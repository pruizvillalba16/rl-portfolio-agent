"""Download, validate, cache and load daily market data.

ETF prices are adjusted for dividends and splits (``auto_adjust=True``), so
close-to-close returns are total returns. Interest-rate series such as ``^IRX``
are stored as annualised yields in percentage points.

Each symbol is cached as an immutable Parquet snapshot plus a ``.meta.json``
sidecar (download time, library versions, SHA-256 of the Parquet file).
Snapshots are never updated incrementally: adjusted prices are rewritten
backwards every time a dividend is paid, so appending new rows to an old
snapshot would mix two different adjustments. Refreshing re-downloads the
full history.

Usage::

    uv run python -m rl_portfolio.data download            # missing symbols only
    uv run python -m rl_portfolio.data download --refresh  # re-download everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

from rl_portfolio.config import DataConfig, load_data_config

logger = logging.getLogger(__name__)

Kind = Literal["price", "rate"]

MARKET_TZ = ZoneInfo("America/New_York")

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")

# Validation thresholds.
MAX_START_DELAY_DAYS = 7  # calendar days between requested start and first row
MAX_GAP_BUSINESS_DAYS = 5  # business days between consecutive rows
EXTREME_DAILY_RETURN = 0.25  # absolute daily return that triggers a warning


class DataValidationError(ValueError):
    """Raised when downloaded or cached data fails a validation check."""


class HoldoutAccessError(RuntimeError):
    """Raised when holdout dates are requested without ``allow_holdout=True``."""


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert a yfinance frame to the cache schema.

    Lower-case ``open/high/low/close/volume`` float64 columns and a tz-naive
    ``DatetimeIndex`` named ``date``. Rows are not sorted or cleaned here, so
    that :func:`validate_ohlcv` can still see any problem in the source.
    """
    df = raw.rename(columns=str.lower)
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {sorted(missing)}")

    df = df.loc[:, list(OHLCV_COLUMNS)].astype("float64")
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize().astype("datetime64[ns]").rename("date")
    df.columns.name = None
    return df


def last_complete_session_end() -> date:
    """Exclusive end date for downloads: today's date in New York.

    yfinance treats ``end`` as exclusive, so the current New York session is
    never downloaded. While the market is open its row is an intraday price,
    not a close, and would be silently frozen into the snapshot.
    """
    return datetime.now(MARKET_TZ).date()


def fetch_ohlcv(symbol: str, start: date) -> pd.DataFrame:
    """Download the daily history of ``symbol`` from ``start`` to the last closed session."""
    raw = yf.download(
        symbol,
        start=start.isoformat(),
        end=last_complete_session_end().isoformat(),
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
        multi_level_index=False,
    )
    if raw is None or raw.empty:
        raise DataValidationError(f"{symbol}: yfinance returned no data")
    return normalize_ohlcv(raw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ohlcv(df: pd.DataFrame, symbol: str, start: date, kind: Kind) -> None:
    """Check a frame in cache schema; raise ``DataValidationError`` on failure.

    Checks: non-empty, unique and increasing dates, no NaN close, strictly
    positive prices (``kind="price"`` only), history starting within
    ``MAX_START_DELAY_DAYS`` of ``start``, and no gap longer than
    ``MAX_GAP_BUSINESS_DAYS``. Daily returns above ``EXTREME_DAILY_RETURN``
    only log a warning, since real crashes produce them.
    """
    if df.empty:
        raise DataValidationError(f"{symbol}: no rows")

    errors: list[str] = []
    index = df.index
    ordered = index.is_unique and index.is_monotonic_increasing
    if not index.is_unique:
        errors.append(f"{index.duplicated().sum()} duplicated dates")
    if not index.is_monotonic_increasing:
        errors.append("dates are not in increasing order")

    n_nan = int(df["close"].isna().sum())
    if n_nan:
        errors.append(f"{n_nan} NaN close values")

    if kind == "price":
        n_non_positive = int((df[list(PRICE_COLUMNS)] <= 0).any(axis=1).sum())
        if n_non_positive:
            errors.append(f"{n_non_positive} rows with non-positive prices")

    first = index.min().date()
    if (first - start).days > MAX_START_DELAY_DAYS:
        errors.append(f"history starts on {first}, expected close to {start} (truncated?)")

    if ordered and len(index) > 1:
        days = index.values.astype("datetime64[D]")
        gaps = np.busday_count(days[:-1], days[1:])
        long_gaps = np.flatnonzero(gaps > MAX_GAP_BUSINESS_DAYS)
        if long_gaps.size:
            examples = ", ".join(f"{index[i].date()}->{index[i + 1].date()}" for i in long_gaps[:3])
            errors.append(
                f"{long_gaps.size} gaps over {MAX_GAP_BUSINESS_DAYS} business days ({examples})"
            )

    if errors:
        raise DataValidationError(f"{symbol}: " + "; ".join(errors))

    if kind == "price":
        returns = df["close"].pct_change().abs()
        extreme = returns[returns > EXTREME_DAILY_RETURN]
        if not extreme.empty:
            logger.warning(
                "%s: %d daily returns above %.0f%% (first on %s); check they are real",
                symbol,
                len(extreme),
                EXTREME_DAILY_RETURN * 100,
                extreme.index[0].date(),
            )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _file_stem(symbol: str) -> str:
    # "^IRX" -> "_IRX": keep file names portable.
    return re.sub(r"[^A-Za-z0-9.-]", "_", symbol)


def parquet_path(symbol: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_file_stem(symbol)}.parquet"


def meta_path(symbol: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_file_stem(symbol)}.meta.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_snapshot(
    df: pd.DataFrame, symbol: str, cache_dir: Path, *, kind: Kind, start: date
) -> Path:
    """Write ``df`` and its metadata to the cache; return the Parquet path.

    Files are written to a temporary name and then renamed, so an interrupted
    write never leaves a half-written snapshot behind.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = parquet_path(symbol, cache_dir)
    tmp = target.with_suffix(".parquet.tmp")
    df.to_parquet(tmp)
    tmp.replace(target)

    meta = {
        "symbol": symbol,
        "kind": kind,
        "requested_start": start.isoformat(),
        "first_date": df.index.min().date().isoformat(),
        "last_date": df.index.max().date().isoformat(),
        "rows": len(df),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "yfinance_version": yf.__version__,
        "pandas_version": pd.__version__,
        "sha256": file_sha256(target),
    }
    meta_file = meta_path(symbol, cache_dir)
    tmp_meta = meta_file.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, indent=2) + "\n")
    tmp_meta.replace(meta_file)
    return target


def read_meta(symbol: str, cache_dir: Path) -> dict:
    path = meta_path(symbol, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"{symbol}: no metadata at {path}; re-download with --refresh")
    return json.loads(path.read_text())


def load_snapshot(symbol: str, cache_dir: Path) -> pd.DataFrame:
    """Read a cached snapshot, check its hash against the metadata, and validate it."""
    path = parquet_path(symbol, cache_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{symbol}: not cached at {path}; run `uv run python -m rl_portfolio.data download`"
        )
    meta = read_meta(symbol, cache_dir)
    if file_sha256(path) != meta["sha256"]:
        raise DataValidationError(f"{symbol}: {path.name} does not match its recorded hash")

    df = pd.read_parquet(path)
    validate_ohlcv(df, symbol, date.fromisoformat(meta["requested_start"]), meta["kind"])
    return df


def download_all(
    config: DataConfig, *, refresh: bool = False, symbols: Iterable[str] | None = None
) -> list[str]:
    """Download, validate and cache symbols; return the ones actually downloaded.

    Symbols that are already cached are skipped unless ``refresh=True``.
    """
    selected = list(symbols) if symbols is not None else list(config.symbols)
    unknown = set(selected) - set(config.symbols)
    if unknown:
        raise ValueError(f"Symbols not in the data config: {sorted(unknown)}")

    downloaded = []
    for symbol in selected:
        if parquet_path(symbol, config.cache_dir).exists() and not refresh:
            logger.info("%s: cached, skipping (use --refresh to re-download)", symbol)
            continue
        kind: Kind = "rate" if symbol in config.rates else "price"
        df = fetch_ohlcv(symbol, config.start)
        validate_ohlcv(df, symbol, config.start, kind)
        save_snapshot(df, symbol, config.cache_dir, kind=kind, start=config.start)
        logger.info(
            "%s: saved %d rows (%s to %s)",
            symbol,
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )
        downloaded.append(symbol)
    return downloaded


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_field(
    symbols: Sequence[str],
    start: date | str | None,
    end: date | str | None,
    *,
    config: DataConfig,
    field: str,
    allow_holdout: bool,
) -> dict[str, pd.Series]:
    """Load one column per symbol, sliced to [start, end] with the holdout guard."""
    if field not in OHLCV_COLUMNS:
        raise ValueError(f"field must be one of {OHLCV_COLUMNS}, got {field!r}")
    if not symbols:
        raise ValueError("No symbols requested")

    start_ts = pd.Timestamp(start if start is not None else config.start)
    end_ts = pd.Timestamp(end) if end is not None else None
    holdout_ts = pd.Timestamp(config.holdout_start)

    if not allow_holdout:
        if start_ts >= holdout_ts or (end_ts is not None and end_ts >= holdout_ts):
            raise HoldoutAccessError(
                f"Requested dates reach the holdout (from {config.holdout_start}). "
                "Pass allow_holdout=True only in the final-test phase."
            )
        if end_ts is None:
            end_ts = holdout_ts - pd.Timedelta(days=1)

    if end_ts is not None and end_ts < start_ts:
        raise ValueError(f"end ({end_ts.date()}) is before start ({start_ts.date()})")

    return {s: load_snapshot(s, config.cache_dir)[field].loc[start_ts:end_ts] for s in symbols}


def _format_dates(dates: pd.DatetimeIndex, limit: int = 3) -> str:
    shown = ", ".join(d.strftime("%Y-%m-%d") for d in dates[:limit])
    return f"{shown}, ..." if len(dates) > limit else shown


def check_same_calendar(indexes: Mapping[str, pd.DatetimeIndex]) -> None:
    """Raise ``DataValidationError`` unless every index holds exactly the same dates.

    The first symbol is the reference; the error lists, for each other symbol,
    the dates it is missing and the extra dates it has.
    """
    items = list(indexes.items())
    if len(items) < 2:
        return
    ref_symbol, ref = items[0]
    problems = []
    for symbol, index in items[1:]:
        missing = ref.difference(index)
        extra = index.difference(ref)
        if len(missing) or len(extra):
            problems.append(
                f"{symbol}: {len(missing)} missing ({_format_dates(missing)}); "
                f"{len(extra)} extra ({_format_dates(extra)})"
            )
    if problems:
        raise DataValidationError(
            f"Ticker calendars differ from {ref_symbol}: "
            + "; ".join(problems)
            + ". If the snapshots were downloaded on different days, re-download "
            "all of them with --refresh."
        )


def load_prices(
    tickers: Sequence[str],
    start: date | str | None = None,
    end: date | str | None = None,
    *,
    config: DataConfig,
    field: str = "close",
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Load one price field for several tickers that share the same trading days.

    Args:
        tickers: ETF symbols (rate series are rejected; use :func:`load_rates`).
        start: First date, inclusive. Defaults to ``config.start``.
        end: Last date, inclusive. Defaults to the day before the holdout, or
            to the last cached date when ``allow_holdout=True``.
        config: Data configuration (cache location, holdout start).
        field: OHLCV column to return.
        allow_holdout: Allow dates from ``config.holdout_start`` on. Only the
            final-test phase may set this.

    Returns:
        Wide frame, one column per ticker, indexed by date.

    Raises:
        HoldoutAccessError: If the dates reach the holdout without permission.
        DataValidationError: If the tickers do not have exactly the same dates
            in the requested range. Nothing is dropped or filled.
    """
    rates = sorted(set(tickers) & set(config.rates))
    if rates:
        raise ValueError(f"{rates} are rate series; load them with load_rates")

    series = _load_field(
        tickers, start, end, config=config, field=field, allow_holdout=allow_holdout
    )
    check_same_calendar({s: values.index for s, values in series.items()})
    frame = pd.concat(series, axis=1)
    frame.index.name = "date"
    return frame


def load_rates(
    rates: Sequence[str],
    start: date | str | None = None,
    end: date | str | None = None,
    *,
    config: DataConfig,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Load interest-rate closes (annualised yield in percent), one column each.

    No alignment to the ETF calendar is done here: rate series follow the bond
    market calendar, and how to align them (e.g. forward-fill onto ETF trading
    days) is decided by the caller.
    """
    not_rates = sorted(set(rates) - set(config.rates))
    if not_rates:
        raise ValueError(f"{not_rates} are not rate series in the data config")

    series = _load_field(
        rates, start, end, config=config, field="close", allow_holdout=allow_holdout
    )
    frame = pd.concat(series, axis=1, join="outer")
    frame.index.name = "date"
    return frame


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m rl_portfolio.data")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="download and cache market data")
    dl.add_argument("--config", type=Path, default=None, help="path to data YAML")
    dl.add_argument("--refresh", action="store_true", help="re-download cached symbols")
    dl.add_argument("--symbols", nargs="+", default=None, help="subset of symbols")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_data_config(args.config) if args.config else load_data_config()

    if args.command == "download":
        downloaded = download_all(config, refresh=args.refresh, symbols=args.symbols)
        logger.info("Done: %d downloaded, cache at %s", len(downloaded), config.cache_dir)


if __name__ == "__main__":
    main()
