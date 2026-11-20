"""Tests for rl_portfolio.data. No network access except tests marked `network`."""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_portfolio import data
from rl_portfolio.config import DataConfig, load_data_config
from rl_portfolio.data import (
    DataValidationError,
    HoldoutAccessError,
    download_all,
    load_prices,
    load_rates,
    load_snapshot,
    normalize_ohlcv,
    parquet_path,
    save_snapshot,
    validate_ohlcv,
)

START = date(2005, 1, 1)
HOLDOUT = date(2025, 1, 1)


def make_ohlcv(
    start: str = "2005-01-03", end: str = "2026-06-30", seed: int = 0, base: float = 100.0
) -> pd.DataFrame:
    """Synthetic OHLCV on business days, in cache schema."""
    index = pd.bdate_range(start, end, name="date")
    rng = np.random.default_rng(seed)
    close = base * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000, 10_000, len(index)).astype("float64"),
        },
        index=index,
    )


@pytest.fixture
def config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        tickers=("AAA", "BBB"),
        rates=("^RATE",),
        start=START,
        holdout_start=HOLDOUT,
        cache_dir=tmp_path / "raw",
    )


@pytest.fixture
def cached(config: DataConfig) -> DataConfig:
    """Config whose cache already holds snapshots for every symbol."""
    save_snapshot(make_ohlcv(seed=1), "AAA", config.cache_dir, kind="price", start=START)
    save_snapshot(make_ohlcv(seed=2), "BBB", config.cache_dir, kind="price", start=START)
    rate = make_ohlcv(seed=3)
    rate["close"] = 2.0
    save_snapshot(rate, "^RATE", config.cache_dir, kind="rate", start=START)
    return config


# --- config ----------------------------------------------------------------


def test_load_data_config_resolves_cache_dir_against_project_root(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "data.yaml").write_text(
        "tickers: [SPY]\nrates: ['^IRX']\nstart: 2005-01-01\n"
        "holdout_start: '2025-01-01'\ncache_dir: data/raw\n"
    )
    cfg = load_data_config(configs / "data.yaml")
    assert cfg.tickers == ("SPY",)
    assert cfg.symbols == ("SPY", "^IRX")
    assert cfg.start == START  # unquoted YAML date
    assert cfg.holdout_start == HOLDOUT  # quoted string
    assert cfg.cache_dir == tmp_path / "data" / "raw"


def test_repo_data_config_is_valid() -> None:
    cfg = load_data_config()
    assert "SPY" in cfg.tickers
    assert cfg.holdout_start == HOLDOUT


def test_config_rejects_start_after_holdout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="before holdout_start"):
        DataConfig(("SPY",), (), date(2025, 6, 1), HOLDOUT, tmp_path)


# --- normalize / fetch -------------------------------------------------------


def test_normalize_ohlcv_matches_cache_schema() -> None:
    index = pd.DatetimeIndex(["2005-01-03 05:00", "2005-01-04 05:00"], tz="America/New_York")
    raw = pd.DataFrame(
        {"Close": [1.0, 2.0], "High": [1, 2], "Low": [1, 2], "Open": [1, 2], "Volume": [5, 6]},
        index=index,
    )
    df = normalize_ohlcv(raw)
    assert list(df.columns) == list(data.OHLCV_COLUMNS)
    assert df.index.name == "date"
    assert df.index.tz is None
    assert (df.index == pd.DatetimeIndex(["2005-01-03", "2005-01-04"])).all()
    assert (df.dtypes == "float64").all()


def test_normalize_ohlcv_rejects_missing_columns() -> None:
    raw = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2005-01-03"]))
    with pytest.raises(DataValidationError, match="Missing columns"):
        normalize_ohlcv(raw)


def test_fetch_ohlcv_requests_adjusted_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_download(symbol, **kwargs):
        calls.update(kwargs, symbol=symbol)
        return make_ohlcv(end="2005-02-01").rename(columns=str.title)

    monkeypatch.setattr(data.yf, "download", fake_download)
    monkeypatch.setattr(data, "last_complete_session_end", lambda: date(2026, 11, 20))
    df = data.fetch_ohlcv("SPY", START)
    assert calls["symbol"] == "SPY"
    assert calls["auto_adjust"] is True
    assert calls["start"] == "2005-01-01"
    assert calls["end"] == "2026-11-20"  # exclusive: today's session is never downloaded
    assert list(df.columns) == list(data.OHLCV_COLUMNS)


def test_last_complete_session_end_uses_new_york_date(monkeypatch: pytest.MonkeyPatch) -> None:
    # 03:00 UTC on Nov 21 is still Nov 20 in New York.
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 21, 3, 0, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(data, "datetime", FrozenDatetime)
    assert data.last_complete_session_end() == date(2026, 11, 20)


def test_fetch_ohlcv_raises_on_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DataValidationError, match="no data"):
        data.fetch_ohlcv("SPY", START)


# --- validation --------------------------------------------------------------


def test_validate_accepts_clean_data() -> None:
    validate_ohlcv(make_ohlcv(), "AAA", START, "price")


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (lambda df: pd.concat([df, df.iloc[[5]]]), "duplicated dates"),
        (lambda df: df.iloc[::-1], "increasing order"),
        (lambda df: df.assign(close=df["close"].where(df.index != df.index[10])), "NaN close"),
        (lambda df: df.assign(low=df["low"].where(df.index != df.index[10], 0.0)), "non-positive"),
        (lambda df: df.loc["2006-01-01":], "truncated"),
        (lambda df: df.drop(df.index[100:110]), "gaps over"),
    ],
    ids=["duplicates", "unsorted", "nan", "non-positive", "late-start", "gap"],
)
def test_validate_rejects_bad_data(corrupt, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_ohlcv(corrupt(make_ohlcv()), "AAA", START, "price")


def test_validate_allows_zero_for_rates() -> None:
    df = make_ohlcv()
    df[list(data.PRICE_COLUMNS)] = 0.0  # T-bill yields were ~0% in 2009-2015
    validate_ohlcv(df, "^IRX", START, "rate")


def test_validate_warns_on_extreme_returns(caplog: pytest.LogCaptureFixture) -> None:
    df = make_ohlcv()
    df.iloc[50:, df.columns.get_loc("close")] *= 0.6  # one-day -40% move
    with caplog.at_level(logging.WARNING, logger="rl_portfolio.data"):
        validate_ohlcv(df, "AAA", START, "price")
    assert "daily returns above 25%" in caplog.text


# --- cache -------------------------------------------------------------------


def test_snapshot_roundtrip_and_metadata(config: DataConfig) -> None:
    df = make_ohlcv()
    path = save_snapshot(df, "AAA", config.cache_dir, kind="price", start=START)
    loaded = load_snapshot("AAA", config.cache_dir)
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)

    meta = json.loads(data.meta_path("AAA", config.cache_dir).read_text())
    assert meta["rows"] == len(df)
    assert meta["kind"] == "price"
    assert meta["first_date"] == "2005-01-03"
    assert meta["sha256"] == data.file_sha256(path)
    assert not list(config.cache_dir.glob("*.tmp"))


def test_snapshot_file_names_are_portable(config: DataConfig) -> None:
    assert parquet_path("^IRX", config.cache_dir).name == "_IRX.parquet"


def test_load_snapshot_detects_modified_file(config: DataConfig) -> None:
    save_snapshot(make_ohlcv(), "AAA", config.cache_dir, kind="price", start=START)
    make_ohlcv(seed=99).to_parquet(parquet_path("AAA", config.cache_dir))
    with pytest.raises(DataValidationError, match="recorded hash"):
        load_snapshot("AAA", config.cache_dir)


def test_load_snapshot_missing_file_explains_how_to_download(config: DataConfig) -> None:
    with pytest.raises(FileNotFoundError, match="rl_portfolio.data download"):
        load_snapshot("AAA", config.cache_dir)


def test_download_all_skips_cached_unless_refresh(
    config: DataConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []

    def fake_fetch(symbol: str, start: date) -> pd.DataFrame:
        fetched.append(symbol)
        return make_ohlcv()

    monkeypatch.setattr(data, "fetch_ohlcv", fake_fetch)

    assert download_all(config) == ["AAA", "BBB", "^RATE"]
    assert download_all(config) == []
    assert download_all(config, refresh=True, symbols=["BBB"]) == ["BBB"]
    assert fetched == ["AAA", "BBB", "^RATE", "BBB"]
    assert json.loads(data.meta_path("^RATE", config.cache_dir).read_text())["kind"] == "rate"


def test_download_all_does_not_cache_invalid_data(
    config: DataConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "fetch_ohlcv", lambda s, start: make_ohlcv().loc["2010":])
    with pytest.raises(DataValidationError, match="truncated"):
        download_all(config, symbols=["AAA"])
    assert not parquet_path("AAA", config.cache_dir).exists()


def test_download_all_rejects_unknown_symbols(config: DataConfig) -> None:
    with pytest.raises(ValueError, match="not in the data config"):
        download_all(config, symbols=["ZZZ"])


# --- holdout guard (look-ahead protection) -----------------------------------


def test_load_prices_default_end_stops_before_holdout(cached: DataConfig) -> None:
    prices = load_prices(["AAA", "BBB"], config=cached)
    assert prices.index.max() < pd.Timestamp(HOLDOUT)
    assert prices.index.max() == pd.Timestamp("2024-12-31")


@pytest.mark.parametrize(
    ("start", "end"),
    [(None, "2025-01-01"), (None, "2026-03-01"), ("2025-02-01", None)],
)
def test_load_prices_blocks_holdout(cached: DataConfig, start, end) -> None:
    with pytest.raises(HoldoutAccessError):
        load_prices(["AAA"], start, end, config=cached)


def test_load_rates_blocks_holdout(cached: DataConfig) -> None:
    with pytest.raises(HoldoutAccessError):
        load_rates(["^RATE"], end="2025-06-30", config=cached)
    assert load_rates(["^RATE"], config=cached).index.max() < pd.Timestamp(HOLDOUT)


def test_load_prices_allows_holdout_when_explicit(cached: DataConfig) -> None:
    prices = load_prices(["AAA"], config=cached, allow_holdout=True)
    assert prices.index.max() == pd.Timestamp("2026-06-30")


# --- loading -----------------------------------------------------------------


def test_load_prices_slices_inclusively(cached: DataConfig) -> None:
    prices = load_prices(["AAA"], "2010-01-04", "2010-01-08", config=cached)
    assert list(prices.index.strftime("%Y-%m-%d")) == [
        "2010-01-04",
        "2010-01-05",
        "2010-01-06",
        "2010-01-07",
        "2010-01-08",
    ]


def test_load_prices_returns_one_column_per_ticker(cached: DataConfig) -> None:
    prices = load_prices(["AAA", "BBB"], config=cached)
    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices.index.name == "date"
    assert not prices.isna().any().any()
    assert prices.index.equals(load_prices(["AAA"], config=cached).index)


# --- calendar consistency across tickers -------------------------------------


def _replace_snapshot(config: DataConfig, symbol: str, df: pd.DataFrame) -> None:
    save_snapshot(df, symbol, config.cache_dir, kind="price", start=START)


def test_load_prices_rejects_missing_date_in_one_ticker(cached: DataConfig) -> None:
    # One missing day is a 2-business-day gap: per-series validation accepts it.
    bbb = make_ohlcv(seed=2).drop(pd.Timestamp("2010-03-10"))
    _replace_snapshot(cached, "BBB", bbb)
    with pytest.raises(DataValidationError, match=r"BBB: 1 missing \(2010-03-10\); 0 extra"):
        load_prices(["AAA", "BBB"], config=cached)


def test_load_prices_rejects_extra_date_in_one_ticker(cached: DataConfig) -> None:
    aaa = make_ohlcv(seed=1)
    saturday = pd.DataFrame(aaa.loc[["2010-03-12"]].to_numpy(), columns=aaa.columns)
    saturday.index = pd.DatetimeIndex(["2010-03-13"], name="date")
    _replace_snapshot(cached, "AAA", pd.concat([aaa, saturday]).sort_index())
    # AAA is the reference, so BBB is reported as missing AAA's extra date.
    with pytest.raises(DataValidationError, match=r"BBB: 1 missing \(2010-03-13\)"):
        load_prices(["AAA", "BBB"], config=cached)
    with pytest.raises(DataValidationError, match=r"AAA: 0 missing \(\); 1 extra \(2010-03-13\)"):
        load_prices(["BBB", "AAA"], config=cached)


def test_load_prices_ignores_mismatch_outside_requested_range(cached: DataConfig) -> None:
    _replace_snapshot(cached, "BBB", make_ohlcv(seed=2).drop(pd.Timestamp("2010-03-10")))
    prices = load_prices(["AAA", "BBB"], "2015-01-01", "2020-12-31", config=cached)
    assert prices.index[0] == pd.Timestamp("2015-01-01")
    assert prices.index[-1] == pd.Timestamp("2020-12-31")
    assert not prices.isna().any().any()


def test_load_prices_rejects_snapshots_with_different_end(cached: DataConfig) -> None:
    # BBB snapshot taken a month earlier than AAA's: both cover the pre-holdout period.
    _replace_snapshot(cached, "BBB", make_ohlcv(seed=2, end="2026-05-29"))
    load_prices(["AAA", "BBB"], config=cached)
    with pytest.raises(DataValidationError, match="--refresh"):
        load_prices(["AAA", "BBB"], config=cached, allow_holdout=True)


def test_load_prices_rejects_rate_series(cached: DataConfig) -> None:
    with pytest.raises(ValueError, match="load_rates"):
        load_prices(["AAA", "^RATE"], config=cached)


def test_load_prices_rejects_unknown_field(cached: DataConfig) -> None:
    with pytest.raises(ValueError, match="field must be one of"):
        load_prices(["AAA"], config=cached, field="adj_close")


# --- live API (opt-in) -------------------------------------------------------


@pytest.mark.network
def test_live_download_spy() -> None:
    df = data.fetch_ohlcv("SPY", date(2024, 1, 1))
    validate_ohlcv(df, "SPY", date(2024, 1, 1), "price")
    assert df.index[0] == pd.Timestamp("2024-01-02")
