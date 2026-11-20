"""Typed configuration loaded from the YAML files in ``configs/``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class DataConfig:
    """Settings for downloading and caching market data.

    Attributes:
        tickers: ETF symbols, stored as adjusted OHLCV prices.
        rates: Interest-rate symbols (e.g. ``^IRX``), stored as yields in percent.
        start: First date to download.
        holdout_start: First day of the final holdout period.
        cache_dir: Absolute directory where Parquet files are stored.
    """

    tickers: tuple[str, ...]
    rates: tuple[str, ...]
    start: date
    holdout_start: date
    cache_dir: Path

    def __post_init__(self) -> None:
        if self.start >= self.holdout_start:
            raise ValueError(
                f"start ({self.start}) must be before holdout_start ({self.holdout_start})"
            )
        overlap = set(self.tickers) & set(self.rates)
        if overlap:
            raise ValueError(f"Symbols listed as both ticker and rate: {sorted(overlap)}")

    @property
    def symbols(self) -> tuple[str, ...]:
        """All symbols to download: tickers first, then rates."""
        return self.tickers + self.rates


def _to_date(value: Any) -> date:
    # PyYAML parses unquoted ISO dates as `date`; quoted ones arrive as strings.
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def load_data_config(path: str | Path = CONFIG_DIR / "data.yaml") -> DataConfig:
    """Load ``DataConfig`` from a YAML file.

    A relative ``cache_dir`` is resolved against the parent of the config file's
    directory (the project root for files in ``configs/``), so results do not
    depend on the current working directory.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())

    cache_dir = Path(raw["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = path.parent.parent / cache_dir

    return DataConfig(
        tickers=tuple(raw["tickers"]),
        rates=tuple(raw.get("rates") or ()),
        start=_to_date(raw["start"]),
        holdout_start=_to_date(raw["holdout_start"]),
        cache_dir=cache_dir,
    )
