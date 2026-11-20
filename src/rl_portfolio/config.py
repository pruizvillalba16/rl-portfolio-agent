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


@dataclass(frozen=True)
class FeatureConfig:
    """Windows and settings for the technical features (see ``configs/features.yaml``)."""

    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vol_window: int = 20
    sma_window: int = 50
    warmup_days: int = 100
    clip: float = 5.0

    def __post_init__(self) -> None:
        windows = {
            "rsi_window": self.rsi_window,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "vol_window": self.vol_window,
            "sma_window": self.sma_window,
        }
        for name, value in windows.items():
            if not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2, got {value!r}")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be shorter than macd_slow")
        if self.warmup_days < self.min_warmup_days:
            raise ValueError(
                f"warmup_days ({self.warmup_days}) must be at least {self.min_warmup_days} "
                "so that every rolling window is full"
            )
        if self.clip <= 0:
            raise ValueError(f"clip must be positive, got {self.clip}")

    @property
    def min_warmup_days(self) -> int:
        """Rows needed before every feature has a value (one row is lost to returns)."""
        return max(
            self.rsi_window + 1,
            self.vol_window + 1,
            self.sma_window,
            self.macd_slow + self.macd_signal,
        )


def load_feature_config(path: str | Path = CONFIG_DIR / "features.yaml") -> FeatureConfig:
    """Load ``FeatureConfig`` from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    return FeatureConfig(**raw)


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
