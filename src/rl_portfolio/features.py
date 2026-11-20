"""Technical features and their normalisation.

Every feature at row ``t`` depends only on closes up to and including ``t``:
indicators use trailing ``rolling`` / ``ewm`` windows and nothing is ever
shifted backwards. ``tests/test_features.py`` proves this by truncating and
perturbing the future.

Inputs are wide close-price frames (one column per ticker) as returned by
:func:`rl_portfolio.data.load_prices`. The indicator functions accept a Series
or a wide DataFrame and work column by column.

Normalisation is done by :class:`FeatureScaler`, which is fitted on the
training part of each walk-forward window and then frozen (see DECISIONS.md).
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from rl_portfolio.config import FeatureConfig

FEATURE_NAMES = ("log_ret", "rsi", "macd", "macd_hist", "vol", "sma_dist")
TRADING_DAYS_PER_YEAR = 252

# Indicators accept one ticker (Series) or many (wide DataFrame) and return the same type.
type Frame = pd.Series | pd.DataFrame


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def log_returns(close: Frame) -> Frame:
    """Daily log return ``log(C_t / C_{t-1})``; the first row is NaN."""
    return np.log(close).diff()


def rsi(close: Frame, window: int = 14) -> Frame:
    """Relative Strength Index with Wilder smoothing, in [0, 100].

    Average gains and losses are exponential means with ``alpha = 1 / window``
    (Wilder's smoothing). The recursion starts from the first change instead of
    a ``window``-day simple mean; the difference fades out during the warm-up.
    Undefined (NaN) when there was no price change at all in the window.
    """
    delta = close.diff()
    smooth = {"alpha": 1 / window, "adjust": False, "min_periods": window}
    avg_gain = delta.clip(lower=0).ewm(**smooth).mean()
    avg_loss = (-delta).clip(lower=0).ewm(**smooth).mean()
    # Same as 100 - 100 / (1 + gain / loss), without dividing by zero when loss == 0.
    return 100 * avg_gain / (avg_gain + avg_loss)


def macd(close: Frame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[Frame, Frame]:
    """MACD line and histogram, both divided by the close (dimensionless).

    ``macd = (EMA_fast - EMA_slow) / C_t`` and
    ``macd_hist = (MACD_raw - EMA_signal(MACD_raw)) / C_t``. Dividing by the
    price removes the dependence on the price level (SPY went from ~80 to ~580).
    """
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line / close, (line - signal_line) / close


def realized_vol(close: Frame, window: int = 20) -> Frame:
    """Annualised standard deviation of daily log returns over ``window`` days."""
    return log_returns(close).rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sma_distance(close: Frame, window: int = 50) -> Frame:
    """Log distance of the close to its simple moving average: ``log(C_t / SMA_t)``."""
    return np.log(close / close.rolling(window).mean())


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------


def _check_prices(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty DataFrame (one column per ticker)")
    if not (prices.index.is_unique and prices.index.is_monotonic_increasing):
        raise ValueError("prices must have unique, increasing dates")
    if prices.isna().any().any():
        raise ValueError("prices contain NaN; features never fill missing values")
    if (prices <= 0).any().any():
        raise ValueError("prices must be strictly positive")


def compute_features(prices: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Compute every feature for every ticker and drop the warm-up rows.

    Args:
        prices: Wide frame of adjusted closes, one column per ticker, no NaN.
        config: Indicator windows and warm-up length.

    Returns:
        Frame indexed by date with two-level columns ``(ticker, feature)``,
        tickers in input order and features in ``FEATURE_NAMES`` order. The
        first ``config.warmup_days`` rows are dropped.

    Compute features once on the longest available history and slice the
    result afterwards: the exponential averages depend on where the input
    starts (not on where it ends), so slicing first would shift the warm-up.
    """
    _check_prices(prices)
    if len(prices) <= config.warmup_days:
        raise ValueError(f"Need more than {config.warmup_days} rows of prices, got {len(prices)}")

    macd_line, macd_hist = macd(prices, config.macd_fast, config.macd_slow, config.macd_signal)
    by_feature = {
        "log_ret": log_returns(prices),
        "rsi": rsi(prices, config.rsi_window),
        "macd": macd_line,
        "macd_hist": macd_hist,
        "vol": realized_vol(prices, config.vol_window),
        "sma_dist": sma_distance(prices, config.sma_window),
    }
    features = pd.concat(by_feature, axis=1, names=["feature", "ticker"])
    features = features.swaplevel(axis=1)
    columns = pd.MultiIndex.from_product(
        [prices.columns, list(FEATURE_NAMES)], names=["ticker", "feature"]
    )
    features = features.reindex(columns=columns).iloc[config.warmup_days :]

    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(f"NaN features after the warm-up in {bad}")
    return features


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class FeatureScaler:
    """Z-score per feature, pooled across tickers, then clipped to ``[-clip, clip]``.

    ``fit`` computes one mean and one standard deviation per feature from all
    tickers together, so differences between assets survive normalisation
    (EEM stays more volatile than TLT). Fit it on the training part of a
    walk-forward window only and reuse it, unchanged, for validation and test.

    Within the training window every row is scaled with statistics from the
    whole window. This is a constant affine transform, so it carries no
    day-specific information about the future (see DECISIONS.md).
    """

    def __init__(self, clip: float = 5.0) -> None:
        if clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}")
        self.clip = clip
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self.fit_start_: pd.Timestamp | None = None
        self.fit_end_: pd.Timestamp | None = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    @staticmethod
    def _check_features(features: pd.DataFrame) -> None:
        if not isinstance(features.columns, pd.MultiIndex) or list(features.columns.names) != [
            "ticker",
            "feature",
        ]:
            raise ValueError("features must have (ticker, feature) columns")
        if features.empty:
            raise ValueError("features are empty")
        if features.isna().any().any():
            raise ValueError("features contain NaN")

    def fit(self, features: pd.DataFrame) -> Self:
        """Learn the per-feature mean and standard deviation from ``features``."""
        self._check_features(features)
        stacked = features.stack(level="ticker", future_stack=True)
        mean = stacked.mean()
        std = stacked.std(ddof=0)
        if (std == 0).any():
            raise ValueError(f"Constant features cannot be scaled: {std[std == 0].index.tolist()}")
        self.mean_, self.std_ = mean, std
        self.fit_start_, self.fit_end_ = features.index[0], features.index[-1]
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted statistics; never refits."""
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler is not fitted; call fit() on training data first")
        self._check_features(features)
        names = set(features.columns.get_level_values("feature"))
        if names != set(self.mean_.index):
            raise ValueError(
                f"Feature names {sorted(names)} differ from the fitted {sorted(self.mean_.index)}"
            )
        z = features.sub(self.mean_, axis="columns", level="feature")
        z = z.div(self.std_, axis="columns", level="feature")
        return z.clip(-self.clip, self.clip)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)
