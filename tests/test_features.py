"""Tests for rl_portfolio.features, including the look-ahead proofs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_portfolio.config import FeatureConfig, load_feature_config
from rl_portfolio.features import (
    FEATURE_NAMES,
    FeatureScaler,
    compute_features,
    log_returns,
    macd,
    realized_vol,
    rsi,
    sma_distance,
)

CONFIG = FeatureConfig()


def make_prices(n: int = 1500, tickers=("AAA", "BBB"), seed: int = 0) -> pd.DataFrame:
    """Random-walk closes on business days; each ticker has a different volatility."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2005-01-03", periods=n, name="date")
    data = {
        t: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01 * (i + 1), n)))
        for i, t in enumerate(tickers)
    }
    return pd.DataFrame(data, index=index)


# --- config ------------------------------------------------------------------


def test_repo_feature_config_is_valid() -> None:
    assert load_feature_config() == FeatureConfig()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"macd_fast": 26, "macd_slow": 12}, "shorter than macd_slow"),
        ({"warmup_days": 30}, "at least 50"),
        ({"rsi_window": 1}, "integer >= 2"),
        ({"clip": 0}, "positive"),
    ],
)
def test_feature_config_rejects_invalid(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FeatureConfig(**kwargs)


def test_load_feature_config_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "features.yaml"
    path.write_text("rsi_window: 10\nwarmup_days: 120\n")
    cfg = load_feature_config(path)
    assert cfg.rsi_window == 10
    assert cfg.warmup_days == 120
    assert cfg.sma_window == 50  # default


# --- indicators: known values ------------------------------------------------


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Plain recursive EMA starting at the first value (reference implementation)."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def test_log_returns_known_values() -> None:
    close = pd.Series([100.0, 110.0, 99.0])
    expected = [np.nan, np.log(1.1), np.log(0.9)]
    np.testing.assert_allclose(log_returns(close), expected)


def test_rsi_extremes() -> None:
    rising = pd.Series(np.arange(1.0, 40.0))
    falling = rising[::-1].reset_index(drop=True)
    assert (rsi(rising, 14).dropna() == 100).all()
    assert (rsi(falling, 14).dropna() == 0).all()
    assert rsi(rising, 14).isna().sum() == 14  # first change at row 1, full at row 14


def test_rsi_matches_wilder_recursion() -> None:
    close = make_prices(200)["AAA"]
    delta = close.diff().to_numpy()[1:]
    gain = _ema(np.clip(delta, 0, None), 1 / 14)
    loss = _ema(np.clip(-delta, 0, None), 1 / 14)
    expected = 100 * gain / (gain + loss)
    np.testing.assert_allclose(rsi(close, 14).to_numpy()[14:], expected[13:])


def test_macd_matches_reference_emas() -> None:
    close = make_prices(200)["AAA"]
    c = close.to_numpy()
    raw = _ema(c, 2 / 13) - _ema(c, 2 / 27)
    line, hist = macd(close, 12, 26, 9)
    # The line is valid from row 25 (slow EMA). The signal EMA starts at the
    # first valid line value and is valid 9 values later, from row 33.
    hist_raw = raw[25:] - _ema(raw[25:], 2 / 10)
    np.testing.assert_allclose(line.to_numpy()[25:], (raw / c)[25:])
    assert line.isna().sum() == 25
    np.testing.assert_allclose(hist.to_numpy()[33:], (hist_raw / c[25:])[8:])
    assert hist.isna().sum() == 33


def test_macd_is_zero_for_constant_price() -> None:
    line, hist = macd(pd.Series(np.full(60, 50.0)))
    assert (line.dropna() == 0).all()
    assert (hist.dropna() == 0).all()


def test_macd_does_not_depend_on_price_level() -> None:
    close = make_prices(200)["AAA"]
    line, hist = macd(close)
    line10, hist10 = macd(close * 10)
    pd.testing.assert_series_equal(line, line10)
    pd.testing.assert_series_equal(hist, hist10)


def test_realized_vol_known_value() -> None:
    close = make_prices(100)["AAA"]
    r = np.diff(np.log(close.to_numpy()))
    expected = np.std(r[-20:], ddof=1) * np.sqrt(252)
    assert realized_vol(close, 20).iloc[-1] == pytest.approx(expected)
    assert realized_vol(close, 20).isna().sum() == 20


def test_sma_distance_known_value() -> None:
    close = make_prices(100)["AAA"]
    expected = np.log(close.iloc[-1] / close.iloc[-50:].mean())
    assert sma_distance(close, 50).iloc[-1] == pytest.approx(expected)
    assert (sma_distance(pd.Series(np.full(60, 5.0)), 50).dropna() == 0).all()


# --- feature table -----------------------------------------------------------


def test_compute_features_layout() -> None:
    prices = make_prices()
    features = compute_features(prices, CONFIG)
    assert features.columns.names == ["ticker", "feature"]
    assert list(features.columns) == [(t, f) for t in ("AAA", "BBB") for f in FEATURE_NAMES]
    assert features.index[0] == prices.index[CONFIG.warmup_days]
    assert features.index.equals(prices.index[CONFIG.warmup_days :])
    assert not features.isna().any().any()


def test_compute_features_ticker_columns_are_independent() -> None:
    prices = make_prices()
    both = compute_features(prices, CONFIG)
    alone = compute_features(prices[["BBB"]], CONFIG)
    pd.testing.assert_frame_equal(both[["BBB"]], alone)


def _with_nan(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out.iloc[500, 0] = np.nan
    return out


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_with_nan, "contain NaN"),
        (lambda p: p.iloc[::-1], "increasing dates"),
        (lambda p: p * 0, "strictly positive"),
        (lambda p: p.iloc[:100], "more than 100 rows"),
    ],
    ids=["nan", "unsorted", "non-positive", "too-short"],
)
def test_compute_features_rejects_bad_prices(corrupt, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compute_features(corrupt(make_prices()), CONFIG)


# --- look-ahead proofs -------------------------------------------------------

CUTS = [CONFIG.warmup_days + 1, 250, 731, 1200, 1499]


@pytest.mark.parametrize("cut", CUTS)
def test_features_unchanged_when_future_is_removed(cut: int) -> None:
    """Features up to day t are identical whether or not days after t exist."""
    prices = make_prices()
    full = compute_features(prices, CONFIG)
    truncated = compute_features(prices.iloc[:cut], CONFIG)
    pd.testing.assert_frame_equal(full.loc[truncated.index], truncated, check_exact=True)


@pytest.mark.parametrize("cut", CUTS)
def test_features_unchanged_when_future_is_altered(cut: int) -> None:
    """Rewriting every price after day t (even wildly) leaves features up to t unchanged."""
    prices = make_prices()
    altered = prices.copy()
    rng = np.random.default_rng(cut)
    altered.iloc[cut:] *= rng.uniform(0.2, 5.0, size=altered.iloc[cut:].shape)

    original = compute_features(prices, CONFIG)
    changed = compute_features(altered, CONFIG)
    past = original.index[original.index < prices.index[cut]]
    pd.testing.assert_frame_equal(original.loc[past], changed.loc[past], check_exact=True)
    # Sanity check: the alteration does reach the features from day t on.
    assert not original.loc[prices.index[cut]].equals(changed.loc[prices.index[cut]])


def test_feature_at_t_reacts_to_close_at_t() -> None:
    """The close of day t is used at t (no accidental lag either)."""
    prices = make_prices()
    t = prices.index[700]
    bumped = prices.copy()
    bumped.loc[t] *= 1.05
    a = compute_features(prices, CONFIG).loc[t]
    b = compute_features(bumped, CONFIG).loc[t]
    assert (a != b).all()


# --- scaler ------------------------------------------------------------------


@pytest.fixture
def features() -> pd.DataFrame:
    return compute_features(make_prices(), CONFIG)


def test_scaler_standardises_training_data_pooled_across_tickers(features) -> None:
    train = features.iloc[:800]
    z = FeatureScaler(clip=100).fit_transform(train)
    pooled = z.stack(level="ticker", future_stack=True)
    np.testing.assert_allclose(pooled.mean(), 0, atol=1e-10)
    np.testing.assert_allclose(pooled.std(ddof=0), 1)
    # Pooled statistics keep differences between assets: BBB is twice as volatile.
    assert z[("BBB", "vol")].mean() > z[("AAA", "vol")].mean()


def test_scaler_statistics_come_only_from_fit_data(features) -> None:
    train, later = features.iloc[:800], features.iloc[800:]
    scaler = FeatureScaler().fit(train)
    mean, std = scaler.mean_.copy(), scaler.std_.copy()

    scaler.transform(later * 3 + 7)  # transforming other data never refits
    pd.testing.assert_series_equal(scaler.mean_, mean)
    pd.testing.assert_series_equal(scaler.std_, std)

    # Changing the later data does not change the statistics fitted on train.
    other = FeatureScaler().fit(pd.concat([train, later * 3 + 7]).iloc[:800])
    pd.testing.assert_series_equal(other.mean_, mean)
    assert scaler.fit_start_ == train.index[0]
    assert scaler.fit_end_ == train.index[-1]


def test_scaler_transform_uses_frozen_statistics(features) -> None:
    train, later = features.iloc[:800], features.iloc[800:]
    scaler = FeatureScaler(clip=100).fit(train)
    z = scaler.transform(later)
    col = ("AAA", "rsi")
    expected = (later[col] - scaler.mean_["rsi"]) / scaler.std_["rsi"]
    pd.testing.assert_series_equal(z[col], expected)


def test_scaler_clips_extremes(features) -> None:
    scaler = FeatureScaler(clip=5).fit(features.iloc[:800])
    extreme = features.iloc[800:810] * 1000
    z = scaler.transform(extreme)
    assert z.abs().max().max() == 5


def test_scaler_requires_fit(features) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        FeatureScaler().transform(features)


def test_scaler_rejects_mismatched_features(features) -> None:
    scaler = FeatureScaler().fit(features)
    with pytest.raises(ValueError, match="differ from the fitted"):
        scaler.transform(features.drop(columns="rsi", level="feature"))


def test_scaler_rejects_nan_and_constant_features(features) -> None:
    with_nan = features.copy()
    with_nan.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="contain NaN"):
        FeatureScaler().fit(with_nan)
    constant = features.copy()
    constant.loc[:, (slice(None), "rsi")] = 50.0
    with pytest.raises(ValueError, match="Constant features"):
        FeatureScaler().fit(constant)
