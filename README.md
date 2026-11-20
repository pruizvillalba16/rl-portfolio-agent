# rl-portfolio-agent

[![CI](https://github.com/pruizvillalba16/rl-portfolio-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/pruizvillalba16/rl-portfolio-agent/actions/workflows/ci.yml)

A reinforcement learning agent that allocates capital across US ETFs, evaluated with
walk-forward validation and an untouched final holdout.

The goal is an **honest, reproducible evaluation**, not beating the benchmarks. Consistently
beating buy-and-hold with public daily data is something most professional funds fail to
do; a rigorous evaluation that explains a negative result is worth more than an inflated
backtest. This is a portfolio project and is **not intended for real-money trading**.

## Plan

| Phase | Scope |
|---|---|
| v1 | SPY only, flat or 100% long, daily decisions. DQN and PPO against buy & hold and a moving-average crossover. |
| v2 | SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ + cash. Long-only weights, weekly rebalance. PPO and SAC against equal weight, 60/40 and minimum variance. |
| Phase 3 | 5-day volatility forecast (GARCH, HAR, LightGBM); compare the agent with and without it. |
| Final test | One run on the 2025–2026 holdout. |

Every result is reported over 5 seeds (mean ± std), with transaction costs of 10 bps per trade
and a sensitivity analysis at 0, 5, 10 and 25 bps.

## Status

| Component | State |
|---|---|
| Tooling: uv, ruff, pytest, pre-commit, GitHub Actions | Done |
| Data: download, validation, Parquet cache, holdout guard | Done |
| Features: indicators, look-ahead tests, walk-forward scaler | Done |
| Walk-forward splits | Next |
| Gymnasium environment (v1) | Planned |
| Benchmarks and metrics | Planned |
| Training, tuning (Optuna) and tracking (MLflow) | Planned |

## Methodology safeguards

- **No look-ahead.** The observation at day *t* uses data up to the close of *t*; an action
  taken at *t* earns the return from close *t* to close *t+1*. Every feature is tested by
  truncating and randomly rewriting the future: features up to *t* must not change.
- **Normalisation is fitted on the training window only** and reused, frozen, for
  validation and test.
- **Walk-forward validation:** 5 years train, 1 year validation, 1 year test, rolled
  forward one year at a time.
- **Final holdout (2025–2026)** is blocked in code: loading those dates raises an error
  unless `allow_holdout=True` is passed explicitly.
- **Data fails loudly.** Downloads are validated, ETF calendars must match exactly, and
  nothing is filled or dropped silently. Cached snapshots carry a SHA-256 hash.

The reasoning behind every design choice is in [DECISIONS.md](DECISIONS.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv if needed.

```bash
uv sync                          # create .venv and install dependencies
uv run pre-commit install        # run ruff and file checks on every commit
uv run python -m rl_portfolio.data download   # download and cache market data
```

Data comes from Yahoo Finance via `yfinance` and is cached in `data/raw/` (not committed).
`download` skips symbols already cached; add `--refresh` to re-download everything.

## Usage

```python
from rl_portfolio.config import load_data_config, load_feature_config
from rl_portfolio.data import load_prices
from rl_portfolio.features import FeatureScaler, compute_features

prices = load_prices(["SPY", "TLT"], config=load_data_config())   # ends 2024-12-31
features = compute_features(prices, load_feature_config())         # (ticker, feature) columns

scaler = FeatureScaler().fit(features.loc[:"2009"])                # training window only
normalised = scaler.transform(features.loc["2010"])                # frozen statistics
```

## Development

```bash
uv run pytest                    # offline test suite
uv run pytest -m network         # live yfinance check (needs internet)
uv run ruff check . && uv run ruff format .
```

## Project layout

```
configs/            YAML parameters (data, features)
src/rl_portfolio/   package: config, data, features
tests/              pytest suite, including look-ahead tests
DECISIONS.md        design decisions and their reasoning
```

## Stack

Python 3.12, uv, pandas, Parquet, yfinance, Gymnasium, Stable-Baselines3, Optuna, MLflow,
TensorBoard, arch, LightGBM, Streamlit, pytest, ruff, pre-commit, GitHub Actions.
Components not yet in the status table above are planned and will be added as each phase
starts.
