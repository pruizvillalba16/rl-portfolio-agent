# rl-portfolio-agent

Personal portfolio project by Pablo Ruiz Villalba (MSc student). A reinforcement
learning agent for ETF allocation, evaluated rigorously. The goal is an honest,
reproducible evaluation, not beating the benchmarks. Not for real-money trading.

## How we work (important)

- The user decides the design; Claude implements it.
- Before writing a new module: explain the design and the alternatives **in Spanish**,
  then wait for the user's choice.
- After writing code: explain the key parts **in Spanish**.
- For every design decision, draft an entry for `DECISIONS.md` (what was decided and
  why, 3–5 lines, in English). The user approves it before it is committed.
- Code, comments, docstrings, commit messages and docs are in **English**.
- The user edits code only when they choose to; do not assign exercises.
- Small, focused commits. Do not push unless the user asks.

## Non-negotiable rules

- **No look-ahead.** The observation at day t uses data up to the close of day t only.
  An action taken at t earns the return from close t to close t+1.
- Normalisation (z-scores, scalers) is fitted on past data only, never on the full history.
- Every look-ahead-sensitive function has a pytest test proving it.
- **2025–2026 is the final holdout.** Never use it for training, tuning or analysis
  until the final-test phase.
- Optuna tunes on validation windows only.
- Seeds are fixed and logged. Report results over 5 seeds (mean ± std).
- Parameters live in YAML files under `configs/`, not hard-coded.

## Key decisions

- **Assets:** v1 = SPY. v2 = SPY, QQQ, IWM, EFA, EEM, TLT, GLD, VNQ + cash.
- **Data:** daily, 2005–present, from yfinance, cached as Parquet in `data/` (not committed).
- **Validation:** walk-forward, 5y train / 1y validation / 1y test, rolling 1y.
- **Positions:** long-only, no leverage. v1: flat or 100% long. v2: weights ≥ 0 summing
  to 1 (cash included), rebalanced weekly at Friday close.
- **Observation:** last 20 log returns, RSI(14), MACD, 20d volatility, distance to 50d
  SMA, current position/weights.
- **Reward:** log return net of costs; differential Sharpe as a comparison experiment.
- **Costs:** 10 bps per trade; sensitivity analysis at 0, 5, 10, 25 bps.
- **Algorithms (Stable-Baselines3):** v1 DQN and PPO; v2 PPO and SAC.
- **Benchmarks:** v1 buy & hold, moving-average crossover. v2 equal weight (monthly
  rebalance), 60/40, minimum-variance (Markowitz).
- **Metrics:** annualised return, Sharpe, Sortino, max drawdown, Calmar, turnover.
- **Phase 3 (ML):** 5-day volatility forecast with GARCH(1,1), HAR and LightGBM; the best
  one is added to the observation and the agent is compared with and without it.

## Stack

Python 3.12, uv, Gymnasium, Stable-Baselines3, Optuna, pandas, Parquet, arch, LightGBM,
TensorBoard, MLflow (local), Streamlit, pytest, ruff, pre-commit, GitHub Actions.

## Environment

- WSL Ubuntu 24.04. CPU only (AMD integrated graphics, no CUDA).
- Git remote uses the SSH alias `github-personal`. Never use the `gh` CLI in this repo:
  it is logged into a different (university) account.