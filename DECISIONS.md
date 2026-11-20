# Decisions

Design decisions for `rl-portfolio-agent`, with the reasoning behind each one.
New entries are drafted by Claude and approved by the author before being committed.
Format: what was decided, why, and the alternatives that were rejected.

---

## D-001 · Goal and success criterion
*2026-11-20*

**Decision:** Build an RL agent for ETF allocation as a portfolio project. Success means an
honest, reproducible evaluation, not beating the benchmarks.

**Why:** Consistently beating buy-and-hold with public daily data is something most
professional funds fail to do. A rigorous evaluation that explains a negative result is more
credible, and more defensible in an interview, than an inflated backtest.

---

## D-002 · Market: US ETFs
*2026-11-20*

**Decision:** Trade US-listed ETFs only.

**Why:** Free, reliable, long daily histories; well-known benchmarks; mature paper-trading APIs
for a later live phase. ETFs avoid the survivorship bias of individual stocks (a stock
universe chosen today excludes companies that went bankrupt).

**Rejected:** European markets (less free data, lower quality); crypto (free 24/7 streaming,
but less credible on a traditional-finance CV).

---

## D-003 · Asset universe
*2026-11-20*

**Decision:** v1 trades SPY alone. v2 allocates across SPY, QQQ, IWM, EFA, EEM, TLT, GLD,
VNQ and cash.

**Why:** A single asset validates the environment and the method before adding complexity.
The v2 basket spans US large caps, tech, small caps, developed and emerging markets, long
bonds, gold and real estate, so the agent has a real allocation problem to learn (e.g.
rotating into bonds or gold in drawdowns).

---

## D-004 · Data frequency: daily
*2026-11-20*

**Decision:** Daily bars for phases 1–3. Intraday data only in the optional phase 4.

**Why:** Daily data is free, clean and decades long, and training is fast on CPU. Intraday
data adds noise and cost without strengthening what the project demonstrates.

---

## D-005 · Period and final holdout
*2026-11-20*

**Decision:** Use data from 2005 to the present. 2025–2026 is a final holdout that is not
used for training, tuning or analysis until the final-test phase, and is run once.

**Why:** 2005 is the earliest year covered by every asset (GLD started trading in late 2004),
and the period includes 2008, COVID and 2022. An untouched holdout is the only protection
against fitting the method to the test data through repeated looks.

---

## D-006 · Walk-forward validation
*2026-11-20*

**Decision:** Rolling walk-forward: 5 years train, 1 year validation, 1 year test, rolled
forward 1 year at a time (about 14 out-of-sample test years).

**Why:** Financial data is non-stationary and ordered in time; a random split leaks the future
into training. Walk-forward mimics how the model would actually be retrained and used, and
gives many out-of-sample periods instead of one.

---

## D-007 · Data source and cache
*2026-11-20*

**Decision:** Download with yfinance and cache locally as Parquet. Switch to Alpaca data only
in the live phase.

**Why:** yfinance is free and convenient but unofficial and occasionally breaks. A local cache
makes every run reproducible and independent of the API.

---

## D-008 · Positions: long-only, no leverage
*2026-11-20*

**Decision:** v1 is either flat (cash) or 100% long. v2 weights are ≥ 0 and sum to 1, with
cash as an asset.

**Why:** Shorting and leverage add borrowing costs and risks that complicate the simulation
without strengthening the project, and long-only is what a retail investor can replicate.

---

## D-009 · Decision frequency
*2026-11-20*

**Decision:** v1 decides daily. v2 rebalances weekly at Friday close.

**Why:** Daily decisions give a single-asset agent enough samples. With nine continuous
weights, daily rebalancing drives turnover and costs up; weekly is more realistic and stable.

---

## D-010 · Observation and normalisation
*2026-11-20*

**Decision:** The agent observes the last 20 log returns per asset, RSI(14), MACD, 20-day
volatility, distance to the 50-day moving average, and its current position or weights.
Features are z-scored using past data only.

**Why:** Including the current position lets the agent learn that trading has a cost.
Normalising with statistics from the full history is the most common form of look-ahead in
projects like this one.

---

## D-011 · Reward
*2026-11-20*

**Decision:** Log return net of transaction costs. Differential Sharpe ratio is run as a
comparison experiment.

**Why:** Net log return is simple and easy to debug. Differential Sharpe accounts for risk but
is less stable; comparing the two is a result in itself.

---

## D-012 · Transaction costs
*2026-11-20*

**Decision:** 10 bps per trade (commission plus slippage), with a sensitivity analysis at 0,
5, 10 and 25 bps.

**Why:** 10 bps is realistic and conservative for liquid ETFs. Without costs an agent learns
to trade constantly. The sensitivity analysis shows conclusions do not depend on a
convenient cost assumption.

---

## D-013 · Algorithms
*2026-11-20*

**Decision:** Stable-Baselines3. v1 (discrete actions): DQN and PPO. v2 (continuous
weights): PPO and SAC.

**Why:** Standard, well-tested implementations, each pairing an on-policy with an off-policy
method. Writing algorithms from scratch adds little; the value of the project is in the
environment and the evaluation.

---

## D-014 · Statistical robustness
*2026-11-20*

**Decision:** Every configuration is trained with 5 seeds and reported as mean ± std. Optuna
runs about 50 trials on validation windows only.

**Why:** In financial RL, variance across seeds is often larger than the difference between
algorithms; a single-seed result would be the first thing a reviewer questions. Tuning only
on validation keeps the test years clean.

---

## D-015 · Benchmarks and metrics
*2026-11-20*

**Decision:** v1 is compared against buy & hold and a moving-average crossover. v2 against
equal weight (monthly rebalance), 60/40 and minimum-variance (Markowitz). Metrics:
annualised return, Sharpe, Sortino, max drawdown, Calmar, turnover.

**Why:** Simple, standard strategies the agent must beat to add value, and metrics that cover
both return and risk.

---

## D-016 · Phase 3: volatility forecasting
*2026-11-20*

**Decision:** Forecast 5-day volatility with GARCH(1,1), HAR and LightGBM; add the best
forecast to the observation and compare the agent with and without it. News sentiment is
an optional extension.

**Why:** Unlike prices, volatility is reasonably predictable, so this phase produces solid
results. The with/without comparison measures whether the ML signal actually helps the
agent. Sentiment needs a lot of extra data work for an uncertain gain.

**Rejected:** Replacing classical ML with deep learning or time-series foundation models.
In finance they do not reliably beat gradient boosting or linear models.

---

## D-017 · Tooling
*2026-11-20*

**Decision:** uv for dependencies, YAML configs, pytest (including look-ahead tests), ruff,
pre-commit and GitHub Actions.

**Why:** Reproducible installs, experiments defined by config rather than code edits,
automatic checks on every commit and push. It is a few hours of setup that separates a
professional project from a notebook.

---

## D-018 · Experiment tracking
*2026-11-20*

**Decision:** TensorBoard for training curves; MLflow (local) for configurations, metrics and
results.

**Why:** Windows × seeds × algorithms means dozens of runs; TensorBoard alone cannot compare
them well. MLflow is an industry standard.

---

## D-019 · Development environment
*2026-11-20*

**Decision:** WSL (Ubuntu 24.04) with Docker, VS Code and Claude Code. Training on CPU, with
Colab or Kaggle if a GPU is ever needed.

**Why:** The data stack (Spark, Kafka) is Linux-first, Docker Desktop already runs on WSL2,
and Linux matches CI and production. Daily data and Stable-Baselines3 train fine on CPU;
the machine has AMD integrated graphics and no CUDA.

---

## D-020 · Language
*2026-11-20*

**Decision:** Code, comments, commits, `DECISIONS.md`, the report and the dashboard are in
English.

**Why:** Consistency for readers and recruiters in any country.

---

## D-021 · Scope of this plan
*2026-11-20*

**Decision:** Phases 0–3 plus deliverables, ending around 2027-04-18. Phase 4 (Kafka/Spark
streaming) and phase 5 (Alpaca paper trading) are planned in detail only after that. The
license is decided at the end of the project.

**Why:** By then it will be clear how much time is left and what worked; planning them now
would be guesswork. A license can be added at any time.

---

## D-022 · Package layout
*2026-11-20*

**Decision:** src layout, with the package at `src/rl_portfolio/` and tests in `tests/`.

**Why:** The package must be installed to be imported, so tests run against what is actually
installed rather than whatever happens to be in the working directory.

**Rejected:** Flat layout (simpler, but allows accidental imports from the repo root).

---

## D-023 · Dependency management
*2026-11-20*

**Decision:** Runtime dependencies are added incrementally, when a module first needs them.
Dev tools (ruff, pytest, pre-commit) live in a uv `dev` group. `uv.lock` is committed and
CI installs with `--locked`.

**Why:** Each dependency enters with a reason, the lock stays small, and any machine can
reproduce the exact environment.

---

## D-024 · Lint rules
*2026-11-20*

**Decision:** Ruff with rule sets E, W, F, I, B, UP, SIM and NPY; line length 100; ruff
also formats the code.

**Why:** Catches real bugs and outdated syntax with little noise. Mandatory docstring (D)
and pandas (PD) rules were rejected for now because they create more noise than value
at this stage.

---

## D-025 · Where checks run
*2026-11-20*

**Decision:** pre-commit runs ruff and basic file-hygiene hooks; pytest runs only in CI.
Ruff runs through `uv run`, so its version comes from `uv.lock`.

**Why:** Commits stay fast, and the full test suite still gates every push. A single
source for the ruff version keeps local and CI results identical.
