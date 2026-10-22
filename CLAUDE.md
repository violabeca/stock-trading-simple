# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A deliberately small framework to backtest and optimize trading strategies.
Read `README.md` first — it explains the architecture in plain language.
The pipeline is always screen → backfill → backtest → optimize → inspect:
`data.load_panel` (DB → wide OHLCV Panel) → `backtest.run/sweep` (one vectorbt
engine) → `optimize` (Optuna, Sharpe + return Pareto front) → `plots`.
Notebooks drive it interactively; the `tsim` CLI runs the same code headless.
The prime directive: **keep it small**. Fewer lines wins; no new dependencies,
layers, or abstractions unless something genuinely cannot be expressed with the
existing ones.

## Commands

```bash
uv sync --all-extras                 # installs deps into .venv (Python 3.13, pinned in .python-version)
uv run pytest -q                     # offline tests, ~5 s — the main gate
uv run pytest tests/test_backtest.py::test_name -q   # run a single test / file
uv run ruff check .                  # style gate (line length 100)
uv run tsim coverage                 # checks the shared DB is reachable
uv run tsim optimize --strategy <name> --n-trials 5   # smoke-runs a strategy end to end
uv run tsim altdata --dataset all --universe ALL --since 2018-01-01  # only what each name is missing (prefer notebooks/03_altdata_update.py — the same ingest plus the coverage audits)
uv run python scripts/marketdata_dump.py --plan   # MarketData.app trial download (var/marketdata/); rerun daily or --wait
uv run python scripts/ibkr_shortstock.py   # IBKR's borrow fees + shares available (var/ibkr/); run daily before the open
(cd collectors && docker compose up -d)   # 24/7 collectors + their own TimescaleDB (collectors/README.md)
git config core.hooksPath .githooks   # once per clone — enables the notebook-clearing pre-commit hook
git ls-files '*.ipynb' | xargs python3 scripts/nbclean.py   # clear notebook outputs by hand
```

## Hard rules

- **The strategy contract is sacred.** A strategy = `strategies/<name>/` with a
  pure `signals(panel, **params) -> dict` (+ optional `valid`, `indicators`)
  and a `config.yaml` with `data / optuna / params` blocks (+ optional `backtest:
  {fill: open | pessimistic, order_notional: <dollars>}` — `pessimistic` buys at the bar's
  high and sells at its low, benchmark included; `order_notional` is the strategy's REAL order
  size, which prices the $1 minimum commission (1 bp on the engine's default $10,000 order, 10 bp
  on the $1,000 orders of a 30-slot book on $30k). Notebooks pass `fill=strat.fill` to every
  direct `bt.run` / `bt.benchmark` / `bt.active_stats` call and take fees from
  `bt.strategy_costs(strat, panel, settings)`, never from `bt.panel_costs` directly). Adding a
  strategy must never require touching `src/trading_simple/` — if it seems to,
  the design is wrong; stop and reconsider.
- **One parameter, one declaration.** `params:` is the whole knob surface: each
  entry carries `doc`, the shipped `value`, and optionally `search` (the Optuna
  range) and `grid` (the coarse-sweep list); a bare scalar means value-only, and
  **no `grid` pins the sweep at `value`**. `strategies._unpack` derives the flat
  `params` / `sweep` / `optuna["space"]` views the engine consumes and **refuses to
  load** a config whose `value` its own `search` cannot reach or whose own `grid`
  excludes. That is not pedantry: both failures are silent, and both shipped here —
  a sweep grid pinned to values the study had long since moved off meant §10b
  walk-forward faithfully validated a strategy nobody ran, twice, with no error
  anywhere. Bump `optuna.study` whenever a `search:` bound changes; a journal that
  mixes two spaces is two experiments in one file and nothing detects it.
- **Two execution models; pick the one the strategy actually is.** `signals()`
  returning `entries`/`exits` runs **per-symbol**: every symbol gets its own
  `init_cash` in its own account. Returning a `weights` frame (target fraction of the
  book per symbol) runs **portfolio mode**: ONE shared cash account of `init_cash`
  *total*, so a buy must be funded by a sell (`from_orders`, `group_by=True`,
  `cash_sharing=True`, `call_seq="auto"`). Per-symbol is right for a per-symbol signal
  and **fatal for a cross-sectional ranker**: cash attached to a name it is flat in sits
  idle, so a strategy holding 40 of 100 names is structurally ~30% invested against a
  100%-invested benchmark and cannot beat it however well it picks. (`informed_flow`
  lost 500/500 trials to buy-and-hold on exactly this.) In portfolio mode the dollar
  amount is a *reporting scale* — with fractional shares everything is scale-invariant —
  so what actually constrains the book is the number of slots, not the $10k. Rebalancing
  is **event-driven per name**: `backtest.rotation_targets` emits a target only on the
  bars a name's *own* weight changes, so a held position is bought once, left to drift
  while other names rotate around it, and sold whole when it leaves — never re-trimmed
  mid-hold (a whole-row mask would re-trim every holding on any single name's rotation).
  The one opt-in exception is a **concentration cap**: a portfolio strategy that returns a
  `retrim_k` scalar alongside `weights` has a held name trimmed back to its slot once its
  realized book weight reaches `retrim_k ×` its target (so a runaway winner cannot dominate
  the book; freed cash rotates into the next pick). It is off by default
  (`backtest._retrim_loop`, a simulate-then-adjust fixed point since realized weight is only
  knowable inside a run), so the benchmark and every strategy that omits it are unchanged;
  `informed_flow` sets it to `2`.
  A signal strategy returns `cash_frac` (a scalar, or a dense per-bar frame read at the
  fill bar — `backtest.cash_frac_size`, the risk-per-trade rule `min(slot, risk / stop)`;
  a NaN cell is refused because vectorbt would silently skip that entry) alongside
  `entries`/`exits` to run the **cash-shared wallet** variant: ONE `init_cash` book, each
  entry deploying that fraction of the FREE cash — total invested can never exceed the
  wallet, the way a real single account trades a multi-symbol scanner. **The wallet (or portfolio mode)
  is the repo default — every shipped strategy trades one shared $10k book**;
  per-symbol books are a diagnostic only (drop `cash_frac`). Wallet-mode shorts work,
  but percent sizing cannot reverse: an opposite-direction entry closes the position
  (`upon_opposite_entry="close"`), never flips it. Shorts **lock cash** (`lock_cash=True`,
  since 2026-09-27): a short's sale proceeds stay reserved for its cover. Before that,
  vectorbt let a later long SPEND them — a $10k wallet ran 280% invested in the regression
  test — and the buy that covers a short was then cash-limited, so a stop covered only part
  of the position and the rest ran on past it (found by burst_scan's label ↔ engine
  reconciliation). Studies of every strategy that can short in the wallet were bumped.
  **Portfolio mode locks short proceeds too** (proposal #31, approved 2026-10-19): a negative
  weight is a short, gross ≤ ~100% when orders are placed, so a dollar-neutral book is ~50/50.
  A portfolio strategy may also return a **`borrow`** frame/scalar of yearly rates: the engine
  charges each short its fee per calendar day held (`backtest._borrow_fees`); without it shorts
  borrow for free. `informed_flow_ls` is the long/short book built on both.
- **Comparability rules (README has the full list).** One shared $10k wallet,
  fractional shares, one global cost model, profits reinvested and reported as
  geometric CAGR only ($10k→$11k→$12.1k over 2y = 10.0%/yr, never the 10.5%
  arithmetic mean), objectives benchmark-relative. `strategies/baseline/` is the
  equal-weight buy-and-hold hurdle expressed as a strategy (test-pinned to
  `backtest.benchmark`); every strategy notebook must show the baseline comparison,
  and `notebooks/04_compare.ipynb` is the cross-strategy scoreboard. `optuna.
  min_trades_per_year` (per config, PER YEAR not absolute — an absolute count is
  implicitly stricter on a short window than a long one) floors the trading rate
  in-objective so few-trade artifacts cannot crown a Pareto front.
- **Beat the benchmark, never the zero line — and never on a cumulative number.**
  Two independent traps, both of which have bitten this repo.
  *(a) Absolute vs relative.* A long-only strategy in a rising market earns a positive
  Sharpe *just by being invested*, so an absolute Sharpe cannot separate skill from
  market exposure, and optimizing it actively rewards exposure. `optimize.OBJECTIVES`
  therefore scores, by default, `excess_cagr` (maximized) against `volatility` (minimized).
  A config's `optuna.objectives` may instead name one or two `optimize.SCOREABLE` metrics
  (directions built in, absolute Sharpe/return deliberately not scoreable; new study name):
  - `excess_cagr` is the return over an equal-weight buy-and-hold of the same panel
    (`backtest.benchmark`).
  - `volatility` is the book's own annualized risk, the "annual risk rate".
  - `alpha_cagr` = `cagr − beta × benchmark cagr` is the ONE sanctioned alternative return
    objective, for a near-zero-beta long/short book that is flat most of the day
    (`level_flow`): excess against a fully invested +47%/yr basket is unreachable by
    construction and would say "nothing beats buy-and-hold" without measuring selection.
    Alpha still charges for the beta actually run (beta ≈ 1 ⇒ alpha ≈ excess).
    `informed_flow` scores it too (alone since v27), for the opposite reason: its return-only
    v25 study crowned a 5-slot, beta-2.1 long book whose holdout excess (+10%/yr) was all
    exposure (alpha −35%/yr) — maximizing excess rewards concentrating in high-beta names.
    And never pair it with `volatility` ↓ when the strategy has an exposure knob: v26's front
    became an exposure dial (every row 12–18% alpha per unit of beta) and the balanced pick was
    a book half in cash that made +0.6%/yr on the holdout against buy-and-hold's +18.4%.
  - `pick_from_front` only picks front rows above the study's hurdle (`optimize.HURDLES`:
    the first objective when it is `excess_cagr` or `alpha_cagr`) — `MIN_PICK_EXCESS`.
  - Absolute Sharpe and return are diagnostics only.
  - Retired pairs: `info_ratio`/`excess_cagr` correlated 0.975 and collapsed fronts to 1–2
    points; `tracking_error` punished de-risking (cash, volatility targeting) as straying
    from the index.
  - Studies are stamped with their objective names, and a journal scored on other objectives
    is refused — bump the study.
  (`informed_flow` learned this the hard way: maximizing Sharpe produced a "0.94 holdout
  Sharpe" strategy that lost to buy-and-hold on both train and holdout.)
  *(b) Per-year vs cumulative.* Every return metric is **annualized** (`cagr`,
  `excess_cagr`). A cumulative return grows with the length of the backtest window, so
  it rewards a long window for being long — it can compare neither two strategies (this
  repo's windows run from 1 month to 4 years) nor, more insidiously, two **walk-forward
  folds of unequal length inside one strategy**. Scoring one annualized objective
  against one cumulative one, as v4 did, is incoherent on its own terms. `years` and
  `trades_per_year` ride along with every metric: annualizing makes a short backtest
  *comparable*, never *trustworthy*.
  What is deliberately **not** normalized away is **time in market** — scaling a
  30%-invested strategy's return up "to compare fairly" inverts the truth, because
  sitting in cash is a real cost paid against a fully-invested benchmark.
  For a cross-sectional ranker, also check the signal itself before the backtest: rank
  IC and decile spread against **cross-sectionally demeaned** forward returns, which
  removes the market and leaves only stock selection.
- **Alternative data rides on `Panel.features`.** Non-OHLCV signals (insider,
  ratings, price targets, short interest) live in the shared DB and reach a
  strategy as `panel.features[name]` — one wide *date × symbol* daily frame per
  signal, built by `features.py`, already point-in-time lagged. A strategy opts in
  with a `data.features:` list in its `config.yaml`; feature builders do only
  DB→daily-aggregate, all modelling stays in `signals()`. Adding an alt-data
  strategy needs no engine change; adding a new *feature type* is one builder in
  `features.py` (proposal-only, like any `src/` change). See `informed_flow`. The `opt_*`
  features (2026-10-19) are the one exception to "from the DB": a daily summary of the local
  MarketData option chains (`providers/marketdata.py`, cache in `var/options/`, built by
  notebook 03 §5b), served through the same as-of lag and masked after 5 days.
- **ALL alt-data ingestion goes through `notebooks/03_altdata_update.py`** (marimo; run
  `uv run marimo edit notebooks/03_altdata_update.py`). Every run fetches only what the
  coverage ledger (`altdata_coverage`) says is MISSING, name by name, for EVERY name in the DB
  (`UNIVERSE = "ALL"`): the history back to `BACKFILL_START` and the days since the name's last
  refresh (`cli._coverage_plan`; per-name sources wait `cli._STALE_DAYS`). Insider history up
  to SEC's newest published quarterly data set comes from the bulk files, only the days after
  it per filing; each finished name is recorded at once, so a stopped run resumes. A ledger's
  `covered_to` is the RUN date, not the newest data: FINRA publishes a settlement ~2 weeks
  after it, so a tail must re-read `cli._OVERLAP_DAYS` (31 for short interest) or a settlement
  dated before a run but published after it is skipped forever. Do not hand the user an
  ad-hoc `tsim altdata` recipe for a backfill; add it to the notebook instead. The
  notebook also audits coverage, and that audit is load-bearing: **a thin alt-data
  table does not fail loudly, it goes quiet.** A cross-sectional ranker imputes a
  neutral median rank where a name has no data, so a sleeve covering 3 of 100 names
  looks scored while actually being a huge arbitrary tilt on those 3 (this shipped,
  in `short_interest`). Check breadth and depth, never just freshness — and breadth at the
  END, in EVERY universe a strategy trades: a single-universe update (the old
  `LiquidLeaders` default) left 395 of informed_flow's 500 names on a weeks-old short
  interest from July 2026, with every check green, and its `"TRADED"` successor reached only
  ~765 of ~2,300 names from July to October 2026 (267 of the 500 names in informed_flow_ls's
  fresh-names test lost short interest from late August). Hence `UNIVERSE = "ALL"` and the §4a tail audit per traded universe;
  `features._SI_MAX_AGE_DAYS` (45) masks a stale level.
  Yahoo's analyst list can also stop for ONE name at the source (META, CAT, FANG, WELL, SMCI …
  end in October 2024; re-ingesting cannot fix it): §4c flags a ratings / price-target feed
  that went quiet against the name's own pace.
- **Engine fixes are proposal-only.** When you spot a bug or improvement in
  `src/trading_simple/` while doing other work, report it with a concrete
  suggested fix — implement only after explicit approval. Keeps strategy
  diffs clean.
- **No look-ahead.** Signals must be execution-timed via `backtest.next_bar` /
  `backtest.exit_after`; fills happen at next-bar adjusted open, engine-enforced.
- **Score out-of-sample windows warm.** Every holdout or walk-forward test backtest passes
  the LONG panel plus `score_from=<window start>` to `bt.evaluate` / `bt.run_strategy`.
  - Never pass a panel sliced to the window: that restarts every rolling indicator on its
    first bar.
  - Measured on informed_flow: 2 names ranked on holdout day one instead of 97; the
    walk-forward test windows read +9.3%/yr cold against +4.3%/yr warm.
  - A 200-bar filter is simply undefined inside a half-year window.
  - A STUDY can score warm too: `optuna.score_from` in a config (or `run_study(...,
    score_from=)`) runs every trial's signals on the whole panel and scores from that date —
    how `level_flow` skips its 20-session level warm-up (otherwise every trial sits flat
    against an invested benchmark, ≈ −3.7%/yr of excess, uniformly) and how a walk-forward
    fold trains warm. The journal is stamped with the window and refuses another one.
- **IBKR is never parallelized** — the historical-request cap is per account.
  One `fetch_batch` connection, paced. yfinance daily may fan out freely. IBKR answers one
  account's history requests one at a time (concurrent requests measured 0.6x the throughput), so
  a request can queue for minutes: `providers/ibkr.fetch_window` waits up to `REQUEST_TIMEOUT_S`
  (240 s) and asks a timed-out window again. ib_async's own 60 s timeout used to look like "the
  start of IB's history" and silently cut symbols short; a symbol that never answers is now left
  out (pending), never stored with a hole.
- **The database is shared** with the sibling `stock-trading` repo (TimescaleDB,
  port 5432). Never drop or ALTER a table, UPSERT only. This repo *does* define
  its own tables in `schema.sql` (prices + the alt-data tables `insider_trades`,
  `analyst_ratings`, `short_interest`) with `CREATE TABLE IF NOT EXISTS`,
  column-for-column identical to the sibling's — additive and idempotent, so it
  coexists with whatever the sibling created and can also populate a fresh DB
  standalone. `prices_ohlcv` PK is `(symbol, bar, ts)`; `ts` is bar OPEN time in
  UTC; `rth=false` marks pre/post-market bars. Alt-data ingestion is this repo's
  own (`providers/{sec,yf,finra}.py` + `tsim altdata`) — no runtime dependency
  on the sibling repo. `tsim init-db` also realigns the alt-data BIGSERIAL id
  sequences (a bulk load elsewhere can leave them stale, breaking inserts here).
- **Docs are load-bearing.** Google-style docstrings on every function; update
  README.md and the strategy READMEs when behavior changes; notebooks are real
  committed `.ipynb` (no generator scripts) and must run top-to-bottom.
- **Notebooks are committed with outputs cleared.** An embedded Plotly figure
  carries its whole raw series, so an un-cleared notebook runs to tens of MB.
  `.githooks/pre-commit` runs `scripts/nbclean.py` over every staged `.ipynb`
  and re-stages it, so this is automatic — but it strips *in place*, so never
  hand a user a "just commit it" instruction that assumes their rendered plots
  survive. Never work around the hook (`--no-verify`) to keep outputs.
- **Plotly only** for figures, and only inside `src/trading_simple/plots.py`
  or notebook cells. Follow the palette/design constants already in `plots.py`.
  Route every long time series through `plots._scatter` (or `_thin` + `_epoch` for
  a bar/marker trace): it thins to a min/max envelope of `MAX_POINTS` and packs the
  time axis as epoch ms, which is what keeps an executed 1m notebook at single-digit
  MB rather than ~80. A figure built inline in a notebook cell bypasses all of that —
  if it plots a full intraday series, thin it. **Never swap the envelope for `[::k]`
  striding**: the envelope keeps each bucket's min *and* max, so it cannot hide a
  spike, wick or drawdown low; striding can, silently. `plots.MAX_POINTS = None`
  draws every bar when someone genuinely needs to zoom.

## Adding a strategy

Use the `/new-strategy` skill (`.claude/skills/new-strategy/SKILL.md`) — it
encodes the full recipe and the validation gate. Summary: copy
`strategies/_template/`, write `signals()` from `backtest.py` helpers, mirror
every param in `config.yaml` (`params` + `optuna.space`), write a synthetic-data
test, adapt a strategy notebook, then pass:
`uv run pytest -q && uv run ruff check . && uv run tsim optimize --strategy <name> --n-trials 5`.

## Gotchas

- **level_flow (1m long/short level scanner) has FOUR notebooks and three engine hooks.**
  `level_flow_visual.ipynb` draws one symbol's levels, pivots, triggers and event outcomes
  (`strategy.explain_levels`) and optimizes nothing — pass it `{**strat.params, ...}`, never a
  bare call: `compose`/`explain_levels` fill unpassed knobs from `signals()`' defaults, which
  differ from the config; `level_flow_trendlines.ipynb` draws SLOPED levels from
  `trendlines.py` (a helper module the strategy loader does not see — load it by path; research
  only, `signals()` does not use it) with candles stamped at their CLOSE, so `ASOF` never sees an
  unfinished candle (daily bars are stored at 00:00 UTC of the trade date), on a LOG price axis
  (the lines are straight in log price); `level_flow.ipynb` optimizes (event study first — the
  polarity is *decided* there) and saves candidates; `level_flow_diagnostics.ipynb` runs the rolling walk-forward with per-fold
  re-optimization against a frozen and a no-tuning control, the warm holdout and the verdict.
  It scores `[alpha_cagr, volatility]`, sizes every entry with a per-entry `cash_frac`
  frame, scores its study warm via `optuna.score_from`, and can trade derived 5m/15m bars via
  `data.resample` — each execution bar is its own study. Its levels (HSAR pivot histogram)
  need up to `level_days` sessions of history, pinned by `warmup_sessions`; the stored 1m
  history starts 2025-06-18, which is why `level_days` is capped at 20.
- **burst_scan (activity bursts, long and short sub-strategies) has FIVE notebooks — run in this order.**
  - `burst_scan_visual` (does the code do what the README says; no statistics) → `burst_scan_events` (gate G0:
    the detectors against two placebos) → `burst_scan_model` (gates a–f: the classifiers) → `burst_scan`
    (optimize: the journal, the candidates file, the baseline comparison) → `burst_scan_diagnostics` (label ↔
    engine reconciliation, look-ahead and purge audits, costs, book anatomy, the one-shot holdout behind
    `READ_HOLDOUT = False`). Each loads its own panel (`SUBSET = 40` for a quick run) and ends with a `CHECKS`
    table and a `SCORECARD` line. Their shared computations live in `strategies/burst_scan/research.py`, loaded
    by path; `strategy.py` never imports it.
  - ONE strategy, two sub-strategies: every knob a side trains on is declared twice (`long_*` / `short_*`), the
    rest are shared; `sides: long | short | both` runs one alone. The two classifiers never see each other's labels.
  - The walk-forward runs INSIDE `signals()` (calendar-month folds; a month is scored by a model fit only on
    events whose outcome finished before it — the purge, needed because holds may run for days). Labels are the
    engine's own trade: barriers from the FILL bar's close, checked from the next bar, stop wins a tie, gaps fill
    at the open (vectorbt's `stop_entry_price="close"`), verified trade by trade in the tests and the diagnostics.
  - scikit-learn is imported inside `make_model()` (every `strategy.py` is loaded by the tests; a cold sklearn
    import costs seconds). A full-universe `signals()` call on 5m bars is seconds; the 1m context in
    `burst_scan_events` holds several GB — run one notebook at a time, never next to a `tsim` process.
  - Five research notebooks sit beside the chain: `options_free_data` (collects, compares and analyses FREE options
    data from Cboe / Yahoo / Nasdaq / OCC / DoltHub via `options_free.py`; data in git-ignored `options_data/`), `move_study`, `move_explorer`, `move_extended` (the pre- and
    post-market) and `move_detectors` (independent LONG / SHORT "+/-MOVE within WINDOW" detectors judged against a
    coin-flip market, `research.sign_surrogate`; see `ONSET_REPORT.md` for why they find movement, not direction). `DIRECTION_REPORT.md`
    is the full direction study: nothing predicts up vs down for these names, and that notebook's 2%/4h direction AUC is NOT a valid
    direction test, because it misses even a planted 35-40 bp edge. Judge direction on a 30-60 min IC / P&L against a coin flip that
    uses one sign per minute for all names. **Extended-hours 1m bars (`session="both"`) need two masks before any use**, both in
    `research.py`:
    - `mask_placeholders`: a third of extended minutes are volume-0 copies of the previous close.
    - `mask_late_reports`: 08:00-08:09 New York carries overnight trades reported late at stale prices. 82% of 08:00
      minutes stick out more than 25 bp. Unmasked, they faked a −33 bp, t −13.5 "pre-market reversal" that
      replicated across names.

    `Panel.rth` reads a missing bar as True; the stored flag ignores half-day closes.
- **darts_forecast (a Darts forecast of the next grid bars, walk-forward) has EIGHT notebooks.**
  `darts_forecast_walkthrough` (start here: 1-hour bars, a 5-day horizon, the whole pipeline in plain language up to a
  price forecast and an honest grade; no trading), `darts_forecast_explorer` (the baseline workbench: 12 rule-picked
  stocks, a multi-step model predicting the whole next PATH, an Optuna search over bar size × input length ×
  prediction length graded by the path's R² against "no change", winners read once on TEST; `SMOKE = True` runs it
  in about a minute), `darts_forecast_intraday` (its intraday sequel, helpers in `intraday.py` (`ix`; it loads
  explorer.py, the notebook uses `ix.ex` / `ix.rs`): one series per stock-day, no path past 16:00, no night in any
  target, every cell of bar size 1-10 min × input span × output span measured in MINUTES; `SMOKE = True`: two
  minutes), `darts_forecast_daystart` (helpers in `daystart.py` (`ds`; the notebook uses `ds.ex` / `ds.rs`): ONE
  example per stock-day, read from 09:30 + offset for L minutes, predict the next H, never past 16:00; bar size ×
  offset × input × output in MINUTES, every cell measured; optional volume and two previous-day inputs; `SMOKE =
  True`: under a minute), then for the 5-minute strategy in this order:
  `darts_forecast_data` → `darts_forecast_model` → `darts_forecast` (the study, trading rule only) →
  `darts_forecast_diagnostics` (TEST, stress, HOLDOUT gate). Their helpers live beside them, loaded by path:
  `research.py` (`rs`), `charts.py` (load it as `charts`: the data notebook uses `ch` for a DataFrame), `checks.py`
  (`ck`) and `explorer.py` (`ex`; it loads research.py itself, the explorer notebook uses `ex.rs`); no notebook
  defines functions.
  - Models: `gbm`, `gbm_clf`, `linear` (ridge), `tide`, `tsmixer`, `tft` (`strategy.DEEP` = the neural ones). The
    gradient-boosting fits go through `strategy.fast_bins`: scikit-learn 1.9 bins a weighted fit with one sort per
    percentile per input (a minute for 1,000 lagged inputs); for the walk-forward's 0/1 weights the edges equal the
    unweighted ones of the kept rows, so it computes those (the same model, tested; at most a last-bit rounding of an
    edge between two values; stock path above the binner's 200k-row subsample; overrides a PRIVATE method).
  - Knobs off in `config.yaml`, on in the explorer: `seasonal` (the same-clock-time mean of past completed outcomes,
    `seasonal_input`) and `lag_history` (a window's first labels read their lagged target from before it; off, the
    first `lookback + horizon − 1` labels of every window are dropped, which handicaps long input horizons). Every
    config param must also be a `signals()` keyword (`test_signals_accepts_every_config_param`).
  - Explorer traps (each measured): a network's predicted paths are louder than what it knows, so a raw
    path-MSE / R² search just rewards the quietest configuration. The paths are scaled by β (0–1, least squares on
    a held-out week before the graded months), training stops early on that week (best epoch kept, in memory), and
    direction (IC, z) is read on the unscaled paths. 1-minute bars bounce (lag-1 autocorrelation negative for all
    12 picks): an AR(1) baseline is graded beside every configuration. Bars mean different time spans per bar size,
    so a configuration needs ≥ 200 training windows, ≥ 10 independent validation paths and a held-out stretch of
    ≥ 5 independent paths per stock. Direction is the path correlation with an outcome-flip luck band (an R²-based
    z is biased when bar moves are autocorrelated).
  - Never scale or floor a target with a statistic over the whole panel: a `nanmedian` floor of the typical-move
    scale let TEST data into every TRAIN label. The explorer's truncation re-run caught it (only where the floor
    binds: U-shaped intraday volatility); `test_path_target_reads_no_later_bar_even_where_its_floor_binds`.
  - The explorer's window is inside TUNE by default; a window past it raises unless `READ_LOCKED = True`. 1-minute
    coverage widened from ~515 to ~1,750 names in March 2026, so "the last 6 months" is mostly the locked HOLDOUT.
  - Intraday notebook rules (each one a measured trap): grade EVERY bar size on ONE truth, the move from the FILL
    (the first trade after the decision; the decision bar's last trade carries the bid-ask bounce) to fixed
    checkpoints, in one typical-move unit built from 1-minute data (`intraday.scales`), never on each grid's own
    leads and units; mask volume-0 placeholder minutes first; fit the scale β month by month on EARLIER months only;
    give every baseline its own month-by-month β, fit the same way (an unscaled 20-day time-of-day mean scores below
    0 by construction: about −1/20 per stock, −0.01 to −0.02 pooled over the 12 stocks);
    pick the ridge's penalty by cross-validation over whole DAYS (a day's 12 decisions share most of their outcome:
    a penalty sized to the row count overfits); plant an INTRADAY effect as the positive control (a gap-driven drift
    is dominated by a few earnings gaps and only tests the static inputs). Half-days and the day after each are left
    out (`intraday.usable`): a half-day series is too short for Darts at long outputs and breaks the next input.
    TEST (one month) gets its own calibration stretch (`TEST_CAL_DAYS`: the refit holds out its last 20 sessions for
    β), since no earlier graded month can add to it as VALIDATION's later months get.
  - A luck band must keep every move's SIZE: flip signs, never shuffle days. A day's parts share its volatility, so a
    day-shuffle null is too narrow: in pure noise with day-to-day volatility the map put 10-22% of cells past 1.96
    (the first version reported a "significant" night → 14:00 cell this way). `intraday.atlas` flips one coin per
    day for every stock, `intraday.grade` one per 10-minute clock block; both in mirror pairs (centre exactly 0).
  - `rs.prune_cache()` deletes every `cache/*.pkl` without `-c<strategy.py version>-` in its name: the intraday
    results carry it, the explorer's `paths-*` / `explorer-*` files do not (a manual prune wipes them).
  - Darts sizes the samples of EVERY series by the LONGEST one: one series a bar too long silently doubles every
    other series' samples, one a bar short raises. `daystart.examples` cuts each stock-day to exactly (L + H) / g
    bars. Darts checks only the NUMBER of static covariates, never their order: build them in one function.
  - A sum over a differently laid-out copy of the same numbers differs in the LAST BIT: a scrambled panel's frames
    were column-major, which failed the bit-exact truncation re-run without any leak. `intraday.minute_arrays` lays
    every array out one way (`np.ascontiguousarray`) before any reduction.
  - plotly.js 3 (plotly 6.8) reads a NUMBER on a date axis in the viewer's time zone and a STRING as written:
    `plots._epoch` (epoch ms) shows a New York clock 1-2 h late in a European browser, and a night break (strings)
    then hides the last hour(s) of the session before it (engine proposal #37, pending). `charts._wall` sends text
    for the intraday pictures that cross a night. Optuna's `GridSampler` calls `study.stop()` when the grid is done:
    drive it with `study.optimize`, not ask/tell (which raises at the last trial).
  - Darts + torch are the optional `darts` extra (PyPI package `darts`, 0.48 since 2026-10-16; the old name `u8darts` stopped at 0.41). Darts 0.48 asks only for `scikit-learn>=1.6`, so the old override is gone.
  - The option inputs (free store, `burst_scan/options_free_data.ipynb`) are attached by `research.attach_options`
    in the notebooks only: a headless `tsim optimize` runs WITHOUT them (`forecast()` warns).
  - Forecasts are cached in-process and on disk (`cache/`, keyed by every forecast knob, the panel's full content
    and the strategy.py version: ANY strategy.py edit recomputes every walk-forward; `rs.prune_cache()`). The study
    searches only the trading rule, so each forecast config (model, `bar_minutes`, horizon, `retrain_minutes`,
    inputs) is its own study.
  - Multi-day horizons: `bar_minutes: 60` (7 buckets, the last 30 minutes), `overnight: true` changes ONLY the target
    (close-to-close over the horizon, nights included, ÷ causal 5-day vol), `expanding: true`. `valid()` rejects
    `overnight` (the trading rule is flat by the close); `forecast_valid()` checks forecaster knobs. Load such panels
    with `rs.load(..., adjust=True)`: the stored 1-minute bars are NOT dividend-adjusted.
  - At a multi-day horizon, forecasts made every hour share most of their outcome: day-clustered t-stats give 24–34%
    false alarms. Judge with the walkthrough's outcome-flip luck band (`rs.flip_outcomes` / `rs.scores`).
  - The stored "regular" 1m session keeps post-13:00 bars on half-days; `strategy.HALF_DAYS` masks them by hand and
    `strategy.HOLIDAYS` lists closures for dates past the data's end (extend both past 2027 / 2026).
- **informed_flow has TWO notebooks.**
  - `informed_flow.ipynb` runs the study on the training panel only — it tops it up TO
    `optuna.n_trials`, never past it — and saves its best distinct admissible trials to
    `studies/<study>.candidates.yaml` (git-ignored). Since v28 NOTHING is searched
    (`n_trials: 1`, like `baseline`): the one trial records the shipped rules.
  - `informed_flow_diagnostics.ipynb` compares any `CANDIDATES` from that file (or `"config"`)
    on the deep checks, the walk-forward and the warm holdout. It ends in a pass/fail verdict
    and an opt-in `config.yaml` promotion.
  - Both pick with `opt.pick_from_front` above `optuna.pick_min_trades_per_year`, one config
    value, so the walk-forward tests the rule that picked the candidate. A one-objective study's
    Pareto front is a single row, so they pick among ALL admissible trials
    (`opt.trials_frame`), which keeps the fallback past a too-quiet best trial.
  - It trades the point-in-time `Liquid500` (497 names since v35: SPY/GLD/SLV removed), every
    trial scored from `optuna.score_from: 2020-07-01`. Since v32-v38 (README "Study history",
    IDEAS.md "v34 → v40"): ONE sleeve, `short_interest` (a pre-registered t ≥ 3 rule; `ratings`
    fails it), days-to-cover AVERAGED over `dtc_window: 42` bars (v34; v35's exhaustive grid picked
    126, which v40 failed to replicate on the rolling universe — reverted), `neutral_groups: 5`, 30 slots, `q_in 0.90 / q_out 0.80`, open
    fills (live = market-on-open), `min_events: 1`. Rejected with measurements: `min_events: 0`
    (v37: admits interlisted names whose FINRA short interest covers only the US line), 20 slots
    (v38: the gain was a few winners), `Liquid1000` (v30). Do not add a `search:` back without a
    walk-forward in which tuning beats not tuning, and never search `q_out`, `max_positions` or
    `neutral_groups`.
  - The short-interest DATA has four read-time screens in `features.py`, all identity-only, never
    returns: a 45-day max age (`_SI_MAX_AGE_DAYS`), a per-bar coverage floor in the strategy
    (`_sleeve_rank`), rows filed under a ticker that then belonged to ANOTHER issuer dropped
    (`_SI_OTHER_ISSUER`: BNY was a muni fund until 2026-05 …), and a renamed company's own rows under
    its OLD ticker mapped to today's (`_SI_PREDECESSOR`: BK→BNY, FB→META …). A new name for either
    map is found by FINRA avg_daily_volume ÷ the stock's own volume: ≈ 1.00 and steady for the
    right issuer, JUMPING between settlements for another one (a split steps it once).
  - A Spearman rank-IC is UNCHANGED by cross-sectional demeaning — "demeaned" removes nothing
    from it. The notebook's fair check removes each name's trailing beta × the market's forward
    return and reads the IC per year, per dollar-volume tercile and by fifth of the ranking.
  - The walk-forward tests the WHOLE procedure: every 6 months from 2022 to the last bar it
    re-reads that signal rule on the period's own training data, holds the sleeves that pass
    (the equal-weight universe if none — never cash, since v32) and trades the fixed book rules.
    Since v36 each period trades a POINT-IN-TIME top 500 re-formed on the 6 months before it
    (`WF_UNIVERSE = "rolling"`), and the rule reads its IC POINT IN TIME (every past membership,
    each name counted only on the dates it was a member). Reading it on the names chosen at the
    END of the training window selects on the future and hides short interest (IC +0.022 vs
    +0.057 on 2020-07 → 2023-12) — the v36 defect. The gates read that **rule arm**; fixed arms,
    flips and §4c's ROAD are controls, never a menu. §4d compares every version with the last
    ADOPTED one re-run in the same walk-forward, PAIRED (same days, same bootstrap blocks, each
    side against its own market): PRIMARY = fixed rules, CLEAN = 2025+ only, LATE = +50 bp/leg,
    a same-decision split and a leave-top-5-out check. Adoption classes and bars are in IDEAS.md
    ("Protocol v2"); the verdict prints one `SCORECARD {json}` line per run. A decision made once
    on the train window and then "walked forward" through the same years is in-sample for it.
  - The verdict's hard gates: the signal rule on train, the rule arm's stitched record (alpha
    with a block-bootstrap P(alpha > 0), beats buy-and-hold, return ÷ volatility ≥ the market's),
    the rule arm against symbol-shuffled books over the WHOLE walk-forward (shuffled inside
    dollar-volume tenths, stitched the same way — the strongest skill test there), shuffled books
    on the HOLDOUT window (a WARNING since v32), completed holds (not vectorbt lots), alpha at the real order size,
    at 2× costs, with every leg 50 bp worse than the open (late fills) and in whole shares at `ACCOUNT` (since v32), no stability spike (§3d draws the neighbourhood over the pinned book rules and
    validates neighbours on top of `config.yaml` — `strategy.valid` counts the composite weights,
    so a bare partial dict is always rejected and the table comes back empty, as v27's did).
    Everything measured on the 2025+ holdout alone is a
    WARNING — that window has been read under eight study versions — as are the risk limits and
    the pessimistic-fill stress. Its random baseline SHUFFLES SYMBOLS; never go back to per-bar
    random ranks, which churn the book daily and make the gate unfailable (v25: z = +267).
  - informed_flow has no `grid:` lists and no coarse sweep. The other strategies still have
    one notebook each.

- **overnight_open (night → first trading hour, research folder, TWO marimo notebooks).**
  `overnight_open_explore_marimo.py` shows one night; `overnight_open_study_marimo.py` holds the
  connection study and its one-shot validation (V1-V4, pre-registered in its README). Data facts:
  - Extended-hours 1m bars exist for ~535 names, 2025-06-18 → 2026-06-26 only (7 names after).
    None of them prints 20:00-04:00.
  - The 16:00 minute holds the closing auction; the 1d close IS the official close, on the 1m
    bars' basis (split-adjusted, no dividends).
  - MarketData's Benzinga news is DATE-only (all items stamped 00:00).
  - **EDGAR's submissions-JSON `acceptanceDateTime` is late by New York's UTC offset for about
    half the filers** (MU, AFRM, AAOI: 16:02 reads 20:02). The filing index page is right:
    `research.unshift` plus a per-filer index-page check. `providers/sec.py` stores the raw stamp
    as insider `known_at` (late, so conservative; proposal in the folder's IDEAS.md).
- **`collectors/` is not the framework.** Its scripts are self-contained PEP 723 scripts
  (`uv run --script`). They never import `trading_simple` and write their own TimescaleDB
  (`collectors/docker-compose.yml`, `market_depth_db`), not the shared `trading` DB, so a 24/7
  VM needs only Docker. `binance_depth.py` rules, pinned by `tests/test_binance_depth.py`:
  - futures continuity is `pu == previous u`; `U` is never previous `u` + 1;
  - levels past the REST snapshot's 1000 are a partial tail (only levels touched since the sync),
    so saved levels are clipped to the snapshot's price range and the book is rebuilt every 30 s.
- mypy is not part of the gates here; rely on pytest + ruff.
- Env vars beat YAML config: `TRADING_<SECTION>__<KEY>` (e.g.
  `TRADING_DB__HOST=otherhost`); a `.env` at the repo root is read automatically.
- psycopg cannot infer NULL parameter types — optional SQL params need explicit
  `CAST(:param AS <type>)` (see `data.py` for the pattern).
- vectorbt 1.0 pins: `pandas < 3.0`, `numpy < 2.5`. The Rust engine comes from
  the `vectorbt-rust` package (`engine: auto` in settings dispatches to it).
- Optuna studies are append-only journal files under `strategies/*/studies/`
  (gitignored). Deleting the file resets the study; same name = resume.
- `uv.lock` is committed for reproducible installs; after changing dependencies
  in `pyproject.toml` run `uv lock` (or `uv sync`) to refresh it.
- **No literal `%` in `schema.sql`** — it is applied via `exec_driver_sql`, and
  psycopg reads `%` as a parameter placeholder, so even a `%` inside a `--` comment
  breaks `tsim init-db`.
- **Plotly `add_hrect`/`add_vrect`/`add_hline` default to `exclude_empty_subplots=True`**,
  so a shape aimed at a subplot that has no *trace yet* is dropped **silently** — no error,
  it just never renders. In a `make_subplots` figure this means bands/spans/reference lines
  must be added **after** the `add_trace` calls for their subplot (`layer="below"` still
  keeps them behind the data). This silently blanked the informed_flow "in or out of the
  book" figure's IN/HOLD bands and position shading. Also: `-open`/`-thin` marker symbols
  have no fill and are stroked from `marker.line.color` (not `marker.color`) — set both, or
  they render the default slate. And a shape does not extend an axis range, but the data
  behind it does: pin `update_xaxes(range=[...])` when a `add_vrect` could reach past a zoom.
- Yahoo's `upgrades_downgrades` frame carries **three undocumented price-target
  columns** (`priceTargetAction`, `currentPriceTarget`, `priorPriceTarget`) that
  yfinance forwards but does not document — `providers/yf.fetch_price_targets` reads
  them from the *same* response as `fetch_ratings`, so the sleeve costs no extra
  requests. They can vanish without notice; ingested rows stay valid. `priorPriceTarget
  == 0.0` means *no prior target*, so it is stored NULL — read as zero it becomes a
  −100% revision. Rank the **revision**, never the target-to-price **level**
  (Han-Kang 2022: price drifts *against* the level, worst in large caps).
- **FINRA's bulk files are pipe-delimited with NO quoting convention**, so `csv` must
  read them with `quoting=csv.QUOTE_NONE` (`providers/finra._parse_bulk_file`). A `"` in
  an issuer name (`"ELEMENTS ""Dogs of the Dow""`) otherwise opens a field that never
  closes and swallows the rest of the file — surfacing as `field larger than field limit`
  on a file whose longest line is 278 chars, and aborting the backfill mid-run. **Never
  answer that error by raising `csv.field_size_limit`**: it "succeeds" while merging
  15,398 rows into 3,881. This silently cost the DB its whole 2018-2019 short-interest
  history until 2026-08-02.
- SEC Form 4 `transactionDate` values are **sometimes wrong in the filing itself** (filers
  keying 2029 for 2020 — 7 such rows are in the DB). Ingestion stores them verbatim, which
  is correct for a point-in-time archive, and nothing downstream breaks because
  `features.py` buckets every sleeve by `known_at` (the filing timestamp), never by the
  event date. Do not "repair" them. The same trade can also be stored TWICE: SEC's quarterly bulk
  data sets round prices to cents where the filing's XML keeps four decimals, and
  `insider_trades_uq` includes the price (68,270 of 757,444 rows on 2026-10-20). Again, never
  repair the table: `features._INSIDER_SQL` keeps one copy per trade (the later `known_at`).
- **SEC XBRL fundamentals are screened at read time, never repaired in the table.** Filers
  make thousand-fold unit slips in share counts (AJG filed 1.9e14 shares for two 2020
  quarters), file zeros and 100-share shell stubs, and abandon tags: multi-class issuers
  stopped filing a non-dimensional cover-page count, which froze V, MA and CMCSA at 2010
  while the parser took the first tag that ever had data. So
  `providers/sec._parse_companyfacts` resolves tag fallbacks per filing (instants) and per
  period (flows, where the EARLIEST filing wins — DEF 14A proxies re-tag net income weeks
  after the 10-K), and `features.py` drops implausible share counts (`_screen_shares`:
  outside `_SHARES_BAND`, or furnished on a 6-K — RY's 2019 6-Ks tagged thousands), keeps
  one row per day that never moves a level backwards (`_newest_period`) and masks levels older
  than `_FUND_MAX_AGE_DAYS`. Every rule is one-sided: never fix a filing by looking at the
  NEXT one, or a train-only load and a full load disagree. `fundamentals` upserts are
  `DO NOTHING`, so a parser change only ever ADDS rows for new keys — dry-run it against
  the DB before re-ingesting (via `notebooks/03_altdata_update.py`, `REFRESH = True`).
- SEC EDGAR (insider provider) requires a User-Agent with a contact email or it
  returns 403; a working default is set, override with `TRADING_SEC_USER_AGENT`.
  Form 4 `primaryDocument` points at the XSL-styled HTML — fetch the raw XML at
  the filing root (`providers/sec._raw_doc`). Ingestion is paced/single-process
  (SEC & FINRA rate-limit per IP), like the IBKR rule.
- Alt-data features align to the panel by UTC session date and are `shift(1)`
  lagged in `features.py` (conservative point-in-time); this supports daily bars.
