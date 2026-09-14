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
uv run tsim altdata --dataset all --universe LiquidLeaders --since 2020-01-01  # ingest alt-data (prefer notebooks/03_altdata_update.ipynb — it audits coverage)
git config core.hooksPath .githooks   # once per clone — enables the notebook-clearing pre-commit hook
git ls-files '*.ipynb' | xargs python3 scripts/nbclean.py   # clear notebook outputs by hand
```

## Hard rules

- **The strategy contract is sacred.** A strategy = `strategies/<name>/` with a
  pure `signals(panel, **params) -> dict` (+ optional `valid`, `indicators`)
  and a `config.yaml` with `data / optuna / params` blocks (+ optional `backtest:
  {fill: open | pessimistic}` — `pessimistic` buys at the bar's high and sells at its
  low, benchmark included; notebooks pass `fill=strat.fill` to every direct
  `bt.run` / `bt.benchmark` / `bt.active_stats` call). Adding a
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
  A signal strategy returns `cash_frac` (a scalar) alongside `entries`/`exits` to run
  the **cash-shared wallet** variant: ONE `init_cash` book, each entry deploying that
  fraction of the free cash — total invested can never exceed the wallet, the way a
  real single account trades a multi-symbol scanner. **The wallet (or portfolio mode)
  is the repo default — every shipped strategy trades one shared $10k book**;
  per-symbol books are a diagnostic only (drop `cash_frac`). Wallet-mode shorts work,
  but percent sizing cannot reverse: an opposite-direction entry closes the position
  (`upon_opposite_entry="close"`), never flips it.
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
  therefore scores `excess_cagr` (maximized) against `tracking_error` (minimized), both
  relative to an equal-weight buy-and-hold of the same panel (`backtest.benchmark`), and
  `pick_from_front` only picks front rows with `excess_cagr > 0` (`MIN_PICK_EXCESS`);
  absolute Sharpe/return are diagnostics only. (The old `info_ratio`/`excess_cagr` pair
  correlated 0.975 and collapsed fronts to 1–2 points. Studies are stamped with their
  objective names, and a journal scored on other objectives is refused — bump the study.)
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
  `features.py` (proposal-only, like any `src/` change). See `informed_flow`.
- **ALL alt-data ingestion goes through `notebooks/03_altdata_update.ipynb`** —
  incremental *and* historic backfill (`MODE = "backfill"`). Do not hand the user an
  ad-hoc `tsim altdata` recipe for a backfill; add it to the notebook instead. The
  notebook also audits coverage, and that audit is load-bearing: **a thin alt-data
  table does not fail loudly, it goes quiet.** A cross-sectional ranker imputes a
  neutral median rank where a name has no data, so a sleeve covering 3 of 100 names
  looks scored while actually being a huge arbitrary tilt on those 3 (this shipped,
  in `short_interest`). Check breadth and depth, never just freshness.
- **Engine fixes are proposal-only.** When you spot a bug or improvement in
  `src/trading_simple/` while doing other work, report it with a concrete
  suggested fix — implement only after explicit approval. Keeps strategy
  diffs clean.
- **No look-ahead.** Signals must be execution-timed via `backtest.next_bar` /
  `backtest.exit_after`; fills happen at next-bar adjusted open, engine-enforced.
- **IBKR is never parallelized** — the historical-request cap is per account.
  One `fetch_batch` connection, paced. yfinance daily may fan out freely.
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
  event date. Do not "repair" them.
- SEC EDGAR (insider provider) requires a User-Agent with a contact email or it
  returns 403; a working default is set, override with `TRADING_SEC_USER_AGENT`.
  Form 4 `primaryDocument` points at the XSL-styled HTML — fetch the raw XML at
  the filing root (`providers/sec._raw_doc`). Ingestion is paced/single-process
  (SEC & FINRA rate-limit per IP), like the IBKR rule.
- Alt-data features align to the panel by UTC session date and are `shift(1)`
  lagged in `features.py` (conservative point-in-time); this supports daily bars.
