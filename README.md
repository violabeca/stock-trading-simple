# stock-trading-simple

A small, readable framework to **backtest and optimize trading strategies**.
Prices live in a TimescaleDB, strategies are pure functions in their own
folders, one shared engine (vectorbt) runs everything, and Optuna tunes
parameters with kill-safe, resumable studies. If you can write a pandas
expression, you can add a strategy.

## The big picture

```mermaid
flowchart LR
    A["📓 01_universe_screener<br>find symbols"] -->|universes.yaml| B["📓 02_backfill<br>load history"]
    B -->|"1d: yfinance (parallel)<br>1m: IBKR (paced)"| C[("TimescaleDB<br>prices_ohlcv")]
    C -->|data.load_panel| D["backtest.py<br>ONE vectorbt engine"]
    S["strategies/&lt;name&gt;/<br>strategy.py + config.yaml"] --> D
    D <--> E["optimize.py<br>Optuna, resumable"]
    D --> F["plots.py<br>plotly figures"]
```

The flow is always the same: **screen → backfill → backtest → optimize →
inspect**. Notebooks drive it interactively; the `tsim` CLI does the heavy
lifting in subprocesses so the same code path runs unattended.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`, or
`curl -LsSf https://astral.sh/uv/install.sh | sh`) — it manages the Python
3.13 interpreter and the `.venv` for you (pinned in `.python-version`).

```bash
# 1. Install — creates .venv and installs every extra (ibkr, notebook, dev)
cd stock-trading-simple
uv sync --all-extras

# 2. Register the Jupyter kernel once, then select "trading-simple" in your editor
uv run python -m ipykernel install --user --name trading-simple --display-name "trading-simple"

# 3. Check the shared database is reachable (it must already be running)
uv run tsim coverage

# 4. Build a universe, then load its history
#    (open in Jupyter/VS Code: notebooks/01_universe_screener.ipynb, then 02_backfill.ipynb)

# 5. Run a strategy end to end
#    open strategies/volume_momentum/volume_momentum.ipynb  →  Run All

# 6. Offline sanity gates (no DB or network needed)
uv run pytest -q && uv run ruff check .
```

Prefer an activated shell? `source .venv/bin/activate` after step 1, then drop
the `uv run` prefix from every command in this README.

## Repo map

```
config/settings.yaml          global knobs: database, IBKR, costs, engine     ─┐ configuration
config/universes.yaml         named symbol lists written by the screener      ─┘
src/trading_simple/
  config.py                   loads the YAML + TRADING_* env overrides
  db.py, schema.sql           engine + idempotent UPSERT writers (shared DB)
  exchanges.py                venue registry: sessions, currencies, suffixes
  providers/yf.py             daily bars from Yahoo (free, parallel-safe)
  providers/ibkr.py           1-minute bars from Interactive Brokers (paced)
  universe.py                 Nasdaq screener → filtered, sector-capped lists
  data.py                     DB → Panel (wide adjusted OHLCV matrices)
  backtest.py                 THE engine: run / sweep / metrics / helpers
  optimize.py                 Optuna: journal storage, Pareto front, pick_best, boundary check
  plots.py                    every figure, one consistent plotly design
  strategies.py               loads a strategy folder by name
  jobs.py                     notebooks → CLI subprocess handoff
  cli.py                      the `tsim` command line
strategies/
  _template/                  copy me to start a new strategy
  volume_momentum/            strategy 1: intraday volume spikes + momentum  (+ notebook)
  donchian_breakout/          strategy 2: intraday breakouts + ATR trailing stops  (+ notebook)
notebooks/
  01_universe_screener.ipynb  find symbols worth loading
  02_backfill.ipynb           load daily AND intraday history
tests/                        55 offline tests (synthetic data, ~5 s)
```

## Concepts in plain language

**Bars and the `rth` flag.** A *bar* is one candle (open/high/low/close/volume)
at some size — `1d` daily, `1m` one-minute. Every stored bar carries `rth`
(*regular trading hours*): `true` for the normal session (09:30–16:00 New
York), `false` for pre/post-market. Both live side by side in the database;
strategies choose at load time with `data.session: regular | extended | both`.
Extended-hours collection is **off by default** (`ibkr.extended_hours` in
settings) — turn it on when you want to explore after-hours behavior.

**Session-consistent indicators.** Intraday volume is U-shaped (huge at the
open, quiet at lunch) and extended-hours volume is thin. An indicator that
averages across all of that is comparing apples to oranges. Two defenses:
loading a single session (the default) makes ordinary rolling windows fair by
construction; and for "is this volume abnormal?" questions the engine provides
`backtest.volume_profile` — the expected volume **at that time of day** over
the last N sessions (optionally smoothed ±N minutes around it), so 09:31 only
ever compares to other 09:31s. It groups by exchange-local clock (DST-safe)
and uses the median (one earnings blowup doesn't distort the next two weeks).
For session-edge discipline there are `backtest.session_mask` (no entries in
the first/last N minutes) and `backtest.session_end` (force intraday
strategies flat on each day's final bar) — both data-driven, so half-days
need no session table.

**Adjusted prices, next-bar fills.** All price fields are split/dividend
adjusted into one consistent space. Signals computed on bar *t* execute at bar
*t+1*'s open — never at the price that produced the signal. This "no
look-ahead" rule is enforced by the engine, not by convention.

**Costs are always on.** Every backtest charges commission, half the quoted
spread, and slippage per leg (the `costs:` block in settings). An edge that
dies from 9 bps of round-trip friction was never an edge.

**Stops.** A strategy may return `sl_stop` / `tp_stop` (stop distance as a
fraction of entry price — e.g. an ATR multiple) and `sl_trail: true` to make
the stop follow the best price since entry. Stops are evaluated *intrabar*
against the high/low, not just at closes.

**The Pareto front.** Optimization maximizes **Sharpe and total return
together**. No single "best" exists — the result is the set of parameter sets
where improving one objective must hurt the other (the Pareto front).
`optimize.pick_best(study, weights=(1.0, 0.5))` reduces it to one trial under
an explicit preference, and `optimize.boundary_report` flags picked parameters
sitting on a search-space bound (log-aware) — evidence the optimum may lie
outside the searched box, so widen that bound and re-run. Beware: more trials
on the same data = more chances to overfit; judge winners on `pct_profitable`
(breadth across symbols) and on held-out data.

## How to add a strategy

No engine edits — ever. Three files in one folder:

1. **Copy the template:** `cp -r strategies/_template strategies/my_idea`
2. **Write `signals()`** in `strategy.py` — a pure function from a price
   `Panel` to entry/exit frames, built from the engine helpers
   (`sma`, `atr`, `volume_profile`, `session_mask`, `session_end`,
   `next_bar`, `exit_after`):

   ```python
   def signals(panel, *, window: int = 20) -> dict:
       breakout = panel.close > bt.sma(panel.close, window)
       entries = bt.next_bar(breakout & ~breakout.shift(1, fill_value=False))
       return {"entries": entries, "exits": bt.exit_after(entries, 10)}
   ```
3. **Fill `config.yaml`** — every `signals()` argument appears under `params:`
   (defaults) and `optuna.space:` (search ranges); optionally `sweep:` lists.

Then copy either strategy notebook, change the name in `strategies.load(...)`,
and Run All — the load/backtest/sweep/optimize/inspect flow is identical for
every strategy. Working with Claude Code? The `/new-strategy` skill walks
through exactly this recipe with a validation gate at the end.

## Configuration

* **Global** (`config/settings.yaml`): database, IBKR connection/pacing,
  trading costs, engine choice, per-symbol exchange overrides, hand-edited
  universes. Every key maps 1:1 onto `src/trading_simple/config.py`.
* **Per strategy** (`strategies/<name>/config.yaml`): four blocks —
  `data` (which panel to load), `params` (single-run defaults),
  `sweep` (grid lists), `optuna` (study name, sampler, trials, search space).
* **Environment overrides** beat YAML: `TRADING_<SECTION>__<KEY>`, e.g.
  `TRADING_IBKR__PORT=4001`, `TRADING_DB__HOST=otherhost`. A `.env` file at
  the repo root is read automatically (see `.env.example`).

## CLI reference

Prefix any of these with `uv run` (or activate `.venv` first):

```
tsim coverage  [--universe X] [--bar 1d]      what's stored (inventory / per symbol)
tsim screen    --name LiquidLeaders --top 100 build + persist a universe
tsim import-universes [--names A,B]           copy universes built by the sibling repo
tsim subset    --universe HighOctane --top 20 pin the most liquid names as <U>Top<N>
                                              (by trailing-90d avg dollar volume)
tsim backfill  --universe X --bar 1d --since 2015-01-01
               --bar 1m [--extended-hours]    provider picked by bar (yfinance/IBKR)
tsim optimize  --strategy volume_momentum [--n-trials 300] [--study S] [--sampler tpe]
tsim init-db                                  apply schema.sql (no-op on the live DB)
```

All output is plain parseable text; `jobs.py` is how notebooks stream it live.
`tsim optimize` prints one line per finished trial (objectives, front size,
parameters) and a ready-to-paste `uvx optuna-dashboard <journal>` command for
a live web view of the running study.

## Speed

* **vectorbt with the Rust engine**: `backtest.engine: auto` dispatches each
  computation to the compiled Rust kernels (`vectorbt-rust`) when supported,
  numba otherwise. One YAML word switches (`rust` | `numba` | `auto`).
* **Chunked column-stacking**: a parameter sweep turns N combos × M symbols
  into the columns of a *single* vectorbt call per chunk (`backtest.chunk_size`),
  instead of N separate calls.
* **Subprocess handoff**: notebooks never do heavy work in-kernel — backfills
  fan out as parallel `tsim` shards (yfinance) and optimizations run as
  `tsim optimize` workers, several of which may share one study.
* **The one anti-rule**: IBKR fetches are *never* parallelized — the request
  cap is per account, so one paced connection is both the safe and the fast
  option.

## Testing & conventions

* `uv run pytest -q` — offline, synthetic data, ~5 seconds. `uv run ruff check .` — style.
* Google-style docstrings on every function; plotly for every figure;
  functions over classes; fewer lines wins.
* Strategy folders own everything strategy-specific (code, config, docs,
  notebook, studies). Engine modules never mention a concrete strategy.

## The shared database

This repo reads and writes the **same TimescaleDB** as the sibling
`stock-trading` platform (`postgresql://trading:trading@localhost:5432/trading`,
tables `securities` + `prices_ohlcv`) — so history backfilled by either repo
serves both. The sibling's screener also builds richer characteristic
universes (**HighOctane** = high volatility, **Trailblazers** = momentum,
**SteadyEddies** = low volatility, **CheapQuality** = value); `tsim
import-universes` copies those lists here, and `tsim subset` carves a pinned
liquid top-N out of any of them (how `HighOctaneTop20`, the donchian_breakout
universe, is defined). It never creates or migrates that database; `tsim
init-db` merely re-applies the identical idempotent schema. Do not point it at
a fresh Postgres without TimescaleDB installed.
