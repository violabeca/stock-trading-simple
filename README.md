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
    B -->|"1d: yfinance (parallel)<br>1m: IBKR (paced)"| C[("TimescaleDB<br>prices_ohlcv + alt-data")]
    G["📓 03_altdata_update<br>insider · ratings · price targets · short interest"] -->|"SEC · Yahoo · FINRA"| C
    C -->|data.load_panel| D["backtest.py<br>ONE vectorbt engine"]
    C -.->|features.attach_features| D
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

# 2b. Enable the git hooks once — clears notebook outputs on commit (see "Committing notebooks")
git config core.hooksPath .githooks

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
  providers/yf.py             daily bars + analyst ratings AND price targets from Yahoo (free)
  providers/ibkr.py           1-minute bars from Interactive Brokers (paced)
  providers/sec.py            SEC EDGAR: Form 4 insider trades + XBRL companyfacts fundamentals (free, PIT-clean)
  providers/finra.py          FINRA short interest / days-to-cover (free)
  universe.py                 Nasdaq screener → filtered, sector-capped lists
  data.py                     DB → Panel (wide adjusted OHLCV matrices)
  features.py                 DB alt-data tables → Panel.features matrices
  backtest.py                 THE engine: run / sweep / metrics / helpers
  optimize.py                 Optuna: journal storage, Pareto front, one picking rule, boundary + stability checks
  plots.py                    every figure, one consistent plotly design
  strategies.py               loads a strategy folder by name
  jobs.py                     notebooks → CLI subprocess handoff
  cli.py                      the `tsim` command line
strategies/
  _template/                  copy me to start a new strategy
  baseline/                   equal-weight buy-and-hold: the benchmark made visible (no notebook — see 04_compare)
  volume_momentum/            strategy 1: intraday volume spikes + momentum  (+ notebook)
  donchian_breakout/          strategy 2: intraday breakouts + ATR trailing stops  (+ notebook)
  keltner_reversion/          strategy 3: intraday dip-buying at the lower Keltner band  (+ notebook)
  squeeze_breakout/           strategy 4: intraday breakouts gated by squeeze/volume/trend confluence  (+ notebook)
  informed_flow/              strategy 5: DAILY alt-data composite (insider + ratings + price targets + short interest)  (+ notebook)
  candle_scalp/               strategy 6: intraday reversal candles at a stretched location, built to be falsified  (+ notebook)
  support_bounce/             strategy 7: intraday tested-support mean reversion  (+ notebook)
  value_factor/               strategy 8: DAILY cross-sectional value + quality on SEC XBRL fundamentals  (+ notebook)
  conviction_flow/            strategy 9: informed_flow plus selectivity gates (quality, agreement, score floor)  (+ notebook)
  gap_and_go/                 strategy 10: intraday opening-range breakout on overnight gaps  (+ notebook)
notebooks/
  01_universe_screener.ipynb  find symbols worth loading
  02_backfill.ipynb           load daily AND intraday price history
  03_altdata_update.ipynb     ingest + audit alternative data (insider, ratings, price targets, short interest, fundamentals)
  04_compare.ipynb            the cross-strategy scoreboard: every strategy vs its buy-and-hold baseline
scripts/nbclean.py            clears notebook outputs; run on staged .ipynb by the pre-commit hook
tests/                        offline tests (synthetic data, ~5 s)
```

## Committing notebooks

Notebooks are committed as **source only, with their outputs cleared**. A single
Plotly figure embeds the whole raw series behind it, so one backtest chart can
weigh tens of megabytes — left unchecked the notebooks here reached 314 MB, and
cleared they total 212 KB. Nothing is lost: every notebook runs top-to-bottom, so
its outputs are always one *Run All* away.

`.githooks/pre-commit` enforces this — it runs `scripts/nbclean.py` over any
staged `.ipynb` and re-stages the cleared file, so a fat notebook cannot reach a
commit by accident. It strips **in place**, meaning your own copy's plots are
cleared too when you commit; re-run the notebook to bring them back.

Enable the hook once per clone (step 2b above); to clear notebooks by hand:

```bash
git ls-files '*.ipynb' | xargs python3 scripts/nbclean.py   # strip in place
python3 scripts/nbclean.py --check <nb>                     # exit 1 if it has outputs
```

### Why an executed notebook stays small

The hook keeps fat notebooks out of git, but it does not make *your* running copy
pleasant. `plots.py` does that, with two measures that between them took an
executed 1-minute notebook from ~80 MB to single-digit megabytes:

- **The time axis ships as epoch milliseconds.** Plotly base64-packs float arrays
  but writes a datetime index out as ISO text — ~29 bytes a point, repeated in
  *every* trace. Packing it costs nothing: the axis, its ticks and the `%{x|…}`
  hovers are unchanged.
- **Long series are thinned to a min/max envelope** of `plots.MAX_POINTS` (10,000)
  points. Each bucket keeps its minimum *and* its maximum, so **no spike, wick or
  drawdown low can be hidden** — a guarantee plain `[::k]` striding cannot make.
  An 82,680-bar panel is ~70 points per pixel at any sane figure width, so there is
  nothing on screen to lose.

Zooming in then shows the envelope rather than every individual bar. When you want
the bars, set the knob and re-run the cell — the figure is still half its old size,
from the epoch packing alone:

```python
from trading_simple import plots
plots.MAX_POINTS = None   # draw every bar
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

**Alternative data.** Beyond prices, the shared database holds *non-OHLCV*
signals — corporate-insider trades (SEC Form 4), analyst rating changes and
analyst price-target revisions (Yahoo), short interest (FINRA), and company
**fundamentals** (SEC EDGAR XBRL `companyfacts`: equity, assets, shares, and
trailing-twelve-month net income and gross profit) — each ingested by this repo's
own providers (no dependency on the sibling repo) and stored point-in-time: every
row carries a `known_at` (when it became public — for fundamentals, the filing
date), so features never see a value before it existed. A strategy opts in with a
`data.features:` list in its config; `features.attach_features` then rides those
matrices on `Panel.features` (one wide *date × symbol* frame per signal, already
lagged), and `signals()` reads them like any price field — forming price-relative
ratios (e.g. book-to-price = equity ÷ market cap) there, since price lives on the
Panel. The `informed_flow` strategy ranks stocks on the insider/ratings/short-
interest composite; `value_factor` ranks them on a value + quality composite from
the fundamentals sleeve. Adding a new alt-data strategy needs no engine change —
just a folder and a `data.features:` line; adding a new *feature type* is one
builder in `features.py` (plus its provider), like the fundamentals sleeve.

**`03_altdata_update.ipynb` is the single entry point for alt-data ingestion**,
incremental *and* historic (`MODE = "backfill"`). The insider backfill pulls SEC's
**quarterly Form 3/4/5 bulk data sets** (`tsim altdata --bulk`) — one ~14 MB download
per quarter serves every issuer, turning a days-long per-filing scrape into minutes
while recording the same per-symbol coverage ledger, so a re-run skips the solid
history; the live per-filing scraper stays the incremental path for the current,
unpublished quarter. It also **audits coverage**, which
matters more than it sounds: a half-ingested table does not fail loudly, it goes
quiet. Because a cross-sectional ranker imputes a *neutral median* rank wherever a
name has no data, a sleeve covering a handful of symbols still looks scored — it
silently becomes a large arbitrary tilt on those few names and noise for everyone
else. So the notebook checks **breadth and depth, not just freshness**, and
`informed_flow` drops any sleeve covering less than half the universe.

**Two execution models.** A strategy returning `entries`/`exits` runs **per-symbol**:
each symbol trades its own `init_cash` in its own account. A strategy returning a
`weights` frame (target fraction of the book per symbol) runs **portfolio mode**: one
shared cash account of `init_cash` *total*, so buying a name must be funded by selling
another. Per-symbol is right for a per-symbol signal — and quietly fatal for a
cross-sectional ranker, because the cash sitting against a name the strategy is *flat*
in can never be spent elsewhere. A book holding 40 of 100 names is then structurally
~30% invested against a 100%-invested benchmark, and cannot beat it however well it
picks. (`informed_flow` lost 500 out of 500 optimizer trials to buy-and-hold on exactly
this, and beat it on 928 of 1038 once the money was pooled.) In portfolio mode the
dollar figure is a *reporting scale* — with fractional shares everything is
scale-invariant — so what actually constrains the book is the number of positions it
may hold, not the `$10,000`. Rebalancing is **event-driven per name**: a target is
placed only when a name's own weight changes (`backtest.rotation_targets` masks per
name), so a held position is bought once, left to drift while other names rotate around
it, and sold whole when it leaves. The one exception is an opt-in **concentration cap**:
a portfolio strategy that returns a `retrim_k` scalar alongside its `weights` has any held
name trimmed back to its slot once its realized book weight reaches `retrim_k ×` its
target (e.g. `2` = double), so a single runaway winner cannot come to dominate the book;
the freed cash rotates into the next name the strategy picks. The cap is off by default
(`backtest._retrim_loop`), so the benchmark and every strategy that omits it are unaffected.

There is a third, wallet-shaped variant of the signal path — and it is **the repo
default**: a strategy returning a scalar `cash_frac` alongside `entries`/`exits` runs
against **one cash-shared wallet** of `init_cash` *total*, each entry deploying that
fraction of the currently free cash — so total invested can never exceed the wallet,
exactly how a single real account would scan many names. Every shipped signal strategy
trades this way; per-symbol books remain available only as a diagnostic (drop
`cash_frac` from the returned dict). One wallet-mode rule to know: percent sizing
cannot *reverse* a position in one order, so an opposite-direction entry closes the
open position instead (`upon_opposite_entry="close"` in the engine); the new direction
opens on a later bar if its signal persists.

**The objectives: benchmark-relative, and per year.** Optimization maximizes
`info_ratio` and `excess_cagr` **together**, both measured against an equal-weight
buy-and-hold of the same panel. Two traps are being avoided at once. *Absolute* Sharpe
rewards market exposure rather than skill — a long-only strategy in a rising market
earns one just by being invested. And a *cumulative* excess return rewards a long
backtest for being long: it can compare neither two strategies (windows here run from
one month to four years) nor, less obviously, two walk-forward folds of unequal length
inside one strategy. So every return metric is annualized (`cagr`, `excess_cagr`), and
`years` and `trades_per_year` travel beside them — annualizing makes a short backtest
*comparable*, never *trustworthy*.

No single "best" exists — the result is the set of parameter sets where improving one
objective must hurt the other (the Pareto front). `optimize.pick_from_front(front)`
reduces it to one row (and `optimize.pick_best(study)` does the same from a study),
under `optimize.PICK_WEIGHTS` — **one balanced preference for the whole repo**, because
this used to be a per-notebook literal and drifted into three different answers, one of
them arguing from the pre-v5 objectives. Pass `weights=` explicitly if a strategy needs
a different trade-off, and say why at the call site.

Both first collapse the front with `optimize.distinct_front`, which keeps one row per
distinct *objective pair*. This is not cosmetic: an **inert** parameter — one that cannot
affect the book, like `informed_flow`'s `max_hold` at a low displacement margin — lets the
sampler mint unlimited "distinct" front entries that are all the same strategy. v17's
front was 35 rows and **one** outcome; v18's was 55 rows and 32. Any statistic taken over
the front is otherwise multiplicity-weighted (`boundary_report`'s edge shares are plain
means, and its `action` fires at 0.5, so clones can vote to widen a bound), and a pick
made over clones is a tie-break wearing the costume of a choice. The picked row carries
`n_tied` so that says so out loud. `optimize.boundary_report` flags picked parameters
sitting on a search-space bound
(log-aware) — evidence the optimum may lie outside the searched box, so widen that bound
and re-run. Beware: more trials on the same data = more chances to overfit; judge winners
on `pct_profitable` (breadth across symbols), on the Deflated Sharpe Ratio, and on
held-out data. A per-strategy `optuna.min_trades_per_year` floor prunes trials
trading below that rate, in-objective — a handful of lucky fills can post a
spectacular `info_ratio`, and the floor keeps such artifacts off the front entirely.
It is a **rate**, not an absolute trade count: an absolute floor is implicitly far
stricter on a short backtest window than a long one for the identical trading style
— and unstable for a strategy with an open-ended `end: null` window, since the
implied rate it demands keeps changing (loosening) as more bars accumulate under
the same fixed count.

**Comparability rules.** Every strategy is measured under the same fixed conditions,
so their numbers can sit in one table (`notebooks/04_compare.ipynb`):

1. **One shared $10,000 wallet.** Each strategy trades a single cash account of
   `init_cash` *total* (the `cash_frac` wallet or a `weights` portfolio — never one
   book per symbol). The dollar amount is a global setting no strategy may override.
2. **Fractional shares.** No share rounding anywhere, so with one shared book every
   result is scale-invariant: $10k is a reporting convention, not a constraint.
3. **Same costs.** One global cost model (`config/settings.yaml`), applied identically
   to every strategy *and* to its benchmark.
4. **Profits stay reinvested; returns are geometric CAGR.** The equity curve
   compounds, and the reported rate is the constant yearly rate that compounds to the
   same endpoint: `(1 + total_return)^(1/years) − 1`. Worked example: $10k → $11k in
   year one, $11k → $12.1k in year two is reported as **10.0%/yr** — the arithmetic
   "average yearly return" of 10.5% overstates what compounding actually delivered,
   which is why it is not used anywhere in this repo.
5. **The hurdle is a strategy you can run.** `strategies/baseline/` is the
   equal-weight buy-and-hold of a universe, expressed through the normal strategy
   contract and pinned by a test to the engine's internal `backtest.benchmark()`.
   The objectives (`excess_cagr`, `info_ratio`) measure distance to it on the
   strategy's own panel; zero = "you could have just bought everything".
6. **Every strategy notebook shows the baseline comparison** (equity overlay +
   `active_stats`), and every new strategy inherits all of the above via
   `strategies/_template` and the `/new-strategy` skill.

One caveat annualization cannot fix: strategies here are tuned on different universes
and windows (`years` and `trades_per_year` ride along to expose this), so the scoreboard
compares each strategy *to its own baseline*, not raw numbers across universes.

## How to add a strategy

No engine edits — ever. Three files in one folder:

1. **Copy the template:** `cp -r strategies/_template strategies/my_idea`
2. **Write `signals()`** in `strategy.py` — a pure function from a price
   `Panel` to signal frames, built from the engine helpers
   (`sma`, `atr`, `volume_profile`, `session_mask`, `session_end`,
   `next_bar`, `exit_after`):

   ```python
   def signals(panel, *, window: int = 20) -> dict:
       breakout = panel.close > bt.sma(panel.close, window)
       entries = bt.next_bar(breakout & ~breakout.shift(1, fill_value=False))
       return {"entries": entries, "exits": bt.exit_after(entries, 10)}
   ```

   Return `weights` instead — a target fraction of the book per symbol — and the
   strategy runs in **portfolio mode** on one shared cash account. Reach for it when
   the idea is "own the best K names", not "trade each name on its own signal":

   ```python
   def signals(panel, *, top_k: int = 10) -> dict:
       rank = score(panel).rank(axis=1, ascending=False)      # your cross-sectional score
       return {"weights": bt.next_bar(rank <= top_k).astype(float) / top_k}
   ```
3. **Fill `config.yaml`** — every `signals()` argument gets **one entry** under
   `params:`, carrying what it means, what ships, and what may vary:

   ```yaml
   params:
     top_k:
       doc: How many names the book holds.
       value: 10                                   # what SHIPS — the source of truth
       search: {type: int, low: 5, high: 30}       # omit -> the study never touches it
       grid: [5, 10, 20]                           # omit -> the sweep PINS it at `value`
     cash_frac: 0.25                               # bare scalar = value only
   ```

   The config refuses to load if `value` is unreachable by its own `search`, or
   missing from its own `grid` — the two ways a sweep or walk-forward ends up
   validating a strategy nobody runs.

Then copy either strategy notebook, change the name in `strategies.load(...)`,
and Run All — the load/backtest/sweep/optimize/inspect flow is identical for
every strategy. Working with Claude Code? The `/new-strategy` skill walks
through exactly this recipe with a validation gate at the end.

## Configuration

* **Global** (`config/settings.yaml`): database, IBKR connection/pacing,
  trading costs, engine choice, per-symbol exchange overrides, hand-edited
  universes. Every key maps 1:1 onto `src/trading_simple/config.py`.
* **Per strategy** (`strategies/<name>/config.yaml`): three blocks —
  `data` (which panel to load), `optuna` (study name, sampler, trials), and
  `params`, where each parameter is declared **once** with its `doc`, its shipped
  `value`, and optionally the `search` range and `grid` list it varies over.
  There is no separate sweep or space block: a knob's shipped value and the
  ranges built around it cannot sit in two places and disagree, which is exactly
  how a walk-forward once spent four folds validating a strategy nobody ran.
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
tsim altdata   --dataset all --universe X --since 2020-01-01
                                              ingest alt-data (prefer notebooks/03 — it audits)
tsim altdata-status [--universe X] [--since D] per-(dataset,symbol,source) coverage + ledger view
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
universe, is defined).

> **`tsim subset` is not point-in-time.** It ranks by dollar volume over
> `now() - 90 days`, so a universe pinned today is *selected using data from
> inside and after* any window you then backtest on. Measured cost:
> `HighOctaneTop20`, pinned 2026-07, has an equal-weight buy-and-hold CAGR of
> **+460%/yr** over the 2025-07 → 2026-04 window five 1m strategies test on —
> which, since the objectives are benchmark-relative, is subtracted from every
> trial. For a backtest, build the universe with a ranking window that closes
> **before** the backtest opens: `notebooks/01_universe_screener.ipynb` §8 does
> exactly that (`Morning200`, hurdle +47%/yr) and needs no new data.

It never creates or migrates that database; `tsim
init-db` merely re-applies the identical idempotent schema. Do not point it at
a fresh Postgres without TimescaleDB installed.
