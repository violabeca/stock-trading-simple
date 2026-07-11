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
uv run tsim altdata --dataset all --universe LiquidLeaders --since 2020-01-01  # ingest alt-data
git config core.hooksPath .githooks   # once per clone — enables the notebook-clearing pre-commit hook
git ls-files '*.ipynb' | xargs python3 scripts/nbclean.py   # clear notebook outputs by hand
```

## Hard rules

- **The strategy contract is sacred.** A strategy = `strategies/<name>/` with a
  pure `signals(panel, **params) -> dict` (+ optional `valid`, `indicators`)
  and a `config.yaml` with `data / params / sweep / optuna` blocks. Adding a
  strategy must never require touching `src/trading_simple/` — if it seems to,
  the design is wrong; stop and reconsider.
- **Alternative data rides on `Panel.features`.** Non-OHLCV signals (insider,
  ratings, short interest) live in the shared DB and reach a strategy as
  `panel.features[name]` — one wide *date × symbol* daily frame per signal,
  built by `features.py`, already point-in-time lagged. A strategy opts in with a
  `data.features:` list in its `config.yaml`; feature builders do only
  DB→daily-aggregate, all modelling stays in `signals()`. Adding an alt-data
  strategy needs no engine change; adding a new *feature type* is one builder in
  `features.py` (proposal-only, like any `src/` change). See `informed_flow`.
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
- SEC EDGAR (insider provider) requires a User-Agent with a contact email or it
  returns 403; a working default is set, override with `TRADING_SEC_USER_AGENT`.
  Form 4 `primaryDocument` points at the XSL-styled HTML — fetch the raw XML at
  the filing root (`providers/sec._raw_doc`). Ingestion is paced/single-process
  (SEC & FINRA rate-limit per IP), like the IBKR rule.
- Alt-data features align to the panel by UTC session date and are `shift(1)`
  lagged in `features.py` (conservative point-in-time); this supports daily bars.
