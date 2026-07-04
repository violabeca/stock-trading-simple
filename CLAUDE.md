# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A deliberately small framework to backtest and optimize trading strategies.
Read `README.md` first — it explains the architecture in plain language.
The prime directive: **keep it small**. Fewer lines wins; no new dependencies,
layers, or abstractions unless something genuinely cannot be expressed with the
existing ones.

## Commands

```bash
source .venv/bin/activate            # venv is python3.13
pytest -q                            # ~55 offline tests, ~5 s — the main gate
ruff check .                         # style gate (line length 100)
tsim coverage                        # checks the shared DB is reachable
tsim optimize --strategy <name> --n-trials 5   # smoke-runs a strategy end to end
```

## Hard rules

- **The strategy contract is sacred.** A strategy = `strategies/<name>/` with a
  pure `signals(panel, **params) -> dict` (+ optional `valid`, `indicators`)
  and a `config.yaml` with `data / params / sweep / optuna` blocks. Adding a
  strategy must never require touching `src/trading_simple/` — if it seems to,
  the design is wrong; stop and reconsider.
- **No look-ahead.** Signals must be execution-timed via `backtest.next_bar` /
  `backtest.exit_after`; fills happen at next-bar adjusted open, engine-enforced.
- **IBKR is never parallelized** — the historical-request cap is per account.
  One `fetch_batch` connection, paced. yfinance daily may fan out freely.
- **The database is shared** with the sibling `stock-trading` repo (TimescaleDB,
  port 5432). Never create/migrate it, never drop tables, UPSERT only.
  `prices_ohlcv` PK is `(symbol, bar, ts)`; `ts` is bar OPEN time in UTC;
  `rth=false` marks pre/post-market bars.
- **Docs are load-bearing.** Google-style docstrings on every function; update
  README.md and the strategy READMEs when behavior changes; notebooks are real
  committed `.ipynb` (no generator scripts) and must run top-to-bottom.
- **Plotly only** for figures, and only inside `src/trading_simple/plots.py`
  or notebook cells. Follow the palette/design constants already in `plots.py`.

## Adding a strategy

Use the `/new-strategy` skill (`.claude/skills/new-strategy/SKILL.md`) — it
encodes the full recipe and the validation gate. Summary: copy
`strategies/_template/`, write `signals()` from `backtest.py` helpers, mirror
every param in `config.yaml` (`params` + `optuna.space`), write a synthetic-data
test, adapt a strategy notebook, then pass:
`pytest -q && ruff check . && tsim optimize --strategy <name> --n-trials 5`.

## Gotchas

- mypy is not part of the gates here; rely on pytest + ruff.
- psycopg cannot infer NULL parameter types — optional SQL params need explicit
  `CAST(:param AS <type>)` (see `data.py` for the pattern).
- vectorbt 1.0 pins: `pandas < 3.0`, `numpy < 2.5`. The Rust engine comes from
  the `vectorbt-rust` package (`engine: auto` in settings dispatches to it).
- Optuna studies are append-only journal files under `strategies/*/studies/`
  (gitignored). Deleting the file resets the study; same name = resume.
