---
name: new-strategy
description: Implement a new trading strategy from a plain-text description, following this repo's strategy contract, with tests and a validation gate. Use when the user describes a trading idea to implement, says "add a strategy", "new strategy", or names a trading rule they want to backtest.
---

# Implement a new trading strategy

Turn the user's plain-text trading idea into a working strategy folder. The
engine never changes — everything happens inside `strategies/<name>/` plus one
test file. Work through the steps in order; do not skip the gate.

## 1 · Understand the idea, map it to the contract

From the user's description, identify:
- **Entry condition(s)** — what makes it a long? a short? If direction is
  unclear, ask.
- **Exit** — time-based (`exit_after`), stop-based (`sl_stop`/`sl_trail`/
  `tp_stop`), opposite-signal reversal, or a mix.
- **Parameters** — every number in the description becomes a tunable param
  with a sensible default and an Optuna range.
- **Bar size and session** — daily or intraday? regular hours or extended?
  Intraday strategies use `bar: 1m`, a deliberately short `data.start`
  (~2 months) while the setup is being tested, session guards
  (`skip_open` / `skip_close` params via `backtest.session_mask`) and an
  `eod_exit` param (`backtest.session_end` OR-ed into the exits) so nothing
  is held overnight — all three guarded by `panel.bar != "1d"` so the same
  `signals()` still works on daily data. Name the study `<name>-1m-v1`.
  For the symbols, prefer a pinned liquid subset over a raw big universe:
  `tsim subset --universe <U> --top 20` writes `<U>Top20` (top names by
  trailing-90d average dollar volume) — see the donchian_breakout README for
  the rationale. Sibling-repo universes arrive via `tsim import-universes`.

Read `strategies/volume_momentum/strategy.py` (indicator-driven, time exits)
and `strategies/donchian_breakout/strategy.py` (breakout, trailing stops) —
one of them is almost always the right starting shape; both are intraday and
show the session-guard pattern.

## 2 · Create the folder

```bash
cp -r strategies/_template strategies/<snake_case_name>
```

## 3 · Write `strategy.py`

- `signals(panel, **params) -> dict` — pure pandas, no I/O, no vectorbt import.
  Build from `trading_simple.backtest` helpers ONLY: `sma`, `atr`,
  `volume_profile` (time-of-day volume baseline), `next_bar` (execution
  timing — ALWAYS shift raw signals with this), `exit_after` (time exits).
  Return any subset of `entries / exits / short_entries / short_exits`
  (bool frames) and `sl_stop / sl_trail / tp_stop` (stops as fraction of
  entry price; use ATR-scaled frames for volatility-aware stops).
- `valid(params) -> bool` — only if some combos are impossible (e.g.
  `fast >= slow`); otherwise delete it.
- `indicators(close, volume, **params) -> dict` — overlay series for the
  chart; delete if nothing useful to draw.
- Module docstring = the thesis in plain language. Google-style docstrings on
  every function.

## 4 · Fill `config.yaml`

Four blocks — `data`, `params`, `sweep`, `optuna`. Every `signals()` keyword
argument MUST appear under `params:` (default); `optuna.space:` covers the
subset worth searching (structural switches like `baseline` and fixed session
guards stay params-only); `sweep:` gets a small hand-picked grid (≤ ~30
combos after `valid`). Comment every parameter with its meaning in one line.
Make each `space` range generously wide rather than pre-tuned — the
post-optimization boundary check (§6) will say if it still needs widening;
use `log: true` for window/horizon-style ranges spanning an order of
magnitude.

## 5 · Write the README and the test

- `README.md`: thesis in 3–5 plain sentences, how a trade works, a parameter
  table with intuition per knob (copy the donchian README's shape).
- `tests/test_<name>.py`: follow `tests/test_strategies.py` — build a small
  synthetic panel (see `tests/conftest.py::make_panel`) with ONE engineered
  signal event, then assert exact entry timing (next bar, never same bar),
  exit behavior, direction routing, and `valid()` if present.

## 6 · Notebook

Copy an existing strategy notebook into the new folder, rename it
`<name>.ipynb`, and change the strategy name in `strategies.load("...")` and
`jobs.run_optimize("...")`, the title, the intro paragraph, the parameter
table in the overview section, and the `plot_heatmap_grid` pair names.
Nothing else should need editing. Every strategy notebook keeps two
strategy-agnostic sections — do NOT drop them when adapting:

- **Strategy overview** (right after loading): a markdown table of each
  param's role/intuition, plus the code cell that renders defaults, sweep
  grid and Optuna ranges straight from `strat.params` / `strat.sweep` /
  `strat.optuna["space"]` — never hand-maintain those numbers.
- **Boundary check** (right after reading the study):
  `opt.boundary_report(best.params, strat.optuna["space"], front=front)`.
  If a parameter is flagged, widen that bound in `config.yaml` and re-run
  the optimize cell (same study name resumes) until the front sits inside
  the searched box.

## 7 · VALIDATION GATE (all three must pass before reporting done)

```bash
uv run pytest -q && uv run ruff check .
uv run tsim optimize --strategy <name> --n-trials 5    # end-to-end smoke (needs the DB)
```

If the smoke run prunes every trial, the params/space are inconsistent with
`valid()` or the signals never fire on the configured data — fix before done.
(`tsim optimize` ends with `boundary ...` lines; at 5 smoke trials they are
meaningless — they matter after real runs.) Report the single-backtest stats
from the notebook (or `backtest.evaluate`) so the user sees the strategy
actually traded.

## 8 · Commit

Only once the gate is green, and only if the user asked for a commit. Stage the
whole strategy folder plus its test:

```bash
git add strategies/<name> tests/test_<name>.py
git commit -m "add <name> strategy"
```

The `.githooks/pre-commit` hook clears the notebook's outputs and re-stages it,
so the committed `.ipynb` is source-only (an un-cleared one runs to tens of MB
of embedded Plotly data). Two things follow from that:

- **It strips in place.** The user's own copy loses its rendered plots on
  commit; say so, and remind them a *Run All* brings them back. Never pass
  `--no-verify` to preserve outputs.
- **The hook must be enabled** (`git config core.hooksPath .githooks`, once per
  clone). If `git config core.hooksPath` comes back empty, run it before
  committing, or the fat notebook lands in history where it is painful to
  remove.

Check `git show --stat HEAD` afterwards — the notebook should be a few dozen KB,
not a few dozen MB.
