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

Read `strategies/volume_momentum/strategy.py` (indicator-driven, time exits)
and `strategies/donchian_breakout/strategy.py` (breakout, trailing stops) —
one of them is almost always the right starting shape.

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
argument MUST appear under `params:` (default) and `optuna.space:` (range);
`sweep:` gets a small hand-picked grid (≤ ~30 combos after `valid`).
Comment every parameter with its meaning in one line.

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
`jobs.run_optimize("...")`, the title, the intro paragraph, and the two
`plot_heatmap` parameter names. Nothing else should need editing.

## 7 · VALIDATION GATE (all three must pass before reporting done)

```bash
uv run pytest -q && uv run ruff check .
uv run tsim optimize --strategy <name> --n-trials 5    # end-to-end smoke (needs the DB)
```

If the smoke run prunes every trial, the params/space are inconsistent with
`valid()` or the signals never fire on the configured data — fix before done.
Report the single-backtest stats from the notebook (or `backtest.evaluate`)
so the user sees the strategy actually traded.
