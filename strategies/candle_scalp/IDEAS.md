# candle_scalp — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Search space

### `wick_mult` is a dead dimension outside `pattern: hammer`

The optuna space searches `wick_mult` 1.0–5.0 unconditionally, but the param only
affects the hammer pattern — every engulfing/`any` trial burns a sample on a knob
that does nothing there. Either gate it in `valid()` (reject non-default
`wick_mult` when `pattern != hammer`) or accept the waste knowingly; with a
1000-trial budget it is real but not fatal. Rejecting is the cheaper edit.

## Signal construction

### Vol-normalized stretch instead of raw ATR multiples

`stretch_mult` measures the dip in ATRs below the band, but the 20 names span very
different vol regimes, so one global multiple is "deep" for GS and "routine" for
IREN. A z-scored stretch (close vs band, normalized by that name's own recent
band-distance distribution) makes the threshold mean the same thing across names —
directly serving the thesis that the edge lives at genuine per-name extremes.

### Same-name re-entry cooldown

`exit_after` schedules a time exit for every raw entry signal, including signals
ignored while a position was already open — clustered reversal candles can emit
phantom exits that clip a later trade early (engine behavior, documented in
squeeze_breakout's docstring). A `cooldown` param (no new raw entry within N bars
of the last) blunts the clustering at the source. Check trade records for
shorter-than-`horizon` time-stop exits first to size how much this actually bites.

## Known caveats (documented, not actionable here)

- The 2 bps default spread understates HighOctane friction 2–5× — the README's
  env-override cost-stress rerun is mandatory before believing any pick.
- `HighOctaneTop20` is a today-snapshot universe (survivorship).
- 1m intrabar path ambiguity: judge on the time-stop/EOD exit population, not
  tight-bracket win rates (README documents both).

## Universe contamination (measured 2026-07-28, not yet acted on here)

`HighOctaneTop20` was pinned on 2026-07-05 by `tsim subset`, which ranks on
`now() - 90 days` — April–July 2026 data — while this strategy backtests
2025-07-01 → 2026-04-30. The universe was therefore **selected using data from
inside and after the window it is tested on**, and it is not a small effect:
equal-weight buy-and-hold of those 20 names returned **+460%/yr** over the train
window (SNDK +2255%, BE +1123%; neither had pre-window daily history — a 2025
spinoff and a 2025 IPO). Because `optimize.OBJECTIVES` is benchmark-relative,
that inflated hurdle is subtracted from every trial: in `gap_and_go`'s v1 study
all 45 scoring trials landed at `excess_cagr ≈ -4.6` and the Pareto front spanned
0.05 — noise, not a trade-off curve.

The fix needs no backfill. `Morning200` (built in
`notebooks/01_universe_screener.ipynb` §8) keeps the 504 symbols with >= 200
sessions of stored 1m data in the window and ranks them by median daily dollar
volume over a period that **closes before the backtest opens**; its buy-and-hold
hurdle is a real +47%/yr. `gap_and_go` has migrated. This strategy has not — its
study and its README numbers should be read as unresolved until it does, and a
migration invalidates the existing journal (new benchmark, new panel).
