# keltner_reversion — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

(The optuna space caps are deliberate — tuned across the keltner-1m study history —
so widening ranges is NOT on this list.)

## Exits

### Tunable exit level (`exit_mult`)

Already flagged in the README as the v2 knob: exit at `midline − exit_mult·ATR`
instead of demanding the full round-trip to the mean. Deep dips often retrace most
of the stretch and then stall; a partial-reversion take keeps the thesis (revert to
*toward* the level) while cutting the tail where the last 20% of the reversion
never arrives and the time stop eats the trade.

## Entries

### Regime-conditional stretch

`band_mult` scales with ATR, but ATR responds slowly to a vol-regime break — the
same 6-ATR stretch is an orderly pullback in a calm regime and the first leg of a
collapse in a fast one. A vol-of-vol conditioner (require deeper stretch when
realized vol is far above its own trailing median) separates the two states the
current single multiplier conflates.

### Market-wide adverse-selection guard

Dip-buying pays worst when the dip is the whole tape, not the name. A cheap guard —
skip entries on bars where the cross-sectional median return of the 20 names is
itself strongly negative — tests whether the edge is name-specific reversion or
disguised market-timing. If the guard helps materially, the honest conclusion is
the latter.

## Diagnostics

- Phantom `exit_after` exits from clustered raw signals (engine behavior shared by
  all the 1m scanners): scan trade records for time-stop exits shorter than
  `horizon` before deciding whether a re-entry cooldown is worth a param.

## Known caveats (documented, not actionable here)

- 2 bps default spread understates 1m HighOctane friction — cost-stress rerun
  mandatory. (Overnight gap-throughs now fill at the true open with slippage —
  engine-pessimistic since 2026-07-19 — so `eod_exit: false` picks pay them in full.)
- `HighOctaneTop20` is a today-snapshot universe (survivorship).

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
