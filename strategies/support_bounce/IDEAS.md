# support_bounce — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Signal construction

### Graded level confluence instead of binary `min_touches`

The Osler / Kavajecz–Odders-White rationale is that levels with more resting
interest hold better — that is a *graded* property the current binary
`min_touches` flattens. A confluence score per level (number of distinct tests ×
a volume-cluster proxy at the level, e.g. summed volume within `touch_tol` of it)
would rank candidate entries by how defended the level actually is, and the
falsification stays built in: if score-weighted entries do no better than
`min_touches: 0`, the "tested level" thesis added nothing — which is exactly what
the README already promises to report honestly.

### Volume confirmation for `bounce` mode

`bounce` mode waits for the reclaim; a reclaim on above-average volume (vs the
time-of-day profile baseline, `bt.volume_profile`) distinguishes a defended level
from a dead-cat drift through it. One `vol_mult`-style gate, ablatable to 0,
mirroring the siblings' idiom.

## Cheap experiments the code already supports

### Coarser-bar variant (5m / 1h)

`_bars` converts every minute-window param per `panel.bar`, so the same code runs
on coarser bars today — only the config never exercises it. A 5m variant sidesteps
most 1m churn and cost sensitivity, at the price of coarser entries. NOTE: no 5m
backfills — derive 5m from the stored 1m bars (agreed convention), so this is a
data.load question, not an ingestion project.

## Known caveats (documented, not actionable here)

- ~~Overnight gap-through-stop fills at the stop (optimistic)~~ — FIXED at the
  engine level 2026-07-19: gap-throughs now fill at the true open with slippage
  charged. The notebook's gap scan stays useful for sizing how much that costs.
- 2 bps default spread understates 1m HighOctane friction — cost-stress mandatory.
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
