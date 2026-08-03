# squeeze_breakout — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

(Context: squeeze-1m-v1 was an honest falsification — squeeze and volume gates
validated mechanically, the trend gate did not earn its keep, and the net edge did
not survive cost-stress or the untouched holdout. Everything below is about the
documented v2, not about re-rolling the same dice.)

## v2 — drop the trend gate

The README's own conclusion from ~2,000 trials: disabling the trend gate barely
moved the front, while the other two gates flipped the sign when removed. v2 is the
same strategy minus `trend_window`, re-searched — two fewer dimensions (the gate
and its window), a denser sample of the space that mattered.

## Entries

### Minimum-expected-move filter

The failure mode was death-by-costs: the breaks were real but too small to pay the
round trip. Make that the entry condition — only take breaks where the compression
depth (squeeze ratio) historically predicts an expansion ≥ round-trip cost in
ATRs for that name. This converts the cost-stress from a post-hoc audit into an
ex-ante filter, which is the only version of it that can save the strategy.

### Per-name eligibility screen

The study's gains concentrated in the fastest names (AAOI, IREN, RKLB, HOOD) and
bled in the slower ones (GS, COHR). A trailing per-name screen — only trade names
whose own breakout follow-through has cleared costs over the last N sessions —
is evidence-backed and cheap. It is also a survivorship trap if done naively:
compute it strictly trailing, never from the full-window result table.

## Known caveats (documented, not actionable here)

- 2 bps default spread understates 1m HighOctane friction — and this strategy is
  the repo's proof of how much that matters.
- `HighOctaneTop20` is a today-snapshot universe (survivorship).
- Phantom `exit_after` exits on clustered signals — already documented in the
  strategy docstring.

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
