# volume_momentum — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Exits

### ATR-scaled stop instead of the fixed-percent `sl_pct`

`sl_pct` is a flat fraction of entry price (default 2%) — the only non-ATR stop in
the repo, and on a 20-name universe spanning very different vol regimes the same 2%
is a noise-floor stop on IREN and a disaster stop on GS. An `sl_mult`-in-ATRs stop
(the siblings' idiom) makes the risk per trade mean the same thing across names.
Keep the search range economically bounded (≥1 ATR; sub-ATR stops just harvest the
spread).

## Signal construction

### The direction filter is the weak leg — test it, then replace it

The volume spike (vs the time-of-day profile baseline — solid construction) decides
*when*; a fast−slow SMA difference decides *which way*. On 1m bars that difference
is a lagging, whipsaw-prone sign that is plausibly near-random in chop, meaning the
volume gate may be doing all the work. Two-step plan:
1. Falsify: compare the shipped strategy against a coin-flip direction on the same
   spike bars (and against long-only). If the SMA sign adds nothing, say so.
2. Replace: candidate signs with actual information at spike time — close location
   within the spike bar's range (a proxy for who initiated the volume), or a
   breakout of the spike bar's own high/low as the direction trigger.

### Continuation-decay check to evidence-set `horizon`

Does post-spike drift actually persist for `horizon` bars, or is it a 1–2 bar pop?
A signal-level decay curve (mean forward return by bars-since-spike, per direction)
would set the `horizon` search range from evidence instead of leaving the optimizer
to find where the noise happens to sum positive.

## Search space

- `fast`/`slow` are drawn independently (fast to 60, slow from 30) so `fast ≥ slow`
  combos are generated and burned by `valid()` — reparameterize as
  `slow = fast + gap` to make every draw admissible.

## Known caveats (documented, not actionable here)

- 2 bps default spread understates 1m HighOctane friction — cost-stress mandatory.
- `HighOctaneTop20` is a today-snapshot universe (survivorship).
- Phantom `exit_after` exits on clustered spike signals (engine behavior; see the
  squeeze_breakout docstring).

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
