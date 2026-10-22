# Ideas parked for level_flow

Not implemented. Each entry says why it might matter and what would falsify it, so it can be
picked up (or discarded) without re-deriving the reasoning.

1. **Merge adjacent histogram bins.** A level straddling a bin edge is split into two weak bins.
   *Why:* the bin grid is absolute in log price, so the split depends on where a round number
   falls. *Test:* count pivots over `(b-1, b, b+1)` for the threshold and compare the event-study
   t-stats and the touch-count distribution against the unsmoothed version.
2. **Re-scan the level pair intraday.** The watchlist is frozen at `scan_minute` and a bounce is
   read against the nearest level to the *live* close, but the activity ranking is not refreshed.
   *Why:* a name that becomes active at 11:00 never gets a seat. *Test:* an hourly re-rank with
   the same seat count; it must beat the frozen scan out of sample, not just trade more.
3. **Pre-market levels.** The database holds extended-hours 1m bars for the whole universe; the
   pre-market high/low is a classic day-trader level. *Why:* it is the most recent tested price
   before the open. *Test:* add `session: both` levels-only (execution stays RTH) and split §5b by
   whether the traded level came from the pre-market.
4. **Round-number levels.** Osler 2003's mechanism is order clustering at round numbers (FX).
   *Test:* nearest `$1 / $5 / $10` multiple by price band as an extra level with its own polarity
   split in §5b; it must show a bounce/break asymmetry that the pivot levels do not.
5. **Meta-labeling gate (v2).** A small classifier on triple-barrier outcomes of the *raw*
   triggers (features: touches, rvol, distance in ATR, time of day, trend state) predicting P(win),
   used to skip or size trades. *Why:* the supervised, sample-efficient answer to "which setups are
   promising" that the RL question was really about. *Test:* trained inside each walk-forward fold
   only; must lift the stitched OOS alpha over the un-gated refit.
6. **Earnings-calendar exclusion.** A level break on an earnings gap is a different population
   from a sympathy break. *Test:* needs an earnings-date feature; split §5b by it.
7. **Limit-fill stand-in for the bounce.** The engine fills at the next open; a real bounce
   trade rests a limit at the level. *Test:* the `touch` entry mode from `support_bounce` as a
   bound; the adverse-selection cost of the limit order is the honest counter-argument.
8. **Longer level windows.** `level_days` is capped at 20 because the stored 1m history starts
   2025-06-18 and the warm-up must cover it. *Test:* backfill 1m bars before 2025-06-18 (IBKR, one
   paced connection, ~200 names) and widen the search to 60 with `warmup_sessions` raised to match.
9. **Anchored vs rolling, and cadence.** The diagnostics notebook re-tunes on rolling 4-month
   windows. *Test:* the anchored variant and a 3-month cadence on the same folds; the answer is
   which procedure's stitched OOS path is better, not which train window scores higher.
10. **Per-day loss limit.** "Stop trading after losing X% today" needs the realized P&L inside
    the run — engine state a pure `signals()` cannot see. Proposal-only; the per-trade
    `risk_per_trade` cap and the slot count bound the daily loss today.
11. **Per-direction stops.** One `sl_stop` frame serves longs and shorts; the asymmetry is
    encoded per entry bar, which is enough for one position per side per session. A strategy
    holding both sides in one name would need two frames (engine change).
12. **Leads from the first event study (README "First results").** (a) `polarity: break` is the
    only trigger with drift (+6–9 bps over 1–2 h, t ≈ 2, doubled by activity selection) but trades
    ~5× less often than the bounce; an exploratory study on `break` with `min_trades_per_year`
    lowered to ~60 would test whether the drift survives costs — say so in the study name.
    (b) The bounce carries information only *with* the 15-minute trend (t 2.1–2.8 at 120–180
    bars); `trend_gate: with` as the shipped default is the cleanest way to give the bounce a
    chance, and `trend_gate` should then be structural in the walk-forward. (c) Levels with 5+
    prior touches break rather than hold: a `max_touches` cap on the bounce, or using the touch
    count as a break-side ranking, is the falsification test. All three are in-sample leads from
    one 10-month window; none is a result.
13. **A stop cap in ATR units, not price units.** `max_stop` (fraction of price) silences the most
    volatile names outright (visual notebook: 0 of 63 AAOI triggers survive 2%), while `sl_atr`
    already scales with the name's range. *Test:* replace or complement `max_stop` with a cap on the
    session ATR as a fraction of price (`max_atr_pct`), and compare which names the two versions
    admit; the event study by ADR bucket says whether volatile names carry the edge or the losses.
14. **Do sloped levels exist, and in which names? (before any trading).** `trendlines.py` draws
    the lines a chart reader would; the notebook's smoke test says they do not yet beat their
    parallel twins. *Test, pre-registered, on 150 tuning names + a disjoint 150 held back:* per
    (symbol, 20-session block), the uplift in held-given-tested of real lines over BOTH placebos
    (parallel shift, and slope scramble around the anchor); an Optuna study over the
    `find_lines` knobs maximizing that uplift's t-stat subject to a coverage cap; then the rank IC
    of a name's uplift in block *b* against block *b+1*. A method whose uplift is ~0, or whose
    levelness does not persist from one block to the next, cannot be screened on and is not wired
    into `signals()`.
