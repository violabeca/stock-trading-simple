# resid_momentum — pending ideas

Parking lot (the repo's per-strategy convention). Each idea names what would settle it.
Ideas that were run are in `RESEARCH_LOG.md`, not here. Numbers are lab v2 (round 2).

## Open questions

- **Combine with the short-interest book** (informed_flow_ls_p1 `v6_si`, the lab's reference
  run). Daily correlation +0.44 on T1 and +0.39 on T2, but −0.38 on T3 (2020-H2/2021) and −0.18
  on TB. A 50/50 sum of the two lab return streams reads T1 alpha +9.8% (P 0.99) at 9.8% vol,
  T2 +4.9%, T3 −0.0%, TB +2.1%. Settles it: a cross-strategy book run through the lab (one
  wallet, real costs), judged on T1/T2 and the locked TL, not a sum of returns. Note that this
  book fails G1 on its own, so a blend must earn G1 rather than inherit it.
- **Hysteresis on the volatility scale.** Rounding to the nearest 0.25 flips the scale near the
  step edges and re-trades the whole book each time: U0v2 places 244 orders a year in T3 and 256
  in T1, against 169 and 183 without the scale (C_resv2). Change the scale only when the raw
  target crosses the next step by a margin. Wins if T3/T1 orders fall >= 30% at Δ alpha >= 0. It
  is a cost fix, not an edge.
- **Short-leg crash in junk rallies** (Nov 2016, Nov 2020, Q1 2021: the short leg −5 to −7%).
  The short leg's ex-ante β·m is 46-65% of that loss (R2.6), but beta-neutral sizing (I2)
  removes it only by giving up the short leg's alpha. Untested: a value or short-interest
  overlay on the SHORT side only (value rallies are what squeezes these losers). Dies if it
  costs T1 alpha > 1 pt.
- **Survivorship.** The DB holds no pre-2026 delistings, so the short leg (losers) looks worse
  than it was, most in 2016-2019, when E0's panels hold ~390 names. Unfixable here. Read E0's
  −8.2% (short leg −5.4%) with that in mind. E0 is the largest negative piece of the G1
  failure; how much of it survivorship explains is not measured.

## Engine proposals (proposal-only: `src/` is not changed by a strategy)

- **`backtest._run_portfolio`: covers before new shorts under `lock_cash`.**
  - Problem: `call_seq="auto"` runs a bar's new shorts (sales) before the covers (buys) that
    free their collateral. On a nearly full book the new short is rejected or part-filled and,
    since `rotation_targets` emits a target only on change, never re-sent.
  - Fix: lab v2's `_call_order` (`strategies/proposals_lab/lab.py`), an explicit per-bar order
    of long cuts, short covers, new shorts, long buys.
  - Would remove this strategy's `phased` option.
  - Settles it: `tests/test_proposals_lab.py`, plus the shared engine reproducing
    `lab.run_portfolio`'s books.
- **A full book's last buy, cut by locked short gains.**
  - Problem: vectorbt's free cash under `lock_cash` leaves out a winning short's gain twice,
    while `targetpercent` sizes against the marked book. A long sent last on a nearly full book
    is part-filled or rejected, and never re-sent. This strategy keeps gross at 0.90 for it.
  - Fix: the engine re-emits a target on the next bar while the held position is short of it by
    more than a tolerance, or at least warns.
  - Settles it: `research.py audit '{"gross": 0.98}'` shows 0 shortfalls.
