# xs_three_book — pending ideas / proposals / thoughts

Parking lot for ideas about this account that are worth keeping but not (yet) implementing.
Nothing in here is a commitment; move an item out when it is implemented or rejected (and say
why). Each idea names what would settle it.

## Open questions

### 1. Replace the SI and MOM sleeves with one composite ranking (design iteration idea 3)

- **Why.** xs_si_mom_composite asks whether one ranking of the two orthogonal signals beats
  holding both books. If it passes its integration bar, the question here is whether signal
  integration and book diversification stack: the composite book at 0.60 of the account, AH at
  0.30.
- **Settles it.** A new pre-registered variant of this account, judged by the protocol's
  iteration bar against this design's final (paired Δ ≥ 0 on TRAIN, T1 and T2, and T1 Δ above
  twice the noise sd). Not run here: it waits for that design's verdict, and this design's
  variant budget (4) is spent.

### 2. Lab level: re-send a refused or part-filled target (adaptive_hedge IDEAS.md 2)

- **Why.** `lab.run_portfolio`, like the engine, sends a name an order only on the bar its
  target changes. Under `lock_cash` an order that vectorbt refuses or part-fills when free cash
  runs out is never sent again. A cover of a short that more than doubled needs free cash beyond
  its own locked proceeds.
- **Settles it.** A fill-retry fixed point in the lab's runner (like `bt._retrim_loop`), then a
  re-run of every lab book that shorts. This account's audit and `diagnose` runs count every
  refused order and stuck position, so the effect here is measured, not guessed
  (RESEARCH_LOG.md).

### 3. A noise bar for challenger rules (a protocol lesson, measured here)

- **Why.** Rule (6) crowned CRISIS over U0 on the sign of two paired deltas. On T2 that delta
  was +0.28 pt, P 0.58: below either book's T2 noise RMS (0.37 / 0.89). Its TRAIN edge was TB
  2020-H1, where AH is in-sample. quality_junk's tuned-values rule failed the same way.
- **Proposal.** A challenger should clear the protocol's iteration bar: Δ ≥ 0 on TRAIN, T1 and
  T2, and a T1 Δ above 2 × the base's noise sd. Under that bar U0 would have stayed (CRISIS's
  T1 Δ is +0.08).
- **Settles it.** The orchestrator's TL read of CRISIS against U0 and the single sleeves. If
  CRISIS − U0 on TL is within ±1 pt, the choice was a coin flip, and the simpler rule should
  have been used.

### 4. Beta drift needs more than an 8% SPY cap (I1's measured failure)

- **Why.** The account's beta drifts.
  - Before the hedge, its month-end ex-ante beta reached 0.42 (2021-07) and 0.30 (2026-09).
  - U0's realized period beta reached +0.41 (T1 2026-H1).
  - I1's hedge, capped at 8% of the account, held the month-end beta at ≤ 0.15 on TB and T2,
    but not on T3 (0.36) or T1 (0.24).
- **Idea.** Raise `hedge_max` to cover the observed drift: (0.42 − 0.15) / β_SPY ≈ 0.30. Or size
  each sleeve's new entries beta-neutral, as informed_flow_ls_p1's `sizing: beta` does.
  - The sleeves' own beta sizing cost alpha in both components (resid_momentum I2 −3.0 on T1;
    informed_flow_ls_p1 −1 pt). So the overlay hedge is the cheaper route.
- **Falsification.** A pre-registered variant with `hedge_max` 0.30 must pass bar (7) as
  written: T1 maxDD ≤ 0.8 × U0's, every month-end post-hedge beta ≤ 0.15, every period beta ≤
  0.25, and Δ ≥ −1 pt on TRAIN, T1 and T2. It dies if the larger hedge costs more than 1 pt
  anywhere: the index short pays the market's drift, which on T3 is +35% a year.
- Note: with 0.30 in SPY, the stock sleeves would share only 0.60.
