# adaptive_hedge — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why). Give each idea a *why* and, where possible,
the falsification that would settle it — "try X" without "X wins if / dies if" is
not an idea, it is a mood.

## Open questions

### 1. Free a carried slot after N unpriced bars (round-2 verifier, minor)

- **Why.** The missing-price carry (`strategy._book`) keeps a held name through any unpriced
  bar. That is right for a one-day gap. It is wrong for a name whose data ENDS: a take-private, a
  delisting, or a gap in the DB's ingest. Such a name keeps its slot to the period end.
- **Where it acts.** On the lab v2 panels, for the final book (TUNEDv2):
  - CWAN and LBRDK, both shorts, are frozen for the last 11 bars of T2 2026-H1, after their data
    stops on 2026-06-12.
  - CWAN has no price at all in T2 2026-H2. The engine drops its opening order at a NaN price,
    so the strategy counts a short slot the book never holds: a dead slot, 3.3% of the book for
    all 76 bars (`reconcile.py`).
  - Measured effect: a typical short slot lost 0.38% of the book in that period, so the dead
    slot flatters T2 by about 0.1 point a year.
- **Rule.** A held name unpriced for N consecutive bars (N = 5) stops counting against the
  slots. The rule is causal: it reads only the bars before the decision.
- **Settles it.** A variant on TB, T3, T1 and T2, judged by the lab's iteration bar. It should
  move nothing outside T2 2026. Not run, because the final had been declared and read on T4
  before the gap was found, and the effect is about one slot in two periods.

### 2. Lab level: re-send refused targets (a proposal for the lab owner; not this folder's)

- **Why.** `lab.run_portfolio`, like the engine, sends a name an order only on the bar its
  target changes. On a book at ~98% gross under `lock_cash`, vectorbt refuses or part-fills an
  order when free cash runs out, and the order is never sent again.
- **The worst case is a cover.** vectorbt lets a cover spend the short's locked proceeds plus an
  equal margin, 2x the entry value. A short that has more than doubled therefore needs free cash
  to be bought back.
  - Example: Wayfair, TB 2020-H1. The chandelier exit was due 2020-04-27, with W at 2.8x its
    entry. The cover was refused, and W stayed short for 46 bars, costing −6.7% of the book.
- **Scale.** Over the 27 open-test periods (`reconcile.py`, RESEARCH_LOG.md §14), TUNEDv2 had:
  - 2 covers refused (W, CNK) and 1 part-filled (COIN, 31%);
  - 3 long buys refused;
  - 32 new shorts and 67 long buys part-filled.
  - Apart from the stuck covers, the missing exposure is small: 0.5-0.8% of the book long and
    0.3-0.5% short.
- **Suggested fix**, any one of:
  - Re-send a refused or part-filled target on the following bars until the position reaches
    it, for example a fill-retry fixed point like `bt._retrim_loop`.
  - Let a cover draw on its own locked proceeds.
  - Keep a free-cash reserve for covers.
  Then re-quote TB and the TRAIN era of every lab book that shorts.
- **Estimated effect here**, with the stuck P&L removed: TB +1.9 → +5.3, TRAIN era +5.7 → +7.6
  (P 0.82 → 0.88), T1 +11.4 → +11.1, T2 +4.3 → +4.1.

### 3. Never run (round 1)

- **`hedge: SPY`**: an index short in every short slot. It removes the single-name squeeze risk,
  but it also removes the short leg's selection, which is most of this book's alpha (+9.0 points
  a year on T1, RESEARCH_LOG.md §13.4 f).
  - Dies if: on T1 the index-hedged book's alpha falls below the long sleeve's own (+5.2 on T1).
- **`short_filter: momentum`**: shorts only names already down over 12-1 months. The TRAIN
  premise split (§1) said those breakdowns rebound more, so it was never run.
