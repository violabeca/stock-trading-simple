# lead_lag — pending ideas / proposals / thoughts

Parking lot for ideas about this study that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why). Give each idea a *why* and, where possible,
the falsification that would settle it — "try X" without "X wins if / dies if" is
not an idea, it is a mood.

## Verdict (2026-09-28, `lead_lag_explore.ipynb`, all data to 2026-06)

**NO-GO at 1m, 5m and 1d** (README "Findings").

At 1 minute, a real and persistent lead-lag network exists beyond the market:
- 268 pairs, against about 1 expected by luck;
- the discovery top 100 keep their sign on validation 95% of the time.

But it is worth about **0.6 bp per trade gross against a 6.8 bp round trip**. Even a zero-commission
account pays the 2 bp spread, so no cost model within reach closes that gap.

Two other channels also fail to pay:
- **The fast market** does not lead the rest; it reverses its own move.
- **Daily lead-lag** is invisible pair by pair (detectable ρ ≈ 0.19) and absent in the group tests.

Any lead-lag strategy built from here must beat this verdict first, on bars after 2026-06-26 (1m)
or 2026-06-12 (1d). This study read everything before those dates.

## Open questions

1. **The 1m network is probably the tail of a sub-minute effect.**
   - *Why:* every top pair's best lag is 1 bar, the first minute. A lead that is spent within a
     minute is market-maker / HFT territory; 1-minute bars only see its tail.
   - *Falsification:* with trade/quote data (not in the DB), measure the same leaders' lead at
     5–30 seconds. It is only worth a strategy if the edge per trade exceeds a round trip at an
     achievable latency.
   - Not recommended: no tick data, and the latency race is not one a $10k account wins.
2. **The fast market reverses its own move within minutes** (t −12.6 at 1m, −4.3 at 5m;
   "fast market's own continuation" in §3).
   - *Why:* the equal-weight of the most-traded fifth gives back part of its last 5 minutes on the
     next bar. That is a time-series effect, not lead-lag. It is likely microstructure (the last
     print and the next open share a side) plus some overreaction.
   - *Falsification:* size it in bp on the tradable target (next open, one-bar and 5-bar holds)
     against one round trip. It dies if gross per trade < 6.8 bp. That is likely, because the
     pair network's gross was 0.6 bp.
3. **Theme clusters as a basket signal at a longer horizon.**
   - *Why:* the 1m screen is dominated by retail "theme" clusters whose leaders move a minute before
     the rest:
     - quantum: QUBT, QBTS, RGTI → AMPX, QS, AEVA;
     - space: RKLB, RDW → BKSY;
     - drones: RCAT ↔ ONDS;
     - nuclear: NNE, OKLO;
     - crypto miners: CIFR, IREN.

     At 5m the top 100 kept their sign 70% of the time against a null of 55%, though the 5m map is
     underpowered. A theme-level lead at 15–60 minutes could be larger per trade than any single
     pair.
   - *Falsification:* freeze the themes from the 1m discovery screen (no re-screening). Then test
     "the theme's leaders' last move → the theme's laggards' next 15/30/60 minutes" on bars after
     2026-06-26. It dies if the net per trade ≤ 0.
4. **Supply chains need a monthly, portfolio-level test.**
   - *Why:* C5c passed only at the margin on the daily fast map: mean percentile 0.602 against a
     0.60 bar, with the reverse direction at 0.564. It failed on the slow (lags 2..22) map. Cohen &
     Frazzini's effect is a monthly customer-momentum *portfolio*, and a daily pair map cannot see a
     correlation below ~0.19.
   - *Falsification:* each month, sort suppliers by their customers' past-month return (the
     89-edge `links.yaml`, extended from 10-K major-customer notes), long the top minus the bottom.
     Test on months after 2026-06, or replicate on 2015–2019 deep names. It dies if the spread ≤ 0.
5. **Index ETFs and mega caps at 1m were never in the window.**
   - *Why:* the fast market here is the most-traded fifth of the HighOctane pool (≤ $2B/day), not
     SPY/QQQ, and the mega caps traded only 18 sessions alongside the pool. Index → stock at 1m is
     the best-documented intraday lead in the literature.
   - *Falsification:* backfill SPY and QQQ at 1m for 2025-06-18 → 2026-06-26 (register them as
     `security_type="etf"` first; about 30 minutes of IBKR time). Re-run §3 with SPY as the fast
     market. It dies if the pooled lag t < 3 or the slow-minus-fast trade stays under 2 round trips.
     The pool's own fast market failed both (t 1.2; 0.8 bp gross), so expect little.
