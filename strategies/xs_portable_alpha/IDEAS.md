# xs_portable_alpha — pending ideas / proposals / thoughts

Parking lot for ideas about this account that are worth keeping but not (yet) implementing.
Nothing in here is a commitment; move an item out when it is implemented or rejected (and say
why). Each idea names what would settle it.

## Open questions

### 1. The overlay as xs_three_book's declared final (design iteration idea 2)

- **Why.** xs_three_book declared CRISIS (MOM + AH, 0.45 each) as its final. AH's negative beta
  (−0.28 alone) trades bull CAGR for falling-market depth. Inside this account the overlay would be
  `{"sleeves": "mom,ah"}` at the same 0.45, with SPY still at 0.45.
- **Why not run here.** The design's variant budget is 2 (U0, I1). As an information-only label it
  would add an unregistered fork beside the declared final. CRISIS's G1 is mostly AH's TB + T3
  tune (xs_three_book §5).
- **Settles it.** A new pre-registered variant with `{"sleeves": "mom,ah"}`, judged by the same
  bars (A1-A4 net of the core). It would be adopted only if it also beat U0 on every bear episode
  without losing A1 on T1 or T2. The overlay's lower beta makes A1 harder there.

### 2. An equal-weight index ETF as the core (design iteration idea 3)

- **Why.** SPY is cap-weighted. The lab's benchmark is the equal-weight buy-and-hold of the same
  names. So every raw number of this account carries SPY's composition alpha against that
  benchmark: +4.4 / +6.1 pts a year on T1 / T2, −1.8 on TB and +5.6 on T3. The net-of-core bars
  remove it only approximately (RESEARCH_LOG §0, the drag term). An equal-weight core (RSP) would
  remove it from A1 and from alpha at the source.
- **Settles it.** The lab gains an RSP column (a `lab.with_spy`-like merge from the shared DB, if
  RSP's daily bars are stored). Then the account re-runs with the RSP core.

### 3. A live account needs a core rebalancing rule

- **Why.** Every lab book restarts from cash each half-year, so the core is reset to 0.45 twice a
  year for free. A live account would hold SPY for years, and the core's weight would drift with
  SPY against the overlay: after 2023-2025 it would sit near 0.55-0.60.
  - The engine re-sends a target only when it changes, so a constant 0.45 is never re-trimmed.
  - The core's unrealized gains also never become free cash (vectorbt's `lock_cash`
    accounting), so a rising core squeezes the overlay's room for new orders.
- **Settles it.** A pre-registered rule (re-trim the core to c when it drifts more than ±0.05,
  checked at month ends) measured on one continuous multi-year book rather than half-year
  resets. That needs a lab mode that does not restart each period.
