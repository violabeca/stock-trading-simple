# peer_reversal — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet) implementing.
Nothing in here is a commitment; move an item out when it is implemented or rejected (and say
why). The verdict is **no edge** (RESEARCH_LOG §12, confirmed on lab v2 in R2.8), and 9 of the 10
variants are used, so each item below would be a new experiment with its own pre-registration.

## Open questions

- **The volume cap was never run.**
  - The proposal named this variant: skip moves made on more than 3× normal volume, which are
    information, not liquidity.
  - It was skipped for the variant budget, not on evidence: the TRAIN formation-volume terciles
    have t −0.74 / +1.47 / −0.10 (§13.3).
  - *Settles:* `volume_cap: 3` against U0, by the lab bar.
  - *Dies if:* on TRAIN, high-volume moves do not continue more than low-volume ones (gross per
    trade, Newey-West t over entry dates).
- **The clock restart lets a persistent winner stay shorted for months.**
  - A held name that stays in its entry zone restarts its `max_hold` clock. In I4v2's 2026-H1
    that kept MRVL short for 65 bars and DELL for 79 (R2.2), the trades that cost the period
    most.
  - A cap on the total hold (for example 2 × `max_hold`) would bound it.
  - *Settles:* the TRAIN anatomy of trades held beyond 2 × `max_hold`. If their gross per trade
    is below the rest's at t < −2, a capped run is worth pre-registering. Do not design it from
    T1's 2026-H1 alone: that would fit one half-year.
- **Crash-time correlation with other books.**
  - The one property that survives is a near-zero-beta book that beat the market in all four
    bear episodes (R2.4).
  - It is worth reusing as a sleeve only if its daily returns are negatively correlated with
    the other proposals' books in those episodes, and the combined book's alpha rises on both
    T1 and T2.
  - Not measured.
- **The real order size.**
  - At a $30k account a 40-slot book's orders are $367-735, not the lab's $1,000. That is 14-27
    bp of commission a leg against the 12 bp the lab priced, so U0's ~6 pts/yr of costs would
    be 1.5-2× larger.
  - Any re-test prices `order_notional` at the real account.
- **Sensitivity to the universe.**
  - Swapping 6 of 500 names (the sector ETFs, from 2024-H2) moved I4's T1 alpha by about
    −5 pts (R2.2). The swap re-forms the peer groups.
  - Read a cluster book's T1 number with its name-noise copies. The lab's noise sd also sits
    below the copies' RMS distance from the label (§13.2).

## Engine (proposal-only, `src/`)

- **The call order under `lock_cash`.**
  - The repo engine runs a bar's orders with `call_seq="auto"`, which puts new shorts before
    the covers that free their cash. A 0.98-gross long/short book under-filled 2% of its
    openings this way (§13.6).
  - Lab v2's `run_portfolio` fixes it for the lab only, in this order: long cuts, short covers,
    new shorts, long buys. It under-fills none (R2.3).
  - A headless `tsim optimize` of this strategy still uses the engine's order.

## Rejected (measured; see the log)

| idea | why rejected |
|---|---|
| correlation-cluster peers | sector codes beat them on TRAIN, T1 and T2 (C1, R2.1); no peers had the higher TRAIN IC (§1) |
| the binary stress switch | C2 − U0 is −0.14 to +0.07 pts (R2.3) |
| stress-only entries | I2: T1 Δ +2.05, below the 2.60 bar; no significant stress-day advantage anywhere (R2.3) |
| monthly formation | I1: T1 Δ −1.48 |
| volatility scaling | I3: T1 Δ −2.22 |
| the tuned corner | T: worse than U0 on T1 and T2; PBO 0.71 |
| monthly formation + stress-only | I4: T1 Δ +0.36 on v2; its gain is the cost saving |
