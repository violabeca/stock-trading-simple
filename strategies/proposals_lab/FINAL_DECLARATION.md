# The overall cross-strategy final — declared before the locked read

*Written 2026-10-24, before `final_batch.py` ran and before any T4 or TL number of a
cross-strategy existed. Every number below is from the OPEN tests on lab v2 (TB + T3 = TRAIN for
these short-interest-based books; T1 = top 500, 2022-01 → 2026-10-16; T2 = names 501-1000), as
reviewed adversarially.*

## The five designs' declared finals, ranked on OPEN by the pre-registered rule

The rule (`xs_design.json`, judge's rationale): every applicable lab gate first, then the
design's own pre-registered bars.

| rank | design / final | gates G1-G8 | design bars | TRAIN | T1 (P) | T2 |
|---|---|---|---|---|---|---|
| **1** | **xs_si_mom_composite / A_U0** | **8 / 8** | **4 / 4** | +1.97% | **+12.88% (0.96)** | +5.07% (repaired +4.15%) |
| 2 | xs_three_book / CRISIS (MOM + AH) | 8 / 8 | bar (2) fails | +5.22% | +9.86% (0.99) | +4.80% |
| 3 | xs_cross_leg / C_U0 (SI longs, MOM shorts) | 8 / 8 | falsification (2), beta (3), (5) fail | +0.65% | +10.62% (0.97) | +3.85% |
| 4 | xs_portable_alpha / U0 (SPY 0.45 + SI/MOM overlay) | G6 fails (by design: beta 0.45) | A1 fails on T2 by 0.06 pt | net +1.21% | net +4.96% (0.99) | net +2.40% |
| 5 | xs_three_premia / B_U0 (+ profitability) | G1, G2 fail | (1), (2), (3), (5) fail | −1.17% | +6.06% (0.88) | +3.63% |

## Declared overall final: `xs_si_mom_composite / A_U0`

One dollar-neutral book (15 + 15 slots, gross 0.90) ranked on the mean of two ranks: minus
FINRA days-to-cover (informed_flow_ls's short-interest signal) and residual 12-1 momentum
(resid_momentum's month-end score), nothing tuned. It is the only design that passes every gate
and every one of its own bars.

## The confirmatory bar for the locked read (fixed now)

- **TL (names 1501-2000; fresh for this book):** alpha > 0 with block-bootstrap P ≥ 0.80, AND
  late-fill (+50 bp a leg) alpha > 0. Pass = the overall final is confirmed on names nobody
  looked at; fail = it is not, whatever T4 says.
- **T4 (names 1001-1500; fresh for this book, read twice before for its component MOM):** the
  same bar, reported as secondary evidence.
- **Integration (reported beside it, same read):** A_U0's alpha against C_mix's (the SI and MOM
  books held side by side, same code and gross) and against C_si and C_mom alone.
- **Caveat stated in advance:** TL's names trade $1-25M a day; the cost model's 2 bp spread is
  optimistic there, so the late-fill alpha is the more realistic number.

## Disclosures in the same batch (never confirmatory, never a post-hoc pick)

The other four designs' finals (xs_three_book/CRISIS, xs_cross_leg/C_U0,
xs_portable_alpha/U0 with its control xs_three_book/C_core, xs_three_premia/B_U0 with its
post-hoc control C_mix3) and the component adaptive_hedge/TUNEDv2 (TL never read; T4 read twice
before). informed_flow_ls (v5/v6) already read names 1501-2000 once in its own harness
(RESEARCH_LOG §34, §38P: +12.2%), so TL is fresh only for the combinations.

---

## Results of the locked read (2026-10-24, after the declaration above)

`final_batch.py` ran once, after this file existed, on the labels `<label>-LOCK`
(`var/proposals_lab/results2/final_batch.csv`, log beside it). Nothing above was changed after it.

### The overall final: CONFIRMED

| xs_si_mom_composite / A_U0 | alpha (P) | late-fill alpha | beta | CAGR vs buy-and-hold | max DD | 2022 bear (market) | Feb-Apr 2025 (market) |
|---|---|---|---|---|---|---|---|
| **TL** (names 1501-2000) | **+8.56% (0.963)** | **+6.50%** | −0.10 | 6.5% vs 19.4% | −13.3% | +6.2% (−16.2%) | +3.4% (−20.4%) |
| **T4** (names 1001-1500) | **+12.35% (0.997)** | **+10.14%** | −0.09 | 11.5% vs 10.2% | −11.7% | +4.5% (−18.4%) | +6.7% (−22.4%) |

- **TL meets the confirmatory bar** (alpha > 0, P ≥ 0.80, late-fill alpha > 0), and so does T4.
- **Execution reconciled after the read, before this note:** every locked period of A_U0 and C_mix
  was rebuilt with vectorbt's order log (`xs_si_mom_composite/research.py`'s `rebuild` / `_gaps`
  / `off_target`, run from a scratch script because `research.py` refuses locked tests) and equals
  the saved returns to < 1e-12. In all 20 periods: **0 refused or part-filled orders and 0
  off-target name-days** (out of 36,060 target name-days per test for A_U0), so no stuck cover
  inflates either number, unlike GME's on T2.
- **Half-year alphas:**
  - T4: −5.8, +21.1, +31.0, +12.0, +20.6, +18.1, +20.6, +8.7, +5.9, −7.9.
  - TL: +4.1, +38.0, +1.0, +6.5, +31.2, +4.4, +17.0, +10.7, −12.8, +9.1.
  - Every half-year from 2022-H2 to 2025-H2 is positive on both sets. 2026 is mixed: T4
    +5.9 then −7.9, TL −12.8 then +9.1. On T1, both 2026 halves lost about 22%/yr.
- **Without the five biggest contributors:** T4 +8.36%, TL +4.26%.
- **Luck band, run after the read (supplementary; it decides nothing):** A_U0 beat all 100 books that
  trade its own signals on names shuffled within dollar-volume tenths (`lab.py luck`). On T4 the
  books' median was −0.05% and their 95th percentile +5.26%; on TL, −1.06% and +4.98%.
- **Not run on locked names:** 2× costs.

### Checks after the read (2026-10-24): supplementary, they decide nothing

- **All 11 locked books reconciled.** Every one of their 220 locked periods was rebuilt from vectorbt's
  order log and equals its saved returns.
  - No refused or part-filled order: A_U0, C_mix, C_si, C_mom, xs_cross_leg C_U0, xs_portable_alpha U0,
    xs_three_book C_core, xs_three_premia C_mix3. CRISIS had none either, only one order dropped at a
    missing price.
  - xs_three_premia B_U0: one cover refused on TL (WVE, 2025-12-09, stuck 16 bars). With its stuck P&L
    removed TL is +3.12% (P 0.73), not +3.37%.
  - adaptive_hedge TUNEDv2: see its row below.
- **The controls pass the same bar.** On both locked sets every control clears alpha > 0, P ≥ 0.80 and
  late-fill alpha > 0:

  | control | T4 alpha (P), late | TL alpha (P), late |
  |---|---|---|
  | C_si | +3.84% (0.83), +2.06% | +10.91% (0.99), +8.81% |
  | C_mix | +4.36% (0.91), +2.31% | +8.71% (0.99), +6.42% |
  | C_mom | +4.64% (0.82), +2.40% | +6.29% (0.89), +3.65% |

  What TL confirms is the short-interest + momentum family on fresh names. The composite's own edge over
  its parts is confirmed on T4 only.
- **2× borrow** (every tier doubled; labels `A_U0-B2`, `A_U0-B2-LOCK`):

  | test | alpha (P) | with late fills |
  |---|---|---|
  | T4 | +11.54% (0.996) | +9.34% |
  | TL | +8.01% (0.951) | +5.92% |
  | TRAIN (open) | +1.41% | — |
  | T1 (open) | +12.33% | — |
  | T2 (open) | +4.42% | — |
- **Whole shares at a $33,000 account:** T1 +11.89%, T2 +4.03%, T4 +11.98%, TL +8.51%, against +12.88%,
  +5.07%, +12.35% and +8.56% in fractional shares.
- **IBKR's borrow file of 2026-10-21:** every name the book shorted in each test's last half-year is
  borrowable, with more shares available than needed.
  - The median fee is 0.30-0.42% a year.
  - Exceptions are business development companies (TSLX 9.2%, ARCC 4.5%, MAIN 4.4%).
- **Beta.**
  - Stitched over each locked set it is −0.09 / −0.10.
  - By half-year it ranged −0.31 to +0.25 (T4) and −0.34 to +0.04 (TL).
  - On the top 500 (T1, open) the book has run net long since 2025-H2: realized +0.60, +0.73, +0.40, and
    ex-ante +0.65 in both 2026 halves.

### Integration: replicated on T4, not on TL

Paired Δ alpha, same days and same bootstrap blocks:

| A_U0 minus | T4 Δ (P, periods won) | TL Δ (P, periods won) |
|---|---|---|
| C_mix (SI and MOM books side by side) | +7.99 (0.997, 7/10) | −0.14 (0.49, 6/10) |
| C_si (short interest alone) | +8.51 (0.97, 7/10) | −2.34 (0.33, 4/10) |
| C_mom (residual momentum alone) | +7.71 (0.96, 7/10) | +2.27 (0.67, 7/10) |

On the smallest names, the plain short-interest book (C_si +10.91%) did as well as the composite or
better. **The book is confirmed. "Signal integration beats holding both books" holds on one of the
two fresh name sets.**

### The disclosed books: never confirmatory, never a post-hoc pick

| book | T4 alpha (P) | TL alpha (P) | note |
|---|---|---|---|
| xs_three_book / CRISIS (design final) | +4.96% (0.94) | **−0.03% (0.54)** | fails TL; TL max DD −30%. Reconciled: 0 refused or part-filled orders, so the failure is real |
| xs_cross_leg / C_U0 (design final) | +3.18% (0.77) | +4.78% (0.89) | |
| xs_portable_alpha / U0 (design final), raw | +4.89% (0.98) | +3.43% (0.89) | |
| ... net of 0.45 × SPY's alpha (its own bar) | +2.78% (0.96) | +4.21% (0.99) | late fills, net: +1.37% / +2.61% |
| ... its design bar A1 (CAGR > EW buy-and-hold) | fails: 7.7% vs 10.2% | fails: 8.6% vs 19.4% | max DD 0.37 / 0.45 of buy-and-hold's |
| xs_three_book / C_core (overlay alone, control) | +5.14% (0.96) | +7.88% (0.99) | |
| xs_three_premia / B_U0 (design final) | +3.38% (0.78) | +3.37% (0.75) | |
| xs_three_premia / C_mix3 (post-hoc control) | +4.02% (0.95) | +3.35% (0.84) | |
| adaptive_hedge / TUNEDv2 (component) | +4.16% (0.84) | **−4.82% (0.205)** | TL max DD −57%, CAGR −11.1%. Reconciled: two covers refused at zero free cash (BKKT 2024-11-19, stuck 29 bars, +3.41% of the book; SGML 2025-12-22, 7 bars, −0.35%). With their stuck P&L removed, TL is −5.49% (P 0.18): the failure is real, and the stuck cover flattered it |
