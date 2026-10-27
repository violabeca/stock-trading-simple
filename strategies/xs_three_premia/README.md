# xs_three_premia — short interest × residual momentum × within-sector profitability

A cross-strategy of the proposals lab's last round (`strategies/proposals_lab`, lab v2): one
dollar-neutral book ranked on THREE signal families at once. The question it answers: does a
third premium, within-sector profitability (PROF, the one lab signal with a TRAIN premise),
make the short-interest × residual-momentum composite of `xs_si_mom_composite` regime-robust —
does one ranking of the three beat holding the three books side by side (`C_mix3`) and beat
the two-signal composite (`xs_si_mom_composite/A_U0`), on alpha and in the 2026-H2 crash and
the drawdowns where profitability was supposed to help?

This is a research folder. `RESEARCH_LOG.md` holds the pre-registration (§0, SHA-256-snapshotted
in `var/proposals_lab/results2/xs_three_premia/prereg/`) and every number.

> **Status (2026-10-24, lab v2, OPEN tests only): no edge as designed — the three-premia
> composite fails G1 and G2 (6 of 8 gates) and design bars (1), (2), (3) and (5).** Declared
> final: `B_U0` (config as shipped, nothing tuned), reported as failing. T4 and TL are not read
> here; the orchestrator reads TL once.
>
> | | TRAIN (TB+T3) | T1 (top 500) | T2 (501-1000) |
> |---|---|---|---|
> | **B_U0** alpha (P) | −1.17% (0.41) | +6.06% (0.883) | +3.63% (0.79) |
> | A_U0, two premia (xs_si_mom_composite, read-only) | +1.97% | +12.88% | +5.07% (+4.15% repaired) |
> | C_mix3, the three books side by side | +0.18% | +7.59% (0.995) | +2.24% |
> | C_prof, profitability alone (this machinery) | −0.78% | +2.56% | +1.45% |
>
> - **The hedge worked; the trade did not.** In 2026-H2, the crash that hit SI, MOM and A_U0
>   (−22.2% / −7.9% a year on T1 / T2), B_U0 made +3.7% / +3.9%, and its T1 drawdown is smaller
>   (−18.9% vs −21.9%). But against A_U0 it gives up 6.8 pts of T1 alpha, 3.1 on TRAIN and 1.4 on
>   T2: a third equal rank pulls the book's 15 + 15 names away from the extremes that carry SI and
>   MOM (only 1.6-1.9 of SI's 15 names a side remain on T1/T2), toward the middle of the
>   cross-section where PROF's IC lives (its TRAIN rank IC +0.030, its top-minus-bottom 15 ≈ 0).
> - B_U0: luck 98th percentile, late fills +4.14%, every bear episode beaten, down months +1.25%
>   (above A_U0's +0.95%), beta −0.21 (PROF tilts equal slots net short), top-5-out +1.74%.
> - Integration loses to holding the three books side by side (C_mix3): TRAIN −1.34, T1 −1.53
>   (T2 +1.39). Read after the declaration (information only): C_mix3 passes all eight gates
>   (G1 a hair-thin +0.18%, T1 P 0.995, luck 100th, max DD −7.6%). It is a control, not this
>   design's final: the pre-registered switch to it applied only if B_U0 had passed bar (1).
> - Iterations, both rejected by the pre-registered rule: PROF at half weight (I1: T1 +4.0, T2
>   +2.2, TRAIN −1.9 vs B_U0) and profitability growth on the long side (I2: TRAIN −3.6).
> - Mechanics: no order refused or part-filled on any open test, nothing off target, and the
>   engine tracks an independent frictionless preview within its costs. Two holes in the price
>   data were carried frozen (PSKY 2026-09 on T2, +0.44% of the book once; WBD's end of data on
>   T1, controls only).
> - In-sample caveats: T1 is partly in-sample for SI; PROF's validation record and its 2026-H2
>   evidence were read before this design; TL is fresh only for the combination.

> **Locked read (2026-10-24, run once by the orchestrator, disclosure only; `strategies/
> proposals_lab/FINAL_DECLARATION.md`):**
>
> - B_U0 read T4 +3.38% (P 0.78) and TL +3.37% (P 0.75), with a max drawdown of −25.5% on both.
>   Rebuilt from vectorbt's order log afterwards: one cover was refused on TL (WVE, 2025-12-09,
>   stuck 16 bars, +1.17% of the book). With its stuck P&L removed TL is +3.12% (P 0.73).
> - The post-hoc control C_mix3 read T4 +4.02% (P 0.95) and TL +3.35% (P 0.84). It is a
>   control, not a pick.
> - The overall final, A_U0, read +12.35% and +8.56% on the same names.

## The rules (`B_U0`, config.yaml as shipped; nothing tuned)

1. **Eligible:** informed_flow_ls_p1's — close >= $5, a days-to-cover and a share count.
2. **SI rank:** minus days-to-cover averaged over 42 bars, percentile inside 5 dollar-volume
   groups, silent below 50% coverage (`informed_flow_ls_p1.compose(...)["ranks"]["si"]`).
3. **MOM rank:** resid_momentum U0v2's `Σe / sd(e)` over bars t−251 .. t−21 (`e = r − β·m`
   against the equal-weight market of names >= $5, 252-bar OLS beta), ranked on month-end rows
   and carried to the next month end; masked where p1 calls the name ineligible.
4. **PROF rank:** quality_junk's `compose(panel, min_price=5)["ranks"]["prof"]` — gross profit
   TTM ÷ total assets, net income TTM ÷ equity (> 0) where gross profit is untagged, ranked
   within sector × measure (a group under 5 names ranks against every name on its measure).
   quality_junk's safety sleeve is not used.
5. **Composite:** `(si.fillna(0.5) + mom.fillna(0.5) + prof.fillna(0.5)) / 3`, averaged over 5
   bars among the eligible names, re-ranked.
6. **Book (A_U0's):** informed_flow_ls_p1's `_select` — a long enters at rank >= 0.90 and leaves
   below 0.80, a short mirrors it; exits first, then free slots best first; `max_hold` 126 (the
   clock restarts in the entry zone); a short is bought back at +100% from its entry close and
   re-enters only after its rank left the short zone once; never short (or keep short) a name
   above 30% short interest / shares; a NaN score exits.
7. **Sizing:** 15 + 15 equal slots of 3.0% (gross 0.90, dollar-neutral), constant while held;
   decided at the close, filled at the next open; a target change on a bar without an open
   waits for the next priced open (adaptive_hedge's `_on_priced_opens`).
8. **Costs:** borrow 0.5% / 3% / 10% a year by short interest / shares (p1's tiers), `fill:
   open`, `order_notional: 1000` (a ~$33k account), longs trimmed at 2× their slot.

Knobs used by the controls and iterations: `long_sleeves` / `short_sleeves` (`si`, `mom`,
`prof`), `w_si` / `w_mom` / `w_prof`, `combine: book` (the single-sleeve books side by side,
netted: `C_mix3`), and this folder's own `w_growth` (quality_junk's profitability-growth rank
added to the LONG composite only: iteration I2; 0 = off = the core's call exactly). The core's
`short_gate` and `neutral` knobs pass through, unused here.

## Code (loaded by path, pinned by SHA-256, never edited)

`strategy.py` is a thin wrapper: it loads `xs_si_mom_composite/strategy.py` by path as
`_xsp_core` (pinned, `CORE_SHA256`), whose own pins load `informed_flow_ls_p1`, `resid_momentum`,
`quality_junk` and `adaptive_hedge` (`_xsc_*`). A changed file refuses to load. The shared
`informed_flow_ls` (under review) is never edited, imported or loaded.

## Files

- `strategy.py`: `signals()` / `valid()` (the core's, plus `w_growth`).
- `research.py`: the mechanics audit (`audit`, `rebuild`, `off_target`, the stuck-order P&L from
  the due open `_gaps` / `gap_pnl`, the frictionless research `preview`), the stuck-order
  estimate with repaired rows (`stuck`), the TRAIN premise check, paired deltas, the
  pre-registered bars (`bars`, `--repaired`), the iteration rule (`adopt`), positions-only
  overlap and sector exposure (`overlap`), noise copies (`noise_sd`, `noise_delta`). Loaded by
  path; `strategy.py` never imports it; it refuses the locked tests T4 and TL.
- `RESEARCH_LOG.md`: pre-registration, every run, the verdict and the declared final.
- `IDEAS.md`: parked ideas.
- `tests/test_xs_three_premia.py`: the PROF sleeve reproduces quality_junk's ranks (re-derived
  by hand, within sector × measure with the MIN_GROUP fallback); the three-way composite is the
  mean of the three ranks before smoothing; `combine: book` = ⅓ of each single-sleeve book;
  causality on truncated panels; config params, `valid()` and the SHA-256 pins; `w_growth = 0`
  is the core's call; I2's long-only growth rank; book invariants; and, for `research.py`, a
  stuck cover and a refused short booked from their due open, off-target name-days both ways,
  the preview equal to a frictionless vectorbt book, and no locked test read.

Run it through the lab: `uv run python strategies/proposals_lab/lab.py run xs_three_premia B_U0
'{}' TB,T3,T1,T2 --costs2`.
