# xs_three_premia — pending ideas / proposals / thoughts

Parking lot for ideas worth keeping but not (yet) implemented. Each idea carries its why and the
falsification that would settle it. Nothing here is a commitment. Every number quoted below was
read on the OPEN tests (TB, T3, T1, T2): none of these ideas can be validated on them again.

## Tested and rejected by their pre-registered bars (RESEARCH_LOG §3-§5)

1. **PROF as a third equal rank (B_U0, the design).** It did hedge the crash it was chosen for
   (2026-H2 +3.7% / +3.9% a year on T1 / T2 against the two-premia A_U0's −22.2% / −7.9%), but it
   cost −6.8 pts of T1 alpha, −3.1 on TRAIN and −1.4 on T2 against A_U0, and lost to holding the
   three books side by side (C_mix3) on TRAIN and T1. It fails G1 and G2.
2. **PROF at half weight (I1, `w_prof: 0.5`).** T1 +10.1% (P 0.96) and T2 +5.8% (P 0.91), but
   TRAIN −3.1%, below both B_U0 and A_U0, and the 2026-H2 hedge is gone on T1 (−22.2%).
3. **Profitability growth on the long side (I2, `w_growth: 1`).** TRAIN −4.8% (T3 −7.1 pts
   against B_U0): quality_junk's wrong-sign growth premise showed up in the book.

## Not tested

- **PROF as a book, not a rank.** C_mix3 (a third each of the SI, MOM and PROF books, netted)
  read T1 +7.6% (P 0.995) at a −7.6% max drawdown and 8.3% volatility, T2 +2.2%, TRAIN +0.2% —
  half A_U0's T1 alpha at a third of its drawdown. Failure mode (f) of the design says this is a
  risk choice for the user, not an adoption, and it is a control here: it was never a candidate.
  It is also nearly xs_three_book's question with PROF in AH's place. Falsification needs data
  no design has read (the half-years after 2026-10-16, or paper trading).
- **PROF's best use in this lab was a GATE, not a rank.** xs_si_mom_composite's I1 (shorts only
  in the unprofitable half) read TRAIN +3.3, T1 +13.8, T2 +8.2 and was rejected there by its T1
  noise bar (+0.95 over A_U0). Its own IDEAS note a month-end-only gate (the daily re-read
  churned orders +21%). Not re-run here: its numbers were known (RESEARCH_LOG §0.4).
- **Beta-sized slots for a composite that includes PROF.** With equal slots PROF tilts the book
  net short (C_prof beta −0.26, B_U0 −0.21 on T1): profitable longs are low-beta, unprofitable
  shorts high-beta. quality_junk's beta sizing kept PROF alone at |beta| ≤ 0.12. Alpha already
  charges the beta run, so this is a risk refinement, not an alpha source; it would need its own
  pre-registration on fresh data.
- **A regime-conditional PROF weight** (raise PROF when the SI and MOM books are in a drawdown)
  is the obvious reading of 2026-H2 and also the most overfit one: one crash, read after the
  fact. Only on fresh data, written before it is read.

## Lab notes (proposal-only; nothing implemented here)

- **Held names whose bars vanish.** PSKY (no DB bars 2026-09-15 → 10-02) and WBD (none after
  2026-09-11) were carried short at a frozen price while the strategy refilled their slots
  (16 shorts, gross 0.93): `_on_priced_opens` waits for an open that the data never supplies.
  PSKY's cover then filled at the first open after the gap, booking +0.44% of the book across
  the hole. The lab could flag such carries in every book (a held name with no bar for k
  sessions) and close them at the last price, so no design inherits an unpriced move.
- **The reconciliation and the frictionless preview belong in the lab.** `research.preview` (a
  vectorbt-free share simulation of the targets, pinned equal to a frictionless vectorbt book)
  and the both-ways off-target count with stuck P&L from the due open are re-implemented in
  every design folder; one lab function would let every result carry them.
