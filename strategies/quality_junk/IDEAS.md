# quality_junk — pending ideas

Parking lot, not commitments (repo convention). The verdict is in `RESEARCH_LOG.md` §13 (round 2,
lab v2; round 1: §9-§10, §12.8): no edge; a low-beta hedge whose 2022 and 2025 crash gains are
its net short beta.

- **Use it as a sleeve, not a book.** Round 1 (lab v1) found its market residuals negatively
  correlated with the short-interest book's (−0.36 on T1; −0.38 to −0.40 on TB/T3), the two
  paying in different eras (QMJ 2018-21, short interest 2022-26); a 50/50 daily blend halved the
  drawdown but did not raise alpha per unit of volatility on T1/T2. Not re-measured on lab v2:
  the lab README asks agents not to touch another agent's results, so it is left to the
  cross-strategy round (`correlate.py blend` holds the round-1 computation).
- **Profitability on smaller names.** Profitability alone (I4) earned +5.0% (P 0.94) on names
  501-1000 at beta ≈ 0 on lab v2 (+4.8%, P 0.93 on v1), nothing on the top 500 (−0.7%) or in TRAIN
  (−0.85%), and +0.3% without its five biggest contributors. If revisited, pre-register it on a
  universe below the top 500 and read the top-5-out number first.
- **A real gross-profit line for the ROE third.** A third of filers never tag gross profit; a
  revenue − cost-of-revenue synthesis per period (the parser does it only where the tag never
  existed) would put them on Novy-Marx's measure instead of ROE (engine proposal, `providers/sec`).
- **Frozen gross-profit levels.** A few percent of the GP/A-measured names carry a GP TTM level
  unchanged for more than 200 days (`research.py staleness`). Falling back to a fresh ROE for them
  lowered the TRAIN premise slightly (IC +0.023 vs +0.026), so U0 keeps them; a `features.py`-level
  check (a TTM level that no new fiscal period moves) would be the cleaner fix, for every strategy.
- **Unadjusted prices for price floors.** Both $5 floors read closes adjusted back from today (for
  later splits and dividends); a split-as-known price would need an unadjusted close in the panels.
- **A tuner objective that reads the stitched record, and a noise bar for tuned values.** The
  lab's robust objective scores half-year alphas, each with its own beta, so a strongly negative
  beta earns a large credit in strong half-years: round 1's TUNE1 picked a −0.8 beta book that
  lost 6-10 points on names 501-1000. Round 2's TUNE2 (lab v2) picked a milder one (T1 β −0.20)
  whose gains over U0 on T1 and T2 (+1.0, +0.1) were inside the noise, which the tuned-values rule
  does not check (the iteration bar does), and which reversed on the locked names (−1.1). Scoring
  the stitched TRAIN alpha, penalizing |beta|, and holding tuned values to the iterations' noise
  bar would make the optimizer's pick comparable to what the gates read (lab-level changes).
- **Not tried:** leverage and payout dimensions of QMJ (issuance needs split-consistent share
  counts, which the DB does not have), accruals (no cash-flow concept ingested), `safety: beta`
  (would only deepen the short beta), the junk short leg against an SPY long (confounded by the
  cap-weight-minus-equal-weight spread against the equal-weight benchmark).
