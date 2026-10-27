# xs_cross_leg — parked ideas (not implemented, not run)

1. **Lab-level re-send of refused or part-filled targets** (adaptive_hedge IDEAS.md 2). Under
   `lock_cash` a cover can spend only the short's locked proceeds plus an equal margin, so a
   short squeezed past twice its entry value needs free cash; a refused cover is never re-sent
   (I1's GME, T2 2024-05-14, RESEARCH_LOG §3). A fill-retry fixed point in `lab.run_portfolio`
   (like `bt._retrim_loop`) would remove the stuck-order estimates every short book here
   carries. A lab change: proposal only.
2. **The design's chandelier iteration (I2c), not run.** adaptive_hedge TUNEDv2's ratcheted
   chandelier (lowest low since entry + 16 × ATR(48)) as an extra exit for the MOM shorts, no
   re-entry before the score leaves the short zone. Fully specified in RESEARCH_LOG §0.7; the
   variant budget went to I2b because bar (3) failed. Its values were tuned on TB + T3 for
   another book, so its TRAIN read would be partly in-sample.
3. **Both-leg beta matching** (not pre-registered; from §7's ex-ante table). The beta drift is
   mostly the LONG leg's (SI's low-days-to-cover names: ~0.8 in 2023-24, ~2.0 in 2026-H2), so
   I2b, which re-sized only the shorts, moved the book's beta the wrong way. p1's `sizing:
   beta` on both legs is the natural test; it cost SI −1.1 and MOM −2.1 pts in their own books,
   and the integrated composite (A_U0) already dominates this book, so the expected value is low.
4. **The reverse cross is not a candidate.** C_rev lost to C_U0 on every open test's point
   estimate and is never adopted after the fact (RESEARCH_LOG §0.3).
