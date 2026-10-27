# xs_si_mom_composite — research log

Cross-strategy round of the proposals lab (`strategies/proposals_lab`, lab v2). The design is
`xs_design.json` → `final.selected[key = xs_si_mom_composite]`, chosen by a design panel and a
judge. Results are written only to `var/proposals_lab/results2/xs_si_mom_composite/`.

## §0 Pre-registration (2026-10-23, written before any number of this strategy)

No backtest, premise or audit number of this strategy existed when this section was written.
The numbers quoted below are the components' published lab-v2 results (`xs_digest.json`) and
the design panel's read-only previews (return sums of saved books). Its SHA-256 snapshot is in
`var/proposals_lab/results2/xs_si_mom_composite/prereg/`.

### §0.1 Question and hypothesis

**Question.** Does ONE ranking of two nearly orthogonal signals (short interest, SI; residual
momentum, MOM) beat holding both books side by side (C_mix) and beat each sleeve alone?

**Hypothesis.** SI and MOM are two alpha sources that replicate across name sets in 2022-26
(SI: T1 +10.4%, P 0.96, T2 +4.0%; MOM: T1 +8.7%, P 0.98, T2 +5.5%, T4 +6.2%). They are nearly
orthogonal in the cross-section (Spearman −0.13 / −0.09 at TRAIN month-ends), they failed in
different TRAIN half-years, and their beta mismatches mirror each other (SI net long beta, MOM
net short). A composite rank keeps names on which both agree, nets the conflicts (SI's long 15
and MOM's short 15 share ~1 name) instead of paying to hold both sides, and balances the tail
betas without beta sizing.

### §0.2 U0 = `A_U0` (config.yaml as shipped; nothing tuned)

1. Eligible `ok` = informed_flow_ls_p1's eligible (close >= $5, a days-to-cover, fund_shares > 0).
2. `r_si` = `p1.compose(panel, dtc_window=42, skew_window=20, w_si=1, w_skew=0, skew_smooth=1,
   score_smooth=1, neutral_groups=5, short_max_si=0.30, si_mid=0.10, si_high=0.20,
   borrow_gc=0.005, borrow_mid=0.03, borrow_high=0.10, min_price=5, beta_window=126)
   ['ranks']['si']`.
3. `r_mom` = `rm.compose(panel, min_price=5)['rank']` on its month-end rows, forward-filled to
   every bar until the next month end (row reindex, `method='ffill'`), then `.where(ok)`.
4. Side composite `c = Σ w_k r_k.fillna(0.5) / Σ w_k`, `w_si = w_mom = 1`.
5. `score = c.where(ok).rolling(5, min_periods=1).mean().where(ok).rank(axis=1, pct=True)`.
6. Book `p1._select(score, close, shortable = si_share <= 0.30, q_in 0.90, q_out 0.80, slots 15,
   max_hold 126, short_stop 1.0, sides 'both')`; two passes when the long and short sleeves
   differ (longs with `sides='long'`, shorts from `short_score.mask(longs)` with
   `sides='short'`).
7. `p1._weights(longs, shorts, beta, gross 0.90, long_share 0.5, slots 15, sizing 'equal')`:
   ±3.0% a slot, constant while held; decided at close t, filled at open t+1.
8. `ah._on_priced_opens(weights, panel.open_)`.
9. `{weights, borrow (p1 tiers 0.5% / 3% / 10%), retrim_k 2.0}`; `fill: open`,
   `order_notional: 1000`.

Components are loaded by path (`_xsc_p1`, `_xsc_rm`, `_xsc_qj`, `_xsc_ah`) and pinned:

| component | SHA-256 of strategy.py |
|---|---|
| informed_flow_ls_p1 | a05e84d4176ef60bcf68c20eca41c4a6cb113e78d17089f37a3f3583b88dee67 |
| resid_momentum | 5cc54c69620cab771e1a268f2900ba0137a107220889794abb3ff6cc3bc52753 |
| quality_junk | 1867bc4de9bf56f66e468bf19008b0f0bf93ae0417fdb7574b1d549c8e48fdc6 |
| adaptive_hedge | 52684a2118b5f6779fe3a0ddbd39c230fde2f1ee1a16cfd27251c013941b936c |

The shared `informed_flow_ls` (under review) is never edited, imported or loaded.

### §0.3 Controls (same code; they decide nothing alone and are not counted as variants)

| label | overrides | what it is |
|---|---|---|
| `C_si` | `{"long_sleeves": "si", "short_sleeves": "si"}` | the SI book in this machinery (≈ v6_si at gross 0.90) |
| `C_mom` | `{"long_sleeves": "mom", "short_sleeves": "mom"}` | the MOM rank in p1's daily machinery |
| `C_mix` | `{"combine": "book"}` | ½·w(C_si) + ½·w(C_mom), each at gross 0.90, netted per name; borrow the max of the two (the same p1 tiers); retrim_k 2 |

Context only (read-only, paired on the same days): `informed_flow_ls_p1/v6_si`,
`resid_momentum/U0v2`.

### §0.4 A known non-identity, measured on the code before this section (no return read)

C_si at gross 0.98 reproduces `informed_flow_ls_p1.signals()` bit for bit when p1's silent
skew sleeve has weight 0 (test 1). p1's config has `w_skew: 1`; without option chains its
composite is `(r_si + 0.5) / 2`, the same ordering in exact arithmetic, but SI percentiles taken
inside 5 liquidity groups tie across groups, and float rounding splits those ties differently.
On the TB 2018-07 panel that changes 286 of 445,658 weight cells (5 names) of p1's own book.
So `v6_si` (run with w_skew 1) and `C_si` differ by that tie effect as well as by gross.

### §0.5 Tests, runs and order (all lab v2; `uv run python strategies/proposals_lab/lab.py …`)

TRAIN = TB + T3 (E0 has no short interest: SI data start 2018-01). VALIDATION = T1, T2. T4 and
TL are LOCKED and are not read by this agent: the orchestrator reads the declared finals' TL
once, in one batch.

1. **Mechanics audit first** (`research.py audit`, on A_U0, C_si, C_mom and C_mix, TB and T1):
   each period rebuilt with vectorbt's order log on lab v2's runner and asserted equal to
   `lab.book` (the research rebuild vs the lab's engine book); per test, orders asked,
   rejected, part-filled (< 99.5% of the shares asked), dropped at a NaN price, off-target
   name-days and the mean gross never held; plus the book invariants on the real weights (no
   name on both sides, ≤ 15 names a side, Σ|w| ≤ gross + 1e-9, one constant weight per hold for
   the rank labels). No return level is read.
   **Execution audit rule:** if rejected plus part-filled orders exceed 1% of the orders asked
   on TB or on T1 for ANY of the four labels, `gross` becomes 0.85 in config.yaml before any
   label is run, for every label.
2. Premise (`research.py premise`, TRAIN only, informational): at TRAIN month-ends, the rank IC
   and the top-15 minus bottom-15 spread against the next 21 bars' beta-adjusted return, for
   the composite beside each sleeve.
3. `run xs_si_mom_composite <label> '<overrides>' TB,T3,T1,T2 --costs2` for A_U0, C_si, C_mom,
   C_mix (no `--hedge`: SPY is not used).
4. `noise xs_si_mom_composite A_U0 5 3`.
5. `luck xs_si_mom_composite A_U0 T1 100` and `luck xs_si_mom_composite C_si T1 100` (C_si's is
   SI's first G4 read).
6. `verdict xs_si_mom_composite A_U0` (market-neutral: G7 applies).
7. `bear xs_si_mom_composite A_U0 C_si C_mom C_mix`.
8. `report xs_si_mom_composite C_mix A_U0 C_si C_mom` (+ the stitched-TRAIN paired Δ from
   `research.py`).
9. I1 (below).
10. The w_mom grid (below), then the tuned point if it differs from the base.
11. One written iteration (written here, before its numbers).
12. The final: declared here, then `verdict` and `bear` on it (and its own luck run when it is
    not A_U0). TL is read by the orchestrator, once.

Machine rules: one heavy process at a time (two only with `memory_pressure` >= 35% free), long
jobs in the background with logs in `var/proposals_lab/results2/xs_si_mom_composite/`.

### §0.6 Pre-registered bars (Δ = paired alpha difference on the same days and bootstrap blocks;
TRAIN Δ is read on the stitched TB + T3 record; "noise sd" = A_U0's `lab.noise` sd of that test)

1. **Works:** every applicable gate G1-G8 for the composite (G1 on TB + T3; G7 applies).
2. **Integration adds value:** Δ(A_U0 − C_mix) >= 0 on TRAIN, T1 and T2, and the T1 Δ > 2 ×
   A_U0's T1 noise sd. If (1) passes and (2) fails, this design's answer is to hold the two
   books side by side (C_mix, or xs_three_book's C_core), not the composite.
3. **Beats the best single sleeve on the same days:** alpha >= max(C_si, C_mom) on TRAIN and on
   T2, and >= that max − 1.0 pt on T1 (T1 is partly in-sample for SI).
4. **Falling markets:** G6, and a down-month mean above C_si's.

**Iteration I1** = `{"short_gate": "prof"}`: a name may be shorted, and stay short, only while its
within-sector profitability rank (quality_junk `compose(panel, min_price=5)['ranks']['prof']`) is
<= 0.5; missing profitability = not shortable; longs untouched. Mechanism: junk-rally crashes
enter through MOM's short side (Nov 2020, Jan-Mar 2021); shorting only the unprofitable half
removes the names whose rebounds hurt most (H37 on SI alone: Δ +0.5, P 0.55; late fills +12.0%
vs +8.0%; orders 161 vs 304 a year). **Adopted only if** Δ(I1 − base) >= 0 on TRAIN, T1 and T2
and the T1 Δ > 2 × noise sd.

**Tuning step.** An exhaustive w_mom grid {0, 0.25, …, 2} (9 points) on TB + T3 on top of the
base as it stands after I1's decision (base books only: no late or 2×-cost runs). Robust score
= mean − ½ sd of the 7 period alphas (`lab.robust`); PBO from the 9 × 7 matrix (`lab.pbo`). The
tuned point (the robust-score maximum) is **adopted only if** it beats the current base on T1
AND T2, paired, each by more than 2 × that test's noise sd. w_mom = 0 is C_si's ranking and
w_mom = 1 the base.

**One written iteration** (variant 4): written in this log from what steps 1-10 show, before its
numbers, with its mechanism; adopted by I1's rule.

**Variant budget: 4** (A_U0, I1, the tuned point, one written iteration). Controls, noise copies,
luck books and the grid's other points are not counted.

**The final.** The base after the adoption decisions (A_U0 unless an iteration or the tuned
point was adopted). If that base passes bar (1) and fails bar (2), the declared final is C_mix
(this design's answer: hold both books), with its own gates read (verdict, bear and a T1
luck run). The final is declared in this log before its verdict and bear are run, and
nothing is read on T4 or TL by this agent.

### §0.7 Expected (from the design, before any number)

- TRAIN (TB + T3): +1.5 to +4% (P 0.6-0.8); TB +1 to +4, T3 −2 to +2.
- T1: +7 to +12% (P >= 0.95), beta −0.10 to +0.15, maxDD −10 to −15% (SI −20.7, MOM −9.5),
  top-5-out > +3%, late fills +5 to +9%. T2: +4 to +7%. Luck >= 95th percentile.
- 150-250 orders a year.
- All four bear episodes beaten (components: Q4 2018 SI −8.2 / MOM −4.8; COVID −1.0 / +4.0; 2022
  +11.2 / +11.8; Feb-Apr 2025 −3.6 / +6.6). Down months +0.3 to +1.0%.
- 2026-H2 negative (both sleeves lost there).
- C_si ≈ v6_si scaled to 0.90 (TRAIN ≈ +1%, T1 ≈ +9.5%).
- I1: T1 Δ −1 to +1.5, orders −10 to −30%, late-fill alpha +0.5 to +2.
- w_mom grid PBO >= 0.5: the tuned point is expected not to be adopted.
- TL (read by the orchestrator): +5 to +10%.

### §0.8 Failure modes to watch

(a) dilution of SI's extremes (T1 below C_si without MOM making up for it; H31, H25);
(b) a shared crash (2026-H2: SI −38.8%, MOM −17.9% annualized), which no composite diversifies;
(c) both premise ICs are about zero on TRAIN (SI −0.017, MOM −0.003), so G1 rests on TB's path,
and MOM's E0 −8.2% is invisible to G1 (no SI there);
(d) junk-rally crashes through MOM's short side (Nov 2020, Jan-Mar 2021);
(e) churn above ~300 orders a year eroding G5;
(f) path noise: moving SI's dollar-volume groups by one name moved T3 by ±4 pts, so T3 is read
through the noise copies only;
(g) rejected or part-filled orders at gross 0.90 (hence the audit first);
(h) MOM's monthly rank changes enter through the 5-bar smoothing over 1-4 days.

### §0.9 In-sample caveats carried into every reading

- T1 is partly in-sample for SI: informed_flow's studies scored from 2020-07, and informed_flow_ls
  was developed on 2022-26 top-500 data. Only MOM is clean on T1.
- AH's TUNEDv2 was tuned on TB + T3 (AH is used here only for `_on_priced_opens`, an execution
  helper: no AH signal enters this book).
- TL is fresh only for the combination: informed_flow_ls §38P already read names 1501-2000 for SI.
- G1 is read on TB + T3; MOM's E0 −8.2% (no SI there) is disclosed beside it.

## §1 Mechanics audit (2026-10-23 12:50-12:52; `results2/xs_si_mom_composite/audit.log`)

`research.py audit A_U0,C_si,C_mom,C_mix TB,T1`: every period rebuilt with vectorbt's order log
on lab v2's runner equals `lab.book` (asserted to 1e-12). No return level was read.

| label | test | orders asked | rejected | part-filled | dropped (NaN price) | off-target name-days | max gross | max names long / short |
|---|---|---|---|---|---|---|---|---|
| A_U0 | TB | 308 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 |
| A_U0 | T1 | 617 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 |
| C_si | TB | 262 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 |
| C_si | T1 | 577 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 |
| C_mom | TB | 335 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 |
| C_mom | T1 | 708 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 |
| C_mix | TB | 586 | 0 | 0 | 0 | 0 | 0.900 | 30 / 30 |
| C_mix | T1 | 1255 | 0 | 0 | 0 | 0 | 0.900 | 30 / 30 |

- **The §0.5 rule:** refused plus part-filled = 0.00% of orders everywhere, so gross stays 0.90.
- Book invariants hold for the three rank labels (no name on both sides, ≤ 15 a side, Σ|w| ≤
  0.90, one weight per hold). C_mix holds up to 30 names a side by construction (two books of 15
  slots at 1.5% each, netted); its gross is ≤ 0.90 too.

## §2 Premise check (TRAIN only, informational; `logs/premise.log`)

At the 33 TRAIN month-ends (TB 18, T3 15), each label's book score against the next 21 bars'
beta-adjusted return (p1's 126-bar beta × the eligible names' equal-weight return):

| score | IC (t) | TB IC (t) | T3 IC (t) | top 15 − bottom 15, per 21 bars (t) |
|---|---|---|---|---|
| A_U0 composite | −0.017 (−0.75) | −0.025 (−0.71) | −0.006 (−0.26) | −0.03% (−0.03) |
| C_si | −0.023 (−1.71) | −0.048 (−2.54) | +0.008 (+0.53) | −0.74% (−0.95) |
| C_mom | +0.000 (0.00) | +0.013 (+0.23) | −0.015 (−0.37) | +0.57% (+0.39) |

As the design warned (failure mode c), neither premise is positive on TRAIN; TB's SI IC is
significantly negative. This decides nothing: G1 is read on the book.

## §3 U0 and the controls (runs 12:52-12:56, noise 12:56-12:59, luck 12:59-13:03)

`lab.py run … TB,T3,T1,T2 --costs2` (logs `logs/run_<label>.log`). Alpha against the
equal-weight buy-and-hold of the same names; TRAIN = TB + T3 stitched.

| | A_U0 | C_si | C_mom | C_mix |
|---|---|---|---|---|
| TRAIN alpha (P) | **+1.97% (0.62)** | +0.64% (0.52) | −0.25% (0.51) | +0.58% (0.55) |
| · TB / T3 | +3.17 / −2.36 | −2.86 / +2.30 | +1.36 / −6.66 | −0.43 / −1.75 |
| · TRAIN late / 2× costs | −0.35 / +1.41 | −1.31 / +0.17 | −2.53 / −0.83 | −1.56 / +0.06 |
| T1 alpha (P) | **+12.88% (0.962)** | +9.40% (0.964) | +9.04% (0.935) | +9.68% (0.982) |
| · beta / vol / maxDD | +0.009 / 16.3% / −21.9% | +0.136 / 11.3% / −19.3% | −0.112 / 16.0% / −18.3% | +0.012 / 11.6% / −14.8% |
| · late / 2× / top-5-out | +10.77 / +12.37 / +8.04 | +7.31 / +8.93 / +2.71 | +6.66 / +8.47 / +4.82 | +7.54 / +9.17 / +5.77 |
| · orders a year | 129 | 121 | 148 | 262 |
| T2 alpha (P) | **+5.07% (0.843)** | +2.53% (0.744) | +1.86% (0.646) | +2.35% (0.733) |
| · beta / maxDD | −0.088 / −15.6% | +0.030 / −12.3% | −0.105 / −22.9% | −0.037 / −15.2% |
| alpha ÷ residual vol: TRAIN / T1 / T2 | 0.16 / 0.79 / 0.38 | 0.09 / 0.85 / 0.25 | −0.01 / 0.57 / 0.13 | 0.07 / 0.83 / 0.23 |
| down months (n 38, market −4.12%) | +0.95% (61% > 0) | −0.02% (53%) | +1.31% (74%) | +0.65% (71%) |
| Q4 2018 (market −18.2%) | −7.0% | −8.7% | −3.3% | −6.0% |
| COVID crash (−37.0%) | −0.1% | −1.0% | +6.7% | +3.1% |
| 2022 bear (−24.5%) | +15.4% | +10.0% | +13.8% | +12.2% |
| Feb-Apr 2025 (−19.5%) | +1.7% | −3.3% | +4.2% | +0.5% |
| CAGR / max DD, 2018-07 → 2026-10 (TB + T3 + T1) | 8.9% / −21.9% | 7.0% / −19.3% | 4.2% / −27.9% | 6.0% / −14.8% |

A_U0 per period (annualized alpha): TB −6.1, +1.5, −11.7, +23.8; T3 +22.4, −14.5, −8.1; T1
+13.2, +17.3, +10.9, +14.5, +49.1, −7.8, +34.9, +23.0, −22.0, −22.2; T2 +18.9, +13.6, −7.1,
−5.9, +14.1, −6.0, +13.7, +3.6, +14.9, −7.9. Calendar years (book / market): 2019 +2.4 / +35.6,
2020 +20.2 / +25.7, 2021 −3.1 / +21.5, 2022 +15.6 / −18.6, 2023 +5.4 / +22.7, 2024 +19.2 / +15.3,
2025 +34.5 / +16.2, 2026 to Oct 16 −8.3 / +11.7. The T1 drawdown runs 2025-10-15 → 2026-07-29
(both 2026 halves −22%/yr of alpha), as expected ("2026-H2 negative"), but deeper than the
−10 to −15% the design expected.

**Noise** (`lab.noise`, 5 names dropped a period, 3 copies; copy − label alpha): TB −0.24, +1.76,
+0.92 (sd 1.01); T3 −1.38, +2.03, +3.33 (sd 2.43); **T1 −1.69, −1.64, −4.19 (sd 1.45)**; T2 −2.61,
+0.79, +0.78 (sd 1.96); stitched TRAIN −0.67, +1.93, +2.10 (sd 1.55). All three T1 copies sit
below the label: A_U0's own T1 path looks like a favourable draw (copies' mean +10.4%).

**Luck** (T1, 100 books with the names shuffled inside dollar-volume tenths): A_U0 real +12.88%,
median +0.87%, p95 +5.15%, **percentile 100**. C_si (SI's first G4 read): real +9.40%, median
+1.01%, p95 +6.19%, **percentile 100**.

**Verdict A_U0** (`logs/verdict_A_U0.log`, market-neutral): G1 +1.97% ✓ · G2 +12.88%, P 0.962 ✓ ·
G3 +5.07% ✓ · G4 100th ✓ · G5 late +10.77%, 2× +12.37% ✓ · G6 all four episodes and down months
+0.95% ✓ · G7 beta +0.009 ✓ · G8 +8.04% ✓. **All eight gates pass.** MOM's E0 −8.2%, invisible to
G1 here (no short interest before 2018), is disclosed beside it.

**Context only** (paired, read-only): A_U0 − v6_si TRAIN +0.92 (P 0.56), T1 +2.47 (P 0.65), T2
+1.12 (P 0.61); A_U0 − resid_momentum U0v2 TRAIN −0.72 (P 0.38), T1 +4.14 (P 0.77), T2 −0.43
(P 0.47).

### §3.1 The pre-registered bars (A_U0)

| bar | reading | result |
|---|---|---|
| (1) works | all eight gates | **pass** |
| (2) Δ vs C_mix ≥ 0 on TRAIN, T1, T2; T1 Δ > 2 × 1.45 = 2.91 | TRAIN +1.39 (P 0.65, 4/7 periods), T1 **+3.20** (P 0.76, 7/10), T2 +2.72 (P 0.78, 8/10) | **pass, thin on T1** (0.29 pt over the bar) |
| (3) ≥ max(C_si, C_mom) on TRAIN and T2, ≥ max − 1 pt on T1 | TRAIN 1.97 vs 0.64; T2 5.07 vs 2.53; T1 12.88 vs 9.40 − 1.0 | **pass** |
| (4) G6 and down months above C_si's | G6 ✓; +0.95% vs −0.02% | **pass** |

Honest reading of bar (2): it passes as pre-registered, but its T1 margin is inside the path noise
that the copies show. Re-centred on its copies' mean (+10.4%), A_U0's T1 lead over C_mix would be
about +0.7 pt; T2 (+2.7, 8/10 periods; the T2 copies centre on the label) and TRAIN (+1.4) carry
more of the case. The composite also runs ~40% more volatility than C_mix (T1 16.3% vs 11.6%):
per unit of residual risk it is level with C_mix and C_si on T1 (0.79 vs 0.83 / 0.85) and ahead
on TRAIN and T2. Its extra alpha is partly concentration: 15 + 15 names on which both signals
agree, against C_mix's ~30 + 30 half-size slots.

*[Review note, §12: this re-centring discounted A_U0's path only, since C_mix had no noise copies
then; §12.3 reads both books' copies on the same reduced panels (paired T1 Δ +3.1 / +3.5 / −0.4).
T2's Δ is +1.8 (P 0.69) once its 2024-05-14 stuck orders are repaired (§9).]*

## §4 Iteration I1, the profitability junk gate on shorts (13:00; `logs/run_I1.log`)

| I1 − A_U0 | TB | T3 | T1 | T2 | TRAIN |
|---|---|---|---|---|---|
| Δ (P) | −0.62 (0.44) | +4.30 (0.98) | **+0.95 (0.67)** | +3.16 (0.82) | +1.33 (0.75) |
| periods better | 3/4 | 2/3 | 8/10 | 6/10 | 5/7 |

I1 alone: TRAIN +3.30%, T1 +13.83% (P 0.976, beta −0.02, vol 16.8%, maxDD −17.9%, late +11.24%),
T2 +8.24% (P 0.942); orders 156 a year on T1 (A_U0 129: +21%, not the −10 to −30% expected — the
gate is re-read daily and names near the sector median flip in and out of it).
I1's long weights are identical to A_U0's in all 27 open periods (a positions-only check added
after §9), so its whole Δ comes from the short side.
**Rule:** Δ ≥ 0 on TRAIN, T1 and T2 ✓, but the T1 Δ +0.95 < 2.91. **Not adopted** (as the design
expected: T1 Δ −1 to +1.5). Base stays A_U0. Variants used: 2 (A_U0, I1).

## §5 The w_mom grid and the tuned point (13:02-13:06; `logs/grid_wmom.log`, `logs/run_TUNED.log`)

Exhaustive grid on TB + T3 on top of A_U0 (base books only):

| w_mom | 0 (= C_si) | 0.25 | 0.5 | 0.75 | 1 (= A_U0) | 1.25 | **1.5** | 1.75 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| robust (mean − ½ sd) | −0.048 | −0.049 | −0.057 | −0.044 | −0.063 | −0.033 | **−0.022** | −0.034 | −0.030 |
| mean period alpha | −0.009 | +0.014 | +0.018 | +0.029 | +0.011 | +0.037 | +0.049 | +0.043 | +0.045 |

PBO = **0.75**, on 9 points × the last six TRAIN periods: `lab.pbo` needs an even number of
periods and drops the oldest, TB 2018-H2 (first written "9 × 7"; corrected in §12).
G_w1.00 reproduces A_U0's seven TRAIN periods exactly. The tuned point
w_mom = 1.5 (`TUNED`, variant 3) on all open tests:

| TUNED − A_U0 | TB | T3 | T1 | T2 | TRAIN |
|---|---|---|---|---|---|
| Δ (P) | +3.42 (0.86) | +3.37 (0.93) | **−3.35 (0.10)** | **+4.71 (0.93)** | +2.92 (0.94) |

TUNED alone: TRAIN +4.89%, T1 +9.53% (P 0.90), T2 +9.78% (P 0.96). **Rule:** beat the base on T1
AND T2 by more than 2 × noise sd (2.91 / 3.92): T2 passes, T1 fails. **Not adopted**, as expected
(PBO >= 0.5). Base stays A_U0. Variants used: 3.

## §6 Pre-registration of the written iteration I2 (variant 4; 2026-10-23, before its code and numbers)

**What §3-§5 showed that motivates it.** About half of the composite's gross is a net sector bet:
the mean Σ|net weight per sector| is 0.38 (TB) and 0.48 (T1) of the 0.90 gross, and the largest
single sector averages 9.7% / 12.5% of the book (C_mix 0.31 / 0.37). In the 2025-H2 + 2026 stretch
that holds the T1 drawdown, the book's Σ weight × return splits by sector into Technology +16.1%,
Industrials −6.2% and Consumer Discretionary −4.2% (positions-only diagnostic,
`scratchpad/xsc_sector.py`). The composite's T1 volatility (16.3%) is ~40% above C_mix's.

**I2 = `{"neutral": "sector"}`** (the design's iteration idea 2), on top of A_U0:
- SI: minus days-to-cover averaged over 42 bars, ranked within (sector × dollar-volume group)
  cells — the lab's constant sector codes, a missing code its own sector — using p1's own
  `_sleeve_rank` (coverage rule unchanged). A cell with fewer than 5 eligible names on a bar
  takes the plain liquidity-group rank instead (quality_junk's MIN_GROUP rule; ~8-11% of names).
- MOM: `rm.compose(panel, min_price=5, factors="market+sector")`: the residual also removes the
  equal-weight return of the name's other sector members (fewer than 5: market only).
- Everything else as A_U0 (composite, smoothing, book, sizing, borrow, execution).

**Mechanism.** Industry crowding (sectors that are heavily shorted as a group) and industry
momentum (Moskowitz & Grinblatt 1999) are sector bets with sector-level crash risk; ranking
within sector keeps the within-industry selection that both signals are claimed to carry.

**Expected.** Lower net sector exposure (Σ|net sector| < 0.30 on T1) and lower T1 volatility
(−1 to −3 pts); alpha Δ −3 to +2 pts (uncertain: resid_momentum's market+sector residual failed
its bar in both rounds; quality_junk's within-sector safety rank cut volatility at no alpha; the
2025-26 Technology gain would be partly given up).

**Adoption (I1's rule):** Δ(I2 − A_U0) ≥ 0 on TRAIN, T1 and T2, and the T1 Δ > 2 × 1.45 = 2.91.
If adopted, I2 becomes the base, and bars (1) and (2) are re-read for it (its own T1 luck run,
verdict and bear; Δ vs C_mix with A_U0's noise sd); the §0.6 final rule then applies to it.
Otherwise the base, and the final, stay A_U0 (bars (1) and (2) passed for it).

**Code.** A new knob `neutral` ("" | "sector", default "" = every label above unchanged) in
strategy.py; the change is verified to leave every earlier label's weights identical on real
panels before I2 is run.

## §7 Iteration I2, the sector-neutral composite (13:12-13:15; `logs/run_I2.log`)

The `neutral` knob was added after §6 was snapshotted and before any I2 number. With its default
(`""`) every earlier label's weights and borrow are bit-identical to the §0 code on four real
panels (TB 2018-07, T3 2021-01, T1 2024-01, T2 2026-07; A_U0, C_si, C_mom, C_mix, I1, TUNED;
`prereg.log`); a new test pins the cell ranks and MOM's sector residual.

| I2 − A_U0 | TB | T3 | T1 | T2 | TRAIN |
|---|---|---|---|---|---|
| Δ (P) | +1.86 (0.66) | +7.72 (0.99) | **−0.03 (0.49)** | +3.34 (0.80) | **+3.33 (0.90)** |
| periods better | 2/4 | 3/3 | 4/10 | 7/10 | 5/7 |

| I2 | TRAIN | T1 | T2 |
|---|---|---|---|
| alpha (P) | +5.30% (0.835) | +12.85% (0.975) | +8.41% (0.972) |
| vol / maxDD | 10.7% / −11.6% | 15.0% / −16.5% | 11.3% / −12.8% |
| alpha ÷ residual vol (A_U0) | 0.51 (0.16) | 0.86 (0.79) | 0.75 (0.38) |
| late / 2× / top-5-out | +2.75 / +4.69 / — | +10.52 / +12.29 / +8.62 | +5.87 / +7.71 / +4.20 |

Down months +0.93%; episodes Q4 2018 −0.6%, COVID −1.6%, 2022 +8.2%, Feb-Apr 2025 −0.1% (all above
the market). Orders 143 a year on T1. Its mechanism did what §6 expected: Σ|net weight per
sector| 0.27 on T1 (A_U0 0.48) and 0.29 on TB (0.38), largest sector 7.0% / 7.7% (12.5% / 9.7%;
`logs/exposure_I2.log`; §6's scratch diagnostic now lives in `research.py exposure` and
reproduces its figures, `logs/exposure.log`).

**Rule:** Δ ≥ 0 on TRAIN ✓ and T2 ✓, but T1 −0.03 < 0 and < 2.91. **Not adopted.** What it
shows, for the record and for IDEAS.md: removing the sector bets kept T1's alpha unchanged, lifted
TRAIN and T2 by about 3.3 pts each, and cut T1 volatility (16.3 → 15.0%) and drawdown (−21.9 →
−16.5%). It was pre-registered as an alpha iteration with I1's rule, and it fails that rule on T1;
nothing here re-labels it as a risk choice after the fact. Variants used: **4 of 4** (A_U0, I1,
TUNED, I2); the budget is spent.

## §8 The final (declared 2026-10-23 13:15 CEST, before the final verdict and bear re-run, and
before any T4 or TL read — neither is read by this agent)

By the §0.6 rule: no iteration and no tuned point was adopted, so the base is **A_U0** (`'{}'`,
config.yaml as shipped). A_U0 passes bar (1) (all eight gates) and bar (2) (thin on T1), so the
§0.6 switch to C_mix does not apply. **Declared final: `xs_si_mom_composite/A_U0`, overrides
`{}`, lab flags `--costs2` (no `--hedge`).** Its luck run exists (100th percentile); its verdict
and bear are re-run below for the record.

**Final verdict and bear, re-run after the declaration** (`logs/verdict_final_A_U0.log`,
`logs/bear_final_A_U0.log`): identical to §3 — all eight gates pass; Q4 2018 −7.0%, COVID −0.1%,
2022 +15.4%, Feb-Apr 2025 +1.7%; CAGR 8.9%, max DD −21.9% over 2018-07 → 2026-10. The verdict
re-computed the T1 luck books, because the luck cache key hashes the params and `neutral` had
joined them: the 100 re-run books under the post-I2 code give the same median (+0.87%), p95
(+5.15%) and percentile (100) to the last digit, which re-confirms that the default path did not
move.

## §9 Mechanics on the tests the §0.5 rule did not cover (post hoc, no decision changes)

The pre-registered audit covered TB and T1. Re-run on T3 and T2 for the four §0.3 labels
(`audit_T3_T2.log`) and on all four open tests for I1, TUNED and I2 (`audit_iterations.log`):

- T3: nothing refused or part-filled for any label.
- **T2, A_U0, 2024-05-14: GME's cover part-filled** (18.2 of 30.1 shares; free cash $768 on a
  $10,076 book) during the May-2024 GME squeeze, and the same bar's new short NTLA was refused
  (NoCashShort). Both were stuck to the period end (32 bars): 0.31% of T2's orders, **64**
  off-target name-days (GME's stuck short 32 + NTLA's missing short 32; first counted as 32,
  because the count required a non-zero target, §12), 0.28% of the book unheld on average over
  T2's bars. It is the mechanism adaptive_hedge's repair found (§14.1 there): under lock_cash a
  cover can spend the short's locked proceeds plus an equal margin, and a squeeze that gaps far
  past the +100% stop asks for more. Here the stuck part MADE money as GME fell back: **+4.57%
  of the book from the due open** ($64.83; 1.72 pts of it on the due bar itself, which closed at
  $48.75), and NTLA's missing short would have made +0.38%. **With both orders repaired** (each
  filled at its due open, the cash it would have freed or used idle): A_U0 T2 **+4.15% (P 0.78)**
  instead of +5.07%; its T2 Δ over C_mix **+1.80 (P 0.69, 8/10 periods)** instead of +2.72.
  *First written as +2.85% / +4.55% (P 0.81) / +2.20 (P 0.73): that estimate started each gap
  at the due bar's CLOSE and so missed the due bar's move (review fix, §12).* C_si's T2 has one
  part-filled new short (MGY, −0.02%). Bars (2) and (3) still pass on the repaired figures; the
  lab number stays the record (`research.py stuck A_U0,C_si,C_mom TB,T3,T1,T2 C_mix`,
  `logs/stuck_v2.log`; the first estimate `logs/stuck_T2.log`).
- I1, TUNED and I2: nothing refused or part-filled on any open test, so their adoption decisions
  are not mechanics artifacts.
- I1, T2 2026-H2: the book invariant breaks for 13 bars (16 shorts, gross 0.93). PSKY's data ends
  while it is held short: its NaN score exits it and the slot is refilled (SGI), but the cover can
  never fill (no open), so `_on_priced_opens` keeps its −3% target. The engine carries a frozen
  position it cannot trade; nothing is refused. A_U0 never shows it.

**For the orchestrator's TL read:** names 1501-2000 are smaller and more squeeze-prone than T2's.
The audit should be repeated on TL (mechanics, before reading returns), since a squeeze cover can
exceed the lock_cash margin at gross 0.90.

## §10 Conclusion (A_U0, lab v2; T4 and TL not read)

**Answer to the design's question: yes, by the pre-registered bars, but the integration margin
on T1 is within the path noise.**

- **Bar (1), works:** all eight gates pass. TRAIN +1.97% (P 0.62; G1 on TB + T3, disclosed beside
  MOM's E0 −8.2% that this design cannot see), T1 +12.88% (P 0.962, beta +0.01, 100th luck
  percentile, without its top five names +8.0%, late fills +10.8%, 2× costs +12.4%), T2 +5.07% (P
  0.84; **+4.15%, P 0.78, with the 2024-05-14 stuck cover and refused short repaired**; first
  written +4.55%, §12), every bear episode beaten, down months +0.95%.
- **Bar (2), integration adds value over C_mix:** TRAIN +1.4, T1 +3.2, T2 +2.7 pts (paired; T2
  +1.8, P 0.69, repaired; first written +2.20, §12); the T1 margin clears 2 × noise sd by 0.29 pt,
  while all three T1 noise copies land 1.6-4.2 pts below the label. The reading "level with
  holding both books once its favourable path is discounted" (first written here) discounted
  A_U0's path only. C_mix's own copies, on the same reduced panels (§12.3), put the paired T1 Δ
  at +3.1 / +3.5 / −0.4 (mean +2.1, sd 2.2): a lead on most paths, about as large as its own
  path noise.
- **Bar (3), beats each sleeve:** TRAIN 1.97 vs 0.64 (C_si), T2 5.07 vs 2.53 (repaired 4.15 vs
  2.54), T1 12.88 vs 9.40.
- **Bar (4), falling markets:** G6 and down months +0.95% vs C_si's −0.02%.
- **Risk:** the composite's T1 volatility (16.3%) and drawdown (−21.9%, 2025-10 → 2026-07) are the
  highest of the four books (C_mix 11.6% / −14.8%); per unit of residual risk it ties C_mix and
  C_si on T1 (0.79 vs 0.83 / 0.85) and leads on TRAIN and T2. It holds 15 + 15 names, of which
  ~3-5 a side are also in C_si's book and ~1-6 in C_mom's: a genuinely new book
  (`research.py exposure`).
- **Iterations:** I1 (junk gate) +1.3 / +0.95 / +3.2 (TRAIN / T1 / T2) and I2 (sector-neutral)
  +3.3 / −0.03 / +3.3, both with lower drawdowns, both rejected by the T1 noise bar; the tuned
  w_mom 1.5 lost T1 by 3.4 (PBO 0.75). Variants: 4 of 4.
- **Regime:** like both sleeves, the edge is a 2022-2025 one; both 2026 halves lose about 22% a
  year of alpha, and the book made −8.3% in 2026 to Oct 16 against the market's +11.7%. Over
  2018-07 → 2026-10 it compounds 8.9% a year, below the equal-weight market (a dollar-neutral book
  cannot keep up with a bull market; shorts lock cash).

**In-sample caveats.** T1 is partly in-sample for SI (informed_flow's studies scored from 2020-07
and informed_flow_ls was developed on 2022-26 top-500 data); only MOM is clean there. AH enters
only as an execution helper (`_on_priced_opens`), so its TB + T3 tuning does not touch this book.
TL is fresh only for the combination: informed_flow_ls §38P already read names 1501-2000 for SI.
The w_mom grid, I1 and I2 were read on TRAIN, T1 and T2 and rejected; the final carries none of
them. C_si and v6_si differ by gross (0.90 vs 0.98) and by p1's float tie-splitting (§0.4).

**Declared final (§8): `xs_si_mom_composite/A_U0`, overrides `{}`, `--costs2`, no `--hedge`.**

## §11 Expected (§0.7) against observed

| item | expected | observed |
|---|---|---|
| TRAIN (TB + T3) | +1.5 to +4% (P 0.6-0.8) | +1.97% (P 0.62) |
| TB / T3 | +1 to +4 / −2 to +2 | +3.17 / −2.36 |
| T1 | +7 to +12% (P >= 0.95) | +12.88% (P 0.962) |
| T1 beta / maxDD | −0.10 to +0.15 / −10 to −15% | +0.009 / **−21.9%** |
| T1 top-5-out / late fills | > +3% / +5 to +9% | +8.04% / +10.77% |
| T2 | +4 to +7% | +5.07% (+4.15% with the stuck orders repaired; first written +4.55%, §12) |
| luck | >= 95th | 100th |
| orders a year | 150-250 | 129-154 |
| bear episodes / down months | all beaten / +0.3 to +1.0% | all beaten / +0.95% |
| 2026-H2 | negative | −22.2% a year |
| C_si ≈ v6_si at 0.90 | TRAIN ≈ +1%, T1 ≈ +9.5% | +0.64%, +9.40% |
| I1 | T1 Δ −1 to +1.5; orders −10 to −30%; late +0.5 to +2 | +0.95; **+21%**; +0.47 |
| w_mom grid | PBO >= 0.5, tuned point not adopted | PBO 0.75, not adopted |
| I2 (§6) | net sector exposure < 0.30, T1 vol −1 to −3, Δ −3 to +2 | 0.27, −1.3, −0.03 |
| TL | +5 to +10% | not read here (orchestrator) |

## §12 Review fixes (2026-10-23, after the adversarial review; T4 and TL still not read)

The review (`var/proposals_lab/results2/xs_si_mom_composite/review/`, `rv_*.py` / `rv_*.log`)
found no look-ahead, reproduced A_U0 exactly, matched the component pins and the snapshots'
order, and found one overstated claim (major) plus five minor points. This pass changes
`research.py` and the tests only: `strategy.py` and `config.yaml` are unchanged (their SHA-256s
are the §7 / end-of-session ones), so every saved lab label stays valid and nothing is re-run
through `lab.py run` except the post-hoc noise copies of §12.0 (4).

### §12.0 What this pass changes and runs (written before any of its numbers)

1. **(major) The stuck-order estimate started each gap at the due bar's close.**
   `research._gaps` booked `gap.shift(1) × Δclose`, so the bar the order was due on never
   counted its own open-to-close move; GME's cover was due at the 2024-05-14 open ($64.83) and
   GME closed that day at $48.75, the largest single piece of the stuck short's windfall
   [*measured in §12.2: the second largest bar; the largest is 2024-06-07*]. The
   gap's shares were also re-estimated from the close (`weight × value / close`) instead of
   taken from what the order asked. Fixed: `gap_pnl` books the due bar as `shares × (close −
   open)`, the bars after it close to close and the next order's bar the overnight move into
   its open (adaptive_hedge/reconcile.py's booking), and `_gaps` takes the gap's shares from
   vectorbt's log (`req_size × value / val_price` minus the position left). Two new tests pin
   it on a toy vectorbt book that reproduces 2024-05-14 (a part-filled cover and a refused new
   short on one bar); the first version fails them (due bar booked 0, the short's target
   −0.55 shares instead of the −0.2 asked).
2. **(minor) Off-target name-days counted only targets not held** (`& (w != 0)`), so a
   position held against a zero target (GME's stuck 40%) never counted. `off_target` now counts
   both directions; a test pins it on the same toy book.
3. **Re-runs, new log names (the old logs stay for the record):** `research.py audit` on all
   seven labels × TB, T3, T1, T2 (`audit_v2.log`: orders, refused, part-filled, off-target
   name-days, the stuck orders' gap P&L from the due open); `research.py stuck A_U0,C_si,C_mom
   TB,T3,T1,T2 C_mix` (`logs/stuck_v2.log`: every refused or part-filled order, each test's
   alpha with the gaps removed, and the paired Δ against C_mix as run and with both sides'
   gaps removed).
4. **Post hoc, informational, decides nothing: C_mix's own noise copies.** §3.1 and §10
   discounted A_U0's T1 path by its copies' mean, but C_mix has no copies, so that re-centring
   was one-sided. `lab.py noise xs_si_mom_composite C_mix 5 3` drops the same five names a
   period as A_U0's copies (same seeds, same panels, same test order), so each copy pair is a
   paired comparison on identical reduced panels: `research.py noise_delta A_U0 C_mix` reads
   Δ(A_U0 copy − C_mix copy) per copy on TRAIN, T1 and T2. Bar (2) stays as pre-registered
   (Δ on the labels, A_U0's noise sd); this only says whether the T1 margin survives both
   books' path noise. Not a variant (noise copies are not counted, §0.6).
5. **Docs:** the corrected T2 figures replace +4.55% / +2.20 / +2.85% in the README and in
   §9-§11 (each marked, first figures kept beside them); §5's PBO is "9 × 6" (lab.pbo drops the
   oldest TRAIN period for an even split); the README's test 1 wording; four README caveats
   (TRAIN late fills, TRAIN Δ vs C_mix and its T3 sign, the repaired T2 Δ, the one-sided
   re-centring); one line on what MOM's month-end decision needs live.

**What this pass can change:** the final cannot change through it. A_U0 was declared by the
§0.6 rule from the saved lab results, which this pass does not touch; the repaired T2 figures
are estimates beside the record, as in §9. If a repaired figure contradicted a bar, it would be
reported as a caveat on that bar, not used to re-decide it.

### §12.1 The corrected audit (`audit_v2.log`, 23:45-23:52; every rebuild equals `lab.book`)

All seven labels × TB, T3, T1, T2 (189 period books, each asserted equal to `lab.book` to
1e-12). Orders were refused or part-filled only in T2:

| label | test | orders asked | refused | part-filled | NaN-price drops | off-target name-days | stuck orders' gap P&L (book) |
|---|---|---|---|---|---|---|---|
| A_U0 | T2 | 647 | 1 (NTLA short) | 1 (GME cover) | 0 | **64** (first counted 32) | **+4.19%** |
| C_si | T2 | 636 | 0 | 1 (MGY short) | 0 | 0 | −0.02% |
| the other 26 label × test cells | | | 0 | 0 | 0 | 0 | 0 |

The invariants read as in §1 and §9 (C_mix holds up to 30 names a side by construction; I1's T2
carries PSKY's frozen short, 16 shorts and gross 0.93 for 13 bars). The §0.5 rule reads TB and
T1: 0% refused or part-filled there, so gross stays 0.90.

### §12.2 The corrected stuck estimate (`logs/stuck_v2.log`, 23:52-23:56)

Every rebuilt period equals the saved returns. A_U0, T2, 2024-05-14, both orders stuck for the
period's last 32 bars:

| order | asked | filled | gap | gap P&L (book) | of which the due bar |
|---|---|---|---|---|---|
| GME cover | +30.13 sh | +18.17 sh | −11.96 sh (still short) | **+4.57%** | +1.72% |
| NTLA new short | −11.64 sh | 0 (NoCashShort) | +11.64 sh (not short) | −0.38% | −0.02% |

C_si's MGY short (2024-05-15) asked −12.58 shares and got −9.26: −0.02%.

GME's stuck short, bar by bar (`gap_pnl` on the rebuilt period, −11.956 shares; total +4.5688%,
the review's figure to the last digit; `logs/gme_daily_T2_2024H1.py` / `.log`): the due bar
2024-05-14 (+1.72%) is the second-largest bar; the largest is 2024-06-07 (+2.13%), the day after
the June-6 spike on which the stuck short LOST 1.70% (then 2024-05-16 +1.35%, 05-15 +1.07%, 06-11
−0.64%). §12.0's "largest single piece" (taken from the review) is corrected to "second largest".
The swings both ways are the point: a stuck short in a live squeeze is a risk that happened to
pay here, not a return source.

| | as run (the record) | both orders repaired | first estimate (from the due close) |
|---|---|---|---|
| A_U0 T2 alpha (P) | +5.07% (0.84) | **+4.15% (0.78)** | +4.55% (0.81) |
| A_U0 − C_mix, T2 (P, periods) | +2.72 (0.78, 8/10) | **+1.80 (0.69, 8/10)** | +2.20 (0.73) |
| C_si T2 alpha | +2.53% | +2.54% | +2.54% |

The review's independent counterfactual (`review/rv_t2_counterfactual.log`, built on
adaptive_hedge/reconcile.py) gives the same +4.15% (P 0.78) and +1.80 (P 0.69), and puts the
2024-H1 period's alpha at +4.8% instead of +14.1%. TRAIN and T1 had nothing refused: their
figures are unchanged (A_U0 − C_mix TRAIN +1.39, P 0.65; T1 +3.20, P 0.76).

**Effect on the gates and bars:** none flips. G3 +4.15% > 0; bar (2) T2 +1.80 ≥ 0; bar (3) T2
4.15 ≥ max(C_si 2.54, C_mom 1.86); G6 and bar (4) read TB, T3 and T1 only. The iterations were
decided on T1 (I1 +0.95, TUNED −3.35, I2 −0.03 against the 2.91 bar), where nothing was refused;
on T2, removing A_U0's windfall could only raise their Δ (none of them had a failure).

### §12.3 Post hoc, decides nothing: C_mix's own noise copies (23:56-23:59)

`logs/noise_C_mix.log`, `logs/noise_delta_A_U0_C_mix.log`. The copies drop the same five names a
period as A_U0's (seeds 100-102; identical benchmark rows asserted). C_mix's own copy − label:
TB −0.14 / +0.47 / +0.40 (sd 0.33), T3 −0.30 / −1.63 / −1.06 (sd 0.67), T1 −1.64 / −1.93 / −0.58
(sd 0.71), T2 −0.85 / −1.85 / −0.63 (sd 0.65). C_mix's path is about half as noisy as A_U0's (T1
sd 0.71 vs 1.45), and its T1 copies also all sit below its label.

Paired Δ(A_U0 copy − C_mix copy), on identical reduced panels:

| | labels Δ (P) | copy 0 | copy 1 | copy 2 | copies mean (sd) |
|---|---|---|---|---|---|
| TRAIN | +1.39 (0.65) | +1.31 (0.63) | +3.86 (0.87) | +3.37 (0.86) | +2.84 (1.35) |
| T1 | +3.20 (0.76) | +3.14 (0.77) | +3.49 (0.79) | −0.41 (0.45) | +2.07 (2.16) |
| T2 (as run) | +2.72 (0.78) | +0.95 (0.62) | +5.36 (0.94) | +4.12 (0.89) | +3.48 (2.27) |

**Reading.** The T1 integration margin survives path noise on average (copies mean +2.1) but not
on every path (one copy −0.4); the margin's own path sd (2.2) is about its size. So "one ranking
beats holding both books on T1" is more likely than not, not established. The first reading
(§3.1 / §10: "about +0.7 pt" once A_U0's path is discounted) discounted one side only and was too
pessimistic, since both books' T1 paths were favourable draws. TRAIN's copies are all positive and
larger than the label. T2's copies are as run: they were not reconciled, and the label's own T2
margin is +1.8 once repaired. Bar (2) stays as pre-registered: pass, thin on T1.

### §12.4 Corrected claims, and what did not change

- §9-§11 and the README: T2 repaired **+4.15% (P 0.78)** replaces +4.55% (P 0.81); the T2 Δ over
  C_mix repaired **+1.80 (P 0.69)** replaces +2.20 (P 0.73); the stuck short's P&L **+4.57%** of
  the book replaces +2.85%; **64** off-target name-days replace 32. Each first figure is kept
  beside its correction.
- §5: PBO 0.75 is on 9 points × the last six TRAIN periods, not 9 × 7 (lab.pbo needs an even
  split and drops TB 2018-H2); the review recomputed it (`review/rv_claims.log`).
- README tests line: C_si reproduces p1's book with its silent skew sleeve at weight 0. p1's config
  itself (`w_skew: 1`) cannot be matched, because of float-split ties (§0.4: 286 of 445,658 cells
  on TB 2018-07; re-measured by the review, `review/rv_tie.log`).
- README caveats added: TRAIN late fills −0.35% (T3 −4.74%; G5 reads T1 only); the TRAIN Δ over
  C_mix is +1.39 (P 0.65, 4/7 periods, T3 −0.61); the repaired T2 Δ is +1.8 (P 0.69); bar (2)'s
  T1 margin is 0.29 pt over its bar (3.20 vs 2.91), with §12.3's paired copies beside it.
- README rule 3: a live run needs an exchange calendar for MOM's month-end decision
  (resid_momentum's `_month_ends` learns a month's last row from the next row, which is exact in a
  backtest; the review's truncation and perturbation runs, `review/rv_lookahead.log`, find weights
  and borrow bit-identical through every cut).
- §3's context comparisons (not bars) used A_U0's as-run T2 too. With A_U0's stuck orders
  repaired (`logs/context_T2_repaired.py` / `.log`): A_U0 − resid_momentum U0v2 T2 −0.43 (P 0.47)
  → **−1.35 (P 0.42)**; A_U0 − v6_si T2 +1.12 (P 0.61) → **+0.20 (P 0.53)**; the same script
  re-checks A_U0 − C_mix → +1.80 (P 0.69). The two published books' own T2 orders were not
  reconciled here. On names 501-1000 the composite does not beat MOM's own published book (U0v2
  +5.50%), and only matches SI's (v6_si +3.95%).
- Unchanged: strategy.py, config.yaml, the component pins, every saved lab label, every gate (G1-G8
  as §3 and §8), the four bars, the iterations' decisions, the variant count (4 of 4) and the final.

### §12.5 The final (re-declared, unchanged)

**Declared final: `xs_si_mom_composite/A_U0`, overrides `{}`, lab flags `--costs2`, no
`--hedge`** (as §8). The repaired T2 and the paired copies are estimates beside the record, read
after the declaration, and change no rule's outcome. Gates: `uv run pytest -q
tests/test_xs_si_mom_composite.py` (17 passed, 3 of them new for `research.py`), `uv run pytest
-q tests/test_strategies.py -k xs_si_mom_composite`, `uv run ruff check` clean. T4 and TL were not
read.

**For the orchestrator's TL read** (as §9): names 1501-2000 are more squeeze-prone than T2's, and
a cover can exceed lock_cash's margin at gross 0.90. The mechanics audit (refused and part-filled
orders, off-target name-days both ways, the stuck P&L from the due open) should be repeated on TL
before its returns are read. This folder's research.py and adaptive_hedge/reconcile.py both refuse
locked tests by design, so that step is the orchestrator's.

## §13 The locked read (2026-10-24, by the orchestrator)

- **The declaration came first.** `strategies/proposals_lab/FINAL_DECLARATION.md` declared A_U0 the
  overall final and fixed the confirmatory bar before any locked number existed. The bar: TL alpha > 0 with
  P ≥ 0.80 and late-fill alpha > 0; T4 is secondary.
- **The batch ran once.** `final_batch.py` then ran A_U0, C_mix, C_si and C_mom on T4 and TL under labels
  `<label>-LOCK`, beside the other designs' finals.
- **Results:**
  - TL: alpha +8.56% (P 0.963), late fills +6.50%, beta −0.10. **Confirmed.**
  - T4: +12.35% (P 0.997), late fills +10.14%.
- **Integration Δ over C_mix:** T4 +7.99 (P 0.997, 7/10 periods), TL −0.14 (P 0.49, 6/10).
- **The §12.5 audit request ran after the read, not before it.** `final_batch.py` printed the returns
  first. The audit used this file's `rebuild`, `_gaps` and `off_target`, called from a scratch script because
  the CLI refuses locked tests. Every locked period of A_U0 and C_mix rebuilt to < 1e-12 of the saved
  returns, with 0 refused or part-filled orders and 0 off-target name-days. Because it found nothing, the
  order of the two steps changes no number.
- **Luck test (`lab.py luck`, 100 books per test, run after the read; supplementary):** A_U0 beat all 100
  books on both sets.
  - T4: books' median −0.05%, 95th percentile +5.26%.
  - TL: books' median −1.06%, 95th percentile +4.98%.
- **Where to read more:** the full table, with the disclosed books, is in `FINAL_DECLARATION.md`, and the
  program's report is `PROPOSALS_RESULTS.md` at the repo root.

### §13.1 Checks after the read (2026-10-24, orchestrator; supplementary, they decide nothing)

- **All 11 locked books of the batch reconciled.** Their 220 periods rebuild exactly.
  - Of this design's labels (A_U0, C_mix, C_si, C_mom), none had a refused or part-filled order.
  - So C_si's TL +10.91%, which makes TL's integration Δ −0.14, is clean.
- **2× borrow** (`A_U0-B2`, `A_U0-B2-LOCK`, every tier doubled):
  - TRAIN +1.41%, T1 +12.33%, T2 +4.42% (Δ −0.5 to −0.65);
  - T4 +11.54% (P 0.996), TL +8.01% (P 0.951); with late fills +9.34% / +5.92%.
- **Whole shares at $33,000** (`lab.book` with `whole_shares` and `init_cash` changed in memory):
  T1 +11.89% (Δ −0.99), T2 +4.03% (−1.04), T4 +11.98% (−0.37), TL +8.51% (−0.05).
- **IBKR borrow, 2026-10-21 snapshot:** every name shorted in the last half-year of T1, T2, T4 and TL is
  in the file, with enough shares available.
  - Median fee 0.30-0.42%, below the model's tiers on average.
  - Business development companies cost more than modelled: TSLX 9.2%, ARCC 4.5%, MAIN 4.4%, GBDC 2.9%.
- **Beta per half-year (realized; ex-ante mean in brackets for 2026):**
  - T1: −0.09, −0.25, −0.40, −0.16, +0.35, +0.06, 0.00, +0.60, +0.73 (+0.65), +0.40 (+0.65).
  - T2: −0.10 … +0.36.
  - T4: −0.31 … +0.25.
  - TL: −0.34 … +0.04.

  The top-500 book has been net long since 2025-H2. In 2018-H2 (realized +0.37) it lost 7.0% in Q4 2018.
- **Post-hoc beta sizing (not a variant of this design, nothing saved as a label).** It used p1's
  `_weights(sizing="beta")`, wrapped in memory, because `strategy.py` is pinned by SHA-256 in xs_cross_leg
  and xs_three_premia.

  | test | alpha | Δ vs A_U0 | beta by half-year | max DD |
  |---|---|---|---|---|
  | T1 | +9.84% | −3.04 (P 0.09) | −0.30 … +0.48 (ex-ante 2026: +0.40) | −15.6% |
  | T2 | +5.78% | +0.71 (P 0.68) | — | — |
  | TB | +1.58% | −1.59 | — | — |
  | T3 | −1.46% | +0.90 | — | — |

  Sizes are frozen while a name is held, so the drift is cut by about 40% (2026 ex-ante +0.65 → +0.40),
  not removed.
- Scripts: the orchestrator's scratchpad (`recon/`); the report is `PROPOSALS_RESULTS.md` §4.


### §13.2 Execution timing (2026-10-27; open tests only, lab v2 runner; decides nothing)

The script is `scratchpad/timing/timing.py`. Its base book equals the saved A_U0 to < 1e-9.

| test | base alpha (decide close t, fill open t+1) | same-day close (MOC) − base | one day late (open t+2) − base |
|---|---|---|---|
| TB | +3.17% | +0.67 (P 0.92) | −0.55 (P 0.09) |
| T3 | −2.36% | +0.30 (P 0.74) | +0.23 (P 0.64) |
| T1 | +12.88% | −0.60 (P 0.19) | −0.30 (P 0.26) |
| T2 | +5.07% | +0.17 (P 0.51) | +0.61 (P 0.82) |

- **Timing does not matter.** Every difference is under 0.7 points a year and the signs are mixed. The
  signal is slow (SI twice a month, MOM monthly, the rank smoothed over 5 days), so the fill time within a
  day either way neither helps nor hurts.
- **The P&L is earned overnight.** Base book, T1, % of the book a year:

  | leg | overnight | intraday |
  |---|---|---|
  | long | +13.2 | +0.8 |
  | short | +2.4 | −1.9 |
  | equal-weight market | +6.8 | +3.5 |

  T2 is similar. Holding only overnight would mean trading every position twice a day: ~450× the book a
  year in turnover, whose costs dwarf the ~1 point it could add.

### §13.3 Counting the last month in MOM (2026-10-27; open tests only, lab v2 runner; decides nothing)

The user asked why MOM skips the last month (bars t−20 … t). Two reads, both after the final was
declared, so neither can change it.

- **Rank IC, 2025-01 → 2026-08** (`explain.skip_test`, 20 month-ends, next month's residual):
  skipped +0.030, counted +0.035, the last month alone +0.021 (no short-term reversal among these
  names). Gap +0.005, t +0.46, counted ahead in 13 of 20 months: noise.
- **Paired book, the lab's open windows** (A_U0 against A_U0 with `resid_momentum.compose(skip=0)`;
  same panels, periods, engine and costs; scratch runner, nothing saved in `results2/`):

  | window | alpha skipped | alpha counted | Δ (P Δ > 0) | late-fill Δ | counted ahead | orders / yr |
  |---|---|---|---|---|---|---|
  | E0 | +0.2% | +0.7% | +0.4 (0.76) | +0.4 | 1/4 | 28 / 26 |
  | TB | +3.2% | +4.7% | +1.5 (0.77) | +1.6 | 3/4 | 154 / 146 |
  | T3 | −2.4% | −3.2% | −0.8 (0.39) | −0.6 | 1/3 | 144 / 135 |
  | T1 | +12.9% | +10.5% | −2.3 (0.17) | −2.2 | 4/10 | 129 / 125 |
  | T2 | +5.1% | +3.3% | −1.8 (0.24) | −1.8 | 1/10 | 135 / 134 |
  | top 500, E0+TB+T3+T1 | +6.7% | +5.9% | **−0.8 (0.26)** | −0.7 | 9/21 | 116 / 111 |

  Counting the month trades about as often (no cost difference) and earns less in the book on
  both universes. The IC's small 2025-26 lead does not reach the traded extremes. **The skip stays.**

### §13.4 Why the top 500 and the $5 rule (2026-10-27; post hoc, decides nothing)

**The tiers** (A_U0 as saved, 2022-01 → 2026-10; tier characteristics at the 2025-07-01 formation):

| tier | alpha (P) | +50 bp a leg | median $ / day (lowest) | est. spread, median (90th pct) | days-to-cover / shares known | IBKR fee 2026-10-21, median (> 5%) |
|---|---|---|---|---|---|---|
| T1 top 500 | +12.9% (0.96) | +10.8% | $334M ($164M) | 2.5 bp (5.6) | 100% / 98% | 0.32% (0%) |
| T2 501-1000 | +5.1% (0.84) | +3.0% | $91M ($56M) | 3.9 bp (7.6) | 99% / 97% | 0.38% (0%) |
| T4 1001-1500 | +12.4% (1.00) | +10.1% | $32M ($20M) | 5.0 bp (9.5) | 99% / 96% | 0.39% (0%) |
| TL 1501-2000 | +8.6% (0.96) | +6.5% | $10M ($3.8M) | 6.4 bp (12.7) | 98% / 95% | 0.40% (1%) |

Spreads: Abdi-Ranaldo within each day on 1m bars (regular session, volume-0 minutes dropped), week of
2025-10-06, names with ≥ 3 full days (T1 471, T2 408, T4 322, TL 222 of 500: the covered small names
are the more liquid ones). The same estimator on DAILY bars read 43-56 bp in every tier: it measures
daily volatility, not the spread — do not use it. The smaller tiers trade fewer minutes (median 314
of 390 in TL vs 388 in T1). **The top 500 is the development universe, not a limit the data
imposes:** every tier passes, and every tier survives +50 bp a leg (8-16× its estimated half-spread).

**The $5 rule** (`min_price`, daily; the formation's own median-price ≥ $5 screen unchanged):
- Name-days below $5 inside their own half-years: T1 0.08%, T2 0.44%, T4 1.0%, TL 2.1%.
- By share price (1m estimate, same week): $2-5 18 bp (the 1-cent tick alone is 25 bp of the
  median price), $5-10 9 bp, $10-20 6 bp, > $50 3.4 bp; daily volatility ~70%/yr under $5 vs ~30%
  above $50. Only 25 DB equities under $5 traded ≥ $1M a day in 2025-H2: too few to test the strategy
  there.
- Paired books, rule off (`min_price: 1`) − on: T1 −2.0 (P 0.10), T2 −0.5 (0.36), T4 −1.5 (0.21),
  TL +1.4 (0.69); same with late fills; orders unchanged. The rule helps in 3 of 4 tiers, none
  conclusively. **It stays.**

### §13.5 The top 2000, with and without the $5 rule (2026-10-27; post hoc, decides nothing)

One book per half-year (the lab's runner, warm), 2022-01 → 2026-10, the shipped rules; only the
universe and the price rule change. Scratch runner `wide/wide.py` (one 2,211-name panel from the DB,
Yahoo split histories for 2,198 of them). Arm A reproduces the lab's T1 (+12.9%) exactly.

- **A** top 500 as shipped; **B** top 2000 (≥ $1M a day), $5 as shipped; **D** top 2000, $5 on the price
  the stock actually traded at (splits undone); **C** top 2000, no price rule. Universes: 1,870-2,000
  names (before 2024 every DB equity with ≥ $1M a day).
- **Costs:** standard; "real" = each order its own commission ($0.005 a share, $1 minimum, $1,000
  orders) + half of max(its tier's 1m spread, one cent ÷ actual price); late = standard + 50 bp a leg.

| arm | alpha, real costs (P) | +50 bp a leg | vs A, paired (P) | CAGR | vol | max DD | gross in ranks > 500 / under $5 actual |
|---|---|---|---|---|---|---|---|
| A top 500, $5 | +12.8% (0.96) | +10.8% | — | +12.9% | 16.3% | −21.9% | 0% / 0.5% |
| B top 2000, $5 adjusted | +14.4% (0.99) | +12.5% | +1.1 (0.57) | +14.1% | 14.6% | −12.7% | 69% / 1.6% |
| D top 2000, $5 actual | +11.7% (0.97) | +9.8% | −1.6 (0.40) | +11.4% | 14.6% | −14.0% | 69% / 0% |
| C top 2000, no rule | +15.2% (0.99) | +13.2% | +1.8 (0.60) | +14.8% | 15.2% | −15.3% | 71% / 5.0% |

(Alpha against each arm's own equal-weight buy-and-hold: top 500 +8.8%/yr, top 2000 +11.4-13.0%. Against
the top-500 buy-and-hold or SPY the ordering is the same. Each arm is ahead of A in 6 of 10 half-years.)

- **C − D is not the cheap stocks.** Per-name P&L in the C and D books: names under $5 added +1.6%
  in total over 4.8 years (long +1.3, short +0.3). The rest of the gap is the $5+ names: C's shorts
  +9.5% vs D's −1.8%. Adding ~40 names re-ranks everything and changes which 15 + 15 fill the slots.
  That is book-path noise of ±2-3 points a year, the same size as every Δ in the table.
- **The shipped $5 rule reads split-ADJUSTED prices** (the DB's `close` is split-adjusted). It drops
  stocks that traded above $5 but split later: NVDA from the 2019-H2 top 500 (stored $3.34, traded
  ~$134). It also admits stocks that traded below $5 but reverse-split later. Both are look-ahead. B
  carries 1.6% of gross in names actually under $5, A 0.5%. A correct rule needs the split history:
  engine proposal #45's split-events table.
- **Survivors only.** All 2,302 DB equities still trade in 2026 (none ever stopped). The smaller and
  cheaper the names, the bigger this bias.
- **Read:** widening to 2000 names gives the same alpha within noise, at lower volatility. The smaller
  drawdown is mostly 2026, when the top-500 book lost and the wider books lost less. Removing the $5
  rule adds nothing measurable (~0.3 pt a year from the cheap names) and brings their survivorship,
  shorting-margin and data risks. **Nothing changes in A_U0.** A top-2000 book is a candidate for paper
  trading, not an adoption: T2, T4 and TL, its 2022-26 data, have been read.
