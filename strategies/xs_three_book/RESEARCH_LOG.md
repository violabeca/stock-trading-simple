# xs_three_book — research log

The cross-strategy round of the proposals lab (`strategies/proposals_lab`, lab v2). The design
was selected by a design panel and a judge (`final.selected[]`, key `xs_three_book`). This log
holds the pre-registration, every run and the declared final. Results are in
`var/proposals_lab/results2/xs_three_book/`. The pre-registrations are SHA-256-snapshotted to
`var/proposals_lab/results2/xs_three_book/prereg/` before the runs they govern.

## §0 Pre-registration (2026-10-23 12:56 CEST, before any lab number of this design)

Nothing of this design had been run on a lab panel when this section was written. The code
existed and its synthetic-panel tests passed. The only measurement on a lab panel was a runtime
check: 1.4 / 0.6 / 1.3 s of component signals on T1 2024-H1, with no book and no return.

### Question

Does one netted account of the three books with T1 selection skill beat its best pair (C_core,
SI + MOM) and its best single sleeve? The three books are:

- **SI**: short interest (informed_flow_ls_p1, lab label `v6_si`);
- **MOM**: residual momentum (resid_momentum `U0v2`);
- **AH**: the adaptive hedge (adaptive_hedge `TUNEDv2`).

Does AH's short leg earn its third through name selection, or only through breadth timing
(C_idx)?

### Mechanism and evidence (from the design; all read-only previews of saved v2 books)

- Each sleeve carries a different test:
  - TB: MOM +6.5% (SI −2.5%);
  - T3: AH +5.2% (MOM −2.9%);
  - T1: all three (SI +10.4%, MOM +8.7%, AH +11.4%);
  - T2: MOM +5.5%.
- TRAIN daily correlations, TB / T3:
  - SI-MOM −0.18 / −0.38;
  - SI-AH −0.04 / −0.04;
  - MOM-AH +0.44 / +0.52.
- T1 ex-ante betas partly offset: SI +0.20, MOM −0.11, AH −0.23.
- AH's short leg is the falling-market engine:
  - down months +2.7%;
  - COVID +20.7%, 2022 bear +25.9%;
  - +9.0 pts a year over an equal-beta index short on T1, +1.2 on T2.
- The return-sum preview of the 1/3 blend:
  - TRAIN (TB + T3) +3.5% (judge +3.3%);
  - T1 +10.4% (judge +9.8%), vol 10.5%, maxDD −12.4%, beta −0.08;
  - T2 +4.8%.

### In-sample cautions (stated beside every result)

1. **TRAIN = TB + T3.** E0 has no short interest. MOM alone reads −8.2% on E0 (and −1.4% on
   E0 + TB + T3). An E0 book of this account would be MOM × 1/3 ≈ −2.7% (SI and AH silent);
   that is disclosed beside G1, computed from the saved MOM `U0v2` E0 book.
2. **T1 is partly in-sample.**
   - SI: informed_flow's studies scored from 2020-07.
   - AH: TUNEDv2 was tuned on TB + T3, which is about 2 pts of this account's G1 (C_ahu0
     measures it), and trend_following's short values were chosen on 2022-26 top-500 data.
   - Only MOM is clean on T1.
3. **TL is fresh only for the combinations.** informed_flow_ls §38P already read names
   1501-2000 for SI. TL is read by the orchestrator, once, in one batch, never by me. T4 and TL
   are not read here.

### U0 (label `U0`, overrides `{}`; the config's values)

The construction, in `strategy.py` `compose()`:

1. `core` = the panel without SPY (informed_flow_ls_p1 `_without`).
2. `w_SI`, `w_MOM`, `w_AH` = each component's own `signals(core, **its config.yaml params)`
   weights. AH is TUNEDv2: its config plus `{max_shorts 20, adx_min 40, entry_window 27,
   atr_mult 16, atr_window 48}`. The components are loaded by path under the prefixed names
   `_xs3_p1`, `_xs3_rm`, `_xs3_ah` and never edited. `strategy.SHA256` pins their six files;
   a test checks the pin.
3. `w = Σ_s b_s × w_s / g_s`, with `b_s = (gross − hedge_max) / n` = 0.30 each and `g_s` = the
   component's own nominal gross (0.98 / 0.90 / 0.98). A name long in one sleeve and short in
   another nets; a summed weight of at most 1e-12 is set to exactly 0.
4. Reindexed to the panel's columns (SPY = 0), then adaptive_hedge `_on_priced_opens`: a target
   change due on a bar without an open waits for the name's next priced open.
5. `borrow` = the element-wise maximum of the sleeves' borrow frames (missing = 0.5%).
   `retrim_k` = 2.
6. **Coverage switch.** Coverage is the share of the core panel's names closing at or above $5
   that carry a days-to-cover. On the fill bar after any decision bar where coverage is below
   `min_coverage` (0.5), the SI and AH sleeves are multiplied by 0, and their budget sits in
   cash. The audit checks that it never binds inside a TB/T3/T1/T2 window.
7. Nothing is rebalanced or estimated: a name trades only when some sleeve changes its target.
   Nominal gross is ≤ 0.90 on every bar (a test checks it).

Config: `backtest: {fill: open, order_notional: 1000}`; the declared account is about $100k
(an SI or AH slot is 1% ≈ $1,000; a MOM slot 0.75% × its scale). `data.features:
[short_interest, fund_shares]`. `optuna: n_trials 1`, `objectives [alpha_cagr]`, no search.

**Every label runs with `--hedge`.** SPY is merged into every panel, so all labels share one
panel and one benchmark. That benchmark is the equal-weight buy-and-hold of the period's names
plus SPY, one name in about 501, while the components' saved labels ran without SPY. Comparisons
with a component's saved book therefore use each book's own alpha, never a paired delta. Only
C_idx and I1 trade SPY.

### Labels (every one: `lab.py run xs_three_book <label> '<overrides>' TB,T3,T1,T2 --costs2 --hedge`)

| label | overrides | role | counted |
|---|---|---|---|
| U0 | `{}` | the design | variant 1 |
| CRISIS | `{"sleeves": "mom,ah"}` (MOM and AH at 0.45 each) | pre-registered challenger | variant 2 |
| I1 | `{"hedge_max": 0.08, "book_vol": 0.10, "beta_cap": 0.15}` | iteration: risk layer | variant 3 |
| I2 | `{"budgets": "0.36,0.36,0.18", "ah_part": "shorts"}` | iteration: crisis sleeve alone | variant 4 |
| C_core | `{"sleeves": "si,mom"}` (0.45 each) | control (AH's third) | no |
| C_idx | `{"ah_hedge": "SPY"}` | control (breadth timing without names) | no |
| C_ahu0 | `{"ah_variant": "u0"}`, **TB,T3 only** | control (AH's in-sample tune in G1) | no |
| S_si, S_mom, S_ah | `{"sleeves": "si"}` / `"mom"` / `"ah"` (0.90 each) | TL-read controls only; not run on OPEN | no |

Variant budget: 4 (U0, CRISIS, I1, I2). Controls are not counted.

Operational definitions of the variants:

- **C_idx.** AH runs on the full panel with its own `hedge: SPY`. AH keeps its longs; each of
  its short slots is held as a SPY short of the slot's size (0.30 / 0.98 × 0.98 / 30 = 1% of the
  account), timed by the same breakdown count. Only AH sees SPY; SI and MOM never do (a test
  checks it).
- **I1.**
  - The sleeves share 0.82 (0.2733 each).
  - (a) The overlay is scaled by `s = min(1, 0.10 / σ)`. σ is the trailing 126-bar realized
    volatility of the UNSCALED summed overlay, after the coverage switch (Σ w × close-to-close
    return, resid_momentum `book_scale`). s is rounded to 0.25 and floored at 0.25; it is
    decided on resid_momentum `_month_ends` rows and applied from the next open.
  - (b) On the first bar after each month end, β = Σ w_i × β_i. Here w is the book about to be
    held (the fill bar's scaled overlay) and β_i the trailing 126-bar beta to the core panel's
    equal-weight market (lab `_ex_ante_beta`'s estimator: at least 63 bars, shifted one bar, so
    it is known at the decision). The SPY weight is `h = −min(max(β − 0.15, 0) / β_SPY, 0.08)`,
    where β_SPY is SPY's own trailing beta to the same market. h is held to the next month end
    and is never a long hedge.
  - **Rounding choice, fixed now:** h is rounded UP in size to the next multiple of 0.02. The
    design says "rounded to 0.02". Rounding to the nearest step would leave up to 0.01 × β_SPY
    of beta above the cap after every hedge, which would fail bar (7)'s "every month-end ex-ante
    beta ≤ +0.15" on rounding alone. Rounding up makes the bar fail only where the 0.08 limit
    binds, which is the informative failure. The scale keeps resid_momentum's nearest rounding.
- **I2.**
  - SI 0.36 and MOM 0.36.
  - AH's short weights only, × (0.18 / 20) / (0.98 / 30): 0.9% a short slot.
  - AH's longs, a duplicate of the SI long rule, are dropped.

### Order of work

1. **Execution audit, before any return is read**: `research.py audit '{}' TB,T3,T1,T2`.
   Mechanics only, at gross 0.90, on lab v2's runner. Every rebuild is checked equal to
   `lab.book`. It counts:
   - rejected orders;
   - part-filled orders: below 99.5% of the shares asked, resid_momentum's definition (below
     99% is also reported);
   - NaN-price drops, stuck positions and position-days off target;
   - the book invariants: nominal gross ≤ 0.90, |SPY| = 0, the coverage switch never off inside
     a window, and the largest realized gross and smallest free cash.

   **Rule:** rejected plus part-filled orders on TB and T1 must be 0. Otherwise **gross drops to
   0.85 for every label**:
   - `config.yaml` ships 0.85, so U0 stays `{}`;
   - I2's budgets become `0.34,0.34,0.17`;
   - I1's overlay runs at 0.77.

   No further drop. Refusals that remain at 0.85 are reported and quantified with every number.
   T3 and T2 counts are reported, but do not trigger the rule.
2. **Plumbing check** (`research.py reproduce`). Each sleeve alone at its own gross, run through
   this strategy, must rebuild the component's saved lab book, on T1 2024-H1, TB 2020-H1 and T2
   2026-H2. The result is the largest daily |difference|, not the returns. It should be exact
   except where this account defers a change at a NaN open (`_on_priced_opens`) and the component
   did not. A larger, unexplained difference is an implementation error.
3. **U0** on TB, T3, T1, T2.
4. **Reconciliation before reading U0** (`research.py diagnose U0`). It uses adaptive_hedge
   `reconcile.py`'s pattern:
   - every refused or part-filled order, by class and with its free cash;
   - every stuck position (held while its target is 0 or on the other side) with its bars and
     its P&L from the due open;
   - position-days off target, NaN-price drops and the missing exposure;
   - the alpha per test and TRAIN with the stuck P&L removed.

   If removing the stuck P&L moves any test's alpha by more than 0.5 pt, that is stated beside
   every number of the label. Then **bar (5)**: |engine T1 alpha − `research.py preview` T1| ≤
   1.5 pts. The preview is the components' saved returns summed at the engine's budgets. A
   larger gap is treated as an implementation error, found and fixed before anything else is
   read.
5. Controls and the challenger: C_core, C_idx, CRISIS (TB, T3, T1, T2) and C_ahu0 (TB, T3); each
   gets `diagnose` (refusals and stuck P&L).
6. `lab.py noise xs_three_book U0 5 3`; `lab.py luck xs_three_book U0 T1 100`; `lab.py verdict
   xs_three_book U0` (market-neutral: G7 applies, no `--not-neutral`); `lab.py bear xs_three_book
   U0 C_core CRISIS C_idx`; `lab.py report xs_three_book C_core U0 CRISIS C_idx C_ahu0`;
   `research.py table` for the stitched TRAIN deltas.
7. I1 and I2, each pre-checked here, run, diagnosed and reported against U0.
8. The challenger rule, then the iteration rules, then the **declared final**, written into this
   log before its verdict and bear table (if it is not U0). No TL, no T4.

### Bars (fixed now)

**(1) Works.** Every applicable gate G1-G8 of `lab.py verdict` passes:
- G1 is read on TB + T3;
- G7 applies (|T1 beta| ≤ 0.3);
- G4 comes from the label's own luck run.

**(2) Combination value against the best single sleeve.** The comparison uses the components'
saved labels, read-only: SI `v6_si`, MOM `U0v2` and AH `TUNEDv2`, on the same tests (TRAIN =
their TB + T3 stitched).
- On T1 and on T2, U0's alpha ÷ volatility (stitched annualized volatility) ≥ the best of the
  three sleeves'.
- U0's alpha ≥ the mean of the three sleeves' alphas on TRAIN, T1 and T2, so that netting and
  the engine do not destroy alpha.

**(3) AH earns its third.** The paired Δ(U0 − C_core) ≥ 0 on the stitched TRAIN and on T2.

**(4) Falsification of the crisis selection.** Δ(U0 − C_idx) > 0, paired, on TRAIN, T1 and T2.
If it fails, the report states that AH's crisis alpha in this account is breadth timing, not
name selection.

**(5) Reconciliation before reading.** |engine U0 T1 alpha − preview T1 alpha| ≤ 1.5 pts.

**(6) Challenger rule.** CRISIS replaces U0 as this design's final only if:
- it passes G1-G8 (its own luck run, made only if the Δ conditions hold);
- AND it beats U0, paired Δ > 0, on TRAIN and on T2. Those are the tests where SI is out of its
  own sample.

**(7) I1 is a risk bar, deliberately not an alpha bar.** I1 is adopted only if all of these
hold:
- T1 max drawdown at most 0.8 × U0's in magnitude;
- every month-end post-hedge ex-ante beta ≤ +0.15. This is measured with the rule's own
  estimator (β + h × β_SPY) at the first bar after each month end inside every OPEN window
  (TB, T3, T1, T2);
- every realized per-period beta ≤ +0.25 (each half-year's beta to its benchmark);
- paired Δ alpha ≥ −1.0 pt on TRAIN, T1 and T2;
- G1-G8 still pass (its own luck run, made only if the other conditions hold).

**(8) I2, the protocol's iteration bar against U0.** Paired Δ ≥ 0 on TRAIN, T1 and T2, AND the
T1 Δ > 2 × U0's T1 noise sd. The sd is the ddof-1 sd of the three noise copies' T1 deltas; their
RMS is reported beside it.

**Order and budget.**
1. The challenger rule comes first. If CRISIS replaces U0, I1 and I2 (built on U0) are reported
   but cannot be adopted, and the final is CRISIS.
2. Otherwise I1, then I2. The first that passes becomes the final.
3. A second that also passes is reported, not adopted: the variant budget of 4 leaves no room
   to re-run it on top of the first.
4. With no adoption, the final is U0.

The final is declared in this log before its verdict and bear table (if not U0) and before any
TL read. TL is the orchestrator's, read once for the declared final with S_si / S_mom / S_ah in
the same read. TL bar: alpha > 0, late-fill alpha > 0, and P(alpha > 0) ≥ the best sleeve's.

**Disclosures (pre-registered).**
- E0 ≈ MOM × 1/3, from the saved MOM `U0v2` E0 book (about −2.7%).
- Every order under $250 at the declared $100k account is hand re-priced at the $1 minimum
  commission. The lab charges 10 bp of its value. The extra cost, in points of the account a
  year, is reported beside G5 (`research.py diagnose`). Re-pricing every order under $1,000,
  and the net over all orders, are reported too.

### Expected (the design's; before any number)

- **U0:**
  - T1 ≈ +9.5% (range +8.5 to +10.5, P ≥ 0.98), beta −0.10 to 0, maxDD −11 to −13%;
  - T1 late fills ≈ +8%, 2× costs ≈ +9%, top-5-out ≈ +6%;
  - T2 ≈ +4.2% (P 0.85-0.90);
  - TB ≈ +1.7%, T3 ≈ +1.4%, TRAIN ≈ +2.8% (P ≈ 0.75);
  - luck ≥ 95th percentile;
  - G6 passes: all four episodes beaten, down months ≈ +1.2%;
  - about 450-550 orders a year;
  - CAGR beats buy-and-hold only on T1 (≈ 9.4% vs 8.8%).
- **Controls:**
  - C_ahu0: TRAIN ≈ +1.2%;
  - C_core: TRAIN ≈ +2.1, T1 ≈ +9.3, T2 ≈ +4.8, Q4 2018 ≈ −6%;
  - C_idx: 2-3 pts below U0 on T1, ≈ 0.5 below on T2, 1-3 below on TRAIN (partly in-sample).
- **Variants:**
  - CRISIS: TRAIN ≈ +4 (in-sample), T1 ≈ +9.5, T2 ≈ +4.5, down months ≈ +1.9%;
  - I1: T1 maxDD ≈ −8 to −9%, alpha −0.5 ± 0.5;
  - I2: T2 1-1.5 pts below U0, better falling-market numbers.
- **Mechanics (my prior):**
  - At 0.90 some long buys may still be cut on TB 2020-H1, where AH's shorts gained 20% in COVID
    and locked those gains twice.
  - So the gross rule may trigger. I would rather see it trigger than read a book that holds
    less than it asks.

### Failure modes the design names (watched, not fixed)

1. G1 is thin and partly in-sample. AH at U0 values previews TRAIN +1.2%, and TRAIN does not
   survive late fills.
2. A negative stitched beta makes G1 sensitive: a 0.05 error in β moves G1 by about 1 pt.
3. T1 is in-sample for SI and for AH's long rule.
4. Diversification decays out of sample: correlations rose on VALIDATION, and all three books
   lost in 2026-H2.
5. Beta drift: the ex-ante beta was +0.27 / +0.43 in 2026-H1 / H2. I1 addresses this.
6. Execution: realized gross above nominal under lock_cash, and a single squeeze (Wayfair cost
   13% of AH, about 4% of this account).
7. Hair orders are under-priced: MOM's scale steps make ~$190 orders.
8. AH's half-year seam is adverse (day-0 returns −7.7% over T1), which makes the lab
   conservative here.

### Machine rules

At most one heavy Python process of mine at a time (two only if `memory_pressure` ≥ 35% free).
Long jobs run in the background, with logs under `var/proposals_lab/results2/xs_three_book/`.
No IBKR, and no other strategy's folder is touched.

## §1 Execution audit and plumbing check (2026-10-23 12:57-13:00; no return read)

**Audit** (`research.py audit '{}' TB,T3,T1,T2`, gross 0.90, lab v2's runner;
`results2/xs_three_book/audit_g090.log`). Every rebuild equals `lab.book` exactly (max
difference 0.0).

| test | orders | rejected | part-filled (< 99.5%) | NaN-price drops | stuck | off-target days | max nominal gross | max realized gross | min free cash |
|---|---|---|---|---|---|---|---|---|---|
| TB | 895 | 0 | 0 | 0 | 0 | 0 | 0.885 | 0.978 | 11.3% |
| T3 | 705 | 0 | 0 | 0 | 0 | 0 | 0.900 | 0.849 | 7.6% |
| T1 | 2,178 | 0 | 0 | 0 | 0 | 0 | 0.900 | 0.972 | 6.1% |
| T2 | 2,199 | 0 | 0 | 1 | 0 | 0 | 0.900 | 1.013 | 4.8% |

- **Rule: rejected + part-filled orders on TB and T1 = 0, so gross stays 0.90** for every label.
  The audit also found 0 on T3 and T2, and 0 below the looser 99% definition.
- The one NaN-price drop is CWAN, T2 2026-H2. It is AH's known dead short slot (adaptive_hedge
  IDEAS.md 1): CWAN has no price at all in that period. Its opening target, −1.0% of the
  account, is never held.
- The invariants hold on every period:
  - |SPY| = 0 everywhere;
  - the coverage switch is never off inside a window;
  - nominal gross ≤ 0.90.
- Realized gross drifts above nominal, to 1.013 at most in T2 2024-H2. Free cash still never
  fell below 4.8% of the book, so no order was cut.
- The Wayfair short of TB 2020-H1 (adaptive_hedge §14) is 1% of this account rather than 3.3%
  of AH's. Its cover found free cash and filled. U0's order log (`wayfair_U0.log`):
  - the short was filled in full on 2020-03-12 (−1.0%);
  - by the cover its weight had grown to −2.9%;
  - the chandelier cover was filled in full on 2020-04-27, with $1,740 of free cash on a ~$10k
    book.
- So my prior that the gross rule might trigger was wrong. At a third of the account, no single
  short can reach the free-cash limit that refused AH's covers.

**Plumbing** (`research.py reproduce`; `reproduce.log`). Each sleeve ran alone through this
strategy at its own gross: SI at 0.98, MOM at 0.90, AH at 0.98. Each rebuilt its component's
saved lab book with a largest daily |difference| of **0.0** in all nine (sleeve, period) pairs:
T1 2024-H1, TB 2020-H1 and T2 2026-H2 for each of SI `v6_si`, MOM `U0v2` and AH `TUNEDv2`. So
the following are all exact:
- the wrapper (loading by path, the SPY strip, budget scaling, the borrow maximum);
- the lab's runner;
- the added SPY column.

## §2 U0 (`lab.py run xs_three_book U0 '{}' TB,T3,T1,T2 --costs2 --hedge`, 13:00-13:01)

**Reconciliation first** (`research.py diagnose U0`; `diagnose_U0.log`, `diag-U0.pkl`). The
rebuild with order logs equals the saved returns (0.0) in all 27 periods.
- Refused orders: 0. Part-filled: 0 (below 99.5%, and below 99%).
- Stuck positions: 0. Position-days off target: 0. Missing exposure: 0.
- One NaN-price drop: CWAN, T2 2026-H2, AH's dead short slot (§1).
- Removing the stuck P&L changes nothing: it is 0.
- **No refusal matters here, so every U0 number below is the account the strategy asked for.**

**Bar (5): passes.** Engine T1 alpha +9.78% against the preview's +9.84% (`research.py preview`,
`preview.log`), a gap of 0.06 pt against the 1.5-pt bar. The preview reproduces the judge's
return sums: U0 TRAIN +3.3%, T1 +9.8%, T2 +4.6%; C_core TRAIN +2.1%, T1 +9.3%, T2 +4.8%;
C_ahu0 TRAIN +1.2%.

| | TB | T3 | **TRAIN (TB + T3)** | T1 | T2 |
|---|---|---|---|---|---|
| alpha (P) | +3.22% (0.77) | +1.64% (0.64) | **+3.88% (0.82)** | **+9.78% (0.995)** | **+4.52% (0.91)** |
| preview (return sum) | +2.20% | +1.61% | +3.28% | +9.84% | +4.59% |
| beta | −0.06 | +0.07 | −0.03 | −0.08 | −0.10 |
| vol / max DD | 6.5% / −7.0% | 6.2% / −7.4% | | 9.9% / −11.8% | 8.9% / −15.1% |
| late fills (+50 bp a leg) | +1.28% | −0.30% | +1.97% | +7.89% | +2.54% |
| 2× costs | +2.75% | +1.17% | +3.41% | +9.34% | +4.05% |
| top-5 names out | +0.14% | −1.48% | | +6.02% | +1.48% |
| CAGR (B&H) | +2.6% (+11.2%) | +4.1% (+35.2%) | | +9.1% (+8.7%) | +3.7% (+7.6%) |
| orders a year | 448 | 470 | | 455 | 460 |

- **Per period.**
  - T1: +15.0, +14.1, +12.1, +5.5, +26.8, −4.8, +14.6, +25.9, −1.7, −25.4. 2026-H2 is the worst
    period, at −25.4 annualized over 3.5 months; all three books lost there, as the design
    expected.
  - T2: +16.6, +3.8, −8.8, −7.0, +24.8, −0.5, +8.1, +11.7, +7.5, −16.8.
  - TB: −4.6, +8.4, +0.7, +13.3.
  - T3: +3.6, −6.3, +12.3.
- **Engine against the preview.**
  - On T1, T2 and T3 the engine is within 0.1 pt of the preview: netting and no rebalancing
    between the sleeves cost about nothing.
  - On TB it is 1.0 pt better. Inside this account, AH's Wayfair cover of 2020-04-27 filled
    (§1), while the preview uses AH's saved book, in which the lab refused that cover. AH's
    stuck P&L removed is +3.4 pts on its own TB; one third of that is about 1.0 pt. The engine
    number is the honest one.
- **The sleeves' arithmetic target contributions**, as alpha: Σ target weight × return, before
  costs and drift (`diagnose`):

  | sleeve | TB | T3 | T1 | T2 |
  |---|---|---|---|---|
  | SI | −1.0 | +1.5 | +3.4 | +1.2 |
  | MOM | +3.0 | −0.9 | +3.2 | +1.7 |
  | AH | +1.9 | +3.0 | +4.2 | +1.6 |

  Each test has a sleeve that carries it. The long leg (realized) is +1.0 / −2.0 / +5.1 / +5.5,
  and the short leg +2.6 / +4.5 / +6.2 / +0.7. The mean gross is 0.37 long and 0.41-0.45 short.
- **Hair orders** (pre-registered disclosure, beside G5). At the declared $100k account, 32-60
  orders a year are under $250. Re-pricing them at the $1 minimum costs 0.03-0.05 pt a year.
  Re-pricing every order under $1,000 costs 0.10-0.15 pt a year, and the net over all orders is
  0.06-0.12 pt. So T1's late-fill alpha (+7.89%) and 2×-cost alpha (+9.34%) stay far above 0.
- **Disclosed beside G1.** MOM's own E0 is −8.2%, and an E0 book of this account would be about
  MOM × 1/3 ≈ −2.7%, with SI and AH silent. G1 is read on TB + T3 only, and it is partly
  in-sample for AH (C_ahu0 measures how much).

## §3 Controls and the challenger (runs 13:04-13:09; `report_controls.log`, `deltas_controls.log`)

Every label ran `TB,T3,T1,T2 --costs2 --hedge` (C_ahu0: TB, T3) and was reconciled first
(`diagnose_<label>.log`). All 27 (7) rebuilds are exact. None of the four controls has a
refused, part-filled or stuck order. CRISIS has the same CWAN NaN-price drop as U0; C_idx has
none, because its AH shorts are SPY.

| label | TB | T3 | **TRAIN** (P) | T1 (P) | T1 beta | T2 (P) | T1 late / 2× / top-5 out | orders/yr T1 |
|---|---|---|---|---|---|---|---|---|
| **U0** | +3.22% | +1.64% | **+3.88% (0.82)** | **+9.78% (0.995)** | −0.08 | **+4.52% (0.91)** | +7.89 / +9.34 / +6.02 | 455 |
| C_core (SI + MOM) | +2.29% | −0.43% | +2.00% (0.71) | +9.36% (0.992) | +0.02 | +4.31% (0.91) | +7.42 / +8.89 / +5.95 | 369 |
| C_idx (AH shorts as SPY) | +2.06% | −3.44% | +1.56% (0.67) | +6.43% (0.981) | −0.04 | +3.55% (0.89) | +4.61 / +5.99 / +3.00 | 418 |
| CRISIS (MOM + AH) | +5.86% | +1.29% | +5.22% (0.84) | +9.86% (0.993) | −0.18 | +4.80% (0.88) | +8.01 / +9.42 / +6.67 | 360 |
| C_ahu0 (AH untuned) | +1.86% | −1.80% | +0.74% (0.57) | — | — | — | — | — |

Paired Δ, P(Δ > 0) on the same days and blocks:

| | TB | T3 | **TRAIN** | T1 | T2 |
|---|---|---|---|---|---|
| U0 − C_core | +0.93 (0.76) | +2.07 (0.80) | **+1.89 (0.92)** | +0.42 (0.62) | **+0.21 (0.56)** |
| U0 − C_idx | +1.16 (0.73) | +5.09 (0.98) | **+2.33 (0.95)** | **+3.35 (0.99)** | **+0.97 (0.76)** |
| CRISIS − U0 | +2.63 (0.88) | −0.35 (0.52) | **+1.34 (0.78)** | +0.08 (0.55) | **+0.28 (0.58)** |
| U0 − C_ahu0 | +1.37 (0.80) | +3.44 (0.94) | +3.14 (0.98) | — | — |

Falling markets (book; the equal-weight market in brackets; G6's episodes and down months):

| | Q4 2018 (−18.1%) | COVID (−37.0%) | 2022 bear (−24.5%) | Feb-Apr 2025 (−19.5%) | down months (n 38, market −4.11%) |
|---|---|---|---|---|---|
| U0 | −4.6% | +7.5% | +15.8% | +1.3% | +1.25% (71% > 0) |
| C_core | −6.2% | +1.6% | +10.4% | +1.5% | +0.60% (63%) |
| C_idx | −3.7% | +4.6% | +11.4% | +2.3% | +0.62% (66%) |
| CRISIS | −3.1% | +11.8% | +18.8% | +3.6% | +1.87% (82%) |

**Bar (3), AH earns its third: PASSES.** U0 − C_core is +1.89 on TRAIN (P 0.92) and +0.21 on T2
(P 0.56).
- The T2 margin is thin.
- The TRAIN margin is partly AH's in-sample tune: U0 − C_ahu0 is +3.14 on TRAIN (P 0.98).

**Bar (4), the falsification of the crisis selection: PASSES.** U0 − C_idx is > 0 on TRAIN
(+2.33, P 0.95), T1 (+3.35, P 0.99) and T2 (+0.97, P 0.76).
- Holding AH's short slots as SPY, with the same breadth timing and no names, gives up 3.4 pts
  on T1 and 1.0 on T2.
- In this account, AH's crisis alpha is name selection, not only breadth timing.
- C_idx is even below C_core on T3, T1 and T2: an index short timed by breakdown breadth costs
  more than it pays.
- The falling-market profile of the index version is weaker (COVID +4.6 vs +7.5, 2022 +11.4 vs
  +15.8).

**C_ahu0, AH's tune: a G1 caveat, larger than the design estimated.** With AH at its untuned U0
values, the three-book TRAIN is +0.74% (P 0.57), and its late-fill TRAIN is −1.35%. The tune is
worth +3.1 pts of U0's +3.9% G1, against the design's estimate of about 2 pts. TB + T3 is where
AH was tuned. So **G1's margin is mostly in-sample**; only T1 (partly) and T2 test the tune out
of sample.

**The challenger's two Δ conditions hold:** CRISIS − U0 is +1.34 on TRAIN (P 0.78) and +0.28 on
T2 (P 0.58). So under rule (6), CRISIS replaces U0 if it passes G1-G8, which needs its own luck
run (queued after U0's noise and luck). Read before its gates, the margins say little:
- the TRAIN edge is TB 2020-H1 (+2.6 on TB), where AH (0.45 here vs 0.30) is in-sample;
- the T2 edge is +0.28 pt with P 0.58;
- T1 is +0.08.

The rule has no noise bar, and I apply it as written. The comparison with U0's noise sd is
reported beside it (§4).

## §4 U0: noise, luck, verdict and bear (13:13-13:25)

**Noise** (`lab.py noise xs_three_book U0 5 3`: 5 names dropped a period, 3 copies;
`U0_noise_sd.log`). The copies' alpha minus the label's:

| test | deltas (pts) | sd | RMS |
|---|---|---|---|
| TB | −1.05, +0.32, −0.80 | 0.73 | 0.78 |
| T3 | −2.16, −1.45, −1.33 | 0.45 | 1.69 |
| T1 | −0.70, −0.78, +0.25 | **0.57** | 0.62 |
| T2 | −0.86, −0.79, −1.01 | 0.12 | 0.89 |

The copies sit below the label on T3 and T2, so the RMS is the honest yardstick there (the
peer_reversal lesson). The I2 bar is a T1 Δ > 2 × 0.57 = **1.15 pts**. With the RMS it would be
1.25; no decision below depends on which.

**Luck** (`lab.py luck xs_three_book U0 T1 100`; `U0_luck.log`). Real +9.78%, against a median
of +0.28% and a 95th percentile of +3.37% for 100 shuffled-name books: **percentile 100**.

**Verdict** (`lab.py verdict xs_three_book U0`, market-neutral, G7 applied; `U0_verdict.log`):

| gate | value | passed |
|---|---|---|
| G1 TRAIN-era (TB + T3) alpha > 0 | +3.88% | yes (mostly AH's in-sample tune: C_ahu0 +0.74%) |
| G2 T1 alpha > 0, P ≥ 0.90 | +9.78% (P 0.995) | yes |
| G3 T2 alpha > 0 | +4.52% | yes |
| G4 T1 beats ≥ 95% of shuffled books | 100 | yes |
| G5 T1 late fills / 2× costs | +7.89% / +9.34% | yes (hair orders re-priced at $1: −0.05 pt a year) |
| G6 every bear episode, down months > 0 | +1.25% (vs −4.11%) | yes |
| G7 \|T1 beta\| ≤ 0.3 | −0.08 | yes |
| G8 T1 without its top 5 names | +6.02% | yes |

**By the lab's definition U0 "works": every applicable gate passes.**

**Bear table** (`lab.py bear xs_three_book U0 C_core CRISIS C_idx`; TB + T3 + T1 stitched,
2018-07 → 2026-10; `bear_U0.log`):

| | Q4 2018 | COVID | 2022 bear | Feb-Apr 2025 | CAGR | max DD |
|---|---|---|---|---|---|---|
| U0 | −4.6% | +7.5% | +15.8% | +1.3% | +6.6% | −11.8% |
| C_core | −6.2% | +1.6% | +10.4% | +1.5% | +6.7% | −12.3% |
| CRISIS | −3.1% | +11.8% | +18.8% | +3.6% | +6.2% | −12.5% |
| C_idx | −3.7% | +4.6% | +11.4% | +2.3% | +4.0% | −11.4% |
| equal-weight market | −18.1% | −37.0% | −24.5% | −19.5% | (+13.8%) | (−37.4%) |
| SPY | −18.7% | −33.4% | −24.1% | −18.6% | (+15.0%) | |

Calendar years, U0 / market:

| | 2018-H2 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|
| U0 | −3.3 | +2.1 | +4.5 | +8.2 | +16.6 | +2.5 | +9.2 | +21.4 | −4.3 |
| market | −6.2 | +35.7 | +25.7 | +21.6 | −18.6 | +22.8 | +15.3 | +16.1 | +11.6 |

- **What U0 is.** U0 beats the market only in the falling years (2018-H2, 2022) and in 2025. Its
  2018-26 CAGR is 6.6%, half the market's 13.8%, at a third of the drawdown (−11.8% vs
  −37.4%).
- **On CAGR, CRISIS is behind.** U0's 6.6% is about C_core's 6.7% and above CRISIS's 6.2%.
  The market's CAGR and drawdown in brackets are the stitched figures of adaptive_hedge's bear
  table (the same names and days, without SPY); `lab.py bear` prints them as NaN.
- **The falling-market engine is AH:**
  - COVID: +7.5 (U0) against +1.6 (C_core) and +11.8 (CRISIS);
  - 2022: +15.8 against +10.4 and +18.8.
- **CAGR beats buy-and-hold only on T1** (9.1% vs 8.7%), as the design expected.

**Bar (2), combination value against the best single sleeve: FAILS on T2 and on the alpha
leg.** The components' saved books (read-only, `components.log`):

| | SI | MOM | AH | best / mean | U0 |
|---|---|---|---|---|---|
| T1 alpha ÷ vol | 0.849 | 0.812 | 0.791 | best 0.849 | **0.990** (passes) |
| T2 alpha ÷ vol | 0.340 | 0.537 | 0.327 | best 0.537 | **0.508** (fails) |
| TRAIN alpha | +1.05% | +2.69% | +5.72% | mean +3.15% | +3.88% (passes) |
| T1 alpha | +10.41% | +8.74% | +11.42% | mean +10.19% | +9.78% (fails by 0.41) |
| T2 alpha | +3.95% | +5.50% | +4.27% | mean +4.57% | +4.52% (fails by 0.05) |

- On T1 the account's return ÷ risk beats every sleeve's (0.99 vs 0.85). On T2, MOM alone has
  the better ratio.
- The alpha leg fails partly by construction. The account holds Σ b_s/g_s = 0.945 of the
  sleeves' average exposure (10% cash against the components' 98% / 90% / 98% gross). At
  matched exposure (Σ b_s/g_s × α_s) the sleeves sum to +2.97% / +9.60% / +4.35% on TRAIN /
  T1 / T2, and U0 beats that on every test (+3.88 / +9.78 / +4.52).
- So netting and the engine do not destroy alpha: bar (5)'s preview and this matched sum both
  say so. But the bar as written fails, and is reported as failed.

## §5 The challenger rule (6): CRISIS's gates (13:21-13:27)

CRISIS met rule (6)'s two Δ conditions (§3), so its own luck run and verdict were read as the
rule requires. The bear table holding CRISIS was pre-registered in step 6.

- **Luck** (`CRISIS_luck.log`): real +9.86%, against a median of +0.10% and a 95th percentile
  of +3.25%: percentile 100.
- **Verdict** (`CRISIS_verdict.log`, G7 applied): every gate passes.

| gate | U0 | CRISIS |
|---|---|---|
| G1 TRAIN-era (TB + T3) | +3.88% | +5.22% |
| G2 T1 (P) | +9.78% (0.995) | +9.86% (0.993) |
| G3 T2 | +4.52% | +4.80% |
| G4 luck percentile | 100 | 100 |
| G5 T1 late / 2× costs | +7.89% / +9.34% | +8.01% / +9.42% |
| G6 down months (all four episodes beaten by both) | +1.25% | +1.87% |
| G7 T1 beta | −0.08 | −0.18 |
| G8 T1 top-5 out | +6.02% | +6.67% |

- CRISIS reconciles cleanly: 0 refused, 0 part-filled, 0 stuck, and the CWAN drop as in U0
  (`diagnose_CRISIS.log`).
- Its hair orders cost less: 13-37 a year under $250, re-priced at +0.01 to +0.03 pt a year.
- **Rule (6) decides: CRISIS replaces U0 as this design's final.** It passes G1-G8 AND beats U0,
  paired, on TRAIN (+1.34, P 0.78) and T2 (+0.28, P 0.58).

**What the decision rests on, stated beside it:**

1. **The margins are within the noise.**
   - T2: +0.28 pt, P 0.58, against U0's T2 noise RMS of 0.89 (sd 0.12).
   - T1: +0.08.
   - The pre-registered rule has no noise bar. It decides on sign alone, as written, and I apply
     it as written.
2. **The TRAIN edge is TB, where AH is in-sample.** CRISIS − U0 is +2.63 on TB and −0.35 on T3.
   AH carries 0.45 of CRISIS against 0.30 of U0, and TUNEDv2 was tuned on TB + T3.
3. **CRISIS's G1 is even more in-sample than U0's.** A read-only disclosure from the saved
   books, not an engine book (`crisis_u0_preview.log`): the return-sum of MOM 0.45 + AH 0.45
   reads TRAIN +4.15% with AH tuned and **+1.11% (P 0.60) with AH at its U0 values**. T2 falls
   from +4.87% to +2.92% the same way.
4. **CRISIS drops the SI book but not short interest.** AH's long side is the SI long rule, so
   CRISIS still owns the names short sellers avoid. What it drops is SI's crowded-name SHORTS.
5. **CRISIS is more net short:** T1 beta −0.18, T2 −0.16, against U0's −0.08 / −0.10. That buys
   the better falling-market numbers: COVID +11.8, 2022 +18.8, down months +1.87% (82%
   positive). Its 2018-26 CAGR is 6.2%, against U0's 6.6%.

The iterations I1 and I2 are built on U0. Under the order rule they are run and reported, but
cannot be adopted (§0, "Order and budget").

## §6 Iterations I1 and I2 (runs 13:23-13:30; `iterations_read.log`, `report_iterations.log`)

Both reconcile cleanly: 0 refused, 0 part-filled, 0 stuck, CWAN's drop as in U0
(`diagnose_I1.log`, `diagnose_I2.log`). Paired Δ against U0, P(Δ > 0):

| | TRAIN | TB | T3 | T1 | T2 | T1 maxDD | T1 vol |
|---|---|---|---|---|---|---|---|
| U0 | +3.88% | +3.22% | +1.64% | +9.78% | +4.52% | −11.8% | 9.9% |
| **I1** risk layer | +3.37% · **Δ −0.51** (0.12) | −0.18 | −0.36 | +9.45% · **Δ −0.33** (0.38) | +4.22% · **Δ −0.30** (0.23) | **−8.0%** | 8.3% |
| **I2** crisis sleeve alone | +3.61% · **Δ −0.27** (0.48) | −0.37 | +1.95 | +9.27% · **Δ −0.51** (0.27) | +2.97% · **Δ −1.55** (0.02) | −12.0% | 10.1% |

**I1: REJECTED by its risk bar (7)** (luck not run: its other conditions fail).

| condition | value | holds |
|---|---|---|
| T1 maxDD ≤ 0.8 × U0's | −8.0% vs −11.8% (0.68) | yes |
| every month-end post-hedge ex-ante beta ≤ +0.15 | max TB 0.146, **T3 0.360**, **T1 0.241**, T2 0.146 | **no** |
| every realized per-period beta ≤ +0.25 | max TB 0.137, T3 0.216, T1 0.248, **T2 0.267** (2026-H1) | **no** |
| paired Δ ≥ −1.0 on TRAIN, T1, T2 | −0.51 / −0.33 / −0.30 | yes |

- The design expected T1 maxDD of −8 to −9% and alpha −0.5 ± 0.5; both came true.
- It fails where its 8% hedge limit binds (`I1_beta_check.log`). Before the hedge, the
  overlay's month-end ex-ante beta reached 0.42 (T3, 2021-07-01) and 0.30 (T1, 2026-09-01). It
  was above 0.15 at 7 of 18 month ends on T3 and 8 of 58 on T1. The hedge hit its 0.08 limit on
  T3, T1 and T2, and could not bring the beta down to 0.15.
- The vol scale bound only on T1 and T2 (0.75 in some months), never on TB or T3.
- U0's own realized period betas reach +0.41 (T1 2026-H1) and +0.29 (T2 2026-H1). I1 cuts them
  to +0.25 and +0.27. This is the design's failure mode 5 (beta drift): measured, and only half
  repaired by an 8% hedge.
- The vol scale and hedge cost 0.3-0.5 pt of alpha. They shrink T1 vol from 9.9% to 8.3%.
- In the falling markets the hedge helps nothing: Q4 2018 −3.9 vs −4.6, COVID +6.9 vs +7.5, 2022
  +14.3 vs +15.8. The vol scale de-risks after a volatile stretch, which is not when these
  episodes started.

**I2: REJECTED by the iteration bar (8):** Δ < 0 on TRAIN (−0.27), T1 (−0.51) and T2 (−1.55, P
0.02); T1 Δ < 2 × 0.57.
- As expected, it is 1.5 pts below U0 on T2, and its falling-market numbers are better:
  - Q4 2018 −2.4 vs −4.6, COVID +10.2 vs +7.5, 2022 +17.1 vs +15.8, Feb-Apr 2025 +3.8 vs +1.3;
  - down months +1.58% (82% positive) vs +1.25%.
- The crisis sleeve's own target alpha is +1.6 / +3.4 / +2.3 / −0.1 on TB / T3 / T1 / T2. AH's
  short leg alone is not a validated book on names 501-1000 (the judge's preview said the same).
- What I2 gives up is AH's long side, the doubled SI long rule. That long side carries T2:
  adaptive_hedge's T2 long-sleeve alpha is +6.2.

Under the order rule, neither could have been adopted anyway, because CRISIS replaced U0 (§5).
Both are reported as pre-registered. Variant count: **4 of 4** (U0, CRISIS, I1, I2); controls
not counted.

## §7 DECLARED FINAL (2026-10-23 13:31 CEST, before any TL or T4 read)

**FINAL = CRISIS** = `{"sleeves": "mom,ah"}`, run with `--hedge` (lab label `CRISIS`):
residual momentum and the adaptive hedge (TUNEDv2), 0.45 each, netted in one account, nominal
gross 0.90.

- It comes from the pre-registered challenger rule (6): CRISIS passes G1-G8, and beats U0, paired,
  on TRAIN (+1.34, P 0.78) and T2 (+0.28, P 0.58).
- I1 and I2 fail their bars, and could not have been adopted after the challenger rule anyway.
- No knob was chosen on a result outside those pre-registered rules.
- CRISIS's verdict (§5) was read as an input of rule (6), as that rule requires. The bear table
  holding CRISIS was pre-registered in step 6. Both are re-read for the declared final below and
  are identical by construction.

**Status, by the lab's definitions: "works"** (every applicable gate, G1-G8 with G7, passes).
U0 also "works". The honest reading of the choice between them:
- CRISIS and U0 are statistically indistinguishable out of sample: T1 Δ +0.08, T2 Δ +0.28 (P
  0.58).
- The rule picked the book that is more AH-weighted (0.45 vs 0.30), more net short (T1 beta
  −0.18 vs −0.08) and better in falling markets.
- Its TRAIN edge is TB 2020-H1, in-sample for AH.

**For the orchestrator's one TL batch** (I do not read TL or T4):

| role | label | overrides | flags |
|---|---|---|---|
| declared final | CRISIS | `{"sleeves": "mom,ah"}` | `--hedge` (`"hedge": true`) |
| same-read control | S_si | `{"sleeves": "si"}` | `--hedge` |
| same-read control | S_mom | `{"sleeves": "mom"}` | `--hedge` |
| same-read control | S_ah | `{"sleeves": "ah"}` | `--hedge` |
| (U0, the design's default, if wanted) | U0 | `{}` | `--hedge` |

- TL bar (pre-registered): alpha > 0, late-fill alpha > 0, and P(alpha > 0) ≥ the best sleeve's.
- On TL (names 1501-2000) the coverage switch may bind: AH sits in cash on bars where fewer
  than half the names closing ≥ $5 carry a days-to-cover.
- TL is fresh only for the combination: informed_flow_ls §38P read names 1501-2000 for SI.

### §7.1 The declared final, re-read (13:32-13:40; `FINAL_CRISIS_verdict.log`, `FINAL_CRISIS_bear.log`, `FINAL_CRISIS_read.log`)

The verdict is identical to §5's: G1-G8 all pass, G4 at percentile 100.

| | TB | T3 | **TRAIN** | T1 | T2 |
|---|---|---|---|---|---|
| alpha (P) | +5.86% (0.83) | +1.29% (0.61) | **+5.22% (0.84)** | **+9.86% (0.993)** | **+4.80% (0.88)** |
| beta | −0.11 | +0.02 | −0.09 | −0.18 | −0.16 |
| vol / max DD | 9.7% / −11.0% | 9.3% / −12.5% | | 11.1% / −12.5% | 10.0% / −19.1% |
| late / 2× costs | +3.94 / +5.45 | −0.62 / +0.83 | +3.34 / +4.80 | +8.01 / +9.42 | +2.84 / +4.33 |
| top-5 out | +2.30 | −3.37 | | +6.67 | +2.24 |
| CAGR (B&H) | +4.6% (+11.2%) | +1.9% (+35.2%) | | +8.3% (+8.7%) | +3.6% (+7.6%) |
| orders a year | 340 | 381 | | 360 | 354 |

- **Per period.**
  - T1: +18.1, +10.9, +9.7, +7.3, +33.0, −5.2, +13.0, +16.6, +3.0, −19.5.
  - T2: +13.0, +7.0, −13.4, −6.0, +30.3, +1.8, +2.8, +14.3, +17.5, −21.3.
- **Falling markets.**
  - Q4 2018 −3.1%, COVID +11.8%, 2022 bear +18.8%, Feb-Apr 2025 +3.6% (market −18.1 / −37.0 /
    −24.5 / −19.5).
  - Down months +1.87% against −4.11% (n 38, 82% positive).
  - 2018-26 CAGR 6.2% (market 13.8%), max DD −12.5% (market −37.4%).
- **Unlike U0, CRISIS does not beat buy-and-hold CAGR even on T1** (8.3% vs 8.7%).
- **Hair orders** at $100k: 13-37 a year under $250. Re-pricing them at $1 costs +0.01 to +0.03
  pt a year, and the net re-pricing of every order is −0.05 to +0.02 pt. G5 is unaffected.

**Disclosed diagnostic, run after the declaration** (it selects nothing): CRISIS noise, `lab.py
noise xs_three_book CRISIS 5 3` (`CRISIS_noise_sd.log`):

| test | deltas (pts) | sd | RMS |
|---|---|---|---|
| TB | −0.87, +0.62, −1.31 | 1.01 | 0.98 |
| T3 | −0.37, −0.48, −0.65 | 0.14 | 0.51 |
| T1 | −0.67, −0.11, +0.83 | 0.76 | 0.62 |
| T2 | −0.21, +0.30, −0.52 | 0.42 | 0.37 |

CRISIS's edge over U0 is +2.63 on TB, 2.7 times its TB noise. It is +0.28 on T2, below either
book's T2 noise RMS (0.37 / 0.89), and +0.08 on T1. The out-of-sample half of rule (6)'s evidence
is noise-sized.

## §8 Conclusion

**Answer to the design's question.** One netted account of SI + MOM + AH (U0) works by the lab's
gates. It beats its best pair (C_core) on every OPEN test, though thinly out of sample. The rule
then crowned its SI-free challenger, and the two are indistinguishable out of sample.

1. **Execution is clean at a third of the account.**
   - No order was refused or part-filled, and no position stuck, in any of the 7 labels × 27
     periods.
   - The Wayfair cover that the lab refused inside adaptive_hedge's own book (§14 there) filled
     here, because a 1% slot needs little free cash. That alone puts U0's TB 1.0 pt above the
     return-sum preview.
   - Every sleeve run alone rebuilds its component's saved book to 0.0.
2. **U0 works** (G1-G8): TRAIN +3.88%, T1 +9.78% (P 0.995, beta −0.08, vol 9.9%, maxDD −11.8%,
   luck 100), T2 +4.52% (P 0.91).
   - Engine and return-sum preview agree within 0.1 pt on T1, T2 and T3.
3. **Bars.**
   - (3) passes: AH earns its third. U0 − C_core is +1.89 on TRAIN and +0.21 on T2, the T2 part
     thin.
   - (4) passes: AH's selection is not breadth timing. U0 − C_idx is +2.33 / +3.35 / +0.97.
   - (5) passes: gap 0.06 pt.
   - (2) fails. On T2, MOM alone has the better alpha ÷ vol (0.537 vs 0.508). The account's alpha
     is 0.41 and 0.05 pt below the sleeves' mean on T1 and T2: it holds only 0.945 of their
     exposure. At matched exposure, the engine beats the sum on every test.
   - (7) fails: I1's hedge limit binds.
   - (8) fails: I2 is below U0 on every test.
4. **The final is CRISIS** (MOM + AH, 0.45 each), by rule (6).
   - It passes every gate. It beats U0 on TRAIN (+1.34) and T2 (+0.28), but within the noise on
     T1 and T2.
   - Its TRAIN edge, and most of its G1, are AH's in-sample tune. With AH untuned, the
     return-sum TRAIN is +1.11%.
   - It is the more net-short book (beta −0.18) and the better falling-market book. Its CAGR is
     the lower one (6.2% vs 6.6%), and it does not beat buy-and-hold CAGR even on T1.
5. **The caveats that bound every number.**
   - G1 rests on AH's TB + T3 tune: about 3 pts of U0's +3.9 (C_ahu0 TRAIN +0.74%).
   - T1 is in-sample for SI and for AH's long rule.
   - TL is fresh only for the combinations.
   - Correlations rose out of sample. Inside U0 the sleeves' daily target-return correlations go
     from TRAIN to VALIDATION as follows:
     - SI-MOM: −0.26 / −0.38 → +0.40 / +0.40;
     - SI-AH: −0.33 / −0.05 → +0.46 / +0.51;
     - MOM-AH: +0.58 / +0.52 → +0.70 / +0.62.
   - All three books lost in 2026-H2: U0 T1 −25.4 annualized, CRISIS −19.5.
   - Beta drift is real: U0's realized period beta reached +0.41 in T1 2026-H1, and an 8% SPY
     cap (I1) could not hold 0.15.

Nothing is committed to git. strategies/xs_three_book/, tests/test_xs_three_book.py and the
results are untracked files.

### §8.1 Housekeeping after the declaration (13:41; no number changes)

- Docstrings were added to the private helpers of `strategy.py` and `research.py`, as the repo
  requires. The behavior is unchanged: U0 and I1 rebuild their saved books on T1 2026-H2 and
  TB 2020-H1 with a max |daily difference| of 0.0 (`rebuild_after_docstrings.log`), and every
  test passes.
- The §0 snapshot keeps the code as it ran (`prereg/strategy_S0_*.py`).

### §8.2 The E0 disclosure for the declared final (read-only; `E0_disclosure.log`)

There is no short interest in E0, so the coverage switch puts AH (and SI) in cash there. An E0
book of the account is its MOM sleeve alone. From MOM's saved `U0v2` E0 book, scaled and summed:
- **U0: −2.72%** (P 0.08; late fills −3.48%), as the design estimated;
- **CRISIS: −4.09%** (P 0.08; late fills −5.21%), since MOM carries 0.45 / 0.90 = half of it.

So if G1 were read on E0 + TB + T3 the way the lab reads it for price-only books, it would be
much thinner for CRISIS than its TB + T3 +5.22% suggests. That is an estimate, not an engine
book: E0 is never run for short-interest designs.
