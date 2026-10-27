# xs_three_premia — research log

Cross-strategy round of the proposals lab (`strategies/proposals_lab`, lab v2). The design is
`xs_design.json` → `final.selected[key = xs_three_premia]`, chosen by a design panel and a
judge. Results are written only to `var/proposals_lab/results2/xs_three_premia/`.

## §0 Pre-registration (2026-10-23, written before any number of this strategy)

No backtest, premise or audit number of this strategy existed when this section was written
(only the synthetic-panel unit tests had run). Its SHA-256 snapshot, with `strategy.py`,
`config.yaml`, `research.py` and `tests/test_xs_three_premia.py`, is in
`var/proposals_lab/results2/xs_three_premia/prereg/`.

**Continuation note (2026-10-24).** The first builder wrote this section and the code on
2026-10-23 and was interrupted (session limit) at about 13:50, before reading any number and
before snapshotting it: `results2/xs_three_premia/` held no result, and its `logs/` and
`prereg/` folders were empty. A second builder re-read the section against the design
(`xs_design.json`, `final.selected[key = xs_three_premia]`) and its rationale, kept every
design element (U0, controls, bars, iterations, variant budget, final rule), and amended it
openly where the execution checks or the disclosures needed it. Amending is legitimate because
no number of B_U0, C_prof or C_mix3 had been read by anyone; each amended passage is marked
*[amended 2026-10-24]* and §0.12 lists them. The amended section was snapshotted before any
command that reads a panel was run.

### §0.1 Question and hypothesis

**Question.** Does a THIRD signal family, within-sector profitability (PROF), make the
short-interest × residual-momentum composite regime-robust? Concretely: does one ranking of
three signals beat (a) holding the three single-sleeve books side by side (C_mix3) and (b) the
two-signal composite `xs_si_mom_composite/A_U0` — on alpha, and in the 2026-H2 crash and the
drawdowns where the mechanism says the third premium should help?

**Hypothesis (the design's mechanism, quoted with its sources).**
1. PROF is the only lab signal with a TRAIN premise: IC +0.026 (t 2.18) and tenth spread
   +0.68%/month (t 1.85) over E0 + TB + T3 (60 month-ends; quality_junk). SI (−0.017) and MOM
   (−0.003) have none.
2. PROF is orthogonal to SI (Spearman −0.02 / −0.06) and mildly correlated with MOM (+0.16 /
   +0.14).
3. PROF is SI's regime hedge: the correlation of per-period alphas over the 27 SI-era half-years
   is −0.38; 2025-H2 SI +49.4% while PROF −15.1%; in 2026-H2, the crash that hit SI (−38.8%),
   MOM (−17.9%) and AH (−23.4%), PROF made +14.1% on T1 and +45.5% on T2 (annualized; judge's
   read of the saved books, PROF = `quality_junk/I4v2`).
4. It does not tilt beta: the three-sleeve composite's tails 1.08 / 0.96 long vs 1.12 / 1.23
   short; the profitability long tail's volatility equals the universe's; PROF alone |beta| <=
   0.12 — so it is not the lottery screen (N3) that cost SI −2.1 / −2.9.
5. Book-level ⅓ blend preview (return sums of saved books, not engine books): TB + T3 +1.48%
   (P 0.69), T1 +6.33% (P 0.99, vol 6.9%, maxDD −7.6%), T2 +5.10% (P 0.989, maxDD −6.0%).
6. PROF alone (`quality_junk/I4v2`): TB +1.9, T3 −0.3, T1 −0.69%, T2 +4.97% (P 0.94, top-5-out
   +0.3%).

### §0.2 U0 = `B_U0` (config.yaml as shipped, overrides `{}`; nothing tuned)

1. Eligible `ok` = informed_flow_ls_p1's eligible (close >= $5, a days-to-cover, fund_shares > 0).
2. `r_si` = `p1.compose(panel, dtc_window=42, skew_window=20, w_si=1, w_skew=0, skew_smooth=1,
   score_smooth=1, neutral_groups=5, short_max_si=0.30, si_mid=0.10, si_high=0.20,
   borrow_gc=0.005, borrow_mid=0.03, borrow_high=0.10, min_price=5, beta_window=126)
   ['ranks']['si']` (exactly A_U0's).
3. `r_mom` = `rm.compose(panel, min_price=5)['rank']` on its month-end rows, forward-filled to
   every bar until the next month end, `.where(ok)` (exactly A_U0's).
4. `r_prof` = `qj.compose(panel, min_price=5)['ranks']['prof']` (`.where(ok)` through the
   composite): gross profit TTM / total assets, ROE (net income TTM / equity > 0) where gross
   profit is untagged; ranked within sector × measure among quality_junk's eligible names (close
   >= $5, a ratio, a 252-bar volatility on >= 126 bars); a (sector, measure) group under 5 names
   ranks against every name on its measure (`MIN_GROUP` 5). quality_junk's safety sleeve is not
   used.
5. Composite `c = (r_si + r_mom + r_prof) / 3`, each `.fillna(0.5)` (w_si = w_mom = w_prof = 1);
   `score = c.where(ok).rolling(5, min_periods=1).mean().where(ok).rank(axis=1, pct=True)`.
6. The book is A_U0's: `p1._select(score, close, shortable = si_share <= 0.30, q_in 0.90,
   q_out 0.80, slots 15, max_hold 126, short_stop 1.0, sides 'both')` (bands mirrored for
   shorts; exits first, then free slots best first; the 126-bar clock restarts in the entry
   zone; squeeze stop +100% with no re-entry before the rank has left the short zone; never
   short, or keep short, above 30% short interest / shares; a NaN score exits); `p1._weights(...,
   gross 0.90, long_share 0.5, slots 15, sizing 'equal')` = ±3.0% a slot, constant while held;
   decided at close t, filled at open t+1; `ah._on_priced_opens(weights, panel.open_)`;
   `{weights, borrow (p1 tiers 0.5% / 3% / 10% a year above 10% / 20% short interest / shares),
   retrim_k 2.0}`.
7. Config: `data.features` [short_interest, fund_shares, fund_gp_ttm, fund_assets, fund_ni_ttm,
   fund_equity] (the lab adds `sector`); `fill: open`, `order_notional: 1000`; optuna objectives
   [alpha_cagr], n_trials 1, no search.

**Code.** `strategy.py` is a thin wrapper of `xs_si_mom_composite/strategy.py`, loaded by path
as `_xsp_core` and pinned at SHA-256 `c6061153760a6caffe65766c68a91bb9e258e134bd32953b75ec6a6a3341ef65`
(the sibling's declared-final file). The core in turn pins its components: informed_flow_ls_p1
`a05e84d4…`, resid_momentum `5cc54c69…`, quality_junk `1867bc4d…`, adaptive_hedge `52684a21…`.
With `w_growth = 0` (B_U0, every control and I1) a call IS the core's call (a test pins it); the
one knob of this folder's own, `w_growth`, exists for iteration I2 and ships off. The shared
`informed_flow_ls` (under review) is never edited, imported or loaded.

### §0.3 Controls (they decide nothing alone; not counted as variants)

| label | overrides | what it is |
|---|---|---|
| `C_prof` | `{"long_sleeves": "prof", "short_sleeves": "prof"}` | PROF alone in this machinery (daily bands, p1 eligibility, equal slots) |
| `C_mix3` | `{"combine": "book"}` | ⅓ each of the SI, MOM and PROF single-sleeve books, each at gross 0.90, netted per name; borrow p1's tiers; retrim_k 2 |

Read-only from `xs_si_mom_composite` (identical code: the same pinned core file; identical
panels: the same feature list, so `lab.panel` serves the same frames): `A_U0`, `C_si`, `C_mom`.
Context only (paired, decide nothing): its `I1` (A_U0 with PROF as a short gate = this design's
iteration idea 2, see §0.4), `informed_flow_ls_p1/v6_si`, `resid_momentum/U0v2`.

### §0.4 What was known when this was written (disclosure)

- The components' lab-v2 numbers (`xs_digest.json`) and the judge's previews quoted in §0.1.
- `xs_si_mom_composite`'s complete log (that design finished before this one started). Its
  A_U0: TRAIN +1.97% (P 0.62), T1 +12.88% (P 0.962, maxDD −21.9%, 2026-H2 −22.2%/yr), T2
  +5.07% (P 0.84, maxDD −15.6%, 2026-H2 −7.9%/yr), down months +0.95%, T1 noise sd 1.45; its
  C_si, C_mom and C_mix; its I1 (PROF short gate: TRAIN +3.30, T1 +13.83, T2 +8.24), its tuned
  w_mom 1.5 and its I2 (sector-neutral).
- None of B_U0's, C_prof's or C_mix3's numbers. B_U0 and both controls are fixed by the design
  (no choice is made here); I1 and I2 below are the design's own iteration ideas, fixed here
  before any number of this strategy.
- **Consequence for the iteration menu.** The design's idea 2 ("use PROF only as a short-side
  gate, xs_si_mom_composite I1") is exactly the sibling's I1 book (A_U0 + `short_gate: prof`:
  the same code and panels), whose numbers are already read. It is NOT run as a variant here: a
  variant whose outcome is known would be a post-hoc pick, and it would re-admit through this
  design a variant the sibling rejected on its own noise bar. It appears only as a read-only
  context row, paired against B_U0.
- Bar (3)'s comparator (A_U0's alphas, its 2026-H2 periods and drawdowns) was known when bar (3)
  was copied here; bar (3) is the design's, unchanged.
- *[amended 2026-10-24]* **Known to the continuing builder** (read before this amendment; none
  of it is a number of B_U0, C_prof or C_mix3):
  - xs_si_mom_composite's §12 review fixes (written after the first builder stopped): A_U0's T2
    holds one execution failure, GME's cover part-filled and NTLA's new short refused on
    2024-05-14, stuck 32 bars; the stuck short made +4.57% of the book from its due open, so
    A_U0's T2 reads **+4.15% (P 0.78) repaired** against +5.07% as run, and its T2 Δ over C_mix
    +1.80 (P 0.69) instead of +2.72. C_mix's own noise copies put the paired T1 Δ(A_U0 − C_mix)
    at +3.1 / +3.5 / −0.4. A_U0's declared final and its gates are unchanged by that review.
  - adaptive_hedge §14 (its round-2 repair: a refused cover under lock_cash, never re-sent) and
    its `reconcile.py`; quality_junk's README and log tables for I4 / I4v2 (PROF in its own
    machinery: monthly, beta sizing, 20 slots, gross 0.98), already quoted by the design.
  - While looking up what "engine vs a research preview" meant in the other cross-strategy
    designs, fragments of xs_three_book's log (its bar (5): engine T1 alpha +9.78% against its
    return-sum preview's +9.84%) and xs_portable_alpha's research.py docstring. Other designs'
    numbers; they bear on nothing here.

### §0.5 Tests, runs and order (lab v2; `uv run python strategies/proposals_lab/lab.py …`)

TRAIN = TB + T3; VALIDATION = T1, T2. E0 is not run: there is no short interest before 2018, so
p1's eligibility (and with it the whole book) is empty there. T4 and TL are LOCKED and are not
read by this agent: the orchestrator reads the declared finals' TL once, in one batch.

1. **Mechanics audit first** (`research.py audit B_U0,C_prof,C_mix3 TB,T3,T1,T2`), before any
   `lab.py run`: each period rebuilt with vectorbt's order log on lab v2's runner (the research
   copy, `rebuild`) and asserted equal to `lab.book` (the engine book the lab scores) to 1e-12;
   per test, orders asked, refused, part-filled (< 99.5% of the shares asked), dropped at a NaN
   price and the mean gross never held; plus the book invariants on the real weights (no name on
   both sides, ≤ 15 names a side — 45 for C_mix3 —, Σ|w| ≤ gross + 1e-9, one constant weight per
   hold for the rank labels). No return level is printed or read.
   *[amended 2026-10-24]* **The reconciliation, on every open test (TB, T3, T1, T2):**
   off-target name-days counted in BOTH directions (a target not held or held on the wrong
   side, and a position held against a zero target: the stuck short of a part-filled cover;
   `research.off_target`), and every refused or part-filled order listed with its stuck P&L
   booked from its due OPEN on the shares the order asked for (`research._gaps` / `gap_pnl`,
   xs_si_mom_composite's review-fixed booking). **Engine vs a research preview:** each period's
   engine book is also compared with `research.preview`, an independent vectorbt-free share
   simulation of the same targets (fills at the open, no costs, no cash limit, the retrim cap
   included; a test pins it equal to a frictionless vectorbt book): the engine's daily gap to
   it must be its costs (reported: the mean gap a year, the mean first-day gap, every later day
   with |gap| > 0.5%). A day above 0.5% that no stuck order accounts for stops the protocol
   until it is explained.
   **Execution audit rule** (xs_si_mom_composite's): if refused plus part-filled orders exceed
   1% of the orders asked on TB or on T1 for ANY of the three labels, `gross` becomes 0.85 in
   config.yaml for every label before any label is run (bar (3)'s comparison with A_U0, run at
   0.90, would then be flagged). T3 and T2: informational (their failures are reconciled and
   their P&L quantified, step 3a).
2. Premise (`research.py premise TB,T3`, TRAIN only, informational): at TRAIN month-ends, the
   rank IC and the top-15 minus bottom-15 spread of the book scores of B_U0, C_prof and A_U0
   (same code) against the next 21 bars' beta-adjusted return.
3. `run xs_three_premia <label> '<overrides>' TB,T3,T1,T2 --costs2` for B_U0, C_prof, C_mix3
   (no `--hedge`: SPY is not used).
   3a. *[amended 2026-10-24]* `research.py stuck B_U0,C_prof,C_mix3 TB,T3,T1,T2 C_mix3`: every
   refused or part-filled order of the saved books (rebuilt and asserted equal to the saved
   returns), its P&L from the due open, and each test's alpha with it removed; and, read-only,
   `research.py stuck xs_si_mom_composite/A_U0,xs_si_mom_composite/C_si,xs_si_mom_composite/C_mom`
   (rebuilt with the sibling's own code and saved overrides), so that every bar of §0.6 can be
   shown repaired beside the record.
4. `noise xs_three_premia B_U0 5 3`.
5. `luck xs_three_premia B_U0 T1 100`.
6. `verdict xs_three_premia B_U0` (market-neutral: G7 applies).
7. `bear xs_three_premia B_U0 C_prof C_mix3`.
8. `report xs_three_premia C_mix3 B_U0 C_prof`.
9. The paired table against `xs_si_mom_composite` A_U0, C_si and C_mom (read-only,
   `research.py delta`), with the §0.3 context rows; `research.py bars B_U0` (§0.6).
10. I1, then I2 (§0.7), each audited (mechanics, all four tests) before its numbers are read.
    *[amended 2026-10-24]* Each runs as U0 does: `run xs_three_premia I1 '{"w_prof": 0.5}'
    TB,T3,T1,T2 --costs2` (then `I2 '{"w_growth": 1.0}'` or `I2h '{"w_prof": 0.5, "w_growth":
    0.5}'`), decided by `research.py adopt <label> <base>`, its stuck orders reconciled
    (`research.py stuck <label> TB,T3,T1,T2 <base>`).
11. The final (§0.8): declared here, then its verdict and bear (and its own T1 luck run when it
    is not B_U0).

Machine rules: one heavy process at a time (two only with `memory_pressure` >= 35% free), long
jobs in the background with logs in `var/proposals_lab/results2/xs_three_premia/logs/`.

### §0.6 Pre-registered bars

Δ = paired alpha difference on the same days and bootstrap blocks (`lab.paired`); the TRAIN Δ
is read on the stitched TB + T3 record; "noise sd" = B_U0's `lab.noise` sd on that test.

1. **Works:** every applicable gate G1-G8 for B_U0 (G1 on TB + T3; G7 applies). Disclosed
   beside G1, invisible to it: MOM's E0 −8.2% and PROF's E0 periods (−10.9, −5.3, +6.5, +7.9).
2. **Integration:** Δ(B_U0 − C_mix3) >= 0 on TRAIN, T1 and T2, and the T1 Δ > 2 × B_U0's T1
   noise sd.
3. **The third premium earns its place**, against `xs_si_mom_composite/A_U0` (read-only, same
   panels): Δ(B_U0 − A_U0) >= 0 on TRAIN and on T2, and >= −1.0 pt on T1; AND the mechanism's
   claim: B_U0's 2026-H2 period alpha (the last period, 2026-07-01 → 2026-10-16) is above
   A_U0's on T1 AND on T2, and B_U0's max drawdown is smaller than A_U0's on T1 AND on T2. (The
   design says: judged against C_mix3 instead if A_U0 failed its own integration bar; A_U0
   passed it, so (3) is judged against A_U0.)
4. **Falling markets:** G6, with a down-month mean >= A_U0's.
5. **Best single sleeve:** alpha >= max(C_si, C_mom, C_prof) on TRAIN and on T2.

*[amended 2026-10-24]* **Record and repaired readings.** Every bar and gate is decided on the
lab's saved books as run (the record), as in xs_si_mom_composite. Where the reconciliation
(§0.5 step 3a) finds refused or part-filled orders in any book a bar reads, the bar is also
computed with each book's stuck P&L removed (`research.py bars B_U0 --repaired`) and reported
beside it. A bar whose outcome differs between the two readings is reported as decided by an
execution artifact (fragile), not re-decided. One such artifact is already known: A_U0's T2
(+5.07% as run, +4.15% repaired, GME 2024-05-14), which bar (3)'s T2 legs read (bar (4) and G6
read TB, T3 and T1 only).
**The 2026-H2 leg of bar (3)** is read on T1 AND on T2: the design's text names no test, and
both validation tests is the stricter reading (fixed by the first builder, kept).

### §0.7 Iterations, the adoption rule and the variant budget

**Variant budget: 3** — B_U0, I1, I2. Controls, noise copies, luck books and read-only rows are
not counted.

**Adoption rule** (the lab protocol's step 5): a variant is adopted only if its Δ against the
current base is >= 0 on TRAIN, T1 and T2 AND its T1 Δ > 2 × B_U0's T1 noise sd. An iteration
whose audit shows refused plus part-filled orders above 1% on TB or T1 is reported as
mechanically compromised and not adopted.

**I1 = `{"w_prof": 0.5}`** (the design's iteration 1, run after the U0 read whatever it shows).
Mechanism: PROF's VALIDATION record is thin (T1 −0.69%; T2's +4.97% rests on five names), and
with three equal sleeves the extremes that carry SI and MOM are diluted (failure modes a, b); a
half-weight third premium keeps more of the two-premia extremes while keeping part of the
hedge. Composite `(r_si + r_mom + 0.5 r_prof) / 2.5`. Expected: between B_U0 and A_U0 on every
test, T1 Δ vs B_U0 0 to +1.5, its 2026-H2 alphas and drawdowns between the two; most likely
within the noise (not adopted).

**I2 = profitability growth on the long side** (the design's iteration 3), run after I1's
decision on top of the base as it then stands: `{"w_growth": <the base's w_prof>}` — label `I2`
= `{"w_growth": 1.0}` if I1 is rejected, `I2h` = `{"w_prof": 0.5, "w_growth": 0.5}` if I1 is
adopted (the growth rank weighs what the profitability rank weighs: the two quality dimensions
alike). The LONG composite adds `w_growth × r_growth` (quality_junk's
`compose(..., w_growth=1)['ranks']['growth']`: the 252-bar change of the name's own
profitability ratio, ranked within sector × measure; NaN without a year of the ratio → 0.5);
the short composite is unchanged; the book runs two passes (longs from the long score, shorts
from the short score with the longs masked). Mechanism: QMJ's growth dimension (Asness, Frazzini
& Pedersen 2019) — improving profitability is underpriced; long side only, so the short side
keeps SI's crowding and MOM's loser extremes. Known before writing: quality_junk's TRAIN premise
of this growth rank had the wrong sign (IC −0.010, t −1.15, 60 TRAIN month-ends, its own
universe and machinery), and it never ran a growth variant. Expected: TRAIN Δ <= 0, T1 and T2 Δ
within ±1.5: not adopted. It is run because the protocol asks for two iterations and the design
names it, while idea 2 cannot be run here (§0.4).

### §0.8 The final

The declared final is the base after the adoption decisions (B_U0 unless I1 or I2 is adopted).
If that base passes bar (1) and fails bar (2), the declared final is C_mix3 instead (this
design's answer would then be: hold the three books side by side), with its own gates read
(verdict, bear and a T1 luck run). If the base fails bar (1), it stays the declared final,
reported as failing (C_mix3's numbers are reported beside it as information: per failure mode
(f), a blend that wins on P or drawdown while losing on alpha is a risk choice for the user, not
an adoption). Bar (3) answers the design's question (does the third premium earn its place over
A_U0?) and is reported as that answer; it does not switch the final — choosing between this
design and `xs_si_mom_composite` is the orchestrator's cross-design ranking (gates first, then
the design bars). The final is declared in this log before its final verdict and bear, and
nothing is read on T4 or TL here. TL is read once by the orchestrator, for the declared final,
disclosing that informed_flow_ls §38P already read names 1501-2000 for SI.
*[amended 2026-10-24]* If an iteration is adopted, bars (1) and (2) are re-read for it (its own
T1 luck run, verdict and bear; Δ vs C_mix3 against B_U0's noise sd) and the rule above applies
to it. Per the design's rationale, the orchestrator ranks the five designs' declared finals on
OPEN (every gate first, then the design bars), declares ONE overall final, and reads TL once in
one batch with the component sleeves as controls; only the overall final's TL result is
confirmatory, the others are disclosures.

### §0.9 Expected (the design's, before any number)

- TRAIN (TB + T3): +1 to +4% (P 0.6-0.85).
- T1: +7 to +11%, "i.e. −1.5 to +0.5 against A_U0". Against the now-known A_U0 (+12.88%) the
  level range means −5.9 to −1.9 pts and the Δ range means +11.4 to +13.4%: the design wrote
  both before A_U0 was run; both are recorded.
- T2: +5 to +8%, i.e. 0 to +2 against A_U0 (A_U0 +5.07%).
- Beta −0.15 to +0.10. Max drawdown and the 2026-H2 loss smaller than A_U0's. Down months at
  least A_U0's (+0.95%).
- Orders about A_U0's (129 a year on T1): profitability ranks move only at filings.
- Luck >= 95th percentile; all four bear episodes beaten.
- C_prof: T1 ≈ −1 to +1, T2 ≈ +3 to +5.
- TL (orchestrator): +4 to +9%.
- I1 and I2: §0.7.

### §0.10 Failure modes to watch (the design's)

(a) PROF's VALIDATION record is thin (T1 −0.69%; T2's +4.97% rests on five names): it may
dilute T1 and add nothing real on T2.
(b) With three sleeves the extremes are moderate on each: the top 15 holds only 2.1-3.2 of SI's
top 15 (the H31 risk, larger than in A_U0).
(c) PROF's IC lives in the middle of the cross-section, so TRAIN may not move (PROF's own book
reads TB + T3 −0.03%).
(d) Fundamentals: 3-9% of gross-profit-measured names carry a level unchanged for more than 200
days; sector codes are today's; about a third of names are on the ROE fallback.
(e) The E0 blind spot: PROF's E0 periods (−10.9, −5.3, +6.5, +7.9) and MOM's E0 −8.2% are
invisible to G1.
(f) A blend (C_mix3) may beat the composite on P and drawdown while losing on alpha: a risk
choice for the user, not an adoption.
Plus, from the sibling's mechanics (§9 there): a squeeze cover can exceed the lock_cash margin
at gross 0.90 (GME on T2, 2024-05); the audit counts it.

### §0.11 In-sample caveats carried into every reading

- T1 is partly in-sample for SI: informed_flow's studies scored from 2020-07 and informed_flow_ls
  was developed on 2022-26 top-500 data. Only MOM is clean on T1.
- PROF's VALIDATION record was read before this design: quality_junk's I4 / I4v2 ran on every
  OPEN test in rounds 1-2, and the judge's 2026-H2 and drawdown evidence for PROF comes from
  those saved T1 / T2 books. Bar (3)'s mechanism legs (2026-H2, drawdowns) were therefore formed
  on T1 / T2 data: they test how the composite integrates PROF, not PROF out of sample.
- AH's TUNEDv2 was tuned on TB + T3; AH enters this book only as the execution helper
  `_on_priced_opens` (no AH signal).
- TL is fresh only for the combination: informed_flow_ls §38P already read names 1501-2000 for SI.
- G1 is read on TB + T3; the E0 periods of MOM and PROF are disclosed beside it (§0.6).

### §0.12 Amendments by the continuing builder (2026-10-24, before any number)

What the second builder changed in §0 and in the code, and why. No design element changed: U0
(B_U0), the controls (C_prof, C_mix3; A_U0, C_si and C_mom read-only), the five bars, the two
iterations (I1 `w_prof` 0.5; I2 / I2h, growth on the long side), the variant budget (3), the
adoption rule and the final rule are the first builder's, which follow the design.

1. **The reconciliation code was the pre-review copy.** `research.py` had been copied from
   xs_si_mom_composite at 13:51, before that design's review found (its §12) that the stuck-order
   estimate started each gap at the due bar's CLOSE (missing the due bar's own move, the second
   largest piece of GME's windfall) and re-estimated the gap's shares from the close, and that
   off-target name-days counted only targets not held (never a position held against a zero
   target). Replaced by the review-fixed `off_target`, `gap_pnl` and `_gaps`; the audit now lists
   every stuck order with its P&L from the due open; `stuck` takes a base for paired Δs, saves
   repaired rows, and reconciles the read-only sibling labels with their own code; every
   function refuses T4 and TL. Tests: the sibling's three reconciliation tests, ported.
2. **Engine vs a research preview** added to the audit (§0.5 step 1): `research.preview`, an
   independent vectorbt-free share simulation of the targets, with a test pinning it equal to a
   frictionless vectorbt book (retrim and a missing open included); the engine's gap to it must
   be its costs.
3. **The reconciliation runs on every open test** for every label of this design and, read-only,
   on A_U0, C_si and C_mom (§0.5 step 3a), and each bar is also shown repaired, beside the
   record (§0.6). The record decides.
4. **Run commands for the iterations** (§0.5 step 10) and the re-reading of bars (1)-(2) for an
   adopted iteration (§0.8), stated explicitly.
5. **Disclosures** (§0.4): what the continuing builder had read, including the sibling's review
   fixes (A_U0's T2 repaired +4.15%).
6. **README.md and IDEAS.md** did not exist; they are written with the results.

`strategy.py`, `config.yaml` and the core pin (`xs_si_mom_composite/strategy.py` c6061153…,
whose own component pins a05e84d4… / 5cc54c69… / 1867bc4d… / 52684a21… all match the current
files on 2026-10-24) are the first builder's, unchanged. Snapshot: `prereg/prereg.log`.

## §1 Mechanics audit (2026-10-24 00:22-00:26; `results2/xs_three_premia/logs/audit.log`)

`research.py audit B_U0,C_prof,C_mix3 TB,T3,T1,T2`, run after the §0 snapshot and before any
`lab.py run`: 81 period books, each rebuilt with vectorbt's order log and asserted equal to
`lab.book` to 1e-12, then compared with the frictionless research preview. No return level was
read.

| label | test | orders asked | refused | part-filled | NaN-price drops | off-target name-days (both ways) | engine − preview a year | first-day gap | days > 0.5% | max gross | max names long / short |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B_U0 | TB | 257 | 0 | 0 | 0 | 0 | −1.06% | −0.24% | 0 | 0.900 | 15 / 15 |
| B_U0 | T3 | 199 | 0 | 0 | 0 | 0 | −0.94% | −0.21% | 0 | 0.900 | 15 / 15 |
| B_U0 | T1 | 588 | 0 | 0 | 0 | 0 | −1.05% | −0.26% | 0 | 0.900 | 15 / 15 |
| B_U0 | T2 | 605 | 0 | 0 | 0 | 0 | −1.21% | −0.28% | 0 | **0.930** | 15 / **16** |
| C_prof | TB | 176 | 0 | 0 | 0 | 0 | −0.68% | −0.26% | 0 | 0.900 | 15 / 15 |
| C_prof | T3 | 150 | 0 | 0 | 0 | 0 | −0.71% | −0.20% | 0 | 0.900 | 15 / 15 |
| C_prof | T1 | 451 | 0 | 0 | 0 | 0 | −0.92% | −0.26% | 0 | **0.930** | 15 / **16** |
| C_prof | T2 | 494 | 0 | 0 | 0 | 0 | −1.00% | −0.25% | 0 | **0.930** | 15 / **16** |
| C_mix3 | TB | 729 | 0 | 0 | 0 | 0 | −0.89% | −0.22% | 0 | 0.880 | 44 / 41 |
| C_mix3 | T3 | 520 | 0 | 0 | 0 | 0 | −0.82% | −0.22% | 0 | 0.880 | 43 / 42 |
| C_mix3 | T1 | 1662 | 0 | 0 | 0 | 0 | −0.88% | −0.22% | 0 | **0.910** | 44 / 44 |
| C_mix3 | T2 | 1792 | 0 | 0 | 0 | 0 | −1.06% | −0.22% | 0 | 0.900 | 44 / 45 |

- **The §0.5 rule:** refused plus part-filled = 0.00% of the orders asked in every cell, so gross
  stays 0.90. Unlike A_U0 (GME on T2), no order of any label was refused or part-filled on any
  open test, so the reconciliation has no stuck position to price.
- **Engine vs the research preview:** the engine trails the vectorbt-free preview by 0.7-1.2%
  a year, which is its costs and borrow (each period's first day, the full rebuild plus the
  borrow charged up front, is −0.20 to −0.28%); no day after a period's first exceeds 0.5% in
  any cell. The engine holds the book the targets ask for.
- **Invariants:** no name on both sides, one weight per hold (rank labels), C_mix3 within its 45
  a side. Four cells hold one slot too many for some bars. Both causes are holes in the price
  data, not orders (`logs/frozen.py`, `frozen2.py`, `psky_gap.py`, positions only):
  - **PSKY, T2 2026-H2 (B_U0, C_prof):** the DB has no PSKY bars from 2026-09-15 to
    2026-10-02 (last close 10.82 on 09-14, first open 9.26 on 10-05). The NaN score exits the
    short on 09-15, `_on_priced_opens` keeps its −3% target until the next priced open, the slot
    is refilled (16 shorts, gross 0.93, 13 bars), and the cover fills at 9.26 on 10-05. The
    engine books the move across the gap, **+0.44% of the book** in that period (B_U0 −28.3
    shares, weight −3.03%): an upper bound on the artifact, since a real book would have covered
    at an unknown price on 09-16. About +0.1 pt a year of T2's stitched alpha.
  - **WBD, T1 2026-H2 (C_prof; C_mix3 at a third):** no bars after 2026-09-11 (close 28.04).
    Carried short at that frozen price for the window's last 24 bars (16 shorts, gross 0.93 /
    0.91): no P&L after the freeze. A real short would have been closed by whatever corporate
    action ended the listing. B_U0 does not hold WBD.
  The preview carries the same frozen positions, so the engine-preview gap cannot see them; the
  same mechanism appeared in xs_si_mom_composite's I1 (PSKY, T2).

## §2 Premise check (TRAIN only, informational; 00:31-00:32, `logs/premise.log`)

At the 33 TRAIN month-ends (TB 18, T3 15), each label's book score against the next 21 bars'
beta-adjusted return (p1's 126-bar beta × the eligible names' equal-weight return):

| score | IC (t) | TB IC (t) | T3 IC (t) | top 15 − bottom 15, per 21 bars (t) |
|---|---|---|---|---|
| B_U0 three-premia composite | +0.007 (+0.32) | +0.002 (+0.05) | +0.014 (+0.45) | −1.07% (−1.10) |
| C_prof | +0.030 (+1.72) | +0.033 (+1.84) | +0.026 (+0.81) | +0.07% (+0.07) |
| A_U0 (same code; the sibling's §2) | −0.017 (−0.75) | −0.025 (−0.71) | −0.006 (−0.26) | −0.03% (−0.03) |

- A_U0's row reproduces xs_si_mom_composite §2 to the printed digit (same code, same panels).
- PROF's rank IC is positive on these p1-eligible names (+0.030, t 1.72; quality_junk's own
  universe read +0.026, t 2.18 over E0 + TB + T3), but its extremes earn nothing (+0.07%, t
  0.07): failure mode (c), "PROF's IC lives in the middle of the cross-section", as the design
  warned.
- Adding PROF moves the composite's IC from −0.017 to +0.007, but its extremes from −0.03% to
  −1.07% (t −1.1): on TRAIN the three-premia book's top and bottom 15 did worse than the
  two-premia book's. Informational; G1 is read on the book.

## §3 U0 and the controls (runs 00:32-00:37, reconciliation 00:37-00:44, noise 00:44-00:48, luck 00:48-00:50, verdict / bear / report / bars 00:50-00:51)

`lab.py run … TB,T3,T1,T2 --costs2` (logs `logs/run_<label>.log`; `logs/chain1.status`). Alpha
against the equal-weight buy-and-hold of the same names; TRAIN = TB + T3 stitched. The
xs_si_mom_composite columns are read-only, from its saved books (same code and panels).

| | **B_U0** | C_prof | C_mix3 | A_U0 (read-only) | C_si (r-o) | C_mom (r-o) |
|---|---|---|---|---|---|---|
| TRAIN alpha (P) | **−1.17% (0.41)** | −0.78% (0.41) | +0.18% (0.51) | +1.97% (0.62) | +0.64% (0.52) | −0.25% (0.51) |
| · TB / T3 | −0.19 / −0.64 | +4.26 / −4.35 | +1.27 / −2.91 | +3.17 / −2.36 | −2.86 / +2.30 | +1.36 / −6.66 |
| · TRAIN late / 2× costs | −3.13 / −1.64 | −2.17 / −1.11 | −1.66 / −0.27 | −0.35 / +1.41 | −1.31 / +0.17 | −2.53 / −0.83 |
| T1 alpha (P) | **+6.06% (0.883)** | +2.56% (0.744) | +7.59% (0.995) | +12.88% (0.962) | +9.40% (0.964) | +9.04% (0.935) |
| · beta / vol / maxDD | −0.205 / 14.2% / −18.9% | −0.260 / 10.5% / −28.9% | −0.077 / 8.3% / −7.6% | +0.009 / 16.3% / −21.9% | +0.136 / 11.3% / −19.3% | −0.112 / 16.0% / −18.3% |
| · late / 2× / top-5-out | +4.14 / +5.60 / +1.74 | +1.14 / +2.22 / −1.63 | +5.75 / +7.13 / +4.90 | +10.77 / +12.37 / +8.04 | +7.31 / +8.93 / +2.71 | +6.66 / +8.47 / +4.82 |
| · orders a year | 123 | 94 | 347 | 129 | 121 | 148 |
| T2 alpha (P) | **+3.63% (0.785)** | +1.45% (0.699) | +2.24% (0.835) | +5.07% (0.843); repaired +4.15% (0.78) | +2.53% (0.744) | +1.86% (0.646) |
| · beta / maxDD | −0.119 / −16.0% | −0.155 / −14.2% | −0.079 / −9.2% | −0.088 / −15.6%; repaired −19.1% | +0.030 / −12.3% | −0.105 / −22.9% |
| alpha ÷ residual vol: TRAIN / T1 / T2 | −0.11 / 0.44 / 0.34 | −0.13 / 0.28 / 0.18 | 0.03 / 0.93 / 0.31 | 0.16 / 0.79 / 0.38 | 0.09 / 0.85 / 0.25 | −0.01 / 0.57 / 0.13 |
| 2026-H2 period alpha, T1 / T2 (a year) | **+3.7 / +3.9** | +16.5 / +42.0 | −20.2 / −1.4 | −22.2 / −7.9 | −36.7 / −8.2 | −33.1 / −29.7 |
| down months (n 38, market −4.12%) | **+1.25%** (66% > 0) | +0.57% (61%) | +0.61% (71%) | +0.95% (61%) | −0.02% (53%) | +1.31% (74%) |
| Q4 2018 (market −18.2%) | −3.5% | +1.6% | −3.4% | −7.0% | −8.7% | −3.3% |
| COVID crash (−37.0%) | −0.3% | +5.1% | +3.8% | −0.1% | −1.0% | +6.7% |
| 2022 bear (−24.5%) | +13.7% | +14.5% | +13.1% | +15.4% | +10.0% | +13.8% |
| Feb-Apr 2025 (−19.5%) | +1.6% | +7.5% | +2.7% | +1.7% | −3.3% | +4.2% |
| CAGR / max DD, 2018-07 → 2026-10 (TB + T3 + T1) | 2.1% / −19.4% | −0.6% / −28.9% | 4.0% / −10.9% | 8.9% / −21.9% | 7.0% / −19.3% | 4.2% / −27.9% |

B_U0 per period (annualized alpha): TB −5.1, −0.7, −7.4, +3.9; T3 +12.8, −14.3, +7.2; T1 +16.0,
+19.5, −1.1, +1.7, +36.9, +3.6, +23.0, −8.4, −10.0, +3.7; T2 −6.7, +7.4, −9.0, +12.1, +30.7, −2.4,
+4.7, +27.8, −14.5, +3.9. Calendar years (book / equal-weight market): 2018-H2 −4.0 / −6.2, 2019
+2.6 / +35.6, 2020 −0.2 / +25.7, 2021 −0.8 / +21.5, 2022 +18.7 / −18.6, 2023 −16.4 / +22.7, 2024
+16.5 / +15.3, 2025 +4.3 / +16.2, 2026 to Oct 16 +1.3 / +11.7.

**Reconciliation** (`logs/stuck_U0_controls.log`, `logs/stuck_sibling.log`): no order of B_U0,
C_prof or C_mix3 was refused or part-filled on any open test, so their repaired figures equal
the record. The read-only rebuild of the sibling's books reproduces its §12 to the last digit:
A_U0's T2 holds GME's part-filled cover (+4.57% of the book, +1.72% on the due bar) and NTLA's
refused short (−0.38%), so A_U0 T2 reads +4.15% (P 0.78) repaired; its T2 max drawdown,
−15.6% as run, is −19.1% repaired (the stuck short's windfall had also cushioned it). C_si's
MGY part-fill: −0.02%.

**Noise** (`lab.noise`, 5 names dropped a period, 3 copies; copy − label alpha): TB −0.30,
+1.00, +0.23 (sd 0.66); T3 −2.58, +1.82, −0.47 (sd 2.20); **T1 +0.47, +0.57, −0.55 (sd 0.62)**;
T2 +0.35, −0.90, +0.53 (sd 0.78); stitched TRAIN sd 1.53. Unlike A_U0 (T1 sd 1.45, all copies
below the label), B_U0's T1 path is not a favourable draw: its copies centre on it.

**Luck** (T1, 100 books with the names shuffled inside dollar-volume tenths): real +6.06%, median
+0.21%, p95 +3.94%, **percentile 98**.

**Verdict B_U0** (`logs/verdict_B_U0.log`, market-neutral): **G1 −1.17% ✗** · **G2 +6.06%, P
0.883 ✗** · G3 +3.63% ✓ · G4 98th ✓ · G5 late +4.14%, 2× +5.60% ✓ · G6 all four episodes, down
months +1.25% ✓ · G7 beta −0.205 ✓ · G8 +1.74% ✓. **Six of eight gates: not "works", and not
"works to some degree"** (which needs G1 and G2). Beside G1, invisible to it: MOM's E0 −8.2% and
PROF's E0 periods (−10.9, −5.3, +6.5, +7.9 in quality_junk's machinery).

**Context** (paired, read-only, decide nothing; `logs/delta_U0.log`):

| B_U0 − … | TRAIN | T1 | T2 |
|---|---|---|---|
| xs_si_mom_composite/C_mix (SI + MOM books, ½ each) | −1.74 (P 0.33) | −3.62 (0.23) | +1.27 (0.66) |
| xs_si_mom_composite/I1 (A_U0 + PROF short gate = design idea 2) | −4.47 (0.11) | −7.77 (0.07) | −4.61 (0.10) |
| informed_flow_ls_p1/v6_si | −2.22 (0.39) | −4.35 (0.26) | −0.32 (0.49) |
| resid_momentum/U0v2 | −3.85 (0.17) | −2.67 (0.28) | −1.87 (0.36) |
| C_prof − quality_junk/I4v2 (PROF in its own machinery) | −0.75 (0.41) | +3.25 (0.88) | −3.52 (0.08) |
| C_mix3 − xs_si_mom_composite/C_mix | −0.40 (0.43) | −2.09 (0.24) | −0.12 (0.49) |

### §3.1 The pre-registered bars (B_U0; `logs/bars_B_U0.log`, `logs/bars_B_U0_repaired.log`)

| bar | reading (record) | repaired reading | result |
|---|---|---|---|
| (1) works: all eight gates | G1 −1.17%, G2 P 0.883 fail | same (nothing of B_U0's was stuck) | **fail** |
| (2) Δ vs C_mix3 ≥ 0 on TRAIN, T1, T2; T1 Δ > 2 × 0.62 = 1.24 | TRAIN −1.34 (P 0.35, 2/7), T1 **−1.53** (P 0.35, 7/10), T2 +1.39 (P 0.67, 5/10) | same | **fail** |
| (3) Δ vs A_U0 ≥ 0 on TRAIN and T2, ≥ −1.0 on T1 | TRAIN −3.13 (P 0.22), T1 **−6.82** (P 0.12), T2 −1.44 (P 0.38) | T2 −0.53 | **fail** |
| (3) mechanism: 2026-H2 alpha above A_U0's on T1 and T2 | T1 +3.7 vs −22.2; T2 +3.9 vs −7.9 | same | holds |
| (3) mechanism: max drawdown smaller than A_U0's on T1 and T2 | T1 −18.9 vs −21.9 ✓; T2 −16.0 vs **−15.6** ✗ | T2 −16.0 vs −19.1 ✓ | fails as run (fragile leg) |
| (4) G6 and down months ≥ A_U0's | G6 ✓; +1.25% vs +0.95% | same | **pass** |
| (5) ≥ max(C_si, C_mom, C_prof) on TRAIN and T2 | TRAIN −1.17 vs C_si +0.64 ✗; T2 3.63 vs C_si 2.53 ✓ | same | **fail** |

No bar's outcome differs between the record and the repaired reading; only the T2 drawdown leg
of bar (3) does, and bar (3) fails on its alpha legs either way.

**Reading.** The design's mechanism did what it said in the crash it was chosen for: with PROF
in the ranking, 2026-H2 turned from A_U0's −22.2% / −7.9% a year into +3.7% / +3.9%, the T1
drawdown shrank from −21.9% to −18.9%, and the falling-market profile improved (down months
+1.25%, Q4 2018 −3.5% vs −7.0%). But the price of that hedge was most of the alpha the two-premia
book earned everywhere else: −6.8 pts on T1 (10 of whose periods B_U0 wins only 5), −3.1 on
TRAIN, −1.4 on T2. Failure modes (a) and (b) both read: PROF's validation record is thin (C_prof
T1 +2.56%, T2 +1.45%, T3 −4.35%), and a third equal rank pulls the book's extremes toward the
middle of the cross-section where PROF's IC lives (§2: the composite's TRAIN extremes −1.07% per
21 bars against A_U0's −0.03%). Integration also loses to holding the three books side by side
(C_mix3), which keeps more of the T1 alpha (+7.59%, P 0.995) at half the drawdown (−7.6%); failure
mode (f) says that is a risk choice, not an adoption. The net short beta (−0.21 on T1, outside
the −0.15 to +0.10 expected) comes from PROF: equal slots on profitable longs and unprofitable
shorts tilt short beta here (C_prof −0.26), unlike quality_junk's beta-sized machinery
(|beta| ≤ 0.12 there).

## §4 Iteration I1, PROF at half weight (00:48-00:54; `logs/audit_I1.log`, `run_I1.log`, `adopt_I1.log`)

Pre-registered in §0.7 (`{"w_prof": 0.5}`, run after the U0 read whatever it showed). It was
launched at 00:48, after B_U0's, C_prof's and C_mix3's runs, the reconciliation and B_U0's noise
copies had been read, while B_U0's luck run was still going (a second process, at 51% free
memory). Its audit came first: nothing refused, part-filled or off target on any open test, invariants intact,
the engine within its costs of the preview (`audit_I1.log`, read before the run).

| I1 − B_U0 | TB | T3 | T1 | T2 | TRAIN |
|---|---|---|---|---|---|
| Δ (P) | −1.97 (0.20) | −3.37 (0.10) | **+4.04 (0.93)** | +2.19 (0.81) | **−1.91 (0.14)** |
| periods better | 1/4 | 0/3 | 7/10 | 5/10 | 1/7 |

I1 alone: TRAIN −3.08% (P 0.28), T1 +10.10% (P 0.960, beta −0.13, vol 15.0%, max DD −19.4%, late
+8.08%, 2× +9.61%, top-5-out +5.58%), T2 +5.81% (P 0.909, max DD −12.4%); 2026-H2 T1 −22.2%, T2
+11.4% a year; down months +1.02%; 128 orders a year on T1.
**Rule:** T1 Δ +4.04 > 2 × 0.62 = 1.24 ✓ and T2 Δ ≥ 0 ✓, but TRAIN Δ −1.91 < 0. **Not adopted.**
Expected: between B_U0 and A_U0 on every test, T1 Δ 0 to +1.5. Observed: T1 and T2 between them
(T1 Δ larger than expected), TRAIN below BOTH (−3.08% against −1.17% and +1.97%), and at half
weight the 2026-H2 hedge is gone on T1 (−22.2%, A_U0's figure). Context (read-only): I1 − A_U0
TRAIN −5.05 (P 0.08), T1 −2.78 (0.31), T2 +0.74 (0.57); I1 − C_mix3 TRAIN −3.26, T1 +2.51, T2
+3.57 (`logs/delta_iterations.log`). Base stays B_U0. Variants used: 2 of 3.

## §5 Iteration I2, profitability growth on the long side (00:54-01:01; `logs/audit_I2.log`, `run_I2.log`, `adopt_I2.log`)

I1 was rejected, so by §0.7 the label is `I2 = {"w_growth": 1.0}` (long composite (si + mom +
prof + growth) / 4, short composite unchanged). Its audit came first and is clean on all four
tests (T2's one extra short for 13 bars is B_U0's PSKY price gap, §1).

| I2 − B_U0 | TB | T3 | T1 | T2 | TRAIN |
|---|---|---|---|---|---|
| Δ (P) | −0.10 (0.49) | −7.09 (0.001) | +2.33 (0.83) | **−0.18 (0.45)** | **−3.63 (0.004)** |
| periods better | 3/4 | 0/3 | 5/10 | 7/10 | 3/7 |

I2 alone: TRAIN −4.79% (P 0.15), T1 +8.39% (P 0.939, beta −0.19, max DD −20.8%), T2 +3.45%
(P 0.80); 2026-H2 T1 −11.3%, T2 −5.8% a year; down months +0.91%.
**Rule:** TRAIN −3.63 and T2 −0.18 < 0. **Not adopted**, as expected (TRAIN Δ ≤ 0; T1 and T2
within ±1.5 — T1's +2.33 is a little outside). quality_junk's wrong-sign growth premise
(IC −0.010 on TRAIN) showed up as a T3 loss (−7.1 pts). Variants used: **3 of 3** (B_U0, I1,
I2); the budget is spent.

## §6 The final (declared 2026-10-24 01:02 CEST, before the final verdict and bear re-run, and before any T4 or TL read — neither is read by this agent)

By the §0.8 rule: no iteration was adopted, so the base is **B_U0** (`'{}'`, config.yaml as
shipped). B_U0 fails bar (1) (G1 −1.17%; G2 P 0.883), so it **stays the declared final, reported
as failing**; the switch to C_mix3 applies only to a base that passes bar (1) and fails bar (2).
C_mix3's numbers are reported beside it as information (failure mode (f): a risk choice for the
user, not an adoption). **Declared final: `xs_three_premia/B_U0`, overrides `{}`, lab flags
`--costs2` (no `--hedge`).** By the lab's definitions it neither "works" (6 of 8 gates) nor
"works to some degree" (G1 and G2 fail). Its luck run exists (98th percentile); its verdict and
bear are re-run below for the record. For the orchestrator's single TL read: names 1501-2000
are more squeeze-prone; this design's research.py refuses locked tests, so the mechanics audit
of TL (refused and part-filled orders, off-target name-days both ways, stuck P&L from the due
open) is the orchestrator's step.

**Final verdict and bear, re-run after the declaration** (`logs/verdict_final_B_U0.log`,
`logs/bear_final_B_U0.log`, 01:02-01:03): identical to §3. G1 −1.17% ✗, G2 +6.06% P 0.883 ✗,
G3-G8 ✓ (luck 98th from the cache); Q4 2018 −3.5%, COVID −0.3%, 2022 +13.7%, Feb-Apr 2025 +1.6%;
CAGR 2.1%, max DD −19.4% over 2018-07 → 2026-10.

## §7 Post hoc, after the declaration: information only, decides nothing (01:03-01:09)

1. **Overlap and sector exposure** (`research.py overlap`, positions only; `logs/overlap_B_U0.log`).
   Mean names held on the same side, of B_U0's 15 a side:

   | | TB | T3 | T1 | T2 |
   |---|---|---|---|---|
   | longs / shorts also in A_U0 | 6.5 / 5.9 | 4.7 / 7.6 | 5.6 / 5.0 | 5.0 / 4.9 |
   | … in C_si | 2.3 / 3.3 | 2.1 / 2.9 | 1.9 / 1.6 | 1.8 / 1.7 |
   | … in C_mom | 1.8 / 2.4 | 2.7 / 2.9 | 3.6 / 2.7 | 3.2 / 2.5 |
   | … in C_prof | 3.6 / 4.5 | 2.9 / 3.7 | 2.5 / 3.9 | 3.0 / 3.7 |
   | Σ|net weight per sector| (A_U0) | 0.27 (0.38) | 0.39 (0.42) | 0.51 (0.48) | 0.48 (0.50) |
   | largest sector (A_U0) | 7.6% (9.7%) | 11.1% (9.7%) | 16.4% (12.5%) | 13.3% (12.6%) |

   Failure mode (b), measured: on T1 and T2 the three-premia book holds only 1.6-1.9 of SI's own
   15 names a side (the design expected 2.1-3.2) and two-thirds of its names are not in A_U0. A
   within-sector PROF rank did not make the book less of a sector bet: on T1 its net sector
   exposure (0.51) and largest sector (16.4%) are above A_U0's.
2. **C_mix3's gates** (`lab.py luck … C_mix3 T1 100`, `verdict`; `logs/luck_C_mix3_posthoc.log`,
   `logs/verdict_C_mix3_posthoc.log`): the control that holds the three single-sleeve books side
   by side passes all eight: G1 +0.18% (P 0.51; TB +1.27, T3 −2.91) · G2 +7.59%, P 0.995 · G3
   +2.24% · G4 100th (median +0.35%, p95 +2.86%) · G5 late +5.75%, 2× +7.13% · G6 every episode,
   down months +0.61% · G7 beta −0.077 · G8 +4.90%. T1 volatility 8.3%, max DD −7.6%; 2026-H2 T1
   −20.2%. **It is not this design's final**: §0.8 switches to C_mix3 only when the base passes
   bar (1) and fails bar (2), and B_U0 failed bar (1); its luck and verdict were read after the
   declaration. Its G1 is hair-thin (P 0.51, T3 −2.91, TRAIN late fills −1.66%), and against the
   two-book C_mix of xs_si_mom_composite it gives up 2.1 pts of T1 alpha (P 0.24) for half the T1
   drawdown (−7.6% vs −14.8%). Reported because the design's failure mode (f) named it — a risk
   choice for the user, not an adoption — and because the orchestrator's cross-design ranking
   may want it as information.

## §8 Conclusion (B_U0, lab v2; T4 and TL not read)

**Answer to the design's question: no. A third premium, within-sector profitability, made the
composite more robust in the 2026-H2 crash, but it cost most of the composite's alpha
everywhere else, and the book fails G1 and G2.**

- **Bar (1), works: fail** (6 of 8 gates). TRAIN −1.17% (P 0.41; TB −0.19, T3 −0.64), T1 +6.06%
  (P 0.883: under the 0.90 bar), T2 +3.63% (P 0.79). G3-G8 pass: luck 98th percentile, late fills
  +4.14%, 2× costs +5.60%, every bear episode beaten, down months +1.25%, beta −0.21, top-5-out
  +1.74%. Not "works to some degree" either (G1 and G2 fail).
- **Bar (2), integration over C_mix3: fail.** TRAIN −1.34 (P 0.35), T1 −1.53 (P 0.35), T2 +1.39
  (P 0.67). Holding the three books side by side kept more of the alpha at half the risk.
- **Bar (3), the third premium earns its place over A_U0: fail.** TRAIN −3.13, T1 −6.82, T2 −1.44
  (−0.53 with A_U0's GME windfall removed). The mechanism's own legs mostly held: 2026-H2 +3.7% /
  +3.9% a year against A_U0's −22.2% / −7.9%, T1 max DD −18.9% vs −21.9%; T2's max DD −16.0% vs
  −15.6% fails as run and passes against A_U0's repaired −19.1%.
- **Bar (4), falling markets: pass.** G6, down months +1.25% vs A_U0's +0.95%.
- **Bar (5), best single sleeve: fail** on TRAIN (−1.17 vs C_si +0.64); T2 passes (3.63 vs 2.53).
- **Iterations:** I1 (PROF at half weight) +4.0 on T1 and +2.2 on T2 but −1.9 on TRAIN; I2
  (profitability growth on the long side) −3.6 on TRAIN. Neither adopted; variants 3 of 3.
- **Mechanics:** nothing refused, part-filled or off target in any of the 135 audited period
  books of this design (B_U0, C_prof, C_mix3, I1, I2 × the 27 periods of TB, T3, T1, T2), and the
  engine tracks an independent frictionless preview within its costs. Two price-data holes
  (PSKY's 13-bar gap on T2, +0.44% of the book in one period; WBD's end of data on T1, C_prof
  only) were carried frozen, which breaks the 15-a-side invariant by one slot for 13-24 bars.

**In-sample caveats.** T1 is partly in-sample for SI (informed_flow's studies scored from
2020-07; informed_flow_ls was developed on 2022-26 top-500 data); only MOM is clean there.
PROF's validation record (quality_junk I4 / I4v2) and the judge's 2026-H2 and drawdown evidence
for it were read on T1 / T2 before this design, so bar (3)'s mechanism legs test integration,
not PROF out of sample. AH's TUNEDv2 was tuned on TB + T3, but AH enters only as the execution
helper `_on_priced_opens`. G1 reads TB + T3 only; beside it, invisible to it: MOM's E0 −8.2% and
PROF's E0 periods (−10.9, −5.3, +6.5, +7.9). TL is fresh only for the combination:
informed_flow_ls §38P already read names 1501-2000 for SI.

**Declared final (§6): `xs_three_premia/B_U0`, overrides `{}`, `--costs2`, no `--hedge` —
failing bar (1).**

## §9 Expected (§0.9) against observed

| item | expected | observed |
|---|---|---|
| TRAIN (TB + T3) | +1 to +4% (P 0.6-0.85) | **−1.17% (P 0.41)** |
| T1 | +7 to +11%; −1.5 to +0.5 against A_U0 | **+6.06% (P 0.883); −6.82 against A_U0** |
| T2 | +5 to +8%; 0 to +2 against A_U0 | **+3.63%; −1.44 against A_U0** (−0.53 repaired) |
| beta | −0.15 to +0.10 | **−0.21 (T1)**, −0.12 (T2) |
| max DD and 2026-H2 loss below A_U0's | both | 2026-H2 ✓ (+3.7 / +3.9 vs −22.2 / −7.9); max DD T1 ✓ (−18.9 vs −21.9), T2 ✗ as run (−16.0 vs −15.6) |
| down months ≥ A_U0's | yes | +1.25% vs +0.95% ✓ |
| orders ≈ A_U0's | ≈ 129 a year on T1 | 123 ✓ |
| luck | ≥ 95th | 98th ✓ |
| bear episodes | all four beaten | all four ✓ |
| C_prof | T1 ≈ −1 to +1, T2 ≈ +3 to +5 | T1 +2.56%, T2 +1.45% |
| I1 | between B_U0 and A_U0; T1 Δ 0 to +1.5 | T1 Δ +4.04, T2 +2.19, TRAIN −1.91 (below both) |
| I2 | TRAIN Δ ≤ 0; T1, T2 within ±1.5 | TRAIN −3.63, T1 +2.33, T2 −0.18 |
| TL | +4 to +9% | not read here (orchestrator) |
