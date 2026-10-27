# xs_cross_leg — research log

Cross-strategy round of the proposals lab (`strategies/proposals_lab`, lab v2). The design is
`xs_design.json` → `final.selected[key = xs_cross_leg]`, chosen by a design panel and a judge.
Results are written only to `var/proposals_lab/results2/xs_cross_leg/` (logs in its `logs/`,
pre-registration snapshots in its `prereg/`).

## §0 Pre-registration (2026-10-24 00:25 CEST, written before any number of this design)

### §0.0 Provenance: a continuation

A first builder of this design was interrupted (session limit) at about 13:50 on 2026-10-23,
before any number of this design existed: no `results2/xs_cross_leg` pickle, empty `logs/` and
`prereg/`, no RESEARCH_LOG. It left `strategy.py`, `config.yaml`, `research.py` and
`tests/test_xs_cross_leg.py`. This section is written by the continuation builder on 2026-10-24,
again before any number of this design. What the continuation changed before this snapshot:

- `strategy.py` and `config.yaml`: inspected against the spec and kept unchanged (the wrapper,
  its SHA-256 pin of the core and every value match the spec).
- `research.py`: rewritten. The inherited version predated xs_si_mom_composite's review fixes
  (its RESEARCH_LOG §12): off-target name-days counted only targets not held (a position held
  against a zero target, a stuck exit, was invisible), stuck orders had no P&L estimate, and
  locked tests were not refused. The composite's fixed `rebuild`, `off_target`, `gap_pnl` and
  `_gaps` are ported; `stuck` (alpha with the stuck P&L removed, paired both ways) is added; the
  audit runs on all four open tests; the research preview is now a frictionless, drift-aware
  replay of the targets outside vectorbt (tested equal to `lab.run_portfolio` without costs, to
  1e-12, retrim cap included).
- Tests: the inherited spec tests are kept; added: the research helpers on a toy squeezed book
  (a part-filled cover and a refused short), the preview against the lab runner, locked-test
  refusal, and this config equal to the composite's except the two sleeve switches.

Numbers seen before writing this section: the components' published lab-v2 results
(`xs_digest.json`), the design's spec and the judge's previews, and xs_si_mom_composite's
RESEARCH_LOG (its A_U0, C_si, C_mom and C_mix results, which are read-only controls of this
design). No number of `C_U0`, `C_rev` or any other label of this design.

### §0.1 Question and hypothesis

**Question.** Does each signal belong on one side of the book: short-interest (SI) longs and
residual-momentum (MOM) shorts? The reverse cross (MOM longs, SI shorts) is the built-in
falsification.

**Hypothesis (the design's mechanism and evidence).** Each signal trades the side where its
documented edge and its beta sit.
- SI's long rule is its best-replicated part: as adaptive_hedge's long sleeve (~0.34 gross) its
  leg alpha is +5.2 (T1) and +6.2 (T2); the long side alone (adaptive_hedge L0v2) reads T2
  +14.2% (P 0.997) and T1 +8.4%.
- MOM's short leg is MOM's most significant leg on T1: +5.1% (P 0.99, lab v1 legs), positive in
  both TRAIN tests (TB +2.5, T3 +1.4).
- The betas match across the cross: SI's long tail 1.04 / 1.30 (TB / T3), MOM's short tail
  1.21 / 1.11, so the ex-ante net beta is about −0.08 / +0.09 at dollar neutrality (SI's own book
  +0.05 / +0.21, up to +0.54 ex ante in 2026-H2; MOM's own −0.16 / −0.03). Crossing removes both
  mismatches without beta sizing, which cost MOM −2.1 and SI −1.1.
- No conflicts: SI's long 15 and MOM's short 15 share about 1 name.
- The reverse cross (MOM longs at beta 0.86 / 1.05 with SI's quiet shorts at 0.94 / 0.84) is
  just as beta-matched. If the leg evidence is real, the primary must beat it.

### §0.2 U0 = `C_U0` (config.yaml as shipped, overrides `{}`; nothing tuned)

`strategy.py` is a thin wrapper: it loads `strategies/xs_si_mom_composite/strategy.py` by path
as `_xsl_core` (SHA-256 pinned) and re-exports its `signals` / `valid`; config.yaml sets
`long_sleeves: si`, `short_sleeves: mom`. Every other value is the composite's (test-pinned).

1. Eligible `ok` = informed_flow_ls_p1's: close >= $5, a days-to-cover and `fund_shares` > 0.
2. SI rank = `p1.compose(panel, dtc_window=42, skew_window=20, w_si=1, w_skew=0,
   skew_smooth=1, score_smooth=1, neutral_groups=5, short_max_si=0.30, si_mid=0.10,
   si_high=0.20, borrow_gc=0.005, borrow_mid=0.03, borrow_high=0.10, min_price=5,
   beta_window=126)['ranks']['si']` (minus days-to-cover averaged over 42 bars, percentile in 5
   dollar-volume groups, silent below 50% coverage).
3. MOM rank = `rm.compose(panel, min_price=5)['rank']` on its month-end rows (resid_momentum
   U0v2: `Σe / sd(e)` over bars t−251 .. t−21, `e = r − β·m`), forward-filled to every bar until
   the next month end, then `.where(ok)`.
4. Each side's score = its one sleeve's rank `.fillna(0.5)`, `.where(ok)`, averaged over 5 bars
   (`min_periods=1`), `.where(ok)`, re-ranked (`pct=True`): the long score from SI, the short
   score from MOM.
5. **Long leg:** `longs = p1._select(long_score, close, shortable, q_in 0.90, q_out 0.80, slots
   15, max_hold 126, short_stop 1.0, sides 'long')[0]`: enter at SI score >= 0.90, leave below
   0.80; the 126-bar clock restarts while the name stays in its entry zone.
6. **Short leg:** `shorts = p1._select(short_score.mask(longs), close, shortable, …, sides
   'short')[1]`: enter at MOM score <= 0.10, leave above 0.20; a name held long at the deciding
   bar is NaN in the short score, so it is never a short candidate and a held short that becomes
   a long exits on that bar; squeeze stop at +100% from the entry close, re-entry only after the
   score has left the short zone once; `shortable` = short interest / shares <= 0.30 (never
   short, or keep short, above it).
7. Sizing `p1._weights(longs, shorts, beta, gross 0.90, long_share 0.5, slots 15, sizing
   'equal')`: ±3% a slot, constant while held, decided at the close, filled at the next open.
8. `ah._on_priced_opens(weights, panel.open_)`: a target change on a bar without an open waits
   for the name's next priced open.
9. Output `{weights, borrow (p1 tiers 0.5% / 3% / 10% a year by short interest / shares),
   retrim_k 2.0}`; `fill: open`, `order_notional: 1000`; `data.features: [short_interest,
   fund_shares]`; optuna `objectives: [alpha_cagr]`, `n_trials: 1`, nothing searched.

### §0.3 Controls (they decide nothing alone and are not counted as variants)

| label | where | overrides | role |
|---|---|---|---|
| `C_rev` | run here | `{"long_sleeves": "mom", "short_sleeves": "si"}` | the reverse cross: the falsification |
| `C_si` | xs_si_mom_composite (read-only) | `{"long_sleeves": "si", "short_sleeves": "si"}` | SI alone in this machinery |
| `C_mom` | xs_si_mom_composite (read-only) | `{"long_sleeves": "mom", "short_sleeves": "mom"}` | MOM alone in this machinery |
| `A_U0` | xs_si_mom_composite (read-only) | `{}` there (`si+mom` both sides) | integration (bar 5, context) |

C_si, C_mom and A_U0 come from the same core `strategy.py` (pinned below) and the same lab
panels; their saved results are lab v2 with `--costs2`, no `--hedge`. C_rev is never adopted
after the fact: pursuing it would cost a new pre-registered variant. Context only (read-only,
paired on the same days): `informed_flow_ls_p1/v6_si`, `resid_momentum/U0v2`.

### §0.4 Code pins (SHA-256 of the current files; none was changed by a review fix)

| file | SHA-256 |
|---|---|
| xs_si_mom_composite/strategy.py (the core, `_xsl_core`) | c6061153760a6caffe65766c68a91bb9e258e134bd32953b75ec6a6a3341ef65 |
| informed_flow_ls_p1/strategy.py (`_xsc_p1`, pinned by the core) | a05e84d4176ef60bcf68c20eca41c4a6cb113e78d17089f37a3f3583b88dee67 |
| resid_momentum/strategy.py (`_xsc_rm`) | 5cc54c69620cab771e1a268f2900ba0137a107220889794abb3ff6cc3bc52753 |
| quality_junk/strategy.py (`_xsc_qj`, loaded only for a `prof` sleeve) | 1867bc4de9bf56f66e468bf19008b0f0bf93ae0417fdb7574b1d549c8e48fdc6 |
| adaptive_hedge/strategy.py (`_xsc_ah`) | 52684a2118b5f6779fe3a0ddbd39c230fde2f1ee1a16cfd27251c013941b936c |

xs_si_mom_composite's review (its §12) changed only its `research.py` and tests; its
`strategy.py` is the file its labels were run with. The adaptive_hedge round-2 repair left its
`strategy.py` at the pinned hash. The shared `informed_flow_ls` (under review) is never edited,
imported or loaded. The components are never edited; a changed file refuses to load.

### §0.5 Tests, runs and order (lab v2; `uv run python strategies/proposals_lab/lab.py …`)

TRAIN = TB + T3 (E0 has no short interest: SI data start 2018-01). VALIDATION = T1, T2. T4 and
TL are LOCKED and are not read by this agent: the orchestrator reads the declared finals' TL
once, in one batch.

1. **Execution audit first** (`research.py audit C_U0,C_rev TB,T3,T1,T2`), before any lab run.
   No return level is read. Per label × test:
   - every period rebuilt with vectorbt's order log on lab v2's runner, asserted equal to
     `lab.book` (1e-12);
   - orders asked, refused, part-filled (< 99.5% of the shares asked), dropped at a NaN price,
     by kind (long buy, long sale, new short, cover); off-target name-days both ways (a target
     not held, and a position held against a zero target); the mean gross never held; every
     stuck order's gap P&L from its due open;
   - book invariants on the weights: no name on both sides, <= 15 names a side, Σ|w| <= 0.90 +
     1e-9, one constant weight per hold; flips (short → long and long → short in one order);
   - **engine vs research preview**: the frictionless replay of the targets (`research.preview`)
     against the engine's daily returns: correlation, tracking error, yearly drag, and the five
     largest daily gaps with the orders that day.

   **Execution audit rule** (the composite's §0.5 rule, read here on all four open tests, not
   only TB and T1): if refused plus part-filled orders exceed 1% of the orders asked on ANY
   open test for C_U0 or C_rev, `gross` becomes 0.85 in config.yaml for every label before any
   label is run, and the audit is re-run. **Preview rule:** expected per test: correlation >=
   0.98, tracking error <= 2%/yr, drag between −4% and 0% a year; any daily gap above 0.5% of
   the book that is not day 0 (opening orders and up-front borrow) and not a logged failed
   order is a defect, investigated (and fixed if it is one) before any lab run.
2. `run xs_cross_leg C_U0 '{}' TB,T3,T1,T2 --costs2` and `run xs_cross_leg C_rev
   '{"long_sleeves":"mom","short_sleeves":"si"}' TB,T3,T1,T2 --costs2` (no `--hedge`: SPY is not
   used). Logs `logs/run_<label>.log`.
3. **The reconciliation on every open test** (`research.py stuck C_U0,C_rev TB,T3,T1,T2`):
   every refused or part-filled order, how long it stayed off target and its P&L from the due
   open, each test's alpha with the stuck P&L removed, and bar (2)'s Δ as run and repaired. The
   lab number is the record; a repaired figure that would flip a bar or a decision is reported
   as a caveat on it, not used to re-decide it.
4. `noise xs_cross_leg C_U0 5 3` (path noise; its T1 sd is the adoption bar's unit).
5. `luck xs_cross_leg C_U0 T1 100`.
6. `verdict xs_cross_leg C_U0` (market-neutral: G7 applies).
7. `bear xs_cross_leg C_U0 C_rev`; `report xs_cross_leg C_rev C_U0`.
8. `research.py betas C_U0,C_rev` (per-period ex-ante leg and book betas, realized beta) and
   `research.py legs` for C_U0, C_rev and the composite's C_si and C_mom (each leg's
   contribution, beta and alpha; adaptive_hedge/legs.py's pattern); `research.py overlap`.
9. `research.py bars C_U0` (bars 2-5) and `research.py delta` against the composite's A_U0,
   C_si, C_mom (read-only) and the context books.
10. The iterations (§0.7), each with the audit (all four open tests) before its numbers, then
    its run, its reconciliation and the adoption rule.
11. The final, declared in this log (§0.8), then `verdict` and `bear` on it (and its own T1 luck
    run when it is not C_U0). TL is read by the orchestrator, once.

Machine rules: one heavy process at a time (two only with `memory_pressure` >= 35% free), long
jobs in the background with logs under `var/proposals_lab/results2/xs_cross_leg/`.

### §0.6 Pre-registered bars (Δ = paired alpha difference on the same days and bootstrap blocks;
TRAIN Δ on the stitched TB + T3 record)

1. **Works:** all applicable gates G1-G8 (G1 read on TB + T3; G7 applies).
2. **Falsification:** Δ(C_U0 − C_rev) > 0 on TRAIN, T1 and T2, with P(Δ > 0) >= 0.80 on T1 and
   on T2. Otherwise leg specialization is rejected, whatever C_U0's own alpha.
3. **Mechanism check, reported:** the realized per-period beta (the period's daily book returns
   on the equal-weight buy-and-hold) is within ±0.15 in at least 8 of the 10 T1 periods. Leg
   contributions are reported on the adaptive_hedge/legs.py pattern.
4. **Best single sleeve:** alpha >= max(C_si, C_mom) on TRAIN and on T2 (C_si +0.64 / +2.53,
   C_mom −0.25 / +1.86, published).
5. **Specialization against integration** (context; xs_si_mom_composite/A_U0, read-only): Δ >= 0
   on TRAIN and T2, and >= −1.0 pt on T1. Read against A_U0 as run; A_U0's T2 with its
   2024-05-14 stuck orders repaired (+4.15% instead of +5.07%, the composite's §12.2) is
   reported beside it.

### §0.7 Iterations (variant budget 3: C_U0, I1, I2; controls, noise copies and luck books
are not counted)

**Adoption rule** (the lab README's protocol, step 5): an iteration is adopted only if its paired
Δ against the current base is >= 0 on TRAIN, T1 and T2, and its T1 Δ exceeds 2 × C_U0's T1 noise
sd. The adopted variant becomes the base; bars (1)-(5) are re-read for it, bar (2) against the
same C_rev.

**I1 = `{"short_sleeves": "si+mom"}`** (design iteration idea 1): SI longs as in C_U0; the short
score is the composite of the two ranks, `(si.fillna(0.5) + mom.fillna(0.5)) / 2` (w_si = w_mom
= 1), averaged over 5 bars among the eligible names and re-ranked, then masked where a name is
held long. No new code (the core's two-pass book). Mechanism: SI's own short leg looks like the
steadier half of its book (about +5 / +4.5 in good half-years, −2 in bad ones); the composite
keeps SI's information on the short side while adding MOM's (failure mode b). Expected (this
builder's guess, not the design's): T1 Δ −1.5 to +2.

**I2, its form decided by bar (3) on the current base** (the design's idea 3 is to be used if
bar (3) fails; otherwise idea 2). Both forms are specified here in full; the code of the one
that runs is added afterwards as a default-off knob, verified to leave C_U0's and C_rev's
weights bit-identical on real panels before its numbers are read.

- **I2b, beta-matched short sizing** (if bar (3) fails, i.e. realized |beta| > 0.15 in more than
  2 of the 10 T1 periods): the long leg is unchanged (equal 3% slots). Each short slot is scaled
  at entry by the bar's median beta ÷ the name's beta — p1's 126-bar beta to the equal-weight
  eligible names (`compose(...)['beta']`), the median over eligible names with beta > 0 —
  clipped to [1/1.5, 1.5] (beta <= 0: 1/1.5; no beta: 1), frozen while held; a bar's new short
  entries are scaled down together when they would lift the short side's nominal gross above
  0.45 (`p1._weights(no longs, shorts, beta, gross 0.45, long_share 0, slots 15, sizing
  'beta', max_size 1.5)` for the short side). Knob `short_sizing: equal | beta`. Mechanism: the
  short tail's higher beta (MOM 1.21 / 1.11 vs SI longs 1.04 / 1.30) is matched name by name,
  so the book's beta stays near zero period by period without touching the long selection.
  Expected: more periods within ±0.15; alpha Δ −2 to +1 (beta sizing cost MOM −2.1).
- **I2c, the ratcheted chandelier on the shorts** (if bar (3) passes): an extra exit for every
  held short, adaptive_hedge TUNEDv2's chandelier. At entry (decision bar t0) best = close(t0)
  and stop = close(t0) + 16 × ATR(48)(t0); on each later bar best = min(best, low), stop =
  min(stop, best + 16 × ATR(48)) (never loosened); the short is covered when the close is above
  the stop (decided at the close, filled at the next open) and is not shorted again until its
  short score has left the short entry zone once (p1's squeeze-stop re-entry rule). ATR =
  `bt.atr(panel, 48)`. Knobs `short_chandelier: 0` (ATR multiple; 0 = off; I2c sets 16) and
  `chandelier_window: 48`. Mechanism: covers recovered losers before junk rallies squeeze them
  (Nov 2020, Jan-Mar 2021). Expected: a few exits a year; Δ −1 to +1.5; a smaller T3 drawdown.
  In-sample: 16 and 48 are AH's TUNEDv2 values, tuned on TB + T3 for another book, so I2c's
  TRAIN Δ is partly in-sample.

I2 runs on top of the base as it stands after I1's decision.

### §0.8 The final

The base after the adoption decisions: C_U0 unless I1 or I2 was adopted. C_rev is never the
final. If bar (2) fails, leg specialization is rejected and reported as such; the final is still
the base (the orchestrator ranks finals on every gate first, then the design bars). The final is
declared in this log before its final verdict and bear; nothing is read on T4 or TL here.

### §0.9 Expected (the design's, before any number)

- TRAIN (TB + T3): −1 to +3% (SI's long leg is TRAIN-weak: leg alpha TB +0.5, T3 −3.5; MOM's
  short leg +2.5 / +1.4).
- T1: +6 to +12%. T2: +4 to +9%. |beta| <= 0.1.
- Volatility above A_U0's (16.3% on T1): both legs come from high-beta, high-volatility tails.
- Bear episodes: Q4 2018 −3 to −8% (market −18.2%); COVID between small positive and small
  negative; 2022 above +10%; Feb-Apr 2025 beaten.
- C_rev 2-5 pts below C_U0 on T1 and T2, if the hypothesis holds.
- 150-220 orders a year.
- TL (the orchestrator's read): +4 to +10%.

### §0.10 Failure modes to watch (the design's)

(a) The cross concentrates both books' crash-prone legs: SI's long leg carries its big losses
(about −7 pts in 2018-H2 and −6 in 2026-H2); MOM's short leg carries the junk-rally crashes (Nov
2020, Jan-Mar 2021). So T3 could be the worst of both.
(b) SI's own short leg looks like the steadier half; dropping it may cost more than MOM's short
leg adds. Then C_rev wins and the hypothesis is rejected.
(c) Leg alphas measured inside other books (other gross, sizing and mechanics) need not add up.
(d) Higher idiosyncratic volatility can push T1's P below 0.90 (G2) even at similar alpha.
(e) MOM's monthly short ranks combined with p1's daily risk exits (squeeze stop, crowding) change
MOM's short leg relative to U0v2.
(f) (execution, this builder's) A short squeezed past twice its entry value needs free cash to
be covered under lock_cash; a refused cover is never re-sent (GME 2024-05-14 in the composite's
T2, Wayfair 2020 in adaptive_hedge). MOM's short tail is high-beta: hence the reconciliation on
every open test.

### §0.11 In-sample caveats carried into every reading

- T1 is partly in-sample for SI: informed_flow's studies scored from 2020-07, and
  informed_flow_ls was developed on 2022-26 top-500 data. Only MOM is clean on T1.
- AH's TUNEDv2 was tuned on TB + T3; AH enters C_U0 only through `_on_priced_opens` (an
  execution helper: no AH signal), and I2c would borrow its chandelier values (above).
- TL is fresh only for the combination: informed_flow_ls §38P already read names 1501-2000 for
  SI.
- G1 is read on TB + T3; MOM's E0 −8.2% (no short interest there) is disclosed beside it.

## §1 Execution audit (2026-10-24 00:23-00:25, before any lab run; `logs/audit.log`)

§0 was written and snapshotted at 00:23:05 (`prereg/S0_20261024-002305.md`, hashes in
`prereg/prereg.log`; the header's "00:25" was the planned time). `research.py audit C_U0,C_rev
TB,T3,T1,T2`: every period's rebuild with vectorbt's order log equals `lab.book` (asserted to
1e-12). No return level was read.

| label | test | orders asked | refused | part-filled | NaN-price drops | off-target name-days | stuck P&L | max gross | max long / short | flips short→long | engine~preview corr | TE / yr | drag / yr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_U0 | TB | 283 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 1 | 0.9996 | 0.23% | −0.82% |
| C_U0 | T3 | 209 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 1 | 0.9996 | 0.24% | −0.83% |
| C_U0 | T1 | 662 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 3 | 0.9998 | 0.29% | −0.86% |
| C_U0 | T2 | 722 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 1 | 0.9998 | 0.29% | −1.08% |
| C_rev | TB | 317 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 2 | 0.9985 | 0.44% | −1.31% |
| C_rev | T3 | 194 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 0 | 0.9992 | 0.51% | −1.04% |
| C_rev | T1 | 629 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 1 | 0.9996 | 0.34% | −0.86% |
| C_rev | T2 | 659 | 0 | 0 | 0 | 0 | 0 | 0.900 | 15 / 15 | 3 | 0.9991 | 0.45% | −1.18% |

- **The §0.5 execution rule:** refused plus part-filled = 0.00% of the orders on every open test
  for both labels, so gross stays 0.90. Nothing is stuck, so the reconciliation's repaired
  figures equal the record for C_U0 and C_rev on TB, T3, T1 and T2.
- **Invariants:** no name on both sides, at most 15 a side, Σ|w| <= 0.90, one constant weight per
  hold, in every period of both labels. Flips are rare (1-3 per test): a held short whose SI
  rank reaches the long zone is covered and bought in one order, as designed; no long → short
  flip occurred.
- **Engine vs research preview:** within the pre-registered ranges everywhere (correlation >=
  0.998 against >= 0.98; tracking error 0.23-0.51%/yr against <= 2%; drag −0.82 to −1.31%/yr,
  i.e. costs and borrow). The five largest daily gaps per test are day 0 (the opening book and
  the up-front borrow, −0.17% to −0.37% of the book) except three, all under the 0.5% rule and
  all explained (`scratchpad gap_days.py`, mechanics only): C_rev T3 2020-10-08 / 09 (+0.21% /
  −0.16%) and T2 2026-03-09 (−0.29%) are the engine's retrim cap selling runaway MOM longs
  (ENPH and SEDG; AMPX), which the cost-free preview trims a bar apart; C_rev TB 2019-02-25
  (−0.07%) is a rotation day's costs (DELL, NOW).

The book the strategy asks for is the book the engine holds. Next: the lab runs (§0.5 step 2).

## §2 C_U0 and the reverse cross C_rev (runs 00:26-00:29, noise 00:29-00:32)

`lab.py run … TB,T3,T1,T2 --costs2` (`logs/run_C_U0.log`, `logs/run_C_rev.log`). Alpha against
the equal-weight buy-and-hold of the same names; TRAIN = TB + T3 stitched. C_si, C_mom and A_U0
are xs_si_mom_composite's saved labels (read-only).

| | **C_U0** (SI long, MOM short) | C_rev (MOM long, SI short) | C_si | C_mom | A_U0 |
|---|---|---|---|---|---|
| TRAIN alpha (P) | **+0.65% (0.56)** | +0.27% (0.48) | +0.64% | −0.25% | +1.97% |
| · TB / T3 | +1.33 / −2.30 | −2.13 / −2.12 | −2.86 / +2.30 | +1.36 / −6.66 | +3.17 / −2.36 |
| · TRAIN late / 2× costs | −1.55 / +0.12 | −2.03 / −0.26 | | | |
| T1 alpha (P) | **+10.62% (0.965)** | +6.30% (0.889) | +9.40% | +9.04% | +12.88% |
| · beta / vol / maxDD | −0.028 / 13.5% / −16.7% | +0.052 / 12.5% / −15.2% | | | |
| · late / 2× / top-5-out | +8.58 / +10.05 / +3.68 | +4.24 / +5.80 / +2.42 | | | |
| · orders a year | 138 | 132 | | | |
| T2 alpha (P) | **+3.85% (0.79)** | +1.27% (0.62) | +2.53% | +1.86% | +5.07% |
| · beta / vol / maxDD | −0.143 / 13.3% / −23.8% | +0.067 / 10.6% / −18.2% | | | |
| · late / top-5-out | +1.41 / −0.39 | −0.81 / −2.27 | | | |
| down months (n 38, market −4.12%) | +0.87% (63% > 0) | +0.33% (61%) | | | |
| Q4 2018 (market −18.2%) | −7.2% | −6.5% | | | |
| COVID crash (−37.0%) | +3.3% | +2.3% | | | |
| 2022 bear (−24.5%) | +9.6% | +14.5% | | | |
| Feb-Apr 2025 (−19.5%) | +3.4% | −2.4% | | | |

C_U0 per period (annualized alpha): TB −12.4, +10.9, −3.6, +17.4; T3 +9.6, −16.4, +8.5; T1
+8.7, +17.5, +10.6, +2.7, +28.4, +3.0, +28.6, +32.5, −11.6, −30.8; T2 +20.7, +1.9, −15.5, −7.5,
+23.7, +3.6, +12.2, +23.9, +4.7, −25.4. Like every component, both 2026 halves of T1 lose (−11.6,
−30.8) and T2's 2026-H2 too (−25.4).

**Noise** (`lab.noise`, 5 names dropped a period, 3 copies; copy − label alpha): TB −0.56, +1.30,
+0.64 (sd 0.94); T3 −0.78, +1.04, −0.25 (sd 0.94); **T1 −0.52, −2.54, −0.97 (sd 1.06)**; T2 −0.55,
−2.55, −1.74 (sd 1.00). All three T1 and T2 copies land below the label: C_U0's validation paths
look like favourable draws (copies' means T1 +9.3%, T2 +2.2%). The adoption bar for iterations
(§0.7) is a T1 Δ above 2 × 1.06 = **2.12 pts**.

### §2.1 The pre-registered bars on C_U0 (`logs/bars_C_U0.log`; G4 and the verdict in §2.2)

| bar | reading | result |
|---|---|---|
| (2) falsification: Δ(C_U0 − C_rev) > 0 on TRAIN, T1, T2 and P >= 0.80 on T1 and T2 | TRAIN +0.38 (P 0.58, 4/7 periods); T1 **+4.31 (P 0.775, 6/10)**; T2 **+2.58 (P 0.683, 5/10)**; TB +3.46 (P 0.79), T3 −0.18 (P 0.51) | **fail**: every Δ is positive, but P is below 0.80 on T1 and T2 |
| (3) realized per-period beta within ±0.15 in >= 8 of 10 T1 periods | −0.10, −0.13, −0.26, −0.29, −0.21, +0.02, −0.08, +0.44, +0.64, +0.15 | **fail** (4/10) |
| (4) alpha >= max(C_si, C_mom) on TRAIN and T2 | TRAIN 0.650 vs 0.635 (C_si); T2 3.85 vs 2.53 (C_si) | **pass** (TRAIN by 0.015 pt) |
| (5) context: Δ(C_U0 − A_U0) >= 0 on TRAIN and T2, >= −1.0 on T1 | TRAIN −1.32 (P 0.41), T1 **−2.26** (P 0.33), T2 −1.22 (P 0.39) | **fail** |

- **Bar (2) fails as pre-registered: leg specialization is rejected, whatever C_U0's own alpha.**
  The primary beats the reverse cross on every open test's point estimate, by +4.3 pts on T1 and
  +2.6 on T2 (the size the design expected: "C_rev 2-5 pts below"), but neither clears P 0.80,
  and TRAIN is level (+0.4, P 0.58; T3 −0.2). Half the periods favour each cross on T2 (5/10).
- **Bar (3) fails, so I2 takes its I2b form** (beta-matched short sizing, §0.7), unless I1 is
  adopted and passes bar (3) itself. The T1 betas run negative in 2022-2024 (−0.29 to −0.08) and
  strongly positive from 2025-H2 (+0.44, +0.64, +0.15): the stitched T1 beta (−0.03) is a
  cancellation, not a neutral book.
- **Bar (5): integration beats specialization.** The composite A_U0 leads by 1.3 (TRAIN), 2.3
  (T1, beyond the −1.0 tolerance) and 1.2 (T2) pts; against A_U0's repaired T2 (+4.15%) the T2
  gap is −0.3.
- Bar (4) passes, but TRAIN's margin over C_si (0.015 pt) is nothing; against the single
  sleeves, paired (`logs/delta_C_U0.log`): C_U0 − C_si TRAIN +0.01 (P 0.55), T1 +1.21 (P 0.65),
  T2 +1.32 (P 0.64); C_U0 − C_mom TRAIN +0.90 (P 0.55), T1 +1.57 (P 0.61), T2 +1.99 (P 0.70).

Context (paired, read-only): C_U0 − v6_si TRAIN −0.40 (P 0.52), T1 +0.21 (P 0.53), T2 −0.10
(P 0.51); C_U0 − resid_momentum U0v2 TRAIN −2.04 (P 0.32), T1 +1.88 (P 0.64), T2 −1.64 (P 0.33).
On these numbers the cross book is about level with the published SI book and does not beat
MOM's own published book outside T1.

### §2.2 G4, the verdict and the bear table (luck 00:32-00:35; `logs/verdict_C_U0.log`,
`logs/bear_C_U0_C_rev.log`)

**Luck** (T1, 100 books with the names shuffled inside dollar-volume tenths): real +10.62%, median
+1.18%, p95 +6.87%, **percentile 100**.

**Verdict C_U0** (market-neutral): G1 +0.65% ✓ (thin; MOM's E0 −8.2%, invisible here, disclosed) ·
G2 +10.62%, P 0.965 ✓ · G3 +3.85% ✓ · G4 100th ✓ · G5 late +8.58%, 2× +10.05% ✓ · G6 all four
episodes and down months +0.87% ✓ · G7 beta −0.028 ✓ · G8 +3.68% ✓. **All eight gates pass**, so
bar (1) passes.

| | Q4 2018 | COVID | 2022 bear | Feb-Apr 2025 | CAGR 2018-07 → 2026-10 | max DD |
|---|---|---|---|---|---|---|
| C_U0 | −7.2% | +3.3% | +9.6% | +3.4% | 6.1% | −16.7% |
| C_rev | −6.5% | +2.3% | +14.5% | −2.4% | 4.7% | −15.2% |
| market (equal weight) | −18.2% | −37.0% | −24.5% | −19.5% | | |

Calendar years, C_U0 / market: 2018-H2 −7.3 / −6.2, 2019 +1.7 / +35.6, 2020 +7.8 / +25.7, 2021
+0.2 / +21.5, 2022 +14.2 / −18.6, 2023 +0.2 / +22.7, 2024 +13.5 / +15.3, 2025 +33.8 / +16.2, 2026
to Oct 16 −7.8 / +11.7.

## §3 Iteration I1, composite shorts (audit 00:33-00:34, run 00:34-00:36; `logs/audit_I1.log`,
`logs/run_I1.log`, `logs/delta_I1.log`, `logs/stuck_I1.log`)

**Audit first** (all four open tests, before its numbers): TB, T3 and T1 clean (0 refused,
part-filled, dropped or off target; invariants hold; engine~preview correlation >= 0.9986). **T2,
2024-05-14, the GME squeeze**: GME's cover part-filled (16.5 of 28.9 shares) and the same bar's
new short NTLA was refused (NoCashShort); both stuck the period's last 32 bars, 64 off-target
name-days, 0.30% of T2's orders (under the 1% rule: gross stays 0.90). It is the composite A_U0's
T2 failure (§9 / §12 there): I1's short side is the same si+mom composite. The preview's T2
tracking error (2.07%/yr against the 2% expectation) and its largest gaps (2024-05-14 / 15 / 16,
06-06, 06-07) are that logged failure.

| I1 − C_U0 (paired) | TB | T3 | T1 | T2 | TRAIN |
|---|---|---|---|---|---|
| Δ (P) | −4.20 (0.07) | −1.22 (0.36) | **+0.99 (0.65)** | +0.75 (0.61) | **−2.42 (0.17)** |
| periods better | 0/4 | 1/3 | 7/10 | 7/10 | 1/7 |

I1 alone: TRAIN −1.77% (P 0.29), T1 +11.61% (P 0.982, beta −0.03), T2 +4.60% (P 0.82) as run.
**Reconciled** (`research.py stuck I1 T2 C_U0`): the stuck GME short made +4.91% of the book from
its due open (+1.86% on the due bar), NTLA's missing short −0.38%; with both orders filled at the
due open I1's T2 is **+3.61% (P 0.75)** and its T2 Δ over C_U0 **−0.24 (P 0.51)**. C_U0 has no
stuck order.

**Rule:** Δ >= 0 on TRAIN ✗ (−2.42), T1 Δ +0.99 < 2.12. **Not adopted.** Base stays C_U0.
Variants used: 2 (C_U0, I1).

## §4 Iteration I2: its form, its code (2026-10-24 00:38-00:45, before any I2 number)

**Form.** I1 was not adopted, so I2 runs on C_U0, and bar (3) on C_U0 failed (4/10 T1 periods
within ±0.15): by §0.7, **I2 = I2b, beta-matched short sizing**, label `I2b`, overrides
`{"short_sizing": "beta"}`. The chandelier form (I2c) is not run and is parked in IDEAS.md.

**Code** (written after §0.7 fixed every rule, before any I2b number): `strategy.py` now has the
explicit signature of the core's knobs plus `short_sizing` (`equal` | `beta`, default `equal`,
added to config.yaml); with `equal` it returns `core.signals(...)` unchanged. With `beta` it builds
the same two-pass book from the core's own `ranks`, `side_score` and `select`, sizes the longs
with `p1._weights(..., sizing="equal")` and the shorts with `p1._weights(no longs, shorts, beta,
gross=gross / 2, long_share=0, slots, sizing="beta", max_size=1.5)`, then
`ah._on_priced_opens`. `valid()` accepts `beta` only with `combine: rank` and no `short_gate`.

**Default path unchanged** (`logs/identity_after_I2b_knob.log`): on TB 2018-07, T3 2021-01, T1
2024-01 and T2 2026-07, C_U0's and C_rev's weights and borrow equal the core's bit for bit, and
`lab.book` reproduces the saved daily returns exactly (max |diff| 0.0). A new test pins I2b on a
synthetic panel: the same longs and the same short names as C_U0, each short sized at entry by
median beta ÷ its beta (clipped, checked by hand on > 20 entries where the cap does not bind),
frozen while held, short gross <= 0.45, gross <= 0.90. Next: I2b's audit, then its run; adoption
by §0.7's rule (Δ >= 0 on TRAIN, T1, T2 and T1 Δ > 2.12).

## §5 Iteration I2b, beta-matched short sizing (audit 00:41-00:42, run 00:42-00:44;
`logs/audit_I2b.log`, `logs/run_I2b.log`, `logs/delta_I2b.log`, `logs/bars_I2b.log`)

(§4's header range "00:38-00:45" should read 00:37-00:41: the identity check ended 00:40:24 and
the S4 snapshot was taken at 00:40:56; §4 is left as snapshotted.)

**Audit first:** clean on all four open tests (0 refused, part-filled, dropped or off target;
invariants hold, max gross 0.90; engine~preview correlation >= 0.9995). It places exactly C_U0's
orders (283 / 209 / 662 / 722): the same names, other short sizes.

| I2b − C_U0 (paired) | TB | T3 | T1 | T2 | TRAIN |
|---|---|---|---|---|---|
| Δ (P) | +0.56 (0.63) | +0.60 (0.64) | **−1.78 (0.02)** | +0.30 (0.64) | +0.67 (0.72) |
| periods better | 2/4 | 1/3 | 2/10 | 4/10 | 3/7 |

I2b alone: TRAIN +1.32% (P 0.61), T1 +8.83% (P 0.94, beta **+0.099**), T2 +4.15% (P 0.81, beta
−0.045); episodes Q4 2018 −7.8%, COVID +0.7%, 2022 +4.8%, Feb-Apr 2025 −0.3% (all above the
market); down months +0.42%.

**Rule:** T1 Δ −1.78 < 0 (and < 2.12). **Not adopted.** Base stays C_U0. Variants used: **3 of 3**
(C_U0, I1, I2b); the budget is spent.

**What it shows (mechanism, not a decision).** I2b did not fix bar (3): 5/10 T1 periods within
±0.15 (realized −0.23 to +0.68), against C_U0's 4/10. Sizing the shorts at median beta ÷ beta
shrinks MOM's high-beta shorts (their ex-ante beta exposure falls from about −0.65 to −0.50) while
the long leg is untouched, so every period's beta moved up, by 0.04-0.16 (T1 stitched −0.03 →
+0.10) *[corrected after the S6 snapshot: first written "−0.11 to +0.68" and "while the SI longs
keep full 3% slots at a beta above the median", which §7's table contradicts for 2023-24]*: that helped
the negative 2022-2024 periods and worsened the positive ones (2025-H2 +0.44 → +0.50, 2026-H1
+0.64 → +0.68, 2026-H2 +0.15 → +0.30). The per-period beta swings are a regime drift of the legs
(negative in 2022-24, strongly positive from 2025-H2), which a beta measured at entry on the
trailing 126 bars cannot neutralize; the design's premise that SI's long tail and MOM's short
tail carry matching betas holds on the stitched T1 (−0.03) but not period by period. I1 fails
bar (3) too (4/10; +0.55, +0.45, +0.36 from 2025-H2).

## §6 The final (declared 2026-10-24 00:45 CEST, before the final verdict and bear re-run, and
before any T4 or TL read — neither is read by this agent)

By the §0.8 rule: neither I1 nor I2b was adopted, so the base is **C_U0**. **Declared final:
`xs_cross_leg/C_U0`, overrides `{}`, lab flags `--costs2`, no `--hedge`.** Its bars: (1) works,
all eight gates ✓; **(2) falsification ✗** (leg specialization rejected: Δ over C_rev positive
everywhere but P 0.775 / 0.683 on T1 / T2); (3) beta check ✗ (4/10); (4) best single sleeve ✓
(TRAIN by 0.015 pt); (5) integration context ✗ (A_U0 leads on TRAIN, T1 and T2). Its luck run
exists (100th percentile); its verdict and bear are re-run below for the record. C_rev is not the
final (§0.3: never adopted after the fact).

**Final verdict and bear, re-run after the declaration** (`logs/verdict_final_C_U0.log`,
`logs/bear_final.log`): identical to §2.2 — all eight gates pass; Q4 2018 −7.2%, COVID +3.3%, 2022
+9.6%, Feb-Apr 2025 +3.4%; CAGR 6.1%, max DD −16.7% over 2018-07 → 2026-10. The verdict
re-computed the T1 luck books (the luck cache key hashes the params, and `short_sizing` had
joined them): the 100 re-run books under the post-§4 code give the same median (+1.18%), p95
(+6.87%) and percentile (100) to the last digit, which re-confirms that the default path did not
move.

## §7 Diagnostics of the final (no decision; 00:45-01:02; `logs/betas.log`, `logs/legs_*.log`,
`logs/overlap.log`, `logs/report_C_rev_base.log`, `logs/noise_C_rev.log`,
`logs/noise_delta_C_U0_C_rev.log`)

**The cross books are recombinations of the controls' legs** (`overlap`, positions only). C_U0's
15 longs ARE C_si's 15 longs on every bar of every test (15.0 / 15), and 13.0-14.1 of its 15
shorts are C_mom's; C_rev's longs are C_mom's (15.0) and 13.7-14.2 of its shorts are C_si's. The
long mask binds on 1.1-2.2 names a bar (C_U0 longs whose MOM score sits in the short entry
zone): the design's "share about 1 name" holds. All four books trade on identical panels (their
benchmark rows are identical; the panels were written 2026-10-23 07:54, before every label).

**Leg attribution** (annualized arithmetic contribution minus beta × market; adaptive_hedge/legs.py):

| leg alpha | TB | T3 | TRAIN | T1 | T2 |
|---|---|---|---|---|---|
| C_U0 long (SI) | −1.6 | +0.5 | +1.0 | **+7.0** | **+5.3** |
| C_U0 short (MOM) | +3.6 | −2.3 | +0.1 | **+6.0** | +1.0 |
| C_rev long (MOM) | +0.3 | −4.1 | +0.6 | +5.1 | +4.3 |
| C_rev short (SI) | −0.8 | +2.9 | +0.6 | +3.4 | −1.6 |
| C_si long / short | −2.3 / −0.3 | +0.3 / +4.0 | +0.3 / +1.3 | +7.0 / +3.7 | +5.1 / −1.5 |
| C_mom long / short | +0.9 / +3.6 | −4.0 / −1.6 | +1.3 / +0.4 | +5.1 / +7.1 | +4.5 / +0.6 |

On validation the point estimates do what the design said: each signal's stronger leg is the
one the cross keeps (T1: SI long +7.0 over SI short +3.4-3.7; MOM short +6.0-7.1 over MOM long
+5.1; T2: SI long +5.3 over SI short −1.5; MOM's legs +4.5 long vs +0.6-1.0 short, so on T2 MOM's
better leg is the LONG one and the cross keeps the weaker). On TRAIN every leg is within ±1.3 pt
of zero: there is no leg specialization to keep there. Costs and cash take 1.3-1.9 pts a year on
T1 and T2.

**Betas, ex ante and realized** (Σ weight × 126-bar beta, per period; the leg columns are the
legs' beta exposures at 0.45 gross a side):

| T1 period | 22-H1 | 22-H2 | 23-H1 | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 |
|---|---|---|---|---|---|---|---|---|---|---|
| ex-ante long (SI) | +0.54 | +0.46 | +0.43 | +0.35 | +0.36 | +0.54 | +0.65 | +0.74 | +0.79 | +0.92 |
| ex-ante short (MOM) | −0.67 | −0.65 | −0.63 | −0.56 | −0.71 | −0.64 | −0.60 | −0.64 | −0.37 | −0.40 |
| ex-ante book | −0.13 | −0.19 | −0.21 | −0.21 | −0.36 | −0.11 | +0.05 | +0.10 | +0.42 | +0.52 |
| realized | −0.10 | −0.13 | −0.26 | −0.29 | −0.21 | +0.02 | −0.08 | +0.44 | +0.64 | +0.15 |
| alpha | +8.7 | +17.5 | +10.6 | +2.7 | +28.4 | +3.0 | +28.6 | +32.5 | −11.6 | −30.8 |

The mismatch is visible ex ante and comes mostly from the LONG leg: SI's low-days-to-cover names
carried an average beta of ~0.8 in 2023-24 (0.35 / 0.45) and ~2.0 in 2026-H2 (0.92 / 0.45), while
MOM's shorts stayed at 1.3-1.6 until 2026, when they fell to ~0.85. The design's matched tail
betas (TB / T3: SI long 1.04 / 1.30, MOM short 1.21 / 1.11) do not hold period by period in
2022-26; the stitched T1 beta (−0.03) is a cancellation of a net-short 2022-24 and a net-long
2025-H2 / 2026. On T2 the book runs net short (realized −0.37 to −0.13 in 2022-24, stitched
−0.14). That is why I2b (shorts only) could not fix it.

**Post hoc, decides nothing: C_rev's own noise copies** (`lab.noise xs_cross_leg C_rev 5 3`, the
same seeds as C_U0's copies; identical benchmark rows asserted, so each pair drops the same names).
C_rev's copy − label: TB +1.27 / +0.19 / −0.29, T3 −0.67 / −1.36 / −1.57, T1 −0.17 / +0.70 /
+1.23 (sd 0.71), T2 −0.61 / −0.85 / +0.69. Paired Δ(C_U0 copy − C_rev copy):

| | labels Δ (P) | copy 0 | copy 1 | copy 2 | copies mean |
|---|---|---|---|---|---|
| TRAIN | +0.38 (0.58) | −1.39 (0.45) | +1.99 (0.68) | +1.35 (0.63) | +0.65 |
| T1 | +4.31 (0.775) | +3.97 (0.76) | +1.07 (0.58) | +2.12 (0.66) | +2.39 |
| T2 | +2.58 (0.683) | +2.64 (0.68) | +0.89 (0.58) | +0.15 (0.51) | +1.23 |

C_U0 leads the reverse cross on T1 and T2 on every path, but on the reduced panels by about half
the labels' margin (T1 +2.4, T2 +1.2) and never at P >= 0.80. The labels' T1 Δ is above all three
copies and its T2 Δ above two of them: C_U0's own T1 and T2 paths are favourable draws (§2),
C_rev's are not. Bar (2)
stays failed as pre-registered; the direction of the leg story is consistent, its size is not
established.

## §8 Conclusion (C_U0, lab v2; T4 and TL not read)

**Answer to the design's question: not established. The cross book works as a book, but the
pre-registered falsification failed, and integrating the two signals (A_U0) beats specializing
them.**

- **Bar (1), works: all eight gates pass.** TRAIN +0.65% (P 0.56; G1 on TB + T3; MOM's E0 −8.2%,
  invisible to it, disclosed), T1 +10.62% (P 0.965, beta −0.03, 100th luck percentile, top-5-out
  +3.68%, late fills +8.58%, 2× costs +10.05%), T2 +3.85% (P 0.79), every bear episode beaten
  (Q4 2018 −7.2% vs −18.2%, COVID +3.3%, 2022 +9.6%, Feb-Apr 2025 +3.4%), down months +0.87% vs
  −4.12%. Nothing was refused, part-filled or stuck on any open test (§1).
- **Bar (2), falsification: fails.** C_U0 − C_rev: TRAIN +0.38 (P 0.58), T1 +4.31 (P 0.775), T2
  +2.58 (P 0.683): positive everywhere, as the design expected in size (2-5 pts), but not
  significant; the paired noise copies halve the margin (§7). Leg specialization is rejected,
  whatever C_U0's own alpha.
- **Bar (3), beta mechanism: fails** (4/10 T1 periods within ±0.15; −0.29 to +0.64). The legs'
  betas drift apart by regime, the long leg most (§7).
- **Bar (4), best single sleeve: passes**, by 0.015 pt on TRAIN (0.650 vs C_si 0.635) and 1.3 on
  T2; paired over C_si / C_mom the margins are +0.0 / +0.9 (TRAIN), +1.2 / +1.6 (T1), +1.3 / +2.0
  (T2), P 0.55-0.70.
- **Bar (5), specialization vs integration (context): fails.** A_U0 leads by 1.3 (TRAIN), 2.3
  (T1) and 1.2 (T2) pts (P 0.33-0.41 for C_U0); with A_U0's T2 stuck orders repaired the T2 gap is
  −0.3.
- **Iterations:** I1 (SI longs, composite shorts) TRAIN −2.4 / T1 +1.0 / T2 +0.8 (repaired T2
  −0.2), not adopted; I2b (beta-matched shorts) +0.7 / −1.8 / +0.3, not adopted, and it raised
  the beta instead of neutralizing it. Variants: 3 of 3.
- **Risk and regime:** T1 volatility 13.5% (below A_U0's 16.3%, against the design's
  expectation), max drawdown −16.7% (T1) and −23.8% (T2). Like every component, the edge is a
  2022-2025 one: both 2026 halves lose on T1 (−11.6, −30.8%/yr of alpha), 2026-H2 on T2 (−25.4).
  2018-07 → 2026-10 CAGR 6.1% against 13.8% for the equal-weight market of the same names
  (stitched TB + T3 + T1 benchmark; a dollar-neutral book cannot keep up with a bull market;
  shorts lock cash).

**In-sample caveats.** T1 is partly in-sample for SI (informed_flow's studies scored from
2020-07; informed_flow_ls was developed on 2022-26 top-500 data): C_U0's long leg IS C_si's long
leg, so its T1 long-leg alpha (+7.0) is SI's partly in-sample number; only MOM's short leg is
clean on T1. AH enters only through `_on_priced_opens` (an execution helper), so its TB + T3
tuning does not touch this book (I2c, which would have borrowed AH's tuned chandelier, was not
run). TL is fresh only for the combination: informed_flow_ls §38P already read names 1501-2000 for
SI. G1 is read on TB + T3; MOM's E0 −8.2% is disclosed beside it. C_si, C_mom and A_U0 are the
composite's saved labels (same core code, same panels: §7).

**Declared final (§6): `xs_cross_leg/C_U0`, overrides `{}`, lab flags `--costs2`, no `--hedge`.**

**For the orchestrator's TL read.** Names 1501-2000 are smaller and more squeeze-prone than T2's.
C_U0's short leg is MOM's losers; I1's composite shorts got stuck in T2's GME squeeze (§3), and a
cover can exceed lock_cash's margin at gross 0.90. `research.py audit` (refused, part-filled and
NaN-dropped orders, off-target name-days both ways, stuck P&L from the due open, the preview
comparison) should be repeated on TL before its returns are read; this folder's `research.py`
refuses locked tests by design, so that step is the orchestrator's.

## §9 Expected (§0.9) against observed

| item | expected | observed (C_U0) |
|---|---|---|
| TRAIN (TB + T3) | −1 to +3% | +0.65% (P 0.56) |
| T1 | +6 to +12% | +10.62% (P 0.965) |
| T2 | +4 to +9% | **+3.85%** (P 0.79) |
| \|beta\| | <= 0.1 | T1 −0.03; **T2 −0.14**; per period up to +0.64 |
| volatility | above A_U0's 16.3% (T1) | **13.5%** (below) |
| Q4 2018 / COVID / 2022 / Feb-Apr 2025 | −3 to −8% / ±small / > +10% / beaten | −7.2% / +3.3% / **+9.6%** / +3.4% |
| C_rev below C_U0 on T1 and T2 | 2-5 pts (if the hypothesis holds) | 4.3 / 2.6 pts, P 0.775 / 0.683 (copies 2.4 / 1.2) |
| orders a year | 150-220 | **138-151** |
| TL | +4 to +10% | not read here (orchestrator) |
