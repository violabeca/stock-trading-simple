# resid_momentum — research log

Proposal 2 of `STRATEGY_OVERVIEW.md` §6, evaluated through the shared lab
(`strategies/proposals_lab/README.md`, `lab.py`). Every number below comes from `lab.py` on the
lab's panels; every step is written here before its numbers are read.

## §0 Pre-registration (2026-10-22 22:50, before any strategy number was computed)

Data checks done before this entry (no returns, no signals): panel shapes, NaN rows and
coverage. Facts: E0's panels hold ~392 names (few names have bars before 2016); three
all-NaN holiday stub rows exist (2026-06-19, 2026-07-03, 2026-09-07); 2-6 names a period stop
trading inside the 2026 windows; short interest exists from 2018-01 only (E0 has none).

**Hypothesis.** In the rolling point-in-time top 500, a dollar-neutral book long the 20 names
with the highest and short the 20 with the lowest *residual* 12-1 momentum (the return left after
removing each name's market beta, divided by its residual volatility), re-ranked only at month
ends with 0.90 / 0.80 bands, and shrunk when its own trailing volatility exceeds 10%/yr, earns
positive alpha against the equal-weight market at |beta| <= 0.3, and loses much less than raw
momentum in momentum crashes (Nov 2020 vaccine rally, Jul-Sep 2026).

**Mechanism.**
1. Investors under-react to firm-specific news, so firm-specific (residual) returns persist for
   months (Jegadeesh-Titman 1993; Grinblatt-Moskowitz 2004; Da-Gurun-Warachka 2014 "frog in the
   pan": steady trends persist more, which dividing by residual volatility rewards).
2. Raw momentum is partly a time-varying market bet: after a fall, the losers are the high-beta
   names, so the short leg is a leveraged long on a rebound (Grundy-Martin 2001; Daniel-Moskowitz
   2016). Ranking the residual removes that bet (Blitz-Huij-Martens 2011; Gutierrez-Prinsky
   2007 find residual momentum about as profitable with ~half the volatility and no rebound
   crash).
3. Momentum's own volatility is predictable and crashes happen in high-volatility regimes, so
   scaling the book by min(1, target / its trailing realized volatility) cuts exposure before
   most crashes (Barroso & Santa-Clara 2015; Moreira-Muir 2017).

**Literature-default config = U0 (`config.yaml` as committed with this entry).**
- residual return e = r − β·m, m = equal-weight return of the panel's names closing >= $5 the
  bar before, β = OLS slope of r on m over the trailing `beta_window` = 252 bars (known at t);
- score = Σe / sd(e) over the formation bars t−251 .. t−21 (`lookback` 252, `skip` 21);
- eligibility: close >= $5, >= 252 bars of history, >= 80% of the formation and beta windows
  observed;
- decisions only at the last trading bar of each month (fills at the next open); the rank is
  held between month ends; long enters at rank >= 0.90 and leaves below 0.80, short mirrored
  (enter <= 0.10, leave > 0.20); 20 + 20 slots, best ranks first; equal slots, dollar-neutral,
  gross 0.98; no stops, no maximum hold;
- crash protection: every slot × scale, scale = min(1, 10% / the unscaled book's trailing
  126-bar realized volatility), recomputed at month ends only, rounded to the nearest 0.25,
  floor 0.25;
- borrow: informed_flow_ls_p1's tiers by short interest / shares outstanding (<= 10%: 0.5%,
  10-20%: 3%, > 20%: 10% a year; no short-interest data: 0.5%); a name above 30% is not shorted
  (and a held short above it is covered at the next month end); `retrim_k` 2 (the engine's long
  concentration cap, as in p1);
- costs: the repo model at `order_notional` 1000, open fills.

**Control arms (pre-registered, each counts as a variant):**
- C_raw: `{"score": "raw", "vol_target": 0}` — 12-1 total return P(t−21)/P(t−252) − 1, no scaling;
- C_res: `{"vol_target": 0}` — the residual ranking without the book-volatility scaling.

**Expected result (honest prior).** Momentum in US large caps since 2000 is weak and crash-prone;
the repo's research loop found raw 12-1 L/S +1.6-1.8%/month gross (t 2.2-2.5) in 2018-2026-06 and
a −7 to −11%/month crash in Jul-Sep 2026. I expect U0 alpha of roughly +1 to +4%/yr on TRAIN and
T1, block-bootstrap P(alpha > 0) of about 0.6-0.8 on T1 (G2 likely FAILS), |beta| < 0.15 for the
residual arms, C_raw's beta negative after the 2022 and 2025 falls, the vol scaling running the
book at ~0.5 most of the time (so U0's alpha below C_res's in calm years), and U0's Jul-Sep 2026
loss at most half of C_raw's. Costs should be small (monthly decisions, ~200-400 orders a year).

**Pass bars.** The lab gates G1-G8 (`lab.verdict`), plus the proposal's own: (a) U0's T1 alpha
> 0 with P >= 0.90 and >= 95% of the shuffled books; (b) its Jul-Sep 2026 loss <= half of
C_raw's; (c) non-negative in >= 3 of the 4 bear episodes; (d) T2 alpha > 0.

**Variant budget: <= 8 through the bar** (U0, C_raw, C_res, the tuned config, iterations, the
final is one of them); tuner trials excluded. Tuning: `lab.tune` on TRAIN only, >= 120 trials,
space written here before it runs.

**Kill:** U0 and both controls negative on TRAIN, or every arm's T1 alpha <= 0.

## Variant count

| # | label (round 1, lab v1) | round 2 (lab v2) | what | status (filled in as run) |
|---|---|---|---|---|
| 1 | U0 | U0v2 | residual + vol-scaled (the pre-registered default; v2: gross 0.90) | **final** both rounds (§6-§7, R2.4-R2.5) |
| 2 | C_raw | C_rawv2 | raw 12-1, no scaling | control (§2, R2.2) |
| 3 | C_res | C_resv2 | residual, no scaling | control (§2, R2.2) |
| 4 | T150 | T150v2 | the tuner's best (150 TRAIN trials; re-tuned on v2) | not adopted (§3, R2.3) |
| 5 | I1 | I1v2 | sector-neutral residuals | not adopted (§4, R2.2) |
| 6 | I2 | I2v2 | beta-neutral slot sizing | not adopted (§5, R2.2) |

6 of the 8 budgeted variants used. Round 2 re-measures the same six on lab v2 and adds none.
Not variants: noise copies, luck books, the leg and crash diagnostics, U0s0v2 (the round-1
final re-measured on v2), and the final's T4 book of each round (U0_T4 on v1, U0v2_T4 on v2:
two reads of T4 in all, both disclosed).

## §1 Premise check — TRAIN only (E0 + TB + T3, 66 decision months 2016-06 → 2021-11)

`research.py premise`: at each month end, the score against the NEXT month's beta-adjusted
return `r_i − β_i r_m` (β = the trailing 252-bar beta known at the decision), eligible names
only. IC = Spearman rank IC per month; spreads = monthly means (top decile − bottom decile;
top 20 − bottom 20, the book's slots); t over months.

| score | IC | IC t | IC>0 | decile/mo (t) | top20 − bot20 /mo (t) | long20 | short20 |
|---|---|---|---|---|---|---|---|
| residual | −0.003 | −0.12 | 48% | +0.21% (0.30) | +0.66% (0.75) | +0.59% | +0.07% |
| raw | −0.019 | −0.76 | 42% | +0.22% (0.28) | +0.25% (0.21) | +0.66% | −0.41% |

By test (residual IC / top-20 spread per month): E0 −0.039 / −1.18%; TB +0.029 / +1.29%;
T3 +0.002 / +2.26%. By year (residual IC): 2016-H2 −0.152 (t −2.4: the Nov-2016 rotation),
2017 +0.007, 2018 −0.005, 2019 +0.032, 2020 +0.060, 2021 −0.024. Raw: 2020 +0.083, 2021 −0.086.

**Read.** The premise is WEAK on TRAIN: no rank IC for either score; only the extremes (top 20
vs bottom 20) earn a positive, insignificant spread, a bit larger for the residual score
(+0.66%/mo ≈ +8%/yr long-short gross, t 0.75). For a 0.49-a-side book that is ~+4%/yr before
scaling and costs, with a standard error larger than the mean. The residual short leg earns
nothing (+0.07%/mo): the spread is long-side. Informational only — it decides nothing; U0 runs
as pre-registered.

## §2 U0 and the two control arms (2026-10-22, `lab.py run … E0,TB,T3,T1,T2 --costs2`)

Stitched alpha per test (block-bootstrap P), beta, max drawdown; late = +50 bp a leg.

| arm | E0 | TB | T3 | TRAIN stitched | T1 (P) | T1 beta | T1 maxDD | T2 (P) | orders/yr T1 |
|---|---|---|---|---|---|---|---|---|---|
| **U0** residual + vol scale | −6.3% | +6.7% | −1.5% | **+0.09%** (P 0.51) | **+10.6%** (0.995) | −0.09 | −9.1% | **+4.6%** (0.905) | 240 |
| C_raw raw 12-1 | −6.0% | +18.6% | −11.2% | | +9.8% (0.85) | +0.11 | −27.0% | +2.7% (0.64) | 161 |
| C_res residual, no scale | −7.1% | +10.6% | −6.3% | | +10.6% (0.96) | −0.11 | −17.7% | +3.2% (0.73) | 177 |

U0 details: T1 late +9.6%, 2× costs +10.4%, top-5-out +8.3%; T2 late +3.4%, top-5-out +2.3%;
volatility 10.5% (T1); mean exposure 0.24-0.45 a side (the scale averages ~0.7). T1 per period:
+15.0, +8.6, +3.0, +3.4, **+56.2** (2024-H1), +0.4, +9.0, +1.0, +14.6, −11.9 (2026-H2).

Paired Δ against U0 (bootstrap P(Δ > 0)): C_raw T1 −0.8 (0.45), T2 −1.8 (0.37), TB +11.9, T3
−9.7; C_res T1 0.0 (0.52), T2 −1.4 (0.29), TB +3.8, T3 −4.8.

**Noise** (`lab.py noise U0`, 5 names dropped a period, 3 copies): sd E0 0.88, TB 0.51, T3 0.34,
**T1 0.21**, T2 0.73 pts. An iteration must clear a T1 Δ of 0.42 pts.
**Luck** (100 shuffled-name books in dollar-volume tenths, T1): real +10.6% vs median +0.4%,
p95 +4.2%: percentile **100**.

Falling markets (book vs equal-weight market): Q4 2018 −4.4% (−18.2%), COVID +4.3% (−37.0%),
2022 bear +14.2% (−25.2%), Feb-Apr 2025 +3.5% (−19.6%); down months mean +1.22% vs −3.97%
(41 months, 73% positive). C_raw: −10.0 / +8.3 / +19.2 / −10.3%; C_res: −6.6 / +4.8 / +18.6 /
+4.8%.

Momentum-crash windows (book return, market in brackets):

| window | U0 | C_raw | C_res |
|---|---|---|---|
| Nov-Dec 2016 (election rotation) | −7.8% (+5.6) | −5.7% | −8.8% |
| Nov 2020 (vaccine rally) | −4.5% (+15.1) | −6.3% | −5.2% |
| Jan-Mar 2021 (junk / value rally) | −5.3% (+10.4) | −9.9% | −9.8% |
| Jul-Sep 2026, T1 | −2.2% (−1.9) | −3.6% | −4.8% |
| Jul-Sep 2026, T2 | −3.7% (−3.6) | −10.1% | −8.0% |
| Jul 1-Oct 16 2026, T1 | −3.5% (−2.1) | −12.7% | −8.5% |

**Read.**
- The pre-registered design holds up on VALIDATION, better than its prior (+1-4% expected; got
  +10.6% on T1 with P 0.995 and 100th luck percentile; +4.6% on T2 with P 0.905), but TRAIN is
  flat (+0.09%): 2016-H2 (−20.9% annualized) and 2021-H1 (−15.6%) are sector-rotation crashes.
  The gap TRAIN vs VALIDATION is a warning, not a result: T1's alpha is carried by 2024-H1
  (+56% annualized; top-5-out still +8.3%).
- Residual vs raw: same T1 alpha (+0.8 pts), but half the drawdown, beta −0.1 instead of
  +0.1, and much better in Feb-Apr 2025 (+4.8% vs −10.3%) and Jan-Mar 2021. Raw won TB (+18.6%).
- The vol scale costs nothing on T1 (Δ 0.0), adds +1.4 pts on T2 and +4.8 on T3, and halves
  max drawdown (−17.7% → −9.1%); it costs 3.8 pts in TB.
- Pass bar (b) "U0's Jul-Sep 2026 loss <= half of C_raw's": T1 −2.2% vs −3.6% = 61% of it:
  **FAIL on T1**; T2 −3.7% vs −10.1% = 37%: pass; through Oct 16 (the crash went on) T1 28%.
  The repo's "−7 to −11%/month" was a raw decile spread; this 20-slot book lost less.

## §3 Tuning — pre-registration (before the study runs)

`lab.tune resid_momentum T150 '{}' <space> 150` on TRAIN (E0 + TB + T3, 11 periods), robust
objective mean − ½ sd of the period alphas, TPE seed 42. Space (8 knobs):
`lookback` 126-252 step 21; `skip` {5, 10, 21}; `beta_window` {126, 252}; `slots` 10-30 step 5;
`q_in` 0.85-0.95 step 0.05; `q_out` 0.6-0.8 step 0.1; `vol_target` 0.06-0.20 step 0.02;
`vol_window` {63, 126, 252}. Adopted only if it beats U0 on BOTH T1 and T2 (paired). Expected:
TRAIN is flat for U0 and the 11 periods are few, so the tuner will mostly fit 2016-H2 / 2021-H1
noise; PBO likely > 0.5; I expect it NOT to be adopted.

**Result (150 trials, PBO 0.615).** Best trial: `lookback` 210, `skip` 21, `beta_window` 252,
`slots` 30, `q_in` 0.95, `q_out` 0.7, `vol_target` 0.06, `vol_window` 126 — robust score −1.7%
(U0 −5.1%), mean TRAIN period alpha +2.2% (U0 +1.5%). The robust objective (mean − ½ sd)
prefers SMALL books: `vol_target` sits on its lower bound and `slots` on its upper one, which
shrinks the period spread more than the mean. Run on OPEN as **T150**:

| | E0 | TB | T3 | T1 (P) | T2 (P) |
|---|---|---|---|---|---|
| U0 | −6.3% | +6.7% | −1.5% | +10.6% (0.995) | +4.6% (0.905) |
| T150 | −2.8% | +5.9% | +1.6% | +4.9% (0.980) | +4.5% (0.978) |
| Δ (P(Δ>0)) | +3.6 (0.91) | −0.8 (0.39) | +3.1 (0.69) | **−5.7 (0.007)** | **−0.0 (0.48)** |

**Not adopted** (it must beat U0 on both T1 and T2; it loses T1 by 5.7 pts and ties T2). As
expected: PBO > 0.5, and the gain on TRAIN is the objective buying less risk.

### Diagnostic: legs of U0 (`research.py legs '{}'`, each side run alone through the engine)

| test | long alpha (beta) | short alpha (beta) |
|---|---|---|
| E0 | −1.1% (+0.54) | −4.5% (−0.50) |
| TB | +2.6% (+0.34) | +2.5% (−0.34) |
| T3 | −2.6% (+0.33) | +1.4% (−0.33) |
| T1 | +5.3% (+0.33, P 0.96) | +5.1% (−0.42, P 0.99) |
| T2 | +3.7% (+0.35, P 0.93) | +1.1% (−0.44) |

Both sides carry T1 equally; the short side has the larger |beta| (residual losers are higher-
beta names), hence the book's −0.09. TRAIN's two losing stretches are on different sides.

## §4 Iteration I1 — sector-neutral residuals (pre-registered before running)

**Change:** `factors: market+sector` — the residual is taken against the market AND the
equal-weight return of the name's other sector members (leave-one-out; a sector needs 5 other
priced members on the bar, else the market stands in): `e = r − β_m m − β_s s`, both slopes
from the same trailing 252-bar OLS. Everything else is U0.
**Mechanism:** a large part of stock momentum is industry momentum (Moskowitz & Grinblatt
1999), and industry momentum reverses in sector rotations. U0's TRAIN losses are exactly those:
Nov-Dec 2016 (−7.8% while the market rose 5.6%, the reflation rotation into energy, banks and
industrials) and Jan-Mar 2021 (−5.3% vs +10.4%). Removing the sector component leaves the
within-sector, firm-specific trend, which the literature finds more persistent (Chaves 2016;
Gutierrez & Prinsky 2007).
**Expected:** better TRAIN (2016-H2, 2021-H1), lower sector concentration; T1 may LOSE,
because 2022's energy longs and 2024's semiconductor longs were sector momentum that paid.
**Bar:** paired Δ >= 0 on TRAIN (stitched), T1 and T2, and T1 Δ > 0.42 pts (2 × noise sd).

**Result I1** (paired Δ vs U0, P(Δ > 0)): TRAIN stitched +0.30 pts (0.56; E0 +1.9, TB −0.8, T3
−0.1), **T1 −0.4 (0.46)**, T2 +0.2 (0.56). Lower max drawdown everywhere (T2 −12.6% → −9.3%),
Feb-Apr 2025 +7.1% (U0 +3.5%) but the 2022 bear +4.5% (U0 +14.2%): U0's 2022 gain was partly a
sector bet (energy long). **Not adopted** (T1 Δ < 0, and far inside 2 × noise anyway). For the
record, TRAIN stitched alphas: U0 +0.09%, C_raw +1.9%, C_res −0.2%, T150 +2.0%, I1 +0.4%.

### Diagnostic: where the crash windows lose (U0, Σ target weight × return, raw P&L)

| window | long leg | short leg | short: high-β half | short: low-β half | mean β short / long |
|---|---|---|---|---|---|
| Nov-Dec 2016 | +0.1% | −7.5% | −4.1% | −3.3% | 1.29 / 0.67 |
| Nov 2020 | +3.8% | −7.9% | −4.4% | −3.4% | 1.42 / 0.66 |
| Jan-Mar 2021 | +1.4% | −5.1% | −3.3% | −1.8% | 1.33 / 0.80 |
| Jul-Oct 16 2026 | −1.4% | −2.2% | −0.1% | −2.2% | 1.13 / 1.44 |

Shorts above 10% short interest / shares are rare (0-15% of short slots) and not the loss.
The TRAIN crash windows are sharp market rallies, and the loss is the SHORT leg: residual
ranking removes beta from the score but not from the book — in a rising market `Σe = Σr −
β̂ Σm` pushes names with over-estimated betas to the short side, so the shorts carry β ~1.3 vs
the longs' ~0.7 and the dollar-neutral book is net short beta (−0.16 at full scale).

## §5 Iteration I2 — beta-neutral slot sizing (pre-registered before running)

**Change:** `sizing: beta` — informed_flow_ls_p1's rule, copied: a slot is scaled AT ENTRY by
the bar's median beta (eligible names) ÷ the name's trailing 252-bar beta, clipped to
[1/1.5, 1.5], and frozen while held (the engine re-trades a name whenever its target changes);
a month end's new entries are scaled down together when the nominal gross would exceed 0.98.
The book scale (vol target) multiplies on top. Everything else U0.
**Mechanism:** the crash windows lose on the short leg because it carries ~2× the long leg's
beta (diagnostic above): the Daniel & Moskowitz (2016) momentum crash is the losers' beta in a
rebound, and residual ranking alone does not remove it from the BOOK. Sizing each slot by
median β / β closes the mismatch, so a sharp rally no longer hits the short side twice as hard.
**Expected:** realized beta ~0 on every test (U0: T1 −0.09, ex-ante −0.11); smaller losses in
Nov 2016 / Nov 2020 / Q1 2021; a smaller 2022-bear gain (U0's +14.2% is partly net short beta);
alpha change uncertain, likely within ±2 pts; more long dollars than short (~0.6 / 0.35).
**Bar:** paired Δ >= 0 on TRAIN, T1 and T2, and T1 Δ > 0.42 pts.

**Result I2** (paired Δ vs U0): TRAIN stitched **−2.2 pts (P 0.015)** (E0 −1.7, TB −1.2, T3
−2.4), **T1 −2.1 (P 0.10)**, T2 +0.2 (0.58). Beta moved as designed (T1 −0.09 → −0.02, T2
−0.09 → −0.04) and the rally windows lost a little less (Nov-Dec 2016 −7.0% vs −7.8%, Nov 2020
−3.2% vs −4.5%, Q1 2021 −4.4% vs −5.3%, Jul-Oct 2026 −2.2% vs −3.5%), but the alpha fell:
shrinking the high-beta shorts gives up the short leg's earnings (+5.1% on T1), and the gross
guard moves the book into larger low-beta longs. **Not adopted.** The beta mismatch is a small
part of the crashes; most of the short-leg loss in a junk rally is the losers' own rebound.

### Diagnostics before the final (not variants)

- U0's volatility scale (unscaled book σ, the 2026 top-500 panel): 1.0 through 2016-2018 (σ
  7.5-10%: the Nov-2016 crash hit at full size), 0.25-0.5 in 2020-21, 0.6-0.8 in 2022-25. The
  downside semi-volatility is ≈ the total (0.105 vs 0.100 … 0.194 vs 0.190), so a downside-risk
  scale (Wang & Yan 2021) would change nothing: not run. *[Round 2: withdrawn as a finding (review
  minor 6). Equal average levels do not imply the same timing of the scale, and no
  downside-scaled book was run. It is an untested guess.]* Rounding to the nearest step flips the
  scale between 0.25 and 0.5 near σ ≈ 27% (2020-09 → 2021-05), each flip re-trading the book:
  U0 places 333 orders a year in T3 vs C_res's 165 (costs: T3 late −1.0 pt, 2× costs −0.25 pt).
- 2024-H1 (+24.8% vs the market's +7.1%) is real momentum: SMCI +3.9 pts, NVDA +2.5, SEDG
  (short) +1.7, VRT +1.6, SPOT +1.3; the largest daily book move in T1/T2 is 3.8%.
- Correlation with the short-interest L/S book (informed_flow_ls_p1 `v6_si`, the lab's reference
  run, read only): daily +0.43 / monthly +0.24 on T1, +0.37 / +0.32 on T2, but −0.39 / −0.53 on
  T3 and −0.17 / −0.08 on TB. A 50/50 blend of the two return streams (a sum, not an engine
  run): T1 alpha +12.0% (P 1.00) at 9.5% vol, T2 +4.2% (P 0.92), T3 +1.1% at 5.4% vol.

## §6 Final candidate (declared 2026-10-22 between the I2 run at 23:42 and the verdict at 23:48, before verdict, bear and T4; first written as "2026-10-23", corrected in the review)

**U0, the pre-registered default** (`config.yaml` as committed: residual + vol-scaled; no
iteration passed the adoption bar). Variants through the bar: 6 of the 8 budgeted (U0, C_raw,
C_res, T150, I1, I2). Next: `lab.py verdict resid_momentum U0`, `lab.py bear resid_momentum U0
C_raw C_res`, then T4 read once (`lab.py run resid_momentum U0_T4 '{}' T4 --costs2`: the final
IS U0, so there is one T4 book).

## §7 Verdict, bear table, T4 (U0 = the final candidate)

**`lab.py verdict resid_momentum U0`: all eight gates pass.** *[Round 2: on lab v1, which starved
new shorts and dropped each period's day 0. On lab v2, G1 fails; see "Round 2".]*

| gate | value | pass |
|---|---|---|
| G1 TRAIN-era (E0+TB+T3) alpha > 0 | +0.09% (P 0.51) | yes — by a hair: TRAIN is FLAT *[Round 2: FAILS on lab v2: −1.37%, P 0.37 (R2.2)]* |
| G2 T1 alpha > 0, P >= 0.90 | +10.6% (P 0.995) | yes |
| G3 T2 alpha > 0 | +4.6% (P 0.905) | yes |
| G4 T1 beats >= 95% of shuffled books | 100th percentile (median +0.4%, p95 +4.2%) | yes |
| G5 T1 late-fill / 2× costs alpha > 0 | +9.6% / +10.4% | yes |
| G6 every bear episode beaten, down months > 0 | +1.22%/month in 41 down months (market −3.97%) | yes |
| G7 \|T1 beta\| <= 0.3 | −0.09 | yes |
| G8 T1 alpha without its top 5 names | +8.3% | yes |

**`lab.py bear resid_momentum U0 C_raw C_res`** (E0+TB+T3+T1 stitched; %):

| | Q4 2018 | COVID crash | 2022 bear | Feb-Apr 2025 | CAGR | max DD |
|---|---|---|---|---|---|---|
| U0 | −4.4 | +4.3 | +14.2 | +3.5 | 4.5 | −18.0 |
| equal-weight market | −18.2 | −37.0 | −25.2 | −19.6 | | |
| SPY | −18.7 | −33.4 | −24.1 | −18.8 | | |
| C_raw | −10.0 | +8.3 | +19.2 | −10.3 | 7.4 | −28.8 |
| C_res | −6.6 | +4.8 | +18.6 | +4.8 | 5.1 | −28.4 |

Calendar years, U0 (market): 2016-H2 −13.9 (+8.8), 2017 −1.2 (+24.6), 2018 +6.0 (−3.8), 2019
+1.2 (+34.8), 2020 +10.2 (+26.0), 2021 −0.1 (+24.1), 2022 +14.0 (−19.4), 2023 −1.1 (+23.9), 2024
+24.7 (+16.4), 2025 +4.2 (+15.7), 2026 to Oct 16 +7.0 (+10.2).

Proposal pass bars: (a) T1 alpha > 0, P >= 0.90, >= 95% luck — PASS; (b) Jul-Sep 2026 loss <=
half of C_raw's — **FAIL on T1** (61%), pass on T2 (37%); (c) non-negative in >= 3 of 4 bear
episodes — PASS (3: COVID, 2022, 2025; Q4 2018 −4.4%); (d) T2 alpha > 0 — PASS.

**T4, read once** (`lab.py run resid_momentum U0_T4 '{}' T4 --costs2`; names 1001-1500, $3M
floor, 2022-01 → 2026-10-16, never looked at): alpha **+6.9%/yr (P 0.96)**, beta −0.07, vol
10.1%, max DD −11.2%, 231 orders/yr; late fills +5.8%, 2× costs +6.6%, top-5-out +4.2%; per
period −0.1, +19.0, +11.2, +8.8, +29.8, +21.0, +1.8, +9.0, **−10.3, −10.6** (both 2026 halves
negative); down months +1.0% vs the market's −4.0% (27 months); 2022 bear +7.7% vs −20.0%;
Feb-Apr 2025 +2.3% vs −22.3%; Jul-Sep 2026 −1.3% vs −4.2%.

## §8 Conclusions

*[Round 2: these are round 1's conclusions on lab v1, kept as written. The review found item 1
overstated even on v1, since G1 was a coin flip under late fills, 2× costs and the noise copies,
and item 4 leaning on a post-hoc window while pre-registered bar (b) had FAILED on T1 (61%). The
corrected conclusions are in "Round 2", R2.6.]*

1. **The pre-registered design passes every lab gate and replicates on the locked names
   1001-1500** (+6.9%, P 0.96), with |beta| < 0.1 and a real bear-market profile (positive in
   COVID, 2022 and Feb-Apr 2025; +1.0-1.2% a month when the market falls). Nothing was tuned.
2. **But the edge is a 2022-2025 phenomenon.** TRAIN (2016-07 → 2021-12) is flat (+0.09%,
   P 0.51; E0 −6.3%, T3 −1.5%), and 2026 is negative on T1-H2 and both T4 halves. The premise
   check on TRAIN found no rank IC (−0.003). The book wins where momentum as a whole won.
3. **Residual ranking beats raw on risk, not on alpha** (T1 +10.6% vs +9.8%, but max DD −17.7%
   vs −27.0% unscaled, beta −0.11 vs +0.11, Feb-Apr 2025 +4.8% vs −10.3%).
4. **The volatility scale is the useful crash protection**: same T1 alpha as no scaling, +1.4
   pts on T2, max DD halved (−17.7% → −9.1%), and Jul-Oct 2026 −3.5% vs −8.5% unscaled.
5. **Crashes are the short leg in junk/value rallies** (Nov 2016, Nov 2020, Q1 2021: short leg
   −5 to −8% while longs were flat or up). Neither sector-neutral residuals (I1: alpha unchanged,
   drawdowns lower, 2022's sector gain lost) nor beta-neutral sizing (I2: −2 pts) fixes it.
6. **Tuning fits the objective, not the edge** (PBO 0.62; the robust score bought less risk and
   lost 5.7 pts on T1).

## Review fixes (2026-10-23)

An adversarial review of §0-§8 reproduced U0 bit for bit and found one critical issue, one
major one and five minor ones. The sections above are kept as they were run, except for the
date in §6. Numbers below come from new labels; the old labels stay on disk for the record.

### R0 Pre-registration (written before any re-run; sha256 snapshots in `var/proposals_lab/results/resid_momentum/prereg/`)

**Critical: the engine drops entries at full gross.** Under `lock_cash` a short's sale proceeds
stay locked until it is covered. With `call_seq="auto"` a bar's sells (new shorts and long
exits) execute before its buys (covers and new longs). At gross 0.98 the book has about 2% free
cash before a rebalance. A new short sent on the same bar as the cover that pays for it
therefore gets that 2% or nothing ("NoCashShort"), and a new long can hit the same wall
("NoCashLong"). `rotation_targets` re-sends a target only when it changes, so the order is
never retried. The strategy believes the slot is held for months; the engine holds nothing or a
sliver. The reviewer's sign check found 840 / 234 / 22 / 184 / 193 position-days off target on
E0 / TB / T3 / T1 / T2. A partial fill (sign right, size about 2% of the book instead of 2.45%
per slot) escapes that check: in the synthetic case of the new engine test, the new short holds
−1.9% of the book instead of −24.5%.

*Fix (strategy-local; `src/` untouched):* the new param `phased: true` (the default) splits
every rebalance over two full rows. At the first open after the month end, what closes or
shrinks trades, and the old side of a flip closes. At the second open, what opens or grows
trades. The cash a cover releases is then free before any new position asks for it. Exits keep
their pre-registered timing; entries and increases fill one session later. `phased: false`
reproduces the original execution, so the old labels can be recomputed.
- Considered and not chosen: gross 0.90. It cuts the pre-registered exposure by 8%, and a month
  whose new shorts exceed the headroom (about 10% of the book) still loses entries.
- Considered and not chosen: retrying rejected orders. Only the engine can do it, so it stays a
  proposal: `_run_portfolio` should re-send a target while the position's sign differs from it,
  or place covers before new shorts under `lock_cash`.
- Not removed by the fix: vectorbt's free cash excludes unrealized P&L, while targets are a
  percentage of the marked-to-market book value. When held positions carry unrealized gains
  above the 2% headroom, the last entries can still fill only partly.

*Acceptance rule (mechanics, judged before any return is read):* `research.py audit` must show,
on every OPEN test, 0 rejected orders and a mean unheld gross ≤ 0.2% of the book for the phased
book. Otherwise `gross` drops one step to 0.95 and the audit is re-run, still before any return.
The old book is audited too, to confirm the review's counts. The audit prints order counts only.

**Re-runs.** New labels carry an `x` (phased execution); everything else is identical:
- U0x: `run … E0,TB,T3,T1,T2 --costs2`, then `noise` and `luck` (T1);
- C_rawx `{"score": "raw", "vol_target": 0}`, C_resx `{"vol_target": 0}`, I1x
  `{"factors": "market+sector"}`, I2x `{"sizing": "beta"}`;
- the tuner re-run on TRAIN as T150x (same 8-knob space as §3, 150 trials, TPE seed 42), its
  best trial run on OPEN;
- the leg attribution of U0x.

Re-running variants 1-6 under the corrected execution corrects them; it adds no new variants
(the count stays 6). `phased: false` vs `true` on U0 is a diagnostic.

**Decision rules (unchanged).**
- T150x is adopted only if it beats U0x on T1 and T2, paired.
- I1x or I2x is adopted only if its paired Δ vs U0x is ≥ 0 on TRAIN, T1 and T2, and its T1 Δ
  exceeds 2 × U0x's T1 noise sd.
- Final = U0x unless one is adopted.
- Then `verdict`, `bear`, and T4 read once more, for the corrected final only. This is the
  SECOND read of T4; the first, U0_T4 at +6.9%, was the uncorrected book. It is reported as the
  second read and selects nothing.

**Expected.** From the reviewer's retry estimate, which re-sent each rejected target one bar
later:
- TRAIN-stitched alpha falls about 0.3 pts, to about −0.2% (P ≈ 0.49), so G1 FAILS;
- T1 ≈ +10.7% and T2 ≈ +4.7%, unchanged within noise;
- T4 unknown.

If G1 fails, the status cannot be "works", and under the lab's definitions it cannot be "works
to some degree" either: that label requires G1.

**Major: the status was overstated.** G1 is now printed with its three noise copies and its
late-fill and 2×-cost TRAIN values. The status follows the lab's definitions literally, and "works
out of sample" is dropped.

**Minor issues.**
1. `order_notional`: the lab-wide $1,000 stays, since all six proposals are priced at it. The
   config comment's "~$1,000 orders on $40k" is corrected with U0x's measured mean order. G1, T1
   and T2 at 2× costs are reported as the bound. 2× costs doubles every cost; the reviewer's
   measured understatement of the $1 minimum was +0.19 to +0.56% of the book a year.
2. Split-adjusted price filter and `si_share`: documented as a limitation of the lab panels. They
   hold back-adjusted closes and volumes and no adjustment factor, so the price as traded at t
   cannot be recovered causally. The review measured the effect at +0.3 (T1) and −1.4 (T2) book
   points over 4.8 years, so it does not flatter the result. Not changed.
3. Pass bar (b) failed on T1 (61% of C_raw's Jul-Sep 2026 loss). It is now stated next to every
   crash-protection claim and in the README.
4. Dates corrected (§6). No git commit is authorized in this session, so each pre-registration
   step is snapshotted with its sha256 before the runs it governs.
5. Three causal explanations above were stated as findings without a test. Each is now tested
   and then kept, reworded or dropped:
   - "beta estimation error pushes over-estimated betas to the short side in a rising market":
     per decision, the short-minus-long ex-ante beta against the sign of the market's
     formation-window return, and each leg's ex-ante beta against its ex-post beta over the next
     126 bars;
   - "U0's 2022 gain was partly a sector bet (energy long)": U0x's 2022-bear P&L by sector and
     leg;
   - "most of a junk-rally loss is the losers' own rebound, not beta": the short leg's
     crash-window P&L split into β·m and residual parts.

### R1 Mechanics audit (order counts only; no return read)

`research.py audit` rebuilds each period's lab book with vectorbt's order log, and checks that
the rebuild equals the lab's own book on every period. Columns:
- **rejected**: orders refused for lack of cash;
- **partly**: filled below 99.5% of the shares asked;
- **off-target days**: name-days whose position has the wrong sign against a non-zero target
  (the review's measure);
- **mean unheld gross**: each shortfall, as a share of the book, counted until that name's next
  order and averaged over the bars.

| execution | E0 | TB | T3 | T1 | T2 |
|---|---|---|---|---|---|
| original (`phased: false`), gross 0.98: rejected / partly / off-target days / mean unheld gross | 10 / 9 / 840 / **5.0%** | 5 / 3 / 234 / 1.3% | 3 / 1 / 22 / 0.29% | 5 / 5 / 184 / 0.44% | 4 / 8 / 193 / 0.78% |
| phased, gross 0.98 | 1 / 8 / 43 / 0.64% | 0 / 1 / 0 / 0.07% | 6 / 1 / 21 / 0.40% | 0 / 3 / 0 / 0.04% | 0 / 3 / 0 / 0.15% |
| phased, gross 0.95 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 2 / 1 / 21 / 0.22% | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

The off-target days of the original execution match the review exactly. The partial fills
are new: on E0 an average 5% of the book's gross was never held.

Phasing removes the first mechanism, new shorts sent before the covers that fund them. What
remains at 0.98 is a second mechanism. Under `lock_cash`, vectorbt's free cash is
`V − Σ long value − Σ short value − 2 × Σ unrealized short gains`: a winning short keeps its
entry proceeds plus collateral locked. Targets, however, are a percentage of the
marked-to-market value `V`. So when held shorts carry gains, the second phase's last orders run
out of cash. These are the longs, which the engine buys last. Examples:
- T3, 2021-12-02: the book scale steps from 0.75 to 1.0, and 6 longs are rejected while 1 fills
  59%;
- E0, 2017-05-02: TTWO is rejected.

**Addendum to the acceptance rule (written after the 0.95 audit, before any return).** At 0.95,
T3 still fails the bar (2 rejections, 0.22%). Gross takes one more step, to **0.90**, the
review's own suggestion. If 0.90 passes the bar it becomes `config.yaml`'s gross for every `x`
label, with `phased: true`. If it fails, gross stays 0.90 and the residual shortfall is reported
next to every number. Gross < 0.98 shrinks the book only while the volatility scale sits at 1:
elsewhere `min(1, 10% / σ)` is computed on the unscaled book at that gross, and so compensates up
to the 0.25-step rounding. It shrinks the unscaled control arms (C_rawx, C_resx) by 8% throughout.

| execution | E0 | TB | T3 | T1 | T2 |
|---|---|---|---|---|---|
| **phased, gross 0.90 (the fix)** | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| same, raw arm unscaled (C_rawx: always at full gross) | 0 / 0 / 0 / 0 | 0 / 1 / 0 / 0.003% | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| same, beta sizing (I2x) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

**Accepted: `phased: true`, `gross: 0.90`** (`config.yaml`). Both mechanisms are pinned by
synthetic engine tests:
- `test_phased_execution_lets_the_engine_hold_every_target_at_full_gross`: the original
  execution holds −1.9% of a −24.5% short;
- `test_winning_shorts_lock_their_gains_so_the_gross_keeps_a_margin`: at 0.98 a phased entry
  fills 96%, at 0.90 fully.

`test_the_engine_runs_the_book_inside_its_gross_limit` now also asserts the engine's sign on
every bar. Every `x` label below runs with this config.

*[Round 2: the `x` labels ran on lab v1, but round 1 ended before their numbers were logged.
Lab v2 supersedes them (R2.0). The tests named above were renamed for lab v2's runner:
`test_a_new_short_needs_its_cover_first_phasing_or_the_lab_runner` and
`test_the_lab_runner_holds_every_target_inside_the_gross_limit`.]*

# Round 2: lab v2 re-evaluation (2026-10-23)

Round 1's numbers (§1-§8 and "Review fixes") came from lab v1. Lab v2
(`strategies/proposals_lab/README.md`, "Lab v2") fixes five lab-level defects that the round-1
reviews found:
- holiday rows in the 2026 panels;
- each period's day 0 dropped, with its opening costs and up-front borrow;
- new shorts starved by the engine's call order under `lock_cash` (this strategy's critical
  review issue);
- borrow mis-charged on hair-sized short re-trims;
- six sector ETFs typed "equity" inside the stock universes.

Lab v2 writes `var/proposals_lab/results2/` and `studies2/`. Round 1's files stay in
`results/`.

## R2.0 Pre-registration (written before any lab-v2 number of this strategy)

**What exists from round 1 after R1.** R1 accepted `phased: true, gross: 0.90`. Round 1 then
ran, under lab v1, U0x with its noise and luck, C_rawx, C_resx and a 150-trial tune T150x
(`results/resid_momentum/*x*`). Round 1 ended before they were logged. Lab v2 supersedes them
and they decide nothing; the v1 → v2 table quotes U0x for the record. Disclosure: before writing
this entry I printed `U0x.log` (lab v1 per-period alphas). No rule below depends on it.

**Mechanics first (no returns).** Lab v2's runner places a bar's orders in this order: long
cuts, short covers, new shorts, long buys. That removes R0's first mechanism, a new short sent
before the cover that funds it. R1's second mechanism is not an ordering problem. vectorbt's
free cash leaves out a winning short's gain twice, while targets are shares of the marked book.
So on a nearly full book the last orders of a bar can still be cut, and under v2 those are the
long buys. `research.py audit` now rebuilds each book on `lab.run_portfolio` (its call order and
borrow fees, day 0 from the initial cash, checked equal to `lab.book` within 1e-12). It runs on
the OPEN tests for:
- (a) gross 0.98, `phased: false`: §0's execution verbatim (the round-1 final, U0);
- (b) gross 0.90, `phased: false`;
- (c) gross 0.90, `phased: true`: R1's accepted fix;
- (d) gross 0.98, `phased: true`: diagnostic only.

**Rule.** R0's bar is unchanged: 0 rejected orders and a mean unheld gross ≤ 0.2% of the book
on every OPEN test. The round-2 base execution is the first of (a), (b), (c) that passes, in
that order. (a) is §0 verbatim. (b) keeps §0's timing; under v2, phasing only delays every entry
by a session. (c) is R1's. If none passes, the base is (c), with the shortfall printed next to
every number. `config.yaml` is set to the base.

**Labels and runs.** All runs are `lab.py run … E0,TB,T3,T1,T2 --costs2`, without `--hedge`
(the book never trades SPY).
- **U0v2** is the base. If the base is not (a), the §0-verbatim book also runs once on OPEN as
  **U0s0v2**. It is the round-1 final re-measured on v2, for the v1 → v2 table, and decides
  nothing.
- `lab.py noise` and `lab.py luck` (T1, 100 books) run for U0v2.
- The round-1 variants are re-measured on the base execution:
  - **C_rawv2** `{"score": "raw", "vol_target": 0}`;
  - **C_resv2** `{"vol_target": 0}`;
  - **I1v2** `{"factors": "market+sector"}`;
  - **I2v2** `{"sizing": "beta"}`;
  - **T150v2**: the tuner re-run on v2 TRAIN (`lab.tune`, the same 8-knob space as §3, 150
    trials, TPE seed 42, journal `studies2/resid_momentum-T150v2.log`), its best trial run on
    OPEN.
- Re-measuring round 1's six variants on corrected accounting adds no variant (6 of 8 used).
  Round 2 may add up to 3 new iterations (the workflow allows 10 variants in all), each
  pre-registered here before it runs.

**Decision rules (round 1's, unchanged).**
- T150v2 is adopted only if it beats U0v2 on T1 AND T2 (paired Δ > 0).
- I1v2 or I2v2 is adopted only if its paired Δ vs U0v2 is ≥ 0 on TRAIN (stitched), T1 and T2,
  and its T1 Δ exceeds 2 × U0v2's T1 noise sd.
- Final = U0v2 unless one is adopted. A final other than U0v2 is declared here before its T4
  read.
- Then `lab.py verdict resid_momentum <final>` (market-neutral, so G7 applies), `lab.py bear`,
  and T4 once for the final (`<final>_T4`; if the final is U0v2, that one book is U0v2's too).
  This is a disclosed SECOND read of T4, on corrected accounting; the first was U0_T4 (+6.9%)
  on lab v1. It selects nothing.
- Status follows the lab's definitions literally. "Works" = every gate; "works to some degree"
  = G1, G2, G3, G5 and G6. Anything else is neither, whatever the validation alphas.
- §0's proposal bars (a)-(d) are re-measured on U0v2 vs C_rawv2 and reported as pass/fail. Bar
  (b) is read on its pre-registered Jul-Sep 2026 window. The Jul 1-Oct 16 window is labelled
  post hoc wherever it appears.

**Review issues fixed in the documents.**
- The status claim.
- Bar (b)'s FAIL, stated in the README status box and next to every crash-protection claim.
- The cost premise (below).
- The dates.
- The two untested causal claims, tested with `research.py betas` and `windows` on U0v2, then
  kept, reworded or dropped.
- Live month-end detection, documented: it needs an exchange calendar.

**Cost premise (review minor 1).** `order_notional` stays $1,000, the lab's common order size
(R0's decision). `research.py orders` on U0v2 measures the median order of a $40k book. The
config comment then states the account size at which $1,000 is the median order. Every gate is
also printed at 2× costs, G1 included, as the bound for a smaller account.

**Expected (honest prior, before any v2 number).**
- Day 0 charges one book rebuild per half-year: −0.1 to −0.15 pt a year at base costs and about
  −0.5 under late fills, on every test.
- Filling the starved shorts: TRAIN about −0.3 pt (the reviewer's retry estimate, +0.09% →
  −0.20%), T1 and T2 about +0.1.
- Holiday rows: little change, since the strategy already skipped stub rows (`_full_rows`).
  ETF exclusion: T1's universe changes a little from 2024-H1.
- Net: TRAIN-stitched alpha about −0.2 to −0.6%, so **G1 FAILS**. T1 ≈ +10% (P > 0.98), T2
  ≈ +4%, and G2-G8 likely pass. With G1 failing, the status is neither "works" nor "works to
  some degree".
- T150v2, I1v2 and I2v2: not adopted, as in round 1.

## R2.1 Mechanics audit on lab v2's runner (order counts only; no return read)

`research.py audit` on `lab.run_portfolio`. Each cell gives rejected / partly filled /
off-target days / mean unheld gross. Every rebuild equals `lab.book` within 1e-12. T1 and T2
now have 1,202 bars instead of 1,205: the three holiday rows are gone.

| execution | E0 | TB | T3 | T1 | T2 |
|---|---|---|---|---|---|
| (a) gross 0.98, unphased (§0) | 1 / 9 / 44 / **0.71%** | 0 / 1 / 0 / 0.08% | **6** / 1 / 0 / **0.29%** | 0 / 2 / 0 / 0.02% | 0 / 4 / 0 / 0.17% |
| **(b) gross 0.90, unphased** | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| (c) gross 0.90, phased (R1) | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| (d) gross 0.98, phased | 1 / 8 / 43 / 0.64% | 0 / 1 / 0 / 0.07% | 9 / 1 / 0 / 0.40% | 0 / 3 / 0 / 0.04% | 0 / 3 / 0 / 0.15% |

For comparison, lab v1 with §0's execution (R1's first table) had 840 / 234 / 22 / 184 / 193
off-target days and 5.0% unheld gross on E0.

**Read.** Lab v2's call order removes the review's critical mechanism: E0's 840 off-target
days fall to 44 and its unheld gross from 5.0% to 0.71%. What is left at 0.98 is R1's second
mechanism, which phasing does not touch ((d) ≈ (a)):
- every rejection is code 7 (vectorbt's "NoCashLong"), i.e. a long buy, the bar's last order,
  with free cash 0;
- E0 still holds 0.71% of the book's gross short of its targets (TTWO rejected 2017-05-01);
- T3 rejects 6 full-scale longs (2.45% each) on 2021-12-01: MRNA, NUE, ORCL, PSA, TGT, WSM.

So (a) fails R0's bar on E0 and T3, and **(b) gross 0.90, `phased: false` is the round-2 base
execution** (U0v2). `config.yaml` is set to it. Phasing is off, so every order fills at the
first open after the decision, as §0 pre-registered. The `phased` option stays, documented as
the workaround for the shared engine's `call_seq="auto"` (a headless `tsim` run). The
§0-verbatim book (gross 0.98) also runs once on OPEN as **U0s0v2**, as pre-registered.

## R2.2 U0v2, the §0 book on v2, noise, luck, and the re-measured variants

All runs are `lab.py run … E0,TB,T3,T1,T2 --costs2` on lab v2. Late = +50 bp a leg.

| | E0 | TB | T3 | **TRAIN stitched (G1)** | T1 (P) | T1 beta / maxDD | T2 (P) | orders/yr T1 |
|---|---|---|---|---|---|---|---|---|
| U0, lab v1 (round 1) | −6.3% | +6.7% | −1.5% | +0.09% (P 0.51) | +10.6% (0.995) | −0.09 / −9.1% | +4.6% (0.905) | 240 |
| U0x, lab v1 (R1 fix, never logged) | −6.9% | +7.1% | −3.2% | −0.67% (P 0.43) | +9.8% (0.992) | −0.09 / −8.2% | +5.2% (0.925) | 243 |
| U0s0v2: §0 verbatim (0.98) on v2 | −7.9% | +6.1% | −1.3% | −0.77% (P 0.43) | +8.9% (0.984) | −0.10 / −10.3% | +4.9% (0.889) | 241 |
| **U0v2**: gross 0.90, unphased, v2 | **−8.2%** | **+6.5%** | **−2.9%** | **−1.37% (P 0.37)** | **+8.7% (0.982)** | **−0.10 / −9.5%** | **+5.5% (0.928)** | 256 |

U0v2 details:
- TRAIN under stress: late fills −3.25%, 2× costs −1.83%, the three noise copies −1.61%,
  −1.35% and −2.09%. G1 fails under every version.
- T1: late +6.90%, 2× costs +8.29%, top-5-out +6.41%, volatility 10.8%, ex-ante beta −0.11.
- T2: late +3.66%, 2× costs +5.02%, top-5-out +3.07%.
- T1 per period: +12.7, +6.1, +3.4, +2.0, **+48.0** (2024-H1), +4.1, +12.5, −3.0, +5.1,
  −17.9 (2026-H2).
- T2 per period: +20.3, +3.6, −5.3, −7.0, +28.7, +13.4, +7.2, +2.0, +2.9, −15.8.

**What moved, T1 / TRAIN / T2 in points.**
- Lab v1 → v2 at the same execution (U0 → U0s0v2): −1.8 / −0.9 / +0.3. Day 0 is counted now,
  which is one book rebuild a half-year plus each period's first session. The panels were
  rebuilt (NYSE sessions only, ETFs out, current DB). The band selection is path-dependent, so
  the holdings drift apart over the years.
- Gross 0.98 → 0.90 on v2 (U0s0v2 → U0v2, paired): −0.1 (P 0.46) / −0.6 (P 0.18) / +0.6
  (P 0.86). T3 alone moves −1.7. Two things differ there, and they were not separated. σ scales
  with gross, so the 0.25-step rounding of the scale falls on other months in 2020-21. And at
  0.98, six longs went unfilled on 2021-12-01.
- By the review's retry estimate (TRAIN +0.09% → −0.20%), the call order is about a third of
  the −0.9 at a fixed execution. The rest is day 0 and the panels; that split was not measured
  separately.

**Noise** (`lab.py noise U0v2`, 5 names dropped a period, 3 copies): sd E0 0.71, TB 0.20,
T3 0.20, **T1 0.38**, T2 0.56 pts. An iteration must clear a T1 Δ of **0.76** pts.
**Luck** (T1, 100 shuffled-name books in dollar-volume tenths): real +8.74% vs median +0.10%,
p95 +2.84%: percentile **100**.

**Re-measured variants**, paired Δ vs U0v2 with P(Δ > 0); the arm's own alpha in brackets:

| | TRAIN | T1 | T2 | decision (round-1 rule) |
|---|---|---|---|---|
| C_rawv2 (control) | +2.93 (0.77) [+1.56%] | −0.51 (0.46) [+8.22%, P 0.83] | −4.43 (0.21) [+1.07%] | control |
| C_resv2 (control) | +0.42 (0.60) [−0.95%] | +0.80 (0.62) [+9.53%, P 0.95] | −2.18 (0.10) [+3.32%] | control |
| I1v2 sector-neutral | +1.09 (0.79) [−0.29%] | +0.81 (0.63) [+9.54%, P 0.99] | **−2.11 (0.18)** [+3.39%] | **not adopted** (T2 Δ < 0) |
| I2v2 beta sizing | −1.12 (0.12) [−2.49%] | **−3.01 (0.03)** [+5.72%] | −1.45 (0.11) [+4.05%] | **not adopted** |

- I1's verdict is unchanged, but the reason moved. Round 1 failed it on T1 (−0.4). On v2 it
  clears the T1 bar (+0.81 > 0.76) and TRAIN, and fails on T2 (−2.1). That is one more sign
  that its effect is not stable across name sets.
- C_raw alone is positive on TRAIN (+1.56%), from TB's +15.9%. It loses T3 (−9.1%) and T2.
  T1 max drawdown is −25.1% vs U0v2's −9.5%, and beta +0.07.
- The volatility scale: against C_resv2 it costs T1 0.8 pt and gains T2 2.2 pts. It halves the
  drawdowns (T1 −18.8% → −9.5%, T2 −21.7% → −13.9%).

**Falling markets** (book, equal-weight market in brackets): Q4 2018 −4.8% (−18.2%), COVID
+4.0% (−37.0%), 2022 bear +11.8% (−24.5%), Feb-Apr 2025 +6.6% (−19.5%). Down months +1.17%
vs −3.98% (41 months, 76% positive).

**Momentum-crash windows** (book return, market in brackets):

| window | U0v2 | C_rawv2 | C_resv2 |
|---|---|---|---|
| Nov-Dec 2016 | −8.0% (+5.6) | −5.8% | −8.0% |
| Nov 2020 | −3.2% (+15.1) | −6.2% | −5.9% |
| Jan-Mar 2021 | −4.7% (+8.2) | −8.9% | −9.4% |
| **Jul-Sep 2026, T1 (bar b)** | **−3.6% (−1.6)** | **−8.7%** | −7.4% |
| Jul-Sep 2026, T2 (bar b) | −3.3% (−3.9) | −11.1% | −6.7% |
| Jul 1-Oct 16 2026, T1 (post hoc) | −5.5% (−1.0) | −17.2% | −11.1% |

**Bar (b) on v2: PASS**, at 42% of C_raw's loss on T1 and 29% on T2. On v1 it FAILED at 61%.
The flip comes from the control arm, not the strategy: C_raw's Jul-Sep 2026 T1 loss grew from
−3.6% to −9.4% at the same execution (gross 0.98; `results2/resid_momentum/diag/craw_days.py`
and `craw_starved.py`, outputs in `.out` files beside them):
- v1 dropped the window's first session, 2026-07-01, which is that period's day 0. On it,
  C_raw's opening book lost −1.95%.
- On the rebuilt panels its path-dependent bands held three shorts that rallied (IT, ZS, CRCL:
  −3.4 pts together) instead of seven other names (AMD, BMNR, BSX, FIG, INTC, STZ, VIAV).
- No order was starved: the shared engine and the lab runner give the same −9.42% there.

So the pass rests on path detail. It is read as fragile, not as crash protection established.

## R2.3 The tuner on v2 (T150v2)

`lab.tune resid_momentum T150v2 '{}' <§3's space> 150` on v2 TRAIN, TPE seed 42 (journal
`studies2/resid_momentum-T150v2.log`). It is the same 8-knob space as round 1, now on the base
execution.
- **Best trial:** `lookback` 210, `skip` 21, `beta_window` 126, `slots` 30, `q_in` 0.95,
  `q_out` 0.8, `vol_target` 0.06, `vol_window` 126. Robust score −1.40% (U0v2: −6.63%), mean
  period alpha +2.8% (U0v2: −0.06%), **PBO 0.738** (round 1: 0.615).
- It is nearly round 1's point again: `vol_target` sits on its lower bound and `slots` on its
  upper one. The robust objective (mean − ½ sd) buys a smaller, more spread book.

On OPEN, paired Δ vs U0v2 (P(Δ > 0)):

| | E0 | TB | T3 | TRAIN | T1 | T2 |
|---|---|---|---|---|---|---|
| T150v2 alpha | −3.8% | +6.6% | +2.8% | +2.21% (P 0.79) | +5.45% (P 0.989) | +5.47% (P 0.987) |
| Δ vs U0v2 | +4.4 (0.94) | +0.2 (0.52) | +5.7 (0.80) | +3.58 (0.92) | **−3.29 (0.12)** | **−0.03 (0.51)** |

T150v2 runs 367-458 orders a year (U0v2: 195-256), with max drawdowns −4% to −10% and beta
−0.06 to −0.08. **Not adopted**, as in round 1: it must beat U0v2 on T1 AND T2 and loses both.
Its positive TRAIN alpha is in-sample (the tuner fitted those 11 periods) and decides nothing.

## R2.4 Final candidate (declared 2026-10-23 08:57, before verdict, bear and T4 on v2; snapshot `prereg/R2.4_20261023-085726.md`)

**U0v2**: the pre-registered default with gross 0.90, unphased (R2.1's mechanics fix), on lab
v2. No round-1 decision changed: T150v2, I1v2 and I2v2 fail their bars, and no new iteration is
run. Lab v2 opens no obvious new direction. Its changes are accounting, and the only arm that
moved toward adoption, I1, now fails on T2 instead of T1. The variant count stays 6 of 8 (U0,
C_raw, C_res, T150, I1, I2, each re-measured on v2). U0s0v2 is the round-1 final re-measured
and is not a variant. Next:
- `lab.py verdict resid_momentum U0v2` (market-neutral);
- `lab.py bear resid_momentum U0v2 C_rawv2 C_resv2`;
- T4 once: `lab.py run resid_momentum U0v2_T4 '{}' T4 --costs2`. The final IS U0v2, so there is
  one T4 book. It is the second read of T4, disclosed; the first was round 1's U0_T4 on lab v1.

## R2.5 Verdict, bear table and T4 on v2 (U0v2 = the final)

**`lab.py verdict resid_momentum U0v2`: G1 FAILS; G2-G8 pass.**

| gate | value | pass |
|---|---|---|
| G1 TRAIN-era (E0+TB+T3) alpha > 0 | −1.37% (P 0.37). Late fills −3.25%, 2× costs −1.83%, noise copies −1.61 / −1.35 / −2.09% | **no** |
| G2 T1 alpha > 0, P >= 0.90 | +8.74% (P 0.982) | yes |
| G3 T2 alpha > 0 | +5.50% (P 0.928) | yes |
| G4 T1 beats >= 95% of shuffled books | 100th percentile (median +0.10%, p95 +2.84%) | yes |
| G5 T1 late-fill / 2× costs alpha > 0 | +6.90% / +8.29% | yes |
| G6 every bear episode beaten, down months > 0 | all four; +1.17% a month in 41 down months (market −3.98%) | yes |
| G7 \|T1 beta\| <= 0.3 | −0.10 | yes |
| G8 T1 alpha without its top 5 names | +6.41% | yes |

**Status, by the lab's definitions (R2.0): neither "works" nor "works to some degree".** Both
labels require G1. What the gates do show is an edge confined to 2022-2026, on the validation
names and on the locked names, and absent in 2016-2021.

**`lab.py bear resid_momentum U0v2 C_rawv2 C_resv2`** (E0+TB+T3+T1 stitched; %):

| | Q4 2018 | COVID crash | 2022 bear | Feb-Apr 2025 | CAGR | max DD |
|---|---|---|---|---|---|---|
| U0v2 | −4.8 | +4.0 | +11.8 | +6.6 | 3.1 | −20.5 |
| equal-weight market | −18.2 | −37.0 | −24.5 | −19.5 | | |
| SPY | −18.7 | −33.4 | −24.1 | −18.6 | | |
| C_rawv2 | −7.4 | +7.7 | +19.6 | −7.4 | 5.7 | −29.3 |
| C_resv2 | −4.8 | +5.3 | +16.0 | +5.8 | 3.6 | −28.4 |

Calendar years, U0v2 (market in brackets): 2016-H2 −15.8 (+9.0), 2017 −2.6 (+24.3), 2018 +3.8
(−2.8), 2019 +0.7 (+35.6), 2020 +13.4 (+25.7), 2021 −2.5 (+21.5), 2022 +11.7 (−18.6), 2023 −1.3
(+22.7), 2024 +23.7 (+15.3), 2025 +3.6 (+16.2), 2026 to Oct 16 +1.8 (+11.7).

**Proposal pass bars (§0) on v2.**
- (a) T1 alpha > 0 with P >= 0.90 and >= 95% luck: PASS.
- (b) Jul-Sep 2026 loss <= half of C_raw's: PASS on T1 (42%) and T2 (29%). It FAILED on v1
  (61%), and the flip is the control arm's path (R2.2), so it is fragile.
- (c) non-negative in >= 3 of the 4 bear episodes: PASS (COVID, 2022, Feb-Apr 2025; Q4 2018
  −4.8%).
- (d) T2 alpha > 0: PASS.

**T4, second read** (`lab.py run resid_momentum U0v2_T4 '{}' T4 --costs2`, 2026-10-23 08:58,
disclosed). Alpha **+6.24%/yr (P 0.949)**, beta −0.07, vol 10.1%, max DD −13.6%, 211 orders a
year; late fills +4.44%, 2× costs +5.81%, top-5-out +3.63%.
- Per period: +1.0, +24.6, +10.6, +5.9, +30.2, +13.8, +3.9, +0.2, **−10.4, −8.5**. Both 2026
  halves are negative again.
- Down months +1.05% vs the market's −4.14% (26 months). 2022 bear +10.6% vs −18.4%. Feb-Apr
  2025 +3.8% vs −22.4%. Jul-Sep 2026 −0.8% vs −4.0%.
- The first read (U0_T4, lab v1) was +6.87% (P 0.96); paired Δ −0.67 (P 0.29).
- It selects nothing.

## R2.6 Round 2 summary

### v1 → v2, the headline numbers

| | round 1: U0, lab v1 | U0s0v2: §0 verbatim, lab v2 | **U0v2: the final, lab v2** |
|---|---|---|---|
| execution | gross 0.98, engine call order | gross 0.98, lab call order | gross 0.90, lab call order |
| TRAIN alpha, G1 (P) | +0.09% (0.51) | −0.77% (0.43) | **−1.37% (0.37)** |
| TRAIN late / 2× costs | −1.18% / −0.17% | −2.84% / −1.27% | −3.25% / −1.83% |
| TRAIN noise copies | −0.71 / +0.09 / −0.92% | — | −1.61 / −1.35 / −2.09% |
| T1 alpha (P) | +10.61% (0.995) | +8.85% (0.984) | **+8.74% (0.982)** |
| T1 late / 2× / top-5-out | +9.55 / +10.37 / +8.26% | +7.02 / +8.41 / +6.36% | +6.90 / +8.29 / +6.41% |
| T1 beta / max DD | −0.09 / −9.1% | −0.10 / −10.3% | −0.10 / −9.5% |
| T1 luck percentile | 100 | — | 100 |
| T2 alpha (P) | +4.55% (0.905) | +4.85% (0.889) | **+5.50% (0.928)** |
| T4 alpha (P) | +6.87% (0.961), 1st read | — | **+6.24% (0.949)**, 2nd read |
| gates passed | 8 / 8 | — | **7 / 8 (G1 fails)** |
| Q4 2018 / COVID / 2022 / Feb-Apr 2025 | −4.4 / +4.3 / +14.2 / +3.5% | −4.5 / +4.3 / +11.8 / +4.7% | −4.8 / +4.0 / +11.8 / +6.6% |
| equal-weight market, same | −18.2 / −37.0 / −25.2 / −19.6% | −18.2 / −37.0 / −24.5 / −19.5% | same as v2 |
| down months, book / market | +1.22 / −3.97% | +1.25 / −3.98% | +1.17 / −3.98% |
| bar (b), T1 vs C_raw | 61%: FAIL | — | 42%: PASS (fragile) |
| status | "works" (overstated) | — | **neither label: G1 fails** |

### Review issues → fix → effect

1. **Critical: new shorts starved by the engine's call order.**
   - Fix: lab v2's runner places covers before new shorts. The audit on it (R2.1) showed a
     second mechanism at gross 0.98: winning shorts' gains are locked twice in vectorbt's free
     cash, which cut the bar's last long buys (E0: 0.71% of the gross never held; T3: 6 longs
     rejected). So gross 0.98 → 0.90, decided on order counts before any return, and phasing is
     off.
   - Effect: 0 rejected or partly filled orders on every OPEN test. TRAIN alpha +0.09% →
     −0.77% (lab v2 at 0.98) → −1.37% (0.90). **G1 now fails**, and the status changes from
     "works" to neither label. T1 moves −1.9 pts, mostly v2's accounting (−1.8 at a fixed
     execution). T2 moves +1.0, mostly gross 0.90 (+0.6, P 0.86) (R2.2).
2. **Major: the status was overstated.**
   - Fix: the status follows the lab's definitions literally. G1 is printed with its late,
     2×-cost and noise-copy versions everywhere. "Works out of sample" is gone.
   - Effect: README status box and conclusions rewritten. On v2, G1 fails under every version.
3. **Minor: the order_notional premise.**
   - Fix: measured (`research.py orders`). A $40k book's median order is $420-900 excluding
     the opening rebuilds (T1 $452, T2 $663, T3 $418), so $1,000 is the median order of a
     ~$45k-95k account. At $40k the $1 minimum costs 0.04-0.36% of the book a year more. The
     config comment says so, and the 2×-cost runs (−0.45 to −0.5 pt) bound it.
   - Effect: no number changes. G1 at 2× costs is −1.83%.
4. **Minor: the $5 filter and `si_share` read split-adjusted data.**
   - Fix: not fixable on the lab panels, which carry no adjustment factor, so the price as
     traded at t cannot be rebuilt causally. It is listed among lab v2's known caveats. The
     review measured +0.3 (T1) and −1.4 (T2) book points over 4.8 years, so it does not
     flatter the result.
   - Effect: documented only.
5. **Minor: failed bar (b) hidden behind a post-hoc window.**
   - Fix: (b) is stated next to every crash-protection claim and in the README. The Jul 1-Oct
     16 window is labelled post hoc.
   - Effect: on v2, (b) PASSES (T1 42%). The flip is the control arm's path: day 0 plus three
     shorts that rallied (R2.2). It is reported as fragile.
6. **Minor: the order of pre-registration rests on document order.**
   - Fix: §6's date was corrected in round 1. Every round-2 pre-registration (R2.0, R2.1,
     R2.4) is snapshotted with sha256 and a timestamp in `results2/resid_momentum/prereg/`
     before the runs it governs. A git commit was not authorized in this session.
   - Effect: none on the numbers.
7. **Minor: causal claims stated as findings.** Each was tested on U0v2 (`research.py betas` /
   `windows`, outputs in `results2/resid_momentum/diag/`):
   - (i) "beta estimation error pushes over-estimated betas to the short side in a rising
     market": **refuted.** The shorts' betas are not over-estimated (ex-ante 1.17, ex-post 1.18
     after rising-market formations). The short-minus-long ex-ante gap is LARGER after falling
     markets (0.65 vs 0.22; correlation with the formation market −0.38). What estimation
     error does is under-estimate the LONGS' beta after rising markets (0.95 ex-ante vs 1.12
     ex-post), so the book is less net-short than its ex-ante beta says.
   - (ii) "a downside-risk scale would change nothing": withdrawn, never tested.
   - (iii) "the 2022 gain was partly a sector bet (energy long)": **holds, with a correction.**
     Of the 2022 bear's +9.8% (costs ignored), half is net short beta (+5.1: the shorts' β·m
     +10.6 against the longs' −5.6) and half residual (+4.7), almost all of it the energy
     longs (residual +4.9).
   - (iv) "most of the short-leg loss in a junk rally is the losers' own rebound, not beta":
     **refuted** for Nov 2020 and Jan-Mar 2021. The short leg's ex-ante β·m part is 65% and
     62% of its loss there, and 46% in Nov-Dec 2016. Only Jul-Oct 2026 is residual.
   - Effect: the claims in §4-§5 are superseded by these readings.
8. **Minor: live month-end detection.**
   - Fix: documented in `_month_ends` and the README. A month's last bar is known only once
     the next one exists, which is exact in the backtest; a live run needs an exchange
     calendar or it decides a session late.
   - Effect: none on the backtests.

### Decisions, v1 → v2

- **Changed:**
  - G1, pass → FAIL.
  - Status, "works" → neither label.
  - Bar (b), FAIL → PASS (fragile).
  - Execution: gross 0.98 → 0.90 and phasing off, decided on mechanics. Round 1's R1 had
    chosen 0.90 with phasing.
- **Unchanged:**
  - The final stays the untuned default.
  - T150v2 is not adopted (T1 −3.3, T2 −0.0).
  - I1v2 is not adopted, now failing on T2 (−2.1) instead of T1.
  - I2v2 is not adopted (T1 −3.0).
  - No new iteration was run: v2's changes are accounting and open no obvious direction.

### Conclusions (they replace §8)

1. **The pre-registered design fails G1 on corrected accounting.** TRAIN-era alpha
   (2016-07 → 2021-12) is −1.37% (P 0.37), negative under late fills, 2× costs and in all
   three noise copies. The TRAIN premise check found no rank IC (−0.003). By the lab's
   definitions the status is neither "works" nor "works to some degree".
2. **On 2022-2026 the edge is real by every other gate.**
   - T1 +8.7% (P 0.98, 100th luck percentile, top-5-out +6.4%), T2 +5.5% (P 0.93), locked T4
     +6.2% (P 0.95, second read). |beta| ≤ 0.1.
   - All four bear episodes are beaten, with +1.0-1.2% a month when the market falls.
   - It is a regime: negative before 2022, and negative in 2026-H2 on T1, T2 and both 2026
     halves of T4.
3. **Round 1's T1 +10.6% was about 1.8 pts too high.** The paired comparison at a fixed
   execution (U0 → U0s0v2) puts that on lab v1's accounting: day 0 dropped and the old panels.
   The starved orders were real but small for T1 and T2. The reviewer's retry estimate was
   about +0.1 pt there, and +0.09% → −0.20% on TRAIN.
4. **Residual vs raw ranking: about the same T1 alpha at a much smaller drawdown.** C_raw: T1
   −0.5, T2 −4.4, T1 max DD −25.1% vs −9.5%, Feb-Apr 2025 −7.4% vs +6.6%. Raw wins TRAIN
   (+1.6%) on TB alone.
5. **The volatility scale halves the drawdowns** (T1 −18.8% → −9.5%). It costs T1 0.8 pt and
   adds 2.2 on T2. Pre-registered bar (b) passes on v2 at 42% but rests on the control arm's
   path, so the crash-protection claim is the drawdown halving, not (b).
6. **The momentum crashes are the short leg in junk rallies**, and about half to two thirds of
   that leg's loss is its ex-ante beta times the market's jump. Neither sector-neutral
   residuals (I1) nor beta-neutral sizing (I2) fixes it without losing alpha elsewhere.
7. **Tuning fits the objective, not the edge** (PBO 0.74; T1 −3.3 pts).
