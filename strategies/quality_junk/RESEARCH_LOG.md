# quality_junk — research log

Proposal 4 of `STRATEGY_OVERVIEW.md` §6, evaluated under the shared protocol of
`strategies/proposals_lab/README.md` (mandatory). Every number below comes from `lab.py` on the
lab's panels; nothing was read from the database. Variant counter at the end of each section.
**Round 2 (§13, 2026-10-23) re-ran everything on lab v2 and supersedes every round-1 number below;
§0-§12 are kept as the record.**

## §0 Pre-registration (2026-10-22, written before any number was computed)

**Hypothesis.** In the lab's point-in-time universes (top 500, names 501-1000), a book that is
long the most profitable (within sector) and least volatile names and short the least profitable,
most volatile names, sized so the two sides carry about the same market beta, earns positive alpha
(CAGR − β × equal-weight buy-and-hold CAGR), and earns it disproportionately in falling markets.

**Mechanism.**
1. *Lottery demand and leverage constraints.* Investors overpay for volatile, unprofitable,
   lottery-like stocks, and leverage-constrained investors bid up high-beta / high-volatility names
   instead of levering safe ones (Frazzini & Pedersen 2014, betting against beta; Bali, Cakici &
   Whitelaw 2011; Asness, Frazzini & Pedersen 2019, quality minus junk). Short-sale costs keep the
   overpricing of junk from being arbitraged away.
2. *Underpriced profitability.* Gross profit ÷ assets predicts returns as strongly as value does
   (Novy-Marx 2013); the market under-weights durable profitability.
3. *Flight to quality.* In a crisis, money leaves junk first: the short side (high beta, losing
   money) falls more than the long side even after beta-neutral sizing, so the book should beat
   the market in drawdowns (QMJ earned most in 2000-02, 2008 and 2022).

**Literature-default config (U0) — nothing tuned.**
- Universe: the lab's test universe of each half-year period; a name is *eligible* on a bar when its
  close ≥ $5, its profitability is defined and its 252-bar volatility exists (≥ 126 returns).
- Profitability: gross profit TTM ÷ total assets (assets > 0); where gross profit is missing (banks,
  insurers, some utilities/telecoms) net income TTM ÷ equity (equity > 0). This hybrid is ranked
  WITHIN sector (the lab's `sector` codes; a name without a sector is its own group), so the sleeve
  is not a sector bet. Never an AND-filter across measures: a name needs one of the two.
- Safety: minus the 252-bar standard deviation of daily close-to-close returns, ranked across all
  eligible names.
- Score: the equal-weight mean of the two percentile ranks, re-ranked across the eligible names.
- Book: long the top tenth (enter at rank ≥ 0.90, leave below 0.80), short the bottom tenth (enter
  ≤ 0.10, leave above 0.20), 20 + 20 slots, best candidates first. Re-ranked ONLY at month ends
  (decided at the last session's close, filled at the next open); between month ends the book is
  held. Daily risk exits for shorts only, copied from informed_flow_ls_p1 v6: the squeeze stop
  (+100% from the entry close; not re-shorted until it has left the short zone) and the crowded
  exit (short interest ÷ shares outstanding > 30%; also never entered).
- Sizing: informed_flow_ls_p1's `_weights` with `sizing: beta` — each slot sized AT ENTRY by the
  bar's median beta ÷ the name's trailing 126-bar beta to the equal-weight eligible names, clipped
  to [1/1.5, 1.5] of the equal slot (gross 0.98 × 0.5 ÷ 20 a side), frozen while held, new entries
  scaled down together if they would lift nominal gross above 0.98. Long cap `retrim_k` 2.
- Borrow: informed_flow_ls_p1's tiers — 0.5% a year, 3% above 10% of shares short, 10% above 20%;
  0.5% where there is no short-interest data (E0 before 2018).
- Costs: the repo's model at `order_notional` $1,000, open fills.

**Expected result (written before any number).**
- Premise check (TRAIN): composite rank IC against beta-adjusted next-month returns small and
  positive, +0.01 to +0.04 (the repo's sector-demeaned gp/ROE hybrid read +0.048, t 1.84 on
  2021-24; low volatility read weak to negative in the 2021-24 mega-cap window). The safety sleeve
  is the bigger risk; profitability the better prior.
- U0: TRAIN-era alpha −2% to +4%/yr (2016-21 rewarded growth and junk; the 2020-H2/2021 junk rally
  should be the worst stretch). T1 (2022-26) alpha +2 to +5%/yr with P(alpha > 0) 0.6-0.9:
  strongly positive in 2022, negative in the 2023-24 mega-cap rally. T2 (names 501-1000, more junk
  to short) similar or better. Bear episodes: ahead of the market in Q4 2018, 2022 and Feb-Apr
  2025; COVID uncertain (quality won the crash, junk won the rebound). Beta in [−0.3, +0.1].
- Survivorship (the DB holds no delisted company) makes the junk short leg look WORSE than it was:
  a bias against this strategy. *[§13.8: it also makes the long leg look better — failed
  "quality" names are missing too; the net direction is unknown.]*

**Pass bar.** The lab's gates G1-G8 (`lab.verdict`, market-neutral, so G7 applies).
"Works" = all of them; "works to some degree" = G1, G2, G3, G5 and G6.

**Protocol and budget.** Premise check on TRAIN only (informational). U0 on OPEN with late fills
and 2× costs, `noise` (5 names dropped, 3 copies), `luck` on T1 (100 shuffled-name books).
Optimization: `lab.tune` on TRAIN only (≥ 120 trials, robust objective, PBO recorded), adopted only
if it beats U0 on T1 AND T2 (paired). At least two further iterations, each pre-registered below
with its mechanism before it is run, adopted only if its paired Δ against the current base is ≥ 0 on
TRAIN, T1 and T2 AND its T1 Δ exceeds twice the noise sd. Budget: at most 8 variants through the
bar (U0 and the tuned candidate included; tuner trials and the noise/luck copies excluded). The
final candidate is declared here before `verdict`, `bear` and the single read of T4 (final vs U0).
TL is never read.

Variants used: 0.

## §1 Data audit and one outcome-blind amendment to U0 (2026-10-22, still no return computed)

`research.py coverage` (identity only: which names carry which data, per OPEN period):
- 90-95% of each period's names are eligible (E0 0.90-0.93, TB 0.94, T3 0.94-0.95, T1 0.91-0.95,
  T2 0.90-0.95). The AND-collapse of `value_factor` does not happen.
- Gross profit / assets covers 0.54-0.67 of names; the ROE fallback carries 0.27-0.38 — far more
  than banks (XBRL filers that never tag gross profit or cost of revenue). Short interest is absent
  before 2018 (E0's first three periods), 0.82-1.00 after.
- The two measures are MIXED inside most sectors (one bar of TB 2019-H1: Consumer Discretionary
  76 / 20, Utilities 13 / 20, Real Estate 10 / 10, Energy 8 / 5, Finance 8 / 42) and on different
  scales (gross profit / assets median 0.30, ROE median 0.12 *[§13.1: per-period medians
  0.26-0.32 and 0.10-0.15; 0.12 is the average over periods]*). Ranking the raw hybrid within a
  sector would therefore rank most ROE-measured names below most gross-profit names — a
  measurement artifact that would steer the short side into ROE-measured names, not junk.

**Amendment (before any number about returns).** A name's profitability rank is its percentile
among the eligible names of its SECTOR MEASURED THE SAME WAY (gross profit / assets or ROE); when
that (sector, measure) group holds fewer than 5 eligible names on a bar, the name is ranked against
every eligible name on its measure instead. Everything else in §0 stands. The literal hybrid
ranking is still computed in the premise check, for information only.

## §2 Premise check — TRAIN only (E0 + TB + T3, 60 month-ends; `research.py premise`)

Rank IC (Spearman) and top-minus-bottom-tenth spread of each signal at a month end against the
next 21 sessions' beta-adjusted return from the next open (trailing 252-bar beta to the
equal-weight universe):

| signal | IC | t | tenth spread / month | t |
|---|---|---|---|---|
| composite score (U0) | +0.032 | 1.46 | +0.18% | 0.26 |
| profitability, within sector × measure (U0) | +0.026 | 2.18 | +0.68% | 1.85 |
| profitability, literal hybrid within sector | +0.027 | 2.34 | +0.67% | 1.87 |
| profitability, raw hybrid (not sector-neutral) | +0.039 | 2.27 | +0.91% | 1.85 |
| safety (−252-bar volatility) | +0.020 | 0.76 | −0.78% | −0.85 |

Composite IC by year: 2016 (H2) −0.146, 2017 +0.085, 2018 +0.060, 2019 +0.049, 2020 −0.049,
2021 +0.122; the safety sleeve drives the swings (2016 −0.168, 2020 −0.105, 2021 +0.136).

Reading (informational): as pre-registered, profitability is the steadier sleeve (t ≈ 2 on both
IC and spread). Safety ranks the middle of the cross-section right (IC > 0) but its extremes wrong
in this era: the most volatile tenth beat the calmest tenth by 0.8%/month even after beta
adjustment, and the two bad years are the junk rallies (2016-H2, the 2020 rebound). The §1
amendment changes almost nothing in IC terms (+0.026 vs +0.027). Expectation for U0 from this: a
book whose TRAIN alpha is held back by the short side's high-volatility names.

## §3 U0 — the untuned default (variant 1)

`lab.py run quality_junk U0 '{}' E0,TB,T3,T1,T2 --costs2` (31 half-year books, late fills and 2×
costs; ~110-125 orders a year — *§12.6: of which only 30-43 are the strategy's own rotation; the
rest is each half-year book buying its 40 names on its first bar*):

| test | alpha | P(α>0) | beta | CAGR | B&H | vol | max DD | late α | 2× costs α | top-5-out α | ex-ante β |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | −5.4% | 0.04 | −0.13 | −7.8% | +18.4% | 6.3% | −16.9% | −5.7% | −5.5% | −7.0% | −0.18 |
| TB | +5.1% | 0.85 | +0.06 | +5.7% | +10.5% | 9.0% | −10.6% | +4.7% | +5.0% | +0.8% | −0.15 |
| T3 | +14.1% | 0.93 | −0.22 | +5.8% | +37.3% | 11.2% | −16.4% | +13.9% | +14.0% | +8.8% | −0.24 |
| **TRAIN stitched** | **+0.9%** | 0.67 | −0.01 | +0.6% | +20.2% | 8.8% | −18.4% | | | | |
| **T1** | **+1.2%** | 0.58 | **−0.32** | −1.5% | +8.6% | 14.0% | −26.6% | +0.9% | +1.1% | −1.3% | −0.40 |
| **T2** | **−0.3%** | 0.48 | −0.13 | −1.3% | +7.7% | 10.3% | −23.0% | −0.7% | −0.4% | −3.0% | −0.19 |

Per-period alphas — E0: −8.1, −1.3, +2.4, −7.3; TB: +16.6, +23.5, +2.1, −2.9; T3: +1.6, +13.2,
+29.5; T1: +12.3, +6.4, −5.2, −7.8, +18.3, +5.6, +7.5, −11.3, +21.8, +10.4; T2: −3.3, +2.9, −6.8,
+8.3, +5.6, −7.3, +3.5, −14.7, +11.4, +25.0.

Reading: no edge yet. Costs are irrelevant (monthly, low turnover). *[Corrected in §12.1: the
lab drops each half-year's first day; charging the trades one continuous book makes at the
boundaries, late fills cost 0.8 point of T1 alpha and G5 fails.]* The book is NOT beta-neutral:
ex-ante −0.15 to −0.40 and T1 realized −0.32 (fails G7). Per-name sizing clipped at 1.5× cannot
offset the gap between calm longs and volatile shorts, so U0 is a net short-beta book that lost
money in absolute terms in T1 (CAGR −1.5%) and is credited for it only through the alpha's
−β × B&H term.

**Noise** (`lab.py noise quality_junk U0`: 5 names dropped a period, 3 copies; alpha Δ vs U0):
E0 sd 0.6 pt, TB 0.5, T3 1.9, **T1 1.2** (Δ −1.2, +1.0, −1.0), T2 0.24 (Δ −1.0, −0.5, −0.7). An
iteration's T1 Δ must therefore exceed **+2.4 points** to be adopted.

**Luck** (`lab.py luck quality_junk U0`, T1, 100 books trading U0's own weights on names shuffled
inside dollar-volume tenths): real +1.2%, shuffled median +0.2%, 95th percentile +5.5% — U0 sits at
the **63rd percentile**. No selection skill shown.

## §4 Optimization — pre-registered space (before any tuning number)

`lab.tune` on TRAIN (E0, TB, T3; 11 periods), 120 TPE trials, robust objective (mean − ½ sd of the
period alphas), over U0's own numeric knobs only (design changes are iterations, §5+):
`w_prof` 0-2 (step 0.25; `w_safe` = 1 is the numeraire, so 0 = safety alone), `vol_window`
63-504 (63), `q_in` 0.80-0.95 (0.05), `q_out` 0.60-0.85 (0.05; invalid pairs pruned), `slots`
10-40 (5), `rebalance` month | quarter, `beta_window` 63-252 (63), `max_size` 1.0-3.0 (0.25).
Adopted only if the best trial beats U0 on T1 AND T2 (paired).

**Result** (116 complete trials, 4 pruned; PBO **0.54**): best robust score +0.085 (mean period
alpha +14.0% in-sample) at `w_prof` 0.25, `vol_window` 63, `q_in` 0.85, `q_out` 0.85 (equal to
q_in: the float 0.8500000000000001 slipped past `valid()`'s strict q_out < q_in — no hysteresis),
`slots` 10, `rebalance` month, `beta_window` 126, `max_size` 1.0 (equal slots). The surface: 82 of
116 trials chose `vol_window` 63 (mean score +0.037; 126: +0.009; ≥ 315: < 0); `w_prof` 0.25-0.5 best
(0: −0.025; ≥ 1.25: ≤ 0). U0's region (w_prof 1, vol 252, 20 slots) is poor on TRAIN.
*[§13.4: TPE sampled `vol_window` 63 in 82 trials (a sampling count, not a vote); the 116 trials
held 96 distinct sets; `w_prof` 1.25 averaged +0.001, only 1.5 and 1.75 were below 0.]*

Label **T** (variant 2), OPEN with late fills and 2× costs:

| test | alpha | P | beta | CAGR | maxDD | paired Δ vs U0 | P(Δ>0) | periods better |
|---|---|---|---|---|---|---|---|---|
| TRAIN stitched | +2.3% | 0.75 | −0.28 | −3.4% | −36.4% | E0 +5.3, TB +1.7, T3 +3.1 | 0.86/0.66/0.68 | 3/4, 4/4, 2/3 |
| T1 | +8.6% | 0.84 | **−0.79** | +1.8% | −37.1% | **+7.4** | 0.94 | 6/10 |
| T2 | −9.9% | 0.10 | −0.61 | −14.6% | **−64.2%** | **−9.6** | 0.02 | 4/10 |

**Not adopted** (fails T2). The tuned book is a net SHORT-beta bet (equal-dollar calm longs against
volatile shorts, ex-ante β −0.5 to −1.0): its T1 "alpha" is mostly the −β × B&H credit (−0.79 ×
8.6%), it beats every bear episode (+9.7%, +3.7%, +8.6%, +15.8%) the way any −0.8 beta book does,
and it loses 64% peak to trough on names 501-1000. A warning for every per-period-alpha tuner:
half-year alphas credit a negative beta generously in strong half-years, so the robust objective
leans toward short-beta books.

## §5 What U0 is made of (diagnostics, `research.py anatomy`, `correlate.py`)

Leg contributions by year (sum of daily realized weight × return, percentage points):

| | 2016-H2 | 2017 | 2018-H2 | 2019 | 2020-H2 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| long (top 500) | −1.1 | +7.1 | +2.7 | +15.1 | +12.4 | +16.2 | −3.8 | +2.8 | +7.0 | +4.1 | +8.4 |
| short (top 500) | −4.9 | −13.0 | +5.7 | −10.2 | −16.6 | −2.0 | +14.5 | −21.2 | −0.5 | −12.9 | −0.2 |
| market | +8.7 | +22.3 | −6.4 | +30.7 | +26.7 | +22.7 | −18.0 | +22.6 | +16.1 | +16.5 | +10.7 |
| long (T2) | | | | | | | −5.6 | +10.4 | +9.9 | +4.4 | +6.5 |
| short (T2) | | | | | | | +6.5 | −11.8 | −12.2 | −9.1 | +2.8 |

Mean exposure on T1: long $0.65 at beta exposure +0.27, short $0.33 at −0.65 (ex-ante net −0.38).
The short side is a market hedge: it loses 10-21 points in every rising year and pays only in the
falling ones. *[Corrected in §12.3: in five of the ten rising periods; it lost 0.2-4.9 in the other
five. Every period is shown there.]* Net of its beta (−0.65 × market) it earned about +2 points a year on T1 and ~0 on T2;
the long side about +1 (T1) and +2 (T2). Sector tilt (mean weight, T1): long Utilities 14%, Consumer
Discretionary 14%, Industrials 10%, Staples 8%, Finance 6%, Technology 1%; short Technology −11%,
Discretionary −6%, Finance −6%, Industrials −6%. The book is partly a sector bet.

**Against the short-interest book** (`informed_flow_ls_p1` v6, the lab's own saved run; read only):
daily-return correlation −0.31 (TB), −0.47 (T3), −0.43 (T1), −0.25 (T2); correlation of the two
books' market residuals −0.38, −0.40, −0.36, −0.23. They worked in different eras (QMJ TB +5.1%,
T3 +14.1%; short interest TB −2.6%, T3 +3.4%, T1 +13.1%). A daily 50/50 blend: TB alpha +1.4%
(vol 4.7%, max DD −5.1%), T3 +9.1% (5.2%, −7.4%), **T1 +7.5% (P 0.996, vol 7.0%, max DD −7.9%,
beta −0.08)**, T2 +2.0% (6.4%, −9.4%).

## §6 Iterations — pre-registered together, run one at a time against the current base

Each is adopted only if its paired Δ vs the base is ≥ 0 on TRAIN (stitched), T1 and T2 AND its T1
Δ exceeds 2 × the base's T1 noise sd (U0: 2 × 1.2 = 2.4 points).

- **I1 — BAB-style side neutrality (`sizing: legs`).** Mechanism: U0's net short beta comes from
  the short side (beta exposure −0.65 vs the long side's +0.27), which makes the junk leg a market
  hedge that loses 10-21 points in every rising year. Scaling the two SIDES so that
  D_L β̄_L = D_S β̄_S (Frazzini & Pedersen's BAB construction; set at month ends, a held name
  re-sized only when its slot size moved > 25%) cancels it and moves the dollars to the long side
  (~0.8 / 0.2), the side whose beta-adjusted contribution was steadier. Prediction: |T1 beta| < 0.1,
  max drawdown down, alpha Δ between −1 and +3 points (smaller 2022 gain, smaller 2023 loss).
- **I2 — short-window safety (`vol_window: 63`).** Mechanism: lottery demand and volatility
  persistence are short-horizon — next month's volatility is best forecast by the last quarter's,
  and the idiosyncratic-volatility and MAX anomalies are measured over the last month (Ang, Hodrick,
  Xing & Zhang 2006; Bali, Cakici & Whitelaw 2011). The TRAIN surface (§4) chose 63 in 82 of 116
  trials *[§13.4: TPE sampled it in 82]*, so TRAIN Δ > 0 is expected by construction; the test is
  whether it carries to T1 and T2.
- **I3 — sector-neutral safety (`safe_by_sector: true`).** Mechanism: U0's sides are a sector bet
  (long utilities and staples, short technology and discretionary, §5), so sector rotations — the
  2023 tech rally (short leg −21 points), 2016-H2, 2020-H2 — dominate its tails; ranking volatility
  within sector keeps the stock-level lottery effect and drops the sector bet (QMJ works within
  industries: Asness, Frazzini & Pedersen 2019). Prediction: lower volatility, a smaller 2023 loss
  and a smaller 2022 gain; alpha Δ ≥ 0 on T1.

Variants used: 2 (U0, T).

### I1 result (variant 3): NOT adopted

| test | alpha | P | beta | maxDD | orders/yr | paired Δ vs U0 | P(Δ>0) | better |
|---|---|---|---|---|---|---|---|---|
| E0 | −6.3% | 0.03 | +0.03 | −13.0% | 217 | −0.9 | 0.16 | 2/4 |
| TB | +0.1% | 0.61 | +0.10 | −22.6% | 157 | −5.1 | 0.19 | 1/4 |
| T3 | +9.8% | 0.86 | −0.08 | −15.1% | 215 | −4.3 | 0.01 | 0/3 |
| TRAIN stitched | −1.2% | 0.48 | +0.06 | −25.9% | | | | |
| T1 | +0.5% | 0.53 | **−0.02** | −17.8% | 162 | −0.8 | 0.29 | 2/10 |
| T2 | +0.2% | 0.53 | +0.03 | −15.1% | 150 | +0.5 | 0.64 | 6/10 |

Episodes (book vs equal-weight market): Q4 2018 +3.8% (U0 +5.5%), COVID crash −12.7% (U0 −6.3%),
2022 bear −6.2% (U0 −0.4%), Feb-Apr 2025 +2.4% (U0 +7.4%); down-month mean +0.04% (U0 +1.27%).

The mechanism did what it was built to do — beta within ±0.1 on every test *[TB +0.10; lab v2:
−0.08 to +0.10]*, max drawdown down on
T1 and T2 — and removed the alpha with it: once the book is beta-neutral, QMJ earns ≈ 0 on every
test (−1.2%, +0.5%, +0.2%). U0's bear-market profile was its net short beta, not quality: a
beta-neutral QMJ lost 12.7% in the COVID crash and 6.2% in the 2022 bear. *[Corrected in §12.2:
I1 still beat the falling market in all four episodes (it passes G6); beta explains U0's 2022-bear
and Feb-Apr 2025 gains, while Q4 2018's cushion (+6.5 points beyond beta) is quality's.]*

Stitched paired Δ (`show.py delta U0 ...`, the bar's three numbers): **I1** TRAIN −2.1 (P 0.11),
T1 −0.8 (P 0.29), T2 +0.5 (P 0.64). For the record, **T**: TRAIN +1.4 (P 0.71), T1 +7.4 (0.94),
T2 −9.6 (0.02).

### I2 result (variant 4): NOT adopted

| test | alpha | P | beta | maxDD | paired Δ vs U0 | P(Δ>0) | better |
|---|---|---|---|---|---|---|---|
| E0 / TB / T3 | −1.7% / +7.0% / +6.9% | | −0.20 / +0.05 / −0.25 | | +3.7 / +1.9 / −7.2 | 0.94 / 0.81 / 0.02 | 4/4, 3/4, 0/3 |
| TRAIN stitched | +0.6% | 0.64 | −0.03 | −24.0% | −0.3 | 0.44 | |
| T1 | +2.9% | 0.70 | −0.34 | −29.5% | +1.7 | 0.81 | 5/10 |
| T2 | −3.8% | 0.18 | −0.25 | −36.7% | −3.6 | 0.05 | 2/10 |

Episodes: Q4 2018 +3.1%, COVID −5.4%, 2022 +2.9%, Feb-Apr 2025 +8.0%. The TRAIN surface's
preference for 63 bars does not survive even the stitched TRAIN record (−0.3: the tuner's
per-period objective and the stitched alpha disagree), T1's +1.7 is under the 2.4 bar and the
fresh names lose 3.6 points.

### I3 result (variant 5): NOT adopted

| test | alpha | P | beta | vol | maxDD | paired Δ vs U0 | P(Δ>0) | better |
|---|---|---|---|---|---|---|---|---|
| E0 / TB / T3 | −7.9% / +6.3% / +12.9% | | −0.11 / +0.04 / −0.20 | | | −2.5 / +1.2 / −1.1 | 0.05 / 0.58 / 0.35 | 1/4, 2/4, 1/3 |
| TRAIN stitched | +0.4% | 0.62 | −0.02 | 7.5% | −20.1% | −0.5 | 0.32 | |
| T1 | +1.3% | 0.62 | −0.26 | 12.3% | −25.9% | +0.1 | 0.51 | 3/10 |
| T2 | +0.3% | 0.53 | −0.12 | 9.2% | −18.3% | +0.6 | 0.62 | 6/10 |

Episodes: Q4 2018 +4.4%, **COVID −1.0%** (U0 −6.3%), **2022 +2.9%** (U0 −0.4%), Feb-Apr 2025
+6.2%. Sector-neutral safety is a RISK improvement (lower volatility everywhere, smaller T2
drawdown, better in the two worst crashes) with no alpha in it.

## §7 Two more iterations (pre-registered before running; variants 6-7 of the budget)

**Growth premise first (TRAIN only, informational).** QMJ's growth dimension — the change in the
name's own profitability ratio over 252 bars, ranked within sector × measure (`w_growth`, built
off by default, test-pinned to leave U0 bit-identical) — reads IC **−0.010 (t −1.15)** and tenth
spread −0.11%/month (t −0.39) on the 60 TRAIN month-ends: the wrong sign. No variant is spent on it.

- **I4 — profitability alone (`w_safe: 0`).** Mechanism: of U0's two sleeves only profitability
  has a steady premise (IC +0.026, t 2.18; tenth spread +0.68%/month, t 1.85), while safety's tails
  are wrong (−0.78%/month) and safety is what makes the book a net short-beta bet (§5). A
  within-sector gross-profitability long/short is Novy-Marx's within-industry result and the repo's
  never-built sector-demeaned hybrid (IC +0.048, t 1.84 on 2021-24). Prediction: |beta| smaller
  than U0's, TRAIN alpha ≥ U0, alpha Δ on T1 between −2 and +4 points.

### I4 result (variant 6): NOT adopted

| test | alpha | P | beta | vol | maxDD | top-5-out | paired Δ vs U0 | P(Δ>0) | better |
|---|---|---|---|---|---|---|---|---|---|
| E0 / TB / T3 | −1.0% / +2.2% / +0.6% | | −0.02 / +0.05 / −0.04 | 4.2 / 5.2 / 6.3% | | | +4.4 / −3.0 / −13.5 | 0.86 / 0.30 / 0.03 | 2/4, 1/4, 0/3 |
| TRAIN stitched | −0.2% | 0.48 | +0.03 | 5.2% | −10.7% | | −1.1 | 0.33 | |
| T1 | −0.9% | 0.42 | −0.12 | 7.9% | −23.0% | −3.2% | −2.1 | 0.30 | 4/10 |
| T2 | **+4.8%** | **0.93** | −0.03 | 7.5% | −15.3% | **+0.4%** | **+5.1** | 0.86 | 7/10 |

T2 periods: −5.0, +11.3, +12.5, +15.8, +8.6, +8.1, +1.3, −5.3, −7.4, +40.8 (the last a 3.5-month
stub). Episodes: Q4 2018 −1.8%, COVID +0.6%, 2022 +2.0%, Feb-Apr 2025 +3.8%. As predicted the
beta all but vanishes (the safety sleeve made U0 net short), and so does the TRAIN and top-500
alpha; profitability earns only on names 501-1000, and there mostly through five names.

**Not run** (reasons, not results): the growth sleeve (wrong-sign premise, §7); the "junk short leg
alone + SPY hedge" — an SPY long against an equal-weight market adds the cap-weight-minus-equal-
weight spread (SPY beat the equal-weight top 500 by ~10 points a year in 2023-24) to the alpha,
which would confound any reading of the junk leg; `safety: beta` — under U0's sizing it can only
deepen the net short beta that I1 showed to be the whole bear-market profile.

## §8 Final candidate — declared before verdict, bear table and T4

No iteration met the adoption bar (T: fails T2; I1: TRAIN and T1 < 0; I2: TRAIN and T2 < 0, T1
under 2.4; I3: TRAIN < 0, T1 +0.1; I4: TRAIN and T1 < 0), so the protocol's base is unchanged: the
**final candidate is U0** (config.yaml as shipped, overrides `{}`). Its T4 read is therefore one
run (final = U0). Variants used: 6 of 8 (U0, T, I1, I2, I3, I4).

## §9 Verdict, bear table, T4 (read once)

*[Lab v1 numbers; superseded by §13.2 and §13.7 (lab v2), where the final candidate is TRv2.]*

`lab.py verdict quality_junk U0` (market-neutral gates):

| gate | value | passed |
|---|---|---|
| G1 TRAIN-era alpha > 0 | +0.9% | yes |
| G2 T1 alpha > 0, P ≥ 0.90 | +1.2%, P 0.58 | no |
| G3 T2 alpha > 0 | −0.3% | no |
| G4 T1 beats ≥ 95% of shuffled books | 63rd percentile | no |
| G5 T1 late-fill and 2× cost alpha > 0 | +0.9% / +1.1% | yes — *§12.1: −0.04% / +0.58% on the boundary reading: FAIL* |
| G6 every bear episode beaten, down months > 0 | +1.27% a down month | yes |
| G7 \|T1 beta\| ≤ 0.3 | −0.32 | no |
| G8 T1 alpha without its top 5 names > 0 | −1.3% | no |

`lab.py bear` (stitched E0 + TB + T3 + T1; %): episodes Q4 2018 / COVID / 2022 / Feb-Apr 2025 —
market −18.2 / −37.0 / −25.2 / −19.6; SPY −18.7 / −33.4 / −24.1 / −18.8; **U0 +5.5 / −6.3 / −0.4 /
+7.4** (CAGR −0.4%, max DD −26.6%); T +9.7 / +3.7 / +8.6 / +15.8; I1 (beta-neutral) +3.8 / −12.7 /
−6.2 / +2.4; I2 +3.1 / −5.4 / +2.9 / +8.0; I3 +4.4 / −1.0 / +2.9 / +6.2; I4 −1.8 / +0.6 / +2.0 /
+3.8. Calendar years, U0 vs market: 2016-H2 −6.1 / +8.8, 2017 −5.9 / +24.6, 2018 +4.4 / −3.8, 2019
+4.8 / +34.8, 2020 −6.2 / +26.0, 2021 +14.1 / +24.1, **2022 +10.5 / −19.4**, 2023 −17.8 / +23.9,
2024 +6.8 / +16.4, 2025 −8.2 / +15.7, 2026 +4.2 / +10.2.

**T4** (names 1001-1500, locked, read once, `lab.py run quality_junk U0 '{}' T4 --costs2`): alpha
**+2.6%** (P 0.74), beta −0.09, CAGR +1.7% vs B&H +9.8%, vol 9.8%, max DD −18.8%, late fills
+1.6%, 2× costs +2.5%, top-5-out +0.2%. Periods: −6.6, −1.8, −2.7, +5.5, +2.8, +19.3, +8.7, +1.5,
−9.2, +33.5 (stub). Positive, not significant, and like T1 it rests on a handful of names.

## §10 Conclusion

*[Lab v1; the round-2 conclusions are §13.9.]*

**No edge.** Quality minus junk, built as the literature specifies on this repo's point-in-time
universes, earns an alpha indistinguishable from zero on every validation set (TRAIN +0.9% P
0.67, T1 +1.2% P 0.58, T2 −0.3% P 0.48, T4 +2.6% P 0.74), sits at the 63rd percentile of
shuffled-name books, and fails five of eight gates. "Works to some degree" needs G2 and G3: no.

What it is instead: **a market hedge**. Its bear-market record (ahead of the market in all four
episodes, +10.5% in 2022) is the net short beta of a book that is long calm stocks and short wild
ones — the short side carries ~0.65 of beta exposure against the long side's ~0.3. Neutralize the
beta (I1) and the crash protection is gone (COVID −12.7%, 2022 −6.2%) and the alpha with it.
*[Corrected in §12.2 and §12.8: any low-beta book beats a falling market here (I1 and I4 pass G6);
beta explains U0's 2022-bear and Feb-Apr 2025 gains, quality cushioned Q4 2018 and lost beyond
beta in COVID and the 2022 bear, and calendar 2022's gain came in its fourth-quarter rally.]*
Neither the tuner (PBO 0.54; its pick, a −0.8 beta book, lost 64% on names 501-1000) nor four
pre-registered design changes found alpha that survives T1 and T2 together.

Partial effects worth carrying, none of them an edge by the bar:
- **Sector-neutral safety (I3)** is a pure risk improvement: lower volatility on every test, COVID
  −1.0% and 2022 +2.9%, same alpha.
- **Profitability alone (I4)** is beta-free (|β| ≤ 0.12) and earns on names 501-1000 (+4.8%,
  P 0.93) but not on the top 500 (−0.9%) or in TRAIN (−0.2%), and T2's number is +0.4% without
  its top five names.
- **Diversification.** QMJ's market residuals are negatively correlated with the short-interest
  book's (−0.36 on T1, −0.38 to −0.40 on TB/T3, −0.23 on T2), and the two worked in different eras.
  A daily 50/50 blend with `informed_flow_ls_p1` v6 halves the drawdown (T1 −7.9% vs −14.9%) and
  turns 2018-21 positive (T3 +9.1%, TB +1.4% against the SI book's +3.4%, −2.6%), but on T1 and
  T2 it lowers alpha per unit of volatility slightly (T1 1.07 vs 1.09).

## §11 Lab issues met (worked around; lab.py not edited)

1. `lab.py run <key> <label> '<overrides>' --costs2` reads `--costs2` as the tests list (`a[4]`)
   and fails with a KeyError; pass the tests explicitly (`... '{}' E0,TB,T3,T1,T2 --costs2`).
2. `lab.luck`'s cache key hashes the strategy's full params, so adding an inert knob to
   `config.yaml` after the luck run makes `lab.py verdict` silently recompute all 1,000 books
   (~15 min; the result reproduced exactly: 63rd percentile).
3. `lab.tune`'s float steps produce values like 0.8500000000000001, which let `q_in == q_out`
   through a strict `q_out < q_in` check (the tuned pick has no hysteresis band). Rounding
   suggestions to the step would avoid it.
4. The robust objective scores per-period (half-year) alphas, each with its own beta; in strong
   half-years a negative beta earns a large alpha credit (mean period alpha +6.3% for U0 against
   a stitched TRAIN alpha of +0.9%). The tuner duly picked a −0.8 beta book that then failed T2.
   Scoring the stitched TRAIN alpha, or penalizing |beta|, would make the optimizer's pick
   comparable to what the gates read.

## §12 Review fixes (fix agent, 2026-10-23)

An adversarial review reproduced U0 bit for bit (T1 re-run: every daily return of the book, the
benchmark, the late-fill and the 2× cost series identical; sample periods of every other label
too) and passed the look-ahead test, but found the claims not supported as written: two major and
five minor issues (the review text handed to this fix stops after the seventh issue; a later one,
if any, was not visible). This section answers each: issue → fix → new numbers. Old labels and
logs are kept as the record. New scripts are `research.py` subcommands; their logs are in
`var/proposals_lab/results/quality_junk/fix/`.

### §12.0 Pre-registration of the fixes (written 2026-10-23 00:35, before any new number)

No new variant: U0 stays the final candidate (variants used stay 6 of 8), T4 is NOT re-read (it was
read once for U0, the candidate is unchanged), TL is never read.

- **R1 (major) — the lab drops each half-year's first day.** `lab.book` takes returns as
  `value.pct_change().fillna(0)`, so the first day of every period (the 40 setup orders' fees,
  slippage and late-fill +50 bp; the borrow the engine charges on day-0 orders for shorts held to
  the period end; the first open-to-close move) is missing for the book AND the benchmark. Fix,
  strategy-local (`lab.py` is not edited): `research.py boundary` re-runs U0's books on the OPEN
  tests (base, late fills, 2× costs) and rebuilds every record three ways. *lab*: as saved.
  *full*: the first day restored for both (value[0] / init_cash − 1) — a full 40-name rebuild and a
  500-name benchmark rebuild every six months, an upper bound on costs. *boundary* (the fair one):
  the first day charges only what ONE continuous book would trade at the boundary — names entering,
  leaving or changing side (a name held on the same side across the boundary is free), the
  benchmark's universe change and re-weighting to 1/N — plus the day-0 borrow charge; the first
  day's market move stays omitted for both, as in the lab. The top 500 is one continuous book from
  2016-07 (E0 → TB → T3 → T1: only 2016-07 pays a full setup); T2 starts with a full setup in
  2022-01. **Decision rule:** G1, G2, G3 and G5 are re-read on the *boundary* records and that
  reading replaces the lab's in the verdict; *full* is reported as the upper bound. **Expected:** a
  boundary trades 30-45% of a full rebuild, so T1's late-fill alpha falls from +0.9% (lab) to about
  0 to +0.5% (G5 borderline), with *full* at −0.6% (the review's number); base alphas move by less
  than 0.3 point; no adoption decision of §4-§7 flips (each rejected variant failed one of its
  three conditions by more than 0.3 point).
- **R2 (major) — the bear-market mechanism was overstated.** Re-read `lab.verdict` for I1 and I4
  (saved results) and split U0's down-month mean and episode returns into the part its stitched
  beta implies (β × the market's) and the rest. Reword the headline, G6, the learnings and the
  bear reading: any near-zero-beta book beats a falling market here; U0's extra cushion is its
  net short beta.
- **R3 (minor) — the leg table was cherry-picked.** Re-run `research.py anatomy` on every OPEN
  test, show every period, and add each leg's return net of its own beta exposure.
- **R4 (minor) — `decision_bars` ignored exchange holidays on the panel's last bar.** Use an NYSE
  holiday calendar (pandas' holiday rules; no new dependency). Tests at Good Friday 2024, Memorial
  Day 2021 and New Year 2023. Expected: no backtest changes (the last bar's decision is filled after
  the panel ends; checked bit-exact on sample periods).
- **R5 (minor) — point-in-time leaks of the data.** Sectors are today's codes; the $5 floors (the
  lab's universe filter and `min_price`) read split-adjusted closes. Count the names `min_price`
  removes on decision bars per OPEN period; document both as caveats and lab issues.
- **R6 (minor) — turnover wording.** From R1's re-run: rotation orders vs the lab's first-bar
  orders, the boundary orders of a continuous book, realized gross (max, mean).
- **R7 (minor) — stale gross profit preferred over fresh ROE.** Measure, at every OPEN period
  start, how many GP/A-measured eligible names and how many HELD names carry a GP TTM level
  unchanged for more than 200 days, and how many of them have a fresh ROE. Documented, not changed
  in U0 (a fallback rule would be a new variant whose reach, 0-1 of 40 held names per the review,
  cannot clear the 2.4-point bar); parked in IDEAS.md.

### §12.1 R1 — the first day of each half-year (major): G5 now FAILS, at zero

`research.py boundary U0` (log `fix/boundary-U0.log`, records `fix/boundary-U0.pkl`) re-ran all
31 OPEN books three times (base, late fills, 2× costs) with the current code: every daily return
equals the saved one (max |diff| 0.0, so R4's code change is verified on every OPEN book too). The
decomposition of each first day is exact (rebuilt from the engine's own fills: the check ratio is
1.000 in every period). Stitched alpha, %/yr:

| reading | TRAIN | late | 2× | T1 (P) | late | 2× | T2 | late | 2× |
|---|---|---|---|---|---|---|---|---|---|
| lab (as saved: first day dropped) | +0.90 | +0.53 | +0.81 | +1.22 (0.58) | +0.89 | +1.14 | −0.29 | −0.68 | −0.37 |
| full (first day restored: a 40-name rebuild every 6 months, upper bound) | +0.04 | −1.36 | −0.29 | +0.73 (0.55) | −0.97 | +0.32 | −0.60 | −2.14 | −0.96 |
| **boundary (pre-registered decision reading)** | **+0.63** | −0.02 | +0.48 | **+0.78 (0.55)** | **−0.04** | **+0.58** | **−0.83** | −2.07 | −1.12 |
| boundary + the first day's market move | +0.21 | −0.45 | +0.05 | +0.94 (0.57) | +0.12 | +0.74 | −0.52 | −1.76 | −0.81 |
| boundary, benchmark at base costs | +0.63 | −0.01 | +0.48 | +0.78 | +0.07 | +0.61 | −0.83 | −1.98 | −1.10 |
| low (only the names that left the universe + one regular month-end rotation) | +0.66 | +0.10 | +0.53 | +0.85 | +0.29 | +0.71 | −0.73 | −1.55 | −0.92 |

The last three rows are sensitivities added after the first boundary run (not pre-registered);
they do not replace the decision reading. Per TRAIN test on the boundary reading: E0 −5.6% (lab
−5.4%), TB +4.9% (+5.1%), T3 +13.7% (+14.1%).
(The review's full-restore late number, −0.61%, kept the benchmark at base costs on its first day;
here the benchmark pays the stress it is built with, as in `lab.book`.)

What the first day holds, per test (TRAIN / T1 / T2):

| | TRAIN | T1 | T2 |
|---|---|---|---|
| lab orders a year = first-bar + rotation | 111 = 81 + 30 | 121 = 85 + 36 | 128 = 85 + 43 |
| one continuous book: rotation + boundary orders | 54 = 30 + 24 | 72 = 36 + 36 | 108 = 43 + 65 |
| boundary turnover ÷ a full rebuild | 0.29 | 0.39 | 0.76 |
| … of it in names that left the universe | 7% | 13% | 21% |
| a regular month-end rebalance trades (share of the book) | 7.3% | 7.1% | 8.4% |
| benchmark boundary turnover ÷ its rebuild | 0.27 | 0.30 | 0.56 |
| borrow charged on day-0 orders (lost by the lab), points a year | 0.20 | 0.32 | 0.35 |

Reading. The pre-registered expectation was half right: G5 is borderline and does fail on the
decision reading (T1 late fills −0.04%), but the base alphas moved by more than the 0.3 point
expected (TRAIN −0.27, T1 −0.44, T2 −0.54): the day-0 borrow (0.2-0.35 point a year) was not in
the estimate. Costs DO matter: on the boundary reading late fills cost 0.82 point of T1 alpha and
2× costs 0.20 (the lab's reading said 0.33 and 0.08). G5's sign at zero depends on defensible
accounting choices (+0.07 with the benchmark at base costs, +0.12 with the first day's move,
+0.29 on the low estimate, −0.97 with full rebuilds); under the pre-registered rule it is a FAIL.
The boundary turnover between two lab books (39% on T1) is probably an upper estimate for a truly
continuous book: each half-year's book is re-formed from 2015 on a different universe, so its
history differs, while a book that carried its positions would trade about the 13% that left the
universe plus a regular month (the *low* row). G1 still passes (+0.63%), G2 and G3 still fail.
No adoption decision of §4-§7 can flip: even if a rejected variant needed no correction at all,
its paired Δ would improve by at most U0's correction (0.27 / 0.44 / 0.54) — T still fails T2
(−9.6), I1 TRAIN (−2.1) and T1, I2 T2 (−3.6), I3 the 2.4-point T1 bar (+0.1), I4 TRAIN (−1.1).

### §12.2 R2 — the bear-market mechanism (major): beta explains 2022 and 2025, not Q4 2018

`lab.py verdict` on the saved I1 and I4 (log `fix/verdict-I1-I4.log`): both PASS G6. G6 compares
a book's absolute return with the market's, so any book whose beta is well below 1 passes it in a
falling market. `research.py bear` (log `fix/bear-split.log`) splits each episode by the book's
realized beta on its test (the beta part = test beta × the market's move; the rest ≈ alpha):

| episode (market) | U0 book | beta part | rest | I1 (β-neutral) book | rest | I4 (prof. alone) book |
|---|---|---|---|---|---|---|
| Q4 2018 (−18.2%) | +5.5% | −1.0 (TB β +0.06) | **+6.5** | +3.8% | +5.7 | −1.8% |
| COVID crash (−37.0%) | −6.3% | −2.1 (β +0.06) | **−4.2** | −12.7% | −8.9 | +0.6% |
| 2022 bear (−25.2%) | −0.4% | +8.1 (T1 β −0.32) | **−8.5** | −6.2% | −6.6 | +2.0% |
| Feb-Apr 2025 (−19.6%) | +7.4% | +6.3 (β −0.32) | +1.1 | +2.4% | +2.1 | +3.8% |
| 41 down months (−3.97%) | +1.27% | +0.62 (β −0.16) | +0.65 | +0.04% | +0.15 | +0.09% |

Calendar 2022 (U0 +10.5% vs −19.4%) is not a bear-market win either: from 2022-01-03 to 10-12 the
book lost 8.5 points net of its beta, and from 10-13 to 12-31 it made +11.0% against the market's
+7.8% (+13.5 net of beta; I1 +11.8). So: U0's gains in the 2022 bear and in Feb-Apr 2025 are its
net short beta (T1 β −0.32); Q4 2018 is the one episode with a quality cushion beyond beta (+6.5
points; the beta-neutral I1 +5.7); in the COVID crash and the 2022 bear quality lost beyond its
beta (U0 −4.2 and −8.5 points, I1 −8.9 and −6.6). Made beta-neutral (I1) the book still beat the
falling market in every episode — that is what near-zero beta does — but lost 12.7% in COVID and
6.2% in the 2022 bear in absolute terms, and its down-month mean fell from +1.27% to +0.04%.

### §12.3 R3 — the junk short leg, every period (minor)

`research.py anatomy '{}' E0,TB,T3,T1,T2` (log `fix/anatomy-U0.log`; sums of daily realized
weight × return, points; the market column is the same sum for the equal-weight market):

| top 500 | 16-H2 | 2017 | 18-H1 | 18-H2 | 2019 | 20-H1 | 20-H2 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | −1.1 | +7.1 | −1.7 | +2.7 | +15.1 | −3.8 | +12.4 | +16.2 | −3.8 | +2.8 | +7.0 | +4.1 | +8.4 |
| short | −4.9 | −13.0 | −2.0 | +5.7 | −10.2 | +2.6 | −16.6 | −2.0 | +14.5 | −21.2 | −0.5 | −12.9 | −0.2 |
| market | +8.7 | +22.3 | +3.8 | −6.4 | +30.7 | +2.7 | +26.7 | +22.7 | −18.0 | +22.6 | +16.1 | +16.5 | +10.7 |

Names 501-1000 (T2), 2022 → 2026: long −5.6, +10.4, +9.9, +4.4, +6.5; short +6.5, −11.8, −12.2,
−9.1, +2.8; market −14.2, +21.6, +13.8, +14.6, +9.6.

Corrected statement: on the top 500 the short leg lost 10-21 points in five of the ten rising
periods (2017, 2019, 2020-H2, 2023, 2025) and 0.2-4.9 in the other five; it gained in 2018-H2
(+5.7), the COVID half-year 2020-H1 (+2.6) and 2022 (+14.5). On names 501-1000 it lost 9-12 points
in 2023-25 and gained in 2022 (+6.5) and in rising 2026 (+2.8). Its ex-ante beta exposure is −0.46
(TB) to −0.65 (T1) (−0.52 on T2); the long side's +0.27 to +0.37. Net of each leg's own realized
beta, points a year: E0 long −4.6 / short −1.0, TB +1.2 / +4.1, T3 +7.5 / +6.5, T1 +1.0 / +1.8,
T2 +1.8 / −0.2 — "+1 to +2 a year" holds on T1 only.

### §12.4 R4 — exchange holidays in `decision_bars` (minor): fixed

`strategy.next_session` steps over the NYSE's holidays (pandas' holiday rules: New Year (Saturday
not moved to Friday), MLK, Presidents, Good Friday, Memorial, Juneteenth from 2022, Independence,
Labor, Thanksgiving, Christmas) and the two one-off closures since 2015; `decision_bars` uses it for
the panel's last bar. New test `test_the_panels_last_bar_is_a_month_end_on_the_exchange_calendar`
(fails on the old `BDay` code): a panel cut at 2024-03-28, 2021-05-28 or 2022-12-30 sees the same
decision bars as the full panel. Backtests are unchanged, as expected (R1's re-run: every OPEN book
bit-identical).

### §12.5 R5 — point-in-time leaks in the data (minor): documented

`research.py floor` (log `fix/floor.log`). The strategy's `min_price` removes 0-2.2 names a decision
bar on the top 500 and 0-4.7 on T2, mostly names whose ADJUSTED close sits under $5 because of
later dividends or splits (PBR in 9 of 21 top-500 periods; NVDA in 2018-H2 and 2019-H1; F, M,
AGNC, CVNA, MARA, SOFI, RIOT, LUMN, ACHR, SBET, SNAP). Upstream, the lab's universe floor drops NVDA
from the top 500 in six of the eight 2016-07 → 2020-01 periods (split-adjusted median under $5),
KGC in three and CVNA / NU in 2023. Both floors read closes adjusted back from today, so
eligibility depends on later splits and dividends; NVDA's absence also lowers the TRAIN-era
benchmark (a lab issue). Sector codes are today's, one per name, applied to 2015-2026. Neither can
be fixed in the strategy without unadjusted prices; both are caveats now.

### §12.6 R6 — turnover wording (minor)

The "~110-125 orders a year" mixed the lab's book rebuild into the strategy's turnover. U0's own
rotation is 30 (TRAIN) / 36 (T1) / 43 (T2) orders a year (month-end band exits and entries, the
squeeze and crowded exits, the 2× retrims; a regular month-end rebalance trades about 7% of the
book); the other 81-85 are each half-year book buying its 40 names on its first bar. One
continuous book would place about 54 / 72 / 108 a year (R1's table). Gross is 0.98 nominal AT
ENTRY, never a hard cap: positions are frozen while prices move and shorts are never re-trimmed,
so realized gross averages 0.99 (0.97-1.00 by test) and reached 1.26 (TRAIN), 1.31 (T1) and 1.22
(T2) at its highest.

### §12.7 R7 — stale gross profit (minor): measured, U0 kept

`research.py staleness` (log `fix/staleness.log`), on every month-end decision bar of every OPEN
window (wider than the pre-registered period starts): 2.8% (T1), 3.5% (E0), 4.8% (T2), 6.6% (T3) and 8.5% (TB) of the eligible names measured on
gross profit / assets carry a GP TTM level unchanged for more than 200 days — up to 781-1,161 days
(`features.py` masks a level only when no filing has refreshed it for 800 days; a level can stay
unchanged far longer while filings keep refreshing it) — and about three quarters of them have a
fresh ROE. The book
holds 0.6 (T1), 0.8 (E0), 1.6 (T2), 2.2 (TB) and 3.8 (T3) of them on average out of 40, at most 6.
The README's "a TTM figure can be up to a few months stale" was wrong. Falling back to ROE for
them does not help the signal (TRAIN premise, `fix/premise-fresh.log`; added after the count,
informational, no variant spent): profitability IC +0.023 (t 1.91) against U0's +0.026 (t 2.18), tenth spread +0.67% against
+0.68% a month, composite IC +0.030 against +0.032. U0 is kept; the rule is in IDEAS.md.

### §12.8 Conclusions re-derived

- **Verdict unchanged: no edge.** Final candidate U0 (variants used 6 of 8; nothing new run
  through the bar). Gates on the pre-registered reading (G1-G3, G5 on the boundary records; the
  rest unchanged): G1 +0.63% pass; G2 +0.78% (P 0.55) fail; G3 −0.83% fail; G4 63rd percentile
  fail; **G5 T1 late fills −0.04% / 2× +0.58% FAIL** (was a pass); G6 pass (+1.24% a down month; any
  low-beta book passes it: I1 and I4 do); G7 β −0.32 fail; G8 −1.3% fail. Six of eight gates fail.
- **T4 is not re-read** (locked; read once for U0, the candidate is unchanged). Its +2.6% (P 0.74),
  late +1.6%, 2× +2.5% are on the lab's reading, which drops the first day like every other lab
  number; on T1 and T2 the boundary reading lowered the base alpha by 0.44 and 0.54 point.
- **What changed in the claims:** (1) costs matter — late fills take T1's alpha to zero; (2) the
  crash record is not "net short beta, not quality" across the board — beta accounts for the 2022
  bear and Feb-Apr 2025, quality cushioned Q4 2018 (+6.5 points beyond beta) and lost beyond beta
  in COVID (−4.2) and the 2022 bear (−8.5); calendar 2022's gain was made in its fourth-quarter
  rally; (3) the short leg lost 10-21 points in half the rising periods, not in all; (4) turnover
  is 30-43 rotation orders a year, gross 0.98 at entry only; (5) 3-9% of gross-profit levels are
  frozen for more than 200 days, which does not hurt the signal.

### §12.9 Lab issues added (worked around; `lab.py` not edited)

5. `lab.book` returns `value.pct_change().fillna(0)`, so every period's first day — the setup
   orders' costs, the borrow the engine charges on day-0 orders for shorts held to the period end
   (0.2-0.35 point a year here), the first open-to-close move — is lost for the book and the
   benchmark. The late / 2× stress therefore reaches only rotation orders (G5), and a strategy's
   alpha is overstated by its setup and day-0 borrow (0.27-0.54 point here). A fair G5 charges the
   turnover between consecutive periods (`research.py boundary`); `r = v / v.shift(1)
   .fillna(init_cash) - 1` would charge a full rebuild every six months instead.
6. G6 is passed by any book with a beta well below 1 (here the beta-neutral I1 and I4): it measures
   beta in falling markets, not crash skill. Splitting episodes by the book's beta (`research.py
   bear`) or demanding a positive return net of beta would separate the two.
7. The universe floor (`ROLL["min_price"]`) and every strategy's price floor read closes adjusted
   back from today for later splits and dividends (NVDA left out of the top 500 in six of eight
   2016-20 periods), and `sector` is today's code for every year.

## Round 2: lab v2 re-evaluation (2026-10-23)

Lab v2 (`strategies/proposals_lab/README.md`, "Lab v2") fixes five lab-level defects that biased
every round-1 number: (1) holiday rows in the 2026 panels (panels are NYSE sessions only now and
were rebuilt from the current DB), (2) each period's day 0 dropped with its opening costs and the
borrow charged up front (now counted from the initial cash, for the book and the benchmark: a full
book rebuild every six months, conservative), (3) new shorts starved by the engine's call order
under `lock_cash` (the lab's runner now executes long cuts, short covers, new shorts, long buys in
that order), (4) borrow over-charged on hair-sized short re-trims (this book never re-trims a
short: `retrim_k` acts on longs only), (5) six sector ETFs typed "equity" in the stock universes
(excluded). v2 results are in `var/proposals_lab/results2/quality_junk/`, the v2 study journals in
`var/proposals_lab/studies2/`; every round-1 label and log stays in `results/` as the record. New
labels end in `v2`.

### §13.0 Pre-registration (written before any v2 number)

- **Code state.** The v2 re-runs use `strategy.py` / `config.yaml` exactly as the round-1 review
  reproduced them (sha1 `325c6a5b` / `ed932657`; lab.py `c2e55bf8`). Nothing in the strategy is
  changed before they finish. Two code fixes follow them, neither able to move a U0 weight:
  `valid()` compares `q_out < q_in` with a tolerance (the tuner's 0.8500000000000001 let an
  equal pair through, review hygiene note), and `_leg_weights`' docstring states that its side
  balance is exact only when both sides are full.
- **Re-runs (OPEN, late fills and 2× costs):** `U0v2` (= the round-1 final, overrides `{}`),
  `Tv2` (round 1's tuned pick, its exact overrides), `I1v2`-`I4v2` (round 1's four iterations).
  Then `noise` and `luck` on U0v2, `lab.py verdict` (market-neutral gates), `lab.py bear`.
- **Decision rules, unchanged:** the tuned values are adopted only if they beat U0v2 on T1 AND T2
  (paired); an iteration only if its paired Δ vs U0v2 is ≥ 0 on stitched TRAIN, T1 and T2 AND its T1
  Δ exceeds 2 × U0v2's T1 noise sd. The verdict is `lab.verdict` on v2; it REPLACES §12.8's
  strategy-local boundary reading (v2 charges the first day itself; `research.py boundary`, ported
  to v2, is reported as a continuous-book sensitivity that decides nothing).
- **The tuning step is re-executed on v2** (`TUNE2`, journal in `studies2/`): the §4 space, seed
  42 and robust objective unchanged, run until at least 120 DISTINCT complete parameter sets have
  been scored (round 1 scored ~96 distinct among 116, review hygiene note); `q_in == q_out` pairs
  are pruned, as §4 intended. No code is edited while it runs. If its pick differs from T, the pick
  goes through the same bar as a new variant (variant 7 of the 10 now allowed).
- **New iterations:** at most 3, only if v2 opens an obvious direction, each pre-registered below
  before it runs.
- **Final candidate:** U0v2 unless a re-checked or new variant passes the bar on v2; declared in
  this log before its T4 read. **T4** is then read ONCE on v2 for U0v2 (a disclosed
  re-measurement of a test read in round 1, on corrected accounting; it selects nothing), and once
  more only for a NEW final. TL is never read.
- **Expected (before any v2 number):** every alpha falls by roughly round 1's full-restore
  correction (§12.1 "full": TRAIN +0.0, T1 +0.7, T2 −0.6), G5 fails (T1 late fills about −1),
  G1 becomes borderline, no adoption decision flips (each rejected variant failed one of its
  conditions by more than the correction), and the status stays "no edge". The call-order fix may
  fill shorts that round 1 starved, which would deepen the net short beta slightly.

### §13.1 U0 on lab v2 (`U0v2`, variant 1 re-measured)

`lab.py run quality_junk U0v2 '{}' E0,TB,T3,T1,T2 --costs2` (log `results2/quality_junk/U0v2.log`),
then `noise`, `luck`, `verdict` (logs `U0v2-noise.log`, `U0v2-luck.log`, `U0v2-verdict.log`). The
code is the reviewed one; after the two §13.0 code fixes `research.py boundary U0v2` re-ran all 93
OPEN books (base, late, 2×) and matched every saved daily return exactly (max |diff| 0.0).

| | round 1 (lab v1) | round 1, §12 boundary reading | **round 2 (lab v2)** |
|---|---|---|---|
| TRAIN stitched alpha (P) | +0.90% (0.67) | +0.63% | **+0.14% (0.59)** |
| T1 alpha (P) | +1.22% (0.58) | +0.78% (0.55) | **−0.38% (0.51)** |
| T2 alpha (P) | −0.29% (0.48) | −0.83% | **−0.05% (0.51)** |
| T1 late fills / 2× costs | +0.89% / +1.14% | −0.04% / +0.58% | **−1.33% / −0.48%** |
| T1 beta / CAGR / B&H / max DD | −0.32 / −1.5% / +8.6% / −26.6% | | −0.33 / −3.3% / +8.8% / −33.2% |
| T1 without its top 5 names | −1.3% | | −2.7% |
| T1 luck (100 shuffled-name books) | 63rd percentile (median +0.2%, p95 +5.5%) | | **50th** (median −0.4%, p95 +4.6%) |
| T1 noise sd → adoption bar | 1.2 → 2.4 points | | **0.86 → 1.72 points** |
| down-month mean (41 months; market −3.98%) | +1.27% | +1.24% | +0.92% |

Noise (5 names dropped a period, 3 copies; alpha Δ vs U0v2, points): E0 +0.8 / −0.1 / −0.4 (sd
0.62), TB −0.1 / −0.4 / +0.3 (0.34), T3 −3.6 / −1.5 / +0.3 (1.92), **T1 +0.7 / +2.2 / +0.7 (0.86)**,
T2 −0.4 / −1.1 / −0.7 (0.35). Per-period alphas — E0 −9.5, −2.2, +2.0, −9.9; TB +15.8, +22.6,
+2.0, −3.1; T3 +2.6, +10.7, +29.3; T1 +10.3, +8.6, −6.9, −8.2, +21.7, −3.3, +7.0, −9.9, +11.7,
+20.2; T2 −3.3, +2.8, −6.2, +7.5, +5.6, −5.9, +2.7, −10.0, +9.7, +29.2.

**Where the numbers moved** (`research.py v1v2 U0 U0v2 --engine`, log `v1v2-U0-engine.log`):
four readings of the same config, each step one change, stitched alpha %/yr.

| reading | TRAIN | E0 | TB | T3 | T1 | T2 |
|---|---|---|---|---|---|---|
| lab v1 (saved round-1 U0) | +0.90 | −5.37 | +5.14 | +14.09 | +1.22 | −0.29 |
| v2 panels, the engine's runner, first day dropped | +0.90 | −5.37 | +5.14 | +14.09 | −0.32 | −0.26 |
| v2 panels, the lab's runner, first day dropped | +1.01 | −5.35 | +5.45 | +14.14 | +0.18 | +0.22 |
| **lab v2** (first day counted) | **+0.14** | **−6.56** | **+4.22** | **+13.65** | **−0.38** | **−0.05** |

- **Panels (fixes 1 and 5): T1 −1.54, T2 +0.03.** With the round-1 code, the rebuilt panels give
  every period up to 2024-H1 to the decimal (the DB refresh changed nothing there), and all of
  the change sits in the five half-years from 2024-H2 on — the periods whose top 500 held the
  six sector ETFs, now replaced by six other names (the ETFs had no fundamentals and never
  entered the book; the replacements shift ranks and the book's names), two of them also the
  periods with the empty US-holiday rows: T1 2024-H2 +5.6 → −0.4, 2025-H1 +7.5 → +8.5, 2025-H2
  −11.3 → −14.2, 2026-H1 +21.8 → +13.7, 2026-H2 +10.4 → +17.7.
- **The lab's runner (fixes 3 and 4): T1 +0.50, T2 +0.48, TRAIN +0.11 (TB +0.31).** New shorts
  that the engine's call order starved now fill (`research.py boundary`: of ~830 new shorts the
  engine's runner left 37 under 80% of target on the OPEN tests, the lab's 17 — the rest are
  cash-limited because drift had pushed realized gross above 1); it matters in the periods with
  starved shorts (T1 2025-H2 −14.2 → −10.9, T2 2025-H2 −18.5 → −12.1, T2 2022-H1 −3.3 → −1.6).
  Fix 4 cannot act here: shorts are never re-trimmed.
- **The first day (fix 2): T1 −0.56, T2 −0.27, TRAIN −0.87** (E0 −1.21, TB −1.23, T3 −0.49).

**Audits on the rebuilt panels** (identity and TRAIN only; logs `coverage-v2.log`,
`premise-v2.log`, `staleness-v2.log`): eligibility 0.90-0.96 of each period's names, GP/A on
0.54-0.67 of them and ROE on 0.27-0.38, as in §1; the two scales by period: median GP/A 0.26-0.32,
median ROE 0.10-0.15 (0.13-0.15 on T1; §1's "0.12" is their average over periods, not every bar —
the review's precision note). The TRAIN premise is identical to the printed digits (composite IC
+0.032, t 1.46; profitability +0.026, t 2.18; safety +0.020, t 0.76; growth −0.010; the fresh-ROE
fallback +0.023), and so are the frozen gross-profit shares (2.8% of GP/A names on T1, 3.5% E0,
4.8% T2, 6.6% T3, 8.5% TB unchanged for more than 200 days, up to 1,161 days; the book holds
0.75 (T1) to 3.8 (T3) of them out of 40, at most 6).

### §13.2 Gates on v2 (`lab.py verdict quality_junk U0v2`, market-neutral)

| gate | round 1 (lab v1) | round 1 after §12 | **v2** |
|---|---|---|---|
| G1 TRAIN-era alpha > 0 | +0.90% pass | +0.63% pass | **+0.14% pass** |
| G2 T1 alpha > 0, P ≥ 0.90 | +1.22%, P 0.58 fail | +0.78%, P 0.55 fail | **−0.38%, P 0.51 fail** |
| G3 T2 alpha > 0 | −0.29% fail | −0.83% fail | **−0.05% fail** |
| G4 T1 ≥ 95% of shuffled books | 63rd fail | 63rd fail | **50th fail** |
| G5 T1 late fills and 2× > 0 | +0.89 / +1.14 pass | −0.04 / +0.58 FAIL | **−1.33 / −0.48 FAIL** |
| G6 every bear episode, down months > 0 | +1.27% pass | +1.24% pass | **+0.92% pass** |
| G7 \|T1 beta\| ≤ 0.3 | −0.32 fail | −0.32 fail | **−0.33 fail** |
| G8 T1 without top 5 > 0 | −1.3% fail | −1.3% fail | **−2.7% fail** |

Six of eight fail, as after §12. **G5 fails on every first-day reading** (`research.py boundary
U0v2`, log `boundary-U0v2-report.log`; the benchmark at base costs, as `lab.verdict` stitches the
stressed books): T1 late fills −1.33% on the lab's full rebuild, −0.57% if one continuous book paid
only the boundary turnover (with the first day's move), −0.28% on the lowest estimate (only names
leaving the universe plus one regular rebalance). The full rebuild over-charges this book by
0.20 point of T1 alpha a year at base costs (−0.38% vs −0.18%) and 0.76 under late fills; the
day-0 borrow alone is 0.20 (TRAIN), 0.32 (T1) and 0.36 (T2) point a year.

### §13.3 Every round-1 decision re-checked on v2 (`lab.py report`, `show.py delta`)

Paired Δ alpha against U0v2 on the same days and bootstrap blocks (points; P(Δ > 0)); the bar is
Δ ≥ 0 on stitched TRAIN, T1 and T2 and a T1 Δ above 1.72 (the tuned values: beat U0v2 on T1 AND T2).

| variant | round 1 Δ: TRAIN / T1 / T2 | **v2 Δ: TRAIN / T1 / T2** | v2 alpha T1 / T2 (β T1) | decision |
|---|---|---|---|---|
| T (tuned) | +1.4 / +7.4 / −9.6 | **+1.27 (0.69) / +9.76 (0.96) / −6.14 (0.11)** | +9.4% / −6.2% (−0.82) | not adopted (T2) |
| I1 `sizing: legs` | −2.1 / −0.8 / +0.5 | **−1.97 (0.13) / +0.42 (0.53) / +0.63 (0.65)** | +0.0% / +0.6% (−0.01) | not adopted (TRAIN, T1 bar) |
| I2 `vol_window: 63` | −0.3 / +1.7 / −3.6 | **+0.08 (0.53) / +2.59 (0.90) / −5.11 (0.01)** | +2.2% / −5.2% (−0.35) | not adopted (T2) |
| I3 `safe_by_sector` | −0.5 / +0.1 / +0.6 | **−0.38 (0.34) / +1.30 (0.73) / +0.95 (0.67)** | +0.9% / +0.9% (−0.27) | not adopted (TRAIN, T1 bar) |
| I4 `w_safe: 0` | −1.1 / −2.1 / +5.1 | **−0.99 (0.34) / −0.31 (0.45) / +5.02 (0.87)** | −0.7% / +5.0% (−0.11) | not adopted (TRAIN, T1) |

**No decision flips.** The strongest rejected variant on v2 is I2: the first to clear two of the
three conditions (TRAIN +0.08, T1 +2.59 over the 1.72 bar, at U0's beta: −0.35 vs −0.33), and it
loses 5.1 points on names 501-1000 (P 0.01, 2 of 10 periods better), where the 63-bar measure
deepens the net short beta (−0.27 vs −0.15) and the drawdown (−44% vs −23%). The tuned T
again is a −0.8 beta book (+9.8 on T1, −6.1 on T2, max DD −62% on T2). I3 gains on both
validation sets (+1.30, +0.95) but loses on TRAIN and stays under the T1 bar; it remains the risk
improvement round 1 found (vol 12.5% vs 14.4% on T1, COVID −1.0%, 2022 bear +2.5%). I4's
profitability-alone T2 result replicates (+5.0%, P 0.94; +0.3% without its top five names) and is
again absent on TRAIN (−0.85%) and T1 (−0.7%). I1v2 is beta-free on every test (β −0.08 to +0.10)
with no alpha (TRAIN −1.8%, T1 +0.0%, T2 +0.6%).

### §13.4 The tuning step re-executed on v2 (`TUNE2`; variant 7 = its pick, `TRv2`)

`lab.py tune quality_junk TUNE2 '{}' '<§4 space>' 150`, resumed to 170 (journal
`studies2/quality_junk-TUNE2.log`; logs `results2/quality_junk/TUNE2.log`, `TUNE2-resume170.log`),
TRAIN only, seed 42, started 08:24:43 after both §13.0 code fixes (strategy.py sha1 `82c53024`);
`strategy.py` and `config.yaml` were not touched while it ran (08:24-09:05; only `research.py`,
which the study does not load, and the docs were edited). 168 complete trials, 2 pruned (`q_out ≥
q_in`: 0.80 / 0.80 and 0.85 / 0.80; no float-noise pair came up, so the new tolerance never
bound), **133 distinct parameter sets** (the first 150 trials gave 114, so the study was resumed
to reach the pre-registered 120). **PBO 0.69** (round 1: 0.54). Best robust score +0.032 (mean
period alpha +8.3% in-sample) at trial 34: `w_prof` 1.0, `vol_window` 189, `q_in` 0.90, `q_out`
0.75, `slots` 15, `rebalance` month, `beta_window` 252, `max_size` 3.0 — closer to U0 than round 1's T (`w_prof` and `q_in` are U0's), with much stronger
beta sizing (low-beta slots up to 3×, high-beta ones down to ⅓). The surface (mean robust score
by value): TPE sampled `vol_window` 63 in 109 of 168 trials (+0.009; 189: −0.001; 126: −0.032;
≥ 252: ≤ −0.020), `slots` 10 in 121 (+0.001; ≥ 20: ≤ −0.014), `beta_window` 252 in 152, `w_prof`
1.0 in 86 (+0.005; 0-0.5: −0.013 to −0.063), `rebalance` quarter only 8 times (−0.033). Round 1's
corrected numbers for the record: TUNE1 scored 96 distinct sets among 116; TPE *sampled*
`vol_window` 63 in 82 of them (§4's "chose" and §6's "won" overstated it), and `w_prof` 1.25
averaged +0.001 (§4's "≥ 1.25: ≤ 0" was wrong for 1.25; 1.5 and 1.75 averaged −0.010, −0.009).

The pick runs on OPEN as label **TRv2** (variant 7, declared in §13.0 before TUNE2 ran), judged by
the tuned-values rule: adopted only if it beats U0v2 on T1 AND T2 (paired).

**TRv2 result (variant 7): ADOPTED by the tuned-values rule, by margins inside the noise.**
`lab.py run quality_junk TRv2 '<the pick>' E0,TB,T3,T1,T2 --costs2` (log `TRv2.log`):

| test | alpha | P | beta | vol | max DD | late α | paired Δ vs U0v2 | P(Δ>0) | periods better |
|---|---|---|---|---|---|---|---|---|---|
| E0 / TB / T3 | +0.5% / +4.7% / +16.8% | 0.56 / 0.83 / 0.95 | −0.15 / +0.12 / −0.17 | | | | +7.1 / +0.5 / +3.2 | 1.00 / 0.52 / 0.80 | 4/4, 1/4, 2/3 |
| TRAIN stitched | +3.5% | 0.87 | +0.03 | 8.8% | −14.6% | | **+3.31** | 0.98 | |
| T1 | +0.6% | 0.56 | −0.20 | 12.6% | −24.5% | −0.6% | **+0.97** | 0.62 | 5/10 |
| T2 | +0.0% | 0.52 | −0.11 | 11.0% | −26.0% | −1.3% | **+0.07** | 0.52 | 4/10 |

The rule pre-registered in §13.0 (round 1's §4 rule: "adopted only if it beats U0 on T1 AND T2,
paired") is met: both paired Δ are positive. Neither is distinguishable from zero: T1's +0.97 is
under two noise sd (1.72) and under the mean Δ of U0v2's own three noise copies (+1.2), T2's +0.07
is a coin flip (P 0.52), and the PBO is 0.69. The gain is on TRAIN, where the tuner looked
(+3.3, E0 +7.1). What the stronger beta sizing changed is the beta (T1 −0.20 vs −0.33), not the
selection. TRv2 becomes the base, as the protocol says; the status cannot change on margins like
these, and the gates will say so.

### §13.5 Iterations against the new base (pre-registered before running; variants 8-9 of 10)

The four round-1 iterations were U0 design changes and were re-checked against U0v2 (§13.3). With
TRv2 the base, the protocol's step 5 judges iterations against TRv2: Δ ≥ 0 on stitched TRAIN, T1
and T2 and a T1 Δ above 2 × TRv2's T1 noise sd (`lab.py noise quality_junk TRv2`, run first).
Two of round 1's mechanisms are re-tested on TRv2, chosen from what v2 showed; nothing else.

- **I5 — sector-neutral safety on TRv2 (`safe_by_sector: true`).** Mechanism (round 1's I3): the
  sides are a sector bet (long utilities, staples and industrials, short technology and
  discretionary), so sector rotations dominate the tails; ranking volatility within sector keeps
  the stock-level lottery effect and drops the sector bet. On v2 it was the one round-1 change
  that gained on both validation sets against U0v2 (T1 +1.30, T2 +0.95) and it cut volatility and
  the COVID loss; it lost 0.38 on TRAIN. Prediction vs TRv2: lower volatility, T1 and T2 Δ between
  −1 and +2, TRAIN Δ between −1.5 and +0.5 (TRv2 is the TRAIN optimum of its space), so most
  likely not adopted.
- **I6 — BAB side neutrality on TRv2 (`sizing: legs`).** Mechanism (round 1's I1), and the check
  of what TRv2's gain is: TRv2's 3× per-name beta sizing shrank the net short beta (T1 −0.20 vs
  −0.33) and its paired gains are small; sizing the two SIDES so that D_L β̄_L = D_S β̄_S removes
  the remaining beta, and keeps whatever TRv2's selection earns beyond beta. Prediction: |β| ≤
  0.1 on every test and alpha Δ between −2 and +1 (the beta-neutral I1v2 earned ≈ 0 everywhere),
  so not adopted unless TRv2's selection carries alpha.

The final candidate is declared in §13.6, after these two, before `verdict`, `bear` and T4.

**Results** (logs `TRv2-noise.log`, `TRv2-luck.log`, `I5v2.log`, `I6v2.log`, `delta-TRv2.log`,
`report-TRv2.log`). TRv2's noise (5 names dropped, 3 copies): T1 Δ −1.2 / −1.7 / −0.5 (sd 0.61 →
**bar 1.21**), T2 −2.4 / −1.2 / −1.4 (sd 0.63), TRAIN periods E0 sd 0.41, TB 1.88, T3 4.01. Every
copy lands below TRv2 on T1 and T2, while U0v2's three copies all landed above U0v2 on T1 (mean
+1.2): the +0.97 by which TRv2 "beats" U0v2 on T1 is smaller than the gap between where each sits
in its own noise. TRv2's luck: real +0.6%, shuffled median −1.0%, p95 +5.5% → 66th percentile.

| variant (vs TRv2) | alpha TRAIN / T1 / T2 | Δ TRAIN / T1 / T2 (P) | β T1 / T2 | decision |
|---|---|---|---|---|
| I5 `safe_by_sector` | +1.0% / +0.6% / −2.6% | **−2.42 (0.09) / +0.05 (0.50) / −2.60 (0.21)** | −0.16 / −0.13 | not adopted |
| I6 `sizing: legs` | +1.8% / +1.1% / −1.4% | **−1.67 (0.05) / +0.52 (0.58) / −1.37 (0.18)** | +0.00 / +0.02 | not adopted |

Both fail TRAIN, T2 and the T1 bar. I5 again lowers T1 risk (vol 11.1% vs 12.6%, max DD −19.3% vs
−24.5%, COVID −1.7% vs −7.4%) but not T2's (max DD −33.1% vs −26.0%); its prediction held for the
T1 volatility only (T2 and TRAIN Δ fell outside the predicted ranges). I6 is beta-neutral on the
validation sets (T1 +0.00, T2 +0.02; TB +0.13 and T3 −0.14 miss the predicted ±0.1) and its alpha
Δ is inside the predicted −2 to +1: what TRv2 earns beyond beta on T1 (+1.1% at zero beta) is not
distinguishable from zero (P 0.61), and on T2 it is negative (−1.4%).

### §13.6 Final candidate (declared here, before `verdict`, `bear` and the T4 reads)

**TRv2** — round 1's §4 space re-tuned on lab v2 (TUNE2) and adopted by the tuned-values rule;
neither step-5 iteration beat it. Overrides: `{"w_prof": 1.0, "vol_window": 189, "q_in": 0.9,
"q_out": 0.75, "slots": 15, "rebalance": "month", "beta_window": 252, "max_size": 3.0}` (the rest
as `config.yaml`). Variants used: 9 of 10 (round 1: U0, T, I1, I2, I3, I4; round 2: TRv2, I5,
I6; the v2 re-measurements of round-1 labels are not new variants). `config.yaml` keeps U0's
values: every saved label with overrides `{}` (U0, U0v2, their noise and luck caches) stays
reproducible, and no config without an edge should be the one a headless run picks up; TRv2 is
the overrides above. T4 is now read twice, each once: **U0v2** (a disclosed re-measurement of the
round-1 final on corrected accounting) and **TRv2** (the new final's one read). Neither selects
anything; TL is never read. strategy.py sha1 `01ed7a33` (docstring-only changes since `82c53024`).

### §13.7 Verdict, bear table and T4 (lab v2)

`lab.py verdict quality_junk TRv2` (market-neutral gates; log `TRv2-verdict.log`), next to U0v2:

| gate | U0v2 | **TRv2 (final)** |
|---|---|---|
| G1 TRAIN-era alpha > 0 | +0.14% pass | **+3.46% pass** (in-sample: tuned there) |
| G2 T1 alpha > 0, P ≥ 0.90 | −0.38%, P 0.51 fail | **+0.59%, P 0.56 fail** |
| G3 T2 alpha > 0 | −0.05% fail | **+0.02% pass** (by 0.02 point) |
| G4 T1 ≥ 95% of shuffled books | 50th fail | **66th fail** |
| G5 T1 late fills and 2× > 0 | −1.33 / −0.48 fail | **−0.62 / +0.30 fail** |
| G6 every bear episode, down months > 0 | +0.92% pass | **+0.67% pass** |
| G7 \|T1 beta\| ≤ 0.3 | −0.33 fail | **−0.20 pass** |
| G8 T1 without top 5 > 0 | −2.7% fail | **−1.9% fail** |

TRv2 passes four of eight; "works to some degree" needs G1, G2, G3, G5 and G6 — G2 fails by a
wide margin (P 0.56 against 0.90), so the status is **no edge**. G5 is the one gate the
first-day convention decides for TRv2 (`research.py boundary TRv2`, log `boundary-TRv2.log`,
every saved return reproduced exactly): late fills −0.62% on the lab's full rebuild, +0.05% /
+0.11% / +0.46% if one continuous book paid only the boundary turnover (without / with the first
day's move / the lowest estimate). Pre-registered: the lab's reading decides; and a G5 pass would
leave G2 and G8 failing. TRv2's own rotation is 17 / 23 / 33 orders a year (TRAIN / T1 / T2) on
top of 61-64 first-bar orders; realized gross mean 0.95-1.01, max 1.21.

`lab.py bear` (stitched E0 + TB + T3 + T1, %; log `bear-final.log`):

| | Q4 2018 | COVID crash | 2022 bear | Feb-Apr 2025 | down-month mean | CAGR | max DD |
|---|---|---|---|---|---|---|---|
| market (equal weight) | −18.2 | −37.0 | −24.5 | −19.5 | −3.98 | | |
| SPY | −18.7 | −33.4 | −24.1 | −18.6 | | | |
| U0v2 | +5.5 | −6.3 | −0.4 | +7.0 | +0.92 | −1.6 | −33.2 |
| **TRv2** | **+4.1** | **−7.4** | **−1.0** | **+8.7** | **+0.67** | +1.6 | −24.5 |
| Tv2 | +10.4 | +5.8 | +9.5 | +14.9 | +2.91 | −2.8 | −47.7 |
| I1v2 (β ≈ 0) | +3.8 | −12.7 | −6.6 | +1.0 | −0.20 | −0.3 | −26.0 |
| I2v2 | +3.2 | −5.1 | +2.7 | +7.4 | +0.99 | −0.7 | −32.8 |
| I3v2 | +4.5 | −1.0 | +2.5 | +6.2 | +0.89 | −1.0 | −27.6 |
| I4v2 | −1.1 | +0.6 | +1.8 | +3.4 | +0.01 | −1.0 | −23.0 |
| I5v2 | +4.7 | −1.7 | +0.8 | +3.9 | +0.39 | +0.2 | −19.3 |
| I6v2 (β ≈ 0) | +2.8 | −8.9 | −3.4 | +2.8 | +0.06 | +2.2 | −17.2 |

Split into beta and the rest (`research.py bear U0v2 TRv2 I1v2 I6v2`, log `bear-split-final.log`;
beta part = the test's realized beta × the market's move): U0v2 Q4 2018 −1.0 / **+6.5**, COVID
−2.1 / −4.2, 2022 bear +8.1 / −8.5, Feb-Apr 2025 +6.5 / +0.6, down months +0.64 / +0.27; TRv2 −2.1
/ **+6.2**, −4.3 / −3.2, +5.0 / −6.0, +4.0 / +4.8, +0.30 / +0.37; I1v2 −1.9 / +5.7, −3.8 / −8.9,
+0.2 / −6.8, +0.1 / +0.9, −0.12 / −0.08. Calendar years, U0v2 / TRv2 / market: 2016 −6.9 / −1.5
/ +9.0, 2017 −6.3 / −0.0 / +24.3, 2018 +2.5 / +4.5 / −2.8, 2019 +3.6 / +1.8 / +35.6, 2020 −5.8 /
+1.0 / +25.7, 2021 +13.7 / +18.1 / +21.5, 2022 +10.7 / +7.1 / −18.6, 2023 −18.1 / −12.1 / +22.7,
2024 +3.8 / +4.8 / +15.3, 2025 −8.6 / −5.9 / +16.2, 2026 −0.8 / +1.6 / +11.7. U0v2's calendar 2022
is again its fourth-quarter rally: −0.4% in the bear (−8.5 net of beta), +11.1% against +7.8%
from 10-13 to 12-31 (+13.7 net of beta; I1v2 +11.7%).

Reading. Every variant, including the two beta-neutral ones, beats the falling market in all four
episodes: that is what a beta well below 1 does. G6's second condition separates them: the
beta-neutral I1v2 now FAILS it (down-month mean −0.20%; it passed on lab v1 at +0.04%, so the
review's "I1 passes G6 too" no longer holds) and I6v2 and I4v2 pass it at +0.06% and +0.01%.
U0v2's +0.92% is two-thirds beta (+0.64), TRv2's +0.67% about half (+0.30). Beyond beta, quality
cushioned Q4 2018 in both books (+6.5, +6.2) and lost in the COVID crash and the 2022 bear;
TRv2 also kept +4.8 beyond beta in Feb-Apr 2025.

**T4 — names 1001-1500, locked, each read once on lab v2** (`lab.py run quality_junk U0v2 '{}'
T4 --costs2` and the same for TRv2's overrides; logs `U0v2-T4.log`, `TRv2-T4.log`; declared in
§13.6; nothing was selected from them):

| | alpha | P | beta | CAGR / B&H | vol | max DD | late | 2× | top-5-out |
|---|---|---|---|---|---|---|---|---|---|
| U0 round 1 (lab v1, read once) | +2.6% | 0.74 | −0.09 | +1.7% / +9.8% | 9.8% | −18.8% | +1.6% | +2.5% | +0.2% |
| **U0v2** (re-measurement) | **+1.7%** | 0.68 | −0.12 | +0.5% / +10.2% | 10.2% | −24.5% | −0.2% | +1.4% | −0.5% |
| **TRv2** (the new final) | **+0.6%** | 0.55 | −0.16 | −1.0% / +10.2% | 12.5% | −36.2% | −1.3% | +0.2% | −2.2% |

Periods — U0v2: −8.0, −1.3, −1.0, +6.4, +5.7, +14.0, +4.6, −2.3, −10.1, +38.0 (a 3.5-month stub);
TRv2: +9.3, −1.9, −1.7, +12.2, +10.4, +12.6, +4.5, −13.9, −28.0, +33.9. Paired on T4, TRv2 − U0v2:
**−1.12 points (P 0.34)**. The tuned final does not carry its noise-level validation margin to the
locked names; neither book's alpha survives late fills or the removal of its five biggest names.

### §13.8 Every review issue → fix → effect

| # | issue (severity) | fix | effect on lab v2 |
|---|---|---|---|
| 1 | G5 and "costs do not matter" rested on the lab dropping each first day (major) | lab v2 counts day 0 (lab fix 2); §12's strategy-local boundary reading is superseded; `research.py boundary` ported to v2 as a continuous-book sensitivity whose stressed books are stitched against the base-cost benchmark, as `lab.verdict` does (round 1's version stitched them against a stressed benchmark, 0.37 point lower on v2's T1; on that convention §12.1's G5 "fail" at −0.04% would have read +0.07%, its own "bench at base costs" row) | U0v2 G5 FAILS on every reading (T1 late −1.33% lab, −0.57% continuous, −0.28% lowest); late fills cost 0.95 point of T1 alpha, 2× costs 0.10. TRv2: −0.62% (fail) on the lab's reading, +0.05 to +0.46 on continuous-book readings |
| 2 | bear mechanism overstated (major) | headline, G6 reading and conclusions rewritten from the v2 episode split (§13.7) | any low-beta book beats every episode (I1v2, I4v2, I6v2 too); U0's down-month cushion is two-thirds beta; I1v2 now fails G6 on its down-month mean (−0.20%); beyond beta U0's quality helped only in Q4 2018 (+6.5; TRv2 +6.2, and +4.8 in Feb-Apr 2025) |
| 3 | leg claim cherry-picked (minor) | every period shown (§13.9 table, `anatomy-U0v2.log`) | short leg lost 10-22 points in five of the eleven rising top-500 periods, 2-5 in five, gained in 2020-H1; its ex-ante beta exposure is −0.46 to −0.67 |
| 4 | `decision_bars` ignored holidays on the last bar (minor) | fixed in round 1 (§12.4: NYSE calendar, test) | nothing moves; v2 panels are NYSE sessions only |
| 5 | static sectors, split-adjusted $5 floors (minor) | caveats (README); lab "known and not fixed" | unchanged |
| 6 | turnover wording (minor) | rotation and period-start orders reported apart; "gross 0.98 at entry" | U0v2 rotation 30 / 35 / 42 orders a year (TRAIN / T1 / T2) + 81-85 first-bar; one continuous book ~54 / 74 / 105; realized gross mean 0.99, max 1.31 |
| 7 | frozen gross profit preferred over fresh ROE (minor) | actual staleness stated (§12.7), re-measured on the v2 panels (§13.1) | 2.8-8.5% of GP/A names stale > 200 days, up to 1,161 days; 0.75-3.8 of the 40 held names; a fresh-ROE fallback does not improve the premise |
| 8a | strategy.py edited while TUNE1 ran (minor) | round 2 edited strategy.py only between runs (08:13, 09:33; `config.yaml`'s header comment after the last run) and never while TUNE2 ran (08:24-09:05) | none |
| 8b | ~96 distinct sets among 116 trials (minor) | TUNE2 run to 133 distinct sets (168 trials); `valid()` rejects a float-noise band (`BAND_TOL`, test) | the pick changed (TRv2) and was adopted, by margins inside the noise |
| 8c | another agent's results read (minor) | round 2 reads none; the blend computation is saved as `correlate.py blend` (round-1 record) | the diversification numbers stay lab v1's, labelled so |
| 9 | wording slips (minor) | "TPE sampled vol_window 63 in 82 of 116" (§13.4); `w_prof` 1.25 averaged +0.001; I1 beta −0.08 to +0.10 (TB +0.10); per-period medians GP/A 0.26-0.32, ROE 0.10-0.15 (docstring, §13.1); survivorship flatters longs and hurts shorts (README); `_leg_weights` balance exact only with full sides (docstring, README); no "alpha falls to about zero" | none |

### §13.9 Conclusions (round 2)

| | round 1 U0 (lab v1) | **U0v2** | **TRv2 (final)** |
|---|---|---|---|
| TRAIN alpha (P) | +0.90% (0.67) | +0.14% (0.59) | +3.46% (0.87), tuned there |
| T1 alpha (P) | +1.22% (0.58) | −0.38% (0.51) | +0.59% (0.56) |
| T2 alpha (P) | −0.29% (0.48) | −0.05% (0.51) | +0.02% (0.52) |
| T4 alpha (P) | +2.6% (0.74) | +1.7% (0.68) | +0.6% (0.55) |
| T1 beta, late-fill alpha | −0.32, +0.89% | −0.33, −1.33% | −0.20, −0.62% |
| T1 luck percentile | 63 | 50 | 66 |
| gates passed | G1, G5, G6 (G5 failed after §12) | G1, G6 | G1, G3, G6, G7 |
| bear episodes (Q4 18 / COVID / 2022 / 2025) | +5.5 / −6.3 / −0.4 / +7.4 | +5.5 / −6.3 / −0.4 / +7.0 | +4.1 / −7.4 / −1.0 / +8.7 |
| down-month mean | +1.27% | +0.92% | +0.67% |

The top 500's legs on U0v2 (sums of daily realized weight × return, points; the halves are the
test boundaries):

| | 16-H2 | 2017 | 18-H1 | 18-H2 | 2019 | 20-H1 | 20-H2 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long | −1.1 | +7.1 | −1.7 | +2.7 | +15.1 | −3.3 | +12.4 | +16.2 | −3.6 | +2.8 | +5.9 | +4.9 | +7.7 |
| short | −4.9 | −13.0 | −2.0 | +5.7 | −10.2 | +2.8 | −16.7 | −2.0 | +15.3 | −21.6 | −2.9 | −13.6 | −4.6 |
| market | +8.9 | +22.0 | +4.1 | −5.7 | +31.3 | +2.7 | +26.4 | +20.6 | −16.9 | +21.7 | +15.1 | +16.9 | +12.0 |

Names 501-1000, 2022 → 2026: long −5.6, +10.4, +11.3, +2.4, +4.7; short +7.5, −11.8, −11.9,
−6.8, +3.2; market −12.7, +20.4, +12.4, +14.7, +10.2. Ex-ante beta exposure: short −0.46 (TB) to
−0.67 (T1), long +0.27 to +0.37. Net of each leg's realized beta, points a year: E0 −4.6 / −0.9,
TB +1.2 / +4.4, T3 +8.2 / +5.4, T1 +0.8 / +0.6, T2 +1.4 / +0.7.

- **Status unchanged: no edge.** Lab v2 lowered the round-1 default's alpha on every test but T2
  (the first day −0.3 to −0.9 point a year, the rebuilt universes −1.5 on T1, the lab's runner
  +0.5), and it fails six gates. The re-tuned final passes four but not G2: +0.6% on the top 500
  at the 66th percentile of shuffled-name books, +0.0% on names 501-1000, +0.6% on the locked
  names, 1.1 points below U0 there.
- **What changed:** the final candidate (U0 → TRv2, through the tuned-values rule, which has no
  noise bar: both margins were inside the noise, and on T1 TRv2's three noise copies all sit below
  it while U0v2's all sit above U0v2); G5 fails on the lab's reading for both books; the
  beta-neutral I1 now fails G6 too; I2 (63-bar volatility) clears TRAIN and the T1 bar against
  U0v2 and still loses 5.1 points on T2.
- **Against the §13.0 expectation:** as expected G5 fails, G1 is borderline (+0.14%), no round-1
  decision flips and the status stays; not expected, T1 fell 1.1 points more than the
  full-restore correction (the rebuilt universes from 2024-H2), T2 rose, and the re-tuned pick
  was adopted (by the rule, on noise-level margins).
- **What did not change:** no round-1 adoption decision flips; the TRAIN premise, coverage and
  staleness are identical on the rebuilt panels; the bear record is net short beta plus a Q4
  2018 quality cushion; I4's profitability-alone T2 result replicates (+5.0%, P 0.94) and rests on
  five names (+0.3% without them).
- Variants used: 9 of 10. TL never read. Lab issues met in round 2: the tuned-values rule has no
  noise bar (a lab-level question), and `lab.tune` still neither rounds float steps nor skips
  duplicates (§11.3; worked around by running to 120 distinct sets).
