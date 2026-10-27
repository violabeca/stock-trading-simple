# etf_trend — research log (proposal 7, STRATEGY_OVERVIEW.md §6 "also considered")

Every section is written BEFORE the numbers it announces are read, unless it says "result".
Evaluation follows `strategies/proposals_lab/README.md` (mandatory protocol), adapted to an ETF
book as the task specifies: the lab's equity TESTS do not apply, so this folder defines its own
half-year periods on its own ETF panel (`research.py`) and reuses `lab.book`, `lab.stitched`,
`lab.alpha_of`, `lab.paired`, `lab.robust`, `lab.period_alpha`, `lab.pbo`.

## §0 Pre-registration (2026-10-22, before any number)

**Hypothesis.** Time-series momentum across asset classes (Moskowitz, Ooi & Pedersen 2012;
Hurst, Ooi & Pedersen 2017; Faber 2007): an asset whose trailing 12-month total return beats
cash tends to keep beating cash over the next month. A long/flat, inverse-volatility weighted
ETF book that holds each asset only while its trend is positive should lose much less than SPY
in PROLONGED bear markets (2008, 2022), when equities trend down while Treasuries, gold or the
dollar trend up, and give up part of the bull-market return (lower beta, whipsaw costs).

**Mechanism.** Slow diffusion of information and herding (under-reaction, then delayed
over-reaction); in a crisis, flight-to-quality flows persist for months. Risk-parity weighting
spreads the book across equities, bonds, commodities and the dollar, so the book is not one
equity bet.

**Data.** 24 ETFs backfilled from yfinance (1993 → 2026-10-16) into the shared DB:
SPY QQQ IWM EFA EEM (equity indices), XLB XLE XLF XLI XLK XLP XLU XLV XLY (sectors), VNQ (REITs),
TLT IEF LQD HYG (bonds), GLD SLV DBC (commodities), UUP (US dollar), SHY (cash proxy). Adjusted
closes are total returns (dividends reinvested).

**Periods** (half-years, `lab.halves`): TRAIN = 2004-01 → 2015-12 (24 periods; 2008), VALID =
2016-01 → 2026-10-16 (22 periods; Q4 2018, COVID, 2022, Feb-Apr 2025), split for replication
into VA = 2016-2020 (10) and VB = 2021-2026 (12). **LOCKED = 2000-01 → 2003-12** (8 periods: the
dot-com bust, assets then live: SPY, sectors, QQQ, IWM, EFA; no bonds, cash at 0%) is this
proposal's T4: read ONCE, for the final candidate against U0, after the final is declared here.
Each period's panel holds the assets with >= 252 bars of history before the period's first day
(an asset joins at the first period start after its first 12 months); the equal-weight
buy-and-hold of that panel is `lab.book`'s bench.

**Benchmarks.** SPY buy-and-hold (primary: the investor's alternative) and the equal-weight
buy-and-hold of the period's ETFs; both reported. alpha vs SPY = CAGR − beta_SPY × SPY CAGR.

**U0 — the literature default, nothing tuned.**
- Decision once a month at the close of the month's first trading day (causal: a new month is
  known when its first bar prints; month-END needs the calendar of the bar after), filled at the
  next open. Weights are constant between decisions.
- Trend: asset on ⇔ its 252-bar total return > SHY's 252-bar total return (excess-return
  TSMOM, MOP); where SHY is not yet live the hurdle is 0.
- Size: inverse volatility, σ = EWMA std of daily returns, centre of mass 60 days (MOP),
  normalised over EVERY live risky asset (on or off): w_i = 0.98 × (1/σ_i) / Σ_j (1/σ_j). An off
  asset's budget goes to SHY (the engine pays 0% on idle cash). A fully-on book is naive risk
  parity, 98% invested.
- Long/flat. A held asset whose state is unchanged keeps its weight unless its new target is
  more than 25% away (`band` 0.25): the engine charges a fee FRACTION, so tiny monthly re-trims,
  which really pay the $1 minimum, would be undercharged. Not tuned (a cost guard, not alpha).
- Costs: the repo's model at `order_notional` 1000, open fills; shorts (variants only) borrow
  at 0.5%/yr (liquid ETFs are general collateral).

**Expected result (written before any number).** TRAIN: SPY ≈ +7-8%/yr with −55% in 2008; the
book ≈ +6-8%/yr at beta ≈ 0.3-0.4, alpha vs SPY ≈ +3 to +5%/yr (G1 pass), max DD −15 to −20%.
VALID: SPY ≈ +14%/yr; the book ≈ +6-8%/yr at beta ≈ 0.35-0.45, alpha ≈ 0 to +2%/yr with
bootstrap P < 0.9 (G2 likely FAIL). Shuffled-asset books are close to the real one in a bull
decade (G4 likely FAIL). The book beats SPY in 2008-09 and 2022 by a wide margin, in Q4 2018,
COVID and Feb-Apr 2025 mostly through diversification (G6 plausible; the down-month mean is the
uncertain half). Most likely verdict: a crisis profile without a significant edge over SPY.

**Premise check (TRAIN only, informational).** Monthly, each live risky asset: is the next
month's excess return over SHY (raw, vol-scaled, and SPY-beta-adjusted: minus beta_252 × SPY's
excess return) higher when the 12-month trend is on? Pooled on/off difference with t over
months, the MOP pooled sign regression, and the per-asset time-series correlation.

**Gates (adapted, `research.verdict`).** G1 TRAIN alpha vs SPY > 0; G2 VALID alpha vs SPY > 0
with block-bootstrap P >= 0.90; G4 VALID alpha beats >= 95% of 100 shuffled-asset books; G5 VALID
alpha > 0 with late fills (+50 bp a leg) AND with 2× costs; G6 beats SPY in every bear episode
(2008-09 2007-10-09 → 2009-03-09, Q4 2018, COVID, 2022, Feb-Apr 2025) AND mean return in SPY's
down months (TRAIN + VALID) > 0; G8 VALID alpha without its single best asset > 0. Reported but
not gated: alpha vs the equal-weight ETF bench.

**Shuffled-asset luck band (G4).** Per period, the trend STATES (on/off timing) are permuted
across the period's live risky assets; each asset keeps its own inverse-vol size, SHY absorbs
the rest. Timing, turnover, number held and the risk budget match; only which asset gets which
timing is random. 100 books on VALID.

**Noise (adoption yardstick).** Two perturbations, 3 copies each: drop 1 random risky asset per
period; move the monthly decision to the 6th / 11th / 16th trading day (rebalance timing luck).
The yardstick is the LARGER sd of Δalpha (vs SPY) on VALID.

**Optimisation.** Optuna TPE on TRAIN periods only, 120 trials, robust objective = mean − ½ sd
of the 24 per-period alphas vs SPY; space: lookback 63-378 bars (step 21), vol_com 20-120 (step
20), skip 0 / 21; PBO recorded. Adopted only if it beats U0 (paired, vs SPY) on TRAIN, VA and
VB.

**Iterations (>= 2, each pre-registered below before its numbers).** Adopted only if the paired
Δ alpha vs SPY >= 0 on TRAIN, VA and VB AND the VALID Δ exceeds twice the noise sd. Candidate
designs: blended lookbacks (1/3/6/12 months), a book-level vol target, long/short, equal risk per
asset class, a SPY-trend overlay on the equity sleeve.

**Variant budget:** at most 8 through the bar (tuner trials excluded), counted in §V.

## §1 Data (2026-10-22, result)

- Registered 24 ETFs (`upsert_securities`, security_type `etf` for the 15 new ones — SPY was
  already `etf`, so 16 rows are `etf` [corrected in round 2; this line said "the 18 new ones"];
  the UPSERT leaves an existing row's type alone, so XLE/XLF/XLI/XLK/XLV/XLY stay `equity` and
  GLD/SLV `metal`) and backfilled `tsim backfill --bar 1d --since 1993-01-01 --until 2026-10-16
  --provider yfinance`: 152,925 rows. Verified in the DB: every ETF yfinance-sourced from its
  inception (SPY 1993-01-29, sectors 1998-12-22, QQQ 1999-03-10, IWM 2000-05-26, EFA 2001-08-27,
  TLT/IEF/LQD/SHY 2002-07-30, EEM 2003-04-14, VNQ 2004-09-29, GLD 2004-11-18, DBC 2006-02-06,
  SLV 2006-04-28, UUP 2007-03-01, HYG 2007-04-11) to 2026-10-16, no NULL price, no session
  missing against SPY's calendar. SPY's total-return CAGR 1993-2026 = 10.8%.
- **Replaced rows.** SPY (2023-10-17 → 2026-10-15) and XLE/XLF/XLI/XLK/XLV/XLY (2023-10 →
  2026-10) were IBKR daily bars with adj_close = close (no dividends), joined to yfinance-adjusted
  history: SPY's stored series jumped +3.43% on 2023-10-17 (real move −0.005%). The full-range
  yfinance UPSERT replaced them with one consistent adjusted series. The 13,406 rows as they were
  are saved in `var/proposals_lab/results/etf_trend/db_backup_before_backfill.pkl`.
- Per-period universes: 17 assets in 2004-H1, 20 from 2006, 22 from 2007-H2, all 24 from
  2008-H2. LOCKED (2000-03): SPY + 9 sectors, QQQ from 2000-H2, IWM from 2001-H2, EFA from 2003;
  no bonds, no SHY (idle cash, 0%).

## §2 Premise check (TRAIN 2004-2015 only; result, informational)

Monthly, first trading day, every live risky ETF (144 months, 3,084 asset-months, 72% "on"):

| next-month excess return over SHY | on | off | on − off (within month) | t (months) |
|---|---|---|---|---|
| raw | +0.69% | −0.01% | +0.37% | 0.84 |
| vol-scaled (z) | +0.156 | −0.018 | +0.102 | 1.25 |
| SPY-beta-adjusted residual | +0.15% | −0.00% | +0.13% | 0.49 |

- The MOP statistic, sign × vol-scaled return pooled per month: +0.118 z/month, **t 2.77**.
- Per-asset time-series corr(sign, z): mean +0.071, positive for 19 of 23. Negative for TLT
  (−0.10), UUP (−0.11), IEF (−0.03), QQQ (−0.02); strongest for HYG +0.19, XLE +0.17, GLD +0.15.
- Read: time-series momentum exists in TRAIN (t 2.8), but the on-minus-off difference AFTER
  removing each asset's SPY beta is +0.13%/month at t 0.5. Most of what the trend state knows is
  when to hold market exposure, not which asset beats SPY. Bonds and the dollar did not trend in
  2004-15. Expectation unchanged: crisis protection yes, alpha vs SPY small.

## §3 U0 — the untuned default (result, 2026-10-22)

`research.py run U0 '{}' --costs2` (late fills and 2× costs books included).

| test | CAGR | SPY | beta (SPY) | **alpha vs SPY** | P | alpha vs EW | vol | max DD | late | 2× costs | best asset out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2004-15 | +6.84% | +6.14% | 0.20 | **+5.63%** | 1.00 | +4.59% | 7.0% | −9.1% | +4.58% | +5.38% | +4.84% |
| VA 2016-20 | +3.54% | +14.37% | 0.26 | −0.21% | 0.47 | −0.92% | 6.1% | −12.8% | −1.67% | −0.56% | −0.85% |
| VB 2021-26 | +5.09% | +14.67% | 0.30 | +0.74% | 0.70 | +0.18% | 6.0% | −10.7% | −0.67% | +0.40% | +0.26% |
| VALID 2016-26 | +4.36% | +14.51% | 0.28 | **+0.34%** | 0.62 | −0.30% | 6.1% | −12.8% | −1.10% | −0.01% | −0.22% |

- 81-95 orders a year; on average 66-74% of the book in risky ETFs, the rest in SHY. The fee
  gap (real per-order $1 minimum minus the engine's fee fraction, at a $30k account) is −13 to
  −18 bp/yr: the cost model OVER-charges this book's mostly full-size orders, so costs are
  conservative here. [Round 2 correction (review): both counts include each half-year's day-0
  rebuild, ~34 orders a year whose cost lab v1 never charged; on the charged orders the fee gap
  is −5 / −8 bp/yr (same sign). Restated on lab v2 in §R2.2: 74 / 92 orders a year, all
  charged, 40 / 58 without the restart; and ~40% of the late-fill cost below sits on SHY legs.]
- Bear episodes (book vs SPY): 2008-09 **+6.2% vs −55.8%**; Q4 2018 −6.1% vs −18.7%; COVID
  −12.2% vs −33.4%; 2022 −8.9% vs −25.3%; Feb-Apr 2025 −6.2% vs −18.6% (also 2011 −8.6% vs
  −19.6%, 2015-16 −3.3% vs −12.2%). Beats SPY in every episode.
- SPY's 89 down months: book −1.03% vs SPY −3.70% (positive in 19% of them; down-month beta
  0.13). **The down-month mean is negative: G6 fails on its second half.**
- Read: exactly the pre-registered profile. In TRAIN the alpha is crash avoidance at beta 0.2
  (2007-H2 +13%, 2009-H2 +10%) [round 2 correction (review): about two thirds of it; without
  the 2007-H2..2009-H1 periods the book still earned ≈ +2%/yr of low-beta carry (+1.96%, P 0.94;
  v2: +1.77%, §R2.2)]. In VALID the book earns what its 0.28 beta predicts and nothing
  more (P 0.62); it loses −1.1% with late fills. In a typical down month the equity sleeve is
  still on (a 12-month trend lags a one-month fall), so the book loses a third of SPY's fall.

**U0 noise** (`research.py noise U0`; Δ alpha vs SPY of each copy minus U0, pts/yr):

| perturbation | TRAIN | VA | VB | VALID |
|---|---|---|---|---|
| drop 1 random ETF a period (3 seeds) | −0.29 −0.07 +0.19 (sd 0.24) | sd 0.59 | sd 0.28 | +0.18 +0.29 −0.48 (sd 0.41) |
| decide on the 6th / 11th / 16th trading day | −0.65 −0.58 −0.78 (sd 0.10) | sd 0.94 | sd 0.65 | +0.26 −0.46 +0.34 (sd 0.44) |

The yardstick is the larger VALID sd, **0.44 pts**: an iteration must beat its base by more than
**0.87 pts** on VALID. (In TRAIN all three later decision days lose 0.6-0.8 pts against the
month's first day, a systematic turn-of-month timing difference, not path noise.)

**U0 luck** (100 shuffled-asset books per period: trend states permuted across the period's
ETFs, sizes kept): VALID real +0.34% vs median +0.51%, p95 +0.93%: **36th percentile (G4
FAIL)**. TRAIN (information, not a gate): real +5.63% vs median +4.73%, p95 +5.19%: 99th
percentile. Most of TRAIN's alpha (+4.7 of +5.6 pts) is the structure that a shuffled book
keeps (inverse-vol sizes, the number of ETFs on, SHY), and which asset gets which trend added
~0.9 pts in 2004-15 and nothing in 2016-26.

## §4 Optimisation (pre-registered in §0; started 2026-10-22)

`research.py tune T1 '{}' <space> 120`: lookback 63-378 step 21, vol_com 20-120 step 20, skip
0/21; TRAIN's 24 half-years, robust = mean − ½ sd of the per-period alphas vs SPY. U0's own
robust score on the same periods, for reference: computed from §3's per-period alphas.
U0's robust score on TRAIN: mean of the 24 per-period alphas +1.85%, sd 4.7% → **−0.50%**.

## §5 Iterations — pre-registered 2026-10-22, after §3 and before any of their numbers

What U0 showed: (1) no alpha vs SPY in VALID (+0.3%, P 0.62) and its asset timing is no
better than a shuffled book's (36th pct); (2) a negative down-month mean (−1.0%): a 12-month
trend lags a one-month fall and the equity sleeve is still on; (3) in TRAIN most of the alpha
is structure (low beta, SHY), not timing. Each change below is tested ONE at a time against the
current base (adoption bar in §0; noise yardstick 0.87 pts on VALID; a new base gets its own
noise check).

- **I1 long/short** (`mode: long_short`). Mechanism: an off asset now sits in SHY, so the book
  only steps aside; shorting downtrending assets earns in persistent falls (2008 equities and
  commodities, 2014-15 commodities, 2022 bonds and equities), the source of managed futures'
  crisis alpha (HOP 2017), and lowers beta, so SPY's down months should improve. Risks: the
  whipsaw of V-shaped rebounds (2009-03, 2020-04, 2023 bonds); the engine pays 0% on locked
  short proceeds (understates the book by ~ the T-bill rate × short gross). Expected: TRAIN Δ > 0,
  VALID Δ between 0 and +1.5 pts, down months better.
- **I2 equal risk per asset class** (`budget: class`). Mechanism: naive inverse vol over 23 ETFs
  gives the 15 correlated equity funds ~60% of the risk, so the book is mostly one equity bet and
  inherits its down months; one quarter each for equities, bonds, commodities and the dollar
  (inverse vol inside a class) gives the flight-to-quality assets (Treasuries, gold, the dollar)
  a real share. Expected: lower beta, better down months, VALID Δ >= 0, TRAIN Δ ≈ 0.
- **I3 blended lookbacks** (`blend: "21,63,126,252"`). Mechanism: averaging the 1/3/6/12-month
  states (HOP 2017 use 1/3/12) halves a falling asset's weight within one to three months instead
  of waiting for the 12-month return to turn. Expected: better down months, more turnover (late
  fills cost more), VALID Δ uncertain.

## §4 result — the tuner (variant 2, "T1")

- 120 trials (67 distinct points; TPE re-samples a small discrete space). Best: lookback 315,
  vol_com 120 (the space's upper bound), skip 0: robust +0.12% (U0 −0.50%), mean per-period alpha
  +2.09% (U0 +1.85%). **PBO 0.62**: the in-sample best lands in the out-of-sample bottom half more
  often than not. The surface: lookbacks 252-336 best (mean −0.1 to −0.5%), 63-231 clearly worse
  (−0.9 to −1.7%); vol_com barely matters (−0.5% at 120 vs −0.9% at 20); the winner is a spike
  (its neighbours 294 / 336 score −0.6% / −0.45%).
- Paired against U0 (alpha vs SPY): TRAIN −0.2 pts (P 0.42; the robust objective is not the
  stitched alpha), VA −0.6 (P 0.23), VB −0.5 (P 0.28), VALID −0.5 (P 0.17, 11/22 periods better);
  worse in all five bear episodes (COVID −15.9% vs −12.2%). **Not adopted: U0 stays.** The repo's
  pattern again: the optimizer found the luckiest point of a flat surface.

## §5 results — I1, I2, I3 (each paired against U0, alpha vs SPY, pts/yr)

| variant | TRAIN Δ (P) | VA Δ (P) | VB Δ (P) | VALID Δ (P, periods) | VALID alpha (P) | down months | adopted |
|---|---|---|---|---|---|---|---|
| T1 tuned (§4) | −0.2 (0.42) | −0.6 (0.23) | −0.5 (0.28) | −0.5 (0.17, 11/22) | −0.2% (0.45) | −1.10% | no |
| I1 long/short | −0.8 (0.23) | **−4.7 (0.00)** | −1.0 (0.26) | −2.7 (0.01, 2/22) | −2.4% (0.07) | −0.44% | no |
| I2 class budget | −0.7 (0.30) | −0.3 (0.41) | **+3.2 (0.998)** | +1.6 (0.98, 16/22) | +1.9% (0.94) | −0.49% | no (TRAIN, VA < 0) |
| I3 blend 1/3/6/12 | −1.4 (0.04) | **+2.2 (0.98)** | −0.8 (0.10) | +0.6 (0.77, 11/22) | +0.9% (0.82) | −0.95% | no (TRAIN, VB < 0; VALID < 0.87) |

- **I1 long/short** is the crisis book the literature describes: 2008-09 **+31.6%** (SPY −55.8%),
  2022 **+0.4%** (SPY −25.3%), COVID −6.4%, Q4 2018 −5.6%, Feb-Apr 2025 −4.3%; beta −0.02 (TRAIN)
  and 0.11 (VALID); 31% of SPY's down months positive. It pays for it in every normal stretch:
  2016-20 CAGR −3.6%/yr (0 of 10 periods better: the shorts of 2016's commodity and EM rebound,
  2019's bond rally), twice U0's orders (167-198 a year), −4.2% with late fills. The engine pays
  0% on the locked short proceeds (about −0.3 to −0.8 pts/yr against a real account), far less
  than the gap. [Round 2 correction (review): every I1 number in this section, §8 and the
  correlations is biased by a borrow over-charge (`build_weights` rescaled the kept shorts by a
  hair at almost every decision; the engine's surcharge turned each hair into up to 54% of the
  order): charged 0.44 / 0.81%/yr of the book against 0.12 / 0.16 owed, and about half the
  167-198 orders were those hairs. Fixed and re-run on lab v2 (§R2.3): TRAIN +5.12%, VA CAGR
  −3.1%/yr, VALID −2.2%, 2022 +1.3%, 83-104 orders a year; still rejected (VA Δ −3.9).]
- **I2 class budget** — every bear episode better than U0 (2022 −2.6% vs −8.9%, Q4 2018 −3.4%
  vs −6.1%), max DD −9.3%, VALID alpha +1.9% (P 0.94), +0.6% with late fills. But the whole gain
  is 2021-26: commodities contribute +3.4 pts/yr there (DBC +1.8, GLD +1.0, SLV +0.7) against
  U0's +1.1, the dollar +0.7 vs +0.3 (those are contribution LEVELS; the deltas vs U0 are DBC
  +1.25, GLD +0.63, SLV +0.40, UUP +0.46 pts/yr — added in round 2, review). In 2004-15 and 2016-20 the smaller equity share costs more
  than the commodity share earns. One regime (the 2021-22 inflation trade and the 2023-26 gold
  run), so the pre-registered rule rejects it; it is the best VALID number so far.
- **I3 blend** — faster exits help the sharp falls of 2016-20 (Q4 2018 −4.1%, COVID −8.1%) and
  whipsaw elsewhere (TRAIN −1.4, P 0.04), at 2.2× U0's orders (188-217 a year).

## §6 Iterations I4-I6 — pre-registered 2026-10-22, before their numbers

Base: U0 (nothing adopted). Same bar. These answer the task's remaining questions.

- **I4 Faber's 10-month average** (`rule: sma`, `lookback: 210`). Mechanism: the close against
  its 10-month average turns earlier than the 12-month point-to-point return (the average's
  centre of mass is 5 months back), the most widely used retail rule (Faber's GTAA). Expected
  from §4's surface (shorter return lookbacks lost in TRAIN): TRAIN Δ < 0, VALID Δ ≈ 0.
- **I5 the SPY trend alone** (`assets: SPY`: hold SPY while its 12-month return beats SHY's,
  else SHY). Mechanism: the purest market-timing overlay, the hedge an equity book would use; it
  avoids prolonged bears (2008, 2022) and pays the whipsaw of V-shaped ones (2020, 2025) and the
  missed rebounds. Answers "the trend signal on SPY alone as an overlay that hedges an equity
  book". Expected: TRAIN alpha vs SPY > 0 (2008), VALID alpha < 0 (2016, 2018-19, 2020, 2023
  re-entry), so Δ vs U0 negative in VALID.
- **I6 vol target** (`vol_target: 0.06`, about U0's own volatility, so it binds in the
  high-volatility half). Mechanism: volatility clusters and U0 lost most in high-vol months
  (COVID −12.2%); scaling the risky book down to a 6% ex-ante volatility (trailing 126-bar
  covariance, monthly, never above 1) cuts exposure after a spike. Expected: smaller COVID and
  2022 losses, missed rebounds (2009, 2020-H2), alpha Δ ≈ 0.

## §6 results — I4, I5, I6 (paired against U0, alpha vs SPY, pts/yr)

| variant | TRAIN Δ (P) | VA Δ (P) | VB Δ (P) | VALID Δ (P, periods) | VALID alpha (P) | down months | adopted |
|---|---|---|---|---|---|---|---|
| I4 Faber 10-month SMA | −0.9 (0.15) | **+3.2 (0.99)** | −0.5 (0.25) | +1.2 (0.91, 12/22) | +1.6% (0.91) | −0.93% | no (TRAIN, VB < 0) |
| I5 SPY trend alone (SPY/SHY) | −1.2 (0.26) | **−8.1 (0.01)** | −0.2 (0.46) | −4.0 (0.04, 12/22) | −3.7% (0.08) | −2.46% | no |
| I6 vol target 6% | −0.7 (0.08) | −0.0 (0.47) | −0.1 (0.27) | −0.1 (0.37, 15/22) | +0.3% (0.60) | −0.89% | no |

- **I4** behaves like I3: the faster signal sidesteps the sharp falls of 2016-20 (Q4 2018 −2.8%
  vs −6.1%, COVID −7.1% vs −12.2%) and gives it back in 2004-15 and 2021-26.
- **I5, the SPY trend as an equity overlay**, protected in exactly one episode: 2008-09 +0.3%
  vs SPY −55.8%. In the four VALID episodes it lost what SPY lost (Q4 2018 −18.4% vs −18.7%,
  COVID −32.8% vs −33.4%, 2022 −22.8% vs −25.3%, Feb-Apr 2025 −18.2% vs −18.6%): a 12-month
  trend turns only after a fast fall is over, then sells near the low and misses the rebound
  (max DD −39% in VALID against SPY's −34%). TRAIN alpha +4.5% (P 0.91, all 2008), VALID −3.7%
  (P 0.08), ~4 trades a year. As a hedge for an equity book this overlay is negative.
- **I6** the 6% target binds rarely (U0's own volatility is ~6-7%) and, decided monthly, never
  inside a crash month: COVID −12.2% unchanged.

## §V Variant count: 8 of 8 (U0, T1, I1, I2, I3, I4, I5, I6). Noise copies and luck books are
not variants.

## §7 FINAL CANDIDATE — declared 2026-10-22, before the verdict and before LOCKED is read

**U0** (`{}`): no change passed the pre-registered bar (Δ >= 0 on TRAIN, VA and VB and VALID
Δ > 0.87 pts). The best VALID numbers (I2 class budget +1.9%, I4 SMA +1.6%) each came from one
sub-period and lost elsewhere. LOCKED (2000-03, dot-com bust) is read once, for U0 only
(the final IS U0), with `research.py run U0 '{}' LOCKED --costs2 --read-locked`.

## §8 Verdict, bear table and the LOCKED read (results, 2026-10-22)

**Gates (U0, `research.py verdict U0`).**

| gate | value | passed |
|---|---|---|
| G1 TRAIN alpha vs SPY > 0 | +5.63% (P 1.00) | **yes** |
| G2 VALID alpha vs SPY > 0, P >= 0.90 | +0.34% (P 0.62) | no |
| G4 VALID beats >= 95% of shuffled-asset books | 36th percentile | no |
| G5 VALID late-fill AND 2× cost alpha > 0 | −1.10% late, −0.01% 2× costs | no |
| G6 beats SPY in every bear episode AND down-month mean > 0 | every episode yes; down months −1.03% | no |
| G8 VALID alpha without its best asset > 0 | −0.22% (the best "asset" is SHY's carry) | no |

**Verdict: no edge over SPY.** One gate of six. U0 is a low-beta (0.20-0.28) multi-asset book
that protects capital in every bear market (max DD −12.8% over 2004-26 against SPY's −56.3%) and
earns, after 2015, what its beta predicts plus the cash proxy's carry and nothing more.

**Bear table** (stitched TRAIN + VALID, 2004-01 → 2026-10; total return over each episode, %):

| | 2008-09 | Q4 2018 | COVID | 2022 | Feb-Apr 2025 | 2011 | 2015-16 | CAGR | max DD | down-month mean |
|---|---|---|---|---|---|---|---|---|---|---|
| SPY | −55.8 | −18.7 | −33.4 | −25.3 | −18.6 | −19.6 | −12.2 | 10.0 | −56.3 | −3.7 |
| EW ETFs | −37.1 | −11.2 | −24.6 | −16.1 | −10.8 | −13.6 | −10.1 | 8.2 | −39.1 | |
| **U0** | **+6.2** | −6.1 | −12.2 | −8.9 | −6.2 | −8.6 | −3.3 | 5.7 | −12.8 | −1.0 |
| I1 long/short | +31.6 | −5.6 | −6.4 | +0.4 | −4.3 | −9.2 | +0.4 | 2.0 | −24.1 | −0.4 |
| I2 class budget | +8.6 | −3.4 | −8.1 | −2.6 | −4.1 | −5.4 | −1.6 | 5.0 | −10.6 | −0.5 |
| I4 10-month SMA | +6.0 | −2.8 | −7.1 | −8.0 | −5.1 | −4.6 | −4.2 | 5.3 | −11.9 | −0.9 |
| I5 SPY trend alone | +0.3 | −18.4 | −32.8 | −22.8 | −18.2 | −19.2 | −10.5 | 7.3 | −39.1 | −2.5 |

U0 by calendar year: 2008 +4.3% (SPY −36.4%), 2018 −3.6% (−5.4%), 2022 −7.9% (−19.5%); it
trails SPY in 17 of the 23 years.

**LOCKED 2000-01 → 2003-12 (the dot-com bust), U0, read once (2026-10-22 23:50).** Equity ETFs
only (SPY, 9 sectors, QQQ from 2000-H2, IWM 2001-H2, EFA 2003), no SHY (idle cash at the
engine's 0%): CAGR −1.1%/yr vs SPY −6.1%/yr, beta 0.27, **alpha vs SPY +0.56% (P 0.59)**, alpha vs
the sectors' EW −0.52%; max DD −19.5% vs −47.5%; 2000-03-24 → 2002-10-09 **−17.5% vs SPY −47.1%**
(EW sectors −37.2%); 2000 −6.3% (SPY −10.1%), 2001 −9.8% (−11.2%), 2002 −2.0% (−20.9%), 2003
+15.4% (+23.0%); down months −1.24% vs −4.06%; late fills −0.50%, 2× costs +0.30%; 46% invested
on average. The same answer a third time: capital protection through low exposure, no timing
alpha. (Real T-bills paid ~1-6% then: on 54% idle cash that is ~+1.6%/yr the engine does not
pay, which would put the alpha near +2%, still with P well below 0.9.)

**Correlations** (daily returns, read-only from the other lab books' U0 files):

| | SPY | informed_flow_ls_p1 v6 | resid_momentum U0 | quality_junk U0 | peer_reversal U0 | earnings_drift U0 | adaptive_hedge U0 |
|---|---|---|---|---|---|---|---|
| U0 2016-21 | 0.83 | 0.20 | 0.19 | 0.10 | 0.17 | 0.06 | −0.37 |
| U0 2022-26 | 0.80 | 0.27 | 0.02 | −0.24 | 0.20 | 0.08 | −0.18 |
| I1 long/short 2022-26 | 0.24 | 0.26 | 0.28 | −0.10 | 0.03 | 0.14 | 0.20 |

U0 is a de-levered SPY in daily terms (corr 0.8 at a third of SPY's volatility); the long/short
version is the only diversifier (corr 0.24-0.43 with SPY) and shares a little with residual
momentum (0.28-0.34).

[Round 2 corrections (review): the rows labelled "2016-21" cover 2016-07..2021-12 (the span
the other lab books trade); from 2016-01, U0/SPY is 0.81 and I1/SPY 0.36, not 0.83 / 0.43.
I1's 0.26 with informed_flow_ls_p1 is the 2022-26 value only (0.04 in 2016-07..2021-12), and
the I1 rows carry the borrow bias above. Recomputed on the v2 books in §R2.5.]

## §9 Conclusions

1. Cross-asset trend following on these 24 ETFs is a **risk reducer, not an alpha source**.
   It cut the loss of every bear market by 63% to all of it (2008-09 +6.2% vs −55.8%; Q4 2018,
   COVID, 2022, Feb-Apr 2025 by 63-67%; 2000-02 −17.5% vs −47.1%) and beat SPY on alpha only in 2004-15 (+5.6%), where 2008 does the work. In 2016-26
   (+0.3%, P 0.62) and 2000-03 (+0.6%, P 0.59) the alpha is zero within noise, the asset timing
   loses to a shuffled book (36th percentile), and what is left is SHY's carry.
2. Every change that helped one era cost another: faster signals (blend, 10-month SMA) win the
   sharp falls of 2016-20 (VA +2.2 / +3.2 pts) and lose 2004-15; equal class budgets win
   2021-26's commodity and gold run (VB +3.2 pts) and lose 2004-20; long/short is the best crisis
   book (2008 +31.6%, 2022 +0.4%) and the worst in normal years (2016-20 −4.7 pts). The tuner's
   pick lost to U0 out of sample (PBO 0.62).
3. **The SPY trend as an overlay does not hedge an equity book on this record**: it sidestepped
   2008 only and lost the full fall in Q4 2018, COVID, 2022 and Feb-Apr 2025 (a 12-month signal
   turns after a fast fall is over), at −3.7%/yr alpha in 2016-26. The repo's earlier finding
   (the 200-day gate cost −5.6 pts) stands on 22 more years of data.
4. What could carry over to a cross-strategy book: U0 or I2 as a low-correlation crisis sleeve
   (I2: 2022 −2.6% vs SPY −25.3%, max DD −10.6%, alpha +1.9% in VALID but regime-dependent), and
   I1's short side as a 2008/2022-type hedge. Neither adds return in a bull market.

(Round 1 conclusions as written on 2026-10-22, lab v1. Round 2 restates every number on lab v2
below; the conclusions hold, see §R2.8. Do not reuse round 1's I1.pkl returns anywhere: use
I1v2.)

## Round 2: lab v2 re-evaluation (2026-10-23)

### §R2.0 Pre-registration (written before any round-2 backtest number)

**Why.** Round 1's adversarial review found one critical defect in this strategy (I1's borrow
over-charged 4-5×: `build_weights` rescaled the KEPT weights whenever the gross exceeded 0.98,
so nearly every held short moved by a hair and the engine's borrow surcharge, a fraction of the
target change, exploded) and five minor ones (the band defeated by the same rescale; order
counts and the fee gap inflated by the uncharged restart bar; ~40% of the late-fill cost on SHY
legs; "all crash avoidance" overstated; the DB side effect of the backfill; label slips). The lab
itself had five defects (README "Lab v2"); lab v2 fixes them in `lab.book` / `lab.run_portfolio`
/ `lab.borrow_fees` and writes to `results2/`.

**What changes (code, before any number).**
1. `strategy.build_weights`: the gross is fitted by scaling only the assets that trade anyway
   (kept weights stay bit for bit; done in an interim fix at 00:27, see below), and NEW
   `min_change: 0.0025` (absolute share of the book, SHY included): a smaller target change
   keeps the held weight and a smaller entry is not opened; an exit always trades. A cost guard
   demanded by the review, not tuned. Measured on the weights alone (no returns): under the
   kept-weight fix it binds on 1-2 SHY re-sizes and 0-2 risky re-sizes per decade in U0 and on
   ~25 SHY slivers per decade in I1. [Round 2 repair, pass 2 (verifier, `verify/check2/
   minchange3.log`, re-run and confirmed): 1-3 SHY re-sizes and 0-1 risky re-sizes per decade
   in U0 (TRAIN 1 SHY order in 12 years; VALID 3 SHY and 1 risky, ≈ 2.8 SHY a decade) and
   21-25 SHY slivers per decade in I1. The pre-registration text above is kept as written.]
2. `research.py` follows lab v2: every book is `lab.book` as the lab runs it (its runner, day 0
   counted from the initial cash, `lab.borrow_fees`); SPY's buy-and-hold is built the same way
   (bought at the window's first open at the book's costs, day 0 counted: `spy_returns`).
   Cross-checks, never gates: `exact borrow` (books with shorts: the borrow owed on the REALIZED
   shorts, 0.5%/yr × short value × calendar days), `late risky` (+50 bp on the risky legs only,
   through `lab.run_portfolio`), and "continuing" orders and fee gap (without each half-year's
   day-0 rebuild, which v2 now charges but one continuous account would not place).
3. Unchanged: the ETF panel (a fresh load from today's DB equals round 1's cache bit for bit;
   no row without SPY, so lab defect 1 does not reach this panel; defect 5 concerns the stock
   universes only), the periods, the adapted gates, the tuner's space and 120 trials, and the
   adoption bar: paired Δ alpha vs SPY ≥ 0 on TRAIN, VA and VB AND VALID Δ > 2 × the VALID noise
   sd, re-measured on U0v2.

**Disclosure.** An interim fix run on lab v1 (labels U0r, T1r, I1r-I6r in `results/etf_trend/`,
2026-10-23 00:34-00:54: kept-weight fix + exact borrow, no `min_change`) was never written up;
its report table was seen before this section (e.g. U0r VALID alpha +0.3%, I1r −1.8%, I2r
+1.9%, tuned pick again lookback 315 / vol_com 120). It decides nothing; it is reported below as
the "strategy fix alone" step between v1 and v2.

**Plan (labels end in v2).** `run U0v2 '{}' --costs2` → `noise U0v2` → `luck U0v2` (VALID; TRAIN
for information) → `tune T1v2` (TRAIN, 120 trials, same space) and `run T1v2 <best>` →
`run I1v2 … I6v2` (the round-1 overrides) → `report U0v2 <all>` → `bear` → `verdict U0v2`. Every
round-1 adoption decision is re-checked (none was adopted; the strongest rejections were I2
class budget, VB +3.2 / VALID +1.6, and I4 10-month SMA, VA +3.2 / VALID +1.2).
**LOCKED (2000-03)** was read once in round 1 for U0. It is re-measured ONCE, for U0v2 (and for
a new final only if a decision changes, which would first be declared here): a disclosed
re-measurement on corrected accounting that selects nothing.

**Expected (before any number).** U0v2 within ~0.2 pts of round 1 (day 0 adds one rebuild's
costs a period to the book AND to SPY's buy-and-hold; ~−0.1 to −0.2 pts of alpha); its late-fill
alpha ~0.5-1 pt lower (the restart is now charged at +50 bp); I1v2's VALID alpha ~+0.5-0.7 pts
better than round 1 (the over-charge gone), still rejected (VA ≈ −4 pts); the other variants'
deltas within ~±0.3 pts of round 1; no decision changes; final = U0v2; status "no edge".
Variant budget: round 1 used 8 of 8; re-measurements are not variants; at most 2 new
pre-registered iterations (to 10) if v2 opens an obvious direction.

### §R2.1 Data and code (result, 2026-10-23)

- The ETF panel loaded fresh from today's DB (`results2/etf_trend/panel.pkl`) equals round 1's
  cache bit for bit (8,487 sessions, 1993-01-29 → 2026-10-16, every frame, the NaN pattern
  too); no row without a SPY bar, no weekend row. Lab defects 1 (holiday rows) and 5 (ETFs in
  the stock universes) do not reach this book; defects 2 (day 0), 3 (call order) and 4 (borrow)
  do, through `lab.book`.
- `min_change` measured on the weights alone (no returns read): it removes 1-3 SHY re-sizes and
  0-1 risky re-sizes per decade in U0 (1 order in TRAIN, 4 in VALID), 21-25 SHY slivers per
  decade in I1 [round 2 repair, pass 2: this said "1-2 SHY ... 0-2 risky ... ~25"; the
  verifier's count, re-run: U0 TRAIN 1 SHY, VALID 3 SHY + 1 risky]. New tests: a kept
  weight (risky and SHY) is bit-identical across decisions and places no order; a hair entry
  is never opened; an exit always trades; every non-exit order in a synthetic long_short book
  is ≥ `min_change`; SPY's benchmark compounds to SPY bought at the first open, costs paid.
- One bug found on the way (before any noise number was read): the drop-one-ETF noise copy
  dropped SPY itself from a period panel and the new SPY benchmark read SPY from the book's
  window. Fixed (SPY's bars come from the full panel, as in round 1); U0v2 re-ran bit-identical.

### §R2.2 U0v2 (result)

`round2.py u0` (= `research.py run U0v2 '{}' --costs2`), alpha vs SPY, %/yr:

| test | U0v2 alpha (P) | round 1 U0 | interim U0r | beta | CAGR | SPY | max DD | late | late risky | 2× costs | best asset out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TRAIN 2004-15 | **+5.25 (1.00)** | +5.63 | +5.59 | 0.20 | +6.48 | +6.23 | −8.7 | +3.17 | +3.82 | +4.75 | +4.46 |
| VA 2016-20 | −0.52 (0.41) | −0.21 | | 0.26 | +3.38 | +14.91 | −12.9 | −2.98 | −2.07 | −1.11 | −1.16 |
| VB 2021-26 | +0.44 (0.62) | +0.74 | | 0.30 | +4.63 | +14.10 | −11.0 | −2.04 | −1.13 | −0.16 | −0.04 |
| VALID 2016-26 | **+0.02 (0.51)** | +0.34 | +0.34 | 0.28 | +4.04 | +14.46 | −12.9 | −2.45 | −1.54 | −0.57 | −0.53 |

- **Where the change comes from.** The kept-weight fix alone (U0r, lab v1) moves U0 by −0.04 /
  0.00 pts; lab v2's day 0 moves it by −0.34 / −0.32: each half-year now pays its opening
  rebuild (~0.24 pts of CAGR a year at 0.12% a leg) and its first day's open-to-close move, and
  SPY's benchmark is built the same way (its CAGR +0.09 / −0.05). `min_change` changes U0 by 1
  order in TRAIN and 4 in VALID.
- **Against the pre-registered expectation** (−0.1 to −0.2 pts, late fills −0.5 to −1): the
  day-0 effect is about twice as large, −0.34 / −0.32 pts [round 2 repair: this said "−0.32
  to −0.38"; the −0.38 was round 1 → round 2 in TRAIN, the strategy fix's −0.04 included.
  Counted one at a time (`day0_repair.log`), the book's own day 0 moves the alpha by −0.33 /
  −0.35 and SPY's by −0.02 / +0.01], and late fills lost 1.35-1.4 pts
  more. The book's day 0 averages −0.15% / −0.16% (TRAIN / VALID): the rebuild costs −0.12%, its
  open-to-close move about −0.03%. SPY's day 0 averages +0.05% / −0.02%: the same −0.12% of
  costs, but its own open-to-close move on the first day of a half-year was +0.17% / +0.10%,
  and the book at beta 0.2-0.3 keeps little of that. Under late fills each rebuild costs a
  further ~0.5% (−1.3 pts a year in the book's day-0 returns). I1's alpha improved less than
  expected (+0.2-0.3 instead of +0.5-0.7) for the same reason; its paired Δ vs U0 improved by
  the expected +0.5.
- **Orders, all charged under v2:** 74 a year in TRAIN, 92 in VALID (round 1: 81 / 95; what
  went were mostly the rescale's hair re-trims, about 7 a year in TRAIN and 2-3 in VALID
  [round 2 repair, pass 2: this said "the ~7 a year"; that is TRAIN's drop, 81.0 → 74.2, against
  VALID's 94.7 → 92.4; the round-1 rule's hair re-trims number 80 in TRAIN and 20 in VALID,
  `verify/check2/kept3.log`]). Without each half-year's day-0
  rebuild ("continuing", what one continuous account places): **40 / 58 a year**. Fee gap (real
  per-order $1 minimum minus the engine's fraction, $30k account): −15 / −17 bp/yr on all
  orders, **−7 / −9 bp/yr on continuing orders**: the cost model over-charges this book.
- **Late fills:** −2.1 (TRAIN) / −2.5 (VALID) pts of alpha, of which 0.66 / 0.90 pts (31% / 37%)
  is the +50 bp on SHY legs; on the risky legs only VALID is −1.54% (G5 still fails). v2 charges
  the restart at +50 bp too; one continuous book (no restart; `analysis2.py continuous`):
  VALID alpha +0.14% (P 0.55), late −1.37%, late risky −0.77%, 2× −0.23%; TRAIN +5.67%,
  late +4.60%. The stitched record is conservative by ~0.1-0.4 pts at base costs and ~1.1-1.4
  pts under late fills (VALID 1.08, TRAIN 1.43) [round 2 repair, pass 2: this said "~1.1 pts",
  VALID's figure only; `verify/check2/continuous3.log`]; no gate depends on it.
- **TRAIN attribution** (years = time held): all +5.32%; without the 2007-H2..2009-H1 crash
  periods **+1.77%/yr (P 0.93, beta 0.44)**; the four crash periods alone +2.65% at beta −0.01.
  About 70% of TRAIN's alpha (in alpha-years) is the 2007-09 crash; the other ten years carry
  ≈ +1.8%/yr of low-beta carry (bonds, SHY, gold in a falling-rate decade). Round 1's "all crash
  avoidance" overstated it (review); "most of it is the 2008 crash" stands.
- **Noise** (`noise U0v2`; Δ alpha vs U0v2, pts/yr): drop one ETF a period: TRAIN sd 0.23, VALID
  sd 0.40 (VA 0.57, VB 0.28); decision on the 6th / 11th / 16th trading day: TRAIN −0.61 −0.56
  −0.75 (sd 0.10: the turn-of-month effect again), VALID +0.28 −0.44 +0.35 (sd 0.43; VA 0.96,
  VB 0.68). Yardstick 0.43 → **the bar stays VALID Δ > 0.87 pts.**
- **Luck** (100 shuffled-asset books a period): VALID real +0.02% vs median +0.21%, p95 +0.65%:
  **31st percentile (G4 FAIL;** round 1: 36th). TRAIN (information): 99th percentile (real +5.25
  vs median +4.50, p95 +4.93).

### §R2.3 The tuner and every round-1 decision re-checked (result)

**Tuner (`tune T1v2`, studies2):** 120 trials, 73 distinct points. Best again lookback 315,
vol_com 120 (the space's bound), skip 0: robust −0.17% (U0v2 −0.78%), mean per-period alpha
+1.87% (U0v2 +1.61%), **PBO 0.55** (round 1 0.62). The surface is the same: lookbacks 252-336
best (mean robust −0.25 to −0.83%), 63-231 worse (−1.2 to −2.0%); vol_com 120 −0.59% vs 20
−1.24%; skip barely matters; 315 is a spike (with vol_com 120 / skip 0: 294 −0.87%, 315
−0.17%, 336 −0.76%).

Paired against U0v2 (alpha vs SPY, pts/yr; bar: Δ ≥ 0 on TRAIN, VA and VB and VALID Δ > 0.87):

| variant | TRAIN Δ (P) | VA Δ (P) | VB Δ (P) | VALID Δ (P, periods) | VALID alpha (P) | down months | round 1 VALID Δ | adopted |
|---|---|---|---|---|---|---|---|---|
| T1v2 tuned | −0.1 (0.45) | −0.5 (0.25) | −0.4 (0.32) | −0.4 (0.20, 12/22) | −0.4% (0.36) | −1.14% | −0.5 | no |
| I1v2 long/short | −0.1 (0.44) | **−3.9 (0.00)** | −0.7 (0.31) | −2.2 (0.03, 3/22) | −2.2% (0.09) | −0.40% | −2.7 | no |
| I2v2 class budget | −0.6 (0.30) | −0.2 (0.42) | **+3.1 (0.998)** | +1.5 (0.98, 16/22) | +1.6% (0.91) | −0.52% | +1.6 | no (TRAIN, VA < 0) |
| I3v2 blend 1/3/6/12 | −1.3 (0.04) | **+2.2 (0.98)** | −0.9 (0.08) | +0.5 (0.74, 10/22) | +0.5% (0.69) | −0.99% | +0.6 | no (TRAIN, VB < 0) |
| I4v2 10-month SMA | −0.8 (0.15) | **+3.2 (0.99)** | −0.6 (0.20) | +1.2 (0.89, 12/22) | +1.2% (0.85) | −0.96% | +1.2 | no (TRAIN, VB < 0) |
| I5v2 SPY trend alone | −1.0 (0.28) | **−7.7 (0.02)** | +0.2 (0.52) | −3.6 (0.07, 13/22) | −3.6% (0.08) | −2.49% | −4.0 | no |
| I6v2 vol target 6% | −0.6 (0.10) | −0.0 (0.47) | −0.1 (0.23) | −0.1 (0.34, 11/22) | −0.1% (0.47) | −0.93% | −0.1 | no |

- Orders a year, TRAIN / VALID, all charged under v2 (continuing, without the day-0 rebuild):
  U0v2 74 / 92 (40 / 58), T1v2 62 / 81 (29 / 46), I1v2 83 / 104 (39 / 56), I2v2 71 / 87
  (37 / 52), I3v2 187 / 214 (147 / 170), I4v2 86 / 102 (54 / 68), I5v2 3 / 4 (1 / 2), I6v2
  88 / 103 (55 / 68). Late-fill VALID alpha: U0v2 −2.45%, I2v2 −0.82%, I4v2 −1.65%, I3v2
  −2.52%, I1v2 −4.79%; 2× costs: −0.57%, +0.98%, +0.50%, −0.20%, −2.82% (`analysis2.log`).
- **No decision changes.** Every variant fails the bar it failed in round 1, on the same
  sub-period; the strongest rejections are again I2 (VB +3.1 only) and I4 (VA +3.2 only).
- **I1v2, the corrected long/short book.** Borrow charged by the lab **0.12%/yr of the book in
  TRAIN and 0.16%/yr in VALID, equal to the borrow owed on the realized shorts (0.12 / 0.16)**;
  round 1 charged 0.44 / 0.81. The `exact borrow` book gives the same alpha (TRAIN +5.11 vs
  +5.12, VALID −2.19 vs −2.19). Orders: 83 / 104 a year (round 1: 167 / 186; continuing 39 / 56).
  Alpha: TRAIN +5.12% (round 1 +4.83; interim I1r +5.55), VA −4.41% (CAGR −3.1%/yr; round 1
  −3.6%), VB −0.28%, **VALID −2.19% (round 1 −2.40%)**. Episodes: 2008-09 +31.4%, 2022 +1.3%
  (round 1 +0.4%), COVID −6.2%, Q4 2018 −6.0%, Feb-Apr 2025 −4.4%; 33% of SPY's down months
  positive (mean −0.40%). Still rejected: whipsawed in 2016-20 (0 of 10 periods better).
- **I2's VB gain, as deltas** (P&L contribution, I2v2 minus U0v2, pts/yr in VB): DBC +1.25, GLD
  +0.63, SLV +0.39, UUP +0.47 (levels 1.77 / 0.95 / 0.66 / 0.74 vs U0v2's 0.52 / 0.32 / 0.27 /
  0.27); every equity fund gives back 0.1-0.26. Round 1 quoted the levels (review).
- **v2 opens no new direction**: the era trade-offs of round 1 reappear number for number, so
  no new iteration is run (variants used: 8 of 8 in round 1; none added).

### §R2.4 FINAL CANDIDATE — declared 2026-10-23, after §R2.3 and before LOCKED is re-read

**U0v2** (`{}`: U0's config on the review-fixed code; `min_change` 0.0025, band 0.25). Nothing
passed the bar on v2. LOCKED (2000-03) is re-measured ONCE for U0v2 with `research.py run U0v2
'{}' LOCKED --costs2 --read-locked`: a disclosed re-measurement of the window read once in round
1 for U0 (2026-10-22 23:50), on corrected accounting; it selects nothing.

### §R2.5 Verdict, bear table, LOCKED re-measurement and correlations (results, 2026-10-23)

**Gates (U0v2, `research.py verdict U0v2`).**

| gate | value | passed |
|---|---|---|
| G1 TRAIN alpha vs SPY > 0 | +5.25% (P 1.00) | **yes** |
| G2 VALID alpha vs SPY > 0, P >= 0.90 | +0.02% (P 0.51) | no |
| G4 VALID beats >= 95% of shuffled-asset books | 31st percentile | no |
| G5 VALID late-fill AND 2× cost alpha > 0 | −2.45% late (risky legs only −1.54%), −0.57% 2× | no |
| G6 beats SPY in every bear episode AND down-month mean > 0 | every episode yes; down months −1.07% | no |
| G8 VALID alpha without its best asset > 0 | −0.53% (the best "asset" is SHY's carry, +0.52 pts/yr) | no |

**Verdict unchanged: no edge over SPY, 1 gate of 6.**

**Bear table** (stitched TRAIN + VALID, 2004-01 → 2026-10, total return over each episode, %;
"SPY" is the per-period benchmark the gate uses, bought at each half-year's first open like the
book; "SPY close to close" is SPY's own record over the same dates, for reference):

| | 2008-09 | Q4 2018 | COVID | 2022 | Feb-Apr 2025 | 2011 | 2015-16 | CAGR | max DD | down-month mean |
|---|---|---|---|---|---|---|---|---|---|---|
| SPY | −54.6 | −18.7 | −33.4 | −24.3 | −18.6 | −18.6 | −12.3 | 10.0 | −55.0 | −3.7 |
| SPY close to close | −55.2 | −19.3 | −33.7 | −24.5 | −18.8 | −18.6 | −13.0 | | | |
| EW ETFs | −36.4 | −11.2 | −24.6 | −15.5 | −10.8 | −13.0 | −10.5 | 8.1 | −37.9 | |
| **U0v2** | **+4.6** | −6.1 | −12.3 | −8.7 | −6.2 | −8.1 | −3.5 | 5.3 | −12.9 | −1.1 |
| I1v2 long/short | +31.4 | −6.0 | −6.2 | +1.3 | −4.4 | −7.9 | +0.8 | 2.2 | −22.6 | −0.4 |
| I2v2 class budget | +7.8 | −3.4 | −8.2 | −2.6 | −4.1 | −5.2 | −1.8 | 4.6 | −10.6 | −0.5 |
| I4v2 10-month SMA | +4.5 | −2.8 | −7.1 | −8.0 | −5.1 | −4.1 | −4.4 | 4.9 | −12.3 | −1.0 |
| I5v2 SPY trend alone | −0.6 | −18.4 | −32.8 | −21.8 | −18.2 | −18.2 | −10.6 | 7.2 | −38.1 | −2.5 |

U0v2 by calendar year: 2008 +3.7% (SPY −36.4%), 2018 −3.6% (−4.5%), 2022 −7.7% (−18.5%); it
trails SPY in 17 of the 23 years. 89 SPY down months: book −1.07% vs −3.69%, positive in 19%.

**LOCKED 2000-01 → 2003-12, U0v2, re-measured once (2026-10-23 09:05; read once for U0 in round
1).** CAGR −1.63%/yr vs SPY −6.37%/yr, beta 0.27, **alpha vs SPY +0.11% (P 0.51)** (round 1
+0.56%, P 0.59); alpha vs the sectors' EW −0.76%; max DD −19.7% (SPY −48.8%); 2000-03-24 → 2002-10-09
**−17.7% vs SPY −48.5%** (EW −39.7%); 2000 −7.5% (SPY −10.9%), 2001 −10.5% (−12.7%), 2002
−2.2% (−22.5%), 2003 +15.6% (+27.5%); down months −1.27% vs −4.04%; late fills −1.34% (no SHY
in this window: all legs are risky), 2× costs −0.24%; 46% invested, 30 orders a year (23
continuing). **The lower alpha is the book's own day 0, not the benchmark** [round 2 repair
correction (verifier): this sentence first said "mostly the benchmark"; the 2×2 on the saved
rows (`day0_repair.log`, no new LOCKED run) shows the opposite]. Zeroing the book's day 0 gives
+0.61%, so counting it costs −0.50 pts a year: its day 0 averages −0.26% a half-year, of which
the rebuild's costs are only −0.05% (43% invested on those days, no SHY) and the first-day
moves −0.21%, most of it 2000-01-03, when the 98%-invested book fell −1.98% with SPY (−2.01%),
and 2001-01-02 (−1.12%). Counting SPY's day 0 LOWERED SPY's CAGR (−6.12% → −6.37%: the −0.12%
buy cost a half-year against first-day open-to-close moves of +0.01% on average) and so RAISED
the alpha by +0.07 pts. 2003-01-02's +2.50% open-to-close did lift SPY's 2003 (+23.0% →
+27.5%), but 2000-02 each lost 0.8-1.6 pts. Both zeroed: +0.57%, round 1's +0.56%. Same answer
a third time: capital protection through low exposure, no timing alpha. It selects nothing.

**Correlations** of daily returns (`corr2.py`; the other books' round-2 files as of 09:00,
read-only; the lab books cover 2016-07..2021-12 (E0+TB+T3) and 2022-01..2026-10 (T1)):

| | SPY | informed_flow_ls_p1 v6_si | resid_momentum U0v2 | quality_junk U0v2 | peer_reversal U0v2 | earnings_drift U0v2 | adaptive_hedge U0v2 |
|---|---|---|---|---|---|---|---|
| U0v2 2016-01..2021-12 | 0.81 | | | | | | |
| U0v2 2016-07..2021-12 | 0.83 | 0.17 | 0.19 | 0.10 | 0.15 | 0.04 | −0.38 |
| U0v2 2022-01..2026-10 | 0.80 | 0.25 | 0.01 | −0.24 | 0.17 | −0.01 | −0.23 |
| I1v2 2016-01..2021-12 | 0.35 | | | | | | |
| I1v2 2016-07..2021-12 | 0.41 | 0.02 | 0.31 | 0.10 | 0.09 | 0.08 | −0.12 |
| I1v2 2022-01..2026-10 | 0.25 | 0.27 | 0.26 | −0.11 | 0.03 | 0.10 | 0.19 |
| I2v2 2016-01..2021-12 | 0.66 | | | | | | |
| I2v2 2022-01..2026-10 | 0.47 | 0.32 | 0.12 | −0.23 | 0.05 | 0.06 | −0.00 |

### §R2.6 Round 1 (lab v1) → round 2 (lab v2): the headline numbers

This book's tests map onto the lab's as TRAIN = TRAIN 2004-15, T1/T2 = VA 2016-20 / VB 2021-26
(the two halves VALID's adoption bar must both pass), T4 = LOCKED 2000-03.

| | round 1 (v1) | round 2 (v2) |
|---|---|---|
| TRAIN alpha (P) | +5.63% (1.00) | +5.25% (1.00) |
| VA / VB alpha | −0.21% / +0.74% | −0.52% / +0.44% |
| VALID alpha (P) | +0.34% (0.62) | +0.02% (0.51) |
| LOCKED (T4) alpha (P) | +0.56% (0.59) | +0.11% (0.51) |
| VALID late / 2× costs | −1.10% / −0.01% | −2.45% / −0.57% (risky-only late −1.54%) |
| luck (VALID percentile) / noise bar | 36th / 0.87 pts | 31st / 0.87 pts |
| gates | 1 of 6 (G1) | 1 of 6 (G1) |
| 2008-09 / Q4 2018 / COVID / 2022 / Feb-Apr 2025 | +6.2 / −6.1 / −12.2 / −8.9 / −6.2 | +4.6 / −6.1 / −12.3 / −8.7 / −6.2 |
| SPY in those episodes | −55.8 / −18.7 / −33.4 / −25.3 / −18.6 | −54.6 / −18.7 / −33.4 / −24.3 / −18.6 |
| down-month mean (SPY) | −1.03% (−3.70%) | −1.07% (−3.69%) |
| orders a year TRAIN / VALID | 81 / 95 (34-35 on an uncharged restart) | 74 / 92, all charged (40 / 58 continuing) |
| tuner pick, PBO | lookback 315, vol_com 120, skip 0; 0.62 | the same; 0.55 |
| I1 VALID alpha; borrow charged / owed | −2.40%; 0.81 / 0.16 %/yr | −2.19%; 0.16 / 0.16 %/yr |
| decisions | nothing adopted; final U0 | nothing adopted; final U0v2 |
| status | no edge | no edge |

### §R2.7 Review issues → fix → effect

1. **Critical — I1's borrow over-charged 4-5×** (`build_weights` rescaled the kept weights;
   the engine's surcharge exploded on the hairs). Fix: the gross is fitted on the trading assets
   only (kept weights bit for bit) plus `min_change`; lab v2's `borrow_fees` defers small
   re-trims; `exact borrow` cross-check. Effect: borrow charged = owed (0.12 / 0.16%/yr);
   I1's orders 167 / 186 → 83 / 104 a year; VALID alpha −2.40 → −2.19% (interim, lab v1: −1.81);
   2022 +0.4 → +1.3%; paired VALID Δ vs U0 −2.7 → −2.2; still rejected (VA Δ −3.9, 0 of 10
   periods better). Round 1's I1 returns must not be reused; I1v2 replaces them.
2. **Minor — the band defeated by the same rescale.** Same fix; tests that a kept weight
   (risky and SHY) is bit-identical and places no order, and that every non-exit order of a
   synthetic long/short book is ≥ `min_change`. Effect: U0's orders 81 → 74 a year (TRAIN),
   95 → 92 (VALID); alpha −0.04 / 0.00 pts; `min_change` itself: 1 and 4 orders.
3. **Minor — orders and the fee gap counted the uncharged restart; ~40% of the late-fill cost
   on SHY legs.** Lab v2 charges day 0; `research.py` reports continuing orders and fee gap and a
   risky-legs-only late stress. Effect: 74 / 92 orders a year (40 / 58 continuing), fee gap
   −7 / −9 bp/yr on continuing orders (the model over-charges); SHY's share of the late cost
   31% / 37%; G5 fails on every reading (risky-only −1.54%, continuous book −1.37%).
4. **Minor — "all crash avoidance" overstated.** Rephrased (§3 note, README). v2: about 70% of
   TRAIN's alpha is the 2007-09 crash; +1.77%/yr (P 0.93, beta 0.44) in the other ten years.
5. **Minor — the backfill's side effect on the shared DB.** The README's "Data" section now
   states it in full (rows, symbols, dates, backup, who reads them, the re-splice risk) and
   proposes an adj_close/close splice check for future refreshes (proposal only: it belongs in
   the refresh path, not in this folder). The lab README's "Lab v2" section records it too; lab
   v2 excludes the six "equity"-typed SPDRs from the stock universes and rebuilt `spy.pkl` from
   the current DB.
6. **Minor — label slips.** Corrected in place (marked "round 2 correction") in §1 (16 `etf`
   rows: 15 new + SPY), §5 (I2's deltas, not levels), §8 (the "2016-21" correlations are
   2016-07..2021-12; I1's 0.26 with informed_flow_ls_p1 is 2022-26 only); v2 values in §R2.5.

### §R2.8 Conclusions (round 2)

1. **The verdict stands on corrected accounting: no edge over SPY; a crash-protection
   sleeve.** U0v2 earns +5.25%/yr of alpha in 2004-15 (P 1.00, about 70% of it the 2007-09
   crash), +0.02% in 2016-26 (P 0.51) and +0.11% in 2000-03 (P 0.51); its asset timing sits at
   the 31st percentile of shuffled books; 1 of 6 gates. It beat SPY in every bear episode
   (2008-09 +4.6% vs −54.6%, 2000-02 −17.7% vs −48.5%) and lost −1.07% in SPY's average down
   month.
2. Lab v2 lowered the long/flat books' alphas by 0.24-0.45 pts a year (U0, round 1 → round
   2: −0.38 TRAIN, −0.31 VALID, −0.45 LOCKED; the strategy fix's share −0.04 / 0.00 / +0.01):
   each half-year now pays its rebuild (~0.12% of the book, ~0.24 pts of CAGR a year in
   TRAIN / VALID) and its first day's move. Almost all of it is the book's own day 0 (−0.33 /
   −0.35 / −0.50 pts); counting SPY's day 0 moved the alpha by −0.02 / +0.01 / +0.07 [round 2
   repair: this item first said "the book and SPY alike"]. The SPY/SHY switch (I5) moved by
   −0.26 / +0.10, all of it lab v2: it held SPY on 20 of its 22 VALID first days and SHY on
   the other two, when SPY's day 0 lost 0.46% and 1.04% [round 2 repair, pass 2: "−0.32 VALID"
   and "−0.25" were rounding slips; `repair2_lab_effect.log`]. The strategy fix moved U0 by
   0.04 pts and I1 by +0.6-0.7 pts (+0.2-0.3 net of v2). No adoption decision changed and the
   tuner found the same spike.
3. The corrected long/short book (I1v2) is still the best crisis book (2008-09 +31.4%, 2022
   +1.3%) and still loses 2016-20 (−4.4%/yr alpha): round 1 overstated the cost of shorting,
   not the verdict.
4. For a cross-strategy book: U0v2 (corr 0.80 with SPY, max DD −12.9%) or I2v2 (corr 0.47 with
   SPY in 2022-26, max DD −10.6%, VALID alpha +1.6% from one regime) as a crisis sleeve, and
   I1v2 (corr 0.25 with SPY in 2022-26) as a 2008/2022-type hedge. None adds return in a bull
   market.

## Round 2 repair (2026-10-23, after the independent verifier)

**What the verifier checked** (its scripts and logs: `results2/etf_trend/verify/`). It re-ran
U0v2 (TRAIN, VALID), I1v2 and I2v2 through `research.py`: every saved series bit-identical. It
confirmed the kept-weight fix on the real periods (no hair re-trims, gross ≤ 0.9825), re-ran a
truncation and poison look-ahead test on 13 configs (largest difference 0.0), traced every
round-2 number to saved files and checked the protocol (§R2.0 written before the first
round-2 backtest, one LOCKED read, TL untouched). Tests passed. It found one major claim problem
and one minor wording slip. Neither changes a number, gate, decision or status.

**1. Major (claim problem): what lowered the LOCKED alpha.** §R2.5 said the drop from +0.56%
(round 1) to +0.11% was "mostly the benchmark" (lab v1's SPY dropped each half-year's first
day). The round-2 summary's learnings said the benchmark construction "inflated LOCKED alpha
by ~0.45 pts". Both are wrong. The check is a 2×2 on the SAVED U0v2 rows: each period's day 0
kept or zeroed, for the book and for SPY separately. No backtest was run and LOCKED was not
re-run. `day0_repair.py` / `day0_repair.log` repeats the verifier's `verify/day0_attrib.log` to
the second decimal and splits each day 0 into its costs (base minus 2× costs) and its move.
Alpha vs SPY, %/yr:

| window | v2 | SPY's day 0 zeroed | book's day 0 zeroed | both zeroed | round 1 | effect of the book's day 0 | effect of SPY's day 0 |
|---|---|---|---|---|---|---|---|
| TRAIN | +5.25 | +5.27 | +5.58 | +5.59 | +5.63 | −0.33 | −0.02 |
| VALID | +0.02 | +0.01 | +0.37 | +0.34 | +0.34 | −0.35 | +0.01 |
| LOCKED | +0.11 | +0.04 | +0.61 | +0.57 | +0.56 | −0.50 | +0.07 |

Each day 0, as a mean per half-year:

| window | book | of it costs | of it first-day move | SPY | of it buy cost | of it open-to-close |
|---|---|---|---|---|---|---|
| TRAIN | −0.151% | −0.118% | −0.033% | +0.047% | −0.120% | +0.167% |
| VALID | −0.158% | −0.118% | −0.040% | −0.019% | −0.120% | +0.101% |
| LOCKED | −0.264% | −0.051% | −0.213% | −0.114% | −0.120% | +0.006% |

- **The book's own day 0 accounts for all of the drop and more** (−0.50 pts against a drop of
  −0.45; SPY's day 0 gives back +0.07). Two of LOCKED's first days carry most of it. On 2000-01-03 the book was 98% invested
  and fell −1.98% with SPY (−2.01%). On 2001-01-02 it was 48% invested and fell −1.12% (SPY
  −2.53%). The book's day-0 losses count in full in its CAGR. SPY's count only through beta
  0.27.
- **SPY's day 0 lowered SPY's LOCKED CAGR** (−6.12% → −6.37%): the −0.12% buy cost each
  half-year outweighed first-day moves that averaged +0.01%. 2003-01-02's +2.50% did lift SPY's
  2003 (+23.0% → +27.5%), but 2000, 2001 and 2002 each lost 0.8-1.6 pts. A lower benchmark CAGR
  at a positive beta RAISES the alpha, by +0.07 pts here.
- **Fix:**
  - §R2.5's sentence is replaced. §R2.8 item 2 ("the book and SPY alike") and §R2.2's
    "−0.32 to −0.38" are made exact. All three are marked "round 2 repair". (The −0.38 was round
    1 → round 2 in TRAIN, the strategy fix's −0.04 included; lab v2's day 0 alone is −0.34 /
    −0.32.)
  - The README's "What round 2 changed" and its status box now say it.
  - In the summary, the learning is rewritten and the "benchmark construction inflated LOCKED
    alpha" item is dropped.
- **Effect:** none on any number, gate, decision or status.

**2. Minor (wording in the summary).** "Each half-year's rebuild costs about 0.24 pts" is now:
each rebuild costs ~0.12% of the book (0.118% in TRAIN / VALID, measured), so the two rebuilds
a year cost ~0.24 pts of CAGR. In LOCKED a rebuild costs 0.05% (the book was 43% invested on
its first days and holds no SHY), ~0.10 pts a year. §R2.2 already said so for TRAIN / VALID.
Side note: the LOCKED line of `extras2.log` repeats the TRAIN / VALID estimate ("costs a
rebuild ~0.118%"). LOCKED's measured cost is 0.051%. No document quoted the LOCKED figure.

**What ran.** Only `day0_repair.py`, which reads saved results (`U0v2.pkl`, round 1's
`U0.pkl`) and SPY's bars from the ETF panel. There was no backtest and no new LOCKED run:
`locked_reads.json` still holds one entry (U0v2, 09:05:26). TL was not touched. Tests:
`tests/test_etf_trend.py` 22 passed, `tests/test_strategies.py -k etf_trend` 1 passed, and
`ruff check` is clean. **Unchanged:** status "no edge", final U0v2, 1 of 6 gates, and every
number in §R2.2-§R2.6.

## Round 2 repair, pass 2 (2026-10-23, after the second verifier check)

**What the second check covered** (scripts and logs: `results2/etf_trend/verify/check2/`).
- **Re-runs.** It re-ran these bit for bit: U0v2 (TRAIN, VALID; 0.0 difference in all 46
  periods), I1v2, I4v2, one drop-one-ETF and one offset noise copy, and the first 3 books of
  each luck cache. T1v2's TRAIN alphas equal the tuner's best trial.
- **The kept-weight fix on the real periods.** U0v2 has 189 held re-sizes and I1v2 247, every
  one beyond the band and ≥ `min_change`; the max gross is 0.9811 / 0.9824. Rebuilt from the
  same `compose()` output, the round-1 rule reproduces the review's hair counts (U0 80 + 20, I1
  1,074 + 893), so the check has power.
- **I1v2's borrow.** Charged equals owed: 0.120 / 0.121 %/yr in TRAIN and 0.165 / 0.164 in
  VALID.
- **Look-ahead.** Truncation and poison tests on 3 panels × 15 configs × 3 cuts gave a largest
  difference of 0.0. The poison moved later weights in 134 of 135 checks.
- **Tests and data.** The tests pass, and the DB side effect matches the backup row for row.

It found one major claim problem and four minor issues. None changes a number, gate, decision
or status.

**1. Major (claim problem, in the relayed summary only).** The summary handed to the check was
round 2's pre-repair one. Its learnings said lab v1's SPY "inflated LOCKED alpha by ~0.45 pts"
and that "each half-year's rebuild costs about 0.24 pts". The docs already had both right
(repair pass 1, above), and so did the repair pass's own summary.
- Fix: the summary returned now is the repair pass's summary, updated. Its day-0 learning
  reads: the book's own day 0 is −0.33 / −0.35 / −0.50 pts (TRAIN / VALID / LOCKED), SPY's
  −0.02 / +0.01 / +0.07. One rebuild costs ~0.12% of the book, so the two a year cost ~0.24
  pts of CAGR (0.05% a rebuild in LOCKED).
- No doc change was needed. Effect: none.

**2. Minor (an overgeneralized lab-v2 effect).** The README said lab v2 lowered "the other
long/flat books'" alphas by 0.25-0.4 pts, and the pre-repair headline said 0.24-0.45. The
SPY/SHY switch (I5) is the exception: its VALID alpha rose.
- New `repair2_lab_effect.py` / `.log` (saved results only) splits every book's round 1 →
  round 2 change into the strategy fix (round 1 → the interim runs `*r`, lab v1) and lab v2
  plus `min_change` (interim → v2). It also counts the book's and SPY's day 0 one at a time
  for every v2 book (TRAIN and VALID).
- The lab-v2 step is −0.23 to −0.40 pts per test (TRAIN / VA / VB / VALID) for U0, T1, I2,
  I3, I4 and I6. In TRAIN and VALID it is all day 0, to within 0.01. The book's own day 0 is
  −0.26 to −0.41 pts; SPY's is −0.02 to +0.02. The strategy fix moved these books by −0.07 to
  +0.02.
- **I5** moved by −0.26 (TRAIN), +0.09 (VA), +0.05 (VB) and +0.10 (VALID), all of it in the
  lab-v2 step (strategy fix 0.00); in TRAIN and VALID all of it is day 0. In VALID it held SPY on 20 of its 22 first days, where its day 0 matched
  SPY's (+0.053% vs +0.054%). It sat in SHY on the other two (2019-07-01, 2023-01-03), when
  SPY's day 0 lost 0.46% and 1.04% and the book lost 0.18-0.19%. On those two days SPY's loss
  lowered the benchmark's CAGR, which at I5's VALID beta of 0.79 raises the alpha by more than
  the book's own small loss lowered its CAGR.
- I1 (long/short): strategy fix +0.42 to +0.76, lab-v2 step −0.29 to −0.46. Its day 0 (book
  and SPY together) costs −0.52 / −0.43; the rest of the lab-v2 step (call order, borrow
  deferral, `min_change`) gives back +0.09 / +0.04.
- Fix: the README's "What round 2 changed" and the summary's headline now say "0.24-0.38 in
  TRAIN / VALID, except the SPY/SHY switch (I5), −0.26 in TRAIN but +0.10 in VALID", with the
  reason. §R2.8 item 2's rounding slips are corrected (U0's VALID change is −0.31, not −0.32;
  I5's TRAIN change −0.26, not −0.25), marked "round 2 repair, pass 2". Effect: none on any
  number, gate or decision.

**3. Minor (three precision slips in §R2.0-§R2.2).** Each is corrected in place, marked "round 2
repair, pass 2". The pre-registration's own words in §R2.0 are kept, with the correction
beside them.
- The stitched record is conservative by ~1.1-1.4 pts under late fills (VALID 1.08, TRAIN
  1.43), not "~1.1".
- The orders that went from round 1 to round 2 were about 7 a year in TRAIN and 2-3 in VALID
  (81.0 → 74.2, 94.7 → 92.4), not "the ~7 a year". The round-1 rule's hair re-trims number 80 in
  TRAIN and 20 in VALID.
- `min_change` removes 1-3 SHY re-sizes and 0-1 risky re-sizes per decade in U0 (TRAIN 1 SHY
  order; VALID 3 SHY and 1 risky), and 21-25 SHY slivers per decade in I1. It had said 1-2 and
  0-2, and ~25. Re-running `verify/check2/minchange3.py` gives the same counts.
- Effect: none.

**4. Minor (provenance of `extras2.log`).** Three inline snippets (owner transcript 07:06:21Z,
07:09:50Z, 07:12:19Z) had appended its last 9 lines: the top contributors, the max DD / vol
lines and the day-0 lines. Their code was not saved.
- Fix: the snippets were recovered verbatim from the transcript and saved into `extras2.py`,
  minus two unused lines (and an unused import). A re-run (`extras2_rerun.log`) regenerates all
  24 lines of `extras2.log` bit for bit (an empty `diff`).
- The script now also says that the day-0 lines' "costs a rebuild ~0.118%" is a hard-coded
  estimate (0.98 × 0.12%), not a measurement. `day0_repair.py` measures it: 0.118% in TRAIN /
  VALID and 0.051% in LOCKED.
- Effect: none. Every quoted value (SHY's VALID carry +0.52 pts/yr, the max DDs, the vols and
  the day-0 means) now comes from saved code.

**5. Minor (the engine half of round 1's critical fix was not filed).** `backtest._borrow_fees`
(src/trading_simple/backtest.py lines 662-710, unchanged) still charges a short's buy-back a fee
fraction `accrued × |w| / (x − w)`, which explodes on a hair-sized re-trim.
- Fix: written up as a proposal-only engine item in the new `IDEAS.md`. It gives the problem
  and two concrete fixes: defer a small re-trim's charge to the next real buy-back, as
  `lab.borrow_fees` does, or charge the accrued borrow in dollars on the actual buy-back. It
  also gives the test that settles it. The summary flags it for the orchestrator to put in the
  repo's engine-proposal ledger. `src/` is untouched.
- Effect: none on this book. Its own fix leaves no hairs, and its borrow charged equals the
  borrow owed.

`IDEAS.md` (new; the repo's per-strategy convention) also parks two open questions (a crisis
sleeve beside an equity book; the TRAIN turn-of-month effect) and the adj_close/close splice
check from the README's "Data" section, each with what would settle it.

**What ran.** No backtest and no LOCKED run: `locked_reads.json` still holds one entry (U0v2,
09:05:26). TL was not touched. Three scripts ran:
- `extras2.py`, on saved results and a read of the T1v2 journal;
- `repair2_lab_effect.py`, on saved results only;
- the verifier's `minchange3.py`, signals on the cached ETF panel with no returns read.

Tests: `tests/test_etf_trend.py` 22 passed, `tests/test_strategies.py -k etf_trend` 1 passed,
and `ruff check strategies/etf_trend tests/test_etf_trend.py` is clean. **Unchanged:** status
"no edge", final U0v2, 1 of 6 gates, and every number in §R2.2-§R2.6.
