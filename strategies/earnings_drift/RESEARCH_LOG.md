# earnings_drift — research log

Proposal 6 of `STRATEGY_OVERVIEW.md` §6 ("also considered": earnings-reaction drift, long/short),
evaluated with the shared lab (`strategies/proposals_lab/`, protocol in its README). Everything is
written here in the order it happened, so the work can be resumed and audited.

## §0 Pre-registration (2026-10-22, written before any result was computed)

Only data properties were looked at before this section was written: event counts (~1,450-1,850
usable earnings events a year on 500 names, ~80-90% of name-quarters; earn_surprise has extreme
outliers, −825 .. +340, because of near-zero estimates; every `earn_flag` sits exactly two sessions
after its `earn_day`), and two facts about the crowding data (below). No return, IC or backtest.

**Hypothesis.** After an earnings report, stocks with the most positive news keep outperforming
stocks with the most negative news for weeks (post-earnings-announcement drift, Ball & Brown 1968,
Bernard & Thomas 1989). The market's own reaction is a cleaner, broader measure of the news than
the EPS surprise alone (earnings-announcement return, Brandt, Kishore, Santa-Clara & Venkatachalam
2008; Chan, Jegadeesh & Lakonishok 1996), and combining both beats either. A dollar-neutral book of
the extreme events earns stock-selection alpha whatever the market does.

**Mechanism.** Under-reaction: limited investor attention (Hirshleifer, Lim & Teoh 2009;
DellaVigna & Pollet 2009) and anchoring on the old earnings level make prices absorb earnings news
slowly; arbitrage is limited by idiosyncratic risk and costs, more so in less liquid names.

**Literature default = U0 (nothing tuned).**
1. An event is every `earn_flag` cell: decided at the close of its usable bar `u` (the second
   session after the report session `p = u − 2`), filled at the next open.
2. SUE = (EPS actual − EPS estimate) / close(p − 1) — the price-scaled surprise (Livnat &
   Mendenhall 2006; the repo's classic_signals measure); chosen over yfinance's % surprise because
   that one explodes for near-zero estimates (a rank makes winsorizing moot).
3. EAR = the stock's return from close(p − 1) to close(p + 1) minus the equal-weight market's (the
   panel's names) over the same window — known at u's close.
4. Each component is ranked (percentile, average ties) among the events whose usable bar lies in
   the trailing 252 bars (same bar included); the event's score = the mean of its available
   component ranks; the score is ranked again among the trailing 252 bars' scores (≥ 100 scored
   events needed). Thresholds are therefore point in time.
5. Long if the score's rank ≥ 0.8 (top fifth), short if ≤ 0.2; close(u) ≥ $5.
6. 30 long + 30 short slots, each 0.98 / 60 of the book (dollar-neutral at full slots, gross ≤
   0.98). Free slots are filled at u's close, the most extreme ranks first; an event that finds no
   free slot is dropped (it never waits).
7. Hold 60 sessions (exit decided at the close of u + 60, filled at the next open). A new scored
   event for a held name re-decides it: the same side restarts the clock, otherwise it is closed
   (and may enter the other side on that bar).
8. Borrow: 0.5% / 3% / 10% a year (informed_flow_ls_p1's rates), tiered by **days-to-cover**
   (> 7 days: 3%, > 10 days: 10%), 0.5% where there is no short-interest data (all of E0); never
   short above 15 days to cover (hard to borrow; a held short reaching it is bought back). Longs
   are trimmed back to their slot at 2× it (`retrim_k` 2, as p1).

**Why not p1's crowding measure (short interest ÷ shares outstanding).** p1 computes it as
days-to-cover × the panel's 20-day median volume ÷ SEC's share count. The lab's volumes are split-
adjusted BACKWARD while `fund_shares` is as reported, so before a forward split the measure is
inflated by the split ratio: NVDA reads 97% shorted in 2018 (true ~1-2%), AVGO 12-20% in 2018-23
before its 2024 10:1 split, TSLA 250% in 2018-19. That is knowledge of future splits — future
winners — kept out of the short book (`informed_flow_ls` v8 fixed it with FINRA's own shares-short
count, which the lab panels do not carry). Days-to-cover is FINRA's own as-of ratio, split-free.
Thresholds 7 / 10 / 15 days ≈ the top 7% / 3% / 0.5% of TRAIN-era name-days (TB, T3 panels: 90th
percentile ~6.3 days, 99th ~13). The rates are unchanged.

**Data caveats, stated with every result.** yfinance coverage ~80-90% of name-quarters; the data
END 2026-06-11, so T1's and T2's last period (2026-07 → 10-16) receives no new events (only
positions opened in May-June run into it, then the book sits in cash); the DB holds survivors only
(shorts look worse, longs better than reality); before 2020 "top 500" is nearly every name.

**Expected result.** PEAD has largely disappeared in large caps since ~2006 (Martineau 2022), so:
T1 (top 500) alpha ≈ 0 (−2 .. +2%/yr), T2 (names 501-1000) +1 .. +4%/yr, the TRAIN era 0 .. +3%.
Most likely outcome: no edge on the top 500 (G2/G4 fail), a weak positive on T2. Prior chance
that U0 passes all gates: ~10%. The repo's evidence (classic_signals: surprise/price top fifth,
R+1 → 60 sessions, +7.7%/yr discovery in illiquid + mid names, failed Jun-Oct 2026) points to the
same size-dependence.

**Premise check (TRAIN only, informational).** Events with usable bar in E0, TB, T3 (each
period's own point-in-time top 500), forward return from the open of u + 1 to the open of u + 61
minus trailing-126-bar beta × the equal-weight market's return over the same window, all prices
≤ 2021-12-31 (events whose window would cross it are dropped). Read: rank IC of the score (and of
each component) per calendar quarter with its t across quarters; mean beta-adjusted return of the
top-fifth minus the bottom-fifth events.

**Budget.** At most 8 variants through the adoption bar (the tuned config + iterations; the task's
cap is 10), counted below. Candidate iteration designs, each to be pre-registered with its
mechanism here before it runs and only if what the earlier steps show motivates it: EAR alone vs
SUE alone; a shorter hold (20/40); abnormal-volume confirmation on the reaction days; the long
side + SPY hedge; slots sized by event strength; a liquidity floor; short interest to pick the
shorts (crowded names with bad news).

**Adoption rule (lab README).** Tuned values only if they beat U0 on T1 AND T2 (paired).
An iteration only if its paired Δ ≥ 0 on TRAIN, T1 and T2 and its T1 Δ > 2 × U0's noise sd.
T4 is read once, for the declared final candidate against U0. TL is never read.

## §1 Implementation (2026-10-22)

`strategy.py` = §0 exactly (tests: `tests/test_earnings_drift.py`, 12 synthetic-panel tests incl.
causality on truncated panels; a 13th was added with I2's `crowd` knob). Mechanics on real panels, no returns read: ~19.5% of scored
events qualify long and ~19.7% short; on TB 2019-H1 the book holds 28.4 longs / 28.8 shorts on
average (13-30), target gross ≤ 0.98, net ≈ 0; ~225-230 entries a year; signals() 0.2-0.8 s a panel.
*[Corrected 2026-10-23, review #4: 0.98 is the gross at ENTRY. Held names are never re-trimmed,
so realized gross drifts above it: up to 1.29 on T1, above 1.0 on ~23% of bars (Review fixes R6).
These mechanics numbers came from a transcript-only script. `diagnostics.py mechanics` now
recomputes them (`diag-mechanics-U0f.log`, review #6).]*
T1 2026-H2 holds 6 + 6 names on average and is empty from September (no events after June).
Lab CLI note: `lab.py run <key> U0 '{}' --costs2` reads `--costs2` as the test list (a[4]);
the tests must be given explicitly (`... '{}' E0,TB,T3,T1,T2 --costs2`).

## §2 Premise check (TRAIN only; `premise.py`, log `var/proposals_lab/results/earnings_drift/premise.log`)

9,334 scored events with usable bar in E0/TB/T3 (2016-07 → 2021-12), forward open-to-open returns
minus trailing beta × the equal-weight market, prices ≤ 2021-12-31; 22 calendar quarters.

| horizon | IC score (t) | IC SUE | IC EAR | top − bottom fifth (t, quarters > 0) |
|---|---|---|---|---|
| 20 sessions | +0.019 (1.4) | +0.009 (0.6) | +0.020 (1.4) | +0.27% (0.8, 15/22) |
| 40 sessions | +0.001 (0.1) | −0.003 | +0.004 | −0.03% (−0.1, 12/22) |
| 60 sessions | +0.019 (0.6) | +0.032 (0.7) | −0.002 | −0.36% (−0.6, 10/21) |

By dollar-volume half (within the top 500): the lower half's EAR IC is positive at every horizon
(+0.022 / +0.022 / +0.031, t 1.1-1.3), the upper half's turns negative at 40-60 (−0.011 / −0.018);
fifth spread at 60: lower +0.06%, upper −0.94%. By test (fwd60 spread): E0 +0.36%, TB −1.36%, T3
+0.20%. **Read: no drift worth trading on the top 500 in 2016-21** — what there is sits in the first
20 sessions and in the less-traded half, the literature's (Martineau 2022) and the repo's
(classic_signals: illiquid + mid names) pattern. Informational; U0 runs as pre-registered.

## §3 U0 — the untuned default (`lab.py run earnings_drift U0 '{}' E0,TB,T3,T1,T2 --costs2`)

T1 and T2's last period (2026-07 → 10-16) gets no new events (data end 2026-06-11): it trades only
the positions opened in May-June, then sits in cash.

| test | alpha | P(α>0) | beta | CAGR | B&H | vol | maxDD | late α | 2× cost α | top-5-out α |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | −0.6% | 0.39 | 0.05 | +0.4% | +18.4% | 2.9% | −3.2% | −4.4% | −1.5% | −1.9% |
| TB | −2.3% | 0.21 | −0.01 | −2.4% | +10.5% | 3.7% | −10.2% | −5.9% | −3.1% | −3.6% |
| T3 | −6.5% | 0.04 | 0.06 | −4.3% | +37.3% | 5.0% | −8.4% | −10.3% | −7.4% | −7.9% |
| **T1** | **+0.3%** | 0.56 | 0.02 | +0.5% | +8.6% | 5.2% | −11.3% | −3.3% | −0.5% | −0.9% |
| **T2** | **−1.0%** | 0.31 | −0.02 | −1.2% | +7.7% | 5.0% | −10.4% | −4.7% | −1.9% | −2.5% |

TRAIN era stitched: alpha −2.1% (P 0.08). T1 periods: −2.7, +2.0, +1.2, **−12.6 (2023-H2)**, +8.4,
+3.2, +6.6, +1.1, −1.9, +4.6. T2 periods: −4.4, +5.0, +2.4, −6.7, −5.8, +2.4, −1.3, +6.1, −3.1, −3.8.
~570-580 orders a year (60 slots turning ~4.8× a year): base costs ≈ 1.1%/yr, +50 bp a leg
≈ 3.6 pts/yr. *[Corrected 2026-10-23, review #3: the measured 1× cost drag (alpha − 2×-cost alpha)
is 0.83-0.94 pts/yr and the late-fill penalty 3.56-3.85 pts/yr (`diag-book.log`). About 125 of
the ~575 orders a year are first-bar re-buys of the carried-in book, and their cost never reaches
the returns (the lab drops each period's first-bar return), so ~450 orders a year are charged.
Entries run ~225-240 a year, i.e. ~3.8 per slot, not 4.8.]* Exposure 0.43-0.49 long / 0.43-0.48 short. Bear episodes (book vs equal-weight
market): Q4 2018 −0.8% vs −18.2%, COVID −1.9% vs −37.0%, 2022 bear −0.1% vs −25.2%, Feb-Apr 2025
−3.9% vs −19.6%; the market's 41 down months: book −0.13% (46% > 0) vs market −3.97%.
**Read: no edge, as the premise and the literature predicted.** Market-neutral as designed (|beta|
≤ 0.06), so it holds up in sell-offs, but it earns nothing for it.

**Noise** (`lab.py noise earnings_drift U0`: 5 names dropped a period, 3 copies), sd of the
copies' alpha − U0's: E0 0.21, TB 0.36, T3 0.54, **T1 0.35**, T2 0.38 pts. An iteration's T1 Δ
must exceed **+0.69 pts**.

**Luck** (`lab.py luck earnings_drift U0`, 100 shuffled-name books on T1): real +0.33%, books'
median −0.91%, 95th percentile +2.04% → **70th percentile** (G4 needs ≥ 95).

**Leg attribution** (`attribution.py '{}'`, gross daily held weight × close-to-close return,
simple yearly sums; the legs carry ~0.43-0.49 of the book each): E0 long +10.2% / short −8.6%
(market +18.4%), TB +2.5 / −4.4 (+10.5), T3 +19.2 / −22.2 (+37.3), T1 +5.8 / −4.3 (+8.6), T2 +3.2
/ −2.9 (+7.7). Against each leg's market share (exposure × market) the long leg is +1.5, −2.4,
+0.9, +2.0, −0.2 pts and the short leg ~0, +0.5, −4.3, −0.6, +0.5 pts — no leg carries a
reliable edge. The short leg's big losses are the junk rallies: T3 2020-H2/2021 and T1 2023-H2
(short leg −22.6%/yr in that half, the book's −12.6% period).

## §4 Optimization T0 (pre-registered before it ran)

`lab.tune` on TRAIN only (E0, TB, T3: 11 periods), 120 TPE trials, robust objective (mean − ½ sd
of the period alphas), PBO recorded. Space = every `search:` in config.yaml: surprise {price,
pct}, w_ear 0-1 (0.25), ear_end 0-2, trail 126-504 (126), q 0.10-0.40 (0.05), hold 10-120 (10),
slots 10-50 (10). Adopted only if it beats U0 on T1 AND T2 (paired, `lab.py report`).

**T0 tuning result** (journal `var/proposals_lab/studies/earnings_drift-T0.log`, 120 trials):
best = surprise **pct**, w_ear **0** (the surprise alone), ear_end 2 (moot at w_ear 0), trail
**126**, q **0.40**, hold **80**, slots **20**: robust score +0.040, TRAIN mean period alpha
+6.7%/yr (E0 +3.6, TB +8.7, T3 +8.1) against U0's −2.9% mean (robust −0.046 *[round 2,
review #7: −0.045; `lab.robust` on U0's 11 TRAIN period alphas gives −0.0447]*). **PBO 0.53**: the
in-sample best lands in the out-of-sample bottom half in half of the period splits — overfit by
the lab's own measure. TPE converged fast (111 of 120 trials on `pct`; many exact duplicates).
Marginal means across trials: `pct` +3.0% vs `price` −0.6%; w_ear 0 +3.1%, 0.25 +2.9%, 1.0 (EAR
alone) −3.3%; q 0.40 +3.7% vs 0.20 −0.9%; hold 80-120 +2-5% vs 10-30 −1-2%; slots 10-20 +3.2% vs
40-50 −0.4..−0.9%. So in-sample the % surprise (not the reaction) carried what there was, at broad
thresholds and long holds. Variant 1 = T0 on OPEN. *[Corrected 2026-10-23, review #1: these
marginals were driven by coarse, split-adjusted `pct` surprises (Review fixes R0, R2). On the
fixed code the same parameters keep 1.3% of T0's 8.1% TRAIN alpha, and the re-tuned study prefers
`price` (`pct` −1.4% vs `price` +0.8%). The marginals are reproduced by `diagnostics.py trials`
(`diag-trials-T0.log`).]*

**T0 on OPEN** (`lab.py run earnings_drift T0 '<best>' E0,TB,T3,T1,T2 --costs2`; `report U0 T0`):

| | E0 | TB | T3 | **T1** | **T2** |
|---|---|---|---|---|---|
| T0 alpha | +2.8% | +10.2% | +7.0% | **+0.6%** (P 0.57) | **−4.7%** (P 0.03) |
| Δ vs U0 (P(Δ>0), periods better) | +3.4 (0.81, 4/4) | +12.5 (1.00, 4/4) | +13.5 (0.98, 3/3) | **+0.3 (0.52, 3/10)** | **−3.7 (0.10, 3/10)** |

T0's T1 beta 0.10, T2 maxDD −20.4%, ~200 orders a year. **Not adopted**: the in-sample gain is all
TRAIN; it does not beat U0 on T2 (and barely moves T1). PBO 0.53 said so in advance.

## §5 Event-time drift and conditioning screens (TRAIN only, informational)

`event_time.py` (log `event_time.log`), 3,703 top/bottom-fifth TRAIN events, market-adjusted:
the next report comes a median **62 sessions** after the usable bar (10-90%: 56-71). Top − bottom
spread from the open after the usable bar: **+0.26% at 5 sessions (t 1.8), +0.45% at 10 (t 1.8),
+0.54% at 20 (t 1.7, 16/22 quarters)**, then +0.35% at 40, +0.24% at 60, ~+0.07% at 65-70. The
next announcement's own reaction: **−0.30% (t −1.4, 7/21)** — the opposite of Bernard & Thomas
(1990): the top fifth's next report disappoints relative to the bottom's. So holding across the
next report would hurt (not pursued), and what drift there is ends within ~20 sessions.

`conditions.py` (log `conditions.log`), the same spreads split (no split reaches |t| 2):
abnormal announcement volume high +0.32% at 20 (t 0.9) vs low +0.13% (t 0.3); dollar volume
upper +0.33% (t 0.7) vs lower +0.20% (t 0.6) at 20, −0.94% vs +0.06% at 60; SUE/EAR agreeing
+0.21% (t 0.7; the 4% disagreeing events are too few); **days-to-cover above the bar's median
(2018+): +0.44% at 20 (t 1.1), +0.97% at 40 (t 1.0); below it −0.15% (t −0.2), −1.49% at 40 (t
−1.6)** — in uncrowded names both sides go the wrong way after 20 sessions.

## §6 Iterations (each pre-registered here before it ran; base = U0)

**I1 — hold 20 sessions** (`{"hold": 20}`; variant 2). *Mechanism:* in today's large caps the
under-reaction is arbitraged within weeks (Chordia, Goyal, Sadka, Sadka & Shivakumar 2009;
Martineau 2022), so the hold should end when the drift does (§5: the spread peaks at 20
sessions); sessions 20-60 add only noise and, for uncrowded names, reversal. Side effect: capacity
triples (30 slots × 252/20 ≈ 380 entries a year per side ≥ ~350 qualifying events), so nearly
every qualifying event gets a slot instead of the first third of each season. Cost: turnover ×3
(~3%/yr at base costs, ~10 pts at +50 bp). *Expected:* T1 Δ −1 .. +1 pt; likely fails G5 (late
fills).

**I2 — crowded names only** (`{"crowd": "crowded"}`; variant 3). An event qualifies (either side)
only if the name's days-to-cover at the decision bar is above the median of the panel's names
with data that bar; no short-interest data (all of E0) = no evidence, qualifies. *Mechanism:*
earnings news in a heavily shorted name meets informed, positioned traders: bad news confirms the
short sellers' thesis and they press (Boehmer, Jones & Zhang 2008; Engelberg, Reed & Ringgenberg
2012: short sellers trade on public news), good news forces covering that takes weeks; in an
uncrowded name nobody is positioned to keep trading the news (§5: the spread there reverses
after 20 sessions). Both sides are crowded, so the book takes no net side on the short-interest
factor itself. *Expected:* TRAIN Δ ≥ 0 (E0 identical), T1/T2 Δ 0 .. +2 pts; weak prior (t ~1).

**I1 and I2 results** (`lab.py report earnings_drift U0 I1 I2`; Δ = paired alpha vs U0, P(Δ>0),
periods better; T1/T2's last period has no new events):

| | E0 Δ | TB Δ | T3 Δ | **T1** alpha / Δ | **T2** alpha / Δ | T1 late α | orders/yr |
|---|---|---|---|---|---|---|---|
| I1 hold 20 | −1.3 (0.25) | −0.4 (0.39) | +6.6 (1.00) | −0.8% / **−1.1** (0.27, 3/10) | −2.0% / **−1.0** (0.31, 5/10) | −6.8% | ~800 |
| I2 crowded | −0.3 (0.34) | −0.2 (0.46) | +5.2 (0.91) | +0.6% / **+0.3** (0.55, 4/10) | +0.3% / **+1.4** (0.72, 6/10) | −3.0% | ~560 |

**I1 rejected** (Δ < 0 on T1, T2, E0, TB): the front-loaded TRAIN drift does not pay its tripled
turnover. *[Corrected 2026-10-23, review #2: "turnover ×3" was the pre-registered prediction, not
a result. Measured (`diag-book.log`): orders rise ×1.40 on T1 (795 vs 568 a year) and ×1.28 on T2.
The 1× cost drag is 1.47 vs 0.87 pts/yr and the late-fill penalty 6.0 vs 3.6. The bigger change
is that a 20-session hold leaves the slots mostly empty between earnings seasons: mean gross 0.48
vs 0.86 on T1 (0.44 vs 0.86 on T2). It loses even per unit of gross: T1 −1.7% vs U0 +0.4%, T2
−4.6% vs −1.2%.]* **I2 rejected**: T1 Δ +0.3 < the +0.69 bar (and E0/TB slightly negative — E0 is not
identical to U0 after all: its 2018-H1 period has short-interest data), though it is
the one change that helped the fresh names (T2 +1.4) and T3 (+5.2) — the crowded-name pattern of
§5 is real enough to note, too weak to adopt.

**Data-timing check** (TRAIN panels; volume and |abnormal return| relative to the name's normal,
by offset from `earn_day`): −1: ×1.22 / ×1.06; **0: ×1.79 / ×2.04; +1: ×1.75 / ×2.17**; +2 (the
usable bar): ×1.32 / ×1.32; +3: ×1.18 / ×1.17. *[Review #6: recomputed by the committed
`diagnostics.py timing` (normal = the median over sessions p−65 .. p−6; medians across 9,319
events; `diag-timing.log`): −1 ×1.21 / ×1.06, 0 ×1.79 / ×2.02, +1 ×1.75 / ×2.17, +2 ×1.32 / ×1.31,
+3 ×1.17 / ×1.16.]* yfinance's report dates are right and the EAR
window [p−1, p+1] holds the reaction; the lab's usable bar (p+2) still carries a little of it,
which the book cannot trade. The absence of drift is not a dating artifact.

**Correlation with the short-interest book** (`informed_flow_ls_p1` v6, its lab result
`v6_si.pkl`, read only): U0 daily / monthly return correlation TB −0.12 / −0.02, T3 +0.36 / +0.05,
**T1 +0.25 / +0.29, T2 +0.12 / +0.32**; I2 T1 +0.17 / +0.36. *[Review #6: reproduced exactly by
`diagnostics.py corr` (`diag-corr.log`). U0f: TB −0.03 / −0.13, T3 +0.20 / −0.06, T1 +0.22 / +0.32,
T2 +0.09 / +0.22.]* Positive — plausibly a shared tilt toward
recent winners (not measured) — so the earnings book is no hedge for it.

**I3 — the reaction alone** (`{"w_ear": 1.0}`; variant 4). *Mechanism:* the proposal's central
claim (Brandt et al. 2008): the market's 2-day reaction aggregates all the news (guidance,
revenue, margins), not just the EPS number, and its drift decayed less than SUE's. Evidence is
mixed: TRAIN IC of EAR +0.020 at 20 sessions (SUE +0.009) but −0.002 at 60 (SUE +0.032), and the
tuner's EAR-alone trials averaged −3.3%. *Expected:* T1 Δ −1 .. +1 pt; likely rejected.

**I3 result** (`report U0 I3`): T1 −1.2% / **Δ −1.5** (P 0.23, 4/10), T2 −1.1% / **Δ −0.1** (P 0.49),
E0 Δ −1.1, TB +1.7, T3 −0.9; late T1 −4.8%. **Rejected.** The reaction alone is worse than the
mean of the two ranks — the Brandt et al. claim does not hold here.

**Stitched TRAIN-era paired Δ vs U0** (E0+TB+T3 as one record; U0 −2.05%): T0 +10.1 (P 1.00,
in-sample), I1 +0.9 (0.75), I2 +1.6 (0.86), I3 −0.4 (0.44).

## §7 Final candidate (declared 2026-10-22, before `verdict`, `bear` and the T4 read)

*[Superseded 2026-10-23: §7-§8 judge U0 on the leaky code. The final candidate is now U0f (Review
fixes R4), with its verdict, bear table and second T4 read in R5. §7-§8 stay as the record.]*

**U0, the pre-registered default** — no variant passed its bar (table below). Iteration stops
here: the one direction with any consistency (I2, crowded names: TRAIN +1.6, T1 +0.3, T2 +1.4)
fails the T1 noise bar, and dialing it further would design from T1/T2 reads. *[Round 2, review
#7: "consistency" overstated it. I2's E0 and TB Δ were negative (−0.3, 1 of 4 periods better;
−0.2, 1 of 4) and its T1 Δ +0.3 was inside the noise; only T3 (+5.2) and T2 (+1.4) helped. On the
fixed code it lost 3.1 pts on T1 (R3).]* T4 (names
1001-1500) is read once, for U0 alone (the final is U0, so "final vs U0" is one book).

## §8 Verdict, bear table, T4 (read once)

`lab.py verdict earnings_drift U0` (luck re-read: 70th percentile, unchanged):

| gate | value | passed |
|---|---|---|
| G1 TRAIN-era alpha > 0 | −2.05% | no |
| G2 T1 alpha > 0, P ≥ 0.90 | +0.33% (P 0.56) | no |
| G3 T2 alpha > 0 | −1.05% | no |
| G4 T1 beats ≥ 95% of shuffled books | 70th percentile | no |
| G5 T1 late-fill and 2×-cost alpha > 0 | −3.26% / −0.5% | no |
| G6 every bear episode beaten and down months > 0 | episodes all beaten; down months −0.13% | no |
| G7 \|T1 beta\| ≤ 0.3 | 0.016 | **yes** |
| G8 T1 alpha without its top 5 names > 0 | −0.89% | no |

`lab.py bear earnings_drift U0 T0 I2` (stitched E0+TB+T3+T1): U0 Q4 2018 −0.8% (equal-weight
market −18.2%, SPY −18.7%), COVID −1.9% (−37.0%, −33.4%), 2022 bear −0.1% (−25.2%, −24.1%),
Feb-Apr 2025 −3.9% (−19.6%, −18.8%); CAGR −0.8%, max drawdown −20.3% (a slow bleed 2018-23).
Calendar years: 2016 +1.9, 2017 +0.6, 2018 −1.3, 2019 −4.1, 2020 −2.1, 2021 −5.4, 2022 −1.4, 2023
−6.7, 2024 +5.8, 2025 +4.8, 2026 +0.4 (equal-weight market −19.4% in 2022). For comparison T0 CAGR
+5.5% (in-sample years), I2 +0.2% (max DD −12.2%).

**T4 (names 1001-1500, $3M floor; read once, U0)**: alpha **−1.6%** (P 0.22), beta −0.01, late
fills −5.2%, 2× costs −2.4%, without its top 5 names −3.1%; periods −11.1, −3.9, +13.4, −5.5,
−0.1, −1.7, −2.3, −4.8, +5.4, +0.8 (the last has no new events). The less-traded names, where the
literature expects the drift to survive, show none either.

**Verdict: no edge.** Post-earnings drift built from yfinance EPS surprises and 2-day reactions
earns nothing on the top 500 (T1 +0.3%), names 501-1000 (−1.0%), names 1001-1500 (−1.6%) or the
2016-21 training era (−2.1%), and it loses 3-5 pts more with late fills. The market-neutral
construction works as designed (beta ≈ 0, < 4% lost in every sell-off), so in falling markets
it is cash with costs, not alpha.

## Variant count

| # | label | what | TRAIN Δ | T1 Δ (bar +0.69) | T2 Δ | adopted |
|---|---|---|---|---|---|---|
| — | U0 | the pre-registered default | — | — | — | base = final |
| 1 | T0 | tuned (pct, w_ear 0, trail 126, q 0.4, hold 80, slots 20; PBO 0.53) | +10.1 | +0.3 | −3.7 | no (fails T2) |
| 2 | I1 | hold 20 | +0.9 | −1.1 | −1.0 | no |
| 3 | I2 | crowded names only | +1.6 | +0.3 | +1.4 | no (T1 < bar) |
| 4 | I3 | the reaction alone (w_ear 1) | −0.4 | −1.5 | −0.1 | no |

Four of the ten allowed variants (eight pre-registered) went through the bar; tuner trials excluded.

## Lab issues found (worked around locally; `lab.py` not edited)

1. **p1's crowding measure leaks future splits** (§0): days-to-cover × the lab's split-adjusted
   volume ÷ SEC's as-reported share count is inflated by every later forward split (NVDA 97% in
   2018, AVGO 12-20% in 2018-23, TSLA 250% in 2018-19). Any lab book that copies
   informed_flow_ls_p1's `si_share` tiers or its "never short above 30%" rule keeps future
   splitters (future winners) out of its shorts. This book uses days-to-cover instead.
2. **CLI:** `lab.py run <key> <label> '<overrides>' --costs2` takes `--costs2` as the test list
   (`a[4]`) and fails; pass the tests explicitly before the flag.
3. **Luck cache keyed on every param:** adding a knob to a config (even one whose default changes
   nothing) changes the key, and `verdict` silently recomputes the 1,000 luck books.
4. **Shared scratchpad:** the six agents share one scratchpad folder, and another agent overwrote
   a helper script of mine with the same generic name mid-session (one summary read returned
   another strategy's numbers; caught and re-run from a uniquely named folder). Use unique names.

## Review fixes (2026-10-23)

An adversarial review (`var/proposals_lab/results/earnings_drift/review/`) reproduced every number
above exactly and found two major and four minor issues. This section records each issue, the fix
and the new numbers. Everything below the "pre-registration" paragraph was computed after it was
written. Old labels (U0, T0, I1, I2, I3, their noise and luck files) are kept as the record of the
leaky code. The new labels end in `f`.

### R0 Pre-registration of the data fix (written before any new backtest or return)

**The issue (major, review #1).** yfinance stores EPS (`earn_actual`, `earn_estimate`) on TODAY's
split basis, rounded to the cent. 100% of the stored values are whole cents (checked on 9,304 TRAIN
events). `earn_surprise` equals (actual − estimate) / |estimate| computed from those rounded values
exactly, so it carries no extra precision. A name that split later therefore has tiny, coarse EPS
before the split: NVDA 2016 actual = estimate = 0.01, CMG 2018 0.04-0.06, KLAC 2018 0.21 (as
reported $2.22: KLAC split 10:1 later). One cent is then a large share of the measure: under `pct`,
1 cent on an estimate of 0.01 is 100%. The *precision* of the surprise thus depends on the name's
future splits, and future splitters are future winners. The data properties on TRAIN (no returns):

- price-scaled SUE: P20 0.000%, P50 0.066%, P80 0.221% of price. The cent quantum 0.01 / close(p−1)
  has median 0.015%, p90 0.048%, p99 0.121%.
- `pct`: P20 0%, P50 5%, P80 18%. The quantum 0.01 / |estimate| has median 1.1%, p90 4.8%, p99 33%.
- Future splitters (the reviewer's proxy: SEC shares ≥ 1.8× higher by 2026) are 13% of the events
  with a known share count. Among events where one cent exceeds a quarter of the P20-P80 range of
  the price SUE (7.9% of events) they are 44%, against 11% elsewhere.

**Why not rebuild the EPS on the as-of basis.** The panels carry no split history. Even with one,
multiplying a rounded value back by the split factor rescales the rounding error instead of
removing it. Both surprise measures are scale-free (price-scaled: both terms on the same basis;
`pct`: a ratio), so their ranks would not change. The coarse values must be taken out, not rescaled.

**The fix (one new knob, `cent_max`, value 0.25, not searched).** An event's surprise is used only
where one cent of EPS is at most `cent_max` × the P20-P80 spread of that surprise measure among
the events of the trailing `trail` bars (the same point-in-time window as the ranks). The quantum is
0.01 / close(p−1) under `price` and 0.01 / |estimate| under `pct`. Elsewhere the surprise is set to
NaN. It leaves the trailing reference distribution too, and the event ranks on its reaction (EAR)
alone, the way U0 already treats a missing estimate. Under `w_ear` 0 (the surprise alone) such an
event is not scored at all. Why 0.25: the rounding error of actual − estimate is at most one cent.
At 0.25, one cent moves an event by at most a quarter of the P20-P80 range, about 15 percentile
points of rank. At 0.5 one cent could carry a median event into a band. The rule flags about 8% of
TRAIN events under `price` and about 10% under `pct`. Not every flagged event is a future splitter:
the rest are genuinely low-priced or high-yield names, whose coarse surprise was real. Dropping
those is conservative. The rule's trigger still depends on the adjusted price, which embeds future
splits and dividends. But a flagged event then ranks on EAR, a split-free ratio of closes, on both
sides alike, so the rounding cannot push future splitters to one side any more.

**Expected.** U0f ≈ U0 (the reviewer found U0's tiny-estimate events lean short, 32 long / 43
short, so U0 was not helped by the leak): every |Δ| < 1 pt, the verdict unchanged. The tuned
config's in-sample gain shrinks. The T0 parameters on the fixed code should lose a large part of
T0's TRAIN +10.1 pts. The reviewer attributes 43% of T0's gross TRAIN P&L to names with |estimate|
< $0.05. A re-tune on the fixed code (T0f, 120 trials, same space) should score lower in-sample, and
its PBO should stay ≥ 0.4.

**Run plan (no new designs: the same variants re-run on the fixed code).**
1. U0f = U0 with `cent_max` 0.25 (the config default) on OPEN with late fills and 2× costs. Then
   `noise` (5 names dropped, 3 copies) and `luck` (100 books, T1).
2. T0q = the old T0 parameters on the fixed code, on OPEN. This is a diagnostic of how much of T0
   was the leak, not a candidate.
3. T0f = `lab.tune` on TRAIN with the fixed code (study `earnings_drift-T0f`, 120 trials, the
   space of §4). Run on OPEN. Adopted only if it beats U0f on T1 AND T2.
4. I1f (hold 20), I2f (crowded) on OPEN against U0f, with the same adoption bar as §6 (Δ ≥ 0 on
   TRAIN, T1, T2 and T1 Δ > 2 × U0f's noise sd). I3 (w_ear 1) does not read the surprise, so its
   signals are identical on the fixed code. Its old result is re-compared against U0f.
5. The final candidate is declared below before `verdict`, `bear` and a SECOND T4 read. That read
   is allowed only because the final book changed (the fixed code), and it is reported as such.
6. Re-run the informational TRAIN screens (premise, event time, conditions) on the fixed code. All
   numbers that lived only in a transcript (correlation with p1, data timing, TPE marginals, §1
   mechanics) are recomputed by committed scripts with logs in the results folder (review #6).

### R1 U0f, the fixed default (`lab.py run earnings_drift U0f '{}' E0,TB,T3,T1,T2 --costs2`)

T1's and T2's last period (2026-07 → 10-16) gets no new events (data end 2026-06-11). The rule
marks as coarse 12-15% of the in-window events in E0, 10-14% in TB, 1-8% in T3, 2.5-6.8% in T1
and 2-5.6% in T2 (`diag-mechanics-U0f.log`). That is more than the ~8% R0 estimated from the
pooled spread, because the trailing 2016-20 spreads are narrower. The rule was not changed.

| test | alpha | P(α>0) | beta | late α | 2× cost α | top-5-out α | maxDD | Δ vs U0 (P, periods better) |
|---|---|---|---|---|---|---|---|---|
| E0 | −0.9% | 0.34 | 0.05 | −4.8% | −1.9% | −2.4% | −4.1% | −0.3 (0.38, 2/4) |
| TB | −3.0% | 0.17 | −0.01 | −6.6% | −3.8% | −5.1% | −11.7% | −0.7 (0.27, 2/4) |
| T3 | −6.3% | 0.04 | 0.03 | −10.0% | −7.2% | −8.2% | −8.8% | +0.2 (0.54, 2/3) |
| **T1** | **+3.1%** | 0.897 | −0.01 | −0.5% | +2.2% | +1.3% | −11.5% | **+2.8** (0.999, 7/10) |
| **T2** | **−0.2%** | 0.48 | −0.03 | −3.8% | −1.1% | −1.9% | −11.4% | +0.8 (0.89, 8/10) |

The stitched TRAIN era is −2.63% (P 0.056), Δ vs U0 −0.6 (P 0.24). T1 periods: +3.7, +2.5,
−2.9, −10.6, +8.5, +2.3, +12.9, +8.7, +5.6, +4.3. T2 periods: −2.5, +6.0, +2.5, −4.8, −7.7, +3.3,
+0.7, +10.8, −2.7, −4.5. Bear episodes (book vs equal-weight market): Q4 2018 −0.7% vs −18.2%,
COVID −2.2% vs −37.0%, 2022 bear **+4.8%** vs −25.2%, Feb-Apr 2025 −2.3% vs −19.6%. The book's
mean in the market's 41 down months is +0.09%.

**Noise** (`U0f-noise.log`), sd of the copies: E0 0.58, TB 1.00, T3 0.37, **T1 0.23**, T2 0.25 pts.
The iteration bar is T1 Δ > +0.46. All three T1 copies landed below U0f (−0.29, −0.72, −0.39).
**Luck** (`U0f-luck.log`): real +3.12%, books' median −0.57%, p95 +2.37%, **96th percentile**.

**Is the T1 move the rule, or just any change to the event set?** The lab's noise check drops 5
names. Re-scoring 5-15% of the events is a much larger perturbation, so the check needs its own
yardstick. `diagnostics.py placebo` re-scores RANDOM events on their reaction alone, as many per
calendar quarter as the rule flags, 20 times. Logs: `diag-placebo.log`, `diag-placebo-train.log`.
Δ is against no rule (= U0, reproduced exactly):

| test | the rule's Δ | 20 placebo Δs: mean ± sd (min, max) | the rule beats |
|---|---|---|---|
| E0 | −0.34 | +0.18 ± 0.63 | 30% |
| TB | −0.69 | −0.35 ± 1.61 | 40% |
| T3 | +0.22 | +0.12 ± 1.49 | 50% |
| **T1** | **+2.79** | +0.87 ± 0.57 (+0.10, +1.88) | **100%** |
| T2 | +0.83 | +0.23 ± 0.54 | 80% |

**Where the T1 gain comes from** (`diag-dropped.log`, by-name attribution). The coarse events
that entered the book on T1 (31 long, 28 short) and T2 (24 / 21) include no future splitter by
the SEC-shares proxy (1 short on T1). On TRAIN they do, on both sides alike: E0 17 of 55 longs
and 21 of 49 shorts, TB 14 of 41 and 10 of 36. Dropping the coarse events instead of scoring them
on EAR gives T1 +1.0%, T2 −0.6%, E0 −0.3%, TB −3.2%, T3 −5.6%. So ~2 of T1's 2.8 pts come from
those ~60 entries of low-priced names traded on their 2-day reaction, and ~0.7 from removing
their cent-level surprises. By name, the T1 gross P&L change is +2.4 pts/yr. Future splitters
(34 names) account for +0.5, the rest for +1.9, and the top 10 names for 28% of the |Δ|.

**Read.** The fix is correct and needed (R2 shows what the leak did to the tuner). Its effect on
U0 was larger than pre-registered on T1 alone. On T1 it is specific to the coarse events (it
beats every placebo), smaller on T2 (80%) and absent on TRAIN (30-50%, within placebo noise).
One test's gain that neither TRAIN nor T2 shows is a weak result. U0f's verdict is read below
without discounting it.

### R2 What the leak did to the tuner: T0q, and the re-tune T0f

**T0q** = the old tuned parameters (pct, w_ear 0, ear_end 2, trail 126, q 0.40, hold 80, slots 20)
on the fixed code. Stitched TRAIN alpha **+1.34%** (P 0.70) against T0's +8.05%: Δ **−6.7 pts**
(P(Δ>0) 0.006). E0 +2.8 → −1.1, TB +10.2 → +4.5, T3 +7.0 → −0.1. So the coarse `pct` surprises
carried ~83% of the tuned book's in-sample alpha. On T1 it rises from +0.6 to +3.0%. T2 goes from
−4.7 to −3.6%.

**T0f** (`lab.tune`, TRAIN only, fixed code, 120 TPE trials, the §4 space, seed 42; journal
`studies/earnings_drift-T0f.log`). The best trial is surprise **price**, w_ear **0.25**, ear_end 1,
trail 126, q **0.30**, hold **50**, slots **10**. Robust score +0.008 against the old study's +0.040,
mean period alpha +4.4% against +6.7%, **PBO 0.42** against 0.53. The marginal means flipped
(`diag-trials-T0f.log` against `diag-trials-T0.log`):

- `pct` −1.4% vs `price` +0.8% (old study: +3.0% vs −0.6%);
- hold 40-50 +0.4 / +2.2%, 80-120 −0.4 .. −4.2% (old: 80 +4.7%);
- w_ear 0.25 +0.9%, 1.0 −1.6%; slots 10 +1.7%.

The old study's "% surprise, broad bands, long holds" was the leak.

T0f on OPEN: TRAIN +3.4% (P 0.91, in-sample), **T1 +3.3%** (P 0.77, max drawdown −23.4%), **T2
−3.0%** (P 0.16). Paired Δ vs U0f: TRAIN +6.0, T1 **+0.2** (P 0.50, 2/10 periods better), T2
**−2.8** (P 0.23). **Not adopted**: it does not beat U0f on T2, and T1 is a tie.

### R3 The iterations re-run on the fixed code (base U0f; bar: Δ ≥ 0 on TRAIN, T1, T2 and T1 Δ > +0.46)

| # | label | change | TRAIN Δ (P) | T1 alpha / Δ (P, better) | T2 alpha / Δ (P) | adopted |
|---|---|---|---|---|---|---|
| 5 | T0f | re-tuned (R2) | +6.0 (0.99) | +3.3% / +0.2 (0.50, 2/10) | −3.0% / −2.8 (0.23) | no (T2) |
| 6 | I1f | hold 20 | +1.5 (0.83) | −0.3% / **−3.5** (0.06, 2/10) | −1.9% / −1.6 (0.23) | no |
| 7 | I2f | crowded names only | +2.8 (0.96) | 0.0% / **−3.1** (0.05, 3/10) | −0.8% / −0.5 (0.40) | no |
| — | I3 | reaction alone (w_ear 1; no surprise, so identical signals, not re-run) | +0.2 (0.55) | −1.2% / **−4.3** (0.02, 1/10) | −1.1% / −0.9 (0.33) | no |

I2's crowded-name hint (vs U0: T1 +0.3, T2 +1.4) does not survive the fix: against U0f it loses
3.1 pts on T1 and 0.5 on T2.

### R4 Final candidate (declared 2026-10-23, before `verdict`, `bear` and the second T4 read)

**U0f** = the pre-registered default U0 with the pre-registered data fix (`cent_max` 0.25). No
variant passed its bar against it (R2, R3). The final book changed because the code was fixed,
not because a variant won. So T4 (names 1001-1500) is read a **second** time, for U0f only. The
first read (U0, leaky code, §8) stays on record. TL is not read.

### R5 Verdict, bear table, T4 (second read)

`lab.py verdict earnings_drift U0f` (`U0f-verdict.log`; luck from its cache, 96th percentile):

| gate | U0 (§8, leaky code) | **U0f** | passed |
|---|---|---|---|
| G1 TRAIN-era alpha > 0 | −2.05% | −2.63% | no |
| G2 T1 alpha > 0, P ≥ 0.90 | +0.33% (P 0.56) | +3.12% (P 0.897) | no |
| G3 T2 alpha > 0 | −1.05% | −0.22% | no |
| G4 T1 beats ≥ 95% of shuffled books | 70th percentile | 96th percentile | **yes** |
| G5 T1 late-fill and 2×-cost alpha > 0 | −3.26% / −0.55% | −0.55% / +2.22% | no |
| G6 every bear episode beaten and down months > 0 | episodes yes; −0.13% | episodes yes; +0.09% | **yes** |
| G7 \|T1 beta\| ≤ 0.3 | 0.016 | −0.009 | **yes** |
| G8 T1 alpha without its top 5 names > 0 | −0.89% | +1.27% | **yes** |

`lab.py bear earnings_drift U0f U0 T0f` (`bear-U0f.log`; stitched E0+TB+T3+T1). U0f: Q4 2018
−0.7% (equal-weight market −18.2%, SPY −18.7%), COVID −2.2% (−37.0%, −33.4%), 2022 bear **+4.8%**
(−25.2%, −24.1%), Feb-Apr 2025 −2.3% (−19.6%, −18.8%). CAGR −0.0%, max drawdown −20.3%. Calendar
years: 2016 +0.4, 2017 −0.4, 2018 +1.4, 2019 −7.0, 2020 −1.9, 2021 −6.4, 2022 +3.0, 2023 −7.6,
2024 +5.4, 2025 +11.5, 2026 +3.3. T0f: CAGR +3.8% (its TRAIN years are in-sample), max DD −23.4%.

**Leg attribution** (`U0f-attribution.log`, gross, against each leg's share of the market): T1
long +3.3 pts, short +0.5. T2 +0.1 / +1.0. TRAIN: E0 +0.5 / +0.8, TB −1.3 / −0.8, T3 +0.3 / −4.9
(the 2020-21 junk rally again).

**T4, names 1001-1500, SECOND read** (U0f only, declared in R4; `U0f-T4.log`). Alpha **−1.93%**
(P 0.18), beta −0.01, late fills −5.5%, 2× costs −2.8%, without its top 5 names −3.6%, max
drawdown −12.7%. Periods: −10.0, −1.9, +16.6, −6.2, −4.4, −2.1, −3.7, −3.9, +3.4, +1.3 (the last
has no new events). The first read (U0) was −1.60%: Δ −0.33 (P 0.29). The fix's T1 gain does not
carry to the less-traded names.

**Verdict: still no edge.** The fixed book passes 4 of 8 gates (U0: 1), all on T1 alone. The
TRAIN era (−2.6%), names 501-1000 (−0.2%) and names 1001-1500 (−1.9%) show nothing, and +50 bp
late fills take T1 to −0.5%. The T1 gain is specific to the coarse events (it beats all 20
placebos) and comes from ~60 entries of low-priced names traded on their reaction (R1). It is not
seen on TRAIN, T2 or T4, so it is not evidence of a drift that can be traded.

### R6 The other review issues

- **#2 (major) I1's mechanism was overstated.** "Turnover ×3" was the pre-registered prediction
  restated as a result. Corrected in §6 and the README with measured numbers (`diag-book.log`):
  orders ×1.40 on T1 (795 vs 568 a year), ×1.28 on T2. The 1× cost drag is 1.47 vs 0.87 pts and
  the late-fill penalty 6.0 vs 3.6. The dominant change is that a 20-session hold leaves the
  slots half empty between seasons: mean gross 0.48 vs 0.86. It loses even per unit of gross (T1
  −1.7% vs +0.4%, T2 −4.6% vs −1.2%). On the fixed code (I1f vs U0f): gross 0.48 vs 0.86, alpha per
  unit of gross T1 −0.7% vs +3.6%, T2 −4.2% vs −0.3%.
- **#3 (minor) cost arithmetic mixed two bases.** Corrected in §3. The measured 1× cost drag
  (alpha − 2×-cost alpha) is 0.83-0.94 pts a year, the +50 bp late-fill penalty 3.56-3.85
  (`diag-book.log`; U0f 0.87-0.92 / 3.62-3.86). `diagnostics.py anatomy U0f` counts ~575 orders a
  year. ~125 of them are first-bar re-buys of the carried-in book, which never reach the returns
  (the lab drops each period's first-bar return, the right accounting for a continuous book), so
  ~450 a year are charged (T1 448, T2 452, E0 469). There are ~212-247 entries a year, 3.5-4.1 per
  slot.
- **#4 (minor) gross is capped only at entry.** Docs corrected (§0/§1, config doc, README): 0.98
  is the entry gross. Realized U0f gross: T1 max 1.29 (above 1.0 on 24% of bars), T2 max 1.20
  (23%), E0 max 1.07 (22%). No code change. The engine's `lock_cash` enforces the cash constraint
  whenever an order is placed. Between rebalances a real margin account drifts the same way
  (Reg-T allows 2× gross). A hard cap would need the realized weights inside `signals()`, which
  only the engine knows (its retrim is longs-only).
- **#5 (minor) engine: new shorts can be rejected on busy days.** Under `lock_cash` with
  `call_seq="auto"`, new short sales can run before the covers that free their collateral. The
  target is never re-emitted, so the name stays unheld for the hold. U0f: T1 6 rejected + 8 filled
  below 90% of 1,615 entries (13 of the 14 shorts), T2 1 + 13 of 1,617 (11 shorts), E0 0 + 1 of
  690. That is small, and `src/` is not edited. **Engine proposal** for
  `backtest._run_portfolio`: when `lock_cash` is on, run each bar's cash-freeing orders (long
  sells, short covers) before the cash-consuming ones (buys, new shorts). An alternative is a
  second pass that re-emits targets the first pass rejected.
- **#6 (minor) numbers that lived only in a transcript.** `diagnostics.py` (commands `precision`,
  `corr`, `timing`, `trials`, `mechanics`, `book`, `anatomy`, `placebo`, `dropped`) recomputes them,
  with logs `diag-*.log` in the results folder. Reproduced: the correlations exactly, the TPE
  marginals exactly, the timing table within ±0.02.

### R7 Variant count (the whole record)

| # | label | base | what | TRAIN Δ | T1 Δ (bar) | T2 Δ | adopted |
|---|---|---|---|---|---|---|---|
| — | U0 | — | pre-registered default (leaky code) | — | — | — | superseded by U0f |
| 1 | T0 | U0 | tuned, old code | +10.1 | +0.3 (0.69) | −3.7 | no |
| 2 | I1 | U0 | hold 20 | +0.9 | −1.1 | −1.0 | no |
| 3 | I2 | U0 | crowded names only | +1.6 | +0.3 | +1.4 | no |
| 4 | I3 | U0 | reaction alone | −0.4 | −1.5 | −0.1 | no |
| — | U0f | — | U0 + the pre-registered data fix | −0.6 vs U0 | +2.8 vs U0 | +0.8 vs U0 | **final** (a fix, not a variant) |
| 5 | T0f | U0f | re-tuned on the fixed code | +6.0 | +0.2 (0.46) | −2.8 | no |
| 6 | I1f | U0f | hold 20 | +1.5 | −3.5 | −1.6 | no |
| 7 | I2f | U0f | crowded names only | +2.8 | −3.1 | −0.5 | no |

Seven of the ten allowed variants went through the bar. I3 was re-compared against U0f (TRAIN
+0.2, T1 −4.3, T2 −0.9) but not re-run: its signals do not read the surprise. T0q (T0's parameters
on the fixed code), the placebos and the "dropped" books are diagnostics, never candidates.

### R8 The TRAIN screens on the fixed code (`premise-U0f.log`, `event_time-U0f.log`, `conditions-U0f.log`)

Premise (9,334 events, beta-adjusted, by quarter):

| horizon | IC score (t) | IC SUE | IC EAR | top − bottom fifth (t, quarters > 0) |
|---|---|---|---|---|
| 20 | +0.016 (1.3) | +0.006 | +0.020 | +0.14% (0.5, 12/22) |
| 40 | +0.000 (0.0) | −0.004 | +0.004 | −0.09% (−0.2, 11/22) |
| 60 | +0.020 (0.6) | +0.028 | −0.002 | −0.39% (−0.7, 10/21) |

Event time: top − bottom +0.32% at 5 sessions (t 2.1), +0.47% at 10 (t 2.0), +0.34% at 20 (t 1.2),
+0.05% at 60. The next report's reaction is −0.32% (t −1.6, 7/21), a median 62 sessions later.
Conditions: crowded names +0.65% at 20 (t 1.9), +1.34% at 40 (t 1.4); uncrowded −0.63% (t −1.1),
**−1.98% at 40 (t −2.2, 4/16 quarters)**. The same picture as §2/§5, a little noisier at 20
sessions. What drift there is fades within 10-20 sessions, and uncrowded names reverse.

### R9 Lab issues found during the fixes (`lab.py` not edited)

5. **The earn_* EPS are on today's split basis, rounded to the cent** (R0). Every lab book that
   reads `earn_surprise`, `earn_actual` or `earn_estimate` gets the precision leak unless it removes
   coarse surprises (this book's `cent_max` rule) or uses price reactions only. Under `pct` it
   drove 83% of this book's tuned in-sample alpha (R2).
6. **The lab universe's $5 floor reads split-adjusted prices.** NVDA's 6-month median adjusted
   price was $0.88 before 2016-07, $1.65 before 2017-01 and $4.52 before 2018-01, so the most
   traded future winner is missing from all of E0 (`ranked.pkl`). SHOP misses 2016-H2 and
   2017-H1. Future splitters are kept out of early universes and benchmarks. The fix would rank
   on the as-traded price, i.e. a split history. The same holds for any strategy's own `min_price`
   on adjusted closes, this book's included.
7. **Closes are also dividend-adjusted backward.** AGNC's mid-2016 close is ≈ $5.7 in the panels
   against ≈ $20 as traded; AAPL's is ×0.91 of its split-adjusted as-traded price. Any ratio of an
   unadjusted quantity (EPS, SEC share counts) to the adjusted close is inflated by FUTURE
   dividends: the price-scaled SUE by ~1/D(t), 2-3.5× for high-yield names in 2016. That inflation
   is symmetric in sign, so it moves high-yield names to both ends of the ranking. The cent rule
   flags most of them (AGNC, F, HBAN, RF, KEY, T, KMI and NLY are among the most-flagged names).
8. **The noise check understates an event book's path noise.** Re-scoring a random ~10% of events
   (the placebo) moves alpha with sd 0.57 pts on T1 and 0.63-1.61 on TRAIN. The check (5 names
   dropped) gives 0.23 and 0.37-1.00. A Δ just above "2 × noise sd" is not safely signal here.

### R10 Re-derived conclusions

1. The review's leak was real. yfinance EPS sit on today's split basis in whole cents, so a future
   splitter's early surprises are coarse and extreme. Under the % surprise they leaned long (59% of
   coarse events vs 40% of all; coarse longs +4.6% over 80 sessions, other longs −0.6%) and carried
   ~83% of the tuned book's in-sample alpha (T0 → T0q: TRAIN +8.05% → +1.34%). "Tuning fitted
   noise" was really "tuning fitted a data leak plus noise". The old marginals ("% surprise,
   broad bands, long holds") reverse on the fixed code.
2. The default book was not flattered by the leak. Fixing it LIFTED T1 (+0.3% → +3.1%, luck 96th
   percentile, 20/20 placebos beaten) and T2 a little (−1.0 → −0.2%), and left TRAIN within noise
   (−2.1 → −2.6%). T4 at the second read gives −1.9%. The verdict is still no edge: one test of four
   is positive, and late fills erase it.
3. The rest of §8's verdict stands. The drift is front-loaded (5-10 sessions, t ~2 on TRAIN) and
   too small to pay ~450 charged orders a year: late fills cost 3.6 pts a year *[round 2: on lab
   v2 every order is charged, ~575 a year, and late fills cost 4.6 pts (V5 #3)]*. Holding to the
   next report hurts (−0.3%, t −1.6). Crowded names drift and uncrowded names reverse, but the
   crowded-only book (I2f) loses 3.1 pts on T1 against the fixed default.

## Round 2: lab v2 re-evaluation (2026-10-23)

Lab v2 (`strategies/proposals_lab/README.md`, "Lab v2") fixes five lab-level defects that biased
every round-1 number: holiday rows in the 2026 panels; each period's day 0 dropped, with its
opening costs and up-front borrow; new shorts starved by the engine's call order under
`lock_cash`; borrow mis-charged on hair-sized short re-trims; six sector ETFs typed "equity" in the
stock universes. The panels were rebuilt (NYSE sessions only, ETFs excluded, current DB).
Round-1 results stay in `var/proposals_lab/results/earnings_drift/`. Everything below is in
`results2/earnings_drift/`, and its labels end in `v2`.

### V0 Pre-registration (written before any lab-v2 book was run)

**The strategy fix the round-2 packet asks for** (neutralize coarse surprises, rank those events
on EAR alone, add a test) has been in the code since R0: `cent_max` 0.25, and U0f is the shipped
config. Before this section, only data properties of the v2 TRAIN panels were read (no returns,
no book): `diagnostics.py tiny`, `diag2-tiny.log`. They were first computed by a scratchpad
script, and the committed command reproduces them exactly. Of 9,315 TRAIN events with an
estimate, 158 (1.7%) have |estimate| < $0.05.
- Under `pct` the rule neutralizes 98% of them. The other 3 have no actual, so they have no
  surprise to neutralize.
- Under `price` it neutralizes 26% of them. The other 117 keep a price-scaled surprise whose
  cent quantum (0.01 / close) is a median 8% of the trailing P20-P80 spread. The rule caps that
  share at 25%. These kept events lean short (22 long, 30 short), as the review found for U0, so
  they do not flatter the book.

A cent on a $30 stock is small whatever the estimate, so under `price` the quantity that decides
precision is 0.01 / close, and the rule reads it. No separate |estimate| floor is added. A new
test pins the reviewer's scenario: a one-cent beat or miss on a tiny split-adjusted estimate
cannot move the book.

**Labels.** All runs use `--costs2` on OPEN (E0, TB, T3, T1, T2). The base for every Δ is U0v2.

| label | overrides | role |
|---|---|---|
| U0v2 | `{}` (= round-1 U0f: U0 + `cent_max` 0.25) | the default and the round-1 final |
| U0Lv2 | `{"cent_max": Infinity}` (= round-1 U0, rule off) | diagnostic: splits v1 → v2 into the fix and the lab |
| T0v2 | T0f's values: price, w_ear 0.25, ear_end 1, trail 126, q 0.30, hold 50, slots 10 | re-measures variant 5 (tuned on the fixed code) |
| I1v2 | `{"hold": 20}` | re-measures variants 2 / 6 |
| I2v2 | `{"crowd": "crowded"}` | re-measures 3 / 7, the strongest rejected direction on the old code |
| I3v2 | `{"w_ear": 1.0}` | re-measures 4 |

Then `noise` and `luck` for U0v2. The adoption bars of §4 / §6 are unchanged:
- T0v2 must beat U0v2 on T1 AND T2 (paired).
- I1v2-I3v2 need Δ ≥ 0 on the stitched TRAIN era, T1 and T2, AND a T1 Δ above 2 × U0v2's T1
  noise sd.

These are re-measurements, not new variants, so the count stays at 7 of 10. The tuning step is
re-checked without a new search. The ten best distinct trials of the round-1 T0f study are
re-scored on the v2 TRAIN books (robust score). A re-tune runs only if v2 reorders them so that
T0f's pick leaves the top three. That re-tune would be a counted variant, with its journal in
`studies2/`.

**Expected, from the mechanics (no v2 number read):**
1. **Day 0** is the main change for this book. Every half-year starts with a nearly full
   carried-in book, because the 60-session holds from the October and April seasons run into
   January and July. v2 charges about one rebuild of ~0.9 gross per period. At 12 bp a leg (fee
   11 bp at $1,000 orders + 1 bp slippage) that is ≈ −0.2 pt a year of alpha. At +50 bp (62 bp
   a leg) it is ≈ −1.1 pts, and at 2× costs ≈ −0.4 pt. Each day 0's open-to-close move is added
   too (zero mean). Configs with emptier books at the period starts pay less: I1 (20-session
   hold) and T0f (50).
2. **Call order.** Round 1 found ~1% of entries rejected or part-filled, mostly new shorts (U0f
   T1: 6 + 8 of 1,615). v2 should fill them. |effect| < 0.2 pt.
3. **Borrow on re-trims.** This book never re-sizes an open short (constant targets; `retrim_k`
   acts on longs), so no effect.
4. **Holiday rows and ETFs.** Events end 2026-06-11 and ETFs have no earnings. Only the
   benchmark and the EAR market change (6 of 500 names since 2023-10), and the June-2026 holds
   count one more real session. Negligible.

So U0v2 ≈ U0f − 0.2 pt: T1 ≈ +2.9%, TRAIN ≈ −2.8%, T2 ≈ −0.4%, T4 ≈ −2.1%, and T1 late-fill
alpha ≈ −1.6%. G1, G2, G3 and G5 still fail, so the status "no edge" is unchanged. G4 (96th
percentile) and G6 (down months +0.09%) were marginal passes and can flip. G7 and G8 pass. No
adoption decision changes: every round-1 Δ against U0f was ≥ 2.8 pts from its bar, and v2 moves
the variants' Δ by ~0.1-0.3 pt (I1 and T0f slightly up).

**Final candidate and T4.** U0v2 (U0f's config) stays the final unless an adoption decision
flips. A new final would be declared here before its verdict and its T4 read. T4 (names
1001-1500) is then run ONCE on v2 for U0v2. That is a disclosed THIRD read of a test read twice
in round 1 (U0 in §8, U0f in R5): a re-measurement on corrected accounting that selects nothing.
TL is not read.

**New iterations** run only if v2 opens a direction the round-1 record did not have: up to 3,
each pre-registered here first, counted, and judged by the lab bar. None is planned, because v2
changes the accounting, not the signal.

### V0a Amendment (2026-10-23, still before any lab-v2 book was run)

The new test of the reviewer's scenario (a one-cent beat or miss on a tiny estimate) FAILED on the
R0 code, and that exposed a gap in the rule. While the trailing window holds fewer than
`min_events` surprises, the spread is undefined, so the R0 code could not flag an event as coarse.
Those raw surprises then stayed in the reference distribution of later events.

On the lab panels this is the first ~90 events of January 2015. Through the two-stage trailing
ranks they still move 214 (`price`) / 301 (`pct`) event ranks in E0's first scored period
(2016-H2) and one in 2017-H1. Nothing after 2017-01-25 changes, so T3, T1, T2 and T4 are
untouched. That count comes from comparing `events()` with and without the change: event ranks
only, no book.

The fix: an unjudged window counts as coarse, so a surprise is used only where its precision was
checked. `cent_max` = inf now switches the rule off exactly; the old "off" value 1e9 would still
flag the warm-up. U0v2 therefore equals round-1 U0f except in E0 2016-H2. *[Wrong, corrected
after the run (V1): the slot book is path-dependent. Different events in 2015-16 mean different
holdings, and so different free slots for months, so V0a moves every E0 period and TB's first
two. Under v2 accounting E0 is −1.0 pt (`diag2-warmup.log`). From TB 2019-H2 on the paths are
identical again.]* U0Lv2 uses
`{"cent_max": Infinity}`, which equals round-1 U0 bit for bit in its signals. The tests now
include the reviewer's scenario and the warm-up rule (19 tests). The rest of V0 stands.

### V1 U0v2, the default on lab v2 (`U0v2.log`, `U0v2-summary.log`)

`lab.py run earnings_drift U0v2 '{}' E0,TB,T3,T1,T2 --costs2` (80 s). T1's and T2's last period
(2026-07 → 10-16) still gets no new events.

| test | alpha | P(α>0) | beta | late α | 2× cost α | top-5-out α | maxDD | round-1 U0f alpha |
|---|---|---|---|---|---|---|---|---|
| E0 | −2.4% | 0.13 | 0.04 | −7.1% | −3.5% | −4.0% | −4.1% | −0.9% |
| TB | −3.1% | 0.16 | −0.01 | −7.6% | −4.2% | −5.3% | −12.1% | −3.0% |
| T3 | −6.5% | 0.03 | 0.02 | −11.2% | −7.6% | −8.5% | −9.0% | −6.3% |
| **T1** | **+2.2%** | 0.81 | −0.01 | −2.5% | +1.1% | +0.6% | −12.5% | +3.1% |
| **T2** | **−0.8%** | 0.37 | −0.03 | −5.4% | −1.9% | −2.5% | −11.9% | −0.2% |

- **Stitched TRAIN era:** −3.30% (P 0.02), against −2.63% for U0f.
- **T1 periods:** +3.5, +2.0, −4.0, −10.8, +8.1, +2.4, +11.5, +5.8, +3.9, +3.7.
- **T2 periods:** −3.6, +5.6, +2.6, −5.3, −6.6, +3.6, −0.3, +9.8, −5.3, −5.6.
- **Market's down months:** the book's mean is +0.09% (n 41, 51% > 0); the market's is −3.98%.
- **Bear episodes** (book vs equal-weight market): Q4 2018 −1.1% vs −18.2%, COVID −2.2% vs
  −37.0%, 2022 bear +4.5% vs −24.5%, Feb-Apr 2025 −2.3% vs −19.5%.
- **Orders:** 570-581 a year on every test.

**Noise** (`U0v2-noise.log`), sd of the 3 copies: E0 0.54, TB 1.00, T3 0.27, **T1 0.18**, T2 0.24
pts. The iteration bar is a T1 Δ above **+0.36**.

**Luck** (`U0v2-luck.log`, 100 shuffled-name books on T1): real +2.21%, books' median −1.11%, p95
+2.65%, **93rd percentile**. Round 1 had 96th, so G4 flips to a fail, as V0 allowed.

**Where v1 → v2 moved, per test** (`diag2-v1acct-U0v2.log`, `diag2-v1acct-U0Lv2.log`,
`diag2-warmup.log`). The config is re-run on the v2 panels under lab v1's accounting, i.e. the
engine's own runner with each period's day 0 dropped. That book reproduces the round-1 record to
the last digit wherever nothing else changed: every period from TB 2019-H2 to T1 2024-H1. So
the reconstruction is exact.

| test | U0f (v1) | v1 accounting, v2 panels | U0v2 | panels + V0a | accounting |
|---|---|---|---|---|---|
| E0 | −0.93% | −2.04% | −2.39% | −1.11 | −0.34 |
| TB | −2.95% | −2.96% | −3.08% | −0.01 | −0.11 |
| T3 | −6.28% | −6.28% | −6.47% | 0.00 | −0.19 |
| T1 | +3.12% | +2.71% | +2.21% | −0.41 | −0.50 |
| T2 | −0.22% | −0.50% | −0.78% | −0.28 | −0.28 |

- **The TRAIN-era panels did not change.** The same decomposition with the rule off (U0Lv2
  against round-1 U0) gives +0.0000 on every E0, TB and T3 period. With the R0 rule on the
  v1-accounting side (`diag2-v1acct-U0v2-r0.log`), U0v2's config reproduces U0f to the last
  digit on every E0 and TB period. So U0v2's E0/TB "panels" column is the V0a code change
  alone, through the book's path: E0 shipped −2.39% vs R0 handling −1.37% under v2 accounting;
  per period −1.7, +0.5, −1.9, −0.8 pts.
- **T1/T2's panel change starts in 2024-H2.** That is the ETF exclusion: since 2024-H2 the six
  sector SPDRs ranked inside the top 500, so excluding them shifts six names into T1 and six
  into T2. The 2026 holiday rows add to it. It is path noise of a size the noise check
  predicts: the rule-off book GAINED +0.38 on T1 from the same panels.
- **The accounting change averages −0.3 pt a year**, close to V0's −0.2 for day 0. It is noisy
  per period (−1.2 .. +1.2) because each day 0 adds the carried-in book's first open-to-close
  move.
- **The TRAIN premise screen** (`premise2-U0v2.log`) reproduces R8 to the third decimal. The IC
  of the score is +0.016 / +0.000 / +0.020 at 20 / 40 / 60 sessions, and the top − bottom fifth
  +0.14% / −0.09% / −0.38%. V0a changes a few early event ranks, not the signal; its −1.0 pt on
  E0 is the book's path.

### V2 The round-1 decisions re-measured (`report2-U0v2.log`, `diag2-paired-train.log`)

Δ = paired alpha vs U0v2 (P(Δ>0), periods better). The bar for T0v2: Δ > 0 on T1 AND T2. The bar
for I1v2-I3v2: Δ ≥ 0 on stitched TRAIN, T1 and T2, and T1 Δ > +0.36.

| label | stitched TRAIN Δ | T1 alpha / Δ | T2 alpha / Δ | round 1 (vs U0f): TRAIN / T1 / T2 | adopted |
|---|---|---|---|---|---|
| T0v2 (T0f's values) | +6.4 (0.99, in-sample) | +2.8% / **+0.6** (0.55, 5/10) | −3.3% / **−2.5** (0.22, 3/10) | +6.0 / +0.2 / −2.8 | no (T2) |
| I1v2 hold 20 | +2.5 (0.93) | −0.6% / **−2.8** (0.11, 2/10) | −2.2% / −1.5 (0.25, 5/10) | +1.5 / −3.5 / −1.6 | no |
| I2v2 crowded only | +2.7 (0.96) | 0.0% / **−2.2** (0.09, 3/10) | −1.1% / −0.3 (0.44, 4/10) | +2.8 / −3.1 / −0.5 | no |
| I3v2 reaction alone | +0.7 (0.65) | −1.7% / **−3.9** (0.03, 1/10) | −2.2% / −1.4 (0.24, 4/10) | +0.2 / −4.3 / −0.9 | no |
| U0Lv2 rule off (diagnostic) | +1.0 (0.90) | +0.3% / −1.9 (0.005, 2/10) | −1.6% / −0.8 (0.07, 2/10) | +0.6 / −2.8 / −0.8 | — |

- **No decision changes.** Every Δ keeps its round-1 sign on T1 and T2, and every variant is
  still ≥ 2 pts from its bar.
- **The cent rule is a fix, not a variant.** Judged as an iteration it would fail the bar, with
  a TRAIN Δ of −1.0 here and −0.6 in round 1. It stays because it removes a data leak (the
  precision of a surprise must not depend on future splits). That is decided on correctness, not
  returns, and a leak is not kept for helping a test. Its T1 gain (+1.9) is reported as the
  coarse-event effect it is (20/20 placebos), not as evidence of drift.
- **Every TRAIN Δ is now ≥ 0.** U0v2's own TRAIN got 1 pt worse through V0a's E0 path; that is
  TRAIN path noise (round 1's placebo: sd 0.6-1.6 pts there).
- **Tuner re-check** (`diag2-rescore-T0f.log`): the round-1 T0f study's 10 best distinct trials,
  re-scored on the v2 TRAIN books, keep their order (rank correlation +0.80). T0f's pick is
  still #1 (robust +0.0105 vs +0.0079). By V0's rule there is no re-tune.
- **New iterations: none.** v2 opened no direction the round-1 record lacked. Every alternative
  that gained on TRAIN still lost on T1. The variant count stays at 7 of 10.

### V3 Final candidate (declared 2026-10-23, before `verdict`, `bear` and the T4 read)

**U0v2**: the shipped config (`{}`, the pre-registered U0 plus the cent rule `cent_max` 0.25,
with V0a's warm-up handling), on lab v2. No adoption decision changed, so the final is the
round-1 final re-measured. T4 (names 1001-1500) is run once for it. That is the disclosed third
read of T4 (U0 in §8, U0f in R5), a re-measurement on corrected accounting that selects nothing.
TL is not read.

### V4 Verdict, bear table, T4 (third read) (`U0v2-verdict.log`, `bear2.log`, `U0v2-T4.log`)

`lab.py verdict earnings_drift U0v2` (luck from its cache, 93rd percentile):

| gate | U0 (v1, leaky) | U0f (v1) | **U0v2** | passed |
|---|---|---|---|---|
| G1 TRAIN-era alpha > 0 | −2.05% | −2.63% | −3.30% | no |
| G2 T1 alpha > 0, P ≥ 0.90 | +0.33% (P 0.56) | +3.12% (P 0.897) | +2.21% (P 0.81) | no |
| G3 T2 alpha > 0 | −1.05% | −0.22% | −0.78% | no |
| G4 T1 beats ≥ 95% of shuffled books | 70th | 96th | **93rd** | **no** (flipped) |
| G5 T1 late-fill and 2×-cost alpha > 0 | −3.26% / −0.55% | −0.55% / +2.22% | −2.46% / +1.08% | no |
| G6 every bear episode beaten, down months > 0 | episodes yes; −0.13% | yes; +0.09% | yes; +0.09% | **yes** |
| G7 \|T1 beta\| ≤ 0.3 | 0.016 | −0.009 | −0.010 | **yes** |
| G8 T1 alpha without its top 5 names > 0 | −0.89% | +1.27% | +0.64% | **yes** |
| gates passed | 1 | 4 | **3** | |

`lab.py bear earnings_drift U0v2 U0Lv2 T0v2 I2v2` (stitched E0+TB+T3+T1):
- **Episodes** (U0v2, with the equal-weight market and SPY): Q4 2018 −1.1% (−18.2%, −18.7%),
  COVID −2.2% (−37.0%, −33.4%), 2022 bear +4.5% (−24.5%, −24.1%), Feb-Apr 2025 −2.3% (−19.5%,
  −18.6%).
- **CAGR −0.8%, max drawdown −22.1%** (U0f: −0.0%, −20.3%).
- **Calendar years:** 2016 −0.8, 2017 −1.9, 2018 +0.8, 2019 −6.8, 2020 −2.6, 2021 −6.5, 2022
  +2.6, 2023 −8.3, 2024 +5.2, 2025 +9.3, 2026 +2.2.
- **For comparison:** U0Lv2 (rule off) CAGR −1.0%; T0v2 +3.1% (its TRAIN years are in-sample,
  max DD −24.6%); I2v2 −0.3%.

**T4, names 1001-1500, THIRD read** (U0v2 only, declared in V3; `U0v2-T4-summary.log`). This is a
re-measurement of the round-1 final on corrected accounting and selects nothing:
- **Alpha −3.40%** (P 0.05), beta −0.02.
- Late fills −7.9%, 2× costs −4.5%, without its top 5 names −5.0%, max drawdown −15.9%.
- Periods: −11.8, −1.7, +15.6, −7.3, −4.0, −4.1, −3.9, −6.7, −1.6, −1.6.

The round-1 reads were −1.60% (U0, §8) and −1.93% (U0f, R5). Against U0f period by period:
- the five periods before 2024-07 move by −0.7 pt on average (accounting plus path noise);
- the five from 2024-07 move by −2.6. That is where lab v2 changed T4's names: the band 1001-1500
  shifts by six once the six ETFs leave the top 500. The 2026 holiday rows fall there too.

No decomposition was run on T4, to keep it at one read. The less-traded names show no drift, on
either accounting.

**Verdict (round 2): still no edge.** The status label is unchanged. On lab v2:
- **Gates:** the final passes 3 of 8 (G6, G7, G8). G4 dropped from a marginal 96th to the 93rd
  percentile.
- **Tests:** TRAIN −3.3%, T1 +2.2% (P 0.81), T2 −0.8%, T4 −3.4%.
- **Costs:** +50 bp late fills take T1 to −2.5%.

The one positive test (T1) rests on the cent rule's coarse-event effect (V2: +1.9 pts, 20/20
placebos). TRAIN, T2 and T4 all have negative alpha, so it is not a tradable drift.

### V5 The review issues: fix and effect on lab v2

| # | issue (severity) | fix | effect on v2 |
|---|---|---|---|
| 1 | yfinance EPS leak future splits: tiny, whole-cent EPS early (major) | The cent rule `cent_max` 0.25 (R0, in the code since round 1). V0a: an unjudged warm-up window counts as coarse, and inf = rule off. Two new tests (one-cent beat vs miss on a tiny estimate cannot move the book; warm-up). No \|estimate\| floor (V0: 98% of tiny estimates are already neutral under `pct`; under `price` the quantum is 0.01 / close). The T0 learning is amended (V8). | Rule vs no rule: TRAIN −1.0, **T1 +1.9** (P 0.995, 20/20 placebos), T2 +0.8 (16/20). V0a alone: E0 −1.0 through the path. In round 1 the leak carried ~83% of the tuned book's TRAIN alpha (T0 → T0q −6.7 pts). |
| 2 | I1 mechanism overstated: "turnover ×3" (major) | Wording corrected in §6, R6 and the README, with measured v2 numbers (`diag2-book.log`) | Hold 20 on T1: orders ×1.40 (800 vs 570 a year; T2 ×1.27), gross 0.48 vs 0.87 (T2 0.44 vs 0.87), 1× cost drag 1.55 vs 1.12 pts, late penalty 6.3 vs 4.7. Alpha per unit of gross: T1 −1.3% vs +2.5%, T2 −5.1% vs −0.9%. |
| 3 | Cost arithmetic mixed two bases (minor) | Restated. Lab v2 charges every order, including each period's first-bar re-buy of the carried-in book (~125 a year). | ~575 orders a year, each ~1.6% of the book. Measured 1× cost drag 1.10-1.15 pts a year (≈ 575 × 1.63% × 12 bp). +50 bp late fills cost 4.55-4.74 more (≈ 575 × 1.63% × 50 bp). Entries ~215 a year (3.6 per slot). |
| 4 | Gross capped only at entry (minor) | Documented as the entry gross (config, README); no code change | Realized gross on v2: T1 mean 0.90, max 1.29, > 1.0 on 30% of bars; T2 max 1.21 (28%); E0 max 1.07 (29%) |
| 5 | Engine starves new shorts under `lock_cash` (minor; engine) | Lab v2's runner orders each bar's trades (covers before new shorts); `src/` untouched | Rejected + partial entries: T1 0 + 2 of 1,618 (v1: 6 + 8), T2 0 + 3 of 1,619 (v1: 1 + 13), E0 0 + 0. Part of the −0.3 pt accounting change. |
| 6 | Numbers only in a transcript (minor) | Every round-2 number comes from a committed command (`diagnostics.py` v1acct, warmup, rescore, paired, book, anatomy, placebo, corr; `lab.py`), logged in `results2/earnings_drift/` | Correlation with p1's v6_si book on v2: T1 daily +0.21 / monthly +0.28, T2 +0.12 / +0.15 (U0f v1: +0.22 / +0.32, +0.09 / +0.22) |
| 7 | Robust −0.046 and "consistent" (minor) | Corrected in place (§4, §7) | — |
| 8 | Juneteenth row (minor; lab) | Lab v2 keeps NYSE sessions only | Inside T1/T2's 2024-H2+ "panels" change (V1) |

The lab-level defects, as they touched this book:
- **Day 0:** −0.3 pt a year, mostly costs.
- **Starved shorts:** #5.
- **Borrow on re-trims:** none; this book never re-sizes an open short.
- **Holiday rows and ETFs:** T1 −0.41, T2 −0.28, and T4's later periods. That is path noise from
  the six-name universe shift.

### V6 v1 → v2, the headline numbers

| | U0 v1 (leaky) | U0Lv2 (leaky, v2) | U0f v1 (round-1 final) | **U0v2 (final)** |
|---|---|---|---|---|
| TRAIN (E0+TB+T3) alpha | −2.05% | −2.30% | −2.63% | **−3.30%** |
| T1 alpha (P) | +0.33% (0.56) | +0.32% (0.57) | +3.12% (0.90) | **+2.21% (0.81)** |
| T2 alpha | −1.05% | −1.59% | −0.22% | **−0.78%** |
| T4 alpha (read) | −1.60% (1st) | — | −1.93% (2nd) | **−3.40% (3rd)** |
| T1 late / 2× costs | −3.26% / −0.55% | −4.27% / −0.79% | −0.55% / +2.22% | **−2.46% / +1.08%** |
| T1 luck percentile | 70 | — | 96 | **93** |
| T1 noise sd (bar) | 0.35 (0.69) | — | 0.23 (0.46) | **0.18 (0.36)** |
| gates passed | 1 (G7) | — | 4 (G4, G6, G7, G8) | **3 (G6, G7, G8)** |
| Q4 2018 / COVID / 2022 / Feb-Apr 2025 | −0.8 / −1.9 / −0.1 / −3.9 | −1.0 / −1.9 / −0.4 / −3.9 | −0.7 / −2.2 / +4.8 / −2.3 | **−1.1 / −2.2 / +4.5 / −2.3** |
| down months (book) | −0.13% | −0.12% | +0.09% | **+0.09%** |
| CAGR / max DD (2016-26) | −0.8% / −20.3% | −1.0% / −22.2% | −0.0% / −20.3% | **−0.8% / −22.1%** |
| status | no edge | — | no edge | **no edge** |

The market in the episodes: −18.2 / −37.0 / −25.2 (v2 −24.5) / −19.6 (v2 −19.5)%. U0Lv2's
numbers are in `U0Lv2-summary.log`, `diag2-paired-train.log` and `bear2.log`.

### V7 Variant count and what round 2 changed

The variants are unchanged at 7 of 10, because round 2 re-measured and added none:

| # | label | what | TRAIN Δ | T1 Δ | T2 Δ | adopted |
|---|---|---|---|---|---|---|
| 1 | T0 | tuned, old code | +10.1 | +0.3 | −3.7 | no |
| 2 | I1 | hold 20 | +0.9 | −1.1 | −1.0 | no |
| 3 | I2 | crowded names only | +1.6 | +0.3 | +1.4 | no |
| 4 | I3 | reaction alone | −0.4 | −1.5 | −0.1 | no |
| 5 | T0f / T0v2 | re-tuned on the fixed code | +6.0 / +6.4 | +0.2 / +0.6 | −2.8 / −2.5 | no / no |
| 6 | I1f / I1v2 | hold 20 | +1.5 / +2.5 | −3.5 / −2.8 | −1.6 / −1.5 | no / no |
| 7 | I2f / I2v2 | crowded names only | +2.8 / +2.7 | −3.1 / −2.2 | −0.5 / −0.3 | no / no |
| — | I3 / I3v2 | reaction alone vs the fixed default | +0.2 / +0.7 | −4.3 / −3.9 | −0.9 / −1.4 | no / no |

What changed in round 2:
- **Numbers:** a little lower everywhere. This is mostly accounting the v1 lab skipped (day 0)
  and path noise from the corrected universes.
- **G4** fell from 96th to 93rd percentile.
- **V0a** (the warm-up rule) moved E0 by −1.0 pt through the book's path. The config's headless
  study name was bumped to `earnings-drift-v3` for it.

What did not change:
- every adoption decision;
- the final (U0 + the cent rule);
- the tuner's pick;
- the status, no edge;
- the reading that the one positive test (T1) is the cent rule's coarse-event effect and not a
  drift that carries to TRAIN, T2 or T4.

### V8 Learnings (round 2)

1. **The T0 learning, amended** (review #1). Round 1 read the tuned book as "tuning fitted
   noise". It fitted near-zero split-adjusted estimates (future splitters included) under the %
   surprise, plus the 2016-21 growth tilt, plus noise. With the cent rule the same parameters
   make +1.3% a year on TRAIN instead of +8.1% (R2). Re-tuned on the fixed code, the study
   prefers `price`, and its pick is stable on lab v2 (V2), but it still loses T2. Any lab book
   that reads `earn_surprise` / `earn_actual` / `earn_estimate` needs the cent rule or must rank
   on price reactions alone.
2. **A first-come slot book is path-dependent.** Re-scoring the ~90 warm-up events of January
   2015 moved every E0 period (−1.0 pt stitched) and TB's first two, through which events found
   free slots. A Δ on E0/TB between two configs that differ in their early events is partly path
   noise. The 5-name noise check understates it, and the placebo (R1) is the better yardstick.
3. **Cost, not signal, is this book's binding number.** All orders charged, the base drag is 1.1
   pts a year and +50 bp late fills cost 4.6 more. That is more than the alpha of any test, on
   either lab version.
4. **How to re-measure across lab versions.** Re-run the config under the OLD accounting on the
   NEW panels (`diagnostics.py v1acct`). Where nothing else changed it reproduces the old record
   to the last digit, which splits a move into data, code and accounting. Here the TRAIN-era data
   were unchanged, and the post-2024-H2 changes are the ETF band shift.
