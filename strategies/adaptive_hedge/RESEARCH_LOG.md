# adaptive_hedge — research log

Proposal 3 of `STRATEGY_OVERVIEW.md` §6, evaluated through the shared lab
(`strategies/proposals_lab/README.md`, protocol mandatory). Every number below comes from
`lab.py` on the lab panels (no DB), unless a section says otherwise. Newest sections at the
bottom; nothing above a section is edited after its numbers were read, except marked `[later]`
notes.

[later, round 2] Every number in §1-§12 was produced by lab v1, whose five defects (holiday rows,
day 0 dropped, starved new shorts, re-trim borrow, ETFs in the universe) biased it. §13 re-measures
everything on lab v2 and supersedes them; the current headline numbers are in §13.6-§13.7.

[later, round-2 repair] §14 adds the target-versus-realized reconciliation. One TB cover that lab
v2 refused moves TB from +1.9% as run to an estimated +5.3%. §13 is corrected in place, with
`[repair]` marks.

## 0. Pre-registration (2026-10-22 22:50, written before any number of this strategy existed)

### Hypothesis

A long book of the least-shorted names (informed_flow v41's rule) plus trend-breakdown shorts that
take slots only when single stocks break down gives a book whose net market exposure falls by
itself when many stocks break down (a falling market) and stays nearly fully long when few do (a
rising market). It should therefore keep most of the long book's alpha in rising markets and beat
equal-weight buy-and-hold in falling ones, without any market-timing rule (the 200-day gate failed
in trend_following: -5.6 points, t -5.7).

### Mechanism

1. **Long side, informed short sellers.** Low days-to-cover names outperform (Boehmer, Jones &
   Zhang 2008; Hong, Li, Ni, Scheinkman & Yan 2016). The repo's only replicated signal.
2. **Short side, time-series momentum on single stocks.** A decisive breakdown (close below the
   prior 9-bar low, ADX >= 34, -DI > +DI) tends to continue (Moskowitz, Ooi & Pedersen 2012;
   Wilcox & Crittenden 2005). The number of such breakdowns is a cross-sectional breadth measure:
   it rises in a sell-off, so the short side is an endogenous hedge whose size follows breadth.
3. **Slot accounting makes the hedge double.** A short that displaces a long moves net exposure
   by -2 slots (-6.5% of the book at 30 slots).

### U0 — the untuned default (the proposal's spec; nothing searched)

- Portfolio mode, 30 slots of gross/30, gross 0.98, `retrim_k` 2 (longs), `fill: open`,
  `order_notional: 1000`.
- **Long side** = informed_flow v41: rank minus days-to-cover averaged over 42 bars inside 5
  dollar-volume groups (63-bar median dollar volume), 50% per-bar coverage floor (share of
  names with a close >= $5 that carry a live days-to-cover), re-ranked across eligible names;
  eligible = close >= $5, days-to-cover present, `fund_shares` > 0. Enter at rank >= 0.90, leave
  below 0.80, `max_hold` 126 (clock restarts while ranked >= 0.90), longs never displace longs.
- **Short side** = trend_following v7's short starts: close < prior 9-bar low, ADX(14) >= 34,
  -DI > +DI (Wilder smoothing, copied from `trend_following/strategy.py`), close >= $5, and short
  interest / shares outstanding <= 30%. Priority when starts outnumber free short slots: the most
  negative 12-1 month relative strength first (trend_following's rule). A start takes a free slot
  at -gross/30; with no slot free it displaces the weakest-ranked long. At most 20 shorts (net
  exposure >= -33%).
- **Short exits**: a close above the ratcheting chandelier (lowest low since entry + 14 x
  ATR(37); starts at the entry close + 14 ATR; never loosens; copied from trend_following's
  `_ride`), an opposite (long) start (close > prior 9-bar high, ADX >= 34, +DI > -DI), the name
  becoming too crowded (> 30% of shares short), or no price. No squeeze stop in U0 (the
  overview's +50% stop is an iteration idea in the task; the 14-ATR chandelier already sits
  ~+30-70% above the low).
- **Borrow** = informed_flow_ls_p1's tiers: 0.5% a year below 10% short interest / shares,
  3% up to 20%, 10% above; 0.5% where there is no short-interest data.

### What is already in-sample (honesty about the defaults)

- The long side's values (42 / 5 groups / 30 slots / 0.90 / 0.80) were chosen by informed_flow's
  walk-forward over 2022-01 -> 2026-09 on a rolling top 500, i.e. the lab's **T1**.
- The short side's values (9 / 14 / 34 / 37 / 14 / 12-1 priority) were optimized by
  trend_following v7 on Liquid500 2022-24 with its 2025-26 holdout read: **T1 is in-sample for
  them too**.
- informed_flow_ls read 2018-07 -> 2020-06 once (§37, alpha -3.5%) and names 501-1000 as its
  "fresh names" test, but never this combination. TB and T3 are clean for the short side; T2 is
  clean for the short side and for the combination. **T1 numbers are therefore the weakest
  evidence here; T2 and TRAIN (TB + T3) matter more.** T4 stays locked until the final candidate.
- Training for this proposal = **TB + T3** only (E0 has no short interest, so the long side would
  be empty there). E0 is not run.

### Expected result (before any number)

- Long side alone (control L0): T1 alpha +5 to +9% at beta ~1.1 (informed_flow's walk-forward:
  +8.8% at 1.14); TB weak or negative (the signal read IC -0.055 on forward-selected names over
  2018-07 -> 2020-06; informed_flow_ls v6 lost 3.5%/yr there).
- U0: alpha 0 to +6% on T1 at beta 0.6-0.9, i.e. the hedge costs 2-5 points in the 2023-24 bull
  (shorts squeezed in rebounds, as trend_following's -19% in 2023). 2022: ahead of buy-and-hold.
  COVID: breakdown shorts need ~2 weeks of falling prices, so they arrive late and get squeezed
  in the April 2020 rebound: probably not much help. Q4 2018: some help.
- **Main risk, stated before measuring:** a 500-name universe has breakdowns every day even in a
  bull market, and the short exits (a 14-ATR chandelier, an opposite start that needs ADX >= 34)
  are rare, so the 20 short slots may simply stay full. Then U0 is a permanent net short (~10
  longs, 20 shorts = -33% net) instead of an adaptive hedge, and loses heavily to buy-and-hold
  in every bull year. The premise check measures how often short starts happen and how long a
  short would be held before U0 is run.

### Pass criteria

- The lab gates (`lab.py verdict adaptive_hedge <label> --not-neutral`): G1 TRAIN-era (TB + T3)
  alpha > 0; G2 T1 alpha > 0 with P >= 0.90; G3 T2 alpha > 0; G4 T1 beats >= 95% of shuffled
  books; G5 late fills and 2x costs; G6 every bear episode + down months > 0; G8 top-5-out. G7
  skipped (not market-neutral).
- The proposal's own bars (STRATEGY_OVERVIEW §6, proposal 3):
  - P1: T1 alpha >= the long side alone (L0) minus 2 points (the hedge costs at most 2 a year);
  - P2: max drawdown <= 2/3 of buy-and-hold's (T1 and TB, stitched);
  - P3: in 2022, net exposure below 50% by mid-year (the book's net exposure on 2022-06-30), and
    the 2022 calendar-year return at least 10 points ahead of buy-and-hold;
  - P4: proposal 1's bear table on 2018-07 -> 2020-06: in Q4 2018 and in the COVID crash the
    book loses less than half of what the market lost.
- Judged also on CAGR vs buy-and-hold and vs SPY (`lab.spy()`), alpha and max drawdown.

### Premise checks (TRAIN only: TB + T3; informational)

1. Long signal: rank IC and top-minus-bottom decile spread of the within-group rank of minus
   days-to-cover (42 bars) against the forward 21- and 63-bar beta-adjusted return (126-bar beta
   to the period's equal-weight names), sampled every h bars inside each period.
2. Short starts: forward 21- and 63-bar return after a short start (next open to the close),
   raw, minus the equal-weight market, and beta-adjusted, averaged per date first (t over dates).
3. The adaptive premise: daily count of short starts vs the market's trailing 21-bar return, by
   up and down months; and the implied short occupancy (how long a short would be held under the
   chandelier / opposite-start exits).

### Budget and adoption rule

- At most 8 variants through the bar (README) and never more than 10 runs of any kind (task cap),
  tuner trials excluded. Planned: U0; L0 (control: `max_shorts: 0`, the long side alone); TUNED
  (lab.tune on TB,T3); B (pre-registered: shorts only among names in the crowded half by
  days-to-cover, i.e. long-side rank <= 0.5); then at least two design iterations written from
  what U0, the premise check and the tune show.
- Tune (pre-registered space, TB,T3, 120 trials, seed 42, robust = mean - 1/2 sd of the 7 period
  alphas, PBO recorded): `max_shorts` int 0-20 step 2, `adx_min` 20-50 step 2, `entry_window`
  int 5-40, `atr_mult` 4-16 step 1, `atr_window` int 14-60. Adopted only if it beats U0 on BOTH
  T1 and T2 (paired, `lab.py report`).
- An iteration is adopted only if its paired delta vs the current base is >= 0 on TRAIN (TB and
  T3), T1 and T2 AND its T1 delta exceeds twice U0's noise sd (`lab.py noise`).
- Final candidate declared here before `lab.py verdict`, `lab.py bear` and the one T4 read
  (final vs U0).

## 1. Premise check, TRAIN only (TB + T3; `premise.py`, 2026-10-22 23:05)

Informational. Output saved in `var/proposals_lab/results/adaptive_hedge/premise.json`.

| check | h = 21 bars | h = 63 bars |
|---|---|---|
| long rank IC vs beta-adjusted forward return (non-overlapping dates) | −0.017 (t −1.5, 38 dates) | +0.005 (t 0.2, 10 dates) |
| long top-minus-bottom decile, beta-adjusted | −0.3% (t −0.8) | +1.2% (t 1.1) |
| after a short start: raw return | +2.1% (t 1.2) | +7.8% (t 1.9) |
| after a short start: minus the equal-weight market | +0.1% (t 0.1) | **+5.1% (t 1.5)** |
| after a short start: beta-adjusted | +0.2% (t 0.2) | +3.9% (t 1.4) |
| · TB / T3, minus the market | +2.0% / −2.5% | +9.8% / −1.9% |

(3,285 start events on 425 dates; t over blocks of h bars.)

Breadth (the adaptive premise):

- Starts are a breadth measure, as hoped: 0.29% of names a day in up months, **2.3% in down
  months** (8x); Spearman with the market's trailing 21-bar return −0.48 (daily).
- **But U0's short side does not adapt: it is saturated.** U0 holds 19.3 shorts on average (cap
  20, at the cap on 72% of bars): 19.0 in up months, 19.9 in down months. Shorts are held a
  median of 203 calendar days (mean 222); over TB + T3 the 20 slots exited only 114 times (51
  chandelier stops, 60 opposite starts, 3 too crowded). The book is therefore ~10.7 longs and
  ~19.3 shorts: a **permanent net short of about −28%**, the main risk named in §0.
- On TRAIN the breakdowns did not continue relative to the market; in TB (Q4 2018, COVID) they
  rebounded (+9.8% over the market within 63 bars). The long signal was flat on TRAIN (as
  informed_flow_ls v6 found: −3.5% alpha on 2018-07 → 2020-06).

What this predicts for U0 (written before running it): large losses against buy-and-hold in
every rising stretch (2019, 2021, 2023-24), a net-short beta, and an alpha that depends on whether
the single-name shorts beat the market (on TRAIN they did not). The design question for the
iterations is visible already: the starts carry the breadth signal, the 200-day holds throw it
away.

## 2. U0, the untuned default (`lab.py run adaptive_hedge U0 '{}' TB,T3,T1,T2 --costs2`)

Variant count: **1** (U0). E0 not run (no short interest before 2018: the long side is empty).

| test | alpha | P | beta | CAGR | B&H | SPY | max DD | B&H max DD | late | 2x | top-5 out | long / short |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TB | +3.3% | 0.74 | −0.17 | +1.6% | +10.5% | +8.8% | −12.1% | −37.4% | +1.2% | +3.0% | −2.3% | 0.33 / −0.59 |
| T3 | −2.5% | 0.37 | −0.21 | −10.4% | +37.3% | +35.3% | −24.5% | −7.8% | −4.4% | −3.0% | −8.6% | 0.35 / −0.62 |
| T1 | +4.4% | 0.80 | −0.28 | +1.9% | +8.6% | +12.1% | −24.2% | −26.0% | +1.2% | +3.9% | −0.9% | 0.36 / −0.57 |
| T2 | +3.4% | 0.81 | −0.29 | +1.1% | +7.7% | +12.1% | −20.0% | −24.6% | +0.9% | +2.8% | −1.0% | 0.36 / −0.57 |

- 2022: book +17.9% vs buy-and-hold −19.4% (+37 points); mean net exposure in 2022-H1 −19%.
- T1 2026-07 → 10-16 (the factor crash): alpha −54.9%, 246 orders. T2 the same period −25.7%.
  [later: not a factor crash: the US-holiday-row artifact (§12.1). With the strategy fix U0's T1
  2026-07 is −23.0% with 60 orders; on lab v2 it is −37.0% (TUNEDv2 −23.4%), §13.6.]
- **Noise** (`lab.py noise`, 5 names dropped a period, 3 copies): sd TB 1.28, T3 0.39, T1 0.56,
  T2 0.27 points. An iteration must beat its base by more than **1.1 points on T1**.
- **Luck** (T1, 100 shuffled-name books): real +4.4%, median −1.1%, 95th percentile +4.0%,
  **percentile 97** (G4 would pass).

Read: exactly the permanent net short predicted in §1 — ~36% long, ~58% short. It makes money
in falling markets (2022 +37 points over buy-and-hold) and loses everywhere else: CAGR below
buy-and-hold AND SPY on every test (T1 +1.9% vs +8.6% / +12.1%). The alpha is positive on TB,
T1, T2 (P 0.74-0.81) but it is the alpha of a −0.2 to −0.3 beta book and does not survive
removing the top five names. Not the proposal's adaptive hedge: a short book with a long
sleeve.

## 3. Optimization (lab.tune on TB,T3, 120 trials, the §0 space)

`lab.py tune adaptive_hedge tune1 '{}' '<§0 space>' 120 TB,T3` (journal
`var/proposals_lab/studies/adaptive_hedge-tune1.log`). Best trial 104: **max_shorts 20, adx_min 40,
entry_window 27, atr_mult 16 (the bound), atr_window 48**; robust score +0.045 (U0 on the same 7
periods −0.028), mean period alpha +10.9%, **PBO 0.10**. By max_shorts, the mean trial score
rises with the cap (0: −0.081 in the one trial, 20: −0.055 over 18 trials [later: a slip; the
journal has 58 trials at 20, mean −0.029, and 27 at 18, mean −0.048; §13.6]): on TRAIN the
optimizer wanted the most shorts, the rarest and most decisive breakdowns and the widest stop.

Run as TUNED (variant **2**) and the long side alone as control L0 (`max_shorts: 0`, variant
**3**, a control, not a candidate):

| test | U0 alpha | TUNED alpha (P) | TUNED beta | TUNED CAGR | B&H | SPY | TUNED max DD | L0 alpha (P) | L0 beta | L0 CAGR |
|---|---|---|---|---|---|---|---|---|---|---|
| TB | +3.3% | +7.5% (0.88) | −0.21 | +5.3% | +10.5% | +8.8% | −9.3% | −7.6% (0.05) | 1.01 | +3.0% |
| T3 | −2.5% | +6.2% (0.75) | −0.01 | +5.9% | +37.3% | +35.3% | −15.6% | −4.3% (0.33) | 1.20 | +40.3% |
| T1 | +4.4% | **+12.9% (0.986)** | −0.21 | +11.1% | +8.6% | +12.1% | −17.9% | +9.5% (0.95) | 1.15 | +19.4% |
| T2 | +3.4% | **+9.4% (0.948)** | −0.23 | +7.7% | +7.7% | +12.1% | −24.3% | +15.8% (0.999) | 0.95 | +23.1% |

Paired vs U0 (`lab.py report`): TUNED TB +4.2 (P 0.77, 3/4 periods better), T3 +8.8 (P 0.90,
3/3), **T1 +8.6 (P 0.97, 8/10), T2 +6.0 (P 0.86, 6/10)**; stitched TRAIN +9.2 (P 0.96).
TUNED: late fills T1 +11.4%, 2x costs +12.5%, top-5-out +7.4%; episodes Q4 2018 −1.5% (market
−18.2%), COVID +20.6% (−37.0%), 2022 bear +30.1% (−25.2%), Feb-Apr 2025 −1.2% (−19.6%); down
months +2.28% vs −4.10% (84% positive); 2022 calendar year +28.0% vs −19.4%.

**Decision: TUNED adopted** (it beats U0 on T1 AND T2, the tune rule). It is the base for the
iterations.

What it is, honestly: still not the proposal's adaptive hedge. `premise.py` on TUNED's starts
(TRAIN, `premise-TUNED.json`): starts are rarer (0.075% of names a day in up months, 0.97% in
down months), but a 16 x ATR(48) chandelier almost never fires — shorts are held a median of
**308 days**, 17.6 on average (16.8 in up months, 19.5 in down months), and the TRAIN starts did
not fall relative to the market afterwards (+0.3% at 21 bars, +6.4% at 63, t 0.95). The book is
a near-constant ~−15% net short of names that broke down hard, plus the long sleeve. CAGR beats
buy-and-hold on T1 (+11.1% vs +8.6%) but not SPY (+12.1%), and ties it on T2.

Proposal bar P1 (hedge cost <= 2 points vs L0): **T1 passes** (TUNED +12.9% vs L0 +9.5%: the
short side ADDS 3.4 points), **T2 fails** (+9.4% vs +15.8%: it costs 6.4). The long sleeve
alone is negative on TRAIN (TB −7.6%, T3 −4.3%): the TRAIN-era alpha of this book is the short
side's.

## 4. Iteration 1 — pre-registration (written before the code existed): the short clock

**Change.** On top of TUNED: `short_hold: 21`. A short is held only while its breakdown goes on:
it is bought back once 21 bars pass without a fresh start on that name (the clock restarts on
every new start while held, as the long side's `max_hold` clock restarts while a name ranks >=
q_in). The chandelier, opposite-start and crowding exits stay.

**Mechanism (from §1 and §3).** The starts carry the breadth signal (13x more frequent in down
months than in up months under TUNED's rule), but 300-day holds throw it away: the cap binds in
every regime. With a one-month clock the short side holds the names breaking down NOW, so its
size becomes the breadth of breakdowns: small in a rising market, full in a falling one. It also
drops the stale shorts that, on TRAIN, rebounded against the market after 63 bars (+6.4%). One
month: the horizon at which the §1 starts had stopped falling relative to the market (+0.1% /
+0.3% at 21 bars) and the short-term reversal horizon (Jegadeesh 1990). Not tuned.

**Expected.** Shorts held in rising stretches fall from ~17 to ~3-8; beta rises from −0.2 to
+0.2..+0.5; CAGR up in T3 and 2023-24; the 2022 hedge mostly kept (breakdowns refresh). Alpha:
within ±3 points of TUNED, sign unknown (the tune's preference for the widest stop says long
holds paid on TRAIN). Adopted only if the paired delta vs TUNED is >= 0 on TB, T3, T1 and T2 AND
the T1 delta exceeds twice the noise sd (max of U0's and TUNED's). Variant **4**.

### Iteration 1 — result (variant 4): REJECTED

`lab.py run adaptive_hedge I1 '{TUNED + "short_hold": 21}' TB,T3,T1,T2 --costs2`

| test | TUNED alpha | I1 alpha | paired Δ (P) | periods better | I1 beta | I1 CAGR | I1 long / short |
|---|---|---|---|---|---|---|---|
| TB | +7.5% | +2.7% | −4.8 (0.22) | 1/4 | +0.05 | +3.2% | 0.66 / −0.23 |
| T3 | +6.2% | −5.6% | −11.8 (0.13) | 1/3 | +0.89 | +27.5% | 0.83 / −0.13 |
| T1 | +12.9% | +5.6% | −7.4 (0.14) | 4/10 | +0.32 | +8.3% | 0.67 / −0.26 |
| T2 | +9.4% | +3.3% | −6.1 (0.19) | 4/10 | +0.17 | +4.6% | 0.65 / −0.27 |

Stitched TRAIN Δ −0.8 (P 0.36). Orders 2.6x TUNED's (T1 404 a year vs 157). 2022: −24.1% vs
buy-and-hold −19.4% (TUNED +28.0%); bear episodes Q4 2018 −7.9%, COVID +11.4%, 2022 −22.5%,
Feb-Apr 2025 −5.1%; down months −1.85% vs −4.10%.

Read: the clock did make the exposure adaptive (short side −0.13 in the T3 bull, beta 0.89
there), but it took the hedge away exactly when it pays: in the 2022 grind the breakdowns did not
make a fresh 27-bar low with ADX >= 40 every month, so shorts timed out mid-decline and were
re-shorted after new lows (whipsaw). **The hedge pays only if the decliners are HELD through the
bear**; a fixed clock cannot tell a decliner that is still falling from one that has recovered.

## 5. Iteration 2 — pre-registration (before code): the short leaves when its own trend reverses

**Change.** On top of TUNED: `exit_window: 27`. A short is also bought back on a close above the
highest high of the prior 27 bars (TUNED's entry window), with no ADX confirmation — the Turtle
exit (Faith 2007: shorts covered on an opposite N-bar breakout). The chandelier and the
ADX-confirmed opposite start stay; 0 = off.

**Mechanism (from §3 and iteration 1).** TUNED holds a short ~300 days because its only
practical exits need either a 16-ATR rally or a new 27-bar high WITH ADX >= 40 — so names that
bottomed and recovered stay short through the next bull (16.8 shorts held in up months on
TRAIN). A time clock removed those but also the decliners still falling (2022). A Donchian exit
separates the two by each name's own trend: a recovered name makes a new 27-bar high and leaves;
a name still falling never does and stays. The short side then shrinks as names recover, i.e.
after the bear, which is what "adaptive" needs.

**Expected.** Fewer shorts in rising stretches (TUNED 16.8 on TRAIN up months → ~8-12), more
turnover than TUNED (but far less than iteration 1), the 2022 hedge kept. Alpha: within ±3 of
TUNED; CAGR up in T3 and 2023-24. Adoption: paired Δ >= 0 on TB, T3, T1, T2 and T1 Δ > 1.1
points (2 x U0's noise sd 0.56; TUNED's own is 0.31). Variant **5**.

### Iteration 2 — result (variant 5): REJECTED

`lab.py run adaptive_hedge I2 '{TUNED + "exit_window": 27}' TB,T3,T1,T2 --costs2`

| test | TUNED alpha | I2 alpha | paired Δ (P) | better | I2 beta | I2 CAGR | I2 long / short |
|---|---|---|---|---|---|---|---|
| TB | +7.5% | −8.2% | −15.6 (0.01) | 0/4 | +0.02 | −8.0% | 0.64 / −0.28 |
| T3 | +6.2% | −2.2% | −8.4 (0.17) | 1/3 | +0.75 | +25.6% | 0.75 / −0.18 |
| T1 | +12.9% | +6.7% | −6.2 (0.12) | 3/10 | +0.09 | +7.4% | 0.55 / −0.35 |
| T2 | +9.4% | +5.0% | −4.4 (0.19) | 3/10 | +0.02 | +5.1% | 0.55 / −0.34 |

Stitched TRAIN Δ −7.5 (P 0.13); orders ~2x TUNED's. 2022 −5.8% vs −19.4% (TUNED +28.0%); bear
episodes Q4 2018 −7.2%, COVID +10.7%, 2022 bear −3.6%, Feb-Apr 2025 −5.9%.

Read: the Turtle exit also makes the book adaptive (short side −0.18 in the T3 bull) and also
gives the alpha back. The names TUNED shorts (decisive 27-bar breakdowns, most negative 12-1
relative strength first) are persistent losers: they rally into new 27-bar highs inside the
decline (bear-market rallies) and then fall further, so covering them on the rally and
re-shorting at the next low is a whipsaw. Two different "shorten the holds" exits (a clock, a
trend reversal) both cost 4-16 points of alpha on every test. On this data **the short side
earns by holding broken stocks for about a year**, which is the short leg of 12-month momentum
with a breakdown trigger, not a hedge that switches on and off.

## 6. Iteration 3 — pre-registration (before the run): variant B, the proposal's own

**Change.** On top of TUNED: `short_filter: crowded` — a start is shorted only if the name sits in
the crowded half by days-to-cover (long rank <= 0.5; names without a rank, i.e. no short-interest
data or no share count, are not shorted).

**Mechanism.** It joins the repo's two independent short signals: short sellers' crowding
(informed_flow's ranking, read in reverse) and a decisive breakdown. A breakdown that informed
short sellers are already positioned for should continue; one they ignore is more likely noise.
TRAIN premise on TUNED's starts (`premise-TUNED.json`): crowded half −0.4% (21 bars) / +0.1%
(63 bars) against the market vs the other half +0.2% / +6.6% — the right sign, t < 1.

**Expected.** Fewer eligible starts, so the cap binds less and the book is a little longer;
borrow slightly dearer (crowded names sit in higher tiers); alpha +0 to +2 over TUNED if the
premise sign holds, but squeeze risk is higher (crowded names). Same adoption bar (Δ >= 0 on
TB, T3, T1, T2; T1 Δ > 1.1). Variant **6**.

### Iteration 3 (variant B) — result (variant 6): REJECTED

`lab.py run adaptive_hedge I3_B '{TUNED + "short_filter": "crowded"}' TB,T3,T1,T2 --costs2`

| test | TUNED alpha | B alpha | paired Δ (P) | better | B beta | B CAGR |
|---|---|---|---|---|---|---|
| TB | +7.5% | +0.4% | −7.1 (0.15) | 0/4 | −0.17 | −1.4% |
| T3 | +6.2% | +1.0% | −5.2 (0.18) | 1/3 | +0.46 | +18.1% |
| T1 | +12.9% | +9.4% | −3.5 (0.12) | 2/10 | −0.18 | +7.9% |
| T2 | +9.4% | +4.4% | −5.0 (0.06) | 2/10 | −0.22 | +2.7% |

Stitched TRAIN Δ −1.7 (P 0.31). B is positive in all four bear episodes (Q4 2018 +0.2%, COVID
+7.2%, 2022 bear +26.4%, Feb-Apr 2025 +1.0%) but worse everywhere on alpha: limiting the shorts
to crowded names loses the uncrowded persistent losers that carried the short leg, and crowded
names carry more squeeze risk. The repo's two short signals do not compound here.

## 7. Leg attribution of TUNED (`legs.py TUNED`; no new variant)

Each leg's daily contribution = weight at the previous close x the name's return, annualized
arithmetically; alpha = contribution − beta x the equal-weight market.

| test | long contrib / beta / alpha | short contrib / beta / alpha | gross long / short |
|---|---|---|---|
| TB | +4.7% / 0.25 / +1.3% | +1.1% / −0.46 / **+7.3%** | 0.32 / 0.58 |
| T3 | +13.9% / 0.54 / −4.0% | −7.9% / −0.55 / **+10.2%** | 0.47 / 0.44 |
| T1 | +8.5% / 0.38 / +4.7% | +3.6% / −0.60 / **+9.6%** | 0.37 / 0.55 |
| T2 | +10.2% / 0.33 / +7.1% | −1.4% / −0.56 / **+3.9%** | 0.38 / 0.56 |

Calendar years, book vs buy-and-hold (T1 names): 2022 +28.0% vs −19.4% (short leg +24.5 points),
2023 +3.5% vs +23.9%, 2024 +3.9% vs +16.4%, 2025 +23.2% vs +15.7%, 2026 −2.5% vs +10.2% [later:
the holiday-row artifact; lab v2: −4.8% vs +11.7%]. TRAIN:
2019 −1.3% vs +34.8%, 2020-H2 −8.1% vs +29.6%, 2021 +18.7% vs +24.1%. T2: 2023 −21.3% vs +22.2%
(short leg −25.6 points: the 2023 rebound of 2022's losers).

Read: the short leg beats an equal-beta index short on every test (+4 to +10 a year): the
single-name breakdown shorts select losers, so a SPY short of the same size would lose that
selection (and, at SPY +12.1% vs the equal-weight +8.6% on T1, ~2 more points). The SPY
iteration was therefore not run: its outcome is predictable from this table. What the book
cannot do is keep up in a rising year: 55% short at beta −0.56 is a crisis-alpha profile
(trend_following's), not the proposal's "nearly fully long in a rising market".

## 8. Iteration 4 — pre-registration (before code): the proposal's +50% squeeze stop

**Change.** On top of TUNED: `short_stop: 0.5` — a short whose close rises 50% above its entry
close is bought back. This is in the proposal's own setup (STRATEGY_OVERVIEW §6: "a +50% squeeze
stop"); U0 left it out because the task text did not list it.

**Mechanism.** The worst short-leg stretches are squeezes of broken names: T2 2023 (−25.6 points
of the book), T3 2020-H2 (−17.2), T1 2026-H2. The chandelier sits 16 x ATR(48) above the low —
~40% for a 2.5%-ATR name but ~80% for a 5%-ATR one — so a volatile short can more than double
before it is covered. A cap on the loss from entry bounds one squeeze.

**Expected.** Small, and the prior is negative: informed_flow_ls v6 found a 50% stop covered at
the squeeze's top (14 stopped shorts lost 29% each, then fell 9%) and 0.75 / 1.0 / none all beat
it. Expected Δ −1 to +1 on T1; a gain on T2 (smaller, more volatile names) is the only plausible
win. Same bar. Variant **7**.

### Iteration 4 — result (variant 7): REJECTED

`lab.py run adaptive_hedge I4 '{TUNED + "short_stop": 0.5}' TB,T3,T1,T2 --costs2`

| test | TUNED alpha | I4 alpha | paired Δ (P) | better |
|---|---|---|---|---|
| TB | +7.5% | +7.9% | +0.4 (0.64) | 1/4 |
| T3 | +6.2% | +4.9% | −1.3 (0.16) | 1/3 |
| T1 | +12.9% | +9.6% | −3.4 (0.025) | 3/10 |
| T2 | +9.4% | +6.9% | −2.5 (0.08) | 4/10 |

As the informed_flow_ls prior said: the stop covers at the top of the squeeze and the name then
falls again; it loses on T1 and T2 (where volatile names were expected to make it pay). The bear
episodes are unchanged (2022 bear +30.1% either way).

Correlations (daily returns, `lab` results): TUNED vs informed_flow_ls_p1/v6_si TB −0.08, T1
+0.51 (monthly −0.04 / +0.41); vs the equal-weight market TB −0.53, T1 −0.25; vs its own long
sleeve L0 TB −0.52, T1 +0.15. v6_si on the same panels: TB alpha −2.6% (beta 0.07), T1 +13.1%
(beta 0.17).

## 9. Final candidate — declared 2026-10-23 before the verdict, the bear table and T4

**FINAL = TUNED** = U0 + `{max_shorts: 20, adx_min: 40, entry_window: 27, atr_mult: 16,
atr_window: 48}` (lab label `TUNED`). The only adopted change; iterations 1-4 (short clock,
Turtle exit, variant B, squeeze stop) were all rejected. Variants through the bar: **7** (U0,
TUNED, control L0, I1, I2, I3_B, I4), within the budget of 8.

Luck (T1, 100 shuffled-name books, `lab.py luck adaptive_hedge TUNED`): real +12.9%, median
−0.2%, 95th percentile +4.3%, **percentile 100**. Noise sd TB 1.58, T3 0.61, T1 0.31, T2 0.31
points; dropping 5 random names a period cost T1 alpha 1.9-2.6 points in all three copies (some
name concentration).

Next, in this order and once: `lab.py verdict adaptive_hedge TUNED --not-neutral`, `lab.py bear
adaptive_hedge U0 TUNED L0`, then T4 for TUNED and U0 (`lab.py run ... T4 --costs2`).

## 10. Verdict, bear table and the one T4 read (TUNED; 2026-10-23)

### Lab gates (`lab.py verdict adaptive_hedge TUNED --not-neutral`)

| gate | value | passed |
|---|---|---|
| G1 TRAIN-era (TB + T3) alpha > 0 | +9.1% | yes |
| G2 T1 alpha > 0, P >= 0.90 | +12.9% (P 0.986) | yes |
| G3 T2 alpha > 0 | +9.4% | yes |
| G4 T1 beats >= 95% of shuffled books | percentile 100 | yes |
| G5 T1 late-fill and 2x-cost alpha > 0 | +11.4% / +12.5% | yes |
| G6 every bear episode + down months > 0 | down months +2.28% (market −4.10%) | yes |
| G7 | not market-neutral: skipped (beta −0.21) | — |
| G8 T1 alpha without its top 5 names | +7.4% | yes |

### Bear table (`lab.py bear adaptive_hedge U0 TUNED L0`; TB + T3 + T1 stitched)

| | Q4 2018 | COVID crash | 2022 bear | Feb-Apr 2025 | CAGR 2018-07 → 2026-10 | max DD |
|---|---|---|---|---|---|---|
| TUNED | −1.5% | +20.6% | +30.1% | −1.2% | +8.7% | −17.9% |
| U0 | +1.5% | +16.8% | +18.4% | +2.2% | −0.5% | −27.1% |
| L0 (long side alone) | −24.7% | −35.7% | −20.3% | −22.7% | +18.6% | −36.2% |
| equal-weight market | −18.2% | −37.0% | −25.2% | −19.6% | +13.8% | −37.4% |
| SPY | −18.7% | −33.4% | −24.1% | −18.8% | +15.1% | |

Calendar years, TUNED vs the equal-weight market: 2018-H2 −0.8 / −6.8, 2019 −1.3 / +34.8, 2020
+4.0 / +26.0, 2021 +18.7 / +24.1, **2022 +28.0 / −19.4**, 2023 +3.5 / +23.9, 2024 +3.9 / +16.4,
2025 +23.2 / +15.7, 2026 −2.5 / +10.2.

### The proposal's own bars (§0)

- P1 hedge cost <= 2 points vs L0: **T1 pass** (+12.9% vs +9.5%: the short side adds 3.4);
  T2 fails (+9.4% vs +15.8%).
- P2 max DD <= 2/3 of buy-and-hold's: TB pass (0.25), stitched 2018-26 pass (0.48), **T1 fails
  narrowly (0.69)**, T2 fails (0.99).
- P3 2022: net exposure −12% on 2022-06-30 (pass); 2022 +28.0% vs −19.4% (+47 points, pass).
- P4 2018-20 bear table: Q4 2018 −1.5% vs −18.2%, COVID +20.6% vs −37.0% (pass).
- The overview's aim, "beat buy-and-hold in both regimes": **fails** — it wins only in falling
  years and 2025; over 2018-07 → 2026-10 it compounds 5 points a year below the equal-weight
  market and 6.4 below SPY.

### T4 (names 1001-1500, locked; read once, after the declaration)

`lab.py run adaptive_hedge TUNED '<final>' T4 --costs2` and `... U0 '{}' T4 --costs2`:

| | alpha | P | beta | CAGR | B&H | SPY | max DD (B&H) | late | 2x | top-5 out | orders/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TUNED | **+4.1%** | 0.82 | −0.30 | +1.2% | +9.8% | +12.1% | −20.2% (−25.6%) | +2.4% | +3.7% | +1.1% | 166 |
| U0 | +3.6% | 0.83 | −0.31 | +0.5% | +9.8% | +12.1% | −15.0% (−25.6%) | +0.9% | +3.0% | +0.6% | 217 |

TUNED − U0 on T4: **+0.6 (P 0.55)**, 7/10 periods better. Years (TUNED vs B&H): 2022 +4.5 /
−13.3, 2023 −8.1 / +24.9, 2024 −7.6 / +17.9, 2025 +16.2 / +10.4, 2026 +2.8 / +11.2.

## 11. Conclusion

- **By the lab's gates TUNED "works"** (G1-G6 and G8 all pass on the open tests; G4 at the
  100th percentile). On the never-read names 1001-1500 the alpha is still positive but a third
  of T1's (+4.1%, P 0.82, below G2's 0.90 bar), and the tune's gain over U0 almost vanishes there
  (+0.6, P 0.55): most of the 8.6-point T1 gain was fitted to the top 1000 names.
- **The proposal's mechanism failed.** Breakdown STARTS are a breadth signal (13x more frequent in
  down months), but the short side never sizes itself: the trend exits are so rare that the
  20 short slots stay ~full in every regime (TRAIN: 16.8 shorts in up months vs 19.5 in down
  months). Every attempt to make it adaptive — a one-month clock, a Turtle trend-reversal exit,
  a squeeze stop — cost 3-16 points of alpha on every test, because the shorts earn by being
  HELD for about a year through the bear-market rallies.
- What the book really is: a near-constant ~−15% net short made of persistent losers (the short
  leg of 12-month momentum with a breakdown trigger) plus informed_flow's long sleeve at ~40%
  gross. A crisis-alpha profile, like trend_following: it beats the market in every falling
  episode (2022: +47 points) and loses 10-35 points to it in every strong bull year [later: a
  slip: 2019 −36.1, 2021 −5.4; on lab v2 it loses 4-37 points in 2019-21, 2023-24 and 2026, §13.6].
- Its alpha is mostly the short leg's selection (+4 to +10 points a year over an equal-beta
  index short on all four open tests), not timing: the shuffled-name books with the same
  exposure path have median alpha −0.2%.
- Status for the cross-strategy round: **works to some degree** — a crisis hedge with positive
  alpha on every test (TB and T3, which the tune was fitted on, and out of sample T1 +12.9%, T2
  +9.4%, T4 +4.1%), but not the proposal's adaptive hedge, below buy-and-hold and SPY on CAGR
  (T1 +11.1% vs SPY +12.1%; T4 +1.2% vs +9.8% / +12.1%), and weak on T4.

## 12. Review fixes (2026-10-23; the fix agent, after an adversarial review)

### 12.1 Pre-registration (written before any re-run number)

An adversarial review reproduced every saved label bit for bit and found one critical and two
major defects (its scripts and logs: `var/proposals_lab/results/adaptive_hedge/review/`).

1. **Critical: market-holiday rows wiped the whole book.** The 2026 lab panels hold three US
   holidays (2026-06-19, 07-03, 09-07) on which no T1 name is priced (T2: only ALC; T4: only
   CFR, Swiss listings). On such a row `_book()` exited every long ("ineligible": the score is
   NaN) and every short ("no_price", this strategy's own addition: trend_following's `_ride`
   holds through a NaN bar). The next bar's book was empty, and the Donchian band
   (`min_periods=entry_window`) blocked every new short for `entry_window` bars after each such
   row. TUNED's T1/T2/T4 2026-H2 books were long-only plus two full-book round trips.
   **Fix (strategy-local; `strategy.py`):**
   - (a) decisions on **trading days only**: `_live_rows` skips a row on which fewer than half
     of the median count of priced names over the 21 rows before it carry a close (causal).
     `compose()` and `_book()` run on the trading days. The decision of the day before a skipped
     row fills at the next trading day's open, and the book is held unchanged on the skipped row.
   - (b) a held name **without a price on a bar is carried**, not sold: no exit is decided for
     it until it is priced again. The "no_price" short exit and the "ineligible" long exit on a
     missing price are gone. Longs without a price cannot be displaced.
   - (c) a target change due on a bar where the name has **no open waits for its next priced
     open** (`_on_priced_opens`). Measured on the engine: an order at a NaN fill price is
     dropped silently, and since the engine trades a name only when its target changes, it is
     never sent again, so the position (or its absence) sticks. This is now a test.
   - Tests: a holiday row (with and without a Swiss name priced on it) gives the same book on
     every other day and holds it on the holiday; the engine places no order on it and values
     the same book; a held long and a held short are carried through a 3-bar gap; the dropped
     order and its deferral; causality with a holiday row and a gap in the panel. Each new test
     fails with the corresponding part of the fix removed (checked by mutation).
   - **Where it can act** (`fix/scan_panels.py`, every panel's whole history from 2015):
     - skipped rows exist only on the three 2026 dates, i.e. in the 2026-01 and 2026-07
       periods of T1, T2 and T4;
     - single-name price gaps exist only in T1 2025-07 (FISV, one bar) and the 2026 periods
       (EA, EQR, PSKY, LBRDK, DDS; 13 names' data stops on 2026-06-12).
     - TB, T3 and every period before 2025-07 are untouched. The tune (TB + T3) and every
       TRAIN delta must therefore reproduce bit for bit; the re-runs check it.
2. **Major: the lab never counts a period's first bar** (init cash → bar-0 close), and with it
   the opening costs, the first day's P&L and the borrow the engine charges up front on a short
   held to the window's end (most of this book's borrow: its shorts are held ~300 days). The
   overnight gap into each period falls in neither period. This is a lab-wide convention that
   applies to book and benchmark alike, and lab.py is not edited. It is reported as a lab issue.
   Workaround: a strategy-local readout `firstbar.py` re-runs a label's books and restates the
   stitched alpha with each first bar counted for book and benchmark (its open→close P&L and the
   up-front borrow; the restart's trading costs left out, the review's correction "B"), and
   measures the seam gaps.
3. **Major: restated headlines.** T2's alpha and P and the 2026 figures are restated after
   1-2. The "factor crash" reading of 2026-H2 (U0 T1 −54.9%) is withdrawn: it was the
   holiday-row artifact. [later] notes mark the affected lines above.
4. **Minor: the noise yardstick.** All three noise copies of TUNED cut T1 alpha by 1.9-2.6
   points, so their sd (0.31) understates the path noise. Besides the sd, the readout reports
   the mean shift and the RMS of the copies' deltas, and checks whether any adoption decision
   depends on which one sets the bar.

### 12.2 Re-run plan (labels end in `f`; the old labels stay for the record)

- `U0f`, `TUNEDf` (the §9 overrides), `L0f` (control), `I1f`, `I2f`, `I3_Bf`, `I4f`, each
  `lab.py run ... TB,T3,T1,T2 --costs2`. These are the same designs re-measured on the fixed
  code, not new variants: the variant count stays at 7.
- `lab.py noise` and `lab.py luck` (T1) for U0f and TUNEDf.
- The tune is re-run only if TB/T3 do not reproduce bit for bit (it scored TB + T3 alone).
- **Declared before any re-run number:** the final candidate stays the §9 design, run on the
  fixed code as `TUNEDf`, unless a re-measured adoption decision flips. Then `lab.py verdict
  adaptive_hedge TUNEDf --not-neutral`, `lab.py bear adaptive_hedge U0f TUNEDf L0f`, and T4
  **read a second time, for TUNEDf only**. The first read (§10) ran the defective code, and the
  review has already re-run T4 on cleaned panels. Nothing is selected on T4. U0 is not re-read
  on T4: the TUNED − U0 comparison there is quoted from the review's cleaned-panel run.

[later, round 2] §12.2's re-runs (labels ending in `f`, lab v1 with the §12.1 strategy fix) ran
on 2026-10-23 00:47-01:02 but were never written up. Lab v2 supersedes them; §13 quotes them
only to separate the strategy fix's effect from the lab's.

## 13. Round 2: lab v2 re-evaluation

### 13.1 Pre-registration (2026-10-23 08:06 CEST, before any lab v2 number of this strategy)

**Why.** Lab v2 (`strategies/proposals_lab/README.md`, "Lab v2") fixes five lab-level defects
that biased every round-1 number: (1) the panels now hold NYSE sessions only; (2) each period's
day 0 is counted from the initial cash, with the opening costs and the borrow the engine charges
up front; (3) short covers execute before new shorts (`lab.run_portfolio`); (4) the borrow on
hair-sized short re-trims is deferred (`lab.borrow_fees`); (5) six sector ETFs typed "equity" are
excluded. Every number in §2-§12 is superseded. The strategy code keeps the §12.1 fix: a held
name is carried through a missing price, rows on which almost no name is priced are skipped, and
a target change waits for the next priced open. On the v2 panels no row is skipped and no row is
empty in any open-test period (`results2/adaptive_hedge/scan_v2.py`). The carry acts only on
FISV (one bar, from T1 2025-07 on), EA (T1 2026-07), EQR and PSKY (T2 2026-07). TB and T3 have no
gap at all.

**Runs** (labels end in `v2`; `lab.py run ... TB,T3,T1,T2 --costs2`; E0 not run, as in round 1):

1. `U0v2` (`{}`), `TUNEDv2` (the §9 overrides), `L0v2` (control, `max_shorts: 0`).
2. Every round-1 iteration re-measured, with the same definitions on top of TUNED: `I1v2`
   (`short_hold: 21`), `I2v2` (`exit_window: 27`), `I3_Bv2` (`short_filter: crowded`), `I4v2`
   (`short_stop: 0.5`). These are re-measurements, not new variants, so the count stays at 7.
3. `lab.py noise` and `lab.py luck` (T1) for U0v2 and TUNEDv2.
4. New: `NCv2`, a control (variant 8): TUNED with `atr_mult: 1000`, so the chandelier never
   fires (the review: "test no chandelier explicitly"; TUNED's 16 sits at the search bound).
   It is not a candidate. Up to two more iterations (variants 9-10) are allowed, only if the v2
   numbers open an obvious direction. Each is pre-registered here before its run.
5. Diagnostics that select nothing:
   - (a) The v1 tune's best trial, its 19 neighbours (the review's definition) and its top 10,
     re-scored on v2 TRAIN (TB + T3, the tune's robust objective). This gives the
     neighbourhood-averaged score on corrected accounting.
   - (b) A borrow stress, `TUNEDv2_b3`: TUNED with `borrow_gc: 0.03` on T1 and T2. Every short
     below the 10% tier, and every short without short-interest data, then pays 3% instead of
     0.5%.
   - (c) The share of TUNEDv2's short exposure that has no `si_share` data (the review: such
     names are shortable at GC borrow).

**Decision rules** (fixed now):

- **R1, the tune** (round 1's only adopted change) stands iff TUNEDv2 − U0v2 has paired Δ > 0
  on T1 AND on T2. Otherwise the final is U0v2.
- **R2, the iterations** are judged against TUNEDv2 with round 1's bar: paired Δ ≥ 0 on TB, T3,
  T1 AND T2, and T1 Δ > 2 × max(noise sd of U0v2, TUNEDv2).
  - Also reported: the same bar with the RMS of the noise copies' deltas in place of their sd
    (review minor). A decision that depends on which yardstick is used is flagged.
  - Order: round 1's (I1, I2, I3_B, I4). The first that passes becomes the base. A later
    iteration that also passed must be re-run on top of that base (a new variant) before it can
    be adopted.
- **R3, the final** = TUNEDv2, unless R1 or R2 changes it. A new final is declared here before
  its verdict, its bear table and its T4 read.
- **R4, the status** follows the lab's definition ("works" = every applicable gate of
  `lab.py verdict --not-neutral`). Any departure is stated as an explicit override, with its
  reason.
- **T4:** the final and U0v2 are read once each, after the verdict and the bear table. This is
  a disclosed re-measurement of the §10 read on corrected accounting, and it selects nothing:
  no decision above may change after it.

**Expected** (from the review's corrected numbers, `review/corrected.log`):

- TUNED: TRAIN-era within ±1.5 points of +9.1%; T1 ≈ +13 ± 1.5; T2 ≈ +6 ± 1.5; T4 ≈ +4.4.
- TUNED − U0 ≈ +6.8 on T1 and ≈ +3.8 on T2, so R1 should stand.
- The iterations should all stay rejected: they failed on TB and T3, which hold no holiday row.
  The cover-first order should help I1 and I2 most (2-2.6x TUNED's short turnover), but not by
  4-16 points.
- The re-trim borrow fix should have no effect: this book never re-sizes a held short, and
  `retrim_k` caps longs only.
- NCv2: within ±3 points of TUNED, sign unknown.
- The gates should pass as in round 1.

The time and SHA-256 of this subsection are written to `results2/adaptive_hedge/prereg.log`
before the first v2 run (review minor: a verifiable order).

### 13.2 The re-checks on lab v2 (runs 08:07-08:34; `results2/adaptive_hedge/run_v2.log`)

| test | U0v2 alpha (P) | TUNEDv2 alpha (P) | TUNEDv2 beta | TUNEDv2 CAGR | B&H | SPY | L0v2 alpha (P), beta |
|---|---|---|---|---|---|---|---|
| TB | +2.4% (0.68) | +1.9% (0.66) | −0.24 | −0.7% | +11.2% | +8.8% | −6.9% (0.07), 1.01 |
| T3 | −3.6% (0.32) | +5.2% (0.71) | +0.00 | +5.2% | +35.2% | +35.3% | −5.4% (0.29), 1.20 |
| TRAIN era (TB + T3) | −0.8% (0.43) | +5.7% (0.82) | −0.19 | +1.8% | +20.9% | | −4.7% (0.13), 1.05 |
| T1 | +6.7% (0.93) | **+11.4% (0.985)** | −0.28 | +9.0% | +8.8% | +11.9% | +8.4% (0.93), 1.15 |
| T2 | +0.1% (0.54) | **+4.3% (0.77)** | −0.26 | +2.3% | +7.6% | +11.9% | +14.2% (0.997), 0.93 |

**R1 (the tune): stands.** TUNEDv2 − U0v2 is +4.7 on T1 (P 0.90, 7/10 periods better) and +4.1
on T2 (P 0.79, 5/10). It is also +8.7 on T3 (P 0.89), but **−0.5 on TB (P 0.47)** as run.
Stitched TRAIN: +6.5 (P 0.89; v1 +9.2).
[repair, §14: the first version called the TB figure "corrected accounting", under which "the
tune no longer beats U0 on half of the data it was fitted to". It is not. Lab v2's runner refused
TUNEDv2's chandelier cover of Wayfair in TB 2020-H1 and never re-sent it (§13.4 e).
- With every stuck position's P&L removed from both books, TUNEDv2 − U0v2 is +2.9 on TB
  (P 0.70) and +8.4 on the stitched TRAIN (P 0.96).
- On T1 it is +4.4 (P 0.88) and on T2 +4.0 (P 0.78). R1 stands either way.]

**R2 (the iterations): every one stays rejected.** Paired Δ against TUNEDv2:

| | TB | T3 | T1 | T2 | orders/yr T1 | 2022 bear | v1 T1 / T2 Δ |
|---|---|---|---|---|---|---|---|
| I1v2 short clock | +4.2 (0.50), 1/4 | −11.7 (0.14) | −6.3 (0.19) | −2.9 (0.33) | 399 | −26.9% | −7.4 / −6.1 |
| I2v2 Turtle exit | −9.9 (0.09) | −9.3 (0.16) | −4.7 (0.21) | −0.9 (0.43) | 297 | −4.6% | −6.2 / −4.4 |
| I3_Bv2 variant B | −0.5 (0.48) | −3.8 (0.25) | −2.2 (0.27) | −0.1 (0.50) | 125 | +24.1% | −3.5 / −5.0 |
| I4v2 squeeze stop | +4.2 (0.92), 1/4 | −1.2 (0.16) | −3.2 (0.03) | −0.5 (0.35) | 116 | +25.9% | −3.4 / −2.5 |
| NCv2 no chandelier (control) | +7.9 (0.91) | −6.1 (0.24) | −3.8 (0.17) | −3.1 (0.21) | 107 | +5.0% | — |

(TUNEDv2: 116 orders a year on T1; 2022 bear +25.9%.)

- **Noise** (5 names dropped a period, 3 copies). The sd of the T1 copies is 0.42 for U0v2 and
  1.08 for TUNEDv2; their RMS is 0.71 and 1.21. The T1 bar is therefore 2.2 points (sd) or 2.4
  (RMS). Every iteration fails on sign first (T1 Δ < 0 and T3 Δ < 0), so no decision depends on
  the yardstick. Other tests' sd for TUNEDv2: TB 1.60, T3 0.85, T2 0.28.
- **The closest rejected changes** are variant B (I3_B) and I4.
  - I3_B is behind on every test. Its T2 gap closed from −5.0 to −0.1.
  - I4 is +4.2 on TB, from one period (it covers Wayfair, §13.4 e). It is behind on T1 with
    P 0.03.
    - Per period it is behind in 7, identical in 2 (2022-H1 and H2) and ahead in 1 (2024-H1).
      [repair: this said "in 9 of 10 periods".]
    - Its TB edge is mostly TUNEDv2's refused W cover: with both books' stuck P&L removed it is
      +0.9 (P 0.68), §14.
- **The no-chandelier control** answers the review's question about the bound. The 16 x ATR(48)
  stop rarely fires, but it is what recycles the 20 short slots. Without it, stale shorts from
  2021 hold the slots, the 2022 breakdowns cannot enter, and the 2022 bear falls from +25.9% to
  +5.0%. The stop costs on TB (+7.9 without it, the 2020 rebound) and pays on T3, T1 and T2. A
  wider bound would move toward the control, which is worse out of sample.
  [later, after §13.4: the mechanism above is wrong in its detail.
  - Both books held 20 shorts on 2022-01-03, and both took only 2 new shorts in the 2022 bear.
  - What differs is WHICH names held the slots. TUNED's 20 had a median age of 172 days
    (oldest 306): 2021 breakdowns such as PYPL, ROKU, PINS, FSLY and RNG. They fell −34.4% in
    2022-H1, against −21.9% for the market.
  - NCv2's 20 had a median age of 840 days (oldest 2,365): APA, DVN, PSX, MRK, O, VTR and
    others. They fell only −15.0%.
  - The stop keeps the short side fresh: it moves slots out of names that recovered and into
    the next breakdowns.
  - On TB the control's gain is one name: its never-recycled slots had no room for the Wayfair
    short of 2020-H1 (§13.4 e).]
  [repair, §14: part of that gain is TUNEDv2's refused W cover. With both books' stuck P&L
  removed, the control's TB edge is +4.5 (P 0.82), not +7.9.]
- **U0v2 luck** (T1, 100 shuffled-name books): real +6.7%, median −0.8%, 95th percentile +4.5%,
  percentile 98.

No adoption decision changes. The variant count is 8: U0, TUNED, L0, I1, I2, I3_B, I4 and the
NCv2 control. No new iteration is run. v2 opened no obvious direction: every iteration still
loses on T1 and T3, and the one new loss (TB 2020-H1, §13.4) is a single TRAIN episode. Fitting a
rule to it would be fitting TRAIN. [repair, §14: about half of that loss is a cover the lab
refused, not a property of the strategy.]

### 13.3 Final candidate: declared 2026-10-23 08:35 CEST, before the verdict, the bear table and T4

**FINAL = TUNEDv2** = U0 + `{max_shorts: 20, adx_min: 40, entry_window: 27, atr_mult: 16,
atr_window: 48}`, unchanged from §9 and re-measured on lab v2. Next, in this order and once:
`lab.py verdict adaptive_hedge TUNEDv2 --not-neutral`, `lab.py bear adaptive_hedge U0v2 TUNEDv2
L0v2`, the §13.1 diagnostics, then T4 for TUNEDv2 and U0v2 (`lab.py run ... T4 --costs2`).

### 13.4 Verdict, bear table and diagnostics (lab v2, read before T4)

**Gates** (`lab.py verdict adaptive_hedge TUNEDv2 --not-neutral`; `results2/.../verdict_bear.log`):

| gate | v1 (§10) | v2 | passed |
|---|---|---|---|
| G1 TRAIN-era (TB + T3) alpha > 0 | +9.1% | +5.7% | yes |
| G2 T1 alpha > 0, P ≥ 0.90 | +12.9% (0.986) | +11.4% (0.985) | yes |
| G3 T2 alpha > 0 | +9.4% | +4.3% | yes |
| G4 T1 beats ≥ 95% of shuffled books | 100 | 100 (median +0.3%, p95 +6.3%) | yes |
| G5 T1 late fills / 2x costs | +11.4% / +12.5% | +9.4% / +10.9% | yes |
| G6 every bear episode + down months | +2.28% vs −4.10% | +2.69% vs −4.12% (82% > 0) | yes |
| G7 \|beta\| ≤ 0.3 | skipped (−0.21) | skipped (−0.28) | — |
| G8 T1 without its top 5 names | +7.4% | +5.9% | yes |

U0v2 fails G1 (TRAIN era −0.8%) and passes the rest (G4 at the 98th percentile).
[repair, §14: TUNEDv2's G1 includes one refused cover. With the stuck P&L removed it is +7.6%
(P 0.88). U0v2's G1 is −0.8% either way.]

**Bear table** (`lab.py bear adaptive_hedge U0v2 TUNEDv2 L0v2 NCv2`; TB + T3 + T1 stitched):

| | Q4 2018 | COVID crash | 2022 bear | Feb-Apr 2025 | CAGR 2018-07 → 2026-10 | max DD |
|---|---|---|---|---|---|---|
| TUNEDv2 | −1.8% | +20.7% | +25.9% | +0.3% | +5.9% | −23.4% |
| U0v2 | −0.1% | +17.9% | +16.0% | +0.8% | +0.3% | −27.8% |
| L0v2 (long side alone) | −24.7% | −35.7% | −19.9% | −22.0% | +17.9% | −36.2% |
| NCv2 (no chandelier) | +0.2% | +22.5% | +5.0% | −0.6% | +3.6% | −19.8% |
| equal-weight market | −18.2% | −37.0% | −24.5% | −19.5% | +13.8% | −37.4% |
| SPY | −18.7% | −33.4% | −24.1% | −18.6% | +15.0% | |

Calendar years, TUNEDv2 / market: 2018-H2 −1.3 / −6.2, 2019 −1.3 / +35.6, 2020 −7.3 / +25.7,
2021 +17.8 / +21.5, **2022 +23.9 / −18.6**, 2023 +1.8 / +22.7, 2024 +0.8 / +15.3, 2025 +24.8 /
+16.2, 2026 −4.8 / +11.7. It loses 4-37 points to the market in 2019, 2020, 2021, 2023, 2024 and
2026, and wins only 2018-H2, 2022 and 2025 (+24.8% against +16.2%). [repair: this said "in every
year the market rose more than 11%", which 2025 contradicts.]

**The proposal's own bars (§0) on v2:**

- P1, the hedge costs at most 2 points against L0: T1 passes (+11.4% vs +8.4%; the short side
  adds 3.0). T2 fails (+4.3% vs +14.2%; it costs 9.9).
- P2, max DD ≤ 2/3 of buy-and-hold's: passes on TB (0.50) and on the stitched 2018-26 record
  (0.63). Fails on T1 (0.71, −17.9% / −25.0%; [repair: was 0.72]) and T2 (1.10).
- P3, 2022: passes. Mean net exposure in 2022-H1 was −19.6%, and the 2022 return was +23.9% vs
  −18.6% for buy-and-hold.
- P4, Q4 2018 and COVID: passes.

**Diagnostics** (§13.1 (a)-(c), plus two that explain the changes; `diag_v2.log`):

- **(a) The neighbourhood on v2 TRAIN** (`rescore_v2.py`): the tuned point (v1 tune trial 104)
  falls from +0.045 to **+0.016** and ranks 4th of 23. Its 19 neighbours average **−0.018** (v1
  −0.010; range −0.078 to +0.018). U0 scores −0.047.
  - Most of the tuned point's TRAIN score was selection. It lost 2.9 points to the corrected
    accounting; its neighbours lost 0.8 on average.
  - The region as a whole still beats U0 on TRAIN.
- **(b) Borrow stress** `TUNEDv2_b3` (3% instead of 0.5% for every low-tier or no-data short):
  T1 +9.9% (P 0.97), T2 +2.9% (P 0.70).
- **(c) Shorts without `si_share`:**
  - Share of short gross: TB 19%, T3 16%, T1 5.6%, T2 6.1%.
  - The names: APA, BKR, XOM and BBWI on TB; PSKY, WBD and BALL on T3 and T1. These are renamed
    tickers and coverage gaps of liquid large caps, so in reality GC names.
  - Charging them 10% instead of 0.5% would cost at most TB 1.15, T3 0.75, T1 0.34 and T2 0.37
    points a year.
  - The code is left as is. The bound sits below every gate's margin, and making such names
    unshortable would drop legitimate shorts of liquid stocks. The 30% crowding cap cannot act on
    them, which is a caveat.
- **(d) The seams** (`seams_v2.py`). Lab v2 counts day 0 but not the overnight gap into it.
  - TUNEDv2's day-0 returns add up to −7.7% over T1's ten periods (benchmark −0.4%) and −5.8% on
    T2 (benchmark +0.2%).
  - That is the rebuild costs and the up-front borrow, plus a real effect: this book's losers
    rally on the first session of January and of July.
  - The overnight gaps into those sessions are counted by neither version. They cost the book
    another 3.6% on T1 (benchmark +1.6%).
  - Counting them would move T1 from +11.42 to +10.68 (P 0.97), T2 from +4.27 to +4.14, TB from
    +1.93 to +1.50 and T3 from +5.18 to +5.60. No gate changes.
- **(e) Why TB fell from +7.5 to +1.9: one short, Wayfair (W) in 2020-H1, for two reasons.**
  [repair, §14: the first version named only the first reason. It also said "filled at ~$33" and
  "v2 held the whole slot ... as W rose to ~$100 before the chandelier fired; W's P&L −13.3%".]
  The short was decided at the 2020-03-11 close ($45.01). It filled at the 2020-03-12 open, $41.81;
  that day's close was $33.00 (`starved.py`, `starved_detail.py`, `reconcile.py`).
  - **Lab v1:** the engine's call order filled 31% of the short's target. Its chandelier cover
    filled as decided, at the 2020-04-27 open ($118.36). W cost the book −2.2%.
  - **Lab v2, reason 1:** the cover-first order filled the short in full (101% of its target).
    The chandelier fired at the 2020-04-24 close, at $122.41 (2.9x the fill). By then W's weight
    had grown from −2.5% to −9.9%, and it had cost the book −6.9%.
  - **Lab v2, reason 2:** the runner refused the cover on 2020-04-27 (Rejected / NoCashLong).
    - Free cash was 0.00 after the 2020-04-14 long buys; FNF's long buy was refused the same day.
    - vectorbt lets a cover spend the short's locked proceeds plus an equal margin, 2x the entry
      value, and W stood at 2.8x.
    - Targets are sent only on change, so the cover was never sent again. W stayed short 46 more
      bars to 2020-06-30 ($197.61); its weight reached −17.3%. That cost another −6.7% of the book,
      counted from the due open.
  - In total W cost −13.3% of the book on v2. The strategy's own exit would have held it to about
    −7%.
  - The TB 2020-H1 alpha went from +27.4% (v1) to +0.8% (v2). With the stuck P&L removed it is
    +15.9%.
  - Across TB and T1, the v1 runner part-filled or rejected 0-2 of every 20-33 new shorts in a
    period. v2 still part-filled one new short in TB: AAP on 2020-02-11, at 43% of its target
    (free cash $139). W is the only name that mattered much, on both versions.
  - It also explains two TB deltas.
    - I4's +50% squeeze stop decided W's cover at the 2020-04-06 close ($71.50), and the cover
      filled at the 04-07 open ($76.90). That gives I4 +4.2 on TB on v2, in 1 of 4 periods.
      [repair: this said "covered W on 2020-04-06".]
    - NCv2 never shorted W: its slots, never recycled, were full.
    - Both edges are partly TUNEDv2's refused cover (§14).
- **(f) Leg attribution of TUNEDv2** (`legs.py TUNEDv2`; arithmetic daily contributions,
  alpha = contribution − beta × the equal-weight market):

| test | long contrib / beta / alpha | short contrib / beta / alpha | gross long / short |
|---|---|---|---|
| TB | +4.2% / 0.26 / +0.5% | −4.3% / −0.49 / **+2.8%** | 0.34 / 0.60 |
| T3 | +13.7% / 0.55 / −3.5% | −7.8% / −0.56 / **+9.5%** | 0.49 / 0.45 |
| T1 | +8.7% / 0.34 / +5.2% | +2.7% / −0.62 / **+9.0%** | 0.34 / 0.61 |
| T2 | +9.2% / 0.33 / +6.2% | −4.3% / −0.58 / **+1.2%** | 0.35 / 0.61 |

  The short leg still beats an equal-beta index short on every test, but on T2 by only 1.2
  points (v1 +3.9). In 2023 the short leg cost the T2 book 26.1 points: 2022's losers
  rebounded. The long sleeve carries T2 (+6.2).
  [repair, §14: the stuck covers all sit in the short leg. With their P&L removed the short
  leg's alpha is +6.0 on TB (was +2.8), +8.7 on T1 and +1.0 on T2.]

### 13.5 T4 (names 1001-1500): the one re-measurement (2026-10-23 08:57; `T4_v2.log`)

This is a disclosed second read of the test read in §10, on corrected accounting. Nothing is
selected on it; every decision above was fixed before it ran.

| | alpha | P | beta | CAGR | B&H | SPY | max DD (B&H) | late | 2x | top-5 out | orders/yr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TUNEDv2 | **+4.2%** | 0.84 | −0.35 | +0.6% | +10.2% | +11.9% | −24.7% (−26.1%) | +2.2% | +3.7% | +1.1% | 125 |
| U0v2 | +3.0% | 0.80 | −0.35 | −0.6% | +10.2% | +11.9% | −20.1% (−26.1%) | +0.4% | +2.4% | +0.4% | 164 |
| v1: TUNED / U0 | +4.1% / +3.6% | 0.82 / 0.83 | −0.30 / −0.31 | +1.2% / +0.5% | +9.8% | +12.1% | | | | | |

- TUNEDv2 − U0v2 on T4: **+1.2 (P 0.61)**, with 5 of 10 periods better (v1: +0.6, P 0.55).
- Years, TUNEDv2 / B&H: 2022 +2.1 / −11.5, 2023 −11.4 / +23.9, 2024 −8.1 / +16.6, 2025 +12.9 /
  +11.5, 2026 +9.6 / +11.6.
- The corrected accounting leaves T4 where it was: positive alpha, below G2's confidence bar, and
  a CAGR 10 points below buy-and-hold. The tune adds nothing that clears noise on these names.

### 13.6 Lab v1 → v2: the headline numbers and what moved them

TUNED, the final. The "fix" column is the §12.1 strategy fix alone, on lab v1 (labels ending
in `f`; never written up before; T4 not run).

| | v1 (§3-§10) | strategy fix, lab v1 | lab v2 |
|---|---|---|---|
| TRAIN era (TB + T3) alpha (P) | +9.1% (0.92) | +9.1% (0.92) | **+5.7% (0.82)** |
| · TB / T3 | +7.5% / +6.2% | +7.5% / +6.2% | +1.9% / +5.2% |
| T1 alpha (P) | +12.9% (0.986) | +14.0% (0.996) | **+11.4% (0.985)** |
| T2 alpha (P) | +9.4% (0.948) | +7.3% (0.90) | **+4.3% (0.77)** |
| T4 alpha (P) | +4.1% (0.82) | — | **+4.2% (0.84)** |
| TUNED − U0: T1 / T2 / T4 | +8.6 / +6.0 / +0.6 | +6.5 / +4.3 / — | +4.7 / +4.1 / +1.2 |
| G4 luck percentile (T1) | 100 | 100 | 100 |
| G5 T1 late / 2x costs | +11.4% / +12.5% | +13.1% / +13.8% | +9.4% / +10.9% |
| G6 down months, book / market | +2.28% / −4.10% | +2.64% / −4.10% | +2.69% / −4.12% |
| G8 T1 top-5 out | +7.4% | +8.6% | +5.9% |
| Q4 2018 / COVID / 2022 bear / Feb-Apr 2025 | −1.5 / +20.6 / +30.1 / −1.2 | the same | −1.8 / +20.7 / +25.9 / +0.3 |
| CAGR 2018-07 → 2026-10 (market, SPY) | +8.7% (+13.8%, +15.1%) | +9.0% | +5.9% (+13.8%, +15.0%) |
| T1 orders a year | 157 | 117 | 116 |
| every applicable gate | pass | pass | pass |

U0: TRAIN era −0.0% → −0.8%, T1 +4.4% → +7.5% (fix) → +6.7%, T2 +3.4% → +3.1% → +0.1%, T4 +3.6% →
+3.0%.

[repair, §14] Lab v2's TB and TRAIN-era numbers above include one refused cover (W, §13.4 e).
With every stuck position's P&L removed (`reconcile.py`), TUNEDv2 reads:
- TB +5.3% (P 0.82) and TRAIN era +7.6% (P 0.88);
- T1 +11.1%, T2 +4.1%;
- TUNEDv2 − U0v2 on TB +2.9 (P 0.70).
These are estimates; the lab numbers above are the record.

[repair] Not every number fell from v1 to v2. A few rose slightly: T4 alpha +4.15 → +4.16, the G6
down-month mean +2.28 → +2.69, TUNED − U0 on T4 +0.6 → +1.2, COVID +20.6 → +20.7 and Feb-Apr 2025
−1.2 → +0.3.

**Each review issue → fix → effect:**

1. **Critical, holiday rows wiped the book.**
   - Fix: the lab v2 panels hold NYSE sessions only. The strategy fix of §12.1 carries a held name
     through a missing price, skips rows on which almost no name is priced, and defers a target
     change to the next priced open.
   - Tests:
     - a holiday row, with and without a Swiss name priced: the same book and no order on it;
     - carry through a 1-bar and a 3-bar gap;
     - new: a `_book` unit test in which no exit of either side is decided on an empty bar. It is
       mutation-checked: removing either side's carry, or restoring the old `no_price` exit,
       fails it;
     - the dropped NaN-open order;
     - causality with a holiday row and a gap.
   - Effect, on lab v1: T1 +12.9 → +14.0 and T2 +9.4 → +7.3; TB and T3 stay bit-identical. U0's
     "2026-H2 factor crash" (−54.9%, 246 orders) becomes −23.0% with 60 orders.
   - On the v2 panels the row skip never acts.
   - [repair, §14: this said "the carry acts on 4 names in 2025-26 only", meaning FISV, EA, EQR
     and PSKY. That count missed every name whose data ENDS before a period does. EA, EQR and
     PSKY are not held during their gaps.] For TUNEDv2 the carry acts on:
     - FISV: 1 bar, 2025-11-12;
     - NOW: T1's last bar, 2026-10-16;
     - CWAN and LBRDK, both shorts: the last 11 bars of T2 2026-H1, after their data ends on
       2026-06-12;
     - CWAN again in T2 2026-H2, where it has no price at all. Its opening order is dropped at a
       NaN price, so the strategy counts a short slot the book never holds. That flatters T2 by
       about 0.1 point a year (IDEAS.md 1).
2. **Major, day 0 dropped.**
   - Fix: lab v2 counts day 0 for book and benchmark. `firstbar.py`, round 1's strategy-local
     workaround, is deleted: it would count day 0 twice on v2 results.
   - Effect: together with the cover-first order, lab v2 moves T1 from +14.0 to +11.4, T2 from
     +7.3 to +4.3 and T3 from +6.2 to +5.2.
   - TUNEDv2's day-0 returns add up to −7.7% over T1's 10 periods (benchmark −0.4%) and −5.8% on
     T2 (§13.4 d).
   - Left: the overnight gap into day 0 is still counted nowhere (T1 −0.7, T2 −0.1; [repair: T2
     was given as −0.2, from rounded figures]).
3. **Major, overstated headlines.**
   - Fix: every number in the README and in §13 is restated on v2, and the "factor crash" reading
     is dropped.
   - T2 is now +4.3% with P 0.77, not +9.4% (0.95). The 2026 year is −4.8% vs +11.7%. The long
     sleeve, not the short leg, lost that year (−2.9 vs +0.6 points of contribution).
   - On v2, 2026-H2 is still T1's worst period (alpha −23.4% over 3.5 months, 31 orders). That is
     a real loss, not a book rebuild.
4. **Lab-level, the cover-first order** (another strategy's review, fixed in v2). This is the
   largest single move on this book, and it is one name (§13.4 e).
   - In v1 the engine filled 31% of the W short of 2020-H1, so W cost only −2.2%. That made TB's
     +7.5.
   - In v2 the short is filled in full, and its chandelier cover is then refused and never
     re-sent (§14). TB is +1.9 as run, and +5.3 with the stuck P&L removed.
   - TUNEDv2 − U0v2 on TB is −0.5 (P 0.47) as run, and +2.9 (P 0.70) with it removed.
   - [repair: this said "**the tune no longer beats U0 on TB**". That is true only of the as-run
     number, and it comes from the refusal, not from corrected accounting.]
5. **Minor, the noise yardstick.** The RMS is reported beside the sd: the T1 bar is 2.2 (sd) or
   2.4 (RMS). No decision depends on which.
6. **Minor, an isolated maximum at the bound.**
   - On v2 TRAIN the tuned point ranks 4th of 23. It scores +0.016, against a neighbourhood mean
     of −0.018.
   - The no-chandelier limit (NCv2) is worse on T3, T1 and T2, so a wider bound would not help.
     The stop moves slots from recovered names into fresh breakdowns: 2021's decliners, which
     kept falling in 2022.
7. **Minor, claim slips.** Corrected:
   - §3's "20: −0.055 over 18 trials" should read: 58 trials at `max_shorts` 20 (mean −0.029)
     and 27 at 18 (−0.048).
   - L0's max-drawdown ratio to buy-and-hold is 0.95-1.54 on v1 (T3 1.54), not 0.95-1.08; on v2
     it is 0.97-1.63.
   - "Loses 10-35 points in strong bull years" was wrong (2019 −36.1, 2021 −5.4). On v2 it loses
     4-37 points in 2019, 2020, 2021, 2023, 2024 and 2026, and wins 2018-H2, 2022 and 2025.
   - The T1 short gross was 0.535 on v1 and is 0.616 on v2, as the mean of the periods' mean
     exposure (`_anatomy`). Time-weighted (`legs.py`) it is 0.552 → 0.607. [repair: this paired
     one measure's v1 value with the other's v2 value.]
8. **Minor, the label departs from the lab's rule.** The override is stated explicitly (§13.7).
9. **Minor, pre-registration timing.**
   - Round 2's pre-registrations are time-stamped and SHA-256-hashed into `results2/
     adaptive_hedge/prereg.log`, with frozen copies (`prereg-13.1.md`, `prereg-13.3.md`), before
     the runs they govern. Every run log carries its start time.
   - Nothing is committed to git: the repo's rule is to commit only when the user asks.
10. **Minor, no `si_share` means a GC short.** Measured (§13.4 c): at most 0.34 points a year on
   T1 and 0.37 on T2 at a 10% fee. Under the 3% stress the alpha is still positive. Code
   unchanged; kept as a caveat.

### 13.7 Conclusion of round 2 (and the status override)

- **Every round-1 decision stands on lab v2.** The tune is adopted (it beats U0 on T1 and T2).
  I1-I4 are rejected (all behind TUNEDv2 on T1 and T3). The final is TUNED, now TUNEDv2.
- **By the lab's gates TUNEDv2 "works":** G1-G6 and G8 pass, G7 is skipped, and G4 sits at the
  100th percentile.
- **Most numbers are smaller than in round 1** (a few rose slightly, §13.6).
  - TRAIN era +5.7%, T1 +11.4% (P 0.985), T2 +4.3% (P 0.77), T4 +4.2% (P 0.84).
  - [repair, §14] TB and the TRAIN era include one refused cover. With the stuck P&L removed,
    TB is +5.3% and the TRAIN era +7.6% (P 0.88).
  - CAGR from 2018-07 to 2026-10 is +5.9% a year, against +13.8% for the equal-weight market and
    +15.0% for SPY.
  - Every bear episode is still won: Q4 2018 −1.8% vs −18.2%, COVID +20.7% vs −37.0%, 2022
    +25.9% vs −24.5%, Feb-Apr 2025 +0.3% vs −19.5%.
- **Status: "works to some degree"**, a deliberate downgrade from the lab's "works" (the label
  the gates give). Four reasons:
  1. The proposal's mechanism failed. The short side is ~full in every regime, so this is a
     near-constant net short (T1: 0.34 long, 0.61 short, beta −0.28), a crisis-alpha book, not
     the adaptive hedge the proposal describes.
  2. Out of sample the alpha shrinks with liquidity rank: T1 +11.4, T2 +4.3, T4 +4.2. On the
     locked T4 it sits below G2's confidence bar (P 0.84), and the tune's edge over U0 there is
     +1.2 (P 0.61).
  3. Its CAGR is 5-11 points a year below buy-and-hold and SPY on T2, on T4 and on the stitched
     2018-26 record. It beats buy-and-hold only on T1 (+9.0% vs +8.8%), and never SPY.
  4. The proposal's own bars P1 (T2) and P2 (T1, T2) fail.
  The gates measure alpha, and the alpha is real; the downgrade says that this alpha comes
  bundled with a permanent short that costs more than it pays outside a bear market.
- **For a cross-strategy book**, its use is as a crisis diversifier.
  - Its returns run against the equal-weight market: beta −0.28 on T1, and every bear episode
    is won.
  - Its short leg's selection carries the alpha on the top 500: +2.8 points a year on TB
    (+6.0 with the refused cover's P&L removed, §14), +9.5 on T3 and +9.0 on T1 (legs,
    §13.4 f). On names 501-1000 it carries only +1.2.
  - The short leg (breakdown shorts held ~300 days, the chandelier as the slot recycler) is
    the component; the whole book is not a standalone.
  - The risk is single-name squeezes. One Wayfair slot cost about 7% of the book in 2020-H1
    before its chandelier fired. [repair, §14: this said 13%; on lab v2 the other half came
    after the stop, from a cover the lab refused.]

## 14. Round 2 repair (2026-10-23, after the round-2 verifier)

An independent verifier reproduced round 2 bit for bit: every saved daily series is identical,
the tests pass, and a truncation test on the real T1 2026-07 panel finds no look-ahead. It
found one major and three minor issues (its logs are in
`var/proposals_lab/results2/adaptive_hedge/verify/r2v/`).

Nothing below is a new variant, nothing is selected, and T4 is not read again. The variant count
stays at 8. The repair's own logs are in `results2/adaptive_hedge/repair/`.

Corrections to §13 are made in place and marked [repair]. They leave the two pre-registrations,
§13.1 and §13.3, untouched: both still hash to the SHA-256 values in `prereg.log`.

### 14.1 Major: a refused cover, blamed on something else

**What happened.** The new diagnostic `reconcile.py` rebuilds each period exactly as `lab.run`
does, with vectorbt's order logs switched on. The rebuild matches the saved returns to 0.0 in
every period of the seven labels read (`repair/reconcile_TUNEDv2_U0v2.log`,
`repair/reconcile_iterations.log`).

- TUNEDv2 decided its chandelier cover of W at the 2020-04-24 close. On 2020-04-27 vectorbt
  refused it (Rejected / NoCashLong).
  - Free cash was 0.00 after the 2020-04-14 long buys.
  - Under `lock_cash` a cover may spend the short's locked proceeds plus an equal margin, 2x the
    entry value. W stood at 2.8x.
  - Targets are sent only on change, so the cover was never sent again. W stayed short for 46 more
    bars and cost −6.7% of the book, counted from the due open (§13.4 e).
- The same mechanism hit two smaller covers, both of which made money while stuck:
  - COIN, T1 2024-H2: filled at 31%, then stuck 16 bars, +1.4% of the book;
  - CNK, T2 2023-H1: refused, stuck 16 bars, +0.8%.
- Round 2 put the whole −13.3% down to the short being "fully placed" before the chandelier fired.
  That claim is restated in §13.2, §13.4 e, §13.6 (item 4 and the notes under the table), §13.7,
  the README and the summary.

**The reconciliation, TUNEDv2 over the 27 open-test periods:**
- refused: 2 covers (W, CNK) and 3 long buys (FNF, STT, RGA), each long buy on the day of a stuck
  cover;
- part-filled: 1 cover (COIN), 32 new shorts and 67 long buys;
- dropped at a NaN price: 1 order (CWAN, T2 2026-H2; §14.2);
- time-averaged missing exposure, by test: long 0.75% / 0.80% / 0.62% / 0.53% of the book and
  short 0.38% / 0.27% / 0.29% / 0.49% on TB / T3 / T1 / T2. The T2 short figure includes CWAN.
- U0v2: 1 new short refused (XEL); 1 cover part-filled (MRNA, stuck 1 bar, +0.05%); 82 new shorts
  and 75 long buys part-filled.

**The estimate with every stuck position's P&L removed.** Each exit fills when the strategy decided
it, and the cash it frees stays idle. The lab numbers remain the record.

| | as run (lab v2) | stuck P&L removed |
|---|---|---|
| TUNEDv2 TB | +1.93% (P 0.66) | **+5.30% (P 0.82)** |
| TUNEDv2 T3 | +5.18% (P 0.71) | +5.18% (P 0.71) |
| TUNEDv2 TRAIN era (TB + T3) | +5.72% (P 0.82) | **+7.63% (P 0.88)** |
| TUNEDv2 T1 | +11.42% (P 0.985) | +11.09% (P 0.98) |
| TUNEDv2 T2 | +4.27% (P 0.77) | +4.10% (P 0.76) |
| U0v2 TB / T3 / T1 / T2 | +2.40 / −3.55 / +6.72 / +0.12 | +2.40 / −3.59 / +6.72 / +0.12 |
| TUNEDv2 − U0v2 on TB | −0.47 (P 0.47, 3/4 better) | **+2.90 (P 0.70, 3/4)** |
| · on T3 | +8.74 (P 0.89) | +8.77 (P 0.89) |
| · on T1 | +4.70 (P 0.90, 7/10) | +4.37 (P 0.88, 7/10) |
| · on T2 | +4.15 (P 0.79, 5/10) | +3.98 (P 0.78, 5/10) |
| · on the TRAIN era | +6.53 (P 0.89) | +8.45 (P 0.96) |
| TUNEDv2 2018-07 → 2026-10 (TB + T3 + T1) | CAGR +5.9%, max DD −23.4%, 2020 −7.3% | CAGR +6.6%, max DD −18.1%, 2020 −0.8% |
| TUNEDv2 short-leg alpha (`legs.py`) TB / T1 / T2 | +2.8 / +9.0 / +1.2 | +6.0 / +8.7 / +1.0 |

**R2 re-checked** (iteration − TUNEDv2, paired, with both books' stuck P&L removed; as run →
stuck removed):

| | TB | T3 | T1 | T2 | TRAIN era |
|---|---|---|---|---|---|
| I1v2 short clock | +4.2 → +0.8 | −11.7 → −11.7 | −6.3 → −6.0 | −2.9 → −2.7 | +3.9 → +1.9 |
| I2v2 Turtle exit | −9.9 → −13.2 | −9.3 → −9.3 | −4.7 → −4.4 | −0.9 → −0.7 | −5.1 → −7.0 |
| I3_Bv2 variant B | −0.5 → −3.9 | −3.8 → −3.8 | −2.2 → −1.9 | −0.1 → −0.2 | +2.4 → +0.5 |
| I4v2 squeeze stop | +4.2 → +0.9 | −1.2 → −1.2 | −3.2 → −2.9 (P 0.04) | −0.5 → −0.7 | +2.1 → +0.2 |
| NCv2 no chandelier | +7.9 → +4.5 | −6.1 → −6.1 | −3.8 → −3.5 | −3.1 → −2.9 | −0.7 → −2.6 |

- I3_B and I4 stick only in T2 2023-H1. Both stick on CNK, and both on FSLY, whose cover filled
  31% under I3_B and almost nothing under I4 and then stuck for 62 bars. I1, I2 and NC never
  stick.
- Every iteration still trails TUNEDv2 on T3 and T1, so no R2 decision changes. The TB edges of
  I4 and NC were largely TUNEDv2's stuck cover.

**Decisions: none changes.**
- R1 stands either way: with the stuck P&L removed, T1 is +4.4 and T2 +4.0.
- The gates stand.
  - G1 is +5.7% as run and an estimated +7.6%.
  - G2-G8 read T1, T2 and the episodes. T1 and T2 move by 0.3 or less: T2 goes from +4.27 to
    +4.10, against G3's bar of 0.
  - No stuck bar falls inside a bear episode, and the G6 down-month mean moves from +2.69% to
    +2.66%.
- The final stays TUNEDv2 and the status stays "works to some degree" (the override of §13.7).
- There is no new final, so there is no T4 read. T4 is not reconciled, because it is locked. Its
  +4.2% may carry the same defect, in either direction.

**The lab-level fix is a proposal** (IDEAS.md 2; this folder does not edit
`strategies/proposals_lab/`). Any one of three changes would do:
- re-send a refused or part-filled target on the following bars until the position reaches it,
  for example as a fill-retry fixed point like `bt._retrim_loop`;
- let a cover draw on its own locked proceeds;
- keep a free-cash reserve for covers.

After the fix, the lab should re-quote TB and the TRAIN era of every book that shorts.

### 14.2 Minor: where the missing-price carry acts (restated)

- §13.1, a hashed pre-registration left as written, and §13.6 item 1 named FISV, EA, EQR and
  PSKY. `scan_v2.py` counted only interior gaps, so it missed every name whose data ENDS before
  the period does.
- For TUNEDv2 (the verifier's `gaps_TUNEDv2.log`, confirmed by `reconcile.py`), the carry acts on:
  - FISV: 1 bar, 2025-11-12;
  - NOW: T1's last bar, 2026-10-16;
  - CWAN and LBRDK, both shorts: the last 11 bars of T2 2026-H1, after their data stops on
    2026-06-12;
  - CWAN again, for all 76 bars of T2 2026-H2, where it has no price. Its opening order is dropped
    at a NaN price, so the strategy counts a short slot the book never holds.
- EA, EQR and PSKY are not held during their gaps.
- The dead slot's effect: a typical short slot lost 0.38% of the book in that period (`legs.py`),
  so the dead slot flatters T2 by about 0.1 point a year.
- The optional fix, freeing a carried slot after N unpriced bars, is parked in IDEAS.md 1.
  Implementing it now would change the declared, T4-read final for a 0.1-point effect.
- strategy.py's docstrings now say that a name whose data ends is carried to the end. Behavior is
  unchanged: a rebuild of TB and T2 with the edited file matches the saved returns to 0.0
  (`repair/rebuild_check_after_docstring.log`).

### 14.3 Minor: the deferral and the displacement rule had no signals()-level test

Two new tests:
- `test_signals_moves_a_target_change_on_a_bar_without_an_open_to_the_next_priced_open`: an
  entry, and separately a cover, each due on a bar where the name has no open, happens one bar
  later, and every other name's weights are untouched.
- `test_a_short_start_on_a_full_book_never_displaces_a_long_without_a_price`: N00, the book's
  first long, is unpriced on the breakdown's bar. It is carried, and N03, the weakest priced
  long, leaves.

**The mutation check** (`repair/mutants.log`). The mutants are scratch copies of strategy.py, and
the repo is untouched. Failing tests per mutant:

| mutant | failing tests |
|---|---|
| M0, no change | 0 |
| M1, no long carry | 4 |
| M2, no short carry | 1 |
| M3, the old no_price exit | 3 |
| M4, no deferral in signals() | **1 (was 0)** |
| M5, no row skip | 1 |
| M6, an unpriced long displaceable | **1 (was 0)** |

§12.1's claim, "each new test fails with the corresponding part of the fix removed", now holds for
all three parts. 22 tests pass.

### 14.4 Minor: claim slips, corrected in place and marked [repair]

- §13.2: I4 is behind TUNEDv2 on T1 in 7 periods, identical in 2 (2022-H1 and H2) and ahead in 1
  (2024-H1). It is not "9 of 10".
- §13.4: "loses 4-37 points in every year the market rose more than 11%" was false for 2025: the
  market made +16.2% and the book +24.8%.
- §13.4 P2: the T1 drawdown ratio is 0.71 (−17.87 / −25.02), not 0.72.
- §13.6 item 2: the seam gaps cost T2 −0.1 (+4.27 → +4.14), not −0.2.
- §13.4 e, Wayfair:
  - W filled at the 2020-03-12 open, $41.81; $33.00 was that day's close.
  - v1's fill was 31% of the target. "About a tenth of the slot" had compared a whole-period mean
    weight with a slot.
  - W rose 2.9x, to $122.41, before the stop. It did not rise "about 4x" or stop "at ~$100".
  - I4's squeeze stop decided W's cover at the 2020-04-06 close and filled it at the 04-07 open.
    It did not cover "on 2020-04-06" (found during the repair).
- §13.6 item 7: the T1 short gross is 0.535 → 0.616 as `_anatomy`'s mean of period means, and
  0.552 → 0.607 time-weighted (`legs.py`).
- The headline "every number is smaller" was too strong. A few rose: T4 alpha +4.15 → +4.16, the
  G6 down-month mean +2.28 → +2.69, TUNED − U0 on T4 +0.6 → +1.2, COVID +20.6 → +20.7 and Feb-Apr
  2025 −1.2 → +0.3.
- Trivial: the fix column's T1 P is 0.996, and the TUNEDv2 T1 noise RMS is 1.21.
- A measurement date, not a slip: §13.2 measures the 2022 slot ages on 2022-01-03 (TUNED median
  172 days, oldest 306; NC 840 and 2,365). The verifier's `slots2022.log` measures them on
  2021-12-31, the last decision of 2021, and gets 167 and 302, and 835 and 2,361. The 2022-H1
  returns (−34.4% against −15.0%; market −21.9%) are the same in both.

### 14.5 What changed, and what did not

- **No number in the lab's record changes.** The round-2 tables stand, with the stuck-P&L
  estimates beside them (TB +5.3, TRAIN era +7.6, T1 +11.1, T2 +4.1).
- **No decision changes:** not R1, not R2, not the final, not the status.
- **Three readings change:**
  1. The tune's TB shortfall against U0 (−0.5) came from the refusal, not from corrected
     accounting. With the stuck P&L removed the tune leads by +2.9 (P 0.70).
  2. Under the strategy's own exits, Wayfair's squeeze cost the book about 7%. It cost 13% only
     because the lab could not cover it.
  3. The biggest v1 → v2 move on this book is one name, for two reasons: v1 part-filled the short,
     and v2 placed it in full and then could not cover it.
