# informed_flow_ls — research log

The autonomous improvement loop of 2026-10-20 (user: "test, optimize on meaningful trial counts,
learn from the output, tune and improve step by step"). Protocol: `IDEAS.md`, "Iteration
protocol". Each entry: the hypothesis first, then what the numbers said, then the decision.

## 0 · v1 as pre-registered (nested Optuna walk-forward, 80 trials × 10 periods)

*Hypothesis:* none — this is the pre-registered test of the design built on 2026-10-19.

## 1 · Diagnosis of the untuned v1 book (point-in-time universe, 2020-07 → 2026-10)

`diag.py` (scratchpad) split the book into its legs (sum of daily returns per year):

| year | long leg | short leg | fees | total | market |
|---|---|---|---|---|---|
| 2020 (H2) | +14.1% | −6.2% | −0.1% | +7.2% | +34.4% |
| 2021 | +15.4% | −1.7% | −0.7% | +12.9% | +28.1% |
| 2022 | −3.4% | +11.3% | −0.6% | +7.8% | −18.1% |
| 2023 | +13.9% | −8.5% | −0.8% | +4.7% | +24.7% |
| 2024 | +10.8% | −0.9% | −0.7% | +8.8% | +23.5% |
| 2025 | +35.8% | −4.8% | −2.3% | +29.2% | +23.3% |
| 2026 | +0.8% | −8.1% | **−10.3%** | −22.4% | +14.4% |

- **The sides are not alike.** Longs: beta 1.11, turnover 3.7% of shares a day, 12-1 momentum
  +49%. Shorts: beta 0.83, turnover 0.8%, momentum +13% (universe: 0.93 / 1.6% / +25%). Low
  days-to-cover mostly means "trades a lot of its shares": the book is long high-turnover,
  high-beta momentum and short low-turnover defensives. The short leg lost money in 6 of 7 years
  (it is mostly a beta hedge).
- **2026's −22% is largely self-inflicted churn.** Orders ran ~280 a month in 2026 (RGTI alone
  bought and sold 60 times, often buy-today / sell-tomorrow); fees took 10.3% of the book and the
  open-to-close timing of the churned trades another ~4.7% (the "residual"). The cause: the skew
  sleeve switched on in October 2025 and its raw 20-day change jumps every day, so names cross
  the 0.9 / 0.8 bands daily. The short-interest legs themselves made +0.8% / −8.1%.

## 2 · Batch 1 hypotheses (written before their numbers were read)

Each runs untuned through the 10 walk-forward test periods, paired against v1 untuned.
Amendment to the protocol, written now: a change whose mechanism exists only where option
chains exist (Oct 2025 →) can only move the last two periods; it is judged on those two
(both better) plus P(Δ > 0) ≥ 0.80, and must not move the other eight.

- **H1 `skew_smooth` 5:** averaging the skew over 5 sessions before its 20-day change removes
  the daily noise; expected: far fewer orders in 2026, Δalpha > 0 in both option-era periods.
- **H2 `score_smooth` 5:** averaging the composite over 5 sessions before ranking cuts
  band-crossing churn everywhere; expected: lower turnover, small Δalpha > 0 in most periods.
- **H3 `si_measure` si_share:** ranking short interest as a share of shares outstanding
  removes the turnover lean; expected: smaller beta gap between the sides; alpha unclear
  (days-to-cover is the published stronger measure).
- **H4 `neutral_by` turnover:** keep days-to-cover but rank it inside turnover groups;
  expected: smaller beta and momentum gap, alpha roughly kept if short interest carries
  information beyond turnover.
- **H5 `long_share` 0.43:** put 43% of the gross long and 57% short so the sides' betas
  (1.11 vs 0.83) roughly cancel; expected: |beta| down, alpha (after beta) about unchanged.

## 3 · v1 nested walk-forward (2026-10-19 22:52 → 23:30, 80 trials × 10 periods)

Stitched 2022-01 → 2026-10-16 (4.8 years):

| arm | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| tuned (primary) | +6.4% | 0.15 | **+5.1%** | 0.80 | −31% |
| untuned config | +4.0% | 0.13 | +2.9% | 0.67 | −31% |
| long side alone (half book) | +9.8% | 0.56 | +4.9% | 0.84 | −20% |
| tuned, no options | +9.4% | 0.19 | **+7.8%** | 0.905 | −19% |
| 2x costs | +3.0% | 0.15 | +1.7% | 0.59 | −39% |
| late fills (+50 bp a leg) | −6.8% | 0.15 | **−8.0%** | 0.18 | −61% |

Verdict **FAIL** on three gates: P(alpha > 0) 0.80 (< 0.90), median PBO 0.55 (> 0.5; ≥ 0.69 in
every option-era study), late fills −8.0%. Passed: beats 99% of random-name books (the
SELECTION is real), tuning beats untuned (+2.2 pts), |beta| 0.15, 7/10 periods positive, 2x
costs and 2x borrow. Periods: +3.0, +30.4, +29.2, +7.5, +15.8, −12.3, +12.4, +29.2, −37.2, −19.9%.
Lessons: the option sleeve (as built) costs alpha (+7.8% without it); turnover is the binding
constraint (late fills); the studies overfit most where the option knobs had data to fit.

## 4 · Batch 1 results (untuned, paired against v1 untuned +2.9%)

| hypothesis | alpha | Δ | P(Δ > 0) | periods better | Δ w/o best | beta | decision |
|---|---|---|---|---|---|---|---|
| H1 skew_smooth 5 | +8.3% | +5.4 | 0.998 | 3/3 option-era, 0 elsewhere | +2.0 | 0.13 | **adopt** |
| H2 score_smooth 5 | +10.3% | +7.4 | 0.994 | 7/10 | +3.6 | 0.12 | re-test on top of H1 |
| H3 si_share | +4.3% | +1.4 | 0.61 | 6/10 | −1.9 | −0.41 | reject |
| H4 turnover groups | +2.5% | −0.3 | 0.48 | 6/10 | −1.7 | 0.06 | reject |
| H5 long_share 0.43 | +2.3% | −0.6 | 0.22 | 3/10 | −1.1 | 0.00 | reject (beta → 0 at no alpha gain; a risk choice for later) |

H1 confirms the diagnosis: the raw skew change was noise that the book traded daily. H3's
short-interest ratio flips the book to beta −0.41 (crowded names by share count are the high-beta
ones), and turnover groups remove alpha with the lean: short interest's information here is
partly the turnover lean itself.

## 5 · Batch 2 hypotheses (on top of v1 + H1)

- **H2b `score_smooth` 5** (re-test): does composite smoothing still help once the skew is
  smoothed? Expected: smaller Δ than in batch 1, still positive outside the option era.
- **H6 a change-in-short-interest sleeve** (`w_dsi` 1, 42-bar change of days-to-cover): new
  short selling is the informed part (Boehmer, Jones & Zhang); expected: Δalpha > 0 in most periods.
- **H7 `rebalance_every` 5:** the score is refreshed only every 5 sessions (weekly trading; stops
  still daily); expected: lower turnover, better late-fill alpha, gross alpha about unchanged.

## 6 · Batch 2 results (untuned, paired against v1 + H1: alpha +8.3%, late fills +3.4%, 236 trades/yr)

| hypothesis | alpha | Δ | P(Δ > 0) | periods better | Δ w/o best | late fills | trades/yr | decision |
|---|---|---|---|---|---|---|---|---|
| H2b score_smooth 5 | +10.5% | +2.3 | 0.885 | 6/10 | +1.2 | +6.5% | 197 | **adopt** |
| H6 Δ short interest | +4.6% | −3.6 | 0.21 | 5/10 | −4.4 | −6.4% | 428 | reject |
| H7 rebalance weekly | +7.2% | −1.1 | 0.33 | 4/10 | −1.7 | +3.1% | 194 | reject |

**Where the trades still come from** (v1 + H1 + H2, exits per year): 2020-2024 between 10 and 40
band exits a side a year; 2025: 40 / 43; **2026: 183 / 183**. Median holding 20 sessions. The
option sleeve (equal weight with short interest since Oct 2025) still turns the book over every
few weeks: short interest moves twice a month, the 20-day skew change moves daily.

## 7 · Batch 3 hypotheses (on top of v1 + H1 + H2; option-era changes, judged per the amendment)

- **H8 `w_skew` 0.5:** half the weight keeps the option information but lets the slow sleeve
  dominate; expected: fewer 2026 exits, Δalpha ≥ 0 in the option-era periods.
- **H9 `w_skew` 0:** no options at all — the control: do the chains add anything net of churn?
- **H10 `skew_window` 60:** a 3-month change in skew moves slower than a 1-month one; expected:
  less churn, similar information.
- **H11 `score_smooth` 10:** more composite smoothing (dose-response of H2); expected: small
  further turnover cut; a smaller gain than H2 or none.

## 8 · Batch 3 results (untuned, paired against v1 + H1 + H2: alpha +10.5%, late fills +6.5%, 197 trades/yr)

| hypothesis | alpha | Δ | P(Δ > 0) | option-era periods Δ | late fills | trades/yr | decision |
|---|---|---|---|---|---|---|---|
| H8 w_skew 0.5 | +11.1% | +0.5 | 0.63 | +4.1, +4.5, −8.3 | +8.5% | 152 | reject (not all better) |
| H9 w_skew 0 (no options) | +10.4% | −0.1 | 0.48 | +2.4, −10.4, +2.2 | +9.3% | 100 | reject |
| H10 skew_window 60 | +6.5% | −4.0 | 0.03 | +2.3, −26.6, −12.8 | +3.3% | 184 | reject |
| H11 score_smooth 10 | +8.1% | −2.4 | 0.13 | 3/10 | +4.9% | 170 | reject |

The option sleeve adds about nothing to the gross alpha and doubles the trading (100 → 197
trades a year): it helped in H1 2026 (+10.4 pts) and cost a little elsewhere. A slower change
(60 days) loses the information, so it is a short-horizon signal. Its weight stays at 1 in the
config and the nested study chooses it per period (range 0-2). Smoothing past 5 hurts: the
5-session window is not a dial to push further.

**v2 = v1 + skew_smooth 5 + score_smooth 5** goes through the full nested walk-forward next.

## 9 · Batch 4 hypotheses (the optimizer; nested walk-forward runs)

- **H12 late-fill-aware objective:** every inner trial is scored with +25 bp of slippage a leg,
  so the study prefers settings whose edge survives worse fills; expected: lower turnover in the
  picks, a better late-fill gate, gross alpha about the same, PBO lower (less fitting of churn).
- **H13 smaller search space:** search only `dtc_window` and `w_skew` (skew_window fixed 20,
  short_stop fixed 0.5); expected: PBO lower, tuned-vs-untuned gain smaller.
- **H14 control:** the long side + SPY hedge against the long/short book, from v2's arms: does
  the single-name short leg add anything over the cheapest hedge?
- **H16 `short_below_ma` 200 (design, fast screen):** short only names below their 200-day
  average (and buy a short back once it climbs above): the short leg lost money in 6 of 7 years,
  mostly in rising markets; expected: a less negative short leg, fewer squeeze losses, Δalpha > 0
  in most periods; risk: more churn around the average.

## 10 · H16 result and its follow-ups

**H16 `short_below_ma` 200** (paired against v2 untuned, +10.5%): alpha **+14.4%**, Δ +3.9,
P(Δ > 0) 0.906, 7/10 periods better, Δ w/o best +2.5, beta 0.127 → 0.075, late fills +7.1% (from
+6.5%), trades 197 → 300 a year. Periods Δ: +16.1, +11.0, −13.6, +4.4, +1.0, +13.6, −1.3, +2.1,
+13.3, −9.9. **Adopted** (all four bars met). The extra trades come from shorts bought back when
they cross above their average.

*Hypotheses (before the numbers):*
- **H16 robustness, 100 and 300 bars:** if the effect is real (do not short uptrends), both
  should also beat no filter; the comparison against 200 is reported, never used to switch.
- **H17 `long_above_ma` 200:** the mirror — buy only names above their average. Expected:
  smaller effect than H16 (the long leg already earns; it holds momentum names that are mostly
  above their averages anyway).

**Results** (paired against v2 + H16, +14.4%):

| variant | alpha | Δ | P(Δ > 0) | periods better | late fills | trades/yr | beta |
|---|---|---|---|---|---|---|---|
| no filter (v2) | +10.5% | — | — | — | +6.5% | 197 | 0.13 |
| ma 100 | +11.5% | −2.9 | 0.10 | 4/10 | +3.1% | 323 | 0.12 |
| **ma 200 (adopted)** | **+14.4%** | — | — | — | +7.1% | 300 | 0.08 |
| ma 300 | +15.9% | +1.5 | 0.80 | 6/10 | +8.7% | 288 | 0.07 |
| H17 long_above_ma 200 | +13.3% | −1.2 | 0.40 | 5/10 | +3.7% | 361 | 0.01 |

The filter helps at every length and more the longer the average: "do not short a stock in a
long uptrend" is the effect, not a lucky window. 200 stays (pre-declared; 300's edge over it is
within noise, P 0.80 with Δ w/o best +0.4). H17 rejected: the long side does not want the mirror
rule (it costs alpha and doubles the churn); the asymmetry fits the diagnosis — the short leg is
where the book bled.

## 11 · v2 nested walk-forward (2026-10-19 23:35 → 23:52)

| arm | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| tuned (primary) | +8.2% | 0.13 | **+7.1%** | **0.94** | −14.5% |
| untuned config | +11.6% | 0.13 | **+10.5%** | 0.98 | −13.6% |
| long side alone | +14.5% | 0.55 | +9.8% | 0.99 | −14.4% |
| long side + SPY hedge | +8.0% | 0.13 | +6.9% | 0.95 | −14.3% |
| late fills (tuned) | +4.6% | 0.13 | +3.5% | 0.76 | −23.9% |

Verdict **FAIL** on two gates: tuning LOSES to the untuned config (−3.4 pts) and the median
probability of overfitting is 0.74. Everything else passes now (P 0.94, late fills +3.5%, 2x
costs +6.2%, 2x borrow +6.9%, luck 99th percentile, 7/10 periods). The smoothing fixed the
execution problem; what fails is the optimizer itself — informed_flow v27's lesson again: the
studies chase noise (w_skew 1.5-2.0 in the option-era periods, where the option sleeve hurt).
**H14** (single-name shorts vs SPY hedge): the untuned long/short book beats long side + SPY
hedge by +3.6 pts (P 0.92) — the short leg earns its borrow; the design stays long/short.

## 12 · Optimizer hypotheses (on the v3 design = v2 + short_below_ma 200)

v3 runs first with the unchanged procedure (the reference). Then, one at a time:
- **O1 smaller space:** search `dtc_window` and `short_stop` only (`skew_window` 20, `w_skew` 1
  fixed): the option knobs had one year of data and the worst picks. Expected: PBO lower, tuned
  ≥ untuned.
- **O2 plateau pick:** the trial whose score with its 4 nearest neighbours is best (a setting
  whose surroundings also work), not the single best. Expected: tuned closer to untuned or above.
- **O3 config guard:** the config's own values are trial 0 and stay unless the pick beats them
  by 2 pts of robust alpha on the training window. Expected: tuned ≥ untuned by construction in
  most periods; the question is whether the guard ever lets a better setting through.
- **O4 late-fill-aware objective (H12):** trials scored with +25 bp a leg.

## 13 · Batch 5 hypotheses (design, fast screens on the v3 base)

- **H18 `short_max_si` 0.2:** never short above 20% of shares (from 30%): less squeeze and
  borrow risk; expected: small Δalpha ≥ 0, lower borrow cost, fewer short-stop exits.
- **H22 `skew_one_sided`:** the skew sleeve only pushes names toward the short side (rising put
  skew = informed downside buying; falling skew earns no long bonus). Expected: option-era Δ ≥ 0
  with less churn on the long side.

## 14 · Batch 5 results, v3 walk-forward, factor attribution

**Batch 5** (paired against v3 untuned, +14.4%): **H18 `short_max_si` 0.2** (si_high 0.15):
+15.6%, Δ +1.2, P 0.84, **8/10 periods better**, Δ w/o best +0.7, beta 0.075 → 0.103 → **adopted**
(v4). **H22 one-sided skew:** Δ −0.1, option-era −3.2 / +13.7 / −12.3 → rejected.

*Process slip, caught:* the config was edited (H18) while the v3 walk-forward was running; the
period jobs read the config when they start. It was reverted within ~30 s; the job start times
(studies created 00:00:37 ×3, 00:02:45, 00:03:16, 00:03:40; the next start after 00:05) show no
job started in that window, so v3 ran on one config. Rule since: no edits to config.yaml,
strategy.py or research.py while a walk-forward runs; design screens use a dev copy.

**v3 nested walk-forward** (v2 + short_below_ma 200; 2026-10-20 00:00 → 00:20):

| arm | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| tuned (primary) | +17.6% | 0.13 | **+16.5%** | 0.998 | −15.7% |
| untuned config | +15.1% | 0.08 | +14.4% | 0.995 | −14.4% |
| late fills (tuned) | +13.0% | 0.13 | +11.8% | 0.98 | −16.3% |
| 2x costs (tuned) | +16.4% | 0.13 | +15.3% | 0.998 | −16.1% |

Periods (tuned): +30.1, +35.5, +14.3, +4.5, +25.0, +4.1, +28.7, +31.8, −3.3, −19.1%. Gates: all pass
except the median probability of overfitting, **0.61**. Tuning now beats not tuning (+2.0 pts);
the book beats 100% of 30 random-name books.

**Is it a factor in disguise?** (`factors.py`: daily returns on the equal-weight market, a 12-1
momentum top-minus-bottom fifth and a low-minus-high turnover fifth, point-in-time universe):

| arm | momentum loading | low-turnover loading | market | alpha after the factors |
|---|---|---|---|---|
| tuned | 0.28 (t 28) | −0.20 (t −14) | 0.02 | **+12.7%/yr (t 3.0)** |
| untuned | 0.23 (t 22) | −0.08 (t −6) | 0.06 | **+11.2%/yr (t 2.6)** |

The book is half a momentum bet by variance (correlation 0.51) and leans to high-turnover names,
as the diagnosis said; but the factors explain ~4-5 pts of its return, not the edge.

## 15 · Batch 6 hypotheses (design, dev copy, on the v4 base)

- **H25 `neutral_by` momentum:** rank the sleeves inside 12-1 momentum fifths instead of
  dollar-volume fifths: the momentum loading (0.23-0.28) is risk the edge does not need;
  expected: lower volatility and momentum loading, alpha after factors kept, raw alpha lower
  (2022-26 paid momentum).

**H25 result:** momentum groups: +9.7%, Δ −6.0, P 0.10, 4/10 periods → rejected. Short
interest's edge here is entangled with momentum; neutralizing it throws the alpha away.

- **H28 `neutral_by` sector** (securities.sector, constant per name): rank inside sectors so
  the book stops making sector bets (long tech/crypto, short staples/telecom); expected: lower
  volatility, alpha roughly kept if the information is within-sector.

**H28 result:** sector groups instead of dollar-volume groups: +8.3%, Δ −7.4, P 0.04 → rejected.
The short-interest information is largely ACROSS sectors (and liquidity groups).

## 16 · Batch 7 hypotheses (turnover; dev copy, on the v4 base: alpha +15.6%, late fills +8.0%, 304 trades/yr)

- **H33 `short_ma_band` 0.03:** a short is entered below its 200-day average but bought back
  only once 3% above it (hysteresis): the filter's buy-backs whipsaw around the line; expected:
  fewer trades, better late-fill alpha, gross alpha kept.
- **H32 `q_out` 0.7:** hold until the rank falls below 0.7 (shorts: above 0.3): fewer band exits;
  a single pre-registered design test, never a search (informed_flow v27's lesson); expected:
  fewer trades, small gross alpha cost.
- **H31 `slots` 20:** more names a side (2.5% each): more diversified, less idiosyncratic noise;
  expected: lower volatility, alpha per year about the same or lower.

**Batch 7 results** (paired against v4 untuned +15.6%, late fills +8.0%, 304 trades/yr):

| hypothesis | alpha | Δ | P(Δ > 0) | periods better | late fills | trades/yr | decision |
|---|---|---|---|---|---|---|---|
| H33 short_ma_band 0.03 | +15.4% | −0.2 | 0.42 | 5/10 | +9.4% | 247 | reject (cheaper, not better) |
| H32 q_out 0.7 | +16.7% | +1.0 | 0.68 | 6/10 | +10.3% | 262 | reject (P < 0.80) |
| H31 slots 20 | +9.9% | −5.7 | 0.004 | **0/10** | +2.2% | 412 | reject |

H31 is the night's clearest result: the 5 extra slots per side fill with weaker-ranked names and
cost alpha in EVERY period: the edge sits in the extreme names. (Fewer slots is NOT tested:
informed_flow's studies always floor-pinned slots — a concentration artifact.)

## 17 · Batch 8 hypotheses (quality filters; fund_ni_ttm, SEC TTM net income, point in time)

- **H36 `long_profitable`:** buy only companies with positive trailing net income. Low
  days-to-cover among unprofitable high-turnover names (2026 losers RGTI, CIFR) is retail
  turnover, not informed confidence; expected: a better long leg, most in junk reversals.
- **H37 `short_unprofitable`:** short only companies with negative trailing net income ("short
  junk"): short interest on profitable defensives (VZ, MO, KMB) is often hedging or merger
  arbitrage; expected: a short leg that loses less in rising markets.

**Batch 8 results** (paired against v4 untuned +15.6%):

| hypothesis | alpha | Δ | P(Δ > 0) | periods better | late fills | trades/yr | beta | decision |
|---|---|---|---|---|---|---|---|---|
| H36 long only profitable | +11.4% | −4.2 | 0.07 | 3/10 | +3.8% | 308 | 0.08 | reject |
| H37 short only unprofitable | +16.2% | +0.5 | 0.55 | 3/10 | **+12.0%** | **161** | 0.21 | reject (P, beta) |

The pattern of the night: the alpha sits in the raw short-interest extremes, and filters that
second-guess them (quality, sectors, momentum, turnover) take it away. What helped was removing
noise (smoothing) and avoiding the two known ways a short goes wrong (uptrends, crowding). H37 is
worth remembering for execution: it halves the trading at the same gross alpha.

- **H38 `si_measure` dtc_z:** days-to-cover relative to the name's own past year (z-score):
  abnormal short selling, not the structural level (low-turnover names always carry high
  days-to-cover). Expected: different names, alpha unclear (H6's 42-bar change failed).

**H38 result:** dtc z-score against the name's own year: −1.2%, Δ −16.8, P 0.002, 1/10 periods →
rejected. With H6, this settles what the signal is: the persistent LEVEL of days-to-cover (which
names are structurally heavily shorted relative to their trading), not new short selling.

## 18 · How the optimizer procedure will be chosen (written before any O-variant result)

The v4 base (unchanged procedure) and O1-O4 each produce a verdict on the same 10 periods. Rule:
among the runs that pass ALL hard gates, take the one with the LOWEST median probability of
overfitting (the gate this batch targets) — not the highest alpha, which would be one more
selection on the same walk-forward. If none passes, the procedure that passes everything but
PBO with the lowest PBO is reported as the best candidate and the verdict stays FAIL.

## 19 · v4 base walk-forward (v3 + short_max_si 0.2; the unchanged procedure)

| arm | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| tuned (primary) | +12.6% | 0.11 | +11.6% | 0.99 | −17.6% |
| untuned config | +16.5% | 0.10 | **+15.6%** | 0.998 | −14.5% |
| late fills (tuned) | +4.5% | 0.11 | +3.5% | 0.73 | −32.0% |

FAIL: tuning loses to the config (−4.0 pts) and median PBO 0.68. Across the three nested runs the
tuning effect is −3.4 (v2), +2.0 (v3), −4.0 (v4): it flips sign with small design changes and
averages −1.8 pts — noise, as the PBO of 0.6-0.7 says. The untuned config improved step by step:
+2.9% (v1) → +10.5% (v2) → +14.4% (v3) → +15.6% (v4).

**The option sleeve on the v4 design** (paired against v4 untuned, w_skew 1, +15.6%):
w_skew 0 → +13.6% (Δ −2.0; option-era periods −4.9, **−22.1**, +5.5; trades 304 → 207); w_skew
0.5 → +14.8% (Δ −0.8). Once its noise is smoothed away the option sleeve ADDS ~2 pts, almost all
of it in H1 2026 — the half-year short interest inverted: the second, independent signal covered
for the first one's worst stretch. (One period of evidence; it stays at weight 1.)

**Survivorship check.** Of the 780 names that were ever members of the point-in-time universe
(2020-07 → 2026-10), only 7 stop before the end of the data — all 2026 acquisitions (EA, EQR,
GTLS, LBRDK, QRVO, SATS, WBD). The database holds no names that left earlier (Twitter,
Activision, VMware, Pioneer, Splunk, Hess, ANSYS, Discover; the 2023 bank failures SVB,
Signature, First Republic), so the universe is formed from today's survivors. Direction for a
long/short book is mixed: missing takeovers would have hurt shorts and helped longs, missing
failures would have paid shorts. It cannot be fixed without delisted-name prices; it is a
limitation of every number here (and of informed_flow's).

**v4 factor attribution** (`factors.py`, 1,202 out-of-sample days): untuned — momentum 0.23 (t 23),
low turnover −0.12 (t −8), market 0.05, **alpha after factors +12.3%/yr (t 2.9)**; tuned — alpha after
factors +9.0% (t 2.2). The short trend filter did not buy its gain with momentum risk (loading
0.23 as before).

## 20 · O1 (smaller search space: dtc_window, short_stop)

Tuned +12.7% vs untuned +15.6% (Δ −2.9), median PBO 0.72 → FAIL (the same two gates). Nine of ten
periods positive; late fills +5.6%. Even a 24-point space overfits: no setting in it is durably
better than the config's.

## 21 · The untuned v4 config as the primary (`untuned_verdict.py`, 100 random-name books)

| arm | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| **config (v4)** | **+16.5%** | 0.10 | **+15.6%** | 0.998 | −14.5% |
| 2x costs | +14.6% | 0.10 | +13.7% | 0.99 | −16.0% |
| 2x borrow | +16.0% | 0.10 | +15.1% | 0.997 | −14.7% |
| late fills | +8.9% | 0.11 | +8.0% | 0.91 | −21.8% |
| long side + SPY hedge | +8.0% | 0.13 | +6.9% | 0.95 | −14.3% |

Periods: +29.8, +30.2, +2.7, +6.8, +14.4, +14.4, +21.5, +46.2, +16.6, −32.1%. Random-name books with
the same rules: median +0.2%, 95th percentile +4.1% — the config beats all 100. Every applicable
gate passes (the two procedure gates — tuning beats not tuning, PBO — do not apply to a book
nothing tunes).

**Caveat that the gates cannot see:** the v4 design was chosen by iterating on these same 10
periods (22 variants tried by now, 4 adopted under the paired bar). The luck band measures the
choice of NAMES, not the choice of DESIGN. Expect the forward numbers to come in lower; paper
trading from 2026-10-20 is the test that counts.

## 22 · A fresh cross-section (pre-registered before it runs)

Every design decision tonight was read on the point-in-time TOP 500. The names ranked 501-1000
by the same rule (same 6-month formations) were never looked at, so they are a fresh test of
the DESIGN — the one thing the walk-forward cannot test any more. Pre-registered:

- **Universe:** each formation's names ranked 501-1000 by median dollar volume (same floors:
  ≥ $5, ≥ 100 sessions; the $50M dollar-volume floor lowered to $10M so the band is filled).
- **Test:** the untuned v1 config and the untuned v4 config through the same 10 test periods,
  paired. **The design generalizes if** v4 beats v1 with P(Δ > 0) ≥ 0.80 AND v4's own alpha is
  > 0 with P ≥ 0.80. If not, tonight's changes are fitted to the top 500 and the README says so.

**Result (§22, names 501-1000): DOES NOT GENERALIZE.**

| | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| v1 untuned | +4.6% | 0.06 | +4.1% | 0.84 | −14.1% |
| v4 untuned | +2.4% | 0.03 | +2.2% | 0.69 | −21.5% |

v4 − v1: Δ −1.9 pts, P 0.28, v4 better in 3/10 periods. By the pre-registered rule, tonight's
design changes are (at least in part) fitted to the top-500 walk-forward. The base signal itself
carries over (v1: +4.1%, P 0.84 on names nobody looked at). Next: each of the four adoptions
alone on the same band (this reads the fresh band for those four components — it is no longer
fresh for them afterwards).

**Each adoption alone on names 501-1000** (paired against v1, +4.1%):

| change | alpha | Δ | P(Δ > 0) | periods better |
|---|---|---|---|---|
| skew_smooth 5 | +4.5% | +0.4 | 0.60 | 3/10 (moves only the option era) |
| score_smooth 5 | +3.2% | −1.0 | 0.32 | 5/10 |
| **short_below_ma 200** | −0.9% | **−5.0** | 0.02 | 3/10 |
| **short_max_si 0.2** | +0.3% | **−3.9** | 0.02 | 3/10 |

The two short-side filters — the night's biggest top-500 gains — HURT on names nobody had looked
at. Either they are specific to large caps (in smaller names the crowded, rising shorts may be
the best shorts) or they were walk-forward luck; the evidence cannot tell which, so they are not
trusted. The smoothing changes, whose need the 2026 fee blow-up showed directly, survive.

**Decision rule (written before the next number):** ship v2 (v1 + both smoothings) if it is at
least as good as v1 on the fresh band (Δ ≥ 0); otherwise ship v1 + skew_smooth only.

**v2 on the fresh band:** +5.5%, Δ +1.3 vs v1, P 0.70, 5/10 periods → by the rule, **v2's design
ships** (both smoothings; short_below_ma and short_max_si 0.2 reverted).

## 23 · O3 (config guard, 2 pts) — and the end of the optimizer question

Tuned +14.4% vs untuned +15.6% (Δ −1.2), PBO ~0.67 → FAIL. The guard kept the config in 4/10
periods; the picks it let through cost (2025-07: +34.8 vs +46.2; 2026-01: −1.3 vs +16.6).

| procedure (v4 design) | tuned alpha | untuned | Δ | median PBO |
|---|---|---|---|---|
| TPE, 4 knobs, best trial | +11.6% | +15.6% | −4.0 | 0.68 |
| O1: 2 knobs | +12.7% | +15.6% | −2.9 | 0.72 |
| O3: config guard | +14.4% | +15.6% | −1.2 | ~0.67 |

Four procedures (with v2, v3: −3.4, +2.0) — the nested search never beat not tuning in a way that
held. **Conclusion: nothing in this space is worth tuning; the book is the design.** O2 (plateau)
and O4 (late-fill objective) were stopped unfinished: they only tested procedures on the
abandoned v4 design, and the answer above would need a large reversal to change.

## 24 · v5 = the v2 design, shipped; a turnover test judged on BOTH universes (pre-registered)

Code cleaned to the shipped design (rejected knobs removed; their results stay in this log).
v5's own nested walk-forward and untuned verdict are running (§25).

**Cost bar (written before the numbers):** a turnover reducer is adopted only if, on BOTH the
top 500 and the 501-1000 band, its late-fill alpha (+50 bp a leg) rises — paired P ≥ 0.8 on the
top 500, Δ ≥ 0 on the band — and its gross alpha falls by no more than 0.5 pt on each.
Candidates: `q_out` 0.75 and 0.7 (hold until the rank falls further; shorts mirror).

**Cost-bar results** (v5 base):

| | top 500: alpha | Δ (P) | late fills | Δ late (P) | band 501-1000: alpha | Δ (P) | late | Δ late (P) |
|---|---|---|---|---|---|---|---|---|
| v5 | +10.5% | — | +6.5% | — | +5.5% (P 0.90) | — | +1.2% | — |
| q_out 0.75 | +7.6% | −3.0 (0.09) | +3.9% | worse | +7.7% | +2.2 (0.82) | +3.8% | +2.6 (0.88) |
| q_out 0.7 | +11.4% | +0.8 (0.63) | +8.0% | +1.5 (0.73) | +5.3% | −0.2 (0.47) | +2.1% | +0.9 (0.63) |

Neither passes (0.75 fails the top 500 outright; 0.7's late-fill gain is P 0.73 < 0.8). The two
universes disagree on which band is better — noise. **v5 stays.** By-product worth having: the
shipped v5 design on names nobody looked at earns +5.5%/yr alpha (P 0.90) — about half its
top-500 level — with a thin +1.2% after late fills (smaller names cost more to trade).

## 25 · Final: v5 (the v2 design) — nested walk-forward and the untuned verdict

**Nested (tuned, the pre-registered primary):** identical to v2 (same design, seeds and data):
alpha +7.1%, P 0.94; FAIL on tuning (−3.4) and PBO (0.74).

**The untuned config (`untuned_verdict.py`, 100 random-name books):**

| arm | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| **config (v5)** | **+11.6%** | 0.13 | **+10.5%** | 0.98 | −13.6% |
| 2x costs | +10.7% | 0.13 | +9.5% | 0.97 | −14.4% |
| 2x borrow | +11.4% | 0.13 | +10.3% | 0.98 | −13.7% |
| late fills | +7.6% | 0.13 | +6.5% | 0.88 | −23.1% |
| long side + SPY hedge | +8.0% | 0.13 | +6.9% | 0.95 | −14.3% |

Periods: +9.9, +21.5, +16.6, +1.9, +12.2, −0.3, +22.6, +43.7, −1.3, −22.4%. Random-name books:
median −0.3%, 95th percentile +4.6%; the config beats all 100. Every applicable gate passes.
Names 501-1000: +5.5%, P 0.90.

## Summary of the loop

- **Tried:** ~30 design screens (each paired on the same days against the current book), 5 full
  nested walk-forwards (+2 optimizer variants), a factor attribution, a survivorship check and a
  fresh-cross-section test.
- **Kept:** `skew_smooth` 5, `score_smooth` 5 — the fix for the churn that cost v1 its 2026.
- **Learned:**
  1. the option sleeve, smoothed, adds a second, independent signal that covered the
     short-interest inversion of H1 2026;
  2. the edge is the LEVEL of short interest in the extreme names (20 slots worse in every period;
     changes and z-scores worthless);
  3. nested tuning never beat the config — the procedure fails its own overfitting gate in
     every variant;
  4. walk-forward iteration overfits: the two filters that won on the top 500 lost on names
     nobody had looked at.
- **Next:** paper trade the v5 config from 2026-10-20 (notebook §7, "config values").

## 26 · H39: a foreign-filer screen (data hygiene; pre-registered)

The v5 paper book's top long is BTI (British American Tobacco ADR), days-to-cover 1.00 — the
v37 problem: FINRA short interest covers only a foreign issuer's US line, so it looks falsely
low and ranks long. The SEC share-count requirement does not catch every foreign filer. The
fundamentals table records filing forms: 304 names filed a 20-F, 40-F or 6-K (foreign private
issuers). **H39:** exclude them from ranking. It is judged as hygiene, not as an edge: adopt if
alpha falls by no more than 0.5 pt on BOTH the top 500 and the names ranked 501-1000.

**H39 results:** excluding foreign private issuers costs −5.4 pts on the top 500 (+10.5% → +5.1%,
P(Δ > 0) 0.05; P(alpha > 0) 0.875) and −3.9 on names 501-1000 (+5.5% → +1.6%, P 0.10) → not
adopted by the pre-registered hygiene rule — but it changes what the edge IS (`foreign.py`):

| | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | mean |
|---|---|---|---|---|---|---|---|---|
| foreign share of long slots | 19% | 14% | 36% | 35% | 35% | 41% | 25% | **30%** |
| foreign share of the universe | 7% | 6% | 7% | 7% | 7% | 7% | 7% | 6.6% |
| long P&L, foreign names | +0.4 | −0.1 | −1.0 | +6.6 | +1.4 | **+21.4** | +1.6 | +30.2 pts |
| long P&L, domestic names | +16.7 | +15.2 | −5.5 | +6.8 | +11.2 | +11.2 | +2.9 | +58.6 pts |

Foreign filers are 4.5× over-represented on the long side: their FINRA short interest covers only
the US line, so their days-to-cover is low by construction (informed_flow v37). A third of the
long side's profit — and most of 2025's — is therefore a "long foreign ADRs" tilt that the
measurement creates, not short sellers' information. The domestic-only book still earns +5.1%/yr
alpha on the top 500, but only +1.6% on the fresh names. **Left for the user to decide**
(README "Status"): ship the book as measured, or exclude foreign filers and accept about half
the backtested alpha for a cleaner signal.

## 27 · Concentration: a few winners carry the alpha (`top5.py`, v5 config, test periods)

Top contributors (sum of daily return contributions over the 10 test periods): NVDA +9.5 pts,
CLS +6.0, PBR +4.6, AMD +4.5, KGC +4.3, QBTS +4.2, HOOD +3.8, OKLO +3.5, TDOC +3.4, TWLO +3.4
(315 names traded). Worst: PANW −3.7, CAR −2.3, LYV −2.1, RCL −2.0, AI −1.9.

| | alpha | P(alpha > 0) |
|---|---|---|
| v5 as is | +10.5% | 0.98 |
| without the top 5 names' P&L | +4.9% | 0.87 |
| without the top 10 | +1.3% | 0.65 |

Half the alpha is five names, almost all of it ten: heavily traded AI / momentum names whose
turnover keeps their days-to-cover low, held long through their runs — the momentum / turnover
tilt again (§14), and informed_flow v38's "the gain was a few winners". With §26 (a third of
the long profit is a foreign-filer measurement tilt), the honest reading of v5 is: a real but
thin short-interest edge, amplified by two tilts that paid in 2023-2025.

## 28 · Fresh in time: 2020-07 → 2021-12 (pre-registered)

Every design decision was judged on the walk-forward periods from 2022-01. The three six-month
periods before (2020-07, 2021-01, 2021-07; each on its own point-in-time members, warm, untuned)
were never used to choose a design. **Test:** v5 vs v1, paired. The design holds in time if v5's
Δ ≥ 0 and v5's own alpha > 0 (low power: 1.5 years — a direction check, not a proof).

**Result: HOLDS.** Per period v1 / v5 alpha: −11.5 / −1.6, −4.5 / +3.6, +6.6 / +6.9%. Stitched: v1
−3.1% (P 0.28), **v5 +2.7% (P 0.66)**; v5 − v1 **+5.8 pts, P 0.96**. No option chains exist
before October 2025, so this tests `score_smooth` alone: the smoothing holds in time as well as
on the fresh names. The level of the edge in 2020-21 is modest — short interest was a weaker
signal then (informed_flow's pre-period IC was even negative).

## Final state (2026-10-20 ~02:00)

Shipped config: v5 (study `informed-flow-ls-1d-v5`) — the v1 design + `skew_smooth` 5 +
`score_smooth` 5, untuned. Evidence for it: walk-forward 2022-26 +10.5%/yr alpha (P 0.98, all
applicable gates), fresh names +5.5% (P 0.90), fresh time +2.7% (P 0.66); the smoothing beats no
smoothing in all three. Against it: half the alpha is five names; a third of the long profit is
a foreign-filer measurement tilt; 2026 is its worst year. Open decision for the user: exclude
foreign filers (cleaner, about half the alpha). Next step either way: paper trading.

## 29 · A three-way protocol for anything further (written before its numbers)

Three tests now exist that are independent of each other: the top-500 walk-forward (2022-26),
the fresh names 501-1000 (same periods), the fresh time 2020-07 → 2021-12 (top 500). Any
further change is adopted only if, against v5 on the same days: top-500 Δ ≥ 0 with P ≥ 0.7, AND
fresh-names Δ ≥ 0, AND fresh-time Δ ≥ 0. Candidates:
- **`dtc_window` 84 and 126:** the nested studies most often picked 63-105; if a longer average
  is genuinely better it should hold on all three.
- **`retrim_k` 1.5:** trim a long once it doubles to 1.5× its slot (not 2×): half the alpha is
  five runaway winners (§27); trimming earlier may lower the dependence or just cut the edge.
- **The domestic-only book (foreign filers out)** is profiled on all three for the user's
  decision (§26), not judged by this bar.

**Three-way results** (alpha per test; Δ against v5 with P(Δ > 0)):

| variant | top 500 | names 501-1000 | 2020-21 | Δ top 500 | Δ names | Δ 2020-21 | adopt |
|---|---|---|---|---|---|---|---|
| **v5** | +10.5% | +5.5% | +2.7% | — | — | — | — |
| dtc_window 84 | +9.8% | +4.7% | +6.4% | −0.8 (0.36) | −0.8 (0.39) | +3.7 (0.81) | no |
| dtc_window 126 | +7.2% | +7.4% | −2.3% | −3.4 (0.18) | +1.9 (0.76) | −4.9 (0.19) | no |
| retrim_k 1.5 | +10.0% | +4.6% | +2.9% | −0.5 (0.11) | −0.9 (0.04) | +0.2 (0.59) | no |
| *domestic only* | *+5.1%* | *+1.6%* | *−0.6%* | *−5.4 (0.05)* | *−3.9 (0.10)* | *−3.3 (0.19)* | *(user's call)* |

The averaging window flips sign across the three tests — noise, so 42 stays (and the nested
studies' preference for 63-105 was noise too). Trimming winners earlier costs a little
everywhere. The foreign-filer tilt helps in ALL THREE independent tests: a persistent feature of
how FINRA measures foreign issuers, not just 2025 — which argues it may persist, though its
source is a measurement artifact rather than short sellers' information.

**v5 stands.** Loop ended here (~02:00): the three-way bar has found nothing further, and each
extra variant tested on the same data adds to the selection the gates cannot see.

## 30 · Can the foreign-filer gap be filled? (2026-10-20, user question)

**Who is affected.** Of the v5 book's foreign long P&L (+30 pts over the walk-forward), by where
each name's home listing trades (FINRA sees only the US line when there is one):

| home market | long P&L (pts) | names (long P&L) | public total short interest? |
|---|---|---|---|
| Canada (TSX) | +12.3 | CLS 5.5, KGC 3.4, B 3.0, CP, MGA, WCN, CCJ (TECK, PAAS, WPM, AG < 0) | CIRO, twice monthly since 2018-11-30, free XLS (behind a Cloudflare check: no scripted download) |
| Brazil (B3) | +5.1 | PBR | B3 open securities-lending positions, daily, free, ~10 days kept online |
| UK (LSE) | +4.9 | BCS 2.9, BTI 2.4, AZN, AU | none before 2026-07-13; since then FCA aggregate of notified positions (≥ 0.1%), not stock-loan totals |
| EU (Euronext, Xetra, Copenhagen) | +3.3 | ASML 1.4, STM 1.3, NVO 1.0, SNY, SAP, MT | none: only individual positions ≥ 0.5% of capital (mega caps: usually nothing) |
| Switzerland, Chile, Hong Kong, Israel | −0.3 | NVS, SQM, EDU, TSEM | HK only (SFC weekly since 2012) |
| **US listing only** (no gap) | +4.8 | IREN 4.6, ICLR, STNE | FINRA already complete |

The shorts include five Canadian large caps (ENB, CNQ, SU, TD, CNI: −1.8 pts together) — the US
line can read too HIGH as well as too low.

**Is the tilt measurement or signal?** (`foreign_ic.py`, `threeway2.py`)
- Inside the foreign pool alone, days-to-cover still sorts the next 21 bars: rank IC +0.038
  (t 1.7, ~30 names) against +0.034 (t 3.5) for domestic names — the same size, noisier. By year
  −0.02 to +0.17 (2025) and −0.105 in 2026.
- Foreign filers as a group beat domestic names in 2022 (+11 pts) and 2025 (+18) and lost in
  2020-21 (−31): a large factor swing the long end rides.
- They are 5% of every fifth of the composite except the long fifth (10%), and 30% of the 15
  long slots: the over-representation sits at the very extreme.
- **Ranking foreign filers in a pool of their own** (so they fill the extremes in proportion)
  loses like excluding them, in all three tests: top 500 +6.5% (Δ −4.1, P 0.08), names 501-1000
  +1.2% (Δ −4.3), 2020-21 −3.2% (Δ −5.9). Excluding: −5.4 / −3.9 / −3.3.

So the low US-line days-to-cover of a foreign name predicted its return better than the
domestic name that would replace it, in three independent tests — including 2020-21, when
foreign names as a group lost 31 pts. Whatever the mechanism, filling the gap would change
exactly these ranks: adding home-market short interest raises a home-listed name's
days-to-cover and moves it out of the long extreme, which is what both tested fixes did. Only
the data itself can say whether total short interest ranks foreign names better than the US
line; the cheapest test of that is Canada (41% of the foreign long P&L, free official data).

**Borrow, from the same data hunt.** IBKR's public short-stock file (`ftp2.interactivebrokers.com`,
user `shortstock`, one file per country, refreshed intraday, current snapshot only) has all 478
eligible names of 2026-10-16: median fee 0.30% in the < 10% tier, 0.40% at 10-20% and 0.41% at
20-30% (the model charges 0.5% / 3% / 10%). All 47 current short-end candidates cost 0.25-0.45%.
On liquid large caps the assumed tiers are conservative by a wide margin (one day's snapshot).

## 31 · Loop 2 (2026-10-21): the refreshed data — pre-registered before any number

**What changed in the data since loop 1.** Short interest is now complete for every name from
July to September 2026: the old update had reached ~765 of ~2,300 names, so in T2 below 267 of
the 500 names lost short interest from late August. The 2026-09-15 and 09-30 settlements are read
for the first time (FINRA started quoting every field; the parser read 0 rows). The insider
de-duplication does not touch this book. Daily prices still end 2026-10-16 (the window below).
The on-disk panel cache predated the ingest (its key has no data content) and was rebuilt.

**Tests** — untuned config values, every variant paired against v5 on the same days:
- **T1** the top-500 walk-forward periods, 2022-01 → 2026-10-16 (10 periods);
- **T2** the names ranked 501-1000 (same periods, $10M dollar-volume floor);
- **T3** 2020-07 → 2021-12 on the top 500 (3 periods);
- **T4 (new, LOCKED)** the names ranked 1001-1500 (same periods as T1, $3M floor). Read ONCE, at
  the end of this loop, for the final candidate against v5. Nothing is decided on it except
  whether the final candidate's change over v5 generalizes; its result is reported either way.

**Adoption bar** (§29 unchanged): T1 Δ ≥ 0 with P(Δ > 0) ≥ 0.7, T2 Δ ≥ 0, T3 Δ ≥ 0, and a
mechanism written down before the numbers. **Budget: at most 12 variants through the bar in this
loop**, counted here; every variant read on the same three tests adds selection the gates cannot
see, so a change that passes narrowly late in the loop is weak evidence.

**Step 0.** v5 re-baselined on T1-T3 (expected: T1 nearly unchanged — the top 500 was mostly
covered — and T2's last period changed).
**Step 1 — rejections read on contaminated data, re-tested (3 of the budget):**
- **R1 `short_below_ma` 200** (do not short, and buy back, above the 200-day average; §10). It
  lost 5.0 pts on T2 (§22) — partly on the half-blank last period.
- **R2 `short_max_si` 0.2** (si_high 0.15; §14): never short above 20% of shares. Lost 3.9 on T2.
- **R3 `dtc_window` 84**: passed T3 (+3.7), failed T1/T2 narrowly (§29).
**Step 2** — new ideas, written in §32 after Step 0-1's inspection, before their numbers.
**Step 3** — the final candidate through the nested Optuna walk-forward (the pre-registered
primary, new study name), the random-name luck band, the cost / borrow / late-fill stresses, and T4.

## 32 · Loop 2, Step 2 hypotheses (from the v5 inspection, `anatomy.py`; before their numbers)

**What the inspection showed** (v5 on the point-in-time universe, 2020-07 → 2026-10, residual =
return − trailing beta × the equal-weight market):
- **Shorts lose in their first two weeks:** −7.6 bp a day over days 1-10 and −2.4 over 11-21,
  then +5.4 over 22-42. Longs earn nothing over days 1-10 (+0.7), then +7.8 over 11-21.
- **2026 churned:** about 170 completed trades per side in 2026 against 6-50 a year before
  (median band-exit hold 20 bars), at ~0 residual per trade. The option-skew sleeve switches on
  in October 2025.
- **Exits are well timed:** a long sold on the band trails by a further 2.2% over the next 21 bars
  and 5.8% over 63; a short covered on the band rises 2.9% over 63. Squeeze-stop exits (14
  trades) lost 29% each and the stocks then fell back 9%.
- The composite ranks the next 21 bars monotonically at both ends (twentieth 1: −116 bp,
  twentieth 20: +19 bp, the middle about −45); in 2026 the long end lost (−116 bp).

**Hypotheses** (each paired against v5 on T1-T3, the §31 bar; budget 5 of 12 with R1-R3):
- **N1 `confirm` 5:** a name enters either book only after 5 consecutive bars in its entry zone.
  *Mechanism:* names reach the short zone right after falls and bounce (the one-month reversal);
  a signal that persists is information, one that flickers is the skew sleeve's noise. *Expected:*
  fewer trades, the first-days loss smaller, T1 Δ ≥ 0, T3 Δ ≥ 0 (the reversal predates options).
- **N2 `dual_band`:** a position leaves on the band only once BOTH the composite and the
  short-interest rank have left it (a missing short-interest rank: the composite alone).
  *Mechanism:* the skew's information lasts 10-21 days, short interest's months; a skew wiggle
  should not sell a name short sellers still avoid. *Expected:* only the option era changes (T3
  identical by construction), fewer 2026 trades, T1 Δ ≥ 0.

## 33 · Loop 2, Steps 0-1: the baseline and the re-tests (`loop2.py`)

**v5 on the refreshed data:** T1 +10.3%/yr (P 0.98, beta 0.13, max DD −14.0%, late fills +6.3%,
296 orders a year); T2 **+4.2%** (P 0.84; was +5.5% with the half-blank last period); T3 +2.7%
(P 0.66, unchanged — no data changed there). Per T1 period: +9.9, +21.5, +16.6, +1.9, +12.2,
−0.3, +22.6, +43.7, −1.3, **−25.2** (2026-07 → 10-16).

| re-test | T1 Δ (P) | T2 Δ (P) | T3 Δ (P) | orders/yr | adopt |
|---|---|---|---|---|---|
| R1 `short_below_ma` 200 | **+4.0** (0.91), 7/10 | −0.4 (0.44), 4/10 | −2.9 (0.29), 1/3 | 497 | no |
| R2 `short_max_si` 0.2 | −0.2 (0.41) | −0.3 (0.44) | −3.8 (0.09), 0/3 | 297 | no |
| R3 `dtc_window` 84 | −0.4 (0.41) | +0.1 (0.52) | +3.7 (0.81) | 275 | no |

The data fix mattered for R1: its fresh-names loss shrank from −5.0 to −0.4 points — most of
§22's verdict on it was the half-blank last period. It still fails the bar: 2020-21, the test
that predates every design decision, loses 2.9 points (in a bull year the filter leaves few
names to short), and it trades 68% more. R2 loses everywhere; R3 flips sign across the tests
again. None adopted (budget 3 of 12 used).

**The worst period, inspected** (`worst.py`, T1 2026-07 → 10-16, alpha −25.2%/yr): the book lost
9.0% against the market's −1.1%; the long leg −8.2 pts, the short leg −1.3. Without the skew
sleeve −21.4% — not the option sleeve. The worst longs: CIFR, TSEM, SMR, AXTI, BE, TTMI, STM, SNDK
— very volatile, very heavily traded crypto / AI / nuclear names that fell together.

- **N3 `long_vol_cap` 0.9 (hypothesis, before its numbers; budget 6 of 12):** no NEW long in a name
  whose 63-bar volatility ranks in the top tenth of the eligible names (a held long is not forced
  out). *Mechanism:* a low days-to-cover can mean "short sellers avoid it" or "everyone trades it";
  the most volatile of the heavily traded names are lottery-like and underperform (Ang, Hodrick,
  Xing & Zhang 2006; Bali, Cakici & Whitelaw 2011), and they crash together when momentum turns.
  *Risk:* it also cuts some of the few huge winners that carry half the alpha (§27: NVDA, CLS).
  *Expected:* T1 Δ near 0 or negative overall but less drawdown in 2026-H2; adopt only on the bar.

**N1-N2 results** (paired against v5):

| | T1 Δ (P), better | T2 Δ (P) | T3 Δ (P) | orders/yr | late fills | adopt |
|---|---|---|---|---|---|---|
| N1 `confirm` 5 | −2.1 (0.06), 1/10 | −1.9 (0.11) | 0.0 (0.51) | 297 | +4.3% | no |
| N2 `dual_band` | −0.8 (0.36), 4/10 | **+2.4** (0.85), 6/10 | −0.6 (0.40) | **131** | **+8.3%** | no |

N1: waiting for a persistent signal just misses part of the move, and does not even cut trading
(the 5-bar score smoothing already filters flicker). The early short loss is not a timing
problem a delay can fix. N2 fails the bar (T1 and T3 below v5) but is the most interesting result
so far: it halves trading (296 → 131 orders a year) and raises the late-fill alpha — the
realistic-cost case for market-on-open orders — by 2.0 points, with T2 +2.4. My expectation that
T3 would not move was wrong: its exit test reads the RAW short-interest rank, not the 5-bar
smoothed score, so holds lengthen before options too. Not adopted under the gross-alpha bar;
reported to the user as a cost choice for live trading (§35).

**N3 result** (`long_vol_cap` 0.9): T1 −2.1 (P 0.29, 4/10), T2 −2.9 (P 0.12, 3/10), T3 −1.1
(P 0.41) — rejected. It does what it was built for: 2026-H2 +14 points, max drawdown −10.9% (from
−14.0%), beta 0.07 (from 0.13). But it gives back 2025 (−14, −8, −12 points over three T1 periods):
the book's alpha and its crash risk come from the same very volatile, heavily traded names.

**Two premise checks that spent no budget** (2022+ trades only, so T2 and T3 stay unread):
- *Volume surges* (20-bar ÷ 126-bar median volume at entry): long trades in the top surge
  quintile did BEST (+3.8% residual per trade). An "abnormal volume" days-to-cover would remove
  winners; not tested.
- *Recent losers as shorts* (21-bar return rank at entry): shorts entered in the bottom quintile
  averaged −0.1% against +1.2% for the rest (55 trades) — too weak for a rule; not tested.

**Step 3 started:** the nested Optuna walk-forward on the refreshed data (80 trials × 10 periods,
dev copy at v5 values, study `informed-flow-ls-dev-loop2`), with the 100 random-name books.

- **N4 `q_out` 0.85 (hypothesis, before its numbers; budget 7 of 12):** leave a long once its rank
  falls below 0.85 (shorts: above 0.15), from 0.8. *Mechanism:* a long sold on the band keeps
  falling — −2.2% residual over the next 21 bars, −5.8% over 63 (§32) — so the decline begins
  while the rank decays from 0.9 to 0.8; leaving earlier avoids part of it. *Cost:* a narrower
  band trades more (N2 showed what fewer trades are worth at late fills). *Expected:* T1 Δ > 0
  with more orders; adopt only on the bar.

**N4 result** (`q_out` 0.85): T1 −0.8 (P 0.37, 4/10), T2 +0.8 (P 0.63), T3 −4.0 (P 0.14, 0/3);
330 orders a year, late fills +4.7% (from +6.3%). Rejected: the post-exit decline is real, but a
narrower band does not capture it — it adds trades and loses the 2020-21 test in every period.

**The foreign-filer question on the refreshed data** (informational, the user's decision; not
counted in the budget): excluding foreign private issuers gives T1 +5.0% (Δ −5.3, P 0.05), T2
+0.5% (Δ −3.7, P 0.10), T3 −0.6% (Δ −3.3, P 0.19) — as in §29, the tilt helps in all three tests.

- **N5 `short_stop` 1.0 (hypothesis, before its numbers; budget 8 of 12):** buy a short back only
  once it is 100% above its entry close (from 50%). *Mechanism:* the 14 stopped shorts lost 29%
  each and then fell back 9% over the next 63 bars (§32) — the stop sells at the squeeze's top, and
  the inner optimizer of the refreshed nested walk-forward picked 0.75-1.0 in each of its first six
  periods (read before this was written: that preference is the motivation, not evidence). The
  crowded-name exit (30% of shares) and the borrow tiers still bound the worst squeezes.
  *Expected:* T1 Δ ≥ 0 with a slightly deeper worst short; adopt only on the bar.

**N5 result** (`short_stop` 1.0): T1 **+12.6%** (Δ **+2.3**, P 0.983, 7/10 periods; per period
+0 +2 +5 +1 +5 −0 +4 +10 +0 +0 — none worse), T2 +4.6% (Δ +0.4, P 0.61, 3/10, mostly unchanged
periods), T3 +3.8% (Δ +1.1, P 0.69, 2/3). Max drawdown unchanged (−14.0%), late fills +8.6% (from
+6.3%), 287 orders a year. **Passes the §31 bar** — the first change of the loop to do so; T2 is
its weak leg.

*Robustness, before their numbers (budget 9-10 of 12; reported, never used to switch, as in §10):*
if the stop really sells at the squeeze's top, `short_stop` 0.75 and NO stop (99) should also beat
0.5 on T1-T3. If either fails badly, 1.0 is a lucky knob setting, not the mechanism, and is not
adopted.

**Stop robustness** (paired against v5):

| `short_stop` | T1 Δ (P), better | T2 Δ (P) | T3 Δ (P) |
|---|---|---|---|
| 0.75 | +1.1 (0.85), 5/10 | +0.5 (0.63) | −0.4 (0.44) |
| **1.0 (N5, pre-declared)** | **+2.3 (0.98), 7/10** | +0.4 (0.61) | +1.1 (0.69) |
| none (99) | +1.3 (0.82), 7/10 | +2.9 (0.95) | −1.4 (0.28) |

Every wider stop beats 0.5 on T1 and T2; T3 (three periods, few stops) is noise either way and
no setting fails it badly. The 50% stop is what costs money — it buys back at the squeeze's top —
so the mechanism, not the knob value, carries the result. 1.0 stays (declared before these).

**N5's tail** (`anatomy.py`, point-in-time universe from 2020-07): squeeze-stop exits fall from 14
to 2; the average short trade from +2.3% to +3.8% residual. The worst single short is deeper:
IBM −65% over 718 bars (a slow climb while it stayed heavily shorted) against v5's worst, CAR
−48%. One short is ~3% of the book, so the worst case costs ~2% of the book over its whole hold;
T1's maximum drawdown is unchanged (−14.0%).

## 34 · Loop 2 decision: v6 = v5 + `short_stop` 1.0

N5 passed the §31 bar (T1 +2.3, P 0.98; T2 +0.4; T3 +1.1), its robustness check holds (every wider
stop beats 0.5 on T1 and T2), and its tail is bounded (above). **Adopted as v6** (config value 1.0,
`signals()` default 1.0, study `informed-flow-ls-1d-v6`). The design phase ends here with 2 of
the 12 variants unspent: every variant read on the same three tests adds selection the gates
cannot see, and the loop's other eight variants all failed.

**T4, the locked test** (names ranked 1001-1500, 500 per formation, $3M floor; read once, now):

| | alpha | P(alpha > 0) |
|---|---|---|
| v5 | +3.3%/yr | 0.79 |
| v6 | +3.6%/yr | 0.82 |

v6 − v5: Δ +0.3 (P 0.53), 4/10 periods better. **The signal itself generalizes** to a third
cross-section nobody looked at (+3.3%, as T2's +4.2% on names 501-1000), but **the stop change
does not carry measurably** to the smallest names: it does no harm there, and its gain is a top-500
gain (T1 +2.3) with a small T2 echo (+0.4). Reported as such.

## 35 · The nested Optuna walk-forward on the refreshed data (Step 3; v5 values, 80 trials × 10)

| arm | CAGR | beta | alpha | P(alpha > 0) | max DD |
|---|---|---|---|---|---|
| tuned (the primary) | +7.7% | 0.13 | +6.6% | 0.92 | −16.5% |
| untuned config (v5) | +11.4% | 0.13 | +10.3% | 0.98 | −14.0% |
| tuned, no options | +9.1% | 0.18 | +7.5% | 0.94 | −14.0% |
| long side + hedge (config) | +14.4% | 0.55 | +9.6% | 0.99 | −14.4% |
| tuned, late fills | +4.0% | 0.13 | +2.9% | 0.72 | −25.8% |

**FAIL**, as in every loop-1 run: tuning costs 3.7 points against the config (gate ≥ 0) and the
median probability of backtest overfitting is 0.74 (gate ≤ 0.5); every other gate passes (beats
99% of the random-name books, beta 0.13, 7/10 periods, all stresses positive). The inner study's
picks wander (dtc_window 63-126, skew_window 20-60, w_skew 0-2) — except `short_stop`, 0.75-1.0 in
9 of 10 periods, the one stable preference, which §34 adopted as a design change after it passed
the independent tests. The optimizer is the right tool for FINDING where the config is weak; it is
not a better config.

## 36 · Loop 2 final state (2026-10-21)

**v6 untuned verdict** (`untuned_verdict.py`, the shipped config, T1): CAGR +13.8%, beta 0.14,
alpha **+12.6%** (P 0.99), max drawdown −14.0%; beats all 100 random-name books (median −0.4%,
95th percentile +4.8%); 2× costs +11.6%, 2× borrow +12.4%, late fills +8.6%; the long side + SPY
hedge +6.8% (the short leg adds 5.8 points); 7 of 10 periods positive. Every applicable gate
passes. Fresh tests: names 501-1000 +4.6% (P 0.88), names 1001-1500 +3.6% (P 0.82), 2020-21 +3.8%
(P 0.71).

**What loop 2 learned.** (1) The data gap had biased one earlier verdict (R1's fresh-names loss),
not the design: re-tested on complete data, nothing reverted earlier changes. (2) The book's alpha
and its crash risk are the same names; every rule that cuts the 2026 drawdown gives back 2025.
(3) The inner optimizer still fits noise, but its one stable preference was a real effect — the
optimizer is useful as a detector of where the config is weak, not as the config. (4) A cost
option exists (N2: half the trades, +2 pts at late fills, −0.8 at normal costs).
The development copy `strategies/informed_flow_ls_dev` was deleted; the loop's drivers and results
are in the session scratchpad (`loop2.py`, `anatomy.py`, `worst.py`, `surge.py`, `reversal.py`).

## 37 · Bear markets, and a period no loop had read (2026-10-21, user question)

**2018-07 → 2020-06** (`bear.py`, v6 as shipped, point-in-time top 500, short interest only): CAGR
**−2.7%/yr** against the market's +10.5%, beta 0.08, **alpha −3.5%/yr (P 0.22)**, max drawdown
−13.4%. Per half-year: H2 2018 −11.5% (market −6.8%), H1 2019 +5.0% (+23.5%), H2 2019 +2.5%
(+9.1%), H1 2020 −0.6% (−2.8%). In the COVID crash the book held flat (February −1.4%, March 0.0%
against the market's −7.6% and −16.8%) and then missed the rebound; in late 2018 it lost on its
own (November −6.2% while the market rose 2.7%). Survivorship is worse this far back.

**Falling markets inside the walk-forward (2022-01 → 2026-10):** in the 25 of 58 months the market
fell, the book averaged +0.5% (market −3.9%) and rose in 60% of them; calendar 2022 +14.2% against
the market's −19.4%; its five worst market months (−6% to −10%): −3.8, +1.4, −0.2, +1.3, +2.1.

**Reading:** a beta of ~0.1 protects in sell-offs (2022, March 2020), but making money in one
depends on the selection working at the time — it did in 2022, not in late 2018. Across eight
years the alpha is concentrated in 2022-2025 (2018-20 −3.5%, 2020-21 +3.8%, 2022-25 strong, 2026
−6.4% YTD while the market rose 10.8%): a real but regime-dependent edge, not a steady one.

## 38P · Proposal 1, three iterations (2026-10-21, a session parallel to loop 3) — pre-registered before any number

**The task** (user, from `STRATEGY_OVERVIEW.md` §6, proposal 1): make the book beta-neutral, test
it on falling markets, then improve it step by step for three iterations, optimization included.

**What is known before any variant runs** (`diag.py`, this session's scratchpad: v6's own weights ×
close-to-close returns per leg, a diagnosis, not the engine's P&L). The *ex-ante* net beta (Σ
weight × trailing 126-bar beta to the equal-weight members) swings from −0.08 to +0.54 between
periods, although the realized beta of the whole walk-forward is 0.13:

| period | market | long leg | short leg | ex-ante beta L / S / net | volatility rank L / S |
|---|---|---|---|---|---|
| 2018-H2 | −6.4% | −11.0% | +0.6% | 0.62 / −0.43 / **0.19** | 0.68 / 0.54 |
| 2022-H1 | −23.1% | −10.4% | +16.4% | 0.58 / −0.49 / 0.10 | 0.64 / 0.50 |
| 2025-H2 | +8.3% | +25.2% | +1.1% | 0.80 / −0.41 / **0.39** | 0.79 / 0.41 |
| 2026-H2 | −1.2% | −7.3% | −1.3% | 0.81 / −0.26 / **0.54** | 0.61 / 0.39 |

Low days-to-cover selects heavily traded, high-beta, volatile names; high days-to-cover selects
quiet, low-beta ones. So a dollar-neutral book is long beta (most of all when the long end is a
momentum crowd, 2025-26), and a sell-off led by high-beta names hits the long leg harder than the
short leg protects (2018-H2: the long leg alone lost 4.6 points more than the market).

**Tests** (untuned config values, every variant paired against the current base, same days;
TB is loop 3's T5 below, TL is not its T6; this work runs in `strategies/informed_flow_ls_p1`):
- **TB** 2018-07 → 2020-06 (4 periods; the point-in-time top 500 is nearly every name with daily
  bars then: ~490). Read once for v6 (§37): from now on development data. No fresh cross-section
  exists for these years.
- **T1, T2, T3** as in §31 (top 500 2022-26; names 501-1000; 2020-07 → 2021-12).
- **TL (LOCKED)**: the names ranked 1501-2000 by dollar volume (2022-26 periods, $1M floor). Read
  ONCE, at the end, for the final candidate against v6.

**The bar** (the goal is robustness in falling markets at about the same alpha; written now):
- *bear:* TB Δ ≥ 0 AND the book's mean return over the market's down months (TB + T1 calendar
  months in which the equal-weight market fell) not lower than the base's;
- *cost:* T1 Δ ≥ −1.0 pt, T2 Δ ≥ −1.0 pt, T3 Δ ≥ −1.0 pt (the price of insurance is capped);
- a mechanism written before its numbers. **Budget: 12 variants over the three iterations.**

**Each iteration:** (1) hypothesis + mechanism here, (2) the paired screen on TB and T1-T3, (3) the
optimizer as a detector: the nested Optuna walk-forward (80 trials × 10 periods, the config's
`search:` knobs including the iteration's new ones) — tuned vs untuned, PBO, and which values the
inner studies prefer stably (loop 2's lesson: a stable preference is worth a design test; the
pick itself is not a config). The next iteration is written from what (2) and (3) showed.
**At the end:** the final candidate's untuned verdict (100 random-name books, stresses), its
bear table by episode (Q4 2018, COVID, 2022, Feb-Apr 2025), and TL.

**Iteration 1 — I1 `sizing: beta` (the proposal).** Each position's size is set at entry from its
trailing beta (126 bars, to the equal-weight eligible names): slot × clip(median beta / beta,
1/1.5, 1.5), frozen while held (no re-trading), with a gross guard (a day's new entries are scaled
down if the book's nominal gross would pass `gross`). *Mechanism:* every position carries about
the same market risk, so the long leg's high-beta names are smaller and the short leg's low-beta
names larger: ex-ante beta near 0 on every bar instead of only on average, and the "lottery"
names that crash together are smaller (the parked inverse-volatility idea, IDEAS.md). *Expected:*
|ex-ante net beta| ≤ 0.1 in every period; TB Δ > 0 (2018-H2); T1 Δ slightly negative (2025's
high-beta winners held smaller) but above −1.0. *Control* (I1-static, budget 2): the static
`long_share` 0.43 of loop 1 (H5) on the refreshed data — is a dynamic split better than a fixed one?

**I1 result** (`harness.py`, paired against v6; v6 reproduces §34-§37 exactly: T1 +12.6%, T2
+4.6%, T3 +3.8%, TB −3.5%):

| | v6 | I1 `sizing: beta` | I1-static `long_share` 0.43 |
|---|---|---|---|
| TB alpha (Δ) | −3.5% | −3.2% (+0.3, 2/4) | −3.6% (−0.2) |
| T1 alpha (Δ, P) | +12.6% | +11.5% (−1.1, 0.27) | +11.4% (−1.2, 0.03) |
| T1 ex-ante beta: mean (max \|.\|) | +0.14 (1.00) | +0.06 (0.59) | +0.00 (0.81) |
| T1 max DD / volatility | −14.0% / 10.7% | **−8.7% / 8.6%** | −12.9% / 9.9% |
| T2 / T3 alpha (Δ) | +4.6% / +3.8% | +2.9% (−1.6) / +2.8% (−1.0) | +3.2% (−1.4) / +5.1% (+1.3) |
| down months (32): mean, share > 0 | +0.03%, 53% | +0.19%, 56% | +0.59%, 59% |

**Neither passes the bar** (I1: T1 −1.1 and T2 −1.6 beyond the −1.0 cap; the static split: T1,
T2, TB). What I1 does is risk, not return: alpha per unit of volatility 1.18 → 1.34, the worst
drawdown −14.0% → −8.7%. And it does NOT repair TB (+0.3): the 2018-20 loss is not beta.

**Why TB lost — two inspections that spent no budget:**
- *Factor anatomy* (`factors.py`: daily returns regressed on the market, VOL = high minus low
  63-bar volatility, MOM = 12-1, TURN = high minus low turnover, all from the test panels):
  pooled loadings VOL +0.30, TURN −0.24, MOM +0.08. T1's +64.6 summed points split into +50.7
  residual, +27.8 VOL, −20.2 TURN, +9.5 MOM; TB's −5.0 into −4.6 residual and −1.8 VOL. The 2018
  loss is the selection itself, almost all in 2018-H2 (alpha −19.7%/yr; the other three TB
  periods +3.4, +1.5, −1.0), all in the long leg (−11.3 pts; shorts +0.8): EDU, BTI, PCG, FCX, SU,
  META, WDAY, AMAT. Short-interest coverage was complete then (~420 names).
- *Foreign filers out* (proposal 1's third item, the user's call; loop 2's X): TB −3.9% (Δ −0.4),
  T1 +5.4% (Δ −7.2), T2 +1.2% (Δ −3.4), T3 −3.6% (Δ −7.4), down months −0.35%. The tilt did not
  cause 2018 and helps everywhere else.

**Iteration 2 — I2 `w_sishare` 1 (hypothesis, before its numbers; budget 3 of 12).** A second
short-interest sleeve beside days-to-cover: short interest as a share of shares outstanding
(the same 42-bar average), equal weight. *Mechanism:* days-to-cover divides short interest by
trading volume, so its low end is the most heavily traded, high-beta names; short interest ÷
shares has no volume in it, and its high end is the crowded high-beta shorts — the two
measurements of one thing (what short sellers hold) carry OPPOSITE beta confounds. A premise check
that read no returns: the two ranks correlate only 0.24-0.59; the betas of the names in the long
/ short tenths are, for days-to-cover, 1.11/0.94 (2018-H2), 1.18/0.84 (2025-H2), 2.03/0.48
(2026-H2); for short interest ÷ shares 0.94/1.23, 0.69/1.81, 0.62/2.56; the blend 0.96/1.03,
0.70/1.14, 0.84/1.03. Loop 1's H3 (short interest ÷ shares INSTEAD of days-to-cover) read T1 Δ
+1.4 (P 0.61) at beta −0.41. *Expected:* the selection's confound shrinks without any sizing
rule: ex-ante beta near 0 in the periods where v6's was highest (2025-26), TB Δ ≥ 0 (2018's long
leg was the high-beta crowd), T1 Δ ≥ −1.0. *Robustness, before its numbers (budget 4-5,
reported, never used to switch):* `w_sishare` 0.5 and 2.0 — if the mechanism is real the effect
grows from 0.5 to 1 and does not need the measure to dominate.

**I2 result** (paired against v6; robustness weights reported, not used to switch):

| | v6 | I2 `w_sishare` 1 | 0.5 | 2.0 |
|---|---|---|---|---|
| TB alpha (Δ) | −3.5% | −5.5% (−2.0) | −5.6% (−2.2) | −11.0% (−7.6) |
| TB Δ by period | | +9 −1 +2 −15 | +11 +5 +5 −23 | +10 −5 −7 −24 |
| T1 alpha (Δ) | +12.6% | +9.8% (−2.9) | +8.1% (−4.5) | +8.0% (−4.6) |
| T1 beta / ex-ante beta | 0.14 / +0.14 | −0.21 / −0.27 | −0.16 / −0.19 | −0.30 / −0.37 |
| T2 / T3 alpha (Δ) | +4.6% / +3.8% | −2.7% (−7.2) / +2.1% (−1.7) | −1.9% / +0.5% | −0.6% / +1.6% |
| down months: mean, share > 0 | +0.03%, 53% | **+1.35%, 62%** | +1.05%, 62% | +1.56%, 75% |

**Rejected, all three.** The mechanism's first half is real — the opposite confound overshoots:
the blend runs net SHORT beta (T1 −0.21), so the falling months improve a lot (+1.35% a month) —
but the second half fails: short interest ÷ shares is not a cleaner selector. It is a crowding /
beta bet: it helps the sell-offs (2018-H2 +9 to +11 pts in all three weights) and loses the
rebounds (2020-H1 −15 to −24, 2025-H2 −49 to −64). Days-to-cover stays the only selector.

**Three premise checks for iteration 3 (no budget; in-sample data, no variant run):**
- *Sector concentration* (securities.sector): the largest sector holds 3-6 of 15 names a side
  (2018-H2: 3.2) — no concentration to cap.
- *A drawdown stop for longs* (v6's own longs, residual return AFTER the first close 15/25/35%
  below the best close since entry, to the real exit): T1 −0.7% (n 73) / +0.7% (22) / +4.7% (5);
  only TB's few (n 8 / 5) kept falling. A long stop would be fitted to 2018. Not tested.
- *Volatility management* (the book's own trailing 63-bar volatility → next month, 98 months
  2018-07 → 2026-10): correlation +0.06 (the wrong sign), the calmest third of months the WEAKEST
  (−0.14% a month), and the worst months (2018-10, 2018-11) came after calm stretches (vol 6-8%).
  Scaling down after volatile stretches would only shave the mean. Not tested.
- *The 200-day filter's empty short slots* (loop 2's R1 `short_below_ma` 200; `_select` alone,
  no returns): the short side holds on average 6.1 of 15 names in 2018-H2, 9.9 / 10.3 in
  2020-H2 / 2021-H1, 10.1 in 2024-H1, 12-15 elsewhere. R1's 2020-21 failure (§33: T3 −2.9) is the
  book drifting net LONG when few names are in downtrends — and 2018-H2, the bear test, is where
  the side would have been emptiest.

**Iteration 3 — I3 `short_below_ma` 200 + `hedge` SPY (hypothesis, before its numbers; budget 6
of 12).** Short a name only while it is crowded AND below its 200-day average (loop 2's R1, bought
back above it), and hold every EMPTY short slot as an SPY short of the same size (the index is
never ranked; it only fills). *Mechanism:* in a falling market the money is in shorting crowded
names that are already falling — the trend shorts that made `trend_following`'s 2022 (+28% of its
book) — while a crowded name in an uptrend is the squeeze / rebound risk that costs the short leg
6 years in 7. The filter alone left the book net long whenever few names were in downtrends
(above); the index fill keeps it dollar-neutral and close to beta-neutral without shorting
uptrends. *Expected:* T1 Δ > 0 (R1 alone +4.0), T3 Δ ≥ 0 (R1's −2.9 repaired), TB Δ ≥ 0 (2018-H2
starts with 9 index slots that gain in the Q4 fall), down months ≥ v6. *Controls, before their
numbers (budget 7-8, reported, never used to switch):* R1 alone on the same panels (is the fill
what repairs it?) and the index fill alone with the filter at 100 bars (dose: a shorter average
leaves fewer empty slots).

**I3 result** (paired against v6: I3 ranks without the index, so its selection is v6's; R1 alone
re-run on the same panels as v6):

| | v6 | I3 MA200 + SPY fill | R1 MA200 alone | I3 MA100 + SPY fill |
|---|---|---|---|---|
| TB alpha (Δ) | −3.5% | −7.1% (−3.6) | −4.3% (−0.8) | −10.7% (−7.2) |
| T1 alpha (Δ, P) | +12.6% | +13.3% (+0.7, 0.58) | +14.3% (+1.7, 0.69) | +11.1% (−1.5) |
| T2 / T3 alpha (Δ) | +4.6% / +3.8% | +3.0% (−1.6) / −0.1% (−3.9) | +3.8% (−0.8) / −1.1% (−4.9) | +3.4% / −0.2% |
| T1 orders/yr, late-fill alpha | 287, +8.6% | 531, +5.0% | 497, +7.0% | 574, +2.1% |
| down months | +0.03% | −0.01% | +0.12% | −0.41% |

**Rejected, all three.** The index fill does not repair the 200-day filter: it costs TB 2.8 points
against R1 alone (the cap-weighted SPY is a poorer short than the equal-weight market the alpha is
measured against) and gives back R1's T1 gain; both double the trading and lose at late fills.

**The most useful number of the iteration is a control's.** v6 on the SAME panels with SPY merged
in — SPY never traded, it only shifts the dollar-volume groups by one name — reads T3 −0.1% against
+3.8% (Δ −3.9, per period −2 / +0 / −9), TB +1.0, T1 −0.2, T2 +0.2. **The book is path-sensitive:
a one-name perturbation of its groups moves the three-period T3 by ±4 points.** Every T3 verdict
in this loop and in loops 1-2 (R1's −2.9, R3's +3.7, I1's −1.0, I3's −3.9) is inside that noise;
T1, T2 and TB move ±0.2-1.0 under the same perturbation.

**The optimizer as a detector, iteration 1** (`wf.py`, nested Optuna walk-forward, 80 trials × 10
periods, I1's space: the four `search:` knobs + `beta_window` + `max_size`, 1.0 = equal sizing):
the inner studies chose `max_size` 1.0 in 6 of 9 periods (1.5, 1.75, 1.75, 2.0 otherwise) — no
stable wish for beta sizing — and `short_stop` 0.75-1.0 (as loop 2). The one stable preference:
**`dtc_window` above 42 in 9 of 9 periods** (63 ×5, 84 ×3, 105 ×1), PBO 0.36-0.90.

**O1 — `dtc_window` 63 (hypothesis, before its numbers; budget 9 of 12; the optimizer's design
test, as loop 2's `short_stop`).** *Mechanism:* FINRA updates short interest twice a month;
the information is in the persistent level (§6, §17), and 42 bars average about three
settlements, 63 about four and a half — a slower level that keeps names in their zone longer
(fewer band exits; late fills should improve as for `dtc_window` 42 in informed_flow v34).
*Prior evidence is mixed:* 84 flipped sign across T1-T3 twice (§29, §33), 126 failed. *Expected:*
T1 Δ ≥ 0 with fewer orders, TB Δ ≥ 0. *Robustness (budget 10, reported, not used to switch):*
`dtc_window` 84 on this harness, TB included.

**O1 result:** `dtc_window` 63: TB −6.1% (Δ −2.6), T1 +9.3% (Δ −3.3, P 0.09, 2/10), T2 Δ −0.8, T3
Δ +0.2; 84: TB Δ −2.5, T1 Δ −1.7 (6/10), T2 Δ −1.8, T3 Δ +1.4. **Rejected.** The inner studies'
unanimous wish for a slower average was in-sample fitting (every inner study trains on windows
from 2020-07 that overlap the next): this time the detector found nothing — `short_stop` was
real, `dtc_window` is not. Budget: 10 of 12 spent; no further variant.

**I1's nested walk-forward** (the I1 space, 80 trials × 10): tuned alpha +4.9% (P 0.86) against
the untuned I1 config +11.5% (P 0.997): tuning costs 6.7 points, median PBO 0.74 — FAIL, as in
every earlier run. Late fills on the tuned arm: +1.7%.

**The bear table** (`episodes.py`, the continuous record TB + T3 + T1, 2018-07 → 2026-10-16; the
book's return over each episode of section 2 of STRATEGY_OVERVIEW.md):

| | Q4 2018 | COVID crash | 2022 bear | Feb-Apr 2025 | 2026-H2 | max DD | CAGR |
|---|---|---|---|---|---|---|---|
| equal-weight market | −18.2% | −37.0% | −25.2% | −19.6% | −1.1% | | |
| SPY | −18.7% | −33.4% | −24.1% | −18.8% | +3.4% | | |
| **v6** | −10.8% | −1.5% | +10.4% | −4.8% | −9.0% | −14.0% | +9.3% |
| I1 beta sizing | −10.0% | −1.7% | +12.7% | −2.7% | −6.0% | −12.9% | +7.3% |
| I2 `w_sishare` 1 | −0.6% | −0.7% | +23.5% | +9.8% | −6.9% | −21.1% | +3.2% |
| I3 MA200 + SPY | −11.8% | −4.8% | +20.6% | −4.6% | −12.0% | −14.8% | +8.2% |

v6 already loses far less than buy-and-hold in every falling market of the data and gains in
2022; its "2018 failure" (§37) is losing 60% as much as the market in Q4 2018 and then missing the
2019 rally — the 2018-20 alpha, not a loss beyond the market's. Its real failure is 2026-H2, a
factor crash in a RISING market. Every variant that protects more in the sell-offs (I2, I3)
pays for it in the rebounds (CAGR 3.2%, 8.2%); I1 protects a little more in four of five
episodes at 2 points of CAGR.

**The final test, declared before its numbers.** No iteration passed the §38P bar, so v6 stays
the shipped config. The candidate for the full test is **I1 itself** — the proposal as designed,
closest to the bar (T1 −1.1 against −1.0) and the one change with a clear risk benefit: its
untuned verdict on T1 (100 random-name books holding random names with the SAME beta sizing,
seed 29; 2× costs, 2× borrow, late fills; the long side + SPY hedge) and **TL, read once now:
names ranked 1501-2000 (2022-26 periods, $1M floor), I1 against v6**. Neither changes the
decision above; they say whether beta sizing is a sound OPTION (a risk choice, like N2 a cost
choice) on data nobody looked at.

**The final test** (`verdict_p1.py`: the untuned verdict on T1 with 100 random-name books that keep
the config's OWN sizing; the v6 run reproduces loop 2's band exactly — median −0.4%, 95th
percentile +4.8% — so the luck function is the same):

| on T1 2022-01 → 2026-10-16 | v6 | I1 `sizing: beta` |
|---|---|---|
| alpha (P) / beta | +12.6% (0.99) / 0.14 | +11.5% (0.997) / **0.05** |
| volatility / max drawdown | 10.7% / −14.0% | **8.6% / −8.7%** |
| random-name books: median, 95th pct; beaten | −0.4%, +4.8%; 100% | −0.2%, +3.4%; 100% |
| 2× costs / 2× borrow / late fills | +11.6% / +12.4% / +8.6% | +10.6% / +11.3% / +7.5% |
| long side + SPY hedge (beaten by) | +6.8% (5.9) | +5.8% (5.7) |
| periods with alpha > 0 | 7/10 | 7/10 |
| **TL (LOCKED, read once): names 1501-2000** | **+12.2% (P 0.99), DD −12.7%** | **+12.1% (Δ −0.1, P 0.43), DD −9.6%** |

Every applicable gate passes for both. On data no loop had read, the signal holds in a third
fresh cross-section (+12.2%; the 2 bp spread flatters names trading $1-8M a day) and beta sizing
costs nothing there (−0.1) while cutting the drawdown by a quarter.

## 38P · Decision (2026-10-21)

- **v6 stands as this loop's reference** (loop 3 has since shipped v7 = v6 + recalibrated
  borrow tiers + separate long/short counts at 15/15; its T1 differs from v6 by +0.1). No
  iteration passed the §38P bar: I1 (beta sizing) missed the cost caps by 0.1 (T1) and 0.6 (T2);
  I2 (short interest ÷ shares) and I3 (200-day short filter + SPY fill) failed decisively; the
  optimizer's one stable preference (`dtc_window` > 42) failed as O1; the nested Optuna
  walk-forward failed again (tuning −6.7 pts, PBO 0.74).
- **`sizing: beta` is a documented RISK option, not an edge:** about −1 pt of alpha on T1-T3 (0
  on TB and TL) for a third less drawdown and a fifth less volatility, ex-ante beta near 0 on
  every bar (the largest |ex-ante beta| 1.00 → 0.59), and a better result in 4 of 5 falling-market
  episodes. The implementation (`_weights`: sized at entry, frozen while held, a gross guard) is
  in `strategies/informed_flow_ls_p1` with tests (`tests/test_informed_flow_ls_p1.py`); folding it
  into the shipped strategy is the user's call.
- **What the loop learned:**
  1. The book's ex-ante beta swings −0.08 → +0.54 between periods because low days-to-cover
     selects the heavily traded high-beta crowd; its realized 0.13 hides that.
  2. v6 already loses far less than buy-and-hold in every falling market in the data (Q4 2018
     −10.8% vs −18.2%; COVID −1.5% vs −37.0%; 2022 +10.4% vs −25.2%; Feb-Apr 2025 −4.8% vs
     −19.6%). The 2018-20 shortfall is selection (2018-H2's long leg), not beta, not foreign
     filers, not sector concentration; its true failure is a factor crash in a RISING market
     (2026-H2 −9.0%).
  3. Every rule that protects more in sell-offs pays it back in the rebounds (I2: CAGR 3.2%,
     I3: 8.2%, against v6's 9.3% over 2018-26).
  4. **The book is path-sensitive:** shifting its dollar-volume groups by one name moves the
     three-period T3 by ±4 points. A bar on T3 alone (here, and §29-§33) reads noise.
  5. The optimizer's stable preferences can be real (`short_stop`, loop 2) or not
     (`dtc_window`, here): only the independent tests can tell.
  6. Volatility management has no premise here: the book does worst after CALM stretches.

Drivers and results stay in this session's scratchpad (`p1/`: `harness.py`, `diag.py`,
`factors.py`, `wf.py`, `verdict_p1.py`, `episodes.py`, `longstop_premise.py`).

## 38 · Loop 3 (2026-10-21, user request): robustness — more stocks, separate long/short counts,
## IBKR borrow rates. Pre-registered before any number.

**Goal:** a book less dependent on a few names and on one regime — not more headline alpha.

**Tests** (all but the last read before; every number below is in-sample to some degree):
T1 top-500 walk-forward 2022-26 (10 periods), T2 names 501-1000 (10), T3 2020-21 (3), T4 names
1001-1500 (10), T5 top 500 2018-07 → 2020-06 (4). **T6 (new, LOCKED): names 501-1000 in
2018-07 → 2020-06** (4 periods), read once at the end for the final candidate against v6.

**Robustness metrics** (v6 is the reference):
- M1 the mean alpha over T1-T5 (each test's stitched alpha, weighted equally);
- M2 the share of positive periods over all 37;
- M3 leave-top-5-out: T1's alpha after removing the five biggest contributors' P&L;
- M4 the worst of the five test alphas; M5 T1's maximum drawdown.

**"More robust"** = (M3 ≥ v6 + 1.0 pt OR M2 ≥ v6 + 5 percentage points) AND M1 ≥ v6 − 1.0 pt AND
no test's paired Δ worse than −2.0 pts. Budget: 8 variants.

**Variants:**
- S1 / S2: 20 / 30 positions a side (from 15): diversification against the five-name dependence.
  Loop 1 found 20 worse on T1 alone ("the edge sits in the extremes"); this asks a different
  question (robustness, five tests).
- S3: 15 longs / 30 shorts; S4: 30 longs / 15 shorts (new `long_slots` / `short_slots`; each side
  keeps its dollar share, so per-name sizes differ). S3's logic: the short edge per name is small
  and squeezes are idiosyncratic, so more shorts hedge the same dollars with less single-name risk.
- U1: the universe widened to the top 1000 (ranked like T2's band, $10M floor), 30 a side; judged
  on T1/T3/T5 re-run on the wider universe (T2 and T4 overlap it).

**Borrow (a cost model, not a variant):** IBKR's live file (2026-10-21, 2,213 DB names) charges a
median 0.40%/yr; its 90th percentile by short-interest tier: < 10% of shares 0.44-0.67%, 10-20%
0.47-0.55%, 20-30% 1.40%. The model charged 0.5% / 3% / 10%. **Recalibrated, conservatively** at
about twice today's 90th percentile: 0.75% / 1% / 3% (one calm snapshot; 2021's fees for crowded
names were far higher, and the 2× borrow stress stays). For live and paper trading the actual
per-name IBKR fee overrides the tiers (a `borrow_fee` feature from the daily IBKR file).

**Batch 1** (`loop3.py`; every variant on the recalibrated borrow tiers; v6c = v6 + those tiers):

| | v6c (15/15) | S1 20/20 | S2 30/30 |
|---|---|---|---|
| T1 alpha | +12.7% | +7.9% (Δ −4.7, P 0.00) | +7.6% (Δ −5.1, P 0.02) |
| M3 T1 w/o top 5 | **+6.8%** | +3.6% | +4.8% |
| T2 / T3 / T4 / T5 | +4.8 / +3.9 / +3.8 / −3.3 | +2.9 / −1.4 / +5.5 / −4.9 | +4.3 / −1.9 / +4.4 / −5.3 |
| M1 mean / M2 periods > 0 / M4 worst | +4.4% / 65% / −3.3% | +2.0% / 60% / −4.9% | +1.8% / 57% / −5.3% |
| T1 max DD / orders/yr / names held | −14.0% / 287 / 67 | −15.4% / 393 / 91 | −12.9% / 619 / 137 |

The recalibrated borrow barely moves v6 (T1 +12.7% against +12.6%: most shorts already sat in the
cheapest tier). More positions per side are worse on EVERY robustness metric, not only alpha:
even with the five biggest winners removed, alpha falls (+6.8% → +3.6-4.8%). The edge sits in the
most extreme names; adding names dilutes it rather than spreading it. S1 and S2 rejected.

- **U2 (hypothesis, before its numbers; budget 6 of 8):** the top-1000 universe with 15 a side.
  *Mechanism:* if the edge sits in the extremes, a twice-larger universe offers more extreme names
  at the same concentration (15 of the top 1.5% instead of the top 3%). Judged like U1 (T1/T3/T5
  on the wider universe).

## 39 · Loop 3, batch 2: separate long and short counts; the borrow model

| | v6c (15/15) | S3 15 longs / 30 shorts | S4 30 longs / 15 shorts | v6 (old tiers) |
|---|---|---|---|---|
| T1 alpha (Δ, P) | +12.7% | +11.2% (−1.5, 0.19) | +9.1% (−3.6, 0.02) | +12.6% (−0.0) |
| M3 T1 w/o top 5 | **+6.8%** | +5.6% | +5.5% | +6.8% |
| T2 / T3 / T4 / T5 | +4.8 / +3.9 / +3.8 / −3.3 | +4.8 / +0.1 / +3.0 / −3.6 | +4.1 / +2.1 / +6.1 / −4.8 | +4.6 / +3.8 / +3.6 / −3.5 |
| M1 / M2 / M4 | +4.4% / 65% / −3.3% | +3.1% / 62% / −3.6% | +3.3% / 62% / −4.8% | +4.2% / 62% / −3.5% |
| T1 beta / orders/yr | 0.14 / 287 | 0.09 / 442 | 0.14 / 463 | 0.14 / 287 |

Neither split is more robust: each dilutes one side's extreme names (S3 lowers beta to 0.09 but
gives up 2020-21; S4 helps only T4). **15/15 stays.** The recalibrated borrow tiers beat the old
ones by 0.1-0.2 points on every test (P(Δ ≤ 0) ≈ 0 for the old ones): a small, consistent cost
correction.

**Built into the shipped strategy (v7)** whatever the counts: `long_slots` / `short_slots`
replace `slots` (both 15; notebooks, luck books and tests follow); the borrow tiers are
0.75% / 1% / 3%; a `borrow_fee` feature overrides them per name where present.
`scripts/ibkr_shortstock.py` saves IBKR's file daily to `var/ibkr/usa-YYYYMMDD.txt`;
`research.load` attaches the newest snapshot's fees for 5 days (`research.borrow_fees`), so the
paper book (walk-forward notebook §7) charges IBKR's own rate and says so per short. History
before 2026-10-21 has no snapshot and keeps the tiers.

## 40 · Loop 3, batch 3: the wider universe — and a sector cap

| | v6c (top 500, 15/15) | U1 top 1000, 30/30 | U2 top 1000, 15/15 |
|---|---|---|---|
| T1 alpha (Δ, P) | +12.7% | +10.1% (−2.7, 0.20) | +10.4% (−2.4, 0.23) |
| T3 2020-21 | +3.9% | **−11.4%** (−14.7, 0.00) | **−13.3%** (−16.8, 0.00) |
| T5 2018-20 | −3.3% | −8.2% (−4.8) | −5.9% (−2.5) |
| M2 periods > 0 (T1/T3/T5) | 65% | 41% | 41% |
| M3 T1 w/o top 5 / max DD | +6.8% / −14.0% | +6.5% / −9.6% | +3.8% / −12.3% |

(T2 and T4 overlap the wider universe and are not run; the U books are paired on v6c's days and
market.) The wider universe collapses in 2020-21 — the smaller names of the meme years — and loses
everywhere else: the signal works best among the most liquid names. **Adding stocks, in every
form tried (more slots, uneven sides, a wider universe), made the book less robust.**

- **V7 sector cap 3 (hypothesis, before its numbers; budget 7 of 8):** no side holds more than 3
  names of one sector (securities.sector, 11 sectors, 772 of 780 universe names); a capped
  candidate is skipped for the next most extreme name of another sector. *Mechanism:* the book's
  concentration is THEMATIC (2026-H2: crypto / AI / nuclear longs fell together; 2025: gold
  miners and AI) — a cap spreads the themes while keeping 15 extreme names, unlike sector-neutral
  ranking (loop 1, H28, rejected), which changes the ranks themselves. *Expected:* M3 and M2 up,
  max drawdown down, alpha about the same.

**V7 result** (sector cap 3): T1 +12.8% (Δ +0.1, P 0.54) with max drawdown **−10.7%** (from
−14.0%) and beta 0.11; M3 +7.2% (+0.4, bar +1.0); but T3 **−2.6%** (Δ −6.5, P 0.04), T5 −6.1%
(Δ −2.8), T2 −0.8, T4 +1.5; M1 +2.7% (from +4.4%), M2 62% (from 65%). **Rejected** by the §38
rule: it calms 2022-26 at the same alpha but loses 2018-21, when the winning extremes sat in few
sectors. A risk lever (like N3's volatility cap), not a robustness gain.

**Loop 3 conclusion (7 of 8 variants):** nothing made the book more robust by the pre-registered
definition. More names, uneven sides, a wider universe and a sector cap all trade the extreme
names' edge for breadth, and lose it. The final candidate is v7 = v6 + the IBKR-calibrated
borrow tiers (a cost correction, +0.1-0.2 points per test), with the per-side counts and the
per-name IBKR fee as built-in capabilities.

**T6 could not be formed:** before 2020 the database holds only 483-499 names that pass the
liquidity rules (100 sessions, $10M a day, $5) — there is no band ranked 501-1000 for 2018-2020.
The same fact qualifies T5: its "top 500" was every liquid name the database has for those years,
all of them survivors. Loop 3 therefore ends without a locked test; its one adopted change is a
cost correction, not a selection.

## 41 · Loop 4 (2026-10-21, user question): a trailing stop against strong losses — pre-registered

**The question:** can a moving stop cut the book's strong losses (2026-H2: the long leg −8.2 pts as
crypto / AI / nuclear longs fell 30-60% together; T1 max drawdown −14.0%) without giving up the
edge? *For:* the book leans on momentum, and a stop-loss tames momentum crashes (Han, Zhou & Zhu
2016). *Against:* the 50% short stop sold squeezes at their top (§32), names that just fell tend
to bounce (the one-month reversal), and half the alpha comes from a few winners (NVDA, IREN, CLS)
that had 30%+ pullbacks along the way.

**Rule:** a long is sold once its close falls a set distance below its highest close since entry
(a short: rises that far above its lowest close since entry), at the next open like every exit; a
stopped name is not re-entered until its rank has left the entry zone once (as the squeeze stop).

**Variants** (on v7; budget 5): **L15 / L20 / L30** — longs, 15% / 20% / 30% from the peak;
**LV2** — longs, a distance scaled to each name's own volatility, 2 × (63-bar daily volatility ×
√21), i.e. two typical monthly moves (≈ 18% for a 2%-a-day stock, ≈ 45% at 5% a day);
**LS25** — both sides, 25%.

**"Fewer strong losses" (decided before the numbers):** T1 max drawdown at least 2 points
shallower AND T1's worst period (2026-H2, −25.2%) better AND the mean alpha over T1-T5 at most 1.0
point lower AND no test's paired Δ below −2.0 points. Reported in full either way.

**Loop 4 results (trailing stops, paired against v7):**

| | T1 alpha (Δ) | max DD | worst period | M1 | adopt |
|---|---|---|---|---|---|
| v7 | +12.7% | −14.0% | −25.2% | +4.4% | — |
| L15 longs 15% | +5.7% (−7.0) | −14.6% | −34.8% | +2.2% | no |
| L20 longs 20% | +7.4% (−5.2) | −15.4% | −33.2% | +3.0% | no |
| L30 longs 30% | +8.1% (−4.6) | −14.2% | −33.0% | +3.2% | no |
| LV2 longs, 2 monthly moves | +10.0% (−2.6) | −14.0% | −25.2% | +4.1% | no |
| LS25 both sides 25% | +6.9% (−5.8) | −14.3% | −33.3% | +1.2% | no |

Every trailing stop fails "fewer strong losses": none shallows the drawdown and the fixed ones make
the worst period WORSE — the stop sells a crashing long, the slot refills from the same zone of
very volatile names, which crash too; and they sell the big winners (NVDA, CLS, PBR, AMD leave the
top contributors). Position-level stops are not the lever for this book.

## 42 · Loop 5 (2026-10-22, user: "think out of the box, find hidden correlations") — pre-registered

A multi-agent search: six discovery agents with different lenses (downside / exposure management;
short-interest construction; option-derived information; interactions with other signals and data;
portfolio construction and timing; regimes) each examine the data (2022+ top 500 only, to keep the
other tests unread by the search) and propose mechanism-backed hypotheses; a selector removes
everything already tried (§0-§41) and keeps up to 14; each is implemented in its own research copy
(`strategies/informed_flow_ls_x<id>`, deleted afterwards), checked for defaults == v7 and causality
on a synthetic panel, and run on T1-T5 against v7 (run name v6c).

**Bars (decided now, applied by the workflow's code, not by the agents):**
- ALPHA: T1 Δ ≥ 0 with P(Δ > 0) ≥ 0.7, T2 Δ ≥ 0, T3 Δ ≥ 0, T4 Δ ≥ −1.0, T5 Δ ≥ −1.0, M1 ≥ v7's.
- RISK ("fewer strong losses"): T1 max drawdown ≥ 2 pts shallower AND T1 worst period better AND
  M1 ≥ v7 − 1.0 AND every test Δ ≥ −2.0.
- A variant passing either bar is then (a) re-run at two neighbouring settings, each of which must
  keep T1 Δ ≥ 0 and M1 ≥ v7 − 0.5 (alpha) or a ≥ 1-pt shallower drawdown (risk); (b) audited by two
  independent agents (look-ahead / leakage; implementation and realism) — any real defect kills it;
  (c) run ONCE on the new locked test **T7: names ranked 1501-2000, 2022-26** ($1M floor; 383-500
  names a formation; never read), where it must not lose more than 0.5 pt against v7.
- With ~14 variants, about one null variant is expected to pass the first bar by luck; the
  neighbour, audit and T7 stages exist for that. Every variant and every number is reported.

**Addendum (user, before any test ran): every hypothesis also gets a wide optimization.** So that no
idea is rejected for want of the right setting, each hypothesis's knobs get WIDE `search:` ranges
(from off to well beyond the plausible extreme; the off value always inside) and a **nested
walk-forward Optuna study over T1's 10 periods** (`opt5.py`): 100 trials a period for one knob, 150
for two, 200 for three or more (TPE), each period's study seeing only its past (robust objective),
trial 0 always the off setting (v7) as the landscape baseline, the pick trading the unseen period
beside v7 on the same days. A wide search judged on the data it searched would always find a
winner; judged only on the periods it did not see, it cannot.
- **NESTED bar:** tuned − v7 stitched alpha Δ ≥ 0 with P(Δ > 0) ≥ 0.8 AND median PBO ≤ 0.6.
- A hypothesis passing the NESTED bar but not a fixed bar at its declared setting is re-run at its
  **consensus setting** (the median of the 10 picks; the mode for categorical knobs) on T1-T5 and
  must then pass the ALPHA or RISK bar; it skips the neighbour stage (the nested study is its
  robustness check) but not the audits or T7.
- Every hypothesis's landscape (per period: the off score, the best and median trial, the share of
  trials beating off) is reported, passing or not.

**A look-ahead found by the search (discovery agent, confirmed by hand 2026-10-22).** The crowding
measure `si_share = days-to-cover × the panel's 20-bar median volume / SEC shares` mixes units: the
stored daily volumes are split-adjusted BACKWARD (NVDA 2024-05-28: 653M shares, the real ~65M was
multiplied by the June 2024 10:1 split), while FINRA's short interest (29M) and SEC's share count
(2.46B) are as reported. Before any forward split the measure read the split factor too high
(NVDA: ~22% instead of 1.2%), so the "never short above 30% of shares" rule and the borrow tiers
used knowledge of future splits — which plausibly kept the book from shorting future big winners
(the stocks that later split). The ranking itself is unaffected (FINRA's own days-to-cover; dollar
volume is split-invariant). Hypothesis 04 implements the fix (FINRA's raw short interest / SEC
shares, both as of their public dates). **Decision, written before its numbers: adopt the fix as a
correction whatever it does to the backtest, if its audits pass** — a leak is not a tunable.

## 43 · Loop 5 results (2026-10-22; 32 agents, 3.8 hours)

Six discovery agents (2022+ top 500 only) proposed 14 ideas; the selector kept 10 (dropped: a
gated duplicate of 06, IV-sized longs ≈ the rejected volatility cap with an exposure cut, a
duplicate of 01, and one more duplicate). Each was implemented in its own copy, checked (defaults
== v7, causal on a synthetic panel), run at its declared setting on T1-T5, and tuned in a nested
walk-forward over its own knobs (100-200 trials a period, wide ranges, off always inside).
Deltas vs v7 in alpha points; P = P(Δ > 0) on T1.

| # | idea | fixed T1 Δ (P) | T2 / T3 / T4 / T5 Δ | M1 | nested Δ (P, PBO) | verdict |
|---|---|---|---|---|---|---|
| 01 | fill shorts with high-beta names first | +2.0 (0.81) | +1.2 / −1.8 / +0.7 / −2.5 | 4.3 | **+2.9 (0.93, 0.52)** | consensus (tilt 0.45, 33-bar beta): T1 +4.7 but T3 −1.1 — fails; part beta credit |
| 02 | fill longs with low-beta names first | −0.0 (0.53) | −1.0 / −1.9 / +1.7 / −1.9 | 3.7 | −1.6 (0.35) | passed RISK (DD −7.5%) but audit: beta credit + one period (2026-H2, the one it was built on); neighbours −2.6 / −5.9 |
| 03 | halve gross while book vol is high | −3.2 (0.10) | −1.2 / −1.1 / −0.5 / −1.4 | 2.9 | −4.5 (0.02) | fails |
| 04 | **crowding measure without the split look-ahead** | −0.1 (0.45) | −0.4 / −0.6 / +0.2 / −1.5 | 3.9 | +1.2 (0.97, 0.62) | **adopted as a correction (v8)** |
| 05 | size longs by the change in 13F holders | +0.3 (0.61) | +2.3 / +0.2 / −0.9 / +0.9 | 4.9 | +0.7 (0.72, 0.81) | fails |
| 06 | put/call open-interest sleeve | **+4.3 (0.98)** | −1.3 / 0.0 / +1.0 / +0.5 | 5.3 | **+4.0 (0.99, 0.58)** | consensus (weight 3.75): T1 +5.3, T2 −2.5 (−21 in 2026-H2) — fails; one year of options |
| 07 | analyst target revisions for longs | +1.8 (0.69) | −2.7 / +4.3 / −0.9 / +0.1 | 4.9 | −3.3 (0.10) | fails |
| 08 | half-size new entries for 21 bars | +0.9 (0.76) | −2.0 / +1.1 / +0.1 / +0.9 | 4.6 | −0.6 (0.38) | fails (DD −7.8% via less exposure in churny 2026) |
| 09 | short sizes scaled by their volatility | +0.9 (0.80) | −0.1 / +2.6 / +0.6 / −0.3 | 5.1 | +1.2 (0.92, 0.63) | near miss (T2 −0.1, PBO 0.63, worst period −32%) |
| 10 | pause entries in high-dispersion weeks | +1.0 (0.69) | +0.6 / +0.7 / 0.0 / −1.5 | 4.5 | +0.3 (0.59) | fails |

**What it means.** No creative idea survived; the wide nested searches did not find a setting the
fixed tests missed (they confirmed the near misses, never rescued a failure). With ten tries, one
or two near misses (01, 09) are what chance alone produces. Three lessons:
1. **Beta credit.** alpha = return − beta × market rewards a lower or negative beta in a rising
   market; 01 and 02 gained partly that way. Judge raw returns beside alpha.
2. **Option signals need more history.** 06 won the top 500 in every option-era period and lost
   the next 500 names in the same months: one year cannot separate signal from luck. Revisit
   after paper trading adds data.
3. **The search's best find was a bug**, not a tweak: the split look-ahead in the crowding measure
   (04). Removing it costs 0.5 points of the five-test mean (T1 unchanged) and is adopted as v8.
A smaller leak of the same kind remains: the $5 minimum price reads the split-adjusted close
(no raw price is stored); its effect is only on names whose adjusted past price crosses $5.

## 44 · v8 adopted; an independent review of the port (2026-10-22)

v8 = v7 + hypothesis 04 (FINRA's shares sold short ÷ SEC's count, `research.attach_si_shares`) +
`si_plausible_max` 1.0. Untuned verdict on T1 (`untuned_verdict.py`, 100 random-name books):
alpha +12.5% (P 0.996), beta 0.12, CAGR +13.6%, volatility 10.6%, max drawdown −14.0%, random
books' 95th percentile +4.6% (beats all), 2× costs +11.5%, 2× borrow +12.3%, late fills +8.6%,
long side + SPY hedge +6.8%, 8/10 periods positive. Against v7 on the same days: T1 −0.1 (P 0.45),
T2 −0.4, T3 −0.6, T4 +0.2, T5 −1.5, **T7 (the lock, names 1501-2000) −2.0 (+10.5% vs +12.5%)**;
M1 3.9 vs 4.4; alpha without the top 5 unchanged (+6.8%). The decision was written before these
numbers (§42): a leak is removed whatever it costs.

A five-lens review of the port (workflow, each finding verified by a skeptic) confirmed:
1. **MAJOR — headless runs silently used the old measure.** `si_shares` is attached only by
   `research.load` / `cached_load`; `tsim optimize`, `04_compare` and any `strategies.load` user
   fell back to days-to-cover × back-adjusted volume under the v8 study name. Fixed now: the
   fallback warns (`compose`), the tests' synthetic panels carry `si_shares`, one test pins the
   warning. The real fix is a `si_shares` builder in `features.py` (engine proposal).
2. **MAJOR (causal, residual) — the weeks after a split.** FINRA's post-split shares short meets
   SEC's pre-split count until the next 10-Q: the ratio reads k× too high (TSCO's 5:1 split, Dec
   2024: covered on 2025-01-13 as "crowded"). ~4% of short name-days are in such a window; above
   100% the cap catches them (unknown, not shorted), below it they over-read the tier. Fixing it
   causally needs the split date (a split-events table: engine proposal).
3. Pickled panels could carry a stale `si_shares`: `cached_load` now re-attaches it on every hit.
4. The tests did not pin the cap's NaN handling or the loader wiring: two tests added
   (`test_an_implausible_crowding_ratio_is_unknown_never_shorted_and_cheap`,
   `test_research_load_and_cache_always_attach_finra_shares_short`).
5. README / signals notebook still described v7 and printed old borrow rates: updated (Table 2's
   labels now come from the params).
6. A parallel session's copy (`informed_flow_ls_p1`, §38P) was measured with the leaky measure;
   its numbers are not comparable to v8's.

Remaining known leak of the same family: the $5 minimum price reads the split-adjusted close.
