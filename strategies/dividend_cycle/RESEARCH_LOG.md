# dividend_cycle — research log

User question (2026-10-27): *is there an edge in buying high-dividend stocks a few weeks before
the payout and shorting them directly after it?* Written in the order it happened.

## §0 Pre-registration (2026-10-27, before any return was computed)

Only data-availability facts were looked at: the shared DB has no dividend table; Yahoo serves
ex-dates + amounts (split-adjusted) and the raw close for every ticker, but NO historical pay
dates (`Ticker.calendar` shows only the next one). So the event day is the **ex-dividend date X**
— the day the price mechanically drops by about the dividend, and the only date a holder, a
buyer or a short seller is affected by. "Payout" is read as X; a pay-date proxy (X + 15 sessions)
is an exploratory read of the event-time chart, never a registered rule.

**Hypothesis.** Buying pressure from income and dividend-capture investors lifts high-yield
stocks in the weeks before X (dividend-month premium, Hartzmark & Solomon 2013; Kalay 1982;
Graham & Kandel 1999); once X has passed the pressure is gone and the stock gives some back. A
long [E, X) then short [X, X+H) book earns timing alpha over the stock's own average return.

**Mechanism against it.** The ex-day drop is ~0.8-1.0 of the dividend (Elton & Gruber 1970), so
holding through X earns at most a small tax-wedge; institutions already harvest the run-up with
size and no costs; the long leg's "run-up" may be the yield accruing (a high-yield stock that
drifts up N sessions before X and falls on X is the dividend itself, which total-return prices
remove). Expected result: no net edge on liquid names after costs.

**Point-in-time rules (the literature default, U0 — nothing tuned).**
1. Universe: the lab's point-in-time top 500 per half-year (`lab.names`), E0 + TB + T3 =
   TRAIN (2016-07 → 2021-12), T1 = VALIDATION (2022-01 → 2026-10), T2 (names 501-1000) =
   second validation sample. T4/TL stay locked.
2. Regular payer at an anchor ex-date `a`: the last 4 ex-dates lie within 400 days and the last
   two gaps are each 80-105 days (quarterly). Only past ex-dates are read.
3. Predicted next ex-date P = `a` + median of the last 3 gaps (calendar days), rounded to the
   next session.
4. High yield: the last four dividends gone ex (within 400 days, ex-date ≤ decision day) ÷ raw
   close of the decision day ≥ 3% (about 2× the index yield). (First drafted as a trailing
   365-day sum; the synthetic test showed it counts a FIFTH quarterly dividend on the ex-date,
   four gaps being 364 days — +25% yield exactly there. Changed before any result was read.) Yield buckets are reported beside it (<1, 1-2, 2-3,
   3-4, ≥ 4%).
5. Long: decide at the close before E = P − 15 sessions, fill at E's open. Exit at the open of
   the ACTUAL next ex-date X (first Yahoo ex-date within P ± 20 days; declared ≥ 10 days ahead
   in practice, so known at exit). The open of X is the ex-price and the holder keeps the
   dividend, so adjusted prices give the exact total return. If no X exists (cut to zero /
   skipped), exit at P's open and there is no short: the event stays in the sample, so a
   dividend that never arrives is paid for by the long.
6. Short: opened at X's open (a short opened on the ex-date owes no dividend), covered at the
   open of X + 10 sessions. Both legs are also reported alone.
7. Excess return of a window = its adjusted open-to-open return minus (a) the equal-weight
   open-to-open return of the same period's universe and (b) the same for all names with a
   trailing yield ≥ 3% on that day, whatever their cycle phase — (b) removes the market AND the
   high-yield style, leaving the cycle position. Placebo: the same windows shifted +30 sessions
   (mid-cycle, never touching an ex-date); event minus placebo is the within-stock timing
   effect.
8. Costs: 8 bp round trip per leg (spread 2 + slippage 1 bp a side + commission at a $10k
   order); the short also pays 1 %/yr borrow. Breakeven reported.
9. Statistics: events share dates (ex-dates bunch at month ends), so the t-statistic is across
   the CALENDAR MONTHS of X (mean of each month's events), not across events.

**Variant budget.** One registered rule: N = 15, H = 10, yield ≥ 3%. Read as a sensitivity
grid, none adopted by hand: N ∈ {10, 15, 20} × H ∈ {5, 10, 20} (9 cells, on TRAIN).

**Pass bar ("an edge").** The registered rule's NET excess (vs (b)) is > 0 with month-clustered
t ≥ 2.5 on TRAIN AND > 0 with t ≥ 2 on T1 and on T2, the placebo does not show the same
effect, and it survives 2× costs. Pre-2020 the DB holds survivors only — the long leg of a
high-yield book is the most flattered thing in it (yield traps that cut the dividend and fell
are missing) — so a TRAIN-only pass is not enough; T1 (2022+) has to carry it.

**Known biases, stated with every result.** Survivorship (above); Yahoo's ex-date list is in
hindsight, but the rule reads only ex-dates before the decision; dividends are before tax (a
short holding is ordinary income, 61-day qualified rule not met).

## §1 Registered run (2026-10-27) — `research.py run`, output in `var/dividend_cycle/run4.log`

Data: Yahoo ex-dates for 1,352 of the 1,359 names in the point-in-time universes (BLD, CPRX, CWAN,
GTLS, JHG, LC, TMHC are no longer served), 26,006 regular-quarterly-payer anchors, 96.6% followed
by a dividend inside P ± 20 days. Net bp per event after costs (8 bp a leg, 1%/yr borrow),
excess vs the high-yield index, t across calendar months:

| sample | events | long | short | both | both vs market |
|---|---|---|---|---|---|
| TRAIN 2016-07 → 2021-12 | 1,745 | +40.9 (t 3.33) | +23.5 (1.32) | **+63.7 (3.25)** | +57.5 (2.26) |
| T1 2022 → 2026-10 | 1,509 | −9.8 (−0.66) | −16.6 (−1.82) | **−26.1 (−1.64)** | −9.3 (−0.18) |
| T2 names 501-1000 | 1,394 | −35.5 (−1.99) | −26.6 (−1.74) | **−61.3 (−2.64)** | −47.6 (−1.02) |

* Before costs (add 20 bp): T1 ≈ −6 bp, T2 ≈ −41 bp — no edge to be eaten by costs.
* Placebo (windows +30 sessions): TRAIN −41 bp, T1 −16, T2 −65. The TRAIN placebo is negative, so
  the TRAIN gap event − placebo is real there (+104 bp); in T1 it is −10, in T2 +4.
* Sensitivity grid N ∈ {10,15,20} × H ∈ {5,10,20}: TRAIN all nine cells +38 … +87 (t 2.3-3.4);
  T1 −35 … +9 (t −1.8 … +0.2); T2 all nine negative (−24 … −52).
* By trailing yield: only TRAIN's ≥ 4% bucket stands out (+114, t 4.05); T1 +4 (t −0.08), T2 −68.
* Raw, un-benchmarked: the long leg returns +166 / +94 / +44 bp (the market's drift plus the
  dividend); the short leg LOSES in every sample (−30 / −48 / −52 bp raw), because stocks
  keep rising after the ex-date like everything else.
* Event time, TRAIN: excess vs the high-yield index rises about +36 bp from day −15 to day 0, then
  −14 bp on day +1. T1: no run-up (−56 → −54). T2: falls (−50 → −83).

**Pass bar: NOT met.** TRAIN passes (t 3.25 ≥ 2.5); T1 and T2 are negative (needed > 0, t ≥ 2).

## §2 Post-hoc diagnostics (read AFTER §1; they explain, they do not rescue)

* **TRAIN is mostly 2020.** By exit year the registered rule earns +182 bp/event in 2020 (435
  events; its placebo −129) and −27 / +70 / 0 / −10 / +70 in 2016 / 17 / 18 / 19 / 21. Without
  2020: **+24.4 bp, t 2.02** — below the 2.5 bar. High-yield names that crashed in March 2020 and
  rebounded are a style-dispersion effect the high-yield index does not remove.
* Pre-2020 the DB holds survivors only; a high-yield book's long leg is the most flattered thing in
  it (dividend cutters that fell and delisted are missing), so even the ex-2020 TRAIN number is an
  upper bound.
* **Pay-date reading** (long held to ex + 15 sessions ≈ 3 calendar weeks, short [ex + 15, ex + 25)):
  TRAIN −1.1 (t 0.91), T1 −23.1 (−1.48), T2 +1.6 (0.87). Nothing.

## §3 Verdict

No edge. The cycle-timing effect (run-up into the ex-date, give-back after) exists in the 2016-2021
sample, is carried by 2020, and is absent from 2022-2026 in both the top 500 and the next 500
names. Not run through the portfolio engine: validation fails on the per-event premise check,
before costs, so sizing and slots cannot rescue it. Not tested: pre-tax vs after-tax (a ≤ 60-day
holding makes the dividend ordinary income, which only lowers the long leg), declaration-date
effects (no declaration dates in hand), non-quarterly payers, and the locked T4/TL samples
(untouched).
