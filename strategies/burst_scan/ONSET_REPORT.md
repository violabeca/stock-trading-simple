# Can the existing features spot where a move starts? (1-minute bars, regular hours)

*burst_scan research, 2026-09-29/30. Eight rounds of research by two dozen agents, then an independent audit
and a final review. The key numbers were re-computed independently before they were accepted. Training window
2025-06-18 → 2026-04-30 only; the May–June 2026 holdout was never loaded. Up moves and down moves were treated
separately throughout.*

**How to read the scores.** Most results are an **AUC**: how well a score ranks the right minutes above the
wrong ones — **0.5 = a coin flip, 1.0 = perfect**. The **coin-flip market** is a fake copy of the real prices
in which every 1-minute bar is flipped up or down at random: it keeps each bar's size, range, volume and time
of day, but contains no direction at all. Anything the features "achieve" there is volatility, not foresight.

## The answer in six sentences

1. **The features tell you *when* and *where* price is about to move — not *which way*.** A detector built on
   the 22 existing features finds the minutes around the start of a 2% run well (AUC 0.86 over 120 names), but
   on the coin-flip market the same features score almost the same (0.85).
2. **Up and down look identical to the features.** Without the price-location features, telling an up-start
   from a down-start scores 0.50 on every name group. With them it rises to about 0.55 — and the coin-flip
   market shows exactly the same, because a swing low is by definition the bottom of a fall: that is
   hindsight, not foresight.
3. **At the moment you would act — price just made a new 30-minute low — it is a coin flip.** Price rises 1%
   before it falls 1% 48% of the time (not reliably different from 50%), and the features predict which at
   0.51.
4. **No knob changes this.** Every knob moves the numbers only mechanically: by selecting more volatile names
   and days, by building more hindsight into the label, or — for price confirmations — by entering later in the
   move.
5. **Nothing trades after costs.** Every rule that looked profitable in-sample turned out to be a few days in a
   few speculative names, market beta, or a flaw in the test — and every rule that was taken to untouched names
   failed there.
6. **What is useful: forecasting how *big* the next move will be.** The forecast explains about two-thirds of
   how large the next 15 minutes' range will be (R² 0.66), and most of that is simply recent volatility and
   the time of day. Use it to size positions and set stops, never to enter.

## How it was tested

| | |
|---|---|
| **Data** | 1-minute bars, 09:45–15:45 of regular trading hours, 120 Morning200 names in four separate groups of 30. The **study** group (GLW, ORCL, ALB and 27 random names) was used for the research. The **fresh** and **third** groups were first used only for checks; later a red team searched on them too. The **fourth** group stayed untouched until the final tests. |
| **A "run" and its onset** | Price moves at least 2% in one direction, the first 2% within 4 hours, and the run ends only on a 1% pullback (the move explorer's definition). Its **onset** is the swing low (up run) or swing high (down run) it starts from — something you only know afterwards. About 4 runs per stock per day, a third of which start in the first 15 minutes; those (the overnight gap resolving) are left out. |
| **Features** | The 22 features of `research.explore_features`: volume and range against the normal for that time of day, daily and intraday ATR, recent and session returns, gap, VWAP distance, position in the day's range, distance to yesterday's high/low, previous-day context, and the clock. |
| **Up vs down** | Always separate: separate labels, separate models, separate trades. |
| **Safeguards** | Walk-forward only (each month is scored by a model trained on earlier months). Uncertainty measured by resampling whole days, so one wild day counts once. A **placebo**: the same model trained to find the NEXT day's run starts at the same clock time, so it can only learn which stock, which day and what time. The **coin-flip market**. Four name groups. Tests whose code and pass criteria were frozen before the data was opened. An independent audit. |
| **Costs** | 24 bp per round trip (the repo's standard). On the violent bars where these signals fire, spreads are 2–4× normal, so a realistic round trip there is **28–43 bp** — every "close to cost" result below is optimistic. |
| **"The market"** | The database has no ETF or futures minute bars, so the market is a home-made equal-weight index of the same group's other 29 names. All market tests and hedges use it. |

## 1 · What the features can do: find activity

| (120 names, average of up and down) | real market | coin-flip market |
|---|---|---|
| How well the detector finds the minutes 0–5 after a swing extreme (AUC) | **0.86** | **0.85** |
| … the placebo, which can only know stock, day and time of day | 0.82 | — |
| … inside one stock on one day: full model vs placebo | 0.77 vs 0.71 | 0.77 vs 0.70 |
| How often its top-1% alarms really sit at a run's start (≈ 2% of all minutes do) | 23% | 23% |

- The top 1% of alarms are ~11× more likely than a random minute to sit at a run's start.
- But most of that is knowing **which stock, which day and what time of day**: the placebo gets most of the way.
- The coin-flip market reproduces **~96% of the detector's score above chance**. Starts of 2% runs are what
  volatility produces, so spotting them is spotting volatility.

## 2 · What they cannot do: tell up from down

| Test (120 names) | result | coin-flip market |
|---|---|---|
| Up-starts vs down-starts, **without** location features (AUC) | **0.50** | 0.50 |
| … **with** location features | 0.55 | 0.55 (hindsight) |
| At a new 30-minute low: share where +1% comes before −1% | **48%** (95% range 46–50%) | 50% |
| … how well the features predict it (AUC) | **0.51** | 0.50 |
| Model trained to spot "+0.5% in the next 30 min": how well it spots its own target vs the opposite "−0.5%" | 0.687 vs **0.682** | — |

- **The 48% is not a signal.** Its range covers 50%, it disappears without the 10 biggest sell-off days, new
  highs do not mirror it, and within a day new lows actually bounce slightly *more* often than random minutes.
  It is worth at most ~3 bp, and every rule built on it loses 11–24 bp after costs.
- **A model trained directly on direction** predicts the opposite outcome just as well as its own: it has
  learned "something will move", not "up".
- **No pocket has direction.** 15 kinds of subset (~400 tests) were tried — time of day, calm vs wild stocks, gap
  days, market up or down, heavy-volume days, big vs small runs, moves net of the market. Three tests passed the
  bar fixed in advance, all in "the market fell over the last 30 minutes"; they turned out to be movement, not
  direction (the up and down detectors scored the same bars), and were worth +0.5 bp live.
- **Not the market either.** Starts do cluster: in 36–40% of cases at least five other stocks of the group start
  a run the same way within 5 minutes (11% by chance). But the market's state at a new low adds nothing, and the
  market's own turns are no easier to call.
- **Not signed flow either.** 16 extra features from the same bars (where each bar closed in its range × volume,
  tick-rule volume, VWAP slope) add 0.000 ± 0.005 to the direction AUC. Their one real effect is the bid-ask
  bounce: a bar that closes at its high is followed by a first trade ~3 bp lower — a move between one bar's last
  trade and the next bar's first, so no order can capture it.
- **Price confirmations** ("close 0.5% off the 30-minute low") know the direction only because the price has
  already moved — and they behave identically on the coin-flip market.

## 3 · Why an "early alarm" is an illusion

| (120 names) | real | coin-flip market |
|---|---|---|
| Runs that the top-1% detector alarms within 30 minutes of the start | **18%** | 16% |
| … of those, the alarm was already ringing **before** the low / high | **85%** | 85% |
| … the detector for the **opposite** direction was ringing too | **67%** | 67% |
| Price confirmation (close 0.5% off the 30-min extreme): runs caught | 71% | 72% |
| … minutes after the start / share of the run already done | 3 min / 24% | 3 min / 24% |
| … false alarms per run caught | **7.5** | 7.0 |

- The detector does not ring *because* a run starts. It rings through the whole fall into the low — and just as
  loudly around the highs.
- The only genuine early warning is a small, **direction-free** rise in activity 1–5 minutes before a swing
  extreme (worth +0.02 AUC; no profitable trade in 108 attempts).
- The first 45 minutes hold most runs and alarms, and are where detection is weakest.

## 4 · The knobs (30 study names; single-knob numbers from the first round)

| Knob | What changing it does | Why | Direction? | Tradable? |
|---|---|---|---|---|
| **move size** (1–4%) | detection 0.84 → 0.86 at a fixed 1% reversal (0.93 at 3% when the reversal grows too); **inside one stock-day it gets worse** (0.81 → 0.75) | bigger runs only happen in the most volatile stocks and days; volatility alone rises just as much | no | no |
| **reversal** (0.5–1.5%) | detection and "direction" look better | a larger reversal forces a deeper fall into the extreme = more hindsight | no (0.50 without location) | no |
| **time window** (30–240 min) | 0.90 at 30 min vs 0.84 at 240 | fast runs = the most volatile stock-days; volatility alone matches | no | no |
| **lag after the extreme** (0–10 min) | 0 looks best ("direction" 0.65) | the label *is* the spike bar at the low — hindsight plus that bar's own activity | no | no |
| **predicting the start 3–10 min before it happens** | "direction" 0.55–0.60 | the fall *before* the low marks it — hindsight | no | no |
| **skip the open** (0/15/30 min) | +0.01 | only the first 15 minutes' volatility | no | no |
| **move scaled to each stock's ATR** | "direction" 0.65–0.78 with location | a deeper counter-move in the stock's own units = hindsight | no (0.50) | no |
| **profile days, mean vs median, ATR days** | nothing (≤ 0.002) | — | no | no |
| **short activity window 5 min** | +0.005 | reacts faster to the burst = movement | no | no |
| **pre-move window 15 min** | "direction" +0.03 | a sharper "price just fell" = hindsight | no | no |
| **confirmation distance** (0.25–1.5%) | fewer false alarms (9 → 1.5 per run caught) but later (1 → 11 min; 14% → 60% of the run done) | first-passage geometry (trap 5); identical on the coin-flip market | — | no |
| **confirmation lookback** (10–120 min) | 10 min catches most runs (83% vs 33%) | a run's low is rarely a 1–2-hour low | — | no |
| **activity filter / model gate on confirmations** | fewer false alarms, far fewer runs caught; never better | the gate just picks the active stock-day | — | no |

Combinations add nothing new: across a 72-combination grid the single-knob effects explain 94–98% of the
variation in each metric, and none of 1,008 combination tests fixed in advance turned on direction or trading.

## 5 · Trading

- Trading the detector's alarms loses after the 24 bp round trip: long after up-alarms −28 bp, short after
  down-alarms −16 bp per trade (30-minute hold, 120 names).
- Every rule that cleared the cost in-sample failed for one of five reasons:
  1. a handful of days in a handful of speculative names (LUNR, QUBT, QBTS, BE, IONQ);
  2. **market beta** — stocks that swing more than the market, bought on up-market days;
  3. a look-ahead in how the "top 1%" was chosen (fixed; leads shrank ~40%);
  4. a hindsight label scored as if it were live;
  5. it did not repeat on stocks the study had never used.
- **How big pure noise looks:** on the coin-flip market, top-1% alarms "earn" −21 to +14 bp over 60 minutes,
  and results with t ≈ 2 appear by chance. Any in-sample ±15 bp is noise-sized.
- Tests fixed in advance, all failed on untouched stocks (returns **before** the 24 bp cost):
  - "buy the violent new low": +2 bp;
  - "buy new lows in the right stock-days": +13 bp raw, +3 bp with the market's move subtracted;
  - the red team's best rules (an agent told to find any tradable rule on 90 names; its best rules were frozen
    and tested on the untouched 30): −16 and −38 bp — the opposite sign (−40 and −62 bp after costs).

## 6 · What to use instead

- **For risk, not entries.** A fixed 0.5% stop is hit on 9% of trades in the calmest stock and 67% in the
  wildest. Scaling stops and position size to a forecast of the next 15–30-minute range levels that out and
  cuts the spread of trade outcomes by 12–24% — it narrows outcomes; it does not change the average trade.
  Recent 15/60-minute range plus the clock gives R² 0.60 (~90% of the job); adding the stock's usual range at
  that time of day gives 0.66, which is everything the 22 features give.
- **Direction, if it can be found at all, would need information these bars do not contain** — order flow from
  quotes (bid/ask), real 1-minute market or sector bars (ETF or futures), or an event calendar (earnings, news).
  These were not tested; the home-made index's current state already added nothing.

## 7 · Traps found on the way (worth knowing for any 1-minute study)

1. **Hindsight location.** An onset is a swing extreme, so "price just fell" marks up-starts. Always test
   without location features and at a live decision point.
2. **Movement is not direction.** Check that the up-detector does not score down-starts just as high.
3. **Tied extremes.** 36% of "new 30-minute lows" only tied an earlier low and were almost never onsets.
4. **Price in the denominator.** Features divided by the current price read high at lows and faked direction.
5. **First-passage geometry.** How far a bar closed above its low sets the odds of "+1% before the low is
   broken", even for a random walk — that one number beat all 22 features (0.72 vs 0.69). (A different question
   from §2's "+1% before −1%", where this geometry cancels out.)
6. **"Beating the stock's day"** is biased for bars picked at lows: even randomly chosen new lows "beat their
   day" by +25 bp while actually losing 3.5 bp.
7. **Averaging per day** over-weights winning days; setting the "top 1%" cut-off from the whole month peeks at
   the future; comparing against many coin-flip markets understates the noise unless you compare day by day.
8. **Market beta** looks like skill in high-beta names — always subtract the market's move.

## Where the work is

- This report is self-contained. The research code and every agent's scripts and tables were kept in a
  temporary scratch folder (`…/scratchpad/onset/`: `onset_lab.py`, `FINDINGS.md` with every verified number,
  `results/it01 … it09/`); it is not part of the repo and will not persist.
- Earlier steps of this line of work: `README.md` ("Move study", "Move explorer", "The extended hours") and
  `IDEAS.md` (#12–#15).
