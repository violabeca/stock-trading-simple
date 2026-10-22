# Can anything predict up or down? The direction study on 1-minute bars

*burst_scan research, 2026-09-30 to 2026-10-01. Two rounds of method agents plus a Darts forecasting track, an
adversarial verifier on every promising claim, two critics, and a one-shot confirmation of every frozen rule on data
no agent had seen. The May–Sep 2026 holdout was never loaded.*

## The short answer

**No.** Nothing tested predicts whether MU, COIN, CSCO, AAL or ANET will make a significant move up or down. That
covers 1 minute to 2 days ahead, using their own 1-minute OHLCV, 497 other stocks' 1-minute bars, pre- and post-market
bars, and daily history. The bar for "predicts" was surviving three things:

- **honest controls:**
  - a coin-flip market with the same volatility but no direction;
  - replication on other stocks;
- **costs:** a 7 bp round trip;
- **unseen data:** a pre-registered test on months no agent had looked at.

The details:

- **The tests could see an edge.** A power study planted known up/down edges into coin-flip data, and the same tests
  find an own-stock edge of 2–5 bp per 30-minute trade. So these negatives mean "no edge big enough to trade", not
  "too little data to tell".
- **All 11 frozen rules failed.** Every rule that looked good on the discovery months was frozen and tested once on
  Feb–Apr 2026 and on 10 untouched stocks. All 11 failed (section 4). The best in-sample looking ones were
  +16 to +48 bp per trade with t 2.3–3.7. Out of sample they were between −13 and +40 bp, never with t ≥ 2.
- **What *looked* like direction was always one of three things:**
  - **microstructure:** the last minute's move is partly given back over the next 1–2 minutes, worth under 0.5 bp;
  - **drift:** this period's overnight premium, about 5× its 2015–2025 level, which vanished in Feb–Apr 2026;
  - **a few trend days:** sector moves that carried on during a handful of volatile October–January days.
- **One robust effect is real, but it is *relative* and it didn't hold up.**
  - A stock's late-day move against the other stocks is partly undone overnight. Across 487 stocks it was the
    strongest result of the whole study (t 6.7), and the same idea surfaced in three different methods.
  - On the locked months its strength halved (t 1.5).
  - A tradable book of about 16 longs and 16 shorts made +16 bp per leg in discovery and **−2 bp** out of sample.
  - On 10 years of daily data, the 4 study stocks with that much history show it only weakly and unstably.
- **Forecasting models (Darts) don't help either.** None of the classical models (ARIMA, ETS, Theta …), global ML
  (LightGBM, random forests, classifiers), 11 deep nets (N-BEATS, TFT, TiDE …), or the pretrained Chronos-2 and
  TimesFM predicts direction. They forecast "no move", drift or volatility (section 5).
- **This corrects an earlier conclusion.** The earlier "LONG/SHORT detectors find movement, not direction" verdict
  (`move_detectors.ipynb`) was right for the wrong reason. Its test, a 2%/4h direction label with a 51-feature model,
  could not see even a planted 35–40 bp edge. This study's verdict rests on tests that can.

## 1 · What was tested

| | |
|---|---|
| **Stocks** | MU, COIN, CSCO, AAL, ANET, 1-minute regular-hours bars |
| **Discovery** | 2025-06-18 → 2026-01-31: 156 sessions, about 135 scored after a 21-session warm-up. All exploration happened here. |
| **Locked confirmation** | 2026-02-01 → 2026-04-30 (60 sessions). Each frozen rule was run once, against a pass bar written before the run. |
| **Fresh stocks** | 10 drawn at random: APLD, BX, CEG, CVNA, HIMS, OKLO, QBTS, RIVN, RKLB, SNOW. Used for confirmation only. |
| **Context data** | 1-minute bars of 497 other stocks, used for an equal-weight market index, sector-peer baskets and a broad cross-section. The database has no 1-minute ETFs. Also: pre- and post-market bars (04:00–20:00), and daily bars 2015–2025. |
| **"Significant move"** | Several definitions: ±1% within 60 min, ±2% within 4 h, 0.75 × daily ATR within 4 h, forward returns from 5 min to 4 h, to the close, overnight, next day, 3 days. |
| **Trade** | Decide at the close of minute t using bars up to t only, enter at the next minute's open. Costs are 7 bp round trip for a $10k order and 24 bp for a $1k order. |

| Method | Tried | Cells | Best discovery result | Fate |
|---|---|---|---|---|
| Serial dependence | own-stock momentum or reversal, 1 min → 4 h; conditional pockets; ±X% persistence | 2,513 | 1–2 min bounce (< 0.5 bp); late-day → overnight relative reversal | bounce untradable; reversal halved, then killed |
| Cross-asset | market and peer lead-lag (1–60 min), divergence/convergence, breadth | 2,770 | "lead-lag" t 5.0 (it is the stock's own bounce); residual momentum +26 bp (t 2.8) | failed out of sample (−3 bp) |
| Clock and flows | leveraged-ETF / gamma flows into the close, Gao-Han-Li-Zhou, gaps, trend days, same-slot seasonality | 241 | the unconditional overnight drift (+29 bp, t 3.1) | drift vanished (−2.5 bp) |
| Shocks and jumps | information vs liquidity jumps, sector-basket shocks | 2,469 | sector shock → 60% follow (t 3.8); morning range break +48 bp (t 2.9) | both failed |
| ML on raw sequences | logistic / GBM / MLP / MiniRocket; pooled training on 50–250 other stocks | 490 | +2 bp on 60-min residual direction | no candidate |
| Daily HF signals | realized skew, signed jumps, semivariance, overnight vs intraday split | 1,340 | literature signals do not replicate; intraday → overnight reversal | near miss on 5 stocks; absent 2015–2025 |
| Adaptive rules | rolling-sign momentum/reversal, online learners, trend filters, regime switches | 543 | IC ≤ 0.06, picked from many tries | no candidate |
| Power and controls | planted edges, minimum detectable effect, harness audit | 143 | calibration | see section 2 |
| Pre/post-market | pre-market → session, post-market → pre-market, closing auction | 199 | post-market fade +19 bp (t 2.35) | failed (−13 bp) |
| Round 2 | realistic execution of the reversal book; sector-basket breadth; news-gap drift over 1–3 days; volatility asymmetry | 384 | book +16 bp/leg at the mid price (t 3.7) | killed (−2 bp); the other three null |
| Darts forecasting | classical, global ML, deep and pretrained models (section 5) | 972 | fading AutoARIMA's 2-h forecast, IC −0.05 (t −2.9) | weak; nets −1.8 bp; not tested on the locked months |

In total that is about 12,000 hypothesis cells. At that count about 30 cells reach |t| ≥ 3 by pure chance, which is
why every claim had to survive replication and an unseen-data test.

## 2 · Why a "no" here means no

- **Planted edges are found.** A rule that biases the next 30 minutes toward the trigger's direction was planted
  into the coin-flip market at 1–3 percentage points of bias, which is 2–11 bp per 30-minute trade. The 30-minute
  IC and P&L find it at t ≥ 3 with at least 80% power. A correctly specified own-stock rule that pays the 7 bp cost
  is detectable here.
- **The old detector test is blind.** The previous study's detector (51 features, 2%/4h direction label) finds
  nothing even at a planted 35–40 bp edge. Two causes: the 2%/4h label needs about 2× the edge, and the 51
  features dilute the signal by another 2× or more.
- **Out of reach: daily or overnight bets on one stock.** One trade per stock per day needs 25–45 bp to be visible
  in 7 months. That is why those questions were also tested on the 487-stock cross-section and on 10 years of daily
  bars.
- **The statistics are calibrated.**
  - Day-clustered t-stats behave like N(0,1) on the coin-flip market.
  - The per-stock coin-flip null is 1.2–1.7× too narrow for statistics pooled across stocks, because it breaks
    their co-movement. Round 2 therefore used a market-preserving coin flip, with one sign per minute for all
    stocks, plus night or day block permutations.

## 3 · What looked like direction, and what it really was

| Looked like | Really | Evidence |
|---|---|---|
| Reversal after any 1-minute move | Bid-ask bounce | 0.02–0.4 bp; strongest in tick-bound AAL and CSCO |
| "The market leads the stock by minutes" | The stock's own 1-minute bounce | none of the IC comes from the index; a day-shifted index scores the same |
| Overnight long, +27 bp a night | Bull-window drift (the market's overnight premium × beta, plus MU's run) | ~5 bp/night over 2015–2025; −2.5 bp on Feb–Apr 2026 |
| Sector shocks continue (60% of next ±1% moves) | Continuation on ~10 volatile trend days; ~1 bp left after removing the sector's own move | +2 bp (t 0.8) on the locked months; 54% across 20 data-built sector baskets, gone without the best 10 days |
| Fading big low-volume jumps (+9 bp, t 7.6) | Untradable prices: 81% of the P&L came from minutes with no trade, whose "price" is a copy of the jump | an honest entry gives +1.5 bp (t 1.4) |
| News gaps keep drifting (+55 bp by day 3) | A few biotech and crypto small caps moving 30–70% | median +3 bp; ±1 ATR first passage 50.2% |
| High volatility → down moves first | The opposite sign *within* a day (volatile days are down days, then rebound); nothing *across* days | no causal trade captures it |
| **Late-day relative move reverses overnight** | **Real in 2025–26 for the speculative cross-section; not tradable; not a property of the 5 stocks** | IC +0.039 (t 6.7) → +0.019 (t 1.5) on the locked months; the book −2 bp; 2015–2025 on the 5: −1 bp/night |

## 4 · The confirmations (one shot each, pre-registered bars)

| Frozen rule | Discovery, 5 stocks | Locked months, 5 stocks | Fresh stocks | Verdict |
|---|---|---|---|---|
| Fade the post-market move in the next pre-market | +19.0 bp, t 2.35 | −13.1, t −0.7 | +4.6, t 0.5 | fail |
| Fade the other stocks' late pre-market move over the first hour | +14.2, t 2.1 | −6.1, t −0.4 | −5.9, t −0.5 | fail |
| Follow a sector-basket 3σ shock (next ±1% within 60 min) | +5.2, t 2.5 | +2.1, t 0.8 | +1.8, t 0.8 | fail |
| Follow a morning 4σ range-break shock for 60 min | +47.9, t 2.9 (73 trades) | +40.1, t 1.7 (29 trades) | +10.4, t 0.55 | fail |
| Residual momentum since the previous close (10:29 → close) | +25.6, t 2.8 | −3.0, t −0.2 | −5.0, t −0.4 | fail |
| Long every night (drift) | +29.0, t 3.1 | −2.5, t −0.1 | +23.6, t 2.1 | fail |
| Fade the day's idiosyncratic move overnight (every stock-night) | +5.2, t 0.7 | +24.4, t 1.7 | +13.3, t 2.0 | near miss |
| Fade big low-volume jumps | 2 trades | 1 trade | 37 trades, −2.2 bp | fail (no trades) |
| Late-day z > 2 fade overnight | 43 trades, −16 bp | 25 trades, +5.5 | −4.6 | fail |
| Relative reversal IC, 487 stocks (frozen by its verifier) | IC +0.039, t 6.7 | IC +0.019, t 1.5 (3 of 3 months > 0) | — | fail by the letter, same sign |
| Liquid-third reversal book, ~16 + 16 names, 15:59 → next open | +16.3 bp/leg, t 3.7 | **−2.0, t −0.2** (0 of 3 months > 7 bp) | — | killed |

The "near miss" fades the day's move net of the market, every stock every night. It has the right sign everywhere:
+24 bp on the 5 stocks in Feb–Apr, 3 of 3 months positive, and 62% of 2%+ overnight moves called right. But it
misses its bar (t 1.7). On 2015–2025 daily bars, for the 4 stocks with that much history (MU, CSCO, AAL, ANET), it
is weak and unstable:
- trading its sign makes −1 bp a night (t −0.7), negative in 9 of 11 years;
- its rank IC is +0.02 (t 2.0), but negative in 2017–2019 and of the opposite sign for MU.

It belongs to this period, not to these stocks.

## 5 · Forecasting models (Darts)

Every main model family in the Darts library was run as a direction predictor on the same data, labels and nulls.
The models used 5-minute steps (and some 1-minute) of the same bars, trained models were fitted walk-forward, and
every run had coin-flip twins. That is about 970 tests.

| Family | Models | Tests | What they actually forecast | Best result after checks |
|---|---|---|---|---|
| Classical, refit on the last 1 / 5 / 20 sessions | naive baselines, AutoARIMA, ETS/Holt, Theta/AutoTheta, CES, MFLES, FFT, Kalman | 704 | zero, the window's drift (momentum), a pull back to a smoothed level (reversal), or a clock pattern | fading the 2-hour AutoARIMA forecast: IC −0.05 on fresh rows (t −2.9). It is below the multiple-testing bar, half of it is market timing, and the trade nets −1.8 bp. |
| Global machine learning, across the 5 stocks | linear, random forest, LightGBM / HistGB (regression, quantile, classifiers), logistic | 129 | the short-horizon reversal of Jul–Aug, which later turned into mild continuation | a next-5-minute classifier, IC +0.015 (t 1.5) |
| Deep nets | N-BEATS, N-HiTS, TCN, TiDE, TSMixer, TFT, DLinear, NLinear, LSTM, Transformer, and a committee of all 11 | 50 | noise that changes with the training seed | every "hit" vanished under another training seed |
| Pretrained, zero-shot | Chronos-2 (Amazon), TimesFM 2.5 (Google) | 89 | the *size* of the next move (spread AUC 0.58–0.73), just as well on coin-flip data | Chronos-2 on 1-minute prices: IC −0.03 (t −1.9) |

- **Costs:** 429 of 432 scored cells lose money at a 7 bp round trip; the other 3 have t ≤ 0.25. None wins at
  24 bp.
- **Forecast error:** no model beats "the price won't move" on ordinary forecast error (MAE / RMSE).
- **Significance:** the largest single-cell |t| is 2.8 in about 970 tests, against a multiple-testing bar of 4.05.
- **Pretrained models:** they know volatility, not direction. Pretraining on millions of series gives a good sense
  of how much a series will move, and no sense of which way an intraday stock price will go.

## 6 · Why direction is so hard here

- **Most of a stock's move is the market's move.** Nothing here predicts the market's direction, and these are
  large, liquid stocks where the remaining, stock-specific part is quickly priced.
- **The structure that does exist is too small or too short-lived:**
  - the 1–2 minute bounce sits below the spread;
  - continuation lasts a few trend days;
  - closing pressure reverses overnight in the speculative cross-section.
- **A 7 bp round trip is the real barrier.** At a 60-minute horizon, paying 7 bp needs a rank IC of about 0.10.
  The best honest ICs found were 0.02–0.06.

## 7 · What it would take

Direction needs information that 1-minute OHLCV does not carry. In order of promise:

1. **Quote and order-book data** (bid/ask sizes, order-flow imbalance). This is the one information class with
   documented short-horizon direction content.
2. **An earnings and news calendar with the surprise itself.** The 1-minute bars only see the gap, which is already
   priced at the open.
3. **Options data** (implied-volatility skew, flows).
4. **Faster leading markets:** index futures (ES/NQ) at second resolution; BTC for COIN.

What the existing features do well is forecast the *size* of the next move. Use them for stops and position size,
not for entries (see `ONSET_REPORT.md`).

## 8 · Traps found on the way (for any future study)

- **The per-stock coin flip is too narrow for pooled cross-stock statistics.** It breaks the stocks' 0.2–0.3
  co-movement, so its z-scores were 1.2–1.7× inflated (one claimed z 14.3 was really 3.4). Use one sign per minute
  for all stocks, or block permutations.
- **The month × stock direction AUC is biased under the null.** It reads 0.40–0.49 for momentum scores and
  0.51–0.60 for reversal scores, and up to 0.67 for trend-shaped forecasts. Compare it with the coin-flip value,
  never with 0.5, or group by day.
- **Long-horizon ICs computed within groups on overlapping windows are biased negative.** This produced fake
  "reversal to the close" cells at t −2.5 to −4.3.
- **16% of minutes in the broad stock set have zero volume.** Their close is a copy of the last trade. Entering or
  exiting there fakes reversals (81% of one rule's P&L).
- **A replication on other stocks over the *same days* is not independent.** In this study, per-day results
  correlated 0.73.
- **Selection has to be nulled too.** A best-of-grid result must be compared with the best-of-grid on the null:
  one rule's t 2.94 was reached in 40% of coin-flip worlds by the same search path.
- **The closing auction is not in the regular-hours bars.** It is the 16:00 bar of the extended data, and its
  volume is 4.4× the 15:59 bar's. A decision at 15:58 cannot use market-on-close orders (the cutoffs are 15:50 and
  15:55), and the market-on-close version of the reversal earned nothing.
- **The extended-hours `rth` flag reads masked minutes as True.** Split the sessions by the clock instead.
- **Small nulls on sparse samples manufacture flags.** A 5-seed coin-flip null on a sparse subsample flagged 27 of
  288 fitted-model cells at |z| ≥ 3, and every one vanished on fresh rows.
- **A deep net's training seed is part of its null.** Direction AUCs of 0.54–0.55 from one seed were 0.48–0.51
  with others.

## Where everything is

The study's code ran in a session scratchpad (`scratchpad/direction/`):
- the shared harness `dirlab.py` (loaders, labels, nulls, statistics);
- `confirm.py` / `confirm_ext.py`, the one-shot runners with their pre-registered bars;
- one folder per method (`m1_serial` … `m9_ext`, `r2_*`, `darts/*`), each with a `RESULTS.md`;
- the critics' and verifiers' folders;
- the lead's log `lead/LOG.md`.

The scratchpad is not part of the repo. The pieces worth keeping can be promoted into `research.py` and a notebook
on request.
