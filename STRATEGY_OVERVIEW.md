# Strategy overview, verdicts and five proposals

*Status as of 2026-10-21. It covers every folder in `strategies/` (28, including the template) and the research
done in and around them. Numbers are the strategy docs' own (README, IDEAS, RESEARCH_LOG, saved diagnostics
notebooks); where a doc and an older research note disagree, the doc wins. Many shipped study names were bumped
after engine fixes (wallet `lock_cash` 2026-09-27, portfolio `lock_cash` 2026-10-19) and have not been re-run:
a number below then comes from the older version named next to it.*

---

## 1. The short answer

Out of 27 strategy and research folders, one signal family has survived every test this repo knows how to run:
**short interest** (FINRA days-to-cover). It drives `informed_flow` (long only, v41: PASS) and
`informed_flow_ls` (long/short, v5: shipped untuned, paper trading). One more strategy shows a real, if
fragile, bear-market profile: **`trend_following`** (daily long/short breakouts made +43% over buy-and-hold
in 2022, almost all of it from the shorts). Everything else either has no edge, an edge too small to pay
its costs, or has not been run on the current engine.

| | folders |
|---|---|
| works to some degree | `informed_flow`, `informed_flow_ls`, `trend_following` (bear-market hedge only) |
| real effect, too small or too fragile to trade | `lead_lag`, `gap_and_go` (fade direction), `options_hedge` (cheap vs dear straddles), `classic_signals` research loop (periodicity, peer catch-up, momentum, earnings drift), `burst_scan` + `darts_forecast` (move SIZE, not direction), `value_factor` (gross profitability only) |
| no edge (falsified) | `momentum_swing`, `turnover_momentum`, `trend_following_1m`, `squeeze_breakout`, `volume_momentum`, `level_flow`, `burst_scan` (as a strategy), `classic_signals` (as a strategy), `tori_trades_trendline`, `darts_forecast` (as a strategy), `fade_movers`, `option_signals` |
| not run on the current engine / unresolved | `donchian_breakout`, `keltner_reversion`, `candle_scalp`, `support_bounce`, `conviction_flow`, `overnight_open`, `orderbook` (data survey only) |

Three folders appear twice: a failed strategy with a real by-product. `baseline` is the buy-and-hold hurdle,
not a strategy; `_template` and `informed_flow_ls_dev` are scaffolding.

The pattern behind the verdicts is consistent across hundreds of tests:

1. **The size of a move is predictable; its direction is not.** Bursts, activity, volatility, the clock and
   option-implied moves all forecast *how much* a stock will move (big-move AUC 0.72, 15-minute range R² 0.66).
   Nothing forecasts *which way* intraday, at any bar size from 1 minute to 1 hour, with any model from
   ridge regression to TimesFM.
2. **Costs decide everything below a day.** A $1,000-$10,000 order pays 6-40 bp per round trip. Every real
   intraday effect found (lead-lag 0.6 bp, peer catch-up 0.4-3 bp, closing periodicity 7 bp) is smaller.
3. **Untuned designs beat tuned ones.** In `informed_flow` (v27-v41), `informed_flow_ls`, `classic_signals`
   and `tori_trades_trendline`, the optimizer fitted noise. Tuning score vs next block: ρ = 0.06. Every
   shipped survivor is a fixed, pre-registered design.
4. **Bear-market evidence is thin.** Only daily strategies can be tested in a falling market (1-minute data
   starts 2025-06 and covers only a bull tape), and in practice only 2022 has been measured.

---

## 2. How things are judged here, and what data exists

**The yardstick.** Every strategy trades one shared $10k book (or a $30k account split into slots), pays one
cost model ($0.005/share, $1 minimum, 2 bp spread, 1 bp slippage, plus the real order size), fills at the
next bar's open, and is scored against an equal-weight buy-and-hold of the same names, never against zero:

- `excess_cagr` = CAGR minus buy-and-hold CAGR (geometric, per year);
- `alpha_cagr` = CAGR − beta × buy-and-hold CAGR, for books that are not fully invested or are long/short.

**"Better than a coin flip"** has a precise meaning in the repo's best tests: the real book against 100-500
books with the same rules whose symbols were shuffled inside dollar-volume tenths. Per-bar random ranks were
abandoned because they churn the book daily and make the test unfailable (z = +267 in `informed_flow` v25).

**Windows.** Train, a warm walk-forward (signals computed on the long history, scored only on the window),
and a holdout read as few times as possible. The 2025+ holdout has now been read 10+ times by `informed_flow`
alone, so its numbers count only as warnings there.

**Data available** (checked against the DB on 2026-10-21):

| data | span | notes |
|---|---|---|
| daily bars | 2015-01 → 2026-10 | 462 names in Jan 2015, 2,277 in Oct 2026 |
| 1-minute bars | 2025-06-18 → 2026 | ~515 names until March 2026, ~1,750 after; extended hours to 2026-06-26 |
| short interest (FINRA) | 2018-01 → 2026-09 | 2,322 names |
| insider trades (SEC Form 4) | 2015 → 2026-10 | 2,112 names |
| fundamentals (SEC XBRL) | multi-year | screened at read time (unit slips, frozen tags) |
| analyst ratings / price targets (Yahoo) | multi-year | the feed stops for some names in Oct 2024 |
| option chains (MarketData, EOD) | 2025-10 → 2026-07 | ~1,750 names; one trial year |
| ETFs | SPY, GLD, SLV since 2015; 6 sector SPDRs since 2023-10 | no TLT, IEF, QQQ, IWM, inverse or low-vol ETFs |

**Survivorship.** The DB holds no company that delisted before 2026 (no SVB, First Republic, Twitter …).
Long books therefore look better than reality and short books worse, and the bias is strongest in bear
markets, when the best shorts went bankrupt.

**The falling markets inside the daily data** (SPY drawdowns deeper than 10%, from the DB):

| episode | peak → trough | SPY drawdown | testable with |
|---|---|---|---|
| 2015-16 | 2015-05-21 → 2016-02-11 | −14.4% | prices, insider (no short interest yet) |
| Q4 2018 | 2018-09-20 → 2018-12-24 | −20.2% | prices, short interest, fundamentals |
| COVID crash | 2020-02-19 → 2020-03-23 | −34.1% | all daily data |
| 2022 bear | 2022-01-03 → 2022-10-12 | −25.4% | all daily data |
| tariff fall | 2025-02-19 → 2025-04-08 | −19.0% | all daily data, 1m for ~515 names |
| *factor crash* | *2026-07 → 2026-09* | *(market up)* | *momentum and short interest both broke here* |

---

## 3. Scoreboard: every folder

| folder | type | status | verdict | headline number | in falling markets |
|---|---|---|---|---|---|
| `baseline` | 1d buy-and-hold | hurdle | n/a | test-pinned to `backtest.benchmark` | is the market |
| `informed_flow` | 1d long-only, short interest | v41 PASS | **works (thin)** | walk-forward alpha +8.8%/yr (P 94%); holdout alpha +1.4% | alpha positive in 2022 (v31-v41: +3.8 to +17.1%) but beta 1.1-1.3, so it falls with the market |
| `informed_flow_ls` | 1d long/short, short interest + put skew | v5 shipped untuned; paper trading | **works (thin)** | walk-forward alpha +10.5%/yr (P 0.98), beta 0.13; fresh names +5.5% | **2022: +7.8% vs −18.1%** (v1; short leg +11.3%) |
| `informed_flow_ls_dev` | research copy | loop 2 running | — | no results yet | — |
| `conviction_flow` | 1d informed_flow fork + filters | 5-trial smoke | not run; fold into informed_flow | — | not measured |
| `value_factor` | 1d long-only value + quality | 5-trial smoke | no selection edge | composite IC ≈ 0; gross profitability IC +0.063 (t 1.67) | not measured |
| `turnover_momentum` | 1d long-only | IC gate failed | **falsified** | top-turnover IC −0.023 (wrong sign) | IC sign flips by year |
| `momentum_swing` | 1d long-only breakouts | v1/v2 run | **falsified twice** | train excess −10.8%, holdout −12.7% | 2022 excess +9.5% (regime gate) |
| `trend_following` | 1d long/short trend starts | v7 run, v8 not | **hedge-like, unproven** | holdout excess +6.1% (−5.5% without LITE) | **2022: +29.4% vs −13.6%** (shorts +28%); lost 2023-24 |
| `trend_following_1m` | 1m same code, 10 stocks | v1 run | no edge | +1.7 bp/day (t 0.84) | flat in falling months |
| `donchian_breakout` | 1m breakouts | v8 not run | unresolved | none documented | not measured |
| `squeeze_breakout` | 1m gated breakouts | v1 run, v4 not | no edge | ~+3% train, dies at 6/5 bp; holdout negative | not measured |
| `volume_momentum` | 1m volume spikes | superseded | no edge | −78.7%/yr at defaults | not measured |
| `keltner_reversion` | 1m dip buying | v5 not run | unresolved | old holdout +8.2% on a contaminated universe | not measured |
| `candle_scalp` | 1m reversal candles | smoke only | not run | — | not measured |
| `support_bounce` | 1m tested-support bounce | v4 not run | unresolved | v3 (old engine) shows an exposure bet | sits out bear regimes by design |
| `level_flow` | 1m level bounce/break, L/S | event study failed; study not run | no edge | placebo levels do as well as real ones | not measured |
| `gap_and_go` | 1m opening-range breakout on gaps | v2 run | **falsified** | pick: train +17% excess → holdout −84% (−96% at the pass-bar costs) | not measured |
| `burst_scan` | 5m activity bursts, L/S | phase 1 | **FAIL both sides** | long −44% alpha, short −29% | not measured |
| `classic_signals` | 1m Bollinger + filters, L/S | full runs | **tuning fits noise** | score → next block ρ = 0.06 | NOW fell, the strategy still lost |
| `tori_trades_trendline` | 15m trendline bounces, L/S | 3 windows | **no edge** | defaults −14.1 bp/trade (t −3.7) on 2025 | shorts lost less; still lost in 2025 |
| `darts_forecast` | 5m-1h ML forecasts | 4 gates + 10 studies | **no edge** | TEST alpha −13.5%/yr, 0/4 gates | not measured |
| `lead_lag` | 1m-1d cross-stock lead-lag | research | **NO-GO** | real, 0.6 bp vs 6.8 bp cost | not measured |
| `fade_movers` | daily/intraday fade of top movers | research | **no edge** | −11.9 bp/day gross (t −2.2) | not measured |
| `overnight_open` | night → first hour | pre-registered | **not run yet** | — | not measured |
| `options_hedge` | straddles on outlier days; option premia | 2 studies | no screen passes; one relative effect confirmed | cheap vs dear straddles +11.0% (z 3.37), ≈ one bid-ask spread | protection costs at least what the move is worth |
| `option_signals` | options → stock direction | research | **0 of 12 pass** | all \|t\| ≤ 1.2 | not measured |
| `orderbook` | free L2 data survey | data only | — | — | — |

---

## 4. Strategy by strategy

### 4.1 Daily, alternative data (the family that works)

#### `informed_flow` — long the stocks short sellers avoid

- **Idea.** Short sellers are informed. A high days-to-cover (short interest ÷ daily volume) is bearish, so
  holding the *least*-shorted names should beat the market over weeks to months. Long only, because the short
  leg's profit mostly goes to the stock lender as a borrow fee.
- **Setup (v41).** Daily, Liquid500 (point-in-time top 500 by dollar volume, 497 names). Rank on minus the
  42-bar average days-to-cover, inside 5 dollar-volume groups. 30 slots; buy at rank ≥ 0.90, sell below 0.80;
  market-on-open fills, $1,000 orders. Nothing is searched (`n_trials: 1`).
- **History.** 41 versions over three months. It started as a five-sleeve composite (insider, analyst ratings,
  price targets, short interest, recency) tuned by Optuna, went through every objective trap (absolute Sharpe
  bought exposure, excess alone bought high-beta concentration, volatility became an exposure dial), and ended
  as one sleeve chosen by a pre-registered rule (sleeve t ≥ 3). Rejected with measurements: Liquid1000,
  20 slots, `min_events: 0`, a 126-bar window (failed to replicate), every risk overlay.
- **Which signals carried information** (beta-adjusted 63-day IC, train):

  | sleeve | IC | t |
  |---|---|---|
  | short interest | +0.052 | 4.7 (positive every year) |
  | analyst ratings | +0.022 | 1.9 (holdout −0.022) |
  | insider | +0.013 | 0.4 |
  | price targets | +0.001 | 0.3 |
  | recency | −0.013 | −0.9 |

  Analyst news and price targets are priced on the day itself (revision day ±1%, t 13-17, nothing after the
  next open).
- **Results (v41).**
  - Walk-forward 2022-01 → 2026-09 on a rolling point-in-time universe: CAGR +19.7% against buy-and-hold
    +9.6% and SPY +12.0%; alpha +8.8% (P 94%); beta 1.14; it beat all 500 shuffled books.
  - Holdout 2025-01 → 2026-09: alpha +1.4% (P 56%). 2025 +9.4%, 2026 −9.4%.
  - The last partial period (2026-07 → 09) had alpha −41% at beta 2.1.
- **Verdict.** "PASS — every hard gate … Against SPY, the index a live account would hold instead, the edge
  is thinner … most of it in 2025." "Paper-trade first."
- **Falling markets.**
  - 2022: alpha positive in every version v31-v41 (+3.8 to +17.1%), and the fixed short-interest book had
    alpha +12.8% / +10.4% in the two halves. But at beta 1.1-1.3 the book still lost money in absolute terms.
  - 2018 Q4 and the 2020 crash were never backtested. On forward-selected names the signal read IC −0.055
    (t −3.4) over 2018-07 → 2020-06.
  - Every overlay tried (cash switch, volatility target) cost alpha. The volatility target cut the book to 25%
    invested at the April 2025 low.

#### `informed_flow_ls` — the same signal, long/short

- **Idea.** Also *short* the most-shorted names, so the book earns the gap between the two ends of the ranking
  and can make money when the market falls. A second signal, the 20-day change in put skew, ranks names short.
- **Setup (v5).** Daily; rolling point-in-time top 500. Portfolio mode, 50/50 dollar-neutral, 15 longs and
  15 shorts on a $30k account. The engine locks short proceeds and charges borrow tiers of 0.5 / 3 / 10% a
  year. Squeeze stop at +50%; never short a name above 30% of shares short.
- **Tested.**
  - A nested Optuna walk-forward with 8 hard gates, run 5 times; about 30 paired design screens.
  - 100 random-name books; factor attribution.
  - A fresh cross-section (names ranked 501-1000), fresh time (2020-07 → 2021-12), a foreign-filer study and
    a concentration check.
- **Results.**
  - Walk-forward 2022-01 → 2026-10: CAGR +11.6%, volatility 10.6%, max drawdown −13.6%, beta 0.13, alpha
    +10.5% (P 0.98). It beat all 100 random books (their 95th percentile was +4.6%).
  - Stress: +9.5% at 2× costs and +6.5% with every leg filled 50 bp late.
  - Fresh names: +5.5% (P 0.90). Fresh time: +2.7% (P 0.66).
  - Tuning never beat the fixed config durably (between −4.0 and +2.2 points; median PBO 0.6-0.74). The
    nested primary therefore FAILED its own gates and the untuned config ships.
- **Caveats.**
  - A third of the long side is a measurement tilt: foreign filers' FINRA short interest covers only the US
    line, so they look lightly shorted. Excluding them halves the alpha to +5.1%.
  - Without its top 5 names (NVDA, CLS, PBR, AMD, KGC) the alpha is +4.9%; without the top 10, +1.3%.
  - 2026 is the book's worst year.
- **Verdict.** "A real but thin short-interest edge, amplified by a momentum/turnover tilt and a foreign-filer
  tilt that both paid in 2023-2025." "What to expect forward: less than the table."
- **Falling markets: the strongest evidence in the repo.** Legs by year for the v1 untuned design:

  | year | long leg | short leg | book | equal-weight market |
  |---|---|---|---|---|
  | 2021 | +15.4 | −1.7 | +12.9 | +28.1 |
  | **2022** | **−3.4** | **+11.3** | **+7.8** | **−18.1** |
  | 2023 | +13.9 | −8.5 | +4.7 | +24.7 |
  | 2025 | +35.8 | −4.8 | +29.2 | +23.3 |

  The short leg lost money in 6 of 7 years and paid in the one bear year: it behaves like a hedge. In v5 the
  2022 halves had alpha +9.9% and +21.5%. Side betas are 1.11 (long) and 0.83 (short), so a dollar-neutral
  book still carries net beta 0.13 overall and 0.39 in 2026. Q4 2018 and the 2020 crash are not measured:
  scoring starts 2020-07.

#### `conviction_flow`, `value_factor`, `turnover_momentum`, `baseline`

- **`conviction_flow`** is an old `informed_flow` fork with extra filters (quality, agreement between sleeves,
  a score floor) on 100 names. Only a 5-trial smoke has run. Its own IDEAS recommends folding the two useful
  levers into `informed_flow` and deleting the folder.
- **`value_factor`** is long-only value (book-to-price, earnings yield) plus quality (gross profitability ÷
  assets) on 100 names, top 15, monthly.
  - Book-to-price IC −0.013 and earnings yield +0.003: dead. Gross profitability IC +0.063 (t 1.67) is the
    only live factor. "So gp is the entire result."
  - After the fundamentals repair: composite IC +0.016 (t 0.42) and excess +3.9%/yr in train. "Selection adds
    little over owning the universe."
  - A sector-demeaned profitability hybrid read IC +0.048 (t 1.84) in a probe and was never built. The full
    study, walk-forward and holdout have never run. Falling markets: not measured.
- **`turnover_momentum`** tested Medhat-Schmeling's "one-month momentum among high-turnover names". Its
  pre-registered IC gate failed on three populations: high-turnover names *reverse* (IC −0.023 to −0.046),
  and no slice reaches |t| > 1.6. "Falsified on this data." The IC sign flips year to year (2018 −0.050,
  2021 −0.077, 2023 +0.006): "regime, not an edge".
- **`baseline`** is equal-weight buy-and-hold, bought once and left to drift. Its rolling top-500 market made
  −18.1% in 2022 and +23-28% in each of 2021, 2023, 2024 and 2025.

### 4.2 Daily, price only

#### `trend_following` — long/short trend starts (the bear-market performer)

- **Idea.** Time-series momentum both ways: join a decisive breakout, ride it with a wide ratcheting ATR stop,
  and let 12-1 month relative strength decide which breakouts get one of 20 slots.
- **Setup (v7).** Daily, Liquid500, portfolio mode with 20 equal slots of ±1/20. Entry on a close beyond the
  9-bar high/low with ADX ≥ 34; a 14 × ATR(37) chandelier stop; exit on the stop or an opposite start. Shorts
  borrow for free (no `borrow` key).
- **Results.**

  | window | strategy | buy & hold | excess | beta |
  |---|---|---|---|---|
  | train 2022-24 | +10.6%/yr | +4.8%/yr | +5.7% | −0.17 |
  | · 2022 | **+29.4%** | **−13.6%** | **+43.1%** | −0.37 |
  | · 2023 | | | −30.7% | −0.68 |
  | · 2024 | | | −13.2% | |
  | holdout 2025-01 → 2026-09 | +24.9%/yr | +18.8%/yr | +6.1% | +0.25 |
  | · holdout without LITE | | | −5.5% | |

  - Shorts made +28% of the book in 2022, then −19% in 2023 and −7% in 2024. **All of the train excess is
    2022.**
  - The sides only work together: short-only alpha +7.7% at beta −0.78; long-only alpha −5.1%.
  - Five names made 75% of train P&L and 92% of holdout P&L.
- **Verdict.** "A crisis-alpha profile … The holdout rests on one trade … The open-fill assumption carries
  most of the edge."
- **Falling markets.** The best bear year in the repo (2022), from shorting breakdowns. But a market regime
  gate (longs only above the 200-day average) was rejected: −5.6 points, t −5.7, because "it pins the book
  short into bear-market rallies".
- **Open.** It needs a rolling walk-forward, which is its own top priority.

#### `momentum_swing` — Qullamaggie-style breakouts and episodic pivots

- **Idea and setup.** Long-only daily breakouts from tight bases in leaders, and gap-ups on 3× volume in
  neglected names, with an ATR stop, a 10-day trail and a fast regime gate, on Liquid500.
- **Results.**
  - The event test failed: the traded signal has t < 2 (breakouts +0.8% at 42 days, t 1.1).
  - Shipped defaults: excess −10.8% (train) and −12.7% (holdout).
  - Best tuned trial: −1.8% on train and −12.1% on the holdout. No trial in the v1 or v2 studies beat the
    hurdle.
- **Verdict.** "Falsified on this data, twice … wins on risk, loses on return."
- **Falling markets.** 2022 was the one positive year (excess +9.5%: −5% against the hurdle's −15%), thanks to
  the regime gate. Every other year was negative.

### 4.3 Intraday, price only (1-minute and 5-minute bars)

Two problems apply to all of these:

- **One regime only.** The intraday data covers one strong bull window, 2025-06 → 2026-06 (the Morning200
  basket made +47%/yr), so none can be judged on falling markets.
- **A contaminated universe.** `HighOctaneTop20` was pinned *after* its own window (its buy-and-hold made
  +460%/yr), which voids every study run on it: `donchian_breakout`, `squeeze_breakout`, `volume_momentum`,
  `keltner_reversion`, `candle_scalp` and `support_bounce`.

| folder | idea | what was measured | verdict |
|---|---|---|---|
| `trend_following_1m` | trend_following's code on 1m bars of Morning10 | +1.7 bp/day (t 0.84); 272 of 311 exits at the close; pessimistic fills −3.6% alpha | "One-minute bars do not rescue the strategy" |
| `donchian_breakout` | 30-minute channel breakouts, trailing ATR stop | old journals (up to 1,000 trials) never written up; no train/holdout split | unresolved |
| `squeeze_breakout` | breakouts gated by squeeze + volume + trend | ~+3% train, Sharpe 0.7; ≈ 0 at 6/5 bp; holdout negative in both rounds. The volume gate is the one that earns its keep | "does not clear costs" |
| `volume_momentum` | volume spike + SMA direction | defaults −78.7%/yr at 24k trades/yr | no edge; superseded by `burst_scan` |
| `keltner_reversion` | buy 6 ATR below the mean in uptrends | v2: holdout +8.2% (Sharpe 3.7) on 20 per-symbol books, old engine, contaminated universe | "every number this strategy has is unverifiable today" |
| `candle_scalp` | hammer / engulfing at an extreme, built to be falsified | smoke only | not run |
| `support_bounce` | buy a twice-tested intraday low | v3 (old engine, never written up): the top 50 trials all hold overnight with 10-ATR stops, i.e. an exposure bet; more touches → *worse* (CAGR 25% → −8%) | unresolved |

#### `level_flow` — multi-week swing levels, bounce or break, long/short

- **Idea.** Take-profit orders cluster at old swing levels (price bounces) and stop orders just beyond them
  (price breaks); it should pay only on high-activity stocks in play.
- **Tested.** An event study on 200 names, and a placebo with levels shifted 0.5-2 ATR.
- **Results.**
  - Bounces: +0.2 bp at 120 bars (t 0.2).
  - Breaks: +9.2 bp (t 2.1), against a 26 bp round trip.
  - The placebo did *better* than the real levels: 73.3% vs 72.7% bounce rate, and +30.6 bp vs +27.8 bp.
    With a median 22 levels a session, 99% of bars sit near *some* level.
  - Sloped trendlines held as often as their parallel twins.
- **Verdict.** By the rule written before the run, neither polarity earned the 1000-trial study.

#### `gap_and_go` — opening-range breakout on overnight gaps

- **Results.**
  - Event study: the gap-and-go thesis loses −10.0 bp at 60 bars (t −2.1), and delaying the fill makes it worse.
    Fading the gap replicates in *direction*.
  - The best pick (fade, both sides) made +63%/yr in-sample at Sharpe 2.6, then −96% excess on the 2-month
    holdout at the pass-bar costs.
- **Verdict.** "Falsified … the direction of the effect replicates, the tradable configuration does not."

#### `burst_scan` — activity bursts, separate long and short classifiers

- **Idea.** A burst (volume ×3 and range ×2 against the same clock time) is new information; learn which
  bursts to buy and which to sell with monthly walk-forward classifiers. Pre-registered gates throughout.
- **The strategy.** Both sides failed every gate.

  | side | within-month AUC | gross bp per pick | book alpha |
  |---|---|---|---|
  | long | 0.536 | −3.5 bp | −44.2% |
  | short | 0.561 | +11.1 bp | −28.7% |

  The books are "turnover times the ~24 bp round trip".
- **The research around it** (move study, move explorer, extended hours, onset, move detectors, direction
  report, free options data) all reached one conclusion:
  - *Occurrence* of a big move within 4 hours is predictable beyond plain volatility: AUC 0.716 against
    0.680, paired t 5.2, in 7 of 7 months.
  - The 15-minute range is forecast at R² 0.66.
  - *Direction* is not: AUC 0.50-0.55, the same on a coin-flip market. All 11 frozen direction rules failed
    out of sample.
  - The test has the power to find planted 2-11 bp edges.
- **Traps found.**
  - The 08:00-08:09 late overnight prints faked a −33 bp, t −13.5 "pre-market reversal".
  - Volume-0 placeholder minutes are a third of extended hours.
  - A within-day "below VWAP" effect (+20 pp, t 8) was selection on the day's own future.
- **Verdict.** "The features tell you *when* and *where* price is about to move — not *which way*." No
  bear-market split exists.

#### `classic_signals` — Bollinger fade and breakout with switchable filters, plus a 120-screen research loop

- **Setup.** 1-minute candles on NOW, GEV and RGTI, long and short. Trend, level, earnings and five option
  filters. Optuna over ~35 knobs, re-tuned walk-forward per stock.
- **Results.**
  - In-sample NOW made +8k to +17k depending on the seed: three seeds picked three different strategies.
  - Tuning score vs the next block: ρ = 0.06 over 117 re-tunings.
  - 10,000 trials: the best score (3.03) sat below the luck line (3.13).
  - 11 stocks, 3,000 trials: validation of the best −$5.6k.
  - NOW fell 27% in 2025 and 33% in H1 2026, and the strategy *still* lost (−6.0k against −5.4k for holding).
- **Verdict.** "The tuning fits noise." "The fade never had an edge" (band-close reversion −5 to +2 bp
  against ~6 bp costs).
- **The research loop (2026-10-19)** screened ~120 ideas with discovery, replication and a Jul-Sep 2026
  validation window. It is recorded only in a temporary scratchpad file
  (`/private/tmp/…/scratchpad/research/LOG.md`); this section keeps its results.

  | family | best result | outcome |
  |---|---|---|
  | 1m ideas on NOW/GEV/COF (gap → day, ORB, hourly reversion, close location) | ≤ +3.7 bp, \|t\| < 2.1 | dead |
  | 15:30 → close periodicity (544 names) | +7.3 bp, t 7.1, both halves | **real**, failed Jul-Sep (+1.7 / −1.5 / 0.0) |
  | last hour → next hour reversal | −3.6 bp, t −4.5 | real, below 12 bp cost |
  | opening-block periodicity / gap reversal | +28.4 bp, t 4.1 | **artifact**: only when entered AT the 09:30 print |
  | options cross-section (1,700 names: skew, call-put IV, O/S, RV−IV …) | all \|t\| < 1.5 | dead |
  | earnings: pre-report straddles, run-up, post-reaction drift | run-up t 3.8 per event → t 1.1-1.4 as a book | dead after costs |
  | Goyal-Saretto straddles | +4% at mid, ≈ 0 at the quotes | dead as traded |
  | weekly / monthly reversal (daily, 500 names) | t 1.0-1.7, **carried by 2020 and 2022** | weak |
  | 12-1 momentum, long/short | **+1.6-1.8%/month, t 2.2-2.5 in both halves 2018-2026-06** | crashed Jul-Sep 2026 (−7 to −11%/month) |
  | peer catch-up (a name toward its correlation cluster) | **t 5-11, held in Jul-Sep** | real, 0.4-3 bp per trade |
  | cluster → cluster lead-lag; limit-order fades | null rate; −2 to −6 bp per fill | dead (adverse selection) |
  | illiquid names (60 bp costs): reversal, volume, Amihud, 52-week high | nothing after costs | dead |
  | **earnings-surprise drift** (Yahoo EPS, 35k reports, top fifth, 60 days) | discovery +7.7%/yr (t 3.0), holdout +13.4%/yr (t 2.7), beat 300 random books | **failed the never-seen Jun-Oct 2026 window** (−5.8%, worse than all 300) |

#### `tori_trades_trendline` — automated top-down trendline bounces

- **Results.**
  - 21 stocks on 15-minute charts: longs −9.4%, shorts −4.6%, against buy-and-hold +20%.
  - The median trade lost 0.95 R.
  - Lines moved *off* hers did as well or better.
  - Every optimizer converged on "barely trade". The one setting that held from train to validation lost in 2025.
- **Verdict.** "No edge… over three windows and two tuning runs, with 2025 read once."
- **Falling markets.** Shorts lost less than longs and tuned more consistently (ρ +0.77), but the short-only
  settings lost in 2025, including the Feb-Apr fall.

### 4.4 Machine-learning forecasts

#### `darts_forecast` — one 5-minute strategy and ten research studies

- **Scope.** Gradient boosting, ridge, TiDE, TSMixer, DLinear, TFT, NHiTS, BlockRNN and TimesFM 3.0
  zero-shot; bar sizes from 1 minute to 1 hour; horizons from 5 minutes to 5 days; 5 to 40 stocks.
- **Results.**
  - The 5-minute strategy: TUNE −26%/yr; TEST alpha −13.5%/yr, and still −6.7% with no costs. "FAIL, 0 of 4
    gates."
  - No path model beats "no change" out of sample: the best R² was +0.0003 on validation and 0.0000 on TEST.
  - Every validation winner failed TEST. Per-stock leaderboards match selection luck: 0 of 40 stocks clear
    their own bar.
- **What is real.**
  - Move size: a two-line "clock time × recent volatility" band beats TimesFM's band (z −6.5).
  - Trailing activity picks the stocks that move (43% of picks against 27% for random).
  - One model pooled over all stocks beats per-stock models.
- **Verdict.** "Nothing in this folder predicts direction better than luck." No bear window was ever tested:
  every window was mostly rising, and "always up" beat a coin.

### 4.5 Research-only studies

| folder | question | result | verdict |
|---|---|---|---|
| `lead_lag` | does one stock's move predict another's (market, industry, supply chain), 1m → 1d? | 268 pairs survive a strict null and persist, but trade at 0.6 bp against a 6.8 bp round trip; the market reverses itself (t −12.6 at 1m); "small caps lag" was really "rarely-printed names lag" | "NO-GO … about ten times too small" |
| `fade_movers` | short the day's top gainers, buy the losers (close, overnight, on a turn) | −11.9 bp/day gross (t −2.2); validation all FAIL; gainers keep rising overnight; days 2-5 drift replicates gross (+16.9 bp) but nets +3.7 | "found no edge" |
| `overnight_open` | does the night (after-hours, pre-market, 8-K filings) predict a strong first hour? | pre-registered (V1-V4), caches built, not run. Found that EDGAR's JSON acceptance times are 4-5 h late for half the filers | not run |
| `option_signals` | do skew, call-put IV, RV−IV, option/stock volume rank stocks? | 0 of 12 cells pass, all \|t\| ≤ 1.2; three of four lean the wrong way | "None of the four signals ranks these stocks" |
| `options_hedge` | (1) buy straddles on outlier days; (2) earn option premia | (1) 0 of 8 pass; options priced the move right (actual ÷ priced 1.00). (2) Cheap vs dear straddles CONFIRMED on the holdout (+11.0%, z 3.37) but "the edge is about the size of the bid-ask spread"; put-writing keeps a −834% tail | protection costs at least what the move is worth |
| `orderbook` | what free level-2 history exists? | OKX / Bybit crypto archives; IEX DEEP (IEX-only book) since 2017; no free consolidated US book | data survey, nothing tested |

---

## 5. Conclusions

### What works, to some degree

1. **Short interest (days-to-cover level), held for months, ranked within liquidity groups.** It is the only
   signal that passed a pre-registered IC rule, a rolling point-in-time walk-forward, shuffled-symbol books, cost
   and late-fill stress, and a fresh cross-section.
   - Its size is modest and shrinking: train alpha +7%, walk-forward +8.8-10.5%, fresh names +5.5%, holdout
     +1.4% (long only).
   - It is concentrated in a few big winners and partly a foreign-filer measurement tilt.
   - It went the wrong way in 2026 Q3.
   - It is still the one thing here that beats a coin flip in a way that has replicated.
2. **Long/short construction turns it into a bear-market asset.** The dollar-neutral book made +7.8% in 2022
   while its market fell 18%, because the short leg (+11.3%) paid exactly when needed.
3. **Shorting breakdowns pays in a persistent bear.** `trend_following` made +43% over buy-and-hold in 2022,
   but lost the same amount back in 2023-24. It works as a hedge, not as a standalone strategy.

### Real effects that do not pay (yet)

- **Movement and volatility are predictable.** Big-move AUC 0.72, range R² 0.66, 30-day IV predicts the next
  session's range. This is useful for sizing and stops, not for direction.
- **Peer catch-up** (t 5-11), **closing-half-hour periodicity** (t 7, faded later), **pair lead-lag**: real,
  10× too small per trade.
- **12-1 momentum** (t 2.2-2.5 both halves) and **earnings-surprise drift** (passed discovery and holdout):
  both broke in the Jul-Oct 2026 window.
- **Cheap vs dear straddles** (Goyal-Saretto): real but the size of the spread.
- **Gross profitability** is the only live fundamental factor; value itself is dead in this universe.

### What does not work

- Predicting intraday direction from price, volume, levels, candles, bands, trendlines, bursts, options or
  any ML model (12,000+ cells in the direction study alone).
- Classic chart patterns as published (Qullamaggie breakouts, gap-and-go, Bollinger fades, Tori trendlines,
  support bounces): each lost to buy-and-hold or to a placebo.
- Fading the day's movers, the overnight drift (a bull-window effect: +29 bp/night in 2025-26, ≈ 5 bp
  2015-25), turnover-conditioned momentum, insider and analyst sleeves in large caps.
- Tuning: optimizers find the luckiest region of a flat surface, and their picks do not carry forward.

### Not yet answered

- Six intraday strategies have never been run on the current engine with a clean universe.
- `overnight_open` is pre-registered but not run. `value_factor`'s full study and holdout have not run.
- No strategy has a backtest through **Q4 2018 or the 2020 crash**. `trend_following` has no rolling
  walk-forward.

### Lessons any new strategy must respect

- Score against buy-and-hold of the same names, with `alpha_cagr` for anything not fully long; never absolute
  Sharpe, never cumulative returns.
- Use a rolling point-in-time universe (a universe chosen on today's liquidity is tomorrow's winners); score
  warm; load the train panel on its own window.
- "Better than a coin flip" means beating symbol-shuffled books inside liquidity tenths, and beating a
  best-of-N luck bar when many settings were tried.
- Pre-register the rule and the pass bar; read every holdout once. Fixed designs beat tuned ones here.
- Price the real order size, stress 2× costs and +50 bp late fills; enter after the print a signal uses.
- Thin alternative data goes quiet instead of failing: check breadth per bar, in every traded universe.
- The DB is survivors only: long results are flattered, short results understated.

---

## 6. Five proposals that should also work in falling markets

### 6.1 How a strategy can beat buy-and-hold when the market falls

A long-only book with beta 1 cannot beat buy-and-hold in a bear market unless its stock picking beats the
fall itself; `informed_flow`'s +9% alpha does not offset a −25% market. There are only three ways to do
better:

1. **Hold less market:** cash or a hedge. Timing rules have failed here: the 200-day gate cost −5.6 points,
   and the volatility target sold the April 2025 low.
2. **Hold negative market:** shorts. These worked in 2022 (`trend_following`, `informed_flow_ls`).
3. **Hold no market at all:** a long/short book whose return is pure stock selection, which is the same in
   a rising or a falling market.

All five proposals below use (2) or (3), and none relies on calling the market's turn. They are ranked by
their chance of a real edge given the evidence above. All are daily, because only daily data covers falling
markets. All fit the strategy contract with no engine change, unless stated otherwise.

**A common test protocol**, the repo's own best practice, written down once:

- **Pre-register** the rules and pass bars in the folder's IDEAS.md before any result is read. Ship literature
  defaults; no search.
- **Universe:** a rolling point-in-time top 500 re-formed every 6 months, scored warm, from the earliest date
  each signal allows.
- **Beat a coin flip:** 100+ symbol-shuffled books inside dollar-volume tenths, plus a block-bootstrap
  P(alpha > 0) ≥ 0.90.
- **Beat buy-and-hold:** alpha against the equal-weight market *and* against SPY.
- **A bear table:** return in each drawdown episode of section 2 (2018 Q4, 2020, 2022, 2025), against the
  market; beta in down months; downside capture. Four episodes are anecdotes, not statistics, so this is a
  *must-not-break* gate. Significance comes from the full record.
- **The 2026-07 → 09 factor crash** as a named stress window.
- **Replication:** names ranked 501-1000, never looked at.
- **Costs:** the real order size, 2× costs, +50 bp late fills, 2× borrow, and the book without its top 5 names.
- **Paper trade** from the untouched data after 2026-10-17 before any money.

One engine detail matters for every proposal: portfolio mode rebalances *per name, only when that name's
target changes*. A book-level scale (a beta-neutral split, a volatility target) must therefore be updated at
scheduled rebalances, not daily, or every holding is re-traded every day.

### Proposal 1: short-interest long/short, made beta-neutral and tested on the two bear markets it has never seen

*A refinement of `informed_flow_ls` v5.*

- **Why it can work.** Short sellers are informed, and borrowing costs keep the effect from being arbitraged
  away (Boehmer-Jones-Zhang; Drechsler-Drechsler). It is the only signal in the repo that has replicated, and
  its long/short version already made money in 2022.
- **Change 1: beta-neutral sizing.** At each rebalance, set the long/short dollar split so the two sides'
  trailing betas cancel. Today's dollar-neutral book carries net beta 0.13, and 0.39 in 2026; in a −30%
  crash a 0.4 beta costs 12 points. The research loop already measured this: `long_share` 0.43 gave beta
  0.00 for −0.6 points of alpha. It was rejected as "a risk choice", and for a book meant to survive bear
  markets it is the right choice.
- **Change 2: a pre-registered bear test on 2018-07 → 2020-06.** This window holds Q4 2018 (−20%) and the
  COVID crash (−34%) and has never been scored.
  - Short interest exists from 2018-01, so the 42-bar window is warm by March. The skew sleeve is silent
    before Oct 2025 anyway, so the book is short-interest-only there.
  - Freeze v5's rules and read the window once.
  - Pass: alpha > 0 over the window, and in each of the two drawdowns the book loses less than half of what
    the market lost.
  - This is a real risk: on forward-selected names the signal read IC −0.055 (t −3.4) over exactly this
    window. That is the reason to run it before trading, not after.
- **Change 3 (needs the user's decision): exclude foreign filers** (RESEARCH_LOG §26). That leaves +5.1% alpha
  on the top 500, without the measurement tilt. A clean version needs a form-type flag in `features.py`, which
  is an engine proposal.
- **Expected.** Alpha around +4-6%/yr at beta ≈ 0: roughly +5% in a year when the market falls 20%.
- **Kill if:** it fails the 2018-2020 test, or paper fills lose more than 50 bp per leg to the open.
- **Effort:** low. Config, plus one cell in the walk-forward notebook.
- **Status (2026-10-21, `informed_flow_ls/RESEARCH_LOG.md` §37, §38P):**
  - **The 2018-20 test ran and failed:** v6 alpha −3.5%. But v6 lost only 60% as much as the market in Q4 2018
    (−10.8% vs −18.2%) and held flat in the COVID crash (−1.5% vs −37%). The shortfall is stock selection in
    2018-H2, not beta.
  - **Beta sizing is built** (`strategies/informed_flow_ls_p1`, `sizing: beta`), along with two further
    iterations and an optimizer-derived variant. None passed the pre-registered bar.
  - **Beta sizing is a risk option:** about −1 point of alpha for max drawdown −14.0% → −8.7%, and it is better
    in 4 of 5 falling-market episodes. On a locked set of names never read before (ranked 1501–2000) it costs
    0.1 point.
  - **The foreign-filer exclusion hurts every test**, 2018-20 included.

### Proposal 2: crash-protected residual momentum, long/short

*New. Daily, prices only.*

- **Idea.** Rank stocks on their 12-1 month return *after removing their market exposure*, scaled by their own
  residual volatility. Long the top, short the bottom, beta-neutral. Shrink the whole book when its own recent
  volatility jumps.
- **Why it can work.**
  - Momentum is the most persistent cross-sectional anomaly (investors under-react to news).
  - Its known flaw is crashes in sharp rebounds after a bear market (−73% in 2009). They happen because the
    short leg fills up with high-beta stocks that rebound violently.
  - Residual momentum (Blitz-Huij-Martens 2011; Gutierrez-Prinsky 2007) removes that beta bet, roughly halves
    volatility and does not crash in rebounds.
  - Scaling exposure by momentum's own trailing volatility (Barroso & Santa-Clara 2015) cuts it before most
    crashes, because momentum's volatility rises first.
- **Repo evidence.**
  - Raw 12-1 momentum long/short made +1.6-1.8%/month, t 2.2-2.5 in *both* halves of 2018 → 2026-06 (research
    loop). That is the second-strongest cross-sectional result in the repo.
  - It crashed −7 to −11%/month in Jul-Sep 2026, a ready-made out-of-sample test of exactly the protection
    this design adds.
  - `trend_following`'s 2022 gains were breakdown shorts, i.e. momentum's short side.
  - `informed_flow_ls`'s alpha already loads 0.23-0.28 on momentum.
- **In falling markets.** Persistent losers keep falling, and the short leg earns. The danger is the rebound,
  which is what the residual ranking and the volatility scaling target.
- **Setup.**
  - Daily, rolling top 500 from 2016 (2015 is warm-up).
  - Residual return from a 252-day regression on the equal-weight market. The signal is the residual return
    from t−252 to t−21, divided by residual volatility.
  - 20 longs and 20 shorts. Monthly re-ranking with `q_in` 0.90 / `q_out` 0.80 bands; beta-neutral split;
    borrow tiers.
  - Book scale = min(1, 10% ÷ the book's trailing 126-day volatility), recomputed monthly.
- **Pre-registered arms** (nothing tuned): raw, residual, and residual + volatility scaling.
- **Pass:**
  - the third arm has alpha > 0 (P ≥ 0.90) and beats the shuffled books;
  - its Jul-Sep 2026 loss is at most half of the raw arm's;
  - it is non-negative in at least 3 of the 4 bear episodes;
  - it is positive on names 501-1000.
- **Risks.**
  - A junk rally (2020-H2/2021, Jul-Sep 2026) hurts every momentum book.
  - Turnover; the bands are there to keep it near 100 trades a year.
  - Survivorship makes the short leg look worse than it would have been, which is conservative.
- **Effort:** low to medium. A new folder; no engine change.

### Proposal 3: the adaptive hedge, long the short-interest book plus breakdown shorts that appear only in falling markets

*A combination of `informed_flow` (v41) and `trend_following`'s short side.*

- **Idea.** A permanently hedged book gives up the bull market; a long-only book gives back everything in a
  bear. Breakdown shorts size themselves:
  - in a rising market few stocks make 9-day lows with ADX ≥ 34, so the book stays nearly fully long and keeps
    `informed_flow`'s bull-market alpha;
  - in a falling market many do, the short side fills up, and net exposure falls toward zero or below.

  The hedge comes from the cross-section, not from a market-timing rule (the 200-day gate failed: −5.6 points).
  Of the five proposals, this is the only one that aims to beat buy-and-hold in both regimes, not just on alpha.
- **Repo evidence.**
  - `informed_flow`: walk-forward alpha +8.8% at beta 1.14.
  - `trend_following` short-only (train 2022-24): alpha +7.7% at beta −0.78.
  - Its shorts made +28% in 2022 and −19%, −7%, +3%, −6% in the following years: over 2022-26 the short side
    was roughly a *free* hedge that paid at the right time.
  - Short starts fell −5.9% over 63 days (raw, t −3.4), though only −0.8% relative to the market (t −1.6):
    much of what the shorts earn is market exposure, which is the point of a hedge.
- **Setup.**
  - Portfolio mode, 30 slots of 1/30.
  - Longs are `informed_flow`'s rule unchanged.
  - A short start (close below the 9-bar low, ADX ≥ 34, −DI > +DI) takes a slot at −1/30. If no slot is free,
    it sells the weakest-ranked long.
  - At most 20 short slots, so net exposure stays above −33%. Shorts exit on `trend_following`'s chandelier or
    an opposite start.
  - Borrow tiers; a +50% squeeze stop; never short above 30% short interest.
- **One pre-registered variant:** shorts only among names in the crowded half by days-to-cover. It combines
  the repo's two independent short signals and is tested as a paired screen against no filter.
- **Pass:**
  - walk-forward alpha at least `informed_flow`'s minus 2 points, i.e. the hedge costs at most 2 points a year;
  - max drawdown at most two-thirds of buy-and-hold's;
  - in 2022, net exposure below 50% by mid-year and the book ahead of buy-and-hold by at least 10 points;
  - the bear table on the 2018-2020 window from proposal 1.
- **Risks.**
  - Bear-market rallies squeeze breakdown shorts; that is how 2023 lost 19%.
  - High-days-to-cover shorts are squeeze candidates.
  - The two parts were each validated alone, never together.
- **Effort:** medium. A new folder that copies both rule sets; no engine change.

### Proposal 4: quality minus junk, beta-neutral

*New. Daily, prices plus SEC fundamentals.*

- **Idea.** Long profitable, low-volatility companies; short unprofitable, volatile "junk". Size the book
  beta-neutral, so there are more dollars long than short.
- **Why it can work.**
  - Investors overpay for lottery-like junk, and leverage-constrained investors bid up high-beta stocks
    (Frazzini-Pedersen's betting-against-beta; Asness-Frazzini-Pedersen's quality-minus-junk; Novy-Marx's
    gross profitability).
  - In a crisis money flees to quality, so the factor historically earned most in bear markets (2000-02,
    2008, 2022).
- **Repo evidence.**
  - Gross profitability is the only live fundamental factor (IC +0.063, t 1.67).
  - A sector-relative profitability hybrid read IC +0.048 (t 1.84) and was never built.
  - `informed_flow_ls`'s variant that shorted only unprofitable firms reached +12.0% alpha after late-fill
    stress.
  - The most robust finding in the whole repo is that volatility persists: that persistence is exactly what a
    low-volatility ranking relies on.
- **In falling markets.** It is short by construction the stocks that fall most (high beta, losing money).
  In 2022 unprofitable growth stocks fell far more than the index.
- **Setup.**
  - Daily, rolling top 500 (or 1000 for more junk), monthly re-ranking with bands.
  - Score = average rank of two sleeves only:
    1. profitability: gross profit ÷ assets, with net income ÷ equity for financials so banks are not dropped;
       ranked within groups, never an AND-filter;
    2. minus 252-day volatility.
  - 20 longs and 20 shorts, beta-neutral, borrow tiers.
  - Gross profit ÷ assets needs no market cap, which sidesteps the share-count and ADR defects found in
    `value_factor`.
- **Pass:** alpha > 0 (P ≥ 0.90); beats the shuffled books; positive in 2022; non-negative in at least 3 of
  the 4 bear episodes; positive on names 501-1000.
- **Risks.**
  - Junk rallies (2020-H2/2021, perhaps Jul-Sep 2026).
  - Low volatility lagged badly in the 2023-24 rally, and even beta-neutral it can lose then.
  - Fundamentals coverage must be checked per bar: thin data goes quiet, it does not fail.
- **Effort:** low. Existing features only; no engine change.

### Proposal 5: peer-relative short-term reversal that leans in when markets are stressed

*New. Daily, prices only.*

- **Idea.** Buy stocks that dropped sharply against their own correlation peers over the past week; short the
  ones that jumped against them; hold 5-10 days. Run the book at full size when market volatility is above its
  trailing median, at half size otherwise.
- **Why it can work.**
  - Short-term reversal is the pay of liquidity providers, and that pay rises with market stress (Nagel 2012:
    reversal profits track VIX and peak in crises).
  - Measuring the move against peers strips out the industry news that makes raw reversal fail
    (Da-Liu-Schaumburg 2014). Residual reversal survives costs in large caps (Blitz et al. 2013).
- **Repo evidence.** Four independent hints point the same way:
  - peer catch-up is real and stable (t 5-11, and it held in Jul-Sep 2026), but at 1-10 minutes it is too
    small to pay;
  - weekly and monthly reversal (t 1.0-1.7) was "carried by 2020 and 2022", the bear years;
  - the most-traded names reverse at one month (`turnover_momentum`);
  - breakdowns beat the market over the next month in 2022-23 (`trend_following`). That reversal is what hurt
    trend shorts, and it is this strategy's profit.
- **In falling markets.** It is built to earn most in stress, when forced sellers pay for liquidity.
- **Setup.**
  - Daily, rolling top 500. Peer groups: 30 correlation clusters of trailing 252-day residual returns,
    re-formed every 6 months (the research loop's method, made point-in-time).
  - Signal: 5-day return minus the cluster's. Skip one day, so entry is the open after next and avoids the
    print artifacts.
  - Long the bottom 10%, short the top 10%, 20 + 20 slots, beta-neutral, a 10-day maximum hold.
  - The stress switch is binary and pre-registered.
  - Variant: drop moves made on more than 3× normal volume. High-volume moves carry information and tend to
    continue.
- **Pass:**
  - alpha > 0 net *at 2× costs* (P ≥ 0.90);
  - beats the shuffled books;
  - most of the profit is earned on stress-on days (if not, the bear-market rationale is false);
  - positive in 2020 and 2022.
- **Risks.**
  - Costs: book turnover could reach ~30× a year, roughly 3-4%/yr in fees. That is why it is fifth, and why
    the pass bar is set at double costs.
  - Reversal has decayed in large caps since the 2000s.
  - Four stress episodes are few, so the switch must not be tuned.
- **Effort:** medium. The clustering runs inside `signals()`; no engine change.

### Also considered, and why not in the five

- **Earnings-reaction drift, long/short.** It is the research loop's best event result: discovery t 3.0,
  holdout +13.4%/yr, beat 300 random books. It ranks sixth because:
  - it failed its never-seen 2026 window;
  - drift in large caps has largely disappeared since 2006 (Martineau 2022);
  - it needs an earnings-date feature builder in `features.py` (engine proposal; SEC 8-K Item 2.02 with the
    #39 timestamp fix).

  That same builder would also unlock ideas parked in `gap_and_go`, `momentum_swing` and `burst_scan`, which
  makes it worth proposing anyway.
- **Cross-asset ETF trend** (stocks, bonds, gold, cash) is the textbook crisis-alpha strategy, and yfinance
  can backfill it to 1993, covering 2000-02 and 2008. It is not stock selection, the DB lacks the ETFs, and
  this repo's market-timing evidence is negative. It is still the right test bench if a timing overlay is
  ever wanted.
- **Options as protection or alpha.** Straddles and puts cost what the moves are worth; cheap vs dear is the
  size of the spread; the engine cannot trade options, and the chain history is one trial year.
- **Anything intraday.** Direction is unpredictable, costs dominate, and the 1-minute history contains no
  falling market.

### Suggested order

1. **Proposal 1.** Lowest effort, strongest prior, and its 2018-2020 test also gives proposal 3 its bear window.
2. **Proposals 2 and 4.** Independent factors, each a low-effort new folder. If both pass, a
   three-sleeve market-neutral book (short interest + residual momentum + quality) answers `informed_flow`'s own
   warning that "one sleeve is one factor".
3. **Proposal 3,** once proposal 1's bear test is read.
4. **Proposal 5,** last, because costs are its main risk.

---

## Appendix: doc inconsistencies noticed while compiling this

- `informed_flow/README.md` opening box ("What ships today (v27) … short interest and analyst rating changes")
  and the `config.yaml` header still describe v27. The `sampler` comment says v41 is "NOT YET RUN", but v41 ran
  on 2026-09-26 and passed.
- `informed_flow_ls/README.md`: "in every run the tuned picks did no better" overstates it. v1 (+2.2) and v3
  (+2.0) tuned did beat untuned; the sign flips between runs, which is the real finding.
- `value_factor/config.yaml` still names the retired `info_ratio` objective; `baseline/README.md` too.
- `trend_following_1m/config.yaml` says buy-and-hold made "~+140%/yr"; the README says +54% over 10 months.
- `level_flow_trendlines.ipynb` cites "the placebo in the level_flow README", which the README does not contain
  (it lives only in a research note).
- `squeeze_breakout.ipynb`'s "Read honestly" cell still calls stop fills optimistic; they are pessimistic since
  2026-07-19.
- None of the seven shipped intraday/daily price-strategy study names (momentum_swing v3, trend_following v8,
  trend_following_1m v2, donchian v8, squeeze v4, volume_momentum v4, keltner v5) has been run.
- The `classic_signals` research-loop log exists only in a temporary scratchpad (`/private/tmp/…/LOG.md`)
  and should be copied into the repo if it is to survive.
