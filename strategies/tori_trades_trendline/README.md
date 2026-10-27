# tori_trades_trendline

Tori Trades' trendline strategy, from her video
[*Once you master this strategy, trading becomes EASY*](https://www.youtube.com/watch?v=TuXOgkcYw9E)
(2026-09-24). She draws trendlines by hand with a top-down analysis (monthly chart down to the
trading chart). She enters when price comes back to a line, keeps her stop beyond the line and
stays in until a candle closes through it.

**Step 1 (built 2026-10-19): the lines.** A marimo notebook draws her trendlines automatically
for any stock in the database.

**Step 2 (built 2026-10-19): the trade, long and short, on one stock.** A second notebook replays
her bounce candle by candle (`bounce.py`), and `study.py` runs it on 21 stocks (results below).
It is a research folder like `classic_signals`: there is no `strategy.py` or `config.yaml`, so
the strategy loader and `tsim` skip it.

```bash
uv run marimo edit strategies/tori_trades_trendline/tori_trades_trendline_marimo.py          # the lines
uv run marimo edit strategies/tori_trades_trendline/tori_trades_trendline_trading_marimo.py  # the trade
uv run python strategies/tori_trades_trendline/study.py run       # 21 stocks, ~25 min on 7 cores
uv run python strategies/tori_trades_trendline/study.py report
```

## How the lines are drawn

The full rules are in the docstring of [topdown.py](topdown.py). In short:

1. **Rays.** Downward lines rest on the candles' highs, upward lines under their lows; each is a
   ray from point A to the right edge.
2. **Start at the extreme.** The first downward line starts at the highest high of all history
   (on the monthly chart), the first upward line at the lowest low.
3. **No candle crosses a line.** From point A the ray turns as steep as it can until a candle
   stops it: the steepest ray through a candle that no later candle goes through. That ray also
   passes closest to every candle, so it has the most touches.
4. **Close enough is a touch.** A candle within *close enough* typical candles of the line touches
   it (a typical candle is the median high-to-low range on the chart). A touch counts once
   *confirm* candles have followed it.
5. **Lines connect.** Point B is the line's most recent touch, and the next line starts there.
   Every later candle is further from the line than B, so the next line is always steeper. A ray
   stopped by a candle that is still too new is a *pending* ray (drawn dashed), not a line.
6. **Top-down.** Monthly → weekly → daily → 4-hour → 1-hour → the trading chart (15 minutes by
   default). Each lower chart shows its last *screen* candles. The walk restarts at the last point
   A or B left of the screen, moved onto this chart's candle that holds its price: the lines
   reaching onto the screen are redrawn on the precise candles, and the steeper lines only this
   chart can see are added. Lines that end left of the screen keep their higher-chart drawing,
   unless a candle on the screen goes through one.
7. **Anchors on the line** (optional, yours, 2026-10-27; `on_line`, off in `topdown.py` and
   `bounce.py`, on in the notebooks together with no poke allowance). Point B is the angle
   candle, so both anchors are candle extremes exactly on the line. No candle crosses a line,
   and every line starts exactly at the previous line's point B.
   - **Why:** before, B was the latest candle within *close enough*, so it could sit off the
     line. On NOW that was up to 0.25 typical candles with her defaults and 1 to 7 with #417's,
     and a poke of 0.1 (0.6) typical candles was allowed.
   - **On a lower chart:** a line taken over from a higher chart keeps its point B on the
     lower chart's candle that holds the same price. It sits 0.01 to 0.08 typical candles off
     the line, because that candle lies later in the higher chart's candle.
   - **Short lines:** the strict rules draw more lines (122 against 84 on 12 NOW dates). Very
     short ones stay rare (5 of 122 under 2 candles, mostly a 1-hour line from the January 2025
     all-time high).
8. **Anchor gap** (optional, yours, 2026-10-27; `gap`, off by default). A line drawn on the
   trading chart counts only when its points A and B are at least *gap* of its candles apart.
   A shorter one stays a pending ray, and the chain ends before it.

Every line also records its `origin`: the chart it was first found on. A line a lower chart
only adjusts (it still starts in the same candle) keeps it, while `tf` says where it was last
drawn. Both notebooks style a line by its origin: blue upward (long), orange downward (short).
Monthly and weekly lines are a deeper shade (monthly solid, weekly dash-dot), daily solid,
4-hour long dashes, 1-hour short dashes, 15-minute dots. So a line looks the same in every
chart, and a key sits at the top of the charts.

Time runs on one trading-session clock on every chart (session *n* starts at *n*; nights,
weekends and holidays take no room), so a line keeps one slope on every chart. A monthly candle
sits at its first session, as a chart places it, which is why the lower charts adjust the lines.

## The notebook

1. **The candles**: what the database holds for the stock and what each chart is built from.
2. **The top-down walk**: one tab per chart, with the lines as they stand after it.
3. **The result** on the trading chart, with the later candles faded. Move the *as of* date back
   to replay a stretch, as she does with TradingView's replay.
4. **How one line is found**: from point A, the steeper rays the stopping candle cuts, the touch
   band, the touches and point B.
5. **The lines and the rule checks**: the lines as a table, and her rules checked on every chart
   (no candle on a screen through a line, each line steeper, each starting at the previous B).

| Control | Default | What it does |
|---|---|---|
| Stock | NOW | any ticker with daily candles; the intraday charts need its 1-minute candles |
| Trading chart | 15-minute | the last chart of the walk (4-hour, 1-hour, 15- or 5-minute) |
| Price scale | plain | TradingView's default; *logarithmic* draws lines straight on a log chart |
| Close enough | 0.25 | a touch comes within this many typical candles of the line |
| Too small to see | 0.1 | a candle may poke through a line by this many typical candles |
| Confirm | 2 | candles that must follow a touch before it counts |
| Screen | 150 | candles in view on every chart after the monthly |
| As of | last session | the lines see the candles up to this session's close |

## Judgment calls

The video leaves these open, so they are choices made here:

- **Touches need no swing test.** Any candle within *close enough* counts. This guarantees that
  each next line is steeper (rule 5).
- **Too small to see.** Added after NOW's 1-hour chart. The candle an hour after the all-time high
  was 3 cents lower and stopped the ray from the top: a one-hour "line" that the real line
  crossed by 0.05 cents.
- **Confirm.** In the video she draws through finished swings. A candle still forming cannot
  anchor a line here, so the chain ends with a pending ray where price is at a line right now.
- **Screen.** The candles in view are a fixed number per chart, like TradingView's view after a
  reset.
- **Regular hours only**, the 4-hour candles counted from 09:30 (as on TradingView). Half-days
  drop their stored after-hours minutes (`topdown.HALF_DAYS`; extend it past 2026).
- **History** is the database's: NOW's daily candles start 2015-01-02, though it listed in June
  2012. Its first upward line therefore starts at the February 2016 low (IDEAS.md has the
  backfill).

## First look: NOW (2026-10-19, defaults)

- **As of 2026-10-16.** One downward line, from the all-time high (2025-01-28, 239.62) to the
  2026-10-16 09:30 spike (154.20); the autumn rally broke every steeper downward line. Five
  upward lines, each steeper than the one before: 2016-02 → 2017-12 → 2026-04-10 → 2026-07-23 →
  2026-09-29 → 2026-10-16.
- **As of 2026-03-31**, inside the downtrend, the downward chain is the one she draws: all-time
  high → 2025-10-30 → 2025-12-12 → 2026-03-09 → 2026-03-19 → 2026-03-31.
- The rule checks pass on every chart, on the plain and the log scale.

## The trade (`bounce.py`)

A long trades the upward lines, a short the downward lines; each is the mirror of the other
(tested: shorts on a market turned upside down make exactly the longs' trades).

1. **Set up**: draw the lines with only the candles known at that moment (the higher charts' last
   candle still forming, as on a live chart).
2. **Wait**: long when a 15-minute candle's low comes within the *entry gap* of an upward line
   and the candle closes above it (short: its high, a downward line, a close below). Enter at the
   next open, on the line price meets first. A close beyond a line breaks it: set up again.
   While waiting, the lines are also drawn again at every session's close.
3. **Manage**: the first stop sits *stop room* beyond the line; every new swing low (high) moves
   it, always beyond the line, only in the trade's favour.
4. **Exit**: the stop is hit, or a candle closes more than *close break* beyond the line (out at
   the next open). Set up again.

Decisions at a candle's close, fills at the next open, stops during the candle; one position at a
time with the whole $10,000 wallet in it, profits reinvested, the repo's costs on every order;
the hurdle is buying and holding the same stock.

| Control | Default | What it does |
|---|---|---|
| Sides | both | long, short or both (one position at a time) |
| Entry gap (`band`) | 0.25 | how close a low (high) must come to a line to be a signal, in typical candles |
| Stop room (`room`) | 0.5 | how far beyond the line the stop sits: the break that stops a trade |
| Close break (`give`) | 0 | how far beyond the line a close must be to end the trade |
| Placebo shift (`shift`) | 0 | moves every line toward price (negative: away), to test whether her lines matter |
| Session refresh | on | draw the lines again at every session's close while waiting |

Judgment calls of the trade:

- **Session refresh.** "Draw the lines after each trade, then wait" taken literally lets the
  lines go stale. On NOW from 2026-04-16 the long-only replay drew its lines after the April
  trade near $82, price rallied to $149 without coming back, and it waited five months. With the
  refresh on (the default) it traded 41 times.
- **Entry at the next open**, not at the touch: she buys "at the pivotal moment" by hand; a
  candle only shows that the line held when it closes.
- **The whole wallet in every trade.** Her example sizes by the dollars at risk (one contract,
  a $420 or $42 stop); here every trade is the same size, so a trade's R and its return differ.
- **Which line is traded.** The line closest to price, which is always the last line of the
  top-down chain, unless `min_span` skips it. Then price passes through it to an older line.
  - **The last line's chart:** it is a 15-minute line whenever the 15-minute chart drew one.
  - **On NOW** (#417 without its trend filter, checked 2026-10-27): 87% of longs and 86% of
    shorts traded the closest line. 34 of the 38 exceptions were lines shorter than #417's
    `min_span` (4.79 sessions between A and B). 28 of the 280 trades were on 1-hour, 4-hour,
    daily or weekly lines.
  - **With the two rules below and `min_span` 0:** 99% and 100% traded the closest 15-minute
    line.

Two knobs added on 2026-10-27 at your request, both off in `bounce.replay` (so earlier studies
replay unchanged) and on in the regimes notebook:

| Knob | Notebook | What it does |
|---|---|---|
| `own_lines` | on | trade only lines drawn on the trading chart; a side with no 15-minute line does not trade |
| `anchor_gap` | 16 | a 15-minute line needs its points A and B at least this many 15-minute bars apart (16 = 4 hours). A shorter one is not a line yet: it stays a pending ray, and the chain ends before it, since the next line would start at its point B. Lines of the higher charts are not affected |

## The 21-stock study (2026-10-19): no edge

`study.py`; the design was fixed before the first run (its docstring). NOW and META MSFT NVDA
GOOGL PLTR JPM GS COF LLY GEHC CAT GEV FLEX COST KMX EXPE FTI CCJ COIN RGTI, 2026-03-30 to
2026-09-29 (the six months all of them have 1-minute candles for; most end 2026-09-29/30),
15-minute trading chart, costs included. A setting's score is the mean over the stocks of the
compounded return of its trades; knobs are picked on the trades entered before 2026-06-29 and
judged on the later ones. `study.py report` prints every table.

**A · her rules (defaults).** Every mode lost, while buying and holding made +20% on average:

| Sides | Score (6 months) | Stocks > 0 | Trades per stock | Win rate | Per trade, net | Per trade, before costs |
|---|---|---|---|---|---|---|
| long | −9.4% | 2 of 21 | 40 | 18% | −21 bp | −15 bp |
| short | −4.6% | 4 of 21 | 37 | 19% | −10 bp | −4 bp |
| both | −8.5% | 6 of 21 | 75 | 19% | | |

The median trade loses 0.95 R: 57% of trades end at the stop (−0.79 R on average), and the rest
close beyond the line (+0.34 R long, +0.67 R short) or run to the end. The winners she shows
(one +9.5 R trade) exist, but they do not pay for the stop-outs. The median stop is 0.47% from
the entry, so a round trip's costs (about 6 bp) are 0.13–0.17 R on every trade.

**B · the knobs** (per trade, pooled over all settings of the other knobs):

- **Entry gap** (how close a low must come to the line): longs least bad at 0.5 typical candles
  (−11 bp net), shorts at 0.1 (−15 bp); no gap made money.
- **Stop room** (the break-through that stops a trade): 0.5 typical candles least bad on both
  sides. At 0.25, 85% of longs are stopped out; at 2, the win rate rises from 15% to 21% but each
  loss is bigger.
- **Close break**: makes no difference (±2 bp).
- **Tuning does not carry over for longs**: the rank correlation between the first and the
  second half's scores over the 48 settings is +0.09. For shorts it is +0.77 (narrow gap, tight
  stop least bad), but the best short setting scores −4% in the first half and +1% in the
  second.

**C · placebo.** Lines moved off hers did no worse. Longs: her lines −9.4%, moved 2 typical
candles away −0.8%, moved 2 toward price +3.3%. That is the rising market paying for being in
more often, not the lines. Shorts: her lines −4.6%, the least bad of the shifts (−5.7% to
−11.6%), but within about one standard error.

**D · two filters from her own rules**, fixed before running:

- **Trend** (long only above, short only below the 50-session daily average): longs −4.1%,
  better than the default in both halves. Still negative, and −19 bp per trade.
- **Higher-chart lines only** (points A and B at least 20 sessions apart): longs −2.4%, 6 of 21
  positive, −15 bp per trade (t = −0.1 across stocks).
- Shorts: neither filter helped consistently. Both sides with the trend filter: −10.3%.

Both filters cut the trades from about 40 to 12–21 per stock, but they do not make a trade
better.

**Verdict:** on 15-minute charts her bounce has no edge in this window, before or after costs,
long or short, at any setting tried. Caveats: one six-month window in a strong rising market, 21
stocks, regular hours only, every trade with the whole wallet. IDEAS.md has what could still be
tested.

## Tuning every knob on many stocks at once (`optimize.py`, 2026-10-20)

The design was fixed before the first trial (the module docstring has it in full):

- **Fitting:** all 13 knobs are tuned on 11 stocks together (TRAIN: every other one of the
  study's 21) over 2026-03-30 to 2026-09-29. The 13 knobs are:
  - trading chart (15-minute or 1-hour);
  - price scale;
  - the four line knobs;
  - session refresh;
  - sides;
  - entry gap, stop room and close break;
  - the trend filter;
  - the shortest line traded.
- **Held out:** the other 10 stocks over the same months, and all 21 over 2025-09-29 to
  2026-03-27.
- **No trade floor** (your rule, 2026-10-20): a setting that finds no signal makes no trade and
  scores 0, so every pick is reported with its trade count. The first run (with a 10-trade
  floor) was stopped. Its random search was carried over, rescored without the floor.
- **Objectives:** mean return, return over buy-and-hold, consistency across stocks (t), the
  Sharpe ratio of the 11-stock book, profit per trade, and return against drawdown.
- **Optimizers:**
  - random search, whose trials are also scored on the held-out stocks, so each knob's fANOVA
    importance can be checked out of sample;
  - TPE on each objective, warm-started with the random trials;
  - CMA-ES on return;
  - NSGA-II on return against drawdown.
- **Trial counts:** batches of 28 trials until two batches bring no gain (at least 3, at most
  16).
- **Validation:** the 5 best trials of every study are replayed on both held-out sets.

`Chartbook` (in `bounce.py`) builds every chart once and cuts it at each setup moment, making a
replay 5–7 times faster. It reproduces the study's trades exactly, and a test pins it to the
original snapshot.

**Result: no knob has an edge; the ones that matter decide how much and which way it trades.**
`optimize.py report` prints every table; the scores below are mean log(1 + return) per stock
over six months.

- **Trials.**
  - Random search: 196 trials, each on 21 stocks.
  - TPE: 84 trials of its own per objective. Every TPE study stopped after 3 batches: none beat
    the random search's best by its tolerance.
  - CMA-ES: 113 trials. NSGA-II: 112.
  - More trials would only fit noise. Over the random trials, the training and held-out-stock
    scores agree weakly for return (rank correlation +0.31) and not at all for profit per trade
    (+0.16) or Sharpe (+0.04).
- **What influences the outcome** (fANOVA on the random trials; importance on TRAIN / OTHER
  STOCKS):
  - For return: refresh 0.74 / 0.01, sides 0.07 / 0.50, shortest line 0.09 / 0.25, trend
    0.02 / 0.10.
  - Their effects point the same way on both sets: refresh off, long only, long lines and a
    small entry gap lose least. All of them make the strategy trade less, or trade long in a
    rising market.
  - The six line-drawing knobs (chart, scale, close enough, too small to see, confirm, screen)
    and the stop and close-break distances barely matter (importance ≤ 0.11).
- **What the optimizers picked.** Without a trade floor, return, excess, consistency and
  profit-per-trade all converged on one setting that barely trades:
  - 1-hour chart, long only, lines at least ~30 sessions long, close enough 1, refresh off,
    entry gap ~1.6, stop room ~0.8, close break ~1.2;
  - 0.7 trades per training stock.

  CMA-ES and NSGA-II found variants of it. Sharpe picked a log-scale variant with a wide stop.

| Setting | TRAIN | OTHER STOCKS | EARLIER (all 21) | Trades / stock (TRAIN, OTHER, EARLIER) |
|---|---|---|---|---|
| her defaults | −11.6% | −5.0% | −11.1% | 77, 73, 84 |
| the barely-trading pick (TPE, random) | +2.2% | +8.7% to +9.9% | −0.8% to −1.8% | 0.7, 1.2–1.7, 1.7 |
| CMA-ES best | +4.6% | +7.0% | −2.9% | 0.9, 1.7, 2.2 |
| best Sharpe with ≥ 10 trades / stock (first run) | +1.9% (Sharpe 0.91) | −8.8% | −2.5% | 20, 26, 24 |

- **No edge.**
  - The barely-trading pick made money on the other stocks while their buy-and-hold made about
    +24%. In the earlier months it lost 25–75 bp per trade.
  - Its few trades are too few to tell skill from luck: about 14 on the 10 held-out stocks, 37
    in the earlier months.
  - Placebo: with its lines moved one typical candle away from price, it did as well on the
    other stocks (+8.0%) and better in the earlier months (+0.4%). So the lines are not what
    makes it work.
  - The active settings that looked best on TRAIN fell apart out of sample.
- **Search strategies and objectives.** The random search found the region; TPE added nothing
  measurable. CMA-ES reached a higher TRAIN return through a different corner that did no
  better out of sample. Profit per trade is meaningless without a trade floor: its best,
  +650 bp a trade, is one or two trades.

**Three more objectives (same day): no edge.**

- **Lower confidence bound on profit per trade:** it picked active settings (50–70 trades per
  stock), but the bound stayed between −14 and −31 bp on every set.
- **Alpha** (return minus beta × the stock's own buy-and-hold): it barely traded again. Alpha
  was +3.5% on TRAIN and −0.3% to +0.7% held out.
- **Worst third of the stocks:** it picked "do not trade".

## Your win/loss objective and stock groups (2026-10-20)

**Step 1, the objective** (+1 for a winning trade, −1 for a losing one), checked on the
21-stock study's trades:

- **Her exits:** every setting wins only 12–25% of its trades. Wins minus losses therefore
  falls with every trade (rank correlation with activity −0.89), so this objective also prefers
  not trading.
- **A profit target** lifts the win rate above 50% (0.5 R: 51–55%). That is the exit's
  geometry, not her lines: moved lines win as often, and the trades still lose about 10 bp
  each.

**Step 2, 200 stocks in 10 groups** (`groups_test.py`; the design is fixed in its docstring):

- **Universe:** the most traded stocks with 1-minute candles since before 2025-10.
- **Groups:** 10, by the 11 price-character numbers of `classic_signals/groups.py`, measured
  2025-07 to 2026-03 only. The groups explain 56% of the differences in character, and 60% of
  stocks keep their group on each half.
- **Test:** her rules, long and short apart, on her lines and on lines moved ±0.5, ±1 and ±2
  typical candles, 2026-03-30 to 2026-09-28, split at 2026-06-29.

| | April–June (discovery) | July–September (validation) |
|---|---|---|
| long: her lines' win rate minus the moved lines' | +2.2 points, t 2.2 | +1.8 points, t 2.1 |
| short: the same | +2.1 points, t 2.9 | +1.5 points, t 2.0 |

- **The pooled test passes** on both sides, but **no single group passes**: the effect is about
  2 points almost everywhere. Group 2 (longs) and group 3 (shorts) each pass one half only.
- **Checks after the fact:**
  - **Longs:** her lines win no more often than lines moved half a typical candle. So the
    exact line does not matter.
  - **Shorts:** her lines beat lines moved 2 typical candles away in both halves (t ≈ 2),
    and earn a little more per trade than the moved lines (per-stock t 2.7 and 2.5). In
    July–September, though, the half-candle shifts did slightly better than her lines. So
    being near a downward line may matter a little; the exact line does not.
  - **No profit either way:** the median stock loses 14–24 bp per trade on her lines, and
    only 25–42% of stocks make money. Wins minus losses is −0.8 to −0.9 per stock per week;
    the edge over the placebo adds about one win per stock per year.
- **Still unread for these 200 stocks:** 2025-10 to 2026-03 (only the 21 study stocks were
  replayed there). It is the place to test the short-side hint with a new design fixed
  beforehand.

## Nine stocks, 15-minute candles, normalized knobs (`tune.py`, 2026-10-20)

**Stocks:** GPI, BIO, ICUI, AMG, ESE, R, KALU, CHE, CACC. ASR was removed at your request;
a first run that included it was stopped before its first batch.

**Windows:**
- **TRAIN:** 2026-01-01 to 2026-07-31.
- **VALID:** 2026-08-01 to 2026-09-28, the last day all nine stocks have 1-minute candles.

**Fixed:** the 15-minute chart, a plain price scale, 150 candles on screen, the session refresh,
and a close-break distance of 0.

**Merged:** the lines' close enough and too small to see are one knob, `tol`, with
`pierce = 0.4 × tol`.

**New knobs** (`bounce.py`):
- `target`: take profit at a multiple of R;
- `reward`: skip a signal with a line of the other kind less than so many R ahead;
- `skip_open`: no signals on the session's first candles.

**Comparable on every stock:** every distance is in typical candles (the stock's own median
15-minute range at the time), in R (the trade's own first risk), in sessions or in candles of
the one chart. So a value means the same on a calm and on a jumpy stock, which is what makes
tuning ten stocks at once meaningful.

**Metrics:** every trap seen so far has metrics of its own, and every trial keeps them all for
post-processing:

- "Do not trade" wins: t and lower bound of the net return per trade. No trade scores worst.
- Market drift passes for skill: excess over random entries held as long, on the same side,
  stock and months, and wins above chance as a z. This is your win/loss idea, made sound.
- A few trades decide: median, trimmed mean, the best trade's share, and the return without
  each stock's best trade.
- Win rate fights profit: payoff ratio, profit factor, expectancy in R.
- Costs eat tight stops: costs in R, and gross and net per trade.
- Stocks and months differ: stocks positive, worst third, alpha, beta, the book's Sharpe, and
  the share of positive months.

**Studies:**
- random search, every trial also replayed on VALID;
- TPE on excess over random holds, the lower bound, wins above chance and the book's Sharpe;
- NSGA-II on excess over random holds against activity, capped at 3 trades per stock per week.

**Judgment:** a setting has an edge only if, on VALID:
- its trades beat random holds (t ≥ 2);
- they make money after costs;
- and they beat lines moved one typical candle.

**Result (2026-10-20): no robust edge.**

Trials: random search 168, each TPE study 84 (all stopped after 3 batches), NSGA-II 148.
`tune.py report` prints every table.

| Setting | TRAIN trades, net bp/trade | VALID trades, net bp/trade |
|---|---|---|
| Her defaults | 725, +1.3 (vs random holds t −1.4) | 244, −15.7 (t −1.3) |
| TPE on excess over random holds, and on the book's Sharpe (short only, lines ≥ 36 sessions) | 6, about +55 | 0 trades |
| TPE on the lower bound (both sides, 1 R target, very tight stop) | ~600, −2 to −3 (+3 gross) | ~200, −2 to −7 |
| TPE on wins above chance, trial 248 (short only, 1 R target) | 156, +0.1 | 18, +34 (t 3.8) |
| NSGA-II #139 (both sides, 1.5 R target) | 328, −4.4 | 102, +1.6 |

**Trial 248 clears the bar, but it doesn't hold up.**
- Its 18 validation trades are clustered: 13 in the last 9 trading days, 7 on ESE, only 10
  distinct days.
- In training it lost 23–29 bp per trade from February to April and made 15–38 bp from May to
  July: a regime, not a steady edge.
- It was the best of about 30 validated settings.

**NSGA-II #139:** her lines beat the moved lines in both windows and both directions, but the
trades make about 0 after costs.

**Small samples win again.** A t or a lower bound on 5–6 trades topped both evidence
objectives, and those settings made no trade in VALID.

**Win-rate metrics follow the exit's geometry.** A close stop and a 1 R target win often, so
wins-above-chance and the lower bound carry over from TRAIN to VALID (rank correlation over the
random trials 0.62–0.63) without any profit. Profit metrics barely carry over: net return
0.13, excess over random holds 0.27, Sharpe 0.16.

**Knobs (fANOVA):**
- **The target is the one consistent effect.** Her exit with no target makes trades worse than
  random holds of the same length (median t −2.8 in TRAIN, −1.8 in VALID); any 1–3 R target
  brings the median setting to about 0. A target removes a handicap; it does not create an edge.
- **The shortest line traded** matters a little on both windows.
- **Line drawing (`tol`, `confirm`)** barely matters.
- **Everything else** (stop distance, trend, skip the open) is important in one window only.

### Second run (v2, 2026-10-20): wider space, a minimum sample, a one-shot 2025 check

**What changed:**
- **Minimum sample:** a setting with fewer than 50 TRAIN trades scores worst. It is a sample
  size for scoring, not a trade rule: the setting still trades only on its own signals.
- **Wider bounds:**
  - `target` adds 0.5 and 0.75 R;
  - `tol` goes up to 2, `confirm` to 10 and `skip_open` to 8;
  - `trend` goes up to 400 candles, `min_span` to 80 sessions and `room` to 5.
- **`entry`:** next open, or a limit order `band` typical candles beyond the line, filled in
  the candle that reaches it.
- **Budget:** at least 6 batches of 28 trials, at most 16; a study stops after 2 batches
  without gain.
- **Studies:** TPE on net bp per trade replaces wins above chance.
- **2025 check, read once:** 2025-01-02 to 2025-12-31 (AMG, BIO and KALU from 2025-06-19). The
  pass bar was fixed before the run: t ≥ 2 against random holds, net > 0, and better than lines
  moved one typical candle in both directions.

**Result: nothing passes 2025.**

Trials: random 224, each TPE study 168–224, NSGA-II 168. Every study stopped after 6–8 batches.
"t" below is the t against random holds.

| Setting | TRAIN trades, net bp/trade | VALID | 2025 |
|---|---|---|---|
| Her defaults | 725, +1.3 (t −1.4) | 244, −15.7 (t −1.3) | 1,109, −14.1 (t −3.7) |
| TPE net bp #387 (short only, 3 R target, 300-candle trend) | 57, +50 | 0 trades | 81, −56 (t −4.3) |
| TPE Sharpe #309 (both, limit, 1.5 R target, ≥ 4 R to the next line) | 87, +38 | 14, +31 (t 1.3) | 126, +7.2 (t 1.0) |
| TPE lower bound #417 (both, limit, 1 R target, 400-candle trend) | 406, +10.6 (t 2.3) | 115, +9.0 (t 2.1) | 605, −7.7 (t −0.9) |
| TPE excess #382 (both, 0.75 R target, stop 0.16 candles out) | 154, +10.6 (t 2.3) | 30, +14.7 (t 2.6) | 193, −6.7 (t −0.3) |
| NSGA-II #85 (short only, 0.75 R target) | 169, +8.9 | 40, +15.4 | 266, −17.6 |
| v1 trial 248 | 156, +0.1 | 18, +34 | 186, −17.4 |

**What it shows:**
- **The one setting that carried over was not her lines.** #417 held from TRAIN to VALID on a
  real sample (t 2.3, then 2.1 on 115 trades) and then lost in 2025. Even in TRAIN and VALID,
  the same lines moved one typical candle away from price did as well or better: +14.1 vs +10.6
  and +12.8 vs +9.0 bp per trade. What worked there was the exit, not where the line was.
- **A minimum sample moves the fluke, it doesn't remove it.** TPE on net bp found settings with
  54–66 TRAIN trades, just above the 50, and no trade at all in VALID. They were shorts behind a
  300-candle downtrend filter, which August–September never met.
- **Profit still doesn't carry over.** Rank correlation TRAIN → VALID over the random trials:
  - profit barely carries: net 0.22, t against random holds 0.23, Sharpe 0.21;
  - what carries is activity and exit shape: trades 0.84, win rate 0.69, payoff ratio 0.67.
- **The knobs that agree in both windows are exit settings, and the median stays below zero:**
  - **Target:** no target (her exit) is the worst, at −34 and −20 bp; 1–1.5 R is the least bad.
  - **Stop distance:** `room` 0.8–1.9 typical candles is the least bad and more than 1.9 the
    worst. It explains 75% of net bp in TRAIN and 43% in VALID.
  - **Bad in both windows:** `skip_open` 8 and `trend` 20.
  - **No effect:** limit entry vs next open (fANOVA ≤ 0.02).
  - **No agreement:** what drives the t against random holds in TRAIN (`band`, `min_span`) is
    not what drives it in VALID (`room`, `sides`).
- **Bounds are still hit.**
  - The #417 family sits at `trend` 400, the new top, with `reward` ≈ 4.5 of 5.
  - The #382 family sits at `room` 0.12–0.16, next to the 0.1 floor.
  - Since profit doesn't carry over, a wider space only gives the optimizer more room for the
    same kind of fit.

**Verdict:** the bounce on her lines has no edge on these nine stocks on the 15-minute chart.
That holds over three windows and two tuning runs, with 2025 read once. 2025 is now read for
these nine stocks.

## The trades one by one, and when it pays (`regimes.py`, 2026-10-23)

**The notebook** `tori_trades_trendline_regimes_marimo.py` replays one setting on one stock's
whole 1-minute history, each side alone, and has two parts.

**Three rules on top of every preset** (2026-10-27, all switchable, all on):
- **Strict anchors** (`strict`: no poke allowance, `on_line`; see "How the lines are drawn",
  rule 7).
- **15-minute lines only** (`own_lines`).
- **An anchor gap of 16 bars** (4 hours) for a 15-minute line (`anchor_gap`; see "The trade").

On NOW (#417 without its trend filter, 15-minute lines only, gap 16), strict anchors changed
the longs from 129 trades at −6.2 bp to 88 at +9.8 bp (+8.6% compounded), and the shorts from
120 at −18.4 bp to 90 at −36.2 bp. That is one stock and one window, with knobs tuned under the
old rules: not evidence either way.

**Part 1, the trades** (your request, 2026-10-27):
- **§1 The trade log.** Every long and every short trade, ranked by return after costs and
  split into profit fifths. Each row carries the line it bounced off (the chart it was drawn
  on, its points A and B) and the setup: when the lines were drawn, why, and how many upward and
  downward lines stood then (`regimes.trade_log`).
- **Picking trades.** The best 5 of each side are picked; the number, the order (best or worst
  first) and the picks can be changed in the tables.
- **Download.** Every line at every trade's setup is one table (`regimes.lines_log`).
- **§2 The picked trades, two charts each.**
  - **The big picture (top, 2026-10-27).** Daily candles from 120 sessions before the entry (a
    slider), with every line that stood at the setup. Each line is drawn in the style of the
    chart of the top-down walk it came from, heavier the higher the chart: monthly to daily
    solid, 4-hour dashed, 1-hour dash-dotted, 15-minute dotted. The traded line is bold, and the
    detailed chart's window is shaded. A line of text counts the lines per chart.
  - **The detail (below).** The 15-minute candles from the point A of the line the trade
    bounced off, or from a few sessions before the entry if that is earlier, to a few sessions
    after the exit (sliders, 3 and 2 by default). Only that line's anchors are marked. An
    anchor before the 15-minute history starts cannot be shown: the chart then starts at the
    line's next anchor, and the text under the trade says so. On NOW the window runs from
    about 150 to 5,000 candles. The chart draws every line that stood at the setup, upward and
    downward, the traded line's anchors (◆ A, ★ the candle that stopped the ray, ◇ B), the
    entry gap, the traded line, the moment the lines were drawn, the entry, the stop at every
    close and the exit, and tells the trade in words.
- **Legends** sit under every chart in two columns. The figure grows with the legend's rows, so
  the legend never covers the axis labels.

**Part 2, profitable time ranges** (§3–§6), below.

**The question** (yours, 2026-10-23): instead of a setting that pays on average, can a
measurable sign tell, better than luck, that a stock is moving into a stretch where the bounce
pays? Step one, in `tori_trades_trendline_regimes_marimo.py`: find every time range where one
fixed, somewhat promising setting made money, long and short, and show the ranges with the
setting. NOW comes first.

**The setting** is a preset, with every knob and its meaning shown in the notebook:
- **Default: #417 without its trend filter.** It was the one setting that held from Jan–Jul to
  Aug–Sep 2026 on a real sample, then lost in 2025.
- **Why the trend filter is off:** a trend filter is itself a regime metric, a candidate for
  the next step, not something to build into the labels. On NOW it leaves the longs no trade
  after November 2025. Without it the random trials did no worse.
- **Other presets:**
  - #417 as tuned;
  - v1's #139, where her lines beat moved lines;
  - her defaults.

**How a range is found:**
- **Each side is replayed alone** over the stock's whole 1-minute history (NOW: from
  2024-12-26). In one wallet, a short holding the money would hide a long stretch.
- **A side's record** is its return after costs, compounded trade by trade. A zigzag cuts it
  into legs: a rising leg ends once the record falls by the *reversal* (default 2%) from its
  high, a falling leg once it rises by the reversal from its low.
- **A profitable range** is a rising leg of at least *min trades* (default 3), from its first
  entry to its last exit.
- **Every leg is listed**, with the stock's own move over it. Losing legs are the contrast the
  next step needs.
- **The ranges are hindsight labels:** the target. A metric that is to predict them must be
  read before a range starts.

**Luck test.** Every record rises some of the time, so a range alone proves nothing.
- **The draws:** keep every trade's size and time and deal its wins and losses out again
  2,000 times (the signs shuffled).
- **What it reads:** persistence. That is the correlation of a trade with the next one and
  with the average of the five before it, and the number of win/loss streaks.
- **Why not the zigzag's own numbers:** they make a poor test. Ten wins then ten losses spend
  no more trades in profitable ranges than 40% of random deals do, because any record rises
  about half the time.

**First read, NOW** (#417 without its trend filter, 2024-12-27 to 2026-10-16, a 2% reversal,
3 trades). Over the window, buy-and-hold lost 35%.

| Side | Trades | Won | Net per trade | Compounded | Profitable ranges | Trades inside them |
|---|---|---|---|---|---|---|
| Long | 147 | 43% | −5.2 bp | −8% | 5 | 59 (40%) |
| Short | 130 | 41% | −19.6 bp | −23% | 6 | 41 (32%) |

- **The largest ranges:**
  - **L6:** Nov 2025 to Mar 2026, 22 trades, 17 won, +12.5%, while the stock fell 32%.
  - **L10:** Jul to Sep 2026, 14 trades, +7.0%, while the stock rose 44%.
  - **S12:** May to Jun 2026, 11 trades, +5.8%.
- **The worst leg:** S13, Jul to Aug 2026, −18% while the stock rallied 33%.
- **The long ranges came in a falling and in a rising market.** The stock's direction over a
  range does not sort them.
- **No clustering on either side.** The luck test's chance by luck is 0.10 to 0.46 for every
  statistic: a trade's result says nothing about the next one's. On NOW the strategy's own
  recent record cannot announce a range, so a predictor would have to come from the market.
- **Other presets on NOW:**
  - **v1 #139:** long 74 trades, −18.8 bp; short 72 trades, −1.6 bp.
  - **#417 as tuned (trend filter on):** the longs make no trade after November 2025, and
    the shorts none before April 2025.

## The lines at one moment, one line per chart (`tori_trades_trendline_topdown_marimo.py`, 2026-10-27)

You asked to step back and check how the lines are drawn and how a setup is made. Then you
specified the method: per side, **one line per chart**, top-down
(`topdown.cascade`):

1. **Daily.** One upward and one downward line, from the extreme of the daily candles in view
   (a slider, 150 by default): the lowest low for the upward line, the highest high for the
   downward one.
2. **4-hour, 1-hour, 15-minute.** Each line starts at the point B of the line above it, moved
   onto this chart's candle that holds the same price.
3. **Every line** is the steepest ray from its point A that no candle of its chart crosses.
   Its point B is the candle that stops it (the angle candle), so both anchors are candle
   extremes on the line.
4. **Not a line yet.** A line whose angle candle is among the newest *confirm* candles is a
   pending ray. So is a line shorter than its chart's minimum, counted in that chart's candles
   between A and B. The defaults are daily 5, 4-hour 4, 1-hour 4 and 15-minute 16 (the last
   two are 4 hours each). The charts below a pending ray draw nothing.
   - **Why a minimum on every line:** 29% of the 4-hour lines and 18% of the 15-minute lines on
     NOW (80 closes) spanned one candle.
   - **Why stop rather than skip:** skipping a short line and drawing on from its B found a line
     on few more charts (22 against 20 fifteen-minute lines in 160 cases). It also loses the
     exact start on the line above.
5. **The setup.** A side trades only when its 15-minute line exists.

**Why the full walk drew several lines per chart.** In her videos she draws as many lines as
she can on every chart until she reaches today's price. The full walk (`topdown.top_down`) does
the same.

**The trade replay.**
- **The switch:** `bounce.replay(method="cascade", minimum=...)` trades the cascade; the
  default stays `"walk"`, so earlier studies replay unchanged.
- **The trading notebook** uses the cascade by default, with the minimum sliders (its
  15-minute minimum is the anchor gap).
- **It trades rarely:** its single 15-minute line is often far from price. On NOW (Dec 2024 to
  Oct 2026):
  - with #417's knobs: 4 longs and 14 shorts;
  - with #417's own `min_span` (4.79 sessions) off: 13 and 30;
  - with `confirm` 2 as well: 21 longs at −2.3 bp a trade and 34 shorts at +30.8 bp.

  That is one stock, a falling one, so it is no evidence.

**The notebook.** It takes a stock, a date and a time of day (on the 15-minute grid, the close
by default). It cuts the charts exactly as the trade replay does (`Chartbook.at`; the higher
charts' last candle may still be forming).

- **Section 1** names the 15-minute line each side would trade, or where the chain stopped.
  It also gives the line's distance to the last close, and every line of each side in a table.
- **Section 2** draws the daily, 4-hour, 1-hour and 15-minute charts. Each shows its own line
  and the lines above it. Each lower chart starts just before the point A of its line, so both
  anchors are in view.

**Line styles** are the same in every chart:
- **Color:** blue upward (long), orange downward (short).
- **Daily:** solid.
- **4-hour:** long dashes.
- **1-hour:** short dashes.
- **15-minute:** dots.
- **Pending ray:** thin dash-dot.
- **The traded line:** tinted.

**First reads on NOW:**
- **2026-10-23 close:**
  - long: the 15-minute line from 29 Sep through 16 Oct;
  - no short: the highest high in view is the newest candle.
- **2026-03-31:** no long (the 15-minute line is pending); the short's 15-minute line runs from
  9 Mar to 19 Mar.
- **Very short lines are possible:** a 1-day daily line from the 11 Dec high to the 12 Dec
  high, 7 cents lower, and a 4-hour line one candle long.

**Section 3, walk forward** (2026-10-27). It trades these lines over the stock's whole
1-minute history: `bounce.replay(method="cascade")`, each side alone, with the line settings of
sections 1–2 (no #417 knobs: `min_span` 0).
- **The trade:** her rules by default: a candle comes to the 15-minute line and closes on its
  side, in at the next open; the stop is 0.5 typical candles beyond the line and follows each
  swing; out on the stop or a close beyond the line; no target. Entry, entry gap, stop room
  and a target in R are sliders.
- **What it shows:** each side's result, every trade in a table, and the top *N* (5) trades of
  each side on the 15-minute chart. Each chart has the lines that stood at the setup, the
  traded line tinted with its anchors, the entry, the stop at every close and the exit.
- **First read, NOW** (2024-12-27 to 2026-10-23; buy-and-hold −27%):
  - **longs:** 8 trades, 12% won, −26 bp a trade;
  - **shorts:** 16 trades, 44% won, +213 bp a trade (+37% compounded). They are a few trades
    riding NOW's fall along their line, such as 26 Jan to 26 Feb 2026.

  That is one falling stock, few trades and lines judged on that same stock: no evidence yet.

**When the walk forward redraws the lines** (2026-10-27): never after every candle.
- **While waiting:** when a candle breaks the 15-minute line without giving a signal (it
  closes beyond it), and at every session's close.
- **In a trade:** never; the trade keeps its line, and the lines are redrawn when it ends.

**Watch** (`bounce.replay(watch=True)`, on in the walk forward). After a break, the new
15-minute line is usually still pending, because the breaking candle is its own angle candle.
Before, the side then waited for the session's close; on NOW it did after all 16 breaks.
- **What watch does:** it keeps that pending line. The line trades once *confirm* candles have
  followed its angle candle. A candle reaching through it before then counts as a break.
- **On NOW** (her trade rules): longs went from 10 trades to 32 (+29 bp a trade), shorts from 16
  to 36 (+90 bp). About two thirds of them are on a watched line (one stock, no evidence).

**Bounce confirmation** (`bounce.replay(hold=N)`, next-open entries, 0 by default; a slider in
the walk forward):
- **The rule:** after the bounce candle, the next N candles must close on the line's side, and
  the entry is at the open after the last of them. A close beyond the line among them means the
  bounce failed: no trade, and the lines are drawn again.
- **Your example, 10 Aug 2026 (NOW):** the 10:15 candle poked 0.11 below the line and closed
  above it. Every one of the next 16 candles also closed above this steep line, until 14:15.
  So with N = 1 or 2 it still trades, only later: in at 10:45 (125.79) or 11:00 (125.78)
  instead of 10:30 (124.93).
- **On NOW** (watch on): with N = 1, longs went from 32 trades at +29 bp to 15 at +73 bp, and
  shorts from 36 at +90 bp to 25 at +95 bp. With N = 2: 13 longs at +80 bp, 23 shorts at
  +85 bp (one stock).

**Session-edge knobs** (2026-10-27, both off by default; counted in 15-minute candles from a
session's open and close, a half-day closing at 13:00):
- **Drawing, (x1, y1):** the 15-minute line leaves out the first x1 and last y1 candles of
  every session (`topdown.cascade`'s `trim`, `bounce.replay`'s `draw_trim`). They can neither
  stop it nor be its point B, so a spike there may poke through it. Its point A still comes
  from the 1-hour line's B, even inside a left-out candle. Starting on the nearest candle left
  in instead moved NOW's 2026-10-05 downward line from 139.57 to about 136.85, nearly flat and
  off the line above. The 15-minute chart shows the left-out candles faded.
- **Trading, (x2, y2):** no position is opened or closed on the first x2 or last y2 candles of
  a session (`bounce.replay`'s `trade_trim`): no signal, no fill, no stop or target fill, and
  no exit on a close beyond the line. A stop crossed then fills on the first candle after them
  (at its open when price is already past it). The stop still trails on them. The trade charts
  show those candles faded.
- **On NOW, with both at 2 and 2:** longs went from 8 trades at −26 bp to 8 at −38 bp; shorts
  from 16 at +213 bp to 10 at +20 bp (one stock).
- **Highlights** (purple, your request) mark the candles a rule kept from mattering:
  - **the 15-minute chart:** a left-out candle whose low (high) reaches or crosses a
    15-minute line right of its point A. It would have stopped that line.
  - **the trade charts:** a no-trade candle where a signal would have opened the trade, or
    where the stop or a close beyond the line would have ended it.

**Noise bands** (2026-10-27, your proposal request; off by default, a switch in the notebook).
The lines were razor sharp: a candle 1 cent through a line stopped it, and a low 1 cent short of
the entry gap was no signal. With bands on, every line and the price get a width sized by the
stock's own noise:
- **The noise** of a chart is the moving average of its candles' true range over the last *L*
  candles (`topdown.noise`; the noise window, 150 by default). It is recomputed at every candle
  of every chart, so the bands widen in a wild market and narrow in a quiet one.
- **Line band** (*k_line*, 0.25 by default). A line is a band of ± *k_line* × its chart's noise
  (`topdown.cascade(band=...)`, `bounce.replay(line_band=...)`). A candle may reach into the
  band without stopping the line. Point B is the latest candle inside the band, not the one
  that would have stopped an exact line.
- **Price band** (*k_price*, 0.25 by default). The price's own uncertainty. With the line band
  it makes the **contact zone** around the traded 15-minute line: ± (*k_line* + *k_price*) ×
  the 15-minute noise as known at each candle's open (`bounce.replay(price_band=...)`).
- **Signal:** a candle reaches into the zone and closes on the line's side. The zone replaces
  the entry gap; a limit order rests at the zone's edge.
- **Break:** a close beyond the zone. A close inside the zone but beyond the line is noise: no
  signal and no break. A trade exits on a close beyond the zone, too.
- **Stop:** the stop room beyond the zone, and the trailing stop keeps the zone's distance from
  the line.
- **The notebook** draws every confirmed line with its band (a translucent strip whose width
  follows the noise) and the contact zone around the traded line (dotted edges), in section 2
  and in the trade charts.
- **First read, NOW** (defaults: watch on, no bounce confirmation, no trims, bands 0.25 and
  0.25 on 150 candles): longs went from 32 trades at +29 bp to 30 at +8 bp (+1% compounded);
  shorts from 36 at +90 bp to 49 at +51 bp (+25%). One stock, so no evidence.

**Settings first** (2026-10-27, your request). Every knob sits at the top of the notebook, one
per row with what it does beside it: the stock, the moment, the lines, the noise bands and the
trade. Only the stock comes before the candles load, because the date picker needs to know the
stored days.

The first visual notebook (`tori_trades_trendline_marimo.py`) is left as it was: her full walk
at a date's close, in tabs.

## Files

| File | What |
|---|---|
| `topdown.py` | the lines: candles on the session clock, the walk, touches, crossings, the loader |
| `bounce.py` | the trade: one stock, long / short / both, candle by candle; `Chartbook` |
| `study.py` | the 21-stock study: defaults, the knob grid, the placebo; `report` prints the tables |
| `optimize.py` | every knob tuned on 11 stocks at once: 11 studies, held-out validation, fANOVA |
| `groups_test.py` | 200 stocks in 10 price-character groups: her lines against moved lines |
| `tune.py` | nine stocks, 15-minute, normalized knobs, trap-derived metrics, validation on Aug–Sep 2026, a one-shot 2025 check |
| `regimes.py` | one setting's profitable time ranges on one stock, long and short: presets, the zigzag, the luck test |
| `tori_trades_trendline_marimo.py` | the visual notebook (the lines) |
| `tori_trades_trendline_trading_marimo.py` | the trading notebook |
| `tori_trades_trendline_topdown_marimo.py` | one line per chart at a chosen moment (`topdown.cascade`): daily, 4-hour, 1-hour and 15-minute, both sides, and the setup a long and a short would trade |
| `tori_trades_trendline_regimes_marimo.py` | every trade logged with its lines, ranked by profit; the picked trades in context; profitable ranges |
| `IDEAS.md` | parked ideas |
| `tests/test_tori_trades_trendline.py` | the lines on hand-made candles (no database) |
| `tests/test_tori_trades_trendline_bounce.py` | the trade on hand-made candles: look-ahead, rules, the mirror |
| `tests/test_tori_trades_trendline_regimes.py` | the zigzag, the legs, the luck test, the cached replay |
