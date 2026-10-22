# classic_signals

A strategy built step by step from classic chart tools, for **ServiceNow (NOW)**, **GE Vernova
(GEV)** and **Rigetti Computing (RGTI**, added in step 4), long and short, on **1-minute
candles**. It should hold up in rising, sideways and falling
markets, and it only trades on a clear signal: missing an opportunity is fine, a muddled trade is
not.

## How a trade happens

Every part is a **switch** that can be turned on or off in the notebook. Built (steps 2 to 5):
the two band triggers, four trend filters, two level filters, the wait after the open, the
earnings guard, the trailing stop and an optional profit target. The filters added in steps 3
and 4 start switched off (a rule's `ON = False`); the earnings guard (step 5) starts on.

1. **A trigger proposes a trade** at a candle's close (two switches):
   - *band fade*: the close was beyond a Bollinger Band, and this candle closes back inside
     (back above the lower band: long; back below the upper band: short);
   - *band breakout*: the close stays beyond a band for a set number of candles in a row (the
     lever `breakout_closes`, 2 by default; 1 = the first close outside), in the move's direction
     (above the upper band: long; below the lower band: short). One close outside alone may
     still turn straight back, which is the fade's case. It fires once per stretch.
   If two triggers propose opposite sides on the same candle, nothing happens.
2. **Every switched-on filter must allow that side**. The four trend filters share one rule:
   up allows longs only, down shorts only, sideways (or not known yet) both:
   - *moving average* (built, step 3): the minute chart's moving average, judged at each
     candle's close: up when the close is above the average AND the average rose over the last
     `ma_rise_candles` candles (10) by more than `ma_flat_moves` typical moves (0.5); down when
     below a falling average; sideways otherwise. The average's length is the notebook's
     "Moving average" setting (60 candles);
   - *stock trend* (built, step 3): the stock's own daily trend (close against its 50-day
     average, and that average's rise), judged at the previous close. It is also the market
     condition every result is split by;
   - *sector trend* (built): the same daily rule on the stock's sector fund (IGV, software, for
     NOW; XLI, industrials, for GEV);
   - *market trend* (built, step 3): the same daily rule on the S&P 500 fund (SPY);
   - *wait after the open* (built): no new trade in the first minutes after 09:30 (the lever
     `open_wait_minutes`, 15 by default): a signal whose trade would fill in that time is dropped;
     trades already open keep their stop;
   - *major levels* (built, step 4): no long while the nearest major level above the close (the
     levels known at the session's open) is closer than `level_room_moves` typical moves (2), no
     short while the nearest one below is; a level right at the close blocks both;
   - *previous high / low* (built, step 4): the same rule on the previous session's high and low
     (`prev_room_moves`, 2);
   - *earnings guard* (built, step 5, on by default): no new trade in the last
     `guard_sessions` sessions (1) before a quarterly report's reaction day (the first session
     that can move on it: the same day for a report before the open, the next for one after the
     close), and an open trade is closed at the close before it (a rule's `flat`: the simulator
     closes any open trade at that candle's close);
   - *options calm* (built 2026-10-18, off by default): no new trade while the options market
     expects a much bigger move than the stock has been making. The expected move is the
     at-the-money straddle (call + put at the strike nearest the price, mid quotes) of the
     nearest expiry at least 2 days out, as a daily move: straddle ÷ (0.798 × price ×
     √trading days); it is divided by the standard deviation of the stock's last 20 daily
     returns and read from the previous session's end-of-day chain (`options_max_ratio`, 1.8).
     About 1.1 on an ordinary day, 2 to 5 before a report. Both sides are blocked: options say
     how much a stock may move, not which way. No option data (before 2025-10-14): both allowed;
   - *iv term*, *put/call*, *dealer gamma* and *option walls* (built 2026-10-18, off by
     default): from implied volatilities and Greeks the strategy solves itself (Black-Scholes on
     the mid quotes, `indicators.option_features`; the chains carry none). *iv term*: no new
     trade while the nearest options' volatility exceeds `iv_term_max` (1.3) × the 30-day
     options'. *put/call*: when the put/call volume is `pc_limit` (1.5) × its 20-day median or
     more, shorts only (and longs only when calls are that unusual), or the other way round
     (`pc_side`). *dealer gamma*: new trades only while dealers' gamma (+ call − put gamma ×
     open interest, over its usual size) is at least `gamma_min` (0). *option walls*: the
     major-levels rule on the strikes with the most call (above) and put (below) open interest
     over the next 45 days (`wall_room_moves`, 1); they can also be the profit target.
3. **Entry**: at the NEXT candle's open, in the same session (09:30-16:00 only). A signal on a
   session's last candle is dropped: the next open comes after a night, when it is stale. One
   trade per stock at a time; signals while a trade is open are ignored.
4. **Exit**: one of three stops (`stop_kind`). A trailing stop a multiple of the stock's typical
   move (a 15-minute average true range by default) away from the entry: after every candle it
   moves to the same distance behind the best price since the entry, only ever in the trade's
   favour. Or the same trailing a number of typical DAILY moves (the average daily range of the
   last 14 sessions) behind the best price. Or a stop fixed at a number of typical daily moves
   from the entry. A candle that trades through it fills at the stop; a candle that OPENS beyond
   it (after a night or a jump) fills at that open. Overnight holding is allowed; an optional
   holding limit closes a trade at the end of its n-th session (1 = never overnight). An
   optional **profit target**: the next major level (or the next major level or previous high or
   low) in the trade's direction, at least one typical move from the entry, or a number of
   typical daily moves from the entry; fixed at the entry. A candle that reaches it closes the trade there, one that
   opens beyond it fills at that better open, and when the stop and the target fall in the same
   candle the stop counts.
5. **Size**: every trade risks the same dollars at its first stop (`simulate.costs`: $100, at
   most $20,000 worth; a gap past the stop loses more). A calm and a jumpy stock, or a calm and
   a wild quarter, then count alike, and several stocks' profits add up on one scale. Every
   distance is measured in the stock's own moves (standard deviations, typical moves, typical
   daily moves), so one setting means the same on every stock.
5. **Costs**, the repo's cost model (`config/settings.yaml`): $10,000 per trade; commission
   $0.005 a share, at least $1 an order (a stock above $50 pays the minimum; Rigetti at $10 is
   1,000 shares, $5 an order); every fill loses 0.02% of the price (half the spread plus
   slippage), but at least half a cent a share, half the smallest possible spread (it binds below
   $25 a share). About 0.06% per round trip for NOW and GEV, several times that for Rigetti.

Daily information (trends, levels) always comes from the previous close, and a trade fills at the
next candle's open: no decision ever uses a candle that has not finished. The tests check this
end to end: cutting the data at any date leaves every earlier signal and closed trade unchanged.

## Results

The **market condition** of a trade is the stock's own daily trend at the entry (up, sideways,
down; judged at the previous close). The notebook splits every result by condition and by side,
draws the profit curve and its drawdown, and gives the bottom line per stock: total profit ÷
worst drawdown of the closed-trade profit curve, next to the same numbers for simply holding the
stock. Every trade can be clicked open: drawn with its stop, and told in words (which trigger,
what each filter said, why it ended).

**First read of step 2** (January 2025 → June 2026, both triggers and the sector filter on, the
breakout on its first close outside): about 800 trades per stock, in a trade 95% of the time
(signals are plentiful on 1-minute candles and only the stop ends a trade), 43% held overnight,
and a fifth of all trades ended with the price opening beyond the stop (the worst: NOW −14.6%
through its April 2026 report). The profit after costs is about zero (+$1,600 on both stocks:
GEV +$5,300, NOW −$3,700; −$3,400 with the default of two closes in a row). For reference,
holding GEV made +$25,300, holding NOW −$5,400. Where most trades fall, trading with the stock's
own trend did better than against it (GEV longs in its up trend, NOW shorts in its down trend):
step 3's own-trend filter will test that.

**The breakout lever** (2026-10-17): each extra close in a row about halves the breakouts (23.5,
10.9, 5.4 and 2.7 per stock and session for 1 to 4). The breakout alone made +$5,100, +$11,300,
+$5,600 and +$14,100 after costs: no steady direction, which is what noise looks like, so the
number is for step 5's walk-forward to choose. With the fade on as well, the fade fills the time
the breakout leaves free (still about 1,600 trades, in a trade 94-95% of the time).

**The wait after the open** (2026-10-17, every switch on): without it, 601 of 1,621 trades (37%)
opened before 10:00, 283 of them in the first five minutes. Those were not the worst trades: by
entry time, 09:30-09:34 averaged −0.02% per trade, 09:35-09:44 +0.20% (the best stretch of the
day), 09:45-09:59 −0.10%, later −0.05%. Waiting 0, 15, 30 or 60 minutes gave −$3,400, −$6,900,
+$700 and −$7,900: no steady direction again, so this lever is for step 5 to choose as well.

**What each switch changes** (step 3, 2026-10-17): the notebook runs the strategy once with
each rule off and once with it on, everything else as set, and splits the difference by market
condition and by half-year. First read (defaults: both triggers, the sector filter and the
15-minute wait on; breakout 2 closes; January 2025 → June 2026, both stocks):

| rule | total P&L off → on | cells where switching it on helped (of 12: per stock, 3 conditions and 3 half-years) |
|---|---|---|
| band fade | +$1,500 → −$6,900 | 3 |
| band breakout | +$3,400 → −$6,900 | 2 |
| moving average | −$6,900 → −$2,300 | 11 |
| stock trend | −$6,900 → −$6,600 | 7 |
| sector trend | −$1,300 → −$6,900 | 3 |
| market trend | −$6,900 → −$1,000 | 8 (one cell 0) |
| wait after the open | −$3,400 → −$6,900 | 5 |

Each trigger alone did better than both together (one trade at a time, so they take each
other's places). Only the moving average filter helped almost everywhere: in every market
condition of both stocks and in five of the six half-years. All of it is one in-sample look at
noisy numbers; step 5's walk-forward is the judge. With every switch on: 1,146 trades, in a
trade 80% of the time, +$1,600.

**The levels** (step 4, 2026-10-17, NOW and GEV, defaults): about three major levels a
session; the previous high or low is usually close to the price (the filter blocks a side on
about 28% of candles), a major level mostly is not (5-7%). Switched on alone: major levels
−$6,900 → +$400 (helped in 10 of the 12 cells), previous high / low −$6,900 → −$8,800 (7 of
12). Profit target at the next major level: −$4,300 (85 trades reached it); at the next major
level or previous high or low: −$9,700 (340 reached it). The same caveat: one in-sample look.

## Tuning (step 5): `classic_signals_tuning_marimo.py`

A notebook of its own, on `tuning.py`. It looks for **one setting for one stock or for several
stocks together** (picked at the top of section 1; NOW and GEV by default), judged on their
combined profit, every stock's own score and validation beside it. Since 2026-10-18 there are
three windows: the optimizer chooses on the
**training** window (2025-01-01 → 2026-06-30); every trial is also scored on the **validation**
window (2026-07-01 → 2026-10-16), to watch, never to choose; everything from 2026-10-17 on is
the **final check**, never loaded here. Its sections:

1. **The data**: the picked stock, training and validation windows, in a `tuning.Lab` (the
   strategy for any settings; its parts built once by `pipeline.py` and kept, so trials mostly
   reuse them).
2. **The search space**: every hyperparameter of `hyperparams.py` (one table both notebooks
   read: the candles and indicators, the rules' on/off switches, the rules' levers, the exit).
   **By default the optimizer decides everything**: every switch on or off, every number in its
   range, every choice. You override only what you want fixed: a switch *always on* / *always
   off*; a number's *from* / *to* / *step* (from = to fixes it); a choice's allowed values (one
   left fixes it). A lever is only drawn while it acts (`hyperparams.active`): a rule's lever
   while its rule is on, `stop_moves` with the typical-move stop, `stop_days` with a daily
   stop, `target_days` with the daily-move target. Table 2 says, row by row, what the optimizer may
   do; *reset* frees everything again. The default ranges live in `hyperparams.py` (a lever's
   in its rule's file). *First trial* (built-in values, or a stock's from an exported file) is
   where a run starts, moved into the space.
3. **The objective**: the **daily Sharpe ratio** on the training window: the profit of every
   session after costs (several stocks: added up, every trade sized by its risk), an open trade
   valued at the session's last close, mean ÷ standard deviation × √252; if any stock trades
   too few times a session, the setting has no score. Beside it, the **luck
   line**: the score the best of as many settings without any edge reaches by chance,
   Φ⁻¹(0.5^(1/n)) × √(252 / sessions); more trials raise it. Chosen on 2026-10-18 after a test
   on 500 random settings of NOW (each objective scored on 2025, read against the profit of
   2026 H1, and back): profit ÷ worst drawdown, total profit, the daily Sharpe and profitable
   months all predicted the other half about equally weakly (rank correlations 0.06–0.19), and
   no setting cleared the luck line in either half. The Sharpe does not grow with a window's
   length, so the training and validation scores compare directly, and it has that known scale
   of luck. (The two passes with a blended score are gone: the blend failed its own test,
   `IDEAS.md`.)
4. **A quick check**: one run with the optimizer's first trial, training and validation.
5. **The optimization**: Optuna with the trials asked; every trial's training and validation
   score, profit, trades and worst drawdown (Figure 1: trial after trial with the best so far,
   the luck line and the validation score of the best so far; training against validation for
   every trial), first trial → best (Table 4), the scores against the luck line (Table 5),
   stock by stock (Table 5b) and its longs and shorts by quarter (Table 5c).
   Every run is written, trial by trial, under its name to the journal `studies/tuning.log`
   (git-ignored); **a run is continued** by picking it under *run* (more trials, with its own
   search space, first trial and minimum trades), and refused when its data, windows or code
   changed. `uvx optuna-dashboard strategies/classic_signals/studies/tuning.log` (from the repo
   folder) shows every run live at <http://127.0.0.1:8080>. A trial with too few trades is
   *pruned*. **Several workers at once** (from a terminal, each its own seed; Optuna's
   "constant liar" keeps them apart): `uv run python strategies/classic_signals/tuning.py
   continue "<run name>" <trials> <worker>`, one command per worker (0, 1, ...). A run is only
   continued with the code it was made with: any edit of the strategy's code files (not the
   notebooks) starts a new run.
6. **Which hyperparameters move the score**: each one across its range on the training window,
   the rest held at the best: its effect (highest minus lowest score) and why one has none (not
   in use, or no trade changed). Table 6 and Figure 2.
7. **The walk forward** inside the training window: the optimization every `refit` sessions,
   each block traded with settings chosen on earlier sessions only (a trade keeps the exit it
   opened with), against no tuning, run in separate processes and kept on disk
   (`var/classic_signals/tuning/`). A walk trades one candle size: leave one in the search
   space.
8. **The export**: the best settings for the stock, one complete set, with what made them, to
   `var/classic_signals/settings/<date>-<name>.json`. The main notebook lists the files under
   **Starting values** (one entry per stock and file): every control there starts at those
   values.

Both notebooks compute the strategy with `pipeline.py`, so the main notebook shows exactly the
trades the tuner scored for the same settings. The heavy steps run on a button and start small.

**An earlier read, before the tuning notebook** (2026-10-17, the first walk forward: one pass
on a moving window, only the stop, the bands and the switched-on rules' levers tuned; the
settings as shipped; re-tuned every 5 sessions, 30 trials; traded late June 2025 → June 2026,
about 253 sessions):

| window | NOW | GEV | RGTI | all three | worst drawdown, all three |
|---|---|---|---|---|---|
| no tuning | −$7,600 | +$3,200 | +$800 | −$3,700 | −$19,500 |
| 20 sessions | −$2,900 | −$3,100 | +$10,400 | +$4,300 | −$12,100 |
| 40 sessions | −$3,000 | +$1,600 | −$700 | −$2,100 | −$14,700 |
| 60 sessions | −$3,800 | −$1,700 | +$900 | −$4,700 | −$15,800 |
| 120 sessions | −$2,500 | +$6,000 | −$2,200 | +$1,300 | −$13,400 |
| all sessions before | −$900 | +$2,700 | +$6,600 | +$8,500 | −$4,100 |

No window beat no tuning on all three stocks; "all sessions before" did on two (NOW, RGTI) and
pooled. But a good score on the tuning window did not predict the next sessions: the rank
correlation between the two is negative for 13 of the 15 stock-window pairs (−0.37 to +0.08),
and the short windows' levers jump between far-apart values from one tuning to the next
(its Figure 13). That is what fitting noise looks like. Part of the gain is trading less: the tuned
windows are in a trade 27-79% of the time against about 90% without tuning. The earnings guard
held no trade through any of the 12 reports of these sessions (NOW's worst trade −4.6% instead
of −14.6%).

**The full optimization of NOW** (2026-10-18, the tuning notebook's procedure at full size,
everything free). In-sample, the two passes on all of EXPLORE (500 + 300 trials, three seeds,
about 4 minutes each) found +$12,800, +$8,000 and +$17,100 (worst drawdowns $1,000–1,800),
against −$6,000 with the built-in values and −$5,400 for holding NOW. None of it is real:

- the three seeds chose three different strategies (candles 15 / 5 / 15 minutes, holding
  limit 2 / 2 / 5 sessions, different filters), so the search does not settle;
- seed 0's winner is a lone spike: one step of band width (1.5 → 2.0) turns +$12,800 into
  −$2,000; band length and band width sit on the lowest value of their ranges; 11 of the 35
  settings change no trade at all (the trend filters are switched off in effect by a flat zone
  of 1.5 typical moves);
- frozen on GEV and RGTI (same months), every NOW winner loses (−$2,500 to −$16,400).

The walk forward (traded 2025-06-27 → 2026-06-30, re-tuned every 20 sessions, 80 + 80 trials,
one family per candle size):

| candles | no tuning | w 0 | w 0.5 | w 1 |
|---|---|---|---|---|
| 1 minute | −$7,600 | +$2,700 | +$1,300 | −$5,100 |
| 5 minutes | +$1,000 | +$900 | −$2,200 | −$700 |
| 15 minutes | −$1,100 | −$2,500 | +$2,200 | −$2,200 |

NOW held over the same sessions: −$5,100. A tuning's score did not predict the next block (rank
correlation 0.06 over 117 re-tunings, p = 0.5), the chosen values jumped (band length changed at
33–100% of the re-tunings, the stop at 42–92%), no tuned walk beat no tuning block by block
(sign tests p ≥ 0.27), and every walk sits within luck (one stock-year: ±$2,500–4,500 for one
standard deviation). The tuning fits noise. Proposed next steps: `IDEAS.md`.

**Pass 1 alone, 1-minute candles, 5000 trials** (2026-10-18, 32 minutes, journal
`studies/tuning.log`): best score 7.7 (+$11,800 on 495 trades, worst drawdown $1,500; the
built-in values −$6,000). The best so far was still climbing at the end (3.8 after 100 trials,
4.8 after 1,000, 6.7 after 3,000, 7.7 after 5,000): more trials found luckier corners, not a
plateau. It is again a spike (stop 3 → 4.5 typical moves: score 7.7 → 3.4; band width 1.5 →
2.0: 7.7 → 2.4; previous high / low off: +$11,800 → −$300), with all its profit in trades held
overnight, and frozen on GEV and RGTI it loses (−$6,000, −$11,400). Not tested out of sample.

**The first run on the new objective** (2026-10-18, 10,000 trials, 1-minute candles, 81 minutes;
journal run "NOW 2026-10-18 11:42 1-minute, 10000 trials"): best training score 3.03 (+$16,400 on
393 trades, worst drawdown $2,800), found at trial 1,978 and never beaten in the 8,000 after it;
the luck line for 9,718 scored trials is 3.13 (3.62 one time in twenty), so the best does not
clear what chance alone reaches. On the validation window the best scored 1.42 (+$2,000, 82
trades), but a 76-session Sharpe swings ±1.8 by chance alone; across all trials training and
validation scores correlate at +0.08. The top trials are one family (band fade only, typical-move
stop, level target, held up to 2 sessions): the percent stop and target were not chosen.

**The first two-stock run** (2026-10-18, NOW + GEV, one setting, risk-sized trades, scored on
their combined daily profit; stopped at 2,035 trials after 25 minutes): best training score 3.10
against a luck line of 2.79 (3.33 one time in twenty), so above typical luck but not clearly.
On the validation window it lost: −1.16 together (NOW +0.18, GEV −2.02). Training predicts
validation broadly (rank correlation +0.48 over 1,974 trials: the bad regions are bad on both),
but not at the top (+0.12 within the best 10%, −0.05 within the best 1%, whose validation
averages about zero): the optimizer finds the region, the fine-tuning inside it is noise. The
region: all of the top 50 use the fixed stop in daily moves (the best: 0.3), the daily-move
target (0.9), hold up to 2 sessions, the band fade only, the slow moving-average and the sector
trend filters, no earnings guard.

**Risk-based sizing, measured** (2026-10-18, the run's best and 200 random settings, against
$10,000 a trade): positions of about $5,000–15,000 (median $8,400 NOW, $7,700 GEV; never the
$20,000 cap; at most $30,000 held across both at once). It evens out NOW's quarters (the
wildest quarter's daily swing 3.7× the calmest's with $10,000 a trade, 1.9× risk-sized) but
hardly GEV's (3.2× → 3.0×). It did not measurably make training predict validation better: of
200 random settings, the 48 that traded enough on both stocks correlated +0.26 risk-sized
against +0.20 at $10,000 a trade (noise about ±0.14).

## Stock groups: `classic_signals_groups_marimo.py`

A notebook of its own, on `groups.py`. It sorts every stock with 1-minute candles into
**groups that behave alike**, so the tuning can look for one setting per group rather than per
stock: a setting judged on several alike stocks rests on more evidence, and the group's other
members show whether it carries over to stocks it never saw.

- **Alike in character, not in industry.** Eleven numbers per stock, about what the rules react
  to, measured on the regular session of **every stored candle** (Dec 2024 to the newest
  download; since 2026-10-20 this includes the tuning notebook's validation months and final
  check, which the first version of the notebook kept out): the typical 15-minute move; a round
  trip's costs in typical moves (the strategy's own size and cost model); whether a minute's
  move continues (lag-1 autocorrelation); 15-minute moves against single minutes (variance
  ratio); after a close outside the Bollinger Band (the built-in 20 / 2), the share whose next
  close is back inside (the fade's candle; the rest are the breakout's second close) and how far
  the price goes in the next 15 minutes, in typical moves; the share of the daily swing made
  overnight (sizes, not squares, so one earnings gap does not decide it); the 15-minute
  correlation with SPY; the share of the session's candle ranges in its first 30 minutes; the
  share of the minutes' swing from each session's biggest 1%; the share of minutes without a
  trade. Every number is a ratio of per-session sums, so the whole window, each half or a moving
  window come exactly from one table per stock; the sums are kept on disk and a stock is
  measured again only when its candles changed (about 15 minutes for the whole pool).
- **Scores, then a tree.** Each number becomes the normal score of its rank among the stocks
  (0 the middle stock; an outlier cannot stretch the scale; every number weighs the same); a
  Ward tree on the scores is cut into the chosen number of groups. A group is said in words by
  the numbers on which its centre sits furthest from the middle.
- **Checks.** Every number's persistence (each stock's sessions cut in two halves: the rank
  correlation of the number across the stocks); which numbers measure the same thing; for every
  number of groups the share of the differences explained, the separation (silhouette) and
  whether each half alone forms the same groups (adjusted Rand index).
- **Tune on a few, check on the rest.** Every group ranks its members by their distance to its
  centre; the most typical (5 by default) are the ones to tune on, the others test the tuned
  setting on stocks it never saw. Figure 10 draws a chosen group's most typical stocks as
  daily candles, one panel each on the same dates (the last 3-18 months of the data,
  12 by default), every stock rebased to 100 at its first close on a log axis, so the same
  percent move is the same height everywhere (one price range for all panels, or each zoomed to
  its own stock).
- **The export** (`var/classic_signals/groups/exports/<date>-<name>.json`): every group's words,
  profile, tune and check members, and what made them. The tuning notebook does not read it yet
  (the next step: each stock then also needs a sector fund).

Two cautions. The groups are formed on all the data, so a walk forward, a validation or the
final check of the tuning step knows each stock's group from months it has not reached yet (mild,
the numbers persist and describe how a stock moves, not what it earned; but not nothing: a
setting tuned on a group's stocks and judged on later months is no longer judged out of sample
for the grouping step). And the pool is whatever was downloaded, partly picked by hindsight (the
HighOctane names were chosen for being the most active): fine for grouping, not for judging how
well a group's setting does.

**First read** (2026-10-18, all 1,759 stocks measured in 25 minutes; 1,755 with at least 40
sessions; every number in): every number persists (rank correlation between the halves 0.46 for
the overnight share up to 0.96 for the minutes without a trade; the market correlation only 0.39
among the stocks with a long history: it changed from 2025 to 2026). The strongest common
direction is liquidity: thinly traded stocks bounce from minute to minute, so the minute
follow-through, the 15-minute variance ratio, the band's follow and the jumps all lean on the
share of minutes without a trade (rank correlations 0.67–0.80). The stocks form one cloud with
denser parts, not islands: separation about 0.10 at 4–15 groups (0.01–0.02 with every number
shuffled across the stocks). The cut lines do not repeat well when each half forms its own
groups (adjusted Rand index 0.14–0.24), but a stock's group mostly lasts: with 6 groups, 61% of
the stocks are closest to their own group on both halves (4% by chance), with 4 groups 66%
(7%). NOW, GEV and RGTI fall in one group at 4, 6, 8 and 10 groups ("moves with the market ·
runs within 15 minutes · trades every minute", 232 stocks at 6 groups; GEV its 9th most typical
member, NOW the 62nd, RGTI the 211th).

**An eleven-stock group** (2026-10-18: NOW, GEV, LNC, JEF, COF, FTI, KMX, EXPE, GEHC, CCJ,
FLEX; one setting for all, everything free but the candle size): the sector funds are in
`data.SECTOR_FUND` (XLF, XLE, XLY, XLV, XLK, URA besides IGV and XLI). Data fixed first: the
daily candles of nine of them were stale (JEF's ended 2026-06-12) and Yahoo's are
dividend-adjusted (LNC's closes 8% below the 1-minute closes in early 2025, which would have
moved every level), so they and the six funds now come from IBKR, unadjusted like the minutes
(daily and 1-minute closes agree within 0.02%). Nine stocks' 1-minute candles end 2026-09-29/30,
so the validation window ends there (`tuning.labs` cuts it to the last day every picked stock
has candles). A 50-trial check: 16 s a trial (11 stocks), a third of the trials pruned (some
stock traded too little), about 1-2 GB of memory; the notebook continues a run started from a
script (`tuning.labs` builds the same labs in both).

**The eleven-stock run** (2026-10-18/19, 3,000 trials by two workers, 5 h 20 min; everything
free but the candle size; validation 2026-07-01 → 09-29): best training score 2.33 against a
luck line of 2.84 (3.37 one time in twenty): below luck. On validation the best scored −4.81
together (−$5,600; 3 of 11 stocks profitable: JEF, EXPE, FLEX), and so did the strategy as a
whole: the trials averaged −3.73, only 0.5% of them made money, the best tenth averaged −4.55
(none positive). Training and validation scores ran AGAINST each other (rank correlation −0.28
over 2,527 trials): what scored best on Jan 2025 – Jun 2026 did worse from July. The losses
were broad (NOW −2.8, GEV −3.4, FTI −3.2 on average over all trials; only JEF +0.8 and EXPE
+0.7 gained). The top 50 trials agree on a structure unlike the two-stock run's: both band
triggers, the moving-average, stock-trend and sector-trend filters, the wait after the open,
dealer gamma (at least −0.4), the typical-move stop, up to 5 sessions held, a target at the
option walls or in daily moves.

## Steps

1. **Look only** (built): the data check and every indicator drawn, nothing traded.
2. **First rules** (built): both band triggers, the sector filter, the trailing stop, the trade
   inspector, results by market condition (up, sideways, down) and by side (long, short).
3. **More switches** (built): the moving average, stock trend and market trend filters, and "what
   each switch changes".
4. **Levels** (built): the major-levels and previous-high/low filters, the profit target, and
   Rigetti as a third stock.
5. **Tuning** (built): the earnings guard, and the tuning notebook (every hyperparameter, the
   switches included; the two passes with the blended score; the effects; the walk forward;
   the export the main notebook loads). Only short runs so far.
6. The final check on the untouched months.

## Files

| File | What it holds |
|---|---|
| `data.py` | loading the 1-minute and daily candles, candles of any size, the data check, report dates (Yahoo, cached in `var/classic_signals/`) |
| `indicators.py` | the maths only: Bollinger Bands, moving averages, typical move, daily trend, major levels, the previous session's range, the nearest level above and below each candle, what was known at a session's open |
| `rules/` | one small file per rule, all with the same shape: `NAME`, `KIND` (trigger or filter), `DESCRIPTION`, `PARAMS` (its levers: label, default value and range; the notebook makes a slider of each, and step 5's optimizer will search the same range), `signal` (a trigger: +1 / -1 / 0 per candle) or `allow` (a filter: long and short allowed per candle), and `explain` (what it saw, in words). A new rule is one new file, listed in `combine.RULE_FILES` |
| `combine.py` | the switchboard: loads the rules and decides, candle by candle, from the switched-on ones |
| `simulate.py` | the trades: next-open entries, the stop (trailing by typical moves or typical daily moves, or fixed), gap fills, the profit target, the holding limit, the size (the same dollars at risk at every first stop, `costs`), costs (the inner loop compiled with numba) |
| `results.py` | summaries by condition, side and setup; the profit curve, drawdown and bottom line; what each switch changes (off against on, by condition and half-year); the walk forward's table; a trade's story in words |
| `hyperparams.py` | every hyperparameter in one table (value, range, searched range, what a change rebuilds), the search space (free by default; overrides fix a value) and the exported settings files |
| `pipeline.py` | from candles to what the rules read: the one computation both notebooks run |
| `tuning.py` | step 5: the lab per stock, the daily Sharpe score and the luck line, the optimization (training and validation; runs kept in the journal `studies/tuning.log` for optuna-dashboard and continued by name), the walk forward, the effects; walks run in separate processes and kept on disk |
| `charts.py` | every figure, on the repo's palette |
| `classic_signals_marimo.py` | the main notebook: `uv run marimo edit strategies/classic_signals/classic_signals_marimo.py` (or open it in VS Code) |
| `classic_signals_tuning_marimo.py` | the tuning notebook (step 5); its export is the main notebook's *Starting values* |
| `groups.py` | stock groups: every stock's character numbers (measured from per-session sums kept on disk, `var/classic_signals/groups/`), their scores, the Ward tree, the groups' quality and stability, each group's members to tune on and to check on, the export; `uv run python strategies/classic_signals/groups.py` measures the pool from a terminal |
| `classic_signals_groups_marimo.py` | the stock-groups notebook: the pool, how each number is measured, the groups, one stock, one group (its members and their candles), the export |
| `IDEAS.md` | ideas parked for later (the two-pass optimization for step 5, small levers offered) |
| `tests/test_classic_signals.py` | offline tests on hand-made candles, including a look-ahead check for every indicator |

This folder has no `strategy.py`: it is a research workbench with its own notebook, not (yet) a
strategy for the repo's shared backtest engine.

## Data

1-minute candles from Interactive Brokers since 2025-01-01 for NOW, GEV, SPY, IGV and XLI, and
daily candles since 2024-01-01 (downloaded 2026-10-16; the repo's backfill tools add newer
days). Rigetti (RGTI) and its sector fund QTUM (the Defiance Quantum ETF, about 70 quantum and
machine-learning companies, Rigetti about 2% of it) were added on 2026-10-17: all 448 sessions
of 1-minute candles for both; Rigetti trades every regular-hours minute, while 23% of QTUM's
have no trade (a thin fund; only its daily trend is used by a rule). QTUM's daily candles came
from Yahoo first, then from IBKR like the others. Besides the regular session, the pre-market (04:00-09:30), after-hours (16:00-20:00) and,
where stored, overnight (20:00-04:00, IBKR's OVERNIGHT venue, `scripts/ibkr_overnight.py`)
minutes are read as information; the strategy trades the regular session only.

End-of-day **option chains** (MarketData.app, `scripts/marketdata_dump.py`,
`var/marketdata/chains/<symbol>/`) for NOW, GEV and RGTI from 2025-10-14 (the trial reaches back
one year) to 2026-07-31 so far: quotes, volume and open interest per contract, no implied
volatility or Greeks. Measured on them (2026-10-18): the straddle's expected daily move runs
above the realized one (median 3.7% against 3.0% for NOW, 3.55% against 2.9% for GEV) and
predicts the size of the next session's move better than the recent realized moves (gap: rank
correlation +0.40 against +0.36 for NOW, +0.26 against +0.12 for GEV); after the top tenth of its
ratio to the realized moves, the next gap was about 1.5 times as large. On the NOW + GEV run's
best setting (found without the filter) the trades opened on high-ratio days were not worse
(17 trades above 1.8: +$40 each, against +$36 overall), so the filter lowered its training score
(3.10 → 2.97 at 1.8) and nudged validation (−1.16 → −0.73, July only): no evidence yet either
way.

**Everything in the chains, measured** (2026-10-18, NOW, GEV and RGTI, 200 days each; each
feature at a close against the next session; rank correlations, about ±0.08 is luck for the
three together). Implied volatility and the term structure forecast how much the next session
moves, on all three stocks (30-day IV: gap +0.31, range +0.38; front ÷ 30-day IV: +0.11,
+0.20). Nothing forecasts its direction (every feature within ±0.06), nothing its intraday mean
reversion (dealer gamma −0.02: the dealer-gamma story does not show here), and nothing the best
NOW + GEV setting's daily profit (all within ±0.11). The 25-delta skew, put/call volume and open
interest, theta and vega of all open interest and the walls' distances: within luck or
inconsistent between stocks. Each new rule switched on, on that best setting: none improved
both training and validation; the best pairs (*iv term* 1.3: 2.84 / −0.61; *options calm* 1.8:
2.97 / −0.73, against 3.10 / −1.16 off) rest on July alone, the only validation month with
option data.
