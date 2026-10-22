# darts_forecast — a Darts forecast of the next bars, walk-forward, retrained on a schedule

**Thesis.** A model reads the current and past state of a stock and predicts its next move:
- intraday detail from 1-minute bars;
- multi-day context;
- options-market data;
- the clock.

If the forecast beats costs, trade it. The model is retrained on a schedule (default: before every session) on
only the last `train_days` sessions, because "the current regime usually lasts a few more days". When the regime
breaks, a **model-health gate** and volatility-scaled stops and sizing limit the damage.

**What the direction study already says** (`strategies/burst_scan/DIRECTION_REPORT.md`):
- 1-minute OHLCV of these names carried no tradable direction;
- the free options features showed no link to next-day or next-week direction (|t| ≤ 1.7).

This strategy is the harness to test that properly with forecasting models, and to test new inputs (order
book, …) the same way. The first runs (below) agree with the study.

## Quick start

**New here? Start with `darts_forecast_walkthrough.ipynb`.** It explains the whole pipeline step by step in plain
language, at 1-hour bars and a 5-day horizon, up to a price forecast and an honest grade of it (no trading).
**Choosing a bar size and horizons? Use `darts_forecast_explorer.ipynb`.** It picks 12 diverse stocks (large / mid
/ small cap × slow / fast) with complete 1-minute data; a multi-step Darts model predicts the whole path of the next
bars, and Optuna chooses the bar size (1–60 minutes), the input length (50–1,000 bars) and the prediction length
(10–100 bars) by how well the predicted path agrees with the actual one, graded against "no change", against luck
and then on an untouched month (`SMOKE = True` runs it end to end in about a minute).
**Intraday only (the night left out)? Use `darts_forecast_intraday.ipynb`.** The same 12 stocks; every prediction
stays inside one session and stops at 16:00, the overnight gap is an input known at the open, and the model is told
when each input bar is from (a "today" flag and the clock). It measures every bar size (1, 2, 3, 5, 10 minutes) ×
input span (30 minutes to a session) × output span (10 minutes to the rest of the day) on one shared truth (the move
from the fill to fixed checkpoints), beside two baselines, a planted positive control and a model-free map of which
part of the day predicts which (`SMOKE = True`: about two minutes).
**From a fixed morning time? Use `darts_forecast_daystart.ipynb`.** The same 12 stocks; ONE example per stock-day:
the model reads the day from 09:30 + an offset for L minutes and predicts the next H minutes, never past 16:00.
Every bar size (1, 2, 3, 5, 10, 15, 30 minutes) × offset (0–60) × input (15–120) × output (30–360 minutes) is measured,
graded like the explorer; optional inputs: each bar's volume and two previous-day numbers (`SMOKE = True`: under a
minute).

1. `uv sync --all-extras` installs Darts and torch (the optional `darts` extra).
2. Collect or refresh the free options history with `strategies/burst_scan/options_free_data.ipynb`. Without it
   the option inputs are missing; their flags say so, and `forecast()` warns.
3. Pick the forecaster in `config.yaml` (or in a notebook's §0 knobs):
   - `model`: `gbm` | `gbm_clf` | `linear` | `tide` | `tsmixer` | `tft`;
   - `bar_minutes`: the forecast resolution, 1 / 5 / 10 / 15 / 30 / 60 (default **5**; 60 = hourly, the session's
     last bucket is 30 minutes);
   - `horizon`: grid bars ahead; with `overnight: true` it may span sessions (60-minute bars: 35 = 5 days);
   - `expanding`: train on all history before each retrain point instead of the last `train_days` sessions;
   - `retrain_minutes`: how often to retrain, in trading minutes. 390 = before every session (default),
     1950 = weekly, 60 = hourly, 0 = once.
4. Run the notebooks in order: data → model → strategy → diagnostics (below).

## The data split (the notebooks print it as a timeline)

| window | dates | used for |
|---|---|---|
| **TUNE** | 2025-06-18 → 2026-01-31 (`data.start` → `data.end`) | Everything a knob may be chosen on. The first 21 sessions are warm-up (daily inputs, volatility scale). Then each retrain point **fits** on the previous `train_days` sessions (**TRAIN**). TiDE / TFT hold out the last `val_days` of those for early stopping (**VALIDATION**). Each fit **forecasts** the bars until the next retrain point, so every scored forecast is out of sample. Scoring starts at `optuna.score_from` (2025-09-16, the first fitted model). |
| **TEST** | 2026-02-01 → 2026-04-30 | The frozen config, scored in `darts_forecast_diagnostics.ipynb`. The walk-forward simply continues into it, warm (`research.load(..., end=WINDOWS["test"][1])`). |
| **HOLDOUT** | 2026-05-01 → last stored bar | Read once, behind `READ_HOLDOUT = True`. |

## The inputs (all on one `bar_minutes` grid; Darts needs one time axis)

| group | inputs | from | known |
|---|---|---|---|
| intraday | return, realized vol, range, close location, signed-volume flow, volume vs its time-of-day norm, return since the open, distance to the session VWAP | the 1-minute bars inside each grid bar | at the grid bar's close |
| market | the other names' average return | the panel | at the grid bar's close |
| daily | previous-day, 5- and 20-day returns (in volatility units), 20-day volatility, position in the 20-day range, previous close location and relative volume; today's gap | the panel's sessions | previous close (gap: the session's first bar) |
| options | ATM IV, skew, term structure, its day-over-day change, IV − HV, OCC customer put/call and volume surge; put/call OI and volume, OI change, dealer gamma | the free options store (`burst_scan/options_free_data.ipynb`) | after the previous close; never staler than 5 days |
| flags | "has data" per option source: `has_dolt_iv`, `has_dolt_hv`, `has_occ`, `has_oi` | — | — |
| seasonal (`seasonal: true`) | the name's mean realized forward target from the same clock time over the previous 20 sessions whose outcome is complete (intraday seasonality, Heston, Korajczyk & Sadka 2010; at a horizon of a day or more a slow momentum) | the target's own past | at the decision (sessions at least ⌈horizon / bars per session⌉ back) |
| future | time of day (sin, cos, fraction), weekday, monthly-expiry week (from the calendar) | the calendar | in advance |
| static | which name (one-hot): one global model for all names | — | — |

The panel is the stored 1-minute panel (`data.bar: 1m`). `bar_minutes` only sets the forecast grid, so the
1-minute detail reaches the model and fills stay at 1-minute resolution. A slower input is repeated on every grid
bar of its session; a faster one is summarized into each grid bar. Missing inputs enter as 0 next to their flag,
because Darts refuses NaN.

The stored "regular" session keeps bars after the 13:00 close on NYSE half-days (engine proposal, `IDEAS.md`).
`strategy.HALF_DAYS` masks them: they are neither inputs, targets nor trading bars.

## The models (`model`)

| model | Darts class | measured speed (5 names, 5-min grid, 40-session window) | output |
|---|---|---|---|
| `gbm` | `SKLearnModel(HistGradientBoostingRegressor)` (the LightGBM algorithm; its own lag length per input; a fixed `gbm_iter`) | ~1 s per fit since `strategy.fast_bins` (was ~5–11 s) | expected return |
| `gbm_clf` | `SKLearnClassifierModel(HistGradientBoostingClassifier)` | similar | P(up) |
| `linear` | `SKLearnModel(Ridge)`, penalty 0.1 × the training samples, on the same lags as `gbm` | well under 1 s per fit | expected return |
| `tide` | `TiDEModel` (MLP encoder-decoder; past, future and static inputs) | ~5–8 s per fit, ~8 min daily | quantiles → median and P(up) |
| `tsmixer` | `TSMixerModel` (MLP mixing time steps and inputs; past, future and static inputs) | ~5 s per fit (explorer: 12 names, 30-min grid) | quantiles → median and P(up) |
| `tft` | `TFTModel` (variable selection, attention) | ~75 s per fit, ~2 h daily; use `finetune_epochs` | quantiles → median and P(up) |

**Fast binning (`strategy.fast_bins`).** scikit-learn 1.9 bins a WEIGHTED gradient-boosting fit with 254
weighted percentiles per input, one sort each, which made a 1,300-input fit take a minute. The walk-forward's
weights are a 0/1 "usable label" mask, for which the bin edges are the unweighted percentiles of the rows kept.
`fast_bins` computes those directly, about 18× faster to bin. The models are the same: the cached walkthrough
and 5-minute forecasts of the previous code version reproduce bit for bit, and the test compares edges and
predictions. An edge can rarely differ in its last bit, a rounding of a midpoint between two data values that
no value can fall on. Recency weights and fits above the binner's 200,000-row subsample take the stock path. It
overrides a private scikit-learn method; the test fails loudly if an upgrade changes it.

**The training window's first labels (`lag_history`).** A fit's target series starts at its first label, so the
first `lookback + horizon − 1` labels of every window have no lagged target and are dropped. That handicaps
long input horizons: a 10-day input at 30-minute bars lost a quarter of the explorer's first fit. With
`lag_history: true` the series starts that much earlier, with weight 0 before the window, so every label of the
window trains. It is off in `config.yaml` (the shipped study and the walkthrough were run without it) and on in
the explorer.

Every model forecasts one point, `horizon` grid bars after its input window (`output_chunk_shift = horizon − 1`).
The target there is the sum of the last `horizon` z-scored grid returns, i.e. exactly the forward return of the
decision bar. Training targets whose decision bar is in another session are weighted zero (no "close → next
open" pairs). `recency_half_life` weights recent sessions more. The quantile models' five quantiles are sorted
before use, since an undertrained net can cross them.

## Retraining (`retrain_minutes`)

| setting | value |
|---|---|
| before every session (default) | 390 |
| weekly | 1950 |
| every 20 sessions | 7800 |
| hourly (intraday retrain points) | 60 |
| once, a static model: the control | 0 |

For TiDE / TFT, `finetune_epochs > 0` continues the previous weights instead of refitting from scratch, which keeps
daily retraining affordable. The model notebook runs the cadences side by side on the same forecasts and the same
null; the strategy notebook trades them.

## Multi-day forecasts (`overnight`, e.g. 1-hour bars and 5 days)

- **The grid.** 60-minute buckets start at the 09:30 open: 09:30, 10:30, …, 14:30 and a 30-minute 15:30 bucket, 7 per
  session. A half-day keeps its empty post-13:00 buckets, so `k × 7` buckets later is always the same clock time `k`
  sessions later: a 35-bucket horizon is "this time of day, 5 trading days from now".
- **The target** (only the target changes): log(close 35 buckets later ÷ close now) — close to close, the five nights
  included (26–44% of a 5-day move happens while the market is closed) — divided by the name's causal 5-day volatility
  (`inputs["tscale"]`), so one unit is one typical 5-day move. The inputs stay the same: a session's bucket returns
  start at its open, and the night enters as the daily `gap` input. A label counts when the decision and the label
  buckets are real ones exactly 35 clock positions apart (a missing hour drops the labels that cross it).
- **Training.** With `expanding: true` every fit learns from all labels realized before its retrain point, from the end
  of the 21-session warm-up; `train_days` is the history required before the first fit. Labels overlap heavily at a
  long horizon, so the gradient-boosting leaves hold several horizons of rows.
- **Prices** must be dividend-adjusted for close-to-close targets: `research.load(..., adjust=True)` multiplies each
  session's 1-minute prices by the daily bars' adjustment factor, relative to the panel's last session (the stored
  1-minute bars are not adjusted; CSCO's ex-dates would otherwise be −0.55% fake overnight losses).
- **Not tradable yet.** The trading rule is flat by the close, so `valid()` rejects `overnight` (`forecast_valid()`
  checks the forecaster's knobs alone).
- **Judging it.** Forecasts made every hour share most of their 5-day outcome, so a day-clustered t-statistic calls
  luck "significant" a quarter of the time. The walkthrough uses an outcome-flip luck band instead (each hourly move
  keeps its size and gets a random direction, one coin per hour for all names).

## The trading rule and risk (the only searched knobs)

1. At each grid bar's close: forecast return over the hold > `edge_bp`, and P(direction) ≥ `p_min` when the model
   gives one, → buy (or short) at the **next 1-minute bar's open**.
2. Hold `hold_bars` grid bars (0 = `horizon`). The book follows an *intended position* per name:
   - a new same-side forecast extends the hold;
   - an opposite one closes the position, and the new side opens one minute later;
   - everything is flat at the session's last real bar (`eod_exit`).

   No decisions are taken in the first `skip_open` or last `skip_close` minutes, nor any whose hold would cross the
   close.
3. **Stop** at `stop_vol` × the name's typical move over the hold. A stop ends the trade; the holding's next same-side
   decision opens a new one.
   **Size** = min(1 / `max_positions`, `risk_per_trade` / stop) of the free cash of the ONE shared $10k wallet. At
   `risk_per_trade` 0.001 the risk rule sets it: the median stop on these names is 0.62%, so a trade takes about
   16% of the free cash and a stop-out loses 0.1% of it.
4. **Model-health gate:** a session is traded only if the model's forecasts over the previous `health_days`
   sessions ranked the realized returns better than `health_min`. It uses completed outcomes only. It stands aside
   after a regime change instead of trading through it.

`order_notional: 2000` is the real order size of a $10k wallet over 5 slots. The $1 minimum commission is then
5 bp per side: **14 bp per round trip** with spread and slippage (`research.round_trip_bp`). The study scores
`alpha_cagr` alone: it is a long/short book that is flat most of the time, and `edge_bp` / `health_min` are
exposure knobs, which a volatility objective would turn into an exposure dial.

## Notebooks (in this order)

1. `darts_forecast_data.ipynb` — the inputs:
   - the split timeline;
   - coverage, every input over time and on a sample day, correlations;
   - the timing/leakage checks;
   - a first univariate look (TUNE only).
2. `darts_forecast_model.ipynb` — the forecaster:
   - one model walk-forward, with TRAIN / VALIDATION / forecast per fit;
   - forecast vs outcome (IC, hit rate, calibration, by month / name / hour);
   - the **retraining study** (once / every 20 sessions / weekly / daily);
   - input-group importance (one static model, strictly out of sample);
   - the coin-flip null;
   - other models.
3. `darts_forecast.ipynb` — the strategy:
   - a backtest against the buy-and-hold baseline;
   - from forecast to trade;
   - the Optuna study on the trading rule, run in-process on the panel with options attached;
   - the pick, candidates and boundary check;
   - the cadence, traded;
   - cost stress;
   - a TEST replay behind `READ_TEST = False`.
4. `darts_forecast_diagnostics.ipynb` — the frozen config on TEST:
   - the truncation check (appending TEST changes nothing in TUNE);
   - forecast quality and the backtest on TEST;
   - pessimistic fills, 2× costs, the health gate's effect;
   - the null;
   - the one-shot HOLDOUT behind `READ_HOLDOUT = False`;
   - CHECKS and a SCORECARD.
5. `darts_forecast_walkthrough.ipynb` — a plain-language walkthrough of the forecaster at **1-hour bars and a 5-day
   horizon** (`research.WALK_DEFAULTS`: `overnight`, `expanding`, daily refits, `options: iv`), from the 1-minute
   bars to a 5-day price forecast, without trading:
   - the split, the data checks, dividends (`load(adjust=True)`) and earnings dates;
   - how the hourly bars and the close-to-close target are built (checked by hand), what one forecast reads, and
     how long its history is (the input window, the training history, the warm-up);
   - the daily walk-forward and its two leak tests (a re-run on scrambled later bars, and a model fit on the data
     up to a day reproducing the walk-forward's forecasts of the next), one forecast end to end, the predicted vs
     the actual price;
   - direction against the obvious guesses (always up, drift, momentum, reversal) with an outcome-flip luck band
     (the path guesses rebuilt in every make-believe future), the classifier's P(up), the evidence ledger, the
     coin-flip leak detector, the retraining comparison (answered in words);
   - the latest forecast, the gaps, a pre-registered verdict, CHECKS and a SCORECARD (TEST behind `READ_TEST`).
6. `darts_forecast_explorer.ipynb` — a workbench for a baseline: which bar size, input length and prediction length
   let a model predict the next PRICE PATH best (no trading). `SMOKE = True` runs every step small in about a
   minute; the full run takes 30–45 minutes, seconds from the cache.
   - 12 stocks picked by rule (`explorer.candidates` / `explorer.pick`): complete 1-minute data on every session of
     the window, a trade in ≥ 90% of its minutes, data over the last 6 months; two each of large / mid / small cap
     (today's cap) × slow / fast (volatility percentile in its cap group before the window), sectors spread;
   - six months inside TUNE (2025-08 → 2026-01). The target: every bar's close-to-close move (the open's bar holds
     the night) in that clock time's typical size over the last 20 sessions, clipped at ±10 (`explorer.path_inputs`).
     One multi-step model (TiDE, TSMixer or DLinear; point forecasts, MSE loss) per configuration for all the stocks
     reads the last *L* bars and predicts the next *H* at once. It is fit once on TRAIN (2 months); its last 5
     sessions stop training early (best epoch kept) and give the scale β (0–1) the paths are multiplied by. It is
     graded on VALIDATION (3 months) at the same decision times for every bar size (10:30 … 15:30). TEST (1 month) is
     read once, refit on TRAIN + VALIDATION. Three leak guards: the fit, the grading windows, the truncation re-run;
   - the grade: R² of the predicted cumulative path against "no change", per stock and lead time, averaged over
     the lead times and equally over the stocks. Beside it: the unscaled paths' R² against a luck band (direction,
     z), the IC per lead time, an AR(1) baseline (1-minute bars bounce: lag-1 autocorrelation negative for all 12
     stocks);
   - the Optuna (TPE) search over bar size (1 · 2 · 5 · 10 · 30 · 60 min) × input (50 · 100 · 200 · 400 · 1,000 bars) ×
     prediction (10 · 20 · 50 · 100 bars), skipping what the data cannot support (≥ 250 training windows and ≥ 10
     independent validation paths per stock: 96 of 120 combinations); the predicted paths of the best configuration
     of each bar size against the actual price, R² and IC by lead time, each knob's effect against luck and the bar
     for the best of k tries, a map per bar size, Optuna's importance, the winners on TEST, CHECKS and a SCORECARD.
7. `darts_forecast_intraday.ipynb` — the explorer's intraday sequel: predicted paths that never pass 16:00 and never
   hold the night (no trading). `SMOKE = True` runs every step small in about two minutes; the full run takes about
   an hour, a minute from the cache.
   - The same picks and window. Half-days and the session after each are left out (`intraday.usable`). Volume-0
     placeholder minutes (a copy of the last price) are masked as no trade (`intraday.prepare`).
   - One series per STOCK-DAY: the previous session's last bars (input only) + the whole session, so every predicted
     bar is today's. Inputs known in advance: the clock and a "today" flag (0 on yesterday's bars, 1 on today's);
     the day's facts as static inputs (the gap, yesterday, 5/20 days, volatility, weekday, calendar days since the
     previous session, expiry week, the stock); the bar-by-bar intraday and market inputs as past inputs.
   - The target: each bar's intraday move (a session's first bar from its own open) in typical moves: the stock's
     RMS 1-minute move over the previous 20 sessions × the time-of-day profile pooled over the stocks
     (`intraday.scales`, one unit for every bar size).
   - ONE truth for every configuration (`intraday.outcomes`): the log move from the FILL (the first trade after the
     decision; the decision bar's last trade carries the bid-ask bounce) to checkpoints 10, 20, 30, 60 … 360 minutes
     later, in that unit; a bar size's predicted bar moves are read at the checkpoints (between two closes on the
     straight line). Decisions every 30 minutes from 10:00, graded when the whole output span ends by 16:00.
   - The paths are scaled month by month by β fit on the held-out week and the earlier months only
     (`intraday.chained_beta`); two baselines get the same treatment: the time-of-day mean path (all stocks, 20
     sessions) and a ridge on a dozen summaries known at the decision (penalty by cross-validation over whole days).
   - A planted 30-minute momentum (`intraday.planted`) checks that the pipeline finds an effect of a known size; a
     smallest-detectable-R² column says what each row could have seen.
   - Every cell of the grid (5 bar sizes × 5 input spans × 6 output spans: 150) is measured through Optuna's grid
     sampler; checkpoint and decision-time curves, each knob's effect, maps, importance; a model-free map of which
     half-hour predicts which (`intraday.atlas`, the "first x minutes → the rest of the day" question; its luck band
     flips each day's sign for every stock, which keeps the day's volatility: a shuffle of days does not, and passed
     10-22% of pure-noise cells at 1.96); ONE configuration, fixed by rule before the grid ran, read on TEST (refit
     on TRAIN + VALIDATION except their last 20 sessions, which give its scale); CHECKS (the stock-day series, a
     random night changing no target, the truncation re-run at 1- and 5-minute bars) and a SCORECARD.
8. `darts_forecast_daystart.ipynb` — the simplest within-day forecast: ONE example per stock-day, anchored in the
   morning (no trading). The full run takes about 15 minutes, seconds from the cache.
   - The same picks, window and split as the explorer, except that `TRAIN_FROM` lets the fit start earlier: from
     the first stored 1-minute day (2025-06-18), each day training once the typical sizes have 10 earlier
     sessions. Half-days are never used; volume-0 placeholder minutes count as no trade (`daystart.prepared`).
   - An example reads the bars from 09:30 + O for L minutes and predicts the next H minutes, at g-minute bars, with
     O + L + H ≤ 390: nothing crosses the close or a night. Every example of one configuration starts at the same
     clock time, so a bar's place in the window is its time of day (no calendar input). Each training series is
     exactly (L + H) / g bars: one Darts sample per stock-day.
   - The target: each bar's intraday move (the day's first bar from its first trade) in typical moves of its clock
     position (RMS over the previous 20 sessions), clipped at ±10. Inputs: the path so far, optionally each bar's
     volume (against the stock's average bar over the previous 20 sessions; a past input) and two previous-day
     numbers as static inputs (`daystart.previous`): the overnight gap in typical nights (withheld until the stock
     has traded) and the last `PREV_DAYS` sessions' move. They are examples of daily information; a test plants a
     gap effect and checks the model picks it up.
   - The explorer's grade: β from the last 10 TRAIN days (early stopping too), R² of the path against "no change"
     per lead and stock, direction against an outcome-flip luck band, the AR(1) baseline. All 372 cells
     (g 1·2·3·5·10·15·30 × O 0·15·30·60 × L 15·30·60·120 × H 30·60·120·240·360 minutes) are measured; each optional input is
     switched off in turn for the worked example and the winner (read on direction: β often makes R² exactly 0);
     the winner is read once on TEST (refit on TRAIN + VALIDATION, its last 20 days held out). CHECKS (boundaries, a
     truncation re-run that scrambles everything from one VALIDATION decision on) and a SCORECARD.
9. `darts_forecast_daystart_marimo.py` — does a Darts model have an EDGE OVER LUCK at one question: at a fixed
   time of day, is the stock up or down at the end of the horizon, and by roughly how much? A marimo notebook
   (`marimo edit <file>`, or the VS Code marimo extension with `.venv` as the interpreter). For now ONE stock
   (`SYMBOLS = ["COLB"]`: a $8B mid cap with median volatility among the mid caps and full 1-minute data from
   2024-12-26 to 2026-09-29; then `["DOCN"]`, DigitalOcean: of the names with full 1-minute data since January
   2025 and every data rule passed over Jan 2025 – Sep 2026, the most movement: 79 bp average 12:00 → 13:00
   move and 77%/yr daily volatility, against COLB's 32 bp and 32%/yr). The split is three explicit date ranges in the settings cell: `TRAIN` 2025 (its
   first 10 sessions are history only, its last 10 held out for early stopping: 227 days learned), `VALIDATION`
   Jan–Jun 2026 (123 days), `TEST` Jul–Sep 2026 (read once); Table 4 and Figure 0 show the exact days. The bars
   are loaded with `research.load` from TRAIN's first day to TEST's last (not `explorer.load_window`, which cuts
   by whole months and counts its warm-up on a holiday calendar that misses some 2025 closures). `READ_LOCKED =
   True` reads the months the darts study series reserves as its TEST / HOLDOUT (`research.WINDOWS`, Feb 2026
   on); COLB was never in those studies.
   - **The question is fixed** (`DECISION` = minutes after 09:30, default 12:00; `HORIZONS`, default `[60]`, each
     its own studies). Searched: the history length (an ordered number between the horizon and 09:30), the inputs
     (volume, previous-day inputs: `SPACE`, Table 1) and the model type's knobs (`SEARCH`, Table 7). One study per
     model type of `MODEL_TYPES` (TSMixer, TiDE, DLinear, TFT, NHiTS, BlockRNN) and horizon, `MAX_TRIALS`
     each, so every type gets a fair search. TFT, NHiTS and BlockRNN are built in the notebook (`make_model`) with
     `explorer.make_path_model`'s training setup (TFT as point forecasts with relative positions); NHiTS reads
     no static inputs, so the previous-day inputs are never drawn for it.
   - **The score is the END-MOVE CORRELATION**: predicted vs actual move from the decision to the end of the
     horizon over all VALIDATION stock-days. It is scale-free, so a muted output earns nothing. The old unscaled
     path R² rewarded muted output: it agreed with direction skill at only 0.31 and drifted with bar size, while
     the end-move correlation does not (614 cached grid results re-scored: rank agreement with the old score
     +0.30, with path direction z +0.69, flat across bar sizes). Reported beside it: hit rate (`hit_p`: binomial),
     size error after a scale β fitted on the held-out days (`mae_skill`), end R², loudness, per-stock values.
   - **The luck check**: every trial's correlation against `DRAWS` sign-flipped outcomes (same sizes, one coin per
     day for all stocks: `luck_z`, `luck_p`); the line above Table 8 compares the best trial's z with
     `explorer.best_of_bar` for the number of trials. Controls on COLB: a planted gap-driven drift is found
     (z +5.5, hit rate 80–86%) only when the model sees the gap (z +1.6 without); random signs give z +0.5.
   - **Training**: a searched `loss` knob, `bar` (Darts' per-bar MSE) or `endpoint` (only the summed move: on the
     planted effect it doubled the size skill, +0.41 vs +0.21), AdamW weight decay, patience `PATIENCE` (5), the
     best pass kept, no pruning (pruning on the first passes cut the trials that ended best). Reversible instance
     norm is never used: its inverse adds the input window's mean back, a drift (median R² −0.39 with it). Seed
     noise is real: the same setup's end correlation moves ±0.06 (bar loss) / ±0.14 (endpoint loss) across seeds,
     so a winner is re-run with several seeds before it is trusted.
   - **Records**: journals in `studies/` (`daystart-<type>-h<horizon>-<fingerprint>`), each trial's predictions in
     `var/daystart_trials/<study>/<n>.npz` (re-scoring needs no retraining), one W&B run per trial through
     OptunaHub's W&B callback (`callbacks/wandb`, pinned), logged once at the trial's step (W&B drops a log at an
     earlier step than the run's last). `.env` sets `WANDB_MODE=offline` and `WANDB_CONSOLE=off` (a stalled W&B
     service once froze a study through the console capture). Results: Table 8 (best per type), Table 9 (all),
     Table 10 and Figure 7 (the best trial per stock and per stock-day), Figures 4–6 (its study).
   - **Two modes, one switch** (settings block 2): `WALK_FORWARD` picks ONE of two blocks; `TEST` (read once) is
     shared. **2a · single split** (`False`): one model learns on `TRAIN` and scores all of `VALIDATION`. **2b ·
     walk-forward** (`True`): `WF_SCORED` (default Mar 2025 – Jun 2026) is cut into steps (`WF_STEP`: a month or
     an ISO week); each step is scored by its own model, which learns on the `WF_WINDOW_MONTHS` months just
     before it (a rolling window, as a model retrained live; `None`: every day since `WF_LEARN_FROM`, an
     expanding window), its last 10 days held out for early stopping and β. Defaults: monthly, 2-month window →
     16 models, each learning on ~31 days (the first on 20), 331 scored days; weekly gives 70 models. A trial
     costs one fit per step (rolling monthly: TSMixer ~10 s, TiDE ~4 s, DLinear ~3 s). Table 2 marks the block
     not in use; Table 4 and Figure 0 (`charts.split_calendar`) show one row per model plus TEST (history only /
     learns / held out / scores). Walk-forward studies carry "wf" in the name, fingerprinted by block 2b;
     single-split names are unchanged. Optuna's per-pass curve is the last model's; W&B's curve table has a
     `fold` column. Every trial records its correlation per calendar month (`month_corr`, `months_positive`) and
     saves each scored stock-day's date and actual end move. The section **Compare** (Table 11, Figure 8) sets
     each model type's study in both modes side by side and re-scores every trial, from its saved predictions,
     on the days BOTH modes score (default Jan–Jun 2026), so the modes compare like for like.

10. `darts_forecast_tsmixer_marimo.py` — the TSMixer deep dive (marimo): ONE TSMixer PER STOCK, one Optuna study
    per stock (`tsmixer-lab-h<H>-<batch>-<stock>`; the batch fingerprints every setting except the stock list and
    `MAX_TRIALS`, plus `tsmixer_lab.py`). `STOCKS` holds 40 stocks in six groups, market cap (large ≥ $10B, mid
    $2–10B, small < $2B) × dollar volume above / below the cap group's median, all with full 1-minute data since
    January 2025 (that pool only has tickers C–E, the backfill's order) and a trade in ≥ 50% of minutes. **run all
    stocks** loops over them with a progress bar (stock, trial, best so far, ETA), tops each study up to
    `MAX_TRIALS` (resumable; a first pass at 10 = the random trials only), skips a stock without data, and saves
    the best trial's weights per stock in `var/tsmixer_lab/models/<batch>/<stock>.ckpt` + `.json`
    (`tsmixer_lab.save_model` / `load_model`: weights only, because Darts' own save pickles the end-move loss,
    a local class). The leaderboard (Table 5) ranks the stocks by best luck z against each one's best-of-N bar,
    with the first 10 trials' mean (the same 10 random settings for every stock: TPE seed 0), Table 6 sums up
    the groups; the best of 40 stocks × N trials needs the bar for all of them (4.16 at 1,600 trials) and a
    TEST read. A picked stock's best model is loaded (not retrained) for the scrambling and SHAP sections. About
    20 s to load a stock, 2–4 s per trial. The earlier pooled design (one model for COLB, DOCN, MU, CSCO, ANET,
    AAL, COIN; best +0.05, z 1.3) stays in the comparison table. Single split only (TRAIN 2025, VALIDATION
    Jan–Jun 2026, TEST Jul–Sep 2026), the same end-move score and luck check as entry 9. The INPUTS: ten of them
    (Table 2), each `True`, `False` or `"search"` in `INPUTS`, chosen from the literature on intraday predictability and
    available at least daily: the stock's own path (always), volume, the minute's range, an order-flow proxy (the
    close's place in the minute's range), the market and the sector (equal-weight baskets of every stock with full
    1-minute data; the sector basket leaves the stock out), the overnight gap, the previous days' move, the same
    clock window on the previous 10 sessions (Heston, Korajczyk & Sadka 2010), and option signals from the previous
    close (IV − HV, skew, customer put/call; only MU, CSCO, ANET, AAL, COIN have option data, so the input refuses
    a stock without it). No sector or market ETF has long 1-minute data (SPY only since March 2026), hence the
    baskets: one SQL aggregation (~2 min), cached in `var/tsmixer_lab/`. Figures 1–2 draw every input of a chosen
    stock-day; Table 10 / Figure 10 compare every batch (one per inputs × settings), including the mean of each
    stock's first 10 trials (random draws, no selection). The helpers live in `tsmixer_lab.py` (builds on
    `daystart.py`, edits nothing there) and `tsmixer_importance.py`. **Never edit `tsmixer_lab.py` while its
    studies matter**: its source hash is part of every batch name, so an edit orphans them.

11. `darts_forecast_stock_marimo.py` — ONE stock under the microscope (`STOCK`, default CRM), on the saved best
    model of entry 10's batch (the notebook stops unless that model reproduces its study score on the dates set):
    the study (every trial, the 10 best), HOW THE SCORE WORKS (simulated correlations of 0 to 0.6 over as many
    days; every VALIDATION day as a dot that helps or hurts, sized by its share; the score added up day by day
    beside two traders acting on every prediction, sized and direction-only, before and after the repo's cost
    model, with the daily P&L split into correlation × spread × spread and average position × average move; nine
    days' price paths with the prediction: the three that helped most, three in between, three that hurt most),
    what carries the best trial's score (rank correlation, the score without the biggest days), the best settings
    retrained with `SEEDS` seeds and their
    averaged model, permutation importance and SHAP (`tsmixer_importance.py`, on the saved model or on the best
    settings retrained with new inputs), and NEW INPUTS: a function of the stock's raw minute arrays under
    `@new_input("day" | "bar", what=, why=)` in one cell. Every new input gets a look-ahead check (recomputed with
    everything after the decision hidden, on 8 sessions), its correlation with the end move on TRAIN and
    VALIDATION and with the saved model's misses, each with a luck z (Table 6, no training), and an ablation:
    the best settings trained with it added (alone, all together; optionally each current input dropped), every
    arm on the same seeds and days, the paired change with its t and the averaged model's change on each half of
    VALIDATION (Table 7); every run is appended to `var/tsmixer_lab/stock/<STOCK>-experiments.csv`. Seven inputs
    from the literature start the list (morning move, the market's morning move, the place in the day's range,
    morning volatility, volume so far, the past month's move, the distance from VWAP). The helpers live in
    `stock_lab.py` (data and raw arrays, the new-input registry and checks, `Inputs`: `tsmixer_lab` plus the new
    inputs, in the shape `tsmixer_importance` expects; `fit_seeds`, `compare`, `robustness`, the log); it edits
    nothing in `tsmixer_lab.py`. About 4 s per fit; TEST is never read.

12. `darts_forecast_movers_marimo.py` — who moves this afternoon, and which way? At 13:30 each stock gets three
    probabilities (down / flat / up, a move being at least `THETA` percent from the fill, the first trade after
    13:30, to the last trade before 15:30) from ONE pooled TSMixer trained with a class cross-entropy, so every
    stock-day counts once. What is scored is the CALL: the `K` = `CAPITAL // POSITION` likeliest movers a day,
    each long or short, every call one vote against random picks and the trailing-20-day-activity picks called on
    the PRIOR side, the side TRAIN's moves leaned to (`movers_lab.ladder`, `prior_side`: against a coin, an almost
    all-long trial scored z +1.5 on 5 stocks in the up-drifting first half of 2026). Training stops on every
    `HELD_EVERY`-th calendar week (`hurdle_lab.spread_held`; the old last-10-days hold-out was the 2025 holidays).
    A 40-trial Optuna study scores the z of the excess against the best-of-N bar; the best trial also gets its
    within-day label-shuffle luck band (Figure 3) and a re-run with other `SEEDS` (Table 7b); a callout warns when
    the stocks are no more than `K` (then nothing is picked and only the sides are tested). TEST is read once.
    Helpers: `movers_lab.py` (builds on `tsmixer_lab.py`, edits nothing there).
13. `darts_forecast_hurdle_marimo.py` — the same question, the same stock-days, labels, calls and ladder, answered
    by gradient-boosted TREES on a short table of numbers instead of a network on the raw path: the control for
    entry 12. A hurdle model (P(move) on every stock-day, P(up | move) on the ones that moved; a direct
    three-class tree beside it) with FIXED knobs, so nothing is searched; only the inputs change, in four steps
    (activity only, + day inputs, + morning summaries, + stock descriptors, never a stock id). Each row of Table 5
    carries its excess z over the activity rule, the 97.5th percentile of the same z with the labels shuffled
    inside each day (the calls kept) and the best-of-rows bar; TEST is read once. Before the steps, "Do the
    stocks differ?" (Table 4b, Figure 4b) asks whether a model PER STOCK could know more than one for all of
    them, without training any: every stock's own slope of each input (log-odds per standard deviation, so
    active and calm stocks share a scale), whether those slopes spread more than when the stocks are relabelled
    inside each day (TRAIN), and whether a stock's TRAIN slope comes back on VALIDATION (`stock_effects.py`).
    The textbook chi-squared p of Cochran's Q is not used: at ~50 moving days a stock it flags 20% of identical
    stocks. Helpers: `hurdle_lab.py` (`table`, `spread_held`, `fit`, `proba`, `shuffle_band`; builds on
    `movers_lab.py`, edits nothing there) and `stock_effects.py`; `tests/test_hurdle_lab.py` and
    `tests/test_stock_effects.py` test them offline. No new dependency (scikit-learn's
    `HistGradientBoostingClassifier`).
14. `darts_forecast_timesfm_marimo.py` — Google's pre-trained TimesFM 3.0 (Darts `TimesFM3Model`) used ZERO-SHOT:
    nothing is trained or tuned. Intraday: from 1-minute bars, the next hour at fixed clock times (never past the
    close); multi-day: from 10-minute bars, the next 5 sessions from each close (nights included). Each runs on the
    price only and on price + volume (a past covariate). Pickers draw one forecast against the actual price (median
    paths, the 10-90% band, "no change", the volume the model read); a gallery spread over the error ranking; the
    path error saved against "no change", the band's coverage and the predicted against the actual end move, each
    against a sign-flip luck band (one coin per session block, shared by every stock). The weights
    (`google/timesfm-3.0-pytorch`, ~1.3 GB, snapshot pinned in `timesfm_lab.REVISION`) download on the first run
    and are under the TimesFM Non-Commercial License (research and backtests, not production trading); forecasts
    are cached in `var/timesfm_lab/`. Section 3 grades the band's SIZE against two cheap Gaussian bands centred on
    "no change" (`vol_bands`: the clock profile of squared returns over 20 sessions, and that profile scaled by
    the last hour's / 5 sessions' realized level), scored by the pinball loss of the 10 % and 90 % quantiles
    (`size_compare`, day-clustered paired z; bar fixed before the run: z > 2 intraday). Result (2026-10-16,
    DirectionFive, TUNE): no direction (inside the luck band everywhere; volume only adds noise), and the cheap
    band beats TimesFM's centred band intraday (−4.4 % band loss, z −6.5; every stock, every clock time, every
    one of 15 cheap-band variants) and multi-day (−4.5 %, z −1.8). TimesFM wins only the first ~5 minutes
    (+1.6 %), carries the morning's volatility into the lunch lull, over-predicts its biggest moves (top tenth
    195 bp predicted, 134 realized) and cannot see the night in the stitched series (first bar after the close:
    23 bp predicted, 227 bp realized). Even an in-sample blend adds at most ~1 % to the cheap band. Helpers:
    `timesfm_lab.py` (loads `research.py` and `charts.py`, edits nothing there); `tests/test_timesfm_lab.py`
    tests it offline without the model (including a no-look-ahead check of the cheap bands).
15. `darts_forecast_pooling_marimo.py` — one model for all stocks, or one model per stock? On entry 13's stocks,
    stock-days and input table (13:30 → 15:30), three ways to learn the same targets with the same kind of small
    boosted tree: **pooled** (blind to the stock), **pooled + stock** (the stock as a tree category: a branch of
    its own where its rows support one) and **per stock** (one tree per stock, from its own rows only). Each
    model chooses its own leaf size and rounds on its own held-out weeks (`hurdle_lab.spread_held`). Two
    targets: moved at least `THETA` (default 0.5%) either way, and up (every day, no threshold). The measure is
    the WITHIN-STOCK AUC on VALIDATION (telling a stock's own days apart; knowing which stock moves more earns
    nothing): head to head (Table 5, Figure 1), as a transfer map (every stock's model rating every stock's days,
    Figure 2: a diagonal that stands out means the stocks really differ) and on planted labels (none / shared /
    flipping between stocks, Table 8) that show what each answer looks like. No calls, profit or luck bars;
    TEST is never loaded. Helpers: `pooling_lab.py`; `tests/test_pooling_lab.py` tests it offline.

The notebooks' helper functions live in modules beside them, each loaded by path (`strategy.py` imports none
of them):
- `research.py`: loading, the option inputs, the forecast cache, the metrics and studies, the notebooks' tables
  (the trading rule's funnel, backtests, stress, monthly returns) and small date and format utilities;
- `charts.py`: the figure helpers, on the engine's palette and thinning (`trading_simple.plots`);
- `checks.py`: the integrity and leakage checks behind the CHECKS rows;
- `explorer.py`: the explorer notebook's stock picks, split, path forecaster (inputs, fit, predict, scale), path
  grade with its luck band, AR(1) baseline, Optuna search and TEST check. It builds on `research.py`, which it
  loads once and exposes as `explorer.rs` (the notebook uses that one copy);
- `intraday.py`: the intraday notebook's data preparation (placeholders, the full clock), the shared typical-move
  unit, the graded truth, the stock-day series, fit and predict, the grade with its luck band, the month-by-month
  scale, the baselines, the planted control, the map of the day, the grid search and the checks. It builds on
  `explorer.py` (`intraday.ex`, with `intraday.rs`);
- `daystart.py`: the day-start notebook's minute arrays, bars and typical moves, the volume and previous-day inputs,
  the stock-day series, fit and predict, one configuration end to end (graded with the explorer's functions), the
  grid, the inputs-off comparison and the truncation check. It builds on `explorer.py` (`daystart.ex`, with
  `daystart.rs`) and imports nothing from `intraday.py`;
- `tsmixer_lab.py`, `stock_lab.py`, `tsmixer_importance.py`, `movers_lab.py`, `hurdle_lab.py` and
  `stock_effects.py`: the helpers of entries 10-13 (each builds on the one before it and edits nothing there;
  `stock_effects.py` reads only `hurdle_lab.table`'s rows);
- `timesfm_lab.py`: entry 14's clock grid, forecast origins, cached TimesFM forecasts, scorecard with its luck
  band and figures;
- `pooling_lab.py`: entry 15's targets, the tree that chooses its own leaf size and rounds, the three
  approaches with their within-stock AUC and transfer map, and the planted labels (reads `hurdle_lab.table`'s
  rows; edits nothing in the other helpers).

`charts.py` and `checks.py` never load `research.py`: a notebook passes in what they need.
`tests/test_darts_forecast_helpers.py` tests them offline.

## Walkthrough results (2026-10-04; 1-hour bars, 5-day horizon, gbm, daily refits, TUNE)

`darts_forecast_walkthrough.ipynb`, all 25 checks passing:

| | value |
|---|---|
| graded forecasts | 3,090 (decisions 2025-09-16 → 2026-01-23), but only ≈ 52 independent 5-day outcomes (18 weeks per stock × 2.9 effective stocks: the five move together) |
| per-name rank IC (the primary number) | −0.078; luck alone gives −0.003 ± 0.091, so z −0.8; the 95% band runs −0.26 … +0.10 |
| direction | right 46.5% of the time against 52.5% for a coin with the same up/down mix: skill −6.0 points (z −1.9). "Always up" was right 56.8% (the TUNE trend) |
| predicted vs actual price | price error 8.8% against 7.1% for "no change" (Theil U 1.25): worse than not forecasting. "No change" alone scores a price R² of 0.30–0.87, which is why a price chart flatters any forecast |
| retraining | per-name IC: once +0.02, weekly −0.04, every session −0.08. It did not help; refitting is borderline worse in skill (z ≈ −2.1 on 2 related comparisons) |
| baselines (drift, momentum, reversal) and the classifier's P(up) | none shows direction skill beyond luck |
| leak detector (the whole pipeline on 3 coin-flip markets) | clean |
| verdict (pre-registered) | **no evidence of a large edge.** The data cannot rule out a small one (IC up to +0.10); only an IC of about 0.25 would have been found reliably |

The 5 overnight gaps carry 31–45% of a 5-day move, and earnings weeks (9% of the forecasts) 17% of the squared moves.
The next steps that could change the answer are more evidence (more stocks: about 510 have 1-minute bars; or daily
bars back to 2015) and new information (an earnings input, order-book data); see `IDEAS.md`.

## Explorer results (2026-10-13; path forecasts, 12 picked stocks, Aug 2025 → Jan 2026, TiDE)

`darts_forecast_explorer.ipynb`, all 10 checks passing (the truncation re-run: 756 paths and the scale identical
before the cut); 33 minutes:

| | value |
|---|---|
| configurations the data supports | 93 of 120 (60-minute bars: one input / prediction pair; 30-minute bars: no 100-bar prediction) |
| the search (60 trials, objective: the scaled paths' R² against "no change") | 3 of 60 above 0, the best +0.0003 (2-minute bars, input 400, predict 100; scale 0.06); 18 scaled to "no change" outright (R² exactly 0) |
| direction | 0 of 60 above luck: the best z is 2.0, against a bar of 3.3 for the best of 60 tries |
| best R² of each bar size | 1 min +0.0000 · 2 min +0.0003 · 5 min +0.0000 · 10 min +0.0000 · 30 min +0.0000 · 60 min −0.0005 |
| the networks' own (unscaled) paths | R² −0.54 to −0.03: worse than "no change" everywhere |
| which knob moves the grade (Optuna fANOVA) | bar size 57%, prediction length 29%, input length 14%: mostly how loud the paths of coarse bars and long predictions come out, which the scale then shrinks |
| models at 5-min bars, input 200, predict 20 | DLinear −0.0008, TiDE −0.0039, TSMixer −0.0079 (direction z −0.3 / −1.1 / +1.2): all inside luck |
| TEST (January 2026, read once, refit on TRAIN + VALIDATION) | winner 0.0000 (scaled to "no change"; direction z −0.2), runner-up −0.0005 (z −1.0), example −0.0008 (z −2.5): none held |
| 1-minute bounce | lag-1 autocorrelation negative for all 12 stocks at 1 minute, 10 of 12 at 2 minutes, gone at 5; the AR(1) baseline reaches R² +0.0001 at best on the graded paths |

For these stocks and months, no bar size, input length or prediction length gives a predicted path that agrees
with the actual one better than "no change", and none points the right way more than luck. The baseline is "no
change"; the next knob should add evidence (more months, more stocks) or a different target (each stock's path
minus the market's) before it adds model complexity (`IDEAS.md` 26–31).

## Intraday results (2026-10-13; paths inside one session, 12 picked stocks, Aug 2025 → Jan 2026, TiDE)

`darts_forecast_intraday.ipynb`, all 20 checks passing (the truncation re-run at 5- and 1-minute bars: 1,392 and
1,512 paths and the first month's scale identical before the cut; a random overnight gap every night changed no
target, graded move or bar input); about 40 minutes. Two independent full runs gave identical numbers for all 150
cells:

| | value |
|---|---|
| the grid (every cell measured; objective: the scaled paths' R² against "no change" at the checkpoints) | 150 configurations: 11 above 0, the same 11 above both baselines; 9 of the 10 best predict 6 hours (10:00 → close), the noisiest cells: one graded path per stock-day |
| the winner | 5-minute bars, input 120 min, predict 360 min: R² +0.0072 (9 of 12 stocks above 0, scale 0.23), direction z +2.2 against a bar of 3.6 for the best of 152 tries; to be seen it needed R² 0.006 as one test, 0.015 as the best of 152 |
| best of each output span | 10 min +0.0000 · 30 min +0.0004 (3-min bars, input 60; z +1.0) · 60 min +0.0000 · 120 min +0.0000 · 240 min +0.0000 · 360 min +0.0072 |
| best of each bar size (all predicting 360 min) | 1 min +0.0029 · 2 min +0.0002 · 3 min +0.0022 · 5 min +0.0072 · 10 min +0.0017 |
| direction above luck (z ≥ 1.96) | 1 of 150 (the winner); luck alone gives about 4 |
| which knob moves the grade (fANOVA) | output span 60%, bar size 25%, input span 15%: mostly where the noisy 6-hour cells land |
| models at 5-minute bars, input 120, predict 60 | TiDE 0.0000 (scaled to "no change"; z −0.3), TSMixer −0.0025 (scale 0.21; z +0.8), DLinear −0.0003 (z +0.1) |
| baselines | the ridge on a dozen summaries +0.0005 (10 min) to −0.011 (240 min) by output span (scaled the same way: nothing it learned held beyond noise), the time-of-day mean path −0.0003 to 0.0000 |
| the planted control (a 30-minute momentum) | planted R² 0.0035 / 0.0101 / 0.0302 → the network recovered −0.0000 (z +1.3) / +0.0012 (z +3.8) / +0.0147 (z +8.2); the ridge −0.0010 / +0.0012 / +0.0138: an intraday effect worth R² ≈ 0.01 at 30 minutes is found, one of 0.0035 is not |
| TEST (January 2026, read once: the VALIDATION winner, refit on TRAIN + VALIDATION, its scale from their last 20 sessions) | R² −0.0331 (scale 0.77; ridge −0.0240), direction z −0.2: **did not hold**; TEST alone could only see R² ≥ 0.031 |
| the map of the day (model-free, TRAIN + VALIDATION, 102 sessions × 12 stocks; luck band: a sign flip per day) | the first 30 / 60 / 120 / 180 minutes against the rest of the day: corr +0.01 / +0.03 / +0.04 / +0.01, all inside luck (z ≤ 0.9). No cell clears the bar for the best of 91 (\|z\| 3.5): the strongest are 15:00 → 15:30 (corr +0.10, z +2.6) and the night → 14:00 (−0.14, z −2.5; a few earnings days: −0.05 without the 5 most influential). 10 of 91 cells pass \|z\| 1.96 (luck: about 5; neighbouring cells share days). The night alone against the whole session: corr −0.11, z −2.2, negative in all 5 months (a gap fade, weak). A first version shuffled days for this band, which is too narrow when days differ in volatility: it reported the night → 14:00 cell at z −3.9 |
| the day | the first minute moves 3.9× the day's average, 10:00 1.6×, midday 0.7×, the close 0.9×; 1-minute bars bounce (lag-1 autocorrelation −0.028, 11 of 12 stocks), gone at 10 minutes; a typical move is 39 bp in 10 minutes and 85 bp in an hour (median stock), so paying a 10 bp round trip by trading the forecast's sign needs R² ≈ 0.10 at 10 minutes, 0.02 at an hour, 0.0035 to the close |

For these stocks and months, no bar size, input span or output span predicts the rest of the trading day's path
better than "no change" in a way that survives the bar for the best of all the tries or a month nobody had looked
at. The planted control says that is informative down to about R² 0.01 at 30 minutes: an intraday pattern that
size, carried by the bars themselves, would have been found. The only lead is weak, model-free and daily, not
bar-by-bar: the overnight gap fading over the day (z −2.2, `IDEAS.md` 39), which needs years of daily bars, not six
months of minutes.

## Day-start results (2026-10-14; one stock-day per example, 12 picked stocks, Aug 2025 → Jan 2026, TiDE)

`darts_forecast_daystart.ipynb` with volume and the two previous-day inputs on, training from the first stored
1-minute day (`TRAIN_FROM = "2025-06-18"`: the fit learns from 2025-07-07, once the typical sizes have 10 earlier
sessions, to 2025-09-16, 51 days, then 10 held out). All 4 checks pass (the truncation re-run cut at 2025-10-15
11:00: 132 paths and the scale 0.340 identical before the cut, 612 later paths changed); about 9 minutes, SMOKE
under a minute. The first run trained from TRAIN's start only (32 days); its numbers are in brackets:

| | value |
|---|---|
| the grid (objective: the scaled paths' R² against "no change", 744 paths per cell) | 234 configurations: 34 above 0 [28], 86 scaled to "no change" outright [91]; direction z over all cells mean +0.07, SD 0.98 [−0.02, 1.01] (luck alone: 0 and 1), 7 at z ≥ 1.96 and 4 at z ≤ −1.96 [7 and 6] |
| the winner | 1-minute bars, read 10:30–11:00, predict to 11:30: R² +0.0077 (9 of 12 stocks above 0, scale 0.15), direction z +2.1; the best z of all cells (+2.9) is below the bar for the best of 234 (3.7) [30-minute bars, same stretch: R² +0.0061, z +1.6] |
| best of each knob | bar size 1 / 5 / 15 / 30 min: +0.0077 / +0.0039 / +0.0051 / +0.0051; offset 0 / 15 / 30 / 60: +0.0043 / +0.0039 / +0.0051 / +0.0077; input 15 / 30 / 60 / 120: +0.0044 / +0.0077 / +0.0040 / +0.0039; output 30 / 60 / 120 / 240 / 360: +0.0077 / +0.0051 / +0.0043 / +0.0044 / +0.0006 |
| the worked example (5-minute bars, read 10:00–11:00, predict to 13:00) | R² −0.0181 (scale 0.34), z −0.5 [−0.0117, z +0.1] |
| each optional input off (VALIDATION, read after the winner was chosen there) | worked example without volume Δz +0.25, without the previous days −0.50; winner without volume +0.48, without the previous days −2.30 (z +2.1 → −0.2), the one change past 2, among four read after the choice. On TEST it reverses (below) |
| TEST (January 2026, read once: the winner refit from 2025-07-07 through VALIDATION, its last 20 days held out) | R² +0.0005 (scale 0.04), direction z +0.7: **did not hold** [R² −0.0027, z −0.4]; without volume z +0.7, without the previous days z +2.1 |

Training on 51 days instead of 32 changes nothing of substance: the z-scores of all 234 cells are still spread as
luck spreads them, the winner moved to another bar size on the same stretch of the day and again did not hold on
TEST, and the previous-day inputs' apparent help on VALIDATION turned into the opposite on TEST.

**With 2-, 3- and 10-minute bars too** (the default ladder since 2026-10-14: 372 configurations; TRAIN and
VALIDATION only, TEST not re-read; about 13 minutes): 57 above 0, 140 scaled to "no change", direction z mean +0.13,
SD 0.98, 10 at z ≥ 1.96 and 6 at z ≤ −1.96; best of each bar size 1 / 2 / 3 / 5 / 10 / 15 / 30 min: +0.0077 / +0.0025
/ +0.0040 / +0.0039 / +0.0063 / +0.0051 / +0.0051. The winner is unchanged (1-minute bars, read 10:30–11:00, predict
to 11:30, z +2.1, below the best-of-372 bar of 3.8); the best new cell is 10-minute bars, read 10:30–11:00, predict to
12:00 (R² +0.0063, z +1.3).

## Explorer, first version (2026-10-13; single-point forecasts, 30-minute bars, gbm)

The first explorer graded ONE point per forecast (the change to the end of the output horizon) with gbm on
30-minute bars; it was replaced the same day by the path explorer above. Its result, all 18 checks passing:

| | value |
|---|---|
| stocks (by rule, §1) | large: OKE, AMT (slow), IREN, SMCI (fast); mid: DAN, MIR (slow), SMR, AAP (fast); small: SMPL, EYE (slow), EOSE, METC (fast) |
| models at input 1d → output 1d, VALIDATION per-name IC (z) | gbm_clf +0.058 (1.4), tsmixer +0.028 (0.6), tide +0.026 (0.6), gbm +0.008 (0.3), tft −0.037 (−1.1), linear −0.041 (−1.1): all inside luck |
| the search (gbm, 30 of 49 cells, objective `ic_z`) | 0 configurations with z ≥ 2; winner input 2d → output 5d, IC +0.107, z 1.5, against a bar of z 3.1 for the best of 30 tries |
| best z per output horizon | 30m 0.8 · 1h 0.4 · 2h 0.4 · 4h 1.3 · 1d 0.7 · 2d 0.4 · 5d 1.5 |
| TEST (January 2026, read once) | winner +0.055 (z 0.5), runner-up 4h → 4h −0.035 (z −0.8), example 1d → 1d +0.003 (z 0.1): none held |
| run time | about 7 minutes (tft 3 of them), about 1.5 minutes from the cache |

No input or output horizon predicts direction better than luck for these stocks over these months; at 5 days
the reversal guess (+0.094 on VALIDATION) is as good as any model cell. The numbers also move with small changes:
adding the first training labels (`lag_history`) took the 1d → 1d example from +0.044 to +0.008. The baseline to
build on is therefore "nothing yet", and the next knob should add evidence (months, names) before it adds
complexity (`IDEAS.md` 26–31).

## First results (2026-10-03; defaults: gbm, 5-minute grid, next bar, daily retrain, all inputs)

**The forecast on TUNE** (95 daily fits, 36,215 out-of-sample forecasts from 2025-09-16; `darts_forecast_model`):

| metric | value |
|---|---|
| rank IC | 0.0058 (t 0.94) |
| hit rate / AUC | 50.0% / 0.501 |
| forecast error vs the zero forecast | worse (MSE skill −0.014) |
| signed return per forecast | +0.23 bp (t 1.44), against a 14 bp round trip |
| against the coin-flip null (3 seeds) | z 0.94: **no edge** |

- **Calibration** is flat: no forecast decile earns its forecast.
- **Machinery:** in the tests a planted signal is learned (IC > 0.5) and noise gives nothing (|IC| < 0.12). This
  input set simply has no next-5-minute direction.
- **Input groups:** only the intraday group moves the static model's IC (by 0.010 of its 0.011); daily, market,
  options, flags and the target's own lags are noise.

**Retraining** (the same bars, IC by cadence): once 0.011, every 20 sessions 0.012, weekly 0.005, daily 0.006. No
schedule differs from training once beyond noise (largest paired |t| 1.19): with these inputs, the "regime" a
fresh fit would catch is not there to catch.

**The trading rule on TUNE** (`darts_forecast`, from 2025-09-16):
- The shipped rule loses **−26%/yr** at 1,243 trades/yr. Its 463 trades come from 439 holdings, the difference being
  re-entries after a stop.
  - The median stop is 0.62%.
  - The risk rule sets the size on 94% of trades, at a median 16% of the free cash.
- The study (`darts_forecast-gbm-5m-v3`, 60 trials, 38 pruned under the 300 trades/yr floor) found no rule with
  positive alpha. The best loses −5.7%/yr at 389 trades/yr. Alpha rises toward zero only as trading falls: the
  cheapest way to trade noise, not an edge.

**TEST** (2026-02 → 04, `darts_forecast_diagnostics`; a re-read after the core fixes, the first read was under the
old code):

| | value |
|---|---|
| forecast rank IC | −0.0127 (t −1.31), z −2.05 against the null |
| shipped rule | alpha −13.5%/yr over 106 trades (buy-and-hold +49.5%) |
| 2× costs / pessimistic fills / no costs | −19.8% / −26.8% / −6.7% |
| appending TEST changes anything in TUNE | no: 0 of 37,045 forecasts differ |
| verdict | **FAIL**, 0 of 4 gates; HOLDOUT unread |

This agrees with the direction study. The setup is ready for inputs that might carry direction: order book or
quotes, earnings, or a cross-sectional target (`IDEAS.md`).

## Setup and caveats

- **Install:** `uv sync --extra darts` (or `--all-extras`) installs Darts + torch (PyPI package `darts`, 0.48;
  the old name `u8darts` stopped at 0.41). Darts 0.48 no longer caps scikit-learn, so the repo keeps
  scikit-learn 1.9 without an override. `tests/test_darts_forecast.py` runs all six models' walk-forward on it.
- **Option inputs** come from `panel.features['opt_*']`, which `research.attach_options` fills from the free
  options store. The notebooks attach them. A headless `tsim optimize` has no options: the inputs become 0 with
  their flags off and `forecast()` warns, until a `features.py` builder reads the store (an engine proposal in
  `IDEAS.md`). So run the study from `darts_forecast.ipynb`.
- **Forecasts are cached** in-process (an Optuna study pays for the walk-forward once) and on disk under `cache/`
  (git-ignored). The disk key covers every forecast knob, the panel's full content and the `strategy.py` version,
  so a changed bar, input or line of code is a new walk-forward, never a stale one. `research.prune_cache()` deletes
  the forecasts of older `strategy.py` versions.
- **Several walk-forwards at once:** gradient boosting uses every core per fit, so parallel jobs oversubscribe the
  machine and each runs several times slower. Give each job `OMP_NUM_THREADS=2`.
