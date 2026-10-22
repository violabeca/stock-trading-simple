# burst_scan — activity bursts on minute bars; a long and a short sub-strategy, each learned

**Status: documented negative after phase 1 (2026-09-27) — both sub-strategies FAIL; see "Results".**
Bursts reliably precede *big* moves, but neither the detectors nor the classifiers find *which way*. Nothing was
searched, nothing ships, the holdout is unread. The config is kept as the record of what was tested; the five
notebooks re-run the whole experiment. The pre-registered gates below were fixed before the full-universe run.

## The idea

A **burst** is a sudden change in how a stock trades: over the last `burst_minutes` its volume and/or its high–low
range far exceed what that name *normally* does **at that time of day** (the same-clock median of the previous
`profile_days` sessions, `bt.volume_profile` — so the routinely busy open never fires by itself). A burst says
*that* something is happening; it does not say *which way* it will go. So the strategy is **one strategy with two
sub-strategies**, trained, validated and judged separately:

* **long** — its own detector (`long_trigger`, `long_impulse`, `long_vol_mult`, `long_range_mult`) flags events;
  its own classifier, trained walk-forward on what happened after past events, predicts which of them will reach a
  target *upward* before a stop of the same size; it buys only its most confident picks.
* **short** — the mirror, with its own `short_*` knobs, its own labels and its own model. The two never see each
  other's labels: one can pass while the other fails, and `sides` then drops the one that failed.

`impulse` lets each side learn continuation *or* fade: a long detector with `impulse: down` buys bursts that
pushed the price DOWN.

## Which notebook answers what (run them in this order)

| notebook | question | reads | ends with |
|---|---|---|---|
| `burst_scan_visual.ipynb` | **Does the code do what this README says?** One name, ~8 sessions: what the detectors saw, what each side decided and why, event cards (entry, target, stop, exit), the engine's fills over the labelled trades. No statistics. | a 40-name panel (the model needs a universe) | 5 engineering checks |
| `burst_scan_events.ipynb` | **Do the detectors find real moves? (G0)** Anatomy, the two placebos, win/stop/time-out rates over 19 hold × bar cells, reliability by burst size, the overnight share, the pre-registered detector grid and its pick. | the 1m panel and its 5m/10m derivatives | the G0 verdict per side |
| `burst_scan_model.ipynb` | **Does the classifier add direction? (a–f)** Folds and a purge audit, out-of-sample AUC (within-month), calibration, returns by decile, what the model learned, the controls, monthly stability. | the primary-bar panel | the a–f verdict per side |
| `burst_scan.ipynb` | **Optimize.** Phase 1: one trial of the shipped rules in the journal, the candidates file, the **baseline comparison** (long, short, both vs buy-and-hold), threshold sensitivity. | the primary-bar panel | the book's numbers |
| `burst_scan_diagnostics.ipynb` | **Is the machinery right?** Label ↔ engine reconciliation, look-ahead and purge audits, cost anatomy and stresses, book anatomy, concentration, runtime, and the one-shot holdout (off by default). | the primary-bar panel | 6 engineering checks |

Every notebook is self-contained (it loads its own panel; `SUBSET = 40` at the top makes a quick first run),
passes `{**strat.params, **OVERRIDES}` to the strategy, ends with a `CHECKS` table and one `SCORECARD {json}` line.
Shared computations live in `research.py`, loaded by path; `strategy.py` never imports it.

Three research notebooks sit beside the chain (sections at the end): `move_study.ipynb` (events defined by their
outcome, 200 names), `move_explorer.ipynb` (whole runs and live confirmations on a few hand-picked names) and
`move_extended.ipynb` (the same with the pre- and post-market).

## The mechanics

1. **Context** (`context()`, once per panel). The burst window's volume ratio and range ratio against their
   same-clock baselines; the window's move (its sign is the burst's direction); the session ATR (Wilder over
   `atr_days` sessions, shifted one session — the one distance unit); VWAP, the session's range so far, the gap,
   the prior day's high/low, a pre-burst trend, the 5-session return, the equal-weight market's move and the
   signed breadth of bursts across the universe, the clock, earlier bursts today, ATR/price and a dollar-volume
   rank over prior sessions. Twenty features (`FEATURES`), all known at the event bar's close.
2. **Detector** (`detect()`, per side). A raw burst clears the side's thresholds (`trigger` = volume, range, both
   or either) with the required move sign (`impulse`). An **event** is the first raw burst after `cooldown` quiet
   minutes, outside the first `skip_open` / last `skip_close` minutes, after `warmup_sessions`, above both floors.
   Shorts skip names under Reg SHO Rule 201 (down 10% from the prior close, that session and the next).
3. **Outcome = the trade** (`outcomes()`). Entry at the next bar's open. Target and stop sit `move_atr` session ATRs
   × √(horizon / 390 minutes) from the **fill bar's close**, checked from the bar after the fill — exactly how
   vectorbt books `sl_stop` / `tp_stop` (`stop_entry_price="close"`; a bar touching both is a stop; a gap through a
   barrier fills at the open). A **win** is the target first. No barrier by `horizon` minutes of trading time → the
   time exit at that bar's open (or at the session's last bar under `eod_exit`). Positions may be held across
   sessions. The random-walk scaling makes one `move_atr` mean the same for a 30-minute and a 5-day hold.
4. **Classifier** (`walk_forward()`, per side). Calendar-month folds, expanding window. Month *m* is scored by a
   model fit ONLY on events whose outcome finished before *m*'s first bar — **the purge**: a multi-day label that
   runs into the month never trains it. Logistic regression (median-imputed, standardized) is the default;
   histogram gradient boosting is the challenger. An event is picked when its probability reaches the
   `1 − trade_q` quantile of the fold's own training probabilities, so no threshold is fit on the month it trades.
5. **Book.** Picks enter at the next bar with the barrier as `sl_stop` and `tp_stop` (per-entry frames on the fill
   bar), `exit_after(horizon)` and, under `eod_exit`, `session_end`. A side's picks on one name are spaced more
   than a horizon apart (`spaced()`), so a position is never re-entered while its time exit is pending. A bar where
   both sides pick the same name trades neither (the clash rule). One shared $10k wallet, `1/max_positions` of the
   free cash per entry; costs at the real order size (`order_notional: 1000` — the $1 minimum commission is 10 bp a
   leg, ~24 bp a round trip).

**The engine's opposite-entry rule.** In the shared wallet an opposite-side signal on a held name *closes* that
position instead of opening its own (`upon_opposite_entry="close"`; percent sizing cannot reverse). The labels do
not know this, so those trades differ from their labels; the diagnostics notebook names, counts and prices them.

## Pre-registration (fixed before the first full-universe run)

Applied to **each sub-strategy separately**, on the training window (to 2026-04-30).

**Primary cell:** 5-minute bars, a one-session hold (390 minutes of trading time, through the close), logistic
regression, scored on the out-of-sample months 2025-10 .. 2026-04. Chosen in advance because longer holds amortize
the ~24 bp round trip, holds past the close are allowed, and the high-volume return premium (Gervais, Kaniel &
Mingelgrin 2001) is a multi-day effect.

**Grid:** 19 cells — 1m × {30, 60, 120 minutes, to the close, 1 session}; 5m and 10m × {30, 60, 120, close, 1, 2,
5 sessions}. A non-primary cell needs t ≥ 3.2 (Bonferroni over 19 cells × 2 models).

**G0 — the detector alone** (`burst_scan_events.ipynb` §8), at the primary cell:
* the detector setting is picked from `research.GRID` (trigger × impulse × volume threshold × range threshold) by
  one rule: the highest win-rate lift over placebo A among settings with ≥ 1 event per name per 5 sessions;
* **placebo A**: a random bar of the same name, month and half hour with no burst within `cooldown`;
  **placebo B**: a bar with the same-size move (same sign, same size decile, same month) on normal volume;
* PASS = win rate ≥ 1.25 × placebo A's (day-clustered t ≥ 3) **and** above placebo B's (t ≥ 2) **and** wins beat
  stops on the same events (t ≥ 2) **and** a positive market-demeaned return. A side that fails stops here.

**a–f — the classifier** (`burst_scan_model.ipynb` §8), on the out-of-sample picks:
* **a** the picks' signed, market-demeaned return has a day-clustered t ≥ 3 and is net positive at 2× costs;
* **b** it beats "take every event" and a model trained on placebo-A bars, paired by day, t ≥ 2 each;
* **c** ≥ 70% of the out-of-sample months are positive after one round trip;
* **d** it stays positive without the five names that contributed most;
* **e** every event's outcome permuted across the sample, the model re-trained on the permuted outcomes and
  scored against THEM: within-month AUC within 0.5 ± max(0.01, 3 null SE) and its picks no better than the
  average permuted event (paired |t| < 2) — otherwise the run is **VOID** (something leaks between training and
  test). *Correction made AFTER the first full-universe run, which printed VOID on the long side:* that
  implementation scored the shuffled model against the TRUE labels, where a model fit to noise still inherits a
  random share of the real volatility structure through the features (within-month AUC 0.519, no leak) — a
  control that tests nothing. The long side fails gates a–d either way, so the fix cannot turn it into a PASS;
  the criterion itself is unchanged;
* **f** gradient boosting replaces logistic regression only if it beats it, paired by day, t ≥ 2.

**Outcomes.** FAIL → that side is written up in IDEAS.md and dropped from `sides`; no search opens for it.
PASS → its cell is pinned, the study bumped, and phase 2 opens (a small search, risk-per-trade sizing). The
holdout (2026-05-01 .. 2026-06-26) is read **once** per side, after the pin, in the diagnostics notebook §7.

## Things to know before reading any number

* **A burst is volatility first.** Bigger bursts mean bigger moves in BOTH directions (`burst_scan_events.ipynb`
  §5): on the same events the long side's win rate and the short side's stop rate are the same number. Any
  direction has to come from the classifier — which is why G0 alone is not enough.
* **These names earned their money overnight** (+11.6 bp/day) and lost intraday (−8 bp/day) in this window
  (`trend_following_1m`). A long book held through the close collects that drift whatever it selects; every
  notebook splits returns into overnight and intraday parts and compares against "take every event".
* **Pooled AUC is confounded by base-rate drift**: a model that only learns which months win more often scores
  above 0.5 pooled. The gates read the **within-month** AUC.
* **Shorts** need a locate, may be squeezed, and pay a borrow fee on every night held — which the engine does not
  model (the diagnostics notebook stresses 3%/yr). Rule 201 is honoured.
* **Survivorship**: Morning200's ranking is point-in-time, its membership (the 1m backfill list) is not.
* **The baseline** is equal-weight buy-and-hold of the same names over the same window; the objective is
  `alpha_cagr` because a book that is flat part of the time cannot out-earn a fully invested +47%/yr basket.

## Results — phase 1, 2026-09-27 (200 names, training window; holdout NOT read)

**Verdict: both sub-strategies FAIL.** Bursts are followed by *significant* moves far more often than ordinary
bars — but not in a predictable *direction*, and the classifiers only learned which events will move. Per the
pre-registration no search opens; the holdout stays unread.

### G0 — the detectors (`burst_scan_events.ipynb`, primary cell: 5m, one-session hold)

| side · detector | events | win | placebo A | lift A (t) | placebo B | t B | stop | t win−stop | demeaned bp (t) | G0 |
|---|---|---|---|---|---|---|---|---|---|---|
| long · config (both/any/3/2) | 22,316 | 18.0% | 11.2% | 1.60 (7.2) | 15.1% | 3.1 | 17.0% | 0.6 | −1.9 (−0.7) | FAIL |
| long · grid pick (both/up/5/2) | 8,177 | 21.4% | 10.3% | 2.07 (10.1) | 15.8% | 4.8 | 19.4% | 1.0 | −0.1 (0.0) | FAIL |
| short · config (both/any/3/2) | 21,514 | 17.2% | 11.4% | 1.51 (5.7) | 13.6% | 3.7 | 17.9% | −0.5 | +4.2 (1.5) | FAIL |
| short · grid pick (both/down/2/3) | 8,915 | 19.9% | 11.3% | 1.76 (7.2) | 15.4% | 4.0 | 21.4% | −0.8 | −1.3 (−0.3) | FAIL |

* **Movement: yes, strongly.** After a burst the target *or* the stop is hit ~1.6× as often as after an ordinary
  bar of the same name, month and half hour (t 5.7–10.1), and more often than after a same-size price move on
  normal volume (t 3.1–4.8) — the volume adds information beyond the move itself.
* **Direction: no.** Wins do not beat stops (t −0.8 … +1.0) and the market-demeaned return is ≈ 0. §5's ladder
  says it cleanly: from ×1.5 to ×8 relative volume the win rate rises 12.8% → 18.2% **and the stop rate 11.9% →
  18.3%** (range ×1.5 → ×6: 13.0% → 24.4% and 12.2% → 23.8%), against a flat ~11% for placebo A. On the same
  events the long side's win rate *is* the short side's stop rate.
* The primary cell's long return (+3.6 bp) is overnight gap (+8.4 bp) minus the session (−4.8 bp).

### a–f — the classifiers (`burst_scan_model.ipynb`, out-of-sample 2025-10 .. 2026-04)

| side | within-month AUC (lr / gbm) | picks | gross bp (t) | vs every event (t) | vs placebo-trained (t) | months net > 0 | verdict |
|---|---|---|---|---|---|---|---|
| long | 0.536 / 0.536 | 5,649 | −3.5 (−0.5) | +1.2 bp (0.1) | +2.2 bp (0.3) | 1/7 | FAIL |
| short | 0.561 / 0.556 | 4,396 | +11.1 (1.6) | +12.8 bp (1.2) | +10.9 bp (1.1) | 1/7 | FAIL |

* The win rate rises steadily across the prediction's deciles (≈15% → 22%) but the returns do not: the models
  learned **which events will move**, not which way. Fig 5 shows it directly — the long and the short model's
  coefficients lie along the diagonal (the same features push both sides up), not the anti-diagonal a direction
  model would draw.
* Leak and pipeline controls all pass: purge audit (flipped running labels change nothing), shuffled-outcome
  control (within-month AUC 0.504 / 0.502 against the permuted outcomes, picks paired t −1.4 / −1.9), the
  label-as-feature canary finds its label (AUC 1.0); gradient boosting does not beat logistic regression (f).

### The book (`burst_scan.ipynb`, warm from 2025-10-01; buy-and-hold of the same names: +18.5%/yr)

| book | CAGR | alpha | beta | trades/yr |
|---|---|---|---|---|
| long | −31.1% | −44.2% | 0.71 | 6,947 |
| short | −36.2% | −28.7% | −0.41 | 5,112 |
| both | −36.1% | −43.0% | 0.38 | 5,703 |

With no selection edge the book is its turnover times the ~24 bp round trip. `max_positions` only sizes a slot
(1/10 of the FREE cash), it caps nothing, so the long side alone ends up a nearly fully invested, high-turnover
long book (beta 0.71). Raising the detector thresholds only lowers the loss by trading less (§6). Scored on the
engine as fixed on 2026-09-27 (study `burst_scan-5m-v2`; see below).

### What the machinery audit found (`burst_scan_diagnostics.ipynb`)

The labels reconcile with the engine trade by trade (max price difference 0), no feature, event or prediction
changes when later data is removed, and the purge holds. The reconciliation named one engine rule the labels do
not model — an opposite-side signal on a held name closes it instead of opening its own (≈ 375 trades each way,
`research.ENGINE_RULES`) — and **found an engine flaw, now fixed**:

* **Short proceeds were spendable** (vectorbt's `lock_cash=False`). A short sale's cash could fund a later long, so
  the wallet ran leveraged (gross up to ~150% in this book; 280% in the engine's regression test), free cash went
  negative on 11% of bars, 199 entries were silently skipped, and the buy that covers a short was cash-limited — a
  stop covered only part of the position and the rest ran past it.
* **Fixed 2026-09-27** (user-approved): the engine now passes `lock_cash=True` in the wallet path (`backtest.run`
  and `sweep`), with a regression test; studies of every strategy that can short in the wallet were bumped. On this
  book: 0 skipped entries, 0 split covers, and the "both" book's beta fell from 0.71 to 0.38 and its CAGR from
  −29.7% to −36.1% — the old engine flattered it. Long-only and short-only books are unchanged. The verdict is
  unaffected (the gates read the labels, not the engine's P&L).

### Where this leaves the idea (see IDEAS.md)

Bursts are a **volatility** signal: they say a big move is coming, not which way. That is useful for sizing,
stop placement or options (a straddle wants exactly this), none of which this equity engine trades. A direction
model would have to be trained on direction alone — outcomes that DID move, win against stop — rather than on a
label that is mostly "did it move"; that, and a direction-aware G0 pick rule, are the two re-registrations worth
making before anything else is tried.

## Move study (2026-09-28): events defined by their outcome — `move_study.ipynb`

**The question, turned around.** Instead of asking what happens after a burst, define the event by what a trade
needs — **a big move within 4 hours** — check every name every 30 minutes, and look for metrics that predict it,
first *whether* it comes, then *which way*. Research only: a strategy is proposed separately if stage 2 passes.

**Why the threshold is scaled.** On these names a fixed ±2% within 4h follows 73% of all moments (a pre-study
count on 60 names: 32% for the calmest fifth, 94% for the wildest), so a fixed 2% mostly measures each stock's
volatility. The event barrier is `k` × the session ATR × √(240/390); fixed 2% runs alongside as a variant.

### Pre-registration (written before the full-universe run)

* **Points:** every 30 minutes (every 6th 5-minute bar) inside the tradable window; training window to
  2026-04-30; out-of-sample months 2025-10 .. 2026-04 (the walk-forward's first predicted month onward).
* **Outcome:** `strategy.outcomes()` with symmetric barriers and a 4-hour horizon (48 bars, may run past the
  close), booked as the engine would book a trade: up first / down first / neither.
* **`k`:** the smallest value on the grid 0.5, 0.75, … 4.0 whose barrier is reached by at most **20%** of the
  decision points before 2025-10-01 (`research.calibrate_k`), fixed before any model is fit.
* **Stage 1 — occurrence (context, no verdict):** the within-month AUC of each of the 22 features for "either
  barrier reached"; the walk-forward model against a baseline that knows only the name's normal range and the time
  of day (paired within-month AUC by month, t ≥ 2 to say it adds anything beyond volatility).
* **Stage 2 — direction (the verdict), per side:** a direction model trained only on points that moved, applied to
  the points stage 1 puts in its top third. Long picks = top third of P(up first); short picks = top third of
  P(down first). PASS needs all of:
  * **a)** among picks that moved, hit rate > 50% with a day-clustered t ≥ 3, **and** mean P&L per pick (timeouts
    included, signed as positioned) positive after 2× the round-trip cost;
  * **b)** the picks beat all of stage 1's selected points and a random pick of the same size, paired by day,
    t ≥ 2 each;
  * **c)** ≥ 70% of months positive after one round trip of cost;
  * **d)** positive without the five names that contributed most;
  * **e)** outcomes shuffled across the sample, the model re-trained and scored against the shuffled outcomes:
    within-month AUC within 0.5 ± max(0.01, 3 null SE) and its picks no better than average (|t| < 2) — else VOID;
  * **f)** gradient boosting replaces logistic regression only if it beats it, paired by day, t ≥ 2.
* **The fixed-2% variant** is a second look at the same question: it needs t ≥ 3.2 on (a).
* **FAIL** → written up in IDEAS.md. **PASS** → a separate strategy is proposed. The holdout stays unread.

### Results (2026-09-28, 200 names, 470,707 decision points; holdout unread)

**Verdict: occurrence is predictable beyond volatility; direction is not. No side passes.**

* **Barrier:** the rule picked `k = 1.25` → median 4.1% of price in 4 hours (10th–90th pct 2.4–7.4%). With it
  the reach rate is flat across volatility (16–17% in every fifth) — the event means the same for every stock.
  Fixed 2% is reached by 25% (calmest fifth) … 88% (wildest) of points.
* **Stage 1 — occurrence:** the model's within-month AUC is **0.716** against **0.680** for the "normal range + time
  of day" baseline, better in every one of the 7 months (paired t 5.2). The strongest metrics alone: the time left
  to the close (a 4-hour window that crosses the night includes the overnight gap, AUC 0.68), earlier bursts
  today (0.66), the session's cumulative relative volume (0.61), the last 60 minutes' range and the burst volume
  (0.58). A busy day stays busy.
* **Stage 2 — direction:**

| variant · side | picks | within-month AUC | hit rate (moved) | hit t | gross bp | net at 2× cost | months > 0 | verdict |
|---|---|---|---|---|---|---|---|---|
| scaled · long | 49,061 | 0.525 | 50.8% | 0.2 | +1.6 | −46 bp | 1/7 | VOID* |
| scaled · short | 25,220 | 0.525 | 46.1% | −1.5 | −13.1 | −61 bp | 0/7 | FAIL |
| fixed 2% · long | 61,243 | 0.508 | 50.3% | 0.2 | +5.7 | −42 bp | 1/7 | FAIL |
| fixed 2% · short | 41,099 | 0.508 | 49.6% | −0.2 | −5.2 | −53 bp | 1/7 | FAIL |

  No single metric predicts direction either: the best within-month AUC is 0.549 (the name's normal range) and
  every rank correlation with the signed 4-hour return is below 0.022.
  \*VOID: the shuffled-outcome control's picks came out at t 2.24 (one seed; |t| < 2 required). The run is void by
  the pre-registered rule — and would fail anyway on a–d (hit t 0.2, 1 of 7 months).
* **Reading:** metrics built from price and volume can tell *when* a big move is coming, well beyond knowing which
  stocks are volatile, but not *which way*. Both studies agree.

## Move explorer (2026-09-28): whole runs, their onset, and the moment you could act — `move_explorer.ipynb`

A fast companion to the move study (three hard-coded names — GLW, ORCL, ALB — every knob at the top: `MOVE_PCT`,
`WINDOW_MIN`, `REVERSAL_PCT`, `CONFIRM_PCT`, `LOOKBACK_MIN`, `PRE_MIN`, `LEAD_MIN`, `PROFILE_DAYS`, …; ~10 s a run).

* **Runs, not 2% slices.** "Any 2% within 4 hours" finds the *last* 2% of a move: on these names the slice had
  already moved a median 1.2% from the real bottom (≥ 1% for 54%, ≥ 2% for 37%). The explorer traces swings
  instead (`research.zigzag`, `swing_runs`): a run is a swing of at least `MOVE_PCT` that ends only on a
  `REVERSAL_PCT` pullback, with its first `MOVE_PCT` inside `WINDOW_MIN` of its **onset** (the swing low / high).
* **Two moments per run.** The onset is known only afterwards — and a swing low is by definition the end of a
  decline, so direction features "before the onset" are partly built in. The **confirmation** — the first close
  `CONFIRM_PCT` off the recent low / high (`research.confirmations`) — is knowable live, and every confirmation is
  kept, including the ones that failed (came back through the low) or stalled: that is the honest test.
* **What it shows (training window, in-sample, three names; `BAR_MIN` 5 and 1 agree):**
  * Runs announce themselves only at the last moment. On 1-minute bars the volume and range burst sits in the
    onset minute itself (≈ 1.4× the clock-time normal) after a 2–3 minute ramp; before that runs trade at ≈ 0.9×
    normal — above ordinary moments (≈ 0.65×) but with no early build-up to identify.
  * Confirmations do not pay: 13–22% of them ran on to `MOVE_PCT`, just below the 27–32% needed at their
    reward ÷ risk, and the mean trade from the confirmation close is −6 … +2.5 bp *before* a 24 bp round trip,
    at both bar sizes. The features that separate "ran" from "failed" best are intraday volatility (AUC ≈ 0.65,
    the same sign on BOTH sides) and time left in the session — movement, not direction — and no fifth of any
    feature clears the cost line.
  * 1-minute bars find 5–14% more runs and confirm earlier (entries nearer the low), but more of those early
    confirmations fail; the expected trade is the same ≈ 0 before costs.
* **Correction (2026-09-29).** The first version of `confirmations` counted a "confirmation" whenever the lookback
  window's extreme CHANGED while price was already past the new level (an older low leaving the window) — 65% of
  5-minute and 78% of 1-minute candidates were such artifacts, with entries far from any low. The numbers first
  reported on 2026-09-28 (31–45% ran, −13 … −24 bp trades, "extended entries lose −50 to −70 bp", ran-vs-failed
  AUC ≈ 0.76 for VWAP distance) came from them and are withdrawn. A confirmation is now a real crossing of the
  level (tested). A second fix the same day: a confirming bar whose close is already at or past the target is dropped (the
  whole run happened inside that bar, and it booked a "run" exiting below its own entry — 9% of confirmations in
  the first half hour, < 1% later). The numbers above moved by less than 0.5 bp and stand.

## The extended hours (2026-09-29): do the pre- and post-market help at the open and the close? — `move_extended.ipynb`

The explorer sees regular hours only, so it is blind exactly where runs cluster: 39% of regular-hours runs start in
the first half hour (half of them on the very first bar), and a run that starts late is cut off at 16:00. This
notebook loads eight names that genuinely trade outside regular hours — MU, INTC, ORCL, HOOD, COIN, SOFI, IONQ,
SMCI: 68-85% of pre-market minutes from 07:00 traded, $24-176M a day pre-market — and traces runs and
confirmations on **the same prints twice**, 04:00-20:00 and 09:30-16:00 (5m bars, `MOVE_PCT` 2%, training window,
holdout unread; ~30 s a run). Any difference is the extended hours alone.

* **Two data traps, masked before anything else.**
  * IBKR placeholders (volume-0 copies of the previous close) are about a third of all extended minutes.
  * **08:00 New York carries stale prints.** 82% of 08:00 minutes stick out more than 25 bp beyond the prints
    either side (0.2% at 07:59; 44% at 08:01, 27% at 08:02, under 2% from 08:07). This is most likely overnight
    trades reported when FINRA's reporting facilities open, stamped at the report time with the overnight price.
    As stored, they produced a striking "pre-market reversal":
    * confirmations from 04:00 to 08:00 lost 9-33 bp each (t −5 to −13.5);
    * it held on all eight names and on both sides, and survived entering at the next print;
    * it **replicated on eight other names** (CRWV, NBIS, RKLB, RGTI, QBTS, SMR, MARA, IREN: −30 bp at 07:00-08:00,
      t −12.8).

    72% of the 07:00-08:00 "failures" were stopped out by an 08:00 print. With 08:00-08:09 masked
    (`research.mask_late_reports`), every pre-market bucket reads −1.5 … +5 bp on both sets. A second set of names
    cannot catch an artifact that every name shares; Fig 1b and §4b show it.
* **The open.**
  * **Where open runs start.** 22% of the regular view's open runs had begun in the pre-market, having done a
    median 0.5% by the regular onset. The rest start at or after the open (median 09:35).
  * **Which way.** At 9:30 (a race to ±2% within 60 minutes of the opening print, 1,624 name-days), the pre-market
    price *is* the gap (correlation 0.999), which regular hours already see at the open. Adding the rest of the
    pre-market to a regular-hours model (volume against its normal and against a session, range, closing trend,
    last evening's after-hours) makes the direction model slightly worse (within-month AUC 0.544 → 0.539).
  * **Whether it moves.** A borderline gain: 0.687 → 0.700, log-loss gain t 2.0. The share that moved on the model's
    top-third days goes from 83.4% to 84.1%. On the second set it is 0.681 → 0.691, t 0.7.
  * **Features and gaps.** No extended feature reaches |t| 2 on the first-hour return. The first hour in the gap's
    direction, by fifth of each feature, clears no cost line; the one lead, last evening's move toward the gap
    (t 1.6), is t 0.9 on the second set.
* **The close.** Runs starting in the last hour are 2% of regular-hours runs. 67% of them are still running at 16:00,
  and 39% peak after it, a median 0.7% further. Yet the trade from a late confirmation is worth the same either way:
  +0.4 bp cut at 16:00, +0.5 bp run on into the evening, before 24 bp of costs.
* **The evening.**
  * 466 after-hours runs (0.27 per name and day, median 2.9%). 31% of them fall on the 3% of evenings with at least
    3× the normal after-hours volume: news nights (earnings), which a price pattern cannot anticipate.
  * Confirmations after 16:00: −0.7 and −1.5 bp.
  * The second set's +11 bp at 16:00-17:00 (t 3.8) and −4 bp at the regular open (t −2.9) do not replicate on the
    first set; with about 20 bucket tests, they are noise until shown otherwise.
* **Tradability (Fig 1).** A traded pre-market minute moves tens of thousands of dollars, against millions in regular
  hours. Print-to-print returns autocorrelate at −0.1 to −0.3 (about −0.02 in regular hours), so trade prices there
  carry the spread, and the 24 bp regular-hours round trip is a floor.
* **Verdict.** The extended hours show where some open runs came from, but they do not help catch them: they add no
  direction, only a borderline occurrence gain, and no confirmation trade near either edge of the day pays before
  costs. The explorer's regular-hours conclusions stand.

## Onset identification on 1-minute bars (2026-09-30) — `ONSET_REPORT.md`

Can the existing 22 features (`research.explore_features`) identify where an up move or a down move starts, on
1-minute regular-hours bars, and what do the knobs change? A ten-round, multi-agent study on 120 Morning200 names
(four disjoint groups of 30, walk-forward, placebo and coin-flip-market nulls, pre-registered out-of-sample tests,
an independent audit). **Read `ONSET_REPORT.md`.** In short:

* The features find WHEN and WHERE price is about to move (AUC 0.86), but a market whose minute-by-minute
  direction is a coin flip lets them score 0.85 — it is volatility, name, day and clock.
* They carry no direction: up- vs down-starts 0.50 without the (hindsight) location features; at a new 30-minute
  low +1% comes first 48% of the time and the features predict it at 0.51. No knob, pocket, market state or
  signed-flow feature changes that.
* Nothing trades after the 24 bp round trip; every in-sample lead failed on untouched names.
* Useful: the size of the next move (R² 0.66, mostly recent volatility) — for stops and position size, not entries.

## Two move detectors (2026-09-30) — `move_detectors.ipynb`

Your picks and knobs at the top (`SYMBOLS` = MU, COIN, CSCO, AAL, ANET; `MOVE_PCT` 2% or `MOVE_ATR`; `WINDOW_MIN`
240; `MODEL`; `ALERT_Q`), ~2.5 min a run. Two **independent** walk-forward detectors on 1-minute regular-hours bars,
each trained only on its own events: **LONG** = price rises `MOVE` within the window (from the next minute's open,
never past the close), **SHORT** = it falls `MOVE`. 51 causal features from the current bar back several weeks
(`research.detector_features`: the 22 explore features + multi-scale returns, realized volatility, volume, 5/20-day
range position, trend slopes, EMA distance, signed flow, previous-day shape, the other names' returns, weekday).
Judged against a **coin-flip market** (`research.sign_surrogate`: every bar flipped up/down at random — same
volatility, no direction). Development window 2025-06-18 .. 2026-04-30; the months after it stay unread.

* **Both are statistically significant alarms — for any move, not their own.** Own-event AUC 0.75 / 0.73, but the
  opposite event scores the same (0.76 / 0.74); their top-5% alerts are followed by their own move 33% / 28% of the
  time (base 12% / 13%) — and by the opposite move 33% / 27%. Direction AUC 0.485 / 0.489 (coin-flip 0.50 / 0.53);
  specificity ≈ 0 (t 0.2). Every knob tried (1-3%, 60-390 min, 0.5 x ATR) gives the same.
* **They look at the same things.** The top three for both: the current 1-minute volatility, the stock's daily
  volatility and the time of day; everything else is ≥ 10x smaller (importance rank correlation 0.56). They differ
  only in the SIGN of price-location features (VWAP distance, gap, distance to recent highs/lows, session return):
  the LONG detector leans on "price below VWAP / after a fall", the SHORT on the mirror — a mean-reversion tilt that
  does not hold out of sample (a model trained on direction alone: AUC 0.48-0.51).
* **Trap met while building:** comparing minutes WITHIN the same day showed "below VWAP → +2% first" +20 pp (t 8).
  It was selection on the day's future path (the coin-flip market shows +8-10 pp); the causal version (price crossed
  VWAP) shows nothing (47.0% vs 46.3%). The notebook's specificity therefore compares alerts with all minutes, never
  with the same day's average.
* **Trading the alerts** (bracket: target `MOVE`, stop at the opposite `MOVE`): +2 / −3 bp before costs, −22 / −27 bp
  after. Nothing passed, so the holdout was not opened.
* **Correction (2026-10-01):** this notebook's direction AUC (a 2%/4h label, 51 features) cannot see even a planted
  35-40 bp edge. Its "no direction" is therefore not evidence. The direction study below rests on tests that can.

## Predicting direction (2026-10-01) — `DIRECTION_REPORT.md`

Can anything predict whether MU, COIN, CSCO, AAL or ANET make a significant move UP or DOWN? Two rounds of method
agents plus a Darts forecasting track, about 12,000 hypothesis cells. Each claim was judged against a market-preserving
coin-flip null, replicated on 487 other names, and confirmed once on locked months (2026-02..04) and 10 fresh names.
**Read `DIRECTION_REPORT.md`.** In short:

* **No.** Nothing predicts direction 1 minute to 2 days ahead at a tradable level, from the names' own 1-minute bars,
  497 other names, the pre/post-market or daily history. All 11 frozen rules failed out of sample, including +16 to
  +48 bp in-sample leaders.
* **The tests could have found an edge.** Planted edges of 2-5 bp per 30-minute trade are detected (t ≥ 3, 80% power).
* **What looked like direction was microstructure, drift or a few trend days.** Those are the 1-2 minute bounce
  (< 0.5 bp), a bull-window overnight premium (5× its 2015-2025 level, gone in Feb-Apr 2026), and sector continuation
  on ~10 trend days.
* **One relative effect was real but did not hold.** A stock's late-day move against the market partly reverses
  overnight (IC +0.039, t 6.7 on 487 names). On the locked months it halved (t 1.5), the tradable book lost 2 bp a
  leg, and in 2015-2025 the 4 long-history names show it only weakly and unstably.
* **Forecasting models don't help (Darts, ~970 tests):** classical (ARIMA/ETS/Theta …), global ML, 11 deep nets and
  the pretrained Chronos-2 / TimesFM 2.5 all forecast "no move", drift or volatility; none beats a coin flip on
  direction or pays 7 bp.
* **Direction needs information 1-minute OHLCV lacks:** quotes/order flow, an earnings calendar with surprises,
  options, or faster leading markets.


## Free options data (2026-10-02) — `options_free_data.ipynb`

The direction study's next information class, gathered from every **free** source, for the 5 study names (`SYMBOLS`
knob). Code: `options_free.py` (loaded by path, never by `strategy.py`); data: `options_data/` (git-ignored,
gzipped CSV; personal research use, no redistribution).

* **Live, ~15 min delayed, current chain only** — Cboe delayed quotes (bid/ask, last, volume, open interest, IV,
  Greeks), Yahoo via yfinance (no Greeks; placeholders outside US hours), Nasdaq (no IV), OCC open interest per
  strike. History exists only from the day collection starts: run §1 daily, ideally ~16:30 New York.
* **History, backfilled incrementally** — DoltHub `post-no-preference/options` end-of-day IV and Greeks (near-the-money
  subset, no volume/OI, some days missing; 2019 on) and its daily IV/HV levels; OCC daily volume per stock split by
  customer / firm / market maker and put/call (**rolling 2 years only**: keep collecting).
* **MarketData.app** (free account; `MARKETDATA_TOKEN` in the git-ignored `.env`) — full end-of-day chains with
  open interest and volume per contract, last 1 year only. Its volume summed per stock is exactly 0.50 x the OCC's
  (OCC counts both sides of every trade) on 1,242 stock-days; its bid/ask match DoltHub's at rank corr. 0.998.
* **Result (§6, §6b, up to 2026-04-30):** no options feature predicts next-day or next-week direction - IV,
  skew, term, IV-HV, OCC customer put/call (630 days, max |t| 1.6) nor put/call OI/volume, OI change and dealer
  gamma (144 days, max |t| 1.7).
* **Cross-checks** — every pair of sources on the same contract. First snapshot: Cboe and Nasdaq agree on 99-100% (the
  same feed); OCC open interest matches the vendors on ~70% of contracts (rank corr. 0.99: update timing).
* **Analysis (§6, runs once the history is complete)** — 30-day ATM IV, 25-delta skew, term structure, their changes,
  IV − HV and the OCC customer put/call and volume-surge features, z-scored per stock, against the next day's and
  week's open-to-close return (day-clustered rank IC). The window stops at 2026-04-30; the holdout stays unread.
