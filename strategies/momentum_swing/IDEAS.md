# momentum_swing — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why). Give each idea a *why* and, where possible,
the falsification that would settle it.

## Engine proposals (proposal-only — `src/` changes need explicit approval)

### 1 · Per-entry cash fraction, for risk-per-trade sizing

**What.** Let `signals()` return `cash_frac` as a *frame* (bar × symbol, fraction of the
free cash to deploy on that entry) as well as a scalar. `backtest.run` currently does
`size=float(signals[CASH_FRAC_KEY])`; vectorbt's `size` broadcasts, so
`size=signals[CASH_FRAC_KEY]` (frame or float) is the whole change there, plus
`sweep()` stacking the per-combo frames the way it stacks `sl_stop`
(`stack([as_frame(s[CASH_FRAC_KEY], ...) for s in combo_signals])` instead of the
`np.repeat` of floats). Roughly ten lines, no new dependency.

**Why.** Position = risk ÷ stop distance is the one risk rule this whole family agrees on,
and it is the one intervention the literature says removes the left tail (Barroso &
Santa-Clara 2015; Daniel & Moskowitz 2016 dynamic strategy). This strategy would then
return `cash_frac = min(risk_frac / stop_frac, max_weight)` per entry — e.g. 1% risk on a
7.5% stop = 13% of the book, capped at 25%. Today the per-trade risk floats between
`cash_frac × 0.5 ATR` and `cash_frac × max_stop`.

**Falsification.** Re-run the v2 study under per-entry sizing; the claim is that
`tracking_error` and `max_drawdown` fall at equal `excess_cagr`. If the front does not
move, fixed-fraction sizing was fine and the proposal is closed.

### 2 · Parallel optimizer workers duplicate trials under a fixed seed

**What.** `jobs.run_optimize(workers=N)` (and N hand-launched `tsim optimize` processes)
each build `make_sampler` with the same `optuna.seed`, so every worker proposes the same
parameter sequence: the v1 study's 450 trials from 3 workers held **300 duplicate
parameter sets** — 150 unique trials for the price of 450. The journal makes the appends
safe; it does not make the samplers different.

**Fix.** Derive the per-process seed from the config seed plus a worker discriminator
(`seed + os.getpid()`, or a `--seed` CLI option that `jobs.run_optimize` sets to
`seed + i`), and say so in the `optuna:` template comment. Until then: **one worker per
study**, or accept the duplicates.

### 3 · An earnings-calendar feature (`earnings_dates`)

Already proposed from `gap_and_go`. It matters more here: the pivot leg pools earnings
gaps (which drift — Brandt et al.) with news/sympathy gaps (which fade — Berkman et al.
2012), and the v1 event study on Liquid500 shows the pooled population carrying nothing
past ten days. One builder in `features.py` (yfinance carries the dates) would allow
`pivot_catalyst: earnings | any` and split the pooled result into its parts.

## Strategy ideas (no engine change)

### 4 · The parabolic short — the spec, kept out on purpose

`short_entries` = close ÷ close 3–5 bars ago ≥ `para_min` (0.5 large caps) AND ≥ 3
consecutive up closes AND the first close below the prior close (first red day);
stop = the run's high (`sl_stop` as a fraction above entry); cover at the first close
below the 10 SMA touch or after `para_hold` bars; run in the same wallet
(`upon_opposite_entry="close"` handles a long/short clash). **Do not ship it without a
borrow-cost stress**: subtract a daily borrow accrual of 25–100%/yr from every short
trade's P&L in the notebook (from `trade_records`, hold days × rate) and require the
edge to survive at 50%/yr. On Liquid500 the setup will rarely fire (names doing $90M+/day
do not go +300% in three days); the population it is meant for is not in the DB.

### 5 · Intraday opening-range entry (1-minute data)

The source enters above the opening-range high and stops at its low; the daily-bar
version fills at the next open and pays the gap. The DB holds 1m bars from 2025-06, so a
`momentum_swing_1m` variant could confirm the daily setup at day t's close and enter at
the ORH break on t+1 (`bar: 1m`, the daily filters computed from resampled bars), stop at
the ORL. Zarattini, Barbon & Aziz (2024) report the intraday version's best numbers *and*
its weakest validation. Falsification: the same setups, ORH entry vs next-open entry,
over the overlapping 2025-06+ window — the claim is that the ORH entry raises the
per-trade expectancy by more than the slippage it adds.

### 6 · A second point-in-time universe for 2015–2020

`Liquid500` is ranked on 2020-H1 dollar volume, so it is honest only from 2020-07. The DB
has ~480 of its names since 2018 and ~460 since 2015, so a `Liquid2015` ranked on
2014-H2 (screener §8b with the dates moved) would double the history and add the 2015-16
chop, the 2018 Q4 break and the 2020 crash — three regimes the current window lacks.
Falsification: the v2 pick's sign of `excess_cagr` on 2015–2020.

### 7 · Partial exits

"Sell a third at 3–5R, trail the rest" is not expressible in one `from_signals` column.
It can be approximated by running two half-size wallets with different exits (one with
`tp_r`, one without) and summing the curves in the notebook; the engine would call that
two strategies. Stonks Capital's rebuild reports the partial sell *hurt*; the v1 study's
best trials chose `tp_r` 7–12 (a distant whole-position target), which points the same
way. Low priority.

### 8 · Base "higher lows" test

The source's base has "higher lows"; the shipped tightness test is a range ratio plus a
depth cap. A slope-of-lows test (`low.rolling(base_window)` regression slope ≥ 0) is one
line; the literature has no directional evidence for any base shape, so this is a
parameter to *ablate*, not to add by default.

### 9 · A per-day signal cap (`max_new_per_day`)

Wallet mode has no position count (README §2i): each entry takes `cash_frac` of the
*remaining* cash, so the v2 pick — a 3%-gap scanner firing several times a day — held up
to 82 positions at once, the median one $2k on a $50k book. A stateless mitigation the
strategy can do itself: rank a day's signals (by leader return, or by gap size) and keep
the top `max_new_per_day`; the engine's `entries` frame then carries at most N True cells
per row. It caps *entries* per day, not concurrent positions — a true concurrency cap
needs the engine to know what is held (an engine feature, not a strategy one).
Falsification: the v2 space with `max_new_per_day` ∈ {1, 2, 4}; the claim is that the
front's `n_trades` and time-in-market fall while `excess_cagr` does not.

## Open questions

- **Why does the optimiser prefer `regime: none`?** In v1 every one of the six best scored
  trials had no gate. The benchmark-relative objective charges time out of the market
  against a +15%/yr equal-weight hurdle, and the gate's job is to cut drawdown (which the
  objectives do not score directly). Read `max_drawdown` and `tracking_error` on the gated
  vs ungated front rows before concluding the gate is useless; the literature says its
  value is in the tail, not the mean.
- **Is the shortfall time-in-market or selection?** The v1 event study says the strict
  breakout event carries +3% market-relative over 10 days (22 events, t 2.0) and the
  relaxed one +0.3–0.8% (762 events, t ~1.2); the pivot carries nothing past ten days.
  Selection is weakly positive; the ~40% time in market against a fully invested hurdle
  is the larger term. A `Liquid2015` run through a bear market is the way to tell.
