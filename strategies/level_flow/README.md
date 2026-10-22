# level_flow — multi-week price levels × today's activity, long/short, flat by the close

**Status: an experiment with pre-registered gates, not a claimed edge.** Nothing here ships
until `level_flow_diagnostics.ipynb` says PASS. Every one of this repo's five earlier 1-minute
strategies lost to equal-weight buy-and-hold of its own universe; this one is built so it either
finds something they did not, or fails fast and legibly.

## The thesis, in the order the evidence supports it

1. **Levels are *where* trends get interrupted, not *which way*.** Published support/resistance
   levels predicted intraday trend interruptions in FX at > 60% bounce frequency
   ([Osler 2000](https://www.newyorkfed.org/research/epr/00v06n2/0007osle.html)); on tick data the
   bounce probability at prior extrema rises with the number of prior bounces
   ([Garzarelli et al. 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC3967202/)); on US stocks the
   HSAR construction used here — swing highs/lows binned by percentage width, a bin with enough
   pivots is a level — predicts interruptions (support better than resistance) but its naive
   trading rules do not beat buy-and-hold
   ([Zapranis & Tsinaslanidis 2012](https://ideas.repec.org/a/taf/apfiec/v22y2012i19p1571-1585.html)).
   The mechanism has **two polarities** ([Osler 2003](https://onlinelibrary.wiley.com/doi/abs/10.1111/1540-6261.00588)):
   take-profit orders cluster *at* a level (trends reverse there — `polarity: bounce`), stop orders
   cluster just *beyond* it (a trend that gets through accelerates — `polarity: break`). The
   polarity is searched, and §5b of the optimize notebook decides it on the naked signal before any
   backtest is trusted.
2. **Geometry alone has no edge net of costs.** 7,846 intraday rules incl. support/resistance and
   channel breakouts on 5-minute SPDR data: none survive data-snooping correction
   ([Marshall, Cahan & Cahan 2008](https://www.sciencedirect.com/science/article/abs/pii/S0927539807000588));
   30-minute technical rules on the S&P decayed after 2000
   ([Schulmeister 2009](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1714980)); lag-1
   autocorrelation of SPY is entirely *magnitude* (bid-ask bounce), not direction
   ([arXiv 2606.29591](https://arxiv.org/abs/2606.29591)). What the literature does find is that
   intraday OHLCV signals only work under **elevated information arrival**
   ([arXiv 2512.12924](https://arxiv.org/abs/2512.12924)) — the "stocks in play" result: a 5-minute
   opening-range breakout on the top-20 names by relative volume, long and short, ATR stops, ~1%
   risk per trade, flat at the close, earned a 2.4–2.8 Sharpe with beta ≈ 0 on 7,000 US stocks
   2016–23 ([Zarattini, Barbon & Aziz 2024](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729284)).
   **Selection on activity is the load-bearing ingredient here**, not the level.
3. **Risk is thesis invalidation, bounded, and never overnight.** Stops add value under momentum
   and regime switching and subtract it under a random walk
   ([Kaminski & Lo 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338)), so a stop
   inside a reversion entry's noise band sells the bounce it just bought: the stop sits `sl_atr`
   session-ATRs *beyond* the level, on the far side from the entry, so it is outside the touch
   band by construction (no combination has to be rejected). Fixed-fractional
   risk per trade is what makes a low-win-rate / high-R book survivable (Zarattini). Shorts obey
   Reg SHO Rule 201 and are never held overnight (no borrow accrues on a same-day round trip).
4. **Multiple timeframes, as an ablatable gate.** A higher-timeframe trend is a regime gate at the
   scale of the trade; the coarser scales are where the little structure there is tends to live
   (Schulmeister; [Heston, Korajczyk & Sadka 2010](https://arxiv.org/abs/1005.3535)). There is no
   peer-reviewed evidence that multi-timeframe *confluence* adds net-of-cost value, and this repo
   measured long trend filters regime-fitting the tape (`keltner_reversion`), so the trend gate ships
   off (`trend_tf: 0` is its one switch), its lengths are capped, and the walk-forward
   decides whether it earns its place.
5. **Re-tuning is a procedure, not a story.** Walk-forward results are window-length dependent;
   the probability of backtest overfitting approaches 1 as the number of tried configurations
   grows ([Bailey, Borwein, López de Prado & Zhu 2017](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253));
   plateaus beat peaks and parameter ensembles beat a single optimum out of sample. The
   diagnostics notebook re-optimizes only the regime-sensitive knobs on rolling 4-month windows
   and compares that against parameters frozen from the first fold and against the shipped
   defaults; if re-tuning does not beat the frozen control, the strategy ships frozen.

Base rate to keep in mind: fewer than 1% of day traders are persistently profitable net of costs
(Barber, Lee, Liu & Odean; Chague, De-Losso & Giovannetti). Assume any green backtest is overfit
until the walk-forward and the holdout say otherwise.

## How a trade happens

1. **Levels.** Each session's level table is built from the *previous* `level_days` sessions:
   swing highs/lows of `tf_levels`-minute candles (a candle whose high/low dominates the
   `pivot_k` candles on each side, confirmed `pivot_k` candles later), binned in log price by
   `bin_pct`; a bin with `min_touches` or more pivots is a level whose price is the mean of its
   pivots. A bar reads the nearest level strictly below its close (support) and strictly above it
   (resistance). Fully vectorized (`_hsar_levels`, ~1 s for 200 symbols × 100k bars).
2. **The scan.** At `scan_minute`, names that clear `min_price`, `min_dollar_volume` and
   `vol_mult` × their time-of-day volume expectation, have a level within `reach_atr` session-ATRs
   of the price, and pass the market `regime` gate are ranked by `rank_by` (relative volume by
   default — the ranking *is* the activity selection; the `vol_mult` floor only binds when fewer
   than `max_candidates` names clear it). The top `max_candidates` are the session's watchlist;
   nobody else trades. A session with no candidate is a flat day — the intended default outcome.
   On the real panel the defaults seat ~4.5 names a session and trade ~3 times a session.
3. **The trigger.** *Bounce*: the low (high) touched the level's `touch_tol` band within
   `touch_minutes` and the close is back outside it, optionally confirmed by `confirm_minutes` of
   momentum. *Break*: the close is `break_tol` beyond the level in force at the previous read, and
   the session's first close was on the inside (a gap over a level is not a break). One trigger per
   session, symbol and side; read on `signal_tf`-minute candle closes (0 = every bar); optionally
   only *with* or *against* the `trend_tf` EMA trend of completed candles; optionally only on the
   `vwap_gate` side of the session VWAP; never a short in a name down 10% from its prior close.
4. **The trade.** Fill at the next bar's open. Stop `sl_atr` session-ATRs beyond the level (a setup
   whose stop would exceed `max_stop` of price is *skipped*), target `tp_ratio` × the stop, time
   stop `horizon` minutes, **flat at the session's last bar, always**. Size =
   `min(1/max_candidates, risk_per_trade / stop)` of the free cash — one stop-out costs at most
   `risk_per_trade` of the book, gaps aside.

All distances are fractions of the **session-scale ATR** (Wilder over `atr_days` sessions of
session true range, shifted one session), never the 1-minute ATR, so the tolerances mean the same
thing on any execution bar. All time knobs are in **minutes**; `data.resample: 5` runs the identical
code on derived 5-minute bars (each execution bar is its own study).

## Capital, shorting and what the numbers cannot see

- One shared $10k wallet (repo rule). Percent sizing is of the *free* cash, so five same-day
  entries at a 1/5 slot deploy 20%, 16%, 12.8%, … of the book (≈ 67% total), never 100%.
- The global cost model (2 bps spread, 1 bp slippage) is a midday large-cap number; the
  diagnostics notebook's pass bar is 10/3 bps and every pick is also re-run under `fill:
  pessimistic`. Wallet child orders (~$2k) pay the $1 minimum commission the model understates.
- Shorts assume borrow is available; hard-to-borrow names and locate failures are not modelled.
  The SSR guard is modelled; the uptick-only *execution* after a trigger is not.
- The 1m history starts 2025-06-18 and the pinned `warmup_sessions` (20) covers every session-level
  window; the study scores warm from `optuna.score_from` so the flat warm-up is not charged against
  an invested benchmark. `level_days` is capped at 20 for the same reason (IDEAS.md).
- The objective is `alpha_cagr` (return after the beta actually run) against `volatility`;
  `excess_cagr` against the +47%/yr equal-weight basket is reported everywhere and will be
  negative — a mostly-flat book cannot out-earn a fully invested one, and is not meant to.

## Read the result honestly (pre-registered)

- **§5b before §9.** The event study measures the naked trigger — forward open-to-open returns,
  cross-sectionally demeaned, signed as positioned — by polarity, side, touch count, relative
  volume and higher-timeframe trend state, with fill delays of 0/2/5 bars. If neither polarity
  shows |t| > 2 at 60 and 120 bars, the backtest cannot rescue it; write it up in IDEAS.md.
- **Controls that must lose:** `rank_by: touches` with `vol_mult: 0` (no activity selection —
  with the default `rank_by: rvol` the volume floor is redundant, the ranking *is* the
  selection), `min_touches: 1` (any swing is a level), `vwap_gate: none` and `trend_gate: none`
  against the picked gates, `regime: none`.
- **Pass bar (diagnostics notebook):** holdout `alpha_cagr > 0` **and** the walk-forward refit
  positive in ≥ 3 of 4 folds with a positive mean **and** the holdout positive at 10/3 bps — all
  hard. Pessimistic fills are reported. The refit's stitched out-of-sample path must beat the
  frozen control's, or the strategy ships frozen. Anything less ships as a finding, not a config.

## First results (2026-09-15, train panel 2025-06-18 → 2026-04-30, scored from 2025-07-17)

The optimize notebook was executed through §5b on the real 200-name panel before any study ran.
Shipped defaults: alpha −5.6%/yr, beta 0.00, 622 trades (794/yr), 40% of names profitable —
untuned, as expected. The **event study** (market-demeaned open-to-open return from the fill,
signed as positioned, bps) is the finding:

| trigger, uncapped | events | 30 bars | 60 bars | 120 bars | 180 bars |
|---|---|---|---|---|---|
| bounce | 19,307 | −0.1 (t −0.1) | +0.1 (t 0.1) | +0.2 (t 0.2) | +0.8 (t 0.9) |
| break | 1,659 | +4.1 (t 1.8) | +6.3 (t **2.0**) | +9.2 (t **2.1**) | +8.3 (t 1.7) |
| break, no activity selection | 2,475 | +1.6 (t 0.9) | +2.6 (t 1.1) | +4.0 (t 1.2) | +4.0 (t 1.1) |
| bounce, 5+ touches | 5,279 | −1.6 (t −1.9) | −2.3 (t **−2.0**) | −2.6 (t −1.9) | −2.2 (t −1.4) |
| bounce, *with* the 15m trend | 10,035 | +0.5 (t 0.7) | +1.5 (t 1.8) | +2.3 (t **2.1**) | +3.4 (t **2.8**) |
| bounce, *against* the 15m trend | 9,273 | −0.6 (t −1.0) | −1.5 (t −1.7) | −2.1 (t −1.9) | −2.0 (t −1.5) |

Read: **the naked bounce is noise** (the largest population, flat at every horizon, unchanged by
fill delay). **The break has a small positive drift** that the activity selection roughly doubles
(4 → 9 bps at 120 bars) — consistent with Osler's stop-clustering polarity and the "stocks in
play" literature — but it misses the pre-registered bar by a hair (t 1.99 at 60 bars) and is
smaller than the 26 bps round trip at the pass bar. **A heavily tested level breaks rather than
holds** (5+ touches: negative), which is the same fact seen from the bounce side. **A bounce is
only worth taking with the higher-timeframe trend** (with vs against ≈ 4.5 bps at 120 bars),
which is also the only positive cell of the timeframe sweep (`signal_tf 5`, `trend_tf 15`,
`trend_gate with`, long only: +0.3% total, Sharpe 0.1). The visual notebook adds one more: on the
universe's most volatile name (AAOI, 10% average daily range) **none of its 63 raw triggers
survives the shipped `max_stop` of 2%** — a 0.3-ATR stop there is ~3% of price — and the ones
that do fire end 86% of the time at the time stop, because a 5% stop and an 8% target are out of
reach in three hours. The cap is the "rather miss" choice made explicit; whether it should scale
with the name's range is an open question (IDEAS). By the rule written before the run,
**neither polarity earned the 1000-trial study**; the leads are in IDEAS.md and the decision to
run an exploratory study on `break` (or the trend-conditioned bounce) is the user's, not this
notebook's.

## Sloped levels: `trendlines.py` (research — `signals()` does not use it)

A horizontal level is a line with slope zero. [`trendlines.py`](./trendlines.py) finds support and
resistance lines of any slope from a chart's own swing points, with the chart reader's rules written
down: every pair of same-side swings whose line price never traded through in between is a candidate
(the edges of the convex hulls of every sub-window); a later swing within `tol_atr` is a test, but a
*separate* test only if a candle closed `leave_atr` away in between; a close beyond breaks it. A kept
line needs `min_touches` tests, must have been traveled along (mean distance from price ≤
`max_gap_atr` — on an accelerating move every chord between two swings clears the path), is only
projected `max_extend` × its tested span past its last test, and one line is kept per idea (lines
sharing two tests, or converging into the same latest test, are one). Time is candle position,
prices are logs, distances are candle ATRs, so one set of knobs reads the same on 15-minute, hourly
and daily candles.

**First read (smoke test, 9 names, 60m, 2026):** the lines hold, when tested, no more often than
their *parallel twins* — the same line moved 0.5–2 ATRs off its swings (resistance 61% vs 60%,
support 54% vs 53%, n≈60–85 each). Drawing lines a trader would draw is not evidence they matter;
this is the same verdict the horizontal levels got. The many-name test is IDEAS #14.

## Notebooks

- [`level_flow_visual.ipynb`](./level_flow_visual.ipynb) — **look before you optimize**: one
  volatile name, one session's level table opened up (candles, confirmed swings, the histogram,
  the levels with their touch counts), every bounce and break trigger on the tape, single events
  zoomed with stop and target, what happened after each event, and how the level construction
  shifts with `tf_levels` / `bin_pct`. Optimizes nothing; the visual check that the mechanics are
  what the README says.
- [`level_flow.ipynb`](./level_flow.ipynb) — train panel only: defaults vs baseline, symbol
  inspection, ablation sweeps, the event study, the study and its pick, candidates file.
- [`level_flow_diagnostics.ipynb`](./level_flow_diagnostics.ipynb) — the candidates through the
  cost stress, stability, rolling walk-forward (refit vs frozen vs defaults, stitched OOS path),
  the untouched holdout, the verdict and the opt-in promotion.
- [`level_flow_trendlines.ipynb`](./level_flow_trendlines.ipynb) — the sloped-level method
  (`trendlines.py`) on real charts: how a handful of lines comes out of hundreds of candidates, the
  lines at one moment and what price did at them next, the same moment on nine charts, one chart
  through time, and the parallel-twin placebo. Candles are stamped at their CLOSE so `ASOF` sees
  only finished candles; price axes are log, because the lines are straight in log price.
