# trend_following — daily long/short trend starts, ridden behind a ratcheting ATR stop

**Thesis.** Prices trend more often than a random walk would, in both directions
(time-series momentum: Moskowitz, Ooi & Pedersen 2012; on single US stocks Wilcox &
Crittenden 2005, Zarattini, Pagani & Wilcox 2025). A trend follower does not forecast.
It waits until a price leaves its range with strength, joins the move, and lets a
volatility-scaled stop decide when the move is over. The payoff is asymmetric by
design: many small losses on false starts, and a few winners that run for months.

**How a trade works** (every decision on a daily close, every fill at the next open):

1. **A trend starts** when the close breaks the prior `entry_window`-bar high (long) or
   low (short). The breakout also needs Wilder's ADX ≥ `adx_min`, with +DI/−DI agreeing
   on the direction, so the move is directional rather than a poke through the channel.
2. **It gets a slot** if one of the book's `max_positions` (20) equal slots is free. When
   more names start than slots are free, the **strongest relative trend** goes first. That
   is the name whose 12-1 month return (`rs_window` bars back to `rs_skip` bars back) beats
   the average name's by the most, or, for a short, trails it by the most. This is
   cross-sectional momentum deciding which of today's breakouts deserve the book.
3. **The stop** starts `atr_mult` ATRs from the signal close. After that it is the
   **chandelier**: the best price since entry (highest high / lowest low) minus/plus
   `atr_mult` × the *current* ATR. It follows the price and adjusts to the name's changing
   volatility. It is **ratcheted**, so it never loosens: an expanding ATR cannot give back
   room already locked in.
4. **The exit** is whichever comes first: a **close** through the stop, or a confirmed start
   in the **opposite** direction. Either way the position is sold at the next open, whole,
   and a new trend in the other direction needs a fresh start.

**Why the stop is walked in `strategy.py`.** The engine's intrabar `sl_stop` fixes its
distance as a fraction of the entry price, so it cannot follow a changing ATR. The chandelier
is therefore walked bar by bar in `_ride` (vectorized across names) and is a
**closing-basis** stop. A bar that gaps past the level fills at that open, so gap risk is in
every number. `compose(panel, **params)` exposes the stop line, the position and every
intermediate for charts and tests.

**Capital model.** One shared book in engine portfolio mode, with 20 equal slots. Each entry
targets ±1/20 of the book's value, drifts untouched and is sold whole. Fixed slots size every
entry off the book's *current* value, so results never depend on the order entries arrived in.
The wallet's free-cash fraction did: its first ~20 entries took the whole book (see v2 below).
`backtest.order_notional: 2500` prices the $1 minimum commission at the real order size of
20 slots on a $50k account (4 bp a leg).

## Parameters

| param | shipped | meaning | intuition |
|---|---|---|---|
| `entry_window` | 9 | Donchian lookback in bars | the study wants a *short* channel: relative strength picks the names, the channel only times the entry (and, via opposite starts, most exits) |
| `adx_window` | 14 | Wilder smoothing of ADX and ±DI | Wilder's standard; not searched |
| `adx_min` | 34 | ADX floor for a start | only strongly directional breakouts count; `0` switches it off (grid ablation) |
| `atr_window` | 37 | ATR window of the stop | a slow ATR: the stop adapts to regimes, not to single days |
| `atr_mult` | 14 | stop distance in ATRs | **wide on purpose**: every tight stop lost here, so it is a disaster stop |
| `direction` | both | long / short / both | the grid runs each side alone |
| `max_positions` | 20 | equal slots in the book | a risk decision, never searched |
| `rs_window` / `rs_skip` | 252 / 21 | 12-1 month relative strength for slot priority | the literature's momentum horizon; the last month reverses |

## Results — read them with the caveats

Universe `Liquid500` (497 names, point-in-time ranking, survivor-only pool), daily bars.
Every window is scored **warm** (signals on the whole history, book from the window's first
bar). Costs are always on, measured against an equal-weight buy-and-hold of the same names.

| window | CAGR | buy & hold | excess | alpha | beta | vol | max DD | long / short P&L |
|---|---|---|---|---|---|---|---|---|
| train 2022–24 (optimized) | +10.6% | +4.8% | **+5.7%/yr** | +11.4% | −0.17 | 13.6% | −13.5% | +22% / +13% |
| · 2022 alone | +29.4% | −13.6% | +43.1% | +24.4% | −0.37 | 19.2% | −13.5% | +1% / +28% |
| · 2023 alone | −9.5% | +21.2% | **−30.7%** | +4.9% | −0.68 | 23.9% | −30.5% | +10% / −19% |
| · 2024 alone | +1.3% | +14.6% | **−13.2%** | −0.6% | +0.13 | 10.5% | −10.3% | +8% / −7% |
| **holdout 2025-01 → 2026-09-11** (read once) | +24.9% | +18.8% | **+6.1%/yr** | +20.1% | +0.25 | 17.4% | −15.2% | +49% / −4% |
| · 2025 alone | +34.0% | +14.0% | +20.0% | +32.9% | +0.07 | 13.3% | −5.1% | +30% / +3% |
| · 2026 alone (to 09-11) | +3.3% | +18.8% | **−15.5%** | +1.6% | +0.09 | 20.7% | −14.9% | +8% / −6% |

Each row is its own warm-scored book (flat on the window's first day) against a fresh
equal-weight buy-and-hold of the same window, so the yearly rows do not chain into the
multi-year ones. P&L is a fraction of the starting book.

On the holdout the original v1 design (same data, its own pick) made **−2.9%/yr** excess
(alpha +8.4%). The improvements helped out of sample. Still, read it honestly:

* **It is a crisis-alpha profile.** All of the train excess is 2022, the bear market in which
  the short side made 28% of the book. It lost heavily to buy-and-hold in 2023 and 2024, when
  the shorts gave back 19% and 7%, and its beta was negative over the train window.
* **The holdout rests on one trade.** LITE contributed +23.5% of the book, about half the
  holdout P&L. With LITE removed from the universe, the holdout excess is **−5.5%/yr** (alpha
  +8.6%). Trend following lives on a few outliers; it also means one holdout cannot confirm it.
* **The open-fill assumption carries most of the edge.** With pessimistic fills (buy at the
  high, sell at the low) excess falls to +1.0%/yr on train and +1.5%/yr on the holdout. Costs
  barely matter at ~30–40 trades a year: 6/5 and 10/3 bps cost about 0.2 points.
* **The ATR stop is a disaster stop.** In the shipped config, 51 of 68 train exits came from
  opposite starts; tight chandelier stops lost in every test (below).
* **The pool is survivor-only.** Names that later delisted, which were the best shorts, are
  missing, so the short side is understated and the long side flattered.
* **The short side is a hedge, not a profit engine.** Names breaking down to new lows
  *outperformed* the market over the next month in 2022–23 (short-term reversal, t ≈ −2 to −3).

## Study history — v1 → v6, five improvement runs, then v7

Protocol: design iterations trained on 2022–23 (scored warm from 2022-01-03) and were judged
on **2024 as a validation year**. 2025+ stayed untouched until the final refit (v7, trained
2022–24) was read once. Each run was a 300–400-trial NSGA-II study on
`[excess_cagr ↑, volatility ↓]`.

| run | change | train q50 / q90 | top-10 → 2024 | pick train / 2024 | decision |
|---|---|---|---|---|---|
| v1 | Donchian + ADX start, chandelier stop, $10k wallet (`cash_frac` 0.1) | −8.3% / −3.1% | −3.9% | +5.2% / −6.1% | baseline; stop width pinned at the upper bound |
| v2 | widen the box (stop ≤ 16 ATR, channel ≤ 300, ADX ≤ 60) | −0.4% / +8.5% | +12.3% | +14.9% / +16.5% | **artifact**, see below |
| v3 | 20 equal slots (portfolio mode), highest ADX admitted first | −7.0% / +1.8% | +11.1% | +8.7% / +10.5% | kept: the order-dependence is gone |
| v4 | 40 slots | −6.0% / −0.6% | +0.9% | +6.1% / +6.0% | rejected: diluted, no smoother |
| v5 | admit by relative strength over the entry window | +0.1% / +11.5% | **−32.5%** | +11.0% / +14.0% | rejected: short horizons overfit 2022–23 |
| v6 | admit by **12-1 month** relative strength | −2.8% / +5.3% | +4.7% | +14.7% / +6.0% | **kept** (paired test) |
| v7 | v6 refit on 2022–24, 600 trials | −0.9% / +8.9% | — | +5.7% / holdout +6.1% | shipped |

What each run taught:

* **v2 was concentration luck, not an edge.** Once the stop stopped firing (16 ATR), the
  wallet's percent-of-free-cash sizing let the first ~20 entries of 2022 own the book. Later
  entries got 0.03% each. Five names made 89% of the P&L. Starting the backtest in July 2022
  instead of January turned +14.9%/yr into −24.4%/yr. Under v3's slot model the same
  parameters scored −2.7% / −3.6%.
* **The landscape is spiky.** On v3's pick, half an ATR of stop width (8.0 → 8.5) moved train
  excess by ~10 points. With the book full, one different exit changes which names get the
  next slots. So from v4 on, candidate changes were screened as **paired comparisons across
  60–100 random parameter sets**, not by single picks.
* **Screens that decided runs 4–5** (train, paired against v3):
  * relative-strength admission: +6.4 points (t 5.1)
  * random admission: +0.3 (so v3's ADX-first ranking was no better than random)
  * low-volatility-first: −2.4 (t −2.8)
  * a market-regime gate: 200-day −5.6 points (t −5.7), 100-day −4.5 (t −5.5), 50-day −0.5
  * a breakeven lock: inert, since the chandelier already implies it
  * tight stops (2–5 ATR): −7 to −22 points in every configuration tested
* **v6 over v3 overrides the rule set before the runs.** That rule ranked versions by the
  median 2024 excess of each study's top-10 train trials, which favoured v3 (+11.1% vs
  +4.7%). A paired test on the *same* 100 random configurations disagrees in both periods:
  12-1 relative strength beats ADX-first by **+6.3 points on train (t 8.5, 84% better) and
  +3.4 points on 2024 (t 2.8)**. The top-10 metric turned out to be dominated by a few noisy
  configurations. v6's horizon was chosen after v5 failed on 2024, so 2024 is not clean
  validation for v6; only the holdout is.
* **v7's pick is not a spike.** 81% of its one-at-a-time neighbours beat buy-and-hold on train
  (median +6.9%/yr), and it sits inside every searched bound.

**Intraday.** The same code runs on minute bars. There, `skip_open` / `skip_close` block
starts in the first and last minutes of a session, and `eod_exit` closes every position
before the session's final bar. All three are ignored on daily bars.
[`trend_following_1m`](../trend_following_1m/README.md) is that experiment on ten liquid
stocks, with its own config and results.

**Run it:** open `trend_following.ipynb`, or `uv run tsim optimize --strategy trend_following`.
Pending ideas and their falsification tests are in `IDEAS.md`.
