# support_bounce — anticipatory longs where a tested intraday support holds

**Thesis.** A horizontal level that recently stopped the selling tends to stop
it again: published support/resistance levels predicted intraday trend
interruptions far better than chance ([Osler, FRBNY 2000](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=888805)),
and such levels sit where resting bid depth peaks in the limit order book
([Kavajecz & Odders-White, RFS 2004](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=315660)).
The strategy buys *near* a recently tested support in a non-bear regime and
manages the position asymmetrically — a wide ATR stop below the level's noise
band, an optional trail to lock in profit once the bounce runs. The goal is
the user's original brief: avoid big drawdowns, accept some early stop-outs.

**What the literature forced into the design.** Three failure modes are baked
into the naive version of this idea, and each got a counter-measure:

1. *Tight stops under a mean-reversion entry sell the bounce*
   ([Kaminski & Lo](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=968338)):
   stops are ATR-sized and `valid()` rejects any combo whose stop could sit
   inside the level's tested noise band (`sl_atr > entry_dist + touch_tol`).
   A fixed 3%-style stop is deliberately not expressible.
2. *Resting limit bids fill exactly when price crashes through the level*
   (adverse selection, the limit order's "free option"): `entry_mode` ships
   both the anticipatory `touch` and the confirmation `bounce` (close must
   reclaim the touch band) so the data can arbitrate.
3. *1-minute churn dies to costs* (this repo's own keltner_reversion tuning):
   the space reaches selective configs (deep zones, tested levels, regime
   gate), `eod_exit: false` lets the bounce carry overnight, and the notebook
   re-verifies every pick under stressed costs.

**How a trade works.**

1. Support = the lowest low of the last `level_window` minutes. Price dips so
   the trailing `tf_minutes` low reaches `touch_tol` ATRs of the level while
   the close stays inside `entry_dist` ATRs above it, the regime is not bear,
   and the level has held ≥ `min_touches` prior distinct tests. In `bounce`
   mode the close must also be back above the touch band.
2. Buy at the **next 1-minute open**. This is the engine's stand-in for the
   real-life resting limit order: the backtest cannot book an intrabar fill
   at the limit price (and "low touched it ⇒ filled" would be optimistic
   queue-wise), so the fill lands ≤60 s after the touch, ± a few bps. Live,
   you would place the actual limit at `support + touch_tol·ATR`.
3. The stop sits `sl_atr` ATRs below entry — beyond the noise band that
   defines the level's tests. With `sl_trail: true` it ratchets up behind the
   best price since entry (the lock-in-profit variant); optionally a
   `tp_mult`-ATR target. Otherwise the trade ends on the `max_hold` time
   stop, a regime flip to bear, or (with `eod_exit: true`) the session close.

**Parameters** (all in `config.yaml`; windows are **minutes**, converted to
bars from `panel.bar`, so the same config runs on 1m/5m/1h/1d panels):

| param | meaning | intuition |
|---|---|---|
| `level_window` | support lookback (min) | the level's lifetime; longer = more meaningful, staler |
| `min_touches` | prior tests required | 0 ablates the "tested level" thesis outright |
| `touch_tol` | touch band (ATRs) | what counts as "testing" the level — and the dip target |
| `entry_dist` | entry zone (ATRs) | how close to the level a close must be; the selectivity knob |
| `entry_mode` | touch \| bounce | anticipatory limit-style dip vs confirmed reclaim |
| `tf_minutes` | trigger candle (min) | 1/5/15m dip window derived from 1m bars — **the interval question** |
| `ema_fast`/`ema_slow`/`regime_dz` | regime EMAs + dead zone | bull / bear / sideways classifier |
| `allow_sideways` | enter in sideways? | bull-only vs bull+sideways (bear never enters) |
| `sl_atr` | stop distance (ATRs) | wide by design; `valid()` keeps it below the noise band |
| `sl_trail` | trail the stop? | lock in profit vs give the bounce room — swept and searched |
| `tp_mult` | target (ATRs) | 0 = run to the time/regime/EOD exits |
| `max_hold` | time stop (min) | bounces that never come get cut |
| `exit_on_bear` | exit on regime flip | drop the bet when its regime justification dies |
| `eod_exit` | flat at close? | `false` lets the overnight gap carry the reversion |
| `skip_open`/`skip_close` | entry blackout (min) | no entries in auction chaos; `skip_open ≥ tf_minutes` |

**Universe.** `HighOctaneTop20` — the pinned top-20 HighOctane names by
trailing-90d dollar volume (`tsim subset`), the same tape the other 1m
strategies trade: volatile enough to test levels intraday, liquid enough for
realistic next-bar-open fills.

**Capital model.** Unlike the per-symbol strategies (each name trading its own
$10k book), this strategy runs the engine's **cash-shared signal mode**: ONE
$10k wallet for the whole universe, and each entry deploys `cash_frac` (25%)
of the currently *free* cash — so total invested can never exceed the wallet,
exactly like a real single account scanning 20 names. The benchmark it must
beat is the same $10k book, equal-weight buy-and-hold. One approximation to
know about: per-leg costs are modeled as a flat fraction assuming ~$10k
orders; the wallet's ~$2.5k child orders would pay ~4 bp/leg minimum
commission instead of the modeled ~2.5 — a small constant understatement that
cannot reorder parameter sets (the notebook's cost-stress rerun brackets it).

**Read the result honestly.** If `touch` with `min_touches: 0` dominates the
Pareto front, the "tested level" story added nothing over generic dip-buying —
that is a finding, not a success. If nothing admissible (≥300 train trades)
survives the holdout and the cost-stress rerun, the edge is not there at this
horizon; say so. Known caveats: overnight gaps through a stop fill at the (worse)
next open with slippage charged (engine-pessimistic since 2026-07-19)
level (engine-optimistic — the notebook scans trade records for those fills),
and on 1m bars a bracket's intrabar win rate is not trustworthy — judge picks
on the time-exit/EOD population.

**Run it:** open `support_bounce.ipynb`, or
`uv run tsim optimize --strategy support_bounce`.
