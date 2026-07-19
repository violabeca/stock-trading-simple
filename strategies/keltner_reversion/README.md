# keltner_reversion — intraday dip-buying at the lower Keltner band, in uptrends only

**Thesis.** High-volatility stocks overshoot: a close more than `band_mult`
ATRs below its short-term mean (the lower **Keltner band**) usually contains a
liquidity component — stops, margin calls, index flows — that snaps back toward
the mean. Academic short-term-reversal studies find this effect, interpreted
as compensation for providing liquidity, is *strongest in volatile names and
stressed moments* — and at the 1-minute scale that liquidity provision is the
whole game. The regime filter only allows longs while the name trades above
its slower intraday SMA (and shorts below it), so dips are bought with the
session's drift rather than against a collapse. It is the style complement of
the repo's two momentum strategies: `donchian_breakout` buys the stretch,
this fades it.

**Capital model.** The whole strategy trades ONE shared $10k wallet (engine
cash-shared mode): each entry deploys `cash_frac` (25%) of the currently *free*
cash, so total invested never exceeds the book — the repo-wide comparability
rule. An opposite-direction entry closes the open position (percent sizing
cannot reverse in one order).

**How a trade works.**

1. Close drops below `SMA(band_window) − band_mult × ATR` while still above
   `SMA(trend_window)` → buy at the next bar's open. (With
   `direction: both/short`, a close *above* the upper band *below* the trend
   SMA opens a short.) Entries are **blocked in the first `skip_open` and
   last `skip_close` minutes** of each session — the auction edges are
   structurally noisy.
2. The position exits at the next bar's open once the close touches the
   midline again (the profit-take), or `horizon` minutes after entry if the
   reversion never comes — whichever is first. `eod_exit: true` forces flat
   on the **day's final bar**; the tuned default is `false` — optimization
   consistently prefers letting deep dips ride overnight, because a large
   part of the reversion arrives with the next day's gap.
3. Optionally a fixed stop-loss sits `sl_mult × ATR` below entry (`sl_mult: 0`
   disables it). There is deliberately **no trailing stop**: the bet is a
   return *to a level*, and a trail would cut it off mid-wiggle.

**Parameters** (all in `config.yaml`):

| param | meaning | intuition |
|---|---|---|
| `band_mult` | entry stretch in ATRs | **the** knob: higher = rarer, deeper, better-paid dips |
| `band_window` | midline SMA (minutes) | what "the mean" is; also the exit level |
| `trend_window` | regime SMA (minutes) | which dips are safe to buy; must be > `band_window` |
| `atr_window` | ATR smoothing (minutes) | how fast the band adapts to a name's volatility |
| `horizon` | time stop (minutes) | how long to wait for a reversion that isn't coming |
| `direction` | long / short / both | fading rips in momentum names is the risky half; default long |
| `sl_mult` | stop distance in ATRs | 0 = ride the dip out until midline/horizon; >0 caps the tail |
| `skip_open` / `skip_close` | entry blackout (minutes) | no fills into the open/close chaos |
| `eod_exit` | flat at session close | intraday means intraday |

**Universe & train/holdout.** The strategy trades `HighOctaneTop20`, the
pinned subset of the sibling repo's **HighOctane** universe (top volatility
tercile within sector) ranked by trailing-90-day dollar volume
(`tsim subset --universe HighOctane --top 20`) — reversion feeds on
volatility, but 1-minute fills need liquidity, and per-leg costs bite a
strategy that trades this often. `data.end` is deliberately **not**
"everything": it is the *train cutoff*. Optuna and the sweep only ever see
data up to it, and the notebook's out-of-sample section replays the picked
parameters on the untouched tail — the one honest number in the whole flow.
The same `signals()` also runs on daily bars (`bar: 1d` — session guards go
inert); a `keltner-1d-v1` study on 40 HighOctane names is kept in `studies/`
for reference. Possible v2 knobs, deliberately left out to keep the search
small: a tunable exit level (`exit_mult`) and a trailing-stop option.

**Run it:** open `keltner_reversion.ipynb`, or
`uv run tsim optimize --strategy keltner_reversion`.
