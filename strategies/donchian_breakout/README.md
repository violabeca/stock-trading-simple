# donchian_breakout — channel breakouts with a tight ATR trailing stop

**Thesis.** A close beyond the highest high / lowest low of the last N bars (the
Donchian channel, the entry of the famous "Turtle" trend systems) marks a
short-term momentum event. Published breakout studies put the directional hit
rate around 52% — barely better than a coin flip, which is *fine*: the profit
comes from asymmetry, not accuracy. A **tight trailing stop** (a multiple of the
Average True Range, so it scales with each name's own volatility) keeps every
loss small while letting the occasional real trend run. The result profile to
expect: many small stop-outs, frequent medium wins, few large losses.

**How a trade works.**

1. Close breaks above the 20-bar high → buy at the next bar's open
   (below the 20-bar low → short).
2. A trailing stop rides `atr_mult × ATR` behind the best price since entry —
   it is both the initial risk cap and the profit lock.
3. The position ends when the stop is hit intrabar, or reverses when the
   opposite band breaks.

**Parameters** (all in `config.yaml`):

| param | meaning | intuition |
|---|---|---|
| `entry_window` | channel lookback (bars) | shorter = more, noisier breakouts |
| `atr_window` | ATR smoothing (bars) | rarely worth tuning far from 14 |
| `atr_mult` | stop distance in ATRs | **the** knob: tighter = smaller losses but more shake-outs |
| `direction` | long / short / both | shorts pay in downtrends, cost carry elsewhere |

**Run it:** open `donchian_breakout.ipynb`, or `tsim optimize --strategy donchian_breakout`.
