# donchian_breakout — intraday channel breakouts with a tight ATR trailing stop

**Thesis.** A close beyond the highest high / lowest low of the last N minutes
(the Donchian channel, the entry of the famous "Turtle" trend systems, here run
inside the trading day) marks a short-term momentum event. Published breakout
studies put the directional hit rate around 52% — barely better than a coin
flip, which is *fine*: the profit comes from asymmetry, not accuracy. A **tight
trailing stop** (a multiple of the Average True Range, so it scales with each
name's own volatility) keeps every loss small while letting the occasional real
intraday trend run. The result profile to expect: many small stop-outs,
frequent medium wins, few large losses.

**Capital model.** The whole strategy trades ONE shared $10k wallet (engine
cash-shared mode): each entry deploys `cash_frac` (25%) of the currently *free*
cash, so total invested never exceeds the book — the repo-wide comparability
rule. An opposite-direction entry closes the open position (percent sizing
cannot reverse in one order).

**How a trade works.**

1. Close breaks above the 30-minute high → buy at the next bar's open
   (below the 30-minute low → short). Entries are **blocked in the first
   `skip_open` and last `skip_close` minutes** of each session — early bars
   trivially "break out" of a channel built across the overnight gap, and the
   closing minutes are auction chaos.
2. A trailing stop rides `atr_mult × ATR` behind the best price since entry —
   it is both the initial risk cap and the profit lock.
3. The position ends when the stop is hit intrabar, reverses when the
   opposite band breaks, or — with `eod_exit` on (the default) — goes flat on
   the **day's final bar**, so nothing is held overnight. Turn `eod_exit` off
   and a position can carry into the next session (still exiting via the stop
   or a reversal); the sweep and Optuna space try **both** so the data decides
   whether closing the overnight gap risk is worth cutting winners short.

**Parameters** (all in `config.yaml`):

| param | meaning | intuition |
|---|---|---|
| `entry_window` | channel lookback (minutes) | **the** knob: shorter = more, noisier breakouts |
| `atr_window` | ATR smoothing (bars) | how fast the stop adapts to a name's volatility |
| `atr_mult` | stop distance in ATRs | tighter = smaller losses but more shake-outs |
| `direction` | long / short / both | shorts pay in downtrends, cost carry elsewhere |
| `skip_open` / `skip_close` | entry blackout (minutes) | no breakouts of the overnight gap or into the close |
| `eod_exit` | flat at close vs hold overnight | `true` = no overnight gap risk; `false` = let winners run past the close — swept and optimized both ways |

**Universe.** The strategy trades `HighOctaneTop20`, a pinned subset of the
sibling `stock-trading` repo's **HighOctane** universe (top volatility tercile
within sector, imported via `tsim import-universes`). The subset method: rank
every HighOctane member by **average daily dollar volume over the trailing 90
days** of stored daily bars and keep the top 20 —
`tsim subset --universe HighOctane --top 20`. Breakouts feed on volatility,
but 1-minute fills need liquidity: the ranking keeps the high-vol character
while ensuring tight spreads and realistic next-bar-open fills, and pinning
the list in `config/universes.yaml` keeps backtests reproducible (re-run
`tsim subset` to refresh the snapshot).

**Data note.** The config deliberately loads only ~2 months of 1-minute bars
(`data.start`) while the setup is being tested; widen it once the pipeline
looks sane. Symbols without 1m history are skipped — run
`notebooks/02_backfill.ipynb` (IBKR) to add more.

**Run it:** open `donchian_breakout.ipynb`, or
`uv run tsim optimize --strategy donchian_breakout`.
