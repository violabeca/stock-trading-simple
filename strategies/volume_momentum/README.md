# volume_momentum — intraday volume spikes, momentum picks the side

**Thesis.** A minute with far more volume than usual marks the arrival of new
information or a large participant, and the burst tends to continue the
prevailing move for a short window. "More than usual" is judged against the
**time-of-day volume profile**: today's 09:35 volume is compared to the
average of the last `profile_days` sessions at 09:35 (± `smooth_minutes`
around it). Because 09:35 only ever compares to other 09:35s, the routinely
heavy open and close minutes never trigger a signal by themselves. Volume says
*when*; a **momentum indicator** — SMA(`fast`) − SMA(`slow`) — reads the
current market condition and says *which way*: positive → long, negative →
short. Both directions are always live; there is no long/short switch.

**Capital model.** The whole strategy trades ONE shared $10k wallet (engine
cash-shared mode): each entry deploys `cash_frac` (25%) of the currently *free*
cash, so total invested never exceeds the book — the repo-wide comparability
rule. An opposite-direction entry closes the open position (percent sizing
cannot reverse in one order).

**How a trade works.**

1. Volume ≥ `threshold` × its time-of-day expectation *and* momentum is
   positive → buy at the next bar's open (momentum negative → short).
   Entries are **blocked in the first `skip_open` and last `skip_close`
   minutes** of each session — the auction edges are structurally noisy.
2. A **stop loss** rides `sl_pct` below the entry (above, for shorts):
   fixed at entry, or following the best price since entry when
   `sl_trail: true`. Evaluated intrabar against the high/low.
3. The position closes `horizon` minutes after entry — or at the stop, or on
   the **day's final bar** (`eod_exit`), so nothing is held overnight.

**Parameters** (all in `config.yaml`):

| param | meaning | intuition |
|---|---|---|
| `fast` / `slow` | momentum SMA windows (bars) | their difference is the direction filter; `fast < slow` enforced |
| `threshold` | spike multiple vs expected volume | **the** knob: higher = rarer, cleaner events |
| `horizon` | bars held after entry | how long the burst is trusted to carry |
| `baseline` | `auto` / `sma` / `profile` | expected-volume model; `auto` = profile intraday, sma on 1d |
| `profile_days` | profile lookback (sessions) | how many past days define "usual" volume |
| `smooth_minutes` | profile smoothing (± minutes) | steadies thin one-minute prints |
| `vol_window` | sma-baseline window (bars) | only used with `baseline: sma` |
| `skip_open` / `skip_close` | entry blackout (minutes) | no trades into the open/close chaos |
| `sl_pct` | stop distance (fraction of entry) | e.g. 0.02 = 2%; 0 disables |
| `sl_trail` | fixed vs trailing stop | trailing locks profits, shakes out more often |
| `eod_exit` | flat at session end | intraday means intraday |

**Data note.** The config deliberately loads only ~1 month of 1-minute bars
(`data.start`) while the setup is being tested; widen it once the pipeline
looks sane. Symbols without 1m history are skipped — run
`notebooks/02_backfill.ipynb` (IBKR) to add more.

**Run it:** open `volume_momentum.ipynb`, or
`uv run tsim optimize --strategy volume_momentum`.
