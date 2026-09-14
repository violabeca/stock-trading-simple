# squeeze_breakout — intraday breakouts taken only on multi-gate confluence

**Thesis.** Most intraday channel breakouts fail — practitioner datasets put
raw false-break rates at 50–70%. `donchian_breakout` monetizes that anyway
with many trades and a tight trail; this strategy is its **selective
complement**: it trades *few* breaks and demands confluent evidence for each.
A break is taken only when the tape was **compressed** before the move
started (volatility clusters — expansion follows compression, the Crabel
NR4/7 / Bollinger BandWidth lineage), **participation** is abnormal (trailing
volume well above its time-of-day expectation, the "stocks in play" effect),
and the break goes **with the session's trend**. The squeeze here is an
ATR-compression ratio — a deliberate lightweight proxy for the TTM
Bollinger-inside-Keltner squeeze from the same own-history compression
family. Donchian trades every break; this one waits for the green lights.

**Capital model.** The whole strategy trades ONE shared $10k wallet (engine
cash-shared mode): each entry deploys `cash_frac` (25%) of the currently *free*
cash, so total invested never exceeds the book — the repo-wide comparability
rule. An opposite-direction entry closes the open position (percent sizing
cannot reverse in one order).

**How a trade works.**

1. At a bar's close, all enabled gates must agree: the lagged squeeze ratio
   `ATR / SMA(ATR, squeeze_window)` sits below `squeeze_thresh`; the close
   breaks the prior `break_window`-bar high (long) or low (short); the last
   `vol_bars` bars' volume sums to at least `vol_mult`× its time-of-day
   expectation; and the close is on the right side of `SMA(trend_window)`.
   Entry at the **next bar's open**. Entries are blocked in the first
   `skip_open` / last `skip_close` minutes of each session.
2. The exit is a layered bracket scaled by **normal** volatility
   (`max(ATR, SMA(ATR, squeeze_window))` — never the squeeze-compressed
   reading the entry certifies): a fixed stop `sl_mult` scales away, an
   optional target `tp_mult` scales away (`0` = no target, winners run), a
   `horizon`-bar time stop for breaks that go nowhere, and with `eod_exit`
   (default) a forced flat on the day's final bar. There is deliberately
   **no trailing stop** — the trail is donchian's exit thesis; a fixed
   bracket is this one's (a searchable trail is the documented v2 knob left
   out to keep the space small).
3. Engine fine print (verified): stop fractions anchor at the fill bar's
   close and activate the next bar; when one bar spans both stop and target
   the **stop** fills (pessimistic); stop legs pay fees but no slippage and
   gap-throughs fill at the bar's true open with slippage charged
   (engine-pessimistic since 2026-07-19, shared by
   all strategies). Treat same-bar R:R as approximate.

**Parameters** (all in `config.yaml`):

| param | meaning | intuition |
|---|---|---|
| `squeeze_thresh` | max lagged ATR/baseline ratio | **the** selectivity knob: lower = quieter tape required; 0 = gate off (ablation) |
| `squeeze_lag` | bars the ratio is lagged | "quiet before the move *started*" — doesn't veto multi-bar breaks |
| `break_window` | channel lookback (minutes) | what counts as a new extreme; donchian's knob |
| `vol_mult` | participation multiple | how abnormal the last `vol_bars` minutes' volume must be; 0 = gate off |
| `vol_bars` | bars in the volume sums | k-bar form rewards building participation, not one print |
| `profile_days` | volume-baseline lookback (sessions) | how many past days define the expected time-of-day volume |
| `trend_window` | regime SMA (minutes) | break only with the drift; must be ≥ 2× `break_window`; 0 = gate off |
| `atr_window` / `squeeze_window` | ATR and its baseline (minutes) | the compression ratio's two legs; baseline also floors the stop scale |
| `sl_mult` | stop distance (stop-scale units) | caps the loss; < 1 is inside 1m noise — searched from 1.0 |
| `tp_mult` | target distance; 0 = none | 0 rides winners to the time stop; > 0 books an explicit R:R |
| `horizon` | time stop (minutes) | dead breaks are dead risk; momentum decays fast |
| `direction` | long / short / both | gates are symmetric; default both |
| `skip_open` / `skip_close` | entry blackout (minutes) | no fills into the auction chaos |
| `eod_exit` | flat at session close | stops cannot price overnight gaps; stays on |

**Universe & train/holdout.** Trades `HighOctaneTop20` (the sibling repo's
HighOctane volatility universe, pinned to the 20 most liquid names by
trailing-90-day dollar volume via `tsim subset`) — compression-expansion
cycles need volatility, 1-minute fills need liquidity. `data.end` is the
**train cutoff**: Optuna and the sweep only ever see data up to it, and the
notebook replays the picked parameters once on the untouched ~2-month tail.

**Reading the optimization honestly.** The objectives are now
benchmark-relative and annualized (`excess_cagr` ↑ against `tracking_error` ↓ vs
equal-weight buy-and-hold; the older studies predate that change and the engine
refuses their journals — re-run before trusting a front). A four-gate strategy
owns a selectivity dial, so the front's ratio end attracts few-trade artifacts.
The notebook therefore **filters the front to ≥ 300 train trades** before
picking, ablates each gate (a gate whose removal doesn't hurt is documented
as not earning its keep — a front piling on a gate's off-edge means the same
thing: remove it, don't widen), stress-tests the pick at punitive costs, and
gates the holdout verdict on ≥ 50 out-of-sample trades. One regime (~10
months), 20 correlated high-vol names: breadth across symbols substitutes
for regime diversity, imperfectly. Deliberate v2 omissions: a trailing-stop
mode and a midday entry blackout.

**First study result (squeeze-1m-v1, ~2,000 trials, honest read).** The gate
ablation *validates the mechanics*: at the picked parameters, disabling the
squeeze gate multiplies trades ~7× and turns +3% into −13%, and disabling the
volume gate more than doubles trades and flips the sign — those two gates do
exactly what the thesis claims. The **trend gate did not earn its keep**
(removing it left results flat-to-slightly-better in both rounds) — the v2
candidate change is dropping it, not widening anything. The *net edge*,
however, did not survive the checks, twice: the best admissible train point
is thin (≈ +3% / Sharpe ≈ 0.7 over ~10 months), decays to ≈ 0 under the
6 bps/5 bps cost stress, and the untouched holdout came back negative for
the pick *and* its front neighbors in both study rounds. Per-symbol, gains
concentrate in the fastest momentum names (AAOI, IREN, RKLB, HOOD) and bleed
in the slower ones (GS, COHR). Conclusion: on 1-minute HighOctane data at
realistic friction, confluence gating dramatically reduces the bleeding of
raw breakouts but does not clear costs in this regime — consistent with the
published falsification studies of bar-level intraday signals.

**Run it:** open `squeeze_breakout.ipynb`, or
`uv run tsim optimize --strategy squeeze_breakout`.
