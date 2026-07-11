# candle_scalp — intraday reversal candles at a stretched location, built to be falsified

**Thesis.** An automated "manual scalper" that watches a handful of highly
fluctuating names and buys a *rejection* candle — a hammer with a long lower
wick, or a bullish engulfing — betting the sellers who pushed price down are
spent. The honest problem: the peer-reviewed evidence on candlestick *shapes* as
intraday signals is close to unanimously **negative** once transaction costs and
data-snooping are accounted for (Marshall-Young-Rose 2006; Duvinage-Mazza-
Petitjean 2013 tested 5-minute candles on the DJIA and *not one rule* beat
buy-and-hold net of costs; Fock-Klein-Zwergel 2005). The gross edge of a shape
is ~1 bp; a round trip on a volatile name is ~10–15 bps. So this strategy does
**not** trade shapes naked. It treats the candle as a *confirmation* layered on
the one intraday edge with real support — short-term reversal at a volatility
extreme, the same edge `keltner_reversion` fades — and every gate has an
off-switch so the optimizer, the holdout and an ablation can answer the only
question that matters: **does reading the candle add anything over the location
edge alone?** If it doesn't, that negative result is the deliverable, and the
config is not shipped.

**How a trade works.**

1. A **reversal candle** prints (`pattern`): a `hammer` (lower wick dominates the
   body + upper wick — a rejection of lower prices; bearish mirror = shooting
   star), an `engulfing` bar (its body engulfs the prior, opposite-coloured
   body), or `any` (no shape required — the ablation control that reduces the
   strategy to a pure location reversal).
2. …**at a stretched location**: the close is more than `stretch_mult` ATRs below
   `SMA(band_window)` for a long (above, for a short). `stretch_mult: 0` turns the
   location gate off and trades the candle alone — a naked-candle test.
3. …**with the regime and (optionally) volume**: close above `SMA(trend_window)`
   for longs (below for shorts); and, when `vol_mult > 0`, the entry bar carries
   at least `vol_mult`× its time-of-day expected volume. Entry fills at the **next
   bar's open**. Entries are blocked in the first `skip_open` / last `skip_close`
   minutes.
4. The position exits on a **bracket**: a `sl_mult`-ATR stop, an optional
   `tp_mult`-ATR target (`0` = run on), a `horizon`-bar time stop, and — intraday
   with `eod_exit` — the session's final bar. Stops are evaluated intrabar;
   same-bar stop-and-target resolves to the **stop** (pessimistic).

**Parameters** (all in `config.yaml`):

| param | meaning | intuition |
|---|---|---|
| `pattern` | hammer / engulfing / any | **the experiment**: `any` is the no-shape control |
| `wick_mult` | hammer rejection-wick strictness | higher = a purer rejection bar (hammer only) |
| `band_window` | midline SMA (minutes) | the mean price overshoots and reverts toward |
| `stretch_mult` | entry stretch in ATRs | **the** location knob; 0 = trade the candle naked |
| `atr_window` | ATR smoothing (minutes) | how fast the band/stops adapt to a name's vol |
| `trend_window` | regime SMA (minutes) | which dips are safe to buy; must be > `band_window` |
| `vol_mult` | entry-bar RVOL gate | 0 = off; >0 demands real participation on the bar |
| `sl_mult` / `tp_mult` | stop / target in ATRs | the target must clear ~10–15 bps of friction to pay |
| `horizon` | time stop (minutes) | how long to wait for a reversion that isn't coming |
| `direction` | long / both | fading rips in momentum names is the risky half |
| `skip_open` / `skip_close` | entry blackout (minutes) | no fills into the open/close chaos |
| `eod_exit` | flat at session close | intraday means intraday |

**Why this is hard (read before trusting a green backtest).**

- **Costs are the executioner.** The global `settings.yaml` charges `spread_bps:
  2.0` — a *liquid-large-cap* number. "Highly fluctuating" names carry 5–15 bps of
  spread alone, so the default understates friction 2–5×. Re-run the picked
  params with `TRADING_COSTS__SPREAD_BPS=10 TRADING_COSTS__SLIPPAGE_BPS=3` (env
  beats YAML, no shared config touched); an edge that survives only at 2 bps was
  never real for these names.
- **Intrabar-path ambiguity.** A 1-minute bar spanning both stop and target has a
  fictional fill; the engine resolves stop-first (conservative), but a tight
  bracket's win rate is still untrustworthy — judge on the time-stop / EOD exits.
- **Data snooping.** `pattern` × a wide space × 20 tickers is a lot of chances to
  find a false winner. Judge on `pct_profitable` (breadth across names), discount
  the winning Sharpe for the number of trials searched (Deflated-Sharpe logic,
  Bailey & López de Prado 2014), and run the **gate ablation**: if `pattern: any`
  or `stretch_mult: 0` wins the front, the candlestick thesis is disproved.
- **Survivorship.** `HighOctaneTop20` is a *snapshot* of today's volatile names; a
  fully honest version would select the universe as-of-date by RVOL/ATR (left out
  of v1). And <1% of real day traders are persistently profitable net of costs
  (Barber-Lee-Liu-Odean, Taiwan) — assume any green backtest is overfit until the
  holdout says otherwise.

**Universe & train/holdout.** Trades `HighOctaneTop20`, the pinned liquid subset
of the sibling repo's **HighOctane** (high-volatility) universe
(`tsim subset --universe HighOctane --top 20`). `data.end` is the **train cutoff**,
not "everything": Optuna and the sweep only see data up to it, and the notebook
replays the picked parameters on the untouched ~2-month tail — the one honest
number in the flow. The same `signals()` also runs on daily bars (`bar: 1d` —
session guards go inert).

**Run it:** open `candle_scalp.ipynb`, or
`uv run tsim optimize --strategy candle_scalp`.

**Selected sources.** Marshall, Young & Rose (2006, *J. Banking & Finance*);
Duvinage, Mazza & Petitjean (2013, *Quantitative Finance*); Fock, Klein & Zwergel
(2005, *J. of Derivatives*); Gao, Han, Li & Zhou (2018, *JFE*, "Market Intraday
Momentum"); Gervais, Kaniel & Mingelgrin (2001, *J. Finance*, the high-volume
premium); Bailey & López de Prado (2014, the Deflated Sharpe Ratio); Barber, Lee,
Liu & Odean (Taiwan day-trader studies).
