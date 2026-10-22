# keltner_reversion — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

(The optuna space caps are deliberate — tuned across the keltner-1m study history —
so widening ranges is NOT on this list.)

## Exits

### Tunable exit level (`exit_mult`)

Already flagged in the README as the v2 knob: exit at `midline − exit_mult·ATR`
instead of demanding the full round-trip to the mean. Deep dips often retrace most
of the stretch and then stall; a partial-reversion take keeps the thesis (revert to
*toward* the level) while cutting the tail where the last 20% of the reversion
never arrives and the time stop eats the trade.

## Entries

### Regime-conditional stretch

`band_mult` scales with ATR, but ATR responds slowly to a vol-regime break — the
same 6-ATR stretch is an orderly pullback in a calm regime and the first leg of a
collapse in a fast one. A vol-of-vol conditioner (require deeper stretch when
realized vol is far above its own trailing median) separates the two states the
current single multiplier conflates.

### Market-wide adverse-selection guard

Dip-buying pays worst when the dip is the whole tape, not the name. A cheap guard —
skip entries on bars where the cross-sectional median return of the 20 names is
itself strongly negative — tests whether the edge is name-specific reversion or
disguised market-timing. If the guard helps materially, the honest conclusion is
the latter.

## Diagnostics

- Phantom `exit_after` exits from clustered raw signals (engine behavior shared by
  all the 1m scanners): scan trade records for time-stop exits shorter than
  `horizon` before deciding whether a re-entry cooldown is worth a param.

## Assessment 2026-09-12 — the one intraday strategy with a positive holdout, and why
## that number cannot be trusted yet

### Every number this strategy has is unverifiable today — fix the instrument first

The positive holdout (+8.2%, Sharpe 3.7, 183 trades over two months) is from study v2: 20
independent $10k books (20× the capital of the wallet it now trades), the pre-realism
engine (optimistic stop fills), the old objective, and a universe whose buy-and-hold
returned +460%/yr. `keltner-1m-v5` — the space this config ships — has never been run.
Order of work: `data.universe: Morning200` (`max_symbols: 200`; `bt.sweep` needs
`chunk_size=2` at that width), run v4, only then read §11.

### Turnover makes the cost model the whole story — add §12 cost stress

This is the only 1m notebook besides donchian/volume_momentum without a cost-stress
section. The v2 pick did ~990 trades in 10 months ≈ 1,200/yr × ~$2.5k ≈ $3M/yr of notional
on a $10k book: ~18%/yr of drag at the modeled 6 bp round trip, ~45%/yr at a realistic
15 bp for these names. Its +14.1%/yr train return is measured *inside* the 6 bp assumption.
Copy `support_bounce.ipynb` §12 (6/5 and 10/3 bp) and treat it as the pass bar.

### Same-bar candidate flood is resolved by column order, not by choice

Deep dips cluster in market-wide selloffs; the wallet funds the first four names at
25/18.75/14/10.5% of free cash in `call_seq` order and misses the rest. Rank same-bar
candidates by z-scored stretch and cap with `max_candidates` (`cash_frac = 1/K`), exactly
as `gap_and_go/strategy.py` does. *Wins if* the ranked book beats the column-order book on
the same panel at the same slot count.

### Test the thesis without the backtest — a market-demeaned event study

Forward return after the dip signal at 30/60/120/240 bars by `band_mult` bucket, with
+1/+2/+5-bar fill delays (`gap_and_go.ipynb` §5b pattern). If the excess is below the cost
bar, no study is needed. Pair it with informed_flow's §10f random-entry control at the same
hold length and slot count: `close > SMA(960)` on high-beta names in a bull tape means much
of the return can be exposure, not reversion.

### Minor

`valid()` wastes trials because `band_window` (5–390) and `trend_window` (60–2000) overlap
(candle_scalp made them disjoint); `indicators()` draws a close-diff range proxy while
`signals()` uses true ATR (the same class as gap_and_go's fixed bug); the README has no
results section at all.

## Known caveats (documented, not actionable here)

- 2 bps default spread understates 1m HighOctane friction — cost-stress rerun
  mandatory. (Overnight gap-throughs now fill at the true open with slippage —
  engine-pessimistic since 2026-07-19 — so `eod_exit: false` picks pay them in full.)
- `HighOctaneTop20` is a today-snapshot universe (survivorship).

## Universe contamination (measured 2026-07-28, not yet acted on here)

`HighOctaneTop20` was pinned on 2026-07-05 by `tsim subset`, which ranks on
`now() - 90 days` — April–July 2026 data — while this strategy backtests
2025-07-01 → 2026-04-30. The universe was therefore **selected using data from
inside and after the window it is tested on**, and it is not a small effect:
equal-weight buy-and-hold of those 20 names returned **+460%/yr** over the train
window (SNDK +2255%, BE +1123%; neither had pre-window daily history — a 2025
spinoff and a 2025 IPO). Because `optimize.OBJECTIVES` is benchmark-relative,
that inflated hurdle is subtracted from every trial: in `gap_and_go`'s v1 study
all 45 scoring trials landed at `excess_cagr ≈ -4.6` and the Pareto front spanned
0.05 — noise, not a trade-off curve.

The fix needs no backfill. `Morning200` (built in
`notebooks/01_universe_screener.ipynb` §8) keeps the 504 symbols with >= 200
sessions of stored 1m data in the window and ranks them by median daily dollar
volume over a period that **closes before the backtest opens**; its buy-and-hold
hurdle is a real +47%/yr. `gap_and_go` has migrated. This strategy has not — its
study and its README numbers should be read as unresolved until it does, and a
migration invalidates the existing journal (new benchmark, new panel).
