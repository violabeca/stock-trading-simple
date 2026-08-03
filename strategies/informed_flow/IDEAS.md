# informed_flow — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Open questions

### Pessimistic fill pricing — when in the day do positions actually open/close?

Worth checking how positions are opened and closed and at which time of the day.
Today every fill (entry and exit) happens at the **next bar's adjusted open**
(`price=panel.open_` in both engine modes), which is symmetric and neutral —
realism comes only from the per-leg `fees`/`slippage` fractions.

For a deliberately **pessimistic** bound, one should probably open a position at
the **day's high** and close it at the **day's low**. That would answer: does the
edge survive the worst plausible intraday execution, or does it live inside the
open-to-open assumption? A middle ground (e.g. slippage stress via the existing
cost-override technique used for candle_scalp) may be enough; the high/low bound
is the extreme end of that spectrum.

Note: fill pricing lives in the engine (`backtest.run` / `_run_portfolio`), so
actually implementing this is an engine change — proposal-only per repo rules.

## Validation & study design

### The two study objectives are ~the same number — ENGINE PROPOSAL

Across ~2,900 completed trials in **both** v13 and v14, `info_ratio` and `excess_cagr`
correlate at **0.98**. That is why a 3,000-trial NSGA-II run collapses to a Pareto front of
2–3 distinct points: the "multi-objective" search is effectively single-objective, and the
front buys almost no diversity of trade-offs to choose between.

Fix would be in `optimize.OBJECTIVES` (`optimize.py:63`): keep `info_ratio` and replace the
second objective with something genuinely orthogonal — turnover/cost, max drawdown, or
breadth of sleeve contribution — or drop to a single objective and spend the saved trials
on a stability penalty (below). **Engine change, proposal-only.**

### Stability-penalized picking — ENGINE PROPOSAL

`pick_best` (`optimize.py:341`) takes the best point on the front with no regard for whether
its parameter *neighbourhood* scores similarly. Given how flat this strategy's objective is
over the sleeve weights, that is close to picking noise. §10d already computes the
neighbourhood; folding it into selection (down-weight configs whose neighbours disagree)
would target the exact overfitting documented below. **Engine change, proposal-only.**

### `slice_panel` / `signal_quality` are copy-pasted across three notebooks — ENGINE PROPOSAL

Both are duplicated verbatim in `informed_flow`, `value_factor` and `conviction_flow`
notebooks with no shared home. Promoting them into `src/` would kill the drift risk, but it
adds engine surface — weigh against the "keep it small" prime directive before doing it.

### Walk-forward says the edge does NOT survive inside the train window (2026-07-28)

Once §10b was fixed to re-pick inside the *shipped* parameter family (see below), it stopped
flattering the strategy:

```
mean IS excess +10.94%  vs  mean OOS excess -3.49%   (paired t=4.65, p=0.02)
folds beating buy-and-hold out of sample: 2/4
```

A paired t-test at p=0.02 says the in-sample-to-out-of-sample degradation is real, not noise.
This is the single most important open item on the strategy: the 2025+ holdout looks strong,
but four expanding folds inside 2020–2024 do not corroborate it. Treat the holdout result as
*unconfirmed* until this is understood. Candidate explanations worth separating: regime
dependence (2025 was a strong momentum tape), the small fold count, and genuine overfitting.

### Sleeve survival across folds (2026-07-28)

From the fixed §10b, how often each sleeve was chosen out of sample:

| sleeve | folds | verdict |
|---|---|---|
| `w_insider` | 3/4 | keep — also the strongest literature prior |
| `w_dtc` (short interest) | 3/4 | keep |
| `w_rating` | 2/4 | keep, weakly |
| `w_ptrev` | 2/4 | keep, weakly |
| `w_trend` | 1/4 | **dead weight** |
| `w_rsi` | 1/4 | **dead weight** |

Consistent with a controlled holdout ablation (removing both price sleeves moved holdout
info ratio 2.02 → 2.02, i.e. not at all). The price sleeves stay at weight 0 and out of the
Optuna space; §4c cell 15 still scores them so the decision stays falsifiable.

## Signal construction

### Decouple the eligibility window from `insider_window`

`insider_window` does double duty: it is the insider sleeve's lookback AND the
eligibility window (`_eligibility`), so an optimizer pick at the 378-bar ceiling may
be buying *breadth* (more names qualify) rather than a longer signal horizon — the
confound is self-documented in `config.yaml`. A separate `elig_window` param would
let the search answer "what horizon?" and "how much coverage?" independently.
Falsification: re-run the study with the split; if the picked `insider_window`
collapses back toward the 1–6 month insider-drift literature while `elig_window`
stays wide, the 378 pick was breadth all along.

### `_coverage` uses the best-covered bar, not coverage over time

`_coverage` takes the **max** over bars of per-bar breadth, so a sleeve that only
reaches breadth late in the window still counts as "present" for the whole window —
early bars sit on neutral 0.5 fills that dilute the composite (and can quietly tilt
it, the same failure mode the short_interest audit exists for). Idea: a bar-by-bar
liveness mask — blend a sleeve only on bars where its own breadth clears
`_MIN_COVERAGE`, instead of all-or-nothing for the window. Check coverage-over-time
in the notebook before assuming this matters for the current DB state.

### Sub-sleeve horizon coherence

The insider literature puts the drift at ~1–6 months, but `max_hold` searches to 252
and `insider_window` to 378. A cheap validation cell: per-sleeve IC decay by horizon,
so the holding-period search range is set from evidence rather than generosity.

### Price-regime market-condition sleeves (trend + RSI) — IMPLEMENTED (2026-07-21, study v10)

Added two price-derived sleeves to the composite — MA-trend (a fast/slow SMA crossover,
`SMA(trend_fast)/SMA(trend_slow)−1`) and Wilder RSI(`rsi_period`) — as a confirmation tilt
against entering names that drift
sideways or down while showing a strong alt-data footprint. They read `panel.close`
directly (no alt-data table, no engine change) and enter `_compose` exactly like every
other sleeve, weighted by `w_trend`/`w_rsi`; a 0 weight ablates and renormalizes. Enabled
by default (both weights 1.0 in `config.yaml`); set either to 0 to disable that sleeve.
Searchable in the v10 Optuna space (`low: 0.0`, so ablation stays reachable) and in the
coarse sweep as `[0.0, 1.0]`.

Design note: implemented as a **soft weighted tilt**, not a hard eligibility gate, so it
degrades to "no effect" at weight 0 rather than needing a separate enable flag. It is
ranked only within the alt-data-eligible cross-section (`_eligibility` still counts
insider+rating events only), so it confirms an already-rankable name and cannot introduce
a pure price-momentum name with no alt-data.

Two open falsifications — they now ship on (weight 1.0), so these validate that default
rather than gate an opt-in; run both, and drop a sleeve's weight to 0 if it fails:
1. Run §4c (rank IC + decile spread vs cross-sectionally demeaned forward returns) with
   `w_trend`/`w_rsi` enabled: does either sleeve carry a positive *demeaned* spread, or is
   it just market beta dressed up as selection?
2. Re-run the study (v10) with them searchable and check whether the optimizer puts weight
   on either sleeve *net of* the four alt-data sleeves, or leaves them at 0.

## Portfolio construction

### Soft re-trim band for runaway winners — IMPLEMENTED (2026-07-21, study v9)

Shipped as the `retrim_k` concentration cap (default `2.0`), enforced in the engine
(`backtest._retrim_loop`). It ended up a **hard** cap rather than the "soft, one order per
breach" band originally sketched here: it iterates run→trim→re-run to a fixed point so no
held name ever exceeds `k ×` its slot, and the freed cash is reinvested by rotating into
the next name the selection picks (not by topping up existing holds — that was rejected as
less faithful to "buy a fresh name"). Deliberately a fixed policy, **not** swept or in the
Optuna space — optimizing over it would just tune the cap to whatever flattered the
in-sample winner, which is the opposite of what it is for. Disable with `retrim_k: null`.
The falsification cell (compare v9 to `retrim_k: null` on the holdout) is still worth
running to price what the insurance costs.

## Known caveats (documented, not actionable here)

- Ratings + price-target sleeves carry survivorship bias (yfinance covers only
  currently-covered names); insider and price data do not.
- `LiquidLeaders` itself is a current-listings snapshot (screener universe).
