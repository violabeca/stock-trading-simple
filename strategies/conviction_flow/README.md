# conviction_flow — informed_flow, made selective and fundamentals-gated

**Thesis.** `informed_flow` ranks a liquid universe on what informed participants
do (insider buying, analyst rating changes, price-target revisions, short-interest
avoidance) and holds the top names. On a ~100-name universe with a 10-slot book,
**far more names clear the entry quantile than there are slots** — most qualifying
names are turned away (`slot_blocked`). Raising `q_in` just narrows on the *same*
axis (composite rank). This strategy narrows on **orthogonal** axes — signal
conviction and company fundamentals — and rotates the book faster into names that
are *decisively* better.

It is a **copy of `informed_flow`**: the same alt-data composite, the same
eligibility model, the same TopK-Drop hysteresis selector, the same portfolio-mode
`weights` output (one shared book, `max_positions` slots of `1/max_positions` each,
bought once and sold whole, never re-trimmed). It adds five open/close levers, and
**every one defaults to OFF** — so at default parameters `conviction_flow` reproduces
`informed_flow` byte-for-byte (a pinned test asserts it). The optimizer then
*discovers* which levers earn their keep rather than being forced to use them.

## The five levers

Three tighten **opening** (a name must clear all enabled gates *and* the `q_in`
rank *and* have a free slot); two speed up **closing**.

| lever | param | what it does | off |
|---|---|---|---|
| **Fundamentals quality gate** | `min_quality_pct` | A name may open only if its cross-sectional **fundamentals-quality** rank clears this floor — a veto *on top of* the flow rank, not another blended score. A hot-flow name that is fundamentally weak is never bought. | `0.0` |
| **Signal-agreement gate** | `min_agree` | Require at least this many *individual* flow sleeves to each rank the name in their own top (`q_in`) quantile — real ranks, not neutral fills. A high *composite* can come from being mildly good everywhere; agreement demands independent conviction. | `0` |
| **Conviction bar** | `min_score` | Require the composite score *level* (the blended rank, in [0, 1]) — not merely its cross-sectional rank — to clear a floor, so a weak field leaves the book in cash. | `0.0` |
| **Margin displacement** | `displace` + `displace_margin` | When the book is full, evict the weakest holding for a challenger only if the challenger outranks it by at least the margin — "close faster when another name is *much* more attractive", without churning on noise. | `displace=false` |
| **Rank-decay exit** | `exit_decay` | Sell a holding whose `score_pct` has fallen at least this far from its **peak while held** (running max since it was bought), before it drifts down to `q_out` — catching a fading thesis early. | `1.0` |

## The fundamentals-quality score

The quality gate reuses `value_factor`'s point-in-time SEC-XBRL factors (credit
where due — the feature builders and the ratio construction come straight from
that strategy):

- **Book-to-price** `fund_equity / mktcap` (value),
- **Earnings yield** `fund_ni_ttm / mktcap` (value),
- **Gross profitability** `fund_gp_ttm / fund_assets` (Novy-Marx quality),

with `mktcap = close * fund_shares`. Each factor is percentile-ranked across the
names that carry it, blended equal-weight, and re-ranked into `quality_pct`. A name
missing a factor is dropped from the quality cross-section (its rank is NaN and a
positive floor rejects it — "we can't verify quality, don't buy"). If the
fundamentals table is too thin to rank at all (below `_MIN_COVERAGE = 0.5`
breadth), the gate **self-disables** and the strategy degrades to `informed_flow` —
the same graceful degradation a missing alt-data sleeve gets. All `fund_*` matrices
are already point-in-time lagged in `features.py`, so there is no look-ahead. Share
counts are also screened there (thousand-fold unit slips dropped, stale counts masked),
so `mktcap` is missing, not wrong, for a name SEC reports no usable share count for.

## Data

`config.yaml` loads the union of the alt-data and fundamentals features:
`insider, ratings, price_target, short_interest` (the flow composite) plus
`fund_shares, fund_equity, fund_ni_ttm, fund_gp_ttm, fund_assets` (the quality
gate). Universe `LiquidLeaders`, daily bars, train 2021–2024, 2025+ untouched
holdout. All features are already ingested (see
`notebooks/03_altdata_update.py`, `--dataset fundamentals`); no new ingestion or
engine change was needed to add this strategy.

## Reading the result

- **The larger search space is the cost.** Five extra knobs multiply the trials
  needed to converge. Start with the `sweep` ablation grid (the four open-gates
  toggled on/off with the flow weights fixed) to see which levers move
  `excess_cagr` / `info_ratio` before committing to a full study — most of the win,
  if any, will come from one or two levers.
- **The motivating metric is `slot_blocked` pressure.** The notebook shows how many
  names are turned away on an average day *before vs after* the gates: a good gate
  cuts the qualifying set toward the number of slots, so the book is choosing among
  genuine contenders rather than rationing a flood.
- **Judge it against `informed_flow`, not just the benchmark.** The whole point is
  to beat the parent strategy on the untouched holdout; `informed_flow` and its v7
  study are untouched, so the two are directly comparable in
  `notebooks/04_compare.ipynb`. Selectivity that does not beat plain `informed_flow`
  out of sample is not worth its complexity — and the honest read may be that the
  flood of qualifiers was already the right breadth.
- The usual caveats from `informed_flow` and `value_factor` apply: the yfinance
  ratings/price-target sleeves are survivor-only; fundamentals are point-in-time via
  the SEC `filed` date; ~4 years of train is a short sample; read `n_trades` beside
  every Sharpe and prefer sub-period consistency to a high headline.
