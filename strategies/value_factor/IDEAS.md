# value_factor — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Portfolio construction

### Anchor the rebalance grid to factor breadth, not bar 0

`_select` marks rebalance bars as `iloc[::rebalance_days]` starting at bar 0 —
inside the fundamentals warm-up (20-F filers take up to ~a year for their first
in-window filing). Early re-picks select from a thin cross-section and the book
runs under-filled, a cash drag charged against a fully-invested benchmark that
dilutes measured `excess_cagr` for reasons that have nothing to do with the
factors. Anchoring the grid at the first bar where factor breadth clears
`min_coverage` (holding cash before it, explicitly and visibly) separates "the
factors don't work" from "the data wasn't there yet".
Note: the engine now seeds `ffill` features from pre-window filings (2026-07-19),
which shortens the warm-up materially — re-check how much dead zone actually
remains before building this.

## Signal construction

### Sector-relative ranking

Raw bp/ey ranks in a 100-name universe are partly a sector bet (banks and energy
persistently screen cheap; software persistently expensive). Demeaning each factor
by sector — or ranking within sector, then blending — isolates the within-sector
value effect the literature actually documents. Needs a sector map (GICS-lite is
enough; the universe screener already touches sector data).

### A second profitability sleeve from the same features

`fund_gp_ttm / fund_equity` (profitability on equity) or a simple accruals proxy
(ΔNI vs Δcash-flow-ish from the existing concepts) would add an orthogonal quality
axis at zero ingestion cost — every input already rides on `panel.features`.
Falsify per-sleeve with the same demeaned-IC cell the notebook uses for bp/ey/gp.

## Known caveats (documented, not actionable here)

- Market cap = adjusted close × as-reported shares: around splits the two are
  transiently inconsistent and one name's rank can distort (README documents it).
- `LiquidLeaders` is a current-listings snapshot — survivorship in universe
  selection; the value premium literature is exactly where dead names matter.
