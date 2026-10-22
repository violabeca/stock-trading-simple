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

## Data quality (found 2026-09-12 while building `turnover_momentum`)

`fund_shares` carries filer-side unit errors from `dei:EntityCommonStockSharesOutstanding`:
AJG reported ×1,000,000 shares for two 2020 quarters (market cap ×1e6 → book-to-price and
earnings yield collapse to ~0 → bottom rank), BALL a literal 0 on 2022-02-14 (dropped), RMD
and CMG ÷1,000 for one quarter each (market cap ÷1,000 → both value ratios ×1,000 → **top
rank** for a quarter), CCL ×1,000. The `fundamentals` table holds 124 rows with a
non-positive share count across 87 symbols. Fixed 2026-09-14: share counts are screened at
read time (`features._screen_shares`) — see the section below.

### Frozen fundamentals series — and gp's coverage sits on the floor (measured 2026-09-14)

The SEC provider locks each concept onto the first XBRL tag that ever had data, so a series
freezes when a filer stops using that tag. On this strategy's LiquidLeaders panel that freezes
the share count of V, MA, ACN and CCZ since 2010 (multi-class issuers stop filing a
non-dimensional cover-page count), and gross profit for AMZN, COST, BMY, PH, SCCO, CVS, ORCL,
VZ, BKNG, C, PBR and TM (the `GrossProfit` tag was abandoned; revenue − COGS is only synthesized
when the tag never existed at all).

It matters because `gp`'s best-bar coverage is 0.52 against the 0.50 `min_coverage` floor.
Config defaults, train 2021–2024, with fundamentals values older than 800 days masked:

| variant | factors present | eligible/day | excess_cagr | info_ratio |
|---|---|---|---|---|
| today | bp, ey, gp | 48 | +4.4% | 0.45 |
| stale values masked | bp, ey | 70 | −3.4% | −0.36 |
| masked, floor lowered to 0.40 so gp stays | bp, ey, gp | 35 | +4.9% | 0.53 |
| today, `w_gp: 0` | bp, ey | 84 | 0.0% | 0.02 |

So gp is the entire result, its breadth leans on stale rows, and masking them without a
per-period fallback silently ablates it. The fallback (revenue − COGS where `GrossProfit`
stops) restores current gross profit for AMZN, COST, BMY, PH and SCCO; VZ, ORCL, BKNG and C
report no gross-profit line and should stay NaN. The proposal is in
`../turnover_momentum/IDEAS.md`.

Separately, ADR market caps are wrong by the ADS ratio, because the cover-page count is
ordinary shares while the stored price is per ADS: stored price × shares at 2024-12-31 gives
TSM ~$5.0T (5× too high), TM ~$2.6T (10×), HSBC ~$0.87T (5×) and MFG ~$12B (5× too low).
BTI, SAP, ASML and the Canadian 40-F banks are 1:1 and fine.

**Fixed 2026-09-14** (what shipped is in `../turnover_momentum/IDEAS.md`). With the per-period
fallback in the parser, masking stale values no longer ablates gp: its best-bar coverage stays at
0.51 with the 800-day cap on, because AMZN, COST, BMY, PH and SCCO carry current gross profit
again. Config defaults, train 2021–2024:

| data | factors present | eligible/day | IC63 | excess_cagr | info_ratio |
|---|---|---|---|---|---|
| before the fix | bp, ey, gp | 48 | +0.004 | +4.4% | 0.45 |
| after the fix | bp, ey, gp | 51 | +0.016 (t 0.42) | +3.9% | 0.38 |

The traded composite's IC rose but is still indistinguishable from zero, so the diagnosis above
stands: selection adds little over owning the universe. V now drops out of the market-cap factors
(no usable SEC share tag since 2010) instead of ranking on its 2010 count.
