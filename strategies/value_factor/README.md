# value_factor — cross-sectional value + quality

Rank a liquid US universe by how **cheap** and how **profitable** each name is,
and hold the top slice. Two of the most-replicated equity factors, combined as an
equal-weighted rank composite and rebalanced monthly.

## Thesis

- **Value** — cheap stocks beat expensive ones over the long run (Fama-French
  HML). Expressed two ways a single price ratio cannot capture: **book-to-price**
  (equity ÷ market cap) and **earnings yield** (trailing-twelve-month net income
  ÷ market cap).
- **Quality** — profitable firms beat unprofitable ones *at the same price*
  (Novy-Marx 2013). Measured as **gross profitability** = TTM gross profit ÷ total
  assets, the cleanest top-of-the-income-statement definition, strongest among
  value names.

All three come from the `fundamentals` sleeve — SEC EDGAR XBRL `companyfacts`,
point-in-time by filing date (see `src/trading_simple/features.py` and the
`fundamentals` builders). The feature layer forward-fills each level from its
filing; the ratios that need price (market cap = price × shares outstanding) are
formed in `signals()`, because price lives on the `Panel`, not the feature table.

## How it trades

1. Form the three factor ratios per name.
2. Percentile-rank each **present** factor across the *eligible* cross-section — a
   name missing any used factor is dropped from the ranking outright (not handed a
   neutral score), so it neither ranks nor distorts anyone else's percentile.
3. Blend the ranks with tunable weights into one composite (equal weight by
   default — a fitted blend overfits at this scale).
4. On every `rebalance_days`-th bar, hold the top `max_positions` names; carry the
   book between rebalances. Value/quality turn over on the scale of quarters, so a
   monthly cadence captures the signal at low cost and needs no stateful
   hysteresis loop.

**One shared book (portfolio mode).** `signals()` returns a `weights` frame, so
the engine runs one shared `init_cash` account split into `max_positions` fixed
slots of `1/max_positions` each — a new name is funded by selling an old one, and
a thin selection holds cash rather than concentrating. (A per-symbol cash account
is fatal for a ranker: cash attached to a name it is flat in sits idle, leaving the
book structurally under-invested against a fully-invested benchmark.) Entries are
next-bar-open executed; every feature is already point-in-time lagged, so there is
no look-ahead.

A factor that is missing or below the `min_coverage` breadth floor is dropped and
its weight renormalized away, so the strategy degrades gracefully.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `w_bp` | 1.0 | Composite weight on book-to-price (value). 0 ablates it. |
| `w_ey` | 1.0 | Composite weight on earnings yield (value). 0 ablates it. |
| `w_gp` | 1.0 | Composite weight on gross profitability (quality). 0 ablates it. |
| `rebalance_days` | 21 | Bars between re-selections; the book is held fixed between them. |
| `max_positions` | 15 | Slots in the book; each holds `1/max_positions` of the capital. |
| `min_coverage` | 0.5 | Drop a factor covering less than this share of the universe. |

Only the *ratios* of the three weights matter (the composite renormalizes by their
sum), so any relative mix is reachable and each factor can be zeroed.

## Reading the result

The `value_factor.ipynb` notebook must show the strategy against
`strategies/baseline` (equal-weight buy-and-hold of the same panel): the equity
overlay plus `excess_cagr` / `info_ratio`. Objectives are **benchmark-relative** —
a long-only ranker earns a positive absolute Sharpe just by being invested, so
only the active numbers separate stock selection from market exposure. The
notebook also runs a **rank-IC / decile-spread** check of the composite against
*cross-sectionally demeaned* forward returns (which removes the market and leaves
only selection), and an out-of-sample replay on the untouched 2025+ tail.

## Caveats

- **Point-in-time via `filed`.** Each XBRL fact is keyed on its SEC filing date
  (`known_at = filed @ 21:00 UTC`), so the backtest only ever sees a figure after
  it was public. `companyfacts` (full amendment history) is used, never the
  `frames` API (which returns the latest restated value → look-ahead).
- **TTM.** Income flows are rolled to trailing-twelve-months in the provider
  (`FY(y-1) + YTD(y) − YTD(y-1)`, collapsing to the full year at year-end), and
  fall back to the last full fiscal year when the quarterly cumulative needed for
  the delta is unavailable.
- **Foreign filers (20-F).** ~15–20 LiquidLeaders names (ASML, AZN, TSM, SAP, …)
  file annually in the `ifrs-full` taxonomy; the IFRS tag fallbacks recover
  book-to-price and last-annual earnings, but their TTM degrades to "last fiscal
  year," and they warm up ~a year after `data.start`.
- **Market cap uses adjusted close × as-reported shares.** Around a split, shares
  lag until the next filing while the adjusted close is already post-split,
  transiently distorting one name's rank. Monthly rebalance + ranking bound the
  impact.
- **Fundamentals are screened at read time** (`src/trading_simple/features.py`). A share
  count outside 1e6–2e11, or one furnished on a 6-K, is dropped as a filer error; a day's
  filings never move a level backwards; and a level no filing has refreshed in 800 days is
  masked — so V, for which SEC has no usable share tag since 2010, drops out of the
  market-cap factors instead of being priced on its 2010 count. ADR market caps are still
  off by the ADS ratio (TSM ×5, TM ×10): SEC reports ordinary shares, the stored price is
  per ADS.
