# informed_flow — daily cross-sectional composite of alternative-data signals

**Thesis.** Three groups of informed participants leave a *public* footprint
that predicts individual-stock drift over weeks to months, and combining their
independent signals is more robust than betting on any one. This is the repo's
first **alternative-data** strategy and its first **daily cross-sectional
ranker**: each day it scores every name in a liquid universe on what insiders,
analysts, and short sellers are doing, and holds the top-ranked names long.

- **Corporate insiders (SEC Form 4).** Open-market insider *buys* are the
  single best-replicated alt-data signal (Cohen-Malloy-Pomorski 2012:
  opportunistic buys ~82 bp/mo; Lakonishok-Lee 2001: buys informative, sells
  not). We use net signed buy dollars **scaled by trailing dollar volume**, so a
  mega-cap's routine transaction cannot swamp a small-cap's conviction buy.
- **Sell-side analysts.** Rating *changes* drift for months (Womack 1996:
  downgrades −9%/6mo, upgrades +2%). Consensus *levels* are unprofitable net of
  costs (Barber et al. 2001), so we use only the up/down change stream,
  EMA-smoothed.
- **Short sellers (FINRA).** High days-to-cover is bearish, but the short leg's
  alpha mostly accrues to the lender as a borrow fee (Drechsler-Drechsler) and
  the value-weighted effect is weak (Asquith-Pathak-Ritter). So this strategy is
  **long-only** and treats *low* days-to-cover as a favourable tilt — it never
  shorts crowded names.

Signals we deliberately **do not** use: analyst price targets and news sentiment
(no usable point-in-time history — the phantom-feature trap), and classic
PEAD/SUE (dead in large caps since ~2006, Martineau 2022).

**How a trade works.**

1. Each day, for every name: rank it cross-sectionally on each present signal
   (percentile rank), combine the ranks with tunable weights (`w_insider`,
   `w_rating`, `w_dtc`) — an equal-weighted rank composite resists overfitting
   far better than ML stacking at this scale (Blitz; Israel-Kelly-Moskowitz) —
   then re-rank into one composite score in [0, 1].
2. **Enter** (next bar's open) when a name's composite rank is at or above `q_in`
   (top quantile) **and** it is *eligible* (had at least `min_events`
   insider+rating events in the trailing `insider_window`, so never-covered
   names and ETFs are excluded). **Hold** under **TopK-Drop hysteresis**: exit
   only when the rank falls below the wider `q_out` band, with a `max_hold`
   time-stop backstop. The `q_in`–`q_out` gap damps turnover.
3. Long only, no stops — the rank-exit band is the exit thesis. Because the
   engine backtests each symbol independently and averages, "top quantile" is a
   per-name rule (each name trades its own cross-sectional rank); the equity
   curve is the equal-weight average across names, and `pct_profitable` reads
   breadth.

**No look-ahead.** Every feature is point-in-time by construction. Each event is
bucketed by its `known_at` (SEC filing *acceptance* time for insiders — never
the transaction date; FINRA *dissemination* for short interest — never
settlement; announcement time for ratings), then `trading_simple.features`
applies a one-bar lag so a feature at bar *D* reflects only what was knowable
strictly before *D*. Combined with next-bar-open execution, an event known on
day *D* moves a position no earlier than the open of *D+2* — deliberately
conservative, immaterial at these horizons.

**Parameters** (all in `config.yaml`):

| param | default | meaning |
|---|---|---|
| `insider_window` | 63 | trailing bars for insider dollars, the dollar-volume normalizer, and eligibility |
| `rating_halflife` | 21 | EMA half-life (bars) for the rating-change stream |
| `w_insider` / `w_rating` / `w_dtc` | 0.4 / 0.3 / 0.3 | composite weights (a weight of 0, or a missing feature, ablates that signal) |
| `q_in` / `q_out` | 0.8 / 0.6 | enter above `q_in`, exit below `q_out` (hysteresis band) |
| `max_hold` | 126 | time-stop backstop (bars) |
| `min_events` | 1 | eligibility floor: insider+rating events in the window |

**Data.** `data.features: [insider, ratings, short_interest]` attaches the
alt-data matrices to the panel at load time. Populate them with
`tsim altdata --dataset all --universe LiquidLeaders --since 2020-01-01` (this
repo's own SEC/Yahoo/FINRA providers — no dependency on the sibling repo), or
run `notebooks/03_altdata_update.ipynb`. If short interest has not been ingested
yet, the strategy simply drops it and runs on the other two signals.

**Backtest honestly.** `informed_flow.ipynb` trains on 2020–2024 and keeps
2025–2026 as an untouched holdout, and carries a validation block: costs-on plus
a cost-doubling stress and breakeven-bps; ~4-fold walk-forward; the Deflated
Sharpe Ratio over the Optuna study; parameter-neighborhood stability; a
sub-period table; a random-entry baseline; and a minimum-trades admissibility
filter on the Pareto front. Report `n_trades` beside every Sharpe.

**Caveats.** yfinance analyst ratings exist only for currently-covered names, so
that one sleeve carries residual survivorship bias even though prices and SEC
insider data do not; claims are restricted to the liquid universe. ~5 years of
history is short for cross-validation — the DSR, random baseline, and holdout
guard against over-reading it.
