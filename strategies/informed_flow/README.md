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
- **Sell-side analysts, ratings.** Rating *changes* drift for months (Womack 1996:
  downgrades −9%/6mo, upgrades +2%). Consensus *levels* are unprofitable net of
  costs (Barber et al. 2001), so we use only the up/down change stream,
  EMA-smoothed.
- **Sell-side analysts, price targets.** A **separate sleeve, not a substitute** for
  the ratings one: the large majority of target revisions (~19 in 20) arrive with the
  *rating unchanged*, which the rating stream scores as exactly zero. Brav-Lehavy
  (2003) find the drift precisely there — conditional on a recommendation
  *reiteration*, the 6-month adjusted CAR is **+6.2%** (top tercile of ΔTP) vs
  **−1.9%** (bottom), while pooled reiterations show no drift at all until you split
  by target revision. Asquith-Mikhail-Au (2005) find targets subsume earnings
  forecasts, not the reverse; Da-Schaumburg (2011) confirm the effect holds in
  S&P 500 names. We rank the **revision** (ΔTP / prior TP) and **never the
  target-to-price level** — Han-Kang (2022) show price drifts *against* the level,
  most strongly in large liquid names, i.e. exactly this universe, so ranking on the
  level would likely invert the sign.
- **Short sellers (FINRA).** High days-to-cover is bearish, but the short leg's
  alpha mostly accrues to the lender as a borrow fee (Drechsler-Drechsler) and
  the value-weighted effect is weak (Asquith-Pathak-Ritter). So this strategy is
  **long-only** and treats *low* days-to-cover as a favourable tilt — it never
  shorts crowded names.

Signals we deliberately **do not** use: **news sentiment** — not for want of data, but
because we cannot reach it. Tetlock-Saar-Tsechansky-Macskassy (2008), *on S&P 500
firms*, publish the cost curve themselves: +7.6%/yr at 5bp, +0.25% at 8bp, **−4.4% at
10bp**. The underreaction is 7.5bp after one day and 2.4bp after two; RavenPack — who
*sell* the data — concede large-cap news effects "decay rapidly (less than 2 days)".
Our features are lagged one bar and fill at the next open, so the earliest we can act
is the **D+2 open**: the alpha is gone before we arrive. Fang-Peress (2009) put the
media premium in small, low-coverage, high-idiosyncratic-vol names — every
conditioning variable points away from this universe. Also excluded: classic PEAD/SUE
(dead in large caps since ~2006, Martineau 2022).

**How a trade works.**

1. Each day, for every name: rank it cross-sectionally on each present signal
   (percentile rank), combine the ranks with tunable weights (`w_insider`,
   `w_rating`, `w_ptrev`, `w_dtc`) — an equal-weighted rank composite resists
   overfitting far better than ML stacking at this scale (Blitz;
   Israel-Kelly-Moskowitz) — then re-rank into one composite score in [0, 1].
2. A name **qualifies** to be bought when its composite rank is at or above `q_in`
   (top quantile). Only *eligible* names are ranked at all — a name needs at least
   `min_events` insider+rating events in the trailing `insider_window`, and one that
   does not is **dropped from the cross-section entirely**, not merely blocked from
   buying. (Ranking it and then refusing to buy it would be the worst of both: it
   could never be held, yet it would still take up rank space and push every other
   name's percentile around.) **Hold** under **TopK-Drop hysteresis**: sell when the
   rank falls below the wider `q_out` band, or when the name drops out of the ranked
   universe altogether, with a `max_hold` time-stop backstop. The `q_in`–`q_out` gap
   damps turnover.
3. **Qualifying is necessary, not sufficient — the book has only `max_positions`
   slots.** All names share one cash account, so a new name is bought only when a slot
   is free, and a slot frees only when a holding is sold. If more names qualify than
   there are slots, the highest-ranked ones win; the rest wait on the bench
   (`slot_blocked` in the diagnostic). With `displace`, a qualifying newcomer that
   outranks the weakest holding may take its slot outright instead of waiting.
4. Long only, no stops — the rank-exit band is the exit thesis. Fills are at the next
   bar's open; between rotations, weights are left to drift rather than trimmed daily,
   so trading stays event-driven and the fee drag stays small.

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
| `ptrev_halflife` | 21 | EMA half-life (bars) for the price-target revision stream |
| `w_insider` / `w_rating` / `w_ptrev` / `w_dtc` | 0.4 / 0.3 / 0.3 / 0.3 | composite weights (a weight of 0, or a missing feature, ablates that signal) |
| `q_in` / `q_out` | 0.8 / 0.6 | qualify above `q_in`, sell below `q_out` (hysteresis band) |
| `max_hold` | 126 | time-stop backstop (bars) — see below; it only bites in the band |
| `min_events` | 1 | eligibility floor: insider+rating events in the window. Below it, a name is dropped from the ranking |
| `max_positions` | 10 | slots in the book. Each is a fixed `1/max_positions` of the capital, so fewer names qualifying means idle cash, never a concentrated bet |
| `displace` | false | when the book is full: `false` = a qualifying newcomer waits for a vacancy; `true` = it evicts the weakest holding if it outranks it |

**What `max_hold` is actually for.** It is the least load-bearing parameter here, and
it exists for exactly one failure case: a name **parked in the hysteresis band**. Such
a name is not strong enough to re-qualify as a fresh buy, but not weak enough to trip
`q_out`, so nothing else can ever sell it — while the evidence that bought it quietly
decays out of the trailing windows (the insider purchase ages past `insider_window`;
the upgrade fades out of the EMA). You end up holding on expired evidence. `max_hold`
is the eviction notice. It deliberately **cannot** evict a name currently ranked at or
above `q_in`: that name has re-qualified on *fresh* evidence, and an old clock has no
business throwing it out.

**Data.** `data.features: [insider, ratings, price_target, short_interest]` attaches
the alt-data matrices to the panel at load time. Populate them by running
**`notebooks/03_altdata_update.ipynb`** — that notebook is the single entry point for
alt-data ingestion, incremental *and* historic (`MODE = "backfill"`), and it carries
the coverage audit. Data is ingested from 2020, but the strategy **trains from 2022**
(`data.start`): the two are separate knobs, so widening the window later is a
one-line config change, not a re-ingestion. Note that ~3 years of train data is
short — read the DSR and the sub-period table before believing a headline number.

**A missing sleeve is dropped, and so is a *thin* one.** A signal is used only if its
weight is positive **and** it covers at least half the universe
(`strategy._MIN_COVERAGE`); otherwise its weight is renormalized away. That floor is
not decoration. Without it, a barely-ingested table still passes a naive "any data?"
check: the handful of covered names get real percentile ranks while everyone else
takes the neutral `.fillna(0.5)`, so the sleeve silently becomes a large arbitrary
tilt on those few names and pure noise for the rest. This shipped for real —
`short_interest` once covered 3 of 100 names — which is why `03_altdata_update.ipynb`
now audits breadth and depth, not just freshness.

**Troubleshooting the buckets — why is this name in (or out)?**

`strategy.explain(panel, symbol, **params)` returns a per-bar audit trail: the
composite, the rank the bands actually compare against, each sleeve's weighted
contribution, the raw point-in-time events, and the resulting book. `signals()` and
`explain()` share one core (`_compose` / `_select`), so the diagnostic **cannot drift
from what the backtest traded** — a test asserts they agree. §4 of the notebook renders
it (rank + bands + real positions; the attribution stack; the triggering events), and
§4b shows every name's state at once.

Four behaviours look like bugs and are not:

- **A high rank does not get you bought.** The book has `max_positions` slots and they
  may all be taken. `slot_blocked` marks a name that qualified and got nothing, and it
  is the honest answer to "its rank is 0.95, why do I not own it?" — the money is
  already spent. This state did not exist under per-symbol cash accounts, where every
  name had its own money and a qualifying name was *always* bought. It is the visible
  face of the capital constraint, and its frequency is worth watching: if names are
  blocked constantly, `max_positions` is the binding parameter, not `q_in`.
- **A name can cross with no event of its own.** The score is a *cross-sectional*
  rank, so a name rises when its peers fall. `own` / `peer` split every bar's rank
  change into exactly those two causes (they sum to it with zero residual), and
  `own_event` says whether anything happened to this name. In a real run, **29% of one
  name's band crossings were pure peer drift.** If you remember one thing, remember
  this one.
- **A name's own value moves with no new event.** The insider signal is a trailing sum
  over a rolling window ÷ trailing dollar volume, so an old buy *rolling off the back
  of the window* moves the score by itself. "Own-driven" and "an event happened" are
  different claims, and are reported separately.
- **A missing signal is a neutral 0.5, not a zero.** `filled_<k>` marks bars where the
  name had no data for a sleeve and was handed the median rank. It looks scored; it
  carries nothing.

**A weight is not a position.** A held name is bought once and then deliberately *not*
re-trimmed — not even when other names rotate around it — and is sold **whole in one
order** only when its own rank leaves the mask. So a winner grows past its slot and a
loser shrinks below it. That keeps trading event-driven *per name* and the fee drag
small (a dense daily rebalance produced ~20x the orders for the same holdings; the
engine's `rotation_targets` masks per name, not per bar), but it means the realized
book drifts away from equal weight, and a runaway winner can come to dominate. The
notebook reports top-name concentration for exactly this reason. A name therefore shows
one buy and one sell per hold in `bt.trade_records` — several sells for one name means
it genuinely left the book and re-entered, not a partial trim. For what was *actually*
held, read `bt.trade_records`, never the weight frame.

**The hurdle: beat equal-weight buy-and-hold, or there is no point.**

A long-only strategy in a rising market earns a positive Sharpe ratio *merely by being
invested*. So an absolute Sharpe cannot tell stock-picking skill from market exposure,
and optimizing it actively rewards the wrong thing. This is not theoretical — the
previous version of this strategy maximized absolute Sharpe, and the optimizer duly
learned to imitate the index: it drove `q_in` to the least selective value permitted,
used `insider_window` as a back door to widen the eligible universe, and put its
largest weight on the sleeve with the *worst* stock selection. It posted a holdout
Sharpe of 0.94 and still **lost to buy-and-hold on both train and holdout**.

So the reference for everything here is an equal-weight buy-and-hold of the same names
over the same window, paying the same costs (`backtest.benchmark`). The optimizer
maximizes `info_ratio` and `excess_cagr` *against that benchmark*
(`optimize.OBJECTIVES`); absolute Sharpe is still recorded, as a diagnostic, never as
a target. Notebook §10a is the pass/fail gate.

`excess_cagr` is **per year**, not cumulative, and that matters more than it sounds.
A cumulative excess return grows with the length of the window, so it quietly rewards a
long backtest for being long — which makes it useless for ranking walk-forward folds of
unequal length against one another, exactly what the fold-picking code does.

**The book is capital-constrained, and that fixed the strategy's biggest defect.**
Earlier versions ran on the engine's per-symbol cash accounts, where every name has its
own money. That is a disaster for a cross-sectional ranker: the cash sitting against a
name the strategy is *flat* in can never be spent elsewhere, so holding 40 of 100 names
means being ~30% invested against a 100%-invested benchmark. It forfeits most of the
market's return to buy a selection edge, and it cannot win that trade — 500 trials of
v4, and not one of them beat buy-and-hold.

Worse, it diluted the edge at the same time. Measured with the engine out of the way (a
fully invested, monthly-rebalanced portfolio on the TRAIN window), the annual active
return by concentration was: **top 10% of names +5.8%, top 20% +4.6%, top 30% +1.7%,
top 40% +0.6%** — and the top 40% is exactly what the strategy held. So it spread the
signal across four times as many names as it could survive.

The strategy now runs in **portfolio mode**: it returns a `weights` frame, so all names
share ONE cash account, split into `max_positions` slots of `1/max_positions` each. A
new name can only be bought by selling an existing one. `displace` decides what happens
when a better name arrives and the book is full — wait for a vacancy, or evict the
weakest holding — and is *searched*, not assumed, because concentration argues for
chasing the top of the ranking and turnover costs argue against it.

One honest caveat: with fractional shares the whole model is scale-invariant, so the
`init_cash` figure is a **reporting convention, not a constraint**. $10,000 and
$1,000,000 give identical returns. What actually binds is the shared pot and the slot
count.

**And check the signal before the backtest.** §4c measures the ranking directly, with
the engine out of the way: the rank information coefficient and the top-minus-bottom
decile spread against *cross-sectionally demeaned* forward returns. Demeaning removes
the market, so what is left is stock selection and nothing else. It runs in seconds,
it is scored per sleeve, and a sleeve with a negative decile spread is not merely
useless — it is dragging the composite the wrong way.

**Backtest honestly.** `informed_flow.ipynb` trains on 2020–2024 and keeps
2025–2026 as an untouched holdout, and carries a validation block: costs-on plus
a cost-doubling stress and breakeven-bps; ~4-fold walk-forward; the Deflated
Sharpe Ratio over the Optuna study; parameter-neighborhood stability; a
sub-period table; a random-entry baseline; and a minimum-trades admissibility
filter on the Pareto front. Report `n_trades` beside every Sharpe.

The holdout section also renders `plots.plot_holdings(panel, pf)` — a
**strategy-specific** figure that lives beside `strategy.py` in
`strategies/informed_flow/plots.py`, not in `trading_simple.plots`, because "in
the book vs out" only means something for a rotating multi-name portfolio. It
puts four x-aligned panels on one screen: the **close price** of every name the
book ever held (faded throughout, bold over the real holding spans from
`bt.trade_records`, with `▲` buy / `▼` sell marks), the **portfolio composition**
(each name's weight as a stacked area with cash on top, shares and dollar value in
the hover — where the no-rebalance drift is visible), the **cumulative realized
P&L** decomposed per name against the total and cumulative fees, and the shared
book's **portfolio value**. Where `plot_equity` shows *how* the book did,
this shows *which names* did it.

**Caveats.**

- **Survivorship.** yfinance returns analyst data only for currently-covered names, so
  *both* analyst sleeves (ratings and price targets) carry residual survivorship bias,
  even though prices and SEC insider data do not. Claims are restricted to the liquid
  universe.
- **The price-target fields are undocumented.** They arrive because yfinance forwards
  unknown JSON keys from what is (per Benzinga's own client spotlight) a commercial
  Benzinga feed; Yahoo could withdraw them without notice. Ingested rows are durable,
  so a *backtest archive* survives — only future ingestion is exposed. The drop-in paid
  fallback is the same feed via Polygon/Massive's Benzinga add-on (~$99/mo, back to
  2011).
- **Short sample.** Training on 2021-2024 is ~4 years, minus up to ~9 months of
  `insider_window` warm-up. That is *less* evidence than the original 5-year window,
  and the DSR will read lower as a direct result — that is the price of regime
  relevance, not a regression. Prefer consistency across the sub-period table to a high
  headline, and remember the data is ingested back to 2020 if you want to widen
  (`data.start` and the ingested history are separate knobs on purpose).
