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
- **How recently any of them acted** (`recency`, added in v21). Not a fifth source but a
  **decay on the other four**: each of them reads its signal through a trailing window or
  an EMA, and *inside* that window an event's age is invisible — a three-day-old insider
  buy scores exactly like a three-month-old one. This sleeve ranks bars since the name's
  last insider or rating event, so freshness is priced explicitly, by a weight the study
  sizes and the sweep can ablate. It exists because that information used to reach the
  book by accident, through the `min_events` filter, and did real damage; see **The
  eligibility cliff** below.

**A price-regime overlay.** The four sleeves above are all
*alternative-data* signals; none of them knows anything about the stock's own price
action, so the ranker will happily surface a name with a strong insider/analyst footprint
that is nonetheless drifting sideways or falling. Two **price-derived** sleeves add that
missing dimension as a *confirmation tilt*:

- **MA-trend** — `SMA(trend_fast) / SMA(trend_slow) − 1`, a fast-over-slow moving-average
  crossover that reads the trend's slope (smoother than a single-MA distance). Names in a
  clean uptrend (fast above slow) rank high, sideways names (fast ≈ slow) land mid-pack,
  downtrends rank low.
- **Wilder RSI** — `RSI(rsi_period)`, the balance of up-closes to down-closes. Persistent
  buying ranks high, choppy/horizontal names sit near the middle, persistent selling low.

Both are computed from `panel.close` alone — no new alt-data, no engine change — and enter
the composite exactly like every other sleeve, weighted by `w_trend` / `w_rsi` (both `0.0`
in `config.yaml`: **off by default and not searched**, because the v13-vs-v14 controlled
holdout ablation put their effect at ~zero — holdout info ratio 2.02 → 2.02 — and the
walk-forward folds kept each of them in only 1 of 4; the sweep grid's `[0.0, 1.0]` columns
keep testing them out of sample). A weight above `0` enables a sleeve and renormalizes the
rest. One real limitation: the
tech sleeves are ranked **only within the alt-data-eligible cross-section**, so they can
down-rank a name that has insider/analyst coverage but a weak price trend — they cannot
pull in a pure price-uptrend name that has no alt-data at all. They confirm the alt-data
ranking; they do not replace it.

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
   `min_events` insider/rating event days in the trailing `insider_window`, and one that
   does not is **dropped from the cross-section entirely**, not merely blocked from
   buying. (Ranking it and then refusing to buy it would be the worst of both: it
   could never be held, yet it would still take up rank space and push every other
   name's percentile around.) `min_events` is a **universe floor and nothing else** —
   pinned at `1`, deliberately unsearched, for reasons the cliff section below explains
   at length. **Hold** under **TopK-Drop hysteresis**: sell when the rank falls below the
   wider `q_out` band, or when the name drops out of the ranked universe altogether, with
   a `max_hold` time-stop backstop. The `q_in`–`q_out` gap damps turnover.
3. **Qualifying is necessary, not sufficient — the book has only `max_positions`
   slots.** All names share one cash account, so a new name is bought only when a slot
   is free, and a slot frees only when a holding is sold. If more names qualify than
   there are slots, the highest-ranked ones win; the rest wait on the bench
   (`slot_blocked` in the diagnostic). A qualifying newcomer that outranks the weakest
   holding **by `displace_margin`** may take its slot outright instead of waiting; at
   `displace_margin = 1.0` — the shipped value — that can never happen and every
   newcomer waits.
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

**Parameters** (the values are what `config.yaml` **ships** today; a study pick overwrites
them there only when §7's "Promoting a pick" cell is run):

| param | shipped | meaning |
|---|---|---|
| `insider_window` | 63 | trailing bars for insider dollars, the dollar-volume normalizer, and eligibility — **triple duty**, so a pick here is not a clean statement about the insider horizon |
| `rating_halflife` | 21 | EMA half-life (bars) for the rating-change stream |
| `ptrev_halflife` | 21 | EMA half-life (bars) for the price-target revision stream |
| `w_insider` / `w_rating` / `w_ptrev` / `w_dtc` | 1.0 / 1.0 / 0.5 / 1.0 | composite weights; only their ratios matter, `w_dtc` is the fixed numeraire. A weight of 0 (grid only — the search floors at 0.25), or a missing feature, ablates that signal |
| `w_recency` | 1.0 | weight on the coverage-recency sleeve (bars since the last insider/rating event, negated). It has **no window of its own**, so staleness is decoupled from `insider_window` |
| `trend_fast` / `trend_slow` | 21 / 126 | fast / slow MA legs (bars) of the price-derived MA-trend crossover sleeve (`valid()` requires `trend_fast < trend_slow`) |
| `rsi_period` | 14 | Wilder-RSI lookback (bars) for the RSI market-condition sleeve |
| `w_trend` / `w_rsi` | 0.0 / 0.0 | weights on the MA-trend and RSI sleeves — **off by default and unsearched** (ablation measured ~zero holdout effect); the sweep grid still tests `1.0` |
| `q_in` / `q_out` | 0.75 / 0.45 | qualify above `q_in`, sell below `q_out` (hysteresis band). Both v13 and v14 converged on exactly this pair from opposite sleeve mixes |
| `max_hold` | 126 | time-stop backstop (bars) — see below; it only bites in the band |
| `min_events` | 1 | eligibility floor: insider/rating event days in the window. Below it, a name is dropped from the ranking. **Pinned, not searched** — see below |
| `max_positions` | 15 | slots in the book. Each is a fixed `1/max_positions` of the capital, so fewer names qualifying means idle cash, never a concentrated bet. The search floor is 10, and every study so far has picked its floor — see **Status** below |
| `displace_margin` | 1.0 | when the book is full, how far a challenger must out-rank the weakest holding to take its slot (one rank place in ~91 names ≈ 0.011). `0.0` evicts on any epsilon; `1.0` is unreachable, so every newcomer waits for a vacancy. One axis, no separate on/off flag — see below |
| `retrim_k` | 2.0 | concentration cap: a held name that grows to `retrim_k ×` its slot is trimmed back to it. Fixed policy, never searched — see below |

**What `max_hold` is actually for.** It is the least load-bearing parameter here, and
it exists for exactly one failure case: a name **parked in the hysteresis band**. Such
a name is not strong enough to re-qualify as a fresh buy, but not weak enough to trip
`q_out`, so nothing else can ever sell it — while the evidence that bought it quietly
decays out of the trailing windows (the insider purchase ages past `insider_window`;
the upgrade fades out of the EMA). You end up holding on expired evidence. `max_hold`
is the eviction notice. It deliberately **cannot** evict a name currently ranked at or
above `q_in`: that name has re-qualified on *fresh* evidence, and an old clock has no
business throwing it out.

**What `displace_margin` is actually for.** `q_in`, `q_out` and `max_hold` all gate a
name's rank **level**. Displacement is an **ordering** comparison — "does anyone outrank
my weakest holding?" — and *none of them can see it*. On the v16 pick's holdout book that
asymmetry was the whole story: **122 of 127 exits were displacements**, 5 were
`q_out`/eligibility, and `max_hold` fired zero times. So the two brakes the strategy
documents at length were dormant, and the undocumented comparison was doing all the
trading. Worse, it was doing it on nothing: the median challenger won by ~2 rank places
out of ~91 eligible names and 16% won by a single place, and because `score_pct` is a
*percentile* a name can lose its slot on a bar where its own composite did not move at
all (MRVL, 2025-12-15: composite bit-identical to the prior bar, one place slipped, whole
position sold, re-bought seven bars later at a *higher* score). An event study over all
121 holdout evictions found the challenger's forward return minus the evicted name's
indistinguishable from zero at every horizon — −0.28% at 5 bars, −0.45% at 10, +0.04% at
21, +1.84% at 63, win rate ~50%, p ≥ 0.67. The swaps were a coin flip paying two legs of
costs each. `displace_margin` is the missing hysteresis on the ordering.

**Why it is one knob and not two.** There used to be a separate `displace` boolean in
front of the comparison, and study v18 proved the two were the *same axis*: `score_pct`
is a percentile in (0, 1], so `challenger ≤ weakest + 1.0` holds for every possible pair
of ranks, and a margin of 1.0 produces a **bit-identical held mask** to `displace=False`.
That is not a resemblance — it is an identity, pinned by
`test_margin_one_is_exactly_the_old_displace_off_flag`.

Keeping both spellings did real damage:

* **A dead parameter.** Every trial that drew `displace=False` carried a
  `displace_margin` that could not do anything, so the value was unidentifiable. **33 of
  v18's 55 Pareto-front trials were one strategy** recorded under 14 different margins,
  all scoring `info_ratio 1.056178 / excess_cagr 0.097963 / 43.458317 trades per year` to
  the last digit, and the pick reported a `displace_margin` that had had no effect in the
  trial that earned it. Clone padding does **not** distort the min–max normalization
  `pick_from_front` applies — min and max do not move when a point is repeated — but it
  does corrupt every statistic taken *over* the front: `boundary_report`'s edge shares are
  plain means over rows, so 33 clones vote 33 times to widen whatever bound they carry.
  `optimize.distinct_front` now collapses ties before either is computed.
* **An unreachable region.** The objective was still improving at the margin's old `0.15`
  ceiling, so "brake harder" could only be expressed by flipping a categorical that
  jumped straight to infinity — leaving everything between 0.15 and 1.0 unexplorable.

One continuous knob is strictly more expressive *and* one dimension smaller. `1.0` ships,
which is exactly the old `displace: false` the config shipped before, bit for bit — so
the collapse changes the search space, not the strategy. The sweep grid `[0.0, 1.0]` is
now the honest ablation: chase the ranking, or pure hysteresis.

**How the choice depends on book width.** Displacement is not good or bad in itself — it
pays only where the *marginal* name in the book still carries signal. Toggling it with
everything else fixed at the v18 pick, on the train window, excess CAGR runs:

| slots | 5 | 8 | 15 | 17 | 20 | 30 |
|---|---|---|---|---|---|---|
| `margin 1.0` (never displace) | 0.061 | **0.074** | **0.090** | **0.090** | **0.064** | −0.007 |
| `margin 0.0` (chase the top) | **0.069** | 0.026 | 0.021 | 0.019 | 0.021 | −0.007 |

Chasing pays at 5–8 slots and stops paying from ~15, which is exactly what the
concentration measurement below predicts: active return is +5.8%/yr in the top decile but
+1.7% at the top 30% and +0.6% at top 40%. A 5-slot book *is* the top decile, so keeping
it exactly sorted chases real signal; a 15–30 slot book's boundary sits where the edge is
already gone, so re-sorting it daily pays costs for noise. At 30 slots the two rows are
identical because the book can no longer fill (median 23 names held), so displacement has
nothing to displace. **Read a margin pick as a statement about book width**, not as a
free parameter — and note v18's four walk-forward folds split 3–1 *against* the main
study's choice, so this is not a converged decision.

**The eligibility cliff — the same bug one layer down (v21).** `displace_margin` fixed a
*rank ordering* with no hysteresis. `min_events` was a *rolling integer count* with no
hysteresis, and it was worse, because it did not merely rotate the book — it emptied
slots. Eligibility is a rolling sum of event days over `insider_window`, so the count
steps **down** when an old filing ages out, on a bar where nothing whatsoever happened to
the name; a held name that falls below the floor is force-sold, since the model has
stopped scoring it. Measured on the v20 pick (which drew `insider_window = 31`,
`min_events = 3`):

| | train 2018-2024 | holdout 2025-2026 |
|---|---|---|
| exits from **eligibility loss** | **140 / 196 (71%)** | **19 / 29 (66%)** |
| …a single event ageing out (3 → 2) | 95% | 95% |
| …median `score_pct` on the last bar held | **0.94** | 0.94 |
| …still above `q_in` when sold | 92% | 89% |
| exits from `q_out` | 52 | 8 |
| exits from displacement | 4 | 2 |
| exits from `max_hold` | 0 | 0 |

So the three brakes this README documents at length governed under a third of exits
between them, and **the SEC/Yahoo filing calendar governed the rest**. Eligibility was
lost 1,606 times on train, and 47% of those names were eligible again within 21 bars —
14% within 3. AMZN's 2-bar holdout position is the mechanism in miniature: PLTR's count
fell 3 → 2 on 2025-11-05 (after a 180-bar hold) and freed a slot, AMZN's rose 2 → 3 the
same bar and took it, then on 11-07 AMZN's fell back while PLTR's recovered and they
swapped again. Four legs of costs, 31 bars apart, and **neither name's composite moved**.

Two things follow, and only the second is obvious. The obvious one: this is turnover
bought with no information. The other: the forced exits were *not* pure noise. An event
study over them (cross-sectionally demeaned forward returns, replacement minus sold)
gives **+1.04% at 5 bars (p = 0.052)** and **+1.57% at 10 (p = 0.030)**, decaying to
+0.62% at 63 (p = 0.78). Coverage recency carries a little real signal — the strategy was
harvesting it through a bookkeeping cliff that also paid four legs of costs on every
calendar coincidence.

The fix is therefore not a grace period but a **sleeve**: `w_recency` prices staleness
continuously and inside the ranking, so a name whose coverage goes stale drifts *down*
the composite and leaves through `q_out` on an ordering, while `min_events` shrinks to a
pinned universe floor (`1` — "has the model seen anything at all?"). Unsearched, and that
matters: leaving the threshold in the space lets the optimizer re-discover the cliff,
which is exactly what v20 did. At the shipped v21 parameters the exit mix inverts:

| exits by cause | train | holdout |
|---|---|---|
| `q_out` | 228 (87%) | 60 (92%) |
| eligibility loss | 35 (13%) | 5 (8%) |
| sold while still ranked ≥ `q_in` | 13 (5%) | 5 (8%) |

**Read that as a mechanism check, not a result.** It says the strategy now exits for the
reason it claims to; it says nothing about whether it makes more money, and it cannot,
because those are config defaults rather than a tuned pick — the v20 journal cannot be
read against v21's space at all. Every de-cliffing variant tested at *v20's* pick scored
below it (train `excess_cagr` 0.446 → 0.25–0.32, holdout 0.780 → 0.28–0.61), and that
comparison is biased beyond use: `insider_window = 31` with `min_events = 3` is precisely
the pair that maximizes flicker, chosen by 1,000 trials that could see and exploit this
exit channel. Bolting a fix onto co-adapted parameters measures the fix against its own
removal. **Run study v21 and the four walk-forward folds before quoting any number.**

**Data.** `data.features: [insider, ratings, price_target, short_interest]` attaches
the alt-data matrices to the panel at load time. Populate them by running
**`notebooks/03_altdata_update.ipynb`** — that notebook is the single entry point for
alt-data ingestion, incremental *and* historic (`MODE = "backfill"`), and it carries
the coverage audit. Alt-data is ingested from 2018 (the coverage ledger's `covered_from`)
and the strategy **trains on 2018–2024** (`data.start` / `data.end`) over the
`max_symbols: 110` best-covered LiquidLeaders names; ingestion depth and the train window
are separate knobs, so narrowing the window later (to skip the 2020 regime, say) is a
one-line config change, not a re-ingestion. Read the DSR and the sub-period table before
believing a headline number.

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

**And the book-wide counterpart: which *rule* is trading this?** `strategy.exit_causes(
panel, **params)` returns one row per completed hold with the rule that ended it —
`q_out`, `ineligible`, `max_hold` or `displaced` — recorded inside `_select` as each
decision is made, so it cannot disagree with the book that traded. §10g runs it on every
holdout, and it is the cheapest check in the notebook: **both of this strategy's worst
bugs were one `value_counts()` away from obvious and invisible in every other statistic.**
Displacement owned 122 of 127 exits on the v16 pick; the eligibility cliff owned 140 of
196 on v20. Neither moves an equity curve in a way you would notice, and both mean the
strategy is not the one the docstrings describe. Read the `cause` column first, then
`score_held` — a name sold while still ranked near the top was not sold by a decayed
thesis. If one mechanism owns most of the exits, **that mechanism is the strategy**.

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

**A weight is not a position.** A held name is bought once and then left to drift — not
re-trimmed when other names rotate around it — and is sold **whole in one order** only
when its own rank leaves the mask. So a winner grows past its slot and a loser shrinks
below it. That keeps trading event-driven *per name* and the fee drag small (a dense
daily rebalance produced ~20x the orders for the same holdings; the engine's
`rotation_targets` masks per name, not per bar), but it means the realized book drifts
away from equal weight, and a runaway winner can come to dominate.

**The concentration cap (`retrim_k`, default `2`).** The one deliberate exception to
"left to drift": when a held name's realized book weight reaches `retrim_k ×` its
`1 / max_positions` slot (double it, by default), the engine trims it back to the slot on
the next bar and the freed cash rotates into the next name the selection picks. This is
risk control, not signal — it exists so a single lucky pop cannot come to dominate the
book and crown a parameter set. It is a **fixed policy, never optimized** (kept out of the
Optuna space and the sweep grid on purpose); set `retrim_k: null` in the config to disable
it and recover the pure drift behavior. The cap lives in the engine
(`backtest._retrim_loop`) because realized weight is only knowable inside a run. The
notebook reports top-name concentration so you can see the cap working (or, with it off,
how far a winner would have run).

So in `bt.trade_records` a name shows one buy and one sell per hold, **plus a partial sell
each time the cap trims it** — several sells for one name is either a genuine exit-and-
re-entry or a `retrim_k` trim, not necessarily the former. For what was *actually* held,
read `bt.trade_records`, never the weight frame.

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
scores `excess_cagr` (maximized) against `tracking_error` (minimized), both *against
that benchmark* (`optimize.OBJECTIVES`), and picks only among front trials that beat it;
absolute Sharpe is still recorded, as a diagnostic, never as a target. Notebook §10a is the pass/fail gate.

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
new name can only be bought by selling an existing one. `displace_margin` decides what
happens when a better name arrives and the book is full — wait for a vacancy, or evict
the weakest holding, and by how clear a margin — and is *searched*, not assumed, because
concentration argues for chasing the top of the ranking and turnover costs argue against
it. Which of the two wins turns out to depend on `max_positions`; see the table above.

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

**Backtest honestly.** `informed_flow.ipynb` trains on 2018–2024 (`data.start` /
`data.end`) and keeps 2025–2026 as an untouched holdout, and carries a validation block: costs-on plus
a cost-doubling stress and breakeven-bps; a 4-fold walk-forward; the Deflated
Sharpe Ratio over the Optuna study; parameter-neighborhood stability; a
sub-period table; a random-entry baseline; and a minimum-trades admissibility
filter on the Pareto front. §9b re-scores the *tuned* ranking's IC on train **and**
holdout. Report `n_trades` beside every Sharpe.

**A validation instrument only works if it tests what you ship.** §10b walk-forward
re-picks each fold from the `sweep` grid, so anything that grid cannot express is a
strategy the walk-forward can never validate. That silently broke: the grid pinned
`max_positions=10`, `q_in ∈ {0.8, 0.9}`, `q_out=0.6` while the study shipped
`15 / 0.75 / 0.45` — **not one shipped value was reachable**, so two 3,000-trial studies
could disagree about whether the insider sleeve belongs at all, with no instrument able
to catch it. The `sweep` block's shape knobs are now pinned to the shipped values and
`tests/test_informed_flow.py::test_sweep_grid_contains_shipped_shape` keeps them there.
Its free dimensions are exactly the sleeve on/off columns, which makes §10b the
**sleeve referee** — sleeve weights are set from its folds, never from the holdout
(selecting on the holdout burns it as an honest test).

**A validation instrument also has to end in an action.** §10d asks whether the picked
*values* are load-bearing: it perturbs one knob at a time by ±5% and ±10% of that knob's
own searched span (`optimize.sensitivity`) and gives each one a verdict with the move it
implies (`optimize.stability_report`) — `inert` (pin it, spend the trials elsewhere),
`plateau` (ship it), `cliff` (step to the neighboring value, which the cell prints),
`spike` (do not ship this pick), `off-center` (refer it to §10b). It replaced a cell that
re-ran §5's *global* ablation grid, called it a neighborhood, and drew a heatmap of a
**pinned** parameter — it never scored a single point near the pick, and its plateau rule
(`median > 0.5 × max`) inverted whenever the best neighbor was negative. On the v15 pick
the answer is blunt: six knobs are locally inert, and `max_positions` is a cliff sitting
on its own low bound — the edge is a concentration effect (+27%/yr at 8 slots, +13% at
10), not a parameter set.

**What the walk-forward says — read this before trusting the holdout.** The first fixed
§10b (2026-07-28) re-picked each fold from the *coarse grid* and reported mean in-sample
excess **+10.94%/yr** against mean out-of-sample **−3.49%/yr** (`t=4.65, p=0.02`, 2 of 4
folds beating buy-and-hold). That decay was **an artifact of the instrument**: a drift gate
added to §9 showed the grid was running a strategy 11 of 12 knobs away from the pick. §10b·i
was rebuilt (2026-08-01) so each fold runs its own 1,000-trial study and picks with the same
`pick_from_front` §7 uses; on the v15 space that gives mean IS **+45.4%/yr** → mean OOS
**+33.6%/yr** (median +23.9%), `t=0.44, p=0.69`, **3 of 4 folds** beating buy-and-hold, mean
OOS info ratio +1.02 — with a fold spread of +105 / −18 / +14 / +33 on half-year windows.
Read that as "no decay *detected* at n=4", not as a clean bill of health, and re-run the
folds on the current space (v22) before quoting them for it.

**Loading the train panel is not a free choice.** Load it on its **own** window; do not
load full history and slice back. A feature *column* exists only where the query window
returned rows, so widening to include the holdout gives a name whose only insider activity
is in 2025 a real all-zero column across 2020–2024 where the train-only load leaves it
all-NaN — and zero is a genuine bottom rank while NaN is the neutral 0.5 fill. Measured on
this panel: train `excess_cagr` 0.084 → 0.104 from that difference alone. Same values,
different missingness, holdout availability leaking into training data.

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
- **Sample.** Training on 2018–2024 is ~7 years, minus up to ~1.5 years of
  `insider_window` warm-up at its 378-bar ceiling — and the analyst sleeves are thin before
  2020 (the price-target feed carries ~11k revisions/yr in 2018–19 against ~47k in 2024,
  across all ingested names). Prefer consistency across the sub-period table and the
  walk-forward folds to a high headline number. (`data.start` and the ingested history are
  separate knobs on purpose, so the window can be narrowed to skip the 2020 COVID regime
  without re-ingesting anything.)
- **The holdout is one regime.** 2025–2026 is a single ~1.5-year window in a strong bull
  tape (equal-weight buy-and-hold did ~+45%/yr), and a 15-name book has high variance.
  The *direction* of a holdout finding can be trusted far more than its *magnitude* —
  and per the walk-forward above, even the direction is currently unconfirmed.

---

## Study history — which journals may not be compared

Optuna journals are append-only and keyed by name, so a study whose *reachable model
space* changed is two different experiments sharing one file. **Bump `optuna.study`
whenever a `search:` bound, the objective, or the execution model changes**; the entries
below record why each bump happened, and are the reason a v9 trial cannot be ranked
against a v10 one. (This history used to live in `config.yaml`; the config now describes
only what runs today.)

| study | what changed | comparable with earlier? |
|---|---|---|
| v3 | maximized absolute Sharpe — rewarded market exposure, not stock selection | — |
| v4 | second objective was cumulative excess *return* | no |
| v5 | book became ONE shared cash account with `max_positions` slots (portfolio mode); second objective became **excess CAGR per year**. Under v4's per-symbol accounts the strategy was structurally ~30% invested against a fully-invested benchmark and could not win whatever it picked | **no** |
| v7 | execution: `rotation_targets` masks per NAME, not per bar — each name is bought once and sold whole, weights drift between its own events | **no** |
| v8 | engine realism: panel-priced costs, benchmark mask, ffill feature seeding | **no** |
| v9 | added the `retrim_k` concentration cap (2× the slot) | **no** |
| v10 | added the price-derived trend + RSI sleeves — a strict superset of v9's space | **no** |
| v13 / v14 | same space, opposite sleeve mixes (v13 kept insider at 1.5, v14 deleted it) and scored within 0.1 information ratio of each other — which is how we learned the objective is nearly **flat** over sleeve weights | v13 ↔ v14 yes |
| v15 | alt-data weight floors `0.0 → 0.25` (study sizes a sleeve, the sweep decides existence); `max_positions` low bound widened; sweep grids re-pinned to the shipped values | **no** |
| v16 | closes the open issue below: same space as v15's *later* trials, cleanly journalled | no |
| v17 | `insider_window` high bound widened `189 → 378`; **and** the new `displace_margin` dimension, added while this journal still held **zero trials** — so it cost no further bump, and v17 has only ever meant the space described above | **no** |
| v18 | `max_positions` low bound raised `5 → 15`. Worth reading as a pair with v17: v17 picked `displace=true` at 5 slots (93% of its trials, 35/35 of its front), v18 picked `displace=false` at 15+ (93%, 55/55) — the same objective, opposite answer, because the answer was always conditional on book width | v17 ↔ v18 **no** (different `max_positions` range) |
| v19 | the `displace` categorical **collapsed into** `displace_margin`, whose range widened `0.0–0.15 → 0.0–1.0`; `displace_margin = 1.0` is bit-identical to the old `displace=false`, so the shipped strategy is unchanged and only the search space is. One dimension smaller, and no longer contains a parameter that silently does nothing | **no** |
| v21 | the **`recency` sleeve** (`w_recency`, searched like any weight) is added, and **`min_events` loses its `search:`** and is pinned at 1. Both halves of one change: the searched threshold on a rolling count had become the strategy's dominant exit rule (see "The eligibility cliff"), and leaving it searchable would let the optimizer re-discover it. A new dimension and one removed; the shipped strategy changes too, not just the space | **no** |
| v20 | `max_positions` low bound `15 → 3`. Both journals are populated (v19 5,005 trials, v20 2,500) and both fronts are a **single dominating trial**, so their picks are not tie-breaks — but v20's pick sits at `max_positions = 3`, the new floor, continuing the unbroken run of floor-pinned picks (30 → 15 → 8 → 5 → 3). Read it as a statement about concentration, not a converged value, and note `excess_cagr = 0.446` on a 3-name book is exactly where a concentration artifact would show up | v19 ↔ v20 **no** |
| v22 | two changes at once: the `max_positions` low bound `3 → 10` (so the floor is a book, not a bet) **and** `max_symbols` `100 → 110` — a different panel and therefore a different benchmark. 1,936 trials run 2026-09-12; the pick again sits on the `max_positions` floor (see **Status** below) | v21 ↔ v22 **no** (space *and* panel differ) |
| v23 | same space and panel as v22; the **engine's objectives** changed from `info_ratio` ↑ / `excess_cagr` ↑ (Spearman 0.975 across v22's trials — a 2-point front) to `excess_cagr` ↑ / `tracking_error` ↓, and the pick only chooses front rows beating buy-and-hold. The engine refuses to load the v22 journal under the new labels | **no** (different objectives) |
| v24 | three v23 bounds that the front was pressing on: the alt-data sleeve weights (`w_insider`, `w_rating`, `w_ptrev`, `w_recency`) may reach **0.0** (were floored at 0.25, which forced the negative-IC insider sleeve and a floor-pinned `w_ptrev` into every pick), `rating_halflife` high `90 → 180` (52% of the front sat at 90), `max_hold` low `21 → 10` (44% sat at 21) | **no** (space differs) |

**Resolved (v15).** `max_positions`'s low bound had been widened `8 → 5` without a study
bump, so `informed-flow-1d-v15.log` held 4,457 trials that could only reach 8 and 412 that
reached below it — two spaces in one journal, and `boundary_report` was unreadable off it.
v16 closed it. Note what the unified `params:` block can and cannot catch: it compares the
config against itself, never against a journal written weeks ago, so **a bound change is
still only caught by a human bumping the study name**.

**Status (2026-09-12): v21 and v22 have both run — read v22 with three caveats.**

`informed-flow-1d-v21.log` holds 2,500 trials plus four 1,000-trial walk-forward folds, and
`informed-flow-1d-v22.log` — the space this config ships — holds **1,936** (1,818 complete,
118 pruned by the 15/yr floor). v22 changed two things at once and is therefore comparable
with v21 on neither axis: the `max_positions` floor moved `3 → 10` (v20's pick sat at 3, the
sixth floor-pinned pick in a row) and `max_symbols` moved `100 → 110`, which changes the
panel *and* the benchmark. The §10b `SLEEVE_KEYS` list also gained `w_recency`, so the fold
ablation now votes on the recency sleeve. The shipped `params:` values are still config
defaults, not v22's pick — §7's "Promoting a pick" cell writes a pick back only when run.

| v22 | |
|---|---|
| best `info_ratio` | **1.27** (trial 1322, excess **+14.4%/yr**): `insider_window 101`, `rating_halflife 10`, `ptrev_halflife 39`, `w_insider 1.5`, `w_rating 1.25`, `w_ptrev 1.0`, `w_recency 2.5`, `q_in 0.82`, `q_out 0.44`, `max_hold 35`, `max_positions 10`, `displace_margin 0.40` |
| best `excess_cagr` | **+17.5%/yr** (trial 1344, IR 1.19; `w_insider 0.25`, `w_ptrev 2.75`, otherwise as above) |
| median trial | IR 0.75, excess +8.4%/yr; 97.6% of complete trials beat buy-and-hold in-sample |
| Pareto front | 10 rows, **4 distinct outcomes** — the clones differ only in `max_hold` (35–246) and `displace_margin` (0.49–0.90) |

1. **`max_positions` sits on its floor again** — 10, in 100 of the top 100 trials. The
   study has said "fewer slots" at every floor it was given (30 → 15 → 8 → 5 → 3 → 10),
   because the edge is monotone in concentration (v15: 8 → 9 → 10 slots gave +27.2 → +17.5
   → +12.9%/yr). Searching it again will not converge; the honest move is to stop searching
   it and pin it as a risk policy, like `retrim_k`, with the concentration curve recorded
   once as the trade-off chosen.
2. **`max_hold` and `displace_margin` are inert at every good pick** — that is what the
   clone rows mean — so they only dilute NSGA-II. §10d said the same of `max_hold` on v15.
3. **`w_recency` took the largest weight in the composite**: 2.5–2.75 in every top-100
   trial, of a total near 9. The sleeve was added to *replace* the filing-calendar exit
   cliff; at this weight it can re-create calendar-driven turnover continuously, and the
   event study behind it found the information gone by bar 63. Run `exit_causes` and
   compare `trades_per_year` against the v21 pick before promoting it; §10b·ii's `0.0`
   column is the out-of-sample vote on whether the sleeve earns any weight at all.

Re-run the four walk-forward folds on the v22 space (§10b·i re-searches per fold, so it is
the instrument that can judge this space) before quoting any tuned number from it.
