# informed_flow — daily cross-sectional composite of alternative-data signals

**Thesis.** Three groups of informed participants leave a *public* footprint
that predicts individual-stock drift over weeks to months, and combining their
independent signals is more robust than betting on any one. This is the repo's
first **alternative-data** strategy and its first **daily cross-sectional
ranker**: each day it scores every name in a liquid universe on what insiders,
analysts, and short sellers are doing, and holds the top-ranked names long.

> **What ships today (v27, 2026-09-19).** Two of the five sleeves below carry the ranking, at
> fixed equal weights: **short interest** and **analyst rating changes**. Insider buying,
> price-target revisions and recency are still built (`strategy.py` is unchanged) but ship at
> weight `0.0`: on **beta-adjusted** forward returns they carry no information in this large-cap
> universe — analyst news is priced on the day it appears, `recency` is direction-blind, and the
> insider table is 96% sells. The measurement, the data defects found along the way and the v26
> result that prompted it are under **Status (2026-09-19)** near the end. The book is 30 slots,
> sold when a name leaves the top fifth of the ranking, scored with open fills and the commission
> priced at the real order size; **one** knob (`q_out`) is searched, on **one** objective.

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
them there only when `informed_flow_diagnostics.ipynb` §7 promotes a candidate that passed
its verdict):

| param | shipped | meaning |
|---|---|---|
| `insider_window` | 126 | trailing bars for insider dollars, the dollar-volume normalizer, and eligibility — **triple duty**. Pinned since v27: with `w_insider` at 0 only the eligibility job is left ("any insider or rating event in ~6 months" ranks ~473 of 500 names a day) |
| `rating_halflife` | 21 | EMA half-life (bars) for the rating-change stream. Pinned since v27 — measured inert: half-lives 5 / 21 / 63 give the same beta-adjusted IC |
| `ptrev_halflife` | 21 | EMA half-life (bars) for the price-target revision stream (sleeve off) |
| `w_insider` / `w_rating` / `w_ptrev` / `w_dtc` | 0.0 / 1.0 / 0.0 / 1.0 | composite weights; only their ratios matter, `w_dtc` is the fixed numeraire. A weight of 0, or a missing feature, ablates that signal. **None is searched since v27**: which sleeves exist is decided by a pre-registered rule on beta-adjusted IC (see **Status (2026-09-19)**), re-read by the diagnostics notebook §2a on every run |
| `w_recency` | 0.0 | weight on the coverage-recency sleeve (bars since the last insider/rating event, negated). It has **no window of its own**, so staleness is decoupled from `insider_window`. Off since v27: it is direction-blind and its IC is zero at 5–63 bars |
| `trend_fast` / `trend_slow` | 21 / 126 | fast / slow MA legs (bars) of the price-derived MA-trend crossover sleeve (`valid()` requires `trend_fast < trend_slow`) |
| `rsi_period` | 14 | Wilder-RSI lookback (bars) for the RSI market-condition sleeve |
| `w_trend` / `w_rsi` | 0.0 / 0.0 | weights on the MA-trend and RSI sleeves — **off by default and unsearched** (ablation measured ~zero holdout effect) |
| `q_in` / `q_out` | 0.90 / 0.80 | qualify above `q_in`, sell below `q_out` (hysteresis band). v13, v14 and the whole v26 front sat near 0.75 / 0.45 — but those studies paid pessimistic fills (~1% per leg), which makes "never sell" optimal whatever the signal says. Measured directly, only the **top fifth** of the ranking pays, so the band now sits there. `q_out` is **the one searched knob** (0.60–0.85); `q_in` is pinned |
| `max_hold` | 126 | time-stop backstop (bars) — see below; it only bites in the band. Pinned since v27 |
| `min_events` | 1 | eligibility floor: insider/rating event days in the window. Below it, a name is dropped from the ranking. **Pinned, not searched** — see below |
| `max_positions` | 30 | slots in the book. Each is a fixed `1/max_positions` of the capital, so fewer names qualifying means idle cash, never a concentrated bet. **Pinned as a risk policy** (every study that searched it picked its floor). 30 since v27: still the top 6% of ~470 ranked names, the same return per name as the top 15 on train, about half the name-specific luck. `backtest.order_notional` is `account / max_positions` — change them together |
| `displace_margin` | 1.0 | when the book is full, how far a challenger must out-rank the weakest holding to take its slot (one rank place in ~470 names ≈ 0.002). `0.0` evicts on any epsilon; `1.0` is unreachable, so every newcomer waits for a vacancy. One axis, no separate on/off flag — see below. Pinned since v27 |
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
**`notebooks/03_altdata_update.py`** (marimo) — that notebook is the single entry point for
alt-data ingestion (every run fetches only what each name is missing), and it carries
the coverage audit. Alt-data is ingested from 2018 (the coverage ledger's `covered_from`)
and the strategy **loads 2018–2024** (`data.start` / `data.end`) but is **scored only from
2020-07-01** (`optuna.score_from`), over the 500 names of `Liquid500` — a point-in-time
universe ranked on 2020-H1 dollar volume, so nothing is scored before its membership was
knowable. (Until v25 it was the 110 best-covered `LiquidLeaders` names, screened on *today's*
liquidity: the book ranked among ex-post winners.) The 2018–2020 bars only warm the rolling
windows. Ingestion depth, the load window and the scored window are separate knobs, so moving
the scored start is a one-line config change, not a re-ingestion. Read the DSR and the sub-period table before
believing a headline number.

**A missing sleeve is dropped, and so is a *thin* one.** A signal is used only if its
weight is positive **and** it covers at least half the universe
(`strategy._MIN_COVERAGE`); otherwise its weight is renormalized away. That floor is
not decoration. Without it, a barely-ingested table still passes a naive "any data?"
check: the handful of covered names get real percentile ranks while everyone else
takes the neutral `.fillna(0.5)`, so the sleeve silently becomes a large arbitrary
tilt on those few names and pure noise for the rest. This shipped for real —
`short_interest` once covered 3 of 100 names — which is why `03_altdata_update.py`
now audits breadth and depth, not just freshness.

**The floor holds bar by bar, and a stale level expires.** Breadth on a sleeve's best bar is
not breadth on every bar. From the 2026-07-15 settlement on, short interest stopped being
ingested for 395 of `Liquid500`'s names (the ingestion notebook defaulted to `LiquidLeaders`),
and they ranked on their 2026-06-30 days-to-cover for months with no error anywhere. Now
`features.py` masks a short-interest level older than 45 days (`_SI_MAX_AGE_DAYS`; no two FINRA
dissemination dates were ever more than 19 days apart), and `strategy._sleeve_rank` silences a
sleeve on any bar where it covers less than half the universe — so a stale table becomes a
silent sleeve (and, if it was the only one, cash), never a tilt toward the names still covered.
The cap also removed a historical defect: FINRA reports by ticker, and six names inherited a
different company's short interest across a rename (Weight Watchers' days-to-cover sat on WTW
for 1,005 days, CoreSite's on COR for 624, Genesis Healthcare's on GEN for 610). Re-scored with
both changes, v31's train alpha moves +10.08% → +10.28% (570 → 567 trades); every v31 number
above predates them.

**Troubleshooting the buckets — why is this name in (or out)?**

`strategy.explain(panel, symbol, **params)` returns a per-bar audit trail: the
composite, the rank the bands actually compare against, each sleeve's weighted
contribution, the raw point-in-time events, and the resulting book. `signals()` and
`explain()` share one core (`_compose` / `_select`), so the diagnostic **cannot drift
from what the backtest traded** — a test asserts they agree.
`informed_flow_diagnostics.ipynb` §2b renders it for the `FOCUS` candidate (rank + bands +
real positions; the attribution stack; the triggering events), and §2c shows every name's
state at once.

**And the book-wide counterpart: which *rule* is trading this?** `strategy.exit_causes(
panel, **params)` returns one row per completed hold with the rule that ended it —
`q_out`, `ineligible`, `max_hold` or `displaced` — recorded inside `_select` as each
decision is made, so it cannot disagree with the book that traded. The diagnostics notebook
runs it on every holdout (§5c), and it is the cheapest check there: **both of this strategy's worst
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
scores `excess_cagr` (maximized; the return *against that benchmark*) against the book's
own `volatility` (minimized) (`optimize.OBJECTIVES`), and picks only among front trials that
beat it;
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

One honest caveat: with fractional shares the simulation is scale-invariant, so the
`init_cash` figure is a **reporting convention, not a constraint** — $10,000 and
$1,000,000 give identical returns. **Real trading is not scale-invariant**, because of the $1
minimum commission: the engine spreads it over a $10,000 order by default (1 bp per leg), but a
30-slot book on a $30k account places $1,000 orders, where it is 10 bp. `config.yaml` therefore
declares `backtest.order_notional: 1000` (account ÷ slots), and the diagnostics notebook re-prices
it for the `ACCOUNT` you set there. (The v26 overlay pick placed 400+ orders a year of $70–670;
the backtest reported $19 of fees where a real $10k account would have paid ~$350.)

**And check the signal before the backtest.** The diagnostics notebook's §2a (train) and §5b
(holdout) measure the ranking directly, with the engine out of the way, per sleeve: the rank
information coefficient, and the forward return of the top `max_positions` names. One trap, which
this README used to fall into: a Spearman rank-IC **does not change** when the market's average is
subtracted, so "cross-sectionally demeaned" removes nothing from it — a ranking that merely leans
toward high-beta names looks skilled in a rising market (a pure-beta ranking scores ≈ +0.02 when
the market rises 20%/yr). The **fair check** therefore removes each name's trailing beta times the
market's forward return, and reads the IC per calendar year, inside three dollar-volume groups,
and by fifth of the ranking — a long-only book can only use information that sits at the top. Its
composite row is the **signal rule**, a hard gate, and the rule that decided which sleeves ship.

**Backtest honestly — two notebooks, two jobs.** Training runs 2018–2024 (`data.start` /
`data.end`, scored from 2020-07) and 2025+ is the holdout — **not untouched**: it has been read
under eight study versions (v13 → v26) and design changes followed those reads, so everything
measured on it alone is a *warning* in the verdict, not a gate.

- **`informed_flow.ipynb` optimizes, on the training panel only** (it never loads the
  holdout). It tops the study up to `optuna.n_trials` and never past it, draws the whole study
  as one curve (alpha against `q_out`), picks with `opt.pick_from_front` among all admissible
  trials, checks the search bounds, shows the pick against buy-and-hold, and saves the best
  distinct admissible trials to `studies/<study>.candidates.yaml`.
- **`informed_flow_diagnostics.ipynb` decides what ships.** `CANDIDATES` defaults to
  `["pick", "config"]` — the tuned pick beside the **no-tuning control** — and it compares them:
  - plain rank IC per sleeve, then the **fair check** (beta-adjusted, per year, per dollar-volume
    group, by fifth) and whether the book holds what the ranking likes;
  - costs at the real order size, 2× costs, and pessimistic fills as a stress row;
  - year by year; parameter-neighbourhood stability; concentration;
  - symbol-shuffled books on train (information) **and on the holdout window** (a gate; shuffled
    freely and within dollar-volume thirds);
  - a walk-forward that re-runs the study every six months from 2022 to the last bar, each
    period's panel loaded on its own window, stitched into one out-of-sample record — beside the
    no-tuning control, and with "ratings off" and "sell threshold 0.47" re-scored per period;
  - the warm holdout, as warnings.

  All of it ends in one **verdict scoreboard** ordered by three questions — is there skill, is it
  worth the risk, can it be run. Only a candidate that passes can be promoted into `config.yaml`
  without `FORCE`, and a PASS is a reason to paper-trade, not to fund.

Report `n_trades` beside every Sharpe.

**A validation instrument only works if it tests what you ship.** An earlier walk-forward
re-picked each fold from a coarse `sweep` grid, so anything the grid could not express was a
strategy the walk-forward could never validate. That silently broke twice: the grid pinned
`max_positions=10`, `q_in ∈ {0.8, 0.9}`, `q_out=0.6` while the study shipped
`15 / 0.75 / 0.45` — **not one shipped value was reachable**. The coarse sweep is now gone.

Instead, every walk-forward period re-runs the real study and picks with the same rule as the
optimization notebook. The trade-rate floor of that rule is one config value,
`optuna.pick_min_trades_per_year`, shared by both notebooks. Whether a sleeve earns its place
is judged there out of sample, never on the holdout: selecting on the holdout burns it as an
honest test.

**A validation instrument also has to end in an action.** The diagnostics notebook's §3d asks whether the picked
*values* are load-bearing: it perturbs one knob at a time by ±5% and ±10% of that knob's
own searched span (`optimize.sensitivity`) and gives each one a verdict with the move it
implies (`optimize.stability_report`) — `inert` (pin it, spend the trials elsewhere),
`plateau` (ship it), `cliff` (step to the neighboring value, which the cell prints),
`spike` (do not ship this pick), `off-center` (refer it to the walk-forward). It replaced a
cell that re-ran the old coarse sweep's *global* ablation grid, called it a neighborhood, and
drew a heatmap of a
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

**The risk overlay: how much of the book is invested at all.**

The selection always wants a full book of `max_positions` names, so on its own the strategy is
always fully invested. That made its risk the market's risk or more: across 453 re-run v23
trials, annual volatility ran 18–28% against buy-and-hold's 21%, and only 1% of trials were
both calmer than buy-and-hold and ahead of it. `risk_method` adds the one lever that can change
that. Each bar, a single exposure number scales every target weight, and the rest sits in cash:

| `risk_method` | What it does | Searched | Pinned |
|---|---|---|---|
| `none` | Always fully invested — the strategy as it was through v23 | — | — |
| `cash_switch` | Trend filter on the panel's own equal-weight index (the benchmark). When the index closes more than `regime_band` **below** its `regime_window`-day average, keep only `risk_off_exposure` invested; once it closes `regime_band` **above** it, go back to fully invested. Inside the band the last state holds, so a sideways market cannot flip the book every few days. | `risk_off_exposure` 0.0–0.8 | `regime_window` 200, `regime_band` 0.02 |
| `vol_target` | Aim the book at an annual risk rate of `target_vol`: exposure = min(1, `target_vol` / the book's realized volatility over `vol_window` days), never above 1. The exposure only moves in `exposure_step` increments, because every change re-trims every held name. | `target_vol` 0.10–0.30 | `vol_window` 21, `exposure_step` 0.10 |

**How the study decides.** The study picks the method itself (a categorical), so "no overlay"
competes on equal terms. Only one knob per method is searched: the other method's knobs are
inert on any trial, and a small search keeps the deflated Sharpe meaningful.

**When the decisions trade.** Both overlays decide on day t's close and trade at day t+1's
open, like every other order. They differ in what they react to:
- **`cash_switch` pays in a slow bear market** (2022, when this book lost 21%). A fast crash
  (2020) is mostly over before a 200-day average reacts.
- **`vol_target` reacts within days** of turbulence starting.

**Two conditions make this measurable.**
- **The objective must be absolute risk.** The overlay is only worth anything because the
  optimizer scores the book's own `volatility`. Under `tracking_error`, sitting in cash counted
  as straying from the index and was punished.
- **Out-of-sample windows must be scored warm** (`score_from`, see the main README). A 200-day
  average is undefined for most of a half-year test window that starts cold.

`informed_flow_diagnostics.ipynb` §4b is the out-of-sample referee: it re-scores each
walk-forward period's own pick under all three methods and reports whether an overlay is calmer
without giving back the excess.

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
| v24 | three v23 bounds that the front was pressing on: the alt-data sleeve weights (`w_insider`, `w_rating`, `w_ptrev`, `w_recency`) may reach **0.0** (were floored at 0.25, which forced the negative-IC insider sleeve and a floor-pinned `w_ptrev` into every pick), `rating_halflife` high `90 → 180` (52% of the front sat at 90), `max_hold` low `21 → 10` (44% sat at 21). **Plus, before v24 ever ran:** the **risk overlay** (`risk_method` none / cash_switch / vol_target, searched with `risk_off_exposure` and `target_vol`); the engine objectives `excess_cagr` ↑ / **`volatility`** ↓ (were `tracking_error` ↓); and warm-start (`score_from`) scoring for every holdout and walk-forward test window | **no** (space, objectives and OOS scoring differ) |
| v25 | the objective became `excess_cagr` **alone** (volatility commented out) and the risk-overlay searches were commented out, with no entry here. The one-trial front picked a 5-slot book trading 16.7/yr — below the 20/yr picking floor, through a silent fallback. Train excess +23.4%/yr at beta 1.32; **holdout** excess +10.1%/yr at beta **2.11**, volatility 46% vs buy-and-hold's 18%, alpha after beta **−34.6%/yr**, composite rank-IC ≈ 0. Verdict FAIL on 4 gates — and three gates it passed were broken (IDEAS.md, "Assessment 2026-09-15") | **no** (one objective) |
| v26 | four changes at once: objectives **`[alpha_cagr, volatility]`**; universe `LiquidLeaders` → point-in-time **`Liquid500`** (a new panel and benchmark); every trial scored from **2020-07-01** (`optuna.score_from`, the first bar after the universe's ranking window); the risk overlay searched again, `max_positions` **pinned** at 15. The diagnostics notebook was rebuilt alongside (bootstrap intervals, a stitched walk-forward, a symbol-shuffled random baseline, beta/drawdown/IC gates) | **no** (objectives, universe and window all differ) |
| v27 | the 2026-09-19 redesign, every axis at once: **one objective** (`alpha_cagr`; the `volatility` objective had turned v26's front into an exposure dial), **one searched knob** (`q_out` 0.60–0.85; everything else pinned), sampler `random` with a budget of **40** trials that is never topped up, **two sleeves** at fixed equal weights (`short_interest`, `ratings`; the other three at 0 by a pre-registered rule on beta-adjusted IC), the risk overlay pinned off, **30 slots**, `q_in 0.90 / q_out 0.80`, **open fills** (pessimistic is now a stress row) and the commission priced at the real order size (`backtest.order_notional: 1000`, a new engine key). The diagnostics gates were rebuilt alongside (signal rule, shuffled books on unseen data, walk-forward to the last bar with a no-tuning control, completed holds not lots, holdout-only numbers and risk limits as warnings, deflated Sharpe removed) | **no** (objective, space, sleeves, slots, fills and costs all differ) |
| v28 | **nothing is searched**: `q_out` pinned at its pre-registered 0.80 (v27's refit picks beat the untuned config in 2 of 9 walk-forward periods), `n_trials: 1`. The strategy that ships is v27's `config` candidate, unchanged; what changed is the **instrument**: the walk-forward re-reads the sleeve-selection rule inside every period (v27's had seen its 2022–2024 test periods when it chose the sleeves), and the stability table works again | v27 `config` ↔ v28 **yes** (same rules); journals no (different space) |
| v29 – v31 | one change each, adoption rule written in IDEAS.md before each run. **v29**: measurement only (the road from the v26 pick and flip-back arms, a naive 50-most-traded basket, shuffled books over the whole walk-forward, SPY, a `SCORECARD` line). **v30**: universe `Liquid500` → `Liquid1000` — **rejected** (rule arm P(alpha > 0) 92% → 75%). **v31**: `neutral_groups: 5` — every sleeve ranked inside dollar-volume fifths, adopted on train evidence alone | v28 ↔ v29 **yes** (same rules); v30, v31 **no** (universe / ranking differ) |
| v32 | **`short_interest` alone** (`w_rating: 0.0`): the sleeve rule's t bar rises 2 → **3**, a walk-forward period in which no sleeve passes holds the **equal-weight universe** (was cash), and the holdout shuffle gate becomes a warning. Instrument added at the same time: the late-fill (+50 bp/leg) and whole-share gates (blockers 2 and 3), §5e (is the open fillable?), and repaired data (a 45-day short-interest max age, a per-bar coverage floor, every traded universe re-ingested). **PASS**: rule arm alpha +6.4% (P 98%), 0% of shuffles ≥ real; t ≥ 3 left 2022-H1/H2 on buy & hold (1.5-2 years of history) and the calendar-year test left 2023-H2 there | **no** (sleeves, rule, fallback and data all differ) |
| v33 | the rule's calendar-year test "IC > 0 in ≥ 80% of years" → "**no full year with IC t < −2**". Same shipped strategy. **PASS**, adopted by its pre-registered rule (v32's test had locked `short_interest` out of 2023-H2): 6/9 full periods positive (was 5/9), stitched alpha unchanged at +6.4% | v32 ↔ v33 **yes** (same strategy; instrument only) |
| v34 | **`dtc_window: 42`** — the short-interest sleeve ranks days-to-cover AVERAGED over the last 42 bars (~4 FINRA settlements) instead of the latest one. Chosen on TRAIN (IC rises 0.046 → 0.055 with the window, top-decile churn halves); change-in-short-interest measured ≈ 0 and dropped. **PASS**, adopted under the protocol's execution-robustness clause: rule arm +7.9% (P 98%, 7/9 periods), completed holds 153 → 47/yr, break-even round trip 110 → 451 bp, train alpha under late fills +0.2% → +6.4% and under pessimistic fills −8.8% → +3.6%. Paired vs v33: rule arm Δ +1.6% (P 67%) — mostly the rule passing short interest one period earlier (2022-H2) — and fixed rules Δ −0.2% (P 46%): the smoothing adds robustness, not return. Holdout alpha +7.3% → +3.4% | **no** (the signal differs) |
| v35 | two changes kept apart. **(a) data integrity (always adopted):** short-interest rows filed under a ticker that then belonged to ANOTHER issuer are dropped (`features._SI_OTHER_ISSUER`: BNY was a municipal fund at FINRA's 1.00 floor until 2026-05, B, DOC, COR, GEN, META, WTW, COHR; CZR, IR and HHH added after the run by a code review), and SPY/GLD/SLV leave Liquid500/Liquid1000 (497/997 names). Size: v34's rules on clean data read walk-forward alpha **+7.1%** (P 97%) against the +7.9% v34 recorded, holdout +3.4% → +0.9%: the wrong-company rows (mostly BNY) had flattered the out-of-sample record. **(b) the first real optimization:** an EXHAUSTIVE grid over `dtc_window` [1 … 126] (8 points), re-run inside every walk-forward period. The main study and all nine re-runs picked **126** (the grid edge; train +13.2% vs +8.6-9.7% for 21-84: a jump, not a plateau); re-tuned vs pinned 42: **+2.2 pts/yr, P 83%, 6/9 periods**, train late-fill alpha +12.1%. Adopted by v35's own pre-registered rule, flagged UNCONFIRMED under protocol v2 (P < 0.971); pinned at 126 | **no** (data and universe differ) |
| v36 | **the walk-forward trades a point-in-time universe re-formed every six months** (`WF_UNIVERSE = "rolling"`: Liquid500's own recipe on the 6 months before each period — by 2026 only 357 of the live top 500 are Liquid500 names), prices and alt-data refreshed for the 258 new pool names first. **FAIL (3 gates) — an instrument defect, found and measured:** each period's sleeve rule read its IC on names chosen by liquidity at the END of the training window, a forward selection that hides short interest (heavily shorted names that later surged are exactly the ones that became liquid). On 2020-07 → 2023-12 the IC reads **+0.022 (t 1.7) on end-selected names, +0.057 (t 5.2) point in time, +0.058 (t 4.6) on the fixed Liquid500**; the rule passed short interest in 3/10 periods and the rule arm fell to −2.2%. The book itself was fine: **fixed rules on the rolling universe +12.4% alpha (P 97%, CAGR +23.8% vs +9.6%)**, PRIMARY Δ vs the fixed Liquid500 −0.0% (P 50%) — the universe does not change what the signal earns. Fixed in v37 | **no** (universe differs) |
| v37 | three things. **(1) instrument fix:** a rolling period's sleeve rule now reads its IC point in time (the union of every past membership, each name counted only on the dates it was a member). **(2) data fix (always adopted):** the recovered companies' own FINRA rows under their old tickers are mapped to today's (`features._SI_PREDECESSOR`: BK→BNY, FB→META, GOLD→B from 2019-01, PEAK→DOC, ABC→COR, SYMC/NLOK→GEN, WLTW→WTW, IIVI→COHR, FI→FISV, ERI→CZR, GDI→IR; each checked at FINRA-ADV ÷ own volume ≈ 1.00). Train alpha −1.8 pts: correct data costs flattery. **(3) `min_events: 0` (class R): REJECTED** — fixed rules Δ −1.1 pts (P 31%), 2025+ −1.8, train −4.8: the names the event gate kept out are mostly interlisted/ADRs whose FINRA short interest covers only the US line, so their days-to-cover reads falsely low and they rank top. On the corrected rolling instrument the rule picks short interest 8/10 but also `ratings` in three early periods (2023-H1: −21%); rule arm +2.8% (P 67%) — **FAIL (2 gates)**, for the incumbent too (≈ +5.7%). Fixed short-interest book on the rolling universe: **+11.2% (P 95%), CAGR +22.5% vs +9.6%**, beta 1.17 (1.6-2.3 in 2025-26) | **no** |
| v38 | `max_positions: 30 → 20` (class A, judged on its mechanism under the amendment fixed before it ran). **REJECTED:** fixed rules Δ +0.7 pts (P 59%; needs 80%), 2025+ −2.7, leave-top-5-out +2.9 → +0.9 (the gain was a handful of winners: AMD, NVDA, OKLO, SNDK …), 18 completed holds a year (gate 20), holdout −3.5% (warning). Train +0.6. Fewer slots bought more lottery tickets, not more skill — the book stays at 30 | **no** |
| v39 | the final confirmation: the adopted chain (short interest alone, days-to-cover over 126 bars, 30 slots, `min_events: 1`) on the rolling universe with 500 + 500 shuffled books; baseline = v33's rules on the fixed Liquid500 (clean data). **FAIL (1 gate: the procedure's P(alpha > 0) 83% < 90%)**, every other gate passes: procedure +5.6% (1% of 500 shuffles ≥ real), fixed book **+12.3% (P 97%), CAGR +23.4% vs +9.6% (SPY +12.0%)**, late fills +10.7%, whole shares +11.2%, pessimistic fills +7.4%; whole night paired: fixed rules +5.8 pts (P 82%), with late fills +9.8 (P 95%) | **no** (rolling universe) |
| v40 | **replication of v35(b):** the shipped `dtc_window: 126` against the a-priori 42 on the ROLLING point-in-time universe (143 of its 500 names by 2026 are not Liquid500). Pre-registered before v39's numbers were read: confirm at P(PRIMARY > 0) ≥ 0.80 and P(CLEAN > 0) ≥ 0.50, revert below 0.50. Result: fixed rules **−0.1 pt (P 0.4965)**, 2025+ −2.4 (P 38%), the procedure −3.2 (P 16%), leave-top-5-out −3.0 → +1.0; train +3.9. **REVERTED to 42** — the longer average won only on the data it was picked on. Config study → v41 (the revert; not yet run) | **no** |
| v41 | the final confirmation of what SHIPS after v40's revert (short interest alone, days-to-cover over 42 bars, 30 slots, `min_events: 1`), rolling universe, 500 + 500 shuffles, baseline = v33's rules on the fixed Liquid500. **PASS — every hard gate:** procedure alpha **+8.8% (P 94%, 7/9 periods)**, CAGR +19.7% vs +9.6% (SPY +12.0%), 0 of 500 shuffles ≥ real; fixed book +12.4% (P 98%), CAGR +23.3%. Whole night paired: fixed rules +5.9 pts (P 86%), late fills +9.5 (P 97%), leave-top-5-out +6.2 → +2.1. Warnings: holdout (fixed Liquid500) alpha +1.4%; the last partial period (2026-07 → 09) alpha −41% at beta 2.1 | **no** |

**Status (2026-09-24): v32 → v40 in one night — what earned what, out of sample.** One change per
version, each pre-registered in IDEAS.md before it ran; from v34 each version is compared PAIRED with the
last adopted one inside the same walk-forward (§4d). Since v36 the walk-forward trades a point-in-time
top 500 re-formed every six months (what a live book could have traded); the old fixed `Liquid500` is
reported beside it.

**What ships** (`config.yaml`, study v41 — **PASS**, every hard gate, 2026-09-26): `short_interest` alone, days-to-cover averaged over **42** bars, ranked
inside five dollar-volume groups, 30 equal slots, buy in the top tenth / sell below the top fifth,
market-on-open orders. Nothing is searched.

| the final confirmation (v39, run with `dtc_window 126`; 42 differs by v40's row below) — walk-forward 2022-01 → 2026-09 on the ROLLING universe | result |
|---|---|
| the book (fixed rules): CAGR · buy-and-hold · SPY | **+23.4%/yr** · +9.6% · +12.0% |
| … alpha after beta (P > 0) · beta · return ÷ vol | **+12.3%** (97%) · 1.15 · 0.92 vs 0.50 |
| … with every order 50 bp worse than the open · whole shares at $30k · buy-high/sell-low | +10.7% · +11.2% · +7.4% |
| the full procedure (sleeves re-read every six months) | +5.6% (P 83% — the one FAILED gate, bar 90%) |
| … against 500 symbol-shuffled books inside liquidity tenths | 1% ≥ real (real − mean +7.6 pts) |
| the whole night, paired: v39 vs v33's rules on the fixed Liquid500, same days | fixed rules **+5.8 pts/yr (P 82%, 7/9 periods)**; with late fills **+9.8 (P 95%)**; 2025+ +9.6; leave-top-5-out +3.3 → +3.0 |
| holdout 2025-01 → 2026-09 (fixed Liquid500, a warning) | CAGR +26.1% vs +18.8%, alpha +3.9% (P 68%), max DD −20.9%, commission 0.23%/yr |
| train (2020-07 → 2024-12) | alpha +11.1%, break-even round trip 895 bp (v32: 110), 30 completed holds a year |

**Which change earned what (paired, out of sample).**

| version | change | out-of-sample effect | decision |
|---|---|---|---|
| v32 | `short_interest` alone, sleeve bar t ≥ 3, buy & hold when nothing passes | rule arm +4.0% (v31) → +6.4%: `ratings` had flipped in and out | adopted |
| v33 | "no bad full year" instead of "80% of years positive" | +0.0 alpha, 6/9 periods (was 5/9) | adopted — a sounder rule, not a gain |
| v34 | days-to-cover averaged over 42 bars | fixed rules −0.2 (P 46%), but trading −70% and late-fill train alpha +0.2 → +6.4% | adopted for execution robustness |
| v35 | (a) wrong-company FINRA rows dropped; (b) exhaustive grid → 126 bars | (a) the old data had FLATTERED the record by ~0.8 pts; (b) +2.2 pts (P 83%) | (a) always; (b) adopted, unconfirmed → v40 |
| v36 | rolling point-in-time universe | fixed rules −0.0 vs Liquid500 (P 50%): the universe does not change what the signal earns | adopted as the instrument; its rule read IC on end-selected names (a defect, fixed in v37) |
| v37 | (a) point-in-time rule reading; (b) predecessor tickers' data; (c) `min_events: 0` | (c) fixed rules −1.1 (P 31%), train −4.8 | (a), (b) always; (c) REJECTED |
| v38 | 20 slots instead of 30 | fixed rules +0.7 (P 59%), 2025+ −2.7, leave-top-5-out +2.9 → +0.9 | REJECTED |
| v39 | final confirmation, 500 + 500 shuffles | see above | FAIL (1 gate), everything else passes |
| v41 | the shipped config's own final confirmation (42 bars) | procedure +8.8% (P 94%); whole night fixed rules +5.9 (P 86%), late fills +9.5 (P 97%) | **PASS** |
| v40 | 126 vs 42 bars, replicated on the rolling universe | 126 − 42: fixed rules −0.1 (P 0.4965), 2025+ −2.4 (P 38%), procedure −3.2 (P 16%), train +3.9 | **REVERTED to 42** by the pre-registered rule (P < 0.50): 126 only won where it was chosen |

**Read it with four caveats.** (1) Every version was judged on the same walk-forward; the paired numbers
count ten looks, and only 2025+ is clean of the choices made on train. (2) 2026 has been bad: 2026-H1
alpha −30% at beta 1.7, 2026-Q3 −3% at beta 2.3 — the rolling universe adds high-beta new listings. (3)
Before the scored window, on names forward-selected for 2020 liquidity, the ranking's IC was −0.055
(2018-07 → 2020-06); tonight showed that selecting names by FUTURE liquidity hides this signal (IC
+0.022 vs +0.057 point in time), which likely explains it, but the evidence FOR the sleeve starts in
2020. (4) 8 of today's 30 names are foreign filers whose FINRA short interest covers only the US line
(IDEAS.md, parked). (5) **Against SPY, the index a live account would hold instead, the edge is thinner:** the fixed-rules book beat SPY by +8.9 / +7.4 / −0.8 / **+41.0** / −0.7 points in 2022 … 2026 (rolling universe), and hedged with its SPY beta it earned +7.6%/yr (+1.7% on the fixed Liquid500) — most of it in 2025 (IDEAS.md, "Measured 2026-09-26"). A PASS was never the bar for funding: **paper-trade first**, market-on-open, and
compare fills with the open.

**Status (2026-09-20, night): v31 — liquidity-neutral ranking — adopted on train evidence, and its
verdict is FAIL (2 gates, narrowly). `config.yaml` ships it; `neutral_groups: 0` restores v29.**
One change: every sleeve is ranked inside dollar-volume fifths (`neutral_groups: 5`), so the
composite cannot lean toward heavily traded names. Adopted by the train-only rule fixed beforehand
(still passes the signal rule, top 30 keep 96% of their forward return, rank correlation with
dollar volume 0.27 → 0.02).

| v31 vs v29 (same universe, same everything else) | plain ranking (v29) | liquidity-neutral (v31) |
|---|---|---|
| train alpha · composite fair IC (t) | +11.4% · +0.048 (6.0) | +10.1% · +0.044 (5.7) |
| walk-forward, **rule arm**: alpha · P(alpha > 0) · periods | +8.4% · 92% · 6/9 | **+4.0% · 87% · 5/9** |
| … against shuffled books inside liquidity tenths, whole walk-forward | 0% ≥ real · real − mean +7.7 pts | **3% ≥ real · real − mean +5.7 pts** |
| config sleeves fixed · **short interest only** · all five sleeves | +12.3% · +8.0% · +6.6% | +8.8% · **+9.5% (P 100%)** · +3.7% |
| holdout alpha · P · shuffles ≥ real (free / within tenths) | +10.7% · 94% · 0% / 3% | +6.3% · 86% · 4% / **10%** |
| train alpha under pessimistic fills | −0.6% | −2.4% |
| verdict | PASS | **FAIL** — rule arm P 87% < 90%; holdout within-tenths shuffle |

**What v31 measured:** about **3.5 points a year of the fixed arm's out-of-sample alpha (12.3 →
8.8), and 4.4 of the rule arm's, was the lean toward heavily traded names** — a bet that paid
through 2022-26 and that nothing in the thesis promises. What is left is smaller but still there:
against books with the same rules, costs and liquidity profile the neutral rule arm earns +5.7
points a year over the whole walk-forward (3 of 100 shuffles did as well). Whether to carry the
lean is a choice, not a finding: v29 (lean in) passes every gate, v31 (lean out) is the cleaner
measurement of stock selection and misses two gates narrowly.

**The one thing that held in every experiment: `short_interest` alone.** Walk-forward alpha +8.0%
(P 96%) on `Liquid500`, +8.4% (P 96%) on `Liquid1000`, +9.5% (P 100%) liquidity-neutral — while
the rule arm swung +8.4 / +3.2 / +4.0 because `ratings` (t ≈ 2 on train, negative on the
holdout) flips in and out of it, and "all five sleeves" swung +6.6 / +10.7 / +3.7. The control arm
without the calendar-year test did *worse* here (+2.0%, P 73%): it let `ratings` in once more.
The next rule to pre-register is a stricter, multiple-testing-aware bar for a sleeve (IDEAS.md).

### Scorecard v26 → v31 (one change per step after v28)

| | v26 | v27 | v28 | v29 | v30 | v31 |
|---|---|---|---|---|---|---|
| the change | — | redesign: 2 sleeves, band at the top fifth, 30 slots, no overlay, one objective, open fills, real-size commission, 1 knob | nothing searched; walk-forward re-reads the sleeve rule each period; stability table fixed | measurement only: road + flips, naive basket, whole-walk-forward shuffles, SPY | universe 500 → 1,000 | liquidity-neutral ranking |
| train alpha | +8.4% | +11.4% | +11.4% | +11.4% | +13.1% | +10.1% |
| walk-forward, honest arm: alpha (P > 0) | −10.8% (1%) | +9.2% (99%) — flattered | **+8.4% (92%)** | +8.4% (92%) | +3.2% (75%) | +4.0% (87%) |
| … `short_interest` only | — | — | +8.0% (96%) | +8.0% (96%) | +8.4% (96%) | +9.5% (100%) |
| … shuffled books, whole walk-forward: share ≥ real | — | — | — | **0%** | 17% | 3% |
| holdout alpha (P > 0) | −9.7% (4%) | +10.7% (94%) | +10.7% (94%) | +10.7% (94%) | +5.5% (74%) | +6.3% (86%) |
| … shuffles inside liquidity tenths ≥ real | — | — | 3% | 3% | 10% | 10% |
| train alpha, pessimistic fills | (scored on them) | −0.6% | −0.6% | −0.6% | +2.5% | −2.4% |
| verdict | FAIL | fixed rules PASS, tuned pick FAIL | PASS | PASS | **FAIL (3)** — rejected | **FAIL (2)** |

**Status (2026-09-20, later): v30 — a broader universe — REJECTED by the rule fixed before it ran.**
One change: `Liquid500` → `Liquid1000` (same point-in-time recipe, floor $50M → $20M a day;
screener notebook §8c). Everything else as v29.

| v30 vs v29 | `Liquid500` (v29) | `Liquid1000` (v30) |
|---|---|---|
| train alpha · composite fair IC (t) · top-30 forward | +11.4% · +0.048 (6.0) · +1.61%/quarter | **+13.1%** · +0.048 (5.5) · +1.73%/quarter |
| train alpha under pessimistic fills · completed holds/yr | −0.6% · 109 | +2.5% · 83 |
| walk-forward, **rule arm**: alpha · P(alpha > 0) · periods | **+8.4% · 92% · 6/9** | **+3.2% · 75% · 8/9** |
| … sleeves the rule chose | `short_interest` 9/10 · `ratings` 4/10 | `ratings` 9/10 · `short_interest` 4/10 |
| … against shuffled books, whole walk-forward | 0% of shuffles ≥ real | **17% ≥ real — FAIL** |
| config sleeves fixed · all five sleeves · 15 slots | +12.3% · +6.6% · +14.3% | +9.3% · **+10.7%** · +7.8% |
| holdout alpha · P · shuffles ≥ real (free / within tenths) | +10.7% · 94% · 0% / 3% | +5.5% · 74% · **7% / 10% — FAIL** |
| verdict | PASS | **FAIL (3 gates)** |

Breadth helped in-sample and hurt everywhere it could not see. Adoption rule (IDEAS.md): keep the
broader universe only if the rule arm's P(alpha > 0) does not fall below v29's — it fell from 92%
to 75%, so `config.yaml` is back on `Liquid500`.

**What v30 exposed is more useful than its result: the pre-registered signal rule has a
knife-edge.** On `Liquid1000`, `short_interest`'s rank-IC for the half-year 2020 is **−0.000**.
The rule asks for "IC > 0 in ≥ 80% of calendar years", so with two, three or four years of
history that one reading fails the sleeve (1/2, 2/3, 3/4 < 0.8) and it stayed locked out of five
walk-forward periods, which then ran on `ratings` alone (OOS alpha +0.3 … +3.9%). A test that
treats −0.0004 like −0.05, counts a half-year like a year, and gets *stricter* the less data
there is, is not robust. v31 adds a control arm that re-reads the rule without that test; the gate
still reads the rule as pre-registered, and a better rule is for the next study to pre-register
(IDEAS.md). Also worth keeping: on the broader universe **all five sleeves beat the two chosen
ones** out of sample (+10.7% vs +9.3%) — the first sign that insider and price-target information
may carry something outside the 500 most-watched names, as finding 4b suggested.

**Status (2026-09-20): v29 — the same strategy under a better instrument. What earned what.**
v29 changed no rule and no universe; it added attribution arms, a naive control, SPY, and a
shuffled-book test over the WHOLE walk-forward (IDEAS.md has the iteration plan v29 → v31 with
the adoption rules fixed before each run). Verdict PASS; every v28 number reproduced.

| v29: stitched walk-forward 2022-01 → 2026-09-11, alpha after beta | alpha | step / cost | P(alpha > 0) | return ÷ vol |
|---|---|---|---|---|
| **THE ROAD** · road 0 — the v26 pick exactly as it ran (5 tuned sleeves, band 0.73/0.47, 15 slots, `vol_target` 0.10, pessimistic fills, $10,000-order commission) | **−0.5%** | — | 44% | 0.40 |
| road 1 — + fills at the open, commission at the real order size | +6.6% | **+7.0** | 98% | 1.04 |
| road 2 — + risk overlay off | +11.9% | **+5.4** | 99% | 1.02 |
| road 3 — + sleeves chosen by the signal rule (equal weights) | +10.8% | −1.2 | 91% | 0.87 |
| road 4 — + sell when a name leaves the top fifth (0.90 / 0.80, `max_hold` 126) | +14.3% | **+3.6** | 99% | 1.05 |
| road 5 — + 30 slots = `config.yaml` today | +12.3% | −2.1 | 100% | **1.09** |
| **FLIP-BACKS** · today's rules with pessimistic fills | +0.1% | **−12.2** | 53% | 0.51 |
| … selling at rank 0.47 | +6.4% | −5.9 | 94% | 0.80 |
| … all five sleeves at equal weight | +6.6% | −5.6 | 90% | 0.78 |
| … the `vol_target` 0.10 overlay | +7.0% | −5.2 | 100% | 1.11 |
| … short interest only (ratings off) | +8.0% | −4.2 | 96% | 0.85 |
| … 15 slots | +14.3% | +2.1 | 99% | 1.05 |
| **CONTROLS** · the rule arm (sleeves re-chosen every period — the honest number) | **+8.4%** | | 92% | 0.84 |
| naive: equal-weight 50 most traded names, re-chosen each period | +5.8% (CAGR +16.0% vs the rule arm's +16.8%) | | 86% | 0.76 |
| SPY, same window | CAGR +12.0% | | | |

Read it as: **(1) the execution model was the biggest single lever** — scoring on buy-high/sell-low
fills turned a +6.6% book into a −0.5% one, and still turns today's +12.3% into +0.1%; **(2) the
risk overlay cost 5 points**, in both directions of the comparison; **(3) selling when a name
leaves the top fifth earns 3.6–5.9 points**; **(4) choosing sleeves by the rule is worth +5.6
under today's band but −1.2 under v26's** — sleeve selection only pays once the book actually
holds the top of the ranking (and the v26 weights had been tuned on these very years);
**(5) 30 slots instead of 15 gives up ~2 points of alpha for less luck** (return ÷ vol 1.09 vs
1.05, P 100% vs 99%). Road steps 3–5 and the fixed flips share one flaw — today's sleeves were
chosen on 2020-07 → 2024-12 — so read differences, not levels; the rule arm is the clean level.

**The strongest skill test so far (new in v29):** the rule arm's own book re-run on 100
symbol-shuffled rankings, shuffled only inside dollar-volume tenths, stitched over the whole
walk-forward: shuffled mean +0.7%, p95 +5.4%, **0 of 100 ≥ the real +8.4%**; real − mean **+7.7
points a year** against books with the same rules, costs and liquidity profile. On the holdout
alone (250 draws): 0% of free shuffles and 3.2% of within-tenths shuffles ≥ real.

**Status (2026-09-19, night): v28 ran — PASS on every hard gate, with three honest caveats.**
v28 ships v27's `config` rules unchanged; what it changed is the instrument (see the v27 block
below for why).

| v28 (run 2026-09-19) | `config` — nothing searched |
|---|---|
| walk-forward of the WHOLE procedure | the signal rule re-read inside each of ten periods (2022-01 → 2026-09-11, 4.69 y), on that period's training data only. **The rule chose `short_interest` in 9/10 periods, `ratings` in 4/10, and `insider` / `price_target` / `recency` in 0/10** — retiring them was not hindsight. One period (2022-H2) no sleeve passed and the book sat in cash. **Rule arm, stitched: CAGR +16.8% vs buy-and-hold +9.4%, alpha +8.4%, P(alpha > 0) 92% (90% interval −1.2% … +19.5%), 6/9 full periods positive, return ÷ vol 0.84 vs 0.55.** IS +7.1% → OOS +10.9% (no decay). This is the honest number; v27's +9.2% / +12.3% had seen 2022–2024 when the sleeves were chosen |
| controls, same periods | config sleeves fixed **+12.3%** (P 100%) · short interest only +8.0% (P 96%) · all five sleeves equal weight +6.6% (P 90%) · config sleeves selling at rank 0.47 +6.4% (P 94%). Every arm is positive; choosing sleeves beats not choosing (+8.4 vs +6.6), the high sell threshold beats the v26 habit (+12.3 vs +6.4) |
| stability (§3d, working again) | 11 neighbours over `q_out`, `q_in`, `max_hold`, `insider_window`: all 11 alpha > 0, median keeps 100%, **0 spikes, 0 cliffs**; `q_out` a plateau (0.75 → +9.8%, 0.85 → +8.3%, centre +11.4%), the other three inert |
| shuffled books on the holdout (150 per kind) | real +10.7% vs p95 **+6.4% (free) / +8.1% (within dollar-volume thirds) / +8.7% (within tenths)**; shuffles ≥ real 0% / 3% / 3%; real − shuffled mean +14.0 / +12.9 / +12.6 points. Passes the strictest control, at roughly p ≈ 0.03 — real, not overwhelming (a p95 from 150 draws is itself ±1.5–2 points) |
| holdout (used) | as v27 `config`: CAGR +34.7% vs +18.9%, alpha +10.7% (beta 1.27), P 94%, return ÷ vol 1.52 vs 1.18, max DD −23.4%. The one warning: composite beta-adjusted IC +0.002 (t 0.1) |
| against what you could actually buy | **SPY** (total return): train +16.8%/yr, walk-forward window +12.0%, holdout +18.4% — the book beats it in all three (+27.0 / +16.8 rule arm / +34.7). An equal-weight basket of the **50 most traded names** (ranked before each window): +18.5 / +15.2 / **+44.8%** — the book beats it on train and over the walk-forward window and **loses to it on the holdout** |
| verdict | **PASS** (8 hard gates; 1 warning) |

**Read the PASS with these three caveats.**

1. *Execution decides everything.* Train alpha is +11.4% with fills at the open and **−0.6% with
   buy-high/sell-low fills**. The strategy turns the book over ~3.6×/yr; it only works if
   market-on-open orders really fill near the open. Only paper trading can measure that.
2. *2025–26 was a tide for heavily traded names*, and this book leans toward them (rank correlation
   ≈ 0.4 between the short-interest sleeve and dollar volume). A naive basket of the 50 most traded
   names beat the book on the holdout. The within-tenths shuffle says the ranking added something
   beyond that lean (p ≈ 0.03), and over 2020–24 the book beat the same naive basket by 8 points a
   year — but the holdout's +10.7% alpha is the thinnest part of the evidence, and its broad IC is
   zero (positive only inside the most traded third).
3. *The rule's latest reading disagrees with `config.yaml`.* Re-read on data to 2026-06-30 the rule
   keeps `short_interest` alone; `ratings` (train t 2.2, holdout IC −0.022 at t −3.0) drops out.
   The gates validate *the procedure* — re-read every six months and hold what passes — so that is
   what should be paper-traded; `config.yaml` still ships the train-window reading
   (`short_interest` + `ratings`) until you promote the newer one.

A PASS is a reason to **paper-trade** (IDEAS.md, "Production, only after a PASS"), not to fund. The
holdout has now been read under ten study versions; the only untouched data is the future.

**Status (2026-09-19, evening): v27 ran — the fixed rules PASS, the tuned pick FAILS — and its
read led to v28** (below the table: what v28 changed and why).

| v27 (run 2026-09-19) | `config` = fixed rules, `q_out 0.80` · `pick #4` = the study's best, `q_out 0.60` |
|---|---|
| study | 40 trials, one knob. Train alpha across `q_out` 0.60 … 0.85: +11.6 / +11.4 / +9.5 / +9.8 / +11.4 / +8.3% — **flat**; the pick is a coin flip against `config` |
| train (2020-07 → 2024-12) | `config`: CAGR +27.0% vs buy-and-hold +14.7%, **alpha +11.4%**, beta 1.06, return ÷ vol 1.38 vs 0.86, max DD −18.5%, 109 holds/yr; alpha every year (+20.2 / +2.7 / +14.7 / +6.4 / +10.8%). Slots 15/20/30/40 → alpha 13.4 / 11.9 / 11.4 / 9.1% (smooth, unlike v26). Held names: mean rank 0.94, +2.4%/quarter beta-adjusted (t 4.4). Shuffled books on train: mean −0.4% (open fills: the construction is no longer a drag), z +3.7 |
| costs | 11 bp fees + 1 bp slippage per leg (`order_notional 1000`); alpha at 2× costs +11.2%; break-even round trip 257 bp. **Pessimistic fills: +11.4% → −0.6%** (`pick`: +11.6% → +5.0%) — execution quality decides everything |
| walk-forward | ten refits 2022-01 → 2026-09-11 (4.69 y), 40 trials each, panels loaded per period. **Tuned picks: CAGR +19.3% vs +9.4%, alpha +9.2%, P(alpha > 0) 99%, 8/9 full periods positive, return ÷ vol 0.95 vs 0.55. No tuning (`config`): CAGR +22.4%, alpha +12.3%, P 100%, return ÷ vol 1.09.** IS +11.5% → OOS +9.8% (p 0.66: no decay detected). **Tuned beat untuned in 2/9 periods.** §4b: "ratings off" loses 5/9 periods to the pick but wins on the mean (+14.5% vs +9.8%); "sell at 0.47" loses 5/9 and on the mean (+7.8%) |
| holdout (2025-01 → 2026-09-11, used) | `config`: CAGR **+34.7% vs +18.9%**, alpha **+10.7%** (beta 1.27), P(alpha > 0) 94% (90% interval −0.2% … +26.5%), return ÷ vol 1.52 vs 1.18, max DD −23.4%, 185 completed holds, real commission 0.65%/yr of $30k; alpha 2025 +4.4%, 2026 +26.5%. `pick`: alpha +3.8%, P 68%. **But** the composite's beta-adjusted IC there is +0.002 (t 0.1; 2025 +0.023, 2026 −0.045), and `ratings` is −0.022 (t −3.0); only the top of the ranking still pays (`short_interest` top 30: +2.8%/quarter, t 2.8) |
| shuffled books on the holdout (150 per kind) | `config`: real +10.7% vs p95 +4.4% (free) / +5.4% (within dollar-volume thirds), **0% of shuffles ≥ real**, real − mean +14%. `pick`: real +3.8% vs p95 +7.0% / +8.7% — **FAIL** |
| verdict | `config` **PASS** · `pick #4` **FAIL (1 gate)** |

**What the v27 read found wrong with v27 itself, and what v28 does.**

1. *The walk-forward did not test the decision that mattered.* v27 chose its two sleeves ONCE, on
   2020-07 → 2024-12, and then walked forward through 2022–2024 — periods that choice had seen.
   Only the refit of `q_out` was out of sample there. → **v28 re-reads the signal rule inside every
   walk-forward period**, on that period's training data alone, and trades whatever passes (cash if
   nothing does). The gates read that **rule arm**; "config sleeves, fixed", "all five sleeves",
   "short interest only" and "sell at 0.47" run beside it as controls.
2. *The optimizer earned nothing* (2/9 periods; +9.2% tuned vs +12.3% untuned; the tuned pick
   fails the shuffle gate the fixed rules pass). By the rule fixed before the run → **v28 pins
   `q_out` at its pre-registered 0.80; nothing is searched**, `n_trials: 1` like `baseline`.
3. *The stability gate could not fail*: `strategy.valid` counts the composite weights, a candidate
   dict holding only searched keys has none, so every neighbour was rejected and the table was
   empty. → **v28 validates neighbours on top of `config.yaml`** and draws the neighbourhood over
   the pinned book rules (`q_out`, `q_in`, `max_hold`, `insider_window`).
4. *"IC > 0" on the holdout passed at +0.002 (t 0.1).* → a warning unless t ≥ 1.

**Status (2026-09-19): v26 ran and FAILED — and the read of it led to v27.** Recorded here
because re-running the notebooks overwrites their saved outputs. The v22 status further down is
kept for the record.

| v26 (run 2026-09-15) | |
|---|---|
| study | **4,000** trials (3,861 complete) against a configured `n_trials: 1000` — the optimize cell added 2,000 on every run. From about trial 1,500 on, 35–45% of new trials reused one ranking (`w_insider 1.0`, `w_recency 1.25` in 93% of the last 2,000). The 33 distinct front rows are **one ranking × an exposure dial** (`vol_target` 0.10–0.18 → `cash_switch` → `none`): alpha per unit of beta is 12–18% on every row |
| pick #3016 | `vol_target`, `target_vol 0.10` — the search floor, flagged "widen low". Train: CAGR +16.1% vs buy-and-hold +14.4%, **alpha +8.4%**, beta 0.53, volatility 10.9%, 119 lots/yr. **Holdout** (2025-01 → 2026-09-11, 52 completed holds): CAGR **+0.6% vs +18.4%**, **alpha −9.7%**, P(alpha > 0) 4%. The overlay cut the book to ~25% invested in April 2025 (low 18.5%), at the low, and was still 40% in May |
| #3218 | the max-alpha front row (`none`, `q_in 0.95`, `max_hold 241`). Train alpha +18.5%, beta 1.09. Holdout CAGR +30.2%, alpha +6.2%, **beta 1.30**, P(alpha > 0) 67% (90% interval −14.3% … +29.8%), volatility 25.8% vs 15.9%. Return ÷ volatility: train 1.60 vs the market's 0.85, **holdout 1.17 vs 1.16** |
| walk-forward | six half-year refits 2022-01 → 2024-12, **1,000 trials each** (not the main study's 4,000). OOS alpha +7.1, −6.0, −11.1, −5.6, +3.3, −26.1. **Stitched: CAGR −6.2% vs +5.9%, alpha −10.8%, P(alpha > 0) 1%.** Mean IS +12.6% → OOS −6.4% (paired p = 0.01). The refits agreed on **no** ranking parameter (`w_insider` 0.25–2.75, `rating_halflife` 18–151, `w_recency` 0.5–3.75); five of six chose `vol_target`. §4b, same picks with the overlay off: +7.1, −1.9, −3.8, −0.6, +19.2, −24.8 (mean **−0.8** vs −6.4) — the overlay cost alpha in every period it was chosen |
| in-sample checks | stability: 4 spikes (`insider_window`, `rating_halflife`, `w_insider`, `q_out`), 3 cliffs, 5 inert. Slots 5/10/15/20/30 → alpha 7.0/5.8/8.4/3.5/−1.4%. Symbol-shuffled books: mean alpha **−6.6%** (p95 −2.7%) — the rules themselves cost ~6%/yr under pessimistic fills, so the walk-forward reads "no demonstrable skill plus the drag", not "negative skill". Fill timing: alpha +13.4% (open) → +8.4% (pessimistic). Deflated Sharpe 0.000 at every `n_eff` |
| rank IC | composite, plain: train +0.020 (1 mo, t 3.1), holdout +0.027 (t 2.2). Holdout exits 98% via `q_out`, median hold 102 bars, median rank at exit 0.49 |
| verdict | pick **FAIL (4 gates)**, #3218 **FAIL (5 gates)** |

**What the 2026-09-19 assessment measured on top (train 2020-07 → 2024-12, read-only).** The
§2a rank IC does not remove beta (a Spearman IC is unchanged by cross-sectional demeaning), so it
was re-measured on **beta-adjusted** 63-bar forward returns, per calendar year and inside
dollar-volume terciles. The beta suspicion was wrong for the train window (rank-vs-beta
correlation 0.02–0.06), and the sleeves separate cleanly:

| sleeve | beta-adjusted IC, 63 bars | t | years > 0 | terciles > 0 |
|---|---|---|---|---|
| `short_interest` | +0.048 | 4.7 | 5/5 | 3/3 |
| `ratings` | +0.025 | 2.2 | 4/5 | 3/3 |
| `insider` | +0.011 | 0.4 | 3/5 | 2/3 |
| `price_target` | +0.005 | 0.3 | 3/5 | 3/3 (≈ 0) |
| `recency` | −0.012 | −0.9 | 3/5 | 0/3 |
| all five, equal weight | +0.033 | 3.4 | 5/5, fading to +0.005 in 2024 | 3/3 |
| **`short_interest` + `ratings`** | **+0.048** | **6.0** | **5/5 (0.042–0.054)** | **3/3** |

Only the top fifth of the two-sleeve ranking pays (+1.3%/quarter vs the average name, t 6.3; the
4th fifth −0.2%); the daily top 30 earn +1.6%/quarter (t 4.3), the top 15 the same (t 3.4). The
v26 pick gave `short_interest` 25% of the composite weight and 56% to `insider` + `recency`.

**Why three sleeves carry nothing here — data defects found, and why they are not the reason.**

- *Price targets.* Yahoo does not consistently split-adjust prior targets: TSLA's 3:1 split reads
  as every analyst cutting −67%, AMZN's and GOOGL's 20:1 as −95% (1.2% of rows are beyond ±50%).
  Screening them changes nothing (sum, mean, raises-minus-cuts, half-life 5: IC −0.005 … +0.008).
  An event study says why: on a clear revision day the stock moves **+0.77% / −1.01%** (t 17 /
  −13, n ≈ 15,500) — so dates and signs are right — and **nothing** follows from the next open on
  (5, 21, 63 bars). Ratings: **+1.77% / −2.00%** on the day, then a small slow drift (+0.2% /
  −0.4% over 63 bars), which is why that sleeve passes narrowly. In the 500 most-watched names
  analyst news is priced within the day; a faster feed would not help a daily strategy.
- *Recency* is direction-blind: `_event_days` counts `abs(flow) > 0`, so a fresh downgrade or
  insider sale ranks as high as a fresh upgrade or buy. IC +0.004 / +0.002 / 0.000 / −0.012 at
  5 / 10 / 21 / 63 bars. A direction-aware version (sign of the last rating change × freshness)
  is the ratings sleeve under another name, and its IC *grows* with horizon (+0.007 → +0.024);
  `rating_halflife` 5, 21 and 63 score the same, so the half-life is inert.
- *Insider.* 96% of rows are **sells** (204,066 vs 8,527 buys), netted against buys although the
  thesis calls sells uninformative; two AIG rows with price = shares = 50,000,000 are worth $4
  quadrillion; 14% of buy rows price outside 0.5–2× the close (LLY "buying" 40.9m shares at
  $18.75 with the stock at $142); corporate "10% owner" rows (GILD, MS, AZN) are filed under the
  owner's own ticker. Screened versions still score zero (buys only +0.001, EMA-weighted −0.011,
  number of buyers +0.003; names with a screened buy in 63 bars beat the rest by +0.13%/quarter,
  t 0.2). The likely reason is the universe: the insider-buy effect is a small-cap effect, and in
  Liquid500 purchases are rare (≈60 names per quarter) and small (median $131k).

The data fixes are proposals (IDEAS.md); none changes v27, which uses neither sleeve.

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
