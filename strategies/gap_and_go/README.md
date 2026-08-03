# gap_and_go — opening-range breaks on overnight gaps, built to be falsified

**Status: falsified. Nothing here ships as a config; it ships as a finding.**

v1's study was uninterpretable for reasons that had nothing to do with the
trading rule (see "What v1 got wrong"). v2 fixed the measurement apparatus and
re-asked the question properly: 1,000 trials, 978 scored, 2.2% pruned.

Two results, one robust and one not.

**Robust: `fade` beats `go`, everywhere.** Across the study (881 `fade` trials
vs 97 `go`) mean `excess_cagr` is −0.215 vs −0.575, mean Sharpe +1.41 vs −1.32,
mean `pct_profitable` 0.53 vs 0.43. Head to head at *identical* parameters the
gap holds on train, on holdout, and at all three cost bars. The event study says
why: gating a morning breakout on an overnight gap flips its excess return
negative, monotonically in the gap's own size.

**Not robust: the winning parameters.** The picked trial posts a beautiful
in-sample number — +63%/yr against a +46%/yr hurdle, Sharpe 2.60, 499 trades —
and then **fails its own pass bar in-sample** (`excess_cagr` −0.02 at 10/3 bps)
and **collapses out of sample** (−0.84 excess, info_ratio −1.60 on 109 trades).
That is textbook overfitting caught by the gate that exists to catch it.

So: the direction of the effect replicates, the tradable configuration does not.
`polarity: fade` stays in the search space as a documented finding; the config
default stays `go` so nobody mistakes an unshipped result for a shipped one.

**Thesis.** An overnight gap on elevated opening participation tends to
*continue* intraday (gap-and-go), and the break of the first-K-minutes opening
range is the trigger that separates continuation from gap-and-fade. Every other
1m strategy in this repo deliberately blacklists the first minutes of the
session (`skip_open`); this one trades **only** a morning window just after the
open — the untraded region where liquid names make their biggest, fastest
moves. That is the high-risk / high-return / fast-money profile on purpose, and
it cuts both ways: the open is also where spreads are widest, so the cost model
flatters this strategy more than any sibling.

**`polarity` — the same event, either side.** `go` takes the position the gap
points at (the thesis). `fade` takes the opposite one on the identical trigger:
buy the gap-down that breaks the range low, short the gap-up that breaks the
range high. Both are in the sweep grid and the search space, because an
in-sample event study is evidence about a *signal* and only a holdout at
stressed costs is evidence about a *strategy*.

**Capital model.** ONE shared $10k wallet (engine cash-shared mode). The
per-entry fraction is not a free parameter — it is `1 / max_candidates`, so a
full watchlist is a fully invested book and never more. Judged, like everything
here, against the equal-weight buy-and-hold baseline on `excess_cagr` /
`info_ratio`.

**How a trade works.**

1. 09:30–09:44 (default `or_minutes: 15`): the opening range forms — its
   high/low freeze at the 15th bar's close, and the first-15-minute cumulative
   volume is compared to its time-of-day expectation (`bt.volume_profile`; the
   baseline *expects* the opening surge, so `vol_mult` means "elevated even for
   the open").
2. Every name clearing the gap, volume, price and opening-range-dollar-volume
   gates is ranked cross-sectionally **at bar K−1** by `rank_by`, and only the
   top `max_candidates` may trade that session. This is the pre-open watchlist
   a real gap scanner builds; ranking on anything that happens *after* the range
   closes would be look-ahead.
3. A `go` long then needs a close above the frozen range high inside bars
   `[K, K + entry_span]`. Shorts mirror on gap-downs through the range low.
   First trigger of the session only. Under `fade` the two legs swap.
4. Fill at the next bar's open (engine-enforced). Exits: ATR trailing stop by
   default (`sl_trail: false` = fixed bracket), optional `tp_ratio` target
   (expressed as a multiple of the stop), `horizon`-bar time stop, and a
   **mandatory** end-of-session flat — overnight risk is this strategy's entry
   signal, never its hold.
5. `bt.atr`'s first bar of each day includes the overnight jump, so stops
   auto-widen on exactly the big-gap mornings. A feature, not a bug.

---

## What v1 got wrong (and why its study says nothing)

Two independent defects, both in the *measurement*, not the rule.

**1. The universe was selected with hindsight.** `HighOctaneTop20` was pinned
2026-07-05 by `tsim subset`, which ranks on `now() - 90 days` — April–July 2026
data — while the backtest window is 2025-07-01 → 2026-04-30. Equal-weight
buy-and-hold of those 20 names returned **+316% over the train window, a CAGR of
+460%/yr** (SNDK +2255%, BE +1123%; neither had pre-window daily history — a
2025 spinoff and a 2025 IPO). Since the objectives are benchmark-relative, that
hurdle was subtracted from every trial: all 45 scoring trials landed at
`excess_cagr ≈ −4.6`, and the entire "Pareto front" spanned 0.05. It was noise
being ranked, and no rule could have beaten it.

**2. 90% of trials never scored.** Of 466 trials, 420 were pruned:

| cause | share | why |
|---|---|---|
| `valid()` rejection | 17% | `entry_until` was drawn log-uniform 20–180 independently of `or_minutes ∈ {5,10,15,30}`, and `tp_mult` 0–10 independently of `sl_mult` 1–5. A fixed fraction of draws was born invalid. |
| `min_trades_per_year: 300` | 73% | 20 symbols × 210 sessions cannot produce 248 selective trades. |

The floor did not prune at random — it deleted precisely the *selective*
region. Measured directly on the v1 panel, trades/year by gate strength:

| gap_min \ vol_mult | 0.0 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|
| 0.010 | 2119 | 1304 | 721 | 429 | **177** | **103** |
| 0.020 | 1102 | 789 | 452 | **287** | **146** | **91** |
| 0.030 | 582 | 474 | 311 | **208** | **110** | **65** |
| 0.050 | **198** | **183** | **135** | **106** | **85** | **53** |

(bold = pruned). The whole `gap_min ≥ 5%` row and everything at `vol_mult ≥ 3`
was unreachable — and that corner was the only place the rule was not losing
money outright.

**The v2 fixes.** `Morning200` instead of `HighOctaneTop20` (hurdle +47%/yr, a
real one); `entry_span` and `tp_ratio` as *relative* parameters so the illegal
region is unrepresentable rather than rejected; `min_trades_per_year: 100`,
which on 200 names no longer binds; and `max_candidates`, without which 200
symbols is just 200× the churn (see "Scaling"). Three genuine bugs in
`signals()` were fixed at the same time: `prev_close` ffilled without limit
(a missing session manufactured a multi-day "gap"), "first cross only" was
really "not consecutive" (a chopping price re-fired), and `indicators()` drew a
close-based range the signal never used.

---

## Results (v2, train window, default costs)

**Ablation grid** — every combo, `excess_cagr` against a +46%/yr buy-and-hold:

| polarity | entry_mode | gap_min | vol_mult | excess_cagr | CAGR | pct_prof |
|---|---|---|---|---|---|---|
| go | time | 0.02 | 1.5 | **−0.44** | **+2.0%** | 0.49 |
| fade | break | 0.02 | 1.5 | −0.50 | −3.7% | 0.36 |
| go | time | 0.02 | 0.0 | −0.51 | −5.2% | 0.47 |
| fade | break | 0.00 | 1.5 | −0.55 | −9.2% | 0.42 |
| go | break | 0.02 | 0.0 | −0.57 | −11.0% | 0.44 |
| go | break | 0.02 | 1.5 | −0.57 | −11.0% | 0.45 |
| go | break | 0.00 | 1.5 | −0.64 | −17.6% | 0.40 |

Two things fall out immediately. The **`time` control beats the `break`
thesis** (−0.44 vs −0.57 at the same gates): the opening-range trigger
*subtracts* value over simply buying the qualifying gap at bar K. And **`fade`
beats `go` on the break trigger** (−0.50 vs −0.57). Neither observation
rescues anything: the best cell in the grid returns +2.0%/yr absolute against a
+46%/yr hurdle.

**Event study** — the naked signal, forward return open-to-open from the fill,
cross-sectionally demeaned so the market is removed, signed as the strategy is
positioned (negative ⇒ `fade` is the better side):

| config | n | 60 bars | t | 120 bars | t |
|---|---|---|---|---|---|
| gates off (any morning break) | 38,263 | **+1.03 bps** | +1.53 | **+2.43 bps** | **+2.97** |
| full thesis (`go`), uncapped | 1,584 | −10.00 bps | −2.08 | −11.62 bps | −1.93 |
| gap only (`time` entry) | 3,726 | −4.39 bps | −1.25 | −8.87 bps | −2.14 |
| big gaps only (4%) | 547 | **−21.12 bps** | −2.09 | −20.98 bps | −1.58 |

This is the clearest statement of the finding. A plain morning breakout with
**no gap gate** carries a small, statistically solid positive drift
(t = +2.97 on 38k events). Gate it on a 2% gap and the sign flips to −10 bps;
gate it on a 4% gap and it doubles to −21 bps. The gap is not a filter — it is
the thing doing the damage, monotonically in its own size.

**Not microstructure.** Delaying the fill makes the effect *stronger*, not
weaker — the opposite of what a bid-ask artifact does:

| fill delay | 60 bars | t |
|---|---|---|
| +0 bars | −10.00 bps | −2.08 |
| +2 bars | −12.62 bps | −2.72 |
| +5 bars | −14.01 bps | −3.13 |

**Why `fade` still does not ship.** The strongest inverted edge measured is
~10–21 bps over 60–120 bars, on 547–1,584 events, in-sample. A round trip
costs ~6 bps at the default cost model and ~46 bps at the opening-auction
conviction bar — and every entry here is taken in the widest-spread window of
the day. The effect and the friction are the same order of magnitude, which
means the honest statement is "not demonstrated", not "profitable inverted".

### The v2 study, and why its winner does not ship

1,000 trials: **978 complete, 22 pruned (2.2%)** — against v1's 90%. The search
converged hard on `fade` (881 of 978 trials) and on one narrow shape:
`or_minutes` 5–10, `entry_span` ~83, a **wide fixed** stop (`sl_mult` 4.5,
`sl_trail: false`), and `horizon` 228–379 on a 390-bar session. Stripped of
jargon that is "fade the morning break and hold it to the close" — `tp_ratio`
made almost no difference between 0.0 and 3.5, so the target essentially never
fired and is a dead dimension in this region.

The picked trial (925: `fade`, `or_minutes` 10, `entry_span` 158,
`gap_min` 0.02, `vol_mult` 1.25, `max_candidates` 5, `sl_mult` 4.5 fixed,
`horizon` 379) against its controls at identical parameters — `excess_cagr`,
train / holdout, at each cost bar:

| config | 2/1 bps | 10/3 bps (pass) | 20/5 bps |
|---|---|---|---|
| **picked (`fade`)** | **+0.170** / −0.840 | **−0.023** / −0.964 | −0.219 / −1.089 |
| `go` control | −0.812 / −1.233 | −0.887 / −1.309 | −0.963 / −1.383 |
| `time` control | −0.137 / −1.401 | −0.431 / −1.503 | −0.692 / −1.592 |
| no hard gap gate | +0.010 / **+0.557** | −0.287 / **+0.075** | −0.559 / −0.366 |

Read the picked row first: **the only positive cell is train at default
costs.** Raise the spread to the pass bar and the in-sample edge is already
gone; the holdout is −0.84 on 109 trades. `pct_profitable` drops 0.544 → 0.431
across the split. There is no reading of that row that ships.

The `go` and `time` rows are the ablation verdict: the ORB trigger and the
continuation direction both *subtract*, consistently, on both splits, at every
cost bar.

**The last row is a trap worth naming.** "No hard gap gate" is the only
configuration positive out of sample, and it clings on at the pass bar
(+0.075). Three reasons not to believe it. It is a **control, not the pick** —
promoting a control because its holdout looked good is precisely the tail-fitting
this notebook warns about. Its evidence runs the wrong way round: **no** edge on
the 10-month train window (+0.010, i.e. exactly the benchmark) and a large edge
on a **2.1-month** holdout (2026-04-30 → 2026-07-02, ~+16% versus the
benchmark's ~+12% cumulative) — that is what a favourable short window looks
like, not what an edge looks like. And it dies at the conviction bar (−0.366).

It is also **not** the gate-free breakout the event study liked. With
`gap_min: 0` the hard threshold is gone but `rank_by: gap` still hands all five
watchlist seats to the biggest gappers, so the gap selection is still there —
just soft. And its `polarity` is `fade`, the opposite side from the event
study's positive ungated row. The two measurements are not in contradiction;
they are measuring different populations, and that gap is exactly why neither
is a verdict on its own.

---

## Scaling to 200 names — why `max_candidates` exists

At 20 symbols the wallet was rarely the binding constraint. At 200 it always
is: 10–20 candidates fire within the same five minutes on a market-wide gap
morning, and one $10k book can fund a handful. Without a selection step the
engine funds whichever columns it reaches first — **alphabetical luck picking
the book** — and adding symbols multiplies churn instead of improving
selection (measured: 4,860 trades/year, −36% CAGR).

`max_candidates` makes the rule scale-invariant. Gate-passing names are ranked
cross-sectionally **at bar K−1**, on information frozen before any break can
happen (the gap, known at bar 0; the opening-range relative volume). Only the
top few get a seat; each entry deploys `1 / max_candidates` of the wallet.
`rank_by` chooses what "conviction" means — `gap`, `rvol`, or their product —
and is searched.

The causality matters. Ranking a session's candidates by *what eventually
happened* would look like a scoring improvement and would be look-ahead;
`test_watchlist_ranking_is_frozen_at_the_range_close` pins that a name whose
price explodes after the range closes still does not take a seat.

---

## Why this is hard (read before trusting a green backtest)

- **Costs are most understated exactly here.** The global `spread_bps: 2.0` is
  a liquid-large-cap *midday* number; quoted spreads in the first 15–45 minutes
  run 2–5× midday, and this strategy takes *every* entry in that window. The
  cost-stress rerun is mandatory and harsher than the siblings': a shipped pick
  must keep holdout `excess_cagr > 0` at `TRADING_COSTS__SPREAD_BPS=10
  TRADING_COSTS__SLIPPAGE_BPS=3` (env overrides beat YAML; the shared config
  stays untouched), with the `20/5` opening-auction worst case reported either
  way. `active_stats` re-benchmarks at the same stressed costs, so the hurdle
  stays fair.
- **Intrabar-path ambiguity is worst at the open.** 1m bars are widest in this
  window; a bar spanning both stop and target resolves stop-first (pessimistic),
  and a gap *through* a stop fills at the bar's true open with slippage charged.
  Judge on the time-stop/EOD exit population, not tight-bracket win rates.
- **The universe is honest now, but not clean.** `Morning200`'s *ranking* is
  point-in-time. Its *membership* is not: the 1-minute backfill list itself was
  assembled in 2026, so which 504 symbols were available to rank at all still
  carries survivorship. A trailing as-of-date universe recomputed monthly would
  close the remaining gap — see IDEAS.md.
- **The gap's cause is unknowable here.** Alt-data is daily and there is no
  earnings-calendar feature, so a 4% earnings gap and a 4% sympathy gap are
  indistinguishable — the literature says they continue differently. This is
  the most likely reason the pooled effect is negative: two populations with
  opposite behavior, averaged. OHLCV-only by design; see IDEAS.md.
- **Shorts assume borrow.** Gap-down shorts ignore borrow cost and
  availability; the engine does not model hard-to-borrow, so the `direction`
  search may flatter `short`/`both` — and under `fade` the short leg is the
  gap-*up* leg, which is where the crowd is.
- **A trade floor can delete the answer.** v1's did. `max_candidates` caps the
  achievable rate at `252 × max_candidates`, so a concentrated watchlist and a
  high `min_trades_per_year` are in direct tension. If the front comes back
  empty, suspect that before suspecting the market.

## Universe & train/holdout

`Morning200` on 1m bars: the 504 symbols with ≥ 200 sessions of stored 1m data
in the window, ranked by median daily dollar volume over
`2025-01-01 → 2025-06-30` — a period that closes **before** the backtest opens
— with $20M/day and $5 floors applied in that same ranking window, top 200 kept.
11 sectors, largest 28%. Built in `notebooks/01_universe_screener.ipynb` §8.
Train `2025-07-01 → 2026-04-30`, the later tail untouched until the holdout
replay (`end: null` there and nowhere else). Half-days work by construction
(sessions are data-defined) but leave `entry_span`/`horizon` little runway —
those trades skew toward time-outs.

**Run it:** open `gap_and_go.ipynb`, or
`uv run tsim optimize --strategy gap_and_go --n-trials 1000`. Budget ~2.5 min
to load the panel and 7–16 s per trial at ~3 GB resident, so a full study is an
overnight job and two workers is the practical ceiling on 16 GB.

**Verdict rule.** A configuration ships only if it beats the `time` control,
the gates-off control **and** the baseline, on train *and* holdout, at the
10/3 bps pass bar. **Nothing in v2 does.** The picked trial fails the pass bar
in-sample and is −0.84 excess on the holdout; the only out-of-sample-positive
configuration is a control whose evidence runs backwards (nothing on 10 months
of train, everything on 2.1 months of holdout) and which dies at the conviction
bar. The strategy stays unpromoted.

What *did* replicate is the direction: `fade` beats `go` across 978 trials, at
identical parameters, on both splits, at every cost bar. That is a real finding
about overnight gaps on this universe, and it is the reason `polarity` stays in
the search space. It is not a strategy.

**Selected sources.** Opening-range breakout is Crabel-lineage folk wisdom with
thin formal support — treated here as a hypothesis, and one this data rejects.
Overnight-gap continuation evidence is mixed; the gap-reversal side has at
least as much support, which is what `polarity: fade` exists to test. The
intraday spread U-shape (widest at the open) is standard microstructure
(McInish & Wood 1992) — it is this strategy's own cost trap, which is why the
stress bars above are the gate.
