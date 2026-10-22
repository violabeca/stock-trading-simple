# informed_flow — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Open questions

### Pessimistic fill pricing — IMPLEMENTED 2026-09-13 (engine, user-commissioned)

Moved out of the parking lot: `config.yaml` now carries `backtest.fill: open | pessimistic`
(`backtest.fill_prices`). `open` is the unchanged default: every order at the next bar's
adjusted open. `pessimistic` fills every **buy at that bar's high and every sell at its low**,
including retrims, rotations and stop legs, and the buy-and-hold benchmark pays the same
fills. It works for any strategy, not just this one. What is still open is the *experiment*:
run the pick under `pessimistic` and read the excess return. If it survives, the edge does
not live inside the open-to-open assumption. Flip the switch together with an
`optuna.study` bump, or the journal mixes two execution models.

## Validation & study design

> **Section numbers in this file** (§4c, §9b, §10a–§10g) refer to the single notebook as it was
> before the **2026-09-14 split**. Those checks now live in `informed_flow_diagnostics.ipynb`
> (§2 understand, §3 stress in-sample, §4 walk-forward, §5 holdout, §6 verdict). The
> optimization itself lives in `informed_flow.ipynb`. The coarse sweep and its grid
> walk-forward (§5, §10b·ii) were removed.

### The two study objectives were ~the same number — IMPLEMENTED 2026-09-13 (study v23)

`info_ratio` and `excess_cagr` correlated at 0.98 in v13/v14 and **0.975** across v22's
2,351 trials (0.994 in gap_and_go v2), so fronts collapsed to 1–2 distinct points. Measured
on 252 re-run v22 trials, the candidates were: redundant — pessimistic-fill excess 0.88,
weaker-half IR 0.90, beta-adjusted appraisal ratio 0.97; useful — relative drawdown +0.27
(5-point front), `excess_cagr` vs `tracking_error` −0.60 (**13-point front**), appraisal vs
|beta − 1| −0.24 (7 points; mean beta 1.14, so part of today's "excess" is beta).

Shipped: `OBJECTIVES = (excess_cagr ↑, tracking_error ↓)`, and `pick_from_front` only picks
rows with `excess_cagr > 0` (the low-TE end of the front is index-huggers). Journals are
stamped with their objective names; the v22 journal is refused under the new labels.
The first full run (v23, 2,500 trials) gave a 25-distinct-point front, and the balanced
min-max pick landed exactly on its max-`info_ratio` knee, so the "is the pick too timid"
worry did not materialize. A tracking-error budget remains an option if a risk cap is wanted.

### v23 pick validation (2026-09-13) — NOT promoted into config.yaml

The pick is trial 2323: train excess +10.9%/yr, TE 6.6%, IR 1.41, 16 slots, 78 trades/yr.

- **Walk-forward (refit per fold):** OOS +15.6 / +9.2 / +14.9 / **−2.5** %/yr, so 3 of 4 folds
  beat buy-and-hold.
- **Holdout:** +3.9%/yr, IR 0.35. **2025 lost** (−5.6%/yr); 8 months of 2026 carry the result.
- **Beta:** train beta 1.04 with alpha after beta **+8.3%/yr**, which is real selection. Holdout beta
  **1.38** with alpha after beta **−9.7%/yr**, so the holdout's excess was a high-beta book in a
  +41%/yr market, not selection.
- **Costs:** the break-even half-spread on excess is **193 bps/leg**, so the spread/commission hurdle
  is irrelevant.
- **Fill timing (`fill: pessimistic`):** train excess falls to **−3.2%/yr**, holdout to **−18.0%/yr**.
  This is an extreme bound (every buy at the high, every sell at the low), but the edge does not
  survive it. The truth lies between the two fill models, and it is the open question: a
  VWAP-like midpoint fill or a slippage sweep would locate where the edge dies.
- **Recency ablation,** on each fold's OWN pick OOS: the picked weight beats recency-off in 4 of 4
  folds, so the heavy recency weight is not an in-sample artifact. It was left uncapped;
  v24's 0.0 floor lets the study decide.
- **Sleeve rank-IC** (the rank correlation between the score and forward returns): insider is
  negative in train (−0.015) and positive in the holdout; price-target and short-interest are
  weak in train and strong in the holdout. The composite IC is only 0.014–0.028 (t < 1).

The completed notebook run (§10b–§10g, v23 pick) added:

- **§10b·i walk-forward (refit per fold):** mean OOS +9.3%/yr against IS +12.0% (p=0.60),
  mean OOS IR 0.87, 3/4 folds positive. The refits AGREE on `w_rating` 1.75 (4/4),
  `displace_margin` ≈ 0.81 (4/4; the pick has 0.51) and `q_in` 0.75 (3/4). They DISAGREE on
  `insider_window` (32–342) and `max_positions` (13–30).
- **§10b·ii grid walk-forward:** mean OOS +1.3% against IS +15.6%, 2/4 folds. `w_ptrev` survives
  4/4, `w_rsi` 0/4. The grid cannot express the pick, so this describes the config defaults.
- **§10c deflated Sharpe:** 0.80–0.88. **FAIL** (want > 0.95).
- **§10d neighbourhood:** 39/39 neighbours beat buy-and-hold, and the median keeps 88%. But ONE
  step on `w_rating`, `w_ptrev`, `w_recency`, `q_out` or `rating_halflife` costs 40–50% of
  the excess; the verdict reads "plateau" only because the cliff threshold is 50%, so the
  pick is a local peak. `insider_window`, `ptrev_halflife`, `q_in`, `max_hold`,
  `max_positions` and `displace_margin` are inert near the pick.
- **§10e:** 2022 lost −21% (the table then showed no excess column and skipped 2018–20; both
  are now fixed).
- **§10f random-entry baseline:** z = +9.3, PASS. Random books lose −10.8%/yr to
  buy-and-hold, so the slot machinery alone costs ~11%/yr and the ranking earns it back.
- **§10g holdout:** Sharpe 1.52, max drawdown −21%. 100% of exits go through `q_out`, and
  30/178 holds last ≤ 5 bars.

### Out-of-sample windows started COLD — IMPLEMENTED 2026-09-14 (`score_from`)

The holdout, every walk-forward test window, §10b·iii, the hurdle's year slices and §10e all
backtest a SLICED panel. `signals()` therefore restarts every rolling window at the slice's
first bar: insider sums, EMAs, the eligibility count, and recency's bars-since-event. Live
trading would have that history. On holdout day 1 the cold slice ranks **2 names** against
**97** with history intact, and stays under 80% of the warm cross-section for 32 days. The
early-January 1-bar round trips (scores 1.0 → 1/7) are this artifact.

Re-scored warm (signals on the full history, book and metrics on the window only):

| Window | Cold excess | Warm excess |
|---|---|---|
| Walk-forward mean OOS | +9.3%/yr | **+4.3%/yr** (still 3/4 folds positive) |
| Holdout | +3.4% | +3.2% |
| Holdout 2025 | −6.2% | −1.8% |

The warm holdout still shows beta 1.29 with alpha after beta **−6.6%/yr**, and pessimistic
fills give **−13.8%/yr**. Cold starts flattered the walk-forward, and the verdict does not
improve.

Proposed fix: `bt.evaluate(..., score_from=...)` (or a `warmup=` history panel) that runs
`signals()` on the long panel and backtests/scores only the window. Portfolio mode can already
express this (slice the `weights` frame, `bt.run` on the window panel); signal mode would slice
`entries`/`exits` the same way. Every notebook's OOS and holdout cells would switch to it.

**Shipped 2026-09-14** as `bt.run_strategy` / `bt.evaluate(..., score_from=...)`. It is used by
every holdout, walk-forward, year-slice and ablation cell in this notebook and in
conviction_flow, value_factor and turnover_momentum. (The 1m scanner notebooks and
momentum_swing already sliced the signals of a full-history run by hand.)

### "High CAGR, low risk": objective + risk overlay — IMPLEMENTED 2026-09-14 (study v24)

1. **Objectives.** `optimize.OBJECTIVES` is now `excess_cagr` ↑ / `volatility` ↓: the book's own
   annual risk rate replaces tracking error. Across 453 re-run v23 trials, volatility tracked
   beta (0.98) and tracking error (0.88), and picked the same trial. The reason to switch
   anyway is that tracking error charges cash as straying, so it would have punished any risk
   lever.
2. **Lever.** The strategy had no risk lever (always fully invested, 3% of trials calmer than
   buy-and-hold), so a volatility objective alone could change nothing.
3. **Overlay.** `risk_method` (none / cash_switch / vol_target) is a searched categorical. See
   the README's risk-overlay section and `strategy._exposure`.
4. **Referee.** `informed_flow_diagnostics.ipynb` §4b judges it out of sample, period by
   period.

Open until v24 has run:
- Does the study pick an overlay at all?
- Does diagnostics §4b show it calmer out of sample without handing back most of the excess?
- Is `regime_window` (pinned at 200) worth searching? Only if `cash_switch` wins and its
  folds disagree.

Next step: run v24, then judge its pick on the WARM walk-forward/holdout excess, volatility
against buy-and-hold, alpha after beta, and the fill-timing bound, before promoting anything.

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

### Walk-forward decay (2026-07-28) — SUPERSEDED 2026-08-01, kept for the record

The `p=0.02` decay below was an artifact of the instrument: the grid walk-forward was
running a strategy 11 of 12 knobs away from the pick. §10b·i now refits a full study per
fold and reports mean IS +45.4%/yr → OOS +33.6%/yr, `t=0.44, p=0.69`, 3/4 folds — "no decay
detected at n=4", not a clean bill of health (fold spread +105 / −18 / +14 / +33). The
original entry:

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
Optuna space. (The pre-optimization trend/RSI IC check was dropped in the 2026-09-14 notebook
split; to re-test them, give `w_trend`/`w_rsi` a `search:` and let the study size them.)

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

## Assessment 2026-09-12 — what the v22 study and a cross-strategy review say

Read of `informed-flow-1d-v22.log` (1,936 trials, finished 2026-09-12) plus a review of every
strategy folder; the v22 numbers are in the README's **Status** section. Items in the order
they should be acted on.

### Stop searching `max_positions`; pin it as a risk policy — IMPLEMENTED 2026-09-15 (study v26)

100 of the top-100 v22 trials sit at 10, the new low bound — the sixth floor-pinned pick in
a row (30 → 15 → 8 → 5 → 3 → 10). The edge is monotone in concentration (v15: 8/9/10 slots
gave +27.2/+17.5/+12.9%/yr), so the study answers "fewer" at any floor and never converges.
Drop the `search:`, keep `value`, record the concentration curve (5/8/10/15 slots, train and
holdout) once in the README as the trade-off chosen — exactly how `retrim_k` is handled.
*Wins if* the freed dimension lets NSGA-II resolve the sleeve weights (front stops being 4
distinct points); *dies if* nothing else in the space carries signal, in which case that is
the finding.

### Pin `max_hold`; narrow `displace_margin` to 0.0–0.5

Both are inert at every good v22 pick (the front's 10 rows are 4 outcomes; the clones
differ only in these two knobs). `max_hold` cannot bind while the clock resets on any bar
with rank ≥ `q_in`; `displace_margin` is flat over 0.4–0.9 here. Check once with
`exit_causes` that displacement still fires anywhere at the pick; if it does not, the
0.0–0.5 range is where the knob trades differently and the rest is noise.

### Cap `w_recency`'s search at 1.5 until an `exit_causes` read of the v22 pick

The sleeve took 2.5–2.75 of a ~9 total in every top-100 trial. It was added to *replace*
the filing-calendar exit cliff; at this weight it can re-create calendar-driven turnover
continuously, and the event study behind it found its information gone by bar 63. Test:
`trades_per_year` and the `cause` mix at trial 1322 vs the v21 pick. *Wins if* turnover is
flat and §10b·ii's `w_recency: 0.0` column loses out of sample; *dies if* turnover rose and
the ablation column is a wash — then the weight is buying churn.

### The universe is the biggest free knob — try a broad point-in-time `Liquid500` — IMPLEMENTED 2026-09-15 (study v26)

Insider buying is documented weakest in large, heavily covered names (Lakonishok–Lee 2001;
Cohen–Malloy–Pomorski 2012) and analyst-revision drift is stronger where coverage is
thinner. Measured in the DB (2026-09-12): LiquidLeaders produced 3–16 insider cluster-buys
(≥ 2 distinct buyers, ≥ $25k, one week) per year; the whole ingested universe produced
240–540. Alt-data is ingested for 2,303 symbols and daily bars exist for 1,951 names with
≥ 95% coverage over 2021–2024. `turnover_momentum` built the `Liquid500` universe
(`notebooks/01_universe_screener.ipynb` §8b: top 500 by median dollar volume over
2020-01 → 2020-06, ≥ $50M/day, ≥ $5) — a second data block pointing informed_flow at it
needs no engine change. Caveat: `securities` holds **zero** delisted names, so the broad
universe is today's listings; the benchmark-relative objective controls for that partially,
absolute numbers do not. *Wins if* the insider sleeve's §4c decile spread is larger there
than on LiquidLeaders; *dies if* the per-sleeve IC is no better and the wider book only adds
names.

### Fold `conviction_flow`'s two orthogonal levers in here (`min_agree`, `exit_decay`)

See `../conviction_flow/IDEAS.md` (2026-09-12): the fork is stale and its quality gate has
rank-IC ≈ 0; the two levers that add information are ~20 lines in `_select`, default off,
sweep-ablatable.

### Verified non-issue: the insider normalizer is a consistent dollar volume

`close × volume` looked like a split leak (adjusted price × unadjusted volume), but Yahoo
stores split-adjusted close **and** volume: NVDA shows no 10× step in either on 2024-06-10,
and 0 of 120 LiquidLeaders names show a split jump in `close/adj_close` over 2018–2024. The
known unit mismatch is `fund_shares` (as-reported) against adjusted prices — see
`../turnover_momentum/strategy.py` for the jump-detection fix and `../value_factor/README.md`
for the market-cap caveat.

### Why it cannot be made much faster than it is

Average hold at 10–15 slots and 40–75 trades/yr is ~2–3 months. A hard time stop without
the `q_in` clock reset would sell and re-buy the same top-ranked name (the exit-wins
conflict drops the same-bar entry, so it re-enters a bar later for two legs of costs).
Speed has to come from the *signal* — shorter half-lives, which the v22 pick already chose
(rating 10, ptrev 39) — not from the exit rule.

## Assessment 2026-09-15 — the v25 read, and what v26 changed

### What the v25 diagnostics run said (pick #1591, verdict FAIL on 4 gates)

v25 scored `excess_cagr` ALONE (volatility commented out, risk overlay searches commented out),
so its "front" was one trial: a 5-slot book trading 16.7/yr — below the 20/yr picking floor, which
the optimize notebook's fallback passed silently.

- **Holdout:** excess +10.1%/yr, but beta **2.11**, volatility 46.1% vs 18.0%, max drawdown
  −35.5%, and alpha after beta **−34.6%/yr** (engine `alpha_cagr`; the notebook's own arithmetic
  recomputation printed −24.3%). 2025 lost (−9.8%), 2026 carried everything (+57.4%).
- **Selection:** composite rank-IC on the holdout −0.006 / −0.010 / +0.005 (1/3/6 months), t ≈ 0;
  on train +0.016 at 1 month, t 0.94. The ranking measurably selected nothing.
- **Walk-forward:** 2/4 half-years, IS +15.5% → OOS +3.4% (p = 0.14); the refits agreed on no knob.
- **Stability:** spikes on `w_ptrev`, `w_recency`, `displace_margin`. Exits: 23 of 24 displacement.

### Diagnostics defects found, all fixed in the v26 notebooks

1. Random-entry null drew i.i.d. ranks every bar → the book churned daily and lost ~90%/yr, so
   z = +267 could never fail. Now a **symbol shuffle** (each name gets another name's whole score
   history: persistence and turnover identical, the name↔return link broken), run through the
   engine's warm path with `retrim_k`.
2. "Pessimistic fills" re-ran the shipped fill (`config.yaml` already says pessimistic): +23.4%
   twice. Now `open` and `pessimistic` side by side; breakeven cost found by bisection, no cap
   (was a 305 bps sweep reporting `inf`).
3. Alpha recomputed as an arithmetic mean × bars/year, overwriting the engine's geometric
   `alpha_cagr`. Now read from `bt.active_stats`.
4. Deflated Sharpe sat at 0.000 at every `n_eff` — a constant. Now INFO, with the trial-IR
   dispersion and expected-max IR shown.
5. No interval anywhere. Now a stationary block bootstrap (21-bar mean blocks) of alpha after beta
   on the holdout and on a **stitched** walk-forward record (refit every 6 months from 2022, folds
   named by training cutoff), both gated on P(alpha > 0) ≥ 0.90.
6. Gates: added trade floor, holdout beta ≤ 1.3, drawdown ≥ −25%, volatility ≤ 1.5× buy-and-hold,
   holdout composite IC > 0; excess-over-buy-and-hold demoted to WARN.
7. Train scored from 2018 with a 378-bar max window: every long-window trial was scored on its own
   warm-up. Now `optuna.score_from` everywhere, including §3b and the stability star (one warm
   `evaluate` per neighbour — `sweep` has neither `score_from` nor `alpha_cagr`).
8. §2a correlated across the whole cross-section and used `t/√h`. Now also the **top-k** names'
   demeaned forward return (k = `max_positions`, what the book owns), with Newey–West t.
9. Promotion now stamps a `# promoted:` provenance line; the §3f concentration curve records what
   the pinned `max_positions` trades away.

### Parked — not done in v26

#### A rolling annual universe rebuild

`Liquid500` is point-in-time as of 2020-07, then fixed: by 2026 it is six years stale, and it
cannot contain names that became liquid later (PLTR, GEV...), which a live trader would hold. A
rolling top-N-by-trailing-dollar-volume membership only adds that if the SUPERSET is broad: within
Liquid500 it can only shrink the list. The broad superset is 1,854 names with ≥ 1,500 daily bars
since 2018 — measured 2026-09-15 at ~1.2 s per warm evaluate on 500 names, so roughly 4× the cost
of every trial, fold and diagnostic, plus an alt-data breadth audit for the small caps first.
*Wins if* the v26 verdict is close and the stale membership is the suspect; not before.

#### Production, only after a PASS

- `tsim signals --strategy informed_flow --as-of <date>`: today's target book, diffed against
  current positions, as an order list (MOO at the next open; the backtest fills at the open).
- A paper-trading log reconciled daily against the backtest's own target weights.
- Monitoring: rolling holdout IC, realized vs expected alpha, a drawdown kill-switch, and an
  alt-data freshness/breadth alarm (a thin table goes quiet, it does not fail).
- An `as_of` reconstruction test: rebuild the book for a few past dates from rows with
  `known_at ≤ date` only, and require it to match the backtest's book.
- Measure the yfinance ratings/price-target restatement risk: diff a fresh `upgrades_downgrades`
  pull against an archived one. Until then two of five sleeves are of unknown PIT quality.
- ~~Costs: `panel_costs` assumes $10k orders~~ — **DONE 2026-09-19 (engine):**
  `backtest.order_notional` in a strategy's `config.yaml` prices the $1 minimum commission at the
  real order size (`bt.strategy_costs`); v27 sets `1000` ($30k ÷ 30 slots → 11 bp fees per leg).

## Assessment 2026-09-19 — the v26 read, the sleeve measurement, and what v27 changed

The v26 study ran (4,000 trials) and **failed**; the numbers are in README "Status (2026-09-19)".
What the read found, and what v27 did about each:

1. **The optimizer fitted luck.** Six walk-forward refits agreed on no ranking parameter; mean IS
   alpha +12.6% → OOS −6.4%; stitched OOS alpha −10.8% (a zero-skill shuffled book expects about
   −6% under those rules, so: no demonstrable skill plus the construction's drag). After ~trial
   1,500 NSGA-II re-tested one ranking. → *v27: one searched knob, 40 random trials, never topped
   up, and the walk-forward always carries a no-tuning control.*
2. **The `volatility` objective bought cash.** The front was one ranking × an exposure dial; the
   balanced pick sat on the `target_vol` floor, and the overlay cost alpha out of sample in every
   period it was chosen. → *v27: one objective, overlay pinned off.*
3. **Three sleeves carry nothing here; two do** (beta-adjusted 63-bar rank-IC, per year, per
   dollar-volume tercile — the pre-registered rule in `config.yaml`). → *v27: `short_interest` +
   `ratings` at equal fixed weights.*
4. **The book held what the ranking had stopped liking** (median exit rank 0.49; only the top
   fifth pays). → *v27: `q_in 0.90`, `q_out` searched 0.60–0.85.*
5. **Pessimistic fills as the yardstick cost more than the signal** (~5%/yr), and the commission
   model undercharged small orders ~10×. → *v27: open fills, `backtest.order_notional`.*
6. **Gates that could not do their job**: the 15-name holdout bootstrap needs ~+17%/yr of measured
   alpha to pass; the shuffle test ran in-sample only; the trade floor counted lots (210/yr on 31
   holds/yr); the deflated Sharpe was a constant; the holdout had been read ≥ 8 times. → *rebuilt
   in the diagnostics notebook (README "Backtest honestly").*

### v27 → v28 (same day): what the v27 run showed, and what is still open

v27's fixed rules passed and its tuned pick failed (README "Status"). Its own read found two
instrument defects — the walk-forward had seen its 2022–2024 test periods when the sleeves were
chosen, and the §3d stability table was empty (`strategy.valid` rejects a partial dict) — and one
pre-registered decision to execute: tuning beat not tuning in 2 of 9 periods, so the optimizer
went. v28 = nothing searched, the signal rule re-read inside every walk-forward period, a working
§3d over the pinned book rules, a within-tenths shuffle control. Verdict PASS; honest stitched
walk-forward alpha +8.4%/yr (P 92%).

**Still open after v28, most important first:**

1. **Measure real fills.** Alpha +11.4% (open fills) → −0.6% (buy-high/sell-low). Paper-trade
   market-on-open orders and record fill − open per order; nothing else in this list matters if
   that slippage is large. (Needs `tsim signals`, below.)
2. **A "most traded names" control arm.** An equal-weight basket of the 50 most traded names
   (ranked before each window) made +44.8%/yr on the 2025-26 holdout against the book's +34.7%
   (train: +18.5 vs +27.0; walk-forward window: +15.2 vs +16.8). Add it to the walk-forward arms
   and the §5a table, and consider scoring alpha against a liquidity-matched benchmark as well as
   the equal-weight one. *Wins if* the book still clears it over the whole record; *dies if* the
   edge is only that lean.
3. **Promote the rule's latest reading?** On data to 2026-06-30 the rule keeps `short_interest`
   alone. A `tsim`-side "re-read the rule" step every six months would make the shipped sleeves a
   product of the procedure instead of a hand edit.
4. **More shuffles for a final decision.** 150 draws put the real book at p ≈ 0.03 inside
   dollar-volume tenths; a p95 from 150 draws moves by ±1.5–2 points. Run 500+ once before funding.
5. **One sleeve is one factor.** The ranking is now essentially the low-short-interest effect: it
   is published, it can go quiet for a year (rule arm −6.6% and −3.7% in 2024-H2 / 2025-H1), and a
   30-name book of it runs beta ≈ 1.1–1.3. A second independent sleeve would help more than any
   tuning — see the smaller-cap universe idea below.

### Iterations v29 → v31 (2026-09-20): one change each, rules fixed before each run

Every iteration changes ONE thing, is scored on the same scorecard (the `SCORECARD` line the
diagnostics notebook prints), and its adoption rule is written here BEFORE the run.

| | the one change | why (a priori) | adopt if |
|---|---|---|---|
| **v29** | *measurement only* — same strategy, same universe. Walk-forward gains (a) the **road** from the v26 pick to today's rules, one step at a time, and **flip-back** arms (one of today's choices reversed at a time); (b) a naive "50 most traded names" basket; (c) **shuffled books over the whole walk-forward** (within dollar-volume tenths) for the rule arm — 4.7 years instead of the holdout's 1.7; (d) SPY; (e) 250 holdout shuffles | v27 changed every axis at once, so nobody can yet say which change earned what; and the holdout shuffle (p ≈ 0.03 on 150 draws) is the thinnest gate | n/a — no strategy change. The new walk-forward shuffle becomes a hard gate |
| **v30** | **universe `Liquid500` → `Liquid1000`** (same point-in-time recipe, dollar-volume floor $50M → $20M); slots, order size, band unchanged | breadth: the top 30 become the top 3% of ~950 ranked names instead of the top 6%; the short-interest effect is documented to be stronger where arbitrage is harder; and the three retired sleeves get one fair re-reading where insider buying is less rare (the rule decides, on train) | the train signal rule still passes AND the rule arm's walk-forward P(alpha > 0) does not fall below v29's. Otherwise revert to `Liquid500` and keep the result as a documented negative |
| **v31** | **liquidity-neutral ranking**: each sleeve ranked *within* dollar-volume fifths (`neutral_groups: 5`), so the composite cannot lean toward heavily traded names | the 2025-26 holdout is contaminated by a tide in the most traded names (a naive top-50 basket beat the book there) and days-to-cover leans that way by construction (volume is its denominator); on train the signal is positive inside every dollar-volume group, so it should not need the lean | decided on TRAIN only: the neutral composite passes the signal rule and its top-k forward return is ≥ 75% of the un-neutralized one. Otherwise it stays a walk-forward control arm |

### After v31: what to pre-register next (in this order)

1. **A stricter bar for a sleeve — or short interest alone.** Five sleeves are re-tested every six
   months at t ≥ 2, and the notebook's own calibration says |t| > 2 turns up on ~8-12% of pure
   noise at these horizons: that is a ~40% chance per reading of letting at least one noise
   sleeve in. `ratings` (train t 2.2, holdout IC −0.022 at t −3.0) is exactly what that looks
   like, and the rule arm swung +8.4 / +3.2 / +4.0 across v29-v31 because it flips in and out,
   while `short_interest` alone held +8.0 / +8.4 / +9.5 (P 96-100%). Candidates: t ≥ 3 (about
   a Bonferroni bar for five tests), or ship `short_interest` alone and treat every other sleeve
   as a research question. Decide BEFORE running; the walk-forward then judges it.
2. **Fix the calendar-year knife-edge** (next item) — but note v31's control arm: simply dropping
   the test made things worse on `Liquid500` (+2.0% vs +4.0%), because it let `ratings` in once
   more. The year test is a bad instrument that happened to block a bad sleeve; fix 1 first.
3. **Lean in or lean out is the user's call, so paper-trade both.** `neutral_groups: 0` (v29,
   PASS, carries ~3.5 points/yr of liquidity lean) and `neutral_groups: 5` (v31, the clean
   measurement, FAIL by a small margin) differ in one line. Run both target books daily and let
   a few months of fills decide — the same paper trade also measures the thing every version
   depends on, fill price against the open.
4. **Insider and price-target sleeves outside the 500 most-watched names.** On `Liquid1000` "all
   five sleeves" beat the two chosen ones out of sample (+10.7% vs +9.3%). Worth a dedicated,
   pre-registered look on the 500 names that are in `Liquid1000` but not in `Liquid500` — with
   the insider row screen (below) applied first.

### v30's lesson: the signal rule needs a better consistency test (PRE-REGISTER before the next study)

The rule's "IC > 0 in ≥ 80% of calendar years" is a knife-edge: on `Liquid1000`,
`short_interest`'s IC for the half-year 2020 was −0.000 and that alone kept the sleeve out of five
walk-forward periods (1/2, 2/3, 3/4 years < 0.8). Candidates, to be chosen BEFORE looking at any
new walk-forward: (a) drop the test and rely on the Newey-West t plus the dollar-volume groups —
v31 runs this as a control arm; (b) count only full calendar years; (c) replace the count with
"no calendar year significantly negative" (yearly IC t > −2); (d) replace in/out with a weight
that grows with t and is capped, so a sleeve fades instead of flipping. Whatever is chosen, the
rule must not get stricter as the history gets shorter.

### v32 → v33 (2026-09-23): PRE-REGISTERED before either ran

Done first, and in force for both runs (README "Status (2026-09-23)"):

- **Data.** Every traded universe has been re-ingested, and `features._SI_MAX_AGE_DAYS` (45) and
  `strategy._sleeve_rank` (a per-bar coverage floor) are in place. On the repaired data, v31's
  train alpha re-scores from +10.08% to +10.28%.
- **Execution.** Orders are market-on-open.
  - Measured: the daily open against the first 1-minute trade differs by a median 0 bp (p90 4 bp;
    21,643 symbol-days, 88 names).
  - A fill anywhere in the first half hour lands a median 52 bp from the open.
  - New hard gate: train alpha > 0 when every leg fills `late_fill_bps` (50) worse than the open.
  - Pessimistic fills stay a warning.
- **Sizing.** Measured, whole shares at the real account cost −0.2 … +0.4 points a year. That holds
  at 30 or 20 slots and at $30k or $50k.
  - So the book stays at 30 slots, and fractional shares are not required.
  - New hard gate: train alpha > 0 with whole shares at `ACCOUNT`. The holdout reading of the
    same check is a warning.

| | the one strategy change (+ its instrument) | why (a priori) | adopt if |
|---|---|---|---|
| **v32** | Ship `short_interest` alone (`w_rating: 0.0`), and raise the sleeve rule's t bar from 2 to **3**. A walk-forward period in which no sleeve passes holds the **equal-weight universe** instead of cash. The holdout shuffle gate becomes a warning | Five sleeves re-tested at t ≥ 2 let in at least one noise sleeve on about 40% of readings. `ratings` (train t 2.2, holdout t −3.0) is what that looks like. t ≥ 3 is roughly a Bonferroni bar for five tests. `short_interest` alone held +8.0 / +8.4 / +9.5%, but that was read after three looks, so the gate reads the PROCEDURE at t ≥ 3, not that arm. Cash in a no-signal period is a bet against the market; the benchmark is the neutral fallback. The holdout has been read under eleven versions | The rule arm has alpha > 0 with P(alpha > 0) ≥ 90%, at least half its full periods positive, beats the p95 of whole-walk-forward shuffled books, and passes every other hard gate. Everything read on the holdout is a warning |
| **v33** | Replace the rule's calendar-year test "IC > 0 in ≥ 80% of calendar years" with "**no FULL calendar year with IC t < −2**" (option (c) above). The shipped strategy is the same as v32's | The old test gets stricter as history gets shorter, and it treats −0.0004 like −0.05 | Only if v32's rule locked `short_interest` out of at least one walk-forward period. In that case, adopt if v33's rule arm has P(alpha > 0) ≥ v32's and passes the same gates. Otherwise v33 is a measurement only |

### v34 → v40 (2026-09-23 night): the protocol, fixed before v34 ran

The user asked for iterations up to v40 (or 05:00 CEST), each built on the last version's
learnings. Every version is judged on the SAME walk-forward, so iterating on it slowly turns it
in-sample. Four rules limit that:

1. **One change per version, with an a-priori mechanism.** A change is chosen from TRAIN evidence
   (2020-07 → 2024-12) or from a documented mechanism, never from walk-forward numbers.
2. **A paired comparison.** Each version is compared with the last ADOPTED version, which is re-run
   inside the same walk-forward (diagnostics §4d, `BASELINE`). Both daily return series are
   bootstrapped on the same blocks.
3. **Adopt only if** all hard gates pass, AND the rule arm's Δ alpha > 0 with P(Δ > 0) ≥ 0.80,
   AND Δ train alpha ≥ 0 (in-sample agrees). One exception: a change made for execution robustness
   may be adopted at Δ alpha ≥ −0.5 pt if its train alpha under late fills improves by ≥ 1 pt.
4. **Count the looks.** The final report quotes the adopted chain with the number of versions
   tried, and with the naive fixed-rules arm beside the rule arm.

The trial budget for a searched version is sized to its space: ≥ 30 TPE trials per searched
dimension, plus a convergence check. The walk-forward then re-runs the same search inside every
period, and the untuned fixed config runs beside it as the control.

**Protocol v2, in force from v36.** It was written after a design workflow (five proposers, three judges and a
sequencer) read v32-v34, and fixed before v36 ran. Every statistic is PAIRED over the same days, with a block
bootstrap of 21-bar blocks and 2,000 draws, and each side is scored against its own market:

- **PRIMARY** = the fixed rules, new minus adopted: the same sleeves in every period, so the mechanism alone.
- **CLEAN** = PRIMARY on 2025+ only, the one stretch no a-priori choice was read from. It is a weak veto.
- **LATE** = PRIMARY with +50 bp per leg on both sides.
- **FLIP** = the rule arm in periods where the two procedures chose different sleeves. It is printed as a
  "known outcome" and never counts toward adoption.

v34 showed why FLIP is excluded. Its rule-arm gain was one flipped period (2022-H2). Its PRIMARY was −0.2,
and on the clean 2025-26 stretch about −5 pts a year.

Adoption by class:

| class | adopt if |
|---|---|
| D, data fix | always; its size is reported |
| I, instrument | reproduces the incumbent |
| V, validity | always, as the instrument; the STRATEGY passes only if the absolute hard gates hold on it |
| A, mechanism | every hard gate, P(PRIMARY > 0) ≥ 0.80, and P(CLEAN > 0) ≥ 0.20. Execution variant: PRIMARY ≥ −0.5 and P(LATE > 0) ≥ 0.80 |
| R, robustness | every hard gate, PRIMARY ≥ −0.5, P(CLEAN > 0) ≥ 0.20, and its stated mechanism check |
| B, searched or walk-forward-driven | P(PRIMARY > 0) ≥ 0.971 (= 1 − 0.20/7 looks), and P(CLEAN > 0) ≥ 0.50 |

**Power, stated plainly.** SE(PRIMARY) ≈ 3.3 pts a year. At P ≥ 0.80 a true +2 pt gain passes about 40% of the
time, and a true 0 passes 20%. So this walk-forward can veto harm; it cannot certify a gain below about 5 pts a
year.

**Recorded before v36: the pre-period.** 2018-07 → 2020-06 lies before the scored window, on Liquid500 names
that were forward-selected for 2020 liquidity. There, the short-interest ranking's beta-adjusted 63-bar IC is
**−0.055 (t −3.4), negative in all three calendar years**. The top-30 names still returned +2.5% a quarter
(t 1.2). All the evidence FOR the sleeve comes from 2020-07 on.

**Amendment, fixed before v38 ran.**

- On the corrected rolling instrument (v37), the rule-arm hard gates fail for the INCUMBENT as well:
  rule arm +2.8% (P 67%), and 11% of shuffles ≥ real. The procedure admitted `ratings` in three early
  periods on thin evidence; in 2023-H1 that cost −21%.
- The fixed short-interest book on the same universe earns +11.2% (P 95%).
- A change to the book therefore cannot be judged by gates that the procedure fails whatever the
  change. From v38, a class A change is adopted on its MECHANISM: P(PRIMARY > 0) ≥ 0.80,
  P(CLEAN > 0) ≥ 0.20, the same sign after leaving out the top-5 names, and Δ train ≥ 0.
- The rule-arm gates are still reported for both arms, never hidden. Protocol v2's ban stands:
  no version may be designed to move a sleeve decision.

| | the one change | why (a priori / train evidence) | adopt if |
|---|---|---|---|
**Results.**

- **v39** (the final confirmation, 500 + 500 shuffles): FAIL on one gate; the procedure's P(alpha > 0) is 83%. Every other gate passes.
  - Fixed book +12.3% alpha, P 97%.
  - Whole night paired: fixed rules +5.8 pts (P 82%), with late fills +9.8 (P 95%).
- **v40:** P(PRIMARY > 0) = 0.4965, so 126 was REVERTED to 42. The config now names study v41, not yet run. README "Status (2026-09-24)" has the full read.

| **v41** | The final confirmation of what SHIPS after v40's revert: short interest alone, days-to-cover averaged over 42 bars, 30 slots, `min_events: 1`, on the rolling universe, with 500 + 500 shuffled books. The baseline is v33's rules on the fixed Liquid500, so §4d gives the whole night's paired effect for the shipped config. The data is unchanged since v39/v40 (last bar 2026-09-11) | v39 confirmed the chain with 126, and v40 reverted to 42, so the shipped config has no verdict of its own yet | Class I: the verdict is reported as it stands. Nothing is adopted or changed by this run |
| **v40** | **Replicate v35(b):** the shipped `dtc_window: 126` against the a-priori 42, on the ROLLING point-in-time universe. The only change is the baseline arm; the shipped config is unchanged | 126 is the one UNCONFIRMED parameter of the adopted chain. It was picked on the fixed Liquid500, at the edge of its grid, with P 83% against a multiple-look bar of 0.971. The rolling universe shares the periods but not the cross-section (by 2026, 143 of its 500 names are not Liquid500), so it is a partial replication, not the same look | Pre-registered 2026-09-24 01:45, before v39's numbers were read. **Confirmed** if P(PRIMARY > 0) ≥ 0.80 AND P(CLEAN > 0) ≥ 0.50. **Reverted to 42** if P(PRIMARY > 0) < 0.50. Otherwise 126 stays, still flagged unconfirmed |
| **v39** | Final confirmation (class I): the adopted chain with 500 + 500 shuffles. The baseline is v33's RULES (`dtc_window 1`, `min_events 1`, 30 slots) on the FIXED Liquid500, on today's clean data, so §4d gives one paired number for the whole night's rule and universe changes | The last look before paper trading. A 150-250-draw p95 moves by ±1.5-2 points | Must reproduce v38's baseline arm (30 slots). The verdict is reported as it stands |
| **v38** | `max_positions: 30 → 20` (class A). `order_notional` stays $1,000 for both arms, which is conservative: real orders would be $1,500 | The book's edge sits at the very top of the ranking. The train concentration curve is monotone in every version since v15; at dtc 42 it reads 15 / 20 / 30 / 40 slots → +11.7 / +10.5 / +8.5 / +7.5%. 20 is the middle step, not the floor (fewer slots = more single-name luck; the design panel warned the edge rides on a few winners) | The amendment above, against 30 slots on the same rolling universe |
| **v37** | `min_events: 1 → 0` (class R). A name is eligible wherever a RANKED sleeve has a live value (for this book, a days-to-cover reading ≤ 45 days old), no longer "an insider or rating event in the last 126 bars" | It is a leftover of the five-sleeve design. For a short-interest-only book it force-sells names whose ranking is still good; on v34's holdout, 9 of 84 exits were forced sells at a median rank of 0.967, all still above q_in. It also makes the book silently depend on two feeds it does not rank: a ratings outage would empty the book. On the rolling universe, newly liquid names have thin analyst/insider coverage, which makes the gate matter more. Train (proposer's replica): ranked names 473 → 489, IC +0.057 → +0.058, entries 47 → 39 a year | Class R, protocol v2: every hard gate passes, PRIMARY Δ ≥ −0.5 pt, P(CLEAN > 0) ≥ 0.20, and its mechanism check (train ineligible exits ≤ 1.5 a year) |
| **v36** | **The walk-forward trades a point-in-time universe re-formed at every period start** (`WF_UNIVERSE = "rolling"`). It is Liquid500's own recipe moved: top 500 equities by median dollar volume over the 6 months before the period (≥ $50M, price ≥ $5, ≥ 100 sessions). The §4d baseline arm trades the fixed Liquid500 in the same periods, and each arm is scored against its OWN equal-weight market. Before it ran: daily prices for 254 pool names (their prices ended 2026-06-12) and alt-data for 258 pool names outside Liquid500 were refreshed (prices by `tsim backfill`, alt-data through notebook 03). The shipped strategy is otherwise v35's | Validity. Liquid500 was ranked on 2020-H1 and, worse, required ≥ 1,000 sessions inside 2020-07 → 2024-12: a SURVIVAL filter read at the end of train, so the 2022-24 walk-forward traded only survivors. By 2026 only 357 of the live top 500 are Liquid500 names, and a live book cannot trade a 2020 list. Five proposers and three judges ranked this the most valuable strategy-level change | Always, as the instrument: the rolling rule arm becomes the walk-forward gate, because it is what a live book could have traded. The STRATEGY passes only if every hard gate holds on it. The paired Δ against the fixed-universe arm is reported as the size of the survivorship / staleness effect, whatever its sign |
| **v35** | Two changes, kept apart. **(a) A data-integrity fix, adopted unconditionally:** short-interest rows filed under a ticker that belonged to another issuer are dropped (`features._SI_OTHER_ISSUER`: BNY a muni fund until 2026-05, B, DOC, COR, GEN, META, WTW, COHR). SPY, GLD and SLV leave Liquid500 and Liquid1000 (497 / 997 names), and the screener keeps equities only. **(b) The first real optimization:** `dtc_window` is searched on an EXHAUSTIVE grid [1, 11, 21, 32, 42, 63, 84, 126] (1-12 settlements), `sampler: grid`, 8 trials. The walk-forward re-runs the same grid inside every period (`retune`) | (a) Known-wrong data is never kept because it scores better. Five design agents and three judges found it and verified it in the DB. (b) The knob has an interior bias-variance trade-off (volume noise vs staleness). A grid is the only search whose optimum is exact, and 8 points cost 80 per-period trials in total | (a) Always. Its size is measured as v35's untuned rule arm (v34's rules on repaired data) against v34's recorded +7.9% (unpaired: only the data differ). (b) The search earns its place only if the re-tuned rule arm beats the pinned-42 rule arm (§4d, paired): Δ > 0, P(Δ > 0) ≥ 0.80, ≥ 5/9 full periods won, and the main-study pick's train alpha ≥ config's. Otherwise `dtc_window` stays pinned at 42 and the search is removed |
| **v34** | `dtc_window: 42`: the short-interest sleeve averages days-to-cover over the last 42 bars (~2 months, ~4 settlements) before ranking | Days-to-cover = SI ÷ FINRA average daily volume, and the noisy denominator makes one settlement's rank jitter. 86-95% of exits happen on the bar a new settlement arrives; the median hold is 26 bars. TRAIN, beta-adjusted 63-bar IC at 1 / 21 / 42 / 63 bars: +0.046 / +0.050 / +0.055 / +0.058, all 5 years positive. Top-30 forward return +1.23 → +1.53%/quarter at 42. Top-decile churn 0.125 → 0.068. Change in short interest (21 / 42 bars) reads IC ≈ 0 and is NOT pursued | The protocol above, against v33 |

### Measured 2026-09-26 — against SPY, the edge is thinner than "alpha" says

Every gate scores alpha against an equal-weight buy-and-hold of the SAME panel. In 2022-26 that
basket lagged SPY by 2.4-2.6 points a year (mega-caps led), so beating it is easier than beating the
index a live trader would actually hold instead.

**Fixed rules, 2022-01 → 2026-09.** The shipped config (dtc 42), read-only. The SPY hedge ratio is the
book's trailing 252-day beta, known at the time.

| universe | book CAGR | SPY | book hedged with SPY | book − SPY by year (2022 … 2026) |
|---|---|---|---|---|
| fixed Liquid500 | +15.1% (DD −22.0%) | +12.0% | **+1.7%** (vol 8.3%, DD −14.0%) | — |
| rolling point-in-time top 500 | +23.4% (DD −25.8%) | +12.2% | **+7.6%** (vol 13.4%, DD −17.8%) | +8.9 · +7.4 · −0.8 · **+41.0** · −0.7 |

What this means:

- Over the index, most of the edge is ONE year: 2025, the rally in newly liquid AI / nuclear / crypto
  names that the rolling universe could hold and the 2020 list could not. Outside it, the book beat
  SPY by about 3-4 points a year on average (one year −0.8, 2026 so far −0.7).
- **Proposal:** add SPY-relative numbers to the diagnostics (alpha after beta against SPY, and a
  hedged record) as warnings first, then as a gate. The equal-weight benchmark stays the
  selection test; SPY is the "is it worth doing at all" test.
- **Cadence (measured the same day).** With `dtc_window: 42` only 9% of exits fall on a FINRA-release
  bar (28% within 2 bars). The 42-bar mean slides daily, so a "trade only on release days" schedule
  would be a DIFFERENT strategy and needs its own test.

### Parked 2026-09-24 — foreign issuers' days-to-cover is mis-measured (NOT run as v40)

FINRA short interest covers only the US line. For a foreign private issuer (a 20-F / 40-F / 6-K filer,
identifiable in `fundamentals.form`), shorting on the home market is invisible, so days-to-cover reads
LOW, i.e. favourable.

- **Train (Liquid500, 2020-07 → 2024-12).** Foreign filers are 4.3% of eligible name-days but 14.7% of
  held slot-days, 3.4× over-represented. They sit at FINRA's 1.00 floor 9.7% of the time, against 2.1%
  for domestic names.
- **Today's book.** 8 of 30 names are foreign filers (AZN, BP, BTI, IREN, PAAS, STM, TSEM, WCN).
- **Why it was not run as v40.** On train the held foreign names EARNED more, not less: beta-adjusted
  63-bar forward return +3.10% against +2.56% for domestic names. An exclusion would be judged on
  measurement grounds alone and would probably cost return.
- **What to do instead.** Watch it live. A `foreign_filer` feature (a 0/1 level from the SEC form, point
  in time by `known_at`) and a strategy switch would be about 20 minutes of work if it is ever wanted.

### Parked — not done in v27

#### Data fixes found on the way (ENGINE PROPOSALS, none changes v27)

All read-time screens, like the fundamentals ones; none alters a stored row. They move
`conviction_flow`'s inputs too, so its study needs a bump when they land.

- **Price targets are not consistently split-adjusted by Yahoo.** TSLA's 3:1 split reads as every
  analyst cutting −67%, AMZN's and GOOGL's 20:1 as −95%; 1.2% of rows are beyond ±50%, and an
  EMA carries a −2.9 summed "revision" for months. Proposal: `AND pt_current / pt_prior BETWEEN
  0.5 AND 1.5` in `features._PRICE_TARGET_SQL`. (Screening changes nothing for the sleeve's IC —
  the news is priced on day 0 — so this is hygiene, not a rescue.)
- **Insider rows.** 96% of rows are sells, netted against buys although the thesis calls sells
  uninformative; two AIG rows (2023-12-21) carry price = shares = 50,000,000 ($4 quadrillion); 14%
  of buy rows price outside 0.5–2× the close (not open-market purchases: LLY 40.9m shares at
  $18.75 with the stock at $142); corporate "10% owner" rows (GILD, MS, AZN) are filed under the
  owner's own ticker — a wrong-issuer attribution in `providers/sec.py`. Proposal: a row screen in
  `features._INSIDER_SQL` (price within a band of the close, drop corporate 10% owners) and the
  attribution fix. This subsumes the older `insider_buyers` builder idea. Measured: screened
  buys-only / EMA-weighted / buyer-count versions all score ≈ 0 in Liquid500.
- **FINRA short interest is keyed by ticker, and tickers get reused** (found 2026-09-23). Six
  names carry another company's rows for the years before their rename: WTW's 2018-19 rows
  are Weight Watchers, COR's to 2021-12 CoreSite, GEN's to 2021-03 Genesis Healthcare, META's
  2021-07 → 2022-01 an ETF that held the ticker before Facebook. `_SI_MAX_AGE_DAYS` (45) now
  stops those values being CARRIED across the gap (1,616 scored train cells), but the rows
  themselves still score inside their own dates. Proposal: a read-time screen dropping a
  symbol's short-interest rows before a gap longer than the max age, when prices show the
  name trading through the gap (the rename signature). Measure first: six of 500 names.
- **The per-bar coverage floor is informed_flow's only.** `conviction_flow`, `value_factor`
  and `turnover_momentum` still judge a sleeve by its best-covered bar (`_coverage`), so a
  table that goes thin at its end tilts them the way it tilted this one. Port
  `_sleeve_rank` if any of them is revived.
- **Events dated on a non-trading day are dropped** by `features._to_grid`'s flow branch
  (`wide.reindex(keys)`): 0.6% of insider rows, 0.1–0.2% of analyst rows. Proposal: roll them to
  the next panel day before the `shift(1)`.

#### Insider and analyst sleeves in a smaller-cap universe

The insider-buy effect is a small-cap effect, and analyst news is priced within the day in the
500 most-watched names (event study: +0.77% / −1.01% on a price-target revision day, +1.77% /
−2.00% on a rating change, ≈ 0 from the next open). Both could carry information where price
discovery is slower. Needs a new point-in-time universe (mid/small caps), an alt-data breadth
audit for it (`notebooks/03_altdata_update.py`), and the §2a signal rule read **before** any
study. *Wins if* a sleeve passes the rule there; *dies if* breadth is too thin to rank.

#### Net insider SELLING as an avoid-filter

Screened net buys−sells scores +0.031 (t 1.7) on the fair check, all from the bottom of the
ranking (the top 30 earn nothing) — heavy insider selling may mark names to avoid. Useless to a
book that buys the top of a ranking; possibly useful as an exclusion rule. It failed the signal
rule (t < 2), so it stays parked.

#### Plateau picking (ENGINE PROPOSAL, restated)

`pick_from_front` takes the best row. With one coarse knob that is harmless (v27's train curve is
flat, +8.3% … +11.6%), and the no-tuning control is the out-of-sample referee. If a later version
searches more than one or two knobs again, pick the centre of a broad good region instead.

## Known caveats (documented, not actionable here)

- Ratings + price-target sleeves carry survivorship bias (yfinance covers only
  currently-covered names); insider and price data do not.
- Until v25 the universe was `LiquidLeaders`, screened on TODAY's liquidity (a forward-selected
  list); v26 trades the point-in-time `Liquid500`, which still holds only names listed today.
- Not modelled: taxes; and the 500-name equal-weight benchmark is not something a $25–50k account
  can buy — the real alternative to this strategy is an index fund.

### Footnote to the `Liquid500` item (2026-09-12)

`turnover_momentum`'s pre-registered test **failed** on `Liquid500` — that says nothing about
the alt-data sleeves (different information entirely), but two things carry over: the
universe file exists and loads (500 names, 1,133 daily bars, ~27 s per 5-trial smoke run), and
`fund_shares` had filer unit errors, screened at read time since 2026-09-14 (see
`../turnover_momentum/IDEAS.md`) — irrelevant here unless a sleeve ever divides by shares.
