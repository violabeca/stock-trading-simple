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

### Out-of-sample windows start COLD — ENGINE PROPOSAL (2026-09-13)

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
**Engine change, proposal-only.**

Next step: approve or reject the warm-start fix, then run v24 (widened bounds, sleeves that
may switch off). Judge its pick on WARM walk-forward/holdout excess, alpha after beta, and the
fill-timing bound before promoting anything.

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
Optuna space; §4c cell 15 still scores them so the decision stays falsifiable.

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

### Stop searching `max_positions`; pin it as a risk policy

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

### The universe is the biggest free knob — try a broad point-in-time `Liquid500`

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

## Known caveats (documented, not actionable here)

- Ratings + price-target sleeves carry survivorship bias (yfinance covers only
  currently-covered names); insider and price data do not.
- `LiquidLeaders` itself is a current-listings snapshot (screener universe).

### Footnote to the `Liquid500` item (2026-09-12)

`turnover_momentum`'s pre-registered test **failed** on `Liquid500` — that says nothing about
the alt-data sleeves (different information entirely), but two things carry over: the
universe file exists and loads (500 names, 1,133 daily bars, ~27 s per 5-trial smoke run), and
`fund_shares` has filer unit errors (see `../turnover_momentum/README.md`, "Data-quality
finding") — irrelevant here unless a sleeve ever divides by shares.
