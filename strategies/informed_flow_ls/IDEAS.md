# informed_flow_ls — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet) implemented.
Every idea carries a *why* and the falsification that would settle it.

## Pre-registration v1 (2026-10-19, written before the walk-forward ran)

**The procedure (user's choice: nested Optuna walk-forward).** Every 6 months from 2022-01 to
2026-10-16 (10 periods, the last partial), one Optuna study (TPE, `optuna.n_trials` = 80 trials,
seed 42) searches the config's four `search:` knobs — `dtc_window`, `skew_window`, `w_skew`,
`short_stop` — on that period's PAST only:

- the training panel is every point-in-time membership of the rolling top 500 since 2020-07,
  each name counted only on the dates it was a member (`member` feature);
- every trial is scored warm from 2020-07-01 to 21 sessions before the test period (`PURGE`);
- the score is NOT the total: the training window is cut into 4 stretches and a trial scores the
  mean of its 4 stretch alphas minus half their standard deviation (`research.robust`), so a set
  that earned everything in one stretch loses to one that earned a little in each;
- the probability of backtest overfitting (`research.pbo`, 8 blocks, 70 splits) is recorded per
  study from every trial's daily returns.

The best trial then trades the next 6 months unseen, on that period's own members, warm.

**Arms, same periods, same costs:** `tuned` (PRIMARY), `untuned` (config values), `long side`
(config, `sides: long`; also SPY-hedged after the fact), `no options` (tuned, `w_skew` 0), and
the stresses of the primary: `2x costs`, `2x borrow`, `late fills` (+50 bp a leg).

**Luck band:** 100 books per period that keep every rule and the period's tuned values but hold
random names (columns relabelled within dollar-volume tenths), stitched like the primary.

**Hard gates (all must pass; `research.GATES`):**

1. stitched walk-forward alpha (CAGR − beta × buy & hold) > 0;
2. block-bootstrap P(alpha > 0) ≥ 0.90;
3. the primary beats ≥ 95% of the random-name books;
4. tuning beats not tuning: alpha(tuned) − alpha(untuned) ≥ 0 on the same days;
5. median probability of overfitting ≤ 0.5;
6. |beta| ≤ 0.3 (it is meant to be market-neutral);
7. alpha > 0 in at least half the periods;
8. alpha > 0 with 2× costs, with 2× borrow and with late fills.

**Warnings (reported, not gating):** max drawdown ≥ −25%; the primary beats the long side +
SPY hedge and the book without options.

**What a FAIL means:** the procedure is not shipped. If only gate 4 fails, the untuned config is
the candidate — but it was not pre-registered as the primary, so it would need its own forward
test. Nothing here is re-run with changed gates; a v2 needs a new pre-registration and the
final check (paper trading from 2026-10-20) is the only data no version has read.

**Known before running (from the signals notebook, 2026-10-19):** on the point-in-time top 500
the short-interest sleeve's 21-day IC is +0.026 (t 3.6) with BOTH ends working (top fifth
+0.50% per 21 days, bottom fifth −0.40%), positive every full year 2021-2025, negative in 2026
(−0.026); the put-skew-change sleeve is weak here (IC +0.003, t 0.5; it was found on ~1,100
optionable names including smaller ones). The untuned book: +7.4%/yr, beta 0.21, alpha +3.3%,
max drawdown −25%, −21% in 2026 so far.

## Iteration protocol (2026-10-20, autonomous session; written before any walk-forward result)

The user asked for a multi-hour trial-and-error loop. Iterating on ONE walk-forward slowly turns
it in-sample (informed_flow's lesson), so every change is judged like this:

1. **One change at a time**, with its hypothesis and expected direction written in
   `RESEARCH_LOG.md` BEFORE its numbers are read.
2. **Fast screen:** the change runs UNTUNED (config values) through all 10 walk-forward test
   periods, paired against the current adopted untuned book on the same days and bootstrap
   blocks (`research.paired`).
3. **Adoption bar** (all of): paired P(Δalpha > 0) ≥ 0.80; Δalpha > 0 in at least 6 of 10
   periods; still > 0 with the best period left out; |beta| not worse by more than 0.05; and a
   reason that is not "it fit 2026".
4. **Full nested walk-forward** (Optuna, 80 trials a period) only for an adopted design, then
   the pre-registered verdict (gates above, unchanged).
5. **The ledger counts every variant tried**; the final report states the number. With ~10
   variants at P ≥ 0.80 one false adoption is expected, which is why the verdict's gates — and
   paper trading — still decide.

## Parked after the 2026-10-20 research loop (RESEARCH_LOG.md has every number)

- **DECISION FOR THE USER: foreign filers.** (Refreshed data, 2026-10-21: excluding them gives
  +5.0% / +0.5% / −0.6% on the top 500 / names 501-1000 / 2020-21, against v5's +10.3 / +4.2 /
  +2.7.) 30% of the long slots are foreign private issuers
  whose short interest FINRA measures on the US line only (RESEARCH_LOG §26); they made a third
  of the long side's profit, most of it in 2025. Excluding them halves the alpha (+10.5% →
  +5.1% on the top 500; +5.5% → +1.6% on names 501-1000). To ship the exclusion the strategy
  needs a foreign-filer flag on the panel: a `features.py` builder reading `fundamentals.form`
  (20-F / 40-F / 6-K) — an engine change, proposal-only. 2026-10-20 (RESEARCH_LOG §30): ranking
  them in a pool of their own loses as much as excluding them (−4.1 / −4.3 / −5.9 pts in the
  three tests), and inside their own pool days-to-cover still sorts returns (IC +0.038). Filling
  the gap with home-market short interest is possible for Canada (CIRO, free, twice monthly
  since 2018-11, 41% of the foreign long P&L), forward-only for Brazil (B3) and Hong Kong, and
  only through a paid lending-data vendor for the UK / EU (Ortex from 2018, S&P Securities
  Finance from 2004, TMX/S3 for Canada + US consolidated). *Test that would settle it:* add CIRO
  short positions and TSX volume to the Canadian names' days-to-cover and run the three-way bar.

- **The two short-side filters, re-tested where they can be judged.** `short_below_ma` 200 (do
  not short a stock above its 200-day average) and `short_max_si` 0.2 helped on the top 500
  (+3.9 and +1.2 pts, P 0.91 / 0.84) but cost 5.0 and 3.9 pts on the never-read names 501-1000
  (P 0.02 each). Both now read as universe-specific or walk-forward luck. *Test that would
  settle it:* a universe neither has seen (e.g. the names ranked 1001-1500, or the top 500 in a
  pre-2020 period once short interest history allows) — adopt only if both hold there.
- **Cutting trades without cutting alpha.** Three changes cut turnover at about the same gross
  alpha (on the v4 base): a 3% band around the short filter (304 → 247 trades/yr, late-fill
  alpha +8.0 → +9.4%), exiting at rank 0.7 (→ 262/yr, +10.3%), shorting only unprofitable firms
  (→ 161/yr, +12.0%, but beta 0.21). None passed the gross-alpha bar; for live trading the
  late-fill alpha is what counts. *Test:* pre-register a cost bar (late-fill alpha up with P ≥
  0.8, gross alpha not down by more than 0.5 pt) and judge them on the v5 base and the fresh band.
- **Inverse-volatility sizing.** Equal slots let a few very volatile names (crypto miners,
  quantum) carry the risk. Under the engine's locked short cash a per-name size leaves cash idle
  or partially fills orders; it needs a sizing rule that keeps each side's gross fixed.
- **Survivorship.** The database holds no name that left before 2026 (no Twitter, VMware, SVB,
  First Republic …): every number here is on survivors. Delisted-name prices would let the
  walk-forward be re-run honestly.
- **Real borrow rates.** Record IBKR's borrow fees and shortable shares daily from now on, to
  replace the tier assumptions for the paper record. The public FTP file works without an
  account (RESEARCH_LOG §30); its 2026-10-20 snapshot charged 0.25-0.45% on every current
  short candidate, far below the 3% / 10% tiers.

- **Loop 2 (2026-10-21) leftovers.** (a) *A cost option:* `dual_band` — leave the band only
  once BOTH the composite and the short-interest rank have left it — halves trading (296 → 131
  orders/yr) and raises the late-fill alpha by 2 pts at −0.8 pts of normal-cost alpha (RESEARCH_LOG
  §32); adopt it if paper-trading fills turn out worse than the open. (b) *A risk choice:* no new
  long in the top tenth of 63-bar volatility cuts the 2026-H2 loss (+14 pts) and the drawdown
  (−14.0% → −10.9%) but gives back 2025; it is the lever if the user wants a calmer book.
  (c) The 200-day short filter, re-tested on complete data: +4.0 on the top 500, −0.4 on names
  501-1000, −2.9 on 2020-21 — still not adopted.

## Ideas (not built)

- **Beta-balanced sides.** TESTED 2026-10-20 as `long_share` 0.43: beta → 0.00 but alpha −0.6
  pts (RESEARCH_LOG §4) — a risk choice, kept as the `long_share` knob at 0.5.
- **A 2026 regime check.** Short interest inverted in 2026 (crowded names rallied). A filter that
  pauses the short side when the bottom fifth has outrun the top over the last 3 months.
  *Danger:* a fitted escape hatch for one bad year; needs its own pre-registration and a
  walk-forward in which it helps in more than one period.
- **The skew sleeve on a broader universe.** It was found on ~1,100 optionable names; the top
  500 may be too efficient for it. *Test:* the sleeve's IC on the 500-1,000 band, point in time.
- **The turnover lean (seen in the first smoke book, 2026-10-19).** TESTED 2026-10-20: ranking
  inside turnover groups (−0.3 pts) and short interest as a share of shares (beta −0.41) both
  rejected; the factor regression puts the high-turnover lean at −0.08..−0.20 and the alpha
  after it at +11..+13%/yr (RESEARCH_LOG §14, §19). Days-to-cover = short interest ÷
  daily volume, so a name that trades a lot of its shares (crypto miners CIFR, MARA; AXTI) has a low
  days-to-cover whatever its short interest (15-19% of shares there) and lands on the LONG side,
  while low-turnover defensives (VZ, MO, KMB) land short. Dollar-volume groups do not remove that:
  turnover (volume ÷ shares) differs inside a group. It makes the book a high-beta-long /
  low-beta-short bet (beta 0.39 in 2026). *Test:* rank inside turnover groups, or rank short
  interest as a share of shares outstanding; *wins if* the walk-forward alpha holds and |beta| falls.
- **Merger-arbitrage shorts.** Acquirers in stock deals are shorted by arbitrageurs as a hedge, not on
  information (KMB/Kenvue, OMC/IPG show days-to-cover of 14-17 in October 2026). *Test:* mask names
  with an announced stock-for-stock deal (SEC 8-K Item 1.01 / S-4 filings) from the short side.
- **Real borrow rates.** IBKR publishes current borrow fees and availability (shortable shares);
  recording them daily from now on would replace the tier assumptions for paper trading.

## Parked after proposal 1 (2026-10-21; RESEARCH_LOG §38P has every number)

- **`sizing: beta` — a risk option, ready to fold in.** Each slot scaled at entry by the median
  beta ÷ the name's trailing 126-bar beta (between 1/1.5 and 1.5), frozen while held, with a gross
  guard; built and tested in `strategies/informed_flow_ls_p1` (`_weights`). On T1: alpha +11.5%
  against +12.6%, max drawdown −8.7% against −14.0%, volatility 8.6% against 10.7%, beta 0.05;
  every untuned gate passes; on the locked names 1501-2000 it costs 0.1 pt. It missed the
  pre-registered cost cap by 0.1-0.6 pts, so it is a choice between a little alpha and a
  third less drawdown — the user's, like N2's cost option (§33). *Folding it in:* copy `_weights`,
  `compose`'s `beta` output and the three knobs into `strategy.py`, and size the luck band with
  `_weights` too (`research.shuffled_returns` sizes equally).
- **Judge T3 with care.** A one-name perturbation of the dollar-volume groups (SPY merged into the
  panels, never traded) moved T3 by −3.9 points (§38P, I3): the three 2020-21 periods are too
  few and too path-sensitive for a ±1-point bar. T1, T2 and TB moved ±0.2-1.0 under the same
  perturbation. A future bar should weigh T3 by its noise, or perturb every candidate a few times.
- **Not worth building (premise checks, §38P):** a long-side drawdown stop (after a 15-25% fall the
  longs did not keep falling outside 2018), volatility-managed gross (the book does worst after
  calm stretches), a sector cap (the largest sector holds 3-6 of 15 names).

## Parked after loop 5 (2026-10-22; RESEARCH_LOG §42-§44 has every number)

- **Put/call open-interest sleeve** (x06): +4.3 on the top 500 in every option-era period, −1.3 on
  names 501-1000 (−21 pts in 2026-H2 at its tuned weight). One year of option data cannot separate
  signal from luck. Revisit after six months of paper trading add history; pre-register the weight
  (1.0, not the tuned 3.75) and judge it on names 501-1000 first.
  Anatomy (2026-10-22, `x06_explain.py` / `x06_attr.py` in the session scratchpad): the whole
  effect is the option year (Oct 2025 → Oct 2026). Top 500 +17 to +20 pts/yr (10-day-block t 1.8-2.0),
  names 501-1000 −6 to −12 (t −0.8 to −1.0), 1001-1500 +4 to +9 (t < 1). Top-500 gain: 10 names
  give more than all of it (MU +5.0 held long while put-heavy; call-heavy ACHR / WULF / CIFR / RKT /
  SMR avoided; RGTI −4.0 the biggest cost); positive in 9 of 13 months. The nested "+4.0, P 0.99"
  is not separate evidence: the 2025-07 fold's training window had NO option data (its pick came
  from noise), and any positive weight wins on the top 500 that year. Even with no data the
  sleeve moves ~2% of ranks (floating-point ties at the bands): ±1-1.5 pts a half-year is noise.
  Longer history: OCC customer put/call VOLUME per stock (rolling 2 years, free; a flow proxy for
  the OI ratio) or paid OI history (MarketData paid plans, ORATS, OptionMetrics).
- **Volatility-scaled short sizes** (x09): +0.9 / +1.2 tuned, T2 −0.1, PBO 0.63, worst period
  −32%. A near miss; worth one re-read on new data, not a re-tune.
- **High-beta shorts first** (x01): the nested tuning found +2.9 (P 0.93) but its consensus
  setting lost 2020-21, and part of the gain is credit for a lower beta. Judge on raw return
  beside alpha if it comes back.
- **13F holder changes** (x05: +2.3 on names 501-1000, +0.3 top 500) and **high-dispersion
  pauses** (x10): no.
- **Fix the remaining split effects causally** (needs a split-events table, engine proposal):
  divide FINRA's shares short by SEC's count × the splits between the 10-Q's period end and the
  settlement date; read the $5 floor on the raw (unadjusted) close.
- **A `si_shares` builder in `features.py`** (engine proposal): so `tsim optimize` and
  `04_compare` read the v8 measure instead of warning and falling back.
