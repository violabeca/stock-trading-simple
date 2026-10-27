# peer_reversal — research log

Proposal 5 of `STRATEGY_OVERVIEW.md` §6, evaluated with the shared lab
(`strategies/proposals_lab/README.md`, `lab.py`). Every number in this file comes from a lab
command or from `premise.py`, `anatomy.py` or `checks.py` in this folder; the command is named
next to it. §0-§13 are round 1 (lab v1); the Round 2 section at the end re-measures everything
on lab v2 and supersedes their numbers.

---

## §0 Pre-registration (written 2026-10-22 22:50, before any number was computed)

### Hypothesis

A stock that fell sharply against its own correlation peers over the past week recovers part of
that gap over the next 5-10 sessions, and a stock that jumped against them gives part of it back.
The effect is stronger when the market is stressed. Measured against peers, the move is mostly
non-information price pressure. The peers' common move (industry or style news) is removed, and
that part continues instead of reversing.

### Mechanism

- **Liquidity provision.** Liquidity-demanding trades push prices away from value: fund flows,
  forced sellers, margin calls, index trades, one big seller. Whoever absorbs the imbalance is
  paid by the reversal (Lehmann 1990; Lo & MacKinlay 1990; Avramov, Chordia & Goyal 2006).
- **Stress raises the price of liquidity.** Liquidity providers' capital is scarcest when
  volatility is high, so expected reversal profits rise with VIX and peak in crises (Nagel 2012,
  "Evaporating Liquidity").
- **Peers strip out news.** Raw short-term reversal mixes two components:
  - industry and common news, which continues (Moskowitz & Grinblatt 1999, industry momentum);
  - idiosyncratic price pressure, which reverses.

  Peer-relative and residual reversal keep only the second component, and residual reversal
  survives costs in large caps (Da, Liu & Schaumburg 2014; Hameed & Mian 2015; Blitz, Huij,
  Lansdorp & Verbeek 2013). Pairs trading is the same idea with a single peer (Gatev, Goetzmann
  & Rouwenhorst 2006; Chen, Chen, Chen & Li 2019).
- **Skipping one day.** It removes bid-ask bounce and closing-print artifacts, which reverse
  mechanically but cannot be captured with market orders.

### The untuned default U0 (literature defaults, every value fixed here)

| element | U0 |
|---|---|
| universe | the lab's point-in-time top 500 (T1; E0/TB/T3 for TRAIN), names 501-1000 (T2) |
| peer groups | **30 correlation clusters**. Input: each name's trailing **252-bar market-residual** daily returns (residual = return − β × the equal-weight return of the panel's names, β fitted on the same 252 bars), z-scored per name. Clustering: **Ward-linkage hierarchical clustering** (`scipy.cluster.hierarchy.linkage(..., "ward")`), cut into at most 30 clusters (`fcluster(..., "maxclust")`). A name needs ≥ 80% valid returns in the window to be clustered. Re-formed every **126 bars**, anchored at the panel's first bar (first formation at bar 252), from past data only. Labels formed at bar b's close are used by decisions from bar b on. |
| peer return | the mean 5-day return of the OTHER members of the name's cluster that have a valid return (leave-one-out). At least 2 such peers, else the name is ineligible on that bar. |
| signal | `R5 − peer mean R5`, where `R5 = close[t−1] / close[t−6] − 1`. That is the 5-day return measured to the close ONE DAY BEFORE the decision bar t (skip one day); the decision is made at t's close and fills at t+1's open. |
| eligibility | close ≥ $5 on the decision bar, a valid signal |
| ranking | percentile rank of the signal among eligible names, per bar |
| entries | long: rank ≤ 0.10 (bottom tenth, fell most against peers); short: rank ≥ 0.90 (top tenth) and shortable. Most extreme candidates first, into free slots. |
| exits | long when its rank rises above 0.5 (back at or above the peer-relative median: the reversal is done), short when it falls below 0.5, or after **10 bars** (`max_hold`). The clock restarts while the name is still in its entry zone: a fresh extreme 5-day move is a new signal, and selling then re-buying it would only pay costs. |
| slots | 20 long + 20 short |
| sizing | dollar-neutral, equal slots. Each slot is `gross / 2 / 20` of the book, set at its entry bar from that bar's stress state and frozen while held (the engine trades a name only when its target changes). |
| stress switch (binary) | stress = the market's 21-bar realized volatility above its trailing 252-bar median, both known at the decision bar's close. The market is the equal-weight mean daily return of the panel's names. **gross 0.98 when stressed, 0.49 otherwise**, applied at entries only. Gross never exceeds 40 × 0.0245 = 0.98. |
| borrow | informed_flow_ls_p1's tiers. Short interest ÷ shares outstanding (days-to-cover × 20-bar median volume ÷ `fund_shares`): ≤ 10% → 0.5%/yr, 10-20% → 3%/yr, > 20% → 10%/yr. No short-interest data (all of E0, before 2018) → 0.5%/yr. A name above 30% is never shorted (hard to borrow; squeeze risk). |
| execution | portfolio mode, fill at the next open, `order_notional` 1000, the repo's cost model |

### Controls (each counts as a variant)

- **C1, sector peers:** the lab's `sector` codes (12 groups) replace the clusters; everything
  else is U0.
- **C2, no stress switch:** constant gross **0.735**, the switch's two levels averaged; stress
  is on about half the days by construction. Same mean exposure as U0, no timing, so the
  paired Δ measures the timing alone. A constant 0.98 would mostly measure more exposure.

### Premise check (TRAIN only: E0 + TB + T3, informational)

`premise.py` will report, on TRAIN periods only:

- the daily cross-sectional Spearman IC of −signal against beta-adjusted forward returns, from
  the fill (`open[t+1]`) to `open[t+1+h]`, h = 5 and 10. Beta is the trailing 126-bar beta to
  the equal-weight market. Positive = reversal.
- the bottom-minus-top decile spread of the same forward returns;
- the same split by stress on / off, and for cluster peers, sector peers and no peers (a raw
  5-day return relative to the market).

### Expected result (priors, not measurements)

- **Premise.** Sign-adjusted IC around +0.01 to +0.03 at 5 days, larger on stress days. Decile
  spread around +0.3-0.6% per 5 days gross. Repo evidence: weekly and monthly reversal among the
  500 most-traded names had t 1.0-1.7, "carried by 2020 and 2022".
- **Turnover and costs.** Book turnover around 25-35× a year. Each leg costs about 12 bp at a
  $1,000 order (the $1 minimum is 10 bp, plus 1 bp spread and 1 bp slippage), so fees run about
  5-8%/yr. U0's net alpha is expected between −4% and +4%/yr.
- **Late fills.** +50 bp a leg costs about 25-35%/yr at this turnover. **G5 is expected to
  FAIL**, and probably every design that holds about a week will fail it too.
- **Market risk.** Beta near 0 (dollar-neutral, peer-relative), so G7 is likely to pass. The
  bear-episode half of G6 should pass, since a neutral book beats a falling market. The
  down-month mean > 0 is uncertain.
- **Overall.** The most likely verdict is "no edge" net of costs. The most likely positive
  finding is a gross reversal premium concentrated on stress days.

### Optimization (step 4)

`lab.tune` on TRAIN only, 120 TPE trials, the lab's robust objective (mean − ½ sd of the 11
period alphas). Searched knobs:

| knob | range |
|---|---|
| `n_clusters` | 10-60, step 5 |
| `formation` | 3-21 bars |
| `q_in` | 0.05-0.20 |
| `q_out` | 0.3-0.7 |
| `max_hold` | 5-25 bars |
| `slots` | 10-30, step 5 |

The stress switch and the skip day are NOT searched; they are pre-registered. The PBO is
recorded. The tuned values are adopted only if they beat U0 on both T1 and T2, paired.

### Iterations (step 5)

At least two, each pre-registered here, with its mechanism, before it is run. Each is adopted
only if its paired Δ against the current base is ≥ 0 on TRAIN, T1 and T2, and its T1 Δ exceeds
twice U0's noise sd. Candidates written down now, from the proposal:

- drop moves made on > 3× normal volume (information, not liquidity);
- 1- vs 5- vs 21-day formation;
- hold length;
- trading only in stress;
- the cluster count;
- turnover-reducing bands.

### Variant budget

At most **8** variants through the bar (tuner trials excluded):

| | variants |
|---|---|
| planned | U0, C1, C2, the tuned values T, and iterations I1 and I2 |
| reserve | 2 |

Hard cap: 10.

### Proposal-specific pass bars (STRATEGY_OVERVIEW §6, Proposal 5)

These are read beside the lab's gates G1-G8:

- alpha > 0 net at 2× costs (P ≥ 0.90);
- beats the shuffled books;
- most of the profit is earned on stress-on days (if not, the bear-market rationale is false);
- positive in 2020 and 2022.

---

## §1 Premise check, TRAIN only (2026-10-22, `premise.py`, log `var/proposals_lab/results/peer_reversal/premise.log`)

**Setup.**
- Decision bars inside the 11 TRAIN periods (E0 + TB + T3, 2016-07 → 2021-12), U0's signal.
- Outcome: the beta-adjusted return from the fill (`open[t+1]` → `open[t+1+h]`).
- IC = daily Spearman of −signal (positive = reversal).
- Spread = bottom-tenth minus top-tenth mean outcome, in % per h days.
- t = Newey-West with h lags.

| peers | h | days | IC | IC t | spread % | spread t |
|---|---|---|---|---|---|---|
| cluster (U0) | 5 | all | +0.0080 | 1.84 | +0.153 | 1.45 |
| cluster | 5 | stress (555 d) | +0.0130 | 1.60 | +0.273 | 1.19 |
| cluster | 5 | calm (765 d) | +0.0043 | 1.01 | +0.066 | 0.93 |
| cluster | 10 | all | +0.0067 | 1.44 | +0.177 | 1.56 |
| sector | 5 | all | +0.0100 | 1.65 | +0.136 | 1.10 |
| sector | 5 | stress | +0.0184 | 1.80 | +0.333 | 1.33 |
| sector | 5 | calm | +0.0038 | 0.56 | −0.007 | −0.07 |
| market (no peers) | 5 | all | +0.0133 | 1.75 | +0.169 | 1.16 |
| market | 5 | stress | +0.0159 | 1.27 | +0.324 | 1.11 |
| market | 5 | calm | +0.0114 | 1.29 | +0.056 | 0.43 |

By year (cluster, h 5), IC / spread:

| 2016 | 2017 | 2018 | 2019 | 2020 | 2021 |
|---|---|---|---|---|---|
| +0.007 / +0.08% | −0.000 / +0.04% | +0.012 / +0.18% | +0.006 / +0.09% | **+0.020 / +0.47%** | +0.003 / +0.02% |

Sector and market peers were negative in 2021 (IC −0.010 / −0.011).

**Read.**
- **The reversal is there but weak** (t 1.4-1.8), and about **4× stronger on stress days** in
  spread: +0.27% vs +0.07%. This matches Nagel's mechanism and the repo's "carried by 2020"
  note. *[Review fix §13.3: point estimates only; the stress and calm spreads have t 1.19 and
  0.93.]*
- **Correlation-cluster peers do NOT beat plain market-relative reversal** on TRAIN: IC 0.008 vs
  0.013. The Da-Liu-Schaumburg "peers strip the news" argument is not visible in this universe
  at this horizon.
- **Economics.** A tenth-vs-tenth spread of 0.15% per 5 days is about 7.7 bp per unit of gross
  per trade, against a 24 bp round trip at the lab's $1,000 order (12 bp a leg). Even on stress
  days, 0.27% / 2 = 13.7 bp is below 24 bp.
- **Implication for U0.** It should lose money net of costs unless its "most extreme 20 of the
  tenth" selection and the band exits capture much more than the tenth's average. Expected U0
  net alpha: about −3 to −6%/yr. Nothing is decided here; U0 runs as pre-registered.

## §2 U0, the untuned default (2026-10-22)

`lab.py run peer_reversal U0 '{}' E0,TB,T3,T1,T2 --costs2`. The tests are passed explicitly: the
CLI reads a 4th positional argument as the test list, so `--costs2` alone is parsed as a test
name (lab issue, worked around).

| test | alpha | P | beta | vol | maxDD | orders/yr | late alpha | 2× costs | top5-out |
|---|---|---|---|---|---|---|---|---|---|
| E0 | −5.45% | 0.02 | 0.00 | 3.6% | −11.1% | 3,116 | −26.2% | −10.8% | −7.4% |
| TB | −5.09% | 0.28 | 0.12 | 11.7% | −27.3% | 3,057 | −29.8% | −11.8% | −9.8% |
| T3 | −8.27% | 0.02 | −0.00 | 7.8% | −15.2% | 3,131 | −27.8% | −13.3% | −10.5% |
| **T1** | **−1.11%** | 0.36 | 0.15 | 9.1% | −12.8% | 3,096 | **−24.2%** | −7.1% | −5.0% |
| **T2** | **−6.69%** | 0.04 | 0.08 | 10.0% | −28.3% | 3,106 | −29.3% | −12.7% | −8.6% |

- **TRAIN stitched:** alpha −7.37% (P 0.03), late −29.2%, 2× costs −13.1%.
- **Per-period T1 alphas:** +0.238 (2022-H1), −0.029, +0.055, −0.106, −0.030, −0.063, +0.028,
  −0.059, −0.010, +0.035.
- **Costs are about 6%/yr** (2× costs − base, T1 −6.0 pts, TRAIN −5.7 pts). Gross of costs, U0
  is about −1.7%/yr on TRAIN, +4.9% on T1 and −0.7% on T2.
- **Bear episodes** (book vs equal-weight market):

  | Q4 2018 | COVID | 2022 bear | Feb-Apr 2025 |
  |---|---|---|---|
  | −0.4% vs −18.2% | −12.6% vs −37.0% | +6.9% vs −25.2% | +2.9% vs −19.6% |

  Down months (41, E0-T1): book mean −0.80% vs market −3.97%, 29% positive.

**Noise** (`lab.py noise peer_reversal U0`: 5 names dropped a period, 3 copies). Alpha deltas
and sd:

| test | deltas | sd |
|---|---|---|
| E0 | −1.7 / −2.1 / −1.6 | 0.30 |
| TB | −1.2 / +1.0 / −5.0 | 3.05 |
| T3 | +1.4 / −0.6 / +0.4 | 0.97 |
| **T1** | −3.3 / −1.9 / −0.5 | **1.40** |
| T2 | +1.8 / +1.3 / −0.5 | 1.16 |

**The adoption bar for an iteration is therefore a T1 Δ > 2.8 points.**

**Luck** (`lab.py luck peer_reversal U0`: T1, 100 shuffled-name books inside dollar-volume
tenths). Real −1.11% vs median −5.89%, p95 −2.61%: **percentile 100**. ~~The selection is real.~~
The real book beats every same-turnover random book by about 4.8 pts/yr, which is about what
costs take (random books ≈ −cost). *[Review fix §13.1: suggestive, not established. The three
noise copies (5 names dropped a period) sit at percentiles 75, 92 and 100 of this band, and no
TRAIN premise statistic reaches t 2.]*

**Trade anatomy on TRAIN** (`anatomy.py '{}'`, log `anatomy-U0.log`; gross, market-adjusted, no
costs):

- 8,422 trades, mean 6.3 bars, **+2.4 bp a trade** against the 24 bp round trip.
- **Longs +11.6 bp (t 1.2), shorts −6.5 bp (t −0.6).** The short side loses even gross. The DB
  is survivors only, which understates shorts.
- **Stress entries +4.2 bp, calm +1.1 bp:** no reliable difference (t ≈ 0.3 each).
- Formation-volume terciles (−9.3 / +17.7 / −1.1 bp) and extremeness terciles (+11.8 / −17.5 /
  +12.9 bp) are non-monotonic: noise.
- **Reversal curve of every entry (bp from the fill, whatever the book did):**

  | | day 1 | day 5 | day 10 | day 15 | day 21 |
  |---|---|---|---|---|---|
  | all | +0.8 | −1.0 | +4.2 | +14.3 | **+28.2** |
  | long | | | +17 | | +59 |
  | short | | | −8 | | −2 |

  The reversal accrues mostly AFTER U0's ~6-bar hold, and the first week is flat: U0 sells
  before it is paid. A longer hold or a slower signal is the first lever. The premise check's
  4× stress premium (§1, all tenth names) does not show in the traded subset.
  *[Review fix §13.3: none of these curve points is significant (Newey-West t over entry dates:
  day 21 all t 1.03, longs 1.10, shorts −0.67), and calm entries reverted MORE than stress
  entries (+36.3 vs +16.9 bp at day 21; difference t −1.08, −1.86 at day 10).]*

## §3 Controls C1 (sector peers) and C2 (no stress switch) — variants 2 and 3 (2026-10-22)

`lab.py run peer_reversal C1 '{"peers": "sector"}' … --costs2` and
`C2 '{"gross_stress": 0.735, "gross_calm": 0.735}'`; `lab.py report peer_reversal U0 C1 C2`.

| | E0 | TB | T3 | T1 | T2 |
|---|---|---|---|---|---|
| U0 alpha | −5.5% | −5.1% | −8.3% | −1.1% | −6.7% |
| C1 sector peers | −3.3% | −1.6% | −8.5% | −2.0% | −0.8% |
| C1 Δ vs U0 (P) | +2.1 (0.89) | +3.5 (0.89) | −0.2 (0.50) | −0.9 (0.40) | **+5.9 (0.97)** |
| C2 constant 0.735 | −4.4% | −6.3% | −10.8% | −0.4% | −6.0% |
| C2 Δ vs U0 (P) | +1.1 (0.84) | −1.2 (0.26) | −2.5 (0.02) | +0.7 (0.75) | +0.7 (0.70) |

Bear episodes (book vs market):

| | Q4 2018 | COVID | 2022 | Feb-Apr 2025 | down months |
|---|---|---|---|---|---|
| C1 | +2.5% | −10.1% | +2.2% | +4.8% | −0.59% |
| C2 | +0.6% | −8.6% | +6.2% | +4.2% | −0.67% |

**Read.**
- **The correlation clusters add nothing over 12 coarse sector codes.** Sector peers are as good
  or better on 3 of 5 tests and clearly better on T2. With the premise check (§1: plain
  market-relative IC ≥ cluster IC), the peer-relative part of the proposal is not supported here.
- **The binary stress switch adds nothing measurable either.**
  - At ~~the same~~ about the same mean exposure, the constant book is level on T1/T2 (+0.7, +0.7)
    and on E0. *[Review fix §13.4: C2 ran 5-8% more realized gross than U0, 0.73-0.74 vs
    0.68-0.71, because stress was on 42-47% of days, not half.]*
  - The switch helps only on T3 (+2.5, P 0.98) and TB (+1.2).
  - Both are TRAIN periods with sharp vol spikes (COVID); the anatomy found no per-trade
    stress premium in the traded names.
- Both are controls, not candidates; U0 stays the base.

## §4 Optimization on TRAIN — variant 4, T (2026-10-22)

`lab.py tune peer_reversal T '{}' '<space §0>' 120 E0,TB,T3` (journal
`var/proposals_lab/studies/peer_reversal-T.log`).

- **Best:** n_clusters 50, formation 18, q_in 0.05, q_out 0.5, max_hold 6, slots 10.
- **Score:** robust +2.91% (mean period alpha +6.32%). **PBO 0.71.** Only 18 of 120 trials score
  above 0.
- **The surface:**
  - Mean score by formation: −6.7% (4-8 bars), −5.3% (9-12), −3.1% (13-16), −1.9% (17-21).
    The rise is monotone across all trials. *[Review fix §13.4: TPE put 8 / 3 / 26 / 83 trials
    in these bins, so the means are confounded by its adaptive sampling; the 20 random start-up
    trials alone point the same way (Spearman +0.58).]*
  - The top trials ~~all~~ sit at the lower bounds q_in 0.05 (7 of the top 10) and slots 10
    (9 of 10).
  - max_hold varies from ~~6~~ 5 to 17 among the top 10: it barely matters.

`lab.py run peer_reversal T '<best>' E0,TB,T3,T1,T2 --costs2`; `lab.py report peer_reversal U0 T`:

| | E0 | TB | T3 | T1 | T2 |
|---|---|---|---|---|---|
| T alpha | (Δ +11.5) | +5.9% | +3.0% | **−2.1%** | **−10.9%** |
| Δ vs U0 (P) | +11.5 (1.00) | +11.0 (0.86) | +11.3 (0.91) | **−1.0 (0.41)** | **−4.2 (0.23)** |
| maxDD | | −12.9% | −9.6% | −29.0% | **−50.4%** |
| orders/yr | 669 | 683 | 639 | 668 | 686 |
| late alpha | | −6.9% | −6.5% | −12.4% | −20.9% |

- **Not adopted.** It is worse than U0 on both T1 and T2. The TRAIN gain is in-sample fit, as the
  PBO said. This is the repo's standing lesson: tuning fits noise.
- **Kept as a lesson, not a value:**
  - A slow (16-21-bar) formation trades about 4.6× less. The TRAIN-wide rise of the score with
    formation is ~~structural~~ visible among the random start-up trials too, so it does not come
    from one lucky point. It matches the anatomy's slow reversal curve (§2). *[Review fix §13.4:
    it is in-sample, mostly the cost saving of slower ranks, and the clean A/B (I1) did not carry
    it to T1 (Δ −1.8).]*
  - The concentrated corner (5% zone, 10 slots) carries fat tails (T2 −50% drawdown).

**U0 stays the base.**

## §5 Iteration I1 — monthly formation at U0's breadth (variant 5). Pre-registered 2026-10-22 23:33, before the run (time corrected in review, §13.5)

- **Change.** `formation: 21` and `max_hold: 21` (the classic one-month reversal: Jegadeesh 1990;
  Lehmann 1990). Everything else is U0: 30 clusters, q_in 0.1, q_out 0.5, 20 + 20 slots,
  skip 1, the stress switch, borrow.
- **Mechanism.**
  1. The anatomy (§2) shows the price pressure of a one-week loser reverses over the next ~3
     weeks, not in the first week. A one-month formation measures the accumulated pressure, the
     horizon where reversal is documented in large caps; the repo found the most-traded names
     reverse at one month.
  2. A 21-bar window moves slowly, so ranks cross the band far less often. Turnover should fall
     about 3-4× (T: 670 orders/yr at 10 slots, so about 1,300 at 20), and costs from ~6% to about
     2%/yr.
  3. U0's breadth is kept on purpose, because the tuned corner (5%, 10 slots) did not carry and
     has fat tails.
- **Expected.** Δ alpha +2 to +5 pts on TRAIN, T1 and T2, mostly from costs; late-fill alpha still
  negative.
- **Adopted only if** Δ ≥ 0 on TRAIN (stitched E0 + TB + T3), T1 and T2, and the T1 Δ is
  > 2.8 pts (2 × U0's noise sd).

### I1 result (`lab.py run peer_reversal I1 '{"formation": 21, "max_hold": 21}' … --costs2`)

Paired against U0; TRAIN is E0 + TB + T3 stitched (`lab.paired` on the stitched record):

| | alpha (P) | Δ vs U0 (P) | maxDD | orders/yr | late | 2× costs |
|---|---|---|---|---|---|---|
| TRAIN | −2.39% (0.26) | +4.98 (0.86) | −18.5% | 962 | −9.9% | −4.2% |
| T1 | −2.88% (0.22) | **−1.77 (0.32)** | −29.5% | 1,016 | −10.7% | −4.8% |
| T2 | −2.60% (0.26) | +4.09 (0.78) | −26.5% | 1,022 | −10.5% | −4.6% |

- **Not adopted:** T1 Δ < 0.
- **Turnover** fell 3× (late-fill alpha −24% → −11%), as predicted. The gross edge did not grow
  enough to make it pay.
- **Per-period T1 alphas** swing widely:

  | 2022-H1 | 2022-H2 | 2023-H1 | 2023-H2 | 2024-H1 | 2024-H2 | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 |
  |---|---|---|---|---|---|---|---|---|---|
  | +0.09 | +0.15 | +0.08 | −0.13 | −0.03 | −0.16 | +0.04 | −0.07 | −0.16 | +0.31 |

  The monthly reversal earns in the 2022 bear and the 2026 factor crash, and loses in the
  trending halves (2023-H2, 2024-H2, 2026-H1), where shorting monthly winners hurts.

**Where the P&L is earned.** Daily book returns minus beta × the market, split by the stress
state at the previous close (`scratchpad/stress_split.py`), in % a year of all days:

| | TRAIN stress / calm | T1 stress / calm | T2 stress / calm |
|---|---|---|---|
| U0 | −3.5 / −3.7 | +1.1 / −2.0 | −2.1 / −4.4 |
| C2 (constant gross) | −2.6 / −4.8 | +1.3 / −1.5 | −1.3 / −4.5 |
| I1 | −0.2 / −1.9 | +0.0 / −2.5 | −0.2 / −2.0 |

- At constant exposure, a calm day costs more than a stress day in every sample: −3.3 vs −2.5
  bp/day on TRAIN, −1.1 vs +1.2 on T1, −3.4 vs −1.1 on T2. That ~~supports~~ points in Nagel's
  direction. *[Review fix §13.3: the stress − calm differences have t +0.27, +0.82, +0.68
  (`checks.py stress`); the calm-day levels are mostly costs, paid every day.]*
- But **no variant earns money on stress days after costs** on TRAIN or T2.
- **The proposal's pass bar "most of the profit is earned on stress days" cannot be met: there
  is no profit.**

## §6 Iteration I2 — trade only in stress (variant 6). Pre-registered 2026-10-22 23:36, before the run (time corrected in review, §13.5)

- **Change.** `gross_calm: 0`: no new position opens on a calm bar. Positions opened in stress
  keep their full size until they exit. Everything else is U0. (I1 was not adopted, so the base
  is U0.)
- **Mechanism.**
  - Nagel 2012: liquidity providers are paid in stress.
  - Premise (§1): the tenth-vs-tenth spread is +0.27%/5 d on stress days, +0.07% on calm days.
  - The split above: calm-day P&L is negative for every variant in every sample, and calm
    entries pay the same 24 bp round trip for an edge near zero.
  - About 58% of U0's entries are calm entries, so turnover and costs should fall by about half.
- **Expected.** Δ +1.5 to +3.5 pts on TRAIN, T1 and T2. Alpha probably still ≤ 0 net.
- **Adopted only if** Δ ≥ 0 on TRAIN, T1 and T2, and the T1 Δ is > 2.8 pts.

### I2 result (`lab.py run peer_reversal I2 '{"gross_calm": 0.0}' … --costs2`)

| | alpha (P) | Δ vs U0 (P) | maxDD | orders/yr | late | 2× costs |
|---|---|---|---|---|---|---|
| TRAIN | −5.61% (0.06) | +1.76 (0.94) | −30.9% | 1,476 | −21.1% | −9.4% |
| T1 | −0.77% (0.39) | +0.34 (0.57) | −14.4% | 1,565 | −17.4% | −4.8% |
| T2 | −6.31% (0.04) | +0.38 (0.56) | −25.8% | 1,684 | −23.3% | −10.7% |

**Not adopted.** Every Δ is ≥ 0, but the T1 Δ of +0.34 is far below the 2.8-pt bar. Halving the
entries halved the orders but saved only about 0.4-1.8 pts.

Stress split for I2, % a year:

| | stress days | calm days |
|---|---|---|
| TRAIN | −4.3 | −1.1 |
| T1 | +0.2 | −0.8 |
| T2 | −3.6 | −2.5 |

The full-size stress entries lose too, after costs.

## §7 Iteration I3 — volatility-scaled signal (variant 7). Pre-registered 2026-10-22 23:38, before the run (time corrected in review, §13.5)

- **Change.** `scale: vol`. The peer-relative 5-day move is divided by the name's own volatility:
  the standard deviation of its daily return minus its peers' mean, over the 63 bars ending
  with the formation window. Everything else is U0. The code was added off by default and
  tested (`test_vol_scaling_ranks_a_move_unusual_for_its_name_as_more_extreme`).
- **Mechanism** (Blitz, Huij, Lansdorp & Verbeek 2013, "Short-term residual reversal"):
  - Raw extremes are dominated by high-volatility names, whose large moves are mostly their
    ordinary noise or news: little reversal, more squeeze risk, larger losses on the short side.
    The TRAIN shorts lose −6.5 bp a trade.
  - Scaling selects moves that are unusual FOR THAT NAME, which is the signature of price
    pressure. Blitz et al. find residual reversal scaled by residual volatility earns
    substantially more per unit of risk and survives costs among large caps.
  - Turnover is expected to stay about U0's.
- **Expected.** A higher gross edge per trade and lower book volatility. Δ +1 to +3 pts if the
  mechanism holds, but costs are unchanged, so the net alpha is still probably < 0.
- **Adopted only if** Δ ≥ 0 on TRAIN, T1 and T2, and the T1 Δ is > 2.8 pts.

### I3 result (`lab.py run peer_reversal I3 '{"scale": "vol"}' … --costs2`)

| | alpha (P) | Δ vs U0 (P) | maxDD | orders/yr | late | 2× costs |
|---|---|---|---|---|---|---|
| TRAIN | −6.38% (0.02) | +0.99 (0.68) | −33.4% | 3,117 | −28.5% | −12.2% |
| T1 | −3.46% (0.04) | **−2.35 (0.20)** | −17.7% | 3,126 | −26.4% | −9.5% |
| T2 | −6.03% (0.01) | +0.66 (0.58) | −29.2% | 3,062 | −28.3% | −11.9% |

**Not adopted** (T1 Δ < 0). Scaling by the name's own volatility does not raise the edge here,
and the residual-reversal result of Blitz et al. does not replicate in this universe and window.

**The volume cap (the proposal's named variant) is not run.** The TRAIN anatomy (§2) already
answers its premise. By formation-volume tercile the gross trade returns are:

| low | mid | high |
|---|---|---|
| −9.3 bp | +17.7 bp | −1.1 bp |

High-volume moves do not continue more than low-volume ones, so a cap at 3× normal volume would
drop the better-than-low tercile. The variant slot goes to I4 instead. *[Review fix §13.3: the
terciles have t −0.74 / +1.47 / −0.10, noise either way; the cap was skipped for the budget, not
on evidence.]*

## §8 Iteration I4 — monthly formation + stress-only entries (variant 8, the last in budget). Pre-registered 2026-10-22 23:39, before the run (time corrected in review, §13.5)

- **Change.** I1 + I2 together: `formation: 21`, `max_hold: 21`, `gross_calm: 0`. Everything else
  is U0.
- **Mechanism.** It attacks the binding constraint, costs, from two independent sides, and cuts
  the regime that loses:
  1. The monthly formation cuts orders 3× (I1: 3,100 → 1,000 a year) and matches the slow
     reversal curve (§2).
  2. Stress-only entries halve them again (I2).
  3. I1's losses sit on calm days (§5 split: calm −1.9 / −2.5 / −2.0 %/yr against stress ≈ 0 on
     TRAIN / T1 / T2). These are the trending calm halves where shorting monthly winners hurts.
     Stress-only removes most of the entries that run into them.
- **Expected.** About 500 orders a year, costs about 1%/yr, late-fill drag about 5%/yr. Δ vs U0
  about +5 TRAIN, 0 to +3 T1, +4 T2. The T1 bar (> 2.8) is likely not met.
- **Adopted only if** Δ ≥ 0 on TRAIN, T1 and T2 and the T1 Δ > 2.8 pts. Otherwise U0 is the final
  candidate.

### I4 result (`lab.py run peer_reversal I4 '{"formation": 21, "max_hold": 21, "gross_calm": 0.0}' … --costs2`)

| | alpha (P) | Δ vs U0 (P) | beta | maxDD | orders/yr | late | 2× costs |
|---|---|---|---|---|---|---|---|
| TRAIN | −2.46% (0.25) | +4.91 (0.87) | 0.08 | −14.4% | 587 | −9.0% | −4.1% |
| **T1** | **+1.73% (0.71)** | **+2.84 (0.77)** | 0.07 | −14.6% | 605 | −5.0% | +0.1% |
| T2 | −0.98% (0.41) | +5.71 (0.86) | 0.08 | −21.4% | 671 | −8.2% | −2.7% |

- **Per-period alphas:**

  | test | per half-year |
  |---|---|
  | T1 | +0.062, +0.175, +0.091, −0.074, +0.069, −0.175, +0.046, −0.024, +0.028, +0.218 |
  | T2 | +0.141, −0.008, −0.021, −0.014, −0.131, −0.000, −0.014, −0.042, +0.184, −0.013 |
  | TRAIN | −0.049, +0.045, −0.035, +0.031, −0.073, +0.120, +0.001, −0.065, −0.090, +0.045, +0.002 |

- **Bear episodes** (book vs market):

  | Q4 2018 | COVID | 2022 bear | Feb-Apr 2025 | down months |
  |---|---|---|---|---|
  | +2.3% vs −18.2% | −1.8% vs −37.0% | +4.6% vs −25.2% | +1.6% vs −19.6% | book +0.10% vs market −3.97% (41% > 0) |

- **Stress split**, % a year:

  | | stress days | calm days |
  |---|---|---|
  | TRAIN | −0.7 | −1.4 |
  | T1 | **+2.5** | −0.5 |
  | T2 | **+1.2** | −1.7 |

  On T1 and T2 the money IS made on stress days, the proposal's criterion. *[Review fix §13.3:
  as point estimates only: t +0.77 / −0.21 on T1, +0.31 / −0.72 on T2; stress − calm t +0.77
  and +0.61.]*

**ADOPTED, by a hair.** Every Δ is ≥ 0. The T1 Δ is **0.02837 against the bar 0.02792**
(2 × 0.013958), a margin of 0.045 pts. This is the 4th iteration read on T1 (8 variants in
all), so the bar's margin is far smaller than the selection effect of having tried four:
treat the adoption as marginal. *[Review fix §13.2: re-read on each of the three noise copies
(the same names dropped from U0 and I4), the T1 Δ is +2.27, +0.14 and +0.96: below the bar every
time, while TRAIN (+6.1 to +7.6) and T2 (+5.0 to +7.5) stay positive. I4 is a lower-turnover
version of U0, not a T1 improvement.]* *[Round 2 (lab v2, R2.1): the T1 Δ is +0.36 against a
2.60 bar, so I4 is NOT adopted; the final is U0.]*

## §9 FINAL CANDIDATE (declared 2026-10-22 23:41, before verdict, bear and T4; time corrected in review, §13.5)

*[Round 2: superseded. Under lab v2 I4 fails the adoption bar, and the final is U0 (R2.1a).]*

**I4 = U0 + `formation: 21`, `max_hold: 21`, `gross_calm: 0`.** A one-month peer-relative
(30-cluster) reversal, skip one day, bottom / top tenth, 20 + 20 slots, exit at the median band
or after 21 bars, opened ONLY when the market is stressed (21-bar vol above its 252-bar median)
at gross 0.98, dollar-neutral *(in its targets; approximately in practice: §13.6)*, borrow
tiers.

**Variants through the bar: 8 of the pre-registered 8:**

| | variant | outcome |
|---|---|---|
| 1 | U0 | the base |
| 2 | C1 | control |
| 3 | C2 | control |
| 4 | T | not adopted |
| 5 | I1 | not adopted |
| 6 | I2 | not adopted |
| 7 | I3 | not adopted |
| 8 | I4 | adopted |

**Next, in this order:**
1. `lab.py luck peer_reversal I4` (G4) and `lab.py noise peer_reversal I4` (informational);
2. `lab.py verdict peer_reversal I4`;
3. `lab.py bear peer_reversal U0 I4`;
4. T4, read once: `lab.py run peer_reversal I4 '<I4>' T4` and `lab.py run peer_reversal U0 '{}' T4`.

## §10 The final candidate I4 through the gates (2026-10-22 23:41-23:46, before T4; date corrected in review, §13.5)

**Luck** (`lab.py luck peer_reversal I4`, T1, 100 shuffled-name books): real +1.73% vs median
−1.13%, p95 +1.87%, **percentile 93**. Compare U0's percentile 100, a 4.8-pt gap to random;
I4's gap is 2.9 pts.

**Noise** (`lab.py noise peer_reversal I4`), alpha deltas and sd:

| test | deltas | sd |
|---|---|---|
| E0 | +1.6 / +0.8 / +0.5 | 0.59 |
| TB | +0.2 / +0.8 / +0.6 | 0.32 |
| T3 | +1.4 / +3.3 / +0.4 | 1.45 |
| **T1** | **−3.9 / −4.6 / −2.4** | 1.13 |
| T2 | +1.0 / +0.7 / +1.3 | 0.32 |

**Every T1 copy loses 2.4-4.6 pts.** I4's T1 alpha rests on the exact name set. Its adopting
Δ (+2.84 vs U0) is about the size of what dropping 5 names a period takes away.

**Verdict** (`lab.py verdict peer_reversal I4`):

| gate | value | passed |
|---|---|---|
| G1 TRAIN-era alpha > 0 | −2.46% | **no** |
| G2 T1 alpha > 0, P ≥ 0.90 | +1.73% (P 0.71) | **no** |
| G3 T2 alpha > 0 | −0.98% | **no** |
| G4 T1 beats ≥ 95% of shuffled books | 93 | **no** |
| G5 T1 late-fill and 2×-cost alpha > 0 | late −4.95% (2× +0.1%) | **no** |
| G6 every bear episode beaten, down months > 0 | down-month mean +0.10% | yes |
| G7 \|T1 beta\| ≤ 0.3 | 0.07 | yes |
| G8 T1 alpha without its top 5 names > 0 | −1.50% | **no** |

**Bear table** (`lab.py bear peer_reversal U0 I4`; stitched E0 + TB + T3 + T1):

| | Q4 2018 | COVID | 2022 bear | Feb-Apr 2025 | CAGR | max DD |
|---|---|---|---|---|---|---|
| U0 | −0.4 | −12.6 | +6.9 | +2.9 | −3.0 | −34.3 |
| **I4** | **+2.3** | **−1.8** | **+4.6** | **+1.6** | **+0.6** | **−14.6** |
| market (EW) | −18.2 | −37.0 | −25.2 | −19.6 | | |
| SPY | −18.7 | −33.4 | −24.1 | −18.8 | | |

Calendar years, %:

| | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U0 | −3.7 | −2.5 | −6.5 | −6.1 | −6.5 | −5.9 | +4.3 | −1.8 | −3.4 | +0.7 | +1.0 |
| I4 | −2.9 | −1.1 | −2.8 | +4.7 | **−5.2** | +3.0 | **+8.1** | +1.5 | −6.3 | +2.1 | +6.3 |
| market (EW) | +8.8 | +24.6 | −3.8 | +34.8 | +26.0 | +24.1 | −19.4 | +23.9 | +16.4 | +15.7 | +10.2 |

**Proposal-specific bars:**

| bar | result | passed |
|---|---|---|
| alpha > 0 at 2× costs, P ≥ 0.90 | T1 +0.1% (P well below 0.90), TRAIN −4.1%, T2 −2.7% | no |
| beats the shuffled books | percentile 93 | no |
| most of the profit on stress days | T1 +2.5 vs −0.5, T2 +1.2 vs −1.7 (%/yr); TRAIN loses in both | T1/T2 only |
| positive in 2020 and 2022 | 2020 −5.2%, 2022 +8.1% | no |

**Verdict before T4: no edge net of costs.** It is a market-neutral book that holds up in every
crash (G6, G7). Its alpha is not distinguishable from zero, depends on a few names, and dies
under late fills.

## §11 T4, read once (2026-10-22 23:47, names 1001-1500, 2022-26; final candidate I4 vs U0; date corrected in review, §13.5)

`lab.py run peer_reversal I4 '{"formation": 21, "max_hold": 21, "gross_calm": 0.0}' T4 --costs2`
and `lab.py run peer_reversal U0 '{}' T4 --costs2`; `lab.py report peer_reversal U0 I4 --tests T4`.

| T4 | alpha (P) | beta | vol | maxDD | orders/yr | late | 2× costs | top5-out |
|---|---|---|---|---|---|---|---|---|
| **I4** | **+1.22% (0.61)** | 0.07 | 10.0% | −17.0% | 634 | −5.9% | −0.5% | −0.8% |
| U0 | −2.05% (0.29) | 0.08 | 8.9% | −15.4% | 3,103 | −25.5% | −8.1% | −5.3% |

- **Δ I4 − U0 = +3.3 pts (P 0.78), 7 of 10 periods better.**
- I4's per-period alphas: −0.137, +0.124, +0.164, +0.160, −0.123, +0.010, +0.204, −0.136,
  +0.119, 0.000.
- In the last period (2026-07 → 10) **I4 placed no order**: that market's volatility never rose
  above its median. *[Round 2, R2.5: half right. Nothing new opened, but the empty book was
  the 2026-06-19 holiday row, which had flattened June's positions; under v2 the period
  trades 68 orders and loses.]*
- T4 replicates the pattern: I4 has a small, insignificant positive alpha and beats U0 by roughly
  its cost saving; both fail on late fills.

## §12 Verdict: **no edge** (net of costs, in this universe and window)

*[Round 2 (R2.8): the verdict stands on lab v2, with U0 as the final; every number below is
lab v1's.]*

- **The effect is ~~real~~ suggestive (§13.1) and too small to trade.**
  - U0's selection beats 100 of 100 shuffled-name books on T1 (+4.8 pts/yr above random).
  - That is less than its own ~6%/yr in costs. Its gross trade edge (+2.4 bp) is a tenth of the
    24 bp round trip at $1,000 orders.
- **The proposal's two distinctive ideas did not survive.**
  - Correlation-cluster peers did no better than 12 sector codes (C1) or than no peers at all
    (premise IC 0.008 vs 0.013).
  - The binary stress switch added nothing at about the same mean exposure (C2; C2 ran 5-8%
    more realized gross, §13.4).
- **What did help is cutting turnover:** a one-month formation plus stress-only entries (I4),
  about 600 orders a year instead of 3,100.
- **The final I4 is a market-neutral book that holds up in every crash**: Q4 2018 +2.3%, COVID
  −1.8% vs −37%, 2022 +4.6%, Feb-Apr 2025 +1.6%, down-month mean +0.10%. But:
  - its alpha is −2.5% (TRAIN), +1.7% (T1, P 0.71), −1.0% (T2) and +1.2% (T4, P 0.61);
  - it fails G1, G2, G3, G4 (percentile 93), G5 (late −5.0%) and G8;
  - its T1 gain disappears when 5 names a period are dropped (noise copies −2.4 to −4.6 pts).
- **Lab gates, final candidate:**

  | gate | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 |
  |---|---|---|---|---|---|---|---|---|
  | I4 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |

  That is neither "works" nor "works to some degree".
- **Caveats.**
  - Survivor-only DB: the long side (buying peer-relative losers) is flattered, the short side
    understated.
  - At a $30k account a 40-slot book's orders are $367-735, not the priced $1,000: U0's real
    commission is 14-27 bp a leg.
  - The 4th iteration was adopted by a 0.045-pt margin over its bar, after four reads of T1.

---

## §13 Review fixes (fix agent, started 2026-10-23 00:20)

An adversarial review (reproduced I4 and U0 bit for bit on T1, look-ahead test passed) found one
major and six minor issues, all in the reading of the numbers, none in `strategy.py`. Each is
answered below: issue → fix → new numbers. The scripts behind every new number are
`strategies/peer_reversal/checks.py` (subcommand named next to each number); their logs are in
`var/proposals_lab/results/peer_reversal/fix/`. No earlier label was changed or deleted.

### §13.0 Pre-registration of the one new lab run: I4-h90, an execution diagnostic (variant 9; written 2026-10-23 00:24:35, before the run)

- **Change.** `gross_stress: 0.90` on top of I4 (label `I4-h90`), OPEN tests, late fills and
  2× costs.
- **Why.** Review issue 6: shorts lock their proceeds, so at a 0.98 target gross the drift of
  held positions leaves too little free cash for some new shorts. `checks.py fills I4 T1`:
  33 of 1,549 T1 openings (2.1%) filled below 90% of target, all of them shorts, 16 not at all;
  realized gross > 1.0 on up to 21% of a period's bars (max 1.114). Eight points of headroom
  should absorb most of that drift.
- **Expected.** Under-filled openings fall well below 1%; every alpha scales with gross
  (× 0.918: T1 ≈ +1.6%, TRAIN ≈ −2.3%, T2 ≈ −0.9%), so paired Δ vs I4 within ±1 pt.
- **Decision rule.** A diagnostic, not a candidate: I4 stays the final whatever I4-h90 shows
  (T4 has been read for I4, and choosing between two execution scales on T1 would be selection
  on noise). If its T1 alpha differs from 0.918 × I4's by more than 2 × I4's T1 noise sd
  (2.3 pts), the verdict is reported as sensitive to the execution model.

### §13.0 result: I4-h90 (`lab.py run peer_reversal I4-h90 '{…, "gross_stress": 0.9}' E0,TB,T3,T1,T2 --costs2`)

| | alpha (P) | Δ vs I4 (P) | 0.918 × I4 | beta | late | 2× costs | under-filled T1 openings |
|---|---|---|---|---|---|---|---|
| TRAIN | −2.48% (0.23) | −0.02 (0.45) | −2.26% | | −8.6% | −4.0% | |
| T1 | +0.49% (0.56) | **−1.24 (0.05)** | +1.59% | 0.046 (I4 0.070) | −5.8% | −1.0% | **1 of 1,549** (I4: 33) |
| T2 | −0.58% (0.45) | +0.40 (0.82) | −0.90% | 0.063 (I4 0.080) | −7.3% | −2.2% | |

- The headroom removes the under-fills (`checks.py fills I4-h90 T1`: 1 opening of 1,549 below
  90%; realized gross above 1.0 on at most 5% of a period's bars).
- The beta falls on every test (T1 0.070 → 0.046): the missing shorts had left I4 slightly net
  long.
- T1 lands 1.1 pts below the scaled expectation, inside the pre-registered 2.3-pt band, so the
  verdict is **not** sensitive to the under-fills. But I4's T1 alpha without them is +0.5%
  (P 0.56): one more sign that I4's T1 number is fragile.
- I4 stays the final candidate, as pre-registered.

### §13.1 Issue 1 (major): "the reversal is real" was overstated → now "suggestive"

**Issue.** The claim rested on one percentile (U0 beat 100 of 100 shuffled-name books on T1) on
the exact name set. The TRAIN premise never reached t 2 (IC +0.008, NW t 1.84; spread t 1.45),
and the per-trade gross edge is +2.4 bp with t 0.22 over entry dates.

**Fix.** Each U0 noise copy (5 names dropped a period) was run against **its own** shuffled
band: the same luck test, re-run on the copy's own panels (`checks.py noise-luck U0`, 3 × 1,000
books). The replay reproduces each saved copy's book to the last bit. The reviewer's
approximation placed the copies in the exact set's band instead.

| T1, U0 | real alpha | band median | band p95 | percentile (own band) | in the exact set's band (review) |
|---|---|---|---|---|---|
| exact name set | −1.11% | −5.89% | −2.61% | **100** | 100 |
| noise copy 0 | −4.40% | −6.70% | −2.74% | **78** | 75 |
| noise copy 1 | −3.00% | −6.37% | −3.59% | **98** | 92 |
| noise copy 2 | −1.60% | −5.69% | −3.03% | **100** | 100 |

**New reading.**
- U0's T1 selection beats the median of its own random band on all four name sets, by +2.3 to
  +4.8 pts/yr. It clears the 95th percentile on three of the four.
- On TRAIN nothing is significant: IC t 1.84, spread t 1.45, per-trade edge t 0.22. Longs earn
  +11.6 bp a trade (t 0.15 over entry dates) and shorts −6.5 bp (t −0.27).
- So the effect is **suggestive**: consistent on T1 selection, unproven on TRAIN, and in every
  case smaller than the ~6%/yr of costs. "Genuinely informative" is withdrawn.
- For I4, the same approximation the review used puts its three copies at percentiles 34, 22
  and 61 of its band (exact set 93). There is no selection evidence for the final.

### §13.2 Issue 2: I4's adoption does not survive the name noise → I4 = "a lower-turnover U0"

`checks.py noise U0 I4`. The U0 and I4 copies drop the same names (same seeds, same columns), so
the adoption Δ can be re-read on each copy:

| I4 − U0 | TRAIN (P) | T1 (P) | T2 (P) | passes the 2.79-pt T1 bar? |
|---|---|---|---|---|
| exact name set | +4.91 (0.87) | +2.84 (0.77) | +5.71 (0.86) | yes, by 0.045 pts |
| copy 0 | +6.10 (0.92) | +2.27 (0.74) | +4.98 (0.84) | no |
| copy 1 | +6.76 (0.95) | +0.14 (0.51) | +5.11 (0.85) | no |
| copy 2 | +7.55 (0.96) | +0.96 (0.59) | +7.50 (0.94) | no |
| T4 (read once, §11) | | | | Δ +3.27 (0.78), 7 of 10 periods better |

**The winner's curse is visible.** I4 was chosen on its T1 Δ:
- On T1, I4's alpha sits ABOVE all three of its copies, by 2.4-4.6 pts.
- On E0, TB, T3 and T2 it sits BELOW all three, by 0.2-3.3 pts.
- U0's T1 copies also all sit below U0, by 0.5-3.3 pts.

**The bar understates this.** The lab's noise sd measures the copies around their own mean, not
around the label. For U0 on T1 that sd is 1.40 pts, while the RMS distance from the label is
2.21 pts (I4: sd 1.13, RMS 3.72). A 2 × RMS bar (4.4 pts) would have rejected I4. Under it U0
would be the final, with the same verdict: U0 fails G1, G2, G3, G5, G6 and G8
(`lab.py verdict peer_reversal U0`; G4 100, G7 0.15). Both have been read once on T4, as the
protocol prescribes for the final and U0, so no T4 read is added.

**Fix.** I4 stays the declared final, but it is presented for what the evidence supports: a
**lower-turnover version of U0**. Its gain over U0 is robust on TRAIN and T2 (+5.0 to +7.6 in
every copy) and on T4 (+3.3), about the fee it saves: the 2×-cost drag falls from 6.0 to 1.6
pts/yr on T1. Its T1 gain (+0.1 to +2.8) is not robust. Its own alpha is about zero.

### §13.3 Issue 3: insignificant point estimates stated as findings → t attached

**Reversal curve, U0 on TRAIN** (`checks.py curve '{}'`). The t is Newey-West over entry dates,
with k lags at day k; bp are per trade.

| entries | day 5 | day 10 | day 21 | held trade |
|---|---|---|---|---|
| all (8,422) | −1.0 (t −0.00) | +4.2 (t −0.22) | +28.2 (t 1.03) | +2.4 (t 0.22) |
| longs | +3.6 (t −0.11) | +17.1 (t −0.17) | +59.2 (t 1.10) | +11.6 (t 0.15) |
| shorts | −5.5 (t −0.07) | −8.4 (t −0.58) | −2.1 (t −0.67) | −6.5 (t −0.27) |
| stress entries | +3.7 (t −0.04) | −10.2 (t −1.37) | +16.9 (t −0.13) | +4.2 (t −0.03) |
| calm entries | −4.3 (t +0.05) | +14.7 (t 1.36) | +36.3 (t 1.39) | +1.1 (t 0.52) |

- Signs can differ between a per-trade mean and its t, because the t reads the per-date means.
  Per date, stress entries average −3.2 bp at day 21.
- Stress − calm per entry date: −48.7 bp at day 10 (t −1.86) and −38.9 bp at day 21 (t −1.08).
- **Calm entries reverted more than stress entries.** That contradicts the stress-only rationale
  of I2 and I4, which the log never noted. "The reversal arrives on days 6-21" and the
  +28 / +59 bp "don't chase" numbers are insignificant point estimates.
- **I4 on TRAIN** (new, `checks.py curve '<I4>'`): 1,513 trades at +26.5 bp a held trade
  (t 1.20) against the 24 bp round trip. Longs +56.4 bp (t 1.20), shorts −3.0 bp (t −0.31).

**P&L by the stress state at the previous close** (`checks.py stress`). Book − beta × market,
in %/yr of all days (t), and the t of the stress − calm difference per day:

| | TRAIN stress / calm (t diff) | T1 stress / calm (t diff) | T2 stress / calm (t diff) |
|---|---|---|---|
| U0 | −3.5 (−0.85) / −3.7 (−3.05) (−0.19) | +1.1 (0.37) / −2.0 (−0.94) (+0.79) | −2.1 (−0.51) / −4.4 (−1.90) (+0.40) |
| C2 | −2.6 (−0.88) / −4.8 (−2.99) (+0.27) | +1.3 (0.56) / −1.5 (−0.61) (+0.82) | −1.3 (−0.40) / −4.5 (−1.59) (+0.68) |
| I1 | −0.2 (−0.05) / −1.9 (−0.98) (+0.33) | +0.0 (0.02) / −2.5 (−0.86) (+0.55) | −0.2 (−0.05) / −2.0 (−0.71) (+0.37) |
| I2 | −4.3 (−1.07) / −1.1 (−1.52) (−0.86) | +0.2 (0.05) / −0.8 (−0.48) (+0.23) | −3.6 (−0.87) / −2.5 (−1.38) (−0.29) |
| I4 | −0.7 (−0.20) / −1.4 (−0.82) (+0.09) | **+2.5 (0.77) / −0.5 (−0.21)** (+0.77) | +1.2 (0.31) / −1.7 (−0.72) (+0.61) |

- The only |t| ≥ 2 are the calm-day LEVELS of U0 and C2 on TRAIN, about −3. Calm days are 58%
  of TRAIN's days and every day carries its share of the ~6%/yr of costs, so these levels are
  mostly costs.
- No stress − calm difference reaches |t| 0.9 in any variant or sample.
- "The money IS made on stress days" (I4, T1 / T2) is a point estimate (t 0.77 and 0.31).

**Decomposing I4** (`checks.py pairs`):
- The stress-only half, I4 − I1: TRAIN −0.07 (P 0.51), T1 **+4.61 (P 0.97)**, T2 +1.62 (0.79).
- The monthly half, I4 − I2: TRAIN +3.15 (0.78), T1 +2.50 (0.73), T2 +5.33 (0.85).
- The stress-only half carries I4's T1 gain (+4.6, P 0.97), adds +1.6 on T2 and nothing on
  TRAIN. T1 is the sample whose Δ the adoption bar read.

**Smaller items.**
- §1's "4× stronger on stress days": the spreads have t 1.19 (stress) and 0.93 (calm).
- §7's volume terciles have t −0.74 / +1.47 / −0.10: noise either way. The volume cap was
  skipped for the budget, not on evidence. Its premise is untested.

### §13.4 Issue 4: numerical misstatements → corrected

1. **C2's exposure.** C2 was "the same mean exposure" as U0 in the pre-registration's
   assumption only. Realized mean gross (`checks.py exposure`):
   - C2: 0.739 / 0.733 / 0.738 (TRAIN / T1 / T2);
   - U0: 0.684 / 0.686 / 0.705.

   That is 5-8% more for C2. Stress was on 42-47% of days, not half, so the switch averaged
   about 0.70, not 0.735. The extra ~0.05 gross costs about 0.4 pts/yr, small against the C2 Δs.
2. **I4's late-fill drag** is −6.5 (TRAIN), −6.7 (T1), −7.2 (T2) and −7.1 (T4) pts. It is not
   "~5-6%/yr".
3. **The G5 yardstick.** A book with U0's turnover needs about 6.0 + 23.1 ≈ **29%/yr** of
   gross-of-cost alpha to pass the late-fill gate. The earlier "> 25%/yr" was too low.
4. **The Jul-Sep 2026 factor crash.** The +5.9% (market −1.9%) is T1 only. On T2, I4 lost 0.4%
   (market −3.6%). On T4 it held nothing (0.0%, market −4.2%).
5. **The tuner surface** (`checks.py tuner`).
   - TPE put 8 / 3 / 26 / 83 of the 120 trials in the formation bins 3-8 / 9-12 / 13-16 /
     17-21, so the bin means are confounded by its adaptive sampling.
   - The 20 random start-up trials point the same way. Spearman(formation, robust score) is
     +0.58. Their bins average −6.7% (8 trials), −5.3% (3), −6.3% (1) and −2.8% (8).
   - So the rise is not a sampling artifact, but it is in-sample and mostly the cost saving of
     slower ranks. The clean A/B (I1 vs U0) did not carry it to T1 (Δ −1.8).
   - The top 10: q_in 0.05 in 7 of 10, slots 10 in 9 of 10, max_hold 5-17, formation 16-18.

### §13.5 Issue 5: invented timestamps → the real ones

The headings of §5-§11 gave later times than the real ones, including dates of 2026-10-23. The
real times come from the run logs' creation stamps (`stat`), which match the builder's transcript
(all on 2026-10-22):

| step | time |
|---|---|
| I1 pre-registered and launched | 23:33:32 |
| I2 | 23:36:07 |
| I3 | 23:38:25 |
| I4 | 23:39:46 |
| final declared (launch of I4's luck run) | 23:41:49 |
| luck, noise and verdict | 23:41-23:46 |
| T4 read | 23:47:00-23:47:49 |
| README and config | 23:49 |

The order was right: each pre-registration was appended in the same command that launched its
run, and the final was declared before luck, noise, verdict, bear and T4. Only the written times
were wrong. They are corrected in the headings above, and the config's "VERDICT 2026-10-23" now
reads 2026-10-22. §0's stamp, 22:50, matches the log file's creation time.

### §13.6 Issue 6: short under-fills; the book is approximately dollar-neutral

**Mechanism** (engine-level, `src/` untouched). Shorts lock their proceeds, and
`call_seq="auto"` runs a bar's sells before its buys. So a bar's NEW shorts (sells) execute
before its short COVERS (buys), which are the orders that would free the locked cash. Once
positions drift, a 0.98-gross book lacks the free cash for some new shorts. This was checked in
vectorbt's source: `portfolio/nb.py` `sell_nb` caps a locked short sale at the free cash, and
`approx_order_value_nb` / `sort_call_seq_out_nb` order a bar's calls by approximate order value,
negative (sells) first.

**I4 on T1** (`checks.py fills I4 T1`):
- 33 of 1,549 openings (2.1%) filled below 90% of target, all of them shorts; 16 were not
  filled at all.
- Realized gross was above 1.0 on 0-21% of a period's bars, with a maximum of 1.114.
- The long and short slot counts differ on 0-68% of a period's bars.
- The net target is at most 0.147 of the book (mean ≤ 0.038). The realized net is −0.02 to −0.03
  (TRAIN / T1 / T2 / T4).

**Fix.**
- The book is described as approximately dollar-neutral, in the log, the README and the
  docstring.
- The headroom diagnostic (§13.0) shows the under-fill does not drive the verdict.
- Any combined book that reuses this sleeve should run it at gross ≤ 0.90.
- Engine proposal (not implemented): within a bar, run the orders that free cash (long sales
  and short covers) before the orders that use it (long buys and short openings). Or size a
  short opening against the free cash after that bar's covers. The same applies to every
  long/short lab book at 0.98 gross.

### §13.7 Issue 7: present-day sector codes

**Issue.** The lab's `sector` is one constant code per name, today's classification, applied
back to 2015. That gives C1 and the premise's sector rows a mild look-ahead.

**Why the conclusion holds.** The look-ahead can only flatter sector peers. The conclusion
"clusters are no better" also rests on the no-peers comparison, which has no look-ahead: in the
premise, plain market-relative reversal scored IC +0.013 (t 1.75) against the clusters' +0.008
(t 1.84), both below t 2. So the correct wording is "no evidence that the clusters help", not
"clusters are worse". The final I4 uses the clusters and has no look-ahead. Reported as a lab
issue.

### §13.8 Lab hygiene: the U0 luck cache

At 23:37 the builder added the `scale` / `scale_window` knobs. Their default, "none", leaves
every book bit-identical, but the knobs changed the params hash in `lab.luck`'s cache key, which
orphaned `luck-U0-2b8adb2b8f.pkl`. The reviewer's recomputation with the current code is
byte-identical to it. It is restored as `luck-U0-e8e22f9ca4.pkl`, so
`lab.py verdict peer_reversal U0` reads it instead of re-running 1,000 books. The old file is
kept.

### §13.9 Variants

**9 of the hard cap of 10:**
- U0, C1, C2, T, I1, I2, I3, I4;
- I4-h90, the execution diagnostic, not a candidate.

The noise-luck, curve, stress, fills and exposure audits re-read existing books; they are not
variants. No TL read. No second T4 read: the final is unchanged.

The only code change is a docstring in `strategy.py` (the sizing paragraph, §13.6). A fresh run
of I4 on T1 after the edit (`fix/I4-fix-repro.pkl`) reproduces I4 bit for bit: the same orders
in every period, and a maximum daily difference of 0.0 across the returns, benchmark, late-fill
and 2×-cost books. Tests (14 + 1) and ruff pass.

### §13.10 Re-derived verdict: **no edge** (unchanged), with the evidence restated

- **Selection.** Peer-relative weekly reversal is suggestive, not established.
  - On T1, U0's selection beats random names on all four name sets, by +2.3 to +4.8 pts/yr,
    significantly on three.
  - On TRAIN no statistic reaches t 2.
  - The selection value is smaller than the ~6%/yr of costs.
- **The final, I4,** is a lower-turnover U0, about 600 orders/yr against 3,100.
  - Its gain over U0 is robust where it is a cost saving: TRAIN, T2, T4 and every noise copy.
  - Its gain is not robust on T1.
  - Its own alpha is about zero: TRAIN −2.5%, T1 +1.7% (+0.5% with headroom), T2 −1.0%,
    T4 +1.2%; none significant.
  - It fails G1, G2, G3, G4, G5 and G8, and passes G6 and G7.
- **The proposal's distinctive parts are unsupported.**
  - Correlation clusters showed no advantage over sector codes, which carry a mild look-ahead,
    or over no peers.
  - The stress switch showed no advantage over constant exposure.
  - The stress-only timing showed no TRAIN advantage, with calm entries reverting more.
- **What survives** is a near-zero-beta book (0.05-0.08) that beat the market in all four bear
  episodes: Q4 2018 +2.3%, COVID −1.8% (market −37.0%), 2022 +4.6% and Feb-Apr 2025 +1.6%. It
  is a candidate hedge-like sleeve only if its crash-time correlation with other books is
  negative, which has not been measured.

---

## Round 2: lab v2 re-evaluation

### R2.0 Pre-registration (written 2026-10-23 08:04:26, before any v2 number of this strategy)

**Why.** Every round-1 number above came from lab v1. The six round-1 reviews found five
lab-level defects (`strategies/proposals_lab/README.md`, "Lab v2"). Lab v2 fixes them, writes to
`var/proposals_lab/results2/`, and the panels were rebuilt (NYSE sessions only, ETFs out, the
current DB). What each defect did to THIS book:

1. **Holiday rows** (2026-06-19, 07-03, 09-07: no US name priced). The move is NaN on the two
   bars that read such a row, so every held name became "ineligible" and exited, and the book
   rebuilt the next bar: about 6 extra whole-book round trips in each 2026 half-year of T1, T2
   and T4. *[Corrected after the runs (R2.2, R2.5): two rebuilds per row (the row and the bar
   after it, then the bar whose formation window starts on it), so 2 in 2026-H1 and 4 in
   2026-H2. The rows also blocked the stress switch for 21 bars each, which this list
   missed.]*
2. **Day 0** (the opening orders' costs, the first open-to-close move, the up-front borrow) was
   dropped.
3. **Starved shorts.** `call_seq="auto"` ran new shorts before the covers that free locked cash.
   That is the under-fill of §13.6: 33 of 1,549 I4 T1 openings, all shorts.
4. **Borrow on hair re-trims.** This book never re-sizes a held weight (§13.6), so no effect is
   expected.
5. **Six sector ETFs** in the top 500 since 2023-10 (T1 from 2024-H1). *[Corrected after the
   runs (R2.2): from 2024-H2. The 2024-H1 universes are identical in v1 and v2, probably
   because the ETFs, stored as equities only since 2023-10, lacked the ranking's 100 sessions
   in 2023-H2.]*

**What is re-measured.** No config changes. Every round-1 variant runs with its round-1
overrides under a new label ending in `v2`, on OPEN (E0, TB, T3, T1, T2) with late fills and 2×
costs:

| label | overrides | round-1 role |
|---|---|---|
| U0v2 | `{}` | the base |
| C1v2 | `peers: sector` | control |
| C2v2 | constant gross 0.735 | control |
| Tv2 | the round-1 tuned values (n_clusters 50, formation 18, q_in 0.05, q_out 0.5, max_hold 6, slots 10), not re-tuned | optimization |
| I1v2 | formation 21, max_hold 21 | iteration |
| I2v2 | gross_calm 0 | iteration |
| I3v2 | scale vol | iteration |
| I4v2 | I1 + I2 | adopted; the round-1 final |
| I4-h90v2 | I4 + gross_stress 0.9 | execution diagnostic |

Then, on v2:
- `lab.py noise` and `lab.py luck` for U0v2 and the v2 final;
- `lab.py verdict` (market-neutral, so G7 applies) and `lab.py bear`;
- the round-1 review checks (`checks.py`): the adoption Δ on every noise copy, U0's noise copies
  against their own shuffled bands, P&L by stress state, realized exposure, under-filled
  openings.

**Diagnostic, not a variant: what each fix did.** `checks.py decompose` re-runs U0 and I4 on the
v2 panels through the ENGINE's runner (`call_seq="auto"`, the engine's borrow) with day 0
dropped: v1's accounting on v2's panels. v1 → v2 then splits into three parts:
- **panels** (holiday rows, ETFs, DB) = v1 saved − v1 accounting on the v2 panels;
- **runner** (call order and borrow) = the v2 runner with day 0 dropped − the engine's runner
  with day 0 dropped;
- **day 0** = the remainder.

TRAIN has no holiday rows and no ETFs, so there the v1-on-v2-panels books should reproduce
round 1 bit for bit unless the DB's prices changed.

**Decision rules (round 1's, unchanged).**
- The adoption bar is 2 × U0v2's T1 noise sd (round 1: 2.79 pts).
- The iterations are replayed in round-1 order (I1v2, I2v2, I3v2, I4v2). Each is judged
  against the base in force at that point under v2: U0v2, unless an earlier one is adopted
  under v2.
- An iteration is adopted only if its paired Δ is ≥ 0 on TRAIN (stitched), T1 and T2, and its
  T1 Δ is above the bar.
- Tv2 is adopted only if it beats U0v2 on both T1 and T2 (paired). C1v2 and C2v2 are controls.
- The v2 final is the last adopted config, or U0v2 if none is adopted. If it is not I4, it is
  declared here before its T4 read.
- T4 is read ONCE for the v2 final and for U0v2. Both were read in round 1, so this is a
  disclosed re-measurement on corrected accounting, and it selects nothing.
- Round 1 used 9 of the hard cap of 10 variants. At most ONE new iteration is possible, only if
  v2 opens an obvious direction, and it is pre-registered here before it runs.

**Expected (priors, not measurements).**
- **Day 0** charges one book rebuild a period: about −0.2 pt/yr of alpha at base costs, and
  about −1 pt/yr more under late fills, for U0 and I4 alike.
- **Holiday rows.** U0 gains in the 2026 half-years of T1 and T2, about +1 pt a 2026 period
  from about 6 fewer rebuilds. I4 gains less.
- **Call order.** The under-filled shorts disappear at 0.98 gross. In round 1, removing them
  (I4-h90) cost I4 1.24 pts on T1, so I4v2's T1 alpha is expected about 1 pt below I4's.
  Its adoption, which cleared the bar by 0.045 pts, is then expected to FAIL under v2, which
  would make U0v2 the final. The TRAIN and T2 Δ should stay positive (the cost saving).
- **Status:** expected unchanged, no edge.

### R2.1 Every round-1 variant on lab v2, and the replayed decisions (runs 08:04-08:21)

`lab.py run peer_reversal <label>v2 '<round-1 overrides>' E0,TB,T3,T1,T2 --costs2` for each
label, then `checks.py v1v2` and `checks.py pairs` (logs in
`var/proposals_lab/results2/peer_reversal/`). Alpha in %/yr, TRAIN = E0 + TB + T3 stitched:

| label | TRAIN v1 → v2 | T1 v1 → v2 | T2 v1 → v2 | T1 late v2 | T1 2× v2 | T1 orders/yr v2 |
|---|---|---|---|---|---|---|
| U0 | −7.37 → **−7.68** | −1.11 → **−3.45** | −6.69 → **−6.54** | −27.3 | −9.8 | 3,080 |
| C1 sector peers | −5.30 → −4.98 | −2.02 → −0.09 | −0.76 → +0.11 | −25.2 | −6.8 | 3,153 |
| C2 constant gross | −7.47 → −7.61 | −0.40 → −3.59 | −5.96 → −6.55 | −27.8 | −10.0 | 3,080 |
| T tuned values | +3.16 → +3.30 | −2.11 → −6.13 | −10.91 → −12.31 | −16.5 | −8.7 | 640 |
| I1 | −2.39 → −2.42 | −2.88 → −4.94 | −2.60 → −2.75 | −12.9 | −6.9 | 948 |
| I2 | −5.61 → −5.75 | −0.77 → −1.41 | −6.31 → −3.92 | −19.9 | −6.2 | 1,657 |
| I3 | −6.38 → −5.89 | −3.46 → −5.68 | −6.03 → −5.15 | −28.9 | −11.8 | 3,107 |
| I4 | −2.46 → **−2.59** | +1.73 → **−3.10** | −0.98 → **+0.08** | −10.3 | −4.9 | 619 |
| I4-h90 | −2.48 → −2.36 | +0.49 → −2.79 | −0.58 → +0.11 | −9.5 | −4.4 | 619 |

**Noise** (`lab.py noise peer_reversal U0v2`). Alpha deltas of the three copies, pts, and sd:

| test | deltas | sd |
|---|---|---|
| E0 | −1.5 / −1.9 / −1.5 | 0.24 |
| TB | +1.3 / +1.7 / −2.7 | 2.40 |
| T3 | +2.0 / −0.2 / +1.6 | 1.13 |
| **T1** | +1.1 / −0.1 / +2.5 | **1.30** |
| T2 | +1.2 / −0.1 / +1.4 | 0.81 |

**The v2 adoption bar is a T1 Δ > 2.60 pts** (round 1: 2.79).

**Luck** (`lab.py luck peer_reversal U0v2`): real −3.45% vs median −6.49%, p95 −3.90%,
**percentile 99** (round 1: 100). The selection still beats its same-turnover random books, by
3.0 pts/yr over the median (round 1: 4.8).

**The decisions, replayed in round-1 order.** Paired Δ against U0v2, pts (P):

| variant | TRAIN | T1 | T2 | v2 decision | round 1 |
|---|---|---|---|---|---|
| I1v2 | +5.26 (0.87) | −1.48 (0.37) | +3.79 (0.78) | not adopted (T1 < 0) | not adopted |
| I2v2 | +1.93 (0.98) | +2.05 (0.90) | +2.62 (0.93) | not adopted (T1 below 2.60) | not adopted |
| I3v2 | +1.79 (0.79) | −2.22 (0.24) | +1.39 (0.71) | not adopted (T1 < 0) | not adopted |
| **I4v2** | +5.09 (0.87) | **+0.36 (0.54)** | +6.62 (0.91) | **not adopted (T1 below 2.60)** | adopted, by 0.045 pts |
| Tv2 | +10.98 (0.99) | −2.68 (0.29) | −5.77 (0.12) | not adopted (worse on T1 and T2) | not adopted |
| C1v2 (control) | +2.70 (0.97) | +3.36 (0.84) | +6.65 (0.99) | control | control |
| C2v2 (control) | +0.07 (0.48) | −0.14 (0.44) | −0.02 (0.49) | control | control |

- **I4 is no longer adopted.** Its T1 Δ fell from +2.84 to +0.36. TRAIN and T2 stay positive
  (the cost saving), as predicted in R2.0.
- **The cause is not the one predicted.** It is the panels, not the call order
  (`checks.py decompose`, R2.2). On the v2 panels, I4's T1 alpha is −3.40% already under v1's
  runner and day-0 accounting. In 2026-H1 the round-1 book never held MRVL; the v2 book shorts
  it for 65 bars and loses 5.4 pts on that name alone.
- Two things changed in the T1 panels. From 2024-H2, six sector ETFs were replaced by the next
  six stocks. In 2026, the holiday rows were removed. Together they moved I4's T1 alpha by
  −5.1 pts and U0's by −2.5 (R2.2). Most of it is the six-name swap, which re-forms the
  clusters. By 06-11, a week before its one holiday row, I4's 2026-H1 book was already 11.7
  pts behind round 1's (14.0 at the period's end; `checks.py period I4 T1 2026-01-01`). That
  is the name noise the round-1 review found, now in the real record.
- **C1 (sector peers) now beats the clusters on all three samples**, by +2.7 / +3.4 / +6.7 pts.
  It is a control by pre-registration, and the sector codes are today's classification, a mild
  look-ahead that can only flatter them.

### R2.1a The v2 final (declared 2026-10-23 08:29:12, before its verdict, bear and T4 read)

**The v2 final is U0v2**, the pre-registered untuned default. No iteration clears the v2 bar,
so the base stays the final, as the protocol prescribes.

- **T4.** U0v2 is read once on T4. It is the v2 final, and it was read in round 1 as the
  comparison. I4v2 is also re-measured once on T4, because round 1 read I4 there as its final
  and the v1 → v2 table needs that row. I4 is no longer a candidate, and neither number can
  change a decision.
- **No new iteration.** Round 1 used 9 of the 10 variants, and v2 opens no direction that could
  change the status:
  - every variant fails G5 by at least 9 pts on T1 (late-fill alpha ≤ −9.5%);
  - every variant except the in-sample tuned T is negative on TRAIN, which fails G1;
  - the one direction v2 points to, sector peers, carries the present-day sector look-ahead. It
    was pre-registered as a control, and promoting it now would be a choice made on validation
    results already seen.

  The last variant is left unspent.

### R2.2 What each fix did (diagnostic; `checks.py decompose U0 | I4 | C1`, `checks.py borrow`)

**TRAIN reproduces round 1 bit for bit** under v1's accounting, for all three configs (largest
daily gap 0.0). So the TRAIN panels did not change, and round 1's TRAIN-only numbers stand as
written: the premise (§1), the anatomy (§2) and the reversal-curve t's (§13.3). The borrow fix
(lab fix 4) never applies to this book: 0 short re-sizes in U0v2 and I4v2 on every test, and
`lab.borrow_fees` equals the engine's `_borrow_fees` exactly. The "runner" column is therefore
the call order alone.

Stitched alpha, pts/yr, at base costs (late fills in brackets):

| config | sample | v1 | panels | runner | day 0 | v2 |
|---|---|---|---|---|---|---|
| U0 | TRAIN | −7.37 | 0.00 | −0.19 (−0.45) | −0.13 (−0.60) | −7.68 |
| U0 | T1 | −1.11 | −2.47 (−1.95) | +0.37 (−0.40) | −0.24 (−0.73) | −3.45 |
| U0 | T2 | −6.69 | −0.70 (−0.19) | +0.83 (+0.25) | +0.01 (−0.51) | −6.54 |
| I4 | TRAIN | −2.46 | 0.00 | −0.27 (−0.35) | +0.13 (−0.34) | −2.59 |
| I4 | T1 | +1.73 | −5.13 (−4.83) | +0.40 (+0.22) | −0.09 (−0.74) | −3.10 |
| I4 | T2 | −0.98 | +0.94 (+1.26) | +0.17 (+0.05) | −0.04 (−0.70) | +0.08 |
| C1 | TRAIN | −5.30 | 0.00 | +0.32 (−0.09) | −0.01 (−0.52) | −4.98 |
| C1 | T1 | −2.02 | +1.76 (+1.35) | +0.66 (−0.06) | −0.49 (−0.91) | −0.09 |
| C1 | T2 | −0.76 | +1.45 (+1.47) | −0.24 (−0.81) | −0.35 (−0.83) | +0.11 |

- **Day 0** costs 0.0-0.5 pts/yr at base costs and 0.3-0.9 under late fills, about as
  predicted (−0.2 and −1).
- **The call order** moves alpha by −0.3 to +0.8 pts and changes no decision. The prediction
  (about −1 pt for I4 on T1, from filling the starved shorts) was wrong in sign: +0.40.
- **The panels** carry the T1 and T2 changes. The T1 universes of 2022-H1 → 2024-H1 are
  identical in v1 and v2 (the benchmark reproduces exactly). From 2024-H2, XLE, XLF, XLI, XLK,
  XLV and XLY left the top 500, the next six names entered, and T2's band shifted by the same
  six. The 2026 periods also lost their holiday rows.
- Annualized per-period alpha change from the panels, base costs:

  | | 2024-H2 | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 |
  |---|---|---|---|---|---|
  | I4 | +0.006 | −0.079 | −0.010 | −0.278 | −0.097 |
  | U0 | −0.040 | −0.020 | −0.069 | −0.080 | −0.016 |

- **The adoption flipped on the panels alone** (`checks.py attribute U0 I4`). I4 − U0,
  paired, pts (P):

  | stage | TRAIN | T1 | T2 |
  |---|---|---|---|
  | v1, saved | +4.91 (0.87) | **+2.84 (0.77)** | +5.71 (0.86) |
  | v1's accounting on the v2 panels | +4.91 (0.87) | **+0.17 (0.52)** | +7.34 (0.94) |
  | lab v2 | +5.09 (0.87) | **+0.36 (0.54)** | +6.62 (0.91) |

  The accounting fixes (day 0 and call order) moved the T1 Δ by +0.19. Added to the round-1
  Δ, that gives about +3.0, above either bar: on the round-1 panels I4 would probably still
  have been adopted. This is an extrapolation, since the v1 panels were not kept.
- A six-name swap re-forms the peer groups and changes which names are extreme. In 2026-H1
  the v2 book shorts MRVL (held 65 bars from 03-27, its clock restarting while it stayed at
  the top of the ranking) and DELL (79 bars); the round-1 book never held MRVL. The swap
  moves sector peers too (C1 +1.8 on T1), but the other way.

### R2.3 The round-1 review checks, re-read on v2

**Issue 1: U0's selection, now the final's** (`checks.py noise-luck U0v2`). Each copy is read
in its own shuffled band:

| T1, U0v2 | real alpha | band median | band p95 | percentile | round 1 |
|---|---|---|---|---|---|
| exact name set | −3.45% | −6.49% | −3.90% | **99** | 100 |
| noise copy 0 | −2.34% | −6.15% | −2.04% | **93** | 78 |
| noise copy 1 | −3.56% | −6.70% | −3.66% | **96** | 98 |
| noise copy 2 | −0.96% | −6.58% | −3.49% | **99** | 100 |

- Under v2 the copies drop other names than in round 1 (the T1 columns changed from 2024-H2),
  so the copy-by-copy comparison is loose.
- On all four name sets the selection beats the median random book of the same turnover, by
  +3.0 to +5.6 pts/yr. It clears p95 on three of the four, as in round 1.
- The reading is unchanged: **suggestive, not established**.
  - TRAIN is unchanged: IC t 1.84, spread t 1.45, +2.4 bp a trade at t 0.22.
  - U0v2's costs are 6.0-6.4 pts/yr (2× − base: TRAIN 5.97, T1 6.32, T2 6.16, T4 6.39). That
    is twice the +3.0 pts by which it beats random.

**Issue 2: I4's adoption under name noise** (`checks.py noise U0v2 I4v2`). I4v2 − U0v2,
paired:

| | TRAIN (P) | T1 (P) | T2 (P) | passes the 2.60 bar? |
|---|---|---|---|---|
| exact name set | +5.09 (0.87) | +0.36 (0.54) | +6.62 (0.91) | no |
| copy 0 | +5.95 (0.91) | +0.34 (0.53) | +5.69 (0.91) | no |
| copy 1 | +6.30 (0.94) | +0.55 (0.56) | +4.15 (0.82) | no |
| copy 2 | +7.10 (0.94) | −2.27 (0.28) | +1.39 (0.61) | no |

On T4 (R2.5, read once) the Δ is +4.41 (P 0.85), with 7 of 10 periods better.

- The review's reading is now the decision. I4 is a lower-turnover U0: its TRAIN, T2 and T4
  gains are about the cost it saves (the 2×-cost drag falls from 6.3 to 1.8 pts/yr on T1). It
  is not a better selector, and on T1 it gains nothing.
- The round-1 winner's-curse pattern (every T1 copy below its label) does not repeat on the
  v2 panels:

  | T1 | copy − label, pts | sd | RMS |
  |---|---|---|---|
  | U0v2 | +1.1 / −0.1 / +2.5 | 1.30 | 1.58 |
  | I4v2 | +1.1 / +0.1 / −0.1 | 0.65 | 0.64 |

**Issue 3: insignificant point estimates.** TRAIN is unchanged (R2.2), so the curve and anatomy
t's of §13.3 stand. The stress split on v2 (`checks.py stress`) gives book − beta × market in
%/yr of all days (t), and the t of the stress − calm difference per day:

| | TRAIN stress / calm (t diff) | T1 stress / calm (t diff) | T2 stress / calm (t diff) |
|---|---|---|---|
| U0v2 | −3.9 (−0.97) / −3.7 (−3.03) (−0.30) | +0.8 (0.27) / −4.1 (−1.82) (+1.17) | −2.0 (−0.47) / −4.3 (−1.78) (+0.37) |
| C2v2 | −2.8 (−0.91) / −4.8 (−3.02) (+0.24) | +1.1 (0.48) / −4.5 (−1.76) (+1.53) | −1.4 (−0.44) / −5.0 (−1.67) (+0.70) |
| I1v2 | −0.3 (−0.08) / −1.8 (−0.94) (+0.29) | −2.1 (−0.65) / −2.5 (−0.66) (−0.01) | −0.7 (−0.19) / −1.6 (−0.58) (+0.17) |
| I2v2 | −4.5 (−1.14) / −1.0 (−1.42) (−0.94) | +0.2 (0.05) / −1.4 (−0.88) (+0.38) | −1.4 (−0.30) / −2.3 (−1.22) (+0.14) |
| I4v2 | −0.9 (−0.25) / −1.4 (−0.79) (+0.04) | −2.3 (−0.66) / −0.4 (−0.12) (−0.43) | +2.1 (0.55) / −1.5 (−0.57) (+0.74) |

- No stress − calm difference reaches |t| 2. The largest is +1.53 (C2v2 on T1); round 1's
  largest was below 0.9.
- I4's "money made on stress days" (round 1, T1: +2.5 vs −0.5) is gone on T1 (−2.3 vs −0.4).
  It shows on T2 only, at t 0.74.
- The only |t| ≥ 2 are again the calm-day levels of U0 and C2 on TRAIN, about −3: the costs
  paid every day.

**Issue 4: numbers, restated on v2.**
1. **C2's exposure** (`checks.py exposure`). C2v2 runs 0.739 / 0.737 / 0.740 gross against
   U0v2's 0.700 / 0.713 / 0.723 (TRAIN / T1 / T2): 2-6% more. Round 1 found 5-8%; U0 now fills
   its shorts. C2 − U0 is −0.14 to +0.07 pts: the switch adds nothing.
2. **I4v2's late-fill drag** (late − base) is −7.1 (TRAIN), −7.2 (T1), −7.7 (T2) and −7.9
   (T4) pts. Day 0 accounts for 0.3-0.7 of it.
3. **The G5 yardstick** at U0's turnover: 6.3 pts of costs plus 23.8 of late-fill drag, about
   30%/yr of gross alpha (round 1: 29).
4. **Jul-Sep 2026**, book (market), `checks.py holidays U0 I4`:

   | | T1 | T2 | T4 |
   |---|---|---|---|
   | I4v2 | +4.8% (−1.6%) | +3.1% (−3.9%) | −6.5% (−4.0%) |
   | U0v2 | +1.7% (−1.6%) | +0.5% (−3.9%) | +0.1% (−4.0%) |

   Round 1's I4 figures (T1 +5.9%, T2 −0.4%, T4 "held nothing") were partly the holiday rows
   (R2.5).
5. **The tuner surface** (§4, §13.4) comes from a TRAIN-only round-1 study and is not re-run.
   Under v2, TRAIN books move by only −0.3 to +0.3 pts (R2.2). Tv2, the round-1 best,
   reproduces its TRAIN (+3.30 vs +3.16) and again loses on T1 and T2.

**Issue 5: timestamps.** Every round-2 time comes from `date`, printed by the command that wrote
the entry and launched the next runs (`results2/peer_reversal/round2.log`):

| step | time |
|---|---|
| pre-registration | 08:04:26 |
| batch 1 (every variant, U0v2's noise and luck) | 08:04:26-08:21:11 |
| final declared | 08:29:12 |
| batch 2 (checks, verdicts, bear) | 08:29:12-08:50:59 |
| T4 | 08:51:48-08:53:01 |

**Issue 6: under-fills, and an approximately dollar-neutral book.** The lab v2 runner fixes the
under-fills by running covers before new shorts. `checks.py fills <label> T1`:

| label | openings | filled < 90% | realized gross max | gross > 1.0 on (worst period) |
|---|---|---|---|---|
| I4, round 1 (lab v1) | 1,549 | 33 (16 not at all) | 1.114 | 21% of bars |
| U0v2 | 7,519 | **0** | 1.074 | 25% |
| I4v2 | 1,584 | **0** | 1.266 | 35% |
| I4-h90v2 | 1,584 | **0** | 1.152 | 13% |

- Realized gross above 1 is now drift alone (shorts that rose), and it no longer blocks an
  opening.
- **Headroom no longer matters.** I4-h90v2 − I4v2 is +0.24 (P 0.81) on TRAIN, +0.30 (0.78) on
  T1 and +0.03 (0.54) on T2. Its T1 alpha (−2.79%) is within 0.06 pts of 0.918 × I4v2's
  (−2.85%). Round 1's advice to "run it at gross 0.90 in a combined book" is withdrawn for
  lab-v2 books. It stands only for the repo engine (`src/`), which still runs
  `call_seq="auto"`.
- The book is still only approximately dollar-neutral. On I4v2's T1 bars the slot counts
  differ on 0-68% of a period's bars and |net target| reaches 0.172. Realized net exposure is
  −0.015 to −0.021 (U0v2) and −0.026 to −0.041 (I4v2).

**Issue 7: present-day sector codes.** v2 does not fix this (a lab caveat), and it now matters
more:
- C1v2 beats the clusters on TRAIN by +2.70 (P 0.97), on T1 by +3.36 (0.84) and on T2 by
  +6.65 (0.99).
- C1's TRAIN reproduces round 1 bit for bit under v1's accounting, so the codes did not change.
- **Reading.** The correlation clusters did worse than present-day sector codes (which the
  look-ahead flatters). On the TRAIN premise they also did worse than no peers at all (IC
  +0.008 vs +0.013, neither at t 2). Round 1's "no evidence that the clusters help" becomes
  "the evidence points against them".

### R2.4 The v2 final, U0v2, through the gates (`lab.py verdict peer_reversal U0v2`, `lab.py bear peer_reversal U0v2 I4v2`)

| gate | U0, round 1 | **U0v2** | I4, round 1 | I4v2 |
|---|---|---|---|---|
| G1 TRAIN-era alpha > 0 | −7.37% ✗ | **−7.68% ✗** | −2.46% ✗ | −2.59% ✗ |
| G2 T1 alpha > 0, P ≥ 0.90 | −1.11% (0.36) ✗ | **−3.45% (0.16) ✗** | +1.73% (0.71) ✗ | −3.10% (0.26) ✗ |
| G3 T2 alpha > 0 | −6.69% ✗ | **−6.54% ✗** | −0.98% ✗ | +0.08% ✓ |
| G4 T1 beats ≥ 95% of shuffled books | 100 ✓ | **99 ✓** | 93 ✗ | 31 ✗ |
| G5 T1 late-fill / 2×-cost alpha > 0 | −24.2% / −7.1% ✗ | **−27.3% / −9.8% ✗** | −4.95% / +0.07% ✗ | −10.3% / −4.9% ✗ |
| G6 every bear episode beaten, down-month mean > 0 | −0.80% ✗ | **−0.78% ✗** | +0.10% ✓ | +0.19% ✓ |
| G7 \|T1 beta\| ≤ 0.3 | 0.145 ✓ | **0.131 ✓** | 0.070 ✓ | 0.056 ✓ |
| G8 T1 alpha without its top 5 names > 0 | −5.0% ✗ | **−7.41% ✗** | −1.50% ✗ | −5.28% ✗ |

U0v2 fails the same six gates as U0 in round 1 (G1, G2, G3, G5, G6, G8). It passes G4 and G7.
That is neither "works" nor "works to some degree".

Bear episodes, stitched E0 + TB + T3 + T1 (% over each episode):

| | Q4 2018 | COVID | 2022 bear | Feb-Apr 2025 | down months: book / market (share > 0) | CAGR | max DD |
|---|---|---|---|---|---|---|---|
| U0, round 1 | −0.4 | −12.6 | +6.9 | +2.9 | −0.80% / −3.97% (29%) | −3.0 | −34.3 |
| **U0v2** | **+0.1** | **−11.5** | **+7.3** | **+1.4** | **−0.78% / −3.98% (27%)** | **−4.6** | **−39.5** |
| I4, round 1 | +2.3 | −1.8 | +4.6 | +1.6 | +0.10% / −3.97% (41%) | +0.6 | −14.6 |
| I4v2 | +2.2 | −1.3 | +4.2 | +0.8 | +0.19% / −3.98% (46%) | −1.8 | −27.2 |
| market (EW), v2 | −18.2 | −37.0 | −24.5 | −19.5 | | | |
| SPY | −18.7 | −33.4 | −24.1 | −18.6 | | | |

Both books beat the market in all four episodes. U0's down-month mean stays negative, which is
why it fails G6.

Calendar years, %:

| | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U0v2 | −4.0 | −2.5 | −5.9 | −7.2 | −9.2 | −6.7 | +4.3 | −3.3 | −6.0 | −1.6 | −4.2 |
| I4v2 | −2.4 | −0.9 | −2.7 | +5.2 | −8.2 | +3.0 | +7.8 | +0.4 | −5.7 | −1.2 | −12.6 |
| market (EW) | +9.0 | +24.3 | −2.8 | +35.6 | +25.7 | +21.5 | −18.6 | +22.7 | +15.3 | +16.2 | +11.7 |

The equal-weight market's years differ from round 1's by up to 2.6 pts (2021), because each
half-year's day 0 now counts and, from 2024-H2, six names were swapped.

**Proposal-specific bars**, for U0v2:

| bar | result | passed |
|---|---|---|
| alpha > 0 at 2× costs, P ≥ 0.90 | T1 −9.8% | no |
| beats the shuffled books | percentile 99 | yes |
| most of the profit on stress days | there is no profit; T1 stress +0.8 vs calm −4.1 %/yr | no |
| positive in 2020 and 2022 | 2020 −9.2%, 2022 +4.3% | no |

### R2.5 T4, re-measured once (08:51:48-08:53:01)

`lab.py run peer_reversal U0v2 '{}' T4 --costs2`, and the same for I4v2 with its overrides;
`lab.py report peer_reversal U0v2 I4v2 --tests T4`.

| T4 | alpha (P) | beta | maxDD | orders/yr | late | 2× costs | top5-out |
|---|---|---|---|---|---|---|---|
| **U0v2, the v2 final** | **−2.91% (0.20)** | 0.064 | −16.3% | 3,100 | −27.0% | −9.3% | −5.7% |
| U0, round 1 | −2.05% (0.29) | 0.075 | −15.4% | 3,103 | −25.5% | −8.1% | −5.3% |
| I4v2 | +1.50% (0.64) | 0.054 | −14.3% | 651 | −6.4% | −0.4% | −1.3% |
| I4, round 1 | +1.22% (0.61) | 0.066 | −17.0% | 634 | −5.9% | −0.5% | −0.8% |

- I4v2 − U0v2 = +4.41 pts (P 0.85), with 7 of 10 periods better (round 1: +3.27, P 0.78,
  7 of 10). On T4, as on TRAIN and T2, the lower-turnover book beats U0 by about its cost
  saving. On T1 it does not.
- Per-period alphas:

  | | per half-year |
  |---|---|
  | U0v2 | −0.198, +0.162, −0.042, +0.027, −0.016, −0.051, +0.023, −0.024, +0.013, −0.023 |
  | I4v2 | −0.119, +0.088, +0.175, +0.158, −0.122, +0.053, +0.170, −0.006, +0.086, −0.175 |

- The read selects nothing. The final (U0v2) was fixed at 08:29:12, and I4 was no longer a
  candidate.
- **A round-1 statement, corrected.** §11 said that in T4's last period (2026-07 → 10) I4
  "placed no order: that market's volatility never rose above its median". Under v2 it placed
  68 orders and lost (period alpha −0.175).
  - The market was indeed never stressed then (0 of 76 bars), so nothing new opened.
  - But I4 entered the period holding the positions opened in June's stress. In round 1 they
    were gone: the 2026-06-19 holiday row had made every name ineligible and flattened the
    book.
  - The holiday rows also blocked the stress switch, because the market return was NaN on
    them and 21-bar volatility was undefined for the next 21 bars. Putting the three rows back
    into the v2 panels (`checks.py holidays`) cuts the stressed bars of 2026-H2 from 15 to 6
    on T1 and from 2 to 0 on T2 (2026-H1 T1: from 69 to 62).

### R2.6 v1 → v2, the headline numbers

| | TRAIN | T1 (P) | T2 | T4 (P) | gates failed | Q4 2018 / COVID / 2022 / Feb-Apr 2025 |
|---|---|---|---|---|---|---|
| U0, v1 | −7.37 | −1.11 (0.36) | −6.69 | −2.05 (0.29) | G1 G2 G3 G5 G6 G8 | −0.4 / −12.6 / +6.9 / +2.9 |
| **U0, v2 (final)** | **−7.68** | **−3.45 (0.16)** | **−6.54** | **−2.91 (0.20)** | **G1 G2 G3 G5 G6 G8** | **+0.1 / −11.5 / +7.3 / +1.4** |
| I4, v1 (round-1 final) | −2.46 | +1.73 (0.71) | −0.98 | +1.22 (0.61) | G1 G2 G3 G4 G5 G8 | +2.3 / −1.8 / +4.6 / +1.6 |
| I4, v2 | −2.59 | −3.10 (0.26) | +0.08 | +1.50 (0.64) | G1 G2 G4 G5 G8 | +2.2 / −1.3 / +4.2 / +0.8 |
| market (EW), v1 → v2 | | | | | | −18.2 / −37.0 / −25.2 → −24.5 / −19.6 → −19.5 |

- Adoption bar: 2.79 pts in v1, 2.60 in v2.
- I4 − U0 on T1: +2.84 (adopted by 0.045 pts) in v1, +0.36 (not adopted) in v2.
- Selection percentile: U0 100 → 99, I4 93 → 31.

### R2.7 Review issues → fix → effect under v2

| # | issue (severity) | fix | effect under v2 |
|---|---|---|---|
| 1 | "the reversal is real" overstated (major) | round 1: "suggestive" | Holds. U0v2 is at percentile 99 on the exact set and 93 / 96 / 99 for its copies in their own bands. TRAIN t's are unchanged (< 2). Costs (6.0-6.4 pts/yr) are twice the +3.0 pts over random. |
| 2 | I4's adoption fails under name noise (minor) | round 1: I4 presented as "a lower-turnover U0" | Now the decision: I4v2's T1 Δ is +0.36 < 2.60 (copies +0.34 / +0.55 / −2.27), and U0v2 is the final. |
| 3 | insignificant point estimates (minor) | round 1: t attached | TRAIN is unchanged. In the v2 stress split no stress − calm difference reaches \|t\| 2 (largest 1.53), and I4's T1 stress-day gain is gone. |
| 4 | numerical misstatements (minor) | round 1: corrected | Restated on v2 (R2.3, item 4). Plus the T4 "no order" correction (R2.5). |
| 5 | invented timestamps (minor) | round 1: the real times | Round-2 times are printed by `date` in each launching command (`round2.log`). |
| 6 | short under-fills; only approximately dollar-neutral (minor, engine) | the lab v2 runner | 0 under-filled openings (U0v2: 7,519; I4v2: 1,584). The call order moves alpha by −0.3 to +0.8 pts and no decision. The headroom advice is withdrawn for lab books. Still approximately neutral. |
| 7 | present-day sector codes (minor, lab) | caveat kept (v2 does not fix it) | C1v2 beats the clusters on all three samples, so the caveat now qualifies a disadvantage of the clusters, not a tie. |

What the five lab fixes did to this book:

| lab fix | effect |
|---|---|
| holiday rows | Part of the 2026 changes: whole-book exits, and the stress switch blocked about 21 bars after each row. I4's T4 2026-H2 "no order" came from this. |
| day 0 | 0.0-0.5 pt/yr at base costs; 0.3-0.9 under late fills. |
| starved shorts | 0 under-fills now; −0.3 to +0.8 pts. |
| borrow on hair re-trims | None in this book: 0 short re-sizes, identical fee frames. |
| ETFs in the universe | The largest effect, through the peer groups: −2.5 (U0), −5.1 (I4), +1.8 (C1) pts on T1, together with the holiday rows. |

### R2.8 Round-2 verdict: **no edge** (unchanged). The final is now U0.

- **Selection: suggestive, not established, and smaller than its costs (unchanged).** On T1,
  U0's selection beats same-turnover random names on all four name sets, by +3.0 to +5.6
  pts/yr (percentiles 99, 93, 96, 99). Its costs are 6.0-6.4 pts/yr, and nothing on TRAIN
  reaches t 2.
- **The final changes from I4 to U0.** I4's T1 gain over U0 (+2.84 in round 1, above the bar
  by 0.045 pts) is +0.36 on the v2 panels, so I4 is not adopted.
  - The cause is a six-name change of the universe (sector ETFs out, from 2024-H2), not the
    accounting fixes. It moved I4's T1 alpha from +1.7% to −3.1%.
  - I4's T1 number was always the exact name set's, as the round-1 noise copies said.
  - I4 stays what the review called it: a lower-turnover U0. It gains +5.1 / +6.6 / +4.4 pts
    on TRAIN / T2 / T4, about the costs it saves.
- **U0v2 fails G1, G2, G3, G5, G6 and G8** (TRAIN −7.7%, T1 −3.5% with P 0.16, T2 −6.5%, T4
  −2.9%; late fills −27%). It passes G4 (99) and G7 (beta 0.13).
- **The proposal's distinctive parts are unsupported, more clearly than in round 1:**
  - sector codes beat the correlation clusters on TRAIN, T1 and T2;
  - the stress switch adds −0.07 to +0.14 pts over a constant gross;
  - no book's stress − calm P&L difference reaches |t| 2 (largest 1.53).
- **The accounting fixes change no conclusion.** Day 0 costs up to 0.5 pt/yr (0.9 under late
  fills). The call order moves alpha by −0.3 to +0.8 pts and removes every under-fill. The
  costs conclusion stands: about 6 pts/yr at U0's turnover, and about 30 pts/yr of gross alpha
  needed to pass G5.
- **What survives** is a near-zero-beta book that beats the market in every bear episode
  (U0v2: +0.1 / −11.5 / +7.3 / +1.4% against −18 / −37 / −25 / −20%). It loses money in the
  average down month and in 10 of 11 calendar years (all but 2022).
