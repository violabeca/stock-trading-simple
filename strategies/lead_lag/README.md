# lead_lag — does one stock's move predict another's LATER move?

**Research only: there is no `strategy.py`, `config.yaml` or study.** This folder is the go/no-go
exploration that precedes a strategy. It asks whether the lead-lag effect — also sold as "cross-asset
momentum" or "delayed correlation trading" — exists on this repo's data at **1-minute, 5-minute and
daily** bars, strongly enough to trade after costs. It also tests the two catalysts usually named
for it: **size** (large caps react faster than small ones) and **economic links** (industries,
supply chains, a metal and its miners).

| file | what |
|---|---|
| `lead_lag_explore.ipynb` | the exploration: §1 data → §2 power → §3 market → §4 pair map → §5 size → §6 links → §7 persistence → §8 tradability → §9 verdict |
| `research.py` | every computation, loaded by path (never through `strategies.load`); tested in `tests/test_lead_lag.py` |
| `links.yaml` | the pre-registered customer→supplier and metal→miner links (frozen; sha1 in the SCORECARD) |
| `IDEAS.md` | what would reopen the question, each with its falsification |
| `studies/lead_lag_explore.executed.ipynb` | the first full run with every output (git-ignored, local only) |

## The broad correlation map

The notebook maps **every leader × follower pair at once**. That is about 264,000 ordered pairs
among the ~514 names with 1m bars, and about 3.5 million among ~1,880 daily names. Each pair gets
a t-statistic for "the leader's last `L` bars predict the follower's next bar".

The map is one float32 matrix product per trading session (`research.pair_t`). The leader's lag
sum is one `cumsum` along the bar axis of a (session × bar × symbol) cube, so a lag cannot reach
into the previous session. The whole 1-minute map takes a fraction of a second, and the rolled
nulls take seconds.

## Five traps, and what the notebook does about each

1. **Stale prints — the #1 false positive.** IBKR reports no minute with fewer than 100 shares
   traded. Such a minute is stored as a flat placeholder at the previous close with volume 0: in a
   2025-10 sample, 100% of 2,230 volume-0 minutes were flat, and the smallest positive volume was
   100. How often that happens is largely a matter of dollar volume (rank correlation −0.56 on
   the 514-name pool, and +0.31 with price: IBKR drops odd lots). FICO (~$1,800 a share) and ALGT
   sit at 40–50% of minutes. A thin name's close-to-close
   return therefore "catches up" one bar late, which looks exactly like "large leads small" and
   survives any time-shift null. The notebook uses two different returns:
   - **Follower target:** each bar's own first-to-last print, `log(close/open)`, on bars with a
     print. That is what a trade entered at the first print after a signal earns; the catch-up
     sits in the excluded gap between bars.
   - **Leader signal:** the last-trade close-to-close return.
   - **5-minute bars:** placeholders are blanked before they are derived, so every 5m candle opens
     on a real print.
   - **Control:** the naive close-to-close map is kept as a diagnostic row.
   - Tests pin all of this, including a stale follower that "leads" close-to-close with T > 5 and
     shows nothing on the tradable target.
2. **Luck across ~264,000 pairs × several lags.**
   - The pair statistic is a **session-clustered t** (weeks at 1d). A plain `ρ·√n` produces about
     40× fake discoveries when volatility moves together across names (pinned in the tests).
   - The null is the same map with the leader **rolled by whole sessions** (20 rolls; at least 63
     days at 1d, 10 rolls). It destroys alignment but keeps each series' own structure.
   - Anything selected from the data is found on a **discovery** half and judged on a separate
     **validation** half.
3. **The market.** Every name follows the market a little late, which lights up the whole map.
   - Pairs are measured in **Granger form**: the follower's innovation after removing the "fast
     market" (the equal-weight of the top point-in-time dollar-volume fifth; SPY at 1d), its lags,
     and the follower's own lags.
   - The leader is taken net of its same-bar market move.
   - The equal-weight of all names would itself be stale (Lo–MacKinlay 1990).
   - Own lags matter too: own autocorrelation × same-bar correlation fakes a cross-lead
     (Boudoukh–Richardson–Whitelaw 1994).
4. **Power.** Before any map, §2 computes, per bar size:
   - the correlation that would pay for costs (`bt.leg_costs` round trip ÷ E|z| of a |z| > 2 trade
     ÷ the hold's σ);
   - the smallest correlation the map can detect;
   - a **canary**: 20 leads planted at the break-even correlation must be recovered.

   At 1d the pair map cannot see a tradable pair: detectable ≈ 0.2 against a break-even ≈ 0.01.
   So the daily verdict rests on group tests (size fifths, industries, links), and the daily pair
   checks read *n/a*.
5. **Time of day.** The open moves several times more than midday. Returns are scaled by a
   **causal** trailing RMS × a pooled time-of-day profile, fitted on discovery. The first 5 minutes'
   targets are blanked, since the auction's spread is not tradable.

## Data (shared DB, 2026-09-28)

- **1m** (IBKR, regular hours, as traded): 514 names with ≥ 200 sessions, 2025-06-18 → 2026-06-26.
  - The pool is the HighOctane pool.
  - Median dollar volume is $46M/day, ranging from about $2M to $2B.
  - **No ETFs and no megacaps inside the window.** AAPL, MSFT, NVDA and TSLA have 1m bars only
    from May/June 2026, so they form a pre-registered **MEGA** leader group, tested only on their
    own sessions. That group has low power and covers a single regime.
- **5m** is derived from 1m.
- **1d** (Yahoo, adjusted): 2,302 names.
  - Equities with ≥ 95% coverage over the window form the universe.
  - The only non-equities are SPY (the market), GLD and SLV (leaders in the links test).
- **Size:**
  - Market cap is `fund_shares` restated into split-adjusted units (turnover_momentum's
    `_in_volume_units`) × close, taken at the last bar before the window.
  - The rival is median dollar volume over the half-year before the window (Chordia–Swaminathan
    2000).
  - ADR share counts are off by the ADS ratio; rank statistics soften that.
- **Industries:** Nasdaq sector/industry from `securities`. This is a 2026 snapshot, not
  point-in-time.

## Windows — all available data (user decision, 2026-09-28)

| | discovery | validation | the first bar a later strategy may call out-of-sample |
|---|---|---|---|
| 1m / 5m | 2025-06-18 → 2025-12-31 | 2026-01-02 → 2026-06-26 | **2026-06-29** |
| 1d | 2020-07-01 → 2023-06-30 | 2023-07-01 → 2026-06-12 | **2026-06-15** |

This study reads the repo's usual holdouts (1d 2025+, 1m May–June 2026). A strategy built on its
findings must therefore validate on bars after these dates, or on paper trading.

## Pre-registration (written 2026-09-28, before the first run)

The thresholds live in `research.THRESH`. The notebook's verdict reads only them.

**Settings per bar size** (`research.SPEC`):

| setting | 1m | 5m | 1d |
|---|---|---|---|
| lag window `L` | 5 bars | 3 bars | 1 day |
| market lags | 1..15 | 1..12 | 1..5 |
| hold for tradability | 5 bars | 3 bars | open → close of the next day |
| target blanked after the open | 5 min | 5 min | — |

The slow daily links map uses lags 2..22.

| | check | pass |
|---|---|---|
| V1 | null calibrated | std of the rolled maps' T in [0.85, 1.20], and a share with \|T\| ≥ 4.5 of at most 3.4e-5 |
| V2 | canary (power) | ≥ 16 of 20 leads planted at the break-even correlation recovered at T ≥ 4.5; otherwise that bar's pair checks read *n/a* |
| V3 | windows stamped | the SCORECARD records the windows read |
| C1 | the market leads | pooled Granger t ≥ 3: the fast market's last `L` bars → the tradable next bar of the equal-weight of all other names, full window. The MEGA group is reported, never gating |
| C2 | pairs beyond the market (discovery) | count of \|T\| ≥ 4.5 above the maximum over rolls; empirical false-discovery rate (mean rolled count ÷ real count) ≤ 0.20; the busiest follower owns ≤ 25% of the discoveries |
| C3m | the market lead persists | Spearman of each name's lag-t, discovery vs validation, ≥ 0.30; pooled validation t ≥ 2 |
| C3p | the pairs persist | the discovery top 100 have a mean signed validation T ≥ 2 and ≥ 65% keep their sign; both above the 95th percentile of roll-selected top 100s |
| C4 | size (descriptive) | Spearman(lag-t, log market cap) ≤ −0.20 and its rank-regression t ≤ −3 given stale share and price; size-fifth map T(big → small) ≥ 3 and T(small → big) ≤ 1 |
| C5a | industries (descriptive) | same- minus cross-industry mean T above the 99th percentile of 1,000 label permutations |
| C5b | Hou 2007 (descriptive) | big → small minus small → big inside industries (≥ 6 names), t ≥ 2 across industries |
| C5c | links, 1d (descriptive) | on the fast OR the slow map: the linked leaders' mean percentile within their follower ≥ 0.60, random-leader p ≤ 0.005, above the reverse direction |
| C6a | market lag tradable (validation, fitted on discovery) | on the 10% largest fast-market moves: slowest-decile minus fastest-decile names (by discovery lag-t), entered at the next open and held `hold` bars. Net of 2 round trips, the mean must be > 0 with cluster t ≥ 2, and still > 0 when entered one bar late |
| C6b | pair network tradable (validation, fitted on discovery) | each follower's 3 strongest discovery leaders, signed. Trade when the prediction's size is in the top 1% of discovery; net of a round trip, the mean must be > 0 with t ≥ 2, and > 0 one bar late |

**GO for a bar size** = V1 ∧ [ (C1 ∧ C3m ∧ C6a) ∨ (V2 ∧ C2 ∧ C3p ∧ C6b) ]. A channel must exist,
persist and pay, all on that same channel. C4 and C5 describe what drives any effect; they shape a
strategy's design and never decide the GO. A GO means "worth building a strategy". It is not a
backtest result.

## How to run

- Open `lead_lag_explore.ipynb` with the trading-simple kernel.
- `SUBSET = 40` gives a quick pass: 40 1m names and 200 daily names.
- `SUBSET = None` is the full run: the 1m load is month by month, and peak memory is a few GB.
  Run one 1m notebook at a time, never next to a `tsim` process.
- Offline tests: `uv run pytest tests/test_lead_lag.py -q`.
- If index ETFs are ever wanted at 1m, register QQQ and IWM as `security_type="etf"` in
  `securities` first. `tsim backfill` inserts `equity` and never corrects the type, which would
  pollute the equity point-in-time screens.

## Findings (first full run, 2026-09-28)

The run covered 514 names at 1m/5m and 1,829 at 1d. It took 10 minutes, with 3.2 GB peak memory.
The executed copy is `studies/lead_lag_explore.executed.ipynb` (git-ignored).

**Verdict: NO-GO at 1m, 5m and 1d.** A lead-lag exists at 1 minute and it is real and persistent,
but it is about ten times too small to pay for a round trip.

| check | 1m | 5m | 1d |
|---|---|---|---|
| V1 rolled nulls calibrated | PASS (T std 0.99–1.01) | PASS | PASS |
| V2 canary (power) | PASS (20/20) | FAIL (0/20) | FAIL (0/20) |
| C1 the market leads | FAIL (t 1.2) | FAIL (t 1.0) | FAIL (t −0.25) |
| C2 pairs beyond the market | **PASS**: 268 pairs vs a roll max of 3, FDR 0.3% | n/a | n/a |
| C3m the market lead persists | FAIL | FAIL | FAIL |
| C3p the pairs persist | **PASS**: top 100, validation T 2.5, 95% keep their sign (null 0.17 / 56%) | n/a | n/a |
| C4 size (descriptive) | FAIL | FAIL | FAIL |
| C5a industries (descriptive) | PASS | PASS | FAIL |
| C5b Hou, big → small in industries (descriptive) | PASS (t 4.1) | PASS (t 2.2) | FAIL |
| C5c links (descriptive) | n/a | n/a | PASS, marginal |
| C6a market lag tradable | FAIL: gross 0.8 bp vs 13.6 bp for 2 round trips | FAIL: 2.3 bp | FAIL: 12.6 bp vs 12.7 |
| C6b pair network tradable | FAIL: gross 0.6 bp vs 6.8 bp | n/a | n/a |

### What exists

1. **A 1-minute lead-lag network beyond the market.**
   - 268 of 263,682 ordered pairs reach |T| ≥ 4.5 on the discovery half; 20 rolled nulls give 0.7
     on average (at most 3).
   - The strongest 100 hold on the separate validation half, with a mean signed T of 2.5. 95%
     keep their sign.
   - The best lag is always the first minute.
   - The top pairs are **theme clusters** and closely linked names, not a size ladder:

     | cluster | leaders → followers |
     |---|---|
     | quantum computing | QUBT, QBTS, RGTI → AMPX, QS, AEVA |
     | space | RKLB, RDW → BKSY |
     | drones | RCAT ↔ ONDS |
     | nuclear | NNE, OKLO → JOBY, ONDS |
     | crypto miners | CIFR, IREN |
     | tankers | FRO → CMBT |
     | European bank ADRs | HSBC → SAN |
     | photonics | LITE → LASR |

     The full 100 are in the notebook's `screens["1m"]`.
2. **Industry structure intraday.** A leader pulls followers in its own industry harder than the
   rest: above all 1,000 label permutations at 1m and 5m. Inside industries the **big half leads
   the small half** (Hou 2007): t 4.1 at 1m and 2.2 at 5m, over 28 industries. Neither survives
   to 1d.
3. **Links at 1d, weakly** (89 pre-registered edges covered).
   - Metal → miners: GLD/SLV rank at the 64th percentile among all leaders of their miners.
   - Customer → supplier: 59th percentile, but the reverse direction reaches 56th. The link
     co-moves either way; the direction adds little.
   - AAPL → SWKS, QRVO and CRUS rank at the 98th–99th percentile.
   - The slow, monthly-style map (lags 2..22) shows nothing: 49th percentile.

### What does not

1. **None of it pays.**
   - The 1m network, traded on validation with everything fitted on discovery, makes **0.6 bp gross
     per trade** over 224,000 trades. Its rank IC is 0.003 (t 2.3). A round trip costs 6.8 bp.
   - Its strongest discovery pairs sit at a correlation of about 0.03 per bar, below the 0.045
     break-even before any shrinkage.
   - The market-lag trade (slowest minus fastest names after the fast market's biggest moves)
     makes 0.8 bp (1m), 2.3 bp (5m) and 12.6 bp (1d) gross against 2 round trips, with t ≈ 0 at 1d.
   - Entering one bar late is worse everywhere.
   - Even a zero-commission account pays the 2 bp spread.
2. **The market does not lead.** The most-traded fifth's last moves do not predict the rest beyond
   the rest's own past: pooled t 1.2, 1.0 and −0.25. Instead the fast market **reverses its own
   move** (t −12.6 at 1m, −4.3 at 5m). The mega caps (18 overlapping sessions) lead nothing
   (t −0.2 and −0.3).
3. **"Small caps lag" is "names that rarely print lag".**
   - At 1m the lag t falls with market cap (Spearman −0.29).
   - Given the no-print share and price, the market-cap coefficient turns positive (t +2.1). The
     no-print share carries it (t +11.4); dollar volume adds nothing (t −0.3).
   - A post-hoc delay measure (the share of a name's market response that arrives late) agrees:
     +0.47 with the no-print share, and t +1.3 for cap once controlled.
   - The Lo–MacKinlay size fifths show no big → small lead at any bar size (T 0.3, 0.6, −0.2).
   - At 1d, big names show the *larger* capturable delay (cap t +12.7; late share t +13). This fits
     small names catching up in the overnight gap, which a next-open trade cannot capture.
4. **Stale prints manufacture most naive lead-lag.** On the 1m discovery half:
   - 36,304 pairs "lead" close-to-close;
   - 904 remain on the tradable target;
   - 268 remain beyond the market.

   A plain lagged-correlation screen on 1-minute closes is therefore mostly a map of which names
   print rarely.
5. **Pair maps are blind at 5m and 1d.**
   - At 5m the map can detect a correlation of 0.053, against a 0.036 break-even per bar.
   - At 1d it can detect 0.19, against a 0.013 break-even.
   - The canaries confirm both (0/20).
   - The daily pair map's discoveries equal its nulls (4 vs 4.2).

**Reading it.** The picture matches the literature. Intraday lead-lag among US stocks has thinned
out as trading got faster (Tóth & Kertész 2006; Curme et al. 2015). What remains at the 1-minute
scale is statistically sharp but economically tiny, and it lives in the first minute, where
1-minute bars see only its tail. What might still be worth a look, each with its falsification, is
parked in `IDEAS.md`: theme baskets at longer horizons, a monthly supply-chain portfolio, and SPY
and QQQ at 1m.

## Literature the design leans on

- Lo & MacKinlay (1990): large portfolios lead small ones, partly through non-synchronous trading.
- Boudoukh, Richardson & Whitelaw (1994): own autocorrelation × contemporaneous correlation.
- Chordia & Swaminathan (2000): trading volume, not size, orders the speed of adjustment.
- Hou & Moskowitz (2005): price delay.
- Hou (2007): big → small inside an industry.
- Cohen & Frazzini (2008) and Menzly & Ozbas (2010): customer → supplier drift at monthly
  horizons.
- Tóth & Kertész (2006) and Curme et al. (2015): intraday lead-lag networks among US stocks thinned
  out as markets got faster.
