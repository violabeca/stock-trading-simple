# informed_flow_ls

`informed_flow` as a **dollar-neutral long/short book**, built 2026-10-19 and improved in an
autonomous research loop on 2026-10-20 (`RESEARCH_LOG.md`). The long-only `informed_flow` (v41)
owned the stocks short sellers avoid. This version also **sells short** the stocks they crowd
into, so it earns on the gap between the two ends of the ranking and can make money when the
market falls. It also listens to **option traders**, and pays a **borrow fee** on every short.

| file | what it does |
|---|---|
| `strategy.py` | the book: two signals → a smoothed composite rank → 15 longs and 15 shorts (pure) |
| `config.yaml` | every parameter; four carry the walk-forward's search range |
| `research.py` | the notebooks' shared helpers (universe, panels + `si_shares` + IBKR fees, statistics, nested walk-forward, verdict) |
| `informed_flow_ls_signals_marimo.py` | data coverage, the two signals at both ends, one book |
| `informed_flow_ls_walkforward_marimo.py` | the nested walk-forward, luck, the verdict, the paper book |
| `RESEARCH_LOG.md` | the 2026-10-20 loop: every hypothesis, number and decision, in order |
| `IDEAS.md` | the pre-registrations and the parked ideas |

Run the notebooks with `uv run marimo edit strategies/informed_flow_ls/<notebook>.py` (or the VS
Code marimo extension). Build the option cache first (`notebooks/03_altdata_update.py`, §5b,
or `providers.marketdata.build_cache()`). The walk-forward notebook's §2 takes ~20 minutes the
first time (cached afterwards), its luck band a few more.

## Status (2026-10-22)

**Ship the untuned config (v8 = v5 + a 100% squeeze stop + IBKR-calibrated borrow fees + a
crowding measure without the split look-ahead); paper trade it before any money.**
Measured on the refreshed data (short interest complete for every name since July 2026, the
2026-09-15 and 09-30 settlements read for the first time), walk-forward 2022-01 → 2026-10-16
(10 six-month periods, the point-in-time top 500):

| the v8 book (config values) | |
|---|---|
| CAGR / volatility / max drawdown | +13.6% / 10.6% / −14.0% |
| beta to the equal-weight market | 0.12 |
| **alpha** (CAGR − beta × buy & hold +8.7%) | **+12.5%/yr**, block-bootstrap P(alpha > 0) 0.996 |
| against 100 random-name books with the same rules | beats all (their 95th percentile: +4.6%) |
| with 2× costs / 2× borrow / every leg 50 bp late | +11.5% / +12.3% / +8.6% |
| the long side alone + a SPY hedge (no single-name shorts) | +6.8% — the short leg adds 5.8 pts |
| without the P&L of its five biggest winners | +6.8% |
| **names ranked 501-1000 (fresh cross-section)** | **+4.4%/yr** |
| **names ranked 1001-1500 (locked until the end of loop 2)** | **+4.0%/yr** |
| **names ranked 1501-2000 (locked until loop 5, read once)** | **+10.5%/yr** |
| **2020-07 → 2021-12 (before every design decision)** | **+3.3%/yr** |
| **2018-07 → 2020-06 (read after loop 2; two sell-offs; every liquid name the DB has)** | **−4.8%/yr** |
| periods with alpha > 0 | 8 of 10 (2024-H2 −0.5%, 2026-H2 −25.1%) |

**v8's one change is a correction, not a tweak** (RESEARCH_LOG §42-§44). The crowding measure
(short interest as a share of shares outstanding, which drives the "never short above 30%" rule
and the borrow tiers) was days-to-cover × the panel's volume ÷ SEC's share count. The stored
volumes are split-adjusted BACKWARD, so before a forward split it read the split factor too high
(NVDA before its 2024 10:1 split: ~22% instead of 1.2%) — knowledge of the future split, which
kept the book from shorting future big winners. v8 reads FINRA's own shares sold short ÷ SEC's
count, each as of its public date (`si_shares`, attached by `research.load`). It costs 0.1 pt on
the top 500, 0.5 pt of the five-test mean and 2.0 pts on names 1501-2000: what the leak was worth.

**v6's one change** — buy a short back at 100% above its entry, not 50% (RESEARCH_LOG §32-§34):
the 50% stop bought back at the squeeze's top (14 stopped shorts lost 29% each, then fell 9%).
Against v5 on the same days: +2.3 pts on the top 500 (P 0.98, no period worse), +0.4 on names
501-1000, +1.1 on 2020-21, +0.3 on the locked names 1001-1500; 0.75 and no stop also beat 0.5.
Its worst single short is deeper (IBM −65% over three years, ~2% of the book).

**The pre-registered primary — the nested Optuna walk-forward — still FAILS its own gates**: on
the refreshed data the tuned picks earn +6.6% against the config's +10.3% (v5) and the median
probability of backtest overfitting is 0.74. Its one stable preference (a wider squeeze stop) is
what v6 adopted after independent tests; nothing else in the searched knobs is worth tuning.

**Read this before trading it — a third of the long side is a measurement tilt.** Foreign
private issuers (20-F / 40-F / 6-K filers, ~7% of the universe) hold 30% of the long slots:
their FINRA short interest covers only the US line, so they look lightly shorted by construction.
Excluding them (refreshed data) leaves +5.0%/yr on the top 500, +0.5% on names 501-1000 and
−0.6% on 2020-21 — the tilt helps in all three tests, and filling the gap is possible only for
Canada for free (RESEARCH_LOG §30). The decision — keep the tilt or exclude them — is open.

**And half the alpha is five names.** Without the P&L of its top 5 contributors (NVDA, CLS,
PBR, AMD, KGC) the walk-forward alpha falls by about half (RESEARCH_LOG §27). The book's alpha
and its crash risk come from the same very volatile, heavily traded names: capping them cuts the
2026 drawdown and the 2025 gains alike (loop 2, N3). The honest reading: a real but thin
short-interest edge, amplified by a momentum/turnover tilt and a foreign-filer tilt.

**In falling markets** (RESEARCH_LOG §37): with a beta of ~0.1 the book is protected, not
guaranteed to profit — 2022 +14.2% while the market fell 19.4%, March 2020 flat while it fell 17%,
but late 2018 −11.5% while it fell 6.8%. The alpha sits mostly in 2022-2025.

**What to expect forward:** less than the table. The design was chosen on these same periods
(two loops, ~40 screens, 3 changes kept); the fresh names earn about a third of the top-500
alpha; the database holds only survivors; and 2026 — the most recent year — is the book's worst
(July-October: the long side's crypto / AI / nuclear names fell together).

## The signals

1. **Short interest**: FINRA days-to-cover, averaged over `dtc_window` bars. Low ranks long,
   high ranks short (Hong, Li, Wang & Zhao; Boehmer, Jones & Zhang). It is the persistent LEVEL
   that carries the information: new short selling (the 42-day change) and short interest
   against the name's own past year both lost badly (RESEARCH_LOG §6, §17).
2. **The change in put skew** over `skew_window` bars. `opt_skew` is the IV of the 95% put minus
   the ATM call, from the end-of-day option chains, averaged over `skew_smooth` (5) sessions
   first. Puts getting dearer ranks short; cheaper ranks long (An, Ang, Bali & Cakici 2014).
   Chains exist from October 2025; before that the sleeve is silent. On the current design it
   adds ~2 pts, almost all in H1 2026 — the half-year short interest inverted.

Each signal is ranked inside five dollar-volume groups (days-to-cover has volume in its
denominator), and the weighted composite is averaged over `score_smooth` (5) sessions before it
is ranked. **Why the smoothing:** v1 traded the raw daily skew change; the book crossed its
bands day after day and in 2026 fees took 10% of the book (RESEARCH_LOG §1). The two averages
took v1's walk-forward alpha from +2.9% to +10.5% and held on the fresh names (+1.3 pts).

## The book

- **Every short pays a borrow fee** (the engine's `borrow` key, per calendar day held). Where
  Interactive Brokers' own rate is known it is charged: `scripts/ibkr_shortstock.py` saves IBKR's
  public short-stock file each morning to `var/ibkr/`, and `research.load` attaches the newest
  one for five days (the paper book marks each short "IBKR" or "tier"). Elsewhere — all of the
  backtest history — tiers calibrated to IBKR's file at about twice its 90th percentile: 0.75%/yr
  up to 10% of shares sold short, 1% up to 20%, 3% above (v7; IBKR's median today is 0.4%).
- **Crowding** = FINRA's shares sold short ÷ SEC's shares outstanding, each as of its own public
  date (v8). A ratio above `si_plausible_max` (100%) is read as unknown — after a split FINRA's new
  count meets SEC's old one until the next 10-Q — so the name is not shorted and pays the lowest
  tier. **Names above 30% of shares sold short are never shorted**; a held short reaching it is
  bought back. **The squeeze stop:** a short 100% above its entry is bought back and not shorted
  again until its rank has left the short zone once (50% until v5: it sold at the squeeze's top).
- **Dollar-neutral, 50% long / 50% short** (no margin). The engine locks short proceeds
  (portfolio mode, 2026-10-19), so the book never exceeds ~100% gross.
- **Eligibility:** a price ≥ $5, a days-to-cover and an SEC share count (most foreign filers
  have none; their FINRA short interest covers only the US line, v37).
- **Bands:** a long enters at rank 0.90 and leaves below 0.80; shorts mirror at 0.10 / 0.20;
  15 longs and 15 shorts (`long_slots` / `short_slots`, settable apart), held at most 126 bars
  unless still in the entry zone; market-on-open orders.

## What the 2026-10-20 loop tried (RESEARCH_LOG.md has the numbers)

Kept: `skew_smooth` 5, `score_smooth` 5. Tried and rejected (paired against the current book on
the same days): short interest as a share of shares; turnover, momentum or sector groups; a
beta-neutral side split; a change-in-short-interest sleeve; weekly rebalancing; a slower skew
change; more smoothing; half or no option weight; buying only above the 200-day average; only
profitable longs; only unprofitable shorts; short interest against its own year; 20 slots a side
(worse in all 10 periods: the edge sits in the extreme names); wider exit bands. Two short
filters (no shorts above the 200-day average; no shorts above 20% of shares) helped on the top
500 and were adopted — then HURT on the names ranked 501-1000 (−5.0 and −3.9 pts) and were
reverted: the loop's clearest lesson in how far a walk-forward can be fitted.

The book's daily returns load 0.23-0.28 on 12-1 momentum and lean to high-turnover names, but
alpha after market, momentum and turnover factors is still +11-13%/yr (t 2.6-3.0).

## What the 2026-10-21 loop tried (loop 2, RESEARCH_LOG §31-§35)

Every change was paired against v5 on three tests at once (top 500, names 501-1000, 2020-21) under
a pre-registered bar and a budget of 12 variants (10 used); a fourth test (names 1001-1500) was
read once at the end. Kept: the 100% squeeze stop. Re-tested on the refreshed data and still
rejected: no shorts above the 200-day average (its old fresh-names loss was mostly the data gap,
but it loses 2020-21), no shorts above 20% of shares, a 84-bar days-to-cover average. New and
rejected: waiting 5 bars before entering, a narrower exit band, a volatility cap on new longs
(cuts the 2026 drawdown but gives back 2025). **A cost option for live trading:** exiting only
when BOTH the composite and the short-interest rank have left the band halves trading (296 → 131
orders a year) and raises the late-fill alpha by 2 points, at −0.8 points of normal-cost alpha —
not adopted under the gross-alpha bar, worth considering if real fills are worse than the open.

## What the 2026-10-21 robustness loop tried (loop 3, RESEARCH_LOG §38-§40)

The question was whether more stocks, uneven long and short counts or a wider universe would
make the book less dependent on a few names and one regime. Judged on five tests at once (top 500,
names 501-1000 and 1001-1500, 2020-21, 2018-20) by pre-registered robustness measures — alpha
without the five biggest winners, the share of positive periods, the worst test — **nothing did:**

| variant | T1 alpha | without top 5 | positive periods | worst test |
|---|---|---|---|---|
| **15 longs / 15 shorts (v7)** | **+12.7%** | **+6.8%** | **65%** | **−3.3%** |
| 20 / 20 | +7.9% | +3.6% | 60% | −4.9% |
| 30 / 30 | +7.6% | +4.8% | 57% | −5.3% |
| 15 longs / 30 shorts | +11.2% | +5.6% | 62% | −3.6% |
| 30 longs / 15 shorts | +9.1% | +5.5% | 62% | −4.8% |
| top-1000 universe, 15 / 15 | +10.4% | +3.8% | 41% | −13.3% |
| no side over 3 names of a sector | +12.8% | +7.2% | 62% | −6.1% |

The edge sits in the most extreme names of the most liquid stocks; every form of breadth dilutes
it. The sector cap is the one useful risk lever (2022-26 max drawdown −14.0% → −10.7% at the same
alpha) but loses 2018-21. Kept from the loop: the IBKR borrow model above, and the per-side
counts as a setting.

## What the 2026-10-21/22 stop and creative loops tried (loops 4-5, RESEARCH_LOG §41-§43)

**Trailing stops** (loop 4) on longs at 15/20/30% below the peak, volatility-scaled, and on both
sides at 25%: all worse. They sell the big winners, and the replacements fall in the same crash.

**Ten creative ideas** (loop 5, found by six discovery agents with different lenses) — each run at
its declared setting on five tests AND tuned in a nested walk-forward with wide search ranges
(100-200 trials per half-year, the off setting always inside). None survived:

| idea | top 500 Δ | verdict |
|---|---|---|
| fill shorts with high-beta names first | +2.0 (tuned +2.9) | loses 2020-21; partly credit for a lower beta |
| fill longs with low-beta names first | 0.0 | drawdown −7.5% but one period, and beta credit |
| halve gross while the book's volatility is high | −3.2 | fails |
| put/call open-interest sleeve | **+4.3** (tuned +4.0) | loses the next 500 names in the same months; one year of options |
| size longs by the change in 13F holders | +0.3 | fails |
| analyst target revisions for longs | +1.8 | loses names 501-1000 |
| half-size new entries for 21 bars | +0.9 | fails |
| short sizes scaled by their volatility | +0.9 (tuned +1.2) | near miss: names 501-1000 −0.1 |
| pause entries in high-dispersion weeks | +1.0 | fails |

The wide searches never rescued a failed idea; they only confirmed the near misses. The search's
best find was the split look-ahead above. The put/call sleeve is the idea to revisit once paper
trading has added option history.

## Limitations

- **Survivorship.** The database holds no name that left before 2026 (no Twitter, VMware, SVB,
  First Republic …): every number here is on survivors.
- **Borrow is modelled, not known**; locates, recalls and hard-to-borrow refusals are not.
- **Two smaller split effects remain.** For a few weeks after a split FINRA's new short interest
  meets SEC's pre-split share count (caught as "unknown" only above 100%; ~4% of short name-days
  read a ratio k× too high), and the $5 minimum price reads the split-adjusted close (no raw
  price is stored). Both need a split-events table (engine proposal).
- **Headless runs read the old measure.** `si_shares` is attached by `research.load`, not by
  `features.py`, so `tsim optimize` and `notebooks/04_compare.ipynb` fall back to the leaky
  days-to-cover × volume measure — with a warning (engine proposal: a `si_shares` builder).
- **Option data covers one year**: the skew sleeve's value rests on one half-year.
- **2026 broke the signal**: short interest inverted and momentum crashed in Q3 (the book's
  last period −22%). One bad year does not kill a signal, but it is the newest evidence.
- **No data is untouched before 2026-10-17**: paper trading is the only clean test.

## Study history

| version | date | what changed | verdict (tuned primary) | untuned config |
|---|---|---|---|---|
| v1 | 2026-10-19 | first version (pre-registration v1) | FAIL: P 0.80, PBO 0.55, late fills −8.0% | +2.9% |
| v2 | 2026-10-20 | skew_smooth 5 + score_smooth 5 | FAIL: tuning −3.4, PBO 0.74 | +10.5% |
| v3 | 2026-10-20 | + no shorts above the 200-day average | FAIL: PBO 0.61 (all else passes) | +14.4% |
| v4 | 2026-10-20 | + no shorts above 20% of shares | FAIL: tuning −4.0, PBO 0.68 | +15.6% |
| v4-O1/O3 | 2026-10-20 | smaller search space / config guard | FAIL: tuning −2.9 / −1.2 | +15.6% |
| v5 | 2026-10-20 | the v2 design (v3-v4's filters failed the fresh names) | FAIL: tuning −3.4, PBO 0.74 | +10.5%, every applicable gate passes |
| v5 (refreshed data) | 2026-10-21 | short interest complete since July 2026 | FAIL: tuning −3.7, PBO 0.74 | +10.3% |
| v6 | 2026-10-21 | + `short_stop` 1.0 | (searched knob; the inner study's stable pick) | +12.6%, every applicable gate passes |
| v7 | 2026-10-21 | + IBKR-calibrated borrow tiers, IBKR's own fee when known, `long_slots` / `short_slots` | (not re-run: no searched knob changed) | +12.7%, every applicable gate passes |
| **v8** | 2026-10-22 | **crowding from FINRA's shares short (the split look-ahead removed), `si_plausible_max`** | (not re-run: no searched knob changed) | **+12.5%, every applicable gate passes** |
