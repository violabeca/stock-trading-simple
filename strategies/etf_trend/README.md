# etf_trend — cross-asset trend following on ETFs (proposal 7)

**Thesis.** Time-series momentum (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017;
Faber 2007): hold each asset while its own trailing 12-month return beats cash, step aside when
it does not, size every asset by the inverse of its volatility. The textbook crisis alpha: in a
prolonged bear market equities trend down while Treasuries, gold or the dollar trend up.

> **Status (round 2, lab v2, 2026-10-23): no edge over SPY; a crash-protection sleeve.**
> The untuned default (U0v2) passes 1 of the 6 adapted gates (G1, TRAIN alpha). Round 2 re-ran
> every evaluation on lab v2 after the adversarial review: the review's critical defect (the
> long/short variant's borrow over-charged 4-5×) is fixed in `build_weights`. Every number
> below is restated on v2. No decision changed, and the final is still the untuned default.
> Two independent verifier passes re-ran U0v2, I1v2, I2v2, I4v2 and two noise copies bit for
> bit and traced every other round-2 number to saved files. They corrected what lowered the
> LOCKED alpha (the book's own day 0, not SPY's) and a few precision slips. No number, gate,
> decision or status changed.
> Research folder of the proposals lab: `RESEARCH_LOG.md` holds the pre-registrations, every
> variant, both rounds and the verdict (§0-§9 round 1, §R2 round 2, then the two "Round 2
> repair" passes). `IDEAS.md` parks the open questions and two proposals for code outside
> this folder (the engine's borrow charge, a splice check for the DB refresh).
> Evaluated by `research.py` on its own ETF panel, not by `tsim optimize` (no named universe).

Alpha vs SPY (CAGR − beta × SPY CAGR), U0v2:

| window | CAGR | SPY | beta | alpha (P) | max DD (SPY) | round 1 alpha |
|---|---|---|---|---|---|---|
| TRAIN 2004-15 | +6.5% | +6.2% | 0.20 | **+5.25% (1.00)** | −8.7% (−55%) | +5.63% |
| (VA 2016-20 / VB 2021-26) | +3.4% / +4.6% | +14.9% / +14.1% | 0.26 / 0.30 | −0.52% / +0.44% | | −0.21% / +0.74% |
| VALID 2016-26 | +4.0% | +14.5% | 0.28 | +0.02% (0.51) | −12.9% (−34%) | +0.34% |
| LOCKED 2000-03 (re-measured once) | −1.6% | −6.4% | 0.27 | +0.11% (0.51) | −19.7% (−49%) | +0.56% |

- About 70% of TRAIN's alpha is the 2007-09 crash. In the other ten years the book earned
  ≈ +1.8%/yr of low-beta carry (P 0.93). In VALID its asset timing lost to shuffled books (31st
  percentile), and without SHY's carry the alpha is negative.
- It beat SPY in every bear episode (2008-09 +4.6% vs −54.6%, 2000-02 −17.7% vs −48.5%, 2022
  −8.7% vs −24.3%, COVID −12.3% vs −33.4%), but lost −1.07% in SPY's average down month and
  trailed SPY in 17 of 23 years. Late fills (+50 bp a leg) cost 2.5 pts a year in VALID, 37% of
  that on SHY legs.
- Nothing beat U0 by the pre-registered bar, on either lab version:
  - the tuned pick (the same spike both rounds, PBO 0.55);
  - long/short (the best crisis book, 2008-09 +31.4% and 2022 +1.3%, but −3.9 pts in 2016-20);
  - equal risk per asset class (+3.1 pts in 2021-26 only, from DBC, gold, silver and the dollar);
  - blended lookbacks and Faber's 10-month average (+2-3 pts in 2016-20 only);
  - a 6% vol target (inert).
- **The SPY trend alone as an equity overlay protected only in 2008.** It lost the full fall in
  Q4 2018, COVID, 2022 and Feb-Apr 2025 (alpha −3.6%/yr in VALID).

**What round 2 changed.**
- Lab v2: each half-year counts its day 0, so it pays its opening rebuild (~0.12% of the book
  in TRAIN / VALID, ~0.24 pts of CAGR a year) and its first day's move. SPY's benchmark is
  built the same way. Shorts run through the lab's order sequence, and small short re-trims
  defer their borrow. This lowered U0's alpha by 0.3-0.4 pts a year (0.45 in LOCKED), and the
  other long/flat books' by 0.24-0.38 in TRAIN / VALID. The exception is the SPY/SHY switch
  (I5): −0.26 in TRAIN but +0.10 in VALID. In VALID it held SPY on 20 of its 22 first days,
  where its day 0 matched SPY's, and SHY on the other two (2019-07-01, 2023-01-03), when SPY's
  day 0 lost 0.46% and 1.04% (`results2/etf_trend/repair2_lab_effect.log`). For U0, almost all
  of the change is the book's own day 0 (−0.33 / −0.35 / −0.50 pts in TRAIN / VALID / LOCKED).
  Counting SPY's day 0 moved the alpha by only −0.02 / +0.01 / +0.07.
- The strategy fix: kept weights no longer move, plus `min_change`. It moved U0 by 0.04 pts and
  the long/short book by +0.6-0.7 pts (+0.2-0.3 net of lab v2).
- Orders are now all charged: 74 / 92 a year (TRAIN / VALID), or 40 / 58 without each
  half-year's rebuild.

## The book (`strategy.py`)

- Universe: 23 risky ETFs (SPY QQQ IWM EFA EEM, the nine sector SPDRs, VNQ, TLT IEF LQD HYG,
  GLD SLV DBC, UUP) plus SHY as the cash asset. An ETF joins at the first half-year period start
  after its first 12 months of history.
- Monthly: decided at the close of each month's first trading day, filled at the next open;
  weights constant in between.
- State: on when the 12-month total return beats SHY's; off budgets go to SHY (the engine pays
  0% on idle cash). Size: inverse EWMA volatility (centre of mass 60) over every live asset.
- A held asset keeps its weight, bit for bit, unless its new target moves by more than 25%
  (`band`). When kept weights push the gross above 0.98, only the assets that trade anyway are
  scaled down. A target change smaller than 0.25% of the book (`min_change`, SHY's included) is
  not traded; an exit always is. These guard against hair-sized orders, which pay the real $1
  minimum and which the engine's borrow model charged ~375× the accrual (round-1 review).
- Variants: `mode: long_short`, `blend` (several lookbacks), `rule: sma`, `vol_target`,
  `budget: class`, `overlay: SPY`, `assets`. Each is documented in `config.yaml`; the log says
  which were tested and why none was adopted.

## The harness (`research.py`)

Half-year periods: TRAIN 2004-2015, VALID 2016-2026-10-16 (split VA 2016-20 / VB 2021-26),
LOCKED 2000-2003 (read once for the final, re-measured once on lab v2). Benchmarks: SPY
buy-and-hold (primary) and the equal-weight buy-and-hold of the period's ETFs.

It reuses `strategies/proposals_lab/lab.py`'s book (lab v2), statistics, robust objective and
PBO. Results go to `var/proposals_lab/results2/etf_trend/` (labels ending in `v2`); round 1's
stay in `results/etf_trend/`. Cross-checks ride along with every result:
- the borrow owed on the realized shorts (`exact borrow`);
- late fills on the risky legs only;
- orders and the fee gap without each half-year's rebuild ("continuing").

```bash
uv run python strategies/etf_trend/research.py panel            # ETF panel from the DB (cached)
uv run python strategies/etf_trend/research.py premise          # TRAIN-only premise check
uv run python strategies/etf_trend/research.py run U0v2 '{}' --costs2
uv run python strategies/etf_trend/research.py report U0v2 <label>...
uv run python strategies/etf_trend/research.py luck U0v2
uv run python strategies/etf_trend/research.py noise U0v2
uv run python strategies/etf_trend/research.py verdict U0v2
uv run python strategies/etf_trend/research.py bear U0v2 <label>...
uv run python var/proposals_lab/results2/etf_trend/round2.py u0 noise luck tune variants report
```

## Data — and a side effect on the shared DB

The 24 ETFs were backfilled from yfinance (1993 → 2026-10-16) into the **shared** DB on
2026-10-22: `tsim backfill --provider yfinance`, an UPSERT of 152,925 daily rows. In round 2 a
fresh load from the DB equals the cached panel bit for bit.

**The UPSERT overwrote 13,406 existing rows that other consumers read.** They were:
- SPY from 2015-01-02: 2,212 yfinance rows to 2023-10-16, then 752 IBKR rows from 2023-10-17;
- GLD and SLV from 2015-01-02: 2,965 yfinance rows each (no dividends, so adjusted = close);
- XLE, XLF, XLI, XLK, XLV and XLY from 2023-10-17/18: 752 IBKR rows each.

The IBKR rows had adj_close = close (no dividend adjustment) and were joined to
yfinance-adjusted history: SPY's stored series jumped +3.43% on 2023-10-17, against a real move
of −0.005%. The new series is one consistent adjusted record per ETF. The rows as they were are
saved in `var/proposals_lab/results/etf_trend/db_backup_before_backfill.pkl`.

Who is affected:
- The six SPDRs are typed `equity`, so they sat in the lab's stock universes until lab v2
  excluded them.
- Lab v2 rebuilt its panels and `spy.pkl` from the current DB, so round-1 lab results of every
  strategy were computed on the old SPY and SPDR bars.
- A later IBKR daily refresh of these symbols can re-splice unadjusted bars onto the adjusted
  history. Proposal only (it belongs in the refresh path, not here): an adj_close/close splice
  check before such a refresh.

Security types are `etf` for 16 of the 24 (15 new + SPY), `equity` for the six SPDRs and
`metal` for GLD and SLV. The UPSERT leaves an existing type alone.
