# proposals_lab — one yardstick for the proposals and the cross-strategies

Built 2026-10-22 to evaluate `STRATEGY_OVERVIEW.md` §6 proposals 2-5, two of the "also considered"
ideas (earnings drift, cross-asset trend) and, in a last round, cross-strategies that combine
what works. Every candidate is judged by the same code (`lab.py`) on the same panels, so numbers
are comparable across strategies. The user granted full authority to depart from repo
conventions where an idea needs it; the **statistical protocol below is what keeps the answers
honest** and is not optional.

## Data (`lab.py build`, cached in `var/proposals_lab/`)

- Daily panels 2015-01 → 2026-10-16 for every test period's names, with `short_interest`
  (FINRA days-to-cover, from 2018-01), `fund_shares/equity/assets/ni_ttm/gp_ttm` (SEC XBRL), the
  `earn_*` frames (yfinance reports in the shared DB's `earnings` table, 2015-01 → **2026-06-11**:
  the last test period has no new reports) and `sector` (constant numeric codes,
  `sectors.json`). SPY separately (`lab.spy()`, `lab.with_spy(panel)`).
- **Earnings timing:** every `earn_*` value sits on its USABLE bar, the second session after the
  report date (decided at that close, filled next open), so a report yfinance dated a day early
  still cannot be traded before it was out. `earn_day` marks the report session itself — for
  measuring a reaction, never as a decision date.
- Only ~470-506 names have daily bars before 2020 (the DB holds survivors only): before 2020
  "top 500" is nearly every name. Shorts look worse and longs better than reality.

## Tests (`lab.TESTS`) — half-year periods, universe ranked on the 6 months before each

| test | window | names | role |
|---|---|---|---|
| E0 | 2016-07 → 2018-06 (4) | top 500 | TRAIN (price/fundamental/earnings signals only: no short interest before 2018) |
| TB | 2018-07 → 2020-06 (4) | top 500 | TRAIN (Q4 2018, COVID) |
| T3 | 2020-07 → 2021-12 (3) | top 500 | TRAIN |
| T1 | 2022-01 → 2026-10-16 (10) | top 500 | VALIDATION (2022 bear, Feb-Apr 2025) |
| T2 | 2022-26 (10) | names 501-1000, $10M floor | VALIDATION (other names) |
| T4 | 2022-26 (10) | names 1001-1500, $3M floor | **LOCKED**: read once, for a proposal's final candidate |
| TL | 2022-26 (10) | names 1501-2000, $1M floor | **LOCKED** for the cross-strategy finals only |

Every backtest is warm (signals on the whole panel from 2015, the book scored from the period's
first day), fills at the next open, pays the repo's cost model at the strategy's
`order_notional`, and is compared with the equal-weight buy-and-hold of the same names.
**alpha** = CAGR − beta × buy-and-hold CAGR.

## The protocol every proposal follows

1. **Pre-register** in the strategy's `RESEARCH_LOG.md`, before any number: hypothesis,
   mechanism, literature-default config, expected result, variant budget (≤ 8).
2. **Premise check on TRAIN only** (an IC / decile spread against beta-adjusted forward
   returns): informational, it decides nothing alone.
3. **U0, the untuned default:** `lab.py run` on OPEN (E0, TB, T3, T1, T2) with late fills and 2×
   costs; `lab.py noise` (5 names dropped a period, 3 copies) = the path noise of this book;
   `lab.py luck` on T1 (100 shuffled-name books).
4. **Optimization:** `lab.tune` on TRAIN only (11 periods, ≥ 120 trials, robust objective =
   mean − ½ sd of the period alphas; PBO recorded). The tuned values are adopted only if they
   also beat U0 on T1 AND T2 (paired); otherwise U0 stays and the result is reported.
5. **Iterations (≥ 2):** each a design change written from what 2-4 showed, pre-registered
   with its mechanism. **Adopted only if** its paired Δ against the current base is ≥ 0 on
   TRAIN, T1 and T2, and its T1 Δ exceeds twice the noise sd. Count every variant.
6. **Final candidate** (declared in the log before reading): `lab.py verdict`, `lab.py bear`,
   then **T4 read once** (final against U0).

## The gates (`lab.verdict`)

| gate | bar |
|---|---|
| G1 | TRAIN-era (E0+TB+T3 stitched) alpha > 0 |
| G2 | T1 alpha > 0 with block-bootstrap P ≥ 0.90 |
| G3 | T2 alpha > 0 |
| G4 | T1 beats ≥ 95% of the shuffled-name books |
| G5 | T1 alpha > 0 with late fills (+50 bp a leg) AND with 2× costs |
| G6 | beats the equal-weight market in EVERY bear episode covered (Q4 2018, COVID, 2022, Feb-Apr 2025) AND mean return in down months > 0 |
| G7 | \|T1 beta\| ≤ 0.3 (market-neutral designs only) |
| G8 | T1 alpha without its five biggest contributors > 0 |

"Works" = all applicable gates. "Works to some degree" = G1, G2, G3, G5 and G6.

## Lab v2 (2026-10-23): what the first round's adversarial reviews found

Every first-round number (`var/proposals_lab/results/`) was produced by lab v1. The six reviews
found five lab-level defects; lab v2 fixes them and writes to `results2/` (`LAB_VERSION = 2`,
in every result's `meta`):

1. **Holiday rows.** The panels' calendar was the union of 2,215 names, and Swiss listings
   (ALC, CFR) trade on US holidays: the 2026 panels had three rows (06-19, 07-03, 09-07) on
   which no US name was priced, and books that read a NaN price as "ineligible" exited
   everything and rebuilt the next day (adaptive_hedge's "2026 factor crash" was this), and a
   NaN market return switched peer_reversal's 21-bar volatility rule off for 21 bars after each
   row. v2 keeps NYSE sessions only (the bars SPY has). [Corrected 2026-10-24: an earlier
   version said the reference short-interest book's 2026-H1 read −7.0% instead of −1.3%
   because of these rows. The −1.3% was a different harness's number (informed_flow_ls_p1's
   own walk-forward, skew sleeve active), so it cannot isolate the rows. The lab v2 record
   of that half-year, with every v2 fix applied, is −11.6%.]
2. **Day 0.** `pct_change().fillna(0)` dropped each half-year's first day — the opening
   orders' costs, the first open-to-close move and the borrow the engine charges UP FRONT for
   shorts held to the window's end (`backtest._borrow_fees`). v2 counts day 0 from the initial
   cash, for the book and the benchmark. This charges one full book rebuild per period, which a
   continuous book would not pay: conservative by ~0.1-0.15 pt of alpha a year at base costs,
   ~0.5 under the late-fill stress.
3. **Starved shorts.** The engine runs a bar's orders with `call_seq="auto"`; under `lock_cash`
   a NEW short needs free cash, and auto runs it with the sells, BEFORE the short covers (buys)
   that would release locked cash — on a nearly full book vectorbt rejects or part-fills it, and
   since targets are emitted only on change it is never re-sent (resid_momentum: 840
   position-days off target in E0). v2 runs portfolio books through its own copy of the engine's
   runner (`lab.run_portfolio`) with an explicit order per bar: long cuts, short covers, new
   shorts, long buys. The shared engine (`src/`) is untouched.
4. **Borrow on hair re-trims.** The engine charges accrued borrow as a fee fraction
   `accrued × |w| / |x − w|` of the order that buys a short back; a hair-sized re-trim makes it
   explode (etf_trend I1: borrow over-charged 4-5×). `lab.borrow_fees` defers the charge of a
   re-trim smaller than a quarter of the short to its next real buy-back.
5. **ETFs in the stock universes.** XLE, XLF, XLI, XLK, XLV, XLY are stored as "equity" (since
   2023-10, in the top 500 by dollar volume): excluded (`lab.EXCLUDE`).

Since 2026-10-24 the shared engine does fixes 3 and 4 itself. It also keeps unfilled orders working, a
fix the lab's runner lacks. The lab keeps its own runner so that its recorded results reproduce.

Known and NOT fixed (caveats): the $5 floor reads split-adjusted closes (0-2 liquid future
splitters a period, e.g. NVDA 2016, are kept out); `sector` is today's classification; yfinance
EPS are on today's share basis, rounded to the cent (a name that split later has tiny, coarse
EPS early: a strategy reading `earn_surprise` must neutralize near-zero estimates); the DB holds
survivors only. The shared DB's SPY and sector SPDRs were re-written by etf_trend's yfinance
backfill (it removed a +3.43% splice on SPY, 2023-10-17); the v2 panels read the current DB.

## Running on a shared machine

At most two of your own Python processes at a time; check `memory_pressure | tail -1` (≥ 20%
free) before starting a heavy one; Optuna sequential. Never touch another strategy's folder,
`informed_flow_ls` (under review), `informed_flow_ls_p1`, or another agent's results. No IBKR.
Read panels through `lab.panel`, not the DB. Write progress into your `RESEARCH_LOG.md` as you
go, so the work can be resumed.
