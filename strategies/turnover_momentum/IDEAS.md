# turnover_momentum — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why). Give each idea a *why* and, where possible,
the falsification that would settle it — "try X" without "X wins if / dies if" is
not an idea, it is a mood.

## Signal construction

### The low-turnover *reversal* leg

The paper's result is a double sort: high-turnover names continue, low-turnover names
reverse. This strategy only ever takes the top-turnover slice (`turnover_rank >= turnover_q`),
so "reversal among the *least* traded" is not expressible. A `turnover_side: high|low` knob
would let the sweep test the full 2×2 (side × polarity). *Wins if* `low + reversal` is the
second-best cell and `low + momentum` the worst — the paper's pattern; *dies if* the sign does
not flip with turnover on this universe, which would say the conditioning is a liquidity
filter here, not the mechanism.

### Industry-relative momentum

Blitz et al.'s composite uses one-month *industry* momentum and *industry-relative* reversal;
a raw one-month return on 500 names is partly a sector bet (2020–2024: energy, then AI). The
`securities.sector` column is in the database, so sector-demeaned `ret` is a small change.
*Wins if* the demeaned IC in §4 is higher with a smaller sector tilt in the book.

### Blend into informed_flow's composite

The two strategies share nothing (price/volume here, filings and analysts there), which is
exactly when a rank-combine helps. informed_flow already rank-combines sleeves with a
`_MIN_COVERAGE` guard; a `w_stmom` sleeve fed by this module's `ret` on the high-turnover
slice is ~10 lines there. Test with informed_flow's §10b·ii `0.0` ablation column, never on
the holdout.

### Volatility scaling / momentum-crash guard

Nothing caps the factor. A trailing-volatility scaler on the formation return (rank on
return ÷ realized vol) is the standard fix and a one-line change to `_compose`. *Wins if*
2022 and Q1-2023 improve in §10c without a lower full-window `excess_cagr`.

## Universe & data

### A trailing point-in-time universe

`Liquid500` is ranked once, as of 2020-06-30. Re-ranking every year on the prior half-year
would let membership follow liquidity through the window instead of freezing it at the start
(also open in `gap_and_go/IDEAS.md`). Needs the engine to accept a time-varying symbol mask
or a per-year panel — proposal territory.

### Share count in volume units — ENGINE PROPOSAL

`_in_volume_units` re-derives the split restatement from jumps in `fund_shares` inside this
strategy. A `fund_shares_adj` feature (or an adjustment-factor column on the daily bars) in
`features.py` / `data.py` would make that one place, fix `value_factor`'s market-cap caveat at
the same time, and close the after-window-split residual by using the backfill date's units
explicitly. Proposal-only per repo rules.

### Anchor the rebalance grid to the end of warm-up

`_select` re-picks on `iloc[::rebalance_days]` from bar 0, so the first two months hold
nothing (warm-up) and the cadence is offset by an accident of the window start. Anchoring
the grid to the first bar where `score` has ≥ `max_positions` names is a two-line change;
the same item is open in `value_factor/IDEAS.md`.

## Known caveats (documented, not actionable here)

- The database retains no delisted names, so `Liquid500` is survivor-only; the
  benchmark-relative objective partially controls for it, absolute numbers do not.
- 20-F filers' share counts are annual and lag; below `min_coverage` breadth the whole
  strategy silently becomes a dollar-volume-conditioned one (§4 prints the source).

## After the 2026-09-12 verdict (see README "Result")

### Replicate the paper's construction before burying the thesis for good

The failed gate used the repo's daily-overlap IC. Medhat & Schmeling form and hold at
month-ends with turnover averaged over the prior year, on all US common stocks. A cheap
month-end replication on the broad pool (formation = the calendar month, hold = the next,
turnover decile across the whole pool) is the only version that could reopen this. *Wins if*
the top-decile month-end spread is positive with a real t; *dies if* it matches the daily
result — then the effect simply is not in 2015–2024 US liquid names.

### The reversal leg is the finding, and it is fading

Top-turnover names carry a one-month reversal IC of +0.02 to +0.05 (|t| < 1.6) in every
population tested, strongest in 2021 and gone in 2023–24. `polarity: reversal` is already
expressible; the reason not to trade it is the same one that killed the overnight anomaly
here — monthly rotation of ~half the book against a 1–2% gross spread. A pre-registered test
of the reversal leg on the 2025+ holdout would at least say whether it survived 2023–24's
fade; run it once, do not tune on it.

### Fundamentals need repairing at read time — ENGINE PROPOSAL (refined 2026-09-14)

Measured on the whole `fundamentals` table, as the feature builder reads it: 181 thousand-fold
unit jumps across 80 symbols, confirmed as filed in SEC's own `companyfacts` (AJG reported
189,621,000,000,000 shares for 2020-Q1); 119 zero share counts and 97 shell stubs of 1–9,999
shares. The bigger problem is **frozen series**: `providers/sec._first_tagged` locks a concept
onto the first XBRL tag that ever had data, so the value stops updating when a filer abandons
that tag. At 2024-12-31, 26 LiquidLeaders ∪ Liquid500 members carry a share count 4–14 years
old (V, MA, CMCSA, ACN, NKE, SCHW, F, REGN …: multi-class issuers stop filing a
non-dimensional cover-page count) and 29 have none (META, CRWD …). The same lock-in freezes
equity (AVGO, QCOM), net income (PH) and gross profit (see `../value_factor/IDEAS.md`).

The fix, in three parts:

1. **`features.py`, share-count units, point-in-time.** Walk each symbol's filings in `known_at`
   order: drop counts below 10,000; a count within 2× of a 1000^k multiple of the last accepted
   count is divided back; new units are accepted only once that run has persisted more than 365
   days. It reads past filings only, so a train-only load and a full load agree. All 26 repairs
   it makes on a universe member since 2018 check out; the 365-day acceptance fires only on nine
   symbols whose very first filing was wrong (2011–2013), and the longest real error run (AAP,
   2011–12) stayed under a year.
2. **`providers/sec.py`, per-period tag fallback.** For each period, the first tag that reports
   it wins; weighted-average basic shares (`us-gaap` / `ifrs-full`) is the last share-count
   resort, and gross profit falls back to revenue − COGS period by period. SEC data shows this
   restores a current share count for 43 of the 54 stale or missing members (not V, F, STZ, TSN,
   PLNT, BP, BIDU, IBN, PSKY, SNDK, XOM). Re-ingest with `REFRESH = True` in
   `notebooks/03_altdata_update.ipynb`; the do-nothing upsert only adds rows.
3. **`features.py`, an 800-day max age on `fund_*` levels.** Domestic filing gaps are 134 days
   at p99, foreign ones 729; the cap removes only genuinely frozen values that no tag can restore
   (V). Scoped to `fund_*` so informed_flow's short-interest sleeve stays bit-identical. It must
   ship with part 2, or value_factor's gross-profit sleeve drops under its coverage floor.

Rejected: a two-sided "differs from both neighbouring filings" rule — it needs the next filing,
so a train-only load and a full load would disagree about the last train-window filings (the
same leak class as loading full history and slicing back). No thousand-fold repair on assets or
equity either: de-SPACs genuinely grow ~1000× (ACHR, BKSY, CIFR, FUBO, GBTG). Never edit stored
rows: the table is a point-in-time archive in a shared database.

This strategy's verdict does not depend on any of it: with parts 1 and 3 applied, the
top-tercile 21d IC moves from −0.023 to −0.019 and the gate still fails.
