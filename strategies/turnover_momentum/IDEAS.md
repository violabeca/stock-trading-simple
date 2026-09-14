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

### `fund_shares` needs a plausibility filter — ENGINE PROPOSAL

`dei:EntityCommonStockSharesOutstanding` carries filer unit errors (AJG ×1e6 for two
quarters, BALL a literal 0, RMD/CMG ÷1,000, CCL ×1,000; 124 non-positive rows across 87
symbols). A feature-layer rule — drop a share count that is ≤ 0 or differs from both
neighbouring filings by ≥ 100× while they agree with each other — would fix this strategy's
turnover, `value_factor`'s and `conviction_flow`'s market cap in one place. Splits (2–20×)
must survive the rule. Proposal-only per repo rules.
