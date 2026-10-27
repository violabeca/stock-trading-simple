# etf_trend — pending ideas / proposals

Parking lot (the repo's per-strategy convention). Each idea names what would settle it. Ideas
that were run are in `RESEARCH_LOG.md`, not here. Numbers are lab v2 (round 2).

## Open questions

- **A crisis sleeve beside an equity book.** U0v2 runs at a third of SPY's volatility (corr
  0.80-0.83, max DD −12.9% vs −55%), and I1v2 is the 2008/2022-type hedge (corr 0.25 with SPY in
  2022-26; 2008-09 +31.4%, 2022 +1.3%). Neither adds return in a bull market. Settles it: the
  blend run through the lab as one book (one wallet, real costs), never a sum of return
  streams. It earns its place only if the blend's VALID alpha AND return ÷ volatility beat the
  equity book alone. Dies if the drag in 2016-20 (I1v2 −4.4%/yr alpha) outweighs the bear
  episodes.
- **The turn of the month.** In TRAIN every later decision day lost to the month's first
  trading day (6th / 11th / 16th: −0.61 / −0.56 / −0.75 pts, sd 0.10). VALID showed only noise
  (+0.28 / −0.44 / +0.35). Every window has now been read, so this can only be settled on data
  this book has not seen: a new asset list or a later year. It wins only if deciding on the
  first trading day beats the later days there by more than 2 × the VALID offset sd (0.87 pts).

## Engine proposal (proposal-only: `src/` is not changed by a strategy)

- **`backtest._borrow_fees` (src/trading_simple/backtest.py, lines 662-710): a hair-sized short
  re-trim blows up the borrow charge.**
  - Problem: the accrued borrow is charged as a fee FRACTION `accrued × |w| / (x − w)` of the
    order that buys a short back. When a short is re-trimmed by a hair, `x − w` is tiny and the
    fraction explodes. In round 1's I1 it reached ~375× the accrual (a 0.543 fee rate on a $78.8
    LQD order on 2022-05-03). vectorbt then applies it to the real target-percent order, whose
    size, and even direction, comes from drift, not from `x − w`. Round 1 charged I1 0.44 /
    0.81 %/yr of the book (TRAIN / VALID) against 0.12 / 0.16 owed.
  - Fix, either one:
    - Defer the charge of a re-trim smaller than a quarter of the short to its next real
      buy-back, as `lab.borrow_fees` does (`strategies/proposals_lab/lab.py`, lab v2).
    - Charge the accrued borrow in dollars on the actual buy-back (vectorbt `fixed_fees`), or
      compute it from the realized short value inside the run, instead of as a fraction derived
      from target deltas.
  - Settles it:
    - A synthetic book that re-trims a short by 0.1% every month: the borrow charged must equal
      rate × short value × calendar days, to within 1%.
    - etf_trend's I1v2 through the engine runner must give the lab's 0.12 / 0.16 %/yr.
  - Who it affects: this folder no longer depends on it. Its own fix (kept weights bit for bit,
    plus `min_change`) leaves no hairs, and I1v2's borrow charged equals the borrow owed. Any
    other portfolio book that re-trims shorts by small amounts outside the lab is still
    exposed.
  - Status: written up here in the round 2 repair, pass 2 (2026-10-23). It still has to be
    filed in the repo's engine-proposal ledger; not implemented.

## Data-path proposal (outside this folder)

- **A splice check before a provider refresh of an ETF's daily bars.**
  - Problem: IBKR daily rows carry adj_close = close. Joined to yfinance-adjusted history, they
    made SPY's stored series jump +3.43% on 2023-10-17, against a real move of −0.005%. Round
    1's yfinance backfill replaced those rows (README "Data"). A later IBKR daily refresh of
    SPY, GLD, SLV or the six SPDRs can splice them back.
  - Fix: before upserting one provider's bars onto another's history, compare adj_close / close
    on the overlap and refuse or flag a step.
  - Settles it: the check must flag the old 2023-10-17 splice in
    `var/proposals_lab/results/etf_trend/db_backup_before_backfill.pkl` and pass the current
    rows.
