# quality_junk — quality minus junk, long/short, beta-neutral

Proposal 4 of `STRATEGY_OVERVIEW.md` §6, built and judged in `strategies/proposals_lab` (round 1
2026-10-22; review fixes and the lab v2 re-evaluation 2026-10-23). The numbers, the decisions and
the verdict are in `RESEARCH_LOG.md`: §13 is round 2 on lab v2, which supersedes every round-1
number; this page explains the rules and quotes lab v2.

> **Verdict (lab v2): no edge — a low-beta hedge, not an alpha source.** The pre-registered
> default (U0, what `config.yaml` ships) earns alpha +0.1% a year on TRAIN (2016-21, P 0.59),
> −0.4% on the top 500 (2022-26, P 0.51: the median of 100 shuffled-name books) and −0.05% on
> names 501-1000. Re-tuned on lab v2 (133 distinct trials on TRAIN, PBO 0.69), the pick — 15
> slots a side, a 189-bar volatility, a wider exit band (0.75), 3× beta sizing — beat U0 on both
> validation sets by margins inside the noise (+1.0 and +0.1 point) and so became the final
> candidate, **TRv2**, under the protocol's tuned-values rule. It earns +3.5% on TRAIN (where it
> was tuned), +0.6% on the top 500 (P 0.56, 66th percentile of shuffled books) and +0.0% on names
> 501-1000, and it passes four of the lab's eight gates (TRAIN alpha, names 501-1000 alpha > 0
> by 0.02 point, the bear gate, |beta| ≤ 0.3) but not alpha significance, luck, late fills or
> the top-5-out check. On the locked names 1001-1500, each read once on lab v2: U0 +1.7% (P 0.68),
> TRv2 +0.6% (P 0.55), 1.1 points below U0. With fills 50 bp late both lose on the top 500 (U0
> −1.3%, TRv2 −0.6%). Both beat the market in all four falling episodes, but so does any low-beta
> book here (both beta-neutral variants do); their mean in the market's down months (+0.9%, +0.7%)
> is largely net short beta (two-thirds and half of it), the beta-neutral I1 fails that gate
> (−0.2%); beyond beta the books gained in Q4 2018 (+6.5 and +6.2 points; TRv2 also in Feb-Apr
> 2025, +4.8) and lost in the COVID crash and the 2022 bear. Round 1's tuned pick, its four
> iterations and two iterations on TRv2 (sector-neutral safety, BAB side neutrality) all failed
> the adoption bar on lab v2. Round 1 found the market residuals negatively correlated with the
> short-interest book's (−0.36 on 2022-26, lab v1, not re-measured): a possible diversifier, a
> cross-strategy question.

## Thesis

Long profitable, low-volatility companies; short unprofitable, volatile "junk".

- Investors overpay for lottery-like stocks, and leverage-constrained investors bid up risky ones
  instead of levering safe ones (Frazzini & Pedersen 2014, betting against beta; Asness, Frazzini
  & Pedersen 2019, quality minus junk).
- Gross profitability predicts returns as strongly as value does (Novy-Marx 2013).
- In a crisis money leaves junk first, so the factor has earned most in falling markets.

## Rules (`strategy.py`)

1. **Profitability.** Gross profit TTM ÷ total assets; where gross profit is missing net income
   TTM ÷ equity (equity > 0). About a third of the names are on the ROE fallback (banks, insurers
   and filers that never tag gross profit), mixed inside most sectors, and the two ratios live on
   different scales (per-period medians 0.26-0.32 and 0.10-0.15), so a name is ranked against the
   names of its sector *measured the same way* (a group under 5 names: against every name on its
   measure). Both ratios are two fundamentals, so no price or share count enters: no share-count
   unit slips, no ADR ratios.
2. **Safety.** Minus the 252-bar volatility of daily returns, ranked across all eligible names
   (`safety: beta` ranks minus beta instead; `safe_by_sector` ranks within sector too). An
   optional third sleeve, profitability growth (`w_growth`, the 252-bar change of the name's own
   ratio), is off: its TRAIN premise had the wrong sign.
3. **Score.** The weighted mean of the two ranks, re-ranked across the eligible names (close ≥ $5,
   a profitability ratio, a volatility).
4. **Book.** Only at month ends (`rebalance`: decided at the last session's close, filled at the
   next open; the panel's last bar counts when the next NYSE session is in another month, so the
   session before Good Friday or a May-31 Memorial Day is a month end): longs leave below rank
   `q_out` 0.80, shorts above 0.20; then free slots fill, best first, from ranks ≥ `q_in` 0.90
   (long) and ≤ 0.10 (short); 20 slots a side. Between month ends the book is held; two risk exits
   act daily on the short side: the squeeze stop (+100% from the entry close) and the crowded exit
   (short interest > 30% of shares outstanding). U0's own rotation is 30-42 orders a year
   (TRAIN 30, top 500 35, names 501-1000 42; a regular month-end rebalance trades 6-9% of the
   book). The lab adds 81-85 orders a year on top by rebuilding each half-year's book on its first
   day; one continuous book would place about 54 / 74 / 105 a year. (TRv2: 15 slots a side, exit
   below 0.75; 17-33 rotation orders a year plus 61-63 first-bar ones.)
5. **Sizing.** `informed_flow_ls_p1`'s beta sizing: a slot is scaled at entry by median beta ÷ the
   name's 126-bar beta (between 1/1.5 and 1.5 of the equal slot), frozen while held, with new
   entries scaled down so nominal gross is at most 0.98 AT ENTRY (positions then drift: realized
   gross averages 0.99 and reached 1.31). The low-beta longs are bigger (0.59-0.66 of the book)
   and the high-beta shorts smaller (0.34-0.41) — but not enough: U0 still runs a net short beta
   (−0.33 realized on the top 500, 2022-26). TRv2 sizes harder (`max_size` 3.0 on a 252-bar
   beta) and runs −0.20. `sizing: legs` (tested, not adopted) is the BAB-style version: at each
   month end the sides get dollars with D_L β̄_L = D_S β̄_S (0.63-0.77 long, 0.19-0.33 short), a
   held name re-sized only when its slot moved more than 25%; a slot is D / slots, so the balance
   is exact only when both sides hold all their names.
6. **Borrow.** 0.5% a year, 3% above 10% of shares short, 10% above 20%; 0.5% without short-interest
   data (before 2018).

Portfolio mode: `signals()` returns `weights`, `borrow` and `retrim_k` (2: a long that doubles its
slot is trimmed back; shorts are never re-trimmed).

## Files

- `strategy.py`, `config.yaml` — the strategy (every config param is a `signals()` keyword).
- `research.py` — not loaded by the strategy: the data-coverage audit, the TRAIN premise check, the
  leg anatomy, and the review re-reads: `boundary` (each half-year's first day charged as one
  continuous book would pay it, and a count of new shorts the engine's runner left short of target),
  `v1v2` (where a label's numbers moved between lab v1 and v2), `bear` (falling markets split into
  beta and the rest), `staleness` (frozen gross-profit levels), `floor` (names the $5 floors
  remove). `show.py` (lab summaries, stitched paired Δ) and `correlate.py` (correlation with, and
  a 50/50 blend with, another lab book: round 1 only) read saved lab results.
- `RESEARCH_LOG.md` — pre-registration, every run, every decision, the verdict, the review fixes,
  round 2.
- `IDEAS.md` — parked ideas.
- `tests/test_quality_junk.py` — synthetic panels: causality, ranking, rebalancing (incl. exchange
  holidays), sizing, stops, borrow, the parameter check.

## Running it

```bash
uv run pytest -q tests/test_quality_junk.py
uv run python strategies/proposals_lab/lab.py run quality_junk U0v2 '{}' E0,TB,T3,T1,T2 --costs2
uv run python strategies/proposals_lab/lab.py run quality_junk TRv2 '{"w_prof": 1.0, "vol_window": 189,
  "q_in": 0.9, "q_out": 0.75, "slots": 15, "rebalance": "month", "beta_window": 252,
  "max_size": 3.0}' E0,TB,T3,T1,T2 --costs2                          # the round-2 final
uv run python strategies/proposals_lab/lab.py verdict quality_junk U0v2
uv run python strategies/proposals_lab/lab.py report quality_junk U0v2 <label>
uv run python strategies/quality_junk/show.py delta U0v2 <label>      # the adoption bar's numbers
uv run python strategies/quality_junk/research.py premise '{}' E0,TB,T3
uv run python strategies/quality_junk/research.py boundary U0v2        # OPEN tests only
```

Lab v2 writes `var/proposals_lab/results2/quality_junk/`; round 1's lab-v1 labels (`U0`, `T`,
`I1`-`I4`) stay in `var/proposals_lab/results/quality_junk/` as the record.

## Caveats

- **Each half-year starts from cash.** Lab v2 counts every period's first day from the initial
  cash, so each half-year pays a full 40-name rebuild (and the benchmark a 500-name one) plus the
  borrow charged up front: conservative. One continuous book would pay only what changes at the
  boundary: on the top 500 that reading (`research.py boundary`) is 0.2 point of alpha a year
  better at base costs for U0 (−0.18% vs −0.38%) and 0.8 better under late fills (−0.57% vs
  −1.33%), and U0's gates read the same. For TRv2 the late-fill gate would pass narrowly on that
  reading (+0.05% to +0.46% instead of −0.62%), which still leaves alpha significance and the
  top-5-out check failing.
- The database holds survivors only. Junk that went bankrupt is missing, so the short side looks
  worse than it was; names that were bought as quality and later failed are missing too, so the
  long side looks better than it was. The two biases pull in opposite directions; the net effect
  on this book is unknown.
- Before 2018 there is no short-interest data: every short borrows at 0.5% a year there.
- Point in time, not quite: sector codes are today's, applied to every year, and the $5 floors (the
  lab's universe filter and `min_price`) read closes adjusted back from today for later splits and
  dividends (0-2 names a month-end on the top 500, e.g. PBR, NVDA in 2018-19; the lab's universe
  left NVDA out of the top 500 in six of eight 2016-20 half-years).
- Fundamentals are the screened, point-in-time SEC XBRL levels of `features.py` (a level no filing
  has refreshed for 800 days is masked). A level can still sit unchanged for long: on month ends,
  3-9% of the names measured on gross profit carry a GP TTM level unchanged for more than 200
  days (up to ~3 years), most of them with a fresh ROE, and U0's book holds 0.75-3.8 of them out
  of 40. Falling back to ROE for them did not improve the TRAIN premise (IC +0.023 vs +0.026), so
  the strategy keeps them.
