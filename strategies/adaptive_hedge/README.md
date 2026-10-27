# adaptive_hedge

Proposal 3 of `STRATEGY_OVERVIEW.md` §6 combines **informed_flow's long short-interest book** with
**trend_following's breakdown shorts** in one book of 30 equal slots. A short start takes a free
slot. When no slot is free, it takes the slot of the weakest-ranked long, so the short side (and
with it the hedge) should grow only when many stocks break down.

This is a research folder for the shared lab (`strategies/proposals_lab`). `RESEARCH_LOG.md` holds
the full record, with every number.

> **Status (round 2 and its repair, lab v2, 2026-10-23): works to some degree.** This is a
> deliberate downgrade: by the lab's own rule it "works", because every applicable gate passes
> (G1-G6 and G8; G7 is skipped). It is downgraded for four reasons:
>
> - The proposal's mechanism failed.
> - The alpha shrinks out of sample: T4 is +4.2% (P 0.84).
> - The CAGR is below buy-and-hold and SPY on T2, on T4 and over 2018-26.
> - The proposal's own bars P1 and P2 fail.
>
> What remains is a crisis-alpha book: a near-constant net short of persistent losers plus
> informed_flow's long sleeve. Every round-1 decision stands on lab v2, and most numbers are
> smaller (RESEARCH_LOG.md §13; the round-2 repair is §14).

> **TL, read once on 2026-10-24: TUNEDv2 fails on the smallest names (names 1501-2000).** The
> orchestrator disclosed this book as a component beside the cross-strategy finals
> (`strategies/proposals_lab/FINAL_DECLARATION.md`).
>
> - TL alpha −4.82% (P 0.205), CAGR −11.1% against buy-and-hold's +19.4%, max drawdown −57%.
> - `reconcile.py`'s `label_run` was called directly, since the CLI refuses locked tests. Two
>   covers were refused at zero free cash: BKKT on 2024-11-19 (stuck 29 bars, +3.41% of the
>   book) and SGML on 2025-12-22 (7 bars). With their stuck P&L removed TL is −5.49% (P 0.18):
>   the failure is real.
> - The same logs show 2 refused long buys and 111 part-filled entries (71 long buys, 40 new
>   shorts) across T4 and TL. These are cash-limited fills at gross 0.98, which left about 2% of the
>   book (long and short together) unheld on average.
> - The same batch re-ran T4: +4.16% (P 0.84), the number already reported above.

## The rules (U0, the pre-registered default)

- **Longs** (informed_flow v41):
  - Rank: minus days-to-cover averaged over 42 bars, ranked inside 5 dollar-volume groups.
  - Enter at rank >= 0.90, leave below 0.80, `max_hold` 126.
  - Eligible: a close of at least $5, a days-to-cover and a share count (`fund_shares`).
- **Shorts** (trend_following v7):
  - Entry: a close below the prior 9-bar low, with ADX(14) >= 34 and -DI > +DI, a close of at
    least $5, and short interest <= 30% of shares.
  - Exit: a close above the chandelier (the lowest low since entry + 14 x ATR(37), ratcheted),
    an opposite start, or crowding.
  - Competing starts: the most negative 12-1 month relative strength goes first. At most 20
    shorts.
- **Book**:
  - 30 slots of 0.98/30, portfolio mode, `fill: open`, `order_notional: 1000`, `retrim_k` 2.
  - Borrow tiers: 0.5 / 3 / 10% a year by short interest / shares; 0.5% without data.
  - A held name with no price on a bar is carried, not sold, and cannot be displaced. A name
    whose data ends is carried to the period end (IDEAS.md 1). A row on which almost no name is
    priced is skipped. A target change waits for the name's next priced open.
- **Knobs off in U0.** Each was tested as an iteration and rejected, except `hedge` and
  `momentum`, which were never run:
  - `short_hold`: a one-month clock since the last fresh start;
  - `exit_window`: cover on a close above the prior N-bar high;
  - `short_stop`: the +50% squeeze stop;
  - `short_filter`: `crowded` (variant B) or `momentum`;
  - `hedge`: an index fund holds every short slot instead of the single names.

## Result on lab v2 (RESEARCH_LOG.md §13)

**Final candidate = TUNED.** It is U0 + `{max_shorts: 20, adx_min: 40, entry_window: 27, atr_mult:
16, atr_window: 48}`, tuned on TB + T3 (120 trials, PBO 0.10) and re-measured on lab v2 as
`TUNEDv2`. `config.yaml` stays at U0, so the lab's saved labels keep their meaning.

| test | alpha (P) | beta | CAGR | buy & hold | SPY | max DD (B&H) |
|---|---|---|---|---|---|---|
| TB 2018-07 → 2020-06 (train) | +1.9% (0.66) | −0.24 | −0.7% | +11.2% | +8.8% | −18.7% (−37.4%) |
| T3 2020-07 → 2021-12 (train) | +5.2% (0.71) | +0.00 | +5.2% | +35.2% | +35.3% | −16.5% (−7.8%) |
| T1 top 500, 2022-26 | +11.4% (0.985) | −0.28 | +9.0% | +8.8% | +11.9% | −17.9% (−25.0%) |
| T2 names 501-1000 | +4.3% (0.77) | −0.26 | +2.3% | +7.6% | +11.9% | −26.4% (−24.0%) |
| T4 names 1001-1500 (locked) | +4.2% (0.84) | −0.35 | +0.6% | +10.2% | +11.9% | −24.7% (−26.1%) |

T4 was read in round 1 and read once more on lab v2, as a disclosed re-measurement that selects
nothing.

TB, and with it the TRAIN era, includes one cover that lab v2's runner refused: Wayfair in
2020-H1 (see the caveats). `reconcile.py` estimates the effect by removing every stuck
position's P&L:
- TB: +5.3% (P 0.82), against +1.9% as run;
- TRAIN era: +7.6% (P 0.88), against +5.7%;
- T1: +11.1%; T2: +4.1%.
These are estimates. The table is the lab's record.

- **Gates.** Every applicable lab gate passes (G7 is skipped: the book is not market-neutral).
  The TRAIN era is +5.7%, late fills give T1 +9.4%, 2x costs +10.9%, and T1 without its top 5
  names +5.9%. T1 beats all 100 shuffled-name books.
- **Bear episodes.** Every one is won: Q4 2018 −1.8% vs −18.2%, COVID +20.7% vs −37.0%, 2022
  +25.9% vs −24.5%, Feb-Apr 2025 +0.3% vs −19.5%. In down months the book averages +2.7%
  against the market's −4.1%.
- **It is not the adaptive hedge the proposal describes.**
  - The short side stays ~full in every regime: trend exits are rare, and shorts are held about
    300 days.
  - Every exit that would make it adaptive cost alpha on T1 and T3: a one-month clock, a Turtle
    exit, the +50% squeeze stop. The crowded-half filter (variant B) cost too.
  - The chandelier is what keeps the short side fresh: it moves slots out of names that
    recovered and into new breakdowns. Without it, the slots hold names shorted years earlier,
    and the 2022 bear falls from +25.9% to +5.0%.
- **What the book really is:**
  - About 34% long and 61% short of the book on T1 (beta −0.28).
  - Most of its alpha is the short leg's selection: +9.0 points a year on T1 over an
    equal-beta index short, against +5.2 from the long sleeve. On T2 it is the other way:
    +1.2 from the short leg, +6.2 from the long sleeve.
  - It loses 4-37 points to the equal-weight market in 2019, 2020, 2021, 2023, 2024 and 2026, and
    wins only 2018-H2, 2022 and 2025.
  - From 2018-07 to 2026-10 it compounds +5.9% a year, against +13.8% for the market and +15.0%
    for SPY. With the refused cover's P&L removed it would compound about +6.6%.
- **Out of sample the alpha shrinks with liquidity rank:** T1 +11.4, T2 +4.3, T4 +4.2.
  - The tune's edge over U0 is +4.7 on T1, +4.1 on T2 and +1.2 on T4 (P 0.61).
  - On TB the edge is −0.5 as run. That figure comes from the refused cover: with the stuck P&L
    removed, it is +2.9 (P 0.70).
- **Caveats:**
  - The overnight gap into each half-year is counted nowhere; counting it moves T1 to +10.7.
  - 6% of T1's short exposure has no short-interest or share-count data, so it borrows at GC
    and escapes the 30% crowding cap. At a 10% fee that would cost at most 0.34 points.
  - With every low-tier short at 3% borrow, T1 is +9.9% and T2 +2.9%.
  - **A single squeezed short can cost the book about 7% before its chandelier fires.**
    - Wayfair in 2020-H1 rose 2.9x between the short's fill and its chandelier stop.
    - On lab v2 the runner then refused its cover: free cash was 0 under `lock_cash`, and the
      order was never sent again. That doubled the loss to 13.3% of the book.
    - `reconcile.py` checks every order against the positions held.
    - The fix, re-sending refused targets, belongs to the lab and is a proposal (IDEAS.md 2).
  - **A name whose data ends is carried to the period end.** CWAN holds a short slot that the
    book never fills in T2 2026-H2. That is worth about 0.1 point a year on T2 (IDEAS.md 1).

**What changed from round 1.** Lab v1 dropped each period's day 0 and part-filled new shorts.
Those defects had flattered the book: in round 1, T1 read +12.9%, T2 +9.4% and the TRAIN era
+9.1%.
- TB's fall from +7.5% to +1.9% is one short, Wayfair in 2020-H1, for two reasons:
  - v1 filled only 31% of the short, so it cost −2.2%;
  - v2 filled it in full and then could not cover it, so it cost −13.3%. About 7% of that came
    before the chandelier fired.
- The 2026-H2 "factor crash" of round 1 was a US-holiday-row artifact.

## Files

| file | what |
|---|---|
| `strategy.py` | `signals()`, `valid()`, `compose()` (every intermediate) and `book()` (held frames, exits) |
| `config.yaml` | the U0 values; `search:` = the pre-registered tune space |
| `premise.py` | the TRAIN-only premise checks (long IC, short-start event study, breadth) |
| `readout.py` | reads saved lab results: per-test table, SPY, 2022, stitched-TRAIN paired delta |
| `legs.py` | re-runs a saved label's book and splits its daily P&L into the long and short legs |
| `reconcile.py` | re-runs a saved label's book with vectorbt's order logs: refused, part-filled and dropped orders, stuck positions, and the alpha with the stuck P&L removed |
| `RESEARCH_LOG.md` | pre-registration, every run, the verdict; round 2 = §13, its repair = §14 |
| `IDEAS.md` | parked: free a slot whose data ended; the lab-level re-send of refused targets |

Run everything through the lab, from the repo root. For example:
`uv run python strategies/proposals_lab/lab.py run adaptive_hedge U0v2 '{}' TB,T3,T1,T2 --costs2`

- Name the tests explicitly. The CLI reads its fifth argument as the test list, so a bare
  `--costs2` in that place is taken for a test name.
- E0 is never run: there is no short interest before 2018, so the long side is empty there.
- The round-2 scratch scripts (`scan_v2`, `rescore_v2`, `seams_v2`, `starved`, `unknown_si`) and
  their logs sit in `var/proposals_lab/results2/adaptive_hedge/`. The repair's logs (the
  reconciliation and the mutation check) sit in its `repair/` folder.
- A reconciliation, for example:
  `uv run python strategies/adaptive_hedge/reconcile.py TUNEDv2 TB,T3,T1,T2 --base U0v2`.
  It refuses the locked tests.
