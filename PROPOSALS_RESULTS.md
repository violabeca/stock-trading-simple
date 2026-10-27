# Proposals 2-5, two "also considered" ideas, and five cross-strategies: results

*Status as of 2026-10-24. This follows `STRATEGY_OVERVIEW.md` §6, which proposed five strategies and, under
"also considered", named four more ideas: earnings-reaction drift, cross-asset ETF trend, options, and
anything intraday. Only the first two were built here. Proposal 1 (short-interest long/short made
beta-neutral) is reported in `strategies/informed_flow_ls/RESEARCH_LOG.md` §38P and in the overview. Every
number below comes from one harness, `strategies/proposals_lab` (lab v2), unless marked otherwise. Each
strategy's folder (README, RESEARCH_LOG and, except earnings_drift, IDEAS) holds its full record. Alphas
are per year: alpha = CAGR − beta × the CAGR of an equal-weight buy-and-hold of the same names. P is the
block-bootstrap probability that the alpha is above zero.*

*An adversarial check of an earlier draft against its sources found 54 errors or overstatements: six
verifiers checked ~700 claims, with a skeptic for each finding and a completeness critic. All are corrected
here. The critic's gaps were closed with the extra checks in §4 and with edits in §1, §7, §8, §9 and §12.
A second check of the rewrite (~810 claims) found 27 more imprecisions, also corrected.*

---

## 1. The short answer

**One declared final passed every test: `xs_si_mom_composite`, label `A_U0`.** It is a dollar-neutral
long/short book of 15 + 15 stocks. Each stock is ranked on the average of two signals: how crowded the short
sellers are (FINRA days-to-cover, the repo's one proven signal) and residual 12-1 momentum. Nothing in it is
tuned.

| test (2022-01 → 2026-10) | names | alpha (P) | with fills 50 bp worse |
|---|---|---|---|
| T1 | top 500 by dollar volume | **+12.9% (0.96)** | +10.8% |
| T2 | names 501-1000 | **+5.1% (0.84)**; +4.2% with its two stuck orders repaired | +3.0% |
| T4, locked | names 1001-1500 | **+12.4% (0.997)** | +10.1% |
| TL, locked; the pre-declared confirmation | names 1501-2000 | **+8.6% (0.963)** | **+6.5%**, the figure the declaration called realistic for these names |

- **It made money in both falls of 2022-26, on every name set.**
  - 2022 bear: +15.4%, +10.0%, +4.5% and +6.2% (T1, T2, T4, TL), while those markets lost 16-25%.
  - Feb-Apr 2025: +1.7%, +1.0%, +6.7% and +3.4%, while they lost 20-22%.
  - In the market's down months it averaged +0.8% to +1.6% against −3.8% to −4.3%.
  - Further back, on the top 500, it lost 7.0% in Q4 2018 (market −18.2%) and was flat in the COVID crash
    (−0.1% vs −37.0%).
- **Its protection depends on its beta at the time, and on the top 500 that beta is now net long.**
  - Stitched over each test the beta is about zero (−0.10 to +0.01).
  - Per half-year on the top 500 it swung from −0.40 (2023-H1) to +0.73 (2026-H1). It was net short through
    2022 and 2023, when it made its bear gains.
  - It has run net long since 2025-H2: the mean ex-ante beta was +0.65 in both 2026 halves. That is the
    configuration in which it lost 7.0% in Q4 2018.
  - On names 501-2000 the half-year beta stayed between −0.34 and +0.36.
  - Proposal 1's beta sizing, tried here post hoc, only cuts the drift by about two-fifths: 2026 ex-ante
    +0.40 instead of +0.65, at −3.0 points of T1 alpha (§4).
- **It beats random stock picking.** It beat all 100 books that trade the same rules on random names of
  the same liquidity:
  - on the top 500 (their median +0.9%, 95th percentile +5.2%);
  - on T4 (−0.0% and +5.3%);
  - on TL (−1.1% and +5.0%; the locked luck runs were added after the read and decide nothing).
- **The confirmation holds for the family, not only for the composite.**
  - All three controls also passed the same locked bar on both sets:
    - the short-interest book alone (C_si): TL +10.9%, P 0.99;
    - the short-interest and momentum books side by side (C_mix): TL +8.7%, P 0.99;
    - the momentum book alone (C_mom): TL +6.3%, P 0.89.
  - The composite's own edge over C_mix is clear only on names 1001-1500: +8.0 points, P 0.997.
- **Against buy-and-hold it won the falling year 2022 on every name set and lost every year the market rose
  20% or more; in between it was mixed.**
  - Calendar years against the equal-weight market:
    - On the top 500 it won 2018-H2, 2022, 2024 and 2025, and lost 2019-2021, 2023 and 2026 (§4).
    - On names 501-2000 it won only 2022 (and 2024 on T4), and lost 2023, 2025 and 2026 on all three sets.
  - Over the whole of 2022-26, against equal-weight buy-and-hold: it beat it on T1 (13.0% vs 8.8% a year)
    and T4 (11.5% vs 10.2%), and lost on T2 (4.4% vs 7.6%) and TL (6.5% vs 19.4%).
  - SPY (11.9% a year over the same days): it beat it only on T1.
  - 2018-07 → 2026-10 on the top 500: it compounded 8.9% a year against 13.7% for the equal-weight market
    and 15.0% for SPY.
  - It is a return stream with little market exposure, not a replacement for owning the market.
- **The edge is concentrated in 2022-2025.**
  - Every half-year from 2022-H2 to 2025-H2 was positive on both locked sets.
  - In 2026, the top 500 lost about 22% a year of alpha in both halves; the other sets were mixed.

**Everything else:**

| idea | folder | verdict |
|---|---|---|
| Proposal 2: crash-protected residual momentum | `resid_momentum` | **Fails G1** (TRAIN 2016-21: −1.4%), so it is neither "works" nor "works to some degree"; it passes the other seven gates. Its edge is a 2022-2025 regime: T1 +8.7%, T2 +5.5%, T4 +6.2%, with 2026-H2 negative on all three. Within TRAIN, TB (2018-07 → 2020-06) was +6.5%. A good component. |
| Proposal 3: adaptive hedge (short-interest longs + breakdown shorts) | `adaptive_hedge` | **Works to some degree on the open tests, then fails on the smallest names.** TRAIN +5.7%, T1 +11.4%; locked T4 +4.2% (P 0.84); locked TL −4.8%, max drawdown −57%. Its CAGR is far below buy-and-hold (5.9% vs 13.8% a year, 2018-26). |
| Proposal 4: quality minus junk | `quality_junk` | **No edge.** T1 +0.6% (P 0.56), T2 +0.0%. |
| Proposal 5: peer-relative short-term reversal | `peer_reversal` | **No edge.** On T1 its stock selection beat random names by about 3 points a year (suggestive, not established), half its ~6 points a year of costs. |
| Also considered: earnings-reaction drift | `earnings_drift` | **No edge.** T1 +2.2%, TRAIN −3.3%, T2 −0.8%, T4 −3.4%. |
| Also considered: cross-asset ETF trend | `etf_trend` | **No alpha; a crash protector.** Its alpha against SPY was +0.02% a year in 2016-26 (beta 0.28; it compounded 4.0% a year against SPY's 14.5%), but it made +4.6% in 2008-09 while SPY lost 54.6%. |
| Cross: three books in one account | `xs_three_book` | Its final (CRISIS: momentum + adaptive hedge) **passes all eight gates on the open tests, but not the design's own bar (2)**: the account does not beat its best single sleeve everywhere (on T2, momentum alone has the better alpha ÷ volatility). It **fails TL** (−0.03%). |
| Cross: short-interest longs, momentum shorts | `xs_cross_leg` | It passes every gate, but **its hypothesis is rejected**, and the composite beats it on every test. |
| Cross: SPY core + market-neutral overlay | `xs_portable_alpha` | **Works to some degree by its own design bars.** A1 fails on T2 and on both locked sets. By the lab's gates it fails G6, by construction: beta ~0.4 from its SPY core, so it is not market-neutral. Highest 2018-26 CAGR of the cross designs (10.1% a year), a better Sharpe than buy-and-hold and SPY (1.06 vs 0.73 / 0.83), about 0.4 of buy-and-hold's drawdown. It still trails buy-and-hold's 13.7% and SPY's 15.0%. |
| Cross: short interest + momentum + profitability | `xs_three_premia` | **No edge as designed.** Profitability hedged the 2026 slump but cost all of the alpha on TRAIN (−1.2% vs +2.0%) and most of it on T1, T4 and TL. On T2 it cost 1.4 points (+3.6% vs +5.1%; 0.5 against the repaired +4.2%). |

---

## 2. What was done

1. **One harness for every idea.** `strategies/proposals_lab/lab.py` builds point-in-time daily panels for
   2015-2026. It runs every candidate through the same backtest:
   - next-open fills;
   - the repo's cost model and `lock_cash`;
   - borrow fees by short-interest tier;
   - warm half-year periods.

   It also holds the statistics: block bootstrap, paired deltas, shuffled-name luck bands, noise copies,
   probability of backtest overfitting (PBO) and gates G1-G8.

   etf_trend is the one exception:
   - it runs the same `lab.book` on its own ETF panel (from 1993), with its own half-years (TRAIN 2004-15,
     VALID 2016-26, LOCKED 2000-03);
   - its long/short variant's shorts pay a flat 0.5% a year (the final is long or flat);
   - its alpha is measured against SPY.
2. **Six ideas built as new strategy folders**, proposals 2-5 plus two of the "also considered" ideas
   (earnings drift and cross-asset ETF trend). Each followed the same protocol:
   - a pre-registration before any number;
   - a training-era premise check;
   - the untuned literature default first;
   - Optuna on TRAIN only (≥ 120 trials, a robust objective, PBO recorded);
   - at least two pre-registered iterations, each needing a paired gain on every open test plus a noise
     margin;
   - a declared final candidate, read on T4 (etf_trend: on its own locked 2000-03 window).
3. **Independent adversarial reviews.** A separate agent reviewed each build for look-ahead, reproduction,
   claims and execution. Across the six, the reviews found five defects in the harness itself (§3). They
   were fixed as **lab v2**. Every strategy was then re-measured on v2, reviewed again and repaired. Three
   decisions changed:
   - resid_momentum lost its training gate: "works" became neither label;
   - quality_junk's final candidate switched from U0 to the re-tuned TRv2;
   - peer_reversal's final candidate switched from I4 to U0.
4. **Five cross-strategies.**
   - Three designers proposed 12 designs, each from a different angle: combining signals, portfolio
     construction and risk, falling markets.
   - A judge selected five of the 12. Its list of 19 other entries, each dropped with a written reason or
     folded in as a control or iteration, covers:
     - six of the seven unselected designs (two share an entry);
     - 14 further ideas: seed ideas, variants, and the component strategies as sleeves.

     The seventh design, a breadth-switched SPY hedge, has no entry.
   - All five designs were built on the three books with T1 stock-selection skill: short interest (SI),
     residual momentum (MOM) and the adaptive hedge (AH).
   - Each design was pre-registered, built, reviewed and fixed.
5. **The locked read.**
   - The overall final (A_U0) and its confirmation bar were written down in
     `strategies/proposals_lab/FINAL_DECLARATION.md` before any T4 or TL number of a cross-strategy existed.
   - T4 had already been read for the proposal finals, and informed_flow_ls had already read names
     1501-2000 for SI (+12.2%).
   - Then `final_batch.py` ran every design's final and the controls once on T4 and TL.
   - Afterwards all 11 locked books were rebuilt from vectorbt's order log and reconciled (§4).

In all, 63 variants were counted against pre-registered budgets: 47 in the proposals and 16 in the
cross-strategies. Tuning studies and diagnostics come on top.

## 3. How it was judged

| test | window | names | role |
|---|---|---|---|
| E0 | 2016-07 → 2018-06 | top 500 | TRAIN (no short interest before 2018; only its last half-year has any) |
| TB | 2018-07 → 2020-06 | top 500 | TRAIN (Q4 2018, COVID) |
| T3 | 2020-07 → 2021-12 | top 500 | TRAIN |
| T1 | 2022-01 → 2026-10-16 | top 500 | VALIDATION (2022 bear, Feb-Apr 2025) |
| T2 | same | names 501-1000 | VALIDATION on other names |
| T4 | same | names 1001-1500 | LOCKED: read once per proposal final by rule; re-read on lab v2 as disclosed re-measurements that selected nothing (earnings_drift three times in all); read again in `final_batch.py` for the cross-strategy finals, their controls and adaptive_hedge's TUNEDv2 (a third run of its final, which reproduced +4.2%) |
| TL | same | names 1501-2000 | LOCKED: read once, in `final_batch.py`, for the cross-strategy finals, their controls and adaptive_hedge's TUNEDv2 (a disclosed component, and proposal 3's final) |

Universes are re-ranked every half-year on the six months before it.

**Gates.** "Works" means passing all of them; "works to some degree" means passing G1, G2, G3, G5 and G6.

| gate | bar |
|---|---|
| G1 | TRAIN-era alpha > 0 |
| G2 | T1 alpha > 0 with P ≥ 0.90 |
| G3 | T2 alpha > 0 |
| G4 | T1 beats ≥ 95% of shuffled-name books |
| G5 | T1 alpha > 0 with late fills (+50 bp a leg) and with 2× costs |
| G6 | beats the market in every bear episode, and has a positive mean in the market's down months |
| G7 | \|beta\| ≤ 0.3 (market-neutral designs) |
| G8 | T1 alpha > 0 without the five biggest contributors |

**Lab v1 → v2, the five harness defects the reviews found** (`strategies/proposals_lab/README.md`, "Lab v2"):

1. **Holiday rows.** The panels' calendar was the union of all names. Swiss listings trade on US holidays,
   so the 2026 panels had three days on which no US stock had a price.
   - Books exited everything and rebuilt the next day; adaptive_hedge's "2026 factor crash" was this.
   - A NaN market return also switched peer_reversal's 21-bar volatility rule off for 21 bars after each
     row.
   - v2 keeps only NYSE sessions.
2. **Day 0.** `pct_change().fillna(0)` dropped each half-year's first day. That day carries the opening
   costs, the first move, and the borrow the engine charges up front for shorts held to the period's end.
3. **Starved new shorts.** The engine's `call_seq="auto"` ran a new short before the covers that free its
   cash, and the refused short was never re-sent. In resid_momentum's E0, 840 position-days were off target.
   v2 orders each bar: long cuts, short covers, new shorts, long buys.
4. **Borrow on tiny re-trims.** The engine charges accrued borrow as a fee fraction of the buy-back order.
   On a hair-sized re-trim that fraction explodes: etf_trend's long/short variant was charged 4-5× its
   borrow.
5. **ETFs in the stock universe.** Six sector SPDRs are stored as equities and sat in the top 500 from
   2024-H2. Their rows began in 2023-10, too few sessions for the 2024-H1 ranking. They are excluded.

v2 moved most alphas by 0.3-2 points, but two proposals' by more:
- peer_reversal's T1 by 2.3-4.8 points: U0 −1.1% → −3.5%; the round-1 final I4 +1.7% → −3.1%;
- adaptive_hedge's TB, T1, T2 and TRAIN era by 2.6-5.6 points: TB +7.5% → +1.9%, mostly one Wayfair short.
  T1's 2.6 is from the holiday-fixed v1 book (from round 1's +12.9%, only 1.5); T3 moved 1.0 and T4 not at
  all.

That was enough to flip resid_momentum's training gate and to undo peer_reversal's round-1 adoption.

---

## 4. The survivor: `xs_si_mom_composite` / A_U0

**Rules (config as shipped; nothing tuned):**

1. **Eligible:** close ≥ $5, a days-to-cover value and a share count.
2. **Short-interest rank:** minus days-to-cover averaged over 42 sessions, as a percentile within five
   dollar-volume groups. A name needs a days-to-cover to be eligible at all, so p1's 50% coverage floor
   never fires for this sleeve. A days-to-cover older than 45 days is masked, and its name drops out.
3. **Momentum rank:** the residual return of months t−12 to t−1. Each day's return is measured against the
   equal-weight market, using a 252-day beta, and the sum is divided by the residuals' standard deviation.
   It is ranked at month ends.
4. **Composite:** the mean of the two ranks, with a missing rank counted as 0.5. It is averaged over 5 days
   and re-ranked.
5. **Book:**
   - A long enters at rank ≥ 0.90 and leaves below 0.80. A short enters at ≤ 0.10 and leaves above 0.20.
   - A position is held at most 126 sessions after it last ranked in its entry zone: the clock restarts
     while it stays there.
   - A short is bought back at +100% from its entry.
   - A name is never shorted above 30% short interest / shares.
6. **Sizing:** 15 + 15 equal slots of 3% (gross 0.90, dollar-neutral). Decided at the close, filled at the
   next open. A long is trimmed back to its slot once it reaches twice its slot.
7. **Costs:** the repo's commission and spread at $1,000 orders (a ~$33k account). Borrow is 0.5% / 3% / 10%
   a year by short-interest tier.

**Results on every test:**

| | TRAIN (TB+T3) | T1 | T2 | T4 (locked) | TL (locked) |
|---|---|---|---|---|---|
| alpha (P) | +2.0% (0.62) | +12.9% (0.962) | +5.1% (0.84) / +4.2% repaired | +12.4% (0.997) | +8.6% (0.963) |
| late fills (+50 bp a leg) | −0.4% | +10.8% | +3.0% | +10.1% | +6.5% |
| 2× borrow | +1.4% | +12.3% | +4.4% | +11.5% | +8.0% (late fills too: +5.9%) |
| whole shares, $33k account | — | +11.9% | +4.0% | +12.0% | +8.5% |
| without the top-5 names | — | +8.0% | +1.2% | +8.4% | +4.3% |
| beta, stitched | +0.07 | +0.01 | −0.09 | −0.09 | −0.10 |
| beta by half-year (realized) | −0.08 to +0.40 | −0.40 to +0.73 | −0.32 to +0.36 | −0.31 to +0.25 | −0.34 to +0.04 |
| CAGR vs equal-weight buy-and-hold | — | 13.0% vs 8.8% | 4.4% vs 7.6% | 11.5% vs 10.2% | 6.5% vs 19.4% |
| CAGR vs SPY (same days) | — | 13.0% vs 11.9% | 4.4% vs 11.9% | 11.5% vs 11.9% | 6.5% vs 11.9% |
| max drawdown | −14.0% | −21.9% | −15.6% (−19.1% repaired) | −11.7% | −13.3% |
| shuffled-name books beaten | — | 100 of 100 | — | 100 of 100 | 100 of 100 |

The T2 rows other than alpha and max drawdown are as run.

- **Gates:**
  - All eight pass on the open tests, as do all four of the design's own pre-registered bars.
  - Bar (2), integration over `C_mix`, passes only thinly on T1. Its Δ of +3.20 clears the 2 × noise bar
    of 2.91 by 0.29 points, and one of three paired noise copies shows −0.4.
  - A_U0's own three T1 noise copies average +10.4%, not +12.9%.
- **Execution:** every locked period of all 11 books in the batch was rebuilt from vectorbt's order log and
  equals the saved returns.
  - A_U0, C_mix, C_si, C_mom, C_U0, the portable-alpha U0, C_core, C_mix3 and CRISIS had no refused or
    part-filled order.
  - xs_three_premia's B_U0 had two refused orders and five part-filled ones, all in TL 2025-H2:
    - WVE's cover, stuck 16 bars (TL +3.37% → +3.12% with its stuck P&L removed);
    - one new short;
    - five part-filled new shorts.

    The short exposure those new shorts missed (1.18% of the book on average that half-year) is not
    priced in the +3.12%.
  - adaptive_hedge's TUNEDv2: see §5.
- **The one T2 failure:** the 2024-05-14 GME squeeze. GME's cover part-filled and the same day's new short
  (NTLA) was refused.
  - Together the two stuck orders flattered T2 by ~0.9 points: GME's stuck short +1.0, NTLA's missing
    short −0.1.
  - The "repaired" T2 figure fills both at their due open.
- **Integration vs simply holding both books** (paired Δ over `C_mix`, the SI and MOM books side by side):
  - TRAIN +1.4, T1 +3.2 and T2 +1.8 (repaired). None is significant alone (P 0.65-0.76).
  - T4 +8.0 (P 0.997).
  - TL −0.1 (P 0.49). There the short-interest book alone was best (+10.9%).
  - Per unit of residual risk the composite ties C_mix and C_si on T1 (0.79 vs 0.83 and 0.85), at 16.3%
    volatility against C_mix's 11.6%.
- **Calendar years on the top 500 (book vs equal-weight market):**

  | 2018-H2 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 to Oct 16 |
  |---|---|---|---|---|---|---|---|---|
  | −5.4 vs −6.2 | +2.4 vs +35.6 | +20.2 vs +25.7 | −3.1 vs +21.5 | +15.6 vs −18.6 | +5.4 vs +22.7 | +19.2 vs +15.3 | +34.5 vs +16.2 | −8.3 vs +11.7 |

- **Borrow, quoted:** IBKR's borrow file of 2026-10-21 (`var/ibkr/usa-20261021.txt`) covers every name
  the book shorted in each test's last half-year (20-22 per name set), with more shares available than a
  $33k book needs.
  - Their median fee is 0.31-0.41% a year (0.30-0.42% for the 15 shorts held on the last bar), below the
    model's tiers on average.
  - The exceptions are business development companies: TSLX 9.2%, ARCC 4.5%, MAIN 4.4%, GBDC 2.9%,
    modelled at 0.5-3%.
  - This is one day's snapshot, not the history.
- **Beta sizing (post hoc, open tests only, decides nothing):**
  - The test swapped equal slots for proposal 1's beta sizing (a slot scaled at entry by the median beta
    over the name's beta), run in memory without editing the pinned strategy.
  - T1: alpha +9.8% (Δ −3.0, P 0.09), half-year beta −0.30 to +0.48, 2026 ex-ante beta +0.40 instead of
    +0.65, max drawdown −15.6% instead of −21.9%.
  - T2: +5.8% (Δ +0.7). TB +1.6% (Δ −1.6), T3 −1.5% (Δ +0.9).
  - Sizes are frozen while a name is held, so the drift is cut by about 40% (2026 ex-ante +0.65 → +0.40),
    not removed.
- **Risks:**
  - T1 volatility was 16.3%, and the worst drawdown was −21.9% (2025-10 → 2026-07).
  - The top-500 book is net long beta now (see §1).
  - Part of T1 is in-sample for the short-interest signal, whose earlier studies used 2020-26 data. TL is
    fresh only for the combination.
  - TRAIN does not survive late fills (−0.4%), and the composite's TRAIN rank IC was −0.017 (t −0.75): the
    training-era case rests on the book path, not on a monthly signal.
- **Rejected by their pre-registered bars:**
  - **a profitability gate on shorts:** +1.0 on T1, below the noise bar;
  - **a sector-neutral version:**
    - same T1 alpha (Δ −0.03), +3.3 on TRAIN and on T2;
    - drawdown −16.5% instead of −21.9%;
    - it also halved the 2022 bear gain (+8.2% vs +15.4%);
    - it was tested as an alpha iteration and failed the T1 noise bar; reading it as a risk choice would
      need its own pre-registration;
  - **the TRAIN-tuned momentum weight of 1.5:** T1 −3.4 (but T2 +4.7), PBO 0.75.

---

## 5. The proposals, one by one

### Proposal 2 — `resid_momentum`: crash-protected residual momentum, long/short

- **Built:**
  - a 12-1 residual momentum score against the market (the residuals' sum over their standard
    deviation), ranked across all eligible names;
  - 20 + 20 slots at gross 0.90;
  - the Barroso-Santa-Clara volatility scale;
  - a cap of 30% short interest on shorts.
- **Result (untuned default, final):**
  - TRAIN 2016-21: −1.4% (P 0.37). E0 −8.2%, TB +6.5%, T3 −2.9%.
  - T1: +8.7% (P 0.98, luck 100th percentile).
  - T2: +5.5% (P 0.93).
  - T4: +6.2% (P 0.95).
  - All four bear episodes beaten: Q4 2018 −4.8%, COVID +4.0%, 2022 +11.8%, Feb-Apr 2025 +6.6%.
- **Optimization:** 150 TPE trials on TRAIN beat the untuned default by 3.6 points in-sample (TRAIN
  alpha +2.2% vs −1.4%) and trailed it by 3.3 points on T1 (+5.4% vs +8.7%; PBO 0.74).
- **Iterations:** sector factors, beta sizing and the controls were not adopted.
- **Why neither "works" nor "works to some degree":** it fails G1 (TRAIN −1.4%, P 0.37) and passes
  G2-G8. The edge is real but a 2022-2025 regime: positive on three disjoint name sets then, negative in
  2016-21 and in 2026-H2.
- **What it gave:**
  - It became the second signal of the survivor.
  - Residual ranking plus the volatility scale keep raw momentum's T1 alpha (+8.7% vs +8.2%) at 38% of
    its drawdown (−9.5% vs −25.1%). The scale alone halves it (−18.8% → −9.5%), for 0.8 points of T1
    alpha.
  - It found the "winning shorts' gains locked twice" cash mechanism, which forced gross down to 0.90
    (§9, #49).

### Proposal 3 — `adaptive_hedge`: short-interest longs plus breakdown shorts that appear in falling markets

- **Built:**
  - informed_flow's long sleeve;
  - trend_following's breakdown shorts (Donchian break + ADX), up to 20, displacing the weakest long;
  - an ATR chandelier exit.
- **Result:**
  - The untuned U0 fails G1 (TRAIN −0.8%).
  - The tuned final (TUNEDv2) passes every applicable gate on the open tests: TRAIN +5.7% (P 0.82), T1
    +11.4% (P 0.985), T2 +4.3% (P 0.77).
  - On the locked T4 (a disclosed second read) it is +4.2% (P 0.84), below G2's confidence bar.
  - Falling markets: COVID +20.7%, 2022 +25.9%, down months +2.7%.
- **Locked TL** (read once, beside the cross-strategy finals): **−4.8% (P 0.21), CAGR −11.1% against
  buy-and-hold's +19.4%, max drawdown −57%.**
  - At gross 0.98 the order log shows 2 covers refused at zero free cash (BKKT 2024-11, SGML 2025-12).
  - It also shows 2 refused long buys and 111 part-filled entries across T4 and TL.
  - With the stuck P&L removed TL is −5.5%: the failure is real.
- **Why only "to some degree":**
  - The proposal's adaptive mechanism failed. Every attempt to make the shorts adaptive lost: a 1-month
    clock, a Turtle exit, a +50% squeeze stop, a crowded-only filter. The shorts earn by holding broken
    stocks for about a year, in every market.
  - Its CAGR is far below buy-and-hold: 5.9% vs 13.8% a year, 2018-26.
  - It now fails on the smallest names.
- **What it gave:**
  - the first refused-cover diagnosis: a Wayfair short in 2020 cost ~13% of the book;
  - the reconciliation tool, `reconcile.py`.

### Proposal 4 — `quality_junk`: quality minus junk, beta-neutral

- **Built:**
  - profitability ranked within sector (and measure);
  - safety (minus 252-bar volatility; 189 bars in TRv2) ranked across all eligible names;
  - 15-20 slots a side, and beta sizing.
- **Result:**
  - The untuned U0: TRAIN +0.1%, T1 −0.4% (P 0.51, exactly the median of shuffled books), T2 −0.05%.
  - The tuned TRv2 was adopted by the tuned-values rule, on margins within the noise: T1 +0.6% (P 0.56),
    T2 +0.0%, T4 +0.6%, which is 1.1 points below U0.
  - It fails G2, G4, G5 and G8.
- **Why:**
  - The quality premium did not show up in these universes, before or after costs. Base trading costs
    take only 0.1-0.3 points a year.
  - Its positive mean in down months (+0.67%) is about half net short beta (U0's +0.92% is two-thirds
    beta). Made beta-neutral, TRv2 averages +0.06% a down month and U0 −0.2%.
  - Profitability alone on names 501-1000 made +5.0%, but rested on five names (+0.3% without them).

### Proposal 5 — `peer_reversal`: buy a stock's drop against its peers, more so under stress

- **Built:**
  - a 5-day return relative to correlation-cluster peers;
  - a stress switch that raises gross when the market is volatile;
  - 40 slots.
- **Result:** TRAIN −7.7%, T1 −3.5%, T2 −6.5%, T4 −2.9%. With late fills −27%.
- **Why:**
  - About 3,100 orders a year cost ~6 points a year.
  - On T1 its stock selection beat same-turnover random names by +3.0 points a year on the exact top
    500 (99th percentile), and by +3 to +6 on three copies with 5 names dropped a period. On the exact
    names that is half the costs.
  - The luck test was not run on TRAIN, T2 or T4, and nothing on TRAIN reaches t 2: the selection is
    suggestive, not established.
  - Sector peers beat the correlation clusters.
  - The stress switch adds nothing (−0.1 to +0.1).
  - Its round-1 adoption (a lower-turnover I4) was undone on v2: the gain was name-set luck.

### Also considered — `earnings_drift`: post-earnings drift, long/short

- **Built:**
  - the yfinance earnings surprise plus the two-day reaction;
  - a 60-session hold, 30 slots a side (30 long + 30 short);
  - a cent rule for coarse EPS data.
- **Result:** TRAIN −3.3% (P 0.02), T1 +2.2% (P 0.81), T2 −0.8%, T4 −3.4%. Late fills take T1 to −2.5%.
- **Why:** costs and late fills are larger than the drift. The one positive test rests on the coarse-data
  rule.
- **Data note:** the earnings data end on 2026-06-11.

### Also considered — `etf_trend`: cross-asset trend

- **Built:**
  - 23 risky ETFs plus SHY as cash, with yfinance history from each fund's inception (SPY 1993). All
    24 are live only from 2008-H2, and the LOCKED 2000-03 test holds equity ETFs only;
  - a 12-month excess-return trend;
  - inverse-volatility sizing;
  - long or flat, with SHY as cash.
- **Result (alpha vs SPY):**
  - TRAIN 2004-15: +5.25%, of which ~70% is 2007-09.
  - VALID 2016-26: +0.02%. Its asset timing ranks at the 31st percentile of shuffled books.
  - LOCKED 2000-03: +0.11%.
- **What it is:** a de-levered SPY with a cash sleeve (correlation 0.80).
  - It truly sidestepped only 2008 (+4.6% vs SPY −54.6%).
  - In every other fall it still lost about a third of SPY's loss: 2000-02 −17.7% vs −48.5%, 2022 −8.7%
    vs −24.3%.
  - It lost 1.1% in SPY's average down month.
- **Variant:** the long/short version made +31.4% in 2008-09 but lost 3.1% a year in 2016-20 (alpha vs
  SPY −4.4% a year).

---

## 6. The cross-strategies

| design (final) | what it combines | TRAIN | T1 (P) | T2 | T4 locked | TL locked | verdict |
|---|---|---|---|---|---|---|---|
| **`xs_si_mom_composite` (A_U0)** | SI × MOM, one ranking | +2.0% | +12.9% (0.96) | +5.1% / +4.2% | **+12.4%** | **+8.6%** | **works, confirmed** |
| `xs_three_book` (CRISIS) | MOM + AH books, one account | +5.2% | +9.9% (0.99) | +4.8% | +5.0% | **−0.0%** | fails TL |
| `xs_cross_leg` (C_U0) | SI longs, MOM shorts | +0.7% | +10.6% (0.97) | +3.9% | +3.2% | +4.8% | hypothesis rejected |
| `xs_portable_alpha` (U0), net of SPY | 0.45 SPY + SI/MOM overlay | +1.2% | +5.0% (0.99) | +2.4% | +2.8% | +4.2% | to some degree by its own bars (bear episodes, drawdown and net alpha pass; A1, CAGR above buy-and-hold, fails on T2, T4 and TL); fails the lab's G6 by construction |
| `xs_three_premia` (B_U0) | SI + MOM + profitability | −1.2% | +6.1% (0.88) | +3.6% | +3.4% | +3.4% | no edge |

**`xs_three_book`:**
- **Design:** the SI, MOM and AH books at 0.30 each in one shared account. The pre-registered challenger
  CRISIS (MOM + AH at 0.45 each) became the declared final.
- **Open tests:** it passed every gate and was the best falling-market book of the five cross-strategy
  finals: COVID +11.8%, 2022 +18.8%, positive in 82% of the market's down months. Its own component,
  adaptive_hedge TUNEDv2, did better in Q4 2018, COVID, 2022 and the down months (§7).
- **What the reviews found:**
  - Most of its TRAIN result is AH's in-sample tune. With AH untuned its TRAIN is about +1.1%, a
    return-sum estimate from the saved books, against +4.2% with AH tuned on the same basis. The three-book
    U0 with AH untuned reads +0.7%.
  - Its margin over the three-book U0 was noise-sized.
- **Locked:** it failed on TL (−0.03%, max drawdown −30%). No order was refused or part-filled there, so
  the failure is real.

**`xs_cross_leg`:**
- **Design:** SI picks the longs and MOM picks the shorts.
- **Gates:** all eight pass.
- **Why the hypothesis is rejected:** the pre-registered falsification failed. The reverse cross (MOM
  longs, SI shorts) is not reliably worse (P 0.78 and 0.68 against a 0.80 bar).
- **Against the composite:** A_U0 beats it on TRAIN, T1 and T2, and on both locked sets.

**`xs_portable_alpha`:** the design asked "how much of a bull market can an index core buy back while still
winning in bear markets?"
- **Design:** SPY at 0.45 of the book, plus the SI/MOM overlay of `xs_three_book`'s C_core (the books side
  by side), not the composite.
- **Over 2018-26:**
  - It compounded 10.1% a year against the overlay's 6.7%, the market's 13.7% and SPY's 15.0%.
  - Sharpe 1.06, the best of the three (buy-and-hold 0.73, SPY 0.83); max drawdown about 0.4 of
    buy-and-hold's (−14.7% vs −37.4%).
  - In every bear episode it lost only 28-64% of the market's loss.
- **Against its own bar:** it fails A1 (CAGR above buy-and-hold) on T2 by 0.06 points, and on both locked
  sets (T4 7.7% vs 10.2%, TL 8.6% vs 19.4%). It never beat SPY's CAGR in any window.
- **Alpha net of the core:** +2.8% on T4 and +4.2% on TL.
- **Verdict:** the best "beat the market's risk, not its return" book, but not a buy-and-hold beater in a
  bull market. It was sized for a ~$150k account.

**`xs_three_premia`:**
- **Design:** adds within-sector profitability as a third rank, as a hedge for the 2026 slump.
- **The hedge worked:** 2026-H2 on T1 was +3.7% a year against A_U0's −22.2%.
- **It cost more than it gave:** −6.8 points of T1 alpha, −3.1 on TRAIN. It fails G1 and G2.
- **Post hoc:** holding the three books side by side (C_mix3) passes every gate, but it was a control, not
  a pick.

---

## 7. Falling markets: what actually protected

**A_U0 on every 2022-26 name set** (book vs that set's equal-weight market):

| A_U0 | 2022 bear | Feb-Apr 2025 | Jul-Sep 2026 (momentum crash) | mean in the market's down months | mean ex-ante beta, 2026-H2 |
|---|---|---|---|---|---|
| T1, top 500 | +15.4% (−24.5%) | +1.7% (−19.5%) | −2.3% (−1.6%) | +1.15% vs −3.86% (64% positive) | +0.65 |
| T2, names 501-1000 | +10.0% (−21.7%) | +1.0% (−21.1%) | +3.5% (−3.9%) | +0.75% vs −4.28% (56%) | +0.38 |
| T4, names 1001-1500 | +4.5% (−18.4%) | +6.7% (−22.4%) | +0.9% (−4.0%) | +1.63% vs −4.14% (73%) | +0.17 |
| TL, names 1501-2000 | +6.2% (−16.2%) | +3.4% (−20.4%) | +5.8% (−3.4%) | +0.96% vs −3.76% (63%) | −0.06 |

**Every book, top 500, since 2018-07** (`*`: since 2016-07):

| book | Q4 2018 | COVID crash | 2022 bear | Feb-Apr 2025 | mean in market's down months |
|---|---|---|---|---|---|
| equal-weight market | −18.2% | −37.0% | −24.5% | −19.5% | −4.1% |
| SPY | −18.7% | −33.4% | −24.1% | −18.6% | |
| **A_U0 (composite)** | −7.0% | −0.1% | **+15.4%** | +1.7% | +0.95% (61% of months positive) |
| C_mom (momentum alone, same harness) | −3.3% | +6.7% | +13.8% | +4.2% | +1.31% |
| CRISIS (MOM + AH) | −3.1% | +11.8% | +18.8% | +3.6% | +1.87% (82%) |
| adaptive_hedge TUNEDv2 | −1.8% | +20.7% | +25.9% | +0.3% | +2.69% (82%) |
| resid_momentum | −4.8% | +4.0% | +11.8% | +6.6% | +1.17% (76%) * |
| SI alone (informed_flow_ls v6, lab copy) | −8.2% | −1.0% | +11.2% | −3.6% | +0.03% |
| xs_portable_alpha (0.45 SPY) | −11.7% | −14.5% | −6.8% | −7.7% | −1.16% (32%) |
| quality_junk TRv2 | +4.1% | −7.4% | −1.0% | +8.7% | +0.67% * |
| peer_reversal | +0.1% | −11.5% | +7.3% | +1.4% | −0.78% (27%) * |
| earnings_drift | −1.1% | −2.2% | +4.5% | −2.3% | +0.09% (51%) * |

etf_trend is measured against SPY over 2004-26: 2008-09 +4.6% (SPY −54.6%), Q4 2018 −6.1%, COVID −12.3%,
2022 −8.7%, Feb-Apr 2025 −6.2%. Its mean in SPY's down months is −1.07%, positive in 19% of them.

**What this shows:**

- **Any low-beta book "beats the market in every bear episode".** Only the alpha and the down-month mean
  separate skill from simply being out of the market.
- **The books that made money in falls are short-side books.**
  - A_U0, CRISIS, AH and resid_momentum all short stocks that are already breaking down (negative
    momentum) or crowded.
  - The SI book alone is the weak one in falls: its 2018-H2 stock selection lost (proposal 1's bear test,
    −3.5% alpha over 2018-20).
- **Momentum lifts the composite in falls; short interest makes it profitable in normal times.**
  - That is why their combination beats either alone on most tests.
  - It is not bear-proof. It did better than SI alone in every episode (down months +0.95% vs +0.03%), but
    it lost 7.0% in Q4 2018 and trailed momentum alone in three of the four episodes.
- **A book's protection is its beta when the fall begins.**
  - In 2018-H2, which holds Q4 2018, A_U0's realized beta was +0.37, and it lost 7.0% in the fall.
  - In 2022 it ran net short and gained.
  - The top-500 book is net long today (+0.65). The smaller-name books are less so (names 501-1000 +0.38,
    names 1001-1500 +0.17); only names 1501-2000 are about flat (−0.06).
- **A 12-month trend on ETFs truly sidestepped only 2008.**
  - In every other fall, long or fast, the ETF book still lost about a third of SPY's loss.
  - The 12-month signal on SPY alone lost nearly the whole 2022 bear (−21.8% vs −24.3%).
  - Its long/short version is a true crisis book, at a large cost in normal years.

---

## 8. Lessons that apply beyond these folders

1. **Tuning rarely carried out of sample.** Six of the seven TRAIN-tuned points lost on at least one
   out-of-sample test, and three lost on every one they were read on:
   - resid_momentum: −3.3 on T1 (T2 −0.0), PBO 0.74;
   - the composite's momentum weight: −3.4 on T1 but +4.7 on T2 (P 0.93), PBO 0.75;
   - quality_junk: +1.0 on T1 and +0.1 on T2, inside the noise (this made it the final), then −1.1 on T4;
   - earnings_drift: +0.6 on T1, −2.5 on T2;
   - etf_trend: −0.4 on VALID;
   - peer_reversal's round-1 values: −2.7 on T1 (T2 −5.8).

   The exception is adaptive_hedge. Its tune (PBO 0.10) beat U0 on T1 (+4.7), T2 (+4.1) and T4 (+1.2) and
   is its final. It then lost 4.8% on TL, where U0 was not read. Outside adaptive_hedge, optimization is
   useful here only to map how flat a surface is.
2. **Signal-level integration beat side-by-side books clearly only once.**
   - The one SI + MOM ranking beat holding the two books side by side by +8.0 on T4 (P 0.997). On TRAIN,
     T1 and T2 its margins (+1.4 to +3.2) are the size of the noise, and on TL it did not beat them (−0.1).
   - Part of its extra alpha looks like concentration: 15 + 15 names on which both signals agree, against
     C_mix's ~30 + 30 half-size slots. That is a reading, not a measured attribution.
   - Adding a third rank (profitability) lost to the three books held side by side.
   - Book-level blends (three_book, portable alpha) are risk tools.
3. **2022-2025 was a regime.** Both proven signals (short interest and residual momentum) were strongest
   then, weaker in 2016-21 and in 2026. Four name sets agreeing within one era is not four eras.
4. **The harness is part of the result.**
   - Lab v1 → v2 moved most alphas by 0.3-2 points, and some by 2.3-5.6.
   - A single stuck cover moved a test by 0.25-3.4 points: Wayfair 2020 (TB, 3.4), GME 2024 (T2, 1.0; 0.9
     net of a refused short), BKKT 2024 (TL, 0.7 together with SGML), WVE 2025 (TL, 0.25).
   - The edges here are 2-13 points, so audit execution on every test, locked ones included, before
     reading returns: reconcile targets against positions from the order log.
5. **Noise is book-specific: ±0.5-4 points of T1 alpha.**
   - Dropping 5 random names a period (the noise copies) moved the composite's T1 alpha by 1.6-4.2 points
     (sd 1.45).
   - The other 15 + 15 books moved less: xs_cross_leg 0.5-2.5 (sd 1.06), xs_three_premia 0.5-0.6 (sd
     0.62).
   - Any adoption margin smaller than a book's own noise is a coin flip. Report the copies' mean, not just
     their spread: all three of the composite's T1 copies landed below the label.
6. **Costs kill daily-turnover ideas at retail order sizes.** peer_reversal's selection beat random names
   by about half its costs, on ~3,100 orders a year, and it loses. Every survivor trades ~130-360 orders a
   year.
7. **A "crisis sleeve" that is short all the time is not adaptive.** AH's shorts made money because they
   were always on, not because they appeared in falls. Adaptive versions lost.
8. **A dollar-neutral book is not a beta-neutral book.** The composite's net beta is left free, and it does
   not follow the market of the same half-year.
   - Its short side carried more beta in 2022-23, its long side since 2025-H2.
   - By half-year on the top 500 it ran from −0.40 (2023-H1, while the equal-weight market rose 12.5%) to
     +0.73 (2026-H1). It stayed at +0.40 in 2026-H2 as the market slipped 1.0%.
   - Equal dollars hide it, and so does a stitched beta of +0.01.

---

## 9. Engine and data findings

**Update 2026-10-24: all seven were approved and implemented in `src/`, with regression tests** (1,096
tests pass):
- #46 `backtest._call_order`;
- #47 working orders and #49 their warning, `backtest._working_orders`;
- #48 the borrow deferral in `backtest._borrow_fees`;
- #50 `data.us_sessions_only`;
- #51 the six SPDRs retyped `etf` in the shared `securities` table;
- #52 `db.adjustment_splice` in `tsim backfill`.

The studies that had journals were bumped: informed_flow v42, conviction_flow v4, turnover_momentum v4
and value_factor v4. informed_flow_ls (v8, under review) was not touched: its study needs a bump by its
owner. Every number in this report was produced before these changes, on lab v2, whose runner already
had #46 and #48. The text below is the proposal as filed.

Numbered after the existing ledger (#45).
- Four of the fixes already exist inside the lab (`strategies/proposals_lab/lab.py`) and can be lifted
  into `src/` once approved:
  - #46 (`lab._call_order`) and #48 (`lab.borrow_fees`) are tested in `tests/test_proposals_lab.py`;
  - #50 (NYSE sessions in `lab.build`) and #51 (`lab.EXCLUDE`) have no test.
- #47 and #49 are not fixed in the lab.
  - Its runner still never re-sends a refused or part-filled order. Stuck orders are only counted after the
    fact, by `xs_si_mom_composite/research.py` and `adaptive_hedge/reconcile.py`.
  - resid_momentum and the five cross-strategies avoid #49 by running at gross 0.90.
  - adaptive_hedge, quality_junk, earnings_drift and etf_trend run at 0.98 (peer_reversal on stressed
    bars), and adaptive_hedge's locked books had 111 cash-limited part-fills.

| # | finding | evidence | suggested fix |
|---|---|---|---|
| 46 | **Portfolio mode starves new shorts.** Under `lock_cash`, `call_seq="auto"` can run a new short before the covers that release its cash; it is refused or part-filled, and never re-sent. | `test_engine_starves_the_new_short_the_lab_runner_fills_it`: the engine fills ~20 of ~500 shares. resid_momentum E0: 840 position-days off target. | Order each bar explicitly: long cuts, short covers, new shorts, long buys (`lab._call_order`). |
| 47 | **A refused or part-filled order is never re-sent.** `rotation_targets` emits a target only when it changes, so a cover refused at zero free cash leaves the short open until the next signal change. | Wayfair 2020 (46 bars stuck, −6.7% of the book), GME 2024 (32 bars), BKKT 2024 (29 bars), WVE 2025 (16 bars). | Re-emit the unfilled remainder on the next bar until held; at least count and warn. |
| 48 | **The borrow-fee fraction explodes on tiny re-trims.** `_borrow_fees` charges accrued × \|w\| / \|x − w\|. | etf_trend's long/short variant was charged 4-5× its borrow; a 0.1% re-trim is charged 55% of its order. | Defer re-trims smaller than 25% of the short to the next real buy-back (`lab.borrow_fees`), or charge accrued borrow in dollars. |
| 49 | **The winning shorts' gains are counted against free cash twice** (vectorbt), so a full long/short book's last buys are cut at gross 0.98. | Cash-limited part-fills near full gross: resid_momentum at 20 + 20 slots; adaptive_hedge (gross 0.98) had 111 on its locked books. | Document a gross cap of ≤ 0.90 for full long/short books, or warn on any part-fill for lack of cash. |
| 50 | **`data.load_panel`'s calendar is the union of its names' bars.** Foreign listings (ALC, CFR) add US holidays on which no US stock has a price. | 2026-06-19, 07-03, 09-07: books exited everything and rebuilt (adaptive_hedge's "2026 factor crash"). | Restrict to NYSE sessions (a reference symbol's bars or an exchange calendar). |
| 51 | **Sector SPDRs are stored as `equity`.** XLE, XLF, XLI, XLK, XLV and XLY ranked into the lab's top-500 sets from 2024-H2. | Since etf_trend's yfinance backfill (2026-10-22) the DB holds them, still typed `equity`, from 1998-12-22, so a ranking built on today's DB admits them in every period. Swapping them for six stocks, together with removing the 2026 holiday rows (#50), moved peer_reversal's T1 by 2.5-5 points, most of it the swap. | Flag them as ETFs in the symbol table, or exclude them in the universe builder. |
| 52 | **An IBKR refresh can splice unadjusted bars onto adjusted history.** | Before etf_trend's backfill, SPY's stored series jumped +3.43% on 2023-10-17 (real move −0.005%), where IBKR rows (adj_close = close) met yfinance-adjusted history. | An adj_close / close splice check in the refresh path (filed in `etf_trend/IDEAS.md`). |

**The shared DB was changed (UPSERT only, as the repo allows).**
- On 2026-10-22, etf_trend's yfinance backfill of 24 ETFs upserted 152,925 daily rows.
- It overwrote 13,406 existing rows that other consumers read: SPY, GLD and SLV from 2015, and the six
  SPDRs from 2023-10-17/18.
- The overwrite removed the SPY splice of #52.
- Every earlier result that read those bars used the old ones: informed_flow, informed_flow_ls, the
  sibling repo and the lab's round 1.
- The old rows are saved in `var/proposals_lab/results/etf_trend/db_backup_before_backfill.pkl`.

**Also noted (not engine):**
- The engine pays 0% on idle cash and on locked short proceeds. A long/short book is understated by up to
  roughly the T-bill rate × its idle cash plus its locked short proceeds (about the whole book for a
  dollar-neutral book), less broker spreads. That is conservative.
- `lab.with_spy` sets SPY's open equal to its close. That is lab-only. It costs xs_portable_alpha's 0.45
  SPY core ~0.07 points of CAGR a year on T1/T2 and ~0.5 on TB.
- The DB holds survivors only.
- The $5 floor reads split-adjusted closes (the existing #45 covers it).

## 10. For the `informed_flow_ls` review (reported only; nothing there was edited, imported or run)

1. **Its walk-forward drops each period's day 0.**
   - `strategies/informed_flow_ls/research.py` line 353 (and 566) computes
     `equity_curve(pf)["value"].pct_change().fillna(0.0)`.
   - Each period's first day then loses its opening costs, its first open-to-close move, and the borrow
     the engine charges up front for shorts held to the period's end.
   - In the lab, dropping day 0 flattered half-yearly books by ~0-0.9 points of alpha a year at base costs
     and ~0.3-1.9 under late fills. adaptive_hedge was flattered much more: its day-0 returns summed to
     −7.7% over T1's ten periods, against the benchmark's −0.4%.
   - It was not measured on informed_flow_ls itself.
2. **It runs through the engine's portfolio mode at gross 0.98**, where the lab saw cash-limited
   part-fills, so #46, #47 and #49 can apply.
   - The reviewers can count them by rebuilding each period through the engine's own runner with
     vectorbt's order log on (`resid_momentum/research.py` `rebuild(..., runner="engine")`).
   - They can then read the log with `xs_si_mom_composite/research.py`'s `_gaps` and `off_target`.
3. **Check its panels for the holiday rows (#50) and the sector ETFs (#51).** A row on which SPY has no
   bar is a holiday row. Its `_select` exits a name whose score is NaN ("ineligible"), so on such a row the
   book would exit everything, as adaptive_hedge's did in the lab.
4. **Any of its numbers computed before 2026-10-22 on SPY, GLD, SLV or the SPDRs used the old bars** (§9,
   the DB change).

## 11. Caveats

- **Survivors only.** The DB holds today's listed names. Before 2020, "top 500" is nearly every name in the
  DB. Missing failures make shorts look worse and longs better, and they flatter buy-and-hold: TL's +19.4% a
  year is probably inflated.
- **In-sample overlap.**
  - T1 is partly in-sample for short interest, whose informed_flow studies scored from 2020-07.
  - T4 had been read for MOM, and TL by informed_flow_ls for SI, so the locked sets are fresh only for the
    combinations.
- **Borrow is modelled, quoted only once.**
  - Three tiers by short interest / shares; 2× borrow costs 0.5-0.8 points a year.
  - IBKR's quote is one day (2026-10-21). It cannot show past hard-to-borrow episodes such as GME's.
- **Every test restarts each half-year,** paying a full rebuild. Against one continuous book this is
  conservative by up to ~0.4 points a year at base costs and up to ~1.4 under late fills (measured on
  etf_trend and quality_junk).
- **Today's sector codes** are used for every year.
- **The earnings data stop on 2026-06-11.**

## 12. Next steps

1. **Paper-trade A_U0 before any real money.**
   - **What paper trading can show.** It is the only declared final confirmed on a locked name set, and
     that set was fresh only for the combination. Its half-year alphas swing by 10-23 points (sd), so one
     year's alpha has a standard error of ~8-16 points. A year of paper trading therefore tests execution,
     data and borrow, not the edge.
   - **Choose the universe** (the user's decision):
     - The top 500 has the largest alpha (+12.9%), but is partly in-sample, lost both 2026 halves and is
       net long beta now (+0.65).
     - Names 1001-2000 have the cleanest record: both locked sets positive, half-year beta at most +0.25,
       +6.5% to +10.1% even with late fills. They also have wider real spreads and more borrow outliers.
     - Paper books cost nothing, so running the two side by side is an option.
   - **Daily:**
     - Refresh short interest and the SEC share counts through `notebooks/03_altdata_update.py` with its
       coverage audits. A stale feed quietly shrinks the book:
       - eligibility needs a days-to-cover and a share count;
       - a days-to-cover older than 45 days is masked, so its name drops out of both ranks and is exited
         (every name, if the feed lapses).
     - Recompute the composite at the close and send market-on-open orders.
     - Before every new short, read `scripts/ibkr_shortstock.py`'s fee and availability.
   - **Monthly:** re-rank momentum on the month's last session. A live run needs an exchange calendar for
     that, or it decides a session late.
   - **Account:** a margin account (shorts), ~$33k for 30 slots of ~$1,000.
     - Whole shares cost 0.0-1.0 points a year (§4).
     - Keep gross at 0.90: GME's cover still part-filled there.
   - **Pre-register the stop rules now**, about execution and risk rather than one year's noisy alpha:
     - fills more than 50 bp a leg worse than the open, averaged over a month;
     - more than 3 of 15 intended shorts unborrowable or above 10% a year;
     - a drawdown beyond −25% (every test's worst was at most −21.9%);
     - after 12 months, a cumulative alpha below zero. That would have stopped the top-500 book in 2026.
2. **Decide what to do about the beta drift** if protection in a fall starting now matters:
   - Beta sizing cuts the drift by a third to two-fifths for −3.0 points of T1 alpha: 2026 ex-ante beta
     +0.40 instead of +0.65, realized 2026-H1 +0.48 instead of +0.73.
   - A full fix would hedge the measured ex-ante beta with an index short at each rebalance. That is
     untested here: xs_three_book's 8% SPY cap could not hold beta at 0.15.
   - Pre-register either as a risk variant and paper-trade it beside A_U0.
3. **Engine proposals #46-#52: approved and implemented on 2026-10-24** (§9). They change results for any
   portfolio-mode book that runs near full gross or meets a squeeze. informed_flow_ls runs at gross 0.98:
   its study should be bumped and its books re-run by its reviewer (§10.2).
4. **Optional risk variants, each needing its own pre-registration:**
   - the sector-neutral composite: −16.5% instead of −21.9% drawdown, but half the 2022 gain;
   - the portable-alpha wrapper, for an account that must also hold the market. It was tested only around
     the side-by-side C_core overlay and sized for ~$150k.
5. **Do not:**
   - re-tune the momentum weight;
   - add profitability as a third rank;
   - revive peer_reversal or earnings_drift at retail order sizes;
   - read T4 or TL again for selection: they are spent.

## 13. Where everything is

- **Harness:**
  - `strategies/proposals_lab/` (`lab.py`, `README.md` with the protocol and lab v2);
  - `FINAL_DECLARATION.md` (the declaration and the locked results);
  - `final_batch.py`.
  - Results and panels are in `var/proposals_lab/` (git-ignored; `results2/` is lab v2).
- **Proposals:** `strategies/{resid_momentum, adaptive_hedge, quality_junk, peer_reversal, earnings_drift,
  etf_trend}/`.
- **Cross-strategies:** `strategies/{xs_si_mom_composite, xs_three_book, xs_cross_leg, xs_portable_alpha,
  xs_three_premia}/`.
- **Proposal 1:** `strategies/informed_flow_ls_p1/` (research copy) and `informed_flow_ls/RESEARCH_LOG.md`
  §38P.
- **Tests:** `tests/test_proposals_lab.py` and one `tests/test_<folder>.py` per folder.
- **The post-lock checks of 2026-10-24** (reconciliation of all 11 locked books, 2× borrow, whole shares,
  the IBKR borrow check, betas, beta sizing) selected nothing. Only the betas just read the saved results.
  - The reconciliation rebuilt every locked period from vectorbt's order log.
  - The 2× borrow, whole-share and IBKR checks re-ran T1, T2, T4 and TL (2× borrow also TRAIN; the IBKR
    check only each test's last half-year's signals).
  - Beta sizing ran in memory on the open tests.
  - The 2× borrow books are lab runs saved as `A_U0-B2` and `A_U0-B2-LOCK` in
    `var/proposals_lab/results2/xs_si_mom_composite/`; the rest are scratch scripts.
  - Beta sizing is reported in §4 and `xs_si_mom_composite/RESEARCH_LOG.md` §13.1, the others in
    `FINAL_DECLARATION.md`.
