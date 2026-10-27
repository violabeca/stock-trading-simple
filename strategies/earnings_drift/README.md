# earnings_drift

Proposal 6 of `STRATEGY_OVERVIEW.md` §6 ("also considered"): **post-earnings-announcement drift as
a dollar-neutral long/short book.** Stocks are supposed to keep drifting in the direction of their
earnings news for weeks after the report (Bernard & Thomas 1989). The market's own reaction is
supposed to be a cleaner measure of that news than the EPS surprise (Brandt, Kishore, Santa-Clara &
Venkatachalam 2008).

It was built and judged on 2026-10-22 with the shared lab (`strategies/proposals_lab/`). On
2026-10-23 it was fixed after the round-1 review and re-judged on lab v2 (round 2). Every step, with
its numbers, is in `RESEARCH_LOG.md`.

| file | what it does |
|---|---|
| `strategy.py` | the book: scored events → 30 long + 30 short slots, held 60 sessions (pure) |
| `config.yaml` | every parameter (the pre-registered default plus the cent rule; no variant was adopted) |
| `premise.py` | the TRAIN-only premise check: rank IC and fifth spread against beta-adjusted forward returns |
| `event_time.py` | TRAIN only: where after an event its drift sits, and the next report's reaction |
| `conditions.py` | TRAIN only: the fifth spread split by announcement volume, liquidity, SUE/EAR agreement, crowding |
| `attribution.py` | a config's long and short legs, per test and period (gross) |
| `diagnostics.py` | every other number the log quotes: the EPS precision, costs and exposure, the lab-v1 → v2 decomposition, placebos, the tuner re-check |
| `RESEARCH_LOG.md` | pre-registration, every variant, every number, the verdicts of round 1 and round 2 |

Run any helper with `uv run python strategies/earnings_drift/<file>.py`. The book itself runs
through the lab: `uv run python strategies/proposals_lab/lab.py run earnings_drift <label> '{}'
E0,TB,T3,T1,T2 --costs2`. The tests must come before the flag.

## How a trade works

1. **An event** is a yfinance earnings report. In the lab's frames it sits on its *usable bar*:
   the second session after the report session. It is decided at that bar's close and traded at
   the next open.
2. **Two measures of the news:**
   - SUE = (EPS actual − estimate) ÷ the close before the report;
   - EAR = the stock's return from the close before the report session to the close of the
     session after it, minus the equal-weight market's.
3. **Coarse surprises are not used.** yfinance stores EPS on TODAY's share basis, rounded to the
   cent. A name that split later therefore has tiny early EPS: NVDA's 2016 actual and estimate
   are both $0.01. One cent can then be an extreme surprise.
   - The quantum of a cent is 0.01 ÷ the close under `price` and 0.01 ÷ |estimate| under `pct`.
   - Where it is more than a quarter of the trailing P20-P80 spread of the surprise, the event
     ranks on its reaction (EAR) alone.
   - The same holds where the trailing window is still too small to judge (a panel's first ~90
     events).
4. **Point-in-time ranks:** each measure is ranked against the events of the trailing 252
   sessions. The two ranks are averaged, and that score is ranked against the trailing scores
   again. The top fifth goes long, the bottom fifth short.
5. **The book:** 30 slots a side, each bought at 0.98/60 of the book (dollar-neutral when full).
   - The most extreme events take the free slots first. An event that finds none is dropped.
   - A position is held 60 sessions, and the name's next report re-decides it.
   - Held positions are not re-sized; only a long grown to 2× its slot is trimmed. So 0.98 is
     the gross at entry, and the realized gross drifts: about 0.9 on average, up to 1.29 on T1.
6. **Shorts** pay 0.5% / 3% / 10% a year by days-to-cover (> 7 and > 10 days). A name above 15
   days is never shorted.
   - This uses days-to-cover, not short interest ÷ shares. That ratio leaks future splits in the
     lab's data: volumes are split-adjusted backward, SEC share counts are as reported.

## Parameters

| knob | default | intuition |
|---|---|---|
| `surprise` | price | `price` = surprise ÷ price; `pct` = yfinance's % surprise (explodes near zero estimates) |
| `w_ear` | 0.5 | EAR's share of the score: 0 = surprise alone, 1 = the market's reaction alone |
| `ear_end` | 1 | the reaction window ends this many sessions after the report session (≤ 2) |
| `trail` | 252 | sessions of past events the ranks are taken against |
| `min_events` | 100 | scored events the trailing window needs before it ranks |
| `q` | 0.2 | the share of the trailing scores that goes long (top) / short (bottom) |
| `hold` | 60 | sessions a position is held |
| `slots` | 30 | positions a side |
| `gross`, `long_share` | 0.98, 0.5 | exposure at entry with full slots; the long side's share |
| `sides` | both | `long` / `short` run one side (diagnostics) |
| `dtc_mid`, `dtc_high`, `short_max_dtc` | 7, 10, 15 | days-to-cover tiers for borrow; no shorts above the cap |
| `borrow_gc`, `borrow_mid`, `borrow_high` | 0.5%, 3%, 10% | yearly borrow rates (informed_flow_ls_p1's) |
| `min_price` | 5 | minimum close at the decision |
| `retrim_k` | 2 | a long grown to 2× its slot is trimmed back |
| `crowd` | all | `crowded` / `uncrowded`: open positions only above / at most the bar's median days-to-cover (iteration I2, rejected) |
| `cent_max` | 0.25 | the surprise counts only where one cent is at most this share of its trailing P20-P80 spread (`.inf` = no rule) |

## Data caveats

- yfinance covers about 80-90% of name-quarters, and the data **end on 2026-06-11**. The last
  test period (2026-07 → 10-16) gets no new events: only positions opened in May and June run
  into it, and then the book sits in cash.
- EPS are on today's share basis and rounded to the cent: see the cent rule above.
- The lab's closes are split- and dividend-adjusted backward, and so is its $5 universe floor.
  Some future splitters are missing from the early TRAIN universes; NVDA is out of all of E0.
- The DB holds survivors only, so shorts look worse and longs better than reality.
- A headless `tsim optimize` has no earnings frames (no `features.py` builder serves them), so it
  runs an empty book.

## Status (2026-10-23, round 2 on lab v2): no edge

The final candidate is the pre-registered default plus the cent rule (U0v2). No variant passed
its bar, in round 1 or on lab v2. Alpha = CAGR − beta × the equal-weight buy-and-hold of the same
names. Lab v2 is the lab with the five accounting and data fixes of its README's "Lab v2" section.

| test | names | alpha | P(α > 0) | late fills | 2× costs | beta |
|---|---|---|---|---|---|---|
| TRAIN era (2016-07 → 2021-12) | top 500 | −3.3% | 0.02 | — | — | ≈ 0 |
| T1 (2022 → 2026-10) | top 500 | +2.2% | 0.81 | −2.5% | +1.1% | −0.01 |
| T2 | names 501-1000 | −0.8% | 0.37 | −5.4% | −1.9% | −0.03 |
| T4 (locked; third read, a re-measurement) | names 1001-1500 | −3.4% | 0.05 | −7.9% | −4.5% | −0.02 |

- **Luck:** T1 sits at the 93rd percentile of 100 shuffled-name books. G4 needs the 95th.
- **Gates:** 3 of 8 pass: G6 (bear episodes and down months), G7 (market neutrality) and G8 (T1
  without its top five names, +0.6%). G1-G5 fail.
- **Bear markets:** in every sell-off the book lost at most 2.3%, and it made +4.5% in 2022. Over
  the same episodes (Q4 2018, COVID, 2022, Feb-Apr 2025) the market lost 18-37%. That is
  neutrality, not alpha: over 2016-26 the book compounded at −0.8% a year (max drawdown −22%).
- **Costs:** about 575 orders a year, each ~1.6% of the book. That is ~215 entries, as many
  exits, and ~125 half-yearly re-buys of the carried-in book, which lab v2 charges. Costs take
  1.1 pts a year at base, and +50 bp late fills take 4.6 more.

The variants, measured on lab v2 against the default. An iteration needs a T1 Δ above +0.36:

- **Tuned on TRAIN** (T0f's values: price, w_ear 0.25, q 0.30, hold 50, 10 slots): +6.4 pts
  in-sample, +0.6 on T1, −2.5 on T2. It is not adopted, because it must beat the default on T1
  AND T2. Round 1's tuner pick is still the best of its study's top 10 on lab v2.
- **Hold 20:** the TRAIN drift is front-loaded (5-10 sessions). But a 20-session hold leaves the
  slots half empty between earnings seasons (gross 0.48 vs 0.87) and raises orders by ~40%. It
  loses 2.8 pts on T1 and 1.5 on T2, and it is worse per unit of gross too.
- **Crowded names only:** +2.7 / −2.2 / −0.3 pts on TRAIN / T1 / T2.
- **The market's reaction alone:** −3.9 pts on T1.

**What the cent rule does.** On lab v2 it lifts T1 by 1.9 pts and T2 by 0.8, and costs TRAIN 1.0.
The T1 gain comes from the ~60 coarse events a test that enter the book on their reaction alone,
mostly low-priced names (round 1's count). It beats all 20 placebos that re-score random events
the same way. TRAIN does not show it, and neither did T4 (round 1's two reads: −0.3 pt with the
rule). Without the rule, in round 1, the coarse % surprise carried ~83% of the tuned book's
in-sample alpha.

**What lab v2 changed:**
- about −0.3 pt a year of accounting (day 0's costs; starved new shorts are now filled);
- path noise from the ETF exclusion from 2024-H2 (T1 −0.4, T2 −0.3);
- one gate: G4 fell from the 96th to the 93rd percentile.

No decision changed. On TRAIN, what drift there is ends within about 20 sessions, and the next
report then partly reverses it.
