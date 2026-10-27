# xs_three_book

The proposals lab's cross-strategy round (2026-10-23): **the three books with T1 selection skill
in one netted cash account, equal thirds.** It is a research folder for the shared lab
(`strategies/proposals_lab`, lab v2). `RESEARCH_LOG.md` holds the pre-registration and every
number.

> **Status (2026-10-23, lab v2, OPEN tests only): works.** By the lab's gates both the design
> (U0, three books) and its declared final pass every applicable gate (G1-G8, G7 applied).
> **Declared final: CRISIS = `{"sleeves": "mom,ah"}`.** That is residual momentum plus the
> adaptive hedge, 0.45 each, chosen by the pre-registered challenger rule.
>
> - **The margins over U0 are within the noise** out of sample: T1 +0.08, T2 +0.28 (P 0.58).
> - **Most of G1 is AH's in-sample tune.** AH was tuned on TB + T3; untuned, the three-book TRAIN
>   is +0.74%.
> - **T1 is partly in-sample** for SI and for AH's long rule.
> - **TL is the orchestrator's one confirmatory read**, and T4/TL were not read here.
>
> | | TRAIN (TB + T3) | T1 top 500 | T2 names 501-1000 | T1 beta | T1 maxDD | 2018-26 CAGR |
> |---|---|---|---|---|---|---|
> | CRISIS (final) | +5.22% (P 0.84) | +9.86% (P 0.993) | +4.80% (P 0.88) | −0.18 | −12.5% | 6.2% |
> | U0 (design) | +3.88% (P 0.82) | +9.78% (P 0.995) | +4.52% (P 0.91) | −0.08 | −11.8% | 6.6% |
> | C_core (SI + MOM) | +2.00% | +9.36% | +4.31% | +0.02 | −12.3% | 6.7% |
> | equal-weight market | | CAGR 8.7% | CAGR 7.6% | | −25% | 13.8% |
>
> Alpha = CAGR − beta × buy-and-hold CAGR.
>
> - **Falling markets** (CRISIS / U0 / market):
>   - Q4 2018 −3.1 / −4.6 / −18.1;
>   - COVID +11.8 / +7.5 / −37.0;
>   - 2022 bear +18.8 / +15.8 / −24.5;
>   - Feb-Apr 2025 +3.6 / +1.3 / −19.5.
> - **It beats buy-and-hold CAGR only in the falling years** (2018-H2 and 2022; U0 also in
>   2025). Neither book beats it over 2018-26. CRISIS does not beat it even on T1 (8.3% vs
>   8.7%); U0 does narrowly (9.1%).

> **Locked read (2026-10-24, run once by the orchestrator; `strategies/proposals_lab/
> FINAL_DECLARATION.md`): CRISIS fails on the smallest names.**
>
> - T4 (names 1001-1500): +4.96% (P 0.94).
> - TL (names 1501-2000): −0.03% (P 0.54), late fills −2.16%, max drawdown −30%.
> - Its locked books were rebuilt from vectorbt's order log: no order was refused or
>   part-filled, so the TL failure is real.
> - The overlay control C_core (SI + MOM, no AH) read T4 +5.14% (P 0.96) and TL +7.88% (P 0.99).
> - The overall final, xs_si_mom_composite A_U0, read T4 +12.35% and TL +8.56%.

## What the lab found (RESEARCH_LOG.md §1-§8)

- **Execution is clean.**
  - No order was refused or part-filled, and no position stuck, in any label (27 periods each).
  - At a third of the account, the Wayfair cover that lab v2 refused inside adaptive_hedge's own
    book filled.
  - Each sleeve run alone rebuilds its component's saved lab book exactly.
- **Netting and the engine cost nothing.** The engine is within 0.1 pt of the return-sum preview
  on T1, T2 and T3, and 1.0 pt better on TB (the Wayfair cover).
- **AH earns its third** (bar 3): U0 − C_core is +1.89 on TRAIN and +0.21 on T2.
- **AH's crisis alpha is name selection, not breadth timing** (bar 4). Holding its short slots
  as SPY (C_idx) loses 2.3 / 3.4 / 1.0 pts on TRAIN / T1 / T2.
- **The account does not beat its best sleeve everywhere** (bar 2 fails).
  - On T2, MOM alone has the better alpha ÷ vol.
  - On T1 and T2 the account's alpha is slightly below the sleeves' mean, because it holds 0.945
    of their exposure.
- **I1 (vol scale + 8% SPY beta cap) is rejected.** It cuts T1 drawdown from −11.8% to −8.0%
  for −0.3 pt of alpha. But the cap cannot hold beta ≤ 0.15 when the overlay's month-end beta
  reaches 0.42 (2021) or 0.30 (2026).
- **I2 (AH's short leg alone as a crisis sleeve) is rejected:** −0.3 / −0.5 / −1.6 on TRAIN /
  T1 / T2.
- **Diversification decays out of sample.** Inside the account, the sleeves' daily correlations
  go from −0.26 to +0.58 on TRAIN to +0.40 to +0.70 on VALIDATION. All three lost in 2026-H2.

## The account

| sleeve | component (lab label) | what it holds | budget |
|---|---|---|---|
| SI | `informed_flow_ls_p1` (`v6_si`) | long the names short sellers avoid, short the ones they crowd: 15 + 15 slots | 0.30 of the account |
| MOM | `resid_momentum` (`U0v2`) | residual 12-1 momentum: 20 + 20 slots, its own volatility scale | 0.30 |
| AH | `adaptive_hedge` (`TUNEDv2`) | the SI long rule plus breakdown shorts held ~300 days: 30 slots | 0.30 |

- **Each component runs unchanged.** Its `signals()` runs from its own `strategy.py` and
  `config.yaml`, loaded by path under a prefixed module name, on the panel without SPY.
  `strategy.SHA256` pins the six files, and a test checks the pin.
- **Each sleeve is scaled to its budget** over its own nominal gross (0.98 / 0.90 / 0.98).
- **The sleeves are summed per name.**
  - A long in one and an equal short in another cancel: no position, no borrow.
  - Same-side positions add (SI and AH share the long rule).
- **Nothing is rebalanced or estimated.** A name trades only when some sleeve changes its
  target.
- **Nominal gross is 0.90:** the headroom `lock_cash` needs.
- **Coverage switch.** The SI and AH sleeves go to cash on the bar after a decision bar where
  fewer than half the names closing at or above $5 carry a days-to-cover. It never binds on
  the open tests; it guards E0 and TL.
- **Borrow** is the dearest sleeve's tier. `retrim_k` is 2. Fills are at the next open, with
  `order_notional: 1000` (the declared account is about $100k: an SI or AH slot ≈ $1,000).

**Switches** (all off in U0):

| switch | what it does |
|---|---|
| `sleeves`, `budgets` | which books, and their budgets |
| `ah_variant: u0` | AH without its tune |
| `ah_part: shorts` | AH's short leg alone, 0.9% a slot (iteration I2) |
| `ah_hedge: SPY` | AH's short slots held as SPY: same breadth timing, no names (control C_idx) |
| `book_vol`, `beta_cap`, `hedge_max` | the risk layer (iteration I1): a month-end volatility scale on the overlay, plus a SPY short that caps its ex-ante beta at 0.15 (at most 8%) |

## Files

| file | what |
|---|---|
| `strategy.py` | `signals()`, `valid()`, `compose()` (every intermediate), the component loader |
| `config.yaml` | U0, the pre-registered default; nothing is searched |
| `research.py` | `audit` (mechanics, no returns), `reproduce` (each sleeve alone rebuilds its component's saved book), `diagnose` (order logs: refusals, stuck positions and their P&L, sleeve contributions, order sizes at $100k), `preview` (return-sum of the components' saved books), `table`, `noise`, `components` |
| `RESEARCH_LOG.md` | the pre-registration (§0, SHA-256-snapshotted in `var/proposals_lab/results2/xs_three_book/prereg/`), every run, the declared final |
| `IDEAS.md` | parked ideas |
| `tests/test_xs_three_book.py` | synthetic-panel tests |

The tests cover:
- the sum identity with the components' own books;
- netting, and SPY isolation;
- causality on truncated panels;
- the coverage switch, and the gross bound;
- I1's steps and month-end timing, and I2's slot size;
- borrow, the lab runner holding every target, and the file pins.

Run everything through the lab, from the repo root, every label with `--hedge`:
`uv run python strategies/proposals_lab/lab.py run xs_three_book U0 '{}' TB,T3,T1,T2 --costs2 --hedge`

- Name the tests explicitly: the CLI reads its fifth argument as the test list.
- E0 is never run: there is no short interest before 2018. An E0 book would be the MOM sleeve
  alone: about −2.7% for U0 (MOM × 1/3) and −4.1% for CRISIS (MOM × 1/2), from MOM's saved E0
  book.
- Labels and overrides: `U0` `{}`, `CRISIS` `{"sleeves": "mom,ah"}`, `C_core` `{"sleeves":
  "si,mom"}`, `C_idx` `{"ah_hedge": "SPY"}`, `C_ahu0` `{"ah_variant": "u0"}` (TB, T3), `I1`
  `{"hedge_max": 0.08, "book_vol": 0.10, "beta_cap": 0.15}`, `I2` `{"budgets": "0.36,0.36,0.18",
  "ah_part": "shorts"}`.
- The TL-read controls are `S_si` / `S_mom` / `S_ah` = `{"sleeves": "si"}` / `"mom"` / `"ah"`.
- Results and logs are in `var/proposals_lab/results2/xs_three_book/`.

**Limits.**
- The DB holds survivors only.
- `sector` is today's classification, and the $5 floors read split-adjusted closes (lab
  caveats).
- $1,000 orders are the declared ~$100k account's slot. Orders under $250 are re-priced at the
  $1 minimum in `research.py diagnose` (≤ 0.06 pt a year).
- A headless `tsim optimize` would run this on the SHARED engine (`call_seq="auto"`), which
  starves new shorts under `lock_cash`. The lab's runner orders covers first. Engine fix:
  proposal-only (resid_momentum IDEAS.md).
- The component files are pinned by SHA-256. If a component changes, the tests fail on purpose,
  and every label here must be re-run.
