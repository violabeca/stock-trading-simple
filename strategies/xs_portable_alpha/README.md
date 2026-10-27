# xs_portable_alpha

The proposals lab's cross-strategy round (2026-10-23), **the absolute-performance design: a
constant SPY core (0.45 of the book) plus the SI + MOM market-neutral overlay (0.45), in one cash
account.** It is a research folder for the shared lab (`strategies/proposals_lab`, lab v2).
`RESEARCH_LOG.md` holds the pre-registration and every number.

> **Status (2026-10-24, lab v2, OPEN tests only): works to some degree.** The design's
> pre-registered bars all pass except A1 on T2. **Declared final: U0 = `{}` with `--hedge`.**
> T4 and TL were not read here; the orchestrator reads TL once.
>
> | | TRAIN (TB + T3) | T1 top 500 | T2 names 501-1000 | 2018-07 → 2026-10 |
> |---|---|---|---|---|
> | CAGR (buy & hold / SPY) | 9.9% (20.9% / 19.4%) | **10.3%** (8.7% / 11.9%) | **7.56%** (7.62% / 11.9%) | 10.1% (13.7% / 15.0%) |
> | alpha net of 0.45 × SPY's own alpha (P) | +1.21% (0.67) | **+4.96% (0.994)** | **+2.40% (0.92)** | +3.40% |
> | max DD (buy & hold) | −14.7% (−37.4%) | −8.1% (−25.0%) | −9.5% (−24.0%) | −14.7% (−37.4%) |
> | Sharpe (buy & hold / SPY) | 0.99 (0.96 / 0.94) | 1.11 (0.54 / 0.74) | 0.88 (0.46 / 0.74) | 1.06 (0.73 / 0.83) |
>
> - **Falling markets** (book / equal-weight market): Q4 2018 −11.7 / −18.1, COVID −14.5 /
>   −37.0, 2022 bear −6.8 / −24.5, Feb-Apr 2025 −7.7 / −19.5. All four are beaten, at 28-64% of
>   the market's loss. Down months −1.16% vs −4.11%: negative by design (beta 0.4).
> - **A1 fails on T2:** CAGR 7.56% vs buy-and-hold's 7.62%. The three noise copies miss it by
>   0.5-0.6 pt.
> - **No c on the a-priori frontier passes both A1 on T2 and the drawdown bar on TB.** c = 0.60
>   wins T2's CAGR but has 0.53 of buy-and-hold's TB drawdown, against a bar of ½.
> - **I1 (vol-managed core) is not adopted.** It is the better risk book (COVID −1.1%, 2022
>   +2.1%), but its CAGR falls and A1 fails on T2 (6.3%).
> - **No hedged design keeps up with a strong bull** without leverage: T3 17% vs 35%.
> - **Caveats:**
>   - T1 is partly in-sample for SI.
>   - Raw alphas carry SPY's composition alpha against the equal-weight benchmark (+2.0 / +2.8
>     pts on T1 / T2); the net column removes it.
>   - TRAIN is TB + T3. An E0 estimate (SI silent) is −2.3% net.
>   - Every lab book restarts each half-year, so a live account needs a core rebalancing rule
>     (IDEAS.md 3).

> **Locked read (2026-10-24, run once by the orchestrator, disclosure only).**
> `final_batch.py` printed raw alpha. The net numbers below are `research._stats` on the saved
> `U0-LOCK` label, called directly because the CLI refuses locked tests.
>
> | U0 | T4 (names 1001-1500) | TL (names 1501-2000) |
> |---|---|---|
> | raw alpha (P) | +4.89% (0.98) | +3.43% (0.89) |
> | net of 0.45 × SPY's alpha (P) | +2.78% (0.96) | +4.21% (0.99) |
> | net, late fills | +1.37% | +2.61% |
> | A1: CAGR vs equal-weight buy-and-hold | 7.7% vs 10.2%: fails | 8.6% vs 19.4%: fails |
> | max DD vs buy-and-hold's | −9.8% vs −26.1% (0.37) | −9.9% vs −22.1% (0.45) |
> | Sharpe (buy-and-hold) | 0.92 (0.57) | 1.00 (0.98) |

## The account (U0, `config.yaml`)

| part | what | weight |
|---|---|---|
| core | SPY, the lab's `--hedge` column (`lab.with_spy`). It is bought on each period's first bar and held: a constant target is never re-sent, so it is never re-trimmed (about 2 orders a year). It is never ranked. | 0.45 |
| overlay | `xs_three_book.signals(panel, sleeves="si,mom", gross=0.45)`: SI = `informed_flow_ls_p1` (`v6_si`), long the names short sellers avoid and short the ones they crowd; MOM = `resid_momentum` (`U0v2`), residual 12-1 momentum. Each runs unchanged on the panel without SPY, at 0.225 of the account, netted per name. | 0.45 |

- **Gross and costs.** Nominal gross is 0.90: the headroom `lock_cash` needs. Borrow is the
  overlay's; `retrim_k` is 2. Fills are at the next open; the lab's SPY column has open = close,
  so the core fills at the fill bar's close.
- **Order size.** `order_notional: 1000` matches the declared ~$150k account: overlay slots of
  0.56-0.75%.
- **Iteration I1 (`core_vol: true`, not adopted).** SPY at 0.45 × m. m = 1 / 0.5 / 0 as SPY's
  21-session variance at the month-end close is at most its trailing 252-session median / at most
  twice it / above. It is decided at month ends, and the freed gross sits in cash.
- **The pins.** xs_three_book's `strategy.py` is loaded by path under `_xspa_tb`.
  `strategy.SHA256` pins it and the six component files it reads, and a changed file refuses to
  load. If a component changes, every label here must be re-run.

## What the lab found (RESEARCH_LOG.md §1-§8)

- **Execution is clean.**
  - Nothing was refused, part-filled, dropped or stuck in any label or period, nor in U0's and
    I1's late-fill and 2×-cost books. In U0's and I1's audits free cash never fell below 6.5% of
    the book.
  - The account at `core: 0` rebuilds xs_three_book's saved `C_core` exactly.
  - The engine is within 0.2 pt of the judge's return-sum preview on T1. TB is −0.9: the core is
    held, not rebalanced, through the COVID V.
- **The c-frontier** (information only; `frontier_table.log`). Each 0.15 of core adds about
  1.1 pt of 2018-26 CAGR and takes about 1.5 pt off T1's net alpha. Return ÷ risk peaks near
  c = 0.30 (Sharpe 1.12).
- **The lab's own verdict** (`--not-neutral`, raw alphas) passes G1-G5 and G8. G6 fails on its
  down-month leg, by construction.

## Files

| file | what |
|---|---|
| `strategy.py` | `signals()`, `valid()`, `core_scale()` (I1's rule), the pinned loader `book()` |
| `config.yaml` | U0, the pre-registered default; nothing is searched |
| `research.py` | `audit` (mechanics and invariants, no returns, base / late / 2× books), `reproduce` (core 0 = C_core), `diagnose` (order logs: refusals, stuck positions, legs, hair orders at $150k), `preview` (the judge's return sum and the engine gap), `table` (CAGR vs buy-and-hold and SPY, net alpha, the core-only book, episodes, drawdown ratios, the stitched record), `bars` (A1-A4, I1's adoption rule), `e0`, `noise` |
| `RESEARCH_LOG.md` | the pre-registration (§0, SHA-256-snapshotted in `var/proposals_lab/results2/xs_portable_alpha/prereg/`), every run, the declared final |
| `IDEAS.md` | parked ideas: the CRISIS overlay, an equal-weight core, a core rebalancing rule |
| `tests/test_xs_portable_alpha.py` | synthetic-panel tests |

The tests cover:
- the core at 0.45 on every bar, bought once;
- I1's three values and its month-end timing, against a loop re-statement of the rule;
- the overlay equal to xs_three_book's book at 0.45, and to half its C_core;
- core 0 = C_core, and the frontier;
- SPY never in a sleeve's input;
- the 0.90 gross bound;
- causality on truncated panels;
- borrow;
- the lab runner holding every target;
- `valid()`, the config and the SHA-256 pins.

Run everything through the lab, from the repo root, every label with `--hedge`:
`uv run python strategies/proposals_lab/lab.py run xs_portable_alpha U0 '{}' TB,T3,T1,T2 --costs2 --hedge`

- Labels: `U0` `{}`, `I1` `{"core_vol": true}`, and the frontier `F15` / `F30` / `F60` =
  `{"core": 0.15}` / `0.30` / `0.60`.
- Verdict with `--not-neutral`.
- The design's bars: `uv run python strategies/xs_portable_alpha/research.py bars U0`.
- Results and logs are in `var/proposals_lab/results2/xs_portable_alpha/`.

**Limits.**
- The DB holds survivors only.
- `sector` is today's classification, and the $5 floors read split-adjusted closes (lab
  caveats).
- **A headless `tsim optimize` cannot run this:** Liquid500 has no SPY column, and the shared
  engine's `call_seq="auto"` starves new shorts under `lock_cash`. The lab's runner orders covers
  first; the engine fix is proposal-only (resid_momentum IDEAS.md).
- **The net-of-core alpha is an approximation.** `table` also reports alpha net of a core-only
  book. The two differ by −1.0 to +0.2 pt.
