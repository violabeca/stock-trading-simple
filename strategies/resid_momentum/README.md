# resid_momentum — crash-protected residual momentum, long/short

Proposal 2 of `STRATEGY_OVERVIEW.md` §6, built and judged in the shared proposals lab
(`strategies/proposals_lab`, lab v2). Daily, prices only; short interest and share counts are
read for the borrow fee alone. **Research status and every number: `RESEARCH_LOG.md`** (round 2
= "Round 2: lab v2 re-evaluation").

> **Status (2026-10-23, round 2 on lab v2): FAILS G1, the TRAIN-era gate, so by the lab's
> definitions it is neither "works" nor "works to some degree".** The design is the untuned
> pre-registered default; its one change is gross 0.90, an execution fix decided on order counts
> only. It passes the other seven gates. Alpha against the equal-weight market:
> - **TRAIN 2016-07 → 2021-12: −1.4%/yr (P 0.37)**, and negative under every stress: late fills
>   −3.3%, 2× costs −1.8%, the three noise copies −1.4 to −2.1%.
> - T1 (top 500, 2022 → 2026-10): +8.7% (P 0.98, beta −0.10, 100th luck percentile, without its
>   top 5 names +6.4%).
> - T2 (names 501-1000): +5.5% (P 0.93).
> - Locked T4 (names 1001-1500), its second read on corrected accounting: +6.2% (P 0.95).
>
> It beat the market in all four bear episodes (Q4 2018 −4.8% vs −18.2%, COVID +4.0%, 2022
> +11.8%, Feb-Apr 2025 +6.6%). **The edge is a 2022-2025 regime:** 2016-2021 is negative, and
> 2026-H2 is negative on T1, T2 and both 2026 halves of T4.
>
> Pre-registered crash bar (b) (Jul-Sep 2026 loss at most half of raw momentum's) **FAILED in
> round 1 (61%)**. It passes on lab v2 (42%), but the flip comes from the control arm's path, so
> it is fragile. The crash protection that holds is the drawdown: T1 −9.5% vs −18.8% unscaled
> and −25.1% for raw momentum.
>
> Round 1 reported "passes all eight gates". That came from lab v1, which starved new shorts
> and dropped each period's first day. G1 never had a margin (P 0.51 then).

## Thesis

Momentum is the most persistent cross-sectional anomaly, but raw momentum is partly a market
bet: after a fall the losers are the high-beta names, so the short leg rebounds violently and the
book crashes (2009, Nov 2020). Ranking the return LEFT after removing each name's market beta,
divided by its residual volatility (Blitz, Huij & Martens 2011), keeps the stock-specific trend
and drops the beta bet. Scaling the whole book down when its own trailing volatility rises
(Barroso & Santa-Clara 2015) cuts exposure before most of the remaining crashes.

What the lab found:
- Residual ranking plus the volatility scale keep raw momentum's T1 alpha (+8.7% vs +8.2%) at
  38% of its drawdown (−9.5% vs −25.1%). The residual ranking alone takes it to −18.8%.
- The book's shorts are still genuinely high-beta names (ex-post beta about 1.2). Half to two
  thirds of the short leg's loss in a junk rally is that beta (`RESEARCH_LOG.md` R2.6).

## Rules (`strategy.py`, defaults in `config.yaml`)

1. Residual return `e = r − β·m`: `m` the equal-weight daily return of the panel's names priced
   at or above $5 the bar before, `β` each name's trailing 252-bar OLS slope on `m`.
2. Score = `Σe / sd(e)` over bars `t−251 .. t−21` (12-1). The control arm `score: raw` ranks
   the 12-1 total return instead.
3. Decisions on the last trading bar of each month only, filled at the next open. Long enters at
   rank ≥ 0.90 and leaves below 0.80; short enters ≤ 0.10 and leaves above 0.20. 20 + 20 slots,
   equal, dollar-neutral slots, gross 0.90. Pre-registered 0.98: at 0.98 the engine cannot fill
   a full book's last long buys, because vectorbt locks a winning short's gain twice in its free
   cash. Every order fills at the first open after the decision.
4. Book scale = `min(1, 10% / σ)`, `σ` the unscaled book's trailing 126-bar realized volatility,
   decided at month ends only, rounded to 0.25, floor 0.25 (`vol_target: 0` = off).
5. Borrow: informed_flow_ls_p1's tiers by short interest / shares outstanding (0.5% / 3% / 10% a
   year; 0.5% with no data); never short above 30%. Longs are trimmed at 2× their slot
   (`retrim_k`).

Eligibility: close ≥ $5, ≥ 252 bars of history, ≥ 80% of both windows observed. Holiday stub
rows (a bar only a few names print) carry no decision and no order.

Options that are OFF:
- `factors: market+sector`: the residual also drops the sector's own trend.
- `sizing: beta`: informed_flow_ls_p1's median-beta ÷ beta slots, frozen while held.
- `phased: true`: closes at the first open, opens at the second. It is a workaround for the
  SHARED engine (`backtest._run_portfolio`, a headless `tsim` run), whose `call_seq="auto"`
  places new shorts before the covers that fund them. The lab's runner orders covers first.

The first two failed their pre-registered bars in both rounds (`RESEARCH_LOG.md` §4-§5, R2.2).

**Limits.**
- *Costs:* $1,000 orders are the median order of a ~$45k-95k account, not of $40k. On $40k the
  $1 minimum costs 0.04-0.36% of the book a year more; the 2×-cost runs bound it.
- *Split basis:* the $5 filter and the short-interest share read split-adjusted data (the lab
  panels carry no adjustment factor; measured immaterial).
- *Survivorship:* the DB holds survivors only, so the short leg looks worse than it was before
  2020.
- *Live use:* a month's last bar is known only once the next exists. A live run needs an
  exchange calendar, or it decides a session late.

## Files

- `strategy.py`: `signals()` / `valid()` / `compose()`, everything the book reads.
- `research.py`: the TRAIN premise check, the mechanics audit on the lab's runner, the order
  sizes, the leg, beta and crash-window diagnostics. It is loaded by path and never imported by
  `strategy.py`.
- `RESEARCH_LOG.md`: the pre-registrations, every run, both rounds' verdicts.
- `tests/test_resid_momentum.py` covers:
  - causality on a truncated panel;
  - the residual removing the beta bet, and the sector regressor;
  - bands, the quarter-step scale, borrow tiers and holiday stubs;
  - the lab runner holding every target;
  - the two cash-lock mechanisms: covers before new shorts, and winning shorts' locked gains.

Run it through the lab (v2):
`uv run python strategies/proposals_lab/lab.py run resid_momentum U0v2 '{}' E0,TB,T3,T1,T2
--costs2`. Name the tests explicitly: a bare `--costs2` in the tests slot is read as a test name.
