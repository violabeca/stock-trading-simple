# xs_si_mom_composite — SI × MOM composite

A cross-strategy of the proposals lab's last round (`strategies/proposals_lab`, lab v2): one
dollar-neutral book ranked on **short interest plus residual momentum**, integrated at the signal
level. The question it answers: does ONE ranking of two nearly orthogonal signals beat holding
the two books side by side (the `C_mix` control) and beat each sleeve alone?

This is a research folder. `RESEARCH_LOG.md` holds the pre-registration (§0, SHA-256-snapshotted
in `var/proposals_lab/results2/xs_si_mom_composite/prereg/`) and every number.

> **Status (2026-10-23, lab v2, OPEN tests only; review fixes in RESEARCH_LOG §12): works — all
> eight lab gates pass, and all four pre-registered design bars pass, bar (2) only thinly on
> T1.** Declared final: `A_U0` (config as shipped, nothing tuned; unchanged by the review). T4
> and TL are not read here; the orchestrator reads TL once.
>
> | | TRAIN (TB+T3) | T1 (top 500) | T2 (501-1000) |
> |---|---|---|---|
> | **A_U0** alpha (P) | +1.97% (0.62) | **+12.88% (0.962)** | +5.07% (0.84) |
> | C_mix (both books side by side) | +0.58% | +9.68% | +2.35% |
> | C_si / C_mom alone | +0.64% / −0.25% | +9.40% / +9.04% | +2.53% / +1.86% |
>
> - T1: beta +0.01, 100th luck percentile, +8.0% without its top 5 names, +10.8% with late fills.
>   Every bear episode beaten (2022 +15.4% vs −24.5%); down months +0.95% vs −4.12%.
> - **Integration vs holding both books** (paired Δ over C_mix): TRAIN +1.4 (P 0.65, 4/7 periods;
>   TB +3.6, T3 −0.6), T1 +3.2 (P 0.76, 7/10), T2 +2.7 as run and **+1.8 (P 0.69)** with T2's
>   stuck orders repaired. None of these is a confident win on its own (P 0.65-0.78).
> - **Bar (2) on T1 is thin:** the Δ clears the pre-registered 2 × noise bar by 0.29 pt (3.20 vs
>   2 × 1.45 = 2.91), and all three A_U0 noise copies land 1.6-4.2 pts below the label's T1.
>   C_mix's own noise copies (post hoc; they drop the same names) put the paired T1 Δ at +3.1,
>   +3.5 and −0.4 (mean +2.1, sd 2.2): integration leads on most paths, but its T1 margin is
>   about as large as its own path noise, and one path in three shows none. TRAIN copies +1.3 /
>   +3.9 / +3.4; T2 +0.9 / +5.4 / +4.1 (as run). The first reading here ("about level with C_mix
>   once its favourable path is discounted") discounted A_U0's path only; both books' T1 paths
>   turned out to be favourable draws.
> - **T2's one execution failure** (2024-05-14, the GME squeeze, a lock_cash limit): GME's cover
>   part-filled (60%) and the same bar's new short NTLA was refused; both stuck 32 bars (64
>   off-target name-days, 0.3% of T2's orders). The stuck short made +4.57% of the book from its
>   due open, so with both orders repaired T2 reads **+4.15% (P 0.78)**, not +5.07%. No gate or
>   bar flips (G3 > 0; bar (2) T2 Δ +1.8 ≥ 0; bar (3) 4.15 ≥ C_si 2.54, C_mom 1.86). Nothing else
>   in TB, T3, T1 or T2 was refused or part-filled for any label, apart from one negligible
>   C_si short (`audit_v2.log`). Against the components' own published books (context, not a
>   bar), the repaired T2 is 1.35 pts below resid_momentum U0v2 (+5.50%; P 0.42) and level with
>   v6_si (+0.20, P 0.53).
> - **TRAIN does not survive late fills:** +50 bp a leg takes TRAIN to −0.35% (T3 −4.74%). G5
>   reads T1 only (+10.8%), so this is not a gate failure, but the TRAIN case rests on open fills.
> - **Risk:** T1 volatility 16.3% and max drawdown −21.9% (2025-10 → 2026-07) are the highest of
>   the four books (C_mix 11.6% / −14.8%). Both 2026 halves lose ~22% a year of alpha: like its
>   sleeves, the edge is a 2022-2025 regime. 2018-07 → 2026-10 CAGR 8.9%, below buy-and-hold.
> - Rejected by their pre-registered bars (T1 noise): the profitability junk gate on shorts (I1,
>   +1.3 / +0.95 / +3.2), the sector-neutral composite (I2, +3.3 / −0.03 / +3.3, lower drawdown),
>   the tuned w_mom 1.5 (T1 −3.4, PBO 0.75 on 9 points × the last six TRAIN periods).
> - In-sample caveats: T1 is partly in-sample for SI (only MOM is clean there); TL is fresh only
>   for the combination (informed_flow_ls already read names 1501-2000 for SI); AH enters only as
>   an execution helper (`_on_priced_opens`), so its TB + T3 tuning does not touch this book.

> **Locked read (2026-10-24): CONFIRMED.** The orchestrator ran it once, after declaring this
> book the overall final in `strategies/proposals_lab/FINAL_DECLARATION.md`, together with the
> other designs' finals (disclosed there).
>
> | A_U0 | alpha (P) | late fills | beta | CAGR vs buy-and-hold | max DD | 2022 bear (market) |
> |---|---|---|---|---|---|---|
> | TL (names 1501-2000) | **+8.56% (0.963)** | +6.50% | −0.10 | 6.5% vs 19.4% | −13.3% | +6.2% (−16.2%) |
> | T4 (names 1001-1500) | **+12.35% (0.997)** | +10.14% | −0.09 | 11.5% vs 10.2% | −11.7% | +4.5% (−18.4%) |
>
> - **Execution reconciled:** every locked period was rebuilt with vectorbt's order log and
>   equals the saved returns. 0 orders were refused or part-filled and 0 name-days were off
>   target, so no stuck cover inflates either number, unlike GME's on T2.
> - **Luck (run after the read, supplementary):** A_U0 beat all 100 shuffled-name books on T4
>   (median −0.05%, 95th percentile +5.26%) and on TL (median −1.06%, 95th +4.98%).
> - **Integration over C_mix replicated on T4 only:** +8.0 (P 0.997) on T4, −0.1 (P 0.49) on
>   TL. On TL the short-interest book alone (C_si) made +10.9%.
> - **Half-years:** every one from 2022-H2 to 2025-H2 is positive on both sets. 2026 is mixed:
>   T4 +5.9 then −7.9, TL −12.8 then +9.1.
> - **The controls pass the same locked bar too:** C_si TL +10.9%, C_mix +8.7%, C_mom +6.3%. What
>   TL confirms is the SI + MOM family on fresh names, not the composite's edge over its parts.
> - **Stresses after the read (decide nothing):**
>   - 2× borrow: T4 +11.5%, TL +8.0% (+5.9% with late fills); T1 +12.3%, T2 +4.4%.
>   - Whole shares at $33k: T1 +11.9%, T2 +4.0%, T4 +12.0%, TL +8.5%.
>   - IBKR's 2026-10-21 borrow file covers every recent short (median fee 0.3-0.4%; BDCs to 9%).
> - **Beta drift:** dollar-neutral is not beta-neutral. On the top 500 the half-year beta ran
>   −0.40 to +0.73, and the book has been net long since 2025-H2 (ex-ante +0.65 in 2026). On T2, T4
>   and TL it stayed between −0.34 and +0.36.
>   - Proposal 1's beta sizing, tried post hoc in memory on the open tests, cuts the 2026 drift by
>     about two-fifths (+0.40) for −3.0 points of T1 alpha (+0.7 on T2).
>   - Full numbers in `PROPOSALS_RESULTS.md` §4 and `FINAL_DECLARATION.md`.

## The standard setting: version B (since 2026-10-27)

The shipped rules (`A_U0`, below) on the point-in-time **top 2000** (≥ $1M a day, median price
≥ $5, re-formed every six months) instead of the top 500 they were built on. One book on all
2000 had the same alpha within noise and lower volatility (RESEARCH_LOG §13.5). Both notebooks
default to it; the lab's tests and every number before §13.5 are the top 500.

## The rules (`A_U0`, config.yaml as shipped; nothing tuned)

1. **Eligible:** informed_flow_ls_p1's — close >= $5, a days-to-cover and a share count.
2. **SI rank:** minus days-to-cover averaged over 42 bars, percentile inside 5 dollar-volume
   groups, silent below 50% coverage (`informed_flow_ls_p1.compose(...)["ranks"]["si"]`).
3. **MOM rank:** resid_momentum U0v2's `Σe / sd(e)` over bars t−251 .. t−21, `e = r − β·m`
   against the equal-weight market of names >= $5 (252-bar OLS beta), ranked on month-end rows
   and carried to the next month end; masked where p1 calls the name ineligible. A month's
   last row is known only once the next row exists (resid_momentum's `_month_ends`), which is
   exact in a backtest; a live run needs an exchange calendar to re-rank at that close, or it
   decides a session late.
4. **Composite:** `(si.fillna(0.5) + mom.fillna(0.5)) / 2`, averaged over 5 bars among the
   eligible names, re-ranked.
5. **Book:** informed_flow_ls_p1's `_select`. A long enters at rank >= 0.90 and leaves below
   0.80; a short enters at <= 0.10 and leaves above 0.20. Exits first, then free slots best
   first; `max_hold` 126 (the clock restarts in the entry zone); a short is bought back at +100%
   from its entry close and re-enters only after its rank left the short zone once; never short
   (or keep short) a name above 30% short interest / shares; a NaN score exits.
6. **Sizing:** 15 + 15 equal slots of 3.0% (gross 0.90, dollar-neutral), constant while held;
   decided at the close, filled at the next open; a target change on a bar without an open waits
   for the next priced open (adaptive_hedge's `_on_priced_opens`).
7. **Costs:** borrow 0.5% / 3% / 10% a year by short interest / shares (p1's tiers), `fill: open`,
   `order_notional: 1000` (a ~$33k account), longs trimmed at 2× their slot.

Knobs used by the controls and iterations: `long_sleeves` / `short_sleeves` (`si`, `mom`,
`prof`; different sleeves per side run two book passes), `w_si` / `w_mom` / `w_prof`,
`combine: book` (the single-sleeve books side by side, netted: `C_mix`), `short_gate: prof`
(shorts only in the unprofitable half by quality_junk's within-sector profitability rank: I1),
`neutral: sector` (SI ranked within sector × liquidity cells, MOM's residual against market +
sector: I2). All are off in the shipped config.

## Components (loaded by path, pinned by SHA-256, never edited)

| key | file | used for |
|---|---|---|
| `_xsc_p1` | `informed_flow_ls_p1/strategy.py` | SI rank, eligibility, borrow, beta, `_select`, `_weights` |
| `_xsc_rm` | `resid_momentum/strategy.py` | the MOM rank (`compose`) |
| `_xsc_qj` | `quality_junk/strategy.py` | the PROF rank (`compose`), I1's gate |
| `_xsc_ah` | `adaptive_hedge/strategy.py` | `_on_priced_opens` |

A changed component refuses to load (`strategy.component`). The shared `informed_flow_ls`
(under review) is never edited, imported or loaded.

## Files

- `strategy.py`: `signals()` / `valid()`, `ranks()`, `side_score()`, `select()`.
- `research.py`: the mechanics audit (`audit`, `rebuild`, `off_target`), the stuck-order
  estimate booked from each order's due open (`stuck`, `gap_pnl`), the TRAIN premise check, the
  w_mom grid, the pre-registered bars, paired deltas, the paired noise copies of two labels
  (`noise_delta`) and the positions-only sector exposure and overlap (`exposure`). Loaded by
  path; `strategy.py` never imports it; it refuses the locked tests T4 and TL.
- `xs_si_mom_composite_explained_marimo.py` (`uv run marimo edit` it): the strategy explained on
  history from 2025-01-01 as ONE continuous book on a chosen account. It shows the point-in-time
  universe, the two signals and the combined rank for any stock, the day-by-day pick (the orders,
  the book and the candidates on any date), performance against point-in-time buy-and-hold and
  SPY, a timeline of every position with its P&L and exit cause, and one position in detail.
  `explain.py` holds its helpers: it recomputes the picks step by step and asserts they equal
  `signals()`. It caches the panel in `var/xs_si_mom_composite/explain/`; tick "reload from the
  database" after a data update.
- `xs_si_mom_composite_tune_marimo.py` (`uv run marimo edit` it): one overnight click searches
  around version B over EVERY knob (33: universe, both signals, ranking, entry/exit bands,
  slots per side, long/short split, sizing, the winner trim, volatility stop-losses on both
  sides, the account up to $100k), on 2022-01 → 2025-12, in three stages: each knob alone
  (`tune.sweep_plan`), the single changes that helped combined best-first (`tune.improve`),
  then small random combinations until the time budget (`tune.pool_plan`). Then the checks a
  pick must pass against version B before paper trading: a walk-forward that reruns the whole
  rule on the past only, PBO, shuffled-name luck books, neighbours, cost stress, missing stocks
  and account size. 2026-01 → today is reserved for one final read. Every order pays its own
  commission, half spread and square-root market impact (`tune.order_fees`), so a bigger
  account is not automatically better. `tune.py` holds the helpers: `weights()` rebuilds the
  strategy from the same components with every knob and equals `signals()` exactly at
  version B (tested), so `strategy.py` and its SHA-256 pins stay untouched; `tune.Search` keeps
  every backtest in an Optuna journal in `studies/` (git-ignored) and resumes. The data cache
  is `var/xs_si_mom_composite/tune/`.
- `RESEARCH_LOG.md`: pre-registration, every run, the verdict and the declared final.
- `tests/test_xs_si_mom_composite.py`: C_si reproduces p1's book (its silent skew sleeve at
  weight 0) exactly, and p1's config (`w_skew: 1`) differs from it only by float-split ties
  (RESEARCH_LOG §0.4); the MOM rank at t is the last month end's; causality on truncated
  panels; book invariants; `combine: book` = ½ + ½; two passes = one pass with identical
  sleeves; the prof gate only removes short candidates; a target change at a NaN open waits;
  the sector-neutral cells; config params, `valid()` and the SHA-256 pins; and, for
  `research.py`, a stuck cover and a refused short booked from their due open, off-target
  name-days counted both ways, and no locked test read.

Run it through the lab: `uv run python strategies/proposals_lab/lab.py run xs_si_mom_composite
A_U0 '{}' TB,T3,T1,T2 --costs2`.
