# xs_cross_leg — SI longs, MOM shorts

A cross-strategy of the proposals lab's last round (`strategies/proposals_lab`, lab v2): one
dollar-neutral book whose **long leg is the short-interest rank (SI)** and whose **short leg is
the residual-momentum rank (MOM)**. The question it answers: does each signal belong on one side
of the book, where its documented edge and its beta sit? The reverse cross (MOM longs, SI
shorts, `C_rev`) is the built-in falsification: if the leg evidence is real, the primary must
beat it.

This is a research folder. `RESEARCH_LOG.md` holds the pre-registration (§0, SHA-256-snapshotted
in `var/proposals_lab/results2/xs_cross_leg/prereg/`) and every number.

> **Status (2026-10-24, lab v2, OPEN tests only): the book passes all eight lab gates, but the
> design's hypothesis is not established — the pre-registered falsification bar fails, and the
> integrated composite (xs_si_mom_composite A_U0) beats it on every open test.** Declared final:
> `C_U0` (config as shipped, nothing tuned; `--costs2`, no `--hedge`). T4 and TL are not read
> here; the orchestrator reads TL once.
>
> | | TRAIN (TB+T3) | T1 (top 500) | T2 (501-1000) |
> |---|---|---|---|
> | **C_U0** alpha (P) | +0.65% (0.56) | **+10.62% (0.965)** | +3.85% (0.79) |
> | C_rev (MOM longs, SI shorts) | +0.27% | +6.30% | +1.27% |
> | Δ C_U0 − C_rev (P) | +0.38 (0.58) | +4.31 (0.775) | +2.58 (0.683) |
> | C_si / C_mom alone (composite's) | +0.64% / −0.25% | +9.40% / +9.04% | +2.53% / +1.86% |
> | A_U0 (one composite ranking) | +1.97% | +12.88% | +5.07% (+4.15% repaired) |
>
> - **Gates (bar 1):** G1 +0.65% (thin; MOM's E0 −8.2% invisible to it), T1 P 0.965, 100th luck
>   percentile, late fills +8.6%, 2× costs +10.1%, top-5-out +3.7%, beta −0.03; every bear
>   episode beaten (Q4 2018 −7.2% vs −18.2%, COVID +3.3%, 2022 +9.6%, Feb-Apr 2025 +3.4%); down
>   months +0.87% vs −4.12%. Nothing refused, part-filled or stuck on any open test.
> - **Falsification (bar 2) fails:** C_U0 beats the reverse cross by the 2-5 pts the design
>   expected, but at P 0.775 (T1) and 0.683 (T2) against the 0.80 bar, and TRAIN is level. C_rev's
>   own noise copies halve the margin (T1 +2.4, T2 +1.2 on average). Leg specialization is
>   rejected as pre-registered.
> - **Beta mechanism (bar 3) fails** (4/10 T1 periods within ±0.15): the book is net short
>   2022-24 and net long from 2025-H2 (realized up to +0.64), mostly because SI's long leg's beta
>   drifts (ex-ante ~0.8 in 2023-24, ~2.0 in 2026-H2). The stitched T1 beta is a cancellation.
> - **Best single sleeve (bar 4) passes**, by 0.015 pt on TRAIN; **integration (bar 5) fails:**
>   A_U0 leads by 1.3 / 2.3 / 1.2 pts.
> - Iterations (budget 3, both rejected): I1 composite shorts (TRAIN −2.4, T1 +1.0, T2 +0.8; T2
>   −0.2 with its GME stuck orders repaired) and I2b beta-matched shorts (TRAIN +0.7, T1 −1.8, T2
>   +0.3; it raised the beta).
> - The cross books are recombinations of the controls' legs: C_U0's longs ARE C_si's, 13-14 of
>   its 15 shorts are C_mom's. Like every component, the edge is 2022-2025: both 2026 halves lose.
> - In-sample: T1 is partly in-sample for SI (C_U0's long leg is SI's long leg); TL is fresh only
>   for the combination (informed_flow_ls §38P read names 1501-2000 for SI).

> **Locked read (2026-10-24, run once by the orchestrator, disclosure only; `strategies/
> proposals_lab/FINAL_DECLARATION.md`):**
>
> - C_U0 read T4 +3.18% (P 0.77) and TL +4.78% (P 0.89); with late fills, +1.00% and +2.39%.
> - On the same names the integrated composite A_U0 read +12.35% and +8.56%.

## The rules (`C_U0`, config.yaml as shipped; nothing tuned)

1. **Eligible:** informed_flow_ls_p1's — close >= $5, a days-to-cover and a share count.
2. **Long score (SI):** minus days-to-cover averaged over 42 bars, percentile inside 5
   dollar-volume groups, silent below 50% coverage; averaged over 5 bars among the eligible
   names and re-ranked.
3. **Short score (MOM):** resid_momentum U0v2's `Σe / sd(e)` over bars t−251 .. t−21, `e = r −
   β·m` against the equal-weight market of names >= $5, ranked on month-end rows and carried to
   the next month end; averaged over 5 bars among the eligible names and re-ranked.
4. **Longs first:** informed_flow_ls_p1's `_select` on the long score alone — enter at >= 0.90,
   leave below 0.80, 15 slots, `max_hold` 126 (the clock restarts in the entry zone).
5. **Then shorts:** the same `_select` on the short score with every name held long masked out
   — enter at <= 0.10, leave above 0.20, 15 slots, squeeze stop at +100% from the entry close
   (no re-entry before the score leaves the short zone once), never short (or keep short) above
   30% short interest / shares. A name held long is never a short candidate, and a held short
   that becomes a long is covered on that bar (one order flips it).
6. **Sizing:** 15 + 15 equal slots of 3.0% (gross 0.90, dollar-neutral), constant while held;
   decided at the close, filled at the next open; a target change on a bar without an open waits
   for the next priced open (adaptive_hedge's `_on_priced_opens`).
7. **Costs:** borrow 0.5% / 3% / 10% a year by short interest / shares (p1's tiers), `fill:
   open`, `order_notional: 1000` (a ~$33k account), longs trimmed at 2× their slot.

`strategy.py` is a thin wrapper: every rule lives in `strategies/xs_si_mom_composite/strategy.py`
(its `long_sleeves` / `short_sleeves` switches; two book passes when they differ), loaded by path
as `_xsl_core` and pinned by SHA-256; the core pins its four components the same way. One knob of
its own, off in the shipped config: `short_sizing: beta` (iteration I2b, rejected) sizes each short
at entry by median beta ÷ its beta (p1's rule, clipped 1/1.5..1.5, frozen while held, short gross
<= 0.45); with `equal` the wrapper returns the core's book unchanged (verified bit for bit on real
panels). C_si,
C_mom and the composite A_U0 are read from xs_si_mom_composite's saved results (the same code and
panels; test-pinned that every shared knob ships the same value).

## Files

- `strategy.py`: the wrapper (`signals` / `valid`: the pinned core's, plus the `short_sizing`
  knob).
- `research.py`: the execution audit (`audit`: rebuild with vectorbt's order log asserted equal
  to `lab.book`, refused / part-filled / NaN-dropped orders, off-target name-days both ways, the
  stuck orders' P&L from their due open, book invariants, and the engine against a frictionless
  research preview of the targets), `stuck` (alpha with the stuck P&L removed), `delta`,
  `betas`, `legs`, `overlap`, `noise_sd` and the pre-registered `bars`. Loaded by path; it
  refuses the locked tests T4 and TL.
- `RESEARCH_LOG.md`: pre-registration, every run, the verdict and the declared final.
- `IDEAS.md`: parked ideas.
- `tests/test_xs_cross_leg.py`: two passes equal p1's one pass with identical sleeves; no name
  long and short on one bar, a short that becomes long covered on that bar; C_rev mirrors the
  legs; causality on truncated panels; book invariants; config params, `valid()` and the
  SHA-256 pins; the config equals the composite's but for the sleeves (and the off knob); I2b's
  beta sizing (same longs and short names, sizes by hand, frozen, short gross <= 0.45); and,
  for `research.py`, a stuck cover and a refused short booked from their due open, off-target
  counted both ways, the preview equal to the lab runner without costs, and no locked test read.

Run it through the lab: `uv run python strategies/proposals_lab/lab.py run xs_cross_leg C_U0 '{}'
TB,T3,T1,T2 --costs2`.
