# turnover_momentum — short-term momentum, only where the tape is deep

Buy last month's winners **among the names that trade the most**, hold them a month, repeat.
The classic one-month effect in stocks is *reversal*; Medhat & Schmeling
([RFS 2022](https://academic.oup.com/rfs/article-abstract/35/3/1480/6286969),
[SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3150525)) show that sorting on
share turnover at the same time splits it in two — low-turnover names reverse, **high-turnover
names continue** — and that the continuation leg is "as profitable and as persistent as
conventional price momentum", "survives transaction costs", and is "strongest among the
largest, most liquid, and most extensively covered stocks". Blitz, Hanauer, Honarvar,
Huisman & van Vliet ([FAJ 2023](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4115411))
add that a composite of such short-term signals earns significant *net* alpha in a liquid
global universe, works long-only, and is robust to implementation lags of several days —
which is what this engine's next-bar-open fill needs.

## Thesis, and how it is falsified

Among liquid names, heavy trading marks where some participants under-appreciate the
information in prices, so a one-month move continues for about another month. This
strategy trades **only the continuation leg, long only**, and makes the turnover
conditioning *the* thesis rather than a tuning knob:

- `turnover_q: 0.0` — everyone eligible — is plain one-month momentum, the control the
  paper says should be weaker or reversed.
- `polarity: reversal` on the high-turnover slice must **lose**.
- Both sit in the sweep grid, so §5 of the notebook is the falsification, not a tuning run.

Before any of that, **§4 is pre-registered**: the rank IC of the formation return against
*cross-sectionally demeaned* forward returns, split by turnover tercile, with no engine
involved. **Pass** = positive IC in the top tercile and lower (or negative) IC in the
bottom one, on the train window. **Fail** = stop; the study is not run.

## How it trades

1. Each day, `ret` = the trailing `formation`-bar return, ending `skip` bars ago.
2. `turnover` = trailing `turnover_window`-bar mean volume ÷ shares outstanding (SEC XBRL
   `fund_shares`, point-in-time), with the share count re-expressed in the units of the
   split-adjusted volume series (see **Caveats**). A name whose turnover rank is below
   `turnover_q` is **dropped from the cross-section** — not handed a neutral score — so it
   neither ranks nor moves anyone else's percentile.
3. Every `rebalance_days` bars, hold the top `max_positions` eligible names by `ret` (the
   bottom, under `polarity: reversal`), one fixed slot of `1/max_positions` each; carry the
   book between rebalances. The monthly cadence *is* the holding period the literature tests,
   and it needs no stateful hysteresis loop.
4. **One shared book, portfolio mode.** `signals()` returns a `weights` frame, so the engine
   runs one $10k account: buys are funded by sells, a thin selection holds cash rather than
   concentrating, and `retrim_k` trims a runaway winner back to its slot. Fills are at the
   next bar's open; there is no look-ahead.

## Universe — `Liquid500`, point-in-time

Built in `notebooks/01_universe_screener.ipynb` §8b with the same construction as
`Morning200`: the daily names with at least 1,000 sessions inside the train window
(2020-07-01 → 2024-12-31), ranked by **median daily dollar volume over 2020-01-01 →
2020-06-30** — a window that closes before the backtest opens — with ≥ $50M/day and ≥ $5
floors measured there too, top 500 kept. Result: 500 names across 12 sectors (largest 22%),
median $215M/day and $71 a share in the ranking window; the smallest name kept traded $88M a
day. It is broad on purpose: the turnover sort needs a real cross-section of turnover, and
`LiquidLeaders`' 120 mega-caps are all high-turnover in absolute terms.

## Parameters

| param | shipped | meaning |
|---|---|---|
| `formation` | 21 | trailing bars for the momentum return (~1 month); pinned in the sweep |
| `skip` | 0 | bars skipped before today (0 = the paper's plain previous month; ~5 skips the last week's reversal); grid `[0, 5]` |
| `turnover_window` | 63 | trailing bars the turnover level is averaged over |
| `turnover_q` | 0.5 | turnover quantile a name must reach to be ranked — **the thesis knob**; grid `[0.0, 0.5, 0.8]` |
| `polarity` | momentum | `momentum` holds the top movers, `reversal` the bottom; grid `[momentum, reversal]` |
| `rebalance_days` | 21 | bars between re-selections = the holding period |
| `max_positions` | 10 | slots, `1/max_positions` each. **Not searched** — a risk decision (informed_flow's studies picked this knob's floor six times running); grid `[10, 20]` |
| `min_coverage` | 0.5 | `fund_shares` breadth floor; below it turnover falls back to dollar volume |
| `retrim_k` | 2.0 | concentration cap; fixed policy, never searched |

## Reading the result

- **§4 first.** If the top-turnover tercile does not carry a positive demeaned IC on train,
  nothing downstream means anything.
- **§10a is the gate**: equal-weight buy-and-hold of the same 500 names, same costs. A
  long-only momentum book in a rising market posts a healthy absolute Sharpe by being
  invested; only `excess_cagr` / `info_ratio` separate selection from exposure.
- **Expected magnitude, honestly.** The long leg's excess over equal-weight of the same
  universe is plausibly +5–15%/yr in trending regimes, with losses in reversal regimes (2022,
  early 2023). Monthly rotation of ~50–70% of a 10-slot book is ~120–170 legs a year ≈ 4–5%/yr
  of drag at the modeled costs — real, but 3–4× smaller than any 1-minute strategy here.
- Read `n_trades` beside every Sharpe, `pct_profitable` for breadth, and the holdout (§10d)
  exactly once.

## Caveats

- **Survivorship.** The shared database holds **no delisted names** (`securities.is_active`
  is true for every row), so `Liquid500` is drawn from today's listings and its history is
  survivor-only. The benchmark-relative objective controls for that partially — strategy and
  hurdle trade the same survivors — but absolute returns and drawdowns are flattered. Only
  the *ranking* inside the pool is point-in-time.
- **Turnover units.** Daily bars store split-adjusted close *and* volume (Yahoo restates the
  history into the latest share units), while `fund_shares` is as reported at each filing.
  Divided naively, pre-split turnover is overstated by the full future split factor — and
  splits follow run-ups, so past winners would be pushed into the high-turnover bucket by
  information from their future. `strategy._in_volume_units` reads every ≥ 1.9× jump between
  consecutive filings as a unit change and restates earlier bars, which is exactly what the
  volume series already carries. Residual: a split *after* the panel ends but before the
  backfill leaves that one name's turnover uniformly overstated across the window. A merger
  that doubles the share count would be misread as a split (rare; documented, not handled).
- **`fund_shares` coverage.** 20-F filers report annually and lag; a name with no filing yet
  has NaN turnover and is simply not ranked. Below `min_coverage` breadth the whole strategy
  falls back to dollar volume — a size × turnover mix — and §4 prints which source ran.
- **Momentum crashes.** `retrim_k` caps single names; nothing caps the factor. Read the
  sub-period table (§10c) for 2022 and early 2023 before believing a headline.
- **Rebalance grid anchored at bar 0.** The first picks land on the first rebalance bar after
  the `formation` / `turnover_window` warm-up (~2 months), so the book is under-invested at the
  very start of the window against a fully invested hurdle.

## Sources

Medhat & Schmeling, *Short-term Momentum*, RFS 35(3) 2022 · Blitz, Hanauer, Honarvar, Huisman
& van Vliet, *Beyond Fama-French Factors: Alpha from Short-Term Signals*, FAJ 79(4) 2023 ·
Jegadeesh (1990) and Lehmann (1990) on one-month reversal, the effect this strategy avoids by
construction.

---

## Result (2026-09-12): the pre-registered test **FAILED** — the study was not run

The §4 gate was run headless on the real panel before any optimization, exactly as
pre-registered, then repeated on two other populations to separate a regime from a
population effect. Everything below is a 21-day rank IC of the 21-day formation return
against cross-sectionally demeaned 21-day forward returns (positive = continuation); the
t-statistic is overlap-adjusted (÷√21). **No slice, in any population, carries |t| above
1.6, and the sign among the most-traded names is the opposite of the thesis.**

| population | window | shares source | top turnover tercile | middle | bottom tercile |
|---|---|---|---|---|---|
| `Liquid500` (500 names, 230 eligible/day) | 2020-07 → 2024-12 | shares, 472/500 covered | **−0.0230** (t −0.96, decile spread −1.01%) | −0.0183 | −0.0010 |
| broad pool (1,951 names, ≥ 95% daily coverage) | 2021-01 → 2024-12 | shares, 1,785/1,951 | −0.0067 (t −0.34); top *quintile* +0.0002 | −0.0228 | −0.0124 |
| `LiquidLeaders` (116 deep names) | 2015-07 → 2019-12 | shares, 108/116 | **−0.0458** (t −1.57) | — | +0.0005 |

Controls on `Liquid500`: the ranking the strategy actually trades (eligible names, momentum)
−0.0241 (spread −1.21%); `skip: 5` −0.0248; no conditioning (`turnover_q: 0.0`) −0.0127;
top quintile (`0.8`) −0.0258. On the broad pool: conditioning on **dollar volume** instead of
shares gives top −0.0053 / bottom −0.0153; a 63-day formation −0.0076 / −0.0067; a 126-day
formation skipping the last 21 bars +0.0131 / +0.0133 — the ordinary medium-term momentum
tilt, identical across turnover terciles, so turnover is not what conditions it.

Per year, top vs bottom tercile (`Liquid500`): 2020 −0.040 / +0.029 · 2021 −0.077 / −0.048 ·
2022 −0.025 / +0.015 · 2023 +0.006 / +0.012 · 2024 +0.011 / +0.034. Pre-2020
(`LiquidLeaders`): 2016 +0.027 / −0.015 · 2017 −0.040 / −0.008 · 2018 −0.050 / +0.103 ·
2019 −0.071 / −0.039. The sign flips year to year; 2021 (the meme/COVID tape) is strongly
reversal, 2023–24 mildly continuation — regime, not an edge.

**Reading.** In this data the most-traded liquid US names show the *classic* one-month
**reversal**, weakly (the mirror image of Medhat & Schmeling's continuation), and the
least-traded names show nothing. That is a finding about these universes and years, not a
strategy: a reversal leg on the top tercile would carry an IC of +0.02 to +0.05 with |t| < 1.6,
fading to zero in 2023–24, on a monthly-rotation cost base. The 5-trial smoke run passes
mechanically (`info_ratio` −0.32 … +0.37 across random draws) and proves nothing either way.

**Status: falsified on this data.** The config default stays `polarity: momentum` so an
unshipped result is not mistaken for a shipped one; the `turnover-momentum-1d-v1` study holds
only the 5 smoke trials and should not be extended. What would reopen it: a **monthly,
non-overlapping** replication (the paper's own construction — month-end formation and
holding) showing the sign flips back, or a universe built from *all* US common stocks where
"high turnover" means the market's top decile rather than the most-traded of an already
liquid pool. Until then this folder is a documented negative, like `gap_and_go`.

**Data-quality finding made along the way** (matters for every strategy that uses
`fund_shares`, i.e. also `value_factor` and `conviction_flow`'s market cap): the SEC XBRL
`dei:EntityCommonStockSharesOutstanding` facts contain filer-side unit errors — AJG reported
shares ×1,000,000 for two quarters in 2020, BALL a literal 0 on 2022-02-14, RMD and CMG
÷1,000 for one quarter each, CCL ×1,000 — and the `fundamentals` table holds **124 rows with
a non-positive share count across 87 symbols**. `_in_volume_units` treats those as splits and
restates a name's history by 1,000×, so the affected names drop out of the turnover ranking;
the fix is an engine proposal (see `IDEAS.md`: a one-sided read-time repair plus a
per-period tag fallback in the SEC provider). The verdict does not depend on it: with
the repair and a staleness cutoff applied, the top-tercile IC moves from −0.023 to
−0.019 and the gate still fails.
