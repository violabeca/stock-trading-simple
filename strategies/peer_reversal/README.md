# peer_reversal — peer-relative short-term reversal, leaning in when markets are stressed

Proposal 5 of `STRATEGY_OVERVIEW.md` §6. Daily, prices only, long/short, portfolio mode.
Evaluated with the shared lab (`strategies/proposals_lab`). The pre-registration, every run and
the verdict are in [`RESEARCH_LOG.md`](RESEARCH_LOG.md).

## The idea in plain words

When someone has to sell a stock *now* (a fund redemption, a margin call, one big seller), the
price drops below what the news justifies, and whoever buys from them is paid when it comes
back. That pay rises when liquidity is scarce, i.e. when the market is volatile (Nagel 2012).

A raw "buy last week's losers" rule also buys stocks that fell on real news, and industry news
tends to continue. So the move is measured **against the stock's own peers**: stocks whose
residual returns move together over the past year. If the whole group fell, nothing happened to
this stock in particular. If it fell alone, the drop is more likely price pressure.

## How a trade works

1. **Peers.** Every 126 bars, the names are grouped into 30 clusters, from past data only. The
   input is each name's last 252 daily returns minus its beta times the equal-weight market
   return, z-scored. The method is Ward hierarchical clustering (`scipy`).
2. **Signal.** Each name's 5-day return minus the mean 5-day return of its other cluster
   members. It is measured to the close one day *before* the decision bar: skipping that day
   keeps the bid-ask bounce and closing-print artifacts out. The signal is ranked across the
   names every day.
3. **Entries.** Buy the bottom tenth (fell most against their peers) and sell short the top
   tenth, 20 positions a side, most extreme first. The decision is made at the close and filled
   at the next open.
4. **Exits.** A long is sold when its rank climbs back above the median (the gap has closed); a
   short is bought back when its rank falls below the median. Either way, a position closes
   after 10 bars, unless the name is still at an extreme (a fresh signal restarts the clock).
5. **Size.** Each slot gets `gross / 2 / 20` of the book, where `gross` is 0.98 on a stressed
   day and 0.49 otherwise. A day counts as stressed when the market's 21-day realized
   volatility is above its trailing one-year median. The size is fixed when the position opens,
   so a regime change never re-trades a holding.
   - The book is only approximately dollar-neutral. Its targets are neutral when both sides
     hold the same number of slots, but the counts can differ (the net target reaches 17% of
     the book on I4's T1 bars), and held positions drift.
   - Shorts lock their proceeds. The repo engine (`call_seq="auto"`) runs a bar's new shorts
     before the covers that free their cash, so at 0.98 gross about 2% of new shorts filled
     only partly or not at all in round 1. The lab v2 runner runs covers first, and no opening
     is under-filled (RESEARCH_LOG R2.3, issue 6).
6. **Borrow.** Shorts pay 0.5%, 3% or 10% a year by crowding (short interest ÷ shares
   outstanding). A name with more than 30% of its shares sold short is never shorted.

## Parameters

| knob | U0 | what it does |
|---|---|---|
| `peers` | cluster | `cluster` (correlation clusters), `sector` (the lab's sector codes), `market` (one group) |
| `n_clusters` | 30 | how many peer groups |
| `cluster_window` / `reform_every` | 252 / 126 | history each clustering reads / bars between re-formations |
| `min_coverage` / `min_peers` | 0.8 / 2 | data a name needs to be clustered / peers it needs for a signal |
| `formation` / `skip` | 5 / 1 | the move's length / days skipped before the decision |
| `scale` / `scale_window` | none / 63 | `vol` divides the move by the name's own peer-relative daily volatility (Blitz et al. 2013) |
| `q_in` / `q_out` | 0.1 / 0.5 | entry zone (tenths) / exit band (the median) |
| `slots` / `max_hold` | 20 / 10 | positions per side / bars held after the last extreme |
| `gross_stress` / `gross_calm` | 0.98 / 0.49 | exposure of slots opened on stressed / calm days |
| `vol_window` / `vol_median_window` | 21 / 252 | the stress switch's volatility and its median |
| `volume_cap` / `volume_base` | 0 (off) / 63 | skip entries whose busiest formation day traded more than `volume_cap` × normal volume |
| `min_price` | 5 | minimum close |
| `short_max_si`, `si_mid`, `si_high`, `borrow_*` | 0.3, 0.1, 0.2, 0.5% / 3% / 10% | borrow tiers and the crowded-short screen |

## Files

| file | contents |
|---|---|
| `strategy.py` | `signals()`, `valid()` and `compose()`, which returns every intermediate frame |
| `config.yaml` | the U0 values |
| `premise.py` | the TRAIN-only premise check: IC and decile spread against beta-adjusted forward returns, by stress regime and peer definition |
| `anatomy.py` | the TRAIN-only trade anatomy: gross edge per trade by side, stress state, volume and extremeness, and the 21-day reversal curve |
| `checks.py` | the review checks (RESEARCH_LOG §13 and Round 2): P&L by stress state with t, the adoption Δ on every noise copy, each noise copy against its own shuffled band, the reversal curve's t, the tuner surface, realized exposure, under-filled orders, paired Δ against any base; round 1 against round 2 (`v1v2`, `period`), what each lab-v2 fix did to the books and to an adoption (`decompose`, `attribute`), the borrow and holiday-row checks |
| `RESEARCH_LOG.md` | the pre-registration, every lab run, the verdict, the review fixes (§13), the lab-v2 re-evaluation (Round 2) |
| `IDEAS.md` | open questions, not run |
| `tests/test_peer_reversal.py` | synthetic-panel tests: clusters, signal, book mechanics, sizing, causality, borrow |

## Results (proposals_lab v2, 2026-10-23): no edge net of costs; the final is U0

The lab trades the point-in-time top 500, warm, at the next open, at $1,000 orders. alpha =
CAGR − beta × the equal-weight buy-and-hold of the same names. Lab v2 applies the round-1
reviews' fixes: NYSE sessions only, day 0 counted, short covers before new shorts, ETFs out of
the universe. It re-measured every round-1 variant. The round-1 (lab v1) numbers stay in
RESEARCH_LOG §2-§13; the v2 numbers are in its Round 2 section.

| | TRAIN 2016-21 | T1 top 500, 2022-26 | T2 names 501-1000 | T4 names 1001-1500 (re-measured once) | orders/yr |
|---|---|---|---|---|---|
| **U0, the final** (as pre-registered) | −7.7% | −3.5% (P 0.16) | −6.5% | −2.9% (P 0.20) | 3,100 |
| I4, the round-1 final (21-day formation and hold, entries only when stressed) | −2.6% | −3.1% (P 0.26) | +0.1% | +1.5% (P 0.64) | 620 |

- **On lab v2 the final changed from I4 to U0.**
  - I4's T1 gain over U0 fell from +2.8 pts (0.045 above the adoption bar) to +0.4, below the
    v2 bar of 2.6. So I4 is not adopted.
  - The cause is the universe, not the accounting fixes. From 2024-H2, six sector ETFs left the
    top 500 and the next six stocks entered, which re-formed the peer groups. The panel changes
    alone (mostly that swap, plus the 2026 holiday rows) moved I4's T1 alpha from +1.7% to
    −3.4% (RESEARCH_LOG R2.2).
  - Day 0 and the call order moved each number by less than 1 pt and changed no decision.
- **The reversal is suggestive, not established.**
  - On T1, U0's selection beats random names on all four name sets tried: the exact names and
    three copies with 5 names dropped a period, each against its own shuffled-name band. It
    wins by +3.0 to +5.6 pts/yr, at the 99th, 93rd, 96th and 99th percentiles.
  - On TRAIN nothing reaches t 2: the premise IC has t 1.84, the decile spread t 1.45, and the
    +2.4 bp a trade t 0.22.
  - Whatever the edge is, it is smaller than U0's 6.0-6.4%/yr of costs (a 24 bp round trip).
- **The proposal's two distinctive parts showed no advantage.**
  - Sector codes beat the correlation-cluster peers on TRAIN, T1 and T2 (+2.7 / +3.4 / +6.7
    pts). The lab's sector codes are present-day, a mild look-ahead that can only flatter them.
    But plain market-relative reversal also had the higher IC on the TRAIN premise (+0.013 vs
    +0.008, neither at t 2).
  - The binary stress switch added −0.1 to +0.1 pts over a constant gross, which ran 2-6% more
    exposure.
  - Calm-day entries reverted more than stress-day entries on TRAIN, against the proposal's
    rationale: 21-day curve +36 vs +17 bp, difference t −1.1 (−1.9 at day 10). In the daily
    P&L of every book, no stress − calm difference reaches |t| 2 (the largest is 1.5).
- **I4 is a lower-turnover U0, not a better selector.** Its gain over U0 is a cost saving:
  TRAIN +5.1, T2 +6.6 and T4 +4.4 pts. On T1 it gains +0.4, and its selection sits at the
  31st percentile of its shuffled books.
- **U0's profile.** T1 beta 0.13. It beat the market in every bear episode:

  | Q4 2018 | COVID | 2022 | Feb-Apr 2025 |
  |---|---|---|---|
  | +0.1% vs −18.2% | −11.5% vs −37.0% | +7.3% vs −24.5% | +1.4% vs −19.5% |

  But its mean down-month return is −0.8%, and it lost money in 10 of 11 calendar years.
- **U0 fails 6 of the lab's 8 gates:**
  - TRAIN (−7.7%), T1 (−3.5%, P 0.16) and T2 (−6.5%) are negative;
  - late fills −27.3% and 2× costs −9.8% (G5);
  - mean down-month return −0.8% (G6);
  - without its top 5 names, −7.4% (G8).

  It passes the shuffled books (G4, percentile 99) and beta (G7).

Full detail: `RESEARCH_LOG.md`. §13 holds the round-1 review fixes; Round 2 holds the lab-v2
re-evaluation, and R2.7 maps every review issue to its fix and its effect. `config.yaml` keeps
the pre-registered U0 values, which are the final. I4 is the overrides
`{"formation": 21, "max_hold": 21, "gross_calm": 0.0}`. On lab v2 it needs no cash headroom,
since no opening is under-filled. On the repo engine (`src/`, `call_seq="auto"`), a combined
book should still run it at `gross_stress` ≤ 0.9.
