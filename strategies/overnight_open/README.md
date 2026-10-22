# overnight_open

Does what happens **between the close and the next open** (the after-hours, the overnight, the
pre-market, the news) connect to a **strong first trading hour** the next day? And does that
first hour follow the night's direction or fade it?

This is the question to answer **before** any trading rule. So this is a research folder like
`classic_signals` and `tori_trades_trendline`: there is no `strategy.py` or `config.yaml`, and the
strategy loader and `tsim` skip it. A strategy comes only if section 6 of the study passes.

```bash
uv run marimo edit strategies/overnight_open/overnight_open_explore_marimo.py   # one night, up close
uv run marimo edit strategies/overnight_open/overnight_open_study_marimo.py     # all nights: the answer
uv run pytest tests/test_overnight_open.py -q
```

The first load builds the night table and caches it under `var/overnight_open/`. It takes about
25 minutes of 1-minute bars (month by month, resumable) and about 4 minutes of SEC requests (name
by name, resumable). After that it loads in seconds. Both notebooks share the cache.

## The night

Everything is in New York time. Session *d*'s night runs from the previous session's close to
*d*'s open:

| part | when | source | used as |
|---|---|---|---|
| previous days | up to the previous close | daily bars | ATR, last day's move / range / volume, 5-day move |
| after-hours | 16:01-19:59 | 1-minute bars | move vs the close, range, volume vs its own normal |
| overnight | 20:00-04:00 | nothing trades for these names | the jump from the last after-hours print to the first pre-market print |
| late reports | 08:00-08:09 | 1-minute bars | volume only: overnight trades reported late at stale prices |
| pre-market | 04:00-09:29 | 1-minute bars | move vs the close, range, volume vs its own normal and vs a day's |
| news | previous session + night | SEC 8-Ks, Benzinga | 8-Ks in the night / in the previous session, earnings 8-Ks, headline counts |

- **Moves** are in units of the stock's daily ATR (Wilder, 14 days, known at the previous close).
  So a 1-ATR night is equally big for a quiet and a wild stock.
- **Volumes** are divided by the stock's own mean of the 20 sessions before (the day itself never
  included).
- **Outcomes** start at the official open, which is what a market-on-open order gets:
  - the first hour's return (09:30-10:29), also with the day's average stock subtracted;
  - its high-low range relative to the stock's usual first hour (the median of the 20 sessions
    before). A **strong** first hour is at least 2× its usual range.

## The data, and what it cannot do

- **Universe:** the ~535 names with extended-hours 1-minute bars, 2025-06-18 → 2026-06-26. After
  that date the database keeps extended hours for 7 names only. These are the HighOctane names:
  volatile, heavily traded, chosen with hindsight. Read every number as "for volatile US stocks".
  The market is always taken out of the direction tests, never left in as a long-only return.
- **The 16:00 minute** holds the closing auction. The daily bar's close is the official close and
  the night's reference price: split-adjusted like the 1-minute bars, not dividend-adjusted.
- **Masks:**
  - IBKR stores a minute with fewer than 100 shares as a copy of the previous close with volume 0.
    Those minutes are dropped.
  - The 08:00-08:09 prices are stale (burst_scan found that they fake a pre-market reversal).
- **Rows left out:**
  - half-days (2025-07-03, 2025-11-28, 2025-12-24) and the session after each;
  - the first ~10 sessions of each name, until its baselines are warm;
  - any day whose 1-minute open is more than 10% from the official open.
- **News timing.**
  - SEC 8-K filings carry EDGAR's acceptance time to the second.
  - **EDGAR's submissions JSON is wrong for about half the filers.** Their
    `acceptanceDateTime` is late by New York's UTC offset (4 h in summer, 5 h in winter). For
    example, Micron's 8-K accepted at 16:02:22 reads `00:02:22Z`, which is 20:02 in New York.
  - The error is a property of the filer. It was checked against the filing index pages on all
    132 filings of 6 names: AAOI, AFRM and MU are always late; AAL, ACHR and ALB are always right.
  - So `research.fetch_filings` reads each filer's offset from the index pages of its oldest and
    newest 8-K. When the two disagree, it reads every filing's page. The corrected times matched
    the index pages to the second on all 132 filings.
  - Uncorrected, a 07:00 pre-market earnings release would land at 11:00, "during the session".
  - Benzinga headlines (MarketData.app, from 2025-10-14) carry a date but no time: every item is
    stamped 00:00. A headline dated on the session day may have come out after the open. So only
    headlines dated up to the day before count as a signal. Same-day headlines appear once, in a
    table labelled **hindsight**, to show what a timed feed could be worth.

## What was known before

burst_scan's `move_extended.ipynb` (16 pre-market-liquid names, 2026-09-29) found:

- the pre-market price is the gap (r 0.999);
- extended hours raised the prediction of a big move near the open only from AUC 0.687 to 0.700,
  and its direction not at all (0.544 → 0.539);
- evening moves clustered on news nights ("an earnings calendar is the missing input").

The direction study (2026-10-01) found nothing that predicts up vs down at a tradable level. It
found one relative overnight reversal, which died out of sample. This study scales the open
question to ~535 names and adds the news.

## Method

- **Discovery:** July 2025 → February 2026. Explore it freely. The model ladder is scored from
  November 2025, once the headline baselines are warm.
- **Validation:** March → June 2026. Read **once**, in section 6 of the study notebook.
- **Errors are clustered by day**, because all stocks share one morning's market and news.
- **The ladder** has four walked-forward models (a small gradient-boosted tree, fixed settings).
  Each adds one group of signals: previous days → + the night's price move → + its trading
  activity → + the news. Each month is predicted by a model fit only on the months before it. The
  score is the out-of-sample AUC. A step's gain gets a 95% interval from resampling whole days.
- **The lift** of a flag compares its nights with unflagged nights whose pre-market moved as far
  (same fifth of the move's size). It answers "does this flag tell more than the move?"
- **Continuation** is the first hour's return beyond the market, signed by the night's direction,
  on nights that moved at least 0.5 ATR. It is in basis points: positive means the first hour went
  on in the night's direction.

## Pre-registration (written 2026-10-20, before any result was computed)

All four tests use a strong first hour of at least 2× usual and a moved night of at least
0.5 ATR, whatever the notebook sliders say. They are scored on the validation months, with models
trained on all months before each one:

| | question | passes when |
|---|---|---|
| V1 | Does the night's price move add to the previous days, for predicting a strong first hour? | the AUC gain's 95% day-bootstrap interval is above 0 |
| V2 | Do the night's trading activity and the news add beyond the move? | the AUC gain (all signals vs previous days + move) has its 95% interval above 0 |
| V3 | Does the first hour continue (or fade) the night's move, beyond the market? | \|t\| ≥ 2 with the same sign as in discovery |
| V4 | Is that continuation different on news nights (an 8-K at night, or ≥ 3 headlines at ≥ 3× normal the day before) than on quiet nights? | \|t\| ≥ 2 with the same sign as in discovery |

- V1 is the "direct connection" the question asks about.
- V2 asks whether anything beyond the gap matters.
- V3 and V4 decide whether a trading rule is worth building. Even when they pass, a first-half-hour
  round trip costs roughly 20-40 bp at $10k (fade_movers' cost table). So a few bp of continuation
  is a connection, not a trade.

## Verdict

Not run yet. Run the study notebook, then write here the discovery reading (sections 3-5) and the
section 6 `SCORECARD` line.

## Files

| file | what |
|---|---|
| `research.py` | loaders (cached), the night table, the flags, the statistics; no figures |
| `overnight_open_explore_marimo.py` | one night: minute candles with the parts shaded, 8-Ks, headlines, its signals vs the stock's normal, all its nights |
| `overnight_open_study_marimo.py` | coverage, strong first hours, size (IC, dose-response, ladder), direction (IC, continuation, ladder), flags, validation |
| `IDEAS.md` | what comes next, parked |
| `tests/test_overnight_open.py` | no look-ahead (nothing from the session's open on reaches a signal), after-hours belong to the next night, filings and headlines land in the right night, half-days left out, the walk-forward never sees its own month, EDGAR's offset undone |
