# option_signals — do option prices tell which stocks will do better?

**Research only: there is no `strategy.py`, `config.yaml` or study.** The options_hedge test found that
buying options on "outlier" days pays the bid-ask spread for nothing: on liquid names the options
price the coming move about right. This folder asks the opposite question: do option prices carry
information about the STOCK's direction? If so it is traded with stocks (no option spread), as a
cross-sectional ranker in the repo's portfolio mode, like informed_flow.

| file | what |
|---|---|
| `option_signals.ipynb` | the test: implied volatilities, the four signals, data checks, the rank-IC gate, the verdict (`CHECKS`, `SCORECARD`); the holdout behind `READ_HOLDOUT` |
| `research.py` | its helpers (loaded by path as `os_rs`; tested in `tests/test_option_signals.py`) |
| `IDEAS.md` | what is parked |

Data: MarketData.app end-of-day chains (`var/marketdata/chains/`, quotes stamped 16:00 New York, NO
implied volatility: computed here), daily bars from the shared DB. Caches in `var/option_signals/`.

## The pre-registered test (written 2026-10-19, before any signal was computed)

**Implied volatility.** Each name-day's chain at 16:00: the expiry 21–60 days out closest to 30
among those quoting an at-the-money call and put ("expiry A"); every standard contract on it with
bid > 0, ask ≥ bid, strike within ±20% of the price. IV = Black-Scholes on the mid price with
r = 3.75% and the stock price net of the present value of the dividends projected before the
expiry (escrowed-dividend model); NaN where the mid is outside the no-arbitrage bounds. Dividends:
the ex-dates and sizes (as a fraction of the price) are read from the steps of the DB's adjusted ÷
raw close ratio; the next ex-date = the last known one + the median gap of the last four, the size =
the median of the last four.

*Changed before any signal was scored:* the first version used a continuous trailing-year yield.
On a 5-name sanity check KO's calls then came out 2 volatility points dearer than its puts in a
month without an ex-date (a 30-day option sees a quarterly dividend or none, never a smooth yield);
that error is the size of the cpiv signal itself. With projected discrete dividends KO's median
call − put IV is −0.1 points.

**Four signals on day t** (the expected sign is the published one):

| signal | definition | expected | source |
|---|---|---|---|
| `skew` | IV of the put nearest K/S = 0.95 (in [0.90, 0.975]) − IV of the call nearest K/S = 1 (in [0.97, 1.03]) | − | Xing, Zhang & Zhao (2010) |
| `cpiv` | open-interest-weighted mean of IV(call) − IV(put) over strikes with both legs quoted, K/S in [0.90, 1.10] (equal weights when all open interest is 0) | + | Cremers & Weinbaum (2010) |
| `vrp` | realized volatility of the last 21 daily log returns (annualized) − ATM IV (mean of the ATM call and put IVs) | − | Bali & Hovakimian (2009) |
| `os` | option contracts traded × 100 ÷ shares traded, both summed over the last 5 sessions (all expiries) | − | Johnson & So (2012) |

**Universe on day t:** names whose ATM call and put IVs exist, raw close ≥ $5, then the 500 with
the highest median dollar volume over the 20 sessions through t. Each signal is scored on the
universe names where it is defined.

**Outcome.** The signal is known at t's close; the trade is at t+1's open. Forward return =
adjusted open of t+1 → adjusted open of t+1+h, for h = 1, 5 and 21 sessions, minus β × the
universe's equal-weight forward return (β: 120-session OLS beta of the name's daily returns on the
equal-weight return of all names, through t). So the market is out and only selection is left.

**Statistic.** Each day, the Spearman rank correlation (IC) between the signal and the residual
forward return across the universe. Mean IC over the window, with a Newey-West t (lag h).
`partial IC`: the same after regressing the signal each day on log dollar volume, the 5- and
21-session past returns and the 21-session realized volatility.

**Pass bar (per signal × horizon, 12 cells), all of:**
1. mean IC in the expected direction with Newey-West t ≥ 2.65 (one-sided ≈ 0.05 / 12);
2. partial IC in the same direction with t ≥ 2.0;
3. IC in the expected direction in at least 60% of the months with ≥ 10 signal days.

Reported with each cell: the minimum detectable IC (80% power at the bar), the IC by month and by
dollar-volume tercile, and the top-minus-bottom-quintile residual return (gross, and net of 5 bp per
side at the measured turnover). Information only.

**Windows.** Discovery: signal days from the first with every input (2025-10-20) to the last whose
forward window ends by **2026-04-30**. Locked holdout: signal days **2026-05-01 → 2026-08-31** (the
last month every name's chain is downloaded for), scored ONCE, only for cells passing discovery, each
at one-sided p ≤ 0.05 / (number of such cells) with the IC in the expected direction. Nothing passes
→ the holdout stays unread.

## What the test found (discovery window, run 2026-10-19)

**None of the four signals ranks these stocks: 0 of 12 cells pass, and none comes close. The
holdout stays unread.**

- **The data is sound.** The computed ATM IVs look like IVs (median 36%, p10–p90 22–75%) and track
  each name's realized volatility across names (correlation 0.84); after dividends the ATM call and
  put IVs agree (median gap +0.4 points, ≈ the rate assumption); puts below the money are dearer
  (median skew +1.2 points). Recomputing the signals on data cut at a date changes nothing before
  the cut; forward returns start at the next open. 500 names a day, 133 signal days.
- **The test.** Mean daily rank IC with market-neutral forward returns, Newey-West t:

  | signal (expected) | h = 1 | h = 5 | h = 21 |
  |---|---|---|---|
  | `skew` (−) | −0.001 (t −0.1) | +0.008 (+0.8) | +0.017 (+1.2) |
  | `cpiv` (+) | +0.004 (+0.7) | +0.002 (+0.3) | +0.008 (+0.7) |
  | `vrp` (−) | +0.009 (+0.9) | +0.016 (+0.9) | +0.022 (+0.9) |
  | `os` (−) | −0.005 (−0.4) | −0.017 (−0.7) | −0.036 (−0.9) |

  Three of the four lean the WRONG way at most horizons. Only `os` points the published way, and only
  after the controls (partial IC t −2.6 at h = 5); its raw IC is t −0.7 and it has the published sign
  in 3 of 7 months. The long-short quintile books are noise: from −36% to +27% a year net of
  costs across the cells, positive in 5 of 12, no pattern by signal or horizon.
- **Power.** One window of 7 months: the smallest IC this test could find 80% of the time is
  0.02–0.04 for `skew` / `cpiv` at 1–5 days and 0.04–0.14 for the rest. The published effects (ICs
  of roughly 0.02–0.03, mostly in older data and smaller names) sit at the edge for `skew` / `cpiv`
  and below it elsewhere. So: no sign of a usable signal in the 500 most liquid optionable names in
  2025-26; small effects of the published size cannot be ruled out with one year of chains.
- **Exploratory (not a verdict):** changes instead of levels (the call − put spread over 5 sessions;
  the call minus put IV change over 21, An, Ang, Bali & Cakici 2014) and a composite of the four:
  all ICs within ±0.01, no |t| above 1.8.
