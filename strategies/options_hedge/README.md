# options_hedge — do "outlier" stocks move more than their options priced in?

**Research only: there is no `strategy.py`, `config.yaml` or study.** The idea under study (user,
2026-10-18): find stocks with unusual activity (news, unusual options trading, a volume spike), buy
the stock together with a put that protects against a drop, sell the stock after a rise and keep the
put in case the price comes back.

Stock + put pays like a call, and selling the stock after a rise leaves a put: the package is a
**long-volatility** trade. It can only make money if the stock then moves MORE than the options
already priced in (plus the bid-ask spreads). The direction guess and the exit rules move profit
around; they cannot create it. So the first question is the only one tested here:

> On the days a screen fires, does the stock then move more than its options said it would?

| file | what |
|---|---|
| `options_hedge.ipynb` | the test: data checks, the four screens, straddle outcomes, the matched-random control, the power check, the verdict (`CHECKS` table, `SCORECARD` line); the holdout behind `READ_HOLDOUT` |
| `options_premia.ipynb` | the second study: cheap vs dear options, earnings options, premium selling (pre-registered below), and the one holdout read |
| `research.py` | its helpers (loaded by path as `rs`; tested in `tests/test_options_hedge.py`) |
| `IDEAS.md` | what is parked |

Data: MarketData.app end-of-day option chains and news (`var/marketdata/`, `scripts/marketdata_dump.py`,
1,753 names, 2025-10-14 on, quotes stamped 16:00 New York, no implied volatility in the history),
daily stock bars from the shared DB, earnings releases from SEC EDGAR (8-K Item 2.02, accepted-at
timestamps). Every derived table is cached in `var/options_hedge/` (pickles; git-ignored with `var/`).

## The pre-registered test (written 2026-10-18, before any screen was scored)

**Measure: a ~30-day at-the-money straddle held to expiry** (call + put, same strike). It is the
symmetric version of the stock + put trade with a mirror rule (on a drop: sell the put, keep the
stock), so no direction is needed, and payoff ÷ price = the actual move ÷ the priced move without
any option model.
- Horizons: `30d` = the expiry 21–45 days out closest to 30 (every optionable name has one);
  `7d` = the shortest expiry 4–14 days out (weekly options, liquid names only); in both cases
  among the expiries that have a qualifying strike (clarified after an independent
  re-implementation read it as "fix the expiry first": 69 of 13,900 rows differ).
- Strike: the listed strike nearest the price with a two-sided quote (bid > 0) on both legs, at
  most 5% from the price; only standard contracts (option root = the ticker).
- Bought at the ASK of both legs, plus $0.65 per contract; settled at intrinsic value at the expiry
  date's close, less $0.65 for the leg that finishes in the money. `r_ask` is that return;
  `r_mid` (bought at mid, no fees) is the "before costs" read.
- Liquidity filter: the straddle's bid-ask spread at entry ≤ 10% of its mid price.

**Timing.** A screen fires on day t using data through t's close (volume, news dated t). The chain's
quotes are the 16:00 snapshot, so the straddle is bought at **day t+1's** close.

**Four screens** (20 sessions of trailing history, the day itself excluded; a screen fires only on
the first day of an episode: not again within 5 sessions):
1. `options_volume`: the day's total option volume (all contracts) ≥ 3× its trailing median, and
   that median ≥ 50 contracts.
2. `stock_volume`: split-adjusted share volume ≥ 3× its trailing median.
3. `news`: headlines dated in the session (weekend items roll to Monday) ≥ 3 and ≥ 3× the trailing
   average count.
4. `outlier_move`: |log return| ≥ 3× the trailing standard deviation of daily log returns.

**Excluded** (from screens and controls alike): any straddle whose window touches an earnings
release (reaction session within [t − 2, expiry + 1] sessions: an 8-K can be filed hours after the
release itself), names without an earnings date in the window (foreign filers report on 6-Ks), and
any split or data break (the chain's price ÷ the DB's adjusted close jumps > 15%) within
[t − 20, expiry]. Earnings-window events are reported separately as information.

**Control: matched random days.** Per screen and horizon, 20,000 random sets of eligible name-days
with the same number of picks per entry date and per implied-move tercile (straddle mid ÷ price,
terciles within each name over the window: expensive vs cheap options), drawn from the name-days on
which that screen could have fired. Same dates remove market-wide shocks; same tercile removes the
"already expensive" effect.

**Pass bar (all of them, per screen × horizon; 8 cells):**
1. tradable: mean `r_ask` of the events > 0;
2. better than chance: one-sided p ≤ 0.003 against the matched random sets (≈ 0.025 / 8);
3. not a few lottery tickets: mean `r_ask` still > 0 without the best entry week and without the
   5 names that contributed most.

**Power check.** Fake screens with the same per-cell counts, drawn at random and planted with a known
edge (payoff × 1.15, i.e. an actual move 15% above the priced one; also × 1.30), are scored against
the same control. A cell whose planted +15% is found less than 80% of the time can only fail
"inconclusive".

**Windows.** Discovery: straddles bought 2025-10-15 on and expiring by **2026-04-30**. Locked
holdout: bought **2026-05-01** on, expiring by **2026-08-31** (the last month every name's chain is
downloaded for), scored ONCE, only for discovery-passing cells, each at one-sided p ≤ 0.05 / (number
of such cells) with mean `r_ask` > 0. If nothing passes discovery, the holdout stays unread. The
2025-10 → 2026-03 unconditional base rate (no screen) was looked at once before this was written:
30-day ATM straddles returned about 0% at mid and −14% at the ask; options already expensive for
their name returned less.

**One change after writing, before any scoring:** the earnings dates were to come from Yahoo, which
answered nothing for 1,265 of 1,753 names and then refused every request (rate limit). They come from
SEC 8-K Item 2.02 filings instead: the company's own earnings release, stamped to the second. A
release accepted at or after 16:00 New York reacts on the next session.

**One warning added after writing, before any scoring:** `p_names`. The control treats events as
independent picks, but one name can fire again a few sessions later and its straddles then overlap.
`p_names` bootstraps whole names over each event's excess above its (date, tercile) cell's mean. It
is never a gate; a passing cell with `p_names` > 0.05 is flagged as a WARNING.

Only a passing cell earns step 2: the exit rules (the user's stock + put rule, the straddle held to
expiry, an early exit) on daily bid / ask marks.

## What the test found (discovery window, run 2026-10-18)

**No screen picks stocks that then move more than their options priced in. The holdout stays
unread.** In numbers (all straddle returns per position, held to expiry):

- **The data is sound.** The chain's stock price matches the DB's day by day (median gap 0.0 bp),
  call − put at the money matches price − strike to 0.40% (about the 30-day carry), 99.99% of
  straddles find their expiry price, the 17 splits / spin-offs / reverse splits in the window are
  caught (NOW 5:1, BKNG 25:1, DD's spin-off …), and recomputing the screens on data cut at a date
  changes nothing before the cut. 1,529 of the 1,733 names with standard contracts have SEC earnings
  dates.
- **Liquidity is the binding constraint.** Only 10% of name-days quote the pre-registered
  at-the-money straddle within a 10% spread (16,770 of 159,413 30-day rows); the earnings exclusion
  halves the rest: 8,861 eligible 30-day name-days on 711 names, 9,238 7-day name-days on 423 (the
  liquid-strike variant below more than doubles this).
- **Base rate (every eligible name-day, no screen):** options priced the move right on average
  (actual ÷ priced 1.00). The straddle returned +0.8% at mid / −2.8% at the ask (30d) and −0.2% /
  −4.2% (7d). The dearest third of each name's days (in hindsight) returned −8.6% at the ask, the
  cheapest +2.0%.
- **The pre-registered test: 0 of 8 cells pass; all 8 are *inconclusive*.** 38–218 events per
  cell, so a +15% edge would have been found only 7–55% of the time. No cell beats its random days:
  on the 30-day horizon all four screens did WORSE than random (stock volume −17.2% vs +2.8%, news
  −15.2% vs −2.2%, outlier move −4.3% vs +3.7%, options volume −2.1% vs −1.6%); the best 7-day cell,
  stock volume, had p = 0.10 on 48 events.
- **Exploratory, with power (not a verdict):** loosening the spread filter to 30% and pooling the
  four screens gives 2,493 (30d) and 2,129 (7d) events, enough to see a +15% edge every time. The
  screened days' excess over random days is **+0.6% (95% band −2.5% to +3.7%)** and **−0.7% (−3.8% to
  +2.4%)**: zero, measured to within about ±3 points of straddle return, which is less than the
  spread paid. Stock-volume spikes lean the wrong way in five of the six filter × horizon looks (−6%
  to −20% vs random, p 0.92–0.98 in the other direction; the exception is 7d at the 10% filter, +13%
  on 48 events): after a spike the options tend to be dearer than the move that follows.
- **Exploratory, the liquid strike:** an independent re-implementation found that the pre-registered
  strike rule (nearest two-sided strike) often lands on a fresh $1 strike with no open interest whose
  wide quote fails the 10% filter, while the standard strike beside it is liquid. Buying the nearest
  strike that passes the filter, as a trader would, more than doubles the pool (19,790 30-day and
  15,965 7-day name-days; power at +15% ≥ 0.99 pooled). The answer does not change: pooled screens
  minus random days **−4.3% (−9.1% to +0.5%)** at 30 days and **+1.1% (−4.2% to +6.4%)** at 7 days;
  no single screen beats random (best p = 0.26).

**For the stock + put idea:** on the days these screens pick, the protection costs at least what
the coming move is worth. The straddle (the direction-free version of the package) bought at the
ask on screened days lost about 3.5% (30d) and 2.5% (7d) per position with the liquid strike, and
4% to 9% once spreads up to 30% are allowed: the same as on random days. Exit rules move that profit
around but cannot create it, so step 2 is not run. What would reopen it: a screen that beats random
days here first (IDEAS.md), or intraday option quotes for an intraday entry.

**Independent check.** A second scorer re-implemented the straddles, two screens (`outlier_move`,
`options_volume`) and every exclusion from this README alone, without reading `research.py`, on 150
random names (30d horizon). Reconciled row by row: 13,831 rows identical in every value (strike,
expiry, prices, spread, `r_ask`, `r_mid`, break / earnings flags, both screens' firings); the 69 extra
rows here are the expiry reading clarified above; one row differs at the 10% spread boundary (float
rounding of exactly 10%). It also confirmed that the DB `close` is split-adjusted back in time for
names that split later (BKNG, SNEX, AGL), so the straddle is settled on the chain's own price, as here.

## Second study: option premia (pre-registered 2026-10-19, before scoring)

Three more ways to trade the options themselves, in the order of their promise. `options_premia.ipynb`
runs them; helpers come from this folder's `research.py` (`rs`) and `option_signals/research.py`
(`os_rs`, the implied volatilities). Same windows as above: discovery = positions closed by
**2026-04-30**; locked holdout = opened **2026-05-01** on and closed by **2026-08-31**, read only for
cells that pass. Straddles use the **liquid strike** (`rs.LIQUID`), learned from the first study.

**A. Cheap vs dear options** (Goyal & Saretto 2009). Two signals on day t:
`hv_iv` = log(realized volatility of the last 252 daily log returns ÷ ATM IV) — high = options
cheap against the stock's own history; `iv_own` = log(ATM IV ÷ its median over the previous 20
sessions) — low = cheap against its recent norm. ATM IV = `os_rs` expiry-A mean of the ATM call and
put. Each day, among the eligible 30-day liquid straddles (the first study's exclusions: spread,
splits, any earnings release in the hold), the cheapest tenth is BOUGHT at the ask and the dearest tenth SOLD at the bid, held to
expiry. Seller return = (bid − payoff − fees) ÷ bid. Control: random eligible straddles with the
same number per entry date (date-matched only: the price level is the thing tested). 4 cells
(2 signals × buy cheap / sell dear). **Pass**: mean return > 0, one-sided p ≤ 0.006 (≈ 0.025 / 4)
against the control, and > 0 without the best week and without the top-5 names.

**B. Earnings options**, two trades the first study did not measure, on every SEC Item 2.02
release whose reaction session r falls in the window, with the liquid ATM straddle of the first
expiry on or after r (strike within 5% and straddle spread ≤ 10% at entry):
1. `crush`: SELL at the bid at the close of r − 1, buy back the same contracts at the ask at the
   close of r (the implied move is priced in, then collapses);
2. `runup`: BUY at the ask at the close of r − 6, sell the same contracts at the bid at the close of
   r − 1 (implied volatility tends to climb into the release; Gao, Xing & Zhang 2018).
Return per $ of premium paid or received, fees $0.65 per contract per trade. **Pass**: mean > 0 with
one-sided p ≤ 0.025 (0.05 / 2) when bootstrapping whole NAMES and also when bootstrapping whole
entry WEEKS, and > 0 without the top-5 names.

**C. Plain premium selling**, as a base rate:
1. `put_write`: every session, sell the expiry-A put nearest 95% of the price (`os_rs` surfaces)
   at the bid on each name of the `os_rs` universe whose put spread is ≤ 20% of its mid; hold to
   expiry (cash-secured; earnings included, as a put-writer holds through them). Return per $ of
   premium. (Monthly entries were the first draft: five entry dates in the window cannot be
   tested.)
2. `short_straddle`: sell every eligible 30-day liquid straddle (no earnings in the hold) at the bid,
   held to expiry; return per $ of premium.
**Pass**: mean > 0 with one-sided p ≤ 0.025 bootstrapping whole entry weeks, and > 0 without the
top-5 names. A pass here is a premium, not an edge: the notebook also shows the worst 1% of
positions, the price of that premium.

### What the second study found (discovery window, run 2026-10-19)

**0 of 8 cells pass. Options here are priced about fairly at the mid; the bid-ask spread decides
almost every trade.** Returns per $ of premium:

- **A · Cheap vs dear** (19,790 eligible 30-day straddles, 802 names). The cheap side DID beat
  random days, in the published direction, but not by enough to pass, and not robustly:

  | cell | events | mean | random | excess (95% band) | p | without top-5 names |
  |---|---|---|---|---|---|---|
  | `hv_iv` buy cheap | 1,977 | +0.4% | −2.6% | +3.0% (−0.3, +6.3) | 0.038 | −6.2% |
  | `hv_iv` sell dear | 1,877 | −1.0% | −5.4% | +4.4% (+0.8, +8.1) | 0.008 | −4.3% |
  | `iv_own` buy cheap | 1,668 | +4.4% | +0.2% | +4.2% (+0.5, +7.8) | 0.014 | −0.9% |
  | `iv_own` sell dear | 1,756 | −9.2% | −8.6% | −0.6% | 0.62 | −11.9% |

  At the mid price each `hv_iv` leg beats random by about 3 points; the spread costs about as much,
  so neither leg makes money on its own after costs (+0.4% and −1.0%).
- **B · Earnings options** (2,485 releases with a liquid straddle; every exit quoted). `crush`:
  −13.2% (win rate 35%, worst 1% −169%); `runup`: −11.2% (win rate 16%). At the MID price both are
  about zero (crush −0.5%, runup +1.6%; −0.5% and −0.3% with expiries within 14 days of the
  release): the event is priced fairly, and the spread, 8% of the premium at entry and 12–16% at the
  exit after the release, is the whole loss. Options priced a 14.8% move (mid ÷ price, time value
  included) where the stock moved 7.0%; the liquid-strike rule let half the straddles drift to long
  expiries (median 23 days, up to 820), which does not change the answer.
- **C · Premium selling** (base rates): `put_write` +2.3% of premium (21,348 puts on 617 names,
  median premium 2.8% of the strike, 75% expire worthless, worst 1% −834%; p 0.43 by weeks, +0.1%
  without the top-5 names) — no premium to speak of in this window, with the crash tail intact.
  `short_straddle` −5.4%: selling at the bid loses the spread.
- **Exploratory, not a verdict:** `hv_iv` with both legs at once beats random by +7.4% per $ of
  premium (z 2.95); `iv_own` +3.5% (z 1.30). This was the one lead, sent to the holdout below.

### Holdout confirmation of ONE exploratory lead (pre-registered 2026-10-19, after discovery, before the holdout was read)

No cell of the second study passed. The one exploratory read with a signal was test A's `hv_iv`
with both legs together (buy the cheapest tenth at the ask, sell the dearest tenth at the bid, same
days): excess over random +7.4% per $ of premium, z 2.95, but with weak robustness (the cheap leg is
negative without its top-5 names; name-clustered p 0.12-0.25). The holdout (straddles bought
2026-05-01 on, expiring by 2026-08-31) is read ONCE for this single hypothesis and nothing else:
**confirmed** if the combined excess over random is > 0 with z ≥ 1.65 (one-sided 5%) AND each leg's
excess is > 0. The holdout is then spent for test A; every other cell of both studies stays unread.

**Holdout result (read once, 2026-10-19): CONFIRMED on the pre-registered bar.** 8,955 eligible
straddles on 554 names, bought 2026-05-01 → 2026-08-07:

| leg | events | mean | random | excess (95% band) | p | without top-5 names |
|---|---|---|---|---|---|---|
| buy cheap (ask) | 913 | −4.0% | −12.2% | +8.2% (+3.9, +12.5) | 0.0001 | −13.7% |
| sell dear (bid) | 851 | +7.3% | +4.5% | +2.8% (−2.0, +7.6) | 0.12 | +1.9% |

Combined excess +11.0% per $ of premium, z 3.37. **What is confirmed is RELATIVE:** straddles whose
implied volatility is low against the stock's own past-year volatility do better than random
straddles bought the same day, and dear ones do worse, in both halves of the year (Goyal & Saretto's
effect, on 2025-26 data). **What is not:** a profit. The absolute result follows the market's
volatility, not the signal: in the stormy discovery months buying straddles paid and selling lost, in
the calm holdout the reverse. The pair, one $ of premium on each leg, averaged −0.3% per position
(discovery) and +1.6% (holdout) after spreads, and the cheap leg leans on a few names (−13.7% without
its top five; name-clustered p 0.07). The edge is about the size of the bid-ask spread.
