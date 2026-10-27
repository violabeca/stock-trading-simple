# The five best strategies in this repo, and what they mean for live trading

*Status 2026-10-24. This is a summary for deciding what to trade. The evidence behind each number is in
`PROPOSALS_RESULTS.md` (the new strategies), `STRATEGY_OVERVIEW.md` (everything older) and each folder's
README. Alpha is the yearly return left after the book's market exposure: CAGR − beta × the CAGR of an
equal-weight buy-and-hold of the same stocks.*

---

## 1. The bottom line

- **One signal has survived everything this repo has thrown at it: short interest.** Stocks that short
  sellers avoid (low days-to-cover) beat stocks they crowd into. All five strategies below use it.
- **Its best form is a long/short book that combines short interest with 12-month momentum**
  (`xs_si_mom_composite`). It is the only strategy confirmed on a locked set of stocks, with the pass bar
  written down before the test. That set was fresh for the combination; informed_flow_ls had already used
  it for short interest alone.
- **These books make money when the market falls.** That holds only if they are not carrying market
  exposure when the fall begins. Right now the top-500 version is net long (beta +0.65). A market hedge
  fixes that (§4).
- **None of them beats buy-and-hold in a strong bull market.** They are a second source of return next to
  owning the market, not a replacement for it.
- **The edge was strongest in 2022-2025 and weak in 2026.** Expect less than the backtests. Paper-trade
  first: a year of paper trading tests the execution, not the edge.

---

## 2. The top five at a glance

| # | strategy | what it does | alpha a year, 2022-26 | in falling markets | vs buy-and-hold | ready? |
|---|---|---|---|---|---|---|
| 1 | **`xs_si_mom_composite`** | 15 long + 15 short, ranked on short interest + momentum | +4% to +13% on four groups of stocks; **+8.6% on the locked test** | made money in 2022 and Feb-Apr 2025 on every group | wins in falling years, loses in strong bull years | **paper-trade** |
| 2 | **`informed_flow_ls`** | 15 long + 15 short on short interest alone | +12.5% (top 500); +4.0% to +10.5% on smaller stocks | 2022 +7.8% (v1) while the market lost 18%; 2018-20 −4.8% | same profile as #1 | under review; re-run first |
| 3 | **`informed_flow`** | 30 longs only, the least-shorted stocks | +8.8% (but beta 1.14) | **loses money** with the market (beta 1.1-1.3) | 19.7% vs 9.6% a year (2022-26); weak since 2025 | re-run first |
| 4 | **`xs_portable_alpha`** | 45% in SPY + a long/short short-interest/momentum overlay | +2.4% to +5.0% net of SPY | loses only 28-64% of the market's fall | 10.1% vs 13.7% a year (2018-26), with ~0.4 of the drawdown | paper-trade (bigger account) |
| 5 | **`adaptive_hedge`** | short-interest longs + shorts of breaking-down stocks | +11.4% on the top 500; **−4.8% on small caps** | COVID +20.7%, 2022 +25.9% | 5.9% vs 13.8% a year (2018-26) | large caps only; a crisis sleeve |

---

## 3. Each strategy: what works, what does not

### 1. `xs_si_mom_composite`: short interest × momentum, long/short

**What it is.** Every day, each stock gets two ranks:
- how lightly it is shorted (days-to-cover, averaged over 42 days);
- its 12-month return after removing its market moves (residual momentum).

The book buys the 15 stocks with the best average rank and shorts the 15 with the worst, with equal
dollars on each side (90% of the account in total). Orders go at the next open. Nothing in it was tuned.

**What works:**

| stocks | alpha a year | with fills 50 bp worse |
|---|---|---|
| top 500 | +12.9% | +10.8% |
| ranked 501-1000 | +4.2% | +3.0% |
| ranked 1001-1500 (locked) | +12.4% | +10.1% |
| ranked 1501-2000 (locked, the pre-declared test) | +8.6% | +6.5% |

- It beat 100 out of 100 copies that use the same rules on random stocks, on each of the three sets where
  that was tested (top 500 and both locked sets).
- **Falling markets, on every group of stocks:**
  - 2022 bear: +4.5% to +15.4%, while those markets lost 16-25%.
  - Feb-Apr 2025: +1.0% to +6.7%, while they lost 20-22%.
  - In the average down month it made +0.8% to +1.6%, while the market lost about 4%.
- Costs are not the risk: doubling borrow fees costs 0.5-0.8 points a year, and trading whole shares in a
  $33k account up to 1 point. Every recent short was borrowable at IBKR, mostly at 0.3-0.4% a year.

**What does not:**
- **Before 2022 the edge was small:** +2.0% a year in 2018-21, and −0.4% with worse fills.
- **2026 has been bad on the top 500:** both halves lost about 22% a year of alpha. The other groups were
  mixed.
- **It is not market-neutral on the top 500 right now.** Equal dollars long and short is not zero market
  exposure. Its beta drifted from −0.40 to +0.73 by half-year, and it has been net long since mid-2025
  (+0.65). That is the state it was in during Q4 2018, when it lost 7.0%. On smaller stocks it stays near
  zero.
- **It trails buy-and-hold in strong bull years.** 2018-26 on the top 500: 8.9% a year against 13.7% for
  equal-weight buy-and-hold and 15.0% for SPY.
- **The simpler versions also passed the locked test.** The short-interest book alone did as well on the
  smallest stocks (+10.9%). Combining the signals clearly helped on one group of stocks, not all.
- Its worst drawdown was −21.9% (top 500, 2025-10 → 2026-07).

**For live trading:** the best candidate. Paper-trade it, decide the stocks to trade (§6), and decide on
the market hedge (§4).

### 2. `informed_flow_ls`: short interest alone, long/short (under review)

**What it is.** The same book as #1 with one signal: long the 15 least-shorted, short the 15 most-shorted
stocks of the top 500, dollar-neutral, with borrow fees from IBKR's quotes and a squeeze stop at +100%.

**What works:**
- **Walk-forward 2022-26:** CAGR +13.6%, alpha +12.5% (P 0.996), beta 0.12, max drawdown −14.0%.
- It beat all 100 random-stock copies.
- **Stresses:** +11.5% with double costs, +12.3% with double borrow, +8.6% with fills 50 bp worse.
- **The bear year:** in v1's record for 2022, its short side made +11.3% while the market fell 18%, and
  the book made +7.8%.

**What does not:**
- **The edge is smaller on stocks it was not designed on:** names 501-1000 +4.4%, 1001-1500 +4.0%,
  1501-2000 +10.5%, and 2020-21 +3.3%.
- **2018-20, with two sell-offs, was −4.8%.**
- Without its five biggest winners the alpha is +6.8%.
- 2026-H2 was −25.1%.

**For live trading:** a simpler alternative to #1 with a thinner fresh-stock record. It was measured on
the old engine (§7), so it needs a re-run before any money.

### 3. `informed_flow`: short interest, long only

**What it is.** It holds the 30 least-shorted stocks of the top 500 and nothing short. It needs no margin
and no borrow.

**What works:**
- Walk-forward 2022-26: CAGR +19.7% a year against +9.6% for buy-and-hold and +12.0% for SPY.
- Alpha +8.8%, and it beat all 500 random copies.

**What does not:**
- **Its beta is 1.14, so it falls with the market:** 2022 lost money even though alpha was positive.
- **The edge has faded since 2025:** alpha +1.4% (P 0.56) from 2025-01 to 2026-09, and 2026 −9.4%.
- The last partial period ran at beta 2.1.

**For live trading:** the easiest to run, but it is a leveraged bet on the market plus a small edge. It
gives no protection in a bear market. Use it only if you want to be long anyway, and re-run it on the new
engine first (§7).

### 4. `xs_portable_alpha`: SPY plus a market-neutral overlay

**What it is.** It holds SPY with 45% of the account. The rest runs the short-interest and momentum books
side by side, long and short (22.5% each).

**What works:**
- **Over 2018-26:** 10.1% a year with a Sharpe of 1.06, against buy-and-hold's 0.73 and SPY's 0.83, and
  about 0.4 of buy-and-hold's worst drawdown (−14.7% vs −37.4%).
- **In every fall it lost only 28-64% of the market's loss:** COVID −14.5% vs −37%; 2022 −6.8% vs −24.5%.

**What does not:**
- **It still trails buy-and-hold and SPY in a bull market.** It never beat SPY's CAGR in any test.
- Its overlay is the side-by-side books, not the #1 composite.
- It was sized for a ~$150k account.

**For live trading:** the right shape if the account must stay invested in the market. It gives a smoother
ride, not a higher return.

### 5. `adaptive_hedge`: short-interest longs plus breakdown shorts

**What it is.** It holds informed_flow's longs. Up to 20 slots go to shorts of stocks that are breaking
down: a new 27-day low with a strong downtrend.

**What works:**
- Top 500: alpha +11.4% (P 0.985).
- The best falling-market record in the repo: COVID +20.7%, 2022 +25.9%, positive in 82% of the market's
  down months.

**What does not:**
- **It fails on small caps:** −4.8% a year on names 1501-2000, with a −57% drawdown from short squeezes.
- **It is net short (beta −0.28), so it compounds slowly:** 5.9% a year (2018-26) against 13.8%.
- Its "adaptive" idea failed: the shorts earn because they are always on.

**For live trading:** a crisis sleeve for large caps only, not a standalone strategy.

---

## 4. The market hedge: what it is

**What it is.** It shorts an index fund in the amount that cancels the book's market exposure, so the book
earns only its stock-picking alpha.
- A book with beta +0.65 behaves as if 65% of its value were invested in the market.
- To cancel that, short 0.65 × the account value of an index ETF, and re-size the hedge at each monthly
  rebalance from the book's measured beta.

**For #1 on the top 500 today**, on a $33k account with beta +0.65:
- **Size:** short about 0.65 × $33k ≈ **$21k** of an index ETF. Use RSP (the equal-weight S&P 500, which
  matches how the beta is measured) or SPY.
- **Gross exposure:** the account then carries $15k long, $15k short and the $21k hedge, about 1.55× its
  equity. A margin account allows that (Reg T allows 2×).
- **Borrow:** IBKR quotes about 0.25% a year for SPY and 0.41% for RSP (2026-10-21), with millions of shares
  available.
- **Smaller stocks:** IWM would be the hedge, at about 0.68% a year.

**What it does:**
- **In a fall:** the hedge gains about beta × the market's fall. With beta 0.65 in a −20% market, that is
  about +13 points that the unhedged book would have lost.
- **In a rally:** it gives the same amount back. That is the point: the book's return becomes its alpha.
- **What it cannot do:** it does not fix a stock-picking loss. The 2026 losses on the top 500 were alpha,
  not market moves.
- **Which books need it:** the smaller-stock versions of #1 already sit near zero beta and need little or
  no hedge.

**What is not done yet:**
- The hedge has not been backtested. xs_three_book tried an index short capped at 8% of the book, which was
  too small to hold beta near 0.15.
- A proper test hedges the measured beta in full. It can run only on the open test periods, because the
  locked ones are spent.
- The engine models no margin (gross stays at ~100% of the account), so the backtest would shrink the stock
  book to make room. For example: 60% in stocks plus up to 40% in the hedge.
- Futures (micro S&P or Russell contracts) are more capital-efficient than an ETF short. But each contract
  has a fixed size (the index level × its multiplier), so check that it divides the hedge finely enough. A
  small account may only be able to hedge in large steps.

---

## 5. What does not work (and why)

| family | folders | why it fails |
|---|---|---|
| Intraday (1-minute and 5-minute) | `level_flow`, `gap_and_go`, `burst_scan`, `classic_signals`, `tori_trades_trendline`, `candle_scalp`, `keltner_reversion`, `squeeze_breakout`, `volume_momentum`, `fade_movers`, `lead_lag`, `trend_following_1m` | Direction is unpredictable intraday (only the size of a move is). The real effects found (0.4-7 bp) are smaller than the 6-40 bp a retail order pays. |
| Machine-learning forecasts | `darts_forecast` (TSMixer, TimesFM, gradient boosting, ridge) | Nothing beats "no change" out of sample. |
| Other daily factors | `quality_junk`, `value_factor`, `turnover_momentum`, `momentum_swing`, `earnings_drift` | No edge after costs here, or it reverses on the most-traded names. |
| Short-term reversal | `peer_reversal` | Some selection skill, but ~3,100 orders a year cost twice what it earns. |
| Market timing and crash protection | `etf_trend`, the 200-day filters | They protect in long bear markets (2008) and give it back in normal years. No alpha. |
| Momentum alone | `resid_momentum` | A real 2022-25 edge, negative in 2016-21. It is useful only inside #1. |
| Other combinations | `xs_three_book`, `xs_cross_leg`, `xs_three_premia` | Failed on small caps, a falsified idea, or diluted the alpha. |
| Options as signals or protection | `option_signals`, `options_hedge` | Puts and straddles cost what the moves are worth. Option-implied signals did not predict returns. |

Two lessons repeat across all of them:
- **Tuning almost always fitted noise.** The strategies that survived are fixed designs.
- **Costs decide everything below a day.**

---

## 6. What this means for live trading

1. **Start with one book: #1.** Paper-trade it for at least 6-12 months before real money.
   - Half-year alphas swing by 10-23 points, so a year of results cannot prove the edge either way.
   - Paper trading proves the execution: the data refresh, the fills and the borrow.
2. **Choose the stocks.**
   - The top 500 has the largest alpha, but it is partly in-sample, lost both 2026 halves, and is net long
     beta today: hedge it.
   - Stocks ranked 1001-2000 have the cleanest record: both locked tests positive, beta near zero. They also
     have wider real spreads and some expensive borrows: business development companies at 3-9% a year.
   - Paper books are free, so run both.
3. **The account:**
   - A margin account for the shorts, about $33k for 30 positions of about $1,000.
   - More if you hedge: the hedge adds about 0.65× the account on the top 500.
   - Keep the stock gross at 90%: one short squeeze (GME 2024) still part-filled there.
4. **The daily routine:**
   - Refresh short interest and share counts with `notebooks/03_altdata_update.py`, and read its coverage
     audits. A stale feed quietly shrinks the book.
   - Recompute at the close and send market-on-open orders.
   - Check borrow and availability with `scripts/ibkr_shortstock.py` before every new short.
   - Re-rank momentum on the month's last session; that needs an exchange calendar.
5. **Expect less than the backtest:**
   - the edge was strongest in 2022-25;
   - the database holds only stocks that still exist, which flatters backtests;
   - fills 50 bp worse cost about 2 points a year.
6. **Stop rules, written down before starting:**
   - fills worse than 50 bp a leg on average;
   - more than 3 of 15 shorts unborrowable or above 10% a year;
   - a drawdown beyond −25% (the worst test was −21.9%);
   - after 12 months, a cumulative alpha below zero.

---

## 7. What the engine changes of 2026-10-24 mean for these numbers

The seven approved engine fixes (#46-#52) are now in `src/`:
- covers execute before new shorts;
- an order the cash could not fill keeps working until it fills;
- no borrow over-charge on tiny re-trims;
- US holidays no longer enter panels;
- the six sector ETFs are no longer treated as stocks;
- adjusted price histories cannot be spliced.

What that means for each strategy:
- **#1 and the other lab strategies:** their numbers were produced by the lab, whose runner already had the
  first and third fixes. The one new effect is working orders. It changes a book only when an order could
  not be filled: in #1's case, the GME squeeze of 2024.
- **`informed_flow` and `informed_flow_ls`:** measured on the old engine, so both must be re-run before
  going live. `informed_flow_ls` runs at 98% gross, where the old engine starved shorts.
  - informed_flow's study was bumped to v42.
  - informed_flow_ls is under review; its owner should bump it.
- **Any book reading the sector ETFs:** they no longer enter point-in-time "top 500 stocks" sets from
  2024-H2, so those universes change slightly.
