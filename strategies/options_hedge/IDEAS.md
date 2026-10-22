# options_hedge — parked ideas

Nothing here is implemented. Each needs its own pre-registration before it is scored. The
first test (README) found no screen that beats random days, so step 2 below is not open.

1. **Step 2: the exit rules** — only if a screen passes discovery AND the holdout. Compare on daily
   bid / ask marks: the user's rule (stock + ATM put, sell the stock after a rise of X, sell the put
   on the way back; mirror rule on a drop), the straddle held to expiry, and an early exit. Needs a
   second pass over the chains keeping the bought contracts' daily quotes (the cache keeps only each
   day's ATM straddle).
2. **Intraday entries.** The repo's movement signals work over hours (burst_scan move study), but
   the chains are end-of-day, so every entry here waits for the next close. Testing an intraday
   entry needs intraday option quotes, which nothing in the repo has.
3. **Sharper "unusual options activity".** The screen uses total volume of all contracts. Variants:
   volume ÷ open interest, the share of short-dated or out-of-the-money volume, call vs put volume.
   Each one is a new cell, so it shares the multiple-testing budget.
4. **Earnings straddles** are a known separate game (options priced up before the release, cheap
   after). The notebook reports earnings-window events for information only.
5. **TESTED 2026-10-19 (`options_premia.ipynb`, test A):** the causal `iv_own` version did not pass;
   the Goyal-Saretto `hv_iv` version beat random straddles in discovery AND on the holdout (README).
   Next, if pursued: a vega-neutral long-short book (buy the cheap tenth, sell the dear tenth) with
   margin and the short leg's tail priced in, and an execution study (fills inside the spread would
   decide it: the edge is about one spread). Needs intraday option quotes and a longer history than
   the MarketData trial's one year. **Original note — cheap options, causally.** In hindsight, each name's cheapest third of days returned +2.0% at
   the ask and its dearest third −8.6% (30d base rate). That split uses the whole window, so it is not
   tradable as it stands. The causal version: the implied move (straddle mid ÷ price) against the
   name's own trailing median, bought when it is low. It is a different question from this folder's
   (it ignores the activity screens) and would need its own pre-registration and the locked holdout.
6. **Any follow-up should buy the liquid strike** (`rs.LIQUID`: the nearest strike whose straddle is
   within the spread filter). The pre-registered "nearest two-sided strike" lands on fresh $1 strikes
   with no open interest and drops more than half of the tradable name-days.
