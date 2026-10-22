# classic_signals — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why). Give each idea a *why* and, where possible,
the falsification that would settle it — "try X" without "X wins if / dies if" is
not an idea, it is a mood.

## Built (2026-10-17)

- **Two optimization passes with the blended score** (below): `tuning.two_passes` and the
  tuning notebook, sections 5 and 7. **Judged on NOW (2026-10-18)**, by the criterion below:
  first traded session 120, re-tuned every 20 sessions, 80 + 80 trials, everything free but the
  candle size. (c) w 0.5 beat both (a) w 0 and (b) w 1 only with 15-minute candles; with 1- and
  5-minute candles pass 1 alone (w 0) did best. So the blend **fails on NOW**; GEV and RGTI not
  run. Every difference is within luck anyway (README, "The full optimization of NOW"). The blend's
  w also does not set the mix: at w 0.5, seed 0's pass 2 ranked its trials like the long score
  (rank correlation 0.95), seed 1's like the recent one (0.96), whichever part spread more.

- **Removed (2026-10-18, user):** pass 2 and the blend, after the test above; the tuning now
  chooses on the long history (the training window) alone. The objective became the daily
  Sharpe ratio with a luck line, every trial is scored on a validation window (2026-07-01 →
  2026-10-16) beside it, and data from 2026-10-17 on is the final check (README, "Tuning").

## Proposed after the full NOW run (2026-10-18; not built, one at a time)

The free tuning fits noise (README): too many decisions for one stock's 373 sessions. Each idea
below is judged by the same walk forward against no tuning on the same sessions.

1. **Tune less: about 4 numbers.** Free: band length, band width, stop, holding limit; the
   filters *always off*, the earnings guard *always on*, the candle size fixed. Needs no code:
   the overrides of the tuning notebook's section 2. *Why:* at the in-sample optimum 11 of 35
   settings did nothing, and in the walks the switches flipped between re-tunings (the earnings
   guard on in 15% of one walk's tunings, 100% of another's). *Wins if* the small space beats no
   tuning with all three candle sizes; *dies if* not: then NOW has nothing to tune.
2. **BUILT 2026-10-18 (user: NOW + GEV first):** one setting for several stocks, every trade
   sized by its risk ($100 at the first stop), scored on the stocks' daily profits added up;
   stops and target in typical daily moves instead of percent. **Pool the three stocks: one
   setting, scored on NOW + GEV + RGTI together.** *Why:* every NOW
   winner lost on GEV and RGTI, and one stock-year cannot tell a $2,000 edge from luck (the luck
   band is ±$2,500–4,500). Three times the trades for every decision. It reverses "each stock
   its own settings" (2026-10-16); per-stock settings can come back later as small departures
   from a shared one. Needs `tuning.py` to score several labs at once. *Wins if* the pooled walk
   beats no tuning on all three stocks.
3. **Pick plateaus, not peaks.** Re-rank pass 1's best ten by the median score of each one and
   its one-step neighbours (a minute more per tuning). *Why:* seed 0's winner collapsed one step
   away. *Wins if* it beats the plain pick in the walk forward.

## Two optimization passes: the long history, then a moving window (user, 2026-10-17; step 5)

**The idea.** Pass 1 tunes the levers on the whole history before a trading day: more data,
less fitting of noise. Pass 2 re-tunes them on a short moving window before each trading day:
closer to live trading, so the levers adapt to the current market and may differ from one
window to the next. Open question: how to combine the two.

**Why.** The long history protects against fitting the noise of a few weeks; the recent window
can follow a market that really changed. Each pass alone gives up one of the two.

**Ways to combine them** (notes from the discussion):

1. **Anchor and neighbourhood.** Pass 1 keeps not one best set but the region of sets that score
   within a few percent of the best on the long history; pass 2 searches only inside it. The
   recent window can steer the levers, but not wander off.
2. **Blended score.** Pass 2 picks the set with the best `w × score(recent window) + (1 − w) ×
   score(all history before it)`. `w = 0` is pass 1 alone, `w = 1` the pure moving window; `w`
   itself is chosen by the walk-forward, never by hand on the same days. This is "a set that
   works OK in both, with a weighting".
3. **Not recommended: averaging the two passes' lever values.** It breaks for whole-number levers
   (closes in a row) and where the best region is not smooth: the average of two good sets can
   be a bad one.

**How to test it fairly.**

- "The long history" always means the history UP TO the trading day (a growing window), never
  the whole study period, or pass 1 knows the future of every window it guides.
- Run three arms on the same out-of-sample days: (a) pass 1 alone, (b) the pure moving window
  (20, 40, 60, 120 days), (c) the blend for a few `w`. Stitch each arm's days into one record.
- Draw the chosen levers window by window: levers that jump between far-apart values from one
  day to the next mean the moving window is fitting noise. Steps 2 and 3 already measured noisy
  scores (one step of a lever moved the total by thousands of dollars, with no steady
  direction), which favours a large weight on the long history unless the market really changes.
- Give every window a minimum number of trades per day, or its score means nothing.

**Wins if** (c) beats both (a) and (b) on the stitched out-of-sample days, for both stocks.
**Dies if** it does not: then the simpler arm that won is the procedure.

## Stock groups: next steps (2026-10-18; the grouping is built, these are not)

- **Wire the groups into the tuning notebook**: a "stock group" choice that loads a group's
  *tune* members as the labs and judges the winner on its *check* members (training and
  validation). Each stock then needs a sector fund for the sector-trend rule: the database has
  daily candles only for IGV, XLI, QTUM and SPY. Proposal: download the 11 SPDR sector funds
  (Yahoo daily, quick) and give each stock the fund its daily returns follow most in the
  training window.
- **Do group-mates share their best settings?** The grouping assumes stocks alike in character
  want alike settings; that is untested. Test: a fixed sample of ~100 random settings scored on
  every stock of a few groups; the rank correlation of the settings' scores between two stocks
  of the same group against two of different groups. **Wins if** same-group pairs agree
  clearly more; **dies if** not (then the groups pool evidence, but not because one setting
  fits them better).
- **Point-in-time groups for a walk forward**: the groups read all the data (since 2026-10-20,
  validation months and final check included); a walk forward could re-form them each block
  from the sessions before it (the sums are per session, so this is cheap), which is also the
  way to get the validation back out-of-sample for the grouping step.

## More from the option chains (2026-10-18; BUILT the same day, user)

Built as switchable rules (*option walls*, *put/call*, *dealer gamma*, plus *iv term*), with
implied volatilities and Greeks solved from the mid quotes; measured in the README ("Everything in
the chains, measured"): volatility information only. The parked notes, kept for the record:

- **Option walls as levels**: the strikes with the most open interest above and below the
  price, as ceilings and floors for the level filters or the profit target (prices are known to
  cluster at large strikes near expiry). *Wins if* adding them to the levels beats the major
  levels alone in the walk forward.
- **Put/call volume** as a sentiment extreme (contrarian): the direction study found options
  carry no direction for these names, so a low prior.
- **Dealer gamma** (positive: moves get damped, fades should work; negative: moves run,
  breakouts should): needs Greeks, so an implied-volatility solver over every contract, and an
  assumption about who holds what. The most work, the most assumptions.

## Open questions

- A fade counterpart to the breakout lever (fade only after short stretches outside the band, or
  N closes back inside): offered 2026-10-17, not asked for.
- A "no new trade in the last minutes of the session" lever, the mirror of the wait after the
  open: not asked for.
