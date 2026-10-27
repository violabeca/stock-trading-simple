# tori_trades_trendline — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet) implementing.
Nothing in here is a commitment; move an item out when it is implemented or rejected (and say
why). Give each idea a *why* and, where possible, the falsification that would settle it.

## Done (2026-10-19)

- **Her trade, the trendline bounce**: built (`bounce.py`, the trading notebook), long and
  short, and tested on 21 stocks with a knob grid, a placebo and two filters (`study.py`,
  README "The 21-stock study"). **No edge** on 15-minute charts from 2026-03-30 to 2026-09-29:
  every mode, setting and filter lost. Her exact lines did no better than lines moved off them.

## Done (2026-10-20)

- **Tuning every knob on many stocks at once** (`optimize.py`, then `tune.py` twice): no
  setting holds out of sample, and the 2025 check of `tune.py` v2 (pre-registered, one shot)
  passes none. The knobs that agree across windows are exit settings, not line settings (README
  "Second run").

## From her second video (2026-10-25)

Her video [*Top Down Analysis Isn't Hard. It's Misunderstood*](https://youtu.be/-emUlKR5n38)
(2026-10-22, gold) repeats the line rules as built: start at the extreme, as many touches as
possible with no candle through the line, every new point A is the previous point B, adjust the
lines on each lower chart. It adds four things:

- **A 30-minute chart** between the 1-hour and the 15-minute chart: monthly, weekly, daily,
  4-hour, 1-hour, 30-minute, 15-minute. `topdown.TIMEFRAMES` has no 30-minute chart (the first
  video did not use one). On her gold chart it added nothing. Here it could: the 15-minute
  screen covers about 6 sessions and the 1-hour screen about 23, so a 30-minute screen (about
  12) can adjust or add lines between them. Adding it is two entries in `topdown.py`, but it
  changes every line drawn so far: every cached line and earlier study result would be stale.
- **Support and resistance, "the bonus step"**: horizontal levels on the trading chart only.
  - **Where:** places price reached several times and turned around, or where it moved
    sideways, as close to today's price as possible. Price may poke through such a level.
  - **In a range:** a ceiling and a floor.
  - **After a breakout:** one key level, where she watches for a rejection, or a break and a
    retest.
  - **How firm:** she calls these rules "more subjective, not as systematic". She uses the
    levels as context for entries and trade management, not as a signal of their own.
  - **Prior result here:** `level_flow`'s horizontal levels (a pivot histogram) did no better
    than a placebo of matched locations (2026-09-19).
  - **Lives if:** a bounce at a line that sits near a level beats a bounce away from any level,
    and beats levels moved off hers, on data not yet read.
- **Direction.** She says the top-down analysis tells "which way the markets are actually
  moving", but gives no rule for it. The `trend` filter (a moving average) is a stand-in here.
  A rule in her spirit would be the direction of the latest lines on the higher charts. It is
  a candidate metric for the regime study (`regimes.py`).
- **The rest of her system is not public.** She calls the top-down analysis "just one part of
  my system": entries, trade management, exits and risk are taught in a paid masterclass. The
  bounce rules here come from her first video's walk-through. "No edge" is a verdict on those
  public rules, not on everything she trades.

Both videos describe touches as swing lows and swing highs ("simply capturing swing lows" on
the higher charts). That supports the parked *swing-only touches* below.

## A slower trading chart

On the 15-minute chart a stop half a typical candle beyond the line is 0.47% of the price, so a
round trip's costs take 0.13–0.17 R from every trade. On the 1-hour or 4-hour chart the stops are
wider and costs a smaller share of R. **Lives if** `study.py` with `tf="1H"` (or `"4H"`) beats
zero before and after costs on both halves; **dies if** longs still lose before costs, as they
did on 15 minutes (−15 bp a trade), where costs were not the whole story.

## A longer and a falling window

The study read one six-month window in a strong rising market (buy-and-hold +20%). The 1-minute
history reaches back to 2024-12-26 for 18 of the 21 stocks; 2025's February–April fall would test
the shorts in a falling market. Same pre-registered score; **lives if** a side beats zero there
and in the 2026 window.

Partly done: `tune.py` v2 read 2025 once for its nine stocks (nothing passed). The 21 study
stocks have not been replayed in the February–April 2025 fall.

## Backfill NOW's daily candles before 2015

She starts from the very first candle. NOW listed on 2012-06-29, but the database's daily
candles start 2015-01-02, so the first upward line starts at the February 2016 low, not the 2012
lows. Fix: a daily backfill from 2012 through `notebooks/02_backfill.ipynb`. It changes only the
first upward line or two; the steep lines near today come from the lower charts.

## The breakout setup

She says price either respects a line or violates it, and only teaches the bounce. A close
through a line ends her trade; whether that same close is an entry the other way is a separate
idea, with the same placebo test. The study's exits hint at it: trades that ended on a close
beyond the line averaged +0.34 R (long) and +0.67 R (short) — but that is the bounce's own
profit before the break, not the move after it.

## Swing-only touches

Touches are any candle within *close enough*. If the lines look too eager in tight
consolidations (many touches, B jumping to the end of a flat stretch), require touches to be swing
points instead: the high (low) of `k` candles on each side. That rule must keep "each line steeper
than the one before" true: a swing touch can sit further inside the line than a later non-swing
candle.
