# gap_and_go — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why).

## Done (moved out of the parking lot)

### ~~As-of-date universe~~ — partially implemented as `Morning200` (2026-07-28)

The Top20 snapshot bias turned out to be far worse than "worth pricing": that
universe's buy-and-hold hurdle was +460%/yr and it voided the entire v1 study.
`notebooks/01_universe_screener.ipynb` §8 now builds a universe whose *ranking*
window closes before the backtest opens. **What is still open** is the harder
half of the original idea: a *trailing* as-of-date universe recomputed monthly,
so membership tracks liquidity through the window instead of being fixed at its
start. Today's version still inherits pool membership from a 1m backfill list
assembled in 2026. Cheap to do (no new data), and the natural next step if
anything here ever looks shippable.

## Open questions

### Earnings-calendar conditioning — now the most promising lead

Earnings gaps and sympathy gaps continue differently in the literature, and the
strategy cannot tell them apart (no earnings-date feature exists). That is now
the leading *explanation* for v2's result, not just a caveat: the pooled event
shows a clean negative excess return that gets worse with gap size, which is
exactly what averaging two populations with opposite behaviour looks like. A
minimal `earnings_dates` feature builder (one date column per symbol; yfinance
carries it) would allow a `skip_earnings` / `only_earnings` ablation and could
split the pooled −10 bps into its parts. Feature builders are `src/` changes:
proposal-only.

### The ungated morning break — two measurements that disagree, and why

v2 produced two positive-looking results about dropping the gap gate, pointing
in **opposite directions**, and reconciling them is the most interesting open
question here.

* The **event study** likes the ungated break in the `go` direction: +2.43 bps
  market-excess at 120 bars, t = +2.97 on 38,263 events, rising with horizon.
* The **backtest** likes it in the `fade` direction: `gap_min: 0` with the
  picked trial's other parameters is the only configuration positive out of
  sample (+0.557 excess, +0.075 at the pass bar).

They are not measuring the same population. The event study row is genuinely
ungated (`max_candidates` = every symbol). The backtest row still has
`rank_by: gap` handing all five watchlist seats to the biggest gappers, so its
gap selection is intact — only the hard threshold is gone. It also carries a
4.5-ATR fixed stop and a hold-to-the-close horizon, which the event study has
no equivalent of.

Two cheap experiments would separate them: (a) run the event study with
`rank_by` set to something gap-free and see whether the sign follows the
ranking or the threshold; (b) test the ungated `go` break in
`donchian_breakout` (drop `skip_open`, add an opening-range variant), where it
belongs if it is real. Until one of those runs, neither number is evidence.

### Gap-size interaction with the range length

Big gaps may need a longer range (more fade risk to filter), small gaps a
shorter one. The space searches `or_minutes` and `gap_min` independently; if
the v2 front shows structure between them, a conditional rule (`or_minutes` as
a function of gap size) is the shaped version — only worth it if the front
actually shows it. Note v2's evidence points the other way: bigger gaps are
monotonically *worse*, with no sign of a length that rescues them.

### The v2 winner's shape is suspicious in a specific, testable way

The search converged on a **wide fixed stop** (4.5 ATR, `sl_trail: false`) with
a **hold-to-the-close horizon** (228–379 bars of a 390-bar session), and
`tp_ratio` made almost no difference between 0.0 and 3.5 — the target never
fired. Strip the labels and that is "enter in the morning, sit until the
close, only bail on a 4.5-ATR move against". Such a configuration earns most of
its return from *being in the market*, not from the entry signal, which would
explain both the beautiful in-sample Sharpe (2.60) and the out-of-sample
collapse. A direct test: compare it against a random-entry control with the
same holding period and slot count. If the random control matches it, the entry
signal contributed nothing and the whole v2 winner was an exposure bet.

### A conviction score beyond `rank_by`

`rank_by` offers `gap`, `rvol` and their product — all frozen at bar K−1, all
crude. A richer watchlist score (gap normalized by the name's own overnight-gap
distribution, so "3% for GLW" and "3% for IREN" are not treated as the same
event) would make `max_candidates` seats go to genuine outliers rather than to
whichever names are simply more volatile. Same causality constraint: anything
in the score must be knowable at K−1.

## Known caveats (documented, not actionable here)

- The 2 bps default spread understates opening friction 2–5×, and *every* entry
  here is in that window — the README's env-override cost-stress rerun is the
  gate, not a formality.
- `Morning200`'s ranking is point-in-time; its membership is not (see above).
- 1m intrabar path ambiguity is worst at the open: judge on the time-stop/EOD
  exit population, not tight-bracket win rates.
- Shorts assume borrow. Under `fade` the short leg is the gap-*up* leg, which
  is exactly where hard-to-borrow bites.
