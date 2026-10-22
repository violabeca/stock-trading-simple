# trend_following_1m — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why). Give each idea a *why* and, where possible,
the falsification that would settle it — "try X" without "X wins if / dies if" is
not an idea, it is a mood.

## Open questions

1. **The money was overnight — test that, not intraday trends.** In train these ten stocks earned
   +11.6 bp a day on the overnight gap and lost −8.0 bp during the session (the overnight/intraday
   split of Lou, Polk & Skouras 2019). The study's highest-alpha configurations (+10-18%/yr on
   train) all held positions overnight. A strategy built on that would be *different*: hold across the close only
   while a (daily-scale) trend is up, flat during the session. *Test:* event study on train — the
   overnight return conditional on the daily trend state vs unconditional, day-clustered t > 2 —
   before any backtest. Beta will be the whole story unless the conditioning adds something.

2. **Breadth before granularity.** Ten highly correlated tech/fintech names give ten slots that never
   compete, so the relative-strength admission — the one component that clearly added value on daily
   bars — is inert. `Morning200` (point-in-time, 1-minute bars) with 10-20 slots would switch it back
   on. *Wins if* a paired screen (≥ 60 random configs, train only) beats this ten-name book on alpha
   with t > 2.

3. **An intraday ATR without the overnight gap.** `bt.atr` puts the opening gap into the first bar's
   true range, so every morning's ATR — and every stop placed before the spike decays — is inflated.
   `level_flow` uses a per-session ATR. *Test:* paired screen, flat-at-close configs only; adopt if
   alpha improves with t > 2 (the event test in the notebook says it cannot rescue the entries).

4. **Fade instead of follow?** Intraday starts showed no continuation, and short starts were followed
   by a small relative bounce. `gap_and_go` found its opening-range breakout FADES. *Test:* the §4
   event table with the sign flipped, on train only — only a day-clustered t > 3 beyond the ~8 bp
   round trip justifies building a fade.

## Tried — with the measurement

* **Five- and fifteen-minute bars** (derived from the 1-minute history, scratch studies of 300
  trials, the same search spans in time): 5 minutes — train alpha +13.0%/yr but day-level t 1.18,
  ORCL alone +8.3% of the book's +9.7%, holdout +2.9% at t 0.47; 15 minutes — not one of 167
  flat-at-close trials had positive alpha, the pick lost alpha on the holdout. Coarser intraday
  bars trade less and lose less to costs; none shows an edge distinguishable from zero (README
  "Granularity").
