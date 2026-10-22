# burst_scan — pending ideas / proposals / thoughts

Parking lot for ideas worth keeping but not (yet) implementing. Nothing here is a commitment; move an item out
when it is implemented or rejected (and say why). Each idea has a *why* and the falsification that would settle it.

## Phase 1 verdict (2026-09-27): both sides FAIL — see README "Results"

Bursts predict MOVEMENT (hit rate of either barrier ~1.6× an ordinary bar, t 5.7–10.1; beats a same-size move on
normal volume, t 3.1–4.8) but not DIRECTION (win vs stop t −0.8 … +1.0; demeaned return ≈ 0). The classifiers
learned movement too (win rate rises across deciles, returns do not; long and short coefficients on the
diagonal). Nothing below is a fix to try on the same data until it is re-registered.

## Open questions

1. **G0's pick rule is direction-blind.** It ranks detector settings by win-rate lift over placebo A, but bursts
   raise the hit rate of BOTH barriers (`burst_scan_events.ipynb` §5: the long win rate and the short stop rate are
   the same number), so the rule tends to pick the most *volatile* setting, not the most *directional* one. On a
   40-name smoke run the long side's pick failed "wins beat stops" while its config detector passed. Candidate
   replacement: rank by the lift of (win − stop), or by the demeaned return, over placebo A. **Needs a new
   pre-registration and a study bump** — never swap it in after seeing which setting wins.
2. **The opposite-entry rule costs trades.** In one wallet an opposite-side signal on a held name closes the
   position and does not open its own. Measured in `burst_scan_diagnostics.ipynb` §1 (count and the labels' mean
   return of the trades that never opened). Options: separate wallets per side (two cash accounts — an engine
   change, proposal only), or suppress an opposite pick while the name is held (the label then matches the book).
   Settles by: the priced cost of the rule in §1 against the complexity.
3. **The same-clock baseline goes stale after news.** After a big news day a name's volume stays several times its
   10-session norm for days, so every re-cross of the threshold fires "a burst" (visible in the visual notebook).
   Candidate: divide the burst ratio by the session's cumulative relative volume so far ("a burst relative to
   today"). Falsification: G0's lift and win − stop gap at the primary cell, same pick rule, new study.
4. **Train direction on direction alone** (was: a two-stage model). Phase 1 showed the single "target first"
   label is mostly "did it move": the models learned movement. A direction model should see only events that
   DID reach a barrier and learn win vs stop (for one detector the long and short labels are then complements,
   so it is one direction model read two ways — the sides stay separate through their own detectors). Needs a
   new pre-registration: the same a–f gates on the resolved events, and a new study.
5. **Use bursts as a volatility signal.** They reliably precede big moves in either direction — what a long
   straddle or a volatility-scaled stop wants. Not tradable in this equity-only engine; as a FILTER it could
   widen stops or cut size of other strategies' entries right after a burst (measure on level_flow/gap_and_go).
6. **A real concurrent-position cap.** `max_positions` only sets the slot (1/N of FREE cash); with ~20 events a
   day the long book ends up nearly fully invested (beta 0.71). A cap needs position state (an engine hook) or a
   per-day signal budget inside `signals()` (momentum_swing IDEAS #9 has the same need).
7. **DONE 2026-09-27 — engine: `lock_cash=True` for wallet shorts.** Short proceeds were spendable, so longs
   bought with them, entries were skipped and covers split. Fixed in `backtest.run`/`sweep` (user-approved);
   this strategy's study bumped to v2 and its book re-scored (README). The same accounting in PORTFOLIO mode
   (`from_orders`, target weights — trend_following) is a separate, still-pending proposal.

12. **Move study (2026-09-28): occurrence yes, direction no** (README "Move study"). A big move within 4h is
   predictable beyond volatility (within-month AUC 0.716 vs 0.680 baseline, t 5.2 — busy days stay busy), but
   no price/volume metric predicts its direction (best AUC 0.549, hit rates 46–51%). Direction would need
   information outside OHLCV (news/earnings calendar, order flow, alt-data like informed_flow's short
   interest); occurrence is usable as a volatility filter or for sizing (#5).

13. **Move explorer (2026-09-28, corrected 2026-09-29): runs start earlier than 2% slices; confirmations don't
   pay** (README "Move explorer"). Both 5m and 1m: the trade from a real confirmation is ≈ 0 bp before 24 bp of
   costs, and what separates runs from failures is volatility and time left, not direction. Next ideas, each a new
   pre-registration: (a) confirmations only right after a 1-minute volume/range burst at the low (Fig 5: the burst
   IS the onset), stop at the low; (b) a volatility filter — take confirmations only in the top intraday-ATR fifth,
   where more runs complete — judged on trade RETURN, not hit rate; (c) scale to the move study's 200 names before
   believing any fifth.

14. **Extended hours (2026-09-29): they explain the open, they don't catch it** (README "The extended hours"). On
   two sets of eight pre-market names, the pre-market adds no direction at 9:30 and only a borderline occurrence
   gain (t 2.0 / 0.7); late-day and evening confirmations are ≈ 0 bp like the rest. Parked:
   * (a) the pre-market's volume ÷ a regular session (`pm_vol_share`, the best single "whether" feature, AUC 0.686)
     as one input to #5's volatility filter;
   * (b) evening runs cluster on news nights, so an earnings calendar is the missing input (see #11);
   * (c) **any extended-hours rule must mask 08:00-08:09** (`research.mask_late_reports`): the stale 08:00 prints
     alone manufactured a −33 bp, t −13.5, every-name "pre-market reversal".

15. **Onset identification at 1 minute (2026-09-30): closed** (`ONSET_REPORT.md`). The existing features detect
   activity, not direction; a coin-flip market reproduces ~96% of the detection edge; no knob, pocket or
   signed-flow feature adds direction; nothing trades. Parked, each needing NEW information: (a) quote-based
   order flow (bid/ask imbalance) — the one information class 1m OHLCV lacks; (b) real 1-minute market/sector
   bars (ETF or futures) as a leading index; (c) an earnings/news calendar (#11). Usable now: a small
   volatility forecast (recent 15/60-min range, the name's time-of-day range profile, clock) for stops and
   sizing (#8).

16. **Two move detectors (2026-09-30): closed** (README "Two move detectors", `move_detectors.ipynb`). A LONG and a
   SHORT detector ("+/-2% within 4 h"), trained independently on 51 multi-scale features, are significant alarms for
   a move in EITHER direction and identical in what they use (volatility + time of day); no knob or feature makes
   them directional. ~~The notebook stays as the harness for any new information~~ **Not as a direction test**
   (#17): its 2%/4h direction AUC with 51 features misses even a planted 35-40 bp edge. Judge new information on a
   30-60 minute IC / P&L against a market-preserving coin flip instead.

17. **Predicting direction (2026-10-01): closed** (`DIRECTION_REPORT.md`, README "Predicting direction"). Nothing
   predicts the up/down direction of the 5 names from 1-minute OHLCV, other names' bars, extended hours or daily
   history. That covers ~11,000 cells and two rounds, plus Darts models (classical, global ML, deep, Chronos-2 /
   TimesFM). All 11 frozen rules failed on the locked months and fresh names. Two findings worth keeping:
   * (a) **Power.** The study's tests find 2-5 bp edges, so the null is real.
   * (b) **The one robust RELATIVE effect.** Late-day residual moves reverse overnight across the cross-section:
     IC +0.039 (t 6.7) on discovery, +0.019 (t 1.5) on the locked months, and the liquid book −2 bp per leg.
     It is not tradable as frozen. Revisit it only as a NEW pre-registration on new data, as a cross-sectional
     book of many names. It cannot be a 5-name rule.

   Parked, each needing new information (see #15): quote/order-book flow, an earnings calendar WITH surprises,
   options IV/skew, and futures or BTC at second resolution.

## Phase 2 (only for a side that passes)

8. **Risk-per-trade sizing**: `min(1/max_positions, risk / stop)` as a per-entry `cash_frac` frame (level_flow's
   rule) instead of equal slots.
9. **Asymmetric barriers** (`tp_ratio`): a target wider than the stop suits a low-win-rate, high-payoff side.
10. **A small search** over `trade_q`, `move_atr` and the horizon — never the detector (frozen by G0).
11. **An earnings-day feature**: gap_and_go's leading explanation for its failure was that earnings and sympathy
   gaps behave oppositely; a burst on an earnings day is a different event. Needs an earnings calendar feed.
