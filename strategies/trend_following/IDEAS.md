# trend_following — pending ideas / proposals / thoughts

Parking lot for ideas about this strategy that are worth keeping but not (yet)
implementing. Nothing in here is a commitment; move an item out when it is
implemented or rejected (and say why). Give each idea a *why* and, where possible,
the falsification that would settle it — "try X" without "X wins if / dies if" is
not an idea, it is a mood.

## Open questions

1. **A walk-forward before anything else.** The shipped config rests on one validation year
   (2024, used for the design decisions) and one holdout (2025-01 → 2026-09, now read once,
   +6.1%/yr excess — half of it one name, LITE). The honest next check is a rolling
   walk-forward over 2022-2026: re-run the study on each trailing two-year window, trade the
   next six months, stitch the out-of-sample record. *Wins if* the stitched excess is > 0 with a
   block-bootstrap P(excess > 0) ≥ 90% and survives leaving out its best five names. *Dies if*
   not — then the edge is 2022 plus one 2025 winner.

2. **Which exit is the strategy actually using?** In the shipped config opposite starts close 51
   of 68 positions; the 14-ATR chandelier is a disaster stop. Two cleaner designs would separate
   the jobs: a Turtle-style `exit_window` (exit on a close beyond the opposite N-bar extreme,
   N searched separately from the entry), or an ATR-only exit (opposite starts ignored). *Test:*
   paired screen, ≥ 60 random configs, train only, against the current rule. *Wins if* the
   paired mean improves with t > 2 and the walk-forward (1) agrees.

3. **The short side is a hedge that bleeds.** Breakdowns to new lows *outperformed* the market over
   the next month in 2022-23 (relative −0.2 to −0.9% for the short at 21 days, t down to −2.9) —
   short-term reversal in liquid large caps. Shorts made 28% of the book in 2022 and gave back 19%
   and 7% in 2023-24. Ideas, each to be event-tested on train first: (a) short only names whose
   12-1 month momentum is already negative (a filter, not just the admission rank); (b) wait for
   the first failed bounce (a lower high after the breakdown) before shorting; (c) keep shorts as
   a hedge but at half a slot. *Wins if* the filtered short starts show a positive relative
   forward return at 21 AND 63 days with monthly-clustered t > 2.

4. **Survivorship understates the short side.** `securities` holds no delisted names, so the
   pool is survivor-only: the stocks that collapsed and delisted — the best shorts — are missing.
   Any verdict on shorts waits for a point-in-time pool with delisted names.

5. **An intrabar chandelier (engine proposal, not implemented).** The stop is closing-basis
   because the engine's `sl_stop` fixes its distance at entry. vectorbt's `from_signals` takes an
   `adjust_sl_func_nb` callback; an engine hook accepting a per-bar stop-PRICE frame would let the
   chandelier fill intrabar at its level. Proposal-only (engine change). Probably moot: every
   tight stop lost here (2-5 ATRs: −7 to −22%/yr of excess on train), and at 14 ATRs intrabar
   versus close changes little. *Test first:* the atr_mult ablation with intrabar fills; drop the
   idea if tight stops still lose.

6. **Volatility-scaled slots (untested).** The Turtles sized so one ATR moved every position by the
   same fraction of equity. Here: weight ∝ 1/ATR%, normalized so the book stays ≤ 100% gross,
   fixed at entry. *Wins if* a paired screen lowers volatility without lowering excess (t > 2).

7. **A concentration cap would fight the thesis.** LITE alone made +23.5% of the book on the
   holdout; portfolio mode's `retrim_k` could trim such a winner back to its slot. Trend following
   lives on those outliers, so expect it to hurt. *Test:* paired screen with `retrim_k` 2 and 3;
   adopt only if excess does not fall and the holdout's leave-top-5-out number rises.

8. **The shipped pick sits next to two cliffs.** `opt.stability_report` (notebook §10) calls
   `entry_window` and `adx_min` cliffs on their HIGH side — 14 bars or ADX 40 drop the train
   excess from +5.7% to about −3.2%/yr — and suggests re-centering one step clear (7 and 31);
   13 of 16 neighbours still beat buy-and-hold. It was NOT re-centered: the holdout had already
   been read, so any move now is tuning against it. *Test:* the walk-forward (1) with the rule
   "re-center one step clear of a cliff" inside each fold; adopt it only if the folds say so.

9. **Is excess the right objective for this book?** Beta swung from −0.68 (2023) to +0.25
   (holdout), so `excess_cagr` against a long-only index mostly measures which way the market
   went. `alpha_cagr` (the repo's sanctioned alternative, used by `level_flow`/`informed_flow`)
   charges for the beta actually run. *Test:* a study on `[alpha_cagr]` alone under a new study
   name, judged by the walk-forward (1). It has no exposure knob (`max_positions`, `direction` are
   fixed), so the CLAUDE.md warning about pairing alpha with volatility does not bite.

## Rejected — with the measurement (do not re-try without new evidence)

* **Market-regime gate** (longs only while the equal-weight index is above its N-day SMA, shorts
  only below): paired over 60 random configs on train, −5.6 points of excess at 200 days
  (t −5.7), −4.5 at 100 (t −5.5), −0.5 at 50. It pins the book short into bear-market rallies.
* **Breakeven lock** (stop no worse than entry once the trade is 1R/2R ahead): inert (+0.3 / 0.0
  points) — the ratcheted chandelier already implies it.
* **40 slots** (v4): the top-10 study trials fell from +11.1% to +0.9%/yr excess on 2024 and the
  pick's neighbourhood was no smoother (every knob a spike). Diluted, not stabilized.
* **Admission by highest ADX** (v3): no better than random on train (+0.3 points, t 0.4).
  **Low-volatility first:** −2.4 points (t −2.8).
* **Relative strength over the entry window** (v5): fitted 2022-23 (train q90 +11.5%) and its
  best train trials lost 22-36%/yr against buy-and-hold in 2024 — one-month returns reverse.
* **The $10k wallet with `cash_frac`** (v1-v2): with a wide stop the first ~20 entries of the
  window owned the book (later entries got 0.03% each); starting in July instead of January
  turned +14.9%/yr into −24.4%/yr. Fixed slots (portfolio mode) replaced it.
