# darts_forecast — pending ideas / proposals / thoughts

Parking lot. Nothing here is a commitment; move an item out when it is implemented or rejected (and say why). Each
idea has a *why* and the falsification that would settle it.

## Engine proposals (proposal-only: `src/trading_simple/` is not touched without approval)

1. **A `features.py` builder for the free options store.** It would let `tsim optimize` (headless) see the option
   inputs: `data.features: [options]` → `panel.features['opt_*']`, built by the same code as
   `research.option_frames` (tz-aware UTC session frames, next-session lag, 5-day staleness cap).
   - **Why:** today only the notebooks attach them, so a headless study runs without options (it warns).
   - **Settles by:** the headless study reproducing the notebook study's first trial.
2. **Half-day closes in the stored session flag (pending engine proposal #32).** The stored "regular" 1m panel
   keeps 13:00–15:59 bars on NYSE early-close days. `strategy.HALF_DAYS` masks them here by hand; the fix
   belongs at ingestion (`rth = false` after an early close), with the calendar in one place.
3. **`plots.plot_equity` subtitle holds two "$" amounts.** "Realized P&L ($X) vs fees paid ($Y)" makes plotly.js
   typeset the text between the two `$` as TeX wherever MathJax is loaded (Jupyter), garbling the title. Fix: write
   `$` as `&#36;` in subplot titles (plotly renders the entity as "$").
4. **A stop touched inside the bar of a planned exit fills at the stop, not at that bar's open.** vectorbt checks the
   stop before the signal exit, so a planned time exit (or reversal) whose bar later trades through the stop is
   booked at the stop price, the worse one (4 of 98 planned exits on TEST, 8–60 bp each; 10 of 249 on TUNE). It is
   conservative and small. Check whether vectorbt can let a same-bar signal exit at the open win over the stop.

5. **Dividend-adjust intraday bars in `load_panel` (pending engine proposal #36).** It applies each row's own
   `adj_close / close`, and 1-minute rows carry `adj_close = close`, so a multi-day intraday return across an ex-date
   shows the dividend as a loss. Fix: take the factor from the daily bars by session date (what
   `research.adjust_dividends` does today, opt-in, relative to the panel's last session).

## Model ideas

6. **Native LightGBM** (`LightGBMModel`) beside `gbm`. It needs `libomp` on macOS: preload torch's copy before
   importing lightgbm. Same algorithm family as `HistGradientBoosting`, so expect the same forecasts; only worth it
   for speed.
7. **Probabilistic gbm.** Quantile `HistGradientBoostingRegressor` heads (0.1 / 0.5 / 0.9) would give a P(up) for
   the regressor, or Darts' `ConformalQRModel` for calibrated intervals around any model.
   - **Settles by:** a better calibration table in the model notebook.
8. **Chronos-2 with covariates** (zero-shot) as a no-training baseline. The Darts track of the direction study
   found nothing zero-shot without covariates.
9. **Multi-horizon in one model** (`output_chunk_length = horizon`), instead of one shifted point.
10. **Cross-sectional target.** Predict each name's return minus the cross-name mean. The direction study's one
   robust effect was relative, not absolute, though it needs a book of ~30 names, not 5.

11. **Restore the best epoch after early stopping** (TiDE / TFT with `val_days > 0`). `EarlyStopping` stops but keeps
    the weights of the last epoch, `patience` (2) epochs past the best validation loss. Darts' `save_checkpoints=True`
    + `load_from_checkpoint(best=True)`, or a small callback that keeps the best `state_dict` in memory, would fix it.
    - **Settles by:** the validation loss of the kept weights equals the best epoch's.

## Input ideas

12. **Order-book / quote inputs** (order-flow imbalance per grid bar), when a source exists. These are the
    information class 1-minute OHLCV lacks.
13. **Earnings calendar** (days to the next report, last surprise) as a future / daily input.
14. **Open interest by strike as a shape** (distance to the largest OI strikes, the "pin") rather than one
    gamma number.

## Process

15. **One study per forecast config.** The study searches only the trading rule. Comparing forecast configs
    (model, grid, horizon, cadence, inputs) belongs in the model notebook, on the same TUNE forecasts and the same
    null. A config is promoted only if it beats the coin-flip null there.
16. **Re-centre the trading-rule search on a forecaster that has an edge.** With the gbm forecasts of v2/v3 every
    trial with `edge_bp` above ~16 bp (and most with `health_min` above ~0.04) trades under the 300/yr floor and is
    pruned: the forecasts rarely exceed 15 bp. Narrowing the ranges to this forecast's distribution would only
    tune the search to a model with no edge; do it (new study) for the first forecast config that beats the null.

## From the 1-hour / 5-day walkthrough (2026-10-04)

17. **More names, no backfill.** About 510 symbols have 1-minute bars over TUNE. Five co-moving names give about
    52 independent 5-day outcomes; 50 such names count as about 5 independent stocks against 2.9 (1.7× the evidence,
    a reliably found edge of IC 0.19 instead of 0.25). More with less correlated names.
    - **Settles by:** the walkthrough's detectable IC on a wider universe.
18. **A daily-bar model for 5-day questions.** The stored daily bars reach back to 2015: about 25× the
    independent weeks of the hourly data. Hourly detail can then only add intraday timing on top.
19. **gbm overfits at a 35-bar horizon.** Theil U is above 1 for every name and the classifier's P(up) points the
    wrong way, although each leaf holds several horizons of rows. Pre-register one of: size the minimum leaf in
    independent outcomes (decision sessions, not rows), or train on one decision per session.
20. **TiDE's 80% range held 25.5% of the outcomes** in one `RUN_DEEP` run: far too narrow. Calibrate the ranges
    (Darts' `ConformalQRModel`) before reading TiDE's P(up) or coverage.
21. **An earnings input.** The DB's `earnings` table (yfinance; report date, no time of day) has every report since
    2015. Windows holding a report are 9% of the forecasts and 17% of the squared 5-day moves.
22. **Daily inputs from the daily bars.** The warm-up rebuilds 20-day inputs from 1-minute bars and spends 21 of
    TUNE's 156 sessions; the stored daily bars would give them from the first session.
23. **Labels that land on a half-day's empty buckets** are dropped (3 decisions per name per half-day) although the
    price there is known (the close). Use the last real close at or before the label time.
24. **Multi-day trading.** `trade_frames` is flat by the close and its health gate counts only completed outcomes;
    holding a 5-day forecast needs overnight positions, a hold that spans sessions and a gate over completed 5-day
    outcomes. Only worth building for a forecaster that passes the walkthrough's verdict.
25. **Next pre-registration: "beats every baseline" on value minus luck centre.** Slow guesses (drift, momentum)
    carry a short-sample bias the walkthrough's luck band now rebuilds; comparing raw ICs (the current, literal rule)
    can flatter or punish them.


## From the explorer (2026-10-13; 12 picked stocks, Aug 2025 → Jan 2026)

26. **Nothing beats "no change", at any bar size or length** (path explorer, TiDE, 60 of 93 feasible
    configurations): 3 above R² 0 by at most +0.0003, 18 scaled to "no change" outright, no direction above luck
    (best z 2.0 against a best-of-60 bar of 3.3), nothing held on TEST. The first, single-point explorer (gbm,
    30-minute bars) found the same: no input or output horizon above luck.
    - **Settles by:** more independent outcomes, more months (the window can start in July 2025) or more stocks
      (about 270 pass the data rules; ~1,750 since March 2026), then the same search.
27. **A cross-sectional path target.** Each stock's path minus the average path of all the stocks: removes the
    market, which no stock's own history predicts, and leaves what the explorer could in principle rank. The
    direction study's one surviving lead (relative overnight reversal) was cross-sectional too.
28. **The scale β is one number per configuration**, fit on the held-out sessions. Per-lead scales (the grade
    weighs every lead equally) or a conformal calibration of the paths' spread would let a model keep its short
    leads and drop its long ones; only worth it once some lead time shows information.
29. **The luck band flips one coin per bar.** That keeps overlap and co-movement but not the bar-to-bar dependence
    of the actual moves (1-minute bounce). The direction statistic (a correlation) is robust to it in the mean; a
    block coin (one per path length) would also keep the dependence in the spread. Check before reading a z near
    the bar.
30. **Multi-scale inputs for the trees** (the single-point explorer's gbm read dense lags: a 10-day window = 130
    lags x 10 inputs). Returns and volume over the last 1, 2, 4, 8 ... bars would cost log(N) inputs instead of N.
31. **The last 6 months are the locked months.** 1-minute coverage widened from ~515 to ~1,750 names in March
    2026, so the explorer's natural "recent" window (Apr → Sep 2026) is the study's TEST + HOLDOUT. Decide once
    whether to spend them on exploration (`READ_LOCKED = True`) or keep them for a frozen config.


## From the intraday study (2026-10-13; `darts_forecast_intraday.ipynb`, same 12 stocks and window)

32. **Power before models.** With 12 stocks (≈ 5.5 independent: their intraday moves correlate ≈ 0.1) and 3
    VALIDATION months, a 30-minute path must reach R² ≈ 0.005 to be seen as one pre-registered test and ≈ 0.01 as
    the best of the grid; the planted control (§7) shows the pipeline recovers about half of a planted R² of
    0.01-0.03. Documented intraday effects on single stocks are of order 0.001.
    - **Settles by:** more stocks (about 270 pass the data rules), the market-relative target (27), and monthly
      expanding refits inside VALIDATION, before any new model.
33. **Train on the graded target.** The truth starts at the FILL (the next trade), the training target at the
    decision bar's close, so a model that learned the bid-ask bounce pays for it at the first checkpoints. A target
    whose first output step starts at the next bar's open would align training and grade.
34. **Half-days back in.** They are left out (2 of 64 VALIDATION sessions, plus the 2 after them): a "minutes to
    the real close" input known in advance would let a half-day be predicted to its 13:00 close, and the next
    morning read yesterday's end correctly.
35. **An earnings-day input** (static, from the DB's earnings dates) and the grade without earnings days: the
    walkthrough found earnings weeks carry 17% of squared 5-day moves; intraday they decide single stocks' R².
36. **A better market input.** `mkt` is the other 11 picks' average move; an equal-weight 1-minute return of the
    ~266 names that pass the data rules, cached once, is the market every stock could lead or lag. SPY 1-minute
    bars exist only from 2026-03-25 (the locked months).
37. **Opening decisions (09:40, 09:50).** The gap's effect concentrates in the first half hour, which decisions at
    10:00 and later never read as an outcome. They are not bar closes on 3-minute bars: drop 3 minutes there, or
    read the decision between two closes.
38. **Per-checkpoint scales** (as 28): one β per configuration and month cannot keep short checkpoints and drop
    long ones.
39. **The only lead is weak, daily and model-free: a gap fade.** In the map (TRAIN + VALIDATION, 102 sessions × 12
    stocks, a sign-flip luck band per day) the night against the whole session correlates −0.11 (z −2.2), negative
    in all 5 months (−0.03 to −0.18); no cell clears the best-of-91 bar (the night → 14:00 cell is z −2.5, a few
    earnings days). The grid's best cells were the 10:00 → close forecasts (the gap is a static input), and the
    winner failed TEST.
    - **Settles by:** the gap against open → close (and against 10:00 → close) on the stored daily bars since 2015,
      for the ~266 names that pass the data rules: by year, stock minus market, earnings days apart. Only daily bars
      carry years of power, and they hold the official open and close.
