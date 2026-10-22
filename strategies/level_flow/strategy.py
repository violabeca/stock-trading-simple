"""Level flow: multi-week price levels x today's activity, long and short, flat by the close.

The thesis, in the order the evidence supports it:

1. **Levels are where things happen, not which way.** Horizontal support/resistance —
   here the HSAR construction of Zapranis & Tsinaslanidis (2012): swing highs/lows of
   coarse candles over the last ``level_days`` sessions, binned by ``bin_pct`` in log
   price, a bin with at least ``min_touches`` pivots is a level — predicts intraday trend
   *interruptions* far better than chance (Osler 2000; Garzarelli et al. 2014, where the
   bounce probability rises with the number of prior touches). A level has TWO polarities
   (Osler 2003): take-profit orders cluster *at* it, so trends reverse there (``polarity:
   bounce``); stop orders cluster just *beyond* it, so a trend that gets through
   accelerates (``polarity: break``). The polarity is searched, never assumed, and the
   notebook's event study decides before the backtest is trusted.
2. **Geometry alone has no edge net of costs** (Marshall-Cahan-Cahan 2008 on 7,846
   intraday rules; Schulmeister 2009; this repo's own five 1m strategies). What the
   literature does find is that intraday OHLCV signals only work under **elevated
   activity** — the "stocks in play" result (Zarattini-Barbon-Aziz 2024). So each session,
   at bar ``scan_minute``, the names that clear the liquidity floors and have a level within
   ``reach_atr`` session-ATRs of the price are ranked by ``rank_by`` (relative volume by
   default) and only the top ``max_candidates`` may trade that day. A day with no such name
   is a flat day, which is the intended default outcome, not a failure.
3. **Risk is thesis invalidation, bounded, and never overnight.** The stop sits
   ``sl_atr`` session-ATRs beyond the level the trade rests on — on the far side from the
   entry, outside the touch band by construction (Kaminski-Lo 2014: a stop inside a
   reversion entry's noise band sells the bounce) — the target is ``tp_ratio``
   times the stop, ``horizon`` minutes is the time stop, and every position is flat at
   the session's last bar — shorts included, so no borrow accrues. A setup whose stop
   would be wider than ``max_stop`` of price is skipped rather than sized down, and every
   entry deploys ``min(1/max_candidates, risk_per_trade / stop)`` of the free cash (the
   engine's per-entry ``cash_frac`` frame), so one trade cannot lose more than
   ``risk_per_trade`` of the book except through a gap. Shorts obey Reg SHO Rule 201: a
   name down 10% from its prior close is not shorted that session or the next.

Multiple timeframes, all derived from the execution grid and all in minutes: ``tf_levels``
is the candle whose swings define the levels; ``signal_tf`` (0 = the execution bar) is
the candle on whose closes the trigger is read; ``trend_tf`` (0 = off, the one switch) is the
candle of the higher-timeframe EMA trend that ``trend_gate`` requires the trade to run with
(or against). Nothing from an unfinished candle ever reaches a decision. The execution bar
itself is the panel's (``data.resample`` in the config derives 5m/15m bars from 1m).

Every distance is a fraction of the SESSION-scale ATR (Wilder over ``atr_days`` sessions
of session true range, shifted one session) — never the 1-minute ATR, so ``touch_tol``,
``break_tol``, ``sl_atr`` and ``reach_atr`` mean the same thing on any bar size.

Causality notes: the level table for session d is built only from pivots *confirmed*
(``pivot_k`` candles later) by the end of session d-1; the live close only selects which
frozen row is read. The watchlist is ranked on the scan bar and held. The stop distance is
computed at the signal bar and shifted onto the fill bar, which is where vectorbt reads it.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trading_simple import backtest as bt
from trading_simple.data import Panel

_POLARITIES = ("bounce", "break")
_DIRECTIONS = ("long", "short", "both")
_VWAP_GATES = ("none", "above", "below")
_TREND_GATES = ("none", "with", "against")
_REGIMES = ("none", "active_only", "calm_only")
_RANKS = ("rvol", "touches", "rvol_x_touches")
#: Reg SHO Rule 201 trigger: this far below the prior close, a name is uptick-only for the
#: rest of the session and the whole next one. The strategy simply does not short it.
_SSR_DROP = 0.10
#: Sessions in the realized-volatility reference the regime gate compares against.
_REGIME_REF_SESSIONS = 60
#: Floor on a stop distance (fraction of price) — a zero stop is not a stop.
_MIN_STOP = 1e-4


def _minutes_per_bar(bar: str) -> int:
    """Minutes in one execution bar (``"1d"`` counts as one RTH day).

    Args:
        bar: Panel bar size (``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``).

    Returns:
        Minutes per bar.

    Raises:
        KeyError: On a bar size that is neither ``1d``/``1h`` nor ``<n>m``.
    """
    if bar == "1d":
        return 390
    if bar == "1h":
        return 60
    if bar.endswith("m") and bar[:-1].isdigit():
        return int(bar[:-1])
    raise KeyError(f"unknown bar size {bar!r}")


def _bars(minutes: float, bar: str, *, exact: bool = False) -> int:
    """Convert a window in minutes to a count of execution bars.

    Args:
        minutes: Window length in minutes.
        bar: Panel bar size.
        exact: Require a whole multiple of the bar (a candle knob), else round up (a
            lookback or horizon).

    Returns:
        The window in bars, at least 1.

    Raises:
        ValueError: If ``exact`` and ``minutes`` is not a multiple of the bar.
    """
    per = _minutes_per_bar(bar)
    if exact and (minutes < per or minutes % per):
        raise ValueError(f"{minutes}-minute candles cannot be built from {bar} bars")
    return max(1, math.ceil(minutes / per))


def _sessions(index: pd.DatetimeIndex, tz: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar session id (0-based, chronological) and bar number within the session.

    Args:
        index: The panel's UTC timestamp index.
        tz: Exchange-local timezone (``panel.tz``).

    Returns:
        ``(codes, bar_no)`` integer arrays.
    """
    day = pd.Series(index.tz_convert(tz).date, index=index)
    codes = pd.factorize(day, sort=True)[0]
    return codes, day.groupby(codes).cumcount().to_numpy()


def _gather(session_frame: pd.DataFrame, codes: np.ndarray,
            index: pd.DatetimeIndex) -> pd.DataFrame:
    """Spread a session-grid frame (one row per session, in order) onto the bar grid."""
    return pd.DataFrame(session_frame.to_numpy()[codes], index=index,
                        columns=session_frame.columns)


def _session_atr(panel: Panel, codes: np.ndarray, atr_days: int) -> pd.DataFrame:
    """Session-scale ATR on the session grid, shifted one session (known at the open).

    Wilder's smoothing over ``atr_days`` sessions of the session true range (high-low,
    or the gap from the prior close if larger). The one distance unit every tolerance
    and stop in this strategy is expressed in.

    Args:
        panel: Price panel.
        codes: Session id per bar (:func:`_sessions`).
        atr_days: Smoothing window in sessions.

    Returns:
        ``(session x symbol)`` frame, NaN through the warm-up.
    """
    hi = panel.high.groupby(codes).max()
    lo = panel.low.groupby(codes).min()
    prev = panel.close.groupby(codes).last().shift(1)
    tr = np.fmax(np.fmax((hi - lo).to_numpy(), (hi - prev).abs().to_numpy()),
                 (lo - prev).abs().to_numpy())
    tr = pd.DataFrame(tr, index=hi.index, columns=hi.columns)
    return tr.ewm(alpha=1.0 / atr_days, min_periods=atr_days).mean().shift(1)


def _candles(frames: dict[str, pd.DataFrame], codes: np.ndarray, bar_no: np.ndarray,
             tf_bars: int) -> tuple[dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """Session-aligned candles of ``tf_bars`` execution bars, on a bucket grid.

    Buckets start at each session's first bar (bucket ``i`` = bars ``i*tf .. i*tf+tf-1``),
    so a half-day's last partial candle stays inside its day. The bucket grid has a fixed
    number of slots per session; slots a session never fills are NaN.

    Args:
        frames: ``{"high" | "low" | "close" | "open" | "volume": bar-grid frame}``.
        codes: Session id per bar.
        bar_no: Bar number within the session.
        tf_bars: Candle length in execution bars.

    Returns:
        ``(candles, key, is_close)``: bucket-grid frames aggregated the way a candle is
        (high max, low min, close last, open first, volume sum), the bucket id of every
        bar, and the mask of bars on which a candle *completes* (its last bar).
    """
    nb = -(-(int(bar_no.max()) + 1) // tf_bars)
    key = codes * nb + np.minimum(bar_no // tf_bars, nb - 1)
    total = (int(codes.max()) + 1) * nb
    how = {"high": "max", "low": "min", "close": "last", "open": "first", "volume": "sum"}
    candles = {name: f.groupby(key).agg(how[name]).reindex(range(total))
               for name, f in frames.items()}
    is_close = np.r_[key[1:] != key[:-1], True]
    return candles, key, is_close


def _hsar_levels(bh: np.ndarray, bl: np.ndarray, close: pd.DataFrame, codes: np.ndarray,
                 nb: int, *, pivot_k: int, level_days: int, bin_pct: float,
                 min_touches: int, return_parts: bool = False) -> tuple:
    """Nearest defended level below and above every bar's close, from prior sessions only.

    The HSAR construction as a rolling histogram: swing highs/lows of the coarse candles
    (a candle whose high is the highest of the ``2*pivot_k+1`` around it, confirmed
    ``pivot_k`` candles later) are counted into log-price bins of width ``bin_pct``; the
    counts of the ``level_days`` sessions *before* session d are session d's level table;
    a bin with ``min_touches`` or more pivots is a level whose price is the mean of its
    pivots. A bar then reads the nearest valid bin strictly below its close bin (support)
    and strictly above it (resistance) — a level sharing the close's bin is neither, which
    keeps the two sides symmetric. Fully vectorized: ~1 s for 200 symbols x 100k bars.

    Args:
        bh: Candle highs on the bucket grid, ``(sessions*nb, symbols)``.
        bl: Candle lows, same shape.
        close: Bar-grid close frame (selects the row to read; sets the output index).
        codes: Session id per bar.
        nb: Buckets per session (candles per session slot count).
        pivot_k: Candles each side a swing must dominate.
        level_days: Sessions of history a level table looks back over (excluding today).
        bin_pct: Log-price bin width (0.005 = half a percent).
        min_touches: Pivots a bin needs to count as a level.
        return_parts: Also return the pieces the table is built from (for
            :func:`explain_levels`): the pivots, the touch tensor, the level prices, the
            per-symbol bin offset and the bin width.

    Returns:
        ``(support, resistance, sup_touches, res_touches)`` bar-grid frames; NaN / 0
        where no level exists on that side — plus the parts dict when asked for.
    """
    total, n_sym = bh.shape
    n_sess = total // nb
    k, w = pivot_k, 2 * pivot_k + 1
    piv_hi = np.zeros((total, n_sym), bool)
    piv_lo = np.zeros((total, n_sym), bool)
    if total >= w:
        # A NaN anywhere in the window (a missing candle) propagates and simply yields no
        # pivot there — fail-safe, and no all-NaN warnings.
        piv_hi[k:total - k] = bh[k:total - k] == sliding_window_view(bh, w, axis=0).max(axis=2)
        piv_lo[k:total - k] = bl[k:total - k] == sliding_window_view(bl, w, axis=0).min(axis=2)
    known = np.minimum((np.arange(total) + k) // nb, n_sess - 1)   # session a pivot is confirmed in
    gi, si = np.nonzero(piv_hi)
    gl, sl = np.nonzero(piv_lo)
    g, s = np.concatenate([gi, gl]), np.concatenate([si, sl])
    p = np.concatenate([bh[gi, si], bl[gl, sl]])

    lw = math.log1p(bin_pct)
    closes = close.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):   # a symbol with no candles at all is all-NaN
        lo_p = np.fmin(np.nanmin(bl, axis=0), np.nanmin(closes, axis=0))
        hi_p = np.fmax(np.nanmax(bh, axis=0), np.nanmax(closes, axis=0))
    lo_p = np.where(np.isfinite(lo_p) & (lo_p > 0), lo_p, 1.0) * (1.0 - bin_pct)
    hi_p = np.where(np.isfinite(hi_p) & (hi_p > 0), hi_p, 1.0) * (1.0 + bin_pct)
    off = np.floor(np.log(lo_p) / lw).astype(int)
    n_bins = int((np.floor(np.log(hi_p) / lw).astype(int) - off).max()) + 1

    counts = np.zeros((n_sess, n_bins, n_sym), np.int16)
    sums = np.zeros((n_sess, n_bins, n_sym), np.float32)
    if len(g):
        b = np.clip(np.floor(np.log(p) / lw).astype(int) - off[s], 0, n_bins - 1)
        np.add.at(counts, (known[g], b, s), 1)
        np.add.at(sums, (known[g], b, s), p.astype(np.float32))
    # Level table for session d = pivots known in sessions [d - level_days, d - 1].
    cum_n = np.cumsum(counts, axis=0, dtype=np.int32)
    cum_p = np.cumsum(sums, axis=0, dtype=np.float64)
    touches = np.zeros_like(cum_n)
    tsum = np.zeros_like(cum_p)
    touches[1:], tsum[1:] = cum_n[:-1], cum_p[:-1]
    if n_sess > level_days + 1:
        touches[level_days + 1:] -= cum_n[:-level_days - 1]
        tsum[level_days + 1:] -= cum_p[:-level_days - 1]
    del counts, sums, cum_n, cum_p
    valid = touches >= max(int(min_touches), 1)
    bins = np.arange(n_bins, dtype=np.int32)[None, :, None]
    sup_bin = np.maximum.accumulate(np.where(valid, bins, -1).astype(np.int32), axis=1)
    res_bin = np.minimum.accumulate(np.where(valid, bins, n_bins).astype(np.int32)[:, ::-1, :],
                                    axis=1)[:, ::-1, :]
    level_px = np.where(valid, tsum / np.maximum(touches, 1), np.nan).astype(np.float32)
    parts = {"pivots": {"bucket": g, "symbol": s, "price": p, "known": known[g],
                        "is_high": np.r_[np.ones(len(gi), bool), np.zeros(len(gl), bool)]},
             "touches": touches, "level_px": level_px, "off": off, "lw": lw}
    del valid, tsum

    with np.errstate(invalid="ignore"):
        cb = np.floor(np.log(closes) / lw)
    priced = np.isfinite(cb)
    cb = np.clip(np.where(priced, cb, 0).astype(np.int32) - off[None, :], 0, n_bins - 1)
    dd = codes[:, None]
    sym = np.broadcast_to(np.arange(n_sym, dtype=np.int32)[None, :], cb.shape)
    below, above = np.maximum(cb - 1, 0), np.minimum(cb + 1, n_bins - 1)
    sb = np.where(cb >= 1, sup_bin[dd, below, sym], -1)
    rb = np.where(cb + 1 < n_bins, res_bin[dd, above, sym], n_bins)

    def read(which, ok):
        idx = np.clip(which, 0, n_bins - 1)
        ok = ok & priced
        px = np.where(ok, level_px[dd, idx, sym], np.nan)
        n = np.where(ok, touches[dd, idx, sym], 0)
        return (pd.DataFrame(px, index=close.index, columns=close.columns),
                pd.DataFrame(n, index=close.index, columns=close.columns))

    support, sup_n = read(sb, sb >= 0)
    resistance, res_n = read(rb, rb < n_bins)
    if return_parts:
        return support, resistance, sup_n, res_n, parts
    return support, resistance, sup_n, res_n


def signals(
    panel: Panel,
    *,
    tf_levels: int = 30,
    pivot_k: int = 2,
    level_days: int = 15,
    bin_pct: float = 0.005,
    min_touches: int = 2,
    signal_tf: int = 0,
    trend_tf: int = 0,
    trend_len: int = 20,
    trend_gate: str = "none",
    touch_tol: float = 0.15,
    touch_minutes: int = 5,
    confirm_minutes: int = 0,
    break_tol: float = 0.10,
    reach_atr: float = 0.5,
    scan_minute: int = 30,
    skip_close: int = 15,
    vol_mult: float = 1.0,
    profile_days: int = 10,
    rank_by: str = "rvol",
    max_candidates: int = 5,
    min_price: float = 5.0,
    min_dollar_volume: float = 1_000_000.0,
    polarity: str = "bounce",
    direction: str = "both",
    vwap_gate: str = "none",
    range_used_max: float = 0.0,
    regime: str = "none",
    regime_days: int = 10,
    atr_days: int = 14,
    warmup_sessions: int = 20,
    sl_atr: float = 0.3,
    max_stop: float = 0.02,
    tp_ratio: float = 1.5,
    sl_trail: bool = False,
    horizon: int = 180,
    risk_per_trade: float = 0.004,
    ssr_guard: bool = True,
    eod_exit: bool = True,
) -> dict:
    """Execution-timed entries, exits, stops and per-entry sizes (see the module docstring).

    Args:
        panel: Intraday price panel (1m stored, or a coarser bar derived from it).
        tf_levels: Candle length in minutes whose swing highs/lows define the levels.
        pivot_k: Candles on each side a swing must dominate (confirmed ``pivot_k`` later).
        level_days: Sessions of history behind a session's level table (excluding today).
        bin_pct: HSAR bin width in log price; pivots within a bin are one level.
        min_touches: Pivots a bin needs to be a level (1 = any swing is a level: ablation).
        signal_tf: Candle length in minutes on whose closes the trigger is read; ``0`` =
            every execution bar.
        trend_tf: Candle length in minutes of the higher-timeframe trend; ``0`` = no trend
            gate (``trend_gate`` is then not read).
        trend_len: EMA length of that trend, in candles.
        trend_gate: ``with`` (a long needs the last completed trend candle above its EMA, a
            short below) | ``against`` (mirrored); ``none`` is accepted and means off.
        touch_tol: Touch band and reclaim margin around a level, in session ATRs.
        touch_minutes: How far back (minutes) the low/high may have touched the band.
        confirm_minutes: Bounce confirmation: the close must exceed the close this many
            minutes earlier (``0`` = off).
        break_tol: A break must close this many session ATRs beyond the level.
        reach_atr: A name is a scan candidate only if a level lies within this many session
            ATRs of its scan-bar close.
        scan_minute: Minute of the session at which the watchlist is frozen; entries only
            after it (the open blackout).
        skip_close: No entries in the last N minutes of the session.
        vol_mult: Relative volume at the scan bar (cumulative volume over its time-of-day
            expectation) a candidate needs; ``0`` = no activity gate (the control).
        profile_days: Sessions in the time-of-day volume baseline.
        rank_by: Watchlist order: ``rvol`` | ``touches`` (of the nearest level) |
            ``rvol_x_touches``.
        max_candidates: Watchlist seats per session; the slot size is ``1/max_candidates``.
        min_price: Price floor at the scan bar.
        min_dollar_volume: Dollar volume traded by the scan bar.
        polarity: ``bounce`` (fade the level) | ``break`` (trade through it).
        direction: ``long`` | ``short`` | ``both``.
        vwap_gate: ``none`` | ``above`` | ``below`` — a long needs the close on that side of
            the session VWAP (a short the opposite side).
        range_used_max: Skip a name whose session range already exceeds this many session
            ATRs; ``0`` = off.
        regime: ``none`` | ``active_only`` | ``calm_only`` — on the panel's equal-weight
            index realized volatility versus its trailing median.
        regime_days: Realized-volatility window in sessions.
        atr_days: Session-ATR smoothing window in sessions.
        warmup_sessions: No trading in the first N sessions (pinned, so every trial shares
            the same dead zone whatever its windows).
        sl_atr: Stop distance beyond the level, in session ATRs.
        max_stop: Skip a setup whose stop would be further than this fraction of price.
        tp_ratio: Target as a multiple of the stop; ``0`` = none.
        sl_trail: Trail the stop behind the best price.
        horizon: Time stop in minutes.
        risk_per_trade: Fraction of the book one stop-out may lose; sizes each entry as
            ``min(1/max_candidates, risk_per_trade / stop)`` of the free cash.
        ssr_guard: Do not short a name down 10% from its prior close (that session and the
            next). Value-only, always on.
        eod_exit: Flat at every session's last bar. Value-only, never off.

    Returns:
        ``{entries, exits, short_entries, short_exits, sl_stop, sl_trail, cash_frac}`` plus
        ``tp_stop`` when ``tp_ratio > 0``.

    Raises:
        ValueError: If a candle knob is not a whole multiple of the panel's bar.
    """
    return _compose(panel, **{k: v for k, v in locals().items() if k != "panel"})["signals"]


def _compose(panel: Panel, **p) -> dict:
    """Everything :func:`signals` computes, intermediates included (see :func:`compose`)."""
    bar = panel.bar
    index, columns = panel.close.index, panel.close.columns
    close, high, low, volume = panel.close, panel.high, panel.low, panel.volume
    codes, bar_no = _sessions(index, panel.tz)
    n_bars, n_sym = close.shape

    def wide(mask) -> np.ndarray:
        """One bool per bar -> full (bar x symbol) array, without copying."""
        return np.broadcast_to(np.asarray(mask)[:, None], (n_bars, n_sym))

    def first(trig: pd.DataFrame) -> pd.DataFrame:
        """First trigger of the session only: one event per (session, symbol, side)."""
        return trig & (trig.groupby(codes).cumsum() == 1)

    # --- the one distance unit: session-scale ATR, frozen per session -------------------
    atr = _gather(_session_atr(panel, codes, p["atr_days"]), codes, index)

    # --- levels from prior sessions' candle swings -------------------------------------
    lvl_bars = _bars(p["tf_levels"], bar, exact=True)
    candles, _, _ = _candles({"high": high, "low": low}, codes, bar_no, lvl_bars)
    nb = len(candles["high"]) // (int(codes.max()) + 1)
    support, resistance, sup_n, res_n = _hsar_levels(
        candles["high"].to_numpy(dtype=float), candles["low"].to_numpy(dtype=float), close, codes,
        nb, pivot_k=p["pivot_k"], level_days=p["level_days"], bin_pct=p["bin_pct"],
        min_touches=p["min_touches"])
    del candles

    # --- intraday context: the last few hours ------------------------------------------
    cum_vol = volume.groupby(codes).cumsum()
    vwap = (close * volume).groupby(codes).cumsum() / cum_vol
    range_used = (high.groupby(codes).cummax() - low.groupby(codes).cummin()) / atr
    prev_close = _gather(close.groupby(codes).last().shift(1), codes, index)
    expected = bt.volume_profile(volume, panel.tz, days=p["profile_days"]).groupby(codes).cumsum()
    rvol = cum_vol / expected
    ssr_now = low.groupby(codes).cummin() <= (1.0 - _SSR_DROP) * prev_close
    ssr = ssr_now | _gather(ssr_now.groupby(codes).last().shift(1, fill_value=False), codes, index)

    # --- session-level gates: warm-up and the market's volatility regime --------------
    warm = codes >= p["warmup_sessions"]
    if p["regime"] == "none":
        regime_ok = np.ones(n_bars, bool)
    else:
        ew_ret = close.groupby(codes).last().pct_change(fill_method=None).mean(axis=1)
        rv = ew_ret.rolling(p["regime_days"], min_periods=p["regime_days"]).std()
        ref = rv.rolling(_REGIME_REF_SESSIONS, min_periods=p["regime_days"]).median()
        active = (rv > ref).shift(1, fill_value=False).to_numpy()[codes]
        regime_ok = active if p["regime"] == "active_only" else ~active

    # --- higher-timeframe trend, from COMPLETED candles only ---------------------------
    trend = None
    if p["trend_tf"] and p["trend_gate"] != "none":       # trend_tf 0 is the off-switch
        tf_bars = _bars(p["trend_tf"], bar, exact=True)
        tcandles, tkey, _ = _candles({"close": close}, codes, bar_no, tf_bars)
        tclose = tcandles["close"]
        ema = tclose.ewm(span=p["trend_len"], min_periods=p["trend_len"]).mean()
        state = np.sign(tclose - ema).shift(1).ffill()         # the last finished candle's side
        trend = pd.DataFrame(state.to_numpy()[tkey], index=index, columns=columns)

    # --- the signal candle: triggers are read on its closes ----------------------------
    sig_bars = _bars(p["signal_tf"], bar, exact=True) if p["signal_tf"] else 1
    if sig_bars > 1:
        _, _, eval_pts = _candles({"close": close}, codes, bar_no, sig_bars)
    else:
        eval_pts = np.ones(n_bars, bool)
    touch_b = max(_bars(p["touch_minutes"], bar), sig_bars)
    dip = low.rolling(touch_b, min_periods=1).min()
    peak = high.rolling(touch_b, min_periods=1).max()

    # --- the tradable window and the scan bar -------------------------------------------
    scan_bar = _bars(p["scan_minute"], bar, exact=True)
    tradable = bt.session_mask(index, panel.tz, skip_close=p["skip_close"]).to_numpy()
    window = wide((bar_no > scan_bar) & tradable & warm & regime_ok)
    is_scan = bar_no == scan_bar

    # --- the watchlist: activity-ranked names with a defended level within reach -------
    gap_dn, gap_up = close - support, resistance - close
    near = np.fmin(gap_dn, gap_up) <= p["reach_atr"] * atr
    nearest_n = pd.DataFrame(np.where(gap_dn.to_numpy() <= gap_up.to_numpy(), sup_n, res_n),
                             index=index, columns=columns).where(near, 0)
    eligible = (wide(is_scan & warm & regime_ok) & near & (close >= p["min_price"])
                & (cum_vol * close >= p["min_dollar_volume"]))
    if p["vol_mult"]:
        eligible &= rvol >= p["vol_mult"]
    score = {"rvol": rvol, "touches": nearest_n.astype(float),
             "rvol_x_touches": rvol * nearest_n}[p["rank_by"]]
    ranked = score.where(eligible)[is_scan]
    on_watchlist = (ranked.rank(axis=1, ascending=False, method="first")
                    .reindex(index).groupby(codes).ffill() <= p["max_candidates"])

    # --- triggers ------------------------------------------------------------------------
    tol = p["touch_tol"] * atr
    confirm_b = _bars(p["confirm_minutes"], bar) if p["confirm_minutes"] else 0
    if p["polarity"] == "bounce":
        long_trig = (dip <= support + tol) & (close > support + tol)
        short_trig = (peak >= resistance - tol) & (close < resistance - tol)
        if confirm_b:
            long_trig &= close > close.shift(confirm_b)
            short_trig &= close < close.shift(confirm_b)
        long_raw = first(long_trig & wide(eval_pts) & window)
        short_raw = first(short_trig & wide(eval_pts) & window)
        lvl_long, lvl_short = support, resistance
    else:
        # The level a break is measured against is the one in force at the previous read
        # point, so the bar that crosses it still sees it (afterwards it is support). A
        # session's first close must be on the inside: a gap over a level is not a break.
        in_session = wide(bar_no > 0)
        prev_res = resistance.where(wide(eval_pts)).ffill().shift(1).where(in_session)
        prev_sup = support.where(wide(eval_pts)).ffill().shift(1).where(in_session)
        first_close = close.groupby(codes).transform("first")
        margin = p["break_tol"] * atr
        long_trig = (close > prev_res + margin) & (first_close <= prev_res) & wide(eval_pts)
        short_trig = (close < prev_sup - margin) & (first_close >= prev_sup) & wide(eval_pts)
        long_raw, short_raw = first(long_trig) & window, first(short_trig) & window
        lvl_long, lvl_short = prev_res, prev_sup

    # --- gates shared by both polarities --------------------------------------------------
    dist_long = ((close - lvl_long).abs() + p["sl_atr"] * atr) / close
    dist_short = ((close - lvl_short).abs() + p["sl_atr"] * atr) / close
    long_raw &= on_watchlist & (dist_long <= p["max_stop"])
    short_raw &= on_watchlist & (dist_short <= p["max_stop"])
    if p["vwap_gate"] == "above":
        long_raw, short_raw = long_raw & (close >= vwap), short_raw & (close <= vwap)
    elif p["vwap_gate"] == "below":
        long_raw, short_raw = long_raw & (close <= vwap), short_raw & (close >= vwap)
    if trend is not None:
        up, down = trend > 0, trend < 0
        if p["trend_gate"] == "with":
            long_raw, short_raw = long_raw & up, short_raw & down
        else:
            long_raw, short_raw = long_raw & down, short_raw & up
    if p["range_used_max"]:
        used = range_used <= p["range_used_max"]
        long_raw, short_raw = long_raw & used, short_raw & used
    if p["ssr_guard"]:
        short_raw &= ~ssr
    none = pd.DataFrame(False, index=index, columns=columns)
    if p["direction"] == "long":
        short_raw = none
    elif p["direction"] == "short":
        long_raw = none
    clash = long_raw & short_raw                     # one bar, both sides: take neither
    long_raw, short_raw = long_raw & ~clash, short_raw & ~clash

    # --- execution: next-bar fills, level-based stops, per-entry sizing, flat by close --
    entries, short_entries = bt.next_bar(long_raw), bt.next_bar(short_raw)
    horizon_b = _bars(p["horizon"], bar)
    exits, short_exits = bt.exit_after(entries, horizon_b), bt.exit_after(short_entries, horizon_b)
    if p["eod_exit"]:
        end = wide(bt.session_end(index, panel.tz).to_numpy())
        exits, short_exits = exits | end, short_exits | end
    stop = (dist_long.where(long_raw, dist_short.where(short_raw)).shift(1)
            .clip(lower=_MIN_STOP))                    # read at the FILL bar
    slot = 1.0 / p["max_candidates"]
    size = np.minimum(slot, p["risk_per_trade"] / stop).where(entries | short_entries).fillna(slot)
    out = {"entries": entries, "exits": exits, "short_entries": short_entries,
           "short_exits": short_exits, "sl_stop": stop, "sl_trail": bool(p["sl_trail"]),
           "cash_frac": size}
    if p["tp_ratio"]:
        out["tp_stop"] = p["tp_ratio"] * stop
    return {"signals": out, "support": support, "resistance": resistance, "sup_touches": sup_n,
            "res_touches": res_n, "atr": atr, "vwap": vwap, "rvol": rvol, "range_used": range_used,
            "on_watchlist": on_watchlist, "eligible": eligible, "long_raw": long_raw,
            "short_raw": short_raw, "trend": trend, "ssr": ssr, "regime_ok": regime_ok,
            "level_long": lvl_long, "level_short": lvl_short, "codes": codes, "bar_no": bar_no}


_DEFAULTS = {name: prm.default for name, prm in inspect.signature(signals).parameters.items()
             if prm.default is not inspect.Parameter.empty}


def compose(panel: Panel, **params) -> dict:
    """Expose the levels, context, watchlist and raw triggers for the notebooks.

    Same computation as :func:`signals` (config defaults under any overrides), returning
    every intermediate: ``support`` / ``resistance`` / ``sup_touches`` / ``res_touches``
    (bar-grid frames), ``atr``, ``vwap``, ``rvol``, ``range_used``, ``on_watchlist``,
    ``eligible``, ``long_raw`` / ``short_raw`` (signal-bar triggers, before the next-bar
    shift), ``trend``, ``ssr``, ``regime_ok``, the level each side rests on, and the engine
    dict under ``"signals"``. The event study reads the raw triggers; the symbol chart
    reads the levels.

    Args:
        panel: Price panel.
        **params: Overrides of :func:`signals`' keyword arguments.

    Returns:
        The intermediates dict described above.
    """
    return _compose(panel, **{**_DEFAULTS, **params})


def explain_levels(panel: Panel, symbol: str, session=None, **params) -> dict:
    """One session's level table for one symbol, with the candles and pivots behind it.

    The same code path as :func:`signals` (session candles, confirmed swings, the log-price
    histogram over the prior ``level_days`` sessions), run on that symbol alone — levels
    are per symbol, only the watchlist is cross-sectional — and opened up so a chart can
    show *why* a level exists: which candle swings were counted, which bins cleared
    ``min_touches``, and which swings the session could not yet know about.

    Args:
        panel: Price panel holding ``symbol``.
        symbol: The symbol to explain.
        session: Exchange-local session date (anything ``pd.Timestamp`` reads); ``None``
            = the panel's last session.
        **params: :func:`signals` keyword overrides.

    Returns:
        ``session`` (date), ``candles`` (``ts/open/high/low/close/session`` of the
        ``level_days`` sessions before the session and the session itself),
        ``pivots`` (``ts/price/side/session/known_session/counted`` — ``counted`` = inside
        the session's lookback and confirmed before it opened), ``levels`` (``level/touches/
        bin_low/bin_high`` the session reads), ``histogram`` (touches per bin across the
        candles' price range), the bar-grid ``support``/``resistance``/``sup_touches``/
        ``res_touches``/``atr`` series of the symbol, and ``params``.

    Raises:
        KeyError: If ``session`` is not a session of the panel.
    """
    p = {**_DEFAULTS, **params}
    one = Panel(close=panel.close[[symbol]], open_=panel.open_[[symbol]], high=panel.high[[symbol]],
                low=panel.low[[symbol]], volume=panel.volume[[symbol]], bar=panel.bar, tz=panel.tz)
    index = one.close.index
    codes, bar_no = _sessions(index, one.tz)
    dates = pd.Series(index.tz_convert(one.tz).date, index=index).groupby(codes).first()
    if session is None:
        d = int(codes.max())
    else:
        want = pd.Timestamp(session).date()
        hits = dates.index[dates == want]
        if not len(hits):
            raise KeyError(f"{want} is not a session of the panel "
                           f"({dates.iloc[0]}..{dates.iloc[-1]})")
        d = int(hits[0])
    lvl_bars = _bars(p["tf_levels"], one.bar, exact=True)
    candles, key, _ = _candles({"open": one.open_, "high": one.high, "low": one.low,
                                "close": one.close}, codes, bar_no, lvl_bars)
    total = len(candles["high"])
    nb = total // (int(codes.max()) + 1)
    support, resistance, sup_n, res_n, parts = _hsar_levels(
        candles["high"].to_numpy(dtype=float), candles["low"].to_numpy(dtype=float), one.close,
        codes, nb, pivot_k=p["pivot_k"], level_days=p["level_days"], bin_pct=p["bin_pct"],
        min_touches=p["min_touches"], return_parts=True)
    stamps = pd.Series(index, index=index).groupby(key).first().reindex(range(total))
    frame = pd.DataFrame({name: candles[name][symbol].to_numpy() for name in candles})
    frame.insert(0, "ts", stamps.to_numpy())
    frame["session"] = np.arange(total) // nb
    lo_s, hi_s = d - p["level_days"], d
    window = frame[(frame["session"] >= lo_s) & (frame["session"] <= hi_s)].dropna(subset=["close"])
    piv = parts["pivots"]
    pivots = pd.DataFrame({"ts": stamps.to_numpy()[piv["bucket"]], "price": piv["price"],
                           "side": np.where(piv["is_high"], "high", "low"),
                           "session": piv["bucket"] // nb, "known_session": piv["known"]})
    pivots["counted"] = (pivots["known_session"] >= lo_s) & (pivots["known_session"] <= d - 1)
    pivots = pivots[(pivots["session"] >= lo_s) & (pivots["session"] <= hi_s)]
    pivots = pivots.sort_values("ts").reset_index(drop=True)
    lw, off = parts["lw"], int(parts["off"][0])
    touches, px = parts["touches"][d, :, 0], parts["level_px"][d, :, 0]
    bins = np.arange(len(touches))
    edges = pd.DataFrame({"bin_low": np.exp((bins + off) * lw),
                          "bin_high": np.exp((bins + off + 1) * lw), "touches": touches})
    on = touches >= max(int(p["min_touches"]), 1)
    levels = edges[on].assign(level=px[on])[["level", "touches", "bin_low", "bin_high"]]
    in_range = ((edges["bin_high"] >= window["low"].min())
                & (edges["bin_low"] <= window["high"].max()))
    atr = _gather(_session_atr(one, codes, p["atr_days"]), codes, index)[symbol]
    return {"session": dates[d], "candles": window.reset_index(drop=True), "pivots": pivots,
            "levels": levels.sort_values("level").reset_index(drop=True),
            "histogram": edges[in_range].reset_index(drop=True),
            "support": support[symbol], "resistance": resistance[symbol],
            "sup_touches": sup_n[symbol], "res_touches": res_n[symbol], "atr": atr, "params": p}


def valid(params: dict) -> bool:
    """Reject incoherent combos (tolerates partial dicts from sweep grids).

    Deliberately thin. Every geometric relation is made unrepresentable rather than
    rejected: the stop sits on the far side of the level from the entry, so it is outside
    the touch band whatever ``sl_atr`` is; ``trend_tf: 0`` is the trend gate's one
    off-switch, so no ``trend_gate`` draw is inert; lookbacks may reach before the scan bar
    (that is past data). The first throwaway study rejected or starved 39 of 40 random
    draws with the earlier rules. What remains: the categoricals, and a warm-up that covers
    every session-level window so no trial trades on an unmeasured level, ATR or volume
    baseline.

    Args:
        params: Candidate parameter combo (may hold only the swept keys).

    Returns:
        True when the combo is coherent.
    """
    g = {**_DEFAULTS, **params}.get
    return (g("polarity") in _POLARITIES and g("direction") in _DIRECTIONS
            and g("vwap_gate") in _VWAP_GATES and g("trend_gate") in _TREND_GATES
            and g("regime") in _REGIMES and g("rank_by") in _RANKS
            and g("warmup_sessions") >= max(g("level_days"), g("atr_days"), g("profile_days"),
                                           2 * g("regime_days"))
            and g("max_candidates") >= 1 and g("max_stop") > 0 and g("horizon") >= 1
            and g("risk_per_trade") > 0 and g("sl_atr") > 0 and g("pivot_k") >= 1)


def indicators(
    close: pd.Series,
    volume: pd.Series,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    tz: str = "America/New_York",
    bar: str = "1m",
    **params,
) -> dict:
    """Per-symbol overlays: the levels in play, their touch counts, session VWAP, rvol.

    Args:
        close: One symbol's adjusted close.
        volume: One symbol's volume.
        high: One symbol's adjusted high (pass it — the levels come from true swings).
        low: One symbol's adjusted low.
        tz: The panel's exchange timezone.
        bar: The panel's bar size.
        **params: :func:`signals` keyword overrides.

    Returns:
        ``{"price", "volume", "volume_marks"}`` overlay series for ``plots.plot_symbol``.
    """
    high = close if high is None else high
    low = close if low is None else low
    one = Panel(close=close.to_frame("s"), open_=close.to_frame("s"), high=high.to_frame("s"),
                low=low.to_frame("s"), volume=volume.to_frame("s"), bar=bar, tz=tz)
    c = compose(one, **{k: v for k, v in params.items() if k in _DEFAULTS})
    return {
        "price": {"support": c["support"]["s"], "resistance": c["resistance"]["s"],
                  "session VWAP": c["vwap"]["s"]},
        "volume": {"rvol": c["rvol"]["s"]},
        "volume_marks": c["on_watchlist"]["s"],
    }
