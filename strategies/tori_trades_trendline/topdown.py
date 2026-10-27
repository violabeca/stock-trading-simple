"""Tori Trades' top-down trendlines, drawn by rule.

Source: the Tori Trades video "Once you master this strategy, trading becomes EASY" (YouTube,
2026-09-24, https://www.youtube.com/watch?v=TuXOgkcYw9E). She draws the lines by hand in
TradingView; this module turns each of her instructions into a rule, for one side at a time:
downward lines lie on the candles' highs, upward lines under their lows.

1. **A line is a ray**: point A, point B, extended to the right without end.
2. **The first line starts at the extreme of all history** on the monthly chart: the highest
   high for the first downward line, the lowest low for the first upward line.
3. **No candle may cross a line.** From point A the ray is turned as steep as it can go until a
   candle stops it: the steepest ray through a candle that no later candle crosses. A candle
   crosses when it goes through the line by more than ``pierce`` typical candle ranges: a poke
   too small to see on the chart does not count (without it, the candle an hour after a top,
   3 cents lower, stopped the ray from the top and made a one-hour "line"). Of all the rays from A
   that no candle crosses, this one also runs closest to every candle, so it has the most touches.
4. **Touches.** A candle touches a line when its high (low) comes within ``touch`` typical candle
   ranges of it ("close enough"); the typical range is the median candle range on the chart. A
   touch counts once ``confirm`` candles have followed it. A ray stopped by a newer candle is not
   a line yet (price is still at that spot): it is kept as the chain's pending ray.
5. **Lines connect.** Point B is the line's most recent touch, and the next line starts there.
   Every later candle stays further from the line than B, so the next line is steeper. The chain
   ends when no confirmed touch is left.
6. **Top-down.** Monthly, weekly, daily, 4-hour, 1-hour, then the trading timeframe. Every chart
   after the monthly shows its last ``screen`` candles, and the lines that reach onto that screen
   are drawn again on its more precise candles. The walk restarts from the last point A or B left
   of the screen, moved onto this chart's candle that holds its price. The line that ends at that
   point takes its most recent touch on this chart. The walk then finds the adjusted lines and the
   steeper ones only this chart can see. A line that ends left of the screen keeps its
   higher-timeframe drawing, unless a candle on the screen crosses it: then it is drawn again from
   its own point A.
7. **Anchors on the line** (optional, ``on_line``; the user's reading of her rule, 2026-10-27).
   Point B is the angle candle, so both anchors are candle extremes on the line and every line
   starts exactly on the one before it. Without it, B is the latest touch within ``touch``.
8. **Anchor gap** (optional, ``gap``; the user's, not hers). A line drawn on the trading chart
   counts only when its points A and B lie at least ``gap`` of its candles apart. A shorter one
   is left as the pending ray, and the chain ends before it.

**Time** runs on one trading-session clock on every chart: session n of the history starts at n,
and a 15-minute candle opening at 10:00 sits at n + 30/390. Nights, weekends and holidays take no
room, as on a TradingView chart, and a line has one slope on every timeframe. A monthly or weekly
candle sits at its first session, because a chart places a candle at its opening time; the day of
its high only shows on a lower chart, which is why the lower charts adjust the lines. **Price** is
plain (TradingView's default scale) or logarithmic (``log=True``: lines straight on a log chart).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The charts of the top-down analysis, in order, with the names the notebook shows.
TIMEFRAMES = {"1M": "monthly", "1W": "weekly", "1D": "daily", "4H": "4-hour", "1H": "1-hour",
              "15m": "15-minute", "5m": "5-minute"}
#: Minutes per candle of the charts built from 1-minute candles.
MINUTES = {"4H": 240, "1H": 60, "15m": 15, "5m": 5}
#: The candle side each kind of line runs along: downward lines on the highs, upward lines under
#: the lows.
SIDES = {"down": "high", "up": "low"}
NY = "America/New_York"
OPEN, SESSION = 570, 390         # the regular session opens 09:30 New York and lasts 390 minutes
#: Sessions that close at 13:00. The database labels their later minutes "regular" (from the clock
#: alone), but those are after-hours trades (classic_signals/data.py has the same list, from 2025).
HALF_DAYS = frozenset(["2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28",
                       "2025-12-24", "2026-11-27", "2026-12-24"])
#: Columns of a line table (:func:`top_down`); ``*_y`` are in the scale the line is straight in.
COLUMNS = ["side", "tf", "origin", "confirmed", "a_time", "a_x", "a_y", "a_price", "b_time",
           "b_x", "b_y", "b_price", "stop_time", "stop_x", "stop_y", "stop_price", "slope", "tol",
           "pierce"]
_OHLC = {"open": ("open", "first"), "high": ("high", "max"), "low": ("low", "min"),
         "close": ("close", "last")}


def load(engine, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Every stored candle of ``symbol``: daily, and 1-minute of the regular session.

    Args:
        engine: Database engine (``trading_simple.db.make_engine``).
        symbol: Ticker, e.g. ``"NOW"``.

    Returns:
        ``(daily, minutes)``. ``daily`` is indexed by session date; a session with minutes but no
        daily candle yet (the daily download can lag a day) gets one built from its minutes.
        ``minutes`` is indexed by New York opening time, without a half-day's after-hours trades,
        or is None when none are stored.

    Raises:
        ValueError: When the database holds no daily candles of ``symbol``.
    """
    from trading_simple.data import load_panel

    def stored(bar: str) -> pd.DataFrame:
        p = load_panel(engine, [symbol], bar, pd.Timestamp("1970-01-01"))
        return pd.DataFrame({"open": p.open_[symbol], "high": p.high[symbol],
                             "low": p.low[symbol], "close": p.close[symbol]}).dropna()

    daily = stored("1d")
    daily.index = daily.index.tz_convert("UTC").tz_localize(None).normalize()
    try:
        minutes = stored("1m")
    except ValueError:                       # no 1-minute candles stored
        return daily, None
    t = minutes.index.tz_convert(NY)
    minutes.index = t
    minutes = minutes[~(np.isin(t.strftime("%Y-%m-%d"), list(HALF_DAYS)) & (t.hour >= 13))]
    days = minutes.groupby(minutes.index.tz_localize(None).normalize()).agg(**_OHLC)
    daily = pd.concat([daily, days[~days.index.isin(daily.index)]]).sort_index()
    return daily, minutes


def clock(daily: pd.DataFrame) -> pd.Series:
    """The trading-session clock: every session date of ``daily`` -> its number (0, 1, 2, ...)."""
    return pd.Series(np.arange(len(daily), dtype=float), index=daily.index)


def candles(daily: pd.DataFrame, minutes: pd.DataFrame | None, tf: str,
            sessions: pd.Series) -> pd.DataFrame:
    """The candles of one chart, placed on the session clock.

    Args:
        daily: Daily candles indexed by session date.
        minutes: Regular-session 1-minute candles indexed by New York time (only the intraday
            charts read them).
        tf: A key of :data:`TIMEFRAMES`.
        sessions: The clock (:func:`clock`), covering every session of both inputs.

    Returns:
        One row per candle in time order, indexed by its opening time (a monthly or weekly candle
        by its first session), with ``open/high/low/close``, ``x`` (where it starts on the clock)
        and ``x_end`` (where the next candle can start).
    """
    if tf in MINUTES:
        size = MINUTES[tf]
        t = minutes.index.tz_localize(None)
        start = (np.asarray(t.hour * 60 + t.minute) - OPEN) // size * size
        opening = t.normalize() + pd.to_timedelta(OPEN + start, unit="min")
        out = minutes.groupby(opening).agg(**_OHLC)
        day = sessions.reindex(out.index.normalize()).to_numpy()
        out["x"] = day + (out.index.hour * 60 + out.index.minute - OPEN) / SESSION
        out["x_end"] = np.minimum(out["x"] + size / SESSION, day + 1)
    else:
        group = daily.index.to_period({"1M": "M", "1W": "W-SUN", "1D": "D"}[tf])
        out = daily.groupby(group).agg(**_OHLC)
        days = daily.index.to_series().groupby(group)
        out.index = pd.DatetimeIndex(days.first().to_numpy())
        out["x"] = sessions.reindex(out.index).to_numpy()
        out["x_end"] = sessions.reindex(days.last().to_numpy()).to_numpy() + 1
    out.index.name = "time"
    return out


def top_down(frames: dict[str, pd.DataFrame], side: str, *, log: bool = False,
             touch: float = 0.25, pierce: float = 0.1, confirm: int = 2,
             screen: int = 150, gap: int = 0, on_line: bool = False) -> dict[str, pd.DataFrame]:
    """Her top-down analysis for one side: the chain of lines after each chart.

    Args:
        frames: Candles of each chart (:func:`candles`) in top-down order, all cut at the same
            moment. The first chart sees all history. A later chart may start later (the 1-minute
            history is shorter): the charts before it fill the time before its first candle.
        side: ``"down"`` (lines on the highs) or ``"up"`` (lines under the lows).
        log: Lines straight on a logarithmic price scale (else on a plain one).
        touch: How close counts as a touch, in typical candle ranges of the chart.
        pierce: How far a candle may poke through a line unseen, in typical candle ranges.
        confirm: Candles that must follow a touch before it counts.
        screen: Candles on screen on every chart after the first.
        gap: The anchor gap (user, 2026-10-27): a line drawn on the LAST chart (the trading
            chart) needs its points A and B at least this many of that chart's candles apart,
            on the session clock (16 on the 15-minute chart = 4 hours). A shorter one is not a
            line yet: it becomes the pending ray, and the chain ends before it, since the next
            line would start at its point B. 0: no gap.
        on_line: Point B is the angle candle, the one that stopped the ray (user, 2026-10-27:
            the anchors are candle extremes ON the line and every line starts on the one before
            it). Off (the first build): point B is the latest candle within ``touch`` of the
            line, which may lie up to ``touch`` off it. When a lower chart takes over a line
            that ends left of its screen, ``on_line`` keeps that line's point B on the lower
            chart's candle that holds its price, instead of moving it to the latest touch.

    Returns:
        ``{chart: lines}``: the lines (:data:`COLUMNS`) as they stand after each chart, in chain
        order. ``tf`` is the chart a line was last drawn on, ``origin`` the chart it was first
        found on (a line adjusted on a lower chart that still starts in the same candle keeps
        it), ``a_*`` its point A, ``b_*`` its point B (the next line's point A: the most recent
        touch, or with ``on_line`` the angle candle), ``stop_*`` the candle that stopped the
        ray (the angle candle), ``slope`` its change per session, ``tol`` and ``pierce`` its
        touch and poke distances (all three in log price when ``log``). A last row with
        ``confirmed`` False is the pending ray: not a line yet (its angle candle is still too
        new, or its anchors are closer than ``gap``).
    """
    lines: list[dict] = []
    start = known = pending = None
    out = {}
    for k, (tf, own) in enumerate(frames.items()):
        if own.empty:                        # no candles of this chart yet: nothing changes
            out[tf] = _table(lines, pending, side, log)
            continue
        bars = pd.DataFrame({"x": own["x"], "x_end": own["x_end"],
                             "y": _y(own[SIDES[side]], log), "tf": tf})
        if known is not None:                # the time before this chart's first candle
            bars = pd.concat([known[known["x_end"] <= bars["x"].iloc[0]], bars])
        known = bars
        first = 0 if k == 0 else max(len(own) - screen, 0)
        x0 = own["x"].iloc[first]
        typical = _typical(own.iloc[first:], log)
        if start is None:
            y = bars["y"].to_numpy()
            start = _point(bars, int(np.argmax(y) if side == "down" else np.argmin(y)))
        anchors = [start] + [line["B"] for line in lines]
        s = max((i for i, a in enumerate(anchors) if a["x"] < x0), default=0)
        shown = bars[bars["x"] >= x0]
        s = next((i for i in range(s) if _crossed(lines[i], shown, side, pierce * typical)), s)
        a = _snap(anchors[s], bars)
        if s == 0:
            start = a
        else:
            if not on_line:
                a = _last_touch(lines[s - 1], bars, a, side, confirm)
            lines[s - 1] = {**lines[s - 1], "B": a}
        apart = (gap * MINUTES.get(tf, SESSION) / SESSION     # in sessions, on the last chart
                 if k == len(frames) - 1 else 0.0)
        walked, pending = _walk(bars, a, side, touch * typical, pierce * typical, confirm, tf,
                                apart, on_line)
        for new, old in zip(walked, lines[s:], strict=False):   # the shorter decides
            # her "slight adjustment": a line redrawn here that still starts in the candle its
            # older drawing started in is that line, so it keeps the chart it was first found on
            if not old["A"]["x"] <= new["A"]["x"] < old["A"]["x_end"]:
                break
            new["origin"] = old.get("origin", old["tf"])
        lines = lines[:s] + walked
        out[tf] = _table(lines, pending, side, log)
    return out


def session_edges(index: pd.DatetimeIndex, tf: str, first: int, last: int) -> np.ndarray:
    """Whether each candle of an intraday chart is one of the first ``first`` or the last
    ``last`` candles of its session.

    Args:
        index: The candles' opening times, New York, without a time zone (:func:`candles`).
        tf: The chart, a key of :data:`MINUTES`.
        first: Candles from the session's open.
        last: Candles to the session's close (13:00 on a :data:`HALF_DAYS` session).

    Returns:
        A boolean array, one value per candle.
    """
    size = MINUTES[tf]
    slot = ((index.hour * 60 + index.minute - OPEN) // size).to_numpy()
    half = np.isin(index.normalize().strftime("%Y-%m-%d"), list(HALF_DAYS))
    count = np.where(half, -(-(SESSION - 180) // size), -(-SESSION // size))
    return (slot < first) | (slot >= count - last)


def cascade(frames: dict[str, pd.DataFrame], side: str, *, window: int = 150,
            confirm: int = 2, minimum: dict[str, int] | None = None,
            trim: tuple[int, int] = (0, 0), band: float = 0.0, noise_window: int = 150,
            log: bool = False) -> pd.DataFrame:
    """One line per chart, top-down (the user's method, 2026-10-27).

    The first chart draws one line, from the extreme of its last ``window`` candles: the lowest
    low for an upward line, the highest high for a downward one. Every chart below draws one
    line too, starting at the point B of the line above it, moved onto this chart's candle that
    holds its price. Each line is the steepest ray from its point A that no candle of its chart
    crosses (no poke allowance), and its point B is the candle that stops it (its angle candle).
    So both anchors are candle extremes on the line, and each line starts on the one above.

    Args:
        frames: Candles of each chart (:func:`candles`) in top-down order, cut at the same
            moment. A chart that starts later than the one above (the 1-minute history is
            shorter) leans on the candles above for the time before its first candle.
        side: ``"down"`` (lines on the highs) or ``"up"`` (lines under the lows).
        window: Candles of the first chart in view: its line starts at their extreme.
        confirm: Candles that must follow an angle candle before the line counts.
        minimum: The shortest line on each chart (user, 2026-10-27): ``{chart: candles}`` that
            must lie between a line's points A and B on that chart, e.g. ``{"1D": 5, "4H": 4,
            "1H": 4, "15m": 16}`` (16 on the 15-minute chart = 4 hours). A shorter line is not
            a line yet: the chain stops there. Skipping it and drawing on from its point B was
            tried: on NOW it found a line on few more charts (22 against 20 fifteen-minute
            lines in 160 cases) and the line no longer starts on the one above.
        trim: ``(first, last)`` (user, 2026-10-27): the LAST chart's line leaves out the first
            ``first`` and the last ``last`` candles of every session. They can neither stop it
            nor be its point B, so an opening or closing spike may poke through it. Its point A
            still comes from the line above, even inside a left-out candle: starting on the
            nearest candle left in instead moved NOW's 2026-10-05 downward line 2.7 below the
            1-hour line's B.
        band: The line band (user, 2026-10-27): a line is a band of half-width ``band`` x the
            chart's :func:`noise` (its average true range over ``noise_window`` candles, so the
            band follows the stock's noise candle by candle). A candle may reach into or
            through the line by up to the band without stopping it, and point B is the latest
            candle inside the band, which the next chart's line starts from. 0: the exact line,
            B the angle candle.
        noise_window: Candles in the noise's moving average.
        log: Lines straight on a logarithmic price scale.

    Returns:
        One row per chart that drew a line (:data:`COLUMNS`), top-down; ``tf`` and ``origin``
        are that chart, ``tol`` the band at its angle candle. A line whose angle candle is still
        too new, or one shorter than its chart's minimum, is the pending ray (``confirmed``
        False, the last row): not a line yet, and the charts below it draw nothing.
    """
    lines, known, a, pending = [], None, None, None
    for k, (tf, own) in enumerate(frames.items()):
        if own.empty:
            break
        bars = pd.DataFrame({"x": own["x"], "x_end": own["x_end"],
                             "y": _y(own[SIDES[side]], log), "tf": tf,
                             "w": band * noise(own, noise_window, log) if band else 0.0})
        out = (session_edges(own.index, tf, *trim)        # left out of this chart's line
               if k == len(frames) - 1 and tf in MINUTES and any(trim)
               else np.zeros(len(own), bool))
        if known is not None:                # the time before this chart's first candle
            before = known[known["x_end"] <= bars["x"].iloc[0]]
            bars, out = pd.concat([before, bars]), np.r_[np.zeros(len(before), bool), out]
        known = bars
        if a is None:
            view = bars.iloc[-window:]
            y = view["y"].to_numpy()
            a = _point(view, int(np.argmax(y) if side == "down" else np.argmin(y)))
        else:
            a = _snap(a, bars)
        apart = (minimum or {}).get(tf, 0) * MINUTES.get(tf, SESSION) / SESSION   # sessions
        w = bars["w"].to_numpy()[~out]         # the line band at each candle (0: exact)
        walked, pending = _walk(bars[~out], a, side, w, w, confirm, tf, apart,
                                on_line=not band, limit=1)
        if not walked:
            break
        lines.append(walked[0])
        a = walked[0]["B"]
    return _table(lines, pending, side, log)


def price_at(lines: pd.DataFrame, x, log: bool) -> np.ndarray:
    """The price of every line at clock positions ``x``: one row per line, one column per x."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = lines["a_y"].to_numpy(float)[:, None] + lines["slope"].to_numpy(float)[:, None] * (
        x[None, :] - lines["a_x"].to_numpy(float)[:, None])
    return np.exp(y) if log else y


def touches(lines: pd.DataFrame, bars: pd.DataFrame, log: bool) -> pd.DataFrame:
    """The candles of a chart that touch each line: right of its point A, within its touch
    distance and not through it by more than its poke allowance.

    Args:
        lines: One side's lines (:func:`top_down`).
        bars: Candles of any chart (:func:`candles`).
        log: The scale the lines were drawn in.

    Returns:
        ``line`` (row number in ``lines``), ``time``, ``x`` and ``price`` (the candle's high or
        low) of every touch.
    """
    if lines.empty or bars.empty:
        return pd.DataFrame(columns=["line", "time", "x", "price"])
    x, gap = _gaps(lines, bars, log)
    line, j = np.nonzero((x[None, :] > lines["a_x"].to_numpy(float)[:, None])
                         & (gap <= lines["tol"].to_numpy(float)[:, None])
                         & (gap >= -lines["pierce"].to_numpy(float)[:, None]))
    return pd.DataFrame({"line": line, "time": bars.index[j], "x": x[j],
                         "price": bars[SIDES[lines["side"].iloc[0]]].to_numpy()[j]})


def crossings(lines: pd.DataFrame, bars: pd.DataFrame, log: bool) -> int:
    """How many (line, candle) pairs cross: a candle right of a line's point A that goes through
    the line by more than the line's poke allowance."""
    if lines.empty or bars.empty:
        return 0
    x, gap = _gaps(lines, bars, log)
    return int(((x[None, :] > lines["a_x"].to_numpy(float)[:, None])
                & (gap < -lines["pierce"].to_numpy(float)[:, None] - 1e-9)).sum())


def _gaps(lines: pd.DataFrame, bars: pd.DataFrame, log: bool) -> tuple[np.ndarray, np.ndarray]:
    """Clock positions of the candles, and how far each candle stays inside each line (negative:
    it crosses), one row per line, in the lines' scale."""
    side = lines["side"].iloc[0]
    x = bars["x"].to_numpy(float)
    y = _y(bars[SIDES[side]], log).to_numpy()
    level = lines["a_y"].to_numpy(float)[:, None] + lines["slope"].to_numpy(float)[:, None] * (
        x[None, :] - lines["a_x"].to_numpy(float)[:, None])
    return x, (level - y[None, :]) if side == "down" else (y[None, :] - level)


def _walk(bars: pd.DataFrame, a: dict, side: str, tol, pierce, confirm: int,
          tf: str, gap: float = 0.0, on_line: bool = False,
          limit: int | None = None) -> tuple[list[dict], dict | None]:
    """The chain of lines from point ``a`` over the candles right of it (rules 3 to 5); with
    ``on_line``, point B is the candle that stopped the ray, else the latest touch; at most
    ``limit`` lines. ``tol`` and ``pierce`` are one distance, or one per candle of ``bars``
    (the noise band, which follows the stock's noise).

    Returns:
        The lines, and the pending ray (stopped by a candle still too new, or with its points A
        and B less than ``gap`` sessions apart) or None.
    """
    x, y = bars["x"].to_numpy(), bars["y"].to_numpy()
    tol, pierce = (np.broadcast_to(np.asarray(v, dtype=float), x.shape) for v in (tol, pierce))
    last = len(x) - 1 - confirm               # the newest candle a touch may sit on
    lines = []
    while True:
        right = np.flatnonzero(x > a["x"])
        if not right.size:
            return lines, None
        dx = x[right] - a["x"]
        slope = (y[right] - a["y"]) / dx
        if side == "down":                    # no candle may poke through by more than pierce
            best = slope[slope >= ((y[right] - pierce[right] - a["y"]) / dx).max()].min()
        else:
            best = slope[slope <= ((y[right] + pierce[right] - a["y"]) / dx).min()].max()
        stop = right[np.flatnonzero(slope == best)[-1]]
        line = {"tf": tf, "origin": tf, "A": a, "slope": float(best), "tol": float(tol[stop]),
                "pierce": float(pierce[stop]), "stop": _point(bars, stop)}
        if stop > last:                       # the candle that stops the ray is still too new
            return lines, {**line, "B": line["stop"]}
        near = right[(_gap(line, x[right], y[right], side) <= tol[right]) & (right <= last)]
        line["B"] = line["stop"] if on_line else _point(bars, near[-1])
        if line["B"]["x"] - a["x"] < gap - 1e-9:   # anchors too close together: not a line yet
            return lines, line
        a = line["B"]
        lines.append(line)
        if limit is not None and len(lines) >= limit:
            return lines, None


def _last_touch(line: dict, bars: pd.DataFrame, a: dict, side: str, confirm: int) -> dict:
    """The most recent confirmed touch of ``line`` on these candles, at or after point ``a``."""
    x, y = bars["x"].to_numpy(), bars["y"].to_numpy()
    near = np.flatnonzero((x >= a["x"]) & (_gap(line, x, y, side) <= line["tol"])
                          & (np.arange(len(x)) <= len(x) - 1 - confirm))
    return _point(bars, near[-1]) if near.size and x[near[-1]] > a["x"] else a


def _gap(line: dict, x: np.ndarray, y: np.ndarray, side: str) -> np.ndarray:
    """How far each candle stays inside ``line`` (negative: it crosses it)."""
    level = line["A"]["y"] + line["slope"] * (x - line["A"]["x"])
    return (level - y) if side == "down" else (y - level)


def _crossed(line: dict, bars: pd.DataFrame, side: str, pierce: float) -> bool:
    """Whether a candle right of the line's point A goes through it by more than ``pierce``."""
    x, y = bars["x"].to_numpy(), bars["y"].to_numpy()
    right = x > line["A"]["x"]
    return bool((_gap(line, x[right], y[right], side) < -pierce - 1e-9).any())


def _snap(p: dict, bars: pd.DataFrame) -> dict:
    """Point ``p`` moved onto the candle of ``bars`` that holds its price, among the candles that
    overlap the candle ``p`` sits on (the same point when none does)."""
    x, x_end, y = bars["x"].to_numpy(), bars["x_end"].to_numpy(), bars["y"].to_numpy()
    over = np.flatnonzero((x < p["x_end"]) & (x_end > p["x"]))
    return _point(bars, over[np.argmin(np.abs(y[over] - p["y"]))]) if over.size else p


def _point(bars: pd.DataFrame, j: int) -> dict:
    """Candle ``j`` of ``bars`` as a point: its time, clock span and price (line scale)."""
    return {"time": bars.index[j], "x": float(bars["x"].iloc[j]),
            "x_end": float(bars["x_end"].iloc[j]), "y": float(bars["y"].iloc[j]),
            "tf": bars["tf"].iloc[j]}


def _y(price: pd.Series, log: bool) -> pd.Series:
    """Prices in the scale the lines are straight in."""
    return np.log(price) if log else price.astype(float)


def noise(c: pd.DataFrame, window: int, log: bool = False) -> np.ndarray:
    """The stock's noise on one chart, candle by candle: the average true range of the last
    ``window`` candles up to and including each one (fewer at the start), in the lines' scale.
    Each value uses only its own candle and the ones before it.

    Args:
        c: Candles of one chart (:func:`candles`), in time order.
        window: Candles in the moving average.
        log: In log price (else in price).

    Returns:
        One value per candle.
    """
    high, low, prev = _y(c["high"], log), _y(c["low"], log), _y(c["close"], log).shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().to_numpy()


def _typical(c: pd.DataFrame, log: bool) -> float:
    """The median true range of the candles ``c``, in the lines' scale."""
    high, low, prev = _y(c["high"], log), _y(c["low"], log), _y(c["close"], log).shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return float(tr.median())


def _table(lines: list[dict], pending: dict | None, side: str, log: bool) -> pd.DataFrame:
    """The lines, then the pending ray if any, as a table (:data:`COLUMNS`)."""
    rows = []
    for line in lines + ([pending] if pending else []):
        row = {"side": side, "tf": line["tf"], "origin": line.get("origin", line["tf"]),
               "confirmed": line is not pending,
               "slope": line["slope"], "tol": line["tol"], "pierce": line["pierce"]}
        for key, p in (("a", line["A"]), ("b", line["B"]), ("stop", line["stop"])):
            row |= {f"{key}_time": p["time"], f"{key}_x": p["x"], f"{key}_y": p["y"],
                    f"{key}_price": float(np.exp(p["y"])) if log else p["y"]}
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
