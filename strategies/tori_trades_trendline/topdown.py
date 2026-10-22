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
COLUMNS = ["side", "tf", "confirmed", "a_time", "a_x", "a_y", "a_price", "b_time", "b_x", "b_y",
           "b_price", "stop_time", "stop_x", "stop_y", "stop_price", "slope", "tol", "pierce"]
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
             screen: int = 150) -> dict[str, pd.DataFrame]:
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

    Returns:
        ``{chart: lines}``: the lines (:data:`COLUMNS`) as they stand after each chart, in chain
        order. ``tf`` is the chart a line was last drawn on, ``a_*`` its point A, ``b_*`` its
        point B (the most recent touch, the next line's point A), ``stop_*`` the candle that
        stopped the ray, ``slope`` its change per session, ``tol`` and ``pierce`` its touch and
        poke distances (all three in log price when ``log``). A last row with ``confirmed``
        False is the pending ray: stopped by a candle still too new, so not a line yet.
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
            a = _last_touch(lines[s - 1], bars, a, side, confirm)
            lines[s - 1] = {**lines[s - 1], "B": a}
        walked, pending = _walk(bars, a, side, touch * typical, pierce * typical, confirm, tf)
        lines = lines[:s] + walked
        out[tf] = _table(lines, pending, side, log)
    return out


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


def _walk(bars: pd.DataFrame, a: dict, side: str, tol: float, pierce: float, confirm: int,
          tf: str) -> tuple[list[dict], dict | None]:
    """The chain of lines from point ``a`` over the candles right of it (rules 3 to 5).

    Returns:
        The lines, and the pending ray (stopped by a candle still too new) or None.
    """
    x, y = bars["x"].to_numpy(), bars["y"].to_numpy()
    last = len(x) - 1 - confirm               # the newest candle a touch may sit on
    lines = []
    while True:
        right = np.flatnonzero(x > a["x"])
        if not right.size:
            return lines, None
        dx = x[right] - a["x"]
        slope = (y[right] - a["y"]) / dx
        if side == "down":                    # no candle may poke through by more than pierce
            best = slope[slope >= ((y[right] - pierce - a["y"]) / dx).max()].min()
        else:
            best = slope[slope <= ((y[right] + pierce - a["y"]) / dx).min()].max()
        stop = right[np.flatnonzero(slope == best)[-1]]
        line = {"tf": tf, "A": a, "slope": float(best), "tol": float(tol),
                "pierce": float(pierce), "stop": _point(bars, stop)}
        if stop > last:                       # the candle that stops the ray is still too new
            return lines, {**line, "B": line["stop"]}
        near = right[(_gap(line, x[right], y[right], side) <= tol) & (right <= last)]
        a = line["B"] = _point(bars, near[-1])
        lines.append(line)


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


def _typical(c: pd.DataFrame, log: bool) -> float:
    """The median true range of the candles ``c``, in the lines' scale."""
    high, low, prev = _y(c["high"], log), _y(c["low"], log), _y(c["close"], log).shift()
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return float(tr.median())


def _table(lines: list[dict], pending: dict | None, side: str, log: bool) -> pd.DataFrame:
    """The lines, then the pending ray if any, as a table (:data:`COLUMNS`)."""
    rows = []
    for line in lines + ([pending] if pending else []):
        row = {"side": side, "tf": line["tf"], "confirmed": line is not pending,
               "slope": line["slope"], "tol": line["tol"], "pierce": line["pierce"]}
        for key, p in (("a", line["A"]), ("b", line["B"]), ("stop", line["stop"])):
            row |= {f"{key}_time": p["time"], f"{key}_x": p["x"], f"{key}_y": p["y"],
                    f"{key}_price": float(np.exp(p["y"])) if log else p["y"]}
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)
