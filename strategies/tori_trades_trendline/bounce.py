"""Tori Trades' trendline bounce, traded long, short or both on one stock: a replay, candle by
candle.

The loop (user's spec, 2026-10-19; shorts added the same day). A long trades the upward lines
under the lows, a short the downward lines over the highs, each the mirror of the other:

1. **Set up.** Draw the lines with her top-down analysis (:func:`topdown.top_down`), using only
   what was known at that moment: complete candles up to the close of the current trading-chart
   candle, and on the higher charts a last candle still forming, as on a live chart.
2. **Wait** for price to come to a line. Long: a candle whose low comes within ``band`` typical
   candles of an upward line and that closes above it. Short: a candle whose high comes within
   ``band`` of a downward line and that closes below it. Enter at the next candle's open, on the
   line price meets first (the highest upward, the lowest downward line). A long and a short
   signal on the same candle is no trade. A candle that closes beyond a traded line (below an
   upward, above a downward one) breaks it: set up again. While waiting, the lines are also
   drawn again at every session's close (``refresh``; always when there is no line at all).
   Without it the lines go stale: on NOW from 2026-04-16 the upward lines drawn after the April
   trade lay near $82, price rallied to $149 without coming back, and the long-only replay
   waited five months while the rally's own, steeper lines were never drawn.
3. **Manage.** The first stop sits ``room`` typical candles beyond the line (below an upward,
   above a downward one). Each new swing (a low for a long, a high for a short: the extreme of
   ``confirm`` candles on each side) moves the stop to ``room`` beyond that swing, or beyond the
   line where the swing lies past it. The stop stays beyond the line and only moves in the
   trade's favour.
4. **Exit** when the stop is hit (at the stop, or at the open when price gaps through it), or
   when a candle closes more than ``give`` typical candles beyond the line (at the next open).
   Then set up again, and wait.

Decisions are made at a candle's close, orders fill at the next candle's open, stops fill during
the candle. One position at a time with the whole wallet in it (profits reinvested), at the repo's
costs (``backtest.leg_costs``) priced at the real share price and order size; a short is sold
for the wallet's worth and bought back. ``band``, ``room`` and ``give`` are measured in typical
candles of the trading chart, the chart she watches when she enters. ``shift`` is a placebo: it
moves every line that many typical candles toward price (negative: away). If her lines mark
something real, lines moved off them should trade worse.

Two optional filters, both off by default, came out of the 21-stock study (``study.py``,
stage D): ``trend`` takes a long only while the price is above the average of the last ``trend``
daily closes and a short only while below (her "always trading with the trend"); ``min_span``
trades only lines whose points A and B lie at least that many sessions apart (the lines of the
higher charts).

Three more, off by default (2026-10-20, for the 10-stock tuning, ``tune.py``): ``target`` takes
the profit at ``target`` R (R = the distance from the entry to the first stop; a gap past it
fills at the open), checked after the stop within a candle; ``skip_open`` ignores signals on the
session's first ``skip_open`` candles; ``reward`` skips a signal when a line of the other kind
(a downward line above a long, an upward line below a short) lies less than ``reward`` R beyond
the signal candle's close: no room to win.

``entry="limit"`` (2026-10-21) enters as she does, at the line: a resting limit order sits at
every traded line plus ``band`` (above an upward line for a long), and the first one price meets
fills inside that candle (at the open when price opened inside the band; never when it opened
beyond the line). The trend filter reads the candle's open instead of the signal close. On the
fill candle the stop may still be hit after the fill (assumed: the worst case, at the stop), and
the target is only checked from the next candle.

Two more, off by default (user, 2026-10-27). ``own_lines`` trades only the lines drawn on the
trading chart. A trade bounces off the line closest to price, the last line of the top-down
chain, unless ``min_span`` skips that one. The last line is a trading-chart line whenever the
walk drew one. So this means: when the trading chart drew no line on a side, that side does not
trade (on NOW with tune.py's trial 417 and no trend filter, 28 of 280 trades were on a 1-hour,
4-hour, daily or weekly line). ``anchor_gap`` is
``topdown.top_down``'s ``gap``: a line drawn on the trading chart needs its points A and B at
least that many of its candles apart (16 on the 15-minute chart = 4 hours). A shorter one is
not a line yet, and the chain ends before it.

``method="cascade"`` (user, 2026-10-27; the default stays ``"walk"``) draws the lines the
user's way, ``topdown.cascade``: one line per side on each of the daily, 4-hour, 1-hour and
trading charts, each starting at the point B of the one above. The daily one starts at the
extreme of the last ``screen`` daily candles. Every line is strict (no candle crosses it, B is
its angle candle), and it is not a line yet when its A and B are closer than its chart's
``minimum`` (``anchor_gap`` for the trading chart). ``touch``, ``pierce`` and ``on_line`` do not
apply to it.

Two session-edge knobs (user, 2026-10-27), both off by default, count candles of the trading
chart from each session's open and close (a half-day closes at 13:00). ``draw_trim = (first,
last)`` leaves those candles out of the cascade's trading-chart line (``topdown.cascade``'s
``trim``). ``trade_trim = (first, last)`` opens and closes no position on them: no signal, no
fill, no stop or target fill, no exit on a close beyond the line. A stop crossed during them
fills on the first candle after them (at its open when price is already beyond it); the stop
still trails on them.

``watch`` (user, 2026-10-27; cascade only, off by default). While waiting, the lines are
redrawn only when a candle breaks a line without giving a signal (and after every trade and at
every session's close), never after every candle. After such a break the cascade's new
trading-chart line is usually still pending: the breaking candle is its own angle candle, too new
to count. Without ``watch`` the side then waits for the next session's close, as it did after
all 16 breaks on NOW. With ``watch`` that pending ray is kept. It becomes a line by itself once
``confirm`` candles have followed its angle candle, and a candle reaching through it before then
changes it, so the lines are redrawn. Only a ray that is pending because its angle candle is too
new is watched; one too short for the chart's minimum never becomes a line.

``hold`` (user, 2026-10-27; next-open entries, 0 by default) confirms a bounce. After the
signal candle (it came to the line and closed on its side), the next ``hold`` candles must close
on the line's side too; the entry is at the open after the last of them. A close beyond the line
among them means the bounce failed: no trade, and since the line broke, the lines are drawn
again. While a bounce waits for its confirmation no other signal is taken and the session-close
redraw waits, so the confirmation can run on into the next session.

Noise bands (user, 2026-10-27; off by default). The stock's noise is the moving average of the
true range (``topdown.noise``: ``noise_window`` candles of each chart), recomputed at every
candle. ``line_band`` draws every line of the cascade as a band of ``line_band`` x its chart's
noise (``topdown.cascade``'s ``band``). ``price_band`` adds the price's own uncertainty: around
the traded line lies a contact zone of (``line_band`` + ``price_band``) x the trading chart's
noise, as known at each candle's open. With bands, the zone replaces the entry gap and the close
break, and pushes the stop out:
- a signal is a candle reaching into the zone that closes on the line's side of it;
- a break (and a failed bounce, and the exit on a close) is a close beyond the zone;
- a close inside the zone, beyond the line, is noise: neither a signal nor a break;
- the stop sits ``room`` typical candles beyond the zone;
- a watched ray changes (a break) when a candle reaches beyond its line band.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("tori_topdown_for_bounce",
                                               Path(__file__).with_name("topdown.py"))
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)

#: Why the lines were drawn again.
START, STOPPED, CLOSED, BROKEN, SESSION, TARGET = (
    "start", "after the stop was hit", "after a close beyond the line",
    "a line broke while waiting", "waiting: again at the session's close",
    "after the target was hit")
#: Each side: the lines it trades and its direction (+1 long, -1 short).
SIDES = {"long": ("up", 1.0), "short": ("down", -1.0)}
#: The exit of a candle that closed beyond the line, by side.
CLOSED_BEYOND = {"long": "closed below the line", "short": "closed above the line"}


def snapshot(daily: pd.DataFrame, minutes: pd.DataFrame, cutoff: pd.Timestamp,
             charts: list[str], sessions: pd.Series) -> dict[str, pd.DataFrame]:
    """The candles of every chart as they stood at ``cutoff``.

    Args:
        daily: Daily candles (``topdown.load``).
        minutes: 1-minute candles (``topdown.load``).
        cutoff: A New York time; only minutes that opened before it are known.
        charts: Keys of ``topdown.TIMEFRAMES``, top-down.
        sessions: The session clock of the full history (``topdown.clock``).

    Returns:
        ``{chart: candles}``. Earlier sessions come from the daily candles, today's from its
        minutes so far, so today's daily, weekly and monthly candles are still forming.
    """
    known = minutes[minutes.index < cutoff]
    today = cutoff.tz_localize(None).normalize()
    now = known[known.index >= today.tz_localize(td.NY)]
    past = daily[daily.index < today]
    if len(now):
        past = pd.concat([past, pd.DataFrame(
            {"open": [now["open"].iloc[0]], "high": [now["high"].max()],
             "low": [now["low"].min()], "close": [now["close"].iloc[-1]]}, index=[today])])
    return {tf: td.candles(past, known, tf, sessions) for tf in charts}


class Chartbook:
    """Every chart's candles built once from all the data, cut at any moment: equal to
    :func:`snapshot` (tested), about twenty times faster, because only each chart's last candle,
    the one still forming at that moment, is rebuilt.

    Args:
        daily: Daily candles (``topdown.load``).
        minutes: 1-minute candles (``topdown.load``).
        charts: Keys of ``topdown.TIMEFRAMES``, top-down.
        sessions: The session clock of the full history (``topdown.clock``).
    """

    _FREQ = {"1M": "M", "1W": "W-SUN"}

    def __init__(self, daily: pd.DataFrame, minutes: pd.DataFrame, charts: list[str],
                 sessions: pd.Series):
        self.daily, self.charts, self.sessions = daily, charts, sessions
        self.full = {tf: td.candles(daily, minutes, tf, sessions) for tf in charts}
        self.periods = {tf: self.full[tf].index.to_period(f) for tf, f in self._FREQ.items()
                        if tf in charts}
        self.times = minutes.index
        self.ohlc = minutes[["open", "high", "low", "close"]].to_numpy(float)

    def _candle(self, i: int, j: int) -> list[float]:
        """Open, high, low and close of minutes ``i`` to ``j - 1``."""
        m = self.ohlc[i:j]
        return [m[0, 0], m[:, 1].max(), m[:, 2].min(), m[-1, 3]]

    def at(self, cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """The candles of every chart as they stood at ``cutoff`` (see :func:`snapshot`)."""
        known = int(self.times.searchsorted(cutoff))         # minutes opened before the cutoff
        today = cutoff.tz_localize(None).normalize()
        first = int(self.times.searchsorted(today.tz_localize(td.NY)))
        past = self.daily.iloc[:int(self.daily.index.searchsorted(today))]
        now = self._candle(first, known) if known > first else None
        if now is not None:
            past = pd.concat([past, pd.DataFrame([now], columns=["open", "high", "low", "close"],
                                                 index=[today])])
        out = {}
        for tf in self.charts:
            full = self.full[tf]
            if tf in td.MINUTES:
                n = int(full.index.searchsorted(cutoff.tz_localize(None)))
                start = (int(self.times.searchsorted(full.index[n - 1].tz_localize(td.NY)))
                         if n else known)
                if start >= known:                          # no minute of it known yet
                    out[tf] = full.iloc[:max(n - 1, 0)]
                    continue
                last = full.iloc[[n - 1]].copy()
                last[["open", "high", "low", "close"]] = [self._candle(start, known)]
                out[tf] = pd.concat([full.iloc[:n - 1], last])
            elif tf == "1D":
                done = full.iloc[:int(full.index.searchsorted(today))]
                if now is None:
                    out[tf] = done
                    continue
                day = self.sessions[today]
                row = pd.DataFrame([[*now, day, day + 1]], columns=full.columns,
                                   index=pd.DatetimeIndex([today], name="time"))
                out[tf] = pd.concat([done, row])
            else:
                period = today.to_period(self._FREQ[tf])
                n = int(self.periods[tf].searchsorted(period))
                begins = period.start_time
                current = past.iloc[int(past.index.searchsorted(begins)):]
                out[tf] = pd.concat([full.iloc[:n], td.candles(current, None, tf, self.sessions)])
        return out


def replay(daily: pd.DataFrame, minutes: pd.DataFrame, *, start, costs, tf: str = "15m",
           sides: str = "long", log: bool = False, touch: float = 0.25, pierce: float = 0.1,
           confirm: int = 2, screen: int = 150, band: float = 0.25, room: float = 0.5,
           give: float = 0.0, shift: float = 0.0, refresh: bool = True, trend: int = 0,
           min_span: float = 0.0, target: float = 0.0, skip_open: int = 0,
           reward: float = 0.0, entry: str = "next_open", own_lines: bool = False,
           anchor_gap: int = 0, on_line: bool = False, method: str = "walk",
           minimum: dict[str, int] | None = None, draw_trim: tuple[int, int] = (0, 0),
           trade_trim: tuple[int, int] = (0, 0), watch: bool = False, hold: int = 0,
           line_band: float = 0.0, price_band: float = 0.0, noise_window: int = 150,
           cash: float = 10_000.0,
           cache: dict | None = None, book: Chartbook | None = None) -> dict:
    """Trade her bounce from ``start`` to the last candle.

    Args:
        daily: Daily candles (``topdown.load``).
        minutes: 1-minute candles (``topdown.load``); the replay ends at the last one.
        start: First session traded; the lines are first drawn at the close before it.
        costs: The ``costs:`` block of the settings (``backtest.leg_costs`` reads it).
        tf: The trading chart, an intraday key of ``topdown.TIMEFRAMES``.
        sides: ``"long"``, ``"short"`` or ``"both"``.
        log: Lines straight on a log price scale.
        touch: The lines' close enough (their touches), in typical candles.
        pierce: How far a candle may poke through a line unseen, in typical candles.
        confirm: Candles that must follow a touch or a swing before it counts.
        screen: Candles in view on every chart after the monthly.
        band: The entry: how close a low (high) must come to the line, in typical candles.
        room: How far beyond the line the stop sits, in typical candles.
        give: How far beyond the line a close must be to end the trade, in typical candles.
        shift: Placebo: every line moved this many typical candles toward price.
        refresh: While waiting, draw the lines again at every session's close (else only after
            a trade or a broken line, and at a session's close when there is no line).
        trend: With N > 0, a long only while the signal candle closes above the average of the
            last N daily closes before its session, a short only while it closes below.
        min_span: Trade only lines whose points A and B are at least this many sessions apart.
        target: Take the profit at this many R (0: no target, her rule).
        skip_open: Ignore signals on the session's first this many candles.
        reward: Skip a signal when a line of the other kind lies less than this many R beyond
            the signal candle's close (0: off).
        entry: ``"next_open"`` (a candle comes to the line and closes on its side: in at the
            next open) or ``"limit"`` (a resting order at the line plus ``band``, filled inside
            the candle that reaches it; at the open when price opened in the band).
        own_lines: Trade only the lines drawn on the trading chart: no trading-chart line on a
            side, no trade on that side.
        anchor_gap: A line drawn on the trading chart needs its points A and B at least this
            many of its candles apart (``topdown.top_down``'s ``gap``; 0: none).
        on_line: Point B is the angle candle, on the line, and every line starts exactly on the
            one before it (``topdown.top_down``'s ``on_line``); else B is the latest touch.
        method: ``"walk"`` (her chain, ``topdown.top_down``: as many lines as each chart
            allows) or ``"cascade"`` (one line per chart, ``topdown.cascade``).
        minimum: With the cascade, ``{chart: candles}`` a line of that chart needs between its
            points A and B (``anchor_gap``, if set, is the trading chart's).
        draw_trim: With the cascade, ``(first, last)`` candles of every session left out of
            the trading chart's line.
        trade_trim: ``(first, last)`` candles of every session on which no position is opened
            or closed (a stop crossed then fills on the first candle after them).
        watch: With the cascade, keep the trading chart's pending ray (its angle candle too
            new) as a line that may trade once ``confirm`` candles have followed its angle
            candle; a candle reaching through it before then redraws the lines.
        hold: Next-open entries: candles after the signal candle that must also close on the
            line's side before the entry (0: in at the next open, her rule).
        line_band: With the cascade, every line is a band of this many times its chart's noise.
        price_band: The price's own band, in the trading chart's noise: with ``line_band`` it
            makes the contact zone that replaces ``band`` and ``give`` (0 and 0: no bands).
        noise_window: Candles in the noise's moving average.
        cash: The wallet at the start.
        cache: A dict that keeps the lines drawn at each candle, to share between replays of
            the same data and line knobs (``tf``, ``log``, ``touch``, ``pierce``, ``confirm``,
            ``screen``, ``anchor_gap``, ``on_line``, ``method``, ``minimum``): a sweep over the
            trading knobs then draws every setup once.
        book: A :class:`Chartbook` of the same ``daily`` and ``minutes`` holding every chart up
            to ``tf``, to share between replays (it builds the charts once); without it each
            setup builds them from the minutes (:func:`snapshot`). Same results either way.

    Returns:
        ``chart`` (the trading-chart candles from the first traded one), ``trades`` (one row per
        trade; ``entry_k``/``exit_k`` count candles of the whole trading chart; ``a_x``,
        ``a_y``, ``slope`` are the action line as traded, shift included; ``band``, ``room`` and
        ``give`` in the lines' scale), ``setups`` (every time the lines were drawn, and why),
        ``lines`` (every line drawn at each setup, by its number: ``{"long": upward lines,
        "short": downward lines}``, both kinds whatever ``sides`` trades, before ``own_lines``,
        ``min_span`` and the shift),
        ``stops`` (the stop at the close of every candle in a trade), ``equity`` (strategy and
        buy-and-hold, marked at every close) and ``summary``.

    Raises:
        ValueError: On an unknown ``sides``, ``entry`` or ``method``, or when ``start`` leaves
            no candle before it or none to trade.
    """
    from trading_simple.backtest import leg_costs

    if sides not in ("long", "short", "both"):
        raise ValueError(f"sides must be long, short or both, got {sides!r}")
    if entry not in ("next_open", "limit"):
        raise ValueError(f"entry must be next_open or limit, got {entry!r}")
    if method not in ("walk", "cascade"):
        raise ValueError(f"method must be walk or cascade, got {method!r}")
    traded = ["long", "short"] if sides == "both" else [sides]
    sessions = td.clock(daily)
    order = list(td.TIMEFRAMES)
    charts = order[:order.index(tf) + 1]
    if book is not None and not set(charts) <= set(book.full):
        raise ValueError(f"the chartbook lacks {sorted(set(charts) - set(book.full))}")
    chart = book.full[tf] if book is not None else td.candles(daily, minutes, tf, sessions)
    first = int(np.searchsorted(chart.index, pd.Timestamp(start)))
    if not 0 < first < len(chart):
        raise ValueError(f"start {start} leaves no candle before it or none to trade")
    o, hi, lo, c, x = (chart[col].to_numpy(float)
                       for col in ("open", "high", "low", "close", "x"))
    extreme = {"long": lo, "short": hi}          # the side of a candle that tests the line
    favour = {"long": hi, "short": lo}           # the side that reaches a target
    slot = ((chart.index.hour * 60 + chart.index.minute - td.OPEN)
            // td.MINUTES[tf]).to_numpy()        # the candle's place in its session, 0 first
    quiet = (td.session_edges(chart.index, tf, *trade_trim) if any(trade_trim)
             else np.zeros(len(chart), bool))    # no position opened or closed on these
    banded = bool(line_band or price_band)
    known = td.noise(chart, noise_window, log) if banded else np.zeros(len(chart))
    known = np.r_[known[:1], known[:-1]]         # the noise as known at each candle's open
    zone = (line_band + price_band) * known      # the contact zone around the traded line
    wline = line_band * known                    # the line band alone
    size, last = pd.Timedelta(minutes=td.MINUTES[tf]), len(chart) - 1
    average = (daily["close"].rolling(trend).mean().shift(1)    # closes before the session
               .reindex(chart.index.normalize()).to_numpy() if trend else None)
    knobs = {"log": log, "touch": touch, "pierce": pierce, "confirm": confirm, "screen": screen,
             "gap": anchor_gap, "on_line": on_line}
    steps = {**(minimum or {}), **({tf: anchor_gap} if anchor_gap else {})}
    setups, drawn, trades, stops = [], [], [], []

    def y(v):
        """A price in the lines' scale."""
        return np.log(v) if log else v

    def price(v):
        """A value in the lines' scale as a price."""
        return float(np.exp(v)) if log else float(v)

    def set_up(k, reason):
        """The traded sides' lines as of the close of candle ``k``, the typical candle, and
        each side's lines as arrays (point A's clock position and value, slope)."""
        key = (k, tf, log, touch, pierce, confirm, screen, anchor_gap, on_line, method,
               tuple(sorted(steps.items())), tuple(draw_trim), line_band, noise_window)
        if cache is None or key not in cache:
            cutoff = (chart.index[k] + size).tz_localize(td.NY)
            frames = (book.at(cutoff) if book is not None
                      else snapshot(daily, minutes, cutoff, charts, sessions))
            kinds = {}
            for kind in ("up", "down"):
                if method == "cascade":           # one line per chart, from the daily one
                    every = td.cascade({c: frames[c] for c in charts[charts.index("1D"):]},
                                       kind, window=screen, confirm=confirm, minimum=steps,
                                       trim=tuple(draw_trim), band=line_band,
                                       noise_window=noise_window, log=log)
                else:
                    every = td.top_down(frames, kind, **knobs)[tf]
                kinds[kind] = every.reset_index(drop=True)      # a pending ray comes last
            found = kinds, td._typical(frames[tf].iloc[-screen:], log)
            if cache is not None:
                cache[key] = found
        else:
            found = cache[key]
        shown = {}                               # each kind's lines, the watched ray included
        for side, (kind, _) in SIDES.items():
            v = found[0][kind]
            ok = v["confirmed"].astype(bool).to_numpy().copy()
            ready = np.zeros(len(v))             # the first candle a line may give a signal on
            if (watch and method == "cascade" and side in traded and len(v) and not ok[-1]
                    and v["tf"].iloc[-1] == tf and v["stop_time"].iloc[-1] in chart.index):
                ray, j = v.iloc[-1], chart.index.get_loc(v["stop_time"].iloc[-1])
                apart = (ray["b_x"] - ray["a_x"]) * td.SESSION / td.MINUTES[tf]
                if j > k - confirm and apart >= steps.get(tf, 0) - 1e-9:   # only too new
                    ok[-1], ready[-1] = True, j + confirm + (entry == "limit")
            shown[side] = v[ok].assign(ready=ready[ok]).reset_index(drop=True)
        lines = {side: shown[side] for side in traded}
        if own_lines:                            # the trading chart's own lines only
            lines = {side: v[v["tf"] == tf].reset_index(drop=True) for side, v in lines.items()}
        if min_span:
            lines = {side: v[(v["b_x"] - v["a_x"]) >= min_span].reset_index(drop=True)
                     for side, v in lines.items()}
        setups.append({"setup": len(setups), "time": chart.index[k], "x": x[k],
                       "reason": reason, **{f"{side} lines": len(lines[side])
                                            for side in traded}})
        drawn.append({side: v.drop(columns="ready") for side, v in shown.items()})
        geo = {side: tuple(v[col].to_numpy(float) for col in ("a_x", "a_y", "slope", "ready"))
               for side, v in lines.items()}
        # the lines of the other kind, for ``reward``: downward lines over a long, upward under
        # a short (every confirmed one drawn, whatever ``min_span``)
        opposite = {side: tuple(shown["short" if side == "long" else "long"].query("confirmed")
                                [col].to_numpy(float) for col in ("a_x", "a_y", "slope"))
                    for side in traded}
        return lines, found[1], (geo, opposite)

    def enter(k, side, line, stop, typical, moved, fill=None, signal=None):
        """Open a position with the whole wallet during candle ``k``: at its open, or at
        ``fill`` (a limit order filled inside the candle); ``signal`` is the candle that came
        to the line (the one before ``k`` by default)."""
        paid = o[k] if fill is None else fill
        fee, slip = leg_costs(costs, price=paid, notional=wallet)
        d = SIDES[side][1]
        shares = wallet / (paid * (1 + slip) * (1 + fee)) if d > 0 else wallet / paid
        goal = (y(paid + d * target * abs(paid - price(stop))) if target else None)
        return {"side": side, "d": d, "setup": len(setups) - 1, "line": line, "moved": moved,
                "ax": float(line["a_x"]), "ay": float(line["a_y"]) + moved,
                "slope": float(line["slope"]), "target": goal, "entry": paid,
                "signal": (signal if signal is not None else k if fill is not None else k - 1),
                "fresh": fill is not None,
                "k": k, "stop": stop, "stop0": stop, "typical": typical, "cost": wallet,
                "shares": shares,
                "proceeds": 0.0 if d > 0 else shares * paid * (1 - slip) * (1 - fee)}

    def worth(k):
        """The open position marked at candle ``k``'s close."""
        if pos["d"] > 0:
            return pos["shares"] * c[k]
        return pos["cost"] + pos["proceeds"] - pos["shares"] * c[k]

    def leave(k, fill, reason):
        """Close the whole position at ``fill`` during candle ``k``; returns the new wallet."""
        fee, slip = leg_costs(costs, price=fill, notional=pos["shares"] * fill)
        if pos["d"] > 0:
            wallet = pos["shares"] * fill * (1 - slip) * (1 - fee)
        else:
            wallet = pos["cost"] + pos["proceeds"] - pos["shares"] * fill * (1 + slip) * (1 + fee)
        line, entry, stop0, d = pos["line"], pos["entry"], price(pos["stop0"]), pos["d"]
        trades.append({
            "trade": len(trades), "side": pos["side"], "setup": pos["setup"],
            "line drawn on": line["tf"], "line A": line["a_time"],
            "line A price": line["a_price"], "line B": line["b_time"],
            "signal time": chart.index[pos["signal"]], "entry time": chart.index[pos["k"]],
            "entry": entry, "initial stop": stop0, "exit time": chart.index[k], "exit": fill,
            "reason": reason, "R": d * (fill - entry) / abs(entry - stop0),
            "return": wallet / pos["cost"] - 1, "candles": k - pos["k"] + 1,
            "entry_k": pos["k"], "exit_k": k, "a_x": line["a_x"],
            "a_y": line["a_y"] + pos["moved"], "slope": line["slope"],
            "band": pos["reach"], "room": room * pos["typical"],
            "give": give * pos["typical"]})
        return wallet

    lines, typical, (geo, opposite) = set_up(first - 1, START)
    wallet, pos, enter_at, close_why, value = float(cash), None, None, None, []
    bounce = None              # a signal waiting for ``hold`` closes: (side, line, candle, left)
    for k in range(first, last + 1):
        if close_why is not None and not quiet[k]:   # closed beyond the line: out at this open
            wallet, pos, close_why = leave(k, o[k], close_why), None, None
        if enter_at is not None:
            side, i, touched = enter_at
            enter_at, line, d = None, lines[side].iloc[i], SIDES[side][1]
            moved = d * shift * typical
            ax, ay, slope, _ = geo[side]
            stop = (ay[i] + slope[i] * (x[k] - ax[i]) + moved
                    - d * (room * typical + zone[touched]))
            if d * (y(o[k]) - stop) > 0:         # an open beyond the stop is no trade
                pos, wallet = enter(k, side, line, stop, typical, moved, signal=touched), 0.0
                pos["reach"] = zone[touched] if banded else band * typical
        if pos is None and entry == "limit":
            # a resting limit order at every traded line, the entry gap beyond it: price fills
            # the first one it meets inside this candle (at the open when it opened in the band)
            fills, broke = [], False
            for side in traded:
                ax, ay, slope, ready = geo[side]
                if not len(ax):
                    continue
                d = SIDES[side][1]
                at = ay + slope * (x[k] - ax) + d * shift * typical
                early = k < ready                 # a watched ray not confirmed yet
                broke = (broke or bool((d * (y(c[k]) - at) < -zone[k]).any())
                         or bool((early & (d * (y(extreme[side][k]) - at) < -wline[k])).any()))
                if ((trend and not d * (o[k] - average[k]) > 0) or slot[k] < skip_open
                        or quiet[k]):
                    continue
                level = at + d * (zone[k] if banded else band * typical)
                live = ((d * (y(o[k]) - at) > 0) & (d * (y(extreme[side][k]) - level) <= 0)
                        & ~early)
                if not live.any():
                    continue
                i = int(np.argmax(np.where(live, d * level, -np.inf)))
                paid = level[i] if d * (y(o[k]) - level[i]) > 0 else y(o[k])
                if reward:                        # room to the nearest line of the other kind
                    oax, oay, oslope = opposite[side]
                    ahead = d * (oay + oslope * (x[k] - oax) - paid)
                    ahead = ahead[ahead > 0]
                    if ahead.size and ahead.min() < reward * (d * (paid - at[i]) + zone[k]
                                                              + room * typical):
                        continue
                fills.append((side, i, paid))
            if len(fills) == 1:                   # both sides at once: no trade
                side, i, paid = fills[0]
                line, d = lines[side].iloc[i], SIDES[side][1]
                moved = d * shift * typical
                ax, ay, slope, _ = geo[side]
                stop = (ay[i] + slope[i] * (x[k] - ax[i]) + moved
                        - d * (room * typical + zone[k]))
                if d * (paid - stop) > 0:
                    pos = enter(k, side, line, stop, typical, moved, fill=price(paid))
                    pos["reach"], wallet = (zone[k] if banded else band * typical), 0.0
            if pos is None:
                closing = (k < last
                           and chart.index[k + 1].normalize() != chart.index[k].normalize())
                if broke:
                    lines, typical, (geo, opposite) = set_up(k, BROKEN)
                elif closing and (refresh or not any(len(v) for v in lines.values())):
                    lines, typical, (geo, opposite) = set_up(k, SESSION)
        elif pos is None and bounce is not None:
            # a bounce waits for its confirmation: this candle must close on the line's side too
            side, i, touched, left = bounce
            ax, ay, slope, _ = geo[side]
            d = SIDES[side][1]
            at = ay[i] + slope[i] * (x[k] - ax[i]) + d * shift * typical
            if d * (y(c[k]) - at) < -zone[k]:     # back beyond the line: the bounce failed
                bounce = None
                lines, typical, (geo, opposite) = set_up(k, BROKEN)
            elif left > 1:
                bounce = (side, i, touched, left - 1)
            else:                                 # confirmed: in at the next open
                bounce = None
                if k < last and not quiet[k + 1]:
                    enter_at = (side, i, touched)
        elif pos is None:
            signals, broke = [], False
            for side in traded:
                ax, ay, slope, ready = geo[side]
                if not len(ax):
                    continue
                d = SIDES[side][1]
                at = ay + slope * (x[k] - ax) + d * shift * typical
                early = k < ready                 # a watched ray not confirmed yet
                if ((d * (y(c[k]) - at) < -zone[k]).any()
                        or (early & (d * (y(extreme[side][k]) - at) < -wline[k])).any()):
                    broke = True                  # a close beyond a line, or a reach through
                    continue                      # a watched ray: draw the lines again
                reach = zone[k] if banded else band * typical
                hit = ((d * (y(extreme[side][k]) - at) <= reach) & (d * (y(c[k]) - at) >= 0)
                       & ~early)                  # reached the line, closed on its side
                if trend and not d * (c[k] - average[k]) > 0:    # against the trend (or unknown)
                    continue
                if slot[k] < skip_open or quiet[k]:     # too early or late in the session
                    continue
                if hit.any():
                    i = int(np.argmax(np.where(hit, d * at, -np.inf)))
                    if reward:                    # room to the nearest line of the other kind
                        risk = d * (y(c[k]) - at[i]) + room * typical + zone[k]
                        oax, oay, oslope = opposite[side]
                        ahead = d * (oay + oslope * (x[k] - oax) - y(c[k]))
                        ahead = ahead[ahead > 0]
                        if ahead.size and ahead.min() < reward * risk:
                            continue
                    signals.append((side, i))
            closing = k < last and chart.index[k + 1].normalize() != chart.index[k].normalize()
            if broke:
                lines, typical, (geo, opposite) = set_up(k, BROKEN)
            elif len(signals) == 1 and hold:
                bounce = (*signals[0], k, hold)   # wait for ``hold`` closes on its side
            elif len(signals) == 1 and k < last and not quiet[k + 1]:
                enter_at = (*signals[0], k)
            elif closing and (refresh or not any(len(v) for v in lines.values())):
                lines, typical, (geo, opposite) = set_up(k, SESSION)
        if pos is not None:
            # a limit filled in THIS candle: its low (high) may still reach the stop after the
            # fill (assumed: the worst case), but a target is not trusted before the next candle
            d, side, fresh = pos["d"], pos["side"], pos["fresh"] and pos["k"] == k
            at = pos["ay"] + pos["slope"] * (x[k] - pos["ax"])
            free = not quiet[k]                   # an exit may happen on this candle
            if free and d * (y(extreme[side][k]) - pos["stop"]) <= 0:
                fill = price(pos["stop"])
                if not fresh:                     # a gap past the stop fills at the open
                    fill = min(o[k], fill) if d > 0 else max(o[k], fill)
                wallet, pos = leave(k, fill, "stop hit"), None
                lines, typical, (geo, opposite) = set_up(k, STOPPED)
            elif (free and pos["target"] is not None and not fresh
                  and d * (y(favour[side][k]) - pos["target"]) >= 0):
                fill = price(pos["target"])               # a gap past it fills at the open
                fill = max(o[k], fill) if d > 0 else min(o[k], fill)
                wallet, pos = leave(k, fill, "target hit"), None
                lines, typical, (geo, opposite) = set_up(k, TARGET)
            elif free and close_why is None and d * (y(c[k]) - at) < -(
                    zone[k] if banded else give * pos["typical"]):
                close_why = CLOSED_BEYOND[side]
                lines, typical, (geo, opposite) = set_up(k, CLOSED)
            else:
                j = k - confirm                  # a swing, confirmed by `confirm` candles
                ext = extreme[side]
                if j >= pos["k"] and d * ext[j] <= (d * ext[max(j - confirm, 0):k + 1]).min():
                    new = (d * min(d * y(ext[j]), d * at - zone[k])
                           - d * room * pos["typical"])
                    if d * (new - pos["stop"]) > 0:
                        pos["stop"] = new
            if pos is not None:
                stops.append({"trade": len(trades), "side": side, "time": chart.index[k],
                              "x": x[k], "stop": price(pos["stop"])})
        value.append(wallet if pos is None else worth(k))
    if pos is not None:
        wallet = leave(last, c[last], close_why or "still open at the last candle")

    fee, slip = leg_costs(costs, price=o[first], notional=cash)
    held = cash / (o[first] * (1 + slip) * (1 + fee)) * c[first:]
    fee, slip = leg_costs(costs, price=c[last], notional=held[-1])
    equity = pd.DataFrame({"x": x[first:], "strategy": value, "buy and hold": held},
                          index=chart.index[first:])
    trades = pd.DataFrame(trades)
    years = (chart.index[last] - chart.index[first]).days / 365.25
    summary = {"trades": len(trades), "years": years}
    for name, end in (("strategy", wallet), ("buy and hold", held[-1] * (1 - slip) * (1 - fee))):
        curve = equity[name]
        summary |= {f"{name} return": end / cash - 1,
                    f"{name} CAGR": (end / cash) ** (1 / years) - 1 if years > 0 else np.nan,
                    f"{name} max drawdown": float((curve / curve.cummax() - 1).min())}
    if len(trades):
        summary |= {"wins": int((trades["return"] > 0).sum()),
                    "average R": float(trades["R"].mean()),
                    "time in market": float(trades["candles"].sum() / len(equity))}
    return {"chart": chart.iloc[first:], "trades": trades, "setups": pd.DataFrame(setups),
            "lines": drawn, "stops": pd.DataFrame(stops), "equity": equity, "summary": summary}
