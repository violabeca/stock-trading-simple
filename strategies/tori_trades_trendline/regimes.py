"""Where a fixed line setting of Tori Trades' bounce made money on one stock: its profitable
time ranges, long and short.

The user (2026-10-23): step back and ask whether some metric predicts, better than luck, that a
stock is moving into a stretch where the bounce pays. Step one is this module: find those
stretches for one fixed setting. They are labelled in hindsight, so they are the TARGET of
the question. A metric meant to predict them must be read before a range starts.

- **One side at a time.** Each side is replayed alone (long only, short only) over the stock's
  whole 1-minute history, so a long stretch is never hidden by a short that held the wallet.
- **Ranges.** A side's record is its net return (after costs) compounded trade after trade. A
  zigzag cuts it into legs: a rising leg ends once the record has fallen ``threshold`` below its
  high, a falling leg once it has risen ``threshold`` above its low. A **profitable range** is a
  rising leg of at least ``min_trades`` trades. It runs from its first trade's entry to its last
  trade's exit.
- **Luck.** Any record of wins and losses has rising legs, so a range is no evidence by itself.
  :func:`luck` asks whether this record's wins cluster more than chance would make them:
  it keeps every trade's size and time and deals the wins and losses out again at random. That
  means keeping each trade's absolute return where it is and shuffling the signs, the same
  number of wins every time. A day's or a week's volatility stays where it was. Only which
  trades won is left to chance. If a trade's result does not follow the ones before it more
  than in those draws, the record shows no persistence. A metric of the market could still
  sort trades, but the record itself would not hint that one exists.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "var" / "tori_trades_trendline" / "regimes"

_spec = importlib.util.spec_from_file_location("tori_bounce_for_regimes",
                                               HERE / "bounce.py")
bounce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bounce)
td = bounce.td

#: Knobs fixed for every preset, as in ``tune.py``: the 15-minute chart, a plain price scale,
#: 150 candles on screen, the session refresh, her close-break rule.
FIXED = {"tf": "15m", "log": False, "screen": 150, "refresh": True, "give": 0.0}

#: Settings worth a look, from the tuning runs (README), in ``tune.py``'s knobs.
PRESETS = {
    "#417 without its trend filter": {
        "knobs": {"tol": 1.5, "confirm": 5, "band": 2.126, "room": 0.569, "target": 1.0,
                  "reward": 4.518, "skip_open": 0, "trend": 0, "min_span": 4.79,
                  "entry": "limit"},
        "why": "tune.py v2, TPE on the lower bound, trial 417: the one setting that held from "
               "Jan-Jul to Aug-Sep 2026 on a real sample (nine other stocks, t 2.3 then 2.1 "
               "against random holds, +10.6 then +9.0 bp a trade), and lost in 2025 (-7.7 bp). "
               "Its 400-session trend filter is off here: a trend filter is itself a regime "
               "metric, one of the candidates this study should test, not build into its "
               "labels (on NOW it leaves the longs no trade after November 2025). Without "
               "a trend filter the random trials did no worse (best median t of every trend "
               "value, in both windows)."},
    "#417 as tuned": {
        "knobs": {"tol": 1.5, "confirm": 5, "band": 2.126, "room": 0.569, "target": 1.0,
                  "reward": 4.518, "skip_open": 0, "trend": 400, "min_span": 4.79,
                  "entry": "limit"},
        "why": "tune.py v2, TPE on the lower bound, trial 417, exactly as tuned, with the "
               "400-session trend filter."},
    "v1 #139 (her lines beat moved lines)": {
        "knobs": {"tol": 1.0, "confirm": 3, "band": 0.341, "room": 0.258, "target": 1.5,
                  "reward": 2.888, "skip_open": 2, "trend": 0, "min_span": 7.81,
                  "entry": "next_open"},
        "why": "tune.py v1, NSGA-II trial 139: her lines beat lines moved one typical candle in "
               "both windows and both directions, but the trades made about 0 after costs "
               "(-4.4 bp Jan-Jul, +1.6 bp Aug-Sep 2026)."},
    "Her defaults": {
        "knobs": {"tol": 0.25, "confirm": 2, "band": 0.25, "room": 0.5, "target": 0.0,
                  "reward": 0.0, "skip_open": 0, "trend": 0, "min_span": 0.0,
                  "entry": "next_open"},
        "why": "Her rules as the video gives them: no profit target, the stop half a typical "
               "candle beyond the line. They lost in every study."},
}

#: What each knob means, for the settings table.
MEANING = {
    "tol": "lines' close enough: a candle within this many typical candles touches a line "
           "(a poke through it of 0.4 x this is not seen)",
    "confirm": "candles that must follow a touch or a swing before it counts",
    "band": "entry gap: a low (high) within this many typical candles of a line is a signal; "
            "with the limit entry, the order rests this far beyond the line",
    "room": "the stop sits this many typical candles beyond the line (or the last swing)",
    "target": "take the profit at this many R (R = the entry's distance to the first stop; "
              "0 = no target, her rule)",
    "reward": "skip a signal when a line of the other kind lies less than this many R ahead",
    "skip_open": "no signal on the session's first this many candles",
    "trend": "long only above, short only below the average of this many daily closes (0: off)",
    "min_span": "trade only lines whose points A and B lie this many sessions apart",
    "entry": "limit: a resting order at the line plus the entry gap, filled in the candle that "
             "reaches it; next_open: the candle closes on the line's side, in at the next open",
    "method": "cascade: one line per chart and side, daily, 4-hour, 1-hour, 15-minute, each "
              "starting at the point B of the one above (your method, 2026-10-27); walk: her "
              "chain, as many lines as each chart allows",
    "minimum": "with the cascade: candles of each chart that must lie between a line's points A "
               "and B (the 15-minute one is the anchor gap); a shorter line is not a line yet",
    "strict": "her anchor rules as you read them (2026-10-27): no candle may cross a line (no "
              "poke allowance), and point B is the angle candle, on the line, so every line "
              "starts exactly on the one before it; off: B is the latest touch within tol and a "
              "candle may poke through by 0.4 x tol",
    "own_lines": "trade only lines drawn on the 15-minute chart: when a side has none (the line "
                 "closest to price comes from a higher chart), that side does not trade",
    "anchor_gap": "a line drawn on the 15-minute chart needs its points A and B at least this "
                  "many 15-minute bars apart (16 = 4 hours); a shorter one is not a line yet, "
                  "and the chain ends before it",
    "tf": "the trading chart",
    "log": "lines straight on a log price scale",
    "screen": "candles in view on every chart after the monthly",
    "refresh": "while waiting, draw the lines again at every session's close",
    "give": "a close this many typical candles beyond the line ends the trade",
}


def replay_kwargs(knobs: dict) -> dict:
    """A preset's knobs as ``bounce.replay`` arguments (``tol`` sets touch and pierce, as in
    ``tune.py``; ``strict`` sets no poke allowance and point B on the line)."""
    k = dict(knobs)
    tol, strict = k.pop("tol"), k.pop("strict", False)
    return {**FIXED, **k, "touch": tol, "pierce": 0.0 if strict else 0.4 * tol,
            "on_line": bool(strict)}


def replay_side(daily: pd.DataFrame, minutes: pd.DataFrame, symbol: str, side: str,
                knobs: dict, costs, *, book: bounce.Chartbook | None = None,
                cache: dict | None = None, folder: Path | None = OUT) -> dict:
    """One side's replay over the stock's whole 1-minute history, kept on disk.

    Args:
        daily: Daily candles (``topdown.load``).
        minutes: 1-minute candles (``topdown.load``).
        symbol: The stock, for the file name.
        side: ``"long"`` or ``"short"``.
        knobs: The setting, in ``tune.py``'s knobs (``tol`` and no ``sides``).
        costs: The ``costs:`` block of the settings.
        book: A shared ``bounce.Chartbook`` of the same candles.
        cache: A shared line cache (``bounce.replay``'s ``cache``).
        folder: Where results are kept, keyed by the stock, side, knobs, the candles' span and
            the source of ``bounce.py`` and ``topdown.py``; None keeps nothing.

    Returns:
        ``bounce.replay``'s result. Trading starts the session after the first 1-minute one.
    """
    start = minutes.index[0].tz_localize(None).normalize() + pd.Timedelta(days=1)
    source = b"".join((HERE / name).read_bytes() for name in ("bounce.py", "topdown.py"))
    key = hashlib.sha1(json.dumps(
        [symbol, side, sorted(knobs.items()), str(minutes.index[0]), str(minutes.index[-1]),
         len(minutes), str(daily.index[-1]), hashlib.sha1(source).hexdigest()],
        default=str).encode()).hexdigest()[:12]
    path = folder / f"{symbol}-{side}-{key}.pkl" if folder is not None else None
    if path is not None and path.exists():
        return pickle.loads(path.read_bytes())
    result = bounce.replay(daily, minutes, start=start, costs=costs, sides=side, book=book,
                           cache=cache, **replay_kwargs(knobs))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(result))
    return result


def trade_log(result: dict) -> pd.DataFrame:
    """One side's trades ranked by their return after costs, each with its setup.

    Args:
        result: ``bounce.replay``'s result for one side.

    Returns:
        The trades (``bounce.replay``'s columns: the line each bounced off is ``line drawn on``,
        ``line A``, ``line A price``, ``line B``, ``a_x``, ``a_y``, ``slope``), best first, plus
        ``rank`` (1 = the best), ``profit fifth`` (1 = the most profitable fifth of the side's
        trades, 5 = the least), ``lines drawn at`` and ``drawn because`` (the setup), and
        ``upward lines`` and ``downward lines`` (how many of each stood then; :func:`lines_at`
        lists them).
    """
    t = result["trades"]
    if t.empty:
        return t
    setups = result["setups"].set_index("setup")
    log = t.assign(**{
        "lines drawn at": setups["time"].reindex(t["setup"]).to_numpy(),
        "drawn because": setups["reason"].reindex(t["setup"]).to_numpy(),
        "upward lines": [len(result["lines"][s]["long"]) for s in t["setup"]],
        "downward lines": [len(result["lines"][s]["short"]) for s in t["setup"]]})
    log = log.sort_values("return", ascending=False, kind="stable").reset_index(drop=True)
    log.insert(0, "rank", np.arange(1, len(log) + 1))
    log.insert(1, "profit fifth", 1 + np.arange(len(log)) * 5 // len(log))
    return log


def lines_at(result: dict, trade: int) -> pd.DataFrame:
    """Every line that stood when one trade was set up, upward and downward.

    Args:
        result: ``bounce.replay``'s result for one side.
        trade: The trade's number (its ``trade`` column).

    Returns:
        The lines (``topdown.COLUMNS``: the chart each was drawn on, its points A and B, the
        candle that stopped the ray, its slope), with ``kind`` (``upward`` or ``downward``)
        and ``traded`` (True for the line the trade bounced off).
    """
    t = result["trades"].iloc[trade]
    drawn = result["lines"][int(t["setup"])]
    out = pd.concat([drawn["long"].assign(kind="upward"),
                     drawn["short"].assign(kind="downward")], ignore_index=True)
    out["traded"] = ((out["kind"] == ("upward" if t["side"] == "long" else "downward"))
                     & (out["a_time"] == t["line A"]) & (out["b_time"] == t["line B"])
                     & (out["tf"] == t["line drawn on"]))
    return out


def lines_log(result: dict) -> pd.DataFrame:
    """:func:`lines_at` for every trade of one side, in one table with a ``trade`` column."""
    if result["trades"].empty:
        return pd.DataFrame()
    return pd.concat([lines_at(result, i).assign(trade=i)
                      for i in result["trades"]["trade"]], ignore_index=True)


def pivots(curve: np.ndarray, threshold: float) -> list[int]:
    """The zigzag's turning points on a record.

    Args:
        curve: The record before the first trade (``curve[0]``) and after each trade.
        threshold: The reversal that ends a leg, in the record's units.

    Returns:
        Positions in ``curve``, from 0 to its last, each a high or a low of the zigzag.
    """
    piv, trend, hi, lo = [0], 0, 0, 0
    for i in range(1, len(curve)):
        v = curve[i]
        if trend == 0:                           # no leg yet: wait for the first swing
            hi = i if v > curve[hi] else hi
            lo = i if v < curve[lo] else lo
            if curve[hi] - curve[lo] >= threshold:
                trend = 1 if hi > lo else -1
                turn = lo if trend > 0 else hi
                if turn > 0:
                    piv.append(turn)
        elif trend > 0:                          # rising: a new high, or the fall that ends it
            if v >= curve[hi]:
                hi = i
            elif curve[hi] - v >= threshold:
                piv.append(hi)
                trend, lo = -1, i
        else:                                    # falling: a new low, or the rise that ends it
            if v <= curve[lo]:
                lo = i
            elif v - curve[lo] >= threshold:
                piv.append(lo)
                trend, hi = 1, i
    last = hi if trend > 0 else lo if trend < 0 else None
    if last is not None and last != piv[-1]:
        piv.append(last)
    if piv[-1] != len(curve) - 1:
        piv.append(len(curve) - 1)
    return piv


def ranges(trades: pd.DataFrame, threshold: float, min_trades: int,
           chart: pd.DataFrame | None = None) -> pd.DataFrame:
    """Cut one side's trades into the zigzag's legs.

    Args:
        trades: ``bounce.replay``'s trades of one side, in order.
        threshold: The reversal that ends a leg, as a return (0.02: 2%), on the compounded net
            record.
        min_trades: Trades a rising leg needs to count as a profitable range.
        chart: The trading chart's candles (``bounce.replay``'s ``chart``); with it, every leg
            also gets the stock's own move from its first entry to its last exit.

    Returns:
        One row per leg: ``leg`` (its number), ``kind`` (``profitable``, ``too few trades``
        for a rising leg below ``min_trades``, ``losing``, or ``in between`` for a first or
        last leg that moved less than ``threshold``), ``first`` and ``last`` (trade numbers),
        ``from`` (first entry), ``to`` (last exit), ``trades``, ``wins``, ``return``
        (compounded, after costs), ``average R``, ``best share`` (the best trade's part of a
        rising leg's gain), ``sessions`` spanned and, with ``chart``, ``stock move``.
    """
    columns = ["leg", "kind", "first", "last", "from", "to", "trades", "wins", "return",
               "average R", "best share", "sessions", "stock move"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    gain = np.log1p(trades["return"].to_numpy(float))
    curve = np.concatenate([[0.0], np.cumsum(gain)])
    h = float(np.log1p(threshold))
    days = None if chart is None else chart.index.normalize()
    turns, rows = pivots(curve, h), []
    for a, b in zip(turns[:-1], turns[1:], strict=True):
        part = trades.iloc[a:b]
        move = curve[b] - curve[a]
        if move >= h - 1e-12:
            kind = "profitable" if b - a >= min_trades else "too few trades"
        elif move <= -h + 1e-12:
            kind = "losing"
        else:
            kind = "in between"
        start, end = part["entry time"].iloc[0], part["exit time"].iloc[-1]
        stock, sessions = np.nan, len(pd.bdate_range(start.normalize(), end.normalize()))
        if chart is not None:
            stock = chart.loc[end, "close"] / chart.loc[start, "open"] - 1
            sessions = int(days[(days >= start.normalize()) & (days <= end)].nunique())
        rows.append({
            "leg": len(rows) + 1, "kind": kind, "first": a + 1, "last": b,
            "from": start, "to": end, "trades": b - a,
            "wins": int((part["return"] > 0).sum()), "return": float(np.expm1(move)),
            "average R": float(part["R"].mean()),
            "best share": float(gain[a:b].max() / move) if move > 0 else np.nan,
            "sessions": sessions, "stock move": stock})
    return pd.DataFrame(rows, columns=columns)


def luck(returns: np.ndarray, window: int = 5, draws: int = 2000,
         seed: int = 0) -> pd.DataFrame:
    """Do this side's wins cluster in time more than chance would make them?

    Each draw keeps every trade's size and place and deals the record's wins and losses out
    again at random (the signs shuffled, the same number of wins). The ranges' own numbers are
    no test: ten wins then ten losses spend no more trades in profitable ranges than 40% of
    the random deals do, because any record rises about half the time. Persistence is read
    directly instead.

    Args:
        returns: The side's net trade returns, in order.
        window: Trades in the recent record that the next trade is compared with.
        draws: Random draws.
        seed: The random seed.

    Returns:
        One row per statistic: the correlation of a trade's return with the next one's, the
        correlation of a trade's return with the average of the ``window`` before it, and the
        number of streaks (runs of wins or of losses; fewer streaks = more clustered). Each
        row gives the real record, the draws' median and their 5th and 95th percentiles, and
        ``chance by luck``: the share of draws at least as clustered as the real record.
    """
    columns = ["statistic", "real", "luck: median", "luck: 5%", "luck: 95%", "chance by luck"]
    gain = np.log1p(np.asarray(returns, dtype=float))
    if len(gain) < window + 3:
        return pd.DataFrame(columns=columns)
    size, signs = np.abs(gain), np.sign(gain)
    recent = np.lib.stride_tricks.sliding_window_view

    def stats(g):
        """Every statistic of one record."""
        before = recent(g[:-1], window).mean(axis=1)
        return (float(np.corrcoef(g[:-1], g[1:])[0, 1]),
                float(np.corrcoef(before, g[window:])[0, 1]),
                float(1 + (np.diff(g > 0) != 0).sum()))

    real = np.asarray(stats(gain))
    rng = np.random.default_rng(seed)
    shuffled = np.array([stats(size * rng.permutation(signs)) for _ in range(draws)])
    more = np.array([1.0, 1.0, -1.0])            # streaks: FEWER is more clustered
    return pd.DataFrame({
        "statistic": ["correlation with the next trade",
                      f"correlation with the average of the {window} trades before",
                      "streaks of wins or losses"],
        "real": real, "luck: median": np.nanmedian(shuffled, axis=0),
        "luck: 5%": np.nanpercentile(shuffled, 5, axis=0),
        "luck: 95%": np.nanpercentile(shuffled, 95, axis=0),
        "chance by luck": (more * shuffled >= more * real - 1e-12).mean(axis=0)},
        columns=columns)
