"""Stock groups for the classic_signals strategy: stocks whose price CHARACTER is alike, so that
one setting of the strategy may fit all of them.

A stock's character is a handful of numbers about what the strategy's rules react to
(:data:`FEATURES`), measured on its 1-minute candles of the regular session: how big its moves
are and how much of one a round trip's costs eat, whether a minute's move tends to continue or to
turn, whether a close outside the Bollinger Band snaps back or runs on, how much of the daily
swing happens overnight, how closely it follows the S&P 500, how jumpy and how thinly traded it
is. Industries do not enter: two stocks of different trades can behave alike.

* :func:`pool`: every stock with 1-minute candles in a window, and its first and last minute.
* :func:`session_sums`: one stock's ingredients, session by session (sums and counts). Every
  character number is a ratio of such sums, so the numbers of ANY set of sessions (the whole
  window, each half, a moving 20 sessions) come exactly from the same table.
* :func:`compute`: every stock's sums, loaded and computed in threads and kept on disk
  (``var/classic_signals/groups/``): a stock is computed again only when its candles changed.
* :func:`characters`: one row of character numbers per stock (the whole window or a half).
* :func:`scores`: each number as a score against the other stocks (the normal score of its rank:
  0 the middle stock, about +1 / -1 the 84th / 16th percentile), so every number weighs the same
  and no outlier stretches the scale.
* :func:`grouping`: the groups: a Ward tree on the scores cut into k groups, each group's
  profile in words, its members ranked by how typical they are (distance to the group's centre):
  the first ``n_tune`` to tune on, the others to check the tuned setting on.
* :func:`quality`: for every number of groups: how much of the stocks' differences the groups
  explain, how well they separate, and whether the two halves of the window give the same groups.
* :func:`persistence`: how much of each number is the stock's own (the same on both halves).
* :func:`nearest`, :func:`flat_map`: the stocks most alike to one; a 2-D map of all of them.
* :func:`save` / :func:`load` / :func:`files`: the exported groups.

Run ``uv run python strategies/classic_signals/groups.py`` to compute every missing stock's sums
from a terminal (the notebook then reads them instantly).
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """One of this strategy's helper files, loaded by path."""
    spec = importlib.util.spec_from_file_location(f"classic_signals_gr_{name}",
                                                  HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IND, PIPE, DATA, HP = (_sibling("indicators"), _sibling("pipeline"), _sibling("data"),
                       _sibling("hyperparams"))
STORE = DATA.VAR / "groups"        # git-ignored: the pool, every stock's sums, the exports
#: Every stored 1-minute candle: a start before the first one (2024-12-26) and an end after the
#: last, so a new download is read without a new window (a stock whose last minute grew is
#: measured again). Not the tuning notebook's training window: the groups also read its
#: validation months and the final check.
WINDOW = ("2024-01-01", "2099-12-31")
BLOCK = 15            # minutes per block: the variance ratios, the market link, the typical move
BACK, FOLLOW = 1, 15  # minutes after a close outside the band: back inside? how far on?
TOP = 0.01            # the biggest 1% of a session's minutes (4 of 390): the jumps
OPEN_END = 600        # 10:00 New York, in minutes after midnight: the end of the first 30 minutes
BAND = (HP.DECLARED["band_length"]["value"], HP.DECLARED["band_width"]["value"])   # 20, 2.0
MOVE = (HP.DECLARED["move_size"]["value"], HP.DECLARED["move_length"]["value"])    # 15, 14
STOP = HP.DECLARED["stop_moves"]["value"]   # typical moves to the first stop: sets the size
N_TUNE = 5            # members of a group the tuning reads; the others check its setting
KS = tuple(range(2, 16))   # the numbers of groups :func:`quality` tries

#: The character numbers: ``label`` (with its unit), ``format`` (how a value is printed),
#: ``words`` (what a low and a high value mean), ``doc`` (how it is computed).
FEATURES = {
    "move_pct": {
        "format": ".2f", "label": "typical 15-minute move, % of the price",
        "words": ("small moves", "big moves"),
        "doc": "The average true range of the session's 15-minute candles (high to low, "
               "counting a jump from the previous candle's close; a session's first candle "
               "counts only its own range), in percent of their close."},
    "cost_moves": {
        "format": ".3f", "label": "a round trip's costs, in typical 15-minute moves",
        "words": ("cheap to trade", "costly to trade"),
        "doc": "What a round trip costs a share (commission and slippage of the repo's cost "
               "model, `simulate.costs`, on the position the strategy takes: $100 at risk at a "
               "first stop 3 typical moves away, at most $20,000 worth) ÷ the typical 15-minute "
               "move in dollars. Low-priced stocks pay more: the commission and the slippage "
               "have a floor per share."},
    "follow_1m": {
        "format": "+.3f", "label": "a minute's move continued by the next (lag-1 autocorrelation)",
        "words": ("minutes bounce back", "minutes follow through"),
        "doc": "The correlation of each minute's return with the previous minute's, inside a "
               "session (pooled: Σ r·r_before ÷ Σ r²). Below 0 a minute's move tends to turn "
               "back in the next one (the bid-ask bounce, quick reversals); above 0 to go on."},
    "ratio_15m": {
        "format": ".2f", "label": "15-minute moves against 15 single minutes (variance ratio)",
        "words": ("turns within 15 minutes", "runs within 15 minutes"),
        "doc": "The squared 15-minute returns ÷ the squared 1-minute returns of the same "
               "minutes (both summed). A random walk gives 1; below 1 the minutes' moves cancel "
               "out within a quarter hour (reversion), above 1 they add up (trend)."},
    "band_back": {
        "format": ".1%",
        "label": "after a close outside the band: back inside at the next close, share",
        "words": ("often a second close outside the band", "back inside the band at once"),
        "doc": "Of the times a 1-minute close lands outside the Bollinger Band (the built-in "
               "20 candles, 2 standard deviations; the close before was not outside on that "
               "side, in the same session), the share whose very next close is back inside: "
               "the candle the band fade fires on. The others are a second close outside, what "
               "the band breakout waits for (2 closes by default)."},
    "band_follow": {
        "format": "+.3f",
        "label": "after a close outside the band: the next 15 minutes, in typical moves",
        "words": ("snaps back after the band", "runs on after the band"),
        "doc": "After the same closes outside the band: the move from that close to the close 15 "
               "minutes later, counted in the excursion's direction (up after a close above the "
               "upper band), in typical 15-minute moves as known then (the stop's yardstick). "
               "Above 0 the move runs on (the breakout's case), below 0 it snaps back (the "
               "fade's). Costs left out."},
    "night_share": {
        "format": ".1%", "label": "share of the daily swing made overnight",
        "words": ("quiet nights", "big overnight gaps"),
        "doc": "The size of the overnight gaps (a session's first open against the previous "
               "session's last close) ÷ that plus the size of the open-to-close moves, both "
               "summed over the sessions (sizes, not squares: one earnings gap alone does not "
               "decide it). A trade held overnight carries this part, and a gap past the stop "
               "fills at the open."},
    "market_corr": {
        "format": ".2f", "label": "15-minute correlation with the S&P 500 fund (SPY)",
        "words": ("goes its own way", "moves with the market"),
        "doc": "The correlation of the stock's 15-minute returns with SPY's in the same quarter "
               "hours (Σ x·y ÷ √(Σ x² · Σ y²)): how much the market filter has to say."},
    "open_share": {
        "format": ".1%", "label": "share of the session's movement in its first 30 minutes",
        "words": ("calm opens", "wild opens"),
        "doc": "The ranges of the 1-minute candles (high to low, in percent of the close) "
               "from 09:30 to 10:00 ÷ those of the whole session (30 of 390 minutes: 0.077 if "
               "every minute ranged alike)."},
    "jump_share": {
        "format": ".1%", "label": "share of the minutes' swing from each session's biggest 1%",
        "words": ("moves smoothly", "moves in jumps"),
        "doc": "In every session, the squared returns of its biggest 1% of minutes (4 of 390) "
               "÷ all its squared minute returns, summed over the sessions. High: a few jumps "
               "(news, spikes) make most of the movement, and they jump over stops."},
    "quiet_share": {
        "format": ".1%", "label": "minutes without a trade, share",
        "words": ("trades every minute", "often no trade in a minute"),
        "doc": "The share of regular-session minutes in which nothing traded (the data then "
               "repeats the last price with no volume)."},
}


def _regular(minutes: pd.DataFrame) -> pd.DataFrame:
    """The regular-session candles."""
    return minutes[minutes["regular"]] if "regular" in minutes else minutes


def _returns(candles: pd.DataFrame) -> np.ndarray:
    """Every candle's log return inside its session: from the previous candle's close, a
    session's first candle from its own open (the night left out). 0 where a price is missing."""
    c, o = candles["close"].to_numpy(float), candles["open"].to_numpy(float)
    day = DATA.session_day(candles.index)
    first = np.r_[True, day[1:] != day[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(c / np.where(first, o, np.r_[np.nan, c[:-1]]))
    return np.where(np.isfinite(r), r, 0.0)


def _blocks(minutes: pd.DataFrame, size: int) -> pd.DataFrame:
    """``size``-minute candles of the regular session, counted from 09:30: the same as
    ``data.resample`` there (a test pins it), without its clock labels of every minute, which
    take most of the time and hold up the other threads."""
    m = _regular(minutes)
    t = m.index
    minute = np.asarray(t.hour * 60 + t.minute)
    start = t.normalize() + pd.to_timedelta(
        DATA.REGULAR_OPEN + (minute - DATA.REGULAR_OPEN) // size * size, unit="min")
    g = m.groupby(start)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "volume": g["volume"].sum()})
    out.index.name = "time"
    return out


def _move(minutes: pd.DataFrame, size: int = MOVE[0], length: int = MOVE[1]) -> pd.Series:
    """The strategy's typical move (the stop's yardstick) in price, as known at every regular
    minute: ``pipeline.move`` (a test pins it), on :func:`_blocks`."""
    m = _regular(minutes)
    typical = IND.typical_move(_blocks(m, size), length, across_nights=False)
    return IND.available_at(typical, size, m.index)


def block_returns(minutes: pd.DataFrame) -> pd.Series:
    """The log return of every 15-minute block of the regular session, indexed by the block's
    opening time: the market's, for :func:`session_sums`."""
    blocks = _blocks(minutes, BLOCK)
    return pd.Series(_returns(blocks), index=blocks.index)


def band_events(minutes: pd.DataFrame, *, band_length: int = BAND[0],
                band_width: float = BAND[1], move_size: int = MOVE[0],
                move_length: int = MOVE[1]) -> pd.DataFrame:
    """Every 1-minute close outside the Bollinger Band whose previous close (in the same session)
    was not outside on that side, with what followed: back inside within :data:`BACK` minutes?
    How far did the price go in the next :data:`FOLLOW` minutes? Closes with fewer than
    :data:`FOLLOW` minutes left in their session are left out. The bands and the typical move are
    the strategy's own (``pipeline.bands`` on the regular session, :func:`_move`).

    Returns:
        ``time``, ``side`` (+1 above the upper band, -1 below the lower), ``close``, ``later``
        (the close :data:`FOLLOW` minutes later), ``back`` (closed back inside the band within
        :data:`BACK` minute(s)) and ``follow`` (``later`` - ``close`` in the excursion's direction,
        in typical moves as known at the close).
    """
    m = _regular(minutes)
    bands = PIPE.bands(m, band_length, band_width, False)
    move = _move(m, move_size, move_length).to_numpy(float)
    c, up, lo = (m["close"].to_numpy(float), bands["upper"].to_numpy(float),
                 bands["lower"].to_numpy(float))
    day = DATA.session_day(m.index)
    same = np.r_[False, day[1:] == day[:-1]]               # the previous close: same session
    with np.errstate(invalid="ignore"):
        above = (c > up) & np.r_[False, c[:-1] <= up[:-1]] & same
        below = (c < lo) & np.r_[False, c[:-1] >= lo[:-1]] & same
    t = np.flatnonzero(above | below)
    t = t[t + FOLLOW < len(c)]
    t = t[day[t + FOLLOW] == day[t]]
    side = np.where(above[t], 1, -1)
    back = np.zeros(len(t), bool)
    with np.errstate(invalid="ignore"):
        for j in range(1, BACK + 1):
            back |= np.where(side > 0, c[t + j] <= up[t + j], c[t + j] >= lo[t + j])
        follow = side * (c[t + FOLLOW] - c[t]) / move[t]
    keep = np.isfinite(follow) & (move[t] > 0)
    return pd.DataFrame({"time": m.index[t], "side": side, "close": c[t], "later": c[t + FOLLOW],
                         "back": back, "follow": follow})[keep].reset_index(drop=True)


def session_sums(minutes: pd.DataFrame, market: pd.Series | None = None, **band) -> pd.DataFrame:
    """One stock's ingredients of every character number, session by session: sums and counts,
    so the numbers of any set of sessions are ratios of their summed rows (:func:`character`).

    A session's row reads only that session and the ones before (the bands and the typical move
    warm up across nights, the overnight gap needs the previous close), never a later one.

    Args:
        minutes: The stock's 1-minute candles (``data.load_minutes``; the regular session is
            used).
        market: The market's :func:`block_returns` (None: no market link).
        band: :func:`band_events`'s settings (the built-in ones by default).

    Returns:
        One row per session (indexed by its day): ``sessions`` (1), ``minutes``, ``quiet``
        (minutes without a trade), ``close_sum`` and ``dollar_volume`` (Σ close, Σ close ×
        volume), ``s2`` / ``s11`` / ``top`` (Σ r², Σ r·r_before, Σ r² of the biggest 1% of
        minutes; r: 1-minute log returns inside the session), ``range_pct`` / ``range_open``
        (Σ of the minutes' high - low in percent of the close, all day and before 10:00),
        ``s15`` (Σ of squared 15-minute returns), ``gap_abs`` and ``oc_abs`` (the size of the
        overnight gap and of the open-to-close return, where the previous session's close is
        known, else 0), ``tr15`` / ``tr15_pct`` /
        ``blocks`` (Σ of the 15-minute true ranges, in price and in percent, and their count),
        ``sxy`` / ``sxx`` / ``syy`` (the market link's sums over the blocks both traded),
        ``events`` / ``back`` / ``follow`` (:func:`band_events`: their count, how many came back
        inside, Σ of their follow).
    """
    m = _regular(minutes)
    day = DATA.session_day(m.index)
    days, sid = np.unique(day, return_inverse=True)
    n = len(days)
    add = lambda w, at=sid: np.bincount(at, weights=w, minlength=n)  # noqa: E731
    r = _returns(m)
    first = np.r_[True, sid[1:] != sid[:-1]]
    sq = r * r
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(r)]
    top = np.array([np.sort(sq[a:b])[len(sq[a:b]) - int(np.ceil(TOP * (b - a))):].sum()
                    for a, b in zip(starts, ends, strict=True)])
    c, o = m["close"].to_numpy(float), m["open"].to_numpy(float)
    volume = np.nan_to_num(m["volume"].to_numpy(float))
    clock = np.asarray(m.index.hour * 60 + m.index.minute)
    oc = add(r)                                           # the session's open-to-close return
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.r_[np.nan, np.log(o[starts[1:]] / c[ends[:-1] - 1])]
        span = np.nan_to_num((m["high"].to_numpy(float) - m["low"].to_numpy(float)) / c * 100)
    known = np.isfinite(gap)
    blocks = _blocks(m, BLOCK)
    bid = np.searchsorted(days, DATA.session_day(blocks.index))
    br = _returns(blocks)
    tr = IND.true_range(blocks, across_nights=False).to_numpy(float)
    out = {"sessions": np.ones(n), "minutes": add(np.ones(len(r))),
           "quiet": add((volume <= 0).astype(float)), "close_sum": add(c),
           "dollar_volume": add(c * volume), "s2": add(sq),
           "s11": add(r * np.where(first, 0.0, np.r_[0.0, r[:-1]])), "top": top,
           "range_pct": add(span), "range_open": add(span * (clock < OPEN_END)),
           "s15": add(br * br, bid), "gap_abs": np.abs(np.where(known, gap, 0.0)),
           "oc_abs": np.where(known, np.abs(oc), 0.0),
           "tr15": add(tr, bid), "tr15_pct": add(tr / blocks["close"].to_numpy(float) * 100, bid),
           "blocks": add(np.ones(len(br)), bid)}
    if market is not None:
        y = market.reindex(blocks.index).to_numpy(float)
        both = np.isfinite(y)
        x, y, at = br[both], y[both], bid[both]
        out.update({"sxy": add(x * y, at), "sxx": add(x * x, at), "syy": add(y * y, at)})
    else:
        out.update({"sxy": np.zeros(n), "sxx": np.zeros(n), "syy": np.zeros(n)})
    ev = band_events(m, **band)
    eid = np.searchsorted(days, DATA.session_day(pd.DatetimeIndex(ev["time"])))
    out.update({"events": add(np.ones(len(ev)), eid),
                "back": add(ev["back"].to_numpy(float), eid),
                "follow": add(ev["follow"].to_numpy(float), eid)})
    return pd.DataFrame(out, index=pd.DatetimeIndex(days, name="day"))


def character(sums, costs: dict, *, stop_moves: float = STOP) -> pd.DataFrame:
    """The character numbers (:data:`FEATURES`) of every row of summed sessions.

    Args:
        sums: :func:`session_sums`, summed over any set of sessions: one row (a Series) or
            several (e.g. a moving sum of 20 sessions).
        costs: ``simulate.costs`` (the size and the cost model).
        stop_moves: Typical moves to the first stop (the position's size follows from it).
    """
    s = sums.to_frame().T if isinstance(sums, pd.Series) else sums
    with np.errstate(divide="ignore", invalid="ignore"):
        price = s["close_sum"] / s["minutes"]
        move = s["tr15"] / s["blocks"]
        shares = np.minimum(costs["risk"] / (stop_moves * move), costs["notional"] / price)
        per_share = 2 * (np.maximum(costs["commission"], costs["per_share"] * shares) / shares
                         + np.maximum(costs["slippage"] * price, costs["min_slip"]))
        out = pd.DataFrame({
            "move_pct": s["tr15_pct"] / s["blocks"],
            "cost_moves": per_share / move,
            "follow_1m": s["s11"] / s["s2"],
            "ratio_15m": s["s15"] / s["s2"],
            "band_back": s["back"] / s["events"],
            "band_follow": s["follow"] / s["events"],
            "night_share": s["gap_abs"] / (s["gap_abs"] + s["oc_abs"]),
            "market_corr": s["sxy"] / np.sqrt(s["sxx"] * s["syy"]),
            "open_share": s["range_open"] / s["range_pct"],
            "jump_share": s["top"] / s["s2"],
            "quiet_share": s["quiet"] / s["minutes"]}, index=s.index)
    return out.astype(float).replace([np.inf, -np.inf], np.nan)


def _part(sums: pd.DataFrame, part: str) -> pd.DataFrame:
    """The sessions of one part of a stock's window: ``whole``, ``first half`` or ``second
    half`` of ITS sessions (a stock that starts late has later halves)."""
    k = len(sums) // 2
    return {"whole": sums, "first half": sums.iloc[:k], "second half": sums.iloc[k:]}[part]


def characters(sums: dict[str, pd.DataFrame], costs: dict, *, part: str = "whole",
               stop_moves: float = STOP) -> pd.DataFrame:
    """One row per stock: its character numbers over the whole window or one half of its
    sessions (:func:`_part`), then ``sessions``, ``first session``, ``last session``, ``price``
    (the average close) and ``dollar volume per session``."""
    if not sums:
        return pd.DataFrame(columns=[*FEATURES, "sessions", "first session", "last session",
                                     "price", "dollar volume per session"])
    parts = {s: _part(f, part) for s, f in sums.items()}
    total = pd.DataFrame({s: f.sum() for s, f in parts.items()}).T
    out = character(total, costs, stop_moves=stop_moves)
    out["sessions"] = total["sessions"].astype(int)
    out["first session"] = [f.index[0].date() if len(f) else None for f in parts.values()]
    out["last session"] = [f.index[-1].date() if len(f) else None for f in parts.values()]
    out["price"] = total["close_sum"] / total["minutes"]
    out["dollar volume per session"] = total["dollar_volume"] / total["sessions"]
    out.index.name = "stock"
    return out


def moving(sums: pd.DataFrame, costs: dict, *, sessions: int = 20,
           stop_moves: float = STOP) -> pd.DataFrame:
    """One stock's character numbers over a moving window of ``sessions`` sessions, session by
    session (NaN until that many exist): how steady each number is over time."""
    return character(sums.rolling(sessions).sum(), costs, stop_moves=stop_moves)


def scores(features: pd.DataFrame) -> pd.DataFrame:
    """Every number as a score against the other stocks: the normal score of its rank,
    Φ⁻¹((rank - ½) / n). 0 is the middle stock, +1 about the 84th percentile, -1 the 16th; a
    stock far out cannot stretch the scale, and every number weighs the same."""
    from scipy.stats import norm

    ranks = features.rank()
    return pd.DataFrame(norm.ppf((ranks - 0.5) / features.notna().sum()), index=features.index,
                        columns=features.columns)


def words(profile: pd.Series, *, n: int = 3, least: float = 0.5) -> str:
    """A group in words: the ``n`` numbers on which its centre sits furthest from the middle
    stock (at least ``least`` score units away), each said in :data:`FEATURES` words."""
    far = profile.abs().sort_values(ascending=False)
    said = [FEATURES[f]["words"][int(profile[f] > 0)] for f in far.index[:n] if far[f] >= least]
    return " · ".join(said) or "close to the middle on every number"


def _closest(sc: pd.DataFrame, centre: pd.DataFrame) -> np.ndarray:
    """The group whose centre is closest to every stock's scores."""
    d = ((sc.to_numpy(float)[:, None, :] - centre.to_numpy(float)[None]) ** 2).sum(axis=2)
    return centre.index.to_numpy()[d.argmin(axis=1)]


def grouping(sc: pd.DataFrame, k: int, *, n_tune: int = N_TUNE, halves=None) -> dict:
    """The stocks cut into ``k`` groups by a Ward tree on their scores (each merge adds the
    least spread inside the groups: compact groups of similar size).

    Args:
        sc: :func:`scores` (stocks x the numbers used; no gaps).
        k: Number of groups.
        n_tune: Members of every group the tuning reads (the most typical); the others check.
        halves: The same stocks' scores on the first and on the second half of their sessions
            (optional): does each half alone put a stock closest to its own group's centre?

    Returns:
        ``labels`` (stock -> group 1 .. k, 1 the largest), ``centre`` (group x number: the mean
        score), ``members`` (stock, group, ``distance`` to the group's centre in score units,
        ``rank`` inside the group, 1 the most typical, ``role`` "tune" or "check", with
        ``halves`` ``stays``: both halves alone are closest to its group's centre; sorted by
        group and rank), ``groups`` (one row per group: size, words, tune and check members,
        with ``halves`` the share that stays) and ``tree`` (the linkage).
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    tree = linkage(sc.to_numpy(float), method="ward")
    raw = pd.Series(fcluster(tree, k, criterion="maxclust"), index=sc.index)
    order = {g: i + 1 for i, g in enumerate(raw.value_counts().index)}
    labels = raw.map(order).rename("group")
    centre = sc.groupby(labels).mean()
    gap = sc.to_numpy(float) - centre.loc[labels].to_numpy(float)
    members = pd.DataFrame({"group": labels, "distance": np.sqrt((gap ** 2).sum(axis=1))})
    members["rank"] = members.groupby("group")["distance"].rank(method="first").astype(int)
    members["role"] = np.where(members["rank"] <= n_tune, "tune", "check")
    if halves is not None:
        members["stays"] = np.logical_and.reduce(
            [_closest(h.loc[sc.index], centre) == labels.to_numpy() for h in halves])
    members = members.sort_values(["group", "rank"])
    members.index.name = "stock"
    rows = []
    for g, part in members.groupby("group"):
        rows.append({"group": int(g), "stocks": len(part), "in words": words(centre.loc[g]),
                     "tune on": ", ".join(part.index[part["role"] == "tune"]),
                     "check on": int((part["role"] == "check").sum()),
                     "typical distance": float(part["distance"].median()),
                     **({"stays on both halves": float(part["stays"].mean())}
                        if "stays" in part else {})})
    return {"labels": labels, "centre": centre, "members": members,
            "groups": pd.DataFrame(rows), "tree": tree}


def quality(sc: pd.DataFrame, first: pd.DataFrame, second: pd.DataFrame,
            ks=KS) -> pd.DataFrame:
    """How good each number of groups is.

    Args:
        sc: :func:`scores` on the whole window.
        first, second: The same stocks' scores on the first and on the second half of their
            sessions.
        ks: The numbers of groups to try.

    Returns:
        One row per number of groups: ``explained`` (the share of the stocks' spread, in score
        units, that lies BETWEEN the groups: 0 the groups say nothing, 1 every stock sits on its
        group's centre), ``separation`` (the silhouette: +1 every stock far closer to its own
        group than to the next, 0 on the border), ``same on both halves`` (the adjusted Rand
        index of the groups formed on each half alone: 1 identical, 0 no more alike than chance),
        ``stays in its group`` (the share of stocks that each half alone puts closest to their
        own group's centre: is a stock's group its lasting character?) and ``stays by chance``
        (the same if the halves said nothing about the stock), and ``smallest group``.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    x = sc.to_numpy(float)
    trees = [linkage(f.loc[sc.index].to_numpy(float), method="ward") for f in (sc, first, second)]
    total = float(((x - x.mean(axis=0)) ** 2).sum())
    rows = []
    for k in ks:
        whole, a, b = (fcluster(t, k, criterion="maxclust") for t in trees)
        within = sum(float(((x[whole == g] - x[whole == g].mean(axis=0)) ** 2).sum())
                     for g in np.unique(whole))
        centre = sc.groupby(whole).mean()
        ga, gb = _closest(first.loc[sc.index], centre), _closest(second.loc[sc.index], centre)
        share = lambda g: pd.Series(g).value_counts(normalize=True)  # noqa: E731
        luck = (share(whole) * share(ga) * share(gb)).fillna(0).sum()
        rows.append({"groups": k, "explained": 1 - within / total,
                     "separation": float(silhouette_score(x, whole)),
                     "same on both halves": float(adjusted_rand_score(a, b)),
                     "stays in its group": float(np.mean((ga == whole) & (gb == whole))),
                     "stays by chance": float(luck),
                     "smallest group": int(np.bincount(whole)[1:].min())})
    return pd.DataFrame(rows)


def persistence(first: pd.DataFrame, second: pd.DataFrame) -> pd.Series:
    """Every number's rank correlation across the stocks between the first and the second half
    of their sessions: near 1 the number is the stock's own character, near 0 mostly chance or a
    passing phase. A number that does not persist cannot define a lasting group."""
    return pd.Series({f: first[f].corr(second[f], method="spearman") for f in first.columns},
                     name="first half against second half")


def nearest(sc: pd.DataFrame, stock: str, n: int = 10) -> pd.DataFrame:
    """The ``n`` stocks closest to ``stock`` in score units (Euclidean distance)."""
    d = np.sqrt(((sc - sc.loc[stock]) ** 2).sum(axis=1)).drop(stock)
    return d.nsmallest(n).rename("distance").to_frame()


def flat_map(sc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The stocks on a 2-D map: the two directions along which they differ most (principal
    components of the scores). Returns the coordinates (``x``, ``y``) and each axis's name: the
    share of the spread it shows and the numbers that pull along it most."""
    x = sc - sc.mean()
    _, s, vt = np.linalg.svd(x.to_numpy(float), full_matrices=False)
    share = s ** 2 / (s ** 2).sum()
    xy = pd.DataFrame(x.to_numpy(float) @ vt[:2].T, index=sc.index, columns=["x", "y"])
    names = []
    for i in range(2):
        load = pd.Series(vt[i], index=sc.columns)
        top = load.abs().sort_values(ascending=False).index[:3]
        names.append(f"map axis {i + 1} ({share[i]:.0%} of the spread): "
                     + ", ".join(f"{'+' if load[f] > 0 else '−'}{f}" for f in top))
    return xy, names


def pool(engine, start: str = WINDOW[0], end: str = WINDOW[1], *, refresh: bool = False,
         folder: Path = STORE) -> pd.DataFrame:
    """Every symbol with 1-minute candles of the regular session between ``start`` and ``end``,
    with its first and last such minute in the whole database (UTC): what :func:`compute` keys a
    stock's sums by, so a stock is computed again when either moves (the sweep added earlier
    minutes, or later ones). Asking the database takes one or two minutes (a time limit per
    stock makes the database plan a slow scan); the answer is kept (``refresh`` asks again).

    Returns:
        ``symbol``, ``first``, ``last``; ``attrs["asked"]``: when the database was asked.
    """
    from sqlalchemy import text

    path = Path(folder) / f"pool-{start}-{end}.pkl"
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    with engine.connect() as c:
        names = pd.read_sql(text("SELECT DISTINCT symbol FROM prices_ohlcv WHERE bar = '1m'"), c)
        out = pd.read_sql(text(
            "SELECT s.symbol, f.first, l.last FROM unnest(CAST(:s AS text[])) AS s(symbol) "
            "CROSS JOIN LATERAL (SELECT min(ts) AS first FROM prices_ohlcv p "
            "WHERE p.symbol = s.symbol AND p.bar = '1m' AND p.rth) f "
            "CROSS JOIN LATERAL (SELECT max(ts) AS last FROM prices_ohlcv p "
            "WHERE p.symbol = s.symbol AND p.bar = '1m' AND p.rth) l"),
            c, params={"s": sorted(names["symbol"])})
    a = pd.Timestamp(start).tz_localize("UTC")
    b = pd.Timestamp(DATA.last_stamp(end, "1m")).tz_localize("UTC")
    out = out.dropna(subset=["first"])
    out = out[(out["first"] < b) & (out["last"] >= a)].sort_values("symbol").reset_index(drop=True)
    out.attrs["asked"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def _code() -> str:
    """A key that changes whenever the code that computes the sums changes (not the grouping
    or the charts, so an edit there keeps every stock's sums)."""
    parts = [_regular, _returns, _blocks, _move, block_returns, band_events, session_sums,
             IND.bollinger, IND.true_range, IND.typical_move, IND.available_at, PIPE._source,
             PIPE.bands, DATA.session_day, DATA.session_part, DATA.load_minutes]
    src = "".join(inspect.getsource(f) for f in parts) + repr((BLOCK, BACK, FOLLOW, TOP,
                                                               OPEN_END, BAND, MOVE))
    return hashlib.sha1(src.encode()).hexdigest()[:10]


def _where(folder: Path, start: str, end: str) -> Path:
    """The folder of every stock's sums for this window and this code."""
    return Path(folder) / "sums" / f"{start}-{end}-{_code()}"


def _file(where: Path, row) -> Path:
    """Where one stock's sums are kept: its name and a key of its first and last minute, so a
    stock whose candles grew (an earlier start, a later end) is computed again."""
    key = hashlib.sha1(f"{row.first}|{row.last}".encode()).hexdigest()[:8]
    return where / f"{row.symbol}-{key}.pkl"


def missing(stocks: pd.DataFrame, start: str = WINDOW[0], end: str = WINDOW[1], *,
            folder: Path = STORE) -> list[str]:
    """The stocks of a :func:`pool` (or a part of it) whose sums are not kept yet for their
    present candles."""
    where = _where(folder, start, end)
    return [row.symbol for row in stocks.itertuples() if not _file(where, row).exists()]


def compute(engine, stocks: pd.DataFrame, start: str = WINDOW[0], end: str = WINDOW[1], *,
            market: str = DATA.MARKET, workers: int = 8, folder: Path = STORE,
            progress=None) -> dict[str, pd.DataFrame]:
    """Every stock's :func:`session_sums`, kept on disk; only the missing ones are computed.

    Each missing stock's candles are loaded and measured on its own (``workers`` threads at a
    time: the database may be read in parallel), so the whole pool never sits in memory.

    Args:
        engine: Database engine.
        stocks: :func:`pool`, or a part of it.
        start, end: The window (both days included).
        market: The market fund (its 15-minute returns give the market link).
        workers: Stocks loaded at once (threads: they overlap the database reads; the
            measuring itself mostly runs one at a time).
        folder: Where the sums are kept.
        progress: Called after each computed stock (e.g. a progress bar's update).

    Returns:
        ``{stock: sums}`` for every stock of ``stocks``.
    """
    where = _where(folder, start, end)
    where.mkdir(parents=True, exist_ok=True)
    files_ = {row.symbol: _file(where, row) for row in stocks.itertuples()}
    todo = [s for s, f in files_.items() if not f.exists()]
    out = {s: pickle.loads(f.read_bytes()) for s, f in files_.items() if f.exists()}
    if not todo:
        return out
    mkt = block_returns(DATA.load_minutes(engine, [market], start, end)[market])

    def one(stock: str) -> tuple[str, pd.DataFrame]:
        with np.errstate(divide="ignore"):       # a typical move of 0: that close is left out
            sums = session_sums(DATA.load_minutes(engine, [stock], start, end)[stock], mkt)
        for old in where.glob(f"{stock}-*.pkl"):          # kept for older candles
            old.unlink()
        tmp = files_[stock].with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(sums))
        tmp.replace(files_[stock])
        return stock, sums

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for job in as_completed([ex.submit(one, s) for s in todo]):
            stock, sums = job.result()
            out[stock] = sums
            if progress is not None:
                progress()
    return {s: out[s] for s in files_}


def save(result: dict, features: pd.DataFrame, *, meta: dict, name: str,
         folder: Path = STORE / "exports") -> Path:
    """Write the groups (:func:`grouping`) with what made them (``meta``: the window, the
    numbers used, the number of groups, ...) to ``<folder>/<date>-<name>.json``: every group's
    words, its profile (the median of every number of ``features``, :func:`characters`), the
    members it tunes on and the members it checks on, each with its rank and distance. Returns
    the file."""
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    members, groups = result["members"], result["groups"].set_index("group")
    numbers = [f for f in FEATURES if f in features]
    out = []
    for g, part in members.groupby("group"):
        profile = features.loc[part.index, numbers].median()
        out.append({"group": int(g), "words": groups.loc[g, "in words"], "stocks": len(part),
                    "tune": list(part.index[part["role"] == "tune"]),
                    "check": list(part.index[part["role"] == "check"]),
                    "profile": {f: round(float(v), 6) for f, v in profile.items()},
                    "members": [{"stock": s, "rank": int(r["rank"]),
                                 "distance": round(float(r["distance"]), 4)}
                                for s, r in part.iterrows()]})
    path = folder / f"{stamp}-{name}.json"
    path.write_text(json.dumps({"made": stamp, "meta": meta, "groups": out}, indent=1,
                               default=str))
    return path


def load(path: Path) -> dict:
    """An exported groups file: ``made``, ``meta`` and ``groups`` (one dict per group)."""
    return json.loads(Path(path).read_text())


def files(folder: Path = STORE / "exports") -> list[Path]:
    """The exported groups files, newest first."""
    return sorted(Path(folder).glob("*.json"), reverse=True) if Path(folder).exists() else []


if __name__ == "__main__":       # compute every missing stock's sums from a terminal
    import argparse
    import time

    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--refresh", action="store_true", help="ask the database for the pool again")
    ap.add_argument("--limit", type=int, default=0, help="only the first n stocks (a test)")
    args = ap.parse_args()
    eng = make_engine(load_settings())
    found = pool(eng, refresh=args.refresh)
    found = found[found["symbol"] != DATA.MARKET]
    found = found.head(args.limit) if args.limit else found
    todo = missing(found)
    print(f"{len(found)} stocks in the pool (asked {found.attrs.get('asked')}), "
          f"{len(todo)} to compute", flush=True)
    t0, done = time.time(), [0]

    def tick():
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(todo):
            rate = (time.time() - t0) / done[0]
            print(f"{done[0]}/{len(todo)} · {rate:.2f} s a stock · "
                  f"{rate * (len(todo) - done[0]) / 60:.0f} min left", flush=True)

    compute(eng, found, workers=args.workers, progress=tick)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
