"""From candles to what the rules read: the one computation both notebooks run.

The main notebook calls these functions cell by cell (each slider redraws its own part); the
tuning notebook calls them through ``tuning.Lab``, which keeps every part it built. So settings
exported by the tuner give the main notebook exactly the trades the tuner scored.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _sibling(name: str):
    """One of this strategy's helper files, loaded by path."""
    spec = importlib.util.spec_from_file_location(f"classic_signals_pl_{name}",
                                                  HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IND, COMBINE, DATA = _sibling("indicators"), _sibling("combine"), _sibling("data")


def _source(candles: pd.DataFrame, extended: bool) -> pd.DataFrame:
    """The candles an intraday indicator reads: the regular session, or every stored minute."""
    return candles if extended else candles[candles["regular"]]


def bands(candles: pd.DataFrame, length: int, width: float, extended: bool) -> pd.DataFrame:
    """The Bollinger Bands on every candle (``indicators.bollinger``)."""
    return IND.bollinger(_source(candles, extended)["close"], length, width).reindex(
        candles.index)


def average(candles: pd.DataFrame, length: int, extended: bool) -> pd.Series:
    """The minute chart's moving average on every candle."""
    return IND.moving_average(_source(candles, extended)["close"], length).reindex(
        candles.index)


def move(minutes: pd.DataFrame, candles: pd.DataFrame, size: int,
         length: int) -> tuple[pd.Series, pd.Series]:
    """The typical move of ``size``-minute regular-session candles (the average of the last
    ``length`` true ranges, the night left out), as known at every regular candle: in price
    (the trailing stop's yardstick) and in percent of the price."""
    blocks = DATA.resample(minutes[minutes["regular"]], size)
    typical = IND.typical_move(blocks, length, across_nights=False)
    regular = candles["regular"]
    return (IND.available_at(typical, size, candles.index).where(regular),
            IND.available_at(typical / blocks["close"] * 100, size, candles.index).where(regular))


def daily_move(days: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """The stock's typical daily move (the average true range of its last 14 daily candles, the
    yardstick of the daily trends and the levels), in price, as known at every candle of
    ``index``: the value at the previous session's close."""
    return IND.previous_session(IND.typical_move(days, 14, across_nights=True), index)


def options(symbol: str, start, end) -> pd.DataFrame:
    """A stock's daily option features (``indicators.option_features`` of ``data.load_chains``):
    one row per date; empty without option data."""
    return IND.option_features(DATA.load_chains(symbol, start, end))


def option_context(options_: pd.DataFrame, days: pd.DataFrame,
                   index: pd.DatetimeIndex) -> pd.DataFrame:
    """What the option rules read, as known at every candle of ``index`` (the previous
    session's end-of-day chain): ``ratio`` (the expected move ÷ the recent daily moves,
    ``indicators.options_ratio``), ``term`` (front ÷ 30-day at-the-money volatility),
    ``put/call`` (the log of the put/call volume over its usual: the last 20 dates' median) and
    ``gamma`` (dealer gamma ÷ its usual size: the last 20 dates' mean absolute value, so one
    setting means the same for every stock)."""
    pcv = np.log(options_["put/call volume"])
    usual = options_["gamma"].abs().rolling(20, min_periods=10).mean()
    daily = pd.DataFrame({
        "ratio": IND.options_ratio(options_["move"], days),
        "term": options_["term"],
        "put/call": pcv - pcv.rolling(20, min_periods=10).median(),
        "gamma": options_["gamma"] / usual})
    return pd.DataFrame({k: IND.previous_session(v, index) for k, v in daily.items()})


def walls(options_: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """The option walls as levels (``day``, ``price``, like ``indicators.major_levels``): each
    chain's call wall and put wall, moved into the daily candles' (split-adjusted) prices, for
    the session after the chain's date."""
    if not len(options_):
        return pd.DataFrame(columns=["day", "price"])
    after = days.index.searchsorted(options_.index, side="right")
    keep = after < len(days.index)
    factor = (days["close"].reindex(options_.index) / options_["price"]).to_numpy()
    rows = [{"day": days.index[a], "price": float(w * f)}
            for a, ok, f, (cw, pw) in zip(after, keep, factor,
                                          options_[["call wall", "put wall"]].to_numpy(),
                                          strict=True)
            if ok and np.isfinite(f) for w in (cw, pw) if np.isfinite(w)]
    return pd.DataFrame(rows, columns=["day", "price"])


def trends(days: dict, length: int, slope: int, flat: float) -> dict:
    """Every symbol's daily trend state (1 up, 0 sideways, -1 down), judged at each close."""
    return {s: IND.trend_state(d, length=length, slope_days=slope, flat=flat)["state"]
            for s, d in days.items()}


def levels(days: pd.DataFrame, span: int, lookback: int, tolerance: float, touches: int,
           mix: bool) -> pd.DataFrame:
    """The major levels known at each session's open (``indicators.major_levels``)."""
    return IND.major_levels(days, span=span, lookback=lookback, tolerance=tolerance,
                            min_touches=touches, mix=mix)


def context(names: dict, candles: pd.DataFrame, bands_: pd.DataFrame, move_: pd.Series,
            trends_: dict, average_: pd.Series, levels_: pd.DataFrame, prev: pd.DataFrame,
            reports: pd.DataFrame, days: pd.DataFrame | None = None,
            options_: pd.DataFrame | None = None) -> dict:
    """What the rules read for one stock (``combine.make_context``), on its regular-session
    candles: the bands, the typical move, the daily trends as known at the open, the moving
    average, the nearest major level and previous high / low above and below each close, the
    sessions until the next report, the typical daily move (from ``days``, the stock's daily
    candles) and what its option chains say (from ``options_``, :func:`options`): the option
    rules' readings (:func:`option_context`) and the option walls as a third kind of level
    (:func:`walls`), all as known at the open.

    Args:
        names: ``{"stock", "sector", "market"}``.
        candles: The stock's candles (every stored hour; the regular ones are traded).
        bands_, move_, trends_, average_, levels_: The parts above.
        prev: ``indicators.previous_range`` of the stock's daily candles.
        reports: ``data.earnings_dates`` (every stock; empty if unknown).
    """
    c = candles[candles["regular"]]
    known = {role: IND.previous_session(trends_[sym], c.index) for role, sym in names.items()}
    near = {"major": IND.nearest_levels(c["close"], levels_),
            "previous": IND.nearest_levels(c["close"], prev)}
    have = options_ is not None and days is not None and len(options_)
    if have:
        near["walls"] = IND.nearest_levels(c["close"], walls(options_, days))
    mine = reports.loc[reports["symbol"] == names["stock"], "reaction day"] if len(reports) \
        else []
    report = IND.sessions_to_event(c.index, mine)
    return COMBINE.make_context(names, c, bands_, move_, known, average_, near, report,
                                None if days is None else daily_move(days, c.index),
                                option_context(options_, days, c.index) if have else None)


def targets(ctx: dict, levels_: pd.DataFrame, prev: pd.DataFrame, kind: str,
            days: float = 0.0, walls_: pd.DataFrame | None = None):
    """The profit target a trade entered at each candle would get (``simulate.run``'s
    ``targets``): the nearest level at least one typical move from the entry (the open), above
    for a long and below for a short (``kind`` "major", "both", or "option walls": the walls of
    :func:`walls`); ``kind`` "daily moves": ``days`` typical daily moves from the entry; None
    when ``kind`` is "none" (or "option walls" without option data)."""
    if kind == "daily moves":
        entry, gap = ctx["candles"]["open"], days * ctx["daily_move"]
        return pd.DataFrame({"long": entry + gap, "short": entry - gap})
    goal = {"none": None, "major": levels_, "both": pd.concat([levels_, prev]),
            "option walls": walls_ if walls_ is not None and len(walls_) else None}[kind]
    if goal is None:
        return None
    return IND.nearest_levels(ctx["candles"]["open"], goal, gap=ctx["move"]).set_axis(
        ["long", "short"], axis=1)
