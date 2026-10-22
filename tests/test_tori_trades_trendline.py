"""tori_trades_trendline: Tori Trades' top-down trendline rules (``topdown.py``) on hand-made
candles, no database. The first point A is the extreme of all history; the ray is the steepest one
no candle crosses; point B is the latest confirmed touch and the next line starts there, steeper; a
stopping candle that is still too new leaves a pending ray; a poke too small to see does not stop
a ray; and the top-down walk keeps the lines left of a lower chart's screen.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

_spec = importlib.util.spec_from_file_location(
    "test_tori_topdown", strategies.STRATEGIES_DIR / "tori_trades_trendline" / "topdown.py")
TD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TD)


def daily(high, *, width=0.5, start="2024-01-01"):
    """Daily candles with the given highs; each low is ``width`` below its high."""
    high = np.asarray(high, dtype=float)
    return pd.DataFrame({"open": high - width / 2, "high": high, "low": high - width,
                         "close": high - width / 2},
                        index=pd.bdate_range(start, periods=len(high)))


def walk(days, side, *, touch=0.0, pierce=0.0, confirm=2, log=False):
    """The lines of one side on the daily chart alone."""
    frames = {"1D": TD.candles(days, None, "1D", TD.clock(days))}
    return TD.top_down(frames, side, log=log, touch=touch, pierce=pierce, confirm=confirm)["1D"]


def unit(days, distance):
    """``distance`` in typical candles of ``days`` (the unit of ``touch`` and ``pierce``)."""
    return distance / TD._typical(days, False)


# x:     0   1  2  3  4  5  6    7  8    9  10   11 12   13
HIGHS = [10, 7, 9, 6, 8, 5, 6.9, 3, 4.5, 2, 3.4, 1, 0.8, 0.6]


def test_chain_by_hand():
    """From the highest high, the ray turns down until x2 and x4 stop it (slope -0.5); x6 comes
    within 0.15 of it, so it is the latest touch: point B. From there the steepest ray no candle
    crosses ends on x10 (-0.875, steeper). From x10 the ray is stopped by the newest candle,
    which is not confirmed yet: a pending ray."""
    days = daily(HIGHS)
    lines = walk(days, "down", touch=unit(days, 0.15))
    first, second, pending = lines.itertuples()
    assert (first.a_x, first.a_y, first.slope, first.stop_x, first.b_x) == (0, 10, -0.5, 4, 6)
    assert (second.a_x, second.stop_x, second.b_x) == (6, 10, 10)
    assert second.slope == pytest.approx(-0.875)
    assert pending.a_x == 10 and pending.stop_x == 13 and not pending.confirmed
    assert first.confirmed and second.confirmed and first.tf == "1D"


def test_without_a_touch_tolerance_point_b_is_the_stopping_candle():
    """With no tolerance only exact touches count: B is the last candle ON the line (x4; x2 also
    lies on it), and the near miss x6 no longer moves it."""
    lines = walk(daily(HIGHS), "down", touch=0.0)
    assert lines["b_x"].iloc[0] == 4 and lines["stop_x"].iloc[0] == 4


def test_upward_lines_mirror_downward_ones():
    """Upward lines under the lows follow the same rules upside down."""
    up = daily(20 - np.asarray(HIGHS) + 0.5, width=0.5)       # lows = 20 - HIGHS
    days = daily(HIGHS)
    down = walk(days, "down", touch=unit(days, 0.15))
    lines = walk(up, "up", touch=unit(up, 0.15))
    assert lines["a_x"].tolist() == down["a_x"].tolist()
    assert lines["stop_x"].tolist() == down["stop_x"].tolist()
    assert np.allclose(lines["slope"], -down["slope"])
    assert lines["a_price"].iloc[0] == 10                    # the lowest low of all history


def test_a_poke_too_small_to_see_does_not_stop_the_ray():
    """The candle after the top is 3 cents lower: with no poke allowed it stops the ray and makes
    a one-candle "line"; allowed a 0.1 poke, the ray runs on to the real touch at x20, and the
    candle after the top pokes through it by only 2 cents."""
    high = [10, 9.97] + [8.0] * 18 + [9.0, 8.0, 8.0]
    days = daily(high)
    assert walk(days, "down")["stop_x"].iloc[0] == 1
    lines = walk(days, "down", pierce=unit(days, 0.1))
    assert lines["stop_x"].iloc[0] == 20
    assert lines["slope"].iloc[0] == pytest.approx(-0.05)
    assert TD.crossings(lines, TD.candles(days, None, "1D", TD.clock(days)), False) == 0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("log", [False, True])
def test_top_down_keeps_her_rules(seed, log):
    """On random monthly + daily candles, after the daily chart: the first point A is the
    extreme of all history; no candle on the daily screen goes through a line; every line starts
    at the previous point B and is steeper; no touch is younger than ``confirm`` candles; lines
    that end left of the screen keep their monthly drawing; and the lines on the screen were
    drawn on the daily chart (the line that ends where the walk restarts may take its latest
    touch on the screen, so a monthly line STARTS left of it)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 700)))
    days = pd.DataFrame({"open": close, "close": close * np.exp(rng.normal(0, 0.01, 700))},
                        index=pd.bdate_range("2020-01-01", periods=700))
    days["high"] = days[["open", "close"]].max(axis=1) * (1 + rng.uniform(0, 0.01, 700))
    days["low"] = days[["open", "close"]].min(axis=1) * (1 - rng.uniform(0, 0.01, 700))
    sessions = TD.clock(days)
    frames = {tf: TD.candles(days, None, tf, sessions) for tf in ("1M", "1D")}
    screen = 120
    x0 = frames["1D"]["x"].iloc[-screen]
    drawn = 0
    for side in TD.SIDES:
        everything = TD.top_down(frames, side, log=log, screen=screen)["1D"]
        # a pending ray alone is right when the last candles stop the first ray: price is at
        # the line from the extreme right now (seed 1's lows, seed 3's highs)
        assert len(everything)
        extreme = days["high"].max() if side == "down" else days["low"].min()
        assert everything["a_price"].iloc[0] == pytest.approx(extreme)
        lines = everything[everything["confirmed"].astype(bool)]
        drawn += len(lines)
        assert TD.crossings(lines, frames["1D"].iloc[-screen:], log) == 0
        assert np.allclose(lines["b_x"].to_numpy(float)[:-1], lines["a_x"].to_numpy(float)[1:])
        steps = np.diff(lines["slope"].to_numpy(float))
        assert (steps < 0).all() if side == "down" else (steps > 0).all()
        assert (lines["b_x"] <= frames["1D"]["x"].iloc[-3]).all()        # confirm = 2
        assert (lines.loc[lines["tf"] == "1M", "a_x"] < x0).all()
        assert (lines.loc[lines["a_x"] >= x0, "tf"] == "1D").all()
    assert drawn >= 2


def test_candles_sit_on_the_session_clock():
    """A 15-minute candle opening at 10:00 on session n sits at n + 30/390; the 1-hour chart's
    15:30 candle is half an hour long; a monthly candle sits at its first session and ends where
    its last session does."""
    days = pd.bdate_range("2025-03-03", periods=25)
    t = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in days[-2:]
                          for m in range(390)]).tz_localize(TD.NY)
    minutes = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5}, index=t)
    bars = daily(np.full(25, 5.0), start="2025-03-03")
    sessions = TD.clock(bars)
    q = TD.candles(bars, minutes, "15m", sessions)
    assert len(q) == 52 and q.loc["2025-04-04 10:00", "x"] == pytest.approx(24 + 30 / 390)
    h = TD.candles(bars, minutes, "1H", sessions)
    assert h.loc["2025-04-04 15:30", "x_end"] == pytest.approx(25)
    m = TD.candles(bars, None, "1M", sessions)
    assert m.index[1] == pd.Timestamp("2025-04-01")
    assert m["x"].iloc[1] == sessions[pd.Timestamp("2025-04-01")]
    assert m["x_end"].iloc[0] == sessions[pd.Timestamp("2025-03-31")] + 1
