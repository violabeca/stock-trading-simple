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


@pytest.mark.parametrize(("gap", "kept"), [(0, 2), (4, 2), (5, 1), (6, 1), (7, 0)])
def test_the_anchor_gap_ends_the_chain_at_the_first_line_too_short(gap, kept):
    """The hand-made chain's lines have their points A and B 6 and 4 candles apart: a gap of 5
    keeps the first line and leaves the second as the pending ray; 7 keeps none. What is kept
    is the same as without a gap."""
    days = daily(HIGHS)
    frames = {"1D": TD.candles(days, None, "1D", TD.clock(days))}
    free = TD.top_down(frames, "down", touch=unit(days, 0.15), pierce=0.0)["1D"]
    lines = TD.top_down(frames, "down", touch=unit(days, 0.15), pierce=0.0, gap=gap)["1D"]
    confirmed = lines[lines["confirmed"].astype(bool)]
    assert len(confirmed) == kept
    pd.testing.assert_frame_equal(confirmed, free.iloc[:kept])
    assert ((confirmed["b_x"] - confirmed["a_x"]) >= gap).all()
    if kept < 2:                                  # the first line too short: it is the pending ray
        pending = lines[~lines["confirmed"].astype(bool)].iloc[0]
        assert pending["a_x"] == (0, 6)[kept] and pending["b_x"] == (6, 10)[kept]


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


@pytest.mark.parametrize("seed", range(6))
def test_with_anchors_on_the_line_every_anchor_is_a_candle_on_it(seed):
    """``on_line`` and no poke allowance, on random monthly + daily candles: point B is the
    angle candle, so it lies exactly on the line (a monthly line's B moves onto the daily candle
    that holds its price), every line starts exactly at the previous
    line's point B, no candle of the daily screen goes through a line, and a line adjusted on
    the daily chart keeps the monthly chart as its origin (never a chart below the one it was
    last drawn on)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 700)))
    days = pd.DataFrame({"open": close, "close": close * np.exp(rng.normal(0, 0.01, 700))},
                        index=pd.bdate_range("2020-01-01", periods=700))
    days["high"] = days[["open", "close"]].max(axis=1) * (1 + rng.uniform(0, 0.01, 700))
    days["low"] = days[["open", "close"]].min(axis=1) * (1 - rng.uniform(0, 0.01, 700))
    sessions = TD.clock(days)
    frames = {tf: TD.candles(days, None, tf, sessions) for tf in ("1M", "1D")}
    order = list(TD.TIMEFRAMES)
    for side in TD.SIDES:
        lines = TD.top_down(frames, side, pierce=0.0, screen=120, on_line=True)["1D"]
        lines = lines[lines["confirmed"].astype(bool)]
        # B is the angle candle: the same price always; the same candle on the line's own
        # chart (the monthly line the daily walk restarts from moves B onto the daily candle
        # that holds its price, inside the monthly angle candle)
        assert (lines["b_y"] == lines["stop_y"]).all() and (lines["b_x"] >= lines["stop_x"]).all()
        here = lines[lines["tf"] == "1D"]
        assert (here["b_x"] == here["stop_x"]).all()
        assert np.allclose(here["a_y"] + here["slope"] * (here["b_x"] - here["a_x"]),
                           here["b_y"])
        assert (lines["b_x"].to_numpy()[:-1] == lines["a_x"].to_numpy()[1:]).all()
        assert (lines["b_y"].to_numpy()[:-1] == lines["a_y"].to_numpy()[1:]).all()
        assert TD.crossings(lines, frames["1D"].iloc[-120:], False) == 0
        assert all(order.index(o) <= order.index(t)
                   for o, t in zip(lines["origin"], lines["tf"], strict=True))


@pytest.mark.parametrize("side", ["up", "down"])
def test_the_cascade_draws_one_line_per_chart_each_starting_on_the_one_above(side):
    """On hand-made 1-minute candles: at most one line per chart, top-down, the daily one from
    the extreme of the daily candles in view; every line below starts at the price of the line
    above's point B, inside that point's candle; point B is the angle candle, on the line; no
    candle of a line's own chart crosses it; a pending ray, if any, comes last. The hand-made
    market drifts up, so the downward lines are drawn on it turned upside down."""
    from tests.test_tori_trades_trendline_bounce import daily_of, market, upside_down

    daily, minutes = market(sessions=60)
    if side == "down":
        minutes = upside_down(minutes)
        daily = daily_of(minutes)
    sessions = TD.clock(daily)
    frames = {tf: TD.candles(daily, minutes, tf, sessions) for tf in ("1D", "4H", "1H", "15m")}
    lines = TD.cascade(frames, side, window=40)
    confirmed = lines[lines["confirmed"].astype(bool)]
    assert len(confirmed) >= 2
    assert list(confirmed["tf"]) == list(frames)[:len(confirmed)]
    assert (lines["confirmed"].astype(bool).to_numpy()[:-1]).all()      # pending: last only
    view = frames["1D"].iloc[-40:][TD.SIDES[side]]
    assert lines["a_price"].iloc[0] == (view.max() if side == "down" else view.min())
    assert (confirmed["b_x"] == confirmed["stop_x"]).all()
    assert np.allclose(confirmed["a_y"] + confirmed["slope"]
                       * (confirmed["b_x"] - confirmed["a_x"]), confirmed["b_y"])
    for above, below in zip(confirmed.iloc[:-1].itertuples(), confirmed.iloc[1:].itertuples(),
                            strict=True):
        assert below.a_price == pytest.approx(above.b_price)
        assert above.b_x <= below.a_x < above.b_x + (1 if above.tf == "1D" else 0.62)
    for line in confirmed.itertuples():
        own = frames[line.tf]
        assert TD.crossings(confirmed[confirmed["tf"] == line.tf], own, False) == 0


def test_the_cascade_stops_at_the_first_line_shorter_than_its_chart_minimum():
    """With a minimum per chart, every line drawn has at least that many candles of its chart
    between its points A and B; the chain is the same as without one up to the first line too
    short, which becomes the pending ray, and nothing is drawn below it."""
    from tests.test_tori_trades_trendline_bounce import market

    daily, minutes = market(sessions=60)
    sessions = TD.clock(daily)
    frames = {tf: TD.candles(daily, minutes, tf, sessions) for tf in ("1D", "4H", "1H", "15m")}
    free = TD.cascade(frames, "up", window=40)
    free = free[free["confirmed"].astype(bool)]
    per = {"1D": 1, **{tf: TD.SESSION / m for tf, m in TD.MINUTES.items()}}   # candles a session
    apart = (free["b_x"] - free["a_x"]) * free["tf"].map(per)
    assert len(free) >= 2
    # a minimum just above the second line's length cuts the chain there
    cut = {free["tf"].iloc[1]: int(np.floor(apart.iloc[1])) + 1}
    lines = TD.cascade(frames, "up", window=40, minimum=cut)
    pd.testing.assert_frame_equal(lines.iloc[:1], free.iloc[:1])
    assert len(lines) == 2 and not lines["confirmed"].astype(bool).iloc[1]
    assert lines["tf"].iloc[1] == free["tf"].iloc[1]
    every = TD.cascade(frames, "up", window=40, minimum={"1D": 2, "4H": 2, "1H": 2, "15m": 4})
    ok = every[every["confirmed"].astype(bool)]
    need = ok["tf"].map({"1D": 2, "4H": 2, "1H": 2, "15m": 4})
    assert ((ok["b_x"] - ok["a_x"]) * ok["tf"].map(per) >= need - 1e-9).all()


def test_session_edges_count_from_the_open_and_from_the_close():
    """The first 2 and last 3 of a regular session's 26 fifteen-minute candles, and of a
    half-day's 14 (it closes at 13:00)."""
    days = [pd.Timestamp("2025-07-02"), pd.Timestamp("2025-07-03")]          # 07-03: half day
    index = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + 15 * i)
                              for d, n in zip(days, (26, 14), strict=True) for i in range(n)])
    edge = TD.session_edges(index, "15m", 2, 3)
    assert edge[:26].nonzero()[0].tolist() == [0, 1, 23, 24, 25]
    assert edge[26:].nonzero()[0].tolist() == [0, 1, 11, 12, 13]


def test_the_cascade_leaves_the_session_edges_out_of_the_fifteen_minute_line():
    """With ``trim`` (4, 4), the 15-minute line's point B (its angle candle) is never one of a
    session's first or last four 15-minute candles, its point A is still the price of the
    line above's point B, and the charts above are unchanged."""
    from tests.test_tori_trades_trendline_bounce import market

    daily, minutes = market(sessions=60)
    sessions = TD.clock(daily)
    frames = {tf: TD.candles(daily, minutes, tf, sessions) for tf in ("1D", "4H", "1H", "15m")}
    found = False
    for side in TD.SIDES:
        free = TD.cascade(frames, side, window=40)
        lines = TD.cascade(frames, side, window=40, trim=(4, 4))
        above = lines[lines["tf"] != "15m"]
        pd.testing.assert_frame_equal(above, free[free["tf"] != "15m"].iloc[:len(above)])
        mine = lines[lines["tf"] == "15m"]
        assert not TD.session_edges(pd.DatetimeIndex(mine["b_time"]), "15m", 4, 4).any()
        if len(mine):
            assert mine["a_price"].iloc[0] == pytest.approx(above["b_price"].iloc[-1])
        found = found or len(mine) > 0
    assert found


def test_the_noise_is_the_moving_average_of_the_true_range():
    """Candle by candle, the mean of the last ``window`` true ranges (fewer at the start); the
    true range reaches back to the close before when the candle gapped."""
    c = pd.DataFrame({"high": [11.0, 12.0, 15.0, 13.0], "low": [9.0, 10.0, 13.0, 12.0],
                      "close": [10.0, 11.0, 14.0, 12.5]})
    tr = [2.0, 2.0, 4.0, 2.0]                     # the third gapped up from 11: 15 - 11
    assert TD.noise(c, 2).tolist() == pytest.approx([2.0, 2.0, 3.0, 3.0])
    assert TD.noise(c, 10).tolist() == pytest.approx(np.cumsum(tr) / np.arange(1, 5))


@pytest.mark.parametrize("side", ["up", "down"])
def test_a_banded_line_lets_candles_reach_in_by_its_band_and_takes_b_inside_it(side):
    """With a line band, no candle of a line's chart goes beyond it by more than the band at
    that candle (the band follows the noise), point B lies inside the band, and the next line
    starts at it; a band of 0 draws exactly the strict lines."""
    from tests.test_tori_trades_trendline_bounce import daily_of, market, upside_down

    daily, minutes = market(sessions=60)
    if side == "down":
        minutes = upside_down(minutes)
        daily = daily_of(minutes)
    sessions = TD.clock(daily)
    frames = {tf: TD.candles(daily, minutes, tf, sessions) for tf in ("1D", "4H", "1H", "15m")}
    pd.testing.assert_frame_equal(TD.cascade(frames, side, window=40, band=0.0),
                                  TD.cascade(frames, side, window=40))
    lines = TD.cascade(frames, side, window=40, band=0.5, noise_window=20)
    ok = lines[lines["confirmed"].astype(bool)]
    assert len(ok) >= 2
    for line in ok.itertuples():
        own = frames[line.tf]
        w = 0.5 * TD.noise(own, 20)
        right = (own["x"] > line.a_x).to_numpy()
        level = line.a_y + line.slope * (own["x"].to_numpy() - line.a_x)
        y = own[TD.SIDES[side]].to_numpy()
        inside = (y - level) if side == "up" else (level - y)       # negative: beyond the line
        assert (inside[right] >= -w[right] - 1e-9).all()
        j = own.index.get_loc(line.b_time)
        assert abs(inside[j]) <= w[j] + 1e-9
    assert (ok["b_time"].to_numpy()[:-1] <= ok["a_time"].to_numpy()[1:]).all()


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
