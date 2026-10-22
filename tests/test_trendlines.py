"""level_flow trendlines: sloped support/resistance from swing points, on engineered charts.

Every chart is a triangle wave whose peaks sit EXACTLY on a known line (in log price), so the
test can assert the line the method must find: its slope, its tests, when it is known, when it
breaks and what a forward test calls it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[1] / "strategies" / "level_flow" / "trendlines.py"
_SPEC = importlib.util.spec_from_file_location("trendlines", _PATH)
tl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tl)

_SLOPE, _AMP, _W = -0.002, 0.06, 0.002   # line slope (log/candle), wave depth, candle half-range
_PEAKS = (10, 40, 70)                     # the wave touches the line here (period 30)


def _chart(n: int = 100, *, spike: int | None = None, breakout: int | None = None):
    """Candles under a descending resistance line, peaks on the line every 30 candles."""
    t = np.arange(n, dtype=float)
    line = np.log(100.0) + _SLOPE * t
    tri = np.abs(((t - _PEAKS[0]) / 15.0) % 2.0 - 1.0)        # 1 at the peaks, 0 at troughs
    mid = line - _W - _AMP * (1.0 - tri)                       # highs (mid + _W) touch the line
    if spike is not None:
        mid[spike] = line[spike] + 0.02                        # trades clean through the line
    if breakout is not None:
        mid[breakout:] = line[breakout:] + 0.03 + 0.001 * (t[breakout:] - breakout)
    close = np.exp(mid)
    return pd.DataFrame({"open": close, "high": np.exp(mid + _W), "low": np.exp(mid - _W),
                         "close": close},
                        index=pd.date_range("2026-01-05 10:30", periods=n, freq="h"))


# The wave is deep next to its candles (about 5 ATRs from the line on average), so the chord
# cut is off here; test_a_line_price_stood_far_from_is_dropped covers it on its own.
_KNOBS = {"pivot_k": 2, "min_gap": 5, "tol_atr": 0.5, "leave_atr": 1.0, "max_slope_atr": 2.0,
          "reach_atr": None, "lookback": 100, "max_gap_atr": None}


def test_finds_the_line_the_peaks_sit_on():
    lines = tl.find_lines(_chart(), None, **_KNOBS)
    res = lines[lines["side"] == "resistance"].iloc[0]
    assert res["touch_pos"] == list(_PEAKS)
    assert res["slope"] == pytest.approx(_SLOPE, abs=1e-9)
    assert res["status"] == "active" and np.isnan(res["break_pos"])
    assert res["level"] == pytest.approx(np.exp(np.log(100.0) + _SLOPE * 99), rel=1e-9)


def test_a_swing_is_used_only_once_it_is_confirmed():
    c = _chart()
    before = tl.find_lines(c, _PEAKS[2] + 1, **_KNOBS)          # the third peak is not known yet
    after = tl.find_lines(c, _PEAKS[2] + 2, **_KNOBS)           # pivot_k = 2 candles later
    assert (before["side"] == "resistance").sum() == 0          # two tests < min_touches 3
    assert after[after["side"] == "resistance"].iloc[0]["touches"] == 3


def test_price_trading_through_between_the_anchors_rejects_the_pair():
    cand = tl.find_lines(_chart(spike=55), None, candidates=True, **_KNOBS)
    res = cand[cand["side"] == "resistance"]
    assert not ((res["x0"] == 10) & (res["x1"] == 70)).any()    # spike at 55 pierces 10 -> 70
    assert ((res["x0"] == 10) & (res["x1"] == 40)).any()         # the unpierced pair survives


def test_a_close_beyond_the_line_breaks_it():
    lines = tl.find_lines(_chart(breakout=85), None, keep_broken=True, **_KNOBS)
    res = lines[lines["side"] == "resistance"].iloc[0]
    assert res["status"] == "broken" and res["break_pos"] == 85


def test_touches_without_a_retreat_are_one_test():
    """A second peak on the line after a shallow dip is the same test, not a new one."""
    c = _chart()
    t = np.arange(41, 46)
    line = np.log(100.0) + _SLOPE * t
    dip = 0.012 * (1.0 - np.abs(t - 43) / 3.0)          # 0 at 40 and 46, deepest at 43
    mid = line - _W - dip
    c.iloc[41:46] = np.exp(np.c_[mid, mid + _W, mid - _W, mid])
    c.iloc[46] = np.exp(np.log(100.0) + _SLOPE * 46 - np.array([_W, 0.0, 2 * _W, _W]))
    knobs = {**_KNOBS, "leave_atr": 3.0}
    pair = lambda f: f[(f["side"] == "resistance") & (f["x0"] == 10) & (f["x1"] == 40)]  # noqa: E731
    merged = pair(tl.find_lines(c, None, candidates=True, **knobs)).iloc[0]
    split = pair(tl.find_lines(c, None, candidates=True, **{**knobs, "leave_atr": 0.0})).iloc[0]
    assert split["touch_pos"] == [10, 40, 46, 70]         # 46 IS a swing on the line ...
    assert merged["touch_pos"] == [10, 40, 70]            # ... but no close left the line


def test_lines_sharing_two_tests_are_one_line():
    c = _chart()
    cand = tl.find_lines(c, None, candidates=True, **_KNOBS)
    res = cand[cand["side"] == "resistance"].set_index(["x0", "x1"])["touches"]
    # Three pairs draw the same line; one starting at 40 does not count the swing at 10.
    assert res.loc[[(10, 40), (10, 70), (40, 70)]].tolist() == [3, 3, 2]
    lines = tl.find_lines(c, None, max_lines=5, min_touches=2, **_KNOBS)
    assert (lines["side"] == "resistance").sum() == 1


def test_forward_tests_call_a_hold_and_a_break():
    c = _chart(130)                                  # the wave keeps touching at 100 and 130
    lines = tl.find_lines(c, 75, **_KNOBS)
    fw = tl.forward_tests(lines, c, 40, hold_atr=2.0)
    res = fw[fw["side"] == "resistance"].iloc[0]
    assert res["outcome"] == "held" and res["test_pos"] == 100
    brk = _chart(130, breakout=100)
    fw = tl.forward_tests(tl.find_lines(brk, 75, **_KNOBS), brk, 40, hold_atr=2.0)
    assert fw[fw["side"] == "resistance"].iloc[0]["outcome"] == "broke"


def test_shifted_twins_keep_the_slope_and_move_away_from_price():
    lines = tl.find_lines(_chart(), None, **_KNOBS)
    twin = tl.shifted(lines, 2.0)
    assert (twin["slope"] == lines["slope"]).all()
    res, twin_res = lines["side"] == "resistance", twin["side"] == "resistance"
    assert (twin.loc[twin_res, "level"] > lines.loc[res, "level"]).all()
    assert twin["dist_atr"].sub(lines["dist_atr"]).abs().round(9).eq(2.0).all()


def test_a_fan_into_one_swing_is_one_line_but_lines_off_one_top_are_two():
    """Lines ending on the same latest test collapse; lines starting at one swing do not."""
    lines = pd.DataFrame({
        "side": "resistance", "x0": [0, 10, 0], "y0": [0.0, 0.0, 0.0],
        "slope": [0.0, 0.001, -0.004], "touches": [3, 3, 3], "score": [5.0, 4.0, 3.0],
        "touch_pos": [[0, 20, 40], [10, 25, 40], [0, 15, 30]], "last_touch": [40, 40, 30],
        "break_pos": np.nan})
    kept = tl._dedup(lines, 60, 1e-6, 5)
    assert kept["score"].tolist() == [5.0, 3.0]     # the fan into 40 keeps its best line


def test_a_line_price_stood_far_from_is_dropped():
    """gap_atr is the mean distance of the highs below the line; max_gap_atr cuts on it."""
    c = _chart()
    kept = tl.find_lines(c, None, **_KNOBS)
    res = kept[kept["side"] == "resistance"].iloc[0]
    atr = kept.attrs["atr"]
    assert res["gap_atr"] == pytest.approx(_AMP / 2 / atr, rel=0.1)   # a triangle wave's mean
    assert not (tl.find_lines(c, None, **{**_KNOBS, "max_gap_atr": 0.9 * res["gap_atr"]})
                ["side"] == "resistance").any()


def test_a_line_goes_stale_once_projected_past_its_own_span():
    """Tested over 10 -> 70 (60 candles): live until candle 130, stale after."""
    c = _chart(200)
    c.iloc[75:] = c.iloc[75:].to_numpy() * np.exp(-0.3)     # price falls away: no more tests
    for asof, alive in ((130, True), (131, False)):
        lines = tl.find_lines(c, asof, **{**_KNOBS, "max_extend": 1.0, "lookback": 200})
        assert ((lines["side"] == "resistance") & (lines["x0"] == 10)).any() == alive
