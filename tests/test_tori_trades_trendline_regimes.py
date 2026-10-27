"""tori_trades_trendline: the profitable time ranges of one line setting (``regimes.py``), on
hand-made records and candles, no database. The zigzag turns where the record reverses by the
threshold, and only there. Every leg gets its trades, kind and compounded return. A clustered
record is judged unlikely by luck, an alternating one likely. The trade log ranks a side's
trades by profit and finds, among the lines of each trade's setup, the one it bounced off. A
side's replay is the plain ``bounce.replay`` from the session after the first minute, and it is
read back from disk the second time.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

_spec = importlib.util.spec_from_file_location(
    "test_tori_regimes", strategies.STRATEGIES_DIR / "tori_trades_trendline" / "regimes.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


@pytest.mark.parametrize(("curve", "turns"), [
    ([0, 1, 2, 1.5, 0.5, 1, 3], [0, 2, 4, 6]),          # up, down, up
    ([0, 0.3, -0.2, 0.1], [0, 3]),                       # never a full swing: one leg
    ([0, -0.5, -1.2, 0.5], [0, 2, 3]),                   # down from the first point, then up
    ([0, -0.4, 0.8], [0, 1, 2]),                         # a small dip, then the first swing up
    ([0, 2, 2, 0.5], [0, 2, 3]),                         # a tied high: the later one turns
])
def test_the_zigzag_turns_only_on_a_full_reversal(curve, turns):
    """A turning point is the high (low) before a fall (rise) of at least the threshold; the
    first and the last point always start and end a leg."""
    assert R.pivots(np.asarray(curve, dtype=float), 1.0) == turns


def _trades(returns):
    """A side's trades, one a day, with the columns :func:`regimes.ranges` reads."""
    days = pd.bdate_range("2026-01-05", periods=len(returns))
    return pd.DataFrame({"entry time": days + pd.Timedelta(hours=10),
                         "exit time": days + pd.Timedelta(hours=11),
                         "return": returns, "R": np.sign(returns)})


def test_every_leg_gets_its_trades_kind_and_return():
    """Three 1% wins, a 2.5% and a 1% loss, a 3% win, at a 2% threshold and 3 trades: a
    profitable range of the first three trades, a losing leg of the next two, and a last rise
    too short to count."""
    trades = _trades([0.01, 0.01, 0.01, -0.025, -0.01, 0.03])
    legs = R.ranges(trades, 0.02, 3)
    assert legs["kind"].tolist() == ["profitable", "losing", "too few trades"]
    assert legs[["first", "last", "trades"]].to_numpy().tolist() == [[1, 3, 3], [4, 5, 2],
                                                                     [6, 6, 1]]
    assert legs["return"].tolist() == pytest.approx([1.01 ** 3 - 1, 0.975 * 0.99 - 1, 0.03])
    assert legs.loc[0, "best share"] == pytest.approx(1 / 3)
    assert legs.loc[0, "from"] == trades["entry time"].iloc[0]
    assert legs.loc[0, "to"] == trades["exit time"].iloc[2]
    assert legs["wins"].tolist() == [3, 0, 1]
    assert R.ranges(trades.iloc[:0], 0.02, 3).empty


def test_a_clustered_record_is_unlikely_by_luck_an_alternating_one_likely():
    """Ten wins then ten losses of the same size cluster far more than any deal of the same
    wins and losses (two streaks); win, loss, win, loss ... (twenty streaks) less than every
    deal."""
    clustered = R.luck(np.r_[np.full(10, 0.01), np.full(10, -0.01)], draws=500)
    alternating = R.luck(np.tile([0.01, -0.01], 10), draws=500)
    assert clustered["real"].iloc[2] == 2 and alternating["real"].iloc[2] == 20
    assert (clustered["chance by luck"] < 0.01).all()
    assert (alternating["chance by luck"].iloc[[0, 2]] == 1.0).all()
    assert R.luck(np.full(7, 0.01)).empty                 # too few trades to read


def test_the_trade_log_ranks_by_profit_and_keeps_every_line_of_the_setup():
    """Best first, in five fifths; every trade counts the lines of its setup and finds, among
    them, exactly the one line it bounced off."""
    from tests.test_tori_trades_trendline_bounce import COSTS, market

    daily, minutes = market(sessions=40)
    knobs = R.PRESETS["Her defaults"]["knobs"]
    for side in ("long", "short"):
        result = R.replay_side(daily, minutes, "TEST", side, knobs, COSTS, folder=None)
        log = R.trade_log(result)
        assert len(log) == len(result["trades"]) >= 5
        assert log["return"].is_monotonic_decreasing and log["rank"].tolist()[0] == 1
        assert log["profit fifth"].tolist() == sorted(log["profit fifth"])
        assert set(log["profit fifth"]) == {1, 2, 3, 4, 5}
        every = R.lines_log(result)
        for row in log.itertuples(index=False):
            lines = every[every["trade"] == row.trade]
            assert lines["traded"].sum() == 1
            counts = lines["kind"].value_counts()
            assert counts.get("upward", 0) == row[log.columns.get_loc("upward lines")]
            assert counts.get("downward", 0) == row[log.columns.get_loc("downward lines")]
            mine = lines[lines["traded"]].iloc[0]
            assert mine["kind"] == ("upward" if side == "long" else "downward")
            assert mine["slope"] == row.slope


def test_a_side_is_replayed_from_the_day_after_the_first_minute_and_kept(tmp_path,
                                                                          monkeypatch):
    """The same trades as ``bounce.replay`` with the preset's knobs; the second call reads the
    file instead of replaying."""
    from tests.test_tori_trades_trendline_bounce import COSTS, market

    daily, minutes = market(sessions=40)
    knobs = R.PRESETS["Her defaults"]["knobs"]
    first = R.replay_side(daily, minutes, "TEST", "long", knobs, COSTS, folder=tmp_path)
    plain = R.bounce.replay(daily, minutes, start="2025-01-03", costs=COSTS, sides="long",
                            **R.replay_kwargs(knobs))
    pd.testing.assert_frame_equal(first["trades"], plain["trades"])
    assert len(list(tmp_path.glob("TEST-long-*.pkl"))) == 1

    def refuse(*args, **kwargs):
        raise AssertionError("replayed again instead of reading the file")

    monkeypatch.setattr(R.bounce, "replay", refuse)
    again = R.replay_side(daily, minutes, "TEST", "long", knobs, COSTS, folder=tmp_path)
    pd.testing.assert_frame_equal(again["trades"], first["trades"])
