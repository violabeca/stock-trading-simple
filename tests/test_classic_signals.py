"""classic_signals: the data helpers (candles of any size, daily candles, the data check, report
dates), the indicators (bands, averages, typical move, trend, levels, what was known at a
session's open) and the figures, on hand-made candles (no database, no network). Every indicator
is checked for look-ahead: changing a later candle never changes an earlier value.
"""

import importlib.util
import sys
import types

import numpy as np
import optuna
import pandas as pd
import pytest

from trading_simple import strategies

_DIR = strategies.STRATEGIES_DIR / "classic_signals"
NY = "America/New_York"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"test_cs_{name}", _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DATA, IND, CH = _load("data"), _load("indicators"), _load("charts")


def minutes(n_days=3, seed=0, start="2025-03-03"):
    """Regular-session 1-minute candles: a random walk, 390 per session, New York times."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start, periods=n_days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in sessions
                            for m in range(390)]).tz_localize(NY)
    close = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    open_ = np.r_[100.0, close[:-1]]
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.02,
                          "low": np.minimum(open_, close) - 0.02, "close": close,
                          "volume": rng.integers(0, 1000, len(idx)).astype(float)}, index=idx)
    frame["traded"] = frame["volume"] > 0
    return frame


def minutes_extended(n_days=2, seed=0, start="2025-03-03"):
    """1-minute candles 04:00-19:59 New York: a trade every regular minute, outside regular hours
    only one minute in three has a trade (the others repeat the last price with no volume)."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start, periods=n_days)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=240 + m) for d in sessions
                            for m in range(960)]).tz_localize(NY)
    close = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    open_ = np.r_[100.0, close[:-1]]
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.02,
                          "low": np.minimum(open_, close) - 0.02, "close": close}, index=idx)
    frame["regular"] = DATA.session_part(idx) == 1
    volume = rng.integers(1, 1000, len(idx)).astype(float)
    volume[~frame["regular"].to_numpy() & (rng.random(len(idx)) > 1 / 3)] = 0.0
    frame["volume"], frame["traded"] = volume, volume > 0
    return frame


def days_frame(close, start="2024-01-01"):
    """Daily candles around a close series (high and low half a dollar away)."""
    c = pd.Series(np.asarray(close, float), index=pd.bdate_range(start, periods=len(close)))
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c + 0.5, "low": c - 0.5,
                         "close": c, "volume": 1e6})


def test_candles_of_any_size_never_span_a_night():
    m = minutes(2)
    five = DATA.resample(m, 5)
    assert len(five) == 2 * 78
    first = five.iloc[0]
    block = m.iloc[:5]
    assert (first["open"], first["high"], first["low"], first["close"]) == (
        block["open"].iloc[0], block["high"].max(), block["low"].min(), block["close"].iloc[-1])
    assert five.index[78].strftime("%H:%M") == "09:30"           # day 2 starts afresh
    hour = DATA.resample(m, 60)
    assert hour.index[6].strftime("%H:%M") == "15:30" and len(hour) == 2 * 7
    assert DATA.resample(m, 1) is m


def test_the_parts_of_the_day_come_from_the_clock_and_a_half_day_closes_at_13():
    t = pd.DatetimeIndex(["2025-03-03 08:00", "2025-03-03 09:30", "2025-03-03 15:59",
                          "2025-03-03 16:00", "2025-07-03 12:59", "2025-07-03 13:00"])
    t = t.tz_localize(NY)
    assert DATA.session_part(t).tolist() == [0, 1, 1, 2, 1, 2]


def test_the_overnight_runs_from_20_to_4_and_belongs_to_the_next_open():
    t = pd.DatetimeIndex(["2025-03-03 03:59", "2025-03-03 04:00", "2025-03-03 19:59",
                          "2025-03-03 20:00"]).tz_localize(NY)
    assert DATA.session_part(t).tolist() == [3, 0, 2, 3]
    rng = np.random.default_rng(6)                    # two full 24-hour days of minutes
    idx = pd.date_range("2025-03-03 00:00", "2025-03-04 23:59", freq="min", tz=NY)
    close = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    m = pd.DataFrame({"open": close, "high": close + 0.02, "low": close - 0.02,
                      "close": close, "volume": 10.0}, index=idx)
    m["traded"], m["regular"] = True, DATA.session_part(idx) == 1
    hour = DATA.resample(m.loc["2025-03-03"], 60)
    assert [t.strftime("%H:%M") for t in hour.index] == [
        "00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00",
        "09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30",
        "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"]
    row = IND.before_open(m).loc[pd.Timestamp("2025-03-04")]
    night = m.loc["2025-03-03 16:00":"2025-03-04 09:29"]   # after-hours, overnight, pre-market
    assert row["minutes traded"] == len(night) == 240 + 480 + 330
    assert row["night high"] == night["high"].max()


def test_coarser_candles_never_mix_the_parts_of_the_day():
    hour = DATA.resample(minutes_extended(1), 60)
    assert [t.strftime("%H:%M") for t in hour.index] == [
        "04:00", "05:00", "06:00", "07:00", "08:00", "09:00",           # pre-market (09:00: 30')
        "09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30",  # regular (15:30: 30')
        "16:00", "17:00", "18:00", "19:00"]                             # after-hours
    assert hour["regular"].tolist() == [False] * 6 + [True] * 7 + [False] * 4


def test_the_hours_before_the_open_run_from_the_previous_close_and_read_traded_minutes():
    m = minutes_extended(3)
    pre = IND.before_open(m, usual_days=2)
    day2 = pd.Timestamp("2025-03-04")
    night = m.loc["2025-03-03 16:00":"2025-03-04 09:29"]       # after-hours, then pre-market
    night = night[~night["regular"] & night["traded"]]
    early = night.loc["2025-03-04"]                           # the pre-market alone
    row = pre.loc[day2]
    assert row["previous close"] == m[m["regular"]].loc["2025-03-03", "close"].iloc[-1]
    assert row["last"] == night["close"].iloc[-1]
    assert row["night high"] == night["high"].max() and row["night low"] == night["low"].min()
    assert row["pre-market high"] == early["high"].max()
    assert row["pre-market low"] == early["low"].min() and row["volume"] == night["volume"].sum()
    assert row["move %"] == pytest.approx((row["last"] / row["previous close"] - 1) * 100)
    assert row["volume vs usual"] == pytest.approx(row["volume"] / pre.iloc[0]["volume"])
    assert np.isnan(pre.iloc[0]["previous close"])
    later = m.copy()                                  # what happens after 09:30 changes nothing
    later.loc["2025-03-04 09:30":"2025-03-04 15:59", ["close", "high", "low"]] *= 1.5
    pd.testing.assert_series_equal(IND.before_open(later, usual_days=2).loc[day2], row)


def test_daily_candles_from_minutes():
    m = minutes(2)
    d = DATA.daily_from_minutes(m)
    assert list(d["candles"]) == [390, 390]
    day1, first = m.iloc[:390], d.iloc[0]
    assert first["open"] == day1["open"].iloc[0] and first["close"] == day1["close"].iloc[-1]
    assert first["high"] == day1["high"].max() and first["volume"] == day1["volume"].sum()


def test_a_daily_load_stops_at_its_own_session_and_a_half_day_ends_at_13():
    # a daily candle is stamped 00:00 UTC of its date: "up to June 30" must not reach July 1st's
    assert DATA.last_stamp("2026-06-30", "1d") == pd.Timestamp("2026-06-30")
    assert DATA.last_stamp("2026-06-30", "1m") == pd.Timestamp("2026-07-01")
    kept = DATA.drop_after_half_day_close(minutes(1, start="2025-07-03"))
    assert len(kept) == 210 and kept.index[-1].strftime("%H:%M") == "12:59"
    assert len(DATA.drop_after_half_day_close(minutes(1, start="2025-07-02"))) == 390


def test_the_data_check_names_complete_half_short_and_missing_sessions():
    counts = pd.DataFrame({"AAA": [390, 210, 300, 0], "BBB": [390, 390, 390, 390]},
                          index=pd.bdate_range("2025-07-01", periods=4))
    counts.attrs["traded"] = counts - 10
    check = DATA.data_check(counts).set_index("symbol")
    assert check.loc["AAA", ["complete (390)", "half-days (210)", "short", "missing"]].tolist() \
        == [1, 1, 1, 1]
    assert check.loc["BBB", "complete (390)"] == 4 and check.loc["BBB", "missing"] == 0
    assert check.loc["AAA", "sessions"] == 3
    assert check.loc["BBB", "minutes without a trade"] == pytest.approx(40 / 1560)
    counts.attrs |= {"pre": counts * 0 + 300, "pre_traded": counts * 0 + 100,
                     "post": counts * 0 + 200, "post_traded": counts * 0 + 50}
    check = DATA.data_check(counts).set_index("symbol")
    assert check.loc["BBB", "pre-market minutes per session"] == 300
    assert check.loc["BBB", "pre-market minutes with a trade"] == pytest.approx(1 / 3)
    assert check.loc["BBB", "after-hours minutes with a trade"] == pytest.approx(0.25)


def test_report_dates_tell_before_the_open_from_after_the_close(monkeypatch, tmp_path):
    times = pd.DatetimeIndex(["2025-07-23 16:05", "2025-10-29 07:00"]).tz_localize(NY)

    class Ticker:
        def __init__(self, name):
            self.name = name

        def get_earnings_dates(self, limit):
            return pd.DataFrame({"EPS Estimate": [1.0, 1.1]}, index=times)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=Ticker))
    monkeypatch.setattr(DATA, "VAR", tmp_path)
    out = DATA.earnings_dates(["NOW"], refresh=True)
    assert list(out["when"]) == ["after the close", "before the open"]
    assert list(out["reaction day"].dt.strftime("%Y-%m-%d")) == ["2025-07-24", "2025-10-29"]
    again = DATA.earnings_dates(["NOW"])                          # read back from the file
    assert list(again["when"]) == list(out["when"])


def test_bollinger_bands_match_their_definition_and_never_look_ahead():
    close = pd.Series([10.0, 11, 12, 11, 10, 11, 12, 13, 12, 11])
    b = IND.bollinger(close, length=4, width=2.0)
    window = close.iloc[3:7]
    assert b["middle"].iloc[6] == pytest.approx(window.mean())
    assert b["upper"].iloc[6] == pytest.approx(window.mean() + 2 * window.std(ddof=0))
    assert b["position"].iloc[6] == pytest.approx(
        (close.iloc[6] - b["lower"].iloc[6]) / (b["upper"].iloc[6] - b["lower"].iloc[6]))
    assert b["middle"].iloc[:3].isna().all()
    flat = IND.bollinger(pd.Series([5.0] * 6), length=3)
    assert flat["position"].isna().all()                          # no band, no position
    changed = close.copy()
    changed.iloc[-1] = 99
    pd.testing.assert_frame_equal(IND.bollinger(changed, 4).iloc[:-1], b.iloc[:-1])


def test_the_typical_move_can_leave_the_overnight_jump_out():
    m = minutes(2)
    m.iloc[390, m.columns.get_loc("open")] += 5                   # day 2 opens with a gap
    m.iloc[390, m.columns.get_loc("high")] += 5
    m.iloc[390, m.columns.get_loc("close")] += 5
    m.iloc[390, m.columns.get_loc("low")] += 5
    inside = IND.true_range(m, across_nights=False)
    across = IND.true_range(m, across_nights=True)
    assert inside.iloc[390] == pytest.approx(m["high"].iloc[390] - m["low"].iloc[390])
    assert across.iloc[390] > inside.iloc[390] + 4
    assert IND.typical_move(m, 10, across_nights=False).iloc[:9].isna().all()


def test_the_typical_move_explained_uses_only_closed_candles_and_the_same_numbers():
    m = minutes(3)
    blocks = DATA.resample(m, 15)
    ex = IND.explain_move(blocks, 14)
    tm = IND.typical_move(blocks, 14, across_nights=False)
    pd.testing.assert_series_equal(ex["typical"], tm, check_names=False)
    np.testing.assert_allclose(ex["range"], IND.true_range(blocks, across_nights=False))
    first = blocks.index.get_loc(blocks.loc["2025-03-04"].index[0])   # day 2, 09:30
    assert ex["range"].iloc[first] == pytest.approx(blocks["high"].iloc[first]
                                                    - blocks["low"].iloc[first])  # no night
    # while a candle runs, the strategy knows the PREVIOUS candle's typical move
    assert ex["known_pct"].iloc[20] == pytest.approx(ex["typical_pct"].iloc[19])
    pd.testing.assert_series_equal(               # the same as the minute chart's reading
        ex["known_pct"].iloc[14:], IND.available_at(ex["typical_pct"], 15, blocks.index).iloc[14:],
        check_names=False)
    fig = CH.move_figure("NOW", blocks, ex, size=15, length=14)
    assert [t.type for t in fig.data] == ["candlestick", "bar", "scatter"]
    assert fig.layout.legend.y < 0


def test_the_trend_is_up_down_or_sideways_and_judged_from_the_past_only():
    t = np.arange(120)
    up = IND.trend_state(days_frame(100 + 0.5 * t))["state"]
    down = IND.trend_state(days_frame(200 - 0.5 * t))["state"]
    rng = np.random.default_rng(0)
    flat = IND.trend_state(days_frame(100 + rng.normal(0, 0.05, 120)))["state"]
    assert up.iloc[-1] == 1 and down.iloc[-1] == -1 and flat.iloc[-1] == 0
    assert up.iloc[:59].isna().all()                              # 50-day average + 10 days
    changed = 100 + 0.5 * t
    changed[-1] = 0                                               # a crash on the last day ...
    pd.testing.assert_series_equal(                               # ... changes no earlier day
        IND.trend_state(days_frame(changed))["state"].iloc[:-1], up.iloc[:-1])


def test_a_daily_value_reaches_the_next_session_not_its_own():
    values = pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2025-03-03", periods=3))
    m = minutes(3)
    got = IND.previous_session(values, m.index)
    assert np.isnan(got.iloc[0])                                  # Monday: nothing before
    assert got.iloc[390] == 1.0 and got.iloc[-1] == 2.0           # Tuesday reads Monday ...


def test_a_coarser_candle_is_known_only_once_it_has_closed():
    m = minutes(1)
    blocks = DATA.resample(m, 15)
    values = pd.Series(np.arange(len(blocks), dtype=float), index=blocks.index)
    got = IND.available_at(values, 15, m.index)
    assert got.iloc[:15].isna().all()                             # 09:30-09:44: none closed
    assert got.iloc[15] == 0.0 and got.iloc[29] == 0.0 and got.iloc[30] == 1.0


def test_major_levels_need_several_turning_points_and_only_confirmed_ones():
    rng = np.random.default_rng(1)
    close = 100 + rng.normal(0, 0.3, 100)
    d = days_frame(close)
    for peak in (20, 40, 60):                                     # three tops at 110
        d.iloc[peak, d.columns.get_loc("high")] = 110.0
    levels = IND.major_levels(d, span=5, lookback=120, tolerance=0.5, min_touches=2)
    at = lambda i: levels[(levels["day"] == d.index[i]) & (levels["price"] > 105)]  # noqa: E731
    assert at(45).empty                                           # the 2nd top not yet confirmed
    assert at(46)["touches"].tolist() == [2]                      # confirmed at day 45's close
    assert at(80)["touches"].tolist() == [3] and at(80)["highs"].tolist() == [3]
    short = IND.major_levels(d.iloc[:70], span=5, lookback=120, tolerance=0.5, min_touches=2)
    early = levels[levels["day"] < d.index[70]].reset_index(drop=True)
    pd.testing.assert_frame_equal(short.reset_index(drop=True), early)


def test_the_level_explanation_gives_exactly_the_strategys_levels():
    rng = np.random.default_rng(3)
    d = days_frame(100 + np.cumsum(rng.normal(0, 1, 160)))
    kw = dict(span=4, lookback=60, tolerance=0.6, min_touches=2)
    levels = IND.major_levels(d, **kw)
    for i in (30, 80, 159):
        x = IND.explain_levels(d, d.index[i], **kw)
        expected = levels[levels["day"] == d.index[i]].reset_index(drop=True)
        pd.testing.assert_frame_equal(x["levels"], expected)
        tp = x["turning_points"]
        assert (tp["confirmed"] < d.index[i]).all()                # known before the open
        assert tp["level"].notna().sum() == expected["touches"].sum()
        assert x["known_until"] == d.index[i - 1]


def test_the_switch_keeps_ceilings_and_floors_apart():
    # up to a top at 110 (day 20), down to 100, up through 110 to 120, back down to a bottom at
    # 110.2 (day 55), up again: the old ceiling's price later acts as a floor
    close = np.interp(np.arange(80), [0, 20, 30, 45, 55, 79], [100, 110, 100, 120, 110.2, 125])
    d = days_frame(close)                                         # high = close + 0.5, low - 0.5
    kw = dict(span=5, lookback=120, tolerance=1.0, min_touches=2)
    near = lambda lv: lv[(lv["day"] == d.index[70]) & lv["price"].between(109, 111.5)]  # noqa: E731
    together = near(IND.major_levels(d, **kw))
    assert together[["highs", "lows"]].values.tolist() == [[1, 1]]  # 110.5 and 109.7: one level
    assert near(IND.major_levels(d, mix=False, **kw)).empty         # apart: each one alone
    rnd = days_frame(100 + np.cumsum(np.random.default_rng(5).normal(0, 1, 150)))
    apart = IND.major_levels(rnd, mix=False, **kw)
    assert ((apart["highs"] == 0) | (apart["lows"] == 0)).all()   # never a mix of both
    for i in (60, 149):
        x = IND.explain_levels(rnd, rnd.index[i], mix=False, **kw)
        expected = apart[apart["day"] == rnd.index[i]].reset_index(drop=True)
        pd.testing.assert_frame_equal(x["levels"], expected)


def test_the_trend_reasons_say_why_in_words():
    t = np.arange(90)
    d = days_frame(100 + 0.5 * t)
    trend = IND.trend_state(d)
    why = IND.explain_trend(d, trend, length=50, slope_days=10, flat=0.5)
    assert why.iloc[0].startswith("not known yet") and why.iloc[-1].startswith("UP: the close")
    flat = days_frame(100 + np.random.default_rng(0).normal(0, 0.05, 90))
    ft = IND.trend_state(flat)
    assert IND.explain_trend(flat, ft, length=50, slope_days=10,
                             flat=0.5).iloc[-1].startswith("SIDEWAYS")


def test_the_session_context_reads_the_previous_close():
    d = days_frame([10.0, 11, 12, 13])
    trend = pd.Series([1.0, 0.0, -1.0, np.nan], index=d.index)
    ctx = IND.session_context(d, {"stock": trend}, d.index[1:])
    assert ctx["previous high"].tolist() == [10.5, 11.5, 12.5]
    assert ctx["stock"].tolist() == ["up", "sideways", "down"]


def test_every_figure_builds_and_minute_charts_skip_the_nights():
    m = minutes(3)
    d = days_frame(100 + np.cumsum(np.random.default_rng(2).normal(0, 1, 150)))
    trend = IND.trend_state(d)["state"]
    levels = IND.major_levels(d)
    reports = pd.DataFrame({"symbol": ["NOW"], "time": [pd.Timestamp("2025-03-04 07:00", tz=NY)],
                            "when": ["before the open"],
                            "reaction day": [pd.Timestamp("2025-03-04")]})
    counts = pd.DataFrame({"NOW": [390, 210, 0]}, index=pd.bdate_range("2025-03-03", periods=3))
    figures = []
    figures.append(CH.coverage_figure(counts))
    assert len(figures[-1].data) == 5
    over = CH.overview_figure("NOW", d, averages={"50-day": IND.moving_average(d["close"], 50)},
                              trends={"NOW": trend, "SPY": trend}, levels=levels,
                              turning_points=IND.pivots(d), reports=reports)
    names = {t.name for t in over.data}
    assert {"turning point: swing high", "turning point: swing low", "trend up"} <= names
    figures += [over, CH.conditions_figure({"NOW": trend, "SPY": trend})]
    assert len(figures[-1].data) == 3
    x = IND.explain_levels(d, d.index[-1])
    figures.append(CH.levels_figure("NOW", d, x))
    assert any(t.name == "major level" for t in figures[-1].data) == (len(x["levels"]) > 0)
    full = IND.trend_state(d)
    figures.append(CH.trend_figure("NOW", d, full, IND.explain_trend(
        d, full, length=50, slope_days=10, flat=0.5), length=50, slope_days=10, flat=0.5))
    bands = IND.bollinger(m["close"], 20)
    lv = levels.assign(day=pd.Timestamp("2025-03-04"))
    ctx = IND.session_context(days_frame([99.0, 100, 101], start="2025-02-28"), {"NOW": trend},
                              ["2025-03-03", "2025-03-04", "2025-03-05"])
    fig = CH.day_figure("NOW", m, bands=bands, average=IND.moving_average(m["close"], 30),
                        average_label="30-candle average", others={"IGV": m["close"] * 1.01},
                        move_pct=IND.typical_move(m, 14, across_nights=False) / m["close"] * 100,
                        context=ctx, levels=lv, reports=reports)
    breaks = fig.layout.xaxis.rangebreaks
    assert len(breaks) == 2                                       # two nights between 3 days
    assert fig.data[0].x[0] == "2025-03-03 09:30"                 # New York wall-clock text
    previous = [t for t in fig.data if t.legendgroup == "previous"]
    assert [t.name for t in previous if t.showlegend] == ["previous session's high and low"]
    ext = minutes_extended(2)                                     # with the hours around them
    pre = IND.before_open(ext)
    ctx2 = IND.session_context(days_frame([99.0, 100, 101], start="2025-02-28"), {"NOW": trend},
                               ["2025-03-03", "2025-03-04"])
    ctx2["pre-market high"] = pre["pre-market high"].to_numpy()
    ctx2["pre-market low"] = pre["pre-market low"].to_numpy()
    fig_ext = CH.day_figure("NOW", ext, bands=IND.bollinger(ext["close"], 20),
                            average=IND.moving_average(ext["close"], 30), average_label="30",
                            others={}, move_pct=ext["close"] * 0, context=ctx2, levels=lv,
                            reports=reports)
    names = {t.name for t in fig_ext.data}
    assert {"1-minute candles, regular hours", "candles outside regular hours",
            "regular session: open and close", "pre-market high and low"} <= names
    figures += [fig_ext, CH.before_open_figure("NOW", pre)]
    figures += [fig, CH.band_position_figure({"NOW": bands["position"]})]
    assert len(figures[-1].data) == 1
    for f in figures:                                             # legends never cover a title
        assert f.layout.legend.y < 0 and f.layout.legend.yanchor == "top"


# --- step 2: the rules, the switchboard, the simulator, the results ---------------------------
COMBINE, SIM, RES = _load("combine"), _load("simulate"), _load("results")
NAMES = {"stock": "NOW", "sector": "IGV", "market": "SPY"}


def candles_from(rows, *, start="2025-03-03 09:30", times=None):
    """Hand-made candles from (open, high, low, close) rows, one minute apart (or at ``times``)."""
    idx = pd.DatetimeIndex(times).tz_localize(NY) if times is not None else pd.date_range(
        start, periods=len(rows), freq="min", tz=NY)
    o, h, lo, c = np.asarray(rows, float).T
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 100.0}, index=idx)


def context(candles, *, upper=101.0, lower=99.0, sector=0.0, stock=0.0, market=0.0, move=1.0,
            average=None, levels=None):
    """A context with flat bands and trends, and a constant typical move."""
    idx = candles.index
    bands = pd.DataFrame({"upper": upper, "lower": lower, "middle": (upper + lower) / 2},
                         index=idx)
    trend = lambda v: pd.Series(v, index=idx, dtype=float)  # noqa: E731
    return COMBINE.make_context(NAMES, candles, bands, pd.Series(move, index=idx),
                                {"stock": trend(stock), "sector": trend(sector),
                                 "market": trend(market)}, average, levels)


def test_the_band_triggers_fire_on_exactly_the_right_candle_and_never_look_ahead():
    rules = COMBINE.load_rules()
    close = [100, 98.5, 99.5, 100, 101.5, 100.5]
    c = candles_from([(x, x, x, x) for x in close])
    ctx = context(c)
    fade, breakout = rules["band fade"], rules["band breakout"]
    one = {"breakout_closes": 1}                                     # the first close outside
    assert fade.signal(ctx, {}).tolist() == [0, 0, 1, 0, 0, -1]      # back inside: long, short
    assert breakout.signal(ctx, one).tolist() == [0, -1, 0, 0, 1, 0]
    assert breakout.signal(ctx, {}).tolist() == [0] * 6               # default: 2 in a row
    assert "back inside" in fade.explain(ctx, {}, 2)
    assert "broke above" in breakout.explain(ctx, one, 4)
    later = candles_from([(x, x, x, x) for x in [*close[:5], 50.0]])  # a crash after candle 4
    for rule, params in ((fade, {}), (breakout, one), (breakout, {})):
        before = rule.signal(ctx, params).iloc[:5].tolist()
        assert rule.signal(context(later), params).iloc[:5].tolist() == before


def test_the_breakout_waits_for_its_closes_in_a_row_and_fires_once_per_stretch():
    rule = COMBINE.load_rules()["band breakout"]
    # one close above that turns straight back, three above in a row, inside, three below
    close = [100, 101.5, 100.5, 101.5, 102, 102.5, 100, 98.5, 98, 97.5]
    ctx = context(candles_from([(x, x, x, x) for x in close]))
    said = {n: rule.signal(ctx, {"breakout_closes": n}).tolist() for n in (1, 2, 3, 4)}
    assert said[1] == [0, 1, 0, 1, 0, 0, 0, -1, 0, 0]
    assert said[2] == [0, 0, 0, 0, 1, 0, 0, 0, -1, 0]     # the lone close at 09:31: no breakout
    assert said[3] == [0, 0, 0, 0, 0, 1, 0, 0, 0, -1]
    assert said[4] == [0] * 10
    assert rule.signal(ctx, {}).tolist() == said[2]
    words = rule.explain(ctx, {"breakout_closes": 2}, 4)
    assert "2 closes in a row above the upper band" in words and "at 09:32, was inside" in words
    assert rule.explain(ctx, {"breakout_closes": 2}, 5) == "band breakout: nothing"
    assert rule.explain(ctx, {"breakout_closes": 3}, 9).startswith("band breakout, short: 3 closes")


def test_every_rule_knob_is_declared_once_with_a_default_its_slider_can_show():
    rules = COMBINE.load_rules()
    names = [k for r in rules.values() for k in r.PARAMS]
    assert len(names) == len(set(names))                  # one dict carries every rule's knobs
    for r in rules.values():
        for p in r.PARAMS.values():
            assert {"label", "value", "start", "stop", "step"} <= set(p)
            assert p["start"] <= p["value"] <= p["stop"] and p["step"] > 0
            assert round((p["value"] - p["start"]) / p["step"], 9).is_integer()


def test_the_sector_filter_allows_the_side_of_its_trend():
    c = candles_from([(100, 100, 100, 100)] * 4)
    ctx = context(c)
    ctx["trend"]["sector"] = pd.Series([1.0, -1.0, 0.0, np.nan], index=c.index)
    allowed = COMBINE.load_rules()["sector trend"].allow(ctx, {})
    assert allowed["long"].tolist() == [True, False, True, True]
    assert allowed["short"].tolist() == [False, True, True, True]


def test_the_switchboard_needs_a_clear_proposal_and_every_filter():
    from types import SimpleNamespace

    c = candles_from([(100, 100, 100, 100)] * 4)
    ctx = context(c)
    s = lambda v: pd.Series(v, index=c.index)  # noqa: E731
    rules = {
        "a": SimpleNamespace(KIND="trigger", signal=lambda ctx, p: s([0, 1, 1, -1])),
        "b": SimpleNamespace(KIND="trigger", signal=lambda ctx, p: s([0, 0, -1, -1])),
        "f": SimpleNamespace(KIND="filter", allow=lambda ctx, p: pd.DataFrame(
            {"long": [True, True, True, True], "short": [True, True, True, False]},
            index=c.index))}
    off = COMBINE.decide(ctx, rules, {"a": True, "b": True, "f": False})
    assert off["signal"].tolist() == [0, 1, 0, -1] and off["conflict"].tolist()[2]
    on = COMBINE.decide(ctx, rules, {"a": True, "b": True, "f": True})
    assert on["signal"].tolist() == [0, 1, 0, 0] and on["blocked"].tolist() == [0, 0, 0, 1]
    assert ((on["signal"] == 0) | (on["signal"] == off["signal"])).all()  # a filter only removes
    assert COMBINE.decide(ctx, rules, {"a": True})["signal"].tolist() == [0, 1, 1, -1]


def run_once(rows, signal, *, times=None, move=1.0, **kw):
    """Simulate hand-made candles with a hand-made signal, no costs unless asked."""
    c = candles_from(rows, times=times)
    ctx = context(c, move=move)
    decisions = pd.DataFrame({"signal": np.asarray(signal, np.int8)}, index=c.index)
    kw = {"stop_moves": 1.0, "slippage": 0.0, "commission": 0.0, **kw}
    return SIM.run(ctx, decisions, **kw)


def test_a_long_trails_its_stop_only_upwards_and_exits_when_a_candle_trades_through_it():
    sim = run_once([(100, 100, 100, 100),          # the signal (a close)
                    (100, 101, 99.8, 100.8),        # entry at 100, stop 99; trails to 100
                    (100.9, 101.5, 100.5, 101),     # stop 100 holds; trails to 100.5
                    (101, 101.1, 100.2, 100.4),     # the low trades through 100.5: out at 100.5
                    (100.4, 100.6, 100.3, 100.5)], [1, 0, 0, 0, 0])
    t = sim["trades"]
    assert len(t) == 1 and t.iloc[0][["side", "entry price", "exit price"]].tolist() == [
        "long", 100.0, 100.5]
    assert t.iloc[0]["exit reason"] == SIM.REASONS[0]
    np.testing.assert_allclose(sim["stop"].to_numpy(), [np.nan, 99.0, 100.0, 100.5, np.nan])
    assert sim["position"].tolist() == [0, 1, 1, 1, 0]


def test_a_gap_through_the_stop_fills_at_the_open_and_a_short_mirrors():
    times = ["2025-03-03 15:58", "2025-03-03 15:59", "2025-03-04 09:30", "2025-03-04 09:31"]
    gap = run_once([(100, 100, 100, 100), (100, 100.2, 99.9, 100),
                    (97, 97.5, 96.5, 97), (97, 97, 97, 97)], [1, 0, 0, 0], times=times)
    row = gap["trades"].iloc[0]
    assert row["exit price"] == 97.0 and row["exit reason"] == SIM.REASONS[1]
    assert row["sessions"] == 2
    short = run_once([(100, 100, 100, 100), (100, 100.2, 99.0, 99.2),   # entry 100, stop 101
                      (99.2, 99.4, 98.5, 98.6),                          # stop trails to 99.5
                      (98.6, 99.6, 98.6, 99.5)], [-1, 0, 0, 0])          # high 99.6: out at 99.5
    row = short["trades"].iloc[0]
    assert row["side"] == "short" and row["exit price"] == 99.5
    assert row["return %"] == pytest.approx((100 - 99.5) / 100 * 100)


def test_daily_move_stops_fixed_or_trailing_a_daily_move_target_and_risk_sizing():
    rows = [(100, 100, 100, 100),           # the signal
            (100, 101, 99.5, 100.8),        # entry at 100: one daily move ($2) away, 98
            (100.8, 103, 100.5, 102.5),     # trailing: $2 behind 101, then behind 103
            (102.5, 102.6, 100.7, 101.0)]   # the low 100.7 trades through 101
    c = candles_from(rows)
    ctx = context(c)
    ctx["daily_move"] = pd.Series(2.0, index=c.index)          # a typical day moves $2

    def run(side, **kw):
        decisions = pd.DataFrame({"signal": np.array([side, 0, 0, 0], np.int8)}, index=c.index)
        return SIM.run(ctx, decisions, stop_moves=1.0, slippage=0.0, commission=0.0, **kw)

    fixed = run(1, stop_kind="daily fixed", stop_days=1.0)
    np.testing.assert_allclose(fixed["stop"].to_numpy()[1:], [98.0, 98.0, 98.0])
    assert fixed["trades"].iloc[0]["exit reason"] == SIM.REASONS[3]       # never hit
    trail = run(1, stop_kind="daily trailing", stop_days=1.0)
    np.testing.assert_allclose(trail["stop"].to_numpy()[1:], [98.0, 99.0, 101.0])
    assert trail["trades"].iloc[0]["exit price"] == 101.0
    assert run(-1, stop_kind="daily fixed", stop_days=1.0)["trades"].iloc[0]["exit price"] == 102
    target = PIPE.targets(ctx, None, None, "daily moves", 1.25)          # $2.50 from the entry
    hit = run(1, stop_kind="daily fixed", stop_days=1.0, targets=target)["trades"].iloc[0]
    assert hit["exit price"] == pytest.approx(102.5) and hit["exit reason"] == SIM.REASONS[4]
    sized = run(1, stop_kind="daily fixed", stop_days=1.0, risk=100.0, notional=20_000.0)
    t = sized["trades"].iloc[0]                       # $100 at risk over a $2 stop: 50 shares
    assert t["shares"] == 50 and t["P&L $"] == pytest.approx(50 * (101 - 100))
    assert t["return %"] == pytest.approx(1.0)                         # of the position's value
    tight = run(1, stop_kind="daily fixed", stop_days=0.1, risk=100.0, notional=20_000.0)
    assert tight["trades"].iloc[0]["shares"] == 200                    # capped at $20,000 worth


def test_a_holding_limit_closes_at_the_session_close_and_costs_are_charged():
    times = ["2025-03-03 15:57", "2025-03-03 15:58", "2025-03-03 15:59", "2025-03-04 09:30"]
    rows = [(100, 100, 100, 100), (100, 100.5, 99.9, 100.4), (100.4, 100.6, 100.3, 100.5),
            (101, 101, 101, 101)]
    one = run_once(rows, [1, 0, 0, 0], times=times, max_sessions=1, slippage=0.001,
                   commission=1.0, notional=10_000.0)
    row = one["trades"].iloc[0]
    assert row["exit reason"] == SIM.REASONS[2] and row["exit time"].strftime("%H:%M") == "15:59"
    fill_in, fill_out = 100 * 1.001, 100.5 * 0.999
    assert row["P&L $"] == pytest.approx(10_000 / fill_in * (fill_out - fill_in) - 2.0)
    open_end = run_once(rows, [1, 0, 0, 0], times=times)
    assert open_end["trades"].iloc[0]["exit reason"] == SIM.REASONS[3]  # still open at the end


def test_results_summaries_curve_and_the_story_of_a_trade():
    rules = COMBINE.load_rules()
    close = [100, 98.5, 99.5, 100, 100.5, 101, 101.2, 99.0, 98.0, 97.5]
    c = candles_from([(x, x + 0.1, x - 0.1, x) for x in close])
    ctx = context(c, sector=1.0, stock=1.0, move=0.5)
    switches = {"band fade": True, "band breakout": False, "sector trend": True}
    decisions = COMBINE.decide(ctx, rules, switches)
    sim = SIM.run(ctx, decisions, stop_moves=1.0, slippage=0.0, commission=0.0)
    trades = sim["trades"]
    assert len(trades) >= 1 and trades.iloc[0]["side"] == "long"
    assert trades.iloc[0]["condition"] == "up"
    assert RES.triggers(trades, decisions, rules, switches).iloc[0] == "band fade"
    story = RES.explain_trade(trades.iloc[0], ctx, decisions, rules, switches, {},
                              exit_={"stop_kind": "moves", "stop_moves": 1.0, "stop_days": 1.0})
    assert "band fade, long" in story and "IGV was up" in story and "Result" in story
    assert "1 typical moves" in story and "trailing" in story
    summ = RES.summary(trades, ["stock", "side"])
    assert summ.iloc[0]["trades"] == len(trades)
    curve = RES.equity(trades)
    assert curve["all stocks"].iloc[-1] == pytest.approx(trades["P&L $"].sum())
    dd = RES.drawdown(pd.Series([0.0, 10.0, 4.0, 12.0, 1.0]))
    assert dd.tolist() == [0.0, 0.0, -6.0, 0.0, -11.0]
    figs = [CH.switchboard_figure("NOW", c, ctx["bands"], decisions, sim, rules, switches),
            CH.results_figure(RES.summary(trades, ["stock", "condition", "side"])),
            CH.equity_figure(curve),
            CH.trade_figure(trades.iloc[0], c, ctx["bands"], sim["stop"])]
    assert all(f.layout.legend.y < 0 for f in figs)
    strips = next(t for t in figs[0].data if t.type == "heatmap")
    assert "band breakout (trigger, off)" in strips.y and "decision" in strips.y


def test_a_signal_on_a_sessions_last_candle_is_dropped_not_filled_after_the_night():
    times = ["2025-03-03 15:58", "2025-03-03 15:59", "2025-03-04 09:30", "2025-03-04 09:31"]
    rows = [(100, 100, 100, 100)] * 4
    stale = run_once(rows, [0, 1, 0, 0], times=times)
    assert stale["trades"].empty and (stale["position"] == 0).all()
    fresh = run_once(rows, [0, 0, 1, 0], times=times)      # the morning's own signal trades
    assert fresh["trades"].iloc[0]["entry time"].strftime("%H:%M") == "09:31"


def test_the_bottom_line_scores_profit_against_the_worst_drawdown_and_holding_is_its_reference():
    exits = pd.DatetimeIndex(["2025-03-03 10:00", "2025-03-03 11:00", "2025-03-03 12:00",
                              "2025-03-03 13:00"]).tz_localize(NY)
    trades = pd.DataFrame({"stock": ["NOW", "NOW", "GEV", "NOW"], "exit time": exits,
                           "P&L $": [100.0, -150.0, 30.0, 200.0]})
    sims = {"NOW": {"position": pd.Series([0, 1, 1, 0])},
            "GEV": {"position": pd.Series([0, 0, 0, 1])}}
    line = RES.bottom_line(trades, sims).set_index("stock")
    assert line.loc["NOW", ["total P&L $", "worst drawdown $"]].tolist() == [150.0, -150.0]
    assert line.loc["NOW", "profit ÷ worst drawdown"] == pytest.approx(1.0)
    assert line.loc["NOW", "in a trade, share of the time"] == 0.5
    assert np.isnan(line.loc["GEV", "profit ÷ worst drawdown"])     # never below its best
    assert line.loc["all stocks", ["trades", "total P&L $", "worst drawdown $"]].tolist() == [
        4, 180.0, -150.0]
    c = candles_from([(100, 101, 99, 100), (100, 100, 90, 90), (90, 120, 90, 110)],
                     times=["2025-03-03 15:59", "2025-03-04 15:59", "2025-03-05 15:59"])
    held = RES.holding("NOW", c, notional=1000.0, commission=0.0, slippage=0.0)
    assert held["total P&L $"] == pytest.approx(100.0)
    assert held["worst drawdown $"] == pytest.approx(-100.0)


def test_the_pictures_draw_each_trades_own_stop_and_only_the_marks_inside_the_window():
    rows = [(100, 100, 100, 100), (100, 101, 99.8, 100.8), (100.9, 101.5, 100.5, 101),
            (101, 101.1, 100.2, 100.4), (100.4, 100.6, 100.3, 100.5),
            (100.5, 100.6, 100.4, 100.5)]
    c = candles_from(rows)
    ctx, rules = context(c), COMBINE.load_rules()
    switches = {n: True for n in rules}
    decisions = COMBINE.decide(ctx, rules, switches)
    decisions["signal"] = np.array([1, 0, 0, 1, 0, 0], np.int8)  # two back-to-back longs
    sim = SIM.run(ctx, decisions, stop_moves=1.0, slippage=0.0, commission=0.0)
    trades = sim["trades"]
    assert trades["entry time"].tolist() == [c.index[1], c.index[4]]
    second = CH.trade_figure(trades.iloc[1], c, ctx["bands"], sim["stop"])
    stop = next(t for t in second.data if t.name == "trailing stop")
    assert [v for v in stop.y if v is not None] == [99.4, 99.6]          # not the first's
    both = CH.trade_figure(trades.iloc[0], c, ctx["bands"], sim["stop"])
    assert list(next(t for t in both.data if t.name == "trailing stop").y) == [
        99.0, 100.0, 100.5, None]
    window = CH.switchboard_figure("NOW", c.iloc[2:], ctx["bands"].iloc[2:],
                                   decisions.iloc[2:], sim, rules, switches)
    named = {t.name: t for t in window.data}
    assert list(named["long entry"].x) == list(CH._wall([c.index[4]]))  # the first entered before
    assert list(named["exit"].x) == list(CH._wall(c.index[[3, 5]]))
    low, high = window.layout.yaxis.range
    assert low < 99.6 and high > 101.1


def test_the_wait_after_the_open_blocks_every_fill_in_its_first_minutes_and_nothing_else():
    from types import SimpleNamespace

    rules = COMBINE.load_rules()
    rule = rules["wait after the open"]
    c = candles_from([(100, 100, 100, 100)] * 21)                       # 09:30 .. 09:50
    ctx, wait = context(c), {"open_wait_minutes": 15}
    allowed = rule.allow(ctx, wait)
    assert allowed["long"].tolist() == [False] * 14 + [True] * 7        # 09:44's close: fill 09:45
    assert allowed["long"].equals(allowed["short"])
    assert rule.allow(ctx, {"open_wait_minutes": 0})["long"].all()
    five = candles_from([(100, 100, 100, 100)] * 4, times=[
        "2025-03-03 09:30", "2025-03-03 09:35", "2025-03-03 09:40", "2025-03-03 09:45"])
    assert rule.allow(context(five), {})["short"].tolist() == [False, False, True, True]
    assert rule.explain(ctx, wait, 0).endswith("1 minute after the open, inside the first 15: "
                                               "no new trade")
    assert "fills at 09:45, 15 minutes after the open" in rule.explain(ctx, wait, 14)
    signal = pd.Series(0, index=c.index)
    signal.iloc[[1, 14]] = 1                                            # fills 09:32 and 09:45
    rules = {**rules, "test": SimpleNamespace(KIND="trigger", signal=lambda ctx, p: signal)}
    on = {"test": True, "wait after the open": True}
    decisions = COMBINE.decide(ctx, rules, on, wait)
    assert decisions["blocked"].iloc[1] and decisions["signal"].iloc[14] == 1
    trades = SIM.run(ctx, decisions, stop_moves=1.0)["trades"]
    assert trades["entry time"].dt.strftime("%H:%M").tolist() == ["09:45"]


def test_the_daily_trend_filters_allow_the_side_of_their_own_trend():
    rules = COMBINE.load_rules()
    c = candles_from([(100, 100, 100, 100)] * 4)
    states = [1.0, -1.0, 0.0, np.nan]
    for name, role, symbol in (("stock trend", "stock", "NOW"), ("market trend", "market", "SPY")):
        ctx = context(c)
        ctx["trend"][role] = pd.Series(states, index=c.index)
        allowed = rules[name].allow(ctx, {})
        assert allowed["long"].tolist() == [True, False, True, True]
        assert allowed["short"].tolist() == [False, True, True, True]
        assert rules[name].explain(ctx, {}, 0) == (f"{name}: {symbol} was up at the previous "
                                                   "close, longs only")
        assert "not known yet" in rules[name].explain(ctx, {}, 3)


def test_the_moving_average_filter_follows_a_rising_or_falling_average_and_allows_both_if_flat():
    rule = COMBINE.load_rules()["moving average"]
    close = [100, 101, 102, 103, 104, 102.0, 104, 103, 102, 101, 100]
    average = [100, 100.5, 101, 101.5, 102, 102.5, 102.6, 102.6, 102.5, 102.0, 101.4]
    c = candles_from([(x, x, x, x) for x in close])
    ctx = context(c, average=pd.Series(average, index=c.index))
    knobs = {"ma_rise_candles": 2, "ma_flat_moves": 0.5}               # flat zone ±0.5
    state = rule.trend(ctx, knobs)["state"].tolist()
    assert np.isnan(state[:2]).all()                                    # rise not known yet
    assert state[2:] == [1, 1, 1, 0, 1, 0, 0, -1, -1]                  # 09:35: close below a rise
    allowed = rule.allow(ctx, knobs)
    assert allowed["long"].tolist() == [True] * 9 + [False] * 2
    assert allowed["short"].tolist() == [True, True, False, False, False, True, False, True, True,
                                         True, True]
    assert not rule.allow(ctx, {**knobs, "ma_flat_moves": 0.0})["short"].iloc[7]
    assert rule.explain(ctx, knobs, 9) == (
        "moving average: the close 101.00 is below its average 102.00, which moved -0.60 over 2 "
        "candles (flat zone ±0.50): down, shorts only")
    cut = context(c.iloc[:8], average=pd.Series(average[:8], index=c.index[:8]))
    assert rule.allow(cut, knobs).equals(allowed.iloc[:8])              # no later candle read
    assert rule.allow(context(c), knobs).all().all()                    # no average: both sides


def test_the_new_filters_start_switched_off_and_the_rest_on():
    rules = COMBINE.load_rules()
    starts = {name: getattr(rule, "ON", True) for name, rule in rules.items()}
    assert [n for n, on in starts.items() if not on] == [
        "moving average", "stock trend", "market trend", "major levels", "previous high / low",
        "options calm", "iv term", "put/call", "dealer gamma", "option walls"]
    assert list(rules)[:2] == ["band fade", "band breakout"]           # triggers first


def test_the_options_expected_move_comes_from_the_nearest_straddle_and_blocks_on_events(
        tmp_path):
    row = lambda day, dte, side, strike, bid, ask: {  # noqa: E731
        "date": day, "expiration": dte, "side": side, "strike": strike, "dte": dte, "bid": bid,
        "ask": ask, "underlyingPrice": 100.4, "openInterest": 10, "volume": 5}
    chains = pd.DataFrame([
        row("2026-03-02", 1, "call", 100.0, 1.0, 1.2), row("2026-03-02", 1, "put", 100.0, 0.8, 1.0),
        *[row("2026-03-02", 7, side, k, b, b + 0.2) for side, k, b in (
            ("call", 95.0, 6.0), ("call", 100.0, 2.9), ("call", 105.0, 0.8),
            ("put", 95.0, 0.6), ("put", 100.0, 2.5), ("put", 105.0, 4.8))],
        row("2026-03-03", 7, "call", 100.0, 2.9, 3.1),
        row("2026-03-03", 7, "put", 100.0, 0.0, 2.6)])
    chains["date"] = pd.to_datetime(chains["date"])
    move = IND.implied_move(chains)                  # the 7-day expiry, strike 100; 03-03: no bid
    straddle, days_left = (2.9 + 3.1) / 2 + (2.5 + 2.7) / 2, 7 * 252 / 365
    assert list(move.index) == [pd.Timestamp("2026-03-02")]
    assert move.iloc[0] == pytest.approx(straddle / (0.798 * 100.4 * np.sqrt(days_left)) * 100)
    days = pd.DataFrame({"close": 100 * np.exp(np.cumsum(np.tile([0.02, -0.02], 15)))},
                        index=pd.bdate_range(end="2026-03-02", periods=30))
    ratio = IND.options_ratio(move, days)            # ÷ the last 20 daily moves (2% each)
    assert ratio.iloc[0] == pytest.approx(move.iloc[0] / (np.std([2.0, -2.0] * 10, ddof=1)))
    rule = COMBINE.load_rules()["options calm"]
    c = candles_from([(100, 100, 100, 100)] * 3)
    ctx = context(c)
    ctx["options"] = pd.DataFrame({"ratio": [np.nan, 1.2, 2.5]}, index=c.index)
    allowed = rule.allow(ctx, {"options_max_ratio": 1.8})
    assert allowed["long"].tolist() == allowed["short"].tolist() == [True, True, False]
    assert "no option data" in rule.explain(ctx, {}, 0) and "no new trade" in rule.explain(
        ctx, {"options_max_ratio": 1.8}, 2)
    folder = tmp_path / "chains" / "NOW"
    folder.mkdir(parents=True)
    chains.assign(date=chains["date"].dt.strftime("%Y-%m-%d")).to_csv(
        folder / "2026-03-01_2026-03-31.csv.gz", index=False)
    loaded = DATA.load_chains("NOW", "2026-03-03", "2026-03-31", folder=tmp_path / "chains")
    assert set(loaded["date"]) == {pd.Timestamp("2026-03-03")}       # cut to the dates asked
    assert DATA.load_chains("GEV", "2026-03-01", "2026-03-31", folder=tmp_path / "chains").empty


def synthetic_chain(day="2026-03-02", price=100.0, *, call_oi=10, put_oi=10):
    """One day's chain priced by Black-Scholes, strikes from 80% to 120% of the price: a front
    expiry (3 days, 45% volatility at the money) and a 30-day one (30%), the volatility rising
    below the price (puts richer); the most call open interest at 110%, the most put open
    interest at 90%; three times as many puts traded as calls."""
    rows = []
    for dte, base in ((3, 0.45), (30, 0.30)):
        for m in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2):
            k, iv = round(price * m, 6), base + 0.2 * (1 - m)
            for call in (True, False):
                v = float(IND.black_scholes(price, k, dte / 365, iv, call)["price"])
                rows.append({"date": pd.Timestamp(day), "expiration": dte, "side":
                             "call" if call else "put", "strike": float(k), "dte": dte,
                             "bid": v - 0.01, "ask": v + 0.01, "underlyingPrice": price,
                             "openInterest": (call_oi * (5 if m == 1.1 else 1)) if call
                             else (put_oi * (5 if m == 0.9 else 1)),
                             "volume": 10 if call else 30})
    return pd.DataFrame(rows)


def test_black_scholes_greeks_and_implied_volatility_hold_together():
    call = IND.black_scholes(100.0, 100.0, 0.25, 0.30, True)
    put = IND.black_scholes(100.0, 100.0, 0.25, 0.30, False)
    assert call["price"] - put["price"] == pytest.approx(100 - 100 * np.exp(-IND.RATE * 0.25))
    assert call["delta"] - put["delta"] == pytest.approx(1.0)            # parity, in deltas
    assert call["gamma"] == pytest.approx(put["gamma"]) and call["vega"] > 0 > call["theta"]
    bump = IND.black_scholes(101.0, 100.0, 0.25, 0.30, True)["price"] - call["price"]
    assert bump == pytest.approx(call["delta"] + call["gamma"] / 2, rel=1e-3)
    prices = np.array([call["price"], put["price"], 0.0])
    iv = IND.implied_vol(prices, 100.0, 100.0, 0.25, np.array([True, False, True]))
    np.testing.assert_allclose(iv[:2], 0.30, atol=1e-9)
    assert np.isnan(iv[2])                                     # no volatility gives a price of 0


def test_option_features_read_term_skew_walls_put_call_and_dealer_gamma_from_a_chain():
    f = IND.option_features(synthetic_chain()).iloc[0]
    assert f["iv"] == pytest.approx(30.0, abs=0.01) and f["front iv"] == pytest.approx(45.0,
                                                                                        abs=0.01)
    assert f["term"] == pytest.approx(1.5, abs=0.001)          # an event priced in front
    assert f["skew"] > 0                                        # puts richer than calls
    assert (f["call wall"], f["put wall"]) == (110.0, 90.0)
    assert f["put/call volume"] == pytest.approx(3.0) and f["price"] == 100.0
    no_puts = synthetic_chain().assign(volume=lambda c: np.where(c["side"] == "put", 0, 10))
    assert np.isnan(IND.option_features(no_puts).iloc[0]["put/call volume"])   # no ratio
    assert IND.option_features(synthetic_chain(call_oi=30)).iloc[0]["gamma"] > 0 > \
        IND.option_features(synthetic_chain(put_oi=30)).iloc[0]["gamma"]


def test_the_new_option_rules_block_on_their_readings_and_allow_without_data():
    rules = COMBINE.load_rules()
    c = candles_from([(100, 100, 100, 100)] * 4)
    ctx = context(c)
    ctx["options"] = pd.DataFrame({"ratio": np.nan, "term": [np.nan, 1.1, 1.6, 1.2],
                                   "put/call": [np.nan, np.log(3), -np.log(3), 0.1],
                                   "gamma": [np.nan, -0.5, 0.4, 0.0]}, index=c.index)
    term = rules["iv term"].allow(ctx, {"iv_term_max": 1.3})
    assert term["long"].tolist() == term["short"].tolist() == [True, True, False, True]
    follow = rules["put/call"].allow(ctx, {"pc_limit": 2.0, "pc_side": 1})   # many puts: short
    assert follow["long"].tolist() == [True, False, True, True]
    assert follow["short"].tolist() == [True, True, False, True]
    against = rules["put/call"].allow(ctx, {"pc_limit": 2.0, "pc_side": -1})
    assert against["long"].tolist() == [True, True, False, True]
    gamma = rules["dealer gamma"].allow(ctx, {"gamma_min": 0.0})
    assert gamma["long"].tolist() == [True, False, True, True]
    assert all("no option data" in rules[n].explain(ctx, {}, 0)
               for n in ("iv term", "put/call", "dealer gamma"))


def test_the_lab_reads_the_option_rules_known_at_the_open_and_the_walls_after_a_split():
    lab = synthetic_lab()
    days = lab.days["NOW"]
    dates = days.index[-45:]
    split = [5 if i < 20 else 1 for i in range(len(dates))]   # 20 days before a 5:1 split
    chain = pd.concat([synthetic_chain(d, price=float(days.loc[d, "close"]) * f)
                       for d, f in zip(dates, split, strict=True)])
    opts = IND.option_features(chain)
    walls = PIPE.walls(opts, days)
    first = walls[walls["day"] == dates[1]]["price"].sort_values().to_numpy()
    close = float(days.loc[dates[0], "close"])                # in the candles' (split) prices
    np.testing.assert_allclose(first, [0.9 * close, 1.1 * close], rtol=1e-9)
    with_options = TU.Lab(lab.names, lab.minutes, lab.days, lab.reports, lab.rules, lab.costs,
                          options=opts)
    s = HP.defaults(lab.rules)
    ctx = with_options.context(s)
    known = PIPE.option_context(opts, days, ctx["candles"].index)
    pd.testing.assert_frame_equal(ctx["options"], known[list(COMBINE.OPTIONS)])
    assert ctx["options"]["term"].notna().any() and lab.context(s)["options"].isna().all().all()
    assert "walls" in ctx["levels"] and "walls" not in lab.context(s)["levels"]
    blocked = {**s, "on: iv term": True, "iv_term_max": 1.0}       # every chain has term 1.5
    assert len(with_options.run(blocked)) < len(with_options.run(s))
    target = with_options.targets({**s, "target_kind": "option walls"})
    assert target is not None and target["long"].notna().any()
    assert lab.targets({**s, "target_kind": "option walls"}) is None      # no option data


def test_the_switch_comparison_reads_every_rule_off_and_on_and_where_the_change_comes_from():
    def trades(rows):
        return pd.DataFrame(rows, columns=["stock", "exit time", "condition", "P&L $"]).assign(
            **{"exit time": lambda t: pd.to_datetime(t["exit time"]).dt.tz_localize(NY)})

    base = trades([("NOW", "2025-03-03 10:00", "up", 100.0),
                   ("NOW", "2025-08-04 10:00", "down", -50.0),
                   ("GEV", "2026-02-02 10:00", "up", 30.0)])
    a_off = trades([("NOW", "2025-03-03 10:00", "up", 100.0),
                    ("NOW", "2025-08-04 10:00", "down", -50.0),
                    ("NOW", "2025-09-01 10:00", "sideways", -80.0),
                    ("GEV", "2026-02-02 10:00", "up", 30.0)])
    b_on = trades([("NOW", "2025-03-03 10:00", "up", 100.0),
                   ("GEV", "2026-02-02 10:00", "up", 30.0)])
    flipped, switches = {"A": a_off, "B": b_on}, {"A": True, "B": False}
    eff = RES.switch_effects(base, flipped, switches, ["NOW", "GEV"]).set_index(["rule", "stock"])
    assert eff.loc[("A", "NOW"), ["as set", "trades off", "trades on"]].tolist() == ["on", 3, 2]
    assert eff.loc[("A", "NOW"), ["P&L off $", "P&L on $", "change $"]].tolist() == [-30, 50, 80]
    assert eff.loc[("A", "NOW"), "profit ÷ drawdown on"] == pytest.approx(1.0)  # 100, then -50
    assert eff.loc[("B", "NOW"), ["as set", "P&L off $", "P&L on $"]].tolist() == ["off", 50, 100]
    assert eff.loc[("B", "all stocks"), "change $"] == 50
    br = RES.effect_breakdown(base, flipped, switches, ["NOW", "GEV"])
    a_now = br[(br["rule"] == "A") & (br["stock"] == "NOW")].set_index("group")["change $"]
    assert a_now.to_dict() == {"up": 0.0, "sideways": 80.0, "down": 0.0, "2025 H1": 0.0,
                               "2025 H2": 80.0}
    figs = [CH.switch_effect_figure(RES.switch_effects(base, flipped, switches, ["NOW", "GEV"])),
            CH.effect_breakdown_figure(br)]
    assert all(f.layout.legend.y < 0 for f in figs)
    heat = next(t for t in figs[1].data if t.type == "heatmap")
    assert list(heat.x) == ["market up", "market sideways", "market down", "2025 H1", "2025 H2",
                            "2026 H1"]


def test_the_nearest_levels_are_read_per_session_with_an_optional_gap():
    days = pd.DataFrame({"high": [105.0, 108.0], "low": [95.0, 97.0]},
                        index=pd.DatetimeIndex(["2025-03-03", "2025-03-04"]))
    prev = IND.previous_range(days)
    assert prev.sort_values("price")[["day", "price"]].values.tolist() == [
        [pd.Timestamp("2025-03-04"), 95.0], [pd.Timestamp("2025-03-04"), 105.0]]
    levels = pd.DataFrame({"day": pd.to_datetime(["2025-03-03", "2025-03-03", "2025-03-04"]),
                           "price": [102.0, 98.0, 110.0]})
    idx = pd.DatetimeIndex(["2025-03-03 09:30", "2025-03-03 09:31", "2025-03-04 09:30",
                            "2025-03-05 09:30"]).tz_localize(NY)
    prices = pd.Series([100.0, 102.0, 100.0, 100.0], index=idx)
    near = IND.nearest_levels(prices, levels)
    assert near["above"].tolist()[:3] == [102.0, 102.0, 110.0]       # a level AT the price counts
    assert near["below"].tolist()[:2] == [98.0, 102.0] and np.isnan(near["below"].iloc[2])
    assert near.iloc[3].isna().all()                                 # no level known that day
    far = IND.nearest_levels(prices, levels, gap=pd.Series(1.5, index=idx))     # 1.5 away at least
    assert far["above"].iloc[0] == 102.0 and np.isnan(far["above"].iloc[1])
    assert far["below"].iloc[:2].tolist() == [98.0, 98.0] and far["above"].iloc[2] == 110.0


def test_the_level_filters_block_a_trade_without_room_to_the_next_level():
    rules = COMBINE.load_rules()
    c = candles_from([(100, 100, 100, 100)] * 3)
    near = pd.DataFrame({"above": [101.5, 103.0, np.nan], "below": [98.5, 97.0, np.nan]},
                        index=c.index)
    for name, key, knob in (("major levels", "major", "level_room_moves"),
                            ("previous high / low", "previous", "prev_room_moves")):
        ctx = context(c, levels={key: near})                         # typical move 1.0
        allowed = rules[name].allow(ctx, {knob: 2.0})
        assert allowed["long"].tolist() == [False, True, True]       # 1.5 away < 2: no long
        assert allowed["short"].tolist() == [False, True, True]
        assert rules[name].allow(ctx, {knob: 1.0}).all().all()
        assert rules[name].allow(context(c), {}).all().all()         # nothing known: no block
        words = rules[name].explain(ctx, {knob: 2.0}, 0)
        assert words.startswith(f"{name}: the nearest")
        assert "101.50, 1.5 typical moves away (needs 2): no long" in words


def test_a_profit_target_closes_the_trade_and_the_stop_wins_a_shared_candle():
    rows = [(100, 100, 100, 100), (100, 100.5, 99.8, 100.4), (100.4, 102.2, 100.3, 101.5),
            (101.5, 101.6, 101.0, 101.2)]
    c = candles_from(rows)
    targets = pd.DataFrame({"long": 102.0, "short": 98.0}, index=c.index)
    decisions = pd.DataFrame({"signal": np.array([1, 0, 0, 0], np.int8)}, index=c.index)

    def first(candles, side=1, **kw):
        """The first trade of a long (or short) signal on the first candle, without costs."""
        n = len(candles)
        return SIM.run(context(candles), side * decisions.iloc[:n], stop_moves=1.0,
                       slippage=0.0, commission=0.0, **kw)["trades"].iloc[0]

    hit = first(c, targets=targets)                                   # stop 99, target 102
    assert hit[["exit reason", "exit price", "target"]].tolist() == [SIM.REASONS[4], 102.0, 102.0]
    assert first(c)[["exit reason", "exit price"]].tolist() == [SIM.REASONS[0], 101.2]
    gapped = candles_from([*rows[:2], (103, 103.5, 102.8, 103.2), rows[3]])
    assert first(gapped, targets=targets)["exit price"] == 103.0      # opened past it: the open
    both = candles_from([rows[0], (100, 102.5, 98.5, 101)])
    assert first(both, targets=targets)["exit reason"] == SIM.REASONS[0]   # stop wins the candle
    down = candles_from([rows[0], (100, 100.2, 97.5, 98)])
    short = first(down, side=-1, targets=targets)
    assert short[["side", "exit reason", "exit price"]].tolist() == ["short", SIM.REASONS[4], 98.0]


def test_commission_per_share_and_the_half_cent_floor_price_a_cheap_stock():
    rows = [(10, 10, 10, 10), (10, 10.2, 9.95, 10.1), (10.1, 10.3, 10.05, 10.2)]
    c = candles_from(rows)
    decisions = pd.DataFrame({"signal": np.array([1, 0, 0], np.int8)}, index=c.index)
    t = SIM.run(context(c, move=0.5), decisions, stop_moves=1.0, commission=1.0, per_share=0.005,
                slippage=0.0002, min_slip=0.005)["trades"].iloc[0]
    fill_in, fill_out = 10 + 0.005, 10.2 - 0.005                       # half a cent beats 2 bp
    shares = 10_000 / fill_in                                          # ~1,000: $5 an order
    assert t[["entry price", "exit price"]].tolist() == pytest.approx([fill_in, fill_out])
    assert t["P&L $"] == pytest.approx(shares * (fill_out - fill_in) - 2 * shares * 0.005)


def test_sessions_to_a_report_count_trading_sessions_and_skip_a_weekend():
    idx = pd.DatetimeIndex(["2025-03-06 15:59", "2025-03-07 09:30", "2025-03-07 15:59",
                            "2025-03-10 09:30", "2025-03-11 09:30"]).tz_localize(NY)
    until = IND.sessions_to_event(idx, ["2025-03-08"])             # a Saturday: Monday reacts
    assert until.tolist()[:4] == [2.0, 1.0, 1.0, 0.0] and np.isnan(until.iloc[4])
    assert IND.sessions_to_event(idx, []).isna().all()


def test_the_earnings_guard_blocks_the_session_before_a_report_and_closes_open_trades():
    from types import SimpleNamespace

    rules = COMBINE.load_rules()
    guard = rules["earnings guard"]
    times = ["2025-03-06 15:58", "2025-03-06 15:59", "2025-03-07 09:30", "2025-03-07 15:58",
             "2025-03-07 15:59", "2025-03-10 09:30", "2025-03-10 09:31"]
    c = candles_from([(100, 100.2, 99.9, 100.1)] * 7, times=times)
    report = IND.sessions_to_event(c.index, ["2025-03-10"])          # reacts on Monday
    ctx = COMBINE.make_context(NAMES, c, context(c)["bands"], pd.Series(1.0, index=c.index),
                               context(c)["trend"], report=report)
    allowed = guard.allow(ctx, {})
    assert allowed["long"].tolist() == [True, True, False, False, False, True, True]
    assert guard.flat(ctx, {}).tolist() == [False, False, False, False, True, False, False]
    assert not guard.allow(ctx, {"guard_sessions": 2})["short"].iloc[0]
    assert "no new trade" in guard.explain(ctx, {}, 2)
    signal = pd.Series([1, 0, 0, 0, 0, 0, 0], index=c.index)
    rules = {**rules, "test": SimpleNamespace(KIND="trigger", signal=lambda ctx, p: signal)}
    decisions = COMBINE.decide(ctx, rules, {"test": True, "earnings guard": True}, only_on=True)
    assert decisions["flat"].tolist() == guard.flat(ctx, {}).tolist()
    trade = SIM.run(ctx, decisions, stop_moves=1.0, slippage=0.0, commission=0.0)[
        "trades"].iloc[0]
    assert trade["exit reason"] == SIM.REASONS[5]
    assert trade["exit time"].strftime("%m-%d %H:%M") == "03-07 15:59"   # not held over the night


def test_a_trade_keeps_the_stop_distance_it_opened_with_when_the_levers_change():
    rows = [(100, 100, 100, 100), (100, 100.5, 99.8, 100.4), (100.4, 101.0, 100.2, 100.9),
            (100.9, 100.95, 100.3, 100.5), (100.5, 100.6, 100.4, 100.5)]
    c = candles_from(rows)
    decisions = pd.DataFrame({"signal": np.array([1, 0, 0, 0, 0], np.int8)}, index=c.index)
    stops = pd.Series([1.0, 1.0, 0.1, 0.1, 0.1], index=c.index)     # re-tuned after the entry
    sim = SIM.run(context(c), decisions, stop_moves=stops, slippage=0.0, commission=0.0)
    trade = sim["trades"].iloc[0]
    assert trade["first stop"] == 99.0                               # 1 typical move, not 0.1
    assert sim["stop"].tolist()[1:] == [99.0, 99.5, 100.0, 100.0]    # trailing 1 behind 101
    assert trade["exit reason"] == SIM.REASONS[3]                    # 0.1 would have hit at 100.9


def test_deciding_only_with_the_switched_on_rules_gives_the_same_signal():
    rules = COMBINE.load_rules()
    close = [100, 98.5, 99.5, 100, 101.5, 101.8, 100.5, 98.0, 97.5, 99.2]
    c = candles_from([(x, x, x, x) for x in close])
    ctx = context(c, sector=1.0)
    switches = {n: getattr(r, "ON", True) for n, r in rules.items()}
    full = COMBINE.decide(ctx, rules, switches)
    fast = COMBINE.decide(ctx, rules, switches, only_on=True)
    assert fast["signal"].equals(full["signal"]) and fast["flat"].equals(full["flat"])
    assert "moving average: long" in full and "moving average: long" not in fast


# --- the tuning notebook: every hyperparameter, two passes, a blended score -------------------
TU, HP, PIPE = _load("tuning"), _load("hyperparams"), _load("pipeline")


def synthetic_lab(n_sessions=40, per=30, seed=5, *, change_after=None):
    """A random-walk stock of ``per`` one-minute regular candles a session, with random daily
    candles for it, its sector and the market; from minute row ``change_after`` on, a
    different future (to check that nothing earlier depends on it)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-03-03", periods=n_sessions)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in days
                            for m in range(per)]).tz_localize(NY)
    close = 100 + np.cumsum(rng.normal(0, 0.3, len(idx)))
    if change_after is not None:
        close[change_after:] += np.random.default_rng(99).normal(0, 5, len(idx) - change_after)
    open_ = np.r_[close[0], close[:-1]]
    minutes = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + 0.1,
                            "low": np.minimum(open_, close) - 0.1, "close": close,
                            "volume": 100.0, "traded": True, "regular": True}, index=idx)

    def daily(k):
        d = pd.bdate_range(end=days[-1], periods=n_sessions + 110)
        c = 100 + np.cumsum(np.random.default_rng(k).normal(0, 1, len(d)))
        return pd.DataFrame({"open": c - 0.2, "high": c + 1, "low": c - 1, "close": c,
                             "volume": 1e6}, index=d)

    reports = pd.DataFrame(columns=["symbol", "time", "when", "reaction day"])
    return TU.Lab(NAMES, minutes, {"NOW": daily(1), "IGV": daily(2), "SPY": daily(3)}, reports,
                  COMBINE.load_rules(), {"commission": 0.0, "slippage": 0.0})


def test_the_hyperparameter_table_holds_every_setting_switch_and_lever_once():
    rules = COMBINE.load_rules()
    rows = HP.table(rules)
    assert set(HP.DECLARED) <= set(rows)
    order = list(rows)
    for name, rule in rules.items():
        assert rows[HP.switch_key(name)]["value"] is getattr(rule, "ON", True)
        for k in rule.PARAMS:
            assert rows[k]["needs"] == HP.switch_key(name)
            assert order.index(HP.switch_key(name)) < order.index(k)     # switch before lever
    for lever, kind in (("stop_moves", "stop_kind"), ("stop_days", "stop_kind"),
                        ("target_days", "target_kind")):
        assert rows[lever]["needs"][0] == kind and order.index(kind) < order.index(lever)
    daily = {**HP.defaults(rules), "stop_kind": "daily fixed"}
    assert HP.active(rows["stop_days"], daily) and not HP.active(rows["stop_moves"], daily)
    assert not HP.active(rows["target_days"], HP.defaults(rules))
    for name, r in rows.items():
        if "choices" in r:
            assert r["value"] in r["choices"], name
        else:
            low, high, step = r["search"]
            assert r["low"] <= low <= high <= r["high"] and r["low"] <= r["value"] <= r["high"]
            assert round((r["value"] - r["low"]) / r["step"], 9).is_integer(), name
    start = HP.defaults(rules)
    assert start["on: band fade"] and not start["on: moving average"]
    assert start["breakout_closes"] == 2 and start["candle_size"] == 1
    assert len(HP.frame(rules)) == len(rows)
    kinds = {name: type(HP.widget(name, r, start[name])).__name__ for name, r in rows.items()}
    assert kinds["extended_bands"] == "switch" and kinds["candle_size"] == "dropdown"
    assert kinds["band_width"] == "slider"


def test_settings_files_round_trip_and_become_the_main_notebooks_starting_values(tmp_path):
    rules = COMBINE.load_rules()
    tuned = {**HP.defaults(rules), "band_width": 2.75, "on: major levels": True,
             "level_room_moves": 3.5, "target_kind": "major", "band_length": 999, "gone": 1}
    path = HP.save({"NOW": tuned}, meta={"note": "test"}, name="test", folder=tmp_path)
    assert HP.files(tmp_path) == [path] and HP.load(path)["meta"]["note"] == "test"
    start = HP.starting(rules, path, "NOW")
    assert start["band_width"] == 2.75 and start["on: major levels"] is True
    assert start["level_room_moves"] == 3.5 and start["target_kind"] == "major"
    assert start["band_length"] == 20 and "gone" not in start            # out of range: left out
    assert HP.starting(rules) == HP.defaults(rules)


def test_the_lab_trades_exactly_what_the_pipeline_builds_step_by_step():
    lab = synthetic_lab()
    s = HP.defaults(lab.rules)
    m = lab.minutes
    ctx = PIPE.context(NAMES, m, PIPE.bands(m, 20, 2.0, False), PIPE.move(m, m, 15, 14)[0],
                       PIPE.trends(lab.days, 50, 10, 0.5), PIPE.average(m, 60, False),
                       PIPE.levels(lab.days["NOW"], 5, 120, 0.5, 2, True),
                       IND.previous_range(lab.days["NOW"]), lab.reports)
    decisions = COMBINE.decide(ctx, lab.rules, HP.switches(s, lab.rules), s)
    direct = SIM.run(ctx, decisions, stop_moves=3.0, slippage=0.0, commission=0.0)["trades"]
    assert len(direct) > 3
    pd.testing.assert_frame_equal(lab.run(s).reset_index(drop=True),
                                  direct.reset_index(drop=True))
    five = lab.run({**s, "candle_size": 5, "on: major levels": True, "target_kind": "both"})
    assert len(five) > 0                                     # every other part builds as well


def test_the_search_space_is_free_by_default_and_overrides_fix_single_values():
    rules = COMBINE.load_rules()
    space = HP.default_space(rules)
    assert all(HP.is_free(spec) for spec in space.values())             # the optimizer decides
    assert space["on: earnings guard"] == [True, False]
    custom = HP.clean_space({**space, "on: earnings guard": [True], "stop_moves": (3.0, 3.0, 0.5),
                             "band_length": (60, 10, 5), "band_width": (1.6, 2.9, 0.25),
                             "candle_size": [], "target_kind": ["major"]}, rules)
    assert custom["on: earnings guard"] == [True] and not HP.is_free(custom["stop_moves"])
    assert custom["band_length"] == (10, 60, 5)                         # put in order
    assert custom["band_width"] == (1.6, 2.85, 0.25)                    # onto its grid
    assert custom["candle_size"] == [1, 5, 15]                          # none allowed: all
    words = HP.describe_space(custom, rules).set_index("hyperparameter")["the optimizer"]
    assert words["on: earnings guard"] == "always on"
    assert words["on: band fade"] == "optimizer decides (on or off)"
    assert words["stop_moves"] == "fixed: 3.0"
    assert words["target_kind"] == "fixed: the next major level"
    assert words["band_length"] == "optimizer decides: 10 to 60, step 5"
    first = HP.project({**HP.defaults(rules), "band_width": 2.0, "band_length": 999}, custom, rules)
    assert first["stop_moves"] == 3.0 and first["band_width"] == 2.1 and first["band_length"] == 60
    assert first["on: earnings guard"] is True and first["target_kind"] == "major"
    assert HP.values(custom["band_length"]) == 11
    noisy, start = 12 * 0.1, HP.defaults(rules)                # Optuna's step 0.1 draws 1.2 so
    drawn = TU.suggest(optuna.trial.FixedTrial({"trend_flat": noisy}), HP.table(rules),
                       HP.fixed_space(start, rules, ["trend_flat"]), start)
    assert noisy != 1.2 and drawn["trend_flat"] == 1.2                  # back on the grid


def test_the_score_is_the_daily_sharpe_of_marked_to_market_profit_and_luck_rises_with_trials():
    d = pd.Series([10.0, -5.0, 20.0, 0.0])
    assert TU.score(d, 4, min_per_day=0.5) == pytest.approx(d.mean() / d.std() * np.sqrt(252))
    assert TU.score(d, 1, min_per_day=0.5) == -1e9                     # too few trades
    times = ["2025-03-03 15:58", "2025-03-03 15:59", "2025-03-04 09:30", "2025-03-04 09:31"]
    rows = [(100, 100, 100, 100), (100, 100.5, 99.9, 100.4), (101, 101.2, 100.9, 101),
            (101, 101, 101, 101)]
    trades = run_once(rows, [1, 0, 0, 0], times=times, stop_moves=5.0)["trades"]
    daily = TU.daily_pnl(trades, candles_from(rows, times=times), 10_000.0)
    assert daily.tolist() == pytest.approx([40.0, 60.0])        # open overnight: 100 × 0.40
    assert daily.sum() == pytest.approx(trades["P&L $"].sum())
    assert TU.luck(1, 252) == pytest.approx(0.0, abs=1e-9)             # one try: no selection
    assert TU.luck(100, 252) == pytest.approx(2.462, abs=0.01)
    assert TU.luck(5000, 373) > TU.luck(100, 373) > TU.luck(100, 1000)
    assert TU.luck(100, 252, chance=0.95) > TU.luck(100, 252)


def test_the_optimizer_tunes_switches_levers_and_exit_kinds_on_the_training_window():
    lab = synthetic_lab()
    fixed = HP.defaults(lab.rules)
    free = ["on: band fade", "on: band breakout", "breakout_closes", "stop_kind", "stop_moves",
            "stop_days", "band_width"]
    space = HP.fixed_space(fixed, lab.rules, free)
    res = TU.optimize(lab, 30, space=space, start=fixed, trials=12, min_per_day=0.0, seed=1)
    assert set(res["best"]) == set(fixed)                              # complete settings
    assert HP.project(res["best"], space, lab.rules) == res["best"]    # exactly on the grid
    log = res["trials"]
    assert len(log) == 12 and {"score", "P&L $", "trades", "luck"} <= set(log.columns)
    assert res["score"] == pytest.approx(log["score"].max())
    off = log[log["on: band breakout"] == False]  # noqa: E712
    assert "breakout_closes" not in off or off["breakout_closes"].isna().all()
    moves = log[log["stop_kind"] == "moves"]
    assert "stop_days" not in moves or moves["stop_days"].isna().all()  # only with its kind
    assert log["luck"].dropna().is_monotonic_increasing


def test_the_validation_window_is_scored_beside_training_and_cannot_change_it():
    lab = synthetic_lab()
    s = HP.defaults(lab.rules)
    a = TU.judge(lab, s, 30, min_per_day=0.0, validate=True)
    assert {"validation score", "validation P&L $", "validation trades"} <= set(a)
    cut = lab.rows(s, 30, 30)[0]
    b = TU.judge(synthetic_lab(change_after=cut), s, 30, min_per_day=0.0, validate=True)
    assert (a["score"], a["P&L $"], a["trades"]) == (b["score"], b["P&L $"], b["trades"])
    assert a["validation P&L $"] != b["validation P&L $"]             # the validation did change
    trades, daily = lab.daily(s)
    assert daily.sum() == pytest.approx(trades["P&L $"].sum())        # closed by the data's end


def test_several_stocks_are_scored_together_on_one_scale_and_each_on_its_own():
    now = synthetic_lab()
    other = synthetic_lab(seed=8)
    gev = TU.Lab({"stock": "GEV", "sector": "IGV", "market": "SPY"}, other.minutes,
                 {"GEV": other.days["NOW"], "IGV": other.days["IGV"], "SPY": other.days["SPY"]},
                 other.reports, other.rules, other.costs)
    s = HP.defaults(now.rules)
    both = TU.judge([now, gev], s, 30, min_per_day=0.0, validate=True)
    one, two = (TU.judge(x, s, 30, min_per_day=0.0, validate=True) for x in (now, gev))
    assert both["P&L $"] == pytest.approx(one["P&L $"] + two["P&L $"])       # added up
    assert both["trades"] == one["trades"] + two["trades"]
    assert (both["NOW score"], both["GEV score"]) == (one["score"], two["score"])
    assert both["worst stock"] == min(one["score"], two["score"])
    assert both["validation P&L $"] == pytest.approx(one["validation P&L $"]
                                                     + two["validation P&L $"])
    assert list(TU.by_stock(both, ["NOW", "GEV"])["stock"]) == ["NOW", "GEV", "together"]
    space = HP.fixed_space(s, now.rules, ["band_width"])
    res = TU.optimize([now, gev], now.sessions()[1][30], space=space, start=s, trials=3,
                      min_per_day=0.0)                  # a date: each stock splits there
    assert len(res["trials"]) == 3 and "GEV score" in res["trials"] and TU.group([now, gev]) \
        == "NOW + GEV"
    walked = TU.walk([now, gev], first=20, refit=10, space=space, start=s, tune=False)
    pd.testing.assert_frame_equal(                     # every stock traded as on its own
        walked["trades"][walked["trades"]["stock"] == "GEV"].reset_index(drop=True),
        gev.run(s, 20).reset_index(drop=True))
    sides = TU.sides([now, gev], s)
    assert set(sides["stock"]) <= {"NOW", "GEV"} and {"long", "short"} <= set(sides.columns)


def test_a_run_is_kept_in_a_journal_continued_by_name_and_refused_with_other_inputs(tmp_path):
    lab = synthetic_lab()
    fixed = HP.defaults(lab.rules)
    space = HP.fixed_space(fixed, lab.rules, ["stop_moves", "band_width"])
    path = tmp_path / "studies" / "tuning.log"
    kw = {"min_per_day": 0.0, "validate": True, "storage": path, "name": "NOW test"}
    first = TU.optimize(lab, 30, space=space, start=fixed, trials=4, **kw)
    assert TU.runs(path) == TU.runs(path, "NOW") == ["NOW test"] and TU.runs(path, "GEV") == []
    assert first["trials"]["validation trades"].notna().all()
    cfg = TU.run_settings(path, "NOW test", lab.rules)
    assert cfg["space"] == space and cfg["trials"] == 4 and cfg["min_per_day"] == 0.0
    more = TU.optimize(lab, 30, space=cfg["space"], start=cfg["start"], trials=3, **kw)
    assert len(more["trials"]) == 7 and more["score"] >= first["score"]   # 4 + 3, one run
    other = HP.fixed_space(fixed, lab.rules, ["stop_moves"])
    with pytest.raises(ValueError, match="other inputs"):
        TU.optimize(lab, 30, space=other, start=fixed, trials=1, **kw)
    none = TU.optimize(lab, 30, space=space, start=fixed, trials=3, min_per_day=1e6)
    assert none["trials"]["score"].isna().all() and len(none["trials"]) == 3   # all pruned
    assert none["best"] == HP.project(fixed, space, lab.rules)


def test_two_workers_add_trials_to_one_run_at_once_without_repeating_each_other(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    fixed = HP.defaults(COMBINE.load_rules())
    space = HP.fixed_space(fixed, COMBINE.load_rules(), ["stop_moves", "band_width"])
    kw = {"space": space, "start": fixed, "min_per_day": 0.0, "storage": tmp_path / "j.log",
          "name": "NOW test"}
    TU.optimize(synthetic_lab(), 30, trials=2, **kw)                     # the run is made
    with ThreadPoolExecutor(2) as pool:                                  # two workers continue
        list(pool.map(lambda k: TU.optimize(synthetic_lab(), 30, trials=4, seed=1000 * (k + 1),
                                            parallel=True, **kw), (0, 1)))
    st = optuna.load_study(study_name="NOW test", storage=TU.journal(tmp_path / "j.log"))
    assert len(st.trials) == 10                                          # 2 + 4 + 4, one run
    drawn = [tuple(sorted(t.params.items())) for t in st.trials[2:]]
    assert len(set(drawn)) >= 6                                          # hardly any repeats


def test_the_walk_reads_no_later_session_and_without_tuning_is_a_plain_run():
    lab = synthetic_lab()
    fixed = HP.defaults(lab.rules)
    space = HP.fixed_space(fixed, lab.rules, ["stop_moves", "band_width", "on: band fade"])
    plain = TU.walk(lab, first=20, refit=5, space=space, start=fixed, tune=False)
    pd.testing.assert_frame_equal(plain["trades"].reset_index(drop=True),
                                  lab.run(fixed, 20).reset_index(drop=True))
    kw = {"first": 20, "refit": 5, "space": space, "start": fixed, "trials": 3,
          "min_per_day": 0.0, "seed": 2}
    full = TU.walk(lab, **kw)
    cut = lab.rows(fixed, 30, 30)[0]
    changed = TU.walk(synthetic_lab(change_after=cut), **kw)
    a, b = full["settings"], changed["settings"]
    pd.testing.assert_frame_equal(a[a["session"] <= 30], b[b["session"] <= 30])
    short = TU.walk(lab, **kw, last=30)                         # stops before session 30
    first_out = lab.sessions(1)[1][30]
    assert short["settings"]["session"].max() < 30
    assert (short["trades"]["entry time"].dt.tz_localize(None) < first_out).all()
    with pytest.raises(ValueError, match="one candle size"):
        TU.walk(lab, first=20, refit=5, space=HP.default_space(lab.rules), start=fixed)


def test_effects_show_which_hyperparameters_move_the_score_and_why_others_do_not():
    lab = synthetic_lab()
    s = HP.defaults(lab.rules)
    eff = TU.effects(lab, s, names=["band_width", "level_room_moves", "on: band fade"],
                     first=10, min_per_day=0.0, points=3)
    summary = TU.effect_summary(eff, s, lab.rules).set_index("hyperparameter")
    assert summary.loc["level_room_moves", "effect"] == 0.0
    assert summary.loc["level_room_moves", "no effect?"].startswith("none: not in use")
    assert summary.loc["band_width", "effect"] > 0
    assert set(eff.loc[eff["hyperparameter"] == "on: band fade", "value"]) == {True, False}
    figs = [CH.effects_figure(eff, summary.reset_index()),
            CH.trials_figure(TU.optimize(
                lab, 30, space=HP.fixed_space(s, lab.rules, ["band_width"]), start=s, trials=3,
                min_per_day=0.0, validate=True))]
    assert all(f.layout.legend.y < 0 for f in figs)


def test_a_trade_keeps_the_holding_limit_it_opened_with():
    times = ["2025-03-03 15:57", "2025-03-03 15:58", "2025-03-03 15:59", "2025-03-04 09:30",
             "2025-03-04 09:31"]
    c = candles_from([(100, 100.1, 99.9, 100)] * 5, times=times)
    decisions = pd.DataFrame({"signal": np.array([1, 0, 0, 0, 0], np.int8)}, index=c.index)
    run = lambda limits: SIM.run(context(c), decisions, stop_moves=5.0, slippage=0.0,  # noqa: E731
                                 commission=0.0, max_sessions=pd.Series(limits, index=c.index))
    assert run([1, 1, 0, 0, 0])["trades"].iloc[0]["exit reason"] == SIM.REASONS[2]
    assert run([0, 0, 1, 1, 1])["trades"].iloc[0]["exit reason"] == SIM.REASONS[3]


def test_walks_run_in_separate_processes_alike(tmp_path):
    lab = synthetic_lab()
    fixed = HP.defaults(lab.rules)
    cfg = {"first": 20, "refit": 10, "space": HP.fixed_space(fixed, lab.rules, ["stop_moves"]),
           "start": fixed, "trials": 2, "min_per_day": 0.0}
    jobs = [(lab, "tuned", cfg), (lab, "no tuning", {**cfg, "tune": False})]
    apart = TU.run_walks(jobs, cache=tmp_path / "apart", workers=2)
    here = TU.run_walks(jobs, cache=tmp_path / "here")
    for key in apart:
        pd.testing.assert_frame_equal(apart[key]["trades"], here[key]["trades"])
        pd.testing.assert_frame_equal(apart[key]["settings"], here[key]["settings"])
    table = RES.walk_table(apart).set_index(["stock", "walk"])
    assert table.loc[("NOW", "no tuning"), "vs no tuning $"] == 0
    figs = [CH.walk_curves_figure(apart), CH.settings_figure(apart, "NOW")]
    assert all(f.layout.legend.y < 0 for f in figs)


# ---- stock groups (groups.py) ------------------------------------------------------------------

GR = _load("groups")
COSTS = {"notional": 20_000.0, "risk": 100.0, "commission": 1.0, "per_share": 0.005,
         "slippage": 0.0002, "min_slip": 0.005}


def walk_minutes(returns, start="2025-03-03", gaps=None):
    """Regular-session 1-minute candles (390 a session) whose returns inside each session are
    exactly ``returns`` (a session's first one from its own open); a session opens at the
    previous close, moved by its overnight gap in ``gaps`` (log, 0 by default)."""
    r = np.asarray(returns, float)
    days = pd.bdate_range(start, periods=len(r) // 390)
    idx = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + k) for d in days
                            for k in range(390)]).tz_localize(NY)
    open_, close, price = np.empty(len(r)), np.empty(len(r)), 100.0
    for d in range(len(days)):
        part = slice(d * 390, (d + 1) * 390)
        first = price * np.exp(0.0 if gaps is None else gaps[d])
        close[part] = first * np.exp(np.cumsum(r[part]))
        open_[part] = np.r_[first, close[part][:-1]]
        price = close[part][-1]
    frame = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.0001,
                          "low": np.minimum(open_, close) * 0.9999, "close": close,
                          "volume": 100.0}, index=idx)
    return frame.assign(traded=True, regular=True)


def ar_returns(phi, n, seed, scale=0.001):
    """1-minute returns where each one carries ``phi`` times the one before."""
    e = np.random.default_rng(seed).normal(0, scale, n)
    r = np.empty(n)
    r[0] = e[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    return r


def test_group_blocks_and_typical_move_are_the_strategys_own():
    m = DATA.drop_after_half_day_close(minutes(4, start="2025-07-01")).assign(regular=True)
    cols = ["open", "high", "low", "close", "volume"]
    pd.testing.assert_frame_equal(GR._blocks(m, 15), DATA.resample(m, 15)[cols])  # half-day
    pd.testing.assert_series_equal(GR._move(m), PIPE.move(m, m, 15, 14)[0], check_names=False)


def test_session_sums_never_read_a_later_session():
    m = minutes(6, seed=3).assign(regular=True)
    market = GR.block_returns(minutes(6, seed=4).assign(regular=True))
    full = GR.session_sums(m, market)
    cut = GR.session_sums(m[m.index < m.index[390 * 4]], market)
    pd.testing.assert_frame_equal(cut, full.iloc[:4])


def test_the_character_of_a_random_walk():
    rng = np.random.default_rng(1)
    m = walk_minutes(rng.normal(0, 0.001, 390 * 40))
    m.loc[m.index[::10], "volume"] = 0.0                  # one minute in ten without a trade
    market = GR.block_returns(walk_minutes(rng.normal(0, 0.001, 390 * 40)))
    f = GR.character(GR.session_sums(m, market).sum(), COSTS).iloc[0]
    assert abs(f["follow_1m"]) < 0.03
    assert abs(f["ratio_15m"] - 1) < 0.15
    assert abs(f["market_corr"]) < 0.12
    assert abs(f["band_follow"]) < 0.1                    # no edge after a close outside
    assert f["quiet_share"] == pytest.approx(0.1)
    assert f["night_share"] == 0                          # every session opens at the close
    assert f["open_share"] == pytest.approx(30 / 390, abs=0.01)
    assert 0 < f["band_back"] < 1 and f["cost_moves"] > 0 and 0 < f["jump_share"] < 1


def test_reverting_and_trending_minutes_read_apart():
    rev, run = (GR.character(GR.session_sums(walk_minutes(ar_returns(phi, 390 * 30, seed)))
                             .sum(), COSTS).iloc[0] for phi, seed in ((-0.3, 5), (0.3, 6)))
    assert rev["follow_1m"] == pytest.approx(-0.3, abs=0.05)
    assert run["follow_1m"] == pytest.approx(0.3, abs=0.05)
    assert rev["ratio_15m"] < 0.75 < 1.3 < run["ratio_15m"]
    assert rev["band_follow"] < run["band_follow"]
    assert rev["band_back"] > run["band_back"]


def test_the_market_link_finds_a_planted_correlation():
    rng = np.random.default_rng(2)
    market = rng.normal(0, 0.001, 390 * 30)
    stock = market + rng.normal(0, 0.001, len(market))     # correlation √½
    sums = GR.session_sums(walk_minutes(stock), GR.block_returns(walk_minutes(market)))
    assert GR.character(sums.sum(), COSTS).iloc[0]["market_corr"] == pytest.approx(
        np.sqrt(0.5), abs=0.06)


def test_the_overnight_share_counts_the_gaps():
    r = np.random.default_rng(3).normal(0, 0.001, 390 * 20)
    gaps = np.r_[0.0, np.full(19, 0.01)]
    f = GR.character(GR.session_sums(walk_minutes(r, gaps=gaps)).sum(), COSTS).iloc[0]
    oc = np.abs(r.reshape(20, 390).sum(axis=1))[1:]       # the first session has no gap
    assert f["night_share"] == pytest.approx(0.19 / (0.19 + oc.sum()), rel=1e-6)


def test_band_events_match_a_plain_loop():
    m = minutes(3, seed=7).assign(regular=True)
    ev = GR.band_events(m)
    b = PIPE.bands(m, *GR.BAND, False)
    c, up, lo = m["close"].to_numpy(), b["upper"].to_numpy(), b["lower"].to_numpy()
    mv, day, f = GR._move(m).to_numpy(), m.index.normalize(), GR.FOLLOW
    want = []
    for t in range(1, len(c) - f):
        if day[t] != day[t - 1] or day[t + f] != day[t] or not mv[t] > 0:
            continue
        side = 1 if c[t] > up[t] and c[t - 1] <= up[t - 1] else \
            -1 if c[t] < lo[t] and c[t - 1] >= lo[t - 1] else 0
        if side:
            back = c[t + 1] <= up[t + 1] if side > 0 else c[t + 1] >= lo[t + 1]
            want.append((m.index[t], side, back, side * (c[t + f] - c[t]) / mv[t]))
    assert len(ev) == len(want) > 10
    assert list(ev["time"]) == [w[0] for w in want]
    assert list(ev["side"]) == [w[1] for w in want] and list(ev["back"]) == [w[2] for w in want]
    np.testing.assert_allclose(ev["follow"], [w[3] for w in want])


def planted_scores(seed=4, noise=0.2):
    """Scores of 60 stocks in three planted groups of 30, 20 and 10, and the truth."""
    rng = np.random.default_rng(seed)
    centres = {"a": [2, 0, 0, 0], "b": [0, 2, 0, 0], "c": [0, 0, -2, 2]}
    sizes = {"a": 30, "b": 20, "c": 10}
    x = np.vstack([np.asarray(centres[g]) + rng.normal(0, noise, (n, 4))
                   for g, n in sizes.items()])
    truth = [g for g, n in sizes.items() for _ in range(n)]
    return pd.DataFrame(x, index=[f"S{i}" for i in range(60)],
                        columns=list(GR.FEATURES)[:4]), truth


def test_grouping_finds_planted_groups_and_ranks_their_members():
    from sklearn.metrics import adjusted_rand_score

    sc, truth = planted_scores()
    out = GR.grouping(sc, 3, n_tune=4)
    assert adjusted_rand_score(truth, out["labels"]) == 1
    assert list(out["groups"]["stocks"]) == [30, 20, 10]         # numbered by size
    for _, part in out["members"].groupby("group"):
        assert part["distance"].is_monotonic_increasing
        assert list(part["rank"]) == list(range(1, len(part) + 1))
        assert list(part["role"]) == ["tune"] * 4 + ["check"] * (len(part) - 4)
    assert out["groups"].loc[0, "in words"] == GR.FEATURES["move_pct"]["words"][1]


def test_stable_groups_score_high_on_quality_and_persistence():
    sc, _ = planted_scores()
    a, b = planted_scores(noise=0.2)[0] + 0.01, planted_scores(noise=0.2)[0] - 0.01
    q = GR.quality(sc, a, b, ks=(2, 3, 4)).set_index("groups").loc[3]
    assert q["same on both halves"] == 1 and q["explained"] > 0.9 and q["separation"] > 0.7
    assert q["stays in its group"] == 1 and q["stays by chance"] == pytest.approx(
        0.5 ** 3 + (1 / 3) ** 3 + (1 / 6) ** 3)
    assert q["smallest group"] == 10
    out = GR.grouping(sc, 3, halves=(a, b))
    assert out["members"]["stays"].all() and (out["groups"]["stays on both halves"] == 1).all()
    moved = b.copy()
    moved.iloc[0] = sc.loc[sc.index[-1]]                  # a group-a stock looks like group c
    assert not GR.grouping(sc, 3, halves=(a, moved))["members"].loc["S0", "stays"]
    assert (GR.persistence(a, b) > 0.99).all()


def test_scores_are_rank_normal_and_an_outlier_cannot_stretch_them():
    s = GR.scores(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 1000.0]}))["x"]
    assert s.is_monotonic_increasing and s.iloc[2] == 0 and s.iloc[-1] == -s.iloc[0]
    assert s.iloc[-1] < 1.5


def test_group_sums_are_kept_and_computed_again_only_for_new_candles(tmp_path, monkeypatch):
    loads = []

    def fake(engine, names, start, end, extended=False):
        loads.append(names[0])
        return {names[0]: minutes(3, seed=sum(map(ord, names[0]))).assign(regular=True)}

    monkeypatch.setattr(GR.DATA, "load_minutes", fake)
    pool = pd.DataFrame({"symbol": ["AAA", "BBB"], "first": ["t0", "t0"], "last": ["t1", "t1"]})
    out = GR.compute(None, pool, folder=tmp_path, workers=2)
    assert sorted(loads) == ["AAA", "BBB", "SPY"] and list(out) == ["AAA", "BBB"]
    loads.clear()
    again = GR.compute(None, pool, folder=tmp_path)
    assert loads == [] and again["AAA"].equals(out["AAA"])
    pool.loc[1, "last"] = "t2"                                    # BBB got newer candles
    assert GR.missing(pool, folder=tmp_path) == ["BBB"]
    GR.compute(None, pool, folder=tmp_path)
    assert sorted(loads) == ["BBB", "SPY"]
    assert len(list(GR._where(tmp_path, *GR.WINDOW).glob("BBB-*.pkl"))) == 1


def test_group_figures_build_and_the_export_reads_back(tmp_path):
    market = GR.block_returns(walk_minutes(np.random.default_rng(99).normal(0, 0.001, 390 * 6)))
    runs = {f"S{i}": walk_minutes(ar_returns((-0.3, 0.0, 0.3)[i % 3], 390 * 6, i,
                                             scale=0.001 * (1 + i % 2))) for i in range(12)}
    sums = {s: GR.session_sums(m, market) for s, m in runs.items()}
    feats = list(GR.FEATURES)
    table = GR.characters(sums, COSTS)
    first, second = (GR.characters(sums, COSTS, part=p) for p in ("first half", "second half"))
    sc = GR.scores(table[feats])
    out = GR.grouping(sc, 3, n_tune=2)
    q = GR.quality(sc, GR.scores(first[feats]), GR.scores(second[feats]), ks=(2, 3, 4))
    xy, axes = GR.flat_map(sc)
    near = GR.nearest(sc, "S0", 3)
    assert len(near) == 3 and "S0" not in near.index
    day = runs["S0"].iloc[:390]
    events = GR.band_events(runs["S0"])
    figs = [CH.history_figure(table["sessions"], least=3),
            CH.feature_figure("S0", "band_follow", GR.moving(sums["S0"], COSTS, sessions=2)[
                "band_follow"], table, (first, second), GR.FEATURES["band_follow"], sessions=2),
            CH.band_events_figure("S0", day, PIPE.bands(runs["S0"], *GR.BAND, False).iloc[:390],
                                  events[events["time"] < runs["S0"].index[390]],
                                  follow=GR.FOLLOW),
            CH.persistence_figure(GR.persistence(first[feats], second[feats]), GR.FEATURES),
            CH.correlation_figure(table[feats], GR.FEATURES), CH.quality_figure(q, 3),
            CH.group_map_figure(xy, axes, out["members"], out["groups"], highlight=["S0"]),
            CH.group_profile_figure(out["centre"], table, out["members"], GR.FEATURES),
            CH.stock_figure("S0", sc, table, out["members"], out["centre"], GR.FEATURES)]
    assert all(f.layout.legend.y < 0 for f in figs)
    path = GR.save(out, table, meta={"groups": 3}, name="test", folder=tmp_path)
    back = GR.load(path)
    assert [g["stocks"] for g in back["groups"]] == list(out["groups"]["stocks"])
    for g in back["groups"]:
        assert len(g["tune"]) == min(2, g["stocks"])
        assert len(g["tune"]) + len(g["check"]) == g["stocks"] == len(g["members"])
    assert set(back["groups"][0]["profile"]) == set(feats)
    assert GR.files(tmp_path) == [path]
