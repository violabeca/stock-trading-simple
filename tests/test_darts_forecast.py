"""darts_forecast: the grid, the inputs' timing, the walk-forward alignment, the trading rule
and the research helpers.

Synthetic 1-minute panels only (no DB). The Darts tests are skipped when the optional ``darts``
extra is not installed.
"""

import importlib.util
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

S = strategies.load("darts_forecast")
M = S.signals.__globals__
_spec = importlib.util.spec_from_file_location(
    "darts_forecast_research", strategies.STRATEGIES_DIR / "darts_forecast" / "research.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
#: A small, fast forecaster (30-minute grid: 13 buckets a session).
SMALL = dict(model="gbm", bar_minutes=30, horizon=1, lookback=4, cov_lags=2, train_days=10,
             val_days=0, retrain_minutes=390, n_epochs=1, finetune_epochs=0,
             recency_half_life=0.0, options="none", use_daily=True, use_market=True, vol_days=5,
             gbm_iter=20, device="cpu", seed=0, overnight=False, expanding=False)


@pytest.fixture(autouse=True)
def _one_thread():
    """Run every test on one OpenMP thread: the tiny gradient-boosting fits here are faster that
    way, and several threads on a busy machine slow them down several-fold."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        yield


def minute_panel(n_days: int = 30, names=("AAA", "BBB"), seed: int = 0, drift=None,
                 start: str = "2025-07-01") -> Panel:
    """``n_days`` sessions of 390 one-minute bars (09:30-15:59 New York) from ``start``; from the
    default start, session 2 (2025-07-03) is a half-day in ``strategy.HALF_DAYS``."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    open_ = [(d + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York") for d in days]
    idx = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(o, periods=390, freq="min").tz_convert("UTC") for o in open_]))
    m, n = len(names), len(idx)
    r = rng.normal(0, 0.0008, (n, m))
    if drift is not None:
        r += np.repeat(drift, 390, axis=0)
    c = 100 * np.exp(np.cumsum(r, axis=0))
    o = np.vstack([c[:1], c[:-1]]) * (1 + rng.normal(0, 1e-4, (n, m)))
    h, lo = np.maximum(c, o) * 1.0003, np.minimum(c, o) * 0.9997
    v = rng.integers(1000, 5000, (n, m)).astype(float)
    f = lambda x: pd.DataFrame(x, index=idx, columns=list(names))  # noqa: E731
    return Panel(close=f(c), open_=f(o), high=f(h), low=f(lo), volume=f(v), bar="1m",
                 tz="America/New_York")


def _copy(p: Panel) -> Panel:
    return replace(p, close=p.close.copy(), open_=p.open_.copy(), high=p.high.copy(),
                   low=p.low.copy(), volume=p.volume.copy())


def _bump_after(p: Panel, row: int) -> Panel:
    """A copy with every bar from ``row`` on moved (prices x1.5, volume x7)."""
    q = _copy(p)
    for frame in (q.close, q.open_, q.high, q.low):
        frame.iloc[row:] *= 1.5
    q.volume.iloc[row:] *= 7
    return q


# --- the grid and the inputs ---------------------------------------------------------------------
def test_grid_buckets_align_to_the_open_and_end_on_their_last_minute():
    p = minute_panel(n_days=2)
    g = M["grid"](p, 30)
    assert len(g["start"]) == 2 * 13 and g["n_pos"] == 13
    assert (g["end"] - g["start"] == 29).all() and g["regular"].all()
    assert g["pos"][:13].tolist() == list(range(13))
    assert g["left"][:13].tolist() == list(range(12, -1, -1))
    local = p.close.index[g["end"][0]].tz_convert("America/New_York")
    assert (local.hour, local.minute) == (9, 59)  # the 09:30 bucket ends on the 09:59 bar
    assert g["first"][[0, 13]].all() and g["sess_first"].tolist() == [0, 13]
    assert np.flatnonzero(g["last_1m"]).tolist() == [389, 779]


def test_half_day_minutes_after_the_close_are_ignored():
    p = minute_panel(n_days=4)  # 2025-07-03 closes at 13:00
    g = M["grid"](p, 30)
    half = g["code"] == 2
    assert g["regular"][half].tolist() == [True] * 7 + [False] * 6
    assert g["left"][half].tolist() == [6, 5, 4, 3, 2, 1, 0] + [-1] * 6
    last = p.close.index[g["last_1m"]].tz_convert("America/New_York")
    assert [(t.hour, t.minute) for t in last] == [(15, 59), (15, 59), (12, 59), (15, 59)]
    f = M["intraday_features"](p, g, vol_days=3)
    for key in ("ret", "clv", "vwap_dist"):
        assert np.isfinite(f[key][half][:7]).all() and np.isnan(f[key][half][7:]).all(), key


def test_an_hourly_grid_has_seven_buckets_and_a_half_hour_close():
    p = minute_panel(n_days=4)  # 2025-07-03 (session 2) closes at 13:00
    g = M["grid"](p, 60)
    sizes = g["end"][:7] - g["start"][:7] + 1
    assert g["n_pos"] == 7 and sizes.tolist() == [60] * 6 + [30] and g["pos"][:7].tolist() == list(
        range(7))
    local = p.close.index[g["end"][:7]].tz_convert("America/New_York")
    assert [t.strftime("%H:%M") for t in local] == ["10:29", "11:29", "12:29", "13:29", "14:29",
                                                    "15:29", "15:59"]
    half = g["code"] == 2
    assert g["regular"][half].tolist() == [True] * 4 + [False] * 3
    assert g["left"][half].tolist() == [3, 2, 1, 0, -1, -1, -1]
    # 35 buckets later is the same clock time 5 sessions later, half-day or not
    assert (g["pos"][35:] == g["pos"][:-35]).all() and (g["code"][35:] - g["code"][:-35] == 5).all()


def test_an_overnight_target_is_close_to_close_in_typical_moves():
    p = minute_panel(n_days=25, seed=21, start="2025-07-07")  # no half-day in these sessions
    g = M["grid"](p, 60)
    b = g["sess_first"][22]
    for frame in (p.close, p.open_, p.high, p.low):  # a 2% gap up at session 22's open
        frame.iloc[g["start"][b]:] *= 1.02
    kw = dict(bar_minutes=60, horizon=7, options="none", vol_days=5)
    on = M["build_inputs"](p, **kw, overnight=True)
    off = M["build_inputs"](p, **kw, overnight=False)
    # the inputs do not change: bucket returns start at each session's open
    assert np.array_equal(on["y"], off["y"], equal_nan=True)
    assert all(np.array_equal(on["past"][k], off["past"][k], equal_nan=True) for k in on["past"])
    # the label 3 buckets into session 22 belongs to the decision one session (7 buckets) before:
    # the close-to-close return, gap included, over the typical one-session move before session 21
    lab, dec = b + 3, b + 3 - 7
    close = p.close.to_numpy()[g["end"]]
    daily = p.close.groupby(g["codes_1m"]).last()
    vol = np.log(daily / daily.shift(1)).rolling(5, min_periods=3).std().shift(1).iloc[21, 0]
    assert on["tscale"][dec, 0] == pytest.approx(vol)  # horizon = one session: sqrt(7 / 7) = 1
    assert on["target"][lab, 0] == pytest.approx(math.log(close[lab, 0] / close[dec, 0]) / vol)
    assert on["valid"][lab, 0] and not off["valid"][lab, 0]  # intraday labels stay in a session
    assert np.allclose(on["close"], close)
    assert np.array_equal(off["tscale"], off["scale"], equal_nan=True)


def test_an_overnight_label_across_a_missing_hour_is_not_a_target():
    p = minute_panel(n_days=20, seed=22)
    g = M["grid"](p, 60)
    hole = np.ones(len(p.close), bool)
    hole[g["start"][g["sess_first"][12] + 2]: g["end"][g["sess_first"][12] + 2] + 1] = False
    q = Panel(close=p.close[hole], open_=p.open_[hole], high=p.high[hole], low=p.low[hole],
              volume=p.volume[hole], bar="1m", tz=p.tz)  # 11:30-12:29 of session 12 is gone
    inp = M["build_inputs"](q, bar_minutes=60, horizon=7, options="none", vol_days=5,
                            overnight=True)
    f = inp["g"]["sess_first"]
    assert np.bincount(inp["g"]["code"])[12] == 6
    # labels whose 7 buckets cross the hole would end an hour late: they are dropped
    assert not inp["valid"][f[12] + 2: f[13] + 2, 0].any()
    assert inp["valid"][f[12] + 1, 0] and inp["valid"][f[13] + 3, 0]


def test_an_overnight_decision_after_a_half_days_close_is_not_a_target():
    p = minute_panel(n_days=20, start="2025-11-17")  # 2025-11-28 (session 9) closes at 13:00
    inp = M["build_inputs"](p, bar_minutes=60, horizon=7, options="none", vol_days=5,
                            overnight=True)
    g, f = inp["g"], inp["g"]["sess_first"]
    assert inp["valid"][f[10] + 3, 0] and not inp["valid"][f[10] + 5, 0]  # decided 14:30: closed
    assert not inp["valid"][f[9] + 5, 0]  # and no label ends after the 13:00 close
    assert np.isfinite(inp["target"][f[10] + 5, 0])  # its price exists; it just is no decision
    assert not g["regular"][f[9] + 5]


def test_bucket_returns_exclude_the_gap_and_intraday_inputs_are_causal():
    p = minute_panel(n_days=25, seed=1)
    g = M["grid"](p, 30)
    f = M["intraday_features"](p, g, vol_days=5)
    b = 13 * 20  # first bucket of session 20
    first_open, end_close = p.open_.iloc[g["start"][b], 0], p.close.iloc[g["end"][b], 0]
    assert f["ret"][b, 0] == pytest.approx(math.log(end_close / first_open))
    q = _bump_after(p, g["end"][b + 3] + 1)  # nothing about buckets <= b + 3 may move
    f2 = M["intraday_features"](q, M["grid"](q, 30), vol_days=5)
    for key in ("ret", "rv", "range", "clv", "flow", "rvol", "ret_open", "vwap_dist", "mkt",
                "scale"):
        np.testing.assert_allclose(f[key][: b + 4], f2[key][: b + 4], equal_nan=True, err_msg=key)


def test_daily_inputs_use_only_previous_sessions_and_the_gap_is_known_at_the_open():
    p = minute_panel(n_days=30, seed=2)
    g = M["grid"](p, 30)
    d = M["daily_features"](p, g)
    b = g["sess_first"][25]
    q = _copy(p)
    for frame in (q.close, q.high, q.low):
        frame.iloc[g["start"][b] + 1:] *= 0.8  # change today after its first bar
    d2 = M["daily_features"](q, M["grid"](q, 30))
    for key in d:
        np.testing.assert_allclose(d[key][b], d2[key][b], equal_nan=True, err_msg=key)
    assert np.allclose(d["prev_ret"][b:b + 13, 0], d["prev_ret"][b, 0])  # one value a session
    prev_close = p.close.iloc[g["start"][b] - 1, 0]
    today_open = p.open_.iloc[g["start"][b], 0]
    r = np.log(p.close.groupby(g["codes_1m"]).last().iloc[:, 0]).diff()
    vol20 = r.rolling(20, min_periods=10).std().shift(1).iloc[25]
    assert d["gap"][b, 0] == pytest.approx(math.log(today_open / prev_close) / vol20)


def test_gap_is_missing_when_the_sessions_first_bar_is_missing():
    p = minute_panel(n_days=25, seed=14)
    b = M["grid"](p, 30)["sess_first"][22]
    row = M["grid"](p, 30)["start"][b]
    for frame in (p.open_, p.high, p.low, p.close):
        frame.iloc[row, 0] = np.nan  # AAA has no 09:30 bar
    d = M["daily_features"](p, M["grid"](p, 30))
    assert np.isnan(d["gap"][b, 0]) and np.isfinite(d["gap"][b, 1])


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_option_inputs_land_on_their_session_and_flag_missing_sources(tz):
    p = minute_panel(n_days=5)
    days = pd.bdate_range("2025-07-01", periods=5, tz=tz)
    skew = pd.DataFrame({"AAA": [0.1, 0.2, np.nan, 0.4, 0.5], "BBB": 0.0}, index=days)
    p.features = {"opt_skew": skew}
    inp = M["build_inputs"](p, bar_minutes=30, horizon=1, options="iv", vol_days=3)
    names = inp["past_names"]
    a = inp["past"]["AAA"][:, names.index("opt_skew")]
    assert np.allclose(a[13:26], 0.2) and np.isnan(a[26:39]).all() and np.allclose(a[52:], 0.5)
    flag = inp["past"]["AAA"][:, names.index("has_dolt_iv")]
    assert flag[13] == 1.0 and flag[30] == 0.0  # day 3 has no implied-vol input at all
    assert (inp["past"]["AAA"][:, names.index("has_dolt_hv")] == 0).all()  # never attached
    assert "has_oi" not in names and inp["options_attached"]  # options="iv" never asks for oi


def test_the_clock_continues_past_the_panel_end_on_business_days():
    g = M["grid"](minute_panel(n_days=2, start="2025-07-17"), 60)  # Thursday, Friday
    tf = M["time_features"](g, extra=8)
    # the 8 points after Friday's last bucket are Monday 09:30 ... and Tuesday's first bucket
    assert np.allclose(tf[14:, 3], [0.0] * 7 + [0.25])  # weekday / 4: Monday, then Tuesday
    assert np.allclose(tf[14:, 2] * 7, [*range(7), 0])  # the session clock restarts
    assert (tf[14:21, 4] == 0).all()  # the week after July's third Friday (07-18) is no expiry week


def test_the_clock_past_the_panel_end_skips_exchange_holidays():
    g = M["grid"](minute_panel(n_days=1, start="2025-07-03"), 60)  # Thursday; Friday 07-04 closed
    tf = M["time_features"](g, extra=7)
    assert np.allclose(tf[7:, 3], 0.0)  # the next session is Monday 07-07, not Friday


def test_opex_week_comes_from_the_calendar():
    dates = pd.to_datetime(["2025-07-11", "2025-07-14", "2025-07-18", "2025-07-21",
                            "2026-06-15", "2026-06-18", "2026-06-22"]).date
    tf = M["time_features"]({"pos": np.zeros(7, int), "n_pos": 13, "date": np.array(dates)})
    # 2026-06-19, June's third Friday, is a holiday (Juneteenth): its week still counts
    assert tf[:, 4].tolist() == [0, 1, 1, 0, 1, 1, 0]


def test_target_is_the_backward_sum_and_never_pairs_two_sessions():
    p = minute_panel(n_days=25, seed=3)
    inp = M["build_inputs"](p, bar_minutes=30, horizon=3, options="none", vol_days=5)
    b = 13 * 21 + 5
    assert inp["target"][b, 0] == pytest.approx(inp["y"][b - 2: b + 1, 0].sum())
    first = inp["g"]["sess_first"][22]
    # the label at p belongs to the decision at p - 3, which must be in the same session
    assert inp["valid"][first - 1, 0] and not inp["valid"][first: first + 3, 0].any()
    assert inp["valid"][first + 3, 0]


@pytest.mark.parametrize("horizon,overnight", [(2, False), (20, True)])
def test_seasonal_input_averages_completed_outcomes_from_the_same_clock_time(horizon, overnight):
    p = minute_panel(n_days=40, seed=5, start="2025-07-07")  # 30-minute grid: 13 buckets a session
    kw = dict(bar_minutes=30, horizon=horizon, options="none", vol_days=5, overnight=overnight)
    on, off = (M["build_inputs"](p, **kw, seasonal=x) for x in (True, False))
    assert on["past_names"] == [*off["past_names"], "seas"] and on["groups"]["seas"] == "seasonal"
    k = on["past_names"].index("seas")
    g, H = on["g"], horizon
    s = g["sess_first"][35] + 4                          # a decision: session 35, bucket 4
    back = max(1, math.ceil(H / 13))                     # 20 bars ahead: 2 sessions must pass
    rows = [g["sess_first"][35 - j] + 4 for j in range(back, back + 20)]
    outcomes = [on["target"][r + H, 0] for r in rows if on["valid"][r + H, 0]]
    assert len(outcomes) >= 5 and all(r + H <= s for r in rows)   # every outcome known at s
    assert on["past"]["AAA"][s, k] == pytest.approx(np.mean(outcomes), rel=1e-5)
    # nothing after the decision bar moves it
    later = M["build_inputs"](_bump_after(p, g["end"][s] + 1), **kw, seasonal=True)
    np.testing.assert_array_equal(on["past"]["AAA"][: s + 1, k], later["past"]["AAA"][: s + 1, k])


def test_schedule_retrains_daily_weekly_once_or_intraday():
    g = M["grid"](minute_panel(n_days=30), 30)
    daily = M["schedule"](g, train_days=5, retrain_minutes=390, warmup=3)
    assert daily.tolist() == g["sess_first"][8:].tolist()
    weekly = M["schedule"](g, train_days=5, retrain_minutes=1950, warmup=3)
    assert weekly.tolist() == g["sess_first"][8::5].tolist()
    once = M["schedule"](g, train_days=5, retrain_minutes=0, warmup=3)
    assert once.tolist() == [g["sess_first"][8]]
    hourly = M["schedule"](g, train_days=5, retrain_minutes=60, warmup=3)
    assert (np.diff(hourly) == 2).all() and hourly[0] == g["sess_first"][8]


def test_p_up_reads_the_quantiles_and_handles_ties():
    q = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 5.0],
                  [-5.0, -4.0, -3.0, -2.0, -1.0], [-1.0, 0.0, 1.0, 2.0, 3.0],
                  [0.0, 0.0, 0.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    tied = 1 - np.mean([0.1, 0.25, 0.5])  # tied quantiles share the mean of their levels
    assert M["_p_up"](q).tolist() == pytest.approx([0.5, 0.95, 0.05, 0.75, tied, 0.95])


# --- the trading rule ----------------------------------------------------------------------------
def _fake_forecast(p: Panel, bar: int = 30, horizon: int = 1) -> dict:
    inp = M["build_inputs"](p, bar_minutes=bar, horizon=horizon, options="none", vol_days=5)
    n, m = inp["target"].shape
    nan = lambda: np.full((n, m), np.nan)  # noqa: E731
    return {"inputs": inp, "edge": nan(), "p_up": nan(), "q10": nan(), "q90": nan(),
            "fit": np.zeros((n, m))}


def _bp(fc: dict, b: int, j: int, bp: float) -> None:
    fc["edge"][b, j] = bp / (fc["inputs"]["scale"][b, j] * 1e4)


def _rule(p: Panel, fc: dict, **kw) -> dict:
    sig = M["trade_frames"](p, fc, **{"health_days": 0, "edge_bp": 10.0, **kw})
    return {k: v.to_numpy() for k, v in sig.items()}


def test_trade_frames_fill_on_the_next_minute_hold_stop_and_size():
    p = minute_panel(n_days=25, seed=4)
    fc = _fake_forecast(p)
    g, scale = fc["inputs"]["g"], fc["inputs"]["scale"]
    b = g["sess_first"][22] + 4  # the 11:30 bucket, 8 buckets left
    _bp(fc, b, 0, 50.0)  # +50 bp forecast for AAA
    _bp(fc, b, 1, -50.0)  # -50 bp forecast for BBB
    sig = _rule(p, fc, hold_bars=2, stop_vol=2.0, risk_per_trade=0.005, max_positions=4)
    fill = g["end"][b] + 1
    assert np.flatnonzero(sig["entries"][:, 0]).tolist() == [fill]
    assert not sig["entries"][:, 1].any()
    assert np.flatnonzero(sig["short_entries"][:, 1]).tolist() == [fill]
    assert np.flatnonzero(sig["exits"][:, 0]).tolist() == [fill + 60]  # two 30-minute bars
    stop = 2.0 * scale[b, 0] * math.sqrt(2)
    assert sig["sl_stop"][fill, 0] == pytest.approx(stop)
    assert sig["cash_frac"][fill, 0] == pytest.approx(min(0.25, 0.005 / stop))
    assert np.isfinite(sig["cash_frac"]).all()
    # a forecast too early in the session, or too close to the close for the hold, never trades
    early, late = g["sess_first"][23], g["sess_first"][23] + 11
    _bp(fc, early, 0, 50.0)
    _bp(fc, late, 0, 50.0)
    assert _rule(p, fc, hold_bars=2, skip_open=45)["entries"][:, 0].sum() == 1


def _trades(p: Panel, fc: dict, **kw) -> pd.DataFrame:
    sig = M["trade_frames"](p, fc, **{"health_days": 0, "edge_bp": 10.0, **kw})
    pf = bt.run(p, sig, fees=0.0, slippage=0.0)
    t = bt.trade_records(pf)
    rows = pd.Series(np.arange(len(p.close)), index=p.close.index)
    return t.assign(entry=rows[t["entry_ts"]].to_numpy(), exit=rows[t["exit_ts"]].to_numpy())


def test_a_new_same_side_forecast_extends_the_hold():
    pytest.importorskip("vectorbt")
    p = minute_panel(n_days=25, seed=4)
    fc = _fake_forecast(p)
    g = fc["inputs"]["g"]
    b = g["sess_first"][22] + 3
    _bp(fc, b, 0, 50.0)
    _bp(fc, b + 1, 0, 50.0)
    t = _trades(p, fc, hold_bars=2, stop_vol=0.0)
    assert t[["entry", "exit"]].values.tolist() == [[g["end"][b] + 1, g["end"][b + 1] + 1 + 60]]


def test_a_decision_after_a_stop_opens_a_new_trade():
    pytest.importorskip("vectorbt")
    p = minute_panel(n_days=25, seed=4)
    fc = _fake_forecast(p)
    g = fc["inputs"]["g"]
    b = g["sess_first"][22] + 3
    f0, f1 = g["end"][b] + 1, g["end"][b + 1] + 1
    q = _copy(p)
    for frame in (q.close, q.open_, q.high, q.low):  # AAA drops 5% ten minutes into the trade
        frame.iloc[f0 + 10: f1, 0] *= 0.95
    _bp(fc, b, 0, 50.0)
    _bp(fc, b + 1, 0, 50.0)
    t = _trades(q, fc, hold_bars=2, stop_vol=2.0)
    assert t["entry"].tolist() == [f0, f1] and t["exit"].tolist()[0] == f0 + 10  # stopped, back in
    assert t["exit"].tolist()[1] == f1 + 60


def test_an_opposite_forecast_closes_then_reverses_one_minute_later():
    p = minute_panel(n_days=25, seed=4)
    fc = _fake_forecast(p)
    g = fc["inputs"]["g"]
    b = g["sess_first"][22] + 3
    _bp(fc, b, 0, 50.0)
    _bp(fc, b + 1, 0, -50.0)
    sig = _rule(p, fc, hold_bars=2)
    f0, f1 = g["end"][b] + 1, g["end"][b + 1] + 1
    assert np.flatnonzero(sig["entries"][:, 0]).tolist() == [f0]
    assert np.flatnonzero(sig["exits"][:, 0]).tolist() == [f1]
    assert np.flatnonzero(sig["short_entries"][:, 0]).tolist() == [f1 + 1]
    assert np.flatnonzero(sig["short_exits"][:, 0]).tolist() == [f1 + 60]


def test_the_book_is_flat_by_the_close():
    p = minute_panel(n_days=25, seed=4)
    fc = _fake_forecast(p)
    g = fc["inputs"]["g"]
    b = g["sess_first"][22] + 10  # the 14:30 bucket: exactly two 30-minute bars left
    _bp(fc, b, 0, 50.0)
    sig = _rule(p, fc, hold_bars=2)
    last = np.flatnonzero(g["last_1m"] & (g["codes_1m"] == 22))[0]
    assert np.flatnonzero(sig["entries"][:, 0]).tolist() == [g["end"][b] + 1]
    assert np.flatnonzero(sig["exits"][:, 0]).tolist() == [last]  # the session's last bar
    # without the EOD exit the hold must END before the close, so this decision is skipped
    assert not _rule(p, fc, hold_bars=2, eod_exit=False)["entries"].any()


def test_health_gate_reads_only_completed_sessions():
    p = minute_panel(n_days=30, seed=5)
    fc = _fake_forecast(p)
    inp = fc["inputs"]
    real = np.full_like(inp["target"], np.nan)
    real[:-1] = inp["target"][1:]
    fc["edge"] = np.where((inp["g"]["code"] < 25)[:, None], real, -real)  # right, then wrong
    h = M["health"](fc, days=3)
    assert h[25] > 0.9 and h[26] > 0.0 and h[28] < -0.9  # session 25 still sees only the past
    assert np.isnan(h[0])


def test_the_health_gate_waits_for_multi_day_outcomes():
    p = minute_panel(n_days=32, seed=5)
    inp = M["build_inputs"](p, bar_minutes=60, horizon=7, options="none", vol_days=5,
                            overnight=True)
    real = np.full_like(inp["target"], np.nan)
    real[:-7] = inp["target"][7:]
    fc = {"inputs": inp, "edge": np.where((inp["g"]["code"] < 25)[:, None], real, -real)}
    h = M["health"](fc, days=4)
    # session 25's decisions are only known one session later, so 26 still reads good forecasts
    assert h[26] > 0.9 and h[29] < -0.9


def test_valid_rejects_impossible_combinations():
    base = dict(S.params)
    assert S.valid(base)
    assert not S.valid({**base, "bar_minutes": 7})
    assert not S.valid({**base, "model": "lstm"})
    assert not S.valid({**base, "retrain_minutes": 700})  # neither intraday nor whole sessions
    assert not S.valid({**base, "val_days": 40})
    assert not S.valid({**base, "p_min": 0.4})
    hourly = {**base, "bar_minutes": 60, "horizon": 35, "overnight": True, "expanding": True}
    assert M["forecast_valid"](hourly) and not S.valid(hourly)  # forecastable, not tradable yet
    assert S.valid({**base, "bar_minutes": 60, "horizon": 1})
    assert M["forecast_valid"]({**hourly, "retrain_minutes": 120})
    assert not M["forecast_valid"]({**hourly, "retrain_minutes": 90})
    assert not M["forecast_valid"]({**hourly, "overnight": False})  # no label inside a session
    assert not M["forecast_valid"]({**hourly, "expanding": False, "train_days": 8})  # 56 buckets


# --- the research helpers ------------------------------------------------------------------------
class _FakeOptions:
    """Stands in for burst_scan/options_free.py: one implied-vol history for AAA, with a copy of
    07-03 filed on the 07-04 holiday (as DoltHub does)."""

    IV = {"2025-07-01": 0.30, "2025-07-02": 0.32, "2025-07-03": 0.35, "2025-07-04": 0.35,
          "2025-07-14": 0.50}

    @staticmethod
    def load_history(source, store):
        if source != "dolthub_chain":
            return pd.DataFrame()
        return pd.DataFrame({"date": list(_FakeOptions.IV), "underlying": "AAA"})

    @staticmethod
    def iv_features(chain):
        return chain.assign(atm_iv=chain["date"].map(_FakeOptions.IV), skew=0.0, term=0.0)


def test_option_frames_are_point_in_time_never_stale_and_score_warm(monkeypatch):
    monkeypatch.setattr(R, "_OF", _FakeOptions)
    sessions = pd.to_datetime(["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-07",
                               "2025-07-08", "2025-07-09", "2025-07-14", "2025-07-15"]).date
    f = R.option_frames(sessions, ["AAA", "BBB"])
    assert str(f["opt_atm_iv"].index.tz) == "UTC"
    iv, d = (f[k]["AAA"].set_axis(f[k].index.strftime("%m-%d")) for k in ("opt_atm_iv",
                                                                        "opt_d_atm_iv"))
    # a day's end-of-day value is first used the NEXT session, and never more than 5 days later
    assert np.isnan(iv["07-01"]) and iv["07-02"] == 0.30 and iv["07-03"] == 0.32
    assert iv["07-07"] == iv["07-08"] == 0.35 and np.isnan(iv["07-09"]) and np.isnan(iv["07-14"])
    assert iv["07-15"] == 0.50 and f["opt_atm_iv"]["BBB"].isna().all()
    # the holiday copy is dropped, and the day-over-day change never spans the 11-day hole
    assert d["07-03"] == pytest.approx(0.02) and d["07-07"] == pytest.approx(0.03)
    assert np.isnan(d["07-15"])
    # tz-aware frames are what the engine's warm-start cut needs
    p = replace(minute_panel(n_days=8), features=f)
    window, _ = bt._score_window(p, {}, "2025-07-08")
    assert window.features["opt_atm_iv"].index[0].strftime("%m-%d") == "07-08"


def test_an_overnight_table_scores_multi_day_forecasts_inside_the_window():
    p = minute_panel(n_days=30, seed=24, start="2025-07-07")
    inp = M["build_inputs"](p, bar_minutes=60, horizon=14, options="none", vol_days=5,
                            overnight=True)
    n, m = inp["target"].shape
    fc = {"inputs": inp, "edge": np.ones((n, m)), "p_up": np.full((n, m), np.nan),
          "fit": np.zeros((n, m))}
    t = R.table(fc)
    days = sorted(t["date"].unique())
    assert len(t) and (t["block"] >= 0).all() and t["minute"].max() == 960  # 15:30 closes 16:00
    # a 2-session horizon: errors cluster by pairs of sessions
    assert t.groupby("block")["date"].nunique().max() == 2
    # the window keeps a decision only when its outcome (2 sessions later) is inside it too
    w = R.table(fc, (days[0], days[20]))
    assert w["date"].max() == days[18] and len(w) < len(t)


def test_dividend_adjustment_adds_the_dividend_back_and_keeps_intraday_returns():
    p = minute_panel(n_days=5, seed=25, start="2025-07-07")
    days = pd.bdate_range("2025-07-07", periods=5)
    f = pd.DataFrame({"AAA": [0.99, 0.99, 1.0, 1.0, 1.0], "BBB": 1.0}, index=days)  # AAA ex day 2
    q = R.adjust_dividends(p, f)
    i0 = 2 * 390  # the first minute of day 2
    raw = math.log(p.open_["AAA"].iloc[i0] / p.close["AAA"].iloc[i0 - 1])
    adj = math.log(q.open_["AAA"].iloc[i0] / q.close["AAA"].iloc[i0 - 1])
    assert adj - raw == pytest.approx(-math.log(0.99))  # the 1% dividend is back in the return
    same_day = np.r_[False, np.diff(np.arange(len(p.close)) // 390) == 0]
    r_p, r_q = (np.log(x / x.shift(1)).to_numpy()[same_day] for x in (p.close, q.close))
    np.testing.assert_allclose(r_q, r_p)  # every intraday return is unchanged
    assert np.allclose(q.close["BBB"], p.close["BBB"]) and (q.volume == p.volume).all().all()
    # a later dividend (factor < 1 on the last session too) leaves the last session's prices traded
    late = R.adjust_dividends(p, f * 0.98)
    assert np.allclose(late.close.iloc[-390:], p.close.iloc[-390:])
    assert np.allclose(late.close, q.close)  # and every return as before


def test_load_reads_through_the_last_session(monkeypatch):
    seen = {}
    monkeypatch.setattr(R, "load_panel", lambda *a, **kw: seen.update(start=a[3], end=a[4]))
    settings = SimpleNamespace(universe={"DirectionFive": ["MU"]}, exchanges=None)
    R.load(settings, None, options=False)
    assert str(seen["start"]) == "2025-06-18" and str(seen["end"]) == "2026-02-01"
    R.load(settings, None, end="all", options=False)
    assert seen["end"] is None


def test_cache_key_holds_every_forecast_knob_and_the_panel_content():
    p = minute_panel(n_days=2)
    base = dict(S.params)
    key = R._key(base, p)
    for k in M["MODEL_KEYS"]:
        v = base[k]
        other = (not v) if isinstance(v, bool) else (v + 1 if isinstance(v, int | float)
                                                     else v + "x")
        assert R._key({**base, k: other}, p) != key, k
    q = _copy(p)
    q.close.iloc[100, 1] *= 1.0001
    assert R._key(base, q) != key and R._key(base, p) == key
    assert f"-c{R.CODE_VERSION[:6]}-" in key  # what prune_cache reads


def test_sign_surrogate_keeps_every_move_size_and_a_late_starting_name():
    p = minute_panel(n_days=3, seed=13)
    for frame in (p.close, p.open_, p.high, p.low):
        frame.iloc[:50, 1] = np.nan  # BBB starts late
    q = R.sign_surrogate(p, seed=1)
    for name in ("AAA", "BBB"):
        r, rq = (np.log(x[name] / x[name].shift(1)).abs().iloc[51:] for x in (p.close, q.close))
        np.testing.assert_allclose(rq, r, rtol=1e-9, atol=1e-12)
    assert q.close["BBB"].iloc[:50].isna().all()
    assert q.open_["BBB"].iloc[50] == pytest.approx(p.open_["BBB"].iloc[50])
    gone = q.close.isna()
    assert ((q.high >= np.maximum(q.open_, q.close) - 1e-9) | gone).all().all()
    assert ((q.low <= np.minimum(q.open_, q.close) + 1e-9) | gone).all().all()


def test_null_z_uses_standard_errors_and_the_scorecard_rounds_deep(capsys):
    null = pd.DataFrame({"ic": [0.01, -0.01, 0.02], "t": [1.0, -1.0, 2.0]})
    z = R.null_z({"ic": 0.05, "t": 5.0}, null)
    assert z == pytest.approx((0.05 - 0.02 / 3) / math.sqrt(0.01 ** 2 + 0.01 ** 2 / 3))
    unlucky = pd.DataFrame({"ic": [-0.01, -0.02, -0.01], "t": [-1.0, -2.0, -1.0]})
    assert R.null_z({"ic": 0.05, "t": 5.0}, unlucky) == 5.0  # never above the t against 0
    R.scorecard({"a": {"b": 0.123456, "c": [np.float64(1 / 3), np.nan]}, "n": np.int64(3)})
    assert capsys.readouterr().out.strip() == (
        'SCORECARD {"a": {"b": 0.1235, "c": [0.3333, null]}, "n": 3}')


# --- Darts (optional extra) ----------------------------------------------------------------------
def test_darts_forecast_is_aligned_learns_a_planted_signal_and_finds_none_in_noise():
    pytest.importorskip("darts")
    rng = np.random.default_rng(7)
    n_days = 45
    drift = rng.choice([-1.0, 1.0], size=(n_days, 2)) * 0.0004
    planted = minute_panel(n_days=n_days, seed=8, drift=drift)
    days = pd.bdate_range("2025-07-01", periods=n_days)
    # a daily "option" input that reveals each day's direction, attached for that day
    planted.features = {"opt_skew": pd.DataFrame(np.sign(drift), index=days,
                                                 columns=["AAA", "BBB"])}
    kw = {**SMALL, "options": "iv", "use_daily": False, "gbm_iter": 40, "retrain_minutes": 1950}
    for panel, lo, hi in ((planted, 0.5, 1.0), (minute_panel(n_days=n_days, seed=9), -0.12, 0.12)):
        fc = M["forecast"](panel, **kw)
        inp = fc["inputs"]
        real = np.full_like(inp["target"], np.nan)
        real[:-1] = inp["target"][1:]
        ok = np.isfinite(fc["edge"]) & np.isfinite(real)
        ic = pd.Series(fc["edge"][ok]).rank().corr(pd.Series(real[ok]).rank())
        assert lo < ic < hi, ic
    log = fc["log"]  # every fit ends before the session it forecasts
    assert (pd.to_datetime(log["train_last"]) < pd.to_datetime(log["session"])).all()
    assert np.flatnonzero(np.isfinite(fc["edge"][:, 0]))[0] == fc["points"][0]


def test_forecast_ignores_everything_after_its_decision_bar():
    pytest.importorskip("darts")
    p = minute_panel(n_days=34, seed=11)
    kw = {**SMALL, "horizon": 2, "train_days": 8}
    fc = M["forecast"](p, **kw)
    s = fc["inputs"]["g"]["sess_first"][31] + 5
    fc2 = M["forecast"](_bump_after(p, fc["inputs"]["g"]["end"][s] + 1), **kw)
    assert np.isfinite(fc["edge"][s]).all()
    np.testing.assert_array_equal(fc["edge"][: s + 1], fc2["edge"][: s + 1])
    assert not np.array_equal(fc["edge"][s + 1:], fc2["edge"][s + 1:], equal_nan=True)


def test_an_overnight_forecast_ignores_everything_after_its_decision_bar():
    pytest.importorskip("darts")
    p = minute_panel(n_days=40, seed=31)
    kw = {**SMALL, "bar_minutes": 60, "horizon": 35, "overnight": True, "expanding": True,
          "lookback": 7, "cov_lags": 3, "retrain_minutes": 1950}
    fc = M["forecast"](p, **kw)
    g = fc["inputs"]["g"]
    s = g["sess_first"][33] + 3
    fc2 = M["forecast"](_bump_after(p, g["end"][s] + 1), **kw)
    assert np.isfinite(fc["edge"][s]).all()
    np.testing.assert_array_equal(fc["edge"][: s + 1], fc2["edge"][: s + 1])
    log = fc["log"]  # expanding: every fit starts after the warm-up and ends before its session
    assert (log["train_from"] == str(g["date"][g["sess_first"][21]])).all()
    assert (pd.to_datetime(log["train_last"]) < pd.to_datetime(log["session"])).all()


def test_an_overnight_model_learns_a_planted_multi_day_signal():
    pytest.importorskip("darts")
    n_days, names = 66, ("AAA", "BBB", "CCC")
    # 12-session up / down regimes, each name in its own phase; a daily input reveals the regime
    regime = np.stack([np.where((np.arange(n_days) + 4 * j) // 12 % 2 == 0, 1.0, -1.0)
                       for j in range(len(names))], axis=1)
    planted = minute_panel(n_days=n_days, names=names, seed=42, drift=regime * 0.00005)
    days = pd.bdate_range("2025-07-01", periods=n_days)
    planted.features = {"opt_skew": pd.DataFrame(regime, index=days, columns=list(names))}
    kw = {**SMALL, "bar_minutes": 60, "horizon": 35, "overnight": True, "expanding": True,
          "options": "iv", "use_daily": False, "lookback": 7, "cov_lags": 3, "gbm_iter": 40,
          "retrain_minutes": 1950, "train_days": 20}  # first fit after 21 + 20 sessions
    noise = minute_panel(n_days=n_days, names=names, seed=43)
    for panel, lo, hi in ((planted, 0.5, 1.0), (noise, -0.35, 0.35)):
        fc = M["forecast"](panel, **kw)
        inp = fc["inputs"]
        real = np.full_like(inp["target"], np.nan)
        real[:-35] = inp["target"][35:]
        ok = np.isfinite(fc["edge"]) & np.isfinite(real)
        ic = pd.Series(fc["edge"][ok]).rank().corr(pd.Series(real[ok]).rank())
        assert lo < ic < hi, ic


def test_signals_accepts_every_config_param():
    """The engine calls signals(panel, **params): a config knob signals() does not take crashes
    every backtest and study."""
    import inspect

    assert set(S.params) <= set(inspect.signature(S.signals).parameters)
    assert set(M["MODEL_KEYS"]) <= set(inspect.signature(S.signals).parameters)


def test_lag_history_trains_on_the_first_labels_without_reading_the_future():
    """With lag_history a window's first labels read their lagged target from before it (weight 0
    before the window), so the fit gains them; nothing after a decision bar moves its forecast."""
    pytest.importorskip("darts")
    from darts import TimeSeries  # noqa: F401  (the weights are Darts series)

    p = minute_panel(n_days=34, seed=11)
    kw = {**SMALL, "horizon": 2, "train_days": 8, "lookback": 6, "lag_history": True}
    inp = M["build_inputs"](p, bar_minutes=30, horizon=2, options="none", vol_days=5)
    lo = int(inp["g"]["sess_first"][14])
    s0 = M["series_start"](lo, 6, 2, True)
    assert s0 == lo - 7 and M["series_start"](lo, 6, 2, False) == lo
    w = M["_weights"](inp, s0, lo + 20, 0.0, first=lo)[0].values()[:, 0]
    assert (w[:7] == 0).all() and (w[7:] == inp["valid"][lo:lo + 20, 0]).all()
    on, off = M["forecast"](p, **kw), M["forecast"](p, **{**kw, "lag_history": False})
    assert not np.array_equal(on["edge"], off["edge"], equal_nan=True)   # it learned more
    s = on["inputs"]["g"]["sess_first"][31] + 5
    later = M["forecast"](_bump_after(p, on["inputs"]["g"]["end"][s] + 1), **kw)
    assert np.isfinite(on["edge"][s]).all()
    np.testing.assert_array_equal(on["edge"][: s + 1], later["edge"][: s + 1])


def test_fast_bins_take_the_stock_path_above_the_binning_subsample():
    """Above 200,000 rows scikit-learn bins a weighted random subsample: the fast path steps
    aside, so the edges stay the stock ones."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(4)
    X = rng.normal(size=(200_500, 2))
    y = X[:, 0] + rng.normal(size=len(X))
    w = (rng.random(len(X)) > 0.1).astype(float)
    hp = dict(max_iter=2, max_leaf_nodes=4, early_stopping=False, random_state=0)
    a = HistGradientBoostingRegressor(**hp).fit(X, y, sample_weight=w)
    b = M["fast_bins"](HistGradientBoostingRegressor)(**hp).fit(X, y, sample_weight=w)
    assert all(np.array_equal(p, q) for p, q in zip(
        a._bin_mapper.bin_thresholds_, b._bin_mapper.bin_thresholds_, strict=True))


def test_fast_bins_is_bit_identical():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    rng = np.random.default_rng(3)
    X = rng.normal(size=(1500, 40))
    X[:, :5] = np.round(X[:, :5], 1)                                   # few distinct values
    X[:, 5:10] = np.repeat(rng.normal(size=(30, 5)), 50, axis=0)        # constant per "session"
    X[rng.random(X.shape) < 0.02] = np.nan
    y = np.nan_to_num(X[:, 20]) + rng.normal(size=1500)
    hp = dict(max_iter=20, max_leaf_nodes=15, min_samples_leaf=40, early_stopping=False,
              random_state=0)
    for w in ((rng.random(1500) > 0.1).astype(float), rng.uniform(0.2, 1.0, 1500)):
        for base, target in ((HistGradientBoostingRegressor, y),
                             (HistGradientBoostingClassifier, (y > 0).astype(int))):
            a = base(**hp).fit(X, target, sample_weight=w)
            b = M["fast_bins"](base)(**hp).fit(X, target, sample_weight=w)
            assert all(np.array_equal(p, q) for p, q in zip(
                a._bin_mapper.bin_thresholds_, b._bin_mapper.bin_thresholds_, strict=True))
            out = "predict_proba" if hasattr(a, "predict_proba") else "predict"
            np.testing.assert_array_equal(getattr(a, out)(X), getattr(b, out)(X))


@pytest.mark.parametrize("model,extra", [("gbm_clf", {}),
                                         ("tide", {"val_days": 2, "finetune_epochs": 1}),
                                         ("tft", {"retrain_minutes": 0}),
                                         ("linear", {"seasonal": True}),
                                         ("tsmixer", {"retrain_minutes": 0})])
def test_every_model_runs_the_walk_forward(model, extra):
    pytest.importorskip("darts")
    kw = {**SMALL, "model": model, "train_days": 5, **extra}
    fc = M["forecast"](minute_panel(n_days=28, seed=12), **kw)
    ok = np.isfinite(fc["edge"])
    p_up = fc["p_up"][ok]
    assert ok.sum() >= 13
    if model == "linear":  # a point forecast: no probability, no range
        assert np.isnan(p_up).all() and np.isnan(fc["q10"][ok]).all()
    else:
        assert ((p_up >= 0) & (p_up <= 1)).all()
    log = fc["log"]
    assert (pd.to_datetime(log["train_last"]) < pd.to_datetime(log["session"])).all()
    if model in M["DEEP"]:
        assert (fc["q10"][ok] <= fc["q90"][ok]).all()
    if extra.get("finetune_epochs"):  # a full fit first, then fine-tuning on a validation slice
        assert log["fresh"].tolist() == [True] + [False] * (len(log) - 1) and len(log) > 1
        assert (log["val_days"] == 2).all()
