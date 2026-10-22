"""lead_lag research helpers: the cube, the clustered pair map and its controls, offline.

Every statistic the exploration notebook gates on is pinned here on synthetic data: a planted lead
must be found at its lag and in its direction, a roll of the leader must destroy it, the clustered
t must stay calibrated when volatility moves together (the naive one must not), a stale follower
must look like a follower close-to-close but not on the tradable target, and the Granger control
must flag only a delayed responder.
"""

import math

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel, resample_panel

_TZ = "America/New_York"


@pytest.fixture(scope="module")
def rs():
    """research.py, loaded by path the way the notebook loads it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "lead_lag_research", strategies.STRATEGIES_DIR / "lead_lag" / "research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index(days: int, bars: int, start: str = "2026-01-05") -> pd.DatetimeIndex:
    opens = [pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz=_TZ)
             for d in pd.bdate_range(start, periods=days)]
    return pd.DatetimeIndex([o + pd.Timedelta(minutes=m) for o in opens
                             for m in range(bars)]).tz_convert("UTC")


def _panel(idx, open_, close, volume, bar="1m") -> Panel:
    frame = lambda a: pd.DataFrame(np.asarray(a, float), index=idx, columns=["A", "B"])  # noqa: E731
    o, c = frame(open_), frame(close)
    return Panel(close=c, open_=o, high=np.maximum(o, c), low=np.minimum(o, c),
                 volume=frame(volume), bar=bar, tz=_TZ)


def test_cube_last_trade_target_gap_and_placeholders(rs):
    idx = _index(2, 6)
    a_o = [10, 10.1, 10.2, 10.3, 10.4, 10.5] * 2
    a_c = [10.1, 10.2, 10.3, 10.4, 10.5, 10.6] * 2
    b_o = [50, 50.5, 51, 52, 52.5, 53] * 2          # bar 2 is IBKR's flat placeholder at 51
    b_c = [50.5, 51, 51, 52.5, 53, 53] * 2
    b_v = [100, 100, 0, 100, 100, 100] * 2
    c = rs.cube(_panel(idx, np.c_[a_o, b_o], np.c_[a_c, b_c], np.c_[[100] * 12, b_v]),
                skip_open=2)
    assert c["x"].shape == (2, 6, 2) and c["present"].all()
    x, y, gap = c["x"][..., 1], c["y"][..., 1], c["gap"][..., 1]
    assert np.isnan(x[:, 0]).all()                             # overnight never enters x
    assert x[0, 2] == 0 and np.isnan(y[0, 2])                  # no print: no move, no target
    assert x[0, 3] == pytest.approx(math.log(52.5 / 51))       # the catch-up lands on the print
    assert y[0, 3] == pytest.approx(math.log(52.5 / 52))       # ... but not in its own bar
    assert gap[0, 3] == pytest.approx(math.log(52 / 51))
    assert np.isnan(c["y"][:, :2]).all()                       # skip_open blanks two minutes
    assert c["stale"][1] == pytest.approx(2 / 12) and c["stale"][0] == 0
    zero = np.c_[a_c, b_c]
    zero[4, 0] = 0.0                                           # a bad (zero) price is no print
    z = rs.cube(_panel(idx, np.c_[a_o, b_o], zero, np.c_[[100] * 12, b_v]))
    assert np.isfinite(z["x"][~np.isnan(z["x"])]).all() and np.isnan(z["y"][0, 4, 0])


def test_masking_before_resampling_opens_the_candle_on_a_real_print(rs):
    idx = _index(1, 10)
    b_o = [50, 51, 51.5, 51.8, 52, 52, 52.1, 52.2, 52.3, 52.4]
    b_c = [50, 51.5, 51.8, 52, 52.2, 52.1, 52.2, 52.3, 52.4, 52.5]
    b_v = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100]     # minute 0: placeholder at 50
    panel = _panel(idx, np.c_[b_o, b_o], np.c_[b_c, b_c], np.c_[[100] * 10, b_v])
    assert resample_panel(panel, 5).open_["B"].iloc[0] == 50   # the stale price opens it
    masked = resample_panel(rs.mask_placeholders(panel), 5)
    assert masked.open_["B"].iloc[0] == 51 and masked.open_["A"].iloc[0] == 50
    c5 = rs.cube(masked)
    assert c5["x"].shape == (1, 2, 2)
    assert c5["y"][0, 0, 1] == pytest.approx(math.log(52.2 / 51))


def test_daily_cube_is_one_session_with_weekly_clusters(rs):
    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=10, tz="UTC"))
    close = np.c_[np.linspace(10, 11, 10), np.linspace(20, 19, 10)]
    open_ = close * 0.99
    c = rs.cube(_panel(idx, open_, close, np.full((10, 2), 1e6), bar="1d"))
    assert c["x"].shape == (1, 10, 2)
    assert c["x"][0, 3, 0] == pytest.approx(math.log(close[3, 0] / close[2, 0]))  # crosses days
    assert c["gap"][0, 3, 0] == pytest.approx(math.log(open_[3, 0] / close[2, 0]))
    assert c["y"][0, 3, 0] == pytest.approx(math.log(1 / 0.99))
    assert rs.clusters(c, "week").tolist() == [0] * 5 + [1] * 5
    assert rs.rows(c, "2024-01-03", "2024-01-09") == slice(2, 6)


def test_lagged_never_crosses_a_session_but_does_cross_days(rs):
    x = np.zeros((2, 4, 1), np.float32)
    x[0, 3, 0] = 1.0                                             # the last bar of session 0
    x[1, 0, 0] = np.nan
    assert not rs.lagged(x, 1, 2)[1].any()                       # never reaches session 1
    assert rs.lagged(x, 0, 0)[0, 3, 0] == 1 and rs.lagged(x, 0, 0)[1, 0, 0] == 0   # NaN -> 0
    daily = x.reshape(1, 8, 1)
    assert rs.lagged(daily, 1, 2)[0, :, 0].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]


def test_pair_t_is_the_clustered_t_by_hand(rs):
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(30, 2)), rng.normal(size=(30, 3))
    cl = np.repeat(np.arange(6), 5)
    s = np.stack([X[cl == g].T @ Y[cl == g] for g in range(6)])
    np.testing.assert_allclose(rs.pair_t(X, Y, cl), s.sum(0) / np.sqrt((s ** 2).sum(0)),
                               rtol=1e-5)
    np.testing.assert_allclose(rs.pair_t(X, Y, cl, roll=5),
                               rs.pair_t(np.roll(X, 5, axis=0), Y, cl), rtol=1e-6)


def _planted(n_s=100, bars=100, lag=2, beta=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_s, bars, 4)).astype(np.float32)
    x[:, lag:, 1] += beta * x[:, :-lag, 0]                     # name 1 follows name 0
    return x


def test_a_planted_lead_is_found_at_its_lag_and_in_its_direction(rs):
    x = _planted()
    cl = rs.clusters({"x": x})
    T = rs.pair_t(rs.lagged(x, 1, 3), x, cl)
    assert T[0, 1] > 8 and abs(T[1, 0]) < 3
    null = ~np.eye(4, dtype=bool)
    null[0, 1] = null[1, 0] = False
    assert np.abs(T[null]).max() < 4                            # no other pair lights up
    profile = [rs.pair_t(rs.lagged(x, k, k), x, cl)[0, 1] for k in (1, 2, 3)]
    assert int(np.argmax(profile)) == 1


def test_a_session_roll_destroys_the_planted_lead(rs):
    x = _planted()
    cl = rs.clusters({"x": x})
    assert abs(rs.pair_t(rs.lagged(x, 1, 3), x, cl, roll=5 * x.shape[1])[0, 1]) < 3


def test_clustered_t_stays_calibrated_when_volatility_moves_together(rs):
    rng = np.random.default_rng(1)
    n_s, bars, n = 60, 50, 40
    vol = np.exp(0.5 * rng.normal(size=(n_s, 1, 1)))            # one level per session, shared
    x = (rng.normal(size=(n_s, bars, n)) * vol).astype(np.float32)
    X = rs.lagged(x, 1, 1)
    off = ~np.eye(n, dtype=bool)
    T = rs.pair_t(X, x, rs.clusters({"x": x}))
    Xf, Yf = X.reshape(-1, n), x.reshape(-1, n)
    naive = (Xf.T @ Yf) / np.sqrt(np.outer((Xf ** 2).sum(0), (Yf ** 2).sum(0)) / len(Xf))
    assert 0.85 <= T[off].std() <= 1.15
    assert naive[off].std() > 1.3                               # rho * sqrt(n) is not


def test_a_stale_follower_leads_close_to_close_but_not_on_the_tradable_target(rs):
    rng = np.random.default_rng(2)
    n_s, bars = 60, 100
    p = 4.0 + np.cumsum(rng.normal(0, 1e-3, size=n_s * bars)).reshape(n_s, bars)
    prev = np.c_[p[:, :1], p[:, :-1]]
    prints = rng.random((n_s, bars)) < 0.26
    prints[:, 0] = True
    b_close = np.exp(p.copy())
    b_open = np.exp(prev.copy())
    for s in range(n_s):                                        # IBKR placeholders: previous close
        for b in range(1, bars):
            if not prints[s, b]:
                b_close[s, b] = b_open[s, b] = b_close[s, b - 1]
    idx = _index(n_s, bars)
    panel = _panel(idx, np.c_[np.exp(prev).ravel(), b_open.ravel()],
                   np.c_[np.exp(p).ravel(), b_close.ravel()],
                   np.c_[np.full(n_s * bars, 100), np.where(prints, 100, 0).ravel()])
    c = rs.cube(panel)
    X, cl = rs.lagged(c["x"][..., :1], 1, 5), rs.clusters(c)
    assert rs.pair_t(X, c["x"][..., 1:], cl)[0, 0] > 5          # the naive map "finds" a lead
    assert abs(rs.pair_t(X, c["y"][..., 1:], cl)[0, 0]) < 3     # the tradable target does not


def test_granger_flags_only_the_delayed_responder(rs):
    rng = np.random.default_rng(3)
    n_s, bars = 100, 200
    m = rng.normal(size=(n_s, bars)).astype(np.float32)
    m1 = rs.lagged(m[..., None], 1, 1)[..., 0]
    e = rng.normal(size=(n_s, bars, 3))
    y = np.empty((n_s, bars, 3), np.float32)
    y[..., 0] = 0.4 * m1 + e[..., 0]                            # delayed responder
    y[..., 1] = 0.8 * m + e[..., 1]                             # synchronous
    ar = np.zeros((n_s, bars))                                  # own autocorrelation only
    for b in range(bars):
        ar[:, b] = 0.5 * m[:, b] + (0.3 * ar[:, b - 1] if b else 0.0) + e[:, b, 2]
    y[..., 2] = ar
    y[:, 0] = np.nan
    cl = rs.clusters({"x": y})
    g = rs.granger(y, [m, m1], own=rs.lagged(y, 1, 1), cl=cl)
    lag_t = g["t"][:, 2]                                        # columns: 1, m, m1, own
    assert lag_t[0] > 10 and abs(lag_t[1]) < 3 and abs(lag_t[2]) < 3
    assert rs.granger(y, [m, m1], cl=cl)["t"][2, 2] > 3         # without the own lags it would
    u = g["u"]
    assert np.isnan(u[:, 0]).all() and abs(np.nanmean(u)) < 1e-6


def test_scale_is_causal_and_takes_the_open_out(rs):
    rng = np.random.default_rng(4)
    r = rng.normal(size=(30, 20, 3)).astype(np.float32)
    r[:, 0] *= 5                                                # the open moves 5x more
    full = rs.scale(r, days=5, fit=slice(0, 12))
    np.testing.assert_allclose(full[:18], rs.scale(r[:18], days=5, fit=slice(0, 12)), rtol=1e-6)
    assert np.isnan(full[:5]).all() and np.isfinite(full[5:]).all()
    assert 0.7 < full[5:, 0].std() / full[5:, 10].std() < 1.4
    daily = r.reshape(1, -1, 3)
    d_full = rs.scale(daily, days=20)
    np.testing.assert_allclose(d_full[:, :300], rs.scale(daily[:, :300], days=20), rtol=1e-6)
    assert np.isnan(d_full[:, :20]).all() and np.isfinite(d_full[:, 20:]).all()


def test_a_name_that_stops_moving_cannot_poison_the_others(rs):
    rng = np.random.default_rng(5)
    r = rng.normal(size=(1, 400, 3)).astype(np.float32)          # daily: one session of days
    clean = rs.scale(r, days=20)
    r[0, 100:200, 2] = 0.0                                       # a halted / pinned name ...
    r[0, 200, 2] = 3.0                                           # ... that then moves once
    out = rs.scale(r, days=20)
    assert not np.isinf(out).any()
    assert np.isnan(out[0, 200, 2])                              # no scale, no value
    np.testing.assert_allclose(out[0, 20:, :2], clean[0, 20:, :2], rtol=0.05)


def test_forward_enters_at_the_next_open_and_never_holds_past_the_close(rs):
    idx = _index(1, 6)
    o = np.array([10, 11, 12, 13, 14, 15.0])
    cl = o + 0.5
    b_v = [100, 100, 0, 100, 100, 100]
    c = rs.cube(_panel(idx, np.c_[o, o], np.c_[cl, cl], np.c_[[100] * 6, b_v]))
    f = rs.forward(c["x"], c["y"], 2)
    assert f[0, 0, 0] == pytest.approx(math.log(cl[2] / o[1]))
    assert np.isnan(f[0, 4:, 0]).all()                          # would hold past the last bar
    assert np.isnan(f[0, 1, 1])                                 # entry bar 2 has no print
    f2 = rs.forward(c["x"], c["y"], 2, delay=2)
    assert f2[0, 0, 0] == pytest.approx(math.log(cl[3] / o[2]))
    assert rs.forward(c["x"], c["y"], 1)[0, 0, 0] == pytest.approx(math.log(cl[1] / o[1]))


def test_spread_portfolios_and_day_t_by_hand(rs):
    pred = np.tile(np.arange(20.0), (6, 1))
    out = rs.spread(pred, pred * 0.01, np.repeat([0, 1, 2], 2), q=0.1, min_names=10)
    assert out["spread"] == pytest.approx(0.185 - 0.005) and out["ic"] == pytest.approx(1.0)
    assert out["rows"] == 6 and (np.diff(out["deciles"]) > 0).all()
    P, groups = rs.portfolios(np.array([[1, 2, np.nan, 4], [np.nan, np.nan, 3, 5.0]]),
                              ["a", "a", "b", None])
    assert groups == ["a", "b"]
    np.testing.assert_allclose(P, [[1.5, np.nan], [np.nan, 3]])
    v = np.array([1.0, 3.0, 2.0, 6.0, -1.0, 1.0, 4.0])
    d = np.array([0, 0, 1, 1, 2, 2, 3])
    sums = np.array([(v[d == g] - v.mean()).sum() for g in range(4)])
    se = math.sqrt((sums ** 2).sum() * 4 / 3) / len(v)
    assert rs.day_t(v, d)["t"] == pytest.approx(v.mean() / se)
    assert rs.spearman([1, 2, 3, np.nan], [2, 4, 9, 1]) == pytest.approx(1.0)
