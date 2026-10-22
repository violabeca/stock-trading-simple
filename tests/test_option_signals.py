"""option_signals research helpers: Black-Scholes implied volatility round-trips, the dividend
yield read from adjusted ÷ raw closes, the surface summary recovers known volatilities from a
synthetic chain (expiry A, ATM, the 0.95 put, the open-interest-weighted call-put spread), the
signals never read a later session, forward returns start at the next open and lose the market,
and the IC / Newey-West machinery. Synthetic data only (no DB)."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

_SPEC = importlib.util.spec_from_file_location(
    "test_option_signals_rs", strategies.STRATEGIES_DIR / "option_signals" / "research.py")
rs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rs)
NYC = "America/New_York"


def test_implied_vol_round_trips_calls_and_puts_with_dividends():
    S, K = np.array([100.0, 100, 50, 50]), np.array([95.0, 105, 50, 45])
    T, q = np.array([0.08, 0.08, 0.2, 0.2]), np.array([0.0, 0.03, 0.01, 0.0])
    call = np.array([True, False, True, False])
    sigma = np.array([0.25, 0.6, 0.9, 0.4])
    price = rs.bs_price(S, K, T, rs.R, q, sigma, call)
    assert rs.implied_vol(price, S, K, T, rs.R, q, call) == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_is_nan_outside_the_no_arbitrage_bounds():
    iv = rs.implied_vol(np.array([0.01, 200.0]), 100.0, 80.0, 0.1, rs.R, 0.0,
                        np.array([True, True]))
    assert np.isnan(iv).all()                    # below intrinsic; above the stock price


def test_dividend_events_and_projection():
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    adj = close.copy()
    ex = [idx[20], idx[83], idx[146], idx[209]]               # quarterly, ~91 calendar days
    for d in ex:
        adj.loc[:d - pd.Timedelta(days=1), "A"] *= 0.99       # 1% dividends
    ev = rs.dividend_events(close, adj)
    assert list(ev) == ["A"]
    assert list(ev["A"]["date"]) == ex
    assert ev["A"]["y"].to_numpy() == pytest.approx(0.01, rel=1e-9)
    gap = int(np.median(np.diff(np.array(ex, dtype="datetime64[D]")).astype(int)))
    nxt = ex[-1] + pd.Timedelta(days=gap)
    d = nxt - pd.Timedelta(days=10)
    f = rs.dividend_fraction(ev["A"], [d, d, ex[1]], [nxt, nxt - pd.Timedelta(days=1), ex[1] +
                                                      pd.Timedelta(days=30)])
    assert f[0] == pytest.approx(0.01 * np.exp(-rs.R * 10 / 365))   # the projected one
    assert f[1] == 0                                          # expiry the day before it
    assert f[2] == 0                                          # next one ~91 days out
    assert rs.dividend_fraction(None, [d], [nxt])[0] == 0


def _chain(date="2026-01-05", S=100.0, ivs=None, oi=None, div=0.0):
    """Calls and puts on two expiries (16 and 32 days), strikes 85..115, priced at ``ivs``."""
    ivs = ivs or {}
    rows = []
    for exp in ("2026-01-21", "2026-02-06"):
        T = (pd.Timestamp(exp) - pd.Timestamp(date)).days / 365
        for k in range(85, 120, 5):
            for side in ("call", "put"):
                iv = ivs.get((side, k), 0.4)
                p = float(rs.bs_price(S * (1 - div), k, T, rs.R, 0.0, iv, side == "call"))
                rows.append(dict(date=date, optionSymbol=f"ABC{pd.Timestamp(exp):%y%m%d}"
                                 f"{side[0].upper()}{k * 1000:08d}",
                                 expiration=int(pd.Timestamp(f"{exp} 16:00", tz=NYC).timestamp()),
                                 side=side, strike=float(k), bid=p - 0.01, ask=p + 0.01,
                                 underlyingPrice=S, volume=10,
                                 openInterest=(oi or {}).get((side, k), 100)))
    return pd.DataFrame(rows)


def test_surface_summary_recovers_the_volatilities_behind_each_signal():
    ivs = {("put", 95): 0.50, ("call", 100): 0.40, ("put", 100): 0.42,
           ("call", 90): 0.45, ("put", 90): 0.47, ("call", 110): 0.38, ("put", 110): 0.36}
    out = rs.surface_day_summary(_chain(ivs=ivs), "ABC").iloc[0]
    assert out["exp"] == pd.Timestamp("2026-02-06")            # 32 days: inside 21-60
    assert out["iv_call"] == pytest.approx(0.40, abs=1e-4)
    assert out["iv_otm_put"] == pytest.approx(0.50, abs=1e-4)  # K/S 0.95
    pairs = {90: 0.45 - 0.47, 95: 0.40 - 0.50, 100: 0.40 - 0.42, 105: 0.0, 110: 0.38 - 0.36}
    assert out["cpiv"] == pytest.approx(np.mean(list(pairs.values())), abs=1e-4)
    assert out["call_vol"] == out["put_vol"] == 2 * 7 * 10


def test_cpiv_weights_by_open_interest():
    ivs = {("call", 100): 0.40, ("put", 100): 0.50}
    oi = {(side, k): 0 for side in ("call", "put") for k in range(85, 120, 5)}
    oi[("call", 100)] = oi[("put", 100)] = 10
    out = rs.surface_day_summary(_chain(ivs=ivs, oi=oi), "ABC").iloc[0]
    assert out["cpiv"] == pytest.approx(-0.10, abs=1e-4)      # only the K=100 pair has weight


def _panel(n=200, m=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-06-02", periods=n)
    cols = [f"S{i}" for i in range(m)]
    mkt = rng.normal(0, 0.01, n)
    beta = np.linspace(0.5, 1.5, m)
    r = mkt[:, None] * beta + rng.normal(0, 0.01, (n, m))
    adj = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=cols)
    return adj, beta


def test_signals_never_read_a_later_session():
    adj, _ = _panel()
    rng = np.random.default_rng(1)
    f = lambda: pd.DataFrame(rng.uniform(0.2, 0.6, adj.shape), index=adj.index,  # noqa: E731
                             columns=adj.columns)
    surf = {k: f() for k in ("iv_call", "iv_put", "iv_otm_put", "cpiv", "call_vol", "put_vol")}
    vol = f() * 1e6
    full = rs.signals(surf, adj, vol)
    cut = rs.signals({k: v.iloc[:120] for k, v in surf.items()}, adj.iloc[:120], vol.iloc[:120])
    for k in rs.SIGNALS:
        pd.testing.assert_frame_equal(full[k].iloc[:120], cut[k])


def test_forward_return_starts_at_the_next_open_and_loses_the_market():
    adj, beta_true = _panel()
    open_adj = adj.copy()
    beta, m = rs.market_beta(adj)
    assert beta.iloc[-1].corr(pd.Series(beta_true, index=adj.columns)) > 0.9
    mask = pd.DataFrame(True, index=adj.index, columns=adj.columns)
    fwd = rs.residual_forward(open_adj, beta, mask, 5)
    raw = open_adj.iloc[11] / open_adj.iloc[6] - 1           # signal day 5: opens of 6 and 11
    exp = raw - beta.iloc[5] * raw.mean()
    pd.testing.assert_series_equal(fwd.iloc[5], exp, check_names=False)
    assert fwd.iloc[-6:].isna().all().all()                  # no forward window left


def test_ic_and_newey_west():
    adj, _ = _panel()
    mask = pd.DataFrame(True, index=adj.index, columns=adj.columns)
    fwd = rs.residual_forward(adj, rs.market_beta(adj)[0], mask, 1)
    ic = rs.daily_ic(fwd, fwd, mask)
    assert ic.dropna().gt(0.999).all()                       # a perfect signal
    noise = pd.DataFrame(np.random.default_rng(3).normal(size=adj.shape), index=adj.index,
                         columns=adj.columns)
    mean, t = rs.nw_t(rs.daily_ic(noise, fwd, mask), 1)
    assert abs(t) < 3
    x = pd.Series(np.r_[np.ones(50), -np.ones(50)] + 0.1)
    assert rs.nw_t(x, 0)[0] == pytest.approx(0.1)
    assert abs(rs.nw_t(x, 10)[1]) < abs(rs.nw_t(x, 0)[1])    # long runs inflate the naive t


def test_residualize_removes_the_control_and_spread_orients_by_sign():
    adj, _ = _panel()
    mask = pd.DataFrame(True, index=adj.index, columns=adj.columns)
    ctrl = pd.DataFrame(np.random.default_rng(4).normal(size=adj.shape), index=adj.index,
                        columns=adj.columns)
    res = rs.residualize(3 * ctrl + 1, [ctrl], mask)
    assert res.abs().max().max() < 1e-9
    fwd = rs.residual_forward(adj, rs.market_beta(adj)[0], mask, 1)
    up, _ = rs.quintile_spread(fwd, fwd, mask, 1)
    down, turn = rs.quintile_spread(fwd, fwd, mask, -1)
    assert (up.dropna() > 0).all() and (down.dropna() < 0).all()
    assert turn.dropna().between(0, 1).all()


def test_a_dividend_inside_the_expiry_is_not_read_as_cheap_puts():
    events = pd.DataFrame({"date": pd.to_datetime(["2025-04-20", "2025-07-20", "2025-10-20"]),
                           "y": [0.01] * 3})
    gap = int(np.median(np.diff(events["date"].to_numpy(dtype="datetime64[D]")).astype(int)))
    nxt = events["date"].iloc[-1] + pd.Timedelta(days=gap)          # 2026-01-19, before expiry
    f = 0.01 * np.exp(-rs.R * (nxt - pd.Timestamp("2026-01-05")).days / 365)
    raw = _chain(div=f)
    seen = rs.surface_day_summary(raw, "ABC", events).iloc[0]
    blind = rs.surface_day_summary(raw, "ABC").iloc[0]
    assert seen["cpiv"] == pytest.approx(0.0, abs=1e-4) and seen["div"] == pytest.approx(f)
    assert blind["cpiv"] < -0.01                                     # puts look dear without it


def test_score_signal_passes_a_planted_signal_and_not_noise():
    adj, _ = _panel(n=260, m=120)
    mask = pd.DataFrame(True, index=adj.index, columns=adj.columns)
    fwd = rs.residual_forward(adj, rs.market_beta(adj)[0], mask, 5)
    rng = np.random.default_rng(7)
    noise = pd.DataFrame(rng.normal(size=adj.shape), index=adj.index, columns=adj.columns)
    planted = -(fwd.rank(axis=1, pct=True).fillna(0.5) + 3 * noise)    # a NEGATIVE signal
    dvol = pd.DataFrame(1.0, index=adj.index, columns=adj.columns).cumsum(axis=1)
    ctrl = [pd.DataFrame(rng.normal(size=adj.shape), index=adj.index, columns=adj.columns)]
    hit = rs.score_signal(planted, fwd, mask, ctrl, dvol, -1, 5)
    assert hit["passed"] and hit["ic"] < 0 and hit["book_gross"] > 0
    assert not rs.score_signal(planted, fwd, mask, ctrl, dvol, +1, 5)["passed"]   # wrong sign
    miss = rs.score_signal(noise, fwd, mask, ctrl, dvol, -1, 5)
    assert not miss["passed"] and miss["mde"] > 0


def test_rebalanced_book_counts_turnover_once_per_rebalance():
    idx = pd.bdate_range("2026-01-05", periods=40)
    cols = [f"S{i}" for i in range(100)]
    sig = pd.DataFrame(np.tile(np.arange(100.0), (40, 1)), index=idx, columns=cols)
    fwd = sig / 1000                                              # top fifth earns most
    mask = pd.DataFrame(True, index=idx, columns=cols)
    b = rs.rebalanced_spread(sig, fwd, mask, 1, 5, cost_bp=5)
    assert b["periods"] == 8 and b["turnover"] == 0              # same legs every time
    gross = (np.arange(80, 100).mean() - np.arange(20).mean()) / 1000 * 252 / 5
    assert b["gross"] == pytest.approx(gross)
    assert b["net"] == pytest.approx(gross - 4 * 5e-4 / 8 * 252 / 5)   # only the first period pays
