"""proposals_lab v2's own portfolio runner (strategies/proposals_lab/lab.py): the engine's
``_run_portfolio`` with short covers executed before new shorts and small short re-trims
deferring their borrow; day 0 counted from the initial cash. Since engine proposals #46 and
#48 (2026-10-24) the shared engine does both itself, so the lab and the engine now agree.
Synthetic panels, no DB."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple.data import Panel

_SPEC = importlib.util.spec_from_file_location(
    "proposals_lab", Path(__file__).resolve().parents[1] / "strategies/proposals_lab/lab.py")
lab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lab)


def _panel():
    idx = pd.date_range("2024-01-01", periods=4, freq="B", tz="UTC")
    close = pd.DataFrame({"A": [10, 10, 10, 10.0], "B": [10, 9, 9, 9.0], "C": [10, 10, 10, 10.0]},
                         index=idx)
    return Panel(close=close, open_=close.copy(), high=close.copy(), low=close.copy(),
                 volume=pd.DataFrame(1e6, index=idx, columns=close.columns), bar="1d")


def _weights(p):
    """A full 0.98-gross book: long A, short B; on bar 2 B is covered and C shorted."""
    return pd.DataFrame({"A": [0.49] * 4, "B": [-0.49, 0.0, 0.0, 0.0],
                         "C": [0.0, -0.49, -0.49, -0.49]}, index=p.close.index)


def test_the_engine_now_fills_the_new_short_like_the_lab_runner():
    """Before #46 the engine's call_seq="auto" part-filled C (~20 of ~500 shares)."""
    p = _panel()
    w = _weights(p)
    kw = dict(price=p.close, fees=0.0, slippage=0.0, init_cash=10_000)
    engine = bt._run_portfolio(p, w, **kw)
    fixed = lab.run_portfolio(p, w, **kw)
    for pf in (engine, fixed):
        assert pf.assets()["C"].iloc[-1] == pytest.approx(-0.49 * pf.value().iloc[1] / 10,
                                                          rel=0.02)
        assert pf.assets()["B"].iloc[-1] == 0
    assert np.allclose(engine.value().to_numpy(), fixed.value().to_numpy())


def test_without_a_rotation_the_lab_runner_is_the_engine():
    p = _panel()
    w = pd.DataFrame({"A": [0.49] * 4, "B": [-0.49] * 4, "C": [0.0] * 4}, index=p.close.index)
    kw = dict(price=p.close, fees=0.001, slippage=0.0005, init_cash=10_000)
    a = bt._run_portfolio(p, w, **kw).value()
    b = lab.run_portfolio(p, w, **kw).value()
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_call_order_sells_covers_shorts_then_buys():
    idx = pd.date_range("2024-01-01", periods=2, freq="B", tz="UTC")
    t = pd.DataFrame({"buy": [0.0, 0.3], "short": [0.0, -0.3], "cover": [-0.3, 0.0],
                      "sell": [0.3, 0.0], "none": [0.1, np.nan]}, index=idx)
    seq = lab._call_order(t)[1]                                 # columns in call order
    assert list(t.columns[seq]) == ["sell", "cover", "short", "buy", "none"]


def test_a_hair_re_trim_defers_the_borrow_to_the_buy_back():
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    tg = pd.DataFrame({"X": [-0.10, np.nan, -0.0999, np.nan, 0.0]}, index=idx)
    engine = bt._borrow_fees(tg, 0.10, 0.0)["X"]
    fixed = lab.borrow_fees(tg, 0.10, 0.0)["X"]
    for fees in (engine, fixed):                               # before #48: a 55% fee on bar 2
        assert fees.iloc[2] == 0.0
        assert fees.iloc[4] == pytest.approx(4 / 365 * 0.10, rel=1e-6)


def test_day_zero_is_counted_from_the_initial_cash():
    p = _panel()
    w = _weights(p)
    pf = lab.run_portfolio(p, w, price=p.close, fees=0.01, slippage=0.0, init_cash=10_000)
    v = bt.equity_curve(pf)["value"]
    r = v / v.shift(1).fillna(10_000) - 1.0
    assert (1 + r).prod() == pytest.approx(v.iloc[-1] / 10_000)
    assert r.iloc[0] < 0                                        # the opening fees land on day 0
