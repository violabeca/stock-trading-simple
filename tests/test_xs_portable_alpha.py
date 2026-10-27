"""xs_portable_alpha (proposals lab, cross-strategy round): a constant SPY core plus xs_three_book's
SI + MOM overlay at 0.45, one cash account — synthetic panels, no DB. The overlay is compared with
xs_three_book loaded by the repo's loader, the core with a loop re-statement of I1's rule."""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

PA = strategies.load("xs_portable_alpha")
MOD = PA.signals.__globals__                     # the loaded strategy.py
TB = strategies.load("xs_three_book")            # the book module, through the repo's loader
I1 = {"core_vol": True}
FRONTIER = ({"core": 0.15}, {"core": 0.30}, {"core": 0.60})
_SPEC = importlib.util.spec_from_file_location(
    "proposals_lab_xspa", Path(__file__).resolve().parents[1] / "strategies/proposals_lab/lab.py")
LAB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LAB)
#: (first bar, bars, volatility multiple) of the market's (and SPY's) volatility bursts.
BURSTS = ((200, 25, 3.0), (300, 30, 4.0))


def _panel(n=420, m=40, seed=5, spy=True, bursts=BURSTS):
    """``m`` names on one market factor (betas 0.6-1.4, drifts mixed), a slowly moving
    days-to-cover per name and, with ``spy``, a SPY column the way ``lab.with_spy`` adds it
    (open = high = low = close, no volume, no features). The market's volatility jumps in
    ``bursts``, so I1's rule takes all three values."""
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.008, n)
    for start, bars, k in bursts:
        market[start:start + bars] *= k
    betas = np.linspace(0.6, 1.4, m)
    drift = np.linspace(-0.0012, 0.0012, m)[rng.permutation(m)]
    rets = np.outer(market, betas) + drift + rng.normal(0.0, 0.012, (n, m))
    names = [f"N{i:02d}" for i in range(m)]
    close = pd.DataFrame(50.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    level = 1.0 + 9.0 * rng.permutation(m) / m
    dtc = pd.DataFrame(np.outer(1.0 + 0.2 * np.sin(np.arange(n) / 30.0), level), index=idx,
                       columns=names)
    feats = {"short_interest": dtc, "fund_shares": pd.DataFrame(1e8, index=idx, columns=names)}
    frames = {"close": close, "open_": close * 0.999, "high": close * 1.01, "low": close * 0.99}
    volume = pd.DataFrame(1e6, index=idx, columns=names)
    if spy:
        s = pd.Series(300.0 * np.cumprod(1 + market), index=idx)
        frames = {k: f.assign(SPY=s) for k, f in frames.items()}
        volume = volume.assign(SPY=np.nan)
    return Panel(volume=volume, bar="1d", features=feats, **frames)


def _cut(p, k):
    """The panel's first ``k`` bars."""
    return Panel(close=p.close.iloc[:k], open_=p.open_.iloc[:k], high=p.high.iloc[:k],
                 low=p.low.iloc[:k], volume=p.volume.iloc[:k], bar="1d",
                 features={f: v.iloc[:k] for f, v in p.features.items()})


PANEL = _panel()


def _run(p=PANEL, **over):
    return PA.signals(p, **{**PA.params, **over})


def _overlay(p=PANEL, gross=0.45, sleeves="si,mom"):
    """xs_three_book's own book at ``gross`` (loaded by the repo's loader, not this module's)."""
    return TB.signals(p, **{**TB.params, "sleeves": sleeves, "gross": gross})


def _first_after_month_end(p=PANEL):
    """True on the first bar after each month end (resid_momentum's ``_month_ends`` on SPY)."""
    rm = MOD["book"]().component("mom")
    rows = p.close["SPY"].notna()
    return rm._month_ends(p.close.index, rows).shift(1, fill_value=False)


def _m_by_hand(spy: pd.Series) -> pd.Series:
    """I1's m_t restated with explicit loops: the month-end decision, applied the next bar."""
    r = spy.pct_change().to_numpy()
    n = len(r)
    var = np.full(n, np.nan)
    for t in range(21, n):                       # returns 1..t: the 21 ending at t
        var[t] = np.var(r[t - 20:t + 1], ddof=1)
    out, m = np.ones(n), 1.0
    month = spy.index.year * 12 + spy.index.month
    for t in range(n):
        out[t] = m                               # in force at bar t: decided at a close < t
        if t + 1 < n and month[t + 1] != month[t]:      # bar t is a month end
            past = var[max(0, t - 251):t + 1]
            past = past[np.isfinite(past)]
            if len(past) >= 126 and np.isfinite(var[t]):
                med = np.median(past)
                m = 1.0 if var[t] <= med else (0.5 if var[t] <= 2 * med else 0.0)
    return pd.Series(out, index=spy.index)


def test_u0_holds_spy_at_045_on_every_bar_and_the_engine_buys_it_once():
    w = _run()["weights"]
    assert (w["SPY"] == 0.45).all()
    assert bt.rotation_targets(w)["SPY"].notna().sum() == 1          # bar 0 only: never re-sent
    for over in FRONTIER:
        assert (_run(**over)["weights"]["SPY"] == over["core"]).all()


def test_i1_core_takes_three_values_and_moves_only_on_the_first_bar_after_a_month_end():
    spy = _run(**I1)["weights"]["SPY"]
    assert set(np.round(spy.unique(), 12)) == {0.0, 0.225, 0.45}
    moved = spy.ne(spy.shift()) & (np.arange(len(spy)) > 0)
    assert moved.any() and not (moved & ~_first_after_month_end()).any()
    np.testing.assert_allclose(spy.to_numpy(), 0.45 * _m_by_hand(PANEL.close["SPY"]).to_numpy(),
                               rtol=0.0, atol=1e-15)


def test_the_overlay_is_xs_three_books_si_mom_book_at_045_exactly():
    for over in ({}, I1):
        w = _run(**over)["weights"]
        ref = _overlay()["weights"]
        assert (ref["SPY"] == 0.0).all()                              # no sleeve trades SPY
        assert list(w.columns) == list(PANEL.close.columns) and w.index.equals(PANEL.close.index)
        names = [c for c in w.columns if c != "SPY"]
        assert (ref[names] > 0).any().any() and (ref[names] < 0).any().any()
        np.testing.assert_array_equal(w[names].to_numpy(), ref[names].to_numpy())
    half = 0.5 * _overlay(gross=0.90)["weights"]                     # xs_three_book's C_core
    np.testing.assert_allclose(_run()["weights"].drop(columns="SPY").to_numpy(),
                               half.drop(columns="SPY").to_numpy(), rtol=0.0, atol=1e-15)


def test_core_zero_is_xs_three_books_c_core_and_the_frontier_overlay_shrinks_with_the_core():
    np.testing.assert_array_equal(_run(core=0.0)["weights"].to_numpy(),
                                  _overlay(gross=0.90)["weights"].to_numpy())
    for over in FRONTIER:
        w = _run(**over)["weights"].drop(columns="SPY")
        ref = _overlay(gross=0.90 - over["core"])["weights"].drop(columns="SPY")
        np.testing.assert_array_equal(w.to_numpy(), ref.to_numpy())


class _Fake:
    """A component module whose ``signals()`` records the panel it is given; every other
    attribute (and the book it returns) is the real module's."""

    def __init__(self, real, key, seen):
        self.real, self.key, self.seen = real, key, seen

    def __getattr__(self, name):
        return getattr(self.real, name)

    def signals(self, panel, **params):
        self.seen[self.key] = {"close": set(panel.close.columns),
                               "features": {f: set(v.columns)
                                            for f, v in (panel.features or {}).items()}}
        return self.real.signals(panel, **params)


def test_spy_never_reaches_a_sleeves_input(monkeypatch):
    tb = MOD["book"]()
    seen = {}
    fakes = {k: _Fake(tb.component(k), k, seen) for k in ("si", "mom", "ah")}
    monkeypatch.setattr(tb, "component", fakes.__getitem__)
    for over in ({}, I1):
        seen.clear()
        _run(**over)
        assert set(seen) == {"si", "mom"}                             # AH is not in the overlay
        for k, s in seen.items():
            assert "SPY" not in s["close"], k
            assert all("SPY" not in cols for cols in s["features"].values()), k


@pytest.mark.parametrize("over", [{}, I1, *FRONTIER, {"core": 0.0}])
def test_nominal_gross_never_exceeds_090(over):
    w = _run(**over)["weights"]
    assert w.abs().sum(axis=1).max() <= 0.90 + 1e-9
    assert w.drop(columns="SPY").abs().sum(axis=1).max() <= 0.90 - over.get("core", 0.45) + 1e-9


@pytest.mark.parametrize("over", [{}, I1, {"core": 0.60}])
def test_a_truncated_panel_reproduces_the_weights_through_its_last_bar(over):
    full = _run(**over)
    first = np.flatnonzero(_first_after_month_end().to_numpy())
    cuts = [300, 352, 391, int(first[first > 320][0]), int(first[first > 320][0]) + 1]
    for k in cuts:
        cut = _run(_cut(PANEL, k), **over)
        for key in ("weights", "borrow"):
            np.testing.assert_allclose(cut[key].to_numpy(), full[key].iloc[:k].to_numpy(),
                                       rtol=0.0, atol=1e-12, err_msg=f"{over} {key} k={k}")
        assert cut["retrim_k"] == full["retrim_k"] == 2.0


def test_borrow_is_the_overlays_with_spy_at_general_collateral():
    b = _run()["borrow"]
    ref = _overlay()["borrow"].reindex(index=PANEL.close.index, columns=PANEL.close.columns)
    np.testing.assert_array_equal(b.to_numpy(), ref.fillna(0.005).to_numpy())
    assert (b["SPY"] == 0.005).all() and (b > 0).all().all()


def test_signals_refuses_a_panel_without_spy():
    with pytest.raises(ValueError, match="SPY"):
        _run(_panel(spy=False))


def test_the_lab_runner_buys_the_core_once_and_holds_every_target():
    for over in ({}, I1):
        sig = _run(**over)
        w = sig["weights"]
        bt.configure()
        pf = LAB.run_portfolio(PANEL, w, price=PANEL.open_, fees=0.0011, slippage=0.0001,
                               init_cash=10_000.0, retrim_k=sig["retrim_k"], borrow=sig["borrow"])
        assert np.isfinite(pf.value()).all() and pf.value().iloc[-1] > 0
        assert (np.sign(pf.assets()) == np.sign(w)).all().all()   # every target, on its side
        spy_orders = (pf.orders.records_readable["Column"] == "SPY").sum()
        changes = int((w["SPY"].ne(w["SPY"].shift()) & (np.arange(len(w)) > 0)).sum())
        assert spy_orders == 1 + changes                           # never a re-trim of the core


def test_every_config_param_is_a_signals_keyword_and_valid_accepts_the_labels():
    assert set(PA.params) <= set(inspect.signature(PA.signals).parameters)
    assert PA.params == {"core": 0.45, "core_vol": False, "sleeves": "si,mom", "gross": 0.9}
    assert PA.valid(PA.params) and PA.fill == "open" and PA.order_notional == 1000.0
    assert PA.optuna["space"] == {} and PA.optuna["n_trials"] == 1
    assert PA.data["features"] == ["short_interest", "fund_shares"]
    for good in (I1, *FRONTIER, {"core": 0.0}, {"sleeves": "mom,ah"}):
        assert PA.valid({**PA.params, **good}), good


def test_valid_rejects_accounts_it_cannot_express():
    for bad in ({"core": -0.1}, {"core": 0.90}, {"core": 0.95}, {"gross": 1.2},
                {"core_vol": 1}, {"core_vol": "yes"}, {"sleeves": "si,si"}, {"sleeves": ""},
                {"sleeves": "si,qj"}, {"budgets": "0.3,0.3"}):
        assert not PA.valid({**PA.params, **bad}), bad


def test_research_core_book_holds_spy_at_its_target_and_resizes_at_the_close():
    spec = importlib.util.spec_from_file_location(
        "xspa_research", Path(__file__).resolve().parents[1]
        / "strategies/xs_portable_alpha/research.py")
    research = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(research)
    idx = pd.date_range("2024-01-02", periods=6, freq="B", tz="UTC")
    px = pd.Series([100.0, 110.0, 99.0, 99.0, 120.0, 60.0], index=idx)
    target = pd.Series([0.45, 0.45, 0.45, 0.225, 0.225, 0.0], index=idx)
    value = (1 + research.core_book(px, target)).cumprod().to_numpy()
    shares, cash = 0.0045, 0.55                       # 0.45 of 1.0 bought at 100 (bar 0's close)
    want = [1.0, cash + shares * 110, cash + shares * 99]
    shares3 = 0.225 * want[2] / 99                    # re-sized at bar 3's close (same price)
    cash3 = want[2] - shares3 * 99
    want += [want[2], cash3 + shares3 * 120, cash3 + shares3 * 60]
    np.testing.assert_allclose(value, want, rtol=1e-12)


def test_the_pinned_files_are_the_files_on_disk_and_a_change_refuses_to_load(monkeypatch):
    assert MOD["file_hashes"]() == MOD["SHA256"]
    tb = MOD["book"]()
    assert tb.__name__ == "_xspa_tb"                                # loaded by path, prefixed
    assert {k: v for k, v in MOD["SHA256"].items() if k != "xs_three_book/strategy.py"} \
        == tb.SHA256                                                # xs_three_book's own pins
    monkeypatch.setitem(MOD["SHA256"], "xs_three_book/strategy.py", "0" * 64)
    MOD["book"].cache_clear()
    try:
        with pytest.raises(RuntimeError, match="changed"):
            MOD["book"]()
    finally:
        monkeypatch.undo()
        MOD["book"].cache_clear()
    assert MOD["book"]().__name__ == "_xspa_tb"
