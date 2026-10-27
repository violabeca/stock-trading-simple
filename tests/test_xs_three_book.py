"""xs_three_book (proposals lab, cross-strategy round): the SI, MOM and AH books summed into one
netted cash account — synthetic panels, no DB. The components run from their own strategy.py
files (loaded by path); the tests compare with the same files loaded by the repo's loader."""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

XS = strategies.load("xs_three_book")
MOD = XS.signals.__globals__                     # the loaded strategy.py
SLEEVES = ("si", "mom", "ah")
GROSS = {"si": 0.98, "mom": 0.90, "ah": 0.98}    # the components' shipped nominal gross
I1 = {"hedge_max": 0.08, "book_vol": 0.10, "beta_cap": 0.15}
I2 = {"budgets": "0.36,0.36,0.18", "ah_part": "shorts"}
_SPEC = importlib.util.spec_from_file_location(
    "proposals_lab_xs3", Path(__file__).resolve().parents[1] / "strategies/proposals_lab/lab.py")
LAB = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(LAB)
#: (name index, first bar, bars, daily return): steady declines that make breakdown shorts.
CRASHES = ((5, 290, 40, -0.025), (17, 300, 40, -0.025), (29, 330, 45, -0.025),
           (33, 280, 30, -0.03))


def _panel(n=420, m=40, seed=3, spy=True, dtc_gap=None, crashes=CRASHES):
    """``m`` names on one market factor (betas 0.6-1.4, drifts mixed), a days-to-cover per name
    that moves slowly, a few steady crashes (AH's breakdowns) and, with ``spy``, a SPY column the
    way ``lab.with_spy`` adds it (prices only, no volume, no features). ``dtc_gap = (a, b)``
    blanks every name's days-to-cover on bars a..b-1."""
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0004, 0.01, n)
    betas = np.linspace(0.6, 1.4, m)
    drift = np.linspace(-0.0012, 0.0012, m)[rng.permutation(m)]
    rets = np.outer(market, betas) + drift + rng.normal(0.0, 0.012, (n, m))
    for j, start, bars, daily in crashes:
        rets[start:start + bars, j] = daily
    names = [f"N{i:02d}" for i in range(m)]
    close = pd.DataFrame(50.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    level = 1.0 + 9.0 * rng.permutation(m) / m
    dtc = pd.DataFrame(np.outer(1.0 + 0.2 * np.sin(np.arange(n) / 30.0), level), index=idx,
                       columns=names)
    if dtc_gap is not None:
        dtc.iloc[dtc_gap[0]:dtc_gap[1]] = np.nan
    feats = {"short_interest": dtc, "fund_shares": pd.DataFrame(1e8, index=idx, columns=names)}
    frames = {"close": close, "open_": close * 0.999, "high": close * 1.01, "low": close * 0.99}
    volume = pd.DataFrame(1e6, index=idx, columns=names)
    if spy:
        s = pd.Series(300.0 * np.cumprod(1 + market), index=idx)
        frames = {k: f.assign(SPY=s * (f.iloc[:, 0] / close.iloc[:, 0])) for k, f in
                  frames.items()}
        volume = volume.assign(SPY=np.nan)
    return Panel(volume=volume, bar="1d", features=feats, **frames)


def _cut(p, k):
    """The panel's first ``k`` bars."""
    return Panel(close=p.close.iloc[:k], open_=p.open_.iloc[:k], high=p.high.iloc[:k],
                 low=p.low.iloc[:k], volume=p.volume.iloc[:k], bar="1d",
                 features={f: v.iloc[:k] for f, v in p.features.items()})


PANEL = _panel()


def _run(p=PANEL, **over):
    return XS.signals(p, **{**XS.params, **over})


def _parts(p=PANEL, **over):
    keys = inspect.signature(MOD["compose"]).parameters
    return MOD["compose"](p, **{k: v for k, v in {**XS.params, **over}.items() if k in keys})


def _component(key, p=PANEL, **over):
    """A component's own signals() on the panel without SPY, through the repo's loader."""
    s = strategies.load(MOD["COMPONENTS"][key][0])
    core = MOD["component"]("si")._without(p, "SPY")
    return s.signals(core, **{**s.params, **(MOD["AH_TUNED"] if key == "ah" else {}), **over})


class _Fake:
    """A component module whose ``signals()`` records its input and returns ``weights`` if
    given (else the real book); every other attribute is the real module's."""

    def __init__(self, real, key, seen, weights=None):
        self.real, self.key, self.seen, self.weights = real, key, seen, weights

    def __getattr__(self, name):
        return getattr(self.real, name)

    def signals(self, panel, **params):
        self.seen[self.key] = {"close": set(panel.close.columns),
                               "features": {f: set(v.columns)
                                            for f, v in (panel.features or {}).items()}}
        if self.weights is None:
            return self.real.signals(panel, **params)
        w = self.weights.reindex(index=panel.close.index, columns=panel.close.columns,
                                 fill_value=0.0)
        return {"weights": w, "borrow": pd.DataFrame(0.005, index=w.index, columns=w.columns)}


def _patch(monkeypatch, weights=None):
    seen = {}
    real = {k: MOD["component"](k) for k in SLEEVES}
    fakes = {k: _Fake(real[k], k, seen, (weights or {}).get(k)) for k in SLEEVES}
    monkeypatch.setitem(MOD, "component", fakes.__getitem__)
    return seen


def test_u0_is_each_components_own_book_at_a_third_of_the_account_summed_per_name():
    w = _run()["weights"]
    books = {k: _component(k)["weights"] for k in SLEEVES}
    for k in SLEEVES:                       # the synthetic panel exercises every sleeve
        assert (books[k] > 0).any().any() and (books[k] < 0).any().any(), k
    want = sum(books[k] * (0.30 / GROSS[k]) for k in SLEEVES)
    want = want.reindex(columns=PANEL.close.columns, fill_value=0.0)
    assert list(w.columns) == list(PANEL.close.columns) and w.index.equals(PANEL.close.index)
    np.testing.assert_allclose(w.to_numpy(), want.to_numpy(), rtol=0.0, atol=1e-15)
    assert (w["SPY"] == 0.0).all()


def test_a_long_in_one_sleeve_and_an_equal_short_in_another_net_to_no_position(monkeypatch):
    idx, cols = PANEL.close.index, [c for c in PANEL.close.columns if c != "SPY"]
    si = pd.DataFrame(0.0, index=idx, columns=cols)
    ah = si.copy()
    si.loc[idx[50:], "N00"] = 0.98 * 0.5 / 15            # one SI long slot ...
    ah.loc[idx[50:], "N00"] = -0.98 / 30                 # ... against one AH short slot
    si.loc[idx[50:], "N01"] = ah.loc[idx[80:], "N01"] = 0.98 * 0.5 / 15   # same side: they add
    _patch(monkeypatch, {"si": si, "ah": ah, "mom": si * 0.0})
    w = _run()["weights"]
    assert (w["N00"] == 0.0).all()                       # nothing held, nothing borrowed
    assert bt.rotation_targets(w)["N00"].dropna().tolist() == [0.0]    # never an order
    assert w["N01"].iloc[60] == pytest.approx(0.01) and w["N01"].iloc[90] == pytest.approx(0.02)


def test_spy_never_reaches_a_sleeve_except_ah_in_its_index_mode(monkeypatch):
    seen = _patch(monkeypatch)
    _run()
    for k in SLEEVES:
        assert "SPY" not in seen[k]["close"], k
        assert all("SPY" not in cols for cols in seen[k]["features"].values()), k
    _run(ah_hedge="SPY")
    assert "SPY" in seen["ah"]["close"]
    assert "SPY" not in seen["si"]["close"] and "SPY" not in seen["mom"]["close"]


def test_ah_index_mode_keeps_ahs_longs_and_holds_its_shorts_as_spy():
    u0, idx = _parts()["sleeves"]["ah"], _parts(ah_hedge="SPY")["sleeves"]["ah"]
    assert (u0 < 0).any().any()
    names = [c for c in u0.columns if c != "SPY"]
    np.testing.assert_allclose(idx[names].to_numpy(), u0[names].clip(lower=0).to_numpy(),
                               atol=1e-15)
    np.testing.assert_allclose(idx["SPY"].to_numpy(),
                               u0[names].clip(upper=0).sum(axis=1).to_numpy(), atol=1e-12)


@pytest.mark.parametrize("over", [{}, I1, I2, {"sleeves": "mom,ah"}, {"ah_hedge": "SPY"}])
def test_a_truncated_panel_reproduces_the_weights_through_its_last_bar(over):
    full = _run(**over)["weights"]
    for k in (300, 352, 391):
        cut = _run(_cut(PANEL, k), **over)["weights"]
        np.testing.assert_allclose(cut.to_numpy(), full.iloc[:k].to_numpy(), rtol=0.0,
                                   atol=1e-12, err_msg=f"{over} k={k}")


def test_the_coverage_switch_puts_si_and_ah_in_cash_on_bars_without_short_interest():
    p = _panel(dtc_gap=(300, 330))
    parts = _parts(p)
    assert not parts["on"].iloc[301:331].any() and parts["on"].iloc[1:301].all()
    assert parts["on"].iloc[331:].all()
    raw_ah = _component("ah", p)["weights"]
    assert (raw_ah.iloc[301:331] < 0).any().any()        # AH alone would stay short there
    for k in ("si", "ah"):
        assert (parts["sleeves"][k].iloc[301:331] == 0.0).all().all(), k
    mom = _component("mom", p)["weights"].reindex(columns=p.close.columns, fill_value=0.0)
    np.testing.assert_allclose(parts["weights"].iloc[301:331].to_numpy(),
                               (mom * (0.30 / 0.90)).iloc[301:331].to_numpy(), atol=1e-15)
    assert _parts()["on"].iloc[1:].all()                 # a full table never trips it


@pytest.mark.parametrize("over", [{}, I1, I2, {"sleeves": "mom,ah"}, {"sleeves": "si,mom"},
                                  {"ah_hedge": "SPY"}, {"ah_variant": "u0"},
                                  {"sleeves": "si"}, {"sleeves": "ah"}])
def test_nominal_gross_never_exceeds_the_accounts_090(over):
    w = _run(**over)["weights"]
    assert w.abs().sum(axis=1).max() <= 0.90 + 1e-12
    if over.get("hedge_max"):
        assert w.drop(columns="SPY").abs().sum(axis=1).max() <= 0.82 + 1e-12


def test_beta_hedge_is_a_short_only_capped_and_stepped_month_end_decision():
    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    beta = pd.Series([0.30, 0.10, 0.19, 0.19, 0.50, 0.12, 0.40, 0.25], index=idx)
    spy = pd.Series([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, -0.1, 0.9], index=idx)
    fills = pd.Series([False, True, True, False, True, True, True, False], index=idx)
    h = MOD["beta_hedge"](beta, spy, fills, beta_cap=0.15, hedge_max=0.08, hedge_step=0.02)
    assert h.tolist() == pytest.approx([0.0, 0.0, -0.06, -0.06, -0.08, 0.0, 0.0, 0.0])
    assert (h <= 0).all() and (h >= -0.08).all()
    assert 0.19 - 0.06 * 0.9 <= 0.15                     # rounded up: below the cap


@pytest.mark.parametrize("cap", [0.15, 0.0])
def test_i1_scale_and_hedge_take_their_steps_and_change_only_after_a_month_end(cap):
    parts = _parts(**{**I1, "beta_cap": cap, "book_vol": 0.005})
    rm = MOD["component"]("mom")
    core = MOD["component"]("si")._without(PANEL, "SPY")
    rows = rm._full_rows(core.close)
    first = rm._month_ends(PANEL.close.index, rows)[rows].shift(1, fill_value=False)
    scale, h, beta = parts["scale"], parts["hedge"], parts["beta"]
    assert set(np.round(scale.unique(), 12)) <= {0.25, 0.5, 0.75, 1.0}
    assert (scale < 1).any()
    for series in (scale, h):
        moved = series.ne(series.shift()) & (np.arange(len(series)) > 0)
        assert not (moved & ~first).any()                # only on a month end's next bar
    assert (h <= 0).all() and (h >= -0.08 - 1e-12).all()
    assert (h[first & (beta <= cap)] == 0.0).all()
    if cap == 0.0:
        assert (h < 0).any()
    betas = MOD["trailing_betas"](PANEL.close, core.close.columns, 126)
    post = beta + h * betas["SPY"]                       # uncapped: brought to the cap
    free = first & (h > -0.08) & (beta > cap)
    assert (post[free] <= cap + 1e-12).all()
    w = _run(**{**I1, "beta_cap": cap})["weights"]
    np.testing.assert_allclose(w["SPY"].to_numpy(),
                               _parts(**{**I1, "beta_cap": cap})["hedge"].to_numpy(), atol=1e-15)


def test_i2s_crisis_sleeve_is_ahs_short_weights_at_09_percent_a_slot():
    parts = _parts(**I2)
    ah = _component("ah")["weights"].reindex(columns=PANEL.close.columns, fill_value=0.0)
    want = ah.clip(upper=0.0) * ((0.18 / 20) / (0.98 / 30))
    np.testing.assert_allclose(parts["sleeves"]["ah"].to_numpy(), want.to_numpy(), atol=1e-15)
    assert (parts["sleeves"]["ah"] <= 0.0).all().all()          # AH's longs are dropped
    assert parts["sleeves"]["ah"].min().min() == pytest.approx(-0.009)   # 0.9% a short slot
    for k in ("si", "mom"):
        book = _component(k)["weights"].reindex(columns=PANEL.close.columns, fill_value=0.0)
        np.testing.assert_allclose(parts["sleeves"][k].to_numpy(),
                                   (book * (0.36 / GROSS[k])).to_numpy(), atol=1e-15)


def test_borrow_is_the_dearest_sleeves_rate_and_gc_elsewhere():
    b = _run()["borrow"]
    rates = [_component(k)["borrow"].reindex(columns=PANEL.close.columns) for k in SLEEVES]
    want = pd.concat(rates).groupby(level=0).max().reindex(PANEL.close.index).fillna(0.005)
    want = want.where(want >= 0.005, 0.005)
    np.testing.assert_allclose(b.to_numpy(), want.to_numpy())
    assert (b["SPY"] == 0.005).all()


def test_the_lab_runner_holds_every_target_of_the_account():
    sig = _run()
    w = sig["weights"]
    bt.configure()
    pf = LAB.run_portfolio(PANEL, w, price=PANEL.open_, fees=0.0011, slippage=0.0001,
                           init_cash=10_000.0, retrim_k=sig["retrim_k"], borrow=sig["borrow"])
    assert np.isfinite(pf.value()).all() and pf.value().iloc[-1] > 0
    assert (np.sign(pf.assets()) == np.sign(w)).all().all()   # every target, on its side


def test_every_config_param_is_a_signals_keyword_and_valid_accepts_the_config():
    assert set(XS.params) <= set(inspect.signature(XS.signals).parameters)
    assert XS.valid(XS.params) and XS.fill == "open" and XS.order_notional == 1000.0
    assert XS.optuna["space"] == {} and XS.optuna["n_trials"] == 1
    for good in (I1, I2, {"sleeves": "mom,ah"}, {"ah_hedge": "SPY"}, {"ah_variant": "u0"},
                 {"sleeves": "si"}, {"gross": 0.85}):
        assert XS.valid({**XS.params, **good}), good


def test_valid_rejects_accounts_it_cannot_express():
    for bad in ({"sleeves": ""}, {"sleeves": "si,si"}, {"sleeves": "si,qj"},
                {"budgets": "0.3,0.3"}, {"budgets": "0.5,0.3,0.2"}, {"budgets": "0.3,-0.1,0.3"},
                {"budgets": "a,b,c"}, {"ah_variant": "v1"}, {"ah_part": "longs"},
                {"ah_hedge": "QQQ"}, {"gross": 1.2}, {"gross": 0.0}, {"hedge_max": 0.9},
                {"beta_cap": -0.1}, {"book_vol": -0.1}, {"retrim_k": 1.0},
                {**I2, "gross": 0.85}):
        assert not XS.valid({**XS.params, **bad}), bad


def test_signals_refuses_spy_rules_without_spy_in_the_panel():
    p = _panel(spy=False)
    _run(p)                                              # U0 never needs SPY
    for over in ({"ah_hedge": "SPY"}, I1):
        with pytest.raises(ValueError, match="SPY"):
            _run(p, **over)


def test_the_pinned_component_files_are_the_files_on_disk():
    assert MOD["file_hashes"]() == MOD["SHA256"]
    assert {k for k, _ in MOD["COMPONENTS"].values()} == {
        k.split("/")[0] for k in MOD["SHA256"]}
    for key, (_, alias) in MOD["COMPONENTS"].items():
        assert MOD["component"](key).__name__ == alias       # loaded by path, prefixed
