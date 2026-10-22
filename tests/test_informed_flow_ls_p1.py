"""informed_flow_ls_p1 (proposal 1's research copy): with its extra knobs off it is v6; beta sizing
is set at entry and frozen; the index fills empty short slots — synthetic panels, no DB."""
import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

P1 = strategies.load("informed_flow_ls_p1")
SMALL = {"neutral_groups": 0, "q_in": 0.8, "q_out": 0.7, "slots": 2, "skew_smooth": 1,
         "score_smooth": 1}


def _panel(betas, n=160, extra=None, seed=3, flip_at=0):
    """Ten names A..J with days-to-cover A=1 .. J=10 (before bar ``flip_at`` the reverse) and
    the given market betas."""
    names = list("ABCDEFGHIJ")
    idx = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.01, n)
    rets = np.outer(market, betas) + rng.normal(0.0, 0.001, (n, len(names)))
    px = pd.DataFrame(50.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=names)
    if extra is not None:
        px = px.join(extra.set_axis(idx))
    cols = px.columns
    dtc = pd.DataFrame(np.tile(np.arange(1.0, 11.0), (n, 1)), index=idx, columns=names)
    dtc.iloc[:flip_at] = dtc.iloc[:flip_at, ::-1].to_numpy()
    feats = {"short_interest": dtc,
             "fund_shares": pd.DataFrame(1e8, index=idx, columns=names)}
    return Panel(close=px, open_=px.copy(), high=px.copy(), low=px.copy(),
                 volume=pd.DataFrame(1e6, index=idx, columns=cols), bar="1d", features=feats)


def test_every_extra_knob_off_is_v6s_dollar_neutral_equal_book():
    w = P1.signals(_panel([2.0, 2.0, 1, 1, 1, 1, 1, 1, 0.5, 0.5]), **{**P1.params, **SMALL})
    last = w["weights"].iloc[-1]
    assert last[["A", "B"]].tolist() == pytest.approx([0.245, 0.245])
    assert last[["I", "J"]].tolist() == pytest.approx([-0.245, -0.245])


def test_beta_sizing_shrinks_high_beta_longs_grows_low_beta_shorts_and_freezes_them():
    p = _panel([2.0, 2.0, 1, 1, 1, 1, 1, 1, 0.5, 0.5], flip_at=60)   # A, B enter long at 61
    w = P1.signals(p, **{**P1.params, **SMALL, "sizing": "beta", "beta_window": 40})["weights"]
    last = w.iloc[-1]
    assert 0 < last["A"] < 0.245 < -last["J"]                  # equal slot: 0.98 / 2 / 2
    assert w.abs().sum(axis=1).max() <= 0.98 + 1e-9            # the gross guard
    held = w["A"][w["A"] > 0]                                  # the long hold after the flip
    assert held.nunique() == 1                                 # frozen while held: no re-trading


def test_index_fills_every_empty_short_slot():
    p = _panel([1.0] * 10, extra=pd.DataFrame({"SPY": np.linspace(400, 480, 160)}))
    rising = p.close.copy()
    rising.loc[:, list("ABCDEFGHIJ")] = np.linspace(40, 60, 160)[:, None]   # nothing in a downtrend
    p = Panel(close=rising, open_=rising.copy(), high=rising.copy(), low=rising.copy(),
              volume=p.volume, bar="1d", features=p.features)
    out = P1.signals(p, **{**P1.params, **SMALL, "short_below_ma": 20, "hedge": "SPY"})
    last = out["weights"].iloc[-1]
    assert (last.drop("SPY") >= 0).all()                       # no single-name short in an uptrend
    assert last["SPY"] == pytest.approx(-0.49)                 # both short slots held as the index
    assert last[["A", "B"]].tolist() == pytest.approx([0.245, 0.245])
    assert out["borrow"]["SPY"].iloc[-1] == P1.params["borrow_gc"]
