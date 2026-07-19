"""value_factor signal logic on synthetic fundamentals (no DB, no network)."""

import numpy as np
import pandas as pd

from trading_simple import strategies
from trading_simple.data import Panel


def _vf(panel, **overrides):
    strat = strategies.load("value_factor")
    return strat.signals(panel, **{**strat.params, **overrides})


def _module():
    """The value_factor strategy module (the loader binds only signals/valid)."""
    return strategies.load("value_factor").signals.__globals__


def _scores(panel, **overrides):
    return _module()["scores"](panel, **overrides)


def _bought(sig, sym) -> np.ndarray:
    """Bars where the book starts wanting a name (target weight 0 -> positive)."""
    weight = sig["weights"][sym]
    return np.where((weight > 0) & (weight.shift(1).fillna(0.0) == 0))[0]


def _panel(symbols=("AAA", "BBB", "CCC", "DDD"), n=60):
    """Flat-price daily panel with fundamentals ranking AAA > BBB > CCC > DDD.

    Constant levels, so every name is eligible from bar 0. Market cap is identical
    across names (same price, same shares), so book-to-price, earnings yield and
    gross profitability all rank on the descending 8/6/4/2 tiers below.
    """
    index = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame(100.0, index=index, columns=list(symbols))
    volume = pd.DataFrame(1_000_000.0, index=index, columns=list(symbols))
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.01,
                  low=close * 0.99, volume=volume, bar="1d")

    def frame(vals):
        return pd.DataFrame({s: vals[i] for i, s in enumerate(symbols)}, index=index)

    tier = ([8, 6, 4, 2] + [1] * len(symbols))[:len(symbols)]   # AAA best .. worst
    panel.features = {
        "fund_shares": frame([1e6] * len(symbols)),
        "fund_equity": frame([t * 1e7 for t in tier]),
        "fund_ni_ttm": frame([t * 1e6 for t in tier]),
        "fund_assets": frame([1e8] * len(symbols)),
        "fund_gp_ttm": frame([t * 1e6 for t in tier]),
    }
    return panel


def test_top_ranked_name_enters_next_bar():
    p = _panel()
    sig = _vf(p, max_positions=1, rebalance_days=21)
    # Selection on bar 0 -> execution at bar 1 (next-bar open), never same bar.
    assert sig["weights"].iloc[0].sum() == 0
    assert _bought(sig, "AAA")[0] == 1
    assert (sig["weights"]["AAA"].iloc[1:] > 0).all()        # cheapest+best: held throughout
    assert not (sig["weights"]["BBB"] > 0).any()             # one slot -> only AAA


def test_the_book_is_always_a_valid_portfolio():
    """Never oversubscribed, never levered, always whole slots (1/max_positions)."""
    p = _panel()
    for max_positions in (1, 2, 4):
        weights = _vf(p, max_positions=max_positions)["weights"]
        slot = 1.0 / max_positions
        held = (weights > 0).sum(axis=1)
        assert held.max() <= max_positions
        assert weights.sum(axis=1).max() <= 1.0 + 1e-12
        nonzero = weights.to_numpy()[weights.to_numpy() > 0]
        assert np.allclose(nonzero, slot)


def test_name_missing_a_used_factor_is_dropped_from_cross_section():
    """A name lacking any present factor is removed from the ranking, not neutral-filled."""
    p = _panel()
    p.features["fund_equity"]["BBB"] = np.nan          # BBB loses book-to-price
    parts = _scores(p)                                 # equal weights -> bp is present
    bar = p.close.index[10]
    assert not parts["eligible"].loc[bar, "BBB"]
    assert np.isnan(parts["composite"].loc[bar, "BBB"])
    assert parts["eligible"].loc[bar, ["AAA", "CCC", "DDD"]].all()
    # and it is never bought
    assert not (_vf(p)["weights"]["BBB"] > 0).any()


def test_low_coverage_factor_is_dropped_and_renormalized():
    """A factor covering too little of the universe must not become a tilt on a few names."""
    p = _panel()  # AAA, BBB, CCC, DDD
    gp = pd.DataFrame(np.nan, index=p.close.index, columns=p.symbols)
    gp["AAA"] = 8e6                                    # only 1 of 4 covered = 25% < 50% floor
    p.features["fund_gp_ttm"] = gp
    assert "gp" not in _scores(p, w_gp=0.9)["present"]
    # With enough coverage it is kept.
    gp["BBB"], gp["CCC"] = 6e6, 4e6                    # 3 of 4 = 75%
    assert "gp" in _scores(p, w_gp=0.9)["present"]


def test_selection_is_held_between_rebalances():
    """Turnover is paced by the cadence: the book does not react until the next rebalance."""
    p = _panel(n=45)
    idx = p.close.index
    # From bar 5, BBB becomes the best name and AAA the worst — but with a 30-bar
    # cadence (rebalances at bar 0 and 30) the book must not change until bar 30.
    for feat, worst, best in [("fund_equity", 1e6, 9e7), ("fund_ni_ttm", 1e5, 9e6),
                              ("fund_gp_ttm", 1e5, 9e6)]:
        p.features[feat].loc[idx[5:], "AAA"] = worst
        p.features[feat].loc[idx[5:], "BBB"] = best
    w = _vf(p, max_positions=1, rebalance_days=30)["weights"]
    assert w["AAA"].iloc[15] > 0 and w["BBB"].iloc[15] == 0   # mid-cycle: still AAA
    assert w["BBB"].iloc[31] > 0 and w["AAA"].iloc[31] == 0   # after the bar-30 rebalance


def test_missing_fundamentals_degrade_to_flat_book():
    """With no usable factor the strategy holds cash rather than erroring."""
    p = _panel()
    p.features = {}                                    # nothing attached
    weights = _vf(p)["weights"]
    assert weights.to_numpy().sum() == 0
    assert weights.shape == p.close.shape


def test_valid_rules():
    strat = strategies.load("value_factor")
    assert strat.valid({"w_bp": 1.0, "w_ey": 1.0, "w_gp": 1.0,
                        "max_positions": 15, "rebalance_days": 21})
    assert not strat.valid({"w_bp": 0.0, "w_ey": 0.0, "w_gp": 0.0})   # no weight
    assert not strat.valid({"w_bp": 1.0, "max_positions": 0})         # a book with no slots
    assert not strat.valid({"w_bp": 1.0, "rebalance_days": 0})        # no cadence
