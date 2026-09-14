"""turnover_momentum signal logic on synthetic prices, volume and shares (no DB, no network)."""

import numpy as np
import pandas as pd

from trading_simple import strategies
from trading_simple.data import Panel


def _tm(panel, **overrides):
    strat = strategies.load("turnover_momentum")
    return strat.signals(panel, **{**strat.params, **overrides})


def _module():
    """The strategy module (the loader binds only signals/valid)."""
    return strategies.load("turnover_momentum").signals.__globals__


def _scores(panel, **overrides):
    return _module()["scores"](panel, **overrides)


def _panel(n=120, shares=True, rates=(0.004, 0.002, 0.0, -0.003)):
    """Four names on smooth daily drifts; DDD trades a tenth of everyone else's volume.

    Defaults: AAA rises fastest, BBB rises, CCC is flat, DDD falls. Shares outstanding are
    identical, so turnover ranks on volume alone — AAA/BBB/CCC tie at the top, DDD is the
    low-turnover name.
    """
    index = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    t = np.arange(n)
    close = pd.DataFrame(
        {sym: 100.0 * (1.0 + r) ** t
         for sym, r in zip(("AAA", "BBB", "CCC", "DDD"), rates, strict=True)},
        index=index)
    volume = pd.DataFrame(1e6, index=index, columns=close.columns)
    volume["DDD"] = 1e5
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.01, low=close * 0.99,
                  volume=volume, bar="1d")
    if shares:
        panel.features = {"fund_shares": pd.DataFrame(1e8, index=index, columns=close.columns)}
    return panel


def test_top_momentum_name_enters_next_bar_after_warmup():
    p = _panel()
    sig = _tm(p, max_positions=1, formation=21, turnover_window=21, rebalance_days=21)
    w = sig["weights"]
    # Rebalance bars are 0, 21, 42, ...; bar 0 is inside the warm-up (no 21-bar return yet),
    # so the first pick is decided on bar 21 and executed at the open of bar 22 — never the
    # same bar.
    assert w.iloc[:22].to_numpy().sum() == 0
    assert w["AAA"].iloc[22] == 1.0
    assert (w["AAA"].iloc[22:] == 1.0).all()                 # fastest riser: held throughout
    assert not (w[["BBB", "CCC", "DDD"]] > 0).any().any()    # one slot -> only AAA
    assert sig["retrim_k"] == 2.0                            # the concentration cap rides along


def test_low_turnover_winner_is_excluded_until_the_gate_is_off():
    """The thesis knob: the best mover is not bought if it is thinly traded."""
    p = _panel(rates=(0.004, 0.002, 0.0, 0.006))             # DDD now rises FASTEST, thin volume
    with_gate = _tm(p, max_positions=1, turnover_window=21, turnover_q=0.5)["weights"]
    assert not (with_gate["DDD"] > 0).any()                  # dropped from the cross-section
    assert (with_gate["AAA"] > 0).any()                      # the best ELIGIBLE name instead
    no_gate = _tm(p, max_positions=1, turnover_window=21, turnover_q=0.0)["weights"]
    assert (no_gate["DDD"] > 0).any()                        # plain momentum: DDD is the pick
    assert not (no_gate["AAA"] > 0).any()


def test_reversal_polarity_picks_the_loser():
    p = _panel()
    w = _tm(p, max_positions=1, turnover_window=21, turnover_q=0.0, polarity="reversal")["weights"]
    assert (w["DDD"] > 0).any()                              # the faller
    assert not (w["AAA"] > 0).any()


def test_selection_is_carried_between_rebalances():
    """Turnover is paced by the cadence: the book does not react until the next rebalance."""
    p = _panel(n=100)
    # BBB jumps 50% on bar 62 — inside the 21-bar window that the bar-80 re-selection reads
    # (bars 59..80) and long before it, so by bar 70 BBB already out-scores AAA.
    p.close.loc[p.close.index[62]:, "BBB"] *= 1.5
    p.open_ = p.close * 0.999
    p.high, p.low = p.close * 1.01, p.close * 0.99
    w = _tm(p, max_positions=1, formation=21, turnover_window=21, rebalance_days=40)["weights"]
    assert w["AAA"].iloc[71] > 0 and w["BBB"].iloc[71] == 0  # mid-cycle: the bar-40 pick is kept
    assert w["BBB"].iloc[81] > 0 and w["AAA"].iloc[81] == 0  # after the bar-80 re-selection


def test_split_in_the_share_count_does_not_step_turnover():
    """A 2:1 jump in as-reported shares is a unit change and must not move the turnover level."""
    p = _panel()
    idx = p.close.index
    p.features["fund_shares"].loc[idx[60]:, "AAA"] = 2e8     # filing after a 2:1 split
    adj = _module()["_in_volume_units"](p.features["fund_shares"])
    assert np.allclose(adj["AAA"], 2e8)                       # earlier bars restated into new units
    assert np.allclose(adj["BBB"], 1e8)                       # untouched names unchanged
    turn = _scores(p, turnover_window=21)["turnover"]["AAA"]
    assert np.isclose(turn.iloc[59], turn.iloc[80])           # constant volume -> constant turnover
    # A small buyback-sized change is NOT a split and passes through as a real level change.
    p.features["fund_shares"].loc[idx[60]:, "BBB"] = 0.97e8
    adj2 = _module()["_in_volume_units"](p.features["fund_shares"])
    assert np.isclose(adj2["BBB"].iloc[0], 1e8) and np.isclose(adj2["BBB"].iloc[-1], 0.97e8)


def test_missing_shares_fall_back_to_dollar_volume():
    p = _panel(shares=False)
    parts = _scores(p, turnover_window=21)
    assert parts["turnover_source"] == "dollar_volume"
    assert (_scores(_panel(), turnover_window=21)["turnover_source"]) == "shares"
    w = _tm(p, max_positions=1, turnover_window=21)["weights"]
    assert (w["AAA"] > 0).any()                              # it still trades


def test_the_book_is_always_a_valid_portfolio():
    """Never oversubscribed, never levered, always whole slots (1/max_positions)."""
    p = _panel()
    for max_positions in (1, 2, 3):
        weights = _tm(p, max_positions=max_positions, turnover_window=21)["weights"]
        held = (weights > 0).sum(axis=1)
        assert held.max() <= max_positions
        assert weights.sum(axis=1).max() <= 1.0 + 1e-12
        nonzero = weights.to_numpy()[weights.to_numpy() > 0]
        assert np.allclose(nonzero, 1.0 / max_positions)


def test_valid_rules():
    strat = strategies.load("turnover_momentum")
    assert strat.valid(dict(strat.params))
    assert not strat.valid({"turnover_q": 1.0})              # nobody could ever be eligible
    assert not strat.valid({"polarity": "sideways"})
    assert not strat.valid({"max_positions": 0})
    assert not strat.valid({"rebalance_days": 0})
    assert not strat.valid({"skip": -1})
