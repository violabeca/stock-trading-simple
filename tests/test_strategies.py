"""Strategy loader contract + both shipped strategies on synthetic data."""

import pandas as pd
import pytest

from trading_simple import strategies


def test_available_lists_strategies_and_skips_template():
    names = strategies.available()
    assert {"volume_momentum", "donchian_breakout"} <= set(names)
    assert "_template" not in names


def test_load_unknown_lists_available():
    with pytest.raises(FileNotFoundError, match="volume_momentum"):
        strategies.load("no_such_strategy")


def test_loaded_strategy_shape():
    strat = strategies.load("volume_momentum")
    assert callable(strat.signals) and callable(strat.valid)
    assert strat.data["bar"] in {"1d", "1m"}
    assert set(strat.optuna["space"]) <= set(strat.params) | {"direction"}


# --- volume_momentum signal logic ---------------------------------------------
def _vm(panel, **overrides):
    strat = strategies.load("volume_momentum")
    params = {**strat.params, "fast": 5, "slow": 20, "vol_window": 10,
              "threshold": 3.0, "horizon": 5, **overrides}
    return strat.signals(panel, **params)


def test_volume_momentum_spike_routes_by_trend(panel):
    sig = _vm(panel)
    # Spike at bar 40 -> execution at bar 41; AAA trends up (long), BBB down (short).
    assert sig["entries"].iloc[41]["AAA"] and not sig["entries"].iloc[41]["BBB"]
    assert sig["short_entries"].iloc[41]["BBB"] and not sig["short_entries"].iloc[41]["AAA"]
    assert not sig["entries"].iloc[40].any()              # never same-bar execution
    assert sig["exits"].iloc[46]["AAA"]                   # horizon 5 bars later
    assert int(sig["entries"].values.sum() + sig["short_entries"].values.sum()) == 2


def test_volume_momentum_direction_narrowing(panel):
    assert not _vm(panel, direction="long")["short_entries"].values.any()
    assert not _vm(panel, direction="short")["entries"].values.any()


def test_volume_momentum_valid_rejects_inverted_mas():
    strat = strategies.load("volume_momentum")
    assert strat.valid({"fast": 10, "slow": 50})
    assert not strat.valid({"fast": 50, "slow": 50})


def test_volume_momentum_indicator_overlays(panel):
    strat = strategies.load("volume_momentum")
    out = strat.indicators(panel.close["AAA"], panel.volume["AAA"],
                           **{**strat.params, "fast": 5, "slow": 20})
    assert set(out) == {"price", "volume"} and len(out["price"]) == 2


# --- donchian_breakout signal logic --------------------------------------------
def _wickless_panel():
    """A panel whose high/low equal close, so every trend bar is a clean breakout."""
    from tests.conftest import make_panel

    p = make_panel()
    return type(p)(close=p.close, open_=p.open_, high=p.close, low=p.close,
                   volume=p.volume, bar=p.bar)


def test_donchian_breaks_out_long_and_short():
    strat = strategies.load("donchian_breakout")
    panel = _wickless_panel()
    sig = strat.signals(panel, entry_window=10, atr_window=5, atr_mult=2.0)
    # With no wicks, a steady uptrend closes above the prior 10-bar high every
    # bar once warmed up (AAA), a steady downtrend below the low (BBB).
    assert sig["entries"]["AAA"].iloc[15:].all()
    assert sig["short_entries"]["BBB"].iloc[15:].all()
    assert not sig["entries"]["BBB"].any() and not sig["short_entries"]["AAA"].any()
    assert sig["sl_trail"] is True
    assert isinstance(sig["sl_stop"], pd.DataFrame)
    valid_stops = sig["sl_stop"].iloc[15:]
    assert (valid_stops > 0).all().all()                  # ATR fraction of price


def test_donchian_channel_excludes_current_bar(panel):
    strat = strategies.load("donchian_breakout")
    sig = strat.signals(panel, entry_window=59, atr_window=5, atr_mult=2.0)
    # Window nearly as long as the data: only the last bars can break out,
    # and a bar can never break out of a channel that includes itself.
    assert not sig["entries"]["AAA"].iloc[:59].any()
