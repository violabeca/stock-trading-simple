"""Strategy loader contract + both shipped strategies on synthetic data."""

import pandas as pd
import pytest

from trading_simple import backtest as bt
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
    assert set(strat.optuna["space"]) <= set(strat.params)


# --- the unified params: block ------------------------------------------------
def _write(tmp_path, params: str):
    """A minimal strategy folder whose config carries the given params block."""
    folder = tmp_path / "strategies" / "probe"
    folder.mkdir(parents=True)
    (folder / "strategy.py").write_text("def signals(panel, **p):\n    return {}\n")
    (folder / "config.yaml").write_text(f"data:\n  bar: 1d\nparams:\n{params}")
    return folder


@pytest.fixture
def probe_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(strategies, "STRATEGIES_DIR", tmp_path / "strategies")
    return tmp_path


def test_params_block_derives_all_three_views(probe_dir):
    _write(probe_dir, """  window:
    doc: how far back
    value: 20
    search: {type: int, low: 5, high: 100}
    grid: [10, 20]
  pinned: 0.25
""")
    strat = strategies.load("probe")
    assert strat.params == {"window": 20, "pinned": 0.25}      # scalar shorthand included
    assert strat.sweep == {"window": [10, 20]}                 # `pinned` has no grid ...
    assert strat.optuna["space"] == {"window": {"type": "int", "low": 5, "high": 100}}
    assert strat.docs == {"window": "how far back"}


def test_value_outside_its_own_search_is_rejected(probe_dir):
    """A shipped value the study can never suggest makes every boundary report a fiction."""
    _write(probe_dir, """  window:
    value: 200
    search: {type: int, low: 5, high: 100}
""")
    with pytest.raises(ValueError, match="search space .* cannot reach"):
        strategies.load("probe")


def test_grid_that_cannot_express_the_shipped_value_is_rejected(probe_dir):
    """THE regression: a walk-forward re-picking from this grid validates another strategy.

    It shipped twice — informed_flow's grid pinned q_in=0.75/max_positions=15 while the
    study ran 0.85/8, so four folds faithfully scored a strategy nobody runs, with no error
    anywhere. The config now refuses to load instead.
    """
    _write(probe_dir, """  window:
    value: 30
    grid: [10, 20]
""")
    with pytest.raises(ValueError, match="validating a strategy this config does not run"):
        strategies.load("probe")


def test_categorical_value_must_be_a_choice(probe_dir):
    _write(probe_dir, """  mode:
    value: sideways
    search: {type: categorical, choices: [long, short]}
""")
    with pytest.raises(ValueError, match="cannot reach"):
        strategies.load("probe")


def test_typo_in_an_entry_key_is_rejected(probe_dir):
    """`serach:` would otherwise mean 'not searched' — silently, and forever."""
    _write(probe_dir, """  window:
    value: 20
    serach: {type: int, low: 5, high: 100}
""")
    with pytest.raises(ValueError, match="unknown key"):
        strategies.load("probe")


def test_backtest_block_selects_the_fill_and_rejects_typos(probe_dir):
    """`backtest.fill` defaults to open; a misspelt key or model fails at load, not silently."""
    cfg = _write(probe_dir, "  window: 20\n") / "config.yaml"
    base = cfg.read_text()
    assert strategies.load("probe").fill == "open"                  # no block -> default
    cfg.write_text(base + "backtest:\n  fill: pessimistic\n")
    assert strategies.load("probe").fill == "pessimistic"
    for bad in ("backtest:\n  fill: worst\n", "backtest:\n  fil: pessimistic\n"):
        cfg.write_text(base + bad)
        with pytest.raises(ValueError, match="backtest"):
            strategies.load("probe")


def test_order_notional_prices_the_minimum_commission_at_the_real_order_size(probe_dir):
    """`backtest.order_notional` is the strategy's real order in dollars; absent = $10,000.

    make_panel's median close is $100, so the $1 minimum binds at both sizes: 1 bp/leg on a
    $10,000 order, 10 bp/leg on the $1,000 orders of a 30-slot book on a $30k account. At the
    default size every fee fraction is unchanged, which is what keeps every other strategy's
    numbers where they were.
    """
    from tests.conftest import make_panel
    from trading_simple.config import Settings

    cfg = _write(probe_dir, "  window: 20\n") / "config.yaml"
    base, panel, settings = cfg.read_text(), make_panel(), Settings()
    plain = strategies.load("probe")
    assert plain.order_notional is None
    assert bt.strategy_costs(plain, panel, settings) == bt.panel_costs(settings.costs, panel)
    cfg.write_text(base + "backtest:\n  order_notional: 1000\n")
    small = strategies.load("probe")
    assert small.order_notional == 1000.0
    fees, slip = bt.strategy_costs(small, panel, settings)
    assert fees == pytest.approx(1.0 / 1_000 + 1.0 / 1e4)      # $1 on $1,000 + half the spread
    assert slip == pytest.approx(1.0 / 1e4)
    cfg.write_text(base + "backtest:\n  order_notional: 0\n")
    with pytest.raises(ValueError, match="order_notional"):
        strategies.load("probe")


@pytest.mark.parametrize("name", strategies.available())
def test_every_shipped_config_satisfies_its_own_invariants(name):
    """Every strategy in the repo — the check that keeps the fleet aligned, not just one.

    The final assert goes one step beyond what :func:`strategies._unpack` can see: it
    builds the real grid, so a shipped value that survives the declaration check but is
    filtered out by ``valid()`` (an exit band crossing its entry band, say) still fails.
    """
    strat = strategies.load(name)                    # raises if either invariant is broken
    for key, grid in strat.sweep.items():
        assert strat.params[key] in grid, f"{name}.{key} ships a value its grid excludes"
    assert set(strat.optuna["space"]) <= set(strat.params)
    assert not set(strat.docs) - set(strat.params)

    combos = bt.grid_from_config(strat.sweep, strat.valid)
    assert combos or not strat.sweep, f"{name}: valid() rejects every grid combo"
    for key in strat.sweep:
        assert strat.params[key] in {c[key] for c in combos}, (
            f"{name}: valid() filters every combo carrying the shipped {key}="
            f"{strat.params[key]}; a walk-forward could never re-pick it")


# --- volume_momentum signal logic ---------------------------------------------
def _vm(panel, **overrides):
    strat = strategies.load("volume_momentum")
    params = {**strat.params, "fast": 5, "slow": 20, "vol_window": 10,
              "threshold": 3.0, "horizon": 5, **overrides}
    return strat.signals(panel, **params)


def test_volume_momentum_spike_routes_by_trend(panel):
    sig = _vm(panel)
    # Spike at bar 40 -> execution at bar 41; AAA trends up (long), BBB down (short) —
    # both directions always live, the momentum sign alone picks the side.
    assert sig["entries"].iloc[41]["AAA"] and not sig["entries"].iloc[41]["BBB"]
    assert sig["short_entries"].iloc[41]["BBB"] and not sig["short_entries"].iloc[41]["AAA"]
    assert not sig["entries"].iloc[40].any()              # never same-bar execution
    assert sig["exits"].iloc[46]["AAA"]                   # horizon 5 bars later
    assert int(sig["entries"].values.sum() + sig["short_entries"].values.sum()) == 2


def test_volume_momentum_stop_params_passthrough(panel):
    sig = _vm(panel, sl_pct=0.02, sl_trail=True)
    assert sig["sl_stop"] == 0.02 and sig["sl_trail"] is True
    assert "sl_stop" not in _vm(panel, sl_pct=0.0)        # 0 disables the stop entirely
    # One shared wallet: the cash fraction rides along unconditionally.
    assert sig["cash_frac"] == 0.25
    assert _vm(panel, cash_frac=0.5)["cash_frac"] == 0.5


def _intraday_panel():
    """Two 91-minute sessions; AAA trends up, BBB down; day-2 volume spikes
    near the open (minute 2), mid-session (minute 45) and near the close
    (minute 87) so the session guards have something to block."""
    import numpy as np

    from trading_simple.data import Panel

    tz = "America/New_York"
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-06-0{d} 09:30", tz=tz) + pd.Timedelta(minutes=m)
                            for d in (1, 2) for m in range(91)]).tz_convert("UTC")
    n = len(idx)
    close = pd.DataFrame({"AAA": np.linspace(100.0, 110.0, n),
                          "BBB": np.linspace(100.0, 92.0, n)}, index=idx)
    volume = pd.DataFrame(1_000.0, index=idx, columns=close.columns)
    volume.iloc[[91 + 2, 91 + 45, 91 + 87]] = 20_000.0
    return Panel(close=close, open_=close * 0.999, high=close * 1.001,
                 low=close * 0.999, volume=volume, bar="1m")


def test_volume_momentum_session_guards_intraday():
    strat = strategies.load("volume_momentum")
    sig = strat.signals(_intraday_panel(), fast=5, slow=20, threshold=3.0, horizon=5,
                        baseline="sma", vol_window=10, skip_open=15, skip_close=15,
                        sl_pct=0.0, eod_exit=True)
    day2 = 91
    assert sig["entries"]["AAA"].iloc[day2 + 46]          # mid-session spike fills next bar
    assert not sig["entries"]["AAA"].iloc[day2 + 3]       # blocked: first 15 minutes
    assert not sig["entries"]["AAA"].iloc[day2 + 88]      # blocked: last 15 minutes
    assert sig["short_entries"]["BBB"].iloc[day2 + 46]    # downtrend routes short
    assert sig["exits"].iloc[90].all()                    # eod_exit: flat on day-1 close
    assert sig["exits"].iloc[-1].all()                    # ... and on day-2 close


def test_volume_momentum_valid_rejects_inverted_mas():
    strat = strategies.load("volume_momentum")
    assert strat.valid({"fast": 10, "slow": 50})
    assert not strat.valid({"fast": 50, "slow": 50})


def test_volume_momentum_indicator_overlays(panel):
    strat = strategies.load("volume_momentum")
    out = strat.indicators(panel.close["AAA"], panel.volume["AAA"],
                           **{**strat.params, "fast": 5, "slow": 20, "vol_window": 10})
    assert set(out) == {"price", "momentum", "volume", "volume_marks"}
    assert len(out["price"]) == 2
    momentum = next(iter(out["momentum"].values()))
    assert momentum.iloc[-1] > 0                          # AAA trends up -> long bias
    assert out["volume_marks"].iloc[40] and out["volume_marks"].sum() == 1  # the bar-40 spike


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
    # One shared wallet: the cash fraction rides along unconditionally.
    assert sig["cash_frac"] == 0.25
    assert strat.signals(panel, entry_window=10, atr_window=5, atr_mult=2.0,
                         cash_frac=0.5)["cash_frac"] == 0.5


def test_donchian_channel_excludes_current_bar(panel):
    strat = strategies.load("donchian_breakout")
    sig = strat.signals(panel, entry_window=59, atr_window=5, atr_mult=2.0)
    # Window nearly as long as the data: only the last bars can break out,
    # and a bar can never break out of a channel that includes itself.
    assert not sig["entries"]["AAA"].iloc[:59].any()
    assert "exits" not in sig                             # daily bars: no eod_exit frames


def test_donchian_session_guards_and_eod_exit_intraday():
    # Wickless intraday panel: with high/low == close every trend bar breaks
    # out, so the session guards' effect is directly observable.
    p = _intraday_panel()
    panel = type(p)(close=p.close, open_=p.open_, high=p.close, low=p.close,
                    volume=p.volume, bar="1m")
    strat = strategies.load("donchian_breakout")
    sig = strat.signals(panel, entry_window=10, atr_window=5, atr_mult=2.0,
                        skip_open=15, skip_close=15, eod_exit=True)
    day2 = 91
    assert sig["entries"]["AAA"].iloc[day2 + 45]          # mid-session breakout fires
    assert not sig["entries"]["AAA"].iloc[:15].any()      # blocked: first 15 minutes
    assert not sig["entries"]["AAA"].iloc[day2 + 76:].any()  # blocked: last 15 minutes
    assert sig["short_entries"]["BBB"].iloc[day2 + 45]    # downtrend routes short
    assert sig["exits"].iloc[90].all()                    # eod_exit: flat on day-1 close
    assert sig["exits"].iloc[-1].all()                    # ... and on day-2 close
