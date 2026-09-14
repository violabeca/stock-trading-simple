"""momentum_swing signal logic on synthetic daily prices (no DB, no network)."""

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

N = 160
BREAK = 100    # LEAD closes above its base on this bar
GAP = 110      # PIVT gaps up on this bar (while LEAD is still running)
CRASH = 130    # LEAD closes far below its 10 SMA on this bar


def _ms(panel, **overrides):
    strat = strategies.load("momentum_swing")
    return strat.signals(panel, **{**strat.params, **overrides})


def _setups(panel, **overrides):
    return strategies.load("momentum_swing").signals.__globals__["setups"](panel, **overrides)


def _panel(dump: bool = False) -> Panel:
    """Four names, one engineered event each.

    ``LEAD`` rises 60% over 80 bars with wide daily ranges, rests 20 bars in a tight base,
    closes above the base on 5x volume at bar ``BREAK``, runs, then crashes through its
    10-day SMA at bar ``CRASH``. ``PIVT`` is flat for 110 bars and gaps up 12% on 8x volume
    at bar ``GAP``, holding the gap. ``UP`` drifts up 0.3% a day forever (keeps the
    equal-weight index rising, never sets up). ``DUMP`` (optional) falls 3% a day, which
    drags the index down so the regime gate has something to block.
    """
    index = pd.date_range("2021-01-04", periods=N, freq="B", tz="UTC")
    t = np.arange(N)
    lead = np.where(t < 80, 100.0 + 60.0 * t / 79.0, 160.0)
    lead[BREAK:] = 166.0 * 1.01 ** (t[BREAK:] - BREAK)      # breakout, then a steady run
    lead[CRASH:] = lead[CRASH - 1] * 0.85                    # ... until it caves
    pivt = np.full(N, 50.0)
    pivt[GAP:] = 57.0
    up = 100.0 * 1.003 ** t
    close = pd.DataFrame({"LEAD": lead, "PIVT": pivt, "UP": up}, index=index)
    # Daily range as a fraction of the close: wide in the leader move, tight in the base.
    rng = pd.DataFrame(0.02, index=index, columns=close.columns)
    rng.loc[index[:80], "LEAD"] = 0.03
    rng.loc[index[80:BREAK], "LEAD"] = 0.005
    high, low = close * (1 + rng), close * (1 - rng)
    open_ = close * 0.999
    volume = pd.DataFrame(1_000.0, index=index, columns=close.columns)
    # The breakout bar: closes near the high on 5x volume.
    high.loc[index[BREAK], "LEAD"], low.loc[index[BREAK], "LEAD"] = 167.0, 161.0
    volume.loc[index[BREAK], "LEAD"] = 5_000.0
    # The gap bar: opens +12%, holds, on 8x volume.
    open_.loc[index[GAP], "PIVT"] = 56.0
    high.loc[index[GAP], "PIVT"], low.loc[index[GAP], "PIVT"] = 58.0, 55.5
    volume.loc[index[GAP], "PIVT"] = 8_000.0
    if dump:
        d = 100.0 * 0.97 ** t
        close["DUMP"], open_["DUMP"] = d, d * 0.999
        high["DUMP"], low["DUMP"], volume["DUMP"] = d * 1.02, d * 0.98, 1_000.0
    return Panel(close=close, open_=open_, high=high, low=low, volume=volume, bar="1d")


def test_breakout_enters_next_bar_with_a_signal_bar_stop():
    p = _panel()
    sig = _ms(p)
    e = sig["entries"]
    assert e["LEAD"].iloc[BREAK + 1]                          # fills the bar AFTER the close
    assert not e["LEAD"].iloc[BREAK]                          # never same-bar
    assert not e["LEAD"].iloc[:BREAK].any()                   # nothing during the move/base
    assert not e["UP"].any()                                  # a smooth drift never sets up
    # The stop is decided on the signal bar (ATR / close there) and applied at the fill bar.
    atr = bt.atr(p, 14)
    want = 1.5 * atr["LEAD"].iloc[BREAK] / p.close["LEAD"].iloc[BREAK]
    assert np.isclose(sig["sl_stop"]["LEAD"].iloc[BREAK + 1], want)
    assert sig["sl_trail"] is False and "tp_stop" not in sig  # fixed stop, no target by default
    assert sig["cash_frac"] == 0.25                           # one shared wallet
    with_tp = _ms(p, tp_r=3.0)
    assert np.isclose(with_tp["tp_stop"]["LEAD"].iloc[BREAK + 1], 3.0 * want)


def test_pivot_enters_after_the_gap_day_and_each_leg_can_be_switched_off():
    p = _panel()
    both = _ms(p)["entries"]
    assert both["PIVT"].iloc[GAP + 1] and not both["PIVT"].iloc[GAP]
    assert int(both["PIVT"].sum()) == 1                       # one gap, one entry
    only_bo = _ms(p, setup="breakout")["entries"]
    assert only_bo["LEAD"].iloc[BREAK + 1] and not only_bo["PIVT"].any()
    only_ep = _ms(p, setup="pivot")["entries"]
    assert only_ep["PIVT"].iloc[GAP + 1] and not only_ep["LEAD"].any()


def test_trail_exit_fires_the_bar_after_a_close_below_the_sma():
    p = _panel()
    x = _ms(p)["exits"]
    assert x["LEAD"].iloc[CRASH + 1]                          # close below SMA 10 -> next open
    assert not x["LEAD"].iloc[BREAK + 1:CRASH].any()          # the run stays above its trail
    slow = _ms(p, trail_sma=40)["exits"]                      # a slower trail: the crash bar
    assert not slow["LEAD"].iloc[CRASH + 1]                   # (−15%) is still above SMA 40


def test_regime_gate_blocks_entries_in_a_falling_tape():
    p = _panel(dump=True)                                     # DUMP drags the EW index down
    assert not _ms(p, regime="fast")["entries"]["LEAD"].any()
    assert not _ms(p, regime="slow", regime_long=50)["entries"]["LEAD"].any()
    assert _ms(p, regime="none")["entries"]["LEAD"].iloc[BREAK + 1]
    parts = _setups(p, regime="fast")
    assert not parts["regime_ok"].iloc[BREAK]                 # the gate, not the setup, said no
    assert parts["breakout"]["LEAD"].iloc[BREAK]


def test_filters_that_kill_the_setup():
    p = _panel()
    assert not _ms(p, max_stop=0.005)["entries"].any().any()      # stop too wide -> skip
    assert not _ms(p, vol_mult=10.0)["entries"]["LEAD"].any()      # no volume -> no breakout
    assert not _ms(p, lead_min=0.9)["entries"]["LEAD"].any()       # not enough of a leader
    assert not _ms(p, contraction=0.1)["entries"]["LEAD"].any()    # base not tight enough
    assert not _ms(p, gap_min=0.2)["entries"]["PIVT"].any()        # gap too small
    assert not _ms(p, gap_vol_mult=20.0)["entries"]["PIVT"].any()  # gap without volume
    assert not _ms(p, adr_min=0.06)["entries"].any().any()         # too quiet a name
    # A name that had ALREADY rallied is a leader, not a neglected pivot.
    assert not _setups(p, lead_min=-0.5)["pivot"]["PIVT"].any()


def test_valid_rules():
    strat = strategies.load("momentum_swing")
    assert strat.valid(dict(strat.params))
    assert not strat.valid({"setup": "short"})
    assert not strat.valid({"regime": "vix"})
    assert not strat.valid({"base_window": 63, "lead_window": 63})   # base must be shorter
    assert not strat.valid({"regime_fast": 20, "regime_slow": 10})
    assert not strat.valid({"stop_atr": 0.0})
    assert not strat.valid({"max_stop": 1.5})
    assert not strat.valid({"tp_r": -1.0})


@pytest.mark.parametrize("regime", ["none", "fast"])
def test_it_runs_through_the_engine_end_to_end(regime):
    """The full contract: signals -> one cash-shared wallet -> stats, no NaN objectives."""
    from trading_simple.config import load_settings

    settings = load_settings()
    strat = strategies.load("momentum_swing")
    stats = bt.evaluate(strat, _panel(), {**strat.params, "regime": regime}, settings=settings)
    assert stats["n_trades"] >= 2                             # LEAD and PIVT both traded
    assert np.isfinite(stats["excess_cagr"]) and np.isfinite(stats["tracking_error"])
