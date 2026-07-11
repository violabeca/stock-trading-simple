"""squeeze_breakout signal logic on synthetic data with one engineered confluence event."""

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

#: Small windows so the 60-bar synthetic panel warms up quickly.
_PARAMS = {"break_window": 10, "trend_window": 40, "atr_window": 5,
           "squeeze_window": 20, "squeeze_lag": 2, "squeeze_thresh": 0.8,
           "vol_mult": 3.0, "vol_bars": 2, "profile_days": 5,
           "sl_mult": 2.0, "tp_mult": 0.0, "horizon": 8}
_EVENT = 45


def _sb(panel, **overrides):
    strat = strategies.load("squeeze_breakout")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _panel(*, tight=True, spike=True, jump_a=4.0, jump_b=-4.0):
    """Two-symbol daily panel with one engineered confluence event at bar 45.

    AAA trends up, BBB down. Bars 30–44 hold a FLAT, tight-range zone (0.2%
    bands instead of 1%), so the short ATR drops well below its 20-bar
    baseline — a genuine squeeze — while no bar breaks the 10-bar channel.
    Bar 45 jumps ``jump_a``/``jump_b`` from the flat level on 10x volume,
    then closes stay at the event level so nothing re-triggers. Each keyword
    knocks out exactly one gate: ``spike=False`` the volume gate,
    ``tight=False`` the squeeze, ``jump=0`` the breakout, and reversed jump
    signs put the break against the trend.
    """
    n = 60
    index = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    a, b = np.linspace(100.0, 130.0, n), np.linspace(100.0, 70.0, n)
    if tight:
        a[30:_EVENT], b[30:_EVENT] = a[30], b[30]
    a[_EVENT], b[_EVENT] = a[_EVENT - 1] + jump_a, b[_EVENT - 1] + jump_b
    a[_EVENT + 1:], b[_EVENT + 1:] = a[_EVENT], b[_EVENT]
    band = np.full(n, 0.01)
    if tight:
        band[30:_EVENT] = 0.002
    close = pd.DataFrame({"AAA": a, "BBB": b}, index=index)
    volume = pd.DataFrame(1_000.0, index=index, columns=close.columns)
    if spike:
        volume.iloc[_EVENT] = 10_000.0
    return Panel(close=close, open_=close * 0.999,
                 high=close * (1.0 + band[:, None]), low=close * (1.0 - band[:, None]),
                 volume=volume, bar="1d")


def test_confluence_enters_next_bar_once():
    sig = _sb(_panel())
    # Event at bar 45 -> execution at 46; the flat squeeze zone never breaks
    # the channel and the flat tail never re-triggers: exactly one long (AAA,
    # with the uptrend) and one short (BBB, with the downtrend).
    assert not sig["entries"].iloc[_EVENT].any()
    assert list(np.where(sig["entries"]["AAA"])[0]) == [_EVENT + 1]
    assert list(np.where(sig["short_entries"]["BBB"])[0]) == [_EVENT + 1]
    assert not sig["entries"]["BBB"].any() and not sig["short_entries"]["AAA"].any()
    # Time stop: the only exits land exactly horizon bars after the entries.
    assert list(np.where(sig["exits"]["AAA"])[0]) == [_EVENT + 1 + 8]
    assert list(np.where(sig["short_exits"]["BBB"])[0]) == [_EVENT + 1 + 8]


def test_volume_gate_vetoes_and_disables():
    quiet = _panel(spike=False)                     # squeeze+break+trend pass, volume fails
    sig = _sb(quiet)
    assert not sig["entries"].any().any() and not sig["short_entries"].any().any()
    sig = _sb(quiet, vol_mult=0.0)                  # 0 disables the gate -> event trades
    assert list(np.where(sig["entries"]["AAA"])[0]) == [_EVENT + 1]
    assert list(np.where(sig["short_entries"]["BBB"])[0]) == [_EVENT + 1]


def test_volume_gate_uses_kbar_sums_on_both_sides():
    # (a) Building participation: volume 5,500 then 1,500 into the break. The
    # 2-bar sum (7,000 >= 3 x 2,000) passes while the break bar alone (1,500 <
    # 3 x 1,000) would not — a regression to a single-bar gate loses this entry.
    building = _panel(spike=False)
    building.volume.iloc[_EVENT - 1] = 5_500.0
    building.volume.iloc[_EVENT] = 1_500.0
    sig = _sb(building)
    assert sig["entries"]["AAA"].iloc[_EVENT + 1]
    assert sig["short_entries"]["BBB"].iloc[_EVENT + 1]
    # (b) Matching baseline sum: 100 then 5,500 totals 5,600 < 3 x 2,000 — the
    # gate must compare the k-bar sum to the k-bar BASELINE sum; comparing it
    # to the per-bar baseline (3 x 1,000) would wrongly admit this entry.
    lopsided = _panel(spike=False)
    lopsided.volume.iloc[_EVENT - 1] = 100.0
    lopsided.volume.iloc[_EVENT] = 5_500.0
    sig = _sb(lopsided)
    assert not sig["entries"].any().any() and not sig["short_entries"].any().any()


def test_squeeze_gate_vetoes_and_disables():
    loud = _panel(tight=False)                      # wide tape: ATR ~ its baseline everywhere
    sig = _sb(loud)
    assert not sig["entries"].any().any() and not sig["short_entries"].any().any()
    sig = _sb(loud, squeeze_thresh=0.0)             # 0 disables the gate -> event trades
    assert list(np.where(sig["entries"]["AAA"])[0]) == [_EVENT + 1]
    assert list(np.where(sig["short_entries"]["BBB"])[0]) == [_EVENT + 1]


def test_trend_gate_vetoes_and_disables():
    # Breaks AGAINST the drift: AAA (uptrend) breaks down, BBB (downtrend) up.
    against = _panel(jump_a=-2.5, jump_b=2.5)
    sig = _sb(against)
    for key in ("entries", "short_entries"):
        assert not sig[key].any().any()
    sig = _sb(against, trend_window=0)              # 0 disables the gate -> both trade
    assert sig["short_entries"]["AAA"].iloc[_EVENT + 1]
    assert sig["entries"]["BBB"].iloc[_EVENT + 1]


def test_no_breakout_no_entry():
    flat = _panel(jump_a=0.0, jump_b=0.0)           # squeeze+volume+trend pass, no break
    sig = _sb(flat)
    assert not sig["entries"].any().any() and not sig["short_entries"].any().any()


def test_direction_gates_sides():
    panel = _panel()
    long_only = _sb(panel, direction="long")
    assert long_only["entries"]["AAA"].iloc[_EVENT + 1]
    assert not long_only["short_entries"].any().any()
    short_only = _sb(panel, direction="short")
    assert short_only["short_entries"]["BBB"].iloc[_EVENT + 1]
    assert not short_only["entries"].any().any()


def test_stop_scale_uses_normal_volatility():
    panel = _panel()
    sig = _sb(panel, tp_mult=3.0)
    rng = bt.atr(panel, _PARAMS["atr_window"])
    base = bt.sma(rng, _PARAMS["squeeze_window"])
    expected = (2.0 * rng.clip(lower=base) / panel.close).clip(lower=1e-4)
    pd.testing.assert_frame_equal(sig["sl_stop"], expected)
    # In the compressed zone the spot ATR is far below its baseline: the stop
    # must reference NORMAL volatility, not the squeeze-certified reading.
    spot_only = float(2.0 * rng.iloc[_EVENT - 1]["AAA"] / panel.close.iloc[_EVENT - 1]["AAA"])
    assert float(sig["sl_stop"].iloc[_EVENT - 1]["AAA"]) > spot_only
    # tp_stop shares the scale (3.0 vs 2.0 multiples) and 0 removes it.
    pd.testing.assert_frame_equal(sig["tp_stop"], expected * 1.5)
    assert "tp_stop" not in _sb(panel, tp_mult=0.0)


def test_valid_rejects_incoherent_combos():
    strat = strategies.load("squeeze_breakout")
    assert strat.valid({"trend_window": 390, "break_window": 60})
    assert not strat.valid({"trend_window": 100, "break_window": 60})   # trend < 2x channel
    assert strat.valid({"trend_window": 0, "break_window": 240})        # gate off -> free
    assert strat.valid({"tp_mult": 0.0, "sl_mult": 5.0})                # no target -> free
    assert not strat.valid({"tp_mult": 0.5, "sl_mult": 2.0})            # degenerate 1:4 bracket
    assert strat.valid({"tp_mult": 1.0, "sl_mult": 2.0})
    assert strat.valid({"squeeze_thresh": 0.7})                         # partial dicts tolerated


def test_intraday_session_guards_and_eod():
    # Two 91-minute sessions, AAA trending up, with two engineered one-bar
    # breakouts on day 2: mid-session (minute 45, tradable) and near the open
    # (minute 5, inside the skip_open blackout). Squeeze/volume gates are
    # disabled (0-switches) to isolate the session logic.
    tz = "America/New_York"
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-06-0{d} 09:30", tz=tz) + pd.Timedelta(minutes=m)
                            for d in (1, 2) for m in range(91)]).tz_convert("UTC")
    n, day2 = len(idx), 91
    close = pd.DataFrame({"AAA": np.linspace(100.0, 110.0, n),
                          "BBB": np.linspace(100.0, 92.0, n)}, index=idx)
    close.loc[close.index[day2 + 45], "AAA"] += 1.2
    close.loc[close.index[day2 + 5], "AAA"] += 1.2
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.001,
                  low=close * 0.999,
                  volume=pd.DataFrame(1_000.0, index=idx, columns=close.columns), bar="1m")

    sig = _sb(panel, squeeze_thresh=0.0, vol_mult=0.0, trend_window=60,
              horizon=30, direction="long", skip_open=15, skip_close=15, eod_exit=True)
    assert list(np.where(sig["entries"]["AAA"])[0]) == [day2 + 46]  # near-open break blocked
    assert sig["exits"].iloc[90].all()                    # eod_exit: flat on day-1 close
    assert sig["exits"].iloc[-1].all()                    # ... and on day-2 close


def test_same_bar_sl_tp_resolves_to_stop():
    # The bracket relies on vectorbt's pessimistic tie-break: when one bar
    # spans both the stop and the target, the STOP fills. Pin it so an engine
    # upgrade that flips the convention is caught here, not in a study.
    n = 10
    index = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    close = pd.DataFrame({"AAA": np.full(n, 100.0)}, index=index)
    high, low = close.copy(), close.copy()
    high.iloc[5], low.iloc[5] = 130.0, 70.0               # hits 5% stop AND 5% target
    panel = Panel(close=close, open_=close * 0.999, high=high, low=low,
                  volume=pd.DataFrame(1_000.0, index=index, columns=close.columns), bar="1d")
    entries = pd.DataFrame(False, index=index, columns=close.columns)
    entries.iloc[3] = True
    pf = bt.run(panel, {"entries": entries, "sl_stop": 0.05, "tp_stop": 0.05},
                fees=0.0, slippage=0.0)
    trades = bt.trade_records(pf)
    assert len(trades) == 1
    assert trades["pnl"].iloc[0] < 0                      # the loss was booked, not the win
    assert abs(trades["exit_price"].iloc[0] - 95.0) < 1.0  # ~the stop level, not ~105
