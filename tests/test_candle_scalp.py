"""candle_scalp signal logic on synthetic data with engineered candles."""

import numpy as np
import pandas as pd

from trading_simple import strategies

#: Small windows so the 60-bar synthetic panel warms up quickly.
_PARAMS = {"band_window": 10, "trend_window": 40, "atr_window": 10,
           "stretch_mult": 0.5, "wick_mult": 2.0, "horizon": 10,
           "vol_mult": 0.0, "sl_mult": 1.0, "tp_mult": 0.0}


def _sig(panel, **overrides):
    strat = strategies.load("candle_scalp")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _event_panel():
    """make_panel with a bullish hammer on AAA and a bearish shooting star on BBB.

    At bar 45 AAA gaps down 8.0 (below its lower band, still above SMA(40)) and
    prints a long lower wick that recovers to the close — a hammer. BBB mirrors
    it upward into a shooting star. Exactly one long (AAA) and one short (BBB)
    setup exist.
    """
    from tests.conftest import make_panel

    p = make_panel()
    close = p.close.copy()
    open_, high, low = (p.close * 0.999).copy(), (p.close * 1.01).copy(), (p.close * 0.99).copy()
    b = close.index[45]
    # AAA: bullish hammer at a stretched dip.
    close.loc[b, "AAA"] -= 8.0
    ca = close.loc[b, "AAA"]
    open_.loc[b, "AAA"], high.loc[b, "AAA"], low.loc[b, "AAA"] = ca - 0.2, ca, ca - 3.0
    # BBB: bearish shooting star at a stretched pop.
    close.loc[b, "BBB"] += 8.0
    cb = close.loc[b, "BBB"]
    open_.loc[b, "BBB"], low.loc[b, "BBB"], high.loc[b, "BBB"] = cb + 0.2, cb, cb + 3.0
    return type(p)(close=close, open_=open_, high=high, low=low, volume=p.volume, bar=p.bar)


def test_hammer_enters_next_bar_and_time_stops():
    sig = _sig(_event_panel(), direction="both")
    # Hammer/star at bar 45 -> execution at bar 46, and the steady trend never
    # re-triggers: exactly one long (AAA) and one short (BBB) entry.
    assert sig["entries"].iloc[46]["AAA"] and not sig["entries"].iloc[45].any()
    assert list(np.where(sig["entries"]["AAA"])[0]) == [46]
    assert sig["short_entries"].iloc[46]["BBB"]
    assert not sig["entries"]["BBB"].any() and not sig["short_entries"]["AAA"].any()
    # No reversion exit — only the horizon time stop, entry + 10 = 56.
    assert not sig["exits"].iloc[47]["AAA"] and sig["exits"].iloc[56]["AAA"]
    assert sig["short_exits"].iloc[56]["BBB"]


def test_direction_gates_sides():
    panel = _event_panel()
    long_only = _sig(panel, direction="long")
    assert long_only["entries"]["AAA"].iloc[46] and not long_only["short_entries"].any().any()
    short_only = _sig(panel, direction="short")
    assert short_only["short_entries"]["BBB"].iloc[46] and not short_only["entries"].any().any()


def test_pattern_any_is_location_only():
    # pattern=any ignores candle shape; the same dip fires purely on the location gate.
    sig = _sig(_event_panel(), pattern="any", direction="long")
    assert list(np.where(sig["entries"]["AAA"])[0]) == [46]


def test_no_entries_without_a_rejection_candle(panel):
    # Steady trends print no hammers and never stretch on the allowed side.
    sig = _sig(panel, direction="both")
    assert not sig["entries"].any().any() and not sig["short_entries"].any().any()


def _engulfing_panel():
    """make_panel with one bullish engulfing on AAA (red bar 44, green bar 45)."""
    from tests.conftest import make_panel

    p = make_panel()
    close = p.close.copy()
    open_, high, low = (p.close * 0.999).copy(), (p.close * 1.01).copy(), (p.close * 0.99).copy()
    i44, i45 = close.index[44], close.index[45]
    open_.loc[i44, "AAA"], close.loc[i44, "AAA"] = 123.0, 121.0     # red bar
    high.loc[i44, "AAA"], low.loc[i44, "AAA"] = 123.5, 120.5
    open_.loc[i45, "AAA"], close.loc[i45, "AAA"] = 120.5, 123.5     # green bar engulfs it
    high.loc[i45, "AAA"], low.loc[i45, "AAA"] = 124.0, 120.0
    return type(p)(close=close, open_=open_, high=high, low=low, volume=p.volume, bar=p.bar)


def test_engulfing_pattern_fires_with_gates_off():
    strat = strategies.load("candle_scalp")
    sig = strat.signals(_engulfing_panel(), **{**strat.params,
                        "pattern": "engulfing", "stretch_mult": 0.0, "trend_window": 0,
                        "vol_mult": 0.0, "sl_mult": 1.0, "horizon": 10, "direction": "long"})
    assert list(np.where(sig["entries"]["AAA"])[0]) == [46]


def test_valid_rules():
    strat = strategies.load("candle_scalp")
    assert strat.valid({"band_window": 10, "trend_window": 40, "sl_mult": 2.0, "pattern": "hammer"})
    assert not strat.valid({"band_window": 40, "trend_window": 40, "sl_mult": 2.0})  # band !< trend
    assert not strat.valid({"sl_mult": 0.0})                                         # needs a stop
    assert not strat.valid({"pattern": "any", "stretch_mult": 0.0, "sl_mult": 2.0})  # naked "any"
    assert strat.valid({"pattern": "any", "stretch_mult": 2.0, "sl_mult": 2.0})      # any+location
    assert strat.valid({"trend_window": 0, "band_window": 100, "sl_mult": 1.0})      # trend off


def test_stop_keys_conditional():
    panel = _event_panel()
    sig = _sig(panel, sl_mult=2.0, tp_mult=0.0)
    assert isinstance(sig["sl_stop"], pd.DataFrame)
    assert (sig["sl_stop"].iloc[15:] > 0).all().all()      # ATR fraction of price
    assert "tp_stop" not in sig                            # 0 disables the target
    assert "sl_trail" not in sig                           # a bracket never trails
    assert "tp_stop" in _sig(panel, sl_mult=2.0, tp_mult=3.0)


def test_intraday_session_guards_and_eod():
    from trading_simple.data import Panel

    # Two 91-minute sessions, AAA trending up, with two engineered hammer dips on
    # day 2: mid-session (minute 45, tradable) and near the open (minute 5, inside
    # the skip_open blackout).
    tz = "America/New_York"
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-06-0{d} 09:30", tz=tz) + pd.Timedelta(minutes=m)
                            for d in (1, 2) for m in range(91)]).tz_convert("UTC")
    n, day2 = len(idx), 91
    close = pd.DataFrame({"AAA": np.linspace(100.0, 130.0, n),
                          "BBB": np.linspace(100.0, 92.0, n)}, index=idx)
    open_, high, low = (close * 0.9995).copy(), (close * 1.001).copy(), (close * 0.999).copy()
    for k in (day2 + 45, day2 + 5):
        b = idx[k]
        close.loc[b, "AAA"] -= 3.0
        c = close.loc[b, "AAA"]
        open_.loc[b, "AAA"], high.loc[b, "AAA"], low.loc[b, "AAA"] = c - 0.02, c, c - 0.5
    panel = Panel(close=close, open_=open_, high=high, low=low,
                  volume=pd.DataFrame(1_000.0, index=idx, columns=close.columns), bar="1m")

    sig = strategies.load("candle_scalp").signals(
        panel, pattern="hammer", wick_mult=2.0, band_window=20, trend_window=60,
        atr_window=10, stretch_mult=0.5, vol_mult=0.0, profile_days=10, sl_mult=1.0,
        tp_mult=0.0, horizon=30, direction="long", skip_open=15, skip_close=15, eod_exit=True)
    assert list(np.where(sig["entries"]["AAA"])[0]) == [day2 + 46]  # near-open dip blocked
    assert sig["exits"].iloc[90].all()                    # eod_exit: flat on day-1 close
    assert sig["exits"].iloc[-1].all()                    # ... and on day-2 close
