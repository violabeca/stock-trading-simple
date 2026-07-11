"""keltner_reversion signal logic on synthetic data with one engineered dip."""

import numpy as np
import pandas as pd

from trading_simple import strategies

#: Small windows so the 60-bar synthetic panel warms up quickly.
_PARAMS = {"trend_window": 40, "band_window": 10, "atr_window": 5,
           "band_mult": 1.0, "horizon": 10}


def _kr(panel, **overrides):
    strat = strategies.load("keltner_reversion")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _event_panel():
    """make_panel with a one-bar flush on AAA (uptrend) and pop on BBB (downtrend).

    At bar 45 AAA closes 8.0 below its trend line — under the lower band but
    still above SMA(40) — and BBB mirrors it upward, so exactly one long and
    one short setup exist.
    """
    from tests.conftest import make_panel

    p = make_panel()
    close = p.close.copy()
    close.loc[close.index[45], "AAA"] -= 8.0
    close.loc[close.index[45], "BBB"] += 8.0
    return type(p)(close=close, open_=close * 0.999, high=close * 1.01,
                   low=close * 0.99, volume=p.volume, bar=p.bar)


def test_dip_enters_next_bar_and_reverts():
    sig = _kr(_event_panel(), direction="both")
    # Dip at bar 45 -> execution at bar 46, and the steady uptrend never
    # re-triggers: exactly one long (AAA) and one short (BBB) entry.
    assert sig["entries"].iloc[46]["AAA"] and not sig["entries"].iloc[45].any()
    assert list(np.where(sig["entries"]["AAA"])[0]) == [46]
    assert sig["short_entries"].iloc[46]["BBB"]
    assert not sig["entries"]["BBB"].any() and not sig["short_entries"]["AAA"].any()
    # Bar 46 closes back at the (dip-depressed) midline -> reversion exit
    # executes at bar 47; the time stop lands horizon bars after entry (56).
    assert not sig["exits"].iloc[46]["AAA"]
    assert sig["exits"].iloc[47]["AAA"] and sig["exits"].iloc[56]["AAA"]
    assert sig["short_exits"].iloc[47]["BBB"]


def test_direction_gates_sides():
    panel = _event_panel()
    long_only = _kr(panel, direction="long")
    assert long_only["entries"]["AAA"].iloc[46] and not long_only["short_entries"].any().any()
    short_only = _kr(panel, direction="short")
    assert short_only["short_entries"]["BBB"].iloc[46] and not short_only["entries"].any().any()


def test_no_entries_without_stretch(panel):
    # Steady trends never leave the band on the allowed side: BBB drifts below
    # its midline but the trend filter blocks buying a downtrend, and AAA never
    # stretches above its band while above trend.
    sig = _kr(panel, direction="both")
    assert not sig["entries"].any().any() and not sig["short_entries"].any().any()


def test_valid_rejects_band_slower_than_trend():
    strat = strategies.load("keltner_reversion")
    assert strat.valid({"band_window": 10, "trend_window": 40})
    assert not strat.valid({"band_window": 40, "trend_window": 40})
    assert not strat.valid({"band_window": 60, "trend_window": 40})


def test_stop_param_conditional():
    panel = _event_panel()
    sig = _kr(panel, sl_mult=2.0)
    assert isinstance(sig["sl_stop"], pd.DataFrame)
    assert (sig["sl_stop"].iloc[5:] > 0).all().all()      # ATR fraction of price
    assert "sl_trail" not in sig                          # reversion bets never trail
    assert "sl_stop" not in _kr(panel, sl_mult=0.0)       # 0 disables the stop entirely


def test_intraday_session_guards_and_eod():
    from trading_simple.data import Panel

    # Two 91-minute sessions, AAA trending up, with two engineered one-bar
    # dips on day 2: mid-session (minute 45, tradable) and near the open
    # (minute 5, inside the skip_open blackout).
    tz = "America/New_York"
    idx = pd.DatetimeIndex([pd.Timestamp(f"2026-06-0{d} 09:30", tz=tz) + pd.Timedelta(minutes=m)
                            for d in (1, 2) for m in range(91)]).tz_convert("UTC")
    n, day2 = len(idx), 91
    close = pd.DataFrame({"AAA": np.linspace(100.0, 110.0, n),
                          "BBB": np.linspace(100.0, 92.0, n)}, index=idx)
    close.loc[close.index[day2 + 45], "AAA"] -= 1.2
    close.loc[close.index[day2 + 5], "AAA"] -= 1.2
    panel = Panel(close=close, open_=close * 0.999, high=close * 1.001,
                  low=close * 0.999,
                  volume=pd.DataFrame(1_000.0, index=idx, columns=close.columns), bar="1m")

    sig = _kr(panel, trend_window=60, horizon=30, direction="long",
              skip_open=15, skip_close=15, eod_exit=True)
    assert list(np.where(sig["entries"]["AAA"])[0]) == [day2 + 46]  # near-open dip blocked
    assert sig["exits"].iloc[90].all()                    # eod_exit: flat on day-1 close
    assert sig["exits"].iloc[-1].all()                    # ... and on day-2 close
