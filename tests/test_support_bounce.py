"""support_bounce signal logic on synthetic data with engineered support tests."""

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

#: Small minute-windows so the two-session synthetic panel warms up quickly
#: (on a 1m panel minutes == bars). Coherent under valid(): 6.0 > 5.0 + 0.5.
_PARAMS = {"level_window": 30, "min_touches": 2, "touch_tol": 0.5,
           "entry_dist": 5.0, "entry_mode": "bounce", "tf_minutes": 1,
           "ema_fast": 5, "ema_slow": 15, "regime_dz": 0.0,
           "allow_sideways": True, "atr_window": 5, "sl_atr": 6.0,
           "sl_trail": False, "tp_mult": 0.0, "max_hold": 10,
           "exit_on_bear": True, "eod_exit": False,
           "skip_open": 15, "skip_close": 15}

_TZ = "America/New_York"
_DAY2 = 91  # bar offset of day 2's first minute


def _sig(panel, **overrides):
    strat = strategies.load("support_bounce")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _index():
    """Two 91-minute sessions (2026-06-01/02, 09:30 New York), UTC index."""
    return pd.DatetimeIndex(
        [pd.Timestamp(f"2026-06-0{d} 09:30", tz=_TZ) + pd.Timedelta(minutes=m)
         for d in (1, 2) for m in range(91)]).tz_convert("UTC")


def _drift_frames(idx):
    """AAA drifts up, BBB drifts down; 0.4-wide bars so ATR is steady (~0.4).

    The up-drift keeps AAA's rolling-min support ~1.1 below the tape, so no
    baseline bar ever touches its band; BBB's down-drift makes every bar a
    touch, leaving the bear gate as the only thing between BBB and an entry.
    """
    n = len(idx)
    close = pd.DataFrame({"AAA": 100.0 + 0.03 * np.arange(n),
                          "BBB": 106.0 - 0.03 * np.arange(n)}, index=idx)
    return close, close.copy(), close + 0.2, close - 0.2


def _panel(close, open_, high, low, bar="1m"):
    volume = pd.DataFrame(1_000.0, index=close.index, columns=close.columns)
    return Panel(close=close, open_=open_, high=high, low=low, volume=volume, bar=bar)


def _support_panel(entry_close=103.7):
    """Drift tape with an engineered, twice-tested support at 99.0 on day 2.

    Day-2 events on AAA (m = minute of day 2):
    * m18 — deep wick to 99.0 pins the level and is touch episode #1.
    * m28 — wick to 99.1 re-tests the level (episode #2), support unchanged.
    * m40 — the entry event: low dips to 99.2 (inside the touch band) while
      the close (``entry_close``) reclaims it. With ``min_touches=2`` this is
      the only bar whose level counts as tested.
    """
    idx = _index()
    close, open_, high, low = _drift_frames(idx)
    low.loc[idx[_DAY2 + 18], "AAA"] = 99.0
    low.loc[idx[_DAY2 + 28], "AAA"] = 99.1
    low.loc[idx[_DAY2 + 40], "AAA"] = 99.2
    close.loc[idx[_DAY2 + 40], "AAA"] = entry_close
    return _panel(close, open_, high, low)


def test_bounce_enters_next_bar_after_tested_level():
    sig = _sig(_support_panel())
    # Signal on the m40 reclaim bar -> execution at m41, and nowhere else:
    # the m18/m28 tests themselves are vetoed by min_touches (0 and 1 prior).
    assert list(np.where(sig["entries"]["AAA"])[0]) == [_DAY2 + 41]
    assert not sig["entries"].iloc[_DAY2 + 40].any()       # never the signal bar
    assert sig["exits"]["AAA"].iloc[_DAY2 + 51]            # max_hold=10 time stop
    assert "short_entries" not in sig and "short_exits" not in sig  # long-only


def test_touch_mode_enters_where_bounce_does_not():
    # Entry-bar close inside the touch band: a touch, but no reclaim. A wide
    # dead zone keeps the regime sideways so the trigger is isolated.
    panel = _support_panel(entry_close=99.5)
    touch = _sig(panel, entry_mode="touch", regime_dz=0.05)
    assert list(np.where(touch["entries"]["AAA"])[0]) == [_DAY2 + 41]
    bounce = _sig(panel, entry_mode="bounce", regime_dz=0.05)
    assert not bounce["entries"]["AAA"].any()


def test_min_touches_vetoes_undertested_level():
    panel = _support_panel()
    assert not _sig(panel, min_touches=3)["entries"].any().any()  # only 2 prior tests
    assert _sig(panel, min_touches=2)["entries"]["AAA"].iloc[_DAY2 + 41]


def test_bear_regime_vetoes_touches():
    panel = _support_panel()
    sig = _sig(panel, min_touches=0, entry_mode="touch")
    assert sig["entries"]["AAA"].any()               # up-drift control fires
    assert not sig["entries"]["BBB"].any()           # bear vetoes permanent touches
    assert sig["exits"]["BBB"].iloc[30:].all()       # exit_on_bear floods the bear tape
    calm = _sig(panel, min_touches=0, entry_mode="touch", exit_on_bear=False)
    assert not calm["exits"]["BBB"].any()


def test_sideways_gate_and_ema_warmup_guard():
    # Flat tape: fast EMA == slow EMA exactly -> sideways; support == the low,
    # so every warmed-up bar is a touch. Entries must start only once the slow
    # EMA has warmed up (bar 14 -> first execution bar 15), not when NaN EMAs
    # masquerade as "sideways".
    idx = _index()
    close = pd.DataFrame({"AAA": np.full(len(idx), 101.0)}, index=idx)
    panel = _panel(close, close.copy(), close + 0.2, close - 0.2)
    kw = dict(min_touches=0, entry_mode="touch", level_window=10,
              skip_open=0, skip_close=0)
    hits = np.where(_sig(panel, **kw)["entries"]["AAA"])[0]
    assert len(hits) and hits[0] == 15
    assert not _sig(panel, allow_sideways=False, **kw)["entries"].any().any()


def test_stop_keys_and_trail_flag():
    panel = _support_panel()
    sig = _sig(panel)
    assert isinstance(sig["sl_stop"], pd.DataFrame)
    assert (sig["sl_stop"].iloc[10:] > 0).all().all()      # ATR fraction of price
    assert "sl_trail" not in sig and "tp_stop" not in sig  # both off by default
    trail = _sig(panel, sl_trail=True, tp_mult=3.0)
    assert trail["sl_trail"] is True and "tp_stop" in trail


def test_session_guards_and_eod():
    # Two touch events on day 2: near the open (m5 -> execution m6, inside the
    # skip_open blackout) and mid-session (m40 -> m41, tradable).
    idx = _index()
    close, open_, high, low = _drift_frames(idx)
    low.loc[idx[_DAY2 + 5], "AAA"] = 99.0
    low.loc[idx[_DAY2 + 40], "AAA"] = 99.2
    panel = _panel(close, open_, high, low)
    sig = _sig(panel, min_touches=0, entry_mode="touch")
    assert list(np.where(sig["entries"]["AAA"])[0]) == [_DAY2 + 41]
    assert not sig["exits"]["AAA"].iloc[90]                # eod off: holds overnight
    eod = _sig(panel, min_touches=0, entry_mode="touch", eod_exit=True)
    assert eod["exits"].iloc[90].all()                     # flat on day-1 close
    assert eod["exits"].iloc[-1].all()                     # ... and on day-2 close


def test_tf_window_widens_the_trigger():
    # A dip bar with no same-bar reclaim, baseline again one bar later: the
    # 1-minute trigger never sees dip+reclaim together, a 5-minute one does.
    idx = _index()
    close, open_, high, low = _drift_frames(idx)
    low.loc[idx[_DAY2 + 40], "AAA"] = 99.2
    close.loc[idx[_DAY2 + 40], "AAA"] = 99.5
    panel = _panel(close, open_, high, low)
    kw = dict(min_touches=0, regime_dz=0.05)   # sideways everywhere: trigger only
    assert not _sig(panel, tf_minutes=1, **kw)["entries"]["AAA"].any()
    assert _sig(panel, tf_minutes=5, **kw)["entries"]["AAA"].any()


def test_valid_rules():
    strat = strategies.load("support_bounce")
    assert strat.valid(dict(strat.params))                        # defaults coherent
    assert strat.valid({"entry_mode": "touch", "tf_minutes": 5, "sl_trail": True})
    assert not strat.valid({"ema_fast": 960, "ema_slow": 960})    # regime roles invert
    assert not strat.valid({"touch_tol": 1.0, "entry_dist": 0.5})  # band outside zone
    assert not strat.valid({"sl_atr": 1.0, "entry_dist": 1.0, "touch_tol": 0.25})
    assert not strat.valid({"tf_minutes": 30})                    # straddles the open


def test_minute_windows_convert_across_bars():
    from tests.conftest import make_panel

    strat = strategies.load("support_bounce")
    p = make_panel()
    sig = strat.signals(p, **strat.params)     # minute windows collapse to >=1 bar
    assert sig["entries"].dtypes.eq(bool).all()
    assert not sig["entries"].iloc[0].any()
    bad = Panel(close=p.close, open_=p.open_, high=p.high, low=p.low,
                volume=p.volume, bar="15m")
    with pytest.raises(KeyError):              # unsupported bar fails loudly
        strat.signals(bad, **strat.params)
