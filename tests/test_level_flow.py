"""level_flow: levels from prior sessions' swings, activity-ranked watchlist, level-based risk.

Every test stamps ONE engineered event into a flat multi-session 1-minute panel and asserts
the exact bar the engine sees it on. Prices sit at 100 with ±0.3 bars, so the flat sessions
form a support level at 99.7 and a resistance level at 100.3 (every 10-minute candle is a
tied swing there) and the session ATR is 0.6; an engineered dip or spike in session 1 adds
the level the events in session 3 (the first session after the 3-session warm-up) trade.
"""

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel, resample_panel

_TZ = "America/New_York"
_N = 80                     # bars per session
_PARAMS = {
    "tf_levels": 10, "pivot_k": 1, "level_days": 3, "bin_pct": 0.002, "min_touches": 1,
    "signal_tf": 0, "trend_tf": 0, "trend_len": 3, "trend_gate": "with",
    "touch_tol": 0.6, "touch_minutes": 1, "confirm_minutes": 0, "break_tol": 0.25,
    "reach_atr": 10.0, "scan_minute": 10, "skip_close": 5,
    "vol_mult": 0.0, "profile_days": 2, "rank_by": "rvol", "max_candidates": 8,
    "min_price": 0.0, "min_dollar_volume": 0.0,
    "polarity": "bounce", "direction": "both", "vwap_gate": "none", "range_used_max": 0.0,
    "regime": "none", "regime_days": 1, "atr_days": 2, "warmup_sessions": 3,
    "sl_atr": 1.0, "max_stop": 1.0, "tp_ratio": 0.0, "sl_trail": False, "horizon": 8,
    "risk_per_trade": 0.004, "ssr_guard": True, "eod_exit": True,
}


def _D(k: int) -> int:
    """Bar offset of session ``k``'s first minute."""
    return k * _N


def _index(days: int = 6) -> pd.DatetimeIndex:
    opens = [pd.Timestamp(f"2026-06-{d:02d} 09:30", tz=_TZ) for d in range(1, days + 1)]
    return pd.DatetimeIndex([o + pd.Timedelta(minutes=m) for o in opens
                             for m in range(_N)]).tz_convert("UTC")


def _flat(symbols=("AAA", "BBB"), days: int = 6) -> tuple[pd.DataFrame, ...]:
    """Flat 100.0 closes with ±0.3 bars: one level below, one above, ATR 0.6."""
    close = pd.DataFrame(100.0, index=_index(days), columns=list(symbols))
    return close, close + 0.3, close - 0.3


def _panel(close, high, low, volume=None, bar: str = "1m") -> Panel:
    if volume is None:
        volume = pd.DataFrame(1_000.0, index=close.index, columns=close.columns)
    return Panel(close=close, open_=close.copy(), high=high, low=low, volume=volume, bar=bar,
                 tz=_TZ)


def _stamp(frames, row: int, symbols, close: float, high: float, low: float) -> None:
    c, h, lo = frames
    for s in symbols:
        c.loc[c.index[row], s], h.loc[h.index[row], s], lo.loc[lo.index[row], s] = close, high, low


def _dip_panel(symbols=("AAA", "BBB"), event: bool = True, volume=None) -> Panel:
    """Session 1 dips to a low of 98.7 (a new level); session 3 bar 42 tests it and reclaims."""
    frames = _flat(symbols)
    _stamp(frames, _D(1) + 30, symbols, 99.0, 99.3, 98.7)
    if event:
        _stamp(frames, _D(3) + 40, symbols, 99.5, 99.55, 99.4)
        _stamp(frames, _D(3) + 41, symbols, 99.6, 99.65, 99.5)
        _stamp(frames, _D(3) + 42, symbols, 99.65, 99.7, 98.9)       # the touch + reclaim
        _stamp(frames, _D(3) + 43, symbols, 99.65, 99.7, 99.5)
        _stamp(frames, _D(3) + 44, symbols, 99.65, 99.7, 99.5)
    return _panel(*frames, volume=volume)


def _spike_panel(symbols=("AAA", "BBB")) -> Panel:
    """Session 1 spikes to a high of 101.3; session 3 bar 42 tests it and rejects."""
    frames = _flat(symbols)
    _stamp(frames, _D(1) + 30, symbols, 101.0, 101.3, 100.7)
    _stamp(frames, _D(3) + 40, symbols, 100.5, 100.6, 100.45)
    _stamp(frames, _D(3) + 41, symbols, 100.4, 100.5, 100.35)
    _stamp(frames, _D(3) + 42, symbols, 100.35, 101.1, 100.3)      # the touch + rejection
    return _panel(*frames)


def _sig(panel, **overrides) -> dict:
    strat = strategies.load("level_flow")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _compose(panel, **overrides) -> dict:
    strat = strategies.load("level_flow")
    return strat.signals.__globals__["compose"](panel, **{**strat.params, **_PARAMS, **overrides})


def _where(frame, col="AAA") -> list[int]:
    return list(np.where(frame[col].to_numpy())[0])


# --- the shipped config is coherent ------------------------------------------------------
def test_shipped_config_loads_and_is_valid():
    strat = strategies.load("level_flow")
    assert strat.valid(strat.params)
    assert strat.valid({**strat.params, **_PARAMS})
    assert strat.optuna["objectives"] == ["alpha_cagr", "volatility"]
    assert strat.optuna["score_from"] and strat.optuna["pick_min_trades_per_year"]


# --- levels and the bounce polarity ------------------------------------------------------
def test_bounce_long_enters_the_bar_after_the_reclaim_and_only_once():
    sig = _sig(_dip_panel())
    assert _where(sig["entries"]) == [_D(3) + 43]                # next bar, not the signal bar
    assert not sig["short_entries"].to_numpy().any()
    c = _compose(_dip_panel())
    assert c["support"]["AAA"].iloc[_D(3) + 42] == pytest.approx(98.7)   # the session-1 dip
    assert c["sup_touches"]["AAA"].iloc[_D(3) + 42] == 1
    assert c["support"]["AAA"].iloc[_D(3) + 10] == pytest.approx(99.7)   # flat lows, below 100
    assert c["resistance"]["AAA"].iloc[_D(3) + 10] == pytest.approx(100.3)


def test_no_trade_before_the_warmup_or_without_a_level():
    frames = _flat()
    _stamp(frames, _D(0) + 30, ("AAA", "BBB"), 99.0, 99.3, 98.7)       # level formed in session 0
    _stamp(frames, _D(2) + 42, ("AAA", "BBB"), 99.65, 99.7, 98.9)      # tested in session 2
    panel = _panel(*frames)
    assert not _sig(panel)["entries"].to_numpy().any()               # session 2 < warmup 3
    assert _where(_sig(panel, warmup_sessions=2)["entries"]) == [_D(2) + 43]
    assert not _sig(_dip_panel(), level_days=1)["entries"].to_numpy().any()   # dip fell out


def test_min_touches_needs_a_defended_level():
    assert not _sig(_dip_panel(), min_touches=2)["entries"].to_numpy().any()
    frames = _flat()
    _stamp(frames, _D(1) + 30, ("AAA", "BBB"), 99.0, 99.3, 98.7)
    _stamp(frames, _D(2) + 30, ("AAA", "BBB"), 99.0, 99.3, 98.7)       # a second test of 98.7
    _stamp(frames, _D(3) + 42, ("AAA", "BBB"), 99.65, 99.7, 98.9)
    sig = _sig(_panel(*frames), min_touches=2)
    assert _where(sig["entries"]) == [_D(3) + 43]


def test_bounce_short_enters_after_the_rejection_at_resistance():
    sig = _sig(_spike_panel())
    assert _where(sig["short_entries"]) == [_D(3) + 43]
    assert not sig["entries"].to_numpy().any()
    assert not _sig(_spike_panel(), direction="long")["short_entries"].to_numpy().any()


# --- the break polarity --------------------------------------------------------------------
def test_break_short_enters_on_the_first_close_through_support():
    frames = _flat()
    for b in range(45, 50):
        _stamp(frames, _D(3) + b, ("AAA", "BBB"), 99.2, 99.3, 99.1)   # through 99.7 by > 0.25 ATR
    sig = _sig(_panel(*frames), polarity="break")
    assert _where(sig["short_entries"]) == [_D(3) + 46]
    assert not sig["entries"].to_numpy().any()


def test_break_before_the_window_is_a_missed_trade_not_a_stale_one():
    frames = _flat()
    for b in range(5, 50):                                           # crossed at bar 5 (< scan 10)
        _stamp(frames, _D(3) + b, ("AAA", "BBB"), 99.2, 99.3, 99.1)
    sig = _sig(_panel(*frames), polarity="break")
    assert not sig["short_entries"].to_numpy().any()


# --- the short-sale restriction ----------------------------------------------------------
def test_ssr_blocks_shorts_for_the_rest_of_the_session_and_the_next():
    frames = _flat()
    _stamp(frames, _D(1) + 30, ("AAA", "BBB"), 101.0, 101.3, 100.7)
    _stamp(frames, _D(3) + 20, ("AAA", "BBB"), 90.0, 90.5, 89.5)       # -10% from the prior close
    for b, (c, h, lo) in zip(range(40, 43), [(100.5, 100.6, 100.45), (100.4, 100.5, 100.35),
                                             (100.35, 101.1, 100.3)], strict=True):
        _stamp(frames, _D(3) + b, ("AAA", "BBB"), c, h, lo)
    _stamp(frames, _D(4) + 40, ("AAA", "BBB"), 95.0, 100.3, 94.9)      # next session: the crash
                                                                      # widened its ATR to 6.6
    panel = _panel(*frames)
    off = _sig(panel, ssr_guard=False)["short_entries"]
    assert _where(off) == [_D(3) + 43, _D(4) + 41]                    # both would trade
    assert not _sig(panel)["short_entries"].to_numpy().any()          # neither may


# --- the activity-ranked watchlist ------------------------------------------------------
def test_watchlist_keeps_the_top_names_by_relative_volume_frozen_at_the_scan_bar():
    syms = tuple(f"S{i}" for i in range(5))
    volume = pd.DataFrame(1_000.0, index=_index(), columns=list(syms))
    volume.iloc[_D(3):_D(3) + 11, 4] = 10_000.0                      # S4 is in play before the scan
    volume.iloc[_D(3) + 11:_D(3) + 40, 1] = 10_000.0                 # S1 only AFTER the scan bar
    panel = _dip_panel(syms, volume=volume)
    sig = _sig(panel, max_candidates=2)
    fired = [s for s in syms if sig["entries"][s].to_numpy().any()]
    assert fired == ["S0", "S4"]                                     # S4 by rvol, S0 by first-tie
    one = _sig(panel, max_candidates=1)["entries"]
    assert [s for s in syms if one[s].to_numpy().any()] == ["S4"]
    assert len([s for s in syms if _sig(panel)["entries"][s].to_numpy().any()]) == 5


# --- stops, sizing, exits -------------------------------------------------------------------
def test_stop_is_level_based_and_sits_on_the_fill_bar():
    panel = _dip_panel()
    sig, c = _sig(panel, tp_ratio=2.0), _compose(panel)
    signal_bar, fill_bar = _D(3) + 42, _D(3) + 43
    atr = c["atr"]["AAA"].iloc[signal_bar]
    assert 0.6 <= atr < 1.0                                          # a session-scale ATR
    want = (abs(99.65 - 98.7) + 1.0 * atr) / 99.65
    assert np.isnan(sig["sl_stop"]["AAA"].iloc[signal_bar])           # nothing at the signal bar
    assert sig["sl_stop"]["AAA"].iloc[fill_bar] == pytest.approx(want, rel=1e-4)   # float32 level
    assert sig["tp_stop"]["AAA"].iloc[fill_bar] == pytest.approx(2.0 * want, rel=1e-4)
    assert "tp_stop" not in _sig(panel)                              # tp_ratio 0 = no target
    assert not _sig(panel, max_stop=0.005)["entries"].to_numpy().any()   # too wide: skipped


def test_cash_frac_frame_sizes_the_entry_off_its_stop_and_is_dense():
    panel = _dip_panel()
    sig = _sig(panel, max_candidates=8, risk_per_trade=0.001)      # 0.1% risk on a ~1.8% stop
    fill_bar = _D(3) + 43
    frac = sig["cash_frac"]
    assert not frac.isna().to_numpy().any()
    stop = sig["sl_stop"]["AAA"].iloc[fill_bar]
    assert frac["AAA"].iloc[fill_bar] == pytest.approx(min(1 / 8, 0.001 / stop))
    assert frac["AAA"].iloc[fill_bar] < 1 / 8                # the stop, not the slot, bound it
    assert frac["AAA"].iloc[fill_bar - 1] == 1 / 8 and frac["BBB"].iloc[0] == 1 / 8
    big = _sig(panel, max_candidates=2, risk_per_trade=0.5)["cash_frac"]
    assert big["AAA"].iloc[fill_bar] == 0.5


def test_eod_flat_and_horizon_exit():
    sig = _sig(_dip_panel())
    for k in range(6):
        last = _D(k) + _N - 1
        assert sig["exits"].iloc[last].all() and sig["short_exits"].iloc[last].all()
    assert sig["exits"]["AAA"].iloc[_D(3) + 43 + 8]                  # horizon 8 minutes = 8 bars
    assert not sig["exits"]["AAA"].iloc[_D(3) + 43 + 7]


# --- multiple timeframes ---------------------------------------------------------------------
def test_signal_candle_reads_the_trigger_on_its_own_closes():
    panel = _dip_panel()
    assert _where(_sig(panel, signal_tf=5)["entries"]) == [_D(3) + 45]   # candle 40-44 closes at 44
    assert _where(_sig(panel)["entries"]) == [_D(3) + 43]


def test_trend_gate_uses_completed_candles_only():
    frames = _flat()
    _stamp(frames, _D(1) + 30, ("AAA", "BBB"), 99.0, 99.3, 98.7)
    for b in range(40):                                              # session 3 drifts DOWN...
        px = 100.2 - 0.005 * b
        _stamp(frames, _D(3) + b, ("AAA", "BBB"), px, px + 0.3, px - 0.3)
    _stamp(frames, _D(3) + 40, ("AAA", "BBB"), 99.5, 99.55, 99.4)   # ...then turns up inside the
    _stamp(frames, _D(3) + 41, ("AAA", "BBB"), 99.6, 99.65, 99.5)   # candle that carries the event
    _stamp(frames, _D(3) + 42, ("AAA", "BBB"), 99.65, 99.7, 98.9)
    panel = _panel(*frames)
    assert _where(_sig(panel)["entries"]) == [_D(3) + 43]            # trend_tf 0: gate off
    trend = dict(trend_tf=10, trend_len=3)
    assert not _sig(panel, trend_gate="with", **trend)["entries"].to_numpy().any()
    assert _where(_sig(panel, trend_gate="against", **trend)["entries"]) == [_D(3) + 43]
    c = _compose(panel, trend_gate="with", **trend)
    assert c["trend"]["AAA"].iloc[_D(3) + 42] < 0                    # the finished candles say down


def test_runs_on_derived_bars_and_refuses_candles_the_bar_cannot_build():
    panel = _dip_panel()
    five = resample_panel(panel, 5)
    sig = _sig(five, scan_minute=10, touch_minutes=5, horizon=40)
    assert set(sig) >= {"entries", "short_entries", "sl_stop", "cash_frac"}
    assert sig["entries"].shape == five.close.shape
    assert not sig["cash_frac"].isna().to_numpy().any()
    with pytest.raises(ValueError, match="candles"):
        _sig(five, signal_tf=2)


# --- the config invariants ------------------------------------------------------------------
def test_valid_rejects_incoherent_combos():
    valid = strategies.load("level_flow").valid
    assert valid({})
    for bad in ({"warmup_sessions": 5, "level_days": 10},
                {"warmup_sessions": 15, "regime_days": 10},
                {"polarity": "fade"}, {"trend_gate": "sideways"}, {"rank_by": "alpha"},
                {"max_stop": 0.0}, {"sl_atr": 0.0}, {"pivot_k": 0}):
        assert not valid(bad), bad
    # Geometry that used to be rejected is representable: the stop is beyond the level on the
    # far side of the band, lookbacks may reach before the scan, trend_tf 0 is the off-switch.
    for ok in ({"sl_atr": 0.1, "touch_tol": 0.5}, {"touch_minutes": 30, "scan_minute": 15},
               {"trend_gate": "with", "trend_tf": 0}, {"signal_tf": 15, "scan_minute": 15}):
        assert valid(ok), ok


# --- the explain hook the visual notebook draws from ------------------------------------------
def test_explain_levels_shows_the_pivot_behind_a_level():
    explain = strategies.load("level_flow").signals.__globals__["explain_levels"]
    panel = _dip_panel()
    day = panel.close.index[_D(3)].tz_convert(_TZ).date()
    x = explain(panel, "AAA", day, **{**strategies.load("level_flow").params, **_PARAMS})
    assert x["session"] == day
    levels = x["levels"]
    dip = levels[(levels["level"] - 98.7).abs() < 0.05]
    assert len(dip) == 1 and dip["touches"].iloc[0] == 1              # one tested pivot at 98.7
    assert (levels["touches"] >= 1).all() and levels["level"].is_monotonic_increasing
    piv = x["pivots"]
    low = piv[(piv["side"] == "low") & ((piv["price"] - 98.7).abs() < 1e-6)]
    assert len(low) == 1 and low["session"].iloc[0] == 1 and bool(low["counted"].iloc[0])
    assert not piv[piv["session"] == 3]["counted"].any()              # today's swings: not yet
    assert set(x["candles"]["session"]) == {0, 1, 2, 3}               # level_days 3 + the session
    assert x["support"].iloc[_D(3) + 42] == pytest.approx(98.7)       # same read as signals()
    with pytest.raises(KeyError, match="not a session"):
        explain(panel, "AAA", "2030-01-01", **_PARAMS)
