"""trend_following signal logic on synthetic daily prices (no DB, no network)."""

from dataclasses import replace

import numpy as np
import pandas as pd

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

N = 140
T0 = 60        # UP and DOWN leave their ranges on this bar
CRASH = 110    # UP falls 40% on this bar — far through its stop
# Small windows so the synthetic history is enough; adx_min 0 so timing is exact.
FAST = dict(entry_window=10, adx_window=5, adx_min=0.0, atr_window=5, atr_mult=3.0,
            direction="both", max_positions=4, rs_window=30, rs_skip=5)


def _strat():
    return strategies.load("trend_following")


def _compose(panel, **overrides):
    return _strat().signals.__globals__["compose"](panel, **{**FAST, **overrides})


def _weights(panel, **overrides):
    return _strat().signals(panel, **{**FAST, **overrides})["weights"]


def _make(**series) -> Panel:
    """OHLC around the given closes: a ±0.5% daily range, opening at the prior close."""
    index = pd.date_range("2022-01-03", periods=N, freq="B", tz="UTC")
    close = pd.DataFrame(series, index=index)
    return Panel(close=close, open_=close.shift(1).fillna(close), high=close * 1.005,
                 low=close * 0.995, volume=pd.DataFrame(1e6, index=index, columns=close.columns),
                 bar="1d")


def _range():
    """100 ± 0.3, alternating: a range with no direction (ADX low, +DI ≈ −DI)."""
    return 100.0 + np.where(np.arange(N) % 2 == 0, 0.3, -0.3)


def _base():
    """UP breaks out and trends +2%/day, then crashes; DOWN breaks down −2%/day; FLAT ranges."""
    t = np.arange(N)
    up, down = _range(), _range()
    up[T0:] = 102.0 * 1.02 ** (t[T0:] - T0)
    up[CRASH:] = up[CRASH - 1] * 0.6
    down[T0:] = 98.0 * 0.98 ** (t[T0:] - T0)
    return _make(UP=up, DOWN=down, FLAT=_range())


def test_starts_enter_on_the_next_bar_long_and_short():
    p = _base()
    parts, w = _compose(p), _weights(p)
    assert parts["long_start"]["UP"].iloc[T0] and parts["short_start"]["DOWN"].iloc[T0]
    assert not parts["long_start"]["UP"].iloc[:T0].any()          # nothing inside the range
    assert w["UP"].iloc[T0] == 0.0 and w["UP"].iloc[T0 + 1] == 1 / 4    # fills the NEXT bar
    assert w["DOWN"].iloc[T0] == 0.0 and w["DOWN"].iloc[T0 + 1] == -1 / 4
    assert (w["FLAT"] == 0).all()                                 # a range never starts
    assert (w.abs().sum(axis=1) <= 1.0 + 1e-12).all()             # never more than the book


def test_chandelier_ratchets_and_a_close_through_it_exits_next_bar():
    p = _base()
    parts, w = _compose(p), _weights(p)
    stop = parts["stop"]["UP"].iloc[T0:CRASH]
    assert stop.notna().all() and (stop.diff().dropna() >= 0).all()   # only ever tightens
    atr = parts["atr"]["UP"]
    # At entry: atr_mult ATRs below the signal close; later: the best high minus 3 ATRs.
    assert np.isclose(stop.iloc[0], p.close["UP"].iloc[T0] - 3.0 * atr.iloc[T0])
    t = CRASH - 1
    assert np.isclose(stop.loc[p.close.index[t]],
                      max(p.high["UP"].iloc[T0 + 1:t + 1].max() - 3.0 * atr.iloc[t],
                          stop.iloc[-2]))
    assert parts["long_exit"]["UP"].iloc[CRASH]                   # the crash closes through it
    assert not parts["long_exit"]["UP"].iloc[T0:CRASH].any()      # a clean trend never does
    assert w["UP"].iloc[CRASH] == 1 / 4 and w["UP"].iloc[CRASH + 1] == 0.0
    # A stop so tight the ±0.5% daily range trips it: out right after the first bar.
    tight = _compose(p, atr_mult=0.01)
    assert tight["long_exit"]["UP"].iloc[T0 + 1]


def test_an_opposite_start_closes_but_never_flips_in_one_bar():
    t = np.arange(N)
    down = _range()
    down[T0:90] = 98.0 * 0.98 ** (t[T0:90] - T0)
    down[90:] = down[89] * 1.05 ** (t[90:] - 89)                  # a sharp reversal up
    p = _make(DOWN=down, FLAT=_range())
    parts = _compose(p, atr_mult=50.0)                            # the stop cannot fire
    rev = int(np.flatnonzero(parts["long_start"]["DOWN"].to_numpy() & (np.arange(N) >= 90))[0])
    assert parts["short_exit"]["DOWN"].iloc[rev]                  # the long start closes it
    assert parts["position"]["DOWN"].iloc[rev] == 0               # ... to flat, not long
    w = _weights(p, atr_mult=50.0)["DOWN"]
    assert w.iloc[rev] == -1 / 4 and w.iloc[rev + 1] == 0.0
    assert not ((w.shift(1) < 0) & (w > 0)).any()                 # short -> long never in one bar


def test_a_full_book_admits_the_strongest_relative_trend_first():
    t = np.arange(N)
    lead, lag = _range(), _range()
    lead[:40], lag[:40] = np.linspace(80, 100, 40), np.linspace(120, 100, 40)
    lead[T0:] = lag[T0:] = 102.0 * 1.02 ** (t[T0:] - T0)          # both break out together
    p = _make(LAG=lag, LEAD=lead)
    parts = _compose(p, max_positions=1)
    assert parts["long_start"]["LEAD"].iloc[T0] and parts["long_start"]["LAG"].iloc[T0]
    assert parts["rel_strength"]["LEAD"].iloc[T0] > parts["rel_strength"]["LAG"].iloc[T0]
    w = _weights(p, max_positions=1)
    assert w["LEAD"].iloc[T0 + 1] == 1.0 and (w["LAG"] == 0).all()   # one slot: the leader


def test_direction_and_adx_switches():
    p = _base()
    assert (_weights(p, direction="long") >= 0).all().all()
    assert (_weights(p, direction="short") <= 0).all().all()
    parts = _compose(p)
    strong = _compose(p, adx_min=30.0)["long_start"]
    assert not (strong & ~parts["long_start"]).any().any()        # a floor only removes starts
    first = int(np.flatnonzero(strong["UP"].to_numpy())[0])
    assert first > T0 and parts["adx"]["UP"].iloc[first] >= 30.0  # confirmation comes later
    above = float(parts["adx"].max().max()) + 0.01                # no breakout is that strong
    assert (_weights(p, adx_min=above) == 0).all().all()


def test_no_look_ahead_future_bars_never_change_past_weights():
    p = _base()
    full = _weights(p)
    for cut in (T0 + 3, CRASH - 5, CRASH + 2):
        idx = p.close.index[:cut]
        short = replace(p, close=p.close.loc[idx], open_=p.open_.loc[idx], high=p.high.loc[idx],
                        low=p.low.loc[idx], volume=p.volume.loc[idx])
        pd.testing.assert_frame_equal(_weights(short), full.loc[idx])


def _minutes():
    """Three 390-minute sessions: UP ranges for an hour, then rises 0.05% a minute to the end."""
    days = ("2024-03-04", "2024-03-05", "2024-03-06")
    index = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"{d} 09:30", f"{d} 15:59", freq="1min", tz="America/New_York")
        for d in days])).tz_convert("UTC")
    n = len(index)
    flat = 100.0 + np.where(np.arange(n) % 2 == 0, 0.05, -0.05)
    up = flat.copy()
    up[60:] = 100.1 * 1.0005 ** np.arange(1, n - 59)
    close = pd.DataFrame({"UP": up, "FLAT": flat}, index=index)
    return Panel(close=close, open_=close.shift(1).fillna(close), high=close * 1.0005,
                 low=close * 0.9995, volume=pd.DataFrame(1e4, index=index, columns=close.columns),
                 bar="1m")


INTRA = dict(FAST, max_positions=2, rs_window=20, rs_skip=2, skip_open=15, skip_close=15)


def test_intraday_blackouts_and_flat_at_the_close():
    p = _minutes()
    parts = _compose(p, **INTRA, eod_exit=True)
    ok = bt.session_mask(p.close.index, p.tz, skip_open=15, skip_close=15).to_numpy()
    starts = parts["long_start"] | parts["short_start"]
    assert not starts.to_numpy()[~ok].any()                       # no starts in the blackouts
    last = bt.session_end(p.close.index, p.tz).to_numpy()
    pos = parts["position"]["UP"].to_numpy()
    assert (pos[last] == 0).all()                                 # flat on every last bar ...
    assert parts["long_exit"]["UP"].to_numpy()[np.roll(last, -1)].all()   # closed the bar before
    w = _weights(p, **INTRA, eod_exit=True)["UP"].to_numpy()
    assert (w[last] == 0).all() and (w > 0).sum() > 3 * 300       # sold at the last bar's open
    first_day2 = int(np.flatnonzero(last)[0]) + 1                 # re-entered after the blackout
    held = np.flatnonzero(pos[first_day2:] == 1)[0]
    assert 15 <= held < 60


def test_intraday_hold_overnight_and_daily_bars_ignore_the_session_rules():
    p = _minutes()
    last = bt.session_end(p.close.index, p.tz).to_numpy()
    w = _weights(p, **INTRA, eod_exit=False)["UP"].to_numpy()
    assert (w[np.flatnonzero(last)[:2] + 1] > 0).all()            # carried across both closes
    daily = _base()
    pd.testing.assert_frame_equal(_weights(daily, skip_open=15, skip_close=15, eod_exit=True),
                                  _weights(daily))


def test_the_one_minute_variant_runs_this_code_with_its_own_config():
    one_m = strategies.load("trend_following_1m")
    assert one_m.signals.__code__.co_filename.endswith("trend_following/strategy.py")
    assert one_m.data["bar"] == "1m" and one_m.params["max_positions"] == 10
    assert one_m.valid(dict(one_m.params))


def test_valid_rules():
    strat = _strat()
    assert strat.valid(dict(strat.params))
    assert not strat.valid({"direction": "sideways"})
    assert not strat.valid({"entry_window": 1})
    assert not strat.valid({"atr_mult": 0.0})
    assert not strat.valid({"adx_min": 100.0})
    assert not strat.valid({"max_positions": 0})
    assert not strat.valid({"rs_window": 21, "rs_skip": 21})


def test_it_runs_through_the_engine_end_to_end():
    """The full contract: weights -> one shared book -> stats, no NaN objectives."""
    from trading_simple.config import load_settings

    strat = _strat()
    stats = bt.evaluate(strat, _base(), {**strat.params, **FAST}, settings=load_settings())
    assert stats["n_trades"] >= 2                                 # UP long and DOWN short
    assert np.isfinite(stats["excess_cagr"]) and np.isfinite(stats["volatility"])
