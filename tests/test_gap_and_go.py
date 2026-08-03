"""gap_and_go signal logic on synthetic data with engineered gaps and breaks."""

import numpy as np
import pandas as pd

from trading_simple import strategies
from trading_simple.data import Panel

#: Small windows so the two-session synthetic panel exercises everything fast
#: (on a 1m panel minutes == bars). The liquidity guards are off by default here
#: so the fixtures stay about gaps and breaks; they get their own test.
_PARAMS = {"or_minutes": 10, "entry_span": 30, "gap_min": 0.02, "vol_mult": 0.0,
           "profile_days": 5, "entry_mode": "break", "polarity": "go", "rank_by": "gap",
           "max_candidates": 8, "min_price": 0.0, "min_or_dollar_volume": 0.0,
           "atr_window": 5, "sl_mult": 2.0, "sl_trail": True, "tp_ratio": 0.0,
           "horizon": 8, "direction": "both", "eod_exit": True}

_TZ = "America/New_York"
_N = 60      # bars per session
_DAY2 = _N   # bar offset of day 2's first minute


def _sig(panel, **overrides):
    strat = strategies.load("gap_and_go")
    return strat.signals(panel, **{**strat.params, **_PARAMS, **overrides})


def _index(days=2, n=_N):
    """`days` n-minute sessions (2026-06-01.., 09:30 New York), UTC index."""
    return pd.DatetimeIndex(
        [pd.Timestamp(f"2026-06-{d:02d} 09:30", tz=_TZ) + pd.Timedelta(minutes=m)
         for d in range(1, days + 1) for m in range(n)]).tz_convert("UTC")


def _panel(close, volume=None):
    volume = volume if volume is not None else pd.DataFrame(
        1_000.0, index=close.index, columns=close.columns)
    return Panel(close=close, open_=close.copy(), high=close + 0.2, low=close - 0.2,
                 volume=volume, bar="1m")


def _gap_panel(gap_close=103.0, break_close=103.5, break_bar=12):
    """Day 1 flat at 100; day 2 AAA gaps +3% and breaks its opening range.

    AAA day 2: bars 0..9 flat at ``gap_close`` (the 10-bar opening range is
    [gap_close - 0.2, gap_close + 0.2]), then ``break_close`` from ``break_bar``
    on. BBB mirrors it downward: gaps to 97 and breaks the range low at the
    same bar.
    """
    idx = _index()
    close = pd.DataFrame({"AAA": 100.0, "BBB": 100.0}, index=idx)
    close.iloc[_DAY2:, 0] = gap_close
    close.iloc[_DAY2 + break_bar:, 0] = break_close
    close.iloc[_DAY2:, 1] = 97.0
    close.iloc[_DAY2 + break_bar:, 1] = 96.5
    return _panel(close)


def test_break_enters_next_bar_and_only_once():
    sig = _sig(_gap_panel())
    # Break of the frozen range high closes at bar 12 -> execution at bar 13,
    # and nowhere else: the first-of-session edge stops the still-above repeats.
    assert list(np.where(sig["entries"]["AAA"])[0]) == [_DAY2 + 13]
    assert not sig["entries"].iloc[_DAY2 + 12].any()       # never the signal bar
    assert not sig["entries"].iloc[:_DAY2].any().any()     # day 1 has no prior close
    # BBB's mirror: gap down + range-low break -> short at the same bar.
    assert list(np.where(sig["short_entries"]["BBB"])[0]) == [_DAY2 + 13]
    assert not sig["short_entries"]["AAA"].any()           # gap-up name never shorts
    assert not sig["entries"]["BBB"].any()                 # gap-down name never longs


def test_break_during_formation_is_absorbed_by_the_range():
    # A "break" INSIDE the formation window is not a break at all: the range is
    # NaN while forming (no signal can fire there), and the move becomes part
    # of the frozen extremes — the tape never clears its own range after.
    sig = _sig(_gap_panel(break_bar=5))
    assert not sig["entries"].any().any()


def test_gap_gate_blocks_small_gaps_and_ablates_to_zero():
    p = _gap_panel(gap_close=100.5, break_close=101.0)     # +0.5% gap, real break
    assert not _sig(p)["entries"]["AAA"].any()             # gate on: too small
    sig = _sig(p, gap_min=0.0)                             # gate off: OR-only control
    assert sig["entries"]["AAA"].iloc[_DAY2 + 13]


def test_entry_window_closes_entry_span_bars_after_the_range():
    # entry_span is an OFFSET from the range close: the window is [10, 10+30].
    sig = _sig(_gap_panel(break_bar=45))                   # break past bar 40
    assert not sig["entries"]["AAA"].any()


def test_only_the_first_trigger_of_a_session_fires():
    # Price clears the range, falls back INSIDE it, then clears it again. The
    # session already had its break event; a re-cross is not a second one.
    # ("not consecutive", the old rule, would have entered twice.)
    p = _gap_panel()
    p.close.iloc[_DAY2 + 14:_DAY2 + 20, 0] = 103.0         # back inside the range
    p.close.iloc[_DAY2 + 20:, 0] = 103.5                   # and out again
    p = _panel(p.close)
    assert list(np.where(_sig(p)["entries"]["AAA"])[0]) == [_DAY2 + 13]


def test_time_mode_enters_at_the_window_open_on_the_gates_alone():
    sig = _sig(_gap_panel(break_close=103.0), entry_mode="time")  # no break at all
    assert list(np.where(sig["entries"]["AAA"])[0]) == [_DAY2 + 11]   # bar 10 -> 11 fill
    assert list(np.where(sig["short_entries"]["BBB"])[0]) == [_DAY2 + 11]


def test_direction_routing_and_eod_flat():
    sig = _sig(_gap_panel(), direction="long")
    assert not sig["short_entries"].any().any()
    assert sig["entries"]["AAA"].iloc[_DAY2 + 13]
    # Mandatory flat on each session's final bar rides on the exit frames.
    assert sig["exits"].iloc[_N - 1].all() and sig["exits"].iloc[-1].all()
    # Time stop: entry fill at bar 13 exits horizon bars later.
    assert sig["exits"]["AAA"].iloc[_DAY2 + 13 + _PARAMS["horizon"]]
    assert sig["sl_trail"] is True


def test_polarity_fade_takes_the_opposite_side_of_the_same_event():
    # Identical event, mirrored position: the gap-up break is shorted and the
    # gap-down break is bought. Same bar, so this is purely a side swap.
    go, fade = _sig(_gap_panel()), _sig(_gap_panel(), polarity="fade")
    assert fade["short_entries"]["AAA"].equals(go["entries"]["AAA"])
    assert fade["entries"]["BBB"].equals(go["short_entries"]["BBB"])
    assert not fade["entries"]["AAA"].any() and not fade["short_entries"]["BBB"].any()
    # direction is applied AFTER polarity, so a long-only fade trades gap-downs.
    only_long = _sig(_gap_panel(), polarity="fade", direction="long")
    assert only_long["entries"]["BBB"].any() and not only_long["short_entries"].any().any()


def _watchlist_panel(gaps=(0.06, 0.05, 0.04, 0.03, 0.025), break_bar=12):
    """Day 1 flat at 100; day 2 every name gaps up by its own amount and breaks."""
    idx = _index()
    close = pd.DataFrame(100.0, index=idx, columns=[f"S{i}" for i in range(len(gaps))])
    for i, gap in enumerate(gaps):
        close.iloc[_DAY2:, i] = 100.0 * (1 + gap)
        close.iloc[_DAY2 + break_bar:, i] = 100.0 * (1 + gap) + 0.5
    return _panel(close)


def test_watchlist_keeps_only_the_top_candidates_of_the_session():
    # Five names all gap and all break. Without the cap the wallet would fund
    # whichever columns the engine reached first; with it, conviction decides.
    assert _sig(_watchlist_panel())["entries"].any().sum() == 5
    top2 = _sig(_watchlist_panel(), max_candidates=2)["entries"]
    assert list(top2.columns[top2.any()]) == ["S0", "S1"]   # the two biggest gaps
    # The wallet is sized off the same knob, so a full watchlist is 100% invested.
    assert _sig(_watchlist_panel(), max_candidates=2)["cash_frac"] == 0.5
    assert _sig(_watchlist_panel(), max_candidates=8)["cash_frac"] == 0.125


def test_watchlist_ranking_is_frozen_at_the_range_close():
    # S4 has the smallest gap but by far the biggest move AFTER the range
    # closes. Ranking on frozen information must not notice — seeing that move
    # would be look-ahead, and the seat still goes to the biggest gappers.
    panel = _watchlist_panel()
    panel.close.iloc[_DAY2 + 12:, 4] = 500.0
    entries = _sig(_panel(panel.close), max_candidates=2)["entries"]
    assert list(entries.columns[entries.any()]) == ["S0", "S1"]


def test_watchlist_ranks_by_opening_volume_when_asked():
    # Same gaps, but S4 trades 10x the opening volume: rank_by rvol hands it the
    # seat that rank_by gap gave to S0. profile_days warm-up needs more sessions,
    # so this uses the raw participation ordering on a flat baseline.
    idx = _index(days=7)
    day7 = 6 * _N
    names = [f"S{i}" for i in range(5)]
    close = pd.DataFrame(100.0, index=idx, columns=names)
    for i, gap in enumerate((0.06, 0.05, 0.04, 0.03, 0.025)):
        close.iloc[day7:, i] = 100.0 * (1 + gap)
        close.iloc[day7 + 12:, i] = 100.0 * (1 + gap) + 0.5
    volume = pd.DataFrame(1_000.0, index=idx, columns=names)
    volume.iloc[day7:day7 + 10, 4] = 10_000.0
    panel = _panel(close, volume)
    by_gap = _sig(panel, max_candidates=1)["entries"]
    by_rvol = _sig(panel, max_candidates=1, rank_by="rvol")["entries"]
    assert list(by_gap.columns[by_gap.any()]) == ["S0"]
    assert list(by_rvol.columns[by_rvol.any()]) == ["S4"]


def test_prior_close_never_reaches_back_more_than_one_session():
    # AAA does not trade at all on day 2. On day 3 it opens 3% above where it
    # last printed — on DAY 1. That is a two-day move, not an overnight gap, and
    # measuring it against a stale close would manufacture a signal out of a
    # data hole (a real risk on a 200-name panel, harmless on a curated 20).
    idx = _index(days=3)
    day3 = 2 * _N
    close = pd.DataFrame({"AAA": 100.0}, index=idx)
    close.iloc[day3:] = 103.0
    close.iloc[day3 + 12:] = 103.5
    with_hole = close.copy()
    with_hole.iloc[_N:day3] = np.nan
    assert _sig(_panel(close))["entries"]["AAA"].any()          # control: day 2 present
    assert not _sig(_panel(with_hole))["entries"]["AAA"].any()  # day 2 missing -> no gap


def test_volume_gate_wants_elevated_opening_participation():
    # Seven sessions warm the time-of-day baseline (profile_days=5); day 7
    # gaps and breaks. With routine opening volume rvol == 1 and the gate
    # blocks; with 3x volume in the first bars it passes.
    idx = _index(days=7)
    day7 = 6 * _N
    close = pd.DataFrame({"AAA": 100.0}, index=idx)
    close.iloc[day7:] = 103.0
    close.iloc[day7 + 12:] = 103.5
    volume = pd.DataFrame(1_000.0, index=idx, columns=close.columns)
    assert not _sig(_panel(close, volume), vol_mult=1.5)["entries"]["AAA"].any()
    volume.iloc[day7:day7 + 10] = 3_000.0                  # elevated first-K volume
    sig = _sig(_panel(close, volume), vol_mult=1.5)
    assert sig["entries"]["AAA"].iloc[day7 + 13]


def test_liquidity_guards_drop_thin_and_cheap_mornings():
    # The opening range trades 10 bars x 1,000 shares at ~103 = ~$1.03M.
    panel = _gap_panel()
    assert _sig(panel, min_or_dollar_volume=5e5)["entries"]["AAA"].any()
    assert not _sig(panel, min_or_dollar_volume=2e6)["entries"]["AAA"].any()
    assert not _sig(panel, min_price=200.0)["entries"].any().any()


def test_valid_rejects_incoherent_combos():
    strat = strategies.load("gap_and_go")
    assert strat.valid({})                                          # defaults cohere
    assert not strat.valid({"entry_mode": "time", "gap_min": 0.0})  # a schedule, not a thesis
    assert strat.valid({"entry_mode": "time", "gap_min": 0.02})
    assert strat.valid({"entry_mode": "break", "gap_min": 0.0, "vol_mult": 0.0})  # the control
    # What used to need rejecting is now unrepresentable: entry_span is an offset
    # from the range close, and tp_ratio is a multiple of the stop.
    assert strat.valid({"entry_span": 5, "or_minutes": 30})
    assert strat.valid({"tp_ratio": 0.5, "sl_mult": 2.0})
