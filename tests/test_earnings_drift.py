"""earnings_drift: point-in-time event ranks, next-bar timing, slots, holds, borrow tiers —
synthetic panels, no DB."""
import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies
from trading_simple.data import Panel

ED = strategies.load("earnings_drift")
mod = SimpleNamespace(**ED.signals.__globals__)     # the strategy module's helpers
NAMES = [f"S{i:02d}" for i in range(20)]
SMALL = {"trail": 126, "min_events": 10, "slots": 2, "hold": 10}


def _panel(n=300, seed=0, jumps=None, surprises=None, dtc=None):
    """20 names reporting every 63 sessions (staggered); ``jumps`` {(name, bar): factor}
    multiplies the close from that bar on, ``surprises`` {(name, report bar): actual - est}
    adds a report (or overrides one)."""
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(seed)
    px = pd.DataFrame(50.0 * np.cumprod(1 + rng.normal(0.0, 0.01, (n, len(NAMES))), axis=0),
                      index=idx, columns=NAMES)
    for (name, bar), factor in (jumps or {}).items():
        px.iloc[bar:, NAMES.index(name)] *= factor
    nan = pd.DataFrame(np.nan, index=idx, columns=NAMES)
    flag, actual, estimate, day, pct = (nan.copy() for _ in range(5))
    reports = {(name, p): rng.normal(0.0, 0.05) for i, name in enumerate(NAMES)
               for p in range(3 + 3 * i, n - 2, 63)}
    reports.update(surprises or {})
    for (name, p), diff in reports.items():
        if p + mod.REPORT_LAG >= n:
            continue
        day.iloc[p, NAMES.index(name)] = 1.0
        u = p + mod.REPORT_LAG
        flag.iloc[u, NAMES.index(name)] = 1.0
        estimate.iloc[u, NAMES.index(name)] = 1.0
        actual.iloc[u, NAMES.index(name)] = 1.0 + diff
        pct.iloc[u, NAMES.index(name)] = diff
    feats = {"earn_flag": flag, "earn_actual": actual, "earn_estimate": estimate,
             "earn_day": day, "earn_surprise": pct}
    if dtc is not None:
        feats["short_interest"] = dtc.set_axis(idx)
    return Panel(close=px, open_=px.copy(), high=px.copy(), low=px.copy(),
                 volume=pd.DataFrame(1e6, index=idx, columns=NAMES), bar="1d", features=feats)


def _cut(panel, end):
    """The same panel with its rows after bar ``end`` removed (what was known then)."""
    sl = lambda f: f.iloc[: end + 1]  # noqa: E731
    return Panel(close=sl(panel.close), open_=sl(panel.open_), high=sl(panel.high),
                 low=sl(panel.low), volume=sl(panel.volume), bar="1d",
                 features={k: sl(v) for k, v in panel.features.items()})


def _run(panel, **over):
    return ED.signals(panel, **{**ED.params, **over})


def test_signals_on_a_truncated_panel_reproduce_the_full_panels_weights():
    full = _panel(n=400, seed=1)
    params = {"trail": 126, "min_events": 10, "slots": 4, "hold": 30}
    w_full = _run(full, **params)["weights"]
    for end in (150, 233, 320):
        w_cut = _run(_cut(full, end), **params)["weights"]
        pd.testing.assert_frame_equal(w_cut, w_full.iloc[: end + 1])
    assert (w_full != 0).any().any()                       # the book traded at all


def test_an_extreme_event_enters_the_bar_after_its_usable_bar_and_holds_hold_sessions():
    p0 = 207                                     # S05's own report (3 + 15 + 63k); next: 270
    panel = _panel(jumps={("S05", p0): 1.15}, surprises={("S05", p0): 0.8})
    u0 = p0 + mod.REPORT_LAG
    ev = mod.events(panel, trail=126, min_events=10)
    row = ev[(ev["symbol"] == "S05") & (ev["bar"] == u0)].iloc[0]
    assert row["pct"] == pytest.approx(1.0)                # the top event of its window
    over = {**SMALL, "slots": 30}                          # free slots: nothing crowds it out
    w = _run(panel, **over)["weights"]["S05"]
    size = 0.98 * 0.5 / 30
    assert w.iloc[u0] == 0.0                               # decided at u0's close ...
    assert (w.iloc[u0 + 1: u0 + 1 + over["hold"]] == size).all()    # ... held `hold` bars
    assert w.iloc[u0 + 1 + over["hold"]] == 0.0


def test_the_reaction_window_ends_ear_end_sessions_after_the_report():
    p0 = 200
    late = _panel(jumps={("S05", p0 + 2): 1.3}, surprises={("S05", p0): 0.0})  # on u itself
    ev = mod.events(late, trail=126, min_events=10, ear_end=1)
    row = ev[(ev["symbol"] == "S05") & (ev["bar"] == p0 + 2)].iloc[0]
    c, mkt = late.close, (1 + late.close.pct_change(fill_method=None).mean(axis=1)
                          .fillna(0.0)).cumprod()
    expect = c["S05"].iloc[p0 + 1] / c["S05"].iloc[p0 - 1] - mkt.iloc[p0 + 1] / mkt.iloc[p0 - 1]
    assert row["ear"] == pytest.approx(expect)             # the u-bar jump is not in it
    ev2 = mod.events(late, trail=126, min_events=10, ear_end=2)
    assert ev2.loc[row.name, "ear"] > row["ear"] + 0.2     # ear_end=2 reads it


def _book(events, n=40, m=6, ok=None, **over):
    """``_select`` on hand-made events [(bar, col, pct)]: (longs, shorts) arrays."""
    bar, col, pct = (np.array(v) for v in zip(*sorted(events), strict=True))
    kw = {"q": 0.2, "hold": 10, "slots": 2, "sides": "both", **over}
    return mod._select(bar, col, pct.astype(float), n, m,
                       ok_short=np.ones((n, m), bool) if ok is None else ok, **kw)


def test_the_most_extreme_qualifying_events_take_the_free_slots():
    longs, shorts = _book([(5, 0, 0.95), (5, 1, 0.85), (5, 2, 0.99), (5, 3, 0.5),
                           (5, 4, 0.10), (5, 5, 0.02)])
    assert longs[5].nonzero()[0].tolist() == [0, 2]        # two slots: 0.99 and 0.95
    assert shorts[5].nonzero()[0].tolist() == [4, 5]       # 0.5 never qualifies
    late, _ = _book([(5, 0, 0.95), (5, 2, 0.99), (6, 1, 1.0)])
    assert not late[6, 1]                                  # full: a later event is dropped ...
    assert not late[30, 1]                                 # ... and never waits for a slot


def test_a_held_names_next_report_redecides_it():
    longs, shorts = _book([(5, 0, 0.9), (12, 0, 0.95), (20, 1, 0.9), (25, 1, 0.1),
                           (30, 2, 0.9), (33, 2, 0.5)], n=60)
    assert longs[5:22, 0].all() and not longs[22, 0]       # same side: the clock restarts
    assert longs[24, 1] and shorts[25, 1] and not longs[25, 1]    # opposite: flipped
    assert longs[32, 2] and not longs[33, 2] and not shorts[33, 2]  # neutral: closed


def test_a_full_book_is_dollar_neutral_within_gross():
    w = _run(_panel(n=400, seed=2), trail=126, min_events=10, slots=3, hold=60)["weights"]
    gross = w.abs().sum(axis=1)
    assert gross.max() <= 0.98 + 1e-9
    full = gross > 0.98 - 1e-9
    assert full.any()
    assert w[full].sum(axis=1).abs().max() < 1e-9          # 3 longs = 3 shorts in size


def test_borrow_tiers_and_the_no_short_cap_read_days_to_cover():
    dtc = pd.DataFrame(np.nan, index=range(300), columns=NAMES)
    dtc["S01"], dtc["S02"], dtc["S03"], dtc["S04"] = 3.0, 8.0, 12.0, 20.0
    out = _run(_panel(dtc=dtc))
    last = out["borrow"].iloc[-1]
    assert last[["S00", "S01", "S02", "S03", "S04"]].tolist() == [0.005, 0.005, 0.03, 0.1, 0.1]
    _, ok = mod.crowding(_panel(dtc=dtc), **{k: ED.params[k] for k in (
        "dtc_mid", "dtc_high", "short_max_dtc", "borrow_gc", "borrow_mid", "borrow_high")})
    assert ok.iloc[-1][["S00", "S03", "S04"]].tolist() == [True, True, False]


def test_a_name_too_crowded_to_borrow_is_never_shorted():
    p0 = 200
    dtc = pd.DataFrame(20.0, index=range(300), columns=NAMES)        # every name > 15 days
    panel = _panel(jumps={("S05", p0): 0.8}, surprises={("S05", p0): -0.8}, dtc=dtc)
    w = _run(panel, **SMALL)["weights"]
    assert (w >= 0).all().all()


def test_without_earnings_frames_the_book_is_empty():
    panel = _panel()
    bare = Panel(close=panel.close, open_=panel.open_, high=panel.high, low=panel.low,
                 volume=panel.volume, bar="1d", features={})
    assert (_run(bare)["weights"] == 0).all().all()


def test_trailing_ranks_ignore_later_events():
    bar = np.array([0, 1, 2, 3, 4, 10])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    r = mod._trailing_pct(bar, x, trail=5, min_n=2)
    assert np.isnan(r[0])                                  # one event: under min_n
    assert r[1] == pytest.approx(1.0) and r[4] == pytest.approx(1.0)
    assert np.isnan(r[5])                                  # bars 6-10 hold only itself
    assert mod._trailing_pct(bar[:5], x[:5], 5, 2) == pytest.approx(r[:5], nan_ok=True)


def test_valid_rejects_what_the_book_cannot_express():
    assert ED.valid(ED.params)
    for bad in ({"q": 0.5}, {"ear_end": 3}, {"w_ear": 1.5}, {"dtc_mid": 12.0},
                {"sides": "net"}, {"surprise": "eps"}, {"gross": 1.2}, {"retrim_k": 1.0},
                {"cent_max": 0.0}):
        assert not ED.valid({**ED.params, **bad}), bad


def test_every_config_param_is_a_signals_keyword():
    keywords = set(inspect.signature(ED.signals).parameters) - {"panel"}
    assert set(ED.params) == keywords


def test_a_coarse_price_surprise_ranks_on_the_reaction_alone():
    # S07 trades near $10: one cent of EPS is 0.1% of its price, more than a quarter of the
    # synthetic SUE's P20-P80 spread (~0.17%), so its surprise is too coarse to rank.
    panel = _panel(jumps={("S07", 0): 0.2})
    ev = mod.events(panel, trail=126, min_events=10)
    s07, rest = ev[ev["symbol"] == "S07"], ev[ev["symbol"] != "S07"]
    ranked = s07["r_ear"].notna()
    assert ranked.any() and s07.loc[ranked, "coarse"].all()
    assert s07.loc[ranked, "r_sue"].isna().all()
    assert s07.loc[ranked, "score"].to_numpy() == pytest.approx(s07.loc[ranked, "r_ear"])
    judged = rest["r_ear"].notna()                          # past the unjudged warm-up
    assert not rest.loc[judged, "coarse"].any()             # $50 names keep their surprise
    loose = mod.events(panel, trail=126, min_events=10, cent_max=np.inf)
    assert not loose["coarse"].any() and loose.loc[s07.index[ranked], "r_sue"].notna().all()


def test_a_coarse_pct_surprise_is_judged_on_the_estimate():
    panel = _panel()
    panel.features["earn_estimate"]["S08"] *= 0.02          # estimate 0.02: 1 cent = 50%
    ev = mod.events(panel, surprise="pct", trail=126, min_events=10)
    s08 = ev[(ev["symbol"] == "S08") & ev["r_ear"].notna()]
    assert len(s08) and s08["coarse"].all() and s08["r_sue"].isna().all()
    assert not ev.loc[(ev["symbol"] != "S08") & ev["r_ear"].notna(), "coarse"].any()
    alone = mod.events(panel, surprise="pct", w_ear=0.0, trail=126, min_events=10)
    assert alone.loc[alone["symbol"] == "S08", "score"].isna().all()   # surprise alone: unscored


def test_the_coarse_rule_is_point_in_time():
    full = _panel(n=400, seed=3, jumps={("S07", 0): 0.2, ("S11", 0): 0.3})
    full.features["earn_estimate"]["S08"] *= 0.02
    for surprise in ("price", "pct"):
        params = {"surprise": surprise, "trail": 126, "min_events": 10, "slots": 4, "hold": 30}
        w_full = _run(full, **params)["weights"]
        ev_full = mod.events(full, surprise=surprise, trail=126, min_events=10)
        assert ev_full["coarse"].any()
        for end in (180, 290):
            cut = _cut(full, end)
            pd.testing.assert_frame_equal(_run(cut, **params)["weights"], w_full.iloc[: end + 1])
            ev_cut = mod.events(cut, surprise=surprise, trail=126, min_events=10)
            pd.testing.assert_frame_equal(ev_cut, ev_full[ev_full["bar"] <= end])


def _tiny_basis(diff):
    """The review's case: a future splitter on TODAY's share basis (CMG 2018: close ~$6, EPS a
    few cents, whole cents). S07 trades near $6 with an estimate of 0.01 and an actual of
    0.01 + ``diff`` on every report."""
    panel = _panel(n=400, seed=4, jumps={("S07", 0): 0.12})
    f = panel.features
    has = f["earn_flag"]["S07"].notna()
    f["earn_estimate"].loc[has, "S07"] = 0.01
    f["earn_actual"].loc[has, "S07"] = 0.01 + diff
    f["earn_surprise"].loc[has, "S07"] = diff / 0.01
    return panel


def test_a_one_cent_beat_or_miss_on_a_tiny_estimate_cannot_move_the_book():
    beat, miss = _tiny_basis(0.01), _tiny_basis(-0.01)
    params = {"trail": 126, "min_events": 10, "slots": 4, "hold": 30}
    for surprise in ("price", "pct"):
        over = {**params, "surprise": surprise}
        ev = mod.events(beat, surprise=surprise, trail=126, min_events=10)
        s07 = ev[(ev["symbol"] == "S07") & ev["pct"].notna()]
        assert len(s07) >= 3 and s07["coarse"].all()            # it is scored, on EAR alone
        pd.testing.assert_frame_equal(_run(beat, **over)["weights"], _run(miss, **over)["weights"])
        off = {**over, "cent_max": np.inf}                       # no rule: the cent decides
        wb, wm = _run(beat, **off)["weights"]["S07"], _run(miss, **off)["weights"]["S07"]
        assert (wb > 0).sum() > (wm > 0).sum() and (wm < 0).sum() > (wb < 0).sum()


def test_a_window_too_small_to_judge_counts_as_coarse():
    panel = _panel()
    ev = mod.events(panel, trail=126, min_events=10)
    first = ev.index[:9]                       # fewer than min_events surprises so far: unjudged
    assert ev.loc[first, "coarse"].all() and not ev["coarse"].iloc[40:].any()
    off = mod.events(panel, trail=126, min_events=10, cent_max=np.inf)
    assert not off["coarse"].any()             # inf: no rule at all


def test_trailing_spread_ignores_later_events():
    bar = np.array([0, 1, 2, 3, 4, 4, 10])
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 100.0])
    s = mod._trailing_spread(bar, x, trail=5, min_n=3)
    assert np.isnan(s[0]) and np.isnan(s[1])                # under min_n finite values
    assert s[2] == pytest.approx(np.quantile([1.0, 2.0, 3.0], 0.8)
                                 - np.quantile([1.0, 2.0, 3.0], 0.2))
    assert s[4] == s[5] == pytest.approx(np.quantile([1, 2, 3, 5, 6], 0.8)
                                         - np.quantile([1, 2, 3, 5, 6], 0.2))
    assert np.isnan(s[6])                                   # bars 6-10 hold only itself
    assert mod._trailing_spread(bar[:6], x[:6], 5, 3) == pytest.approx(s[:6], nan_ok=True)


def test_crowd_opens_positions_only_in_names_above_the_median_days_to_cover():
    p0 = 207                                     # S05's own report; S06's is 210
    dtc = pd.DataFrame(1.0, index=range(300), columns=NAMES)
    dtc["S06"] = 9.0                                                 # S06 crowded, S05 not
    panel = _panel(jumps={("S05", p0): 1.15, ("S06", 210): 1.15},
                   surprises={("S05", p0): 0.8, ("S06", 210): 0.8}, dtc=dtc)
    over = {**SMALL, "slots": 30}
    every = _run(panel, **over)["weights"]
    crowded = _run(panel, **over, crowd="crowded")["weights"]
    assert every["S05"].iloc[p0 + 3] > 0 and crowded["S05"].iloc[p0 + 3] == 0
    assert crowded["S06"].iloc[210 + 3] > 0
    assert _run(panel, **over, crowd="uncrowded")["weights"]["S06"].iloc[213] == 0
