"""burst_scan: bursts after quiet, engine-exact barrier labels, a purged walk-forward per side.

Two kinds of panel. Engineered ones stamp ONE event into flat 1-minute sessions (close 100,
±0.1 bars, 1,000 shares a minute, a routine 5,000 at every session's bar 5) and assert the exact
bar the detector, the label and the engine see it on. A seeded random panel (random walks with
volume-and-price bursts) runs the whole pipeline — walk-forward included — and is reconciled
trade by trade against vectorbt.
"""

import math

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel, resample_panel

_TZ = "America/New_York"
_N = 40                     # bars per session
_P = {                       # small windows so a few weeks of synthetic sessions warm everything
    "burst_minutes": 3, "profile_days": 3, "atr_days": 3, "dv_days": 3, "warmup_sessions": 6,
    "min_train_sessions": 15, "skip_open": 3, "skip_close": 3, "cooldown": 10,
    "trend_minutes": 10, "min_dollar_volume": 0.0, "min_price": 0.0,
    "long_vol_mult": 3.0, "short_vol_mult": 3.0, "long_range_mult": 0.0, "short_range_mult": 0.0,
    "long_trigger": "volume", "short_trigger": "volume", "long_horizon": 20, "short_horizon": 20,
}


def _strat():
    return strategies.load("burst_scan")


def _mod():
    return _strat().signals.__globals__


def _params(**overrides) -> dict:
    return {**_strat().params, **_P, **overrides}


def _index(days: int, start: str = "2026-01-05") -> pd.DatetimeIndex:
    opens = [pd.Timestamp(f"{d:%Y-%m-%d} 09:30", tz=_TZ)
             for d in pd.bdate_range(start, periods=days)]
    return pd.DatetimeIndex([o + pd.Timedelta(minutes=m) for o in opens
                             for m in range(_N)]).tz_convert("UTC")


def _D(k: int) -> int:
    """Bar offset of session ``k``'s first minute."""
    return k * _N


def _flat(days: int = 20, symbols=("AAA", "BBB")) -> dict[str, pd.DataFrame]:
    idx = _index(days)
    close = pd.DataFrame(100.0, index=idx, columns=list(symbols))
    volume = pd.DataFrame(1_000.0, index=idx, columns=list(symbols))
    volume.iloc[[_D(k) + 5 for k in range(days)]] = 5_000.0          # the routine busy minute
    return {"close": close, "open_": close.copy(), "high": close + 0.1, "low": close - 0.1,
            "volume": volume}


def _panel(f: dict, bar: str = "1m") -> Panel:
    return Panel(close=f["close"], open_=f["open_"], high=f["high"], low=f["low"],
                 volume=f["volume"], bar=bar, tz=_TZ)


def _stamp(f: dict, row: int, sym: str, *, o=None, h=None, lo=None, c=None, v=None) -> None:
    for key, val in (("open_", o), ("high", h), ("low", lo), ("close", c), ("volume", v)):
        if val is not None:
            f[key].iloc[row, f[key].columns.get_loc(sym)] = val


def _random_panel(days: int = 90, n_sym: int = 6, seed: int = 0) -> Panel:
    """Random walks with bursts: a few mid-session bars get 8x volume and a 0.8% move."""
    rng = np.random.default_rng(seed)
    idx = _index(days)
    shape = (len(idx), n_sym)
    ret = rng.normal(0.0, 0.001, shape)
    vol = np.tile(np.r_[np.full(3, 5_000.0), np.full(_N - 3, 1_000.0)], days)[:, None]
    vol = vol * rng.uniform(0.7, 1.3, shape)
    for _ in range(days * n_sym // 2):
        r, c = int(rng.integers(0, len(idx))), int(rng.integers(0, n_sym))
        if 8 <= r % _N <= _N - 8:
            vol[r, c] *= 8.0
            ret[r, c] += rng.choice([-1.0, 1.0]) * 0.008
    cols = [f"S{i}" for i in range(n_sym)]
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(ret, axis=0)), index=idx, columns=cols)
    opn = close.shift(1).fillna(close.iloc[0])
    wick = pd.DataFrame(np.abs(rng.normal(0.0, 5e-4, shape)), index=idx, columns=cols)
    return Panel(close=close, open_=opn, high=np.maximum(close, opn) * (1 + wick),
                 low=np.minimum(close, opn) * (1 - wick),
                 volume=pd.DataFrame(vol, index=idx, columns=cols), bar="1m", tz=_TZ)


@pytest.fixture(scope="module")
def rnd():
    """The seeded random panel and its composed default run (computed once)."""
    panel = _random_panel()
    return panel, _mod()["compose"](panel, **_params())


def _reconcile(panel: Panel, c: dict) -> pd.DataFrame:
    """Every traded event joined to the engine's trade on (symbol, entry bar)."""
    trades = bt.trade_records(bt.run(panel, c["signals"], fees=0.0, slippage=0.0))
    ix = panel.close.index
    rows = []
    for side, t in c["events"].items():
        mine = t[t["traded"]].assign(entry_ts=lambda x: ix[x["row"].to_numpy() + 1])
        theirs = trades[trades["direction"] == side.capitalize()]
        rows.append(theirs.merge(mine, on=["symbol", "entry_ts"], how="outer", indicator=True)
                    .assign(side=side))
    return pd.concat(rows, ignore_index=True)


# --- the shipped config ----------------------------------------------------------------------
def test_shipped_config_is_coherent_and_searches_nothing():
    strat = _strat()
    assert strat.valid(strat.params) and strat.valid(_params())
    assert strat.optuna["space"] == {} and strat.optuna["n_trials"] == 1
    assert strat.optuna["objectives"] == ["alpha_cagr", "volatility"]
    assert strat.order_notional == 1000.0
    longs = {k[5:] for k in strat.params if k.startswith("long_")}
    shorts = {k[6:] for k in strat.params if k.startswith("short_")}
    assert longs == shorts and len(longs) == 8          # every sub-strategy knob exists twice
    assert set(strat.params) == set(_mod()["_DEFAULTS"])


def test_valid_rejects_incoherent_combos():
    v = _strat().valid
    assert not v({"skip_open": 5, "burst_minutes": 10})    # a window straddling the open
    assert not v({"cooldown": 5, "burst_minutes": 10})
    assert not v({"long_trigger": "sometimes"})
    assert not v({"short_trade_q": 1.0})
    assert not v({"long_horizon": 5 * 390 + 1})
    assert not v({"warmup_sessions": 5, "profile_days": 10})


# --- the detector ----------------------------------------------------------------------------
def _detect(panel: Panel, side: str = "long", **overrides) -> np.ndarray:
    m, p = _mod(), _params(**overrides)
    q = m["_side_params"](p, side)
    ctx = m["context"](panel, **p)
    return m["detect"](ctx, trigger=q["trigger"], impulse=q["impulse"], vol_mult=q["vol_mult"],
                       range_mult=q["range_mult"], cooldown=m["_bars"](p["cooldown"], panel.bar))


def _burst_panel() -> dict:
    """Session 10, bar 20: AAA trades 8x its volume and jumps +1% on that one minute."""
    f = _flat()
    _stamp(f, _D(10) + 20, "AAA", c=101.0, h=101.1, lo=99.9, v=8_000.0)
    for r in range(_D(10) + 21, _D(20)):
        _stamp(f, r, "AAA", o=101.0, c=101.0, h=101.1, lo=100.9)
    return f


def test_mid_session_burst_fires_once_and_the_routine_busy_minute_never_does():
    ev = _detect(_panel(_burst_panel()))
    assert list(np.flatnonzero(ev[:, 0])) == [_D(10) + 20]     # first bar of the cluster only
    assert not ev[:, 1].any()                                  # BBB: the 5x minute is routine
    assert not ev[[_D(k) + 5 for k in range(20)]].any()


def test_impulse_and_each_sides_own_thresholds_select_the_events():
    panel = _panel(_burst_panel())
    assert _detect(panel, "long", long_impulse="up")[_D(10) + 20, 0]
    assert not _detect(panel, "long", long_impulse="down")[:, 0].any()
    assert _detect(panel, "short", short_impulse="up")[_D(10) + 20, 0]       # a fade short
    assert not _detect(panel, "short", short_vol_mult=10.0)[:, 0].any()
    assert _detect(panel, "long", short_vol_mult=10.0)[_D(10) + 20, 0]       # untouched


def test_range_and_either_triggers():
    f = _flat()
    _stamp(f, _D(12) + 25, "BBB", c=101.5, h=101.6, lo=99.9)                # a move, no volume
    panel = _panel(f)
    kw = {"long_range_mult": 3.0}
    assert _detect(panel, long_trigger="range", **kw)[_D(12) + 25, 1]
    assert _detect(panel, long_trigger="either", **kw)[_D(12) + 25, 1]
    assert not _detect(panel, long_trigger="volume", **kw)[:, 1].any()
    assert not _detect(panel, long_trigger="both", **kw)[:, 1].any()


def test_quiet_first_spacing_always_exceeds_the_cooldown():
    rng = np.random.default_rng(3)
    raw = rng.random((500, 4)) < 0.08
    n = 7
    kept = raw & ~_mod()["_recent"](raw, n)
    for col in range(4):
        r = np.flatnonzero(kept[:, col])
        assert (np.diff(r) > n).all()
        for t in np.flatnonzero(raw[:, col] & ~kept[:, col]):     # dropped = a burst was recent
            assert raw[max(t - n, 0):t, col].any()


def test_spaced_keeps_same_symbol_events_more_than_a_horizon_apart():
    rows = np.array([0, 5, 11, 30, 40, 3])
    cols = np.array([0, 0, 0, 0, 0, 1])
    assert list(_mod()["spaced"](rows, cols, 10)) == [True, False, True, True, False, True]


def test_context_and_events_are_unchanged_by_later_data(rnd):
    panel, full = rnd
    m = _mod()
    cut = _D(60)                                               # a session boundary
    short = Panel(close=panel.close.iloc[:cut], open_=panel.open_.iloc[:cut],
                  high=panel.high.iloc[:cut], low=panel.low.iloc[:cut],
                  volume=panel.volume.iloc[:cut], bar="1m", tz=_TZ)
    part = m["compose"](short, **_params())
    for side in ("long", "short"):
        a, b = full["events"][side], part["events"][side]
        a = a[a["row"] < cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(a[["row", "col", *m["FEATURES"]]],
                                      b[["row", "col", *m["FEATURES"]]])
        np.testing.assert_array_equal(a["p"].to_numpy(), b["p"].to_numpy())


# --- outcomes: the label is the trade --------------------------------------------------------
def _outcome(f: dict, row: int, *, side: int = 1, horizon: int = 10, frac: float = 0.01,
             eod: bool = False) -> pd.Series:
    m, panel = _mod(), _panel(f)
    ctx = m["context"](panel, **_params())
    return m["outcomes"](panel, ctx, np.array([row]), np.array([0]), side=side, horizon=horizon,
                         frac=np.array([frac]), eod=eod).iloc[0]


def test_target_first_is_a_long_win_and_a_short_loss_on_the_right_bar():
    r = _D(8) + 20                                             # fill r+1: open 100, close 100
    f = _flat()
    _stamp(f, r + 4, "AAA", h=101.5)
    win = _outcome(f, r)
    assert win["reason"] == 1 and win["win"] and win["exit_row"] == r + 4
    assert win["exit_px"] == pytest.approx(101.0) and win["bars"] == 3
    assert win["ret"] == pytest.approx(0.01)
    loss = _outcome(f, r, side=-1)
    assert loss["reason"] == -1 and not loss["win"] and loss["exit_px"] == pytest.approx(101.0)
    assert loss["ret"] == pytest.approx(-0.01)


def test_quiet_path_times_out_at_the_horizon_open_and_both_touched_is_a_stop():
    r = _D(8) + 10
    quiet = _outcome(_flat(), r, horizon=12)
    assert quiet["reason"] == 0 and quiet["exit_row"] == r + 1 + 12 and quiet["complete"]
    f = _flat()
    _stamp(f, r + 3, "AAA", h=101.5, lo=98.5)                  # both barriers inside one bar
    both = _outcome(f, r)
    assert both["reason"] == -1 and both["exit_px"] == pytest.approx(99.0)


def test_overnight_gap_through_the_stop_fills_at_the_next_open():
    r = _D(8) + _N - 6                                         # near the close, held overnight
    f = _flat()
    for row in range(_D(9), _D(20)):
        _stamp(f, row, "AAA", o=98.0, c=98.0, h=98.1, lo=97.9)
    held = _outcome(f, r, horizon=30)
    assert held["reason"] == -1 and held["exit_row"] == _D(9) and held["nights"] == 1
    assert held["exit_px"] == pytest.approx(98.0) and held["ret"] == pytest.approx(-0.02)
    capped = _outcome(f, r, horizon=30, eod=True)             # flat by the close instead
    assert capped["reason"] == 0 and capped["exit_row"] == _D(9) - 1 and capped["nights"] == 0


def test_a_trade_the_data_cannot_finish_is_incomplete():
    f = _flat()
    out = _outcome(f, _D(19) + 30, horizon=50)
    assert not out["complete"]


def test_barrier_width_scales_with_the_square_root_of_the_horizon(rnd):
    panel, base = rnd
    quarter = _mod()["compose"](panel, **_params(long_horizon=5))
    a, b = base["events"]["long"], quarter["events"]["long"]
    assert len(a) == len(b)
    ratio = (a["frac"] / b["frac"]).dropna()
    assert np.allclose(ratio, math.sqrt(20 / 5))


# --- the walk-forward ------------------------------------------------------------------------
def _wf_inputs(seed: int = 1):
    """Synthetic events on the random panel's clock: feature 0 predicts the label."""
    panel = _random_panel(days=90, n_sym=2)
    m = _mod()
    ctx = m["context"](panel, **_params())
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(np.arange(_D(6), len(panel.close) - 30), 1500, replace=False))
    X = rng.normal(size=(len(rows), 4))
    y = (X[:, 0] + rng.normal(0, 0.5, len(rows)) > 0).astype(float)
    exit_rows = rows + 1 + rng.integers(1, 3 * _N, len(rows))      # some labels run for days
    return m, ctx, rows, X, y, exit_rows


def _wf(m, ctx, rows, X, y, exit_rows, model="lr"):
    return m["walk_forward"](X, y, rows, exit_rows, np.ones(len(rows), bool), ctx, model=model,
                             trade_q=0.33, warmup_sessions=6, min_train_sessions=15)


def test_walk_forward_is_purged_of_labels_still_running_at_the_fold():
    m, ctx, rows, X, y, exit_rows = _wf_inputs()
    base = _wf(m, ctx, rows, X, y, exit_rows)
    month, first = m["month_starts"](ctx["index"], ctx["tz"])
    folds = [f for f in np.unique(base["fold"]) if f >= 0]
    assert len(folds) >= 2
    fm = folds[1]
    start = first[fm]
    running = exit_rows >= start                    # started before, ends inside or later
    poisoned = np.where(running, 1.0 - y, y)
    out = _wf(m, ctx, rows, X, poisoned, exit_rows)
    this = (base["fold"] == fm).to_numpy()
    np.testing.assert_array_equal(base["p"][this], out["p"][this])
    assert not np.array_equal(base["p"][month[rows] > fm], out["p"][month[rows] > fm])


def test_walk_forward_waits_for_training_sessions_and_abstains_on_one_class():
    m, ctx, rows, X, y, exit_rows = _wf_inputs()
    out = _wf(m, ctx, rows, X, y, exit_rows)
    first_session = ctx["codes"][rows]
    month, first = m["month_starts"](ctx["index"], ctx["tz"])
    early = ctx["codes"][first[month[rows]]] - 6 < 15
    assert out["p"][early].isna().all() and out["p"][~early].notna().any()
    assert first_session[out["p"].notna().to_numpy()].min() >= 21
    assert _wf(m, ctx, rows, X, np.zeros(len(rows)), exit_rows)["p"].isna().all()


@pytest.mark.parametrize("model", ["lr", "gbm"])
def test_walk_forward_is_deterministic_and_learns_the_planted_signal(model):
    m, ctx, rows, X, y, exit_rows = _wf_inputs()
    a = _wf(m, ctx, rows, X, y, exit_rows, model)
    b = _wf(m, ctx, rows, X, y, exit_rows, model)
    pd.testing.assert_frame_equal(a, b)
    scored = a["p"].notna().to_numpy()
    assert np.corrcoef(a["p"][scored], X[scored, 0])[0, 1] > 0.8
    picked = a["picked"].to_numpy()
    assert 0.2 < picked[scored].mean() < 0.5 and y[picked].mean() > y[scored].mean()


def test_the_two_sides_never_see_each_others_labels(rnd):
    panel, base = rnd
    other = _mod()["compose"](panel, **_params(short_move_atr=3.0, short_trade_q=0.1))
    keep = ["row", "col", "win", "p", "thr", "picked"]
    pd.testing.assert_frame_equal(base["events"]["long"][keep], other["events"]["long"][keep])
    assert not base["events"]["short"]["win"].equals(other["events"]["short"]["win"])


# --- the book ------------------------------------------------------------------------------------
def test_engine_books_exactly_the_labelled_trades(rnd):
    panel, c = rnd
    j = _reconcile(panel, c)
    assert (j["_merge"] == "both").all() and len(j) > 20
    assert set(j["side"]) == {"long", "short"}
    ix = panel.close.index
    np.testing.assert_allclose(j["exit_price"], j["exit_px"], rtol=1e-9)
    np.testing.assert_array_equal(ix.get_indexer(j["exit_ts"]), j["exit_row"])


def test_entries_are_next_bar_and_barrier_frames_sit_on_the_fill_bar(rnd):
    panel, c = rnd
    sig = c["signals"]
    for side, key in (("long", "entries"), ("short", "short_entries")):
        t = c["events"][side]
        t = t[t["traded"]]
        assert set(zip(*np.nonzero(sig[key].to_numpy()), strict=True)) == set(
            zip(t["row"] + 1, t["col"], strict=True))
        got = sig["sl_stop"].to_numpy()[t["row"] + 1, t["col"]]
        np.testing.assert_allclose(got, t["frac"])
    assert np.array_equal(sig["sl_stop"].isna().to_numpy(), sig["tp_stop"].isna().to_numpy())
    assert sig["cash_frac"] == pytest.approx(0.1)


def test_sides_switch_and_clash_rule(rnd):
    panel, both = rnd
    m = _mod()
    only_long = m["compose"](panel, **_params(sides="long"))
    only_short = m["compose"](panel, **_params(sides="short"))
    assert not only_long["signals"]["short_entries"].to_numpy().any()
    assert not only_short["signals"]["entries"].to_numpy().any()
    assert "short" not in only_long["events"] and "long" not in only_short["events"]
    union = only_long["raw"]["long"].to_numpy() | only_short["raw"]["short"].to_numpy()
    kept = both["raw"]["long"].to_numpy() | both["raw"]["short"].to_numpy()
    assert (kept <= union).all()
    assert int(union.sum() - kept.sum()) == 2 * both["clash"]


def test_positions_are_held_overnight_unless_eod_exit(rnd):
    panel, c = rnd
    sig = c["signals"]
    h = _mod()["_bars"](20, "1m")
    pd.testing.assert_frame_equal(sig["exits"], bt.exit_after(sig["entries"], h))
    flat = _mod()["compose"](panel, **_params(long_horizon=200, long_eod_exit=True))["signals"]
    end = bt.session_end(panel.close.index, _TZ).to_numpy()
    assert flat["exits"].to_numpy()[end].all()
    held = _mod()["compose"](panel, **_params(long_horizon=200))
    nights = held["events"]["long"].query("traded")["nights"]
    assert (nights > 0).any()


def test_runs_on_resampled_bars_and_still_reconciles():
    panel = resample_panel(_random_panel(days=90, n_sym=4, seed=5), 5)
    # A one-minute 8x spike is ~2.4x a five-minute bar's volume, so the threshold drops.
    c = _mod()["compose"](panel, **_params(burst_minutes=5, skip_open=5, cooldown=15,
                                           long_horizon=60, short_horizon=60,
                                           long_vol_mult=1.8, short_vol_mult=1.8))
    assert panel.bar == "5m"
    j = _reconcile(panel, c)
    assert len(j) and (j["_merge"] == "both").all()
    np.testing.assert_allclose(j["exit_price"], j["exit_px"], rtol=1e-9)


# --- research.py (the notebooks' shared computations) ------------------------------------------
@pytest.fixture(scope="module")
def rs():
    """research.py, loaded by path the way the notebooks load it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "burst_research", strategies.STRATEGIES_DIR / "burst_scan" / "research.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_day_t_is_the_clustered_t_by_hand(rs):
    v = np.array([1.0, 3.0, 2.0, 6.0, -1.0, 1.0, 4.0])
    d = np.array([0, 0, 1, 1, 2, 2, 3])
    m = v.mean()
    sums = np.array([(v[d == g] - m).sum() for g in range(4)])
    se = math.sqrt((sums ** 2).sum() * 4 / 3) / len(v)
    out = rs.day_t(v, d)
    assert out["mean"] == pytest.approx(m) and out["t"] == pytest.approx(m / se)
    assert out["n"] == 7 and out["days"] == 4
    same = rs.day_t(np.r_[v, np.nan], np.r_[d, 9])           # NaN dropped, day with it too
    assert same["t"] == pytest.approx(out["t"])


def test_placebos_match_their_events_and_never_sit_on_a_burst(rs, rnd):
    panel, c = rnd
    ctx = c["ctx"]
    t = c["events"]["long"]
    rows, cols = t["row"].to_numpy(), t["col"].to_numpy()
    n = rs.S["_bars"](_P["cooldown"], "1m")
    pool = rs.placebo_pool(ctx, cooldown_bars=n)
    pr, pc, matched = rs.draw_placebo_a(pool, ctx, rows, cols, seed=1)
    assert matched.mean() > 0.9 and len(pr) == matched.sum()
    month, _ = rs.S["month_starts"](ctx["index"], ctx["tz"])
    tod = ctx["feat"]["tod"]
    er, ec = rows[matched], cols[matched]
    assert (pc == ec).all() and (month[pr] == month[er]).all()
    assert (tod[pr] // 30 == tod[er] // 30).all()
    busy = (ctx["vol_ratio"] >= rs.S["_CONTEXT_MULT"]) & ctx["inside"][:, None]
    assert not rs._near(busy, n)[pr, pc].any() and ctx["window"][pr, pc].all()
    # Placebo B: the synthetic bursts' 0.8% jumps have few normal-volume twins, so only part of
    # the events match — the notebook reports that rate. The matched ones must be twins.
    br, bc, bm = rs.draw_placebo_b(ctx, rows, cols, seed=1, sample=1.0)
    assert bm.any() and len(br) == bm.sum()
    assert (ctx["vol_ratio"][br, bc] < 1.5).all() and (month[br] == month[rows[bm]]).all()
    assert (np.sign(ctx["move"][br, bc]) == np.sign(ctx["move"][rows[bm], cols[bm]])).all()
    size = np.abs(ctx["feat"]["burst_atr"])
    edges = np.nanquantile(size[rows, cols], np.linspace(0.1, 0.9, 9))
    np.testing.assert_array_equal(np.searchsorted(edges, size[br, bc]),
                                  np.searchsorted(edges, size[rows[bm], cols[bm]]))


def test_reconcile_passes_the_real_book_and_flags_a_tampered_label(rs, rnd):
    panel, c = rnd
    assert (rs.reconcile(panel, c)["status"] == "ok").all()
    bad = {**c, "events": {s: t.copy() for s, t in c["events"].items()}}
    t = bad["events"]["long"]
    first = t.index[t["traded"]][0]
    t.loc[first, "exit_px"] *= 1.01                          # a label the engine never booked
    status = rs.reconcile(panel, bad)["status"]
    assert (status == "exit differs").sum() == 1


def test_purge_audit_passes_the_strategy_and_catches_a_leaky_walk_forward(rs, rnd, monkeypatch):
    panel, _ = rnd
    c = _mod()["compose"](panel, **_params(long_horizon=200, long_move_atr=2.5,
                                           sides="long"))   # 5-session holds, many still running
    table, p = c["events"]["long"], c["params"]
    audit = rs.purge_audit(table, c["ctx"], p, folds=2)
    assert audit["unchanged"].all()
    assert (audit["labels flipped"] > audit["events"]).all()      # labels from before the fold
    real = rs.S["walk_forward"]

    def leaky(X, y, rows, exit_rows, complete, ctx, **kw):
        return real(X, y, rows, rows, complete, ctx, **kw)    # "ends" at its own bar: no purge

    monkeypatch.setitem(rs.S, "walk_forward", leaky)
    assert not rs.purge_audit(table, c["ctx"], p, folds=2)["unchanged"].all()


def test_overnight_split_adds_up(rs, rnd):
    panel, c = rnd
    t = c["events"]["long"].query("traded & complete")
    parts = rs.split_overnight(panel, c["ctx"], t, "long")
    np.testing.assert_allclose(parts["total"], parts["overnight"] + parts["intraday"])
    np.testing.assert_allclose(np.expm1(parts["total"]), t["ret"], rtol=1e-9, atol=1e-12)
    assert (parts.loc[t["nights"].eq(0).to_numpy(), "overnight"] == 0).all()


# --- the move study (events defined by their outcome) --------------------------------------------
def test_decision_points_are_every_step_inside_the_window(rs, rnd):
    _, c = rnd
    ctx = c["ctx"]
    rows, cols = rs.decision_points(ctx, every_minutes=10)       # 1m panel: every 10th bar
    assert len(rows) and (ctx["bar_no"][rows] % 10 == 0).all()
    assert ctx["window"][rows, cols].all()
    assert len(rows) == int((ctx["window"] & (ctx["bar_no"] % 10 == 0)[:, None]).sum())


def test_scaled_barrier_follows_atr_and_the_square_root_of_time(rs, rnd):
    _, c = rnd
    ctx = c["ctx"]
    rows, cols = rs.decision_points(ctx)
    a = rs.scaled_barrier(ctx, rows, cols, 1.0, minutes=390)
    np.testing.assert_allclose(a, ctx["feat"]["atr_pct"][rows, cols].astype(float))
    np.testing.assert_allclose(rs.scaled_barrier(ctx, rows, cols, 2.0, minutes=97.5), a)


def test_calibrate_k_picks_the_first_k_at_or_below_the_target_rate(rs, rnd):
    panel, c = rnd
    ctx = c["ctx"]
    rows, cols = rs.decision_points(ctx)
    k, table = rs.calibrate_k(panel, ctx, rows, cols, target=0.2)
    assert table["reach"].is_monotonic_decreasing                # a wider barrier is reached less
    chosen = table.loc[table["k"] == k, "reach"].iloc[0]
    assert chosen <= 0.2 and (table.loc[table["k"] < k, "reach"] > 0.2).all()


def test_metric_auc_is_the_mean_of_monthly_aucs(rs):
    from sklearn.metrics import roc_auc_score
    t = pd.DataFrame({"x": [0.1, 0.9, 0.2, 0.8, 0.5, 0.4, 0.7, 0.3],
                      "month": [0, 0, 0, 0, 1, 1, 1, 1]})
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    want = np.mean([roc_auc_score(y[:4], t["x"][:4]), roc_auc_score(y[4:], t["x"][4:])])
    got = rs.metric_auc(t, y, ["x"]).iloc[0]
    assert got["auc"] == pytest.approx(want) and got["edge"] == pytest.approx(abs(want - 0.5))


def test_extra_features_only_use_bars_up_to_the_point(rs, rnd):
    panel, c = rnd
    ctx = c["ctx"]
    rows, cols = rs.decision_points(ctx)
    cut = _D(60)
    keep = rows < cut
    full = rs.extra_features(panel, ctx, rows[keep], cols[keep])
    short = Panel(close=panel.close.iloc[:cut], open_=panel.open_.iloc[:cut],
                  high=panel.high.iloc[:cut], low=panel.low.iloc[:cut],
                  volume=panel.volume.iloc[:cut], bar="1m", tz=_TZ)
    ctx_s = _mod()["context"](short, **_params())
    part = rs.extra_features(short, ctx_s, rows[keep], cols[keep])
    pd.testing.assert_frame_equal(full, part)
    assert full.notna().any().all()


# --- move explorer -------------------------------------------------------------------------
def _ramp_panel(sign: float = 1.0) -> Panel:
    """Flat 100 sessions; session 10 drifts +0.5% at bar 10, then ramps 3% over bars 20-25."""
    f = _flat(days=20)
    path = np.full(_N, 100.0)
    path[10:] = 100.0 * (1 + sign * 0.005)
    path[20:26] = 100.0 * (1 + sign * (0.005 + np.linspace(0.005, 0.03, 6)))
    path[26:] = path[25]
    for b in range(_N):
        r = _D(10) + b
        _stamp(f, r, "AAA", o=path[b - 1] if b else path[b], c=path[b],
               h=max(path[b], path[b - 1] if b else path[b]) + 0.01,
               lo=min(path[b], path[b - 1] if b else path[b]) - 0.01)
    return _panel(f)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_move_episodes_finds_the_takeoff_and_the_hit(rs, sign):
    panel = _ramp_panel(sign)
    ev = rs.move_episodes(panel, move_pct=0.02, window_min=60)
    assert list(ev["symbol"]) == ["AAA"]
    e = ev.iloc[0]
    assert e["direction"] == int(sign) and not e["overnight"]
    assert _D(10) + 9 <= e["start"] < e["hit"] <= _D(10) + 25
    c, h, lo = (panel.close["AAA"].to_numpy(), panel.high["AAA"].to_numpy(),
                panel.low["AAA"].to_numpy())
    edge = h if sign > 0 else lo
    assert sign * (edge[e["hit"]] / e["anchor"] - 1) >= 0.02                  # the move is real
    later = c[e["start"] + 1]                                                 # one bar later ...
    reached = sign * (edge[e["start"] + 2:e["hit"] + 1] / later - 1) >= 0.02
    assert not reached.any()                                                  # ... it no longer was


def test_move_episodes_ignores_flat_tape_and_never_overlaps(rs, rnd):
    assert rs.move_episodes(_panel(_flat()), move_pct=0.02, window_min=60).empty
    panel, _ = rnd
    ev = rs.move_episodes(panel, move_pct=0.005, window_min=30)
    assert len(ev)
    for _, g in ev.groupby("col"):
        g = g.sort_values("start")
        assert (g["start"].to_numpy()[1:] > g["hit"].to_numpy()[:-1]).all()
    h, lo = panel.high.to_numpy(), panel.low.to_numpy()
    for r in ev.head(50).itertuples():
        px = h[r.hit, r.col] if r.direction > 0 else lo[r.hit, r.col]
        assert r.direction * (px / r.anchor - 1) >= 0.005 - 1e-12              # it really moved


def test_quiet_controls_share_the_clock_and_never_move(rs, rnd):
    panel, _ = rnd
    ev = rs.move_episodes(panel, move_pct=0.005, window_min=30)
    q = rs.quiet_controls(panel, ev, move_pct=0.005, window_min=30, seed=1)
    assert len(q)
    moves = rs.move_episodes(panel, move_pct=0.005, window_min=30)
    starts = set(zip(moves["col"], moves["start"], strict=True))
    assert not any((r.col, r.start) in starts for r in q.itertuples())
    c, h, lo = (panel.close.to_numpy(), panel.high.to_numpy(), panel.low.to_numpy())
    for r in q.head(50).itertuples():
        a = c[r.start, r.col]
        assert h[r.start + 1:r.start + 31, r.col].max() < a * 1.005
        assert lo[r.start + 1:r.start + 31, r.col].min() > a * 0.995


def test_explore_features_only_use_bars_up_to_the_point(rs, rnd):
    panel, _ = rnd
    cut = _D(60)
    short = Panel(close=panel.close.iloc[:cut], open_=panel.open_.iloc[:cut],
                  high=panel.high.iloc[:cut], low=panel.low.iloc[:cut],
                  volume=panel.volume.iloc[:cut], bar="1m", tz=_TZ)
    full = rs.explore_features(panel, profile_days=3)
    part = rs.explore_features(short, profile_days=3)
    for name, frame in part.items():
        pd.testing.assert_frame_equal(full[name].iloc[:cut], frame, check_freq=False, obj=name)


# --- runs and confirmations ----------------------------------------------------------------
def test_zigzag_alternates_and_flags_the_open_last_swing(rs):
    # up 3%, back 1.5%, up 2.5%: with a 1% reversal -> L, H, L, H(provisional)
    path = np.r_[np.linspace(100, 103, 7), np.linspace(102.5, 101.45, 4),
                 np.linspace(102, 104, 6)]
    piv = rs.zigzag(path, path, 0.01)
    assert [k for _, k, _ in piv] == ["L", "H", "L", "H"]
    assert [p for _, _, p in piv] == [False, False, False, True]
    assert piv[1][0] == 6 and path[piv[2][0]] == pytest.approx(101.45)
    assert rs.zigzag(np.full(10, 100.0), np.full(10, 100.0), 0.01) == []    # never moves


def _swing_panel(sign: float = 1.0) -> Panel:
    """Session 10: bars 5-9 slide 0.8% (onset at bar 9), then bars 10-20 run 3% the other way."""
    f = _flat(days=20)
    path = np.full(_N, 100.0)
    path[5:10] = 100.0 * (1 - sign * np.linspace(0.002, 0.008, 5))
    path[10:21] = path[9] * (1 + sign * np.linspace(0.003, 0.03, 11))
    path[21:] = path[20]
    _stamp_path(f, path)
    return _panel(f)


def _stamp_path(f: dict, path: np.ndarray, session: int = 10) -> None:
    """Write a close path into AAA's session; each bar spans its previous close to its close."""
    for b in range(_N):
        prev = path[b - 1] if b else path[b]
        _stamp(f, _D(session) + b, "AAA", o=prev, c=path[b], h=max(path[b], prev),
               lo=min(path[b], prev))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_swing_runs_find_the_onset_confirmation_and_hit(rs, sign):
    panel = _swing_panel(sign)
    runs = rs.swing_runs(panel, reversal_pct=0.004, min_run_pct=0.02, max_run_min=60,
                         confirm_pct=0.005)
    runs = runs[runs["symbol"] == "AAA"]
    assert len(runs) == 1
    r = runs.iloc[0]
    assert r["direction"] == int(sign) and r["onset"] == _D(10) + 9      # the true bottom / top
    base = r["base"]
    c = panel.close["AAA"].to_numpy()
    edge = (panel.high if sign > 0 else panel.low)["AAA"].to_numpy()
    assert sign * (c[r["confirm"]] / base - 1) >= 0.005                  # confirmed on a close
    assert sign * (c[r["confirm"] - 1] / base - 1) < 0.005 or r["confirm"] == r["onset"]
    assert sign * (edge[r["hit"]] / base - 1) >= 0.02                    # the 2% was reached
    assert r["onset"] < r["confirm"] < r["hit"] <= r["peak"] and r["size"] >= 0.02


def test_swing_runs_reject_a_run_slower_than_the_window(rs):
    runs = rs.swing_runs(_swing_panel(), reversal_pct=0.004, min_run_pct=0.02, max_run_min=5,
                         confirm_pct=0.005)
    assert not (runs["symbol"] == "AAA").any()                                  # 2% took ~10 min


def test_confirmations_label_runs_and_failures(rs):
    runs_panel = _swing_panel()
    conf = rs.confirmations(runs_panel, confirm_pct=0.005, lookback_min=30, min_run_pct=0.02,
                            max_run_min=60)
    up = conf[(conf["symbol"] == "AAA") & (conf["direction"] == 1)
              & (conf["ref"] == _D(10) + 9)]
    assert len(up) == 1 and up.iloc[0]["outcome"] == 1                           # it ran
    assert up.iloc[0]["r_multiple"] > 1
    e = up.iloc[0]
    assert e["ret"] == pytest.approx(e["base"] * 1.02 / e["entry"] - 1)          # exits at target
    # a pop of 0.6% that falls back through its low fails
    f = _flat(days=20)
    path = np.full(_N, 100.0)
    path[10], path[11], path[12], path[13:] = 99.5, 100.2, 99.3, 99.3
    _stamp_path(f, path)
    conf = rs.confirmations(_panel(f), confirm_pct=0.005, lookback_min=30, min_run_pct=0.02,
                            max_run_min=60)
    pop = conf[(conf["symbol"] == "AAA") & (conf["direction"] == 1) & (conf["ref"] == _D(10) + 10)]
    assert len(pop) == 1 and pop.iloc[0]["outcome"] == -1
    assert pop.iloc[0]["ret"] == pytest.approx(pop.iloc[0]["base"] / pop.iloc[0]["entry"] - 1)
    # every confirmation is causal: its reference lies at or before it, inside the lookback
    assert (conf["ref"] <= conf["t"]).all() and ((conf["t"] - conf["ref"]) < 30).all()


def test_confirmations_need_a_real_crossing_not_a_window_roll(rs):
    """An old low leaving the lookback must not 'confirm' a move that happened long ago."""
    f = _flat(days=20)
    path = np.full(_N, 101.0)
    path[:5] = 100.0
    path[10] = 99.5                     # the deep low; price then sits at 101 (1.5% past it)
    path[11:25] = 101.0
    path[25] = 100.0                    # a shallower dip: 101 is 1% past it
    path[26:] = 101.0
    _stamp_path(f, path)
    conf = rs.confirmations(_panel(f), confirm_pct=0.005, lookback_min=20, min_run_pct=0.02,
                            max_run_min=60)
    a = conf[(conf["symbol"] == "AAA") & (conf["direction"] == 1)
             & (conf["t"] >= _D(10)) & (conf["t"] < _D(11))]
    # bar 5 crosses off the opening low (bar 0) and bar 11 off the deep low (bar 10) — real moves.
    # The bar-25 dip bounced while bar 10 was the reference; when bar 10 leaves the 20-bar window
    # (bar 30) the reference becomes bar 25, already passed: no confirmation may appear for it
    assert set(a["ref"] - _D(10)) == {0, 10}
    assert not (a["t"] - _D(10) >= 30).any()
    c = _panel(f).close["AAA"].to_numpy()
    lvl = a["base"].to_numpy() * 1.005
    prev = c[a["t"].to_numpy() - 1]
    assert ((prev < lvl) | (a["t"] == a["ref"]).to_numpy()).all()


@pytest.mark.parametrize("jump, kept", [(102.5, False), (101.0, True)])
def test_confirmations_drop_a_bar_that_jumps_past_the_target(rs, jump, kept):
    """A bar that clears the whole 2% leaves nothing to confirm; a 1% bar is a confirmation."""
    f = _flat(days=20)
    path = np.full(_N, 100.0)
    path[10:] = jump
    _stamp_path(f, path)
    conf = rs.confirmations(_panel(f), confirm_pct=0.005, lookback_min=20, min_run_pct=0.02,
                            max_run_min=60)
    hit = conf[(conf["symbol"] == "AAA") & (conf["direction"] == 1) & (conf["t"] == _D(10) + 10)]
    assert len(hit) == int(kept)
    if kept:
        assert hit.iloc[0]["entry"] < hit.iloc[0]["base"] * 1.02


def test_confirmations_skip_unprinted_bars(rs):
    """A no-print bar (NaN) before the crossing: the previous close is the last one printed."""
    f = _flat(days=20)
    path = np.full(_N, 100.0)
    path[10:] = 101.0
    _stamp_path(f, path)
    _stamp(f, _D(10) + 9, "AAA", o=np.nan, h=np.nan, lo=np.nan, c=np.nan, v=0.0)
    conf = rs.confirmations(_panel(f), confirm_pct=0.005, lookback_min=20, min_run_pct=0.02,
                            max_run_min=60)
    up = conf[(conf["symbol"] == "AAA") & (conf["direction"] == 1) & (conf["t"] == _D(10) + 10)]
    assert len(up) == 1 and np.isfinite(up["ret"]).all()


def _ext_panel(days: int = 25, bar: int = 5) -> dict:
    """``days`` full days 04:00-20:00 of flat ``bar``-minute candles, rth flagged by clock."""
    rows = [pd.Timestamp(f"{d:%Y-%m-%d} 04:00", tz=_TZ) + pd.Timedelta(minutes=m)
            for d in pd.bdate_range("2026-01-05", periods=days) for m in range(0, 960, bar)]
    idx = pd.DatetimeIndex(rows).tz_convert("UTC")
    close = pd.DataFrame(100.0, index=idx, columns=["AAA", "BBB"])
    loc = idx.tz_convert(_TZ)
    clock = loc.hour * 60 + loc.minute
    rth = pd.DataFrame(np.repeat(((clock >= 570) & (clock < 960))[:, None], 2, axis=1),
                       index=idx, columns=close.columns)
    return {"close": close, "open_": close.copy(), "high": close + 0.1, "low": close - 0.1,
            "volume": pd.DataFrame(1_000.0, index=idx, columns=close.columns), "rth": rth}


def _ext(f: dict, bar: int = 5) -> Panel:
    return Panel(close=f["close"], open_=f["open_"], high=f["high"], low=f["low"],
                 volume=f["volume"], bar=f"{bar}m", tz=_TZ, rth=f["rth"])


def _row(f: dict, day: int, hhmm: str) -> int:
    loc = f["close"].index.tz_convert(_TZ)
    days = sorted(set(loc.date))
    return int(np.flatnonzero((loc.date == days[day]) & (loc.strftime("%H:%M") == hhmm))[0])


def test_mask_placeholders_blanks_prices_and_keeps_volume(rs):
    f = _ext_panel(days=2)
    f["volume"].iloc[3, 0] = 0.0
    out = rs.mask_placeholders(_ext(f))
    assert out.close.iloc[3].isna().tolist() == [True, False]
    assert out.high.iloc[3, 0] != out.high.iloc[3, 0] and out.volume.iloc[3, 0] == 0.0
    assert out.close.notna().sum().sum() == out.close.size - 1


def test_mask_late_reports_blanks_08_00_to_08_09_only(rs):
    f = _ext_panel(days=2, bar=1)
    out = rs.mask_late_reports(_ext(f, bar=1))
    clock = out.close.index.tz_convert(_TZ).hour * 60 + out.close.index.tz_convert(_TZ).minute
    gone = out.close.isna().all(axis=1).to_numpy()
    assert (gone == ((clock >= 480) & (clock < 490))).all()
    assert (out.volume.to_numpy() == 1_000.0).all()


def test_session_parts_and_split_ignore_the_missing_bar_flag(rs):
    """The loader reads a missing bar's rth as True; an unprinted row must not open the session."""
    f = _ext_panel(days=2)
    r = _row(f, 0, "05:00")
    f["close"].iloc[r] = np.nan                       # nobody printed at 05:00 ...
    f["rth"].iloc[r] = True                           # ... and the loader says 'rth'
    parts = rs.session_parts(_ext(f))
    loc = f["close"].index.tz_convert(_TZ)
    clock = loc.hour * 60 + loc.minute
    want = np.where(clock < 570, "pre", np.where(clock < 960, "rth", "post"))
    assert (parts == want).all()
    ext, reg = rs.split_sessions(_ext(f))
    assert len(reg.close) == 2 * 78 and (reg.close.index.tz_convert(_TZ).hour >= 9).all()
    assert (ext.rth.iloc[:, 0].to_numpy() == (want == "rth")).all()


def _day(f: dict, day: int):
    return sorted(set(f["close"].index.tz_convert(_TZ).date))[day]


def _open_day(race_high: float | None = 105.0) -> Panel:
    """Day 20: yesterday closes 100, after-hours ends 101, pre-market 101 → 102, opens 102.5."""
    f = _ext_panel()
    _stamp(f, _row(f, 19, "19:55"), "AAA", o=101.0, h=101.0, lo=101.0, c=101.0)
    _stamp(f, _row(f, 20, "08:55"), "AAA", o=101.0, h=101.0, lo=101.0, c=101.0)
    _stamp(f, _row(f, 20, "09:25"), "AAA", o=102.0, h=102.0, lo=102.0, c=102.0)
    first, last = _row(f, 20, "04:00"), _row(f, 20, "09:25")
    f["volume"].iloc[first:last + 1, 0] = 2_000.0                   # a pre-market twice as busy
    for r in range(_row(f, 20, "09:30"), _row(f, 20, "15:55") + 1):
        _stamp(f, r, "AAA", o=102.5, h=102.6, lo=102.4, c=102.5)
    if race_high is not None:
        _stamp(f, _row(f, 20, "09:40"), "AAA", h=race_high)
    return _ext(f)


def test_open_table_knows_the_extended_hours_and_races_from_the_open(rs):
    ext, _ = rs.split_sessions(_open_day())
    t = rs.open_table(ext, move_pct=0.02, window_min=60, trend_min=30)
    d20 = _day(_ext_panel(), 20)
    aaa20 = (t["symbol"] == "AAA") & (t["day"] == d20)
    row = t[aaa20].iloc[0]
    assert row["gap"] == pytest.approx(0.025) and row["pm_ret"] == pytest.approx(0.02)
    assert row["ah_ret"] == pytest.approx(0.01)
    assert row["pm_trend"] == pytest.approx(102 / 101 - 1)                  # 08:55 → 09:25
    assert row["open_vs_pm"] == pytest.approx(102.5 / 102 - 1)
    assert row["pm_vol_rel"] == pytest.approx(2.0)
    assert row["pm_vol_share"] == pytest.approx(66 * 2_000 / (78 * 1_000))
    assert row["race"] == 1 and row["ret"] == pytest.approx(0.0)
    assert set(rs.OPEN_BASE) | set(rs.OPEN_EXT) <= set(t.columns)
    # both barriers inside one bar is ambiguous; neither is 0
    ext2, _ = rs.split_sessions(_open_day(race_high=None))
    assert rs.open_table(ext2, move_pct=0.02, window_min=60)[aaa20]["race"].iloc[0] == 0
    f = _ext_panel()
    for r in range(_row(f, 20, "09:30"), _row(f, 20, "10:25") + 1):
        _stamp(f, r, "AAA", h=103.0, lo=97.0)
    t3 = rs.open_table(rs.split_sessions(_ext(f))[0], move_pct=0.02, window_min=60)
    assert np.isnan(t3[aaa20]["race"].iloc[0])


def test_open_table_features_ignore_everything_after_the_open(rs):
    a = rs.open_table(rs.split_sessions(_open_day())[0], move_pct=0.02, window_min=60)
    b = rs.open_table(rs.split_sessions(_open_day(race_high=110.0))[0], move_pct=0.02,
                      window_min=60)
    cols = list(rs.OPEN_BASE) + list(rs.OPEN_EXT)
    upto = a["day"] <= _day(_ext_panel(), 20)          # later days' ATR rightly sees day 20
    pd.testing.assert_frame_equal(a.loc[upto, cols], b.loc[upto, cols])


def test_monthly_walk_predicts_each_month_from_earlier_months_only(rs):
    rng = np.random.default_rng(0)
    months = np.repeat([f"2026-{m:02d}" for m in range(1, 7)], 200)
    x = rng.normal(size=len(months))
    table = pd.DataFrame({"month": months, "x": x, "y": (x + rng.normal(size=len(x)) > 0) * 1.0})
    p = rs.monthly_walk(table, ["x"], "y", warm_months=2)
    warm = table["month"] < "2026-03"
    assert p[warm].isna().all() and p[~warm].notna().all()
    assert rs.auc(p.dropna(), table["y"][p.notna()]) > 0.7
    later = table.copy()
    later.loc[later["month"] == "2026-06", "y"] = 1 - later.loc[later["month"] == "2026-06", "y"]
    q = rs.monthly_walk(later, ["x"], "y", warm_months=2)
    pd.testing.assert_series_equal(p[table["month"] < "2026-06"], q[table["month"] < "2026-06"])


def test_match_runs_follows_a_run_into_the_wider_view(rs):
    idx_in = pd.date_range("2026-01-05 14:30", periods=10, freq="5min", tz="UTC")
    idx_out = pd.date_range("2026-01-05 12:00", periods=60, freq="5min", tz="UTC")
    inner = pd.DataFrame({"col": [0, 0, 1], "direction": [1, -1, 1], "onset": [0, 5, 0],
                          "peak": [4, 8, 3], "base": [100.0, 101.0, 50.0], "size": [.03] * 3})
    outer = pd.DataFrame({"col": [0, 0], "direction": [1, 1], "onset": [20, 40], "peak": [35, 50],
                          "base": [99.0, 98.0], "size": [0.05, 0.04]})
    m = rs.match_runs(inner, idx_in, outer, idx_out)
    assert m.loc[0, "x_onset_ts"] == idx_out[20] and m.loc[0, "x_base"] == 99.0
    assert pd.isna(m.loc[1, "x_onset_ts"]) and pd.isna(m.loc[2, "x_onset_ts"])   # side / name


def test_rank_ic_is_spearman_with_a_day_clustered_t(rs):
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    days = np.repeat(np.arange(100), 4)
    same = rs.rank_ic(x, np.exp(x), days)
    assert same["ic"] == pytest.approx(1.0) and same["n"] == 400
    noise = rs.rank_ic(x, rng.normal(size=400), days)
    assert abs(noise["ic"]) < 0.15 and abs(noise["t"]) < 3
    y = x + rng.normal(size=400)
    assert rs.rank_ic(x, y, days)["ic"] == pytest.approx(
        pd.Series(x).rank().corr(pd.Series(y).rank()))


# --- move detectors -------------------------------------------------------------------------------
def _detector_panel() -> tuple[dict, Panel]:
    """Session 10: flat 100, then 101 at bar 10 and 102.5 from bar 11 (a +2.5% rise)."""
    f = _flat(days=20)
    path = np.full(_N, 100.0)
    path[10], path[11:] = 101.0, 102.5
    _stamp_path(f, path)
    return f, _panel(f)


def test_move_events_label_the_first_touch_and_the_bracket(rs):
    _, panel = _detector_panel()
    rows = np.array([_D(10) + 5, _D(11) + 5])
    ev = rs.move_events(panel, rows, move=0.02, window=20)
    a = panel.symbols.index("AAA")
    assert ev["long"][0, a] and not ev["short"][0, a]
    assert ev["t_long"][0, a] == 6 and ev["first"][0, a] == 1                 # bar 11, entry bar 6
    assert ev["ret_long"][0, a] == pytest.approx(0.02)
    assert ev["ret_short"][0, a] == pytest.approx(-0.02)                      # up first = its stop
    assert not ev["long"][1, a] and ev["first"][1, a] == 0                    # a flat session
    assert ev["ret_long"][1, a] == pytest.approx(0.0)


def test_move_events_both_in_one_bar_is_a_stop_and_never_cross_the_close(rs):
    f = _flat(days=20)
    _stamp(f, _D(10) + 10, "AAA", h=103.0, lo=97.0)
    _stamp(f, _D(11), "AAA", h=103.0)                        # a jump in the NEXT session
    panel = _panel(f)
    a = panel.symbols.index("AAA")
    ev = rs.move_events(panel, np.array([_D(10) + 5, _D(10) + 30]), move=0.02, window=20)
    assert np.isnan(ev["first"][0, a])
    assert ev["ret_long"][0, a] == pytest.approx(-0.02)
    assert ev["ret_short"][0, a] == pytest.approx(-0.02)
    assert not ev["long"][1, a]                              # the window ends at the close


def test_sign_surrogate_keeps_bar_sizes_and_removes_direction(rs, rnd):
    rnd, _ = rnd
    s0, s1 = rs.sign_surrogate(rnd, 0), rs.sign_surrogate(rnd, 1)
    prev = np.vstack([rnd.open_.to_numpy()[:1], rnd.close.to_numpy()[:-1]])
    prev0 = np.vstack([s0.open_.to_numpy()[:1], s0.close.to_numpy()[:-1]])
    real_move = np.abs(np.log(rnd.close.to_numpy() / prev))
    surr_move = np.abs(np.log(s0.close.to_numpy() / prev0))
    np.testing.assert_allclose(surr_move, real_move, atol=1e-9)
    np.testing.assert_allclose(np.log(s0.high.to_numpy() / s0.low.to_numpy()),
                               np.log(rnd.high.to_numpy() / rnd.low.to_numpy()), atol=1e-9)
    assert not np.allclose(s0.close.to_numpy(), s1.close.to_numpy())
    signs = np.sign(np.log(s0.close.to_numpy() / prev0) * np.log(rnd.close.to_numpy() / prev))
    assert abs(np.nanmean(signs)) < 0.05                     # half the bars flipped


def test_detector_features_only_use_bars_up_to_the_point(rs, rnd):
    panel, _ = rnd
    t = _D(60) + 17                                          # mid-session
    k = np.where(np.arange(len(panel.close)) > t, 1.3, 1.0)[:, None]
    moved = Panel(close=panel.close * k, open_=panel.open_ * k, high=panel.high * k,
                  low=panel.low * k, volume=panel.volume * k, bar="1m", tz=_TZ)
    a, b = rs.detector_features(panel), rs.detector_features(moved)
    for name in a:
        np.testing.assert_allclose(a[name][:t + 1], b[name][:t + 1], equal_nan=True, err_msg=name)


def test_causal_alerts_use_only_the_previous_month(rs):
    table = pd.DataFrame({"month": ["2026-01"] * 100 + ["2026-02"] * 100})
    score = np.r_[np.arange(100.0), np.linspace(0, 200, 100)]
    a = rs.causal_alerts(table, score, 0.1)
    assert not a[:100].any()                                  # the first month raises nothing
    thr = np.quantile(np.arange(100.0), 0.9)
    assert (a[100:] == (score[100:] >= thr)).all()
    other = score.copy()
    other[150:] = -1.0                                        # the scored month's other rows
    assert rs.causal_alerts(table, other, 0.1)[120] == a[120]


def test_walk_detector_scores_each_month_from_earlier_months(rs):
    rng = np.random.default_rng(0)
    months = np.repeat([f"2026-{m:02d}" for m in range(1, 7)], 300)
    x = rng.normal(size=len(months))
    table = pd.DataFrame({"month": months, "x": x, "y": (x + rng.normal(size=len(x)) > 0) * 1.0})
    p = rs.walk_detector(table, ["x"], "y", kind="lr", warm_months=2)
    warm = table["month"] < "2026-03"
    assert p[warm].isna().all() and p[~warm].notna().all()
    flipped = table.copy()
    last = flipped["month"] == "2026-06"
    flipped.loc[last, "y"] = 1 - flipped.loc[last, "y"]
    q = rs.walk_detector(flipped, ["x"], "y", kind="lr", warm_months=2)
    pd.testing.assert_series_equal(p[table["month"] < "2026-06"], q[table["month"] < "2026-06"])
