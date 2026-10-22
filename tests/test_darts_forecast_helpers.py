"""darts_forecast's notebook helpers (research.py, charts.py, checks.py, explorer.py, intraday.py,
daystart.py): fast, offline, synthetic.

The notebooks load these modules by path, and so do these tests (as ``test_darts_forecast.py``
loads ``research.py``). No database and no backtest; the few tests that fit a tiny Darts model
are skipped without the optional ``darts`` extra.
"""

import importlib.util
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from trading_simple import strategies
from trading_simple.config import load_settings
from trading_simple.data import Panel

_DIR = strategies.STRATEGIES_DIR / "darts_forecast"


def _load(name: str):
    """One of the strategy folder's helper modules, loaded by path like the notebooks do."""
    spec = importlib.util.spec_from_file_location(f"darts_forecast_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R, CH, CK = _load("research"), _load("charts"), _load("checks")


def _frames(**arrays) -> dict:
    """Bool arrays as the 1-minute signal frames ``strategy.trade_frames`` returns."""
    return {k: pd.DataFrame(np.asarray(v, bool)) for k, v in arrays.items()}


# --- the position path: holdings, their starts, and the check against the rule's decisions --------
def test_held_and_holding_starts():
    """An entry while already held (trade_frames' re-entry marker after a stop) starts nothing."""
    e = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 1], [0, 0], [0, 0]], bool)
    x = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]], bool)
    h = R.held(e, x)
    assert h[:, 0].tolist() == [False, True, True, True, True, False, False]
    assert h[:, 1].tolist() == [False, False, False, False, True, True, False]
    flat = np.zeros_like(e)
    longs = R.holding_starts(_frames(entries=e, exits=x, short_entries=flat, short_exits=flat))
    assert np.argwhere(longs).tolist() == [[1, 0], [4, 1]]
    shorts = R.holding_starts(_frames(entries=flat, exits=flat, short_entries=e, short_exits=x))
    assert np.array_equal(shorts, longs)  # both sides count


def test_path_misses_counts_disagreements_with_the_decisions():
    """A decision at a grid bar's close fills one 1-minute row later and must be held from there."""
    end = np.array([1, 3, 5, 7, 9])  # grid bar k ends on 1-minute row 2k + 1
    d = {"long": np.array([[0], [1], [0], [0], [0]], bool), "short": np.zeros((5, 1), bool)}

    def path(entry: int, exit_: int) -> dict:
        e, x, z = np.zeros((10, 1), bool), np.zeros((10, 1), bool), np.zeros((10, 1), bool)
        e[entry], x[exit_] = True, True
        return _frames(entries=e, exits=x, short_entries=z, short_exits=z)

    assert CK.path_misses(d, path(4, 6), end, R.held) == 0  # filled on row 4, held rows 4-5
    # an entry on no fill bar, and a decision whose side is never held
    assert CK.path_misses(d, path(7, 9), end, R.held) == 2


# --- the leakage checks' perturbations and comparisons --------------------------------------------
def test_scrambled_replaces_only_the_masked_bars():
    """Fresh random bars after the cut, the bars before it untouched, a missing bar missing."""
    idx = pd.date_range("2025-07-01 13:30", periods=40, freq="min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-3, (40, 2)), axis=0))
    f = lambda a: pd.DataFrame(a.copy(), index=idx, columns=["A", "B"])  # noqa: E731
    p = Panel(close=f(c), open_=f(c), high=f(c * 1.001), low=f(c * 0.999),
              volume=f(np.full((40, 2), 1e3)), bar="1m", tz="America/New_York")
    p.close.iloc[30, 1] = np.nan
    q = CK.scrambled(p, np.arange(40) >= 25, seed=0)
    for field in ("open_", "high", "low", "close", "volume"):
        a, b = getattr(p, field).to_numpy(), getattr(q, field).to_numpy()
        assert np.array_equal(a[:25], b[:25])
        assert not np.allclose(a[25:30], b[25:30])
    assert np.isnan(q.close.iloc[30, 1]) and np.isnan(q.volume.iloc[30, 1])
    top = (q.high - np.maximum(q.open_, q.close)).to_numpy()[25:]
    assert np.nanmin(top) >= 0  # a valid bar: the high above the open and the close


def test_same_inputs_compares_up_to_a_bar():
    """Two build_inputs-like results agree before a changed bar, and not over all of them."""
    rng = np.random.default_rng(0)
    a = {"names": ["A"], "past": {"A": rng.normal(size=(10, 3))}, "y": rng.normal(size=(10, 1)),
         "scale": np.ones((10, 1)), "target": rng.normal(size=(10, 1)),
         "valid": np.ones((10, 1), bool)}
    b = {**{k: v.copy() for k, v in a.items() if isinstance(v, np.ndarray)},
         "names": ["A"], "past": {"A": a["past"]["A"].copy()}}
    assert CK.same_inputs(a, b)
    b["past"]["A"][7, 1] += 1.0
    assert CK.same_inputs(a, b, upto=7) and not CK.same_inputs(a, b)
    b["past"]["A"][7, 1] -= 1.0
    b["valid"][2, 0] = False
    assert not CK.same_inputs(a, b, upto=7)


def test_vs_close_tells_the_labelled_day_from_the_next():
    """A file holding the labelled day's close passes; one holding the next session's fails."""
    sessions = pd.DatetimeIndex(pd.bdate_range("2025-07-01", periods=30))
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, (30, 2)), axis=0))
    i, j = np.repeat(np.arange(29), 2), np.tile([0, 1], 29)
    days, names = sessions[i], np.array(["A", "B"])[j]
    kw = {"sessions": sessions, "symbols": ["A", "B"]}
    same = CK.vs_close(close[i, j], days, names, close, "same day", **kw)
    late = CK.vs_close(close[i + 1, j], days, names, close, "a day late", **kw)
    assert same["ok"] and same["rows"] == 58 and not late["ok"]
    # rows on the last session (no next one), of another name, or without a value are left out
    v, si, sj = CK.per_day([1.0, 2.0, np.nan], [sessions[-1], sessions[0], sessions[1]],
                           ["A", "ZZZ", "B"], sessions, ["A", "B"])
    assert len(v) == len(si) == len(sj) == 0


# --- date, grid and format utilities --------------------------------------------------------------
def test_date_and_format_utilities():
    """The clock, the New York day, session offsets and the small text formats."""
    assert R.clock(0, 5) == "09:35" and R.clock(0, 5, end=False) == "09:30"
    assert R.clock(77, 5) == "16:00"
    assert R.ymd("2025-12-18 15:30") == "2025-12-18"
    ny = "America/New_York"
    assert R.ny_day("2025-12-19 02:00", ny) == pd.Timestamp("2025-12-18")  # naive = UTC
    assert R.ny_day(pd.Timestamp("2025-12-18 15:00", tz="UTC"), ny) == pd.Timestamp("2025-12-18")
    t = pd.DatetimeIndex(["2025-12-18 14:31"], tz="UTC")
    assert list(R.ny(t, ny)) == [pd.Timestamp("2025-12-18 09:31")]
    assert R.iso(R.ny(t, ny)) == ["2025-12-18 09:31:00"]
    s = ["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-07"]
    assert R.sess("2025-07-03", 1, s) == "2025-07-07" and R.sess("2025-07-03", -2, s) == s[0]
    assert R.sessions_span(1, 3, s) == "2025-07-02 → 2025-07-07 (3)"
    assert R.sessions_span(2, 1, s) == "—"
    fit = {"train_from": "2025-07-01", "train_last": "2025-07-02", "session": "2025-07-07"}
    assert R.fit_windows(fit, 0, s) == "TRAIN 2025-07-01 → 2025-07-02"
    assert R.fit_windows(fit, 1, s).endswith(" + VALIDATION 2025-07-03 → 2025-07-03")
    assert R.runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 2), (4, 4)]
    assert R.runs(np.zeros(3, bool)) == []
    assert [R.cadence_label(c) for c in (0, 390, 1950, 7800, 60)] == [
        "once", "every session", "weekly", "every 20 sessions", "every 60 min (intraday)"]
    assert [R.cadence_name(c) for c in (0, 390, 1950, 60)] == [
        "once", "daily", "every 5 sessions", "every 60 min"]
    assert R.lag_txt(1) == "s (latest)" and R.lag_txt(6) == "s-5 .. s (6)"
    assert R.pct(0.1234) == "+12.3%" and R.pct(None) == "n/a" and R.pct(np.nan) == "n/a"
    assert [R.state(v) for v in (True, False, None, np.nan)] == ["pass", "fail", "not run",
                                                                  "not run"]
    assert R.fmt_space({"type": "float", "low": 1, "high": 2, "log": True}) == "1 – 2 (log)"
    assert R.fmt_space({"type": "int", "low": 1, "high": 9, "step": 2}) == "1 – 9, step 2"
    assert R.fmt_space({"type": "categorical", "choices": [0, 2]}) == "0 | 2"
    assert R.missing(np.array([[1.0, np.nan], [np.nan, np.nan]]), ["A", "B"]) == {
        "A % missing": 50.0, "B % missing": 100.0}


def test_grid_helpers():
    """Grid bars as sessions side by side, the retrain moment, and walk_forward's slices."""
    g = {"code": np.repeat([0, 1, 2], 4), "pos": np.tile(np.arange(4), 3), "n_pos": 4,
         "sess_first": np.array([0, 4, 8]),
         "date": np.repeat(np.array(["2025-07-01", "2025-07-02", "2025-07-03"], object), 4)}
    assert R.x_of(4, 8, g) == (1.0, 1.0) and R.x_of(5, 7, g) == (1.25, 0.5)
    assert R.when(4, g, 30) == "2025-07-02 before the open"
    assert R.when(6, g, 30) == "2025-07-02 after 10:30"
    c = ("train", "val", "forecast")
    assert R.slices(8, 12, 0, g, 2, c) == [(0, 8, "train", "TRAIN (fitted)"),
                                           (8, 8, "val", "VALIDATION (early stopping)"),
                                           (8, 12, "forecast", "FORECAST (out of sample)")]
    assert [s[:2] for s in R.slices(8, 12, 1, g, 2, c)] == [(0, 4), (4, 8), (8, 12)]


def test_tables_of_trials_trades_and_forecasts():
    """as_params' types, marks, how_closed's endings and the cross-sectional IC."""
    space = {"edge_bp": {"type": "float"}, "hold_bars": {"type": "categorical",
                                                         "choices": [0, 2, 3, 6]},
             "n": {"type": "int"}}
    p = R.as_params(pd.Series({"edge_bp": 0.8500000000000001, "hold_bars": 3.0, "n": 4.0}), space)
    assert p == {"edge_bp": 0.85, "hold_bars": 3, "n": 4}
    assert type(p["hold_bars"]) is int and type(p["n"]) is int
    sig = pd.DataFrame({"A": [True, False], "B": [False, True]}, index=["t0", "t1"])
    assert R.marks(sig) == {("t0", "A"), ("t1", "B")}

    ts = pd.date_range("2025-07-01 13:30", periods=4, freq="min", tz="UTC")
    open_ = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0]}, index=ts)
    slip = 0.0005
    kw = {"planned": {"Long": {(ts[2], "A")}, "Short": set()},
          "opposite": {"Long": {(ts[3], "A")}, "Short": set()},
          "following": dict(zip(ts[:-1], ts[1:], strict=True)), "last_real": set()}

    def trade(price, at, status="Closed"):
        return {"status": status, "exit_ts": at, "symbol": "A", "direction": "Long",
                "exit_price": price}

    at_open = 102 * (1 - slip)  # a planned sell at ts[2]'s open, after slippage
    assert R.how_closed(trade(at_open, ts[2]), open_, slip, **kw) == "reversal"
    assert R.how_closed(trade(101.5, ts[2]), open_, slip, **kw) == "stop"  # not at the open
    assert R.how_closed(trade(101 * (1 - slip), ts[1]), open_, slip, **kw) == "stop"  # unplanned
    assert R.how_closed(trade(0.0, ts[2], "Open"), open_, slip, **kw) == "open at the end"
    assert R.how_closed(trade(at_open, ts[2]), open_, slip,
                        **{**kw, "last_real": {ts[2]}}) == "session close"
    assert R.how_closed(trade(at_open, ts[2]), open_, slip,
                        **{**kw, "opposite": {"Long": set()}}) == "time exit"

    t = pd.DataFrame({"date": ["d1"] * 6 + ["d2"] * 3, "minute": [600] * 3 + [605] * 3 + [600] * 3,
                      "edge": [1, 2, 3, 3, 2, 1, 1, 2, 3], "real": [10, 20, 30] * 2 + [5] * 3})
    ic = R.xs_ic(t)
    assert ic["d1"] == 0.0  # +1 at 10:00, -1 at 10:05
    assert np.isnan(ic["d2"])  # every name the same outcome: nothing to rank


def test_with_costs_scales_or_overrides_every_cost():
    """2x every cost, or a fixed spread / slippage; the settings passed in stay as they were."""
    s = load_settings()
    k, k2 = s.costs, R.with_costs(s, 2.0).costs
    assert (k2.commission_per_share, k2.min_commission) == (2 * k.commission_per_share,
                                                           2 * k.min_commission)
    assert (k2.spread_bps, k2.slippage_bps) == (2 * k.spread_bps, 2 * k.slippage_bps)
    k3 = R.with_costs(s, spread=10.0, slippage=3.0).costs
    assert (k3.spread_bps, k3.slippage_bps, k3.min_commission) == (10.0, 3.0, k.min_commission)
    assert s.costs is k


# --- charts ------------------------------------------------------------------------------------
def test_timeline_bars_show_one_legend_entry_per_name():
    """bar / span / seg give each name one legend entry; below moves the legend under the plot."""
    fig = go.Figure()
    CH.bar(fig, "row 1", "2025-07-01", "2025-07-03", "#000000", "TRAIN")
    CH.bar(fig, "row 2", "2025-07-02", "2025-07-04", "#000000", "TRAIN")
    CH.span(fig, "row 3", "2025-07-01", "2025-07-01", "#111111", "TEST")
    CH.seg(fig, "row 4", "2025-07-01", "2025-07-05", "#222222", "TEST")
    assert [t.showlegend for t in fig.data] == [True, False, True, False]
    assert fig.data[0].x[0] == 2 * 86_400_000  # bar: from a to the day before b, in ms
    assert fig.data[2].x[0] == 86_400_000  # span: from the start of a to the end of b
    assert CH.below(fig, -0.2).layout.legend.y == -0.2


# --- the walkthrough (1-hour bars, 5 sessions ahead) --------------------------------------------
def _hourly(n_days: int = 24, horizon: int = 14, seed: int = 5) -> tuple:
    """A synthetic 1-hour, multi-day forecast: real inputs, random forecasts (every bar)."""
    from test_darts_forecast import minute_panel

    p = minute_panel(n_days=n_days, seed=seed, start="2025-07-07")
    inp = R.SM["build_inputs"](p, bar_minutes=60, horizon=horizon, options="none", vol_days=5,
                               overnight=True)
    n, m = inp["target"].shape
    rng = np.random.default_rng(seed)
    fc = {"inputs": inp, "edge": rng.normal(size=(n, m)), "p_up": np.full((n, m), np.nan),
          "q10": np.full((n, m), np.nan), "q90": np.full((n, m), np.nan),
          "fit": np.zeros((n, m))}
    return p, inp, fc


def test_horizon_table_keeps_tables_rows_and_reads_prices_off_them():
    """Same rows as table(); prices, % changes and target times read off the grid."""
    _, inp, fc = _hourly()
    t, base = R.horizon_table(fc, earnings={"AAA": ["2025-07-31"]}), R.table(fc)
    pd.testing.assert_frame_equal(t[base.columns], base)
    assert np.allclose(t["price_now"] * (1 + t["r_pct"]), t["price_then"])
    assert np.allclose(t["f_pct"], np.expm1(t["edge_bp"] / 1e4))  # edge x tscale = log change
    assert np.allclose(np.log1p(t["r_pct"]), t["real_bp"] / 1e4)
    assert ((t["target_ts"] - t["ts"]).dt.days >= 2).all() and (t["target_session"]
                                                              - t["session"] == 2).all()
    assert (t["ts"].dt.strftime("%H:%M").isin(["10:30", "11:30", "12:30", "13:30", "14:30",
                                               "15:30", "16:00"])).all()
    # a report on 07-31 (a Thursday): flagged when the window holds 07-31's or 08-01's open
    sf = inp["g"]["sess_first"]
    days = [str(d) for d in inp["g"]["date"][sf]]
    opens = sf[[days.index("2025-07-31"), days.index("2025-08-01")]]
    want = (t["name"] == "AAA") & np.any([(t["bucket"] < b) & (b <= t["target_bucket"])
                                          for b in opens], axis=0)
    assert t["earn"].equals(pd.Series(want, name="earn")) and want.any() and not want.all()


def test_decision_times_close_the_half_day_at_13():
    """The 30-minute last bar closes at 16:00; a half-day's 12:30 bar at 13:00."""
    from test_darts_forecast import minute_panel

    g = R.SM["grid"](minute_panel(n_days=4), 60)                # 2025-07-03 closes at 13:00
    t = R.decision_times(g).strftime("%m-%d %H:%M")
    assert list(t[:7]) == [f"07-01 {h}" for h in ("10:30", "11:30", "12:30", "13:30", "14:30",
                                                  "15:30", "16:00")]
    assert t[14 + 3] == "07-03 13:00" and t[14 + 4] == "07-03 14:30"   # then the empty bars


def test_flip_outcomes_are_signed_sums_and_rebuild_the_path_guesses():
    """All +1 gives back the real outcome and add_baselines' guesses, all -1 their mirror; a draw
    is the signed sum of the row's own bars (one coin per bar, shared by the names), in the row's
    units, and its guesses are rebuilt from the same make-believe path."""
    _, inp, fc = _hourly()
    t = R.add_baselines(R.horizon_table(fc), inp)
    n, H, ts = len(inp["close"]), inp["horizon"], inp["tscale"]
    ones = np.ones((1, n))
    same, mirror = R.flip_outcomes(inp, t, signs=ones), R.flip_outcomes(inp, t, signs=-ones)
    for c, real in (("outcome", "real"), ("b_drift", "b_drift"), ("b_mom", "b_mom"),
                    ("b_rev", "b_rev")):
        np.testing.assert_allclose(same[c][0], t[real], atol=1e-12)
        np.testing.assert_allclose(mirror[c][0], -t[real], atol=1e-12)
    sign = np.random.default_rng(1).choice([-1.0, 1.0], size=(1, n))
    one = R.flip_outcomes(inp, t, signs=sign)
    move = np.r_[np.zeros((1, 2)), np.log(inp["close"][1:] / inp["close"][:-1])]
    cum = np.cumsum(sign[0][:, None] * move, axis=0)
    for i in (len(t) // 2, len(t) - 1):
        s, j = int(t["bucket"].iloc[i]), ["AAA", "BBB"].index(t["name"].iloc[i])
        assert one["outcome"][0, i] == pytest.approx((cum[s + H, j] - cum[s, j]) / ts[s, j])
        assert one["b_mom"][0, i] == pytest.approx((cum[s, j] - cum[s - H, j]) / ts[s, j])
        labels = [(cum[q, j] - cum[q - H, j]) / ts[q - H, j] for q in range(H, s + 1)
                  if inp["valid"][q, j]]                # every outcome completed by the decision
        assert one["b_drift"][0, i] == pytest.approx(np.mean(labels))
    flips = R.flip_outcomes(inp, t, draws=300, seed=0)
    assert flips["outcome"].shape == flips["b_drift"].shape == (300, len(t))
    assert abs(flips["outcome"].mean()) < 0.2 * flips["outcome"].std()
    assert set(R.flip_outcomes(inp, t, draws=3, guesses=False)) == {"outcome"}


def test_baselines_use_only_outcomes_known_at_the_decision():
    """Drift and the base rate change only with labels at or before the decision bar."""
    _, inp, fc = _hourly()
    t = R.add_baselines(R.horizon_table(fc), inp)
    s = int(t["bucket"].iloc[len(t) // 2])
    late = {**inp, "target": inp["target"].copy()}
    late["target"][s + 1:] = 99.0                       # every outcome completed after s changes
    t2 = R.add_baselines(t, late)
    early = (t["bucket"] <= s).to_numpy()
    for c in ("b_drift", "base_rate"):
        np.testing.assert_array_equal(t.loc[early, c], t2.loc[early, c])
        assert not np.allclose(t.loc[~early, c], t2.loc[~early, c])
    j = t["name"].map({"AAA": 0, "BBB": 1}).to_numpy()
    row = int(np.flatnonzero(early)[-1])
    lab = np.where(inp["valid"], inp["target"], np.nan)[: s + 1, j[row]]
    assert t["b_drift"].iloc[row] == pytest.approx(np.nanmean(lab))
    H, b = inp["horizon"], int(t["bucket"].iloc[row])
    mom = np.log(inp["close"][b, j[row]] / inp["close"][b - H, j[row]]) / inp["tscale"][b, j[row]]
    assert t["b_mom"].iloc[row] == pytest.approx(mom) == -t["b_rev"].iloc[row]
    assert (t["b_up"] == 1.0).all()


def test_scores_hand_example():
    """Hit, chance, skill and the per-name IC of a hand example, with the luck band's z; a path
    guess is judged against its copies rebuilt in each make-believe future."""
    t = pd.DataFrame({"name": ["A"] * 4 + ["B"] * 4,
                      "edge": [1, 1, -1, 1, -2, 1, 3, -1], "b_up": 1.0,
                      "real": [2, -1, -3, 4, -1, 2, 1, 1.5]})
    t["b_mom"] = -t["edge"]
    real = t["real"].to_numpy()
    flips = {"outcome": np.stack([real, -real]), "b_mom": np.stack([real, real])}
    sc = R.scores(t, flips, cols=("edge", "b_up", "b_mom"), pairs=(("edge", "b_up"),))
    v = sc[sc["forecast"] == "edge"].set_index("metric")
    # A: calls up 3/4, up 2/4, hit 3/4 -> chance .5; B: calls up 2/4, up 3/4, hit 3/4 -> chance .5
    assert v.loc["up_share", "value"] == 0.625 and v.loc["hit", "value"] == 0.75
    assert v.loc["chance", "value"] == 0.5 and v.loc["skill", "value"] == 0.25
    ic = np.mean([pd.Series(e).corr(pd.Series(r), method="spearman") for e, r in
                  ((t["edge"][:4], t["real"][:4]), (t["edge"][4:], t["real"][4:]))])
    assert v.loc["ic_name", "value"] == pytest.approx(ic)
    # the band: the same forecasts against +real and -real
    null = (v.loc["skill", "value"] + R.scores(t.assign(real=-real), {"outcome": flips[
        "outcome"][:1]}, cols=("edge",), pairs=()).set_index("metric").loc["skill", "value"]) / 2
    assert v.loc["skill", "null_mean"] == pytest.approx(null)
    assert v.loc["skill", "z"] == pytest.approx((0.25 - null) / v.loc["skill", "null_sd"])
    assert v.loc["ic_name", "lo"] == pytest.approx(ic - 1.96 * v.loc["ic_name", "null_sd"])
    up = sc[sc["forecast"] == "b_up"].set_index("metric")
    assert up.loc["skill", "value"] == 0 and np.isnan(up.loc["ic_name", "value"])
    d = sc[sc["forecast"] == "edge - b_up"].set_index("metric")
    assert d.loc["hit", "value"] == pytest.approx(0.75 - 0.625)
    # b_mom's copies: the outcome itself in future 1 (IC +1), against its mirror in future 2 (-1)
    mom = sc[sc["forecast"] == "b_mom"].set_index("metric")
    assert mom.loc["ic_name", "value"] == pytest.approx(-ic)
    assert mom.loc["ic_name", "null_mean"] == pytest.approx(0.0)
    assert mom.loc["ic_name", "null_sd"] == pytest.approx(1.0)
    conf = R.scores(t, flips, cols=("edge",), by="confidence", bins=2, pairs=())
    assert set(conf["group"]) == {"1. 1/2", "2. 2/2"}


def test_target_by_hand_matches_the_pipeline():
    """The 5-day change from the raw 1-minute closes equals the pipeline's target x tscale."""
    p, inp, _ = _hourly()
    tab, (_, ok, _) = CK.target_by_hand(p, inp, n=4, include=[(40, 1)])
    assert ok and len(tab) == 5 and tab["difference"].abs().max() < 1e-9
    assert tab["decision"].iloc[0][-5:] in ("10:30", "11:30", "12:30", "13:30", "14:30",
                                            "15:30", "16:00")


@pytest.mark.parametrize("extra", [{}, {"seasonal": True, "lag_history": True}])
def test_latest_forecast_refits_the_walk_forwards_own_model(extra):
    """A model fit on the panel cut after day D (the latest forecast's fit) gives exactly the
    walk-forward's forecasts of D+1, and the latest forecast from D's last bar is that model's,
    stamped H bars ahead - also with the seasonal input and the lag history (the refit must
    build the same inputs and the same training window as the walk-forward)."""
    pytest.importorskip("darts")
    from test_darts_forecast import SMALL, minute_panel
    from threadpoolctl import threadpool_limits

    p = minute_panel(n_days=36, seed=31, start="2025-07-07")
    prm = {**R.STRAT.params, **SMALL, "bar_minutes": 60, "horizon": 35, "overnight": True,
           "expanding": True, "lookback": 7, "cov_lags": 1, "train_days": 8, "vol_days": 5,
           "retrain_minutes": 390, **extra}            # fits before sessions 29 and 30
    with threadpool_limits(limits=1):
        fc = R.SM["forecast"](p, **R.model_params(prm))
        g = fc["inputs"]["g"]
        T = int(fc["points"][1])     # the 2nd fit: at the 1st, expanding and rolling coincide
        day = str(g["date"][T - 1])                    # the last session before that fit
        first, edge = R.refit_forecast(p, prm, day, fc["inputs"])
        lat = R.latest_forecast(p, prm, asof=day)
        short = R.SM["build_inputs"](R._upto(p, day), **R.input_kw(prm))
        seg = R.SM["predict_segment"](R._fit_last(short, prm), fc["inputs"], "gbm", T - 1, T - 1)
    assert first == T and np.isfinite(edge).all()
    np.testing.assert_allclose(edge, fc["edge"][T:T + 7], rtol=0, atol=1e-12)
    assert CK.refit_check(fc, first, edge, day)[0][1]
    assert not CK.refit_check(fc, first, edge + 1e-6, day)[0][1]
    assert lat.attrs["last_bar"] == T - 1 and lat.attrs["target_bar"] == T - 1 + 35
    assert lat["decision time"].iloc[0] == pd.Timestamp(f"{day} 16:00")
    assert lat["target time"].iloc[0] == pd.Timestamp(f"{day} 16:00") + pd.offsets.BDay(5)
    np.testing.assert_allclose(lat["forecast change %"],
                               100 * np.expm1(seg["edge"][0] * fc["inputs"]["tscale"][T - 1]))


def test_walkthrough_charts_build_on_a_synthetic_forecast():
    """Every walkthrough figure builds, with the traces and shapes its notebook text describes,
    also for a decision or a day near the end of the data."""
    p, inp, fc = _hourly()
    t = R.add_baselines(R.horizon_table(fc), inp).assign(p_up=0.6, earn=False)
    flips = R.flip_outcomes(inp, t, draws=50)
    sc = R.scores(t, flips)
    days = [str(d) for d in inp["g"]["date"][inp["g"]["sess_first"]]]
    fig = CH.split_timeline(days, R.WINDOWS, warmup=5, first_fit=days[8],
                            last_decision=days[-3])
    assert [tr.name for tr in fig.data if tr.showlegend][:3] == ["warm-up",
                                                                  "history before the 1st fit",
                                                                  "graded"]
    assert len(CH.hourly_day(p, inp, "AAA", days[10], days=3).data) == 3
    assert len(CH.hourly_day(p, inp, "AAA", days[-2]).data) == 3      # 2 sessions left, not 5
    pic = CH.target_picture(inp, "AAA", int(t["bucket"].iloc[5]), before=5)
    assert len(pic.data) == 2 and len(pic.layout.shapes) == 3
    assert len(CH.target_picture(inp, "AAA", int(t["bucket"].iloc[-1])).layout.shapes) == 3
    hist = CH.outcome_hist(t)
    assert len(hist.data) == 2 and len(hist.layout.shapes) == 4      # 0 and the mean, per name
    assert [tr.name for tr in CH.price_vs_forecast(t, "AAA").data] == [
        "actual price", "model's predicted price", "'no change' guess"]
    assert len(CH.lag_profile(R.lag_profile(t, "AAA", max_lag=3)).data) == 2
    assert len(CH.return_scatter(t).data) == 3
    dots = CH.dots(sc, "skill")
    assert list(dots.data[1].y[:2]) == ["model (gbm)", "always up"]
    assert len(CH.reliability(R.reliability(t.assign(p_up=np.linspace(0, 1, len(t))),
                                            flips=flips)).data) == 2
    g, n = inp["g"], len(inp["g"]["code"])
    sf = g["sess_first"]
    log = pd.DataFrame({"fit": [0, 1], "retrain_bucket": sf[[12, 16]],
                        "train_from": [days[5]] * 2, "session": [days[12], days[16]]})
    fit = np.full((n, 2), np.nan)
    fit[sf[12]:sf[16]], fit[sf[16]:n - inp["horizon"]] = 0, 1
    tl = CH.fit_timeline({"inputs": inp, "log": log, "fit": fit})
    assert [tr.name for tr in tl.data] == ["learns from: decisions with a known outcome",
                                           "skipped: outcome not known yet",
                                           "forecasts: out of sample"]
    assert len(tl.data[0].y) == 2 and len(tl.layout.shapes) == 1     # the warm-up
    last = pd.Timestamp(f"{days[-1]} 16:00")
    latest = pd.DataFrame({"decision time": [last] * 2, "target time": [last + pd.Timedelta(
        days=7)] * 2, "price now": [100.0, 50.0], "forecast change %": [1.0, -2.0],
        "predicted price": [101.0, 49.0], "P(up)": [0.6, 0.4], "typical 10% change": [-5.0] * 2,
        "typical 90% change": [6.0] * 2}, index=pd.Index(["AAA", "BBB"], name="name"))
    fan = CH.fan(latest, inp, "AAA", sessions=40)                     # more than the data holds
    assert [tr.name for tr in fan.data][-1] == "predicted price" and fan.data[-1].y[0] == 101.0


# --- the walkthrough's verdict, its plain words and its checks ---------------------------------
def _sc(rows: dict, *, sd: float = 0.1, group: str = "all") -> pd.DataFrame:
    """A scores()-like frame: ``{(forecast, metric): (value, z)}``."""
    return pd.DataFrame([{"forecast": f, "group": group, "metric": m, "value": v,
                          "null_mean": v - z * sd, "null_sd": sd, "z": z, "lo": v - 1.96 * sd,
                          "hi": v + 1.96 * sd} for (f, m), (v, z) in rows.items()])


def test_verdict_applies_the_pre_registered_rule_literally():
    """Promising, no evidence, inconclusive and void, and a TEST read that confirms or not."""
    def case(ic: float, z: float, *, clean: bool = True, sd: float = 0.1) -> dict:
        sc = _sc({("edge", "ic_name"): (ic, z), ("edge", "skill"): (0.05 * np.sign(ic), z),
                  ("edge", "hit"): (0.55, 0), ("edge", "chance"): (0.5, 0),
                  ("b_drift", "ic_name"): (-0.2, 0), ("b_mom", "ic_name"): (-0.1, 0),
                  ("b_rev", "ic_name"): (0.1, 0)}, sd=sd)
        by_name = pd.concat([_sc({("edge", "ic_name"): (ic, z)}, group=n) for n in "ABCDE"])
        by_earn = _sc({("edge", "ic_name"): (ic, z)}, group=False)
        null = pd.DataFrame(index=[0])
        null.attrs = {"clean": clean}
        return {"sc": sc, "null": null, "breakdowns": {"name": by_name, "earnings": by_earn}}

    v = R.verdict(**case(0.3, 3.0))
    assert v["status"] == "promising" and all(v["tests"].values())
    assert v["detectable"] == pytest.approx(0.28)
    assert R.verdict(**case(-0.05, -0.5))["status"] == "no evidence of a large edge"
    assert R.verdict(**case(0.05, 0.5, sd=0.1))["status"] == "inconclusive"     # band top 0.25
    assert R.verdict(**case(0.3, 3.0, clean=False))["status"].startswith("void")
    hit, miss = (_sc({("edge", "ic_name"): (x, z)}) for x, z in ((0.2, 2.0), (0.2, 1.0)))
    assert R.verdict(**case(0.3, 3.0), test=hit)["status"] == "promising, confirmed on TEST"
    assert "not confirmed" in R.verdict(**case(0.3, 3.0), test=miss)["status"]
    ledger = pd.DataFrame({"value": [18.0]}, index=["independent weeks per name"])
    words = R.verdict_words(R.verdict(**case(-0.05, -0.5)), case(-0.05, -0.5)["sc"], ledger)
    assert "up to a per-name IC of about +0.15" in words and "18 independent weeks" in words


def test_cadence_words_answer_the_retraining_question():
    """'helped' only when a schedule beats training once beyond luck on the primary number."""
    rng = np.random.default_rng(3)
    real = rng.normal(size=400)
    tab = pd.DataFrame({"name": np.repeat(["A", "B"], 200), "real": real, "once": rng.normal(
        size=400), "weekly": rng.normal(size=400), "every session": real + rng.normal(size=400)})
    flips = {"outcome": real * rng.choice([-1.0, 1.0], size=(300, 400))}
    pairs = [("weekly", "once"), ("every session", "once")]
    words = R.cadence_words(R.scores(tab, flips, cols=("once", "weekly", "every session"),
                                     pairs=pairs))
    assert words.startswith("Retraining helped.")
    calm = R.cadence_words(R.scores(tab, flips, cols=("once", "weekly"), pairs=pairs[:1]))
    assert calm.startswith("Retraining did not help.") and "all within luck" in calm


def test_walkthrough_checks_fail_on_their_corruption():
    """Each check passes on clean data and fails on the corruption it exists for."""
    p, inp, fc = _hourly()
    days = sorted({f"{d:%Y-%m-%d}" for d in p.close.index.tz_convert(p.tz).date})
    factors = pd.DataFrame(1.0, index=pd.to_datetime(days), columns=["AAA", "BBB"])
    kw = {"factors": factors, "last_allowed": days[-1]}
    assert all(ok for _, ok, _ in CK.data_checks(p, inp, **kw))
    hole = p.close.copy()                                    # BBB loses a whole session
    hole.loc[p.close.index.tz_convert(p.tz).strftime("%Y-%m-%d") == days[10], "BBB"] = np.nan
    rows = dict((c, ok) for c, ok, _ in CK.data_checks(replace(p, close=hole), inp, **kw))
    assert not rows["every session holds all 390 one-minute bars for every name"]
    rows = dict((c, ok) for c, ok, _ in CK.data_checks(p, inp, **{**kw, "factors": factors.drop(
        factors.index[-1])}))
    assert not rows["every 1-minute session has its dividend factor for every name"]
    extra = pd.concat([factors, pd.DataFrame(1.0, index=[pd.Timestamp(days[-1]) + pd.Timedelta(
        days=3)], columns=factors.columns)])            # a daily-bar session past the 1-minute end
    rows = dict((c, ok) for c, ok, _ in CK.data_checks(p, inp, **{**kw, "factors": extra}))
    assert not rows["every daily-bar session has its 1-minute bars"]

    sf, n = inp["g"]["sess_first"], len(inp["g"]["code"])
    log = pd.DataFrame({"fit": [0, 1], "retrain_bucket": sf[[21, 22]],
                        "session": [days[21], days[22]], "train_from": [days[21]] * 2,
                        "train_last": [days[20], days[21]], "val_days": [0, 0]})
    edge, fit = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    edge[sf[21]:], fit[sf[21]:sf[22]], fit[sf[22]:] = 0.1, 0, 1
    wf = {"inputs": inp, "log": log, "edge": edge, "fit": fit}
    prm = {"vol_days": 5, "retrain_minutes": 390, "expanding": True, "val_days": 0}
    assert all(ok for _, ok, _ in CK.walk_forward_checks(wf, prm))
    old = fit.copy()
    old[sf[22] + 2] = 0                                      # a forecast credited to an older fit
    assert not CK.walk_forward_checks({**wf, "fit": old}, prm)[0][1]
    peek = log.assign(train_last=[days[21], days[22]])       # a fit trained through its session
    assert not CK.walk_forward_checks({**wf, "log": peek}, prm)[1][1]

    t = R.horizon_table(fc)
    window = (days[0], days[-1])
    assert all(ok for _, ok, _ in CK.window_checks(t, window, expected=t))
    late = CK.window_checks(t, (days[0], days[-3]))
    assert not dict((c, ok) for c, ok, _ in late)[
        "every graded outcome lies inside the window (no later price is read)"]
    assert not CK.window_checks(t.iloc[1:], window, expected=t)[0][1]


def test_truncation_check_catches_a_fit_that_learned_past_its_first_forecast():
    """The leak test's cut sits one bar after a fit: an honest walk-forward passes, and one whose
    fit also learned from a label realized after its first forecast (11:30 for 10:30) fails."""
    p, _, _ = _hourly()

    def walk_forward(leak: int):
        def run(q, params=None, extra=""):
            inp = R.SM["build_inputs"](q, bar_minutes=60, horizon=14, options="none", vol_days=5,
                                       overnight=True)
            sf, n = inp["g"]["sess_first"], len(inp["g"]["code"])
            lab = np.where(inp["valid"], inp["target"], np.nan)
            edge, pts = np.full((n, 2), np.nan), sf[8:]
            for k, T in enumerate(pts):                   # a "fit" = the mean of its labels
                edge[T:pts[k + 1] if k + 1 < len(pts) else n] = np.nanmean(lab[:T + leak], axis=0)
            return {"inputs": inp, "edge": edge, "points": pts}
        return run

    honest, leaky = walk_forward(0), walk_forward(2)
    assert all(ok for _, ok, _ in CK.truncation_check(p, {}, honest(p), honest, cut_sessions=5))
    rows = CK.truncation_check(p, {}, leaky(p), leaky, cut_sessions=5)
    assert rows[0][1] and not rows[1][1] and rows[2][1]


# --- the explorer (darts_forecast_explorer.ipynb): picks, split, path forecasts, search ----------
EX = _load("explorer")


def _paths_panel(n_days: int = 40, seed: int = 3):
    """A synthetic 1-minute panel, no half-day, and its split (warm-up 21, TRAIN, VALIDATION,
    TEST) as ``explorer.load_window`` would cut it."""
    from test_darts_forecast import minute_panel

    p = minute_panel(n_days=n_days, names=("AAA", "BBB", "CCC"), seed=seed, start="2025-07-07")
    sess = sorted({f"{d}" for d in p.close.index.tz_convert(p.tz).date})
    sp = {"sessions": sess, "warmup": (sess[0], sess[20]), "train": (sess[21], sess[29]),
          "validation": (sess[30], sess[35]), "test": (sess[36], sess[-1]),
          "window": (sess[21], sess[-1]), "train_sessions": 9}
    return p, sp


def test_knobs_shrink_everything_for_a_smoke_run():
    cut = EX.months(("2025-08-01", "2026-01-31"), train_months=2, test_months=1)
    assert {k: (str(a.date()), str(b.date())) for k, (a, b) in cut.items()} == {
        "train": ("2025-08-01", "2025-09-30"), "validation": ("2025-10-01", "2025-12-31"),
        "test": ("2026-01-01", "2026-01-31")}
    with pytest.raises(ValueError, match="no VALIDATION"):
        EX.months(("2025-08-01", "2025-10-31"), train_months=2, test_months=1)
    k = EX.knobs(smoke=True, window=("2025-08-01", "2026-01-31"), train_months=2, n_trials=60,
                 granularities=EX.GRANULARITIES, epochs=10)
    assert k["window"] == ("2025-11-01", "2026-01-31") and k["n_trials"] < 60
    assert k["epochs"] < 10 and set(k["granularities"]) < set(EX.GRANULARITIES)
    full = EX.knobs(smoke=False, window=("a", "b"))
    assert full["window"] == ("a", "b") and full["min_windows"] == EX.MIN_WINDOWS
    for ladder in (EX.GRANULARITIES, EX.INPUT_LENGTHS, EX.PRED_LENGTHS):    # no small steps
        assert all(b / a >= 1.6 for a, b in zip(ladder, ladder[1:], strict=False))


def test_path_target_is_each_bars_move_in_its_clock_times_typical_size():
    """y = close-to-close log return (the open's bar holds the gap) / the RMS of the same clock
    position over the previous sessions, clipped; nothing after a bar moves its y or inputs."""
    p, _ = _paths_panel()
    pin = EX.path_inputs(p, 30)
    g, y, u = pin["g"], pin["y"], pin["u"]
    close = pin["close"]
    b = g["sess_first"][25] + 3                             # session 25, 4th bar (pos 3)
    r = np.log(close[b, 0] / close[b - 1, 0])
    past = [g["sess_first"][25 - k] + 3 for k in range(1, 21)]
    rms = np.sqrt(np.mean([np.log(close[q, 0] / close[q - 1, 0]) ** 2 for q in past]))
    assert u[b, 0] == pytest.approx(rms, rel=1e-9) and y[b, 0] == pytest.approx(r / rms)
    o = g["sess_first"][25]                                 # the open's bar: the gap is in it
    assert y[o, 0] == pytest.approx(np.log(close[o, 0] / close[o - 1, 0]) / u[o, 0])
    assert np.nanmax(np.abs(y)) <= EX.CLIP and np.isnan(y[: g["sess_first"][3]]).all()
    from test_darts_forecast import _bump_after

    later = EX.path_inputs(_bump_after(p, g["end"][b] + 1), 30)
    np.testing.assert_array_equal(y[: b + 1], later["y"][: b + 1])
    assert all(np.array_equal(pin["past"][n][: b + 1], later["past"][n][: b + 1], equal_nan=True)
               for n in pin["names"])


def test_path_target_reads_no_later_bar_even_where_its_floor_binds():
    """U-shaped intraday volatility (quiet middays, where the floor of the typical-move scale
    binds) and every bar after mid-panel replaced by louder random bars: each target value and
    scale before the cut stays bit-identical (a floor taken over the whole panel failed this)."""
    from test_darts_forecast import minute_panel

    p = minute_panel(n_days=50, names=("AAA", "BBB"), seed=9, start="2025-07-07")
    local = p.close.index.tz_convert(p.tz)
    minute = np.asarray(local.hour * 60 + local.minute) - 570
    shape = 0.05 + 2.0 * ((minute - 195) / 195.0) ** 2        # loud open and close, quiet midday
    logc = np.log(p.close.to_numpy())
    c = np.exp(logc[0] + np.cumsum(np.diff(logc, axis=0, prepend=logc[:1]) * shape[:, None],
                                   axis=0))
    o = np.vstack([c[:1], c[:-1]])
    frame = lambda x: pd.DataFrame(x, index=p.close.index, columns=p.close.columns)  # noqa: E731
    q = replace(p, close=frame(c), open_=frame(o), high=frame(np.maximum(o, c) * 1.0001),
                low=frame(np.minimum(o, c) * 0.9999))
    cut = len(q.close) // 2
    a = EX.path_inputs(q, 30)
    b = EX.path_inputs(CK.scrambled(q, np.arange(len(q.close)) >= cut, 1), 30)
    k = int(np.searchsorted(a["g"]["end"], cut))              # bars [:k] end before the cut
    np.testing.assert_array_equal(a["y"][:k], b["y"][:k])
    np.testing.assert_array_equal(a["u"][:k], b["u"][:k])
    assert not np.array_equal(a["y"][k:], b["y"][k:], equal_nan=True)


@pytest.mark.parametrize("g", [1, 2, 5, 10, 30, 60])
def test_decisions_sit_at_the_same_clock_times_on_every_grid(g):
    p, sp = _paths_panel()
    pin = EX.path_inputs(p, g)
    s, j = EX.decisions(pin, sp["validation"], 3)
    t = pin["times"]
    assert set(t[s].strftime("%H:%M")) == set(EX.DECISION_TIMES)
    assert (t[s].strftime("%Y-%m-%d") >= sp["validation"][0]).all()
    assert (t[s + 3].strftime("%Y-%m-%d") <= sp["validation"][1]).all()        # inside its part
    act = EX.actual_paths(pin, s, j, 3)
    assert np.isfinite(act).all() and act.shape == (len(s), 3)
    assert (np.bincount(j) == len(s) // 3).all()                               # every stock


def test_feasibility_counts_training_windows_and_independent_paths():
    """Windows: the labels between the first trained bar and the held-out sessions (at least 5,
    more when 5 independent paths need more), inputs reaching back into the warm-up; paths:
    validation bars / prediction length."""
    sess = [f"d{i:03d}" for i in range(147)]
    sp = {"sessions": sess, "train": (sess[21], sess[62]), "validation": (sess[63], sess[126])}
    assert EX.held_out_sessions(5, 20) == EX.ES_DAYS                  # 100 bars < 5 sessions
    assert EX.held_out_sessions(60, 20) == math.ceil(EX.ES_PATHS * 20 / 7)   # 5 paths: 15
    ok, why = EX.feasible(60, 100, 10, sp)                  # 7 bars a session
    lo, hi = 21 * 7, (63 - EX.held_out_sessions(60, 10)) * 7
    assert EX.windows(hi, lo, 100, 10) == hi - 10 - max(100, lo - 9) + 1
    assert ok == (EX.windows(hi, lo, 100, 10) >= EX.MIN_WINDOWS) and "windows" in why
    assert not EX.feasible(60, 1000, 10, sp)[0]             # 143 sessions of input: impossible
    assert not EX.feasible(60, 50, 50, sp)[0]               # the held-out paths eat TRAIN
    assert EX.feasible(5, 1000, 100, sp)[0]
    short = {**sp, "validation": (sess[63], sess[67])}      # 5 validation sessions
    bad, why = EX.feasible(30, 50, 50, short)
    assert not bad and f"{5 * 13 / 50:.0f} independent" in why
    t = EX.feasibility_table(sp, (5, 60), (50, 1000), (10, 100))
    assert t.loc[(5, 1000), "predict 100"] == "ok" and t.loc[(60, 1000), "predict 10"] != "ok"


def test_path_r2_against_no_change_by_hand():
    """R² per stock and lead = 1 - sum (pred - actual)^2 / sum actual^2 of the CUMULATIVE paths,
    averaged over leads, then equally over stocks; a perfect path scores 1, "no change" 0."""
    rng = np.random.default_rng(0)
    act = rng.normal(size=(40, 3)).astype(np.float32)
    j = np.repeat([0, 1], 20)
    s = np.arange(40) * 5
    names = ["AAA", "BBB"]
    assert EX.score_paths(act, act, s, j, names)["r2"] == pytest.approx(1.0)
    assert EX.score_paths(0 * act, act, s, j, names)["r2"] == pytest.approx(0.0)
    half = EX.score_paths(0.5 * act, act, s, j, names)                 # 1 - 0.25 = 0.75
    assert half["r2"] == pytest.approx(0.75) and half["share_pos"] == 1.0
    pred = rng.normal(size=(40, 3)).astype(np.float32)
    pc, ac = np.cumsum(pred, 1), np.cumsum(act, 1)
    want = np.mean([np.mean(1 - ((pc[j == k] - ac[j == k]) ** 2).sum(0)
                            / (ac[j == k] ** 2).sum(0)) for k in (0, 1)])
    corr = np.mean([np.corrcoef(pc[j == k, h], ac[j == k, h])[0, 1] for k in (0, 1)
                    for h in range(3)])
    out = EX.score_paths(pred, act, s, j, names, draws=100)
    assert out["r2"] == pytest.approx(want, rel=1e-6) and out["corr"] == pytest.approx(corr)
    assert abs(out["null_mean"]) < 3 * out["null_sd"] / 10 and np.isfinite(out["z"])  # luck: ~0
    assert out["r2_h"].shape == (3,) and list(out["r2_by_name"].index) == names
    flat = EX.score_paths(0 * act, act, s, j, names, draws=50)     # a flat path has no direction
    assert np.isnan(flat["corr"]) and "z" not in flat


def test_calibration_scales_a_path_by_what_proved_right():
    rng = np.random.default_rng(1)
    act = rng.normal(size=(200, 5))
    assert EX.calibration(2 * act, act) == pytest.approx(0.5)          # twice too loud
    assert EX.calibration(-act, act) == 0.0                            # pointing the wrong way
    assert EX.calibration(0 * act, act) == 0.0 and EX.calibration(0.5 * act, act) == 1.0
    assert np.array_equal(EX.scaled(act, 0.0), 0 * act.astype(np.float32))


def test_ar1_baseline_reads_the_bounce_inside_sessions():
    rng = np.random.default_rng(2)
    n, rho = 4000, -0.3
    y = np.zeros((n, 1))
    for t in range(1, n):
        y[t] = rho * y[t - 1] + rng.normal()
    code = np.repeat(np.arange(n // 100), 100)
    pin = {"y": y, "g": {"code": code}}
    est = EX.ar1(pin, 0, n)
    assert est[0] == pytest.approx(rho, abs=0.05)
    path = EX.ar1_paths(pin, np.array([10]), np.array([0]), est, 3)
    assert np.allclose(path[0], y[10, 0] * est[0] ** np.arange(1, 4))


def test_search_paths_measures_distinct_feasible_cells_and_finds_a_planted_best(monkeypatch):
    """Every trial measures a new configuration the data supports; infeasible ones are never
    measured; with trials to spare every feasible one is; a planted best is found."""
    _, sp = _paths_panel()
    grans, ins, outs = (1, 5, 60), (50, 200, 1000), (10, 50)
    best_cell = (5, 200, 10)

    def fake_run(panel, sp_, cell, **kw):
        a = sum(abs(np.log(x) - np.log(y)) for x, y in zip(cell, best_cell, strict=True))
        r2 = 0.01 - 0.002 * a
        sc = {"r2": r2, "r2_h": np.array([r2]), "corr_h": np.array([0.0]), "share_pos": 0.5,
              "worst": r2, "corr": 0.0, "outcomes": 10.0, "n_paths": 100}
        return {"cell": tuple(cell), "score": sc, "beta": 0.5,
                "score_raw": {**sc, "z": r2 * 100, "null_mean": 0.0, "null_sd": 0.01},
                "ar1": {"r2": 0.0}, "fit": {"stride": 1, "seconds": 0.0, "epochs": 1}}

    monkeypatch.setattr(EX, "run_path", fake_run)
    kw = dict(kind="tide", grans=grans, inputs=ins, outputs=outs, epochs=1, min_windows=50,
              min_outcomes=3, verbose=False)
    ok = {(g, L, H) for g in grans for L in ins for H in outs
          if EX.feasible(g, L, H, sp, min_windows=50, min_outcomes=3)[0]}
    assert 0 < len(ok) < len(grans) * len(ins) * len(outs)
    res, study = EX.search_paths(None, sp, n_trials=6, **kw)
    cells = set(zip(res["granularity"], res["input"], res["prediction"], strict=True))
    assert len(res) == 6 and len(cells) == 6 and cells <= ok
    full, _ = EX.search_paths(None, sp, n_trials=500, **kw)
    assert len(full) == len(ok) and list(full["trial"]) == list(range(1, len(ok) + 1))
    top = EX.best(full).iloc[0]
    assert (top["granularity"], top["input"], top["prediction"]) == best_cell


def test_fit_and_predict_paths_end_to_end_without_reading_the_future():
    """A tiny DLinear on the synthetic panel: one model for every stock, paths of the right
    shape; nothing after a cut moves a path predicted before it, and the fit itself reads no bar
    of the graded part (refit on a panel changed there: the same model)."""
    pytest.importorskip("darts")
    from test_darts_forecast import _bump_after

    p, sp = _paths_panel()
    pin = EX.path_inputs(p, 30)
    sf = pin["g"]["sess_first"]
    lo, hi = int(sf[21]), int(sf[30])
    est, info = EX.fit_paths(pin, "dlinear", 26, 5, lo, hi, epochs=2)
    assert info["windows"] > 0 and info["held_out"][1] < sp["validation"][0]
    s, j = EX.decisions(pin, sp["validation"], 5)
    pred = EX.predict_paths(est, pin, s, j, 26, 5)
    assert pred.shape == (len(s), 5) and np.isfinite(pred).all()
    cut = int(s[len(s) // 2])
    alt = EX.path_inputs(_bump_after(p, pin["g"]["end"][cut] + 1), 30)
    again = EX.predict_paths(est, alt, s, j, 26, 5)
    before = s <= cut
    np.testing.assert_array_equal(pred[before], again[before])
    assert not np.array_equal(pred[~before], again[~before])
    est2, _ = EX.fit_paths(alt, "dlinear", 26, 5, lo, hi, epochs=2)   # graded part changed only
    np.testing.assert_array_equal(EX.predict_paths(est2, pin, s, j, 26, 5), pred)


def test_explorer_path_charts_build():
    """Every explorer figure builds with the traces its notebook text describes."""
    p, sp = _paths_panel()
    pin = EX.path_inputs(p, 30)
    rng = np.random.default_rng(4)
    rows = []
    for i, (g, L, H) in enumerate([(5, 50, 10), (5, 200, 10), (30, 50, 20), (60, 50, 10)]):
        r2 = rng.normal(0, 0.01)
        rows.append({"trial": i + 1, "granularity": g, "input": L, "prediction": H, "r2": r2,
                     "z": r2 * 100, "corr": 0.0, "r2_h": rng.normal(0, 0.01, H),
                     "corr_h": rng.normal(0, 0.05, H)})
    res = pd.DataFrame(rows)
    assert [tr.name for tr in CH.lead_curves(res).data][-1].startswith("60 min")
    assert len(CH.lead_curves(res, metric="corr_h").data) == len(res) + 3
    params = {"granularity": ("bar size", (5, 30, 60)), "input": ("input", (50, 200)),
              "prediction": ("predict", (10, 20))}
    eff = CH.param_effects(res, params, metric="z")
    names = [tr.name for tr in eff.data]
    assert names.count("luck, one config (95%)") == 3 and names.count("best") == 3
    bar = [tr for tr in eff.data if tr.name == "luck, best of k"][0]
    assert np.allclose(bar.y, [EX.best_of_bar(2), EX.best_of_bar(1), EX.best_of_bar(1)])
    assert "luck, best of k" not in [tr.name for tr in CH.param_effects(res, params).data]
    maps = CH.granularity_maps(res, (5, 30, 60), (50, 200), (10, 20))
    assert len(maps.data) == 3 and len(maps.layout.shapes) == 1
    assert len(CH.importance_bars(pd.Series({"a": 0.7, "b": 0.3})).data) == 1
    assert len(CH.search_history(res).data) == 2
    b = pin["g"]["sess_first"][31] + 5
    ex_paths = [{"label": "x", "granularity": 30, "beta": 0.5, "times": pin["times"][b:b + 4],
                 "price": np.full(4, 100.0), "price_raw": np.full(4, 101.0)}]
    fig = CH.path_picture(p, ex_paths, "AAA", pin["times"][b])
    assert [tr.name for tr in fig.data][:2] == ["actual price (1-minute closes)", "x (scale 0.50)"]
    assert len(fig.data) == 3                                   # the unscaled path, dotted
    split = CH.explorer_split(sp, [(sp["validation"][0], "fit"), (sp["test"][0], "refit")])
    assert len(split.data) == 4 and len(split.layout.shapes) == 2
    c = pd.DataFrame({"market_cap": [5e9, 50e9, 1e9], "vol": [0.3, 0.6, 0.9], "ok": True},
                     index=["A", "B", "C"])
    picks = pd.DataFrame({"symbol": ["B", "C"], "cap_group": ["large", "small"],
                          "speed": ["slow", "fast"], "market_cap": [50e9, 1e9], "vol": [0.6, 0.9]})
    assert len(CH.universe_map(c, picks).data) == 3


def test_coverage_counts_missing_sessions_traded_minutes_and_recent_months():
    days = pd.bdate_range("2025-08-01", periods=10)
    rows = [{"symbol": s, "day": d, "n": 390, "traded": 390 if s != "THIN" else 195}
            for s in ("AAA", "THIN", "GAP") for d in days if not (s == "GAP" and d == days[3])]
    recent = pd.bdate_range("2026-04-01", "2026-09-30")
    r = [{"symbol": s, "day": d, "n": 390} for s in ("AAA", "THIN", "GAP") for d in recent
         if not (s == "GAP" and d.month == 9)]
    c = EX._coverage(pd.DataFrame(rows), pd.DataFrame(r), 6, (days[0], days[-1]), frozenset())
    assert c.loc["GAP", "missing"] == 1 and c.loc["AAA", "missing"] == 0
    assert c.loc["THIN", "traded"] == pytest.approx(0.5) and c.loc["AAA", "traded"] == 1.0
    assert c.loc["AAA", "recent"] == 1.0 and 0.75 < c.loc["GAP", "recent"] < 0.9
    assert c.attrs["sessions"] == 10 and c.attrs["recent_to"] == "2026-09-30"
    # a calendar, not a vote: many names that start mid-window must not hide a late start
    late = [{"symbol": f"N{k}", "day": d, "n": 390, "traded": 390} for k in range(5)
            for d in days[5:]]
    c = EX._coverage(pd.DataFrame(rows + late), pd.DataFrame(r), 6, (days[0], days[-1]),
                     frozenset())
    assert c.attrs["sessions"] == 10 and (c.loc[[f"N{k}" for k in range(5)], "missing"] == 5).all()
    # an exchange holiday is no session; a weekday with no data at all is none either
    hol = EX._coverage(pd.DataFrame(rows), pd.DataFrame(r), 6, (days[0], days[-1]),
                       frozenset({days[3].strftime("%Y-%m-%d")}))
    assert hol.attrs["sessions"] == 9 and hol.loc["GAP", "missing"] == 0


def test_pick_takes_the_nearest_percentile_and_spreads_the_sectors():
    """Slow = nearest the 10th volatility percentile of its cap group, fast = the 90th; a sector
    already picked is skipped while a nearby name of a new sector exists."""
    rng = np.random.default_rng(0)
    n = 60
    c = pd.DataFrame({"cap_group": np.repeat(["large", "mid", "small"], n // 3),
                      "vol": rng.uniform(0.1, 1.0, n),
                      "sector": rng.choice(list("ABCDEFGHIJKL"), n),
                      "name": "x", "market_cap": 1.0, "traded": 1.0, "ok": True},
                     index=[f"S{i}" for i in range(n)])
    c.loc["S0", "ok"] = False                                  # a failed name is never picked
    c["vol_pct"] = c[c["ok"]].groupby("cap_group")["vol"].rank(pct=True)
    p = EX.pick(c)
    assert len(p) == 12 and p["symbol"].is_unique and "S0" not in set(p["symbol"])
    assert list(p.groupby(["cap_group", "speed"], sort=False).size()) == [2] * 6
    slow, fast = p[p["speed"] == "slow"], p[p["speed"] == "fast"]
    # 20 names a group: the 8 nearest reach about 0.2 from the target percentile
    assert slow["vol_pct"].max() <= 0.3 and fast["vol_pct"].min() >= 0.6
    assert slow["vol_pct"].mean() < 0.2 and fast["vol_pct"].mean() > 0.8
    assert p["sector"].nunique() >= 10                         # 12 sectors to choose from
    one = EX.pick(c, {("large", "slow"): 1}, near=1)           # near=1: the nearest, whatever
    x = c[c["ok"] & (c["cap_group"] == "large")]
    assert one["symbol"].iloc[0] == (x["vol_pct"] - 0.10).abs().idxmin()


def test_load_window_guards_the_locked_months_and_missing_names(monkeypatch):
    """A window past TUNE needs read_locked; a requested name without 1-minute data is an error,
    not a silent drop; the warm-up is exactly the inputs'; a zero market cap fails rule 4."""
    from test_darts_forecast import minute_panel

    p = minute_panel(n_days=88, names=("AAA", "BBB"), seed=2, start="2025-06-02")
    monkeypatch.setattr(EX.rs, "load", lambda *a, **k: p)
    window = ("2025-07-01", "2025-09-30")               # TRAIN Jul, VALIDATION Aug, TEST Sep
    with pytest.raises(ValueError, match="locked"):
        EX.load_window(None, None, ["AAA"], ("2026-04-01", "2026-09-30"))
    with pytest.raises(ValueError, match="CCC"):
        EX.load_window(None, None, ["AAA", "CCC"], window, train_months=1, test_months=1)
    panel, sp = EX.load_window(None, None, ["AAA", "BBB"], window, train_months=1, test_months=1)
    assert sp["sessions"].index(sp["train"][0]) == max(21, EX.SETTINGS["vol_days"] + 1)
    assert (sp["train"][0], sp["validation"][0], sp["test"][0]) == (
        "2025-07-01", "2025-08-01", "2025-09-01")
    rule = EX.RULES["listed on a US exchange, with a market cap"]
    assert rule(pd.DataFrame({"market_cap": [0.0, np.nan, 5e9]})).tolist() == [False, False, True]


# --- the intraday study (darts_forecast_intraday.ipynb): no night inside, one truth, causal -------
IX = _load("intraday")


def _with(p: Panel, **frames) -> Panel:
    """A copy of ``p`` with some frames replaced by arrays."""
    f = lambda x: pd.DataFrame(x, index=p.close.index, columns=p.close.columns)  # noqa: E731
    return replace(p, **{k: f(v) for k, v in frames.items()})


def test_intraday_target_and_truth_start_inside_the_session():
    """A session's first bar moves from its own open (the night is never a target), every later
    bar from the previous close; the graded move runs from the FILL (the first trade after the
    decision) to the last close before each checkpoint (carried over a minute without a trade);
    volume-0 placeholder minutes are no trade, and a decision with no trade in its next 10
    minutes is skipped."""
    p, _ = _paths_panel()
    v = p.volume.to_numpy().copy()
    s, k = 25, 0
    v[s * 390 + 30:s * 390 + 40, k] = 0                      # 10:00-10:09: placeholders only
    v[s * 390 + 60, k] = 0                                   # 10:30: one placeholder
    v[s * 390 + 69, k] = 0                                   # 10:39: the 10-minute checkpoint's
    pp = IX.prepared(_with(p, volume=v))
    m, oc = pp["m"], pp["oc"]
    op, C = p.open_.to_numpy().reshape(-1, 390, 3), p.close.to_numpy().reshape(-1, 390, 3)
    assert np.isnan(m["close"][s, 30:40, k]).all() and np.isfinite(m["close"][s, 40, k])
    gi = IX.grid_inputs(pp, 5)
    b = gi["sf"][s]                                          # 09:30-09:34, the session's first
    assert gi["y"][b, k] * gi["u"][b, k] == pytest.approx(np.log(C[s, 4, k] / op[s, 0, k]))
    assert gi["y"][b + 1, k] * gi["u"][b + 1, k] == pytest.approx(np.log(C[s, 9, k] / C[s, 4, k]))
    row = lambda t: int(np.flatnonzero((oc["s"] == s) & (oc["t"] == t) & (oc["k"] == k))[0])  # noqa: E731
    assert not oc["ok"][row(30)] and not oc["usable"][row(30)]   # no trade 10:00-10:09: no fill
    r60 = row(60)                                            # fills at 10:31's open instead
    assert oc["A"][r60, 0] == pytest.approx(np.log(C[s, 68, k] / op[s, 61, k]))   # carried
    assert oc["A"][r60, IX.CHECKPOINTS.index(60)] == pytest.approx(np.log(C[s, 119, k]
                                                                         / op[s, 61, k]))
    assert np.isnan(oc["A"][row(360), IX.CHECKPOINTS.index(60)])   # 15:30 + 60 min: past 16:00
    assert all(ok for _, ok, _ in IX.gap_invariance(p, 5))


def test_ridge_reads_the_gap_only_once_the_day_has_traded():
    """A stock whose first trade of the day comes at 10:05: its 10:00 decision cannot know the
    gap (that first trade would BE its fill), its 10:30 decision can."""
    p, _ = _paths_panel(seed=11)
    v = p.volume.to_numpy().copy()
    s, k = 27, 1
    v[s * 390:s * 390 + 35, k] = 0                           # no trade 09:30-10:04
    pp = IX.prepared(_with(p, volume=v))
    X, names = IX.ridge_features(pp)
    oc, j = pp["oc"], names.index("gap")
    row = lambda t: int(np.flatnonzero((oc["s"] == s) & (oc["t"] == t) & (oc["k"] == k))[0])  # noqa: E731
    assert X[row(30), j] == 0.0 and X[row(60), j] != 0.0


def test_intraday_scales_are_causal_and_shared_by_every_bar_size():
    """The same numbers laid out row-major or column-major give bit-identical scales (a sum over
    another layout rounds differently in the last bit unless the arrays are laid out one way:
    the real panel and its scrambled copy differ so); sigma = RMS 1-minute move over the previous
    20 sessions; the time-of-day profile averages 1 over a session; a 10-minute bar's unit is the
    two 5-minute ones' combined; every scale and ridge input before a cut is bit-identical when
    the bars after it are replaced."""
    p, _ = _paths_panel()
    ohlcv = ("open_", "high", "low", "close", "volume")
    prep = IX.prepare(p)                                     # the same numbers, both layouts
    as_c = _with(prep, **{f: np.ascontiguousarray(getattr(prep, f).to_numpy()) for f in ohlcv})
    as_f = _with(prep, **{f: np.asfortranarray(getattr(prep, f).to_numpy()) for f in ohlcv})
    assert as_c.close.to_numpy().flags.c_contiguous and as_f.close.to_numpy().flags.f_contiguous
    sc_c, sc_f = (IX.scales(IX.minute_arrays(x)) for x in (as_c, as_f))
    for key in ("sigma", "prof2", "cum"):
        np.testing.assert_array_equal(sc_c[key], sc_f[key])
    pp = IX.prepared(p)
    m, sc = pp["m"], pp["sc"]
    s = 30
    day = np.nanmean(m["r1"][s - 20:s] ** 2, axis=1)
    assert sc["sigma"][s] == pytest.approx(np.sqrt(day.mean(axis=0)), rel=1e-9)
    assert np.allclose(sc["prof2"][21:].mean(axis=1), 1.0)
    u5, u10 = IX.grid_inputs(pp, 5)["u"], IX.grid_inputs(pp, 10)["u"]
    assert np.allclose(u10[s * 39 + 4] ** 2, u5[s * 78 + 8] ** 2 + u5[s * 78 + 9] ** 2)
    cut = 33 * 390 + 200
    pa = IX.prepared(CK.scrambled(p, np.arange(len(p.close)) >= cut, 1))
    for key in ("sigma", "prof2", "cum"):
        np.testing.assert_array_equal(sc[key][:34], pa["sc"][key][:34])
    early = (pp["oc"]["s"] < 33) | ((pp["oc"]["s"] == 33) & (pp["oc"]["t"] <= 180))
    np.testing.assert_array_equal(IX.ridge_features(pp)[0][early],
                                  IX.ridge_features(pa)[0][early])


def test_stock_day_series_hold_yesterdays_end_then_today(monkeypatch):
    """Each series: the previous session's last L-1 bars and the whole session, on consecutive
    global bars, the 'today' flag 1 exactly on the session's bars; the static inputs carry the
    calendar days since the previous session (3 after a weekend); half-days and the day after one
    are left out, and the check says so only when it is true."""
    pytest.importorskip("darts")
    p, sp = _paths_panel()
    assert all(ok for _, ok, _ in IX.series_checks(p, sp, (5, 60, 30)))
    pp = IX.prepared(p)
    gi = IX.grid_inputs(pp, 5)
    dates = pp["m"]["dates"]
    mon = next(i for i, d in enumerate(dates) if i > 21 and pd.Timestamp(str(d)).weekday() == 0)
    ser, fut = IX.day_series(gi, 1, mon, 12), IX.future(gi, mon, 12)
    assert len(ser) == 11 + 78 and ser.static_covariates["days_since"].iloc[0] == 3 / 4
    assert ser.static_covariates["is_BBB"].iloc[0] == 1.0
    flag = fut.values()[:, IX.CLOCK.index("today")]
    assert flag[:11].sum() == 0 and flag[11:].all()
    assert len(IX.day_series(gi, 1, mon, 12, end=gi["sf"][mon] + 5)) == 11 + 6
    d = ["2025-11-26", "2025-11-28", "2025-12-01", "2025-12-02", "2025-12-24", "2025-12-26"]
    assert IX.usable(d).tolist() == [False, False, False, True, False, False]
    q, _ = _paths_panel(seed=13)                             # a half-day the module fails to drop
    monkeypatch.setitem(IX.SM, "HALF_DAYS", frozenset({sp["train"][0]}))
    monkeypatch.setattr(IX, "usable", lambda dd: np.r_[False, np.ones(len(dd) - 1, bool)])
    rows = IX.series_checks(q, sp, (5, 60, 30))
    assert not rows[3][1] and sp["train"][0] in rows[3][2]


def test_checkpoints_read_bar_closes_with_each_bars_unit():
    """Predicted bar moves become log moves with each predicted bar's OWN unit (from the bar after
    the decision); a checkpoint on a bar close reads it, one between two closes (10 minutes on
    3-minute bars) the straight line between."""
    u = 0.01 * (1 + np.arange(200) / 10)                      # every bar its own unit
    gi = {"g": 3, "sf": np.array([0]), "u": u[:, None]}
    pred = np.arange(1, 21, dtype=float)[None, :]                 # bars 1..20, 3 minutes each
    b = 30 // 3 - 1                                               # the bar closing at 10:00
    P = np.r_[0.0, np.cumsum(pred[0] * u[b + 1:b + 21])]
    out = IX.checkpoint_moves(pred, gi, np.array([0]), np.array([30]), np.array([0]), 60)
    assert out[0, IX.CHECKPOINTS.index(30)] == pytest.approx(P[10])
    assert out[0, 0] == pytest.approx(P[3] + (10 / 3 - 3) * (P[4] - P[3]))
    assert out[0, IX.CHECKPOINTS.index(60)] == pytest.approx(P[20])
    assert np.isnan(out[0, IX.CHECKPOINTS.index(90)])            # past the 60-minute span


def test_intraday_grade_scale_and_luck_by_hand():
    """R² per stock and checkpoint against 'no change' (NaN columns not graded); the scale of a
    month comes from the held-out week and the earlier months only; the luck band is centred on
    exactly 0 (each coin and its mirror image); the interval resamples whole sessions, so it is
    wider than one resampling single decisions when a session's decisions share their outcome."""
    rng = np.random.default_rng(0)
    n = 120
    act = np.full((n, len(IX.CHECKPOINTS)), np.nan)
    act[:, :4] = rng.normal(size=(n, 4))
    k = np.repeat([0, 1], n // 2)
    names = ["AAA", "BBB"]
    assert IX.grade(act, act, k, names)["r2"] == pytest.approx(1.0)
    assert IX.grade(0.5 * act, act, k, names)["r2"] == pytest.approx(0.75)
    assert IX.grade(0 * act, act, k, names)["r2"] == pytest.approx(0.0)
    month = np.repeat(["2025-10", "2025-11"], n // 2)
    held = rng.normal(size=(30, 4))
    beta, by = IX.chained_beta(2 * held, held, act[:, :4], act[:, :4], month)
    assert by["2025-10"] == pytest.approx(0.5)                   # the held-out week alone
    assert by["2025-11"] == pytest.approx(IX.calibration(np.r_[2 * held, act[:60, :4]],
                                                         np.r_[held, act[:60, :4]]))
    assert (beta[:60] == by["2025-10"]).all() and (beta[60:] == by["2025-11"]).all()
    s = np.tile(np.arange(10).repeat(6), 2)                       # 10 sessions, 6 decisions each
    B = rng.normal(size=(n, 36))
    key = s[:, None] * 39 + np.arange(36)
    U = np.ones((n, len(IX.CHECKPOINTS)))
    flips = np.where(np.isfinite(act), np.cumsum(B, axis=1)[:, np.array(IX.CHECKPOINTS) // 10 - 1],
                     np.nan)
    out = IX.grade(flips + rng.normal(size=flips.shape), flips, k, names, key=key, B=B, U=U,
                   draws=40)
    assert abs(out["null_mean"]) < 1e-12 and out["z"] > 3
    day = rng.normal(size=(10, 1)).repeat(6, axis=0)              # a shared part per session
    a = np.where(np.isfinite(act), np.tile(day, (2, 1)) * 3 + rng.normal(size=act.shape), np.nan)
    pred = 0.5 * np.where(np.isfinite(act), np.tile(day, (2, 1)) * 3, np.nan)
    lo, hi = IX.bootstrap(pred, a, s, k, n=400)
    assert lo <= IX.grade(pred, a, k, names)["r2"] <= hi
    rows = np.random.default_rng(1).integers(0, n, size=(400, n))  # single decisions instead
    r2 = [IX.grade(pred[r], a[r], k[r], names)["r2"] for r in rows]
    assert hi - lo > 1.5 * (np.percentile(r2, 95) - np.percentile(r2, 5))
    assert IX.mde(0.01) == pytest.approx(((1.96 + 0.84) * 0.01) ** 2, rel=1e-3)
    assert IX.mde(0.01, 100) > IX.mde(0.01)


def test_planted_momentum_leaves_the_gaps_and_its_oracle_finds_it():
    """The planted 30-minute momentum moves no overnight gap, and its known part alone grades
    well above 0 at the size planted."""
    p, sp = _paths_panel()
    pl, info = IX.planted(p, sp, 0.3)
    g0, g1 = (IX.day_open_gap(IX.prepared(x)["m"])[1] for x in (p, pl))
    np.testing.assert_allclose(g0[1:], g1[1:], rtol=1e-9)
    assert all(v > 0 for v in info["kappa"].values())
    assert 0.03 < IX.oracle(pl, sp, info, 30) < 0.3


def _atlas_panel(seed: int, *, link: float = 0.0, day_vol: float = 0.0, n_days: int = 60):
    """A synthetic panel for the map: each stock-day's volatility drawn log-normal with spread
    ``day_vol``, and the last half-hour continuing the one before it by ``link``."""
    from test_darts_forecast import minute_panel

    p = minute_panel(n_days=n_days, names=("AAA", "BBB", "CCC", "DDD"), seed=seed,
                     start="2025-07-07")
    rng = np.random.default_rng(seed + 100)
    r = np.diff(np.log(p.close.to_numpy()), axis=0, prepend=np.log(p.close.to_numpy()[:1]))
    r = r.reshape(-1, 390, 4) * np.exp(day_vol * rng.normal(size=(n_days, 1, 4)))
    r[:, 360:] += link * r[:, 330:360].sum(axis=1, keepdims=True) / 30
    c = np.exp(np.log(p.close.to_numpy()[:1]) + np.cumsum(r.reshape(-1, 4), axis=0))
    o = np.vstack([c[:1], c[:-1]]) * np.exp(rng.normal(0, 0.003, c.shape)) ** np.repeat(
        np.r_[1, np.zeros(389)][None].repeat(n_days, 0).ravel()[:, None], 4, axis=1)
    sess = sorted({f"{d}" for d in p.close.index.tz_convert(p.tz).date})
    sp = {"train": (sess[21], sess[40]), "validation": (sess[41], sess[-1])}
    return _with(p, close=c, open_=o, high=np.maximum(o, c), low=np.minimum(o, c)), sp


def test_atlas_finds_a_planted_half_hour_link_and_keeps_its_band_honest():
    """The last half-hour made to continue the one before it: the map's 15:00 -> 15:30 cell stands
    far outside its luck band. In pure noise whose stock-days differ in volatility (as earnings
    days do), about 5% of cells pass 1.96, not the 10-20% a shuffle of days would give."""
    p, sp = _atlas_panel(4, link=0.8)
    at = IX.atlas(p, sp, draws=60)
    i, j = at["labels"].index("15:00"), at["labels"].index("15:30")
    assert at["corr"][i, j] > 0.3 and at["z"][i, j] > 4 and abs(at["corr"][0, 0] - 1) < 1e-9
    assert list(at["first"].index) == [0, 30, 60, 120, 180] and "strongest" in IX.atlas_words(at)
    assert "the night alone" in IX.atlas_words(at)
    up = np.triu(np.ones((14, 14), bool), 1)
    share = np.mean([np.mean(np.abs(IX.atlas(*_atlas_panel(seed, day_vol=0.5), draws=80)["z"]
                                    [up]) >= 1.96) for seed in (21, 22, 23)])
    assert share < 0.09


def test_intraday_run_end_to_end_without_reading_the_future(tmp_path, monkeypatch):
    """A tiny DLinear on the synthetic panel: graded rows at the decision times inside
    VALIDATION, boundaries as built, the truncation check passing (network, baselines, first
    month's scale); the saved file carries strategy.py's version, so prune_cache keeps it; the
    TEST read fits its scale on a held-out stretch before TEST."""
    pytest.importorskip("darts")
    monkeypatch.setattr(IX.rs, "CACHE", tmp_path)
    p, sp = _paths_panel()
    r = IX.run(p, sp, (10, 30, 30), kind="dlinear", epochs=1, draws=10)
    assert r["pred"].shape == (len(r["rows"]), len(IX.CHECKPOINTS)) and r["bars"] == (3, 3)
    assert np.isfinite(r["raw"][:, :3]).all() and np.isnan(r["raw"][:, 3:]).all()
    assert all(ok for _, ok, _ in IX.run_checks(r, sp, p))
    assert set(r["base"]) == {"ridge", "seasonal"} and set(r["r2_time"]) <= set(IX.DECISION_MINUTES)
    assert all(ok for _, ok, _ in IX.truncation(p, sp, (10, 30, 30), kind="dlinear", epochs=1,
                                                cut_session=2))
    again = IX.run(p, sp, (10, 30, 30), kind="dlinear", epochs=1, draws=10)    # from the cache
    assert again["score"]["r2"] == r["score"]["r2"]
    files = list(tmp_path.glob("intraday-*.pkl"))
    assert len(files) == 3 and IX.rs.prune_cache() == [] and len(list(tmp_path.glob("*.pkl"))) == 3
    train, es, cal = IX.training_sessions(sp, IX.prepared(p)["m"]["dates"], "test")
    n_use = len(train) + len(cal)                               # TRAIN + VALIDATION, usable
    assert train.max() < cal.min() and set(es) <= set(cal)
    assert len(cal) == max(IX.ES_DAYS, min(IX.TEST_CAL_DAYS, n_use // 3))
    big = {"train": ("d000", "d041"), "validation": ("d042", "d101")}   # the real window's size
    tr2, _, cal2 = IX.training_sessions(big, [f"d{i:03d}" for i in range(102)], "test")
    assert len(cal2) == IX.TEST_CAL_DAYS and len(tr2) == 102 - 1 - IX.TEST_CAL_DAYS
    rt = IX.run(p, sp, (10, 30, 30), kind="dlinear", epochs=1, draws=10, part="test")
    assert rt["fit"]["first_graded"] >= sp["test"][0] and list(rt["beta"]) == ["2025-08"]


def test_intraday_grid_search_measures_every_cell_once(monkeypatch):
    """Optuna's grid sampler visits every (bar size, input, output) cell exactly once, in a random
    order; a planted best is ranked first."""
    best_cell = (5, 60, 30)

    def fake_run(panel, sp, cell, **kw):
        a = sum(abs(np.log(x) - np.log(y)) for x, y in zip(cell, best_cell, strict=True))
        sc = {"r2": 0.01 - 0.002 * a, "r2_cp": np.zeros(10), "share_pos": 0.5, "worst": 0.0,
              "n": 100}
        return {"cell": tuple(cell), "bars": (cell[1] // cell[0], math.ceil(cell[2] / cell[0])),
                "score": sc, "score_raw": {**sc, "corr": 0.0, "corr_cp": np.zeros(10), "z": 0.0,
                                           "null_mean": 0.0, "null_sd": 0.01},
                "beta": {"2025-10": 0.5}, "base": {"ridge": 0.0, "seasonal": 0.0},
                "base_cp": {"ridge": np.zeros(10), "seasonal": np.zeros(10)}, "r2_time": {},
                "fit": {"sessions": 37, "epochs": 1, "stride": 1, "seconds": 0.0}}

    monkeypatch.setattr(IX, "run", fake_run)
    grans, ins, outs = (1, 5, 10), (30, 60), (10, 30, 60)
    res, study = IX.search(None, None, kind="tide", grans=grans, inputs=ins, outputs=outs,
                           epochs=1, verbose=False)
    cells = list(zip(res["granularity"], res["input"], res["prediction"], strict=True))
    assert len(cells) == len(set(cells)) == 18 and len(study.trials) == 18
    assert tuple(IX.best(res).iloc[0][["granularity", "input", "prediction"]]) == best_cell
    assert IX.grid_table(grans, ins, outs, 37).loc["10", "predict 10 min"] == 1
    assert IX.results_table(res, 20).shape[0] == 18 and "Winner" in IX.search_words(res, 20)


def test_intraday_example_paths_draw_each_decisions_own_scale(monkeypatch):
    """The worked decision's predicted prices: the decision bar's close carried along the scaled
    and unscaled checkpoint moves, at the checkpoint times, labelled with the scale of the
    decision's own month."""
    p, sp = _paths_panel()
    pp = IX.prepared(p)
    oc = pp["oc"]
    s = IX.session_index(pp["m"]["dates"], *sp["validation"])[1]
    row = int(np.flatnonzero((oc["s"] == s) & (oc["t"] == 30) & (oc["k"] == 0))[0])
    raw = np.full((1, len(IX.CHECKPOINTS)), np.nan)
    raw[0, :3] = [0.5, 1.0, -0.5]
    month = oc["month"][row]
    r = {"cell": (10, 30, 30), "rows": np.array([row]), "raw": raw, "pred": 0.2 * raw,
         "beta": {"1999-01": 0.9, month: 0.2}}
    ts = f"{pp['m']['dates'][s]} 10:00"
    e = IX.example_paths(p, [("x", r)], "AAA", ts)[0]
    assert e["beta"] == 0.2 and pd.Timestamp(e["times"][3]) == pd.Timestamp(ts) + pd.Timedelta(
        minutes=30)
    U, p0 = oc["U"][row], oc["p0"][row]
    assert e["price"][0] == p0 and e["price"][2] == pytest.approx(p0 * np.exp(0.2 * U[1]))
    assert e["price_raw"][3] == pytest.approx(p0 * np.exp(-0.5 * U[2]))


def test_intraday_charts_draw_what_they_say():
    """The figures' data, not just their traces: wall-clock text times (right in any viewer's
    time zone), closes stamped at their close, the coloured lines the best rows', the baselines
    of the row (or each part) they belong to, the frame on the best cell, the map's upper
    triangle only."""
    pytest.importorskip("darts")
    p, sp = _paths_panel()
    shape = IX.profile(p, sp)
    assert len(shape) == 390 and shape.index[0] == "09:31" and shape.index[-1] == "16:00"
    assert len(CH.day_shape(shape).data) == 1
    ts = f"{sp['validation'][0]} 10:30"
    d = IX.day_picture(p, "AAA", ts, (5, 60, 30))
    assert d["first"] == 11 and d["decision"] - d["start"] == 11 and d["end"] - d["decision"] == 6
    assert pd.Timestamp(d["times"][d["decision"]]) == pd.Timestamp(ts)
    pp = IX.prepared(p)
    s = list(pp["m"]["dates"]).index(sp["validation"][0])
    assert d["price"][d["decision"]] == pytest.approx(pp["m"]["cf"][s, 59, 0])   # 10:29's close
    fig = CH.stock_day(d)
    price = fig.data[0]
    assert isinstance(price.x[0], str) and price.x[d["decision"]] == f"{ts}:00"
    rects = [sh for sh in fig.layout.shapes if sh.type == "rect"]
    t = pd.DatetimeIndex(d["times"])
    assert rects[1].x0 == str(t[d["start"]] - pd.Timedelta(minutes=5)) and rects[2].x0 == f"{ts}:00"
    assert "today flag (0 / 1)" in [tr.name for tr in fig.data]
    flag = [tr for tr in fig.data if tr.name == "today flag (0 / 1)"][0]
    assert list(flag.y[:11]) == [0] * 11 and set(flag.y[11:]) == {1}
    act = CH.path_picture(p, [], "AAA", ts).data[0]
    close = pd.Series(p.close["AAA"].to_numpy(), index=p.close.index.tz_convert(p.tz)
                      .tz_localize(None) + pd.Timedelta(minutes=1))
    i = list(act.x).index(f"{ts}:00")
    assert act.y[i] == pytest.approx(close[pd.Timestamp(ts)])        # the close of 10:29-10:30
    rng = np.random.default_rng(5)
    rows = [{"granularity": g, "input": L, "prediction": S, "r2": r2, "z": rng.normal(),
             "r2_cp": rng.normal(0, 0.01, 10), "corr_cp": rng.normal(0, 0.1, 10),
             "ridge_cp": rng.normal(0, 0.01, 10), "seasonal_cp": rng.normal(0, 0.01, 10),
             "r2_time": {30: r2, 60: -r2}, "trial": i + 1}
            for i, (g, L, S, r2) in enumerate([(1, 30, 10, -0.002), (1, 60, 30, 0.001),
                                               (10, 30, 60, 0.003), (10, 60, 60, -0.001)])]
    res = pd.DataFrame(rows)
    cc = CH.checkpoint_curves(res)
    best = IX.best_per(res, "granularity")
    colour = [tr for tr in cc.data if tr.name.endswith(" min") and "bars" in tr.name]
    for tr, (_, r) in zip(colour, best.iterrows(), strict=True):
        assert np.allclose(tr.y, r["r2_cp"])
    top = IX.best(res).iloc[0]
    ridge = [tr for tr in cc.data if tr.name.startswith("ridge baseline")][0]
    assert np.allclose(ridge.y, top["ridge_cp"])
    two = res.iloc[[1, 2]].assign(label=["VALIDATION", "TEST"])
    lab = CH.checkpoint_curves(two, by="label")
    for part in ("VALIDATION", "TEST"):
        tr = [x for x in lab.data if x.name == f"ridge baseline, {part}"][0]
        assert np.allclose(tr.y, two.set_index("label").loc[part, "ridge_cp"])
    tc = CH.time_curves(res)
    by_span = IX.best_per(res, "prediction")
    for tr, (_, r) in zip(tc.data, by_span.iterrows(), strict=True):
        assert list(tr.x) == ["10:00", "10:30"] and np.allclose(tr.y, [r["r2"], -r["r2"]])
    maps = CH.granularity_maps(res, (1, 10), (30, 60), (10, 30, 60), unit="min")
    frame = maps.layout.shapes[0]
    assert maps.layout.xaxis.title.text == "predict (min)"
    assert frame.x0 + 0.5 == [10, 30, 60].index(int(top["prediction"]))
    assert frame.y0 + 0.5 == [30, 60].index(int(top["input"]))
    assert "min" in CH.search_history(res, unit="min").data[0].text[0]
    tab = IX.planted_table([{"planted correlation": 0.1, "oracle R²": 0.01, "network R²": 0.004,
                             "network z": 3.0, "network scale β": 0.3, "ridge R²": 0.005}])
    bars = CH.planted_bars(tab).data
    assert list(bars[1].y) == [0.004] and list(bars[2].y) == [0.005]
    at = {"labels": [f"x{i}" for i in range(14)], "corr": rng.normal(0, 0.1, (14, 14)),
          "z": rng.normal(size=(14, 14)), "days": 10, "stocks": 3}
    z = np.asarray(CH.atlas_map(at).data[0].z, float)
    assert np.isnan(z[1, 0]) and np.isfinite(z[0, 1]) and np.isnan(np.diag(z)).all()


# --- the day-start forecast (darts_forecast_daystart.ipynb): one stock-day per example ----------
DS = _load("daystart")


def test_daystart_bars_start_inside_the_day_and_never_hold_the_night():
    """The grid has 372 cells inside one session; a session's first bar runs from its first
    TRADED open (a volume-0 opening print is no trade), every later bar from the previous close,
    in typical moves of its clock position (that bar's RMS over the previous 20 sessions); the gap
    is the night in typical nights; giving every session its own price level moves the
    previous-day inputs and nothing intraday; the gap is withheld until the stock has traded
    before the decision (a first trade AT the decision minute is too late)."""
    grid = DS.cells()
    assert len(grid) == 372 and all(o + i + h <= 390 and not (o % g or i % g or h % g)
                                    for g, o, i, h in grid)
    p, _ = _paths_panel()
    s, k = 25, 0
    v, c = p.volume.to_numpy().copy(), p.close.to_numpy().copy()
    v[s * 390, k] = 0                                        # 09:30: a stale placeholder print
    c[s * 390, k] *= 1.05
    pp = DS.prepared(_with(p, volume=v, close=c))
    op, C = (x.to_numpy().reshape(-1, 390, 3) for x in (p.open_, p.close))
    b = DS.bars(pp, 5)
    assert b["y"][s, 0, k] * b["u"][s, 0, k] == pytest.approx(np.log(C[s, 4, k] / op[s, 1, k]))
    assert b["y"][s, 1, k] * b["u"][s, 1, k] == pytest.approx(np.log(C[s, 9, k] / C[s, 4, k]))
    r = np.log(C[s - 20:s, 9, k] / C[s - 20:s, 4, k])       # bar 1 in sessions s-20 .. s-1
    assert b["u"][s, 1, k] == pytest.approx(np.sqrt(np.mean(r ** 2)))
    raw = np.log(op[:, 0, k] / np.r_[np.nan, C[:-1, -1, k]])
    raw[s] = np.log(op[s, 1, k] / C[s - 1, -1, k])          # from the first TRADED open
    night = 1.4826 * np.median(np.abs(raw[s - 20:s]))
    assert DS.previous(pp)["gap"][s, k] == pytest.approx(np.clip(raw[s] / night, -4, 4))
    rng = np.random.default_rng(0)
    f = np.repeat(np.exp(rng.normal(0, 0.05, (40, 3))), 390, axis=0)
    moved = _with(p, **{x: getattr(p, x).to_numpy() * f for x in ("open_", "high", "low", "close")})
    a, m = DS.prepared(p), DS.prepared(moved)
    for g in (5, 30):
        ba, bm = DS.bars(a, g), DS.bars(m, g)
        for key in ("y", "u", "volume"):
            assert np.allclose(ba[key], bm[key], rtol=1e-6, atol=1e-6, equal_nan=True)
    assert not np.allclose(a["gap"], m["gap"], equal_nan=True)
    v2 = p.volume.to_numpy().copy()
    v2[27 * 390:27 * 390 + 30, 1] = 0                        # the first trade opens AT 10:00
    q = DS.prepared(_with(p, volume=v2))
    assert DS.statics(q, (5, 0, 30, 60), prev=True)[27, 1, 0] == 0.0     # decided at 10:00
    assert DS.statics(q, (5, 0, 60, 60), prev=True)[27, 1, 0] != 0.0     # decided at 10:30
    assert DS.statics(q, (5, 0, 30, 60), prev=False).shape[-1] == 0


def test_daystart_inputs_read_nothing_after_their_bar(monkeypatch):
    """Every target, scale, volume input and previous-day input up to a cut is bit-identical when
    the bars after it are replaced by random ones, with the original laid out column-major (as a
    database panel) and the scrambled copy row-major: a sum over another layout would round
    differently in the last bit."""
    monkeypatch.setattr(DS, "_PREP", {})
    monkeypatch.setattr(DS, "_BARS", {})
    p, _ = _paths_panel()
    p = _with(p, **{f: np.asfortranarray(getattr(p, f).to_numpy())
                    for f in ("open_", "high", "low", "close", "volume")})
    cut = 33 * 390 + 200
    zp = CK.scrambled(p, np.arange(len(p.close)) >= cut, 1)
    assert p.close.to_numpy().flags.f_contiguous and zp.close.to_numpy().flags.c_contiguous
    a, z = DS.prepared(p), DS.prepared(zp)
    for g in (1, 5, 30):
        ba, bz = DS.bars(a, g), DS.bars(z, g)
        n = 200 // g
        for key in ("y", "u", "volume"):
            np.testing.assert_array_equal(ba[key][:33], bz[key][:33])
            np.testing.assert_array_equal(ba[key][33, :n], bz[key][33, :n])
        assert not np.array_equal(ba["y"][34:], bz["y"][34:])      # the check can fail
    pa, pz = DS.previous(a, 2), DS.previous(z, 2)
    for key in ("gap", "prev_move"):
        np.testing.assert_array_equal(pa[key][:34], pz[key][:34])


def _gap_panel(n_days: int = 90, n_names: int = 6, effect: float = 1.0, busy: float = 0.5,
               seed: int = 7):
    """A synthetic panel with two planted effects on the hour after 10:00: the overnight gap
    (about 1%) moves it against itself, a busy first half hour (its volume x exp(z)) with itself.
    Each 30-minute bar from 10:00 to 11:00 drifts by (``busy`` x z - ``effect`` x the gap in
    typical nights) typical 30-minute moves. Its split as ``explorer.load_window`` would cut it."""
    rng = np.random.default_rng(seed)
    names = [f"S{i}" for i in range(n_names)]
    days = pd.bdate_range("2025-07-07", periods=n_days)
    idx = pd.DatetimeIndex(np.concatenate([pd.date_range(
        (d + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York"), periods=390,
        freq="min").tz_convert("UTC") for d in days]))
    sm = 0.0008
    r = rng.normal(0, sm, (n_days, 390, n_names))
    gap = rng.normal(0, 0.01, (n_days, n_names))
    z = rng.normal(0, 1, (n_days, n_names))
    r[:, 30:90] += (busy * z - effect * gap / 0.01)[:, None, :] * sm * np.sqrt(30) / 30
    start = np.cumsum(np.c_[np.zeros((1, n_names)).T, r.sum(axis=1).T].T[:-1], axis=0) \
        + np.cumsum(gap, axis=0)
    logp = start[:, None, :] + np.concatenate([np.zeros((n_days, 1, n_names)),
                                               np.cumsum(r, axis=1)[:, :-1]], axis=1)
    o, c = 100 * np.exp(logp), 100 * np.exp(logp + r)
    vol = rng.integers(1000, 5000, c.shape).astype(float)
    vol[:, :30] *= np.exp(z)[:, None, :]
    f = lambda x: pd.DataFrame(x.reshape(-1, n_names), index=idx, columns=names)  # noqa: E731
    p = Panel(close=f(c), open_=f(o), high=f(np.maximum(o, c) * 1.0003),
              low=f(np.minimum(o, c) * 0.9997), volume=f(vol), bar="1m", tz="America/New_York")
    sess = [str(d.date()) for d in days]
    sp = {"sessions": sess, "warmup": (sess[0], sess[20]), "train": (sess[21], sess[69]),
          "validation": (sess[70], sess[79]), "test": (sess[80], sess[-1]),
          "window": (sess[21], sess[-1]), "train_sessions": 49}
    return p, sp


def test_daystart_model_reads_its_optional_inputs():
    """The positive control of both optional inputs: with the gap planted to move the next hour
    against itself and a busy first half hour to move it with itself, a TiDE fit with both inputs
    predicts a lower path for the same mornings when told the night was up (+2) than down (-2),
    and a higher one when told the input bar was busy (+2) than quiet (-2); without the
    previous-day inputs the gap is not among its inputs."""
    pytest.importorskip("darts")
    p, sp = _gap_panel()
    pp, cell = DS.prepared(p), (30, 0, 30, 60)
    b = DS.bars(pp, 30)
    assert np.corrcoef(pp["gap"][21:].ravel(), b["y"][21:, 1].ravel())[0, 1] < -0.3
    assert np.corrcoef(b["volume"][21:, 0].ravel(), b["y"][21:, 1].ravel())[0, 1] > 0.15
    fit_d, held_d, graded = DS.split_days(pp, sp, "validation")
    kw = dict(volume=True, prev=True, prev_days=1)
    est, _ = DS.fit(b, pp, cell, "tide", fit_d, held_d, epochs=30, **kw)
    ser, vol, _, _ = DS.examples(b, pp, cell, graded, full=False, **kw)

    def path(gap: float | None = None, busy: float | None = None) -> np.ndarray:
        told = [x if gap is None else x.with_static_covariates(x.static_covariates.assign(gap=gap))
                for x in ser]
        busy_ = [x if busy is None else x.with_values(np.full_like(x.values(), busy)) for x in vol]
        out = est.predict(n=2, series=told, past_covariates=busy_, verbose=False)
        return np.stack([q.values()[:, 0] for q in out]).sum(axis=1)

    assert (path(gap=2.0) - path(gap=-2.0)).mean() < -2.0      # planted: about -6 (u holds both)
    assert (path(busy=2.0) - path(busy=-2.0)).mean() > 1.0     # planted: about +3
    off, _, _, _ = DS.examples(b, pp, cell, graded[:1], full=False, volume=False, prev=False)
    assert "gap" not in off[0].static_covariates.columns


def test_daystart_run_end_to_end_without_reading_the_future(tmp_path, monkeypatch):
    """A tiny DLinear on the synthetic panel, at offsets 0 and 30: one graded path per stock-day,
    decided at 09:30 + O + L, boundaries as built, the leak check passing, the saved result read
    back without a refit (and kept by prune_cache); the TEST read refits on TRAIN + VALIDATION
    with its last days held out; an earlier first training day reaches into the warm-up, never
    before the typical sizes have 10 earlier sessions. The series Darts reads are exactly the
    input + output bars from the offset, with the statics in one order."""
    pytest.importorskip("darts")
    monkeypatch.setattr(DS.rs, "CACHE", tmp_path)
    p, sp = _paths_panel()
    kw = dict(kind="dlinear", volume=True, prev=True, prev_days=1, epochs=2, draws=10)
    for cell in ((30, 0, 30, 60), (30, 30, 30, 60)):
        g, off, L, H = cell
        r = DS.run(p, sp, cell, **kw)
        assert r["pred"].shape == (len(r["s"]), H // g) and np.isfinite(r["raw"]).all()
        assert len(r["s"]) == 6 * 3 and ((r["s"] % (390 // g)) == (off + L) // g - 1).all()
        assert all(ok for _, ok, _ in r["checks"])
        assert all(ok for _, ok, _ in DS.truncation(p, sp, cell, cut_session=2, **kw))
        with monkeypatch.context() as mp:                    # from the cache: no refit
            mp.setattr(DS, "fit", lambda *a, **k: pytest.fail("refit, not read"))
            assert DS.run(p, sp, cell, **kw)["score"]["r2"] == r["score"]["r2"]
    assert DS.rs.prune_cache() == [] and len(list(tmp_path.glob("daystart-*.pkl"))) == 4
    rt = DS.run(p, sp, (30, 0, 30, 60), part="test", **kw)
    assert all(ok for _, ok, _ in rt["checks"]) and rt["fit"]["first_graded"] >= sp["test"][0]
    assert rt["fit"]["fit"] == (sp["train"][0], sp["validation"][0])        # TRAIN + VALIDATION
    assert rt["fit"]["held"] == (sp["sessions"][31], sp["validation"][1])   # minus the last 5
    fit_d, held_d, _ = DS.split_days(DS.prepared(p), sp, "validation", sp["sessions"][0])
    assert fit_d[0] == DS.VOL_DAYS // 2 and held_d[-1] == 29    # from the warm-up, sizes ready
    re = DS.run(p, sp, (30, 0, 30, 60), fit_from=sp["sessions"][0], **kw)
    assert all(ok for _, ok, _ in re["checks"]) and re["fit"]["fit"][0] == sp["sessions"][10]
    assert DS.warmup("2025-06-18", "2025-08-01") == 30 and DS.warmup(None, "2025-08-01") is None
    pp = DS.prepared(p)
    b = DS.bars(pp, 30)
    ser, vol, _, _ = DS.examples(b, pp, (30, 30, 30, 60), [25], full=True, volume=True,
                                 prev=True)
    assert len(ser) == 3 and len(ser[0]) == len(vol[0]) == 3
    np.testing.assert_allclose(ser[0].values()[:, 0], b["y"][25, 1:4, 0], rtol=1e-6)
    np.testing.assert_allclose(vol[0].values()[:, 0], b["volume"][25, 1:4, 0], rtol=1e-6)
    assert list(ser[0].static_covariates.columns) == ["is_AAA", "is_BBB", "is_CCC", "gap",
                                                      "prev_move"]
    with pytest.raises(ValueError, match="whole"):
        DS.run(p, sp, (30, 15, 30, 60), **kw)
