"""etf_trend: monthly cross-asset trend book — causality, timing, sizing and the variants on
synthetic panels (no DB)."""
import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

ET = strategies.load("etf_trend")
SMALL = {"lookback": 63, "min_history": 63, "vol_com": 20}


def _panel(drifts: dict, n: int = 400, vols: dict | None = None, seed: int = 5,
           start: str = "2019-01-01") -> Panel:
    """Daily total-return prices (00:00 UTC stamps, business days) with the given daily drifts
    and volatilities."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    vols = {"SHY": 0.0003, **(vols or {})}                     # cash barely moves
    cols = {}
    for name, mu in drifts.items():
        r = mu + rng.normal(0.0, vols.get(name, 0.006), n)
        cols[name] = 100.0 * np.cumprod(1 + r)
    px = pd.DataFrame(cols, index=idx)
    return Panel(close=px, open_=px.copy(), high=px * 1.002, low=px * 0.998,
                 volume=pd.DataFrame(1e6, index=idx, columns=px.columns), bar="1d")


def _weights(p, **over):
    return ET.signals(p, **{**ET.params, **SMALL, **over})["weights"]


UPDOWN = {"SPY": 0.0015, "XLK": 0.002, "TLT": -0.0015, "GLD": 0.0, "SHY": 0.0001}


def test_signals_reproduce_on_a_truncated_panel():
    """Causality: the weights on a panel cut at a date equal the full panel's up to that date."""
    p = _panel(UPDOWN, n=420)
    full = _weights(p)
    for cut in (200, 263, 330):
        q = Panel(close=p.close.iloc[:cut], open_=p.open_.iloc[:cut], high=p.high.iloc[:cut],
                  low=p.low.iloc[:cut], volume=p.volume.iloc[:cut], bar="1d")
        part = _weights(q)
        pd.testing.assert_frame_equal(part, full.iloc[:cut])
    for over in ({"mode": "long_short"}, {"blend": "21,63", "vol_target": 0.05},
                 {"budget": "class", "overlay": "SPY", "offset": 4}):
        full = _weights(p, **over)
        q = Panel(close=p.close.iloc[:300], open_=p.open_.iloc[:300], high=p.high.iloc[:300],
                  low=p.low.iloc[:300], volume=p.volume.iloc[:300], bar="1d")
        pd.testing.assert_frame_equal(_weights(q, **over), full.iloc[:300])


def test_uptrend_held_downtrend_flat_and_its_budget_goes_to_cash():
    w = _weights(_panel(UPDOWN)).iloc[-1]
    assert w["SPY"] > 0 and w["XLK"] > 0
    assert w["TLT"] == 0.0
    assert w["SHY"] > 0
    assert w.sum() == pytest.approx(0.98)


def test_trades_only_the_bar_after_a_months_first_trading_day():
    p = _panel(UPDOWN, n=420)
    w = _weights(p)
    changed = w.ne(w.shift()).any(axis=1).to_numpy()
    changed[0] = False
    fill_days = np.flatnonzero(np.roll(ET.signals.__globals__["decision_bars"](p.close.index), 1))
    assert set(np.flatnonzero(changed)) <= set(fill_days)
    assert changed.sum() >= 3


def test_inverse_volatility_sizes():
    p = _panel({"A": 0.003, "B": 0.003, "SHY": 0.0}, vols={"A": 0.003, "B": 0.012}, n=300)
    w = _weights(p, band=0.0).iloc[-1]
    assert w["A"] / w["B"] == pytest.approx(4.0, rel=0.35)       # vol ratio 0.012 / 0.003
    assert w["A"] + w["B"] == pytest.approx(0.98)                # both on: SHY gets nothing


def test_band_keeps_a_held_weight_while_its_target_drifts_a_little():
    p = _panel({"A": 0.002, "B": 0.002, "SHY": 0.0}, n=400)
    held = _weights(p, band=0.5)["A"].iloc[150:]
    retrim = _weights(p, band=0.0)["A"].iloc[150:]
    assert held.nunique() < retrim.nunique()
    assert (held > 0).all()


def _frames(rows: list[dict]):
    """Hand-made states (1 = on) and sizes, one decision bar per row, plus one bar after the
    last decision (targets are shifted one bar: decided at a close, filled at the next open)."""
    idx = pd.date_range("2024-01-01", periods=len(rows) + 1, freq="B", tz="UTC")
    size = pd.DataFrame([*rows, {}], index=idx).fillna(0.0)
    return (size != 0).astype(float), size, np.r_[np.ones(len(rows), dtype=bool), False]


def test_gross_fit_scales_only_the_assets_that_trade():
    """Kept weights never move: when they push the gross above `gross`, only the entering
    asset is scaled down (the old code rescaled every weight by a hair, review 2026-10-23)."""
    build = ET.signals.__globals__["build_weights"]
    state, size, dec = _frames([{"A": 0.40, "B": 0.30, "C": 0.28},
                                {"A": 0.32, "C": 0.30, "D": 0.38}])   # B exits, D enters
    w = build(state, size, dec, ["A", "B", "C", "D"], cash=None, gross=0.98, band=0.25)
    last = w.iloc[-1]                                  # shifted: the 2nd decision's targets
    assert last["A"] == 0.40 and last["C"] == 0.28     # kept, bit for bit
    assert last["B"] == 0.0
    assert last["D"] == pytest.approx(0.30)            # 0.38 scaled to what the gross leaves
    assert last.abs().sum() == pytest.approx(0.98)


def test_a_scaled_target_inside_the_band_keeps_its_weight():
    build = ET.signals.__globals__["build_weights"]
    state, size, dec = _frames([{"A": 0.40, "B": 0.30, "C": 0.28},
                                {"A": 0.45, "B": 0.39, "C": 0.33}])   # B moves 30%, A/C < 25%
    w = build(state, size, dec, ["A", "B", "C"], cash=None, gross=0.98, band=0.25)
    # kept A + C = 0.68 leave 0.30 for B: inside B's band, so B keeps its 0.30 too
    assert list(w.iloc[-1]) == [0.40, 0.30, 0.28]


def test_held_weights_change_only_beyond_the_band_in_long_short():
    """long_short fills `gross` at every decision, so kept weights push it over whenever their
    targets fall: no held weight may then move by a hair (each is an order: the real $1
    minimum, and the engine's borrow surcharge on a short scales with 1 / the change)."""
    p = _panel({"SPY": 0.0015, "XLK": 0.002, "TLT": -0.0015, "GLD": 0.0, "EEM": -0.001,
                "IEF": 0.0005, "SHY": 0.0001}, n=900,
               vols={"SPY": 0.01, "XLK": 0.014, "TLT": 0.009, "GLD": 0.008, "EEM": 0.013,
                     "IEF": 0.004})
    full = _weights(p, mode="long_short")
    w = full.drop(columns="SHY")
    prev = w.shift()
    moved = w.ne(prev) & (prev != 0) & (np.sign(w) == np.sign(prev))
    moved.iloc[0] = False
    rel = ((w - prev).abs() / prev.abs())[moved]
    assert moved.to_numpy().sum() > 0                            # some re-sizes happened
    assert np.nanmin(rel.to_numpy()) > 0.25                      # each beyond the band
    assert full.abs().sum(axis=1).max() <= 0.98 + ET.params["min_change"]
    step = full.diff().abs().iloc[1:]                            # every order, SHY included
    traded = step.gt(0) & full.iloc[1:].ne(0)                    # an exit may be any size
    assert step[traded].min().min() >= ET.params["min_change"]


def test_changes_below_min_change_place_no_order():
    """A re-size (beyond the band) or an entry smaller than `min_change` keeps the held weight,
    the cash asset's included, so rotation_targets emits no order; an exit always trades."""
    build = ET.signals.__globals__["build_weights"]
    cols = ["A", "B", "C", "E", "SHY"]
    state, size, dec = _frames([{"A": 0.004, "B": 0.30, "E": 0.10},
                                {"A": 0.006, "B": 0.30, "C": 0.002, "E": 0.10},  # A +50%, C hair
                                {"B": 0.405}])           # A, E exit; B +35%; SHY moves 0.001
    w = build(state, size, dec, cols, cash="SHY", gross=0.98, band=0.25, min_change=0.0025)
    assert list(w["A"].iloc[1:]) == [0.004, 0.004, 0.0]          # kept, then exits whole
    assert (w["C"] == 0.0).all()                                 # the hair entry never opens
    assert w["SHY"].iloc[3] == w["SHY"].iloc[1]                  # bit for bit
    assert w.iloc[3].sum() <= 0.98 + 0.0025
    orders = bt.rotation_targets(w).iloc[1:]
    assert orders["A"].notna().sum() == 2 and orders["SHY"].notna().sum() == 1
    assert orders.iloc[1].isna().all()                           # decision 2: no order at all
    loose = build(state, size, dec, cols, cash="SHY", gross=0.98, band=0.25)
    assert loose["A"].iloc[2] == 0.006 and loose["C"].iloc[2] == 0.002   # no tolerance


def test_spy_benchmark_counts_day_zero_from_the_initial_cash():
    """research.spy_returns compounds to SPY bought at the window's first open, costs paid
    (lab v2's day 0), not to the close-to-close return after day 0."""
    rs = _research()
    idx = pd.date_range("2024-01-01", periods=30, freq="B", tz="UTC")
    close = pd.DataFrame({"SPY": np.linspace(100.0, 110.0, 30), "SHY": 100.0}, index=idx)
    opn = close * 0.99
    p = Panel(close=close, open_=opn, high=close * 1.01, low=opn * 0.99,
              volume=pd.DataFrame(1e6, index=idx, columns=close.columns), bar="1d")
    fees, slip = 0.001, 0.0001
    r = rs.spy_returns(idx, fees, slip, "open", source=p)
    want = close["SPY"].iloc[-1] / (opn["SPY"].iloc[0] * (1 + slip) * (1 + fees)) - 1
    assert (1 + r).prod() - 1 == pytest.approx(want, rel=1e-9)
    assert r.iloc[0] == pytest.approx(close["SPY"].iloc[0] / (opn["SPY"].iloc[0] * (1 + slip)
                                                              * (1 + fees)) - 1, rel=1e-9)


def _research():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "strategies" / "etf_trend" / "research.py"
    spec = importlib.util.spec_from_file_location("etf_trend_research", path)
    rs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rs)
    return rs


def test_research_book_charges_borrow_on_the_realized_short():
    """research.borrow_owed against the engine's own borrow model where that one is exact: a
    short opened once and bought back whole, on a price that does not move."""
    rs = _research()
    idx = pd.date_range("2024-01-01", periods=300, freq="B", tz="UTC")
    px = pd.DataFrame({"A": 50.0, "SHY": 100.0}, index=idx)
    p = Panel(close=px, open_=px.copy(), high=px.copy(), low=px.copy(),
              volume=pd.DataFrame(1e6, index=idx, columns=px.columns), bar="1d")
    w = pd.DataFrame(0.0, index=idx, columns=["A", "SHY"])
    w.iloc[10:200, 0] = -0.5                                     # short A for 190 bars
    rate = 0.05
    pf = bt.run(p, {"weights": w}, fees=0.0, slippage=0.0)
    owed = rs.borrow_owed(pf, rate)
    days = (idx[200] - idx[10]).days
    assert owed.sum() == pytest.approx(rate * 0.5 * 10_000 * days / 365, rel=1e-6)
    engine = bt.run(p, {"weights": w, "borrow": rate}, fees=0.0, slippage=0.0)
    charged = engine.orders.records_readable["Fees"].sum()
    assert charged == pytest.approx(owed.sum(), rel=1e-3)


def test_cash_hurdle_turns_off_an_asset_that_lags_cash():
    p = _panel({"A": 0.0003, "SHY": 0.0008}, vols={"A": 0.0005, "SHY": 0.0005}, n=300)
    assert _weights(p).iloc[-1]["A"] == 0.0                      # positive but below cash
    assert _weights(p, hurdle="zero").iloc[-1]["A"] > 0


def test_long_short_shorts_the_downtrend_within_gross():
    w = _weights(_panel(UPDOWN), mode="long_short")
    assert w["TLT"].iloc[-1] < 0 and w["XLK"].iloc[-1] > 0
    assert w.abs().sum(axis=1).max() <= 0.98 + 1e-9


def test_young_assets_wait_for_their_history():
    p = _panel(UPDOWN, n=300)
    p.close.loc[p.close.index[:200], "XLK"] = np.nan             # XLK lists at bar 200
    w = _weights(p)
    assert (w["XLK"].iloc[:263] == 0).all()                      # needs 63 closes + a decision


def test_spy_overlay_switches_the_equity_sleeve_off():
    p = _panel({"SPY": -0.002, "XLK": 0.002, "TLT": 0.001, "SHY": 0.0})
    w = _weights(p).iloc[-1]
    assert w["XLK"] > 0                                          # its own trend is up
    w = _weights(p, overlay="SPY").iloc[-1]
    assert w["XLK"] == 0.0 and w["SPY"] == 0.0 and w["TLT"] > 0


def test_vol_target_only_scales_the_risky_book_down():
    p = _panel({"A": 0.002, "B": 0.002, "SHY": 0.0}, vols={"A": 0.02, "B": 0.02})
    plain, capped = _weights(p).iloc[-1], _weights(p, vol_target=0.05).iloc[-1]
    assert capped[["A", "B"]].sum() < plain[["A", "B"]].sum()
    assert capped.sum() == pytest.approx(0.98)                   # the rest waits in SHY
    loose = _weights(p, vol_target=5.0).iloc[-1]
    pd.testing.assert_series_equal(loose, plain)                 # never levers up


def test_class_budget_gives_each_class_an_equal_share():
    p = _panel({"SPY": 0.002, "XLK": 0.002, "XLE": 0.002, "TLT": 0.002, "SHY": 0.0}, n=300)
    w = _weights(p, budget="class", band=0.0, min_change=0.0).iloc[-1]
    assert w[["SPY", "XLK", "XLE"]].sum() == pytest.approx(w["TLT"], rel=1e-9)


def test_sma_rule_reads_the_close_against_its_average():
    px = np.r_[np.linspace(100, 150, 200), np.linspace(150, 120, 100)]   # up, then a slide
    idx = pd.date_range("2019-01-01", periods=len(px), freq="B", tz="UTC")
    close = pd.DataFrame({"A": px, "SHY": np.linspace(100, 101, len(px))}, index=idx)
    p = Panel(close=close, open_=close.copy(), high=close.copy(), low=close.copy(),
              volume=pd.DataFrame(1e6, index=idx, columns=close.columns), bar="1d")
    sma = _weights(p, rule="sma", lookback=50)["A"]
    ret = _weights(p, rule="return", lookback=250)["A"]
    assert sma.iloc[150] > 0 and sma.iloc[-1] == 0.0           # below its 50-bar average
    assert ret.iloc[-1] > 0                                    # still up over 250 bars


def test_assets_restricts_the_risky_universe():
    w = _weights(_panel(UPDOWN), assets="SPY").iloc[-1]
    assert w["SPY"] == pytest.approx(0.98) and w["XLK"] == 0.0 and w["SHY"] == 0.0


def test_decision_bars_are_the_offset_th_bar_of_each_month():
    idx = pd.date_range("2024-01-01", "2024-04-30", freq="B", tz="UTC")
    dec = ET.signals.__globals__["decision_bars"](idx, 0)
    assert [d.day for d in idx[dec]] == [1, 1, 1, 1]
    dec2 = ET.signals.__globals__["decision_bars"](idx, 2)
    assert [str(d.date()) for d in idx[dec2]][:2] == ["2024-01-03", "2024-02-05"]


def test_valid_rejects_what_the_book_cannot_express():
    assert ET.valid(ET.params)
    assert not ET.valid({"mode": "both"})
    assert not ET.valid({"overlay": "SPY", "mode": "long_short"})
    assert not ET.valid({"blend": "21,x"})
    assert not ET.valid({"gross": 1.2})
    assert not ET.valid({"offset": 20})


def test_engine_runs_the_book_and_trades_monthly():
    p = _panel(UPDOWN, n=420)
    sig = ET.signals(p, **{**ET.params, **SMALL})
    pf = bt.run(p, sig, fees=0.001, slippage=0.0001)
    orders = pf.orders.records_readable
    assert len(orders) > 0
    days = pd.DatetimeIndex(orders["Timestamp"]).unique()
    assert len(days) <= 420 // 15                                # ~ one rebalance a month
