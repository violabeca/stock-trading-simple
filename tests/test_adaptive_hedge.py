"""adaptive_hedge (proposal 3): a long short-interest book whose slots are taken by breakdown shorts
when stocks break down — synthetic panels, no DB."""
import inspect

import numpy as np
import pandas as pd
import pytest

from trading_simple import backtest as bt
from trading_simple import strategies
from trading_simple.data import Panel

AH = strategies.load("adaptive_hedge")
BOOK = AH.signals.__globals__["book"]      # the diagnostics view of the same book
# A small book: 4 slots, at most 2 shorts, one ranking group; N00..N04 rank >= 0.6 (5 names for
# 4 slots), so the long side is always full and the weakest held long is N03.
SMALL = {"neutral_groups": 0, "q_in": 0.6, "q_out": 0.55, "slots": 4, "max_shorts": 2,
         "adx_window": 10, "atr_window": 10, "rs_window": 40, "rs_skip": 5}
SIZE = 0.98 / 4


def _panel(n=200, crash=(), rebound=(), seed=0, names=12, shares=1e8, extra=None):
    """``names`` gently rising names N00..; days-to-cover = 1 + i (N00 ranks best). ``crash``:
    ``(name, start, bars, daily)`` steady declines; ``rebound``: the same for rallies after."""
    idx = pd.date_range("2023-01-02", periods=n, freq="B", tz="UTC")
    cols = [f"N{i:02d}" for i in range(names)]
    rng = np.random.default_rng(seed)
    rets = 0.002 + rng.normal(0.0, 0.0005, (n, names))
    for name, start, bars, daily in (*crash, *rebound):
        rets[start:start + bars, cols.index(name)] = daily
    close = pd.DataFrame(50.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)
    if extra is not None:
        close = close.join(extra.set_axis(idx))
    dtc = pd.DataFrame(np.tile(1.0 + np.arange(names), (n, 1)), index=idx, columns=cols)
    feats = {"short_interest": dtc, "fund_shares": pd.DataFrame(shares, index=idx, columns=cols)}
    return Panel(close=close, open_=close * 0.999, high=close * 1.005, low=close * 0.995,
                 volume=pd.DataFrame(1e6, index=idx, columns=close.columns), bar="1d",
                 features=feats)


def _run(p, **over):
    return AH.signals(p, **{**AH.params, **SMALL, **over})


def _holiday(p, k, priced=("CH",)):
    """``(base, holiday)``: ``p`` without its row ``k``, and ``p`` with row ``k`` unpriced for
    every name but ``priced`` (the lab's 2026 US-holiday rows, priced only for a Swiss name)."""
    keep = np.arange(len(p.close)) != k
    base = Panel(close=p.close[keep], open_=p.open_[keep], high=p.high[keep], low=p.low[keep],
                 volume=p.volume[keep], bar="1d",
                 features={n: v[keep] for n, v in p.features.items()})

    def hole(frame):
        frame = frame.copy()
        blank = [c for c in frame.columns if c not in priced]
        frame.iloc[k, frame.columns.get_indexer(blank)] = np.nan
        return frame

    hol = Panel(close=hole(p.close), open_=hole(p.open_), high=hole(p.high), low=hole(p.low),
                volume=hole(p.volume), bar="1d", features=p.features)
    return base, hol


def test_every_config_param_is_a_signals_keyword_and_the_config_is_valid():
    keywords = set(inspect.signature(AH.signals).parameters)
    assert set(AH.params) <= keywords
    assert AH.valid(AH.params)
    assert AH.fill == "open" and AH.order_notional == 1000.0


def test_valid_rejects_books_it_cannot_express():
    for bad in ({"q_out": 0.95}, {"max_shorts": 31}, {"gross": 1.2}, {"short_filter": "x"},
                {"si_mid": 0.25}, {"rs_skip": 300}, {"atr_mult": 0.0}, {"max_shorts": -1}):
        assert not AH.valid(bad), bad


def test_without_breakdowns_the_book_is_the_long_side_alone():
    w = _run(_panel())["weights"]
    last = w.iloc[-1]
    assert last[["N00", "N01", "N02", "N03"]].tolist() == pytest.approx([SIZE] * 4)
    assert (w >= 0).all().all()                      # no name broke down: no short at all


def test_a_breakdown_takes_the_weakest_longs_slot_at_minus_one_slot():
    p = _panel(crash=[("N10", 100, 40, -0.025)])
    out = BOOK(p, **{**AH.params, **SMALL})
    first = out["shorts"]["N10"].idxmax()
    assert out["shorts"]["N10"].any()
    assert not out["longs"].loc[first, "N03"]        # the weakest-ranked long was displaced
    assert out["longs"].loc[first, ["N00", "N01", "N02"]].all()
    assert any(x[:2] == ("long", "N03") and x[3] == first and x[4] == "displaced"
               for x in out["exits"])
    w = _run(p)["weights"]
    held = w["N10"][w["N10"] != 0]
    assert held.unique().tolist() == pytest.approx([-SIZE])   # one size, never re-traded


def test_max_shorts_caps_the_short_side_and_gross_never_exceeds_its_bound():
    crashes = [(f"N{i:02d}", 100 + i, 40, -0.025) for i in (6, 7, 8, 9, 10, 11)]
    w = _run(_panel(crash=crashes))["weights"]
    assert (w < 0).sum(axis=1).max() == 2
    assert w.abs().sum(axis=1).max() <= 0.98 + 1e-12
    assert (w > 0).sum(axis=1).iloc[-1] >= 2         # two longs kept: net >= 0 with 2 of 4
    assert _run(_panel(crash=crashes), max_shorts=0)["weights"].min().min() >= 0


def test_a_short_leaves_on_its_chandelier_and_on_an_opposite_start():
    crash, up = [("N10", 100, 30, -0.025)], [("N10", 130, 40, 0.03)]
    stop = BOOK(_panel(crash=crash, rebound=up), **{**AH.params, **SMALL, "atr_mult": 2.0})
    causes = [x[4] for x in stop["exits"] if x[0] == "short" and x[1] == "N10"]
    assert causes and causes[0] == "stop"
    late = BOOK(_panel(crash=crash, rebound=up),
                    **{**AH.params, **SMALL, "atr_mult": 60.0, "adx_min": 20.0})
    causes = [x[4] for x in late["exits"] if x[0] == "short" and x[1] == "N10"]
    assert causes and causes[0] == "opposite"
    exit_bar = [x[3] for x in late["exits"] if x[0] == "short" and x[1] == "N10"][0]
    assert late["parts"]["long_start"].loc[exit_bar, "N10"]


def test_the_short_clock_buys_back_a_breakdown_that_stopped_making_new_lows():
    p = _panel(crash=[("N10", 100, 30, -0.025)])     # 30 bars of new lows, then a gentle rise
    held = BOOK(p, **{**AH.params, **SMALL})
    out = BOOK(p, **{**AH.params, **SMALL, "short_hold": 10})
    starts = out["parts"]["short_start"]["N10"]
    exit_ = [x for x in out["exits"] if x[0] == "short" and x[1] == "N10"][0]
    assert exit_[4] == "short_hold"
    last_start = starts[starts & (starts.index < exit_[3])].index.max()
    assert starts.index.get_loc(exit_[3]) - starts.index.get_loc(last_start) == 10
    assert held["shorts"].loc[exit_[3], "N10"]       # off by default: still short on that bar


def test_the_turtle_exit_covers_a_short_on_a_close_above_the_prior_high():
    p = _panel(crash=[("N10", 100, 30, -0.025)], rebound=[("N10", 130, 20, 0.03)])
    base = {**AH.params, **SMALL, "atr_mult": 60.0}
    out = BOOK(p, **{**base, "exit_window": 3})
    exit_ = [x for x in out["exits"] if x[0] == "short" and x[1] == "N10"][0]
    assert exit_[4] == "reversal"
    prior_high = p.high["N10"].rolling(3).max().shift(1)
    assert p.close.loc[exit_[3], "N10"] > prior_high.loc[exit_[3]]
    off = [x for x in BOOK(p, **base)["exits"] if x[0] == "short" and x[1] == "N10"]
    assert not off or off[0][3] > exit_[3]           # without it the short is held longer


def test_the_squeeze_stop_covers_a_short_that_rises_past_it_from_entry():
    idx = pd.date_range("2024-01-01", periods=12, freq="B", tz="UTC")
    close = pd.DataFrame({"X": [10, 10, 10, 9.8, 10.2, 10.5, 10.9, 11.2, 11.0, 11.5, 12, 12]},
                         index=idx, dtype=float)
    false = pd.DataFrame(False, index=idx, columns=close.columns)
    parts = {"score": close * np.nan, "close": close, "low": close * 0.99,
             "atr": close * 0 + 0.1, "rel": close * 0, "short_start": false.copy(),
             "long_start": false, "short_ok": ~false, "crowded": false, "cover": false}
    parts["short_start"].iloc[3] = True                  # shorted at the 9.8 close
    _book = AH.signals.__globals__["_book"]
    kw = {"q_in": 0.9, "q_out": 0.8, "slots": 2, "max_hold": 99, "max_shorts": 1,
          "atr_mult": 50.0}
    _, shorts, exits = _book(parts, **kw, short_stop=0.1)
    assert exits == [("short", "X", idx[3], idx[6], "squeeze")]      # 10.9 >= 9.8 x 1.1
    assert shorts["X"].tolist() == [False] * 3 + [True] * 3 + [False] * 6
    assert _book(parts, **kw)[1]["X"].iloc[-1]                     # off: still short


def test_a_name_too_crowded_to_borrow_is_never_shorted_and_borrow_follows_the_tiers():
    p = _panel(crash=[("N10", 100, 40, -0.025)], shares=1e8)
    shares = p.features["fund_shares"].copy()
    shares["N10"] = 1e6 * 11 / 0.35                  # 11 days x 1e6 shares = 35% of shares
    p.features["fund_shares"] = shares
    out = _run(p)
    assert (out["weights"]["N10"] == 0).all()
    shares["N09"] = 1e6 * 10 / 0.15                  # 15%: the middle tier
    shares["N08"] = 1e6 * 9 / 0.25                   # 25%: the top tier
    b = _run(p)["borrow"].iloc[-1]
    assert b["N09"] == pytest.approx(0.03) and b["N08"] == pytest.approx(0.10)
    assert b["N00"] == pytest.approx(0.005)


def test_variant_b_shorts_only_the_crowded_half():
    p = _panel(crash=[("N05", 100, 40, -0.025)])     # rank 7/12 > 0.5, below q_in: not held
    assert (_run(p)["weights"]["N05"] < 0).any()
    assert (_run(p, short_filter="crowded")["weights"]["N05"] >= 0).all()
    q = _panel(crash=[("N10", 100, 40, -0.025)])     # rank 2/12: crowded, shorted under B
    assert (_run(q, short_filter="crowded")["weights"]["N10"] < 0).any()


def test_the_index_hedge_holds_every_short_slot_as_the_index():
    spy = pd.DataFrame({"SPY": np.linspace(400.0, 480.0, 200)})
    p = _panel(crash=[("N10", 100, 40, -0.025)], extra=spy)
    single = _run(_panel(crash=[("N10", 100, 40, -0.025)]))["weights"]
    out = _run(p, hedge="SPY")
    w = out["weights"]
    assert (w.drop(columns="SPY") >= 0).all().all()
    assert w["SPY"].tolist() == pytest.approx((-(single < 0).sum(axis=1) * SIZE).tolist())
    assert w.drop(columns="SPY").clip(lower=0).equals(single.clip(lower=0))
    assert out["borrow"]["SPY"].iloc[-1] == pytest.approx(0.005)
    with pytest.raises(ValueError, match="hedge"):
        _run(_panel(), hedge="SPY")


def test_a_holiday_row_is_skipped_so_it_neither_empties_the_book_nor_stops_new_shorts():
    n = 201
    p = _panel(n=n, crash=[("N10", 100, 40, -0.025), ("N09", 135, 30, -0.025)],
               extra=pd.DataFrame({"CH": np.linspace(80.0, 90.0, n)}))
    base, hol = _holiday(p, 130)
    wb, wh = _run(base)["weights"], _run(hol)["weights"]
    day = hol.close.index[130]
    pd.testing.assert_frame_equal(wh.drop(index=day), wb)       # the same book, day for day
    pd.testing.assert_series_equal(wh.loc[day], wh.iloc[129], check_names=False)  # held
    assert (wh["N10"].iloc[120:n] < 0).all()                    # the short rides through it
    assert (wh["N09"] < 0).any()                                # a start right after it trades
    assert (wh.abs().sum(axis=1).iloc[60:] > 0).all()           # never an empty book
    _, empty = _holiday(p, 130, priced=())                      # nothing priced at all
    pd.testing.assert_frame_equal(_run(empty)["weights"].drop(index=day), wb)


def test_the_engine_trades_nothing_on_the_holiday_and_values_the_same_book():
    n = 201
    p = _panel(n=n, crash=[("N10", 100, 40, -0.025)],
               extra=pd.DataFrame({"CH": np.linspace(80.0, 90.0, n)}))
    base, hol = _holiday(p, 130)
    day = hol.close.index[130]
    runs = [bt.run(q, _run(q), fees=0.0011, slippage=0.0001, init_cash=10_000.0)
            for q in (base, hol)]
    orders = [pf.orders.records_readable for pf in runs]
    assert (orders[1]["Timestamp"] != day).all() and len(orders[0]) == len(orders[1])
    np.testing.assert_allclose(runs[1].value().drop(index=day).to_numpy(),
                               runs[0].value().to_numpy(), rtol=1e-9)


@pytest.mark.parametrize("bars", [1, 3])
def test_a_held_name_without_a_price_is_carried_not_sold(bars):
    p = _panel(crash=[("N10", 100, 40, -0.025)])
    for frame in (p.close, p.open_, p.high, p.low, p.volume):
        frame.iloc[150:150 + bars, frame.columns.get_indexer(["N00", "N10"])] = np.nan
    out = BOOK(p, **{**AH.params, **SMALL})
    assert np.isnan(out["parts"]["score"].iloc[150]["N00"])    # unranked on the empty bar
    assert out["longs"]["N00"].iloc[140:170].all() and out["shorts"]["N10"].iloc[140:170].all()
    assert not [x for x in out["exits"] if x[1] in ("N00", "N10") and x[3] > p.close.index[140]]
    w = _run(p)["weights"]
    assert w["N00"].iloc[141:171].nunique() == 1 and w["N10"].iloc[141:171].nunique() == 1


def test_the_book_never_exits_on_a_bar_without_a_price():
    """One empty bar (no close, so no score, low or ATR) for a held long and a held short: both
    ride through it and nothing exits, not even on an input that is not a price (the short's
    crowding flag, set on that bar only). A PRICED bar without a score is different: that long
    is ineligible (informed_flow's rule) and leaves."""
    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    close = pd.DataFrame(10.0, index=idx, columns=["L", "S", "Z"])
    false = pd.DataFrame(False, index=idx, columns=close.columns)
    parts = {"score": pd.DataFrame({"L": 0.95, "S": 0.1, "Z": 0.5}, index=idx),
             "close": close, "low": close * 0.99, "atr": close * 0 + 0.1, "rel": close * 0,
             "short_start": false.copy(), "long_start": false, "short_ok": ~false,
             "crowded": false, "cover": false}
    parts["short_start"].iloc[1, 1] = True              # S breaks down on bar 1
    kw = {"q_in": 0.9, "q_out": 0.8, "slots": 2, "max_hold": 99, "max_shorts": 1,
          "atr_mult": 5.0}
    _book = AH.signals.__globals__["_book"]
    empty = {k: v.copy() for k, v in parts.items()}
    for k in ("score", "close", "low", "atr"):
        empty[k].iloc[4, :2] = np.nan                   # bar 4: L and S unpriced
    empty["crowded"].iloc[4, 1] = True
    longs, shorts, exits = _book(empty, **kw)
    assert exits == []
    assert longs["L"].all() and shorts["S"].iloc[1:].all() and not longs["Z"].any()
    unranked = {**parts, "score": empty["score"]}       # priced, but no score on bar 4
    assert _book(unranked, **kw)[2][0] == ("long", "L", idx[0], idx[4], "ineligible")


def test_a_target_change_on_a_bar_without_an_open_waits_for_the_next_priced_open():
    idx = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    close = pd.DataFrame({"A": [10, 10.5, 11, np.nan, 11.5, 12, 12, 12]}, index=idx)
    w = pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    moved = AH.signals.__globals__["_on_priced_opens"](w, close)
    assert moved["A"].tolist() == [0.5] * 4 + [0.0] * 4
    p = Panel(close=close, open_=close, high=close, low=close, volume=close * 0 + 1e6,
              bar="1d")
    sold, lost = (bt.run(p, {"weights": x}, fees=0.0, slippage=0.0, init_cash=1e4)
                  for x in (moved, w))
    assert len(sold.orders.records_readable) == 2            # bought, then sold on bar 4
    assert len(lost.orders.records_readable) == 1            # the engine dropped the sell


def test_signals_moves_a_target_change_on_a_bar_without_an_open_to_the_next_priced_open():
    """signals() itself defers the change, not only the helper: a short's entry and its cover,
    each due on a bar where the name has no open, happen one bar later, and every other name's
    weights are untouched (the decisions read no open)."""
    crash, up = [("N10", 100, 30, -0.025)], [("N10", 130, 40, 0.03)]
    w = _run(_panel(crash=crash, rebound=up), atr_mult=2.0)["weights"]
    short = w["N10"].lt(0).to_numpy()
    enter = int(np.argmax(short))                       # filled at this bar's open
    leave = enter + int(np.argmin(short[enter:]))       # covered at this bar's open
    assert 0 < enter < leave < len(w) - 2
    for bar in (enter, leave):
        p = _panel(crash=crash, rebound=up)
        p.open_.iloc[bar, p.open_.columns.get_loc("N10")] = np.nan
        v = _run(p, atr_mult=2.0)["weights"]
        moved = w["N10"].copy()
        moved.iloc[bar] = w["N10"].iloc[bar - 1]          # the change waits one bar
        pd.testing.assert_series_equal(v["N10"], moved)
        pd.testing.assert_frame_equal(v.drop(columns="N10"), w.drop(columns="N10"))


def test_a_short_start_on_a_full_book_never_displaces_a_long_without_a_price():
    """The breakdown takes the slot of the weakest PRICED long. N00, the book's first long, is
    unpriced on the deciding bar: it is carried, and N03 (the weakest priced long) leaves."""
    crash = [("N10", 100, 40, -0.025)]
    day = BOOK(_panel(crash=crash), **{**AH.params, **SMALL})["shorts"]["N10"].idxmax()
    p = _panel(crash=crash)
    k = p.close.index.get_loc(day)
    for frame in (p.close, p.open_, p.high, p.low, p.volume):
        frame.iloc[k, frame.columns.get_loc("N00")] = np.nan
    out = BOOK(p, **{**AH.params, **SMALL})
    assert out["shorts"].loc[day, "N10"] and out["longs"].loc[day, ["N00", "N01", "N02"]].all()
    left = [(x[0], x[1], x[4]) for x in out["exits"] if x[3] == day]
    assert left == [("long", "N03", "displaced")]
    w = _run(p)["weights"].iloc[k + 1]                  # the decisions of ``day``, filled
    assert w["N00"] == pytest.approx(SIZE) and w["N03"] == 0 and w["N10"] == pytest.approx(-SIZE)


def test_signals_are_causal_a_truncated_panel_reproduces_the_weights():
    rng = np.random.default_rng(11)
    n, names = 320, 40
    idx = pd.date_range("2022-01-03", periods=n, freq="B", tz="UTC")
    cols = [f"S{i:02d}" for i in range(names)]
    trend = rng.normal(0.0, 0.004, names)
    rets = trend + rng.normal(0.0, 0.02, (n, names))
    close = pd.DataFrame(30.0 * np.cumprod(1 + rets, axis=0), index=idx, columns=cols)
    dtc = pd.DataFrame(rng.uniform(1, 10, (n, names)), index=idx, columns=cols).rolling(
        15, min_periods=1).mean()
    dtc.iloc[:30] = np.nan                           # short interest starts late
    p = Panel(close=close, open_=close.shift(1).fillna(close),
              high=close * (1 + rng.uniform(0, 0.02, (n, names))),
              low=close * (1 - rng.uniform(0, 0.02, (n, names))),
              volume=pd.DataFrame(rng.uniform(5e5, 2e6, (n, names)), index=idx, columns=cols),
              bar="1d", features={"short_interest": dtc,
                                  "fund_shares": pd.DataFrame(1e8, index=idx, columns=cols)})
    for frame in (p.close, p.open_, p.high, p.low):  # a holiday row and a 3-bar price gap
        frame.iloc[200] = np.nan
        frame.iloc[240:243, :3] = np.nan
    params = {**AH.params, "slots": 8, "max_shorts": 4, "rs_window": 60, "rs_skip": 5,
              "atr_window": 14, "atr_mult": 4.0, "adx_min": 25.0}
    full = AH.signals(p, **params)
    assert (full["weights"] < 0).any().any() and (full["weights"] > 0).any().any()
    for cut in (150, 200, 201, 233, 241, 301):
        q = Panel(close=close.iloc[:cut], open_=p.open_.iloc[:cut], high=p.high.iloc[:cut],
                  low=p.low.iloc[:cut], volume=p.volume.iloc[:cut], bar="1d",
                  features={k: v.iloc[:cut] for k, v in p.features.items()})
        part = AH.signals(q, **params)
        pd.testing.assert_frame_equal(part["weights"], full["weights"].iloc[:cut])
        pd.testing.assert_frame_equal(part["borrow"], full["borrow"].iloc[:cut])


def test_the_engine_runs_the_book_and_shorts_lock_their_cash():
    crashes = [(f"N{i:02d}", 100 + 3 * i, 40, -0.02) for i in (8, 9, 10, 11)]
    p = _panel(crash=crashes)
    sig = _run(p)
    pf = bt.run(p, sig, fees=0.0011, slippage=0.0001, init_cash=10_000.0)
    value = pf.value()
    assets = pf.asset_value(group_by=False)
    assert np.isfinite(value).all() and (assets < 0).any().any()
    assert (assets.abs().sum(axis=1) / value).max() <= 1.0 + 1e-6
