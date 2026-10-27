"""tori_trades_trendline: the trendline bounce replay (``bounce.py``), long and short, on
hand-made 1-minute candles, no database. Cutting the data off later changes nothing before the
cut; every trade follows the rules it reports (the signal candle came to the line and closed on
its side, the stop starts beyond the line and only moves in the trade's favour, each exit happened
for its stated reason at the stated price); shorts on a market turned upside down make exactly
the longs' trades; the wallet compounds trade by trade; without the session refresh the lines are
only drawn again after a trade or a broken line; ``own_lines`` trades only 15-minute lines and
the anchor gap keeps every 15-minute line's points A and B apart.
"""

import importlib.util
import types

import numpy as np
import pandas as pd
import pytest

from trading_simple import strategies

_spec = importlib.util.spec_from_file_location(
    "test_tori_bounce", strategies.STRATEGIES_DIR / "tori_trades_trendline" / "bounce.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
TD = B.td
COSTS = types.SimpleNamespace(commission_per_share=0.005, min_commission=1.0, spread_bps=2.0,
                              slippage_bps=1.0)
FREE = types.SimpleNamespace(commission_per_share=0.0, min_commission=0.0, spread_bps=0.0,
                             slippage_bps=0.0)
START = "2025-03-03"


def market(sessions=60, seed=3):
    """Daily and regular-session 1-minute candles of a random walk with overnight gaps, shaped
    like ``topdown.load``'s output."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=sessions)
    steps = rng.normal(0.00004, 0.0011, (sessions, 390))
    steps[:, 0] += rng.normal(0, 0.006, sessions)
    close = 100 * np.exp(np.cumsum(steps.ravel()))
    open_ = np.r_[close[0], close[:-1]]
    wick = rng.uniform(0, 0.0006, (2, close.size))
    t = pd.DatetimeIndex([d + pd.Timedelta(minutes=570 + m) for d in days for m in range(390)])
    minutes = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wick[0]),
                            "low": np.minimum(open_, close) * (1 - wick[1]), "close": close},
                           index=t.tz_localize(TD.NY))
    return daily_of(minutes), minutes


def daily_of(minutes):
    """Daily candles built from the minutes."""
    return minutes.groupby(minutes.index.tz_localize(None).normalize()).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))


def upside_down(candles, top=300.0):
    """The candles mirrored: every price p becomes top - p, so highs become lows."""
    return pd.DataFrame({"open": top - candles["open"], "high": top - candles["low"],
                         "low": top - candles["high"], "close": top - candles["close"]},
                        index=candles.index)


@pytest.fixture(scope="module", params=["long", "short", "both"])
def run(request):
    """One replay of the whole hand-made market per ``sides``, with the candles it traded."""
    daily, minutes = market()
    result = B.replay(daily, minutes, start=START, costs=COSTS, sides=request.param)
    return result, TD.candles(daily, minutes, "15m", TD.clock(daily))


def test_it_trades(run):
    """The hand-made market gives the rules something to do: one side alone trades at least
    three times with both exit kinds; both sides together trade each side at least once (one
    position at a time: the third trade, a long, runs to the end)."""
    result, _ = run
    trades = result["trades"]
    sides = set(trades["side"])
    if sides == {"long", "short"}:
        assert len(trades) >= 3
    else:
        assert len(trades) >= 3 and "stop hit" in set(trades["reason"])
        assert set(trades["reason"]) & {"closed below the line", "closed above the line"}


def test_the_chartbook_cuts_every_chart_like_a_fresh_snapshot():
    """At random moments (inside a candle, at a candle's open, at the open and the close of a
    session, a Monday and a month's first session), every chart the Chartbook cuts from its
    candles built once equals the one built from the minutes known then."""
    daily, minutes = market()
    sessions = TD.clock(daily)
    charts = list(TD.TIMEFRAMES)[:6]
    book = B.Chartbook(daily, minutes, charts, sessions)
    rng = np.random.default_rng(0)
    stamps = minutes.index[rng.choice(len(minutes), 25, replace=False)]
    special = [pd.Timestamp(t).tz_localize(TD.NY) for t in (
        "2025-02-03 09:31", "2025-02-03 09:45", "2025-02-03 16:00", "2025-03-03 10:00",
        "2025-01-02 09:31", "2025-02-14 15:59", "2025-03-07 13:30")]
    for cutoff in [*stamps, *special]:
        fresh = B.snapshot(daily, minutes, cutoff, charts, sessions)
        for tf, frame in book.at(cutoff).items():
            pd.testing.assert_frame_equal(frame, fresh[tf], check_freq=False,
                                          obj=f"{tf} at {cutoff}")


def test_cutting_the_data_later_changes_nothing_before():
    """The same replay on the data up to a later session: every setup and every trade that was
    over before the cut's last candle is the same (no decision read a later candle)."""
    daily, minutes = market()
    result = B.replay(daily, minutes, start=START, costs=COSTS, sides="both")
    end = daily.index[50]
    cut = B.replay(daily[daily.index <= end],
                   minutes[minutes.index.tz_localize(None).normalize() <= end],
                   start=START, costs=COSTS, sides="both")
    last = cut["chart"].index[-1]
    early = result["setups"][result["setups"]["time"] < last]
    pd.testing.assert_frame_equal(early, cut["setups"][cut["setups"]["time"] < last])
    for i in early.index:
        for side in ("long", "short"):
            pd.testing.assert_frame_equal(result["lines"][i][side], cut["lines"][i][side])
    done = result["trades"][result["trades"]["exit time"] < last]
    assert len(done)
    pd.testing.assert_frame_equal(done, cut["trades"][cut["trades"]["exit time"] < last])


def test_every_trade_follows_the_rules(run):
    """Signal candle: its low (high) within the band of the action line, its close on the
    line's side; in at the next open; the first stop ``room`` beyond the line; the stop only
    moves in the trade's favour and stays ``room`` beyond the line; a stop exit at the stop (or
    a worse open), a close-beyond exit at the next open after a close beyond the line."""
    result, chart = run
    o, hi, lo, c, x = (chart[k].to_numpy() for k in ("open", "high", "low", "close", "x"))
    stops = result["stops"]
    trades = result["trades"].rename(columns=lambda name: name.replace(" ", "_"))
    for t in trades.itertuples():
        d, ext = (1, lo) if t.side == "long" else (-1, hi)

        def line(k, t=t):
            return t.a_y + t.slope * (x[k] - t.a_x)
        s = t.entry_k - 1
        assert d * (ext[s] - line(s)) <= t.band + 1e-9 and d * (c[s] - line(s)) >= 0
        assert t.entry == o[t.entry_k]
        assert t.initial_stop == pytest.approx(line(t.entry_k) - d * t.room)
        path = stops[stops["trade"] == t.trade]           # the stop at each close in the trade
        assert (d * np.diff(path["stop"]) >= -1e-12).all()
        ks = np.searchsorted(chart.index, path["time"])
        beyond = d * (path["stop"].to_numpy() - np.array([line(k) for k in ks]))
        assert (beyond <= -t.room + 1e-9).all()
        if t.reason == "stop hit":
            stop = path["stop"].iloc[-1] if len(path) else t.initial_stop
            assert d * (ext[t.exit_k] - stop) <= 1e-9
            assert t.exit == pytest.approx(min(o[t.exit_k], stop) if d > 0
                                           else max(o[t.exit_k], stop))
        elif t.reason.startswith("closed"):
            k = t.exit_k - 1
            assert d * (c[k] - line(k)) < -t.give and t.exit == o[t.exit_k]


@pytest.mark.parametrize(("entry", "method", "bands"),
                         [("next_open", "walk", 0.0), ("limit", "walk", 0.0),
                          ("next_open", "cascade", 0.0), ("next_open", "cascade", 0.25)])
def test_shorts_on_an_upside_down_market_make_the_longs_trades(entry, method, bands):
    """Mirror every price (p -> 300 - p): downward lines become upward lines, highs become lows,
    so the short-only replay of the mirror must trade exactly when and how the long-only replay
    of the original does (same candles, same R), cost-free, with either entry and either way of
    drawing the lines (the cascade trades rarely: it gets a longer market)."""
    daily, minutes = market() if method == "walk" else market(sessions=120, seed=5)
    knobs = {"entry": entry, "method": method, "screen": 40 if method == "cascade" else 150,
             "line_band": bands, "price_band": bands, "noise_window": 20}
    longs = B.replay(daily, minutes, start=START, costs=FREE, sides="long", **knobs)["trades"]
    shorts = B.replay(upside_down(daily), upside_down(minutes), start=START, costs=FREE,
                      sides="short", **knobs)["trades"]
    assert len(longs) >= 3
    for col in ("entry time", "exit time", "line A"):
        assert shorts[col].tolist() == longs[col].tolist()
    assert np.allclose(shorts["R"], longs["R"])
    assert (shorts["reason"].str.replace("above", "below") == longs["reason"]).all()


def test_the_wallet_compounds(run):
    """The strategy's final return is the product of its trades' returns after costs."""
    result, _ = run
    total = np.prod(1 + result["trades"]["return"]) - 1
    assert result["summary"]["strategy return"] == pytest.approx(total)


def test_the_filters_keep_to_their_rules():
    """``min_span`` 5: every traded line has its points A and B at least five sessions apart
    (the setups keep every line drawn, the filter acts on what is traded). ``trend`` 20: every
    long's signal candle closed above the average of the 20 daily closes before its session,
    every short's below."""
    daily, minutes = market()
    result = B.replay(daily, minutes, start=START, costs=COSTS, sides="both", min_span=5.0,
                      trend=20)
    average = daily["close"].rolling(20).mean().shift(1)
    chart = TD.candles(daily, minutes, "15m", TD.clock(daily))
    trades = result["trades"]
    assert len(trades)
    for t in trades.itertuples():
        drawn = result["lines"][t.setup][t.side]
        line = drawn[(drawn["a_x"] == t.a_x) & (drawn["slope"] == t.slope)]
        assert len(line) == 1 and (line["b_x"] - line["a_x"]).iloc[0] >= 5.0
        signal = t.entry_k - 1
        d = 1 if t.side == "long" else -1
        assert d * (chart["close"].iloc[signal]
                    - average[chart.index[signal].normalize()]) > 0


def test_target_skip_open_and_reward_keep_to_their_rules():
    """``target`` 1: a target exit fills at entry + 1 R (or a better open) and earns at least
    1 R; ``skip_open`` 2: no signal on a session's first two candles; ``reward`` 1.5: at every
    signal, the nearest line of the other kind beyond the close lies at least 1.5 R away."""
    daily, minutes = market()
    result = B.replay(daily, minutes, start=START, costs=FREE, sides="both", target=1.0,
                      skip_open=2, reward=1.5)
    chart = TD.candles(daily, minutes, "15m", TD.clock(daily))
    trades = result["trades"]
    assert len(trades) >= 3 and "target hit" in set(trades["reason"])
    o, c, x = (chart[k].to_numpy() for k in ("open", "close", "x"))
    for t in trades.rename(columns=lambda name: name.replace(" ", "_")).itertuples():
        d = 1 if t.side == "long" else -1
        if t.reason == "target hit":
            goal = t.entry + d * abs(t.entry - t.initial_stop)
            assert t.exit == pytest.approx(max(o[t.exit_k], goal) if d > 0
                                           else min(o[t.exit_k], goal))
            assert t.R >= 1 - 1e-9
        s = t.entry_k - 1
        signal = chart.index[s]
        assert (signal.hour * 60 + signal.minute - 570) // 15 >= 2
        line = t.a_y + t.slope * (x[s] - t.a_x)
        risk = d * (c[s] - line) + t.room
        other = result["lines"][t.setup]["short" if d > 0 else "long"]
        ahead = d * (other["a_y"] + other["slope"] * (x[s] - other["a_x"]) - c[s])
        ahead = ahead[ahead > 0]
        assert not len(ahead) or ahead.min() >= 1.5 * risk - 1e-9


def test_own_lines_and_the_anchor_gap_keep_to_their_rules():
    """``own_lines``: every trade bounced off a line drawn on the 15-minute chart.
    ``anchor_gap`` 16: every 15-minute line drawn at any setup has its points A and B at least
    16 candles (4 hours) apart, while without the gap some are closer (the knob bites)."""
    daily, minutes = market()
    free = B.replay(daily, minutes, start=START, costs=FREE, sides="both")
    result = B.replay(daily, minutes, start=START, costs=FREE, sides="both", own_lines=True,
                      anchor_gap=16)
    trades = result["trades"]
    assert len(trades) >= 3 and set(trades["line drawn on"]) == {"15m"}

    def apart(drawn):
        """Candles between points A and B of every 15-minute line of every setup."""
        lines = pd.concat([v for setup in drawn for v in setup.values()])
        lines = lines[lines["tf"] == "15m"]
        return (lines["b_x"] - lines["a_x"]) * 26

    assert (apart(result["lines"]) >= 16 - 1e-9).all()
    assert (apart(free["lines"]) < 16).any()
    for t in trades.itertuples(index=False):
        drawn = result["lines"][t.setup][t.side]
        assert ((drawn["a_time"] == t[trades.columns.get_loc("line A")])
                & (drawn["tf"] == "15m")).any()


def test_the_cascade_trades_only_its_one_fifteen_minute_line():
    """``method="cascade"`` with 15-minute lines only: every setup holds at most one line per
    chart and side, each trade bounced off its setup's 15-minute line, and every line is at
    least its chart's minimum long. Its one 15-minute line is often far from price, so it
    trades rarely: it gets a longer market."""
    daily, minutes = market(sessions=120, seed=5)
    least = {"1D": 3, "4H": 2, "1H": 2, "15m": 8}
    result = B.replay(daily, minutes, start=START, costs=FREE, sides="both", method="cascade",
                      minimum=least, own_lines=True, screen=40)
    trades = result["trades"]
    assert len(trades) >= 2 and set(trades["line drawn on"]) == {"15m"}
    per = {"1D": 1, **{tf: TD.SESSION / m for tf, m in TD.MINUTES.items()}}
    for setup in result["lines"]:
        for lines in setup.values():
            assert lines["tf"].is_unique
            assert ((lines["b_x"] - lines["a_x"]) * lines["tf"].map(per)
                    >= lines["tf"].map(least) - 1e-9).all()
    for t in trades.itertuples(index=False):
        drawn = result["lines"][t.setup][t.side]
        mine = drawn[drawn["tf"] == "15m"]
        assert len(mine) == 1 and mine["a_time"].iloc[0] == t[trades.columns.get_loc("line A")]


@pytest.mark.parametrize("entry", ["next_open", "limit"])
def test_no_position_opens_or_closes_on_the_session_edges(entry):
    """``trade_trim`` (3, 3): without it some trades open or close on a session's first or last
    three 15-minute candles; with it no signal, entry or exit does (a close beyond the line on
    them exits at the first candle after them; a trade still open at the data's end is only
    valued there). With ``draw_trim`` too, trades still bounce off the 15-minute line."""
    daily, minutes = market(sessions=120, seed=5)
    knobs = {"entry": entry, "method": "cascade", "screen": 40, "own_lines": True}

    def on_edges(times):
        return TD.session_edges(pd.DatetimeIndex(times), "15m", 3, 3)

    free = B.replay(daily, minutes, start=START, costs=FREE, sides="long", **knobs)["trades"]
    assert on_edges(free["entry time"]).any() or on_edges(free["exit time"]).any()
    for draw in ((0, 0), (2, 2)):
        trades = B.replay(daily, minutes, start=START, costs=FREE, sides="long",
                          trade_trim=(3, 3), draw_trim=draw, **knobs)["trades"]
        assert len(trades) >= 3 and set(trades["line drawn on"]) == {"15m"}
        done = trades[trades["reason"] != "still open at the last candle"]
        assert not on_edges(trades["signal time"]).any()
        assert not on_edges(trades["entry time"]).any()
        assert not on_edges(done["exit time"]).any()


def test_a_watched_ray_trades_only_once_its_angle_candle_is_confirmed():
    """``watch``: some trades bounce off a 15-minute ray that was still pending when the lines
    were drawn (its angle candle among the newest ``confirm``); every trade's signal comes at
    least ``confirm`` candles after its line's angle candle. Without ``watch`` no trade uses a
    line pending at its setup."""
    daily, minutes = market(sessions=120, seed=5)
    chart = TD.candles(daily, minutes, "15m", TD.clock(daily))
    knobs = {"method": "cascade", "screen": 40, "own_lines": True, "confirm": 2}

    def pending_at_setup(result):
        """Per trade: was its line's angle candle too new when the lines were drawn?"""
        t, setups = result["trades"], result["setups"]
        made = chart.index.get_indexer(setups["time"].iloc[t["setup"]])
        angle = chart.index.get_indexer(t["line B"])
        return angle > made - 2, angle, chart.index.get_indexer(t["signal time"])

    plain = B.replay(daily, minutes, start=START, costs=FREE, sides="both", **knobs)
    assert len(plain["trades"]) and not pending_at_setup(plain)[0].any()
    watched = B.replay(daily, minutes, start=START, costs=FREE, sides="both", watch=True,
                       **knobs)
    late, angle, signal = pending_at_setup(watched)
    assert late.any()
    assert (signal >= angle + 2).all()


@pytest.mark.parametrize("sides", ["long", "short"])
def test_a_bounce_is_confirmed_by_hold_closes_on_the_lines_side(sides):
    """``hold`` 2: every trade enters at the open three candles after its signal candle, and
    the two candles in between closed on the line's side; with ``hold`` 0 some trades take a
    bounce that the next candles did not confirm."""
    daily, minutes = market()
    if sides == "short":
        daily, minutes = upside_down(daily), upside_down(minutes)
    chart = TD.candles(daily, minutes, "15m", TD.clock(daily))
    c, x = chart["close"].to_numpy(), chart["x"].to_numpy()

    def held(trades, n):
        """Per trade: did the ``n`` candles after its signal close on its line's side?"""
        out = []
        for t in trades.itertuples(index=False):
            d = 1 if t.side == "long" else -1
            s = chart.index.get_loc(t[trades.columns.get_loc("signal time")])
            after = np.arange(s + 1, s + 1 + n)
            line = t.a_y + t.slope * (x[after] - t.a_x)
            out.append(bool((d * (c[after] - line) >= 0).all()))
        return np.array(out)

    plain = B.replay(daily, minutes, start=START, costs=FREE, sides=sides)["trades"]
    held2 = B.replay(daily, minutes, start=START, costs=FREE, sides=sides, hold=2)["trades"]
    assert len(held2) >= 2 and not held(plain, 2).all()
    assert held(held2, 2).all()
    signal = chart.index.get_indexer(held2["signal time"])
    assert (held2["entry_k"].to_numpy() == signal + 3).all()


def test_with_noise_bands_trades_keep_to_the_contact_zone():
    """Line band 0.3 and price band 0.2, the noise over 20 candles: every signal candle reached
    into the zone (0.5 x the noise known at its open) around its line and closed on the line's
    side; every first stop lies beyond the zone; every exit on a close came after a close
    beyond the zone (out at the next open)."""
    daily, minutes = market(sessions=120, seed=5)
    chart = TD.candles(daily, minutes, "15m", TD.clock(daily))
    n = TD.noise(chart, 20)
    zone = 0.5 * np.r_[n[:1], n[:-1]]
    lo, c, x = (chart[k].to_numpy() for k in ("low", "close", "x"))
    result = B.replay(daily, minutes, start=START, costs=FREE, sides="long", method="cascade",
                      screen=40, own_lines=True, line_band=0.3, price_band=0.2,
                      noise_window=20)
    trades = result["trades"]
    assert len(trades) >= 3
    for t in trades.rename(columns=lambda name: name.replace(" ", "_")).itertuples():
        s = chart.index.get_loc(t.signal_time)
        line = t.a_y + t.slope * (x[s] - t.a_x)
        assert lo[s] - line <= zone[s] + 1e-9 and c[s] >= line - 1e-9
        assert t.initial_stop <= t.a_y + t.slope * (x[t.entry_k] - t.a_x) - zone[s] + 1e-9
        if t.reason == "closed below the line":
            k = t.exit_k - 1
            assert c[k] < t.a_y + t.slope * (x[k] - t.a_x) - zone[k] + 1e-9


def test_limit_entries_fill_at_the_line():
    """``entry="limit"``: the order sits at the line plus the entry gap; the fill candle opened
    on the line's side and reached the order, the fill is the order price (or the open when it
    opened inside the band), the first stop sits ``room`` beyond the line, a stop on the fill
    candle fills exactly at the stop, and a target is never taken on the fill candle."""
    daily, minutes = market()
    result = B.replay(daily, minutes, start=START, costs=FREE, sides="both", entry="limit",
                      band=0.25, target=1.0)
    chart = TD.candles(daily, minutes, "15m", TD.clock(daily))
    o, hi, lo, x = (chart[k].to_numpy() for k in ("open", "high", "low", "x"))
    trades = result["trades"].rename(columns=lambda name: name.replace(" ", "_"))
    assert len(trades) >= 3 and {"stop hit", "target hit"} <= set(trades["reason"])
    for t in trades.itertuples():
        d, ext = (1, lo) if t.side == "long" else (-1, hi)
        k = t.entry_k
        line = t.a_y + t.slope * (x[k] - t.a_x)
        level = line + d * t.band
        assert t.signal_time == t.entry_time
        assert d * (o[k] - line) > 0 and d * (ext[k] - level) <= 1e-9
        assert t.entry == pytest.approx(level if d * (o[k] - level) > 0 else o[k])
        assert t.initial_stop == pytest.approx(line - d * t.room)
        if t.reason == "target hit":
            assert t.exit_k > k
        if t.reason == "stop hit" and t.exit_k == k:
            assert t.exit == pytest.approx(t.initial_stop)


def test_without_refresh_lines_are_drawn_again_only_after_a_trade_or_a_break():
    """With ``refresh`` off, a session-close setup only happens when there was no line."""
    daily, minutes = market()
    setups = B.replay(daily, minutes, start=START, costs=COSTS, refresh=False)["setups"]
    again = setups.index[setups["reason"] == B.SESSION]
    assert (setups.loc[again - 1, "long lines"] == 0).all()
