import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tori Trades · the trendline bounce, traded (long and short)

    This notebook trades the trendline bounce from Tori Trades' video
    [*Once you master this strategy, trading becomes EASY*](https://www.youtube.com/watch?v=TuXOgkcYw9E)
    on one stock, candle by candle, from a start date (six months before the stock's last
    1-minute candle by default) to its last candle. A **long** trades the upward lines under the
    lows, a **short** the downward lines over the highs; each is the mirror of the other. The lines
    are drawn by the rules of the visual notebook (`topdown.py`); the trading is `bounce.py`. Each
    round goes like this:

    1. **Set up.** Draw the lines with her top-down analysis (monthly down to the trading chart),
       using only the candles known at that moment. The higher charts' last candle is still
       forming, as on a live chart.
    2. **Wait** for price to come to a line. Long: a candle's low comes within the *entry gap* of
       an upward line and the candle closes above it. Short: a candle's high comes within the
       *entry gap* of a downward line and the candle closes below it. **Enter at the next
       candle's open**, on the line price meets first (the *action line*). A long and a short
       signal on the same candle is no trade. A candle that closes beyond a traded line breaks
       it: set up again. While waiting, the lines are also drawn again at every session's close
       (the switch below).
    3. **Manage.** The first stop sits *stop room* typical candles beyond the action line (below
       an upward line, above a downward one). Each new swing (a low for a long, a high for a short:
       the extreme of *confirm* candles on each side) moves the stop to just beyond that swing, or
       just beyond the line if the swing lies past it. The stop stays beyond the line and only
       moves in the trade's favour.
    4. **Exit** when the stop is hit (at the stop, or at the open if price gaps through it), or
       when a candle closes more than *close break* typical candles beyond the line (at the next
       open). Then set up again and wait.

    **Money**: a $10,000 wallet, all of it in each trade, profits reinvested; a short is sold for
    the wallet's worth and bought back. Costs are the repo's: $0.005 a share with a $1 minimum,
    half the 2 bp spread and 1 bp slippage, on every order. **The hurdle** is buying the same
    stock at the first traded open and holding it to the end.

    **The session refresh.** Taken literally, "draw the lines after each trade and wait" makes
    the lines go stale. On NOW from 2026-04-16, the upward lines drawn after the April trade lay
    near $82. Price rallied to $149 without coming back, so the long-only replay waited five months
    and the rally's own, steeper lines were never drawn. Switch the refresh off to see that version.
    **Placebo shift** moves every line toward price (negative: away). If her lines mark
    something real, moved lines should trade worse. Two filters are off by default. The **trend
    filter** takes a long only above, and a short only below, an N-session daily average. The
    **shortest line traded** skips lines whose points A and B lie fewer sessions apart.

    **What 21 stocks said** (`study.py`, 2026-03-30 to 2026-09-29, README): at these defaults
    every mode lost (long −9%, short −5%, both −8% a stock over six months, against +20% for
    buy-and-hold). No setting of the entry gap, stop room or close break made either side
    profitable, and lines moved off hers did no worse. The filters only trade less; they do not
    trade better.
    """)
    return


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload bounce.py and topdown.py (after they changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from trading_simple import strategies
    from trading_simple.plots import _LONG as LONG
    from trading_simple.plots import _LOSS as LOSS
    from trading_simple.plots import _MUTED as MUTED
    from trading_simple.plots import _SHORT as SHORT
    from trading_simple.plots import _WIN as WIN
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as base_layout

    _ = reload_helpers.value     # a click reruns this cell: the helper files are read again
    _path = strategies.STRATEGIES_DIR / "tori_trades_trendline" / "bounce.py"
    _spec = importlib.util.spec_from_file_location("tori_trades_bounce", _path)
    assert _spec is not None and _spec.loader is not None, _path
    bounce = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(bounce)
    td = bounce.td                    # topdown.py, as bounce.py loaded it
    return LONG, LOSS, MUTED, SERIES, SHORT, WIN, base_layout, bounce, go, np, pd, td


@app.cell
def _():
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    settings = load_settings()
    engine = make_engine(settings)
    costs = settings.costs
    return costs, engine


@app.cell
def _(mo, td):
    stock = mo.ui.text(value="NOW", label="Stock")
    trading = mo.ui.dropdown(
        options={td.TIMEFRAMES[_k]: _k for _k in ("4H", "1H", "15m", "5m")},
        value="15-minute", label="Trading chart")
    sides = mo.ui.radio(options=["long", "short", "both"], value="both", label="Sides",
                        inline=True)
    scale = mo.ui.radio(options={"plain (TradingView's default)": False, "logarithmic": True},
                        value="plain (TradingView's default)", label="Price scale", inline=True)
    band = mo.ui.slider(0.0, 2.0, step=0.05, value=0.25, show_value=True,
                        label="Entry gap: a low (high) this many typical candles from a line")
    room = mo.ui.slider(0.0, 3.0, step=0.05, value=0.5, show_value=True,
                        label="Stop room: the stop sits this many typical candles beyond the line")
    give = mo.ui.slider(0.0, 3.0, step=0.05, value=0.0, show_value=True,
                        label="Close break: a close this far beyond the line ends the trade")
    shift = mo.ui.slider(-3.0, 3.0, step=0.25, value=0.0, show_value=True,
                         label="Placebo shift: lines moved this many typical candles toward price")
    trend = mo.ui.slider(0, 200, step=10, value=0, show_value=True,
                         label="Trend filter: long above, short below this daily average (0: off)")
    min_span = mo.ui.slider(0, 40, step=1, value=0, show_value=True,
                            label="Shortest line traded: sessions between its points A and B")
    touch = mo.ui.slider(0.0, 1.0, step=0.05, value=0.25, show_value=True,
                         label="Lines' close enough: a touch is within this many typical candles")
    pierce = mo.ui.slider(0.0, 0.5, step=0.05, value=0.1, show_value=True,
                          label="Too small to see: a poke through a line this small is ignored")
    confirm = mo.ui.slider(1, 6, step=1, value=2, show_value=True,
                           label="Confirm: candles after a touch or a swing before it counts")
    screen = mo.ui.slider(50, 400, step=10, value=150, show_value=True,
                          label="Screen: candles in view on every chart after the monthly")
    refresh = mo.ui.checkbox(value=True,
                             label="While waiting, draw the lines again at every session's close")
    mo.vstack([mo.hstack([stock, trading, sides, scale], justify="start", gap=2),
               mo.md("**The trade**"),
               mo.hstack([band, room], justify="start", gap=2),
               mo.hstack([give, shift], justify="start", gap=2),
               mo.hstack([trend, min_span], justify="start", gap=2),
               mo.md("**The lines**"),
               mo.hstack([touch, pierce], justify="start", gap=2),
               mo.hstack([confirm, screen], justify="start", gap=2),
               mo.hstack([refresh], justify="start")])
    return (band, confirm, give, min_span, pierce, refresh, room, scale, screen, shift, sides,
            stock, touch, trading, trend)


@app.cell
def _(engine, mo, stock, td):
    symbol = stock.value.strip().upper()
    try:
        daily, minutes = td.load(engine, symbol)
    except (ValueError, KeyError) as _error:
        mo.stop(True, mo.callout(mo.md(f"No daily candles of **{symbol}** in the database "
                                       f"({_error})."), kind="danger"))
    mo.stop(minutes is None, mo.callout(mo.md(
        f"**{symbol}** has no 1-minute candles: the trading chart is built from them."),
        kind="danger"))
    return daily, minutes, symbol


@app.cell
def _(minutes, mo, pd):
    _first = minutes.index[0].tz_localize(None).normalize() + pd.Timedelta(days=1)
    _last = minutes.index[-1].tz_localize(None).normalize()
    start = mo.ui.date(start=_first.date(), stop=_last.date(),
                       value=max(_last - pd.DateOffset(months=6), _first).date(),
                       label=f"Start: the first session traded (1-minute candles end "
                             f"{_last:%Y-%m-%d})")
    run = mo.ui.run_button(label="run the replay")
    mo.hstack([start, run], justify="start", gap=2)
    return run, start


@app.cell
def _(band, bounce, confirm, costs, daily, give, min_span, minutes, mo, pd, pierce, refresh,
      room, run, scale, screen, shift, sides, start, td, touch, trading, trend):
    mo.stop(not run.value, mo.md("Press **run the replay** (15 to 30 seconds for six months "
                                 "of 15-minute candles)."))
    with mo.status.spinner(title="Replaying the trades ..."):
        result = bounce.replay(
            daily, minutes, start=pd.Timestamp(start.value), costs=costs, tf=trading.value,
            sides=sides.value, log=scale.value, touch=touch.value, pierce=pierce.value,
            confirm=confirm.value, screen=screen.value, band=band.value, room=room.value,
            give=give.value, shift=shift.value, refresh=refresh.value, trend=trend.value,
            min_span=float(min_span.value))
        # every candle of the trading chart, from the first stored minute: the history the
        # close-ups show (the replay's own chart starts at the first traded candle)
        bars = td.candles(daily, minutes, trading.value, td.clock(daily))
    log, tf = scale.value, trading.value
    return bars, log, result, tf


@app.cell
def _(LONG, LOSS, MUTED, SERIES, SHORT, WIN, base_layout, go, np, pd, td):
    COLOR = {"long": LONG, "short": SHORT}
    BAND = {"long": "rgba(42, 120, 214, 0.14)", "short": "rgba(235, 104, 52, 0.14)"}
    LINES = {"long": "upward", "short": "downward"}
    STOP, CANDLE = SERIES[2], "#7d7b75"

    def stamp(t):
        """A candle's time as text: its date, plus the clock time for an intraday candle."""
        t = pd.Timestamp(t)
        return f"{t:%Y-%m-%d}" if t == t.normalize() else f"{t:%Y-%m-%d %H:%M}"

    def to_price(v, log):
        """Values in the lines' scale as prices."""
        return np.exp(v) if log else np.asarray(v, dtype=float)

    def line_at(a_x, a_y, slope, x, log):
        """A line's price at clock positions ``x``."""
        return to_price(a_y + slope * (np.asarray(x, dtype=float) - a_x), log)

    def candle_trace(bars, name):
        """Candles in one color: hollow when the close is above the open, filled when below."""
        return go.Candlestick(
            x=bars["x"], open=bars["open"], high=bars["high"], low=bars["low"],
            close=bars["close"], text=[stamp(t) for t in bars.index], name=name,
            increasing={"line": {"color": CANDLE, "width": 1}, "fillcolor": "white"},
            decreasing={"line": {"color": CANDLE, "width": 1}, "fillcolor": CANDLE})

    def finish(fig, bars, fmt, log, *, title, height, prices=True):
        """The shared look, the legend under the plot, dates on the session-clock axis and,
        for a candle chart, the price range of its candles."""
        base_layout(fig, title=title, height=height)
        plot = max(height - 50 - 150, 80)
        fig.update_layout(margin={"b": 150}, hovermode="closest", legend={
            "orientation": "h", "x": 0.0, "xanchor": "left", "y": -60 / plot, "yanchor": "top",
            "entrywidth": 0.33, "entrywidthmode": "fraction"})
        pick = np.unique(np.linspace(0, len(bars) - 1, 9).round().astype(int))
        fig.update_xaxes(tickvals=bars["x"].to_numpy()[pick],
                         ticktext=[f"{t:{fmt}}" for t in bars.index[pick]],
                         rangeslider_visible=False)
        if prices:
            lo, hi = float(bars["low"].min()), float(bars["high"].max())
            if log:
                lo, hi = np.log10(lo), np.log10(hi)
            fig.update_yaxes(type="log" if log else "linear",
                             range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        return fig

    def named(trades):
        """The trades with column names usable as attributes."""
        return trades.rename(columns=lambda name: name.replace(" ", "_"))

    def exit_traces(fig, trades):
        """The exits as crosses, green with a profit and red with a loss."""
        for won, color, name in ((True, WIN, "exit with a profit"),
                                 (False, LOSS, "exit with a loss")):
            part = named(trades[(trades["return"] > 0) == won])
            fig.add_trace(go.Scatter(
                x=part["exit_x"], y=part["exit"], mode="markers", name=name,
                marker={"symbol": "x", "size": 10, "color": color},
                hovertext=[f"trade {t.trade + 1} ({t.side}): {t.reason}, {stamp(t.exit_time)} "
                           f"at {t.exit:.2f}<br>{t.R:+.2f} R" for t in part.itertuples()],
                hoverinfo="text"))

    def equity_chart(result, *, title):
        """The wallet at every close: the strategy against buying and holding."""
        eq = result["equity"]
        fig = go.Figure()
        for name, color in (("strategy", LONG), ("buy and hold", MUTED)):
            fig.add_trace(go.Scatter(
                x=eq["x"], y=eq[name], mode="lines", name=name,
                line={"color": color, "width": 1.8}, hoverinfo="text",
                hovertext=[f"{name}, {stamp(t)}: ${v:,.0f}"
                           for t, v in zip(eq.index, eq[name], strict=True)]))
        fig = finish(fig, eq, "%Y-%m-%d", False, title=title, height=420, prices=False)
        fig.update_yaxes(title_text="wallet ($)")
        return fig

    def trades_chart(result, log, *, title):
        """Every trade on the trading chart: its action line from the signal candle to the
        exit, the entry, the stop at every close, the exit."""
        bars, stops = result["chart"], result["stops"]
        trades = result["trades"].copy()
        for key in ("signal", "entry", "exit"):
            trades[f"{key}_x"] = bars["x"].reindex(trades[f"{key} time"]).to_numpy()
        fig = go.Figure(candle_trace(bars, "candles"))
        for side, part in trades.groupby("side"):
            xs, ys = [], []
            for t in named(part).itertuples():
                xs += [t.signal_x, t.exit_x, None]
                ys += [*line_at(t.a_x, t.a_y, t.slope, [t.signal_x, t.exit_x], log), None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     name=f"action line of each {side}",
                                     line={"color": COLOR[side], "width": 2}))
        if len(stops):
            xs, ys = [], []
            for _, path in stops.groupby("trade"):
                xs += [*path["x"], None]
                ys += [*path["stop"], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="stop at each close",
                                     line={"color": STOP, "width": 1.5, "shape": "hv"},
                                     hoverinfo="skip"))
        for side, part in trades.groupby("side"):
            fig.add_trace(go.Scatter(
                x=part["entry_x"], y=part["entry"], mode="markers",
                name=f"{side} entry (next open)", hoverinfo="text",
                marker={"symbol": "triangle-up" if side == "long" else "triangle-down",
                        "size": 11, "color": COLOR[side]},
                hovertext=[f"trade {t.trade + 1}: {side} {stamp(t.entry_time)} at {t.entry:.2f},"
                           f" stop {t.initial_stop:.2f}" for t in named(part).itertuples()]))
        exit_traces(fig, trades)
        return finish(fig, bars, "%Y-%m-%d", log, title=title, height=640)

    def trade_chart(result, bars, i, log, tf, before, *, title):
        """One trade up close, on ``bars`` (every candle of the trading chart): ``before``
        candles before the lines were drawn, through the exit; every line drawn at that moment
        (upward and downward), the band that triggered the entry, the entry, the stop at every
        close and the exit. Returns the figure and the story in words."""
        t = named(result["trades"]).iloc[i]
        side, d = t["side"], 1.0 if t["side"] == "long" else -1.0
        setup, drawn = result["setups"].iloc[t["setup"]], result["lines"][t["setup"]]
        k0 = int(bars.index.searchsorted(min(setup["time"], t["signal_time"])))
        k1 = int(bars.index.searchsorted(t["exit_time"]))
        after = max(int(390 / td.MINUTES[tf]) + 1, (k1 - k0) // 5)   # a session at least
        shown = bars.iloc[max(k0 - before, 0):min(k1 + after, len(bars) - 1) + 1]
        x0, x1 = float(shown["x"].iloc[0]), float(shown["x"].iloc[-1])
        sx, ex, xx = (float(bars.loc[t[f"{k}_time"], "x"]) for k in ("signal", "entry", "exit"))
        fig = go.Figure(candle_trace(shown, f"{td.TIMEFRAMES[tf]} candles"))
        for kind, lines in drawn.items():
            xs, ys, tips = [], [], []
            for line in lines.itertuples():
                left = max(line.a_x, x0)
                tip = (f"{LINES[kind]} line drawn on the {td.TIMEFRAMES[line.tf]} chart<br>"
                       f"point A {stamp(line.a_time)} at {line.a_price:.2f}, point B "
                       f"{stamp(line.b_time)} at {line.b_price:.2f}")
                xs += [left, x1, None]
                ys += [*line_at(line.a_x, line.a_y, line.slope, [left, x1], log), None]
                tips += [tip, tip, None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", hovertext=tips, hoverinfo="text",
                name=f"{LINES[kind]} lines drawn at setup {t['setup'] + 1}",
                line={"color": COLOR[kind], "width": 1.2}, opacity=0.6))
            # the candles that fixed each line: point A, the candle that stopped the ray (its
            # angle), point B (the latest touch, the next line's A)
            points = []
            for n, line in enumerate(lines.itertuples(), start=1):
                head = f"{LINES[kind]} line {n} (drawn on the {td.TIMEFRAMES[line.tf]} chart)"
                points += [
                    (line.a_x, line.a_price, "diamond",
                     f"{head}<br>point A: {stamp(line.a_time)} at {line.a_price:.2f}"),
                    (line.stop_x, line.stop_price, "star",
                     f"{head}<br>the candle that stopped the ray: {stamp(line.stop_time)} at "
                     f"{line.stop_price:.2f}"),
                    (line.b_x, line.b_price, "diamond-open",
                     f"{head}<br>point B, the latest touch: {stamp(line.b_time)} at "
                     f"{line.b_price:.2f}")]
            points = [p for p in points if x0 <= p[0] <= x1]
            if points:
                px, py, symbols, notes = zip(*points, strict=True)
                fig.add_trace(go.Scatter(
                    x=px, y=py, mode="markers", hovertext=notes, hoverinfo="text",
                    name=f"{LINES[kind]} lines' anchors: ◆ A · ★ stopping candle · ◇ B",
                    marker={"symbol": list(symbols), "size": 10, "color": COLOR[kind],
                            "line": {"color": COLOR[kind], "width": 1.5}}))
        span = [max(float(setup["x"]), x0), ex]
        edge = t["a_y"] + t["slope"] * (np.asarray(span) - t["a_x"])
        fig.add_trace(go.Scatter(
            x=[*span, *span[::-1], span[0]],
            y=to_price([*edge, *(edge + d * t["band"])[::-1], edge[0]], log), mode="lines",
            fill="toself", fillcolor=BAND[side], line={"width": 0}, hoverinfo="skip",
            name="entry gap: a low (high) in here is a signal"))
        left = max(t["a_x"], x0)
        fig.add_trace(go.Scatter(
            x=[left, xx], y=line_at(t["a_x"], t["a_y"], t["slope"], [left, xx], log),
            mode="lines", name="the action line", hoverinfo="skip",
            line={"color": COLOR[side], "width": 3}))
        lo, hi = float(shown["low"].min()), float(shown["high"].max())
        fig.add_trace(go.Scatter(x=[setup["x"]] * 2, y=[lo, hi], mode="lines",
                                 name="the lines were drawn here", hoverinfo="skip",
                                 line={"color": MUTED, "width": 1, "dash": "dot"}))
        path = result["stops"][result["stops"]["trade"] == t["trade"]] if len(
            result["stops"]) else result["stops"]
        if len(path):
            fig.add_trace(go.Scatter(x=path["x"], y=path["stop"], mode="lines",
                                     name="stop at each close", hoverinfo="skip",
                                     line={"color": STOP, "width": 1.5, "shape": "hv"}))
        tested = "low" if side == "long" else "high"
        fig.add_trace(go.Scatter(
            x=[sx], y=[bars.loc[t["signal_time"], tested]], mode="markers",
            name=f"signal: the {tested} came to the line, the close held",
            marker={"symbol": "circle-open", "size": 13, "color": COLOR[side],
                    "line": {"color": COLOR[side], "width": 2}}, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[ex], y=[t["entry"]], mode="markers", name=f"{side} entry (next open)",
            hoverinfo="skip", marker={"symbol": "triangle-up" if d > 0 else "triangle-down",
                                      "size": 13, "color": COLOR[side]}))
        fig.add_trace(go.Scatter(
            x=[xx], y=[t["exit"]], mode="markers", name=f"exit: {t['reason']}",
            hoverinfo="skip", marker={"symbol": "x", "size": 12,
                                      "color": WIN if t["return"] > 0 else LOSS}))
        risk = abs(t["entry"] - t["initial_stop"])
        band = (f"{np.expm1(t['band']):.2%}" if log else f"{t['band']:.2f}")
        how = {"stop hit": "The stop was hit",
               "closed below the line": "A candle closed below the line; out at the next open",
               "closed above the line": "A candle closed above the line; out at the next open",
               }.get(t["reason"], "Still open at the last candle, valued at its close")
        verb, held = ("bought", "above") if d > 0 else ("sold short", "below")
        story = (
            f"**Trade {i + 1} ({side}).** The lines were drawn at {stamp(setup['time'])} "
            f"({setup['reason']}): {len(drawn['long'])} upward and {len(drawn['short'])} "
            f"downward line(s), all on the chart (the thin lines). At "
            f"{stamp(t['signal_time'])} the candle's {tested} came within {band} of the "
            f"{LINES[side]} line drawn on the {td.TIMEFRAMES[t['line_drawn_on']]} chart (point A "
            f"{stamp(t['line_A'])} at {t['line_A_price']:.2f}) and the candle closed {held} it: "
            f"{verb} at the next open, {stamp(t['entry_time'])} at {t['entry']:.2f}, first stop "
            f"{t['initial_stop']:.2f} ({risk:.2f} away: 1 R). {how}: {stamp(t['exit_time'])} at "
            f"{t['exit']:.2f}, **{t['R']:+.2f} R**, {t['return']:+.2%} after costs, "
            f"{t['candles']} candles.")
        return finish(fig, shown, "%m-%d %H:%M", log, title=title, height=580), story

    return equity_chart, stamp, trade_chart, trades_chart


@app.cell(hide_code=True)
def _(equity_chart, mo, pd, result, symbol):
    _s, _t = result["summary"], result["trades"]
    _rows = pd.DataFrame({
        "": ["return over the window", "per year (CAGR)", "worst drop (max drawdown)"],
        "strategy": [f"{_s['strategy return']:+.1%}", f"{_s['strategy CAGR']:+.1%}",
                     f"{_s['strategy max drawdown']:.1%}"],
        "buy and hold": [f"{_s['buy and hold return']:+.1%}", f"{_s['buy and hold CAGR']:+.1%}",
                         f"{_s['buy and hold max drawdown']:.1%}"]})
    _sides = (" · ".join(
        f"{_side}: {len(_p)} trades, {(_p['return'] > 0).sum()} with a profit, average "
        f"{_p['R'].mean():+.2f} R, compounded {(1 + _p['return']).prod() - 1:+.1%}"
        for _side, _p in _t.groupby("side")) if len(_t) else "no trade")
    _setups = ", ".join(f"{_n} × {_r}"
                        for _r, _n in result["setups"]["reason"].value_counts().items())
    mo.vstack([mo.md(f"""
    ## 1 · The result

    {symbol} from {result['chart'].index[0]:%Y-%m-%d} to {result['chart'].index[-1]:%Y-%m-%d}
    ({_s['years']:.2f} years). {_sides}. In the market {_s.get('time in market', 0):.0%} of the
    time; the lines were drawn {len(result['setups'])} times ({_setups}). An R is a trade's first
    risk: the distance from entry to first stop. Over a window this short the per-year rates are
    noise, not a forecast.
    """), mo.ui.table(_rows, selection=None, show_download=False),
        equity_chart(result, title=f"{symbol} · the wallet at every close")])
    return


@app.cell(hide_code=True)
def _(log, mo, result, symbol, td, tf, trades_chart):
    mo.stop(result["trades"].empty, mo.md("*No trade in this window.*"))
    mo.vstack([mo.md("""
    ## 2 · Every trade on the chart

    Each trade's action line runs from the candle that signalled to the exit: blue for a long,
    orange for a short. Triangles are entries, pointing the trade's way; crosses are exits (green
    with a profit, red with a loss); amber is the stop at every close. Zoom in with the mouse;
    hover over an entry or an exit for the trade.
    """), trades_chart(result, log,
                       title=f"{symbol} · {td.TIMEFRAMES[tf]} chart · {len(result['trades'])} "
                             "trades")])
    return


@app.cell(hide_code=True)
def _(mo, result, stamp):
    mo.stop(result["trades"].empty, mo.md("*No trade in this window.*"))
    pick_trade = mo.ui.dropdown(
        options={f"trade {_t + 1} · {_s} · {stamp(_e)} · {_r} · {_R:+.2f} R": _t
                 for _t, _s, _e, _r, _R in zip(
                     result["trades"]["trade"], result["trades"]["side"],
                     result["trades"]["entry time"], result["trades"]["reason"],
                     result["trades"]["R"], strict=True)},
        value=None, label="Trade to look at")
    before = mo.ui.slider(50, 2000, step=50, value=400, show_value=True,
                          label="History: candles shown before the lines were drawn")
    mo.vstack([mo.md("""
    ## 3 · One trade up close

    Pick a trade (the first one by default). The chart shows the history before the moment the
    lines were drawn (the dotted line; the slider sets how much) and **every line drawn at that
    moment**: upward lines blue, downward lines orange, each with the candles that fixed it:
    ◆ point A where the ray starts, ★ the candle that stopped the ray (its angle), ◇ point B,
    the latest touch, where the next line starts. Over them: the band where a low (high) is a
    signal, the action line (thick), the signal candle, the entry, the stop at every close and
    the exit. Hover over a line or a point for its details.
    """), mo.hstack([pick_trade, before], justify="start", gap=2)])
    return before, pick_trade


@app.cell(hide_code=True)
def _(bars, before, log, mo, pick_trade, result, symbol, tf, trade_chart):
    _i = 0 if pick_trade.value is None else pick_trade.value
    _fig, _story = trade_chart(result, bars, _i, log, tf, before.value,
                               title=f"{symbol} · trade {_i + 1}")
    mo.vstack([mo.md(_story), _fig])
    return


@app.cell(hide_code=True)
def _(mo, pd, result, stamp, td):
    _t = result["trades"]
    _trades = pd.DataFrame({
        "trade": _t["trade"] + 1, "side": _t["side"], "signal": _t["signal time"].map(stamp),
        "entry time": _t["entry time"].map(stamp), "entry": _t["entry"].round(2),
        "first stop": _t["initial stop"].round(2), "exit time": _t["exit time"].map(stamp),
        "exit": _t["exit"].round(2), "why": _t["reason"], "R": _t["R"].round(2),
        "return": (100 * _t["return"]).round(2).astype(str) + "%", "candles": _t["candles"],
        "line drawn on": _t["line drawn on"].map(td.TIMEFRAMES),
        "line A": _t["line A"].map(stamp)}) if len(_t) else _t
    _setups = result["setups"].assign(setup=lambda f: f["setup"] + 1,
                                      time=lambda f: f["time"].map(stamp)).drop(columns="x")
    mo.vstack([mo.md("## 4 · The trades and the setups\n\nEvery trade:"),
               mo.ui.table(_trades, selection=None, page_size=15, show_download=False),
               mo.md("Every time the lines were drawn, why, and how many lines each traded side "
                     "had:"),
               mo.ui.table(_setups, selection=None, page_size=15, show_download=False)])
    return


if __name__ == "__main__":
    app.run()
