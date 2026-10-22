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
    # Tori Trades · top-down trendlines

    This notebook draws the trendlines that **Tori Trades** teaches in her video
    [*Once you master this strategy, trading becomes EASY*](https://www.youtube.com/watch?v=TuXOgkcYw9E)
    (2026-09-24) automatically, for any stock in the database. It only draws the lines: no entries,
    no exits, no backtest (her trade rules are parked in `IDEAS.md`).

    What she does by hand, and *the exact rule used here* (`topdown.py`):

    1. **Rays.** She draws every line with TradingView's ray tool: point A, point B, extended to
       the right; downward lines connect swing highs, upward lines swing lows. *A line is a ray
       from point A to the right edge; downward lines rest on the candles' highs, upward lines
       under their lows.*
    2. **Start at the extreme.** She starts on the monthly chart with all history in view: the
       first downward line starts at the highest point on the chart, the first upward line at the
       lowest. *The same, over every daily candle the database holds (section 1).*
    3. **Price never cuts through a line.** She turns the ray as far as it goes and catches as
       many touches as possible. *The steepest ray from point A that touches a candle and that no
       later candle goes through; a poke too small to see does not count (the "too small to see"
       slider). Of all the rays from A that no candle crosses, this one also passes closest to
       every candle, so it has the most touches.*
    4. **Close enough is a touch.** *A candle whose high (low) comes within "close enough" typical
       candles of the line touches it; a typical candle is the median high-to-low range of the
       candles on the chart. A touch counts once "confirm" candles have followed it.*
    5. **Lines connect.** Each new line starts at the previous line's point B, its most recent
       touch, and is steeper. *Point B is the most recent touch, and the next line starts there.
       Every later candle stays further from the line than B, so the next line is always steeper.
       A ray stopped by a candle that is still too new is not a line yet: it is drawn dashed
       (pending).*
    6. **Top-down.** Then the weekly, daily, 4-hour and 1-hour charts, and finally the trading
       chart (15 minutes in her example, anything from 5 minutes to 1 hour); on each lower chart
       she tidies the lines and adds the steeper ones it shows. *Every chart after the monthly
       shows its last "screen" candles. The walk restarts at the last point A or B left of the
       screen, moved onto the candle of this chart that holds its price: that redraws the lines
       reaching onto the screen with the precise candles and adds the steeper lines only this
       chart can see. Lines that end left of the screen keep their drawing from the higher chart,
       unless a candle on this screen goes through one: then it is drawn again too.*

    **Time** runs on one trading-session clock on every chart. Session *n* starts at *n*, and a
    15-minute candle opening at 10:00 sits at *n* + 30/390. Nights, weekends and holidays take no
    room (as on a TradingView chart), so a line has one slope on every chart. A monthly or weekly
    candle sits at its first session, as a chart places a candle at its opening time. The day of its
    high only shows on a lower chart: that is why the lower charts adjust the lines.
    """)
    return


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload topdown.py (after it changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from trading_simple import strategies
    from trading_simple.plots import _MUTED as MUTED
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as base_layout

    _ = reload_helpers.value     # a click reruns this cell: topdown.py is read again
    _path = strategies.STRATEGIES_DIR / "tori_trades_trendline" / "topdown.py"
    _spec = importlib.util.spec_from_file_location("tori_trades_topdown", _path)
    assert _spec is not None and _spec.loader is not None, _path
    td = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(td)
    return MUTED, SERIES, base_layout, go, np, pd, td


@app.cell
def _():
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    engine = make_engine(load_settings())
    return (engine,)


@app.cell
def _(mo, td):
    stock = mo.ui.text(value="NOW", label="Stock")
    trading = mo.ui.dropdown(
        options={td.TIMEFRAMES[_k]: _k for _k in ("4H", "1H", "15m", "5m")},
        value="15-minute", label="Trading chart (the last chart of the walk)")
    scale = mo.ui.radio(options={"plain (TradingView's default)": False, "logarithmic": True},
                        value="plain (TradingView's default)", label="Price scale", inline=True)
    touch = mo.ui.slider(0.0, 1.0, step=0.05, value=0.25, show_value=True,
                         label="Close enough: a touch comes within this many typical candles")
    pierce = mo.ui.slider(0.0, 0.5, step=0.05, value=0.1, show_value=True,
                          label="Too small to see: a candle may poke through by this many")
    confirm = mo.ui.slider(1, 6, step=1, value=2, show_value=True,
                           label="Confirm: candles that must follow a touch before it counts")
    screen = mo.ui.slider(50, 400, step=10, value=150, show_value=True,
                          label="Screen: candles in view on every chart after the monthly")
    mo.vstack([mo.hstack([stock, trading, scale], justify="start", gap=2),
               mo.hstack([touch, pierce], justify="start", gap=2),
               mo.hstack([confirm, screen], justify="start", gap=2)])
    return confirm, pierce, scale, screen, stock, touch, trading


@app.cell
def _(engine, mo, stock, td):
    symbol = stock.value.strip().upper()
    try:
        daily, minutes = td.load(engine, symbol)
    except (ValueError, KeyError) as _error:
        mo.stop(True, mo.callout(mo.md(f"No daily candles of **{symbol}** in the database "
                                       f"({_error})."), kind="danger"))
    sessions = td.clock(daily)
    return daily, minutes, sessions, symbol


@app.cell
def _(daily, mo):
    asof = mo.ui.date(start=daily.index[0].date(), stop=daily.index[-1].date(),
                      value=daily.index[-1].date(),
                      label="As of: the lines see the candles up to this session's close")
    mo.hstack([asof], justify="start")
    return (asof,)


@app.cell
def _(asof, confirm, daily, minutes, pd, pierce, scale, screen, sessions, td, touch, trading):
    day = pd.Timestamp(asof.value)
    log = scale.value
    _daily = daily[daily.index <= day]
    _minutes = (None if minutes is None
                else minutes[minutes.index.tz_localize(None).normalize() <= day])
    _order = list(td.TIMEFRAMES)
    charts = _order[:3]                  # monthly, weekly, daily
    if _minutes is not None and len(_minutes):
        charts += _order[3:_order.index(trading.value) + 1]
    frames = {_tf: td.candles(_daily, _minutes, _tf, sessions) for _tf in charts}
    walk = {_side: td.top_down(frames, _side, log=log, touch=touch.value, pierce=pierce.value,
                               confirm=confirm.value, screen=screen.value)
            for _side in td.SIDES}
    return charts, day, frames, log, walk


@app.cell
def _(MUTED, SERIES, base_layout, go, np, pd, td):
    COLOR = {"down": SERIES[5], "up": SERIES[0]}       # downward lines red, upward lines blue
    WORD = {"down": "downward", "up": "upward"}
    WIDTH = dict(zip(td.TIMEFRAMES, (3.6, 3.0, 2.5, 2.1, 1.8, 1.5, 1.3), strict=True))
    CANDLE, LATER = "#7d7b75", "#d2d0c9"
    BAND = "rgba(137, 135, 129, 0.18)"
    TICK = {"1M": "%Y-%m", "1W": "%Y-%m-%d", "1D": "%Y-%m-%d"}

    def stamp(t):
        """A candle's time as text: its date, plus the clock time for an intraday candle."""
        t = pd.Timestamp(t)
        return f"{t:%Y-%m-%d}" if t == t.normalize() else f"{t:%Y-%m-%d %H:%M}"

    def per_session(slope, log):
        """A line's slope in words: its change per trading session."""
        return f"{np.expm1(slope):+.2%} per session" if log else f"{slope:+.3f} per session"

    def level(line, x, log):
        """The price of one line (a table row) at clock positions ``x``."""
        return td.price_at(line.to_frame().T, x, log)[0]

    def candle_trace(bars, name, color):
        """Candles in one color: hollow when the close is above the open, filled when below."""
        return go.Candlestick(
            x=bars["x"], open=bars["open"], high=bars["high"], low=bars["low"],
            close=bars["close"], text=[stamp(t) for t in bars.index], name=name,
            increasing={"line": {"color": color, "width": 1}, "fillcolor": "white"},
            decreasing={"line": {"color": color, "width": 1}, "fillcolor": color})

    def finish(fig, every, tf, log, *, title, height):
        """The shared look, the legend under the plot, dates on the clock axis, and the price
        range of the candles (lines run off it, as on a chart)."""
        base_layout(fig, title=title, height=height)
        plot = max(height - 50 - 150, 80)
        fig.update_layout(margin={"b": 150}, hovermode="closest", legend={
            "orientation": "h", "x": 0.0, "xanchor": "left", "y": -60 / plot, "yanchor": "top",
            "entrywidth": 0.33, "entrywidthmode": "fraction"})
        pick = np.unique(np.linspace(0, len(every) - 1, 9).round().astype(int))
        fmt = TICK.get(tf, "%m-%d %H:%M")
        fig.update_xaxes(tickvals=every["x"].to_numpy()[pick],
                         ticktext=[f"{t:{fmt}}" for t in every.index[pick]],
                         rangeslider_visible=False)
        lo, hi = float(every["low"].min()), float(every["high"].max())
        if log:
            lo, hi = np.log10(lo), np.log10(hi)
        fig.update_yaxes(type="log" if log else "linear",
                         range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        return fig

    def line_traces(fig, side, lines, x0, x1, log):
        """Every line of one side as a ray from point A (or the left edge) to ``x1``: one legend
        entry per chart the lines were drawn on, the pending ray dashed."""
        for (tf, confirmed), group in lines.groupby(["tf", "confirmed"], sort=False):
            xs, ys, tips = [], [], []
            for _, line in group.iterrows():
                left = max(float(line["a_x"]), x0)
                tip = (f"{WORD[side]} line drawn on the {td.TIMEFRAMES[tf]} chart<br>"
                       f"point A: {stamp(line['a_time'])} at {line['a_price']:.2f}<br>"
                       f"stopped by the candle of {stamp(line['stop_time'])}"
                       f"{'' if confirmed else ' (still too new to count)'}<br>")
                if confirmed:
                    tip += (f"point B (latest touch): {stamp(line['b_time'])} at "
                            f"{line['b_price']:.2f}<br>")
                tip += per_session(line["slope"], log)
                xs += [left, x1, None]
                ys += [*level(line, [left, x1], log), None]
                tips += [tip, tip, None]
            name = (f"{WORD[side]} lines · drawn on the {td.TIMEFRAMES[tf]} chart" if confirmed
                    else f"{WORD[side]} pending ray · stopping candle still too new")
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", name=name, hovertext=tips, hoverinfo="text",
                opacity=1.0 if confirmed else 0.7,
                line={"color": COLOR[side], "width": WIDTH[tf] if confirmed else 1.3,
                      "dash": "solid" if confirmed else "dash"}))

    def point_traces(fig, side, lines, shown, x0, x1, log):
        """Points A and B of the confirmed lines, and the candles on the chart that touch them."""
        lines = lines[lines["confirmed"].astype(bool)].reset_index(drop=True)
        if lines.empty:
            return
        n = len(lines)
        what = [f"point A of {WORD[side]} line 1" if i == 0 else
                f"point B of line {i} = point A of line {i + 1}" for i in range(n)]
        pts = pd.DataFrame({"x": [*lines["a_x"], lines["b_x"].iloc[-1]],
                            "price": [*lines["a_price"], lines["b_price"].iloc[-1]],
                            "time": [*lines["a_time"], lines["b_time"].iloc[-1]],
                            "what": [*what, f"point B of line {n} (its latest touch)"]})
        pts = pts[pts["x"].between(x0, x1)]
        fig.add_trace(go.Scatter(
            x=pts["x"], y=pts["price"], mode="markers",
            name=f"{WORD[side]} points A and B (B = latest touch)",
            marker={"symbol": "diamond", "size": 10, "color": COLOR[side],
                    "line": {"color": "white", "width": 1}},
            hovertext=[f"{w}: {stamp(t)} at {p:.2f}"
                       for w, t, p in zip(pts["what"], pts["time"], pts["price"], strict=True)],
            hoverinfo="text"))
        hits = td.touches(lines, shown, log)
        fig.add_trace(go.Scatter(
            x=hits["x"], y=hits["price"], mode="markers",
            name=f"{WORD[side]} touches (within close enough)",
            marker={"symbol": "circle-open", "size": 8, "color": COLOR[side],
                    "line": {"color": COLOR[side], "width": 1.5}},
            hovertext=[f"touches {WORD[side]} line {i + 1}: {stamp(t)} at {p:.2f}"
                       for i, t, p in zip(hits["line"], hits["time"], hits["price"], strict=True)],
            hoverinfo="text"))

    def chart(tf, shown, state, log, *, title, later=None, height=580):
        """One chart of the walk: its candles, the lines of both sides as they stand after it,
        points A and B, the touches, and (faded) candles that came later."""
        later = shown.iloc[:0] if later is None else later
        every = pd.concat([shown, later])
        x0, x1 = float(every["x"].iloc[0]), float(every["x"].iloc[-1])
        x1 += 0.03 * (x1 - x0)                         # rays run on past the last candle
        fig = go.Figure(candle_trace(shown, f"{td.TIMEFRAMES[tf]} candles", CANDLE))
        if len(later):
            fig.add_trace(candle_trace(later, "later candles (unseen by the lines)", LATER))
        for side, lines in state.items():
            line_traces(fig, side, lines, x0, x1, log)
            point_traces(fig, side, lines, shown, x0, x1, log)
        return finish(fig, every, tf, log, title=title, height=height)

    def step_text(k, tf, state, shown):
        """What the walk did on one chart, in words."""
        view = (f"all {len(shown)} candles (all history)" if k == 0 else
                f"its last {len(shown)} candles, from {stamp(shown.index[0])}")
        parts = []
        for side, lines in state.items():
            conf = lines[lines["confirmed"].astype(bool)]
            here = int((conf["tf"] == tf).sum())
            text = f"**{WORD[side]}**: {here} line(s) drawn or adjusted on this chart"
            if len(conf) > here:
                text += f", {len(conf) - here} kept from the charts above"
            pend = lines[~lines["confirmed"].astype(bool)]
            if len(pend):
                text += (f"; the next ray, from {stamp(pend['a_time'].iloc[0])}, is stopped by "
                         f"the candle of {stamp(pend['stop_time'].iloc[0])}, still too new to "
                         "count (dashed)")
            if k == 0 and len(lines):
                first = "highest high" if side == "down" else "lowest low"
                text += (f". Its first point A is the {first} of all history: "
                         f"{stamp(lines['a_time'].iloc[0])} at {lines['a_price'].iloc[0]:.2f}")
            parts.append(text)
        return f"**The {td.TIMEFRAMES[tf]} chart**, {view}. " + " · ".join(parts) + "."

    def explain(lines, i, view, view_tf, own, log, confirm):
        """How line ``i`` (of one side's confirmed lines) was found, drawn on the candles
        ``view`` of chart ``view_tf``; ``own`` are the candles of the chart it was drawn on."""
        line, side = lines.iloc[i], lines["side"].iloc[0]
        nxt = lines.iloc[i + 1] if i + 1 < len(lines) else None
        ax, ay = float(line["a_x"]), float(line["a_y"])
        span = float(view["x"].iloc[-1]) - ax
        shown = view[view["x"] >= ax - 0.08 * span]
        x1 = float(shown["x"].iloc[-1]) + 0.03 * span
        to_price = np.exp if log else np.asarray
        fig = go.Figure(candle_trace(shown, f"{td.TIMEFRAMES[view_tf]} candles", CANDLE))
        # steeper rays from A: in each sixth of the line's life, the ray to the candle that comes
        # closest to stopping it; the stopping candle goes through every one of them
        right = own[own["x"] > ax]
        y = np.log(right[td.SIDES[side]]) if log else right[td.SIDES[side]]
        slopes = ((y - ay) / (right["x"] - ax)).to_numpy()
        down = side == "down"
        xs, ys = [], []
        for part in np.array_split(np.arange(len(slopes)), 6):
            s = slopes[part]
            s = s[(s < line["slope"]) if down else (s > line["slope"])]
            if s.size:
                xs += [ax, x1, None]
                ys += [*to_price(ay + (s.max() if down else s.min()) * (np.array([ax, x1]) - ax)),
                       None]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", hoverinfo="skip",
            name="steeper rays from A (the stopping candle cuts each)",
            line={"color": MUTED, "width": 1, "dash": "dot"}))
        edge = ay + line["slope"] * (np.array([ax, x1]) - ax)
        inner = edge - line["tol"] if side == "down" else edge + line["tol"]
        fig.add_trace(go.Scatter(
            x=[ax, x1, x1, ax, ax], y=to_price([edge[0], edge[1], inner[1], inner[0], edge[0]]),
            mode="lines", fill="toself", fillcolor=BAND, line={"width": 0}, hoverinfo="skip",
            name="close enough (a candle reaching in touches)"))
        fig.add_trace(go.Scatter(
            x=[ax, x1], y=to_price(edge), mode="lines", hoverinfo="skip",
            name=f"the line ({per_session(line['slope'], log)})",
            line={"color": COLOR[side], "width": 3}))
        if nxt is not None:
            fig.add_trace(go.Scatter(
                x=[nxt["a_x"], x1], y=level(nxt, [nxt["a_x"], x1], log),
                mode="lines", hoverinfo="skip",
                name="the next line (from B, steeper)",
                line={"color": COLOR[side], "width": 1.5, "dash": "dash"}))
        hits = td.touches(lines.iloc[[i]], own, log)       # on the line's own chart
        hits = hits[hits["x"] >= float(shown["x"].iloc[0])]
        fig.add_trace(go.Scatter(
            x=hits["x"], y=hits["price"], mode="markers", name="touches",
            marker={"symbol": "circle-open", "size": 9, "color": COLOR[side],
                    "line": {"color": COLOR[side], "width": 1.5}},
            hovertext=[f"touch: {stamp(t)} at {p:.2f}"
                       for t, p in zip(hits["time"], hits["price"], strict=True)],
            hoverinfo="text"))
        for key, name, symbol in (("a", "point A", "diamond"),
                                  ("stop", "the candle that stops the ray", "star"),
                                  ("b", "point B (latest touch, next line's A)", "diamond-open")):
            fig.add_trace(go.Scatter(
                x=[line[f"{key}_x"]], y=[line[f"{key}_price"]], mode="markers", name=name,
                marker={"symbol": symbol, "size": 14, "color": COLOR[side],
                        "line": {"color": COLOR[side] if "open" in symbol else "white",
                                 "width": 2 if "open" in symbol else 1}},
                hovertext=[f"{name}: {stamp(line[f'{key}_time'])} at {line[f'{key}_price']:.2f}"],
                hoverinfo="text"))
        tol = f"{np.expm1(line['tol']):.2%}" if log else f"{line['tol']:.2f}"
        turn = "down" if side == "down" else "up"
        text = (
            f"From **point A** ({stamp(line['a_time'])}, {line['a_price']:.2f}) the ray turns "
            f"{turn} as far as it can, until the candle of **{stamp(line['stop_time'])}** "
            f"(star) stops it: every steeper ray (dotted) runs through that candle. The line "
            f"changes {per_session(line['slope'], log)}. A candle whose "
            f"{td.SIDES[side]} comes within {tol} of it (the grey band, close enough on the "
            f"{td.TIMEFRAMES[line['tf']]} chart) touches it: {len(hits)} "
            f"{td.TIMEFRAMES[line['tf']]} candles (open circles). The latest "
            f"touch followed by at least {confirm} candles is **point B** "
            f"({stamp(line['b_time'])}, {line['b_price']:.2f})"
            + (": the next line (dashed) starts there." if nxt is not None
               else ". No later candle can stop a ray from B yet, so the chain ends here."))
        if view_tf != line["tf"]:
            text += (f" The line was drawn on the {td.TIMEFRAMES[line['tf']]} chart; its life is "
                     f"too long to show in those candles, so they are the "
                     f"{td.TIMEFRAMES[view_tf]} ones.")
        title = f"How {WORD[side]} line {i + 1} was found"
        return finish(fig, shown, view_tf, log, title=title, height=560), text

    return WORD, chart, explain, per_session, stamp, step_text


@app.cell(hide_code=True)
def _(charts, daily, day, frames, minutes, mo, pd, screen, stamp, symbol, td):
    _rows = [{"chart": td.TIMEFRAMES[_tf], "candles": len(frames[_tf]),
              "first": stamp(frames[_tf].index[0]), "last": stamp(frames[_tf].index[-1]),
              "on screen": len(frames[_tf]) if _k == 0 else min(len(frames[_tf]), screen.value)}
             for _k, _tf in enumerate(charts) if len(frames[_tf])]
    _minutes = ("none are stored" if minutes is None else
                f"{minutes.index[0]:%Y-%m-%d} to {minutes.index[-1]:%Y-%m-%d}")
    mo.vstack([mo.md(f"""
    ## 1 · The candles

    **{symbol}**: daily candles from {daily.index[0]:%Y-%m-%d} to {daily.index[-1]:%Y-%m-%d}
    ({len(daily)} sessions); 1-minute candles of the regular session: {_minutes}. The walk sees
    everything up to the close of **{day:%Y-%m-%d}**. Monthly and weekly candles are built from the
    daily ones; the 4-hour, 1-hour, 15- and 5-minute candles from the 1-minute ones, regular session
    only, each counted from the 09:30 open (a 4-hour chart has a 09:30 and a 13:30 candle, as on
    TradingView). Before the first 1-minute session, the intraday charts lean on the daily candles.

    Tori starts from the very first candle a stock ever traded. Here, all history means the
    database's daily history: if {symbol} traded before {daily.index[0]:%Y-%m-%d}, those highs and
    lows are missing, and a side's first line can start elsewhere than on her chart (ServiceNow,
    for one, listed in June 2012).
    """), mo.ui.table(pd.DataFrame(_rows), selection=None, show_download=False)])
    return


@app.cell(hide_code=True)
def _(chart, charts, frames, log, mo, screen, step_text, symbol, td, walk):
    _tabs = {}
    for _k, _tf in enumerate(charts):
        _shown = frames[_tf] if _k == 0 else frames[_tf].iloc[-screen.value:]
        _state = {_side: walk[_side][_tf] for _side in td.SIDES}
        _title = f"{symbol} · {td.TIMEFRAMES[_tf]} chart · lines after step {_k + 1}"
        _tabs[f"{_k + 1} · {td.TIMEFRAMES[_tf]}"] = mo.vstack([
            mo.md(step_text(_k, _tf, _state, _shown)),
            chart(_tf, _shown, _state, log, title=_title)])
    mo.vstack([mo.md("""
    ## 2 · The top-down walk, chart by chart

    Each tab is one chart of the walk, with the lines as they stand **after** it. **Red** lines run
    along the highs (downward lines), **blue** lines under the lows (upward lines). A line is the
    thicker, the higher the chart it was drawn on. Diamonds are points A and B (B is a line's latest
    touch and the next line's A); open circles are candles that touch a line. A **dashed** ray is
    pending: the candle that stops it is among the last *confirm* candles, so it is no touch yet.
    Hover over a line for its points and slope.
    """), mo.ui.tabs(_tabs)])
    return


@app.cell(hide_code=True)
def _(chart, charts, daily, day, frames, log, minutes, mo, screen, sessions, symbol, td,
      trading, walk):
    _tf = charts[-1]
    _shown = frames[_tf].iloc[-screen.value:]
    _all = td.candles(daily, minutes, _tf, sessions)
    _later = _all[_all["x"] > _shown["x"].iloc[-1]].iloc[:max(screen.value // 3, 1)]
    _missing = ("" if _tf == trading.value else f"\n\n*No 1-minute candles up to "
                f"{day:%Y-%m-%d}: the walk stops at the daily chart.*")
    mo.vstack([mo.md(f"""
    ## 3 · The result on the {td.TIMEFRAMES[_tf]} chart

    Every line of the walk on the trading chart, as of the close of **{day:%Y-%m-%d}**. Light
    candles came later; the lines never saw them. Move the as-of date back to replay a stretch,
    as Tori does with TradingView's replay.{_missing}
    """), chart(_tf, _shown, {_side: walk[_side][_tf] for _side in td.SIDES}, log,
                title=f"{symbol} · {td.TIMEFRAMES[_tf]} chart · as of {day:%Y-%m-%d}",
                later=_later, height=660)])
    return


@app.cell(hide_code=True)
def _(WORD, charts, mo, stamp, td, walk):
    _options = {}
    for _side in td.SIDES:
        _lines = walk[_side][charts[-1]]
        _lines = _lines[_lines["confirmed"].astype(bool)]
        for _i, (_, _line) in enumerate(_lines.iterrows()):
            _options[f"{WORD[_side]} line {_i + 1} · drawn on the "
                     f"{td.TIMEFRAMES[_line['tf']]} chart · A {stamp(_line['a_time'])} → "
                     f"B {stamp(_line['b_time'])}"] = (_side, _i)
    pick_line = mo.ui.dropdown(options=_options, value=next(iter(_options), None),
                               label="Line to explain")
    mo.vstack([mo.md("""
    ## 4 · How one line is found

    Pick a line of the result. The figure shows the line's chart from point A on: the rays from A
    that are steeper than the line (each goes through the candle that stopped it), the band within
    which a candle touches, the touches, and point B, where the next line starts. The three
    sliders are the same as at the top.
    """), pick_line])
    return (pick_line,)


@app.cell(hide_code=True)
def _(charts, confirm, explain, frames, log, mo, pick_line, pierce, touch, walk):
    mo.stop(pick_line.value is None, mo.md("*No confirmed line to explain.*"))
    _side, _i = pick_line.value
    _lines = walk[_side][charts[-1]]
    _lines = _lines[_lines["confirmed"].astype(bool)].reset_index(drop=True)
    _line = _lines.iloc[_i]
    # the view: the lowest chart, at or above the line's own, that shows the line's whole life in
    # at most 800 candles (a 15-minute line from 2022 would be 20,000 candles)
    _above = charts[:charts.index(_line["tf"]) + 1]
    _view = next((_tf for _tf in reversed(_above)
                  if (frames[_tf]["x"] >= _line["a_x"]).sum() <= 800), charts[0])
    _fig, _text = explain(_lines, _i, frames[_view], _view, frames[_line["tf"]], log,
                          confirm.value)
    mo.vstack([mo.hstack([touch, pierce, confirm], justify="start", gap=2), mo.md(_text), _fig])
    return


@app.cell(hide_code=True)
def _(WORD, charts, frames, log, mo, np, pd, per_session, screen, stamp, td, walk):
    _final = charts[-1]
    _last = frames[_final].iloc[-1]
    _rows = []
    for _side in td.SIDES:
        _lines = walk[_side][_final]
        _lines = _lines[_lines["confirmed"].astype(bool)]
        _now = td.price_at(_lines, [_last["x"]], log)[:, 0] if len(_lines) else []
        for _i, (_, _line) in enumerate(_lines.iterrows()):
            _rows.append({
                "line": f"{WORD[_side]} {_i + 1}", "drawn on": td.TIMEFRAMES[_line["tf"]],
                "point A": stamp(_line["a_time"]), "A price": round(_line["a_price"], 2),
                "stopped by": stamp(_line["stop_time"]),
                "point B (latest touch)": stamp(_line["b_time"]),
                "B price": round(_line["b_price"], 2),
                "slope": per_session(_line["slope"], log),
                "line at the last candle": round(float(_now[_i]), 2),
                "last close vs line": f"{_last['close'] / _now[_i] - 1:+.1%}"})
    _checks = []
    for _k, _tf in enumerate(charts):
        _shown = frames[_tf] if _k == 0 else frames[_tf].iloc[-screen.value:]
        for _side in td.SIDES:
            _all = walk[_side][_tf]
            _lines = _all[_all["confirmed"].astype(bool)]
            _slope = np.diff(_lines["slope"].to_numpy(float))
            _checks.append({
                "chart": td.TIMEFRAMES[_tf], "side": WORD[_side], "lines": len(_lines),
                "pending ray": "yes" if len(_all) > len(_lines) else "no",
                "candles on screen through a line": td.crossings(_lines, _shown, log),
                "each line steeper than the one before": bool(
                    (_slope < 0).all() if _side == "down" else (_slope > 0).all()),
                "each line starts at the previous point B": bool(np.allclose(
                    _lines["b_x"].to_numpy(float)[:-1], _lines["a_x"].to_numpy(float)[1:]))})
    mo.vstack([
        mo.md(f"""
    ## 5 · The lines, and the rule checks

    Every confirmed line on the {td.TIMEFRAMES[_final]} chart: its points, the candle that stopped
    the ray, its slope, where it stands at the last candle and how far the last close
    ({_last['close']:.2f}) is from it.
    """),
        mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=20, show_download=False),
        mo.md("""
    Her rules, checked on every chart of the walk: no candle on a chart's screen goes through a
    line (beyond *too small to see*), every line is steeper than the one before, and every line
    starts at the previous line's point B. The first column should read 0 and the other two True.
    """),
        mo.ui.table(pd.DataFrame(_checks), selection=None, page_size=20, show_download=False)])
    return


if __name__ == "__main__":
    app.run()
