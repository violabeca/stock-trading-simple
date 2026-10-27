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
    # Tori Trades · the bounce, trade by trade: its best and worst trades, and when it pays

    This notebook replays one fixed line setting of her trendline bounce over a stock's whole
    1-minute history, long and short, starting with NOW. It logs every trade with its trendlines.
    Each side is replayed alone: long only, then short only. In one shared wallet, a short
    holding the money would hide a long trade, and the other way round.

    **How the lines are drawn** (2026-10-27): by default your way, one line per chart and side
    (daily, 4-hour, 1-hour, 15-minute, each starting at the point B of the one above, each at
    least its chart's minimum long; `topdown.cascade`), or her full walk. Three rules sit on
    top of every preset, all on by default:
    - **Strict anchors.** No candle may cross a line, and point B is the angle candle, so both
      anchors are candle extremes on the line and every line starts exactly at the previous
      line's point B.
    - **15-minute lines only.** A trade always bounces off the line closest to price, the last
      line of the top-down chain. That line is a 15-minute line whenever the walk drew one on
      the 15-minute chart. When it drew none on a side, that side does not trade.
    - **The anchor gap.** A 15-minute line counts only when its points A and B are at least
      16 bars of 15 minutes apart (4 hours). A shorter one is not a line yet, and the chain
      ends before it.

    **Part 1 · the trades**
    - **§1 The trade log.** Every long and every short trade, ranked by its return after costs
      and split into fifths. Each row has the line the trade bounced off and how many upward
      and downward lines stood when it was set up. The best five of each side are picked; click
      rows to pick others.
    - **§2 The picked trades, each in its own chart.** The 15-minute candles from a few
      sessions before the entry to a few after the exit. Every line that stood at the setup is
      drawn, upward and downward, with its anchors. The chart also shows the entry gap, the
      entry, the stop at every close and the exit, and the trade is told in words.

    **Part 2 · when does it pay?**
    - **§3–§5 Profitable time ranges.** A zigzag cuts each side's record into rising and falling
      legs. A rising leg of a few trades is a profitable range. The ranges are labelled in
      hindsight: they are what a regime metric would have to predict.
    - **§6 Luck.** Do the wins cluster more than random deals of the same trades would?
    """)
    return


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(
        label="reload regimes.py, bounce.py and topdown.py (after they changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from trading_simple import strategies
    from trading_simple.plots import _LONG as LONG
    from trading_simple.plots import _LOSS as LOSS
    from trading_simple.plots import _MUTED as MUTED
    from trading_simple.plots import _SHORT as SHORT
    from trading_simple.plots import _WIN as WIN
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as base_layout

    _ = reload_helpers.value     # a click reruns this cell: the helper files are read again
    _path = strategies.STRATEGIES_DIR / "tori_trades_trendline" / "regimes.py"
    _spec = importlib.util.spec_from_file_location("tori_trades_regimes", _path)
    assert _spec is not None and _spec.loader is not None, _path
    rg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rg)
    td = rg.td                        # topdown.py, as regimes.py loaded it
    return (LONG, LOSS, MUTED, SERIES, SHORT, WIN, base_layout, go, make_subplots, np, pd, rg,
            td)


@app.cell
def _():
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    settings = load_settings()
    engine = make_engine(settings)
    costs = settings.costs
    return costs, engine


@app.cell
def _(mo, rg):
    stock = mo.ui.text(value="NOW", label="Stock")
    preset = mo.ui.dropdown(options=list(rg.PRESETS), value="#417 without its trend filter",
                            label="Line setting")
    method = mo.ui.radio(options={"one line per chart (your method)": "cascade",
                                  "her full walk (many lines per chart)": "walk"},
                         value="one line per chart (your method)", label="Lines", inline=True)
    shortest = mo.ui.dictionary({
        _tf: mo.ui.slider(0, _top, step=1, value=_value, show_value=True,
                          label=f"Shortest {_name} line: candles from A to B")
        for _tf, (_top, _value, _name) in {"1D": (30, 5, "daily"), "4H": (20, 4, "4-hour"),
                                           "1H": (32, 4, "1-hour")}.items()})
    strict = mo.ui.checkbox(value=True,
                            label="Strict anchors: no candle crosses a line, point B on the line "
                                  "(the one-line method always is)")
    own_only = mo.ui.checkbox(value=True,
                              label="Trade only 15-minute lines (none on a side: no trade)")
    anchor_gap = mo.ui.slider(0, 52, step=1, value=16, show_value=True,
                              label="Anchor gap: 15-minute bars between a 15-minute line's "
                                    "points A and B, at least (16 = 4 hours)")
    run = mo.ui.run_button(label="run the replay (long, then short)")
    mo.vstack([mo.hstack([stock, preset, run], justify="start", gap=2),
               mo.hstack([method], justify="start"),
               mo.hstack([shortest["1D"], shortest["4H"], shortest["1H"]], justify="start",
                         gap=2),
               mo.hstack([strict, own_only, anchor_gap], justify="start", gap=2)])
    return anchor_gap, method, own_only, preset, run, shortest, stock, strict


@app.cell(hide_code=True)
def _(anchor_gap, method, mo, own_only, pd, preset, rg, shortest, strict):
    knobs = {**rg.PRESETS[preset.value]["knobs"], "method": method.value,
             "minimum": dict(shortest.value), "strict": strict.value,
             "own_lines": own_only.value, "anchor_gap": anchor_gap.value}
    _every = {**knobs, **rg.FIXED}
    _rows = pd.DataFrame({
        "knob": list(_every), "value": [f"{_v:g}" if isinstance(_v, float) else str(_v)
                                        for _v in _every.values()],
        "": ["fixed in every preset" if _k in rg.FIXED else
             "set by the controls above"
             if _k in ("method", "minimum", "strict", "own_lines", "anchor_gap") else ""
             for _k in _every],
        "what it means": [rg.MEANING[_k] for _k in _every]})
    setting_line = (f"{preset.value}: " + " · ".join(
        f"{_k} {_v:g}" if isinstance(_v, (int, float)) and not isinstance(_v, bool)
        else f"{_k} {_v}" for _k, _v in knobs.items()) + " · 15-minute chart")
    mo.vstack([mo.md(f"**The line setting: {preset.value}.** {rg.PRESETS[preset.value]['why']}"
                     " Distances are in typical candles (the stock's median 15-minute range at "
                     "the time), so a value means the same on a calm and on a jumpy stock."),
               mo.ui.table(_rows, selection=None, show_download=False, page_size=20)])
    return knobs, setting_line


@app.cell
def _(engine, mo, rg, stock, td):
    symbol = stock.value.strip().upper()
    try:
        daily, minutes = td.load(engine, symbol)
    except (ValueError, KeyError) as _error:
        mo.stop(True, mo.callout(mo.md(f"No daily candles of **{symbol}** in the database "
                                       f"({_error})."), kind="danger"))
    mo.stop(minutes is None, mo.callout(mo.md(
        f"**{symbol}** has no 1-minute candles: the trading chart is built from them."),
        kind="danger"))
    _charts = list(td.TIMEFRAMES)[:list(td.TIMEFRAMES).index("15m") + 1]
    book = rg.bounce.Chartbook(daily, minutes, _charts, td.clock(daily))   # every chart, once
    line_cache = {}        # the lines drawn at each candle, shared by both sides and presets
    return book, daily, line_cache, minutes, symbol


@app.cell
def _(book, costs, daily, knobs, line_cache, minutes, mo, rg, run, symbol):
    mo.stop(not run.value, mo.md(
        f"Press **run the replay**: {symbol}'s 1-minute history runs from "
        f"{minutes.index[0]:%Y-%m-%d} to {minutes.index[-1]:%Y-%m-%d}. The first replay of a "
        "setting takes three to five minutes (the long side draws the lines, the short side "
        "reuses them); a result computed before for the same stock, setting and candles is read "
        "from disk at once (`var/tori_trades_trendline/regimes/`)."))
    results = {}
    with mo.status.spinner(title="Replaying ...") as _spinner:
        for _side in ("long", "short"):
            _spinner.update(f"Replaying the {_side} side of {symbol} ...")
            results[_side] = rg.replay_side(daily, minutes, symbol, _side, knobs, costs,
                                            book=book, cache=line_cache)
    return (results,)


@app.cell
def _(LONG, LOSS, MUTED, SERIES, SHORT, WIN, base_layout, go, make_subplots, np, pd, rg,
      td):
    COLOR = {"long": LONG, "short": SHORT}
    BAND = {"long": "rgba(42, 120, 214, 0.16)", "short": "rgba(235, 104, 52, 0.16)"}
    KEY = {"long": "rgba(42, 120, 214, 0.45)", "short": "rgba(235, 104, 52, 0.45)"}
    LINES = {"long": "upward", "short": "downward"}
    TAG = {"long": "L", "short": "S"}
    STOP, CANDLE = SERIES[2], "#7d7b75"
    VIEW = "rgba(125, 123, 117, 0.14)"          # the detailed chart's window, on the daily one
    DEEP = {"long": "#1b4f8f", "short": "#a8461d"}  # monthly and weekly lines: a deeper shade
    #: One style for every figure, the same as the top-down notebook's: the color tells the
    #: side (blue upward = long, orange downward = short), the dash and width the chart a line
    #: was first found on.
    DASH = {"1M": ("solid", 4.4), "1W": ("12px,4px,3px,4px", 3.4), "1D": ("solid", 3.0),
            "4H": ("14px,7px", 2.6), "1H": ("6px,5px", 2.2), "15m": ("dot", 3.0),
            "5m": ("dot", 2.0)}
    EXITS = {"stop hit": "the stop was hit", "target hit": "the profit target was hit",
             "closed below the line": "a candle closed below the line, out at the next open",
             "closed above the line": "a candle closed above the line, out at the next open"}

    def stamp(t):
        """A time as text: its date, plus the clock time for an intraday one."""
        t = pd.Timestamp(t)
        return f"{t:%Y-%m-%d}" if t == t.normalize() else f"{t:%Y-%m-%d %H:%M}"

    def line_at(a_x, a_y, slope, x):
        """A line's price at clock positions ``x`` (plain price scale, as every preset)."""
        return a_y + slope * (np.asarray(x, dtype=float) - a_x)

    def candle_trace(bars, name):
        """Candles in one color: hollow when the close is above the open, filled when below."""
        return go.Candlestick(
            x=bars["x"], open=bars["open"], high=bars["high"], low=bars["low"],
            close=bars["close"], text=[stamp(t) for t in bars.index], name=name,
            increasing={"line": {"color": CANDLE, "width": 1}, "fillcolor": "white"},
            decreasing={"line": {"color": CANDLE, "width": 1}, "fillcolor": CANDLE})

    def finish(fig, bars, fmt, *, title, height):
        """The shared look, dates on the session-clock axis and the legend under the plot in two
        columns. ``height`` is the plot's own; the figure grows by the legend's rows, so the
        legend never runs into the axis labels or off the figure."""
        rows = -(-sum(1 for tr in fig.data if tr.showlegend is not False and tr.name) // 2)
        bottom = 55 + 24 * rows
        base_layout(fig, title=title, height=50 + height + bottom)
        fig.update_layout(margin={"b": bottom}, hovermode="closest", legend={
            "orientation": "h", "x": 0.0, "xanchor": "left", "y": -45 / height,
            "yanchor": "top", "entrywidth": 0.5, "entrywidthmode": "fraction",
            "traceorder": "normal"})         # legend groups would each start a new row
        pick = np.unique(np.linspace(0, len(bars) - 1, 9).round().astype(int))
        fig.update_xaxes(tickvals=bars["x"].to_numpy()[pick],
                         ticktext=[f"{t:{fmt}}" for t in bars.index[pick]],
                         rangeslider_visible=False)
        return fig

    def key(fig, color, name):
        """A legend entry for a shaded span (a shape has none of its own)."""
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker={"symbol": "square", "size": 12, "color": color}))

    def window(bars, t, before, after, start=None):
        """The candles of ``bars`` from ``before`` sessions before trade ``t``'s entry, or from
        the session holding clock position ``start`` when that is earlier (at most from the
        first candle), to ``after`` sessions after its exit."""
        day = bars.index.normalize()
        sessions = day.unique()
        i0, i1 = (int(sessions.searchsorted(t[f"{k} time"].normalize()))
                  for k in ("entry", "exit"))
        first = sessions[max(i0 - before, 0)]
        if start is not None:
            first = min(first, day[max(int(bars["x"].searchsorted(start, side="right")) - 1, 0)])
        return bars[(day >= first) & (day <= sessions[min(i1 + after, len(sessions) - 1)])]

    def reach(result, trade, first):
        """Where the detailed chart must start to show the anchors of the line a trade bounced
        off, and that line. ``first`` is the clock position of the first 15-minute candle: an
        anchor before it cannot be shown, so the chart then starts at the line's first anchor
        after it (the angle candle or point B)."""
        lines = rg.lines_at(result, trade)
        line = lines[lines["traded"]]
        anchors = line[["a_x", "stop_x", "b_x"]].to_numpy(float)
        anchors = anchors[anchors >= first]
        return (float(anchors.min()) if anchors.size else None), line

    def line_style(owner, tf):
        """The look of one side's lines (``owner`` long or short) first found on chart
        ``tf``."""
        dash, width = DASH[tf]
        return {"color": DEEP[owner] if tf in ("1M", "1W") else COLOR[owner], "width": width,
                "dash": dash}

    def style_key():
        """The key to the line styles: a sample for every chart and both sides."""
        rows = [tf for tf in DASH if tf != "5m"]
        fig = go.Figure()
        for r, tf in enumerate(rows):
            for c, owner in enumerate(("long", "short")):
                fig.add_trace(go.Scatter(x=[1.5 * c, 1.5 * c + 1.1], y=[-r, -r], mode="lines",
                                         line=line_style(owner, tf), hoverinfo="skip",
                                         showlegend=False))
            fig.add_annotation(x=-0.15, y=-r, text=f"{td.TIMEFRAMES[tf]} line", xanchor="right",
                               showarrow=False)
        for c, head in enumerate(("upward lines · long", "downward lines · short")):
            fig.add_annotation(x=1.5 * c + 0.55, y=0.9, text=f"<b>{head}</b>", showarrow=False)
        base_layout(fig, title="The line styles, the same in every chart", height=300)
        fig.update_layout(width=760, margin={"l": 20, "r": 20, "t": 45, "b": 10})
        fig.update_xaxes(visible=False, range=[-1.9, 2.8])
        fig.update_yaxes(visible=False, range=[-len(rows) + 0.4, 1.4])
        return fig

    def line_traces(fig, lines, x0, x1):
        """Every line in ``lines`` as a ray from point A (or the left edge) to ``x1``, in the
        style of its side and of the chart it was first found on: one legend entry per side
        and chart, top-down. Returns the count of lines per chart and kind."""
        counts = {}
        groups = lines.groupby(["kind", "origin"], sort=False)
        for (kind, origin), part in sorted(
                groups, key=lambda g: (g[0][0] != "upward", list(DASH).index(g[0][1]))):
            counts[(origin, kind)] = len(part)
            owner = "long" if kind == "upward" else "short"
            xs, ys, tips = [], [], []
            for line in part.itertuples():
                left = max(line.a_x, x0)
                found = f"first found on the {td.TIMEFRAMES[origin]} chart"
                if line.tf != origin:
                    found += f", adjusted on the {td.TIMEFRAMES[line.tf]} chart"
                tip = (f"{kind} line, {found}" + (", the one traded" if line.traded else "")
                       + f"<br>point A {stamp(line.a_time)} at {line.a_price:.2f}, point B "
                       f"{stamp(line.b_time)} at {line.b_price:.2f}")
                xs += [left, x1, None]
                ys += [*line_at(line.a_x, line.a_y, line.slope, [left, x1]), None]
                tips += [tip, tip, None]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines", hovertext=tips, hoverinfo="text",
                name=f"{kind} · {td.TIMEFRAMES[origin]} lines ({len(part)})",
                line=line_style(owner, origin)))
        return counts

    def traded_tint(fig, t, x0, out):
        """A broad tint under the line the trade bounced off, to its exit: the line keeps its
        own style on top of it."""
        left = max(float(t["a_x"]), x0)
        fig.add_trace(go.Scatter(
            x=[left, out], y=line_at(t["a_x"], t["a_y"], t["slope"], [left, out]),
            mode="lines", name="the line the trade bounced off (tinted)", hoverinfo="skip",
            opacity=0.22, line={"color": COLOR[t["side"]], "width": 14}))

    def anchors(fig, lines, x0, x1, size, name):
        """The anchors of ``lines`` in view, colored by the line's side: ◆ points A and B
        (each B is the next line's A) and ★ the angle candle where it is not point B (B the
        latest touch, or a line a lower chart took over)."""
        diamonds, stars = [], []
        for line in lines.itertuples():
            head = f"{line.kind} line, first found on the {td.TIMEFRAMES[line.origin]} chart"
            color = COLOR["long" if line.kind == "upward" else "short"]
            diamonds += [(line.a_x, line.a_price, color,
                          f"{head}<br>point A: {stamp(line.a_time)} at {line.a_price:.2f}"),
                         (line.b_x, line.b_price, color,
                          f"{head}<br>point B: {stamp(line.b_time)} at {line.b_price:.2f}")]
            if abs(line.stop_x - line.b_x) > 1e-9:
                stars.append((line.stop_x, line.stop_price, color,
                              f"{head}<br>angle candle: {stamp(line.stop_time)} at "
                              f"{line.stop_price:.2f}"))
        for points, symbol, label in ((diamonds, "diamond", f"{name}: ◆ points A and B"),
                                      (stars, "star", f"{name}: ★ angle candle (not B)")):
            points = [p for p in points if x0 <= p[0] <= x1]
            if points:
                px, py, colors, notes = zip(*points, strict=True)
                fig.add_trace(go.Scatter(
                    x=px, y=py, mode="markers", hovertext=notes, hoverinfo="text", name=label,
                    marker={"symbol": symbol, "size": size, "color": list(colors),
                            "line": {"color": "white", "width": 1}}))

    def topdown_chart(result, days, bars, trade, horizon, before, after, start, *, title):
        """The trade's setup in the big picture, on daily candles from ``horizon`` sessions
        before its entry: every line that stood at the setup, in the style of the chart of the
        top-down walk it was drawn on (the higher the chart, the heavier the line), the
        anchors, the line traded, the entry, the exit, and the window of the detailed chart
        (shaded; it reaches back to ``start``). Returns the figure and the walk in words."""
        t = result["trades"].iloc[trade]
        side = t["side"]
        lines = rg.lines_at(result, trade)
        setup = result["setups"].iloc[int(t["setup"])]
        detail = window(bars, t, before, after, start)
        k0 = int(days.index.searchsorted(detail.index[0].normalize()))
        k1 = int(days.index.searchsorted(detail.index[-1].normalize(), side="right"))
        shown = days.iloc[max(k0 - horizon, 0):k1]
        x0, x1 = float(shown["x"].iloc[0]), float(shown["x_end"].iloc[-1])
        fig = go.Figure(candle_trace(shown, "daily candles"))
        counts = line_traces(fig, lines, x0, x1)
        walk = [f"{td.TIMEFRAMES[tf]} {counts.get((tf, 'upward'), 0)} up, "
                f"{counts.get((tf, 'downward'), 0)} down"
                for tf in td.TIMEFRAMES if (tf, "upward") in counts or (tf, "downward") in counts]
        anchors(fig, lines, x0, x1, 9, "the lines' anchors")
        out = float(bars.loc[t["exit time"], "x_end"])
        traded_tint(fig, t, x0, out)
        lo, hi = float(shown["low"].min()), float(shown["high"].max())
        if x0 <= float(setup["x"]) <= x1:
            fig.add_trace(go.Scatter(x=[setup["x"]] * 2, y=[lo, hi], mode="lines",
                                     name="the lines were drawn here", hoverinfo="skip",
                                     line={"color": MUTED, "width": 1, "dash": "dot"}))
        fig.add_trace(go.Scatter(
            x=[bars.loc[t["entry time"], "x"]], y=[t["entry"]], mode="markers",
            hoverinfo="skip", name=f"{side} entry",
            marker={"symbol": "triangle-up" if side == "long" else "triangle-down", "size": 13,
                    "color": COLOR[side]}))
        fig.add_trace(go.Scatter(
            x=[bars.loc[t["exit time"], "x"]], y=[t["exit"]], mode="markers",
            name=f"exit: {t['reason']}", hoverinfo="skip",
            marker={"symbol": "x", "size": 12, "color": WIN if t["return"] > 0 else LOSS}))
        key(fig, VIEW, "the window of the detailed chart below (shaded)")
        fig.add_vrect(x0=float(detail["x"].iloc[0]), x1=float(detail["x_end"].iloc[-1]),
                      fillcolor=VIEW, line_width=0, layer="below")
        fig.update_yaxes(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        finish(fig, shown, "%Y-%m-%d", title=title, height=460)
        return fig, ("The top-down walk at the setup, by the chart each line was first found "
                     "on: " + "; ".join(walk) + ".")

    def trade_chart(result, bars, trade, before, after, start, entry, *, title):
        """One trade in context on the 15-minute chart, from ``before`` sessions before its
        entry (or from ``start``, the earliest anchor it must show, when that is earlier) to
        ``after`` sessions after its exit: every line that stood at its setup (upward blue,
        downward orange) with its anchors (◆ A, ★ the candle that stopped the ray, ◇ B), the
        entry gap, the line it bounced off (tinted), the moment the lines were drawn, the entry,
        the stop at every close, the exit and the trade (shaded). Returns the figure and the
        trade in words."""
        t = result["trades"].iloc[trade]
        side, d = t["side"], 1.0 if t["side"] == "long" else -1.0
        setup = result["setups"].iloc[int(t["setup"])]
        lines = rg.lines_at(result, trade)
        shown = window(bars, t, before, after, start)
        x0, x1 = float(shown["x"].iloc[0]), float(shown["x_end"].iloc[-1])
        sx, ex, xx = (float(bars.loc[t[f"{k} time"], "x"]) for k in ("signal", "entry", "exit"))
        out = float(bars.loc[t["exit time"], "x_end"])
        fig = go.Figure(candle_trace(shown, "15-minute candles"))
        line_traces(fig, lines, x0, x1)
        anchors(fig, lines[lines["traded"]], x0, x1, 11, "the traded line")
        span = np.asarray([max(float(setup["x"]), x0), ex])
        edge = line_at(t["a_x"], t["a_y"], t["slope"], span)
        fig.add_trace(go.Scatter(
            x=[*span, *span[::-1], span[0]], y=[*edge, *(edge + d * t["band"])[::-1], edge[0]],
            mode="lines", fill="toself", fillcolor=BAND[side], line={"width": 0},
            hoverinfo="skip", name="entry gap: a low (high) in here is a signal"))
        traded_tint(fig, t, x0, out)
        path = result["stops"]
        path = path[path["trade"] == t["trade"]] if len(path) else path
        if len(path):
            fig.add_trace(go.Scatter(x=path["x"], y=path["stop"], mode="lines",
                                     name="stop at each close", hoverinfo="skip",
                                     line={"color": STOP, "width": 1.5, "shape": "hv"}))
        lo = float(min(shown["low"].min(), t["initial stop"], t["exit"]))
        hi = float(max(shown["high"].max(), t["initial stop"], t["exit"]))
        if x0 <= float(setup["x"]) <= x1:
            fig.add_trace(go.Scatter(x=[setup["x"]] * 2, y=[lo, hi], mode="lines",
                                     name="the lines were drawn here", hoverinfo="skip",
                                     line={"color": MUTED, "width": 1, "dash": "dot"}))
        tested = "low" if d > 0 else "high"
        if entry != "limit":
            fig.add_trace(go.Scatter(
                x=[sx], y=[bars.loc[t["signal time"], tested]], mode="markers",
                name=f"signal: the {tested} came to the line, the close held",
                marker={"symbol": "circle-open", "size": 13, "color": COLOR[side],
                        "line": {"color": COLOR[side], "width": 2}}, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[ex], y=[t["entry"]], mode="markers", hoverinfo="skip",
            name=f"{side} entry ({'limit order filled' if entry == 'limit' else 'next open'})",
            marker={"symbol": "triangle-up" if d > 0 else "triangle-down", "size": 13,
                    "color": COLOR[side]}))
        fig.add_trace(go.Scatter(
            x=[xx], y=[t["exit"]], mode="markers", name=f"exit: {t['reason']}",
            hoverinfo="skip", marker={"symbol": "x", "size": 12,
                                      "color": WIN if t["return"] > 0 else LOSS}))
        key(fig, KEY[side], "the trade, entry to exit (shaded)")
        fig.add_vrect(x0=ex, x1=out, fillcolor=BAND[side], line_width=0, layer="below")
        finish(fig, shown, "%m-%d", title=title, height=460)
        starts = shown.groupby(shown.index.normalize())["x"].first()
        fmt = "%a %m-%d" if len(starts) <= 8 else "%Y-%m-%d"
        starts = starts.iloc[::max(1, -(-len(starts) // 8))]   # at most 8 labels, level
        fig.update_xaxes(tickvals=starts.to_numpy(), tickangle=0,
                         ticktext=[f"{s:{fmt}}" for s in starts.index])
        fig.update_yaxes(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        traded = lines[lines["traded"]]
        b = (f", point B {stamp(traded['b_time'].iloc[0])} at "
             f"{traded['b_price'].iloc[0]:.2f}" if len(traded) else "")
        verb = "bought" if d > 0 else "sold short"
        if entry == "limit":
            how_in = (f"At {stamp(t['entry time'])} price came into the entry gap, "
                      f"{t['band']:.2f} wide {'above' if d > 0 else 'below'} the line, where "
                      f"the resting order sat: {verb} at {t['entry']:.2f}")
        else:
            how_in = (f"At {stamp(t['signal time'])} the candle's {tested} came within "
                      f"{t['band']:.2f} of the line and the candle closed "
                      f"{'above' if d > 0 else 'below'} it: {verb} at the next open, "
                      f"{stamp(t['entry time'])} at {t['entry']:.2f}")
        story = (
            f"The lines were drawn at {stamp(setup['time'])} ({setup['reason']}): "
            f"{(lines['kind'] == 'upward').sum()} upward and "
            f"{(lines['kind'] == 'downward').sum()} downward line(s), all on the chart. The "
            f"trade's line was drawn on the {td.TIMEFRAMES[t['line drawn on']]} chart (point A "
            f"{stamp(t['line A'])} at {t['line A price']:.2f}{b}). {how_in}, first stop "
            f"{t['initial stop']:.2f} (1 R = {abs(t['entry'] - t['initial stop']):.2f}). Out: "
            f"{EXITS.get(t['reason'], 'still open at the last candle, valued at its close')}, "
            f"{stamp(t['exit time'])} at {t['exit']:.2f}: **{t['R']:+.2f} R, "
            f"{t['return']:+.2%} after costs**, {t['candles']} candles held.")
        return fig, story

    def span(chart, leg):
        """Where a leg starts and ends on the session clock: its first entry candle to the end
        of its last exit candle."""
        return float(chart.loc[leg["from"], "x"]), float(chart.loc[leg["to"], "x_end"])

    def overview_chart(results, legs, days, *, title):
        """The stock's daily candles over the replay, each side's record after every trade, and
        the profitable ranges shaded: long ones blue, short ones orange."""
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(candle_trace(days, "daily candles"), row=1, col=1)
        for row, side in ((2, "long"), (3, "short")):
            r, t = results[side], results[side]["trades"]
            if t.empty:
                continue
            x = np.r_[r["chart"]["x"].iloc[0], r["chart"]["x"].reindex(t["exit time"])]
            record = 100 * (np.r_[1.0, np.cumprod(1 + t["return"].to_numpy())] - 1)
            fig.add_trace(go.Scatter(
                x=x, y=record, mode="lines", hoverinfo="skip",
                name=f"{side} record: compounded return after costs (%)",
                line={"color": COLOR[side], "width": 1.6, "shape": "hv"}), row=row, col=1)
            won = t["return"].to_numpy() > 0
            for flag, color, label in ((True, WIN, "a trade with a profit"),
                                       (False, LOSS, "a trade with a loss")):
                pick = np.flatnonzero(won == flag)
                fig.add_trace(go.Scatter(
                    x=x[pick + 1], y=record[pick + 1], mode="markers", name=label,
                    legendgroup=label, showlegend=side == "long",
                    marker={"color": color, "size": 6}, hoverinfo="text",
                    hovertext=[f"{side} trade {i + 1}: {stamp(t['entry time'].iloc[i])} to "
                               f"{stamp(t['exit time'].iloc[i])}, "
                               f"{t['return'].iloc[i]:+.2%}, {t['reason'].iloc[i]}"
                               for i in pick]), row=row, col=1)
            fig.update_yaxes(title_text=f"{side} (%)", row=row, col=1)
        lo, hi = float(days["low"].min()), float(days["high"].max())
        fig.update_yaxes(title_text="price", range=[lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)],
                         row=1, col=1)
        for side, edge in (("long", "top"), ("short", "bottom")):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", marker={"symbol": "square", "size": 12,
                                                            "color": KEY[side]},
                name=f"profitable range of the {side}s (price chart: {edge} strip)"),
                row=1, col=1)
        # the shading last: a shape aimed at a subplot without a trace yet is dropped silently
        for row, side in ((2, "long"), (3, "short")):
            good = legs[side][legs[side]["kind"] == "profitable"]
            strip = (0.95, 1.0) if side == "long" else (0.0, 0.05)
            for _, leg in good.iterrows():
                x0, x1 = span(results[side]["chart"], leg)
                fig.add_vrect(x0=x0, x1=x1, y0=strip[0], y1=strip[1], fillcolor=KEY[side],
                              line_width=0, row=1, col=1)
                fig.add_vrect(x0=x0, x1=x1, fillcolor=BAND[side], line_width=0, layer="below",
                              row=row, col=1, annotation_text=f"{TAG[side]}{leg['leg']}",
                              annotation_position="top left",
                              annotation_font={"size": 10, "color": COLOR[side]})
        return finish(fig, days, "%Y-%m-%d", title=title, height=700)

    def range_chart(result, bars, leg, side, before, *, title):
        """One leg up close on the 15-minute chart: ``before`` candles of history, every trade
        of the leg with the line it traded (from its point A, ◆ A and ◇ B marked), the entry,
        the stop at every close and the exit."""
        t = result["trades"].iloc[leg["first"] - 1:leg["last"]]
        k0, k1 = (int(bars.index.searchsorted(leg[k])) for k in ("from", "to"))
        shown = bars.iloc[max(k0 - before, 0):min(k1 + 27, len(bars) - 1) + 1]
        x0, x1 = float(shown["x"].iloc[0]), float(shown["x"].iloc[-1])
        fig = go.Figure(candle_trace(shown, "15-minute candles"))
        xs, ys, tips, points = [], [], [], []
        for _, tr in t.iterrows():
            drawn = result["lines"][tr["setup"]][side]
            same = drawn[(drawn["a_time"] == tr["line A"]) & (drawn["b_time"] == tr["line B"])
                         & (drawn["tf"] == tr["line drawn on"])]
            out = float(bars.loc[tr["exit time"], "x_end"])
            left = max(float(tr["a_x"]), x0)
            head = (f"trade {tr['trade'] + 1}: the {LINES[side]} line drawn on the "
                    f"{td.TIMEFRAMES[tr['line drawn on']]} chart")
            tip = f"{head}<br>point A {stamp(tr['line A'])} at {tr['line A price']:.2f}"
            if len(same):                        # its anchors, from the lines of its setup
                line = same.iloc[0]
                tip += f", point B {stamp(line['b_time'])} at {line['b_price']:.2f}"
                points += [(line["a_x"], line["a_price"], "diamond", f"{head}<br>point A"),
                           (line["b_x"], line["b_price"], "diamond-open",
                            f"{head}<br>point B")]
            xs += [left, out, None]
            ys += [*line_at(tr["a_x"], tr["a_y"], tr["slope"], [left, out]), None]
            tips += [tip, tip, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hovertext=tips, hoverinfo="text",
                                 name="the line each trade bounced off (to its exit)",
                                 line={"color": COLOR[side], "width": 2}))
        points = [p for p in points if x0 <= p[0] <= x1]
        if points:
            px, py, symbols, notes = zip(*points, strict=True)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers", hovertext=notes, hoverinfo="text",
                name="their anchors in view: ◆ point A · ◇ point B",
                marker={"symbol": list(symbols), "size": 10, "color": COLOR[side],
                        "line": {"color": COLOR[side], "width": 1.5}}))
        stops = result["stops"]
        stops = stops[stops["trade"].isin(t["trade"])] if len(stops) else stops
        if len(stops):
            sx, sy = [], []
            for _, path in stops.groupby("trade"):
                sx += [*path["x"], None]
                sy += [*path["stop"], None]
            fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="stop at each close",
                                     hoverinfo="skip",
                                     line={"color": STOP, "width": 1.5, "shape": "hv"}))
        tx = {k: bars["x"].reindex(t[f"{k} time"]).to_numpy() for k in ("entry", "exit")}
        fig.add_trace(go.Scatter(
            x=tx["entry"], y=t["entry"], mode="markers", name=f"{side} entry",
            marker={"symbol": "triangle-up" if side == "long" else "triangle-down", "size": 11,
                    "color": COLOR[side]}, hoverinfo="text",
            hovertext=[f"trade {n + 1}: in {stamp(e)} at {p:.2f}"
                       for n, e, p in zip(t["trade"], t["entry time"], t["entry"],
                                          strict=True)]))
        for flag, color, label in ((True, WIN, "exit with a profit"),
                                   (False, LOSS, "exit with a loss")):
            pick = (t["return"] > 0).to_numpy() == flag
            fig.add_trace(go.Scatter(
                x=tx["exit"][pick], y=t["exit"][pick], mode="markers", name=label,
                marker={"symbol": "x", "size": 10, "color": color}, hoverinfo="text",
                hovertext=[f"trade {n + 1}: {why}, {stamp(e)} at {p:.2f}, {r:+.2f} R, "
                           f"{ret:+.2%}" for n, why, e, p, r, ret in zip(
                               t["trade"][pick], t["reason"][pick], t["exit time"][pick],
                               t["exit"][pick], t["R"][pick], t["return"][pick],
                               strict=True)]))
        key(fig, KEY[side], "the leg (shaded)")
        a, b = span(result["chart"], leg)
        fig.add_vrect(x0=a, x1=b, fillcolor=BAND[side], line_width=0, layer="below")
        lo, hi = float(shown["low"].min()), float(shown["high"].max())
        fig.update_yaxes(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        fmt = "%Y-%m-%d" if leg["sessions"] > 3 else "%m-%d %H:%M"
        return finish(fig, shown, fmt, title=title, height=460)

    return (TAG, overview_chart, range_chart, reach, stamp, style_key, topdown_chart,
            trade_chart)


@app.cell(hide_code=True)
def _(mo):
    order = mo.ui.radio(options=["most profitable first", "biggest losses first"],
                        value="most profitable first", label="Order", inline=True)
    top_n = mo.ui.slider(0, 20, step=1, value=5, show_value=True,
                         label="Pick the first N trades of each side")
    mo.vstack([mo.md("""
    ## 1 · The trade log

    Every trade of each side, ranked by its return after costs. **profit fifth** splits a
    side's trades into five equal groups, 1 the most profitable fifth and 5 the least. Each row
    has the line the trade bounced off: the chart it was drawn on and its points A and B. It
    also has the setup: when the lines were drawn, why, and how many upward and downward lines
    stood then (§2 draws them all). The first *N* rows of each table are picked; tick or untick
    rows to pick others. The tables can be sorted, searched and downloaded.
    """), mo.hstack([order, top_n], justify="start", gap=2)])
    return order, top_n


@app.cell(hide_code=True)
def _(mo, order, pd, results, rg, stamp, td, top_n):
    logs = {_s: rg.trade_log(results[_s]) for _s in ("long", "short")}

    def _view(log):
        """The trade log as shown: times as text, returns in %, in the chosen order."""
        if log.empty:
            return pd.DataFrame()
        view = pd.DataFrame({
            "rank": log["rank"], "profit fifth": log["profit fifth"],
            "trade": log["trade"] + 1, "return %": (100 * log["return"]).round(2),
            "R": log["R"].round(2), "entry time": log["entry time"].map(stamp),
            "exit time": log["exit time"].map(stamp), "why it ended": log["reason"],
            "candles held": log["candles"], "entry": log["entry"].round(2),
            "first stop": log["initial stop"].round(2), "exit": log["exit"].round(2),
            "line drawn on": log["line drawn on"].map(td.TIMEFRAMES),
            "line A": log["line A"].map(stamp), "line A price": log["line A price"].round(2),
            "line B": log["line B"].map(stamp), "lines drawn at": log["lines drawn at"].map(stamp),
            "drawn because": log["drawn because"], "upward lines": log["upward lines"],
            "downward lines": log["downward lines"]})
        return (view if order.value == "most profitable first"
                else view.iloc[::-1].reset_index(drop=True))

    def _table(side):
        """One side's trades as a table, the first ``top_n`` rows picked."""
        view = _view(logs[side])
        return mo.ui.table(view, selection="multi", page_size=10, label=f"{side} trades",
                           initial_selection=list(range(min(top_n.value, len(view)))))

    pick_long, pick_short = _table("long"), _table("short")

    def _summary(side):
        _r, _t = results[side], results[side]["trades"]
        return {"side": side, "trades": len(_t),
                "with a profit": f"{(_t['return'] > 0).mean():.0%}" if len(_t) else "",
                "net per trade": f"{1e4 * _t['return'].mean():+.1f} bp" if len(_t) else "",
                "best": f"{_t['return'].max():+.2%}" if len(_t) else "",
                "worst": f"{_t['return'].min():+.2%}" if len(_t) else "",
                "compounded": f"{_r['summary']['strategy return']:+.1%}",
                "buy and hold": f"{_r['summary']['buy and hold return']:+.1%}"}

    _fifths = pd.concat([
        _l.groupby("profit fifth").agg(
            trades=("return", "size"), highest=("return", "max"), lowest=("return", "min"),
            average=("return", "mean"), average_R=("R", "mean"),
            candles_held=("candles", "mean"),
            usual_exit=("reason", lambda _x: _x.value_counts().index[0]))
        .assign(side=_s).reset_index() for _s, _l in logs.items() if len(_l)],
        ignore_index=True) if any(len(_l) for _l in logs.values()) else pd.DataFrame()
    if len(_fifths):
        _fifths = _fifths.assign(**{_c: (100 * _fifths[_c]).round(2)
                                    for _c in ("highest", "lowest", "average")})
        _fifths = _fifths[["side", "profit fifth", "trades", "highest", "lowest", "average",
                           "average_R", "candles_held", "usual_exit"]].round(2).rename(
            columns={"highest": "best %", "lowest": "worst %", "average": "average %",
                     "average_R": "average R", "candles_held": "candles held",
                     "usual_exit": "most common exit"})
    mo.vstack([
        mo.ui.table(pd.DataFrame([_summary(_s) for _s in ("long", "short")]), selection=None,
                    show_download=False),
        mo.md("**By profit fifth** (returns after costs, in %):"),
        mo.ui.table(_fifths, selection=None, show_download=False, page_size=10),
        mo.md("**Long trades**"), pick_long, mo.md("**Short trades**"), pick_short])
    return logs, pick_long, pick_short


@app.cell(hide_code=True)
def _(mo, pd, results, rg, stamp, td):
    _every = []
    for _side in ("long", "short"):
        _l = rg.lines_log(results[_side])
        if len(_l):
            _every.append(pd.DataFrame({
                "side": _side, "trade": _l["trade"] + 1, "line": _l["kind"],
                "traded": _l["traded"], "drawn on": _l["tf"].map(td.TIMEFRAMES),
                "point A": _l["a_time"].map(stamp), "A price": _l["a_price"].round(2),
                "point B": _l["b_time"].map(stamp), "B price": _l["b_price"].round(2),
                "stopping candle": _l["stop_time"].map(stamp),
                "stopping price": _l["stop_price"].round(2),
                "slope per session": _l["slope"].round(4)}))
    mo.accordion({"Every line that stood at every trade's setup (to download)": mo.ui.table(
        pd.concat(_every, ignore_index=True) if _every else pd.DataFrame(), selection=None,
        page_size=15)})
    return


@app.cell(hide_code=True)
def _(mo, setting_line, style_key):
    days_before = mo.ui.slider(0, 10, step=1, value=3, show_value=True,
                               label="Sessions shown before the entry, at least")
    days_after = mo.ui.slider(0, 10, step=1, value=2, show_value=True,
                              label="Sessions shown after the exit")
    horizon = mo.ui.slider(20, 750, step=10, value=120, show_value=True,
                           label="Big picture: daily candles shown before the detail")
    mo.vstack([mo.md(f"""
    ## 2 · The picked trades, two charts each

    Every trade picked in §1, long trades first, in the tables' order, with two charts.

    - **The big picture (top).** Daily candles from some sessions before the entry (the
      slider), with every line that stood when the trade was set up, each in the style of the
      chart of the top-down walk it was first found on (the key below; the same style in every
      chart of both notebooks). The line the trade bounced off is tinted, and the window of the
      detailed chart is shaded.
    - **The detail (below).** The 15-minute candles from the point A of the line the trade
      bounced off, or from a few sessions before the entry if that is earlier, to a few
      sessions after the exit. It shows the same lines, and the candles that fixed the traded
      line: ◆ points A and B, and ★ the angle candle where it is not B. When the traded line starts
      before the stock's 15-minute history, the note under the trade says so; its point A is
      then on the big picture only. Also: the entry gap, the moment the lines were drawn
      (dotted), the entry, the stop at every close, the exit and the trade (shaded). Zoom in
      with the mouse; double-click to zoom back out.

    Upward lines are blue, downward lines orange. Hover over a line or an anchor for its
    details; the table under each trade lists its lines. **Setting** {setting_line}.
    """), style_key(), mo.hstack([days_before, days_after, horizon], justify="start", gap=2)])
    return days_after, days_before, horizon


@app.cell(hide_code=True)
def _(book, days_after, days_before, horizon, knobs, logs, mo, pd, pick_long, pick_short,
      reach, results, rg, stamp, symbol, td, topdown_chart, trade_chart):
    _MAX = 20                                    # charts per side, at most
    _out = []
    for _side, _pick in (("long", pick_long), ("short", pick_short)):
        _rows = _pick.value if _pick.value is not None else pd.DataFrame()
        _n = len(_rows)
        _out.append(mo.md(f"### {_side.capitalize()} trades: {_n} picked"
                          + (f", the first {_MAX} drawn" if _n > _MAX else "")))
        for _number in (_rows["trade"].head(_MAX).tolist() if _n else []):
            _i = int(_number) - 1
            _log = logs[_side].set_index("trade").loc[_i]
            _start, _line = reach(results[_side], _i, float(book.full["15m"]["x"].iloc[0]))
            _note = (" The traded line starts before the 15-minute history "
                     f"({book.full['15m'].index[0]:%Y-%m-%d}): its point A is on the big "
                     "picture only."
                     if (_line["a_x"] < book.full["15m"]["x"].iloc[0]).any() else "")
            _top, _walk = topdown_chart(
                results[_side], book.full["1D"], book.full["15m"], _i, horizon.value,
                days_before.value, days_after.value, _start,
                title=f"{symbol} · {_side} trade {_i + 1} · the big picture: its lines on daily "
                      "candles")
            _fig, _story = trade_chart(
                results[_side], book.full["15m"], _i, days_before.value, days_after.value,
                _start, knobs["entry"],
                title=(f"{symbol} · {_side} trade {_i + 1} · rank {_log['rank']} of "
                       f"{len(logs[_side])} · {_log['return']:+.2%} after costs · "
                       f"{_log['R']:+.2f} R"))
            _lines = rg.lines_at(results[_side], _i)
            _table = pd.DataFrame({
                "line": _lines["kind"], "traded": _lines["traded"],
                "drawn on": _lines["tf"].map(td.TIMEFRAMES),
                "point A": _lines["a_time"].map(stamp), "A price": _lines["a_price"].round(2),
                "point B": _lines["b_time"].map(stamp), "B price": _lines["b_price"].round(2),
                "stopping candle": _lines["stop_time"].map(stamp),
                "slope per session": _lines["slope"].round(4)})
            _out.append(mo.vstack([
                mo.md(f"**{_side.capitalize()} trade {_i + 1}** (rank {_log['rank']} of "
                      f"{len(logs[_side])}, profit fifth {_log['profit fifth']}). {_story} "
                      f"{_walk}{_note}"),
                _top, _fig,
                mo.accordion({f"the {len(_lines)} lines at this trade's setup": mo.ui.table(
                    _table, selection=None, show_download=False, page_size=20)})]))
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    reversal = mo.ui.slider(0.5, 10.0, step=0.5, value=2.0, show_value=True,
                            label="Reversal (%): a leg ends when the record turns this much")
    min_trades = mo.ui.slider(1, 10, step=1, value=3, show_value=True,
                              label="Min trades: a rising leg needs this many to count")
    mo.vstack([mo.md("""
    # Part 2 · When does it pay? Profitable time ranges

    The question behind this part: can something we can measure tell, better than luck, that a
    stock is moving into a stretch where the bounce pays? A side's record is its return after
    costs, compounded trade by trade. A zigzag cuts the record into legs: a rising leg ends once
    the record falls by the *reversal* from its high, a falling leg once it rises by the
    *reversal* from its low. A rising leg of at least *min trades* trades is a **profitable
    range**, from its first trade's entry to its last trade's exit. A range is known only after
    it ends: the ranges are the target of the question, and a metric meant to predict them must
    be read before a range starts. Both sliders apply at once, without replaying.
    """), mo.hstack([reversal, min_trades], justify="start", gap=2)])
    return min_trades, reversal


@app.cell
def _(min_trades, results, reversal, rg):
    legs = {_side: rg.ranges(_r["trades"], reversal.value / 100, min_trades.value, _r["chart"])
            for _side, _r in results.items()}
    return (legs,)


@app.cell(hide_code=True)
def _(book, legs, mo, overview_chart, pd, results, setting_line, symbol):
    def _summary(side):
        _r, _t, _g = results[side], results[side]["trades"], legs[side]
        _good = _g[_g["kind"] == "profitable"]
        _days = _r["chart"].index.normalize().unique()
        _inside = sum(((_days >= f.normalize()) & (_days <= t)).sum()
                      for f, t in zip(_good["from"], _good["to"], strict=True))
        return {
            "side": side, "trades": len(_t),
            "profitable ranges": len(_good),
            "trades inside them": f"{_good['trades'].sum() / max(len(_t), 1):.0%}",
            "sessions inside them": f"{_inside / len(_days):.0%}"}

    _first = min(_r["chart"].index[0] for _r in results.values()).normalize()
    _days = book.full["1D"][book.full["1D"].index >= _first]
    mo.vstack([
        mo.md(f"""
    ## 3 · Where the setting made money

    {symbol}, every session from {_first:%Y-%m-%d} to {_days.index[-1]:%Y-%m-%d}. **Setting**
    {setting_line}. Top: the stock's daily candles. Below: each side's record after every
    trade (a dot per trade: green with a profit, red with a loss). The shaded spans on each
    record are its profitable ranges, numbered L1, L2 ... for the longs and S1, S2 ... for the
    shorts, as in the tables below; on the price chart the same ranges are strips, the longs'
    along its top edge and the shorts' along its bottom edge. Zoom with the mouse; hover over a
    dot for its trade.
    """),
        mo.ui.table(pd.DataFrame([_summary(_s) for _s in ("long", "short")]),
                    selection=None, show_download=False),
        overview_chart(results, legs, _days, title=f"{symbol} · profitable ranges of each side")])
    return


@app.cell(hide_code=True)
def _(TAG, legs, mo, pd, stamp):
    _rows = pd.concat([_g.assign(side=_s) for _s, _g in legs.items()], ignore_index=True)
    _table = pd.DataFrame({
        "range": [f"{TAG[_s]}{_n}" for _s, _n in zip(_rows["side"], _rows["leg"], strict=True)],
        "side": _rows["side"], "kind": _rows["kind"],
        "from": _rows["from"].map(stamp), "to": _rows["to"].map(stamp),
        "sessions": _rows["sessions"], "trades": _rows["trades"],
        "wins": _rows["wins"], "return": (100 * _rows["return"]).round(2),
        "average R": _rows["average R"].round(2),
        "best trade's share": _rows["best share"].round(2),
        "stock move": (100 * _rows["stock move"]).round(2)}) if len(_rows) else _rows
    mo.vstack([mo.md("""
    ## 4 · Every leg of each record

    Every leg the zigzag found, profitable or not: the profitable ranges are the target, the
    losing legs the contrast the next step needs. **return** is the leg's own return after costs
    (%), **stock move** the stock's from the leg's first entry to its last exit (%), **best
    trade's share** the part of a rising leg's gain that came from its best trade (near 1: one
    trade made the range). The table can be sorted, filtered and downloaded.
    """), mo.ui.table(_table, selection=None, page_size=20)])
    return


@app.cell(hide_code=True)
def _(TAG, legs, mo):
    _options = {}
    for _side, _g in legs.items():
        for _, _leg in _g.iterrows():
            _options[(f"{TAG[_side]}{_leg['leg']} · {_side} · {_leg['kind']} · "
                      f"{_leg['from']:%Y-%m-%d} to {_leg['to']:%Y-%m-%d} · "
                      f"{_leg['trades']} trades · {_leg['return']:+.1%}")] = (_side, _leg["leg"])
    mo.stop(not _options, mo.md("*No trade, so no leg to look at.*"))
    _first = next((_k for _k in _options if " profitable " in _k), next(iter(_options)))
    pick_leg = mo.ui.dropdown(options=_options, value=_first, label="Leg to look at")
    history = mo.ui.slider(0, 1000, step=25, value=100, show_value=True,
                           label="History: 15-minute candles shown before the leg")
    mo.vstack([mo.md("""
    ## 5 · One range up close

    Pick a leg (the first profitable range by default). The chart shows the 15-minute candles
    over the leg (shaded) with some history before it, and for every trade the line it bounced
    off, drawn from its point A (◆) to the trade's exit, point B (◇) where in view, the entry,
    the stop at every close and the exit. Hover over a line for the chart it was drawn on and
    its anchors.
    """), mo.hstack([pick_leg, history], justify="start", gap=2)])
    return history, pick_leg


@app.cell(hide_code=True)
def _(TAG, book, history, legs, mo, pd, pick_leg, range_chart, results, setting_line, stamp,
      symbol):
    _side, _n = pick_leg.value
    _leg = legs[_side].set_index("leg").loc[_n].copy()
    _leg["leg"] = _n
    _t = results[_side]["trades"].iloc[_leg["first"] - 1:_leg["last"]]
    _fig = range_chart(results[_side], book.full["15m"], _leg, _side, history.value,
                       title=f"{symbol} · {TAG[_side]}{_n} ({_side}, {_leg['kind']})")
    _story = (
        f"**{TAG[_side]}{_n}** ({_side}, {_leg['kind']}): {_leg['trades']} trades from "
        f"{stamp(_leg['from'])} to {stamp(_leg['to'])} ({_leg['sessions']} sessions), "
        f"{_leg['wins']} with a profit, **{_leg['return']:+.2%} after costs**; the stock moved "
        f"{_leg['stock move']:+.2%} over the same span. Setting {setting_line}.")
    _rows = pd.DataFrame({
        "trade": _t["trade"] + 1, "entry time": _t["entry time"].map(stamp),
        "entry": _t["entry"].round(2), "first stop": _t["initial stop"].round(2),
        "exit time": _t["exit time"].map(stamp), "exit": _t["exit"].round(2),
        "why": _t["reason"], "R": _t["R"].round(2),
        "return": (100 * _t["return"]).round(2), "line drawn on": _t["line drawn on"],
        "line A": _t["line A"].map(stamp), "line B": _t["line B"].map(stamp)})
    mo.vstack([mo.md(_story), _fig,
               mo.ui.table(_rows, selection=None, page_size=15, show_download=False)])
    return


@app.cell(hide_code=True)
def _(mo, pd, results, rg):
    _tables, _verdicts = [], []
    for _side in ("long", "short"):
        _l = rg.luck(results[_side]["trades"]["return"].to_numpy())
        if _l.empty:
            _verdicts.append(f"**{_side}**: too few trades to read.")
            continue
        _tables.append(_l.assign(side=_side).set_index(["side", "statistic"]).round(3))
        _hits = _l.loc[_l["chance by luck"] < 0.05, "statistic"].tolist()
        _verdicts.append(
            f"**{_side}** ({len(results[_side]['trades'])} trades): "
            + (f"more clustered than luck on {', '.join(_hits)}." if _hits else
               "no sign of clustering: every statistic lies where random deals of the same "
               "trades put it."))
    _verdict = "  \n".join(_verdicts)
    mo.vstack([mo.md(f"""
    ## 6 · Do the wins cluster more than luck would make them?

    A range is no evidence by itself: any record of wins and losses has rising legs. The test
    keeps every trade's size and time and deals the record's wins and losses out again at
    random, 2,000 times (the same number of wins each time). It reads persistence directly: the
    correlation of a trade's return with the next one's, with the average of the five before
    it, and the number of streaks of wins or losses (fewer streaks = more clustered). **chance by
    luck** is the share of random deals at least as clustered as the real record. The zigzag's
    own numbers would make a poor test: ten wins in a row then ten losses spend no more trades
    in profitable ranges than 40% of the random deals do.

    {_verdict}

    With a few dozen trades a side, only strong clustering can show here. If none shows, the
    record gives no hint that stretches exist. A market metric could still sort the trades, so
    the next step tests candidate metrics directly against the legs.
    """), mo.ui.table(pd.concat(_tables).reset_index() if _tables else pd.DataFrame(),
                      selection=None, show_download=False)])
    return


if __name__ == "__main__":
    app.run()
