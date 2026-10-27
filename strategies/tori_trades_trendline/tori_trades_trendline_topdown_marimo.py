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
    # Tori Trades · the top-down lines at one moment, one line per chart

    Pick a stock, a date and a time of day. For each side the notebook draws **one line per
    chart**, top-down, as you described it (2026-10-27): one upward line (for longs) and one
    downward line (for shorts) each on the daily, the 4-hour, the 1-hour and the 15-minute
    chart (`topdown.cascade`). Section 3 then trades these lines over the stock's whole history.

    1. **Daily.** The line starts at the extreme of the daily candles in view: the lowest low
       for the upward line, the highest high for the downward one.
    2. **4-hour, 1-hour, 15-minute.** Each line starts at the point B of the line above it,
       moved onto this chart's candle that holds the same price.
    3. **Every line** is the steepest ray from its point A that no candle of its chart crosses.
       The candle that stops it fixes its angle, and that candle is point B.
    4. **Noise bands** (optional). The stock's noise is the moving average of its candles' true
       range, on each chart and recomputed at every candle. With bands on, a line is a band of
       *line band* × noise: a candle may reach into it without stopping the line, and point B
       is the latest candle inside it. Around the traded 15-minute line, the price's own *price
       band* widens it into a contact zone. A touch reaches into the zone, a break closes beyond
       it, and the stop sits beyond it.
    5. **Not a line yet.** A line whose angle candle is still among the newest *confirm*
       candles is a pending ray, and so is a line shorter than its chart's minimum. The charts
       below a pending ray draw nothing.
    6. **The setup.** A side trades only when its 15-minute line exists. Section 1 names that
       line, or says where the chain stopped.

    Every chart is cut at the chosen moment, exactly as the trade replay (`bounce.py`) cuts it.
    The last candle of a higher chart may still be forming: at 11:00 today's daily candle holds
    only the first hour and a half. Every knob sits in the settings below, one per row.
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
    _path = strategies.STRATEGIES_DIR / "tori_trades_trendline" / "regimes.py"
    _spec = importlib.util.spec_from_file_location("tori_trades_regimes_topdown", _path)
    assert _spec is not None and _spec.loader is not None, _path
    rg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rg)
    bounce, td = rg.bounce, rg.td     # bounce.py and topdown.py, as regimes.py loaded them
    return (LONG, LOSS, MUTED, SERIES, SHORT, WIN, base_layout, bounce, go, np, pd, rg, td)


@app.cell
def _():
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    _settings = load_settings()
    engine = make_engine(_settings)
    costs = _settings.costs
    return costs, engine


@app.cell(hide_code=True)
def _(mo):
    def knob(widget, text):
        """One setting: its control, and what it does beside it."""
        return mo.hstack([widget, mo.md(text)], justify="start", align="center", gap=2,
                         widths=[2, 3])

    stock = mo.ui.text(value="NOW", label="Stock")
    mo.vstack([mo.md("## Settings\n\n### The stock"),
               knob(stock, "Any ticker with daily and 1-minute candles in the database.")])
    return knob, stock


@app.cell
def _(bounce, engine, mo, stock, td):
    symbol = stock.value.strip().upper()
    try:
        daily, minutes = td.load(engine, symbol)
    except (ValueError, KeyError) as _error:
        mo.stop(True, mo.callout(mo.md(f"No daily candles of **{symbol}** in the database "
                                       f"({_error})."), kind="danger"))
    mo.stop(minutes is None, mo.callout(mo.md(
        f"**{symbol}** has no 1-minute candles: the 4-hour to 15-minute charts are built from "
        "them."), kind="danger"))
    CHARTS = ["1D", "4H", "1H", "15m"]
    book = bounce.Chartbook(daily, minutes, list(td.TIMEFRAMES)[:6], td.clock(daily))
    line_cache = {}        # the lines drawn at each candle, shared by both sides' replays
    return CHARTS, book, daily, line_cache, minutes, symbol


@app.cell(hide_code=True)
def _(knob, minutes, mo, td):
    _days = minutes.index.tz_localize(None).normalize()
    day = mo.ui.date(start=_days[0].date(), stop=_days[-1].date(), value=_days[-1].date(),
                     label="Date")
    _closes = [td.OPEN + 15 * _k for _k in range(1, 27)]          # 09:45 ... 16:00
    clock = mo.ui.dropdown(
        options={f"{_m // 60:02d}:{_m % 60:02d}" + (" (the close)" if _m == 960 else ""): _m
                 for _m in _closes},
        value="16:00 (the close)", label="Time")
    later = mo.ui.slider(0, 10, step=1, value=0, show_value=True, label="Later sessions")
    window = mo.ui.slider(20, 750, step=10, value=150, show_value=True,
                          label="Daily candles in view")
    confirm = mo.ui.slider(1, 10, step=1, value=2, show_value=True, label="Confirm")
    min_1d = mo.ui.slider(0, 30, step=1, value=5, show_value=True, label="Shortest daily line")
    min_4h = mo.ui.slider(0, 20, step=1, value=4, show_value=True, label="Shortest 4-hour line")
    min_1h = mo.ui.slider(0, 32, step=1, value=4, show_value=True, label="Shortest 1-hour line")
    min_15m = mo.ui.slider(0, 52, step=1, value=16, show_value=True,
                           label="Shortest 15-minute line")
    draw_first = mo.ui.slider(0, 10, step=1, value=0, show_value=True,
                              label="Leave out: first candles")
    draw_last = mo.ui.slider(0, 10, step=1, value=0, show_value=True,
                             label="Leave out: last candles")
    banded = mo.ui.checkbox(value=False, label="Noise bands")
    noise_window = mo.ui.slider(5, 300, step=5, value=150, show_value=True,
                                label="Noise window")
    line_band = mo.ui.slider(0.0, 1.0, step=0.05, value=0.25, show_value=True,
                             label="Line band")
    price_band = mo.ui.slider(0.0, 1.0, step=0.05, value=0.25, show_value=True,
                              label="Price band")
    entry = mo.ui.radio(options={"next open (her rule)": "next_open",
                                 "limit order at the line": "limit"},
                        value="next open (her rule)", label="Entry", inline=True)
    gap_band = mo.ui.slider(0.0, 3.0, step=0.05, value=0.25, show_value=True, label="Entry gap")
    room = mo.ui.slider(0.0, 3.0, step=0.05, value=0.5, show_value=True, label="Stop room")
    target = mo.ui.slider(0.0, 5.0, step=0.25, value=0.0, show_value=True,
                          label="Profit target (R)")
    hold = mo.ui.slider(0, 5, step=1, value=0, show_value=True, label="Bounce confirmation")
    watch = mo.ui.checkbox(value=True, label="Watch the pending line")
    trade_first = mo.ui.slider(0, 10, step=1, value=0, show_value=True,
                               label="No trades: first candles")
    trade_last = mo.ui.slider(0, 10, step=1, value=0, show_value=True,
                              label="No trades: last candles")
    top_n = mo.ui.slider(1, 10, step=1, value=5, show_value=True, label="Trades drawn per side")
    walk = mo.ui.run_button(label="walk forward: trade the whole history (long, then short)")
    mo.vstack([
        mo.md("### The moment (sections 1 and 2)"),
        knob(day, "The session whose lines you want to see."),
        knob(clock, "New York time: the lines see only the candles before it."),
        knob(later, "Sessions after the moment, drawn faded: the lines never see them."),
        mo.md("### The lines (every section)"),
        knob(window, "The daily line starts at the extreme (lowest low, highest high) of these "
                     "daily candles."),
        knob(confirm, "Candles that must follow a line's angle candle before the line counts."),
        knob(min_1d, "Daily candles a daily line needs between its points A and B."),
        knob(min_4h, "4-hour candles a 4-hour line needs between its points A and B."),
        knob(min_1h, "1-hour candles a 1-hour line needs between its points A and B (4 = 4 "
                     "hours)."),
        knob(min_15m, "15-minute candles a 15-minute line needs between A and B (16 = 4 hours)."),
        knob(draw_first, "The first 15-minute candles of every session neither stop the "
                         "15-minute line nor become its point B (shown faded)."),
        knob(draw_last, "The same for the last 15-minute candles of every session."),
        mo.md("### Noise bands (every section)"),
        knob(banded, "Draw every line as a band and trade a contact zone, both sized by the "
                     "stock's noise. Off: exact lines."),
        knob(noise_window, "Candles in the moving average of the true range: the noise, on each "
                           "chart, recomputed at every candle."),
        knob(line_band, "Half-width of a line's band, in noise of its chart: a candle may reach "
                        "into it without stopping the line, and point B is the latest candle "
                        "inside it."),
        knob(price_band, "The price's own uncertainty, in 15-minute noise: with the line band "
                         "it makes the contact zone around the traded line."),
        mo.md("### The trade (section 3)"),
        knob(entry, "Next open: a candle comes to the line and closes on its side, in at the "
                    "next open. Limit: an order rests at the line plus the entry gap."),
        knob(gap_band, "How close (in typical candles) a low (high) must come to the line to "
                       "be a signal. With noise bands the contact zone sets it instead."),
        knob(room, "The stop starts this many typical candles beyond the line (beyond the "
                   "contact zone with noise bands) and follows each new swing."),
        knob(target, "Take the profit at this many R (R = the entry's distance to the first "
                     "stop). 0: no target, her rule."),
        knob(hold, "Candles after the bounce that must close on the line's side before the "
                   "entry (next-open entries). A close beyond the line among them: no trade."),
        knob(watch, "After a break, keep the new pending 15-minute line: it trades once its "
                    "angle candle is confirmed, without waiting for the session's close."),
        knob(trade_first, "No position is opened or closed on the first 15-minute candles of a "
                          "session."),
        knob(trade_last, "The same for the last 15-minute candles of a session."),
        knob(top_n, "The most profitable trades of each side drawn in section 3."),
        knob(walk, "Trades the whole 1-minute history; a first run takes a minute or so, the "
                   "same settings are then read from disk.")])
    return (banded, clock, confirm, day, draw_first, draw_last, entry, gap_band, hold, later,
            line_band, min_15m, min_1d, min_1h, min_4h, noise_window, price_band, room, target,
            top_n, trade_first, trade_last, walk, watch, window)


@app.cell
def _(banded, draw_first, draw_last, line_band, min_15m, min_1d, min_1h, min_4h,
      noise_window, price_band):
    minimum = {"1D": min_1d.value, "4H": min_4h.value, "1H": min_1h.value, "15m": min_15m.value}
    draw_trim = (draw_first.value, draw_last.value)
    bands = ((line_band.value, price_band.value) if banded.value else (0.0, 0.0))
    noise_n = noise_window.value
    return bands, draw_trim, minimum, noise_n


@app.cell
def _(CHARTS, bands, book, clock, confirm, day, draw_trim, minimum, minutes, mo, noise_n, pd,
      td, window):
    _date = pd.Timestamp(day.value)
    _on = minutes.index.tz_localize(None).normalize() == _date
    mo.stop(not _on.any(), mo.callout(mo.md(
        f"**{_date:%Y-%m-%d}** is not a session with 1-minute candles (a weekend, a holiday or "
        "outside the stored history): pick another date."), kind="warn"))
    cutoff = (_date + pd.Timedelta(minutes=clock.value)).tz_localize(td.NY)
    cutoff = min(cutoff, minutes.index[_on][-1] + pd.Timedelta(minutes=1))   # half-days: 13:00
    _cut = book.at(cutoff)
    frames = {_tf: _cut[_tf] for _tf in CHARTS}
    chain = {_side: td.cascade(frames, _side, window=window.value, confirm=confirm.value,
                               minimum=minimum, trim=draw_trim, band=bands[0],
                               noise_window=noise_n)
             for _side in td.SIDES}
    return chain, cutoff, frames


@app.cell
def _(LONG, LOSS, MUTED, SERIES, SHORT, WIN, base_layout, go, np, pd, rg, td):
    COLOR = {"up": LONG, "down": SHORT}
    WORD = {"up": "upward", "down": "downward"}
    TRADE = {"up": "long", "down": "short"}
    #: One style for every figure (the regimes notebook's too): the color tells the side (blue
    #: upward = long, orange downward = short), the dash and width the chart the line is on.
    DASH = {"1D": ("solid", 3.0), "4H": ("14px,7px", 2.6), "1H": ("6px,5px", 2.2),
            "15m": ("dot", 3.0)}
    #: Candles of each chart in one session.
    PER_SESSION = {"1D": 1.0, **{_tf: td.SESSION / _m for _tf, _m in td.MINUTES.items()}}
    TICK = {"1D": "%Y-%m-%d"}
    CANDLE, LATER, EDGE = "#7d7b75", "#d2d0c9", "#bfbcb4"
    IGNORED = "#4a3aa7"                # a left-out candle that would have mattered (purple)
    #: A line's band (its own noise band) and the traded line's contact zone, by side.
    FILL = {"up": "rgba(42, 120, 214, 0.13)", "down": "rgba(235, 104, 52, 0.13)"}
    ZONE = {"up": "rgba(42, 120, 214, 0.07)", "down": "rgba(235, 104, 52, 0.07)"}

    def stamp(t):
        """A candle's time as text: its date, plus the clock time for an intraday candle."""
        t = pd.Timestamp(t)
        return f"{t:%Y-%m-%d}" if t == t.normalize() else f"{t:%Y-%m-%d %H:%M}"

    def apart(line):
        """How far apart a line's points A and B are, in candles of its chart, as text."""
        n = (line["b_x"] - line["a_x"]) * PER_SESSION[line["tf"]]
        return f"{n:.0f} {td.TIMEFRAMES[line['tf']]} candles"

    def line_style(side, tf, confirmed=True):
        """The look of one side's line on chart ``tf``, and its opacity: a pending ray is thin,
        dash-dotted and faint, whatever its chart."""
        if not confirmed:
            return {"color": COLOR[side], "width": 1.4, "dash": "dashdot"}, 0.55
        dash, width = DASH[tf]
        return {"color": COLOR[side], "width": width, "dash": dash}, 1.0

    def noises(charts, window):
        """Each chart's noise, candle by candle: ``{chart: (clock positions, noise)}``."""
        return {tf: (c["x"].to_numpy(float), td.noise(c, window)) for tf, c in charts.items()}

    def noise_at(series, xs, lag=0):
        """A chart's noise at clock positions ``xs``: the value of the latest candle starting at
        or before each (``lag`` candles earlier: 1 = as known at that candle's open)."""
        cx, nz = series
        i = np.searchsorted(cx, xs, side="right") - 1 - lag
        return nz[np.clip(i, 0, len(nz) - 1)]

    def band_trace(fig, line, xs, half, fill, name, first, outline=None):
        """A band of half-width ``half`` (one value per x of ``xs``) around ``line``; the legend
        shows it once (``first``)."""
        level = line["a_y"] + line["slope"] * (xs - line["a_x"])
        fig.add_trace(go.Scatter(
            x=[*xs, *xs[::-1]], y=[*(level + half), *(level - half)[::-1]], mode="lines",
            fill="toself", fillcolor=fill, hoverinfo="skip", name=name, legendgroup=name,
            showlegend=first, line={"width": 1 if outline else 0, "color": outline or fill,
                                    "dash": "dot"}))

    def span_xs(shown, left, right):
        """Clock positions to draw a band at: the candles in view from ``left`` to ``right``
        (at most about 800 of them) and both ends."""
        xs = shown["x"].to_numpy(float)
        xs = xs[(xs > left) & (xs < right)]
        xs = xs[::max(1, len(xs) // 800)]
        return np.r_[left, xs, right]

    def style_key(bands):
        """The key to the line styles: a sample for every chart and both sides (and the bands,
        when they are on)."""
        rows = [*((tf, f"{td.TIMEFRAMES[tf]} line") for tf in DASH),
                (None, "pending ray: not a line yet")]
        fig = go.Figure()
        for r, (tf, label) in enumerate(rows):
            for c, side in enumerate(("up", "down")):
                look, opacity = line_style(side, tf or "15m", tf is not None)
                if any(bands) and tf is not None:
                    fig.add_trace(go.Scatter(
                        x=[1.5 * c, 1.5 * c + 1.1, 1.5 * c + 1.1, 1.5 * c],
                        y=[-r + 0.22, -r + 0.22, -r - 0.22, -r - 0.22], fill="toself",
                        fillcolor=FILL[side], line={"width": 0}, mode="lines",
                        hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=[1.5 * c, 1.5 * c + 1.1], y=[-r, -r], mode="lines",
                                         line=look, opacity=opacity, hoverinfo="skip",
                                         showlegend=False))
            fig.add_annotation(x=-0.15, y=-r, text=label, xanchor="right", showarrow=False)
        for c, head in enumerate(("upward line · long", "downward line · short")):
            fig.add_annotation(x=1.5 * c + 0.55, y=0.9, text=f"<b>{head}</b>", showarrow=False)
        title = "The line styles, the same in every chart"
        if any(bands):
            title += " (each line drawn with its noise band)"
        base_layout(fig, title=title, height=270)
        fig.update_layout(width=760, margin={"l": 20, "r": 20, "t": 45, "b": 10})
        fig.update_xaxes(visible=False, range=[-1.9, 2.8])
        fig.update_yaxes(visible=False, range=[-len(rows) + 0.4, 1.4])
        return fig

    def candle_trace(bars, name, color):
        """Candles in one color: hollow when the close is above the open, filled when below."""
        return go.Candlestick(
            x=bars["x"], open=bars["open"], high=bars["high"], low=bars["low"],
            close=bars["close"], text=[stamp(t) for t in bars.index], name=name,
            increasing={"line": {"color": color, "width": 1}, "fillcolor": "white"},
            decreasing={"line": {"color": color, "width": 1}, "fillcolor": color})

    def finish(fig, every, tf, *, title, height):
        """The shared look, dates on the session-clock axis, the price range of the candles
        (lines run off it, as on a chart) and the legend under the plot in two columns; the
        figure grows by the legend's rows, so the legend never covers the axis labels."""
        rows = -(-sum(1 for tr in fig.data if tr.showlegend is not False and tr.name) // 2)
        bottom = 55 + 24 * rows
        base_layout(fig, title=title, height=50 + height + bottom)
        fig.update_layout(margin={"b": bottom}, hovermode="closest", legend={
            "orientation": "h", "x": 0.0, "xanchor": "left", "y": -45 / height,
            "yanchor": "top", "entrywidth": 0.5, "entrywidthmode": "fraction",
            "traceorder": "normal"})
        pick = np.unique(np.linspace(0, len(every) - 1, 8).round().astype(int))
        fmt = TICK.get(tf, "%m-%d %H:%M")
        fig.update_xaxes(tickvals=every["x"].to_numpy()[pick], tickangle=0,
                         ticktext=[f"{t:{fmt}}" for t in every.index[pick]],
                         rangeslider_visible=False)
        lo, hi = float(every["low"].min()), float(every["high"].max())
        fig.update_yaxes(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        return fig

    def faded(fig, shown, tf, ends, name):
        """Candles of ``shown`` (an intraday chart) among the first or last ``ends`` of their
        session, drawn faded under ``name``; returns the other candles."""
        if not any(ends):
            return shown
        edge = td.session_edges(shown.index, tf, *ends)
        if edge.any():
            fig.add_trace(candle_trace(shown[edge], name, EDGE))
        return shown[~edge]

    def ignored(fig, points, symbol, name):
        """Highlight candles that a session-edge rule kept from mattering: an open ``symbol``
        around each point ``(x, price, note)``."""
        if points:
            px, py, notes = zip(*points, strict=True)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers", hovertext=notes, hoverinfo="text", name=name,
                marker={"symbol": symbol, "size": 17, "color": IGNORED,
                        "line": {"color": IGNORED, "width": 2.2}}))

    def would_stop(shown, chain, trim, half):
        """The left-out candles of the 15-minute chart whose low (high) reaches beyond a
        15-minute line's band (``half`` at each candle of ``shown``; 0 without bands) right of
        its point A: without the rule they would have stopped it."""
        edge = td.session_edges(shown.index, "15m", *trim)
        rows, w = shown[edge], half[edge]
        points = []
        for side, lines in chain.items():
            for _, line in lines[lines["tf"] == "15m"].iterrows():
                right = (rows["x"] > line["a_x"] + 1e-9).to_numpy()
                part, wr = rows[right], w[right]
                level = line["a_y"] + line["slope"] * (part["x"] - line["a_x"])
                y = part["low"] if side == "up" else part["high"]
                reach = (y <= level - wr) if side == "up" else (y >= level + wr)
                points += [
                    (row["x"], value, f"{stamp(when)}: left out, but its "
                     f"{'low' if side == 'up' else 'high'} {value:.2f} reaches the {WORD[side]} "
                     f"15-minute line ({lv:.2f}): it would have stopped the line")
                    for (when, row), value, lv in zip(part[reach].iterrows(), y[reach],
                                                     level[reach], strict=True)]
        return points

    def chart(tf, shown, later, chain, cut, traded, *, title, trim=(0, 0), noise=None,
              bands=(0.0, 0.0)):
        """One chart: its candles in view, faded later ones, every side's lines from the daily
        one down to this chart's (each with its noise band when ``bands`` are on), the anchors
        ◆ A and B, the moment chosen and, on the 15-minute chart, the line each side would
        trade (tinted, with its contact zone)."""
        every = pd.concat([shown, later])
        x0, x1 = float(every["x"].iloc[0]), float(every["x_end"].iloc[-1])
        x1 += 0.03 * (x1 - x0)                          # rays run on past the last candle
        fig = go.Figure()
        kept = (faded(fig, shown, tf, trim, f"left out of the 15-minute line (first {trim[0]}, "
                                            f"last {trim[1]} of each session)")
                if tf == "15m" else shown)
        fig.add_trace(candle_trace(kept, f"{td.TIMEFRAMES[tf]} candles", CANDLE))
        if tf == "15m" and any(trim):
            half = (bands[0] * noise_at(noise["15m"], shown["x"].to_numpy(float))
                    if bands[0] else np.zeros(len(shown)))
            ignored(fig, would_stop(shown, chain, trim, half), "square-open",
                    "left out, but it would have stopped a 15-minute line")
        if len(later):
            fig.add_trace(candle_trace(later, "later candles (unseen by the lines)", LATER))
        points, first_band = [], True
        upto = list(DASH)[:list(DASH).index(tf) + 1]
        for side, lines in chain.items():
            for _, line in lines[lines["tf"].isin(upto)].iterrows():
                confirmed = bool(line["confirmed"])
                left = max(float(line["a_x"]), x0)
                if bands[0] and confirmed:
                    xs = span_xs(every, left, x1)
                    band_trace(fig, line, xs, bands[0] * noise_at(noise[line["tf"]], xs),
                               FILL[side], "line bands: line band × each chart's noise",
                               first_band)
                    first_band = False
                look, opacity = line_style(side, line["tf"], confirmed)
                tip = (f"{WORD[side]} {td.TIMEFRAMES[line['tf']]} line"
                       + ("" if confirmed else ": not a line yet") + "<br>"
                       f"point A {stamp(line['a_time'])} at {line['a_price']:.2f}<br>"
                       f"point B {stamp(line['b_time'])} at {line['b_price']:.2f}, "
                       f"{apart(line)} after A")
                fig.add_trace(go.Scatter(
                    x=[left, x1],
                    y=line["a_y"] + line["slope"] * (np.array([left, x1]) - line["a_x"]),
                    mode="lines", line=look, opacity=opacity, hovertext=[tip, tip],
                    hoverinfo="text",
                    name=(f"{WORD[side]} · {td.TIMEFRAMES[line['tf']]} line" if confirmed
                          else f"{WORD[side]} · {td.TIMEFRAMES[line['tf']]} pending ray")))
                if confirmed:
                    points += [(line[f"{k}_x"], line[f"{k}_price"], COLOR[side],
                                f"{WORD[side]} {td.TIMEFRAMES[line['tf']]} line: point {k.upper()}"
                                f", {stamp(line[f'{k}_time'])} at {line[f'{k}_price']:.2f}")
                               for k in ("a", "b")]
        points = [p for p in points if x0 <= p[0] <= x1]
        if points:
            px, py, colors, notes = zip(*points, strict=True)
            fig.add_trace(go.Scatter(
                x=px, y=py, mode="markers", hovertext=notes, hoverinfo="text",
                name="anchors ◆: points A and B (B starts the line below)",
                marker={"symbol": "diamond", "size": 11, "color": list(colors),
                        "line": {"color": "white", "width": 1}}))
        for side, line in traded.items():
            if line is None:
                continue
            left = max(float(line["a_x"]), x0)
            if any(bands):
                xs = span_xs(every, left, x1)
                band_trace(fig, line, xs, sum(bands) * noise_at(noise["15m"], xs, lag=1),
                           ZONE[side], f"contact zone of the {TRADE[side]}: (line band + price "
                           "band) × 15-minute noise", True, outline=COLOR[side])
            fig.add_trace(go.Scatter(        # a broad tint under the line keeps its own style
                x=[left, x1], y=line["a_y"] + line["slope"] * (np.array([left, x1]) - line["a_x"]),
                mode="lines", hoverinfo="skip", opacity=0.22,
                name=f"the line a {TRADE[side]} would trade now (tinted)",
                line={"color": COLOR[side], "width": 14}))
        lo, hi = float(every["low"].min()), float(every["high"].max())
        fig.add_trace(go.Scatter(x=[cut, cut], y=[lo, hi], mode="lines", hoverinfo="skip",
                                 name="the moment chosen",
                                 line={"color": MUTED, "width": 1, "dash": "dot"}))
        return finish(fig, every, tf, title=title, height=520)

    KIND = {"upward": "up", "downward": "down"}
    SIDE = {"long": "up", "short": "down"}
    STOP = SERIES[2]
    BAND = {"up": "rgba(42, 120, 214, 0.16)", "down": "rgba(235, 104, 52, 0.16)"}
    EXITS = {"stop hit": "the stop was hit", "target hit": "the profit target was hit",
             "closed below the line": "a candle closed below the line, out at the next open",
             "closed above the line": "a candle closed above the line, out at the next open"}

    def trade_chart(result, bars, trade, entry, before, after, *, title, quiet=(0, 0), hold=0,
                    noise=None, bands=(0.0, 0.0)):
        """One trade on the 15-minute chart, from the point A of the line it bounced off (or
        ``before`` sessions before its entry, if that is earlier) to ``after`` sessions after
        its exit: the lines of both sides that stood at its setup, one per chart, in the shared
        style (with their noise bands when ``bands`` are on); the traded line tinted, with its
        anchors ◆ A and B and its contact zone; the entry gap, the moment the lines were drawn,
        the entry, the stop at every close, the exit and the trade (shaded). Returns the figure
        and the trade in words."""
        t = result["trades"].iloc[trade]
        key = SIDE[t["side"]]
        d = 1.0 if key == "up" else -1.0
        setup = result["setups"].iloc[int(t["setup"])]
        lines = rg.lines_at(result, trade)
        mine = lines[lines["traded"]]
        day = bars.index.normalize()
        sessions = day.unique()
        i0, i1 = (int(sessions.searchsorted(t[f"{k} time"].normalize()))
                  for k in ("entry", "exit"))
        first = sessions[max(i0 - before, 0)]
        if len(mine) and float(mine["a_x"].iloc[0]) >= float(bars["x"].iloc[0]):
            k = int(bars["x"].searchsorted(float(mine["a_x"].iloc[0]), side="right")) - 1
            first = min(first, day[max(k, 0)])
        shown = bars[(day >= first) & (day <= sessions[min(i1 + after, len(sessions) - 1)])]
        x0, x1 = float(shown["x"].iloc[0]), float(shown["x_end"].iloc[-1])
        ex, xx = (float(bars.loc[t[f"{k} time"], "x"]) for k in ("entry", "exit"))
        out = float(bars.loc[t["exit time"], "x_end"])
        fig = go.Figure()
        kept = faded(fig, shown, "15m", quiet, f"no position opened or closed (first "
                                               f"{quiet[0]}, last {quiet[1]} of each session)")
        fig.add_trace(candle_trace(kept, "15-minute candles", CANDLE))
        groups = sorted(lines.groupby(["kind", "tf"], sort=False),
                        key=lambda g: (g[0][0] != "upward", list(DASH).index(g[0][1])))
        first_band = True
        for (kind, tf), part in groups:
            look, opacity = line_style(KIND[kind], tf)
            xs, ys, tips = [], [], []
            for _, line in part.iterrows():
                left = max(float(line["a_x"]), x0)
                if bands[0]:
                    pts = span_xs(shown, left, x1)
                    band_trace(fig, line, pts, bands[0] * noise_at(noise[tf], pts),
                               FILL[KIND[kind]], "line bands: line band × each chart's noise",
                               first_band)
                    first_band = False
                tip = (f"{kind} {td.TIMEFRAMES[tf]} line"
                       + (", the one traded" if line["traded"] else "")
                       + f"<br>point A {stamp(line['a_time'])} at {line['a_price']:.2f}, "
                       f"point B {stamp(line['b_time'])} at {line['b_price']:.2f}")
                xs += [left, x1, None]
                ys += [*(line["a_y"] + line["slope"] * (np.array([left, x1]) - line["a_x"])),
                       None]
                tips += [tip, tip, None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=look, opacity=opacity,
                                     hovertext=tips, hoverinfo="text",
                                     name=f"{kind} · {td.TIMEFRAMES[tf]} line"))
        if len(mine):
            m = mine.iloc[0]
            pts = [(m[f"{k}_x"], m[f"{k}_price"], f"point {k.upper()}: {stamp(m[f'{k}_time'])} "
                    f"at {m[f'{k}_price']:.2f}") for k in ("a", "b")]
            pts = [p for p in pts if x0 <= p[0] <= x1]
            if pts:
                px, py, notes = zip(*pts, strict=True)
                fig.add_trace(go.Scatter(
                    x=px, y=py, mode="markers", hovertext=notes, hoverinfo="text",
                    name="the traded line's anchors ◆ A and B",
                    marker={"symbol": "diamond", "size": 11, "color": COLOR[key],
                            "line": {"color": "white", "width": 1}}))
        left = max(float(t["a_x"]), x0)
        traded_line = {"a_x": t["a_x"], "a_y": t["a_y"], "slope": t["slope"]}
        if any(bands):
            pts = span_xs(shown, left, out)
            band_trace(fig, traded_line, pts, sum(bands) * noise_at(noise["15m"], pts, lag=1),
                       ZONE[key], "contact zone: (line band + price band) × 15-minute noise",
                       True, outline=COLOR[key])
        span = np.asarray([max(float(setup["x"]), x0), ex])
        edge = t["a_y"] + t["slope"] * (span - t["a_x"])
        if not any(bands):
            fig.add_trace(go.Scatter(
                x=[*span, *span[::-1], span[0]],
                y=[*edge, *(edge + d * t["band"])[::-1], edge[0]], mode="lines", fill="toself",
                fillcolor=BAND[key], line={"width": 0}, hoverinfo="skip",
                name="entry gap: a low (high) in here is a signal"))
        fig.add_trace(go.Scatter(
            x=[left, out], y=t["a_y"] + t["slope"] * (np.array([left, out]) - t["a_x"]),
            mode="lines", hoverinfo="skip", opacity=0.22, line={"color": COLOR[key], "width": 14},
            name="the line the trade bounced off (tinted)"))
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
                x=[float(bars.loc[t["signal time"], "x"])], y=[bars.loc[t["signal time"], tested]],
                mode="markers", name=f"signal: the {tested} came to the line, the close held",
                marker={"symbol": "circle-open", "size": 13, "color": COLOR[key],
                        "line": {"color": COLOR[key], "width": 2}}, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[ex], y=[t["entry"]], mode="markers", hoverinfo="skip",
            name=f"{t['side']} entry ({'limit order filled' if entry == 'limit' else 'next open'})",
            marker={"symbol": "triangle-up" if d > 0 else "triangle-down", "size": 13,
                    "color": COLOR[key]}))
        fig.add_trace(go.Scatter(
            x=[xx], y=[t["exit"]], mode="markers", name=f"exit: {t['reason']}",
            hoverinfo="skip", marker={"symbol": "x", "size": 12,
                                      "color": WIN if t["return"] > 0 else LOSS}))
        if any(quiet):
            q = shown[td.session_edges(shown.index, "15m", *quiet)]
            level = t["a_y"] + t["slope"] * (q["x"] - t["a_x"])
            zone = (sum(bands) * noise_at(noise["15m"], q["x"].to_numpy(float), lag=1)
                    if any(bands) else np.zeros(len(q)))
            ext = q["low"] if d > 0 else q["high"]
            before_in = (q["x"] >= float(setup["x"])) & (q["x"] < ex)
            near = d * (ext - level) <= t["band"]
            held = (d * (q["close"] - level) >= 0) if entry != "limit" else (
                d * (q["open"] - level) > 0)
            pick = q[before_in & near & held]
            ignored(fig, [(row["x"], row[tested], f"{stamp(when)}: no trade then, but its "
                           f"{tested} came to the line: a signal")
                          for when, row in pick.iterrows()],
                    "triangle-up-open" if d > 0 else "triangle-down-open",
                    "no-trade candle: a signal would have opened the trade here")
            stop_at = (path.set_index("time")["stop"].shift(1) if len(path)
                       else pd.Series(dtype=float))
            during = (q["x"] > ex) & (q["x"] <= xx)
            active = stop_at.reindex(q.index).fillna(t["initial stop"])
            hit = q[during & (d * (ext - active) <= 0)]
            ignored(fig, [(row["x"], row[tested], f"{stamp(when)}: no trade then, but its "
                           f"{tested} reached the stop ({active[when]:.2f})")
                          for when, row in hit.iterrows()],
                    "square-open", "no-trade candle: the stop would have closed the trade here")
            beyond = q[during & (d * (q["close"] - level) < -zone)]
            ignored(fig, [(row["x"], row["close"], f"{stamp(when)}: no trade then, but it "
                           f"closed beyond the line ({row['close']:.2f})")
                          for when, row in beyond.iterrows()],
                    "diamond-open",
                    "no-trade candle: a close beyond the line would have ended the trade")
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 name="the trade, entry to exit (shaded)",
                                 marker={"symbol": "square", "size": 12, "color": BAND[key]}))
        fig.add_vrect(x0=ex, x1=out, fillcolor=BAND[key], line_width=0, layer="below")
        finish(fig, shown, "15m", title=title, height=480)
        fig.update_yaxes(range=[lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)])
        verb = "bought" if d > 0 else "sold short"
        reach = (f"came into the contact zone ({t['band']:.2f} either side of the line)"
                 if any(bands) else f"came within {t['band']:.2f} of the line")
        how_in = (f"at {stamp(t['entry time'])} price came into the entry gap "
                  f"({t['band']:.2f} wide), where the resting order sat: {verb} at "
                  f"{t['entry']:.2f}" if entry == "limit" else
                  f"at {stamp(t['signal time'])} the candle's {tested} {reach} and it closed "
                  f"{'above' if d > 0 else 'below'} the line"
                  + (f", and the next {hold} candle(s) did not close beyond it" if hold else "")
                  + f": {verb} at the next open, {stamp(t['entry time'])} at {t['entry']:.2f}")
        story = (
            f"The lines were drawn at {stamp(setup['time'])} ({setup['reason']}). The trade's "
            f"15-minute line runs from point A {stamp(t['line A'])} at {t['line A price']:.2f} "
            f"through point B {stamp(t['line B'])}; {how_in}, first stop "
            f"{t['initial stop']:.2f} (1 R = {abs(t['entry'] - t['initial stop']):.2f}). Out: "
            f"{EXITS.get(t['reason'], 'still open at the last candle, valued at its close')}, "
            f"{stamp(t['exit time'])} at {t['exit']:.2f}: **{t['R']:+.2f} R, "
            f"{t['return']:+.2%} after costs**, {t['candles']} candles held.")
        return fig, story

    return TRADE, WORD, apart, chart, noise_at, noises, stamp, style_key, trade_chart


@app.cell
def _(TRADE, WORD, apart, bands, chain, confirm, cutoff, frames, mo, noise_at, noise_n,
      noises, np, pd, stamp, symbol, td, window):
    _bars = frames["15m"]
    _close = float(_bars["close"].iloc[-1])
    _now = float(_bars["x_end"].iloc[-1])
    _typical = td._typical(_bars.iloc[-150:], False)
    _noise = noise_at(noises({"15m": _bars}, noise_n)["15m"], np.array([_now]))[0] \
        if any(bands) else None
    traded, _text, _rows = {}, [], []
    for _side in ("up", "down"):
        _lines = chain[_side]
        _ok = _lines[_lines["confirmed"].astype(bool)]
        _own = _ok[_ok["tf"] == "15m"]
        _line = _own.iloc[0] if len(_own) else None
        traded[_side] = _line
        _who = f"**{TRADE[_side].capitalize()}** ({WORD[_side]} lines)"
        if _line is None:
            _pend = _lines[~_lines["confirmed"].astype(bool)]
            _extreme = "highest high" if _side == "down" else "lowest low"
            _why = (f"no daily line: the {_extreme} of the {window.value} daily candles in view "
                    f"is one of the newest {confirm.value} (price is at its "
                    f"{window.value}-session {'high' if _side == 'down' else 'low'}), so no "
                    f"{WORD[_side]} line can start yet. Pick an earlier date, or show more daily "
                    "candles"
                    if _lines.empty else
                    f"the chain stops at the {td.TIMEFRAMES[_pend['tf'].iloc[0]]} chart, whose "
                    "line is not a line yet (its angle candle is still too new, or it is "
                    "shorter than that chart's minimum)" if len(_pend) else
                    f"the chain stops at the {td.TIMEFRAMES[_ok['tf'].iloc[-1]]} chart")
            _text.append(f"{_who}: no 15-minute line, no trade: {_why}.")
        else:
            _level = _line["a_y"] + _line["slope"] * (_now - _line["a_x"])
            _away = (_close - _level) if _side == "up" else (_level - _close)
            _text.append(
                f"{_who}: the 15-minute line from point A {stamp(_line['a_time'])} "
                f"({_line['a_price']:.2f}) through point B {stamp(_line['b_time'])} "
                f"({_line['b_price']:.2f}), {apart(_line)} apart. It stands at {_level:.2f} "
                f"now, {_away:+.2f} ({_away / _close:+.2%}, {_away / _typical:+.1f} typical "
                f"candles) {'below' if _side == 'up' else 'above'} the last close "
                f"{_close:.2f}.")
        for _, _l in _lines.iterrows():
            _rows.append({
                "side": WORD[_side], "chart": td.TIMEFRAMES[_l["tf"]],
                "line yet": "yes" if _l["confirmed"] else "no (pending ray)",
                "point A": stamp(_l["a_time"]), "A price": round(_l["a_price"], 2),
                "point B": stamp(_l["b_time"]), "B price": round(_l["b_price"], 2),
                "A to B": apart(_l), "slope per session": round(_l["slope"], 3),
                "band at B": round(_l["tol"], 2)})
    _setup = "  \n".join(_text)
    _bands = (f"Noise bands on: the 15-minute noise is {_noise:.2f} now (the average true range "
              f"of the last {noise_n} candles). Each line's band is ±{bands[0]:g} × its chart's "
              f"noise (±{bands[0] * _noise:.2f} on the 15-minute chart now), and the contact "
              f"zone ±{sum(bands):g} × the 15-minute noise (±{sum(bands) * _noise:.2f})."
              if any(bands) else "Noise bands off: exact lines.")
    mo.vstack([mo.md(f"""
    ## 1 · The setup at {cutoff:%Y-%m-%d %H:%M} New York

    **{symbol}**, last 15-minute close {_close:.2f}; a typical 15-minute candle is
    {_typical:.2f} (the median range of the last 150). {_bands}

    {_setup}

    Each side's lines, top-down (each point B is the next line's point A):
    """), mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=20)])
    return (traded,)


@app.cell(hide_code=True)
def _(CHARTS, bands, book, chain, chart, cutoff, draw_trim, frames, later, mo, noise_n, noises,
      pd, style_key, symbol, td, traded, window):
    _noise = noises(frames, noise_n)
    _out = [mo.md("""
    ## 2 · Chart by chart

    Each chart shows its own line and the lines of the charts above it, with their noise bands
    when those are on. The daily chart shows its candles in view; every lower chart starts a
    little before the point A of its own line, so both anchors are in view. Hover over a line or
    an anchor for its points. The dotted vertical line is the moment chosen.
    """), style_key(bands)]
    for _tf in CHARTS:
        _own = frames[_tf]
        _mine = pd.concat([chain[_s][chain[_s]["tf"] == _tf] for _s in chain])
        if _tf == "1D":
            _shown = _own.iloc[-window.value:]
        elif len(_mine):
            _a = float(_mine["a_x"].min())
            _span = float(_own["x_end"].iloc[-1]) - _a
            _shown = _own[_own["x"] >= _a - 0.08 * _span]
        else:
            _shown = _own.iloc[-150:]
        _after = book.full[_tf][book.full[_tf].index > _own.index[-1]]
        _days = _after.index.normalize().unique()[:later.value]
        _after = _after[_after.index.normalize().isin(_days)]
        _out.append(chart(_tf, _shown, _after, chain, float(_own["x_end"].iloc[-1]),
                          traded if _tf == "15m" else {},
                          title=f"{symbol} · {td.TIMEFRAMES[_tf]} chart · as of "
                                f"{cutoff:%Y-%m-%d %H:%M}",
                          trim=draw_trim, noise=_noise, bands=bands))
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Walk forward: trading these lines

    The replay (`bounce.replay` with `method="cascade"`) walks the stock's whole 1-minute
    history candle by candle, from the session after its first 1-minute one. It draws these
    same lines (one per chart and side, with the settings above) using only what was known at
    each moment. It trades the 15-minute line: a side with none does not trade. Long and short
    are replayed separately, so a short never blocks a long, and the other way round.

    **When the lines are drawn again** (never after every candle):
    - **While waiting:** when a candle breaks the 15-minute line without giving a signal (it
      closes beyond the line, or beyond the contact zone with noise bands), and at every
      session's close.
    - **In a trade:** never. The trade keeps its line until it ends, and the lines are drawn
      again right after.
    - **After a break:** the new 15-minute line is usually still pending, because the candle
      that broke the old one is its angle candle. With *watch* on it is kept, and it trades
      once *confirm* candles have followed its angle candle.

    **The trade:**
    - **In:** a candle comes to the line and closes on its side, in at the next open; or a limit
      order resting at the line plus the entry gap.
    - **Bounce confirmation:** the next *N* candles must not close beyond the line.
    - **Stop:** it starts the stop room beyond the line, and follows each new swing.
    - **Out:** the stop, a close beyond the line (out at the next open), or the profit target.
    - **Session edges:** no position is opened or closed on the left-out candles.
    - **Noise bands:** with them on, "the line" is the contact zone. A touch reaches into it, a
      break and an exit close beyond it, and the stop sits beyond it. A close inside it, beyond
      the line, is noise: neither a signal nor a break.
    - **Money:** the whole $10,000 wallet in every trade, at the repo's costs.

    Press **walk forward** in the settings.
    """)
    return


@app.cell
def _(bands, book, confirm, costs, daily, draw_trim, entry, gap_band, hold, line_cache,
      minimum, minutes, mo, noise_n, rg, room, symbol, target, trade_first, trade_last, walk,
      watch, window):
    mo.stop(not walk.value, mo.md("*Press **walk forward** in the settings to trade the whole "
                                  "history.*"))
    walk_knobs = {"tol": 0.0, "confirm": confirm.value, "band": gap_band.value,
                  "room": room.value, "target": target.value, "reward": 0.0,
                  "skip_open": 0, "trend": 0, "min_span": 0.0, "entry": entry.value,
                  "method": "cascade", "minimum": minimum, "strict": True,
                  "own_lines": True, "anchor_gap": minimum["15m"], "screen": window.value,
                  "draw_trim": draw_trim, "trade_trim": (trade_first.value, trade_last.value),
                  "watch": watch.value, "hold": hold.value, "line_band": bands[0],
                  "price_band": bands[1], "noise_window": noise_n}
    walked = {}
    with mo.status.spinner(title="Walking forward ...") as _spinner:
        for _side in ("long", "short"):
            _spinner.update(f"Trading the {_side} side of {symbol} ...")
            walked[_side] = rg.replay_side(daily, minutes, symbol, _side, walk_knobs, costs,
                                           book=book, cache=line_cache)
    return walk_knobs, walked


@app.cell(hide_code=True)
def _(mo, pd, rg, stamp, walked):
    logs = {_s: rg.trade_log(walked[_s]) for _s in ("long", "short")}

    def _summary(side):
        _r, _t = walked[side], walked[side]["trades"]
        return {"side": side, "trades": len(_t),
                "with a profit": f"{(_t['return'] > 0).mean():.0%}" if len(_t) else "",
                "net per trade": f"{1e4 * _t['return'].mean():+.1f} bp" if len(_t) else "",
                "best": f"{_t['return'].max():+.2%}" if len(_t) else "",
                "worst": f"{_t['return'].min():+.2%}" if len(_t) else "",
                "compounded": f"{_r['summary']['strategy return']:+.1%}",
                "buy and hold": f"{_r['summary']['buy and hold return']:+.1%}",
                "from": f"{_r['chart'].index[0]:%Y-%m-%d}",
                "to": f"{_r['chart'].index[-1]:%Y-%m-%d}"}

    _rows = []
    for _side, _log in logs.items():
        for _, _t in _log.iterrows():
            _rows.append({
                "side": _side, "rank": _t["rank"], "trade": _t["trade"] + 1,
                "return %": round(100 * _t["return"], 2), "R": round(_t["R"], 2),
                "entry time": stamp(_t["entry time"]), "exit time": stamp(_t["exit time"]),
                "why it ended": _t["reason"], "candles held": _t["candles"],
                "entry": round(_t["entry"], 2), "first stop": round(_t["initial stop"], 2),
                "exit": round(_t["exit"], 2), "line A": stamp(_t["line A"]),
                "line A price": round(_t["line A price"], 2), "line B": stamp(_t["line B"]),
                "lines drawn at": stamp(_t["lines drawn at"]),
                "drawn because": _t["drawn because"]})
    mo.vstack([
        mo.md("**The result**, each side on its own $10,000 wallet:"),
        mo.ui.table(pd.DataFrame([_summary(_s) for _s in ("long", "short")]), selection=None,
                    show_download=False),
        mo.md("**Every trade**, best first within each side (sort, filter or download it):"),
        mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=15)])
    return (logs,)


@app.cell(hide_code=True)
def _(book, logs, mo, noises, symbol, top_n, trade_chart, walk_knobs, walked):
    _noise = (noises({_tf: book.full[_tf] for _tf in ("1D", "4H", "1H", "15m")},
                     walk_knobs["noise_window"])
              if walk_knobs["line_band"] or walk_knobs["price_band"] else None)
    _out = [mo.md(f"""
    ### The {top_n.value} most profitable trades of each side

    Each chart shows the 15-minute candles from the point A of the line the trade bounced off
    (or three sessions before the entry, if that is earlier) to two sessions after its exit,
    with every line that stood when it was set up, one per chart and side, in the styles of the
    key above. The traded line is tinted, with its anchors ◆ A and B (and its contact zone with
    noise bands).
    """)]
    for _side in ("long", "short"):
        _best = logs[_side].head(top_n.value)
        _out.append(mo.md(f"#### {_side.capitalize()}: {len(_best)} of "
                          f"{len(logs[_side])} trades"))
        for _, _t in _best.iterrows():
            _fig, _story = trade_chart(
                walked[_side], book.full["15m"], int(_t["trade"]), walk_knobs["entry"], 3, 2,
                quiet=walk_knobs["trade_trim"], hold=walk_knobs["hold"], noise=_noise,
                bands=(walk_knobs["line_band"], walk_knobs["price_band"]),
                title=f"{symbol} · {_side} trade {int(_t['trade']) + 1} · rank {_t['rank']} · "
                      f"{_t['return']:+.2%} after costs · {_t['R']:+.2f} R")
            _out += [mo.md(f"**{_side.capitalize()} trade {int(_t['trade']) + 1}** (rank "
                           f"{_t['rank']}). {_story}"), _fig]
    mo.vstack(_out)
    return


if __name__ == "__main__":
    app.run()
