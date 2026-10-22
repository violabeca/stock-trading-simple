"""The figures of the classic_signals notebook, on the repo's palette (``trading_simple.plots``).

Colors keep one meaning everywhere: blue rising / red falling / gray sideways (candles, trends,
stretches), gray for the Bollinger Bands, black for turning points and major levels, pink for
report days, and blue / green / purple for the stock / its sector fund / the market fund.
Legends sit under the charts, so they never cover a title.

Minute charts send New York wall-clock times as text (a European browser would shift numeric
times by its own time zone) and cut nights, weekends and holidays out of the time axis, so the
candles of consecutive sessions sit side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_simple.plots import _GRID, _INK, _LONG, _MUTED, _SHORT, DIVERGING, SERIES, _base

UP, DOWN, SIDEWAYS = "#2a78d6", "#d03b3b", "#b9b7b0"     # rising, falling, neither
UP_PALE, DOWN_PALE = "#a9c9ef", "#eeb2b2"                 # the same outside regular hours
BAND, BAND_FILL = _MUTED, "rgba(137, 135, 129, 0.12)"
LEVEL, REPORT = _INK, SERIES[6]
ZONE_FILL = "rgba(11, 11, 11, 0.07)"
AVERAGES = (SERIES[2], SERIES[3])                          # amber, dark green
ROLES = {"stock": SERIES[0], "sector": SERIES[1], "market": SERIES[4]}
MOVE = SERIES[3]
STATUS = {"complete": "#0ca30c", "half-day": "#9bc4ee", "short": "#eda100",
          "missing": "#d03b3b"}
WORDS = {1.0: "up", 0.0: "sideways", -1.0: "down"}
#: The switchboard's strip colors: code -> (color, meaning).
STRIP = {0: ("#f4f3ef", "nothing"), 1: (_LONG, "long"), 2: (_SHORT, "short"),
         3: ("#dcdad3", "both sides allowed"), 4: ("#5f5d58", "blocked / neither side")}
_MARGIN_TOP, _MARGIN_BOTTOM = 50, 120


def _legend_below(fig: go.Figure, height: int) -> go.Figure:
    """Put the legend under the plot (about 60 pixels below its lower edge, under the axis
    labels), so it never covers the title."""
    plot = max(height - _MARGIN_TOP - _MARGIN_BOTTOM, 80)
    fig.update_layout(margin={"b": _MARGIN_BOTTOM},
                      legend={"orientation": "h", "x": 0.0, "xanchor": "left",
                              "y": -60 / plot, "yanchor": "top"})
    return fig


def _finish(fig: go.Figure, *, title: str, height: int) -> go.Figure:
    """The shared look (``plots._base``) with the legend under the plot."""
    return _legend_below(_base(fig, title=title, height=height), height)


def _wall(times) -> np.ndarray:
    """Times as New York wall-clock text ("2026-03-05 10:31")."""
    t = pd.DatetimeIndex(times)
    t = t.tz_convert("America/New_York").tz_localize(None) if t.tz is not None else t
    return np.asarray(t.strftime("%Y-%m-%d %H:%M"))


def _breaks(times, *, min_gap: str = "5min") -> list[dict]:
    """Axis breaks that hide every gap between candles longer than ``min_gap`` (nights,
    weekends, holidays), as value + length pairs on wall-clock text."""
    t = pd.DatetimeIndex(times)
    t = t.tz_convert("America/New_York").tz_localize(None) if t.tz is not None else t
    if len(t) < 2:
        return []
    step = pd.Series(t).diff().median()
    gaps = pd.Series(t).diff()
    return [{"values": [(t[i - 1] + step).strftime("%Y-%m-%d %H:%M")],
             "dvalue": int((gap - step).total_seconds() * 1000)}
            for i, gap in gaps[gaps > pd.Timedelta(min_gap)].items()]


def _span(times) -> list[str]:
    """An x range from the first to the last time (wall-clock text). Plotly pads its own range
    in clock time, nights included, which leaves a wide empty strip once the nights are cut."""
    w = _wall(times)
    return [w[0], w[-1]] if len(w) else None


def _day_breaks(index: pd.DatetimeIndex) -> list[dict]:
    """Axis breaks for a daily chart: every calendar day without a session (weekends and
    holidays) as one list of dates. Mixing a weekend PATTERN break with a list of dates crashes
    plotly.js."""
    missing = pd.date_range(index.min(), index.max(), freq="D").difference(index)
    return [{"values": [d.strftime("%Y-%m-%d") for d in missing]}]


def _discrete(colors: list[str]) -> list:
    """A step color scale for the codes 0 .. n-1 (use ``zmin=-0.5, zmax=n-0.5``)."""
    n, out = len(colors), []
    for i, c in enumerate(colors):
        out += [[i / n, c], [(i + 1) / n, c]]
    return out


def _segments(x0, x1, y) -> tuple[list, list]:
    """Horizontal line pieces from ``x0[i]`` to ``x1[i]`` at ``y[i]``, as one trace's x and y
    (pieces separated by None)."""
    xs, ys = [], []
    for a, b, v in zip(x0, x1, y, strict=True):
        xs += [a, b, None]
        ys += [v, v, None]
    return xs, ys


def _key(fig: go.Figure, name: str, color: str, symbol: str = "square", **where) -> None:
    """A legend entry without data (to name the colors of a heatmap or a ribbon)."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                             marker={"color": color, "size": 10, "symbol": symbol}), **where)


def _ribbon(fig: go.Figure, index, trends: dict[str, pd.Series], **where) -> None:
    """One colored row per trend (blue up, gray sideways, red down; blank: not known yet)."""
    labels = list(trends)
    z = np.vstack([trends[k].reindex(index).to_numpy(float) for k in labels])
    words = np.vectorize(lambda v: WORDS.get(v, "not known yet"))(np.nan_to_num(z, nan=9.0))
    fig.add_trace(go.Heatmap(
        z=z + 1, x=index, y=labels, colorscale=_discrete([DOWN, SIDEWAYS, UP]), zmin=-0.5,
        zmax=2.5, showscale=False, ygap=2, customdata=words,
        hovertemplate="%{y} · %{x|%Y-%m-%d}: trend %{customdata}<extra></extra>"), **where)


def _why_turning_point(r: pd.Series, tolerance: float | None, same_kind: bool = False) -> str:
    """The hover text of one turning point: what it is, from when the strategy may use it and
    why, and (with the grouping distance of one day's levels) why it is or is not part of a
    major level."""
    more = "higher high" if r["kind"] == "high" else "lower low"
    other = f"swing {r['kind']}" if same_kind else "turning point"
    lines = [f"<b>swing {r['kind']} on {r['day']:%Y-%m-%d} at {r['price']:.2f}</b>",
             f"usable from the session after {r['confirmed']:%Y-%m-%d}: only once the "
             f"{int(r.get('span', 0))} sessions",
             f"after it had passed without a {more} was it clear that the price turned here"]
    if tolerance is not None:
        gap = r.get("nearest", np.nan)
        if np.isfinite(r["level"]):
            lines.append(f"part of the major level {r['level']:.2f}: its nearest other {other} "
                         f"is {gap:.2f} away,")
            lines.append(f"within the grouping distance of {tolerance:.2f}")
        elif r["touches"] >= 2:
            lines.append(f"not part of a major level: its group has only {int(r['touches'])} "
                         "turning points, fewer than needed")
        elif np.isfinite(gap):
            lines.append(f"stands alone: its nearest other {other} is {gap:.2f} away,")
            lines.append(f"more than the grouping distance of {tolerance:.2f}")
        else:
            lines.append(f"stands alone: it is the only {other} in the memory window")
    return "<br>".join(lines)


def _turning_points(fig: go.Figure, tp: pd.DataFrame, *, tolerance: float | None = None,
                    same_kind: bool = False, **where) -> None:
    """Turning points as black triangles: a swing high points down from above, a swing low
    points up from below. With ``tolerance`` (the grouping distance of one day's levels,
    ``indicators.explain_levels``), points that belong to a major level are filled, the others
    hollow, and the hover says why; ``same_kind``: highs only group with highs and lows with
    lows."""
    for kind, symbol in (("high", "triangle-down"), ("low", "triangle-up")):
        part = tp[tp["kind"] == kind]
        if tolerance is None:
            groups = [("", part, _INK, symbol)]
        else:
            groups = [(" (part of a major level)", part[part["level"].notna()], _INK, symbol),
                      (" (not part of one)", part[part["level"].isna()], _MUTED,
                       symbol + "-open")]
        for suffix, rows, color, sym in groups:
            if not len(rows):
                continue
            fig.add_trace(go.Scatter(
                x=rows["day"], y=rows["price"], mode="markers",
                name=f"turning point: swing {kind}{suffix}",
                marker={"symbol": sym, "size": 9, "color": color,
                        "line": {"color": color, "width": 1.5}},
                text=[_why_turning_point(r, tolerance, same_kind) for _, r in rows.iterrows()],
                hovertemplate="%{text}<extra></extra>"), **where)


def coverage_figure(counts: pd.DataFrame, *, title: str = "") -> go.Figure:
    """One row per symbol, one cell per session: green = complete (390 one-minute candles),
    light blue = half-day (the market closed at 13:00), amber = some minutes missing, red = no
    candle at all on a day the others traded.

    Args:
        counts: Sessions x symbols (``data.coverage_counts``).
        title: Figure title.
    """
    labels = list(STATUS)
    code = np.select([counts == 390, counts == 210, counts > 0], [0, 1, 2], 3).T
    words = np.array(labels)[code]
    fig = go.Figure(go.Heatmap(
        z=code, x=counts.index, y=list(counts.columns), colorscale=_discrete(list(STATUS.values())),
        zmin=-0.5, zmax=len(labels) - 0.5, showscale=False, xgap=0, ygap=2,
        customdata=np.dstack([counts.T.to_numpy(), words]),
        hovertemplate="%{y} · %{x|%a %Y-%m-%d}: %{customdata[0]} candles "
                      "(%{customdata[1]})<extra></extra>"))
    for label, color in STATUS.items():
        _key(fig, label, color)
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title or "Data check: one-minute candles per session", height=300)


def overview_figure(name: str, days: pd.DataFrame, *, averages: dict[str, pd.Series],
                    trends: dict[str, pd.Series], levels: pd.DataFrame,
                    turning_points: pd.DataFrame | None = None,
                    reports: pd.DataFrame | None = None, title: str = "") -> go.Figure:
    """The daily chart of one stock and, below it, its trend ribbons.

    Top: candles, moving averages, the turning points (black triangles: a swing high points
    down from above, a swing low up from below), the major levels (a short black dash on every
    day a level was known: a long row of dashes is a level that held for months, a thicker dash
    has more turning points) and the report days (pink). Bottom: one ribbon per trend (the
    stock, its sector fund, the market fund), each day colored by the trend judged at that
    day's close: blue up, gray sideways, red down.

    Args:
        name: The stock.
        days: Its daily candles (the days shown).
        averages: Label -> daily moving average.
        trends: Ribbon label -> daily trend state (1, 0, -1).
        levels: ``indicators.major_levels`` of the stock.
        turning_points: ``indicators.pivots`` of the stock, or None.
        reports: Report rows of the stock (``data.earnings_dates``), or None.
        title: Figure title.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=days.index, open=days["open"], high=days["high"], low=days["low"], close=days["close"],
        name="daily candles", increasing={"line": {"color": UP}, "fillcolor": UP},
        decreasing={"line": {"color": DOWN}, "fillcolor": DOWN}), row=1, col=1)
    for (label, line), color in zip(averages.items(), AVERAGES, strict=False):
        fig.add_trace(go.Scatter(x=line.index, y=line, name=label, mode="lines",
                                 line={"color": color, "width": 1.6}), row=1, col=1)
    if turning_points is not None and len(turning_points):
        shown = turning_points[turning_points["day"].between(days.index.min(), days.index.max())]
        _turning_points(fig, shown, row=1, col=1)
    if len(levels):
        lv = levels[levels["day"].isin(days.index)]
        fig.add_trace(go.Scatter(
            x=lv["day"], y=lv["price"], mode="markers", name="major level (a dash per day)",
            marker={"symbol": "line-ew", "size": 9, "color": LEVEL,
                    "line": {"color": LEVEL, "width": np.clip(lv["touches"], 2, 5)},
                    "opacity": 0.55},
            customdata=np.column_stack([lv["touches"], lv["highs"], lv["lows"]]),
            hovertemplate="%{x|%Y-%m-%d}: level %{y:.2f} · %{customdata[0]} turning points "
                          "(%{customdata[1]} highs, %{customdata[2]} lows)<extra></extra>"),
            row=1, col=1)
    if reports is not None and len(reports):
        lo, hi = float(days["low"].min()), float(days["high"].max())
        shown = reports[(reports["reaction day"] >= days.index.min())
                        & (reports["reaction day"] <= days.index.max())]
        xs, ys, text = [], [], []
        for _, r in shown.iterrows():
            xs += [r["reaction day"], r["reaction day"], None]
            ys += [lo, hi, None]
            text += [f"report {r['when']}"] * 2 + [None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="report day",
                                 line={"color": REPORT, "width": 1.2, "dash": "dot"},
                                 text=text, hovertemplate="%{x|%Y-%m-%d}: %{text}<extra></extra>"),
                      row=1, col=1)
    _ribbon(fig, days.index, trends, row=2, col=1)
    for word, color in (("trend up", UP), ("trend sideways", SIDEWAYS), ("trend down", DOWN)):
        _key(fig, word, color, row=2, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=_day_breaks(days.index))
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="price", row=1, col=1)
    return _finish(fig, title=title or f"{name}: the daily picture", height=700)


def levels_figure(name: str, days: pd.DataFrame, explained: dict, *, title: str = "") -> go.Figure:
    """How the major levels known at the open of one session are built.

    The daily candles of the memory window (pale), every turning point in it (black when it
    belongs to a major level, hollow gray when it stands alone), and each major level: a dashed
    line at the average price of its turning points, inside a gray zone that spans those
    turning points plus half the grouping distance on each side.

    Args:
        name: The stock.
        days: Its daily candles.
        explained: ``indicators.explain_levels`` for the session.
        title: Figure title.
    """
    window = days.loc[explained["window_start"]:explained["known_until"]]
    tp, lv, tol = explained["turning_points"], explained["levels"], explained["tolerance"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=window.index, open=window["open"], high=window["high"], low=window["low"],
        close=window["close"], name="daily candles", opacity=0.4,
        increasing={"line": {"color": UP}, "fillcolor": UP},
        decreasing={"line": {"color": DOWN}, "fillcolor": DOWN}))
    if len(window):
        x0, x1 = window.index[0], window.index[-1]
        for i, (_, r) in enumerate(lv.iterrows()):
            lo, hi = r["bottom"] - tol / 2, r["top"] + tol / 2
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0], y=[lo, lo, hi, hi, lo], mode="lines", fill="toself",
                fillcolor=ZONE_FILL, line={"width": 0}, name="grouping zone",
                legendgroup="zone", showlegend=i == 0, hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[r["price"], r["price"]], mode="lines", name="major level",
                legendgroup="level", showlegend=i == 0,
                line={"color": LEVEL, "width": 1.5, "dash": "dash"},
                hovertemplate=f"major level {r['price']:.2f}: {r['touches']} turning points "
                              f"({r['highs']} highs, {r['lows']} lows)<extra></extra>"))
            fig.add_annotation(x=x1, y=r["price"], xanchor="right", yanchor="bottom",
                               showarrow=False, font={"size": 11, "color": _INK},
                               text=f"{r['price']:.2f} · {r['touches']} turning points")
    if len(tp):
        _turning_points(fig, tp, tolerance=tol, same_kind=not explained.get("mix", True))
    fig.update_xaxes(rangeslider_visible=False,
                     rangebreaks=_day_breaks(window.index) if len(window) else [])
    fig.update_yaxes(title_text="price")
    return _finish(fig, title=title or f"{name}: how the major levels are built", height=560)


def trend_figure(name: str, days: pd.DataFrame, trend: pd.DataFrame, reasons: pd.Series, *,
                 length: int, slope_days: int, flat: float, title: str = "") -> go.Figure:
    """How the daily trend is judged, day by day, in three panels.

    1. The daily close (a dot per day, colored by the trend: blue up, gray sideways, red down)
       and its moving average. Hover a dot for the reason in words.
    2. How much the average rose (or fell) over the last ``slope_days`` days, against the flat
       zone (gray band, ± ``flat`` typical daily moves): above the band it is rising, below it
       falling, inside it flat.
    3. The resulting trend ribbon.

    Args:
        name: The symbol.
        days: Its daily candles (the days shown).
        trend: ``indicators.trend_state`` of those days.
        reasons: ``indicators.explain_trend`` of those days.
        length: Days in the average.
        slope_days: Days the rise is measured over.
        flat: The flat zone, in typical daily moves.
        title: Figure title.
    """
    tr = trend.reindex(days.index)
    why = reasons.reindex(days.index)
    zone = flat * tr["typical_move"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.55, 0.3, 0.15],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=days.index, y=days["close"], mode="lines", name="daily close",
                             line={"color": _MUTED, "width": 1}, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days.index, y=tr["average"], mode="lines",
                             name=f"{length}-day average", line={"color": AVERAGES[1], "width": 2},
                             hovertemplate="%{x|%Y-%m-%d}: average %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    for value, word, color in ((1.0, "up", UP), (0.0, "sideways", SIDEWAYS),
                               (-1.0, "down", DOWN)):
        m = (tr["state"] == value).to_numpy()
        fig.add_trace(go.Scatter(x=days.index[m], y=days["close"][m], mode="markers",
                                 name=f"a day judged {word}", marker={"color": color, "size": 6},
                                 text=why[m], hovertemplate="%{x|%Y-%m-%d}: %{text}"
                                                            "<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days.index, y=zone, mode="lines", line={"width": 0},
                             showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=days.index, y=-zone, mode="lines", line={"width": 0},
                             fill="tonexty", fillcolor=BAND_FILL,
                             name=f"flat zone (± {flat:g} typical daily moves)",
                             hovertemplate="%{x|%Y-%m-%d}: flat zone ± %{customdata:.2f}"
                                           "<extra></extra>", customdata=zone), row=2, col=1)
    fig.add_trace(go.Scatter(x=days.index, y=tr["rise"], mode="lines",
                             name=f"the average's rise over {slope_days} days",
                             line={"color": AVERAGES[1], "width": 1.5},
                             hovertemplate="%{x|%Y-%m-%d}: the average moved %{y:+.2f} over "
                                           f"{slope_days} days<extra></extra>"), row=2, col=1)
    _ribbon(fig, days.index, {"trend": tr["state"]}, row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=_day_breaks(days.index))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="rise", row=2, col=1)
    return _finish(fig, title=title or f"{name}: how the trend is judged", height=640)


def conditions_figure(trends: dict[str, pd.Series], *, title: str = "") -> go.Figure:
    """For every symbol, the share of days its daily trend was up, sideways or down (days whose
    trend is not known yet are left out).

    Args:
        trends: Label -> daily trend state over the same period.
        title: Figure title.
    """
    fig = go.Figure()
    labels = list(trends)
    for value, word, color in ((-1.0, "down", DOWN), (0.0, "sideways", SIDEWAYS),
                               (1.0, "up", UP)):
        share = [float((trends[k].dropna() == value).mean()) for k in labels]
        days = [int((trends[k].dropna() == value).sum()) for k in labels]
        fig.add_trace(go.Bar(y=labels, x=share, name=word, orientation="h",
                             marker={"color": color, "line": {"color": "white", "width": 2}},
                             customdata=days,
                             hovertemplate=f"%{{y}}: {word} on %{{x:.0%}} of the days "
                                           "(%{customdata} days)<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title or "How the days split into up, sideways and down",
                   height=380)


def day_figure(name: str, minutes: pd.DataFrame, *, bands: pd.DataFrame, average: pd.Series,
               average_label: str, others: dict[str, pd.Series], move_pct: pd.Series,
               context: pd.DataFrame, levels: pd.DataFrame, reports: pd.DataFrame,
               move_label: str = "typical move", title: str = "") -> go.Figure:
    """A few sessions of one stock, minute by minute, in five panels sharing one time axis.

    1. Candles with the Bollinger Bands (gray, shaded between), the moving average, the major
       levels known at each session's open (black dashes) and the previous session's high and
       low (gray dots). With the hours outside the regular session loaded (a ``regular``
       column), their candles are pale, dotted lines mark each regular open and close, and the
       pre-market's high and low (gray dash-dots) run across the regular session.
    2. Where the close sits in the band: 0 on the lower band, 1 on the upper band; dots mark the
       closes outside the band (blue above, red below).
    3. The stock against its sector fund and the market fund, in percent since the first candle
       shown.
    4. The typical move: how far the price typically travels within one candle of the chosen
       size (high to low, average of the last few candles), in percent of the price, as known at
       each moment. The trailing stop will keep a multiple of it behind the price.
    5. Volume.

    Args:
        name: The stock.
        minutes: Its candles for the sessions shown.
        bands: ``indicators.bollinger`` on those candles.
        average: A moving average on those candles.
        average_label: Its legend label.
        others: Label -> close of the sector fund and the market fund (any index; aligned here).
        move_pct: Typical move in percent of the price, on the same candles.
        move_label: What the typical move is, for the legend and the hover.
        context: ``indicators.session_context`` rows of the sessions shown.
        levels: ``indicators.major_levels`` of the stock.
        reports: Report rows of the stock.
        title: Figure title.
    """
    x = _wall(minutes.index)
    regular = (minutes["regular"].to_numpy(bool) if "regular" in minutes
               else np.ones(len(minutes), bool))
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[0.44, 0.12, 0.16, 0.12, 0.16])
    for part, label, up, down in ((regular, "1-minute candles, regular hours", UP, DOWN),
                                  (~regular, "candles outside regular hours", UP_PALE,
                                   DOWN_PALE)):
        if part.any():
            c = minutes[part]
            fig.add_trace(go.Candlestick(
                x=x[part], open=c["open"], high=c["high"], low=c["low"], close=c["close"],
                name=label, increasing={"line": {"color": up, "width": 1}, "fillcolor": up},
                decreasing={"line": {"color": down, "width": 1}, "fillcolor": down}),
                row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=bands["upper"], mode="lines", name="Bollinger Bands",
                             legendgroup="bands", line={"color": BAND, "width": 1}),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=bands["lower"], mode="lines", showlegend=False,
                             legendgroup="bands", line={"color": BAND, "width": 1},
                             fill="tonexty", fillcolor=BAND_FILL), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=bands["middle"], mode="lines", name="band middle",
                             line={"color": BAND, "width": 1, "dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=average, mode="lines", name=average_label,
                             line={"color": AVERAGES[0], "width": 1.4}), row=1, col=1)
    session = minutes.index.tz_localize(None).normalize()
    starts, ends = {}, {}                       # each session's regular open and close (as x)
    for day, part in pd.Series(x[regular], index=session[regular]).groupby(level=0):
        starts[day], ends[day] = part.iloc[0], part.iloc[-1]
    if (~regular).any() and starts:
        lo, hi = float(minutes["low"].min()), float(minutes["high"].max())
        xs, ys = [], []
        for d in starts:
            xs += [starts[d], starts[d], None, ends[d], ends[d], None]
            ys += [lo, hi, None, lo, hi, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="regular session: open and close",
                                 line={"color": _MUTED, "width": 1, "dash": "dot"},
                                 hoverinfo="skip"), row=1, col=1)
    lv = levels[levels["day"].isin(list(starts))] if len(levels) else levels
    if len(lv):
        xs, ys = _segments([starts[d] for d in lv["day"]], [ends[d] for d in lv["day"]],
                           lv["price"])
        hover = [f"major level {p:.2f} · {t} turning points" for p, t in
                 zip(lv["price"], lv["touches"], strict=True) for _ in range(3)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="major level",
                                 line={"color": LEVEL, "width": 1.5, "dash": "dash"},
                                 text=hover, hovertemplate="%{text}<extra></extra>"),
                      row=1, col=1)
    ctx = context[context["day"].isin(list(starts))]
    for column, label in (("pre-market high", "pre-market high"),
                          ("pre-market low", "pre-market low")):
        if column in ctx and ctx[column].notna().any():
            xs, ys = _segments([starts[d] for d in ctx["day"]], [ends[d] for d in ctx["day"]],
                               ctx[column])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="pre-market high and low",
                                     legendgroup="pre", showlegend=column == "pre-market high",
                                     line={"color": _MUTED, "width": 1.2, "dash": "dashdot"},
                                     hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"),
                          row=1, col=1)
    for column, label in (("previous high", "previous session's high"),
                          ("previous low", "previous session's low")):
        if len(ctx) and ctx[column].notna().any():
            xs, ys = _segments([starts[d] for d in ctx["day"]], [ends[d] for d in ctx["day"]],
                               ctx[column])
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     name="previous session's high and low",
                                     legendgroup="previous", showlegend=column == "previous high",
                                     line={"color": _MUTED, "width": 1, "dash": "dot"},
                                     hovertemplate=f"{label}: %{{y:.2f}}<extra></extra>"),
                          row=1, col=1)
    shown = reports[reports["reaction day"].isin(list(starts))] if len(reports) else reports
    for _, r in shown.iterrows():
        at = starts[r["reaction day"]]
        fig.add_trace(go.Scatter(x=[at, at], y=[minutes["low"].min(), minutes["high"].max()],
                                 mode="lines", name="report day",
                                 line={"color": REPORT, "width": 2, "dash": "dot"},
                                 hovertemplate=f"report {r['when']}<extra></extra>"),
                      row=1, col=1)
    pos = bands["position"]
    fig.add_trace(go.Scatter(x=x, y=pos, mode="lines", name="position in the band",
                             line={"color": BAND, "width": 1}, showlegend=False,
                             hovertemplate="%{x}: %{y:.2f} of the way from lower to upper band"
                                           "<extra></extra>"), row=2, col=1)
    for ref in (0.0, 1.0):
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[ref, ref], mode="lines", showlegend=False,
                                 hoverinfo="skip", line={"color": _GRID, "width": 1}),
                      row=2, col=1)
    for mask, color, word in ((pos > 1, UP, "above the upper band"),
                              (pos < 0, DOWN, "below the lower band")):
        fig.add_trace(go.Scatter(x=x[mask.to_numpy()], y=pos[mask], mode="markers",
                                 name=f"close {word}", marker={"color": color, "size": 4},
                                 hovertemplate=f"%{{x}}: close {word}<extra></extra>"),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=(minutes["close"] / minutes["close"].iloc[0] - 1) * 100,
                             mode="lines", name=name, line={"color": ROLES["stock"],
                                                            "width": 1.4}), row=3, col=1)
    for (label, close), color in zip(others.items(), (ROLES["sector"], ROLES["market"]),
                                     strict=False):
        aligned = close.reindex(minutes.index).ffill()
        fig.add_trace(go.Scatter(x=x, y=(aligned / aligned.dropna().iloc[0] - 1) * 100
                                 if aligned.notna().any() else aligned,
                                 mode="lines", name=label, line={"color": color, "width": 1.2}),
                      row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=move_pct, mode="lines", name=move_label,
                             line={"color": MOVE, "width": 1.2, "shape": "hv"},
                             hovertemplate=f"%{{x}}: {move_label}: %{{y:.2f}}% of the price"
                                           "<extra></extra>"), row=4, col=1)
    rising = (minutes["close"] >= minutes["open"]).to_numpy()
    colors = np.where(regular, np.where(rising, UP, DOWN), np.where(rising, UP_PALE, DOWN_PALE))
    fig.add_trace(go.Bar(x=x, y=minutes["volume"], name="volume", showlegend=False,
                         marker={"color": colors, "opacity": 0.6, "line": {"width": 0}}),
                  row=5, col=1)
    fig.update_xaxes(type="date", rangebreaks=_breaks(minutes.index), rangeslider_visible=False,
                     range=_span(minutes.index))
    for row, label in ((1, "price"), (2, "in band"), (3, "% change"), (4, "typical move %"),
                       (5, "volume")):
        fig.update_yaxes(title_text=label, row=row, col=1)
    return _finish(fig, title=title or f"{name}: minute by minute", height=1000)


def move_figure(name: str, candles: pd.DataFrame, explained: pd.DataFrame, *, size: int,
                length: int, title: str = "") -> go.Figure:
    """How the typical move is computed, candle by candle, in two panels sharing a time axis.

    1. The ``size``-minute candles.
    2. Each candle's move as a gray bar: its high minus its low, stretched to the previous
       candle's close if that lay outside (the "true range"; the overnight gap is left out), in
       percent of the price. The line is the typical move the strategy knows while that candle
       runs: the average of the last ``length`` completed candles' moves. A bar above the line
       is a candle that moved more than usual.

    Args:
        name: The stock.
        candles: Its ``size``-minute candles for the sessions shown.
        explained: ``indicators.explain_move`` of those candles (computed on a longer history,
            so the average is ready on the first candle shown).
        size: Minutes per candle.
        length: Candles averaged.
        title: Figure title.
    """
    x = _wall(candles.index)
    ex = explained.reindex(candles.index)
    ends = _wall(candles.index + pd.Timedelta(minutes=size))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.45])
    fig.add_trace(go.Candlestick(
        x=x, open=candles["open"], high=candles["high"], low=candles["low"],
        close=candles["close"], name=f"{size}-minute candles",
        increasing={"line": {"color": UP, "width": 1}, "fillcolor": UP},
        decreasing={"line": {"color": DOWN, "width": 1}, "fillcolor": DOWN}), row=1, col=1)
    fig.add_trace(go.Bar(
        x=x, y=ex["range_pct"], name="this candle's move (high to low), % of the price",
        marker={"color": BAND, "opacity": 0.55, "line": {"width": 0}},
        customdata=np.column_stack([np.asarray([e[-5:] for e in ends]), ex["range"]]),
        hovertemplate="%{x|%H:%M}-%{customdata[0]}: this candle moved %{y:.2f}% "
                      "(%{customdata[1]:.2f})<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=ex["known_pct"], mode="lines", line={"color": MOVE, "width": 2, "shape": "hv"},
        name=f"typical move known at the candle's start (average of the last {length})",
        hovertemplate=f"%{{x}}: a {size}-minute candle typically moves %{{y:.2f}}% "
                      f"(average of the last {length} completed candles)<extra></extra>"),
        row=2, col=1)
    fig.update_xaxes(type="date", rangebreaks=_breaks(candles.index, min_gap=f"{size + 1}min"),
                     rangeslider_visible=False, range=_span(candles.index))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="move %", row=2, col=1)
    return _finish(fig, title=title or f"{name}: how the typical move is computed", height=620)


def before_open_figure(name: str, pre: pd.DataFrame, *, title: str = "") -> go.Figure:
    """The hours before each open (after-hours, overnight, pre-market), session by session, in
    two panels.

    1. The move before the open: from the previous session's close to the last trade before
       09:30, in percent: blue up, red down.
    2. The volume traded since the previous close against its usual level (its average over the
       previous sessions): 1 is a usual night (dashed line), 3 means three times as busy.

    Args:
        name: The stock.
        pre: ``indicators.before_open`` rows of the sessions shown.
        title: Figure title.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.55, 0.45])
    move = pre["move %"]
    fig.add_trace(go.Bar(
        x=pre.index, y=move, name="move before the open (%)",
        marker={"color": np.where(move.fillna(0) >= 0, UP, DOWN), "line": {"width": 0}},
        customdata=np.column_stack([pre["previous close"], pre["last"]]),
        hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}% (previous close %{customdata[0]:.2f}, last "
                      "trade before the open %{customdata[1]:.2f})<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(
        x=pre.index, y=pre["volume vs usual"], name="volume since the previous close ÷ usual",
        marker={"color": _MUTED, "line": {"width": 0}}, customdata=pre["volume"],
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} × the usual volume between close and open "
                      "(%{customdata:,.0f} shares)<extra></extra>"), row=2, col=1)
    if len(pre):
        fig.add_trace(go.Scatter(x=[pre.index[0], pre.index[-1]], y=[1, 1], mode="lines",
                                 name="a usual night (1 ×)", hoverinfo="skip",
                                 line={"color": _INK, "width": 1, "dash": "dash"}), row=2, col=1)
        fig.update_xaxes(rangebreaks=_day_breaks(pre.index))
    fig.update_yaxes(title_text="move %", row=1, col=1)
    fig.update_yaxes(title_text="× usual", row=2, col=1)
    return _finish(fig, title=title or f"{name}: the hours before each open", height=560)


def band_position_figure(positions: dict[str, pd.Series], *, title: str = "") -> go.Figure:
    """How often the close sits where in its Bollinger Band, over every minute: one histogram
    per stock, clipped to -1 .. 2 (0 = lower band, 1 = upper band). The tails beyond 0 and 1 are
    the stretched closes a band trigger reacts to.

    Args:
        positions: Stock -> ``bollinger(...)["position"]``.
        title: Figure title.
    """
    fig = go.Figure()
    edges = np.linspace(-1.0, 2.0, 61)                     # bins of 0.05, counted here: the
    for (name, pos), color in zip(positions.items(), SERIES, strict=False):   # figure keeps 60
        counts, _ = np.histogram(pos.dropna().clip(-1, 2), bins=edges)        # numbers, not
        share = counts / max(counts.sum(), 1) * 100                           # every minute
        fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=share, width=0.05, name=name,
                             opacity=0.6, marker={"color": color, "line": {"width": 0}},
                             hovertemplate=f"{name}: %{{y:.2f}}% of minutes near %{{x:.2f}}"
                                           "<extra></extra>"))
    fig.update_layout(barmode="overlay")
    for ref, word in ((0.0, "lower band"), (1.0, "upper band")):
        fig.add_vline(x=ref, line={"color": _INK, "width": 1, "dash": "dash"},
                      annotation_text=word, annotation_position="top")
    fig.update_xaxes(title_text="where the close sits in the band")
    fig.update_yaxes(title_text="% of minutes")
    return _finish(fig, title=title or "Where the close sits in its band, all minutes",
                   height=420)


def _stop_paths(stop: pd.Series, trades: pd.DataFrame, index) -> tuple[list, list]:
    """Each trade's own trailing stop from its entry to its exit, cut to ``index``, as one
    trace's x (wall-clock text) and y, with a gap between trades: back-to-back trades would
    otherwise join into one line."""
    xs, ys = [], []
    for entry, out in zip(trades["entry time"], trades["exit time"], strict=True):
        path = stop.loc[entry:out]
        path = path[path.index.isin(index)].dropna()
        if len(path):
            xs += [*_wall(path.index), None]
            ys += [*path.to_numpy(float), None]
    return xs, ys


def _price_range(candles: pd.DataFrame, more: list, pad: float = 0.08) -> list[float]:
    """A price axis that fits the candles and ``more`` prices (None skipped), with a margin of
    ``pad`` of the span on each side. A band that balloons after a gap may then run off the
    chart instead of squashing the candles."""
    extra = np.asarray([v for v in more if v is not None], float)
    lo = float(np.nanmin(np.append(candles["low"].to_numpy(float), extra)))
    hi = float(np.nanmax(np.append(candles["high"].to_numpy(float), extra)))
    margin = (hi - lo) * pad or hi * 0.001
    return [lo - margin, hi + margin]


def _trade_marks(fig: go.Figure, trades: pd.DataFrame, *, within=None, **where) -> None:
    """Entries (blue triangle up for a long, orange triangle down for a short) and exits (black
    x) of the trades, with their result on hover; with ``within``, only the entries and exits
    at those times."""
    entries = trades if within is None else trades[trades["entry time"].isin(within)]
    exits = trades if within is None else trades[trades["exit time"].isin(within)]
    for side, symbol, color in (("long", "triangle-up", _LONG), ("short", "triangle-down", _SHORT)):
        t = entries[entries["side"] == side]
        if len(t):
            fig.add_trace(go.Scatter(
                x=_wall(t["entry time"]), y=t["entry price"], mode="markers", name=f"{side} entry",
                marker={"symbol": symbol, "size": 12, "color": color,
                        "line": {"color": "white", "width": 1}},
                customdata=np.column_stack([t["return %"], t["exit reason"]]),
                hovertemplate=f"{side} entry at %{{y:.2f}}<br>result %{{customdata[0]:+.2f}}% "
                              "(exit: %{customdata[1]})<extra></extra>"), **where)
    if len(exits):
        fig.add_trace(go.Scatter(
            x=_wall(exits["exit time"]), y=exits["exit price"], mode="markers", name="exit",
            marker={"symbol": "x", "size": 10, "color": _INK},
            customdata=np.column_stack([exits["return %"], exits["exit reason"]]),
            hovertemplate="exit at %{y:.2f}: %{customdata[1]}<br>result %{customdata[0]:+.2f}%"
                          "<extra></extra>"), **where)


def _level_lines(fig: go.Figure, idx: pd.DatetimeIndex, candles: pd.DataFrame,
                 levels: pd.DataFrame | None, previous: pd.DataFrame | None, **where) -> list:
    """Draw each session's major levels (black dashes) and previous high and low (gray dots)
    from its first to its last candle; return the prices worth keeping on the price axis (the
    ones within half the candles' span of them)."""
    day = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    walls = pd.Series(_wall(idx), index=day)
    starts, ends = walls.groupby(level=0).first(), walls.groupby(level=0).last()
    lo, hi = float(candles["low"].min()), float(candles["high"].max())
    near = []
    for table, name, line in ((levels, "major level", {"color": LEVEL, "width": 1.5,
                                                       "dash": "dash"}),
                              (previous, "previous session's high and low",
                               {"color": _MUTED, "width": 1, "dash": "dot"})):
        t = table[table["day"].isin(starts.index)] if table is not None and len(table) else None
        if t is None or not len(t):
            continue
        xs, ys = _segments(starts[t["day"]].tolist(), ends[t["day"]].tolist(), t["price"])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, line=line,
                                 hovertemplate=f"{name} %{{y:.2f}}<extra></extra>"), **where)
        near += [v for v in t["price"] if lo - (hi - lo) / 2 <= v <= hi + (hi - lo) / 2]
    return near


def switchboard_figure(name: str, candles: pd.DataFrame, bands: pd.DataFrame,
                       decisions: pd.DataFrame, sim: dict, rules: dict, switches: dict, *,
                       average: pd.Series | None = None, levels: pd.DataFrame | None = None,
                       previous: pd.DataFrame | None = None, title: str = "") -> go.Figure:
    """What every switch says, candle by candle, and what the strategy did.

    Top: the candles with the Bollinger Bands, the minute chart's moving average (amber), the
    major levels (black dashes) and the previous session's high and low (gray dots) of each
    session, the trailing stop while a trade is open (black dotted steps) and the entries and
    exits. Bottom: one colored row per rule, then the decision and whether a trade is open:

    * a trigger row: blue = proposes long, orange = proposes short, pale = nothing;
    * a filter row: light gray = both sides allowed, blue = longs only, orange = shorts only, dark
      gray = neither side;
    * the decision: blue / orange = a long / short signal (the trade fills at the next candle's
      open), dark gray = a proposal a filter blocked, or two triggers that disagreed;
    * in a trade: blue = long, orange = short.

    A rule that is switched off keeps its row (marked "off"): it shows what it would have said,
    but it takes no part in the decision.

    Args:
        name: The stock.
        candles: Its regular-session candles of the sessions shown.
        bands: ``indicators.bollinger`` of those candles.
        decisions: ``combine.decide`` of those candles.
        sim: ``simulate.run`` of the whole stock (its stop, position and trades are cut here).
        rules: ``combine.load_rules``.
        switches: Rule name -> on / off.
        average: The minute chart's moving average (drawn when given).
        levels: ``indicators.major_levels`` (drawn when given).
        previous: ``indicators.previous_range`` (drawn when given).
        title: Figure title.
    """
    idx = candles.index
    x = _wall(idx)
    rows = []
    for rname, rule in rules.items():
        off = "" if switches.get(rname) else ", off"
        if rule.KIND == "trigger":
            s = decisions[rname].to_numpy()
            rows.append((f"{rname} (trigger{off})", np.select([s == 1, s == -1], [1, 2], 0),
                         np.select([s == 1, s == -1], ["proposes long", "proposes short"],
                                   "nothing")))
        else:
            lo, sh = (decisions[f"{rname}: long"].to_numpy(bool),
                      decisions[f"{rname}: short"].to_numpy(bool))
            rows.append((f"{rname} (filter{off})", np.select([lo & sh, lo, sh], [3, 1, 2], 4),
                         np.select([lo & sh, lo, sh],
                                   ["both sides allowed", "longs only", "shorts only"],
                                   "neither side")))
    sig = decisions["signal"].to_numpy()
    blocked, conflict = decisions["blocked"].to_numpy(bool), decisions["conflict"].to_numpy(bool)
    rows.append(("decision", np.select([sig == 1, sig == -1, blocked | conflict], [1, 2, 4], 0),
                 np.select([sig == 1, sig == -1, blocked, conflict],
                           ["LONG signal: buy at the next open", "SHORT signal: sell short at the "
                            "next open", "a proposal a filter blocked",
                            "the triggers disagree: no trade"], "nothing")))
    pos = sim["position"].reindex(idx).fillna(0).to_numpy()
    rows.append(("in a trade", np.select([pos == 1, pos == -1], [1, 2], 0),
                 np.select([pos == 1, pos == -1], ["long", "short"], "flat")))
    strips = 28 * len(rows)                    # every strip keeps its height as rules are added
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[440, strips])
    fig.add_trace(go.Candlestick(
        x=x, open=candles["open"], high=candles["high"], low=candles["low"],
        close=candles["close"], name="candles",
        increasing={"line": {"color": UP, "width": 1}, "fillcolor": UP},
        decreasing={"line": {"color": DOWN, "width": 1}, "fillcolor": DOWN}), row=1, col=1)
    b = bands.reindex(idx)
    fig.add_trace(go.Scatter(x=x, y=b["upper"], mode="lines", name="Bollinger Bands",
                             legendgroup="bands", line={"color": BAND, "width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=b["lower"], mode="lines", showlegend=False,
                             legendgroup="bands", line={"color": BAND, "width": 1},
                             fill="tonexty", fillcolor=BAND_FILL), row=1, col=1)
    if average is not None:
        fig.add_trace(go.Scatter(x=x, y=average.reindex(idx), mode="lines",
                                 name="moving average", line={"color": AVERAGES[0], "width": 1.4}),
                      row=1, col=1)
    near = _level_lines(fig, idx, candles, levels, previous, row=1, col=1)
    tr = sim["trades"]
    in_view = tr[(tr["entry time"] <= idx[-1]) & (tr["exit time"] >= idx[0])] if len(idx) else tr
    sx, sy = _stop_paths(sim["stop"], in_view, idx)
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="trailing stop",
                             line={"color": _INK, "width": 1.5, "dash": "dot", "shape": "hv"},
                             hovertemplate="stop %{y:.2f}<extra></extra>"), row=1, col=1)
    _trade_marks(fig, in_view, within=idx, row=1, col=1)
    fig.add_trace(go.Heatmap(
        z=np.vstack([r[1] for r in rows]), x=x, y=[r[0] for r in rows],
        colorscale=_discrete([c for c, _ in STRIP.values()]), zmin=-0.5, zmax=len(STRIP) - 0.5,
        showscale=False, ygap=2, customdata=np.vstack([r[2] for r in rows]),
        hovertemplate="%{y} · %{x}: %{customdata}<extra></extra>"), row=2, col=1)
    for code in (1, 2, 3, 4):
        _key(fig, STRIP[code][1], STRIP[code][0], row=2, col=1)
    fig.update_xaxes(type="date", rangebreaks=_breaks(idx), rangeslider_visible=False,
                     range=_span(idx))
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="price", range=_price_range(candles, [*sy, *near]), row=1, col=1)
    return _finish(fig, title=title or f"{name}: what every switch says", height=640 + strips)


def results_figure(summary: pd.DataFrame, *, title: str = "") -> go.Figure:
    """The average result per trade (percent, after costs) by market condition, long and short
    side by side, one panel per stock; the bar's label is the number of trades.

    Args:
        summary: ``results.summary(trades, ["stock", "condition", "side"])``.
        title: Figure title.
    """
    stocks = list(dict.fromkeys(summary["stock"])) if len(summary) else []
    fig = make_subplots(rows=1, cols=max(len(stocks), 1), shared_yaxes=True,
                        subplot_titles=stocks or [""], horizontal_spacing=0.05)
    for col, stock in enumerate(stocks, start=1):
        part = summary[summary["stock"] == stock]
        for side, color in (("long", _LONG), ("short", _SHORT)):
            p = part[part["side"] == side].set_index("condition").reindex(
                ["up", "sideways", "down"])
            fig.add_trace(go.Bar(
                x=p.index, y=p["average return %"], name=side, legendgroup=side,
                showlegend=col == 1, marker={"color": color, "line": {"width": 0}},
                text=p["trades"].fillna(0).astype(int).astype(str) + " trades",
                textposition="outside",
                customdata=np.column_stack([p["trades"].fillna(0), p["winners"],
                                            p["total P&L $"]]),
                hovertemplate=(f"{side}, market %{{x}}: %{{y:+.3f}}% per trade<br>"
                               "%{customdata[0]:.0f} trades, %{customdata[1]:.0%} winners, "
                               "total %{customdata[2]:+,.0f} $<extra></extra>")), row=1, col=col)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="average result per trade, %", row=1, col=1)
    fig.update_yaxes(zeroline=True, zerolinecolor=_INK)
    return _finish(fig, title=title or "Results by market condition and side", height=460)


def equity_figure(curve: pd.DataFrame, *, title: str = "") -> go.Figure:
    """The profit curve: the cumulative P&L in dollars after every exit, per stock and for all
    together (black), and below it how far the total is under its best level so far (the
    drawdown).

    Args:
        curve: ``results.equity(trades)``.
        title: Figure title.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.04)
    if len(curve):
        x = _wall(curve.index)
        stocks = [c for c in curve.columns if c != "all stocks"]
        for stock, color in zip(stocks, SERIES, strict=False):
            fig.add_trace(go.Scatter(x=x, y=curve[stock], mode="lines", name=stock,
                                     line={"color": color, "width": 1.4, "shape": "hv"}),
                          row=1, col=1)
        total = curve["all stocks"]
        fig.add_trace(go.Scatter(x=x, y=total, mode="lines", name="all stocks",
                                 line={"color": _INK, "width": 2, "shape": "hv"}), row=1, col=1)
        dd = total - total.cummax().clip(lower=0.0)
        fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", name="drawdown (all stocks)",
                                 line={"color": DOWN, "width": 1, "shape": "hv"},
                                 fill="tozeroy", fillcolor="rgba(208, 59, 59, 0.15)"),
                      row=2, col=1)
    fig.update_xaxes(type="date")
    fig.update_yaxes(title_text="P&L $", row=1, col=1)
    fig.update_yaxes(title_text="below best, $", row=2, col=1)
    return _finish(fig, title=title or "The profit curve", height=520)


def trade_figure(trade: pd.Series, candles: pd.DataFrame, bands: pd.DataFrame, stop: pd.Series,
                 *, before: int = 60, after: int = 30, title: str = "") -> go.Figure:
    """One trade from its signal to its exit: the candles (from ``before`` candles before the
    signal to ``after`` candles after the exit), the Bollinger Bands, the trailing stop as it
    moved, the signal (dotted line), the entry and the exit.

    Args:
        trade: A row of ``simulate.run(...)["trades"]``.
        candles: The stock's regular-session candles.
        bands: Their ``indicators.bollinger``.
        stop: ``simulate.run(...)["stop"]``.
        before: Candles shown before the signal.
        after: Candles shown after the exit.
        title: Figure title.
    """
    i0 = max(candles.index.get_loc(trade["signal time"]) - before, 0)
    i1 = min(candles.index.get_loc(trade["exit time"]) + after, len(candles) - 1)
    c = candles.iloc[i0:i1 + 1]
    x = _wall(c.index)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=c["open"], high=c["high"], low=c["low"], close=c["close"], name="candles",
        increasing={"line": {"color": UP, "width": 1}, "fillcolor": UP},
        decreasing={"line": {"color": DOWN, "width": 1}, "fillcolor": DOWN}))
    b = bands.reindex(c.index)
    fig.add_trace(go.Scatter(x=x, y=b["upper"], mode="lines", name="Bollinger Bands",
                             legendgroup="bands", line={"color": BAND, "width": 1}))
    fig.add_trace(go.Scatter(x=x, y=b["lower"], mode="lines", showlegend=False,
                             legendgroup="bands", line={"color": BAND, "width": 1},
                             fill="tonexty", fillcolor=BAND_FILL))
    fig.add_trace(go.Scatter(x=x, y=b["middle"], mode="lines", name="band middle",
                             line={"color": BAND, "width": 1, "dash": "dash"}))
    one = trade.to_frame().T
    sx, sy = _stop_paths(stop, one, c.index)          # this trade's stop only
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", name="trailing stop",
                             line={"color": _INK, "width": 1.5, "dash": "dot", "shape": "hv"},
                             hovertemplate="stop %{y:.2f}<extra></extra>"))
    at = _wall([trade["signal time"]])[0]
    fig.add_trace(go.Scatter(x=[at, at], y=[c["low"].min(), c["high"].max()], mode="lines",
                             name="the signal (a candle's close)", hoverinfo="skip",
                             line={"color": _MUTED, "width": 1, "dash": "dot"}))
    target = float(trade.get("target", np.nan))
    if np.isfinite(target):
        span = _wall([trade["entry time"], trade["exit time"]])
        fig.add_trace(go.Scatter(x=span, y=[target, target], mode="lines", name="profit target",
                                 line={"color": _LONG if trade["side"] == "long" else _SHORT,
                                       "width": 1.5, "dash": "dash"},
                                 hovertemplate="profit target %{y:.2f}<extra></extra>"))
        sy = [*sy, target]
    _trade_marks(fig, one.astype({"entry price": float, "exit price": float, "return %": float}))
    fig.update_xaxes(type="date", rangebreaks=_breaks(c.index), rangeslider_visible=False,
                     range=_span(c.index))
    fig.update_yaxes(title_text="price", range=_price_range(c, sy))
    return _finish(fig, title=title or f"{trade['stock']} {trade['side']}", height=560)


def switch_effect_figure(effects: pd.DataFrame, *, title: str = "") -> go.Figure:
    """What each switch changes: one panel per stock (and all together), one line per rule from
    its total P&L with the switch off (hollow circle) to the total with it on (filled: blue when
    switching it on made more, red when less). Every other switch stays as set; the label says
    how the rule is set now.

    Args:
        effects: ``results.switch_effects``.
        title: Figure title.
    """
    stocks = list(dict.fromkeys(effects["stock"]))
    fig = make_subplots(rows=1, cols=max(len(stocks), 1), shared_yaxes=True,
                        subplot_titles=stocks or [""], horizontal_spacing=0.04)
    for col, stock in enumerate(stocks, start=1):
        e = effects[effects["stock"] == stock]
        label = (e["rule"] + " (" + e["as set"] + " now)").tolist()
        sx, sy = _segments(e["P&L off $"], e["P&L on $"], label)
        fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line={"color": _MUTED, "width": 2},
                                 hoverinfo="skip", showlegend=False), row=1, col=col)
        info = np.column_stack([e["trades off"], e["trades on"], e["change $"],
                                e["profit ÷ drawdown off"], e["profit ÷ drawdown on"]])
        fig.add_trace(go.Scatter(
            x=e["P&L off $"], y=label, mode="markers", name="switched off", legendgroup="off",
            showlegend=col == 1, customdata=info,
            marker={"symbol": "circle-open", "size": 11, "color": _INK,
                    "line": {"color": _INK, "width": 2}},
            hovertemplate="%{y}, switched OFF: %{x:+,.0f} $ from %{customdata[0]:.0f} trades, "
                          "profit ÷ worst drawdown %{customdata[3]:.2f}<extra></extra>"),
            row=1, col=col)
        fig.add_trace(go.Scatter(
            x=e["P&L on $"], y=label, mode="markers", name="switched on", showlegend=False,
            customdata=info,
            marker={"size": 12, "color": np.where(e["change $"] >= 0, UP, DOWN).tolist(),
                    "line": {"color": "white", "width": 1}},
            hovertemplate="%{y}, switched ON: %{x:+,.0f} $ from %{customdata[1]:.0f} trades "
                          "(%{customdata[2]:+,.0f} $), profit ÷ worst drawdown "
                          "%{customdata[4]:.2f}<extra></extra>"),
            row=1, col=col)
        fig.add_vline(x=0, line={"color": _INK, "width": 1}, row=1, col=col)
    _key(fig, "switched on: more profit", UP, "circle", row=1, col=1)
    _key(fig, "switched on: less profit", DOWN, "circle", row=1, col=1)
    fig.update_xaxes(title_text="total P&L after costs, $", tickformat=",.0f")
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title or "What each switch changes", height=200 + 46 * max(
        effects["rule"].nunique(), 1))


def effect_breakdown_figure(breakdown: pd.DataFrame, *, title: str = "") -> go.Figure:
    """Where each switch's change comes from: one panel per stock, one row per rule, one column
    per market condition (up, sideways, down) and per half-year; each cell is the change in P&L
    from switching the rule on (blue: more, red: less, the number printed in it). Empty: no trade
    there either way.

    Args:
        breakdown: ``results.effect_breakdown``.
        title: Figure title.
    """
    stocks = list(dict.fromkeys(breakdown["stock"]))
    rules = list(dict.fromkeys(breakdown["rule"]))
    groups = [g for g in ("up", "sideways", "down") if g in set(breakdown["group"])]
    groups += sorted(g for g in set(breakdown["group"]) if g not in groups)
    top = float(breakdown["change $"].abs().max()) if len(breakdown) else 1.0
    fig = make_subplots(rows=1, cols=max(len(stocks), 1), shared_yaxes=True,
                        subplot_titles=stocks or [""], horizontal_spacing=0.04)
    for col, stock in enumerate(stocks, start=1):
        z = breakdown[breakdown["stock"] == stock].pivot_table(
            index="rule", columns="group", values="change $", aggfunc="sum").reindex(
            index=rules, columns=groups)
        text = z.map(lambda v: "" if pd.isna(v) else f"{v:+,.0f}")
        fig.add_trace(go.Heatmap(
            z=z.to_numpy(), x=[f"market {g}" if g in WORDS.values() else g for g in groups],
            y=rules, zmin=-top, zmax=top, zmid=0,
            colorscale=[[0.0, DOWN], [0.5, "#f4f3ef"], [1.0, UP]], showscale=col == 1,
            colorbar={"title": {"text": "$"}, "len": 0.8}, text=text.to_numpy(),
            texttemplate="%{text}", xgap=2, ygap=2,
            hovertemplate="%{y} switched on, %{x}: %{z:+,.0f} $<extra></extra>"), row=1, col=col)
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title or "Where each switch's change comes from",
                   height=200 + 46 * max(len(rules), 1))


def walk_curves_figure(walks: dict, *, control: str = "no tuning", title: str = "") -> go.Figure:
    """The walk forward's profit curves: one panel per stock, one line per walk (the control in
    black): the P&L in dollars added up trade after trade over the traded sessions, each
    session traded with settings chosen on earlier sessions only.

    Args:
        walks: (stock, walk name) -> ``tuning.walk``.
        control: The walk drawn in black.
        title: Figure title.
    """
    stocks = list(dict.fromkeys(s for s, _ in walks))
    names = list(dict.fromkeys(n for _, n in walks))
    colors = dict(zip([n for n in names if n != control], SERIES, strict=False))
    fig = make_subplots(rows=1, cols=max(len(stocks), 1), subplot_titles=stocks or [""],
                        horizontal_spacing=0.05)
    for col, stock in enumerate(stocks, start=1):
        for name in names:
            t = walks[(stock, name)]["trades"].sort_values("exit time")
            fig.add_trace(go.Scatter(
                x=_wall(t["exit time"]), y=t["P&L $"].cumsum(), mode="lines", name=name,
                legendgroup=name, showlegend=col == 1,
                line={"color": _INK if name == control else colors[name],
                      "width": 2.2 if name == control else 1.3, "shape": "hv"},
                hovertemplate=f"{name}: %{{y:+,.0f}} $ (%{{x}})<extra></extra>"), row=1, col=col)
        fig.add_hline(y=0, line={"color": _MUTED, "width": 1}, row=1, col=col)
    fig.update_xaxes(type="date")
    fig.update_yaxes(title_text="P&L after costs, $", tickformat=",.0f", row=1, col=1)
    return _finish(fig, title=title or "The walk forward: profit over the traded sessions",
                   height=480)


def settings_figure(walks: dict, stock: str, *, title: str = "") -> go.Figure:
    """The settings each walk chose for one stock, tuning after tuning: one panel per tuned
    hyperparameter that changed (the first: the winner's training score), one step line per
    walk. Values
    that jump between far-apart settings from one tuning to the next mean the tuning follows
    noise.

    Args:
        walks: (stock, walk name) -> ``tuning.walk``.
        stock: The stock shown.
        title: Figure title.
    """
    tuned = [(n, w["settings"]) for (s, n), w in walks.items() if s == stock
             and w["settings"]["score"].notna().any()]
    skip = {"session", "day", "score"}
    cols = [c for c in (tuned[0][1].columns if tuned else []) if c not in skip]
    names = ["score", *[c for c in cols if any(lv[c].astype(str).nunique() > 1
                                                      for _, lv in tuned if c in lv)]]
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=["the winner's score on the training sessions"
                                        if n == "score" else n for n in names])
    for k, (walk_name, lv) in enumerate(tuned):
        for row, name in enumerate(names, start=1):
            if name not in lv:
                continue
            y = lv[name].where(lv[name] > -1e8) if name == "score" else lv[name]
            y = y.astype(float) if y.dtype == bool else y
            fig.add_trace(go.Scatter(
                x=lv["day"], y=y, mode="lines", name=walk_name, legendgroup=walk_name,
                showlegend=row == 1, line={"color": SERIES[k % len(SERIES)], "width": 1.3,
                                           "shape": "hv"},
                hovertemplate=f"{walk_name}, {name}: %{{y}} (from %{{x|%Y-%m-%d}})"
                              "<extra></extra>"), row=row, col=1)
    fig.update_xaxes(type="date")
    return _finish(fig, title=title or f"{stock}: the settings each walk chose",
                   height=140 + 120 * len(names))


def trials_figure(result: dict, *, title: str = "") -> go.Figure:
    """An optimization run (``tuning.optimize``). Left, trial after trial: every trial's training
    score, the best so far, the luck line (the best score that as many trials without any edge
    reach by chance) and the validation score of the best so far: flat or falling while the best
    climbs means the search fits noise. Right: every scored trial's training against its
    validation score. A trial with too few trades has no score (a hollow mark on the floor).

    Args:
        result: ``tuning.optimize``.
        title: Figure title.
    """
    log = result["trials"].sort_values("trial").reset_index(drop=True)
    x, scored = log["trial"], log["score"].astype(float)
    ok = scored.notna()
    val = (log["validation score"].astype(float) if "validation score" in log
           else pd.Series(np.nan, index=log.index))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
                        subplot_titles=["trial after trial",
                                        "training against validation, every scored trial"])
    fig.add_trace(go.Scatter(
        x=x[ok], y=scored[ok], mode="markers", name="a trial: its training score",
        marker={"color": SERIES[0], "size": 5, "opacity": 0.45},
        customdata=log.loc[ok, ["P&L $", "trades"]].to_numpy(),
        hovertemplate="trial %{x}: score %{y:.2f} (%{customdata[0]:,.0f} $, %{customdata[1]} "
                      "trades)<extra></extra>"), row=1, col=1)
    if (~ok).any():
        fig.add_trace(go.Scatter(
            x=x[~ok], y=[float(scored.min()) if ok.any() else 0.0] * int((~ok).sum()),
            mode="markers", name="too few trades: no score",
            marker={"symbol": "circle-open", "size": 5, "color": _MUTED,
                    "line": {"color": _MUTED}}, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=scored.cummax(), mode="lines", name="best so far",
                             line={"color": _INK, "width": 1.5, "shape": "hv"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=log["luck"], mode="lines",
                             name="luck line: the best that as many trials reach without an edge",
                             line={"color": _SHORT, "width": 1.5, "dash": "dash"}), row=1, col=1)
    if val[ok].notna().any():
        pick, top, curve = -1, -np.inf, []
        for i, v in enumerate(scored):
            if v > top:
                pick, top = i, v
            curve.append(val.iat[pick] if pick >= 0 else np.nan)
        fig.add_trace(go.Scatter(x=x, y=curve, mode="lines",
                                 name="validation score of the best so far",
                                 line={"color": SERIES[2], "width": 1.5, "shape": "hv"}),
                      row=1, col=1)
        rho = scored[ok].rank().corr(val[ok].rank())
        fig.add_trace(go.Scatter(
            x=scored[ok], y=val[ok], mode="markers",
            name=f"a trial: training and validation score (rank correlation {rho:+.2f})",
            marker={"color": SERIES[0], "size": 5, "opacity": 0.45},
            customdata=log.loc[ok, ["trial", "validation P&L $", "validation trades"]].to_numpy(),
            hovertemplate="trial %{customdata[0]}: training %{x:.2f}, validation %{y:.2f} "
                          "(%{customdata[1]:,.0f} $, %{customdata[2]} trades)<extra></extra>"),
            row=1, col=2)
        fig.add_hline(y=0, line={"color": _GRID, "width": 1}, row=1, col=2)
    fig.update_xaxes(title_text="trial", row=1, col=1)
    fig.update_xaxes(title_text="training score", row=1, col=2)
    fig.update_yaxes(title_text="score: daily Sharpe ratio, per year", row=1, col=1)
    fig.update_yaxes(title_text="validation score", row=1, col=2)
    return _finish(fig, title=title or "The optimization, trial by trial", height=480)


def effects_figure(effects: pd.DataFrame, summary: pd.DataFrame, *, top: int = 12,
                   title: str = "") -> go.Figure:
    """What each hyperparameter does to the score: one small panel per hyperparameter (the
    ``top`` strongest), its values across, the score up, the value as set marked black.

    Args:
        effects: ``tuning.effects``.
        summary: ``tuning.effect_summary`` (its order: strongest first).
        top: How many panels.
        title: Figure title.
    """
    names = list(summary["hyperparameter"])[:top]
    cols = 4
    nrows = max(-(-len(names) // cols), 1)
    fig = make_subplots(rows=nrows, cols=cols, subplot_titles=names or [""],
                        horizontal_spacing=0.06, vertical_spacing=0.12)
    for k, name in enumerate(names):
        g = effects[effects["hyperparameter"] == name]
        y = g["score"].where(g["score"] > -1e8)
        x = g["value"].astype(str)
        r, c = k // cols + 1, k % cols + 1
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", showlegend=k == 0,
                                 name="score at that value", line={"color": _LONG, "width": 1.5},
                                 marker={"size": 7}, customdata=g[["trades", "P&L $"]].to_numpy(),
                                 hovertemplate=f"{name} = %{{x}}: score %{{y:.2f}}, "
                                               "%{customdata[0]} trades, "
                                               "%{customdata[1]:+,.0f} $<extra></extra>"),
                      row=r, col=c)
        mark = g["as set"].to_numpy(bool)
        fig.add_trace(go.Scatter(x=x[mark], y=y[mark], mode="markers", showlegend=k == 0,
                                 name="the value as set", marker={"size": 11, "color": _INK},
                                 hoverinfo="skip"), row=r, col=c)
        few = y.isna().to_numpy()
        if few.any():                       # too few trades: no score, shown at the bottom
            floor = float(y.min()) if y.notna().any() else 0.0
            fig.add_trace(go.Scatter(x=x[few], y=[floor] * int(few.sum()), mode="markers",
                                     name="trades too little: no score",
                                     showlegend="few" not in {t.legendgroup for t in fig.data},
                                     legendgroup="few", marker={"symbol": "x-thin", "size": 10,
                                                                "color": DOWN,
                                                                "line": {"color": DOWN,
                                                                         "width": 2}},
                                     hoverinfo="skip"), row=r, col=c)
        fig.update_xaxes(type="category", row=r, col=c)
    return _finish(fig, title=title or "What each hyperparameter does to the score",
                   height=120 + 230 * nrows)



# ---- stock groups (groups.py) ------------------------------------------------------------------

_GROUP_MARKS = ("circle", "square", "diamond", "triangle-up")


def _group_style(group: int) -> tuple[str, str]:
    """A group's color and marker: the palette's eight colors, then again with another marker."""
    i = int(group) - 1
    return SERIES[i % len(SERIES)], _GROUP_MARKS[(i // len(SERIES)) % len(_GROUP_MARKS)]


def _value(value, spec: str) -> str:
    """A number as text in its format, a dash when it is missing."""
    return format(float(value), spec) if value is not None and np.isfinite(value) else "-"


def history_figure(sessions: pd.Series, *, least: int, title: str = "") -> go.Figure:
    """How many sessions of data each stock of the pool has (bars of 10
    sessions): stocks with fewer than ``least`` (the dashed line) are left out of the grouping,
    their numbers would rest on too few sessions.

    Args:
        sessions: Sessions per stock.
        least: The least sessions a stock needs to be grouped.
        title: Figure title.
    """
    edges = np.arange(0, int(sessions.max()) + 11, 10)
    counts, _ = np.histogram(sessions, bins=edges)
    lo, hi = edges[:-1], edges[1:] - 1
    fig = go.Figure()
    for name, sel, color in (("stocks grouped", lo >= least, _LONG),
                             ("left out: fewer sessions than the least", lo < least, _GRID)):
        sel = sel & (counts > 0)
        fig.add_trace(go.Bar(x=lo[sel] + 5, y=counts[sel], width=9, name=name,
                             marker={"color": color, "line": {"width": 0}},
                             customdata=np.column_stack([lo[sel], hi[sel]]),
                             hovertemplate="%{customdata[0]}-%{customdata[1]} sessions: %{y} "
                                           "stocks<extra></extra>"))
    fig.add_trace(go.Scatter(x=[least, least], y=[0, int(counts.max())], mode="lines",
                             name=f"the least sessions a stock needs: {least}",
                             line={"color": _INK, "width": 1, "dash": "dash"}, hoverinfo="skip"))
    fig.update_xaxes(title_text="sessions of data")
    fig.update_yaxes(title_text="stocks")
    return _finish(fig, title=title or "How much history each stock has", height=360)


def feature_figure(stock: str, feature: str, moving: pd.Series, table: pd.DataFrame,
                   halves: tuple[pd.DataFrame, pd.DataFrame], info: dict, *,
                   sessions: int = 20, title: str = "") -> go.Figure:
    """One character number of one stock, and where it sits among the other stocks.

    Left, over the stock's sessions: the number over the last ``sessions`` sessions, session by
    session (how steady it is); the stock's value on each half of its sessions (the persistence
    check compares these, stock by stock); and on the whole window (the value the grouping
    reads). Right: the same number for every stock of the pool, this stock's value marked.

    Args:
        stock: The stock.
        feature: The number's name (a key of ``groups.FEATURES``).
        moving: ``groups.moving(...)[feature]`` of the stock.
        table: ``groups.characters`` of every stock (whole window).
        halves: The same on the first and on the second half of every stock's sessions.
        info: ``groups.FEATURES[feature]``.
        sessions: The moving window's sessions.
        title: Figure title.
    """
    spec, label = info["format"], info["label"]
    whole = float(table.loc[stock, feature])
    rank = float((table[feature] < whole).mean())
    fig = make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], horizontal_spacing=0.08,
                        subplot_titles=[f"{stock}, session by session",
                                        "every stock of the pool"])
    m = moving.dropna()
    idx = moving.index
    k = len(idx) // 2
    fig.add_trace(go.Scatter(
        x=m.index, y=m, mode="lines", name=f"over the last {sessions} sessions",
        line={"color": ROLES["stock"], "width": 1.5},
        hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:{spec}}} over the {sessions} sessions up to this "
                      "one<extra></extra>"), row=1, col=1)
    for part, (a, b), t in ((0, (0, k - 1), "first half"), (1, (k, len(idx) - 1), "second half")):
        v = float(halves[part].loc[stock, feature])
        fig.add_trace(go.Scatter(
            x=[idx[a], idx[b]], y=[v, v], mode="lines", name="on each half of its sessions",
            showlegend=part == 0, legendgroup="halves", line={"color": SERIES[2], "width": 3},
            hovertemplate=f"{t}: {_value(v, spec)}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[idx[0], idx[-1]], y=[whole, whole], mode="lines",
        name="on the whole window (what the grouping reads)",
        line={"color": _INK, "width": 1.5, "dash": "dash"},
        hovertemplate=f"whole window: {_value(whole, spec)}<extra></extra>"), row=1, col=1)
    values = table[feature].dropna()
    edges = np.histogram_bin_edges(values, bins=40)
    counts, _ = np.histogram(values, bins=edges)
    fig.add_trace(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges) * 0.9,
        name="stocks with that value", marker={"color": BAND, "line": {"width": 0}},
        customdata=np.column_stack([edges[:-1], edges[1:]]),
        hovertemplate=f"%{{customdata[0]:{spec}}} to %{{customdata[1]:{spec}}}: %{{y}} stocks"
                      "<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[whole, whole], y=[0, int(counts.max())], mode="lines",
        name=f"{stock}: {_value(whole, spec)}, above {rank:.0%} of the stocks",
        line={"color": ROLES["stock"], "width": 2}, hoverinfo="skip"), row=1, col=2)
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_xaxes(title_text="value", row=1, col=2)
    fig.update_yaxes(title_text="stocks", row=1, col=2)
    fig = _finish(fig, title=title or f"{stock}: {label}", height=460)
    return fig.update_layout(margin={"t": 90})


def band_events_figure(stock: str, minutes: pd.DataFrame, bands: pd.DataFrame,
                       events: pd.DataFrame, *, follow: int, title: str = "") -> go.Figure:
    """How the two band numbers are measured, on one session's 1-minute closes: every close
    that lands outside the Bollinger Band (the close before was not), marked green when the very
    next close is back inside (the candle the fade fires on) and amber when the next close is
    outside too (the breakout's second close); a dotted line from each to the close ``follow``
    minutes later, how far the price went on (+) or came back (-), in typical moves.

    Args:
        stock: The stock.
        minutes: Its 1-minute candles of the session.
        bands: ``pipeline.bands`` on those candles (computed on a longer history).
        events: ``groups.band_events`` of that session.
        follow: Minutes to the later close.
        title: Figure title.
    """
    x = np.asarray(minutes.index.strftime("%H:%M"))          # one session: the clock is enough
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=bands["lower"], mode="lines", name="Bollinger Band",
                             legendgroup="band", line={"color": BAND, "width": 1},
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=bands["upper"], mode="lines", showlegend=False,
                             legendgroup="band", line={"color": BAND, "width": 1},
                             fill="tonexty", fillcolor=BAND_FILL, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=minutes["close"], mode="lines", name="1-minute close",
                             line={"color": ROLES["stock"], "width": 1.2},
                             hovertemplate="%{x}: close %{y:.2f}<extra></extra>"))
    at = minutes.index.get_indexer(pd.DatetimeIndex(events["time"]))
    ok = at >= 0
    ev, at = events[ok], at[ok]
    xs, ys = [], []
    for a, c0, c1 in zip(at, ev["close"], ev["later"], strict=True):
        xs += [x[a], x[a + follow], None]
        ys += [c0, c1, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"to the close {follow} minutes later",
                             line={"color": _INK, "width": 1, "dash": "dot"}, hoverinfo="skip"))
    side = np.where(ev["side"] > 0, "above the upper band", "below the lower band")
    later = [x[a + follow] for a in at]
    words = np.where(ev["follow"] > 0, "went on", "came back")
    for back, name, color in ((True, "close outside, the next close back inside (the fade)",
                               SERIES[3]),
                              (False, "close outside, the next close outside too (the breakout)",
                               SERIES[2])):
        sel = (ev["back"] == back).to_numpy()
        fig.add_trace(go.Scatter(
            x=x[at[sel]], y=ev["close"][sel], mode="markers", name=name,
            marker={"color": color, "size": 9, "line": {"color": _INK, "width": 0.5}},
            customdata=np.column_stack([side[sel], np.asarray(later)[sel], ev["later"][sel],
                                        ev["follow"][sel], words[sel]]),
            hovertemplate="%{x}: close %{y:.2f}, %{customdata[0]}<br>at %{customdata[1]}: "
                          "%{customdata[2]:.2f}, it %{customdata[4]} %{customdata[3]:+.2f} "
                          "typical moves<extra></extra>"))
    fig.update_xaxes(type="category", nticks=14)
    fig.update_yaxes(title_text="price")
    return _finish(fig, title=title or f"{stock}: closes outside the band and what followed",
                   height=480)


def persistence_figure(persist: pd.Series, features: dict, *, title: str = "") -> go.Figure:
    """Each number's rank correlation between the two halves of every stock's sessions: near 1
    the number is the stock's own (a stock high on it in the first half is high in the second),
    near 0 it is mostly chance or a passing phase, and cannot define a lasting group."""
    p = persist.sort_values()
    fig = go.Figure(go.Bar(
        x=p, y=list(p.index), orientation="h", marker={"color": _LONG},
        name="rank correlation across the stocks: first half against second half",
        customdata=[features[f]["label"] for f in p.index],
        hovertemplate="%{customdata}: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="rank correlation, first half against second half",
                     range=[min(-0.1, float(p.min()) - 0.05), 1])
    return _finish(fig, title=title or "How much of each number is the stock's own",
                   height=140 + 32 * len(p))


def correlation_figure(features: pd.DataFrame, info: dict, *, title: str = "") -> go.Figure:
    """The numbers' rank correlations with each other across the stocks: two numbers near +1 or
    -1 measure much the same thing and together weigh twice in the grouping."""
    rho = features.corr(method="spearman")
    names = list(rho.columns)
    labels = [info[n]["label"] for n in names]
    hover = np.array([[f"{labels[i]}<br>{labels[j]}" for j in range(len(names))]
                      for i in range(len(names))])
    fig = go.Figure(go.Heatmap(
        z=rho.to_numpy(), x=names, y=names, zmin=-1, zmax=1, colorscale=DIVERGING,
        text=np.round(rho.to_numpy(), 2), texttemplate="%{text:.2f}", customdata=hover,
        colorbar={"title": {"text": "rank<br>correlation"}},
        hovertemplate="%{customdata}<br>rank correlation %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title or "Which numbers measure the same thing", height=560)


def quality_figure(quality: pd.DataFrame, k: int, *, title: str = "") -> go.Figure:
    """How good each number of groups is (``groups.quality``): the share of the stocks'
    differences that lies between the groups (rises with every group), how well they separate
    (the silhouette), whether each half of the window alone forms the same groups (the adjusted
    Rand index), and the share of stocks that each half alone puts in their own group (purple;
    dotted: the same by chance). The dashed line marks the number chosen (``k``)."""
    q = quality
    fig = go.Figure()
    for col, name, color in (
            ("explained", "explained: share of the stocks' differences between the groups",
             SERIES[0]),
            ("separation", "separation: how much closer a stock is to its own group than to "
                           "the next (silhouette)", SERIES[2]),
            ("same on both halves", "same on both halves: the groups each half alone forms "
                                    "agree (adjusted Rand index)", SERIES[3]),
            ("stays in its group", "stays in its group: each half alone puts the stock "
                                   "closest to its own group", SERIES[4])):
        fig.add_trace(go.Scatter(
            x=q["groups"], y=q[col], mode="lines+markers", name=name,
            line={"color": color, "width": 2}, customdata=q["smallest group"],
            hovertemplate=f"%{{x}} groups: {col} %{{y:.2f}} (smallest group: %{{customdata}} "
                          "stocks)<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=q["groups"], y=q["stays by chance"], mode="lines", name="stays in its group by chance",
        line={"color": SERIES[4], "width": 1.5, "dash": "dot"},
        hovertemplate="%{x} groups: by chance %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[k, k], y=[0, 1], mode="lines", name=f"chosen: {k} groups",
                             line={"color": _INK, "width": 1, "dash": "dash"}, hoverinfo="skip"))
    fig.update_xaxes(title_text="number of groups", dtick=1)
    fig.update_yaxes(range=[0, 1])
    return _finish(fig, title=title or "How many groups?", height=420)


def group_map_figure(xy: pd.DataFrame, axes: list[str], members: pd.DataFrame,
                     groups: pd.DataFrame, *, highlight=(), title: str = "") -> go.Figure:
    """Every stock on a map of the two directions along which the stocks differ most
    (``groups.flat_map``), colored by its group: a big outlined mark is a stock its group tunes
    on (the most typical), a small one a stock the group checks the tuned setting on. Stocks
    close together on the map are alike on the numbers that pull along its axes (not on every
    number: the groups use all of them).

    Args:
        xy: ``groups.flat_map``'s coordinates.
        axes: Its axis names.
        members: ``groups.grouping(...)["members"]``.
        groups: ``groups.grouping(...)["groups"]``.
        highlight: Stocks named on the map.
        title: Figure title.
    """
    say = groups.set_index("group")["in words"]
    fig = go.Figure()
    for g, part in members.groupby("group"):
        color, mark = _group_style(g)
        p = xy.loc[part.index]
        for tune in (False, True):
            sel = (part["role"] == "tune").to_numpy() == tune
            fig.add_trace(go.Scatter(
                x=p["x"][sel], y=p["y"][sel], mode="markers", legendgroup=str(g),
                name=f"group {g} ({len(part)} stocks): {say[g]}", showlegend=not tune,
                marker={"color": color, "symbol": mark, "size": 12 if tune else 6,
                        "opacity": 1.0 if tune else 0.55,
                        "line": {"color": _INK, "width": 1.2 if tune else 0}},
                customdata=np.column_stack([part.index[sel], part["rank"][sel],
                                            part["role"][sel]]),
                hovertemplate=f"%{{customdata[0]}}: group {g}, the %{{customdata[1]}}. most "
                              "typical (%{customdata[2]})<extra></extra>"))
    _key(fig, "big outlined mark: a stock the group tunes on", _MUTED)
    _key(fig, "small mark: a stock the group checks on", _GRID)
    shown = [s for s in highlight if s in xy.index]
    if shown:
        fig.add_trace(go.Scatter(
            x=xy.loc[shown, "x"], y=xy.loc[shown, "y"], mode="markers+text", text=shown,
            textposition="top center", name="the strategy's stocks so far",
            marker={"symbol": "circle-open", "size": 18, "color": _INK,
                    "line": {"color": _INK, "width": 2}}, hoverinfo="skip"))
    fig.update_xaxes(title_text=axes[0], zeroline=False)
    fig.update_yaxes(title_text=axes[1], zeroline=False)
    return _finish(fig, title=title or "The stocks on a map, by group",
                   height=700 + 22 * len(groups))


def group_profile_figure(centre: pd.DataFrame, table: pd.DataFrame, members: pd.DataFrame,
                         info: dict, *, title: str = "") -> go.Figure:
    """Every group's character: one row per group, one column per number. The color is where
    the group's centre sits among all stocks (its mean score: blue above the middle stock, red
    below), the text the median value of its members, in the number's own unit."""
    names = list(centre.columns)
    sizes = members.groupby("group").size()
    median = table.loc[members.index, names].groupby(members["group"]).median()
    rows = [f"group {g} ({sizes[g]})" for g in centre.index]
    text = [[_value(median.loc[g, n], info[n]["format"]) for n in names] for g in centre.index]
    hover = [[f"{info[n]['label']}: median {text[i][j]}" for j, n in enumerate(names)]
             for i, _ in enumerate(centre.index)]
    fig = go.Figure(go.Heatmap(
        z=centre.to_numpy().clip(-1.5, 1.5), x=names, y=rows, zmin=-1.5, zmax=1.5,
        colorscale=DIVERGING, text=text, texttemplate="%{text}", customdata=hover,
        colorbar={"title": {"text": "score of the<br>group's centre"}},
        hovertemplate="%{y} · %{customdata}<br>its centre's score %{z:+.2f} (0: the middle "
                      "stock)<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title or "What each group is like",
                   height=200 + 34 * len(rows))


def stock_figure(stock: str, sc: pd.DataFrame, table: pd.DataFrame, members: pd.DataFrame,
                 centre: pd.DataFrame, info: dict, *, title: str = "") -> go.Figure:
    """One stock's character: its score on every number (bars: 0 the middle stock, +1 above
    about 84% of them) against its group's centre (black marks): where the bar and the mark
    agree the stock is typical of its group."""
    g = int(members.loc[stock, "group"])
    names = list(sc.columns)
    text = [f"{info[n]['label']}: {_value(table.loc[stock, n], info[n]['format'])}"
            for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sc.loc[stock], y=names, orientation="h", name=f"{stock}'s score",
                         marker={"color": ROLES["stock"]}, customdata=text,
                         hovertemplate="%{customdata}<br>score %{x:+.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=centre.loc[g, names], y=names, mode="markers",
                             name=f"the centre of group {g}",
                             marker={"color": _INK, "symbol": "diamond", "size": 10},
                             hovertemplate="group's centre: %{x:+.2f}<extra></extra>"))
    fig.update_xaxes(title_text="score: where the stock sits among the stocks (0 the middle)",
                     range=[-3.2, 3.2])
    fig.update_yaxes(autorange="reversed")
    return _finish(fig, title=title or f"{stock}: its character against its group's",
                   height=160 + 34 * len(names))


def group_candles_figure(days: dict[str, pd.DataFrame], members: pd.DataFrame, *,
                         shared: bool = True, title: str = "") -> go.Figure:
    """The daily candles of a group's stocks, one panel each, one above the other on the same
    dates, every stock rebased so that they compare: its first close of the window is 100, and the
    price axis is logarithmic, so the same percent move is the same height in every panel (a
    dotted line marks 100). Each panel is titled with the stock's most-typical rank in its
    group, its role (tune / check), its move over the window and its typical day's range.

    Args:
        days: ``{stock: daily candles}`` (``data.load_days``), in the order to draw.
        members: ``groups.grouping(...)["members"]`` (rank and role of each stock).
        shared: One price range for every panel (a quiet stock draws small candles beside a wild
            one); else each panel is fitted to its own stock's range.
        title: Figure title.
    """
    rebased = {s: d[["open", "high", "low", "close"]] / d["close"].iloc[0] * 100
               for s, d in days.items()}
    first = min(d.index[0] for d in days.values())
    labels = []
    for s, d in days.items():
        move = d["close"].iloc[-1] / d["close"].iloc[0] - 1
        day = ((d["high"] - d["low"]) / d["close"]).median()
        late = f" from {d.index[0]:%Y-%m-%d}" if d.index[0] - first > pd.Timedelta(days=10) else ""
        labels.append(f"{s} · #{int(members.loc[s, 'rank'])} ({members.loc[s, 'role']}) · "
                      f"{move:+.0%} over the window{late} · typical day's range {day:.1%}")
    n = len(rebased)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=labels,
                        vertical_spacing=min(0.06, 0.5 / max(n - 1, 1)))
    for i, (s, r) in enumerate(rebased.items(), start=1):
        fig.add_trace(go.Candlestick(
            x=r.index, open=r["open"], high=r["high"], low=r["low"], close=r["close"], name=s,
            showlegend=False, increasing={"line": {"color": UP}, "fillcolor": UP},
            decreasing={"line": {"color": DOWN}, "fillcolor": DOWN}), row=i, col=1)
        fig.add_hline(y=100, line={"color": _MUTED, "width": 1, "dash": "dot"}, row=i, col=1)
    fig.update_xaxes(rangeslider_visible=False,
                     rangebreaks=_day_breaks(pd.concat(rebased.values(), axis=1).index))
    fig.update_yaxes(type="log", title_text="price, 100 at the start")
    if shared:
        lo = min(float(r["low"].min()) for r in rebased.values())
        hi = max(float(r["high"].max()) for r in rebased.values())
        fig.update_yaxes(range=[np.log10(lo / 1.04), np.log10(hi * 1.04)])
    fig = _finish(fig, title=title or "The group's stocks, candle by candle", height=140 + 220 * n)
    return fig.update_layout(showlegend=False, margin={"t": 90, "b": 50})
