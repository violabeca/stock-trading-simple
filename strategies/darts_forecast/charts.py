"""darts_forecast's figure helpers: the pieces its notebooks build their Plotly figures from.

Loaded by path from the notebooks, like ``research.py``; ``strategy.py`` never imports it, and it
never loads ``research.py`` (a notebook passes in what a helper needs). Like
``strategies/informed_flow/plots.py`` it lives in the strategy folder because it is specific to
these notebooks (the split timelines, the book against buy-and-hold), and it reuses the engine's
palette, layout and thinning (:mod:`trading_simple.plots`) so the figures share the repo's look
and its committed-notebook size discipline.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_simple import backtest as bt
from trading_simple.plots import (
    _GRID,
    _INK,
    _MUTED,
    DIVERGING,
    SERIES,
    _base,
    _intraday_breaks,
    _scatter,
)


def below(fig: go.Figure, y: float = -0.12) -> go.Figure:
    """Put the legend under the plot: the default top-right legend sits on the title and the
    subplot titles.

    Args:
        fig: The figure.
        y: The legend's top, in paper units (negative = below the plot area).

    Returns:
        The same figure, for chaining (``ch.below(fig).show()``).
    """
    fig.update_layout(
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": y, "yanchor": "top"}
    )
    return fig


def _named(fig: go.Figure) -> set:
    """The trace names already in a figure: a timeline shows one legend entry per name."""
    return {t.name for t in fig.data}


def bar(fig: go.Figure, row: str, a, b, color: str, name: str, label: str = "",
        opacity: float = 1.0, pattern: str = "") -> None:
    """One bar of a date timeline (the data notebook's Fig 2a) on ``row``, from day ``a``
    (included) to day ``b`` (excluded). The first bar of each ``name`` carries its legend entry.

    Args:
        fig: The figure (an overlay bar chart with a date x axis).
        row: The category on the y axis.
        a: First day.
        b: The day after the last.
        color: The bar's color.
        name: The legend entry.
        label: Text inside the bar.
        opacity: The bar's opacity.
        pattern: A plotly fill pattern ("/", "x", ...), "" for none.
    """
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    fig.add_trace(
        go.Bar(
            y=[row],
            x=[(b - a) / pd.Timedelta(milliseconds=1)],
            base=[a.strftime("%Y-%m-%d")],
            orientation="h",
            marker={"color": color, "opacity": opacity, "pattern": {"shape": pattern}},
            text=[label],
            textposition="inside",
            insidetextanchor="middle",
            name=name,
            legendgroup=name,
            showlegend=name not in _named(fig),
            hovertemplate=f"<b>{name}</b><br>{a:%Y-%m-%d} -> "
            f"{b - pd.Timedelta(days=1):%Y-%m-%d}<extra></extra>",
        )
    )


def seg(fig: go.Figure, y: str, a, b, color: str, name: str, dash: str | None = None) -> None:
    """One thick line segment of a date timeline (the strategy notebook's Fig 1), from ``a`` to
    ``b`` on row ``y``. The first segment of each ``name`` carries its legend entry.

    Args:
        fig: The figure.
        y: The row (a category).
        a: Start date.
        b: End date.
        color: The segment's color.
        name: The legend entry.
        dash: A plotly dash style ("dot", ...), None for solid.
    """
    fig.add_trace(
        go.Scatter(
            x=[pd.Timestamp(a), pd.Timestamp(b)],
            y=[y, y],
            mode="lines",
            name=name,
            legendgroup=name,
            showlegend=name not in _named(fig),
            line={"color": color, "width": 14, "dash": dash},
            hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<extra>{y}</extra>",
        )
    )


def fit_window(fig: go.Figure, y: str, r, val: int, sess: pd.Index) -> None:
    """The TRAIN (and VALIDATION) sessions of the fit in fit-log row ``r``, as :func:`seg`
    segments on timeline row ``y``.

    Args:
        fig: The figure.
        y: The row.
        r: A row of the fit log (``train_from``, ``train_last``, ``session``).
        val: VALIDATION sessions (0 for gbm: none drawn).
        sess: The panel's session dates (``datetime.date``), in order.
    """
    seg(fig, y, r["train_from"], r["train_last"], SERIES[1], "TRAIN")
    if val:
        i = sess.get_loc(pd.Timestamp(r["session"]).date())
        seg(fig, y, sess[i - val], sess[i - 1], SERIES[4], "VALIDATION (early stopping)")


def span(fig: go.Figure, row: str, a, b, color: str, name: str, *, pattern: str | None = None,
         opacity: float = 1.0) -> None:
    """A bar of a date timeline (the diagnostics notebook's Fig 2) on ``row``, from the start of
    day ``a`` to the end of day ``b``. The first bar of each ``name`` carries its legend entry.

    Args:
        fig: The figure (an overlay bar chart with a date x axis).
        row: The category on the y axis.
        a: First day.
        b: Last day (included).
        color: The bar's color.
        name: The legend entry.
        pattern: A plotly fill pattern, None for none.
        opacity: The bar's opacity.
    """
    a, b = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1)
    marker = {"color": color, **({"pattern": {"shape": pattern}} if pattern else {})}
    fig.add_trace(
        go.Bar(
            y=[row],
            x=[(b - a).total_seconds() * 1e3],
            base=[a],
            orientation="h",
            name=name,
            legendgroup=name,
            showlegend=name not in _named(fig),
            opacity=opacity,
            marker=marker,
            hovertemplate=f"{name}<br>{a:%Y-%m-%d} → {b - pd.Timedelta(days=1):%Y-%m-%d}"
            "<extra></extra>",
        )
    )


def overlay(books: dict, title: str, base, init_cash: float) -> go.Figure:
    """Cumulative return of each book against the baseline's, all from the same first bar
    (thinned like every long series in the repo).

    Args:
        books: ``{label: vectorbt portfolio}``, drawn in :data:`SERIES` order.
        title: The figure title.
        base: The baseline's (buy-and-hold) portfolio, drawn dashed in gray.
        init_cash: The starting cash.

    Returns:
        The figure (legend under the plot).
    """
    fig = go.Figure()
    curve = bt.equity_curve(base, init_cash=init_cash)["value"] / init_cash - 1.0
    fig.add_trace(
        _scatter(
            curve,
            mode="lines",
            name="baseline: equal-weight buy & hold",
            line={"color": _MUTED, "dash": "dash", "width": 1.5},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:+.1%}<extra>baseline</extra>",
        )
    )
    for i, (label, book) in enumerate(books.items()):
        curve = bt.equity_curve(book, init_cash=init_cash)["value"] / init_cash - 1.0
        fig.add_trace(
            _scatter(
                curve,
                mode="lines",
                name=label,
                line={"color": SERIES[i % len(SERIES)], "width": 1.5},
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:+.1%}<extra>" + label + "</extra>",
            )
        )
    _base(fig, title=title, height=440)
    fig.update_xaxes(type="date")
    fig.update_yaxes(tickformat="+.0%", title_text="cumulative return")
    return below(fig)


# ---------------------------------------------------------------------------------------------
# The walkthrough's figures (darts_forecast_walkthrough.ipynb). Hourly series sit on a TRADING
# time axis: session number + the fraction of the session at the bar's close (nights, weekends
# and holidays take no room); the ticks show the dates.
# ---------------------------------------------------------------------------------------------
_LABEL = {"edge": "model (gbm)", "p_up": "classifier P(up)", "b_up": "always up",
          "b_drift": "drift", "b_mom": "momentum", "b_rev": "reversal"}


def _clock(g: dict, b) -> np.ndarray:
    """Grid bars on the trading-time axis: session number + the share of the session's 390
    minutes elapsed at the bar's close."""
    return g["code"][b] + np.minimum((g["pos"][b] + 1) * g["bar_minutes"], 390) / 390


def _days(fig: go.Figure, g: dict, first: int, last: int, *, extra: dict | None = None,
          row: int | None = None) -> None:
    """Date ticks on a trading-time x axis for sessions ``first`` .. ``last`` (about 10 ticks,
    clipped to the sessions in the data; ``extra``: more ``{x: label}`` ticks, e.g. a target day
    past the data)."""
    dates = g["date"][g["sess_first"]]
    first, last = max(first, 0), min(last, len(dates) - 1)
    step = max(1, round((last - first + 1) / 10))
    ticks = {k: f"{pd.Timestamp(dates[k]):%b %d}" for k in range(first, last + 1, step)}
    ticks.update(extra or {})
    kw = {"row": row, "col": 1} if row else {}
    fig.update_xaxes(tickvals=list(ticks), ticktext=list(ticks.values()), **kw)


def split_timeline(sessions: list[str], windows: dict, *, warmup: int, first_fit: str,
                   last_decision: str, read_test: bool = False) -> go.Figure:
    """Which months train, which are graded and which stay locked, as a date timeline.

    Args:
        sessions: The panel's session dates ("YYYY-MM-DD"), in order.
        windows: ``research.WINDOWS``.
        warmup: Warm-up sessions (inputs only).
        first_fit: The session of the first fit (the first graded decision).
        last_decision: The last graded decision's session (its outcome ends inside TUNE).
        read_test: Whether TEST is read.

    Returns:
        The figure.
    """
    fig = go.Figure()
    i = sessions.index(first_fit)
    tune = f"TUNE {windows['tune'][0]} → {windows['tune'][1]}"
    nxt = sessions[sessions.index(last_decision) + 1]
    span(fig, tune, sessions[0], sessions[warmup - 1], _MUTED, "warm-up")
    span(fig, tune, sessions[warmup], sessions[i - 1], SERIES[1], "history before the 1st fit")
    span(fig, tune, first_fit, last_decision, SERIES[0], "graded")
    span(fig, tune, nxt, windows["tune"][1], SERIES[2], "graded nowhere", pattern="/")
    a, b = windows["test"]
    span(fig, "TEST", a, b, SERIES[4] if read_test else _GRID,
         "TEST: read once" if read_test else "locked", pattern="" if read_test else "x")
    span(fig, "HOLDOUT", windows["holdout"][0], f"{pd.Timestamp.today():%Y-%m-%d}", _GRID,
         "locked", pattern="x")
    _base(fig, title="The split: what is learned from, what is graded, what stays locked",
          height=300)
    fig.update_layout(barmode="overlay", bargap=0.35)
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed")
    return below(fig, -0.2)


def hourly_day(panel, inp: dict, name: str, day: str, days: int = 5) -> go.Figure:
    """One stock over a few sessions: the 1-minute closes, the hourly decision closes built from
    them, and the overnight gaps between sessions.

    Args:
        panel: The 1-minute panel (``inp``'s).
        inp: ``strategy.build_inputs`` output.
        name: The stock.
        day: The first session ("YYYY-MM-DD").
        days: Sessions to draw (fewer when the data ends sooner).

    Returns:
        The figure.
    """
    g, j = inp["g"], inp["names"].index(name)
    sess = [str(d) for d in g["date"][g["sess_first"]]]
    k0 = sess.index(day)
    days = min(days, len(sess) - k0)
    local = panel.close.index.tz_convert(panel.tz)
    minute = np.asarray(local.hour * 60 + local.minute) - 570
    keep = (g["codes_1m"] >= k0) & (g["codes_1m"] < k0 + days) & g["regular_1m"]
    rows = np.flatnonzero(keep)
    x = g["codes_1m"][rows] + (minute[rows] + 1) / 390
    y = panel.close[name].to_numpy()[rows]
    cut = np.flatnonzero(np.diff(g["codes_1m"][rows]) > 0) + 1          # break between sessions
    line = pd.Series(np.insert(y, cut, np.nan), index=np.insert(x, cut, np.nan))
    fig = go.Figure(_scatter(line, mode="lines", name="1-minute close", connectgaps=False,
                             line={"color": _MUTED, "width": 1}))
    b = np.flatnonzero((g["code"] >= k0) & (g["code"] < k0 + days) & g["regular"])
    fig.add_trace(go.Scatter(x=_clock(g, b), y=inp["close"][b, j], mode="markers",
                             name="hourly close = a decision point",
                             marker={"color": SERIES[0], "size": 8}))
    gx, gy = [], []
    for k in cut:                                       # previous close -> the first trade
        gx += [x[k], x[k], None]
        gy += [y[k - 1], panel.open_[name].to_numpy()[rows[k]], None]
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", name="overnight gap",
                             line={"color": SERIES[7], "width": 3}))
    _base(fig, title=f"{name}: from 1-minute bars to 7 hourly decision points a day", height=420)
    _days(fig, g, k0, k0 + days - 1)
    fig.update_yaxes(title_text="price ($)")
    return below(fig, -0.12)


def target_picture(inp: dict, name: str, s: int, before: int = 20, after: int = 3,
                   lags: int = 7) -> go.Figure:
    """One forecast's question on the hourly closes: what the model reads (the daily summaries
    reach 20 sessions back; hourly, its own 5-day change at the last ``lags`` closes and the last
    hour's inputs) and what it must predict (the change to the same clock time ``horizon`` bars
    later).

    Args:
        inp: ``strategy.build_inputs`` output.
        name: The stock.
        s: The decision's grid bar (``research.bucket_at``).
        before: Sessions drawn before the decision.
        after: Sessions drawn after the target.
        lags: The target's own past values the model reads (``lookback``).

    Returns:
        The figure.
    """
    g, H, j = inp["g"], inp["horizon"], inp["names"].index(name)
    k = int(g["code"][s])
    b = np.flatnonzero((g["code"] >= k - before) & (g["code"] <= g["code"][s + H] + after)
                       & g["regular"])
    x, xs, xt = _clock(g, b), _clock(g, s), _clock(g, s + H)
    now, then = inp["close"][s, j], inp["close"][s + H, j]
    fig = go.Figure(go.Scatter(x=x, y=inp["close"][b, j], mode="lines", name="hourly close",
                               line={"color": _INK, "width": 1.5}))
    fig.add_trace(go.Scatter(x=[xs, xt], y=[now, then], mode="markers+text",
                             name="price now, price 5 sessions later",
                             text=[f"now ${now:,.2f}", f"then ${then:,.2f}"],
                             textposition=["top center", "bottom center"],
                             marker={"color": SERIES[5], "size": 11}))
    fig.add_annotation(x=xt, y=then, ax=xs, ay=now, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=SERIES[5], text="")
    fig.add_annotation(x=(xs + xt) / 2, y=(now + then) / 2, text=f"{then / now - 1:+.1%}",
                       showarrow=False, xshift=34, font={"color": SERIES[5], "size": 14})
    for x0, x1, color, label, where in (
            (k - 20, xs, SERIES[1], "daily summaries: 20 sessions back", "top left"),
            (_clock(g, s - lags), xs, SERIES[0],
             f"hourly: its own 5-day change<br>at the last {lags} closes", "bottom left"),
            (xs, xt, SERIES[2], "what it predicts: 5 sessions", "top right")):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.13, line_width=0, layer="below",
                      annotation_text=label, annotation_position=where)
    _base(fig, title=f"{name}: the question one forecast answers", height=440)
    _days(fig, g, max(0, k - before), int(g["code"][s + H]) + after)
    fig.update_yaxes(title_text="price ($)")
    return below(fig, -0.12)


def outcome_hist(t: pd.DataFrame) -> go.Figure:
    """The graded 5-day changes, one small panel per name: how wide they are and which way they
    leaned (the share of up-weeks is in each title).

    Args:
        t: A ``research.horizon_table``.

    Returns:
        The figure.
    """
    names = sorted(t["name"].unique())
    up = t.groupby("name")["r_pct"].apply(lambda r: (r > 0).mean())
    fig = make_subplots(rows=1, cols=len(names), shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=[f"{n}: up {up[n]:.0%}" for n in names])
    lim = float(np.ceil(t["r_pct"].abs().quantile(0.995) * 100))
    for i, n in enumerate(names, start=1):
        r = 100 * t.loc[t["name"] == n, "r_pct"]
        fig.add_trace(go.Histogram(x=r, xbins={"start": -lim, "end": lim, "size": lim / 15},
                                   marker_color=SERIES[i - 1], name=n, showlegend=False),
                      row=1, col=i)
        fig.add_vline(x=0, line={"color": _INK, "width": 1}, row=1, col=i)
        fig.add_vline(x=float(r.mean()), line={"color": _INK, "width": 1, "dash": "dot"},
                      row=1, col=i)
    _base(fig, title="The 5-day changes being predicted (%, dotted line = the mean)", height=330)
    fig.update_xaxes(range=[-lim, lim])
    fig.update_yaxes(title_text="graded forecasts", row=1, col=1)
    fig.update_layout(margin={"t": 80})
    return fig


def fit_timeline(fc: dict) -> go.Figure:
    """The walk-forward, one row per fit: what each fit learned from (decisions whose 5-day
    outcome was already known), the recent decisions it had to skip (outcome still open) and
    the session it then forecast.

    Args:
        fc: A forecast (its ``log`` and grid).

    Returns:
        The figure.
    """
    inp, log = fc["inputs"], fc["log"]
    g, H = inp["g"], inp["horizon"]
    day = lambda b: pd.Timestamp(g["date"][b])  # noqa: E731
    pts = log["retrain_bucket"].to_numpy(int)
    bars, cols = np.nonzero(np.isfinite(fc["fit"]))
    last = pd.Series(bars).groupby(fc["fit"][bars, cols].astype(int)).max()
    fig, y = go.Figure(), log["fit"].to_numpy() + 1
    ms = pd.Timedelta(milliseconds=1)
    one = pd.Timedelta(days=1)
    parts = [("learns from: decisions with a known outcome",
              [pd.Timestamp(a) for a in log["train_from"]], [day(T - 1 - H) + one for T in pts],
              SERIES[0]),
             ("skipped: outcome not known yet", [day(T - 1 - H) + one for T in pts],
              [day(T) for T in pts], SERIES[2]),
             ("forecasts: out of sample", [day(T) for T in pts],
              [day(last[k]) + one for k in log["fit"]], SERIES[1])]
    for label, a, b, color in parts:
        fig.add_trace(go.Bar(y=y, base=[x.isoformat() for x in a],
                             x=[(q - p) / ms for p, q in zip(a, b, strict=True)],
                             orientation="h", name=label, width=1.0,
                             marker={"color": color, "line": {"width": 0}},
                             hovertemplate=label + " · fit %{y}<extra></extra>"))
    fig.add_vrect(x0=day(0), x1=pd.Timestamp(log["train_from"].iloc[0]), fillcolor=_MUTED,
                  opacity=0.15, line_width=0, layer="below", annotation_text="warm-up",
                  annotation_position="top left")
    _base(fig, title="The walk-forward: every fit learns from the past only", height=520)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed", title_text="fit")
    return below(fig, -0.1)


def _tt(t: pd.DataFrame) -> np.ndarray:
    """A table's target times on the trading-time axis."""
    m = t["target_ts"].dt.hour * 60 + t["target_ts"].dt.minute - 570
    return (t["target_session"] + m / 390).to_numpy()


def price_vs_forecast(t: pd.DataFrame, name: str) -> go.Figure:
    """The flattering picture: the actual price, the model's predicted price and the "no change"
    price, each drawn at the time it is about (the target time).

    Args:
        t: A ``research.horizon_table``.
        name: The stock.

    Returns:
        The figure.
    """
    x = t[t["name"] == name].sort_values("target_bucket")
    tx = _tt(x)
    fig = go.Figure()
    for col, label, line in (("price_then", "actual price", {"color": _INK, "width": 1.5}),
                             ("pred_price", "model's predicted price",
                              {"color": SERIES[0], "width": 1.2}),
                             ("price_now", "'no change' guess",
                              {"color": _MUTED, "width": 1.2, "dash": "dot"})):
        fig.add_trace(_scatter(pd.Series(x[col].to_numpy(), index=tx), mode="lines", name=label,
                               line=line))
    _base(fig, title=f"{name}: predicted vs actual price, drawn at the target time", height=440)
    sess = x.drop_duplicates("target_session")
    step = max(1, len(sess) // 10)
    fig.update_xaxes(tickvals=sess["target_session"].to_numpy()[::step],
                     ticktext=[f"{d:%b %d}" for d in sess["target_ts"].iloc[::step]])
    fig.update_yaxes(title_text="price ($)")
    return below(fig, -0.12)


def lag_profile(prof: pd.DataFrame) -> go.Figure:
    """How far the predicted price sits from the actual price k sessions before its target time
    (``research.lag_profile``): the closest k says what the forecast really tracks.

    Args:
        prof: ``research.lag_profile`` output.

    Returns:
        The figure.
    """
    best = prof.loc[prof["typical gap %"].idxmin()]
    fig = go.Figure(go.Scatter(x=prof["sessions back"], y=prof["typical gap %"],
                               mode="lines+markers", name="typical gap",
                               line={"color": SERIES[0], "width": 2}))
    fig.add_trace(go.Scatter(x=[best["sessions back"]], y=[best["typical gap %"]],
                             mode="markers+text", text=[f"closest: {best['sessions back']:.0f}"
                                                        " sessions back"],
                             textposition="top right", name="closest",
                             marker={"color": SERIES[5], "size": 12}))
    _base(fig, title="The predicted price is closest to the price when it was made", height=360)
    fig.update_xaxes(title_text="k: the actual price k sessions before the target time",
                     dtick=1)
    fig.update_yaxes(title_text="typical gap to the predicted price (%)")
    return below(fig, -0.25)


def return_scatter(t: pd.DataFrame, horizon: str = "5-day") -> go.Figure:
    """The honest picture: the predicted change over the horizon against the actual one, every
    graded forecast, one colour per name. A forecast with skill tilts the cloud up along the
    diagonal.

    Args:
        t: A ``research.horizon_table``.
        horizon: The horizon in words (titles and axes).

    Returns:
        The figure.
    """
    fig = go.Figure()
    for i, (n, x) in enumerate(t.groupby("name")):
        fig.add_trace(go.Scatter(x=100 * x["f_pct"], y=100 * x["r_pct"], mode="markers", name=n,
                                 marker={"color": SERIES[i % len(SERIES)], "size": 4,
                                         "opacity": 0.35}))
    lim = float(100 * np.nanpercentile(np.abs(t[["f_pct", "r_pct"]].to_numpy()), 99.5))
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                             name="perfect forecast (diagonal)",
                             line={"color": _INK, "width": 1, "dash": "dash"}))
    fig.add_hline(y=0, line={"color": _MUTED, "width": 1})
    fig.add_vline(x=0, line={"color": _MUTED, "width": 1})
    _base(fig, title=f"Predicted vs actual {horizon} change, every graded forecast",
          height=520)
    fig.update_xaxes(title_text=f"predicted {horizon} change (%)", range=[-lim, lim])
    fig.update_yaxes(title_text=f"actual {horizon} change (%)", range=[-lim, lim])
    return below(fig, -0.12)


def dots(sc: pd.DataFrame, metric: str, *, rows: str = "forecast", where: dict | None = None,
         scale: float = 100.0, unit: str = " pts", title: str = "") -> go.Figure:
    """THE estimate-against-luck chart: one row per forecast (or group), the grey bar is where
    luck alone lands 95% of the time (the luck band's mean ± 1.96 SD), the dot is the measured
    value. A dot outside its grey bar is more than luck.

    Args:
        sc: ``research.scores`` output (or any frame with its columns).
        metric: The metric to draw (e.g. ``"skill"``, ``"ic_name"``).
        rows: The column naming the rows (``"forecast"`` or ``"group"``).
        where: Further filters, ``{column: value}``.
        scale: Multiplies the values (100: fractions as points).
        unit: Hover unit.
        title: The figure title.

    Returns:
        The figure.
    """
    x = sc[sc["metric"] == metric]
    for k, v in (where or {}).items():
        x = x[x[k] == v]
    x = x[np.isfinite(x["null_sd"])]
    names = [_LABEL.get(r, str(r)).replace(" - ", " − ") for r in x[rows]]
    for a, b in _LABEL.items():
        names = [n.replace(a, b) for n in names]
    v, mu, sd = (scale * x[c].to_numpy(float) for c in ("value", "null_mean", "null_sd"))
    color = [_INK if str(r).startswith("b_") else SERIES[4] if r == "p_up" else
             _MUTED if "coin-flip" in str(r) else SERIES[0] for r in x[rows]]
    fig = go.Figure(go.Scatter(x=mu, y=names, mode="markers", name="luck alone (95% of draws)",
                               marker={"color": _MUTED, "size": 5, "symbol": "line-ns-open"},
                               error_x={"type": "data", "array": 1.96 * sd, "color": _GRID,
                                        "thickness": 12, "width": 0}))
    fig.add_trace(go.Scatter(x=v, y=names, mode="markers", name="measured value",
                             marker={"color": color, "size": 11},
                             customdata=np.c_[x["z"].to_numpy()],
                             hovertemplate="%{y}: %{x:+.2f}" + unit +
                             " (z %{customdata[0]:+.1f})<extra></extra>"))
    fig.add_vline(x=0, line={"color": _MUTED, "width": 1, "dash": "dot"})
    height = 120 + 42 * len(names)
    _base(fig, title=title, height=height)
    fig.update_yaxes(autorange="reversed")
    return below(fig, -45 / (height - 90))                 # about 45 px under the plot


def reliability(rel: pd.DataFrame) -> go.Figure:
    """Is "P(up)" honest? Each dot is one fifth of the forecasts: the mean P(up) it was given
    against the share that really went up, with its error bar (the share ± 1.96 luck-SDs); the
    diagonal is an honest probability, a flat row is one that knows nothing.

    Args:
        rel: ``research.reliability`` output.

    Returns:
        The figure.
    """
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="honest (diagonal)",
                               line={"color": _MUTED, "width": 1, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=rel["mean P(up)"], y=rel["real up share"], mode="markers",
                             name="each fifth of the forecasts (error bar: ± 1.96 luck-SDs)",
                             marker={"color": SERIES[4], "size": 12},
                             error_y={"type": "data", "symmetric": False,
                                      "array": rel["hi"] - rel["real up share"],
                                      "arrayminus": rel["real up share"] - rel["lo"],
                                      "color": SERIES[4]}))
    a = rel.attrs
    _base(fig, title=f"P(up) against what happened · Brier {a['brier']:.3f} (coin 0.250, the "
          f"name's past up share {a['brier_trailing']:.3f})", height=440)
    fig.update_xaxes(title_text="predicted P(up)", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title_text="share that went up", range=[0, 1], tickformat=".0%")
    return below(fig, -0.18)


def fan(latest: pd.DataFrame, inp: dict, name: str, sessions: int = 20) -> go.Figure:
    """The end of the pipeline for one stock: the last sessions' hourly closes, the predicted
    price 5 sessions ahead, and the name's typical range of 5-day changes around today's price.

    Args:
        latest: ``research.latest_forecast`` output.
        inp: The inputs it was made from (the same panel).
        name: The stock.
        sessions: History sessions drawn.

    Returns:
        The figure.
    """
    g, H, j = inp["g"], inp["horizon"], inp["names"].index(name)
    r = latest.loc[name]
    last = len(g["code"]) - 1
    b = np.flatnonzero((g["code"] > g["code"][last] - sessions) & g["regular"])
    x0 = float(_clock(g, last))
    x1 = x0 + H / g["n_pos"]
    lo, hi = (r["price now"] * (1 + r[c] / 100) for c in ("typical 10% change",
                                                          "typical 90% change"))
    fig = go.Figure(go.Scatter(x=_clock(g, b), y=inp["close"][b, j], mode="lines",
                               name="hourly close", line={"color": _INK, "width": 1.5}))
    fig.add_trace(go.Scatter(x=[x0, x1, x1, x0], y=[r["price now"], hi, lo, r["price now"]],
                             fill="toself", mode="lines", name="typical range of 5-day changes "
                             "(10% to 90%)", fillcolor="rgba(137,135,129,0.18)",
                             line={"color": _MUTED, "width": 0.5}))
    fig.add_trace(go.Scatter(x=[x0, x1], y=[r["price now"]] * 2, mode="lines",
                             name="'no change'", line={"color": _MUTED, "dash": "dot"}))
    label = f"${r['predicted price']:,.2f} ({r['forecast change %']:+.1f}%)"
    fig.add_trace(go.Scatter(x=[x1], y=[r["predicted price"]], mode="markers+text",
                             name="predicted price", text=[label], textposition="middle right",
                             marker={"color": SERIES[0], "size": 12}))
    _base(fig, title=f"{name}: the latest forecast, made {r['decision time']:%Y-%m-%d %H:%M} "
          f"for {r['target time']:%Y-%m-%d %H:%M}", height=440)
    _days(fig, g, int(g["code"][last]) - sessions + 1, int(g["code"][last]),
          extra={x1: f"{r['target time']:%b %d}"})
    fig.update_xaxes(range=[_clock(g, b[0]) - 0.2, x1 + 4])
    fig.update_yaxes(title_text="price ($)")
    return below(fig, -0.12)


# ---------------------------------------------------------------------------------------------
# The explorer (darts_forecast_explorer.ipynb): the picks, the split, the path search.
# ---------------------------------------------------------------------------------------------
def universe_map(c: pd.DataFrame, picks: pd.DataFrame) -> go.Figure:
    """Every name that passes the data rules by market cap and volatility, the picks marked
    (circles: slow, diamonds: fast; one colour per cap group and speed).

    Args:
        c: ``explorer.candidates``.
        picks: ``explorer.pick``.

    Returns:
        The figure.
    """
    ok = c[c["ok"]]
    fig = go.Figure(go.Scatter(
        x=ok["market_cap"] / 1e9, y=100 * ok["vol"], mode="markers",
        name="passes the rules", text=ok.index,
        marker={"color": _GRID, "size": 6, "line": {"color": _MUTED, "width": 0.5}},
        hovertemplate="%{text}: $%{x:.1f}B, %{y:.0f}% a year<extra></extra>"))
    for i, ((grp, spd), x) in enumerate(picks.groupby(["cap_group", "speed"], sort=False)):
        fig.add_trace(go.Scatter(
            x=x["market_cap"] / 1e9, y=100 * x["vol"], mode="markers+text", text=x["symbol"],
            textposition="top center", name=f"{grp}, {spd}",
            marker={"color": SERIES[i % len(SERIES)], "size": 12,
                    "symbol": "circle" if spd == "slow" else "diamond"},
            hovertemplate="%{text}: $%{x:.1f}B, %{y:.0f}% a year<extra></extra>"))
    for v, label in ((2, "$2B"), (10, "$10B")):        # a log axis: annotations sit at log10(x)
        fig.add_vline(x=v, line={"color": _MUTED, "width": 1, "dash": "dot"})
        fig.add_annotation(x=np.log10(v), y=1.0, yref="paper", text=label, showarrow=False,
                           yanchor="bottom", font={"color": _MUTED})
    _base(fig, title="The candidates and the picks: market cap vs how fast they move", height=520)
    fig.update_xaxes(type="log", title_text="market cap today ($ billion, log scale)")
    fig.update_yaxes(type="log", title_text="volatility before the window (% a year, log scale)")
    return below(fig, -0.13)


def price_paths(panel, picks: pd.DataFrame | None, sp: dict) -> go.Figure:
    """What the names did: each one's daily close as a share of its close on the first TRAIN
    session (100 = unchanged, log scale), one colour per cap group and speed (the second name of
    a group dotted), VALIDATION shaded. It stops where TEST starts: the TEST month stays unseen
    until it is graded.

    Args:
        panel: The 1-minute panel.
        picks: ``explorer.pick`` (None: one colour per name).
        sp: ``explorer.load_window``'s split.

    Returns:
        The figure.
    """
    day = pd.Index(panel.close.index.tz_convert(panel.tz).date)
    daily = panel.close.groupby(day).last()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.loc[:pd.Timestamp(sp["validation"][1])]
    rel = 100 * daily / daily.loc[pd.Timestamp(sp["train"][0]):].iloc[0]
    cat = picks.set_index("symbol")[["cap_group", "speed"]] if picks is not None else None
    groups = [] if cat is None else list(dict.fromkeys(map(tuple, cat.to_numpy())))
    fig, seen = go.Figure(), {}
    for i, n in enumerate(rel.columns):
        k = tuple(cat.loc[n]) if cat is not None and n in cat.index else None
        color = SERIES[groups.index(k) % len(SERIES)] if k in groups else SERIES[i % len(SERIES)]
        seen[k] = seen.get(k, 0) + 1
        fig.add_trace(go.Scatter(
            x=rel.index, y=rel[n], mode="lines", name=n + (f" ({k[0]}, {k[1]})" if k else ""),
            line={"color": color, "width": 1.6, "dash": "dot" if seen[k] > 1 else "solid"},
            hovertemplate=n + " %{x|%Y-%m-%d}: %{y:.0f}<extra></extra>"))
    a, b = sp["validation"]
    fig.add_vrect(x0=pd.Timestamp(a), x1=pd.Timestamp(b) + pd.Timedelta(days=1),
                  fillcolor=SERIES[0], opacity=0.07, line_width=0, layer="below",
                  annotation_text="VALIDATION", annotation_position="top left")
    fig.add_hline(y=100, line={"color": _MUTED, "width": 1, "dash": "dot"})
    _base(fig, title="What the stocks did up to TEST (daily close, 100 = the first TRAIN session)",
          height=500)
    fig.update_yaxes(type="log", title_text="price, 100 = start (log scale)")
    return below(fig, -0.1)


def _wall(times) -> np.ndarray:
    """Times as wall-clock text ("2025-12-08 10:00:00"). plotly.js 3 reads a NUMBER on a date axis
    in the viewer's own time zone but text as written: epoch milliseconds (``plots._epoch``) put a
    New York clock an hour or two late in a European browser, and a night break (given as text)
    then hides the last hour of the session before it. Text is right everywhere."""
    return np.asarray(pd.DatetimeIndex(times).strftime("%Y-%m-%d %H:%M:%S"))


def _mark(fig: go.Figure, at, text: str, **kw) -> None:
    """A dotted vertical line at time ``at`` with a label at the top (``add_vline``'s own label
    cannot take a time)."""
    x = _wall([at])[0]
    fig.add_vline(x=x, line={"color": _MUTED, "width": 1, "dash": "dot"}, **kw)
    fig.add_annotation(x=x, y=1.0, yref="paper", text=text, showarrow=False, xanchor="left",
                       yanchor="bottom", font={"color": _MUTED})


def _best_of_bar(k: int, level: float = 0.975) -> float:
    """Φ⁻¹(level^(1/k)): the z the best of ``k`` independent tries exceeds by luck only
    ``1 - level`` of the time (``explorer.best_of_bar``)."""
    from scipy.stats import norm

    return float(norm.ppf(level ** (1 / max(int(k), 1))))


#: How each part of the data split is drawn in :func:`split_calendar`: (color, fill pattern).
SPLIT_PARTS = {"history only": (_MUTED, "/"), "learns": (SERIES[0], None),
               "held out (early stop, β)": (SERIES[2], None), "scores": (SERIES[1], None),
               "TEST (read once)": (SERIES[5], "x")}


def split_calendar(rows: list, *, title: str) -> go.Figure:
    """Figure 0: how the days are used, one row per model (a single split has one, a
    walk-forward one per scored month) and one for TEST, each part a bar on the calendar.

    Args:
        rows: ``[(row label, [(part, first day, last day)])]``, parts from :data:`SPLIT_PARTS`.
        title: The figure's title.

    Returns:
        The figure.
    """
    fig = go.Figure()
    for label, parts in rows:
        for part, a, b in parts:
            color, pattern = SPLIT_PARTS[part]
            span(fig, label, a, b, color, part, pattern=pattern)
    _base(fig, title=title, height=170 + 32 * len(rows))
    fig.update_layout(barmode="overlay", margin={"l": 90})
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed")
    return below(fig, -0.25 if len(rows) < 4 else -0.12)


def explorer_split(sp: dict, fits: list | None = None) -> go.Figure:
    """The window on a calendar: warm-up, TRAIN, VALIDATION and TEST, and a dotted line at every
    fit (``[(date, label)]``: the VALIDATION fit after TRAIN, the TEST refit after VALIDATION).

    Args:
        sp: ``explorer.load_window``'s split.
        fits: The fits to mark.

    Returns:
        The figure.
    """
    fig = go.Figure()
    parts = (("warmup", "warm-up: inputs", _MUTED, "/"),
             ("train", "TRAIN: learned", SERIES[1], None),
             ("validation", "VALIDATION: scored", SERIES[0], None),
             ("test", "TEST: read once", SERIES[5], "x"))
    for k, row, color, pattern in parts:
        a, b = sp[k]
        n = sum(a <= d <= b for d in sp["sessions"])
        span(fig, row, a, b, color, row, pattern=pattern)
        fig.add_annotation(x=pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2,
                           y=row, text=f"{n} sessions", showarrow=False,
                           font={"color": "white" if k != "warmup" else _INK},
                           bgcolor="rgba(0,0,0,0.25)" if k != "warmup" else None)
    for d, label in fits or []:
        fig.add_vline(x=pd.Timestamp(d), line={"color": _INK, "width": 1, "dash": "dot"},
                      annotation_text=label, annotation_position="top")
    _base(fig, title="The window: what each part is for", height=320)
    fig.update_layout(barmode="overlay", showlegend=False, margin={"l": 140})
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed")
    return fig


def path_picture(panel, examples: list[dict], name: str, ts, *, before: int = 90) -> go.Figure:
    """One decision, several configurations: the actual price (1-minute closes, black) from
    ``before`` trading minutes before the decision to the end of the longest predicted path, and
    each configuration's predicted price path (``explorer.example_paths`` or
    ``intraday.example_paths``): solid with dots the graded (scaled) path, dotted the network's own.
    Every price sits at the time it was a close (a 1-minute bar at its end); nights and weekends
    are cut out of the time axis.

    Args:
        panel: The 1-minute panel.
        examples: ``explorer.example_paths`` output.
        name: The stock.
        ts: The decision time (New York).
        before: Trading minutes drawn before the decision.

    Returns:
        The figure.
    """
    t0 = pd.Timestamp(ts)
    end = max((pd.Timestamp(e["times"][-1]) for e in examples), default=t0)
    idx = panel.close.index.tz_convert(panel.tz).tz_localize(None) + pd.Timedelta(minutes=1)
    close = pd.Series(panel.close[name].to_numpy(float), index=idx).ffill()
    lo, hi = max(0, idx.searchsorted(t0) - before), idx.searchsorted(end, side="right")
    act = close.iloc[lo:hi]
    fig = go.Figure(go.Scatter(x=_wall(act.index), y=act.to_numpy(), mode="lines",
                               name="actual price (1-minute closes)",
                               line={"color": _INK, "width": 1.4}))
    for i, e in enumerate(examples):
        x, color = _wall(e["times"]), SERIES[i % len(SERIES)]
        fig.add_trace(go.Scatter(x=x, y=e["price"], mode="lines+markers",
                                 name=f"{e['label']} (scale {e['beta']:.2f})",
                                 line={"color": color, "width": 2}, marker={"size": 4},
                                 legendgroup=e["label"]))
        if "price_raw" in e:
            fig.add_trace(go.Scatter(x=x, y=e["price_raw"], mode="lines", showlegend=False,
                                     name=f"{e['label']}, unscaled", legendgroup=e["label"],
                                     line={"color": color, "width": 1, "dash": "dot"}))
    _mark(fig, t0, "decision")
    _base(fig, title=f"{name}: predicted price paths against the actual price, from "
          f"{t0:%Y-%m-%d %H:%M}", height=480)
    fig.update_xaxes(type="date", rangebreaks=_intraday_breaks(act.index),
                     range=list(_wall([act.index[0], max(end, act.index[-1])])))
    fig.update_yaxes(title_text="price ($)")
    return below(fig, -0.14)


def model_io(d: dict) -> go.Figure:
    """Figure 1: one day-start example, before and after training. Three panels share the day's
    clock. 1a: the day's price (1-minute closes) with the skipped offset (grey), the input window
    (blue), the output window (green) and the unused rest; with a trained model also its
    predicted price from the decision (dashed red). 1b: the target, bar by bar, in typical moves:
    the input bars (blue), the actual output bars (green: the truth) and the model's output (red
    line; until ``pred`` is given a flat dashed line at 0 = "no change"). 1c (with volume): the
    volume input of the input bars; the output window's volume is never an input.

    Args:
        d: ``name``, ``day`` ("YYYY-MM-DD"), ``cell`` ((bar, offset, input, output) minutes),
            ``times`` (the session's 390 one-minute close times, New York wall clock), ``price``
            (the last traded close at each), ``y_in`` / ``y_out`` (the input and the actual
            output bars' moves, typical moves), ``volume`` (the input bars' volume input, or
            None), ``pred`` (the model's output bars, typical moves, or None: not trained) and
            ``pred_price`` (the same as prices: the decision's price, then one per output bar).

    Returns:
        The figure.
    """
    g, off, L, H = d["cell"]
    day, vol, pred = pd.Timestamp(d["day"]), d.get("volume"), d.get("pred")
    at = lambda m: day + pd.Timedelta(minutes=570 + m)  # noqa: E731  (m minutes after 09:30)
    centre = lambda start, n: _wall([at(start + (i + 0.5) * g) for i in range(n)])  # noqa: E731
    x_in, x_out = centre(off, len(d["y_in"])), centre(off + L, len(d["y_out"]))
    wide = g * 60_000.0                                  # a bar's width on a date axis: ms
    rows = 3 if vol is not None else 2
    titles = ["1a · Price and the windows", "1b · Target: bar moves (input, truth, model output)",
              "1c · Volume input"][:rows]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.42, 0.34, 0.24] if vol is not None else [0.55, 0.45],
                        subplot_titles=titles)
    fig.update_annotations(x=0, xanchor="left")          # panel names at the left edge
    fig.add_trace(go.Scatter(x=_wall(d["times"]), y=d["price"], mode="lines",
                             name="price (1-minute closes)", line={"color": _INK, "width": 1.2}),
                  row=1, col=1)
    if d.get("pred_price") is not None:
        fig.add_trace(go.Scatter(
            x=_wall([at(off + L + i * g) for i in range(len(d["pred_price"]))]),
            y=d["pred_price"], mode="lines", name="model: predicted price (before scaling)",
            line={"color": SERIES[5], "width": 2, "dash": "dash"}), row=1, col=1)
    fig.add_trace(go.Bar(x=x_in, y=d["y_in"], width=wide, marker={"color": SERIES[0]},
                         name="input: the bars the model reads"), row=2, col=1)
    fig.add_trace(go.Bar(x=x_out, y=d["y_out"], width=wide, marker={"color": SERIES[1]},
                         name="truth: the bars it must predict"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_out, y=np.zeros(len(x_out)) if pred is None else pred, mode="lines",
        name="model output: not trained yet (flat line at 0 = no change)" if pred is None
        else "model output (the network's own, before scaling)",
        line={"color": _MUTED if pred is None else SERIES[5], "width": 2,
              "dash": "dash" if pred is None else "solid"}), row=2, col=1)
    if vol is not None:
        fig.add_trace(go.Bar(x=x_in, y=vol, width=wide, marker={"color": SERIES[0]},
                             name="volume input (log, against the usual bar)"), row=3, col=1)
    bands = ((0, off, "rgba(137,135,129,0.25)", f"skipped: the first {off} min"),
             (off, off + L, "rgba(42,120,214,0.18)", f"input: {L} min = {L // g} bars"),
             (off + L, off + L + H, "rgba(27,175,122,0.22)",
              f"output: {H} min = {H // g} bars"),
             (off + L + H, 390, "rgba(137,135,129,0.10)", "not used: the rest of the day"))
    for a, b, color, label in bands:                     # after the traces: empty rows drop shapes
        if b > a:
            for r in range(1, rows + 1):
                fig.add_vrect(x0=_wall([at(a)])[0], x1=_wall([at(b)])[0], fillcolor=color,
                              line_width=0, layer="below", row=r, col=1)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                     marker={"symbol": "square", "size": 12, "color": color}),
                          row=1, col=1)
    _mark(fig, at(off + L), "decision", row="all", col="all")
    _base(fig, title=f"Figure 1 · {d['name']} on {d['day']} ({g}-minute bars)",
          height=760 if vol is not None else 600)
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(type="date", range=list(_wall([at(0), at(390)])))
    fig.update_yaxes(title_text="price ($)", row=1, col=1)
    fig.update_yaxes(title_text="bar move (typical moves)", row=2, col=1)
    if vol is not None:
        fig.update_yaxes(title_text="volume (log vs usual)", row=3, col=1)
    return below(fig, -0.12)


def static_inputs(d: dict) -> go.Figure:
    """Figure 2: the previous-day inputs of one stock-day, the two numbers the model gets once
    for the whole day beside the bars: the overnight gap and the previous days' move, in typical
    units, on the axis they are clipped to.

    Args:
        d: ``name``, ``day``, ``gap`` (typical nights), ``prev_move`` (typical moves),
            ``prev_days`` (the sessions behind the move) and ``limit`` (the clip).

    Returns:
        The figure.
    """
    vals = [d["gap"], d["prev_move"]]
    labels = ["overnight gap: today's open vs yesterday's close",
              f"previous {d['prev_days']}-day move: close to close"]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker={"color": SERIES[0]},
                           text=[f"{v:+.2f}" for v in vals], textposition="outside",
                           showlegend=False))
    fig.add_vline(x=0, line={"color": _MUTED, "width": 1})
    _base(fig, title=f"Figure 2 · Previous-day inputs of {d['name']} on {d['day']}, known "
          "before the open", height=240)
    fig.update_layout(bargap=0.5, margin={"l": 330})
    fig.update_xaxes(range=[-d["limit"], d["limit"]],
                     title_text=f"typical units (clipped at ±{d['limit']:g})")
    fig.update_yaxes(autorange="reversed")
    return fig


def training_curve(rows: list[dict], baseline: float, *, epochs: int) -> go.Figure:
    """Figure 3: the training progress, one point per pass. 3a: the mean loss over the training
    days' batches (dropout on) and the loss on the held-out days, against the dashed loss of
    "no change" (predicting 0) on the same held-out days: the loss to beat. 3b: the held-out loss
    as R² against "no change", 1 - loss / baseline: above 0 the model beats saying nothing. The
    pass with the lowest held-out loss, the one kept, is starred.

    Args:
        rows: ``[{"pass", "train", "validation"}]``, one per pass so far.
        baseline: The held-out days' loss of "no change".
        epochs: The most passes (the title).

    Returns:
        The figure.
    """
    p = [r["pass"] for r in rows]
    tr, va = [r["train"] for r in rows], [r["validation"] for r in rows]
    r2 = [1 - v / baseline for v in va]
    best = int(np.argmin(va))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        row_heights=[0.6, 0.4],
                        subplot_titles=["3a · Loss per pass", "3b · Held-out R² against "
                                        "\"no change\", from the loss (all bars pooled)"])
    fig.update_annotations(x=0, xanchor="left")
    fig.add_trace(go.Scatter(x=p, y=tr, mode="lines+markers", line={"color": SERIES[0]},
                             name="training days (mean over batches, dropout on)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=p, y=va, mode="lines+markers", line={"color": SERIES[1]},
                             name="held-out days"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.5, max(p) + 0.5], y=[baseline, baseline], mode="lines",
                             line={"color": _MUTED, "dash": "dash"},
                             name="\"no change\" on the held-out days (the loss to beat)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=p, y=r2, mode="lines+markers", line={"color": SERIES[1]},
                             showlegend=False, name="held-out R²"), row=2, col=1)
    for r, y in ((1, va[best]), (2, r2[best])):
        fig.add_trace(go.Scatter(x=[p[best]], y=[y], mode="markers", showlegend=r == 1,
                                 name="best pass (kept)",
                                 marker={"symbol": "star", "size": 14, "color": SERIES[2],
                                         "line": {"color": _INK, "width": 1}}), row=r, col=1)
    fig.add_hline(y=0, line={"color": _MUTED, "width": 1}, row=2, col=1)  # after the traces
    _base(fig, title=f"Figure 3 · Training progress: pass {len(p)} of at most {epochs}",
          height=560)
    fig.update_xaxes(range=[0.5, max(p) + 0.5], dtick=1)
    fig.update_xaxes(title_text="pass (epoch)", row=2, col=1)
    fig.update_yaxes(title_text="loss (typical moves squared)", row=1, col=1)
    fig.update_yaxes(title_text="R² vs no change", row=2, col=1)
    return below(fig, -0.22)


def endpoint_scatter(pred: np.ndarray, act: np.ndarray, labels, *, title: str) -> go.Figure:
    """Figure 7: predicted vs actual move to the end of the horizon, one dot per stock-day (the
    stock in the hover). Dots in the top-right and bottom-left quarters got up/down right; the
    dashed diagonal is a perfect forecast.

    Args:
        pred: Predicted end moves (typical moves, already scaled).
        act: Actual end moves (typical moves).
        labels: The stock of each dot.
        title: The figure's title.

    Returns:
        The figure.
    """
    pred, act = np.asarray(pred, float), np.asarray(act, float)
    right = np.sign(pred) == np.sign(act)
    fig = go.Figure()
    for ok, name, color in ((True, "up/down right", SERIES[1]), (False, "up/down wrong",
                                                                  SERIES[5])):
        m = right == ok
        fig.add_trace(go.Scatter(x=pred[m], y=act[m], mode="markers", text=np.asarray(labels)[m],
                                 name=f"{name} ({int(m.sum())})",
                                 marker={"size": 7, "color": color, "opacity": 0.75},
                                 hovertemplate="%{text}<br>predicted %{x:.2f}<br>actual %{y:.2f}"
                                               "<extra></extra>"))
    lim = float(np.nanmax(np.abs(np.r_[pred, act]))) * 1.05 if len(pred) else 1.0
    fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines", name="perfect forecast",
                             line={"color": _MUTED, "width": 1, "dash": "dash"}))
    fig.add_hline(y=0, line={"color": _GRID, "width": 1})  # after the traces
    fig.add_vline(x=0, line={"color": _GRID, "width": 1})
    _base(fig, title=title, height=520)
    fig.update_xaxes(title_text="predicted end move (typical moves)", range=[-lim, lim])
    fig.update_yaxes(title_text="actual end move (typical moves)", range=[-lim, lim])
    return below(fig, -0.14)


def inputs_picture(d: dict) -> go.Figure:
    """Figure 1: every per-minute input of one stock-day. Top: the day's price (1-minute closes)
    with the input (blue) and output (green) windows. Below, one panel per input over the input
    window; the path panel also shows the output window's actual minutes (the truth, green).

    Args:
        d: ``name``, ``day``, ``cell`` ((1, offset, input, output) minutes), ``times`` (the
            session's 390 one-minute close times, New York wall clock), ``price``, ``bars``
            (``{label: values over the input window}``, the path first), ``truth`` (the output
            window's actual moves).

    Returns:
        The figure.
    """
    _, off, L, H = d["cell"]
    D = off + L
    t = pd.DatetimeIndex(d["times"])
    rows = 1 + len(d["bars"])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[2] + [1] * (rows - 1),
                        subplot_titles=["1a · price"] + [f"1{chr(98 + i)} · {k}"
                                                         for i, k in enumerate(d["bars"])])
    fig.update_annotations(x=0, xanchor="left", font={"size": 11})
    fig.add_trace(go.Scatter(x=_wall(t), y=d["price"], mode="lines", name="price",
                             line={"color": _INK, "width": 1.2}, showlegend=False), row=1, col=1)
    x_in, x_out = _wall(t[off:D]), _wall(t[D:D + H])
    for r, (label, vals) in enumerate(d["bars"].items(), start=2):
        fig.add_trace(go.Scatter(x=x_in, y=vals, mode="lines", name=label, showlegend=False,
                                 line={"color": SERIES[0], "width": 1}), row=r, col=1)
        if r == 2:  # the path: the truth after the decision
            fig.add_trace(go.Scatter(x=x_out, y=d["truth"], mode="lines", showlegend=False,
                                     name="truth", line={"color": SERIES[1], "width": 1}),
                          row=r, col=1)
    for a, b, color in ((off, D, "rgba(42,120,214,0.15)"), (D, D + H, "rgba(27,175,122,0.18)")):
        for r in range(1, rows + 1):  # after the traces: empty rows drop shapes
            fig.add_vrect(x0=_wall([t[a] - pd.Timedelta(minutes=1)])[0], x1=_wall([t[b - 1]])[0],
                          fillcolor=color, line_width=0, layer="below", row=r, col=1)
    _base(fig, title=f"Figure 1 · {d['name']} on {d['day']}: the per-minute inputs (blue: the "
          f"{L} minutes read; green: the {H} minutes to predict)", height=180 + 95 * rows)
    fig.update_xaxes(type="date", range=list(_wall([t[0] - pd.Timedelta(minutes=1), t[-1]])))
    return fig


def day_inputs_picture(values: dict, *, title: str, limit: float = 4.0) -> go.Figure:
    """Figure 2: the per-day inputs of one stock-day, one bar each, on the axis they are clipped
    to (typical units; "–" = no value that day).

    Args:
        values: ``{label: value or NaN}``.
        title: The figure's title.
        limit: The clip.

    Returns:
        The figure.
    """
    labels = list(values)
    vals = [values[k] for k in labels]
    fig = go.Figure(go.Bar(x=[0 if not np.isfinite(v) else v for v in vals], y=labels,
                           orientation="h", marker={"color": SERIES[0]}, showlegend=False,
                           text=["–" if not np.isfinite(v) else f"{v:+.2f}" for v in vals],
                           textposition="outside"))
    fig.add_vline(x=0, line={"color": _MUTED, "width": 1})
    _base(fig, title=title, height=120 + 34 * len(labels))
    fig.update_layout(bargap=0.4, margin={"l": 230})
    fig.update_xaxes(range=[-limit, limit], title_text=f"typical units (clipped at ±{limit:g})")
    fig.update_yaxes(autorange="reversed")
    return fig


def lead_curves(res: pd.DataFrame, *, metric: str = "r2_h", title: str = "") -> go.Figure:
    """How far ahead the predicted path agrees with the actual one: per configuration measured
    (grey), the grade at every lead time against the minutes ahead (log scale); the best
    configuration of each granularity in colour. 0 is "no change".

    Args:
        res: ``explorer.search_paths`` results (``r2_h`` / ``corr_h`` per row).
        metric: ``"r2_h"`` (R² against "no change") or ``"corr_h"`` (direction: the
            correlation of predicted and actual change).
        title: The figure title.

    Returns:
        The figure.
    """
    fig = go.Figure()
    for k, (_, r) in enumerate(res.iterrows()):
        fig.add_trace(go.Scatter(x=r["granularity"] * np.arange(1, r["prediction"] + 1),
                                 y=r[metric], mode="lines", line={"color": _GRID, "width": 1},
                                 name="one configuration", legendgroup="all",
                                 showlegend=k == 0, hoverinfo="skip"))
    rank = [c for c in ("r2", "z") if c in res]       # ties at R² = 0 broken by direction
    top = res.sort_values(rank, ascending=False).drop_duplicates("granularity").sort_values(
        "granularity")
    for i, (_, r) in enumerate(top.iterrows()):
        fig.add_trace(go.Scatter(
            x=r["granularity"] * np.arange(1, r["prediction"] + 1), y=r[metric], mode="lines",
            line={"color": SERIES[i % len(SERIES)], "width": 2.5},
            name=f"{r['granularity']:.0f} min (input {r['input']:.0f}, predict "
                 f"{r['prediction']:.0f})",
            hovertemplate="%{x:.0f} min ahead: %{y:+.4f}<extra></extra>"))
    fig.add_hline(y=0, line={"color": _INK, "width": 1})
    words = {"r2_h": "R² against 'no change'", "corr_h": "direction (correlation)"}
    _base(fig, title=title or "Agreement by lead time: the best configuration of each "
          "granularity (VALIDATION)", height=460)
    fig.update_xaxes(type="log", title_text="minutes ahead (trading time, log scale)")
    fig.update_yaxes(title_text=words.get(metric, metric),
                     tickformat="+.4f" if metric == "r2_h" else "+.3f")
    return below(fig, -0.18)


def param_effects(res: pd.DataFrame, params: dict, *, metric: str = "r2") -> go.Figure:
    """How each hyperparameter moves the grade on its own: one panel per hyperparameter, every
    configuration measured (grey dots), the best (solid) and the average (dotted) of each step.
    For ``z`` the grey band is where ONE configuration lands 95% of the time by luck and the dashed
    line is the bar the BEST of a step's tries must clear (:func:`_best_of_bar`); for R² the
    black line at 0 is "no change".

    Args:
        res: ``explorer.search_paths`` results.
        params: ``{column: (title, steps in order)}``.
        metric: ``"r2"``, ``"z"`` (direction against luck) or ``"corr"`` (direction).

    Returns:
        The figure.
    """
    keys = list(params)
    fig = make_subplots(rows=1, cols=len(keys), shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=[params[k][0] for k in keys])
    for col, k in enumerate(keys, start=1):
        have = set(res[k])
        steps = [v for v in params[k][1] if v in have]
        labels = [str(v) for v in steps]
        g = res.groupby(k)
        first = col == 1
        if metric == "z":
            n = g.size().reindex(steps).to_numpy()
            fig.add_trace(go.Scatter(x=labels + labels[::-1],
                                     y=[1.96] * len(steps) + [-1.96] * len(steps),
                                     fill="toself", mode="lines",
                                     fillcolor="rgba(137,135,129,0.18)", line={"width": 0},
                                     name="luck, one config (95%)", showlegend=first,
                                     legendgroup="luck", hoverinfo="skip"), row=1, col=col)
            fig.add_trace(go.Scatter(x=labels, y=[_best_of_bar(c) for c in n],
                                     mode="lines+markers", name="luck, best of k",
                                     line={"color": _MUTED, "width": 1.5, "dash": "dash"},
                                     marker={"size": 5}, showlegend=first, legendgroup="bar"),
                          row=1, col=col)
        fig.add_trace(go.Scatter(x=res[k].astype(str), y=res[metric], mode="markers",
                                 marker={"color": _MUTED, "size": 7, "opacity": 0.6},
                                 name="one configuration", showlegend=first, legendgroup="one"),
                      row=1, col=col)
        for stat, label, line in (("max", "best", {"color": SERIES[0], "width": 2.5}),
                                  ("mean", "average",
                                   {"color": SERIES[0], "width": 1.5, "dash": "dot"})):
            fig.add_trace(go.Scatter(x=labels, y=g[metric].agg(stat).reindex(steps),
                                     mode="lines+markers", line=line, name=label,
                                     showlegend=first, legendgroup=label), row=1, col=col)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=labels,
                         row=1, col=col)
    words = {"r2": "R² against 'no change'", "z": "direction: z against luck",
             "corr": "direction (correlation)"}
    _base(fig, title=f"How each hyperparameter moves the grade: {words.get(metric, metric)} "
          "(VALIDATION)", height=460)
    fig.update_layout(margin={"t": 85})
    fig.add_hline(y=0, line={"color": _INK, "width": 1})
    fig.update_yaxes(title_text=words.get(metric, metric), row=1, col=1)
    fig.update_yaxes(tickformat={"r2": "+.4f", "corr": "+.3f"}.get(metric, "+.1f"))
    return below(fig, -0.16)


def granularity_maps(res: pd.DataFrame, grans, inputs, outputs, *,
                     metric: str = "r2", unit: str = "bars") -> go.Figure:
    """One small map per granularity: input length up, prediction length across, coloured by
    ``metric`` on one shared scale (blue above 0, red below), each cell with its R² and z; the
    best configuration overall is framed, blank cells were not measured (or cannot be).

    Args:
        res: ``explorer.search_paths`` results.
        grans: The granularities (one map each, in order).
        inputs: The input lengths (rows).
        outputs: The prediction lengths (columns).
        metric: The column coloured.
        unit: What the input and prediction lengths count ("bars" or "min").

    Returns:
        The figure.
    """
    grans = [g for g in grans if g in set(res["granularity"])]
    cols = min(3, len(grans))
    rows = math.ceil(len(grans) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{g}-minute bars" for g in grans],
                        horizontal_spacing=0.07, vertical_spacing=0.16)
    lim = float(np.nanmax(np.abs(res[metric]))) or 1.0
    x_lab, y_lab = [str(o) for o in outputs], [str(i) for i in inputs]
    top = res.sort_values([c for c in (metric, "z") if c in res], ascending=False).iloc[0]
    for n, g in enumerate(grans):
        r, c = n // cols + 1, n % cols + 1
        sub = res[res["granularity"] == g]

        def grid(col: str, sub=sub) -> np.ndarray:
            return sub.pivot_table(index="input", columns="prediction", values=col).reindex(
                index=list(inputs), columns=list(outputs)).to_numpy()

        val, r2, zz = grid(metric), grid("r2"), grid("z")
        txt = [[f"{a:+.4f}<br>z {b:+.1f}" if np.isfinite(a) else "" for a, b in
                zip(ra, rb, strict=True)] for ra, rb in zip(r2, zz, strict=True)]
        fig.add_trace(go.Heatmap(z=val, x=x_lab, y=y_lab, text=txt, texttemplate="%{text}",
                                 coloraxis="coloraxis", xgap=2, ygap=2,
                                 hovertemplate=f"{g}-min bars, input %{{y}}, predict %{{x}}: "
                                 f"{metric} %{{z:+.4f}}<extra></extra>"), row=r, col=c)
        fig.update_xaxes(type="category", title_text=f"predict ({unit})", row=r, col=c)
        fig.update_yaxes(type="category", title_text=f"input ({unit})" if c == 1 else None,
                         row=r, col=c)
        if g == top["granularity"]:
            i, j = list(inputs).index(top["input"]), list(outputs).index(top["prediction"])
            fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5,
                          line={"color": _INK, "width": 3}, row=r, col=c)
    fig.update_layout(coloraxis={"colorscale": DIVERGING, "cmin": -lim, "cmax": lim,
                                 "colorbar": {"title": {"text": metric},
                                              "tickformat": "+.4f" if metric == "r2" else ""}})
    _base(fig, title=f"The search, granularity by granularity ({metric} on VALIDATION)",
          height=60 + 300 * rows)
    return fig


def importance_bars(imp: pd.Series) -> go.Figure:
    """Which hyperparameter moves the grade most (``explorer.importance``: Optuna's fANOVA, the
    share of the grade's variation across the trials each one explains)."""
    x = imp.sort_values()
    fig = go.Figure(go.Bar(x=x.to_numpy(), y=list(x.index), orientation="h",
                           marker={"color": SERIES[0]},
                           hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
    _base(fig, title="Which hyperparameter moves the grade most", height=120 + 50 * len(x))
    fig.update_xaxes(tickformat=".0%", title_text="share of the grade's variation explained")
    return fig


def search_history(res: pd.DataFrame, objective: str = "r2", unit: str = "") -> go.Figure:
    """The optimizer's path: each configuration's grade in the order measured and the best so
    far (the explorer: the first quarter are random draws, then the sampler aims at the good
    region; the intraday grid: every cell in a random order). ``unit`` labels the lengths."""
    x = res.sort_values("trial")
    u = f" {unit}" if unit else ""
    text = (x["granularity"].astype(str) + "-min, input " + x["input"].astype(str) + u
            + ", predict " + x["prediction"].astype(str) + u)
    fig = go.Figure(go.Scatter(
        x=x["trial"], y=x[objective], mode="markers", name="configuration measured",
        marker={"color": SERIES[0], "size": 8}, text=text,
        hovertemplate="trial %{x}: %{text}: %{y:+.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x["trial"], y=x[objective].cummax(), mode="lines",
                             name="best so far", line={"color": _INK, "width": 1.5,
                                                       "shape": "hv"}))
    _base(fig, title="The search, trial by trial", height=340)
    fig.update_xaxes(title_text="trial")
    fig.update_yaxes(title_text=objective)
    return below(fig, -0.25)


# ---------------------------------------------------------------------------------------------
# The intraday study (darts_forecast_intraday.ipynb).
# ---------------------------------------------------------------------------------------------
def day_shape(prof: pd.DataFrame) -> go.Figure:
    """The day's shape: the typical 1-minute move at every minute of the session against the
    day's average (``intraday.profile``, TRAIN): large after the open, quiet at lunch, rising into
    the close. It is the unit every bar size's target and the graded truth are measured in."""
    clock = pd.to_datetime("2000-01-01 " + prof.index.astype(str))
    fig = go.Figure(go.Scatter(x=clock, y=prof.iloc[:, 0], mode="lines",
                               line={"color": SERIES[0], "width": 2},
                               hovertemplate="%{x|%H:%M}: %{y:.2f}× the day's average"
                               "<extra></extra>"))
    fig.add_hline(y=1.0, line={"color": _MUTED, "width": 1, "dash": "dot"})
    fig.add_annotation(x=1, xref="paper", y=0, yref="y", text="the day's average",  # log10(1)
                       showarrow=False, xanchor="right", yanchor="bottom",
                       font={"color": _MUTED})
    _base(fig, title="The day's shape: the typical 1-minute move by time of day (TRAIN)",
          height=340)
    fig.update_xaxes(tickformat="%H:%M", title_text="minute of the session (New York)")
    fig.update_yaxes(type="log", title_text="× the day's average (log scale)")
    return fig


def stock_day(d: dict) -> go.Figure:
    """One stock-day series as the model reads it at one decision (``intraday.day_picture``).
    Top: the bar closes from the previous session's end (grey: read as input only) to today's
    close, the input window (blue band), the decision, the predicted span (green band) and the
    graded checkpoints. Bottom: the inputs known in advance that tell the model when each bar is
    from: the "today" flag and the time of day. The night between the sessions is cut out of the
    axis."""
    g, Lm, Sm = d["cell"]
    t = pd.DatetimeIndex(d["times"])
    x = _wall(t)
    opens = _wall(t - pd.Timedelta(minutes=g))         # a band of bars starts at its first open
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=x, y=d["price"], mode="lines+markers", name=f"{g}-minute closes",
                             line={"color": _INK, "width": 1.2}, marker={"size": 3}),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["today"], mode="lines", name="today flag (0 / 1)",
                             line={"color": SERIES[5], "width": 2, "shape": "hv"}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["tod_frac"], mode="lines", name="time of day (0 = 09:30)",
                             line={"color": SERIES[4], "width": 1.5}), row=2, col=1)
    edge = lambda i: x[min(max(i, 0), len(x) - 1)]  # noqa: E731
    start = lambda i: opens[min(max(i, 0), len(x) - 1)]  # noqa: E731
    for a, b, color, label in ((0, d["first"] - 1, "rgba(137,135,129,0.25)",
                                "the previous session (input only)"),
                               (d["start"], d["decision"], "rgba(42,120,214,0.18)",
                                f"input: {Lm} min = {d['bars'][0]} bars"),
                               (d["decision"], d["end"], "rgba(27,175,122,0.22)",
                                f"predicted: {Sm} min = {d['bars'][1]} bars")):
        if b >= a:
            x0 = edge(a) if a == d["decision"] else start(a)
            fig.add_vrect(x0=x0, x1=edge(b), fillcolor=color, line_width=0, layer="below",
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                     marker={"symbol": "square", "size": 12, "color": color}),
                          row=1, col=1)
    _mark(fig, t[d["decision"]], "decision", row=1, col=1)
    cpt = pd.DatetimeIndex(d["checkpoints"])
    at = np.interp(cpt.asi8, t.asi8, np.asarray(d["price"], float))
    fig.add_trace(go.Scatter(x=_wall(cpt), y=at, mode="markers", name="graded checkpoints",
                             marker={"symbol": "diamond", "size": 9, "color": SERIES[1],
                                     "line": {"color": _INK, "width": 1}}), row=1, col=1)
    _base(fig, title=f"{d['name']}: one stock-day series at {t[d['decision']]:%Y-%m-%d %H:%M} "
          f"({g}-minute bars)", height=580)
    fig.update_xaxes(type="date", rangebreaks=_intraday_breaks(t), range=[opens[0], x[-1]])
    fig.update_yaxes(title_text="price ($)", row=1, col=1)
    fig.update_yaxes(title_text="known in advance", range=[-0.1, 1.1], row=2, col=1)
    return below(fig, -0.16)


def checkpoint_curves(res: pd.DataFrame, *, by: str = "granularity", metric: str = "r2_cp",
                      title: str = "") -> go.Figure:
    """How far ahead the predicted path agrees with the actual one: R² against "no change" at
    every checkpoint (minutes after the decision, log scale), every configuration measured in
    grey, the best of each ``by`` value in colour, and the overall best's two baselines (ridge
    dashed, time-of-day mean dotted; with a ``label`` column, each coloured row's own, in its
    colour: VALIDATION against TEST). 0 is "no change".

    Args:
        res: ``intraday.search`` results (or one row: the worked example); a ``label`` column
            names the coloured lines.
        by: The column whose best rows are coloured.
        metric: ``"r2_cp"`` or ``"corr_cp"`` (direction).
        title: The figure title.

    Returns:
        The figure.
    """
    cps = np.array([10, 20, 30, 60, 90, 120, 180, 240, 300, 360])
    fig = go.Figure()
    for i, (_, r) in enumerate(res.iterrows()):
        y = np.asarray(r[metric], float)
        fig.add_trace(go.Scatter(x=cps, y=y, mode="lines", line={"color": _GRID, "width": 1},
                                 name="one configuration", legendgroup="all",
                                 showlegend=i == 0, hoverinfo="skip"))
    top = res.sort_values(["r2", "z"], ascending=False).drop_duplicates(by).sort_values(by)
    for i, (_, r) in enumerate(top.iterrows()):
        name = r["label"] if "label" in r else (f"{r['granularity']:.0f}-min bars, input "
                                                f"{r['input']:.0f}, predict {r['prediction']:.0f}"
                                                " min")
        fig.add_trace(go.Scatter(
            x=cps, y=np.asarray(r[metric], float), mode="lines+markers",
            line={"color": SERIES[i % len(SERIES)], "width": 2.5}, marker={"size": 5},
            name=name, hovertemplate="%{x:.0f} min ahead: %{y:+.4f}<extra></extra>"))
    if metric == "r2_cp" and "label" in res:        # rows of different parts: own baselines
        for i, (_, r) in enumerate(top.iterrows()):
            for col, dash, label in (("ridge_cp", "dash", "ridge"),
                                     ("seasonal_cp", "dot", "time-of-day")):
                fig.add_trace(go.Scatter(x=cps, y=np.asarray(r[col], float), mode="lines",
                                         name=f"{label} baseline, {r['label']}",
                                         line={"color": SERIES[i % len(SERIES)], "width": 1.2,
                                               "dash": dash}))
    elif metric == "r2_cp":
        best = res.sort_values(["r2", "z"], ascending=False).iloc[0]
        for col, dash, label in (("ridge_cp", "dash", "ridge baseline"),
                                 ("seasonal_cp", "dot", "time-of-day baseline")):
            fig.add_trace(go.Scatter(x=cps, y=np.asarray(best[col], float), mode="lines",
                                     name=f"{label} (predict {best['prediction']:.0f} min)",
                                     line={"color": _INK, "width": 1.5, "dash": dash}))
    fig.add_hline(y=0, line={"color": _INK, "width": 1})
    words = {"r2_cp": "R² against 'no change'", "corr_cp": "direction (correlation)"}
    _base(fig, title=title or "Agreement by checkpoint (VALIDATION)", height=470)
    fig.update_xaxes(type="log", tickvals=cps, title_text="minutes after the decision (log scale)")
    fig.update_yaxes(title_text=words.get(metric, metric),
                     tickformat="+.4f" if metric == "r2_cp" else "+.3f")
    return below(fig, -0.2)


def time_curves(res: pd.DataFrame, *, by: str = "prediction") -> go.Figure:
    """Which decision times the paths are predicted best from: R² against "no change" by decision
    time for the best configuration of each ``by`` value (a long output span is graded only where
    it ends by 16:00)."""
    top = res.sort_values(["r2", "z"], ascending=False).drop_duplicates(by).sort_values(by)
    fig = go.Figure()
    for i, (_, r) in enumerate(top.iterrows()):
        t = sorted(r["r2_time"])
        clock = [f"{(570 + m) // 60}:{(570 + m) % 60:02d}" for m in t]
        fig.add_trace(go.Scatter(x=clock, y=[r["r2_time"][m] for m in t], mode="lines+markers",
                                 line={"color": SERIES[i % len(SERIES)], "width": 2},
                                 name=f"predict {r['prediction']:.0f} min ({r['granularity']:.0f}"
                                      f"-min bars, input {r['input']:.0f})"))
    fig.add_hline(y=0, line={"color": _INK, "width": 1})
    _base(fig, title="Agreement by decision time: the best configuration of each output span "
          "(VALIDATION)", height=430)
    fig.update_xaxes(type="category", title_text="decision time (New York)")
    fig.update_yaxes(title_text="R² against 'no change'", tickformat="+.4f")
    return below(fig, -0.22)


def planted_bars(tab: pd.DataFrame) -> go.Figure:
    """The positive control (``intraday.planted_table``): per planted effect, the R² that was
    there to find (the oracle) against what the network and the ridge recovered; the network's
    direction z is written on its bar."""
    x = [f"{v:.2f}" for v in tab.index]
    fig = go.Figure()
    for col, color, label in (("oracle R²", _MUTED, "planted (the most there is)"),
                              ("network R²", SERIES[0], "the network recovered"),
                              ("ridge R²", SERIES[2], "the ridge recovered")):
        text = [f"z {z:+.1f}" for z in tab["network z"]] if col == "network R²" else None
        fig.add_trace(go.Bar(x=x, y=tab[col], name=label, marker={"color": color}, text=text,
                             textposition="outside"))
    fig.add_hline(y=0, line={"color": _INK, "width": 1})
    _base(fig, title="The positive control: a planted 30-minute momentum, found or not "
          "(VALIDATION)", height=400)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="planted correlation with the 30-minute move")
    fig.update_yaxes(title_text="R² against 'no change'", tickformat="+.3f")
    return below(fig, -0.25)


def atlas_map(at: dict) -> go.Figure:
    """Which part of the day predicts which (``intraday.atlas``): the night and the thirteen
    half-hours, the earlier part down, the later part across, coloured by how many luck-SDs their
    correlation sits from a day-shuffled centre (cells with the later part not after the earlier
    one are blank), each cell with its correlation."""
    lab = at["labels"]
    n = len(lab)
    z = np.where(np.arange(n)[None, :] > np.arange(n)[:, None], at["z"], np.nan)
    c = np.where(np.isfinite(z), at["corr"], np.nan)
    txt = [[f"{c[i, j]:+.2f}" if np.isfinite(c[i, j]) else "" for j in range(n)]
           for i in range(n)]
    lim = max(3.0, float(np.nanmax(np.abs(z))))
    fig = go.Figure(go.Heatmap(z=z, x=lab, y=lab, text=txt, texttemplate="%{text}",
                               colorscale=DIVERGING, zmin=-lim, zmax=lim, xgap=1, ygap=1,
                               colorbar={"title": {"text": "z vs luck"}},
                               hovertemplate="%{y} → %{x}: corr %{text}, z %{z:+.1f}"
                               "<extra></extra>"))
    _base(fig, title=f"Which part of the day predicts which ({at['days']} sessions × "
          f"{at['stocks']} stocks; half-hours named by their start)", height=560)
    fig.update_xaxes(type="category", title_text="the later part (what is predicted)")
    fig.update_yaxes(type="category", autorange="reversed",
                     title_text="the earlier part (what predicts)")
    return fig
