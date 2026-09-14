"""All figures in one place: a small, consistent plotly design system.

Every figure shares one look (colors, fonts, grid, hover) via :func:`_base`.
Colors follow the job they do, using a pre-validated colorblind-safe palette:

* **categorical** series take :data:`SERIES` hues in fixed order,
* **polarity** (positive vs negative metric) uses the diverging blue/red pair,
* **win/loss** trades use the reserved status greens/reds,
* chrome (grid, axes, labels) stays recessive gray so data carries the figure.

Long series are thinned to a min/max envelope of :data:`MAX_POINTS` points and
their time axis is packed as epoch milliseconds, which is what keeps an executed
1-minute notebook at single-digit megabytes instead of ~80. Set ``MAX_POINTS =
None`` to draw every bar; figures then switch to ``Scattergl`` above
:data:`_GL_THRESHOLD` points so they stay responsive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_simple import backtest

#: Fixed-order categorical palette (colorblind-safe, validated; do not cycle).
SERIES = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834")
#: Diverging poles for signed metrics (blue = positive, red = negative), gray midpoint.
DIVERGING = [(0.0, "#d03b3b"), (0.5, "#f0efec"), (1.0, "#2a78d6")]
_WIN, _LOSS = "#0ca30c", "#d03b3b"          # reserved status colors: profit / loss
_LONG, _SHORT = "#2a78d6", "#eb6834"        # trade direction pair (blue long, orange short)
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"
_GL_THRESHOLD = 10_000
#: Point budget per trace, split into min/max buckets; ``None`` draws every bar.
MAX_POINTS: int | None = 10_000
#: Metric/bookkeeping columns of sweep and trial frames; everything else is a parameter.
_METRIC_COLS = backtest.METRIC_COLS | {"trial"}
#: Metrics measured against the buy-and-hold benchmark, so zero is a meaningful line.
_RELATIVE = {"excess_cagr", "info_ratio"}
#: Rate metrics that read as percentages (besides every ``*return`` column).
_PERCENT = {"cagr", "excess_cagr", "tracking_error", "max_drawdown"}


def _base(fig: go.Figure, *, title: str, height: int = 420) -> go.Figure:
    """Apply the shared layout: white surface, hairline grid, unified hover."""
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": _INK}},
        template="plotly_white", height=height,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif',
              "size": 12, "color": _INK},
        margin={"l": 60, "r": 20, "t": 50, "b": 40},
        hoverlabel={"bgcolor": "white", "bordercolor": _GRID, "font": {"size": 12}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID,
                     tickfont={"color": _MUTED})
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_GRID,
                     tickfont={"color": _MUTED})
    return fig


def _keep(*series: pd.Series) -> pd.Index | None:
    """Row positions of a min/max envelope shared by every given series.

    Splits the index into equal buckets and keeps, per bucket and per series,
    the position of the minimum and of the maximum, plus the first and last row.
    Extremes survive by construction, so no spike, wick or drawdown low can be
    hidden — which plain ``[::k]`` striding cannot promise. Series passed
    together come back on one x grid (``hovermode="x unified"`` needs that), and
    the bucket count is split among them so their union still fits the budget.

    NaN-safe by filling ``±inf`` rather than using ``nanargmin``: the panel index
    is a symbol union, so a series genuinely carries gaps, and an all-NaN bucket
    would otherwise raise.

    Args:
        *series: Equal-length series sharing one index.

    Returns:
        Sorted positions to keep, or ``None`` when nothing needs dropping
        (:data:`MAX_POINTS` is ``None``, or the series already fit).
    """
    n = len(series[0])
    if MAX_POINTS is None or n <= MAX_POINTS:
        return None
    label = pd.RangeIndex(n) * max(1, MAX_POINTS // (2 * len(series))) // n
    keep = {0, n - 1}
    for s in series:
        v = pd.Series(s.to_numpy(dtype="float64"))
        keep |= set(v.fillna(float("inf")).groupby(label).idxmin())
        keep |= set(v.fillna(float("-inf")).groupby(label).idxmax())
    return pd.Index(sorted(keep))


def _thin(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Reduce a long series/frame to the min/max envelope of :func:`_keep`.

    Args:
        data: A series, or a frame whose columns are all plotted.

    Returns:
        The reduced object, or the input itself when no reduction applies.
    """
    cols = [data[c] for c in data.columns] if isinstance(data, pd.DataFrame) else [data]
    keep = _keep(*cols)
    return data if keep is None else data.iloc[keep]


def _epoch(index: pd.Index):
    """A datetime index as float64 epoch milliseconds.

    Plotly base64-packs float arrays but writes a datetime index out as ISO text,
    once per trace — ~29 bytes a point against ~11, and that repetition is most of
    what made an executed 1-minute notebook ~80 MB. Pair with ``xaxis.type="date"``
    to keep the axis, its ticks and the ``%{x|...}`` hovers unchanged. Dropping the
    timezone changes nothing on screen: plotly.js parses no UTC offset and already
    renders these timestamps as a naive wall clock.

    Args:
        index: A trace's x values.

    Returns:
        Epoch milliseconds for a ``DatetimeIndex``, else ``index`` untouched.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return index
    naive = index.tz_localize(None) if index.tz is not None else index
    return naive.to_numpy("datetime64[ms]").astype("float64")


def _scatter(y: pd.Series, **kwargs) -> go.Scatter | go.Scattergl:
    """A thinned Scatter on a packed date axis, upgrading to WebGL for large series.

    Args:
        y: The series to draw; its index supplies x.
        **kwargs: Passed through to the trace constructor.

    Returns:
        The trace.
    """
    y = _thin(y)
    cls = go.Scattergl if len(y) > _GL_THRESHOLD else go.Scatter
    return cls(x=_epoch(y.index), y=y, **kwargs)


def _intraday_breaks(index: pd.DatetimeIndex) -> list[dict] | None:
    """Axis rangebreaks hiding the gaps between trading sessions.

    Without them an intraday time axis draws a bridge line from each close to
    the next open, and overnight/weekend gaps dominate the picture. One break
    per gap observed in the data, which covers overnights, weekends and
    holidays alike (all in values+dvalue form: mixing a pattern break with a
    values break crashes plotly.js). Returns None for daily series — and for
    series above the WebGL threshold, where Scattergl ignores rangebreaks.
    """
    tod = index - index.normalize()
    if tod.nunique() <= 1 or len(index) > _GL_THRESHOLD:
        return None
    diffs = pd.Series(index).diff()
    step = diffs.median()
    return [{"values": [str(index[i - 1] + step)],
             "dvalue": int((gap - step).total_seconds() * 1000)}
            for i, gap in diffs[diffs > step].items()] or None


def plot_sweep(stats: pd.DataFrame, *, metric: str = "excess_cagr", top: int = 20) -> go.Figure:
    """Ranked horizontal bars of a sweep's best parameter combos.

    Bar color encodes the metric's sign (blue positive, red negative); the
    hover carries the full parameter set and the breadth guard
    (``pct_profitable``).

    Args:
        stats: Output of ``backtest.sweep``.
        metric: Column to rank by.
        top: Number of combos shown.

    Returns:
        The figure.
    """
    param_cols = [c for c in stats.columns if c not in _METRIC_COLS]
    best = stats.dropna(subset=[metric]).nlargest(top, metric).iloc[::-1]
    labels = best[param_cols].astype(str).agg(" ".join, axis=1)
    hover = ("<b>%{customdata[0]}</b><br>" + metric + ": %{x:.3f}<br>"
             "return: %{customdata[1]:.1%} | profitable names: %{customdata[2]:.0%}"
             " | trades: %{customdata[3]}<extra></extra>")
    fig = go.Figure(go.Bar(
        x=best[metric], y=labels, orientation="h",
        marker={"color": ["#2a78d6" if v >= 0 else "#d03b3b" for v in best[metric]]},
        customdata=best[[*param_cols[:1], "total_return", "pct_profitable", "n_trades"]]
                   .assign(label=labels)[["label", "total_return", "pct_profitable", "n_trades"]],
        hovertemplate=hover,
    ))
    fig.update_yaxes(title=f"params ({', '.join(param_cols)})", tickfont={"size": 10})
    fig.update_xaxes(title=metric)
    return _base(fig, title=f"Top {len(best)} parameter combos by {metric}",
                 height=max(300, 24 * len(best) + 120))


def plot_heatmap(stats: pd.DataFrame, *, x: str, y: str, metric: str = "excess_cagr") -> go.Figure:
    """Mean ``metric`` over two parameters (marginalizing the others).

    Uses the diverging blue/red scale centered at zero, so sign is readable
    at a glance.

    Args:
        stats: Output of ``backtest.sweep``.
        x: Parameter for the x axis.
        y: Parameter for the y axis.
        metric: Metric to average.

    Returns:
        The figure.
    """
    grid = stats.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=grid.columns.astype(str), y=grid.index.astype(str),
        colorscale=DIVERGING, zmid=0.0, colorbar={"title": metric},
        texttemplate="%{z:.2f}", textfont={"size": 11},
        hovertemplate=f"{x}=%{{x}} {y}=%{{y}}<br>{metric}=%{{z:.3f}}<extra></extra>",
    ))
    fig.update_xaxes(title=x)
    fig.update_yaxes(title=y)
    return _base(fig, title=f"Mean {metric} by {x} and {y}")


def plot_equity(curve: pd.DataFrame, *, title: str = "Portfolio equity") -> go.Figure:
    """Stacked panels: portfolio value ($), cumulative return (%), and — when the
    curve carries them — the money split into fees paid and realized P&L ($).

    The third panel answers "where did the money go?": cumulative **fees paid**
    (amber, the total spent on commission+slippage) against cumulative
    **realized P&L** (green, the profit/loss booked as positions close, net of
    their own fees). Both final totals are stamped into the panel title. The
    panel is omitted for curves without the ``fees``/``realized_pnl`` columns.

    Args:
        curve: Output of ``backtest.equity_curve`` (``value``, ``return`` and,
            optionally, ``fees`` + ``realized_pnl``).
        title: Figure title.

    Returns:
        The figure.
    """
    has_costs = {"fees", "realized_pnl"} <= set(curve.columns)
    subtitles = ["Value ($)", "Cumulative return"]
    if has_costs:
        won, spent = float(curve["realized_pnl"].iloc[-1]), float(curve["fees"].iloc[-1])
        won_str = f"-${abs(won):,.0f}" if won < 0 else f"${won:,.0f}"
        subtitles.append(f"Realized P&L ({won_str}) vs fees paid (${spent:,.0f})")
    # Read the session gaps off the true index (the step is its median bar spacing),
    # then thin the whole frame at once so every trace shares one x grid — which is
    # what keeps the unified hover below showing all of them together.
    breaks = _intraday_breaks(curve.index)
    curve = _thin(curve)
    fig = make_subplots(rows=len(subtitles), cols=1, shared_xaxes=True,
                        vertical_spacing=0.07, subplot_titles=subtitles)
    fig.add_trace(_scatter(curve["value"], mode="lines", name="value",
                           showlegend=False, line={"color": SERIES[0], "width": 2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(_scatter(curve["return"], mode="lines", name="return",
                           showlegend=False, line={"color": SERIES[1], "width": 2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1%}<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    if has_costs:
        fig.add_trace(_scatter(curve["realized_pnl"], mode="lines",
                               name="realized P&L", line={"color": SERIES[3], "width": 2},
                               hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"
                                             "realized $%{y:,.0f}<extra></extra>"),
                      row=3, col=1)
        fig.add_trace(_scatter(curve["fees"], mode="lines", name="fees paid",
                               line={"color": SERIES[2], "width": 2},
                               hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"
                                             "fees $%{y:,.0f}<extra></extra>"),
                      row=3, col=1)
        fig.update_yaxes(tickprefix="$", row=3, col=1)
    fig.update_layout(showlegend=has_costs, hovermode="x unified")
    fig.update_xaxes(type="date", rangebreaks=breaks)
    return _base(fig, title=title, height=660 if has_costs else 520)


def plot_symbol(
    close: pd.Series,
    volume: pd.Series,
    trades: pd.DataFrame,
    *,
    symbol: str,
    overlays: dict | None = None,
    equity: pd.Series | None = None,
) -> go.Figure:
    """Per-symbol chart: price with trades, then momentum/volume/equity panels.

    Trades draw as entry→exit segments colored by outcome (green win, red
    loss) with direction-explicit markers: **▲ long entry** (blue), **▼ short
    entry** (orange), **× exit** (gray). A strategy's ``indicators()`` hook
    supplies the ``overlays``; every key is optional:

    * ``"price"`` — ``{name: series}`` lines on the price panel,
    * ``"momentum"`` — ``{name: series}`` in an own panel, filled blue above
      zero / red below (the up-vs-down direction filter at a glance),
    * ``"volume"`` — ``{name: series}`` lines on the volume panel,
    * ``"volume_marks"`` — boolean series; marked bars (e.g. volume spikes)
      get an amber dot on the volume panel.

    Args:
        close: The symbol's adjusted close.
        volume: The symbol's volume.
        trades: ``backtest.trade_records`` output (all symbols; filtered here).
        symbol: Which symbol to draw.
        overlays: Overlay dict as above.
        equity: Optional per-symbol equity curve for an extra panel.

    Returns:
        The figure.
    """
    overlays = overlays or {}
    momentum = overlays.get("momentum") or {}
    panels = ["price", *(["momentum"] if momentum else []), "volume",
              *(["equity"] if equity is not None else [])]
    row = {name: i + 1 for i, name in enumerate(panels)}
    titles = {"price": "", "momentum": "Momentum — above 0 → long bias, below 0 → short bias",
              "volume": "Volume — amber dots mark the spikes the strategy acts on",
              "equity": "Equity ($)"}
    weights = {"price": 0.46, "momentum": 0.17, "volume": 0.21, "equity": 0.16}
    heights = [weights[p] for p in panels]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[h / sum(heights) for h in heights],
                        subplot_titles=[titles[p] for p in panels])

    fig.add_trace(_scatter(close, mode="lines", name="close",
                           line={"color": _INK, "width": 1.2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>close</extra>"),
                  row=1, col=1)
    for i, (name, series) in enumerate(overlays.get("price", {}).items()):
        fig.add_trace(_scatter(series, mode="lines", name=name,
                               line={"color": SERIES[i % len(SERIES)], "width": 1.5}),
                      row=1, col=1)

    mine = trades[trades["symbol"] == symbol]
    for wins, color, label in ((True, _WIN, "winning trade"), (False, _LOSS, "losing trade")):
        seg = mine[(mine["pnl"] > 0) == wins]
        xs, ys = [], []
        for _, t in seg.iterrows():
            xs += [t["entry_ts"], t["exit_ts"], None]
            ys += [t["entry_price"], t["exit_price"], None]
        if xs:
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=label,
                                     line={"color": color, "width": 2.5}), row=1, col=1)
    for direction, marker, color in (("Long", "triangle-up", _LONG),
                                     ("Short", "triangle-down", _SHORT)):
        seg = mine[mine["direction"] == direction]
        if len(seg):
            fig.add_trace(go.Scatter(
                x=seg["entry_ts"], y=seg["entry_price"], mode="markers",
                name=f"{direction.lower()} entry",
                marker={"symbol": marker, "size": 11, "color": color,
                        "line": {"color": "white", "width": 1}},
                hovertemplate=direction + " entry %{x|%Y-%m-%d %H:%M} @ %{y:.2f}"
                              "<extra></extra>"), row=1, col=1)
    if len(mine):
        fig.add_trace(go.Scatter(
            x=mine["exit_ts"], y=mine["exit_price"], mode="markers", name="exit",
            marker={"symbol": "x", "size": 8, "color": _MUTED},
            customdata=mine[["pnl", "return"]],
            hovertemplate="exit %{x|%Y-%m-%d %H:%M} @ %{y:.2f}<br>"
                          "pnl $%{customdata[0]:,.0f} (%{customdata[1]:+.1%})<extra></extra>"),
            row=1, col=1)

    for name, series in momentum.items():
        # Thin once, before the clip: both halves then land on one x grid, so the
        # positive and negative `tozeroy` fills meet cleanly instead of on two.
        series = _thin(series)
        for part, color, show in ((series.clip(lower=0), _LONG, True),
                                  (series.clip(upper=0), _LOSS, False)):
            fig.add_trace(_scatter(part, mode="lines", name=name,
                                   fill="tozeroy", showlegend=show,
                                   line={"color": color, "width": 1},
                                   hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:+.1%}"
                                                 f"<extra>{name}</extra>"),
                          row=row["momentum"], col=1)

    marks = overlays.get("volume_marks")
    spike = (marks.reindex(volume.index).fillna(False).astype(bool)
             if marks is not None else pd.Series(False, index=volume.index))
    bars = _thin(volume)  # the bars thin; the spikes below stay at full resolution
    fig.add_trace(go.Bar(x=_epoch(bars.index), y=bars, name="volume", showlegend=False,
                         marker={"color": _MUTED, "opacity": 0.6},
                         hovertemplate="%{y:,.0f}<extra>volume</extra>"),
                  row=row["volume"], col=1)
    if spike.any():
        # Markers, not bars: dense daily bars render sub-pixel thin and would hide the spikes.
        fig.add_trace(go.Scatter(x=_epoch(volume.index[spike]), y=volume[spike], mode="markers",
                                 name="volume spike",
                                 marker={"color": SERIES[2], "size": 7,
                                         "line": {"color": "white", "width": 1}},
                                 hovertemplate="%{y:,.0f}<extra>volume spike</extra>"),
                      row=row["volume"], col=1)
    vol_colors = (SERIES[4], SERIES[2], SERIES[6])  # amber slot 2 matches the spike markers
    for i, (name, series) in enumerate(overlays.get("volume", {}).items()):
        fig.add_trace(_scatter(series, mode="lines", name=name,
                               line={"color": vol_colors[i % len(vol_colors)], "width": 1.5,
                                     "dash": "dash" if i else "solid"}),
                      row=row["volume"], col=1)

    if equity is not None:
        fig.add_trace(_scatter(equity, mode="lines", name="equity",
                               showlegend=False, line={"color": SERIES[0], "width": 2}),
                      row=row["equity"], col=1)

    n_win, n_loss = int((mine["pnl"] > 0).sum()), int((mine["pnl"] <= 0).sum())
    fig = _base(fig, title=f"{symbol} — price (adjusted), {len(mine)} trades "
                           f"({n_win} wins, {n_loss} losses)",
                height={1: 420, 2: 560, 3: 760, 4: 900}[len(panels)])
    # Too many series for a horizontal legend row: park it vertically beside the panels.
    fig.update_layout(legend={"orientation": "v", "yanchor": "top", "y": 1.0,
                              "xanchor": "left", "x": 1.005},
                      margin={"r": 175})
    if momentum:
        fig.update_yaxes(tickformat="+.0%", row=row["momentum"], col=1)
    fig.update_xaxes(type="date", rangebreaks=_intraday_breaks(close.index))
    return fig


def _trial_hover(frame: pd.DataFrame) -> pd.DataFrame:
    """Customdata frame for trial scatters: number + a ``k=v`` parameter string."""
    params = [c for c in frame.columns if c not in _METRIC_COLS]
    text = frame[params].apply(lambda r: ", ".join(f"{k}={v}" for k, v in r.items()), axis=1)
    return pd.DataFrame({"trial": frame["trial"], "params": text})


def plot_volume_profile(volume: pd.Series, baseline: pd.Series, tz: str, *, days: int,
                        threshold: float | None = None, symbol: str = "",
                        date=None) -> go.Figure:
    """Validation view of the time-of-day volume baseline for one symbol.

    Overlays, on one exchange-local clock axis: the ``days`` input sessions'
    raw volume (thin gray lines, darker = more recent), the expected-volume
    profile in use on the target session (bold purple — pass the
    ``backtest.volume_profile`` output the strategy actually consumes), the
    ``threshold``× spike trigger (dashed amber), and the target session's own
    volume (gray bars, amber dots where they cross the trigger).

    Args:
        volume: One symbol's volume (UTC index).
        baseline: Matching expected-volume series for that symbol.
        tz: Exchange-local timezone (``panel.tz``).
        days: Number of input sessions the profile averages over.
        threshold: Spike multiple; draws the trigger when given.
        symbol: Name for the title.
        date: Target session (exchange-local date, e.g. ``"2026-06-24"``);
            default: the last stored session.

    Returns:
        The figure.
    """
    volume = volume.dropna()  # the panel index is a symbol union; keep this symbol's bars
    baseline = baseline.reindex(volume.index)
    local = volume.index.tz_convert(tz)
    sessions = pd.unique(local.date)
    target = pd.Timestamp(date).date() if date is not None else sessions[-1]
    inputs = [d for d in sessions if d < target][-days:]
    clock = pd.Timestamp("2000-01-01") + (local - local.normalize())  # shared clock axis
    fig = go.Figure()
    for i, day in enumerate(inputs):
        m = local.date == day
        alpha = 0.2 + 0.35 * (i + 1) / len(inputs)
        fig.add_trace(go.Scatter(
            x=clock[m], y=volume[m], mode="lines", name=f"input {day}",
            line={"color": f"rgba(137, 135, 129, {alpha:.2f})", "width": 1},
            hovertemplate="%{x|%H:%M} — %{y:,.0f}<extra>" + str(day) + "</extra>"))
    m = local.date == target
    fig.add_trace(go.Scatter(
        x=clock[m], y=baseline[m], mode="lines", name="profile used (avg of inputs)",
        line={"color": SERIES[4], "width": 2.5},
        hovertemplate="%{x|%H:%M} — %{y:,.0f}<extra>profile</extra>"))
    if threshold:
        fig.add_trace(go.Scatter(
            x=clock[m], y=threshold * baseline[m], mode="lines",
            name=f"spike trigger ({threshold}x)",
            line={"color": SERIES[2], "width": 1.5, "dash": "dash"},
            hovertemplate="%{x|%H:%M} — %{y:,.0f}<extra>trigger</extra>"))
    fig.add_trace(go.Scatter(x=clock[m], y=volume[m], mode="lines",
                             name=f"{target} volume (target day)",
                             line={"color": SERIES[0], "width": 1.3},
                             hovertemplate="%{x|%H:%M} — %{y:,.0f}<extra>target day</extra>"))
    if threshold:
        spike = volume[m] >= (threshold * baseline[m]).to_numpy()
        if spike.any():
            fig.add_trace(go.Scatter(
                x=clock[m][spike.to_numpy()], y=volume[m][spike.to_numpy()], mode="markers",
                name="volume spike", marker={"color": SERIES[2], "size": 8,
                                             "line": {"color": "white", "width": 1}},
                hovertemplate="%{x|%H:%M} — %{y:,.0f}<extra>volume spike</extra>"))
    fig.update_xaxes(tickformat="%H:%M", title=f"time of day ({tz})")
    # Log scale: the open/close U-shape would otherwise flatten the mid-day
    # detail this figure exists to show; 4x above the profile is a constant gap.
    fig.update_yaxes(title="volume (log)", type="log")
    fig = _base(fig, title=f"{symbol} — {len(inputs)} input sessions vs the "
                           f"averaged profile in use on {target}")
    fig.update_layout(legend={"orientation": "v", "yanchor": "top", "y": 1.0,
                              "xanchor": "left", "x": 1.005},
                      margin={"r": 195})
    return fig


def plot_pareto(front: pd.DataFrame, all_trials: pd.DataFrame | None = None,
                *, picked: int | None = None, axes: tuple[str, str] | None = None) -> go.Figure:
    """The Pareto front of an optimization study, on the study's own two objectives.

    By default tracking error runs across and excess return up — the usual picture of an
    efficient frontier: up is more return, left is less straying from the benchmark.
    Completed trials draw as a recessive gray cloud; the front as a blue
    staircase (the exact boundary of the dominated region — a diagonal would
    claim unattained trade-offs — stepped the right way for each objective's
    direction), each front trial direct-labeled with its number; the picked trial
    as an orange star. Hovers carry the full parameter set, so promising trials
    can be read straight off the figure.

    A dashed zero line marks the benchmark on a benchmark-relative axis: anything below
    (or left of) it **loses to simply owning the universe**, however healthy its absolute
    Sharpe looks — and the pick never lands there (``optimize.MIN_PICK_EXCESS``).

    Args:
        front: ``optimize.pareto_frame`` output (needs the two ``axes`` columns
            plus ``trial``).
        all_trials: Optional frame of every completed trial (context cloud).
        picked: Trial number to highlight (from ``optimize.pick_best``).
        axes: ``(x, y)`` metric columns; defaults to the objectives, second one across.

    Returns:
        The figure.
    """
    from trading_simple.optimize import DIRECTIONS, OBJECTIVES

    x, y = axes or (OBJECTIVES[1], OBJECTIVES[0])
    pct = {m: m.endswith("return") or m in _PERCENT for m in (x, y)}
    hover = (f"<b>trial %{{customdata[0]}}</b><br>{x} %{{x:{'.1%' if pct[x] else '.2f'}}} | "
             f"{y} %{{y:{'.1%' if pct[y] else '.2f'}}}<br>%{{customdata[1]}}<extra></extra>")
    # The dominated region hugs the front from the side its x-axis is worse on: step
    # vertically first when larger x is better, horizontally first when smaller x is.
    shape = "vh" if dict(zip(OBJECTIVES, DIRECTIONS, strict=True)).get(x) != "minimize" else "hv"
    fig = go.Figure()
    if all_trials is not None and len(all_trials):
        fig.add_trace(go.Scatter(
            x=all_trials[x], y=all_trials[y], mode="markers",
            name=f"all {len(all_trials)} trials",
            marker={"color": _MUTED, "size": 5, "opacity": 0.35},
            hovertemplate=hover, customdata=_trial_hover(all_trials)))
    front = front.sort_values(x)
    # Labels dodge their left neighbor: same-y steps get labeled above, not beside.
    ys = front[y].to_numpy()
    span = max(ys.max() - ys.min(), 1e-9) if len(ys) else 1.0
    positions = ["middle left"] + ["top center" if abs(b - a) / span < 0.08 else "middle left"
                                   for a, b in zip(ys, ys[1:], strict=False)]
    fig.add_trace(go.Scatter(
        x=front[x], y=front[y], mode="lines+markers+text",
        name="Pareto front", line={"color": SERIES[0], "width": 2, "shape": shape},
        marker={"size": 10, "color": SERIES[0], "line": {"color": "white", "width": 1}},
        text=[f"trial {t} " for t in front["trial"]], textposition=positions,
        textfont={"size": 11, "color": _INK},
        hovertemplate=hover, customdata=_trial_hover(front)))
    if picked is not None and picked in set(front["trial"]):
        row = front[front["trial"] == picked].iloc[0]
        fig.add_trace(go.Scatter(x=[row[x]], y=[row[y]], mode="markers",
                                 name=f"picked: trial {picked}",
                                 marker={"symbol": "star", "size": 17, "color": SERIES[7],
                                         "line": {"color": "white", "width": 1}},
                                 hovertemplate=hover, customdata=_trial_hover(front[
                                     front["trial"] == picked])))
    if x in _RELATIVE or y in _RELATIVE:  # zero = the buy-and-hold benchmark
        line = {"dash": "dash", "color": _LOSS, "width": 1}
        if x in _RELATIVE:
            fig.add_vline(x=0, line=line, annotation_text="loses to buy & hold →",
                          annotation_position="top left")
        if y in _RELATIVE:
            fig.add_hline(y=0, line=line, annotation_text="loses to buy & hold ↓",
                          annotation_position="bottom right")
    fig.update_xaxes(title=x.replace("_", " "), tickformat=".0%" if pct[x] else None)
    fig.update_yaxes(title=y.replace("_", " "), tickformat=".0%" if pct[y] else None)
    return _base(fig, title=f"Optimization Pareto front — {x} vs {y}")


def plot_heatmap_grid(stats: pd.DataFrame, *, pairs: list[tuple[str, str]],
                      metrics: tuple[str, ...] = ("excess_cagr", "sharpe")) -> go.Figure:
    """Small-multiple heatmaps: one row per metric, one column per parameter pair.

    The multi-objective view of a sweep: the same parameter pairs judged by
    every metric at once, so a region that pays on return but not on Sharpe
    (or breadth) is visible immediately. Each row shares one diverging color
    scale centered at 0 (0.5 for the rate metrics ``win_rate`` /
    ``pct_profitable``).

    Args:
        stats: Output of ``backtest.sweep``.
        pairs: ``[(x_param, y_param), ...]`` — one column of heatmaps per pair.
        metrics: One row of heatmaps per metric.

    Returns:
        The figure.
    """
    rows, cols = len(metrics), len(pairs)
    fig = make_subplots(rows=rows, cols=cols, horizontal_spacing=0.09, vertical_spacing=0.16,
                        subplot_titles=[f"mean {m} — {y} × {x}"
                                        for m in metrics for x, y in pairs])
    for r, metric in enumerate(metrics):
        axis = "coloraxis" if r == 0 else f"coloraxis{r + 1}"
        for c, (x, y) in enumerate(pairs):
            grid = stats.pivot_table(index=y, columns=x, values=metric, aggfunc="mean")
            fig.add_trace(go.Heatmap(
                z=grid.values, x=grid.columns.astype(str), y=grid.index.astype(str),
                coloraxis=axis, texttemplate="%{z:.2f}", textfont={"size": 10},
                hovertemplate=f"{x}=%{{x}} {y}=%{{y}}<br>{metric}=%{{z:.3f}}<extra></extra>"),
                row=r + 1, col=c + 1)
            fig.update_xaxes(title_text=x, title_standoff=2, row=r + 1, col=c + 1)
            fig.update_yaxes(title_text=y, title_standoff=2, row=r + 1, col=c + 1)
        fig.update_layout({axis: {
            "colorscale": DIVERGING,
            "cmid": 0.5 if metric in {"win_rate", "pct_profitable"} else 0.0,
            "colorbar": {"title": metric, "len": 0.85 / rows, "thickness": 12,
                         "y": 1 - (r + 0.5) / rows, "yanchor": "middle"}}})
    fig = _base(fig, title="Sweep response surfaces — every metric × parameter pair",
                height=290 * rows + 90)
    fig.update_layout(margin={"t": 80})
    return fig


def _pad(values: pd.Series, frac: float) -> list[float] | None:
    """An explicit axis range hugging ``values``, padded by ``frac`` of their span.

    Pinning a range is the only way to keep an axis off a shape: plotly grows an
    *autoranged* axis to contain its shapes, so a ``y=0`` reference line drags a
    panel all the way to zero however tight the data is.

    Args:
        values: The points the axis must show.
        frac: Padding on each side, as a fraction of the data's span.

    Returns:
        ``[lo, hi]``, or ``None`` when the span is zero or unusable (no finite
        points, one repeated value) and autorange is the better answer.
    """
    nums = pd.to_numeric(values, errors="coerce")
    finite = nums[nums.abs() != float("inf")].dropna()
    if finite.empty:
        return None
    lo, hi = float(finite.min()), float(finite.max())
    return None if hi == lo else [lo - frac * (hi - lo), hi + frac * (hi - lo)]


def plot_sensitivity(star: pd.DataFrame, *, metric: str = "excess_cagr",
                     cols: int = 3) -> go.Figure:
    """Local robustness of one picked parameter set: a tornado over 1-D profiles.

    Two questions, two panels, because they are answered differently:

    * **Which knobs matter?** The tornado — one bar per parameter, spanning the
      worst and best its own perturbations produced. Long bar = the result hangs
      on that value; short bar = the knob is not load-bearing. A bar is drawn red
      when it reaches ``0``, i.e. when some nearby value gives the benchmark back.
    * **Is the pick on a plateau or a spike?** The profiles — one mini-chart per
      parameter, the metric against that parameter's actual values, with the picked
      value marked. A flat top is a plateau; a lone peak between two drops is the
      overfit case; a value sitting on the shoulder of a drop is a cliff to step
      away from.

    Each profile is scaled to **its own** neighborhood, both axes pinned to the
    points it draws: a knob whose neighborhood spans a tenth of a percent gets a
    tenth-of-a-percent y-axis, so its shape is readable next to a knob that swings
    ten points. The zero line therefore only appears where the neighborhood
    actually reaches the benchmark — which is the case it is there to flag — and
    the shared, comparable view of all knobs at once is the tornado above. (The
    ranges are set explicitly rather than left to autorange because plotly grows
    an autoranged axis to fit its shapes, so the ``y=0`` line alone was stretching
    every panel down to zero and flattening the curve against the top of it.)

    The verdicts and the actions they imply come from
    :func:`optimize.stability_report`, which reads the same frame.

    Args:
        star: Output of :func:`optimize.sensitivity` — the center row plus the
            one-at-a-time perturbations, labelled by ``perturbed``/``offset``.
        metric: Column to judge on; benchmark-relative by default, so ``0`` is the
            meaningful reference line rather than an arbitrary threshold.
        cols: Profile charts per row.

    Returns:
        The figure.
    """
    center = star[star["perturbed"] == ""]
    groups = [(name, g) for name, g in
              star[star["perturbed"] != ""].groupby("perturbed", sort=False)]
    mid = float(center[metric].iloc[0])
    fmt = ".3f" if metric in {"sharpe", "info_ratio", "n_trades", "trades_per_year"} else ".2%"

    rows = 1 + -(-len(groups) // cols)
    specs = ([[{"colspan": cols}] + [None] * (cols - 1)]
             + [[{} for _ in range(cols)] for _ in range(rows - 1)])
    titles = ([f"how far each knob moves {metric} — bar spans its whole neighborhood"]
              + [name for name, _ in groups] + [""] * ((rows - 1) * cols - len(groups)))
    # The tornado grows with the knob count: on a fixed share of the height, plotly drops
    # every other category label as soon as a strategy searches more than a few knobs.
    tornado_h, profile_h = 30 * len(groups) + 70, 200
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles,
                        row_heights=[tornado_h] + [profile_h] * (rows - 1),
                        horizontal_spacing=0.07,
                        vertical_spacing=min(0.11, 0.8 / max(rows - 1, 1)))
    fig.update_annotations(font_size=12)      # the subplot titles; vline labels come later

    # --- the tornado, widest bar on top ---------------------------------------
    span = {name: (min(float(g[metric].min()), mid), max(float(g[metric].max()), mid))
            for name, g in groups}
    order = sorted(span, key=lambda name: span[name][1] - span[name][0])
    lo = [span[name][0] for name in order]
    hi = [span[name][1] for name in order]
    fig.add_trace(go.Bar(
        y=order, x=[h - low for low, h in zip(lo, hi, strict=True)], base=lo, orientation="h",
        # The outline is load-bearing: a knob whose whole neighborhood scores identically
        # has a zero-width bar, and without a stroke it renders as nothing at all — which
        # reads as missing data rather than as the perfectly flat result it is.
        marker={"color": [_LOSS if low <= 0 else SERIES[0] for low in lo],
                "line": {"color": [_LOSS if low <= 0 else SERIES[0] for low in lo],
                         "width": 1.5}},
        customdata=list(zip(lo, hi, strict=True)), showlegend=False,
        hovertemplate=(f"%{{y}}<br>worst %{{customdata[0]:{fmt}}}"
                       f" · best %{{customdata[1]:{fmt}}}<extra></extra>")), row=1, col=1)
    fig.update_xaxes(tickformat=fmt, row=1, col=1)
    fig.update_yaxes(type="category", dtick=1, automargin=True, tickfont={"size": 10},
                     row=1, col=1)

    # --- one profile per parameter --------------------------------------------
    for i, (name, g) in enumerate(groups):
        r, c = 2 + i // cols, 1 + i % cols
        pts = pd.concat([g[[name, metric]], center[[name, metric]]]).sort_values(name)
        numeric = (pd.api.types.is_numeric_dtype(pts[name])
                   and not pd.api.types.is_bool_dtype(pts[name]))
        x = pts[name] if numeric else pts[name].astype(str)
        picked = center[name].iloc[0] if numeric else str(center[name].iloc[0])
        fig.add_trace(go.Scatter(
            x=x, y=pts[metric], mode="lines+markers", showlegend=False,
            line={"color": SERIES[0], "width": 1.5}, marker={"size": 5, "color": SERIES[0]},
            hovertemplate=f"{name}=%{{x}}<br>{metric}=%{{y:{fmt}}}<extra></extra>"), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=[picked], y=[mid], mode="markers", showlegend=False,
            marker={"size": 10, "color": _INK, "symbol": "diamond"},
            hovertemplate=f"picked: {name}=%{{x}}<br>{metric}=%{{y:{fmt}}}<extra></extra>"),
            row=r, col=c)
        # Reference lines go in AFTER their subplot's traces: add_hline/add_vline default
        # to exclude_empty_subplots=True and are dropped SILENTLY on a traceless subplot.
        fig.add_hline(y=0.0, row=r, col=c, line={"color": _GRID, "width": 1}, layer="below")
        fig.update_xaxes(title_text=name, title_standoff=2, title_font={"size": 10},
                         tickfont={"size": 9}, range=_pad(x, 0.06) if numeric else None,
                         row=r, col=c)
        # Percent ticks at 2dp read as one repeated label once the whole neighborhood
        # fits inside 0.5pp, which is exactly the panel this zoom exists for.
        span = float(pts[metric].max() - pts[metric].min())
        fine = {".2%": ".3%", ".3f": ".4f"}[fmt]
        fig.update_yaxes(tickformat=fine if 0 < span < (0.005 if fmt == ".2%" else 0.05) else fmt,
                         tickfont={"size": 9}, range=_pad(pts[metric], 0.12), row=r, col=c)

    fig.add_vline(x=0.0, row=1, col=1, line={"color": _GRID, "width": 1}, layer="below")
    fig.add_vline(x=mid, row=1, col=1, line={"color": _INK, "width": 1, "dash": "dot"},
                  annotation_text="picked", annotation_font_size=10)
    fig = _base(fig, title=f"Neighborhood stability around the picked parameters — {metric}",
                height=tornado_h + profile_h * (rows - 1) + 110)
    fig.update_layout(margin={"t": 80}, bargap=0.45)
    return fig


def plot_coverage(timeline: pd.DataFrame, *, bar: str = "1d", freq: str = "MS") -> go.Figure:
    """When each symbol actually has data — a symbol × time coverage map.

    One cell per (symbol, bucket), shaded by how complete that bucket is
    *relative to the best-covered symbol in the same bucket*. That normalization
    needs no exchange calendar and works for any bar size: a full cell means
    "as many bars as the fullest name that period", a pale cell means a partial
    period, and an empty cell means nothing stored at all.

    This is the figure that answers "do I really have history since 2015?" — a
    total bar count cannot, because a late listing, an early stop and a hole in
    the middle all look identical once the time axis is collapsed. Symbols are
    ordered by first bucket, so genuine IPO/listing starts form a clean staircase
    at the bottom and a failed backfill shows up as a hole that breaks it.

    Args:
        timeline: ``data.coverage_timeline`` output for one bar size.
        bar: Bar size, for the title.
        freq: Pandas offset alias matching the timeline's bucket width
            (``"MS"`` for monthly, ``"D"`` daily, ``"YS"`` yearly).

    Returns:
        The figure.
    """
    z = timeline.pivot(index="symbol", columns="period", values="bars")
    # Reindex to a gapless bucket axis: a period missing for EVERY symbol has no
    # rows at all, and without this the heatmap would silently close the gap up.
    z = z.reindex(columns=pd.date_range(z.columns.min(), z.columns.max(), freq=freq))
    z = z.loc[z.notna().idxmax(axis=1).sort_values(kind="stable").index]  # earliest start first
    pct = z.div(z.max(axis=0), axis=1)                                    # vs the fullest name

    ext = timeline.pivot(index="symbol", columns="period", values="extended")
    ext = ext.reindex(index=z.index, columns=z.columns)
    fig = go.Figure(go.Heatmap(
        z=pct.to_numpy(), x=_epoch(pct.columns), y=pct.index.to_numpy(),
        customdata=np.dstack([z.to_numpy(), ext.to_numpy()]),
        # Floor the ramp at a visible blue: a month holding 1 bar of 21 must not
        # fade into the white that means "nothing stored at all".
        colorscale=[(0.0, "#cfe0f4"), (1.0, SERIES[0])], zmin=0, zmax=1,
        hoverongaps=False,
        colorbar={"title": {"text": "of fullest name<br>(blank = nothing stored)",
                            "side": "top", "font": {"size": 10}},
                  "tickformat": ".0%", "thickness": 12, "len": 0.35, "y": 1,
                  "yanchor": "top", "outlinewidth": 0, "tickfont": {"size": 10}},
        hovertemplate="<b>%{y}</b>  %{x|%b %Y}<br>%{customdata[0]:,.0f} bars "
                      "(%{customdata[1]:,.0f} extended)<br>%{z:.0%} of fullest<extra></extra>",
    ))
    first, last = z.columns.min(), z.columns.max()
    early = int((z.notna().idxmax(axis=1) == first).sum())
    current = int((z.iloc[:, ::-1].notna().idxmax(axis=1) == last).sum())
    fig.update_yaxes(tickfont={"size": 8}, autorange="reversed")
    fig.update_xaxes(type="date", showgrid=False)  # _epoch packs x; type pins the axis back
    fig = _base(fig, title=f"{bar} coverage over time — {early}/{len(z)} symbols reach back to "
                           f"{first:%b %Y}, {current}/{len(z)} run to {last:%b %Y}",
                height=max(420, 70 + 11 * len(z)))
    return fig
