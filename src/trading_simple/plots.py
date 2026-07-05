"""All figures in one place: a small, consistent plotly design system.

Every figure shares one look (colors, fonts, grid, hover) via :func:`_base`.
Colors follow the job they do, using a pre-validated colorblind-safe palette:

* **categorical** series take :data:`SERIES` hues in fixed order,
* **polarity** (positive vs negative metric) uses the diverging blue/red pair,
* **win/loss** trades use the reserved status greens/reds,
* chrome (grid, axes, labels) stays recessive gray so data carries the figure.

Figures switch to ``Scattergl`` above :data:`_GL_THRESHOLD` points so intraday
panels stay responsive.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#: Fixed-order categorical palette (colorblind-safe, validated; do not cycle).
SERIES = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834")
#: Diverging poles for signed metrics (blue = positive, red = negative), gray midpoint.
DIVERGING = [(0.0, "#d03b3b"), (0.5, "#f0efec"), (1.0, "#2a78d6")]
_WIN, _LOSS = "#0ca30c", "#d03b3b"          # reserved status colors: profit / loss
_LONG, _SHORT = "#2a78d6", "#eb6834"        # trade direction pair (blue long, orange short)
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"
_GL_THRESHOLD = 10_000
#: Metric/bookkeeping columns of sweep and trial frames; everything else is a parameter.
_METRIC_COLS = {"trial", "total_return", "sharpe", "max_drawdown", "win_rate",
                "pct_profitable", "n_trades"}


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


def _scatter(x, y, **kwargs) -> go.Scatter | go.Scattergl:
    """A Scatter that upgrades itself to WebGL for large series."""
    cls = go.Scattergl if len(x) > _GL_THRESHOLD else go.Scatter
    return cls(x=x, y=y, **kwargs)


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


def plot_sweep(stats: pd.DataFrame, *, metric: str = "sharpe", top: int = 20) -> go.Figure:
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


def plot_heatmap(stats: pd.DataFrame, *, x: str, y: str, metric: str = "sharpe") -> go.Figure:
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
    fig = make_subplots(rows=len(subtitles), cols=1, shared_xaxes=True,
                        vertical_spacing=0.07, subplot_titles=subtitles)
    fig.add_trace(_scatter(curve.index, curve["value"], mode="lines", name="value",
                           showlegend=False, line={"color": SERIES[0], "width": 2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(_scatter(curve.index, curve["return"], mode="lines", name="return",
                           showlegend=False, line={"color": SERIES[1], "width": 2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1%}<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    if has_costs:
        fig.add_trace(_scatter(curve.index, curve["realized_pnl"], mode="lines",
                               name="realized P&L", line={"color": SERIES[3], "width": 2},
                               hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"
                                             "realized $%{y:,.0f}<extra></extra>"),
                      row=3, col=1)
        fig.add_trace(_scatter(curve.index, curve["fees"], mode="lines", name="fees paid",
                               line={"color": SERIES[2], "width": 2},
                               hovertemplate="%{x|%Y-%m-%d %H:%M}<br>"
                                             "fees $%{y:,.0f}<extra></extra>"),
                      row=3, col=1)
        fig.update_yaxes(tickprefix="$", row=3, col=1)
    fig.update_layout(showlegend=has_costs, hovermode="x unified")
    breaks = _intraday_breaks(curve.index)
    if breaks:
        fig.update_xaxes(rangebreaks=breaks)
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

    fig.add_trace(_scatter(close.index, close, mode="lines", name="close",
                           line={"color": _INK, "width": 1.2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>close</extra>"),
                  row=1, col=1)
    for i, (name, series) in enumerate(overlays.get("price", {}).items()):
        fig.add_trace(_scatter(series.index, series, mode="lines", name=name,
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
        for part, color, show in ((series.clip(lower=0), _LONG, True),
                                  (series.clip(upper=0), _LOSS, False)):
            fig.add_trace(_scatter(series.index, part, mode="lines", name=name,
                                   fill="tozeroy", showlegend=show,
                                   line={"color": color, "width": 1},
                                   hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:+.1%}"
                                                 f"<extra>{name}</extra>"),
                          row=row["momentum"], col=1)

    marks = overlays.get("volume_marks")
    spike = (marks.reindex(volume.index).fillna(False).astype(bool)
             if marks is not None else pd.Series(False, index=volume.index))
    fig.add_trace(go.Bar(x=volume.index, y=volume, name="volume", showlegend=False,
                         marker={"color": _MUTED, "opacity": 0.6},
                         hovertemplate="%{y:,.0f}<extra>volume</extra>"),
                  row=row["volume"], col=1)
    if spike.any():
        # Markers, not bars: dense daily bars render sub-pixel thin and would hide the spikes.
        fig.add_trace(go.Scatter(x=volume.index[spike], y=volume[spike], mode="markers",
                                 name="volume spike",
                                 marker={"color": SERIES[2], "size": 7,
                                         "line": {"color": "white", "width": 1}},
                                 hovertemplate="%{y:,.0f}<extra>volume spike</extra>"),
                      row=row["volume"], col=1)
    vol_colors = (SERIES[4], SERIES[2], SERIES[6])  # amber slot 2 matches the spike markers
    for i, (name, series) in enumerate(overlays.get("volume", {}).items()):
        fig.add_trace(_scatter(series.index, series, mode="lines", name=name,
                               line={"color": vol_colors[i % len(vol_colors)], "width": 1.5,
                                     "dash": "dash" if i else "solid"}),
                      row=row["volume"], col=1)

    if equity is not None:
        fig.add_trace(_scatter(equity.index, equity, mode="lines", name="equity",
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
    breaks = _intraday_breaks(close.index)
    if breaks:
        fig.update_xaxes(rangebreaks=breaks)
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
                *, picked: int | None = None) -> go.Figure:
    """The Sharpe-vs-return Pareto front of an optimization study.

    Completed trials draw as a recessive gray cloud; the front as a blue
    staircase (the exact boundary of the dominated region — a diagonal would
    claim unattained trade-offs), each front trial direct-labeled with its
    number; the picked trial as an orange star. Hovers carry the full
    parameter set, so promising trials can be read straight off the figure.

    Args:
        front: ``optimize.pareto_frame`` output (needs ``sharpe``,
            ``total_return``, ``trial`` columns).
        all_trials: Optional frame of every completed trial (context cloud).
        picked: Trial number to highlight (from ``optimize.pick_best``).

    Returns:
        The figure.
    """
    hover = ("<b>trial %{customdata[0]}</b><br>sharpe %{x:.2f} | return %{y:.1%}"
             "<br>%{customdata[1]}<extra></extra>")
    fig = go.Figure()
    if all_trials is not None and len(all_trials):
        fig.add_trace(go.Scatter(
            x=all_trials["sharpe"], y=all_trials["total_return"], mode="markers",
            name=f"all {len(all_trials)} trials",
            marker={"color": _MUTED, "size": 5, "opacity": 0.35},
            hovertemplate=hover, customdata=_trial_hover(all_trials)))
    front = front.sort_values("sharpe")
    # Labels dodge their left neighbor: same-return steps get labeled above, not beside.
    ys = front["total_return"].to_numpy()
    span = max(ys.max() - ys.min(), 1e-9) if len(ys) else 1.0
    positions = ["middle left"] + ["top center" if abs(b - a) / span < 0.08 else "middle left"
                                   for a, b in zip(ys, ys[1:], strict=False)]
    fig.add_trace(go.Scatter(
        x=front["sharpe"], y=front["total_return"], mode="lines+markers+text",
        name="Pareto front", line={"color": SERIES[0], "width": 2, "shape": "vh"},
        marker={"size": 10, "color": SERIES[0], "line": {"color": "white", "width": 1}},
        text=[f"trial {t} " for t in front["trial"]], textposition=positions,
        textfont={"size": 11, "color": _INK},
        hovertemplate=hover, customdata=_trial_hover(front)))
    if picked is not None and picked in set(front["trial"]):
        row = front[front["trial"] == picked].iloc[0]
        fig.add_trace(go.Scatter(x=[row["sharpe"]], y=[row["total_return"]], mode="markers",
                                 name=f"picked: trial {picked}",
                                 marker={"symbol": "star", "size": 17, "color": SERIES[7],
                                         "line": {"color": "white", "width": 1}},
                                 hovertemplate=hover, customdata=_trial_hover(front[
                                     front["trial"] == picked])))
    fig.update_xaxes(title="Sharpe ratio")
    fig.update_yaxes(title="Total return", tickformat=".0%")
    return _base(fig, title="Optimization Pareto front — Sharpe vs total return")


def plot_heatmap_grid(stats: pd.DataFrame, *, pairs: list[tuple[str, str]],
                      metrics: tuple[str, ...] = ("sharpe", "total_return")) -> go.Figure:
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


def plot_coverage(cov: pd.DataFrame) -> go.Figure:
    """Stored bars per symbol, split into regular vs extended session.

    Args:
        cov: ``data.coverage`` output for a universe (per-symbol rows).

    Returns:
        The figure.
    """
    fig = go.Figure([
        go.Bar(x=cov["symbol"], y=cov["regular"], name="regular hours",
               marker={"color": SERIES[0]}),
        go.Bar(x=cov["symbol"], y=cov["extended"], name="extended hours",
               marker={"color": SERIES[2]}),
    ])
    fig.update_layout(barmode="stack", bargap=0.25)
    fig.update_yaxes(title="bars stored")
    return _base(fig, title="Price coverage per symbol (regular vs extended session)")
