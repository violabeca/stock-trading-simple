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
_INK, _MUTED, _GRID = "#0b0b0b", "#898781", "#e1e0d9"
_GL_THRESHOLD = 10_000


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
    metric_cols = {"total_return", "sharpe", "max_drawdown", "win_rate",
                   "pct_profitable", "n_trades"}
    param_cols = [c for c in stats.columns if c not in metric_cols]
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
    """Two stacked panels: portfolio value ($) and cumulative return (%).

    Args:
        curve: Output of ``backtest.equity_curve`` (``value`` + ``return``).
        title: Figure title.

    Returns:
        The figure.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Value ($)", "Cumulative return"))
    fig.add_trace(_scatter(curve.index, curve["value"], mode="lines", name="value",
                           line={"color": SERIES[0], "width": 2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>$%{y:,.0f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(_scatter(curve.index, curve["return"], mode="lines", name="return",
                           line={"color": SERIES[1], "width": 2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1%}<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(showlegend=False, hovermode="x unified")
    return _base(fig, title=title, height=520)


def plot_symbol(
    close: pd.Series,
    volume: pd.Series,
    trades: pd.DataFrame,
    *,
    symbol: str,
    overlays: dict | None = None,
    equity: pd.Series | None = None,
) -> go.Figure:
    """Per-symbol chart: price with trades, volume, and (optionally) equity.

    Winning trades draw as green entry→exit segments, losing ones red; entries
    are triangles, exits are crosses. A strategy's ``indicators()`` hook
    supplies the ``overlays`` (extra lines on the price and volume panels).

    Args:
        close: The symbol's adjusted close.
        volume: The symbol's volume.
        trades: ``backtest.trade_records`` output (all symbols; filtered here).
        symbol: Which symbol to draw.
        overlays: ``{"price": {name: series}, "volume": {name: series}}``.
        equity: Optional per-symbol equity curve for a third panel.

    Returns:
        The figure.
    """
    rows = 3 if equity is not None else 2
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.55, 0.25, 0.2][:rows],
                        subplot_titles=(f"{symbol} price (adjusted)", "Volume",
                                        "Equity ($)")[:rows])
    fig.add_trace(_scatter(close.index, close, mode="lines", name="close",
                           line={"color": _INK, "width": 1.2},
                           hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra>close</extra>"),
                  row=1, col=1)
    overlays = overlays or {}
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
    if len(mine):
        fig.add_trace(go.Scatter(
            x=mine["entry_ts"], y=mine["entry_price"], mode="markers", name="entry",
            marker={"symbol": "triangle-up", "size": 9, "color": _INK},
            hovertemplate="entry %{x|%Y-%m-%d %H:%M} @ %{y:.2f}<extra></extra>"), row=1, col=1)

    fig.add_trace(go.Bar(x=volume.index, y=volume, name="volume",
                         marker={"color": _GRID}, hovertemplate="%{y:,.0f}<extra>volume</extra>"),
                  row=2, col=1)
    for i, (name, series) in enumerate(overlays.get("volume", {}).items()):
        fig.add_trace(_scatter(series.index, series, mode="lines", name=name,
                               line={"color": SERIES[(i + 2) % len(SERIES)], "width": 1.5,
                                     "dash": "dot" if i else "solid"}),
                      row=2, col=1)
    if equity is not None:
        fig.add_trace(_scatter(equity.index, equity, mode="lines", name="equity",
                               line={"color": SERIES[0], "width": 2}), row=rows, col=1)
    n_win, n_loss = int((mine["pnl"] > 0).sum()), int((mine["pnl"] <= 0).sum())
    return _base(fig, title=f"{symbol} — {len(mine)} trades ({n_win} wins, {n_loss} losses)",
                 height=250 * rows)


def plot_pareto(front: pd.DataFrame, all_trials: pd.DataFrame | None = None,
                *, picked: int | None = None) -> go.Figure:
    """The Sharpe-vs-return Pareto front of an optimization study.

    Non-optimal trials draw as recessive gray dots, the front as connected
    blue markers, and the picked trial as an annotated orange star.

    Args:
        front: ``optimize.pareto_frame`` output (needs ``sharpe``,
            ``total_return``, ``trial`` columns).
        all_trials: Optional frame of every completed trial (context cloud).
        picked: Trial number to highlight (from ``optimize.pick_best``).

    Returns:
        The figure.
    """
    fig = go.Figure()
    if all_trials is not None and len(all_trials):
        fig.add_trace(go.Scatter(
            x=all_trials["sharpe"], y=all_trials["total_return"], mode="markers",
            name="all trials", marker={"color": _GRID, "size": 6},
            hovertemplate="trial %{customdata}<br>sharpe %{x:.2f} | return %{y:.1%}<extra></extra>",
            customdata=all_trials["trial"]))
    front = front.sort_values("sharpe")
    fig.add_trace(go.Scatter(
        x=front["sharpe"], y=front["total_return"], mode="lines+markers", name="Pareto front",
        line={"color": SERIES[0], "width": 2}, marker={"size": 9},
        hovertemplate="trial %{customdata}<br>sharpe %{x:.2f} | return %{y:.1%}<extra></extra>",
        customdata=front["trial"]))
    if picked is not None and picked in set(front["trial"]):
        row = front[front["trial"] == picked].iloc[0]
        fig.add_trace(go.Scatter(x=[row["sharpe"]], y=[row["total_return"]], mode="markers",
                                 name="picked", marker={"symbol": "star", "size": 16,
                                                        "color": SERIES[7]}))
        fig.add_annotation(x=row["sharpe"], y=row["total_return"], text=f"trial {picked}",
                           showarrow=True, arrowhead=2, ax=30, ay=-30)
    fig.update_xaxes(title="Sharpe ratio")
    fig.update_yaxes(title="Total return", tickformat=".0%")
    return _base(fig, title="Optimization Pareto front — Sharpe vs total return")


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
