"""informed_flow's own figure: the whole book on one screen.

The engine's :func:`trading_simple.plots.plot_equity` shows the book as a single
aggregate curve — value, return, fees, realized P&L — which answers *how did the
book do* but hides *which names did it*. This strategy holds a rotating basket of
up to ``max_positions`` names, so the question that actually matters here is
per-name: who was I holding, when did I buy and sell each one, and what did each
contribute to the total. :func:`plot_holdings` answers exactly that in four
stacked, x-aligned panels:

1. **Close price of every name ever held.** Each name's adjusted close is drawn
   faded across the whole window and **bold over the spans it was actually in the
   book** (read straight off ``backtest.trade_records`` — the real fills, not a
   reconstruction), so "in the portfolio now" vs "out" is legible at a glance.
   ``▲`` marks each buy at its fill price and ``▼`` each sell, colored by name.
2. **Portfolio composition.** Each held name's **weight** — its position value as a
   fraction of the whole book — as a stacked area (colored to match panel 1), with
   **cash** riding on top so the stack fills to 100%. The hover carries the two
   absolute figures behind each weight: the **share count** held and the **dollar
   value** of the position. Because a held name is never re-trimmed, this is where
   the book's drift away from equal weight is visible — a winner's band swells while
   the cash band shrinks or grows with turnover.
3. **Cumulative realized P&L, decomposed.** One step line per name (its booked
   profit accumulating as its trades close), the **total** realized P&L in ink,
   and cumulative **fees paid** in amber — so a name carrying the book, or fees
   quietly eating the edge, both show up.
4. **Portfolio value ($)** — the one shared cash account's mark-to-market equity,
   with a dashed line at the starting capital.

It lives in the strategy folder, not in :mod:`trading_simple.plots`, because it is
specific to a rotating multi-name portfolio: it assumes portfolio mode (one shared
book) and reads meaning into "held vs not held" that a per-symbol strategy does not
have. It reuses the engine palette and the thinning/epoch-packing helpers so it
shares the repo's look and its committed-notebook size discipline.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_simple import backtest as bt
from trading_simple.data import Panel
from trading_simple.plots import _INK, _MUTED, SERIES, _base, _epoch, _scatter, _thin


def _rgba(hex_color: str, alpha: float) -> str:
    """A ``#rrggbb`` palette color as a semi-transparent ``rgba(...)`` string.

    Args:
        hex_color: A ``#rrggbb`` hex color (one of :data:`SERIES` or the gray chrome).
        alpha: Opacity in ``[0, 1]`` for the stacked-area fill.

    Returns:
        The matching CSS ``rgba(r, g, b, alpha)`` string.
    """
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _composition(pf, names: list[str]):
    """Per-symbol share count, position value ($) and book weight, plus the cash weight.

    Read straight from the portfolio, so it is what was actually held (mark-to-market on
    the adjusted close), not a reconstruction from the weight frame the strategy asked
    for — the two differ precisely because a held position is left to drift.

    Args:
        pf: The portfolio-mode ``vectorbt.Portfolio``.
        names: Symbols to report, in draw order.

    Returns:
        ``(shares, value, weight, cash)`` — three ``(bars x names)`` frames (share count,
        dollar value, and value as a fraction of the whole book) and one ``cash`` Series
        (the fraction of the book held in cash), so each bar's ``weight`` row plus
        ``cash`` sums to 1.
    """
    shares = pf.assets()[names]
    value = pf.asset_value(group_by=False)[names]
    total = pf.value()                                  # the whole book, incl. cash
    weight = value.div(total, axis=0).fillna(0.0)
    cash = (1.0 - weight.sum(axis=1)).clip(lower=0.0)
    return shares, value, weight, cash


def plot_holdings(
    panel: Panel,
    pf,
    *,
    title: str = "informed_flow — holdings, P&L and portfolio value",
    max_names: int | None = None,
) -> go.Figure:
    """One figure tying the book together: prices held, P&L per name, and value.

    Args:
        panel: The price panel the backtest ran on (supplies each name's close).
        pf: The portfolio-mode ``vectorbt.Portfolio`` for the run to visualize
            (e.g. ``backtest.run_strategy(strat, holdout, params=picked, ...)``).
        title: Figure title.
        max_names: Cap on how many held names to draw, keeping the busiest by
            absolute realized P&L. ``None`` (default) draws every name the book
            ever held. A cap only trims the panels' clutter; it is stated in the
            first panel's subtitle so nothing is dropped silently.

    Returns:
        A four-row plotly figure (close prices with buy/sell marks; portfolio
        composition — per-name weight stacked with cash, shares + $ in the hover;
        cumulative realized P&L per name plus total and fees; portfolio value). When
        the run made no trades, a titled empty figure.
    """
    trades = bt.trade_records(pf)
    curve = bt.equity_curve(pf)
    close = panel.close

    if trades.empty:
        return _base(go.Figure(), title=f"{title} — no trades in this run", height=300)

    booked = trades[trades["status"] == "Closed"]
    # Order names by absolute realized P&L so the biggest contributors get the first,
    # most-distinct palette hues and a `max_names` cap keeps what matters most.
    ranked = booked.groupby("symbol")["pnl"].sum()
    impact = ranked.abs().sort_values(ascending=False)
    # A still-open name with no closed trade yet is held but absent from `booked`; keep it.
    order = list(dict.fromkeys([*impact.index, *trades["symbol"]]))
    names = order if max_names is None else order[:max_names]
    kept = set(names)
    color = {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}

    trimmed = "" if max_names is None or len(order) <= len(names) else \
        f" · showing {len(names)} of {len(order)} names by |realized P&L|"
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.40, 0.22, 0.20, 0.18],
        subplot_titles=[
            f"Close price ($) — bold while held, faded when out of the book · "
            f"▲ buy ▼ sell{trimmed}",
            "Portfolio composition — each name's share of book value (stacked; "
            "gray = cash) · hover for shares & $",
            "Cumulative realized P&L ($) — per name, total (ink), and fees paid (amber)",
            "Portfolio value ($)",
        ],
    )

    # --- panel 1: prices, faded everywhere then bold over the real holding spans ----
    for n in names:
        series = close[n].dropna()
        fig.add_trace(_scatter(
            series, mode="lines", name=n, legendgroup=n, opacity=0.22,
            line={"color": color[n], "width": 1},
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra>" + n + " · out</extra>"),
            row=1, col=1)
    for _, t in trades.iterrows():
        if t["symbol"] not in kept:
            continue
        held = close[t["symbol"]].loc[t["entry_ts"]:t["exit_ts"]].dropna()
        if held.empty:
            continue
        fig.add_trace(_scatter(
            held, mode="lines", name=t["symbol"], legendgroup=t["symbol"], showlegend=False,
            line={"color": color[t["symbol"]], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra>" + t["symbol"] + " · held</extra>"),
            row=1, col=1)

    # Buys and sells each as ONE trace with per-point color, so the legend stays two
    # entries while every marker still reads which name it belongs to. Sells are drawn
    # only for closed trades: a position vectorbt closes at the last bar is not a sell.
    buys = trades[trades["symbol"].isin(kept)]
    fig.add_trace(go.Scatter(
        x=_epoch(pd.DatetimeIndex(buys["entry_ts"])), y=buys["entry_price"], mode="markers",
        name="buy", marker={"symbol": "triangle-up", "size": 9,
                            "color": [color[s] for s in buys["symbol"]],
                            "line": {"color": "white", "width": 1}},
        customdata=buys[["symbol"]],
        hovertemplate="buy %{customdata[0]} · %{x|%Y-%m-%d} @ $%{y:.2f}<extra></extra>"),
        row=1, col=1)
    sells = booked[booked["symbol"].isin(kept)]
    fig.add_trace(go.Scatter(
        x=_epoch(pd.DatetimeIndex(sells["exit_ts"])), y=sells["exit_price"], mode="markers",
        name="sell", marker={"symbol": "triangle-down", "size": 9,
                             "color": [color[s] for s in sells["symbol"]],
                             "line": {"color": "white", "width": 1}},
        customdata=sells[["symbol", "pnl", "return"]],
        hovertemplate="sell %{customdata[0]} · %{x|%Y-%m-%d} @ $%{y:.2f}<br>"
                      "pnl $%{customdata[1]:,.0f} (%{customdata[2]:+.1%})<extra></extra>"),
        row=1, col=1)

    # --- panel 2: composition — each name's weight (stacked), shares + $ in the hover -
    shares, posval, weight, cash = _composition(pf, names)
    # Thin every series on ONE shared grid: stacked areas must share an x axis, and
    # per-series thinning (what `_scatter` does) would desynchronize the stack. The
    # `w:`/`s:`/`v:` prefixes keep the three quantities apart through the one `_thin`.
    packed = _thin(pd.concat(
        [weight.add_prefix("w:"), shares.add_prefix("s:"), posval.add_prefix("v:"),
         cash.rename("cash")], axis=1))
    xe = _epoch(packed.index)
    for n in names:
        fig.add_trace(go.Scatter(
            x=xe, y=packed[f"w:{n}"], mode="lines", name=n, legendgroup=n, showlegend=False,
            line={"color": color[n], "width": 0.5}, stackgroup="book",
            fillcolor=_rgba(color[n], 0.55),
            customdata=packed[[f"s:{n}", f"v:{n}"]].to_numpy(),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%} of book · %{customdata[0]:,.2f} sh · "
                          "$%{customdata[1]:,.0f}<extra>" + n + "</extra>"),
            row=2, col=1)
    # Cash on top, so the stack fills to 100% and the gray band's height is "how much of
    # the book is uninvested" — which a no-rebalance book lets drift with turnover.
    fig.add_trace(go.Scatter(
        x=xe, y=packed["cash"], mode="lines", name="cash", legendgroup="cash",
        line={"color": _MUTED, "width": 0.5}, stackgroup="book",
        fillcolor=_rgba(_MUTED, 0.30),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%} cash<extra>cash</extra>"), row=2, col=1)

    # --- panel 3: realized P&L, per name and in aggregate, plus fees ----------------
    # Each name's booked profit accumulates as its trades close; the per-name lines sum
    # to `curve["realized_pnl"]` by construction (same closed trades, grouped two ways).
    for n in names:
        pnl = (booked[booked["symbol"] == n].groupby("exit_ts")["pnl"].sum()
               .reindex(close.index, fill_value=0.0).cumsum())
        fig.add_trace(_scatter(
            pnl, mode="lines", name=n, legendgroup=n, showlegend=False,
            line={"color": color[n], "width": 1},
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + n + " · realized</extra>"),
            row=3, col=1)
    fig.add_trace(_scatter(
        curve["realized_pnl"], mode="lines", name="total realized P&L",
        line={"color": _INK, "width": 2.4},
        hovertemplate="%{x|%Y-%m-%d}<br>total $%{y:,.0f}<extra></extra>"), row=3, col=1)
    fig.add_trace(_scatter(
        curve["fees"], mode="lines", name="fees paid",
        line={"color": SERIES[2], "width": 1.6, "dash": "dot"},
        hovertemplate="%{x|%Y-%m-%d}<br>fees $%{y:,.0f}<extra></extra>"), row=3, col=1)

    # --- panel 4: the shared book's value, against its starting capital --------------
    fig.add_trace(_scatter(
        curve["value"], mode="lines", name="portfolio value",
        line={"color": SERIES[0], "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>"), row=4, col=1)
    # After the trace: add_hline drops onto a subplot with no trace yet (silently).
    fig.add_hline(y=float(curve["value"].iloc[0]), line={"dash": "dash", "color": _MUTED,
                  "width": 1}, row=4, col=1)

    for r in (1, 3, 4):
        fig.update_yaxes(tickprefix="$", row=r, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], row=2, col=1)   # composition = weights
    fig.update_xaxes(type="date")
    fig = _base(fig, title=title, height=1160)
    # Many names -> a vertical legend beside the panels reads better than a header row.
    fig.update_layout(hovermode="closest",
                      legend={"orientation": "v", "yanchor": "top", "y": 1.0,
                              "xanchor": "left", "x": 1.005, "font": {"size": 10}},
                      margin={"r": 160})
    return fig
