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
    # overnight_open · one night, up close

    What a stock did between yesterday's close and today's first trading hour, minute by minute,
    and what was known before the open. Pick a stock and one of its nights. The companion
    notebook `overnight_open_study_marimo.py` asks whether any of it adds up over ~530 stocks.

    **The night** (New York time):

    | part | when | what we have |
    |---|---|---|
    | previous days | up to yesterday's close | daily bars: moves, ranges, volume, ATR |
    | after-hours | 16:01-19:59 | 1-minute bars |
    | overnight | 20:00-04:00 | no trades: only the price jump across it, and its volume (*) |
    | pre-market | 04:00-09:29 | 1-minute bars |
    | news | previous session + night | SEC 8-Ks (timed to the second), headlines (dated) |

    (*) Overnight trades are reported late, at 08:00-08:09: their volume counts as overnight
    trading, their prices are stale and ignored. Benzinga headlines carry a date but no time.

    The **first trading hour** (09:30-10:29) is the outcome, measured from the official opening
    print (what a market-on-open order gets).

    1. **The night table**: built once for all ~530 stocks and cached. The first time takes about
       half an hour: the 1-minute bars month by month, then the SEC filings name by name.
    2. **One night**: its minute candles, the filings and the headlines.
    3. **Its signals**: each one against the stock's own normal, and which flags fired.
    4. **All of the stock's nights**: did the first hour follow the night?
    """)
    return


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload research.py (after it changed)")
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
    from trading_simple.db import make_engine
    from trading_simple.plots import _INK as INK
    from trading_simple.plots import _MUTED as MUTED
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as base_layout

    _ = reload_helpers.value     # a click reruns this cell: research.py is read again
    _spec = importlib.util.spec_from_file_location(
        "overnight_open_research", strategies.STRATEGIES_DIR / "overnight_open" / "research.py")
    rs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rs)
    engine = make_engine()

    UP, DOWN = "#2a78d6", "#d03b3b"                  # regular-hours candles
    UP_PALE, DOWN_PALE = "#a9c9ef", "#eeb2b2"        # extended-hours candles
    SHADE = {"after-hours (16:01-19:59)": "rgba(237, 161, 0, 0.13)",
             "pre-market (04:00-09:29)": "rgba(74, 58, 167, 0.10)",
             "first trading hour (09:30-10:29): the outcome": "rgba(27, 175, 122, 0.15)"}
    FILING = SERIES[6]

    def finish(fig, title, height=480):
        """The repo's look (``plots._base``) with the legend under the plot."""
        base_layout(fig, title=title, height=height)
        fig.update_layout(margin={"b": 120},
                          legend={"orientation": "h", "x": 0.0, "xanchor": "left",
                                  "y": -60 / max(height - 170, 80), "yanchor": "top"})
        return fig

    return (DOWN, DOWN_PALE, FILING, INK, MUTED, SERIES, SHADE, UP, UP_PALE, engine, finish,
            go, make_subplots, np, pd, rs)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The night table

    One row per stock and session: every signal known before 09:30 and what the first hour did.
    The columns are listed in `research.FEATURES`; moves are in units of the stock's daily ATR
    (its typical daily range), volumes against the stock's own mean of the 20 sessions before.
    """)
    return


@app.cell
def _(mo):
    build_button = mo.ui.run_button(
        label="load the night table (about half an hour the first time, then seconds)")
    mo.hstack([build_button], justify="start")
    return (build_button,)


@app.cell
def _(build_button, engine, mo, rs):
    mo.stop(not build_button.value, mo.md("_Click the button to load the night table._"))
    table, filings, daily = rs.build(
        engine, progress=lambda xs, title: mo.status.progress_bar(xs, title=title))
    mo.md(f"**{len(table):,} nights** of **{table['symbol'].nunique()} stocks**, "
          f"{table['day'].min().date()} → {table['day'].max().date()}.")
    return daily, filings, table


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · One night

    **Flags** mark an unusual night. A move flag fires when the after-hours or the pre-market
    moved at least *move* daily ATRs away from the close (either way). A volume flag fires when
    the after-hours, the 08:00 late reports or the pre-market traded at least *volume* times the
    stock's own normal. The news flags fire on an SEC 8-K accepted during the night (16:00 to
    09:30), on one carrying earnings (Item 2.02), or on at least *headlines* times the usual
    number of Benzinga headlines dated the day before (and at least 3 of them).
    """)
    return


@app.cell
def _(mo, table):
    _names = sorted(table["symbol"].unique())
    stock = mo.ui.dropdown(options=_names, value="MU" if "MU" in _names else _names[0],
                           searchable=True, label="Stock")
    only_flagged = mo.ui.switch(value=True, label="list only flagged nights")
    move_atr = mo.ui.slider(0.25, 3.0, step=0.25, value=1.0, show_value=True,
                            label="move: at least this many daily ATRs")
    vol_x = mo.ui.slider(1.5, 10.0, step=0.5, value=3.0, show_value=True,
                         label="volume: at least this many times its normal")
    news_x = mo.ui.slider(1.5, 10.0, step=0.5, value=3.0, show_value=True,
                          label="headlines: at least this many times the usual count")
    mo.vstack([mo.hstack([stock, only_flagged], justify="start", gap=2),
               mo.hstack([move_atr, vol_x, news_x], justify="start", gap=2)])
    return move_atr, news_x, only_flagged, stock, vol_x


@app.cell
def _(move_atr, news_x, rs, table, vol_x):
    flagged = rs.flags(table, move_atr=move_atr.value, vol_x=vol_x.value, news_x=news_x.value)
    return (flagged,)


@app.cell
def _(flagged, mo, only_flagged, stock, table):
    _rows = table[table["symbol"] == stock.value]
    if only_flagged.value:
        _rows = _rows[flagged.loc[_rows.index].any(axis=1)]

    def _label(i):
        _r = table.loc[i]
        _on = [k for k, v in flagged.loc[i].items() if v] or ["no flag"]
        return (f"{_r['day'].date()} · {', '.join(_on)} · first hour {_r['fh_ret']:+.1%}, "
                f"range {_r['fh_range_rel']:.1f}x usual")

    _options = {_label(i): i for i in _rows.sort_values("day", ascending=False).index}
    _start = _label(_rows["fh_range_rel"].idxmax()) if len(_rows) else None
    night = mo.ui.dropdown(options=_options, value=_start, searchable=True,
                           label=f"Night ({len(_options)}, newest first; starts on the "
                                 "strongest first hour)")
    mo.hstack([night], justify="start")
    return (night,)


@app.cell
def _(daily, engine, mo, night, pd, rs, table):
    mo.stop(night.value is None, mo.md("_No night to show: switch off 'list only flagged "
                                       "nights' or lower the flags._"))
    row = table.loc[night.value]
    _own = daily[daily["symbol"] == row["symbol"]].set_index("day").sort_index()
    prev_day = _own.index[_own.index < row["day"]][-1]
    prev_close = float(_own.loc[prev_day, "close"])
    official_open = float(_own.loc[row["day"], "open"])
    bars = rs.minute_bars(row["symbol"], prev_day, row["day"], engine)
    bars = bars[(bars["time"] >= prev_day + pd.Timedelta(hours=9, minutes=30))
                & (bars["time"] <= row["day"] + pd.Timedelta(hours=16))].reset_index(drop=True)
    return bars, official_open, prev_close, prev_day, row


@app.cell
def _(DOWN, DOWN_PALE, FILING, INK, MUTED, SHADE, UP, UP_PALE, bars, filings, finish, go,
      make_subplots, np, official_open, pd, prev_close, prev_day, row):
    _x = np.arange(len(bars))
    _t = bars["time"]
    _hover = _t.dt.strftime("%a %d %b %H:%M").to_numpy()
    _today = (_t.dt.normalize() == row["day"]).to_numpy()
    _part = bars["part"].to_numpy()
    _regular = _part == "rth"
    _late = _part == "late"
    _ext = ~_regular & ~_late
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24],
                        vertical_spacing=0.03)
    for _mask, _up, _down, _name in (
            (_regular, UP, DOWN, "regular-hours candle (blue up, red down)"),
            (_ext, UP_PALE, DOWN_PALE, "extended-hours candle (pale)")):
        _b = bars[_mask]
        fig.add_trace(go.Candlestick(
            x=_x[_mask], open=_b["open"], high=_b["high"], low=_b["low"], close=_b["close"],
            increasing={"line": {"color": _up, "width": 1}, "fillcolor": _up},
            decreasing={"line": {"color": _down, "width": 1}, "fillcolor": _down},
            hovertext=_hover[_mask], name=_name), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=_x[_late], y=bars.loc[_late, "close"], mode="markers", hovertext=_hover[_late],
        marker={"symbol": "x-thin", "size": 7, "color": MUTED,
                "line": {"color": MUTED, "width": 2}},
        name="08:00-08:09 late reports (stale prices, ignored)"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, len(bars) - 1], y=[prev_close, prev_close], mode="lines", hoverinfo="skip",
        line={"color": MUTED, "dash": "dash", "width": 1.5},
        name=f"previous official close {prev_close:.2f} (night moves start here)"),
        row=1, col=1)
    _first_today = int(np.flatnonzero(_today & _regular)[0]) if (_today & _regular).any() else 0
    fig.add_trace(go.Scatter(
        x=[_first_today, len(bars) - 1], y=[official_open, official_open], mode="lines",
        hoverinfo="skip", line={"color": INK, "dash": "dot", "width": 1.5},
        name=f"official open {official_open:.2f} (the first hour starts here)"), row=1, col=1)

    _night_start = int(np.flatnonzero(_today)[0]) if _today.any() else len(bars)
    _lo, _hi = float(bars["low"].min()), float(bars["high"].max())
    fig.add_trace(go.Scatter(
        x=[_night_start - 0.5] * 2, y=[_lo, _hi], mode="lines", hoverinfo="skip",
        line={"color": INK, "dash": "dot", "width": 1},
        name="the overnight gap: no trades from 20:00 to 04:00"), row=1, col=1)

    _f = filings[(filings["symbol"] == row["symbol"]) & filings["ts"].notna()]
    _ft = _f["ts"].dt.tz_localize(None)
    _f = _f[(_ft >= prev_day + pd.Timedelta(hours=9, minutes=30))
            & (_ft <= row["day"] + pd.Timedelta(hours=16))]
    _ft = _f["ts"].dt.tz_localize(None).to_numpy()
    _pos = np.minimum(np.searchsorted(_t.to_numpy(), _ft), len(bars) - 1)   # next printed minute
    if len(_f):
        _text = [f"{pd.Timestamp(a):%a %H:%M:%S} · {b} items {c}"
                 for a, b, c in zip(_ft, _f["form"], _f["items"], strict=True)]
        fig.add_trace(go.Scatter(
            x=_pos - 0.5, y=[_hi * 1.004] * len(_pos), mode="markers", hovertext=_text,
            hoverinfo="text", marker={"symbol": "triangle-down", "size": 13, "color": FILING},
            name="SEC 8-K accepted (hover: time and items)"), row=1, col=1)

    _colors = np.where(_regular, "#9db8d9", np.where(_late, MUTED, "#d9c79d"))
    fig.add_trace(go.Bar(x=_x, y=bars["volume"], marker_color=_colors, hovertext=_hover,
                         name="shares traded per minute (log scale)"), row=2, col=1)
    for _p in _pos:                                  # shapes after the traces of both panels
        fig.add_vline(x=_p - 0.5, line={"color": FILING, "width": 1})

    _spans = {"after-hours (16:01-19:59)": (~_today) & np.isin(_part, ["close", "post"]),
              "pre-market (04:00-09:29)": _today & np.isin(_part, ["early", "late", "pre"]),
              "first trading hour (09:30-10:29): the outcome":
              _today & _regular & (_t.dt.hour * 60 + _t.dt.minute < 630).to_numpy()}
    for _name, _mask in _spans.items():
        _k = np.flatnonzero(_mask)
        for _row in (1, 2) if len(_k) else ():       # one legend entry, both panels shaded
            fig.add_vrect(x0=_k[0] - 0.5, x1=_k[-1] + 0.5, fillcolor=SHADE[_name], line_width=0,
                          layer="below", name=_name, showlegend=_row == 1, row=_row, col=1)

    _new_day = np.r_[True, _today[1:] != _today[:-1]]
    _crowded = np.zeros(len(bars), bool)           # no hour tick just before a date tick
    for _s in np.flatnonzero(_new_day):
        _crowded[max(0, _s - 25):_s] = True
    _ticks = np.flatnonzero(((_t.dt.minute == 0).to_numpy() & ~_crowded) | _new_day)
    _text = [_t.iloc[k].strftime("%a %d %b<br>%H:%M" if _new_day[k] else "%H:%M")
             for k in _ticks]
    fig.update_xaxes(tickmode="array", tickvals=_ticks, ticktext=_text, rangeslider_visible=False)
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(type="log", title_text="volume", row=2, col=1)
    fig = finish(fig, f"{row['symbol']} · {prev_day:%a %d %b} close → {row['day']:%a %d %b} "
                      "16:00", height=640)          # an assignment: shown once, by the next cell
    return (fig,)


@app.cell(hide_code=True)
def _(fig, mo):
    mo.vstack([fig, mo.md("""
    **How to read it.** Time runs left to right, one slot per printed minute: minutes without a
    trade are left out, so the 8-hour overnight gap is just the dotted vertical line.
    Blue and red candles are regular hours; pale ones are extended hours. Amber shading is the
    previous session's after-hours, purple this morning's pre-market, green the first trading
    hour, the outcome. The dashed gray line is the previous official close: every night move
    (after-hours, pre-market, the gap) is measured from it. The dotted black line is today's
    official open: the first hour is measured from it. Gray crosses at 08:00-08:09 are overnight
    trades reported late at stale prices: their volume counts as overnight trading, their prices
    are ignored. Pink triangles and lines are SEC 8-K filings (hover for the time and the
    items). The volume panel is on a log scale, or the regular session would flatten the rest.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · The night's signals

    Each signal against this stock's own history: its median over all of the stock's nights,
    and the share of its nights with a smaller value. For a move, that share is taken on the
    size (a 2-ATR drop is as unusual as a 2-ATR rise). Above 95% means a top-5% night for this
    stock.
    """)
    return


@app.cell
def _(flagged, mo, pd, row, rs, table):
    _own = table[table["symbol"] == row["symbol"]]
    _rows = []
    for _c, (_grp, _words) in rs.FEATURES.items():
        _v = row[_c]
        _size = _c in rs.MOVES or _c.endswith("range_atr")
        _hist = _own[_c].abs() if _size else _own[_c]
        _rows.append({"group": _grp, "signal": _c, "what it measures": _words,
                      "this night": round(_v, 3) if pd.notna(_v) else None,
                      "the stock's median": round(_own[_c].median(), 3),
                      "share of its nights below": (round((_hist < (abs(_v) if _size else _v))
                                                          .mean(), 3)
                                                    if pd.notna(_v) else None)})
    _on = [k for k, v in flagged.loc[row.name].items() if v]
    _text = [f"**Flags:** {', '.join(_on) or 'none'}.",
             f"**First hour:** {row['fh_ret']:+.2%} from the official open "
             f"({row['fh_ret_x']:+.2%} beyond the average stock that morning); its high-low "
             f"range {row['fh_range']:.2%} = **{row['fh_range_rel']:.1f}×** this stock's usual "
             f"first hour ({row['fh_norm']:.2%}).",
             f"**Gap at the open:** {row['gap']:+.2%}. **Rest of the day** (10:30 → close): "
             f"{row['rest_ret']:+.2%}."]
    if pd.notna(row["news_same"]):
        _text.append(f"_Hindsight only:_ {row['news_same']:.0f} Benzinga headline(s) dated "
                     "this day, possibly published after the open.")
    mo.vstack([mo.md("  \n".join(_text)),
               mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=25)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The news around this night

    SEC 8-K filings from the previous session's open to today's close, with EDGAR's acceptance
    time (New York; corrected, see the README), and the Benzinga headlines dated from the
    previous session to today. **Headlines carry a date but no time**: one dated today may have
    come out after the open, so the study never uses it as a signal.
    """)
    return


@app.cell
def _(filings, mo, pd, prev_day, row, rs):
    _f = filings[(filings["symbol"] == row["symbol"]) & filings["ts"].notna()].copy()
    _ft = _f["ts"].dt.tz_localize(None)
    _f = _f[(_ft >= prev_day + pd.Timedelta(hours=9, minutes=30))
            & (_ft <= row["day"] + pd.Timedelta(hours=16))].sort_values("ts")

    def _items(code):
        return ", ".join(f"{c} {rs.ITEMS.get(c, '')}".strip() for c in str(code).split(","))

    _lines = ["| accepted (New York) | form | items | document |", "|---|---|---|---|"]
    _lines += [f"| {t:%a %d %b %H:%M:%S} | {fm} | {_items(it)} | [open]({u}) |"
               for t, fm, it, u in zip(_f["ts"], _f["form"], _f["items"], _f["url"], strict=True)]
    _h = rs.read_headlines(row["symbol"], str(prev_day.date()), str(row["day"].date()))
    _hl = ["| dated | headline |", "|---|---|"]
    _hl += [f"| {d:%a %d %b}{' (today: maybe after the open)' if d == row['day'] else ''} | "
            f"[{h.replace('|', '/')}]({s}) |"
            for d, h, s in zip(_h["date"], _h["headline"], _h["source"], strict=True)]
    mo.vstack([
        mo.md("**SEC 8-K filings**\n\n" + ("\n".join(_lines) if len(_f) else "_none_")),
        mo.md("**Benzinga headlines**\n\n" + ("\n".join(_hl) if len(_h) else
              "_none (or no news file for these dates: headlines start on 2025-10-14)_")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · All of this stock's nights

    Every night of the stock: how far the pre-market moved from the close (in daily ATRs)
    against what the first hour then did beyond the average stock. Points on the rising
    diagonal (top right, bottom left) continued the night's move; points in the other two
    corners faded it. One stock gives a picture, not a test: the study notebook pools all
    stocks, with errors clustered by day.
    """)
    return


@app.cell
def _(FILING, INK, SERIES, finish, flagged, go, mo, np, row, rs, stock, table):
    _own = table[table["symbol"] == stock.value]
    _news = flagged.loc[_own.index, ["SEC filing at night", "earnings at night",
                                     "headlines"]].any(axis=1)
    _sizes = 6 + 10 * np.clip(np.log2(_own["pm_vol_rel"].fillna(1).clip(lower=1)), 0, 2) / 2
    _fig = go.Figure()
    for _mask, _color, _name in ((~_news, SERIES[0], "quiet night"),
                                 (_news, FILING, "news night (8-K at night or a headline flag)")):
        _d = _own[_mask]
        _fig.add_trace(go.Scatter(
            x=_d["pm_ret_atr"], y=100 * _d["fh_ret_x"], mode="markers", name=_name,
            marker={"color": _color, "size": _sizes[_mask], "opacity": 0.7,
                    "line": {"width": 0}},
            hovertext=[f"{d:%Y-%m-%d} · pre-market {p:+.2f} ATR · first hour {f:+.2%}"
                       for d, p, f in zip(_d["day"], _d["pm_ret_atr"], _d["fh_ret_x"],
                                          strict=True)], hoverinfo="text"))
    if row["symbol"] == stock.value:
        _fig.add_trace(go.Scatter(
            x=[row["pm_ret_atr"]], y=[100 * row["fh_ret_x"]], mode="markers",
            name="the night shown above",
            marker={"symbol": "circle-open", "size": 22, "color": INK,
                    "line": {"color": INK, "width": 2}}))
    _fig.add_hline(y=0, line={"color": "#b9b7b0", "width": 1})
    _fig.add_vline(x=0, line={"color": "#b9b7b0", "width": 1})
    _fig.update_xaxes(title_text="pre-market move at 09:29, in daily ATRs (0 = the close)")
    _fig.update_yaxes(title_text="first-hour return beyond the average stock, %")
    finish(_fig, f"{stock.value}: the night's move against the first hour "
                 "(bigger dot = busier pre-market)", height=520)
    _ic = rs.rank_ic(_own["pm_ret_atr"], _own["fh_ret_x"], _own["day"])
    mo.vstack([_fig, mo.md(
        f"Rank correlation over {_ic['n']} nights: **{_ic['ic']:+.2f}** (positive = the first "
        "hour tends to continue the night, negative = to fade it). Nights without a pre-market "
        "print are not drawn.")])
    return


if __name__ == "__main__":
    app.run()
