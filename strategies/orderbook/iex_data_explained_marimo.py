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
    # IEX · what the free feed holds, and what a strategy could take from it

    IEX (Investors Exchange) is one US stock exchange, about 2-3% of US share volume. Its free
    **HIST** download is its DEEP market-data feed of a past trading day, for the whole market, a
    day late: every change of its displayed order book, every trade, and the status of every stock.
    This notebook opens one day of it and shows, with pictures, **what each kind of message holds**
    (§1-§3), **what information a strategy could extract and how each piece is computed** (§4), and
    **what testing that would cost** (§5).

    In four lines:

    * **The book** is IEX's own, displayed orders only, summed per price: no single orders, no queue
      positions, no hidden liquidity. It is thin and often many ticks wide.
    * **The trades** are every execution on IEX, hidden liquidity included, with flags (intermarket
      sweep, odd lot, extended hours ...) but no buy / sell side. Many print inside the displayed
      spread.
    * **The status messages** cover about 13,000 symbols at once: halts, the short-sale restriction
      (Rule 201), IEX's opening, a retail-interest flag. This is the part that bars cannot give you.
    * **Auctions and official prices** exist only for IEX-listed securities. In the file you load,
      those are test symbols.

    **How to use it.** Run it with `uv run marimo edit
    strategies/orderbook/iex_data_explained_marimo.py` (or the VS Code marimo extension). §1 asks
    for a day and a click: the first MB of the day's file are downloaded (a stored file is not
    fetched again) and decoded, about a minute the first time. From §3 on, the pictures react to the
    controls above them: the stock, the width of the ladder, the moment drawn in Fig 4, the bucket
    of Fig 5. Code: `iex_features.py` and `orderbook_free.py` (tests: `tests/test_iex_features.py`,
    `tests/test_orderbook_free.py`). Data: `var/orderbook_free/` (git-ignored; IEX's historical data
    is under IEX's terms of use). The message formats are IEX's DEEP specification v1.08. **Nothing
    here is a trading result**: no feature below has been tested for predicting returns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(df, rows=15, wrapped=None, digits=3, widths=None):
        """A DataFrame as a marimo table (an unnamed index is dropped, a named one kept)."""
        named = df.index.name or df.index.nlevels > 1
        flat = df.reset_index() if named else df.reset_index(drop=True)
        return mo.ui.table(
            flat.round(digits),
            selection=None,
            page_size=rows,
            wrapped_columns=wrapped,
            column_widths=widths,
            show_column_summaries=False,
            show_download=False,
        )

    return (table,)


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(
        label="reload iex_features.py and orderbook_free.py (after they changed)"
    )
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util
    import time

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from trading_simple import plots, strategies

    def _load(name):
        """A helper module of strategies/orderbook, loaded by path."""
        spec = importlib.util.spec_from_file_location(
            name, strategies.STRATEGIES_DIR / "orderbook" / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None, name
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _ = reload_helpers.value  # a click reruns this cell: both helper files are read again
    ob, feat = _load("orderbook_free"), _load("iex_features")
    return feat, go, make_subplots, np, ob, pd, plots, time


@app.cell(hide_code=True)
def _(np, pd):
    NY = "America/New_York"

    def below(fig, y=-0.15):
        """Legend under the plot."""
        fig.update_layout(
            legend={"orientation": "h", "x": 0, "xanchor": "left", "y": y, "yanchor": "top"}
        )
        return fig

    def et(ns):
        """New York clock text of nanosecond timestamps (plotly draws text as written)."""
        stamps = pd.to_datetime(np.asarray(ns), unit="ns", utc=True).tz_convert(NY)
        return stamps.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

    return NY, below, et


@app.cell
def _():
    TICK = 0.01  # price step of the ladder picture (Fig 2)
    SHOW_BEFORE_S, SHOW_AFTER_S = 10, 75  # seconds around the open shown in Fig 2
    GRID_S = 0.5  # seconds between the columns of Fig 2
    LEVELS = 3  # levels per side in the multi-level imbalance (Fig 4)
    return GRID_S, LEVELS, SHOW_AFTER_S, SHOW_BEFORE_S, TICK


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · What IEX gives away for free

    IEX's file index (`iextrading.com/api/1.0/hist`) lists, for each trading day, a capture of each
    feed: **TOPS** (best bid and offer, last sale), **DEEP** (the price-level book, trades, status
    messages) and the newer **DPLS / DPLC** files (from 2024 and 2026; presumably the order-by-order
    DEEP+, not opened here). They are next-day, free, and large: the tables count them and size
    them. One technical fact decides what is cheap: a gzip file can only be read from its START, so
    the **opening of a day is cheap** (a prefix) while its afternoon and close cost the whole file.
    """)
    return


@app.cell
def _(mo, ob, pd, table):
    try:
        _index_json = ob.iex_index()
    except Exception as _error:
        mo.stop(
            True,
            mo.callout(mo.md(f"IEX's file index could not be read ({_error})."), kind="danger"),
        )
    _index = pd.DataFrame(
        [
            {"day": day, "feed": f["feed"], "gb": int(f["size"]) / 1e9}
            for day, files in _index_json.items()
            for f in files
        ]
    )
    deep = _index[_index["feed"] == "DEEP"].assign(year=lambda d: d["day"].str[:4])
    _feeds = _index.groupby("feed").agg(
        files=("day", "size"),
        first_day=("day", "min"),
        last_day=("day", "max"),
        median_gb=("gb", "median"),
        latest_gb=("gb", "last"),
    )
    _years = deep.groupby("year").agg(
        days=("day", "size"), median_gb=("gb", "median"), total_tb=("gb", lambda s: s.sum() / 1000)
    )
    mo.vstack(
        [
            mo.md("**Table · the feeds in IEX's index**"),
            table(_feeds),
            mo.md("**Table · DEEP files per year** (`total_tb` = all the year's files together)"),
            table(_years),
        ]
    )
    return (deep,)


@app.cell
def _(deep, mo):
    _dates = [f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in sorted(deep["day"], reverse=True)]
    load_form = (
        mo.md("{day} {mb} {stocks}")
        .batch(
            day=mo.ui.dropdown(
                options=_dates, value=_dates[0], searchable=True, label="Trading day"
            ),
            mb=mo.ui.slider(
                50, 600, step=50, value=200, show_value=True, label="MB of the day's gzip"
            ),
            stocks=mo.ui.text(value="INTC, SPY, NVDA, AAPL", label="Stocks to rebuild"),
        )
        .form(submit_button_label="download and decode the day")
    )
    mo.vstack(
        [
            mo.md(
                "**Pick the day** (the newest one IEX has published is the default), **how much of "
                "its file to read** (200 MB reach about 3 minutes after the open) and **which "
                "stocks' books to rebuild**, then click the button."
            ),
            load_form,
        ]
    )
    return (load_form,)


@app.cell
def _(NY, et, feat, load_form, mo, ob, pd, time):
    mo.stop(
        load_form.value is None,
        mo.md("_Pick a day above and click the button: the first MB of the day's file are "
              "downloaded (a stored file is not fetched again) and decoded._"),
    )
    DAY = load_form.value["day"]
    _typed = load_form.value["stocks"].upper().replace(",", " ").replace(";", " ")
    _wanted = list(dict.fromkeys(_typed.split()))
    sys_ns = pd.Timestamp(f"{DAY} 08:00:00", tz=NY).value
    open_ns = pd.Timestamp(f"{DAY} 09:30:00", tz=NY).value
    with mo.status.spinner(title="downloading and decoding the day ..."):
        got = ob.fetch_iex(DAY, "DEEP", load_form.value["mb"])
        _t0 = time.perf_counter()
        iex = ob.decode_iex(got["path"], _wanted)
        decode_s = time.perf_counter() - _t0
        st = feat.status_tables(iex["others"])
    STOCKS = [t for t in _wanted if iex["events"][t]]
    mo.stop(
        not STOCKS,
        mo.callout(
            mo.md(f"None of {_wanted} had a book update in this part of the file."), kind="danger"
        ),
    )
    _end = et([iex["last_ns"]])[0][11:19]
    mo.stop(
        iex["last_ns"] < open_ns + 80 * 10**9,
        mo.callout(
            mo.md(
                f"This part of the file ends at {_end}, less than 80 seconds after the 09:30 open, "
                "so the open cannot be shown. Raise the MB slider (the first minutes take more MB "
                "on a busy day) or pick another day."
            ),
            kind="danger",
        ),
    )
    n_msgs = sum(iex["types"].values())
    _sysev = st["system_event"]
    _clock = dict(zip(_sysev["event"], et(_sysev["t_ns"]).str[11:19], strict=True))
    _how = "stored already" if got["cached"] else f"downloaded in {got['seconds']:.0f} s"
    mo.md(
        f"**{DAY}**: IEX's DEEP file for the day is {got['full_bytes'] / 1e9:.1f} GB; the first "
        f"{got['bytes'] / 1e6:.0f} MB are used here ({_how}).\n\n"
        f"Decoded {n_msgs:,} messages in {decode_s:.0f} s "
        f"({n_msgs / decode_s / 1e6:.2f} M per second) "
        f"from a {iex['format']} capture. System events (New York): start of messages "
        f"{_clock.get('O')}, system hours {_clock.get('S')}, regular hours {_clock.get('R')}. "
        f"Book updates run from {et([iex['first_ns']])[0][11:19]} to {_end}; "
        f"{iex['symbols']:,} symbols had at least one. Books rebuilt for: {', '.join(STOCKS)}."
    )
    return DAY, STOCKS, decode_s, got, iex, n_msgs, open_ns, st, sys_ns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · What is inside one DEEP day

    The capture is a stream of small binary messages. The table lists every message type, what it
    tells you, which securities it covers, and how many there are in the part of the file you
    fetched: nearly all are price-level updates (the displayed size at one price changed); trades
    and status messages are about one percent each. **Fig 1** is the feed's pulse: the start-of-day
    status spin, the quiet pre-market, and the burst when the regular session starts and every book
    is rebuilt at once.
    """)
    return


@app.cell
def _(feat, iex, pd, table):
    table(
        pd.DataFrame(
            [
                {
                    "type": t,
                    "name": c[0],
                    "what it tells you": c[1],
                    "covers": c[2],
                    "messages": iex["types"].get(t, 0),
                }
                for t, c in feat.CATALOGUE.items()
            ]
        ),
        rows=13,
        wrapped=["what it tells you"],
    )
    return


@app.cell
def _(DAY, below, et, feat, go, iex, open_ns, plots, sys_ns):
    _rh = feat.rhythm_frame(iex["rhythm"], iex["bucket_s"])
    _fig = go.Figure()
    for _i, _col in enumerate(_rh.columns):
        _fig.add_trace(
            go.Scatter(
                x=et(_rh.index.astype("int64").to_numpy()),
                y=_rh[_col].where(_rh[_col] > 0),
                mode="lines",
                name=f"{_col}, per {iex['bucket_s']} s",
                line={"color": plots.SERIES[_i]},
            )
        )
    for _ns, _label in ((sys_ns, "08:00 IEX opens for orders"), (open_ns, "09:30 regular session")):
        _fig.add_trace(
            go.Scatter(
                x=et([_ns, _ns]),
                y=[1, _rh.to_numpy().max()],
                mode="lines",
                name=_label,
                line={"color": "#898781", "dash": "dot"},
            )
        )
    _fig.update_yaxes(type="log", title="messages per bucket (log scale)")
    _title = f"Fig 1 · the pulse of IEX's DEEP feed ({DAY}, New York time)"
    plots._base(_fig, title=_title, height=420)
    below(_fig, -0.2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading Fig 1.** Each line counts one kind of message per 10-second bucket, on a log scale.
    The **status messages** come in one wave before 08:00 (every symbol's trading status,
    operational halt, short-sale status and retail flag) and again at 09:30:00, when each stock's
    opening completes. **Book updates** trickle in from 08:00, then jump by two to three orders of
    magnitude at 09:30:00: IEX publishes the whole queued book of every stock within a second.
    Trades follow the same shape. After the first seconds the rate settles at a level far above the
    pre-market's.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · One stock's book, as a picture

    **Fig 2** is the DEEP book of one stock from a few seconds before to about a minute after the
    open. Price runs up, time across, and the colour is the shares resting at that price (blue =
    bids, orange = asks, darker = more). The dark lines are the best bid and ask, the grey dots are
    trades on IEX. Look for three things: (1) **the left edge**, before 09:30:00: how much book is
    there near the market?; (2) **how fast the book appears** at the open; (3) **how sparse and
    uneven it is**: a few levels, gaps, a spread of several ticks. DEEP shows displayed orders only,
    summed per price: one 300-share order and three 100-share orders look the same, and hidden or
    midpoint-peg liquidity does not appear at all, although it trades (§4.4). The controls choose
    the stock (for this figure and all that follow) and how many ticks around the mid the ladder
    keeps (a stock with a wide book, such as SPY, needs a larger number).
    """)
    return


@app.cell
def _(STOCKS, mo):
    focus = mo.ui.dropdown(options=STOCKS, value=STOCKS[0], label="Stock")
    max_ticks = mo.ui.slider(
        10,
        300,
        step=10,
        value=40,
        show_value=True,
        debounce=True,
        label="Ladder: ticks kept around the mid",
    )
    mo.hstack([focus, max_ticks], justify="start", gap=2)
    return focus, max_ticks


@app.cell
def _(LEVELS, STOCKS, feat, iex):
    qf = {s: feat.quotes(iex["events"][s], LEVELS) for s in STOCKS}
    return (qf,)


@app.cell
def _(
    DAY,
    GRID_S,
    SHOW_AFTER_S,
    SHOW_BEFORE_S,
    TICK,
    below,
    et,
    feat,
    focus,
    go,
    iex,
    max_ticks,
    np,
    ob,
    open_ns,
    plots,
    qf,
):
    _grid = np.arange(
        open_ns - SHOW_BEFORE_S * 10**9, open_ns + SHOW_AFTER_S * 10**9, round(GRID_S * 1e9)
    )
    _run = ob.replay(iex["events"][focus.value], grid_s=1, at_ns=list(_grid))
    _times, _prices, _bid_z, _ask_z = feat.ladder(
        _run["books"], tick=TICK, max_ticks=max_ticks.value
    )
    _q = qf[focus.value]
    _qw = _q[(_q["t_ns"] >= _grid[0]) & (_q["t_ns"] <= _grid[-1])]
    _trw = iex["trades"][focus.value]
    _trw = _trw[(_trw["t_ns"] >= _grid[0]) & (_trw["t_ns"] <= _grid[-1])]

    _fig = go.Figure()
    _blue, _orange = ("#d6e4fa", "#154c9a"), ("#fde3d6", "#b8400f")
    for _z, (_lo, _hi), _side in ((_bid_z, _blue, "bid"), (_ask_z, _orange, "ask")):
        _fig.add_trace(
            go.Heatmap(
                x=et(_times),
                y=_prices,
                z=np.log10(_z),
                colorscale=[[0, _lo], [1, _hi]],
                showscale=False,
                hoverongaps=False,
                customdata=_z,
                hovertemplate=_side + " size %{customdata:,.0f} shares at %{y:.2f}<extra></extra>",
            )
        )
    for _name, _color in (
        ("resting bid size (darker = more shares)", "#2a78d6"),
        ("resting ask size (darker = more shares)", "#eb6834"),
    ):
        _fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"symbol": "square", "size": 11, "color": _color},
                name=_name,
            )
        )
    for _col, _name, _color in (("bid", "best bid", "#154c9a"), ("ask", "best ask", "#b8400f")):
        _s = plots._thin(_qw.set_index("t_ns")[_col])
        _fig.add_trace(
            go.Scatter(
                x=et(_s.index),
                y=_s.to_numpy(),
                mode="lines",
                name=_name,
                line={"color": _color, "width": 1.5, "shape": "hv"},
            )
        )
    _fig.add_trace(
        go.Scatter(
            x=et(_trw["t_ns"]),
            y=_trw["price"],
            mode="markers",
            name="trade on IEX (marker grows with shares)",
            marker={
                "size": 4 + 2 * np.sqrt(_trw["size"] / 100),
                "color": "rgba(11,11,11,0.55)",
                "line": {"color": "white", "width": 0.5},
            },
            customdata=_trw["size"],
            hovertemplate="%{y:.2f} x %{customdata} shares<extra>trade</extra>",
        )
    )
    _after = _qw[_qw["t_ns"] >= open_ns]
    if len(_after):
        _fig.update_yaxes(
            range=[_after["bid"].quantile(0.01) - 0.1, _after["ask"].quantile(0.99) + 0.1]
        )
    _fig.update_yaxes(title="price")
    _title = f"Fig 2 · {focus.value}'s IEX order book around the open ({DAY}, New York time)"
    plots._base(_fig, title=_title, height=520)
    below(_fig, -0.12)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · What a strategy could extract, and how each piece is computed

    Five kinds of information come out of the feed. For each: **how it is computed** (with a picture
    of the calculation on real data), **what a strategy could use it for**, and **what limits it
    here**. The pointers (author, year) are where the idea comes from; they were not checked for
    this notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.1 · Liquidity state: spread and depth

    **How it is computed.** From the book after every update: the quoted spread is (best ask − best
    bid) / mid, in basis points; depth is the dollar value (price × shares) resting within 10 bps of
    the mid, both sides together. Both are sampled every second. **Use:** a cost estimate (a
    strategy that trades at the touch pays about half the spread each way), an entry filter (skip
    names whose book is wide or thin), execution timing (Fig 3 shows how the book builds after the
    open), a regime variable for other signals. **Limit:** this is IEX's book, not the market's: the
    consolidated spread is usually tighter. The table compares the stocks before and after the open.
    """)
    return


@app.cell
def _(STOCKS, iex, np, ob, open_ns, pd, sys_ns, table):
    liq = {s: ob.replay(iex["events"][s], grid_s=1) for s in STOCKS}
    _rows = []
    for _s in STOCKS:
        _smp = liq[_s]["samples"]
        _smp_ns = _smp.index.astype("int64").to_numpy()
        _ev_ns = np.array([m.t_ns for m in iex["events"][_s]])
        _periods = (("08:00-09:30", sys_ns, open_ns), ("from 09:30", open_ns, iex["last_ns"]))
        for _label, _lo, _hi in _periods:
            _part = _smp[(_smp_ns >= _lo) & (_smp_ns < _hi)]
            _n_events = ((_ev_ns >= _lo) & (_ev_ns < _hi)).sum()
            _levels = _part["bid_levels"] + _part["ask_levels"]
            _usd = _part["bid_usd_10"] + _part["ask_usd_10"]
            _rows.append(
                {
                    "stock": _s,
                    "period": _label,
                    "book events per second": _n_events / ((_hi - _lo) / 1e9),
                    "levels, both sides (median)": _levels.median(),
                    "spread, bps (median)": _part["spread_bps"].median(),
                    "USD within 10 bps (median)": _usd.median(),
                }
            )
    table(pd.DataFrame(_rows).set_index(["stock", "period"]), rows=8)
    return (liq,)


@app.cell
def _(below, et, focus, go, liq, make_subplots, mo, open_ns, plots):
    _smp = liq[focus.value]["samples"]
    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "quoted spread, bps of the mid (log scale)",
            "USD resting within 10 bps of the mid, both sides (log scale)",
        ),
    )
    _series = (
        (_smp["spread_bps"], f"{focus.value}: spread (bps)"),
        (_smp["bid_usd_10"] + _smp["ask_usd_10"], f"{focus.value}: USD within 10 bps"),
    )
    for _row, (_y, _name) in enumerate(_series, start=1):
        _y = plots._thin(_y.where(_y > 0))
        _fig.add_trace(
            go.Scatter(
                x=et(_y.index.astype("int64").to_numpy()),
                y=_y,
                mode="lines",
                name=_name,
                line={"color": plots.SERIES[0]},
            ),
            _row,
            1,
        )
        _fig.add_trace(
            go.Scatter(
                x=et([open_ns, open_ns]),
                y=[_y.min(), _y.max()],
                mode="lines",
                name="09:30 regular session",
                showlegend=_row == 1,
                line={"color": "#898781", "dash": "dot"},
            ),
            _row,
            1,
        )
    _fig.update_yaxes(type="log")
    _title = f"Fig 3 · {focus.value}'s liquidity on IEX, 08:00 to just after the open"
    plots._base(_fig, title=_title, height=520)
    mo.vstack([mo.hstack([focus], justify="start"), below(_fig, -0.1)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.2 · Queue imbalance and microprice

    **How they are computed** (the left half of Fig 4 draws them on one real instant). With `Qb` and
    `Qa` the shares queued at the best bid and ask:

    * **queue imbalance** `OBI(1) = (Qb − Qa) / (Qb + Qa)`, in [−1, +1]: +1 means only bids are
      queued at the touch, −1 only asks. `OBI(3)` (the setting `LEVELS`) uses the sizes of the best
      three levels per side.
    * **microprice** `(bid × Qa + ask × Qb) / (Qb + Qa)`: the mid, pulled toward the side with LESS
      size, because a thin queue is exhausted first and the price tends to step that way.

    **Use:** a forecast of the next price move over seconds. Published work finds that queue
    imbalance helps forecast the next move (Lipton, Pesavento & Sotiropoulos 2013, mainly for
    large-tick stocks; Cao, Hansch & Wang 2009; Stoikov 2018). **Limit:** it fades within seconds,
    so spread and fees decide whether anything survives; and these are IEX's queues, a few percent
    of the market's. The instant in Fig 4 is, by default, the most lopsided queue in the first
    minute (so the formula has something to show): it is not a typical moment. Switch the automatic
    pick off to choose the moment yourself. The right half shows how the same quantities move around
    it.
    """)
    return


@app.cell
def _(focus, iex, mo, open_ns):
    auto_pick = mo.ui.switch(value=True, label="auto: the most lopsided queue")
    at_secs = mo.ui.slider(
        1,
        max(2, int((iex["last_ns"] - open_ns) / 1e9) - 5),
        step=1,
        value=20,
        show_value=True,
        debounce=True,
        label="or the moment: seconds after the open",
    )
    mo.hstack([focus, auto_pick, at_secs], justify="start", gap=2)
    return at_secs, auto_pick


@app.cell
def _(at_secs, auto_pick, focus, open_ns, qf):
    q_focus = qf[focus.value]
    _lop = q_focus[
        (q_focus["t_ns"] >= open_ns + 10 * 10**9) & (q_focus["t_ns"] <= open_ns + 60 * 10**9)
    ]
    _lop = _lop[_lop[["bid_qty", "ask_qty"]].min(axis=1) >= 50]
    if auto_pick.value and len(_lop):
        at_ns = int(_lop.loc[_lop["obi1"].abs().idxmax(), "t_ns"]) + 1
    else:
        at_ns = open_ns + at_secs.value * 10**9
    return at_ns, q_focus


@app.cell
def _(
    DAY,
    LEVELS,
    TICK,
    at_ns,
    below,
    et,
    feat,
    focus,
    go,
    iex,
    make_subplots,
    ob,
    plots,
    q_focus,
):
    _bids, _asks = ob.replay(iex["events"][focus.value], grid_s=1, at_ns=[at_ns])["books"][at_ns]
    _f = feat.book_features(_bids, _asks, LEVELS)
    _mid = _f["mid"]

    _fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.4, 0.6],
        horizontal_spacing=0.1,
        subplot_titles=(
            f"{focus.value}'s book at {et([at_ns])[0][11:23]}",
            "the same quantities around it",
        ),
    )
    _sides = (
        ("bid", _bids, sorted(_bids, reverse=True)[:LEVELS], "#2a78d6"),
        ("ask", _asks, sorted(_asks)[:LEVELS], "#eb6834"),
    )
    for _side, _book, _top, _color in _sides:
        _parts = (
            ("best level (all of OBI(1))", _top[:1], 1.0),
            (f"levels 2-{LEVELS} (added in OBI({LEVELS}))", _top[1:], 0.45),
        )
        for _label, _ps, _opacity in _parts:
            _fig.add_trace(
                go.Bar(
                    x=[_book[p] for p in _ps],
                    y=_ps,
                    orientation="h",
                    width=TICK * 0.8,
                    marker={"color": _color, "opacity": _opacity},
                    name=f"{_side}: {_label}",
                ),
                1,
                1,
            )
    _xmax = max(book[p] for _name, book, top, _tint in _sides for p in top) * 1.25
    for _y, _color, _dash in ((_mid, "#0b0b0b", "dot"), (_f["micro"], "#4a3aa7", "solid")):
        _fig.add_trace(
            go.Scatter(
                x=[0, _xmax],
                y=[_y, _y],
                mode="lines",
                showlegend=False,
                line={"color": _color, "dash": _dash, "width": 2},
            ),
            1,
            1,
        )
    _fig.update_xaxes(title="shares resting at the price", range=[0, _xmax], row=1, col=1)
    _fig.update_yaxes(title="price", range=[_mid - 15 * TICK, _mid + 15 * TICK], row=1, col=1)
    _fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.98,
        y=0.02,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e1e0d9",
        text=(
            f"OBI(1) = ({_f['bid_qty']:.0f} − {_f['ask_qty']:.0f}) / "
            f"({_f['bid_qty']:.0f} + {_f['ask_qty']:.0f}) = {_f['obi1']:+.2f}<br>"
            f"OBI({LEVELS}) = ({_f['bid_qty_k']:.0f} − {_f['ask_qty_k']:.0f}) / "
            f"({_f['bid_qty_k']:.0f} + {_f['ask_qty_k']:.0f}) = {_f['obik']:+.2f}<br>"
            f"microprice {_f['micro']:.4f} vs mid {_mid:.4f} = "
            f"{(_f['micro'] - _mid) / TICK:+.2f} ticks"
        ),
    )
    _w = q_focus[(q_focus["t_ns"] >= at_ns - 5 * 10**9) & (q_focus["t_ns"] <= at_ns + 5 * 10**9)]
    _lines = (
        ("bid", "best bid", "#2a78d6", "solid"),
        ("ask", "best ask", "#eb6834", "solid"),
        ("mid", "mid", "#0b0b0b", "dot"),
        ("micro", "microprice", "#4a3aa7", "solid"),
    )
    for _col, _name, _color, _dash in _lines:
        _fig.add_trace(
            go.Scatter(
                x=et(_w["t_ns"]),
                y=_w[_col],
                mode="lines",
                name=_name,
                line={"color": _color, "dash": _dash, "shape": "hv", "width": 1.5},
            ),
            1,
            2,
        )
    _tw = iex["trades"][focus.value]
    _tw = _tw[(_tw["t_ns"] >= at_ns - 5 * 10**9) & (_tw["t_ns"] <= at_ns + 5 * 10**9)]
    _fig.add_trace(
        go.Scatter(
            x=et(_tw["t_ns"]),
            y=_tw["price"],
            mode="markers",
            name="trade on IEX",
            marker={"size": 6, "color": "rgba(11,11,11,0.6)"},
        ),
        1,
        2,
    )
    _fig.add_trace(
        go.Scatter(
            x=et([at_ns, at_ns]),
            y=[_w["bid"].min(), _w["ask"].max()],
            mode="lines",
            name="the moment drawn on the left",
            line={"color": "#898781", "dash": "dot"},
        ),
        1,
        2,
    )
    _title = f"Fig 4 · queue imbalance and microprice, {focus.value} ({DAY})"
    plots._base(_fig, title=_title, height=460)
    below(_fig, -0.18)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3 · Order-flow imbalance (OFI)

    **How it is computed** (Cont, Kukanov & Stoikov 2014). Compare each new best-bid / best-ask
    state with the one before. A bid that is higher or equal adds its whole size to buying pressure;
    a bid that is lower or equal takes the old size away (at an unchanged price both apply, so the
    net is the change of size); asks work the mirror way (an ask that is lifted away or shrunk is
    buying pressure). The first table applies the rule to six consecutive real states, starting at
    the moment of Fig 4 (`e_bid + e_ask = ofi`). Then OFI is summed over short buckets and set
    against the change of the mid over the same bucket (Fig 5); the bucket length is a control.

    **Use:** it explains the price change of the *same* interval (price impact: the slope is roughly
    inverse to the depth), and is an input to impact models and short-horizon forecasts. That is a
    description, not a forecast: whether one bucket's OFI predicts the NEXT bucket's move is a
    separate question that this notebook does not answer. **Limit:** the last table shows that even
    the same-interval relation depends on the bucket length and on the stock, and is clearest in the
    shortest buckets (the long buckets have few observations: see the noise band printed under the
    table).
    """)
    return


@app.cell
def _(focus, mo):
    ofi_bucket = mo.ui.dropdown(
        options={"0.1 s": 0.1, "0.25 s": 0.25, "0.5 s": 0.5, "1 s": 1, "2 s": 2},
        value="0.1 s",
        label="Bucket length",
    )
    mo.hstack([focus, ofi_bucket], justify="start", gap=2)
    return (ofi_bucket,)


@app.cell
def _(
    TICK,
    at_ns,
    below,
    et,
    feat,
    focus,
    go,
    iex,
    mo,
    np,
    ofi_bucket,
    open_ns,
    plots,
    q_focus,
    table,
):
    _parts = feat.ofi(q_focus, parts=True)
    _tbl = q_focus.join(_parts)
    _i0 = int(np.searchsorted(_tbl["t_ns"].to_numpy(), at_ns))
    _rule = _tbl.iloc[_i0 : _i0 + 6].copy()
    _rule["time"] = et(_rule["t_ns"]).str[11:]

    _t_end = min(iex["last_ns"], open_ns + 150 * 10**9)
    _b = feat.bucket_ofi(q_focus, open_ns, _t_end, ofi_bucket.value)
    _slope, _intercept = np.polyfit(_b["ofi"], _b["dmid"] / TICK, 1)
    _r = _b["ofi"].corr(_b["dmid"])
    _xs = np.array([_b["ofi"].min(), _b["ofi"].max()])
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_b["ofi"],
            y=_b["dmid"] / TICK,
            mode="markers",
            name=f"{ofi_bucket.value} s buckets after the open (n = {len(_b):,})",
            marker={"size": 5, "color": plots.SERIES[0], "opacity": 0.5},
            customdata=et(_b["t_ns"]).str[11:],
            hovertemplate=(
                "%{customdata}<br>OFI %{x:,.0f} shares<br>mid %{y:+.1f} ticks<extra></extra>"
            ),
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_xs,
            y=_intercept + _slope * _xs,
            mode="lines",
            name=f"least-squares line: {100 * _slope:+.2f} ticks per 100 shares, r = {_r:+.2f}",
            line={"color": "#0b0b0b"},
        )
    )
    _fig.update_xaxes(title="order-flow imbalance in the bucket (shares)")
    _fig.update_yaxes(title="change of the mid over the bucket (ticks)")
    _title = (
        f"Fig 5 · order-flow imbalance against the mid change, {focus.value}, "
        "first 150 s after the open"
    )
    plots._base(_fig, title=_title, height=420)
    mo.vstack(
        [
            mo.md(f"**Table · the rule on six consecutive states of {focus.value}**"),
            table(
                _rule[["time", "bid", "bid_qty", "ask", "ask_qty", "e_bid", "e_ask", "ofi"]],
                rows=6,
            ),
            below(_fig, -0.2),
        ]
    )
    return


@app.cell
def _(STOCKS, feat, iex, mo, np, open_ns, pd, qf, table):
    _t_end = min(iex["last_ns"], open_ns + 150 * 10**9)
    _buckets = (0.1, 0.25, 0.5, 1, 2)
    sens = {}
    for _s in STOCKS:
        sens[_s] = {}
        for _bucket in _buckets:
            _bb = feat.bucket_ofi(qf[_s], open_ns, _t_end, _bucket)
            sens[_s][f"{_bucket} s"] = _bb["ofi"].corr(_bb["dmid"])
    _n = {_bucket: (_t_end - open_ns) / 1e9 / _bucket for _bucket in _buckets}
    _band = ", ".join(f"{_bucket} s ±{2 / np.sqrt(_k):.2f}" for _bucket, _k in _n.items())
    mo.vstack(
        [
            mo.md(
                "**Table · correlation of OFI with the mid change of the same bucket**, by stock "
                "and bucket length (first 150 s after the open)"
            ),
            table(pd.DataFrame(sens).T.rename_axis("stock"), digits=2),
            mo.md(f"Noise band (2 / sqrt(n) with n = buckets): {_band}."),
        ]
    )
    return (sens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.4 · Trades: where they print, and who pushed

    DEEP's trade report has price, size and **flags**, but not the side. A trade is labelled buyer-
    or seller-initiated by comparing its price with the IEX quote just before it (Lee & Ready 1991):
    at the ask or above the mid = buy, at the bid or below the mid = sell, exactly at the mid = the
    tick rule (the sign of the last price change). DEEP sends a trade report *before* the book
    update it causes, with the same timestamp, so the quote used must be strictly earlier. **Table
    1** says where trades print against IEX's own quote: a trade *inside* the displayed spread
    executed against liquidity the book does not show (midpoint or hidden orders). **Table 2**
    counts the flags over every stock after the open. **Fig 6** shows the chosen stock's trades
    labelled this way, and the running total of signed shares.

    **Use:** signed volume per interval (order-flow pressure; cf. Chordia & Subrahmanyam 2004), the
    share of intermarket sweeps (urgent, liquidity-taking orders), odd-lot activity, trades per
    second as an activity measure. Volume-bucketed versions of signed flow have been proposed as a
    "toxicity" measure (VPIN; Easley, López de Prado & O'Hara 2012; disputed). **Limit:** IEX's
    trades are a small part of the market; the sign is inferred; and a print at the midpoint says
    little about direction.
    """)
    return


@app.cell
def _(STOCKS, feat, iex, mo, open_ns, pd, qf, table):
    signed = {}
    for _s in STOCKS:
        _tr = iex["trades"][_s]
        signed[_s] = feat.sign_trades(_tr[_tr["t_ns"] >= open_ns], qf[_s])
    _where = pd.DataFrame({s: signed[s]["where"].value_counts(normalize=True) for s in STOCKS})
    _where = _where.reindex(list(feat.WHERE)).fillna(0).mul(100).round(0)
    _where = _where.rename_axis("where it printed")

    _ta = iex["trades_all"]
    _ta = _ta[_ta["t_ns"] >= open_ns]
    _flag_rows = []
    for _name, _mask in (
        ("intermarket sweep order (ISO)", 0x80),
        ("extended hours", 0x40),
        ("odd lot (under 100 shares)", 0x20),
        ("exempt from the trade-through rule", 0x10),
        ("single-price cross", 0x08),
    ):
        _hit = (_ta["flags"] & _mask) > 0
        _flag_rows.append(
            {
                "flag": _name,
                "% of trades": 100 * _hit.mean(),
                "% of shares": 100 * _ta["size"][_hit].sum() / _ta["size"].sum(),
            }
        )
    mo.vstack(
        [
            mo.md(
                "**Table 1 · where trades print against IEX's own quote just before them** "
                "(% of each stock's trades after the open)"
            ),
            table(_where, digits=0),
            mo.md(
                f"**Table 2 · the flags of all {len(_ta):,} trades in "
                f"{_ta['symbol'].nunique():,} symbols after the open**"
            ),
            table(pd.DataFrame(_flag_rows).set_index("flag"), digits=1),
        ]
    )
    return (signed,)


@app.cell
def _(
    below,
    et,
    focus,
    go,
    iex,
    make_subplots,
    mo,
    np,
    open_ns,
    plots,
    qf,
    signed,
):
    _sg = signed[focus.value]
    _sg = _sg.assign(time=et(_sg["t_ns"]))
    _fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.08,
        subplot_titles=("IEX best bid / ask and the trades", "cumulative signed shares"),
    )
    _q = qf[focus.value]
    _qa = _q[(_q["t_ns"] >= open_ns) & (_q["t_ns"] <= iex["last_ns"])]
    for _col, _name, _color in (("bid", "best bid", "#2a78d6"), ("ask", "best ask", "#eb6834")):
        _s = plots._thin(_qa.set_index("t_ns")[_col])
        _fig.add_trace(
            go.Scatter(
                x=et(_s.index),
                y=_s.to_numpy(),
                mode="lines",
                name=_name,
                line={"color": _color, "width": 1, "shape": "hv"},
            ),
            1,
            1,
        )
    for _side, _name, _color, _symbol in (
        (1, "trade labelled a buy", "#2a78d6", "triangle-up"),
        (-1, "trade labelled a sell", "#eb6834", "triangle-down"),
    ):
        _t = _sg[_sg["side"] == _side]
        _fig.add_trace(
            go.Scatter(
                x=_t["time"],
                y=_t["price"],
                mode="markers",
                name=_name,
                marker={
                    "symbol": _symbol,
                    "size": 4 + 2 * np.sqrt(_t["size"] / 100),
                    "color": _color,
                },
                customdata=np.stack([_t["size"], _t["where"]], axis=-1),
                hovertemplate=(
                    "%{y:.2f} x %{customdata[0]} shares<br>%{customdata[1]}<extra></extra>"
                ),
            ),
            1,
            1,
        )
    _fig.add_trace(
        go.Scatter(
            x=_sg["time"],
            y=(_sg["side"] * _sg["size"]).cumsum(),
            mode="lines",
            name="cumulative signed shares (buys minus sells)",
            line={"color": plots.SERIES[4], "shape": "hv"},
        ),
        2,
        1,
    )
    _title = f"Fig 6 · {focus.value}'s trades on IEX, labelled by the quote before them"
    plots._base(_fig, title=_title, height=560)
    mo.vstack([mo.hstack([focus], justify="start"), below(_fig, -0.1)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.5 · Market-wide status: halts, short-sale restriction, the opening, retail interest

    These messages are published for **every** symbol, so one file gives a cross-section of about
    13,000 names, not four. Each symbol gets a status "spin" before 08:00, then a message whenever
    something changes:

    * **Short sale price test (Reg SHO rule 201)**: in effect once a stock has fallen 10% from its
      previous close; the restriction lasts the rest of that day and the next. Names already
      restricted at the start of the day are recent big losers; names switched on after 09:30:00 are
      stocks trading at least 10% below yesterday's close right now, stamped to the millisecond.
    * **Trading status**: `H` = halted across all US markets (reason `NA` for names not listed on
      IEX), `T` = trading. A halt and its release are events.
    * **Operational halt**: halted on IEX only. **Security event `O`**: IEX has finished opening the
      stock's book.
    * **Retail liquidity indicator**: `A` / `B` / `C` when IEX holds retail-eligible buy / sell /
      both-side interest, blank otherwise. It says that interest exists; not how much, not at what
      price.

    **Use:** event lists the instant the market opens (who gaps down 10%, who is halted and when it
    is released), a "do not trade" filter, and a cross-sectional retail-attention measure.
    **Limit:** only a few dozen events in the first minutes, so a test needs many days; and the
    retail flag's link to returns is not established.
    """)
    return


@app.cell
def _(mo, np, open_ns, pd, st, sys_ns, table):
    def _phase(t_ns):
        """Which part of the day each timestamp falls in."""
        names = ["before 08:00 (the spin)", "08:00-09:30"]
        return np.select([t_ns < sys_ns, t_ns < open_ns], names, "from 09:30")

    kinds = {
        "halted across US markets (H)": st["trading_status"].query("status == 'H'"),
        "halted on IEX only (O)": st["operational_halt"].query("status == 'O'"),
        "short-sale restriction in effect": st["short_sale"].query("in_effect == 1"),
        "retail interest flag set (A / B / C)": st["retail_interest"].query("indicator != ' '"),
        "IEX opening complete (E O)": st["security_event"].query("event == 'O'"),
    }
    _rows = []
    for _label, _d in kinds.items():
        _counts = pd.Series(_phase(_d["t_ns"].to_numpy())).value_counts().to_dict()
        _rows.append({"messages": _label, **_counts, "distinct symbols": _d["symbol"].nunique()})
    _columns = ["before 08:00 (the spin)", "08:00-09:30", "from 09:30", "distinct symbols"]
    _counts_table = pd.DataFrame(_rows).set_index("messages")
    _counts_table = _counts_table.reindex(columns=_columns, fill_value=0)

    _ssr, _tstat = st["short_sale"], st["trading_status"]
    changes = pd.concat(
        [
            _ssr[(_ssr["in_effect"] == 1) & (_ssr["t_ns"] >= sys_ns)].assign(
                event="short-sale restriction switched on"
            ),
            _tstat[_tstat["t_ns"] >= sys_ns].assign(
                event=lambda d: np.where(d["status"] == "H", "trading halted", "trading (status T)")
            ),
        ]
    )[["t_ns", "symbol", "event"]].sort_values("t_ns").reset_index(drop=True)
    changes["seconds after the open"] = (changes["t_ns"] - open_ns) / 1e9
    mo.vstack(
        [
            mo.md("**Table · status messages by part of the day** (`distinct symbols` = how many "
                  "different stocks they concern)"),
            table(_counts_table.fillna(0).astype(int), rows=6),
            mo.md(f"**Status changes after 08:00** ({len(changes)}), by event:"),
            table(changes["event"].value_counts().rename_axis("event").to_frame("count"), rows=6),
        ]
    )
    return changes, kinds


@app.cell
def _(changes, iex, mo, open_ns, table):
    _post = iex["trades_all"]
    _post = _post[_post["t_ns"] >= open_ns].sort_values("t_ns")
    _seen = _post.groupby("symbol").agg(
        iex_trades=("price", "size"), first_price=("price", "first"), last_price=("price", "last")
    )
    _newly = (
        changes[changes["event"] == "short-sale restriction switched on"]
        .merge(_seen, left_on="symbol", right_index=True, how="left")
        .reset_index(drop=True)
    )
    _newly["change since the open, %"] = (_newly["last_price"] / _newly["first_price"] - 1) * 100
    _show = [
        "symbol",
        "seconds after the open",
        "iex_trades",
        "first_price",
        "last_price",
        "change since the open, %",
    ]
    _many = _newly[_newly["iex_trades"] >= 3]
    _at_open = (_newly["seconds after the open"].abs() < 1).sum()
    _traded = _newly["iex_trades"].notna().sum()
    _median = _many["change since the open, %"].median()
    _minutes = (iex["last_ns"] - open_ns) / 6e10
    mo.vstack(
        [
            mo.md(
                f"**{len(_newly)} stocks were switched to short-sale restriction after 08:00** "
                f"({_at_open} within a second of 09:30:00). "
                f"{_traded} traded on IEX after the open; the first ones:"
            ),
            table(_newly.sort_values("seconds after the open").head(12)[_show], rows=12),
            mo.md(
                f"Of the {len(_many)} with at least 3 IEX trades, the median price change from the "
                f"first to the last IEX trade in the file is {_median:+.1f}% (a description of a "
                f"handful of names over ~{_minutes:.0f} minutes, not a result)."
            ),
        ]
    )
    return


@app.cell
def _(DAY, go, iex, kinds, mo, open_ns, pd, plots, st, table):
    _rli = st["retail_interest"]
    _flagged = _rli[_rli["indicator"] != " "]
    _final = _rli.sort_values("t_ns").groupby("symbol").tail(1)
    _rli_table = (
        pd.DataFrame(
            {
                "flag messages in the file": _flagged["indicator"].value_counts(),
                "symbols flagged at the end of the file": _final["indicator"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
        .rename_axis("indicator")
    )
    _top = _flagged.groupby("symbol").size().sort_values(ascending=False).head(8)
    _opened = st["security_event"].query("event == 'O'")
    fast = (_opened["t_ns"] - open_ns < 10**6).mean()
    _late = _opened[_opened["t_ns"] - open_ns > 10**9]["symbol"].tolist()

    _n_sym = {label: d["symbol"].nunique() for label, d in kinds.items()}
    scale = {
        "symbols in the start-of-day status spin": st["trading_status"]["symbol"].nunique(),
        "symbols with a book update in the file": iex["symbols"],
        "short-sale restriction at some point": _n_sym["short-sale restriction in effect"],
        "retail flag set at some point": _flagged["symbol"].nunique(),
        "halted across US markets at some point": _n_sym["halted across US markets (H)"],
        "halted on IEX only": _n_sym["halted on IEX only (O)"],
        "IEX-listed (security directory)": st["security_directory"]["symbol"].nunique(),
    }
    _fig = go.Figure(
        go.Bar(
            x=list(scale.values()),
            y=list(scale),
            orientation="h",
            marker_color=plots.SERIES[0],
            text=[f"{v:,}" for v in scale.values()],
            textposition="outside",
        )
    )
    _fig.update_xaxes(type="log", title="symbols (log scale)")
    _fig.update_yaxes(autorange="reversed")
    _title = f"Fig 7 · how many symbols each market-wide field covers ({DAY})"
    plots._base(_fig, title=_title, height=380)
    mo.vstack(
        [
            mo.md(
                "**Table · the retail interest flag**: messages in the file, and the flag each "
                "symbol carried at the end of the file (blank = none)"
            ),
            table(_rli_table, rows=6),
            mo.md(
                "Most flag changes: " + ", ".join(f"{k} ({v})" for k, v in _top.items()) + ". "
                f"IEX opened {fast:.2%} of {len(_opened):,} books within 1 ms of 09:30:00; "
                f"later: {_late}."
            ),
            _fig,
        ]
    )
    return fast, scale


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **What the auction and official-price messages hold.** They exist only for IEX-listed
    securities, and the security directory (also IEX-listed only) says who that is. The tables below
    show the directory and what the auction and official-price messages say. If the directory lists
    only test securities (the `test_security` column), the auction information, the imbalance fields
    and the official prices carry nothing about real stocks. The auction schedule (information once
    a second from two minutes before the open) and the fields (paired shares, imbalance side and
    size, indicative price, collars) are what a closing-imbalance signal would use on an exchange
    with listed stocks.
    """)
    return


@app.cell
def _(mo, st, table):
    _directory = st["security_directory"].assign(test_security=lambda d: (d["flags"] & 0x80) > 0)
    _items = [
        mo.md("**Table · the security directory (IEX-listed securities)**"),
        table(
            _directory[["symbol", "test_security", "round_lot", "prior_close", "luld_tier"]], rows=6
        ),
    ]
    if len(st["auction"]):
        _auctions = (
            st["auction"]
            .groupby(["symbol", "auction_type"])
            .agg(
                messages=("t_ns", "size"),
                paired_shares_max=("paired_shares", "max"),
                imbalance_shares_max=("imbalance_shares", "max"),
                imbalance_sides=("imbalance_side", lambda s: "".join(sorted(set(s)))),
            )
        )
        _items += [mo.md("**Table · auction information messages**"), table(_auctions, rows=6)]
    if len(st["official_price"]):
        _official = st["official_price"]
        _prices = dict(zip(_official["symbol"], _official["price"], strict=True))
        _items.append(mo.md(f"Official prices sent: {_prices}."))
    mo.vstack(_items)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · In context: from data to a strategy

    The table sets each piece of information against what a strategy could do with it, where the
    idea comes from in the literature (pointers, not checked here) and what limits it with this
    data.
    """)
    return


@app.cell
def _(pd, table):
    _rows = [
        (
            "Liquidity state (§4.1)",
            "book updates",
            "cost estimate, entry filter, execution timing, a regime variable",
            "—",
            "IEX's own book: thin, often wider than the market's",
        ),
        (
            "Queue imbalance, microprice (§4.2)",
            "sizes at the best prices",
            "direction of the next price move, over seconds",
            "Lipton, Pesavento & Sotiropoulos 2013; Cao, Hansch & Wang 2009; Stoikov 2018",
            "fades in seconds; IEX is a few percent of volume; spread and fees decide",
        ),
        (
            "Order-flow imbalance (§4.3)",
            "changes of the best bid / ask",
            "explains the same interval's price change; input to impact and forecast models",
            "Cont, Kukanov & Stoikov 2014",
            "same-interval, not predictive; strength depends on bucket length and stock",
        ),
        (
            "Signed trade flow, trade flags (§4.4)",
            "trade reports plus the quote before",
            "buy / sell pressure, urgency (sweeps), odd-lot activity, activity",
            "Lee & Ready 1991; Chordia & Subrahmanyam 2004; "
            "Easley, López de Prado & O'Hara 2012 (disputed)",
            "no aggressor side, the sign is inferred; many prints inside the spread say "
            "little about direction; IEX trades only",
        ),
        (
            "Market-wide status (§4.5): short-sale restriction, halts, opening",
            "status, short-sale and security-event messages",
            "whole-market event lists at 09:30:00: who opened 10% down, who is halted and "
            "when released; a 'do not trade' filter",
            "Corwin & Lipson 2000 (halts); SEC Rule 201; compare this repo's gap_and_go",
            "a few dozen events in the first minutes: needs many days to test",
        ),
        (
            "Retail interest flag (§4.5)",
            "retail liquidity indicator",
            "breadth of retail buy vs sell interest across names in the first minutes",
            "Boehmer, Jones, Zhang & Zhang 2021 measure retail flow from other data",
            "says interest exists on a side, not how much or where; flickers; "
            "no known link to returns",
        ),
        (
            "Auction, official price (§4.5)",
            "auction information, official price",
            "opening / closing imbalance signals",
            "—",
            "IEX-listed securities only: test symbols here, nothing for real stocks",
        ),
    ]
    _text = [
        "information",
        "made from",
        "what a strategy could do with it",
        "literature",
        "limits here",
    ]
    table(
        pd.DataFrame(_rows, columns=_text),
        rows=8,
        wrapped=_text,
        widths={
            "information": 190,
            "made from": 150,
            "what a strategy could do with it": 280,
            "literature": 260,
            "limits here": 320,
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where I would look first (a judgement, not a result).** (1) The **status fields at the open**:
    they are the only thing here that no bar data has, they span the whole market (thousands of
    names, so every kind of stock, though only a handful of events per day), and they sit in the
    first seconds of the file, which is the cheap part. (2) **Opening-minute book and trade
    features** (spread, depth, signed flow over the first 1-3 minutes) against the following 5-30
    minutes' returns, which the repo's 1-minute bars in the database could supply for about 1,750
    names. (3) Tick-level imbalance and OFI strategies last: they need full-day files, they see only
    IEX's thin book, and seconds-long edges meet the spread. The next table puts numbers on the
    cost.
    """)
    return


@app.cell
def _(decode_s, deep, got, iex, mo, n_msgs, open_ns, pd, table):
    _secs_after = (iex["last_ns"] - open_ns) / 1e9
    _n_after = sum(
        n
        for b, c in iex["rhythm"].items()
        if b * iex["bucket_s"] * 10**9 >= open_ns
        for n in c.values()
    )
    _mb = got["bytes"] / 1e6
    _mb_per_min = _mb * _n_after / n_msgs / _secs_after * 60
    _last_year = deep.sort_values("day").tail(250)
    _whole_day_min = got["full_bytes"] / got["bytes"] * decode_s / 60
    _feasibility = pd.DataFrame(
        [
            ("DEEP file of the day, all of it", f"{got['full_bytes'] / 1e9:.1f} GB", "IEX's index"),
            ("fetched, and the time it reaches", f"{_mb:.0f} MB, {_secs_after:.0f} s", "measured"),
            (
                "gzip volume right after the open",
                f"{_mb_per_min:.0f} MB per minute",
                "estimate (a)",
            ),
            (
                "first 3 minutes of the session, 250 days",
                f"{3 * _mb_per_min * 250 / 1000:.0f} GB",
                "estimate (a)",
            ),
            ("decode speed here", f"{n_msgs / decode_s / 1e6:.2f} M messages/s", "measured"),
            ("decoding a whole day here", f"{_whole_day_min:.0f} minutes", "estimate"),
            (
                "a year of DEEP files (last 250 days listed)",
                f"{_last_year['gb'].sum() / 1000:.1f} TB",
                "IEX's index",
            ),
        ],
        columns=["what", "value", "how"],
    ).set_index("what")
    mo.vstack(
        [
            table(_feasibility, rows=8),
            mo.md("(a) estimate: assumes the gzip bytes follow the message counts"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Checks
    """)
    return


@app.cell
def _(STOCKS, fast, focus, iex, liq, pd, scale, sens, signed, st, table):
    _sg = signed[focus.value]
    _checks = [
        (
            "the file decoded: book updates, trades and status messages all present",
            bool(iex["types"].get("8") and iex["types"].get("T") and iex["types"].get("H")),
        ),
        (
            "IEX-listed securities are only test symbols (so auctions carry no real stock)",
            bool(((st["security_directory"]["flags"] & 0x80) > 0).all()),
        ),
        (
            "the status spin covers more than 10,000 symbols",
            scale["symbols in the start-of-day status spin"] > 10_000,
        ),
        ("IEX opened at least 99% of the books within 1 ms of 09:30:00", bool(fast >= 0.99)),
        (
            "no crossed book in any rebuilt stock",
            all(liq[s]["counts"].get("crossed", 0) == 0 for s in STOCKS),
        ),
        (
            "a quote exists before at least 99% of the chosen stock's trades",
            bool((_sg["where"] != "no quote").mean() >= 0.99),
        ),
        (
            "OFI and the same-bucket mid change are positively related in at least 3 of the stocks",
            bool((pd.DataFrame(sens).T["0.1 s"] > 0).sum() >= min(3, len(STOCKS))),
        ),
    ]
    table(pd.DataFrame(_checks, columns=["check", "ok"]), rows=10)
    return


if __name__ == "__main__":
    app.run()
