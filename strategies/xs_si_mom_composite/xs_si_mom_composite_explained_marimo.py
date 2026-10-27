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
    # xs_si_mom_composite · how the strategy works, and how it did from 1 January 2025

    **The idea in one paragraph.** Each day the strategy gives every one of the most traded US
    stocks (the top 2000 in the standard setting, version B) two scores:
    - **Short interest.** How little short sellers bet against the stock: FINRA's days-to-cover.
      Few short sellers is good, a crowded short is bad.
    - **Momentum.** How steadily the stock beat what its market exposure explains over the past
      year, leaving out the last month.

    It averages the two ranks. It **buys the 15 best** and **sells short the 15 worst**, with the
    same dollars on each side (45% of the account each). It holds each one until its rank falls
    back, and trades at the next day's open. Nothing in it was tuned.

    **What this notebook shows, on real data from 2025-01-02 to the last bar in the database:**

    1. **The universe:** which stocks can be picked, re-formed every six months.
    2. **How one stock is scored,** step by step: days-to-cover → its rank; one year of returns →
       its momentum rank; the two ranks → the combined rank → the book's entry and exit rules.
    3. **The stock-picking assessment on any day you choose** (the last day = what the strategy
       holds now): every eligible stock ranked, the 15 + 15 picks, and the orders for the next open.
    4. **How the book performed** against buying and holding the same universe, and against SPY.
    5. **A timeline of every position:** when it was opened and closed, its P&L and why it closed.
    6. **One position up close:** why it was bought, what it did, why it was sold.

    Nothing here is tuned or selected: the rules are `config.yaml` as shipped. The first run reads
    the database (about 15 minutes for the top 2000); later runs read a cached copy in `var/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(df, rows=15):
        """A DataFrame as a marimo table; a named index becomes a column."""
        return mo.ui.table(df.reset_index() if df.index.name else df, selection=None,
                           page_size=rows)

    return (table,)


@app.cell
def _():
    import importlib.util
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine
    from trading_simple.plots import _GRID as GRID
    from trading_simple.plots import _LONG as LONG
    from trading_simple.plots import _LOSS as LOSS
    from trading_simple.plots import _MUTED as MUTED
    from trading_simple.plots import _SHORT as SHORT
    from trading_simple.plots import _WIN as WIN
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as layout

    _spec = importlib.util.spec_from_file_location(
        "xs_si_mom_composite_explain",
        strategies.STRATEGIES_DIR / "xs_si_mom_composite" / "explain.py")
    ex = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ex)
    strat = strategies.load("xs_si_mom_composite")
    settings = load_settings()
    engine = make_engine(settings)
    START, WARMUP = "2025-01-01", "2022-01-01"     # first traded day; first bar loaded
    CACHE = Path(strategies.STRATEGIES_DIR).parent / "var" / "xs_si_mom_composite" / "explain"

    def finish(fig, title, height=420):
        """The repo's chart style, with the legend UNDER the chart (about 40 px below it)."""
        layout(fig, title=title, height=height)
        fig.update_layout(legend={"orientation": "h", "yanchor": "top",
                                  "y": -40 / max(height - 150, 100), "xanchor": "left", "x": 0.0},
                          margin={"l": 60, "r": 20, "t": 50, "b": 100})
        return fig

    def days(index):
        """Daily timestamps as date strings (a date axis that no browser time zone shifts)."""
        return pd.DatetimeIndex(index).strftime("%Y-%m-%d")

    return (CACHE, GRID, LONG, LOSS, MUTED, SERIES, SHORT, START, WARMUP, WIN, days,
            engine, ex, finish, go, make_subplots, np, pd, settings, strat)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Load the data and run the strategy

    - **Prices:** daily bars, split- and dividend-adjusted, from 2022. Momentum needs about two
      years of history before the first traded day.
    - **Short interest:** FINRA's days-to-cover, usable the day after FINRA publishes it.
    - **Share counts:** SEC filings, used for the "too crowded to short" cap and the borrow fee.
    - **The universe:** version B, the standard since 2026-10-27: the 2000 most traded stocks.
      Version A is the top 500, where the strategy was built. Both are re-formed every six
      months.
    - **The account:** its size only scales the dollar figures; the returns do not depend on it.
      $33,000 gives the 30 positions of about $1,000 that the cost model prices: $0.005 a share,
      at least $1, plus 3 bp of spread and slippage.
    """)
    return


@app.cell
def _(mo):
    account = mo.ui.number(start=1_000, stop=10_000_000, step=1_000, value=33_000,
                           label="account ($)")
    universe = mo.ui.dropdown(options={"B · top 2000 (standard)": "B", "A · top 500": "A"},
                              value="B · top 2000 (standard)", label="universe")
    refresh = mo.ui.checkbox(label="reload from the database (else the cached copy)")
    load_button = mo.ui.run_button(label="load the data and run the strategy")
    mo.hstack([universe, account, refresh, load_button], justify="start", gap=2)
    return account, load_button, refresh, universe


@app.cell
def _(CACHE, START, WARMUP, account, engine, ex, load_button, mo, pd, refresh, settings,
      strat, universe):
    mo.stop(not load_button.value, mo.md("_Click **load the data and run the strategy**._"))
    _roll, _tag = (ex.ROLL_B, "-B") if universe.value == "B" else (ex.ROLL, "")
    with mo.status.spinner("universe and prices (5-15 minutes the first time) ..."):
        history = ex.cached(CACHE / f"formations{_tag}-{START}.pkl",
                            lambda: ex.formations(engine, START, pd.Timestamp.today(), _roll),
                            refresh=refresh.value)
        panel = ex.cached(CACHE / f"panel{_tag}-{WARMUP}.pkl",
                          lambda: ex.load(engine, settings, history, warmup=WARMUP),
                          refresh=refresh.value)
    with mo.status.spinner("scoring every stock and running the book ..."):
        w = ex.walk(strat, panel)
        res = ex.backtest(strat, panel, settings, START, account.value)
        bench = ex.member_benchmark(res["window"], history)
        spy = ex.spy_returns(engine, settings, res["returns"].index)
        pos = ex.positions(res, w)
        pnl = ex.legs(res)
    LAST = res["returns"].index[-1]
    _si_last = w["dtc"].dropna(how="all").index[-1]
    mo.md(f"""
    **Loaded:**
    - {panel.close.shape[1]} stocks (every name that was ever in the universe);
    - prices {panel.close.index[0].date()} → **{LAST.date()}**;
    - the newest short-interest value on {_si_last.date()};
    - the book traded from {res['returns'].index[0].date()}: {len(pos)} positions.

    Engine messages: {res['warnings'] or 'none'}.

    The last price bar is the newest the database holds. Refresh prices and short interest
    (`notebooks/03_altdata_update.py`, `tsim backfill`) before reading §3 as today's picks.
    """)
    return LAST, bench, history, panel, pnl, pos, res, spy, w


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The universe: which stocks can be picked

    **Formation.** Every six months (1 January and 1 July) the universe is re-formed from the six
    months before:
    - the **2000 stocks with the highest median daily dollar volume** (500 in version A);
    - with a median price of at least $5, at least 100 sessions and at least $1M traded a day
      ($50M in version A);
    - common stocks only, so ETFs are left out.

    **Point in time.** A stock counts only while it is a member. When it drops out at a
    re-formation, the strategy sells (or covers) it at the next open.

    **Eligible on a given day.** A member is eligible when, that day:
    - it closed at $5 or more;
    - FINRA has a days-to-cover for it (masked once it is 45 days old);
    - SEC has a share count.

    Only eligible stocks are ranked.

    **Why these limits** (the checks are in RESEARCH_LOG §13.4 and §13.5):
    - **Why 2000, when it was built on 500.** The next three groups of 500 (ranked 501-2000)
      were kept apart as tests. From 2022 to 2026 the strategy beat the market on all of them:
      alpha +5% to +12% a year, still positive with every fill 0.5% worse.
    - **One book on all 2000** had the same alpha as the top 500 within noise (+14.4% against
      +12.8% a year) and lower volatility (14.6% against 16.3%). That made it the standard.
    - **The smaller stocks cost a little more to trade:** about 6.4 bp spread for ranks
      1501-2000 against 2.5 bp in the top 500.
    - **$5:** prices move in 1-cent steps, and one step is 0.25% of a $4 stock. Stocks under $5
      also move about twice as much as large ones, and a broker must hold the full value of a
      short below $5, against 30% above it. Without the rule, the cheap stocks added only about
      0.3 points a year. One flaw to know: the database stores split-adjusted prices, so "$5"
      is judged on today's split history. NVDA was dropped in 2019 at a stored $3.34, while it
      actually traded near $134.
    """)
    return


@app.cell
def _(SERIES, days, finish, go, history, mo, panel, pd, res, table, w):
    _rows = []
    _prev = None
    for _f, _names in sorted(history.items()):
        _s = set(_names)
        _rows.append({"formed": _f.date(), "members": len(_s),
                      "joined": len(_s - _prev) if _prev else "",
                      "left": len(_prev - _s) if _prev else ""})
        _prev = _s
    _idx = res["window"].close.index
    _member = panel.features["member"].reindex(_idx).sum(axis=1)
    _elig = w["eligible"].reindex(_idx).sum(axis=1)
    _fig = go.Figure([
        go.Scatter(x=days(_idx), y=_member, name="members of the universe", mode="lines",
                   line={"color": SERIES[0]}),
        go.Scatter(x=days(_idx), y=_elig, name="eligible (≥ $5, days-to-cover and shares known)",
                   mode="lines", line={"color": SERIES[1]})])
    finish(_fig, "Universe size per day", 320)
    mo.vstack([table(pd.DataFrame(_rows)), _fig,
               mo.md("The blue line counts members and the green line eligible names. The gap "
                     "is members without a days-to-cover or share count, or below $5 that day.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · How one stock is scored

    Pick a stock: the list holds every stock the strategy traded, biggest P&L first. The three
    charts below follow it through the strategy's three steps:
    1. its short-interest rank;
    2. its momentum rank;
    3. its combined rank and the book's entry and exit rules.
    """)
    return


@app.cell
def _(mo, pos):
    _by = pos.groupby("symbol").agg(pnl=("P&L $", "sum"), n=("side", "size"),
                                    sides=("side", lambda s: "/".join(sorted(set(s)))))
    _by = _by.reindex(_by["pnl"].abs().sort_values(ascending=False).index)
    _options = {f"{s} · {r.sides} · {r.n} position(s) · P&L ${r.pnl:,.0f}": s
                for s, r in _by.iterrows()}
    stock = mo.ui.dropdown(options=_options, value=next(iter(_options)), label="stock")
    mo.hstack([stock], justify="start")
    return (stock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1 · Short interest → the short-interest rank

    - **Days-to-cover** = shares sold short ÷ the average daily volume: how many days of normal
      trading the short sellers would need to buy their shares back. FINRA publishes it twice a
      month, so it moves in steps.
    - **It arrives late.** FINRA counts the positions on a settlement date (mid-month and month
      end) and publishes them about 8 business days later; the strategy uses them from the next
      session. So the newest value is 9-20 business days old, about three weeks on a typical
      day. The dotted line in the chart below dates each value by its settlement date, so the
      gap between the dotted and the solid steps is that delay.
    - **The strategy averages it over the last 42 sessions,** about two months, so one report
      does not jump the score.
    - **It then ranks that average** against the other eligible stocks of similar liquidity: five
      dollar-volume groups, so a giant is not compared with a small cap.
    - **Low days-to-cover gets a high rank** (close to 1): short sellers avoid the stock, which
      is good for a long. High days-to-cover gets a low rank (close to 0): a crowded short.
    """)
    return


@app.cell
def _(LOSS, SERIES, START, GRID, WIN, days, engine, ex, finish, make_subplots, mo, panel, pd,
      stock, w):
    _s = stock.value
    _from = pd.Timestamp(START, tz="UTC") - pd.DateOffset(months=6)
    _dtc, _avg = w["dtc"][_s].loc[_from:], w["dtc_avg"][_s].loc[_from:]
    _rank = w["si"][_s].loc[_from:]
    _set = ex.settlements(engine, _s)
    _set = _set[(_set["settled"] >= _from) & (_set["settled"] <= _dtc.index[-1])]
    _fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.45, 0.3, 0.25],
                         subplot_titles=("daily candles (adjusted)", "days-to-cover",
                                         "short-interest rank (1 = least shorted)"))
    _fig.add_candlestick(x=days(_dtc.index), open=panel.open_[_s].loc[_from:],
                         high=panel.high[_s].loc[_from:], low=panel.low[_s].loc[_from:],
                         close=panel.close[_s].loc[_from:], name="daily candle (adjusted)",
                         increasing={"line": {"color": WIN}, "fillcolor": WIN},
                         decreasing={"line": {"color": LOSS}, "fillcolor": LOSS}, row=1, col=1)
    _fig.add_scatter(x=days(_set["settled"]), y=_set["dtc"],
                     name="days-to-cover at its settlement date",
                     mode="lines", line={"color": SERIES[2], "shape": "hv", "dash": "dot"},
                     row=2, col=1)
    _fig.add_scatter(x=days(_dtc.index), y=_dtc,
                     name="days-to-cover as the strategy knew it",
                     mode="lines", line={"color": SERIES[2], "shape": "hv"}, row=2, col=1)
    _fig.add_scatter(x=days(_avg.index), y=_avg, name="its 42-session average (what is ranked)",
                     mode="lines", line={"color": SERIES[0], "width": 2.5}, row=2, col=1)
    _fig.add_scatter(x=days(_rank.index), y=_rank,
                     name="short-interest rank in its liquidity group", mode="lines",
                     line={"color": SERIES[1]}, row=3, col=1)
    _fig.add_hline(y=0.5, line={"color": GRID, "dash": "dot"}, row=3, col=1)
    _fig.update_xaxes(rangeslider_visible=False, rangebreaks=[{"bounds": ["sat", "mon"]}])
    _fig.update_yaxes(range=[0, 1.02], row=3, col=1)
    finish(_fig, f"{_s}: price, short interest → rank", 760)
    mo.vstack([_fig, mo.md(
        "- **Top:** the daily candles, green for a day that closed above its open and red for "
        "one that closed below.\n"
        "- **Middle:** the yellow steps are the days-to-cover. Dotted, each value sits at its "
        "settlement date, when the short positions were counted. Solid, it sits where the strategy "
        "could first use it, about two weeks later. The thick blue line is the 42-session average "
        "that the strategy ranks.\n"
        "- **Bottom:** the green line is the rank, high when the stock is lightly shorted "
        "compared with similar stocks. The dotted line marks the middle (0.5).")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2 · One year of returns → the momentum rank

    Raw momentum ("it went up") is mostly the market. The strategy uses **residual momentum**:
    1. **The market part.** Each day's return of the stock minus β × the equal-weight market's
       return, with β the stock's sensitivity to the market over the last 252 sessions. What is
       left (the residual) is the part the market cannot explain.
    2. **The window.** The residuals are summed over months t−12 … t−1. **The last month is
       skipped.** Why, and whether it matters, is explained below the chart.
    3. **The score.** The sum is divided by the residuals' standard deviation, so a steady climb
       scores higher than one lucky jump.
    4. **The rank.** On the last session of each month every eligible stock is ranked on that
       score (1 = the strongest), and the rank holds for the whole next month.

    Choose a month-end to see the calculation behind that month's rank.
    """)
    return


@app.cell
def _(START, mo, pd, w):
    _ends = w["rebalance"][w["rebalance"]].index
    _ends = [d for d in _ends if d >= pd.Timestamp(START, tz="UTC") - pd.Timedelta(days=31)]
    month_end = mo.ui.dropdown(options={d.strftime("%Y-%m-%d"): d for d in _ends},
                               value=_ends[-1].strftime("%Y-%m-%d"), label="month-end")
    mo.hstack([month_end], justify="start")
    return (month_end,)


@app.cell
def _(SERIES, GRID, days, finish, make_subplots, mo, month_end, np, stock, w):
    _s, _t = stock.value, month_end.value
    _idx = w["returns"].index
    _i = _idx.get_loc(_t)
    _win = _idx[max(0, _i - 251):_i + 1]
    _r = w["returns"][_s].reindex(_win).fillna(0.0)
    _m = w["market"].reindex(_win).fillna(0.0)
    _beta = float(w["mom_beta"][_s].loc[_t])
    _e = _r - _beta * _m
    _used = _win[:-21]
    _sd = float(_e.loc[_used].std())
    _score = float(w["mom_value"][_s].loc[_t])
    _rank = float(w["mom"][_s].loc[_t])
    _score0 = float(w["mom_value_noskip"][_s].loc[_t])
    _rank0 = float(w["mom_noskip"][_s].loc[_t])
    # the month this rank is used: from the session after _t to the next month-end (or last bar)
    _later = w["rebalance"][w["rebalance"]].index
    _later = _later[_later > _t]
    _end = _later[0] if len(_later) else _idx[-1]
    _span = _idx[(_idx >= _win[0]) & (_idx <= _end)]
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         row_heights=[0.6, 0.4],
                         subplot_titles=(f"the 12 months before {_t.date()} (cumulative, sum of "
                                         "daily returns)", "momentum rank, set at each "
                                         "month-end and held for the next month"))
    for _y, _name, _c, _dash in ((_r.cumsum(), "the stock", SERIES[0], None),
                                 ((_beta * _m).cumsum(), f"β × market (β = {_beta:.2f})",
                                  SERIES[2], "dot"),
                                 (_e.cumsum(), "residual = stock − β × market", SERIES[3], None)):
        _fig.add_scatter(x=days(_y.index), y=100 * _y, name=_name, mode="lines",
                         line={"color": _c, "dash": _dash,
                               "width": 2.5 if _c == SERIES[3] else 1.5},
                         row=1, col=1)
    _daily = w["mom"][_s].reindex(_span)
    _fig.add_scatter(x=days(_span), y=_daily, name="momentum rank (1 = strongest)",
                     mode="lines", line={"color": SERIES[4], "shape": "hv"}, row=2, col=1)
    _fig.add_scatter(x=[days([_t])[0]], y=[_rank], name="the chosen month-end",
                     mode="markers", marker={"color": SERIES[5], "size": 12}, row=2, col=1)
    for _x0, _x1, _c, _name in ((_win[-21], _win[-1], GRID, "skipped last month (not scored)"),
                                (_t, _end, SERIES[4], "the month this rank is used")):
        _fig.add_vrect(x0=days([_x0])[0], x1=days([_x1])[0], fillcolor=_c,
                       opacity=0.6 if _c == GRID else 0.12, line_width=0, layer="below",
                       row="all", col=1)
        _fig.add_scatter(x=[days([_x0])[0]], y=[None], mode="markers", name=_name,
                         marker={"color": _c, "symbol": "square", "size": 12,
                                 "opacity": 0.6 if _c == GRID else 0.3}, row=1, col=1)
    _fig.update_xaxes(range=[days(_span)[0], days(_span)[-1]])
    _fig.update_yaxes(title_text="%", row=1, col=1)
    _fig.update_yaxes(range=[0, 1.02], row=2, col=1)
    finish(_fig, f"{_s}: residual momentum at {_t.date()}", 560)
    mo.vstack([_fig, mo.md(f"""
    - **The lines.** Blue is what the stock returned. Dotted yellow is what its market exposure
      alone would have given (β × the equal-weight market). Thick green is the difference: the
      residual.
    - **This month-end.** The residual over the scored eleven months summed to
      {100 * float(_e.loc[_used].sum()):+.1f}%, against a daily standard deviation of
      {100 * _sd:.2f}%.
    - **The score and rank.** The score is that sum divided by the standard deviation:
      **{_score:+.2f}**. That ranks **{_rank:.2f}** among the eligible stocks (purple line, red
      dot).
    - **Had the skipped month counted,** its residual of
      {100 * float(_e.loc[_win[-21:]].sum()):+.1f}% would have been added. The score would be
      {_score0:+.2f} and the rank **{_rank0:.2f}** instead of {_rank:.2f}.
    - **The shading.** The grey band is the skipped last month. The light purple band is the
      month that uses this rank: it starts at the chosen month-end and runs to the next one.
    - **One time axis.** Both charts share it, so each step in the purple line sits under the
      month-end it was computed at.
    """) if np.isfinite(_score) else mo.md("_No momentum score that month (too little history)._")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Why the last month is skipped

    Stock returns show two patterns that point in opposite directions:
    - **Over several months, trends tend to continue.** A stock that beat the market for most
      of a year tends to keep beating it for a while. This is momentum, and it is what the
      strategy wants to buy.
    - **Over about one month, moves tend to partly reverse.** Part of a sharp move in the last
      few weeks is often temporary. A big buyer pushed the price up, or the news was over-read,
      and some of the move is given back the next month.

    The standard momentum recipe, which this strategy copies, therefore leaves the last month
    out, so the "keep going" signal is not mixed with a "give some back" signal. Nothing is lost
    for good: a month later the window has moved on, and this month is part of the eleven
    scored months.

    **Does it matter here?** The give-back is strongest in small, thinly traded stocks. This
    strategy trades the 500 most traded US stocks, where it is weak. The table checks this on
    the notebook's data. At each month-end it asks how well each ranking lines up with the
    next month's return beyond the market:
    - **The IC** is the correlation between the ranking and the next month's returns. 0 means
      no link and 1 a perfect one. A useful stock signal typically scores only 0.02-0.05.
    - **The t-statistic** shows whether that average could be luck. Below about 2, it could.
    """)
    return


@app.cell
def _(START, ex, mo, table, w):
    _tab = ex.skip_test(w, START)
    _a, _b = _tab["average IC"].iloc[0], _tab["average IC"].iloc[1]
    _last = _tab["average IC"].iloc[2]
    mo.vstack([table(_tab, 5), mo.md(f"""
    - **Skipped vs counted:** {_a:+.3f} against {_b:+.3f}.
    - **The last month on its own:** {_last:+.3f}. A negative number would mean that last
      month's winners give some back, which is the reason for the skip. Around 0 or above means
      that did not happen in these stocks over this period.
    - **How to read it:** about {_tab['months with IC > 0'].iloc[0].split(' of ')[1]}
      month-ends is a short sample, so a gap between the first two rows smaller than about
      0.01-0.02 is noise.
    - **The longer test points the other way.** The full book was run both ways over ten years
      (2016-2026, RESEARCH_LOG §13.3). Counting the month traded about as often, but earned
      about 0.8 points a year LESS alpha on the 500 most traded stocks. On the next 500 it
      earned 1.8 points less. It was ahead in only 9 of 21 half-years. The IC above ranks all
      ~490 stocks, but the book only trades the extremes. So the skip stays.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3 · The combined rank, and when the stock is bought or sold

    1. **The combined rank** = the average of the two ranks (a missing rank counts as 0.5),
       averaged over the last 5 sessions and ranked again among the eligible stocks.
    2. **A long** enters when its combined rank reaches **0.90** (the top 10%) and a long slot is
       free; with more candidates than slots, the best go first. It leaves when the rank falls
       **below 0.80**.
    3. **A short** enters at **0.10** or below (the bottom 10%) and leaves **above 0.20**.
    4. **Other exits:**
       - after **126 sessions** without the rank returning to its entry zone;
       - a short that rises **100%** from its entry (the squeeze stop);
       - a short whose short interest exceeds **30% of its shares**. Shorting crowded stocks is
         the whole short side (that is what a low rank means). The cap removes only the extreme
         end, about 1 in 170 short-zone stock-days in 2025-26. There borrowing is scarce, costly
         and can be recalled; squeezes happen; and the backtest's borrow-fee tiers (at most 10%
         a year) cannot price them. The ratio is also least reliable there: right after a split
         it reads too high for weeks;
       - a stock that leaves the universe or loses its data.
    5. **Execution:**
       - every decision is made at the close and filled at the **next open**;
       - 15 longs and 15 shorts of **3% of the account** each, sized at that open;
       - a long that grows to 6% (twice its slot) is trimmed back to 3%.
    """)
    return


@app.cell
def _(SERIES, START, LONG, MUTED, SHORT, days, finish, make_subplots, mo, panel, pd, pos,
      stock, w):
    _s = stock.value
    _from = pd.Timestamp(START, tz="UTC") - pd.DateOffset(months=1)
    _sc = w["score"][_s].loc[_from:]
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                         row_heights=[0.6, 0.4],
                         subplot_titles=("ranks (1 = best)", "price (adjusted) and the fills"))
    _fig.add_scatter(x=days(_sc.index), y=w["si"][_s].loc[_from:], name="short-interest rank",
                     mode="lines", line={"color": SERIES[1], "width": 1}, opacity=0.7, row=1, col=1)
    _fig.add_scatter(x=days(_sc.index), y=w["mom"][_s].loc[_from:],
                     name="momentum rank (set each month-end)", mode="lines",
                     line={"color": SERIES[4], "width": 1, "shape": "hv"}, opacity=0.7,
                     row=1, col=1)
    _fig.add_scatter(x=days(_sc.index), y=_sc, name="combined rank (5-day average, re-ranked)",
                     mode="lines", line={"color": SERIES[0], "width": 2.5}, row=1, col=1)
    for _y, _label, _c in ((0.9, "long entry 0.90", LONG), (0.8, "long exit 0.80", LONG),
                           (0.2, "short exit 0.20", SHORT), (0.1, "short entry 0.10", SHORT)):
        _fig.add_hline(y=_y, line={"color": _c, "dash": "dash" if "exit" in _label else "solid",
                                   "width": 1}, row=1, col=1)
        _fig.add_scatter(x=[days(_sc.index)[0]], y=[None], mode="lines", name=_label,
                         line={"color": _c, "dash": "dash" if "exit" in _label else "solid"},
                         row=1, col=1)
    _px = panel.close[_s].loc[_from:]
    _fig.add_scatter(x=days(_px.index), y=_px, name="price (adjusted close)", mode="lines",
                     line={"color": MUTED}, row=2, col=1)
    _mine = pos[pos["symbol"] == _s]
    for _, _p in _mine.iterrows():
        _c = LONG if _p["side"] == "long" else SHORT
        _end = _p["exit"] if _p["status"] == "closed" else _px.index[-1]
        for _row in (1, 2):
            _fig.add_vrect(x0=days([_p["entry"]])[0], x1=days([_end])[0], fillcolor=_c,
                           opacity=0.12, line_width=0, row=_row, col=1)
    for _side, _c in (("long", LONG), ("short", SHORT)):
        _m = _mine[_mine["side"] == _side]
        if len(_m):
            _fig.add_scatter(x=days(_m["entry"]), y=_m["entry price"], mode="markers",
                             marker={"symbol": ("triangle-up" if _side == "long"
                                                else "triangle-down"),
                                     "size": 12, "color": _c},
                             name=f"{_side} opened (fill at the open)", row=2, col=1)
            _c2 = _m[_m["status"] == "closed"]
            _fig.add_scatter(x=days(_c2["exit"]), y=_c2["exit price"], mode="markers",
                             marker={"symbol": "x", "size": 11, "color": _c},
                             name=f"{_side} closed", text=_c2["cause"],
                             hovertemplate="%{x}: closed at %{y:.2f}<br>%{text}<extra></extra>",
                             row=2, col=1)
    _fig.add_scatter(x=[days(_sc.index)[0]], y=[None], mode="markers",
                     marker={"color": LONG, "symbol": "square", "opacity": 0.3, "size": 12},
                     name="held long (shaded)", row=1, col=1)
    _fig.add_scatter(x=[days(_sc.index)[0]], y=[None], mode="markers",
                     marker={"color": SHORT, "symbol": "square", "opacity": 0.3, "size": 12},
                     name="held short (shaded)", row=1, col=1)
    _fig.update_yaxes(range=[0, 1.02], row=1, col=1)
    finish(_fig, f"{_s}: combined rank and the book's decisions", 640)
    mo.vstack([_fig, mo.md(
        "- **Upper chart:** the thick blue line is the combined rank. It is built from the thin "
        "green short-interest rank and the thin purple momentum rank.\n"
        "- **The thresholds:** blue lines are the long thresholds (solid entry 0.90, dashed exit "
        "0.80), orange lines the short ones (solid entry 0.10, dashed exit 0.20).\n"
        "- **Shading:** blue spans are held long, orange spans held short.\n"
        "- **Lower chart:** triangles are the fills that opened a position and crosses the fills "
        "that closed it; hover over a cross for the reason.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The stock-picking assessment on one day

    This is the whole decision the strategy makes at one close.
    - **Choose a day:** the default is the last bar, i.e. what the strategy holds now and what it
      would trade at the next open.
    - **Every eligible stock** gets its two ranks and its combined rank.
    - **The 15 + 15 picks:** the 15 longs and 15 shorts are the stocks inside the bands that won
      a slot, and positions already held keep their slots.
    """)
    return


@app.cell
def _(LAST, START, mo, pd):
    day = mo.ui.date(start=pd.Timestamp(START).date(), stop=LAST.date(), value=LAST.date(),
                     label="deciding close")
    mo.hstack([day], justify="start")
    return (day,)


@app.cell
def _(SERIES, LONG, MUTED, SHORT, day, ex, finish, go, mo, table, w):
    a = ex.assessment(w, day.value)
    _at = a.attrs["day"]
    _ranked = a[a["combined rank"].notna()]
    _groups = (("held long", a["book"].eq("long") & a["action at next open"].eq("hold"), LONG,
                "circle", 10),
               ("new long (buy at the next open)", a["action at next open"].str.startswith("buy ("),
                LONG, "star", 15),
               ("held short", a["book"].eq("short") & a["action at next open"].eq("hold"), SHORT,
                "circle", 10),
               ("new short (sell short at the next open)",
                a["action at next open"].str.startswith("sell short"), SHORT, "star", 15),
               ("closed at the next open", a["action at next open"].isin(
                   ["sell (long exit)", "buy back (cover)"]), SERIES[5], "x", 12),
               ("not in the book", a["book"].eq(""), MUTED, "circle", 5))
    _fig = go.Figure()
    for _name, _mask, _c, _sym, _size in _groups:
        _d = _ranked[_mask.reindex(_ranked.index).fillna(False)]
        _fig.add_scatter(x=_d["short-interest rank"], y=_d["momentum rank"], mode="markers",
                         name=f"{_name} ({len(_d)})", text=_d.index,
                         customdata=_d[["combined rank"]],
                         marker={"color": _c, "symbol": _sym, "size": _size,
                                 "opacity": 0.35 if _name == "not in the book" else 0.95},
                         hovertemplate="%{text}<br>short-interest rank %{x:.2f}<br>momentum rank "
                                       "%{y:.2f}<br>combined %{customdata[0]:.2f}<extra></extra>")
    _fig.update_xaxes(title_text="short-interest rank (1 = least shorted)", range=[0, 1.02])
    _fig.update_yaxes(title_text="momentum rank (1 = strongest)", range=[0, 1.02])
    finish(_fig, f"The assessment at the close of {_at.date()}: every eligible stock", 560)
    _orders = a[~a["action at next open"].isin(["", "hold"])]
    _longs = a[a["book"].eq("long")]
    _shorts = a[a["book"].eq("short")]
    _cand_l = _ranked[(_ranked["book"] == "") & (_ranked["combined rank"] >= 0.8)].head(5)
    _cand_s = _ranked[(_ranked["book"] == "")
                      & (_ranked["combined rank"] <= 0.2)].sort_values("combined rank").head(5)
    _cols = ["combined rank", "short-interest rank", "momentum rank", "days-to-cover (avg)",
             "momentum score", "short interest / shares", "action at next open"]
    mo.vstack([
        mo.md(f"""
        **{_at.date()}:**
        - **{len(_ranked)} eligible stocks** were ranked.
        - **The book after this close:** {len(_longs)} longs and {len(_shorts)} shorts.
        - **Orders for the next open:** {len(_orders)}.

        The best stocks sit top right (lightly shorted AND strong momentum) and the worst bottom
        left. Blue marks the longs and orange the shorts; a star is a new position bought or
        sold short at the next open, a red cross a position closed there, and grey dots are not
        held.
        """),
        _fig,
        mo.md("**Orders for the next open**" if len(_orders) else
              "_No orders for the next open: the book is unchanged._"),
        table(_orders[_cols].round(3)) if len(_orders) else mo.md(""),
        mo.hstack([mo.vstack([mo.md("**The longs (held after this close)**"),
                              table(_longs[_cols].round(3))]),
                   mo.vstack([mo.md("**The shorts**"), table(_shorts[_cols].round(3))])]),
        mo.hstack([mo.vstack([mo.md("**Next in line for a long slot** (not held, rank ≥ 0.80)"),
                              table(_cand_l[_cols].round(3), 5)]),
                   mo.vstack([mo.md("**Next in line for a short slot** (not held, rank ≤ 0.20)"),
                              table(_cand_s[_cols].round(3), 5)])]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · How the book performed

    **The comparisons:**
    - **Equal-weight buy-and-hold of the same universe:** at each re-formation it buys every
      member equally and holds until the next one. It pays no costs.
    - **SPY.**

    **Two numbers matter:**
    - **Beta** is how much of the market's move the book carries. Long and short in equal
      dollars still leaves some.
    - **Alpha** is the yearly return left after that market exposure: CAGR − beta × the
      benchmark's CAGR. It is what the stock picking earned.
    """)
    return


@app.cell
def _(bench, ex, mo, res, spy, table):
    _sum = ex.summary(res["returns"], bench, spy)
    _show = _sum.copy()
    for _c in ("total return", "CAGR", "volatility", "max drawdown", "alpha a year"):
        _show[_c] = (100 * _show[_c]).map("{:+.1f}%".format)
    _show["Sharpe"] = _show["Sharpe"].map("{:.2f}".format)
    _show["beta vs equal-weight"] = _show["beta vs equal-weight"].map("{:.2f}".format)
    _show.index.name = "from the first traded day"
    _s = _sum.loc["the strategy"]
    mo.vstack([table(_show, 5), mo.md(f"""
    **Reading it:**
    - The strategy compounded **{100 * _s['CAGR']:+.1f}% a year**, carried a beta of
      **{_s['beta vs equal-weight']:.2f}** to the universe, and earned an alpha of
      **{100 * _s['alpha a year']:+.1f}% a year**.
    - Its worst drawdown was {100 * _s['max drawdown']:.1f}%.
    - A market-neutral book is not built to beat buy-and-hold in a rising market. Its job is to
      earn the alpha whatever the market does.
    """)])
    return


@app.cell
def _(SERIES, account, bench, days, finish, make_subplots, mo, res, spy):
    _curves = {"the strategy": (1 + res["returns"]).cumprod(),
               "equal-weight buy-and-hold of the universe": (1 + bench).cumprod(),
               "SPY": (1 + spy).cumprod()}
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.7, 0.3],
                         subplot_titles=("account value ($)", "drawdown from the high (%)"))
    for (_name, _c), _col in zip(_curves.items(), (SERIES[0], SERIES[2], SERIES[4]), strict=True):
        _fig.add_scatter(x=days(_c.index), y=account.value * _c, name=_name, mode="lines",
                         line={"color": _col, "width": 2.5 if _name == "the strategy" else 1.5},
                         row=1, col=1)
        _fig.add_scatter(x=days(_c.index), y=100 * (_c / _c.cummax() - 1), mode="lines",
                         line={"color": _col, "width": 1}, showlegend=False, name=_name,
                         row=2, col=1)
    finish(_fig, "The book against buy-and-hold and SPY", 560)
    mo.vstack([_fig, mo.md(
        "Blue is the strategy, yellow the equal-weight buy-and-hold of the same universe, purple "
        "SPY. All three start with the same account. The lower chart shows each one's drawdown "
        "from its own high, in the same colors.")])
    return


@app.cell
def _(SERIES, LONG, SHORT, bench, days, finish, make_subplots, mo, res):
    _pf = res["pf"]
    _val = _pf.value()
    _av = _pf.asset_value(group_by=False)
    _long = _av.clip(lower=0).sum(axis=1) / _val
    _short = -_av.clip(upper=0).sum(axis=1) / _val
    _a = _pf.assets()
    _nl, _ns = (_a > 0).sum(axis=1), (_a < 0).sum(axis=1)
    _r = res["returns"]
    _beta = _r.rolling(63).cov(bench) / bench.rolling(63).var()
    _fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=("exposure (% of the account)", "positions held",
                                         "beta to the universe (rolling 63 sessions)"))
    _fig.add_scatter(x=days(_long.index), y=100 * _long, name="long exposure", mode="lines",
                     line={"color": LONG}, row=1, col=1)
    _fig.add_scatter(x=days(_short.index), y=100 * _short, name="short exposure", mode="lines",
                     line={"color": SHORT}, row=1, col=1)
    _fig.add_scatter(x=days(_long.index), y=100 * (_long - _short), name="net (long − short)",
                     mode="lines", line={"color": SERIES[3], "dash": "dot"}, row=1, col=1)
    _fig.add_scatter(x=days(_nl.index), y=_nl, name="longs held", mode="lines",
                     line={"color": LONG, "shape": "hv"}, row=2, col=1)
    _fig.add_scatter(x=days(_ns.index), y=_ns, name="shorts held", mode="lines",
                     line={"color": SHORT, "shape": "hv"}, row=2, col=1)
    _fig.add_scatter(x=days(_beta.index), y=_beta, name="rolling beta", mode="lines",
                     line={"color": SERIES[5]}, row=3, col=1)
    _fig.add_hline(y=0, line={"color": "#898781", "width": 1}, row=3, col=1)
    finish(_fig, "What the book held", 680)
    mo.vstack([_fig, mo.md(
        "- **Exposure:** blue is the long side's share of the account and orange the short "
        "side's. The dotted green line is the difference; it moves because winners grow and "
        "losers shrink between trades.\n"
        "- **Positions:** the middle chart counts the positions on each side.\n"
        "- **Beta:** the red line is the book's measured market exposure over the last 63 "
        "sessions. Above 0 the book rises and falls with the market, below 0 against it.")])
    return


@app.cell
def _(SERIES, LONG, SHORT, bench, days, finish, go, mo, pd, pnl, res, spy, table):
    _c = pnl[["long", "short", "costs", "total"]].cumsum()
    _fig = go.Figure()
    for _k, _col, _w in (("long", LONG, 1.5), ("short", SHORT, 1.5), ("costs", SERIES[5], 1.5),
                         ("total", SERIES[3], 3)):
        _fig.add_scatter(x=days(_c.index), y=_c[_k], mode="lines",
                         name={"long": "long side P&L", "short": "short side P&L",
                               "costs": "costs (commission, spread, slippage, borrow)",
                               "total": "total P&L"}[_k], line={"color": _col, "width": _w})
    finish(_fig, "Where the profit came from (cumulative, $)", 420)
    _month = res["returns"].index.tz_localize(None).to_period("M")
    _m = pd.DataFrame({"strategy": (1 + res["returns"]).groupby(_month).prod() - 1,
                       "buy-and-hold": (1 + bench).groupby(_month).prod() - 1,
                       "SPY": (1 + spy).groupby(_month).prod() - 1})
    _d = pnl.groupby(pnl.index.tz_localize(None).to_period("M"))[["long", "short", "costs"]].sum()
    _tab = pd.concat([(100 * _m).round(1).astype(str) + "%", _d.round(0)], axis=1)
    _tab.index = _tab.index.astype(str)
    _tab.index.name = "month"
    _tab.columns = ["strategy", "buy-and-hold", "SPY", "long $", "short $", "costs $"]
    _fall = [("2025-02-19", "2025-04-08", "the Feb-Apr 2025 fall")]
    _lines = []
    for _a, _b, _n in _fall:
        _sl = slice(pd.Timestamp(_a, tz="UTC"), pd.Timestamp(_b, tz="UTC"))
        _book = 100 * ((1 + res['returns'][_sl]).prod() - 1)
        _lines.append(f"- **{_n}:** the strategy {_book:+.1f}%,"
                      f" buy-and-hold {100 * ((1 + bench[_sl]).prod() - 1):+.1f}%,"
                      f" SPY {100 * ((1 + spy[_sl]).prod() - 1):+.1f}%.")
    mo.vstack([_fig, mo.md(
        "Blue is what the longs made, orange what the shorts made, red the costs and thick green "
        "the total. Over a rising market the shorts lose money; they earn when the market or "
        "their own stocks fall."),
        mo.md("**In the falls inside this window:**\n" + "\n".join(_lines)),
        mo.md("**Month by month**"), table(_tab, 24)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Every position on a timeline

    Each bar is one position, from the open it was bought (or sold short) at to the open it was
    closed at. Positions still held run to the last bar.
    - **Color is the result:** green made money, red lost money. Solid bars are longs, hatched
      bars shorts.
    - **Hover a bar** for its prices, its P&L, its ranks and why it closed.
    """)
    return


@app.cell
def _(mo):
    side_pick = mo.ui.dropdown(options=["all", "long", "short"], value="all", label="side")
    order_pick = mo.ui.dropdown(options=["entry date", "P&L"], value="entry date",
                                label="sort by")
    mo.hstack([side_pick, order_pick], justify="start", gap=2)
    return order_pick, side_pick


@app.cell
def _(LAST, LOSS, WIN, finish, go, mo, np, order_pick, pd, pos, side_pick):
    _p = pos if side_pick.value == "all" else pos[pos["side"] == side_pick.value]
    _p = _p.sort_values("entry" if order_pick.value == "entry date" else "P&L $",
                        ascending=order_pick.value == "entry date").reset_index(drop=True)
    _p = _p.assign(label=[f"{s} {'▲' if d == 'long' else '▼'} #{i + 1}"
                          for i, (s, d) in enumerate(zip(_p["symbol"], _p["side"], strict=True))],
                   end=_p["exit"].where(_p["status"] == "closed", LAST))
    _ms = (pd.to_datetime(_p["end"]) - pd.to_datetime(_p["entry"])).dt.total_seconds() * 1000
    _p = _p.assign(ms=_ms.clip(lower=86_400_000))
    _fig = go.Figure()
    for _side in ("long", "short"):
        for _won, _c in ((True, WIN), (False, LOSS)):
            _d = _p[(_p["side"] == _side) & ((_p["P&L $"] > 0) == _won)]
            if not len(_d):
                continue
            _fig.add_bar(
                y=_d["label"], x=_d["ms"], base=pd.to_datetime(_d["entry"]).dt.strftime("%Y-%m-%d"),
                orientation="h", name=f"{_side}, {'profit' if _won else 'loss'}",
                marker={"color": _c, "opacity": 0.85,
                        "pattern": {"shape": "/" if _side == "short" else ""}},
                customdata=np.column_stack([
                    pd.to_datetime(_d["entry"]).dt.strftime("%Y-%m-%d"),
                    pd.to_datetime(_d["end"]).dt.strftime("%Y-%m-%d"),
                    _d["entry price"].map("${:,.2f}".format),
                    _d["exit price"].map("${:,.2f}".format),
                    _d["P&L $"].map("{:,.0f}".format), (100 * _d["return"]).map("{:+.1f}%".format),
                    _d["sessions"], _d["cause"], _d["rank at entry"].map("{:.2f}".format),
                    _d["rank at exit"].map("{:.2f}".format), _d["status"]]),
                hovertemplate=("%{y}<br>opened %{customdata[0]} at %{customdata[2]}"
                               "<br>closed %{customdata[1]} at %{customdata[3]} (%{customdata[10]})"
                               "<br>P&L $%{customdata[4]} (return %{customdata[5]})"
                               "<br>%{customdata[6]} sessions · %{customdata[7]}"
                               "<br>combined rank at entry %{customdata[8]}, at exit "
                               "%{customdata[9]}<extra></extra>"))
    _fig.update_xaxes(type="date")
    _fig.update_yaxes(categoryorder="array", categoryarray=list(_p["label"])[::-1],
                      tickfont={"size": 9})
    finish(_fig, f"{len(_p)} positions from {pd.to_datetime(_p['entry']).min().date()}",
           max(420, 16 * len(_p) + 140))
    _fig.update_layout(barmode="overlay", bargap=0.25)
    mo.vstack([_fig])
    return


@app.cell
def _(mo, pos, table):
    _p = pos.copy()
    _p["won"] = _p["P&L $"] > 0
    _side = _p.groupby("side").agg(positions=("symbol", "size"), win_rate=("won", "mean"),
                                   total_pnl=("P&L $", "sum"), mean_return=("return", "mean"),
                                   median_sessions=("sessions", "median"))
    _side.loc["all"] = [len(_p), _p["won"].mean(), _p["P&L $"].sum(), _p["return"].mean(),
                        _p["sessions"].median()]
    _side["win_rate"] = (100 * _side["win_rate"]).map("{:.0f}%".format)
    _side["mean_return"] = (100 * _side["mean_return"]).map("{:+.1f}%".format)
    _side["total_pnl"] = _side["total_pnl"].map("${:,.0f}".format)
    _side.index.name = "side"
    _cause = _p.groupby("cause").agg(positions=("symbol", "size"), total_pnl=("P&L $", "sum"),
                                     mean_return=("return", "mean"))
    _cause["mean_return"] = (100 * _cause["mean_return"]).map("{:+.1f}%".format)
    _cause["total_pnl"] = _cause["total_pnl"].map("${:,.0f}".format)
    _cause.index.name = "why it closed"
    _show = _p[["symbol", "side", "entry", "exit", "sessions", "status", "entry price",
                "exit price", "P&L $", "return", "cause", "rank at entry", "SI rank at entry",
                "momentum rank at entry", "rank at exit"]].copy()
    for _c in ("entry", "exit"):
        _show[_c] = _show[_c].map(lambda t: "" if t is None or t != t else str(t)[:10])
    _show["return"] = (100 * _show["return"]).round(1)
    mo.vstack([mo.md("**By side**"), table(_side, 5), mo.md("**By the reason it closed**"),
               table(_cause, 8),
               mo.md("**Every position** (prices adjusted, slippage included; P&L after "
                     "commission and borrow; return in %)"),
               table(_show.round(2), 20)])
    return


@app.cell
def _(LOSS, WIN, finish, go, mo, pd, pos):
    _by = pos.groupby(["symbol", "side"])["P&L $"].sum().sort_values()
    _both = pd.concat([_by.head(10), _by.tail(10)])
    _both = _both[~_both.index.duplicated()]
    _fig = go.Figure(go.Bar(
        x=_both.values, y=[f"{s} ({d})" for s, d in _both.index], orientation="h",
        marker={"color": [WIN if v > 0 else LOSS for v in _both.values]},
        name="P&L per stock and side (green profit, red loss)", showlegend=True,
        hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
    finish(_fig, "The 10 best and 10 worst stocks (all their positions, $)", 560)
    mo.vstack([_fig, mo.md("Green bars made money, red bars lost it. A few names usually carry "
                           "much of the result; the timeline above shows when they were held.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · One position up close

    Pick a position (biggest results first). The charts show the stock around the holding
    period, its combined rank against the bands, and the position's running P&L.
    """)
    return


@app.cell
def _(mo, pos):
    _order = pos.reindex(pos["P&L $"].abs().sort_values(ascending=False).index)
    _opts = {f"{r['symbol']} {r['side']} · {str(r['entry'])[:10]} → "
             f"{str(r['exit'])[:10] if r['status'] == 'closed' else 'open'} · "
             f"${r['P&L $']:,.0f}": i for i, r in _order.iterrows()}
    one = mo.ui.dropdown(options=_opts, value=next(iter(_opts)), label="position")
    mo.hstack([one], justify="start")
    return (one,)


@app.cell
def _(LAST, SERIES, LONG, MUTED, SHORT, days, finish, make_subplots, mo, one, panel, pd,
      pos, w):
    _p = pos.loc[one.value]
    _s, _c = _p["symbol"], (LONG if _p["side"] == "long" else SHORT)
    _end = _p["exit"] if _p["status"] == "closed" else LAST
    _idx = panel.close.index
    _lo = _idx[max(0, _idx.get_loc(_p["entry"]) - 60)]
    _hi = _idx[min(len(_idx) - 1, _idx.get_loc(_end) + 30)]
    _px = panel.close[_s].loc[_lo:_hi]
    _sc = w["score"][_s].loc[_lo:_hi]
    _held = panel.close[_s].loc[_p["entry"]:_end]
    _sign = 1 if _p["side"] == "long" else -1
    _run = _sign * _p["shares"] * (_held - _p["entry price"])
    _fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         row_heights=[0.45, 0.3, 0.25],
                         subplot_titles=("price (adjusted close)", "combined rank",
                                         "the position's running P&L ($, before the exit's costs)"))
    _fig.add_scatter(x=days(_px.index), y=_px, name="price", mode="lines",
                     line={"color": MUTED}, row=1, col=1)
    _fig.add_scatter(x=days([_p["entry"]]), y=[_p["entry price"]], mode="markers",
                     marker={"symbol": "triangle-up" if _sign > 0 else "triangle-down",
                             "size": 14, "color": _c}, name="opened (fill at the open)",
                     row=1, col=1)
    if _p["status"] == "closed":
        _fig.add_scatter(x=days([_p["exit"]]), y=[_p["exit price"]], mode="markers",
                         marker={"symbol": "x", "size": 13, "color": _c},
                         name="closed (fill at the open)", row=1, col=1)
    _fig.add_scatter(x=days(_sc.index), y=_sc, name="combined rank", mode="lines",
                     line={"color": SERIES[0], "width": 2.5}, row=2, col=1)
    _bands = (0.9, 0.8) if _sign > 0 else (0.1, 0.2)
    for _y, _dash, _n in ((_bands[0], "solid", "entry threshold"),
                          (_bands[1], "dash", "exit threshold")):
        _fig.add_hline(y=_y, line={"color": _c, "dash": _dash, "width": 1}, row=2, col=1)
        _fig.add_scatter(x=[days(_sc.index)[0]], y=[None], mode="lines", name=f"{_n} {_y:.2f}",
                         line={"color": _c, "dash": _dash}, row=2, col=1)
    _fig.add_scatter(x=days(_run.index), y=_run, name="running P&L", mode="lines",
                     line={"color": SERIES[3], "width": 2}, fill="tozeroy", row=3, col=1)
    for _r in (1, 2, 3):                          # after the traces: see CLAUDE.md gotchas
        _fig.add_vrect(x0=days([_p["entry"]])[0], x1=days([_end])[0], fillcolor=_c,
                       opacity=0.1, line_width=0, row=_r, col=1)
    _fig.update_yaxes(range=[0, 1.02], row=2, col=1)
    finish(_fig, f"{_s} {_p['side']}: {str(_p['entry'])[:10]} → "
                 f"{str(_end)[:10]}", 680)
    _decided = ("before the window (it was already held on 2025-01-01)"
                if _p["held from before the start"]
                else f"at the close of {str(_p['decided'])[:10]}")
    _why_in = (f"its combined rank {'reached 0.90' if _sign > 0 else 'fell to 0.10'} "
               f"({_p['rank at entry']:.2f}: short-interest rank {_p['SI rank at entry']:.2f}, "
               f"momentum rank {_p['momentum rank at entry']:.2f}) and a slot was free")
    _why_out = ("it is still held" if _p["status"] == "open" else
                f"{_p['cause']} (decided at the close of {str(_p['exit decided'])[:10]}"
                + (f", combined rank {_p['rank at exit']:.2f})" if pd.notna(_p["rank at exit"])
                   else ")"))
    mo.vstack([_fig, mo.md(f"""
    **The story of this position:**
    - **Why it opened:** decided {_decided}, because {_why_in}.
    - **The fill:** {'bought' if _sign > 0 else 'sold short'} at the next open,
      {str(_p['entry'])[:10]}: {_p['shares']:.2f} shares at ${_p['entry price']:.2f}.
    - **Why it closed:** {_why_out}.
      {'Closed' if _p['status'] == 'closed' else 'Marked'} at ${_p['exit price']:.2f}.
    - **The result:** **${_p['P&L $']:,.0f}** ({100 * _p['return']:+.1f}%) over
      {_p['sessions']} sessions, after commission and borrow.

    The triangle is where the position opened and the cross where it closed. The shaded span is
    the holding period. The middle chart shows the rank against the entry (solid) and exit
    (dashed) thresholds.
    """)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · What to keep in mind

    - **This is one continuous book.** The research (`PROPOSALS_RESULTS.md`) restarted the book
      every half-year and measured it on four groups of stocks. This notebook runs one book from
      2025-01-02 on, as a live account would, so its numbers differ somewhat from the research
      tables.
    - **The benchmark is point in time.** It buys the universe as it stood at each re-formation.
      An equal-weight buy-and-hold of every name in the panel would include stocks that only
      joined in 2026 because they had risen: look-ahead.
    - **Fills are at the next open, with modelled costs and borrow.** Every recent short was
      borrowable at IBKR, mostly at 0.3-0.4% a year. A few business development companies cost
      3-9%.
    - **The "too crowded to short" cap inherits a known flaw from informed_flow_ls v6.** It reads
      short interest ÷ shares with a volume that is split-adjusted backward. Before a forward
      split that overstates the ratio, so a stock about to split can be kept out of the short
      book: a small look-ahead. informed_flow_ls v8 fixed it in its own code; here it touches the
      cap and the borrow tier, never the ranks.
    - **The database holds only stocks that still exist** (survivors), which flatters the long
      side and the benchmark.
    - **On the top 500 the book has carried net market exposure since mid-2025** (rolling beta,
      §4). See `TOP_STRATEGIES.md` §4 for the market hedge.
    """)
    return


if __name__ == "__main__":
    app.run()
