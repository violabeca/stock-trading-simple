import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Classic signals · stock groups

    This notebook sorts every stock with 1-minute candles into **groups of stocks that behave
    alike**, so that the tuning notebook can look for **one setting per group** instead of one
    per stock: a setting judged on several alike stocks rests on more evidence than one stock's
    history, and the group's other members show whether it carries over to stocks it never saw.

    "Alike" means alike in what the strategy's rules react to, its **character**: how big the
    stock's moves are and how much of one a round trip's costs eat; whether a minute's move
    tends to continue or turn back; whether a close outside the Bollinger Band snaps back or runs
    on; how much of its daily swing comes overnight; how closely it follows the S&P 500; how
    jumpy and how thinly traded it is. Industries play no part: a bank and a chip maker can
    behave alike.

    1. **The pool**: every stock with 1-minute candles, and how much history each has
    2. **The numbers**: how each one is measured (drawn for any stock), how much of it is the
       stock's own, which ones measure the same thing
    3. **The groups**: how many, the stocks on a map, what each group is like
    4. **One stock**: its character against its group's, its closest stocks
    5. **One group**: its members, the most typical first: the ones to tune on, then the ones to
       check the tuned setting on, and the candles of the top ones side by side
    6. **The export**: the groups as a file for the tuning step

    **Every stored candle is read** (the first is from December 2024, the last from the newest
    download). Unlike the tuning notebook's training window (Jan 2025 – Jun 2026), that includes
    its validation months and its final check: once the tuning step reads these groups, those
    months are no longer unseen by the grouping (mild: the numbers describe how a stock moves,
    not what it earned).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(df, rows=12):
        """A DataFrame as a marimo table; a named index becomes a column."""
        return mo.ui.table(
            df.reset_index() if df.index.name else df, selection=None, page_size=rows
        )

    return (table,)


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload the helper files (after they changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util

    import numpy as np
    import pandas as pd

    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    _ = reload_helpers.value     # a click reruns this cell: the helper files are read again

    def _helper(name):
        """One of this strategy's helper files, loaded by path."""
        path = strategies.STRATEGIES_DIR / "classic_signals" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"classic_signals_g_{name}", path)
        assert spec is not None and spec.loader is not None, path
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    data, charts, groups = _helper("data"), _helper("charts"), _helper("groups")
    pipeline, simulate = _helper("pipeline"), _helper("simulate")
    settings = load_settings()
    engine = make_engine(settings)
    COSTS = simulate.costs(settings)     # the size and costs the strategy trades with
    FEATURES = groups.FEATURES
    STRATEGY_STOCKS = list(data.SECTOR)  # NOW, GEV, RGTI: marked wherever they appear
    return (
        COSTS,
        FEATURES,
        STRATEGY_STOCKS,
        charts,
        data,
        engine,
        groups,
        np,
        pd,
        pipeline,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The pool

    Every symbol with 1-minute candles of the regular session in the database (all of them, from
    the first stored minute to the newest download), except the S&P 500 fund (SPY): it is the
    market every stock is compared with. Funds (IGV, XLI, QTUM, ...) are in like any stock.

    For every stock, its candles are read once and boiled down to a few sums per session
    (`groups.session_sums`); the sums are kept on disk (`var/classic_signals/groups/`), so the
    next run reads them in seconds. A stock is measured again when its candles changed: an
    earlier first or a later last minute (the 2025 download still adds history to many
    stocks). Measuring the whole pool takes about 20 minutes; from a terminal:
    `uv run python strategies/classic_signals/groups.py`.

    **Asking the database** which stocks it has takes a minute or two, so the answer is kept
    too: click *ask the database again* after new downloads.
    """)
    return


@app.cell
def _(mo):
    refresh_pool = mo.ui.run_button(label="ask the database again")
    measure = mo.ui.run_button(label="measure the missing stocks")
    workers = mo.ui.slider(1, 12, step=1, value=8, show_value=True,
                           label="stocks read at once (threads)")
    least = mo.ui.slider(20, 300, step=10, value=40, show_value=True,
                         label="least sessions a stock needs to be grouped")
    mo.vstack([mo.hstack([refresh_pool, measure, workers], justify="start", gap=2),
               mo.hstack([least], justify="start")])
    return least, measure, refresh_pool, workers


@app.cell(hide_code=True)
def _(data, engine, groups, mo, refresh_pool):
    with mo.status.spinner(title="asking the database for every stock with 1-minute candles"):
        POOL = groups.pool(engine, refresh=refresh_pool.value)
    _asked = POOL.attrs.get("asked", "?")
    POOL = POOL[POOL["symbol"] != data.MARKET].reset_index(drop=True)
    POOL.attrs["asked"] = _asked
    MISSING = groups.missing(POOL)
    mo.md(f"**{len(POOL):,} stocks** in the pool, with candles from "
          f"{POOL['first'].min():%Y-%m-%d} to {POOL['last'].max():%Y-%m-%d} (the database was "
          f"asked {_asked}); **{len(MISSING):,}** not measured yet.")
    return MISSING, POOL


@app.cell(hide_code=True)
def _(MISSING, POOL, engine, groups, measure, mo, workers):
    mo.stop(bool(MISSING) and not measure.value, mo.md(
        f"_**{len(MISSING):,} of {len(POOL):,} stocks** are not measured yet (about "
        f"{len(MISSING) * 0.6 / 60:.0f} minutes): click **measure the missing stocks**, or run "
        "`uv run python strategies/classic_signals/groups.py` in a terminal and rerun this "
        "cell._"))
    if MISSING:
        with mo.status.progress_bar(total=len(MISSING), title="measuring stocks") as _bar:
            SUMS = groups.compute(engine, POOL, workers=workers.value,
                                  progress=lambda: _bar.update())
    else:
        SUMS = groups.compute(engine, POOL)
    return (SUMS,)


@app.cell(hide_code=True)
def _(COSTS, FEATURES, SUMS, charts, groups, least, mo, pd):
    _all = groups.characters(SUMS, COSTS)
    _keep = _all.index[_all["sessions"] >= least.value]
    _sums = {_s: SUMS[_s] for _s in _keep}
    TABLE = _all.loc[_keep]                                    # the whole window
    HALVES = tuple(groups.characters(_sums, COSTS, part=_p)    # each half of a stock's sessions
                   for _p in ("first half", "second half"))
    _gaps = TABLE[list(FEATURES)].isna().any(axis=1)
    _first = pd.to_datetime(TABLE["first session"])
    _rows = pd.DataFrame({
        "stocks grouped": [len(TABLE)],
        "left out: too little history": [int((_all["sessions"] < least.value).sum())],
        "with history from Dec 2024 – Jan 2025": [int((_first < "2025-02-01").sum())],
        "from mid-2025": [int(((_first >= "2025-02-01") & (_first < "2025-08-01")).sum())],
        "from 2026 only": [int((_first >= "2026-01-01").sum())],
        "with a number missing (left out below)": [int(_gaps.sum())]})
    mo.vstack([
        charts.history_figure(_all["sessions"], least=least.value,
                              title="Figure 1 · How many sessions of data each stock has"),
        mo.md("""
        Most stocks started at one of three dates: December 2024 – January 2025, June 2025
        (the first 1-minute download) or late March 2026 (the widened one). A stock with little
        history gets noisier numbers: the persistence check below shows how much that matters.
        The slider above sets the least.
        """),
        mo.md("### Table 1 · The pool"), mo.ui.table(_rows, selection=None)])
    return HALVES, TABLE


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · The numbers

    **Table 2** lists every number: what it measures, how it is computed, its spread over the
    pool (the stock at the 10th, 50th and 90th percentile) and its **persistence**: each stock's
    sessions are cut in two halves, the number is measured on each, and the rank correlation
    across the stocks says how much a stock high on it in the first half is high in the second.
    Near 1 the number is the stock's own character; near 0 mostly chance or a passing phase,
    and then it cannot define a lasting group (**Figure 2**).

    **Figure 3** shows which numbers measure the same thing (rank correlations across the
    stocks): two numbers near +1 or -1 weigh twice in the grouping. Thinly traded stocks, for
    instance, bounce from minute to minute (a trade at the bid, the next at the ask), so their
    minutes' moves turn back, a close outside the band comes straight back, and a few minutes
    make most of their movement: several numbers lean on how often a minute has no trade.
    """)
    return


@app.cell(hide_code=True)
def _(FEATURES, HALVES, TABLE, charts, groups, mo, pd, table):
    _f = list(FEATURES)
    PERSIST = groups.persistence(HALVES[0][_f], HALVES[1][_f])
    _q = TABLE[_f].quantile([0.1, 0.5, 0.9])
    _rows = pd.DataFrame([{
        "number": _n, "what it measures": FEATURES[_n]["label"],
        "10%": format(_q.loc[0.1, _n], FEATURES[_n]["format"]),
        "50%": format(_q.loc[0.5, _n], FEATURES[_n]["format"]),
        "90%": format(_q.loc[0.9, _n], FEATURES[_n]["format"]),
        "persistence": round(float(PERSIST[_n]), 2),
        "how it is computed": FEATURES[_n]["doc"]} for _n in _f])
    mo.vstack([
        mo.md("### Table 2 · The numbers"), table(_rows, rows=len(_f)),
        mo.hstack([charts.persistence_figure(PERSIST, FEATURES, title="Figure 2 · How much "
                                             "of each number is the stock's own"),
                   charts.correlation_figure(TABLE[_f], FEATURES, title="Figure 3 · Which "
                                             "numbers measure the same thing")],
                  widths=[0.4, 0.6])])
    return (PERSIST,)


@app.cell(hide_code=True)
def _(FEATURES, STRATEGY_STOCKS, TABLE, mo):
    _stocks = sorted(TABLE.index)
    explain_stock = mo.ui.dropdown(_stocks, value=next(
        (_s for _s in STRATEGY_STOCKS if _s in TABLE.index), _stocks[0]), label="stock",
        searchable=True)
    explain_feature = mo.ui.dropdown({FEATURES[_n]["label"]: _n for _n in FEATURES},
                                     value=FEATURES["band_follow"]["label"], label="number")
    moving_sessions = mo.ui.slider(5, 60, step=5, value=20, show_value=True,
                                   label="moving window, sessions")
    mo.vstack([
        mo.md("""
        ### How a number is measured, for any stock

        Pick a stock and a number. **Figure 4**, left: the number over a moving window of
        sessions (blue: how steady it is from month to month), its value on each half of the
        stock's sessions (amber: what the persistence check compares) and on the whole window
        (black dashed: what the grouping reads). Right: the same number for every stock of the
        pool, the picked stock's value marked in blue.
        """),
        mo.hstack([explain_stock, explain_feature, moving_sessions], justify="start", gap=2)])
    return explain_feature, explain_stock, moving_sessions


@app.cell(hide_code=True)
def _(
    COSTS,
    FEATURES,
    HALVES,
    SUMS,
    TABLE,
    charts,
    explain_feature,
    explain_stock,
    groups,
    mo,
    moving_sessions,
):
    _s, _n = explain_stock.value, explain_feature.value
    _moving = groups.moving(SUMS[_s], COSTS, sessions=moving_sessions.value)[_n]
    mo.vstack([
        charts.feature_figure(_s, _n, _moving, TABLE, HALVES, FEATURES[_n],
                              sessions=moving_sessions.value,
                              title=f"Figure 4 · {_s}: {FEATURES[_n]['label']}"),
        mo.md(f"**How it is computed:** {FEATURES[_n]['doc']}")])
    return


@app.cell(hide_code=True)
def _(SUMS, explain_stock, mo):
    _days = [str(_d.date()) for _d in SUMS[explain_stock.value].index]
    band_day = mo.ui.dropdown(_days, value=_days[-1], label="session", searchable=True)
    mo.vstack([
        mo.md(f"""
        ### The two band numbers, drawn on one session of {explain_stock.value}

        **Figure 5**: the 1-minute closes (blue) inside the Bollinger Band (gray: the built-in
        20 candles, 2 standard deviations). Every close that lands outside the band (the one
        before was not) is marked: **green** when the very next close is back inside (the
        candle the band fade fires on), **amber** when the next close is outside too (the band
        breakout's second close). The dotted line runs to the close 15 minutes later; hover a
        mark for how far the price went on (+) or came back (-), in typical 15-minute moves.
        Over all sessions, the share of green marks is *band_back* and the average of those
        moves is *band_follow*.
        """),
        mo.hstack([band_day], justify="start")])
    return (band_day,)


@app.cell(hide_code=True)
def _(band_day, charts, data, engine, explain_stock, groups, mo, pd, pipeline):
    _s, _day = explain_stock.value, pd.Timestamp(band_day.value)
    _m = data.load_minutes(engine, [_s], _day - pd.Timedelta(days=10), _day)[_s]
    _bands = pipeline.bands(_m, *groups.BAND, False)
    _ev = groups.band_events(_m)
    _in = data.session_day(_m.index) == _day
    _ev_day = _ev[data.session_day(pd.DatetimeIndex(_ev["time"])) == _day]
    charts.band_events_figure(_s, _m[_in], _bands[_in], _ev_day, follow=groups.FOLLOW,
                              title=f"Figure 5 · {_s} on {band_day.value}: {len(_ev_day)} "
                                    f"closes outside the band, {int(_ev_day['back'].sum())} "
                                    "back inside at the next close")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · The groups

    **How they are formed.** Every number becomes a **score** against the other stocks (the
    normal score of its rank: 0 is the middle stock, +1 above about 84% of them, -1 below), so
    every number weighs the same and no stock far out stretches the scale. A **Ward tree**
    then joins the stocks step by step, always the two groups whose merger adds the least
    spread inside the groups, and is cut into the number of groups you choose. The **numbers
    used** below pick which numbers count: leave out one that does not persist (Figure 2) or one
    that repeats another (Figure 3).

    **How many groups?** **Figure 6** helps choose. *Explained* (blue): the share of the
    stocks' differences that lies between the groups; it rises with every group, so look for
    where it stops rising much. *Separation* (amber): how much closer a stock is to its own
    group than to the next (1 far apart, 0 on the border); stocks spread evenly over the map
    give low values at any number of groups: then the groups are slices of one cloud, still
    useful if they are stable. *Same on both halves* (green): the groups formed on each half of
    the stocks' sessions alone, compared (adjusted Rand index: 1 identical, 0 no more alike than
    chance): a number of groups that does not repeat on the halves is cut from noise. *Stays in
    its group* (purple): with the groups as formed on the whole window, the share of stocks that
    each half alone puts closest to their own group's centre (dotted: the same by chance): how
    far a stock's group is its lasting character, which is what trading a group's setting on it
    later relies on. More groups cut finer, and fewer stocks stay.
    **Tuned on** sets how many of every group's most typical members the tuning reads; the
    rest of the group checks the tuned setting.
    """)
    return


@app.cell
def _(FEATURES, PERSIST, mo):
    used = mo.ui.multiselect({FEATURES[_n]["label"]: _n for _n in FEATURES},
                             value=[FEATURES[_n]["label"] for _n in FEATURES],
                             label="numbers used")
    k = mo.ui.slider(2, 15, step=1, value=6, show_value=True, label="number of groups")
    n_tune = mo.ui.slider(2, 15, step=1, value=5, show_value=True,
                          label="tuned on: most typical members per group")
    _weak = [_n for _n in FEATURES if PERSIST[_n] < 0.4]
    mo.vstack([used, mo.hstack([k, n_tune], justify="start", gap=2),
               mo.md(f"<span style='color:#77756f'>Persisting least (below 0.4): "
                     f"{', '.join(_weak) or 'none'}</span>")])
    return k, n_tune, used


@app.cell(hide_code=True)
def _(HALVES, TABLE, groups, mo, used):
    mo.stop(len(used.value) < 2, mo.md("_Pick at least two numbers._"))
    USED = list(used.value)
    _ok = TABLE[USED].notna().all(axis=1) & HALVES[0][USED].notna().all(axis=1) & \
        HALVES[1][USED].notna().all(axis=1)
    STOCKS = list(TABLE.index[_ok])      # every number known on the whole window and the halves
    SCORES = groups.scores(TABLE.loc[STOCKS, USED])
    HALF_SCORES = tuple(groups.scores(_h.loc[STOCKS, USED]) for _h in HALVES)
    with mo.status.spinner(title="trying every number of groups"):
        QUALITY = groups.quality(SCORES, *HALF_SCORES)
    return HALF_SCORES, QUALITY, SCORES, STOCKS, USED


@app.cell(hide_code=True)
def _(QUALITY, STOCKS, TABLE, charts, k, mo, table):
    mo.vstack([
        charts.quality_figure(QUALITY, k.value,
                              title=f"Figure 6 · How many groups? ({len(STOCKS):,} stocks)"),
        mo.md(f"_{len(TABLE) - len(STOCKS)} stocks left out: a number unknown on the whole "
              "window or on a half (e.g. no close outside the band)._"),
        mo.accordion({"Table 3 · The same as numbers": table(QUALITY.round(3), rows=14)})])
    return


@app.cell(hide_code=True)
def _(HALF_SCORES, SCORES, groups, k, n_tune):
    RESULT = groups.grouping(SCORES, k.value, n_tune=n_tune.value, halves=HALF_SCORES)
    MAP, AXES = groups.flat_map(SCORES)
    return AXES, MAP, RESULT


@app.cell(hide_code=True)
def _(AXES, MAP, RESULT, STRATEGY_STOCKS, charts, mo, table):
    mo.vstack([
        mo.md("### Table 4 · The groups"),
        table(RESULT["groups"].round(2), rows=15),
        charts.group_map_figure(MAP, AXES, RESULT["members"], RESULT["groups"],
                                highlight=STRATEGY_STOCKS,
                                title="Figure 7 · The stocks on a map, by group"),
        mo.md("""
        **Figure 7** puts every stock on the two directions along which the stocks differ most
        (the axis titles name the numbers that pull along each, + where a high value lies
        right or up, - where it lies left or down). A big outlined mark is a stock its group
        tunes on, a small one a stock the group checks on; the circled names are the
        strategy's stocks so far. Two stocks close on the map are alike on those directions
        only: the groups use every number, so a group can look mixed on the map.
        """)])
    return


@app.cell(hide_code=True)
def _(FEATURES, RESULT, TABLE, USED, charts):
    charts.group_profile_figure(RESULT["centre"][USED], TABLE, RESULT["members"], FEATURES,
                                title="Figure 8 · What each group is like (color: the group's "
                                      "centre among all stocks; text: its members' median)")
    return


@app.cell(hide_code=True)
def _(RESULT, STRATEGY_STOCKS, mo, pd, table):
    _m = RESULT["members"]
    _say = RESULT["groups"].set_index("group")["in words"]
    _rows = pd.DataFrame([{"stock": _s, "group": int(_m.loc[_s, "group"]),
                           "in words": _say[_m.loc[_s, "group"]],
                           "rank in its group": int(_m.loc[_s, "rank"]),
                           "of": int((_m["group"] == _m.loc[_s, "group"]).sum()),
                           "role": _m.loc[_s, "role"],
                           "stays on both halves": bool(_m.loc[_s, "stays"])}
                          for _s in STRATEGY_STOCKS if _s in _m.index])
    mo.vstack([mo.md("### Table 5 · Where the strategy's stocks land"),
               table(_rows) if len(_rows) else mo.md("_None of them is grouped._")])
    return


@app.cell(hide_code=True)
def _(STRATEGY_STOCKS, STOCKS, mo):
    _stocks = sorted(STOCKS)
    one_stock = mo.ui.dropdown(_stocks, value=next(
        (_s for _s in STRATEGY_STOCKS if _s in STOCKS), _stocks[0]), label="stock",
        searchable=True)
    mo.vstack([
        mo.md("""
        ## 4 · One stock

        **Figure 9**: the stock's score on every number (blue bars: 0 is the middle stock)
        against its group's centre (black marks). Where a bar and its mark agree the stock is
        typical of its group; hover a bar for the number in its own unit. **Table 6**: the
        stocks closest to it on all the numbers used, and their groups.
        """),
        mo.hstack([one_stock], justify="start")])
    return (one_stock,)


@app.cell(hide_code=True)
def _(FEATURES, RESULT, SCORES, TABLE, charts, groups, mo, one_stock, table):
    _s = one_stock.value
    _near = groups.nearest(SCORES, _s, 12)
    _near["group"] = RESULT["labels"].loc[_near.index]
    _near["same group"] = _near["group"] == RESULT["labels"].loc[_s]
    mo.vstack([
        charts.stock_figure(_s, SCORES, TABLE, RESULT["members"], RESULT["centre"], FEATURES,
                            title=f"Figure 9 · {_s}: its character against group "
                                  f"{int(RESULT['labels'].loc[_s])}'s"),
        mo.md(f"### Table 6 · The stocks closest to {_s}"), table(_near.round(2), rows=12)])
    return


@app.cell(hide_code=True)
def _(RESULT, mo):
    one_group = mo.ui.dropdown({f"group {int(_r['group'])} ({_r['stocks']}): {_r['in words']}":
                                int(_r["group"]) for _, _r in RESULT["groups"].iterrows()},
                               value=None, label="group")
    mo.vstack([
        mo.md("""
        ## 5 · One group

        **Table 7**: a group's members, the most typical first (the smallest distance to the
        group's centre, in score units). The first ones (*tune*) are what the tuning reads; the
        others (*check*) test the tuned setting on stocks it never saw. Every number in its own
        unit beside them. *stays*: each half of the stock's sessions alone is also closest to
        this group.
        """),
        mo.hstack([one_group], justify="start")])
    return (one_group,)


@app.cell(hide_code=True)
def _(FEATURES, RESULT, TABLE, USED, mo, one_group, table):
    _g = one_group.value or 1
    _m = RESULT["members"]
    _part = _m[_m["group"] == _g]
    _rows = _part[["rank", "role", "stays", "distance"]].join(
        TABLE[["price", "sessions", *USED]])
    for _n in USED:
        _rows[_n] = [format(_v, FEATURES[_n]["format"]) for _v in _rows[_n]]
    _rows["price"] = _rows["price"].round(2)
    _rows["distance"] = _rows["distance"].round(2)
    mo.vstack([mo.md(f"### Table 7 · Group {_g}: {len(_part)} stocks"), table(_rows, rows=20)])
    return


@app.cell(hide_code=True)
def _(mo):
    show_n = mo.ui.slider(1, 12, step=1, value=5, show_value=True,
                          label="stocks shown (most typical first)")
    candle_months = mo.ui.slider(3, 18, step=1, value=12, show_value=True,
                                 label="months shown")
    candle_scale = mo.ui.dropdown(["one scale for all", "each stock's own range"],
                                  value="one scale for all", label="price axis")
    mo.vstack([
        mo.md("""
        ### Look at the group's stocks

        **Figure 10**: the daily candles of the group's most typical stocks (the *tune* ones
        first, then *check* ones), one panel each on the same dates, over the last months of the
        data. Every stock is **rebased to 100 at its first close** and drawn on a
        **logarithmic** axis, so the same percent move is the same height in every panel,
        whatever the stock's price. With *one scale for all* the panels share their price range
        (a calmer stock draws smaller candles beside a wilder one); with *each stock's own
        range* every panel is zoomed to its own stock, to compare shapes. Blue candles closed
        above their open, red below. Stocks of one group should look alike in how they move (the
        size of the days, how choppy), not in where they went: a group does not share a trend.
        """),
        mo.hstack([show_n, candle_months, candle_scale], justify="start", gap=2)])
    return candle_months, candle_scale, show_n


@app.cell(hide_code=True)
def _(POOL, RESULT, candle_months, data, engine, mo, one_group, pd, show_n):
    _g = one_group.value or 1
    _m = RESULT["members"]
    TOP = _m[_m["group"] == _g].sort_values("rank").head(show_n.value)
    _end = POOL["last"].max().tz_localize(None).normalize()      # the newest session stored
    with mo.status.spinner(title=f"reading the daily candles of {len(TOP)} stocks"):
        DAYS = data.load_days(engine, list(TOP.index), _end - pd.DateOffset(
            months=candle_months.value), _end)
    return DAYS, TOP


@app.cell(hide_code=True)
def _(DAYS, TOP, candle_months, candle_scale, charts, one_group):
    charts.group_candles_figure(
        DAYS, TOP, shared=candle_scale.value == "one scale for all",
        title=f"Figure 10 · Group {one_group.value or 1}: its {len(TOP)} most typical stocks, "
              f"the last {candle_months.value} months")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · The export

    Writes the groups as they stand above to
    `var/classic_signals/groups/exports/<date>-<name>.json`: for every group its words, its
    profile (every number's median), the members to tune on and the members to check on (each
    with its rank and distance), and what made them (the window, the numbers used, the number of
    groups, the least sessions, the pool). The tuning notebook will read it in the next step.
    """)
    return


@app.cell
def _(k, mo):
    export_name = mo.ui.text(value=f"{k.value}-groups", label="name")
    export_now = mo.ui.run_button(label="export the groups")
    mo.hstack([export_name, export_now], justify="start", gap=2)
    return export_name, export_now


@app.cell(hide_code=True)
def _(
    PERSIST,
    POOL,
    QUALITY,
    RESULT,
    STOCKS,
    TABLE,
    USED,
    export_name,
    export_now,
    groups,
    k,
    least,
    mo,
    n_tune,
):
    mo.stop(not export_now.value, mo.md("_Click **export the groups** to write the file._"))
    _q = QUALITY.set_index("groups").loc[k.value]
    _meta = {"window": [f"{POOL['first'].min():%Y-%m-%d}", f"{POOL['last'].max():%Y-%m-%d}"],
             "numbers used": USED, "groups": k.value,
             "tuned on per group": n_tune.value, "least sessions": least.value,
             "stocks grouped": len(STOCKS), "pool": len(POOL),
             "pool asked": POOL.attrs.get("asked"),
             "quality": {_c: float(_q[_c]) for _c in ("explained", "separation",
                                                      "same on both halves")},
             "persistence": {_n: round(float(PERSIST[_n]), 3) for _n in USED},
             "scores": "normal scores of the ranks; Ward tree on them"}
    _path = groups.save(RESULT, TABLE, meta=_meta, name=export_name.value)
    mo.md(f"Written: `{_path}`")
    return


if __name__ == "__main__":
    app.run()
