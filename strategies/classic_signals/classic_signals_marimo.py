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
    # Classic signals: the indicators, the trading rules and their results

    This is the workbench of a new strategy for **ServiceNow (NOW)**, **GE Vernova (GEV)** and
    **Rigetti Computing (RGTI)**: Bollinger Bands, moving averages, the trend of each stock's
    sector and of the market, and major price levels, each one a switch that can be turned on or
    off.

    **Step 1 trades nothing.** It shows the data and every indicator, so you can check that they
    look right before any rule relies on them:

    1. **Data check**: are all the one-minute candles there?
    2. **The daily picture**: candles, moving averages, major levels, report days, and the trend
       of the stock, its sector fund and the market fund
    3. **Minute by minute**: a few sessions with the bands, the moving average, the levels, the
       sector and the market, the typical move and the volume
    4. **Band behaviour**: how often the price closes outside its bands

    **Step 2 trades**, with the first rules as switches:

    5. **The first trading rules**: two band triggers (fade and breakout), the sector-trend
       filter and the trailing stop, and what each switch says candle by candle
    6. **Results** by market condition (up, sideways, down) and by side (long, short), the profit
       curve and the bottom line (profit ÷ worst drawdown)
    7. **Every trade, and why**: click a trade to see it drawn and explained in words

    **Step 3** adds three more filters as switches (the minute chart's moving average, the
    stock's own daily trend and the market's daily trend) and:

    8. **What each switch changes**: every rule switched off and on, everything else as set, and
       where the difference comes from (market condition, half-year)

    **Step 4** adds the levels: two filters that keep a trade from running straight into a
    major level or the previous session's high or low (sections 5 and 8), and an optional
    profit target at the next level (under "The exit"). Rigetti joins as a third stock, with
    QTUM (a quantum-computing fund) as its sector.

    **Step 5** adds the earnings guard (a switch: no trade held through a quarterly report). The
    tuning itself lives in its own notebook, `classic_signals_tuning_marimo.py`; the settings it
    exports can be loaded here under **Starting values** as every control's starting value.

    The sliders change every chart at once. The final-check months (July to October 2026) are
    only counted in the data check, never drawn or traded: they stay untouched until the very
    end.
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
    mo.hstack([reload_helpers, mo.md("The notebook reads its helper files (`data.py`, `rules/`, "
                                     "…) once; this button reads them again and reruns "
                                     "everything, without restarting the kernel.")],
              justify="start", gap=1)
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
        spec = importlib.util.spec_from_file_location(f"classic_signals_{name}", path)
        assert spec is not None and spec.loader is not None, path
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    data, ind, charts = _helper("data"), _helper("indicators"), _helper("charts")
    combine, simulate, results = _helper("combine"), _helper("simulate"), _helper("results")
    hyperparams, pipeline = _helper("hyperparams"), _helper("pipeline")
    RULES = combine.load_rules()     # every file listed in combine.RULE_FILES, in that order
    settings = load_settings()
    engine = make_engine(settings)
    return (
        RULES,
        charts,
        combine,
        data,
        engine,
        hyperparams,
        ind,
        np,
        pd,
        pipeline,
        results,
        settings,
        simulate,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Starting values

    Every slider, switch and lever below starts at a value from `hyperparams.py` (the one table
    of every hyperparameter both notebooks read), or, if you pick one here, at the values the
    tuning notebook (`classic_signals_tuning_marimo.py`) exported for one stock. A file holds one
    set per stock; the set you pick applies to every stock in this notebook, so pick the one of
    the stock you want to inspect. Changing the pick resets every control to its values.
    """)
    return


@app.cell
def _(hyperparams, mo):
    _options = {"built-in values (hyperparams.py)": None}
    for _f in hyperparams.files():
        for _s in hyperparams.load(_f)["stocks"]:
            _options[f"{_f.stem} · tuned for {_s}"] = (str(_f), _s)
    start_from = mo.ui.dropdown(_options, value="built-in values (hyperparams.py)",
                                label="starting values")
    mo.hstack([start_from], justify="start")
    return (start_from,)


@app.cell(hide_code=True)
def _(RULES, hyperparams, start_from):
    START = hyperparams.starting(RULES, *(start_from.value or ()))   # name -> starting value
    HP = hyperparams.table(RULES)                                      # name -> its row
    return HP, START


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    The stocks, the fund that stands for each one's sector, the market fund, and the periods.
    The sector funds: IGV (software) for ServiceNow, XLI (industrials) for GE Vernova, and QTUM
    (the Defiance Quantum ETF, about 70 quantum-computing and machine-learning companies) for
    Rigetti. QTUM holds Rigetti at only about 2%, so its trend is the theme's, not Rigetti's own.

    - **EXPLORE**: the months every chart and (later) every test uses
    - **FINAL_CHECK**: kept aside for one honest look at the very end; here only counted
    - **DAILY_FROM**: daily candles start a year earlier, so a 50-day trend and the major
      levels are ready on the first day of EXPLORE
    """)
    return


@app.cell
def _():
    STOCKS = {"NOW": "IGV", "GEV": "XLI", "RGTI": "QTUM"}  # each stock and its sector fund
    MARKET = "SPY"  # the S&P 500 fund: the market as a whole
    EXPLORE = ("2025-01-01", "2026-06-30")  # what the charts show
    FINAL_CHECK = ("2026-07-01", "2026-10-15")  # untouched until the very end
    DAILY_FROM = "2024-01-01"  # daily candles for the trends and levels start here
    return DAILY_FROM, EXPLORE, FINAL_CHECK, MARKET, STOCKS


@app.cell(hide_code=True)
def _(MARKET, STOCKS, data):
    SYMBOLS = data.symbols(STOCKS, MARKET)  # stocks, then sector funds, then the market fund
    return (SYMBOLS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Data check

    How many one-minute candles each symbol has on each session, from the first day of EXPLORE
    to the last day of FINAL_CHECK (counted in the database, no prices loaded). A regular
    session has **390** candles (09:30 to 16:00); a **half-day** has 210 (the market closes at
    13:00, e.g. the day after Thanksgiving).

    Two things to know about ServiceNow:

    - **Minutes without a trade**: candles where nobody traded; Interactive Brokers then repeats
      the last price with zero volume. ServiceNow has many of them in 2025 (up to 18% of the
      minutes in June 2025, mostly around midday) and none since January 2026: before its
      5-for-1 split in December 2025 one share cost about $1,000, so fewer shares changed hands
      each minute. The prices here are split-adjusted throughout
    - **Half-days**: its stored candles continue after the 13:00 close on the three half-days
      (after-hours trades labelled as regular hours). They are counted as after-hours ("half-day
      candles after 13:00")

    The **pre-market** (04:00-09:30, up to 330 minutes), **after-hours** (16:00-20:00, up to
    240 minutes) and **overnight** (20:00-04:00, up to 480 minutes; only special venues trade
    then, Sunday to Thursday nights) columns give the typical number of stored minutes per
    session and the share of them with an actual trade. Outside regular hours far fewer shares
    change hands, so many minutes have no trade at all. Nothing trades on Saturday and Sunday
    daytime.
    """)
    return


@app.cell(hide_code=True)
def _(EXPLORE, FINAL_CHECK, SYMBOLS, charts, data, engine, mo, table):
    _counts = data.coverage_counts(engine, SYMBOLS, EXPLORE[0], FINAL_CHECK[1])
    mo.vstack([
        mo.md("### Table 1 · Sessions per symbol"),
        table(data.data_check(_counts).round(4)),
        charts.coverage_figure(_counts, title="Figure 1 · One-minute candles per session "
                               "(hover a cell for the count)")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Loading the candles

    The one-minute candles of all five symbols for EXPLORE, the regular session (09:30-16:00)
    AND every hour around it: pre-market (04:00-09:30), after-hours (16:00-20:00) and overnight
    (20:00-04:00, where stored), and the daily candles from DAILY_FROM. The strategy will only
    trade in the regular session; the other hours are read as information.

    **Table 2** compares each day's last regular-session close with the stored daily close: they
    should be within a few hundredths of a percent (the official close comes from the closing
    auction a moment after 15:59). A large gap would mean the two disagree about a split or a
    dividend.
    """)
    return


@app.cell(hide_code=True)
def _(DAILY_FROM, EXPLORE, STOCKS, SYMBOLS, data, engine, mo, np, pd, pipeline, table):
    MIN = data.load_minutes(engine, SYMBOLS, *EXPLORE, extended=True)
    DAYS = data.load_days(engine, SYMBOLS, DAILY_FROM, EXPLORE[1])
    REPORTS = data.earnings_dates(list(STOCKS))
    OPTIONS = {_s: pipeline.options(_s, *EXPLORE) for _s in STOCKS}   # what the chains say
    _rows = []
    for _s in SYMBOLS:
        _reg = MIN[_s][MIN[_s]["regular"]]
        _built = data.daily_from_minutes(_reg)["close"]
        _gap = ((_built / DAYS[_s]["close"].reindex(_built.index) - 1).abs() * 100).dropna()
        _rows.append({"symbol": _s, "regular-hours minutes": len(_reg),
                      "minutes outside regular hours": len(MIN[_s]) - len(_reg),
                      "sessions": int(_built.size), "daily candles": len(DAYS[_s]),
                      "first daily candle": DAYS[_s].index.min().date(),
                      "close gap % (median)": float(np.median(_gap)) if len(_gap) else np.nan,
                      "close gap % (largest)": float(_gap.max()) if len(_gap) else np.nan})
    _reports = (f"**{len(REPORTS)}** report dates from Yahoo Finance." if len(REPORTS) else
                "**No report dates**: Yahoo Finance did not answer (the charts work without).")
    _options = "; ".join(f"{_s} {len(_m)} days ({_m.index.min().date()} to "
                         f"{_m.index.max().date()})" if len(_m) else f"{_s} none"
                         for _s, _m in OPTIONS.items())
    mo.vstack([mo.md("### Table 2 · What was loaded, and do minutes and days agree?"),
               table(pd.DataFrame(_rows).round(4)), mo.md(_reports),
               mo.md(f"**Option chains** (end of day, for the option rules): {_options}.")])
    return DAYS, MIN, OPTIONS, REPORTS


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The indicator settings

    Every slider redraws the charts below. They start at the values chosen under *Starting
    values* (built-in, or exported by the tuning notebook).

    - **Bollinger Bands**: an average of the last closes with a band above and below, a set
      number of standard deviations away. The bands widen when prices swing and narrow when
      they calm down
    - **Moving average**: an average of the last closes, drawn on the minute chart. The
      *moving average* filter (step 3) only trades in the direction it is going
    - **The switch "bands and average also read the hours outside the regular session"**: off,
      both read the regular session only (09:30-16:00), and the hours before the open enter as
      their own numbers (Figure 4c); on, both run through every stored minute, pre-market,
      after-hours and overnight included. They react earlier then, but the thin trading outside
      regular hours makes them jumpier
    - **Daily trend**: up when the close is above its average and the average is rising; down
      when below and falling; sideways otherwise. "Flat" means the average moved less than a
      fraction of a typical day's move (Figure 2b shows it day by day)
    - **Major levels**: prices where the daily chart turned (a high or low that stands out from
      the days around it) several times (Figure 2a shows how they are built)
    - **Typical move**: how far a candle typically moves (average true range). The trailing
      stop will be measured in it
    """)
    return


@app.cell
def _(HP, START, hyperparams, mo):
    def _w(name):     # a control built from the table, starting at the chosen value
        return hyperparams.widget(name, HP[name], START[name])

    candle_size, band_length, band_width = _w("candle_size"), _w("band_length"), _w("band_width")
    average_length, extended_bands = _w("average_length"), _w("extended_bands")
    trend_length, trend_slope, trend_flat = (_w("trend_length"), _w("trend_slope"),
                                             _w("trend_flat"))
    level_span, level_lookback = _w("level_span"), _w("level_lookback")
    level_tolerance, level_touches, level_mix = (_w("level_tolerance"), _w("level_touches"),
                                                 _w("level_mix"))
    move_size, move_length = _w("move_size"), _w("move_length")
    mo.vstack([
        mo.md("**Bollinger Bands**"),
        mo.hstack([candle_size, band_length, band_width], justify="start", gap=2),
        mo.md("**Moving average** (minute chart)"),
        mo.hstack([average_length, extended_bands], justify="start", gap=2),
        mo.md("**Daily trend** (stock, sector fund, market fund)"),
        mo.hstack([trend_length, trend_slope, trend_flat], justify="start", gap=2),
        mo.md("**Major levels**"),
        mo.hstack([level_span, level_lookback, level_tolerance, level_touches, level_mix],
                  justify="start", gap=2),
        mo.md("**Typical move** (the yardstick of the trailing stop)"),
        mo.hstack([move_size, move_length], justify="start", gap=2)])
    return (
        average_length,
        band_length,
        band_width,
        candle_size,
        extended_bands,
        level_lookback,
        level_mix,
        level_span,
        level_tolerance,
        level_touches,
        move_length,
        move_size,
        trend_flat,
        trend_length,
        trend_slope,
    )


@app.cell(hide_code=True)
def _(MIN, candle_size, data):
    CANDLES = {_s: data.resample(_m, candle_size.value) for _s, _m in MIN.items()}
    return (CANDLES,)


@app.cell(hide_code=True)
def _(CANDLES, STOCKS, band_length, band_width, extended_bands, pipeline):
    BANDS = {_s: pipeline.bands(CANDLES[_s], band_length.value, band_width.value,
                                extended_bands.value) for _s in STOCKS}
    return (BANDS,)


@app.cell(hide_code=True)
def _(CANDLES, STOCKS, average_length, extended_bands, pipeline):
    AVERAGE = {_s: pipeline.average(CANDLES[_s], average_length.value, extended_bands.value)
               for _s in STOCKS}
    return (AVERAGE,)


@app.cell(hide_code=True)
def _(CANDLES, MIN, STOCKS, move_length, move_size, pipeline):
    MOVE, MOVE_PCT = {}, {}     # in price (the trailing stop's yardstick) and in percent
    for _s in STOCKS:     # measured in the regular session, where the strategy trades
        MOVE[_s], MOVE_PCT[_s] = pipeline.move(MIN[_s], CANDLES[_s], move_size.value,
                                               move_length.value)
    return MOVE, MOVE_PCT


@app.cell(hide_code=True)
def _(MIN, STOCKS, ind):
    PRE = {_s: ind.before_open(MIN[_s]) for _s in STOCKS}   # the hours before each open
    return (PRE,)


@app.cell(hide_code=True)
def _(DAYS, pipeline, trend_flat, trend_length, trend_slope):
    TRENDS = pipeline.trends(DAYS, trend_length.value, trend_slope.value, trend_flat.value)
    return (TRENDS,)


@app.cell(hide_code=True)
def _(
    DAYS,
    STOCKS,
    level_lookback,
    level_mix,
    level_span,
    level_tolerance,
    level_touches,
    pipeline,
):
    LEVELS = {_s: pipeline.levels(DAYS[_s], level_span.value, level_lookback.value,
                                  level_tolerance.value, level_touches.value, level_mix.value)
              for _s in STOCKS}
    return (LEVELS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · The daily picture

    **Figure 2**, top: one stock's daily candles with a 20-day and the trend's moving average,
    and the material of the major levels:

    - **turning points** (black triangles): a day whose high is the highest of the days around
      it (a swing high, ▼ above the candle) or whose low is the lowest (a swing low, ▲ below it)
    - **major levels** (short black dashes, one on every day the level was known): prices where
      several turning points line up. A long row of dashes is a level that held for months; a
      thicker dash has more turning points behind it. **Figure 2a** shows, for one day, exactly
      which turning points make which level
    - **report days** (pink dotted lines)

    **Figure 2**, bottom: the three colored **trend ribbons**, explained right under the figure
    and day by day in **Figure 2b**. **Figure 3** counts how many days were up, sideways or down.
    """)
    return


@app.cell
def _(STOCKS, mo):
    pick = mo.ui.dropdown(list(STOCKS), value=list(STOCKS)[0], label="stock")
    mo.hstack([pick], justify="start")
    return (pick,)


@app.cell(hide_code=True)
def _(
    DAYS,
    EXPLORE,
    LEVELS,
    MARKET,
    REPORTS,
    STOCKS,
    TRENDS,
    charts,
    ind,
    level_span,
    mo,
    pd,
    pick,
    trend_flat,
    trend_length,
    trend_slope,
):
    _s, _sector = pick.value, STOCKS[pick.value]
    _all = DAYS[_s]
    _shown = _all.loc[pd.Timestamp(EXPLORE[0]):]
    _avg = {"20-day average": ind.moving_average(_all["close"], 20).loc[_shown.index],
            f"{trend_length.value}-day average (the trend's)":
                ind.moving_average(_all["close"], trend_length.value).loc[_shown.index]}
    _ribbons = {f"{_s} (the stock)": TRENDS[_s], f"{_sector} (its sector)": TRENDS[_sector],
                f"{MARKET} (the market)": TRENDS[MARKET]}
    _fig2 = charts.overview_figure(
        _s, _shown, averages=_avg, trends=_ribbons, levels=LEVELS[_s],
        turning_points=ind.pivots(_all, level_span.value),
        reports=REPORTS[REPORTS["symbol"] == _s] if len(REPORTS) else REPORTS,
        title=f"Figure 2 · {_s}: daily candles, turning points, major levels and the trends")
    _ribbon_words = mo.md(f"""
    **The colored ribbons under Figure 2** show the daily trend of three symbols: the stock
    ({_s}), its sector fund ({_sector}) and the market fund ({MARKET}). Each day is judged at its
    own close, by the same rule for all three:

    - **blue, up**: the close is above its {trend_length.value}-day average, AND that average
      rose over the last {trend_slope.value} days by more than {trend_flat.value:g} typical daily
      moves
    - **red, down**: the close is below its {trend_length.value}-day average, AND that average
      fell by more than {trend_flat.value:g} typical daily moves
    - **gray, sideways**: everything else: the average barely moved, or the price and the
      average disagree (the price below a rising average, or above a falling one)

    A "typical daily move" is the average size of the last 14 daily candles, gaps included, so
    the rule is equally strict for a calm fund and a jumpy stock. The strategy will act on a
    day's trend from the NEXT session on, since that close is not known during the day. The
    sliders under "Daily trend" change it; **Figure 2b** shows the calculation day by day.
    """)
    mo.vstack([_fig2, _ribbon_words])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### How the major levels are built (Figure 2a)

    Pick a day. Figure 2a zooms into the sessions before it and shows how the levels known at
    its open come about, in three steps, each with its own lever:

    1. **Find the turning points.** A day is a swing high when its high is the highest of the
       days on both sides of it (lever: *turning point: highest/lowest of ± days*), a swing low
       when its low is the lowest. A turning point only counts once those days after it have
       passed: before that, nobody could know the price would not go further. More days = fewer,
       more important turning points
    2. **Keep the recent ones.** Only turning points from the last *memory* sessions count.
       A longer memory brings in older levels
    3. **Group the ones that line up.** Turning points whose prices are closer than the
       *grouping distance* count as the same level (lever: *same level if closer than*, in
       typical daily moves; a typical daily move is the average size of the last 14 daily
       candles, overnight gaps included, and the price amount is printed under the figure). A
       group with at least *turning points needed* members is a **major level**, at the average
       price of its turning points. A wider distance merges more points into fewer, broader
       levels; more turning points needed keeps only the strongest

    The switch *a ceiling and a floor may form one level* decides whether swing highs and swing
    lows group together. **On**: a price where the stock turned up (a floor) and later turned
    down (a ceiling) is one level, since an old floor that breaks often becomes a ceiling.
    **Off**: swing highs make only ceiling (resistance) levels and swing lows only floor
    (support) levels. A point joins a group when it is within the grouping distance of its
    nearest neighbour in it, so a level of several points can be a little wider than the
    distance itself.

    Black triangles belong to a major level, hollow gray ones do not. **Hover a triangle**: it
    says from which session on the strategy may use it, and why it is (or is not) part of a
    level. The gray zone around each dashed level spans its turning points plus half the
    grouping distance on each side. The four levers below are the same as in the settings
    panel: moving one moves both.
    """)
    return


@app.cell
def _(
    DAYS,
    EXPLORE,
    level_lookback,
    level_mix,
    level_span,
    level_tolerance,
    level_touches,
    mo,
    pd,
    pick,
):
    _days = DAYS[pick.value].loc[pd.Timestamp(EXPLORE[0]):].index
    levels_day = mo.ui.date(start=_days[0].date(), stop=_days[-1].date(),
                            value=_days[-1].date(), label="levels known at the open of")
    mo.vstack([mo.hstack([levels_day], justify="start"),
               mo.hstack([level_span, level_lookback, level_tolerance, level_touches,
                          level_mix], justify="start", gap=2)])
    return (levels_day,)


@app.cell(hide_code=True)
def _(
    DAYS,
    charts,
    ind,
    level_lookback,
    level_mix,
    level_span,
    level_tolerance,
    level_touches,
    levels_day,
    mo,
    pd,
    pick,
    table,
):
    _s = pick.value
    _x = ind.explain_levels(DAYS[_s], levels_day.value, span=level_span.value,
                            lookback=level_lookback.value, tolerance=level_tolerance.value,
                            min_touches=level_touches.value, mix=level_mix.value)
    _day = _x["day"].date()
    _fig = charts.levels_figure(_s, DAYS[_s], _x, title=f"Figure 2a · {_s}: how the major "
                                f"levels known at the open of {_day} are built")
    _tp, _lv = _x["turning_points"], _x["levels"]
    _rows = []
    for _, _r in _lv.iterrows():
        _members = _tp[_tp["level"] == _r["price"]].sort_values("day")
        _rows.append({"level": round(_r["price"], 2), "turning points": int(_r["touches"]),
                      "swing highs": int(_r["highs"]), "swing lows": int(_r["lows"]),
                      "lowest": round(_r["bottom"], 2), "highest": round(_r["top"], 2),
                      "width": round(_r["top"] - _r["bottom"], 2),
                      "turning points on": ", ".join(
                          f"{_d:%Y-%m-%d} ({_k})" for _d, _k in
                          zip(_members["day"], _members["kind"], strict=True))})
    _words = mo.md(
        f"At the open of **{_day}**, the turning points from **{_x['window_start'].date()}** to "
        f"**{_x['known_until'].date()}** count: **{len(_tp)}** of them. A typical daily move (the "
        f"average size of the last 14 daily candles, gaps included) was "
        f"**{_x['typical_move']:.2f}**, so turning points closer than **{_x['tolerance']:.2f}** "
        f"({level_tolerance.value:g} typical moves) group together. **{len(_lv)}** groups have "
        f"at least {level_touches.value} turning points: the major levels. "
        f"**{int(_tp['level'].isna().sum())}** turning points stand alone.")
    mo.vstack([_fig, _words, mo.md("### Table 2a · The major levels of that day"),
               table(pd.DataFrame(_rows)) if _rows else mo.md("_No major level on that day._")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### How the trend is judged (Figure 2b)

    Pick the stock, its sector fund or the market fund. Figure 2b shows the rule at work, day by
    day, in three panels:

    1. **the daily close**, one dot per day colored by the result (blue up, gray sideways, red
       down), and the trend average. Hover a dot: it says why, in words
    2. **how much the average rose or fell** over the last days, against the gray **flat zone**:
       above the zone the average counts as rising, below it as falling, inside it as flat
    3. **the resulting ribbon**, the same as under Figure 2

    The three levers below are the same as in the settings panel.
    """)
    return


@app.cell
def _(MARKET, STOCKS, mo, pick, trend_flat, trend_length, trend_slope):
    _s = pick.value
    trend_of = mo.ui.dropdown({f"{_s} (the stock)": _s, f"{STOCKS[_s]} (its sector)": STOCKS[_s],
                               f"{MARKET} (the market)": MARKET},
                              value=f"{_s} (the stock)", label="explain the trend of")
    mo.vstack([mo.hstack([trend_of], justify="start"),
               mo.hstack([trend_length, trend_slope, trend_flat], justify="start", gap=2)])
    return (trend_of,)


@app.cell(hide_code=True)
def _(DAYS, EXPLORE, charts, ind, pd, trend_flat, trend_length, trend_of, trend_slope):
    _sym = trend_of.value
    _kw = dict(length=trend_length.value, slope_days=trend_slope.value, flat=trend_flat.value)
    _trend = ind.trend_state(DAYS[_sym], **_kw)
    charts.trend_figure(
        _sym, DAYS[_sym].loc[pd.Timestamp(EXPLORE[0]):], _trend,
        ind.explain_trend(DAYS[_sym], _trend, **_kw), **_kw,
        title=f"Figure 2b · How the trend of {_sym} is judged: {trend_length.value}-day average, "
              f"its rise over {trend_slope.value} days, flat zone ± {trend_flat.value:g} typical "
              "daily moves")
    return


@app.cell(hide_code=True)
def _(EXPLORE, MARKET, STOCKS, TRENDS, charts, pd):
    _labels = {}
    for _stock, _fund in STOCKS.items():
        _labels[f"{_stock} (stock)"] = _stock
        _labels[f"{_fund} (sector of {_stock})"] = _fund
    _labels[f"{MARKET} (market)"] = MARKET
    charts.conditions_figure(
        {_k: TRENDS[_v].loc[pd.Timestamp(EXPLORE[0]):] for _k, _v in _labels.items()},
        title="Figure 3 · How the days of EXPLORE split into up, sideways and down "
              "(the strategy must work in all three)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Minute by minute

    Pick a first session and how many to show. **Figure 4** has five panels on one time axis
    (nights and weekends are cut out):

    1. **Candles** with the Bollinger Bands (gray, shaded between), the moving average (amber),
       the major levels known at each session's open (black dashes) and the previous
       session's high and low (gray dots). Candles outside regular hours are pale (choose which
       hours to show: regular only, 04:00-20:00, or all 24); dotted lines mark the regular open
       (09:30) and close; the pre-market's high and low (gray dash-dots) run across the regular
       session
    2. **Where the close sits in the band**: 0 on the lower band, 1 on the upper band; dots mark
       the closes outside the band, the moments a band signal will look at
    3. **The stock against its sector and the market**, in percent since the first candle
    4. **The typical move**: how far the price typically travels within one candle of the
       size set under "Typical move" (15 minutes by default: its high minus its low, averaged
       over the last 14 such candles), in percent of the price. The trailing stop will keep a
       multiple of it behind the price. **Figure 4b** shows how it is computed
    5. **Volume**

    **Table 3** lists what was known at each session's open, the night before it included.
    """)
    return


@app.cell
def _(MIN, mo, pick):
    _days = MIN[pick.value].index.tz_localize(None).normalize().unique()
    first_day = mo.ui.date(start=_days[0].date(), stop=_days[-1].date(),
                           value=_days[-3].date(), label="first session")
    n_sessions = mo.ui.slider(1, 10, step=1, value=3, show_value=True, label="sessions shown")
    hours_shown = mo.ui.dropdown({"regular hours (09:30-16:00)": "regular",
                                  "04:00-20:00 (+ pre-market and after-hours)": "extended",
                                  "all 24 hours (+ overnight)": "all"},
                                 value="04:00-20:00 (+ pre-market and after-hours)",
                                 label="hours shown")
    mo.hstack([first_day, n_sessions, hours_shown], justify="start", gap=2)
    return first_day, hours_shown, n_sessions


@app.cell(hide_code=True)
def _(
    AVERAGE,
    BANDS,
    CANDLES,
    DAYS,
    LEVELS,
    MARKET,
    MOVE_PCT,
    PRE,
    REPORTS,
    STOCKS,
    TRENDS,
    average_length,
    candle_size,
    charts,
    first_day,
    ind,
    mo,
    move_length,
    move_size,
    n_sessions,
    pd,
    data,
    hours_shown,
    pick,
    table,
):
    _s, _sector = pick.value, STOCKS[pick.value]
    _c = CANDLES[_s]
    _day = pd.Series(_c.index.tz_localize(None).normalize(), index=_c.index)
    _sessions = pd.DatetimeIndex(_day.unique())
    _start = min(int(_sessions.searchsorted(pd.Timestamp(first_day.value))), len(_sessions) - 1)
    _show = _sessions[_start:_start + n_sessions.value]
    _part = data.session_part(_c.index)             # 0 pre, 1 regular, 2 after, 3 overnight
    _keep = {"regular": _part == 1, "extended": _part < 3, "all": _part >= 0}[hours_shown.value]
    _in = _day.isin(_show).to_numpy() & _keep
    _ctx = ind.session_context(DAYS[_s], {"stock trend": TRENDS[_s],
                                          "sector trend": TRENDS[_sector],
                                          "market trend": TRENDS[MARKET]}, _show)
    _pre = PRE[_s].reindex(_show)
    _ctx["move before the open %"] = _pre["move %"].to_numpy()
    _ctx["pre-market high"] = _pre["pre-market high"].to_numpy()
    _ctx["pre-market low"] = _pre["pre-market low"].to_numpy()
    _ctx["night volume × usual"] = _pre["volume vs usual"].to_numpy()
    _reports = REPORTS[REPORTS["symbol"] == _s] if len(REPORTS) else REPORTS
    _fig = charts.day_figure(
        _s, _c[_in], bands=BANDS[_s][_in], average=AVERAGE[_s][_in],
        average_label=f"{average_length.value}-candle average",
        others={f"{_sector} (sector)": CANDLES[_sector]["close"],
                f"{MARKET} (market)": CANDLES[MARKET]["close"]},
        move_pct=MOVE_PCT[_s][_in], context=_ctx, levels=LEVELS[_s], reports=_reports,
        move_label=f"typical move of a {move_size.value}-minute candle "
                   f"(average of the last {move_length.value})",
        title=f"Figure 4 · {_s}, {candle_size.selected_key} candles, "
              f"{pd.Timestamp(_show[0]).date()} → {pd.Timestamp(_show[-1]).date()}")
    _lv = LEVELS[_s]
    _ctx["major levels"] = [", ".join(f"{p:.2f} ({t}x)" for p, t in zip(
        _lv.loc[_lv["day"] == d, "price"], _lv.loc[_lv["day"] == d, "touches"], strict=True))
        or "none" for d in _ctx["day"]]
    _ctx["day"] = _ctx["day"].dt.date
    mo.vstack([_fig, mo.md("### Table 3 · Known at each session's open"),
               table(_ctx.round(2))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What the typical move is (Figure 4b)

    The typical move says how far the price normally travels in a short while, so that a stop
    can be set wide enough not to be hit by ordinary noise, and no wider. It is computed from
    candles of a fixed size (lever: *candle size*, 15 minutes by default) in three steps:

    1. **Each candle's move**: its high minus its low. If the previous candle closed outside
       that range, the move is stretched to reach it (this is called the "true range"). The jump
       over the night is left out, so one gap at the open does not inflate the whole morning
    2. **Average the last few**: the typical move is the average of the last *candles averaged*
       moves (14 by default). It only uses candles that have closed: while a candle runs, its own
       move is not known yet
    3. **In percent of the price**, so a $100 and a $1,000 stock compare

    **Figure 4b** shows the sessions of Figure 4 on candles of that size: the gray bars are each
    candle's move, the green line is the typical move the strategy knows while the candle runs.
    A bar above the line is a candle that moved more than usual. The trailing stop (step 2)
    will sit a multiple of the line behind the best price: with a typical move of 0.5%, a stop
    of 3 typical moves trails 1.5% behind.
    """)
    return


@app.cell
def _(mo, move_length, move_size):
    mo.hstack([move_size, move_length], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(MIN, charts, data, first_day, ind, move_length, move_size, n_sessions, pd, pick):
    _s, _size = pick.value, move_size.value
    _blocks = data.resample(MIN[_s][MIN[_s]["regular"]], _size)
    _explained = ind.explain_move(_blocks, move_length.value)
    _day = pd.Series(_blocks.index.tz_localize(None).normalize(), index=_blocks.index)
    _sessions = pd.DatetimeIndex(_day.unique())
    _start = min(int(_sessions.searchsorted(pd.Timestamp(first_day.value))), len(_sessions) - 1)
    _show = _sessions[_start:_start + n_sessions.value]
    _in = _day.isin(_show).to_numpy()
    charts.move_figure(
        _s, _blocks[_in], _explained[_in], size=_size, length=move_length.value,
        title=f"Figure 4b · {_s}: how the typical move of a {_size}-minute candle is computed "
              f"(average of the last {move_length.value}), {pd.Timestamp(_show[0]).date()} → "
              f"{pd.Timestamp(_show[-1]).date()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The hours before the open (Figure 4c)

    Everything that happens between the previous session's close (16:00) and the next open
    (09:30): the after-hours, the overnight (where stored) and the pre-market, summed up per
    session. All of it is known at 09:30, before the strategy may trade:

    - **move before the open**: from the previous close to the last trade before 09:30, in
      percent
    - **pre-market high and low**: the highest and lowest trade from 04:00 to 09:30 (drawn in
      Figure 4 as gray dash-dots across the regular session); the night's own high and low
      (since the previous close) are kept too
    - **volume against its usual level**: the shares traded between the close and the open,
      divided by their average over the previous 20 sessions. 1 is a usual night, 3 means three
      times as busy, which often comes with news

    Only minutes with a trade count: a minute without one just repeats an older price.
    **Figure 4c** shows these numbers for every session of EXPLORE; **Table 3b** how big they
    typically are.
    """)
    return


@app.cell(hide_code=True)
def _(EXPLORE, PRE, charts, mo, np, pd, pick, table):
    _s = pick.value
    _p = PRE[_s].loc[pd.Timestamp(EXPLORE[0]):]
    _move = _p["move %"].abs()
    _summary = pd.DataFrame([{
        "sessions": int(_p["move %"].notna().sum()),
        "typical size of the move before the open (median), %": float(np.nanmedian(_move)),
        "sessions moving more than 1% before the open": float((_move > 1).mean()),
        "sessions moving more than 3%": float((_move > 3).mean()),
        "sessions with 3 × the usual volume or more": float((_p["volume vs usual"] >= 3).mean()),
        "minutes with a trade between close and open (median)": float(
            _p["minutes traded"].median())}])
    mo.vstack([
        charts.before_open_figure(_s, _p, title=f"Figure 4c · {_s}: the move before each open "
                                  "and the volume since the previous close against its usual "
                                  "level"),
        mo.md("### Table 3b · How big the hours before the open typically are"),
        table(_summary.round(3))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Band behaviour

    Over every regular-hours candle of EXPLORE (where the strategy trades): where the close sits
    in its band (**Figure 5**), and how often it closes outside (**Table 4**). The share outside
    the bands is how often a band signal can even start; wider bands make it rarer.
    """)
    return


@app.cell(hide_code=True)
def _(BANDS, CANDLES, band_length, band_width, charts, mo, pd, table):
    _pos = {_s: _b["position"][CANDLES[_s]["regular"]].dropna() for _s, _b in BANDS.items()}
    _rows = [{"stock": _s, "regular-hours candles": int(_p.size),
              "close above the upper band": float((_p > 1).mean()),
              "close below the lower band": float((_p < 0).mean()),
              "inside the band": float(((_p >= 0) & (_p <= 1)).mean())}
             for _s, _p in _pos.items()]
    mo.vstack([
        charts.band_position_figure(_pos, title=f"Figure 5 · Where the close sits in its band "
                                    f"({band_length.value} candles, {band_width.value:g} "
                                    "standard deviations)"),
        mo.md("### Table 4 · How often the close is outside the band"),
        table(pd.DataFrame(_rows).round(4))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · The trading rules (steps 2 and 3)

    From here on the strategy trades: in the regular session only (09:30-16:00), each stock on
    its own, one trade at a time, long or short. Every rule is a switch. The three filters added
    in step 3 (moving average, stock trend, market trend) start switched off, so the default is
    still step 2's strategy; section 8 shows what each one changes.

    **Triggers** propose a trade at a candle's close:

    - **band fade**: the close comes back inside the bands after a close outside them: long when
      it climbs back above the lower band, short when it falls back below the upper band. The
      bet: the stretch snaps back
    - **band breakout**: the close stays outside the bands for a set number of candles in a row
      (lever *closes in a row outside the band*, 2 by default): long above the upper band, short
      below the lower band. One close outside alone may still turn straight back inside, which
      is the fade's case; the breakout fires once per stretch, at the close of its n-th candle.
      The bet: the move carries on

    **Filters** allow or block a side. The four trend filters share one rule: up allows longs
    only, down shorts only, sideways (or not known yet) both:

    - **moving average** (step 3): the trend of the minute chart's moving average (the amber
      line of Figure 4), judged at each candle's close: up when the close is above the average
      AND the average rose over the last candles (lever) by more than a flat zone (lever, in
      typical moves); down when below a falling average; sideways otherwise
    - **stock trend** (step 3): the stock's own daily trend (Figure 2b), judged at the previous
      session's close. It is also the market condition every result is split by
    - **sector trend**: the same daily rule on the sector fund (IGV for NOW, XLI for GEV, QTUM
      for RGTI)
    - **market trend** (step 3): the same daily rule on the market fund (SPY)
    - **major levels** (step 4): no long while the nearest major level above the close (the
      levels of Figure 2a, known at the open) is closer than a set room (lever, in typical
      moves: 2 by default), no short while the nearest one below is; a level right at the close
      blocks both
    - **previous high / low** (step 4): the same rule on the previous session's high and low
      (whichever is above the price is a ceiling, whichever is below a floor)
    - **wait after the open**: no new trade in the first minutes after 09:30 (lever, 15 by
      default): a signal whose trade would fill in that time is dropped. Trades already open
      keep their stop
    - **earnings guard** (step 5, on by default): no new trade in the session(s) before a
      quarterly report's reaction day (the first session that can move on it: the same day for
      a report before the open, the next for one after the close), and an open trade is
      closed at the close before it. A report can move a stock 10% or more over a night
    - **options calm** (off by default): no new trade while the options market expects a much
      bigger move than the stock has been making: the at-the-money straddle of the nearest
      expiry at least 2 days out, as a daily move, more than a set multiple (lever, 1.8 by
      default) of the stock's last 20 daily moves, read from the previous session's end-of-day
      chain. About 1.1 on an ordinary day, 2 to 5 before a report: an event ahead, when gaps run
      through stops. Options tell how much a stock may move, not which way, so both sides are
      blocked. The chains start in October 2025; a day without them allows both sides
    - **iv term** (off by default): no new trade while the nearest options' implied volatility
      is more than a set multiple (1.3) of the 30-day options' (an event priced ahead). The
      volatilities are solved from the chain's mid quotes by Black-Scholes (the chains carry
      none)
    - **put/call** (off by default): when far more puts than usual traded the day before (the
      put/call volume over its 20-day median, from 1.5x), shorts only, and longs only when far
      more calls did ("with the option buyers"); or the other way round (a lever)
    - **dealer gamma** (off by default): new trades only while the option dealers' gamma (+ call
      − put gamma over all open interest, over its usual size) is at least a set level (0:
      positive). Positive gamma damps moves, which suits fades
    - **option walls** (off by default): the major-levels rule on the strikes with the most call
      open interest above and put open interest below the price (the next 45 days); they can
      also be the profit target

    **How they decide together.** At every candle's close the switched-on triggers propose. If
    two of them propose opposite sides, the signal is not clear and nothing happens. A proposed
    side also needs every switched-on filter to allow it. The trade is then bought (long) or sold
    short at the NEXT candle's open, in the same session: a signal on a session's last candle is
    dropped, since the next open comes after a night. While a trade is open, new signals are
    ignored.

    **How a trade ends: the stop**, one of three kinds (under "The exit"):

    - **trailing, typical moves** (as from step 2): it starts a set number of typical moves
      (Figure 4b) away from the entry price. After every candle it moves up to the same distance
      below the highest price since the entry (for a long; above the lowest price for a short),
      and never back
    - **trailing, typical daily moves**: the same, a set number of typical DAILY moves (the
      average daily range of the last 14 sessions) behind the best price
    - **fixed, typical daily moves**: a set number of typical daily moves from the entry price,
      never moved

    Daily moves make one setting mean the same for a calm and a jumpy stock.

    A candle that trades through the stop closes the trade at the stop price. A candle that
    OPENS beyond it (after a night, or a jump) closes the trade at that open, which is the real
    loss. Optionally, a holding limit closes the trade at the close of its n-th session.

    **The profit target** (optional, under "The exit"), fixed when the trade opens: the next
    level in the trade's direction, at least one typical move from the entry (the next major
    level, or the next major level or previous session's high or low), or a set number of
    typical daily moves from the entry. A candle that reaches it closes the trade there; one
    that opens beyond it fills at that better open. If the stop and the target fall in the same
    candle, the stop counts (the cautious reading).

    The **market condition** of a trade is the stock's own daily trend at the entry (judged at
    the previous close): up, sideways or down. The strategy is meant to hold up in all three.
    """)
    return


@app.cell
def _(HP, RULES, START, hyperparams, mo):
    switches = mo.ui.dictionary({     # every rule's switch, at its starting position
        _n: mo.ui.switch(value=bool(START[hyperparams.switch_key(_n)]), label=_n)
        for _n in RULES})
    knobs = mo.ui.dictionary({     # every rule's PARAMS, one slider each
        _k: hyperparams.widget(_k, HP[_k], START[_k])
        for _r in RULES.values() for _k in _r.PARAMS})
    exits = mo.ui.dictionary({     # the stop, the holding limit and the profit target
        _k: hyperparams.widget(_k, _r, START[_k]) for _k, _r in HP.items()
        if _r["stage"] == "exit"})
    _rows = [mo.md("**The rules** (a new rule is one new file in `rules/`; its switch and its "
                   "levers appear here)")]
    for _n, _r in RULES.items():
        _rows.append(mo.hstack([switches[_n], mo.md(f"*{_r.KIND}*: {_r.DESCRIPTION}")],
                               justify="start", gap=1))
        if _r.PARAMS:
            _rows.append(mo.hstack([knobs[_k] for _k in _r.PARAMS], justify="start", gap=2))
    mo.vstack([*_rows, mo.md("**The exit** (a stop's distance and the daily-move target only "
                             "act with their kind)"),
               mo.hstack([exits["stop_kind"], exits["stop_moves"], exits["stop_days"]],
                         justify="start", gap=2),
               mo.hstack([exits["max_sessions"], exits["target_kind"], exits["target_days"]],
                         justify="start", gap=2)])
    return exits, knobs, switches


@app.cell
def _(
    average_length,
    band_length,
    band_width,
    candle_size,
    extended_bands,
    mo,
    move_length,
    move_size,
    trend_flat,
    trend_length,
    trend_slope,
):
    # the indicator sliders of the settings panel, shown again here (in their own cell, so moving
    # one never resets the switches above)
    mo.accordion({"The indicator settings the rules read (the same sliders as at the top)":
                  mo.vstack([
                      mo.md("**Bollinger Bands** (band fade, band breakout)"),
                      mo.hstack([candle_size, band_length, band_width], justify="start", gap=2),
                      mo.md("**Moving average** (the moving average filter)"),
                      mo.hstack([average_length, extended_bands], justify="start", gap=2),
                      mo.md("**Daily trend** (stock, sector and market trend)"),
                      mo.hstack([trend_length, trend_slope, trend_flat], justify="start", gap=2),
                      mo.md("**Typical move** (the trailing stop, the moving average's flat zone)"),
                      mo.hstack([move_size, move_length], justify="start", gap=2)])})
    return


@app.cell(hide_code=True)
def _(CANDLES, STOCKS, mo, pd, settings, simulate, table):
    COSTS = simulate.costs(settings)     # the same size and costs as the tuning notebook's
    _rows = []
    for _s in STOCKS:
        _price = float(CANDLES[_s].loc[CANDLES[_s]["regular"], "close"].median())
        _shares = 10_000.0 / _price          # a $10,000 position, to show the costs' scale
        _order = max(COSTS["commission"], _shares * COSTS["per_share"])
        _fill = _shares * max(_price * COSTS["slippage"], COSTS["min_slip"])
        _rows.append({"stock": _s, "typical price $": _price, "shares in $10,000": _shares,
                      "commission per order $": _order, "spread and slippage per fill $": _fill,
                      "round trip $": 2 * (_order + _fill),
                      "round trip %": 2 * (_order + _fill) / 10_000.0 * 100})
    mo.vstack([mo.md(f"""
    **Size.** Every trade risks **${COSTS['risk']:,.0f}** at its first stop: it buys as many
    shares as lose that much if the stop is hit (a gap past the stop loses more), at most
    **${COSTS['notional']:,.0f}** worth. So a calm and a jumpy stock, or a calm and a wild
    quarter, count alike in the results.

    **Costs** (the repo's cost model, `config/settings.yaml`). Commission **${COSTS['per_share']}
    a share, at least ${COSTS['commission']:.2f} an order** (a $10,000 position of a stock above
    $50 pays the minimum; Rigetti at $10 is 1,000 shares, $5 an order).
    Every fill loses **{COSTS['slippage'] * 100:.2f}%** of the price (half the bid-ask spread plus
    slippage), but at least **half a cent a share**: half the smallest possible spread of one
    cent, which is more than 0.02% below $25 a share. The table shows what a round trip of a
    $10,000 position costs at each stock's typical price: a trade must move more than that
    before it makes money.
    """), table(pd.DataFrame(_rows).round(3))])
    return (COSTS,)


@app.cell(hide_code=True)
def _(
    AVERAGE,
    BANDS,
    CANDLES,
    COSTS,
    DAYS,
    LEVELS,
    MARKET,
    MOVE,
    OPTIONS,
    REPORTS,
    RULES,
    STOCKS,
    TRENDS,
    combine,
    exits,
    hyperparams,
    ind,
    knobs,
    pd,
    pipeline,
    results,
    simulate,
    switches,
):
    CTX, DECISIONS, SIMS, TARGETS, PREV = {}, {}, {}, {}, {}
    for _s, _sector in STOCKS.items():     # the same computation as the tuning notebook's
        _names = {"stock": _s, "sector": _sector, "market": MARKET}
        PREV[_s] = ind.previous_range(DAYS[_s])      # the previous session's high and low
        CTX[_s] = pipeline.context(_names, CANDLES[_s], BANDS[_s], MOVE[_s], TRENDS,
                                   AVERAGE[_s], LEVELS[_s], PREV[_s], REPORTS, DAYS[_s],
                                   OPTIONS[_s])
        TARGETS[_s] = pipeline.targets(CTX[_s], LEVELS[_s], PREV[_s],
                                       exits.value["target_kind"], exits.value["target_days"],
                                       pipeline.walls(OPTIONS[_s], DAYS[_s]))
        DECISIONS[_s] = combine.decide(CTX[_s], RULES, switches.value, knobs.value)
        SIMS[_s] = simulate.run(CTX[_s], DECISIONS[_s], **hyperparams.stop_args(exits.value),
                                targets=TARGETS[_s], **COSTS)
        SIMS[_s]["trades"]["setup"] = results.triggers(SIMS[_s]["trades"], DECISIONS[_s],
                                                       RULES, switches.value)
    TRADES = pd.concat([SIMS[_s]["trades"] for _s in STOCKS], ignore_index=True)
    return CTX, DECISIONS, PREV, SIMS, TARGETS, TRADES


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What every switch says (Figure 6)

    The sessions picked below, regular hours only. Top: the candles, the Bollinger Bands, the
    moving average (amber), the trailing stop of an open trade (black dots, one step per
    candle) and the trades (blue ▲ a
    long entry, orange ▼ a short entry, ✕ the exit; hover a marker for the result). Bottom: one
    row per rule, then the decision and whether a trade is open. Hover a cell for its meaning:

    - a **trigger** row: blue = proposes long, orange = proposes short
    - a **filter** row: light gray = both sides allowed, blue = longs only, orange = shorts only,
      dark gray = neither side
    - **decision**: blue / orange = a long / short signal (filled at the next candle's open);
      dark gray = a proposal that a filter blocked, or two triggers that disagreed
    - **in a trade**: blue = long, orange = short

    A switched-off rule keeps its row, marked "off": it shows what it would have said, but it
    takes no part in the decision. A signal while a trade is open is ignored.
    """)
    return


@app.cell
def _(first_day, mo, n_sessions, pick):
    mo.hstack([pick, first_day, n_sessions], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(
    CTX,
    DECISIONS,
    LEVELS,
    PREV,
    RULES,
    SIMS,
    candle_size,
    charts,
    first_day,
    n_sessions,
    pd,
    pick,
    switches,
):
    _s = pick.value
    _c = CTX[_s]["candles"]
    _day = _c.index.tz_localize(None).normalize()
    _sessions = _day.unique()
    _start = min(int(_sessions.searchsorted(pd.Timestamp(first_day.value))), len(_sessions) - 1)
    _show = _sessions[_start:_start + n_sessions.value]
    _in = _day.isin(_show)
    charts.switchboard_figure(
        _s, _c[_in], CTX[_s]["bands"][_in], DECISIONS[_s][_in], SIMS[_s], RULES, switches.value,
        average=CTX[_s]["average"][_in], levels=LEVELS[_s], previous=PREV[_s],
        title=f"Figure 6 · {_s}, {candle_size.selected_key} candles, {_show[0].date()} → "
              f"{_show[-1].date()}: what every switch says")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Results by market condition and side

    Every trade of EXPLORE, every stock, after costs. **Figure 7** shows the average result per
    trade by the market condition at the entry (the stock's own daily trend) and by side; the
    label on each bar is the number of trades. A strategy that works in every market has bars
    above zero in all three conditions. **Table 5** gives the details, **Table 5b** splits the
    same trades by setup (the trigger that fired).

    The columns: *winners* is the share of trades that made money; *average* and *median return
    %* are per trade, in percent of the $10,000; the *profit factor* is the dollars won divided
    by the dollars lost (above 1: more won than lost); *sessions held* counts the entry's own
    session as 1.
    """)
    return


@app.cell(hide_code=True)
def _(TRADES, charts, mo, results, table):
    _by = results.summary(TRADES, ["stock", "condition", "side"])
    mo.vstack([
        charts.results_figure(_by, title="Figure 7 · Average result per trade (after costs) by "
                              "market condition and side"),
        mo.md("### Table 5 · By stock, market condition and side"), table(_by.round(3)),
        mo.md("### Table 5b · By stock, setup and side"),
        table(results.summary(TRADES, ["stock", "setup", "side"]).round(3))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The profit curve and the bottom line (Figure 8, Table 6)

    **Figure 8**: the profit in dollars, added up trade after trade (a step at every exit), per
    stock and for all together; below it, how far the total is under its best level so far (the
    drawdown). **Table 6** holds the number the optimizer will judge later (step 5), for each
    stock on its own: the total profit divided by the worst drawdown. 2 means the strategy made
    twice as much as it ever gave back from a peak. For reference, the same numbers for simply
    buying $10,000 of the stock on the first day and holding it to the last.

    The strategy's drawdown is read at every exit, holding's at every session's close: a loss
    inside an open trade only counts once the trade has closed.
    """)
    return


@app.cell(hide_code=True)
def _(COSTS, CTX, SIMS, STOCKS, TRADES, charts, mo, pd, results, table):
    _costs = {_k: _v for _k, _v in COSTS.items() if _k != "risk"}
    _held = [results.holding(_s, CTX[_s]["candles"], **{**_costs, "notional": 10_000.0})
             for _s in STOCKS]           # a held position has no stop: $10,000 of the stock
    _line = pd.concat([results.bottom_line(TRADES, SIMS), pd.DataFrame(_held)],
                      ignore_index=True)
    mo.vstack([
        charts.equity_figure(results.equity(TRADES), title="Figure 8 · The profit curve (closed "
                             "trades, after costs) and how far it is below its best"),
        mo.md("### Table 6 · The bottom line: profit ÷ worst drawdown"), table(_line.round(3))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7 · Every trade, and why (Table 7, Figure 9)

    **Table 7** lists every trade; search, sort and filter it in the table. **Click a row**:
    Figure 9 draws that trade from an hour of candles before its signal to half an hour after
    its exit, with the bands, the trailing stop as it moved, the signal, the entry and the exit.
    The text under it tells the trade's story: which trigger fired and why, what each filter
    said, where the stop started and why the trade ended. Without a click it shows the first
    trade.
    """)
    return


@app.cell(hide_code=True)
def _(TRADES, mo):
    _times = {_k: TRADES[_k].dt.strftime("%Y-%m-%d %H:%M")
              for _k in ("signal time", "entry time", "exit time")}
    _shown = TRADES.assign(**{"#": TRADES.index}, **_times)[
        ["#", "stock", "side", "setup", "condition", "signal time", "entry time", "exit time",
         "entry price", "exit price", "first stop", "exit reason", "sessions", "return %",
         "P&L $"]].round(3)
    trade_list = mo.ui.table(_shown, selection="single", page_size=10,
                             label="Table 7 · Every trade (click a row to inspect it)")
    mo.vstack([trade_list])
    return (trade_list,)


@app.cell(hide_code=True)
def _(
    CTX,
    DECISIONS,
    RULES,
    SIMS,
    TRADES,
    charts,
    exits,
    knobs,
    mo,
    results,
    switches,
    trade_list,
):
    mo.stop(TRADES.empty, mo.md("_No trades: switch on at least one trigger._"))
    _picked = trade_list.value
    _i = int(_picked["#"].iloc[0]) if len(_picked) else int(TRADES.index[0])
    _t = TRADES.loc[_i]
    _s = _t["stock"]
    mo.vstack([
        charts.trade_figure(_t, CTX[_s]["candles"], CTX[_s]["bands"], SIMS[_s]["stop"],
                            title=f"Figure 9 · Trade #{_i}: {_s} {_t['side']}, "
                                  f"{_t['entry time']:%Y-%m-%d %H:%M} → "
                                  f"{_t['exit time']:%Y-%m-%d %H:%M}, {_t['return %']:+.2f}%"),
        mo.md(results.explain_trade(_t, CTX[_s], DECISIONS[_s], RULES, switches.value,
                                    knobs.value, exit_=exits.value))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8 · What each switch changes (step 3)

    For every rule, the whole strategy runs twice: once with that rule switched off and once
    with it switched on, every other switch, lever and setting as set above.

    - **Figure 10**: one line per rule, per stock and for all together. The hollow circle is the
      total profit after costs with the rule off, the filled one with it on (blue when switching
      it on made more, red when less). The label says how the rule is set now. Hover for the
      trades and profit ÷ worst drawdown
    - **Figure 11**: where that change comes from. Each cell is the change in profit from
      switching the rule on, within one market condition (the stock's own daily trend at the
      entry) or one half-year (by the exit). An empty cell had no trade either way
    - **Table 8**: the numbers

    **How to read it.** Step 2 showed that the total moves by several thousand dollars with
    small changes of any lever, by chance alone. So read a switch as helpful only if it helps in
    most cells of Figure 11, for every stock: in each market condition and in each half-year,
    not only in the total.
    """)
    return


@app.cell(hide_code=True)
def _(
    COSTS,
    CTX,
    RULES,
    STOCKS,
    TARGETS,
    TRADES,
    combine,
    exits,
    hyperparams,
    knobs,
    pd,
    results,
    simulate,
    switches,
):
    FLIPPED = {}     # rule -> the trades with only that rule's switch flipped
    for _rule in RULES:
        _sw = {**switches.value, _rule: not switches.value[_rule]}
        FLIPPED[_rule] = pd.concat([
            simulate.run(CTX[_s], combine.decide(CTX[_s], RULES, _sw, knobs.value),
                         **hyperparams.stop_args(exits.value), targets=TARGETS[_s],
                         **COSTS)["trades"]
            for _s in STOCKS], ignore_index=True)
    EFFECTS = results.switch_effects(TRADES, FLIPPED, switches.value, list(STOCKS))
    BREAKDOWN = results.effect_breakdown(TRADES, FLIPPED, switches.value, list(STOCKS))
    return BREAKDOWN, EFFECTS


@app.cell(hide_code=True)
def _(BREAKDOWN, EFFECTS, charts, mo, table):
    mo.vstack([
        charts.switch_effect_figure(EFFECTS, title="Figure 10 · What each switch changes: total "
                                    "profit after costs with the rule off (○) and on (●)"),
        charts.effect_breakdown_figure(BREAKDOWN, title="Figure 11 · Where the change comes "
                                       "from: switching the rule on, by market condition and by "
                                       "half-year ($)"),
        mo.md("### Table 8 · What each switch changes"),
        table(EFFECTS.round(2), rows=24)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next: tuning (step 5), then the final check (step 6)

    The tuning has its own notebook, `classic_signals_tuning_marimo.py`: every hyperparameter of
    this notebook (the rule switches included) in one table, the two passes with the blended
    score, which hyperparameters move the score at all, the walk forward of the whole
    procedure, and an export of the winning settings per stock. Pick an exported file under
    *Starting values* at the top to inspect those settings here.

    The final check (step 6): the procedure chosen there runs once on the untouched months,
    July to October 2026, and is judged there, without changing anything after looking.
    """)
    return


if __name__ == "__main__":
    app.run()
