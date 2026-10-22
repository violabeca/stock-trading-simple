import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # TimesFM 3.0, zero-shot: how well does a pre-trained model forecast stock prices?

    **TimesFM 3.0** is Google's pre-trained forecasting model (330 million parameters, released
    August 2026, shipped in Darts 0.48 as `TimesFM3Model`). It is used here **zero-shot**: it
    reads a stock's recent closes and forecasts the next bars without ever being trained or tuned
    on stocks. Every forecast comes as three paths: the **median** (half the outcomes above, half
    below) and a **10-90 % band** that the price should stay inside 80 % of the time.

    Two questions, one section each:

    1. **Intraday**: from 1-minute bars, the next hour (never past the close)
    2. **Multi-day**: from 10-minute bars, the next five sessions (nights included)

    and a third that reuses both sections' forecasts: 3. **Size**: does TimesFM's band know how
    big the coming move will be better than a cheap volatility rule?

    Each section first shows single forecasts against the actual price, then grades every
    forecast in the window three ways:

    - **path error**: how far the median path is from the actual one, against the forecast
      **"no change"** (the price stays where it is). Prices are close to a random walk, so
      "no change" is the honest bar: a forecast that cannot beat it adds nothing.
    - **direction**: does the median end on the right side of the price when the forecast is
      made?
    - **calibration**: do the actual closes stay inside the 10-90 % band 80 % of the time?

    Everything runs twice, on the **price only** and on the **price + volume** (volume as a past
    input, which TimesFM 3 can read beside the price). Beside every number sits a **luck band**:
    the same forecasts graded against the actual moves with their signs flipped at random (one
    coin per day, shared by every stock), which keeps each move's size and destroys any direction
    the model knows. A number inside its luck band is what a model without a clue gets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(df, rows=15):
        """A DataFrame as a marimo table; a named index becomes a column."""
        return mo.ui.table(
            df.reset_index() if df.index.name else df, selection=None, page_size=rows
        )

    return (table,)


@app.cell
def _():
    import importlib.util
    import logging
    import warnings

    import numpy as np
    import pandas as pd

    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    warnings.filterwarnings("ignore")
    for _name in ("darts", "pytorch_lightning", "lightning.pytorch"):
        logging.getLogger(_name).setLevel(logging.ERROR)  # Darts' and Lightning's notes

    _path = strategies.STRATEGIES_DIR / "darts_forecast" / "timesfm_lab.py"
    _spec = importlib.util.spec_from_file_location("darts_timesfm_lab", _path)
    tfm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tfm)
    rs = tfm.rs  # research.py, as loaded by timesfm_lab
    settings = load_settings()
    engine = make_engine(settings)
    return engine, np, pd, rs, settings, tfm


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    The next cell holds every setting. Bars are counted in the section's own bar size: 1,170
    one-minute bars are three sessions, 780 ten-minute bars are twenty.

    1. **The stocks and the window**: the five stocks of the direction study (liquid, very
       different movers), on the TUNE months only. The later months are locked in this folder
       (TEST, HOLDOUT): nothing here is trained, but what you see could still steer a choice.
    2. **Intraday**: the bar size, how many bars the model reads, how many it forecasts, and the
       clock times a forecast is made each day
    3. **Multi-day**: the same, one forecast after each session's close
    4. **The run and the charts**: the budget, the device, the luck-band draws and how much
       history the charts draw before each forecast
    """)
    return


@app.cell
def _():
    # ── 1 · The stocks and the window ─────────────────────────────────
    SYMBOLS = None  # None: the strategy's universe (DirectionFive: MU COIN CSCO AAL ANET)
    WINDOW = ("2025-06-18", "2026-01-31")  # first and last session read (the TUNE months)
    READ_LOCKED = False  # True allows sessions after 2026-01-31 (TEST and HOLDOUT)
    # ── 2 · Intraday ──────────────────────────────────────────────────
    INTRA_BAR = 1  # bar size, minutes
    INTRA_CONTEXT = 1170  # bars read before each forecast (1,170 one-minute bars = 3 sessions)
    INTRA_HORIZON = 60  # bars forecast (60 one-minute bars = the next hour)
    INTRA_CLOCKS = ("10:00", "11:00", "12:00", "13:00", "14:00", "15:00")  # forecasts a day
    # ── 3 · Multi-day ─────────────────────────────────────────────────
    MULTI_BAR = 10  # bar size, minutes (39 bars a session)
    MULTI_CONTEXT = 780  # bars read (780 ten-minute bars = 20 sessions)
    MULTI_HORIZON = 195  # bars forecast (195 = the next 5 sessions), from each session's close
    # ── 4 · The run and the charts ────────────────────────────────────
    MAX_FORECASTS = 5000  # per section and setup; more are thinned evenly over the window
    DEVICE = "mps"  # "mps": Apple GPU, ~35 ms a forecast | "cpu": the same numbers, slower
    FLIPS = 1000  # sign draws behind every luck band
    INTRA_SHOW = 120  # bars drawn before an intraday forecast (the model reads INTRA_CONTEXT)
    MULTI_SHOW = 195  # bars drawn before a multi-day forecast
    # ── 5 · The size test (section 3; fixed before the run, not searched) ─────────
    PROFILE_DAYS = 20  # sessions behind the cheap bands' clock profile (at least half present)
    INTRA_LEVEL = 60  # bars behind "recent", intraday (the last hour, nights left out)
    MULTI_LEVEL = 195  # bars behind "recent", multi-day (the last 5 sessions)
    return (
        DEVICE,
        FLIPS,
        INTRA_BAR,
        INTRA_CLOCKS,
        INTRA_CONTEXT,
        INTRA_HORIZON,
        INTRA_LEVEL,
        INTRA_SHOW,
        MAX_FORECASTS,
        MULTI_BAR,
        MULTI_CONTEXT,
        MULTI_HORIZON,
        MULTI_LEVEL,
        MULTI_SHOW,
        PROFILE_DAYS,
        READ_LOCKED,
        SYMBOLS,
        WINDOW,
    )


@app.cell(hide_code=True)
def _(
    INTRA_BAR,
    INTRA_CLOCKS,
    INTRA_HORIZON,
    MULTI_BAR,
    MULTI_HORIZON,
    READ_LOCKED,
    WINDOW,
    np,
    rs,
    tfm,
):
    _tune_end = rs.WINDOWS["tune"][1]
    if WINDOW[1] > _tune_end and not READ_LOCKED:
        raise ValueError(f"WINDOW ends {WINDOW[1]}, after the TUNE months ({_tune_end}): set "
                         "READ_LOCKED = True to read the locked months")
    for _c in INTRA_CLOCKS:
        _m = int(_c[:2]) * 60 + int(_c[3:]) - tfm.OPEN
        if not (_m > 0 and _m % INTRA_BAR == 0 and _m + INTRA_HORIZON * INTRA_BAR <= 390):
            raise ValueError(f"INTRA_CLOCKS {_c}: after 09:30, at a {INTRA_BAR}-minute bar's "
                             f"close and {INTRA_HORIZON * INTRA_BAR} minutes before 16:00")
    PER_DAY = -(-390 // MULTI_BAR)  # multi-day bars in a full session
    INTRA_LEAD = INTRA_BAR * np.arange(1, INTRA_HORIZON + 1)  # minutes ahead
    MULTI_LEAD = np.arange(1, MULTI_HORIZON + 1) / PER_DAY  # sessions ahead
    MULTI_BLOCK = -(-MULTI_HORIZON // PER_DAY)  # sessions per luck-band coin
    return INTRA_LEAD, MULTI_BLOCK, MULTI_LEAD, PER_DAY


@app.cell(hide_code=True)
def _(DEVICE, mo, tfm):
    def run(B, org, *, context, horizon, label):
        """Both setups' forecasts for these origins: read from disk when saved, else computed
        with a progress bar (the model is loaded for the run and freed after it)."""
        out = {}
        for name in tfm.SETUPS:
            kw = {"context": context, "horizon": horizon, "volume": name != tfm.SETUPS[0]}
            if tfm.cached(B, org, **kw):
                out[name] = tfm.forecast(B, org, **kw)
                continue
            with mo.status.progress_bar(total=-(-len(org) // tfm.CHUNK),
                                        title=f"{label}: {name}",
                                        subtitle=f"{len(org):,} forecasts on {DEVICE}") as bar:
                out[name] = tfm.forecast(B, org, **kw, device=DEVICE, progress=bar.update)
        return out

    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The data

    The stored 1-minute bars of the regular session (09:30-16:00 New York), adjusted for
    dividends (a multi-day move across an ex-date would otherwise count the dividend as a loss).
    Each session is laid on a full clock grid: a minute without a stored bar takes the previous
    close and zero volume; half-days end at 13:00. The sessions are then joined end to end, so
    the model reads ONE series in which the next morning's first bar follows the close: the
    night's gap is a jump between two neighbouring bars. Every bar is stamped at its **close**
    time (the 1-minute bar from 09:59 to 10:00 is "10:00"), so a forecast "made at 10:00" reads
    the closes up to 10:00 and nothing later.
    """)
    return


@app.cell
def _(INTRA_BAR, MULTI_BAR, SYMBOLS, WINDOW, engine, rs, settings, tfm):
    panel = rs.load(settings, engine, start=WINDOW[0], end=WINDOW[1], options=False,
                    symbols=SYMBOLS, adjust=True)
    B_INTRA, B_MULTI = tfm.bars(panel, INTRA_BAR), tfm.bars(panel, MULTI_BAR)
    return B_INTRA, B_MULTI


@app.cell(hide_code=True)
def _(B_INTRA, mo, pd, table):
    _c, _days = B_INTRA["close"], B_INTRA["day"]
    _rows = []
    for _k, _s in enumerate(B_INTRA["symbols"]):
        _rows.append({
            "stock": _s, "sessions": int(_days.max() + 1),
            "first bar": f"{B_INTRA['time'][0]:%Y-%m-%d %H:%M}",
            "last bar": f"{B_INTRA['time'][-1]:%Y-%m-%d %H:%M}",
            "minutes without a stored bar": f"{100 * B_INTRA['filled'][_s]:.2f}%",
            "first close": f"${_c[0, _k]:,.2f}", "last close": f"${_c[-1, _k]:,.2f}",
            "move over the window": f"{100 * (_c[-1, _k] / _c[0, _k] - 1):+.0f}%",
        })
    mo.vstack([table(pd.DataFrame(_rows))])
    return


@app.cell(hide_code=True)
def _(mo, tfm):
    mo.md(f"""
    ## The model

    - **Weights**: `{tfm.HUB}`, snapshot `{tfm.REVISION[:10]}…` (pinned: a new upload never
      changes a result silently). The first run downloads them (~1.3 GB) into the Hugging Face
      cache; later runs read them from there. They are under the **TimesFM Non-Commercial
      License**: fine for research and backtests, not for production trading.
    - **Zero-shot**: `fit()` only builds the network and loads the weights; nothing is trained.
    - **One pass**: each section has its own model, reading exactly the context and forecasting
      exactly the horizon in one pass, as Google's own code does (a longer output or a variable
      input length moves the paths by a few bp).
    - **What it reads**: the closes (price only), or the closes plus `log(1 + volume)` as a
      second series (price + volume). It scales every series itself, and if the context is
      close to a straight line (R² above about 0.75) it continues that line into the forecast:
      on a smooth trend it is a trend follower by design.
    - **Output**: the 10 %, 50 % and 90 % quantile of every future bar's close.
    - **Speed and cache**: about 35 ms a forecast on the Apple GPU (~3 minutes per section and
      setup at the defaults). Results are saved under `var/timesfm_lab/`, keyed by every
      setting and the exact numbers read; a rerun loads them in a second. Apple GPU and CPU
      agree to about 0.00002 dollars a share; the run is deterministic.
    """)
    return


@app.cell(hide_code=True)
def _(INTRA_CLOCKS, INTRA_CONTEXT, INTRA_HORIZON, mo):
    mo.md(f"""
    ## 1 · Intraday: 1-minute bars, the next {INTRA_HORIZON} minutes

    Every session, at {", ".join(INTRA_CLOCKS)}, the model reads the last **{INTRA_CONTEXT:,}
    bars** (about {INTRA_CONTEXT / 390:.1f} sessions: the earlier days' minutes, then today's so
    far) and forecasts the next **{INTRA_HORIZON}**. No forecast runs past the close: a half-day
    keeps only the forecasts that end by 13:00.
    """)
    return


@app.cell
def _(B_INTRA, INTRA_CLOCKS, INTRA_CONTEXT, INTRA_HORIZON, MAX_FORECASTS, run, tfm):
    O_INTRA = tfm.origins(B_INTRA, context=INTRA_CONTEXT, horizon=INTRA_HORIZON,
                          clocks=INTRA_CLOCKS, limit=MAX_FORECASTS)
    Q_INTRA = run(B_INTRA, O_INTRA, context=INTRA_CONTEXT, horizon=INTRA_HORIZON,
                  label="Intraday")
    return O_INTRA, Q_INTRA


@app.cell
def _(B_INTRA, FLIPS, O_INTRA, Q_INTRA, tfm):
    S_INTRA = {_k: tfm.score(B_INTRA, O_INTRA, _q, block=1, flips=FLIPS)
               for _k, _q in Q_INTRA.items()}
    return (S_INTRA,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### One forecast up close

    Pick a stock, a day and a time. **Figure 1** draws the actual price (black) before and after
    the forecast, the median path of each setup (blue: price only; green, dashed: price +
    volume), the price-only 10-90 % band (shaded) and the "no change" line (dotted). Below:
    the volume, dark up to the forecast, light after it (the model never sees the light part).
    The title gives this forecast's path error as a multiple of "no change" (below 1: better).
    """)
    return


@app.cell
def _(B_INTRA, INTRA_CLOCKS, O_INTRA, mo):
    _days = sorted(O_INTRA["time"].dt.strftime("%Y-%m-%d").unique())
    intra_stock = mo.ui.dropdown(options=B_INTRA["symbols"], value=B_INTRA["symbols"][0],
                                 label="stock")
    intra_day = mo.ui.dropdown(options=_days, value=_days[len(_days) // 2], label="day")
    intra_clock = mo.ui.dropdown(options=list(INTRA_CLOCKS),
                                 value=INTRA_CLOCKS[len(INTRA_CLOCKS) // 2],
                                 label="forecast made at")
    mo.hstack([intra_stock, intra_day, intra_clock], justify="start")
    return intra_clock, intra_day, intra_stock


@app.cell(hide_code=True)
def _(
    B_INTRA,
    INTRA_HORIZON,
    INTRA_SHOW,
    O_INTRA,
    Q_INTRA,
    S_INTRA,
    intra_clock,
    intra_day,
    intra_stock,
    mo,
    np,
    tfm,
):
    _m = ((O_INTRA["symbol"] == intra_stock.value)
          & (O_INTRA["time"].dt.strftime("%Y-%m-%d") == intra_day.value)
          & (O_INTRA["time"].dt.strftime("%H:%M") == intra_clock.value)).to_numpy()
    if not _m.any():
        _out = mo.md("No forecast for this choice: a half-day, or thinned by `MAX_FORECASTS`.")
    else:
        _i = int(np.flatnonzero(_m)[0])
        _r = S_INTRA[tfm.SETUPS[0]]["ratio"][_i]
        _out = tfm.forecast_figure(
            B_INTRA, O_INTRA, Q_INTRA, _i, before=INTRA_SHOW,
            title=f"Figure 1 · {intra_stock.value}: the next {INTRA_HORIZON} minutes from "
                  f"{O_INTRA['time'].iat[_i]:%a %d %b %Y, %H:%M} · path error {_r:.2f}× "
                  f"no change (price only)")
    mo.vstack([_out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### The scorecard

    Every forecast of the section, graded. **Path error saved**: the share of the "no change"
    path error the median saves (0 % = no better, negative = worse). **Direction right**: the
    share of forecasts whose median ends on the right side of the starting price.
    **End-move corr**: how well the size and sign of the predicted end move line up with the
    actual one. **Inside 10-90 % band**: 80 % if the band is honest; below the 10 % line and
    above the 90 % line should each be 10 %. **Says up / went up** shows a lean: in a rising
    window a model that always says "up" is right more than half the time without knowing
    anything, and the luck band (signs flipped) does not catch that.
    """)
    return


@app.cell(hide_code=True)
def _(B_INTRA, O_INTRA, Q_INTRA, S_INTRA, mo, table, tfm):
    mo.vstack([
        mo.md(tfm.verdict(S_INTRA)),
        table(tfm.summary(S_INTRA)),
        mo.md("**Per stock** (each against its own luck band: wider, a few hundred forecasts "
              "each)"),
        table(tfm.by_stock(B_INTRA, O_INTRA, Q_INTRA, block=1)),
    ])
    return


@app.cell(hide_code=True)
def _(B_INTRA, INTRA_SHOW, O_INTRA, Q_INTRA, S_INTRA, tfm):
    _sc = S_INTRA[tfm.SETUPS[0]]
    tfm.gallery(B_INTRA, O_INTRA, Q_INTRA, tfm.picks(_sc), _sc, before=INTRA_SHOW,
                title="Figure 2 · Six forecasts from best to worst (2.5th, 20th, 40th, 60th, "
                      "80th, 97.5th percentile of the path error, price only)")
    return


@app.cell(hide_code=True)
def _(INTRA_LEAD, S_INTRA, tfm):
    tfm.skill_figure(S_INTRA, INTRA_LEAD, xlabel="minutes ahead",
                     title="Figure 3 · Path error saved against \"no change\", by lead time "
                           "(shaded: the luck band)")
    return


@app.cell(hide_code=True)
def _(INTRA_LEAD, S_INTRA, tfm):
    tfm.coverage_figure(S_INTRA, INTRA_LEAD, xlabel="minutes ahead",
                        title="Figure 4 · Is the 10-90% band honest? By lead time")
    return


@app.cell(hide_code=True)
def _(B_INTRA, INTRA_HORIZON, O_INTRA, S_INTRA, tfm):
    tfm.end_scatter(S_INTRA, O_INTRA, B_INTRA["symbols"],
                    title=f"Figure 5 · Predicted against actual move over the {INTRA_HORIZON} "
                          "minutes (bp; the axes differ in scale)")
    return


@app.cell(hide_code=True)
def _(MULTI_BAR, MULTI_CONTEXT, MULTI_HORIZON, PER_DAY, mo):
    mo.md(f"""
    ## 2 · Multi-day: {MULTI_BAR}-minute bars, the next {MULTI_HORIZON / PER_DAY:g} sessions

    After every session's close, the model reads the last **{MULTI_CONTEXT:,} bars** (about
    {MULTI_CONTEXT / PER_DAY:.0f} sessions) and forecasts the next **{MULTI_HORIZON}** (about
    {MULTI_HORIZON / PER_DAY:g} sessions, {PER_DAY} bars each). The first forecast bar is the
    next morning's first {MULTI_BAR} minutes, so the overnight gap is part of what it forecasts.
    Forecasts made on neighbouring days share most of their future, so the luck band flips one
    coin per {-(-MULTI_HORIZON // PER_DAY)} sessions.
    """)
    return


@app.cell
def _(B_MULTI, MAX_FORECASTS, MULTI_CONTEXT, MULTI_HORIZON, run, tfm):
    O_MULTI = tfm.origins(B_MULTI, context=MULTI_CONTEXT, horizon=MULTI_HORIZON,
                          limit=MAX_FORECASTS)
    Q_MULTI = run(B_MULTI, O_MULTI, context=MULTI_CONTEXT, horizon=MULTI_HORIZON,
                  label="Multi-day")
    return O_MULTI, Q_MULTI


@app.cell
def _(B_MULTI, FLIPS, MULTI_BLOCK, O_MULTI, Q_MULTI, tfm):
    S_MULTI = {_k: tfm.score(B_MULTI, O_MULTI, _q, block=MULTI_BLOCK, flips=FLIPS)
               for _k, _q in Q_MULTI.items()}
    return (S_MULTI,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### One forecast up close

    The same picture as Figure 1, made at the close of the chosen day; the nights and weekends
    are cut out of the time axis.
    """)
    return


@app.cell
def _(B_MULTI, O_MULTI, mo):
    _days = sorted(O_MULTI["time"].dt.strftime("%Y-%m-%d").unique())
    multi_stock = mo.ui.dropdown(options=B_MULTI["symbols"], value=B_MULTI["symbols"][0],
                                 label="stock")
    multi_day = mo.ui.dropdown(options=_days, value=_days[len(_days) // 2],
                               label="forecast made at the close of")
    mo.hstack([multi_stock, multi_day], justify="start")
    return multi_day, multi_stock


@app.cell(hide_code=True)
def _(
    B_MULTI,
    MULTI_HORIZON,
    MULTI_SHOW,
    O_MULTI,
    PER_DAY,
    Q_MULTI,
    S_MULTI,
    mo,
    multi_day,
    multi_stock,
    np,
    tfm,
):
    _m = ((O_MULTI["symbol"] == multi_stock.value)
          & (O_MULTI["time"].dt.strftime("%Y-%m-%d") == multi_day.value)).to_numpy()
    if not _m.any():
        _out = mo.md("No forecast for this choice (thinned by `MAX_FORECASTS`).")
    else:
        _i = int(np.flatnonzero(_m)[0])
        _r = S_MULTI[tfm.SETUPS[0]]["ratio"][_i]
        _out = tfm.forecast_figure(
            B_MULTI, O_MULTI, Q_MULTI, _i, before=MULTI_SHOW,
            title=f"Figure 6 · {multi_stock.value}: the next {MULTI_HORIZON / PER_DAY:g} "
                  f"sessions from the close of {O_MULTI['time'].iat[_i]:%a %d %b %Y} · path "
                  f"error {_r:.2f}× no change (price only)")
    mo.vstack([_out])
    return


@app.cell(hide_code=True)
def _(B_MULTI, MULTI_BLOCK, O_MULTI, Q_MULTI, S_MULTI, mo, table, tfm):
    mo.vstack([
        mo.md("### The scorecard\n\nThe same columns as in section 1.\n\n"
              + tfm.verdict(S_MULTI)),
        table(tfm.summary(S_MULTI)),
        mo.md("**Per stock** (each against its own luck band: about a hundred forecasts each, "
              "most of them overlapping, so the bands are wide)"),
        table(tfm.by_stock(B_MULTI, O_MULTI, Q_MULTI, block=MULTI_BLOCK)),
    ])
    return


@app.cell(hide_code=True)
def _(B_MULTI, MULTI_SHOW, O_MULTI, Q_MULTI, S_MULTI, tfm):
    _sc = S_MULTI[tfm.SETUPS[0]]
    tfm.gallery(B_MULTI, O_MULTI, Q_MULTI, tfm.picks(_sc), _sc, before=MULTI_SHOW,
                title="Figure 7 · Six forecasts from best to worst (2.5th, 20th, 40th, 60th, "
                      "80th, 97.5th percentile of the path error, price only)")
    return


@app.cell(hide_code=True)
def _(MULTI_LEAD, S_MULTI, tfm):
    tfm.skill_figure(S_MULTI, MULTI_LEAD, xlabel="sessions ahead",
                     title="Figure 8 · Path error saved against \"no change\", by lead time "
                           "(shaded: the luck band)")
    return


@app.cell(hide_code=True)
def _(MULTI_LEAD, S_MULTI, tfm):
    tfm.coverage_figure(S_MULTI, MULTI_LEAD, xlabel="sessions ahead",
                        title="Figure 9 · Is the 10-90% band honest? By lead time")
    return


@app.cell(hide_code=True)
def _(B_MULTI, MULTI_HORIZON, O_MULTI, PER_DAY, S_MULTI, tfm):
    tfm.end_scatter(S_MULTI, O_MULTI, B_MULTI["symbols"],
                    title=f"Figure 10 · Predicted against actual move over the "
                          f"{MULTI_HORIZON / PER_DAY:g} sessions (bp; the axes differ in scale)")
    return


@app.cell(hide_code=True)
def _(INTRA_LEVEL, MULTI_BLOCK, MULTI_LEVEL, PROFILE_DAYS, mo, tfm):
    mo.md(f"""
    ## 3 · Does TimesFM know how BIG the move will be?

    Sections 1 and 2 found no direction, but the intraday band was honest (about 80 % of the
    closes inside). A band is only worth a 330-million-parameter model if it beats a cheap one,
    so here it meets two volatility rules of a few lines each, both Gaussian in log price and
    centred on "no change", with every number known when the forecast is made:

    - **clock profile**: each future bar's variance is the average squared return of the same bar
      of the session (the same clock time; a session's first bar carries the night) over the
      {PROFILE_DAYS} sessions before the forecast's own;
    - **{tfm.REFERENCE}**: that profile scaled by how busy the stock is now: the realized
      variance of the last {INTRA_LEVEL} bars (multi-day: {MULTI_LEVEL}) over what the profile
      expected for them, nights left out.

    TimesFM is scored as is and with its band **centred** on "no change" (each quantile divided
    by its median): that keeps what it knows about the SIZE of the move and drops the wandering
    median. The score is the **band loss**, the pinball loss of the 10 % plus the 90 % quantile
    in bp: lower is better; it rewards a band that is narrow where the move is small and wide
    where it is large, and charges every close outside it.

    **Decided before the run**: TimesFM's centred band (price only) is a better size forecaster
    if it beats *{tfm.REFERENCE}* intraday with z above 2 (the paired difference over its
    standard error, one cluster per session shared by every stock; multi-day one per
    {MULTI_BLOCK} sessions). The cheap bands' settings were fixed before the run and not searched.

    Columns: **saved vs {tfm.REFERENCE}** (above 0: lower band loss than it) · **inside
    10-90 %** (80 % if honest) · **inside at the 1st bar** (the next minute; multi-day the first
    bar after the night) · **width ↔ |move|** (rank correlation of the predicted size with the
    realized size at the horizon's end: does the band tell big moves from small ones?).
    """)
    return


@app.cell
def _(
    B_INTRA,
    B_MULTI,
    INTRA_HORIZON,
    INTRA_LEVEL,
    MULTI_BLOCK,
    MULTI_HORIZON,
    MULTI_LEVEL,
    O_INTRA,
    O_MULTI,
    PROFILE_DAYS,
    Q_INTRA,
    Q_MULTI,
    tfm,
):
    SIZE_INTRA = tfm.size_compare(
        B_INTRA, O_INTRA, Q_INTRA,
        tfm.vol_bands(B_INTRA, O_INTRA, INTRA_HORIZON, level=INTRA_LEVEL, days=PROFILE_DAYS),
        block=1)
    SIZE_MULTI = tfm.size_compare(
        B_MULTI, O_MULTI, Q_MULTI,
        tfm.vol_bands(B_MULTI, O_MULTI, MULTI_HORIZON, level=MULTI_LEVEL, days=PROFILE_DAYS),
        block=MULTI_BLOCK)
    return SIZE_INTRA, SIZE_MULTI


@app.cell(hide_code=True)
def _(SIZE_INTRA, mo, table, tfm):
    mo.vstack([
        mo.md("### Intraday: the next hour\n\n" + tfm.size_verdict(SIZE_INTRA)),
        table(SIZE_INTRA["table"]),
    ])
    return


@app.cell(hide_code=True)
def _(INTRA_LEAD, SIZE_INTRA, tfm):
    tfm.size_figure(SIZE_INTRA, INTRA_LEAD, xlabel="minutes ahead",
                    title="Figure 11 · Intraday: band loss saved and band coverage, by lead time")
    return


@app.cell(hide_code=True)
def _(SIZE_INTRA, tfm):
    tfm.deciles_figure(SIZE_INTRA, title="Figure 12 · Intraday: predicted against realized size "
                                         "of the hour's move, by tenth of the prediction")
    return


@app.cell(hide_code=True)
def _(SIZE_MULTI, mo, table, tfm):
    mo.vstack([
        mo.md("### Multi-day: the next sessions\n\n" + tfm.size_verdict(SIZE_MULTI)),
        table(SIZE_MULTI["table"]),
    ])
    return


@app.cell(hide_code=True)
def _(MULTI_LEAD, SIZE_MULTI, tfm):
    tfm.size_figure(SIZE_MULTI, MULTI_LEAD, xlabel="sessions ahead",
                    title="Figure 13 · Multi-day: band loss saved and band coverage, by lead time")
    return


@app.cell(hide_code=True)
def _(SIZE_MULTI, tfm):
    tfm.deciles_figure(SIZE_MULTI, title="Figure 14 · Multi-day: predicted against realized size "
                                         "of the end move, by tenth of the prediction")
    return


@app.cell(hide_code=True)
def _(SIZE_INTRA, SIZE_MULTI, mo, table, tfm):
    def _rows(cmp):
        s = cmp["stocks"]
        return s[s["band"].isin([tfm.PRIMARY, "clock profile"])]

    mo.vstack([
        mo.md(f"### Per stock\n\nTimesFM's centred band and the clock profile, each against "
              f"*{tfm.REFERENCE}*, intraday then multi-day."),
        table(_rows(SIZE_INTRA)),
        table(_rows(SIZE_MULTI)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reading the results

    - **Beating "no change" is the bar.** A median path that wanders while the price wanders
      the other way costs more than the flat line. A positive path error saved, above its luck
      band, is the one sign the model knows where the price goes.
    - **Direction and lean are different things.** Compare "says up" with "went up": a model
      that continues the trend of a rising window says "up" more often and gets paid for the
      window, not for skill. The luck band keeps every move and flips its sign, so it removes
      the lean along with the direction.
    - **A good band is still useful without direction, if nothing cheaper does it.** Closes
      inside the 10-90 % band about 80 % of the time mean the model knows the SIZE of the coming
      moves (the earlier studies in this folder found size predictable and direction not), which
      is what risk limits, stops and position sizes need. Section 3 asks whether TimesFM knows it
      better than two lines of realized volatility that know the clock: the lunch lull and the
      night, which a series with the nights cut out never tells the model.
    - **The settings are choices.** Nothing is trained, but trying many contexts, horizons or
      clock times and keeping the best one is tuning: a best-of-many result needs a higher bar
      than one luck band, and the locked months as the final check.
    """)
    return


if __name__ == "__main__":
    app.run()
