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
    # overnight_open · does the night predict the first hour?

    The question, before any trading: **is there a direct connection between what happens
    after the close, overnight, in the pre-market and in the news, and a strong first trading
    hour the next day?** And does that first hour follow the night's direction, or fade it?
    `overnight_open_explore_marimo.py` shows single nights, minute by minute.

    **Data.**
    - ~530 stocks with extended-hours 1-minute bars, 2025-06-18 → 2026-06-26. After that the
      database keeps extended hours for only 7 names.
    - The stocks were chosen as volatile, heavily traded names, with hindsight (the HighOctane
      list). So every number reads as "for volatile US stocks", and the market's own rise is
      always taken out.
    - SEC 8-K filings, timed to the second.
    - Benzinga headlines from 2025-10-14, dated but not timed.

    **What was found before.** burst_scan's `move_extended.ipynb` (16 names, 2026-09-29):
    - The pre-market price IS the gap (r 0.999).
    - Extended hours barely improved the prediction of a big move near the open
      (AUC 0.687 → 0.700) and not its direction at all (0.544 → 0.539).
    - Evening moves clustered on news nights: "an earnings calendar is the missing input".

    This notebook scales that to ~530 names and adds the news.

    **Two windows.** *Discovery*, July 2025 → February 2026: explore it freely; §1-§5 read only
    this window. *Validation*, March → June 2026: read ONCE in §6, with four questions whose pass
    bars were written into the README before anything was run.

    1. **The night table** and how much of it each source covers
    2. **A strong first hour**: twice the stock's usual first-hour range
    3. **Size**: which signals go with a strong first hour, and does each part of the night add
       to what the previous days already tell?
    4. **Direction**: does the first hour continue the night's move or fade it?
    5. **Flags**: unusual nights, compared with nights that moved just as much
    6. **Validation**: the four pre-registered questions on the months nobody looked at
    """)
    return


@app.cell
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload research.py (after it changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(mo, reload_helpers):
    import importlib.util
    import json

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

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
    GROUP_COLOR = dict(zip(rs.GROUPS, (SERIES[0], SERIES[2], SERIES[4], SERIES[6]), strict=True))

    def finish(fig, title, height=460):
        """The repo's look (``plots._base``) with the legend under the plot."""
        base_layout(fig, title=title, height=height)
        fig.update_layout(margin={"b": 120},
                          legend={"orientation": "h", "x": 0.0, "xanchor": "left",
                                  "y": -60 / max(height - 170, 80), "yanchor": "top"})
        return fig

    def show(df, rows=20, digits=3):
        """A frame as a marimo table, numbers rounded."""
        return mo.ui.table(df.round(digits), selection=None, page_size=rows)

    def ic_figure(ics, title):
        """Rank correlations as bars, ±2 standard errors, one color per group of signals."""
        fig = go.Figure()
        for grp in rs.GROUPS:
            d = ics[ics["group"] == grp]
            fig.add_trace(go.Bar(
                y=d["signal"], x=d["ic"], orientation="h", name=grp,
                marker_color=GROUP_COLOR[grp],
                error_x={"type": "data", "array": 2 * (d["ic"] / d["t"]).abs(),
                         "color": INK, "thickness": 1},
                hovertext=[f"{w}: IC {i:+.3f}, t {t:+.1f}, {n:,} nights"
                           for w, i, t, n in zip(d["what"], d["ic"], d["t"], d["n"],
                                                 strict=True)], hoverinfo="text"))
        fig.add_vline(x=0, line={"color": MUTED, "width": 1})
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title_text="rank correlation (IC), bars = ±2 standard errors "
                                    "clustered by day")
        return finish(fig, title, height=620)

    def ladder_figure(lad, title):
        """Out-of-sample AUC per ladder step with its 95% day-bootstrap interval."""
        fig = go.Figure(go.Bar(
            x=lad["step"], y=lad["auc"], marker_color=SERIES[0], name="out-of-sample AUC",
            error_y={"type": "data", "array": lad["hi"] - lad["auc"],
                     "arrayminus": lad["auc"] - lad["lo"], "color": INK, "thickness": 1},
            text=[f"{a:.3f}<br>gain {g:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  for a, g, lo, hi in zip(lad["auc"], lad["gain"], lad["gain_lo"],
                                          lad["gain_hi"], strict=True)],
            textposition="inside", insidetextanchor="start", textfont={"color": "white"}))
        fig.add_hline(y=0.5, line={"color": MUTED, "dash": "dash"},
                      annotation_text="0.5 = a coin flip", annotation_position="bottom left")
        fig.update_yaxes(range=[0.45, max(0.75, float(lad["hi"].max()) + 0.05)],
                         title_text="AUC")
        return finish(fig, title, height=460)

    return (GROUP_COLOR, INK, MUTED, SERIES, engine, finish, go, ic_figure, json,
            ladder_figure, np, pd, rs, show)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The night table

    One row per stock and session: every signal known before 09:30 (`research.FEATURES`, in
    four groups: previous days, the night's price move, the night's trading activity, the news)
    and what the first hour did. Moves are in daily ATRs (the stock's typical daily range), so a
    1-ATR night is equally big for a quiet and a wild stock; volumes are against the stock's own
    mean of the 20 sessions before. The first build takes about half an hour (cached after).
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
    disc = table[table["split"] == "discovery"].copy()
    mo.md(f"**{len(table):,} nights** of **{table['symbol'].nunique()} stocks**: "
          f"{len(disc):,} in discovery ({disc['day'].min().date()} → "
          f"{disc['day'].max().date()}), {len(table) - len(disc):,} in validation (locked "
          "until §6).")
    return disc, filings, table


@app.cell
def _(disc, mo, pd, show):
    _g = disc.groupby("month")
    _cov = pd.DataFrame({
        "stocks": _g["symbol"].nunique(),
        "nights": _g.size(),
        "after-hours print": _g["ah_ret_atr"].apply(lambda s: s.notna().mean()),
        "pre-market print": _g["pm_ret_atr"].apply(lambda s: s.notna().mean()),
        "8-K filer": _g["sec_night"].apply(lambda s: s.notna().mean()),
        "8-K at night": _g["sec_night"].apply(lambda s: (s > 0).mean()),
        "earnings at night": _g["earnings_night"].apply(lambda s: (s > 0).mean()),
        "headlines covered": _g["news_prev"].apply(lambda s: s.notna().mean()),
    }).reset_index()
    mo.vstack([
        mo.md("**Table 1 · Coverage per month** (share of nights). A stock without an "
              "after-hours or pre-market print that night has no value for those signals; a "
              "stock that never files 8-Ks (a foreign issuer, a fund) has none for the 8-K "
              "columns; headlines start on 2025-10-14."),
        show(_cov, rows=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · A strong first hour

    The first hour's high-low range (the official open included) divided by the stock's
    **usual** first-hour range: its median over the 20 sessions before. 1 = an ordinary first
    hour, 2 = twice as wide. A *strong* first hour reaches the threshold below. The
    pre-registered tests (§6) use 2, whatever the slider says.
    """)
    return


@app.cell
def _(mo):
    strong_x = mo.ui.slider(1.25, 4.0, step=0.25, value=2.0, show_value=True,
                            label="strong first hour: at least this many times its usual range")
    mo.hstack([strong_x], justify="start")
    return (strong_x,)


@app.cell
def _(disc, finish, go, mo, np, strong_x, SERIES):
    nights = disc.assign(strong=(disc["fh_range_rel"] >= strong_x.value).astype(float),
                         up_x=(disc["fh_ret_x"] > 0).astype(float).where(disc["fh_ret_x"].notna()))
    _fig = go.Figure(go.Histogram(x=np.clip(disc["fh_range_rel"], 0, 6), nbinsx=60,
                                  marker_color=SERIES[0], name="discovery nights"))
    _fig.add_vline(x=strong_x.value, line={"color": SERIES[5], "width": 2},
                   annotation_text=f"strong: ≥ {strong_x.value:g}×")
    _fig.update_xaxes(title_text="first-hour range / the stock's usual first-hour range "
                                 "(clipped at 6)")
    _fig.update_yaxes(title_text="nights")
    finish(_fig, "How wide first hours are, relative to each stock's normal", height=400)
    mo.vstack([_fig, mo.md(f"**{nights['strong'].mean():.1%}** of discovery nights are strong at "
                           f"{strong_x.value:g}× ({int(nights['strong'].sum()):,} nights).")])
    return (nights,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Size: what goes with a strong first hour?

    **3a · Each signal alone.** The rank correlation (IC) of every signal with the first hour's
    range relative to normal: +0.10 means nights ranked higher on the signal tend to have wider
    first hours. Moves enter as their size (a 2-ATR drop counts like a 2-ATR rise). Errors are
    clustered by day, because all stocks share one morning's market.

    One correlation is partly built in: `fh_norm`, the usual first-hour range, is the outcome's
    denominator. When it happens to read low, the same first hour counts as wider, so it leans
    negative by construction.
    """)
    return


@app.cell
def _(ic_figure, mo, nights, rs, show):
    size_ic = rs.ic_table(nights, "fh_range_rel", size=True)
    mo.vstack([ic_figure(size_ic, "Which signals go with a wide first hour (discovery)"),
               show(size_ic, rows=20)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **3b · Dose-response.** Nights sorted into ten equal groups by one signal: is the share of
    strong first hours higher where the signal is higher, and smoothly so? A real connection
    rises step by step; one group sticking out is more likely noise. Bars are ±2 standard
    errors clustered by day; the dashed line is the overall share.
    """)
    return


@app.cell
def _(mo, rs):
    signal_pick = mo.ui.dropdown(options=list(rs.FEATURES), value="pm_vol_rel",
                                 searchable=True, label="signal")
    dose_y = mo.ui.radio(options={"share of strong first hours": "strong",
                                  "first-hour range / usual (mean)": "fh_range_rel"},
                         value="share of strong first hours", inline=True, label="outcome")
    mo.hstack([signal_pick, dose_y], justify="start", gap=2)
    return dose_y, signal_pick


@app.cell
def _(INK, MUTED, SERIES, dose_y, finish, go, mo, nights, rs, signal_pick):
    _c = signal_pick.value
    _size = _c in rs.MOVES
    _ylabel = {"strong": "share of strong first hours",
               "fh_range_rel": "first-hour range / usual (mean)"}[dose_y.value]
    _d = rs.deciles(nights.assign(x=nights[_c].abs() if _size else nights[_c]), "x", dose_y.value)
    _fig = go.Figure(go.Scatter(
        x=_d["bin"], y=_d["y"], mode="lines+markers", line={"color": SERIES[0]},
        error_y={"type": "data", "array": 2 * _d["se"], "color": INK, "thickness": 1},
        name=_ylabel,
        hovertext=[f"tenth {b}: median {'|' + _c + '|' if _size else _c} = {x:.3g}, "
                   f"{y:.3f}, {n:,} nights"
                   for b, x, y, n in zip(_d["bin"], _d["x"], _d["y"], _d["n"], strict=True)],
        hoverinfo="text"))
    _fig.add_hline(y=nights[dose_y.value].mean(), line={"color": MUTED, "dash": "dash"})
    _fig.update_xaxes(title_text=f"tenth of {'|' + _c + '|' if _size else _c} "
                                 "(1 = lowest, 10 = highest)", dtick=1)
    _fig.update_yaxes(title_text=_ylabel)
    finish(_fig, f"{rs.FEATURES[_c][1]}", height=420)
    mo.vstack([_fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **3c · The ladder: does each part of the night add?** Four models, each with one more
    group of signals: the previous days alone, then the night's price move, then its trading
    activity, then the news. Every model is walked forward: each month is predicted by a model
    fit only on the months before it (a small gradient-boosted tree, fixed settings). The score
    is the **AUC**, the chance that a strong night is ranked above an ordinary one: 0.5 = a coin
    flip, 1 = perfect. A step's **gain** is its AUC minus the previous step's, with a 95%
    interval from resampling whole days. Scored on November 2025 → February 2026 (the headlines
    need a month to build their baseline); trained on everything before each month.

    This is the core answer to the question: if the night's move and activity add a clear gain
    over the previous days, the night *is* connected to the first hour's size. (Part of the
    previous days' own score is the built-in `fh_norm` effect of 3a, one more reason to read the
    gains, not the first bar.)
    """)
    return


@app.cell
def _(mo):
    ladder_button = mo.ui.run_button(label="run the size ladder (about a minute)")
    mo.hstack([ladder_button], justify="start")
    return (ladder_button,)


@app.cell
def _(disc, ladder_button, ladder_figure, mo, rs, show, strong_x):
    mo.stop(not ladder_button.value, mo.md("_Click the button to run the ladder._"))
    with mo.status.spinner("walking four models forward month by month ..."):
        size_ladder, _ = rs.ladder(
            disc.assign(strong=(disc["fh_range_rel"] >= strong_x.value).astype(float)),
            "strong", first_month=rs.SCORE_FROM, last_month="2026-02")
    mo.vstack([ladder_figure(size_ladder, f"Predicting a strong first hour "
                                          f"(≥ {strong_x.value:g}× usual), out of sample"),
               show(size_ladder)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Direction: does the first hour follow the night?

    **4a · Each signal alone**, now with its sign: the IC of every signal with the first hour's
    return **beyond the average stock that morning** (the market taken out). A positive IC for
    the pre-market move means the first hour tends to continue the night; negative, to fade it.
    """)
    return


@app.cell
def _(ic_figure, mo, nights, rs, show):
    dir_ic = rs.ic_table(nights, "fh_ret_x", size=False)
    mo.vstack([ic_figure(dir_ic, "Which signals go with the first hour's direction (discovery)"),
               show(dir_ic, rows=20)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **4b · Continue or fade, by kind of night.** On nights whose pre-market moved at least
    0.5 daily ATRs from the close: the first hour's return beyond the average stock, signed by
    the night's direction, in basis points (1 bp = 0.01%). Positive = it went on in the night's
    direction. For scale: a round trip in the first half hour costs roughly 20-40 bp for a
    $10k order (fade_movers' cost table), so a few bp is a connection, not a trade.
    """)
    return


@app.cell
def _(INK, SERIES, finish, go, mo, nights, pd, rs, show):
    _f = rs.flags(nights)
    _m = nights["pm_ret_atr"].abs()
    _news = rs.news_night(nights)
    _groups = {
        "all moved nights": None,
        "night move 0.5-1 ATR": _m < 1,
        "night move 1-2 ATR": (_m >= 1) & (_m < 2),
        "night move 2+ ATR": _m >= 2,
        "news night": _news,
        "quiet night": ~_news,
        "earnings at night": _f["earnings at night"],
        "busy pre-market (≥ 3× volume)": _f["pre-market volume"],
        "quiet pre-market": ~_f["pre-market volume"],
    }
    _rows = pd.DataFrame([{"nights": k, **rs.continuation(nights, v)}
                          for k, v in _groups.items()])
    _fig = go.Figure(go.Bar(
        x=_rows["nights"], y=_rows["bp"], marker_color=SERIES[0], name="continuation, bp",
        error_y={"type": "data", "array": 2 * _rows["se"], "color": INK, "thickness": 1},
        hovertext=[f"{n:,} nights on {d} days, t {t:+.1f}, {s:.0%} continued"
                   for n, d, t, s in zip(_rows["n"], _rows["days"], _rows["t"], _rows["share"],
                                         strict=True)], hoverinfo="text"))
    _fig.add_hline(y=0, line={"color": "#b9b7b0"})
    _fig.update_yaxes(title_text="first hour beyond the market, signed by the night (bp)")
    finish(_fig, "Does the first hour continue the night's move? (± 2 standard errors)",
           height=440)
    mo.vstack([_fig, show(_rows.rename(columns={"share": "share continued"}), digits=2)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **4c · The direction ladder.** The same four walked-forward models as 3c, now predicting
    whether a stock's first hour beats the average stock that morning. 0.5 = no better than a
    coin flip. The earlier 16-name study found 0.54 at most, so expect little here.
    """)
    return


@app.cell
def _(mo):
    dir_button = mo.ui.run_button(label="run the direction ladder (about a minute)")
    mo.hstack([dir_button], justify="start")
    return (dir_button,)


@app.cell
def _(dir_button, disc, ladder_figure, mo, rs, show):
    mo.stop(not dir_button.value, mo.md("_Click the button to run the ladder._"))
    with mo.status.spinner("walking four models forward month by month ..."):
        dir_ladder, _ = rs.ladder(
            disc.assign(up_x=(disc["fh_ret_x"] > 0).astype(float)
                        .where(disc["fh_ret_x"].notna())),
            "up_x", first_month=rs.SCORE_FROM, last_month="2026-02")
    mo.vstack([ladder_figure(dir_ladder, "Predicting a first hour above the market, "
                                         "out of sample"), show(dir_ladder)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Flags: unusual nights

    The flags of the explorer notebook, over all stocks. For each flag:
    - **strong**: the share of its nights with a strong first hour (§2's threshold);
    - **strong if moved as much**: the share among nights WITHOUT the flag whose pre-market
      moved just as far (same fifth of the move's size; no pre-market print is its own group);
    - **lift**: the difference, with its t clustered by day. A big lift says the flag tells
      something the move alone does not;
    - **continuation**: as in 4b, on the flag's moved nights.

    The co-occurrence table shows how often one flag comes with another (row = given this flag,
    column = share that also has that one): earnings nights are usually busy, moving nights too.
    """)
    return


@app.cell
def _(mo):
    move_atr = mo.ui.slider(0.25, 3.0, step=0.25, value=1.0, show_value=True,
                            label="move flag: at least this many daily ATRs")
    vol_x = mo.ui.slider(1.5, 10.0, step=0.5, value=3.0, show_value=True,
                         label="volume flag: at least this many times its normal")
    news_x = mo.ui.slider(1.5, 10.0, step=0.5, value=3.0, show_value=True,
                          label="headline flag: at least this many times the usual count")
    mo.hstack([move_atr, vol_x, news_x], justify="start", gap=2)
    return move_atr, news_x, vol_x


@app.cell
def _(mo, move_atr, news_x, nights, pd, rs, show, vol_x):
    fl = rs.flags(nights, move_atr=move_atr.value, vol_x=vol_x.value, news_x=news_x.value)
    _tab = rs.flag_table(nights, fl, "strong")
    _ints = fl.astype(int)
    _co = (_ints.T @ _ints).div(_ints.sum(), axis=0)
    _co.index.name = "given"
    _same = nights["news_same"] >= 3
    _hind = pd.DataFrame({
        "nights": ["≥ 3 headlines dated the SAME day (hindsight)", "fewer"],
        "count": [int(_same.sum()), int((~_same & nights["news_same"].notna()).sum())],
        "strong": [nights.loc[_same, "strong"].mean(),
                   nights.loc[~_same & nights["news_same"].notna(), "strong"].mean()]})
    mo.vstack([
        mo.md("**Table 5a · The flags** (discovery)"), show(_tab, rows=12),
        mo.md("**Table 5b · Co-occurrence**: share of the row flag's nights that also carry "
              "the column flag"), show(_co.reset_index(), rows=10, digits=2),
        mo.md("**Table 5c · Hindsight only**: headlines dated on the session day itself, which "
              "may have come out after the open, so never a signal. It shows how much a TIMED "
              "news feed could be worth: if same-day headlines go with far more strong first "
              "hours than the previous day's, the timing is what is missing."),
        show(_hind, rows=3),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Validation: the four pre-registered questions (read once)

    Written into the README before anything ran. Each is scored on discovery (November 2025 →
    February 2026) and on the locked validation months (March → June 2026); the models are
    walked forward exactly as in §3c (trained on all months before the one scored), with a
    strong first hour fixed at 2× usual and a moved night at 0.5 ATR.

    - **V1** · Does the night's price move add to the previous days, for a strong first hour?
      Passes when the AUC gain's 95% interval is above 0.
    - **V2** · Do the night's trading activity and the news add beyond the move? Passes when
      the AUC gain's 95% interval is above 0.
    - **V3** · Does the first hour continue (or fade) the night's move, beyond the market?
      Passes when |t| ≥ 2 with the same sign as in discovery.
    - **V4** · Does it continue more (or less) on news nights than on quiet ones? Passes when
      |t| ≥ 2 with the same sign as in discovery.

    Read it **once**: a second look after changing anything turns the validation months into
    discovery months. Turn the switch on, then click.
    """)
    return


@app.cell
def _(mo):
    read_validation = mo.ui.switch(value=False, label="I have finished exploring: read validation")
    validate_button = mo.ui.run_button(label="score V1-V4 (about three minutes)")
    mo.hstack([read_validation, validate_button], justify="start", gap=2)
    return read_validation, validate_button


@app.cell
def _(json, mo, pd, read_validation, rs, show, table, validate_button):
    mo.stop(not (read_validation.value and validate_button.value),
            mo.md("_Validation is locked: switch it on and click to read it._"))
    with mo.status.spinner("discovery and validation, walked forward ..."):
        _disc = rs.prereg(table, first_month=rs.SCORE_FROM, last_month="2026-02")
        _val = rs.prereg(table, first_month=rs.VALIDATION_FROM, last_month="2026-06")
    _rows = []
    for _k in ("V1", "V2"):
        _g, _lo, _hi = _val[_k]
        _rows.append({"test": _k, "discovery": f"{_disc[_k][0]:+.4f} [{_disc[_k][1]:+.4f}, "
                                               f"{_disc[_k][2]:+.4f}]",
                      "validation": f"{_g:+.4f} [{_lo:+.4f}, {_hi:+.4f}]",
                      "pass": bool(_lo > 0)})
    _v3d, _v3v = _disc["V3"], _val["V3"]
    _rows.append({"test": "V3", "discovery": f"{_v3d['bp']:+.1f} bp, t {_v3d['t']:+.1f}",
                  "validation": f"{_v3v['bp']:+.1f} bp, t {_v3v['t']:+.1f}",
                  "pass": bool(abs(_v3v["t"]) >= 2 and _v3v["bp"] * _v3d["bp"] > 0)})
    _v4d, _v4v = _disc["V4"], _val["V4"]
    _rows.append({"test": "V4", "discovery": f"{_v4d['diff']:+.1f} bp, t {_v4d['t']:+.1f}",
                  "validation": f"{_v4v['diff']:+.1f} bp, t {_v4v['t']:+.1f}",
                  "pass": bool(abs(_v4v["t"]) >= 2 and _v4v["diff"] * _v4d["diff"] > 0)})
    _verdict = pd.DataFrame(_rows)
    _card = {"study": "overnight_open-v1", "pass": _verdict.set_index("test")["pass"].to_dict(),
             "V3_bp": round(_v3v["bp"], 2), "V4_bp": round(_v4v["diff"], 2)}
    print("SCORECARD " + json.dumps(_card))
    mo.vstack([show(_verdict), mo.md(
        f"**{int(_verdict['pass'].sum())} of 4 pass.** V1-V2 are AUC gains with 95% intervals; "
        "V3 is the first hour's continuation of the night beyond the market, V4 its "
        "difference between news and quiet nights (bp, t clustered by day). Write the result "
        "into the README's verdict section.")])
    return


if __name__ == "__main__":
    app.run()
