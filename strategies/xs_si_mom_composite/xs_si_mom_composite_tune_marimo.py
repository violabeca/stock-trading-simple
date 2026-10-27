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
    # xs_si_mom_composite · tuning around version B, overnight

    The standard setting is **version B**: the shipped rules on the 2000 most traded stocks
    (RESEARCH_LOG §13.5). A first search that changed all 33 knobs at random found nothing
    better: almost every random setting carries several bad choices at once, and 100 points
    cannot cover 33 knobs. This version searches **around version B** instead, in three stages,
    and then checks whether its best setting would really have done better **without knowing
    the future**.

    **One click runs the whole night** (set the time budget in step 2; 8 hours by default):
    1. **Each knob alone:** every listed value of every knob, everything else at version B
       (about 140 backtests, about an hour).
    2. **Combine the winners:** start from version B and add the single changes that helped,
       best first, keeping each one only if the combination scores higher still.
    3. **Small random combinations** of 2 to 5 changes around version B, until only the time
       the checks need is left.
    4. **The checks** (about 45 minutes), then the verdict.

    **Everything is saved as it runs.** If the night is interrupted, click again: settings
    already backtested are read back, not rerun. Keep the Mac awake, for example by running
    `caffeinate -i` in a terminal. The last step, the reserved months, is read by hand, once.
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
    import hashlib
    import importlib.util

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine
    from trading_simple.plots import _GRID as GRID
    from trading_simple.plots import _LOSS as LOSS
    from trading_simple.plots import _MUTED as MUTED
    from trading_simple.plots import _WIN as WIN
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as layout

    _folder = strategies.STRATEGIES_DIR / "xs_si_mom_composite"
    _spec = importlib.util.spec_from_file_location("xs_si_mom_composite_tune", _folder / "tune.py")
    tune = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(tune)
    settings = load_settings()
    engine = make_engine(settings)
    STUDIES = _folder / "studies"
    TUNE_SOURCE = hashlib.sha1((_folder / "tune.py").read_bytes()).hexdigest()[:8]

    def finish(fig, title, height=420):
        """The repo's chart style, with the legend UNDER the chart."""
        layout(fig, title=title, height=height)
        fig.update_layout(legend={"orientation": "h", "yanchor": "top",
                                  "y": -40 / max(height - 150, 100), "xanchor": "left", "x": 0.0},
                          margin={"l": 60, "r": 20, "t": 50, "b": 100})
        return fig

    def days(index):
        """Daily timestamps as date strings (a date axis that no browser time zone shifts)."""
        return pd.DatetimeIndex(index).strftime("%Y-%m-%d")

    return (GRID, LOSS, MUTED, SERIES, STUDIES, TUNE_SOURCE, WIN, days, engine, finish, go,
            hashlib, make_subplots, pd, settings, tune)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The data

    - **One panel** of every stock the widest universe (the 2000 most traded, any price) held
      since 2022: daily prices from mid-2019, FINRA short interest, SEC share counts and
      profitability, and sectors. Narrower universes are cut from it. It loads from the cache
      in `var/` (the first load takes about 10 minutes).
    - **The search window is 2022-01 → 2025-12.** Before mid-2020 the database holds only about
      500 stocks, and momentum needs a year of history. **2026-01 → today is reserved** for the
      last step.
    - **The check below** rebuilds version B with this notebook's code and compares it with the
      shipped strategy. 0 means it trades exactly the strategy.
    """)
    return


@app.cell
def _(mo):
    reload_button = mo.ui.run_button(label="reload from the database")
    mo.hstack([reload_button], justify="start")
    return (reload_button,)


@app.cell
def _(engine, mo, reload_button, settings, tune):
    with mo.status.spinner("loading the data ..."):
        data = tune.load(engine, settings, refresh=reload_button.value)
    with mo.status.spinner("checking that this notebook trades the shipped strategy ..."):
        _diff = tune.check(data)
    _p = data["panel"]
    _same = ("identical: the knobs below change the real strategy" if _diff < 1e-9
             else "NOT identical: stop and look")
    mo.md(f"""
    **Loaded:** {_p.close.shape[1]:,} stocks, {_p.close.index[0].date()} →
    **{_p.close.index[-1].date()}**, {len(data['ranked'])} universe formations.

    **Check:** the largest difference between this notebook's version B and the shipped
    `signals()` is **{_diff:.1e}** ({_same}).
    """)
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · What to search

    **All 33 knobs** are listed below with version B's value and every value the search tries.
    Untick a knob to keep it at version B.

    **Stop-losses** (group "stops"; version B has none):
    - A long is sold when its close falls **k daily volatilities** below its entry (or below
      its best close since entry, with "best close"). A short is covered when its close rises
      that far. The volatility is the stock's own: the standard deviation of its daily returns
      over the 20 sessions before the entry. Example: a stock that moves about 2% a day, k = 5:
      the stop sits 10% away.
    - Like every decision of this strategy, a stop is read at the close and traded at the next
      open. A real stop order would fill during the day it is hit, often a little better.
    - A stopped stock is not bought (or shorted) again until its rank has left the entry zone
      once.

    **The account size is a knob too, up to $100k.** Every order pays its own costs:
    - commission at its real size ($0.005 a share, at least $1);
    - half the spread (about 1-3 bp by how much the stock trades, never below one cent);
    - market impact, which grows with the order against the stock's daily trading
      (volatility × √(order ÷ daily dollar volume)).

    A bigger account pays less commission per dollar but moves thin stocks more. **If your
    account is fixed, untick `account`:** every setting then uses the amount chosen below.

    **Kept fixed, never searched:** the borrow-fee model (0.5% a year, 3% above 10% of the
    shares shorted, 10% above 20%), the cost model above, and market-on-open fills. They
    describe what trading costs, not choices.

    **What "better" means** (the score, all against equal-weight buy-and-hold of version B's
    2000 stocks, so every universe is judged on one yardstick):
    - **robust** (recommended): each half-year's alpha, mean minus half their spread. A steady
      edge beats one lucky stretch.
    - **alpha:** the yearly return beyond what the book's market exposure explains.
    - **alpha ratio:** alpha per unit of the book's own stock-specific risk.

    A setting must trade at least 20 times a year to be picked.
    """)
    return


@app.cell
def _(table, tune):
    table(tune.knob_table(), 33)
    return


@app.cell
def _(mo, tune):
    include_pick = mo.ui.multiselect(options=tune.NAMES, value=tune.NAMES,
                                     label="knobs to search")
    account_pick = mo.ui.dropdown(options={f"${a:,.0f}": a for a in tune.GRID["account"]},
                                  value="$33,000", label="your account (version B's)")
    how = mo.ui.dropdown(options={"robust (steady half-years)": "robust", "alpha": "alpha",
                                  "alpha ratio (per unit of risk)": "alpha_ratio"},
                         value="robust (steady half-years)", label="better means")
    hours = mo.ui.number(start=1, stop=24, step=0.5, value=8, label="time budget (hours)")
    n_luck = mo.ui.number(start=10, stop=500, step=10, value=100, label="luck books (check C)")
    mo.vstack([include_pick, mo.hstack([account_pick, how, hours, n_luck], justify="start",
                                       gap=2)])
    return account_pick, how, hours, include_pick, n_luck


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Version B, the setting to beat

    The shipped rules on the top 2000, run on the search window (2022-01 → 2025-12) with your
    account. Every metric below is also computed for every setting the search tries.
    """)
    return


@app.cell
def _(SERIES, account_pick, data, days, finish, go, how, mo, pd, settings, table, tune):
    base_knobs = {**tune.standard(), "account": int(account_pick.value)}
    with mo.status.spinner("running version B ..."):
        base_m, _r = tune.evaluate(data, base_knobs, settings=settings, how=how.value)
    _b = tune.benchmark(data, _r.index)
    _spy = data["spy"].reindex(_r.index).fillna(0.0)
    _fig = go.Figure()
    for _x, _name, _c in ((_r, "version B", SERIES[0]),
                          (_b, "equal-weight buy-and-hold of B's stocks", SERIES[2]),
                          (_spy, "SPY", SERIES[1])):
        _fig.add_scatter(x=days(_x.index), y=(1 + _x).cumprod(), name=_name, mode="lines",
                         line={"color": _c})
    finish(_fig, "Growth of $1 on the search window", 360)
    _show = pd.Series(base_m).to_frame("version B").rename_axis("metric")
    mo.vstack([_fig, table(_show.round(4), 40)])
    return base_knobs, base_m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The overnight run

    - **Saved as it runs** in `studies/`, under a name built from the searched knobs, the score,
      your account and this code's version. Change any of them and a fresh search starts; click
      again with the same choices and it resumes.
    - **Stages 1 and 3 never look at results** (a fixed list, and random draws from a fixed
      seed). Stage 2 does, so the walk-forward check reruns it on the past only, half-year by
      half-year. That tests exactly the rule that picks the winner.
    - The checks run straight after the search, inside the same click.
    """)
    return


@app.cell
def _(STUDIES, TUNE_SOURCE, base_knobs, data, hashlib, how, include_pick, mo, settings,
      tune):
    include = [n for n in tune.NAMES if n in include_pick.value]
    _key = repr((include, how.value, base_knobs["account"], tune.FIRST, tune.TRAIN_END,
                 TUNE_SOURCE))
    study_name = f"xs-around-b-{hashlib.sha1(_key.encode()).hexdigest()[:10]}"
    search = tune.Search(data, settings, how.value, base_knobs, STUDIES, study_name)
    _done = search.results() is not None
    mo.md(f"Search `{study_name}`: **{len(search.runs)} settings backtested so far**"
          + (", checks finished." if _done else "."))
    return include, search, study_name


@app.cell
def _(mo):
    night_button = mo.ui.run_button(label="run the search and the checks (overnight)")
    mo.hstack([night_button], justify="start")
    return (night_button,)


@app.cell
def _(hours, include, mo, n_luck, night_button, search):
    if night_button.value:
        res = search.overnight(include, float(hours.value), int(n_luck.value),
                               progress=lambda it, title: mo.status.progress_bar(it, title=title))
    else:
        res = search.results()
    mo.stop(res is None, mo.md(
        "_No finished night yet: click **run the search and the checks (overnight)**._"))
    runs = search.table()
    _secs = sorted(search.seconds)
    mo.md(f"**Done:** {len(search.runs)} settings backtested (median "
          f"{_secs[len(_secs) // 2]:.0f} seconds each), {res['n_pool']} of them in stage 3.")
    return res, runs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · What the search found

    ### Stage 1: each knob alone
    Every dot is version B with **one** knob changed to one of its values. Right of the line is
    a better score than version B, left a worse one. Hover for the value. A knob whose dots all
    sit near the line hardly matters; one with dots far to the right is worth a look, but a
    single dot far out on one value is often luck.
    """)
    return


@app.cell
def _(GRID, LOSS, WIN, base_m, finish, go, mo, runs, tune):
    _s = runs[runs["stage"] == "sweep"].copy()
    _s["knob"] = _s["changes vs B"].str.split(" = ").str[0]
    _s["value"] = _s["changes vs B"].str.split(" = ").str[1]
    _s["delta"] = _s["score"] - base_m["score"]
    _order = [n for n in tune.NAMES if n in set(_s["knob"])]
    _fig = go.Figure()
    for _good, _c, _name in ((True, WIN, "better than version B"),
                             (False, LOSS, "worse than version B")):
        _x = _s[(_s["delta"] > 0) == _good]
        _fig.add_scatter(x=_x["delta"], y=_x["knob"], mode="markers", name=_name,
                         marker={"color": _c, "size": 9, "opacity": 0.8}, text=_x["value"],
                         hovertemplate="%{y} = %{text}<br>score change %{x:+.3f}<extra></extra>")
    _fig.add_vline(x=0, line={"color": GRID, "width": 2})
    _fig.update_yaxes(categoryorder="array", categoryarray=_order[::-1])
    _fig.update_xaxes(title_text="score change against version B")
    finish(_fig, "Each knob alone: how the score changes", 30 + 22 * len(_order) + 150)
    mo.vstack([_fig, mo.md("Green dots beat version B, red dots lose to it; the grey line is "
                           "version B. Version B's own value of each knob sits on that line.")])
    return


@app.cell
def _(include, mo):
    knob_pick = mo.ui.dropdown(options=include, value=include[0] if include else None,
                               label="one knob in detail")
    mo.hstack([knob_pick], justify="start")
    return (knob_pick,)


@app.cell
def _(SERIES, base_knobs, base_m, finish, knob_pick, make_subplots, mo, runs, tune):
    _k = knob_pick.value
    _s = runs[(runs["stage"] == "sweep") & runs["changes vs B"].str.startswith(f"{_k} = ")]
    _rows = [{"value": str(v).split(" = ")[1], "score": sc, "alpha": al}
             for v, sc, al in zip(_s["changes vs B"], _s["score"], _s["alpha"], strict=True)]
    _rows.append({"value": str(base_knobs[_k]), "score": base_m["score"],
                  "alpha": base_m["alpha"]})
    _grid = [str(v) for v in tune.GRID[_k]]
    _rows.sort(key=lambda r: _grid.index(r["value"]) if r["value"] in _grid else 99)
    _fig = make_subplots(rows=1, cols=2, subplot_titles=("score", "alpha a year"))
    for _col, _i in (("score", 1), ("alpha", 2)):
        _fig.add_scatter(x=[r["value"] for r in _rows], y=[r[_col] for r in _rows],
                         mode="lines+markers", line={"color": SERIES[0]},
                         marker={"size": 10, "color": [SERIES[3] if r["value"] == str(
                             base_knobs[_k]) else SERIES[0] for r in _rows]},
                         name=_col, showlegend=False, row=1, col=_i)
    _fig.add_scatter(x=[None], y=[None], mode="markers", marker={"color": SERIES[3], "size": 10},
                     name="version B's value")
    _fig.update_xaxes(type="category")
    finish(_fig, f"{_k}: every value, everything else at version B", 380)
    mo.vstack([_fig, mo.md(f"_{dict((k[0], k[4]) for k in tune.KNOBS)[_k]}_")])
    return


@app.cell
def _(mo, pd, res, table):
    _p = pd.DataFrame(res["path"])
    mo.vstack([mo.md("""
    ### Stage 2: combining the single changes that helped (whole search window)
    Best first; "kept" means the combination scored higher than without that change.
    """), table(_p.round(4), 30) if len(_p) else mo.md("_No single change beat version B._")])
    return


@app.cell
def _(mo, runs, table):
    _cols = ["trial", "stage", "changes vs B", "score", "alpha", "cagr", "volatility", "sharpe",
             "max drawdown", "worst month", "beta", "alpha vs SPY", "trades / yr",
             "stop exits / yr", "median days held", "costs / yr", "long exposure",
             "short exposure", "order % of daily $ volume (p95)"]
    _t = runs[[c for c in _cols if c in runs]].sort_values("score", ascending=False)
    mo.vstack([mo.md("""
    ### Every setting backtested
    Best score first. Stages: B, sweep (one knob), greedy (stage 2), pool (stage 3), neighbour
    (check D). Returns, alphas and costs are yearly rates. Click a column to sort by it.
    """), table(_t.round(4), 15)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · The checks: is the pick really better?

    **The pick** is the best-scoring setting on the whole search window among version B, every
    stage-1 and stage-3 setting, and the stage-2 combination. The verdict needs **A to F** to
    pass.

    - **A walk-forward.** For every half-year from 2024 on, the whole rule (stage 2 on the past
      half-years, then the best on the past) picks a setting; the next half-year is traded with
      it. *Passes:* the stitched record beats version B with P(really better) ≥ 0.80.
    - **B overfitting (PBO).** Over every split of the half-years into two halves: how often
      does the best setting on one half land in the bottom half on the other? *Passes:* below
      0.5 (0.5 is picking blindly).
    - **C luck.** Does the pick beat the same book on randomly swapped stocks of similar
      liquidity? *Passes:* it beats 95% of them.
    - **D neighbours.** Do the settings one listed value away on any knob keep the gain, or is it
      one lucky point? *Passes:* their median alpha is above version B's.
    - **E costs.** Fills 0.5% worse, every cost doubled, whole shares. *Passes:* alpha above 0
      every time, and at least version B's with the worse fills.
    - **F missing stocks.** 5% of the stocks removed at random, three times. *Passes:* alpha
      above 0 each time, on average at least version B's.
    - **G account size** (information only): the pick at every account size up to $100k.
    """)
    return


@app.cell
def _(base_knobs, base_m, mo, pd, res, search, table, tune):
    best = res["best"]
    _best_m = search.runs[tune.label(best, base_knobs)]["metrics"]
    _diff = pd.DataFrame([{"knob": n, "version B": base_knobs[n], "pick": v}
                          for n, v in tune.changes(best, base_knobs).items()])
    _keys = ["score", "alpha", "cagr", "volatility", "sharpe", "max drawdown", "worst month",
             "beta", "trades / yr", "stop exits / yr", "median days held", "costs / yr",
             "order % of daily $ volume (p95)"]
    _cmp = pd.DataFrame({"version B": [base_m[k] for k in _keys],
                         "pick": [_best_m.get(k) for k in _keys]},
                        index=_keys).rename_axis("metric")
    _same = tune.label(best, base_knobs) == "version B"
    mo.vstack([mo.md("### The pick" + (": version B itself (nothing beat it)" if _same else "")),
               table(_diff, 33) if len(_diff) else mo.md(""), table(_cmp.round(4), 20)])
    return (best,)


@app.cell
def _(mo, res, table):
    wf = res["wf"]
    pbo = res["pbo"]
    mo.vstack([mo.md(f"""
    ### A · Walk-forward
    Each half-year is traded with the setting the whole rule picked on the half-years **before**
    it.
    - **Tuned, picked on the past:** alpha **{wf['tuned']:+.1%}** a year
    - **Version B on the same days:** **{wf['standard']:+.1%}**
    - **Difference {wf['delta']:+.1%}**, P(tuning really better) **{wf['P(delta > 0)']:.2f}**
    """), table(wf["table"].round(4), 10), mo.md(f"""
    ### B · Overfitting (PBO)
    **{pbo:.2f}**: how often the best setting on half of the half-years lands in the bottom half
    on the other half. 0 means the search finds real differences, 0.5 is picking blindly.
    """)])
    return pbo, wf


@app.cell
def _(LOSS, MUTED, finish, go, mo, res):
    luck = res["luck"]
    _fig = go.Figure(go.Histogram(x=luck["books"], nbinsx=25, marker={"color": MUTED},
                                  name="the same book on swapped stocks"))
    _fig.add_vline(x=luck["real"], line={"color": LOSS, "width": 3})
    _fig.add_scatter(x=[None], y=[None], mode="lines", line={"color": LOSS, "width": 3},
                     name="the pick")
    _fig.update_xaxes(title_text="alpha a year", tickformat=".0%")
    finish(_fig, "C · Luck: the pick against swapped-stock books", 380)
    mo.vstack([_fig, mo.md(
        f"The grey bars are the same rules, timing and turnover, on stocks swapped at random "
        f"within the same liquidity tenth. The pick (red line, alpha {luck['real']:+.1%}) beats "
        f"**{luck['percentile']:.0f}%** of them.")])
    return (luck,)


@app.cell
def _(base_m, mo, res, table):
    near = res["near"]
    _med = float(near["alpha"].median()) if len(near) else float("nan")
    mo.vstack([mo.md(f"""
    ### D · Neighbours
    {len(near)} settings one listed value away on one knob. Median alpha **{_med:+.1%}** against
    version B's **{base_m['alpha']:+.1%}**. A real improvement sits on a plateau; a lucky one is
    a spike its neighbours do not share.
    """), table(near.sort_values("alpha").round(4), 12) if len(near) else mo.md("")])
    return (near,)


@app.cell
def _(mo, res, table):
    costs = res["costs"]
    mo.vstack([mo.md("### E · Costs\nThe same books with harsher trading. Live fills are rarely "
                     "exactly at the open, and an edge that needs more trading pays for it here."),
               table(costs.round(4), 6)])
    return (costs,)


@app.cell
def _(base_m, mo, res, table):
    missing = res["missing"]
    mo.vstack([mo.md(f"""
    ### F · Missing stocks
    The pick with 5% of the stocks removed at random, three times: alpha
    **{', '.join(f'{a:+.1%}' for a in missing)}** (version B with all stocks:
    {base_m['alpha']:+.1%}). A setting that needs a few particular stocks falls apart here.

    ### G · Account size (information)
    Larger accounts pay less commission per dollar (the $1 minimum) but more market impact as the
    orders grow against the stocks' daily trading.
    """), table(res["capacity"].round(4), 6)])
    return (missing,)


@app.cell
def _(base_knobs, base_m, best, costs, luck, missing, mo, near, pbo, pd, table, tune, wf):
    _late = costs.iloc[1]
    _gates = [
        ("A walk-forward", wf["delta"] > 0 and wf["P(delta > 0)"] >= 0.8,
         f"Δ {wf['delta']:+.1%}, P {wf['P(delta > 0)']:.2f}"),
        ("B overfitting", pbo < 0.5, f"PBO {pbo:.2f}"),
        ("C luck", luck["percentile"] >= 95, f"beats {luck['percentile']:.0f}% of swapped books"),
        ("D neighbours", len(near) > 0 and near["alpha"].median() > base_m["alpha"],
         f"median {near['alpha'].median():+.1%} vs B {base_m['alpha']:+.1%}" if len(near)
         else "no neighbours"),
        ("E costs", bool((costs["pick alpha"] > 0).all())
         and _late["pick alpha"] >= _late["B alpha"],
         f"worst {costs['pick alpha'].min():+.1%}; +50 bp: {_late['pick alpha']:+.1%}"
         f" vs B {_late['B alpha']:+.1%}"),
        ("F missing stocks", min(missing) > 0 and sum(missing) / len(missing) >= base_m["alpha"],
         ", ".join(f"{a:+.1%}" for a in missing))]
    verdict = pd.DataFrame([{"check": g, "result": "PASS" if ok else "FAIL", "numbers": n}
                            for g, ok, n in _gates])
    _new = tune.label(best, base_knobs) != "version B"
    _all = bool((verdict["result"] == "PASS").all()) and _new
    _acct = ("" if best["account"] == base_knobs["account"] else
             f" Note: it assumes an account of ${best['account']:,.0f}, not your "
             f"${base_knobs['account']:,.0f}.")
    mo.vstack([mo.md("## Verdict"), table(verdict, 8), mo.callout(mo.md(
        f"**Paper-trade the pick next to version B.** It passed every check. Before real money, "
        f"read the reserved months below once.{_acct}" if _all else
        "**Keep version B.** The pick did not pass every check, so its advantage cannot be told "
        "apart from luck."), kind="success" if _all else "warn")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · The reserved months (2026-01 → today)

    The search and the checks never saw these months. Read them **once**, when you are done
    tuning. Looking, changing a knob and looking again turns them into one more search window.
    It is a short stretch: a gap of a few points a year between two settings is noise here
    (RESEARCH_LOG §13.5).

    **Fair warning:** version B's own 2026 has been looked at before (RESEARCH_LOG §13.5, and
    while this notebook was built: alpha about −2% a year). For a new setting these months
    are unseen; for version B they are not.
    """)
    return


@app.cell
def _(mo):
    final_ok = mo.ui.checkbox(label="I have finished tuning: show the reserved months")
    final_button = mo.ui.run_button(label="read them")
    mo.hstack([final_ok, final_button], justify="start", gap=2)
    return final_button, final_ok


@app.cell
def _(SERIES, base_knobs, best, data, days, final_button, final_ok, finish, go, how, mo,
      settings, tune):
    mo.stop(not (final_button.value and final_ok.value),
            mo.md("_Tick the box and click **read them**._"))
    _rows = {}
    for _label, _k in (("version B", base_knobs), ("the pick", best)):
        _rows[_label] = tune.evaluate(data, _k, settings=settings, how=how.value,
                                      start=tune.FINAL, end=None)
    _fig = go.Figure()
    for (_label, (_m, _r)), _c in zip(_rows.items(), (SERIES[0], SERIES[4]), strict=True):
        _fig.add_scatter(x=days(_r.index), y=(1 + _r).cumprod(), name=_label, mode="lines",
                         line={"color": _c})
    _b = tune.benchmark(data, next(iter(_rows.values()))[1].index)
    _fig.add_scatter(x=days(_b.index), y=(1 + _b).cumprod(), mode="lines",
                     line={"color": SERIES[2]}, name="equal-weight buy-and-hold of B's stocks")
    finish(_fig, "The reserved months: growth of $1", 360)
    _t = "\n".join(f"- **{_l}:** alpha {_m['alpha']:+.1%} a year, max drawdown "
                   f"{_m['max drawdown']:.1%}" for _l, (_m, _) in _rows.items())
    mo.vstack([_fig, mo.md(_t)])
    return


if __name__ == "__main__":
    app.run()
