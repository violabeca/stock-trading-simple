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
    # One stock under the microscope

    A deep dive into ONE stock of the per-stock run (`darts_forecast_tsmixer_marimo.py`): its
    study, its saved best model, and new inputs. Same question as there: at the decision time,
    is the stock up or down at the end of the horizon, and by roughly how much?

    1. **The study**: every trial of the stock, and the best trial's settings
    2. **How the score works**: the end correlation in pictures and in dollars, and nine
       VALIDATION days up close (the best, some in between, the worst)
    3. **Is the best trial real?** Which days carry its score, and what other random seeds give
    4. **Which inputs does the model use?** Permutation importance and SHAP
    5. **New inputs**: write one as a short function, check it for look-ahead and for
       information, then train the best trial's settings with and without it over several
       seeds: does the score rise by more than the seeds move it?

    Everything is scored on VALIDATION; TEST stays unread here.
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
    import json
    import logging
    import warnings

    import numpy as np
    import optuna
    import pandas as pd
    import plotly.graph_objects as go
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    from plotly.subplots import make_subplots

    from trading_simple import backtest as bt
    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    warnings.filterwarnings("ignore")
    logging.getLogger("darts").setLevel(logging.ERROR)  # Darts' notes about optional models
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _helpers(name):
        """One of the strategy folder's helper modules, loaded by path."""
        path = strategies.STRATEGIES_DIR / "darts_forecast" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"darts_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    sl, imp, charts = _helpers("stock_lab"), _helpers("tsmixer_importance"), _helpers("charts")
    lab, ds = sl.lab, sl.ds  # tsmixer_lab and daystart.py, as stock_lab loads them
    settings = load_settings()
    engine = make_engine(settings)
    return (
        JournalFileBackend,
        JournalStorage,
        bt,
        charts,
        ds,
        engine,
        go,
        imp,
        json,
        lab,
        make_subplots,
        np,
        optuna,
        pd,
        settings,
        sl,
        strategies,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Settings

    The stock, the per-stock batch whose best model is studied, and the dates of that batch
    (the notebook checks that the saved model reproduces its score on them). `SEEDS`: how many
    random seeds every comparison trains: more seeds, smaller noise, longer runs.
    """)
    return


@app.cell
def _():
    STOCK = "CRM"
    BATCH = None  # the per-stock batch, e.g. "ba8dbcc8b" (None: the newest with a STOCK model)
    TRAIN = ("2025-01-01", "2025-12-31")  # as in the batch
    VALIDATION = ("2026-01-01", "2026-06-30")  # as in the batch: everything here is scored on it
    TEST = ("2026-07-01", "2026-09-30")  # as in the batch: loaded, never read here
    SEEDS = 5  # random seeds per comparison (the seed alone moves the score by ±0.05 to ±0.14)
    EPOCHS = 60  # as in the batch
    PATIENCE = 5  # as in the batch
    DRAWS = 200  # random up/down outcomes behind every luck check
    return (
        BATCH,
        DRAWS,
        EPOCHS,
        PATIENCE,
        SEEDS,
        STOCK,
        TEST,
        TRAIN,
        VALIDATION,
    )


@app.cell(hide_code=True)
def _(
    BATCH,
    JournalFileBackend,
    JournalStorage,
    STOCK,
    TEST,
    TRAIN,
    VALIDATION,
    engine,
    json,
    lab,
    mo,
    np,
    optuna,
    settings,
    sl,
    strategies,
):
    _models = lab.BASKET_DIR / "models"
    _found = ([_models / BATCH / f"{STOCK}.json"] if BATCH else
              sorted(_models.glob(f"*/{STOCK}.json"), key=lambda p: p.stat().st_mtime))
    mo.stop(not _found or not _found[-1].exists(),
            mo.md(f"_No saved model for {STOCK} in `{_models}`: run it in "
                  "`darts_forecast_tsmixer_marimo.py` first._"))
    meta = json.loads(_found[-1].read_text())
    cell, base_feats = tuple(meta["cell"]), list(meta["inputs"])
    prev, knobs = meta["prev_days"], meta["knobs"]
    _log = strategies.STRATEGIES_DIR / "darts_forecast" / "studies" / f"{meta['study']}.log"
    study = optuna.load_study(study_name=meta["study"],
                              storage=JournalStorage(JournalFileBackend(str(_log))))
    with mo.status.spinner(title=f"Loading {STOCK} (about 20 seconds) ..."):
        data = sl.load(settings, engine, STOCK, train=TRAIN, validation=VALIDATION, test=TEST,
                       decision=cell[1] + cell[2])
    saved_model = lab.load_model(_found[-1].with_suffix(".ckpt"), cell, knobs)
    _raw, _d, _k = lab.predict(saved_model, data["F"], cell, data["val"], base_feats,
                               prev_days=prev)
    saved = {"pe": _raw.sum(1) * meta["beta"], "ae": lab.end_moves(data["F"], cell, _d, _k),
             "d": _d, "k": _k, "dates": np.array([data["F"]["dates"][x] for x in _d]),
             "paths": _raw}  # the predicted minute moves (typical moves, before β)
    _now = float(np.corrcoef(saved["pe"], saved["ae"])[0, 1])
    if abs(_now - meta["end_corr"]) > 1e-6:
        raise ValueError(f"the saved model scores {_now:+.4f} here but {meta['end_corr']:+.4f} "
                         "in its study: TRAIN / VALIDATION / TEST must be the batch's")
    fits = {}  # every seed's fit of the comparisons below, kept while the stock stays
    return base_feats, cell, data, fits, knobs, meta, prev, saved, saved_model, study


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The study

    **Figure 1**: every trial's end correlation on VALIDATION. The first 10 are random draws,
    the others chosen by the optimizer. **Table 1** lists the best trials: when the top trials
    agree on their settings, the stock's signal is more likely real than when one trial stands
    out alone.
    """)
    return


@app.cell(hide_code=True)
def _(base_feats, cell, charts, ds, go, knobs, meta, mo, np, pd, prev, sl, study, table):
    _done = [t for t in study.trials if t.state.name == "COMPLETE"]
    _vals = np.array([t.value for t in _done])
    _first = [t.value for t in sorted(_done, key=lambda t: t.number)[:10]]
    _bar = sl.ex.best_of_bar(len(_done))
    _fig = go.Figure()
    for _name, _sel, _i in (("random (first 10)", lambda t: t.number < 10, 1),
                            ("chosen by the optimizer", lambda t: t.number >= 10, 0)):
        _part = [t for t in _done if _sel(t)]
        _fig.add_trace(go.Scatter(
            x=[t.number for t in _part], y=[t.value for t in _part], mode="markers", name=_name,
            marker={"size": 8, "color": charts.SERIES[_i]},
            text=[f"luck z {t.user_attrs['luck_z']:+.2f}" for t in _part]))
    _fig.add_hline(y=0, line={"color": charts._MUTED, "width": 1})  # after the traces
    charts._base(_fig, title="Figure 1 · Every trial's end correlation (0 = no skill)",
                 height=380)
    _fig.update_xaxes(title_text="trial")
    _top = pd.DataFrame([{
        "trial": t.number, "end corr": t.value, "luck z": t.user_attrs["luck_z"],
        "hit rate": t.user_attrs["hit_rate"], "history": t.params["history"],
        "loss": t.params["loss"], "prev days": t.params.get("prev_days", "–"),
        "hidden": t.params["hidden_size"], "blocks": t.params["num_blocks"],
        "lr": t.params["lr"]} for t in sorted(_done, key=lambda t: -t.value)[:10]])
    _, _off, _L, _H = cell
    _knobs = ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                       for k, v in knobs.items())
    mo.vstack([
        mo.md(f"**{meta['stock']}** ({meta['group']}), batch `{meta['batch']}`: "
              f"{len(_done)} trials. Best: trial #{meta['trial']}, end correlation "
              f"**{meta['end_corr']:+.3f}**, luck z **{meta['luck_z']:+.2f}** against its bar "
              f"{_bar:.2f} (the z the best of {len(_done)} trials reaches by luck 2.5% of the "
              f"time). Without selection: the first 10 trials average "
              f"**{np.mean(_first):+.3f}**, the median trial is **{np.median(_vals):+.3f}**. "
              f"Its model reads {ds.clock(_off)}–{ds.clock(_off + _L)} and predicts to "
              f"{ds.clock(_off + _L + _H)}, with {', '.join(base_feats)} (previous days: "
              f"{prev}); knobs: {_knobs}."),
        _fig, mo.md("### Table 1 · The 10 best trials"), table(_top.round(4), rows=10),
    ])
    return


@app.cell(hide_code=True)
def _(cell, mo, np, saved):
    _n, _H = len(saved["ae"]), cell[3]
    mo.md(f"""
    ## 2 · How the score works

    Every VALIDATION day the model makes **one number** at the decision: how far the stock will
    move by the end of the horizon, in typical moves (one ordinary minute's size, so an ordinary
    {_H}-minute stretch moves about ±{np.sqrt(_H):.0f}). The score, the **end correlation**,
    lines these {_n} numbers up with what happened:

    - each day: (prediction − the average prediction) × (actual − the average actual), each
      divided by its spread; the score is the average of these products, between −1 and +1
    - a day where both sit on the **same side of their averages adds** (green in Figure 3), a day
      on opposite sides **subtracts** (red). A big move adds or subtracts a lot, a quiet day almost
      nothing. The averages, not zero: if the model's average prediction is slightly down, a
      smaller "down" counts as leaning up
    - only the **direction and the relative size** count: a model that says +0.1 before every +10
      and −0.1 before every −10 scores a perfect 1. The size is set apart, by β on the held-out
      days
    - **Figure 2**: what correlations of 0, 0.1, 0.3 and 0.6 look like over {_n} days. Without
      skill, luck alone reaches about ±{2 / np.sqrt(_n):.2f} in one try out of 20; the best of
      many tries reaches more (the bars in section 1)
    - **in money (Figure 4b)**: a trader who buys or sells at the decision in proportion to the
      prediction, and closes at the horizon's end, earns per day about **correlation × typical
      position × typical move** (Grinold & Kahn's "fundamental law": return ≈ IC × volatility).
      In quant finance a correlation of 0.05 that holds up is already a good forecast of
      short-term prices; costs come off every day
    """)
    return


@app.cell(hide_code=True)
def _(bt, cell, charts, data, ds, go, make_subplots, mo, np, saved, settings, sl):
    _p, _a, _n = saved["pe"], saved["ae"], len(saved["ae"])
    _rng = np.random.default_rng(7)  # Figure 2: simulated days, the same noise in every panel
    _x, _e = _rng.standard_normal(_n), _rng.standard_normal(_n)
    _ys = {r: r * _x + np.sqrt(1 - r ** 2) * _e for r in (0.0, 0.1, 0.3, 0.6)}
    _sim = make_subplots(rows=1, cols=4, shared_yaxes=True, subplot_titles=[
        f"correlation {r:.1f} (drawn: {np.corrcoef(_x, y)[0, 1]:+.2f})" for r, y in _ys.items()])
    for _i, _y in enumerate(_ys.values(), start=1):
        _sim.add_trace(go.Scatter(x=_x, y=_y, mode="markers", showlegend=False, marker={
            "size": 5, "color": charts.SERIES[0], "opacity": 0.6}), row=1, col=_i)
    charts._base(_sim, height=320, title=f"Figure 2 · What a correlation looks like: {_n} "
                 "simulated days, prediction (across) against outcome (up)")
    _share = sl.contributions(_p, _a)
    _size = np.sqrt(np.abs(_share) / np.abs(_share).max())
    _sc = go.Figure()
    for _label, _m, _col in (("helped: both on the same side of their averages", _share >= 0,
                              charts.SERIES[0]),
                             ("hurt: on opposite sides", _share < 0, charts.SERIES[3])):
        _sc.add_trace(go.Scatter(
            x=_p[_m], y=_a[_m], mode="markers", name=_label,
            marker={"size": 5 + 25 * _size[_m], "color": _col, "opacity": 0.7},
            text=[f"{d}: {s:+.1%} of the score" for d, s in zip(saved["dates"][_m], _share[_m],
                                                                 strict=True)],
            hovertemplate="%{text}<br>predicted %{x:+.2f}, actual %{y:+.1f}<extra></extra>"))
    _sc.add_vline(x=_p.mean(), line={"color": charts._MUTED, "dash": "dot"})  # after the traces
    _sc.add_hline(y=_a.mean(), line={"color": charts._MUTED, "dash": "dot"})
    charts._base(_sc, height=480, title=f"Figure 3 · Every VALIDATION day of the best trial: "
                 f"end correlation {np.corrcoef(_p, _a)[0, 1]:+.3f} (dotted: the two averages; "
                 "dot size: the day's share of the score)")
    charts.below(_sc, y=-0.18)  # the default legend sits on the long title
    _sc.update_layout(margin={"b": 110})
    _sc.update_xaxes(title_text="predicted end move (typical moves, × β)")
    _sc.update_yaxes(title_text="actual end move (typical moves)")
    _pc, _ac = _p - _p.mean(), _a - _a.mean()
    _cum = np.cumsum(_pc * _ac) / np.sqrt(np.sum(_pc ** 2) * np.sum(_ac ** 2))
    _moves = sl.real_moves(data["F"], cell, saved["d"])
    _fee, _slip = bt.leg_costs(settings.costs, price=float(np.nanmedian(data["raw"]["close"])),
                               notional=10_000)
    _cost = 2 * (_fee + _slip)  # a round trip, as a share of the position
    _t = sl.traders(_p, _moves, notional=10_000, cost=_cost)
    _money = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                           subplot_titles=["4a · the score, added up day by day (it ends at the "
                                           "end correlation)",
                                           "4b · a trader acting on every prediction: "
                                           "cumulative dollars"])
    _money.add_trace(go.Scatter(x=saved["dates"], y=_cum, mode="lines+markers",
                                name="the score so far", marker={"size": 4},
                                line={"color": charts._INK}), row=1, col=1)
    for _who, _name, _col in (("sized", "sized by the prediction ($10,000 on a typical day)",
                               charts.SERIES[0]),
                              ("direction", "direction only (±$10,000)", charts.SERIES[2])):
        for _kind, _dash in (("gross", "solid"), ("net", "dot")):
            _money.add_trace(go.Scatter(
                x=saved["dates"], y=_t[f"{_who} {_kind}"].cumsum(), mode="lines",
                name=f"{_name}, {'before' if _kind == 'gross' else 'after'} costs",
                line={"color": _col, "dash": _dash}), row=2, col=1)
    for _r in (1, 2):  # after the traces of each row
        _money.add_hline(y=0, line={"color": charts._MUTED, "width": 1}, row=_r, col=1)
    charts._base(_money, height=700, title="Figure 4 · From the score to money")
    charts.below(_money, y=-0.08)  # the default legend sits on the title and subplot titles
    _money.update_layout(margin={"t": 90, "b": 130})
    _money.update_yaxes(title_text="score", row=1, col=1)
    _money.update_yaxes(title_text="$", row=2, col=1)
    _w = _t["sized position"].to_numpy()
    _skill = np.corrcoef(_w, _moves)[0, 1] * _w.std() * _moves.std()
    _lean = _w.mean() * _moves.mean()
    _hit = np.mean(np.sign(_p) == np.sign(_moves))
    _, _off, _L, _H = cell
    mo.vstack([
        _sim, _sc, _money,
        mo.md(f"Trading every VALIDATION day from {ds.clock(_off + _L)} to "
              f"{ds.clock(_off + _L + _H)} (fills at the last trade; costs: the repo's model, "
              f"{1e4 * _cost:.1f} bp a round trip): the trader **sized by the prediction** made "
              f"**${_t['sized gross'].sum():,.0f}** before costs and "
              f"**${_t['sized net'].sum():,.0f}** after, ${_t['sized gross'].mean():+.2f} a day "
              f"before costs. Of that daily average, **${_skill:+.2f}** is what the correlation "
              f"measures: correlation of position and move "
              f"{np.corrcoef(_w, _moves)[0, 1]:+.2f} × typical position ${_w.std():,.0f} × "
              f"typical move {_moves.std():.2%}; the other **${_lean:+.2f}** is the average "
              "position × the average move (leaning one way while the stock drifted that way: "
              "not a forecast of the hour). The **direction-only** trader was right on "
              f"{_hit:.0%} of the days: ${_t['direction gross'].sum():,.0f} before costs, "
              f"${_t['direction net'].sum():,.0f} after."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Nine VALIDATION days up close

    **Figure 5**: the three days that **helped** the score most, three **in between** and the
    three that **hurt** it most. Each panel shows the price in % from the price at the decision
    (0): grey over the minutes the model read, black over the horizon (what happened), the
    prediction dashed (× β: its honest size) and the model's own louder output dotted.
    **Table 2** lists the same days.
    """)
    return


@app.cell(hide_code=True)
def _(cell, charts, data, ds, go, make_subplots, meta, mo, np, pd, saved, sl, table):
    _share = sl.contributions(saved["pe"], saved["ae"])
    _order = np.argsort(_share)
    _mid = _order[np.round(np.array([0.35, 0.5, 0.65]) * (len(_order) - 1)).astype(int)]
    _groups = {"helped most": _order[::-1][:3], "in between": _mid, "hurt most": _order[:3]}
    _paths = sl.day_paths(data["F"], cell, saved["d"], saved["paths"], meta["beta"])
    _picked = [(g, i) for g, idx in _groups.items() for i in idx]
    _fig = make_subplots(rows=3, cols=3, vertical_spacing=0.12, horizontal_spacing=0.06,
                         subplot_titles=[
                             f"{g} · {saved['dates'][i]}<br>predicted "
                             f"{_paths['predicted'][i, -1]:+.2f}%, actual "
                             f"{_paths['actual'][i, -1]:+.2f}% · {_share[i]:+.0%} of the score"
                             for g, i in _picked])
    _x = _paths["minutes"]
    for _j, (_g, _i) in enumerate(_picked):
        _rc = {"row": _j // 3 + 1, "col": _j % 3 + 1}
        for _name, _xs, _ys, _line in (
                ("price, the minutes the model read", _x[_x <= 0], _paths["actual"][_i][_x <= 0],
                 {"color": charts._MUTED}),
                ("price, the horizon (what happened)", _x[_x >= 0],
                 _paths["actual"][_i][_x >= 0], {"color": charts._INK}),
                ("prediction × β", _paths["ahead"], _paths["predicted"][_i],
                 {"color": charts.SERIES[0], "dash": "dash", "width": 2.5}),
                ("the model's raw output", _paths["ahead"], _paths["raw"][_i],
                 {"color": charts.SERIES[3], "dash": "dot"})):
            _fig.add_trace(go.Scatter(x=_xs, y=_ys, mode="lines", name=_name, line=_line,
                                      legendgroup=_name, showlegend=_j == 0), **_rc)
        _fig.add_vline(x=0, line={"color": charts._MUTED, "width": 1}, **_rc)  # after the traces
    _, _off, _L, _H = cell
    charts._base(_fig, height=1080, title=f"Figure 5 · Nine VALIDATION days: % from the price at "
                 f"{ds.clock(_off + _L)}, against minutes from the decision")
    charts.below(_fig, y=-0.05)  # the default legend sits on the title and the subplot titles
    _fig.update_layout(margin={"t": 120, "b": 100})
    _fig.update_annotations(font_size=11)
    _days = pd.DataFrame([{
        "group": g, "date": saved["dates"][i],
        "predicted (typical moves, × β)": saved["pe"][i],
        "actual (typical moves)": saved["ae"][i],
        "predicted %": _paths["predicted"][i, -1], "actual %": _paths["actual"][i, -1],
        "right direction (up / down)": bool(np.sign(saved["pe"][i]) == np.sign(saved["ae"][i])),
        "share of the score": _share[i]} for g, i in _picked])
    mo.vstack([_fig, mo.md("### Table 2 · The nine days"), table(_days.round(3), rows=9)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Is the best trial real?

    **Table 3** asks whether a few days carry the score: a correlation is pulled by big moves,
    so one crash the model happened to call can make a whole half-year look good (Figure 4a
    shows where the score was made). The rank correlation weighs every day alike.
    """)
    return


@app.cell(hide_code=True)
def _(meta, mo, saved, sl, table):
    _rob = sl.robustness(saved["pe"], saved["ae"])
    _months = ", ".join(f"{m[2:]} {v:+.2f}" for m, v in meta["month_corr"].items())
    mo.vstack([
        mo.md(f"Month by month: {_months}."),
        mo.md("### Table 3 · What carries the best trial's score"),
        table(_rob.assign(value=_rob["value"].round(3))),
    ])
    return


@app.cell(hide_code=True)
def _(SEEDS, mo):
    mo.md(f"""
    ### Other seeds

    The study trained every trial with seed 0. **retrain** trains the best trial's settings
    with {SEEDS} seeds (0 to {SEEDS - 1}): if the score was a lucky seed, the others fall back.
    The **averaged model** (the mean of the {SEEDS} seeds' predictions) is steadier than any
    one seed. These fits are kept: the comparisons in section 5 reuse them as their baseline.
    """)
    return


@app.cell
def _(mo):
    run_seeds = mo.ui.run_button(label="retrain the best trial with every seed")
    mo.hstack([run_seeds], justify="start")
    return (run_seeds,)


@app.cell(hide_code=True)
def _(
    DRAWS,
    EPOCHS,
    PATIENCE,
    SEEDS,
    base_feats,
    cell,
    data,
    fits,
    knobs,
    meta,
    mo,
    prev,
    run_seeds,
    sl,
    table,
):
    mo.stop(not run_seeds.value, mo.md("_Press the button (a few seconds per seed)._"))
    with mo.status.progress_bar(total=SEEDS, title="Retraining the best trial") as _bar:
        _r = sl.fit_seeds(sl.Inputs(), data, cell, base_feats, knobs, prev_days=prev,
                          seeds=range(SEEDS), epochs=EPOCHS, patience=PATIENCE, draws=DRAWS,
                          cache=fits, on_fit=lambda: _bar.update(1))
    _s, _e = _r["seeds"], _r["ensemble"]
    mo.vstack([
        mo.md(f"Over {SEEDS} seeds the best trial's settings score **{_s['end_corr'].mean():+.3f}"
              f"** ± {_s['end_corr'].std(ddof=1):.3f} (the study: {meta['end_corr']:+.3f} with "
              f"seed 0). The averaged model: end correlation **{_e['end_corr']:+.3f}**, luck z "
              f"**{_e['luck_z']:+.2f}**, hit rate {_e['hit_rate']:.0%}."),
        mo.md("### Table 4 · One row per seed"), table(_s.round(3)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Which inputs does the model use?

    Measured on the saved best model (or, with new inputs picked below, on the best trial's
    settings trained again with them, seed 0), in two ways that both ask what happens when an
    input is **absent**: read from another day of the stock, so it keeps its usual size and only
    its link to that day's outcome is cut. VALIDATION days only.

    - **Permutation importance** (Table 5, Figure 6; about a minute): the score again with one
      input scrambled, ten times each. A drop well above its sd, with most scrambles hurting:
      the model leans on it, and it helps on days the model never learned from. Below 0: the
      input hurts. Market + sector are also scrambled together (they overlap)
    - **SHAP** (Table 6, Figures 7 and 8; about 1.5 seconds per day): how far each input moved
      each prediction, signed, whether or not that helped. High here and near 0 in Table 5: the
      model moves with the input but gains nothing from it
    - if the model's score is within luck, every drop is small and noisy: the tables then say
      what this one model leans on, not what predicts
    """)
    return


@app.cell
def _(NEW, mo):
    explain_new = mo.ui.multiselect(options=list(NEW), value=[],
                                    label="explain with these new inputs added (section 5)")
    run_importance = mo.ui.run_button(label="measure the inputs")
    mo.hstack([explain_new, run_importance], justify="start", align="end")
    return explain_new, run_importance


@app.cell(hide_code=True)
def _(
    EPOCHS,
    PATIENCE,
    base_feats,
    cell,
    data,
    explain_new,
    inputs,
    knobs,
    mo,
    prev,
    run_importance,
    saved_model,
):
    mo.stop(not run_importance.value, mo.md("_Press the button to measure the inputs._"))
    if explain_new.value:
        _feats = base_feats + list(explain_new.value)
        with mo.status.spinner(title="Training the best trial with the new inputs ..."):
            _model, _ = inputs.train(data["F"], cell, data["fit"], data["held"], _feats,
                                     {**knobs, "patience": PATIENCE}, epochs=EPOCHS,
                                     prev_days=prev)
        explained = {"model": _model, "feats": _feats,
                     "label": "the best trial + " + ", ".join(explain_new.value) + " (seed 0)"}
    else:
        explained = {"model": saved_model, "feats": base_feats, "label": "the saved best model"}
    return (explained,)


@app.cell(hide_code=True)
def _(cell, charts, data, explained, go, imp, inputs, mo, prev, table):
    with mo.status.spinner(title="Scrambling every input, ten times each ..."):
        _base, perm_scores = imp.importance(inputs, explained["model"], data["F"], cell,
                                            data["val"], explained["feats"], prev_days=prev,
                                            repeats=10)
    _show = perm_scores.round(3).assign(hurt=perm_scores["hurt"].map("{:.0%}".format)).rename(
        columns={"drop": "drop in end corr", "sd": "sd over scrambles",
                 "hurt": "scrambles that hurt", "scrambled corr": "end corr when scrambled"})
    _fig = go.Figure(go.Bar(
        x=perm_scores["input"], y=perm_scores["drop"], showlegend=False,
        error_y={"type": "data", "array": perm_scores["sd"], "visible": True}))
    _fig.add_hline(y=0, line={"color": charts._MUTED, "width": 1})  # after the trace
    charts._base(_fig, height=420, title="Figure 6 · How far the end correlation falls when one "
                 "input is scrambled (± sd over the scrambles)")
    mo.vstack([
        mo.md(f"**{explained['label']}** scores **{_base:+.3f}** on VALIDATION."),
        mo.md("### Table 5 · What the end correlation loses when an input is scrambled"),
        table(_show, rows=15), _fig,
    ])
    return (perm_scores,)


@app.cell
def _(mo):
    shap_explain = mo.ui.number(start=10, stop=120, step=10, value=40, label="days explained")
    run_shap = mo.ui.run_button(label="compute SHAP values")
    mo.hstack([shap_explain, run_shap], justify="start", align="end")
    return run_shap, shap_explain


@app.cell(hide_code=True)
def _(
    cell,
    charts,
    data,
    explained,
    go,
    imp,
    inputs,
    make_subplots,
    mo,
    np,
    pd,
    perm_scores,
    prev,
    run_shap,
    shap_explain,
    table,
):
    mo.stop(not run_shap.value, mo.md("_Press the button to compute SHAP values._"))
    with mo.status.spinner(title="Shapley values: every input present or absent, many orders ..."):
        _r = imp.shap_values(inputs, explained["model"], data["F"], cell, data["val"],
                             explained["feats"], prev_days=prev, explain=int(shap_explain.value))
    _v, _names = _r["values"], _r["names"]
    _gap = float(np.abs(_r["base"] + _v.sum(axis=1) - _r["pred"]).max())
    _shap = pd.DataFrame({
        "input": _names, "mean |SHAP|": np.abs(_v).mean(axis=0), "mean SHAP": _v.mean(axis=0),
        "sd": _v.std(axis=0), "pushes up": (_v > 0).mean(axis=0)})
    _shap = _shap.merge(perm_scores[["input", "drop"]].rename(
        columns={"drop": "drop in end corr (Table 5)"}), on="input", how="left")
    _shap = _shap.sort_values("mean |SHAP|", ascending=False).reset_index(drop=True)
    _strip = go.Figure()
    for _name in _shap["input"][::-1]:  # the biggest on top of a horizontal chart
        _strip.add_trace(go.Box(
            x=_v[:, _names.index(_name)], name=_name, orientation="h", boxpoints="all",
            jitter=0.4, pointpos=0, marker={"size": 4}, showlegend=False, text=list(_r["date"]),
            hovertemplate="%{x:+.3f} typical moves<br>%{text}<extra></extra>"))
    _strip.add_vline(x=0, line={"color": charts._MUTED, "width": 1})  # after the traces
    charts._base(_strip, height=120 + 46 * len(_names), title="Figure 7 · SHAP value of every "
                 "input on every explained day (typical moves of the predicted end move)")
    _out = [
        mo.md(f"**{len(_v)}** days explained ({explained['label']}). The values add up to the "
              f"model's own predictions to within **{_gap:.1e}**."),
        mo.md("### Table 6 · What each input does to the predictions"),
        table(_shap.round(4).assign(**{"pushes up": _shap["pushes up"].map("{:.0%}".format)}),
              rows=15), _strip]
    _scalars = list(_r["x"].columns)
    if _scalars:
        _dep = make_subplots(rows=1, cols=len(_scalars), subplot_titles=_scalars,
                             shared_yaxes=True)
        for _i, _f in enumerate(_scalars, start=1):
            _dep.add_trace(go.Scatter(
                x=_r["x"][_f], y=_v[:, _names.index(_f)], mode="markers", showlegend=False,
                marker={"size": 5}, text=list(_r["date"]),
                hovertemplate=f"{_f} %{{x:+.2f}}<br>SHAP %{{y:+.3f}}<br>%{{text}}"
                              "<extra></extra>"), row=1, col=_i)
            _dep.add_hline(y=0, line={"color": charts._MUTED, "width": 1}, row=1, col=_i)
        charts._base(_dep, height=380, title="Figure 8 · A per-day input's value (typical units) "
                     "against its SHAP value: a rising cloud = a high value pushes the "
                     "prediction up")
        _out.append(_dep)
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · New inputs

    **How to add one**: write a function of `raw` in the next cell and put
    `@new_input("day" | "bar", what=..., why=...)` above it. A **day** input returns one value
    per session (a static input), a **bar** input one value per session and minute (a past
    covariate, read over the model's input minutes). `raw` holds the stock's sessions × 390
    minutes: `open`, `high`, `low`, `close` (NaN without a trade), `volume`, `y` (the minute
    moves in typical moves: the model's own path), `market` / `sector` (the baskets' minute
    moves); per session `day_close` (its last trade); and `decision` (the decision minute).

    - **only what is known at the decision**: a day input may use today's minutes before
      `raw["decision"]` and earlier sessions; a bar input at minute t only minutes up to t. The
      look-ahead check (Table 7) recomputes every input with everything after the decision
      hidden and flags any change
    - **in typical units**: values of about ±1 on an ordinary day; `sl.typical(x)` scales a
      per-minute array by the same clock minute's usual size, `sl.causal_z(x)` a per-session
      series by its own previous 60 sessions (both read earlier sessions only). Day inputs are
      clipped at ±4, bar inputs at ±10; NaN counts as 0

    Then the two checks: **Table 7** (seconds, no training) and **Table 8** (training, several
    seeds). The seven inputs below are ideas from the literature, as a start.
    """)
    return


@app.cell
def _(np, pd, sl):
    NEW, new_input = sl.registry()

    @new_input("day", what="the stock's move from 09:30 to the decision (÷ √minutes)",
               why="Gao, Han, Li & Zhou (2018): the morning's move predicts the last half hour")
    def morning_move(raw):
        D = raw["decision"]
        return np.nansum(raw["y"][:, :D], axis=1) / np.sqrt(D)

    @new_input("day", what="the market basket's move from 09:30 to the decision (÷ √minutes)",
               why="Gao et al. (2018): the effect is strongest for the MARKET's morning move")
    def market_morning(raw):
        D = raw["decision"]
        return np.nansum(raw["market"][:, :D], axis=1) / np.sqrt(D)

    @new_input("day", what="where the last price sits in the day's range so far (-1 at the "
               "low, +1 at the high)",
               why="George & Hwang (2004): prices anchor on recent highs and lows")
    def range_position(raw):
        D = raw["decision"]
        hi = np.nanmax(raw["high"][:, :D], axis=1)
        lo = np.nanmin(raw["low"][:, :D], axis=1)
        last = pd.DataFrame(raw["close"][:, :D]).ffill(axis=1).to_numpy()[:, -1]
        return np.where(hi > lo, (2 * last - hi - lo) / (hi - lo), 0.0)

    @new_input("day", what="how volatile the morning was: log of the root-mean-square minute "
               "move (typical moves; 0 = an ordinary morning)",
               why="Gao et al. (2018): intraday momentum is stronger on volatile days")
    def morning_volatility(raw):
        D = raw["decision"]
        return np.log(np.sqrt(np.nanmean(raw["y"][:, :D] ** 2, axis=1)))

    @new_input("day", what="the volume traded so far today against the same part of the "
               "previous 20 sessions (log)",
               why="Gervais, Kaniel & Mingelgrin (2001): unusual volume carries information")
    def volume_so_far(raw):
        D = raw["decision"]
        v = pd.Series(np.nansum(raw["volume"][:, :D], axis=1))
        usual = v.rolling(20, min_periods=10).mean().shift(1)
        return np.log((1 + v) / (1 + usual)).to_numpy()

    @new_input("day", what="the last 20 sessions' close-to-close move, in typical months",
               why="Jegadeesh (1990): the past month's return reverses")
    def month_move(raw):
        c = pd.Series(np.log(raw["day_close"]))
        scale = c.diff().shift(1).rolling(60, min_periods=20).std() * np.sqrt(20)
        return ((c.shift(1) - c.shift(21)) / scale).to_numpy()

    @new_input("bar", what="each minute's distance from the day's volume-weighted average "
               "price so far (log, in typical sizes)",
               why="practitioners' reference price (Berkowitz, Logue & Noser 1988): a "
               "stretched price may pull back to it")
    def vwap_gap(raw):
        pv = np.nancumsum(raw["close"] * raw["volume"], axis=1)
        vwap = pv / np.nancumsum(np.where(np.isfinite(raw["close"]), raw["volume"], 0), axis=1)
        last = pd.DataFrame(raw["close"]).ffill(axis=1).to_numpy()
        return sl.typical(np.log(last / vwap))

    return (NEW,)


@app.cell(hide_code=True)
def _(DRAWS, NEW, base_feats, cell, data, lab, np, pd, prev, saved, sl):
    _raw = data["raw"]
    new_values = sl.compute(NEW, _raw)
    inputs = sl.Inputs(new_values)
    _F, _D = data["F"], cell[1] + cell[2]
    _checked = np.unique(np.r_[data["fit"], data["val"]][np.linspace(
        0, len(data["fit"]) + len(data["val"]) - 1, 8).round().astype(int)])
    _ok = lab.valid(_F, cell)[:, 0]
    _parts = {"TRAIN": np.array([d for d in data["fit"] if _ok[d]]),
              "VALIDATION": saved["d"]}
    _ends = {p: lab.end_moves(_F, cell, d, np.zeros_like(d)) for p, d in _parts.items()}
    _miss = saved["ae"] - saved["pe"]  # what the saved model gets wrong

    def _ic(x, a, d):
        """The input's correlation with ``a`` on days ``d``, and its luck z."""
        s, _, _ = lab.grade(np.nan_to_num(x), a, d, np.zeros_like(d), _F, beta=1.0,
                            draws=DRAWS)
        return s["end_corr"], s["luck_z"]

    _rows, decision_values = [], {}
    _current = {"gap": lab.day_inputs(_F, cell, ["gap"])[:, 0, 0],
                **{f"prev ({p} days)": _F["prev_move"][p][:, 0] for p in (1, 2, 5)},
                "same_hour": lab.day_inputs(_F, cell, ["same_hour"])[:, 0, 0]}
    for _name, _x in [*((n, (v, "day", None)) for n, v in _current.items()),
                      *((n, (v["values"], v["level"], v["fn"])) for n, v in new_values.items())]:
        _vals, _level, _fn = _x
        _at = {p: sl.at_decision(_vals, _level, d, _D) for p, d in _parts.items()}
        decision_values[_name] = _at
        _tr, _va = (_ic(_at[p], _ends[p], _parts[p]) for p in _parts)
        _mi = _ic(_at["VALIDATION"], _miss, saved["d"])
        _leak = None if _fn is None else sl.leak_check(_fn, _level, _raw, _checked)
        _rows.append({
            "input": _name, "kind": "current" if _fn is None else "new", "level": _level,
            "in the best model": (_name == f"prev ({prev} days)" and "prev" in base_feats)
            if _name.startswith("prev") else _name in base_feats,
            "look-ahead": "built in" if _leak is None else "none" if _leak == 0 else
            f"LEAK ({_leak:.2g})",
            "has a value": float(np.isfinite(_at["TRAIN"]).mean()),
            "corr TRAIN": _tr[0], "z TRAIN": _tr[1], "corr VALIDATION": _va[0],
            "z VALIDATION": _va[1], "corr with the model's misses": _mi[0],
            "z misses": _mi[1], "what": NEW[_name]["what"] if _name in NEW else ""})
    information = pd.DataFrame(_rows)
    return decision_values, information, inputs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Does a new input carry information? (no training)

    **Table 7**: every new input (and the current per-day ones, for scale) as the model sees it
    at the decision: a day input's value, a bar input's last minute. Its correlation with the
    end move on TRAIN and on VALIDATION, each with a luck z (random up/down outcomes), and with
    the saved model's **misses** (actual minus predicted end move, VALIDATION): an input that
    lines up with the misses carries something the model does not use yet.

    - a z beyond ±2 on TRAIN AND the same sign on VALIDATION is a candidate worth training
    - a single correlation says nothing about combinations: a weak input can still help the
      model through another one (Gao et al.: momentum on volatile days). Table 8 tests that
    - the model can read a bar input's whole input window, not just the last minute shown here
    """)
    return


@app.cell(hide_code=True)
def _(information, mo, table):
    _bad = information.loc[information["look-ahead"].str.startswith("LEAK"), "input"].tolist()
    mo.vstack([
        mo.md("**Look-ahead found in: " + ", ".join(_bad) + "**: fix these before training."
              if _bad else "No look-ahead in any new input."),
        mo.md("### Table 7 · Every input against the end move, as seen at the decision"),
        table(information.round(3), rows=20),
    ])
    return


@app.cell
def _(decision_values, mo):
    ic_pick = mo.ui.dropdown(list(decision_values), value=list(decision_values)[-1],
                             label="input")
    mo.hstack([ic_pick], justify="start")
    return (ic_pick,)


@app.cell(hide_code=True)
def _(cell, charts, data, decision_values, go, ic_pick, lab, mo, np):
    _F = data["F"]
    _ok = lab.valid(_F, cell)[:, 0]
    _fig = go.Figure()
    for _i, (_part, _days) in enumerate((("TRAIN", [d for d in data["fit"] if _ok[d]]),
                                         ("VALIDATION", [d for d in data["val"] if _ok[d]]))):
        _days = np.array(_days)
        _fig.add_trace(go.Scatter(
            x=decision_values[ic_pick.value][_part],
            y=lab.end_moves(_F, cell, _days, np.zeros_like(_days)), mode="markers", name=_part,
            marker={"size": 6, "color": charts.SERIES[_i], "opacity": 0.7},
            text=[_F["dates"][d] for d in _days],
            hovertemplate="%{text}<br>input %{x:+.2f}<br>end move %{y:+.1f}<extra></extra>"))
    _fig.add_hline(y=0, line={"color": charts._MUTED, "width": 1})  # after the traces
    charts._base(_fig, height=420, title=f"Figure 9 · {ic_pick.value} at the decision against "
                 "the end move (typical moves), one dot per day")
    _fig.update_xaxes(title_text=ic_pick.value)
    mo.vstack([_fig])
    return


@app.cell(hide_code=True)
def _(SEEDS, mo):
    mo.md(f"""
    ### Does a new input make the model better? (training)

    **compare** trains the best trial's settings ({SEEDS} seeds each) as they are (the
    baseline: the same fits as in section 3), with each picked new input added, with all of
    them together, and, if ticked, with each current input dropped. **Table 8** then compares
    every arm with the baseline on the same days and the same seeds:

    - **change (paired)** and its **t**: the seeds' mean change in end correlation; |t| > 2 is a
      change bigger than the seeds move it, |t| < 1 is noise
    - **averaged model**: the mean of the {SEEDS} seeds' predictions, scored with its luck z
    - **1st / 2nd half**: the averaged model's change on each half of VALIDATION. A real gain
      holds in both halves; a gain from one stretch of days (like a crash) does not
    - the knobs stay the best trial's, tuned without the new input: an input the knobs do not
      suit can look worse than it is. Every run is appended to the stock's experiment log
      (Table 9), so ideas tried on earlier days stay comparable. Picking the best arm out of
      many is selection again: confirm a winner on TEST once, at the very end
    """)
    return


@app.cell
def _(NEW, SEEDS, base_feats, mo):
    arm_new = mo.ui.multiselect(options=list(NEW), value=list(NEW),
                                label="add, one at a time")
    arm_all = mo.ui.checkbox(value=True, label="add all of them together")
    arm_drop = mo.ui.checkbox(value=False, label="drop each current input, one at a time")
    run_compare = mo.ui.run_button(label="compare")
    _n = 1 + len(NEW) + 1 + len(base_feats)
    mo.vstack([mo.hstack([arm_new, arm_all, arm_drop], justify="start", align="center"),
               mo.md(f"_At most {_n} arms × {SEEDS} seeds; about 4 seconds per fit (fits "
                     "already made are reused)._"), run_compare])
    return arm_all, arm_drop, arm_new, run_compare


@app.cell(hide_code=True)
def _(
    DRAWS,
    EPOCHS,
    PATIENCE,
    SEEDS,
    STOCK,
    arm_all,
    arm_drop,
    arm_new,
    base_feats,
    cell,
    charts,
    data,
    fits,
    go,
    inputs,
    knobs,
    meta,
    mo,
    prev,
    run_compare,
    sl,
    table,
):
    mo.stop(not run_compare.value, mo.md("_Pick the arms and press **compare**._"))
    _arms = {"baseline (the best trial)": base_feats}
    _arms |= {f"+ {n}": base_feats + [n] for n in arm_new.value}
    if arm_all.value and len(arm_new.value) > 1:
        _arms["+ all picked new inputs"] = base_feats + list(arm_new.value)
    if arm_drop.value:
        _arms |= {f"− {f}": [x for x in base_feats if x != f] for f in base_feats}
    with mo.status.progress_bar(total=len(_arms) * SEEDS, title="Comparing", show_eta=True,
                                completion_title="Compared") as _bar:
        _res = {}
        for _arm, _feats in _arms.items():
            _bar.update(0, subtitle=_arm)
            _res[_arm] = sl.fit_seeds(inputs, data, cell, _feats, knobs, prev_days=prev,
                                      seeds=range(SEEDS), epochs=EPOCHS, patience=PATIENCE,
                                      draws=DRAWS, cache=fits, on_fit=lambda: _bar.update(1))
    _cmp = sl.compare(_res, "baseline (the best trial)")
    _log = sl.log(STOCK, _cmp.assign(
        inputs=[", ".join(_arms[a]) for a in _cmp["arm"]],
        version=[inputs.tag(_arms[a]) for a in _cmp["arm"]]),  # changes with an input's code
        batch=meta["batch"], trial=meta["trial"])
    _fig = go.Figure()
    for _arm, _r in _res.items():
        _fig.add_trace(go.Scatter(x=[_arm] * SEEDS, y=_r["seeds"]["end_corr"], mode="markers",
                                  marker={"size": 8, "color": charts.SERIES[0], "opacity": 0.6},
                                  showlegend=False, hovertemplate="seed end corr %{y:+.3f}"))
        _fig.add_trace(go.Scatter(x=[_arm], y=[_r["ensemble"]["end_corr"]], mode="markers",
                                  marker={"size": 13, "symbol": "diamond",
                                          "color": charts.SERIES[3]}, showlegend=False,
                                  hovertemplate="averaged model %{y:+.3f}"))
    _fig.add_hline(y=_cmp["mean end corr"].iloc[0],
                   line={"color": charts._MUTED, "dash": "dash"})  # after the traces
    charts._base(_fig, height=440, title="Figure 10 · Every arm's end correlation: one dot per "
                 "seed, the diamond = the averaged model (dashed: the baseline's seed mean)")
    mo.vstack([
        mo.md("### Table 8 · Every arm against the baseline (same days, same seeds)"),
        table(_cmp.round(3), rows=25), _fig,
        mo.md(f"### Table 9 · {STOCK}'s experiment log, newest first "
              f"(`var/tsmixer_lab/stock/{STOCK}-experiments.csv`)"),
        table(_log.round(3), rows=10),
    ])
    return


if __name__ == "__main__":
    app.run()
