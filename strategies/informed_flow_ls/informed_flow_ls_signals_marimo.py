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
    # informed_flow_ls · data, signals and one book

    `informed_flow` owned the stocks short sellers avoid. This version also **sells short** the
    stocks they crowd into, adds what **option traders** say (the put skew getting dearer), and
    pays a **borrow fee** on every short. This notebook looks at the ingredients; the companion
    `informed_flow_ls_walkforward_marimo.py` decides whether the strategy works.

    1. **The universe and the data**: the point-in-time top 500 (re-formed every 6 months on the
       6 months before), and how much of it each input covers, month by month: days-to-cover,
       share counts, option chains; how crowded the names are and what borrowing them costs
    2. **The two signals, point in time**: does each one rank the next month's market-adjusted
       returns, and at BOTH ends of the ranking — the long end and the short end separately?
    3. **One book with the config's values**: the long/short book from July 2020 on, against
       buying and holding everything: its returns, exposure, beta, borrow paid and exits

    Everything up to 2026-10-16 is read here (informed_flow's eleven versions read all of it
    already); the final check is paper trading from 2026-10-20 (walk-forward notebook, §8).
    Section 1's load takes a few minutes the first time.
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
def _(mo):
    reload_helpers = mo.ui.run_button(label="reload research.py (after it changed)")
    mo.hstack([reload_helpers], justify="start")
    return (reload_helpers,)


@app.cell
def _(reload_helpers):
    import importlib.util

    import numpy as np
    import pandas as pd
    from plotly.subplots import make_subplots

    from trading_simple import backtest as bt
    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine
    from trading_simple.plots import SERIES
    from trading_simple.plots import _base as layout
    from trading_simple.plots import _scatter as scatter

    _ = reload_helpers.value
    _spec = importlib.util.spec_from_file_location(
        "informed_flow_ls_research", strategies.STRATEGIES_DIR / "informed_flow_ls" / "research.py")
    rs = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rs)
    strat = strategies.load("informed_flow_ls")
    settings = load_settings()
    engine = make_engine(settings)
    mod = strat.signals.__globals__
    return (SERIES, bt, engine, make_subplots, mod, np, pd,
            layout, rs, scatter, settings, strat)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · The universe and the data

    **The universe** is informed_flow's rolling one (v36-v41): every six months from July 2020
    the 500 equities with the highest median dollar volume over the six months before (price at
    least $5, at least $50M a day). A name counts only on the dates it was a member — the
    strategy reads that as its `member` feature — so nothing is chosen by how liquid a stock
    LATER became (v36's forward-selection defect hid the short-interest signal).

    **Inputs:** FINRA days-to-cover (`short_interest`, twice a month, masked after 45 days),
    SEC share counts (`fund_shares`: the borrow tier and the foreign-filer screen), and the put
    skew from the end-of-day option chains (`opt_skew`, October 2025 on; `providers/marketdata`).
    """)
    return


@app.cell
def _(mo):
    load_button = mo.ui.run_button(label="load the point-in-time panel (2018 → 2026-10-16)")
    mo.hstack([load_button], justify="start")
    return (load_button,)


@app.cell
def _(engine, load_button, mo, pd, rs, settings, strat):
    mo.stop(not load_button.value, mo.md("_Click the button to load the panel._"))
    with mo.status.spinner("memberships and prices (a few minutes the first time) ..."):
        LAST = "2026-10-16"
        history = rs.pit_history(engine, LAST)
        names = sorted(set().union(*history.values()))
        panel = rs.load(engine, settings, strat, LAST, names, history)
    member = panel.features["member"].astype(bool)
    scored = panel.close.index >= pd.Timestamp(rs.SCORE_FROM, tz="UTC")
    mo.md(f"**{len(history)} formations, {len(names)} names over all of them, "
          f"{len(panel.close)} sessions** ({panel.close.index[0].date()} → "
          f"{panel.close.index[-1].date()}).")
    return LAST, member, panel, scored


@app.cell
def _(member, mod, panel, rs, scored, strat):
    parts = mod["compose"](panel, **rs.compose_params(mod, strat.params))
    month = panel.close.index.to_period("M")

    def share(frame):
        """Per month: the share of member name-days where `frame` holds."""
        return frame.where(member).mean(axis=1)[scored].groupby(month[scored]).mean()

    coverage = {
        "members": member[scored].sum(axis=1).groupby(month[scored]).mean().round(0),
        "days-to-cover": share(panel.features["short_interest"].notna()),
        "share count": share(panel.features["fund_shares"].notna()),
        "option chain": share(panel.features["opt_skew"].notna()),
        "eligible": share(parts["eligible"]),
        "shortable (of eligible)": share(parts["shortable"]) / share(parts["eligible"]),
    }
    si = parts["si_share"].where(member)[scored]
    _p = strat.params

    def above(level):
        """Per month: the share of member names with a known ratio above ``level``."""
        return (si > level).where(si.notna()).mean(axis=1).groupby(month[scored]).mean()

    crowd = {
        "short interest / shares, median": si.median(axis=1).groupby(month[scored]).mean(),
        f"> {_p['si_mid']:.0%} ({_p['borrow_mid']:.2%}/yr borrow)": above(_p["si_mid"]),
        f"> {_p['si_high']:.0%} ({_p['borrow_high']:.2%}/yr)": above(_p["si_high"]),
        f"> {_p['short_max_si']:.0%} (never shorted)": above(_p["short_max_si"]),
        f"> {_p['si_plausible_max']:.0%} (implausible: unknown)": (
            parts["si_share"].isna() & parts["eligible"]).where(member)[scored].mean(axis=1)
        .groupby(month[scored]).mean(),
    }
    return coverage, crowd, parts


@app.cell
def _(coverage, crowd, mo, pd, table):
    _cov = pd.DataFrame(coverage)
    _cov.index = _cov.index.astype(str)
    _cov.index.name = "month"
    _crowd = pd.DataFrame(crowd)
    _crowd.index = _cov.index
    mo.vstack([
        mo.md("**Table 1 · Coverage per month** (share of member name-days). Eligible = a price "
              "of at least $5, a days-to-cover and a share count: names without an SEC share "
              "count (most foreign filers) are left out on purpose."),
        table(_cov.round(3), rows=12),
        mo.md("**Table 2 · How crowded the shortable universe is**: FINRA's shares sold short "
              "as a share of SEC's shares outstanding (each as of its own public date: the "
              "`si_shares` feature `research.load` attaches), and the share of eligible names "
              "in each borrow tier. Where IBKR's own fee is known it replaces the tier. The "
              "last column counts eligible names whose ratio is above the plausible maximum "
              "(after a split, FINRA's new count over SEC's old one): those are not shorted."),
        table(_crowd.round(3), rows=12),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · The two signals, point in time

    For each signal and for the composite: the daily **rank correlation (IC)** between the
    ranking and the next 21 / 63 sessions' returns from the day after (a trade fills at the next
    open) **minus trailing beta × the market's** — so being in the market earns nothing here and
    only the choice of stocks counts — over member names only. Newey-West t (lag 2 × horizon).

    **Both ends separately** (Table 4): a long/short book earns the top fifth's return minus the
    bottom fifth's. The old strategy only ever used the top. If all the edge sits at one end,
    the other end only adds costs and borrow.

    The skew sleeve exists from October 2025 only: its rows cover one year.
    """)
    return


@app.cell
def _(member, panel, parts, pd, rs, scored):
    def sleeve_tables(h):
        rel = rs.residual_forward(panel.close, h).where(member)
        rel = rel.sub(rel.mean(axis=1), axis=0)
        frames = {"short interest": parts["ranks"]["si"], "put skew change": parts["ranks"]["skew"],
                  "composite": parts["score"]}
        summary, ends = {}, {}
        for label, frame in frames.items():
            f = frame.where(member)[scored]
            r = rel[scored]
            ic = rs.rank_ic(f, r)
            if not len(ic):
                continue
            by_year = ic.groupby(ic.index.year).mean()
            summary[label] = {"from": str(ic.index[0].date()), "days": len(ic),
                              "IC": ic.mean(), "t": rs.nw_t(ic, 2 * h),
                              **{f"IC {y}": v for y, v in by_year.items()}}
            pct = f.rank(axis=1, pct=True)
            top = r.where(pct > 0.8).mean(axis=1)
            bottom = r.where(pct <= 0.2).mean(axis=1)
            ends[label] = {"top fifth (long end)": top.mean(), "t top": rs.nw_t(top, 2 * h),
                           "bottom fifth (short end)": bottom.mean(),
                           "t bottom": rs.nw_t(bottom, 2 * h),
                           "spread": (top - bottom).mean(),
                           "t spread": rs.nw_t(top - bottom, 2 * h)}
        return pd.DataFrame(summary).T, pd.DataFrame(ends).T

    ic_tables = {h: sleeve_tables(h) for h in (21, 63)}
    return (ic_tables,)


@app.cell
def _(ic_tables, mo, table):
    _blocks = []
    for _h, (_summary, _ends) in ic_tables.items():
        _blocks += [
            mo.md(f"**Table 3 · Rank IC, {_h} sessions ahead** (market-adjusted, member names; "
                  "a positive IC = the ranking points the right way)"),
            table(_summary.round(4), rows=5),
            mo.md(f"**Table 4 · Both ends, {_h} sessions ahead**: the average market-adjusted "
                  "return of the top and the bottom fifth against the average name "
                  "(fractions: 0.01 = 1%). For the short end a NEGATIVE number is what the "
                  "short side earns."),
            table(_ends.round(4), rows=5),
        ]
    mo.vstack(_blocks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · One book with the config's values

    The long/short book as `config.yaml` ships it (no tuning), run warm on the point-in-time
    universe from July 2020: 15 longs and 15 shorts, each about 3.3% of the book, borrow paid on
    every short, fills at the next open. **Not an out-of-sample test** — the config's values
    were set after informed_flow's eleven versions read all of this — but it shows what the
    book DOES. The baseline is buying and holding every member name in equal weight.
    """)
    return


@app.cell
def _(mo):
    book_button = mo.ui.run_button(label="run the book (about a minute)")
    mo.hstack([book_button], justify="start")
    return (book_button,)


@app.cell
def _(book_button, bt, mo, np, panel, pd, rs, settings, strat):
    mo.stop(not book_button.value, mo.md("_Click the button to run the book._"))
    with mo.status.spinner("running the long/short book and its long side alone ..."):
        one = rs.book(strat, panel, {}, settings, rs.SCORE_FROM)
        longs_only = rs.book(strat, panel, {"sides": "long"}, settings, rs.SCORE_FROM)
    pf = one["pf"]
    value = pf.value()
    assets = pf.asset_value(group_by=False)
    exposure = pd.DataFrame({"long": assets.clip(lower=0).sum(axis=1) / value,
                             "short": assets.clip(upper=0).sum(axis=1) / value})
    exposure["net"] = exposure.sum(axis=1)
    rows = {}
    for label, b in (("long/short book", one), ("long side alone (half the book)", longs_only)):
        r, be = b["returns"], b["bench"]
        years = bt._years(r.index)
        cagr, bcagr, beta, alpha = (float(x[0]) for x in rs.alpha_of(r.to_numpy(), be.to_numpy(),
                                                                       years))
        curve = (1 + r).cumprod()
        rows[label] = {"CAGR": cagr, "beta": beta, "alpha": alpha,
                       "volatility": float(r.std() * np.sqrt(len(r) / years)),
                       "max drawdown": float((curve / curve.cummax() - 1).min()),
                       "trades / year": b["stats"]["trades_per_year"],
                       "buy & hold CAGR": bcagr}
    stats = pd.DataFrame(rows).T
    by_year = pd.DataFrame({k: (1 + b["returns"]).groupby(b["returns"].index.year).prod() - 1
                            for k, b in (("book", one), ("long side", longs_only))})
    by_year["buy & hold"] = (1 + one["bench"]).groupby(one["bench"].index.year).prod() - 1
    by_year.index.name = "year"
    return by_year, exposure, longs_only, one, stats


@app.cell
def _(SERIES, exposure, layout, make_subplots, mo, one, scatter):
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=["Growth of $1", "Exposure (share of the book)"])
    for _i, (_name, _r) in enumerate((("long/short book", one["returns"]),
                                      ("buy & hold (equal weight)", one["bench"]))):
        _fig.add_trace(scatter((1 + _r).cumprod(), mode="lines", name=_name,
                                line={"color": SERIES[_i], "width": 2}), row=1, col=1)
    for _i, _col in enumerate(("long", "short", "net")):
        _fig.add_trace(scatter(exposure[_col], mode="lines", name=_col,
                                line={"color": SERIES[2 + _i], "width": 1.5}), row=2, col=1)
    _fig.update_xaxes(type="date")
    _fig.update_yaxes(tickformat=".0%", row=2, col=1)
    mo.vstack([mo.md("**Figure 1 · The book against buying everything**, and its exposure: long "
                     "and short legs and the net (dollar-neutral = 0)."),
               layout(_fig, title="informed_flow_ls · config values, point-in-time universe",
                     height=620)])
    return


@app.cell
def _(bt, by_year, mo, one, stats, table):
    _trades = bt.trade_records(one["pf"])
    _borrow = one["pf"].orders.records_readable
    mo.vstack([
        mo.md("**Table 5 · The book's numbers** (geometric CAGR; alpha = CAGR − beta × buy & "
              "hold). The long side alone runs at half the book, so its numbers are half-sized."),
        table(stats.round(3), rows=5),
        mo.md("**Table 6 · Year by year**"),
        table(by_year.round(3), rows=10),
        mo.md(f"**Trades:** {len(_trades)} positions closed or open; fees (commission, "
              f"spread, slippage and borrow) paid: ${float(_borrow['Fees'].sum()):,.0f} on a "
              f"$10,000 book."),
    ])
    return


@app.cell
def _(mo, mod, panel, parts, pd, strat, table):
    _p = strat.params
    _l, _s, _exits = mod["_select"](parts["score"], panel.close, parts["shortable"],
                                    q_in=_p["q_in"], q_out=_p["q_out"],
                                    slots=(_p["long_slots"], _p["short_slots"]),
                                    max_hold=_p["max_hold"], short_stop=_p["short_stop"])
    _x = pd.DataFrame(_exits, columns=["side", "symbol", "opened", "closed", "cause"])
    _x["side"] = _x["side"].map({1: "long", -1: "short"})
    _causes = _x.pivot_table(index="cause", columns="side", values="symbol",
                             aggfunc="count").fillna(0)
    mo.vstack([mo.md("**Table 7 · Why positions were closed** (whole history, decided at the "
                     "close, traded at the next open)"), table(_causes, rows=8)])
    return


if __name__ == "__main__":
    app.run()
