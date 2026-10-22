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
    # 🛰️ Alternative data — short interest, insider, fundamentals, ratings, price targets

    **The single entry point for alt-data ingestion.** This repo's own providers — FINRA (short
    interest), SEC EDGAR (Form 4 insider trades and XBRL `companyfacts` fundamentals) and Yahoo
    Finance (analyst rating changes and price targets) — write to the **shared TimescaleDB**; the
    option chains of §5b are local files. No dependency on the sibling `stock-trading` repo.

    **Every run fetches only what is missing, for every name in the database.** The coverage
    ledger (`altdata_coverage`) records, per dataset and name, the window already fetched. A run
    fetches, name by name, the history back to `BACKFILL_START` that a name lacks and the days
    since its last refresh (re-reading a short overlap, `cli._OVERLAP_DAYS`, for data published
    late). A name refreshed recently costs nothing; one never ingested gets its whole history.
    Insider history up to SEC's newest quarterly data set comes from those bulk files (one
    download serves every company), only the days after it from the per-filing scraper. Stop a
    run whenever you like: every finished name is recorded at once, and the next run continues
    after it. §2 shows the plan before anything is downloaded; the button in §3 starts it.

    Point-in-time by construction: every row is stamped `known_at` (SEC acceptance, FINRA
    dissemination, the rating's announcement), so features never see a value before it was
    public. Writes UPSERT on natural keys, so a re-run is safe and idempotent.

    > **Why the audits matter.** A half-ingested table does not fail loudly — it goes *quiet*.
    > `short_interest` once shipped covering 3 of 100 names and read as fresh, because the check
    > only looked at the newest row; the strategy ranked those 3 and gave 97 a neutral 0.5. In
    > 2026 a one-universe update left `informed_flow`'s 500 names on weeks-old short interest
    > with every check green, and from July 2026 short interest reached only ~765 of ~2,300
    > names. So this notebook now always covers the whole registry, §4 checks breadth and depth
    > (not just recency), and §4a audits the NEWEST data of every universe a strategy trades.

    Flow: **parameters → what is missing → ingest → ledger → freshness, breadth & depth →
    per-symbol coverage → option chains → schema.**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def table(df, rows=12):
        """A DataFrame as a marimo table; a named index becomes a column."""
        return mo.ui.table(df.reset_index() if df.index.name else df, selection=None,
                           page_size=rows)

    return (table,)


@app.cell
def _():
    import warnings

    warnings.filterwarnings("ignore")

    from datetime import date

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sqlalchemy import text

    from trading_simple import cli, features, jobs, strategies
    from trading_simple.config import load_settings
    from trading_simple.db import all_symbols, coverage_overview, load_coverage, make_engine
    from trading_simple.plots import SERIES
    from trading_simple.providers.sec import bulk_published_until

    settings = load_settings()
    engine = make_engine(settings)
    TABLES = {"short_interest": "short_interest", "insider": "insider_trades",
              "fundamentals": "fundamentals", "ratings": "analyst_ratings",
              "price_target": "analyst_price_targets"}
    return (
        SERIES, TABLES, all_symbols, bulk_published_until, cli, coverage_overview, date, engine,
        features, go, jobs, load_coverage, make_subplots, pd, settings, strategies, text,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 · Parameters
    """)
    return


@app.cell
def _(cli):
    # "ALL" = every name in the shared DB (the default: a name missing nothing costs nothing).
    # A config universe name, or "TRADED" (the union of the universes the alt-data strategies
    # trade), restricts a run to those names.
    UNIVERSE = "ALL"
    # Short interest first: the live books read it, and it takes about a minute.
    DATASETS = ["short_interest", "insider", "fundamentals", "ratings", "price_target"]
    # The deepest history wanted. It is free to hold, so widening a train window later is a
    # config change rather than a re-ingestion (fundamentals: a 2019 filing seeds a 2021 start).
    BACKFILL_START = "2018-01-01"
    # Days a name's newest data may age before it is refreshed. The defaults (cli._STALE_DAYS)
    # refresh short interest and insider trades on every run and the per-name sources (Yahoo,
    # companyfacts: one request per name, 20-40 minutes each over ~2,300 names) once a week.
    # Override one like this: {**cli._STALE_DAYS, "ratings": 0}
    STALE_DAYS = {**cli._STALE_DAYS}
    # True ignores the ledger and re-fetches every name's whole window (hours; rarely needed).
    REFRESH = False
    # Names above this get no name-by-name audit (§5); the traded names are audited instead.
    AUDIT_MAX = 600

    # SEC EDGAR REQUIRES a contact email in the User-Agent or it returns 403.
    import os as _os

    if not _os.environ.get("TRADING_SEC_USER_AGENT"):
        _os.environ["TRADING_SEC_USER_AGENT"] = "your-name your-email@example.com"
    return AUDIT_MAX, BACKFILL_START, DATASETS, REFRESH, STALE_DAYS, UNIVERSE


@app.cell
def _(BACKFILL_START, UNIVERSE, all_symbols, date, engine, mo, pd, settings, strategies):
    # What the strategies actually read, from every config that attaches alt-data features:
    # {universe: {"strategies": ..., "datasets": ...}}. §4a audits each universe on its own — a
    # union can look fresh while one member is stale.
    TRADED = {}
    for _name in strategies.available():
        _data = strategies.load(_name).data
        for _feat in _data.get("features") or []:
            _entry = TRADED.setdefault(_data["universe"], {"strategies": set(), "datasets": set()})
            _entry["strategies"].add(_name)
            _entry["datasets"].add("fundamentals" if _feat.startswith("fund_") else _feat)
    traded_names = sorted(set().union(*(settings.universe_symbols(_u) for _u in TRADED)))
    if UNIVERSE.upper() == "ALL":
        uni = all_symbols(engine)
    elif UNIVERSE.upper() == "TRADED":
        uni = traded_names
    else:
        uni = settings.universe_symbols(UNIVERSE)
    today = pd.Timestamp(date.today(), tz="UTC")
    want_start = pd.Timestamp(BACKFILL_START, tz="UTC")
    mo.md(f"DB `{settings.db.host}:{settings.db.port}/{settings.db.name}` · **{UNIVERSE}: "
          f"{len(uni)} names** · traded: "
          + "; ".join(f"{_u} ({', '.join(sorted(_e['strategies']))})"
                      for _u, _e in TRADED.items()))
    return TRADED, today, traded_names, uni, want_start


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 · What is missing

    The plan the ingest will follow, read from the ledger (nothing is downloaded here). **up to
    date**: refreshed within `STALE_DAYS` and complete back to `BACKFILL_START`. **to fetch**: a
    refresh of the days since its last run, plus — **with missing history** — the history it
    lacks (new names get everything). Re-read after every ingest: a finished run leaves every
    name up to date.
    """)
    return


@app.cell
def _(
    BACKFILL_START, DATASETS, REFRESH, STALE_DAYS, bulk_published_until, cli, date, engine,
    load_coverage, mo, pd, read_at, table, uni,
):
    _start = date.fromisoformat(BACKFILL_START)
    _bulk_end = bulk_published_until()
    _rows = []
    for _ds in DATASETS:
        _specs = cli._coverage_plan(
            uni, load_coverage(engine, _ds), _start, date.today(), kind=cli._SEMANTICS[_ds],
            overlap=cli._OVERLAP_DAYS[_ds], stale_days=STALE_DAYS[_ds], refresh=REFRESH)
        _todo = {_sp.symbol for _sp in _specs}
        _rows.append({
            "dataset": _ds, "stale after (days)": STALE_DAYS[_ds],
            "up to date": len(uni) - len(_todo), "to fetch": len(_todo),
            "with missing history": len({_sp.symbol for _sp in _specs if _sp.since == _start}),
            "earliest day fetched": min((_sp.since for _sp in _specs), default=None),
            "source": ("SEC bulk files to " + str(_bulk_end) + ", per filing after"
                       if _ds == "insider" else ""),
        })
    mo.vstack([mo.md(f"**The plan, read at {read_at}**"), table(pd.DataFrame(_rows))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 · Ingest what is missing

    Each dataset runs as **one paced `tsim altdata` subprocess**: SEC and FINRA rate-limit per
    IP, so — like the IBKR rule — there is no fan-out. Progress streams below the cell, one line
    per name (`[i/n]`). `ratings` and `price_target` come from the same Yahoo response; they are
    two tables because they are two signals (most target revisions arrive with the rating
    unchanged).
    """)
    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="ingest what is missing")
    mo.hstack([run], justify="start")
    return (run,)


@app.cell
def _(BACKFILL_START, DATASETS, REFRESH, STALE_DAYS, UNIVERSE, jobs, mo, pd, run, uni):
    if run.value:
        for _ds in DATASETS:
            print(f"\n=== {_ds} ===", flush=True)
            jobs.run_altdata(_ds, since=BACKFILL_START, universe=UNIVERSE,
                             symbols=None if UNIVERSE.upper() == "ALL" else uni,
                             refresh=REFRESH, stale_days=STALE_DAYS[_ds])
    # Every audit below reads the DB again whenever this changes.
    read_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    mo.md(f"_Ingest finished at {read_at}; the plan and every audit below re-read the DB._"
          if run.value else "_Press the button to fetch what §2 lists._")
    return (read_at,)


@app.cell
def _(engine, mo, pd, read_at, table, text, uni):
    # The ledger: per dataset, how many names are tracked and the depth/recency fetched.
    _led = pd.read_sql(text(
        "SELECT dataset, count(*) AS symbols, min(covered_from) AS deepest_from, "
        "max(covered_to) AS latest_to, min(covered_to) AS oldest_refresh "
        "FROM altdata_coverage WHERE symbol = ANY(:s) GROUP BY dataset ORDER BY dataset"),
        engine, params={"s": uni})
    mo.vstack([mo.md(f"**The coverage ledger for these {len(uni)} names** (read at {read_at}; "
                     "`oldest_refresh` is the name refreshed longest ago)"),
               table(_led) if not _led.empty else mo.md("_ledger empty — the first ingest "
                                                       "fills it_")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 · Freshness, breadth & depth

    Three questions, not one. **Fresh** (is the newest row recent?), **broad** (how many of the
    names are present?) and **deep** (how far back does it go?). The old version of this check
    asked only the first, which is exactly how a 3-of-100-name `short_interest` table passed as
    healthy. A `⚠` means the sleeve is **not safe to trade on**: a ranker drops a signal
    covering less than half its names, but one broad enough to survive that floor yet still
    patchy quietly distorts the ranking. These read the whole history; §4a asks whether the
    *newest* data is broad in *every* universe a strategy trades.
    """)
    return


@app.cell
def _(BACKFILL_START, TABLES, engine, mo, pd, read_at, table, text, today, uni, want_start):
    MIN_COVERAGE = 0.5   # a cross-sectional sleeve below this is dropped (strategy._MIN_COVERAGE)
    _rows = []
    with engine.connect() as _c:
        for _label, _tbl in TABLES.items():
            _stat = _c.execute(text(
                f"SELECT count(*), min(known_at), max(known_at), count(DISTINCT symbol) "
                f"FROM {_tbl} WHERE symbol = ANY(:s)"), {"s": uni}).one()
            _rows.append({"dataset": _label, "rows": _stat[0], "first_known_at": _stat[1],
                          "last_known_at": _stat[2], "symbols": _stat[3]})
    report = pd.DataFrame(_rows)
    report["coverage"] = report["symbols"] / len(uni)
    report["stale_days"] = report["last_known_at"].apply(
        lambda t: (today - pd.Timestamp(t)).days if t is not None else None)
    report["short_by_days"] = report["first_known_at"].apply(
        lambda t: max(0, (pd.Timestamp(t) - want_start).days) if t is not None else None)

    def _flag(row):
        """Why this sleeve is not trustworthy — empty string when it is."""
        if not row["rows"]:
            return "⚠ EMPTY — press ingest"
        if row["coverage"] < MIN_COVERAGE:
            return f"⚠ THIN — {row['symbols']}/{len(uni)} names; a ranker will DROP it"
        if row["short_by_days"] and row["short_by_days"] > 90:
            return f"⚠ SHALLOW — history starts {row['short_by_days']}d after {BACKFILL_START}"
        return ""

    report["verdict"] = report.apply(_flag, axis=1)
    _bad = report[report["verdict"] != ""]
    _msg = (f"✅ all {len(report)} datasets cover ≥{MIN_COVERAGE:.0%} of the names back to "
            f"{BACKFILL_START}" if _bad.empty else "❌ not safe to trade on: "
            + "; ".join(f"{r['dataset']} {r['verdict']}" for _, r in _bad.iterrows()))
    mo.vstack([mo.md(f"**{_msg}** (read at {read_at})"),
               table(report[["dataset", "rows", "symbols", "coverage", "first_known_at",
                             "last_known_at", "stale_days", "verdict"]]
                     .assign(coverage=report["coverage"].map("{:.0%}".format)))])
    return MIN_COVERAGE, report


@app.cell
def _(MIN_COVERAGE, SERIES, UNIVERSE, go, mo, report):
    _fig = go.Figure(go.Bar(
        x=report["dataset"], y=report["coverage"],
        marker_color=[SERIES[5] if v < MIN_COVERAGE else SERIES[0] for v in report["coverage"]],
        text=[f"{s} names" for s in report["symbols"]], textposition="outside"))
    _fig.add_hline(y=MIN_COVERAGE, line={"dash": "dash", "color": SERIES[5]},
                   annotation_text=f"a ranker drops a sleeve below {MIN_COVERAGE:.0%}")
    _fig.update_layout(
        title=f"Alt-data breadth — share of {UNIVERSE} covered (red = will be dropped)",
        template="plotly_white", height=360,
        yaxis={"title": "coverage", "tickformat": ".0%", "range": [0, 1.15]})
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4a · Is the newest data broad — in every universe a strategy trades?

    Breadth over the whole history cannot see a table that went stale at its **end**. So, for
    every universe in `TRADED` and every dataset its strategies read: the share of its names the
    ledger **refreshed in the last `FRESH_DAYS`**, and — short interest being scheduled — the
    share the **newest settlement** covers. A `❌` here is not safe for a live book; a backtest
    that ends before the gap is unaffected.
    """)
    return


@app.cell
def _(TABLES, TRADED, engine, mo, pd, read_at, settings, table, text, today):
    FRESH_DAYS = 14   # a name the ledger last refreshed longer ago than this is stale for a book
    MIN_FRESH = 0.9   # ... and a universe with a smaller share of fresh names is flagged
    _ledger = pd.read_sql(text("SELECT dataset, symbol, covered_to FROM altdata_coverage"),
                          engine, parse_dates=["covered_to"])
    _cutoff = today.tz_localize(None) - pd.Timedelta(days=FRESH_DAYS)
    with engine.connect() as _c:
        _newest = _c.execute(text("SELECT max(settlement_date) FROM short_interest")).scalar()
        _in_newest = set(_c.execute(text(
            "SELECT symbol FROM short_interest WHERE settlement_date = :d"),
            {"d": _newest}).scalars())
    _tail = []
    for _u, _entry in TRADED.items():
        _names = set(settings.universe_symbols(_u))
        _row = {"universe": _u, "traded by": ", ".join(sorted(_entry["strategies"])),
                "names": len(_names)}
        for _ds in TABLES:
            if _ds in _entry["datasets"]:
                _fresh = _ledger[(_ledger["dataset"] == _ds) & _ledger["symbol"].isin(_names)
                                 & (_ledger["covered_to"] >= _cutoff)]
                _row[f"{_ds} refreshed"] = len(_fresh) / len(_names)
        if "short_interest" in _entry["datasets"]:
            _row[f"settlement {_newest}"] = len(_names & _in_newest) / len(_names)
        _tail.append(_row)
    _tail = pd.DataFrame(_tail)
    _shares = [c for c in _tail.columns if c not in ("universe", "traded by", "names")]
    _stale = [(r["universe"], c, r[c]) for _, r in _tail.iterrows() for c in _shares
              if pd.notna(r[c]) and r[c] < MIN_FRESH]
    _msg = (f"✅ every traded universe refreshed within {FRESH_DAYS} days, and the newest "
            f"short-interest settlement ({_newest}) covers each of them" if not _stale else
            f"❌ stale for a live book (below {MIN_FRESH:.0%} fresh): "
            + "; ".join(f"{u} {c} {s:.0%}" for u, c, s in _stale) + " → press ingest")
    mo.vstack([mo.md(f"**{_msg}** (read at {read_at})"),
               table(_tail.assign(**{c: _tail[c].map(lambda v: "—" if pd.isna(v)
                                                     else f"{v:.0%}") for c in _shares}))])
    return


@app.cell
def _(BACKFILL_START, coverage_overview, engine, mo, pd, read_at, table, uni):
    # Provenance: what data exists, since when, via which METHOD (the per-row `source`, so
    # insider splits into sec_edgar / sec_edgar_bulk / ...), and the ledger window.
    # `recorded_but_empty` is the ledger-aware "thin table went quiet" alarm: names the ledger
    # claims but the table has 0 rows for (mostly event-free names; a whole dataset high here
    # is a red flag). Same view as `tsim altdata-status`.
    _prov = pd.DataFrame(coverage_overview(engine, uni, pd.Timestamp(BACKFILL_START).date()))
    mo.vstack([mo.md(f"**Provenance** (read at {read_at})"),
               table(_prov[["dataset", "source", "rows", "symbols", "first_known_at",
                            "last_known_at", "ledger_from", "ledger_to",
                            "recorded_but_empty"]])])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4b · Fundamentals: is any concept frozen?

    A company that keeps filing keeps its fundamentals rows arriving — even when one concept
    inside them stopped updating years ago. That is how the share counts of V, MA and CMCSA sat
    on their 2010 values. The parser now falls back tag by tag per filing, but a filer can still
    stop reporting a concept, so audit per concept:

    * **frozen** — the concept's newest row is more than `MAX_AGE_DAYS` behind the company's
      newest filing. The feature layer masks such values, so the name drops out of any ranking
      that needs the concept.
    * **missing** — the concept has no row at all for that name.
    """)
    return


@app.cell
def _(engine, features, mo, pd, read_at, table, text, uni):
    MAX_AGE_DAYS = features._FUND_MAX_AGE_DAYS   # older fundamentals levels are masked
    with engine.connect() as _c:
        _fund = pd.read_sql(text(
            "SELECT symbol, concept, max(known_at) AS last FROM fundamentals "
            "WHERE symbol = ANY(:s) GROUP BY symbol, concept"), _c, params={"s": uni})
    _behind = (_fund.groupby("symbol")["last"].transform("max") - _fund["last"]).dt.days
    _fund["frozen"] = _behind > MAX_AGE_DAYS
    _freshness = _fund.groupby("concept").agg(covered=("symbol", "nunique"),
                                              frozen=("frozen", "sum"))
    _freshness["missing"] = len(uni) - _freshness["covered"]
    _lines = [f"frozen {c} ({len(n)}): {', '.join(sorted(n)[:20])}"
              for c, n in _fund[_fund["frozen"]].groupby("concept")["symbol"]]
    mo.vstack([mo.md(f"**Fundamentals by concept** (read at {read_at})"), table(_freshness),
               mo.md("\n\n".join(_lines) or "✅ no concept frozen")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4c · Analyst feeds: has any name's feed been cut?

    A feed that stops for ONE name does not fail, it goes quiet. Yahoo's analyst list (the only
    source of `ratings` and `price_target`) ends in October 2024 for SMCI, CAT, META, FANG, WELL …
    although those names are covered as heavily as ever, and Yahoo itself returns nothing newer,
    so re-ingesting cannot fix it. A ranker would read two years of silence as "no analyst news".

    A name is flagged when its silence, up to the last time the ledger refreshed it, is at least
    `MIN_SILENT_DAYS` **and** would have produced at least `CUT_EXPECTED` actions at its own pace
    over the two years before its last action (zero events at that pace has probability ≈ e⁻¹⁰).
    An acquired or delisted name flags too; either way its data after `last` must be MASKED by a
    feature that uses it, never read as zero.
    """)
    return


@app.cell
def _(TABLES, engine, mo, pd, read_at, table, text, uni):
    MIN_SILENT_DAYS = 180   # longer than the lull between two earnings seasons of a busy name
    CUT_EXPECTED = 10       # ... and this many actions expected at the name's own pace
    _cut = []
    with engine.connect() as _c:
        for _label in ("ratings", "price_target"):
            _cut.append(pd.read_sql(text(f"""
                WITH s AS (SELECT symbol, max(rated_at) AS last FROM {TABLES[_label]}
                           WHERE symbol = ANY(:s) GROUP BY symbol)
                SELECT s.symbol, s.last, COALESCE(l.covered_to, CURRENT_DATE) AS refreshed,
                       count(*) FILTER (WHERE t.rated_at > s.last - 730) AS actions_2y
                FROM s JOIN {TABLES[_label]} t USING (symbol)
                LEFT JOIN altdata_coverage l ON l.symbol = s.symbol AND l.dataset = :ds
                GROUP BY s.symbol, s.last, l.covered_to"""), _c,
                params={"s": uni, "ds": _label}, parse_dates=["last", "refreshed"],
            ).assign(dataset=_label))
    _cut = pd.concat(_cut, ignore_index=True)
    _cut["silent_days"] = (_cut["refreshed"] - _cut["last"]).dt.days
    _cut["expected"] = (_cut["actions_2y"] / 730 * _cut["silent_days"]).round()
    _cut = (_cut[(_cut["silent_days"] >= MIN_SILENT_DAYS) & (_cut["expected"] >= CUT_EXPECTED)]
            .sort_values(["dataset", "expected"], ascending=[True, False]))
    _msg = ("✅ no analyst feed went quiet against its own pace" if _cut.empty else
            "❌ " + "; ".join(f"{d}: {len(n)} names' feed went quiet"
                              for d, n in _cut.groupby("dataset")["symbol"])
            + " — mask them after `last`, never read the silence as 'no news'")
    mo.vstack([mo.md(f"**{_msg}** (read at {read_at})"),
               table(_cut[["dataset", "symbol", "last", "refreshed", "actions_2y",
                           "silent_days", "expected"]]) if not _cut.empty else mo.md("")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5 · Per-symbol coverage & gaps

    An average hides exactly the failure that matters: a dataset can look 80% covered while a
    fifth of the names have *no data at all*. This section goes name by name — over the names
    the strategies trade when the run covers more than `AUDIT_MAX` names.

    **What counts as a gap depends on the dataset:** `short_interest` is **scheduled** (FINRA,
    about twice a month), so a missing settlement really is a gap. `insider`, `ratings` and
    `price_target` are **irregular event streams**: a quiet stretch is not a gap, but a name with
    **zero events over the whole window** (it rides the neutral 0.5 rank forever) or a silence
    **wildly out of line** with the limit is.
    """)
    return


@app.cell
def _(AUDIT_MAX, TABLES, engine, mo, pd, read_at, table, text, today, traded_names, uni,
      want_start):
    SCHEDULED = {"short_interest"}          # ~24 disseminations a year
    EXPECTED_PER_YEAR = 24
    # fundamentals file quarterly (domestic) or annually (foreign 20-F issuers): allow more.
    GAP_LIMIT = {"fundamentals": 500}
    DEFAULT_GAP = 400
    audit_names = uni if len(uni) <= AUDIT_MAX else traded_names
    events = {}
    with engine.connect() as _c:
        for _label, _tbl in TABLES.items():
            events[_label] = pd.read_sql(text(
                f"SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d, count(*) AS n "
                f"FROM {_tbl} WHERE symbol = ANY(:s) AND known_at >= :since "
                f"GROUP BY symbol, d ORDER BY symbol, d"),
                _c, params={"s": audit_names, "since": want_start.to_pydatetime()},
                parse_dates=["d"])

    def _per_symbol(label, frame):
        """First/last event, count, and longest silence for every audited name."""
        out = []
        span_years = max((today - want_start).days / 365.25, 1e-9)
        for sym in audit_names:
            g = frame[frame["symbol"] == sym]
            if g.empty:
                out.append({"symbol": sym, "dataset": label, "events": 0, "first": pd.NaT,
                            "last": pd.NaT, "max_gap_d": None, "verdict": "⚠ NEVER COVERED"})
                continue
            dates = pd.to_datetime(g["d"])
            # Silence includes the tail: the stretch from the last event until today.
            edges = pd.concat([pd.Series([want_start.tz_localize(None)]), dates,
                               pd.Series([today.tz_localize(None)])])
            max_gap = int(edges.diff().max().days)
            n = int(g["n"].sum())
            verdict = ""
            if label in SCHEDULED:
                expected = EXPECTED_PER_YEAR * span_years
                if n < 0.7 * expected:
                    verdict = f"⚠ SPARSE — {n} of ~{expected:.0f} expected disseminations"
            elif max_gap > GAP_LIMIT.get(label, DEFAULT_GAP):
                verdict = f"⚠ {max_gap}d silence — check for an interrupted ingest"
            out.append({"symbol": sym, "dataset": label, "events": n, "first": dates.min(),
                        "last": dates.max(), "max_gap_d": max_gap, "verdict": verdict})
        return pd.DataFrame(out)

    _per = pd.concat([_per_symbol(k, v) for k, v in events.items()], ignore_index=True)
    _problems = _per[_per["verdict"] != ""]
    _summary = (_problems.groupby("dataset")["symbol"]
                .agg(flagged="count", examples=lambda s: ", ".join(s[:6])))
    mo.vstack([
        mo.md(f"**{len(audit_names)} names × {len(TABLES)} datasets: {len(_problems)} flagged** "
              f"(read at {read_at})" + ("" if not _problems.empty
                                        else " — ✅ every name has usable history")),
        table(_summary) if not _problems.empty else mo.md(""),
        table(_problems.sort_values(["dataset", "symbol"])) if not _problems.empty
        else mo.md(""),
    ])
    return audit_names, events


@app.cell
def _(SERIES, TABLES, audit_names, events, go, make_subplots, mo, pd, today, want_start):
    # Symbol × month, one panel per dataset. A dark cell = events that month; a pale cell = none.
    # Missing history reads as a solid blank block.
    _months = pd.date_range(want_start.tz_localize(None), today.tz_localize(None), freq="MS")
    _order = sorted(audit_names)
    _fig = make_subplots(rows=1, cols=len(TABLES), shared_yaxes=True,
                         subplot_titles=list(TABLES), horizontal_spacing=0.012)
    for _i, _label in enumerate(TABLES, start=1):
        _frame = events[_label]
        _grid = pd.DataFrame(0.0, index=_order, columns=_months)
        if not _frame.empty:
            _counts = (_frame.assign(m=_frame["d"].dt.to_period("M").dt.to_timestamp())
                       .pivot_table(index="symbol", columns="m", values="n", aggfunc="sum"))
            _grid = _counts.reindex(index=_order, columns=_months).fillna(0.0)
        _fig.add_trace(go.Heatmap(
            z=_grid.to_numpy(), x=_months, y=_order,
            colorscale=[(0.0, "#f4f3ef"), (0.001, "#cfe0f5"), (1.0, SERIES[0])],
            zmin=0, zmax=max(float(_grid.to_numpy().max()), 1.0), showscale=False,
            hovertemplate=(f"<b>{_label}</b><br>%{{y}}<br>%{{x|%b %Y}}"
                           "<br>%{z:.0f} events<extra></extra>")), row=1, col=_i)
    _fig.update_layout(
        title=(f"Alt-data coverage — {len(_order)} names × {len(_months)} months "
               "(pale = no events; a blank BLOCK = missing history)"),
        template="plotly_white", height=max(420, 13 * len(_order)),
        margin={"l": 70, "r": 20, "t": 90, "b": 40})
    _fig.update_yaxes(tickfont={"size": 8}, autorange="reversed")
    _fig.update_xaxes(tickfont={"size": 9})
    mo.as_html(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5b · Option chains (local files, not the DB)

    The `opt_*` features (`informed_flow_ls`) come from MarketData.app's end-of-day option
    chains, which `scripts/marketdata_dump.py` downloads into `var/marketdata/chains/` (one gzip
    CSV per name and month; rerun it daily). They carry no implied volatilities, so
    `providers.marketdata.build_cache()` solves them and reduces each name-day to a few numbers
    (ATM IV, put skew, call − put IV, volumes) in `var/options/`. The first build reads every
    file (about 5 minutes); later builds redo only the files the daily dump touched.

    The audit asks the same breadth question as §4a: does the NEWEST day of chains cover the
    names the strategies trade? A name whose chains stop is masked by the feature layer after
    `features._OPT_MAX_AGE_DAYS` days, never carried.
    """)
    return


@app.cell
def _(mo):
    options_button = mo.ui.run_button(label="build the option cache and audit it")
    mo.hstack([options_button], justify="start")
    return (options_button,)


@app.cell
def _(TRADED, features, mo, options_button, pd, settings, table):
    mo.stop(not options_button.value, mo.md("_Press the button to build and audit the cache._"))
    from trading_simple.providers import marketdata

    _options = marketdata.build_cache()          # incremental: only new or changed chain files
    _newest = _options["d"].max()
    _per_month = _options.groupby(_options["d"].dt.to_period("M"))["symbol"].nunique()
    _last_seen = _options.groupby("symbol")["d"].max()
    _rows = []
    for _u, _entry in TRADED.items():
        if not any(_d.startswith("opt_") for _d in _entry["datasets"]):
            continue
        _seen = _last_seen.reindex(settings.universe_symbols(_u))
        _rows.append({
            "universe": _u, "traded by": ", ".join(sorted(_entry["strategies"])),
            "names": len(_seen), f"chain on {_newest.date()}": f"{(_seen == _newest).mean():.0%}",
            "never": f"{_seen.isna().mean():.0%}",
            f"stale > {features._OPT_MAX_AGE_DAYS}d":
                f"{((_newest - _seen).dt.days > features._OPT_MAX_AGE_DAYS).mean():.0%}"})
    mo.vstack([
        mo.md(f"**Option summaries: {len(_options):,} name-days, "
              f"{_options['symbol'].nunique()} names, {_options['d'].min().date()} → "
              f"{_newest.date()}**"),
        table(_per_month.rename("names with chains").to_frame().T),
        table(pd.DataFrame(_rows)) if _rows else mo.md("no traded universe reads an opt_* feature"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6 · Schema sanity

    Each alt-data table must still have the columns the feature layer reads. They are defined in
    this repo's `schema.sql` with `CREATE TABLE IF NOT EXISTS` (`insider_trades`,
    `analyst_ratings` and `short_interest` column-for-column identical to the sibling repo's), so
    this also catches drift if the sibling altered anything. A missing table: run
    `uv run tsim init-db`.
    """)
    return


@app.cell
def _(engine, mo, text):
    EXPECTED = {
        "insider_trades": ["symbol", "txn_type", "shares", "price", "traded_at", "known_at"],
        "analyst_ratings": ["symbol", "firm", "action", "grade_to", "rated_at", "known_at"],
        "analyst_price_targets": ["symbol", "firm", "pt_action", "pt_current", "pt_prior",
                                  "rated_at", "known_at"],
        "short_interest": ["symbol", "settlement_date", "days_to_cover", "known_at"],
        "fundamentals": ["symbol", "concept", "period_end", "val", "known_at"],
    }
    _lines = []
    with engine.connect() as _c:
        for _tbl, _cols in EXPECTED.items():
            _have = {r[0] for r in _c.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
                {"t": _tbl})}
            _missing = [c for c in _cols if c not in _have]
            _lines.append(f"`{_tbl}`: " + ("MISSING TABLE — run `uv run tsim init-db`"
                                           if not _have else
                                           "OK" if not _missing else f"missing {_missing}"))
    mo.md("\n\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Next steps

    If §4 and §5 are green, open a strategy notebook to backtest what ranks stocks on these
    signals — `strategies/informed_flow_ls/` (short interest, share counts, option skew),
    `strategies/informed_flow/informed_flow.ipynb` (short interest) or
    `strategies/value_factor/value_factor.ipynb` (fundamentals).

    The same ingest runs headless (no audit) — every name, only what is missing:

    ```bash
    uv run tsim altdata --dataset all --universe ALL --since 2018-01-01
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
