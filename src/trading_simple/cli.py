"""The ``tsim`` command line: schema, screening, backfill, optimization, coverage.

Notebooks hand long-running work to these commands as subprocesses (see
:mod:`trading_simple.jobs`), so interactive exploration and unattended runs
share one code path. Every command prints plain, parseable lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def init_db() -> None:
    """Apply schema.sql (idempotent; a no-op against the live shared database)."""
    from trading_simple.db import apply_schema, make_engine

    apply_schema(make_engine())
    typer.echo("schema: applied (idempotent)")


@app.command()
def screen(
    name: str = typer.Option("LiquidLeaders", help="Universe name to write"),
    top: int = typer.Option(100, help="Universe size"),
    min_cap: float = typer.Option(2e9, help="Minimum market cap in $"),
    min_price: float = typer.Option(5.0, help="Minimum last price in $"),
    sector_cap: float = typer.Option(0.25, help="Max fraction of one sector"),
) -> None:
    """Build a universe from the Nasdaq screener and persist it."""
    from trading_simple import universe as uv
    from trading_simple.db import make_engine

    listed = uv.fetch_listed_stocks()
    screened = uv.screen_candidates(listed, min_market_cap=min_cap, min_price=min_price)
    picks = uv.liquid_leaders(screened, n=top, sector_cap=sector_cap)
    path = uv.write_universe(name, picks)
    n_reg = uv.register_securities(make_engine(), screened[screened["symbol"].isin(picks)])
    typer.echo(f"universe {name}: {len(picks)} symbols -> {path} ({n_reg} securities registered)")


@app.command()
def import_universes(
    path: str = typer.Option(None, help="Source universes.yaml "
                                        "(default: ../stock-trading/config/universes.yaml)"),
    names: str = typer.Option(None, help="Comma-separated universe names "
                                         "(default: all not yet defined here)"),
    overwrite: bool = typer.Option(False, help="Replace universes already defined here"),
) -> None:
    """Copy universes built by the sibling stock-trading repo into this one.

    Both repos share one database, so symbol lists the sibling's screener
    built (HighOctane, Trailblazers, SteadyEddies, CheapQuality) are directly
    usable here. Imports are snapshots — re-run after the sibling regenerates.
    """
    from trading_simple import universe as uv
    from trading_simple.config import PROJECT_ROOT

    src = path or PROJECT_ROOT.parent / "stock-trading" / "config" / "universes.yaml"
    wanted = [n.strip() for n in names.split(",") if n.strip()] if names else None
    written = uv.import_universes(src, names=wanted, overwrite=overwrite)
    for name, count in written.items():
        typer.echo(f"universe {name}: {count} symbols imported")
    typer.echo(f"universes: {len(written)} imported from {src}")


@app.command()
def subset(
    universe: str = typer.Option(..., help="Source universe name"),
    top: int = typer.Option(20, help="How many symbols to keep"),
    days: int = typer.Option(90, help="Dollar-volume lookback in calendar days"),
    name: str = typer.Option(None, help="Universe name to write (default: <universe>Top<top>)"),
) -> None:
    """Pin a universe's most liquid names as a new universe: <universe>Top<N>.

    THE method for carving a tradable subset out of a big universe (e.g. for
    slow 1-minute backfills and intraday strategies): rank the members by
    average daily dollar volume (mean close x volume) over the trailing
    --days of stored daily bars, keep the --top, and persist the list to
    config/universes.yaml so backtests are reproducible. Re-running refreshes
    the snapshot; strategies keep referencing the name.
    """
    from trading_simple import universe as uv
    from trading_simple.config import load_settings
    from trading_simple.db import make_engine

    settings = load_settings()
    ranked = uv.rank_by_dollar_volume(make_engine(settings),
                                      settings.universe_symbols(universe), days=days)
    if ranked.empty:
        raise typer.BadParameter(f"no daily bars stored for {universe!r} — backfill 1d first")
    picks = ranked.head(top)
    target = name or f"{universe}Top{top}"
    uv.write_universe(target, list(picks.index))
    for sym, row in picks.iterrows():
        typer.echo(f"subset {sym}: ${row['dollar_volume']:,.0f}/day")
    typer.echo(f"universe {target}: {len(picks)} symbols written")


@app.command()
def backfill(
    universe: str = typer.Option(None, help="Universe name from config"),
    symbols: str = typer.Option(None, help="Comma-separated symbols (overrides --universe)"),
    bar: str = typer.Option("1d", help="Bar size: 1d | 1m | 5m | 1h"),
    since: datetime = typer.Option(..., formats=["%Y-%m-%d"], help="First day (YYYY-MM-DD)"),
    until: datetime = typer.Option(None, formats=["%Y-%m-%d"], help="Last day (default today)"),
    provider: str = typer.Option(None, help="yfinance | ibkr (default: by bar size)"),
    extended_hours: bool = typer.Option(None, "--extended-hours/--regular-only",
                                        help="IBKR only: include pre/post-market bars"),
) -> None:
    """Backfill price bars into the shared database (idempotent UPSERT).

    Prints one ``prices SYM BAR: N bars`` line per symbol and a final
    ``prices: N rows written`` — the notebook helpers parse these.
    """
    from trading_simple.config import load_settings
    from trading_simple.db import Security, make_engine, upsert_prices, upsert_securities

    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    syms = ([s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols
            else settings.universe_symbols(universe or "starter"))
    start = since.date()
    end = until.date() if until else date.today()
    provider = provider or ("yfinance" if bar == "1d" else "ibkr")
    if extended_hours is not None:
        settings.ibkr.extended_hours = extended_hours

    engine = make_engine(settings)
    upsert_securities(engine, [Security(symbol=s) for s in syms])  # FK before prices
    total = 0

    if provider == "ibkr":
        from trading_simple.providers.ibkr import fetch_batch

        def store(sym: str, bars: list) -> None:
            nonlocal total
            total += upsert_prices(engine, bars) if bars else 0
            typer.echo(f"prices {sym} {bar}: {len(bars)} bars")

        fetch_batch(syms, bar, start, end, settings=settings.ibkr,
                    exchanges=settings.exchanges, on_symbol=store)
    elif provider == "yfinance":
        from trading_simple.providers.yf import fetch_daily

        if bar != "1d":
            raise typer.BadParameter(
                "yfinance is daily-only here; use --provider ibkr for intraday")
        for sym in syms:
            try:
                bars = fetch_daily(sym, start, end, exchanges=settings.exchanges)
            except RuntimeError as exc:
                typer.echo(f"prices {sym} {bar}: 0 bars ({exc})")
                continue
            total += upsert_prices(engine, bars)
            typer.echo(f"prices {sym} {bar}: {len(bars)} bars")
    else:
        raise typer.BadParameter(f"Unknown provider {provider!r} (yfinance | ibkr)")
    typer.echo(f"prices: {total} rows written")


@app.command()
def altdata(
    dataset: str = typer.Option(
        "all", help="insider | ratings | price_target | short_interest | fundamentals | all"),
    universe: str = typer.Option(None, help="Universe name from config, or ALL (every symbol)"),
    symbols: str = typer.Option(None, help="Comma-separated symbols (overrides --universe)"),
    since: datetime = typer.Option(..., formats=["%Y-%m-%d"], help="First day (YYYY-MM-DD)"),
    skip_covered: bool = typer.Option(
        False, "--skip-covered/--no-skip-covered",
        help="Skip (dataset, symbol) already backfilled to at least --since (uses the ledger)"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-fetch even symbols the ledger marks as already covered"),
    bulk: bool = typer.Option(
        False, "--bulk",
        help="Insider only: backfill from SEC's quarterly Form 345 data sets (whole-market "
             "downloads) instead of per-filing scraping — far faster for closed quarters"),
) -> None:
    """Ingest alternative data into the shared database (idempotent UPSERT).

    This repo's own providers — SEC EDGAR Form 4 (insider) and XBRL companyfacts
    (fundamentals), Yahoo Finance (analyst rating changes and price targets),
    FINRA (short interest) — populate the alt-data tables consumed by
    :mod:`trading_simple.features`, with no dependency on the sibling repo. Prints
    one ``altdata <ds> <SYM>: N rows`` line per symbol and a final
    ``altdata: N rows written`` (the notebook helpers parse these).

    ``--skip-covered`` consults the ``altdata_coverage`` ledger and skips symbols
    already backfilled to ``--since``, so adding one name to a universe re-fetches
    only that name (``--refresh`` forces a re-fetch). ``--universe ALL`` targets
    every registered symbol. Historic backfills should be run from
    ``notebooks/03_altdata_update.ipynb``, which also carries the coverage audit.
    """
    from trading_simple.config import load_settings
    from trading_simple.db import Security, all_symbols, make_engine, upsert_securities

    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    engine = make_engine(settings)
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif (universe or "").upper() == "ALL":
        syms = all_symbols(engine)
    else:
        syms = settings.universe_symbols(universe or "starter")
    start = since.date()
    datasets = (("insider", "ratings", "price_target", "short_interest", "fundamentals")
                if dataset == "all" else (dataset,))

    upsert_securities(engine, [Security(symbol=s) for s in syms])  # FK before alt-data rows
    total = sum(_run_altdata(ds, engine, settings, syms, start,
                             skip_covered=skip_covered, refresh=refresh, bulk=bulk)
                for ds in datasets)
    typer.echo(f"altdata: {total} rows written")


@app.command("altdata-status")
def altdata_status(
    universe: str = typer.Option(None, help="Universe name from config, or ALL (every symbol)"),
    symbols: str = typer.Option(None, help="Comma-separated symbols (overrides --universe)"),
    since: datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Backfill start to judge coverage against (YYYY-MM-DD)"),
    per_symbol: bool = typer.Option(
        False, "--per-symbol", help="One line per (dataset, symbol), not per (dataset, source)"),
) -> None:
    """Report what alt-data exists, since when, and via what source/method.

    For the given universe and a candidate ``--since``, prints per ``(dataset,
    source)``: row count, breadth, the actual data date range, the coverage-ledger
    window, whether a ``REFRESH=False`` backfill at ``--since`` would SKIP it
    (``will_skip``: every ledgered symbol already reaches back to ``--since``), and a
    ``recorded_but_empty`` count (symbols the ledger claims but the table has no rows
    for). ``--per-symbol`` drills down to one line per name. Read-only.
    """
    from trading_simple.config import load_settings
    from trading_simple.db import (
        all_symbols,
        coverage_overview,
        coverage_overview_symbols,
        make_engine,
    )

    settings = load_settings()
    engine = make_engine(settings)
    if symbols:
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    elif (universe or "").upper() == "ALL":
        syms = all_symbols(engine)
    else:
        syms = settings.universe_symbols(universe or "starter")
    want_from = since.date()

    if per_symbol:
        rows = coverage_overview_symbols(engine, syms, want_from)
        for r in rows:
            flag = "SKIP" if r["will_skip"] else ("EMPTY!" if r["recorded_but_empty"] else "fetch")
            typer.echo(f"altdata-status {r['dataset']:14} {r['symbol']:8} {flag:6} "
                       f"rows={r['rows']:6} src={r['sources'] or '-':30} "
                       f"cov={r['covered_from']}..{r['covered_to']}")
        empties = sum(1 for r in rows if r["recorded_but_empty"])
        typer.echo(f"altdata-status: {len(rows)} rows; {empties} recorded-but-empty")
        return

    for r in coverage_overview(engine, syms, want_from):
        typer.echo(
            f"altdata-status {r['dataset']:14} {str(r['source'] or '-'):16} "
            f"rows={r['rows']:8} {r['symbols']:5}/{len(syms)} syms  "
            f"known {str(r['first_known_at'])[:10]}..{str(r['last_known_at'])[:10]}  "
            f"ledger {r['ledger_from']}..{r['ledger_to']}  "
            f"will_skip={'yes' if r['will_skip'] else 'no'}  rec_empty={r['recorded_but_empty']}")


# The earliest date recorded as ``covered_from`` for a "whole-history" dataset
# (SEC companyfacts): one fetch stores the symbol's entire history back to
# inception, so it is complete for ANY earlier ``BACKFILL_START`` — the sentinel
# makes the generic ``covered_from <= want_from`` skip test unconditionally true.
COMPLETE_SENTINEL = date(1900, 1, 1)

# Per-dataset meaning of ``since``, which decides how an already-covered symbol is
# deepened when the requested window is widened earlier:
#   "range"      — ``since`` limits the fetch (insider filings/quarters, FINRA
#                  settlements), so fetch ONLY the missing early gap
#                  [want_from, covered_from), bounded by ``until``.
#   "whole"      — one fetch returns the ENTIRE history regardless of ``since``
#                  (companyfacts), so once covered it is complete to inception
#                  (recorded with COMPLETE_SENTINEL) and never re-fetched to deepen.
#   "monolithic" — one request returns the whole frame and ``since`` is a
#                  client-side row filter (yfinance ratings/price targets), so
#                  deepen with ONE re-fetch using the earlier ``since``.
_SEMANTICS = {"insider": "range", "short_interest": "range", "fundamentals": "whole",
              "ratings": "monolithic", "price_target": "monolithic"}


@dataclass(frozen=True, slots=True)
class FetchSpec:
    """One symbol's effective fetch window plus the coverage to record for it.

    ``until is None`` means "open to today" (a full/monolithic fetch); a set
    ``until`` marks a bounded gap-fill that must NOT advance the recorded
    ``covered_to`` (``rec_to`` stays the pre-existing value).
    """

    symbol: str
    since: date
    until: date | None
    rec_from: date
    rec_to: date


def _coverage_plan(syms: list[str], coverage: dict[str, tuple[date, date]],
                   want_from: date, want_to: date, *, kind: str,
                   skip_covered: bool) -> tuple[list[FetchSpec], list[str]]:
    """Plan each symbol's effective fetch window from the ledger (pure).

    A symbol already covered back to ``want_from`` is skipped; a partially-covered
    ``range`` symbol is narrowed to its missing early gap ``[want_from, covered_from)``
    (leaving the covered tail untouched); ``whole``/``monolithic`` partials and
    brand-new symbols get a full fetch. With ``skip_covered`` off, everything is
    fetched — the pre-ledger behaviour.

    Args:
        syms: Symbols requested for this run.
        coverage: ``{symbol: (covered_from, covered_to)}`` from :func:`db.load_coverage`.
        want_from: The run's requested start date.
        want_to: The run's requested end date (today).
        kind: The dataset's coverage semantics (see :data:`_SEMANTICS`).
        skip_covered: Whether to honour the ledger.

    Returns:
        ``(specs, skipped)`` — fetch plans and the skipped symbols, preserving order.
    """
    rec_from_full = COMPLETE_SENTINEL if kind == "whole" else want_from
    if not skip_covered:
        return [FetchSpec(s, want_from, None, rec_from_full, want_to) for s in syms], []
    specs: list[FetchSpec] = []
    skipped: list[str] = []
    for s in syms:
        cov = coverage.get(s)
        if cov is not None and cov[0] <= want_from:          # already deep enough
            skipped.append(s)
        elif cov is not None and kind == "range":            # partial range → gap only
            specs.append(FetchSpec(s, want_from, cov[0], want_from, cov[1]))
        else:                                                # new, or monolithic/whole re-fetch
            specs.append(FetchSpec(s, want_from, None, rec_from_full, want_to))
    return specs, skipped


def _group_by_window(specs: list[FetchSpec]) -> dict[tuple[date, date | None], list[FetchSpec]]:
    """Group specs by their ``(since, until)`` window (pure), preserving first-seen order.

    Batched providers (short interest, insider bulk) issue one request per window,
    so a normal backfill collapses to one or two groups (deep symbols share a
    window, brand-new ones another).
    """
    groups: dict[tuple[date, date | None], list[FetchSpec]] = {}
    for sp in specs:
        groups.setdefault((sp.since, sp.until), []).append(sp)
    return groups


def _cov_row(sp: FetchSpec) -> dict:
    """Coverage-ledger row for a completed batched-fetch spec (``rows`` audit unknown)."""
    return {"symbol": sp.symbol, "covered_from": sp.rec_from, "covered_to": sp.rec_to,
            "rows": None}


def _coverage_targets(syms: list[str], coverage: dict[str, tuple[date, date]], since: date,
                      *, skip_covered: bool) -> tuple[list[str], list[str]]:
    """Back-compat shim: the ``(to_fetch, skipped)`` partition from :func:`_coverage_plan`.

    The partition is independent of ``kind``/``want_to``, so this preserves the
    original behaviour (skip iff ``skip_covered`` and ``covered_from <= since``).
    """
    specs, skipped = _coverage_plan(syms, coverage, since, since, kind="range",
                                    skip_covered=skip_covered)
    return [sp.symbol for sp in specs], skipped


def _run_altdata(ds: str, engine, settings, syms: list[str], start: date,
                 *, skip_covered: bool = False, refresh: bool = False, bulk: bool = False) -> int:
    """Ingest one alt-data dataset for ``syms`` since ``start``; return rows written.

    Per-symbol network sources (insider per-filing, ratings, price_target,
    fundamentals) are fetched one symbol at a time so a single failure is isolated
    and reported, never aborting the run; short interest and bulk insider are
    batched by fetch window. :func:`_coverage_plan` turns the ``altdata_coverage``
    ledger into a per-symbol plan (``skip_covered`` honoured unless ``refresh``):
    a symbol already covered back to ``start`` is skipped; a partially-covered
    ``range`` dataset (insider, short_interest) fetches ONLY the missing early gap
    ``[start, covered_from)`` and leaves the covered tail; a ``whole`` dataset
    (fundamentals) is complete once covered; a ``monolithic`` dataset (ratings,
    price_target) re-fetches once to deepen. Each cleanly-fetched symbol widens its
    ledger window.

    ``bulk`` (insider only) swaps the per-filing scrape for SEC's quarterly Form 345
    data sets: a whole-market download per window.

    Args:
        ds: Dataset name (``insider`` | ``ratings`` | ``price_target`` |
            ``short_interest`` | ``fundamentals``).
        engine: Database engine.
        settings: Resolved settings (for exchange overrides).
        syms: Symbols to ingest.
        start: Earliest date to fetch.
        skip_covered: Skip symbols already backfilled to ``start`` (per the ledger).
        refresh: Ignore the ledger and re-fetch every symbol.
        bulk: Insider only — use the quarterly Form 345 bulk data sets.

    Returns:
        Number of rows written for this dataset.
    """
    from trading_simple.db import (
        load_coverage,
        upsert_coverage,
        upsert_fundamentals,
        upsert_insider,
        upsert_price_targets,
        upsert_ratings,
        upsert_short_interest,
    )

    kind = _SEMANTICS.get(ds)
    if kind is None:
        raise typer.BadParameter(
            f"Unknown dataset {ds!r} "
            "(insider | ratings | price_target | short_interest | fundamentals | all)")
    today = date.today()
    coverage = load_coverage(engine, ds)
    specs, skipped = _coverage_plan(syms, coverage, start, today, kind=kind,
                                    skip_covered=skip_covered and not refresh)
    if skipped:
        typer.echo(f"altdata {ds}: skipping {len(skipped)} symbol(s) already covered "
                   f"back to {start} (--refresh to force)")
    gaps = sum(1 for sp in specs if sp.until is not None)
    if gaps:
        typer.echo(f"altdata {ds}: {gaps} symbol(s) gap-fill only (fetching before covered_from)")
    if not specs:
        typer.echo(f"altdata {ds}: nothing to do — all {len(syms)} symbols already covered")
        return 0

    if ds == "insider" and bulk:
        from trading_simple.providers.sec import iter_insider_bulk
        # Stream quarter-by-quarter per window: UPSERT each quarter's rows and release
        # them, so a full-universe backfill never holds millions of trades at once.
        written = 0
        for (since, until), group in _group_by_window(specs).items():
            gsyms = [sp.symbol for sp in group]
            for quarter, rows in iter_insider_bulk(gsyms, since, until=until):
                written += upsert_insider(engine, rows)
                typer.echo(f"altdata insider (bulk) {quarter}: {len(rows)} rows")
            upsert_coverage(engine, ds, [_cov_row(sp) for sp in group])
        typer.echo(f"altdata insider (bulk): {written} rows across {len(specs)} symbols")
        return written
    if ds == "short_interest":
        # Bulk CDN files (one per settlement, all symbols, no auth) — the Query API's
        # symbol domainFilter 504s on a full universe. Stream per settlement, UPSERT and
        # release, mirroring the insider bulk path.
        from trading_simple.providers.finra import iter_short_interest_bulk
        written = 0
        for (since, until), group in _group_by_window(specs).items():
            gsyms = [sp.symbol for sp in group]
            try:
                for settle, rows in iter_short_interest_bulk(gsyms, since, until=until):
                    written += upsert_short_interest(engine, rows)
                    typer.echo(f"altdata short_interest {settle}: {len(rows)} rows")
            except Exception as exc:  # noqa: BLE001 — a FINRA outage must not abort the run
                typer.echo(f"altdata short_interest: error ({exc})")
                continue
            upsert_coverage(engine, ds, [_cov_row(sp) for sp in group])
        typer.echo(f"altdata short_interest (bulk): {written} rows across {len(specs)} symbols")
        return written
    if ds == "insider":
        from trading_simple.providers.sec import fetch_insider
        return _ingest_per_symbol(ds, engine, specs,
                                  lambda s, since, until: fetch_insider(s, since, until=until),
                                  upsert_insider)
    if ds == "ratings":
        from trading_simple.providers.yf import fetch_ratings
        return _ingest_per_symbol(
            ds, engine, specs,
            lambda s, since, until: fetch_ratings(s, since, exchanges=settings.exchanges),
            upsert_ratings)
    if ds == "price_target":
        from trading_simple.providers.yf import fetch_price_targets
        return _ingest_per_symbol(
            ds, engine, specs,
            lambda s, since, until: fetch_price_targets(s, since, exchanges=settings.exchanges),
            upsert_price_targets)
    # fundamentals (the only remaining kind)
    from trading_simple.providers.sec import fetch_fundamentals
    return _ingest_per_symbol(ds, engine, specs,
                              lambda s, since, until: fetch_fundamentals(s, since),
                              upsert_fundamentals)


def _ingest_per_symbol(ds: str, engine, specs: list[FetchSpec], fetch, upsert) -> int:
    """Fetch + upsert each spec one at a time, recording ledger coverage on success.

    A failing symbol is logged and skipped — and NOT recorded, so it retries next
    run; a symbol that fetches cleanly (even with zero rows) IS recorded (with its
    spec's ``covered_from``/``covered_to``), so a genuinely event-free name is not
    re-attempted forever.

    Args:
        ds: Dataset name (for log lines and the ledger).
        engine: Database engine.
        specs: Per-symbol fetch plans from :func:`_coverage_plan`.
        fetch: ``(symbol, since, until) -> rows`` provider call.
        upsert: ``(engine, rows) -> n`` writer.

    Returns:
        Total rows written across all specs.
    """
    from trading_simple.db import upsert_coverage

    total, covered = 0, []
    for sp in specs:
        try:
            rows = fetch(sp.symbol, sp.since, sp.until)
        except Exception as exc:  # noqa: BLE001 — isolate a bad symbol
            typer.echo(f"altdata {ds} {sp.symbol}: 0 rows ({exc})")
            continue
        total += upsert(engine, rows)
        covered.append({"symbol": sp.symbol, "covered_from": sp.rec_from,
                        "covered_to": sp.rec_to, "rows": len(rows)})
        typer.echo(f"altdata {ds} {sp.symbol}: {len(rows)} rows")
    upsert_coverage(engine, ds, covered)
    return total


@app.command()
def optimize(
    strategy: str = typer.Option(..., help="Strategy folder name, e.g. volume_momentum"),
    n_trials: int = typer.Option(None, help="Trials to run (default: strategy config)"),
    timeout_s: float = typer.Option(None, help="Stop after this many seconds"),
    study: str = typer.Option(None, help="Study name (default: strategy config)"),
    sampler: str = typer.Option(None, help="nsga2 | tpe | random (default: strategy config)"),
) -> None:
    """Run or RESUME an Optuna study for a strategy (journal-backed, kill-safe).

    Re-running the same command continues where it stopped; every finished
    trial is already checkpointed in the study's journal file.
    """
    from trading_simple import optimize as opt
    from trading_simple import strategies
    from trading_simple.config import load_settings
    from trading_simple.data import load_panel, symbols_with_bars
    from trading_simple.db import make_engine

    settings = load_settings()
    strat = strategies.load(strategy)
    if study:
        strat.optuna["study"] = study
    if sampler:
        strat.optuna["sampler"] = sampler
    d = strat.data
    engine = make_engine(settings)
    syms = symbols_with_bars(engine, settings.universe_symbols(d["universe"]), d["bar"],
                             d["start"], d.get("end"), limit=d.get("max_symbols"))
    panel = load_panel(engine, syms, d["bar"], d["start"], d.get("end"),
                       session=d.get("session", "regular"), exchanges=settings.exchanges)
    if d.get("features"):  # attach alt-data feature matrices for alt-data strategies
        from trading_simple import features
        features.attach_features(panel, engine, d["features"], settings=settings)
    typer.echo(f"optimize {strategy}: {len(syms)} symbols, study={strat.optuna['study']}")
    typer.echo("dashboard: uvx optuna-dashboard "
               f"{opt.study_path(strat, strat.optuna['study'])}  "
               "(run in another terminal, live at http://127.0.0.1:8080)")
    result = opt.run_study(strat, panel, settings=settings,
                           n_trials=n_trials, timeout_s=timeout_s, verbose=True)
    typer.echo(f"study {result.study_name}: {len(result.trials)} trials total, "
               f"{len(result.best_trials)} on the Pareto front")
    if result.best_trials:
        report = opt.boundary_report(opt.pick_best(result).params,
                                     strat.optuna.get("space", {}),
                                     front=opt.pareto_frame(result))
        for name, row in report[report["action"] != "ok"].iterrows():
            typer.echo(f"boundary {name}: best={row['best']:g} "
                       f"searched=[{row['low']:g}, {row['high']:g}] -> {row['action']}")


@app.command()
def coverage(
    universe: str = typer.Option(None, help="Universe name (default: whole-DB inventory)"),
    bar: str = typer.Option(None, help="Restrict to one bar size"),
) -> None:
    """Show what price data is stored (per bar, or per symbol for a universe)."""
    from trading_simple.config import load_settings
    from trading_simple.data import coverage as cov
    from trading_simple.db import make_engine

    settings = load_settings()
    syms = settings.universe_symbols(universe) if universe else None
    frame = cov(make_engine(settings), symbols=syms, bar=bar)
    typer.echo(frame.to_string(index=False))


if __name__ == "__main__":
    app()
