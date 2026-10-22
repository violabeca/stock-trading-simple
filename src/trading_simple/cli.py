"""The ``tsim`` command line: schema, screening, backfill, optimization, coverage.

Notebooks hand long-running work to these commands as subprocesses (see
:mod:`trading_simple.jobs`), so interactive exploration and unattended runs
share one code path. Every command prints plain, parseable lines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

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


@app.command("backfill-sweep")
def backfill_sweep(
    bar: str = typer.Option("1m", help="Bar size (IBKR only): 1m | 5m | 1h"),
    since: datetime = typer.Option(..., formats=["%Y-%m-%d"], help="First day (YYYY-MM-DD)"),
    until: datetime = typer.Option(None, formats=["%Y-%m-%d"], help="Last day (default today)"),
    tracker_path: str = typer.Option(
        "var/backfill_1m_tracker.csv",
        help="Progress ledger CSV: one row per symbol, alphabetical"),
    retry_empty: bool = typer.Option(
        False, "--retry-empty", help="Also re-attempt symbols the ledger marked 'empty'"),
    skip_covered: bool = typer.Option(
        True, "--skip-covered/--no-skip-covered",
        help="Skip a symbol without an IBKR call if the DB already spans the window"),
    extended_hours: bool = typer.Option(None, "--extended-hours/--regular-only",
                                        help="Include pre/post-market bars"),
) -> None:
    """Sweep EVERY symbol in every universe, alphabetically, through one paced
    IBKR connection, checkpointed to a resumable CSV ledger.

    Unlike ``backfill``, there is no ``--universe``/``--symbols`` — the target is
    always the full union of every universe in ``config/universes.yaml``. Safe to
    kill (Ctrl-C, Gateway drop, machine restart) and re-run with the SAME
    --bar/--since/--until: the ledger at --tracker-path records one row per symbol
    with its target window and status ('pending' | 'done' | 'empty'), and a re-run
    only re-fetches rows still 'pending' (plus 'empty' ones with --retry-empty).
    Widening --since/--until resets the affected rows back to 'pending' rather than
    trusting stale coverage. ``--skip-covered`` (on by default) also checks the
    database directly before ever calling IBKR, so symbols an earlier notebook
    backfill already covers cost no request at all. This can run for days for a
    large universe/window — run it under ``nohup``/``tmux`` with IB Gateway logged
    in throughout.
    """
    from trading_simple import jobs
    from trading_simple.config import load_settings

    logging.basicConfig(level=logging.WARNING)
    settings = load_settings()
    start = since.date().isoformat()
    end = (until.date() if until else date.today()).isoformat()
    tracker = jobs.run_ibkr_sweep(
        settings.all_symbols(), bar=bar, since=start, until=end, tracker_path=tracker_path,
        extended_hours=extended_hours, retry_empty=retry_empty, skip_covered=skip_covered)
    done = int((tracker["status"] != "pending").sum())
    typer.echo(f"sweep: {done}/{len(tracker)} symbols done -> {tracker_path}")


@app.command()
def altdata(
    dataset: str = typer.Option(
        "all", help="insider | ratings | price_target | short_interest | fundamentals | all"),
    universe: str = typer.Option(None, help="Universe name from config, or ALL (every symbol)"),
    symbols: str = typer.Option(None, help="Comma-separated symbols (overrides --universe)"),
    since: datetime = typer.Option(
        ..., formats=["%Y-%m-%d"], help="Deepest day of history wanted (YYYY-MM-DD)"),
    refresh: bool = typer.Option(
        False, "--refresh", help="Ignore the ledger: re-fetch every symbol's whole window"),
    stale_days: int = typer.Option(
        None, help="Days a symbol's newest data may age before it is refreshed "
                   "(default: per dataset, cli._STALE_DAYS)"),
) -> None:
    """Ingest the alternative data each symbol is MISSING into the shared database (UPSERT).

    This repo's own providers — SEC EDGAR Form 4 (insider) and XBRL companyfacts
    (fundamentals), Yahoo Finance (analyst rating changes and price targets),
    FINRA (short interest) — populate the alt-data tables consumed by
    :mod:`trading_simple.features`, with no dependency on the sibling repo. Prints
    one ``altdata <ds> [i/n] <SYM>: N rows`` line per symbol and a final
    ``altdata: N rows written``.

    The ``altdata_coverage`` ledger decides what is missing, name by name: the history
    back to ``--since`` and the days since the last refresh (:func:`_coverage_plan`).
    A fully covered name costs nothing, so a run over ``--universe ALL`` (every
    registered symbol) is cheap when little is missing. Insider history comes from
    SEC's quarterly bulk data sets, only the days after the newest published quarter
    from the per-filing scraper. Run it from ``notebooks/03_altdata_update.py``,
    which also carries the coverage audit.
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
    total = sum(_run_altdata(ds, engine, settings, syms, start, refresh=refresh,
                             stale_days=stale_days)
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

# Per-dataset meaning of ``since``, which decides how a symbol's missing days are fetched:
#   "range"      — ``since``/``until`` bound the fetch (insider filings/quarters, FINRA
#                  settlements), so each missing piece is fetched on its own.
#   "whole"      — one fetch returns the ENTIRE history regardless of ``since``
#                  (companyfacts), so it is complete to inception once fetched
#                  (recorded with COMPLETE_SENTINEL) and only ever refreshed whole.
#   "monolithic" — one request returns the whole frame and ``since`` is a
#                  client-side row filter (yfinance ratings/price targets), so ONE
#                  fetch from the earliest missing day covers every missing piece.
_SEMANTICS = {"insider": "range", "short_interest": "range", "fundamentals": "whole",
              "ratings": "monolithic", "price_target": "monolithic"}

#: Days before a symbol's recorded ``covered_to`` that a refresh re-reads. ``covered_to`` is
#: the RUN date, not the newest data: FINRA publishes a settlement about two weeks after it,
#: so a settlement dated before a run can still be unpublished at that run (31 days re-reads
#: the last two settlement files); a Form 4 accepted on the run date after the run is caught
#: by 2 days. One Yahoo request returns a name's whole analyst list, so re-writing 30 days of
#: its rows is free; companyfacts is always the whole history.
_OVERLAP_DAYS = {"insider": 2, "short_interest": 31, "ratings": 30, "price_target": 30,
                 "fundamentals": 0}

#: Days a symbol's newest data may age before it counts as missing (``--stale-days``
#: overrides). 0 = anything not refreshed today. A refresh of the per-name sources costs one
#: request per name whatever is new (Yahoo's analyst lists, SEC companyfacts: ~2,300 requests,
#: 20-40 minutes each over the full registry), so they wait a week; short interest is two
#: downloads for every name and insider only the filings since the last run.
_STALE_DAYS = {"insider": 0, "short_interest": 0, "ratings": 6, "price_target": 6,
               "fundamentals": 6}


@dataclass(frozen=True, slots=True)
class FetchSpec:
    """One symbol's fetch window plus the coverage to record for it once fetched.

    ``until is None`` means "open to today"; a set ``until`` bounds a piece of missing
    history. ``rec_from``/``rec_to`` are the window recorded in the ledger:
    :func:`db.upsert_coverage` widens the stored window, and every piece touches it.
    """

    symbol: str
    since: date
    until: date | None
    rec_from: date
    rec_to: date


def _coverage_plan(syms: list[str], coverage: dict[str, tuple[date, date]],
                   want_from: date, want_to: date, *, kind: str, overlap: int = 0,
                   stale_days: int = 0, refresh: bool = False) -> list[FetchSpec]:
    """Plan only what each symbol is missing, from the ledger (pure).

    The ledger holds one window per symbol, ``(covered_from, covered_to)``. A symbol can
    miss two pieces: the HISTORY from ``want_from`` up to ``covered_from``, and the TAIL
    after ``covered_to`` once that is more than ``stale_days`` old (re-read from
    ``covered_to - overlap``, see :data:`_OVERLAP_DAYS`). Per ``kind``:

    * ``range``: each piece is its own fetch, so a covered name costs only its tail;
    * ``whole``: either piece means one full fetch, recorded complete to inception;
    * ``monolithic``: one fetch from the earliest missing day covers both pieces.

    A symbol not in the ledger — or every symbol under ``refresh`` — gets the whole
    window ``[want_from, want_to]``.

    Args:
        syms: Symbols requested for this run.
        coverage: ``{symbol: (covered_from, covered_to)}`` from :func:`db.load_coverage`.
        want_from: The deepest day of history wanted.
        want_to: The run's end date (today).
        kind: The dataset's coverage semantics (see :data:`_SEMANTICS`).
        overlap: Days before ``covered_to`` a tail re-reads.
        stale_days: Days ``covered_to`` may age before the tail counts as missing.
        refresh: Ignore the ledger.

    Returns:
        Fetch plans in input order; a symbol without one is up to date.
    """
    whole_from = COMPLETE_SENTINEL if kind == "whole" else want_from
    specs: list[FetchSpec] = []
    for s in syms:
        cov = None if refresh else coverage.get(s)
        if cov is None:
            specs.append(FetchSpec(s, want_from, None, whole_from, want_to))
            continue
        history = cov[0] > want_from
        tail = (want_to - cov[1]).days > stale_days
        tail_from = max(want_from, cov[1] - timedelta(days=overlap))
        if kind == "range":
            if history:
                specs.append(FetchSpec(s, want_from, cov[0], want_from, cov[0]))
            if tail:
                specs.append(FetchSpec(s, tail_from, None, tail_from, want_to))
        elif history or tail:
            since = tail_from if kind == "monolithic" and not history else want_from
            specs.append(FetchSpec(s, since, None, whole_from if since == want_from else since,
                                   want_to))
    return specs


def _split_at(specs: list[FetchSpec], day: date) -> tuple[list[FetchSpec], list[FetchSpec]]:
    """Split each spec into its part up to ``day`` and its part after it (pure).

    Insider history comes from SEC's quarterly bulk data sets, which end with the newest
    published quarter (``day``): only the days after it need the per-filing scraper.

    Args:
        specs: Fetch plans from :func:`_coverage_plan`.
        day: The last day the bulk source covers.

    Returns:
        ``(upto, after)`` — the bounded parts on or before ``day`` and the parts after it.
    """
    upto: list[FetchSpec] = []
    after: list[FetchSpec] = []
    for sp in specs:
        end = sp.until or sp.rec_to
        if sp.since <= day:
            stop = min(end, day)
            upto.append(FetchSpec(sp.symbol, sp.since, stop, sp.rec_from, stop))
        if end > day:
            start = max(sp.since, day + timedelta(days=1))
            after.append(FetchSpec(sp.symbol, start, sp.until, start, sp.rec_to))
    return upto, after


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


def _run_altdata(ds: str, engine, settings, syms: list[str], start: date,
                 *, refresh: bool = False, stale_days: int | None = None) -> int:
    """Ingest what ``syms`` are missing of one alt-data dataset; return rows written.

    :func:`_coverage_plan` reads the ``altdata_coverage`` ledger and plans, name by
    name, only the missing history back to ``start`` and the days since the name's
    last refresh. Short interest is batched (one FINRA file per settlement serves
    every name). Insider history up to SEC's newest published quarterly data set
    comes from those bulk files, only the days after it from the per-filing scraper.
    The per-name sources (that scraper, Yahoo's ratings and price targets, SEC
    companyfacts) go one symbol at a time, so one failure is isolated and reported.
    Every piece widens the symbol's ledger window as soon as it is fetched, so an
    interrupted run resumes where it stopped.

    Args:
        ds: Dataset name (``insider`` | ``ratings`` | ``price_target`` |
            ``short_interest`` | ``fundamentals``).
        engine: Database engine.
        settings: Resolved settings (for exchange overrides).
        syms: Symbols to ingest.
        start: The deepest day of history wanted.
        refresh: Ignore the ledger and re-fetch every symbol's whole window.
        stale_days: Days a symbol's newest data may age before it is refreshed
            (``None``: :data:`_STALE_DAYS`).

    Returns:
        Number of rows written for this dataset.
    """
    from trading_simple.db import (
        load_coverage,
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
    specs = _coverage_plan(
        syms, load_coverage(engine, ds), start, date.today(), kind=kind,
        overlap=_OVERLAP_DAYS[ds], refresh=refresh,
        stale_days=_STALE_DAYS[ds] if stale_days is None else stale_days)
    todo = len({sp.symbol for sp in specs})
    history = sum(sp.until is not None for sp in specs)
    typer.echo(f"altdata {ds}: {len(syms) - todo} of {len(syms)} symbol(s) up to date; "
               f"fetching {todo} ({history} with missing history)")
    if not specs:
        return 0

    if ds == "short_interest":
        from trading_simple.providers.finra import iter_short_interest_bulk
        return _ingest_batched(ds, engine, specs, iter_short_interest_bulk,
                               upsert_short_interest)
    if ds == "insider":
        from trading_simple.providers.sec import (
            bulk_published_until,
            fetch_insider,
            iter_insider_bulk,
        )
        try:
            last = bulk_published_until()
        except Exception as exc:  # noqa: BLE001 — SEC unreachable: retry next run, record nothing
            typer.echo(f"altdata insider: SEC unreachable ({exc}); skipped, run again later")
            return 0
        upto, after = _split_at(specs, last) if last else ([], specs)
        written = _ingest_batched(ds, engine, upto, iter_insider_bulk, upsert_insider)
        return written + _ingest_per_symbol(
            ds, engine, after, lambda s, since, until: fetch_insider(s, since, until=until),
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


def _ingest_batched(ds: str, engine, specs: list[FetchSpec], iterate, upsert) -> int:
    """Fetch windows from a whole-market source, recording each window once it is done.

    Specs sharing a ``(since, until)`` window share its downloads (one FINRA file per
    settlement, one SEC zip per quarter, each serving every symbol); rows are upserted
    per file and released, so memory stays bounded. A window that fails (an outage) is
    reported and left unrecorded, so the next run fetches it again.

    Args:
        ds: Dataset name (for log lines and the ledger).
        engine: Database engine.
        specs: Fetch plans from :func:`_coverage_plan`.
        iterate: ``(symbols, since, until=) -> iter[(label, rows)]`` bulk provider.
        upsert: ``(engine, rows) -> n`` writer.

    Returns:
        Total rows written.
    """
    from trading_simple.db import upsert_coverage

    written = 0
    for (since, until), group in _group_by_window(specs).items():
        try:
            for label, rows in iterate([sp.symbol for sp in group], since, until=until):
                written += upsert(engine, rows)
                typer.echo(f"altdata {ds} (bulk) {label}: {len(rows)} rows")
        except Exception as exc:  # noqa: BLE001 — an outage must not abort the run
            typer.echo(f"altdata {ds} (bulk) {since}..{until or 'today'}: error ({exc})")
            continue
        upsert_coverage(engine, ds, [_cov_row(sp) for sp in group])
    typer.echo(f"altdata {ds} (bulk): {written} rows for {len({sp.symbol for sp in specs})} "
               "symbol(s)")
    return written


def _ingest_per_symbol(ds: str, engine, specs: list[FetchSpec], fetch, upsert) -> int:
    """Fetch + upsert each spec one at a time, recording its ledger coverage at once.

    A failing symbol is logged and skipped — and NOT recorded, so it retries next
    run; a symbol that fetches cleanly (even with zero rows) IS recorded (with its
    spec's ``rec_from``/``rec_to``) right away, so a genuinely event-free name is not
    re-attempted forever and an interrupted run resumes after the last finished name.

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

    total = 0
    for i, sp in enumerate(specs, 1):
        try:
            rows = fetch(sp.symbol, sp.since, sp.until)
        except Exception as exc:  # noqa: BLE001 — isolate a bad symbol
            typer.echo(f"altdata {ds} [{i}/{len(specs)}] {sp.symbol}: 0 rows ({exc})")
            continue
        total += upsert(engine, rows)
        upsert_coverage(engine, ds, [{"symbol": sp.symbol, "covered_from": sp.rec_from,
                                      "covered_to": sp.rec_to, "rows": len(rows)}])
        typer.echo(f"altdata {ds} [{i}/{len(specs)}] {sp.symbol}: {len(rows)} rows "
                   f"(since {sp.since})")
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
                       session=d.get("session", "regular"), exchanges=settings.exchanges,
                       resample=d.get("resample"))
    if d.get("features"):  # attach alt-data feature matrices for alt-data strategies
        from trading_simple import features
        features.attach_features(panel, engine, d["features"], settings=settings)
    typer.echo(f"optimize {strategy}: {len(syms)} symbols @ {panel.bar}, "
               f"study={strat.optuna['study']}, objectives={list(opt.objectives_for(strat))}")
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
