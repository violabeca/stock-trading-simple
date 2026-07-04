"""The ``tsim`` command line: schema, screening, backfill, optimization, coverage.

Notebooks hand long-running work to these commands as subprocesses (see
:mod:`trading_simple.jobs`), so interactive exploration and unattended runs
share one code path. Every command prints plain, parseable lines.
"""

from __future__ import annotations

import logging
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
    typer.echo(f"optimize {strategy}: {len(syms)} symbols, study={strat.optuna['study']}")
    result = opt.run_study(strat, panel, settings=settings,
                           n_trials=n_trials, timeout_s=timeout_s)
    typer.echo(f"study {result.study_name}: {len(result.trials)} trials total, "
               f"{len(result.best_trials)} on the Pareto front")


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
