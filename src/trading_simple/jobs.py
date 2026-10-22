"""Hand long-running work from notebooks to ``tsim`` CLI subprocesses.

Notebooks stay responsive and the heavy lifting runs through the exact same
code path as unattended runs. Two parallelism rules:

* **yfinance (daily)** fans out freely: symbols are split into shards, each
  shard runs as its own ``tsim backfill`` subprocess in a thread pool.
* **IBKR (intraday)** must NOT fan out (the request cap is account-wide):
  one streamed subprocess handles the whole symbol list over one connection.
* **Optuna** workers are separate ``tsim optimize`` processes appending to the
  same journal-backed study — safe by design, no coordination needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from trading_simple.config import PROJECT_ROOT

_CLI = [sys.executable, "-m", "trading_simple.cli"]
DEFAULT_TRACKER_PATH = PROJECT_ROOT / "var" / "backfill_1m_tracker.csv"
TRACKER_COLUMNS = ["symbol", "bar", "since", "until", "status", "bars", "updated_at"]


def backfill_cmd(
    symbols: list[str], *, bar: str, since: str, until: str | None = None,
    provider: str | None = None, extended_hours: bool | None = None,
) -> list[str]:
    """Compose a ``tsim backfill`` command line (pure; unit-testable).

    Args:
        symbols: Symbols for this invocation.
        bar: Bar size.
        since: First day, ``YYYY-MM-DD``.
        until: Optional last day.
        provider: Optional provider override.
        extended_hours: Optional IBKR extended-hours override.

    Returns:
        The argv list.
    """
    cmd = [*_CLI, "backfill", "--symbols", ",".join(symbols), "--bar", bar, "--since", since]
    if until:
        cmd += ["--until", until]
    if provider:
        cmd += ["--provider", provider]
    if extended_hours is not None:
        cmd += ["--extended-hours" if extended_hours else "--regular-only"]
    return cmd


def run_backfill(
    symbols: list[str], *, bar: str = "1d", since: str, until: str | None = None,
    workers: int = 8, shard_size: int = 30,
) -> pd.DataFrame:
    """Backfill daily bars via parallel yfinance CLI shards.

    Args:
        symbols: All symbols to load.
        bar: Bar size (yfinance path supports ``"1d"`` only).
        since: First day, ``YYYY-MM-DD``.
        until: Optional last day.
        workers: Concurrent subprocesses.
        shard_size: Symbols per subprocess.

    Returns:
        One row per shard: symbols, ok flag, rows written.
    """
    shards = [symbols[i:i + shard_size] for i in range(0, len(symbols), shard_size)]

    def run(shard: list[str]) -> dict:
        proc = subprocess.run(backfill_cmd(shard, bar=bar, since=since, until=until,
                                           provider="yfinance"),
                              capture_output=True, text=True, cwd=PROJECT_ROOT)
        return {"symbols": f"{shard[0]}..{shard[-1]} ({len(shard)})",
                "ok": proc.returncode == 0,
                "rows": parse_rows_written(proc.stdout),
                "error": proc.stderr.strip()[-200:] if proc.returncode else ""}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [f.result() for f in as_completed(pool.submit(run, s) for s in shards)]
    done = pd.DataFrame(rows)
    print(f"backfill {bar}: {done['rows'].sum()} rows across {len(shards)} shards "
          f"({int(done['ok'].sum())}/{len(shards)} ok)")
    return done


def altdata_cmd(
    dataset: str, *, since: str, universe: str | None = None,
    symbols: list[str] | None = None, refresh: bool = False, stale_days: int | None = None,
) -> list[str]:
    """Compose a ``tsim altdata`` command line (pure; unit-testable).

    Args:
        dataset: Dataset name (``insider`` | ``ratings`` | ``price_target`` |
            ``short_interest`` | ``fundamentals`` | ``all``).
        since: The deepest day of history wanted, ``YYYY-MM-DD``.
        universe: Universe name (used when ``symbols`` is omitted), or ``ALL``.
        symbols: Explicit symbol list (overrides ``universe``).
        refresh: Ignore the coverage ledger and re-fetch every symbol's whole window.
        stale_days: Days a symbol's newest data may age before it is refreshed
            (``None``: the dataset's default, ``cli._STALE_DAYS``).

    Returns:
        The argv list.
    """
    cmd = [*_CLI, "altdata", "--dataset", dataset, "--since", since]
    if symbols:
        cmd += ["--symbols", ",".join(symbols)]
    elif universe:
        cmd += ["--universe", universe]
    if refresh:
        cmd += ["--refresh"]
    if stale_days is not None:
        cmd += ["--stale-days", str(stale_days)]
    return cmd


def run_altdata(
    dataset: str, *, since: str, universe: str | None = None,
    symbols: list[str] | None = None, refresh: bool = False, stale_days: int | None = None,
) -> None:
    """Ingest what is missing of one alt-data dataset via ONE streamed ``tsim altdata``.

    Never fans out: SEC EDGAR and FINRA rate-limit per IP, so one paced process
    is both the safe and the correct option (the same rule as IBKR). Per-symbol
    progress lines stream live into the notebook.

    Args:
        dataset: Dataset name (``insider`` | ``ratings`` | ``price_target`` |
            ``short_interest`` | ``fundamentals`` | ``all``).
        since: The deepest day of history wanted, ``YYYY-MM-DD``.
        universe: Universe name (used when ``symbols`` is omitted), or ``ALL``.
        symbols: Explicit symbol list (overrides ``universe``).
        refresh: Ignore the coverage ledger and re-fetch every symbol's whole window.
        stale_days: Days a symbol's newest data may age before it is refreshed
            (``None``: the dataset's default, ``cli._STALE_DAYS``).
    """
    cmd = altdata_cmd(dataset, since=since, universe=universe, symbols=symbols,
                      refresh=refresh, stale_days=stale_days)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=PROJECT_ROOT)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()


def run_ibkr_backfill(
    symbols: list[str], *, since: str, until: str | None = None, bar: str = "1m",
    port: int | None = None, client_id: int = 10, extended_hours: bool | None = None,
    on_symbol: Callable[[str, int], None] | None = None,
) -> pd.DataFrame:
    """Backfill intraday bars via ONE streamed IBKR CLI subprocess.

    Never fans out — IBKR's historical-request cap is account-wide, so a single
    paced connection is both the safe and the fast option. Per-symbol progress
    lines stream live into the notebook as they happen.

    Args:
        symbols: All symbols to load (one connection handles them in turn).
        since: First day, ``YYYY-MM-DD``.
        until: Optional last day.
        bar: Bar size, default ``"1m"``.
        port: Override the Gateway port (else settings/env).
        client_id: IBKR client id for this run.
        extended_hours: Include pre/post-market bars (else settings default).
        on_symbol: Optional ``(symbol, bar_count)`` callback fired as each
            symbol's line streams in — e.g. :func:`run_ibkr_sweep` uses this to
            checkpoint its progress ledger without waiting for the whole batch.

    Returns:
        One row per symbol with the bar count (0-bar symbols are visible).
    """
    env = os.environ | {"TRADING_IBKR__CLIENT_ID": str(client_id)}
    if port is not None:
        env["TRADING_IBKR__PORT"] = str(port)
    cmd = backfill_cmd(symbols, bar=bar, since=since, until=until,
                       provider="ibkr", extended_hours=extended_hours)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=PROJECT_ROOT, env=env)
    counts: dict[str, int] = {}
    for line in proc.stdout:
        print(line, end="")
        parsed = parse_symbol_bars(line)
        if parsed:
            counts[parsed[0]] = parsed[1]
            if on_symbol is not None:
                on_symbol(*parsed)
    proc.wait()
    return pd.DataFrame({"symbol": list(counts), "bars": list(counts.values())}).set_index("symbol")


def seed_tracker(symbols: list[str], *, bar: str, since: str, until: str) -> pd.DataFrame:
    """Build a fresh, alphabetical, all-``pending`` tracker frame for one sweep target."""
    syms = sorted(set(symbols))
    return pd.DataFrame({"symbol": syms, "bar": bar, "since": since, "until": until,
                         "status": "pending", "bars": 0, "updated_at": ""})


def merge_tracker(existing: pd.DataFrame, seed: pd.DataFrame) -> pd.DataFrame:
    """Carry over ledger rows whose ``(bar, since, until)`` target still matches ``seed``.

    A symbol that is new, or whose stored target window no longer matches (e.g.
    ``--until`` moved forward), comes back ``"pending"`` — so widening the sweep's
    window re-closes gaps instead of trusting stale coverage.

    Args:
        existing: Previously persisted tracker (``TRACKER_COLUMNS``, possibly empty).
        seed: This run's freshly seeded, all-``pending`` target frame.

    Returns:
        The merged tracker, alphabetical, one row per symbol in ``seed``.
    """
    if existing.empty:
        return seed
    carry = existing.merge(seed[["symbol", "bar", "since", "until"]],
                           on=["symbol", "bar", "since", "until"], how="inner").set_index("symbol")
    out = seed.set_index("symbol")
    keep = carry.index.intersection(out.index)
    cols = ["status", "bars", "updated_at"]
    out.loc[keep, cols] = carry.loc[keep, cols]
    return out.reset_index()[TRACKER_COLUMNS]


def mark_symbol(tracker: pd.DataFrame, symbol: str, bars: int) -> pd.DataFrame:
    """Set one symbol's row to ``"done"`` (``bars > 0``) or ``"empty"``, timestamped now."""
    status = "done" if bars > 0 else "empty"
    now = datetime.now(UTC).isoformat(timespec="seconds")
    tracker.loc[tracker["symbol"] == symbol, ["status", "bars", "updated_at"]] = [status, bars, now]
    return tracker


def read_tracker(path: str | Path) -> pd.DataFrame:
    """Load a tracker CSV, or an empty ``TRACKER_COLUMNS`` frame if it doesn't exist yet."""
    path = Path(path)
    return (pd.read_csv(path, keep_default_na=False, dtype={"bars": int}) if path.exists()
            else pd.DataFrame(columns=TRACKER_COLUMNS))


def write_tracker(tracker: pd.DataFrame, path: str | Path) -> None:
    """Persist the tracker CSV, creating its parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tracker.to_csv(path, index=False)


def covered_symbols(cov: pd.DataFrame, *, since: str, until: str,
                    tolerance_days: int = 5) -> dict[str, int]:
    """Filter a :func:`trading_simple.data.coverage` frame to symbols already
    spanning ``[since, until]`` — pure, so it needs no live database to test.

    ``until``'s edge tolerates ``tolerance_days`` of lag (a weekend, a holiday, or
    today's bar not closed yet) so "one day behind" isn't treated as a gap.

    Args:
        cov: A ``coverage(engine, symbols=..., bar=...)`` result (``symbol``,
            ``bars``, ``first_ts``, ``last_ts`` columns).
        since: First day, ``YYYY-MM-DD``.
        until: Last day, ``YYYY-MM-DD``.
        tolerance_days: Slack allowed on the ``until`` edge.

    Returns:
        ``{symbol: bars}`` for symbols that already fully cover the window.
    """
    if cov.empty:
        return {}
    since_ts = pd.Timestamp(since, tz="UTC")
    until_ts = pd.Timestamp(until, tz="UTC") - pd.Timedelta(days=tolerance_days)
    ok = cov[(cov["first_ts"] <= since_ts) & (cov["last_ts"] >= until_ts)]
    return dict(zip(ok["symbol"], ok["bars"], strict=True))


def run_ibkr_sweep(
    symbols: list[str], *, bar: str = "1m", since: str, until: str | None = None,
    tracker_path: str | Path = DEFAULT_TRACKER_PATH, port: int | None = None,
    client_id: int = 10, extended_hours: bool | None = None, retry_empty: bool = False,
    skip_covered: bool = True,
) -> pd.DataFrame:
    """Backfill every ``symbols`` name, alphabetically, via ONE paced IBKR connection,
    checkpointing progress to a CSV ledger so a multi-day sweep survives being killed.

    Resuming (same ``bar``/``since``/``until``) skips symbols the ledger already marks
    ``"done"`` (and ``"empty"`` unless ``retry_empty``), so re-running the same command
    is always safe and picks up exactly where it left off — the same idempotent-UPSERT
    philosophy as the rest of the backfill path, applied at symbol granularity.

    Args:
        symbols: All symbols to sweep (alphabetical order is enforced here).
        bar: Bar size, default ``"1m"``.
        since: First day, ``YYYY-MM-DD``.
        until: Last day (default: today).
        tracker_path: Where to read/write the progress ledger.
        port: Override the Gateway port (else settings/env).
        client_id: IBKR client id for this run.
        extended_hours: Include pre/post-market bars (else settings default).
        retry_empty: Also re-attempt symbols the ledger marked ``"empty"`` (0 bars
            returned last time — either no IBKR history that far back, or a
            transient fetch error; the ledger can't tell the two apart).
        skip_covered: Before fetching a ``pending`` symbol, check the DATABASE's own
            stored coverage (:func:`covered_symbols`) and mark it ``"done"`` without
            an IBKR call if it already spans the window — e.g. names an earlier
            notebook run already backfilled. IBKR is still the only source ever
            re-requested for a genuine gap; this only skips symbols needing no
            request at all.

    Returns:
        The final tracker frame.
    """
    until = until or date.today().isoformat()
    seed = seed_tracker(symbols, bar=bar, since=since, until=until)
    tracker = merge_tracker(read_tracker(tracker_path), seed)

    todo = {"pending", "empty"} if retry_empty else {"pending"}
    pending = tracker.loc[tracker["status"].isin(todo), "symbol"].tolist()

    if skip_covered and pending:
        from trading_simple.config import load_settings
        from trading_simple.data import coverage as _coverage
        from trading_simple.db import make_engine

        engine = make_engine(load_settings())
        cov = _coverage(engine, symbols=pending, bar=bar)
        covered = covered_symbols(cov, since=since, until=until)
        if covered:
            now = datetime.now(UTC).isoformat(timespec="seconds")
            for sym, bars in covered.items():
                tracker.loc[tracker["symbol"] == sym, ["status", "bars", "updated_at"]] = \
                    ["done", bars, now]
            pending = [s for s in pending if s not in covered]
            print(f"sweep: {len(covered)} symbols already covered in the DB, skipped")

    write_tracker(tracker, tracker_path)  # persist the (re)seeded state before fetching anything
    print(f"sweep {bar} {since}..{until}: {len(pending)}/{len(tracker)} symbols to fetch "
          f"({len(tracker) - len(pending)} already done)")
    if not pending:
        return tracker

    def checkpoint(symbol: str, bars: int) -> None:
        nonlocal tracker
        tracker = mark_symbol(tracker, symbol, bars)
        write_tracker(tracker, tracker_path)

    run_ibkr_backfill(pending, since=since, until=until, bar=bar, port=port, client_id=client_id,
                      extended_hours=extended_hours, on_symbol=checkpoint)
    return tracker


def ibkr_preflight(port: int | None = None, symbol: str = "AAPL") -> dict:
    """Check that an IB Gateway is reachable and serves historical data.

    Never raises — returns ``{"ok": bool, "message": str}`` so a notebook can
    assert with a readable hint.

    Args:
        port: Gateway port to probe (else settings default).
        symbol: Probe symbol.

    Returns:
        Status dict.
    """
    from trading_simple.config import load_settings

    cfg = load_settings().ibkr
    port = port or cfg.port
    try:
        import asyncio

        try:  # Jupyter kernels already run an event loop; nest it so connect() works
            asyncio.get_running_loop()
            import nest_asyncio

            nest_asyncio.apply()
        except RuntimeError:
            pass
        from ib_async import IB, Stock

        ib = IB()
        ib.connect(cfg.host, port, clientId=99, timeout=10)
        try:
            head = ib.reqHeadTimeStamp(Stock(symbol, "SMART", "USD"), "TRADES", useRTH=True)
        finally:
            ib.disconnect()
        return {"ok": True, "message": f"Gateway on port {port} OK; {symbol} history since {head}"}
    except Exception as exc:  # noqa: BLE001 — a preflight reports, never raises
        return {"ok": False,
                "message": f"IB Gateway not usable on port {port}: {exc}. "
                           "Is the Gateway running and logged in? Paper GW = 4002."}


def run_optimize(strategy: str, *, n_trials: int | None = None, study: str | None = None,
                 timeout_s: float | None = None, workers: int = 1) -> None:
    """Run/resume a strategy's Optuna study in ``tsim optimize`` subprocess(es).

    ``workers > 1`` launches N identical processes against the SAME study; the
    journal storage makes concurrent appends safe. Re-run any time to resume.

    Args:
        strategy: Strategy folder name.
        n_trials: Trials per worker (default: strategy config).
        study: Study name override.
        timeout_s: Per-worker time budget in seconds.
        workers: Number of parallel optimizer processes.
    """
    cmd = [*_CLI, "optimize", "--strategy", strategy]
    if n_trials is not None:
        cmd += ["--n-trials", str(n_trials)]
    if study:
        cmd += ["--study", study]
    if timeout_s is not None:
        cmd += ["--timeout-s", str(timeout_s)]

    def run(i: int) -> None:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, cwd=PROJECT_ROOT)
        for line in proc.stdout:
            print(line if workers == 1 else f"[w{i}] {line}", end="")
        proc.wait()

    if workers == 1:
        run(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, range(workers)))


def parse_rows_written(stdout: str) -> int:
    """Extract N from the final ``prices: N rows written`` CLI line (0 if absent)."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("prices:") and line.endswith("rows written"):
            return int(line.split()[1])
    return 0


def parse_symbol_bars(line: str) -> tuple[str, int] | None:
    """Parse a ``prices SYM BAR: N bars`` CLI line into ``(symbol, count)``."""
    parts = line.split()
    if len(parts) >= 5 and parts[0] == "prices" and parts[4] == "bars":
        try:
            return parts[1], int(parts[3])
        except ValueError:
            return None
    return None
