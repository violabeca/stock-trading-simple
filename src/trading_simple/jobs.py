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
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from trading_simple.config import PROJECT_ROOT

_CLI = [sys.executable, "-m", "trading_simple.cli"]


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
    symbols: list[str] | None = None,
) -> list[str]:
    """Compose a ``tsim altdata`` command line (pure; unit-testable).

    Args:
        dataset: Dataset name (``insider`` | ``ratings`` | ``short_interest`` | ``all``).
        since: First day, ``YYYY-MM-DD``.
        universe: Universe name (used when ``symbols`` is omitted).
        symbols: Explicit symbol list (overrides ``universe``).

    Returns:
        The argv list.
    """
    cmd = [*_CLI, "altdata", "--dataset", dataset, "--since", since]
    if symbols:
        cmd += ["--symbols", ",".join(symbols)]
    elif universe:
        cmd += ["--universe", universe]
    return cmd


def run_altdata(
    dataset: str, *, since: str, universe: str | None = None,
    symbols: list[str] | None = None,
) -> None:
    """Ingest one alt-data dataset via a single streamed ``tsim altdata`` subprocess.

    Never fans out: SEC EDGAR and FINRA rate-limit per IP, so one paced process
    is both the safe and the correct option (the same rule as IBKR). Per-symbol
    progress lines stream live into the notebook.

    Args:
        dataset: Dataset name (``insider`` | ``ratings`` | ``short_interest`` | ``all``).
        since: First day, ``YYYY-MM-DD``.
        universe: Universe name (used when ``symbols`` is omitted).
        symbols: Explicit symbol list (overrides ``universe``).
    """
    cmd = altdata_cmd(dataset, since=since, universe=universe, symbols=symbols)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=PROJECT_ROOT)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()


def run_ibkr_backfill(
    symbols: list[str], *, since: str, until: str | None = None, bar: str = "1m",
    port: int | None = None, client_id: int = 10, extended_hours: bool | None = None,
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
    proc.wait()
    return pd.DataFrame({"symbol": list(counts), "bars": list(counts.values())}).set_index("symbol")


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
