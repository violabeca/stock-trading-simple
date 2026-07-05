"""Universe building: find symbols worth backfilling and give the set a name.

Pipeline (driven interactively by ``notebooks/01_universe_screener.ipynb``):

1. **Fetch** every US-listed stock from the free Nasdaq screener API.
2. **Screen** down to liquid common stocks (drop warrants/funds/preferreds,
   apply market-cap and price floors).
3. **Pick** the top names, capped per sector so no universe is a hidden sector bet.
4. **Persist**: write the list to ``config/universes.yaml`` (merged into
   ``settings.universe`` at load time) and register ``securities`` rows so
   price backfills have valid foreign keys.

:func:`rank_by_dollar_volume` ranks names by trading activity using prices
already in the database — the cheap way to decide which symbols deserve slow
1-minute IBKR history (``tsim subset`` pins the top of that ranking as its own
named universe). :func:`import_universes` copies universes built by the
sibling ``stock-trading`` repo — same shared database — so its richer
characteristic universes (HighOctane, Trailblazers, ...) are usable here too.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
import yaml
from sqlalchemy import Engine, text

from trading_simple.config import UNIVERSES_FILE
from trading_simple.db import Security, upsert_securities

log = logging.getLogger(__name__)

_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"),
    "Accept": "application/json",
}
#: Non-common issues excluded by keyword in the security NAME.
EXCLUDE_NAME_KEYWORDS = (
    "warrant", "unit", "preferred", " notes", "depositary", "right", "convertible",
    "%", "etf", "etn", "fund", "trust", "spac", "acquisition corp", "index",
)


def fetch_listed_stocks(
    exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX"), *, timeout: int = 30
) -> pd.DataFrame:
    """Fetch all US-listed stocks with name, price, market cap, sector, volume.

    One request per exchange against the (unofficial, keyless) Nasdaq screener
    API. A single failing exchange only degrades coverage; the call raises only
    when every exchange fails.

    Args:
        exchanges: Exchange feeds to query.
        timeout: Per-request timeout in seconds.

    Returns:
        One row per symbol with columns ``symbol, name, price, market_cap,
        volume, sector, industry, exchange`` (duplicates keep the richest row).

    Raises:
        RuntimeError: If no exchange returned data (endpoint down or blocking).
    """
    frames, errors = [], []
    for exch in exchanges:
        try:
            resp = requests.get(_SCREENER_URL, headers=_HEADERS, timeout=timeout,
                                params={"download": "true", "limit": "10000", "exchange": exch})
            resp.raise_for_status()
            rows = (resp.json() or {}).get("data", {}).get("rows") or []
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{exch}: {exc}")
            log.warning("screener fetch failed for %s: %s", exch, exc)
            continue
        if rows:
            frames.append(pd.DataFrame(rows).assign(exchange=exch))
    if not frames:
        raise RuntimeError(
            f"Nasdaq screener returned no data ({'; '.join(errors) or 'empty payloads'}). "
            "The unofficial endpoint may be down — retry later, or hand-edit "
            "config/universes.yaml instead."
        )
    raw = pd.concat(frames, ignore_index=True)
    col = lambda c: raw.get(c, pd.Series([None] * len(raw)))  # noqa: E731
    out = pd.DataFrame({
        "symbol": raw["symbol"].str.upper().str.strip(),
        "name": col("name"),
        "price": raw["lastsale"].map(_to_float),
        "market_cap": raw["marketCap"].map(_to_float),
        "volume": col("volume").map(_to_float),
        "sector": col("sector").replace("", None),
        "industry": col("industry").replace("", None),
        "exchange": raw["exchange"],
    })
    return (out.sort_values("market_cap", ascending=False)
               .drop_duplicates(subset="symbol").reset_index(drop=True))


def screen_candidates(
    df: pd.DataFrame, *, min_market_cap: float = 2e9, min_price: float = 5.0,
    exclude_name_keywords: tuple[str, ...] = EXCLUDE_NAME_KEYWORDS,
) -> pd.DataFrame:
    """Filter a listing down to liquid common stocks, ranked by market cap.

    Pure function. Keeps plain 1–5 letter symbols, drops names containing a
    non-common-stock keyword, and applies market-cap and price floors.

    Args:
        df: Output of :func:`fetch_listed_stocks`.
        min_market_cap: Minimum market cap in dollars.
        min_price: Minimum last price in dollars.
        exclude_name_keywords: Lower-case keywords that disqualify a name.

    Returns:
        The surviving rows, sorted by market cap descending.
    """
    out = df[df["symbol"].str.fullmatch(r"[A-Z]{1,5}")]
    name_l = out["name"].fillna("").str.lower()
    drop = pd.Series(False, index=out.index)
    for kw in exclude_name_keywords:
        drop |= name_l.str.contains(kw, regex=False)
    out = out[~drop]
    out = out[(out["market_cap"] >= min_market_cap) & (out["price"] >= min_price)]
    return out.sort_values("market_cap", ascending=False).reset_index(drop=True)


def liquid_leaders(screened: pd.DataFrame, *, n: int = 100, sector_cap: float = 0.25) -> list[str]:
    """Pick the top ``n`` names by market cap, greedily capped per sector.

    The cap (default: no sector may exceed 25% of the universe) keeps a
    backtested edge from being a hidden sector bet.

    Args:
        screened: Output of :func:`screen_candidates` (sorted by market cap).
        n: Universe size.
        sector_cap: Maximum fraction of the universe from one sector.

    Returns:
        The selected symbols, biggest first.
    """
    cap = max(1, int(sector_cap * n))
    counts: defaultdict[str, int] = defaultdict(int)
    keep: list[str] = []
    for _, row in screened.iterrows():
        sec = row.get("sector") or "Unknown"
        if counts[sec] >= cap:
            continue
        keep.append(row["symbol"])
        counts[sec] += 1
        if len(keep) >= n:
            break
    return keep


def rank_by_dollar_volume(
    engine: Engine, symbols: list[str], *, bar: str = "1d", days: int = 90
) -> pd.DataFrame:
    """Rank symbols by average daily dollar volume, from prices already stored.

    The cheap "which names are actually active?" lens — used to pick which
    symbols deserve slow 1-minute IBKR backfills.

    Args:
        engine: Database engine.
        symbols: Candidate symbols (must have ``bar`` data to appear).
        bar: Bar size to measure on.
        days: Trailing window in calendar days.

    Returns:
        DataFrame indexed by symbol with ``dollar_volume`` (mean close×volume),
        sorted descending.
    """
    sql = text("""
        SELECT symbol, avg(close * volume) AS dollar_volume
        FROM prices_ohlcv
        WHERE symbol = ANY(:symbols) AND bar = :bar
          AND ts >= now() - make_interval(days => :days)
        GROUP BY symbol ORDER BY dollar_volume DESC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"symbols": symbols, "bar": bar, "days": days}).fetchall()
    return pd.DataFrame(rows, columns=["symbol", "dollar_volume"]).set_index("symbol")


def write_universe(name: str, symbols: list[str], *, path: Path = UNIVERSES_FILE) -> Path:
    """Persist ``name -> symbols`` into ``config/universes.yaml`` (merging).

    Existing universes in the file are kept; the named one is replaced.
    ``load_settings`` merges this file into ``settings.universe``.

    Args:
        name: Universe name, e.g. ``"LiquidLeaders"``.
        symbols: The member symbols (stored sorted).
        path: Target YAML file (overridable for tests).

    Returns:
        The path written.
    """
    existing = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    existing = existing.get("universe", existing) if isinstance(existing, dict) else {}
    existing[name] = sorted(symbols)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Machine-generated universes (notebooks/01_universe_screener.ipynb,\n"
              "# tsim import-universes, tsim subset). Merged into settings.universe\n"
              "# by trading_simple.config.load_settings.\n")
    path.write_text(header + yaml.safe_dump({"universe": existing}, sort_keys=True))
    return path


def import_universes(src: Path | str, *, names: list[str] | None = None,
                     overwrite: bool = False, path: Path = UNIVERSES_FILE) -> dict[str, int]:
    """Copy named universes from another repo's ``universes.yaml`` into this one.

    The sibling ``stock-trading`` platform builds richer, characteristic
    universes against the same shared database (HighOctane, Trailblazers,
    SteadyEddies, CheapQuality — see its ``notebooks/00_universe_screener.ipynb``).
    This merges chosen lists into this repo's ``config/universes.yaml`` so any
    strategy here can reference them by name. Lists are copied as a snapshot,
    not a live link — re-run after the sibling regenerates its file.

    Args:
        src: Source ``universes.yaml`` (``{universe: {name: [symbols]}}``).
        names: Universes to import (default: every one not yet defined here).
        overwrite: Replace universes that already exist in the target file.
        path: Target YAML file (overridable for tests).

    Returns:
        ``{name: symbol_count}`` for the universes actually written; names
        skipped because they already exist (and ``overwrite`` is off) are absent.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        KeyError: If a requested name is missing from ``src`` (message lists
            what the source offers).
    """
    raw = yaml.safe_load(Path(src).read_text()) or {}
    source = raw.get("universe", raw) if isinstance(raw, dict) else {}
    missing = [n for n in names or [] if n not in source]
    if missing:
        raise KeyError(f"{missing} not in {src}. Available: {sorted(source)}")
    current = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
    current = current.get("universe", current) if isinstance(current, dict) else {}
    written = {}
    for name in names or sorted(source):
        if name in current and not overwrite:
            continue
        write_universe(name, source[name], path=path)
        written[name] = len(source[name])
    return written


def register_securities(engine: Engine, screened: pd.DataFrame) -> int:
    """Upsert ``securities`` rows so price backfills have valid foreign keys.

    Args:
        engine: Database engine.
        screened: Frame with at least a ``symbol`` column; ``name``,
            ``exchange``, ``sector``, ``industry`` are carried through if present.

    Returns:
        Number of rows written.
    """
    opt = lambda v: str(v) if pd.notna(v) and v != "" else None  # noqa: E731
    items = [
        Security(symbol=row["symbol"], name=opt(row.get("name")),
                 exchange=opt(row.get("exchange")), sector=opt(row.get("sector")),
                 industry=opt(row.get("industry")))
        for _, row in screened.iterrows()
    ]
    return upsert_securities(engine, items)


def _to_float(value: object) -> float:
    """Parse screener strings like ``'$205.19'`` or ``'4,965,598,000'`` to float."""
    s = str(value or "").strip().replace("$", "").replace(",", "").replace("%", "")
    if not s or s in {"NA", "N/A", "--"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")
