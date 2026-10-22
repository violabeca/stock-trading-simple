#!/usr/bin/env python3
"""Collect Finviz's free analyst-action table for every name the DB ingests ratings for.

A comparison source for Yahoo's ``upgrades_downgrades`` feed, which stopped for some heavily
covered names in October 2024 (see ``notebooks/03_altdata_update.py`` §4c). Finviz's free
quote page shows each name's LAST 20 analyst actions (date only; most target-only revisions
omitted), so this is a partial history, never a replacement for a full feed.

Scraping is against Finviz's terms of use: one connection, one page every 6-10 s, and the run
STOPS at the first sign of blocking (403, 429, 503 or a Cloudflare challenge). Resumable: each
symbol's parsed rows land in ``var/finviz/raw/<SYMBOL>.csv`` (an empty file = no table), and a
rerun skips symbols already there. Names whose Yahoo feed is oldest go first.

Every run (also ``--assemble``) rewrites two CSVs with the DB tables' columns, for comparison
with ``analyst_ratings`` / ``analyst_price_targets``:

- ``var/finviz/analyst_ratings_finviz.csv``: action Upgrade/Downgrade/Initiated/Resumed/
  Reiterated -> upgrade/downgrade/initiate/initiate/maintain; ``grade_from`` is empty when
  Finviz shows one grade.
- ``var/finviz/analyst_price_targets_finviz.csv``: only rows with a target; ``pt_action``
  Raises/Lowers/Maintains from "$a -> $b", Announces for a lone target on an initiation, empty
  for a lone target otherwise (its prior is unknown).

``known_at`` is the end of the action's UTC day, as for Yahoo rows.

Usage:
    uv run python scripts/finviz_ratings.py                # crawl what is left (~5 h for 2,300)
    uv run python scripts/finviz_ratings.py --limit 10     # a few names
    uv run python scripts/finviz_ratings.py --assemble     # only rebuild the two CSVs
"""

from __future__ import annotations

import argparse
import html
import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "var" / "finviz"
URL = "https://finviz.com/stock"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/129.0 Safari/537.36",
           "Accept": "text/html,application/xhtml+xml", "Accept-Language": "en-US,en;q=0.9"}
ACTIONS = {"Upgrade": "upgrade", "Downgrade": "downgrade", "Initiated": "initiate",
           "Resumed": "initiate", "Reiterated": "maintain"}
RAW_COLS = ["date", "action", "firm", "rating", "target"]
_ROW = re.compile(r'<tr class="styled-row[^"]*">(.*?)</tr>', re.S)
_TD = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_TAG = re.compile(r"<[^>]+>")
_BLOCK = ("cf-chl", "Just a moment...", "Access denied", "captcha")


class Blocked(Exception):
    """Finviz answered with a block, a rate limit or a challenge page."""


def parse(page: str) -> pd.DataFrame:
    """The analyst-action table of a Finviz quote page.

    Args:
        page: The page's HTML.

    Returns:
        ``RAW_COLS`` as shown (``date`` parsed); empty when the page has no table.
    """
    start = page.find("js-table-ratings")
    if start < 0:
        return pd.DataFrame(columns=RAW_COLS)
    table = page[start:page.find("</table>", start)]
    rows = [[html.unescape(_TAG.sub("", td)).strip() for td in _TD.findall(r)]
            for r in _ROW.findall(table)]
    df = pd.DataFrame([r[:5] for r in rows if len(r) >= 5], columns=RAW_COLS)
    today = datetime.now(ZoneInfo("America/New_York")).date()  # Finviz writes today's as "Today"
    df["date"] = df["date"].replace({"Today": today.strftime("%b-%d-%y"),
                                     "Yesterday": (today - timedelta(days=1)).strftime("%b-%d-%y")})
    df["date"] = pd.to_datetime(df["date"], format="%b-%d-%y").dt.date
    return df


def fetch(session: requests.Session, symbol: str) -> pd.DataFrame:
    """One symbol's table.

    Args:
        session: HTTP session (keeps Finviz's cookies).
        symbol: Ticker.

    Returns:
        The parsed table (empty for an unknown ticker or a name without analyst actions).

    Raises:
        Blocked: On 403 / 429 / 503 or a challenge page.
    """
    r = session.get(URL, params={"t": symbol, "p": "d"}, headers=HEADERS, timeout=30)
    if r.status_code in (403, 429, 503) or any(b in r.text[:20000] for b in _BLOCK):
        raise Blocked(f"{symbol}: HTTP {r.status_code}, retry-after={r.headers.get('retry-after')}")
    if r.status_code == 404:
        return pd.DataFrame(columns=RAW_COLS)
    r.raise_for_status()
    return parse(r.text)


def symbols() -> list[str]:
    """Every symbol the ratings ledger covers, the stalest Yahoo history first.

    Returns:
        Symbols ordered by their newest ``analyst_ratings`` action (none = last).
    """
    from sqlalchemy import text

    from trading_simple.db import make_engine

    sql = text("""SELECT l.symbol FROM altdata_coverage l
                  LEFT JOIN analyst_ratings r USING (symbol) WHERE l.dataset = 'ratings'
                  GROUP BY l.symbol ORDER BY max(r.rated_at) NULLS LAST, l.symbol""")
    with make_engine().connect() as conn:
        return list(conn.execute(sql).scalars())


def _split(cell: str) -> tuple[str | None, str | None]:
    """``"a → b"`` -> (a, b); ``"b"`` -> (None, b); ``""`` -> (None, None)."""
    parts = [p.strip() for p in cell.split("→")] if cell else []
    return (parts[0], parts[1]) if len(parts) == 2 else (None, parts[0] if parts else None)


def _price(value: str | None) -> float | None:
    """``"$1,234.5"`` -> 1234.5 (None when empty)."""
    return float(value.replace("$", "").replace(",", "")) if value else None


def assemble(out: Path = OUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn every stored raw table into the two DB-shaped CSVs.

    Args:
        out: Output root.

    Returns:
        ``(ratings, price_targets)`` as written.
    """
    frames = [pd.read_csv(f, dtype=str, keep_default_na=False).assign(symbol=f.stem)
              for f in sorted((out / "raw").glob("*.csv"))]  # a name without a table: header only
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=[*RAW_COLS, "symbol"])
    known = pd.to_datetime(raw["date"]).dt.tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59,
                                                                              seconds=59)
    grades = raw["rating"].map(_split)
    targets = raw["target"].map(_split)
    common = {"symbol": raw["symbol"], "firm": raw["firm"], "rated_at": raw["date"],
              "known_at": known, "source": "finviz"}
    ratings = pd.DataFrame({**common, "action": raw["action"].map(ACTIONS),
                            "grade_from": grades.str[0], "grade_to": grades.str[1]})
    prior, current = targets.str[0].map(_price), targets.str[1].map(_price)
    pt_action = pd.Series(None, index=raw.index, dtype=object)
    pt_action[prior.notna() & (current > prior)] = "Raises"
    pt_action[prior.notna() & (current < prior)] = "Lowers"
    pt_action[prior.notna() & (current == prior)] = "Maintains"
    pt_action[prior.isna() & raw["action"].isin(["Initiated", "Resumed"])] = "Announces"
    pts = pd.DataFrame({**common, "pt_action": pt_action, "pt_current": current,
                        "pt_prior": prior})[current.notna()]
    cols_r = ["symbol", "firm", "action", "grade_from", "grade_to", "rated_at", "known_at",
              "source"]
    cols_p = ["symbol", "firm", "pt_action", "pt_current", "pt_prior", "rated_at", "known_at",
              "source"]
    ratings[cols_r].to_csv(out / "analyst_ratings_finviz.csv", index=False)
    pts[cols_p].to_csv(out / "analyst_price_targets_finviz.csv", index=False)
    return ratings[cols_r], pts[cols_p]


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; see the module docstring.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).

    Returns:
        Process exit code (2 when Finviz blocked the run).
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--limit", type=int, help="fetch at most N symbols this run")
    p.add_argument("--pause", type=float, nargs=2, default=(6.0, 10.0), metavar=("MIN", "MAX"),
                   help="seconds between requests, drawn uniformly (default 6 10)")
    p.add_argument("--assemble", action="store_true", help="only rebuild the two CSVs")
    args = p.parse_args(argv)
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    code = 0
    if not args.assemble:
        todo = [s for s in symbols() if not (raw / f"{s}.csv").exists()][:args.limit]
        print(f"{datetime.now():%H:%M} {len(todo)} symbols to fetch", flush=True)
        session = requests.Session()
        for i, sym in enumerate(todo, 1):
            try:
                df = fetch(session, sym)
            except Blocked as e:
                print(f"BLOCKED after {i - 1} pages this run: {e}", flush=True)
                code = 2
                break
            except (requests.RequestException, ValueError) as e:  # network, or an unparsable page
                print(f"{sym}: {e} (retried next run)", flush=True)
            else:
                df.to_csv(raw / f"{sym}.csv", index=False)
            if i % 50 == 0:
                print(f"{datetime.now():%H:%M} {i}/{len(todo)}", flush=True)
            time.sleep(random.uniform(*args.pause))
    ratings, pts = assemble()
    print(f"{ratings['symbol'].nunique()} symbols: {len(ratings):,} rating rows, "
          f"{len(pts):,} price-target rows -> {OUT}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
