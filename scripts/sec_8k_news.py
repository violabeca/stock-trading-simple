"""Company press releases from SEC 8-K exhibits (EX-99.1) into the shared ``news`` table.

A company's own press release (results, guidance, deals, executive changes) is attached to its
8-K as exhibit EX-99.1, free on EDGAR, with the filing's acceptance time to the second. This fills
what the other news stores lack: press releases before March 2025 and a first-party,
precisely timed copy of the wire ones afterwards.

Steps (all resumable, EDGAR paced to ~8 requests a second):

1. every 8-K of every universe name in the window, with its TRUE acceptance time
   (``strategies/overnight_open/research.fetch_filings``: EDGAR's JSON stamp is late by New
   York's UTC offset for about half the filers and is corrected from the index pages);
2. each filing's index page -> its EX-99.1 document -> plain text, headline and a body prefix;
3. one ``news`` row per filing (``source = 'sec_8k'``, ``url`` = the exhibit, ``symbols`` = every
   universe ticker sharing the filer, ``published_at = known_at`` = acceptance time).

  uv run python scripts/sec_8k_news.py --since 2025-01-01 --until 2025-06-30
  uv run python scripts/sec_8k_news.py --since 2025-01-01 --until 2025-06-30 \
      --symbols MU,AAPL --no-db
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import re
import time

import pandas as pd
from sqlalchemy import text

from trading_simple.config import PROJECT_ROOT, load_settings
from trading_simple.db import make_engine
from trading_simple.providers import sec

OUT = PROJECT_ROOT / "var" / "news_8k"
BODY_CHARS = 10_000
ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
HREF = re.compile(r'href="([^"]+)"')
BLOCK = re.compile(r"</?(?:p|br|div|tr|li|ul|ol|h[1-6]|table|center)\b[^>]*>", re.I)
NOISE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
SKIP = re.compile(r"^(exhibit\b|news release|press release|for immediate release|contacts?\b|"
                  r"investor|media\b|source:|\(|\[|page \d)", re.I)
DATELINE = re.compile(r"/prnewswire/|\(business wire\)|globe newswire|\bglobenewswire\b|"
                      r"^[A-Z][A-Za-z .,'&-]{2,40}, [A-Z][a-z]{2,8}\.? \d{1,2}, 20\d\d", re.I)


def research():
    """Load ``strategies/overnight_open/research.py`` by path (it is not an importable package)."""
    spec = importlib.util.spec_from_file_location(
        "overnight_open_research", PROJECT_ROOT / "strategies/overnight_open/research.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plain_text(raw: str) -> str:
    """Plain text of an HTML exhibit: block tags become line breaks, entities are decoded.

    Args:
        raw: The document as served.

    Returns:
        Text with one paragraph per line and no blank runs.
    """
    t = html.unescape(re.sub(r"<[^>]+>", "", BLOCK.sub("\n", NOISE.sub(" ", raw))))
    lines = (re.sub(r"[ \t\xa0]+", " ", ln).strip() for ln in t.split("\n"))
    return "\n".join(ln for ln in lines if ln)


def headline_of(body: str) -> str:
    """The press release's title: the first real line, joined with its continuation lines.

    Letterhead (``News Release``, contacts) and datelines are skipped; a title that the HTML
    broke over several lines (``... RESULTS FOR THE`` / ``SECOND QUARTER ...``) is re-joined.

    Args:
        body: Output of :func:`plain_text`.

    Returns:
        Up to 300 characters; the first line when nothing better is found.
    """
    lines = body.split("\n")
    for i, ln in enumerate(lines[:25]):
        if len(ln.split()) >= 4 and not SKIP.match(ln) and not DATELINE.search(ln):
            parts, n = [], 0
            for nxt in lines[i:i + 4]:
                if parts and (len(nxt.split()) <= 2 or SKIP.match(nxt) or DATELINE.search(nxt)):
                    break
                parts.append(nxt)
                n += len(nxt.split())
                if nxt.endswith((".", "!", "?", ":")) or n >= 12:
                    break
            return " ".join(parts)[:300]
    return (lines[0] if lines else "")[:300]


def exhibit_of(cik: int, accession: str) -> str | None:
    """URL of a filing's EX-99.1 (press release) document, from its EDGAR index page.

    Args:
        cik: The filer's CIK.
        accession: Accession number with dashes.

    Returns:
        The absolute URL, or ``None`` when the filing has no EX-99.1.
    """
    url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
           f"{accession}-index.htm")
    page = sec._get(url, as_json=False).decode("utf-8", "replace")
    for row in ROW.findall(page):
        cells = CELL.findall(row)
        kind = re.sub(r"<[^>]+>|&nbsp;|\s", "", cells[3]) if len(cells) >= 4 else ""
        if re.fullmatch(r"EX-99(\.0?1)?", kind):
            m = HREF.search(cells[2])
            if m:
                return "https://www.sec.gov" + re.sub(r"^/ix\?doc=", "", m.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--since", required=True, help="first filing date YYYY-MM-DD")
    ap.add_argument("--until", required=True, help="last filing date YYYY-MM-DD")
    ap.add_argument("--symbols", help="comma-separated tickers (default: the whole universe)")
    ap.add_argument("--no-db", action="store_true", help="fetch and cache only, do not insert")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    symbols = (args.symbols.split(",") if args.symbols else load_settings().all_symbols())

    r = research()
    filings = r.fetch_filings(symbols, since=args.since, until=args.until,
                              path=OUT / f"filings_{args.since}_{args.until}.pkl")
    f = filings[(filings.form == "8-K") & filings.accession.notna()].copy()
    f["cik"] = f.url.str.extract(r"/data/(\d+)/")[0].astype(int)
    by_acc = (f.groupby("accession").agg(cik=("cik", "first"), ts=("ts", "first"),
                                         symbols=("symbol", lambda s: sorted(set(s))),
                                         items=("items", "first")).reset_index())
    print(f"{len(by_acc)} 8-K filings by {f.symbol.nunique()} names", flush=True)

    cache = OUT / f"exhibits_{args.since}_{args.until}.pkl"
    have = pd.read_pickle(cache) if cache.exists() else {}
    t0, n_new = time.monotonic(), 0
    for i, row in enumerate(by_acc.itertuples(), 1):
        if row.accession in have:
            continue
        try:
            url = exhibit_of(row.cik, row.accession)
            raw = sec._get(url, as_json=False).decode("utf-8", "replace") if url else ""
            body = plain_text(raw) if raw else ""
        except Exception as exc:  # noqa: BLE001 - one bad filing must not stop the sweep
            print("skip", row.accession, type(exc).__name__, exc, flush=True)
            continue
        have[row.accession] = {"url": url, "headline": headline_of(body) if body else "",
                               "body": body[:BODY_CHARS], "chars": len(body)}
        n_new += 1
        if n_new % 200 == 0:
            pd.to_pickle(have, cache)
            per = (time.monotonic() - t0) / n_new
            found = sum(bool(v["url"]) for v in have.values())
            print(f"  {i}/{len(by_acc)} filings | {found} with EX-99.1 | {per:.1f}s each | "
                  f"~{per * (len(by_acc) - i) / 60:.0f} min left", flush=True)
    pd.to_pickle(have, cache)

    rows = [{"source": "sec_8k", "url": ex["url"], "headline": ex["headline"], "body": ex["body"],
             "symbols": list(r_.symbols), "published_at": r_.ts.tz_convert("UTC").to_pydatetime(),
             "known_at": r_.ts.tz_convert("UTC").to_pydatetime()}
            for r_ in by_acc.itertuples() if (ex := have.get(r_.accession)) and ex["url"]]
    print(f"{len(rows)} press releases (EX-99.1) of {len(by_acc)} filings", flush=True)
    if args.no_db or not rows:
        return
    sql = text("INSERT INTO news (source, url, headline, body, symbols, published_at, known_at) "
               "VALUES (:source, :url, :headline, :body, CAST(:symbols AS text[]), "
               ":published_at, :known_at) "
               "ON CONFLICT (url) DO NOTHING")
    with make_engine().begin() as c:
        res = c.execute(sql, rows)
    print(f"inserted {res.rowcount} new rows (existing URLs skipped)")


if __name__ == "__main__":
    main()
