"""Ticker news from Massive (ex-Polygon) into the shared ``news`` and ``news_sentiment`` tables.

The free key (``MASSIVE_API_KEY`` in ``.env``) serves two years of history at 5 calls a minute:
Motley Fool, GlobeNewswire (press releases), Benzinga and Investing.com. One UTC day at a time,
all tickers at once (~250 articles on a weekday). Every article carries its tickers, a publish
time to the minute, a ~440-character description and ``insights`` = the vendor's own per-ticker
sentiment (positive / neutral / negative / mixed with a reason).

1. each day's articles are cached as JSON in ``var/massive_news/`` (a re-run asks for nothing twice);
2. one ``news`` row per article that names a universe ticker (``source = 'massive/<publisher>'``,
   ``url`` = the article, ``body`` = the description, ``published_at = known_at``); an article whose URL
   is already stored (Benzinga items also come through Alpaca) keeps its old row;
3. one ``news_sentiment`` row per (article, universe ticker) with ``backend = 'massive'``,
   ``label`` as given and ``score`` = +1 / -1 for positive / negative, 0 for neutral and mixed.

  uv run python scripts/massive_news.py --since 2024-12-01 --until 2024-12-31
  uv run python scripts/massive_news.py --since 2024-12-01 --until 2024-12-31 --no-db
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date, datetime, timedelta

import requests
from dotenv import dotenv_values
from sqlalchemy import text

from trading_simple.config import PROJECT_ROOT, load_settings
from trading_simple.db import make_engine

URL = "https://api.massive.com/v2/reference/news"
OUT = PROJECT_ROOT / "var" / "massive_news"
GAP_S = 12.5  # free tier: 5 calls a minute
SCORE = {"positive": 1.0, "negative": -1.0}
_last = [0.0]


def get(key: str, params: dict) -> dict:
    """One paced, retried call to the news endpoint.

    Args:
        key: The API key.
        params: Query parameters (the key is added here).

    Returns:
        The decoded JSON body.

    Raises:
        RuntimeError: When the call still fails after five tries.
    """
    for attempt in range(5):
        time.sleep(max(0.0, GAP_S - (time.monotonic() - _last[0])))
        r = requests.get(URL, params={**params, "apiKey": key}, timeout=60)
        _last[0] = time.monotonic()
        if r.status_code == 200:
            return r.json()
        print(f"  HTTP {r.status_code} {r.text[:100]} - retry {attempt + 1}", flush=True)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"Massive news call failed: {params}")


def fetch_day(key: str, day: date) -> list[dict]:
    """All articles published on one UTC day, from the cache or the API.

    Args:
        key: The API key.
        day: The UTC day.

    Returns:
        The raw article dicts.
    """
    path = OUT / f"{day}.json"
    if path.exists():
        return json.loads(path.read_text())
    params = {"published_utc.gte": str(day), "published_utc.lt": str(day + timedelta(days=1)),
              "limit": 1000, "order": "asc"}
    rows = []
    while True:
        body = get(key, params)
        rows += body.get("results", [])
        nxt = body.get("next_url")
        if not nxt:
            break
        params = {"cursor": re.search(r"cursor=([^&]+)", nxt).group(1)}
    path.write_text(json.dumps(rows))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--since", required=True, help="first UTC day YYYY-MM-DD")
    ap.add_argument("--until", required=True, help="last UTC day YYYY-MM-DD (inclusive)")
    ap.add_argument("--no-db", action="store_true", help="fetch and cache only, do not insert")
    args = ap.parse_args()
    key = dotenv_values(PROJECT_ROOT / ".env").get("MASSIVE_API_KEY")
    if not key:
        raise SystemExit("MASSIVE_API_KEY is empty in .env")
    OUT.mkdir(parents=True, exist_ok=True)
    universe = set(load_settings().all_symbols())

    articles, day = [], date.fromisoformat(args.since)
    while day <= date.fromisoformat(args.until):
        got = fetch_day(key, day)
        articles += got
        print(f"{day}: {len(got)} articles", flush=True)
        day += timedelta(days=1)

    news, senti = [], {}
    for a in articles:
        syms = sorted(set(a.get("tickers") or []) & universe)
        if not syms or not a.get("article_url"):
            continue
        at = datetime.fromisoformat(a["published_utc"].replace("Z", "+00:00"))
        news.append({"source": f"massive/{a['publisher']['name']}", "url": a["article_url"],
                     "headline": a["title"], "body": a.get("description") or "", "symbols": syms,
                     "published_at": at, "known_at": at})
        senti[a["article_url"]] = [(i["ticker"], i["sentiment"]) for i in a.get("insights") or []
                                   if i.get("ticker") in universe and i.get("sentiment")]
    print(f"{len(articles)} articles, {len(news)} name a universe ticker, "
          f"{sum(map(len, senti.values()))} ticker sentiments", flush=True)
    if args.no_db or not news:
        return
    ins = text("INSERT INTO news (source, url, headline, body, symbols, published_at, known_at) "
               "VALUES (:source, :url, :headline, :body, CAST(:symbols AS text[]), :published_at, "
               ":known_at) ON CONFLICT (url) DO NOTHING")
    ins_s = text("INSERT INTO news_sentiment (news_id, symbol, backend, score, label, scored_at) "
                 "VALUES (:news_id, :symbol, 'massive', :score, :label, now()) "
                 "ON CONFLICT (news_id, symbol, backend) DO NOTHING")
    with make_engine().begin() as c:
        res = c.execute(ins, news)
        ids = dict(c.execute(text("SELECT url, id FROM news WHERE url = ANY(:u)"),
                             {"u": list(senti)}).all())
        rows = [{"news_id": ids[u], "symbol": s, "label": lab, "score": SCORE.get(lab, 0.0)}
                for u, pairs in senti.items() if u in ids for s, lab in pairs]
        res_s = c.execute(ins_s, rows) if rows else None
    print(f"inserted {res.rowcount} new news rows (existing URLs skipped) and "
          f"{res_s.rowcount if res_s else 0} sentiment rows")


if __name__ == "__main__":
    main()
