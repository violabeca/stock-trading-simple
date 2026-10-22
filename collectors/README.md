# collectors/ — 24/7 market-data collection

Scripts that record data nobody sells retroactively, for later backtests. They run apart from
the backtest framework: each script declares its own dependencies inline (PEP 723), so `uv run`
needs nothing from this repo's environment, and `docker-compose.yml` runs them next to their own
TimescaleDB, on a laptop or a VM.

Order books that exchanges already publish as free daily files (OKX, Bybit, IEX) need no collector:
see `strategies/orderbook/`. A collector is for books nobody publishes, such as Binance's.

| script | records | table |
|---|---|---|
| `binance_depth.py` | Binance USDⓈ-M futures order books, top 1000 levels a side every 5 s, 10 pairs | `market_depth` in `market_depth_db` |

## binance_depth.py

Ported from `trading_strategy/src/data_extraction/stream_order_book.py` (the newest of the old
collectors). It writes the same `market_depth` table (time, symbol, side, price, quantity),
so existing analysis code reads old and new data alike. The module docstring documents the
sync procedure and three data-quality fixes over the old version:

- continuity is checked on `pu`, so syncs no longer fail;
- the stored levels are clipped to the range the REST snapshot really covered;
- out-of-sync or stale books are no longer saved.

**Volume.** At the defaults, 10 pairs × up to 2 × 1000 levels every 5 s is ~340 M rows a day.
Measured on a 9-minute local sample (2026-10-02): 83 bytes/row before compression and 3.1 after,
so about **1 GB/day, ~30 GB/month**. The full one-hour chunks a long run produces should compress
at least as well. On top of that sit the newest ~3 hours, not compressed yet (one-hour chunks,
compressed two hours after they close): ~4 GB. The old collector's layout took 14.4 bytes/row,
~7 GB/day at its 2000 levels. To store less, cut `--levels`, `--interval` or `DEPTH_SYMBOLS`.

### Run locally

```bash
cd collectors
cp .env.example .env        # set POSTGRES_PASSWORD; DB_PORT=5433 if your main cluster owns 5432
docker compose up -d        # TimescaleDB + collector; the table is created on first start
docker compose logs -f binance_depth
docker compose down         # stop (data kept); `down -v` deletes it
```

Without Docker, against any TimescaleDB:
`uv run collectors/binance_depth.py --dsn postgresql://USER:PW@HOST:PORT/DB [--symbols BTCUSDT,ETHUSDT] [--levels 200]`.
An existing `market_depth` table is checked, never altered.

### Check it

```bash
docker compose exec timescaledb psql -U postgres -d market_depth_db -c "
  SELECT symbol, count(DISTINCT time) AS snapshots, max(time) AS last,
         round(count(*)::numeric / count(DISTINCT time)) AS rows_per_snapshot
  FROM market_depth WHERE time > now() - interval '10 minutes' GROUP BY 1 ORDER BY 1"
docker compose exec timescaledb psql -U postgres -d market_depth_db -c "
  SELECT pg_size_pretty(hypertable_size('market_depth'))"
```

### Notes

- **Binance refuses US IPs** (HTTP 451). The script checks this at startup and exits with that
  reason, so a cloud VM must sit in a non-US region. Tokyo (`asia-northeast1`) is closest to
  Binance's matching engine; any EU region works.
- Postgres is published on `127.0.0.1` only. Reach a remote one through an SSH tunnel, never
  by opening the port.
- `uv` resolves the script's dependencies as of the `exclude-newer` date in its header, so every
  machine runs the same versions. Bump the date to upgrade.
