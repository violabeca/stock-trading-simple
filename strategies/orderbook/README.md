# orderbook — what do the FREE historical order books really contain?

**Research only: there is no `strategy.py`, `config.yaml` or study.** The goal is to explore strategies
that use order-book data (US stocks or crypto). The framework's engine runs on bars, so this folder
answers the data question first: which free level-2 history exists, and what is really in the files.

| file | what |
|---|---|
| `orderbook_free_data.ipynb` | downloads ONE day from OKX, Bybit and IEX, decodes the three formats into one message stream, replays the books, and compares file formats, message rates, depth and defects (4 figures, a `Checks` table) |
| `orderbook_free.py` | its code: downloads, decoders (JSON lines; pcap / pcapng with IEX-TP and DEEP 1.0), `replay` (rebuild the book, check it, sample it), `summary`, `compare_mid` (loaded by path as `ob`; tested in `tests/test_orderbook_free.py`) |
| `iex_data_explained_marimo.py` | a **marimo** notebook: what the free IEX feed holds (every message type, one stock's book as a heat map, the market-wide status messages) and what a strategy could extract from it, each piece with a "how it is computed" figure: liquidity state, queue imbalance and microprice, order-flow imbalance, signed trades, halts / short-sale restriction / retail interest; ends with what testing it would cost (7 figures, a `Checks` table). Pick the day, the MB and the stocks in a form; the pictures then react to the stock, the ladder width, the moment of Fig 4 and the bucket of Fig 5 |
| `iex_features.py` | its code: `status_tables`, `quotes` / `book_features`, `ofi` / `bucket_ofi`, `sign_trades`, `ladder`, `rhythm_frame` (tested in `tests/test_iex_features.py`; uses `orderbook_free.decode_iex`) |

Data: `var/orderbook_free/` (git-ignored). Run `orderbook_free_data` top to bottom: about 290 MB the first
time (OKX 130 MB, Bybit 60 MB, a 100 MB prefix of IEX's day), 1-2 minutes. The IEX notebook runs with
`uv run marimo edit strategies/orderbook/iex_data_explained_marimo.py`: its form downloads a 200 MB prefix
by default (about a minute), and changing a control afterwards does not download again. Stored files are
never fetched again.

## The free sources (probed 2026-10-19)

No key and no login for any of them. Dates were checked against real files, not copied from docs.

| source | what it is | coverage | size per day | access |
|---|---|---|---|---|
| **OKX** | tick-by-tick L2, 400 levels a side for spot, swaps and dated futures; a level is `[price, size, orders]`; swaps also have 5,000-level files since about the turn of 2025/26 | spot from at least 2023-02, swaps checked from 2024-06, through at least 2026-10-17 | 45-525 MB zipped per instrument | `GET https://www.okx.com/api/v5/public/market-data-history?module=4&instType=SPOT&instIdList=BTC-USDT&dateAggrType=daily&begin=<ms>&end=<ms>` returns the file URL. Swaps and futures: `instFamilyList=BTC-USDT`; `module=5` = the 5,000-level swaps; limit 5 requests / 2 s |
| **Bybit** | the exchange's websocket stream: a snapshot, then deltas with an update id; 200 levels a side (500 for USDT perpetuals until mid-August 2025) | USDT perpetuals from 2023-01, spot from about 2025-04-30, through 2026-10-18 | 30-380 MB zipped per symbol | `https://quote-saver.bycsi.com/orderbook/{spot\|linear}/BTCUSDT/<day>_BTCUSDT_ob200.data.zip` |
| **Binance** `bookDepth` | NOT a book: cumulative depth within ±0.2% … ±5% of the mid every ~30 s | 2023-01-01 to 2026-10-18, 1,000+ futures symbol folders | 0.5 MB | `https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-<day>.zip` (USD-M and COIN-M). No spot book files; `bookTicker` (best bid / ask) ends 2024-03-30 |
| **IEX** HIST | IEX's own price-level book (DEEP) and best bid / offer (TOPS) in a pcap(ng) capture; IEX is ~2-3% of US volume | DEEP daily from 2017-05-15 to 2026-10-16; old links still download | 3-15 GB gzipped, the WHOLE market, no per-symbol file | file index `https://iextrading.com/api/1.0/hist`; HTTP Range works, so a prefix of the gzip is the start of the day |
| **Nasdaq ITCH samples** | every order add / cancel / replace / execute, all names | about 17 full days: 2019-2021, 2025-11-28 and 2025-12-08..12, 2026-06-12 | 3.5-18 GB gzipped | `https://emi.nasdaq.com/ITCH/Nasdaq%20ITCH/` |
| **Databento** | all venues, order-by-order or 10 levels | new accounts get $125 of credit (6 months); their example: order-by-order data for 199 energy stocks over 175 days | priced per GB | not free beyond the credit |

Not tested hands-on: Tardis.dev (docs: the first day of each month is free without a key, the rest is
paid), Hyperliquid (requester-pays S3 bucket of L2 snapshots) and Alpaca (free best bid / ask quotes
only). The OKX and Bybit order-book files hold no trades; OKX (history API `module=1`) and Bybit
(`public.bybit.com`) publish them separately. IEX's capture carries its own trade reports.

## IEX DEEP in practice (the 2026-10-19 file; its first 200 MB cover 07:06 to 09:32:54 New York)

Measured with `iex_data_explained_marimo.py`'s helpers, spec = IEX DEEP v1.08:

* **Messages**: 6.78 M, of which 97.6% price-level updates (displayed size at one price changed), 1.3%
  trades (86,604; 72,542 after the open, in 4,485 symbols) and the rest status messages. 11,097
  symbols had a book update; 13,297 got a status spin at 07:06:52 (IEX opens for orders at 08:00).
* **The book** is IEX's displayed orders only, summed per price. It is sparse and wide: over the quote
  states after the open INTC's median spread is 5 ticks (5.3 bps), SPY's 23 ticks (3.0 bps); SPY has
  about 20 levels in total, a few near the touch and clusters of 80-share orders ~100 ticks away. 39% (SPY) to 85% (AAPL) of trades print strictly INSIDE
  the displayed spread: midpoint or hidden liquidity that DEEP does not show.
* **Opening**: at 09:30:00 the queued book of every stock is published within a second (99.985% of
  13,268 books opened within 1 ms; SQBR and ACIV, both halted, 23 s later). SPY's book at 09:30:00 is
  two placeholder quotes (750 / 796) and a real book one second later.
* **Trades** carry flags, not a side: after the open 32.7% of trades (37.8% of shares) are intermarket
  sweeps and 44.7% of trades (7.2% of shares) are odd lots; median size 99 shares.
* **Short-sale restriction (Rule 201)**: 762 symbols in effect at the start of the day, 42 more switched
  on after 08:00 (23 within a second of the open: stocks that opened 10% below yesterday's close).
  **Halts**: 41 symbols (37 before 08:00, 1 pre-market, 3 after the open); 6 symbols halted on IEX only
  (5 of them also halted market-wide).
  **Retail interest flag**: 3,989 flag messages in 456 symbols in the first ~170 s (the most flagged are
  ETFs: NVD, EFA, SQQQ, TQQQ ...); at the end 98 symbols had a sell-side and 45 a buy-side flag.
* **Auctions, official prices, the security directory** cover IEX-listed securities only, and the
  only ones are three test symbols (ZIEXT, ZEXIT, ZXIET): nothing for real stocks.
* **Order-flow imbalance** against the mid change of the same 0.1 s bucket (first 150 s after the open):
  r = +0.48 (SPY), +0.19 (AAPL), +0.20 (NVDA), +0.11 (INTC); weaker in longer buckets. A description of
  one window, not a forecast and not a result.
* **Cost**: the whole file is 12 GB gzipped (median 3.1 GB over the years, up to 20 GB), the first
  3 minutes of the regular session about 1 MB per second (an estimate from message counts); a gzip file can only be read from its start, so the open is
  cheap and the close is not. Decoding runs at ~0.6 M messages per second in pure Python.

## Parked until the data question is answered

* Which horizon? Slow depth features (imbalance per minute or longer) could ride on Binance's 30 s
  `bookDepth` and the bar engine; millisecond microstructure needs an event-driven simulator outside it.
* Trades next to the books (OKX module 1, Bybit `public.bybit.com`) for order-flow features.
* Binance's real book exists only in `collectors/binance_depth.py`'s own recording.
