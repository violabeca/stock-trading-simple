"""Free historical order books from OKX, Bybit and IEX: download one day, decode, compare.

Loaded by path from ``orderbook_free_data.ipynb`` (like ``options_free.py``); nothing else imports
it. No keys, no sign-ups. Three sources, all LEVEL 2 (price levels with their total size, never
single orders):

* ``okx``: OKX's public history download (``/api/v5/public/market-data-history``, module 4). One
  tar.gz per instrument and UTC day holding one ``.data`` file of JSON lines: a ``snapshot`` of 400
  levels a side, then an ``update`` whenever the book changes (about every 10 ms). A level is
  ``[price, size, number of orders]``; size 0 deletes it.
* ``bybit``: Bybit's public archive (``quote-saver.bycsi.com``). One zip per symbol and UTC day of
  the exchange's websocket stream, JSON lines: a ``snapshot`` (200 levels a side; 500 for USDT
  perpetuals until mid-August 2025), then ``delta`` messages carrying an update id ``u``.
* ``iex``: IEX's free HIST download (index ``iextrading.com/api/1.0/hist``). One capture per
  trading day of IEX-TP / DEEP 1.0 packets for the WHOLE market, 3-15 GB gzipped; classic pcap in
  2017, pcapng since 2019 (the file name says ``.pcap.gz`` either way). Only a PREFIX of the gzip
  is fetched (HTTP Range): the start of the day, so every book builds from empty. IEX alone is
  about 2-3% of US volume.

Every source becomes the same stream of :class:`Msg` (a time, whether it replaces the whole book,
and the ``(price, size)`` levels it sets on each side). :func:`replay` applies a stream to a book,
checks it and samples it on a time grid, so the sources are compared on the same measures.
Downloads are kept under ``STORE`` (git-ignored ``var/``), one file per source and day.

Terms: personal research. IEX's data is under its own terms of use (iextrading.com); read OKX's and
Bybit's before redistributing anything.
"""

from __future__ import annotations

import json
import struct
import tarfile
import time
import zipfile
import zlib
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import requests

STORE = Path(__file__).resolve().parents[2] / "var" / "orderbook_free"
UA = {"User-Agent": "Mozilla/5.0 (research; orderbook_free.py)"}
OKX_API = "https://www.okx.com/api/v5/public/market-data-history"
BYBIT = "https://quote-saver.bycsi.com/orderbook"
IEX_INDEX = "https://iextrading.com/api/1.0/hist"
BANDS_BPS = (5, 10, 25)
#: IEX DEEP message types (first byte), as named in IEX's DEEP 1.0 specification.
TYPE_NAMES = {
    0x53: "System Event", 0x44: "Security Directory", 0x48: "Trading Status",
    0x49: "Retail Liquidity Indicator", 0x4F: "Operational Halt Status",
    0x50: "Short Sale Price Test Status", 0x45: "Security Event",
    0x38: "Price Level Update (buy)", 0x35: "Price Level Update (sell)",
    0x54: "Trade Report", 0x58: "Official Price", 0x42: "Trade Break", 0x41: "Auction Information",
}
_PLU = struct.Struct("<xBq8sIq")        # price level update: flags, time, symbol, size, price
_TRADE = struct.Struct("<xBq8sIqq")     # trade report: flags, time, symbol, size, price, trade id
_PCAP = {b"\xd4\xc3\xb2\xa1": ("<", 1000), b"\x4d\x3c\xb2\xa1": ("<", 1),
         b"\xa1\xb2\xc3\xd4": (">", 1000), b"\xa1\xb2\x3c\x4d": (">", 1)}
_LINK_HEADER = {1: 14, 113: 16, 101: 0}     # Ethernet, Linux cooked capture, raw IP


class Msg(NamedTuple):
    """One book message.

    Attributes:
        t_ns: Time of the message, nanoseconds since the epoch (UTC).
        snap: True when the message replaces the whole book.
        bids: ``(price, size)`` levels set on the bid side: the NEW total size at that price,
            0 meaning the level is gone.
        asks: The same for the ask side.
        seq: The source's update id when it has one (Bybit ``u``), else None.
    """

    t_ns: int
    snap: bool
    bids: list
    asks: list
    seq: int | None = None


# ---------------------------------------------------------------------------------------------
# Downloads.
# ---------------------------------------------------------------------------------------------
def _retry(call, *args, tries: int = 3, **kwargs):
    """``call(*args, **kwargs)``, asked again after a connection error or timeout (these hosts
    sometimes take several seconds to accept a first connection)."""
    for k in range(tries):
        try:
            return call(*args, **kwargs)
        except (requests.ConnectionError, requests.Timeout):
            if k == tries - 1:
                raise
            time.sleep(2**k)


def _download(url: str, dest: Path, *, prefix_mb: float | None = None) -> dict:
    """Stream ``url`` to ``dest`` unless it is stored already.

    With ``prefix_mb`` only the first bytes are asked for (HTTP Range); a server that ignores the
    range is cut off after that many bytes. A partial download lives in ``dest.part`` until it is
    complete, so ``dest`` is never half a file.

    Args:
        url: What to fetch.
        dest: Where to store it.
        prefix_mb: Fetch only this many MB from the start (``None`` = the whole file).

    Returns:
        ``{'path', 'bytes', 'seconds', 'cached'}``.
    """
    want = None if prefix_mb is None else int(prefix_mb * 1_000_000)
    if dest.exists() and (want is None or dest.stat().st_size >= want):
        return {"path": dest, "bytes": dest.stat().st_size, "seconds": 0.0, "cached": True}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    headers = {**UA, **({"Range": f"bytes=0-{want - 1}"} if want else {})}
    t0, size = time.time(), 0
    r = _retry(requests.get, url, headers=headers, stream=True, timeout=(10, 120))
    with r, open(part, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            f.write(chunk[:want - size] if want else chunk)
            size += len(chunk)
            if want and size >= want:
                break
    part.rename(dest)
    return {"path": dest, "bytes": dest.stat().st_size, "seconds": time.time() - t0,
            "cached": False}


def okx_url(day: str, inst: str = "BTC-USDT", market: str = "spot") -> tuple[str, str]:
    """File name and URL of OKX's level-2 file for one UTC day, from OKX's public history API.

    Args:
        day: ``'YYYY-MM-DD'``.
        inst: ``'BTC-USDT'`` (spot instrument, or the instrument FAMILY for a swap).
        market: ``'spot'`` or ``'swap'`` (400 levels a side for both).

    Returns:
        ``(file name, download URL)``.

    Raises:
        FileNotFoundError: OKX lists no such file.
        RuntimeError: The API answered with an error.
    """
    ms = round(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    params = {"module": 4, "dateAggrType": "daily", "begin": ms, "end": ms}
    params |= ({"instType": "SPOT", "instIdList": inst} if market == "spot"
               else {"instType": market.upper(), "instFamilyList": inst})
    for _ in range(5):
        j = _retry(requests.get, OKX_API, params=params, headers=UA, timeout=(10, 30)).json()
        if j.get("code") != "50011":                   # rate limit: 5 requests per 2 s
            break
        time.sleep(2)
    if j.get("code") != "0":
        raise RuntimeError(f"OKX {j.get('code')}: {j.get('msg')}")
    files = [g for d in j["data"] for e in d["details"] for g in e["groupDetails"]]
    hit = [g for g in files if g["filename"].endswith(f"-{day}.tar.gz")]
    if not hit:
        raise FileNotFoundError(f"OKX lists no {market} order book of {inst} for {day}")
    return hit[0]["filename"], hit[0]["url"]


def fetch_okx(day: str, inst: str = "BTC-USDT", market: str = "spot", store: Path = STORE) -> dict:
    """Download OKX's level-2 file of ``inst`` for ``day`` (kept; never fetched twice).

    Args:
        day: ``'YYYY-MM-DD'``.
        inst: See :func:`okx_url`.
        market: ``'spot'`` or ``'swap'``.
        store: Folder for downloads.

    Returns:
        ``{'source', 'url', 'path', 'bytes', 'seconds', 'cached'}``.
    """
    name, url = okx_url(day, inst, market)
    return {"source": "okx", "url": url, **_download(url, store / "okx" / name)}


def bybit_url(day: str, symbol: str = "BTCUSDT", market: str = "spot") -> str:
    """URL of Bybit's order-book archive of one symbol and UTC day.

    The depth in the file name changed: 200 levels now, 500 for USDT perpetuals until mid-August
    2025. The first of the two that exists is used.

    Args:
        day: ``'YYYY-MM-DD'``.
        symbol: ``'BTCUSDT'``.
        market: ``'spot'`` or ``'linear'`` (USDT perpetual).

    Returns:
        The download URL.

    Raises:
        FileNotFoundError: Neither depth exists for that day.
    """
    for depth in (200, 500):
        url = f"{BYBIT}/{market}/{symbol}/{day}_{symbol}_ob{depth}.data.zip"
        if _retry(requests.head, url, headers=UA, timeout=(10, 30)).status_code == 200:
            return url
    raise FileNotFoundError(f"Bybit has no {market} order book of {symbol} for {day}")


def fetch_bybit(day: str, symbol: str = "BTCUSDT", market: str = "spot",
                store: Path = STORE) -> dict:
    """Download Bybit's order-book archive of ``symbol`` for ``day`` (kept; never fetched twice).

    Args:
        day: ``'YYYY-MM-DD'``.
        symbol: ``'BTCUSDT'``.
        market: ``'spot'`` or ``'linear'``.
        store: Folder for downloads.

    Returns:
        ``{'source', 'url', 'path', 'bytes', 'seconds', 'cached'}``.
    """
    url = bybit_url(day, symbol, market)
    return {"source": "bybit", "url": url,
            **_download(url, store / "bybit" / url.rsplit("/", 1)[-1])}


def iex_index(store: Path = STORE) -> dict:
    """IEX's HIST file index ``{'YYYYMMDD': [{'feed', 'version', 'link', 'size', ...}]}``.

    Fetched once a day and kept.

    Args:
        store: Folder for downloads.

    Returns:
        The decoded JSON index.
    """
    path = store / "iex" / f"hist_{date.today()}.json"
    if not path.exists():
        r = _retry(requests.get, IEX_INDEX, headers=UA, timeout=(10, 60))
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        for old in path.parent.glob("hist_*.json"):
            old.unlink()
        path.write_text(r.text)
    return json.loads(path.read_text())


def latest_iex_day(feed: str = "DEEP", store: Path = STORE) -> str:
    """The newest ``'YYYY-MM-DD'`` that IEX's index lists a ``feed`` file for.

    Args:
        feed: ``'DEEP'`` (price-level book) or ``'TOPS'`` (best bid and offer).
        store: Folder for downloads.

    Returns:
        The trading day.
    """
    d = max(k for k, v in iex_index(store).items() if any(f["feed"] == feed for f in v))
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def fetch_iex(day: str, feed: str = "DEEP", mb: float = 100, store: Path = STORE) -> dict:
    """Download the first ``mb`` MB of IEX's gzipped ``feed`` capture of one trading day.

    A prefix is the start of the day (pre-market first): 100 MB of a recent DEEP file reaches
    about the opening minute. The whole file would be 3-15 GB.

    Args:
        day: ``'YYYY-MM-DD'``; must be a trading day IEX lists (see :func:`latest_iex_day`).
        feed: ``'DEEP'``.
        mb: How many MB to fetch.
        store: Folder for downloads.

    Returns:
        ``{'source', 'url', 'path', 'bytes', 'seconds', 'cached', 'full_bytes'}``.

    Raises:
        FileNotFoundError: IEX lists no such file.
    """
    files = [f for f in iex_index(store).get(day.replace("-", ""), []) if f["feed"] == feed]
    if not files:
        raise FileNotFoundError(f"IEX lists no {feed} file for {day}")
    got = _download(files[0]["link"], store / "iex" / f"{day}_{feed}_first{mb:g}MB.pcap.gz",
                    prefix_mb=mb)
    return {"source": "iex", "url": files[0]["link"], "full_bytes": int(files[0]["size"]), **got}


# ---------------------------------------------------------------------------------------------
# Decoders: every source -> a stream of Msg.
# ---------------------------------------------------------------------------------------------
def iter_okx(path: Path) -> Iterator[Msg]:
    """Messages of an OKX file (the tar holds one ``.data`` file of JSON lines).

    Args:
        path: The downloaded ``.tar.gz``.

    Yields:
        One :class:`Msg` per line; the order count of a level is dropped.
    """
    with tarfile.open(path) as tar:
        for line in tar.extractfile(tar.next()):
            m = json.loads(line)
            yield Msg(int(m["ts"]) * 1_000_000, m["action"] == "snapshot",
                      [(float(p), float(s)) for p, s, *_ in m["bids"]],
                      [(float(p), float(s)) for p, s, *_ in m["asks"]])


def iter_bybit(path: Path) -> Iterator[Msg]:
    """Messages of a Bybit archive (the zip holds one ``.data`` file of JSON lines).

    Args:
        path: The downloaded ``.zip``.

    Yields:
        One :class:`Msg` per line, ``seq`` = Bybit's update id.
    """
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        for line in f:
            m = json.loads(line)
            d = m["data"]
            yield Msg(m["ts"] * 1_000_000, m["type"] == "snapshot",
                      [(float(p), float(s)) for p, s in d["b"]],
                      [(float(p), float(s)) for p, s in d["a"]], d["u"])


def raw_head(path: Path, n: int = 3, width: int = 220) -> list[str]:
    """First ``n`` raw lines of an OKX (tar.gz) or Bybit (zip) file, cut at ``width`` characters.

    Args:
        path: The downloaded file.
        n: How many lines.
        width: Cut each line after this many characters.

    Returns:
        The lines as text.
    """
    path = Path(path)
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path) as tar:
            f = tar.extractfile(tar.next())
            lines = [f.readline() for _ in range(n)]
    else:
        with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
            lines = [f.readline() for _ in range(n)]
    return [ln.decode().rstrip()[:width] + (" ..." if len(ln) > width else "") for ln in lines]


def _gunzip(path: Path, chunk: int = 4 << 20) -> Iterator[bytes]:
    """The decompressed bytes of a gzip file, or of a truncated prefix of one, chunk by chunk."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            yield d.decompress(block)


def _tsresol_ns(opts: bytes, endian: str) -> int:
    """Nanoseconds per timestamp tick from a pcapng interface block's options (default: us)."""
    o = 0
    while o + 4 <= len(opts):
        code, size = struct.unpack_from(endian + "HH", opts, o)
        if code == 0:
            break
        if code == 9:
            r = opts[o + 4]
            return max(1, round(10.0 ** (9 - r))) if r < 128 else round(1e9 / 2 ** (r - 128))
        o += 4 + (size + 3) // 4 * 4
    return 1000


def iter_packets(chunks: Iterable[bytes]) -> Iterator[tuple[int, int, bytes]]:
    """Packets of a pcap or pcapng capture handed over as consecutive chunks of bytes.

    A capture cut short ends cleanly after its last whole packet.

    Args:
        chunks: The capture's bytes in order, split anywhere.

    Yields:
        ``(capture time in ns since the epoch, link type, packet bytes)``.

    Raises:
        ValueError: The bytes are neither pcap nor pcapng.
    """
    buf, ng, endian, unit, link = b"", None, "<", 1000, 1
    links, units = [], []
    for chunk in chunks:
        buf, pos = buf + chunk, 0
        if ng is None:
            if len(buf) < 24:
                continue
            ng = buf[:4] == b"\n\r\r\n"
            if not ng:
                if buf[:4] not in _PCAP:
                    raise ValueError(f"not a pcap or pcapng capture (starts with {buf[:4].hex()})")
                endian, unit = _PCAP[buf[:4]]
                link, pos = struct.unpack_from(endian + "I", buf, 20)[0], 24
        while True:
            if not ng:
                if pos + 16 > len(buf):
                    break
                sec, frac, size, _ = struct.unpack_from(endian + "IIII", buf, pos)
                if pos + 16 + size > len(buf):
                    break
                yield sec * 10**9 + frac * unit, link, buf[pos + 16:pos + 16 + size]
                pos += 16 + size
                continue
            if pos + 12 > len(buf):
                break
            if buf[pos:pos + 4] == b"\n\r\r\n":              # section header: sets the byte order
                endian = "<" if buf[pos + 8:pos + 12] == b"\x4d\x3c\x2b\x1a" else ">"
                links, units = [], []
            kind, size = struct.unpack_from(endian + "II", buf, pos)
            if size < 12:
                raise ValueError("corrupt pcapng block")
            if pos + size > len(buf):
                break
            body = buf[pos + 8:pos + size - 4]
            if kind == 1:                                      # interface description
                links.append(struct.unpack_from(endian + "H", body)[0])
                units.append(_tsresol_ns(body[8:], endian))
            elif kind == 6:                                    # enhanced packet
                iface, hi, lo, cap, _ = struct.unpack_from(endian + "IIIII", body)
                yield (hi << 32 | lo) * units[iface], links[iface], body[20:20 + cap]
            pos += size
        buf = buf[pos:]


def iter_iex_messages(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """The DEEP messages of an IEX-TP capture, in order (heartbeat packets carry none).

    Args:
        chunks: The decompressed capture, in order.

    Yields:
        One message's bytes (its first byte is the type).

    Raises:
        ValueError: The packets are not IEX DEEP (protocol id 0x8004) or the link type is unknown.
    """
    for _, link, pk in iter_packets(chunks):
        if link not in _LINK_HEADER:
            raise ValueError(f"unsupported link type {link}")
        off = _LINK_HEADER[link] + (4 if link == 1 and pk[12:14] == b"\x81\x00" else 0)
        if pk[off] >> 4 != 4:
            continue
        off += (pk[off] & 0x0F) * 4 + 8                         # IPv4 header, UDP header
        _, _, proto, _, _, _, count = struct.unpack_from("<BBHIIHH", pk, off)
        if proto != 0x8004:
            raise ValueError(f"not a DEEP capture (IEX-TP message protocol {proto:#x})")
        pos = off + 40                                          # IEX-TP header
        for _ in range(count):
            size = struct.unpack_from("<H", pk, pos)[0]
            yield pk[pos + 2:pos + 2 + size]
            pos += 2 + size


def _msg_time(m: bytes | None) -> int | None:
    """The nanosecond timestamp (bytes 2-9) of a DEEP price level update or trade report."""
    return None if m is None else struct.unpack_from("<q", m, 2)[0]


def decode_iex(path: Path, symbols: Iterable[str], bucket_s: int = 10) -> dict:
    """Decode an IEX DEEP capture (a gzipped prefix is fine).

    Price level updates are grouped into events for the chosen symbols: IEX flags the last update
    of an event, and the book is consistent only after it. Each event becomes one :class:`Msg` (no
    snapshots: a book starts empty). Every trade and every status-type message is kept for ALL
    symbols, and every message is counted by type and time.

    Args:
        path: The downloaded ``.pcap.gz``.
        symbols: The tickers to keep books for.
        bucket_s: Seconds per bucket of the ``rhythm`` counts.

    Returns:
        A dict: ``format`` ('pcap' or 'pcapng'), ``events`` (symbol -> list of :class:`Msg`),
        ``trades`` (symbol -> DataFrame ``t_ns, price, size, flags``, chosen symbols),
        ``trades_all`` (the same for every symbol, plus a ``symbol`` column), ``others`` (type ->
        list of message bytes for every type except price level updates and trades, every symbol;
        decode with ``iex_features.status_tables``), ``rhythm`` (bucket number = time //
        ``bucket_s`` -> Counter of message type), ``bucket_s``, ``types`` (message type -> count
        over the whole capture), ``symbols`` (distinct symbols with a price level update),
        ``first_ns`` / ``last_ns`` (the first and last price level update) and ``examples`` (type
        -> the first message of that type, as bytes).
    """
    want = {s.upper().ljust(8).encode(): s.upper() for s in symbols}
    pend = {s: ([], []) for s in want.values()}
    events = {s: [] for s in want.values()}
    types, seen, examples = Counter(), set(), {}
    rhythm, others, all_trades = defaultdict(Counter), defaultdict(list), []
    bucket_ns = bucket_s * 10**9
    first = last = None
    fmt = "pcapng" if next(_gunzip(path), b"")[:4] == b"\n\r\r\n" else "pcap"
    for m in iter_iex_messages(_gunzip(path)):
        t = m[0]
        ch = chr(t)
        types[ch] += 1
        rhythm[struct.unpack_from("<q", m, 2)[0] // bucket_ns][ch] += 1
        examples.setdefault(ch, m)
        if t in (0x38, 0x35):
            last = m
            first = first or m
            seen.add(m[10:18])
            sym = want.get(m[10:18])
            if sym is None:
                continue
            flags, ts, _, size, px = _PLU.unpack_from(m)
            pend[sym][0 if t == 0x38 else 1].append((px / 1e4, float(size)))
            if flags & 1:
                events[sym].append(Msg(ts, False, *pend[sym]))
                pend[sym] = ([], [])
        elif t == 0x54:
            flags, ts, sym, size, px, _ = _TRADE.unpack_from(m)
            all_trades.append((ts, sym.decode().strip(), px / 1e4, size, flags))
        else:
            others[ch].append(m)
    trades_all = pd.DataFrame(all_trades, columns=["t_ns", "symbol", "price", "size", "flags"])
    return {"format": fmt, "events": events, "types": types, "symbols": len(seen),
            "trades_all": trades_all, "others": dict(others), "rhythm": dict(rhythm),
            "bucket_s": bucket_s,
            "trades": {s: trades_all[trades_all["symbol"] == s].drop(columns="symbol")
                       .reset_index(drop=True) for s in want.values()},
            "first_ns": _msg_time(first), "last_ns": _msg_time(last), "examples": examples}


def describe_iex_message(m: bytes) -> dict:
    """One DEEP message as a readable dict (price level updates and trades decoded; any other
    type shows its name and its first bytes in hex).

    Args:
        m: A message as yielded by :func:`iter_iex_messages`.

    Returns:
        The fields.
    """
    t = m[0]
    out = {"type": chr(t), "name": TYPE_NAMES.get(t, "?"), "bytes": len(m)}
    if t in (0x38, 0x35):
        flags, ts, sym, size, px = _PLU.unpack_from(m)
        out |= {"time": str(pd.Timestamp(ts, unit="ns", tz="UTC")), "symbol": sym.decode().strip(),
                "size": size, "price": px / 1e4, "event_complete": bool(flags & 1)}
    elif t == 0x54:
        flags, ts, sym, size, px, trade_id = _TRADE.unpack_from(m)
        out |= {"time": str(pd.Timestamp(ts, unit="ns", tz="UTC")), "symbol": sym.decode().strip(),
                "size": size, "price": px / 1e4, "trade_id": trade_id}
    else:
        out["hex"] = m[:24].hex()
    return out


# ---------------------------------------------------------------------------------------------
# Replay: apply a message stream to a book, check it, sample it.
# ---------------------------------------------------------------------------------------------
def _sample(t: int, bids: dict, asks: dict, bb: float | None, ba: float | None,
            bands: tuple) -> dict | None:
    """The book at grid time ``t``: top of book and USD depth within bands of the mid."""
    if bb is None or ba is None:
        return None
    mid = (bb + ba) / 2
    row = {"t_ns": t, "bid": bb, "ask": ba, "mid": mid, "spread_bps": (ba - bb) / mid * 1e4,
           "bid_levels": len(bids), "ask_levels": len(asks)}
    for side, book, sign in (("bid", bids, -1), ("ask", asks, 1)):
        p = np.fromiter(book, float, len(book))
        usd = p * np.fromiter(book.values(), float, len(book))
        away = sign * (p - mid) / mid * 1e4                    # bps from the mid, outwards
        for b in bands:
            row[f"{side}_usd_{b}"] = float(usd[away <= b].sum())
    return row


def replay(msgs: Iterable[Msg], *, grid_s: float = 10.0, bands_bps: tuple = BANDS_BPS,
           at_ns: Iterable[int] = ()) -> dict:
    """Apply a message stream to an empty book, check it and sample it on a time grid.

    The best bid and ask are tracked incrementally (a full scan only when the best level is
    deleted), so a crossed book is caught after every message.

    Args:
        msgs: Time-ordered :class:`Msg`.
        grid_s: Seconds between samples.
        bands_bps: Distances from the mid (bps) for the depth columns.
        at_ns: Times at which to keep a copy of the whole book (see :func:`depth_profile`).

    Returns:
        A dict: ``samples`` (DataFrame on a UTC grid: ``bid``, ``ask``, ``mid``, ``spread_bps``,
        ``bid_levels``, ``ask_levels`` and ``bid_usd_<b>`` / ``ask_usd_<b>``, the USD resting
        within ``b`` bps of the mid), ``counts`` (messages, snapshots, deletes, crossed, backwards
        = messages earlier than their predecessor, seq_gaps), ``times`` (message times, ns),
        ``sizes`` (levels set per message), ``books`` (``{time: (bids, asks)}``) and
        ``first_snapshot`` (levels a side of the first snapshot, or None).
    """
    step = round(grid_s * 1e9)
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    bb = ba = next_t = prev_t = prev_seq = first_snap = None
    rows, books, n = [], {}, Counter()
    pending = sorted(at_ns, reverse=True)
    times, sizes = array("q"), array("I")
    for m in msgs:
        t = m.t_ns
        if next_t is None:
            next_t = (t // step + 1) * step
        while t >= next_t:
            row = _sample(next_t, bids, asks, bb, ba, bands_bps)
            if row:
                rows.append(row)
            next_t += step
        while pending and t >= pending[-1]:
            books[pending.pop()] = (dict(bids), dict(asks))
        n["messages"] += 1
        times.append(t)
        sizes.append(len(m.bids) + len(m.asks))
        if prev_t is not None and t < prev_t:
            n["backwards"] += 1
        prev_t = t
        if m.seq is not None:
            if not m.snap and prev_seq is not None and m.seq != prev_seq + 1:
                n["seq_gaps"] += 1
            prev_seq = m.seq
        if m.snap:
            n["snapshots"] += 1
            first_snap = first_snap or (len(m.bids), len(m.asks))
            bids.clear()
            asks.clear()
            bb = ba = None
        for p, s in m.bids:
            if s > 0:
                bids[p] = s
                bb = p if bb is None or p > bb else bb
            else:
                n["deletes"] += 1
                bids.pop(p, None)
                if p == bb:
                    bb = max(bids) if bids else None
        for p, s in m.asks:
            if s > 0:
                asks[p] = s
                ba = p if ba is None or p < ba else ba
            else:
                n["deletes"] += 1
                asks.pop(p, None)
                if p == ba:
                    ba = min(asks) if asks else None
        if bb is not None and ba is not None and bb >= ba:
            n["crossed"] += 1
    cols = ["t_ns", "bid", "ask", "mid", "spread_bps", "bid_levels", "ask_levels",
            *(f"{side}_usd_{b}" for side in ("bid", "ask") for b in bands_bps)]
    samples = pd.DataFrame(rows, columns=cols)
    samples.index = pd.to_datetime(samples.pop("t_ns"), unit="ns", utc=True)
    return {"samples": samples, "counts": dict(n), "times": np.frombuffer(times, dtype=np.int64),
            "sizes": np.frombuffer(sizes, dtype=np.uint32), "books": books,
            "first_snapshot": first_snap}


def depth_profile(bids: dict, asks: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative USD resting at or inside each distance from the mid, per side.

    Args:
        bids: ``{price: size}``, as kept in :func:`replay`'s ``books``.
        asks: The same for the ask side.

    Returns:
        ``(bid curve, ask curve)``: DataFrames of ``bps`` (distance from the mid, outwards,
        ascending) and ``usd`` (cumulative price x size).
    """
    mid = (max(bids) + min(asks)) / 2
    out = []
    for book, sign in ((bids, -1), (asks, 1)):
        p = np.fromiter(book, float, len(book))
        usd = p * np.fromiter(book.values(), float, len(book))
        away = sign * (p - mid) / mid * 1e4
        order = np.argsort(away)
        out.append(pd.DataFrame({"bps": away[order], "usd": np.cumsum(usd[order])}))
    return out[0], out[1]


def summary(runs: dict[str, dict]) -> pd.DataFrame:
    """One column per replayed book, one row per measure, the same measures for every source.

    Args:
        runs: ``{name: replay(...) result}``.

    Returns:
        Rows: span, messages and their rate (mean and 95th percentile per second), levels set
        per message (median, 95th percentile), share of level changes that delete, snapshots and
        first-snapshot size, spread (bps) and USD depth within 10 bps (median, both sides),
        longest silence, and the checks (crossed books, update-id gaps, messages out of order).
    """
    cols = {}
    for name, r in runs.items():
        t, lv, c, s = r["times"], r["sizes"], r["counts"], r["samples"]
        per_s = np.bincount(((t - t[0]) // 10**9).astype(int))
        snap = r["first_snapshot"]
        depth = s["bid_usd_10"] + s["ask_usd_10"] if "bid_usd_10" in s else pd.Series(dtype=float)
        cols[name] = {
            "span_hours": (t[-1] - t[0]) / 3.6e12, "messages": c["messages"],
            "msgs_per_s_mean": c["messages"] / len(per_s),
            "msgs_per_s_p95": float(np.percentile(per_s, 95)),
            "levels_per_msg_p50": float(np.median(lv)),
            "levels_per_msg_p95": float(np.percentile(lv, 95)),
            "pct_changes_that_delete": 100 * c.get("deletes", 0) / max(1, int(lv.sum())),
            "snapshots": c.get("snapshots", 0),
            "first_snapshot_levels": sum(snap) if snap else np.nan,
            "spread_bps_p50": float(s["spread_bps"].median()) if len(s) else np.nan,
            "spread_bps_p95": float(s["spread_bps"].quantile(0.95)) if len(s) else np.nan,
            "depth10bps_usd_p50": float(depth.median()) if len(depth) else np.nan,
            "longest_silence_s": float(np.diff(t).max() / 1e9) if len(t) > 1 else np.nan,
            "crossed_books": c.get("crossed", 0), "update_id_gaps": c.get("seq_gaps", 0),
            "out_of_order": c.get("backwards", 0)}
    return pd.DataFrame(cols)


def compare_mid(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """How two sampled books of the SAME instrument agree: mid prices on their common grid.

    Args:
        a: ``replay(...)['samples']`` of one source.
        b: The same of another source (same grid).

    Returns:
        ``matched`` samples, the median and the median absolute / 95th-percentile absolute
        difference of the mids in bps (a minus b), and the correlation of their 1-minute returns.
    """
    j = pd.DataFrame({"a": a["mid"], "b": b["mid"]}).dropna()
    d = (j["a"] - j["b"]) / j["b"] * 1e4
    r = np.log(j.resample("1min").last()).diff().dropna()
    return {"matched": len(j), "median_diff_bps": float(d.median()),
            "median_abs_diff_bps": float(d.abs().median()),
            "p95_abs_diff_bps": float(d.abs().quantile(0.95)),
            "corr_1min_returns": float(r["a"].corr(r["b"]))}
