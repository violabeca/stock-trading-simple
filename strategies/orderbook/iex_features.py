"""What a strategy could take from IEX's DEEP feed, as small tested functions.

Loaded by path from ``iex_data_explained_marimo.py``. It works on what ``orderbook_free.decode_iex``
returns (book events, trades, raw status messages) and on ``orderbook_free.replay`` results; no
network, no files, only numpy and pandas. Each function is ONE piece of information:

* :func:`rhythm_frame`: how many messages of each kind per time bucket (the feed's pulse);
* :func:`status_tables`: halts, short-sale restriction, retail interest, the opening, auctions ...
  decoded to tables; these cover every symbol at once;
* :func:`book_features`, :func:`quotes`: the top of one stock's book after every event, with the
  spread, mid, microprice and queue imbalance it gives;
* :func:`ofi`, :func:`bucket_ofi`: order-flow imbalance (Cont, Kukanov & Stoikov 2014) per event
  and per time bucket, against the change of the mid;
* :func:`sign_trades`: a buy / sell label for each trade from the quote just before it;
* :func:`ladder`: the book over time as matrices for a heat map.

DEEP shows only the DISPLAYED orders resting on IEX, summed per price level. Every book feature
here therefore describes IEX's own book, not the market's.
"""

from __future__ import annotations

import struct
from collections import Counter

import numpy as np
import pandas as pd

#: Message type -> (name, what it tells you, which securities). Names and codes follow IEX's
#: DEEP specification v1.08.
CATALOGUE = {
    "8": ("Price Level Update (buy)",
          "total displayed size now resting at one bid price on IEX; 0 = the level is gone; the "
          "last update of an event carries a flag", "every symbol"),
    "5": ("Price Level Update (sell)", "the same for one ask price", "every symbol"),
    "T": ("Trade Report",
          "an execution on IEX, displayed or hidden liquidity: price, size, flags (intermarket "
          "sweep, odd lot, extended hours, trade-through exempt, single-price cross); no buy / "
          "sell side", "every symbol"),
    "H": ("Trading Status",
          "T trading, H halted across US markets (P and O: pause and order acceptance, IEX-listed "
          "only); one per symbol at the start of the day, then changes", "every symbol"),
    "O": ("Operational Halt Status", "N normal, O halted on IEX only", "every symbol"),
    "P": ("Short Sale Price Test Status",
          "Reg SHO rule 201 short-sale restriction in effect (1) or not (0)", "every symbol"),
    "E": ("Security Event",
          "O: IEX's opening process for the symbol is complete, orders queued before 09:30 are "
          "live; C: closing process complete", "every symbol"),
    "I": ("Retail Liquidity Indicator",
          "whether IEX holds retail-eligible interest: A buy side, B sell side, C both, blank "
          "none; no size, no price", "every symbol"),
    "S": ("System Event",
          "start of messages, of system hours (08:00), of regular hours (09:30), and their ends",
          "the market"),
    "D": ("Security Directory",
          "round lot size, prior official close, LULD tier, test / ETP flags", "IEX-listed only"),
    "A": ("Auction Information",
          "once a second before an auction: paired shares, imbalance side and size, indicative "
          "price, collars", "IEX-listed only"),
    "X": ("Official Price", "IEX official opening / closing price", "IEX-listed only"),
    "B": ("Trade Break", "a trade of the same day cancelled", "every symbol"),
}
_GROUP = {"8": "book updates", "5": "book updates", "T": "trades", "I": "retail interest flags",
          "H": "status messages", "O": "status messages", "P": "status messages",
          "E": "status messages"}
_OTHER = "auction, official price, directory, system"
_AUCTION = struct.Struct("<xcq8sIqqIcBIqqqq")
_FIELDS = ("bid", "bid_qty", "ask", "ask_qty", "bid_qty_k", "ask_qty_k", "mid", "spread_bps",
           "micro", "obi1", "obik")
#: Where a trade can print against the quote before it, from the sell side to the buy side.
WHERE = ("at or below the bid", "inside, below the mid", "at the mid", "inside, above the mid",
         "at or above the ask", "no quote")


def rhythm_frame(rhythm: dict, bucket_s: int) -> pd.DataFrame:
    """Message counts per time bucket and kind, from ``decode_iex``'s ``rhythm``.

    Args:
        rhythm: ``{bucket number: Counter of message type}`` (bucket number = time // bucket_s).
        bucket_s: Seconds per bucket.

    Returns:
        One row per bucket (UTC start time), one column per group: book updates, trades, retail
        interest flags, status messages and everything else.
    """
    rows = {}
    for b, counts in rhythm.items():
        grouped = Counter()
        for kind, n in counts.items():
            grouped[_GROUP.get(kind, _OTHER)] += n
        rows[b] = grouped
    df = pd.DataFrame.from_dict(rows, orient="index").fillna(0).sort_index()
    df.index = pd.to_datetime(df.index.to_numpy() * bucket_s * 10**9, unit="ns", utc=True)
    return df


def _time(m: bytes) -> int:
    """The nanosecond timestamp (bytes 2-9) every DEEP message carries."""
    return struct.unpack_from("<q", m, 2)[0]


def _symbol(m: bytes) -> str:
    """The symbol (bytes 10-17) of a message; empty for the 10-byte System Event."""
    return m[10:18].decode().strip()


def _auction(m: bytes) -> tuple:
    """The auction fields of an Auction Information message, prices in dollars."""
    typ, _, _, paired, ref, ind, imb, side, ext, sched, book, cref, lo, hi = _AUCTION.unpack_from(m)
    return (typ.decode(), paired, ref / 1e4, ind / 1e4, imb, side.decode(), ext, sched,
            book / 1e4, cref / 1e4, lo / 1e4, hi / 1e4)


#: type -> (table name, columns after time and symbol, parser of the message bytes)
_STATUS = {
    "S": ("system_event", ("event",), lambda m: (chr(m[1]),)),
    "D": ("security_directory", ("flags", "round_lot", "prior_close", "luld_tier"),
          lambda m: (m[1], struct.unpack_from("<I", m, 18)[0],
                     struct.unpack_from("<q", m, 22)[0] / 1e4, m[30])),
    "H": ("trading_status", ("status", "reason"), lambda m: (chr(m[1]), m[18:22].decode().strip())),
    "O": ("operational_halt", ("status",), lambda m: (chr(m[1]),)),
    "P": ("short_sale", ("in_effect", "detail"), lambda m: (m[1], chr(m[18]))),
    "E": ("security_event", ("event",), lambda m: (chr(m[1]),)),
    "I": ("retail_interest", ("indicator",), lambda m: (chr(m[1]),)),
    "X": ("official_price", ("kind", "price"),
          lambda m: (chr(m[1]), struct.unpack_from("<q", m, 18)[0] / 1e4)),
    "A": ("auction", ("auction_type", "paired_shares", "reference_price", "indicative_price",
                      "imbalance_shares", "imbalance_side", "extensions", "scheduled_s",
                      "book_clearing_price", "collar_reference", "lower_collar", "upper_collar"),
          _auction),
}


def status_tables(others: dict) -> dict[str, pd.DataFrame]:
    """Decode the status-type messages of a capture into one table per kind.

    Args:
        others: ``decode_iex``'s ``others``: message type -> list of message bytes.

    Returns:
        Table name -> DataFrame with ``t_ns``, ``symbol`` and the message's own columns, for
        every kind: system_event, security_directory, trading_status, operational_halt,
        short_sale, security_event, retail_interest, official_price, auction. A kind the capture
        holds no message of (a quiet morning has no operational halt) is an EMPTY table, never a
        missing one.
    """
    return {
        name: pd.DataFrame(
            [(_time(m), _symbol(m), *parse(m)) for m in others.get(kind, [])],
            columns=["t_ns", "symbol", *cols],
        )
        for kind, (name, cols, parse) in _STATUS.items()
    }


def book_features(bids: dict, asks: dict, depth: int = 5) -> dict | None:
    """The top of one book: best prices and sizes, the sizes of the best ``depth`` levels, and
    the mid, spread, microprice and queue imbalance they give.

    * ``mid`` = (bid + ask) / 2; ``spread_bps`` = (ask - bid) / mid in basis points;
    * ``micro`` (microprice) = (bid x ask size + ask x bid size) / (bid size + ask size): the mid
      pulled toward the side with LESS size, where the price is likelier to move next;
    * ``obi1`` = (bid size - ask size) / (bid size + ask size) at the best level, and ``obik``
      the same with the sizes of the best ``depth`` levels; both lie in [-1, 1].

    Args:
        bids: ``{price: size}``.
        asks: ``{price: size}``.
        depth: How many levels per side go into ``obik``.

    Returns:
        The fields, or None unless both sides have a level.
    """
    if not bids or not asks:
        return None
    bp, ap = sorted(bids, reverse=True)[:depth], sorted(asks)[:depth]
    bid, ask = bp[0], ap[0]
    bq, aq = bids[bid], asks[ask]
    bqk, aqk = sum(bids[p] for p in bp), sum(asks[p] for p in ap)
    mid = (bid + ask) / 2
    return {"bid": bid, "bid_qty": bq, "ask": ask, "ask_qty": aq, "bid_qty_k": bqk,
            "ask_qty_k": aqk, "mid": mid, "spread_bps": (ask - bid) / mid * 1e4,
            "micro": (bid * aq + ask * bq) / (bq + aq), "obi1": (bq - aq) / (bq + aq),
            "obik": (bqk - aqk) / (bqk + aqk)}


def quotes(events: list, depth: int = 5) -> pd.DataFrame:
    """The top of one stock's book after every event (see :func:`book_features`).

    Args:
        events: ``decode_iex``'s ``events[symbol]``: time-ordered book messages.
        depth: Levels per side that go into the multi-level imbalance.

    Returns:
        One row per CHANGE of the top of the book (``t_ns`` = time of the event) while both sides
        have a level, with the columns of :func:`book_features`.
    """
    bids, asks, rows, last = {}, {}, [], None
    for m in events:
        if m.snap:
            bids.clear()
            asks.clear()
        for book, levels in ((bids, m.bids), (asks, m.asks)):
            for p, s in levels:
                if s > 0:
                    book[p] = s
                else:
                    book.pop(p, None)
        f = book_features(bids, asks, depth)
        if f is None:
            continue
        key = tuple(f[k] for k in _FIELDS[:6])
        if key != last:
            rows.append({"t_ns": m.t_ns, **f})
            last = key
    return pd.DataFrame(rows, columns=["t_ns", *_FIELDS])


def ofi(q: pd.DataFrame, parts: bool = False) -> pd.Series | pd.DataFrame:
    """Order-flow imbalance of every quote state against the one before it.

    The rule of Cont, Kukanov & Stoikov (2014), from the best bid ``Pb, Qb`` and ask ``Pa, Qa``:
    ``e_bid = Qb_n if Pb_n >= Pb_n-1, minus Qb_n-1 if Pb_n <= Pb_n-1`` (a higher or equal bid
    adds its whole size, a lower or equal one removes the old size) and
    ``e_ask = Qa_n-1 if Pa_n >= Pa_n-1, minus Qa_n if Pa_n <= Pa_n-1``. OFI = e_bid + e_ask:
    positive when buying pressure rose (bid size added or the bid lifted, ask size removed or the
    ask lifted away), negative when selling pressure did. The first state has no predecessor: 0.

    Args:
        q: :func:`quotes` output.
        parts: Return the two sides and their sum as a DataFrame instead of the sum.

    Returns:
        OFI per row (shares), or a frame with ``e_bid``, ``e_ask`` and ``ofi``.
    """
    pb, pa, qb, qa = q["bid"], q["ask"], q["bid_qty"], q["ask_qty"]
    pb0, pa0, qb0, qa0 = pb.shift(), pa.shift(), qb.shift(), qa.shift()
    e_bid = qb.where(pb >= pb0, 0.0) - qb0.where(pb <= pb0, 0.0)
    e_ask = qa0.where(pa >= pa0, 0.0) - qa.where(pa <= pa0, 0.0)
    total = e_bid + e_ask
    return pd.DataFrame({"e_bid": e_bid, "e_ask": e_ask, "ofi": total}) if parts else total


def bucket_ofi(q: pd.DataFrame, t0_ns: int, t1_ns: int, bucket_s: float) -> pd.DataFrame:
    """OFI summed per time bucket, next to the change of the mid over the same bucket.

    A bucket is (edge, next edge]; the mid at an edge is the last quote state at or before it.

    Args:
        q: :func:`quotes` output.
        t0_ns: Start of the first bucket (ns since the epoch).
        t1_ns: The last bucket ends at or after this time.
        bucket_s: Seconds per bucket.

    Returns:
        One row per bucket with a known mid change: ``t_ns`` (bucket start), ``ofi`` (shares) and
        ``dmid`` (price change of the mid, in dollars).
    """
    if q.empty:
        return pd.DataFrame(columns=["t_ns", "ofi", "dmid"])
    step = round(bucket_s * 1e9)
    edges = np.arange(t0_ns, t1_ns + step, step)
    t = q["t_ns"].to_numpy()
    last = np.searchsorted(t, edges, side="right") - 1
    mid = np.where(last >= 0, q["mid"].to_numpy()[np.maximum(last, 0)], np.nan)
    which = np.searchsorted(edges, t, side="left") - 1
    ok = (which >= 0) & (which < len(edges) - 1)
    flow = np.bincount(which[ok], weights=ofi(q).to_numpy()[ok], minlength=len(edges) - 1)
    out = pd.DataFrame({"t_ns": edges[:-1], "ofi": flow, "dmid": np.diff(mid)})
    return out.dropna(subset=["dmid"]).reset_index(drop=True)


def sign_trades(trades: pd.DataFrame, q: pd.DataFrame) -> pd.DataFrame:
    """Label each trade buyer- or seller-initiated from the IEX quote just BEFORE it.

    DEEP's trade report has no aggressor side, so it is inferred (Lee & Ready 1991): a trade at or
    above the ask, or above the mid, is a buy (+1); at or below the bid, or below the mid, a sell
    (-1); exactly at the mid, or with no quote yet, the tick rule decides (the sign of the last
    price change). The quote used is strictly earlier than the trade: DEEP sends a trade report
    BEFORE the book update it causes, with the same timestamp.

    Args:
        trades: ``t_ns``, ``price``, ``size`` (and any other columns), one stock.
        q: :func:`quotes` output of the same stock.

    Returns:
        The trades in time order with the quote used (``q_ns``, ``bid``, ``ask``), ``where`` (one
        of :data:`WHERE`) and ``side`` (+1 / -1; 0 only for a first trade at the mid or unquoted).
    """
    t = trades.sort_values("t_ns").reset_index(drop=True)
    before = q[["t_ns", "bid", "ask"]].rename(columns={"t_ns": "q_ns"})
    m = pd.merge_asof(t, before, left_on="t_ns", right_on="q_ns", allow_exact_matches=False)
    mid = (m["bid"] + m["ask"]) / 2
    where = np.select(
        [m["bid"].isna(), m["price"] >= m["ask"], m["price"] <= m["bid"], m["price"] > mid,
         m["price"] < mid],
        [WHERE[5], WHERE[4], WHERE[0], WHERE[3], WHERE[1]], default=WHERE[2])
    tick = np.sign(m["price"].diff()).replace(0, np.nan).ffill().fillna(0).to_numpy()
    side = np.select([np.isin(where, [WHERE[4], WHERE[3]]), np.isin(where, [WHERE[0], WHERE[1]])],
                     [1, -1], default=tick)
    return m.assign(where=where, side=side.astype(int))


def ladder(books: dict, tick: float = 0.01, max_ticks: int = 40
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One stock's book over time as two matrices (bid sizes, ask sizes) for a heat map.

    Args:
        books: ``orderbook_free.replay``'s ``books``: ``{time ns: (bids, asks)}``.
        tick: Price step of the rows.
        max_ticks: Keep levels at most this many ticks from that moment's mid (a pre-open book
            holds placeholder quotes far away); a moment without both sides stays empty.

    Returns:
        ``(times, prices, bid_sizes, ask_sizes)``: times (ns, sorted), the row prices (rising),
        and matrices of shape (prices, times) holding the resting size, NaN where no level rests.
    """
    times = sorted(books)
    cols = []
    for t in times:
        bids, asks = books[t]
        if not bids or not asks:
            cols.append(None)
            continue
        mid_i = round((max(bids) + min(asks)) / 2 / tick)
        near = {side: {round(p / tick): s for p, s in lv.items()
                       if abs(round(p / tick) - mid_i) <= max_ticks}
                for side, lv in (("b", bids), ("a", asks))}
        cols.append(near)
    rows = [i for c in cols if c for lv in c.values() for i in lv]
    if not rows:
        empty = np.empty((0, len(times)))
        return np.array(times), np.array([]), empty, empty.copy()
    lo, hi = min(rows), max(rows)
    bid_z = np.full((hi - lo + 1, len(times)), np.nan)
    ask_z = bid_z.copy()
    for j, c in enumerate(cols):
        if c:
            for i, s in c["b"].items():
                bid_z[i - lo, j] = s
            for i, s in c["a"].items():
                ask_z[i - lo, j] = s
    return np.array(times), (lo + np.arange(hi - lo + 1)) * tick, bid_z, ask_z
