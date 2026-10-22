"""Offline tests of strategies/orderbook/orderbook_free.py (no network: requests is stubbed)."""

import gzip
import importlib.util
import io
import json
import struct
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

_P = Path(__file__).resolve().parents[1] / "strategies" / "orderbook" / "orderbook_free.py"
_spec = importlib.util.spec_from_file_location("orderbook_free", _P)
ob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ob)


class _Resp:
    """A stand-in for requests.Response."""

    def __init__(self, payload=None, text="", chunks=(), status=200):
        self._payload, self.text, self._chunks, self.status_code = payload, text, chunks, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, n):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _m(t, bids=(), asks=(), snap=False, seq=None):
    """A book message at second ``t``."""
    return ob.Msg(t * 10**9, snap, list(bids), list(asks), seq)


# --- building captures --------------------------------------------------------------------------
def _plu(ts, sym, side, size, price, last=True):
    """A DEEP price level update (30 bytes)."""
    return struct.pack("<BBq8sIq", 0x38 if side == "b" else 0x35, int(last), ts,
                       sym.ljust(8).encode(), size, round(price * 1e4))


def _trade(ts, sym, size, price, trade_id, flags=0):
    """A DEEP trade report (38 bytes)."""
    return struct.pack("<BBq8sIqq", 0x54, flags, ts, sym.ljust(8).encode(), size,
                       round(price * 1e4), trade_id)


def _status(kind, ts, sym, code, tail=b""):
    """A DEEP status-type message: type, one code byte, time, symbol, then any extra bytes."""
    return struct.pack("<BBq8s", ord(kind), code if isinstance(code, int) else ord(code), ts,
                       sym.ljust(8).encode()) + tail


def _packet(messages, proto=0x8004):
    """Ethernet + IPv4 + UDP + IEX-TP header + messages."""
    payload = b"".join(struct.pack("<H", len(m)) + m for m in messages)
    iex = struct.pack("<BBHIIHHqqq", 1, 0, proto, 1, 1, len(payload), len(messages), 0, 1, 0)
    udp = struct.pack(">HHHH", 1, 2, 8 + 40 + len(payload), 0) + iex + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0, b"\0" * 4,
                     b"\0" * 4) + udp
    return b"\0" * 12 + b"\x08\x00" + ip


def _pcapng(packets):
    """A pcapng capture with nanosecond timestamps 0, 1, 2, ..."""
    def block(kind, body):
        body += b"\0" * (-len(body) % 4)
        return struct.pack("<II", kind, len(body) + 12) + body + struct.pack("<I", len(body) + 12)

    shb = block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    idb = block(1, struct.pack("<HHI", 1, 0, 65535) + struct.pack("<HH", 9, 1) + b"\x09\0\0\0"
                + struct.pack("<HH", 0, 0))
    epb = [block(6, struct.pack("<IIIII", 0, 0, i, len(p), len(p)) + p)
           for i, p in enumerate(packets)]
    return shb + idb + b"".join(epb)


def _pcap(packets):
    """A classic little-endian pcap capture with microsecond timestamps 1 s + i us."""
    out = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    for i, p in enumerate(packets):
        out += struct.pack("<IIII", 1, i, len(p), len(p)) + p
    return out


# --- packets and IEX -----------------------------------------------------------------------------
@pytest.mark.parametrize("make, times", [(_pcapng, [0, 1, 2]),
                                         (_pcap, [10**9, 10**9 + 1000, 10**9 + 2000])])
def test_iter_packets_reads_both_formats_however_the_bytes_are_split(make, times):
    pkts = [b"abc" * 10, b"defg" * 7, b"h" * 61]
    raw = make(pkts)
    for cut in (1, 7, 13, 50, len(raw)):
        got = list(ob.iter_packets([raw[i:i + cut] for i in range(0, len(raw), cut)]))
        assert [g[2] for g in got] == pkts and [g[0] for g in got] == times
        assert {g[1] for g in got} == {1}
    assert len(list(ob.iter_packets([raw[:-5]]))) == 2            # cut short: whole packets only
    with pytest.raises(ValueError, match="not a pcap"):
        list(ob.iter_packets([b"x" * 40]))


def test_decode_iex_groups_updates_into_events_and_reads_trades(tmp_path):
    p1 = _packet([_plu(10, "SPY", "b", 100, 500.00, last=False), _plu(10, "SPY", "b", 50, 499.99)])
    p2 = _packet([_plu(11, "SPY", "s", 70, 500.05), _plu(11, "QQQ", "s", 10, 400.0),
                  _trade(12, "SPY", 30, 500.02, 7), _trade(12, "QQQ", 5, 400.0, 8, flags=0xA0),
                  _status("H", 9, "SPY", "T", b"    "), _status("P", 9, "QQQ", 1, b"N")])
    p3 = _packet([_plu(13, "SPY", "b", 0, 500.00)])
    path = tmp_path / "day.pcap.gz"
    path.write_bytes(gzip.compress(_pcapng([p1, p2, p3]), mtime=0))
    d = ob.decode_iex(path, ["spy"])
    assert d["format"] == "pcapng"
    assert [(e.t_ns, e.snap, e.bids, e.asks) for e in d["events"]["SPY"]] == [
        (10, False, [(500.0, 100.0), (499.99, 50.0)], []),
        (11, False, [], [(500.05, 70.0)]),
        (13, False, [(500.0, 0.0)], [])]
    assert list(d["trades"]) == ["SPY"]                              # only the chosen symbols
    assert d["trades"]["SPY"].to_dict("records") == [
        {"t_ns": 12, "price": 500.02, "size": 30, "flags": 0}]
    assert d["trades_all"].to_dict("records") == [          # every symbol, with the flag byte
        {"t_ns": 12, "symbol": "SPY", "price": 500.02, "size": 30, "flags": 0},
        {"t_ns": 12, "symbol": "QQQ", "price": 400.0, "size": 5, "flags": 0xA0}]
    assert dict(d["types"]) == {"8": 3, "5": 2, "T": 2, "H": 1, "P": 1} and d["symbols"] == 2
    assert sorted(d["others"]) == ["H", "P"] and len(d["others"]["P"]) == 1   # kept for ALL symbols
    assert d["rhythm"] == {0: Counter({"8": 3, "5": 2, "T": 2, "H": 1, "P": 1})}   # 10 s buckets
    assert (d["first_ns"], d["last_ns"]) == (10, 13)
    assert ob.describe_iex_message(d["examples"]["T"])["trade_id"] == 7
    assert ob.describe_iex_message(d["examples"]["8"]) == {
        "type": "8", "name": "Price Level Update (buy)", "bytes": 30,
        "time": "1970-01-01 00:00:00.000000010+00:00", "symbol": "SPY", "size": 100,
        "price": 500.0, "event_complete": False}
    assert ob.describe_iex_message(b"S\x00" + bytes(8))["name"] == "System Event"


def test_decode_iex_reads_classic_pcap_and_survives_a_truncated_gzip(tmp_path):
    pkts = [_packet([_plu(i, "SPY", "b", 10 + i, 500.0)]) for i in range(40)]
    gz = gzip.compress(_pcap(pkts), mtime=0)
    full = tmp_path / "a.pcap.gz"
    full.write_bytes(gz)
    d = ob.decode_iex(full, ["SPY"])
    assert d["format"] == "pcap" and len(d["events"]["SPY"]) == 40
    cut = tmp_path / "b.pcap.gz"
    cut.write_bytes(gz[:-30])                                    # a prefix, as fetch_iex stores
    assert 0 < len(ob.decode_iex(cut, ["SPY"])["events"]["SPY"]) <= 40


def test_iex_rejects_a_tops_capture(tmp_path):
    path = tmp_path / "tops.pcap.gz"
    path.write_bytes(gzip.compress(_pcapng([_packet([_plu(1, "SPY", "b", 1, 1.0)], proto=0x8003)])))
    with pytest.raises(ValueError, match="not a DEEP capture"):
        ob.decode_iex(path, ["SPY"])


# --- OKX and Bybit files -------------------------------------------------------------------------
def test_iter_okx_iter_bybit_and_raw_head_read_the_files(tmp_path):
    okx = [{"instId": "X", "action": "snapshot", "ts": "1000",
            "asks": [["101.5", "2", "3"], ["102", "1", "1"]], "bids": [["100", "4", "2"]]},
           {"instId": "X", "action": "update", "ts": "1010",
            "asks": [["101.5", "0", "0"]], "bids": [["100.5", "1", "1"]]}]
    body = ("\n".join(map(json.dumps, okx)) + "\n").encode()
    p = tmp_path / "X-L2orderbook-400lv-2026-10-16.tar.gz"
    with tarfile.open(p, "w:gz") as tar:
        info = tarfile.TarInfo("X.data")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    a, b = list(ob.iter_okx(p))
    assert a == ob.Msg(1000 * 10**6, True, [(100.0, 4.0)], [(101.5, 2.0), (102.0, 1.0)], None)
    assert (b.snap, b.bids, b.asks) == (False, [(100.5, 1.0)], [(101.5, 0.0)])
    assert len(ob.raw_head(p, 2)) == 2 and ob.raw_head(p, 1, 30)[0].endswith(" ...")

    rows = [{"topic": "orderbook.200.X", "ts": 5, "type": "snapshot", "cts": 4,
             "data": {"s": "X", "b": [["100", "4"]], "a": [["101", "2"]], "u": 1, "seq": 9}},
            {"topic": "orderbook.200.X", "ts": 6, "type": "delta", "cts": 5,
             "data": {"s": "X", "b": [["100", "0"]], "a": [], "u": 2, "seq": 10}}]
    z = tmp_path / "2026-10-16_X_ob200.data.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("2026-10-16_X_ob200.data", "\n".join(map(json.dumps, rows)) + "\n")
    m1, m2 = list(ob.iter_bybit(z))
    assert m1 == ob.Msg(5 * 10**6, True, [(100.0, 4.0)], [(101.0, 2.0)], 1)
    assert (m2.snap, m2.bids, m2.seq) == (False, [(100.0, 0.0)], 2)
    assert ob.raw_head(z, 1)[0].startswith('{"topic"')


# --- replay --------------------------------------------------------------------------------------
def test_replay_applies_snapshot_deltas_and_deletes_and_samples_the_grid():
    msgs = [_m(1, [(100.0, 2.0), (99.0, 3.0)], [(101.0, 1.0), (102.0, 4.0)], snap=True, seq=1),
            _m(12, [(100.0, 0.0), (98.0, 5.0)], [(101.5, 2.0)], seq=2),    # best bid deleted
            _m(25, [], [(101.0, 0.0)], seq=3),                              # best ask deleted
            _m(31, [(99.5, 1.0)], [], seq=4)]
    r = ob.replay(msgs, grid_s=10, bands_bps=(100,))
    s = r["samples"]
    assert list(s.index.astype("int64") // 10**9) == [10, 20, 30]
    assert list(s["bid"]) == [100.0, 99.0, 99.0] and list(s["ask"]) == [101.0, 101.0, 101.5]
    assert s["spread_bps"].iloc[0] == pytest.approx(1 / 100.5 * 1e4)
    # within 100 bps of the first mid (100.5): bid 100 x 2 and ask 101 x 1
    assert (s["bid_usd_100"].iloc[0], s["ask_usd_100"].iloc[0]) == (200.0, 101.0)
    assert (s["bid_levels"].iloc[1], s["ask_levels"].iloc[1]) == (2, 3)
    assert r["counts"] == {"messages": 4, "snapshots": 1, "deletes": 2}
    assert r["first_snapshot"] == (2, 2) and list(r["sizes"]) == [4, 3, 1, 1]


def test_replay_of_a_one_sided_book_gives_an_empty_but_well_formed_sample_frame():
    r = ob.replay([_m(1, [(100.0, 1.0)], []), _m(30, [(99.0, 1.0)], [])], grid_s=10)
    s = r["samples"]
    assert len(s) == 0 and isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None
    assert list(s.columns[:4]) == ["bid", "ask", "mid", "spread_bps"] and "ask_usd_25" in s
    assert len(s.loc[: pd.Timestamp("2026-01-01", tz="UTC")]) == 0
    assert np.isnan(ob.summary({"x": r}).loc["spread_bps_p50", "x"])      # no two-sided sample


def test_replay_flags_crossed_books_sequence_gaps_and_time_going_backwards():
    msgs = [_m(1, [(100.0, 1.0)], [(101.0, 1.0)], snap=True, seq=1),
            _m(2, [(101.0, 1.0)], [], seq=2),                 # the bid reaches the ask
            _m(3, [], [], seq=4),                             # update id jumps 2 -> 4
            _m(2, [], [], seq=5)]                             # time goes back
    c = ob.replay(msgs)["counts"]
    assert (c["crossed"], c["seq_gaps"], c["backwards"]) == (3, 1, 1)


def test_replay_keeps_a_copy_of_the_book_at_requested_times_and_profiles_it():
    msgs = [_m(1, [(100.0, 2.0), (99.0, 3.0)], [(101.0, 1.0), (102.0, 4.0)], snap=True),
            _m(20, [(100.0, 0.0)], [])]
    r = ob.replay(msgs, at_ns=[10 * 10**9])
    bids, asks = r["books"][10 * 10**9]                       # the state before the message at 20 s
    assert bids == {100.0: 2.0, 99.0: 3.0} and asks == {101.0: 1.0, 102.0: 4.0}
    b, a = ob.depth_profile(bids, asks)                       # mid 100.5
    assert list(b["usd"]) == pytest.approx([200.0, 497.0]) and list(a["usd"]) == pytest.approx(
        [101.0, 509.0])
    assert b["bps"].iloc[0] == pytest.approx(0.5 / 100.5 * 1e4)


def test_summary_has_one_column_per_book_and_compare_mid_measures_the_gap():
    msgs = [_m(1, [(100.0, 2.0)], [(101.0, 1.0)], snap=True, seq=1)]
    msgs += [_m(2 + i, [(100.0, 2.0 + i)], [], seq=2 + i) for i in range(30)]
    r = ob.replay(msgs, grid_s=5)
    s = ob.summary({"a": r, "b": r})
    assert list(s.columns) == ["a", "b"]
    assert (s.loc["messages", "a"], s.loc["snapshots", "a"], s.loc["first_snapshot_levels", "a"]
            ) == (31, 1, 2)
    assert s.loc["crossed_books", "a"] == 0 and s.loc["update_id_gaps", "a"] == 0
    assert s.loc["longest_silence_s", "a"] == pytest.approx(1.0)
    assert s.loc["pct_changes_that_delete", "a"] == 0

    idx = pd.date_range("2026-01-05 14:00", periods=600, freq="10s", tz="UTC")
    mid = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 1e-4, len(idx))))
    a, b = pd.DataFrame({"mid": mid}, index=idx), pd.DataFrame({"mid": mid * 1.0002}, index=idx)
    c = ob.compare_mid(a, b)
    assert c["matched"] == 600 and c["median_diff_bps"] == pytest.approx(-2.0, abs=0.01)
    assert c["corr_1min_returns"] == pytest.approx(1.0)


# --- downloads -----------------------------------------------------------------------------------
def test_okx_url_asks_for_the_day_retries_the_rate_limit_and_picks_the_file(monkeypatch):
    good = {"code": "0", "msg": "", "data": [{"details": [{"groupDetails": [
        {"filename": "BTC-USDT-L2orderbook-400lv-2026-10-15.tar.gz", "url": "u15"},
        {"filename": "BTC-USDT-L2orderbook-400lv-2026-10-16.tar.gz", "url": "u16"}]}]}]}
    answers, calls = [{"code": "50011", "msg": "Too Many Requests", "data": []}, good], []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        return _Resp(answers.pop(0))

    monkeypatch.setattr(ob.requests, "get", fake_get)
    monkeypatch.setattr(ob.time, "sleep", lambda s: None)
    assert ob.okx_url("2026-10-16", "BTC-USDT") == (
        "BTC-USDT-L2orderbook-400lv-2026-10-16.tar.gz", "u16")
    assert len(calls) == 2 and calls[0]["module"] == 4
    assert (calls[0]["instType"], calls[0]["instIdList"]) == ("SPOT", "BTC-USDT")
    assert calls[0]["begin"] == calls[0]["end"] == 1792108800000      # 2026-10-16 00:00 UTC, ms

    answers.append(good)
    ob.okx_url("2026-10-16", "BTC-USDT", "swap")
    assert calls[-1]["instType"] == "SWAP" and calls[-1]["instFamilyList"] == "BTC-USDT"
    assert "instIdList" not in calls[-1]

    answers.append({"code": "50016", "msg": "bad", "data": []})
    with pytest.raises(RuntimeError, match="50016"):
        ob.okx_url("2026-10-16")
    answers.append({"code": "0", "msg": "", "data": [{"details": [{"groupDetails": []}]}]})
    with pytest.raises(FileNotFoundError):
        ob.okx_url("2026-10-16")


def test_bybit_url_tries_the_200_level_file_then_the_500_level_one(monkeypatch):
    def head_for(suffix):
        return lambda url, headers=None, timeout=None: _Resp(
            status=200 if url.endswith(suffix) else 404)

    monkeypatch.setattr(ob.requests, "head", head_for("_ob500.data.zip"))
    assert ob.bybit_url("2024-09-03", "BTCUSDT", "linear").endswith(
        "linear/BTCUSDT/2024-09-03_BTCUSDT_ob500.data.zip")
    monkeypatch.setattr(ob.requests, "head", head_for("_ob200.data.zip"))
    assert ob.bybit_url("2026-10-16").endswith("spot/BTCUSDT/2026-10-16_BTCUSDT_ob200.data.zip")
    monkeypatch.setattr(ob.requests, "head", head_for("nothing"))
    with pytest.raises(FileNotFoundError):
        ob.bybit_url("2026-10-16")


def test_download_keeps_files_cuts_a_server_that_ignores_the_range_and_never_refetches(
        tmp_path, monkeypatch):
    seen = []

    def fake_get(url, headers=None, stream=False, timeout=None):
        seen.append(headers.get("Range"))
        return _Resp(chunks=[b"a" * 600_000, b"b" * 600_000, b"c" * 600_000])

    monkeypatch.setattr(ob.requests, "get", fake_get)
    dest = tmp_path / "x" / "f.gz"
    r = ob._download("u", dest, prefix_mb=1)
    assert dest.stat().st_size == 1_000_000 and not r["cached"] and seen == ["bytes=0-999999"]
    assert not list(dest.parent.glob("*.part"))
    assert ob._download("u", dest, prefix_mb=1)["cached"] and len(seen) == 1
    whole = ob._download("u", tmp_path / "x" / "g.gz")
    assert whole["bytes"] == 1_800_000 and seen[-1] is None


def test_iex_index_is_fetched_once_a_day_and_latest_day_needs_the_feed(tmp_path, monkeypatch):
    index = {"20260105": [{"feed": "DEEP", "link": "u5", "size": "100"}],
             "20260106": [{"feed": "TOPS", "link": "t6", "size": "7"}],
             "20260107": [{"feed": "TOPS", "link": "t7", "size": "7"},
                          {"feed": "DEEP", "link": "u7", "size": "300"}]}
    hits = []

    def fake_get(url, headers=None, stream=False, timeout=None):
        hits.append(url)
        return _Resp(text=json.dumps(index)) if url == ob.IEX_INDEX else _Resp(chunks=[b"x" * 3000])

    monkeypatch.setattr(ob.requests, "get", fake_get)
    assert ob.latest_iex_day(store=tmp_path) == "2026-01-07"
    index.pop("20260107")
    assert ob.latest_iex_day("DEEP", store=tmp_path) == "2026-01-07"   # read from disk, not fetched
    assert hits.count(ob.IEX_INDEX) == 1
    r = ob.fetch_iex("2026-01-07", mb=0.002, store=tmp_path)
    assert r["full_bytes"] == 300 and r["bytes"] == 2000 and not r["cached"]
    assert r["path"].name == "2026-01-07_DEEP_first0.002MB.pcap.gz"
    with pytest.raises(FileNotFoundError):
        ob.fetch_iex("2026-01-06", store=tmp_path)                        # only TOPS that day
