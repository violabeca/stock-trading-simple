"""Offline tests of strategies/orderbook/iex_features.py (synthetic messages, no network)."""

import importlib.util
import struct
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_DIR = Path(__file__).resolve().parents[1] / "strategies" / "orderbook"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ob, feat = _load("orderbook_free"), _load("iex_features")


def _sym(s):
    return s.ljust(8).encode()


def _quotes(rows):
    """A quote frame from (t seconds, bid, bid_qty, ask, ask_qty) rows."""
    df = pd.DataFrame(rows, columns=["t_ns", "bid", "bid_qty", "ask", "ask_qty"])
    df["t_ns"] = df["t_ns"] * 10**9
    df["mid"] = (df["bid"] + df["ask"]) / 2
    return df


# a hand-worked sequence of top-of-book states for the order-flow rule
_STATES = [(1, 100.0, 10, 101.0, 10), (2, 100.0, 15, 101.0, 10), (3, 100.5, 7, 101.0, 10),
           (4, 100.0, 12, 101.0, 10), (5, 100.0, 12, 101.5, 4), (6, 100.0, 12, 101.0, 6)]


def test_rhythm_frame_groups_the_kinds_and_stamps_the_bucket_start():
    df = feat.rhythm_frame({0: Counter({"8": 3, "5": 2, "T": 1}),
                            2: Counter({"I": 4, "H": 1, "A": 7})}, 10)
    assert list(df.index.astype("int64") // 10**9) == [0, 20]
    assert (df["book updates"].iloc[0], df["trades"].iloc[0]) == (5, 1)
    assert (df["retail interest flags"].iloc[0], df["retail interest flags"].iloc[1]) == (0, 4)
    assert df["status messages"].iloc[1] == 1
    assert df["auction, official price, directory, system"].iloc[1] == 7


def test_catalogue_names_agree_with_the_decoder_and_cover_every_type():
    names = {chr(k): v for k, v in ob.TYPE_NAMES.items()}
    assert {k: v[0] for k, v in feat.CATALOGUE.items()} == names


def test_status_tables_decode_every_kind_of_message():
    t = 1_700_000_000_000_000_000
    others = {
        "S": [struct.pack("<BBq", ord("S"), ord("R"), t)],
        "H": [struct.pack("<BBq8s4s", ord("H"), ord("H"), t, _sym("ABC"), b"NA  ")],
        "O": [struct.pack("<BBq8s", ord("O"), ord("N"), t, _sym("ABC"))],
        "P": [struct.pack("<BBq8sB", ord("P"), 1, t, _sym("ABC"), ord("N"))],
        "E": [struct.pack("<BBq8s", ord("E"), ord("O"), t, _sym("ABC"))],
        "I": [struct.pack("<BBq8s", ord("I"), ord("B"), t, _sym("ABC"))],
        "X": [struct.pack("<BBq8sq", ord("X"), ord("Q"), t, _sym("ZIEXT"), 1_000_000)],
        "D": [struct.pack("<BBq8sIqB", ord("D"), 0x80, t, _sym("ZIEXT"), 100, 990_000, 1)],
        "A": [struct.pack("<Bcq8sIqqIcBIqqqq", 0x41, b"O", t, _sym("ZIEXT"), 1000, 10_000, 10_100,
                          50, b"B", 0, 1_700_000_000, 10_200, 10_000, 9_000, 11_000)],
    }
    tabs = feat.status_tables(others)
    assert set(tabs) == {"system_event", "trading_status", "operational_halt", "short_sale",
                         "security_event", "retail_interest", "official_price",
                         "security_directory", "auction"}

    def row(name):
        return tabs[name].iloc[0].to_dict()

    assert row("system_event") == {"t_ns": t, "symbol": "", "event": "R"}
    assert row("trading_status")["status"] == "H" and row("trading_status")["reason"] == "NA"
    assert row("operational_halt")["status"] == "N" and row("security_event")["event"] == "O"
    assert (row("short_sale")["in_effect"], row("short_sale")["detail"]) == (1, "N")
    assert row("retail_interest")["indicator"] == "B"
    assert (row("official_price")["kind"], row("official_price")["price"]) == ("Q", 100.0)
    d = row("security_directory")
    assert (d["flags"], d["round_lot"], d["prior_close"], d["luld_tier"]) == (0x80, 100, 99.0, 1)
    a = row("auction")
    assert (a["auction_type"], a["paired_shares"], a["reference_price"]) == ("O", 1000, 1.0)
    assert (a["imbalance_shares"], a["imbalance_side"], a["scheduled_s"]) == (
        50, "B", 1_700_000_000)
    assert (a["lower_collar"], a["upper_collar"]) == (0.9, 1.1)


def test_status_tables_are_empty_not_missing_when_a_kind_never_occurs():
    t = 1_700_000_000_000_000_000
    tabs = feat.status_tables({"O": [struct.pack("<BBq8s", ord("O"), ord("N"), t, _sym("ABC"))]})
    assert len(tabs) == 9 and len(tabs["operational_halt"]) == 1
    quiet = tabs["retail_interest"]
    assert quiet.empty and list(quiet.columns) == ["t_ns", "symbol", "indicator"]
    assert quiet.query("indicator != ' '")["symbol"].nunique() == 0
    assert all(tab.empty for tab in feat.status_tables({}).values())


def test_book_features_and_quotes_collapse_unchanged_states_and_follow_snapshots():
    def m(t, bids=(), asks=(), snap=False):
        return ob.Msg(t, snap, list(bids), list(asks))

    events = [m(1, [(100.0, 3.0), (99.0, 5.0)]),            # no ask yet: no quote
              m(2, asks=[(101.0, 1.0), (102.0, 4.0)]),       # first two-sided state
              m(3, [(100.0, 3.0)]),                          # nothing changes: no new row
              m(4, [(100.0, 0.0)]),                          # the best bid is removed
              m(5, [(10.0, 1.0)], [(11.0, 1.0)], snap=True)]    # a snapshot replaces the book
    q = feat.quotes(events)
    assert list(q["t_ns"]) == [2, 4, 5]
    r = q.iloc[0]
    assert (r.bid, r.bid_qty, r.ask, r.ask_qty, r.bid_qty_k, r.ask_qty_k) == (100, 3, 101, 1, 8, 5)
    assert r.mid == 100.5 and r.spread_bps == pytest.approx(1 / 100.5 * 1e4)
    assert r.micro == pytest.approx((100 * 1 + 101 * 3) / 4)       # pulled toward the thin ask
    assert r.obi1 == pytest.approx(0.5) and r.obik == pytest.approx(3 / 13)
    assert (q["bid"].iloc[1], q["bid_qty"].iloc[1], q["bid_qty_k"].iloc[1]) == (99, 5, 5)
    assert (q["bid"].iloc[2], q["ask"].iloc[2]) == (10, 11)
    assert feat.book_features({}, {1.0: 1.0}) is None and feat.quotes([]).empty


def test_ofi_follows_the_published_rule_side_by_side():
    q = _quotes(_STATES)
    parts = feat.ofi(q, parts=True)
    assert list(parts["ofi"]) == [0, 5, 7, -7, 10, -6]
    assert list(parts["e_bid"]) == [0, 5, 7, -7, 0, 0]       # size added, bid lifted, bid dropped
    assert list(parts["e_ask"]) == [0, 0, 0, 0, 10, -6]      # ask lifted away, ask lowered
    assert list(feat.ofi(q)) == list(parts["ofi"])


def test_bucket_ofi_sums_the_flow_and_measures_the_mid_change_of_each_bucket():
    q = _quotes(_STATES)
    b = feat.bucket_ofi(q, 0, 6 * 10**9, 1)
    assert list(b["t_ns"]) == [1e9, 2e9, 3e9, 4e9, 5e9]     # bucket 0 has no mid before it
    assert list(b["ofi"]) == [5, 7, -7, 10, -6]
    assert list(b["dmid"]) == pytest.approx([0, 0.25, -0.25, 0.25, -0.25])
    assert feat.bucket_ofi(q.iloc[:0], 0, 10**9, 1).empty


def test_sign_trades_uses_the_quote_strictly_before_the_trade():
    q = _quotes([(10, 100.0, 5, 101.0, 5), (20, 100.0, 5, 102.0, 5)])
    sec = 10**9
    trades = pd.DataFrame({"t_ns": [5, 15, 20, 25, 30, 35]})
    trades["t_ns"] *= sec
    trades["price"] = [100.5, 101.0, 100.0, 101.0, 100.5, 100.0]
    trades["size"] = 100
    out = feat.sign_trades(trades, q)
    assert list(out["where"]) == [
        "no quote",                  # before any quote
        "at or above the ask",       # 101 >= ask 101
        "at or below the bid",       # t=20 = the quote change's own time: the OLD quote is used
        "at the mid",                # new quote 100 x 102, mid 101
        "inside, below the mid",
        "at or below the bid"]
    assert list(out["side"]) == [0, 1, -1, 1, -1, -1]       # the 4th is at the mid: 101 > 100 = buy
    assert set(feat.WHERE) >= set(out["where"])


def test_ladder_keeps_levels_near_the_mid_and_leaves_empty_moments_empty():
    books = {1 * 10**9: ({100.00: 5.0, 99.99: 3.0}, {100.02: 4.0}),
             2 * 10**9: ({}, {}),
             3 * 10**9: ({100.01: 7.0, 90.0: 1.0}, {100.03: 2.0, 100.05: 6.0})}
    times, prices, bid_z, ask_z = feat.ladder(books, tick=0.01, max_ticks=3)
    assert list(times) == [1e9, 2e9, 3e9] and bid_z.shape == ask_z.shape == (7, 3)
    assert prices[0] == pytest.approx(99.99) and prices[-1] == pytest.approx(100.05)
    assert bid_z[1, 0] == 5 and bid_z[0, 0] == 3 and bid_z[2, 2] == 7     # 100.00, 99.99, 100.01
    assert ask_z[3, 0] == 4 and ask_z[4, 2] == 2 and ask_z[6, 2] == 6     # 100.02, 100.03, 100.05
    assert np.isnan(bid_z[:, 1]).all() and np.isnan(ask_z[:, 1]).all()
    assert np.isnan(bid_z).sum() == 21 - 3 and 90.0 not in prices        # the far quote is dropped
    assert feat.ladder({1: ({}, {})})[1].size == 0
