"""Offline tests of strategies/burst_scan/options_free.py (no network: fetchers are stubbed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_P = Path(__file__).resolve().parents[1] / "strategies" / "burst_scan" / "options_free.py"
_spec = importlib.util.spec_from_file_location("options_free", _P)
of = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(of)


def test_parse_occ_symbol_handles_roots_strikes_and_nasdaq_urls():
    assert of.parse_occ_symbol("MU261002C00005000") == ("MU", "2026-10-02", "C", 5.0)
    assert of.parse_occ_symbol("2MU261016P01077500") == ("2MU", "2026-10-16", "P", 1077.5)
    assert of.parse_occ_symbol("mu----261002c00005000") == ("MU", "2026-10-02", "C", 5.0)
    assert of.parse_occ_symbol("garbage") is None


def test_parse_occ_series_gives_call_and_put_open_interest():
    text = ("Series Search Results for MU\n\nProductSymbol\tyear\tMonth\tDay\tInteger\tDec\tC/P\t"
            "Call\tPut\tPosition Limit\t\n"
            "MU    \t\t2026\t10\t02\t5\t000\tC P \t55\t23\t25000000\n"
            "MU    \t\t2026\t10\t16\t1077\t500\tC P \t7\t0\t25000000\n")
    rows = of.parse_occ_series(text)
    assert rows[0] == {"root": "MU", "expiration": "2026-10-02", "cp": "C", "strike": 5.0,
                       "open_interest": 55.0}
    assert rows[1]["cp"] == "P" and rows[1]["open_interest"] == 23.0
    assert rows[2]["strike"] == 1077.5 and len(rows) == 4


def test_parse_nasdaq_splits_rows_and_reads_dashes_and_commas():
    rows = [{"expirygroup": "October 2, 2026", "drillDownURL": None},
            {"c_Bid": "1,084.65", "c_Ask": "1,099.25", "c_Last": "1,036.54", "c_Volume": "70",
             "c_Openinterest": "20", "p_Bid": "--", "p_Ask": "0.01", "p_Last": "0.01",
             "p_Volume": "3", "p_Openinterest": "23",
             "drillDownURL": "/market-activity/stocks/mu/option-chain/call-put-options/"
                             "mu----261002c00005000"}]
    out = of.parse_nasdaq(rows)
    assert len(out) == 2
    assert out[0]["bid"] == 1084.65 and out[0]["open_interest"] == 20
    assert np.isnan(out[1]["bid"]) and out[1]["cp"] == "P" and out[1]["strike"] == 5.0


def test_clean_yahoo_marks_off_hours_placeholders_missing():
    df = of._frame([{"root": "X", "expiration": "2026-10-16", "strike": 5.0, "cp": "C",
                     "bid": 0.0, "ask": 0.0, "last": 8.1, "open_interest": 0, "iv": 1e-5},
                    {"root": "X", "expiration": "2026-10-16", "strike": 6.0, "cp": "C",
                     "bid": 1.0, "ask": 1.2, "last": 1.1, "open_interest": 0, "iv": 0.4}],
                   "yfinance", "X", "t")
    out = of.clean_yahoo(df)
    assert out["bid"].isna().iloc[0] and out["iv"].isna().iloc[0]
    assert out["bid"].iloc[1] == 1.0 and out["iv"].iloc[1] == 0.4
    assert out["open_interest"].isna().all()          # a chain-wide zero OI is a placeholder
    assert out["last"].iloc[0] == 8.1                  # the last trade is real


def test_compare_matches_contracts_and_measures_agreement():
    base = {"root": "X", "expiration": "2026-10-16", "cp": "C"}
    a = of._frame([{**base, "strike": k, "bid": k, "open_interest": 10 * k} for k in range(1, 9)],
                  "a", "X", "t")
    b = of._frame([{**base, "strike": k, "bid": k + (1 if k == 8 else 0),
                    "open_interest": 10 * k} for k in range(1, 10)], "b", "X", "t")
    c = of.compare({"a": a, "b": b}, fields=["bid", "open_interest"])
    bid = c[c.field == "bid"].iloc[0]
    assert bid.matched == 8 and bid.agree_share == pytest.approx(7 / 8)
    assert c[c.field == "open_interest"].iloc[0].agree_share == 1.0


def test_iv_features_pick_the_30_day_expiry_and_the_delta_points():
    rows = []
    for exp, iv0 in (("2026-01-31", 0.50), ("2026-01-08", 0.70)):   # 30 and 7 days out
        for cp, deltas in (("C", (0.5, 0.25)), ("P", (-0.5, -0.25))):
            for dl in deltas:
                iv = iv0 + (0.05 if (cp == "P" and dl == -0.25) else 0.0)
                rows.append({"date": "2026-01-01", "underlying": "X", "expiration": exp,
                             "cp": cp, "delta": dl, "iv": iv})
    f = of.iv_features(pd.DataFrame(rows)).iloc[0]
    assert f.dte == 30 and f.atm_iv == 0.50
    assert f["skew"] == pytest.approx(0.05) and f["term"] == pytest.approx(0.20)
    # with quotes: a contract nobody bids, or with a spread over half its mid, is skipped
    junk = {"date": "2026-01-01", "underlying": "X", "expiration": "2026-01-31", "cp": "C",
            "delta": 0.5, "iv": 0.99}
    quoted = [{**r, "bid": 1.0, "ask": 1.1} for r in rows]
    for bid, ask in ((0.0, 0.5), (1.0, 3.0)):
        f = of.iv_features(pd.DataFrame([{**junk, "bid": bid, "ask": ask}, *quoted])).iloc[0]
        assert f.atm_iv == 0.50


def test_occ_features_ratio_and_surge_are_causal():
    days = pd.bdate_range("2026-01-01", periods=30).strftime("%Y-%m-%d")
    rows = []
    for i, d in enumerate(days):
        q = 100 if i < 29 else 400
        rows += [{"date": d, "underlying": "X", "actype": "C", "porc": "C", "quantity": q},
                 {"date": d, "underlying": "X", "actype": "C", "porc": "P", "quantity": q},
                 {"date": d, "underlying": "X", "actype": "M", "porc": "C", "quantity": q}]
    f = of.occ_features(pd.DataFrame(rows)).set_index("date")
    assert f["cust_pc"].iloc[0] == pytest.approx(0.0)
    assert f["cust_share"].iloc[0] == pytest.approx(2 / 3)
    assert f["vol_surge"].iloc[-1] == pytest.approx(np.log(4))   # vs the PREVIOUS 20 days
    assert f["vol_surge"].iloc[:10].isna().all()


def test_backfill_skips_stored_days_and_keeps_old_holidays(tmp_path, monkeypatch):
    calls = []

    def fake(day, symbols):
        calls.append(day)
        return pd.DataFrame({"x": [1]}) if day != "2025-01-08" else pd.DataFrame()

    monkeypatch.setitem(of.HISTORY, "occ_volume", fake)
    log = of.backfill("occ_volume", "2025-01-06", "2025-01-10", store=tmp_path)
    assert len(log) == 5 and len(calls) == 5
    assert (tmp_path / "occ_volume" / "2025-01-08.csv.gz").exists()   # empty, old: stored
    calls.clear()
    of.backfill("occ_volume", "2025-01-06", "2025-01-10", store=tmp_path)
    assert calls == []                                                 # nothing asked twice
    hist = of.load_history("occ_volume", store=tmp_path)               # empty marker is skipped
    assert len(hist) == 4 and set(hist["date"]) == {"2025-01-06", "2025-01-07", "2025-01-09",
                                                     "2025-01-10"}
    assert of.inventory(store=tmp_path).set_index("source").loc["occ_volume", "days_with_data"] == 4


def test_ic_table_finds_a_planted_relation():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2025-01-01", periods=200).strftime("%Y-%m-%d")
    df = pd.DataFrame([{"date": d, "underlying": s, "f": rng.normal()} for d in days
                       for s in ("A", "B")])
    df["ret_1d"] = 0.5 * df["f"] + rng.normal(size=len(df))
    t = of.ic_table(df, ["f"], ["ret_1d"]).iloc[0]
    assert t.ic > 0.3 and t.t > 5 and t.days == 200


def test_parse_marketdata_columns_to_contract_rows():
    j = {"s": "ok", "optionSymbol": ["MU251017P00035000", "MU251017C00200000"],
         "underlying": ["MU", "MU"], "expiration": [1760731200, 1760731200],
         "side": ["put", "call"], "strike": [35, 200], "bid": [0.0, 3.1], "bidSize": [0, 5],
         "ask": [0.02, 3.3], "askSize": [1, 7], "last": [0.04, 3.2], "openInterest": [83, 1200],
         "volume": [0, 450], "underlyingPrice": [183.75, 183.75], "iv": [None, None],
         "delta": [None, None], "gamma": [None, None], "theta": [None, None],
         "vega": [None, None]}
    df = of.parse_marketdata(j, "2025-10-02")
    assert list(df["cp"]) == ["P", "C"] and list(df["expiration"]) == ["2025-10-17"] * 2
    assert df["open_interest"].tolist() == [83, 1200] and df["root"].tolist() == ["MU", "MU"]
    assert df["iv"].isna().all() and df["ask_size"].tolist() == [1, 7]
    assert of.parse_marketdata({"s": "no_data"}, "2025-10-02").empty


def test_md_features_put_call_ratios_and_gamma_sign():
    base = {"date": "2026-01-02", "underlying": "X", "expiration": "2026-01-30",
            "underlying_price": 100.0}
    rows = [{**base, "cp": "C", "strike": 100.0, "open_interest": 300, "volume": 30},
            {**base, "cp": "P", "strike": 100.0, "open_interest": 100, "volume": 90}]
    f = of.md_features(pd.DataFrame(rows)).iloc[0]
    assert f["oi_pc"] == pytest.approx(np.log(101 / 301))
    assert f["vol_pc"] == pytest.approx(np.log(91 / 31))
    assert f["gex"] == pytest.approx(0.5)          # (300 - 100) / (300 + 100), same strike/gamma
    rows[1]["open_interest"] = 900
    assert of.md_features(pd.DataFrame(rows)).iloc[0]["gex"] == pytest.approx(-0.5)
