"""Offline tests for scripts/marketdata_dump.py: what a resumed run plans and how chunks close."""

import importlib.util
from datetime import date
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "marketdata_dump", Path(__file__).resolve().parents[1] / "scripts" / "marketdata_dump.py")
md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(md)


def test_plan_fetches_only_uncovered_days_oldest_month_first(tmp_path):
    md.write(pd.DataFrame({"t": [1]}), tmp_path / "chains" / "AAA" / "2026-01-05_2026-01-20.csv.gz")
    md.write(pd.DataFrame(), tmp_path / "news" / "AAA" / "2026-02-01_2026-02-10.csv.gz")
    tasks = md.plan(["AAA", "BBB"], ["chains", "news"], date(2026, 1, 5), date(2026, 2, 10),
                    tmp_path)
    assert tasks[:4] == [
        ("chains", "AAA", date(2026, 1, 21), date(2026, 1, 31)),  # the rest of a partial month
        ("chains", "BBB", date(2026, 1, 5), date(2026, 1, 31)),
        ("news", "AAA", date(2026, 1, 5), date(2026, 1, 31)),
        ("news", "BBB", date(2026, 1, 5), date(2026, 1, 31)),
    ]
    assert ("news", "AAA", date(2026, 2, 1), date(2026, 2, 10)) not in tasks  # empty = asked
    assert ("news", "BBB", date(2026, 2, 1), date(2026, 2, 10)) == tasks[-1]
    assert md.load("chains", out=tmp_path)["symbol"].tolist() == ["AAA"]


def test_fetch_closes_a_recent_chunk_at_its_last_session(tmp_path, monkeypatch):
    served = {"2026-10-05", "2026-10-06"}  # later sessions are not published yet
    monkeypatch.setattr(md, "call", lambda path, p, tok, res: (
        {"s": "ok", "optionSymbol": ["X"], "volume": [1]} if p["date"] in served
        else {"s": "no_data"}))
    rows = md.fetch("chains", "AAA", date(2026, 10, 1), date(2026, 10, 9), "", 0,
                    date(2026, 10, 10), tmp_path)
    assert rows == 2
    assert [f.name for f in (tmp_path / "chains" / "AAA").iterdir()] == [
        "2026-10-01_2026-10-06.csv.gz"]
    served.clear()  # a recent chunk with nothing yet is left for the next run, not marked done
    assert md.fetch("chains", "AAA", date(2026, 10, 7), date(2026, 10, 9), "", 0,
                    date(2026, 10, 10), tmp_path) == 0
    assert len(list((tmp_path / "chains" / "AAA").iterdir())) == 1
