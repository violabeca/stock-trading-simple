"""Offline tests for scripts/finviz_ratings.py: the table parser and the DB-shaped mapping."""

import importlib.util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_spec = importlib.util.spec_from_file_location(
    "finviz_ratings", Path(__file__).resolve().parents[1] / "scripts" / "finviz_ratings.py")
fz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fz)

_ROW = ('<tr class="styled-row is-hoverable"><td align="left">{d}</td><td align="left">'
        '<span class="fv-label is-small">{a}</span></td><td>{f}</td><td>{r}</td>'
        '<td class="tabular-nums">{t}</td>\n</tr>')
PAGE = ('<table class="js-table-ratings styled-table-new"><thead><tr><th>Date</th></tr></thead>'
        + _ROW.format(d="Today", a="Upgrade", f="Citigroup", r="Neutral &rarr; Buy", t="$45")
        + _ROW.format(d="Mar-24-25", a="Downgrade", f="Goldman", r="Neutral &rarr; Sell",
                      t="$40 &rarr; $32")
        + _ROW.format(d="Mar-10-25", a="Initiated", f="Rosenblatt", r="Buy", t="$60")
        + _ROW.format(d="Feb-27-25", a="Reiterated", f="Barclays", r="Equal Weight", t="$1,059")
        + _ROW.format(d="Feb-11-25", a="Upgrade", f="CFRA", r="Hold &rarr; Buy", t="")
        + "</table>")


def test_parse_and_assemble_map_to_the_db_columns(tmp_path):
    (tmp_path / "raw").mkdir()
    fz.parse(PAGE).to_csv(tmp_path / "raw" / "SMCI.csv", index=False)
    fz.parse("<html>no table</html>").to_csv(tmp_path / "raw" / "NONE.csv", index=False)
    ratings, pts = fz.assemble(tmp_path)
    assert ratings.iloc[0]["rated_at"] == str(datetime.now(ZoneInfo("America/New_York")).date())
    ratings, pts = ratings.iloc[1:], pts.iloc[1:]  # "Today" is a date like any other from here
    assert ratings["action"].tolist() == ["downgrade", "initiate", "maintain", "upgrade"]
    assert ratings.iloc[0][["grade_from", "grade_to"]].tolist() == ["Neutral", "Sell"]
    assert ratings.iloc[1]["grade_from"] is None
    assert str(ratings.iloc[1]["rated_at"]) == "2025-03-10"
    assert str(ratings.iloc[0]["known_at"]) == "2025-03-24 23:59:59+00:00"
    assert pts["pt_action"].fillna("").tolist() == ["Lowers", "Announces", ""]  # lone: no prior
    assert pts["pt_current"].tolist() == [32.0, 60.0, 1059.0]
    assert (tmp_path / "analyst_price_targets_finviz.csv").exists()
