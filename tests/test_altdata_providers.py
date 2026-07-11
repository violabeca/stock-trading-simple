"""Pure alt-data provider parsers: SEC Form 4, FINRA short interest, yfinance ratings.

No network — each exercises the parse/transform half of a provider on a fixture.
"""

from datetime import UTC, date, datetime

import pandas as pd

from trading_simple.providers import finra, sec, yf

# --- SEC Form 4 -------------------------------------------------------------
_FORM4 = b"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>DOE JANE</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>1</isDirector>
      <isOfficer>1</isOfficer>
      <officerTitle>CEO</officerTitle>
      <isTenPercentOwner>0</isTenPercentOwner>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-03-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>150.25</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-03-02</value></transactionDate>
      <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>500</value></transactionShares>
        <transactionPricePerShare><value>152.00</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>D</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2024-03-03</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>200</value></transactionShares>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>"""

_KNOWN = datetime(2024, 3, 4, 18, 3, 1, tzinfo=UTC)


def test_parse_form4_keeps_only_open_market_and_maps_side():
    trades = sec._parse_form4(_FORM4, "AAPL", "0001-24-000001",
                              known_at=_KNOWN, filed_at=_KNOWN)
    # The 'A' grant is dropped; only the P (buy) and S (sell) survive.
    assert [t.txn_type for t in trades] == ["buy", "sell"]
    buy = trades[0]
    assert buy.symbol == "AAPL" and buy.shares == 1000.0 and buy.price == 150.25
    assert buy.traded_at == date(2024, 3, 1)
    assert buy.known_at == _KNOWN and buy.accession == "0001-24-000001"
    assert buy.insider == "DOE JANE" and "Director" in buy.role and "CEO" in buy.role


def test_parse_form4_strips_namespace():
    ns = _FORM4.replace(b"<ownershipDocument>",
                        b'<ownershipDocument xmlns="http://www.sec.gov/edgar/ownership">')
    trades = sec._parse_form4(ns, "AAPL", "acc", known_at=_KNOWN, filed_at=_KNOWN)
    assert [t.txn_type for t in trades] == ["buy", "sell"]


def test_raw_doc_strips_xsl_prefix():
    # primaryDocument points at the XSL-styled HTML; the raw XML is at the root.
    assert sec._raw_doc("xslF345X06/form4.xml") == "form4.xml"
    assert sec._raw_doc("wk-form4_123.xml") == "wk-form4_123.xml"    # already raw


# --- FINRA short interest ---------------------------------------------------
def test_finra_records_parse_and_derive_dtc():
    records = [
        {"symbolCode": "AAPL", "settlementDate": "2024-01-31",
         "currentShortPositionQuantity": "1000000", "averageDailyVolumeQuantity": "250000",
         "daysToCoverQuantity": ""},                       # dtc blank -> derived
        {"symbolCode": "msft", "settlementDate": "2024-01-31",
         "currentShortPositionQuantity": "600000", "averageDailyVolumeQuantity": "300000",
         "daysToCoverQuantity": "2.0"},                    # dtc provided
        {"symbolCode": None, "settlementDate": "2024-01-31"},   # dropped (no symbol)
    ]
    rows = finra._records_to_rows(records)
    assert len(rows) == 2
    aapl = rows[0]
    assert aapl.symbol == "AAPL" and aapl.days_to_cover == 4.0        # 1e6 / 250000
    assert rows[1].symbol == "MSFT" and rows[1].days_to_cover == 2.0  # symbol upcased
    # known_at is FINRA dissemination: several business days after settlement, evening UTC.
    assert aapl.known_at.date() > date(2024, 1, 31) and aapl.known_at.hour == 21
    assert aapl.disseminated_at == aapl.known_at.date()


# --- yfinance analyst ratings ----------------------------------------------
def _ratings_frame():
    idx = pd.DatetimeIndex(["2019-06-01", "2021-02-10", "2021-05-20"], name="GradeDate")
    return pd.DataFrame({
        "Firm": ["Old Co", "Big Bank", "Boutique"],
        "ToGrade": ["Buy", "Overweight", "Sell"],
        "FromGrade": ["", "Equal-Weight", "Hold"],
        "Action": ["init", "up", "down"],
    }, index=idx)


def test_ratings_frame_maps_action_and_filters_since():
    rows = yf._ratings_frame_to_rows("AAPL", _ratings_frame(), date(2020, 1, 1))
    # The 2019 row is before `since` and dropped; two remain, actions normalized.
    assert [r.action for r in rows] == ["upgrade", "downgrade"]
    up = rows[0]
    assert up.symbol == "AAPL" and up.firm == "Big Bank" and up.grade_to == "Overweight"
    assert up.rated_at == date(2021, 2, 10)
    assert up.known_at == datetime(2021, 2, 10, 23, 59, 59, tzinfo=UTC)
