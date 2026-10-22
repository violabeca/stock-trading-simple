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


# --- SEC bulk Form 3/4/5 data sets ------------------------------------------
_SUBMISSION = (
    b"ACCESSION_NUMBER\tFILING_DATE\tDOCUMENT_TYPE\tISSUERTRADINGSYMBOL\n"
    b"acc-p-s\t05-FEB-2024\t4\tAAPL\n"       # kept: AAPL Form 4 after `since`
    b"acc-amend\t06-FEB-2024\t4/A\tAAPL\n"   # dropped: amendment (double-count risk)
    b"acc-old\t05-FEB-2020\t4\tAAPL\n"       # dropped: before `since`
    b"acc-msft\t05-FEB-2024\t4\tMSFT\n"      # dropped: not in the symbol set
)
_OWNERS = (
    b"ACCESSION_NUMBER\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
    b"acc-p-s\tDOE JANE\tDirector,Officer\tCTO\n"
    b"acc-msft\tROE JOHN\tTenPercentOwnerOther\t\n"
)
_TRANS = (
    b"ACCESSION_NUMBER\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\n"
    b"acc-p-s\t01-FEB-2024\tP\t1000\t150.25\n"
    b"acc-p-s\t02-FEB-2024\tS\t500\t152.00\n"
    b"acc-p-s\t03-FEB-2024\tA\t200\t\n"       # dropped: 'A' grant is not open-market
    b"acc-amend\t01-FEB-2024\tP\t999\t10.00\n"
    b"acc-old\t01-FEB-2020\tP\t111\t9.00\n"
    b"acc-msft\t01-FEB-2024\tP\t50\t400.00\n"
)


def test_parse_form345_tables_filters_and_maps():
    trades = sec._parse_form345_tables(_SUBMISSION, _OWNERS, _TRANS, {"AAPL"}, date(2024, 1, 1))
    # Only the AAPL Form 4's P (buy) + S (sell) survive: the 'A' grant, the 4/A
    # amendment, the pre-`since` filing, and the MSFT filing are all dropped.
    assert [t.txn_type for t in trades] == ["buy", "sell"]
    buy, sell = trades
    assert buy.symbol == "AAPL" and buy.shares == 1000.0 and buy.price == 150.25
    assert buy.traded_at == date(2024, 2, 1) and buy.accession == "acc-p-s"
    assert buy.insider == "DOE JANE" and buy.role == "Director/CTO"
    assert buy.source == "sec_edgar_bulk"   # distinct provenance from the per-filing path
    # known_at is the FILING_DATE at 21:00 UTC (conservative point-in-time), NOT trade date.
    assert buy.known_at == datetime(2024, 2, 5, 21, tzinfo=UTC) and buy.filed_at == buy.known_at
    assert sell.shares == 500.0 and sell.traded_at == date(2024, 2, 2)


def test_parse_form345_tables_respects_until_upper_bound():
    # `until` before the only kept filing (2024-02-05) drops it; on/after keeps it.
    assert sec._parse_form345_tables(_SUBMISSION, _OWNERS, _TRANS, {"AAPL"},
                                     date(2024, 1, 1), date(2024, 1, 31)) == []
    trades = sec._parse_form345_tables(_SUBMISSION, _OWNERS, _TRANS, {"AAPL"},
                                       date(2024, 1, 1), date(2024, 2, 5))
    assert [t.txn_type for t in trades] == ["buy", "sell"]


def test_role_from_relationship_decodes_concatenated_tokens():
    # 'TenPercentOwnerOther' has no delimiter; substring decode still finds the role.
    assert sec._role_from_relationship("TenPercentOwnerOther", "") == "10% owner"
    assert sec._role_from_relationship("Director,Officer", "CEO") == "Director/CEO"
    assert sec._role_from_relationship("", None) is None


def test_quarters_since_spans_year_boundary():
    assert sec._quarters_since(date(2024, 11, 15), date(2025, 2, 1)) == ["2024q4", "2025q1"]
    assert sec._quarters_since(date(2024, 1, 1), date(2024, 1, 31)) == ["2024q1"]


def test_cik_map_prefers_the_lowest_cik_on_ticker_collisions(monkeypatch):
    """A recently-registered shell must not shadow the operating company.

    EDGAR's directory listed XOM under both Exxon Mobil Corp (CIK 34088) and an
    "ExxonMobil Holdings Corp" shell (CIK 2115436); last-write-wins resolved to
    the shell and every SEC-sourced sleeve silently returned zero XOM rows.
    Operating companies carry old, low CIKs — the lowest must win either way
    the directory orders the rows.
    """
    directory = {
        "0": {"ticker": "XOM", "cik_str": 34088, "title": "EXXON MOBIL CORP"},
        "1": {"ticker": "XOM", "cik_str": 2115436, "title": "ExxonMobil Holdings Corp"},
        "2": {"ticker": "BRK-B", "cik_str": 1067983, "title": "BERKSHIRE HATHAWAY INC"},
    }
    monkeypatch.setattr(sec, "_cik_cache", None)
    monkeypatch.setattr(sec, "_get", lambda url, **kw: directory)
    assert sec._cik_map()["XOM"] == 34088
    assert sec._cik_map()["BRK.B"] == 1067983          # dotted alias still resolves
    monkeypatch.setattr(sec, "_cik_cache", None)       # shell FIRST — order must not matter
    monkeypatch.setattr(sec, "_get", lambda url, **kw: dict(reversed(directory.items())))
    assert sec._cik_map()["XOM"] == 34088


def test_parse_ddmonyyyy_handles_sec_format_and_junk():
    assert sec._parse_ddmonyyyy("05-FEB-2024") == date(2024, 2, 5)
    assert sec._parse_ddmonyyyy("") is None and sec._parse_ddmonyyyy("garbage") is None


# --- FINRA short interest ---------------------------------------------------
_BULK_HEADER = ("accountingYearMonthNumber|symbolCode|issueName|issuerServicesGroupExchangeCode|"
                "marketClassCode|currentShortPositionQuantity|previousShortPositionQuantity|"
                "stockSplitFlag|averageDailyVolumeQuantity|daysToCoverQuantity|revisionFlag|"
                "changePercent|changePreviousNumber|settlementDate")


def test_parse_bulk_file_is_pipe_delimited_and_feeds_records_to_rows():
    # Pipe-delimited despite the .csv name; columns match the Query-API field names.
    text = (_BULK_HEADER + "\n"
            "202401|AAPL|Apple Inc.|R|NNM|1000000|900000||250000||||100000|2024-01-31\n")
    recs = finra._parse_bulk_file(text)
    assert recs[0]["symbolCode"] == "AAPL" and recs[0]["settlementDate"] == "2024-01-31"
    rows = finra._records_to_rows(recs)                       # the shipped parser, reused as-is
    assert rows[0].symbol == "AAPL" and rows[0].days_to_cover == 4.0   # 1e6/250000, derived


def test_a_quote_in_an_issuer_name_cannot_swallow_the_file():
    """The bug that silently cost this repo the 2018-2019 short-interest history.

    FINRA's pipe format has no quoting convention, so a ``"`` in an issuer name is data.
    Read with :mod:`csv`'s default quoting, the unbalanced quotes below open a field that
    never closes and every later row is absorbed into it — surfacing as ``field larger
    than field limit`` on a file whose longest line is a few hundred characters, which
    aborted the backfill and left a two-year hole that read as "FINRA has no data".

    The rows after the offending one are what matters: losing them is the silent part.
    """
    text = (_BULK_HEADER + "\n"
            "202401|AAPL|Apple Inc.|R|NNM|1000000|900000||250000||||100000|2024-01-31\n"
            '201803|DOD|"ELEMENTS ""Dogs of the Dow""|E|ARCA|613|2318||4445|1.00|'
            "||-1705|2024-01-31\n"
            "202401|MSFT|Microsoft Corp|R|NNM|2000000|1900000||500000||||100000|2024-01-31\n")
    recs = finra._parse_bulk_file(text)
    assert [r["symbolCode"] for r in recs] == ["AAPL", "DOD", "MSFT"]
    # The row AFTER the quoted name survives intact — that is the regression.
    assert recs[2]["currentShortPositionQuantity"] == "2000000"
    assert {row.symbol for row in finra._records_to_rows(recs)} == {"AAPL", "DOD", "MSFT"}


def test_a_fully_quoted_bulk_file_parses_like_the_old_format():
    """FINRA quotes every field since the 2026-09-15 settlement; under QUOTE_NONE the quotes
    stayed in every key, ``symbolCode`` was never found and every file matched no symbol."""
    quoted = "|".join(f'"{c}"' for c in _BULK_HEADER.split("|")) + "\n" + (
        '"20260915"|"AAPL"|"Apple Inc."|"R"|"NNM"|"1000000"|"900000"||"250000"|"4.00"||||'
        '"2026-09-15"\n'
        '"20260915"|"DOD"|"ELEMENTS ""Dogs of the Dow"""|"E"|"ARCA"|"613"|"2318"||"4445"|"1.00"'
        '||||"2026-09-15"\n')
    recs = finra._parse_bulk_file(quoted)
    assert [r["symbolCode"] for r in recs] == ["AAPL", "DOD"]
    assert recs[1]["issueName"] == 'ELEMENTS "Dogs of the Dow"'
    assert finra._records_to_rows(recs)[0].days_to_cover == 4.0


def test_an_unreadable_or_unmatched_bulk_file_fails_loudly(monkeypatch):
    class _Resp:
        status_code = 200

        def __init__(self, text):
            self.text = text

    monkeypatch.setattr(finra, "settlement_dates", lambda since, until=None: [date(2026, 9, 15)])
    row = "202609|AAPL|Apple Inc.|R|NNM|1000000|900000||250000||||100000|2026-09-15\n"
    for body, symbols in (("bad|header\n1|2\n", ["AAPL"]),                 # format change
                          (_BULK_HEADER + "\n" + row, [f"S{i}" for i in range(60)])):  # no match
        monkeypatch.setattr("requests.get", lambda url, timeout, b=body: _Resp(b))
        try:
            list(finra.iter_short_interest_bulk(symbols, date(2026, 9, 1)))
        except ValueError:
            continue
        raise AssertionError("a file that matches nothing must raise, not yield zero rows")


def test_si_request_body_adds_upper_bound_only_when_until_set():
    open_body = finra._si_request_body(["aapl", "MSFT"], date(2024, 1, 1), None, 0)
    assert open_body["compareFilters"] == [
        {"fieldName": "settlementDate", "compareType": "gte", "fieldValue": "2024-01-01"}]
    assert open_body["domainFilters"][0]["values"] == ["AAPL", "MSFT"]   # upper-cased
    bounded = finra._si_request_body(["AAPL"], date(2024, 1, 1), date(2024, 6, 30), 1000)
    pairs = {(f["compareType"], f["fieldValue"]) for f in bounded["compareFilters"]}
    assert ("gte", "2024-01-01") in pairs and ("lte", "2024-06-30") in pairs
    assert bounded["offset"] == 1000


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


# --- SEC fundamentals (XBRL companyfacts) ----------------------------------
def _fact(end, val, filed, *, start=None, fy=None, fp=None, form=None, accn=None):
    """Build one raw companyfacts data point (``start`` present only for flows)."""
    f = {"end": end, "val": val, "filed": filed, "accn": accn, "fy": fy, "fp": fp, "form": form}
    if start is not None:
        f["start"] = start
    return f


#: Calendar fiscal periods (Q3 year-to-date and full year, 2022-2023) for the fixtures below.
_PERIODS = dict(
    q322=dict(start="2022-01-01", filed="2022-11-01", accn="q322", fy=2022, fp="Q3", form="10-Q"),
    fy22=dict(start="2022-01-01", filed="2023-02-15", accn="a22", fy=2022, fp="FY", form="10-K"),
    q323=dict(start="2023-01-01", filed="2023-11-01", accn="q323", fy=2023, fp="Q3", form="10-Q"),
    fy23=dict(start="2023-01-01", filed="2024-02-15", accn="a23", fy=2023, fp="FY", form="10-K"),
)


def _companyfacts():
    """A two-fiscal-year (calendar) companyfacts fixture exercising every branch.

    Includes instants (equity/assets/shares) with a re-reported comparative,
    income flows as FY + fiscal-YTD cumulatives, GrossProfit *absent* (so it is
    synthesized from Revenues - CostOfRevenue), and Revenues reached via the
    second tag fallback.
    """
    usd = "USD"
    q322, fy22, q323, fy23 = (_PERIODS[k] for k in ("q322", "fy22", "q323", "fy23"))
    return {"facts": {
        "us-gaap": {
            "StockholdersEquity": {"units": {usd: [
                _fact("2022-12-31", 5000, "2023-02-15", accn="a22", fy=2022, fp="FY", form="10-K"),
                _fact("2023-12-31", 5500, "2024-02-15", accn="a23", fy=2023, fp="FY", form="10-K"),
            ]}},
            "Assets": {"units": {usd: [
                _fact("2022-12-31", 20000, "2023-02-15", accn="a22", fy=2022, fp="FY", form="10-K"),
                _fact("2023-12-31", 22000, "2024-02-15", accn="a23", fy=2023, fp="FY", form="10-K"),
            ]}},
            "NetIncomeLoss": {"units": {usd: [
                _fact("2022-09-30", 700, **q322), _fact("2022-12-31", 1000, **fy22),
                _fact("2023-09-30", 900, **q323), _fact("2023-12-31", 1200, **fy23),
            ]}},
            # RevenueFromContractWithCustomerExcludingAssessedTax is ABSENT, so the
            # builder must fall through to the `Revenues` tag.
            "Revenues": {"units": {usd: [
                _fact("2022-09-30", 2100, **q322), _fact("2022-12-31", 3000, **fy22),
                _fact("2023-09-30", 2700, **q323), _fact("2023-12-31", 3600, **fy23),
            ]}},
            "CostOfRevenue": {"units": {usd: [
                _fact("2022-09-30", 1300, **q322), _fact("2022-12-31", 1800, **fy22),
                _fact("2023-09-30", 1500, **q323), _fact("2023-12-31", 2000, **fy23),
            ]}},
        },
        "dei": {
            "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                _fact("2022-12-31", 100, "2023-02-15", accn="a22", fy=2022, fp="FY", form="10-K"),
                _fact("2023-09-30", 105, "2023-11-01", accn="q323", fy=2023, fp="Q3", form="10-Q"),
                # the 2023 10-K re-reports the 2022 figure (a comparative) — must not duplicate:
                _fact("2022-12-31", 100, "2024-02-15", accn="a23", fy=2023, fp="FY", form="10-K"),
                _fact("2023-12-31", 110, "2024-02-15", accn="a23", fy=2023, fp="FY", form="10-K"),
            ]}},
        },
    }}


def _by_concept(rows):
    out: dict[str, list] = {}
    for r in rows:
        out.setdefault(r.concept, []).append(r)
    return out


def test_companyfacts_instants_are_monotonic_and_drop_comparatives():
    by = _by_concept(sec._parse_companyfacts(_companyfacts(), "AAPL"))
    shares = sorted(by["shares"], key=lambda r: r.period_end)
    # one row per successively-newer balance-sheet date; the 2023-10-K's re-report
    # of the 2022 figure creates no duplicate and never regresses the level.
    assert [(r.period_end, r.val) for r in shares] == [
        (date(2022, 12, 31), 100.0), (date(2023, 9, 30), 105.0), (date(2023, 12, 31), 110.0)]
    assert {r.period_end: r.val for r in by["equity"]}[date(2023, 12, 31)] == 5500.0


def test_companyfacts_ttm_is_annual_plus_ytd_delta_no_double_count():
    by = _by_concept(sec._parse_companyfacts(_companyfacts(), "AAPL"))
    ni = {r.period_end: r.val for r in by["ni_ttm"]}
    assert ni[date(2023, 12, 31)] == 1200.0        # 10-K: TTM == full year, no double count
    assert ni[date(2023, 9, 30)] == 1200.0         # FY22 + YTD23Q3 - YTD22Q3 = 1000+900-700
    assert date(2022, 9, 30) not in ni             # no prior-year data -> not emitted


def test_companyfacts_gross_profit_synthesized_from_revenue_minus_cogs():
    by = _by_concept(sec._parse_companyfacts(_companyfacts(), "AAPL"))
    gp = {r.period_end: r for r in by["gp_ttm"]}
    assert gp[date(2023, 12, 31)].val == 1600.0             # (3600-2000)
    assert gp[date(2023, 9, 30)].val == 1600.0              # 1200 + 1200 - 800
    assert all(r.tag == "computed:GrossProfit" for r in gp.values())


def test_companyfacts_known_at_is_filed_date_at_21_utc():
    rows = sec._parse_companyfacts(_companyfacts(), "AAPL")
    r = next(r for r in rows if r.concept == "equity" and r.period_end == date(2023, 12, 31))
    assert r.filed == date(2024, 2, 15)
    assert r.known_at == datetime(2024, 2, 15, 21, tzinfo=UTC)


def test_companyfacts_prefers_tagged_gross_profit_when_present():
    facts = {"facts": {"us-gaap": {"GrossProfit": {"units": {"USD": [
        _fact("2023-12-31", 1700, "2024-02-15", start="2023-01-01", fy=2023, fp="FY",
              form="10-K", accn="a23"),
    ]}}}}}
    gp = [r for r in sec._parse_companyfacts(facts, "AAPL") if r.concept == "gp_ttm"]
    assert gp and gp[0].val == 1700.0 and gp[0].tag == "us-gaap:GrossProfit"


def test_companyfacts_equity_follows_the_best_tag_each_filing_reports():
    """A filer that drops ``StockholdersEquity`` must not freeze the equity series.

    Resolving the fallback once for the whole history locked AVGO and QCOM onto a tag
    they had stopped filing. Per filing, the best tag that filing reports wins.
    """
    se = "StockholdersEquity"
    nci = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    facts = {"facts": {"us-gaap": {
        se: {"units": {"USD": [_fact("2022-12-31", 5000, "2023-02-15", accn="a22")]}},
        nci: {"units": {"USD": [
            _fact("2021-12-31", 4000, "2022-02-15", accn="a21"),
            _fact("2022-12-31", 5100, "2023-02-15", accn="a22"),
            _fact("2023-12-31", 5600, "2024-02-15", accn="a23"),
        ]}},
    }}}
    equity = [r for r in sec._parse_companyfacts(facts, "AVGO") if r.concept == "equity"]
    assert [(r.period_end, r.val, r.tag) for r in equity] == [
        (date(2021, 12, 31), 4000.0, f"us-gaap:{nci}"),
        (date(2022, 12, 31), 5000.0, f"us-gaap:{se}"),     # the filing's preferred tag
        (date(2023, 12, 31), 5600.0, f"us-gaap:{nci}"),    # the series keeps moving
    ]


def test_companyfacts_shares_fall_back_to_weighted_average_basic():
    """A filing with no share count contributes its latest, shortest weighted average.

    Multi-class issuers stopped filing a non-dimensional cover-page count, which froze
    MA and CMCSA at 2010 and left META with no share count at all.
    """
    waso = "WeightedAverageNumberOfSharesOutstandingBasic"
    facts = {"facts": {
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            _fact("2023-02-01", 100, "2023-02-15", accn="a22")]}}},
        "us-gaap": {waso: {"units": {"shares": [
            _fact("2022-12-31", 99, "2023-02-15", start="2022-01-01", accn="a22"),
            _fact("2023-09-30", 104, "2023-11-01", start="2023-07-01", accn="q323"),  # quarter
            _fact("2023-09-30", 103, "2023-11-01", start="2023-01-01", accn="q323"),  # 9 months
            _fact("2022-09-30", 98, "2023-11-01", start="2022-07-01", accn="q323"),   # comparative
            _fact("2023-12-31", 106, "2024-02-15", start="2023-01-01", accn="a23"),
        ]}}},
    }}
    shares = [r for r in sec._parse_companyfacts(facts, "MA") if r.concept == "shares"]
    assert [(r.period_end, r.val) for r in shares] == [
        (date(2023, 2, 1), 100.0), (date(2023, 9, 30), 104.0), (date(2023, 12, 31), 106.0)]
    assert [r.tag for r in shares] == ["dei:EntityCommonStockSharesOutstanding",
                                      f"us-gaap:{waso}", f"us-gaap:{waso}"]
    assert all(r.period_start is None for r in shares)


def test_companyfacts_shares_prefer_a_point_in_time_count_over_the_average():
    """A filing that reports a share count never contributes its weighted average instead."""
    facts = {"facts": {"us-gaap": {
        "CommonStockSharesOutstanding": {"units": {"shares": [
            _fact("2023-12-31", 110, "2024-02-15", accn="a23")]}},
        "WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": [
            _fact("2023-12-31", 107, "2024-02-15", start="2023-10-01", accn="a23")]}},
    }}}
    shares = [r for r in sec._parse_companyfacts(facts, "AAPL") if r.concept == "shares"]
    assert [(r.period_end, r.val, r.tag) for r in shares] == [
        (date(2023, 12, 31), 110.0, "us-gaap:CommonStockSharesOutstanding")]


def test_companyfacts_net_income_continues_on_profitloss():
    """Net income keeps updating after a filer stops tagging ``NetIncomeLoss``."""
    p = _PERIODS
    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [
            _fact("2022-09-30", 700, **p["q322"]), _fact("2022-12-31", 1000, **p["fy22"])]}},
        "ProfitLoss": {"units": {"USD": [
            _fact("2022-09-30", 710, **p["q322"]), _fact("2022-12-31", 1010, **p["fy22"]),
            _fact("2023-09-30", 910, **p["q323"]), _fact("2023-12-31", 1210, **p["fy23"])]}},
    }}}
    ni = [r for r in sec._parse_companyfacts(facts, "PH") if r.concept == "ni_ttm"]
    assert [(r.period_end, r.val, r.tag) for r in ni] == [
        (date(2022, 12, 31), 1000.0, "us-gaap:NetIncomeLoss"),  # same filing: priority decides
        (date(2023, 9, 30), 1210.0, "us-gaap:ProfitLoss"),      # FY22 1010 + 910 - 710
        (date(2023, 12, 31), 1210.0, "us-gaap:ProfitLoss"),
    ]


def test_companyfacts_gross_profit_is_synthesized_after_the_tag_stops():
    """Revenue minus COGS takes over the periods a filer no longer tags ``GrossProfit``.

    Synthesis used to run only when the tag had NEVER existed, so AMZN's gross profit sat
    on its 2009 figure.
    """
    p = _PERIODS
    facts = {"facts": {"us-gaap": {
        "GrossProfit": {"units": {"USD": [_fact("2022-12-31", 1200, **p["fy22"])]}},
        "Revenues": {"units": {"USD": [
            _fact("2022-09-30", 2100, **p["q322"]), _fact("2022-12-31", 3000, **p["fy22"]),
            _fact("2023-09-30", 2700, **p["q323"]), _fact("2023-12-31", 3600, **p["fy23"])]}},
        "CostOfRevenue": {"units": {"USD": [
            _fact("2022-09-30", 1300, **p["q322"]), _fact("2022-12-31", 1800, **p["fy22"]),
            _fact("2023-09-30", 1500, **p["q323"]), _fact("2023-12-31", 2000, **p["fy23"])]}},
    }}}
    gp = [r for r in sec._parse_companyfacts(facts, "AMZN") if r.concept == "gp_ttm"]
    assert [(r.period_end, r.val, r.tag) for r in gp] == [
        (date(2022, 12, 31), 1200.0, "us-gaap:GrossProfit"),
        (date(2023, 9, 30), 1600.0, "computed:GrossProfit"),    # 1200 + 1200 - 800
        (date(2023, 12, 31), 1600.0, "computed:GrossProfit"),
    ]


def test_a_proxy_statement_never_displaces_the_10k_income():
    """A period comes from the filing that disclosed it FIRST, whatever the tag order.

    DEF 14A pay-versus-performance tables tag ``NetIncomeLoss``; merging by tag priority
    took MA's FY2025 net income from the proxy filed ~75 days after the 10-K.
    """
    fy22 = _PERIODS["fy22"]
    proxy = dict(fy22, filed="2023-04-27", accn="p23", form="DEF 14A")
    facts = {"facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [_fact("2022-12-31", 1000, **proxy)]}},
        "ProfitLoss": {"units": {"USD": [_fact("2022-12-31", 1000, **fy22)]}},
    }}}
    ni = [r for r in sec._parse_companyfacts(facts, "MA") if r.concept == "ni_ttm"]
    assert [(r.period_end, r.accn) for r in ni] == [(date(2022, 12, 31), "a22")]


def test_synthesized_gross_profit_never_borrows_a_later_filings_cogs():
    """Revenue and COGS must come from the SAME filing; matching on the period leaked.

    The FY23 10-K re-presents FY22. Matching on the period alone paired the FY22 10-K's
    revenue with that later COGS and stamped the result with the earlier filing date.
    """
    fy22, fy23 = _PERIODS["fy22"], _PERIODS["fy23"]
    fy22_again = dict(fy23, start="2022-01-01")         # FY22 as re-presented in the FY23 10-K
    facts = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _fact("2022-12-31", 3000, **fy22), _fact("2022-12-31", 2600, **fy22_again),
            _fact("2023-12-31", 3600, **fy23)]}},
        "CostOfRevenue": {"units": {"USD": [
            _fact("2022-12-31", 1800, **fy22), _fact("2022-12-31", 1700, **fy22_again),
            _fact("2023-12-31", 2000, **fy23)]}},
    }}}
    rows = sec._parse_companyfacts(facts, "PH")
    gp = {r.period_end: r.val for r in rows if r.concept == "gp_ttm"}
    assert gp[date(2022, 12, 31)] == 1200.0            # 3000 - 1800, never 3000 - 1700
    assert gp[date(2023, 12, 31)] == 1600.0
