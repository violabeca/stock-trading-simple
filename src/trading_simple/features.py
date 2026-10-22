"""Alternative-data features: shared-DB event tables -> panel-aligned matrices.

This is the one engine extension that lets a strategy see beyond OHLCV. It reads
the point-in-time alt-data tables (``insider_trades``, ``analyst_ratings``,
``short_interest`` — ingested by this repo's own providers) and turns each into a
wide ``(panel-index x symbol)`` daily matrix that rides on :attr:`Panel.features`,
so ``signals(panel, **params)`` can consume it with no further plumbing. Adding a
future alt-data strategy needs only a new builder here and a ``data.features:``
line in its config — never a change to the strategy contract.

**Division of labour.** Builders do only DB -> daily aggregate (e.g. net insider
dollars per day). ALL modelling — rolling windows, EMAs, cross-sectional ranks,
weights — stays in the strategy's ``signals()``, where it is tunable.

**Point-in-time discipline.** Every builder buckets events by their ``known_at``
(the public-availability timestamp) in UTC, and :func:`_to_grid` then applies a
one-bar ``shift`` so a feature at bar ``D`` reflects only events knowable
*strictly before* ``D``. Combined with the engine's next-bar-open execution
(``bt.next_bar`` in the strategy), an event known during day ``D`` moves a
position no earlier than the open of ``D+2`` — deliberately conservative, and
immaterial at these signals' weeks-to-months horizons.

**Fundamentals are screened, never repaired.** SEC XBRL carries filers' own
mistakes — thousand-fold unit slips, zero and 100-share stub counts — and the table
stores them as filed, like any point-in-time archive. The feature layer is where they
stop: :func:`_screen_shares` drops share counts that cannot be real, :func:`_newest_period`
picks one row per day in a total order and never lets a series move backwards, and
:func:`_to_grid` masks a level no filing has
refreshed in :data:`_FUND_MAX_AGE_DAYS`. Every rule is causal, so a train-only load and
a full load agree bar for bar — a rule that looked at the NEXT filing would break that.

**Insider trades are counted once.** The same trade can arrive by two routes (SEC's bulk
data sets and the filing itself) with differently rounded prices, so the table keeps both;
the insider query keeps one copy per trade (:data:`_INSIDER_SQL`).

**Short interest expires too.** FINRA publishes twice a month, so a days-to-cover level
older than :data:`_SI_MAX_AGE_DAYS` means the table stopped being refreshed for that name,
not that its short interest stood still — it is masked, never carried.

**Option chains are local files.** The ``opt_*`` features (:data:`_OPTION_FEATURES`) come
from a daily summary of MarketData's end-of-day chains on disk
(``providers.marketdata``: implied volatilities solved locally), not from the DB; they are
levels like short interest, served through the same as-of lag and age mask.

Daily bars only: the alignment keys on the UTC calendar date, which for
UTC-midnight daily bars equals the trading session's date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

# One row per (symbol, UTC event-date): net signed open-market insider dollars, each trade
# counted ONCE. SEC's quarterly bulk data sets round prices to cents (5.06) where the filing's
# own XML keeps four decimals (5.0620), and ``insider_trades_uq`` includes the price, so a trade
# ingested by both routes is stored twice (68,270 of 757,444 rows on 2026-10-20). Identity-only
# and causal, like the other screens: of the copies known by ``as_of``, the later stamp wins.
_INSIDER_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d,
           sum(CASE WHEN txn_type = 'buy'  THEN shares * price
                    WHEN txn_type = 'sell' THEN -shares * price ELSE 0 END) AS value
    FROM (SELECT DISTINCT ON (accession, insider, traded_at, txn_type, shares)
                 symbol, known_at, txn_type, shares, price
          FROM insider_trades
          WHERE symbol = ANY(:symbols) AND known_at >= :start AND known_at <= :as_of
          ORDER BY accession, insider, traded_at, txn_type, shares, known_at DESC) AS t
    GROUP BY symbol, d
"""

# One row per (symbol, UTC event-date): net rating change (+1 upgrade, -1 downgrade).
_RATINGS_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d,
           sum(CASE WHEN action = 'upgrade'   THEN 1
                    WHEN action = 'downgrade' THEN -1 ELSE 0 END) AS value
    FROM analyst_ratings
    WHERE symbol = ANY(:symbols) AND known_at >= :start AND known_at <= :as_of
    GROUP BY symbol, d
"""

# One row per (symbol, UTC event-date): summed analyst price-target REVISION.
#
# The revision (new/prior - 1), never the target-to-price LEVEL. Brav & Lehavy (2003)
# locate the drift in the revision — notably on events where the *rating* did not
# change, which `_RATINGS_SQL` scores as zero — while Han & Kang (2022) show price
# drifts *against* the level, most strongly in large, liquid names. Ranking on the
# level would likely invert the sign in exactly this universe.
#
# `pt_prior > 0` drops first-ever targets (stored NULL, no prior to revise from).
_PRICE_TARGET_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d,
           sum(pt_current / pt_prior - 1.0) AS value
    FROM analyst_price_targets
    WHERE symbol = ANY(:symbols) AND known_at >= :start AND known_at <= :as_of
      AND pt_prior > 0 AND pt_current > 0
    GROUP BY symbol, d
"""

# One row per settlement: days-to-cover level (a stock), keyed by UTC dissemination date.
# No lower time bound: a level is seeded from the last observation BEFORE the panel
# start, so bar 0 is not a dead zone waiting for the first in-window dissemination. The
# settlement date rides along for :func:`_screen_short_interest`; two rows landing on one
# dissemination date are averaged by :func:`_to_grid`, exactly as a GROUP BY would.
_SHORT_INTEREST_SQL = """
    SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d, settlement_date,
           days_to_cover AS value
    FROM short_interest
    WHERE symbol = ANY(:symbols) AND known_at <= :as_of
"""

def _fundamental_sql(concept: str) -> str:
    """SQL for one fundamentals concept: every stored row knowable by ``:as_of``.

    The choice between rows is made in pandas by :func:`_newest_period`, not here: it
    needs a total order and a memory of earlier days that ``DISTINCT ON`` cannot express.
    Like all levels there is no lower time bound, so the panel's first bars are seeded
    from the last filing BEFORE the window instead of sitting NaN until the first
    in-window one lands. ``concept`` is a fixed code literal (never user input), so
    interpolating it is safe.

    Args:
        concept: The stored concept name (e.g. ``"equity"``, ``"ni_ttm"``).

    Returns:
        A parameterized SQL string returning ``(symbol, d, period_end, accn, form, value)``.
    """
    return f"""
        SELECT symbol, (known_at AT TIME ZONE 'UTC')::date AS d, period_end, accn, form,
               val AS value
        FROM fundamentals
        WHERE symbol = ANY(:symbols) AND concept = '{concept}' AND known_at <= :as_of
    """


#: Feature name -> (SQL, fill mode). ``zero`` = event flow (no event = 0);
#: ``ffill`` = a level that persists between observations (short interest, and the
#: fundamentals levels, which hold from one filing to the next).
_BUILDERS = {
    "insider": (_INSIDER_SQL, "zero"),
    "ratings": (_RATINGS_SQL, "zero"),
    "price_target": (_PRICE_TARGET_SQL, "zero"),
    "short_interest": (_SHORT_INTEREST_SQL, "ffill"),
    # Fundamentals (SEC XBRL companyfacts): balance-sheet levels + TTM income flows,
    # each forward-filled from its filing. Ratios vs price are formed in signals().
    "fund_equity": (_fundamental_sql("equity"), "ffill"),
    "fund_assets": (_fundamental_sql("assets"), "ffill"),
    "fund_shares": (_fundamental_sql("shares"), "ffill"),
    "fund_ni_ttm": (_fundamental_sql("ni_ttm"), "ffill"),
    "fund_gp_ttm": (_fundamental_sql("gp_ttm"), "ffill"),
}

#: The fundamentals levels: picked by :func:`_newest_period` and age-capped in
#: :func:`load_features`. Of the other features only short interest is age-capped
#: (:data:`_SI_MAX_AGE_DAYS`); the event flows are read exactly as before.
_FUNDAMENTALS = frozenset(name for name in _BUILDERS if name.startswith("fund_"))

#: A share count outside this band is a filer error, not a company. SEC stores
#: thousand-fold unit slips (AJG filed 1.9e14 shares for two 2020 quarters), zeros and
#: 100-share shell stubs as filed. The smallest real count in the data is ~1.2e6 (SEB) and
#: the largest 1.9e11 (BSAC, a Chilean bank's ordinary shares); every universe error since
#: 2018 falls outside it. Stateless on purpose: repairing each filing against the previous
#: one manufactures errors for up to a year wherever a company's FIRST filing is the bad one.
_SHARES_BAND = (1e6, 2e11)

#: Share counts furnished on this form are ignored. Foreign private issuers FURNISH interim
#: results on Form 6-K (unaudited, not filed), and those XBRL share counts are the unreliable
#: ones: RY's 2019 6-Ks tagged its weighted-average count in thousands (1,437,074 for 1.44
#: billion shares — inside the band), while ASM, CCEC and DHT tagged theirs a thousand-fold
#: too high. Ignoring them leaves no thousand-fold step in any universe name since 2018; the
#: annual 20-F / 40-F count carries until the next report, well inside the max age.
_SHARES_SKIP_FORM = "6-K"

#: A fundamentals level older than this many days at the as-of date is masked, not carried.
#: Normal filing gaps are far shorter (99th percentile: 134 days for domestic filers, 729
#: for annual foreign filers); what it removes is a series no filing refreshes any more —
#: V's share count from 2010, which no SEC tag can replace.
_FUND_MAX_AGE_DAYS = 800

#: A short-interest level older than this many days at the as-of date is masked, not carried.
#: No two FINRA dissemination dates in 2018-2026 were more than 19 days apart (99.9% of
#: per-name gaps are <= 19 days), so 45 tolerates two missed files. What it removes is a
#: table that stopped being refreshed: from the 2026-07-15 settlement on, 395 of Liquid500's
#: names went un-ingested and ranked on their 2026-06-30 days-to-cover with no error anywhere.
#: It also stops a reused ticker carrying the previous company's value across the gap (FINRA
#: reports by ticker: Weight Watchers' days-to-cover sat on WTW for 1,005 days).
_SI_MAX_AGE_DAYS = 45

#: FINRA files short interest by TICKER, and a ticker that changed hands carries the previous
#: issuer's rows under today's name: symbol -> (first, last) settlement date of rows that belong
#: to ANOTHER issuer (None = open-ended). Each case was confirmed 2026-09-23 against the DB: the
#: ratio of FINRA's average daily volume to the stock's own volume is steady (~1.0; a split steps
#: it once) for the right issuer and JUMPS between settlements for another one — lower or higher,
#: since the old issuer can be the bigger company. BNY was a municipal fund at FINRA's 1.00
#: days-to-cover floor until 2026-05 (BNY Mellon's own rows were never filed under it), which
#: ranked a megacap bank at the top of its group on another issuer's data for eight years; B was
#: Barnes Group before Barrick; DOC Physicians Realty; COR CoreSite; GEN Genesis Healthcare; META
#: an ETF before Facebook took the ticker; WTW Weight Watchers; COHR Coherent before II-VI; CZR
#: the old Caesars Entertainment before Eldorado took the name; IR Ingersoll-Rand plc (today's
#: TT) before Gardner Denver took it; HHH another security before Howard Hughes Holdings.
#: Identity is hindsight about WHICH COMPANY a row describes, never about returns. A broader
#: universe needs its own audit on that jump signature.
_SI_OTHER_ISSUER = {
    "BNY": (None, "2026-05-28"), "B": (None, "2025-05-14"), "DOC": (None, "2024-02-29"),
    "COR": (None, "2023-08-30"), "GEN": (None, "2022-11-14"), "META": (None, "2022-05-31"),
    "WTW": (None, "2022-01-13"), "COHR": (None, "2022-09-14"), "CZR": (None, "2020-07-15"),
    "IR": (None, "2020-02-28"), "HHH": (None, "2022-06-15"),
}

#: The same company's OWN rows, filed under the ticker it used before: symbol -> [(predecessor,
#: first, last settlement date)]. The price table keeps a renamed company's history under today's
#: ticker (Barrick's volume sits under B back to 2018), so these rows — ingested under their own
#: ticker, stored verbatim — are what describe that history; :data:`_SI_OTHER_ISSUER` drops the
#: other company's rows in the same windows. Stateless and identity-only, like that map.
_SI_PREDECESSOR = {
    "BNY": [("BK", None, "2026-05-28")], "META": [("FB", None, "2022-05-31")],
    "B": [("GOLD", "2019-01-15", "2025-05-14")],   # GOLD was Randgold until Barrick took it
    "DOC": [("PEAK", None, "2024-02-29")],
    "COR": [("ABC", None, "2023-08-30")],
    "GEN": [("SYMC", None, "2019-11-04"), ("NLOK", "2019-11-05", "2022-11-14")],
    "WTW": [("WLTW", None, "2022-01-13")], "COHR": [("IIVI", None, "2022-09-14")],
    "FISV": [("FI", "2023-06-01", "2025-11-13")], "CZR": [("ERI", None, "2020-07-15")],
    "IR": [("GDI", None, "2020-02-28")],
}

#: Option-chain features: name -> column of the daily summary table that
#: ``providers.marketdata.build_cache`` builds from the local MarketData chains (not the DB).
#: Levels quoted at a session's 16:00 close, so the as-of lag serves them from the next bar.
_OPTION_FEATURES = {"opt_atm_iv": "atm_iv", "opt_skew": "skew", "opt_put_skew": "put_skew",
                    "opt_cpiv": "cpiv", "opt_call_volume": "call_volume",
                    "opt_put_volume": "put_volume", "opt_call_oi": "call_oi",
                    "opt_put_oi": "put_oi"}
#: An option summary older than this is a chain that stopped arriving, not a quiet market.
_OPT_MAX_AGE_DAYS = 5

#: Feature name -> max age in days of a carried level. A feature not listed is never masked
#: (the event flows are not carried at all).
_MAX_AGE_DAYS = {"short_interest": _SI_MAX_AGE_DAYS,
                 **dict.fromkeys(_FUNDAMENTALS, _FUND_MAX_AGE_DAYS),
                 **dict.fromkeys(_OPTION_FEATURES, _OPT_MAX_AGE_DAYS)}


def load_features(
    engine: Engine, index: pd.DatetimeIndex, symbols: list[str], names: list[str],
) -> dict[str, pd.DataFrame]:
    """Build point-in-time feature matrices aligned to a panel's index.

    Args:
        engine: Shared-database engine.
        index: The panel's (tz-aware UTC) daily timestamp index.
        symbols: Panel symbols; every returned frame has exactly these columns.
        names: Feature names to build (subset of :data:`_BUILDERS` and
            :data:`_OPTION_FEATURES`).

    Returns:
        ``{name: frame}`` where each frame is ``(index x symbols)``, already
        PIT-lagged (see module docstring). Unknown names raise ``KeyError``.

    Raises:
        KeyError: If a requested name has no builder.
        FileNotFoundError: If an ``opt_*`` feature is asked for before the option summary
            cache exists (``providers.marketdata.build_cache``).
    """
    start = pd.Timestamp(index[0]).to_pydatetime()
    as_of = pd.Timestamp(index[-1]).to_pydatetime()
    out: dict[str, pd.DataFrame] = {}
    for name in [n for n in names if n in _OPTION_FEATURES]:
        out[name] = _option_grid(name, index, symbols)
    names = [n for n in names if n not in _OPTION_FEATURES]
    if not names:
        return out
    with engine.connect() as conn:                # one connection serves every builder
        for name in names:
            sql, fill = _BUILDERS[name]
            # Flows are window-bounded; levels carry no :start bind (they fetch the
            # full pre-window history so the first bars are seeded — see builder SQL).
            params = {"symbols": symbols, "as_of": as_of}
            if name == "short_interest":            # + each name's own rows under an old ticker
                params["symbols"] = symbols + sorted(
                    {old for sym in symbols for old, *_ in _SI_PREDECESSOR.get(sym, [])})
            if fill == "zero":
                params["start"] = start
            long = pd.read_sql(text(sql), conn, params=params)
            fundamental = name in _FUNDAMENTALS
            if fundamental:
                long = _newest_period(_screen_shares(long) if name == "fund_shares" else long)
            elif name == "short_interest":
                long = _screen_short_interest(long)
            grid = _to_grid(long, index, fill=fill, max_age_days=_MAX_AGE_DAYS.get(name))
            out[name] = grid.reindex(columns=symbols)
    return out


def _option_grid(name: str, index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """One ``opt_*`` feature on the panel grid, from the local option summary table.

    A chain quoted at session ``d``'s close is a level observed on ``d``: the same as-of join
    as every level (:func:`_to_grid`) serves it from bar ``d + 1``, and masks it once it is
    more than :data:`_OPT_MAX_AGE_DAYS` old.
    """
    from trading_simple.providers.marketdata import load_summaries

    table = load_summaries()
    as_of = pd.Timestamp(index[-1])
    as_of = as_of.tz_convert("UTC").tz_localize(None) if as_of.tz is not None else as_of
    rows = table[table["symbol"].isin(symbols) & (table["d"] <= as_of.normalize())]
    long = rows[["symbol", "d"]].assign(value=rows[_OPTION_FEATURES[name]]).dropna()
    grid = _to_grid(long, index, fill="ffill", max_age_days=_OPT_MAX_AGE_DAYS)
    return grid.reindex(columns=symbols)


def attach_features(panel, engine: Engine, names: list[str], *, settings=None) -> None:
    """Populate ``panel.features`` in place with the requested alt-data matrices.

    Args:
        panel: A :class:`~trading_simple.data.Panel` (daily bars).
        engine: Shared-database engine.
        names: Feature names to build (e.g. ``["insider", "ratings"]``).
        settings: Accepted for call-site symmetry; unused (kept so callers can
            pass it uniformly alongside other loaders).
    """
    panel.features = load_features(engine, panel.close.index, panel.symbols, names)


def _screen_shares(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop share-count rows that cannot describe a listed company's float (pure).

    A count outside :data:`_SHARES_BAND` is a filer error — a thousand-fold unit slip, a
    zero, a 100-share shell stub — and a count furnished on :data:`_SHARES_SKIP_FORM` is not
    trusted at all. Runs before :func:`_newest_period` picks a day's row, so an invalid row
    never hides a valid one filed the same day.

    Args:
        rows: The ``fund_shares`` long frame from :func:`_fundamental_sql`.

    Returns:
        The rows that can be a real, reliably reported share count.
    """
    plausible = rows["value"].between(*_SHARES_BAND)
    furnished = rows["form"].fillna("").str.startswith(_SHARES_SKIP_FORM)
    return rows[plausible & ~furnished]


def _screen_short_interest(rows: pd.DataFrame) -> pd.DataFrame:
    """Drop short-interest rows filed under a ticker that then belonged to another issuer (pure).

    Args:
        rows: ``_SHORT_INTEREST_SQL`` output: ``symbol``, ``d``, ``settlement_date``, ``value``.

    Returns:
        ``(symbol, d, value)`` without the :data:`_SI_OTHER_ISSUER` windows, and with each
        :data:`_SI_PREDECESSOR` row relabelled to today's ticker inside its window (rows of an
        old ticker outside it keep their own symbol and fall away with the panel's columns).
    """
    settle = pd.to_datetime(rows["settlement_date"])

    def within(symbol, first, last):
        hit = rows["symbol"] == symbol
        if first is not None:
            hit &= settle >= pd.Timestamp(first)
        if last is not None:
            hit &= settle <= pd.Timestamp(last)
        return hit

    wrong = pd.Series(False, index=rows.index)
    for symbol, (first, last) in _SI_OTHER_ISSUER.items():
        wrong |= within(symbol, first, last)
    out = rows.loc[~wrong, ["symbol", "d", "value"]].copy()
    for symbol, spans in _SI_PREDECESSOR.items():   # the company's own rows, old ticker -> today's
        for old, first, last in spans:
            out.loc[within(old, first, last).loc[out.index], "symbol"] = symbol
    return out


def _newest_period(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, day) the newest reported period, kept only if it moves the series forward.

    Three rules, all causal — a day's outcome depends only on that day and the days before
    it, so a train-only load and a full load agree on every shared bar:

    1. **Screen first.** Missing values are dropped here, and implausible ones by the caller
       (:func:`_screen_shares`) BEFORE this runs — so an invalid row (a zero share count)
       can never hide a valid one filed the same day.
    2. **Newest period, total order.** Among a day's rows take the latest ``period_end``,
       then the highest accession. Every fundamentals ``known_at`` is 21:00 UTC, so
       without the accession a tie had no defined winner.
    3. **Never regress.** Keep the day only if its period is later than every earlier
       day's. The SEC parser emits monotone series, but the table is insert-or-ignore:
       rows from an older parse stay beside a newer parse's, and a late proxy statement or
       10-K/A re-reporting an old period must not pull the level backwards.

    Args:
        rows: Long frame with ``symbol``, ``d`` (the UTC ``known_at`` date),
            ``period_end``, ``accn`` and ``value`` — the output of :func:`_fundamental_sql`.

    Returns:
        A ``(symbol, d, value)`` frame with at most one row per (symbol, day), ready for
        :func:`_to_grid`.
    """
    rows = rows[rows["value"].notna()]
    rows = rows.assign(pe=pd.to_datetime(rows["period_end"]), accn=rows["accn"].fillna(""))
    rows = rows.sort_values(["symbol", "d", "pe", "accn"], kind="stable")
    day = rows.groupby(["symbol", "d"], sort=False).tail(1)
    earlier = day.groupby("symbol")["pe"].cummax().groupby(day["symbol"]).shift()
    return day.loc[earlier.isna() | (day["pe"] > earlier), ["symbol", "d", "value"]]


def _to_grid(long: pd.DataFrame, index: pd.DatetimeIndex, *, fill: str = "zero",
             max_age_days: int | None = None) -> pd.DataFrame:
    """Pivot a ``(symbol, d, value)`` long frame onto the panel index (pure).

    Reindexes daily event aggregates onto the panel's calendar (matching on UTC
    date) and lags them one day, so bar ``D`` sees only events knowable strictly
    before ``D`` — the conservative point-in-time lag. ``"zero"`` flows live on
    panel days only (no event = 0, lag = previous panel bar). ``"ffill"`` levels
    are as-of joins over ALL observation dates: a level observed before the panel
    start seeds bar 0 (no dead zone waiting for the first in-window filing), and
    one observed on a non-panel day (weekend dissemination) is carried onto the
    next bar instead of being dropped.

    Args:
        long: Columns ``symbol``, ``d`` (a date), ``value``; one row per pair.
        index: Target panel index (tz-aware UTC or naive).
        fill: ``"zero"`` (flows) or ``"ffill"`` (levels).
        max_age_days: ``"ffill"`` only — mask a carried level observed more than this many
            days before the as-of date (D-1). ``None`` carries levels indefinitely, which
            is the exact pre-existing behaviour.

    Returns:
        A ``(index x symbol)`` frame; empty (index-only) when ``long`` is empty.
    """
    if long.empty:
        return pd.DataFrame(index=index, dtype=float)
    aggfunc = "mean" if fill == "ffill" else "sum"
    wide = long.pivot_table(index="d", columns="symbol", values="value", aggfunc=aggfunc)
    wide.index = pd.to_datetime(wide.index)  # date -> naive midnight Timestamp

    keys = pd.DatetimeIndex(index)
    keys = keys.tz_convert("UTC") if keys.tz is not None else keys
    keys = keys.normalize().tz_localize(None) if keys.tz is not None else keys.normalize()

    if fill == "ffill":
        # As-of at D-1: the last observation on any date <= D-1, i.e. strictly
        # before D — the same lag the flows' shift(1) expresses, but seedable from
        # observations before keys[0] and not confined to panel days.
        asof = keys - pd.Timedelta(days=1)
        filled = wide.reindex(wide.index.union(keys)).ffill()
        carried = filled.reindex(asof, method="ffill")
        if max_age_days is not None:
            # The day each carried value was observed, pushed through the SAME as-of
            # join, so the age always belongs to the value actually carried.
            seen = pd.DataFrame(np.where(wide.notna(), _day_number(wide.index)[:, None], np.nan),
                                index=wide.index, columns=wide.columns)
            seen = seen.reindex(seen.index.union(keys)).ffill().reindex(asof, method="ffill")
            carried = carried.where(_day_number(asof)[:, None] - seen.to_numpy() <= max_age_days)
        wide = carried
    else:
        wide = wide.reindex(keys).fillna(0.0)
        wide = wide.shift(1)      # PIT: bar D sees only events strictly before D
    wide.index = index            # restore the panel's real (tz-aware) index
    return wide


def _day_number(index: pd.DatetimeIndex) -> np.ndarray:
    """Whole days since the epoch for a naive daily index, as floats (pure).

    Args:
        index: A tz-naive ``DatetimeIndex`` at day resolution.

    Returns:
        A float array, so it can sit beside NaN in the max-age arithmetic.
    """
    return np.asarray((index - pd.Timestamp(0)).days, dtype=float)
