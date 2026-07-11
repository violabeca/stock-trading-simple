-- ===========================================================================
-- Schema for trading-simple. Column-for-column IDENTICAL to the tables of the
-- sibling `stock-trading` repo (securities, prices_ohlcv, and the alt-data
-- tables below), because both repos share one TimescaleDB. Running this against
-- the live shared database is a no-op: every statement is IF NOT EXISTS /
-- idempotent, and we only ever ADD tables here — never DROP or ALTER an existing
-- one. That keeps this repo standalone (it can populate its own alt-data on a
-- fresh database) while staying compatible with whatever the sibling created.
--
-- POINT-IN-TIME (PIT) DISCIPLINE for the alt-data tables: every event row stores
-- BOTH when the event happened and when it first became public (`known_at`).
-- Feature code only ever reads rows with `known_at <= as_of` — the single most
-- important defense against look-ahead bias.
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Master list of tradable instruments. Delisted names are retained
-- (is_active=false) for survivorship-bias-free backtesting.
CREATE TABLE IF NOT EXISTS securities (
    symbol        TEXT PRIMARY KEY,
    name          TEXT,
    security_type TEXT NOT NULL DEFAULT 'equity',  -- equity | etf | metal | index
    exchange      TEXT,
    currency      TEXT NOT NULL DEFAULT 'USD',
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    listed_at     DATE,
    delisted_at   DATE,
    sector        TEXT,
    industry      TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Candle data at any bar size ('1m', '1d', ...). High-volume, so it is a
-- TimescaleDB hypertable partitioned by time and symbol.
CREATE TABLE IF NOT EXISTS prices_ohlcv (
    symbol     TEXT        NOT NULL REFERENCES securities(symbol),
    bar        TEXT        NOT NULL,        -- '1m', '5m', '1d', ...
    ts         TIMESTAMPTZ NOT NULL,        -- bar OPEN time, in UTC
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION,
    adj_close  DOUBLE PRECISION,            -- split/dividend-adjusted close
    volume     DOUBLE PRECISION,
    source     TEXT,                        -- provider that produced this row
    rth        BOOLEAN NOT NULL DEFAULT TRUE, -- TRUE = regular session (or daily bar);
                                              -- FALSE = pre/post-market bar
    PRIMARY KEY (symbol, bar, ts)
);

SELECT create_hypertable(
    'prices_ohlcv', 'ts',
    partitioning_column => 'symbol',
    number_partitions   => 8,
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);


-- ===========================================================================
-- ALTERNATIVE DATA (non-OHLCV signals consumed by alt-data strategies via
-- trading_simple.features). Ingested by this repo's own providers (SEC EDGAR,
-- Yahoo Finance, FINRA) — no dependency on the sibling repo — but with columns
-- identical to the sibling's so the shared database stays consistent whichever
-- repo populated them. Each carries `known_at` (public-availability timestamp)
-- for point-in-time reads, a NULLS-NOT-DISTINCT natural key for idempotent
-- UPSERT, and a (symbol, known_at) index for the as-of feature queries.
-- ===========================================================================

-- SEC Form 4 corporate-insider transactions. `traded_at` is the transaction
-- date; `known_at` is the SEC filing-ACCEPTANCE timestamp — most of the signal's
-- value precedes disclosure, so features may only act on `known_at`.
CREATE TABLE IF NOT EXISTS insider_trades (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT NOT NULL REFERENCES securities(symbol),
    insider     TEXT,                         -- person/entity name
    role        TEXT,                         -- 'CEO', 'CFO', 'Director', ...
    txn_type    TEXT,                         -- 'buy' | 'sell'
    shares      DOUBLE PRECISION,
    price       DOUBLE PRECISION,
    traded_at   DATE NOT NULL,
    filed_at    TIMESTAMPTZ NOT NULL,
    known_at    TIMESTAMPTZ NOT NULL,
    accession   TEXT,                         -- SEC filing accession number
    source      TEXT,
    -- price is part of the key so split-fill lots (one order filled at several
    -- prices, reported as separate rows with identical share counts) stay
    -- distinct. NULLS NOT DISTINCT keeps re-runs idempotent for priceless rows.
    CONSTRAINT insider_trades_uq
        UNIQUE NULLS NOT DISTINCT (accession, insider, traded_at, txn_type, shares, price)
);

-- Sell-side analyst rating / grade changes (upgrades, downgrades) — the raw
-- material for rating-change momentum. `known_at` is the announcement timestamp.
CREATE TABLE IF NOT EXISTS analyst_ratings (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT NOT NULL REFERENCES securities(symbol),
    firm        TEXT,
    action      TEXT,                          -- initiate | upgrade | downgrade | maintain
    grade_from  TEXT,
    grade_to    TEXT,
    rated_at    DATE NOT NULL,
    known_at    TIMESTAMPTZ NOT NULL,
    source      TEXT,
    -- NULLS NOT DISTINCT so rows with a null firm or grade_to still dedupe.
    CONSTRAINT analyst_ratings_uq
        UNIQUE NULLS NOT DISTINCT (symbol, firm, rated_at, grade_to)
);

-- FINRA consolidated bi-monthly equity short interest. `settlement_date` is the
-- position-measurement date; `known_at` is FINRA's dissemination timestamp
-- (~8 business days later, after 16:00 ET) — the only time a trader could act.
-- `days_to_cover` = short_interest / avg_daily_volume (the classic ranking).
CREATE TABLE IF NOT EXISTS short_interest (
    symbol           TEXT NOT NULL REFERENCES securities(symbol),
    settlement_date  DATE NOT NULL,
    short_interest   DOUBLE PRECISION,          -- consolidated shares short
    avg_daily_volume DOUBLE PRECISION,          -- FINRA-reported average daily volume
    days_to_cover    DOUBLE PRECISION,          -- short_interest / avg_daily_volume
    disseminated_at  DATE,                       -- FINRA publication date
    known_at         TIMESTAMPTZ NOT NULL,
    source           TEXT,
    CONSTRAINT short_interest_uq UNIQUE NULLS NOT DISTINCT (symbol, settlement_date)
);

CREATE INDEX IF NOT EXISTS idx_insider_symbol_known ON insider_trades  (symbol, known_at);
CREATE INDEX IF NOT EXISTS idx_ratings_symbol_known ON analyst_ratings (symbol, known_at);
CREATE INDEX IF NOT EXISTS idx_short_int_sym_known  ON short_interest  (symbol, known_at);
