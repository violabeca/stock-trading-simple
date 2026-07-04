-- ===========================================================================
-- Schema for trading-simple. IDENTICAL (column for column) to the securities +
-- prices_ohlcv tables of the sibling `stock-trading` repo, because both repos
-- share one TimescaleDB. Running this against the live shared database is a
-- no-op: every statement is IF NOT EXISTS / idempotent.
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
