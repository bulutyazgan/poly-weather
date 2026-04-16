-- TradeBot PostgreSQL schema
-- Run: psql -d tradebot -f src/db/schema.sql

-- Forecasts (raw model output per station per run)
CREATE TABLE IF NOT EXISTS forecasts (
    id              BIGSERIAL PRIMARY KEY,
    station_id      VARCHAR(8)   NOT NULL,
    model_name      VARCHAR(16)  NOT NULL,  -- gfs, ecmwf, hrrr, mos
    run_time        TIMESTAMPTZ  NOT NULL,
    valid_date      DATE         NOT NULL,
    ensemble_mean   FLOAT,
    ensemble_std    FLOAT,
    mos_high        FLOAT,
    mos_low         FLOAT,
    hrrr_temp_f     FLOAT,
    hrrr_dewpoint_f FLOAT,
    hrrr_wind_kt    FLOAT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_forecasts_station_date
    ON forecasts (station_id, valid_date);

-- Observations (METAR / ASOS)
CREATE TABLE IF NOT EXISTS observations (
    id              BIGSERIAL PRIMARY KEY,
    station_id      VARCHAR(8)   NOT NULL,
    observed_time   TIMESTAMPTZ  NOT NULL,
    temp_f          FLOAT        NOT NULL,
    dewpoint_f      FLOAT,
    wind_speed_kt   FLOAT,
    wind_dir        FLOAT,
    pressure_mb     FLOAT,
    cloud_cover     VARCHAR(4),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_observations_station_time
    ON observations (station_id, observed_time);

-- Market prices (Polymarket CLOB snapshots)
CREATE TABLE IF NOT EXISTS market_prices (
    id              BIGSERIAL PRIMARY KEY,
    token_id        VARCHAR(128) NOT NULL,
    ts              TIMESTAMPTZ  NOT NULL,
    bid             FLOAT        NOT NULL,
    ask             FLOAT        NOT NULL,
    mid             FLOAT        NOT NULL,
    volume_24h      FLOAT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_market_prices_token
    ON market_prices (token_id, ts);

-- Trades (all signals — including SKIPs — full audit trail)
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        VARCHAR(128) UNIQUE,
    market_id       VARCHAR(128) NOT NULL,
    station_id      VARCHAR(8)   NOT NULL,
    direction       VARCHAR(8)   NOT NULL,  -- BUY_YES, BUY_NO
    action          VARCHAR(8)   NOT NULL,  -- TRADE, SKIP
    edge            FLOAT        NOT NULL,
    kelly_size      FLOAT        NOT NULL DEFAULT 0.0,
    entry_price     FLOAT,
    outcome         BOOLEAN,
    pnl             FLOAT,
    resolved        BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_trades_station
    ON trades (station_id, created_at);

-- Calibration (running verification stats per station/regime)
CREATE TABLE IF NOT EXISTS calibration (
    id              BIGSERIAL PRIMARY KEY,
    station_id      VARCHAR(8)   NOT NULL,
    regime          VARCHAR(16)  NOT NULL,  -- HIGH, MEDIUM, LOW
    brier_score     FLOAT,
    sample_count    INT          NOT NULL DEFAULT 0,
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (station_id, regime)
);

-- Regimes (classified regime per station per day)
CREATE TABLE IF NOT EXISTS regimes (
    id                  BIGSERIAL PRIMARY KEY,
    station_id          VARCHAR(8)   NOT NULL,
    valid_date          DATE         NOT NULL,
    regime              VARCHAR(32)  NOT NULL,
    confidence          VARCHAR(8)   NOT NULL,  -- HIGH, MEDIUM, LOW
    confidence_score    FLOAT        NOT NULL DEFAULT 0.5,
    spread_percentile   FLOAT,
    active_flags        TEXT[],
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (station_id, valid_date)
);
CREATE INDEX IF NOT EXISTS idx_regimes_station_date
    ON regimes (station_id, valid_date);
