-- ============================================================================
-- Data Service — Esquema de base de datos (PostgreSQL 16 + TimescaleDB)
-- Plataforma de Predicción de Volatilidad SP500 / NASDAQ
-- ============================================================================
-- Cubre los casos de uso:
--   1. Guardar datos crudos (row data) de SP500, NASDAQ, VIX, VXN.
--   2. Guardar datos procesados de SP500+VIX y NASDAQ+VXN.
--   3. Consultar la data para entrenar modelos de ML.
--   4. Registro de logs de cada corrida del pipeline.
--   5. Registro de versiones (de esquema de features y de código del pipeline).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 0. EXTENSIÓN
-- ----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ----------------------------------------------------------------------------
-- 1. TABLAS DE CATÁLOGO (referencia/lookup)
-- ----------------------------------------------------------------------------
-- Catálogo de instrumentos. Generaliza el diseño para no hardcodear
-- "SP500" y "VIX": agregar un nuevo índice en el futuro (ej. Dow Jones)
-- es una fila nueva aquí, no un cambio de esquema.
CREATE TABLE instruments (
    ticker        TEXT PRIMARY KEY,                 -- '^GSPC', '^VIX', '^IXIC', '^VXN'
    display_name  TEXT NOT NULL,
    asset_class   TEXT NOT NULL CHECK (asset_class IN ('index', 'volatility_index')),
    exchange      TEXT,
    currency      TEXT NOT NULL DEFAULT 'USD',
    is_active     BOOLEAN NOT NULL DEFAULT true,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE instruments IS 'Catálogo de tickers soportados por el Data Service.';

-- Pares índice + su volatilidad asociada. Aquí vive la relación de negocio
-- "SP500 se evalúa junto con VIX" / "NASDAQ se evalúa junto con VXN".
CREATE TABLE asset_pairs (
    pair_code          TEXT PRIMARY KEY,             -- 'SP500_VIX', 'NASDAQ_VXN'
    index_ticker       TEXT NOT NULL REFERENCES instruments(ticker),
    volatility_ticker  TEXT NOT NULL REFERENCES instruments(ticker),
    description        TEXT,
    is_active          BOOLEAN NOT NULL DEFAULT true,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (index_ticker, volatility_ticker)
);
COMMENT ON TABLE asset_pairs IS 'Pares índice + índice de volatilidad sobre los que se calculan features.';

-- Versiones del esquema de features (RNF12: invalidación automática de modelos
-- entrenados con un esquema anterior).
CREATE TABLE feature_schema_versions (
    version     TEXT PRIMARY KEY,                    -- 'v1', 'v2'...
    columns     JSONB NOT NULL,                       -- columnas y tipos vigentes en esta versión
    is_current  BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE feature_schema_versions IS 'Historial de versiones del esquema de features, para trazabilidad y compatibilidad de modelos.';

-- Solo puede haber una versión marcada como vigente a la vez.
CREATE UNIQUE INDEX idx_one_current_schema_version
    ON feature_schema_versions (is_current)
    WHERE is_current;

-- ----------------------------------------------------------------------------
-- 2. DATOS CRUDOS (row data) — Caso de uso 1
-- ----------------------------------------------------------------------------
CREATE TABLE raw_ohlc (
    ticker       TEXT NOT NULL REFERENCES instruments(ticker),
    date         DATE NOT NULL,
    open         DOUBLE PRECISION NOT NULL,
    high         DOUBLE PRECISION NOT NULL,
    low          DOUBLE PRECISION NOT NULL,
    close        DOUBLE PRECISION NOT NULL,
    volume       BIGINT,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, date),
    CHECK (high >= low),
    CHECK (open BETWEEN low AND high),
    CHECK (close BETWEEN low AND high),
    CHECK (high > 0 AND low > 0)
);
COMMENT ON TABLE raw_ohlc IS 'Velas diarias OHLC crudas, tal como las devuelve Yahoo Finance. Fuente de verdad inmutable.';

SELECT create_hypertable('raw_ohlc', 'date', chunk_time_interval => INTERVAL '1 year');

-- ----------------------------------------------------------------------------
-- 3. DATOS PROCESADOS (features) — Caso de uso 2
-- ----------------------------------------------------------------------------
CREATE TABLE features (
    pair_code               TEXT NOT NULL REFERENCES asset_pairs(pair_code),
    date                     DATE NOT NULL,
    schema_version           TEXT NOT NULL REFERENCES feature_schema_versions(version),

    -- variables del índice principal (SP500 o NASDAQ, según pair_code)
    main_log_return         DOUBLE PRECISION,
    main_log_range          DOUBLE PRECISION,
    main_body_log            DOUBLE PRECISION,
    main_upper_wick_log      DOUBLE PRECISION,
    main_lower_wick_log      DOUBLE PRECISION,
    main_vol_5d              DOUBLE PRECISION,
    main_vol_10d             DOUBLE PRECISION,

    -- variables del índice de volatilidad asociado (VIX o VXN)
    vol_idx_log_close        DOUBLE PRECISION,
    vol_idx_log_range        DOUBLE PRECISION,
    vol_idx_log_return       DOUBLE PRECISION,

    day_of_week              SMALLINT CHECK (day_of_week BETWEEN 0 AND 4),
    target_range_next_day    DOUBLE PRECISION,   -- se llena 1 día después (evita data leakage)

    computed_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (pair_code, date)
);
COMMENT ON TABLE features IS 'Variables derivadas por par índice+volatilidad, listas para entrenar o predecir.';
COMMENT ON COLUMN features.target_range_next_day IS 'Rango real observado al día siguiente. NULL hasta que se conoce (previene data leakage).';

SELECT create_hypertable('features', 'date', chunk_time_interval => INTERVAL '1 year');

-- ----------------------------------------------------------------------------
-- 4. LOGGING Y AUDITORÍA — Caso de uso 4 y 5 (versión de código del pipeline)
-- ----------------------------------------------------------------------------
CREATE TABLE ingestion_log (
    id                BIGSERIAL PRIMARY KEY,
    run_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    pipeline_stage    TEXT NOT NULL CHECK (pipeline_stage IN ('extraction', 'feature_engineering')),
    ticker            TEXT REFERENCES instruments(ticker),     -- usado en stage='extraction'
    pair_code         TEXT REFERENCES asset_pairs(pair_code),   -- usado en stage='feature_engineering'
    date_from         DATE,
    date_to           DATE,
    rows_affected     INT,
    status            TEXT NOT NULL CHECK (status IN ('success', 'partial', 'error')),
    error_message     TEXT,
    pipeline_version  TEXT,        -- commit/tag del código que ejecutó esta corrida
    duration_ms       INT,
    CHECK (
        (pipeline_stage = 'extraction' AND ticker IS NOT NULL)
        OR (pipeline_stage = 'feature_engineering' AND pair_code IS NOT NULL)
    )
);
COMMENT ON TABLE ingestion_log IS 'Auditoría de cada corrida del pipeline (extracción o feature engineering), con la versión de código que la ejecutó.';

CREATE INDEX idx_ingestion_log_ticker_run_at ON ingestion_log (ticker, run_at DESC);
CREATE INDEX idx_ingestion_log_pair_run_at ON ingestion_log (pair_code, run_at DESC);

-- ----------------------------------------------------------------------------
-- 5. TRIGGER DE AUDITORÍA (updated_at automático)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_raw_ohlc_updated_at
    BEFORE UPDATE ON raw_ohlc
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_features_updated_at
    BEFORE UPDATE ON features
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ----------------------------------------------------------------------------
-- 6. ÍNDICES ADICIONALES
-- ----------------------------------------------------------------------------
CREATE INDEX idx_features_schema_version ON features (schema_version);

-- ----------------------------------------------------------------------------
-- 7. COMPRESIÓN (TimescaleDB) — optimización de almacenamiento a largo plazo
-- ----------------------------------------------------------------------------
ALTER TABLE raw_ohlc SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker',
    timescaledb.compress_orderby = 'date'
);
SELECT add_compression_policy('raw_ohlc', INTERVAL '180 days');

ALTER TABLE features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'pair_code',
    timescaledb.compress_orderby = 'date'
);
SELECT add_compression_policy('features', INTERVAL '180 days');

-- ----------------------------------------------------------------------------
-- 8. VISTAS DE CONSUMO — Caso de uso 3 (consultar data para entrenar ML)
-- ----------------------------------------------------------------------------
-- El ML Service consulta estas vistas por HTTP (vía el Data Service), nunca
-- las tablas base directamente: aísla al consumidor de cambios internos de
-- esquema, siempre que la vista mantenga su contrato de columnas.

CREATE VIEW v_training_dataset AS
SELECT
    f.pair_code,
    ap.description AS pair_description,
    f.date,
    f.schema_version,
    f.main_log_return, f.main_log_range, f.main_body_log,
    f.main_upper_wick_log, f.main_lower_wick_log,
    f.main_vol_5d, f.main_vol_10d,
    f.vol_idx_log_close, f.vol_idx_log_range, f.vol_idx_log_return,
    f.day_of_week,
    f.target_range_next_day
FROM features f
JOIN asset_pairs ap ON ap.pair_code = f.pair_code
WHERE f.target_range_next_day IS NOT NULL;   -- solo filas con label conocido: listas para entrenar

COMMENT ON VIEW v_training_dataset IS 'Dataset de entrenamiento: features con su target ya conocido, listo para Walk-Forward Validation.';

CREATE VIEW v_latest_features AS
SELECT DISTINCT ON (pair_code) *
FROM features
ORDER BY pair_code, date DESC;

COMMENT ON VIEW v_latest_features IS 'Última fila de features por par, usada por el endpoint /features/latest (modo Predicción de Mañana).';

-- ----------------------------------------------------------------------------
-- 9. SEGURIDAD — rol de aplicación con privilegio mínimo (RNF06, RNF07)
-- ----------------------------------------------------------------------------
-- La contraseña se inyecta por variable de entorno en el momento del deploy,
-- nunca se guarda en este script ni en el repositorio.
-- CREATE ROLE data_service_app LOGIN PASSWORD :'data_service_app_password';
-- GRANT CONNECT ON DATABASE data_service_db TO data_service_app;
-- GRANT USAGE ON SCHEMA public TO data_service_app;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO data_service_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO data_service_app;
-- -- Sin DELETE ni privilegios DDL: la aplicación nunca borra filas ni altera el esquema.

-- ----------------------------------------------------------------------------
-- 10. SEED DE CATÁLOGO (datos de referencia iniciales)
-- ----------------------------------------------------------------------------
INSERT INTO instruments (ticker, display_name, asset_class, exchange) VALUES
    ('^GSPC', 'S&P 500', 'index', 'NYSE'),
    ('^VIX',  'CBOE Volatility Index', 'volatility_index', 'CBOE'),
    ('^IXIC', 'NASDAQ Composite', 'index', 'NASDAQ'),
    ('^VXN',  'CBOE NASDAQ-100 Volatility Index', 'volatility_index', 'CBOE');

INSERT INTO asset_pairs (pair_code, index_ticker, volatility_ticker, description) VALUES
    ('SP500_VIX',   '^GSPC', '^VIX', 'S&P 500 con VIX como índice de volatilidad asociado'),
    ('NASDAQ_VXN',  '^IXIC', '^VXN', 'NASDAQ Composite con VXN como índice de volatilidad asociado');

INSERT INTO feature_schema_versions (version, columns, is_current) VALUES
    ('v1', '{"columns": ["main_log_return","main_log_range","main_body_log","main_upper_wick_log","main_lower_wick_log","main_vol_5d","main_vol_10d","vol_idx_log_close","vol_idx_log_range","vol_idx_log_return","day_of_week"]}'::jsonb, true);

-- ============================================================================
-- Fin del script
-- ============================================================================
