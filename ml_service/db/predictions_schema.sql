-- predictions_schema.sql
-- ------------------------
-- Historial de predicciones día a día del ML Service (petición
-- explícita: poder consultarlas/graficarlas después). Vive en un
-- SQLite propio dentro de un volumen Docker del ML Service — no en el
-- Data Service ni en el Web Service (RNF17: cada servicio con estado
-- es dueño exclusivo de su propia base de datos).

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date         TEXT NOT NULL,          -- fecha objetivo de la predicción (t+1)
    predicted_at        TEXT NOT NULL,          -- timestamp UTC de cuándo se generó
    pair_code           TEXT NOT NULL,
    mode                TEXT NOT NULL CHECK (mode IN ('testing', 'automatic', 'backtest')),
    predicted_log_range REAL NOT NULL,          -- log(High/Low) predicho, escala del modelo (RF16)
    predicted_range_pct REAL NOT NULL,          -- e^predicted_log_range - 1
    anchor_close        REAL NOT NULL,          -- cierre real usado para anclar la conversión a puntos
    predicted_range_points REAL NOT NULL,       -- anchor_close * predicted_range_pct
    actual_range        REAL,                   -- se completa después, cuando se conoce (RF24)
    model_name          TEXT NOT NULL,
    model_version        TEXT,
    mlflow_run_id       TEXT,
    trace_id            TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_target_date ON predictions (target_date);
CREATE INDEX IF NOT EXISTS idx_predictions_mode ON predictions (mode);