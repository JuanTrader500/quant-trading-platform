# SP500 Data Pipeline

Data pipeline for SP500 and VIX financial data. Extracts daily OHLCV data from Yahoo Finance, cleans it, and generates engineered features for machine learning models.

## Data Pipeline

The pipeline has two stages:

### 1. Extraction (`src/data_pipeline/extraction.py`)

Downloads daily OHLCV data from Yahoo Finance for the configured assets.

**Assets downloaded:**
- `^GSPC` (SP500 Index) — saved as `data/raw/sp500_df_data_daily.csv`
- `^VIX` (Volatility Index) — saved as `data/raw/vix_2026_data_daily.csv`

For each asset, it computes:
- `pct_move` (for SP500): `(high - low) / open * 100`
- `avg_hl` (for VIX): `(high + low) / 2`

**Data integrity:** When the CSV already exists, the extractor merges the new data with the existing data instead of overwriting. If the API response is missing dates that were present in the existing file (e.g., partial data from Yahoo Finance), those dates are preserved from the historical CSV. A warning is logged for each asset when missing dates are detected and recovered.

Run standalone:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/data_pipeline/extraction.py
```

Assets are configured in `config/assets.yaml`.

### 2. Preparation (`src/features/preparation.py`)

Loads the raw CSVs, merges SP500 and VIX data on date, cleans outliers using IQR (trained only on data up to 2020-12-31 to prevent leakage), and engineers features.

**Feature engineering outputs (saved to `data/processed/processed_features.csv`):**
- Log-transformed prices (`close_log`, `high_log`, `low_log`)
- Target variables: `log_target` (log return), `target_high`, `target_low`
- Lagged features: `log_return`, `Upper_Wick_lag1`, `Lower_Wick_lag1`, `close_log_lag1`
- VIX-based features: `VIX_Vol_Diaria_lag1`, `VIX_Rango_Log_lag1`

Run standalone:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/features/preparation.py
```

### 3. Pipeline Manager (`src/pipeline_manager.py`)

Orchestrates the full data pipeline programmatically. It checks whether the processed data needs updating (via `model_metadata.pkl` from the training step), and if so, runs extraction and preparation sequentially via subprocess.

**Logic:**
1. Looks for `models/artifacts/model_metadata.pkl` (generated when you train a model).
2. If the file is missing or the model is older than 1 week, it runs the full pipeline (extraction → preparation).
3. If the model is up-to-date, it skips the pipeline and logs that no update is needed.

This is the recommended way to run the pipeline when it's part of an automated MLOps workflow:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/pipeline_manager.py
```

Or step by step:
```
export PYTHONPATH=$PYTHONPATH:$(pwd) && \
python3 src/data_pipeline/extraction.py && \
python3 src/features/preparation.py
```

## Project Structure

```
├── config/
│   └── assets.yaml              # Asset ticker configuration
├── data/
│   ├── raw/                     # Raw CSV from Yahoo Finance
│   │   ├── sp500_df_data_daily.csv
│   │   └── vix_2026_data_daily.csv
│   └── processed/               # Engineered features
│       └── processed_features.csv
├── src/
│   ├── data_pipeline/
│   │   └── extraction.py        # Yahoo Finance downloader
│   ├── features/
│   │   └── preparation.py       # Feature engineering & cleaning
│   └── pipeline_manager.py      # Pipeline orchestrator
├── tests/
│   └── test_pipeline.py         # Unit tests
├── requirements.txt
└── README.md
```

## Model Training (WIP)

This repository provides the data pipeline. ML model training code can be added in a new `src/models/` directory. The processed features in `data/processed/processed_features.csv` are ready for consumption by any regression model (the target column is `log_target`).

## Quick Start

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 src/data_pipeline/extraction.py
python3 src/features/preparation.py
```

## Scheduling the pipeline weekly

You can run the pipeline automatically every week. Two common approaches:

- Cron example (run every Monday at 02:00):

```bash
# Edit the crontab with `crontab -e` and add:
0 2 * * 1 cd /home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops && /usr/bin/python3 src/pipeline_manager.py >> pipeline.log 2>&1
```

- systemd user timer example (create two files under `~/.config/systemd/user/`):

`sp500-pipeline.service`:
```
[Unit]
Description=Run SP500 data pipeline

[Service]
Type=oneshot
WorkingDirectory=/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops
ExecStart=/usr/bin/python3 src/pipeline_manager.py
```

`sp500-pipeline.timer` (run weekly):
```
[Unit]
Description=Weekly SP500 data pipeline timer

[Timer]
OnCalendar=Mon *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start with:

```bash
systemctl --user daemon-reload
systemctl --user enable --now sp500-pipeline.timer
```