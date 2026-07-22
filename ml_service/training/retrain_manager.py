import os
import sys
import logging
import subprocess
import joblib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from arch import arch_model

from .validation_utils import (
    calculate_rmse, calculate_mae, calculate_directional_bias, walk_forward_splits
)

# Configuración de rutas absolutas
ROOT_DIR = Path("/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops")
DATA_PROCESSED_PATH = ROOT_DIR / "data_service" / "pipeline" / "data" / "processed" / "processed_sp500.csv"
PIPELINE_MANAGER_PATH = ROOT_DIR / "data_service" / "pipeline" / "pipeline_manager.py"
ARTIFACTS_ROOT = ROOT_DIR / "ml_service" / "artifacts"
LOG_FILE = ROOT_DIR / "logs" / "ml_service" / "retraining.log"

# Logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class RetrainManager:
    def __init__(self):
        self.rf_params = {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 5}
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.artifacts_dir = ARTIFACTS_ROOT / self.date_str
        
    def run_data_pipeline(self):
        """Ejecuta el pipeline de ingesta y procesamiento (RF01-RF06)."""
        logger.info("Iniciando disparador de ingesta de datos...")
        try:
            # Ejecutar como módulo para mantener consistencia de rutas
            result = subprocess.run(
                [sys.executable, "-m", "data_service.pipeline.pipeline_manager"],
                cwd=ROOT_DIR / "data_service",
                capture_output=True, text=True, check=True
            )
            logger.info("DataPipeline completado exitosamente.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error en DataPipeline: {e.stderr}")
            raise RuntimeError("El pipeline de datos falló. Abortando reentrenamiento.")

    def train_and_validate(self):
        """Entrena modelos y realiza validación Walk-Forward (RF08)."""
        logger.info("Cargando datos procesados...")
        df = pd.read_csv(DATA_PROCESSED_PATH, index_col="date", parse_dates=["date"]).sort_index()
        
        # Preparación de Features
        target_col = "target"
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Configuración Walk-Forward
        N = len(df)
        initial_train_size = int(N * 0.7)
        step_size = 21 # Mensual
        folds = list(walk_forward_splits(N, initial_train_size, step_size))
        
        # 1. RandomForest (ML Model)
        logger.info("Entrenando RandomForest...")
        rf = RandomForestRegressor(**self.rf_params, random_state=42)
        scaler = StandardScaler()
        
        # Entrenamiento final sobre todo el dataset para producción
        X_scaled = scaler.fit_transform(X)
        rf.fit(X_scaled, y)
        
        # 2. GARCH(1,1) Baseline (RF10)
        logger.info("Entrenando Baseline GARCH...")
        returns = df["sp500_log_return"] * 100
        am = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal", mean="Zero", rescale=False)
        garch_res = am.fit(disp="off")
        
        # Calibración GARCH (target ~ a*sigma + b) usando el último fold de entrenamiento
        # para evitar leakage en la métrica
        train_idx, _ = folds[0]
        sigma_in_sample = garch_res.conditional_volatility.iloc[train_idx] / 100
        y_train_sample = y.iloc[train_idx]
        calib_coeffs = np.polyfit(sigma_in_sample, y_train_sample, 1)

        # 3. Validación de Rendimiento (Sexto fold como representación)
        # Usamos el último fold disponible para reportar métricas actuales
        last_train, last_test = folds[-1]
        X_tr, X_te = X.iloc[last_train], X.iloc[last_test]
        y_te = y.iloc[last_test]
        
        # Validar RF
        s_val = StandardScaler().fit(X_tr)
        y_pred_rf = rf.predict(s_val.transform(X_te))
        
        # Validar GARCH
        sigma_forecast = np.sqrt(garch_res.forecast(horizon=len(last_test), reindex=False).variance.values[-1]) / 100
        y_pred_garch = np.polyval(calib_coeffs, sigma_forecast)

        metrics = {
            "rf": {
                "rmse": calculate_rmse(y_te, y_pred_rf),
                "mae": calculate_mae(y_te, y_pred_rf),
                "bias": calculate_directional_bias(y_te, y_pred_rf)
            },
            "garch": {
                "rmse": calculate_rmse(y_te, y_pred_garch),
                "mae": calculate_mae(y_te, y_pred_garch),
                "bias": calculate_directional_bias(y_te, y_pred_garch)
            }
        }
        
        return rf, scaler, garch_res, calib_coeffs, metrics

    def save_artifacts(self, rf, scaler, garch_res, calib, metrics):
        """Serializa modelos y metadatos (RF11)."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(rf, self.artifacts_dir / "rf_model.pkl")
        joblib.dump(scaler, self.artifacts_dir / "scaler.pkl")
        joblib.dump(garch_res, self.artifacts_dir / "garch_model.pkl")
        
        metadata = {
            "date": self.date_str,
            "rf_params": self.rf_params,
            "garch_calibration": calib.tolist(),
            "metrics": metrics
        }
        
        with open(self.artifacts_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Artefactos guardados en {self.artifacts_dir}")

    def execute(self):
        try:
            self.run_data_pipeline()
            rf, scaler, garch_res, calib, metrics = self.train_and_validate()
            self.save_artifacts(rf, scaler, garch_res, calib, metrics)
            logger.info("Ciclo de reentrenamiento completado con éxito.")
        except Exception as e:
            logger.error(f"Error crítico en el manager: {str(e)}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    manager = RetrainManager()
    manager.execute()
