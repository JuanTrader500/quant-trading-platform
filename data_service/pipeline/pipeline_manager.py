"""
pipeline_manager.py
--------------------
Orquesta extracción + preparación y decide si un reentrenamiento es
necesario.

RNF10: no depende de Django ni de ningún componente web — se ejecuta
como script o módulo independiente, o puede invocarse desde un
management command / scheduler sin acoplarse a las vistas.

Uso
---
    python -m DataPipeline.pipeline_manager
"""

from datetime import datetime
from pathlib import Path

import joblib
from dateutil.relativedelta import relativedelta

from .extraction import DataExtractor
from .logging_config import get_logger
from .preparation import DataPreparer
from .settings import DEFAULT_START_DATE, MODEL_METADATA_PATH, RETRAINING_INTERVAL_WEEKS

logger = get_logger(__name__)


class PipelineManager:
    """Punto de entrada único: decide si correr el pipeline y lo ejecuta."""

    def __init__(self, model_metadata_path: str | Path | None = None, start_date: str = DEFAULT_START_DATE):
        self.model_metadata_path = Path(model_metadata_path) if model_metadata_path else MODEL_METADATA_PATH
        self.start_date = start_date

    def check_retraining_needed(self) -> bool:
        if not self.model_metadata_path.exists():
            logger.info("No existe metadata de modelo → se requiere reentrenamiento.")
            return True
        try:
            metadata = joblib.load(self.model_metadata_path)
            last_trained = datetime.strptime(metadata["last_trained"], "%Y-%m-%d")
        except Exception as exc:
            logger.warning(f"No se pudo leer la metadata ({exc}) → se requiere reentrenamiento.")
            return True

        deadline = last_trained + relativedelta(weeks=RETRAINING_INTERVAL_WEEKS)
        if datetime.now() >= deadline:
            logger.info(f"Modelo del {last_trained.date()} supera {RETRAINING_INTERVAL_WEEKS} semana(s) → reentrenar.")
            return True

        logger.info(f"Modelo vigente (entrenado {last_trained.date()}) → se omite el pipeline.")
        return False

    def run_full_pipeline(self) -> bool:
        logger.info("Iniciando pipeline de datos …")

        if not self._run_extraction():
            logger.error("Pipeline abortado en la etapa de extracción.")
            return False
        if not self._run_preparation():
            logger.error("Pipeline abortado en la etapa de preparación.")
            return False

        logger.info("Pipeline de datos completado exitosamente.")
        return True

    def execute(self) -> None:
        """Punto de entrada: solo corre el pipeline si hace falta reentrenar."""
        if not self.check_retraining_needed():
            logger.info("Nada que hacer — se usa el modelo vigente para inferencia.")
            return

        if not self.run_full_pipeline():
            logger.error("Pipeline falló — se conserva el modelo vigente para inferencia (RNF04).")

    # ------------------------------------------------------------------
    # Etapas internas
    # ------------------------------------------------------------------

    def _run_extraction(self) -> bool:
        try:
            extractor = DataExtractor(start_date=self.start_date)
            results = extractor.download_all(DataExtractor.load_config())
            failed = [name for name, ok in results.items() if not ok]
            if failed:
                logger.error(f"Activos que fallaron: {failed}")
                return False
            return True
        except Exception as exc:
            logger.error(f"Extracción lanzó una excepción: {exc}", exc_info=True)
            return False

    def _run_preparation(self) -> bool:
        try:
            results = DataPreparer().run_pipeline()
            if not results:
                logger.error("La preparación no produjo ningún dataset.")
                return False
            logger.info(f"Datasets preparados: {list(results.keys())}")
            return True
        except Exception as exc:
            logger.error(f"Preparación lanzó una excepción: {exc}", exc_info=True)
            return False


if __name__ == "__main__":
    PipelineManager().execute()
