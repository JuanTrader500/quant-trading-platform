"""
pipeline_manager.py
--------------------
Orquesta extracción + preparación del Data Service.

RNF10: no depende de Django ni de ningún componente web — se ejecuta
como script, como módulo independiente, o desde un endpoint del propio
Data Service (FastAPI, ver app/main.py) sin acoplarse a otras capas.

Nota de diseño: a diferencia de versiones anteriores del proyecto,
este orquestador ya NO decide "si hace falta reentrenar" leyendo
metadata de un modelo de ML. Esa decisión pertenece al ML Service, que
vive en otro proceso con su propio Model Registry (MLflow) — leer sus
artefactos desde aquí violaría el aislamiento de datos por servicio
(RNF17). Este módulo siempre corre extracción + preparación cuando se
invoca; la periodicidad (ej. diaria) se define fuera de este archivo,
vía cron, un scheduler o un CronJob de Kubernetes que llame a
`PipelineManager().execute()` o al endpoint POST /pipeline/run.

Uso
---
    python -m pipeline.pipeline_manager
"""

from .extraction import DataExtractor
from .logging_config import get_logger
from .preparation import DataPreparer

logger = get_logger(__name__)


class PipelineManager:
    """Punto de entrada único: corre extracción y luego preparación."""

    def run_full_pipeline(self) -> bool:
        """Corre las dos etapas en orden. Se detiene en la primera que
        falle (la preparación depende de que la extracción haya
        escrito datos frescos en raw_ohlc)."""
        logger.info("Iniciando pipeline de datos …")

        if not self._run_extraction():
            logger.error("Pipeline abortado en la etapa de extracción.")
            return False
        if not self._run_preparation():
            logger.error("Pipeline abortado en la etapa de preparación.")
            return False

        logger.info("Pipeline de datos completado exitosamente.")
        return True

    def execute(self) -> bool:
        """Alias explícito, pensado para ser llamado por un scheduler
        externo o por el endpoint POST /pipeline/run del servicio."""
        return self.run_full_pipeline()

    # ------------------------------------------------------------------
    # Etapas internas
    # ------------------------------------------------------------------

    def _run_extraction(self) -> bool:
        try:
            results = DataExtractor().download_all()
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
            logger.info(f"Pares procesados: {list(results.keys())}")
            return True
        except Exception as exc:
            logger.error(f"Preparación lanzó una excepción: {exc}", exc_info=True)
            return False


if __name__ == "__main__":
    ok = PipelineManager().execute()
    raise SystemExit(0 if ok else 1)
