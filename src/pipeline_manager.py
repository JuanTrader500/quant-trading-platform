#PipelineManager
"""
pipeline_manager.py
-------------------
Orchestrates the full MLOps data refresh + retraining decision.

Execution order
---------------
1.  check_retraining_needed()
    • If no model metadata exists, or the model is > 1 week old → run pipeline
    • Otherwise → skip and proceed directly to inference

2.  run_full_pipeline()
    a.  DataExtractor  → downloads / updates raw CSVs for all assets
    b.  DataPreparer   → builds processed feature sets (one per asset pair)

Classes are imported directly — no subprocess calls — so errors surface
cleanly and the call stack is fully traceable.

Usage
-----
    python -m src.pipeline.pipeline_manager
    # or
    python src/pipeline/pipeline_manager.py
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when the file is executed directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_pipeline.extraction import DataExtractor  # noqa: E402
from features.preparation import DataPreparer       # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RETRAINING_INTERVAL_WEEKS = 1


class PipelineManager:
    """
    Central orchestrator for data extraction → feature preparation.

    Parameters
    ----------
    model_metadata_path : optional path to the model_metadata.pkl artifact.
        Defaults to  <project_root>/models/artifacts/model_metadata.pkl
    start_date : earliest date for data extraction (passed to DataExtractor).
    config_path : optional path to assets.yaml.
    """

    def __init__(
        self,
        model_metadata_path: str | Path | None = None,
        start_date: str = "2005-01-01",
        config_path: str | Path | None = None,
    ):
        self.project_root        = _PROJECT_ROOT
        self.start_date          = start_date
        self.config_path         = config_path
        self.model_metadata_path = (
            Path(model_metadata_path)
            if model_metadata_path
            else self.project_root / "models" / "artifacts" / "model_metadata.pkl"
        )

    # ------------------------------------------------------------------
    # Retraining gate
    # ------------------------------------------------------------------

    def check_retraining_needed(self) -> bool:
        """Return True when a fresh training run is required."""
        if not self.model_metadata_path.exists():
            logger.info("No model metadata found → retraining required.")
            return True

        try:
            metadata     = joblib.load(self.model_metadata_path)
            last_trained = datetime.strptime(metadata["last_trained"], "%Y-%m-%d")
        except Exception as exc:
            logger.warning(f"Could not read model metadata ({exc}) → retraining required.")
            return True

        deadline = last_trained + relativedelta(weeks=RETRAINING_INTERVAL_WEEKS)
        if datetime.now() >= deadline:
            logger.info(
                f"Model trained on {last_trained.date()} is older than "
                f"{RETRAINING_INTERVAL_WEEKS} week(s) → retraining required."
            )
            return True

        logger.info(f"Model is up-to-date (last trained {last_trained.date()}) → skipping pipeline.")
        return False

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def run_extraction(self) -> bool:
        """Step 1 — download / update all raw CSVs."""
        logger.info("═" * 60)
        logger.info("STEP 1 — Data Extraction")
        logger.info("═" * 60)
        try:
            extractor = DataExtractor(
                start_date=self.start_date,
                data_dir=Path.home() / "Documents" / "Develops" / "sp500_range_proyect" / "sp500_MLops" / "data" / "raw",
            )
            config  = DataExtractor.load_config(self.config_path)
            results = extractor.download_all(config)

            failed = [name for name, ok in results.items() if not ok]
            if failed:
                logger.error(f"Assets that failed to download: {failed}")
                return False

            logger.info("All assets downloaded/updated successfully.")
            return True
        except Exception as exc:
            logger.error(f"Extraction step raised an exception: {exc}", exc_info=True)
            return False

    def run_preparation(self) -> bool:
        """Step 2 — build processed feature datasets."""
        logger.info("═" * 60)
        logger.info("STEP 2 — Feature Preparation")
        logger.info("═" * 60)
        try:
            preparer = DataPreparer(
                raw_data_dir=Path.home() / "Documents" / "Develops" / "sp500_range_proyect" / "sp500_MLops" / "data" / "raw",
                processed_data_dir=Path.home() / "Documents" / "Develops" / "sp500_range_proyect" / "sp500_MLops" / "data" / "processed",
            )
            results = preparer.run_pipeline()

            if not results:
                logger.error("Preparation produced no output datasets.")
                return False

            logger.info(f"Prepared datasets: {list(results.keys())}")
            return True
        except Exception as exc:
            logger.error(f"Preparation step raised an exception: {exc}", exc_info=True)
            return False

    def run_full_pipeline(self) -> bool:
        """Run extraction then preparation. Returns True only if both succeed."""
        logger.info("Starting full data pipeline …")

        if not self.run_extraction():
            logger.error("Pipeline aborted at extraction step.")
            return False

        if not self.run_preparation():
            logger.error("Pipeline aborted at preparation step.")
            return False

        logger.info("Full data pipeline completed successfully.")
        return True

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def execute(self) -> None:
        """
        Gate-and-run: only execute the full pipeline when retraining is needed.
        Call this from  __main__  or from a scheduler (APScheduler, Airflow, cron).
        """
        if not self.check_retraining_needed():
            logger.info("Nothing to do — proceed to inference with the current model.")
            return

        success = self.run_full_pipeline()
        if success:
            logger.info("Pipeline finished — model artifacts are ready for retraining.")
        else:
            logger.error("Pipeline failed — existing model artifacts will be used for inference.")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    manager = PipelineManager()
    manager.execute()