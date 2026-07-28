"""
scheduler/scheduler.py
-------------------------
RF08: scheduler interno (APScheduler) que corre el reentrenamiento
(incluye la Walk-Forward Validation, ver `training/walk_forward.py`)
el primero de cada mes, para estar alineado con el entorno de
producción real. No depende de un CronJob externo de Kubernetes ni de
Django: vive dentro del propio proceso del ML Service.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from core.logging_config import get_logger, set_trace_id
from core.settings import (
    RETRAIN_SCHEDULE_DAY,
    RETRAIN_SCHEDULE_HOUR,
    RETRAIN_SCHEDULE_MINUTE,
    RETRAIN_SCHEDULE_TIMEZONE,
)
from training.retrain_manager import run_retraining

logger = get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def _scheduled_retrain_job() -> None:
    set_trace_id(None)  # nuevo trace_id para esta corrida disparada por cron
    try:
        outcome = run_retraining()
        logger.info(f"Job de reentrenamiento mensual finalizado: {outcome}")
    except Exception:
        logger.error("El job de reentrenamiento mensual falló.", exc_info=True)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler(timezone=RETRAIN_SCHEDULE_TIMEZONE)
    _scheduler.add_job(
        _scheduled_retrain_job,
        trigger=CronTrigger(
            day=RETRAIN_SCHEDULE_DAY,
            hour=RETRAIN_SCHEDULE_HOUR,
            minute=RETRAIN_SCHEDULE_MINUTE,
            timezone=RETRAIN_SCHEDULE_TIMEZONE,
        ),
        id="monthly_retrain",
        name="Reentrenamiento mensual (Walk-Forward Validation)",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    _scheduler.start()
    logger.info(
        f"Scheduler iniciado: reentrenamiento el día {RETRAIN_SCHEDULE_DAY} de cada mes "
        f"a las {RETRAIN_SCHEDULE_HOUR:02d}:{RETRAIN_SCHEDULE_MINUTE:02d} ({RETRAIN_SCHEDULE_TIMEZONE})."
    )
    return _scheduler


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
