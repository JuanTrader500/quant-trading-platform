"""
logging_config.py
------------------
Logger único compartido por todo el paquete `pipeline`.

RF06: cada corrida del pipeline debe quedar registrada en logs (fecha
de ejecución, rango de fechas obtenido, errores de conexión). Este
logger escribe simultáneamente a consola (para desarrollo/Docker logs)
y a un archivo persistente (para auditoría posterior). El detalle
estructurado por corrida (rango de fechas, filas afectadas, estado)
se registra además en la tabla `ingestion_log` vía `db.log_run` — este
logger cubre la traza legible por humanos, la tabla cubre la traza
consultable.
"""

import logging

from .settings import LOG_FILE

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado para `name`, reutilizando handlers
    si ya fue creado antes (evita duplicar líneas de log en re-imports)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FORMAT))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
