"""
logging_config.py
------------------
Logger único compartido por todo el paquete.

RF06: cada corrida del pipeline debe quedar registrada en logs
(fecha de ejecución, rango de fechas obtenido, errores de conexión).
Escribe simultáneamente a consola y a archivo persistente.
"""

import logging

from .settings import LOG_FILE

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # evita handlers duplicados en re-imports
        return logger

    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT))

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_FORMAT))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
