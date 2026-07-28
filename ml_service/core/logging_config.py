"""
core/logging_config.py
-----------------------
Logger estructurado (JSON) compartido por todo el ML Service (RNF19):
cada línea de log es un objeto JSON con timestamp, nivel, logger,
mensaje y trace_id de correlación, para poder enviarse tal cual a un
stack de logging centralizado (ELK, Datadog, etc.).

El trace_id se origina en el Web Service (o, si la petición llega
directo durante pruebas, en este mismo servicio) y se propaga por el
header HTTP `X-Trace-Id` hacia el Data Service, permitiendo rastrear
una petición de punta a punta entre los 3 servicios.
"""

import contextvars
import json
import logging
import uuid
from datetime import datetime, timezone

from .settings import LOG_FILE

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex


def set_trace_id(trace_id: str | None) -> str:
    """Fija el trace_id del contexto actual (por petición). Si no viene
    uno desde el Web Service, se genera uno nuevo aquí mismo."""
    trace_id = trace_id or new_trace_id()
    _trace_id_var.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    return _trace_id_var.get()


class _TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Devuelve un logger configurado para `name`, reutilizando handlers
    si ya fue creado antes (evita duplicar líneas de log en re-imports)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    trace_filter = _TraceIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(_JsonFormatter())
    console.addFilter(trace_filter)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(_JsonFormatter())
    file_handler.addFilter(trace_filter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
