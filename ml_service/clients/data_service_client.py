"""
clients/data_service_client.py
-------------------------------
Cliente HTTP hacia el Data Service (RF39: llamadas REST directas y
síncronas, ML Service -> Data Service, sin bus de eventos).

RF41: cada llamada incluye la clave de servicio interna en el header
`X-Service-Key`. El Data Service actual (main.py provisto) todavía no
la valida, pero se envía igual para no requerir otro cambio en este
cliente cuando se agregue esa verificación del otro lado.

RF42: timeouts y reintentos explícitos con backoff. Si el Data Service
no responde a tiempo tras agotar los reintentos, se lanza
`DataServiceUnavailableError`, que la capa de arriba (endpoints FastAPI)
traduce en un error controlado (RNF03) sin exponer trazas técnicas.

RNF19: el trace_id del contexto actual se propaga por el header
`X-Trace-Id` hacia el Data Service.
"""

from datetime import date
import time

import httpx

from core.logging_config import get_logger, get_trace_id
from core.settings import (
    DATA_SERVICE_BASE_URL,
    HTTP_MAX_RETRIES,
    HTTP_RETRY_BACKOFF_SECONDS,
    HTTP_TIMEOUT_SECONDS,
    INTERNAL_SERVICE_KEY,
)

logger = get_logger(__name__)


class DataServiceUnavailableError(Exception):
    """El Data Service no respondió a tiempo tras agotar los reintentos,
    o devolvió un error de servidor. RNF03: quien atrape esta excepción
    debe mostrar un mensaje claro sin stack trace en producción."""


class DataServiceNotFoundError(Exception):
    """El Data Service respondió 404 (ej. pair_code desconocido o sin
    features calculadas todavía)."""


def _headers() -> dict:
    return {
        "X-Service-Key": INTERNAL_SERVICE_KEY,
        "X-Trace-Id": get_trace_id(),
    }


def _get_with_retries(path: str, params: dict | None = None) -> httpx.Response:
    url = f"{DATA_SERVICE_BASE_URL}{path}"
    last_exc: Exception | None = None

    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = client.get(url, params=params, headers=_headers())
            if response.status_code == 404:
                raise DataServiceNotFoundError(response.json().get("detail", "No encontrado"))
            if response.status_code >= 500:
                raise DataServiceUnavailableError(
                    f"Data Service respondió {response.status_code} en {path}"
                )
            response.raise_for_status()
            return response
        except DataServiceNotFoundError:
            raise
        except (httpx.TimeoutException, httpx.ConnectError, DataServiceUnavailableError) as exc:
            last_exc = exc
            logger.error(
                f"Intento {attempt}/{HTTP_MAX_RETRIES} fallido llamando a {path}: {exc}"
            )
            if attempt < HTTP_MAX_RETRIES:
                time.sleep(HTTP_RETRY_BACKOFF_SECONDS * attempt)

    raise DataServiceUnavailableError(
        f"Data Service no disponible en {path} tras {HTTP_MAX_RETRIES} intentos"
    ) from last_exc


def health() -> bool:
    """Healthcheck simple del Data Service, usado por /health del ML
    Service para reportar el estado de sus dependencias."""
    try:
        response = _get_with_retries("/health")
        return response.json().get("status") == "ok"
    except Exception:
        return False


def get_latest_features(pair_code: str) -> dict:
    """Última fila de features calculada para el par (RF15: modo
    "Predicción de Mañana")."""
    response = _get_with_retries("/features/latest", params={"pair_code": pair_code})
    return response.json()


def get_training_dataset(
    pair_code: str,
    date_from: date | None = None,
    date_to: date | None = None,
) -> list[dict]:
    """Histórico de features con target ya conocido, para entrenar /
    walk-forward validar (RF08, RF09). Sin fuga de datos: el Data
    Service ya garantiza (RF03) que cada fila usa solo datos <= t."""
    params: dict = {"pair_code": pair_code}
    if date_from is not None:
        params["date_from"] = date_from.isoformat()
    if date_to is not None:
        params["date_to"] = date_to.isoformat()
    response = _get_with_retries("/features/history", params=params)
    return response.json()
