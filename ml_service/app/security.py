"""
app/security.py
------------------
RF41: cada llamada HTTP entre Web Service, ML Service y Data Service
debe incluir una clave de servicio interna que identifique al servicio
emisor. Este módulo valida esa clave en las peticiones ENTRANTES al ML
Service (el envío saliente hacia el Data Service vive en
`clients/data_service_client.py`).

RNF20: el ML Service no debe exponerse directamente a internet, solo
alcanzable dentro de la red interna de Docker; esta clave es una capa
adicional de defensa en profundidad, no el único control.
"""

from fastapi import Header, HTTPException, status

from core.settings import ML_SERVICE_API_KEY, REQUIRE_API_KEY


def verify_service_key(x_service_key: str | None = Header(default=None)) -> None:
    if not REQUIRE_API_KEY:
        return
    if not ML_SERVICE_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ML_SERVICE_API_KEY no está configurada en el servidor.",
        )
    if x_service_key != ML_SERVICE_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clave de servicio inválida.")
