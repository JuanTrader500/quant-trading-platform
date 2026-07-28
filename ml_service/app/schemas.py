"""
app/schemas.py
----------------
Esquemas Pydantic de entrada/salida de la API del ML Service.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class _DomainModel(BaseModel):
    """Varios campos del dominio (model_used, model_version, model_name)
    empiezan legítimamente con "model_"; se desactiva el namespace
    protegido de pydantic para no arrastrar warnings en cada arranque."""

    model_config = ConfigDict(protected_namespaces=())


class TestingPredictionRequest(_DomainModel):
    """RF14/RF21: modo Testing, OHLC ingresado manualmente."""

    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    as_of: date = Field(default_factory=date.today, description="Fecha que representa el OHLC ingresado.")
    pair_code: str | None = Field(default=None, description="Por defecto, el configurado en el servidor.")


class PredictionResponse(_DomainModel):
    """RF16: salida mínima de una predicción."""

    predicted_range: float
    target_date: date
    model_used: str
    model_version: str | None = None
    mode: str


class MetricsResponse(_DomainModel):
    """RF17/RF18: métricas del modelo vigente."""

    model_name: str
    model_version: str | None
    metrics_by_model: dict
    comparison_vs_garch: dict


class ReloadResponse(_DomainModel):
    reloaded: bool
    model_name: str | None
    model_version: str | None
