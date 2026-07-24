"""Paquete `pipeline`: ingesta, validación y feature engineering del
Data Service. Ver README.md para el flujo completo y la cobertura de
requerimientos."""

from .db import get_engine
from .extraction import DataExtractor
from .pipeline_manager import PipelineManager
from .preparation import DataPreparer

__all__ = ["DataExtractor", "DataPreparer", "PipelineManager", "get_engine"]
