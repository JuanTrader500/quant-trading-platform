"""
registry.py
-----------
Catálogo único de activos y pares índice + índice de volatilidad
soportados por el Data Service.

Es la fuente de verdad que usan tanto `extraction.py` (qué tickers
descargar de Yahoo Finance) como `preparation.py` (qué par de columnas
generar): agregar un activo o un par nuevo es una entrada aquí, nunca
un cambio de lógica en otro módulo (RNF11).

Debe reflejar exactamente los datos semilla de las tablas `instruments`
y `asset_pairs` de `docs/data_service_schema.sql`. Si agregas un
activo aquí, agrega también su INSERT correspondiente en ese script.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetInfo:
    """Un instrumento individual descargable de Yahoo Finance."""

    name: str  # nombre corto interno usado en logs, ej. "sp500"
    ticker: str  # símbolo real de Yahoo Finance; debe coincidir con instruments.ticker
    is_volatility: bool  # True si es un índice de volatilidad (VIX, VXN, ...)


@dataclass(frozen=True)
class PairInfo:
    """Un par índice principal + su índice de volatilidad asociado."""

    pair_code: str  # debe coincidir con asset_pairs.pair_code
    index_asset: str  # AssetInfo.name del índice principal
    volatility_asset: str  # AssetInfo.name del índice de volatilidad asociado


# Activos soportados. Agregar uno nuevo (ej. Dow Jones + su índice de
# volatilidad) es agregar dos entradas aquí y una entrada en PAIRS.
ASSETS: dict[str, AssetInfo] = {
    "sp500": AssetInfo(name="sp500", ticker="^GSPC", is_volatility=False),
    "vix": AssetInfo(name="vix", ticker="^VIX", is_volatility=True),
    "nq": AssetInfo(name="nq", ticker="^IXIC", is_volatility=False),
    "vxn": AssetInfo(name="vxn", ticker="^VXN", is_volatility=True),
}

# Pares sobre los que se calculan features. Cada pair_code debe existir
# también como fila en la tabla asset_pairs.
PAIRS: dict[str, PairInfo] = {
    "SP500_VIX": PairInfo(pair_code="SP500_VIX", index_asset="sp500", volatility_asset="vix"),
    "NASDAQ_VXN": PairInfo(pair_code="NASDAQ_VXN", index_asset="nq", volatility_asset="vxn"),
}


def all_assets() -> list[AssetInfo]:
    """Devuelve todos los activos registrados, usado por extraction.py."""
    return list(ASSETS.values())


def all_pairs() -> list[PairInfo]:
    """Devuelve todos los pares registrados, usado por preparation.py."""
    return list(PAIRS.values())
