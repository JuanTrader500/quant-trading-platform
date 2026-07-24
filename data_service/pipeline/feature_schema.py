"""
feature_schema.py
------------------
RNF12: versiona el esquema de features (nombres y orden de columnas).

A diferencia de versiones anteriores del proyecto, el esquema es
único y genérico (no uno distinto por activo): todas las columnas se
nombran `main_*` (índice principal: SP500 o NASDAQ) y `vol_idx_*`
(índice de volatilidad asociado: VIX o VXN). Un solo esquema describe
cualquier par índice+volatilidad, así que agregar un par nuevo no
crea una versión de esquema nueva — solo agrega filas en registry.py.

La versión vigente se registra en la tabla `feature_schema_versions`
(ver db.py, llamado desde preparation.py al inicio de cada corrida).
Si en el futuro cambias las columnas de aquí, sube SCHEMA_VERSION: la
próxima corrida registrará la nueva versión y las filas de `features`
calculadas con el esquema anterior quedan trazablemente distinguidas
por su columna `schema_version`.
"""

SCHEMA_VERSION = "v1"

# Orden exacto de columnas de valor de la tabla `features` (sin contar
# las claves pair_code/date/schema_version, que van aparte). Debe
# coincidir 1:1 con las columnas de la tabla `features` en
# docs/data_service_schema.sql.
FEATURE_COLUMNS: list[str] = [
    "main_log_return",
    "main_log_range",
    "main_body_log",
    "main_upper_wick_log",
    "main_lower_wick_log",
    "main_vol_5d",
    "main_vol_10d",
    "vol_idx_log_close",
    "vol_idx_log_range",
    "vol_idx_log_return",
    "day_of_week",
    "target_range_next_day",
]


def enforce_schema(df):
    """Reordena y valida las columnas de `df` contra el esquema esperado.

    Falla rápido (ValueError) si falta una columna requerida, en vez de
    dejar que un cambio silencioso upstream en `preparation.py` llegue
    a persistirse con columnas inconsistentes en la base de datos.
    """
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas del esquema esperado: {sorted(missing)}")
    return df[FEATURE_COLUMNS]


def register_current_version() -> None:
    """Registra (si hace falta) y marca como vigente en la base de datos
    la versión de esquema activa en este código.

    Import perezoso de `db` para evitar un ciclo de imports a nivel de
    módulo (db.py importa FEATURE_COLUMNS de este archivo).
    """
    from . import db

    db.register_schema_version(SCHEMA_VERSION, FEATURE_COLUMNS)
