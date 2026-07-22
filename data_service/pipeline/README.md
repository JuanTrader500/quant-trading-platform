# DataPipeline

Módulo de ingesta y procesamiento de datos del proyecto **Predicción de
Volatilidad SP500**. Cubre el **Módulo 1 (Data Engineering)** de los
requerimientos del sistema, y está diseñado para ejecutarse de forma
totalmente independiente del backend web (RNF10).

## Ubicación en el proyecto

```
sp500_MLops/
├── src/
│   ├── DataPipeline/          ← código (este paquete)
│   │   ├── __init__.py
│   │   ├── settings.py        # rutas y constantes centralizadas
│   │   ├── logging_config.py  # logger compartido (consola + archivo)
│   │   ├── extraction.py      # descarga OHLC vía yfinance
│   │   ├── validation.py      # integridad de datos crudos
│   │   ├── preparation.py     # feature engineering
│   │   ├── feature_schema.py  # versionado del esquema de features
│   │   ├── pipeline_manager.py# orquestador + gate de reentrenamiento
│   │   ├── config/
│   │   │   └── assets.yaml    # activos a extraer
│   │   └── README.md
│   └── data/                  ← datos generados (no versionar en git)
│       ├── raw/                #  <asset>_data_daily.csv
│       └── processed/          #  processed_<dataset>.csv + *_schema.json
├── models/
│   └── artifacts/
│       └── model_metadata.pkl
└── logs/
    └── data_pipeline.log
```

Todas las rutas se resuelven en `settings.py` a partir de la posición
del paquete — nada está hardcodeado en el resto de los módulos.

## Flujo de ejecución

```
PipelineManager.execute()
  ├── check_retraining_needed()   → True si no hay metadata o el modelo
  │                                  tiene más de 1 semana
  └── run_full_pipeline()
        ├── DataExtractor.download_all()   (extraction.py)
        │     └── DataValidator.validate() (validation.py)
        └── DataPreparer.run_pipeline()    (preparation.py)
              └── enforce_schema() + write_schema_manifest() (feature_schema.py)
```

## Uso
# Desde src/ (recomendado para producción/cron)
python -m DataPipeline.pipeline_manager

# Directamente dentro de la carpeta (como lo probaste tú)
cd src/DataPipeline
python pipeline_manager.py
```

Desde Django (o cualquier orquestador), basta con importar la clase:

```python
from DataPipeline import PipelineManager
PipelineManager().execute()
```

No hay llamadas a `subprocess`: los errores se propagan con traceback
completo, y el módulo no importa nada de Django, por lo que puede
correr en un cron/worker aislado (RNF10).

## Cobertura de requerimientos

| Requerimiento | Dónde se cumple |
|---|---|
| **RF01** — Extraer OHLC de SP500/VIX vía Yahoo Finance | `extraction.py` (`DataExtractor`, `config/assets.yaml`) |
| **RF02** — Calcular y almacenar las features derivadas | `preparation.py` (`_engineer_features`) |
| **RF03** — Evitar data leakage (solo info hasta t) | `preparation.py`: todas las features se calculan en t; `target` es el único valor desplazado (`shift(-1)`) |
| **RF04** — Validar integridad antes de persistir (nulos, duplicados, gaps) | `validation.py` (`DataValidator`), invocado dentro de `download_asset` antes de guardar |
| **RF05** — Actualización incremental sin re-descargar todo | `extraction.py` (`_merge_with_integrity`) |
| **RF06** — Logging de fecha de ejecución, rango obtenido y errores de API | `logging_config.py` + logs en cada etapa de `extraction.py` → `logs/data_pipeline.log` |
| **RNF10** — Desacoplar pipeline de la capa web | Paquete independiente, sin dependencias de Django, ejecutable como script/módulo |
| **RNF11** — Estructura modular para agregar activos/algoritmos sin tocar otras capas | `DATASET_CONFIGS` en `preparation.py` y `assets.yaml`: agregar un activo no requiere tocar lógica existente |
| **RNF12** — Versionar el esquema de features | `feature_schema.py`: cada dataset procesado se acompaña de `<dataset>_schema.json` con versión y hash de columnas |

## Esquema de features versionado

Cada corrida de `preparation.py` escribe, junto al CSV procesado, un
manifiesto de esquema:

```json
{
  "dataset": "sp500",
  "schema_version": "1.0.0",
  "columns": ["sp500_log_return", "...", "target"],
  "hash": "a1b2c3d4e5f6"
}
```

El módulo de entrenamiento debe comparar este `hash` contra el
guardado en `model_metadata.pkl` del modelo vigente: si difiere,
implica que el esquema cambió y el modelo debe reentrenarse antes de
usarse en producción (evita `RNF12` — predicciones con features
inconsistentes).

## Variables y justificación estadística

La definición matemática de cada feature (`sp500_log_return`,
`vix_log_range`, `day_of_week`, etc.) y su justificación como señal de
volatilidad clustering, sesgo direccional o efecto calendario está
documentada en el **Diccionario de Variables** del proyecto (README
raíz), no se repite aquí para evitar desincronización entre
documentos.
