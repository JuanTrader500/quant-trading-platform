# Data Service

Servicio de ingesta y procesamiento de datos de la plataforma de
predicción de volatilidad SP500 / NASDAQ. Cubre el **Módulo 1 (Data
Engineering)** de los requerimientos del sistema y corre de forma
totalmente independiente de los demás servicios (RNF10, RNF17: tiene
su propia base de datos, nadie más se conecta a ella directamente).

> **Cambio respecto a versiones anteriores del proyecto:** el pipeline
> ya no lee/escribe archivos CSV en disco. Toda la persistencia es en
> PostgreSQL + TimescaleDB, conectado vía la variable de entorno
> `DATABASE_URL`. El esquema completo (DDL) está en
> `docs/data_service_schema.sql` — pídeselo a quien administre la base
> de datos, o ejecútalo tú mismo siguiendo el tutorial de esa
> documentación.

## Estructura del proyecto

```
data_service/
├── pipeline/
│   ├── __init__.py
│   ├── settings.py         # configuración leída de variables de entorno (.env)
│   ├── logging_config.py   # logger compartido (consola + archivo)
│   ├── registry.py         # catálogo único de activos y pares índice+volatilidad
│   ├── db.py                # capa de acceso a datos (PostgreSQL/TimescaleDB)
│   ├── extraction.py        # descarga OHLC vía yfinance (RF01, RF05)
│   ├── validation.py         # integridad de datos crudos (RF04)
│   ├── preparation.py         # feature engineering (RF02, RF03)
│   ├── feature_schema.py       # versionado del esquema de features (RNF12)
│   └── pipeline_manager.py      # orquestador: extracción + preparación
├── app/
│   └── main.py               # API FastAPI (health, trigger, lectura de features)
├── docs/
│   └── data_service_schema.sql # DDL completo de la base de datos
├── .env.example
├── requirements.txt
├── Dockerfile
└── README.md
```

## Configuración

1. Copia `.env.example` a `.env`.
2. Completa `DATABASE_URL` con la cadena de conexión real a tu
   instancia de PostgreSQL/TimescaleDB (host, usuario, password, base
   de datos). El resto de variables tienen valores por defecto
   razonables.
3. Asegúrate de que la base de datos ya tenga el esquema aplicado
   (`docs/data_service_schema.sql`).

`settings.py` es la única fuente de verdad de configuración: ningún
otro módulo lee variables de entorno directamente ni hardcodea rutas o
credenciales (RNF06).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Correr el pipeline completo (extracción + preparación)

```bash
python -m pipeline.pipeline_manager
```

Termina con código de salida 0 si todo salió bien, 1 si alguna etapa
falló (útil para cron/CI).

### Correr solo una etapa

```bash
python -m pipeline.extraction     # solo descarga OHLC hacia raw_ohlc
python -m pipeline.preparation    # solo calcula features hacia features
```

### Levantar la API

```bash
uvicorn app.main:app --reload
```

Endpoints:

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Healthcheck. |
| POST | `/pipeline/run` | Corre extracción + preparación de forma síncrona. |
| GET | `/features/latest?pair_code=SP500_VIX` | Última fila de features del par (modo "Predicción de Mañana", RF15). |
| GET | `/features/history?pair_code=SP500_VIX` | Histórico con target ya conocido, listo para entrenar ML (caso de uso 3). Acepta `date_from`/`date_to` opcionales. |

### Con Docker

```bash
docker build -t data-service .
docker run --env-file .env -p 8000:8000 data-service
```

## Flujo de ejecución

```
PipelineManager.execute()
  └── run_full_pipeline()
        ├── extraction.DataExtractor.download_all()   (por cada activo del registry)
        │     ├── db.get_latest_raw_date()             → calcula el rango incremental (RF05)
        │     ├── yfinance.download()
        │     ├── validation.DataValidator.validate()   → nulos, duplicados, gaps (RF04)
        │     ├── db.upsert_raw_ohlc()                  → persiste en raw_ohlc
        │     └── db.log_run("extraction", ...)          → auditoría en ingestion_log (RF06)
        └── preparation.DataPreparer.run_pipeline()     (por cada par del registry)
              ├── feature_schema.register_current_version() → RNF12
              ├── db.fetch_raw_ohlc()  x2 (índice + volatilidad)
              ├── feature engineering (RF02/RF03, sin data leakage)
              ├── feature_schema.enforce_schema()
              ├── db.upsert_features()                   → persiste en features
              └── db.log_run("feature_engineering", ...)  → auditoría en ingestion_log (RF06)
```

## Activos y pares soportados

Definidos en `pipeline/registry.py`, fuente de verdad única (RNF11):
agregar un activo o un par nuevo es agregar entradas ahí, no cambiar
lógica en otros módulos. Debe mantenerse sincronizado con los `INSERT`
de `instruments`/`asset_pairs` en `docs/data_service_schema.sql`.

| Par | Índice principal | Índice de volatilidad |
|---|---|---|
| `SP500_VIX` | SP500 (`^GSPC`) | VIX (`^VIX`) |
| `NASDAQ_VXN` | NASDAQ (`^IXIC`) | VXN (`^VXN`) |

## Prevención de data leakage (RF03)

Todas las columnas de features (`main_*`, `vol_idx_*`, `day_of_week`)
se calculan usando exclusivamente datos hasta el cierre del día `t`.
La única excepción es `target_range_next_day`, calculada
deliberadamente con `shift(-1)` porque ES el valor que el modelo debe
aprender a predecir — y por eso queda `NULL` en la fila del día más
reciente, hasta que `raw_ohlc` tenga el dato de `t+1` y una corrida
posterior de `preparation.py` la complete.

## Versionado de esquema (RNF12)

`feature_schema.py` define `SCHEMA_VERSION` y `FEATURE_COLUMNS` como
constantes de código. Cada corrida de `preparation.py` registra esa
versión en la tabla `feature_schema_versions` (marcándola vigente) y
`enforce_schema()` falla rápido si el DataFrame calculado no tiene
exactamente esas columnas — evita persistir datos con un esquema
inconsistente por un cambio silencioso en el feature engineering.

## Logging y auditoría (RF06)

Dos mecanismos complementarios:

- **Logs de texto** (`logging_config.py`): a consola y a
  `logs/data_service/data_pipeline.log`, para lectura humana durante
  desarrollo o revisión de incidentes.
- **`ingestion_log`** (tabla en base de datos, vía `db.log_run`): un
  registro estructurado y consultable por SQL de cada corrida —
  etapa, ticker o par, rango de fechas, filas afectadas, estado,
  mensaje de error y versión del código que corrió
  (`pipeline_version`).

## Decisión de diseño: sin acoplamiento al ML Service

Versiones anteriores de este pipeline decidían si "hacía falta
reentrenar" leyendo el archivo de metadata del modelo de ML antes de
correr. Se eliminó esa lógica: el Data Service no debe leer artefactos
de otro servicio con su propia base de datos (RNF17). Ahora
`PipelineManager` siempre corre cuando se invoca; la periodicidad
(diaria, por ejemplo) se decide fuera de este código — vía cron, un
scheduler, o llamando a `POST /pipeline/run` desde un orquestador
externo.

## Cobertura de requerimientos

| Requerimiento | Dónde se cumple |
|---|---|
| **RF01** — Extraer OHLC de SP500/NASDAQ/VIX/VXN vía Yahoo Finance | `extraction.py` + `registry.py` |
| **RF02** — Calcular y almacenar las features derivadas | `preparation.py` (`_engineer_features`) |
| **RF03** — Evitar data leakage | `preparation.py`: todas las features usan datos ≤ t; solo `target_range_next_day` usa `shift(-1)` y queda NULL hasta conocerse |
| **RF04** — Validar integridad antes de persistir | `validation.py`, invocado en `extraction.py` antes del upsert; reforzado con CHECK constraints en la base de datos |
| **RF05** — Actualización incremental sin re-descargar todo | `extraction.py._resolve_date_range()` + `db.get_latest_raw_date()` |
| **RF06** — Logging de fecha de ejecución, rango obtenido y errores | `logging_config.py` + tabla `ingestion_log` vía `db.log_run()` en ambas etapas |
| **RNF10** — Desacoplar pipeline de la capa web | Paquete `pipeline/` sin dependencias de Django/frameworks web; `app/main.py` es una capa delgada que solo lo invoca |
| **RNF11** — Estructura modular para agregar activos sin tocar otras capas | `registry.py`: agregar un activo/par es agregar una entrada, no cambiar lógica |
| **RNF12** — Versionar el esquema de features | `feature_schema.py` + tabla `feature_schema_versions` |
| **RNF17** — Database per service | Este servicio es dueño exclusivo de su base de datos; ningún otro servicio se conecta directamente a ella (consumen vía la API de `app/main.py`) |

## Validado antes de esta entrega

Este código fue ejecutado de punta a punta contra una base PostgreSQL
real antes de entregarse (no solo escrito): creación del esquema,
upsert de datos crudos con rechazo correcto de OHLC inválido por los
CHECK constraints, corrida completa de `preparation.py` con
verificación de que la fila más reciente conserva `target_range_next_day`
en NULL sin ser descartada, los 5 endpoints de la API, la lógica de
actualización incremental (RF05) y la validación de integridad (RF04).

## Pendiente (fuera del alcance de este entregable)

- Tests automatizados (`pytest`) migrados a este esquema.
- Conexión real a TimescaleDB con las hypertables y la política de
  compresión activas (probado en este entorno solo contra PostgreSQL
  estándar, por restricciones de red del entorno de desarrollo; la
  sintaxis de `docs/data_service_schema.sql` ya fue validada por
  separado).
- Scheduler (cron / APScheduler / CronJob de Kubernetes) que invoque
  `POST /pipeline/run` con la periodicidad deseada.
