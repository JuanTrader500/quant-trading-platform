# SP500 / NASDAQ MLOps — Data Service

Documentación consolidada de todo lo construido para el **Data Service**:
infraestructura con Docker, base de datos, el código del pipeline y la
API que expone. Este documento es el punto de entrada; los documentos
más profundos de cada tema (SQL completo, guía de Docker paso a paso)
se referencian donde corresponde.

## Índice

1. [Resumen y alcance](#1-resumen-y-alcance)
2. [Arquitectura general del sistema](#2-arquitectura-general-del-sistema)
3. [Base de datos](#3-base-de-datos)
4. [Infraestructura con Docker](#4-infraestructura-con-docker)
5. [El paquete `pipeline`](#5-el-paquete-pipeline)
6. [Documentación de la API](#6-documentación-de-la-api)
7. [Cómo levantar todo desde cero](#7-cómo-levantar-todo-desde-cero)
8. [Notebook de entrenamiento](#8-notebook-de-entrenamiento)
9. [Cobertura de requerimientos](#9-cobertura-de-requerimientos)
10. [Pendiente / próximos pasos](#10-pendiente--próximos-pasos)

---

## 1. Resumen y alcance

El **Data Service** es uno de los 3 microservicios de la plataforma
(junto a **ML Service** y **Web Service**). Es responsable de todo el
**Módulo 1 (Data Engineering)** de los requerimientos: extraer datos
de mercado, validarlos, calcular features y exponerlos por API para
que el ML Service los consuma — sin que ningún otro servicio toque su
base de datos directamente (RNF17).

Todo lo documentado aquí fue **probado de punta a punta** contra una
base de datos PostgreSQL real y contra la API real corriendo (no solo
escrito): creación del esquema, upserts con rechazo de datos
inválidos, cálculo de features preservando la fila más reciente sin
target conocido, los 4 endpoints de la API, y un notebook completo de
56 celdas ejecutado sin errores contra la API.

---

## 2. Arquitectura general del sistema

```
                    ┌──────────────────┐
   Usuario ────────▶│   Web Service    │  (Django, único punto público)
                    │  Auth · Tienda · │
                    │  Formularios     │
                    └────────┬─────────┘
                             │ HTTP interno
                             ▼
                    ┌──────────────────┐        ┌───────────────────┐
                    │    ML Service    │───────▶│   Model Registry    │
                    │  (FastAPI)       │  HTTP   │   (MLflow)           │
                    │  Entrena/predice │◀────────┘   Versiona modelos   │
                    └────────┬─────────┘
                             │ HTTP interno (GET /features/*)
                             ▼
                    ┌──────────────────┐
                    │   Data Service    │  ◀── este documento
                    │  (FastAPI)         │
                    │  Ingesta + features │
                    └────────┬───────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  PostgreSQL /       │
                    │  TimescaleDB          │  (dueño exclusivo: Data Service)
                    └──────────────────┘
```

Ningún servicio se conecta a la base de datos de otro. La única forma
de pedir datos entre servicios es HTTP, sin Gateway ni bus de eventos
(decisión tomada por simplicidad para el tamaño de este proyecto — ver
conversaciones anteriores de arquitectura).

---

## 3. Base de datos

**Motor:** PostgreSQL 16 + TimescaleDB (corriendo en Docker — ver
sección 4). Se generalizó el esquema desde el inicio para soportar
múltiples pares índice + índice de volatilidad (no solo SP500+VIX,
también NASDAQ+VXN), evitando duplicar tablas por activo.

### Tablas

| Tabla | Qué guarda |
|---|---|
| `instruments` | Catálogo de tickers soportados (`^GSPC`, `^VIX`, `^IXIC`, `^VXN`), con su clase de activo. |
| `asset_pairs` | Qué índice va con qué índice de volatilidad (`SP500_VIX`, `NASDAQ_VXN`). |
| `raw_ohlc` | Velas diarias OHLCV crudas, tal como las devuelve Yahoo Finance. Hypertable particionada por `date`. |
| `features` | Variables derivadas por par (`main_*` del índice principal, `vol_idx_*` de su volatilidad asociada) + `target_range_next_day`. Hypertable particionada por `date`. |
| `feature_schema_versions` | Historial de versiones del esquema de features (RNF12), con la vigente marcada (`is_current`). |
| `ingestion_log` | Auditoría de cada corrida del pipeline: etapa, ticker/par, rango de fechas, filas afectadas, estado, error, versión de código (RF06). |

Dos vistas de solo lectura, pensadas para que el ML Service nunca
tenga que conocer el detalle interno del esquema:

- **`v_latest_features`** — última fila de features por par (modo
  "Predicción de mañana").
- **`v_training_dataset`** — filas con `target_range_next_day` ya
  conocido, listas para entrenar.

### Decisiones de diseño relevantes

- **`raw_ohlc` y `features` son tablas separadas** — así se puede
  recalcular todas las features desde cero (nuevo `schema_version`)
  sin volver a llamar a Yahoo Finance.
- **`target_range_next_day` se llena un día después**, en una corrida
  posterior, nunca en la misma fila que la calcula — hace físicamente
  imposible que el pipeline "vea el futuro" por accidente (RF03).
- **CHECK constraints a nivel de base de datos** (`high >= low`, OHLC
  dentro de rango) como segunda capa de defensa, independiente de la
  validación en Python (RF04/RF20).
- **`raw_ohlc → features` no es una FK real** — es una dependencia de
  datos (agregación de ventanas móviles de dos tickers), documentada y
  protegida con tests, no con `REFERENCES`.

El DDL completo está en `data_service/docs/data_service_schema.sql`,
con seed de catálogo incluido. Se ejecuta automáticamente al levantar
el contenedor de la base de datos (ver sección 4).

---

## 4. Infraestructura con Docker

**Por qué Docker en vez de instalar TimescaleDB nativo:** TimescaleDB
no publica paquetes oficiales para Fedora, y versiones muy nuevas de
PostgreSQL (18) suelen ir por delante de lo que la extensión soporta.
Corriendo la base de datos en un contenedor (`timescale/timescaledb:latest-pg16`),
el sistema operativo host es irrelevante — el contenedor trae su
propio PostgreSQL 16 + la extensión ya compilada.

### Estructura

```
sp500_mlops/
├── docker-compose.yml
├── .env                    ← credenciales, no se sube al repo
├── data_service/
│   ├── docs/data_service_schema.sql
│   ├── pipeline/
│   ├── app/
│   └── Dockerfile
└── logs/data_service/       ← montado desde el contenedor al host
```

### Servicios del `docker-compose.yml`

| Servicio | Imagen / build | Rol |
|---|---|---|
| `timescaledb` | `timescale/timescaledb:latest-pg16` | Base de datos. Volumen con nombre (`timescale-data`) para persistencia. Aplica `data_service_schema.sql` automáticamente la primera vez (vía `/docker-entrypoint-initdb.d`). |
| `data_service` | build desde `./data_service/Dockerfile` | API FastAPI del pipeline. Espera a que la base esté healthy antes de arrancar. |

### Persistencia de datos

Con un **volumen con nombre** (`timescale-data`), los datos sobreviven
a `stop`, `start`, `restart` y `down` (sin `-v`). Solo se pierden con
`docker compose down -v` o `docker volume rm` — comandos explícitos,
nunca accidentales por un simple apagado.

### Comandos esenciales

```bash
# Levantar todo (primera vez o tras cambiar código)
docker compose up -d --build

# Ver estado
docker compose ps

# Apagar sin perder datos
docker compose stop

# Reinicio completo (⚠️ borra los datos, útil para pruebas)
docker compose down -v && docker compose up -d --build

# Backup manual
docker compose exec timescaledb pg_dump -U data_service_app data_service_db > backup_$(date +%F).sql
```

La guía completa, paso a paso desde instalar Docker en Fedora hasta
hacer queries, está en **`GUIA_DOCKER_PASO_A_PASO.md`** (documento
aparte, con troubleshooting de los errores más comunes: puertos
ocupados, permisos, confusión `localhost` vs. nombre del servicio).

---

## 5. El paquete `pipeline`

```
data_service/pipeline/
├── registry.py         # catálogo único de activos y pares (fuente de verdad)
├── settings.py         # configuración leída de .env
├── logging_config.py   # logger compartido (consola + archivo)
├── db.py                # capa de acceso a datos (SQLAlchemy + SQL crudo)
├── extraction.py         # descarga OHLC vía yfinance, incremental (RF01, RF05)
├── validation.py          # integridad de datos crudos (RF04)
├── preparation.py          # feature engineering, sin data leakage (RF02, RF03)
├── feature_schema.py        # versionado del esquema (RNF12)
└── pipeline_manager.py       # orquesta extracción + preparación
```

### Flujo de ejecución

```
PipelineManager.execute()
  └── run_full_pipeline()
        ├── DataExtractor.download_all()        (por cada activo del registry)
        │     ├── db.get_latest_raw_date()        → rango incremental (RF05)
        │     ├── yfinance.download()
        │     ├── DataValidator.validate()          → nulos, duplicados, gaps (RF04)
        │     ├── db.upsert_raw_ohlc()                → persiste en raw_ohlc
        │     └── db.log_run("extraction", ...)         → auditoría (RF06)
        └── DataPreparer.run_pipeline()          (por cada par del registry)
              ├── feature_schema.register_current_version() → RNF12
              ├── db.fetch_raw_ohlc()  ×2 (índice + volatilidad)
              ├── feature engineering (RF02/RF03)
              ├── feature_schema.enforce_schema()
              ├── db.upsert_features()                   → persiste en features
              └── db.log_run("feature_engineering", ...)   → auditoría (RF06)
```

### Activos y pares (`registry.py`)

| Par | Índice principal | Índice de volatilidad |
|---|---|---|
| `SP500_VIX` | SP500 (`^GSPC`) | VIX (`^VIX`) |
| `NASDAQ_VXN` | NASDAQ (`^IXIC`) | VXN (`^VXN`) |

Agregar un activo o par nuevo es agregar entradas en `registry.py` (y
su seed correspondiente en el SQL) — ningún otro módulo cambia
(RNF11).

### Prevención de data leakage (RF03)

Todas las columnas de features (`main_*`, `vol_idx_*`, `day_of_week`)
usan exclusivamente datos hasta el cierre del día `t`. La única
excepción es `target_range_next_day` (`shift(-1)` deliberado), que
queda `NULL` en la fila más reciente hasta que el día siguiente ya
tiene dato — verificado en pruebas reales (ver sección 1).

---

## 6. Documentación de la API

Base URL en desarrollo: `http://localhost:8000` (host) o
`http://data_service:8000` (desde otro contenedor de la misma red de
Docker). Este servicio **no debe exponerse a internet**; solo lo
consumen el ML Service y el Web Service dentro de la red interna.

Documentación interactiva autogenerada (Swagger UI) disponible en
`http://localhost:8000/docs` mientras el servicio está corriendo.

### `GET /health`

Healthcheck. Sin parámetros.

**Respuesta 200:**
```json
{"status": "ok"}
```

---

### `POST /pipeline/run`

Corre extracción + preparación de forma síncrona (bloquea hasta
terminar). Pensado para ser llamado por un scheduler externo (cron,
CronJob de Kubernetes) o manualmente.

**Respuesta 200:**
```json
{"status": "completed"}
```

**Respuesta 500** (alguna etapa falló — ver `logs/data_service/data_pipeline.log`
y la tabla `ingestion_log` para el detalle):
```json
{"detail": "El pipeline falló. Ver logs/data_service/data_pipeline.log y la tabla ingestion_log."}
```

**Ejemplo:**
```bash
curl -X POST http://localhost:8000/pipeline/run
```

---

### `GET /features/latest`

Última fila de features calculada para un par. Usada por el modo
"Predicción de Mañana" del ML Service (RF15).

**Parámetros de query:**

| Parámetro | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `pair_code` | string | Sí | `SP500_VIX` o `NASDAQ_VXN` |

**Respuesta 200:**
```json
{
  "pair_code": "SP500_VIX",
  "date": "2026-07-22",
  "schema_version": "v1",
  "main_log_return": 0.00284,
  "main_log_range": 0.01823,
  "main_body_log": -0.00415,
  "main_upper_wick_log": 0.00612,
  "main_lower_wick_log": 0.00721,
  "main_vol_5d": 0.00891,
  "main_vol_10d": 0.00764,
  "vol_idx_log_close": 2.7912,
  "vol_idx_log_range": 0.0532,
  "vol_idx_log_return": -0.0187,
  "day_of_week": 2,
  "target_range_next_day": null,
  "computed_at": "2026-07-22T21:03:11.204Z",
  "updated_at": "2026-07-22T21:03:11.204Z"
}
```
`target_range_next_day` es `null` en la fila más reciente hasta que el
día siguiente ya tenga dato en `raw_ohlc` (comportamiento esperado,
no un error).

**Respuesta 404** — `pair_code` no existe:
```json
{"detail": "pair_code desconocido: XXXX"}
```

**Respuesta 404** — el par existe pero no hay features calculadas aún:
```json
{"detail": "Sin features calculadas todavía para este par."}
```

**Ejemplo:**
```bash
curl "http://localhost:8000/features/latest?pair_code=SP500_VIX"
```

---

### `GET /features/history`

Histórico de features con `target_range_next_day` ya conocido — listo
para entrenar (Walk-Forward Validation). Es el endpoint que consume el
notebook/ML Service para el caso de uso "consultar la data para
entrenar modelos de ML".

**Parámetros de query:**

| Parámetro | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `pair_code` | string | Sí | `SP500_VIX` o `NASDAQ_VXN` |
| `date_from` | date (`YYYY-MM-DD`) | No | Filtro inclusive, desde esta fecha |
| `date_to` | date (`YYYY-MM-DD`) | No | Filtro inclusive, hasta esta fecha |

**Respuesta 200** — lista de objetos, mismo formato que `/features/latest`
pero sin filas con `target_range_next_day` nulo:
```json
[
  {
    "pair_code": "SP500_VIX",
    "date": "2021-01-15",
    "schema_version": "v1",
    "main_log_return": 0.000284,
    "...": "...",
    "target_range_next_day": 0.01245,
    "computed_at": "2026-07-20T10:00:00Z",
    "updated_at": "2026-07-20T10:00:00Z"
  }
]
```

**Respuesta 404** — `pair_code` no existe:
```json
{"detail": "pair_code desconocido: XXXX"}
```

**Ejemplos:**
```bash
# histórico completo del par
curl "http://localhost:8000/features/history?pair_code=SP500_VIX"

# acotado por fechas
curl "http://localhost:8000/features/history?pair_code=SP500_VIX&date_from=2023-01-01&date_to=2023-12-31"
```

### Resumen de endpoints

| Método | Ruta | Uso |
|---|---|---|
| GET | `/health` | Healthcheck |
| POST | `/pipeline/run` | Disparar extracción + preparación |
| GET | `/features/latest?pair_code=...` | Última fila (predicción diaria) |
| GET | `/features/history?pair_code=...&date_from=...&date_to=...` | Histórico para entrenamiento |

---

## 7. Cómo levantar todo desde cero

```bash
# 1. Clonar/descomprimir el proyecto
cd ~/sp500_mlops

# 2. Configurar variables de entorno
cp .env.example .env
nano .env   # cambiar POSTGRES_PASSWORD como mínimo

# 3. Levantar base de datos + API
docker compose up -d --build

# 4. Verificar que está sano
docker compose ps
curl http://localhost:8000/health

# 5. Correr el pipeline por primera vez (descarga histórico completo)
curl -X POST http://localhost:8000/pipeline/run

# 6. Confirmar que hay datos
curl "http://localhost:8000/features/latest?pair_code=SP500_VIX"
```

Detalle exhaustivo de cada paso (incluyendo instalar Docker en Fedora
desde cero) en `GUIA_DOCKER_PASO_A_PASO.md`.

---

## 8. Notebook de entrenamiento

`01_exploracion_seleccion_modelo_adaptado.ipynb` (fuera de este
repositorio de servicio, vive en el espacio de trabajo del ML Service)
pide los datos exactamente como lo hará el ML Service en producción:
por HTTP contra `GET /features/history`, no importando código del Data
Service ni leyendo CSV. Esto es intencional — los microservicios no
comparten código (RNF17), así que el "contrato" entre ambos es la API,
y el notebook valida ese contrato en su primera celda:

```python
EXPECTED_FEATURE_COLUMNS = [
    "main_log_return", "main_log_range", "main_body_log",
    "main_upper_wick_log", "main_lower_wick_log",
    "main_vol_5d", "main_vol_10d",
    "vol_idx_log_close", "vol_idx_log_range", "vol_idx_log_return",
    "day_of_week", "target_range_next_day",
]
```

Si el Data Service cambiara de esquema sin avisar, esta celda falla de
inmediato en vez de dejar entrenar un modelo con columnas
inconsistentes.

---

## 9. Cobertura de requerimientos

| Requerimiento | Dónde se cumple |
|---|---|
| **RF01** — Extraer OHLC de SP500/NASDAQ/VIX/VXN | `extraction.py` + `registry.py` |
| **RF02** — Calcular y almacenar features derivadas | `preparation.py` |
| **RF03** — Evitar data leakage | Features usan datos ≤ t; `target_range_next_day` es NULL hasta conocerse |
| **RF04** — Validar integridad antes de persistir | `validation.py` + CHECK constraints en BD |
| **RF05** — Actualización incremental | `db.get_latest_raw_date()` + `extraction._resolve_date_range()` |
| **RF06** — Logging de cada corrida | `logging_config.py` + tabla `ingestion_log` |
| **RNF06** — Credenciales por variable de entorno | `.env` / `.env.example`, nunca hardcodeado |
| **RNF10** — Pipeline desacoplado de la web | Paquete `pipeline/` sin dependencias de Django; `app/main.py` es una capa delgada |
| **RNF11** — Estructura modular | `registry.py`: agregar activos/pares no cambia lógica existente |
| **RNF12** — Versionar esquema de features | `feature_schema.py` + tabla `feature_schema_versions` |
| **RNF15** — Contenedorización | `Dockerfile` del Data Service |
| **RNF17** — Database per service | Única base de datos, dueño exclusivo el Data Service; otros servicios consumen vía API |

---

## 10. Pendiente / próximos pasos

- **ML Service**: envolver el notebook validado en un servicio FastAPI
  con endpoints `/predict`, `/retrain`, `/metrics`, e integrar MLflow
  como Model Registry (ver conversación de diseño de MLflow).
- **Web Service**: proyecto Django (auth, tienda de algoritmos, pagos)
  aún no iniciado.
- **Tests automatizados** (`pytest`) del paquete `pipeline`, migrados
  al esquema actual.
- **Scheduler** que llame `POST /pipeline/run` con periodicidad diaria
  (cron, APScheduler, o CronJob de Kubernetes) — hoy se dispara
  manualmente.
- **TimescaleDB en producción real**: lo probado en este entorno de
  desarrollo fue contra PostgreSQL estándar (por restricciones de red
  del sandbox); la sintaxis específica de TimescaleDB (hypertables,
  compresión) está en el script pero debe confirmarse en el primer
  despliegue real con Docker.
