# ML Service

FastAPI, stateless (RNF17), entrenamiento/reentrenamiento/predicción del
rango de volatilidad del SP500 (Módulos 2, 3 y 9 de los requerimientos).
Consume al **Data Service** por HTTP (RF39) y no debe exponerse a
internet (RNF20) — solo lo consume el Web Service.

## 1. Qué trae este entregable

```
ml_service/
├── app/                 API FastAPI (endpoints, seguridad, esquemas, estado en memoria)
├── clients/             Cliente HTTP hacia el Data Service (timeouts + reintentos, RF42)
├── core/                Settings + logging estructurado con trace_id (RNF19)
├── features/            Validación OHLC (RF20) + feature engineering modo Testing (RF14)
├── models/               Modelo confirmado (Gradient Boosting), GARCH baseline (RF10), métricas (RF17/RF18)
├── predictions/          SQLite de historial de predicciones + logging espejo en MLflow
├── registry/             Wrapper de MLflow como Model Registry (RF43, RF44)
├── scheduler/            APScheduler: reentrenamiento el día 1 de cada mes (RF08)
├── training/             Walk-Forward Validation, orquestador de reentrenamiento, backtest histórico
├── tests/                 Pytest (no dependen de servicios externos: usan mocks/datos sintéticos)
├── db/predictions_schema.sql
├── Dockerfile
├── requirements.txt / requirements-test.txt
├── .env.example
└── docker-compose.ml_service.snippet.yml   <- fusionar a mano en tu docker-compose.yml
```

## 2. Cómo integrarlo a tu proyecto

1. Descomprime este zip como carpeta `ml_service/` en la raíz de tu repo
   (al lado de `data_service/`).
2. Abre `docker-compose.ml_service.snippet.yml` y copia los servicios
   `mlflow` y `ml_service`, y los volúmenes `mlflow-data` /
   `ml-service-data`, dentro de tu `docker-compose.yml` real (junto a
   `timescaledb` y `data_service`). No se hizo automáticamente porque
   pediste revisarlo todo a mano.
3. Copia las variables de `ml_service/.env.example` a tu `.env` de la
   raíz (el mismo que ya usa `data_service`), y define además
   `INTERNAL_SERVICE_KEY` y `ML_SERVICE_API_KEY` (RF41) — deben ser
   iguales a las que uses también en el Data Service cuando le agregues
   la verificación (ver limitación #1 abajo).
4. `docker compose up -d --build`.
5. Primer arranque: el ML Service no tendrá modelo todavía (no ha
   corrido ningún reentrenamiento). Dispáralo manualmente una vez:
   ```bash
   curl -X POST http://localhost:8001/admin/retrain -H "X-Service-Key: <ML_SERVICE_API_KEY>"
   ```
   A partir de ahí, el scheduler interno lo repite automáticamente el
   día 1 de cada mes (configurable con `RETRAIN_SCHEDULE_*`).

## 3. Endpoints

| Método | Ruta | Requerimiento |
|---|---|---|
| GET | `/health` | healthcheck propio + del Data Service |
| POST | `/predict/testing` | RF14/RF20/RF21 — OHLC manual |
| POST | `/predict/tomorrow` | RF15 — modo automático |
| GET | `/model/metrics` | RF17/RF18/RF19 — RMSE, MAE, sesgo, comparación vs GARCH |
| GET | `/predictions/history` | RF24 — historial consultable (filtros `pair_code`, `mode`, fechas) |
| GET | `/predictions/export/historical` | descarga del CSV `date,predict` 2022→hoy |
| POST | `/admin/reload-model` 🔒 | RF44 — recarga manual del modelo vigente |
| POST | `/admin/retrain` 🔒 | dispara un reentrenamiento fuera de cadencia |

🔒 = requiere header `X-Service-Key: <ML_SERVICE_API_KEY>`.

Swagger/OpenAPI (RF37): `http://localhost:8001/docs`.

## 4. El CSV histórico 2022 → actualidad que pediste

**No se genera solo.** Es una corrida batch (no vive en el flujo de
predicción en vivo). Corre, dentro del contenedor ya levantado:

```bash
docker compose exec ml_service python -m training.historical_backtest --start 2022-01-01
```

Esto:

1. Pide al Data Service **todo** el histórico de features vía
   `GET /features/history` (incluye datos anteriores a 2022, necesarios
   para entrenar el primer mes del backtest).
2. Elige un algoritmo **una sola vez** con Walk-Forward Validation
   sobre los datos previos a 2022-01-01 (ver limitación #2 abajo).
3. Para cada mes calendario desde 2022-01 hasta hoy: entrena solo con
   datos **estrictamente anteriores** a ese mes, y predice cada día de
   ese mes con ese modelo "congelado" — replicando la cadencia mensual
   real de producción (RF08), sin usar nunca un dato del mes que
   predice ni de meses futuros.
4. Escribe `date,predict` en
   `/app/data/exports/predictions_SP500_VIX_2022-01-01_<hoy>.csv`
   dentro del volumen `ml-service-data`.

Descárgalo con:
```bash
curl -O -J "http://localhost:8001/predictions/export/historical?filename=predictions_SP500_VIX_2022-01-01_<hoy>.csv"
```
o directamente del volumen Docker (`docker cp`, o copiándolo a tu host
por el mismo volumen).

Las predicciones día a día del modo automático en producción se
acumulan aparte, en la tabla `predictions` del SQLite
(`/app/data/predictions.db`) desde que el sistema arranque en
producción — nunca podrán tener datos de 2022, por eso ese rango se
resuelve con el backtest batch de arriba y no con el log en vivo.

## 5. Dónde quedan las predicciones (tu pregunta sobre almacenamiento)

- **SQLite propio** (`/app/data/predictions.db`, volumen Docker
  `ml-service-data`): fuente de verdad del historial día a día,
  consultable vía `GET /predictions/history` para tus gráficas o
  comparaciones futuras.
- **MLflow**: además de trackear cada reentrenamiento (hiperparámetros
  del modelo confirmado, métricas de validación + GARCH, versión
  promovida), cada
  predicción diaria del modo automático también se loguea como métrica
  dentro de un run "production_monitoring" de larga duración, para que
  puedas verla y graficarla en la UI de MLflow (`http://localhost:5000`)
  junto a las métricas de reentrenamiento.
- **CSV histórico 2022→hoy**: archivo aparte, generado por el batch de
  la sección 4.

## 6. Limitaciones / decisiones de diseño que debes revisar

1. **API key entre servicios (RF41):** el `main.py` del Data Service
   que revisé **no valida todavía** ninguna clave de servicio. El ML
   Service ya envía `X-Service-Key` en cada llamada saliente, lista
   para cuando agregues esa verificación al Data Service — pero hoy no
   hace nada del otro lado.
2. **Modo Testing (RF14) y datos que el Data Service no expone:** el
   Data Service solo expone features ya calculadas
   (`/features/latest`, `/features/history`), nunca el OHLC crudo
   histórico. Por eso, en modo Testing, `main_log_return`,
   `main_vol_5d`, `main_vol_10d` y las columnas `vol_idx_*` (VIX, que
   la persona usuaria no ingresa manualmente) se completan con el
   último valor real conocido en vez de recalcularse para el día
   hipotético ingresado. Está documentado en
   `features/feature_engineering.py`. Si quieres precisión exacta ahí,
   se necesitaría un endpoint adicional en el Data Service que exponga
   el cierre crudo del día anterior.
3. **`pair_code` fijo en `SP500_VIX`** por defecto (configurable vía
   `DATA_SERVICE_PAIR_CODE`), porque RF01-RF20 solo hablan de SP500/VIX.
4. **Modelo de producción confirmado:** Gradient Boosting con
   hiperparámetros fijos (`learning_rate=0.05, max_depth=2,
   min_samples_leaf=5, n_estimators=200, subsample=0.9,
   random_state=42`), definidos en `models/algorithms.py`. Ya no se
   comparan Ridge/Lasso/ElasticNet — el walk-forward y el backtest
   histórico solo entrenan/validan este modelo (más GARCH como
   baseline de comparación, RF19).
5. **Puerto 8001 mapeado al host** en el snippet de docker-compose:
   está ahí solo para que puedas probar con curl/Swagger durante tu
   revisión manual. RNF20 dice que el ML Service no debe exponerse a
   internet — quita ese mapeo de puertos cuando despliegues de verdad.

## 7. Tests

```bash
pip install -r requirements-test.txt
PYTHONPATH=. pytest
```

Todos los tests corren con mocks/datos sintéticos, sin necesitar el
Data Service, MLflow ni Docker levantados.
