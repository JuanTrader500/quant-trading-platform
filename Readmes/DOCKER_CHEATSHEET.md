# Docker Cheat Sheet — sp500_MLops

Comandos de referencia rápida para el día a día del stack
(`data_service`, `ml_service`, `mlflow`, `data-service-db`). Corre
todo desde la raíz del repo (donde está `docker-compose.yml`).

---

## 1. Prender / apagar todo el stack

```bash
# Levantar todo (crea contenedores si no existen, reusa si ya existen)
docker compose up -d

# Apagar todo, sin borrar nada (contenedores se destruyen, volúmenes quedan)
docker compose down

# Apagar TODO y además borrar los volúmenes (⚠️ borra la base de datos,
# el histórico de predicciones y los modelos entrenados en MLflow)
docker compose down -v

# Reiniciar un servicio puntual sin tocar los demás
docker compose restart ml_service
```

---

## 2. Cuando cambias código

Regla simple: **¿tocaste algo dentro de `ml_service/` o `data_service/`?**
→ necesitas `build`, no solo `restart` (`restart` reusa la imagen vieja).

```bash
# Reconstruir SOLO el servicio que cambiaste
docker compose build ml_service
docker compose up -d ml_service

# Ídem para data_service
docker compose build data_service
docker compose up -d data_service

# Atajo: reconstruye y levanta TODO lo que haya cambiado
docker compose up -d --build

# Si sospechas de caché vieja de Docker (cambios que "no se ven")
docker compose build --no-cache ml_service
docker compose up -d ml_service
```

**No hace falta rebuild** si solo cambiaste el `.env` — con esto alcanza:
```bash
docker compose up -d ml_service
```
(recrea el contenedor con las variables nuevas, sin reconstruir la imagen).

---

## 3. Ver que todo esté sano (healthcheck)

```bash
# Vista rápida: nombres, estado, puertos
docker compose ps

# Ídem pero de TODO Docker (útil si sospechas de contenedores viejos duplicados)
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Salud propia de cada servicio vía HTTP
curl http://localhost:8000/health                                    # Data Service
curl "http://localhost:8000/raw/latest?pair_code=SP500_VIX"           # Data Service tiene datos crudos
curl "http://localhost:8000/features/latest?pair_code=SP500_VIX"      # Data Service tiene features
curl http://localhost:8001/health                                    # ML Service (mira "model_loaded")
curl http://localhost:5000/health || echo "revisa http://localhost:5000 en el navegador"  # MLflow

# Ver la UI de MLflow (experimentos, runs, modelo registrado)
xdg-open http://localhost:5000   # o pégalo directo en el navegador
```

---

## 4. Entrenamiento manual del modelo

```bash
# Forma recomendada: por HTTP, actualiza el modelo en memoria de inmediato
curl -X POST http://localhost:8001/admin/retrain -H "X-Service-Key: <tu ML_SERVICE_API_KEY>"

# Confirmar que quedó cargado
curl http://localhost:8001/health          # "model_loaded": true
curl http://localhost:8001/model/metrics   # RMSE, MAE, sesgo, comparación vs GARCH

# Alternativa por CLI (proceso aparte — requiere el reload extra de abajo)
docker compose exec ml_service python -m training.retrain_manager
curl -X POST http://localhost:8001/admin/reload-model -H "X-Service-Key: <tu ML_SERVICE_API_KEY>"

# Backtest histórico batch (CSV date,predict 2022 -> hoy, no actualiza el modelo en memoria)
docker compose exec ml_service python -m training.historical_backtest
```

---

## 5. Predicciones

```bash
# Modo automático (usa la última fila de features del Data Service)
curl -X POST "http://localhost:8001/predict/tomorrow?pair_code=SP500_VIX"

# Modo Testing (OHLC manual)
curl -X POST http://localhost:8001/predict/testing \
  -H "Content-Type: application/json" \
  -d '{"open": 5300, "high": 5340, "low": 5280, "close": 5321, "pair_code": "SP500_VIX"}'

# Historial de predicciones guardadas
curl "http://localhost:8001/predictions/history?pair_code=SP500_VIX&limit=20"
```

---

## 6. Logs

```bash
# En vivo, todos los servicios
docker compose logs -f

# En vivo, solo uno
docker compose logs -f ml_service
docker compose logs -f data_service

# Solo las últimas N líneas, sin quedarte viendo
docker compose logs --tail=100 ml_service

# Filtrar por palabra clave
docker compose logs ml_service | grep -i "retrain\|error"

# Archivo persistente en tu host (no depende de Docker)
tail -f ./logs/ml_service/ml_service.log
tail -f ./logs/data_service/data_pipeline.log
```

---

## 7. Diagnóstico rápido cuando algo falla

```bash
# Ver variables de entorno reales dentro de un contenedor
docker compose exec ml_service printenv ML_SERVICE_API_KEY
docker compose exec ml_service printenv DATA_SERVICE_BASE_URL

# Entrar a una shell dentro del contenedor
docker compose exec ml_service bash
docker compose exec data_service bash

# Ver qué proceso tiene ocupado un puerto en tu máquina (fuera de Docker)
sudo ss -ltnp | grep :8001

# Ver contenedores duplicados o huérfanos de otros compose viejos
docker ps -a --filter "name=data-service-db"
docker ps -a --filter "name=mlflow-server"
```

---

## 8. Referencia de puertos (en esta máquina)

| Servicio | Puerto host | URL |
|---|---|---|
| Data Service | 8000 | http://localhost:8000 |
| Data Service DB (Postgres) | 5433 | (no HTTP, solo psql/clientes DB) |
| ML Service | 8001 | http://localhost:8001 |
| MLflow | 5000 (o el que hayas cambiado) | http://localhost:5000 |

---

## 9. El secreto de un buen día

```bash
docker compose ps            # todo "healthy"?
docker compose logs -f       # algo gritando en rojo?
```

Si esos dos se ven bien, probablemente todo lo demás también.
