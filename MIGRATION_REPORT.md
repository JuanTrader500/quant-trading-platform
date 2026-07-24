# MIGRATION REPORT: Monolith to 3-Service Architecture

## 1. Mapping Table (Absolute Paths)

| Origin Path | Destination Path |
| :--- | :--- |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/src/DataPipeline/` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/data_service/pipeline/` |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/src/data/` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/data_service/pipeline/data/` |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/tests/data_pipeline/` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/data_service/tests/` |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/src/RetrainingPipeline/` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/ml_service/training/` |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/models/artifacts/` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/ml_service/artifacts/` |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/tests/model_pipeline/` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/ml_service/tests/` |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/logs/data_pipeline.log` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/logs/data_service/data_pipeline.log` |
| `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/logs/retraining.log` | `/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops/logs/ml_service/retraining.log` |

## 2. Updated Imports & Paths

| File | Before | After |
| :--- | :--- | :--- |
| `data_service/pipeline/settings.py` | `src/data/` | `data_service/pipeline/data/` |
| `data_service/pipeline/settings.py` | `models/artifacts/` | `ml_service/artifacts/` |
| `data_service/pipeline/settings.py` | `logs/` | `logs/data_service/` |
| `ml_service/training/retrain_manager.py` | `src/data/processed/...` | `data_service/pipeline/data/processed/...` |
| `ml_service/training/retrain_manager.py` | `src/DataPipeline/pipeline_manager.py` | `data_service/pipeline/pipeline_manager.py` |
| `ml_service/training/retrain_manager.py` | `models/artifacts/` | `ml_service/artifacts/` |
| `ml_service/training/retrain_manager.py` | `logs/retraining.log` | `logs/ml_service/retraining.log` |
| `ml_service/training/retrain_manager.py` | `subprocess ... -m DataPipeline.pipeline_manager` | `subprocess ... -m data_service.pipeline.pipeline_manager` |
| `ml_service/training/retrain_manager.py` | `cwd=ROOT_DIR / "src"` | `cwd=ROOT_DIR / "data_service"` |

## 3. Pending TODOs

### data_service
- [ ] Implement endpoints `/features/latest` and `/features/history` in `app/main.py`.
- [ ] Migrate local CSV data to a database.
- [ ] Refine `Dockerfile` for production.

### ml_service
- [ ] Implement endpoints `/predict`, `/retrain`, and `/metrics` in `app/main.py`.
- [ ] Implement `registry/` logic (Model Registry client).
- [ ] Migrate local `.pkl` artifacts to a Model Registry.
- [ ] Refine `Dockerfile` for production.

### web_service
- [ ] Initialize Django project.
- [ ] Implement Auth, Forms, Algorithm Store, and Payments.
- [ ] Connect to `ml_service` via HTTP.

## 4. Unmoved Files
No files were left behind. All business logic was successfully migrated to their respective services.

## 5. Git Commands Executed
- `git checkout -b migracion/arquitectura-3-servicios`
- `git mv` (various files and folders)
- `git add .`
- `git commit -m "Fase 1 & 2: Reorganización de estructura a microservicios y actualización de rutas básicas"`
- `git commit -m "Fase 3: Creación de esqueletos FastAPI, requirements, Dockerfiles y docker-compose"`
- `git commit -m "Fase 4: Actualización de CI workflows para los nuevos servicios"`
