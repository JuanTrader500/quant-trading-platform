# Retraining Pipeline - MLOps
Este módulo implementa el sistema de reentrenamiento automático para el modelo de predicción de volatilidad del S&P 500, cumpliendo con los requerimientos RF08, RF11, RF17, RF18 y RF19.

## Contexto Lógico
En series temporales financieras, el rendimiento de un modelo degrada con el tiempo debido al cambio de régimen de volatilidad. Para mitigar esto, el pipeline automatiza el ciclo de vida del modelo:
1. **Sincronización de Datos**: Ejecuta el `DataPipeline` para obtener los datos más recientes.
2. **Entrenamiento**: Ajusta un RandomForestRegressor (Modelo ML) y un GARCH(1,1) (Baseline).
3. **Validación Walk-Forward**: Evalúa la capacidad predictiva simulando el avance del tiempo, evitando el *data leakage* al ajustar los escaladores solo en la ventana de entrenamiento.
4. **Serialización**: Guarda los artefactos en rutas fechadas para permitir trazabilidad y rollbacks.

## Componentes del Código
- `retrain_manager.py`: Punto de entrada. Coordina la ejecución de los scripts de datos, el entrenamiento de modelos y la persistencia de artefactos.
- `validation_utils.py`: Contiene la lógica matemática para el cálculo de RMSE, MAE y la métrica de sesgo direccional (subestimación vs sobreestimación).

## Flujo de Ejecución
`pipeline_manager.py` $\rightarrow$ `RandomForest` + `GARCH` $\rightarrow$ `Walk-Forward Validation` $\rightarrow$ `Artifacts Storage`
