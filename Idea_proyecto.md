# Roadmap del Proyecto: Multi-Model Quant Trading System (S&P 500 & VIX)

Este proyecto consiste en el desarrollo de un sistema de trading cuantitativo híbrido. Utiliza un **Modelo de Regresión** para predecir los límites del rango diario (High/Low) y un **Modelo de Clasificación** para predecir la dirección del mercado (Alcista/Bajista). Ambas predicciones se combinan para alimentar la lógica de un Expert Advisor (EA).

---

## 📌 1. Plan de Trabajo (Fases de Desarrollo)

### Fase 1: Arquitectura de Datos y Blindaje (Anti-Data Leakage)
- [ ] **Configuración del Index:** Asegurar que el dataset (`2005 - 2026`) use `Date` como un `DatetimeIndex` estricto en Pandas.
- [ ] **División Temporal Inicial (Split Primario):**
  - **Train Set:** 2005 al 31 de diciembre de 2019 (Diseño, EDA y entrenamiento inicial).
  - **Walk-Forward Validation Set:** 2020 al 2026 (Bloqueado en "caja fuerte" hasta la fase de simulación).
- [ ] **Alineación de Variables (Lagging):** Verificar matemáticamente que las variables `_lag1` correspondan al día $t-1$ y que el Target corresponda al día $t+1$ con respecto a la información disponible al cierre de hoy ($t$).

### Fase 2: Modelo de Regresión (Predicción de Rangos)
- [ ] **EDA del Set de Entrenamiento (2005-2019):** Analizar correlaciones de volatilidad entre el VIX y las variables logarítmicas de los rangos.
- [ ] **Feature Engineering:** Crear funciones de transformación (logaritmos, rolling volatilities) que solo utilicen datos del pasado del registro actual.
- [ ] **Pipeline de Regresión:** Crear un `Pipeline` en Scikit-Learn que incluya el escalador (`StandardScaler`) y el modelo (ej. Random Forest Regressor o XGBoost Regressor).
- [ ] **Métricas de Evaluación:** Medir rendimiento usando MAE (Mean Absolute Error) y RMSE enfocados en la precisión de los límites.

### Fase 3: Modelo de Clasificación (Dirección del Mercado)
- [ ] **Definición del Target Binario:** Crear la etiqueta `Target_Direccion` ($1$ si sube, $0$ si baja) probando dos enfoques en el Train Set:
  - Cierre de mañana vs Cierre de hoy.
  - Cierre de mañana vs Apertura de mañana (evitando distorsión por Gaps).
- [ ] **Pipeline de Clasificación:** Configurar un pipeline gemelo al de regresión pero optimizado para clasificación (ej. Support Vector Classifier o LightGBM).
- [ ] **Métricas de Evaluación:** Ignorar el Accuracy simple. Evaluar con **Precision** (minimizar falsos positivos), **Recall** y el área bajo la curva (ROC-AUC).

### Fase 4: Pipeline de Validación Walk-Forward (2020 - 2026)
- [ ] **Implementación del Backtest Temporal:** Diseñar el bucle de simulación usando una **Ventana Rodante (Rolling Window)** de 15 años.
- [ ] **Simulación de Reentrenamiento:** Configurar el script para que avance mes a mes o trimestre a trimestre reentrenando los modelos con la nueva data del pasado acumulada.
- [ ] **Registro de Predicciones:** Guardar en un CSV histórico todas las predicciones generadas en el periodo 2020-2026 para el análisis final.

### Fase 5: Diseño de la Lógica del EA y Gestión de Riesgo
- [ ] **Matriz de Decisión Híbrida (Stacking):**
  - Si Clasificador dice *Alcista* -> Buscar solo Compras.
  - Si Clasificador dice *Bajista* -> Buscar solo Ventas.
- [ ] **Cálculo Dinámico de Niveles:** Usar las salidas del regresor (`target_high` y `target_low`) para colocar los niveles automáticos de Stop Loss y Take Profit.
- [ ] **Esperanza Matemática:** Calcular el ratio Riesgo:Beneficio promedio y la tasa de acierto real del sistema combinado.

---

## 💡 2. Banco de Ideas y Mejoras Futuras

### Ideas para Características (Features)
* **VIX Term Structure:** Incorporar la diferencia entre el VIX de corto plazo (9 días) y el VIX estándar (30 días) para medir el miedo inmediato del mercado.
* **Regímenes de Volatilidad:** Clasificar los días en 3 estados (Alta, Media, Baja volatilidad) usando un algoritmo no supervisado (K-Means) en el train set, y pasar este "estado" como variable categórica a los modelos principales.
* **Días de la semana / Efectos Calendario:** Añadir variables *dummy* para los días de la semana (los lunes y viernes suelen tener comportamientos direccionales distintos a mitad de semana).

### Ideas para Modelado Avanzado
* **Meta-Labeling (Enfoque de Marcos López de Prado):** Usar el modelo de regresión para decidir si entrar o no al mercado (tamaño de la posición) y el de clasificación solo para la dirección. Si la confianza de ambos es baja, el EA se queda en liquidez (0 lotes).
* **Optimización Bayesiana por Ventana:** Usar `Optuna` para encontrar los hiperparámetros óptimos del modelo, pero asegurando que la optimización ocurra *dentro* de cada ventana de entrenamiento del Walk-Forward, evitando mirar el futuro.

### Ideas para la Gestión de Riesgo (Money Management)
* **Asignación de Tamaño por Confianza:** Si el modelo de clasificación predice "Alcista" con un 51% de probabilidad, asignar un riesgo del 0.5% de la cuenta. Si la probabilidad es del 70%, asignar el 1.5%.
* **Filtro de Amplitud Estrecha:** Si el modelo de regresión predice un rango diario extremadamente pequeño (poca volatilidad esperada), programar al EA para que no opere ese día, ya que las comisiones/spreads podrían comerse el beneficio.