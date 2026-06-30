# README: Diccionario de Variables y Justificación Técnica del Modelo

Este documento contiene la especificación completa del dataset optimizado para el **Sistema de Trading Cuantitativo Híbrido (S&P 500 & VIX)**. Explica la definición matemática de cada variable, su rol en la matriz de datos y la **justificación fundamental/estadística** de por qué es estrictamente necesaria para el entrenamiento del modelo de Machine Learning.

---

## 1. Variable Objetivo (Target / $y$)

### `target`
* **Definición:** `(ln(High_mañana) - ln(Low_mañana))` o implementado en código como `(np.log(df['high']) - np.log(df['low'])).shift(-1)`
* **¿Por qué es necesaria?** Es la variable que el modelo de regresión intentará predecir. Al usar la diferencia de logaritmos en lugar de la resta de puntos absolutos ($High - Low$), neutralizamos por completo el crecimiento exponencial del S&P 500 a lo largo de las décadas (2005-2026). Esto permite que el modelo intente adivinar una **amplitud porcentual relativa**, haciendo que el aprendizaje del pasado sea aplicable a los precios del futuro. El desfasaje `.shift(-1)` asegura que estemos intentando predecir el rango que ocurrirá *mañana* con los datos que tenemos *hoy*.

---

## 2. Características de Entrada (Features / $X$)

Todas las variables de este bloque son **estacionarias** (oscilan de forma estable alrededor de una media constante), garantizando que algoritmos basados en árboles (XGBoost, Random Forest) o distancias (SVM) no sufran fallas de extrapolación ante máximos históricos del mercado.

### 🔹 Bloque A: Estructura y Microestructura del S&P 500 (Hoy - $t$)

#### 1. `sp_gap_log` (El Movimiento Nocturno)
* **Definición:** `ln(Open_hoy) - ln(Close_ayer)`
* **¿Por qué es necesaria?** El mercado nunca duerme. Mientras la sesión oficial de Nueva York está cerrada, ocurren eventos macroeconómicos, reportes de ganancias corporativas y sesiones operativas en Asia y Europa. Esta variable captura la "sorpresa" o el sentimiento acumulado durante la noche. Descomponer el rendimiento total y aislar el Gap evita la multicolinealidad con el cuerpo de la vela y le da al modelo una señal pura de cómo arranca el día.

#### 2. `sp_body_log` (El Movimiento Diurno / Intradía)
* **Definición:** `ln(Close_hoy) - ln(Open_hoy)`
* **¿Por qué es necesaria?** Mide la fuerza direccional neta de los participantes institucionales durante el horario de negociación regular. Un cuerpo verde largo indica acumulación agresiva; un cuerpo rojo largo indica distribución. Al separarlo del `sp_gap_log`, resolvemos el problema de alta correlación (Pearson ~0.8) y permitimos al modelo evaluar si el movimiento diurno continuó o revirtió la tendencia de la noche.

#### 3. `sp_log_range` (Amplitud de la Sesión Actual)
* **Definición:** `ln(High_hoy) - ln(Low_hoy)`
* **¿Por qué es necesaria?** En finanzas existe un fenómeno estadístico llamado **Volatility Clustering** (Agrupamiento de Volatilidad): los días de alta volatilidad tienden a ser seguidos por días de alta volatilidad, y los días de calma por días de calma. El rango logarítmico de hoy es el predictor lineal más fuerte del rango de mañana. Proporciona la "inercia" actual del mercado.

#### 4. `sp_upper_wick_log` (Mecha Superior)
* **Definición:** `ln(High_hoy) - ln(max(Open_hoy, Close_hoy))`
* **¿Por qué es necesaria?** Representa el "rechazo de precios altos". Mide la capacidad de los vendedores para empujar el precio hacia abajo después de que este alcanzara el máximo del día. Un valor alto indica agotamiento comprador o resistencia fuerte, lo que suele preceder a una contracción del rango o un cambio de dirección bajista al día siguiente.

#### 5. `sp_lower_wick_log` (Mecha Inferior)
* **Definición:** `ln(min(Open_hoy, Close_hoy)) - ln(Low_hoy)`
* **¿Por qué es necesaria?** Representa el "rechazo de precios bajos" (compras de pánico o absorción). Mide el contraataque de los compradores tras una caída intradía. Si el mercado cae con fuerza pero recupera el terreno antes del cierre dejando una mecha larga, el modelo interpretará esta resiliencia como una señal de soporte que afectará la estructura de volatilidad de la jornada posterior.

#### 6. `sp_vol_5d` y `sp_vol_20d` (Regímenes de Volatilidad Histórica)
* **Definición:** Desviación estándar rodante de los retornos logarítmicos en ventanas de 5 días (una semana) y 20 días (un mes de trading).
* **¿Por qué son necesarias?** El mercado cambia de estados de ánimo. Pasar de un mercado alcista lento y tranquilo a un mercado bajista rápido y violento altera la distribución de los rangos diarios. Estas variables le otorgan al modelo el **contexto macro**. Una misma señal estructural (ej. una mecha larga) no significa lo mismo en un entorno de pánico generalizado (`sp_vol_20d` alto) que en un mercado lateral y plano (`sp_vol_20d` bajo).

---

### 🔹 Bloque B: El Sentimiento del Mercado y Cobertura (VIX - $t$)

El VIX (CBOE Volatility Index) mide la volatilidad implícita de las opciones del S&P 500 a 30 días. Es considerado por excelencia el "índice del miedo".

#### 7. `vix_log_close` (El Nivel de Miedo Absoluto)
* **Definición:** `ln(VIX_Close_hoy)`
* **¿Por qué es necesaria?** A diferencia de las acciones, el VIX es una serie de tiempo estacionaria por naturaleza (tiende a regresar a su media histórica de ~18-20 puntos). No crece al infinito. Su nivel absoluto al cierre de hoy determina las expectativas matemáticas de riesgo de las instituciones financieras. Un VIX alto significa que las opciones de cobertura son caras porque los creadores de mercado esperan movimientos gigantescos mañana.

#### 8. `vix_log_range` (Nerviosismo Intradía del VIX)
* **Definición:** `ln(VIX_High_hoy) - ln(VIX_Low_hoy)`
* **¿Por qué es necesaria?** Mide la estabilidad o inestabilidad del miedo dentro del mismo día. Si el VIX oscila violentamente en un rango amplio hoy, significa que la convicción de los operadores institucionales es frágil y están cambiando de opinión rápidamente. Esta inestabilidad emocional se traslada directamente como una señal de ensanchamiento de rango para el S&P 500 al día siguiente.

#### 9. `vix_log_return` (Aceleración del Miedo)
* **Definición:** `ln(VIX_Close_hoy) - ln(VIX_Close_ayer)`
* **¿Por qué es necesaria?** Mide el *impulso (momentum)* del pánico. No es lo mismo tener un VIX en 25 puntos que viene bajando desde 30 (el pánico se está calmando), que tener un VIX en 25 puntos que acaba de subir disparado desde 15 (el pánico está acelerando). El modelo necesita saber la velocidad del cambio del sentimiento para predecir si el rango de mañana se expandirá exponencialmente.

---

### 🔹 Bloque C: Estructura del Tiempo (Efectos Calendario)

#### 10. `day_of_week`
* **Definición:** Entero del 0 al 4 extraído del índice de fechas (`df.index.dayofweek`), donde 0 = Lunes y 4 = Viernes.
* **¿Por qué es necesaria?** El comportamiento institucional varía según el día de la semana debido a la liquidez y las normativas de riesgo. Los lunes suelen experimentar reajustes de flujos por el fin de semana; los viernes sufren cierres de posiciones por parte de traders que no quieren asumir el riesgo de noticias durante el sábado y el domingo. Esta variable permite al algoritmo capturar la estacionalidad del calendario.

---

## 3. Columnas Excluidas (Metadata / Materia Prima)

* **Variables:** `open`, `high`, `low`, `close`, `vix_open`, `vix_high`, `vix_low`, `vix_close`
* **¿Por qué se deben eliminar antes del entrenamiento?**
  Estas columnas son indispensables para calcular todas las variables anteriores. Sin embargo, **nunca deben entrar al método `model.fit()`**. Al ser precios absolutos indexados a la moneda (USD), carecen de cota superior. Dejarlas provocaría que tu modelo sea incapaz de generalizar en el futuro, destruyendo tu estrategia de trading automatizada (EA) cuando los precios alcancen niveles jamás vistos por el algoritmo durante su etapa de entrenamiento.
