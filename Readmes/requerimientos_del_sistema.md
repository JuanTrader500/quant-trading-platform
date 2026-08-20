Requerimientos del Sistema — Plataforma de
Predicción de Volatilidad SP500
Módulo 1: Ingesta y Procesamiento de Datos (Data Engineering)
RF01: El sistema debe extraer datos históricos de velas diarias (OHLC) del SP500 y del VIX
mediante la API de Yahoo Finance.
RF02: El sistema debe calcular y almacenar las siguientes variables derivadas a partir de
los datos crudos: sp500_log_return, sp500_log_range, sp500_body_log,
sp500_upper_wick_log, sp500_lower_wick_log, sp500_vol_5d, sp500_vol_10d,
vix_log_close, vix_log_range, vix_log_return y day_of_week.
RF03: El sistema debe garantizar que el cálculo de cada feature utilice únicamente
información disponible hasta el cierre del día t para predecir el rango del día t+1, evitando
data leakage.
RF04: El sistema debe validar la integridad de los datos descargados (valores nulos,
duplicados, gaps de fechas por días no hábiles) antes de persistirlos.
RF05: El sistema debe permitir la actualización incremental del dataset histórico sin
necesidad de re-descargar la serie completa.
RF06: El sistema debe registrar en logs cada proceso de extracción de datos, incluyendo
fecha de ejecución, rango de fechas obtenido y errores de conexión con la API externa.
Módulo 2: Entrenamiento y Gestión del Modelo ML (MLOps)
RF07: El sistema debe permitir la exportación del modelo entrenado en formato .pkl
(pickle) desde el entorno de experimentación (Jupyter Notebook).
RF08: El sistema debe implementar un script de reentrenamiento automático del modelo
utilizando la técnica de Walk-Forward Validation, ejecutable con una periodicidad
configurable (mensual por defecto).
RF09: El sistema debe entrenar y comparar al menos los siguientes algoritmos: Ridge
Regression, Lasso, ElasticNet y modelos basados en árboles.
RF10: El sistema debe calcular un modelo GARCH como baseline de referencia para el
mismo objetivo (predicción de rango/volatilidad diaria).
RF11: El sistema debe versionar cada modelo entrenado (timestamp, hiperparámetros,
métricas de validación) para permitir trazabilidad y rollback a una versión anterior.RF12: El sistema debe cargar el modelo .pkl vigente en memoria al iniciar el servicio de
predicción, sin requerir reentrenamiento en cada solicitud.
RF13: El sistema debe exponer un mecanismo (management command de Django o script
independiente) para reemplazar el modelo en producción sin downtime del servicio web.
Módulo 3: Motor de Predicción (Backend ML)
RF14: El sistema debe ofrecer un modo "Testing" donde el usuario ingresa manualmente
valores OHLC del SP500 y el sistema transforma dichos datos en las features requeridas
por el modelo.
RF15: El sistema debe ofrecer un modo "Predicción de Mañana" donde el sistema consulta
automáticamente la API de Yahoo Finance para obtener los datos más recientes y generar
la predicción del rango del siguiente día hábil.
RF16: El sistema debe devolver como salida de la predicción, como mínimo: el valor de
rango porcentual esperado, la fecha objetivo de la predicción y el modelo utilizado.
RF17: El sistema debe calcular y exponer las métricas de evaluación RMSE y MAE del
modelo vigente sobre el conjunto de validación más reciente.
RF18: El sistema debe calcular una métrica propia de sesgo direccional que distinga entre
subestimación de volatilidad (predicción menor a la volatilidad real) y sobreestimación de
volatilidad (predicción mayor a la volatilidad real).
RF19: El sistema debe comparar automáticamente el desempeño (RMSE, MAE, métrica de
sesgo) del modelo ML contra el modelo GARCH baseline y mostrar el resultado de dicha
comparación.
RF20: El sistema debe rechazar y notificar al usuario cuando los valores OHLC ingresados
manualmente en modo Testing sean matemáticamente inválidos (ej. Low > High, Open o
Close fuera del rango [Low, High]).
Módulo 4: Interfaz Web de Predicción (Frontend/Django Views)
RF21: El sistema debe presentar un formulario web para el modo Testing con campos de
entrada para Open, High, Low y Close.
RF22: El sistema debe presentar un botón/acción de "Predecir mañana" que dispare la
consulta automática a Yahoo Finance y muestre el resultado sin intervención manual de
datos.
RF23: El sistema debe mostrar visualmente (gráfico o indicador) si la predicción representa
un régimen de volatilidad alta, media o baja respecto al histórico reciente.RF24: El sistema debe mostrar un historial de predicciones anteriores junto con el resultado
real observado una vez disponible, para fines de transparencia.
Módulo 5: Tienda de Algoritmos (Producto)
RF25: El sistema debe listar los algoritmos de trading disponibles, cada uno con una
descripción funcional de su lógica de operación.
RF26: El sistema debe mostrar, por cada algoritmo, métricas de backtesting que incluyan
como mínimo: Ratio de Sharpe y Ratio de Sortino.
RF27: El sistema debe permitir la descarga de una versión demo del algoritmo en formato
ejecutable .mq5.
RF28: El sistema debe generar dinámicamente el archivo .mq5 de demo, modificando la
función OnInit() para inyectar una fecha de expiración calculada como fecha_actual + 10
días.
RF29: El sistema debe impedir la ejecución del algoritmo .mq5 una vez superada la fecha
de expiración embebida en el código de la demo.
RF30: El sistema debe registrar cada descarga de demo (usuario, algoritmo, fecha de
generación, fecha de expiración) para control interno y evitar regeneraciones abusivas de la
ventana de prueba.
RF31: El sistema debe integrar una pasarela de pagos (con fines educativos/de
aprendizaje) que permita procesar una transacción de prueba para la suscripción o compra
de un algoritmo.
RF32: El sistema debe actualizar el estado de acceso del usuario (habilitado/deshabilitado)
a la versión completa del algoritmo en función del resultado de la transacción de pago.
Módulo 6: Gestión de Usuarios y Autenticación
RF33: El sistema debe permitir el registro y autenticación de usuarios mediante el sistema
de autenticación nativo de Django.
RF34: El sistema debe asociar cada predicción realizada en modo Testing y cada descarga
de demo a un usuario autenticado.
RF35: El sistema debe restringir la funcionalidad de compra/suscripción de algoritmos
únicamente a usuarios autenticados.Requerimientos No Funcionales
Rendimiento
RNF01: El sistema debe generar una predicción (modo Testing o modo Automático) en un
tiempo de respuesta inferior a 3 segundos bajo condiciones normales de operación.
RNF02: El sistema debe soportar la limitación del número de solicitudes de predicción por
usuario/IP (rate limiting) para mitigar sobrecarga en caso de que el modelo o la consulta a
Yahoo Finance incrementen su latencia.
Confiabilidad y Manejo de Errores
RNF03: El sistema debe manejar de forma controlada la indisponibilidad de la API de Yahoo
Finance, mostrando un mensaje de error claro al usuario sin exponer trazas técnicas (stack
traces) en producción.
RNF04: El sistema debe mantener un modelo .pkl de respaldo (última versión estable)
disponible en caso de que el proceso de reentrenamiento automático falle o produzca
métricas inferiores al modelo vigente.
RNF05: El sistema debe deshabilitar el modo DEBUG de Django en el entorno de producción
para prevenir la exposición de información sensible (rutas, variables de entorno, stack
traces), dado el precedente de vulnerabilidad de debug detectado en proyectos previos del
equipo.
Seguridad
RNF06: El sistema debe almacenar las credenciales de la pasarela de pagos y claves de
API mediante variables de entorno, nunca en el código fuente ni en el repositorio.
RNF07: El sistema debe validar y sanear toda entrada del usuario (formularios OHLC,
formularios de pago) contra inyección SQL y XSS.
RNF08: El sistema debe generar el archivo .mq5 de demo de forma que la fecha de
expiración embebida no pueda ser trivialmente editada por el usuario final sin conocimientos
de compilación MQL5.
RNF09: El sistema debe registrar (logging) todos los intentos de transacción de pago,
exitosos y fallidos, sin almacenar datos sensibles de tarjeta en la base de datos propia
(cumplimiento básico tipo PCI-DSS a nivel educativo).
Mantenibilidad y Escalabilidad
RNF10: El sistema debe desacoplar la lógica de extracción/transformación de datos
(pipeline extraction.py, preparation.py) de la lógica de servicio web (views de
Django), permitiendo ejecutar el pipeline de forma independiente vía script o management
command.RNF11: El sistema debe estructurar el pipeline de entrenamiento de forma modular para
permitir agregar nuevos algoritmos de ML sin modificar el código de las vistas o la API de
predicción.
RNF12: El sistema debe versionar el esquema de features (nombres y orden de columnas)
de forma que un cambio en las features utilizadas invalide automáticamente modelos
entrenados con un esquema anterior, evitando predicciones inconsistentes.
Usabilidad
RNF13: El sistema debe presentar las métricas de backtesting (Sharpe, Sortino) y de
predicción (RMSE, MAE) en un formato comprensible para usuarios sin formación técnica
en Machine Learning, mediante tooltips o descripciones breves.
RNF14: El sistema debe ser responsivo (responsive design), permitiendo el uso del
formulario de predicción y la tienda de algoritmos desde dispositivos móviles.