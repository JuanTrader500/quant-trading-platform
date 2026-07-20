# Automation Scripts - Retraining
Este directorio contiene los scripts necesarios para la automatización del pipeline de reentrenamiento en el sistema operativo.

## Setup de Cron
El script `setup_retrain_cron.sh` automatiza la creación de una tarea programada en el `crontab` del usuario.

### Lógica de Ejecución
La tarea está configurada para ejecutarse el **día 28 de cada mes a las 02:00 AM**. 
Se ha seleccionado este horario para minimizar el impacto en los recursos del sistema y asegurar que la mayoría de los datos del mes en curso ya estén disponibles.

### Comando Ejecutado
El script invoca al `retrain_manager.py` utilizando el ejecutable de Python del sistema, asegurando que el contexto de trabajo sea la raíz del proyecto para la resolución de rutas.

## Instrucciones de Uso
1. Otorgar permisos de ejecución: `chmod +x scripts/setup_retrain_cron.sh`
2. Ejecutar el script: `./scripts/setup_retrain_cron.sh`
3. Verificar la tarea: `crontab -l`
