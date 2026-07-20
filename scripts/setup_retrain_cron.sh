#!/bin/bash

# Rutas absolutas
PROJECT_ROOT="/home/juan500/Documents/Develops/sp500_range_proyect/sp500_MLops"
PYTHON_BIN=$(which python3)
RETRAIN_SCRIPT="$PROJECT_ROOT/src/RetrainingPipeline/retrain_manager.py"
LOG_FILE="$PROJECT_ROOT/logs/retraining.log"

# Definición del cron: Día 28, 02:00 AM
# Formato: min hora día mes día_semana comando
CRON_JOB="0 2 28 * * cd $PROJECT_ROOT && $PYTHON_BIN -m src.RetrainingPipeline.retrain_manager >> $LOG_FILE 2>&1"

# Eliminar entrada previa si existe para evitar duplicados
crontab -l | grep -v "retrain_manager.py" > mycron
crontab mycron
rm mycron

# Agregar la nueva tarea
(crontab -l ; echo "$CRON_JOB") | crontab -

echo "✅ Tarea de reentrenamiento configurada exitosamente."
echo "📅 Programada para: Día 28 de cada mes a las 02:00 AM"
echo "📂 Log: $LOG_FILE"
