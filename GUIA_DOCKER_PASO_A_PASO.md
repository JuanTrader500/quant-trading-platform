# Guía paso a paso: base de datos y pipeline con Docker (desde cero)

Esta guía asume que nunca has creado una base de datos con Docker.
Todos los comandos están probados para **Fedora**. Al final vas a
tener: PostgreSQL/TimescaleDB corriendo en un contenedor con datos
persistentes, el esquema ya aplicado, el pipeline corriendo también en
Docker, y vas a saber dónde y cómo consultar los datos.

Estructura de carpetas que asume esta guía:

```
sp500_mlops/                    ← raíz del proyecto (créala tú)
├── docker-compose.yml          ← adjunto
├── .env                        ← lo creas en el paso 4
├── data_service/               ← contenido del zip que ya tienes
│   ├── docs/data_service_schema.sql
│   ├── pipeline/
│   ├── app/
│   ├── Dockerfile
│   └── ...
└── logs/                       ← se crea sola al levantar los contenedores
```

---

## Paso 1 — Instalar Docker en Fedora

Fedora no trae Docker por defecto (trae Podman). Instala Docker Engine
desde el repositorio oficial de Docker, no desde los repos de Fedora
(suelen ir desactualizados):

```bash
# 1. Quitar versiones viejas o en conflicto, si existen
sudo dnf remove -y docker docker-client docker-client-latest docker-common \
  docker-latest docker-latest-logrotate docker-logrotate docker-selinux \
  docker-engine-selinux docker-engine podman-docker

# 2. Agregar el repositorio oficial de Docker
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager addrepo --from-repofile=https://download.docker.com/linux/fedora/docker-ce.repo

# 3. Instalar Docker Engine + Compose plugin
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 4. Habilitar y arrancar el servicio
sudo systemctl enable --now docker

# 5. Permitir usar docker sin sudo (opcional pero recomendado)
sudo usermod -aG docker $USER
```

**Importante:** después del paso 5, cierra sesión y vuelve a entrar
(o reinicia) para que el cambio de grupo tenga efecto. Sin esto, cada
comando `docker` de aquí en adelante necesita `sudo` delante.

### Verificar la instalación

```bash
docker --version
docker compose version
docker run hello-world
```

Si el último comando descarga una imagen y muestra un mensaje de
bienvenida, Docker está funcionando correctamente.

---

## Paso 2 — Preparar la carpeta del proyecto

```bash
mkdir -p ~/sp500_mlops
cd ~/sp500_mlops
```

Descomprime ahí el `data_service.zip` que ya tienes, de modo que quede
`~/sp500_mlops/data_service/...`. Coloca el `docker-compose.yml` y el
`.env.example` adjuntos directamente en `~/sp500_mlops/` (al mismo
nivel que la carpeta `data_service/`, **no** dentro de ella).

Verifica que quedó así:

```bash
ls ~/sp500_mlops
# data_service  docker-compose.yml  .env.example
```

---

## Paso 3 — Entender qué hace el docker-compose.yml

Dos servicios:

- **`timescaledb`**: el contenedor de la base de datos. Usa un
  **volumen con nombre** (`timescale-data`) para que los datos
  sobrevivan a reinicios del contenedor. Además monta
  `data_service/docs/data_service_schema.sql` en una carpeta especial
  (`/docker-entrypoint-initdb.d/`) que Postgres ejecuta **automáticamente
  la primera vez** que el contenedor se crea con un volumen vacío —
  así el esquema completo (extensión TimescaleDB, tablas, datos
  semilla) se aplica solo, sin que tengas que correr nada a mano.
- **`data_service`**: construye la imagen a partir del `Dockerfile`
  que ya está en `data_service/` y corre la API (FastAPI) del
  pipeline. Espera a que la base de datos esté lista (`depends_on` +
  `healthcheck`) antes de arrancar.

---

## Paso 4 — Configurar las variables de entorno

```bash
cd ~/sp500_mlops
cp .env.example .env
nano .env    # o el editor que prefieras
```

Cambia como mínimo `POSTGRES_PASSWORD` por una contraseña real. Este
`.env` lo lee automáticamente `docker-compose` (no hace falta pasarlo
con `--env-file`, basta con que esté en la misma carpeta).

---

## Paso 5 — Levantar todo por primera vez

```bash
cd ~/sp500_mlops
docker compose up -d --build
```

- `-d` = en segundo plano (detached).
- `--build` = construye la imagen del `data_service` (solo hace falta
  la primera vez, o cuando cambies el código/Dockerfile).

Esto va a:
1. Descargar la imagen de TimescaleDB (puede tardar un par de minutos
   la primera vez).
2. Crear el volumen `timescale-data`.
3. Arrancar Postgres y ejecutar automáticamente
   `data_service_schema.sql` (solo esta primera vez).
4. Construir y arrancar el contenedor del `data_service`.

### Verificar que ambos contenedores están arriba

```bash
docker compose ps
```

Deberías ver `data-service-db` y `data-service-app` con estado `Up` (o
`healthy` para la base de datos).

### Ver logs si algo no arrancó

```bash
docker compose logs timescaledb
docker compose logs data_service
```

---

## Paso 6 — Confirmar que el esquema se aplicó

```bash
docker compose exec timescaledb psql -U data_service_app -d data_service_db -c "\dt"
```

Debe listar las tablas: `instruments`, `asset_pairs`, `raw_ohlc`,
`features`, `feature_schema_versions`, `ingestion_log`.

```bash
docker compose exec timescaledb psql -U data_service_app -d data_service_db \
  -c "SELECT * FROM asset_pairs;"
```

Debe mostrar las dos filas semilla: `SP500_VIX` y `NASDAQ_VXN`.

---

## Paso 7 — Comprobar que los datos persisten (opcional, pero hazlo una vez)

```bash
docker compose stop
docker compose start
docker compose exec timescaledb psql -U data_service_app -d data_service_db \
  -c "SELECT * FROM asset_pairs;"
```

Las mismas filas siguen ahí después de apagar y prender. Eso confirma
que el volumen está funcionando.

---

## Paso 8 — Correr el pipeline de datos desde Docker

Tienes dos formas, según lo que necesites:

### Opción A — Vía la API (recomendada para uso normal)

El contenedor `data_service` ya está corriendo la API en el puerto
8000. Dispara el pipeline con una petición HTTP:

```bash
curl -X POST http://localhost:8000/pipeline/run
```

Corre extracción + preparación de forma síncrona y te devuelve
`{"status": "completed"}` cuando termina (puede tardar según cuántos
datos históricos falten la primera vez).

### Opción B — Ejecutando el script directamente dentro del contenedor

Útil para depurar o ver el log en vivo:

```bash
docker compose exec data_service python -m pipeline.pipeline_manager
```

### Ver el resultado

```bash
curl "http://localhost:8000/features/latest?pair_code=SP500_VIX"
```

O revisa los logs de texto (montados al host en el paso 5):

```bash
tail -f ~/sp500_mlops/logs/data_service/data_pipeline.log
```

---

## Paso 9 — Cómo y dónde hacer queries a la base de datos

Tres formas, de más simple a más cómoda:

### 9.1 — psql dentro del contenedor (no necesitas instalar nada en Fedora)

```bash
docker compose exec timescaledb psql -U data_service_app -d data_service_db
```

Te deja en una sesión interactiva de `psql`. Algunas queries útiles
para explorar:

```sql
-- ver las últimas velas descargadas
SELECT * FROM raw_ohlc ORDER BY date DESC LIMIT 10;

-- ver las últimas features calculadas de un par
SELECT * FROM features WHERE pair_code = 'SP500_VIX' ORDER BY date DESC LIMIT 5;

-- ver el historial de corridas del pipeline (auditoría, RF06)
SELECT run_at, pipeline_stage, ticker, pair_code, status, rows_affected
FROM ingestion_log
ORDER BY run_at DESC LIMIT 10;

-- ver la versión de esquema vigente
SELECT * FROM feature_schema_versions WHERE is_current;
```

Para salir: `\q`.

### 9.2 — psql desde tu Fedora, sin entrar al contenedor

Como el puerto 5432 está publicado (`ports: "5432:5432"` en el
compose), puedes conectarte desde el host si tienes el cliente `psql`
instalado (`sudo dnf install postgresql`):

```bash
psql "postgresql://data_service_app:TU_PASSWORD@localhost:5432/data_service_db"
```

**Nota:** aquí sí usas `localhost`, a diferencia del `DATABASE_URL`
que usa el contenedor `data_service` internamente (que usa
`timescaledb` como host, porque están en la misma red de Docker). Esta
es la confusión más común al empezar con Docker — recuérdalo:

| Quién se conecta | Host a usar |
|---|---|
| Un contenedor de la misma red de docker-compose (ej. `data_service`) | `timescaledb` (nombre del servicio) |
| Tu Fedora, fuera de Docker (psql, DBeaver, tu propio script) | `localhost` |

### 9.3 — Con una herramienta gráfica (opcional, más cómodo para explorar)

Cualquier cliente de PostgreSQL sirve: DBeaver, pgAdmin, TablePlus.
Conéctate con:

- Host: `localhost`
- Puerto: `5432`
- Base de datos: `data_service_db`
- Usuario / contraseña: los que pusiste en `.env`

---

## Comandos del día a día (referencia rápida)

```bash
# Levantar todo (si ya existía, no reconstruye salvo que cambie el código)
docker compose up -d

# Apagar sin borrar datos
docker compose stop

# Apagar y quitar los contenedores (el volumen de datos sigue intacto)
docker compose down

# ⚠️ Apagar y BORRAR también los datos (úsalo solo a propósito)
docker compose down -v

# Reconstruir la imagen del data_service tras cambiar código
docker compose up -d --build data_service

# Ver logs en vivo
docker compose logs -f data_service

# Entrar a una shell dentro del contenedor del pipeline
docker compose exec data_service bash

# Backup manual de la base de datos
docker compose exec timescaledb pg_dump -U data_service_app data_service_db > backup_$(date +%F).sql
```

---

## Problemas comunes

**`permission denied` al correr `docker`**
No cerraste sesión después de `usermod -aG docker $USER` (paso 1.5).
Cierra sesión y vuelve a entrar, o usa `sudo` delante de cada comando
mientras tanto.

**`port is already allocated` en el puerto 5432**
Ya tienes tu PostgreSQL 18 nativo escuchando en ese puerto. Cambia el
mapeo en `docker-compose.yml` a, por ejemplo, `"5433:5432"`, y ajusta
el puerto en las conexiones desde el host (paso 9.2/9.3). El contenedor
`data_service` no se ve afectado porque internamente sigue usando el
puerto 5432 dentro de la red de Docker.

**El script SQL no se aplicó (tablas vacías)**
`/docker-entrypoint-initdb.d` solo se ejecuta la **primera vez** que el
volumen está vacío. Si ya habías levantado el contenedor antes (aunque
haya fallado), el volumen ya existe y el script no se vuelve a correr.
Solución para forzar una reinicialización limpia (⚠️ borra los datos
actuales):
```bash
docker compose down -v
docker compose up -d --build
```

**`data_service` no logra conectarse a la base de datos**
Revisa que `DATABASE_URL` dentro del contenedor use `timescaledb`
como host (ya viene así en el `docker-compose.yml` adjunto) y no
`localhost` — ese es el error más común (ver tabla del paso 9.2).
