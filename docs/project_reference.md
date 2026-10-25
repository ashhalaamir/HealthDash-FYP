# HealthDash reference

One document for everything. This first part covers **running live mode**: the database, the worker and
Airflow. Further parts (architecture, spec format, evaluation) get added here as they are written.

**Contents:** 1 What live mode is · 2 Database setup · 3 Running live mode · 4 Airflow · 5 Command cheat sheet · 6 Troubleshooting

---

## 1. What live mode is

HealthDash has two modes that share one engine (cleaning, profiling, comparison, findings, summaries):

| | Explore mode | Live mode |
|---|---|---|
| Data comes from | a file you upload in the app | files dropped into `data/incoming/`, stored in a database |
| Dates | you pick two dates or periods | a rolling window (last 7 days vs the 7 before) anchored to the newest data |
| Runs | when you click | on a schedule (worker or Airflow) |
| Output | a dashboard on screen | a stored **snapshot** per refresh, so you can compare "now" with the previous refresh |

"Live" here means **a refresh every N minutes** (micro-batch), not streaming. Latency equals the schedule interval.

```
data/incoming/<dataset>/file.csv  ->  ingest (clean, dedupe)  ->  database  ->  refresh  ->  snapshot  ->  dashboard
                                      src/ingest.py                src/db.py    src/live.py
                                      repeated by src/worker.py or the Airflow DAG
```

The subfolder name is the dataset name: `data/incoming/opd/week1.csv` is loaded into dataset `opd`.

---

## 2. Database setup

You have three options. **Start with A**; move to B when you want the real thing.

### Option A: SQLite (default, nothing to install)

Do nothing. The app creates `data/healthdash.db` on first use. Good for development and demos on one computer.
Limits: one machine, and writes are one at a time (fine for a worker plus a dashboard reading).

### Option B: PostgreSQL in Docker (recommended)

**You need:** Docker Desktop. On Windows it needs WSL 2 and virtualization enabled in the BIOS (Docker Desktop's installer
guides you through both). Check it works:

```
docker --version
docker compose version
```

**Steps** (all from the repository root, the folder containing `docker-compose.yml`):

1. Start the database. The first run downloads the image, which takes a minute.
   ```
   docker compose up -d db
   ```
2. Wait until it is healthy (the STATUS column should say `healthy`, usually within 15 seconds):
   ```
   docker compose ps
   ```
3. Create a file named `.env` (copy `.env.example`) and set:
   ```
   DATABASE_URL=postgresql+psycopg2://healthdash:healthdash@localhost:5432/healthdash
   ```
4. Install the Python packages (this includes the Postgres driver `psycopg2-binary`):
   ```
   pip install -r requirements.txt
   ```
5. Check the connection. You should see `Database: postgresql+psycopg2://healthdash:***@localhost:5432/healthdash`
   and "no datasets yet":
   ```
   python -m src.db status
   ```
6. **Verify it works** by running the tests against Postgres too. Add this line to `.env`:
   ```
   TEST_DATABASE_URL=postgresql+psycopg2://healthdash:healthdash@localhost:5432/healthdash
   ```
   then run:
   ```
   python -m pytest tests -q
   ```
   Every database test now runs twice, once on SQLite and once on Postgres. The expected result is **no failures**.
   Each Postgres test runs in its own temporary schema that is deleted afterwards, so the test run **cannot touch your real
   tables**.

**Look inside the database:**
```
docker compose exec db psql -U healthdash -d healthdash
```
Then at the `healthdash=#` prompt: `\dt` lists tables, `select count(*) from records;` counts stored rows,
`select * from pipeline_runs order by id desc limit 5;` shows the last refreshes, `\q` quits.

**Stop, back up, reset:**

| Goal | Command |
|---|---|
| Stop, keep the data | `docker compose stop` (start again with `docker compose up -d db`) |
| Back up | `docker compose exec db pg_dump -U healthdash healthdash > backup.sql` |
| Restore | `docker compose exec -T db psql -U healthdash -d healthdash < backup.sql` |
| **Delete all data** and start fresh | `docker compose down -v` (the `-v` removes the data volume: this cannot be undone) |

**Moving from SQLite to Postgres:** data is not copied across automatically. Point `DATABASE_URL` at Postgres and run
`python -m src.ingest` again. Ingestion is safe to repeat, and your source files in `data/incoming/` are the record of what to load.

### Option C: PostgreSQL installed directly on Windows (no Docker)

1. Download the installer from postgresql.org (the Windows installer), run it, keep the default port 5432, and set a password
   for the `postgres` user. Remember it.
2. Open **SQL Shell (psql)** from the Start menu, log in as `postgres`, and run:
   ```sql
   CREATE USER healthdash WITH PASSWORD 'choose-a-password';
   CREATE DATABASE healthdash OWNER healthdash;
   ```
3. Put this in `.env` and continue from step 4 above:
   ```
   DATABASE_URL=postgresql+psycopg2://healthdash:choose-a-password@localhost:5432/healthdash
   ```
   If the password contains special characters, they must be URL-encoded in that line (for example `@` becomes `%40`).

### Security notes

* The credentials in `docker-compose.yml` are development defaults and the port is bound to your own computer only
  (`127.0.0.1`). Do not change that to expose it to a network.
* `.env` holds secrets. It must be listed in `.gitignore`, and never committed. Suggested `.gitignore` lines:
  ```
  .env
  .venv/
  __pycache__/
  .pytest_cache/
  data/*.db
  data/incoming/
  data/output/
  data/generated/
  ```
* Real patient data brings legal and ethical obligations: institutional approval, de-identification, access control and
  encryption. Keep demos on synthetic data. The LLM only ever receives summaries (counts, percentages, category names),
  never patient rows, but category names can themselves be sensitive, so check with your supervisor before using a
  cloud LLM on real data.

---

## 3. Running live mode

**One-time check (safe to repeat):**
```
python scripts/generate_fake_data.py          # makes sample data in data/generated/
```

**By hand, step by step:**
```
# 1. put a file in a dataset folder
mkdir data\incoming\opd                       # macOS/Linux: mkdir -p data/incoming/opd
copy data\generated\opd_visits.csv data\incoming\opd\week1.csv

# 2. load it into the database
python -m src.ingest

# 3. build a dashboard snapshot for a rolling 30-day window
python -m src.live opd --window 30d

# 4. see what is stored
python -m src.db status
```

**Automatically, with the worker** (the simplest scheduler; works on any computer):
```
python -m src.worker --every 30m              # keep running; Ctrl+C stops it
python -m src.worker --every 2m               # a fast cycle for a demo
python -m src.worker --once                   # one cycle then exit (for Task Scheduler / cron)
python -m src.worker --every 30m --llm        # the LLM writes the layout (once) and the summary (when findings change)
```
Each cycle: scan `data/incoming/` for new files, load them, then refresh every dataset. A refresh with nothing new is
skipped cheaply, and a file is never loaded twice, even if renamed. A file still being copied (changed in the last 2
seconds) waits for the next cycle. Run **one** worker per database.

**A demo script:** start the worker with `--every 1m`, copy `week1.csv` into `data/incoming/opd/`, wait a minute, then copy
`week2.csv`. The next snapshot reports what changed since the previous refresh.

**Windows Task Scheduler instead of a running worker:** create a task that runs every 30 minutes with the program
`python`, arguments `-m src.worker --once`, and "Start in" set to the repository folder.

**LLM key:** put it in `.env` (see `.env.example`). If the model name stops working, run `python -m src.llm models`.

---

## 4. Airflow (optional, for the scheduler your supervisor mentioned)

`airflow/dags/dashboard_dag.py` defines the DAG `healthdash_refresh`: every 30 minutes, `ingest_new_files` then
`refresh_snapshots`. It calls the same functions as the worker, so results are identical. Only one run can be active at a
time, a failed refresh fails the task (so it shows red in Airflow and is retried twice), and a bad data file does not.

**What has been verified:** the DAG was parsed and executed end to end by **Apache Airflow 3.3.2** installed with pip
(`airflow dags test healthdash_refresh`): both tasks succeeded and a snapshot was written. **Not verified:** running Airflow
inside Docker, which is the route for Windows, because Airflow does not run natively on Windows as far as I know. Use WSL 2
or Docker. Follow the **official Airflow Docker Compose guide for your Airflow version**
(https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html), then make these changes:

1. In the Airflow `docker-compose.yaml`, under `x-airflow-common`, set the DAG folder and mount the project:
   ```yaml
   environment:
     HEALTHDASH_HOME: /opt/airflow/healthdash
     HEALTHDASH_WINDOW: 7d
     DATABASE_URL: postgresql+psycopg2://healthdash:healthdash@host.docker.internal:5432/healthdash
     _PIP_ADDITIONAL_REQUIREMENTS: "pandas sqlalchemy psycopg2-binary requests python-dotenv openpyxl"
   volumes:
     - /full/path/to/HealthDash-FYP:/opt/airflow/healthdash
     - /full/path/to/HealthDash-FYP/airflow/dags:/opt/airflow/dags
   ```
   `host.docker.internal` lets the Airflow containers reach the Postgres you started with this project's `docker-compose.yml`.
   `_PIP_ADDITIONAL_REQUIREMENTS` is the official image's quick way to add packages; it is meant for testing only.
2. Run `docker compose up airflow-init`, then `docker compose up`, open the Airflow web page, and switch the
   `healthdash_refresh` DAG on. To change how often it runs, set `HEALTHDASH_SCHEDULE` (a cron expression).
3. Do not run the worker and Airflow at the same time against the same database.

If Airflow in Docker costs you more time than it is worth, the worker above gives the same behaviour and you can describe
Airflow as the production scheduler in your report.

---

## 5. Command cheat sheet

| Task | Command |
|---|---|
| Start Postgres | `docker compose up -d db` |
| Stop Postgres (keep data) | `docker compose stop` |
| Check the database | `python -m src.db status` |
| Load new files once | `python -m src.ingest` |
| Load one file into a dataset | `python -m src.ingest path/to/file.csv --dataset opd` |
| Build one snapshot | `python -m src.live opd --window 7d` |
| Force a snapshot even if nothing changed | `python -m src.live opd --window 7d --force` |
| Keep it fresh automatically | `python -m src.worker --every 30m` |
| One cycle only | `python -m src.worker --once` |
| List usable LLM models | `python -m src.llm models` |
| Run all tests | `python -m pytest tests -q` |
| Upload-mode app | `streamlit run app/streamlit_app.py` |

---

## 6. Troubleshooting

| What you see | Likely cause and fix |
|---|---|
| `Connection refused ... port 5432` | The database is not running. `docker compose up -d db`, then `docker compose ps` until `healthy`. |
| `password authentication failed for user "healthdash"` | The password in `DATABASE_URL` differs from the one the database was created with. If you changed `POSTGRES_PASSWORD` after the first start, the old value is still stored: `docker compose down -v` (deletes data) and start again. |
| `No module named 'psycopg2'` | `pip install -r requirements.txt` in the same Python environment you run from (check `python -m pip --version`). |
| `port is already allocated` / `address already in use` | Another Postgres is using 5432. Stop it, or change the left number in `"127.0.0.1:5432:5432"` (for example `5433`) and use `:5433` in `DATABASE_URL`. |
| Docker says it cannot connect to the daemon | Start Docker Desktop and wait for it to say it is running. On Windows, check WSL 2 is enabled. |
| `database "healthdash" does not exist` | You used Option C but skipped step 2, or you changed `POSTGRES_DB`. Create the database, or recreate the Docker volume with `docker compose down -v`. |
| `Unknown dataset 'x'` | No file has been loaded for that dataset yet. Put one in `data/incoming/x/` and run `python -m src.ingest`. |
| A file shows `FAILED` with a message | Read the message: missing date column, empty file, or columns that differ from the dataset's first file. The failure is logged and the file is not retried; fix the file (any change makes it a new file) and re-add it. |
| `Nothing new since the last snapshot` | Correct behaviour. Use `--force` if you want a snapshot anyway. |
| The worker prints `still being written` | The file changed in the last 2 seconds. It will load on the next cycle. |
| `404` from the LLM | The model name was retired. `python -m src.llm models`, then set `GEMINI_MODEL` in `.env`. |