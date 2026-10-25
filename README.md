# HealthDash

An automated business-intelligence dashboard for clinical data. Give it a table of records (outpatient visits, ICU
admissions, pharmacy stock, anything with a date column) and it **builds the dashboard itself**, **compares the current
period with an earlier one**, and has an LLM **explain what changed in plain English** for people who are not data
specialists. It works in two modes:

| | Explore mode | Live mode |
|---|---|---|
| Data | upload a CSV or Excel file | files dropped into a folder, stored in a database (SQLite or PostgreSQL) |
| Dates | you pick two dates or two periods | a rolling window (last N days against the N days before) |
| Updates | when you click | on a schedule (a worker, or Airflow); the page checks for new snapshots on a timer |
| Extra | | "since the last refresh" comparison, staleness and failure warnings |

## What makes it different from a fixed dashboard

* **One engine, any dataset.** A column profiler decides what each column is (date, category, number, id). A short *spec*
  (written by an LLM or by simple rules) says which KPIs and charts to show. The same code ran unchanged on outpatient and
  ICU data.
* **Python does all the arithmetic; the LLM only writes words.** Every number in an LLM summary is checked against the
  numbers Python computed. One invented figure and the summary is replaced by an automatic one.
* **Five ways to compare two periods:** side by side, biggest movers, mix shift (with a statistical check that separates
  real shifts from noise), contribution waterfall, and a sortable table.
* **Filters change the story, not just the chart.** Picking a gender or age group recomputes every number, finding and
  summary for that slice, and the summary button explains exactly the view on screen.
* **It degrades gracefully.** No API key, a retired model name, a malformed LLM reply, a bad file or a failed refresh each
  fall back to something sensible and say why, instead of crashing.

## Quick start

You need Python 3.10 or newer (tested on 3.12).

```
git clone <your repository url>
cd HealthDash-FYP
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_fake_data.py      # makes sample OPD and ICU data in data/generated/
python -m pytest tests -q                 # should end with no failures
streamlit run app/streamlit_app.py        # opens the app in your browser
```

In the app, pick `opd_visits.csv` under "...or use a sample file". That is **Explore mode**. Switch **Mode** at the top of
the sidebar to **Live (database)** for live mode (see below).

Without the app, one command builds a dashboard file:

```
python -m src.pipeline data/generated/icu_admissions.csv --out data/output/icu.html
```

Open `data/output/icu.html` in a browser. Useful options: `--spec specs/opd_example.json` (use a given layout),
`--filter gender=Female`, `--current 2026-10-01 2026-10-31 --baseline 2026-09-01 2026-09-30`, `--llm` (let the LLM design
the layout), `--narrate` (let the LLM write the summary).

## Live mode

Live mode keeps a database and refreshes it on a schedule. SQLite works with no setup; PostgreSQL is one setting away.

```
mkdir data\incoming\opd                                  # macOS/Linux: mkdir -p data/incoming/opd
copy data\generated\opd_visits.csv data\incoming\opd\week1.csv
python -m src.ingest                                     # load the file into the database
python -m src.live opd --window 30d                      # build a dashboard snapshot
python -m src.worker --every 30m                         # ...or do both automatically, forever
```

The subfolder name is the dataset name. A file is loaded once (even if renamed), a repeated row is stored once, and a file
that does not fit its dataset is rejected with a reason. Then run the app, choose **Live (database)**, and drop a second
file into the folder: the next snapshot shows what changed.

**PostgreSQL:** `docker compose up -d db`, then put `DATABASE_URL=postgresql+psycopg2://healthdash:healthdash@localhost:5432/healthdash`
in `.env`. The step-by-step guide, troubleshooting and the Airflow DAG are in [`docs/project_reference.md`](docs/project_reference.md).

## Using an LLM (optional)

Everything works without one. With one, the LLM designs the dashboard layout (once per dataset) and writes the summaries.

1. Copy `.env.example` to `.env` (never commit `.env`).
2. Get a key from Google AI Studio and set `GEMINI_API_KEY=` in `.env`. OpenAI and a local Ollama model also work: set
   `LLM_PROVIDER` and the matching settings (see `.env.example`).
3. Model names are retired regularly. If a call fails with a 404, run `python -m src.llm models` to see which names your key
   can use, then set `GEMINI_MODEL` in `.env`.

The LLM receives summaries only (counts, percentages, category names), never patient rows. Category names can still be
sensitive, so check before using a cloud LLM on real data.

## Configuration (`.env`)

| Setting | Meaning | Default |
|---|---|---|
| `DATABASE_URL` | Where live mode stores data | a local SQLite file, `data/healthdash.db` |
| `TEST_DATABASE_URL` | Also run the database tests on PostgreSQL (each test uses a throwaway schema; your data is safe) | off |
| `LLM_PROVIDER` | `gemini`, `openai` or `ollama` | `gemini` |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Gemini settings | none, `gemini-3.8-flash` |
| `HEALTHDASH_SCHEDULE`, `HEALTHDASH_WINDOW`, `HEALTHDASH_USE_LLM` | Airflow DAG settings | every 30 minutes, `7d`, off |

## How it works

```
 file or database rows
        |
   cleaning        validate columns, fix dates, drop duplicates, label blanks        src/cleaning.py
        |
   profiler        which columns are dates, categories, numbers, ids                 src/profiler.py
        |
   spec            which KPIs and charts to show (LLM or rules, validated)            src/spec_generator.py
        |
   aggregator      the numbers for both periods, with full-period totals              src/aggregator.py, compare.py
        |
   insights        ranked findings: KPI changes, category shifts, spikes, data gaps   src/insights.py
        |
   summariser      plain-English summary, checked against the findings                src/summariser.py
        |
   renderer        one self-contained HTML dashboard                                  src/renderer.py, templates/
```

Live mode wraps the same engine: `src/ingest.py` loads files into `src/db.py`, `src/live.py` builds a snapshot per refresh,
and `src/worker.py` (or `airflow/dags/dashboard_dag.py`) repeats that on a schedule.

## Project layout

```
app/            Streamlit app (streamlit_app.py) and the Live page (live_view.py)
airflow/dags/   the Airflow DAG
docs/           project_reference.md: setup guide, running live mode, troubleshooting, command cheat sheet
scripts/        generate_fake_data.py: sample data with planted changes
specs/          example dashboard layouts (opd, icu)
src/            the engine
templates/      dashboard.html, the page every dashboard is rendered into
tests/          the test suite
docker-compose.yml   PostgreSQL for live mode
```

## Testing

```
python -m pytest tests -q
```

The suite generates its own sample data and checks the engine against facts planted in that data (a visit spike, a rising
condition, changing pain scores), so the tests show that the dashboard *finds what is really there*. Database tests run on
SQLite, and also on PostgreSQL when `TEST_DATABASE_URL` is set. The run should be identical in any timezone.

## Known limitations

* "Live" means a refresh every N minutes (micro-batch), not streaming.
* The sample data is synthetic. Real patient data brings consent, de-identification and security obligations that this
  project does not implement.
* The LLM summary check catches invented numbers, but not a real number used in the wrong place.
* The dashboard page has been tested headlessly and by reading its code, not by automated browser tests.
* Airflow has been run end to end with a pip installation; running it inside Docker follows Airflow's own guide and is
  described, not tested, here.