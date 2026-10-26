# HealthDash project overview

This document describes how the project currently works, including its Python
engine, Streamlit pages, and HTML dashboard template. It is an architectural
companion to the setup and operating instructions in
[project_reference.md](./project_reference.md).

## What the project does

HealthDash turns dated clinical or operational records into an interactive
dashboard. It can analyze an uploaded file for an on-demand comparison
(**Explore mode**) or keep records in a database and build repeatable dashboard
snapshots (**Live mode**). Both modes use the same cleaning, profiling,
aggregation, comparison, insight, summary, and rendering components.

The LLM is optional. Without it, HealthDash uses a rule-based dashboard layout
and generates summaries from its findings. With it, the model can propose a
layout and write a narrative summary; it does not replace the Python analysis.

## Main runtime paths

### Explore mode

The app entry point is `app/streamlit_app.py`. Its main flow is:

1. The user uploads a CSV/Excel file or selects a generated sample file.
2. `src/pipeline.py` calls `prepare`, which reads the file, cleans it, finds
   the date column, and profiles the dataset.
3. The user chooses a dashboard design, comparison dates or periods, and
   optional filters. A design can come from a JSON file in `specs/`, the
   rule-based generator, or an LLM proposal.
4. `make_dashboard` applies filters before calculating the view, builds
   current/baseline data, derives findings, and creates a summary.
5. `src/renderer.py` embeds the resulting payload in `templates/dashboard.html`.
   Streamlit displays the generated HTML in an iframe and offers it for
   download.

The app stores the prepared dataset and generated designs in Streamlit session
state so ordinary control changes do not reload and clean the source file.
Expensive AI narration is requested only when the user presses the summary
button.

The same pipeline can run without Streamlit:

```text
python -m src.pipeline <input-file> --out <output-html>
```

### Live mode

Live mode is selected from the sidebar in `app/streamlit_app.py`; it delegates
the page to `app/live_view.py`. It reads database snapshots rather than
rebuilding a dashboard on every page render. The two page actions that do
write data are **Refresh now** and **Save to database**.

The scheduled data path is:

```text
data/incoming/<dataset>/<file>
        -> src/ingest.py
        -> src/db.py
        -> src/worker.py or airflow/dags/dashboard_dag.py
        -> src/live.py
        -> database snapshot
        -> app/live_view.py
```

`src/ingest.py` infers the dataset name from the incoming subfolder, applies
the shared cleaning rules, and records accepted rows. File-content hashes
prevent reprocessing the same file; row hashes prevent storing a duplicate row
within a dataset. The first file establishes the dataset's date and column
shape. Later files must match that shape (additional columns are ignored);
unreadable-date rows are recorded as rejected, and file-level failures are
logged.

`src/worker.py` runs ingestion and refresh as separate steps. For each dataset,
`src/live.py` anchors the rolling current and baseline windows to the newest
record date, obtains a saved or generated spec, runs the shared dashboard
engine, and stores a snapshot and pipeline-run status. If the data, window, and
layout have not changed, it can skip the snapshot. When a previous snapshot
exists, it also records changes since that refresh.

The optional Airflow DAG defines those same ingest and refresh steps as
separate tasks. It is an alternative scheduler, not a second analytics
implementation; the worker and Airflow should not refresh the same database at
the same time.

On the Live page, the user selects a dataset and may filter the snapshot.
Filters cause the page to load the relevant rows from the snapshot's stored
date windows and recompute the dashboard, so the summary and findings describe
the selected slice. The page periodically checks for a newer snapshot and
displays refresh health, recent file loads, and snapshot history.

## Shared analysis pipeline

`src/pipeline.py` composes the analysis stages and is the common entry point
for the app and command-line workflow:

| Stage | Module | Responsibility |
|---|---|---|
| Load and clean | `src/cleaning.py` | Read CSV or Excel; normalize headers and dates; reject unusable data; label blank text values; remove duplicate rows; return a cleaning report. |
| Profile | `src/profiler.py` | Classify columns as dates, identifiers, categories, numeric values, or high-cardinality text; produce a compact, JSON-safe description. |
| Choose a layout | `src/spec_generator.py`, `src/fallback_spec.py` | Use a supplied or generated widget spec. The fallback builds a basic layout from profiled columns; LLM suggestions are validated and can fall back to rules. |
| Apply filters and calculate | `src/filters.py`, `src/aggregator.py` | Filter rows consistently, select date windows, validate widgets, and calculate KPIs, trends, category breakdowns, and combo charts. |
| Compare | `src/compare.py` | Calculate per-category changes, percentage changes, share changes, and a corrected statistical significance check for count-based mix shifts. |
| Find noteworthy changes | `src/insights.py` | Rank KPI changes, category shifts, spikes, small-sample warnings, top shares, and missing-value findings from the calculated payload. |
| Summarize | `src/summariser.py`, `src/llm.py` | Build a template summary or request an optional LLM summary. Reject summaries that are too long or contain unsupported numbers. |
| Render | `src/renderer.py`, `templates/dashboard.html` | Serialize the dashboard data into a standalone HTML page and optionally write it to disk. |

`src/aggregator.py` owns the widget data contract and date-window calculations.
It supports `kpi`, `trend`, `breakdown`, and `combo` widgets. A spec identifies
the widget type, measure or category, labels, and optional display settings;
`validate_spec` drops invalid widgets and reports warnings rather than letting
an unusable widget break the whole dashboard.

For a normal Explore dashboard, the default comparison is the last 30 days in
the data against the preceding 30 days. Users can instead select cumulative
dates or a custom current period. Live refreshes use a configured rolling
window and its immediately preceding, equal-length period.

## Layouts and LLM boundaries

Dashboard layouts are JSON objects containing a title, a list of widgets, and
optionally a list of filter columns. Example layouts are in `specs/`. The
fallback layout uses the profiled categories, numeric columns, and identifiers
to choose sensible initial widgets.

`src/llm.py` provides one interface for Gemini, OpenAI, and Ollama, configured
through environment variables. The layout prompt receives the column profile,
not source records. The summary prompt receives ranked findings and the
selected scope, not source records. Both generated layouts and summaries are
validated before use; a failed or unusable response produces a warning and
falls back to project-generated output.

## Database and supporting modules

`src/db.py` uses SQLAlchemy and creates the schema on engine initialization.
SQLite is the default; PostgreSQL can be selected through `DATABASE_URL`.
Live mode stores:

| Table | Purpose |
|---|---|
| `datasets` | Dataset name and its established date/column metadata. |
| `ingested_files` | Each processed file, its content hash, status, and row counts. |
| `records` | Deduplicated, cleaned records and their event dates. |
| `rejected_records` | Rows excluded from ingestion, with the rejection reason. |
| `dataset_specs` | The chosen layout retained for a dataset across refreshes. |
| `snapshots` | Dashboard payloads and findings hashes from successful refreshes. |
| `pipeline_runs` | Per-refresh status, timing, and error message. |

Other supporting modules include:

- `src/worker.py`: polling loop, one-shot cycle, interval parsing, and scheduler
  results.
- `src/live.py`: refresh orchestration, snapshot reads, filtered views, and
  pipeline health.
- `src/filters.py`: filter options, row selection, and readable filter scope.
- `src/insights.py`: shared findings logic used by both dashboard modes and
  refresh-to-refresh comparisons.
- `src/aggregator.py`: validates specs and produces the numeric payload consumed
  by the renderer.

## App and dashboard template

`app/streamlit_app.py` provides the shared page shell, Explore controls, file
selection, spec and date controls, filters, warnings, and dashboard download.
It routes Live mode to `app/live_view.py`, which contains database-specific
controls and embeds the selected snapshot.

`templates/dashboard.html` is the browser-facing dashboard. The renderer
replaces its JSON payload block with the current spec and calculated data. The
template uses Chart.js for chart drawing and JavaScript to build the title,
summary, findings, KPI cards, charts, and comparison controls. Its comparison
views include side-by-side values, biggest movers, mix shift, contribution,
and a sortable table. Live payloads can additionally show snapshot age and
changes since the previous refresh.

The template is intended to render the output of the Python engine; it is not a
separate data-processing backend. It loads Chart.js from a CDN, so rendering
charts in a browser requires access to that library.

## Project boundaries and related files

- `scripts/generate_fake_data.py` generates synthetic OPD/ICU example data.
- `specs/` contains example dashboard layouts.
- `airflow/dags/dashboard_dag.py` is the optional scheduler integration.
- `tests/` covers the pipeline, app behavior, ingestion, database, worker, and
  analytics components.
- `docs/project_reference.md` covers setup, operating live mode, commands, and
  troubleshooting.

Live mode is scheduled batch processing, not real-time streaming. The project
includes synthetic data and does not itself provide the approvals, controls,
or protections required to handle real patient information.
