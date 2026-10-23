# HealthDash-FYP

HealthDash turns outpatient (OPD), intensive care (ICU), or other date-based CSV and Excel data into an interactive dashboard. It cleans and profiles the data, compares dates or periods, supports filters, and highlights changes. An LLM can optionally generate dashboard designs and summaries.

Final Year Project for Fall 2026–Spring 2027 by Faraz Ansari and Ashhal Aamir, supervised by Dr. Tariq Mehmood.

## Requirements

- Python 3.10 or newer
- pip
- Git (if you are cloning the repository)

## Setup

Open a terminal in the project folder (the folder containing this README) and create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

On macOS or Linux, activate with:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Start the app

From the project folder, run:

```shell
python -m streamlit run app/streamlit_app.py
```

Streamlit prints a local URL in the terminal (usually [http://localhost:8501](http://localhost:8501)); open it in your browser. Keep the terminal open while using the app. To stop it, press `Ctrl+C` in that terminal.

## Try the included demo data

The generated demo CSVs are not included in Git. Create them locally from the project folder:

```shell
python scripts/generate_fake_data.py
```

This creates three months of reproducible OPD and ICU data in `data/generated/`, including:

- `opd_visits.csv` — clean outpatient visit records
- `opd_visits_dirty.csv` — outpatient data with examples of duplicates, missing values, and mixed date formats
- `icu_admissions.csv` — clean ICU admission records

Refresh the app. In the sidebar, select one of the CSVs under **“…or use a sample file”**. Sample files are generated locally and are ignored by Git.

## Use your own data

In the app sidebar, upload a `.csv`, `.xlsx`, or `.xls` file. The file needs a date column with values that can be read as dates. HealthDash cleans and profiles the data before displaying the dashboard. If a file cannot be loaded, check the error message and confirm it has usable dates and tabular data.

## Using the dashboard

1. Choose an uploaded file or a generated sample in the sidebar.
2. Choose **Automatic (rules)** for a dashboard design that does not require an AI provider. You can also select a bundled example spec or an AI-designed layout.
3. Choose **Two dates** to compare cumulative data up to two dates, or **Two periods** to compare a period with the previous period.
4. Apply optional filters in the sidebar. The dashboard updates to reflect the selected dates and filters.
5. Select **Summarise this view (AI)** to request an AI-written summary, or use the automatic insights without an AI service.
6. Use **Download this dashboard (HTML)** to save a standalone HTML copy of the current dashboard.

## Optional AI setup

The rule-based dashboard and insights work without an API key. AI-designed layouts and the AI summary button require a configured LLM provider.

To use Gemini, copy the example environment file and add your API key:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`, setting `GEMINI_API_KEY` to your key. The default provider is Gemini. The `.env` file is local configuration and must not be committed or shared.

Other supported providers are OpenAI and Ollama. Set `LLM_PROVIDER` to `openai` or `ollama` and configure the corresponding values shown in `.env.example`. For Ollama, make sure the Ollama service is running and the selected model is available locally. Provider/model availability can change; update the model setting in `.env` if a request reports an unavailable model.

## Run tests

With the virtual environment active and dependencies installed:

```shell
python -m pytest -q
```

## Project layout

- `app/streamlit_app.py` — Streamlit user interface
- `src/` — data preparation, analysis, insights, LLM integration, and rendering
- `scripts/generate_fake_data.py` — reproducible demo data generator
- `specs/` — example dashboard specifications
- `templates/dashboard.html` — dashboard HTML template
- `tests/` — automated tests
- `airflow/dags/` — Airflow DAG
