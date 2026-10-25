"""HealthDash: upload data, pick two dates, filter, and get a dashboard with insights.

Run from the repo root:
    streamlit run app/streamlit_app.py

Cheap work (cleaning, aggregation, findings, template summary) re-runs on every change.
The LLM is only called when the Summarise button is pressed (and optionally for the AI-designed layout).
"""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.aggregator import cumulative_windows, previous_period
from src.filters import filter_options, filterable_columns
from src.llm import call_llm
from src.pipeline import default_periods, make_dashboard, prepare
from src.spec_generator import generate_spec

st.set_page_config(page_title="HealthDash", layout="wide")

# Let the dashboard use the whole page: no page padding, no top bar clutter, iframe as tall as the window.
st.markdown("""<style>
.block-container, [data-testid="stMainBlockContainer"]{padding:0!important;max-width:100%!important}
[data-testid="stMain"] [data-testid="stVerticalBlock"]{gap:0!important}
[data-testid="stHeader"]{background:transparent!important;height:0!important}
[data-testid="stToolbar"], .stAppDeployButton, #MainMenu, footer{display:none!important}
iframe{height:100vh!important;width:100%!important;border:0!important;display:block}
</style>""", unsafe_allow_html=True)
ss = st.session_state
ss.setdefault("file_id", 0)
ss.setdefault("sig", None)
ss.setdefault("specs", {})
ss.setdefault("llm_views", {})        # view key -> dashboard built with the AI summary
ss.setdefault("summary_cache", {})    # summariser cache: same findings = no second LLM call
nice = lambda c: c.replace("_", " ").strip().capitalize()

# ------------------------------------------------------------------ data
st.sidebar.title("HealthDash")
mode = st.sidebar.radio("Mode", ["Explore (upload a file)", "Live (database)"], key="mode")
if mode.startswith("Live"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from live_view import render_live
    render_live()
    st.stop()
st.sidebar.header("Data")
upload = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"], key="upload")
samples = {p.name: p for d in ("data/generated", "data/raw") for p in sorted((ROOT / d).glob("*.csv"))}
names = ["(none)"] + list(samples)
sample = st.sidebar.selectbox("...or use a sample file", names, index=names.index("opd_visits.csv") if "opd_visits.csv" in names else 0, key="sample")

if upload is not None:
    sig, label = ("upload", upload.name, upload.size), upload.name
elif sample != "(none)":
    sig, label = ("sample", sample), sample
else:
    st.info("Upload a CSV or Excel file with a date column, or pick a sample file in the sidebar.\n\n"
            "No sample files? Run `python scripts/generate_fake_data.py` first.")
    st.stop()

if ss.sig != sig:                                   # a new file: clean and profile it once
    try:
        if upload is not None:
            tmp = Path(tempfile.mkdtemp()) / upload.name
            tmp.write_bytes(upload.getvalue())
            source = tmp
        else:
            source = samples[sample]
        prep = prepare(source)
    except Exception as e:                          # DataValidationError messages are written for users
        st.error(f"Could not use this file: {e}")
        st.stop()
    ss.prep, ss.sig, ss.file_id = prep, sig, ss.file_id + 1
    ss.specs, ss.llm_views, ss.summary_cache = {}, {}, {}
prep, fid = ss.prep, ss.file_id
title = Path(label).stem

with st.sidebar.expander("Data checks"):
    r = prep.report
    st.write(f"Rows: {r['rows_in']} in, {r['rows_out']} kept. Duplicates removed: {r['duplicates_removed']}. "
             f"Unreadable dates dropped: {r['bad_dates_dropped']}.")
    if r["blanks_filled"]:
        st.write("Blank values labelled 'Not recorded': " + ", ".join(f"{nice(c)} ({n})" for c, n in r["blanks_filled"].items()))

# ------------------------------------------------------------------ dashboard design
spec_files = {f"Spec: {p.stem}": p for p in sorted((ROOT / "specs").glob("*.json"))}
design = st.sidebar.selectbox("Dashboard design", ["Automatic (rules)", "AI-designed (LLM)", *spec_files], key="design")
if (fid, design) not in ss.specs:
    if design in spec_files:
        ss.specs[(fid, design)] = (json.loads(spec_files[design].read_text()), "file", [])
    else:
        with st.spinner("Designing the dashboard..."):
            spec_, info = generate_spec(prep.profile, prep.df, title, call_llm if design.startswith("AI") else None)
        ss.specs[(fid, design)] = (spec_, info["source"], info["warnings"])
spec, spec_source, spec_notes = ss.specs[(fid, design)]

# ------------------------------------------------------------------ dates
st.sidebar.header("Compare")
dmin, dmax = prep.df[prep.date_col].min().date(), prep.df[prep.date_col].max().date()
if dmin >= dmax:
    st.warning("The data covers a single day, so there is nothing to compare. Upload a file that spans several days.")
    st.stop()
mode = st.sidebar.radio("Comparison", ["Two dates (everything up to each date)", "Two periods (current vs the one before)"], key=f"mode_{fid}")
if mode.startswith("Two dates"):
    b, c = st.sidebar.slider("Baseline date and current date", dmin, dmax, (dmin + (dmax - dmin) // 3, dmax), key=f"pair_{fid}")
    current, baseline = cumulative_windows(prep.df, prep.date_col, b.isoformat(), c.isoformat())
else:
    (cs, ce), _ = default_periods(prep)
    cs, ce = max(dmin, date.fromisoformat(cs)), date.fromisoformat(ce)
    s, e = st.sidebar.slider("Current period", dmin, dmax, (cs, ce), key=f"period_{fid}")
    current = (s.isoformat(), e.isoformat())
    baseline = previous_period(*current)

# ------------------------------------------------------------------ filters
st.sidebar.header("Filters")
cols = filterable_columns(prep.profile, spec)
options = filter_options(prep.df, cols)
selected = {col: st.sidebar.multiselect(nice(col), options[col], key=f"f_{col}_{fid}") for col in cols}
active = {k: v for k, v in selected.items() if v}

st.sidebar.header("Summary")
go = st.sidebar.button("Summarise this view (AI)", type="primary", key="summarise",
                       help="Asks the LLM to write a summary of exactly this filter and date selection.")

# ------------------------------------------------------------------ build
view_key = json.dumps([fid, design, current, baseline, active], sort_keys=True)
build = dict(current=current, baseline=baseline, spec=spec, title=title, filters=active)
try:
    if go:
        with st.spinner("Writing the summary..."):
            ss.llm_views[view_key] = make_dashboard(prep, llm=call_llm, narrate=True, cache=ss.summary_cache, **build)
    out = ss.llm_views.get(view_key) or make_dashboard(prep, **build)
except ValueError as e:
    st.error(str(e))
    st.stop()

ins = out["insights"]
st.sidebar.caption(f"Design: {spec_source} | Summary: {'AI' if ins['source'] == 'llm' else 'automatic'} | "
                   f"{len(prep.df):,} records after cleaning")
for f in ins["findings"]:                           # a tiny baseline or view deserves a warning next to the controls
    if f["kind"] in ("baseline_small", "small_sample"):
        st.sidebar.warning(f["text"])
        break
if ins["notes"]:
    st.sidebar.warning(" ".join(ins["notes"]))
if spec_notes or out["warnings"]:
    with st.sidebar.expander("Notes about this dashboard"):
        for n in dict.fromkeys(spec_notes + out["warnings"]):
            st.write(f"- {n}")
st.sidebar.download_button("Download this dashboard (HTML)", out["html"], file_name=f"{title}_dashboard.html", mime="text/html")

# st.components.v1.html is deprecated; st.iframe replaces it. The CSS above stretches the frame to the window height.
# The page writes every LLM-derived string with textContent, so the model's text cannot become markup.
if hasattr(st, "iframe"):
    st.iframe(out["html"], width="stretch", height=1000)
else:                                                        # older Streamlit without st.iframe
    import streamlit.components.v1 as components
    components.html(out["html"], height=1000, scrolling=True)