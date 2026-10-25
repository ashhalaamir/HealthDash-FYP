"""Live mode page for the Streamlit app (called from streamlit_app.py when Mode = Live).

Everything here only READS snapshots that the worker or Airflow builds, except two buttons:
  Refresh now   runs one worker cycle in-process (scan the watch folder, then refresh every dataset)
  Save to database   puts an uploaded file into the database
The dashboard panel re-checks for a newer snapshot on a timer and only redraws when one exists.
"""
import json
import os
import re
from datetime import datetime

import streamlit as st

from src.db import DEFAULT_URL, dataset_summary, get_engine
from src.ingest import INCOMING, ingest_bytes
from src.live import latest_snapshot, pipeline_status, recent_files, snapshot_history, view_dashboard, view_filters
from src.worker import cycle, refresh_step

AUTO = {"Off": None, "10 seconds": 10, "30 seconds": 30, "1 minute": 60, "5 minutes": 300}
nice = lambda c: c.replace("_", " ").strip().capitalize()


@st.cache_resource
def _engine(url):
    return get_engine(url)


def render_live():
    from src.llm import call_llm                    # looked up at call time so tests can substitute it
    ss, sb = st.session_state, st.sidebar
    ss.setdefault("live_llm_views", {})
    ss.setdefault("live_cache", {})
    ss.setdefault("live_msg", "")
    engine = _engine(os.environ.get("DATABASE_URL") or DEFAULT_URL)

    sb.header("Live data")
    names = [s["dataset"] for s in dataset_summary(engine)]
    ds = sb.selectbox("Dataset", names, key="live_dataset") if names else None
    folder = sb.text_input("Watch folder", str(INCOMING), key="live_folder")
    window = sb.selectbox("Window for new snapshots", ["1d", "7d", "14d", "30d"], index=1, key="live_window")
    ai = sb.checkbox("AI summaries (needs an API key)", value=False, key="live_ai")
    llm = call_llm if ai else None
    if sb.button("Refresh now", key="live_refresh", help="Load new files from the watch folder and build new snapshots."):
        with st.spinner("Refreshing..."):
            ss["live_msg"] = cycle(engine, folder, window, llm=llm).line()
        st.rerun()

    with sb.expander("Add a file to the database"):
        up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], key="live_upload")
        name = re.sub(r"[^a-z0-9_]+", "_", st.text_input("Dataset name", value=ds or "", key="live_dsname").strip().lower()).strip("_")
        if st.button("Save to database", key="live_save", disabled=up is None or not name):
            r = ingest_bytes(engine, up.name, up.getvalue(), name)
            note = r.line()
            if r.status == "loaded":
                note += " | " + " ; ".join(x.line() for x in refresh_step(engine, window, llm, [name]))
            ss["live_msg"] = note
            st.rerun()
    if ss["live_msg"]:
        sb.caption(ss["live_msg"])

    if ds is None:
        st.info("The database has no data yet. Put a CSV or Excel file in a dataset folder inside the watch folder "
                "(for example `data/incoming/opd/week1.csv`) and press **Refresh now**, or use **Add a file to the database**.")
        st.stop()
    snap = latest_snapshot(engine, ds)
    if snap is None:
        st.info(f"'{ds}' has data but no dashboard snapshot yet. Press **Refresh now** in the sidebar.")
        st.stop()

    sb.header("Filters")
    options = view_filters(engine, ds, snap)
    selected = {}
    for col, values in options.items():
        key = f"live_f_{ds}_{col}"
        if ss.get(key):                              # a value may have left the window since it was chosen
            ss[key] = [v for v in ss[key] if v in values]
        selected[col] = sb.multiselect(nice(col), values, key=key)
    active = {c: v for c, v in selected.items() if v}

    sb.header("Summary")
    go = sb.button("Summarise this view (AI)", key="live_summarise",
                   help="Asks the LLM to write a summary of exactly this filter selection.")
    sb.header("Page")
    auto = sb.selectbox("Check for new snapshots", list(AUTO), index=2, key="live_auto")
    stale_after = sb.number_input("Warn if no refresh for (minutes)", min_value=5, max_value=10080, value=90, key="live_stale")
    with sb.expander("Recent file loads"):
        files = recent_files(engine, ds)
        st.dataframe([{"file": f["filename"], "status": f["status"], "new rows": f["rows_new"], "duplicates": f["rows_duplicate"],
                       "rejected": f["rows_rejected"], "note": f["message"] or "", "when (UTC)": f["ingested_at"]} for f in files]
                     or [{"file": "(none yet)"}], hide_index=True)
    with sb.expander("Snapshot history"):
        hist = snapshot_history(engine, ds)
        st.dataframe([{"snapshot": h["id"], "when (UTC)": h["created_at"], "window": h["window"],
                       "summary": h["summary"], "findings": h["findings"]} for h in hist], hide_index=True)

    @st.fragment(run_every=AUTO[auto])
    def panel():
        now = latest_snapshot(engine, ds)            # newest snapshot at every tick
        status = pipeline_status(engine, ds, stale_after)
        if status["failed"]:
            st.error(status["message"])
        elif status["stale"]:
            st.warning(status["message"])
        key = json.dumps([ds, now["id"], active], sort_keys=True)
        if go:
            with st.spinner("Writing the summary..."):
                ss["live_llm_views"][key] = view_dashboard(engine, ds, now, active, llm=call_llm, narrate=True,
                                                          cache=ss["live_cache"], stale_after=stale_after)
        view = ss["live_llm_views"].get(key) or view_dashboard(engine, ds, now, active, stale_after=stale_after)
        st.iframe(view["html"], width="stretch", height=1000)

    panel()