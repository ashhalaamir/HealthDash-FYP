"""Live page: the logic behind it (src/live.py) and the Streamlit page itself, driven headlessly."""
import json
import os
import time
import re
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select, update

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import ingest as ing
from src import live
from src.db import get_engine, pipeline_runs, snapshots, utcnow
from src.renderer import PAYLOAD_RE, render_html

GEN = ROOT / "data/generated"
pytestmark = pytest.mark.skipif(not (GEN / "opd_visits.csv").exists(), reason="run scripts/generate_fake_data.py first")


@pytest.fixture
def parts(tmp_path):
    df = pd.read_csv(GEN / "opd_visits.csv")
    a, b = tmp_path / "part1.csv", tmp_path / "part2.csv"
    df[df.visit_date <= "2026-10-13"].to_csv(a, index=False)
    df[df.visit_date > "2026-10-13"].to_csv(b, index=False)
    return a, b


@pytest.fixture
def seeded(engine, parts):
    """A database with the first half of the OPD data and one snapshot."""
    ing.ingest_file(engine, parts[0], "opd")
    live.run_once(engine, "opd", "30d")
    return engine


def count(engine, table):
    with engine.connect() as c:
        return c.execute(select(func.count()).select_from(table)).scalar()


def payload_of(html):
    return json.loads(PAYLOAD_RE.search(html).group(2).replace("<\\/", "</"))


def fake_llm(prompt):
    return json.dumps({"summary": "Visits and unique patients both increased."})


# ------------------------------------------------------------------ pipeline status
def test_status_before_any_run(engine):
    s = live.pipeline_status(engine, "opd")
    assert s["stale"] and not s["failed"] and s["message"] == "No refresh has run yet."


def test_status_healthy_then_stale_then_failed(seeded):
    ok = live.pipeline_status(seeded, "opd", 90)
    assert not ok["stale"] and ok["message"] == "Healthy." and ok["age_minutes"] < 5
    later = live.pipeline_status(seeded, "opd", 90, now=utcnow() + timedelta(hours=5))
    assert later["stale"] and "No successful refresh for 3" in later["message"]
    live.run_once(seeded, "opd", "7x")                                   # a failing run
    bad = live.pipeline_status(seeded, "opd", 90)
    assert bad["failed"] and bad["message"].startswith("The last refresh failed: ")
    assert not bad["stale"]                                              # the earlier success is still recent


def test_no_change_counts_as_a_successful_check(seeded):
    live.run_once(seeded, "opd", "30d")                                  # nothing new
    with seeded.begin() as c:
        c.execute(update(pipeline_runs).where(pipeline_runs.c.status == "ok").values(finished_at=utcnow() - timedelta(hours=9)))
    assert not live.pipeline_status(seeded, "opd", 90)["stale"]


# ------------------------------------------------------------------ history and options
def test_recent_files_and_snapshot_history(seeded, parts):
    ing.ingest_file(seeded, parts[1], "opd")
    live.run_once(seeded, "opd", "30d")
    files = live.recent_files(seeded, "opd")
    assert [f["filename"] for f in files] == ["part2.csv", "part1.csv"] and files[0]["status"] == "loaded"
    hist = live.snapshot_history(seeded, "opd")
    assert [h["id"] for h in hist] == [2, 1] and hist[0]["summary"] == "template" and hist[0]["findings"] > 0


def test_filter_options_come_from_the_layout_and_the_window(seeded):
    snap = live.latest_snapshot(seeded, "opd")
    opts = live.view_filters(seeded, "opd", snap)
    assert set(opts) <= set(snap["payload"]["spec"].get("filters", [])) and "gender" in opts
    assert sorted(opts["gender"]) == ["Female", "Male"]


# ------------------------------------------------------------------ views
def test_unfiltered_view_is_the_stored_snapshot_with_live_info(seeded):
    snap = live.latest_snapshot(seeded, "opd")
    v = live.view_dashboard(seeded, "opd", snap, stale_after=90)
    p = payload_of(v["html"])
    assert not v["filtered"] and p["live"]["snapshot_id"] == snap["id"] and p["live"]["updated_at"].endswith("Z")
    assert p["live"]["stale_minutes"] == 90 and "since_last" not in p          # the first snapshot has nothing to compare with
    assert p["insights"]["summary"] == snap["payload"]["insights"]["summary"]


def test_since_last_refresh_reaches_the_page(seeded, parts):
    ing.ingest_file(seeded, parts[1], "opd")
    live.run_once(seeded, "opd", "30d")
    p = payload_of(live.view_dashboard(seeded, "opd", live.latest_snapshot(seeded, "opd"))["html"])
    assert p["since_last"]["rows_new"] > 0 and isinstance(p["since_last"]["findings"], list)


def test_a_filtered_view_is_recomputed_for_that_slice(seeded):
    snap = live.latest_snapshot(seeded, "opd")
    whole = payload_of(live.view_dashboard(seeded, "opd", snap)["html"])
    women = live.view_dashboard(seeded, "opd", snap, {"gender": ["Female"]})
    p = payload_of(women["html"])
    assert women["filtered"] and p["insights"]["scope"] == "Gender: Female" and p["live"]["view"] == "filtered view"
    total = lambda x: x["data"]["widgets"]["rows"]["current"]["value"]               # the automatic layout's "Total records"
    assert 0 < total(p) < total(whole)
    assert p["data"]["widgets"]["by_gender"]["current"]["labels"] == ["Female"]
    assert "since_last" not in p and p["data"]["dates"] == whole["data"]["dates"]        # same windows as the snapshot


def test_ai_summary_on_demand_for_whole_and_filtered_views(seeded):
    snap = live.latest_snapshot(seeded, "opd")
    for flt in (None, {"gender": ["Male"]}):
        p = payload_of(live.view_dashboard(seeded, "opd", snap, flt, llm=fake_llm, narrate=True)["html"])
        assert p["insights"]["source"] == "llm" and p["insights"]["summary"].startswith("Visits and unique")
    def broken(_): raise RuntimeError("quota exceeded")
    p = payload_of(live.view_dashboard(seeded, "opd", snap, None, llm=broken, narrate=True)["html"])
    assert p["insights"]["source"] == "template" and "quota exceeded" in p["insights"]["notes"][0]


def test_renderer_passes_live_and_since_last_through():
    p = {"spec": {"title": "T", "widgets": []}, "data": {"dates": {}, "widgets": {}},
         "live": {"snapshot_id": 7}, "since_last": {"rows_new": 3, "findings": []}}
    out = payload_of(render_html(p))
    assert out["live"] == {"snapshot_id": 7} and out["since_last"]["rows_new"] == 3


def test_template_writes_the_new_panels_with_textcontent_only():
    html = (ROOT / "templates/dashboard.html").read_text(encoding="utf-8")
    seg = html[html.index("if (live){"): html.index("$('sinceList').appendChild(li);")]       # just the two new blocks
    assert "innerHTML" not in seg and "textContent" in seg and "since_last" in seg


def test_uploaded_bytes_are_ingested_once(engine, parts):
    data = parts[0].read_bytes()
    assert ing.ingest_bytes(engine, "upload.csv", data, "opd").status == "loaded"
    assert ing.ingest_bytes(engine, "again.csv", data, "opd").status == "skipped"


# ------------------------------------------------------------------ the page
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


@pytest.fixture
def db(tmp_path, monkeypatch, parts):
    for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    url = f"sqlite:///{(tmp_path / 'page.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    return get_engine(url)


def open_live():
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=90).run()
    return at.sidebar.radio(key="mode").set_value("Live (database)").run()


def page(at):
    return payload_of(at.get("iframe")[0].proto.srcdoc)


def test_page_with_an_empty_database_explains_what_to_do(db):
    at = open_live()
    assert not at.exception and any("no data yet" in i.value for i in at.info)


def test_page_with_data_but_no_snapshot_says_so(db, parts):
    ing.ingest_file(db, parts[0], "opd")
    at = open_live()
    assert not at.exception and any("no dashboard snapshot yet" in i.value for i in at.info)


def test_page_shows_the_latest_snapshot(db, parts):
    ing.ingest_file(db, parts[0], "opd"); live.run_once(db, "opd", "30d")
    at = open_live()
    assert not at.exception and page(at)["live"]["snapshot_id"] == 1 and not at.warning and not at.error
    assert "gender" in [m.key.split("_")[-1] for m in at.sidebar.multiselect]


def test_filter_recomputes_and_keeps_the_scope_visible(db, parts):
    ing.ingest_file(db, parts[0], "opd"); live.run_once(db, "opd", "30d")
    at = open_live()
    at.sidebar.multiselect(key="live_f_opd_gender").set_value(["Female"]).run()
    assert not at.exception and page(at)["insights"]["scope"] == "Gender: Female"


def test_refresh_now_loads_a_dropped_file_and_shows_what_changed(db, parts, tmp_path):
    inbox = tmp_path / "incoming" / "opd"; inbox.mkdir(parents=True)
    ing.ingest_file(db, parts[0], "opd"); live.run_once(db, "opd", "30d")
    (inbox / "part2.csv").write_bytes(parts[1].read_bytes())
    old = time.time() - 600
    os.utime(inbox / "part2.csv", (old, old))
    at = open_live()
    at.sidebar.multiselect(key="live_f_opd_gender").set_value(["Female"]).run()          # a filter chosen before the refresh
    at.sidebar.text_input(key="live_folder").set_value(str(tmp_path / "incoming")).run()
    at.sidebar.button(key="live_refresh").click().run()
    assert not at.exception and count(db, snapshots) == 2
    p = page(at)
    assert p["live"]["snapshot_id"] == 2 and p["insights"]["scope"] == "Gender: Female"   # new data, same slice
    at.sidebar.multiselect(key="live_f_opd_gender").set_value([]).run()
    assert page(at)["since_last"]["rows_new"] > 0


def test_summarise_button_without_a_key_falls_back_and_says_why(db, parts):
    ing.ingest_file(db, parts[0], "opd"); live.run_once(db, "opd", "30d")
    at = open_live()
    at.sidebar.button(key="live_summarise").click().run()
    ins = page(at)["insights"]
    assert not at.exception and ins["source"] == "template" and "template summary" in ins["notes"][0]


def test_summarise_button_with_a_working_llm(db, parts, monkeypatch):
    from src import llm
    monkeypatch.setattr(llm, "call_llm", fake_llm)
    ing.ingest_file(db, parts[0], "opd"); live.run_once(db, "opd", "30d")
    at = open_live()
    at.sidebar.button(key="live_summarise").click().run()
    assert page(at)["insights"]["source"] == "llm"
    at.sidebar.multiselect(key="live_f_opd_gender").set_value(["Male"]).run()
    assert page(at)["insights"]["source"] == "template"                    # a summary of another view must not linger


def test_stale_and_failed_pipelines_are_flagged_on_the_page(db, parts):
    ing.ingest_file(db, parts[0], "opd"); live.run_once(db, "opd", "30d")
    with db.begin() as c:
        c.execute(update(pipeline_runs).values(finished_at=utcnow() - timedelta(hours=5)))
    at = open_live()
    assert any("No successful refresh" in w.value for w in at.warning)
    live.run_once(db, "opd", "7x")
    at = open_live()
    assert any("The last refresh failed" in e.value for e in at.error)


def test_switching_back_to_explore_mode(db):
    at = open_live()
    at.sidebar.radio(key="mode").set_value("Explore (upload a file)").run()
    assert not at.exception and "Dashboard design" in [s.label for s in at.sidebar.selectbox]