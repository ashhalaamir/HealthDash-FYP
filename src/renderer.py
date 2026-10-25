"""Step 4 of the pipeline: put a payload into templates/dashboard.html and save the result."""
import json
import re
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "dashboard.html"
PAYLOAD_RE = re.compile(r'(<script id="payload" type="application/json">)(.*?)(</script>)', re.S)


def render_html(payload, template_path=TEMPLATE):
    """The standalone dashboard page as a string. `payload` needs "spec" and "data"; "insights" is optional."""
    html = Path(template_path).read_text(encoding="utf-8")
    if not PAYLOAD_RE.search(html):
        raise ValueError("Template has no payload block (<script id=\"payload\" type=\"application/json\">).")
    keys = [k for k in ("spec", "data", "insights", "since_last", "live") if payload.get(k) is not None]
    body = json.dumps({k: payload[k] for k in keys}, ensure_ascii=False)
    body = body.replace("</", "<\\/")          # data must never be able to close the script tag
    html = PAYLOAD_RE.sub(lambda m: m.group(1) + "\n" + body + "\n" + m.group(3), html, count=1)
    title = payload["spec"].get("title") or "Dashboard"
    html = re.sub(r"<title>.*?</title>", lambda _: f"<title>{title}</title>".replace("&", "&amp;"), html, count=1)
    return html


def render_dashboard(payload, out_path, template_path=TEMPLATE):
    """Write the dashboard to a file and return its path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(payload, template_path), encoding="utf-8")
    return out_path