"""Rule-based dashboard spec built from a profile. Used when no LLM is available or its answer is
unusable, and as the safe baseline to compare LLM specs against."""

nice = lambda col: col.replace("_", " ").strip().capitalize()


def make_fallback_spec(profile, title="Dashboard"):
    cats = [c for c in profile["columns"] if c["role"] == "category"]
    nums = [c for c in profile["columns"] if c["role"] == "numeric"]
    ids = [c for c in profile["columns"] if c["role"] == "id" and 1 < c["distinct"] < profile["rows"]]
    w = [{"id": "rows", "type": "kpi", "label": "Total records", "agg": "count"}]

    if ids:  # the id that repeats (e.g. patient_id) is the entity worth counting
        e = min(ids, key=lambda c: c["distinct"])
        w.append({"id": "uniq", "type": "kpi", "label": f"Unique {nice(e['name']).lower()}s", "agg": "nunique", "column": e["name"]})
    for c in nums[:2]:
        w.append({"id": f"avg_{c['name']}", "type": "kpi", "label": f"Average {nice(c['name']).lower()}", "agg": "mean", "column": c["name"]})

    w.append({"id": "trend", "type": "trend", "label": "Records over time", "agg": "count", "span": 12})
    for c in cats[:4]:
        small = c["distinct"] <= 4
        w.append({"id": f"by_{c['name']}", "type": "breakdown", "chart": "donut" if small else "bar",
                  "label": f"Records by {nice(c['name']).lower()}", "by": c["name"], "top": 8, "span": 4 if small else 6})

    wide = [c for c in cats if c["distinct"] >= 3]
    if wide and nums:
        w.append({"id": "combo", "type": "combo", "label": f"Records and average {nice(nums[0]['name']).lower()} by {nice(wide[0]['name']).lower()}",
                  "by": wide[0]["name"], "top": 8, "span": 12, "bar": {"agg": "count"},
                  "line": {"agg": "mean", "column": nums[0]["name"]}, "bar_label": "Records", "line_label": "Average"})
    filters = [c["name"] for c in cats if 2 <= c["distinct"] <= 15][:4]
    return {"title": title, "widgets": w, "filters": filters}