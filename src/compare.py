"""Comparison maths behind the dashboard's 'Change over time' views.

compare_rows(current, baseline) takes {category: value} for both periods and returns one row per
category: change, % change, share of total in each period, the share shift in percentage points, and
whether that shift is statistically notable. Shares use each period's own total, so a busier month with
the same mix shows no shift. Notable = two-proportion z-test with a Bonferroni correction for the number
of categories shown, so a chart with many bars does not light up by chance alone.
"""
from math import sqrt
from statistics import NormalDist

MIN_COMBINED = 10     # fewer than this many records across both periods is never called notable


def compare_rows(cur, base, counts=True, alpha=0.05, totals=None):
    labels = list(cur)
    c, b = dict(cur), {l: base.get(l, 0) for l in labels}
    t = totals or {}          # full-period totals, so categories hidden by a top-N cut still count
    tc, tb = t.get("current", sum(c.values())), t.get("baseline", sum(b.values()))
    cut = NormalDist().inv_cdf(1 - alpha / (2 * max(len(labels), 1)))
    rows = []
    for l in labels:
        x, y = c[l], b[l]
        row = {"label": l, "current": x, "baseline": y, "delta": round(x - y, 2),
               "pct": round((x - y) / abs(y) * 100, 1) if y else None,
               "share_cur": None, "share_base": None, "shift_pp": None, "z": None, "significant": False}
        if counts and tc and tb:
            pc, pb, pooled = x / tc, y / tb, (x + y) / (tc + tb)
            se = sqrt(pooled * (1 - pooled) * (1 / tc + 1 / tb))
            z = (pc - pb) / se if se else 0.0
            row.update(share_cur=round(pc * 100, 1), share_base=round(pb * 100, 1), shift_pp=round((pc - pb) * 100, 1),
                       z=round(z, 2), significant=bool(abs(z) >= cut and x + y >= MIN_COMBINED))
        rows.append(row)
    return rows