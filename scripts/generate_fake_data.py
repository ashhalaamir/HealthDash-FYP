"""Generate fake OPD and ICU datasets (3 months) with deliberately planted changes.

Run from the repo root:  python scripts/generate_fake_data.py

Writes to data/generated/:
  opd_visits.csv, icu_admissions.csv   clean datasets
  opd_visits_dirty.csv                 same data plus duplicates, blanks, mixed date formats
  planted_facts.json                   ground truth measured from the data, for evaluation
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START, END = "2026-08-01", "2026-10-31"
OUT = Path(__file__).resolve().parents[1] / "data" / "generated"
rng = np.random.default_rng(SEED)


def age_group(ages):
    return pd.cut(ages, [0, 17, 39, 59, 120], labels=["Under 18", "18-39", "40-59", "60+"]).astype(str)


def make_patients(n):
    ages = np.clip(rng.normal(44, 18, n).round(), 3, 90).astype(int)
    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(1, n + 1)],
        "gender": rng.choice(["Male", "Female"], n, p=[0.63, 0.37]),
        "age": ages,
        "age_group": age_group(ages),
    })


def pick(options, weights, n):
    w = np.array(weights, dtype=float)
    return rng.choice(options, n, p=w / w.sum())


# ---------------------------------------------------------------- OPD
OPD_DIAG = {  # diagnosis -> (chief complaint, base weight)
    "LBP non-radiation": ("Lower back pain", 15), "Muscular back pain": ("Back pain", 9),
    "Lower back pain": ("Lower back pain", 7), "Muscular spasm": ("Back pain", 7),
    "Not recorded": ("Back pain", 7), "Shoulder dislocation": ("Shoulder pain", 6),
    "Bilateral knee OA": ("Knee pain", 5), "Sprain": ("Elbow pain", 5),
    "Mechanical wrist pain": ("Wrist pain", 4), "Morning stiffness": ("Neck pain", 3),
}


def make_opd():
    pts = make_patients(900)
    diags = list(OPD_DIAG)
    rows = []
    for day in pd.date_range(START, END):
        mean = 2.0 if day.weekday() >= 5 else 5.0   # quieter weekends
        if day == pd.Timestamp("2026-10-14"):
            mean = 28                                 # planted spike
        n = rng.poisson(mean)
        if n == 0:
            continue
        oct_ = day.month == 10
        w = [OPD_DIAG[d][1] * (2.0 if oct_ and d == "Lower back pain" else 1) for d in diags]  # planted shift
        w = [x * (0.3 if oct_ and d == "Not recorded" else 1) for x, d in zip(w, diags)]       # fewer blanks later
        diag = pick(diags, w, n)
        p = pts.sample(n, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        vas_mu = 1.3 if oct_ else 1.75                # planted: pain scores improve in October
        rows.append(pd.DataFrame({
            "visit_date": day.strftime("%Y-%m-%d"),
            "patient_id": p["patient_id"], "gender": p["gender"], "age": p["age"], "age_group": p["age_group"],
            "diagnosis": diag, "chief_complaint": [OPD_DIAG[d][0] for d in diag],
            "treatment": pick(["Meds", "Meds + I.A inj", "Meds + physio", "Meds + backslab"], [69, 13, 12, 6], n),
            "investigation": pick(["X-Ray", "X-Ray + Lab", "Lab", "None"], [77, 15, 5, 3], n),
            "follow_up_vas": np.clip(rng.normal(vas_mu, 0.8, n), 0, 10).round(1),
            "additional_visit": rng.random(n) < (0.12 if oct_ else 0.07),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "visit_id", [f"V{i:05d}" for i in range(1, len(df) + 1)])   # unique per row, like an encounter id
    df["additional_visit"] = df["additional_visit"].map({True: "Yes", False: "No"})
    return df


def make_dirty(df):
    d = df.copy()
    d = pd.concat([d, d.sample(15, random_state=1)], ignore_index=True)           # duplicates
    d.loc[d.sample(10, random_state=2).index, "visit_date"] = None                # blank dates
    d.loc[d.sample(12, random_state=3).index, "diagnosis"] = ""                   # blank categories
    idx = d.sample(10, random_state=4).index
    d.loc[idx, "visit_date"] = pd.to_datetime(d.loc[idx, "visit_date"]).dt.strftime("%d/%m/%Y")  # mixed format
    return d.sample(frac=1, random_state=5).reset_index(drop=True)


# ---------------------------------------------------------------- ICU
ICU_DIAG = {"Sepsis": 14, "Respiratory failure": 18, "Cardiac arrest": 8, "Post-surgical": 22,
            "Stroke": 10, "Trauma": 12, "DKA": 6}


def make_icu():
    pts = make_patients(500)
    diags = list(ICU_DIAG)
    rows = []
    for day in pd.date_range(START, END):
        surge = pd.Timestamp("2026-10-10") <= day <= pd.Timestamp("2026-10-20")   # planted sepsis surge
        n = rng.poisson(3.5)
        if n == 0:
            continue
        oct_ = day.month == 10
        w = [ICU_DIAG[d] * (3.0 if surge and d == "Sepsis" else 1) for d in diags]
        diag = pick(diags, w, n)
        p = pts.sample(n, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        vent_p = 0.42 if oct_ else 0.28                                           # planted: more ventilation
        los_mu = 6.5 if oct_ else 4.8                                             # planted: longer stays
        deceased = rng.random(n) < (np.where(diag == "Sepsis", 0.22, 0.09))
        rows.append(pd.DataFrame({
            "admit_date": day.strftime("%Y-%m-%d"),
            "patient_id": p["patient_id"], "gender": p["gender"], "age_group": p["age_group"],
            "diagnosis": diag,
            "ventilated": np.where(rng.random(n) < vent_p, "Yes", "No"),
            "length_of_stay_days": np.clip(rng.gamma(2.2, los_mu / 2.2, n), 1, 40).round(1),
            "apache_score": np.clip(rng.normal(18, 7, n), 3, 45).round().astype(int),
            "outcome": np.where(deceased, "Deceased", pick(["Discharged", "Transferred"], [80, 20], n)),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "admission_id", [f"A{i:05d}" for i in range(1, len(df) + 1)])
    return df


# ------------------------------------------------- ground truth for evaluation
def planted_facts(opd, icu):
    opd["m"] = opd["visit_date"].str[:7]
    icu["m"] = icu["admit_date"].str[:7]
    daily = opd.groupby("visit_date").size()
    sep_oct = lambda d, col, val: {m: int((g[col] == val).sum()) for m, g in d.groupby("m")}
    return {
        "opd": {
            "total_rows": len(opd),
            "busiest_day": daily.idxmax(), "busiest_day_visits": int(daily.max()),
            "avg_vas_by_month": opd.groupby("m")["follow_up_vas"].mean().round(2).to_dict(),
            "lower_back_pain_by_month": sep_oct(opd, "diagnosis", "Lower back pain"),
            "not_recorded_by_month": sep_oct(opd, "diagnosis", "Not recorded"),
        },
        "icu": {
            "total_rows": len(icu),
            "sepsis_by_month": sep_oct(icu, "diagnosis", "Sepsis"),
            "ventilated_share_by_month": (icu["ventilated"].eq("Yes").groupby(icu["m"]).mean().round(3).to_dict()),
            "avg_los_by_month": icu.groupby("m")["length_of_stay_days"].mean().round(2).to_dict(),
        },
    }


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    opd, icu = make_opd(), make_icu()
    facts = planted_facts(opd.copy(), icu.copy())
    opd.to_csv(OUT / "opd_visits.csv", index=False)
    make_dirty(opd).to_csv(OUT / "opd_visits_dirty.csv", index=False)
    icu.to_csv(OUT / "icu_admissions.csv", index=False)
    (OUT / "planted_facts.json").write_text(json.dumps(facts, indent=2))
    print(f"OPD rows: {len(opd)}  ICU rows: {len(icu)}")
    print(json.dumps(facts, indent=2))