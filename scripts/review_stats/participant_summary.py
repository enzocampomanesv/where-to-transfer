"""
participant_summary.py  (review round 1: R1-M5, R2-5, R3-1)
===========================================================

Per-city participant statistics from the raw workshop database, for the
Supplementary dataset table. Writes AGGREGATES ONLY: the raw database holds
personal data (names, phone numbers) and is never released.

  * participants, votes, votes per participant (median, IQR)
  * within-rater consistency: share of pairs a participant judged more than once
    on which all their judgements agree
  * between-rater agreement: probability that two different participants who
    judged the same pair chose the same location (first judgement per participant)
  * demographics: share of women, age bands, education

Community workshop dates (planners' workshops and test votes excluded); Accra and
Tema share one session and are separated by date. The selection reproduces the
aggregated training tables to within 1% (Chimoio exactly).

  python participant_summary.py --raw-db ../../results/data/bulk_15052026.sqlite
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd

SESSIONS = {"ACC": "ACC_sel_200m_test_p3", "TEM": "ACC_sel_200m_test_p3", "NBO": "NBO_sel_200m",
            "KIS": "KIS_sel_200m", "BEI": "BEI_sel_200m", "CHI": "CHI_sel_200m", "MUM": "MUM_sel_200m"}
DATES = {"ACC": ["14/8/2024", "16/8/2024", "19/8/2024"],
         "TEM": ["21/8/2024", "22/8/2024", "23/8/2024"],
         "NBO": ["14/9/2025", "15/9/2025", "16/9/2025", "17/9/2025"],
         "KIS": ["29/1/2025", "30/1/2025", "31/1/2025"],
         "BEI": ["26/2/2025", "27/2/2025"],
         "CHI": ["11/2/2025", "13/2/2025"],
         "MUM": ["25/2/2026", "26/2/2026", "27/2/2026"]}


def city_stats(v, users, city):
    d = v[(v["session"] == SESSIONS[city]) & (v["date"].isin(DATES[city]))].copy()
    w, l = d["wins"].astype(int), d["loose"].astype(int)
    d["a"], d["b"] = np.minimum(w, l), np.maximum(w, l)
    d["left_won"] = w == d["a"]
    per = d.groupby("uid").size()
    rep = d.groupby(["uid", "a", "b"])["left_won"].agg(["size", "mean"])
    rep = rep[rep["size"] > 1]
    first = d.drop_duplicates(["uid", "a", "b"])
    gb = first.groupby(["a", "b"])["left_won"].agg(["size", "mean"])
    gb = gb[gb["size"] > 1]
    agree = ((gb["mean"] ** 2 + (1 - gb["mean"]) ** 2) * gb["size"] - 1) / (gb["size"] - 1)
    u = users[users["uid"].isin(d["uid"].unique())]
    row = {"city": city, "workshop_dates": "; ".join(DATES[city]), "participants": d["uid"].nunique(),
           "votes": len(d), "votes_per_participant_median": per.median(),
           "votes_per_participant_q1": per.quantile(0.25), "votes_per_participant_q3": per.quantile(0.75),
           "pairs_judged_twice_by_same_participant": len(rep),
           "within_rater_consistency": rep["mean"].isin([0, 1]).mean(),
           "pairs_judged_by_several_participants": len(gb),
           "between_rater_agreement": np.average(agree, weights=gb["size"] * (gb["size"] - 1) / 2),
           "women_share": (u["u_sex"] == "f").mean()}
    u = u.assign(u_age=u["u_age"].replace({"0-14": "15-29"}))  # one entry slip, corrected by the authors; no participant was under 18
    for col, prefix in [("u_age", "age"), ("u_studies", "education")]:
        for k, share in u[col].fillna("not given").value_counts(normalize=True).items():
            row[f"{prefix}: {k}"] = share
    return row


def main(args):
    con = sqlite3.connect(f"file:{args.raw_db}?mode=ro", uri=True)
    v = pd.read_sql("select session, date, uid, wins, loose from votes", con)
    users = pd.read_sql("select uid, u_sex, u_age, u_studies from users", con)   # no names or phone numbers
    con.close()
    out = pd.DataFrame([city_stats(v, users, c) for c in DATES]).fillna(0)
    os.makedirs(args.out_dir, exist_ok=True)
    out.to_csv(os.path.join(args.out_dir, "participant_summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-db", default="../../results/data/bulk_15052026.sqlite")
    ap.add_argument("--out-dir", default="../../results/review_stats")
    main(ap.parse_args())
