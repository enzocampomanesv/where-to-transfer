"""
t1_overlap_percentiles.py  (review round 1: R1-M7)
==================================================

Recomputes the Tier 1 overlap ratio (t1_metrics.compute_overlap_ratio) with the
coverage threshold at the 90th, 95th (paper) and 99th percentile of within-source
k-NN distances, for k = 10 and 20, on the same embedding parquets. CPU only, a few
minutes. The output feeds scripts/review_stats/coverage_thresholds.py
(--percentile-csv).

Run on CRIB from scripts/finetune_experiments/t1/:

  PYTHONNOUSERSITE=1 python t1_overlap_percentiles.py

Writes {output_dir}/overlap_percentiles.csv with columns source_city, target_city,
overlap_p{90,95,99}_k{10,20} and overlap_p{90,95,99} (mean over k, as in the paper).
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from t1_metrics import CITY_CODES, K_VALUES, TRANSFER_PATH, compute_overlap_ratio

PERCENTILES = [90.0, 95.0, 99.0]


def main(args):
    emb_dir = Path(args.embeddings_dir)
    data = {}
    for p in sorted(emb_dir.glob("embeddings_*.parquet")):
        src = p.stem.replace("embeddings_", "").upper()
        if src in CITY_CODES:
            data[src] = pd.read_parquet(p)
    if not data:
        raise FileNotFoundError(f"No embedding parquets in {emb_dir}")

    rows = []
    for src, tgt in product(CITY_CODES, CITY_CODES):
        if src == tgt or src not in data:
            continue
        df = data[src]
        cols = sorted([c for c in df.columns if c.startswith("emb_")], key=lambda c: int(c.split("_")[1]))
        s = df.loc[df["city"] == src, cols].values.astype("float32")
        t = df.loc[df["city"] == tgt, cols].values.astype("float32")
        row = {"source_city": src, "target_city": tgt}
        for pct in PERCENTILES:
            for k in K_VALUES:
                row[f"overlap_p{pct:.0f}_k{k}"] = compute_overlap_ratio(s, t, k=k, coverage_percentile=pct)
            row[f"overlap_p{pct:.0f}"] = sum(row[f"overlap_p{pct:.0f}_k{k}"] for k in K_VALUES) / len(K_VALUES)
        print(src, tgt, "  ".join(f"p{pct:.0f}={row[f'overlap_p{pct:.0f}']:.3f}" for pct in PERCENTILES))
        rows.append(row)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "overlap_percentiles.csv", index=False)
    print(f"\n-> {out / 'overlap_percentiles.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings_dir", default=f"{TRANSFER_PATH}/results/tier1/embeddings/")
    ap.add_argument("--output_dir", default=f"{TRANSFER_PATH}/results/tier1/metrics/")
    main(ap.parse_args())
