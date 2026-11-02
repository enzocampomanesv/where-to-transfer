"""
phase3.py
=========

Turn a target city's completed run directories (raw per-patch BT in scores.csv)
into the comparison layer:

  1. discover runs from their manifests; identify the reference (source==target,
     mode==zeroshot). Hard-fail if the reference is absent.
  2. reference-ECDF normalise every run against the reference distribution.
  3. difference maps: variant_ecdf - reference_ecdf, in [-1, 1].
  4. full-city Spearman vs reference, per run (a summary table).
  5. subset-consistency QA: Spearman between a run's full-city BT (restricted to the
     clustering-subset patches) and the subset BT recomputed with that run's model.

The AoA/OOD overlay (item 6) is deferred to land with the v2 iso-forest port.

Normalisation and rasterisation reuse the already-tested normalize.py / rasterize.py.
The core (discover/normalise/difference/spearman/subset comparison) is testable on
synthetic score CSVs + real gpkgs; the subset-BT RECOMPUTE needs a model + tiles and
runs on infra (score_city over the human-vote pairs).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import normalize as nz
import rasterize as rz


# ===========================================================================
# Discovery
# ===========================================================================
def load_run(run_dir: str) -> Tuple[dict, pd.DataFrame]:
    """Load (manifest, scores_df) for a run directory."""
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    scores = pd.read_csv(os.path.join(run_dir, "scores.csv"))
    return manifest, scores


def is_reference(manifest: dict) -> bool:
    return (manifest.get("source") == manifest.get("target")
            and manifest.get("mode") == "zeroshot")


def discover_runs(runs_dir: str, target: str) -> Dict[str, object]:
    """Group runs for a target by reading manifests; separate reference from variants.

    Returns {'reference': (manifest, scores) | None, 'variants': [(manifest, scores), ...]}.
    """
    reference = None
    variants = []
    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        man_path = os.path.join(run_dir, "manifest.json")
        if not os.path.exists(man_path):
            continue
        with open(man_path) as f:
            manifest = json.load(f)
        if manifest.get("target") != target:
            continue
        if manifest.get("status") not in (None, "ok"):
            continue
        manifest, scores = load_run(run_dir)
        manifest["_run_dir"] = run_dir
        if is_reference(manifest):
            reference = (manifest, scores)
        else:
            variants.append((manifest, scores))
    return {"reference": reference, "variants": variants}


# ===========================================================================
# Core comparison (testable; no model/tiles)
# ===========================================================================
def _variant_label(manifest: dict) -> str:
    if manifest.get("mode") == "zeroshot":
        return f"{manifest['src_code']}__zeroshot"
    return (f"{manifest['src_code']}__{manifest.get('sampling')}"
            f"_{manifest.get('sample_size')}")


def full_city_spearman_table(
    reference_scores: pd.DataFrame,
    variants: List[Tuple[dict, pd.DataFrame]],
    *,
    key: str = "image_id",
    value_col: str = "bt_score",
) -> pd.DataFrame:
    """Spearman of each variant's full-city BT vs the reference, on shared patches."""
    ref = reference_scores[[key, value_col]].rename(columns={value_col: "ref"})
    rows = []
    for manifest, sc in variants:
        merged = sc[[key, value_col]].merge(ref, on=key, how="inner")
        rho, _ = spearmanr(merged[value_col].to_numpy(), merged["ref"].to_numpy())
        rows.append({
            "variant": _variant_label(manifest),
            "src_code": manifest.get("src_code"),
            "mode": manifest.get("mode"),
            "sampling": manifest.get("sampling"),
            "sample_size": manifest.get("sample_size"),
            "spearman_vs_reference": float(rho),
            "n_shared": int(len(merged)),
        })
    return pd.DataFrame(rows).sort_values("spearman_vs_reference",
                                          ascending=False).reset_index(drop=True)


def normalized_and_difference(
    reference_scores: pd.DataFrame,
    variant_scores: pd.DataFrame,
    *,
    key: str = "image_id",
    value_col: str = "bt_score",
) -> Dict[str, pd.DataFrame]:
    """Return reference-ECDF for the variant + its difference vs reference-ECDF.

    reference_ecdf is the reference scored against its own distribution (~uniform).
    difference = variant_ecdf - reference_ecdf per patch, aligned by key.
    """
    ref_ecdf = nz.normalize_map(reference_scores, reference_scores,
                                value_col=value_col, key=key, method="ecdf")
    var_ecdf = nz.normalize_map(variant_scores, reference_scores,
                                value_col=value_col, key=key, method="ecdf")
    ecdf_col = f"{value_col}_ecdf"
    diff = nz.difference_map(var_ecdf, ref_ecdf, value_col=ecdf_col, key=key,
                             out_col="difference")
    return {"variant_ecdf": var_ecdf, "reference_ecdf": ref_ecdf,
            "difference": diff, "ecdf_col": ecdf_col}


def subset_consistency(
    full_scores: pd.DataFrame,
    subset_bt: Dict[int, float],
    *,
    key: str = "image_id",
    value_col: str = "bt_score",
) -> Dict[str, float]:
    """Spearman between full-city BT (restricted to subset patches) and subset BT.

    ``subset_bt`` maps image_id -> BT score recomputed on the human-vote pairs with
    the SAME model. High (but not identical) is expected -- the two comparison graphs
    differ. A low value flags a scoring-pipeline problem.
    """
    sub = pd.DataFrame({"image_id": list(subset_bt.keys()),
                        "subset_bt": list(subset_bt.values())})
    merged = full_scores[[key, value_col]].merge(sub, left_on=key, right_on="image_id",
                                                 how="inner")
    if len(merged) < 3:
        return {"spearman": None, "n_shared": int(len(merged))}
    rho, _ = spearmanr(merged[value_col].to_numpy(), merged["subset_bt"].to_numpy())
    return {"spearman": float(rho), "n_shared": int(len(merged))}


# ===========================================================================
# Orchestration (rasterises via tested rasterize.py)
# ===========================================================================
def build_comparison(
    target: str,
    runs: Dict[str, object],
    gdf,
    out_dir: str,
    *,
    key: str = "image_id",
    value_col: str = "bt_score",
    rasterize_fn: Callable = rz.rasterize_frame,
    crs=None,
) -> Dict[str, object]:
    """Produce ECDF + difference GeoTIFFs per variant and a Spearman table.

    Requires runs['reference'] to be present (hard error otherwise).
    """
    if runs.get("reference") is None:
        raise RuntimeError(
            f"No reference run for target {target!r} (need source==target, "
            f"mode==zeroshot). Run it first: --source {target} --target {target} "
            f"--mode zeroshot.")
    ref_manifest, ref_scores = runs["reference"]
    variants = runs["variants"]
    os.makedirs(out_dir, exist_ok=True)

    # reference ECDF map (uniform by construction) -- rasterise once
    ref_ecdf = nz.normalize_map(ref_scores, ref_scores, value_col=value_col,
                                key=key, method="ecdf")
    rasterize_fn(ref_ecdf, gdf, out_dir, value_cols=[f"{value_col}_ecdf"],
                 basename=f"{target}__reference", crs=crs,
                 tags={"target": target, "kind": "reference_ecdf"})

    outputs = {"reference": f"{target}__reference"}
    per_variant = {}
    for manifest, sc in variants:
        label = _variant_label(manifest)
        nd = normalized_and_difference(ref_scores, sc, key=key, value_col=value_col)
        # ECDF map
        rasterize_fn(nd["variant_ecdf"], gdf, out_dir, value_cols=[nd["ecdf_col"]],
                     basename=f"{target}__{label}__ecdf", crs=crs,
                     tags={"target": target, "variant": label, "kind": "variant_ecdf"})
        # difference map (join difference back onto image_id for rasterizing)
        diff_frame = nd["difference"].rename(columns={"difference": "difference"})
        rasterize_fn(diff_frame, gdf, out_dir, value_cols=["difference"],
                     basename=f"{target}__{label}__diff", crs=crs,
                     tags={"target": target, "variant": label, "kind": "difference",
                           "range": "[-1,1]"})
        per_variant[label] = {"ecdf": f"{target}__{label}__ecdf",
                              "diff": f"{target}__{label}__diff"}

    # Spearman table
    table = full_city_spearman_table(ref_scores, variants, key=key, value_col=value_col)
    table_path = os.path.join(out_dir, f"{target}__spearman_vs_reference.csv")
    table.to_csv(table_path, index=False)

    outputs["variants"] = per_variant
    outputs["spearman_table"] = table_path
    outputs["spearman_table_df"] = table
    return outputs


# ===========================================================================
# Subset-BT recompute (infra: needs model + tiles) -- item 5
# ===========================================================================
def recompute_subset_bt(
    model,
    target_city: str,
    root_path: str,
    sub_bands,
    *,
    voter: str = "communities",
    preprocess_fn: Callable = None,
    score_fn: Callable = None,          # citywide_score.score_city
    load_pairs_fn: Callable = None,     # finetune_data.load_pairs
) -> Dict[int, float]:
    """Recompute subset BT: score the model over the human-vote pairs -> {image_id: bt}.

    The human-vote pairs (load_pairs) already carry image PATHS in left/right, so they
    feed score_city directly. Runs on infra.
    """
    pairs = load_pairs_fn(target_city, voter, root_path)
    if "y" not in pairs.columns:
        pairs = pairs.copy(); pairs["y"] = 0
    scored = score_fn(model, pairs, preprocess_fn=preprocess_fn)
    frame = scored["frame"]
    return dict(zip(frame["image_id"].astype(int), frame["bt_score"]))


# ===========================================================================
# CLI (items 1-4; needs geopandas + the gpkg, but no TF)
# ===========================================================================
def main():  # pragma: no cover
    import argparse
    import geopandas as gpd
    import registry as reg

    ap = argparse.ArgumentParser(description="Phase 3: comparison layer for a target city")
    ap.add_argument("--target", required=True)
    ap.add_argument("--runs-dir", default="finetuning_results/runs")
    ap.add_argument("--out-dir", default=None, help="default: finetuning_results/phase3/{target}")
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--clustering-dir", default="clustering")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join("finetuning_results", "phase3", args.target)
    runs = discover_runs(args.runs_dir, args.target)
    n_var = len(runs["variants"])
    print(f"[phase3] {args.target}: reference "
          f"{'FOUND' if runs['reference'] else 'MISSING'}, {n_var} variant(s)")
    if runs["reference"] is None:
        raise SystemExit(
            f"No reference run for {args.target} (need --source {args.target} "
            f"--target {args.target} --mode zeroshot). Run it first.")

    cfg = reg.get_city_config(args.target, args.root_path, clustering_dir=args.clustering_dir)
    gdf = gpd.read_file(cfg.gpkg_path)

    out = build_comparison(args.target, runs, gdf, out_dir, crs=gdf.crs)
    print(f"\n[phase3] wrote {n_var} ECDF + difference map pairs to {out_dir}")
    print("\nFull-city Spearman vs reference:")
    print(out["spearman_table_df"].to_string(index=False))
    print(f"\nTable: {out['spearman_table']}")


if __name__ == "__main__":
    main()
