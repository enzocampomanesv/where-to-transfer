"""
run_pipeline.py
===============

One source->target variant per invocation. Resolves inputs, builds/loads the frozen
target pair graph, runs zero-shot or random-finetune (with the reproducibility gate),
scores the whole city, writes a raw per-patch score CSV (Phase-3 hand-off), a raw-BT
GeoTIFF, and a run manifest. Finetune mode also saves the re-finetuned checkpoint.

Modes:
  zeroshot : load the saved source model, score, map. A wiring CHECK compares the
             zero-shot test metrics to the recorded sample_size=0 values (catches a
             mis-resolved model / preprocessing drift) -- hard-stop on mismatch.
  finetune : (random only in v1) re-finetune, GATE against the original three-seed
             band, hard-stop on fail, save checkpoint, then score + map.
             sampling='informed' raises NotImplementedError until v2.

Normalisation (reference-ECDF) is deferred to Phase 3; each run emits RAW BT only
and is fully independent / order-free. The frozen graph is keyed on (target, D,
generation_seed) -- shared across every variant of a target so map differences
reflect the model, not the sampling.

The orchestration is in ``run_variant`` with injectable heavy deps (model loader,
preprocess, score, finetune) so it is testable without TensorFlow or tiles. ``main``
wires the real implementations.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

import registry as reg
import pairgraph as pgmod
import rasterize as rzmod
import gate as gatemod

GEN_SEED = 20240101   # fixed graph generation seed (NOT the finetune seed)


# ===========================================================================
# Frozen target graph: build once per (target, D, gen_seed), reuse across sources
# ===========================================================================
def resolve_graph(
    target_cfg: reg.CityConfig,
    comparisons_per_patch: int,
    *,
    gen_seed: int = GEN_SEED,
    graph_cache_dir: str,
    validate_paths: bool = True,
    build_fn: Callable = pgmod.build_pair_graph,
) -> "tuple[pd.DataFrame, dict, str]":
    """Load the frozen graph for a target if cached, else build and cache it."""
    os.makedirs(graph_cache_dir, exist_ok=True)
    key = f"{target_cfg.code}_D{comparisons_per_patch}_seed{gen_seed}"
    parquet = os.path.join(graph_cache_dir, f"{key}.parquet")
    report_path = os.path.join(graph_cache_dir, f"{key}.report.json")

    if os.path.exists(parquet) and os.path.exists(report_path):
        pairs_df = pd.read_parquet(parquet)
        with open(report_path) as f:
            report = json.load(f)
        report["_loaded_from_cache"] = True
        return pairs_df, report, parquet

    pairs_df, report = build_fn(
        gpkg_path=target_cfg.gpkg_path, city=target_cfg.code,
        root_path=target_cfg.root_path, year=target_cfg.year, d_src=target_cfg.d_src,
        comparisons_per_patch=comparisons_per_patch, seed=gen_seed,
        validate_paths=validate_paths, out_parquet=parquet, out_report=report_path,
    )
    report["_loaded_from_cache"] = False
    return pairs_df, report, parquet


# ===========================================================================
# Orchestration core (injectable heavy deps -> testable without TF/tiles)
# ===========================================================================
def run_variant(
    *,
    source: str,
    target: str,
    mode: str,                       # 'zeroshot' | 'finetune'
    comparisons_per_patch: int,
    root_path: str,
    clustering_dir: str = "clustering",
    metrics_csv_dir: str,
    out_dir: str,
    graph_cache_dir: str,
    ft_save_dir: Optional[str] = None,
    iso_model_dir: Optional[str] = None,
    sampling: str = "random",
    sample_size: int = 5000,
    seed: int = 42,
    d_src: str = "S1_200m",
    year: Optional[int] = None,
    gen_seed: int = GEN_SEED,
    tol_k: float = gatemod.DEFAULT_TOL_K,
    tol_floor: float = gatemod.DEFAULT_TOL_FLOOR,
    zeroshot_check: str = "hard",    # 'hard' | 'warn' | 'off'
    validate_paths: bool = True,
    # ---- injectable heavy deps (real impls wired in main) ----
    preprocess_fn: Callable = None,
    sub_bands=None,
    model_loader: Callable = None,          # (path) -> model
    score_fn: Callable = None,              # (model, pairs_df, ...) -> {frame, ...}
    finetune_fn: Callable = None,           # (...) -> {model, metrics, selected_ids, ...}
    save_model_fn: Callable = None,         # (model, path) -> None
    zeroshot_eval_fn: Callable = None,      # (model, target, ...) -> {accuracy, spearman}
    rasterize_fn: Callable = rzmod.rasterize_frame,
    resolve_inputs_fn: Callable = reg.resolve_inputs,
) -> Dict[str, object]:
    """Run a single variant end to end. Returns the manifest dict (also written to disk)."""
    t0 = time.time()
    if mode not in ("zeroshot", "finetune"):
        raise ValueError(f"mode must be 'zeroshot' or 'finetune', got {mode!r}")
    if mode == "finetune" and sampling not in ("random", "informed", "informed_t1"):
        raise NotImplementedError(
            f"finetune sampling={sampling!r} not supported."
        )

    # 1. resolve model + target config (hard-fail if missing)
    resolved = resolve_inputs_fn(
        source, target, root_path, clustering_dir=clustering_dir, d_src=d_src, year=year,
    )
    src_code = resolved.src_code
    target_cfg = resolved.target_cfg

    run_id = (f"{src_code}__{target}__{mode}"
              + ("" if mode == "zeroshot" else f"__{sampling}_{sample_size}")
              + f"__seed{seed}__D{comparisons_per_patch}")
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    manifest: Dict[str, object] = {
        "run_id": run_id, "source": source, "target": target, "src_code": src_code,
        "mode": mode, "sampling": sampling if mode == "finetune" else None,
        "sample_size": sample_size if mode == "finetune" else None,
        "seed": seed, "comparisons_per_patch": comparisons_per_patch,
        "gen_seed": gen_seed, "model_path": resolved.model_path,
        "gpkg_path": target_cfg.gpkg_path, "d_src": target_cfg.d_src,
        "year": target_cfg.year,
    }

    # 2. frozen target graph (shared across sources for this target)
    pairs_df, graph_report, graph_path = resolve_graph(
        target_cfg, comparisons_per_patch, gen_seed=gen_seed,
        graph_cache_dir=graph_cache_dir, validate_paths=validate_paths,
    )
    manifest["graph"] = {"path": graph_path, "n_pairs": graph_report.get("n_pairs"),
                         "n_patches": graph_report.get("n_with_imagery"),
                         "loaded_from_cache": graph_report.get("_loaded_from_cache")}

    metrics_df = gatemod.load_metrics(metrics_csv_dir)

    # 3. obtain the (scored) model
    if mode == "zeroshot":
        model = model_loader(resolved.model_path)
        # wiring check against recorded zero-shot metrics
        band = gatemod.band_for(metrics_df, src_code, target, 0, sampling, zero_shot=True)
        if band is not None and zeroshot_eval_fn is not None:
            zmetrics = zeroshot_eval_fn(model, target, root_path, target_cfg,
                                        preprocess_fn, sub_bands)
            gate_res = gatemod.evaluate_gate(zmetrics, band, tol_k=tol_k, tol_floor=tol_floor)
            manifest["zeroshot_check"] = {"metrics": zmetrics, "result": gate_res}
            if not gate_res["passed"]:
                msg = (f"Zero-shot wiring check FAILED for {src_code}->{target}: "
                       f"{gate_res['metrics']}")
                if zeroshot_check == "hard":
                    manifest["status"] = "zeroshot_check_failed"
                    _write_manifest(manifest, run_dir)
                    raise RuntimeError(msg + "  (set --zeroshot-check warn to override)")
                elif zeroshot_check == "warn":
                    print("[WARN] " + msg)
        else:
            manifest["zeroshot_check"] = {"skipped": True,
                                          "reason": "no recorded zero-shot band"
                                          if band is None else "no eval fn"}

    else:  # finetune (random)
        band = gatemod.band_for(metrics_df, src_code, target, sample_size, sampling)
        if band is None:
            manifest["status"] = "missing_metric_band"
            _write_manifest(manifest, run_dir)
            raise RuntimeError(
                f"No original metrics for {src_code}->{target} "
                f"sample_size={sample_size} sampling={sampling}; cannot gate -> refusing to map."
            )
        ft = finetune_fn(
            model_path=resolved.model_path, target_city=target, root_path=root_path,
            sub_bands=sub_bands, preprocess_fn=preprocess_fn, sampling=sampling,
            sample_size=sample_size, seed=seed,
            iso_model_dir=iso_model_dir, source_code=src_code,
        )
        gate_res = gatemod.evaluate_gate(ft["metrics"], band, tol_k=tol_k, tol_floor=tol_floor)
        manifest["gate"] = {"refit_metrics": ft["metrics"], "result": gate_res}
        if not gate_res["passed"]:
            manifest["status"] = "gate_failed"
            _write_manifest(manifest, run_dir)
            raise RuntimeError(
                f"GATE FAILED for {src_code}->{target} s={sample_size} {sampling}: "
                f"{gate_res['metrics']}  (re-finetune out of the original three-seed band)"
            )
        model = ft["model"]

        # save the re-finetuned checkpoint (weights-only; the missing-model fix)
        if ft_save_dir and save_model_fn is not None:
            os.makedirs(ft_save_dir, exist_ok=True)
            ckpt = os.path.join(
                ft_save_dir,
                f"{src_code}__ft_{target}_{sampling}_{sample_size}_seed{seed}_{d_src}.ckpt")
            saved_path = save_model_fn(model, ckpt)
            manifest["checkpoint"] = saved_path or ckpt
            manifest["checkpoint_format"] = "weights_only"
            manifest["architecture_source"] = resolved.model_path

        # log selected finetune ids
        sel = pd.DataFrame(ft.get("selected_ids", []), columns=["left_id", "right_id"])
        sel_path = os.path.join(run_dir, "selected_ids.csv")
        sel.to_csv(sel_path, index=False)
        manifest["selected_ids_csv"] = sel_path

    # 4. score the whole city with the obtained model
    scored = score_fn(model, pairs_df, d_src=target_cfg.d_src,
                      preprocess_fn=preprocess_fn)
    frame = scored["frame"]
    scores_csv = os.path.join(run_dir, "scores.csv")
    frame.to_csv(scores_csv, index=False)
    manifest["scores_csv"] = scores_csv
    manifest["n_patches_scored"] = int(frame["image_id"].nunique())

    # 5. rasterize raw BT -> GeoTIFF
    import geopandas as gpd
    gdf = gpd.read_file(target_cfg.gpkg_path)
    no_img = graph_report.get("dropped_ids") or []
    rz = rasterize_fn(frame, gdf, run_dir, value_cols=["bt_score"], basename=run_id,
                      no_imagery_ids=no_img,
                      tags={"source": src_code, "target": target, "mode": mode,
                            "sampling": sampling, "sample_size": sample_size,
                            "seed": seed, "D": comparisons_per_patch,
                            "normalization": "raw"})
    manifest["geotiff"] = rz["geotiffs"]["bt_score"]
    manifest["coverage_mask"] = rz.get("coverage_mask")

    manifest["status"] = "ok"
    manifest["seconds"] = round(time.time() - t0, 1)
    _write_manifest(manifest, run_dir)
    return manifest


def _write_manifest(manifest: Dict, run_dir: str):
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ===========================================================================
# CLI (real run -- needs TF, GDAL, model, tiles)
# ===========================================================================
def _real_zeroshot_eval(model, target, root_path, target_cfg, preprocess_fn, sub_bands):
    """Compute zero-shot test-set accuracy + BT spearman exactly as the study did."""
    from finetune_data import reconstruct_split
    from finetune_core import create_dataset, evaluate_model
    from BT_utils.evalBT import evaluate_model_bt_recovery
    _, test_data = reconstruct_split(target, "communities", root_path)
    ds = create_dataset(test_data, root_path, sub_bands, preprocess_fn)
    preds, probs = evaluate_model(model, ds)
    y = np.array(test_data["y"], dtype=int).flatten()
    acc = float((y == np.array(preds, dtype=int).flatten()).mean())
    bt = evaluate_model_bt_recovery(test_data, probs)
    return {"accuracy": acc, "spearman": bt.get("spearman")}


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Run one source->target city-wide map variant")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--mode", required=True, choices=["zeroshot", "finetune"])
    ap.add_argument("--sampling", default="random")
    ap.add_argument("--sample-size", type=int, default=5000)
    # D=120 was fixed by the ACC seed-stability sweep; change only if recalibrated.
    ap.add_argument("--comparisons-per-patch", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    # Machine-specific defaults (this deployment). Override on other machines.
    ap.add_argument("--root-path", default=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"))
    ap.add_argument("--clustering-dir", default="clustering",
                    help="gpkg dir under root_path: {root}/{clustering_dir}/{CITY}/{CITY}_FN_200m.gpkg")
    ap.add_argument("--iso-model-dir", default=os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"))
    ap.add_argument("--metrics-csv-dir",
                    default=os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/transferLearning"))
    ap.add_argument("--out-dir", default="finetuning_results/runs")
    ap.add_argument("--graph-cache-dir", default="finetuning_results/graphs")
    ap.add_argument("--ft-save-dir", default="trained_models/voter")
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--zeroshot-check", default="hard", choices=["hard", "warn", "off"])
    ap.add_argument("--preprocess", default="project_io:load_and_preprocess_image")
    args = ap.parse_args()

    import tensorflow as tf
    from calibrate_degree import _load_callable
    from citywide_score import score_city, default_get_sub_bands
    from finetune_core import finetune_variant, save_finetuned_weights

    preprocess_fn = _load_callable(args.preprocess)
    sub_bands, _, _, _ = default_get_sub_bands(args.d_src)

    manifest = run_variant(
        source=args.source, target=args.target, mode=args.mode,
        comparisons_per_patch=args.comparisons_per_patch, root_path=args.root_path,
        clustering_dir=args.clustering_dir, metrics_csv_dir=args.metrics_csv_dir,
        out_dir=args.out_dir, graph_cache_dir=args.graph_cache_dir,
        ft_save_dir=args.ft_save_dir, iso_model_dir=args.iso_model_dir,
        sampling=args.sampling,
        sample_size=args.sample_size, seed=args.seed, d_src=args.d_src, year=args.year,
        zeroshot_check=args.zeroshot_check,
        preprocess_fn=preprocess_fn, sub_bands=sub_bands,
        model_loader=lambda p: tf.keras.models.load_model(p),
        score_fn=score_city,
        finetune_fn=finetune_variant,
        save_model_fn=save_finetuned_weights,
        zeroshot_eval_fn=_real_zeroshot_eval,
    )
    print(f"\n[{manifest['status']}] {manifest['run_id']}")
    print(f"  scores: {manifest.get('scores_csv')}")
    print(f"  geotiff: {manifest.get('geotiff')}")
    if "gate" in manifest:
        print(f"  gate passed: {manifest['gate']['result']['passed']}")


if __name__ == "__main__":
    main()
