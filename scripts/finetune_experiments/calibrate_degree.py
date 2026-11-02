"""
calibrate_degree.py
===================

One-off calibration to FIX the single ``comparisons_per_patch`` (D) used by every
city-wide map in the study. It does NOT produce any map.

Idea: BT map trustworthiness depends on how many times each patch is compared, and
every comparison is inference cost paid per model variant (x14 variants, up to MUM
scale). So we find the cheapest degree whose ranking has already converged.

Method: on ONE city (ACC) with its zero-shot local model, fit city-wide BT at a set
of sweep degrees and at a high-degree reference (cpp=200 -- deep in the converged
regime, NOT the full 15.2M-pair complete graph, which is pointless here). Report
Spearman of each sweep degree vs the reference; the recommended D is the smallest
degree whose Spearman >= a threshold (default 0.999). That D is then frozen and used
everywhere.

Efficiency: all graphs share the SAME patches (only edges differ), so patch tiles
are preprocessed ONCE into an in-memory cache and reused across every degree -- a
local optimisation inside this script, not a change to score_city.

The heavy run needs the real model + tiles (your infra). ``run_calibration`` is the
testable core (inject a model + preprocess fn); ``main`` is the CLI that resolves
the registry, loads the Keras model, and calls the core.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

import pairgraph as pgmod
from citywide_score import (
    build_patch_cache, score_pairs_with_cache, default_get_sub_bands,
)
from bt_sparse import estimate_bt_scores_from_predictions


# ===========================================================================
# Shared helpers (used by both the sweep and the diagnostics)
# ===========================================================================
def _build_graph(D, seed, *, gpkg_path, _node_ids, city, root_path, year, d_src,
                 validate_paths):
    pairs_df, rep = pgmod.build_pair_graph(
        gpkg_path=gpkg_path or "(injected)", city=city, root_path=root_path,
        year=year, d_src=d_src, comparisons_per_patch=D, seed=seed,
        validate_paths=validate_paths, _node_ids=_node_ids, _crs=None,
    )
    return pairs_df, rep


def _cache_for_graphs(graph_frames, sub_bands, preprocess_fn, norm_data,
                      expected_shape, dtype, progress=False):
    """Preprocess the union of all patches across the given graphs ONCE."""
    all_paths = pd.unique(np.concatenate(
        [np.concatenate([g["left"].to_numpy(), g["right"].to_numpy()])
         for g in graph_frames]
    ))
    if progress:
        print(f"[cache] preprocessing {len(all_paths)} unique patches once ...")
    A, path_to_row = build_patch_cache(
        all_paths, sub_bands, preprocess_fn, norm_data, expected_shape, dtype
    )
    return A, path_to_row


def _bt_for_graph(model, pairs_df, A, path_to_row, batch_size=256, progress=False):
    """Score one graph against a shared cache and fit BT -> {path: score}."""
    li = np.fromiter((path_to_row[p] for p in pairs_df["left"].to_numpy()),
                     dtype=np.int64, count=len(pairs_df))
    ri = np.fromiter((path_to_row[p] for p in pairs_df["right"].to_numpy()),
                     dtype=np.int64, count=len(pairs_df))
    probs = score_pairs_with_cache(model, li, ri, A, batch_size=batch_size,
                                   progress=progress)
    return estimate_bt_scores_from_predictions(pairs_df, probs)


def _spearman_shared(bt_a, bt_b):
    """Spearman + Kendall of two {path: score} dicts over shared patches."""
    common = sorted(set(bt_a) & set(bt_b))
    a = np.array([bt_a[k] for k in common])
    b = np.array([bt_b[k] for k in common])
    rho, _ = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    return float(rho), float(tau), len(common)


# ===========================================================================
# Testable core
# ===========================================================================
def run_calibration(
    model,
    preprocess_fn: Callable,
    *,
    gpkg_path: Optional[str] = None,
    _node_ids: Optional[Sequence[int]] = None,
    city: str = "ACC",
    root_path: str = "",
    year=2022,
    d_src: str = "S1_200m",
    degrees: Sequence[int] = (10, 20, 40, 60),
    reference_degree: int = 200,
    seed: int = 20240101,
    voter: str = "communities",
    batch_size: int = 256,
    norm_data: bool = False,
    dtype=np.float32,
    get_sub_bands_fn: Callable = default_get_sub_bands,
    validate_paths: bool = True,
    spearman_target: float = 0.999,
    progress: bool = True,
) -> Dict[str, object]:
    """Run the degree sweep and return curve + recommended D.

    Provide either ``gpkg_path`` (real) or ``_node_ids`` (tests). ``model`` needs
    ``predict_on_batch([left, right])``; ``preprocess_fn(path, sub_bands, norm_data)``
    returns an HxWxC array. All degrees share one patch cache.
    """
    sub_bands, n_w, n_h, n_c = get_sub_bands_fn(d_src)
    expected_shape = (n_h, n_w, n_c)

    all_degrees = list(degrees) + [reference_degree]

    # --- build one graph per degree (same node set, different edges) -----------
    graphs: Dict[int, pd.DataFrame] = {}
    reports: Dict[int, dict] = {}
    for D in all_degrees:
        pairs_df, rep = _build_graph(
            D, seed, gpkg_path=gpkg_path, _node_ids=_node_ids, city=city,
            root_path=root_path, year=year, d_src=d_src, validate_paths=validate_paths)
        graphs[D] = pairs_df
        reports[D] = rep
        if progress:
            print(f"[graph] D={D}: {rep['n_pairs']} pairs, "
                  f"mean_deg={rep['mean_degree']:.1f}, N={rep['n_with_imagery']}")

    # --- preprocess every patch ONCE (shared across all degrees) ---------------
    A, path_to_row = _cache_for_graphs(
        list(graphs.values()), sub_bands, preprocess_fn, norm_data,
        expected_shape, dtype, progress=progress)

    # --- score each degree's graph, fit BT -------------------------------------
    bt_by_degree: Dict[int, Dict[str, float]] = {}
    for D in all_degrees:
        bt_by_degree[D] = _bt_for_graph(model, graphs[D], A, path_to_row,
                                        batch_size=batch_size, progress=progress)
        if progress:
            print(f"[bt] D={D}: {len(bt_by_degree[D])} patch scores")

    # --- compare each sweep degree to the high-degree reference ----------------
    ref_scores = bt_by_degree[reference_degree]
    rows = []
    for D in degrees:
        rho, tau, n_shared = _spearman_shared(bt_by_degree[D], ref_scores)
        rows.append({
            "degree": D,
            "n_pairs": reports[D]["n_pairs"],
            "mean_degree": reports[D]["mean_degree"],
            "n_shared": n_shared,
            "spearman_vs_ref": rho,
            "kendall_vs_ref": tau,
        })
    curve = pd.DataFrame(rows).sort_values("degree").reset_index(drop=True)

    # --- recommend the smallest degree meeting the target ----------------------
    meeting = curve[curve["spearman_vs_ref"] >= spearman_target]
    if len(meeting):
        recommended = int(meeting["degree"].iloc[0])
        rationale = f"smallest swept degree with Spearman >= {spearman_target}"
    else:
        recommended = int(reference_degree)
        rationale = (f"no swept degree reached Spearman {spearman_target}; "
                     f"falling back to reference_degree={reference_degree}")

    return {
        "city": city,
        "curve": curve,
        "reference_degree": reference_degree,
        "recommended_degree": recommended,
        "recommended_rationale": rationale,
        "spearman_target": spearman_target,
        "graph_reports": reports,
    }


# ===========================================================================
# Diagnostics: seed-stability at chosen D, and reference-stability (200 vs 400)
# ===========================================================================
def run_diagnostics(
    model,
    preprocess_fn: Callable,
    *,
    gpkg_path: Optional[str] = None,
    _node_ids: Optional[Sequence[int]] = None,
    city: str = "ACC",
    root_path: str = "",
    year=2022,
    d_src: str = "S1_200m",
    diag_degree: int = 60,
    diag_seeds: Sequence[int] = (20240101, 20240102),
    ref_degrees: Sequence[int] = (200, 400),
    ref_seed: int = 20240101,
    batch_size: int = 256,
    norm_data: bool = False,
    dtype=np.float32,
    get_sub_bands_fn: Callable = default_get_sub_bands,
    validate_paths: bool = True,
    progress: bool = True,
) -> Dict[str, object]:
    """Two decision-relevant checks that don't depend on the reference being converged.

    seed-stability: two graphs at ``diag_degree`` with different seeds, scored by the
        same model -> Spearman between the resulting maps. This is the reproducibility
        of the map you will actually ship at that D. Freeze D if it clears ~0.996.
    reference-stability: ``ref_degrees`` (200 vs 400), same seed -> Spearman. If the
        "converged" reference isn't settled, the sweep's ruler is wobbly and true
        convergence is slower than it looked.

    All four graphs share ONE patch cache (patches preprocessed once).
    """
    if len(diag_seeds) != 2:
        raise ValueError("diag_seeds must be exactly two seeds.")
    if len(ref_degrees) != 2:
        raise ValueError("ref_degrees must be exactly two degrees.")

    sub_bands, n_w, n_h, n_c = get_sub_bands_fn(d_src)
    expected_shape = (n_h, n_w, n_c)

    gkw = dict(gpkg_path=gpkg_path, _node_ids=_node_ids, city=city,
               root_path=root_path, year=year, d_src=d_src, validate_paths=validate_paths)
    g_seedA, rep_A = _build_graph(diag_degree, diag_seeds[0], **gkw)
    g_seedB, rep_B = _build_graph(diag_degree, diag_seeds[1], **gkw)
    g_ref0, rep_r0 = _build_graph(ref_degrees[0], ref_seed, **gkw)
    g_ref1, rep_r1 = _build_graph(ref_degrees[1], ref_seed, **gkw)

    A, path_to_row = _cache_for_graphs(
        [g_seedA, g_seedB, g_ref0, g_ref1], sub_bands, preprocess_fn,
        norm_data, expected_shape, dtype, progress=progress)

    bt_seedA = _bt_for_graph(model, g_seedA, A, path_to_row, batch_size, progress)
    bt_seedB = _bt_for_graph(model, g_seedB, A, path_to_row, batch_size, progress)
    bt_ref0 = _bt_for_graph(model, g_ref0, A, path_to_row, batch_size, progress)
    bt_ref1 = _bt_for_graph(model, g_ref1, A, path_to_row, batch_size, progress)

    seed_rho, seed_tau, seed_n = _spearman_shared(bt_seedA, bt_seedB)
    ref_rho, ref_tau, ref_n = _spearman_shared(bt_ref0, bt_ref1)

    return {
        "city": city,
        "seed_stability": {
            "degree": diag_degree, "seeds": list(diag_seeds),
            "spearman": seed_rho, "kendall": seed_tau, "n_shared": seed_n,
            "n_pairs_each": [rep_A["n_pairs"], rep_B["n_pairs"]],
        },
        "reference_stability": {
            "degrees": list(ref_degrees), "seed": ref_seed,
            "spearman": ref_rho, "kendall": ref_tau, "n_shared": ref_n,
            "n_pairs_each": [rep_r0["n_pairs"], rep_r1["n_pairs"]],
        },
    }


def write_diagnostics(result: Dict[str, object], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{result['city']}_degree_diagnostics.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


# ===========================================================================
# Seed-stability sweep: freeze the smallest D whose two seeded maps agree >= target
# ===========================================================================
def run_seed_stability_sweep(
    model,
    preprocess_fn: Callable,
    *,
    gpkg_path: Optional[str] = None,
    _node_ids: Optional[Sequence[int]] = None,
    city: str = "ACC",
    root_path: str = "",
    year=2022,
    d_src: str = "S1_200m",
    degrees: Sequence[int] = (80, 100, 120, 160),
    seeds: Sequence[int] = (20240101, 20240102),
    stability_target: float = 0.996,
    batch_size: int = 256,
    norm_data: bool = False,
    dtype=np.float32,
    get_sub_bands_fn: Callable = default_get_sub_bands,
    validate_paths: bool = True,
    progress: bool = True,
) -> Dict[str, object]:
    """Measure ship-map reproducibility at each D directly and freeze the cheapest safe one.

    For each D in ``degrees`` we build two graphs with different ``seeds``, score both
    with the same model, and report the Spearman between the two resulting maps -- i.e.
    "if I regenerate this map with a different sampling seed, how much does the ranking
    move." The recommended D is the smallest whose seed-stability >= ``stability_target``.

    This is the decision-relevant curve for freezing D: it depends only on sampling
    reproducibility, not on whether any high-degree reference is converged. All graphs
    (every D x both seeds) share ONE patch cache.
    """
    if len(seeds) != 2:
        raise ValueError("seeds must be exactly two seeds.")

    sub_bands, n_w, n_h, n_c = get_sub_bands_fn(d_src)
    expected_shape = (n_h, n_w, n_c)
    gkw = dict(gpkg_path=gpkg_path, _node_ids=_node_ids, city=city,
               root_path=root_path, year=year, d_src=d_src, validate_paths=validate_paths)

    # build all graphs first (2 per degree), so the shared cache covers everything
    graphs: Dict[int, tuple] = {}
    for D in degrees:
        gA, repA = _build_graph(D, seeds[0], **gkw)
        gB, repB = _build_graph(D, seeds[1], **gkw)
        graphs[D] = (gA, gB, repA, repB)
        if progress:
            print(f"[graph] D={D}: {repA['n_pairs']} / {repB['n_pairs']} pairs "
                  f"(seeds {seeds[0]}/{seeds[1]})")

    frames = [f for D in degrees for f in (graphs[D][0], graphs[D][1])]
    A, path_to_row = _cache_for_graphs(
        frames, sub_bands, preprocess_fn, norm_data, expected_shape, dtype,
        progress=progress)

    rows = []
    for D in degrees:
        gA, gB, repA, repB = graphs[D]
        btA = _bt_for_graph(model, gA, A, path_to_row, batch_size, progress)
        btB = _bt_for_graph(model, gB, A, path_to_row, batch_size, progress)
        rho, tau, n = _spearman_shared(btA, btB)
        rows.append({
            "degree": D,
            "n_pairs_seedA": repA["n_pairs"], "n_pairs_seedB": repB["n_pairs"],
            "seed_stability_spearman": rho, "seed_stability_kendall": tau,
            "n_shared": n,
        })
        if progress:
            print(f"[seed-stability] D={D}: Spearman={rho:.4f}")

    curve = pd.DataFrame(rows).sort_values("degree").reset_index(drop=True)
    meeting = curve[curve["seed_stability_spearman"] >= stability_target]
    if len(meeting):
        recommended = int(meeting["degree"].iloc[0])
        rationale = f"smallest D with seed-stability >= {stability_target}"
    else:
        recommended = None
        rationale = (f"no swept D reached seed-stability {stability_target}; "
                     f"extend degrees upward")

    return {
        "city": city,
        "seeds": list(seeds),
        "stability_target": stability_target,
        "curve": curve,
        "recommended_degree": recommended,
        "recommended_rationale": rationale,
    }


def write_seed_sweep(result: Dict[str, object], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    curve: pd.DataFrame = result["curve"]
    paths = {}
    csv_path = os.path.join(out_dir, f"{result['city']}_seed_stability_sweep.csv")
    curve.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    summary = {k: v for k, v in result.items() if k != "curve"}
    summary["curve"] = curve.to_dict(orient="records")
    json_path = os.path.join(out_dir, f"{result['city']}_seed_stability_sweep.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    paths["json"] = json_path
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curve["degree"], curve["seed_stability_spearman"], "o-")
        ax.axhline(result["stability_target"], ls="--", color="grey",
                   label=f"target {result['stability_target']}")
        if result["recommended_degree"] is not None:
            ax.axvline(result["recommended_degree"], ls=":", color="red",
                       label=f"freeze D={result['recommended_degree']}")
        ax.set_xlabel("comparisons_per_patch (mean degree)")
        ax.set_ylabel("seed-stability Spearman (two seeded maps)")
        ax.set_title(f"{result['city']} seed-stability sweep")
        ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(out_dir, f"{result['city']}_seed_stability_sweep.png")
        fig.savefig(png, dpi=140)
        plt.close(fig)
        paths["png"] = png
    except Exception as e:  # pragma: no cover
        print(f"[plot] skipped ({e})")
    return paths


# ===========================================================================
# Output helpers
# ===========================================================================
def write_outputs(result: Dict[str, object], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    curve: pd.DataFrame = result["curve"]
    paths = {}

    csv_path = os.path.join(out_dir, f"{result['city']}_degree_calibration.csv")
    curve.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    summary = {k: v for k, v in result.items() if k not in ("curve", "graph_reports")}
    summary["curve"] = curve.to_dict(orient="records")
    json_path = os.path.join(out_dir, f"{result['city']}_degree_calibration.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    paths["json"] = json_path

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(curve["degree"], curve["spearman_vs_ref"], "o-", label="Spearman vs ref")
        ax.axhline(result["spearman_target"], ls="--", color="grey",
                   label=f"target {result['spearman_target']}")
        ax.axvline(result["recommended_degree"], ls=":", color="red",
                   label=f"recommended D={result['recommended_degree']}")
        ax.set_xlabel("comparisons_per_patch (mean degree)")
        ax.set_ylabel(f"Spearman vs cpp={result['reference_degree']} reference")
        ax.set_title(f"{result['city']} degree calibration")
        ax.legend(fontsize=8)
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{result['city']}_degree_calibration.png")
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        paths["png"] = png_path
    except Exception as e:  # pragma: no cover
        print(f"[plot] skipped ({e})")

    return paths


def _load_callable(spec: str) -> Callable:
    """Load 'package.module:function' dynamically (for project hooks)."""
    if ":" not in spec:
        raise ValueError(f"Expected 'module:function', got {spec!r}")
    mod_name, fn_name = spec.split(":", 1)
    return getattr(importlib.import_module(mod_name), fn_name)


# ===========================================================================
# CLI (real run -- needs TensorFlow + GDAL + real model/tiles)
# ===========================================================================
def main():  # pragma: no cover  (exercised on the user's infra)
    ap = argparse.ArgumentParser(description="ACC degree calibration for city-wide BT maps")
    ap.add_argument("--city", default="ACC")
    ap.add_argument("--root-path", required=True, help="PARTIMAP data root")
    ap.add_argument("--gpkg-path", default=None, help="full-city gpkg (else via --gpkg-dir)")
    ap.add_argument("--gpkg-dir", default=None)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--d-src", default="S1_200m")
    ap.add_argument("--degrees", default="10,20,40,60")
    ap.add_argument("--reference-degree", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20240101)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--spearman-target", type=float, default=0.999)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--diagnostics", action="store_true",
                    help="run seed-stability + reference-stability instead of the sweep")
    ap.add_argument("--diag-degree", type=int, default=60)
    ap.add_argument("--diag-seeds", default="20240101,20240102")
    ap.add_argument("--ref-degrees", default="200,400")
    ap.add_argument("--seed-stability-sweep", default=None,
                    help="comma-separated degrees, e.g. '80,100,120,160'; runs the "
                         "seed-stability sweep and freezes the smallest D >= target")
    ap.add_argument("--stability-target", type=float, default=0.996)
    ap.add_argument("--preprocess", default="BT_utils.dataLoad:load_and_preprocess_image",
                    help="project hook 'module:function' for load_and_preprocess_image")
    args = ap.parse_args()

    import tensorflow as tf  # noqa
    import registry as reg

    cfg = reg.get_city_config(args.city, args.root_path, gpkg_dir=args.gpkg_dir,
                              gpkg_path=args.gpkg_path, year=args.year, d_src=args.d_src)
    model_path = reg.resolve_model_path(args.city, args.city, args.root_path, d_src=args.d_src)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model not found: {model_path}")
    if not os.path.exists(cfg.gpkg_path):
        raise FileNotFoundError(f"gpkg not found: {cfg.gpkg_path}")

    print(f"[load] model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    preprocess_fn = _load_callable(args.preprocess)

    if args.seed_stability_sweep:
        sweep = run_seed_stability_sweep(
            model, preprocess_fn, gpkg_path=cfg.gpkg_path, city=args.city,
            root_path=args.root_path, year=cfg.year, d_src=args.d_src,
            degrees=[int(x) for x in args.seed_stability_sweep.split(",")],
            seeds=[int(x) for x in args.diag_seeds.split(",")],
            stability_target=args.stability_target, batch_size=args.batch_size,
            validate_paths=True, progress=True,
        )
        paths = write_seed_sweep(sweep, args.out_dir)
        print("\n=== SEED-STABILITY SWEEP ===")
        print(sweep["curve"].to_string(index=False))
        if sweep["recommended_degree"] is not None:
            print(f"\nFreeze D = {sweep['recommended_degree']}  ({sweep['recommended_rationale']})")
        else:
            print(f"\n{sweep['recommended_rationale']}")
        print(f"Outputs: {paths}")
        return

    if args.diagnostics:
        diag = run_diagnostics(
            model, preprocess_fn, gpkg_path=cfg.gpkg_path, city=args.city,
            root_path=args.root_path, year=cfg.year, d_src=args.d_src,
            diag_degree=args.diag_degree,
            diag_seeds=[int(x) for x in args.diag_seeds.split(",")],
            ref_degrees=[int(x) for x in args.ref_degrees.split(",")],
            ref_seed=args.seed, batch_size=args.batch_size,
            validate_paths=True, progress=True,
        )
        path = write_diagnostics(diag, args.out_dir)
        ss = diag["seed_stability"]; rs = diag["reference_stability"]
        print("\n=== DIAGNOSTICS ===")
        print(f"seed-stability @ D={ss['degree']} (seeds {ss['seeds']}): "
              f"Spearman={ss['spearman']:.4f}, Kendall={ss['kendall']:.4f}")
        print(f"reference-stability {rs['degrees']}: "
              f"Spearman={rs['spearman']:.4f}, Kendall={rs['kendall']:.4f}")
        verdict = ("FREEZE this D" if ss["spearman"] >= 0.996
                   else "consider a higher D")
        ref_ok = ("reference is settled" if rs["spearman"] >= 0.999
                  else "reference NOT fully settled -- true convergence is slower")
        print(f"-> seed-stability suggests: {verdict}")
        print(f"-> {ref_ok}")
        print(f"Output: {path}")
        return

    result = run_calibration(
        model, preprocess_fn, gpkg_path=cfg.gpkg_path, city=args.city,
        root_path=args.root_path, year=cfg.year, d_src=args.d_src,
        degrees=[int(x) for x in args.degrees.split(",")],
        reference_degree=args.reference_degree, seed=args.seed,
        batch_size=args.batch_size, spearman_target=args.spearman_target,
        validate_paths=True, progress=True,
    )
    paths = write_outputs(result, args.out_dir)
    print("\n=== CALIBRATION RESULT ===")
    print(result["curve"].to_string(index=False))
    print(f"\nRecommended D = {result['recommended_degree']}  "
          f"({result['recommended_rationale']})")
    print(f"Outputs: {paths}")


if __name__ == "__main__":
    main()
