"""
test_phase3.py
==============

Phase 3 core on synthetic run dirs + the real ACC gpkg (as a stand-in target):
  * discovery from manifests; reference identification; reference-missing hard-fail
  * reference-ECDF + difference arithmetic (reference vs itself -> ~uniform; a run
    differenced against the reference it equals -> ~0)
  * full-city Spearman-vs-reference table ordering
  * subset-consistency comparison logic
  * build_comparison writes ECDF + difference GeoTIFFs + table against a real gpkg

Run:  python test_phase3.py
"""

import json
import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd

import phase3
import normalize as nz

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
ACC_GPKG = "clustering/ACC/ACC_FN_200m.gpkg"


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


def _write_run(base, run_id, manifest, scores):
    d = os.path.join(base, run_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    scores.to_csv(os.path.join(d, "scores.csv"), index=False)
    return d


def _scores_for(ids, seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"image_id": ids, "path": [f"p_{i}.tif" for i in ids],
                         "bt_score": rng.randn(len(ids))})


def test_discovery_and_reference():
    print("Test 1: discovery groups by target and finds the reference")
    ids = list(range(100))
    with tempfile.TemporaryDirectory() as tmp:
        _write_run(tmp, "TEM_ref",
                   {"source": "TEM", "target": "TEM", "mode": "zeroshot",
                    "src_code": "TEM", "status": "ok"}, _scores_for(ids, 0))
        _write_run(tmp, "ACC_TEM_ft",
                   {"source": "ACC", "target": "TEM", "mode": "finetune",
                    "src_code": "ACC", "sampling": "random", "sample_size": 5000,
                    "status": "ok"}, _scores_for(ids, 1))
        _write_run(tmp, "NBO_ACC_ft",   # different target -> ignored
                   {"source": "NBO", "target": "ACC", "mode": "finetune",
                    "src_code": "NBO", "status": "ok"}, _scores_for(ids, 2))
        runs = phase3.discover_runs(tmp, "TEM")
        ok = True
        ok &= check("reference found", runs["reference"] is not None)
        ok &= check("reference is TEM->TEM zeroshot",
                    runs["reference"][0]["src_code"] == "TEM")
        ok &= check("one variant for TEM", len(runs["variants"]) == 1,
                    f"{len(runs['variants'])}")
        ok &= check("other-target run excluded",
                    all(m["target"] == "TEM" for m, _ in runs["variants"]))
        return ok


def test_reference_missing_hardfail():
    print("Test 2: build_comparison hard-fails when reference is absent")
    ids = list(range(50))
    runs = {"reference": None,
            "variants": [({"src_code": "ACC", "mode": "finetune", "sampling": "random",
                           "sample_size": 5000}, _scores_for(ids, 1))]}
    gdf = gpd.read_file(ACC_GPKG)
    try:
        phase3.build_comparison("TEM", runs, gdf, "/tmp/x")
        return check("raised on missing reference", False)
    except RuntimeError as e:
        return check("raised on missing reference", "reference" in str(e).lower())


def test_ecdf_and_difference():
    print("Test 3: reference ECDF ~uniform; reference differenced with itself ~0")
    ids = list(range(500))
    ref = _scores_for(ids, 0)
    nd = phase3.normalized_and_difference(ref, ref)  # variant == reference
    ok = True
    ecdf = nd["reference_ecdf"]["bt_score_ecdf"].to_numpy()
    ok &= check("reference ECDF in [0,1]", ecdf.min() >= 0 and ecdf.max() <= 1)
    ok &= check("variant==reference -> difference ~0",
                np.allclose(nd["difference"]["difference"].to_numpy(), 0, atol=1e-9),
                f"max |diff| {np.max(np.abs(nd['difference']['difference'])):.2e}")
    return ok


def test_spearman_table():
    print("Test 4: Spearman-vs-reference table ranks a near-copy above a shuffle")
    ids = list(range(300))
    ref = _scores_for(ids, 0)
    # near-copy: reference + tiny noise -> high spearman
    near = ref.copy(); near["bt_score"] = ref["bt_score"] + np.random.RandomState(9).randn(300) * 0.01
    # unrelated: fresh random -> low spearman
    far = _scores_for(ids, 123)
    variants = [({"src_code": "NEAR", "mode": "finetune", "sampling": "random",
                  "sample_size": 5000}, near),
                ({"src_code": "FAR", "mode": "zeroshot"}, far)]
    table = phase3.full_city_spearman_table(ref, variants)
    ok = True
    ok &= check("near-copy ranked first", table.iloc[0]["src_code"] == "NEAR",
                f"{table.iloc[0]['src_code']}")
    ok &= check("near spearman > far", table.iloc[0]["spearman_vs_reference"]
                > table.iloc[1]["spearman_vs_reference"])
    return ok


def test_subset_consistency():
    print("Test 5: subset-consistency Spearman on shared patches")
    ids = list(range(200))
    full = _scores_for(ids, 0)
    # subset BT = full restricted to 60 patches, monotonically related -> high spearman
    sub_ids = ids[:60]
    subset_bt = {i: float(full[full["image_id"] == i]["bt_score"].iloc[0] * 2 + 0.1)
                 for i in sub_ids}
    res = phase3.subset_consistency(full, subset_bt)
    ok = True
    ok &= check("uses only shared patches", res["n_shared"] == 60, f"{res['n_shared']}")
    ok &= check("monotone subset -> spearman ~1", abs(res["spearman"] - 1.0) < 1e-9,
                f"{res['spearman']:.4f}")
    return ok


def test_build_comparison_real_gpkg():
    print("Test 6: build_comparison writes ECDF + difference GeoTIFFs (real ACC gpkg)")
    gdf = gpd.read_file(ACC_GPKG)
    ids = gdf["PARTIMAP_I"].to_numpy()
    ref_scores = _scores_for(ids, 0)
    var_scores = _scores_for(ids, 1)
    runs = {"reference": ({"source": "ACC", "target": "ACC", "mode": "zeroshot",
                           "src_code": "ACC"}, ref_scores),
            "variants": [({"source": "NBO", "target": "ACC", "mode": "finetune",
                           "src_code": "NBO", "sampling": "random", "sample_size": 5000},
                          var_scores)]}
    with tempfile.TemporaryDirectory() as tmp:
        out = phase3.build_comparison("ACC", runs, gdf, tmp)
        import rasterio
        ecdf_tif = os.path.join(tmp, "ACC__NBO__random_5000__ecdf__bt_score_ecdf.tif")
        diff_tif = os.path.join(tmp, "ACC__NBO__random_5000__diff__difference.tif")
        ok = True
        ok &= check("ecdf geotiff written", os.path.exists(ecdf_tif))
        ok &= check("difference geotiff written", os.path.exists(diff_tif))
        ok &= check("spearman table written", os.path.exists(out["spearman_table"]))
        if os.path.exists(diff_tif):
            with rasterio.open(diff_tif) as src:
                data = src.read(1)
                fin = data[np.isfinite(data)]
            ok &= check("difference values within [-1,1]",
                        fin.min() >= -1.0 and fin.max() <= 1.0,
                        f"[{fin.min():.3f},{fin.max():.3f}]")
        return ok


def main():
    tests = [test_discovery_and_reference, test_reference_missing_hardfail,
             test_ecdf_and_difference, test_spearman_table, test_subset_consistency,
             test_build_comparison_real_gpkg]
    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception:
            print(f"  [{FAIL}] {t.__name__} raised:")
            traceback.print_exc()
            results.append(False)
        print()
    n = sum(results)
    print("=" * 60)
    print(f"RESULT: {n}/{len(results)} test groups passed")
    print("=" * 60)
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
