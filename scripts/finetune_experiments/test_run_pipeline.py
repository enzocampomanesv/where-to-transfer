"""
test_run_pipeline.py
====================

Exercises run_variant end-to-end with REAL registry/pairgraph/gate/rasterize (against
the real ACC gpkg) and MOCKED model/score/finetune (no TF, no tiles). Covers:
  * finetune (random) happy path: gate passes -> checkpoint saved, scores + GeoTIFF +
    manifest written, selected ids logged.
  * gate hard-stop: refit out of band -> RuntimeError, status 'gate_failed'.
  * missing band -> RuntimeError, status 'missing_metric_band'.
  * zeroshot happy path with wiring check passing.
  * graph caching: second call for same (target,D,seed) loads from cache.
  * informed finetune -> NotImplementedError (v1).

Run:  python test_run_pipeline.py
"""

import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd
import geopandas as gpd

import run_pipeline as rp
import registry as reg

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

ACC_GPKG = "clustering/ACC/ACC_FN_200m.gpkg"
METRICS_DIR = os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/transferLearning")
ROOT = os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP")


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


# ---- mocks ----------------------------------------------------------------
def _fake_frame_for_gpkg(gpkg, n=None):
    gdf = gpd.read_file(gpkg)
    ids = gdf["PARTIMAP_I"].to_numpy()
    rng = np.random.RandomState(0)
    return pd.DataFrame({
        "path": [f"votes/2022/ACC/S1_200m/ACC_S1_200m_{i}.tif" for i in ids],
        "image_id": ids,
        "bt_score": rng.randn(len(ids)),
    })


def _mock_score_fn(model, pairs_df, **kw):
    # ignore model; produce a frame covering all ACC cells
    return {"frame": _fake_frame_for_gpkg(ACC_GPKG), "n_patches": 5523, "n_pairs": len(pairs_df)}


def _resolve_inputs_no_fs(source, target, root_path, **kw):
    """Bypass filesystem existence checks; build a plausible ResolvedInputs for ACC."""
    src_code = reg.effective_source_code(source, target)
    model_path = reg.resolve_model_path(source, target, root_path)
    cfg = reg.CityConfig(code=target, root_path=root_path, year=2022, d_src="S1_200m",
                         gpkg_path=ACC_GPKG)
    return reg.ResolvedInputs(source=source, target=target, src_code=src_code,
                              model_path=model_path, target_cfg=cfg,
                              checks={"model_exists": True, "gpkg_exists": True})


def _make_finetune_fn(spearman, accuracy):
    def _ft(**kw):
        return {"model": object(), "metrics": {"spearman": spearman, "accuracy": accuracy},
                "selected_ids": [(1, 2), (3, 4)], "n_finetune": kw.get("sample_size"),
                "sampling": kw.get("sampling"), "sample_size": kw.get("sample_size"),
                "seed": kw.get("seed")}
    return _ft


def _common_kwargs(tmp):
    return dict(
        source="PRI_NBO" if False else "ACC",  # placeholder; set per test
        target="TEM", mode="finetune", comparisons_per_patch=40,
        root_path=ROOT, clustering_dir="clustering", metrics_csv_dir=METRICS_DIR,
        out_dir=os.path.join(tmp, "runs"), graph_cache_dir=os.path.join(tmp, "graphs"),
        ft_save_dir=os.path.join(tmp, "ckpts"), sampling="random", sample_size=5000,
        seed=42, validate_paths=False,   # no tiles in sandbox
        preprocess_fn=lambda p, sb: np.zeros((20, 20, 2)), sub_bands=[0, 1],
        model_loader=lambda p: object(), score_fn=_mock_score_fn,
        save_model_fn=lambda m, p: open(p, "w").close(),
        resolve_inputs_fn=_resolve_inputs_no_fs,
    )


def test_finetune_gate_pass():
    print("Test 1: finetune (random) gate PASS -> full outputs + checkpoint")
    import gate as gatemod
    with tempfile.TemporaryDirectory() as tmp:
        df = gatemod.load_metrics(METRICS_DIR)
        band = gatemod.band_for(df, "PRI_NBO", "TEM", 5000, "random")
        # choose refit metrics inside the band
        sp = band["spearman"]["mean"]; ac = band["accuracy"]["mean"]
        kw = _common_kwargs(tmp); kw["source"] = "PRI_NBO"
        kw["finetune_fn"] = _make_finetune_fn(sp, ac)
        man = rp.run_variant(**kw)
        ok = True
        ok &= check("status ok", man["status"] == "ok", man["status"])
        ok &= check("gate passed", man["gate"]["result"]["passed"])
        ok &= check("scores.csv written", os.path.exists(man["scores_csv"]))
        ok &= check("geotiff written", os.path.exists(man["geotiff"]))
        ok &= check("checkpoint saved", os.path.exists(man["checkpoint"]))
        ok &= check("selected ids logged", os.path.exists(man["selected_ids_csv"]))
        ok &= check("all ACC cells scored", man["n_patches_scored"] == 5523,
                    f"{man['n_patches_scored']}")
        return ok


def test_finetune_gate_fail():
    print("Test 2: finetune gate FAIL -> hard stop, status gate_failed")
    with tempfile.TemporaryDirectory() as tmp:
        kw = _common_kwargs(tmp); kw["source"] = "PRI_NBO"
        kw["finetune_fn"] = _make_finetune_fn(0.30, 0.30)  # far below band
        try:
            rp.run_variant(**kw)
            return check("raised on gate fail", False)
        except RuntimeError as e:
            ok = check("raised RuntimeError", "GATE FAILED" in str(e))
            # manifest with status should still be written
            man_path = os.path.join(kw["out_dir"],
                                    "PRI_NBO__TEM__finetune__random_5000__seed42__D40",
                                    "manifest.json")
            ok &= check("gate_failed manifest written", os.path.exists(man_path))
            return ok


def test_missing_band():
    print("Test 3: no original metrics for cell -> hard stop, missing_metric_band")
    with tempfile.TemporaryDirectory() as tmp:
        kw = _common_kwargs(tmp); kw["source"] = "ACC"; kw["sample_size"] = 777  # nonexistent size
        kw["finetune_fn"] = _make_finetune_fn(0.7, 0.7)
        try:
            rp.run_variant(**kw)
            return check("raised on missing band", False)
        except RuntimeError as e:
            return check("raised RuntimeError (cannot gate)", "cannot gate" in str(e))


def test_graph_cache():
    print("Test 4: frozen graph is cached and reused across calls for same (target,D,seed)")
    import gate as gatemod
    with tempfile.TemporaryDirectory() as tmp:
        df = gatemod.load_metrics(METRICS_DIR)
        band = gatemod.band_for(df, "PRI_NBO", "TEM", 5000, "random")
        sp, ac = band["spearman"]["mean"], band["accuracy"]["mean"]
        kw = _common_kwargs(tmp); kw["source"] = "PRI_NBO"
        kw["finetune_fn"] = _make_finetune_fn(sp, ac)
        m1 = rp.run_variant(**kw)
        m2 = rp.run_variant(**kw)
        ok = True
        ok &= check("first build not from cache", m1["graph"]["loaded_from_cache"] is False)
        ok &= check("second load from cache", m2["graph"]["loaded_from_cache"] is True)
        ok &= check("same pair count", m1["graph"]["n_pairs"] == m2["graph"]["n_pairs"])
        return ok


def test_zeroshot_check_pass():
    print("Test 5: zeroshot wiring check passes when metrics match recorded zero-shot")
    import gate as gatemod
    with tempfile.TemporaryDirectory() as tmp:
        df = gatemod.load_metrics(METRICS_DIR)
        zb = gatemod.band_for(df, "ACC", "TEM", 0, "random", zero_shot=True)
        if zb is None:
            return check("zero-shot band exists for ACC->TEM", False)
        sp = zb["spearman"]["mean"]; ac = zb["accuracy"]["mean"]
        kw = _common_kwargs(tmp); kw["source"] = "ACC"; kw["mode"] = "zeroshot"
        kw["zeroshot_eval_fn"] = lambda *a, **k: {"spearman": sp, "accuracy": ac}
        man = rp.run_variant(**kw)
        ok = True
        ok &= check("status ok", man["status"] == "ok", man["status"])
        ok &= check("zeroshot check passed", man["zeroshot_check"]["result"]["passed"])
        ok &= check("geotiff written", os.path.exists(man["geotiff"]))
        return ok


def test_informed_flows():
    print("Test 6: finetune sampling='informed' now flows through the gate (v2)")
    import gate as gatemod
    with tempfile.TemporaryDirectory() as tmp:
        df = gatemod.load_metrics(METRICS_DIR)
        inf = df[(df["sampling"] == "informed_t1") & (df["sample_size"] == 5000)]
        if not len(inf):
            return check("informed_t1 5000 cell exists", False)
        s, t = inf.iloc[0]["source"], inf.iloc[0]["target"]
        band = gatemod.band_for(df, s, t, 5000, "informed_t1")
        sp, ac = band["spearman"]["mean"], band["accuracy"]["mean"]
        kw = _common_kwargs(tmp)
        kw["source"] = s; kw["target"] = t; kw["sampling"] = "informed"
        kw["iso_model_dir"] = "/iso"
        # resolver mock ignores target gpkg mismatch (uses ACC gpkg); fine for flow test
        kw["finetune_fn"] = _make_finetune_fn(sp, ac)
        try:
            man = rp.run_variant(**kw)
            ok = check("informed run reached status ok", man["status"] == "ok", man["status"])
            ok &= check("gate evaluated for informed", man["gate"]["result"]["passed"])
            return ok
        except NotImplementedError:
            return check("informed no longer raises NotImplementedError", False)


def main():
    tests = [test_finetune_gate_pass, test_finetune_gate_fail, test_missing_band,
             test_graph_cache, test_zeroshot_check_pass, test_informed_flows]
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
