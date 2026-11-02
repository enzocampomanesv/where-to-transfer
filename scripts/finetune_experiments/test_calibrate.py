"""
test_calibrate.py
=================

registry.py       -- source-code resolution incl. leave-target-out aggregates,
                     model-path templating, city-config/year precedence.
calibrate_degree  -- run_calibration core with a mock model + mock preprocessing:
                     sweep converges to the reference, patch cache built ONCE across
                     all degrees, recommended-D rule fires correctly.

Run:  python test_calibrate.py   (exit 0 = all passed)
"""

import sys
import traceback

import numpy as np
import pandas as pd

import registry as reg
import calibrate_degree as cal

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


# ===========================================================================
# registry
# ===========================================================================
def test_effective_source_code():
    print("Test G1: source-code resolution (leave-target-out for aggregates)")
    ok = True
    ok &= check("single-city source unchanged",
                reg.effective_source_code("NBO", "ACC") == "NBO")
    ok &= check("aggregate PRI on member ACC -> PRI_ACC (no leakage)",
                reg.effective_source_code("PRI", "ACC") == "PRI_ACC")
    ok &= check("aggregate PRI on non-member CHI -> PRI (full)",
                reg.effective_source_code("PRI", "CHI") == "PRI")
    ok &= check("aggregate AFR on member TEM -> AFR_TEM",
                reg.effective_source_code("AFR", "TEM") == "AFR_TEM")
    ok &= check("aggregate AFR on non-member MUM -> AFR",
                reg.effective_source_code("AFR", "MUM") == "AFR")
    return ok


def test_model_path():
    print("Test G2: model path templating")
    root = "/data/PARTIMAP"
    p = reg.resolve_model_path("ACC", "TEM", root)
    ok = True
    ok &= check("single-city path",
                p == f"{root}/trained_models/voter/ACC_voter_communities_ensembleVoter_S1_200m_1_bradley_terry.ckpt",
                p)
    p2 = reg.resolve_model_path("PRI", "ACC", root)
    ok &= check("aggregate leave-out path uses PRI_ACC",
                "PRI_ACC_voter_communities_ensembleVoter" in p2, p2)
    return ok


def test_city_config_precedence():
    print("Test G3: city-config year/gpkg precedence")
    ok = True
    c = reg.get_city_config("ACC", "/data", clustering_dir="clustering")
    ok &= check("default year 2022", c.year == 2022, f"{c.year}")
    ok &= check("gpkg from clustering_dir under root",
                c.gpkg_path == "/data/clustering/ACC/ACC_FN_200m.gpkg", c.gpkg_path)
    c2 = reg.get_city_config("MUM", "/data", clustering_dir="clustering",
                             year_overrides={"MUM": 2023})
    ok &= check("year override applied", c2.year == 2023, f"{c2.year}")
    c3 = reg.get_city_config("ACC", "/data", gpkg_path="/explicit/x.gpkg", year=2021)
    ok &= check("explicit year + gpkg win", c3.year == 2021 and c3.gpkg_path == "/explicit/x.gpkg")
    # clustering_dir has a default now, so no missing-info error path to test
    return ok


# ===========================================================================
# calibration core (mock model + mock preprocessing)
# ===========================================================================
def _strengths(n, seed=0):
    return np.random.RandomState(seed).randn(n)


def _id_of(path):
    return int(path.split("_")[-1].split(".")[0]) - 30000


class MockPreprocess:
    def __init__(self, strengths):
        self.strengths = strengths
        self.calls = 0

    def __call__(self, fp, sub_bands, norm_data=False):
        self.calls += 1
        return np.full((20, 20, 2), self.strengths[_id_of(fp)], dtype=np.float64)


class MockModel:
    """P(left) = sigmoid(mean_left - mean_right); deterministic 'voter'."""
    def predict_on_batch(self, inputs):
        lb, rb = inputs
        sl = lb.reshape(lb.shape[0], -1).mean(axis=1)
        sr = rb.reshape(rb.shape[0], -1).mean(axis=1)
        return (1.0 / (1.0 + np.exp(-(sl - sr)))).reshape(-1, 1)


def test_calibration_core():
    print("Test C1: sweep converges to reference; cache built once; D recommended")
    n_nodes = 400
    strengths = _strengths(n_nodes, seed=3)
    ids = list(range(30000, 30000 + n_nodes))
    pre = MockPreprocess(strengths)

    result = cal.run_calibration(
        MockModel(), pre,
        _node_ids=ids, city="ACC", root_path="/x", year=2022, d_src="S1_200m",
        degrees=(10, 20, 40, 60), reference_degree=200, seed=7,
        validate_paths=False, progress=False, spearman_target=0.99,
    )
    curve = result["curve"]
    ok = True
    # Spearman should rise with degree and be high at the top
    sp = curve["spearman_vs_ref"].to_numpy()
    ok &= check("Spearman non-decreasing-ish and high at top degree",
                sp[-1] >= sp[0] and sp[-1] > 0.95,
                f"{np.round(sp, 4).tolist()}")
    # cache built ONCE: preprocess calls == unique patches (not summed over degrees)
    n_unique = n_nodes  # every node appears (min-degree >= 1)
    ok &= check("patch cache preprocessed once across all degrees",
                pre.calls == n_unique, f"{pre.calls} calls for {n_unique} patches")
    # recommendation is one of the swept degrees (or reference fallback)
    ok &= check("recommended D is valid",
                result["recommended_degree"] in list(curve["degree"]) + [200],
                f"{result['recommended_degree']}")
    return ok


def test_recommendation_rule():
    print("Test C2: recommended D = smallest degree meeting the Spearman target")
    # craft a curve directly and re-run the selection logic via a tiny stub
    curve = pd.DataFrame({
        "degree": [10, 20, 40, 60],
        "spearman_vs_ref": [0.90, 0.995, 0.9995, 0.9999],
        "n_pairs": [0, 0, 0, 0], "mean_degree": [10, 20, 40, 60], "n_shared": [1, 1, 1, 1],
        "kendall_vs_ref": [0, 0, 0, 0],
    })
    target = 0.999
    meeting = curve[curve["spearman_vs_ref"] >= target]
    rec = int(meeting["degree"].iloc[0]) if len(meeting) else 200
    ok = check("picks 40 (first >= 0.999), not 20 or 60", rec == 40, f"{rec}")
    # fallback when none meet
    curve2 = curve.copy(); curve2["spearman_vs_ref"] = [0.8, 0.9, 0.95, 0.98]
    meeting2 = curve2[curve2["spearman_vs_ref"] >= target]
    rec2 = int(meeting2["degree"].iloc[0]) if len(meeting2) else 200
    ok &= check("falls back to reference when none meet target", rec2 == 200, f"{rec2}")
    return ok


def test_diagnostics_core():
    print("Test C3: diagnostics core (seed- and reference-stability; cache once)")
    n_nodes = 500
    strengths = _strengths(n_nodes, seed=8)
    ids = list(range(30000, 30000 + n_nodes))
    pre = MockPreprocess(strengths)

    diag = cal.run_diagnostics(
        MockModel(), pre,
        _node_ids=ids, city="ACC", root_path="/x", year=2022, d_src="S1_200m",
        diag_degree=40, diag_seeds=(11, 12),
        ref_degrees=(80, 120), ref_seed=5,
        validate_paths=False, progress=False,
    )
    ss = diag["seed_stability"]; rs = diag["reference_stability"]
    ok = True
    ok &= check("seed-stability high (two seeds agree at fixed D)",
                ss["spearman"] > 0.95, f"{ss['spearman']:.4f}")
    ok &= check("reference-stability high (80 vs 120 agree)",
                rs["spearman"] > 0.95, f"{rs['spearman']:.4f}")
    # all four graphs share one cache -> each unique patch preprocessed once
    ok &= check("patch cache preprocessed once across all four graphs",
                pre.calls == n_nodes, f"{pre.calls} calls for {n_nodes} patches")
    ok &= check("seeds/degrees recorded",
                ss["degree"] == 40 and rs["degrees"] == [80, 120])
    return ok


def test_seed_stability_sweep():
    print("Test C4: seed-stability sweep (rises with D; freezes smallest >= target; cache once)")
    n_nodes = 500
    strengths = _strengths(n_nodes, seed=13)
    ids = list(range(30000, 30000 + n_nodes))
    pre = MockPreprocess(strengths)

    sweep = cal.run_seed_stability_sweep(
        MockModel(), pre,
        _node_ids=ids, city="ACC", root_path="/x", year=2022, d_src="S1_200m",
        degrees=(20, 40, 80, 160), seeds=(11, 12),
        stability_target=0.98, validate_paths=False, progress=False,
    )
    curve = sweep["curve"]
    sp = curve["seed_stability_spearman"].to_numpy()
    ok = True
    ok &= check("seed-stability rises with D (top >= bottom)",
                sp[-1] >= sp[0], f"{np.round(sp, 4).tolist()}")
    # cache built once across all 8 graphs (4 degrees x 2 seeds)
    ok &= check("patch cache preprocessed once across all 8 graphs",
                pre.calls == n_nodes, f"{pre.calls} calls for {n_nodes} patches")
    # recommendation logic: smallest D meeting target, or None
    meeting = curve[curve["seed_stability_spearman"] >= 0.98]
    expected = int(meeting["degree"].iloc[0]) if len(meeting) else None
    ok &= check("recommends smallest D meeting target",
                sweep["recommended_degree"] == expected, f"{sweep['recommended_degree']}")
    return ok


def main():
    tests = [
        test_effective_source_code,
        test_model_path,
        test_city_config_precedence,
        test_calibration_core,
        test_recommendation_rule,
        test_diagnostics_core,
        test_seed_stability_sweep,
    ]
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
