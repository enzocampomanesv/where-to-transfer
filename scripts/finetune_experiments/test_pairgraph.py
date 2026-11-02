"""
test_pairgraph.py
=================

Two layers:
  (A) Graph invariants on synthetic node counts -- no gpkg, no tiles on disk.
  (B) Real-ingestion smoke test on the uploaded ACC_FN_200m.gpkg with
      validate_paths=False (exercises gpkg read + path templating minus disk checks,
      which only the user's machine can do).

Run:  python test_pairgraph.py
Exit code 0 = all checks passed.
"""

import os
import sys
import traceback

import numpy as np

import pairgraph as pg

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(condition)


def _degrees(n, edges):
    deg = np.zeros(n, dtype=int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return deg


# ---------------------------------------------------------------------------
# A1 -- core invariants across a range of N and target degree
# ---------------------------------------------------------------------------
def test_core_invariants():
    print("Test A1: k-out graph invariants (min-degree, no self/dup, connected)")
    ok = True
    for n in [50, 500, 2416, 5523]:          # incl. CHI-size and ACC-size
        for cpp in [10, 40, 50]:
            edges, diag = pg.build_edges_for_nodes(n, cpp, seed=123)
            out_deg = diag["out_degree"]

            # no self-loops
            no_self = all(a != b for a, b in edges)
            # canonical + de-duplicated (set of tuples already guarantees dedup;
            # verify canonical ordering a<b)
            canonical = all(a < b for a, b in edges)
            # min degree >= out_degree (k-out guarantee)
            deg = _degrees(n, edges)
            min_ok = deg.min() >= out_deg
            # connected
            connected = diag["components_after_repair"] == 1
            # achieved mean degree must match the EXACT k-out expectation
            # (N-1)*[1-(1-k/(N-1))^2], which is ~2k only when k << N. Checking
            # against the naive 2k would (correctly) fail in the small-N corner.
            exp_mean = diag["expected_mean_degree"]
            mean_ok = abs(diag["mean_degree"] - exp_mean) <= max(0.5, 0.03 * exp_mean)

            passed = no_self and canonical and min_ok and connected and mean_ok
            ok &= passed
            if not passed:
                check(f"N={n}, cpp={cpp}", False,
                      f"self_ok={no_self} canon={canonical} min_ok={min_ok} "
                      f"conn={connected} mean={diag['mean_degree']:.1f} "
                      f"(expected {exp_mean:.1f})")
    check("all (N, cpp) combinations satisfy invariants", ok)
    return ok


# ---------------------------------------------------------------------------
# A2 -- determinism: same seed => identical edges; different seed => different
# ---------------------------------------------------------------------------
def test_determinism():
    print("Test A2: determinism across runs")
    e1, _ = pg.build_edges_for_nodes(1000, 40, seed=7)
    e2, _ = pg.build_edges_for_nodes(1000, 40, seed=7)
    e3, _ = pg.build_edges_for_nodes(1000, 40, seed=8)
    ok = True
    ok &= check("same seed -> identical edge list", e1 == e2)
    ok &= check("different seed -> different edge list", e1 != e3)
    return ok


# ---------------------------------------------------------------------------
# A3 -- degree saturation clamp -> complete graph
# ---------------------------------------------------------------------------
def test_saturation_clamp():
    print("Test A3: degree saturation clamps to complete graph")
    n = 8
    # cpp high enough that out_degree = ceil(cpp/2) >= n-1
    edges, diag = pg.build_edges_for_nodes(n, comparisons_per_patch=20, seed=1)
    expected = n * (n - 1) // 2
    ok = True
    ok &= check("flagged saturated", diag["saturated_to_complete"] is True)
    ok &= check("produced complete graph", diag["n_pairs"] == expected,
                f"{diag['n_pairs']} vs {expected}")
    return ok


# ---------------------------------------------------------------------------
# A4 -- guards
# ---------------------------------------------------------------------------
def test_guards():
    print("Test A4: input guards")
    ok = True
    for bad_n in [0, 1]:
        try:
            pg.build_edges_for_nodes(bad_n, 10, seed=1)
            ok &= check(f"N={bad_n} raises", False)
        except ValueError:
            ok &= check(f"N={bad_n} raises", True)
    try:
        pg.build_edges_for_nodes(100, 0, seed=1)
        ok &= check("cpp=0 raises", False)
    except ValueError:
        ok &= check("cpp=0 raises", True)
    return ok


# ---------------------------------------------------------------------------
# A5 -- connectivity repair actually triggers and fixes a fragmented graph
# ---------------------------------------------------------------------------
def test_connectivity_repair():
    print("Test A5: connectivity repair on a deliberately fragmented graph")
    # Two disjoint cliques -> 2 components; repair must link them.
    edges = {(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)}
    repaired, n_before, n_repair = pg._repair_connectivity(6, edges, seed=1)
    dsu = pg._DSU(6)
    for a, b in repaired:
        dsu.union(a, b)
    ok = True
    ok &= check("detected 2 components before", n_before == 2, f"{n_before}")
    ok &= check("added exactly 1 repair edge", n_repair == 1, f"{n_repair}")
    ok &= check("single component after", len(dsu.components()) == 1)
    return ok


# ---------------------------------------------------------------------------
# A6 -- full build via injected node ids (no gpkg): path materialization + report
# ---------------------------------------------------------------------------
def test_full_build_injected():
    print("Test A6: full build with injected ids (paths + report, no disk)")
    ids = [30260, 30261, 30262, 30486, 30487, 58746, 40000, 41234, 41235, 41236]
    pairs_df, report = pg.build_pair_graph(
        gpkg_path="(injected)", city="ACC", root_path="/data/PARTIMAP",
        year=2022, d_src="S1_200m", comparisons_per_patch=6, seed=42,
        validate_paths=False, _node_ids=ids, _crs="EPSG:32630",
    )
    ok = True
    ok &= check("columns present",
                list(pairs_df.columns) == ["left", "right", "y", "left_id", "right_id"])
    # a sample path matches the expected template
    sample = pairs_df.iloc[0]["left"]
    ok &= check("path template correct",
                sample.startswith("/data/PARTIMAP/votes/2022/ACC/S1_200m/ACC_S1_200m_")
                and sample.endswith(".tif"), sample)
    # left_id/right_id are real ids and left path corresponds to left_id
    lid = int(pairs_df.iloc[0]["left_id"])
    ok &= check("left path id matches left_id",
                sample.endswith(f"_{lid}.tif"), f"lid={lid}")
    ok &= check("y is all zero", (pairs_df["y"] == 0).all())
    ok &= check("report coverage == 1.0 (validate off)", report["coverage_fraction"] == 1.0)
    ok &= check("report records seed + crs",
                report["seed"] == 42 and report["crs"] == "EPSG:32630")
    return ok


# ---------------------------------------------------------------------------
# B  -- real ACC gpkg ingestion (validate_paths=False; no tiles needed)
# ---------------------------------------------------------------------------
def test_real_acc_gpkg():
    print("Test B: real ACC_FN_200m.gpkg ingestion (validate_paths=False)")
    gpkg = "clustering/ACC/ACC_FN_200m.gpkg"
    pairs_df, report = pg.build_pair_graph(
        gpkg_path=gpkg, city="ACC", root_path=os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"),
        year=2022, d_src="S1_200m", comparisons_per_patch=50, seed=20240101,
        validate_paths=False,
    )
    ok = True
    ok &= check("read N=5523 patches", report["n_total_patches"] == 5523,
                f"{report['n_total_patches']}")
    ok &= check("CRS read from gpkg (EPSG code, version-independent)",
                report["crs_epsg"] == 32630, f"{report['crs_epsg']}")
    ok &= check("connected (1 component)", report["components_after_repair"] == 1)
    ok &= check("min degree >= out_degree",
                report["min_degree"] >= report["out_degree"],
                f"min={report['min_degree']} out={report['out_degree']}")
    ok &= check("mean degree ~ cpp target (50)",
                45 <= report["mean_degree"] <= 51, f"{report['mean_degree']:.1f}")
    # every path matches the expected on-disk convention
    p = pairs_df.iloc[0]["left"]
    ok &= check("path convention matches sample from earlier turns",
                "/votes/2022/ACC/S1_200m/ACC_S1_200m_" in p and p.endswith(".tif"), p)
    # ids in paths are within the gpkg id range seen during inspection
    ok &= check("pair count ~ N * out_degree",
                abs(report["n_pairs"] - 5523 * report["out_degree"]) / report["n_pairs"] < 0.15,
                f"{report['n_pairs']} pairs, out_degree={report['out_degree']}")
    print(f"       ACC: N={report['n_total_patches']}, out_degree={report['out_degree']}, "
          f"pairs={report['n_pairs']}, mean_deg={report['mean_degree']:.1f}, "
          f"build={report['build_seconds']}s")
    return ok


def main():
    tests = [
        test_core_invariants,
        test_determinism,
        test_saturation_clamp,
        test_guards,
        test_connectivity_repair,
        test_full_build_injected,
        test_real_acc_gpkg,
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
    n_pass = sum(results)
    print("=" * 60)
    print(f"RESULT: {n_pass}/{len(results)} test groups passed")
    print("=" * 60)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
