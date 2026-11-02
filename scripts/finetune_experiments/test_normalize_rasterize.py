"""
test_normalize_rasterize.py
===========================

normalize.py  -- ECDF/z-score properties, difference maps.
rasterize.py  -- geometry round-trip on the REAL ACC (EPSG:32630) and
                 MUM (EPSG:32643) gpkgs: write a GeoTIFF, read it back, and confirm
                 every cell's value lands in the pixel containing its centroid, in
                 the right CRS and pixel size, with no collisions.

Run:  python test_normalize_rasterize.py   (exit 0 = all passed)
"""

import os
import sys
import tempfile
import traceback

import numpy as np
import pandas as pd

import geopandas as gpd
import rasterio

import normalize as nz
import rasterize as rz

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    return bool(cond)


# ===========================================================================
# normalize.py
# ===========================================================================
def test_ecdf_reference_uniform():
    print("Test N1: reference normalised against itself is ~uniform and order-preserving")
    rng = np.random.RandomState(0)
    ref = rng.randn(2000)
    frame = pd.DataFrame({"image_id": np.arange(2000), "bt_score": ref})
    out = nz.normalize_map(frame, frame, method="ecdf")
    v = out["bt_score_ecdf"].to_numpy()
    ok = True
    ok &= check("values in [0,1]", v.min() >= 0.0 and v.max() <= 1.0,
                f"[{v.min():.3f},{v.max():.3f}]")
    # approx uniform: sorted values close to linspace
    order_pres = np.corrcoef(np.argsort(np.argsort(ref)),
                             np.argsort(np.argsort(v)))[0, 1]
    ok &= check("strictly order-preserving vs raw (rank corr = 1)",
                abs(order_pres - 1.0) < 1e-12, f"{order_pres:.6f}")
    sorted_v = np.sort(v)
    max_dev = np.max(np.abs(sorted_v - np.linspace(sorted_v[0], sorted_v[-1], len(v))))
    ok &= check("distribution ~uniform (max dev from linspace < 0.02)",
                max_dev < 0.02, f"{max_dev:.4f}")
    return ok


def test_ecdf_saturation_and_ties():
    print("Test N2: out-of-range saturation and tie handling")
    ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ok = True
    below = nz.reference_ecdf(np.array([-5.0]), ref)[0]
    above = nz.reference_ecdf(np.array([100.0]), ref)[0]
    ok &= check("below-range -> 0.0", below == 0.0, f"{below}")
    ok &= check("above-range -> 1.0", above == 1.0, f"{above}")
    # ties: value equal to a ref point gets midrank
    reft = np.array([0.0, 0.0, 1.0, 1.0])   # two ties at each
    mid = nz.reference_ecdf(np.array([0.0]), reft)[0]   # #<0=0, #<=0=2 -> (0+2)/8=0.25
    ok &= check("tie midrank correct", abs(mid - 0.25) < 1e-12, f"{mid}")
    return ok


def test_difference_self_zero():
    print("Test N3: difference of a normalised map with itself is exactly zero")
    rng = np.random.RandomState(1)
    frame = pd.DataFrame({"image_id": np.arange(500), "bt_score": rng.randn(500)})
    norm = nz.normalize_map(frame, frame, method="ecdf")
    diff = nz.difference_map(norm, norm, value_col="bt_score_ecdf")
    return check("all differences == 0", np.allclose(diff["difference"], 0.0, atol=1e-15),
                 f"max |diff| {np.max(np.abs(diff['difference'])):.2e}")


def test_difference_alignment():
    print("Test N4: difference aligns by image_id, not row order")
    a = pd.DataFrame({"image_id": [1, 2, 3, 4], "v": [0.9, 0.8, 0.7, 0.6]})
    b = pd.DataFrame({"image_id": [4, 3, 2, 1], "v": [0.1, 0.2, 0.3, 0.4]})  # shuffled
    diff = nz.difference_map(a, b, value_col="v").sort_values("image_id")
    expected = np.array([0.9 - 0.4, 0.8 - 0.3, 0.7 - 0.2, 0.6 - 0.1])
    return check("differences respect image_id join",
                 np.allclose(diff["difference"].to_numpy(), expected),
                 f"{diff['difference'].to_numpy()}")


# ===========================================================================
# rasterize.py -- real gpkgs
# ===========================================================================
def _roundtrip_city(gpkg, expected_epsg):
    gdf = gpd.read_file(gpkg)
    # value = the cell id itself -> unambiguous round-trip check
    frame = pd.DataFrame({
        "image_id": gdf["PARTIMAP_I"].to_numpy(),
        "bt_score": gdf["PARTIMAP_I"].to_numpy().astype(float),
    })
    with tempfile.TemporaryDirectory() as td:
        res = rz.rasterize_frame(
            frame, gdf, td, value_cols=["bt_score"], basename="rt",
            tags={"test": "roundtrip"},
        )
        tif = res["geotiffs"]["bt_score"]
        with rasterio.open(tif) as src:
            epsg = src.crs.to_epsg()
            a, e = src.transform.a, src.transform.e
            data = src.read(1)
            # sample each cell's centroid, compare to the id we burned
            cent = gdf.geometry.centroid
            rows, cols = rasterio.transform.rowcol(
                src.transform, cent.x.to_numpy(), cent.y.to_numpy())
            rows = np.asarray(rows); cols = np.asarray(cols)
            sampled = data[rows, cols]
            truth = gdf["PARTIMAP_I"].to_numpy().astype(float)
            mism = int(np.sum(sampled != truth))
            n_finite = int(np.sum(np.isfinite(data)))
    return {
        "epsg": epsg, "px_a": a, "px_e": e,
        "mismatches": mism, "n_cells": len(gdf), "n_finite": n_finite,
        "counts": res.get("coverage_counts"),
    }


def test_rasterize_acc():
    print("Test R1: ACC round-trip (EPSG:32630, 200 m)")
    r = _roundtrip_city("clustering/ACC/ACC_FN_200m.gpkg", 32630)
    ok = True
    ok &= check("CRS EPSG:32630", r["epsg"] == 32630, f"{r['epsg']}")
    ok &= check("pixel 200 x -200", r["px_a"] == 200 and r["px_e"] == -200,
                f"{r['px_a']} x {r['px_e']}")
    ok &= check("every cell value round-trips to its centroid pixel",
                r["mismatches"] == 0, f"{r['mismatches']} mismatches")
    ok &= check("exactly N finite pixels (one per cell, no collisions)",
                r["n_finite"] == r["n_cells"], f"{r['n_finite']} vs {r['n_cells']}")
    return ok


def test_rasterize_mum():
    print("Test R2: MUM round-trip (EPSG:32643, 200 m) -- different CRS + largest N")
    r = _roundtrip_city("clustering/MUM/MUM_FN_200m.gpkg", 32643)
    ok = True
    ok &= check("CRS EPSG:32643", r["epsg"] == 32643, f"{r['epsg']}")
    ok &= check("pixel 200 x -200", r["px_a"] == 200 and r["px_e"] == -200,
                f"{r['px_a']} x {r['px_e']}")
    ok &= check("every cell value round-trips", r["mismatches"] == 0,
                f"{r['mismatches']} mismatches")
    ok &= check("one finite pixel per cell", r["n_finite"] == r["n_cells"],
                f"{r['n_finite']} vs {r['n_cells']}")
    return ok


def test_rasterize_nodata_and_mask():
    print("Test R3: unscored cells -> NaN; coverage mask categories correct")
    gdf = gpd.read_file("clustering/ACC/ACC_FN_200m.gpkg")
    ids = gdf["PARTIMAP_I"].to_numpy()
    # score only 80%; mark 5% as 'no imagery'; rest unscored-other
    rng = np.random.RandomState(0)
    scored_mask = rng.rand(len(ids)) < 0.80
    scored_ids = ids[scored_mask]
    remaining = ids[~scored_mask]
    no_img = remaining[: max(1, len(remaining) // 2)]
    frame = pd.DataFrame({"image_id": scored_ids,
                          "bt_score": rng.randn(len(scored_ids))})
    with tempfile.TemporaryDirectory() as td:
        res = rz.rasterize_frame(frame, gdf, td, value_cols=["bt_score"],
                                 basename="cov", no_imagery_ids=no_img)
        with rasterio.open(res["geotiffs"]["bt_score"]) as src:
            data = src.read(1)
        n_finite = int(np.sum(np.isfinite(data)))
        counts = res["coverage_counts"]
    ok = True
    ok &= check("finite pixels == scored cells",
                n_finite == len(scored_ids), f"{n_finite} vs {len(scored_ids)}")
    ok &= check("mask scored count matches", counts["scored"] == len(scored_ids))
    ok &= check("mask no_imagery count matches", counts["no_imagery"] == len(no_img))
    ok &= check("all AOI cells categorised",
                sum(counts.values()) == len(ids),
                f"{sum(counts.values())} vs {len(ids)}")
    return ok


def test_metadata_tags():
    print("Test R4: metadata tags are written and readable")
    gdf = gpd.read_file("clustering/TEM/TEM_FN_200m.gpkg")
    frame = pd.DataFrame({"image_id": gdf["PARTIMAP_I"].to_numpy(),
                          "bt_score": np.arange(len(gdf), dtype=float)})
    with tempfile.TemporaryDirectory() as td:
        res = rz.rasterize_frame(
            frame, gdf, td, value_cols=["bt_score"], basename="meta",
            tags={"source": "ACC", "target": "TEM", "sample_size": 5000,
                  "sampling": "informed", "seed": 42, "D": 50, "normalization": "raw"})
        with rasterio.open(res["geotiffs"]["bt_score"]) as src:
            t = src.tags()
    ok = True
    ok &= check("source/target tags present",
                t.get("source") == "ACC" and t.get("target") == "TEM")
    ok &= check("sampling/seed tags present",
                t.get("sampling") == "informed" and t.get("seed") == "42")
    return ok


def main():
    tests = [
        test_ecdf_reference_uniform,
        test_ecdf_saturation_and_ties,
        test_difference_self_zero,
        test_difference_alignment,
        test_rasterize_acc,
        test_rasterize_mum,
        test_rasterize_nodata_and_mask,
        test_metadata_tags,
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
