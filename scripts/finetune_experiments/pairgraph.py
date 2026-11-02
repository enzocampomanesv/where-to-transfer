"""
pairgraph.py
============

Generate a per-city, degree-targeted, connectivity-guaranteed random comparison
graph over *all* patches of a city, for city-wide Bradley-Terry livability mapping.

WHY A DEGREE-TARGETED GRAPH (not "N% of all pairs")
---------------------------------------------------
BT score precision for a patch depends on how many times it is compared, not on
what *fraction* of the quadratic pair space is sampled. Sampling a fixed fraction
(e.g. 10% of C(N,2)) wastes inference -- and inference (running the ensembleVoter
over every pair, per model variant) is the dominant cost. Targeting a fixed number
of comparisons per patch makes cost linear in N and keeps BT precision uniform.

The same frozen graph is reused across every model variant for a city, so that
differences between maps reflect the *model*, not the pair sampling. It is
regenerated per city (IDs, CRS, and imagery coverage all differ).

DEGREE SEMANTICS (read this)
----------------------------
The user-facing knob is ``comparisons_per_patch`` == the *mean degree* of the
undirected graph == roughly how many times each patch is compared. Internally we
build a "k-out" graph: each node draws ``out_degree = ceil(comparisons_per_patch/2)``
distinct partners. After undirecting and de-duplicating, mean degree is
approximately ``2 * out_degree`` (a hair less, because a few picks are mutual).
The k-out construction guarantees a **minimum** out-degree, so no patch is left
under-compared with a wild BT score. The report records the *achieved* mean/min/max
degree so the target-vs-reality gap is always visible.

OUTPUT
------
A DataFrame (optionally frozen to parquet) with, per comparison:
    left, right        : image *paths* (feed create_dataset / the BT estimator directly)
    y                  : dummy 0 (create_dataset needs the column; the value is unused
                         -- votes come from the model at inference time)
    left_id, right_id  : integer PARTIMAP_I ids (diagnostics / joins)

plus a JSON provenance report (coverage, degree stats, connectivity repair, seed,
path template, CRS, timestamp) -- the record that ties every derived map back to
exactly this pair set.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_PATH_TEMPLATE = "{root}/votes/{year}/{city}/{d_src}/{city}_{d_src}_{id}.tif"


# ===========================================================================
# Union-Find (for connectivity check + repair)
# ===========================================================================
class _DSU:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

    def components(self) -> Dict[int, List[int]]:
        comps: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            comps.setdefault(self.find(i), []).append(i)
        return comps


# ===========================================================================
# Core graph construction (index-space; no gpkg, no disk -- unit testable)
# ===========================================================================
def _sample_partners(rng: np.random.RandomState, n: int, k: int, self_idx: int) -> List[int]:
    """Sample ``k`` distinct partners in [0, n) \\ {self_idx} via Floyd's algorithm.

    O(k) per node, deterministic given ``rng``. We sample k distinct values from
    the (n-1) non-self indices [0, n-1), then remap any value >= self_idx by +1 so
    the result lives in [0, n) and never equals self_idx.
    """
    m = n - 1  # number of allowable partners
    chosen: set = set()
    # Floyd's algorithm for a distinct sample of size k from range [0, m)
    for j in range(m - k, m):
        t = rng.randint(0, j + 1)  # inclusive of j
        v = t if t not in chosen else j
        chosen.add(v)
    # remap [0, m) -> [0, n) \ {self_idx}
    return [(v + 1) if v >= self_idx else v for v in chosen]


def _kout_edges(n_nodes: int, out_degree: int, seed: int) -> set:
    """Build canonical undirected edges from a seeded k-out draw.

    Each node picks ``out_degree`` distinct partners; edges are stored canonically
    as (min, max) and de-duplicated. Deterministic given (n_nodes, out_degree, seed):
    nodes are visited in index order and the RNG is consumed sequentially.
    """
    rng = np.random.RandomState(seed)
    edges: set = set()
    for i in range(n_nodes):
        for j in _sample_partners(rng, n_nodes, out_degree, i):
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    return edges


def _complete_graph_edges(n_nodes: int) -> set:
    """All C(n,2) undirected edges -- used when the requested degree saturates N."""
    edges = set()
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edges.add((i, j))
    return edges


def _repair_connectivity(n_nodes: int, edges: set, seed: int) -> Tuple[set, int, int]:
    """Ensure a single connected component by linking components minimally.

    Returns (edges, n_components_before, n_repair_edges). Repair links the
    representative of each extra component to the representative of the largest
    component (minimises distortion vs a random chain). Deterministic.
    """
    dsu = _DSU(n_nodes)
    for a, b in edges:
        dsu.union(a, b)
    comps = dsu.components()
    n_before = len(comps)
    if n_before <= 1:
        return edges, n_before, 0

    # Largest component as the hub; sort comp roots by size desc, then by root id
    roots_by_size = sorted(comps.keys(), key=lambda r: (-len(comps[r]), r))
    hub = comps[roots_by_size[0]][0]  # a concrete node in the largest component
    n_repair = 0
    edges = set(edges)
    for r in roots_by_size[1:]:
        node = comps[r][0]
        a, b = (hub, node) if hub < node else (node, hub)
        if (a, b) not in edges:
            edges.add((a, b))
            n_repair += 1
    return edges, n_before, n_repair


def _degree_stats(n_nodes: int, edges: set) -> Dict[str, float]:
    deg = np.zeros(n_nodes, dtype=np.int64)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    return {
        "mean_degree": float(deg.mean()),
        "min_degree": int(deg.min()),
        "max_degree": int(deg.max()),
        "median_degree": float(np.median(deg)),
        "n_isolated": int((deg == 0).sum()),
    }


def build_edges_for_nodes(
    n_nodes: int,
    comparisons_per_patch: int,
    seed: int,
) -> Tuple[List[Tuple[int, int]], Dict]:
    """Index-space graph builder. Returns (sorted_edge_list, diagnostics).

    Separated from gpkg/disk I/O so graph invariants are unit-testable without
    any tiles present. Handles the degree-saturation clamp.
    """
    if n_nodes < 2:
        raise ValueError(f"Need >= 2 nodes to build a comparison graph, got {n_nodes}.")
    if comparisons_per_patch < 1:
        raise ValueError(f"comparisons_per_patch must be >= 1, got {comparisons_per_patch}.")

    out_degree = max(1, math.ceil(comparisons_per_patch / 2))
    saturated = out_degree >= (n_nodes - 1)

    if saturated:
        edges = _complete_graph_edges(n_nodes)
        n_before, n_repair = 1, 0
        clamp_note = (
            f"comparisons_per_patch={comparisons_per_patch} implies out_degree="
            f"{out_degree} >= N-1={n_nodes - 1}; built COMPLETE graph "
            f"({len(edges)} pairs)."
        )
    else:
        edges = _kout_edges(n_nodes, out_degree, seed)
        edges, n_before, n_repair = _repair_connectivity(n_nodes, edges, seed)
        clamp_note = None

    # Final connectivity assertion (must hold after repair)
    dsu = _DSU(n_nodes)
    for a, b in edges:
        dsu.union(a, b)
    n_after = len(dsu.components())
    assert n_after == 1, f"graph still disconnected after repair: {n_after} components"

    # Theoretical mean degree of a k-out graph after undirected de-dup:
    #   an unordered pair is an edge iff either endpoint picks the other, so
    #   E[mean degree] = (N-1) * [1 - (1 - k/(N-1))^2].
    # This equals ~2k when k << N (the real regime) and falls below 2k only when
    # k is a large fraction of N (a small-city / very-high-degree corner we never hit).
    if saturated:
        expected_mean_degree = float(n_nodes - 1)
    else:
        p_not = 1.0 - out_degree / (n_nodes - 1)
        expected_mean_degree = float((n_nodes - 1) * (1.0 - p_not * p_not))

    diagnostics = {
        "out_degree": out_degree,
        "saturated_to_complete": saturated,
        "clamp_note": clamp_note,
        "n_pairs": len(edges),
        "components_before_repair": n_before,
        "n_repair_edges": n_repair,
        "components_after_repair": n_after,
        "expected_mean_degree": round(expected_mean_degree, 3),
        **_degree_stats(n_nodes, edges),
    }
    return sorted(edges), diagnostics


# ===========================================================================
# Path resolution + validation
# ===========================================================================
def build_path(root: str, year, city: str, d_src: str, id_val, template: str) -> str:
    return template.format(root=root.rstrip("/"), year=year, city=city, d_src=d_src, id=id_val)


def _validate_paths(paths: Sequence[str]) -> np.ndarray:
    """Boolean mask of which paths exist on disk."""
    return np.array([os.path.exists(p) for p in paths], dtype=bool)


# ===========================================================================
# Top-level: gpkg -> frozen pair set + provenance report
# ===========================================================================
def build_pair_graph(
    gpkg_path: str,
    city: str,
    root_path: str,
    *,
    year,
    d_src: str = "S1_200m",
    comparisons_per_patch: int = 50,
    seed: int = 20240101,
    id_col: str = "PARTIMAP_I",
    path_template: str = DEFAULT_PATH_TEMPLATE,
    validate_paths: bool = True,
    min_coverage_warn: float = 0.5,
    out_parquet: Optional[str] = None,
    out_report: Optional[str] = None,
    # Test/injection hooks (avoid gpkg + geopandas in unit tests):
    _node_ids: Optional[Sequence[int]] = None,
    _crs: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Build (and optionally freeze) a city-wide degree-targeted comparison graph.

    Parameters
    ----------
    gpkg_path : str
        Path to the city's FULL patch gpkg (e.g. ACC_FN_200m.gpkg). Must contain
        ``id_col`` and a geometry. Ignored if ``_node_ids`` is provided.
    city, root_path, year, d_src :
        Used to construct per-patch image paths via ``path_template``.
    comparisons_per_patch : int
        Target MEAN degree (~how many times each patch is compared). Out-degree is
        derived as ceil(cpp/2). Provisional until the ACC calibration D-sweep fixes it.
    seed : int
        Seed for the k-out draw and connectivity repair. Freeze this per city.
    validate_paths : bool
        If True, drop patches whose .tif is missing before graph construction, so
        no pair points at a nonexistent tile. Set False for dry runs / sandbox tests
        where imagery is absent.
    out_parquet, out_report : str, optional
        If given, write the frozen pair set (parquet) and provenance report (JSON).
    _node_ids, _crs :
        Test hooks to bypass gpkg reading. Do not use in production.

    Returns
    -------
    (pairs_df, report)
        pairs_df columns: [left, right, y, left_id, right_id]  (left/right are paths)
        report: dict of provenance + diagnostics.
    """
    t0 = time.time()

    # --- 1. node ids + CRS (from gpkg, or injected for tests) -------------------
    if _node_ids is not None:
        node_ids = np.asarray(list(_node_ids))
        crs = _crs
        crs_epsg = None
    else:
        import geopandas as gpd  # lazy: only needed for real gpkg reads
        gdf = gpd.read_file(gpkg_path)
        if id_col not in gdf.columns:
            raise ValueError(f"{id_col!r} not in gpkg columns: {list(gdf.columns)}")
        node_ids = gdf[id_col].to_numpy()
        crs = str(gdf.crs) if gdf.crs is not None else None
        # Canonical, version-independent CRS id (str(crs) case varies by pyproj version)
        crs_epsg = gdf.crs.to_epsg() if gdf.crs is not None else None

    node_ids = np.asarray(node_ids)
    if node_ids.size != np.unique(node_ids).size:
        raise ValueError("Duplicate ids in gpkg; PARTIMAP_I must be unique.")

    n_total = int(node_ids.size)

    # --- 2. resolve paths -------------------------------------------------------
    all_paths = [
        build_path(root_path, year, city, d_src, int(i), path_template) for i in node_ids
    ]

    # --- 3. validate imagery presence ------------------------------------------
    dropped_ids: List[int] = []
    if validate_paths:
        mask = _validate_paths(all_paths)
        if not mask.all():
            dropped_ids = [int(i) for i, m in zip(node_ids, mask) if not m]
            node_ids = node_ids[mask]
            all_paths = [p for p, m in zip(all_paths, mask) if m]
    n_with_imagery = int(node_ids.size)

    if n_with_imagery < 2:
        raise ValueError(
            f"After path validation only {n_with_imagery} patches have imagery; "
            f"cannot build a graph. Check root_path/year/d_src/path_template."
        )
    coverage = n_with_imagery / n_total if n_total else 0.0

    # --- 4. build the graph in index space -------------------------------------
    edges, diag = build_edges_for_nodes(n_with_imagery, comparisons_per_patch, seed)

    # --- 5. materialize with paths (canonical left<right by index) -------------
    li = np.fromiter((a for a, _ in edges), dtype=np.int64, count=len(edges))
    ri = np.fromiter((b for _, b in edges), dtype=np.int64, count=len(edges))
    paths_arr = np.asarray(all_paths, dtype=object)
    ids_arr = np.asarray(node_ids)

    pairs_df = pd.DataFrame({
        "left":     paths_arr[li],
        "right":    paths_arr[ri],
        "y":        np.zeros(len(edges), dtype=np.int64),   # dummy; unused at inference
        "left_id":  ids_arr[li],
        "right_id": ids_arr[ri],
    })

    # --- 6. provenance report ---------------------------------------------------
    report = {
        "city": city,
        "gpkg_path": None if _node_ids is not None else gpkg_path,
        "crs": crs,
        "crs_epsg": crs_epsg,
        "year": year,
        "d_src": d_src,
        "id_col": id_col,
        "path_template": path_template,
        "root_path": root_path,
        "comparisons_per_patch_target": comparisons_per_patch,
        "seed": seed,
        "validate_paths": validate_paths,
        "n_total_patches": n_total,
        "n_with_imagery": n_with_imagery,
        "n_dropped_no_imagery": len(dropped_ids),
        "coverage_fraction": round(coverage, 6),
        "dropped_ids": dropped_ids,
        "build_seconds": round(time.time() - t0, 2),
        **diag,
    }

    if validate_paths and coverage < min_coverage_warn:
        report["coverage_warning"] = (
            f"Only {coverage:.1%} of patches have imagery (< {min_coverage_warn:.0%}). "
            f"Verify path_template/year/d_src."
        )

    # --- 7. freeze ------------------------------------------------------------
    if out_parquet is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_parquet)), exist_ok=True)
        # dictionary encoding keeps repeated path strings small
        pairs_df.to_parquet(out_parquet, index=False)
        report["out_parquet"] = out_parquet
    if out_report is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_report)), exist_ok=True)
        with open(out_report, "w") as f:
            json.dump(report, f, indent=2)
        report["out_report"] = out_report

    return pairs_df, report
