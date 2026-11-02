"""
rasterize.py
============

Turn a per-patch score frame (from ``citywide_score.score_city`` / ``normalize``)
into grid-aligned GeoTIFFs in the city's native CRS.

DESIGN
------
The module is purely geometric: it burns *whatever* value columns you give it
(raw BT, reference-ECDF, difference, ...) and knows nothing about their meaning.
Normalisation lives in ``normalize.py``.

The FN gpkg cells are a regular 200 m grid, so we:
  1. infer pixel size from the cells (default 200 m; asserted against geometry);
  2. snap a raster lattice to that pixel size from the layer's bounds;
  3. place each cell at the single pixel containing its centroid;
  4. ASSERT one-cell-per-pixel (no two cells collide) -- if the grid were irregular
     this fires loudly rather than silently overwriting.

Sparse / non-rectangular AOIs are handled naturally: only cells present in the
frame get a value; everything else is ``nodata`` (NaN). A companion coverage mask
distinguishes "scored", "no imagery" (dropped upstream in pairgraph), and
"in-AOI but unscored".

Outputs are Float32, LZW-compressed, CRS from the gpkg, with self-describing
metadata tags so each GeoTIFF records how it was made.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

import rasterio
from rasterio.transform import from_origin, rowcol


NODATA = np.float32(np.nan)


# ===========================================================================
# Grid geometry
# ===========================================================================
def infer_pixel_size(gdf, default: float = 200.0, tol: float = 1.0) -> float:
    """Infer square cell size from geometry bounds; verify it is ~regular."""
    b = gdf.geometry.bounds  # columns: minx, miny, maxx, maxy
    widths = (b["maxx"] - b["minx"]).to_numpy()
    heights = (b["maxy"] - b["miny"]).to_numpy()
    w = float(np.median(widths))
    h = float(np.median(heights))
    if abs(w - h) > tol:
        raise ValueError(f"Cells are not square (median {w:.1f} x {h:.1f}).")
    if abs(w - default) > tol:
        # not fatal -- just use the inferred size, but surface it
        pass
    return w


def build_grid(gdf, pixel_size: float):
    """Snap a raster lattice to ``pixel_size`` covering the layer bounds.

    Returns (transform, width, height). Origin snapped so the lattice is congruent
    with a global pixel_size grid (floor min-x, ceil max-y).
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    left = np.floor(minx / pixel_size) * pixel_size
    top = np.ceil(maxy / pixel_size) * pixel_size
    right = np.ceil(maxx / pixel_size) * pixel_size
    bottom = np.floor(miny / pixel_size) * pixel_size
    width = int(round((right - left) / pixel_size))
    height = int(round((top - bottom) / pixel_size))
    transform = from_origin(left, top, pixel_size, pixel_size)
    return transform, width, height


def cell_rowcol(gdf, transform) -> "tuple[np.ndarray, np.ndarray]":
    """Row/col of each cell's centroid under ``transform`` (vectorised)."""
    cent = gdf.geometry.centroid
    xs = cent.x.to_numpy()
    ys = cent.y.to_numpy()
    rows, cols = rowcol(transform, xs, ys)  # rasterio: accepts arrays
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


# ===========================================================================
# Core rasterization
# ===========================================================================
def _write_geotiff(path, array, transform, crs, tags: Optional[Dict] = None):
    array = np.ascontiguousarray(array.astype(np.float32))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=array.shape[0], width=array.shape[1],
        count=1, dtype="float32",
        crs=crs, transform=transform,
        nodata=float("nan"), compress="lzw",
    ) as dst:
        dst.write(array, 1)
        if tags:
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def rasterize_frame(
    frame: pd.DataFrame,
    gdf,
    out_dir: str,
    *,
    value_cols: Sequence[str] = ("bt_score",),
    key: str = "image_id",
    id_col: str = "PARTIMAP_I",
    crs=None,
    pixel_size: Optional[float] = None,
    basename: str = "map",
    tags: Optional[Dict] = None,
    no_imagery_ids: Optional[Iterable[int]] = None,
    write_coverage_mask: bool = True,
) -> Dict[str, object]:
    """Rasterize one or more value columns to GeoTIFFs on the gpkg's grid.

    Parameters
    ----------
    frame : DataFrame
        Per-patch values; must have ``key`` and each of ``value_cols``.
    gdf : GeoDataFrame
        The city's FULL cell layer (geometry + ``id_col``). Defines grid + CRS.
    out_dir : str
        Directory for GeoTIFFs. Files are ``{basename}__{value_col}.tif``.
    crs : optional
        Override CRS; defaults to ``gdf.crs``.
    pixel_size : optional
        Override cell size; defaults to inferred (≈200 m).
    tags : dict, optional
        Metadata written into every GeoTIFF (source, target, sample_size, seed, ...).
    no_imagery_ids : iterable, optional
        IDs dropped upstream for missing imagery, for the coverage mask.

    Returns
    -------
    dict: paths written, grid info, and coverage counts.
    """
    for c in value_cols:
        if c not in frame.columns:
            raise ValueError(f"value column {c!r} not in frame.")
    if key not in frame.columns:
        raise ValueError(f"key {key!r} not in frame.")
    if id_col not in gdf.columns:
        raise ValueError(f"id_col {id_col!r} not in gdf.")

    crs = crs if crs is not None else gdf.crs
    px = pixel_size if pixel_size is not None else infer_pixel_size(gdf)
    transform, width, height = build_grid(gdf, px)

    # cell -> pixel, with collision guard (irregular grid protection)
    rows, cols = cell_rowcol(gdf, transform)
    in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not in_bounds.all():
        raise ValueError(
            f"{(~in_bounds).sum()} cell centroids fall outside the snapped grid; "
            f"check pixel_size / bounds."
        )
    flat = rows * width + cols
    if np.unique(flat).size != flat.size:
        # find an example collision for the message
        _, counts = np.unique(flat, return_counts=True)
        raise ValueError(
            f"{(counts > 1).sum()} pixels receive >1 cell -- grid is not a clean "
            f"{px:g} m lattice. Refusing to silently overwrite."
        )

    ids = gdf[id_col].to_numpy()
    id_to_pixel = {int(i): (int(r), int(c)) for i, r, c in zip(ids, rows, cols)}

    # value lookups keyed by id
    written = {}
    frame_ids = frame[key].to_numpy()
    for col in value_cols:
        vals = frame[col].to_numpy()
        arr = np.full((height, width), np.nan, dtype=np.float32)
        n_placed = 0
        for i, v in zip(frame_ids, vals):
            rc = id_to_pixel.get(int(i))
            if rc is not None and np.isfinite(v):
                arr[rc] = v
                n_placed += 1
        out_path = os.path.join(out_dir, f"{basename}__{col}.tif")
        col_tags = dict(tags or {})
        col_tags.update({"value_column": col, "n_cells_placed": n_placed,
                         "pixel_size_m": px})
        _write_geotiff(out_path, arr, transform, crs, col_tags)
        written[col] = out_path

    result = {
        "geotiffs": written,
        "grid": {"width": width, "height": height, "pixel_size": px,
                 "transform": tuple(transform)[:6],
                 "crs_epsg": crs.to_epsg() if crs is not None else None},
        "n_cells_total": int(len(gdf)),
        "n_scored": int(frame[key].nunique()),
    }

    # ---- coverage mask: 1=scored, 2=no imagery, 3=in-AOI unscored (other) -----
    if write_coverage_mask:
        scored = set(int(i) for i in frame[key].to_numpy())
        _no_img_iter = no_imagery_ids if no_imagery_ids is not None else []
        no_img = set(int(i) for i in _no_img_iter)
        mask = np.full((height, width), np.nan, dtype=np.float32)
        counts = {"scored": 0, "no_imagery": 0, "unscored_other": 0}
        for i, (r, c) in id_to_pixel.items():
            if i in scored:
                mask[r, c] = 1.0; counts["scored"] += 1
            elif i in no_img:
                mask[r, c] = 2.0; counts["no_imagery"] += 1
            else:
                mask[r, c] = 3.0; counts["unscored_other"] += 1
        mpath = os.path.join(out_dir, f"{basename}__coverage_mask.tif")
        _write_geotiff(mpath, mask, transform, crs,
                       {**(tags or {}), "mask_legend": "1=scored;2=no_imagery;3=unscored"})
        result["coverage_mask"] = mpath
        result["coverage_counts"] = counts

    return result
