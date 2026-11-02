"""
registry.py
===========

Single source of truth for resolving, from a (source, target) city pair:
  * the source model checkpoint path,
  * per-city year / d_src / gpkg / tile-root,

with explicit overrides everywhere it could silently go wrong. Convention is used
only where being wrong would *fail loudly* (path templates); anything whose error
would be invisible (which model, which year, aggregate leakage) is explicit.

MODEL NAMING (fixed for this transferability study)
---------------------------------------------------
    model_name = f"{src_code}_voter_communities_ensembleVoter"
    model_path = f"{root}/trained_models/voter/{model_name}_{d_src}_1_bradley_terry.ckpt"

  * voter is hardcoded to "communities" (the only vote set used here).
  * the "_1_" is a fixed filename token, not a variable.
  * ``src_code`` handles LEAVE-TARGET-OUT for aggregate sources to avoid leakage:
      - single-city source (e.g. ACC)                 -> "ACC"
      - aggregate whose members INCLUDE the target    -> "{AGG}_{TARGET}"  e.g. PRI on ACC -> "PRI_ACC"
      - aggregate whose members EXCLUDE the target    -> "{AGG}"           e.g. PRI on CHI -> "PRI"
    (CHI is not in PRI, so full PRI is safe; ACC is in PRI, so PRI_ACC must be used.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

VOTER = "communities"          # fixed for this study
DEFAULT_D_SRC = "S1_200m"
DEFAULT_YEAR = 2022            # ACC sample path uses 2022; override per-city as needed
MODEL_TOKEN = "1_bradley_terry"

SINGLE_CITY_CODES = ["ACC", "NBO", "BEI", "KIS", "CHI", "TEM", "MUM"]

# Aggregate source memberships (African aggregates only; MUM is cross-continental).
AGGREGATE_MEMBERS: Dict[str, list] = {
    "PRI": ["ACC", "NBO", "BEI"],
    "SEC": ["TEM", "KIS", "CHI"],
    "AFR": ["ACC", "NBO", "BEI", "TEM", "KIS", "CHI"],
}

# Per-city acquisition year. Fill in any city whose imagery year != DEFAULT_YEAR.
# Kept explicit because a wrong year silently loads the wrong tiles.
CITY_YEARS: Dict[str, int] = {
    # "MUM": 2023,   # <- example; set real values on your machine
}

DEFAULT_TILE_TEMPLATE = "{root}/votes/{year}/{city}/{d_src}/{city}_{d_src}_{id}.tif"
# gpkg lives under root_path at clustering/{CITY}/{CITY}_FN_200m.gpkg
DEFAULT_CLUSTERING_DIR = "clustering"
DEFAULT_GPKG_TEMPLATE = "{root}/{clustering_dir}/{city}/{city}_FN_200m.gpkg"
DEFAULT_MODEL_TEMPLATE = "{root}/trained_models/voter/{model_name}_{d_src}_{token}.ckpt"


# ===========================================================================
# Source-code resolution (leave-target-out for aggregates)
# ===========================================================================
def is_aggregate(source: str) -> bool:
    return source in AGGREGATE_MEMBERS


def effective_source_code(source: str, target: str) -> str:
    """Resolve the model's source code, applying leave-target-out for aggregates."""
    if is_aggregate(source):
        members = AGGREGATE_MEMBERS[source]
        if target in members:
            return f"{source}_{target}"     # leakage-avoiding leave-one-out variant
        return source                        # target not a member -> full aggregate is safe
    return source


def resolve_model_path(
    source: str,
    target: str,
    root_path: str,
    *,
    voter: str = VOTER,
    d_src: str = DEFAULT_D_SRC,
    template: str = DEFAULT_MODEL_TEMPLATE,
    token: str = MODEL_TOKEN,
) -> str:
    """Full checkpoint path for the source model to apply to the target."""
    src_code = effective_source_code(source, target)
    model_name = f"{src_code}_voter_{voter}_ensembleVoter"
    return template.format(root=root_path.rstrip("/"), model_name=model_name,
                           d_src=d_src, token=token)


# ===========================================================================
# City config
# ===========================================================================
@dataclass
class CityConfig:
    code: str
    root_path: str
    year: int
    d_src: str = DEFAULT_D_SRC
    gpkg_path: Optional[str] = None
    tile_template: str = DEFAULT_TILE_TEMPLATE
    extras: Dict = field(default_factory=dict)


def get_city_config(
    code: str,
    root_path: str,
    *,
    clustering_dir: str = DEFAULT_CLUSTERING_DIR,
    gpkg_path: Optional[str] = None,
    year: Optional[int] = None,
    d_src: str = DEFAULT_D_SRC,
    year_overrides: Optional[Dict[str, int]] = None,
    tile_template: str = DEFAULT_TILE_TEMPLATE,
    gpkg_template: str = DEFAULT_GPKG_TEMPLATE,
) -> CityConfig:
    """Assemble a city's config, resolving year and gpkg path.

    ``year`` (explicit) wins over ``year_overrides``/``CITY_YEARS`` wins over
    ``DEFAULT_YEAR``. ``gpkg_path`` (explicit) wins over the resolved
    ``{root}/{clustering_dir}/{city}/{city}_FN_200m.gpkg``.
    """
    yr = year
    if yr is None:
        merged = {**CITY_YEARS, **(year_overrides or {})}
        yr = merged.get(code, DEFAULT_YEAR)

    gpath = gpkg_path
    if gpath is None:
        gpath = gpkg_template.format(root=root_path.rstrip("/"),
                                     clustering_dir=clustering_dir.strip("/"), city=code)

    return CityConfig(code=code, root_path=root_path, year=yr, d_src=d_src,
                      gpkg_path=gpath, tile_template=tile_template)


# ===========================================================================
# Resolution + validation for a full run
# ===========================================================================
@dataclass
class ResolvedInputs:
    source: str
    target: str
    src_code: str
    model_path: str
    target_cfg: CityConfig
    checks: Dict[str, bool]


def resolve_inputs(
    source: str,
    target: str,
    root_path: str,
    *,
    clustering_dir: str = DEFAULT_CLUSTERING_DIR,
    target_gpkg_path: Optional[str] = None,
    d_src: str = DEFAULT_D_SRC,
    year: Optional[int] = None,
    year_overrides: Optional[Dict[str, int]] = None,
    require_exist: bool = True,
) -> ResolvedInputs:
    """Resolve model + target-city config for a run, optionally asserting existence.

    Only the TARGET gpkg is resolved (that is the city being mapped); the source
    contributes a model, not geometry. Raises FileNotFoundError (when
    ``require_exist``) with a precise message if the model checkpoint or target
    gpkg is missing.
    """
    model_path = resolve_model_path(source, target, root_path, d_src=d_src)
    tgt = get_city_config(target, root_path, clustering_dir=clustering_dir,
                          gpkg_path=target_gpkg_path, d_src=d_src, year=year,
                          year_overrides=year_overrides)

    checks = {
        "model_exists": os.path.exists(model_path),
        "gpkg_exists": os.path.exists(tgt.gpkg_path),
    }
    if require_exist:
        if not checks["model_exists"]:
            raise FileNotFoundError(
                f"Source model not found for {source}->{target}: {model_path}\n"
                f"(effective src_code={effective_source_code(source, target)})"
            )
        if not checks["gpkg_exists"]:
            raise FileNotFoundError(f"Target gpkg not found for {target}: {tgt.gpkg_path}")

    return ResolvedInputs(
        source=source, target=target,
        src_code=effective_source_code(source, target),
        model_path=model_path, target_cfg=tgt, checks=checks,
    )
