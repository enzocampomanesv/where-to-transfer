# Data

The statistics and figures need only the files in [`../results/`](../results/). Rerunning the experiments needs the raw data, which is too large for git and is archived on Zenodo:

> **Zenodo DOI: to be added**

## Contents of the Zenodo archive

| Folder | Content |
|---|---|
| `votes/db/{CITY}/{CITY}_voterAI.db` | Pairwise comparisons by residents of each city (SQLite): image pair IDs and the chosen image |
| `votes/{year}/{CITY}/S1_200m/` | Satellite image patches (200 m) shown in the comparisons, as GeoTIFF |
| `clustering/{CITY}/{CITY}_FN_200m.gpkg` | Patch grid per city |
| `trained_models/voter/` | The trained city models (`*_bradley_terry.ckpt`), one per city and per multi-city aggregate |
| `where_transfer/models/` | SAT isolation forests, scalers and PCA per source city (`*.joblib`) |
| `where_transfer/results/tier1/embeddings/` | Source-model embeddings of every patch, used for the coverage measures |
| `pool_features.zip` | Pool features for replaying the SAT sampler (`scripts/review_stats/sampling_sensitivity.py`) |

City codes: ACC Accra, TEM Tema, NBO Nairobi, KIS Kisumu, BEI Beira, CHI Chimoio, MUM Mumbai.

## Using it

Unpack the archive and point the scripts to it:

```bash
export WT_DATA_ROOT=/path/to/archive               # votes/, clustering/, trained_models/
export WT_ROOT=/path/to/archive/where_transfer     # models/, results/
```

Copy `pool_features.zip` to `results/` in this repository if you want to rerun `sampling_sensitivity.py`.
