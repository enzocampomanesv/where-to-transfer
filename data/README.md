# Data

The statistics and figures need only the files in [`../results/`](../results/). Rerunning the experiments needs the raw data, which is too large for git and is archived on Zenodo under CC BY 4.0:

> Zenodo: [https://doi.org/10.5281/zenodo.23910257](https://doi.org/10.5281/zenodo.23910257)

| Archive | Content | Unpacks to |
|---|---|---|
| `votes.zip` | Residents' pairwise comparisons per city (SQLite and CSV) | `data_root/votes/db/`, `data_root/votes/csv/` |
| `patches.zip` | Every satellite image patch used in a comparison (GeoTIFF, 200 m) | `data_root/votes/` |
| `clustering.zip` | Patch grid per city | `data_root/clustering/` |
| `city_models.zip` | Trained city and aggregate models | `data_root/trained_models/voter/` |
| `sat_models.zip` | SAT isolation forests, scalers, PCA and score-feature models | `wt_root/models/` |
| `embeddings.zip` | Source-model embeddings for the coverage measures | `wt_root/results/tier1/embeddings/` |
| `pool_features.zip` | Pool features for `scripts/review_stats/sampling_sensitivity.py` | `wt_root/results/pool_features/` |
| `MANIFEST.csv` | Every file with size and SHA-256 | |

City codes: ACC Accra, TEM Tema, NBO Nairobi, KIS Kisumu, BEI Beira, CHI Chimoio, MUM Mumbai.

Unzip everything into one folder and point the scripts to it:

```bash
export WT_DATA_ROOT=/path/to/folder/data_root
export WT_ROOT=/path/to/folder/wt_root
```

To rerun `sampling_sensitivity.py`, copy `pool_features.zip` to `results/` in this repository.
