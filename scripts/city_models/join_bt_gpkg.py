"""
join_bt_scores_to_gpkg.py
=========================
Joins per-location BT scores (from the bt_scores mode runs) to their
respective city GeoPackages, producing one output GPKG per target city
with each stage as a separate column.

Column naming convention
------------------------
    bt_zs                   — zero-shot
    bt_r_{n}                — random sampling, n fine-tuning pairs
    bt_t1_{n}               — informed t1 sampling, n fine-tuning pairs

e.g. bt_r_1000, bt_t1_1000

Input
-----
    BT_SCORES_DIR/*.csv     — one CSV per stage per pair

Output
------
    OUTPUT_DIR/{city}_bt_progression.gpkg   — one per target city

Usage
-----
    python join_bt_scores_to_gpkg.py
"""

import os
import glob
import pandas as pd
import geopandas as gpd

# ── Config ────────────────────────────────────────────────────────────────

BT_SCORES_DIR = os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/bt_scores")
GPKG_TEMPLATE = os.path.join(os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"), "clustering/{city}/{city}_FN_200m.gpkg")
OUTPUT_DIR    = os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/bt_score_maps")

# Source-target pairs whose BT scores were saved
BT_SCORE_PAIRS = [
    ('ACC', 'CHI'),
    ('NBO', 'ACC'),
    ('NBO', 'BEI'),
    ('TEM', 'CHI'),
    ('CHI', 'NBO'),
    ('MUM', 'BEI'),
]

# Unique target cities (determines which GPKGs to produce)
TARGET_CITIES = sorted(set(tgt for _, tgt in BT_SCORE_PAIRS))

SAMPLE_SIZES = [100, 500, 1000, 5000]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Column naming ─────────────────────────────────────────────────────────

def col_name(source, sampling_tag, sample_size):
    """Build a short column name for a given stage."""
    if sampling_tag == 'zero_shot' or sample_size == 0:
        return f'bt_{source}_zs'
    if sampling_tag == 'random':
        return f'bt_{source}_r_{sample_size}'
    tag = sampling_tag.replace('informed_', '')   # 't1', 't20', ...
    return f'bt_{source}_{tag}_{sample_size}'


# ── Load all BT score CSVs ────────────────────────────────────────────────

def load_all_bt_scores():
    """
    Returns a dict:
        {target_city: DataFrame with columns [image_id, bt_zs, bt_r_100, ...]}
    """
    csv_paths = glob.glob(os.path.join(BT_SCORES_DIR, '*_bt_scores.csv'))
    if not csv_paths:
        raise FileNotFoundError(f'No BT score CSVs found in {BT_SCORES_DIR}')
    print(f'Found {len(csv_paths)} BT score CSVs')

    # Accumulate one wide DataFrame per target city
    city_dfs = {city: None for city in TARGET_CITIES}

    for path in sorted(csv_paths):
        df = pd.read_csv(path)

        # Sanity check
        required = {'source', 'target', 'sample_size', 'sampling', 'image_id', 'bt_score'}
        missing  = required - set(df.columns)
        if missing:
            print(f'  WARNING: {os.path.basename(path)} missing columns {missing} — skipping')
            continue

        source       = df['source'].iloc[0]
        target       = df['target'].iloc[0]
        if target not in TARGET_CITIES:
            continue

        sample_size  = int(df['sample_size'].iloc[0])
        sampling_tag = df['sampling'].iloc[0]
        col          = col_name(source, sampling_tag, sample_size)

        # Average across seeds if multiple rows per image_id
        pivot = (df.groupby('image_id')['bt_score']
                   .mean()
                   .reset_index()
                   .rename(columns={'bt_score': col}))

        print(f'  {os.path.basename(path):55s} → col={col:<18s} ({len(pivot)} locations)')

        if city_dfs[target] is None:
            city_dfs[target] = pivot
        else:
            city_dfs[target] = city_dfs[target].merge(pivot, on='image_id', how='outer')

    return city_dfs


# ── Join to GPKG and save ─────────────────────────────────────────────────

def join_and_save(target_city, scores_df):
    gpkg_path = GPKG_TEMPLATE.format(city=target_city)
    if not os.path.exists(gpkg_path):
        print(f'  WARNING: GPKG not found at {gpkg_path} — skipping {target_city}')
        return

    gdf = gpd.read_file(gpkg_path)
    print(f'\n[{target_city}]  {len(gdf)} features in GPKG')
    print(f'  Score columns to join: {[c for c in scores_df.columns if c != "image_id"]}')

    # Join on PARTIMAP_I (int) == image_id (int)
    gdf['PARTIMAP_I'] = gdf['PARTIMAP_I'].astype(int)
    scores_df['image_id'] = scores_df['image_id'].astype(int)

    merged = gdf.merge(
        scores_df.rename(columns={'image_id': 'PARTIMAP_I'}),
        on='PARTIMAP_I',
        how='left'
    )

    n_matched = merged[[c for c in scores_df.columns if c != 'image_id']].notna().any(axis=1).sum()
    print(f'  Matched: {n_matched}/{len(merged)} features have at least one BT score')

    # Sort columns: geometry last, BT score columns grouped
    def _sort_key(c):
        # c = bt_{source}_zs  |  bt_{source}_r_{n}  |  bt_{source}_{tag}_{n}
        parts = c.split('_')   # ['bt', src, 'zs'] or ['bt', src, 'r', n] etc.
        is_zs = parts[-1] == 'zs'
        is_r  = 'r' in parts
        n     = int(parts[-1]) if parts[-1].isdigit() else 0
        order = 0 if is_zs else (1 if is_r else 2)
        return (parts[1], order, n)   # group by source, then strategy, then n

    bt_cols  = sorted([c for c in merged.columns if c.startswith('bt_')],
                      key=_sort_key)
    meta_cols = [c for c in merged.columns if not c.startswith('bt_') and c != 'geometry']
    merged   = merged[meta_cols + bt_cols + ['geometry']]

    out_path = os.path.join(OUTPUT_DIR, f'{target_city}_bt_progression.gpkg')
    merged.to_file(out_path, driver='GPKG')
    print(f'  ✓ Saved → {out_path}')
    print(f'    BT columns: {bt_cols}')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('Joining BT scores to city GeoPackages')
    print('=' * 65)

    city_dfs = load_all_bt_scores()

    for city in TARGET_CITIES:
        if city_dfs[city] is None:
            print(f'\n[{city}]  No BT scores found — skipping')
            continue
        join_and_save(city, city_dfs[city])

    print('\n' + '=' * 65)
    print('Done.')
    print(f'Output GPKGs in: {OUTPUT_DIR}')
    print('=' * 65)


if __name__ == '__main__':
    main()