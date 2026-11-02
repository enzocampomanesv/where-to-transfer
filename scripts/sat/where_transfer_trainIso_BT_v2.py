import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize, StandardScaler
from utils.ACC_train_bestVoter_ensemble import *
import joblib

root_path = os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP")

city = sys.argv[2]
base_name = sys.argv[1]
d_src = "S1_200m"
tbl_name = f'{city}_{base_name}_votes'
se=1
euclid=0
n_epochs = 1000
n_width = 20
n_height = 20
n_channels = 2
n_batch = 64
BATCH_SIZE = 64

db_path = f'{root_path}/votes/db/{city}/{city}_voterAI.db'
base_model = f"{city}_voter_{base_name}"
model_name = f"{base_model}_ensembleVoter"
model_dir = os.path.join(os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP"), "trained_models/voter", "")+model_name+"_"+d_src+"_1_bradley_terry.ckpt"

def retrieve_db(db_path, sql):
    con = sqlite3.connect(db_path)
    return con, pd.read_sql(sql, con = con)
def import_votes(db_path, tbl_name, n_votes=-1):
    sql = f"select * from " + tbl_name
    con, all_data = retrieve_db(db_path, sql)
    if n_votes != -1:
        all_data = all_data.sample(n=n_votes)
    return all_data
def load_and_preprocess_image(fp, sub_bands, norm_data=False):
    # Load image using GDAL
    # print(fp,end='\r')
    img_dataset = gdal.Open(fp)
    img_array = img_dataset.ReadAsArray()
    if norm_data:
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    if sub_bands is not None:
        img_array = img_array[sub_bands,:,:]
    return np.transpose(img_array[:, :20, :20], (1, 2, 0))

# 1. Load the model and extract the encoder
full_model = tf.keras.models.load_model(model_dir)
nested_model = full_model.layers[2]
single_encoder = nested_model.get_layer('feature_extractor')

sub_bands, n_width, n_height, n_channels = get_sub_bands(d_src)
# data_frac = np.arange(0.1,1.1,0.1)
data_frac = [1]
norm_data = 0
# Data loading
all_data = import_votes(db_path, tbl_name)

print(f"PROCESSING {len(all_data)} VOTES")

print("\n" + "="*60)
print("EXTRACTING PAIRWISE EMBEDDINGS")
print("="*60)
 
train_pair_concat = []  # Concatenated: [features_L, features_R] (512D)
train_pair_diff = []    # Difference: features_L - features_R (256D)
 
print(f"Processing {len(all_data)} pairs in batches of {BATCH_SIZE}...")
 
for i in range(0, len(all_data), BATCH_SIZE):
    if i % (BATCH_SIZE * 10) == 0:
        print(f"  Processed {i}/{len(all_data)} pairs", end='\r')
    
    batch_df = all_data.iloc[i: i + BATCH_SIZE]
    
    # Load images
    batch_l = [load_and_preprocess_image(root_path + "/" + row.left, sub_bands) 
               for row in batch_df.itertuples()]
    batch_r = [load_and_preprocess_image(root_path + "/" + row.right, sub_bands) 
               for row in batch_df.itertuples()]
    
    # Extract features using single encoder
    features_l = single_encoder(np.array(batch_l)).numpy()  # (batch, 2, 2, 64)
    features_r = single_encoder(np.array(batch_r)).numpy()  # (batch, 2, 2, 64)
    
    # Flatten spatial dimensions
    features_l_flat = features_l.reshape(features_l.shape[0], -1)  # (batch, 256)
    features_r_flat = features_r.reshape(features_r.shape[0], -1)  # (batch, 256)
    
    # Create pairwise representations
    # 1. Concatenation (similar to previous joint representation)
    pair_concat = np.concatenate([features_l_flat, features_r_flat], axis=1)  # (batch, 512)
    train_pair_concat.extend(pair_concat)
    
    # 2. Difference (BT-aligned representation)
    pair_diff = features_l_flat - features_r_flat  # (batch, 256)
    train_pair_diff.extend(pair_diff)
 
train_pair_concat = np.array(train_pair_concat)  # (N_pairs, 512)
train_pair_diff = np.array(train_pair_diff)      # (N_pairs, 256)
 
print(f"\n✓ Extracted {len(train_pair_concat)} pairwise embeddings")
print(f"  - Concatenated shape: {train_pair_concat.shape}")
print(f"  - Difference shape: {train_pair_diff.shape}")

# ============================================================================
# Extract Single-Image Embeddings
# ============================================================================
 
print("\n" + "="*60)
print("EXTRACTING SINGLE-IMAGE EMBEDDINGS")
print("="*60)
 
unique_images = pd.concat([all_data['left'], all_data['right']]).unique()
print(f"Processing {len(unique_images)} unique images...")
 
train_single_embeddings = []
 
for i in range(0, len(unique_images), BATCH_SIZE):
    if i % (BATCH_SIZE * 10) == 0:
        print(f"  Processed {i}/{len(unique_images)} images", end='\r')
    
    batch_paths = unique_images[i: i + BATCH_SIZE]
    
    # Load images
    batch_imgs = [load_and_preprocess_image(root_path + "/" + path, sub_bands) 
                  for path in batch_paths]
    
    # Extract features
    features = single_encoder(np.array(batch_imgs)).numpy()  # (batch, 2, 2, 64)
    
    # Flatten
    features_flat = features.reshape(features.shape[0], -1)  # (batch, 256)
    train_single_embeddings.extend(features_flat)
 
train_single_embeddings = np.array(train_single_embeddings)  # (N_images, 256)
 
print(f"\n✓ Extracted {len(train_single_embeddings)} single-image embeddings")
print(f"  - Shape: {train_single_embeddings.shape}")

# ============================================================================
# PCA Configuration
# ============================================================================
 
# How many components to keep for each embedding type?
PCA_COMPONENTS = {
    'single': 64,        # 256D → 64D  (75% reduction)
    'pair_concat': 128,  # 512D → 128D (75% reduction)
    'pair_diff': 64      # 256D → 64D  (75% reduction)
}
 
# Alternative: Keep components that explain X% of variance
USE_VARIANCE_THRESHOLD = False
VARIANCE_THRESHOLD = 0.95  # Keep 95% of variance

print(f"Original embedding shapes:")
print(f"  Single:     {train_single_embeddings.shape}")
print(f"  Concat:     {train_pair_concat.shape}")
print(f"  Difference: {train_pair_diff.shape}")

# ============================================================================
# Apply PCA to Each Embedding Type
# ============================================================================
 
print("\n" + "="*60)
print("APPLYING PCA DIMENSIONALITY REDUCTION")
print("="*60)
 
# --- 1. Single-Image Embeddings ---
print("\n1. Single-image embeddings (256D → 64D)")
 
scaler_single = StandardScaler()
train_single_scaled = scaler_single.fit_transform(train_single_embeddings)
 
if USE_VARIANCE_THRESHOLD:
    pca_single = PCA(n_components=VARIANCE_THRESHOLD, svd_solver='full')
else:
    pca_single = PCA(n_components=PCA_COMPONENTS['single'])
 
train_single_pca = pca_single.fit_transform(train_single_scaled)
 
variance_explained = np.sum(pca_single.explained_variance_ratio_)
print(f"✓ Reduced to {train_single_pca.shape[1]}D")
print(f"  Variance explained: {variance_explained:.1%}")
 
# --- 2. Concatenated-Pair Embeddings ---
print("\n2. Concatenated-pair embeddings (512D → 128D)")
 
scaler_concat = StandardScaler()
train_concat_scaled = scaler_concat.fit_transform(train_pair_concat)
 
if USE_VARIANCE_THRESHOLD:
    pca_concat = PCA(n_components=VARIANCE_THRESHOLD, svd_solver='full')
else:
    pca_concat = PCA(n_components=PCA_COMPONENTS['pair_concat'])
 
train_concat_pca = pca_concat.fit_transform(train_concat_scaled)
 
variance_explained = np.sum(pca_concat.explained_variance_ratio_)
print(f"✓ Reduced to {train_concat_pca.shape[1]}D")
print(f"  Variance explained: {variance_explained:.1%}")
 
# --- 3. Difference Embeddings (BT-specific) ---
print("\n3. Difference embeddings (256D → 64D)")
 
scaler_diff = StandardScaler()
train_diff_scaled = scaler_diff.fit_transform(train_pair_diff)
 
if USE_VARIANCE_THRESHOLD:
    pca_diff = PCA(n_components=VARIANCE_THRESHOLD, svd_solver='full')
else:
    pca_diff = PCA(n_components=PCA_COMPONENTS['pair_diff'])
 
train_diff_pca = pca_diff.fit_transform(train_diff_scaled)
 
variance_explained = np.sum(pca_diff.explained_variance_ratio_)
print(f"✓ Reduced to {train_diff_pca.shape[1]}D")
print(f"  Variance explained: {variance_explained:.1%}")

# ============================================================================
# Train Isolation Forests on PCA-Reduced Embeddings
# ============================================================================
 
print("\n" + "="*60)
print("TRAINING ISOLATION FORESTS")
print("="*60)
 
output_dir = os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "models")
 
# --- 1. Single-Image Isolation Forest ---
print("\n1. Training single-image Isolation Forest...")
iso_single = IsolationForest(
    n_estimators=200, 
    contamination='auto', 
    random_state=42,
    n_jobs=-1
)
iso_single.fit(train_single_pca)  # Using PCA-reduced embeddings
 
# Save both the PCA and Isolation Forest
joblib.dump(scaler_single, f'{output_dir}/{city}_{base_name}_single_scaler.joblib')
joblib.dump(pca_single, f'{output_dir}/{city}_{base_name}_single_pca.joblib')
joblib.dump(iso_single, f'{output_dir}/{city}_{base_name}_single_iso_pca.joblib')
 
print(f"✓ Saved single-image models")
print(f"  Input:  256D → StandardScaler → PCA → {train_single_pca.shape[1]}D → IsolationForest")
 
# --- 2. Concatenated-Pair Isolation Forest ---
print("\n2. Training concatenated-pair Isolation Forest...")
iso_pair_concat = IsolationForest(
    n_estimators=200, 
    contamination='auto', 
    random_state=42,
    n_jobs=-1
)
iso_pair_concat.fit(train_concat_pca)
 
joblib.dump(scaler_concat, f'{output_dir}/{city}_{base_name}_pair_concat_scaler.joblib')
joblib.dump(pca_concat, f'{output_dir}/{city}_{base_name}_pair_concat_pca.joblib')
joblib.dump(iso_pair_concat, f'{output_dir}/{city}_{base_name}_pair_concat_iso_pca.joblib')
 
print(f"✓ Saved concatenated-pair models")
print(f"  Input:  512D → StandardScaler → PCA → {train_concat_pca.shape[1]}D → IsolationForest")
 
# --- 3. Difference Isolation Forest ---
print("\n3. Training difference Isolation Forest (BT-specific)...")
iso_pair_diff = IsolationForest(
    n_estimators=200, 
    contamination='auto', 
    random_state=42,
    n_jobs=-1
)
iso_pair_diff.fit(train_diff_pca)
 
joblib.dump(scaler_diff, f'{output_dir}/{city}_{base_name}_pair_diff_scaler.joblib')
joblib.dump(pca_diff, f'{output_dir}/{city}_{base_name}_pair_diff_pca.joblib')
joblib.dump(iso_pair_diff, f'{output_dir}/{city}_{base_name}_pair_diff_iso_pca.joblib')
 
print(f"✓ Saved difference models")
print(f"  Input:  256D → StandardScaler → PCA → {train_diff_pca.shape[1]}D → IsolationForest")

# ============================================================================
# Score-Feature ISO Forest
# ============================================================================
# Trains the three models needed by compute_score_feature_misalignment and
# the informed sampling pipeline in ACC_transferLearn_BT.py:
#   - MLP regressor:   bt_score ~ pca_single_features
#   - scaler_residual: StandardScaler fitted on training residuals
#   - iso_score_feature: IsolationForest on (pca_features, scaled_residual)
#
# A location is a score-feature outlier at inference time if its image is
# visually familiar (inlier in iso_single) but its BT rank deviates more
# from what the MLP predicts than any source location's rank does — meaning
# the appearance-to-rank mapping differs from this source city's distribution.
# ============================================================================
print("\n" + "="*60)
print("TRAINING SCORE-FEATURE ISO FOREST")
print("="*60)

from evalBT import estimate_bt_scores_from_predictions
from sklearn.neural_network import MLPRegressor

# Estimate BT scores from all source city annotations.
# Uses the same LogisticRegression BT MLE as the rest of the pipeline.
print(f"\nEstimating BT scores from {len(all_data)} source city annotations...")
true_bt_scores = estimate_bt_scores_from_predictions(all_data, all_data['y'].values)
print(f"  ✓ Scores estimated for {len(true_bt_scores)} locations")

# Build location → PCA feature mapping.
# unique_images[i] corresponds to train_single_pca[i] by construction above.
# Decode bytes paths so keys match those returned by estimate_bt_scores_from_predictions.
loc_to_pca = {}
for i, path in enumerate(unique_images):
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    loc_to_pca[str(path)] = train_single_pca[i]

sf_locs = [loc for loc in loc_to_pca if loc in true_bt_scores]
print(f"  ✓ {len(sf_locs)} locations matched between PCA features and BT scores")

if len(sf_locs) >= 10:
    from sklearn.model_selection import KFold

    sf_pca = np.array([loc_to_pca[loc] for loc in sf_locs])
    sf_bt  = np.array([true_bt_scores[loc] for loc in sf_locs])

    # Step 1: Compute OUT-OF-FOLD residuals via KFold cross-validation.
    #
    # Using in-sample residuals (fitting the MLP then evaluating on the same
    # data) produces near-zero residuals for all training points, so the ISO
    # decision scores cluster near zero and lose all discriminative power.
    # OOF residuals reflect what the MLP actually gets wrong on unseen data,
    # giving the ISO real variance to learn a meaningful boundary from.
    print("\n1. Computing out-of-fold residuals (5-fold CV)...")
    n_splits = min(5, len(sf_locs))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_residuals = np.zeros(len(sf_locs))

    for fold, (train_idx, val_idx) in enumerate(kf.split(sf_pca)):
        mlp_fold = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            alpha=0.01,
            max_iter=1000,
            random_state=42
        )
        mlp_fold.fit(sf_pca[train_idx], sf_bt[train_idx])
        oof_residuals[val_idx] = sf_bt[val_idx] - mlp_fold.predict(sf_pca[val_idx])

    print(f"  OOF residuals: mean={oof_residuals.mean():.3f}, "
          f"std={oof_residuals.std():.3f}, "
          f"range=[{oof_residuals.min():.3f}, {oof_residuals.max():.3f}]")

    # Step 2: Train the final MLP on ALL data — used at inference time.
    print("\n2. Fitting final MLP on all data (for inference)...")
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        alpha=0.01,
        max_iter=1000,
        random_state=42
    )
    mlp_regressor.fit(sf_pca, sf_bt)
    r2 = mlp_regressor.score(sf_pca, sf_bt)
    print(f"  ✓ MLP fitted | in-sample R² = {r2:.3f}")

    # Step 3: Scale OOF residuals to match PCA feature scale.
    scaler_residual = StandardScaler()
    oof_residuals_scaled = scaler_residual.fit_transform(oof_residuals.reshape(-1, 1))

    # Step 4: Train ISO on scaled OOF residuals (1D).
    # Using only the residual dimension means the ISO directly learns what
    # range of residuals is normal for the source city, without being
    # overwhelmed by the 64 PCA dimensions.
    print("\n3. Training score-feature Isolation Forest on OOF residuals (1D)...")
    iso_score_feature = IsolationForest(
        n_estimators=200,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso_score_feature.fit(np.abs(oof_residuals_scaled))

    joblib.dump(mlp_regressor,
                f'{output_dir}/{city}_{base_name}_score_feature_mlp.joblib')
    joblib.dump(scaler_residual,
                f'{output_dir}/{city}_{base_name}_score_feature_residual_scaler.joblib')
    joblib.dump(iso_score_feature,
                f'{output_dir}/{city}_{base_name}_score_feature_iso.joblib')

    print(f"✓ Saved score-feature models:")
    print(f"  {city}_{base_name}_score_feature_mlp.joblib")
    print(f"  {city}_{base_name}_score_feature_residual_scaler.joblib")
    print(f"  {city}_{base_name}_score_feature_iso.joblib")

else:
    print(f"  ⚠ Only {len(sf_locs)} matched locations — skipping (need ≥ 10)")

# ============================================================================
# Summary
# ============================================================================
 
print("\n" + "="*60)
print("TRAINING COMPLETE WITH PCA")
print("="*60)
 
summary_data = [
    ['Single-Image', f'256D → {train_single_pca.shape[1]}D', 
     f"{np.sum(pca_single.explained_variance_ratio_):.1%}",
     len(train_single_pca)],
    ['Concat-Pair', f'512D → {train_concat_pca.shape[1]}D', 
     f"{np.sum(pca_concat.explained_variance_ratio_):.1%}",
     len(train_concat_pca)],
    ['Difference', f'256D → {train_diff_pca.shape[1]}D', 
     f"{np.sum(pca_diff.explained_variance_ratio_):.1%}",
     len(train_diff_pca)]
]
 
print("\nPipeline for each embedding type:")
print("  Raw Features → StandardScaler → PCA → IsolationForest")
print("\nDimensionality Reduction:")
for name, dims, var, n_samples in summary_data:
    print(f"  {name:15} {dims:20} {var:>7} variance | {n_samples:,} samples")
 
print("\nSaved models for each type (9 files total for ISO forests + 3 for score-feature):")
print("  ISO forests — StandardScaler + PCA + IsolationForest (×3 types)")
print("  Score-feature — MLPRegressor + StandardScaler + IsolationForest")
print("="*60)