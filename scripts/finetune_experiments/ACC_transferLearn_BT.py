"""
Fine-Tuning Transfer Learning Script
Tests impact of fine-tuning source models with small target city samples
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # suppress retracing and other performance warnings
import numpy as np
import sqlite3
from osgeo import gdal
import joblib
import gc
from sklearn.model_selection import train_test_split
from BT_utils.aoaBT import add_bt_metrics_to_categories
from BT_utils.evalBT import evaluate_model_bt_recovery, estimate_bt_scores_from_predictions
from BT_utils.decompBT import compute_bt_score_decompositions

# ============================================================================
# TensorFlow Memory Configuration
# ============================================================================

# Prevent TensorFlow from allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s), enabled memory growth")
    except RuntimeError as e:
        print(f"GPU memory config error: {e}")

# Limit CPU threads to prevent resource exhaustion
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ============================================================================
# Memory Monitoring
# ============================================================================

def report_memory():
    """Report current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / 1024**3
        print(f"  [MEM: {mem_gb:.2f} GB]", end='')
        return mem_gb
    except ImportError:
        return None
# ============================================================================
# Configuration
# ============================================================================

root_path = os.environ.get("WT_DATA_ROOT", "/home/jovyan/private/slums/data/PARTIMAP")
iso_model_dir = os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer")
voter = 'communities'
d_src = "S1_200m"
# src_ct = sys.argv[1] # 'PRI','SEC','GHA','KEN','MOZ','AFR', 'NBO','BEI','TEM','KIS','CHI',
# Target list: cities only for now (pri-sec and country targets deferred).
_TGT_CITIES = ['ACC','NBO','BEI','TEM','KIS','CHI','MUM']
_TGT_FULL   = ['ACC','NBO','BEI','TEM','KIS','CHI','MUM',
                'GHA','KEN','MOZ','PRI','SEC']

# Constituent individual cities of every city and aggregate target.
# Used to check whether a target overlaps with a source's training set.
_MEMBERS = {
    'ACC': {'ACC'}, 'NBO': {'NBO'}, 'BEI': {'BEI'},
    'TEM': {'TEM'}, 'KIS': {'KIS'}, 'CHI': {'CHI'}, 'MUM': {'MUM'},
    'GHA': {'ACC', 'TEM'},
    'KEN': {'NBO', 'KIS'},
    'MOZ': {'BEI', 'CHI'},
    'PRI': {'ACC', 'NBO', 'BEI'},
    'SEC': {'TEM', 'KIS', 'CHI'},
    'AFR': {'ACC', 'NBO', 'BEI', 'TEM', 'KIS', 'CHI'},
}

# Individual cities each source model was trained on.
_SRC_TRAIN = {
    'ACC': {'ACC'}, 'NBO': {'NBO'}, 'BEI': {'BEI'},
    'TEM': {'TEM'}, 'KIS': {'KIS'}, 'CHI': {'CHI'}, 'MUM': {'MUM'},
    'PRI':     {'ACC', 'NBO', 'BEI'},
    'SEC':     {'TEM', 'KIS', 'CHI'},
    'AFR':     {'ACC', 'NBO', 'BEI', 'TEM', 'KIS', 'CHI'},
    'PRI_ACC': {'NBO', 'BEI'},
    'PRI_NBO': {'ACC', 'BEI'},
    'PRI_BEI': {'ACC', 'NBO'},
    'SEC_TEM': {'KIS', 'CHI'},
    'SEC_KIS': {'TEM', 'CHI'},
    'SEC_CHI': {'TEM', 'KIS'},
    'AFR_ACC': {'NBO', 'BEI', 'TEM', 'KIS', 'CHI'},
    'AFR_NBO': {'ACC', 'BEI', 'TEM', 'KIS', 'CHI'},
    'AFR_BEI': {'ACC', 'NBO', 'TEM', 'KIS', 'CHI'},
    'AFR_TEM': {'ACC', 'NBO', 'BEI', 'KIS', 'CHI'},
    'AFR_KIS': {'ACC', 'NBO', 'BEI', 'TEM', 'CHI'},
    'AFR_CHI': {'ACC', 'NBO', 'BEI', 'TEM', 'KIS'},
}
# Cities to test
src_choice = int(sys.argv[1]) # 0=GHA, 1=KEN, 2=MOZ, 3=countries, 4=pri-sec, 5=OTH
if src_choice == 0:
    exp = "GHA"
    src_cities = ['ACC','TEM']
    tgt_cities = _TGT_CITIES
elif src_choice == 1:
    exp = "KEN"
    src_cities = ['NBO','KIS']
    tgt_cities = _TGT_CITIES
elif src_choice == 2:
    exp = "MOZ"
    src_cities = ['BEI','CHI']
    tgt_cities = _TGT_CITIES
elif src_choice == 3:
    exp = "OTH"
    src_cities = ['MUM']
    tgt_cities = _TGT_CITIES
elif src_choice == 4:
    exp = "AGG"
    src_cities = ['PRI', 'SEC', 'AFR']
    tgt_cities = _TGT_FULL
elif src_choice == 5:
    exp = "PRI"
    src_cities = ['PRI_ACC', 'PRI_NBO', 'PRI_BEI']
    tgt_cities = _TGT_FULL
elif src_choice == 6:
    exp = "SEC"
    src_cities = ['SEC_TEM', 'SEC_KIS', 'SEC_CHI']
    tgt_cities = _TGT_FULL
elif src_choice == 7:
    exp = "AFR"
    src_cities = ['AFR_ACC', 'AFR_NBO', 'AFR_BEI']
    tgt_cities = _TGT_FULL
elif src_choice == 8:
    exp = "AFR"
    src_cities = ['AFR_TEM', 'AFR_KIS', 'AFR_CHI']
    tgt_cities = _TGT_FULL


# ── Sampling strategy ────────────────────────────────────────────────────────
# sys.argv[2] accepts either:
#
#   'random'   — uniform random draw; does not use pool categorisation.
#                Kept as a named keyword because it is qualitatively different
#                from any temperature value (no ISO inference at sampling time).
#
#   'informed' — original hard-fill priority queue (Structural → SF →
#                Cultural → Relational → Consensus).  Kept as a named keyword
#                for backward compatibility with existing result files.
#
#   <float>    — temperature T ≥ 0 for the unified informed sampler (fixed-
#                weight), which uses the revised priority order and a softmax-
#                weighted draw:
#
#                  T = 0          near-uniform draw
#                  0 < T < HARD   soft mix; T=1 is the recommended operating
#                                 point (probabilities equal base weights)
#                  T ≥ HARD       deterministic hard fill in priority order
#
#   <float>a   — same as <float> but with adaptive weighting: each category's
#                base weight is additionally scaled by its fraction in the
#                train pool, so the draw responds to the actual shift profile
#                of the specific source→target pair. Append 'a' to the float:
#                e.g. '1a' for adaptive T=1, '20a' for adaptive hard fill.
#
# Examples:
#   python ACC_transferLearn_BT.py 0 random    # baseline
#   python ACC_transferLearn_BT.py 0 informed  # original order (legacy)
#   python ACC_transferLearn_BT.py 0 0         # near-uniform informed
#   python ACC_transferLearn_BT.py 0 1         # soft mix, T=1 (recommended)
#   python ACC_transferLearn_BT.py 0 20        # hard fill, revised order
#   python ACC_transferLearn_BT.py 0 1a        # adaptive soft mix, T=1
#   python ACC_transferLearn_BT.py 0 20a       # adaptive hard fill

_raw_arg = sys.argv[2] if len(sys.argv) > 2 else 'random'

if _raw_arg in ('random', 'informed'):
    sampling_strategy  = _raw_arg
    informed_temperature = None
    informed_adaptive    = False
else:
    # Strip optional trailing 'a' for adaptive mode
    informed_adaptive = _raw_arg.endswith('a')
    _float_str = _raw_arg[:-1] if informed_adaptive else _raw_arg
    try:
        informed_temperature = float(_float_str)
        assert informed_temperature >= 0, "Temperature must be ≥ 0"
        sampling_strategy = 'informed_t'
    except (ValueError, AssertionError) as _e:
        raise ValueError(
            f"sys.argv[2] must be 'random', 'informed', a float ≥ 0, or a "
            f"float ≥ 0 followed by 'a' for adaptive mode (got '{_raw_arg}'): {_e}"
        ) from _e

# Threshold above which the probabilistic draw is replaced by deterministic
# hard fill (avoids numerical edge cases at extreme temperatures).
HARD_FILL_TEMPERATURE = 15.0

# ── Revised priority order and base rank weights ──────────────────────────────
# Used by both the fixed-weight and adaptive informed samplers.
#
# INFORMED_PRIORITY_ORDER defines which categories are preferred (most → least
# informative). The order is grounded in the offline correlation analysis
# between each category's sample proportion and finetuning improvement:
#   SF misalignment is the only category with a consistently *positive*
#   correlation with improvement (Spearman ρ ≈ +0.36 across all experiment
#   levels and both accuracy and Spearman metrics). Structural Ignorance has
#   the strongest *negative* correlation (ρ ≈ -0.33).
#
# INFORMED_BASE_WEIGHTS assigns a base importance score to each category.
# The specific values (7:1:1:0.7:0.3) are chosen to reflect the relative
# magnitudes suggested by the correlation analysis:
#
#   SF_Misaligned        → 7.0  strongly positive ρ (+0.36) — dominant share
#   Cultural Disagreement→ 1.0  mild negative ρ (-0.13)    — equal moderate share
#   Relational Ignorance → 1.0  mild negative ρ (-0.11)    — equal moderate share
#   Consensus            → 0.7  mild negative ρ (-0.11), but kept slightly above
#                                Structural because it stabilises training and
#                                does not actively resist improvement
#   Structural Ignorance → 0.3  strong negative ρ (-0.33) — small but non-zero
#                                floor so the budget can still be filled when the
#                                pool is dominated by structural pairs
#
# At T=1 (recommended operating point), probabilities equal the normalised
# base weights directly: ~70% SF, ~10% Cultural, ~10% Relational, ~7%
# Consensus, ~3% Structural.
#
# These weights are intentionally simple and dataset-agnostic. Adjust freely
# for new datasets or contexts where the correlation structure may differ.

INFORMED_PRIORITY_ORDER = [
    'SF_Misaligned',
    'Cultural Disagreement',
    'Relational Ignorance',
    'Consensus',
    'Structural Ignorance',
]

INFORMED_BASE_WEIGHTS = {
    'SF_Misaligned':         7.0,
    'Cultural Disagreement': 1.0,
    'Relational Ignorance':  1.0,
    'Consensus':             0.7,
    'Structural Ignorance':  0.3,
}

# ── Print active configuration ────────────────────────────────────────────────
if sampling_strategy == 'random':
    sampling_tag = 'random'
    print(f"Sampling strategy: random")
elif sampling_strategy == 'informed':
    sampling_tag = 'informed'
    print(f"Sampling strategy: informed (original order, hard fill)")
else:
    T   = informed_temperature
    _a  = 'a' if informed_adaptive else ''
    sampling_tag = f'informed_t{T:g}{_a}'
    mode = ('hard fill (revised order)' if T >= HARD_FILL_TEMPERATURE
            else f'soft mix (T={T:g})' if T > 0
            else 'near-uniform draw (T=0)')
    adapt_label = ' — adaptive weights' if informed_adaptive else ''
    print(f"Sampling strategy: {sampling_tag}  [{mode}{adapt_label}]")

    if T >= HARD_FILL_TEMPERATURE:
        print(f"  Priority order: {' → '.join(INFORMED_PRIORITY_ORDER)}")
        if informed_adaptive:
            print(f"  Adaptive: hard-fill order unchanged; "
                  f"pool fractions govern depletion speed naturally.")
    else:
        # Show fixed-weight effective probabilities (T=1: probs = base weights)
        _bw = np.array([INFORMED_BASE_WEIGHTS[c] for c in INFORMED_PRIORITY_ORDER],
                       dtype=float)
        if T == 0:
            _eff = np.ones(len(_bw)) / len(_bw)
        else:
            _log = np.log(_bw) * T
            _log -= _log.max()
            _eff = np.exp(_log) / np.exp(_log).sum()
        print("  Fixed base probabilities: " + ", ".join(
            f"{c.split()[0]}={p:.3f}"
            for c, p in zip(INFORMED_PRIORITY_ORDER, _eff)
        ))
        if informed_adaptive:
            print(f"  Adaptive: above probabilities further scaled by pool "
                  f"category fractions at runtime (pair-specific).")


# Fine-tuning sample sizes to test
SAMPLE_SIZES = [100, 500, 1000, 5000]

# ── Training strategy (sys.argv[3]) ─────────────────────────────────────────
# Controls how the model is finetuned:
#
#   'full'      — all layers trainable, single Adam optimizer at BASE_LR.
#                 Simple baseline; use to compare against layerwise.
#
#   'layerwise' — all layers trainable except BatchNormalization (frozen).
#                 Three Adam optimizers with decayed LRs per layer group:
#                   Block 1 (conv/dense):  BASE_LR × α²        (≈ 1% of head LR)
#                   Block 2 (conv/dense):  BASE_LR × α         (≈ 10% of head LR)
#                   Block 3 (conv/dense):  BASE_LR × √α        (≈ 32% of head LR)
#                   strength_output head:  HEAD_LR_MULTIPLIER × BASE_LR
#                 where α = LAYERWISE_DECAY = 0.1
#                 Implemented via a custom train_step with GradientTape so
#                 model.fit / EarlyStopping / callbacks work unchanged.
#
# Usage:
#   python ACC_transferLearn_BT.py 0 1 full      1e-4
#   python ACC_transferLearn_BT.py 0 1 layerwise 1e-4

_raw_train_arg = sys.argv[3] if len(sys.argv) > 3 else 'layerwise'
assert _raw_train_arg in ('full', 'layerwise'), \
    f"sys.argv[3] must be 'full' or 'layerwise', got '{_raw_train_arg}'"
training_strategy = _raw_train_arg

# ── Base learning rate (sys.argv[4]) ────────────────────────────────────────
# Controls the LR of the head (strength_output) in layerwise mode, and the
# single optimizer LR in full mode. Accepts any float > 0.
# Recommended values: 1e-4 (more aggressive), 1e-5 (conservative original).
_raw_lr_arg = sys.argv[4] if len(sys.argv) > 4 else '1e-4'
try:
    BASE_LR = float(_raw_lr_arg)
    assert BASE_LR > 0, "BASE_LR must be > 0"
except (ValueError, AssertionError) as _e:
    raise ValueError(
        f"sys.argv[4] must be a float > 0 (got '{_raw_lr_arg}'): {_e}"
    ) from _e

# ── Layerwise LR constants ───────────────────────────────────────────────────
# α (LAYERWISE_DECAY) controls the LR decay per block from head → early layers.
# At α=0.1: Block1=BASE_LR×0.01, Block2=BASE_LR×0.1, Block3=BASE_LR×0.316.
# HEAD_LR_MULTIPLIER scales BASE_LR upward for the strength_output layer, which
# needs more room to shift the preference mapping toward the target city.
LAYERWISE_DECAY      = 0.1
HEAD_LR_MULTIPLIER   = 5.0   # strength_output LR = BASE_LR × 5

# Derived LRs (printed at startup for verification)
_LR_HEAD   = BASE_LR * HEAD_LR_MULTIPLIER
_LR_BLOCK3 = BASE_LR * (LAYERWISE_DECAY ** 0.5)
_LR_BLOCK2 = BASE_LR * LAYERWISE_DECAY
_LR_BLOCK1 = BASE_LR * (LAYERWISE_DECAY ** 2)

# Fine-tuning strategies dict (consumed by run_finetuning_iteration)
STRATEGIES = {
    'full': {
        'name':         'Full Fine-Tuning',
        'learning_rate': BASE_LR,
        'description':  f'All layers, single Adam LR={BASE_LR:.2e}',
    },
    'layerwise': {
        'name':         'Layerwise LR Fine-Tuning',
        'learning_rate': BASE_LR,   # head LR; block LRs derived at runtime
        'description':  (
            f'BN frozen; Block1={_LR_BLOCK1:.1e} Block2={_LR_BLOCK2:.1e} '
            f'Block3={_LR_BLOCK3:.1e} Head={_LR_HEAD:.1e}'
        ),
    },
}

print(f"Training strategy : {training_strategy}")
print(f"Base LR           : {BASE_LR:.2e}")
if training_strategy == 'layerwise':
    print(f"  Block1={_LR_BLOCK1:.1e}  Block2={_LR_BLOCK2:.1e}  "
          f"Block3={_LR_BLOCK3:.1e}  Head={_LR_HEAD:.1e}")

# Fine-tuning hyperparameters
FINE_TUNE_EPOCHS = 50    # Few epochs to avoid overfitting
BATCH_SIZE = 32
PATIENCE = 10             # Early stopping patience

# Random seeds for robustness (multiple runs)
RANDOM_SEEDS = [42, 43, 44]  # 3 random splits per sample size

# Output
# ── Run tag — unique identifier for this combination of settings ─────────────
# Combines sampling strategy, training strategy, and base LR into a single
# string used for the output folder and CSV filenames. Double underscore
# separates the sampling tag (which may contain single underscores) from the
# training tag for clean parsing downstream.
#   e.g. 'informed_t1__layerwise_lr1e-04'
#        'random__full_lr1e-04'
run_tag = f"{sampling_tag}__{training_strategy}_lr{BASE_LR:.0e}"
print(f"Run tag           : {run_tag}")

RESULTS_DIR = os.path.join(os.environ.get("WT_ROOT", "/home/jovyan/private/where_transfer"), "results/transferLearning", run_tag)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# Helper Functions
# ============================================================================
def get_sub_bands(d_src):
    n_width = 10
    n_height = 10
    if d_src == "RGBN":
        sub_bands = [0, 1, 2, 6]
    elif d_src == "RGB":
        sub_bands = [0, 1, 2]
    elif d_src == "RGN":
        sub_bands = [0, 1, 6]
    elif d_src == "all":
        sub_bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    elif d_src == "S1":
        sub_bands = [0, 1]
    elif d_src == "wv3":
        sub_bands = [0, 1, 2, 3]
        n_width = 50
        n_height = 50
    elif d_src == "S1_200m":
        sub_bands = [0, 1]
        n_width = 20
        n_height = 20
    n_channels = len(sub_bands)
    return sub_bands, n_width, n_height, n_channels

def load_and_preprocess_image(fp, sub_bands, norm_data=False):
    img_dataset = gdal.Open(fp)
    img_array = img_dataset.ReadAsArray()
    img_dataset = None
    if norm_data:
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    if sub_bands is not None:
        img_array = img_array[sub_bands,:,:]
    return np.transpose(img_array[:, :20, :20], (1, 2, 0))

def retrieve_db(db_path, sql):
    con = sqlite3.connect(db_path)
    df = pd.read_sql(sql, con=con)
    con.close()
    return df

def _try_load_joblib(path):
    """Load a joblib file if it exists, return None if not found yet."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None

def _save_results(results, output_path):
    """
    Save the current results list to CSV with consistent column ordering.
    Called after every source-target pair so progress is never lost.
    """
    if not results:
        return
    results_df = pd.DataFrame(results)

    core_cols = [
        'source', 'target', 'sample_size', 'strategy', 'sampling', 'random_seed',
        'duration_seconds', 'accuracy', 'improvement'
    ]
    category_cols = [
        'consensus', 'cultural_disagreement', 'relational_ignorance',
        'structural_ignorance', 'relational_ratio'
    ]
    compat_cols = [
        'pool_structural_pct', 'pool_sf_pct', 'pool_relational_pct',
        'pool_learnable_pct',
        'pair_compatibility', 'ft_benefit_ceiling_est', 'ft_spearman_ceiling_est',
    ]
    remaining_cols = [col for col in results_df.columns
                      if col not in core_cols + category_cols + compat_cols]
    existing_cols = [col for col in core_cols + category_cols + compat_cols + remaining_cols
                     if col in results_df.columns]
    results_df[existing_cols].to_csv(output_path, index=False)
    print(f"  ✓ Checkpoint saved ({len(results_df)} rows) → {output_path}")

def _informed_sample(train_pool, aoa_labels, sf_pair_flags, sample_size, seed):
    """
    Sample from train_pool prioritising pairs by AoA category and score-feature
    misalignment.

    Priority order (most → least informative):
        1. Structural Ignorance  — OOD images; expands visual coverage
        2. SF_Misaligned         — familiar image but unexpected rank;
                                   teaches the target city's appearance-to-rank
                                   mapping (only available if SF models loaded)
        3. Cultural Disagreement — familiar image, model wrong, expected rank
        4. Relational Ignorance  — familiar images, foreign comparison strategy
        5. Consensus             — model already handles these; least informative

    Pairs that are BOTH Structural Ignorance AND SF-misaligned go into tier 1
    (SF analysis is uninformative for OOD images, so the AoA label dominates).

    Selection within each tier is random, controlled by seed.
    Budget spills into the next tier when a tier is exhausted.
    """
    from collections import Counter
    rng = np.random.RandomState(seed)

    priority_order = [
        'Structural Ignorance',
        'SF_Misaligned',
        'Cultural Disagreement',
        'Relational Ignorance',
        'Consensus',
    ]

    # Assign effective tier to each pair
    effective = []
    for aoa, sf in zip(aoa_labels, sf_pair_flags):
        if aoa == 'Structural Ignorance':
            effective.append('Structural Ignorance')   # AoA dominates
        elif sf:
            effective.append('SF_Misaligned')
        else:
            effective.append(aoa)

    # Group indices by tier and shuffle within each
    tier_indices = {t: [] for t in priority_order}
    for i, t in enumerate(effective):
        tier_indices[t].append(i)
    for t in priority_order:
        rng.shuffle(tier_indices[t])

    # Fill budget in priority order
    selected = []
    for t in priority_order:
        needed = sample_size - len(selected)
        if needed <= 0:
            break
        selected.extend(tier_indices[t][:needed])

    # Log and return breakdown
    breakdown = Counter(effective[i] for i in selected)
    print(f"      [Informed] " + ", ".join(
        f"{t.split()[0]}: {breakdown.get(t, 0)}"
        for t in priority_order
    ))

    sample_breakdown = {
        'sample_structural':    breakdown.get('Structural Ignorance', 0),
        'sample_sf_misaligned': breakdown.get('SF_Misaligned', 0),
        'sample_cultural':      breakdown.get('Cultural Disagreement', 0),
        'sample_relational':    breakdown.get('Relational Ignorance', 0),
        'sample_consensus':     breakdown.get('Consensus', 0),
    }

    return train_pool.iloc[selected].reset_index(drop=True), sample_breakdown

def _informed_sample_unified(train_pool, aoa_labels, sf_pair_flags,
                              sample_size, seed, temperature):
    """
    Unified informed sampler — revised priority order, temperature-controlled
    draw (probabilistic soft mix or deterministic hard fill).

    Priority order (fixed, grounded in correlation analysis):
        1. SF_Misaligned         — familiar image, unexpected rank;
                                   teaches target appearance-to-rank mapping
        2. Cultural Disagreement — familiar image, model wrong, expected rank
        3. Relational Ignorance  — familiar images, foreign comparison logic
        4. Consensus             — model already handles; useful as filler
        5. Structural Ignorance  — OOD images; resists improvement at small n;
                                   deprioritised so it does not crowd out more
                                   informative pairs at limited budgets

    Temperature behaviour
    ─────────────────────
    T = 0
        Near-uniform draw: every pair has roughly equal probability regardless
        of its category. Useful as an ablation or when no prior knowledge about
        category importance is available.

    0 < T < HARD_FILL_TEMPERATURE
        Soft mix: sampling probabilities are proportional to
            p_i ∝ exp( log(base_weight[category_i]) × T )
        Higher T concentrates more mass on SF_Misaligned; lower T flattens the
        distribution toward uniform. The draw is without replacement so the
        actual sample composition reflects probabilities, not hard cuts.

    T ≥ HARD_FILL_TEMPERATURE  (default threshold: 15)
        Hard fill: budget is filled in strict priority order (deterministic),
        spilling into the next tier only when the current tier is exhausted.
        This is the behaviour of the hard-fill revised-order strategy and avoids
        floating-point edge cases at extreme temperatures.

    Parameters
    ----------
    temperature : float ≥ 0
        Controls sharpness. Passed from informed_temperature (module-level
        constant set from sys.argv[2]).

    Design notes
    ────────────
    • Effective tier assignment is identical across all informed samplers:
      AoA label dominates for Structural Ignorance; SF flag overrides for
      image-inliers with unexpected BT rank (SF_Misaligned tier).
    • INFORMED_PRIORITY_ORDER and INFORMED_BASE_WEIGHTS are module-level
      constants so they can be tuned without touching this function.
    • The logged breakdown reflects the *actual* drawn sample, making it
      directly comparable across temperatures and with the random baseline.
    """
    from collections import Counter
    rng = np.random.RandomState(seed)

    # ── Assign effective tier to each pair ───────────────────────────────────
    effective = []
    for aoa, sf in zip(aoa_labels, sf_pair_flags):
        if aoa == 'Structural Ignorance':
            effective.append('Structural Ignorance')   # AoA dominates for OOD
        elif sf:
            effective.append('SF_Misaligned')
        else:
            effective.append(aoa)

    # ── Select pairs ─────────────────────────────────────────────────────────
    if temperature >= HARD_FILL_TEMPERATURE:
        # Deterministic hard fill in priority order
        tier_indices = {t: [] for t in INFORMED_PRIORITY_ORDER}
        for i, t in enumerate(effective):
            tier_indices[t].append(i)
        for t in INFORMED_PRIORITY_ORDER:
            rng.shuffle(tier_indices[t])   # random within each tier
        selected = []
        for t in INFORMED_PRIORITY_ORDER:
            needed = sample_size - len(selected)
            if needed <= 0:
                break
            selected.extend(tier_indices[t][:needed])

    else:
        # Softmax-weighted draw without replacement
        base_w = np.array(
            [INFORMED_BASE_WEIGHTS.get(t, 1e-6) for t in effective], dtype=float
        )
        if temperature == 0:
            prob = np.ones(len(effective)) / len(effective)
        else:
            log_w = np.log(np.maximum(base_w, 1e-12)) * temperature
            log_w -= log_w.max()
            prob = np.exp(log_w)

        prob_sum = prob.sum()
        prob = prob / prob_sum if prob_sum > 0 else np.ones(len(effective)) / len(effective)

        actual_size = min(sample_size, len(train_pool))
        selected = rng.choice(len(train_pool), size=actual_size, replace=False, p=prob).tolist()

    # ── Log and package breakdown ─────────────────────────────────────────────
    breakdown = Counter(effective[i] for i in selected)
    mode_str  = ('hard-fill' if temperature >= HARD_FILL_TEMPERATURE
                 else f'T={temperature:g}')
    print(f"      [Informed {mode_str}] " + ", ".join(
        f"{t.split()[0]}: {breakdown.get(t, 0)}"
        for t in INFORMED_PRIORITY_ORDER
    ))

    sample_breakdown = {
        'sample_structural':    breakdown.get('Structural Ignorance', 0),
        'sample_sf_misaligned': breakdown.get('SF_Misaligned', 0),
        'sample_cultural':      breakdown.get('Cultural Disagreement', 0),
        'sample_relational':    breakdown.get('Relational Ignorance', 0),
        'sample_consensus':     breakdown.get('Consensus', 0),
    }
    return train_pool.iloc[selected].reset_index(drop=True), sample_breakdown


def _informed_sample_adaptive(train_pool, aoa_labels, sf_pair_flags,
                               sample_size, seed, temperature):
    """
    Adaptive informed sampler — revised priority order, temperature-controlled
    draw, with per-pair weight scaling by pool category fractions.

    Identical to _informed_sample_unified in every respect except how the
    probability vector is built for the soft-mix path (T < HARD_FILL_TEMPERATURE):

        Fixed-weight path (_informed_sample_unified):
            prob_i ∝ exp( log(base_weight[category_i]) × T )

        Adaptive path (this function):
            prob_i ∝ exp( log(base_weight[category_i] × pool_frac_i) × T )

    where pool_frac_i = count(category_i) / len(pool).

    Motivation
    ──────────
    The fixed-weight sampler applies the same priority regardless of how much
    of each shift type is actually present in the pool. A pair whose pool is
    60% SF-misaligned and one whose pool is only 5% SF-misaligned receive the
    same starting weights, with the difference entering only implicitly through
    without-replacement depletion.

    The adaptive sampler makes the dependency explicit: the effective weight of
    a category is proportional to both how informative it is (base weight) and
    how available it is (pool fraction). A category that is rare in the pool
    contributes less to the draw even if it is high-priority, which prevents
    over-sampling a thin slice and broadens the mix naturally.

    Trade-off vs fixed-weight
    ─────────────────────────
    Adaptive sampling is pair-specific — two source→target pairs with different
    shift profiles receive different sample compositions even at the same T.
    This makes it a better "best possible" sampler for each individual pair but
    confounds direct cross-pair comparisons of finetuning outcomes (differences
    in improvement may reflect sampling differences rather than transferability
    differences). Use fixed-weight for controlled comparisons; use adaptive as
    an upper-bound or "oracle" reference.

    The hard-fill path (T ≥ HARD_FILL_TEMPERATURE) is identical to
    _informed_sample_unified — deterministic fill in INFORMED_PRIORITY_ORDER
    already adapts implicitly through depletion, so no explicit pool-fraction
    scaling is needed or meaningful.

    Parameters
    ----------
    temperature : float ≥ 0
        Same semantics as _informed_sample_unified.
    """
    from collections import Counter
    rng = np.random.RandomState(seed)

    # ── Assign effective tier ────────────────────────────────────────────────
    effective = []
    for aoa, sf in zip(aoa_labels, sf_pair_flags):
        if aoa == 'Structural Ignorance':
            effective.append('Structural Ignorance')
        elif sf:
            effective.append('SF_Misaligned')
        else:
            effective.append(aoa)

    # ── Select pairs ─────────────────────────────────────────────────────────
    if temperature >= HARD_FILL_TEMPERATURE:
        # Hard fill — identical to fixed-weight path; adaptive scaling is
        # implicit through category depletion.
        tier_indices = {t: [] for t in INFORMED_PRIORITY_ORDER}
        for i, t in enumerate(effective):
            tier_indices[t].append(i)
        for t in INFORMED_PRIORITY_ORDER:
            rng.shuffle(tier_indices[t])
        selected = []
        for t in INFORMED_PRIORITY_ORDER:
            needed = sample_size - len(selected)
            if needed <= 0:
                break
            selected.extend(tier_indices[t][:needed])

    else:
        # Adaptive softmax-weighted draw without replacement.
        # Compute pool fractions for each category.
        pool_counts = Counter(effective)
        pool_total  = len(effective) or 1
        pool_fracs  = {t: pool_counts.get(t, 0) / pool_total
                       for t in INFORMED_PRIORITY_ORDER}

        # Effective weight = base_weight × pool_fraction (floored to avoid zeros)
        eff_w = np.array(
            [max(INFORMED_BASE_WEIGHTS.get(t, 1e-6) * pool_fracs.get(t, 0), 1e-9)
             for t in effective],
            dtype=float
        )

        if temperature == 0:
            prob = np.ones(len(effective)) / len(effective)
        else:
            log_w = np.log(eff_w) * temperature
            log_w -= log_w.max()
            prob = np.exp(log_w)

        prob_sum = prob.sum()
        prob = prob / prob_sum if prob_sum > 0 else np.ones(len(effective)) / len(effective)

        actual_size = min(sample_size, len(train_pool))
        selected = rng.choice(len(train_pool), size=actual_size, replace=False, p=prob).tolist()

    # ── Log and package breakdown ─────────────────────────────────────────────
    breakdown = Counter(effective[i] for i in selected)
    mode_str  = ('hard-fill' if temperature >= HARD_FILL_TEMPERATURE
                 else f'T={temperature:g}')
    print(f"      [Informed adaptive {mode_str}] " + ", ".join(
        f"{t.split()[0]}: {breakdown.get(t, 0)}"
        for t in INFORMED_PRIORITY_ORDER
    ))

    sample_breakdown = {
        'sample_structural':    breakdown.get('Structural Ignorance', 0),
        'sample_sf_misaligned': breakdown.get('SF_Misaligned', 0),
        'sample_cultural':      breakdown.get('Cultural Disagreement', 0),
        'sample_relational':    breakdown.get('Relational Ignorance', 0),
        'sample_consensus':     breakdown.get('Consensus', 0),
    }
    return train_pool.iloc[selected].reset_index(drop=True), sample_breakdown


def load_pairs(city, voter, root_path):
    db_path = f'{root_path}/votes/db/{city}/{city}_voterAI.db'
    tbl_name = f'{city}_{voter}_votes'
    sql = f"SELECT * FROM {tbl_name}"
    city_data = retrieve_db(db_path, sql)
    return city_data

# Use Keras Sequence (recognized by model.fit)
class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, left_paths, right_paths, labels, sub_bands, batch_size):
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.labels = np.array(labels, dtype=np.int32)
        self.sub_bands = sub_bands
        self.batch_size = batch_size
        self.n = len(labels)

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n)

        # Load batch
        batch_l = []
        batch_r = []
        for i in range(start_idx, end_idx):
            img_l = load_and_preprocess_image(self.left_paths[i], self.sub_bands)
            img_r = load_and_preprocess_image(self.right_paths[i], self.sub_bands)
            batch_l.append(img_l)
            batch_r.append(img_r)

        # Convert to numpy arrays
        batch_l = np.array(batch_l, dtype=np.float32)
        batch_r = np.array(batch_r, dtype=np.float32)
        batch_labels = self.labels[start_idx:end_idx]

        return (batch_l, batch_r), batch_labels

    def on_epoch_end(self):
        """Called at the end of each epoch (optional shuffle here)"""
        pass

def create_dataset(pairs_df, root_path, sub_bands, batch_size=32):
    """Create Keras Sequence (avoids closure leaks and works with model.fit)"""
    
    # Extract paths and labels into simple lists
    left_paths = []
    right_paths = []
    labels = []
    
    for idx, row in pairs_df.iterrows():
        lp = row['left']
        rp = row['right']
        
        # Decode bytes
        if isinstance(lp, bytes):
            lp = lp.decode('utf-8')
        if isinstance(rp, bytes):
            rp = rp.decode('utf-8')
        
        left_paths.append(str(lp))
        right_paths.append(str(rp))
        labels.append(int(row['y']))
    
    return ImageSequence(left_paths, right_paths, labels, sub_bands, batch_size)

def evaluate_model(model, test_dataset):
    """Evaluate model and return predictions + probabilities"""
    all_preds = []
    all_probs = []
    
    # Iterate through batches (works with both Sequence and tf.data.Dataset)
    for i in range(len(test_dataset)):
        (images_l, images_r), labels = test_dataset[i]
        
        # Use predict_on_batch to prevent memory leaks and dropped batches
        predictions = model.predict_on_batch([images_l, images_r])
        
        # Convert to numpy to free TF graph memory immediately
        if tf.is_tensor(predictions):
            predictions = predictions.numpy()
            
        if predictions.shape[1] == 1:
            probs = predictions.flatten()
            preds = (probs > 0.5).astype(int)
        else:
            preds = np.argmax(predictions, axis=1)
            probs = predictions[:, 1]
        
        all_preds.extend(preds)
        all_probs.extend(probs)
    
    return np.array(all_preds), np.array(all_probs)

def categorize_with_iso(test_df, preds, iso_models, model, root_path, sub_bands):
    """
    Apply Isolation Forest categorization to test results.
    Returns category counts.
    """
    
    # Load ISO models
    scaler_single = iso_models['scaler_single']
    pca_single = iso_models['pca_single']
    iso_single = iso_models['iso_single']
    
    scaler_diff = iso_models['scaler_diff']
    pca_diff = iso_models['pca_diff']
    iso_pair_diff = iso_models['iso_pair_diff']
    
    # Get feature extractor
    nested_model = model.layers[2]
    single_encoder = nested_model.get_layer('feature_extractor')
    
    # Process test images in batches
    all_iso_l_class = []
    all_iso_r_class = []
    all_iso_diff_class = []
    
    batch_size = 64
    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i+batch_size]
        
        # Load images (decode bytes if needed)
        batch_l = []
        batch_r = []
        for row in batch.itertuples():
            left_path = row.left
            right_path = row.right
            
            # Decode if bytes
            if isinstance(left_path, bytes):
                left_path = left_path.decode('utf-8')
            if isinstance(right_path, bytes):
                right_path = right_path.decode('utf-8')
            
            left_path = f"{root_path}/{left_path}"
            right_path = f"{root_path}/{right_path}"
            batch_l.append(load_and_preprocess_image(left_path, sub_bands))
            batch_r.append(load_and_preprocess_image(right_path, sub_bands))
        
        batch_l = np.array(batch_l)
        batch_r = np.array(batch_r)
        
        # Extract features
        # features_l = single_encoder(batch_l).numpy()
        # features_r = single_encoder(batch_r).numpy()
        features_l = single_encoder.predict_on_batch(batch_l)
        features_r = single_encoder.predict_on_batch(batch_r)
        
        # Ensure they are numpy arrays
        if tf.is_tensor(features_l):
            features_l = features_l.numpy()
            features_r = features_r.numpy()
        
        features_l_flat = features_l.reshape(features_l.shape[0], -1)
        features_r_flat = features_r.reshape(features_r.shape[0], -1)
        
        # Single image ISO
        features_l_pca = pca_single.transform(scaler_single.transform(features_l_flat))
        features_r_pca = pca_single.transform(scaler_single.transform(features_r_flat))
        
        iso_l_class = iso_single.predict(features_l_pca)
        iso_r_class = iso_single.predict(features_r_pca)
        
        # Diff ISO
        pair_diff = features_l_flat - features_r_flat
        pair_diff_pca = pca_diff.transform(scaler_diff.transform(pair_diff))
        iso_diff_class = iso_pair_diff.predict(pair_diff_pca)
        
        all_iso_l_class.extend(iso_l_class)
        all_iso_r_class.extend(iso_r_class)
        all_iso_diff_class.extend(iso_diff_class)
    
    # Add to test_df
    test_df = test_df.copy()
    test_df['model_prediction'] = preds
    test_df['iso_left_class'] = all_iso_l_class
    test_df['iso_right_class'] = all_iso_r_class
    test_df['iso_pair_diff_class'] = all_iso_diff_class
    
    # Categorize
    def categorize(row):
        correct = (row['y'] == row['model_prediction'])
        if row['iso_pair_diff_class'] == 1:
            return 'Consensus' if correct else 'Cultural Disagreement'
        else:
            if row['iso_left_class'] == 1 and row['iso_right_class'] == 1:
                return 'Relational Ignorance'
            else:
                return 'Structural Ignorance'
    
    test_df['category'] = test_df.apply(categorize, axis=1)
    
    # Count categories
    category_counts = test_df['category'].value_counts()
    
    # Extract category labels as list  ← ADD THIS
    category_labels = test_df['category'].tolist()
    
    return {
        'consensus': category_counts.get('Consensus', 0) / len(test_df),
        'cultural_disagreement': category_counts.get('Cultural Disagreement', 0) / len(test_df),
        'relational_ignorance': category_counts.get('Relational Ignorance', 0) / len(test_df),
        'structural_ignorance': category_counts.get('Structural Ignorance', 0) / len(test_df),
        'category_labels': category_labels
    }

def train_new_iso_forests(finetune_data, model, root_path, sub_bands, contamination=0.1):
    """
    Train new Isolation Forests on fine-tuned model's features.
    Returns dict of new ISO models.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    
    # print("      Training new Isolation Forests...")
    
    # Get feature extractor
    nested_model = model.layers[2]
    single_encoder = nested_model.get_layer('feature_extractor')
    
    # Extract features from fine-tune data
    all_features_l = []
    all_features_r = []
    
    batch_size = 64
    for i in range(0, len(finetune_data), batch_size):
        batch = finetune_data.iloc[i:i+batch_size]
        
        # Load images (decode bytes if needed)
        batch_l = []
        batch_r = []
        for row in batch.itertuples():
            left_path = row.left
            right_path = row.right
            
            # Decode if bytes
            if isinstance(left_path, bytes):
                left_path = left_path.decode('utf-8')
            if isinstance(right_path, bytes):
                right_path = right_path.decode('utf-8')
            
            batch_l.append(load_and_preprocess_image(left_path, sub_bands))
            batch_r.append(load_and_preprocess_image(right_path, sub_bands))
        
        batch_l = np.array(batch_l)
        batch_r = np.array(batch_r)
        
        features_l = single_encoder.predict_on_batch(batch_l)
        features_r = single_encoder.predict_on_batch(batch_r)
        
        # Ensure they are numpy arrays
        if tf.is_tensor(features_l):
            features_l = features_l.numpy()
            features_r = features_r.numpy()
        
        features_l_flat = features_l.reshape(features_l.shape[0], -1)
        features_r_flat = features_r.reshape(features_r.shape[0], -1)
        
        all_features_l.append(features_l_flat)
        all_features_r.append(features_r_flat)
    
    all_features_l = np.vstack(all_features_l)
    all_features_r = np.vstack(all_features_r)
    
    # Stack for single-image ISO (both left and right)
    all_features_single = np.vstack([all_features_l, all_features_r])
    
    # Train single-image pipeline
    scaler_single = StandardScaler()
    features_single_scaled = scaler_single.fit_transform(all_features_single)
    
    pca_single = PCA(n_components=64, random_state=42)
    features_single_pca = pca_single.fit_transform(features_single_scaled)
    
    iso_single = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_single.fit(features_single_pca)
    
    # Train pair-diff pipeline
    pair_diff = all_features_l - all_features_r
    
    scaler_diff = StandardScaler()
    pair_diff_scaled = scaler_diff.fit_transform(pair_diff)
    
    pca_diff = PCA(n_components=64, random_state=42)
    pair_diff_pca = pca_diff.fit_transform(pair_diff_scaled)
    
    iso_pair_diff = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_pair_diff.fit(pair_diff_pca)

    # ── Score-feature ISO forest (residual approach) ──────────────────────────
    iso_pair_diff.fit(pair_diff_pca)

    # SF ISO is intentionally NOT retrained here. The source pre-trained ISO
    # defines a fixed reference: "what does the source city consider a normal
    # appearance-to-rank relationship?" Fine-tuning SF evaluation always compares
    # against that fixed source standard via iso_models (passed separately).
    # Retraining on target finetune data would move the reference point and make
    # zero-shot vs fine-tuning comparisons meaningless.

    return {
        'scaler_single':    scaler_single,
        'pca_single':       pca_single,
        'iso_single':       iso_single,
        'scaler_diff':      scaler_diff,
        'pca_diff':         pca_diff,
        'iso_pair_diff':    iso_pair_diff,
    }

def _get_sf_misalignment_flags(data, model, root_path, sub_bands, iso_models,
                                all_target_data=None):
    """
    Per-location score-feature misalignment classification.
 
    Requires an already-loaded model so it can be called from within a function
    that has the model in scope (avoiding an extra model load).
 
    Parameters
    ----------
    data : pd.DataFrame
        Pairs used to determine which locations to evaluate (test set).
    all_target_data : pd.DataFrame, optional
        Full target city data (train pool + test set) used ONLY for BT score
        estimation. More pairs → more stable BT estimates per location.
        If None, falls back to using `data` (30% test set), which produces
        noisy BT scores and near-zero residuals that collapse the ISO scores.
 
    Returns a dict {path: flag} where:
        True  — image-inlier with OOD (image, rank) pairing (misaligned)
        False — image-inlier with expected (image, rank) pairing
        None  — image-outlier (SF analysis uninformative for this location)
    Returns {} if SF ISO models are not available.
    """
    if iso_models.get('iso_score_feature') is None:
        return {}
 
    scaler_single       = iso_models['scaler_single']
    pca_single          = iso_models['pca_single']
    iso_single          = iso_models['iso_single']
    mlp_score_regressor = iso_models['mlp_score_regressor']
    scaler_residual     = iso_models['scaler_residual']
    iso_score_feature   = iso_models['iso_score_feature']
 
    nested_model   = model.layers[2]
    single_encoder = nested_model.get_layer('feature_extractor')
 
    # Collect unique locations from the evaluation set
    seen, unique_locs = set(), []
    for _, row in data.iterrows():
        for col in ('left', 'right'):
            p = row[col]
            if isinstance(p, bytes): p = p.decode('utf-8')
            p = str(p)
            if p not in seen:
                seen.add(p)
                unique_locs.append(p)
 
    # BT score estimation — use all target data when available for stable estimates.
    # With only 30% of pairs, scores per location are poorly constrained and
    # residuals don't reliably reflect the true appearance-to-rank mismatch.
    bt_data = all_target_data if all_target_data is not None else data
    true_bt_scores = estimate_bt_scores_from_predictions(bt_data, bt_data['y'].values)
 
    # Extract features for unique locations in batches
    batch_size, loc_to_feat = 64, {}
    for i in range(0, len(unique_locs), batch_size):
        batch_locs = unique_locs[i:i + batch_size]
        images = np.array([
            load_and_preprocess_image(f"{root_path}/{p}", sub_bands)
            for p in batch_locs
        ])
        feats = single_encoder.predict_on_batch(images)
        if tf.is_tensor(feats): feats = feats.numpy()
        feats_flat = feats.reshape(feats.shape[0], -1)
        for j, p in enumerate(batch_locs):
            loc_to_feat[p] = feats_flat[j]
 
    locs        = list(loc_to_feat.keys())
    feat_matrix = np.array([loc_to_feat[p] for p in locs])
    feat_pca    = pca_single.transform(scaler_single.transform(feat_matrix))
    img_cls     = iso_single.predict(feat_pca)   # 1=inlier, -1=outlier
 
    flags = {loc: None for loc, cls in zip(locs, img_cls) if cls == -1}
 
    # Score-feature classification for image-inliers with BT scores
    inlier_indices = [i for i, c in enumerate(img_cls)
                      if c == 1 and locs[i] in true_bt_scores]
    if inlier_indices:
        sf_locs     = [locs[i] for i in inlier_indices]
        sf_pca      = feat_pca[np.array(inlier_indices)]
        sf_bt       = np.array([true_bt_scores[loc] for loc in sf_locs])
        residuals   = sf_bt - mlp_score_regressor.predict(sf_pca)
        res_scaled  = scaler_residual.transform(residuals.reshape(-1, 1))
        # ISO predict on absolute scaled residual — consistent with training
        sf_cls    = iso_score_feature.predict(np.abs(res_scaled))
        sf_scores = iso_score_feature.decision_function(np.abs(res_scaled))
        print(f"      [SF] residuals: mean={residuals.mean():.3f} std={residuals.std():.3f} "
              f"range=[{residuals.min():.3f}, {residuals.max():.3f}]")
        print(f"      [SF] ISO scores: mean={sf_scores.mean():.3f} "
              f"min={sf_scores.min():.3f} max={sf_scores.max():.3f} "
              f"outliers={(sf_cls==-1).sum()}/{len(sf_cls)}")
        for loc, cls in zip(sf_locs, sf_cls):
            flags[loc] = bool(cls == -1)   # Python bool — numpy.bool_ would break 'is True' checks
 
    # Image-inliers with no BT score: treat as not misaligned
    for loc, cls in zip(locs, img_cls):
        if cls == 1 and loc not in flags:
            flags[loc] = False
 
    return flags


def compute_score_feature_misalignment(test_data, model, root_path, sub_bands,
                                        iso_models, all_target_data=None):
    """
    Aggregate score-feature misalignment statistics over the test set.
    Delegates per-location classification to _get_sf_misalignment_flags.

    Parameters
    ----------
    all_target_data : pd.DataFrame, optional
        Full target city data passed through to _get_sf_misalignment_flags
        for more stable BT score estimation (see that function's docstring).
    """
    null = {'sf_misalignment_pct': None,
            'sf_image_inlier_pct': None,
            'sf_n_evaluated':      None}

    if iso_models.get('iso_score_feature') is None:
        return null

    try:
        flags = _get_sf_misalignment_flags(
            test_data, model, root_path, sub_bands, iso_models,
            all_target_data=all_target_data
        )
        if not flags:
            return null

        n_total          = len(flags)
        inlier_flags     = {loc: f for loc, f in flags.items() if f is not None}
        n_inliers        = len(inlier_flags)
        image_inlier_pct = float(n_inliers / n_total) if n_total > 0 else None

        n_misaligned = sum(1 for f in inlier_flags.values() if f)
        sf_n_evaluated      = n_inliers
        sf_misalignment_pct = float(n_misaligned / n_inliers) if n_inliers >= 5 else None

        return {
            'sf_misalignment_pct': sf_misalignment_pct,
            'sf_image_inlier_pct': image_inlier_pct,
            'sf_n_evaluated':      sf_n_evaluated,
        }

    except Exception as e:
        print(f"      [Score-feature misalignment error: {e}]")
        import traceback
        traceback.print_exc()
        return null


def categorize_pool_for_sampling(model_path, train_pool, root_path, sub_bands,
                                  iso_models, all_target_data=None):
    """
    Categorize all pairs in train_pool for informed sampling.

    Called once per source-target pair before the sample_size / seed loops.
    Loads the model once and computes both:
      - AoA category labels (pair-level, from categorize_with_iso)
      - SF misalignment flags (pair-level, derived from _get_sf_misalignment_flags)

    The SF flags are computed while the model is already loaded so no extra
    model load is needed.

    Returns
    -------
    aoa_labels : list[str]
        AoA category per pair, in train_pool row order.
    sf_pair_flags : list[bool]
        True if either location in the pair is score-feature misaligned
        (image-inlier with unexpected BT rank). False otherwise (including
        when SF models are unavailable or the location is an image-outlier).
    """
    from collections import Counter

    print(f"  Categorizing train pool for informed sampling ({len(train_pool)} pairs)...")
    tf.keras.backend.clear_session()

    model        = tf.keras.models.load_model(model_path)
    pool_dataset = create_dataset(train_pool, root_path, sub_bands, BATCH_SIZE)
    pool_preds, _ = evaluate_model(model, pool_dataset)

    # AoA categories (pair-level)
    categories = categorize_with_iso(
        train_pool, pool_preds, iso_models, model, root_path, sub_bands
    )
    aoa_labels = categories['category_labels']

    # SF misalignment flags (location-level → pair-level)
    # Computed here while the model is in scope — no extra model load needed.
    sf_loc_flags = _get_sf_misalignment_flags(
        train_pool, model, root_path, sub_bands, iso_models,
        all_target_data=all_target_data
    )

    del model, pool_dataset
    tf.keras.backend.clear_session()
    gc.collect()

    # A pair is SF-misaligned if either location is flagged True.
    # None (image-outlier) and False (inlier, expected rank) both map to False.
    sf_pair_flags = []
    for _, row in train_pool.iterrows():
        left  = row['left'].decode('utf-8')  if isinstance(row['left'],  bytes) else str(row['left'])
        right = row['right'].decode('utf-8') if isinstance(row['right'], bytes) else str(row['right'])
        sf_pair_flags.append(
            sf_loc_flags.get(left) is True or sf_loc_flags.get(right) is True
        )

    # Log AoA breakdown
    counts = Counter(aoa_labels)
    total  = len(aoa_labels)
    print(f"    AoA: " + ", ".join(
        f"{cat.split()[0]} {counts.get(cat, 0)} ({100 * counts.get(cat, 0) / total:.0f}%)"
        for cat in ['Structural Ignorance', 'Cultural Disagreement',
                    'Relational Ignorance', 'Consensus']
    ))
    if sf_loc_flags:
        n_sf = sum(sf_pair_flags)
        print(f"    SF-misaligned pairs: {n_sf} ({100 * n_sf / total:.0f}%)")
    else:
        print(f"    SF models not available — SF tier will be skipped")

    return aoa_labels, sf_pair_flags


def compute_pair_compatibility_flags(aoa_labels, zero_accuracy,
                                      sf_pair_flags=None,
                                      zero_spearman=None,
                                      zero_categories_dict=None):
    """
    Post-hoc diagnostic: estimate whether a source→target pair is likely to
    benefit from fine-tuning, without running any additional model inference.

    Called once per source-target pair, using information already available
    after the zero-shot evaluation and pool categorisation:
      • aoa_labels            — AoA category label for every pair in the train pool
      • zero_accuracy         — zero-shot pairwise accuracy on the held-out test set
      • sf_pair_flags         — bool list, one per pool pair; True = SF-misaligned
                                (from categorize_pool_for_sampling). If supplied,
                                the SF override is applied before computing fractions
                                so categories are mutually exclusive (same rule as
                                the informed samplers). If None, SF is omitted from
                                learnable_pct and pool_sf_pct is 0.
      • zero_spearman         — zero-shot Spearman ρ on the held-out test set
      • zero_categories_dict  — unused; retained for backward compatibility

    Compatibility label
    ───────────────────
    Derived directly from the Spearman ceiling estimate, so the label is fully
    consistent with the ceiling formula. Three tiers:

        'good'         — ceiling ≥ 0.50: fine-tuning expected to reach useful
                         performance with moderate annotation effort.
        'caution'      — ceiling 0.30–0.50: fine-tuning may help at larger n
                         but outcome is uncertain; proceed with awareness of risk.
        'incompatible' — ceiling < 0.30: fine-tuning unlikely to reach useful
                         performance at any realistic n; local data collection
                         is recommended instead.

    When zero_spearman is not supplied the ceiling cannot be computed and
    the label falls back to 'unknown'.

    Ceiling estimates
    ─────────────────
    ft_benefit_ceiling_est (pairwise accuracy)
        Weighted blend of zero-shot accuracy and learnable-pair uplift.
        Formula: 0.6 × ZS_acc + 0.4 × (ZS_acc + learnable_pct × 0.3),
        capped at 0.80 (empirical ceiling from existing results).

    ft_spearman_ceiling_est (Spearman rank correlation)
        Calibrated ceiling estimate for rank recovery after fine-tuning at
        n=5,000. Designed so the diagonal (predicted = actual) cuts through
        the centre of the distribution (~52% of pairs fall below the ceiling).

        Formula:
            sp_ceiling = 0.66
                       + 0.45 × ZS_sp
                       + 0.25 × learnable_pct
                       − 0.90 × cultural_disagreement_pool
                       − 0.30 × structural_ignorance_pool
            capped at 0.95.

        Term interpretation:
          0.66              — empirical base uplift from fine-tuning.
          +0.45 × ZS_sp     — partial anchor on zero-shot quality; strong ZS
                              signals transferable feature representations.
          +0.25 × learnable — learnable_pct = SF_Misaligned + Relational Ignorance.
                              These categories provide genuine corrective signal:
                              familiar images with correctable errors. Cultural
                              Disagreement is intentionally excluded — it reflects
                              genuine annotator disagreement that fine-tuning
                              cannot resolve, not a learnable mapping error.
                              Note: in practice relational ignorance is near-zero
                              in the current dataset, so learnable_pct ≈ pool_sf_pct.
                              pool_relational_pct is stored separately so future
                              datasets with more relational pairs are handled
                              correctly without formula changes.
          −0.90 × CD        — cultural disagreement is the dominant ceiling
                              suppressor. Persistent annotator conflict cannot
                              be corrected through limited fine-tuning; it
                              represents a genuine preference misalignment.
          −0.30 × SI        — structural ignorance (OOD images) limits what
                              fine-tuning can learn at moderate sample sizes.

        Returns None if zero_spearman is not supplied.

    Parameters
    ----------
    aoa_labels : list[str]
        AoA category label for each pair in the train pool.
    zero_accuracy : float
        Zero-shot pairwise accuracy on the held-out test set.
    sf_pair_flags : list[bool] or None
        Per-pair SF misalignment flags from categorize_pool_for_sampling.
        If supplied, the SF override is applied before computing fractions.
    zero_spearman : float or None
        Zero-shot Spearman ρ on the held-out test set.
    zero_categories_dict : dict or None
        Unused; retained for backward compatibility.

    Returns
    -------
    dict with keys:
        pool_structural_pct        — fraction of pool pairs: Structural Ignorance
                                     (after SF override)
        pool_sf_pct                — fraction of pool pairs: SF_Misaligned
                                     (after SF override; 0.0 if sf_pair_flags
                                     not supplied)
        pool_relational_pct        — fraction of pool pairs: Relational Ignorance
                                     (after SF override)
        pool_learnable_pct         — SF_Misaligned + Relational Ignorance
                                     (after SF override); Cultural Disagreement
                                     excluded — it is a ceiling suppressor, not
                                     a learnable signal
        pair_compatibility         — 'good' | 'caution' | 'incompatible' |
                                     'unknown'
        ft_benefit_ceiling_est     — float in [0, 1]; pairwise accuracy ceiling
        ft_spearman_ceiling_est    — float in [-1, 1] or None; Spearman ceiling
    """
    from collections import Counter

    total = len(aoa_labels) or 1   # guard division by zero

    # ── Apply SF override to get mutually exclusive effective categories ──────
    # Same rule as informed samplers:
    #   Structural Ignorance → keeps AoA label (SF uninformative for OOD)
    #   Image-inlier + SF-flagged → SF_Misaligned (overrides AoA label)
    #   Otherwise → raw AoA label
    if sf_pair_flags is not None and len(sf_pair_flags) == len(aoa_labels):
        effective = []
        for aoa, is_sf in zip(aoa_labels, sf_pair_flags):
            if aoa == 'Structural Ignorance':
                effective.append('Structural Ignorance')
            elif is_sf:
                effective.append('SF_Misaligned')
            else:
                effective.append(aoa)
    else:
        # No SF flags available — use raw AoA labels; SF_Misaligned = 0
        effective = list(aoa_labels)

    counts = Counter(effective)

    structural_pct = counts.get('Structural Ignorance', 0) / total
    sf_pct         = counts.get('SF_Misaligned', 0)        / total
    cultural_pct   = counts.get('Cultural Disagreement', 0) / total
    relational_pct = counts.get('Relational Ignorance', 0)  / total

    # Learnable = categories that provide genuine corrective signal to fine-tuning:
    #   SF_Misaligned      — familiar images, wrong score-feature mapping
    #   Relational Ignorance — familiar images, uncertain comparison direction
    # Cultural Disagreement is intentionally excluded: it reflects genuine
    # annotator preference misalignment that cannot be resolved by more examples.
    # Consensus and Structural Ignorance are excluded as before.
    learnable_pct = sf_pct + relational_pct

    # ── Pairwise accuracy ceiling ─────────────────────────────────────────────
    acc_ceiling = min(
        0.6 * zero_accuracy + 0.4 * (zero_accuracy + learnable_pct * 0.3),
        0.80
    )

    # ── Spearman ceiling ──────────────────────────────────────────────────────
    if zero_spearman is not None:
        sp_ceiling = min(
            0.66
            + 0.45 * zero_spearman
            + 0.25 * learnable_pct
            - 0.90 * cultural_pct
            - 0.30 * structural_pct,
            0.95
        )
        sp_ceiling_out = float(sp_ceiling)
    else:
        sp_ceiling_out = None

    # ── Compatibility label — derived from Spearman ceiling ───────────────────
    if sp_ceiling_out is None:
        compatibility = 'unknown'
    elif sp_ceiling_out >= 0.50:
        compatibility = 'good'
    elif sp_ceiling_out >= 0.30:
        compatibility = 'caution'
    else:
        compatibility = 'incompatible'

    _zs_sp_str   = f"{zero_spearman:.3f}"  if zero_spearman   is not None else 'n/a'
    _sp_ceil_str = f"{sp_ceiling_out:.3f}" if sp_ceiling_out is not None else 'n/a'
    print(f"    [Compatibility] structural={structural_pct:.2f} sf={sf_pct:.2f} "
          f"relational={relational_pct:.2f} cultural={cultural_pct:.2f} "
          f"learnable={learnable_pct:.2f} "
          f"ZS_acc={zero_accuracy:.3f} ZS_sp={_zs_sp_str} "
          f"→ {compatibility} "
          f"(acc_ceil={acc_ceiling:.3f} sp_ceil={_sp_ceil_str})")

    return {
        'pool_structural_pct':      float(structural_pct),
        'pool_sf_pct':              float(sf_pct),
        'pool_relational_pct':      float(relational_pct),
        'pool_learnable_pct':       float(learnable_pct),
        'pair_compatibility':       compatibility,
        'ft_benefit_ceiling_est':   float(acc_ceiling),
        'ft_spearman_ceiling_est':  sp_ceiling_out,
    }


def run_zeroshot_evaluation(model_path, test_data, root_path, sub_bands, target_city,
                             iso_models, all_target_data=None):
    """Encapsulates zero-shot evaluation to prevent initial memory bloat."""
    
    # 1. Clear session before starting
    tf.keras.backend.clear_session()
    
    # 2. Load model and create dataset
    zero_model = tf.keras.models.load_model(model_path)
    test_dataset = create_dataset(test_data, root_path, sub_bands, BATCH_SIZE)
    
    # 3. Evaluate
    zero_shot_preds, zero_shot_probs = evaluate_model(zero_model, test_dataset)
    
    # Calculate accuracy safely
    y_true_zero = np.array(test_data['y'], dtype=int).flatten()
    y_pred_zero = np.array(zero_shot_preds, dtype=int).flatten()
    zero_accuracy = (y_true_zero == y_pred_zero).mean()
    
    # 4. BT Metrics
    zero_bt_metrics = evaluate_model_bt_recovery(
        test_data, zero_shot_probs
    )
    
    # 4b. BT Score-Space Decompositions (location-level, output-space)
    zero_bt_decomp = compute_bt_score_decompositions(
        test_data=test_data,
        test_probs=zero_shot_probs
    )
    
    # 5. Categories
    zero_categories = categorize_with_iso(
        test_data, zero_shot_preds, iso_models,
        zero_model, root_path, sub_bands
    )
    
    # 6. Category BT metrics
    zero_category_bt = add_bt_metrics_to_categories(
        test_data=test_data,
        test_probs=zero_shot_probs,
        categories_dict=zero_categories
    )

    # 7. Score-feature misalignment
    zero_sf_metrics = compute_score_feature_misalignment(
        test_data=test_data,
        model=zero_model,
        root_path=root_path,
        sub_bands=sub_bands,
        iso_models=iso_models,
        all_target_data=all_target_data
    )
    
    # Calculate relational ratio safely
    rel_ign = zero_categories.get('relational_ignorance', 0)
    str_ign = zero_categories.get('structural_ignorance', 0)
    zero_relational_ratio = (rel_ign / (rel_ign + str_ign + 1e-10)) if (rel_ign + str_ign) > 0 else 0
    zero_categories.pop('category_labels', None)
    
    # 8. Package only primitive results
    final_results = {
        'accuracy': float(zero_accuracy),
        'improvement': 0.0,
        'spearman': float(zero_bt_metrics['spearman']),
        **zero_bt_metrics,
        **zero_categories,
        'relational_ratio': float(zero_relational_ratio),
        **zero_category_bt,
        **zero_bt_decomp,
        **zero_sf_metrics
    }
    
    # 8. Aggressive cleanup
    del zero_model
    del test_dataset
    tf.keras.backend.clear_session()
    
    return final_results

def run_finetuning_iteration(model_path, strategy_config, train_pool, test_data,
                              sample_size, seed, root_path, sub_bands, target_city,
                              zero_accuracy, zero_bt_metrics, iso_models,
                              sampling_strategy='random', train_pool_categories=None,
                              all_target_data=None):
    """Encapsulates a single training run to guarantee memory release."""
    # 1. ALWAYS clear session before loading a new model
    tf.keras.backend.clear_session()
    
    # 2. Load fresh from disk (No cloning!)
    model = tf.keras.models.load_model(model_path)
    
    # ── Configure training ────────────────────────────────────────────────────
    # Two paths:
    #   'full'      — all layers trainable, single Adam, standard model.fit.
    #   'layerwise' — BN frozen, three Adam optimizers with decayed LRs per
    #                 block group + a higher LR for the strength_output head.
    #                 Uses a custom train_step so model.fit / EarlyStopping
    #                 work unchanged.

    if training_strategy == 'layerwise':
        # ── Variable-level approach ───────────────────────────────────────────
        # Rather than fighting Keras's layer.trainable propagation across nested
        # models (which is unreliable after load_model), we:
        #   1. Set all layers trainable=True so model.trainable_variables is complete
        #   2. Collect all trainable variables
        #   3. Exclude BN variables by name at the optimizer assignment step
        #   4. Assign remaining variables to block groups by Conv2D filter count
        #
        # This avoids the layer.trainable=False propagation bug entirely — BN
        # variables are simply never passed to any optimizer, so they receive
        # zero gradient updates (equivalent to being frozen).

        for layer in model.layers:
            layer.trainable = True
            if hasattr(layer, 'layers'):
                for sub in layer.layers:
                    sub.trainable = True
                    if hasattr(sub, 'layers'):
                        for subsub in sub.layers:
                            subsub.trainable = True

        def _find_layer(root, name):
            """Recursively find the first layer with the given name."""
            for layer in root.layers:
                if layer.name == name:
                    return layer
                if hasattr(layer, 'layers'):
                    found = _find_layer(layer, name)
                    if found is not None:
                        return found
            return None

        def _find_all_by_type(root, layer_type, _seen=None):
            """Recursively collect all layers of a given type in traversal order."""
            if _seen is None:
                _seen = set()
            results = []
            for layer in root.layers:
                if id(layer) in _seen:
                    continue
                _seen.add(id(layer))
                if isinstance(layer, layer_type):
                    results.append(layer)
                if hasattr(layer, 'layers'):
                    results.extend(_find_all_by_type(layer, layer_type, _seen))
            return results

        # All trainable variables — BN variables identified by name for exclusion
        all_vars = model.trainable_variables
        is_bn    = lambda v: 'batch_normalization' in v.name

        # strength_output head — search by name, fallback to Dense(1) linear
        strength_layer = _find_layer(model, 'strength_output')
        if strength_layer is None:
            all_dense = _find_all_by_type(model, tf.keras.layers.Dense)
            dense_1   = [l for l in all_dense
                         if l.units == 1
                         and getattr(l.activation, '__name__', '') == 'linear']
            if dense_1:
                strength_layer = dense_1[0]
                print(f"      [Layerwise] 'strength_output' not found by name; "
                      f"using '{strength_layer.name}' (Dense(1) linear)")
            else:
                raise RuntimeError(
                    "Cannot find strength output layer. "
                    f"Top-level layers: {[l.name for l in model.layers]}"
                )
        head_var_ids = {id(v) for v in strength_layer.variables}
        vars_head    = [v for v in all_vars if id(v) in head_var_ids and not is_bn(v)]

        # Conv2D blocks in traversal order — gives us anchor variable sets
        # Block 1: conv 0–1 (16 filters), Block 2: conv 2–3 (32 filters),
        # Block 3: conv 4–6 (64 filters)
        all_conv  = _find_all_by_type(model, tf.keras.layers.Conv2D)
        b1_ids = {id(v) for l in all_conv[0:2] for v in l.variables}
        b2_ids = {id(v) for l in all_conv[2:4] for v in l.variables}
        b3_ids = {id(v) for l in all_conv[4:7] for v in l.variables}

        # SE block Dense layers and any other non-BN, non-head variables:
        # assign to the block whose Conv2D layers they sit closest to by
        # partitioning the full variable list into three equal-ish segments
        # based on the index positions of the Conv2D anchor variables.
        all_var_ids = [id(v) for v in all_vars]

        def _block_range(anchor_ids):
            """Return (min_idx, max_idx) of anchor variables in all_vars."""
            idxs = [i for i, v in enumerate(all_vars) if id(v) in anchor_ids]
            return (min(idxs), max(idxs)) if idxs else (0, 0)

        b1_range = _block_range(b1_ids)
        b2_range = _block_range(b2_ids)
        b3_range = _block_range(b3_ids)
        head_range = _block_range(head_var_ids)

        # Boundaries: vars before midpoint(b1,b2) → block1,
        #             vars before midpoint(b2,b3) → block2,
        #             vars before head             → block3,
        #             head vars                    → head
        mid12 = (b1_range[1] + b2_range[0]) // 2
        mid23 = (b2_range[1] + b3_range[0]) // 2
        mid3h = (b3_range[1] + head_range[0]) // 2

        vars_block1, vars_block2, vars_block3 = [], [], []
        for i, v in enumerate(all_vars):
            if is_bn(v) or id(v) in head_var_ids:
                continue   # BN → excluded; head → separate group
            if i <= mid12:
                vars_block1.append(v)
            elif i <= mid23:
                vars_block2.append(v)
            else:
                vars_block3.append(v)

        n_bn = sum(1 for v in all_vars if is_bn(v))
        print(f"      [Layerwise] vars — "
              f"block1={len(vars_block1)} block2={len(vars_block2)} "
              f"block3={len(vars_block3)} head={len(vars_head)} "
              f"bn_excluded={n_bn}")

        # ── Three Adam optimizers ─────────────────────────────────────────────
        _base = strategy_config['learning_rate']
        opt_block1 = tf.keras.optimizers.Adam(learning_rate=_base * (LAYERWISE_DECAY**2))
        opt_block2 = tf.keras.optimizers.Adam(learning_rate=_base * LAYERWISE_DECAY)
        opt_block3 = tf.keras.optimizers.Adam(learning_rate=_base * (LAYERWISE_DECAY**0.5))
        opt_head   = tf.keras.optimizers.Adam(learning_rate=_base * HEAD_LR_MULTIPLIER)

        loss_fn = tf.keras.losses.BinaryCrossentropy()

        # ── Custom train_step via GradientTape ────────────────────────────────
        # Compute gradients once, apply with each group's optimizer.
        # Overriding train_step keeps model.fit, callbacks, and EarlyStopping
        # working without modification.
        @tf.function
        def layerwise_train_step(data):
            x, y = data
            with tf.GradientTape() as tape:
                y_pred = model(x, training=True)
                loss   = loss_fn(y, y_pred)

            grads = tape.gradient(loss, model.trainable_variables)
            # Map each gradient to its variable group
            grad_map = {id(v): g for v, g in zip(model.trainable_variables, grads)
                        if g is not None}

            def _apply(opt, var_list):
                gv = [(grad_map[id(v)], v) for v in var_list if id(v) in grad_map]
                if gv:
                    opt.apply_gradients(gv)

            _apply(opt_block1, vars_block1)
            _apply(opt_block2, vars_block2)
            _apply(opt_block3, vars_block3)
            _apply(opt_head,   vars_head)

            # Update metrics so EarlyStopping can monitor 'loss'
            model.compiled_metrics.update_state(y, y_pred)
            results = {m.name: m.result() for m in model.metrics}
            results['loss'] = loss
            return results

        model.train_step = layerwise_train_step

        # Compile with a dummy optimizer (never used — gradients are applied
        # manually above, but compile is required for model.fit to work).
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=_base),
            loss=loss_fn,
            metrics=['accuracy']
        )

    else:
        # ── Full finetuning — single optimizer, all layers trainable ──────────
        for layer in model.layers:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=strategy_config['learning_rate']
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    # Sample data — random, original informed, or unified temperature-based.
    # All paths produce an identical `sample_breakdown` dict so nothing
    # downstream needs to change.

    if sampling_strategy == 'informed' and train_pool_categories is not None:
        # Original hard-fill, original priority order — backward-compat path.
        aoa_labels, sf_pair_flags = train_pool_categories
        finetune_data, sample_breakdown = _informed_sample(
            train_pool, aoa_labels, sf_pair_flags, sample_size, seed
        )
    elif sampling_strategy == 'informed_t' and train_pool_categories is not None \
            and not informed_adaptive:
        # Unified fixed-weight sampler — consistent policy across all pairs.
        aoa_labels, sf_pair_flags = train_pool_categories
        finetune_data, sample_breakdown = _informed_sample_unified(
            train_pool, aoa_labels, sf_pair_flags, sample_size, seed,
            temperature=informed_temperature
        )
    elif sampling_strategy == 'informed_t' and train_pool_categories is not None \
            and informed_adaptive:
        # Adaptive sampler — weights scaled by pool category fractions.
        aoa_labels, sf_pair_flags = train_pool_categories
        finetune_data, sample_breakdown = _informed_sample_adaptive(
            train_pool, aoa_labels, sf_pair_flags, sample_size, seed,
            temperature=informed_temperature
        )
    else:
        finetune_data = train_pool.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        # Categorize the random sample for breakdown logging.
        # Apply the same SF override used by the informed samplers so that
        # categories are mutually exclusive and sum to sample_size — making
        # the random breakdown directly comparable to informed breakdowns.
        # Rule: Structural Ignorance keeps its AoA label (SF uninformative for OOD);
        #       for image-inliers, SF flag overrides the AoA label.
        from collections import Counter
        _ft_dataset = create_dataset(finetune_data, root_path, sub_bands, BATCH_SIZE)
        _ft_preds, _ = evaluate_model(model, _ft_dataset)
        del _ft_dataset
        _ft_cats = categorize_with_iso(
            finetune_data, _ft_preds, iso_models, model, root_path, sub_bands
        )
        _ft_sf_flags = _get_sf_misalignment_flags(
            finetune_data, model, root_path, sub_bands, iso_models,
            all_target_data=all_target_data
        )
        _ft_aoa_labels = _ft_cats.get('category_labels', [])

        # Build effective labels with SF override — mirrors the informed samplers
        _ft_effective = []
        for _, row in finetune_data.iterrows():
            left  = row['left'].decode('utf-8')  if isinstance(row['left'],  bytes) else str(row['left'])
            right = row['right'].decode('utf-8') if isinstance(row['right'], bytes) else str(row['right'])
            sf_flagged = (_ft_sf_flags.get(left) is True
                          or _ft_sf_flags.get(right) is True)
            _ft_effective.append(sf_flagged)

        _eff_counts = Counter()
        for aoa, is_sf in zip(_ft_aoa_labels, _ft_effective):
            if aoa == 'Structural Ignorance':
                _eff_counts['Structural Ignorance'] += 1
            elif is_sf:
                _eff_counts['SF_Misaligned'] += 1
            else:
                _eff_counts[aoa] += 1

        sample_breakdown = {
            'sample_structural':    _eff_counts.get('Structural Ignorance', 0),
            'sample_sf_misaligned': _eff_counts.get('SF_Misaligned', 0),
            'sample_cultural':      _eff_counts.get('Cultural Disagreement', 0),
            'sample_relational':    _eff_counts.get('Relational Ignorance', 0),
            'sample_consensus':     _eff_counts.get('Consensus', 0),
        }
        _bd_total = sum(sample_breakdown.values())
        print(f"      [Random] "
              f"Structural: {sample_breakdown['sample_structural']}, "
              f"SF: {sample_breakdown['sample_sf_misaligned']}, "
              f"Cultural: {sample_breakdown['sample_cultural']}, "
              f"Relational: {sample_breakdown['sample_relational']}, "
              f"Consensus: {sample_breakdown['sample_consensus']} "
              f"(total={_bd_total})")
    
    # Create datasets
    train_dataset = create_dataset(finetune_data, root_path, sub_bands, BATCH_SIZE)
    test_dataset = create_dataset(test_data, root_path, sub_bands, BATCH_SIZE)
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='loss', 
        patience=PATIENCE, 
        restore_best_weights=True
    )
    
    # Train
    model.fit(train_dataset, epochs=FINE_TUNE_EPOCHS, callbacks=[early_stop], verbose=0)
    
    # Evaluate (using the predict_on_batch fix we added earlier)
    test_preds, test_probs = evaluate_model(model, test_dataset)
    
    # Calculate accuracy safely (using the int casting fix)
    y_true_ft = np.array(test_data['y'], dtype=int).flatten()
    y_pred_ft = np.array(test_preds, dtype=int).flatten()
    accuracy = (y_true_ft == y_pred_ft).mean()
    improvement = accuracy - zero_accuracy
    
    # BT Metrics
    bt_metrics = evaluate_model_bt_recovery(
        test_data, test_probs
    )
    
    # BT Score-Space Decompositions (location-level, output-space)
    bt_decomp = compute_bt_score_decompositions(
        test_data=test_data,
        test_probs=test_probs
    )
    
    overall_spearman = bt_metrics['spearman']
    sp_improvement = overall_spearman - zero_bt_metrics['spearman']
    
    print(f"→ Acc: {accuracy:.3f} (Δ{improvement:+.3f})", end=' ')
    print(f"→ Spearman: {overall_spearman:.3f} (Δ{sp_improvement:+.3f})")
    # =========================================================
    # Categorization (conditional on strategy and sample size)
    # =========================================================

    should_categorize = True
    iso_models_to_use = train_new_iso_forests(
        finetune_data, model, root_path, sub_bands
    )

    # Categorize if applicable
    if should_categorize:
        categories = categorize_with_iso(
            test_data, test_preds, iso_models_to_use,
            model, root_path, sub_bands
        )
        category_bt_metrics = add_bt_metrics_to_categories(
            test_data=test_data,
            test_probs=test_probs,
            categories_dict=categories
        )
        category_labels_list = categories.pop('category_labels', None)
    else:
        # No categorization
        categories = {
            'consensus': None,
            'cultural_disagreement': None,
            'relational_ignorance': None,
            'structural_ignorance': None,
        }
        category_bt_metrics = {
            'consensus_bt_error': None,
            'consensus_spearman': None,
            'cultural_bt_error': None,
            'cultural_spearman': None,
            'relational_bt_error': None,
            'relational_spearman': None,
            'structural_bt_error': None,
            'structural_spearman': None,
        }
    relational_ratio = None
    if categories['relational_ignorance'] is not None:
        relational_ratio = categories['relational_ignorance'] / (
            categories['relational_ignorance'] +
            categories['structural_ignorance'] + 1e-10
        )

    # Score-feature misalignment always uses the source pre-trained iso_models —
    # the fixed reference against which both zero-shot and fine-tuning are compared.
    sf_metrics = compute_score_feature_misalignment(
        test_data=test_data,
        model=model,
        root_path=root_path,
        sub_bands=sub_bands,
        iso_models=iso_models,
        all_target_data=all_target_data
    )

    final_results = {
        'accuracy': float(accuracy),
        'improvement': float(improvement),
        'spearman': float(bt_metrics['spearman']),
        **bt_metrics,
        **categories,
        'relational_ratio': relational_ratio,
        **category_bt_metrics,
        **bt_decomp,
        **sf_metrics,
        **sample_breakdown
    }
    
    del model
    del train_dataset
    del test_dataset
    del finetune_data
    
    # Clear session AT THE END of the function so the graph is destroyed immediately
    tf.keras.backend.clear_session()
    
    # Return result
    return final_results

# ============================================================================
# Main Fine-Tuning Loop
# ============================================================================

# Build test sets once per target city
target_test_sets = {}
for target_city in tgt_cities:
    target_data = load_pairs(target_city, voter, root_path)
    target_data = target_data.sample(frac=1, random_state=42).reset_index(drop=True)
    train_pool, test_data = train_test_split(target_data, test_size=0.3, random_state=42)
    target_test_sets[target_city] = (train_pool, test_data)

for source_city in src_cities:
    model_name = f"{source_city}_voter_{voter}_ensembleVoter"
    model_path = f'{root_path}/trained_models/voter/{model_name}_{d_src}_1_bradley_terry.ckpt'
    
    for target_city in tgt_cities:
        if _SRC_TRAIN.get(source_city, {source_city}) & _MEMBERS.get(target_city, {target_city}):
            print(f"SKIPPING SOURCE: {source_city} | TARGET: {target_city} (training overlap)")
            continue
        pair_results = []
        pair_output_path = f"{RESULTS_DIR}/{exp}_{source_city}_{target_city}_{run_tag}.csv"
        
        pair_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"SOURCE: {source_city} → TARGET: {target_city}")
        print(f"{'='*70}")
        
        try:
            # Load target city data
            # target_data = load_pairs(target_city, voter, root_path)
            # print(f"Loaded {len(target_data)} pairs from {target_city}")
            
            sub_bands, n_width, n_height, n_channels = get_sub_bands(d_src)
            
            # Load Isolation Forest models (for categorization)
            iso_models = {
                'scaler_single': joblib.load(f'{iso_model_dir}/models/{source_city}_{voter}_single_scaler.joblib'),
                'pca_single': joblib.load(f'{iso_model_dir}/models/{source_city}_{voter}_single_pca.joblib'),
                'iso_single': joblib.load(f'{iso_model_dir}/models/{source_city}_{voter}_single_iso_pca.joblib'),
                'scaler_diff': joblib.load(f'{iso_model_dir}/models/{source_city}_{voter}_pair_diff_scaler.joblib'),
                'pca_diff': joblib.load(f'{iso_model_dir}/models/{source_city}_{voter}_pair_diff_pca.joblib'),
                'iso_pair_diff': joblib.load(f'{iso_model_dir}/models/{source_city}_{voter}_pair_diff_iso_pca.joblib'),
                # Score-feature models (optional — only available if pre-training
                # pipeline has been run for this source city).
                'mlp_score_regressor': _try_load_joblib(f'{iso_model_dir}/models/{source_city}_{voter}_score_feature_mlp.joblib'),
                'scaler_residual':     _try_load_joblib(f'{iso_model_dir}/models/{source_city}_{voter}_score_feature_residual_scaler.joblib'),
                'iso_score_feature':   _try_load_joblib(f'{iso_model_dir}/models/{source_city}_{voter}_score_feature_iso.joblib'),
            }
            
            # ================================================================
            # Prepare test data (hold-out for all evaluations)
            # ================================================================
            
            print(f"  Preparing test set...")
            
            # Shuffle and split into train pool and test set
            # target_data = target_data.sample(frac=1, random_state=42).reset_index(drop=True)
            # train_pool, test_data = train_test_split(
            #     target_data,
            #     test_size=0.3,
            #     random_state=42
            # )
            train_pool, test_data = target_test_sets[target_city]
            # Full target data (train pool + test) used only for BT score estimation
            # inside the SF misalignment analysis — more pairs give more stable scores.
            all_target_data = pd.concat([train_pool, test_data]).reset_index(drop=True)
            
            print(f"    Train pool: {len(train_pool)} pairs")
            print(f"    Test set: {len(test_data)} pairs")
            
            # ================================================================
            # Baseline: Zero-shot (no fine-tuning)
            # ================================================================
            
            zero_start_time = time.time()
            
            print(f"\n  Evaluating zero-shot baseline...")
            
            zero_shot_metrics = run_zeroshot_evaluation(
                model_path=model_path,
                test_data=test_data,
                root_path=root_path,
                sub_bands=sub_bands,
                target_city=target_city,
                iso_models=iso_models,
                all_target_data=all_target_data
            )
            
            zero_end_time = time.time()
            zero_duration_sec = zero_end_time - zero_start_time
            
            # Extract these specific values because the fine-tuning loop needs
            # them to calculate the delta (improvement) later on
            zero_accuracy = zero_shot_metrics['accuracy']
            zero_bt_metrics = {'spearman': zero_shot_metrics['spearman']}
            
            # Save zero-shot results
            pair_results.append({
                'source': source_city,
                'target': target_city,
                'sample_size': 0,
                'strategy': 'zero_shot',
                'sampling': 'n/a',
                'random_seed': None,
                'duration_seconds': zero_duration_sec,
                **zero_shot_metrics
            })
            
            print(f"    Zero-shot accuracy: {zero_accuracy:.3f}, Spearman: {zero_shot_metrics['spearman']:.3f}")
            print(f"    [TIME] Zero-shot finished in {zero_duration_sec:.1f} seconds")

            # ── Pair-compatibility diagnostic + pool categorisation ───────────
            # categorize_pool_for_sampling is called exactly once here.
            # It returns both the AoA labels and SF flags needed by:
            #   (a) compute_pair_compatibility_flags — uses AoA labels only
            #   (b) the informed samplers — use both AoA labels and SF flags
            # Both consumers reuse these results; no second call is needed.
            _compat_aoa_labels, _compat_sf_flags = categorize_pool_for_sampling(
                model_path=model_path,
                train_pool=train_pool,
                root_path=root_path,
                sub_bands=sub_bands,
                iso_models=iso_models,
                all_target_data=all_target_data
            )
            compat_flags = compute_pair_compatibility_flags(
                aoa_labels=_compat_aoa_labels,
                zero_accuracy=zero_accuracy,
                sf_pair_flags=_compat_sf_flags,
                zero_spearman=zero_shot_metrics.get('spearman'),
                zero_categories_dict=zero_shot_metrics,
            )
            # Patch the already-appended zero-shot result row with the flags
            pair_results[-1].update(compat_flags)

            # Pass the categorisation results to the informed samplers.
            # No second model load needed — results from above are reused directly.
            train_pool_categories = None
            if sampling_strategy in ('informed', 'informed_t'):
                train_pool_categories = (_compat_aoa_labels, _compat_sf_flags)

            # Run garbage collection before starting the massive fine-tuning loop
            gc.collect()
            
            # ================================================================
            # Fine-tuning with different sample sizes AND strategies
            # ================================================================
            
            for sample_size in SAMPLE_SIZES:
                
                # Skip if not enough data in train pool
                if sample_size >= len(train_pool):
                    print(f"\n  Skipping sample_size={sample_size} (not enough data in train pool)")
                    continue
                
                print(f"\n{'─'*60}")
                print(f"SAMPLE SIZE: {sample_size}")
                print(f"{'─'*60}")
                
                for strategy_name, strategy_config in STRATEGIES.items():
                    # Only run the strategy selected via sys.argv[3]
                    if strategy_name != training_strategy:
                        continue
                    
                    print(f"\n  Strategy: {strategy_config['name']}")
                    print(f"  {strategy_config['description']}")
                    
                    for seed in RANDOM_SEEDS:
                        
                        print(f"    Random seed: {seed}", end=' ')
                        report_memory()
                        print()
                        
                        seed_start_time = time.time()
                        
                        # Run the isolated function
                        iteration_metrics = run_finetuning_iteration(
                            model_path=model_path,
                            strategy_config=strategy_config,
                            train_pool=train_pool,
                            test_data=test_data,
                            sample_size=sample_size,
                            seed=seed,
                            root_path=root_path,
                            sub_bands=sub_bands,
                            target_city=target_city,
                            zero_accuracy=zero_accuracy,
                            zero_bt_metrics=zero_bt_metrics,
                            iso_models=iso_models,
                            sampling_strategy=sampling_strategy,
                            train_pool_categories=train_pool_categories,
                            all_target_data=all_target_data
                        )
                        
                        seed_end_time = time.time()
                        seed_duration_sec = seed_end_time - seed_start_time
                        print(f"      [TIME] Iteration finished in {seed_duration_sec:.1f} seconds")
                        
                        # Combine loop variables with the returned metrics
                        pair_results.append({
                            'source': source_city,
                            'target': target_city,
                            'sample_size': sample_size,
                            'strategy': strategy_name,
                            'sampling': sampling_tag,
                            'random_seed': seed,
                            'duration_seconds': seed_duration_sec,
                            **iteration_metrics
                        })

                        # A single garbage collection call is now sufficient
                        gc.collect()
                        
        
        except Exception as e:
            print(f"ERROR processing {source_city}→{target_city}: {e}")
            import traceback
            traceback.print_exc()
            _save_results(pair_results, pair_output_path)   # preserve whatever completed so far
            continue
        
        # Clean up after each source-target pair
        del train_pool, test_data, iso_models
        gc.collect()
        
        pair_end_time = time.time()
        pair_duration_min = (pair_end_time - pair_start_time) / 60.0
        print(f"\n[TIME] Completed {source_city} -> {target_city} in {pair_duration_min:.2f} minutes")

        # Save this pair's results to its own file
        _save_results(pair_results, pair_output_path)
        
# ============================================================================
# Done
# ============================================================================

print("\n" + "="*70)
print("COMPLETE!")
print(f"Individual pair results saved to: {RESULTS_DIR}/")
print(f"File pattern: {exp}_{{source}}_{{target}}.csv")
print("="*70)