"""
finetune_core.py  (v1 -- random sampling path)
==============================================

Re-finetune a saved source ensembleVoter on a target city and return the trained
MODEL plus the gate metrics (accuracy, BT spearman) on the study's held-out test
set. Ported in logic verbatim from ``ACC_transferLearn_BT.py`` so a re-finetuned
variant matches the study; the gate (metrics within the original three-seed band)
is the acceptance test for that fidelity.

v1 SCOPE: sampling='random' only. The informed path (categorize_pool_for_sampling
+ iso-forests + _informed_sample_unified) is deferred to v2.

Ported verbatim from the source:
  * ImageSequence / create_dataset / evaluate_model
  * the layerwise variable partitioning + three-optimizer custom train_step
  * fixed constants (LAYERWISE_DECAY=0.1, HEAD_LR_MULTIPLIER=5.0, FINE_TUNE_EPOCHS=50,
    BATCH_SIZE=32, PATIENCE=10, BASE_LR=1e-4)

Added for this pipeline:
  * tf.random.set_seed(42) before fit (shuffle left at default True, to keep the
    training trajectory drawn from the same distribution as the original band)
  * return of the trained model + selected sample ids + gate metrics
  * preprocess_fn injected into the Sequence (so it uses the project's real
    load_and_preprocess_image; no module-global dependency)

NOTE ON KERAS VERSION: the custom train_step uses ``model.compiled_metrics`` and
attribute-assignment of ``model.train_step`` -- Keras 2 idioms, matching the infra
that produced the study. Under Keras 3 these differ; run this on the same
TF/Keras that trained the originals.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from finetune_data import reconstruct_split, random_sample

# Fixed constants (verbatim from the source script)
LAYERWISE_DECAY = 0.1
HEAD_LR_MULTIPLIER = 5.0
FINE_TUNE_EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 10
BASE_LR = 1e-4


# ===========================================================================
# Data pipeline (ported verbatim; preprocess_fn injected)
# ===========================================================================
class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, left_paths, right_paths, labels, sub_bands, batch_size,
                 preprocess_fn, root_path=""):
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.labels = np.array(labels, dtype=np.int32)
        self.sub_bands = sub_bands
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.root_path = root_path
        self.n = len(labels)

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def _resolve(self, p):
        # Option 2: prepend root_path so relative DB paths resolve regardless of CWD.
        # os.path.join leaves already-absolute paths unchanged.
        import os
        return os.path.join(self.root_path, p) if self.root_path else p

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n)
        batch_l, batch_r = [], []
        for i in range(start_idx, end_idx):
            batch_l.append(self.preprocess_fn(self._resolve(self.left_paths[i]), self.sub_bands))
            batch_r.append(self.preprocess_fn(self._resolve(self.right_paths[i]), self.sub_bands))
        batch_l = np.array(batch_l, dtype=np.float32)
        batch_r = np.array(batch_r, dtype=np.float32)
        batch_labels = self.labels[start_idx:end_idx]
        return (batch_l, batch_r), batch_labels

    def on_epoch_end(self):
        pass


def create_dataset(pairs_df, root_path, sub_bands, preprocess_fn, batch_size=BATCH_SIZE):
    left_paths, right_paths, labels = [], [], []
    for _, row in pairs_df.iterrows():
        lp, rp = row["left"], row["right"]
        if isinstance(lp, bytes):
            lp = lp.decode("utf-8")
        if isinstance(rp, bytes):
            rp = rp.decode("utf-8")
        left_paths.append(str(lp))
        right_paths.append(str(rp))
        labels.append(int(row["y"]))
    return ImageSequence(left_paths, right_paths, labels, sub_bands, batch_size,
                         preprocess_fn, root_path=root_path)


def evaluate_model(model, test_dataset):
    all_preds, all_probs = [], []
    for i in range(len(test_dataset)):
        (images_l, images_r), labels = test_dataset[i]
        predictions = model.predict_on_batch([images_l, images_r])
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


# ===========================================================================
# Layerwise setup -- partitioning extracted so it is testable on a toy model
# ===========================================================================
def _find_layer(root, name):
    for layer in root.layers:
        if layer.name == name:
            return layer
        if hasattr(layer, "layers"):
            found = _find_layer(layer, name)
            if found is not None:
                return found
    return None


def _find_all_by_type(root, layer_type, _seen=None):
    if _seen is None:
        _seen = set()
    results = []
    for layer in root.layers:
        if id(layer) in _seen:
            continue
        _seen.add(id(layer))
        if isinstance(layer, layer_type):
            results.append(layer)
        if hasattr(layer, "layers"):
            results.extend(_find_all_by_type(layer, layer_type, _seen))
    return results


def partition_variables(model):
    """Partition trainable variables into (block1, block2, block3, head), excluding BN.

    Verbatim port of the source's layerwise variable grouping (transfer script
    lines ~1649-1748). Extracted as a pure function so the fidelity-critical
    partitioning can be unit-tested on a toy two-tower model without training.

    Returns dict with the four variable lists + counts + the strength layer name.
    """
    for layer in model.layers:
        layer.trainable = True
        if hasattr(layer, "layers"):
            for sub in layer.layers:
                sub.trainable = True
                if hasattr(sub, "layers"):
                    for subsub in sub.layers:
                        subsub.trainable = True

    all_vars = model.trainable_variables
    # BN detection: identify BN variables by BatchNormalization-LAYER membership,
    # not by the 'batch_normalization' name substring the source used. On Keras 2
    # both agree (var names contain 'batch_normalization'); on Keras 3 variable
    # names are bare ('gamma'/'beta') so the name check finds NONE and BN would
    # silently train. Structural membership is version-robust and excludes exactly
    # the same variables on the study's Keras 2 -> behaviourally identical there.
    bn_layers = _find_all_by_type(model, tf.keras.layers.BatchNormalization)
    bn_var_ids = {id(v) for l in bn_layers for v in l.variables}
    if bn_layers and not bn_var_ids:
        raise RuntimeError("BatchNormalization layers found but no BN variables "
                           "resolved -- cannot guarantee BN freezing.")
    is_bn = lambda v: id(v) in bn_var_ids

    strength_layer = _find_layer(model, "strength_output")
    if strength_layer is None:
        all_dense = _find_all_by_type(model, tf.keras.layers.Dense)
        dense_1 = [l for l in all_dense
                   if l.units == 1 and getattr(l.activation, "__name__", "") == "linear"]
        if dense_1:
            strength_layer = dense_1[0]
        else:
            raise RuntimeError(
                "Cannot find strength output layer. "
                f"Top-level layers: {[l.name for l in model.layers]}"
            )
    head_var_ids = {id(v) for v in strength_layer.variables}
    vars_head = [v for v in all_vars if id(v) in head_var_ids and not is_bn(v)]

    all_conv = _find_all_by_type(model, tf.keras.layers.Conv2D)
    b1_ids = {id(v) for l in all_conv[0:2] for v in l.variables}
    b2_ids = {id(v) for l in all_conv[2:4] for v in l.variables}
    b3_ids = {id(v) for l in all_conv[4:7] for v in l.variables}

    def _block_range(anchor_ids):
        idxs = [i for i, v in enumerate(all_vars) if id(v) in anchor_ids]
        return (min(idxs), max(idxs)) if idxs else (0, 0)

    b1_range = _block_range(b1_ids)
    b2_range = _block_range(b2_ids)
    b3_range = _block_range(b3_ids)
    head_range = _block_range(head_var_ids)

    mid12 = (b1_range[1] + b2_range[0]) // 2
    mid23 = (b2_range[1] + b3_range[0]) // 2
    # (mid3h computed in source but unused; omitted)

    vars_block1, vars_block2, vars_block3 = [], [], []
    for i, v in enumerate(all_vars):
        if is_bn(v) or id(v) in head_var_ids:
            continue
        if i <= mid12:
            vars_block1.append(v)
        elif i <= mid23:
            vars_block2.append(v)
        else:
            vars_block3.append(v)

    n_bn = sum(1 for v in all_vars if is_bn(v))
    return {
        "block1": vars_block1, "block2": vars_block2, "block3": vars_block3,
        "head": vars_head, "n_bn": n_bn, "n_total": len(all_vars),
        "strength_layer_name": strength_layer.name,
    }


def setup_layerwise_training(model, base_lr=BASE_LR):
    """Configure the model for layerwise finetuning (verbatim from the source).

    Sets up three block optimizers + head optimizer, a custom GradientTape
    train_step, and compiles. Returns the partition info for logging.
    """
    part = partition_variables(model)
    vars_block1, vars_block2 = part["block1"], part["block2"]
    vars_block3, vars_head = part["block3"], part["head"]

    _base = base_lr
    opt_block1 = tf.keras.optimizers.Adam(learning_rate=_base * (LAYERWISE_DECAY ** 2))
    opt_block2 = tf.keras.optimizers.Adam(learning_rate=_base * LAYERWISE_DECAY)
    opt_block3 = tf.keras.optimizers.Adam(learning_rate=_base * (LAYERWISE_DECAY ** 0.5))
    opt_head = tf.keras.optimizers.Adam(learning_rate=_base * HEAD_LR_MULTIPLIER)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def layerwise_train_step(data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        grad_map = {id(v): g for v, g in zip(model.trainable_variables, grads)
                    if g is not None}

        def _apply(opt, var_list):
            gv = [(grad_map[id(v)], v) for v in var_list if id(v) in grad_map]
            if gv:
                opt.apply_gradients(gv)

        _apply(opt_block1, vars_block1)
        _apply(opt_block2, vars_block2)
        _apply(opt_block3, vars_block3)
        _apply(opt_head, vars_head)

        model.compiled_metrics.update_state(y, y_pred)
        results = {m.name: m.result() for m in model.metrics}
        results["loss"] = loss
        return results

    model.train_step = layerwise_train_step
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=_base),
        loss=loss_fn, metrics=["accuracy"],
    )
    return part


# ===========================================================================
# Top-level: re-finetune one variant (random path) -> model + gate metrics
# ===========================================================================
def _extract_id(path):
    import re, os
    m = re.search(r"(\d+)\.", os.path.basename(str(path)))
    return int(m.group(1)) if m else None


def teardown_training_state(model):
    """Detach the custom train_step and optimizer so the model is a clean inference model.

    The layerwise setup assigns ``model.train_step`` (a tf.function closing over the
    four optimizers) -- this is what makes SavedModel serialization fail (it captures
    optimizer resource tensors). Removing the instance attribute reverts to the class
    method; predict() is unaffected.
    """
    try:
        del model.train_step
    except Exception:
        pass
    try:
        model.optimizer = None
    except Exception:
        pass


def save_finetuned_weights(model, ckpt_path: str) -> str:
    """Save WEIGHTS ONLY (robust across Keras 2/3); returns the actual path written.

    Weights-only sidesteps SavedModel serialization of the custom train_step. The
    architecture is not saved -- reload by building the source model and calling
    load_weights (every consumer here has the source architecture). Uses the
    '.weights.h5' extension, which Keras 3 requires and Keras 2 accepts as HDF5.
    """
    teardown_training_state(model)
    base = ckpt_path
    for ext in (".ckpt", ".weights.h5", ".h5", ".keras"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    out = base + ".weights.h5"
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    model.save_weights(out)
    return out


def finetune_variant(
    model_path: str,
    target_city: str,
    root_path: str,
    sub_bands,
    *,
    preprocess_fn: Callable,
    sampling: str = "random",
    sample_size: int = 5000,
    seed: int = 42,
    voter: str = "communities",
    base_lr: float = BASE_LR,
    batch_size: int = BATCH_SIZE,
    epochs: int = FINE_TUNE_EPOCHS,
    patience: int = PATIENCE,
    bt_recovery_fn: Optional[Callable] = None,
    iso_model_dir: Optional[str] = None,
    source_code: Optional[str] = None,
    progress: bool = True,
) -> Dict[str, object]:
    """Re-finetune (random or informed-t1) and return trained model + gate metrics.

    bt_recovery_fn defaults to evalBT.evaluate_model_bt_recovery so the gate's
    spearman is computed EXACTLY as the study's CSVs were.

    For sampling='informed'/'informed_t1', iso_model_dir and source_code are
    required (to load the source's iso-forest artifacts); temperature=1, adaptive
    is not used (t1 soft-mix).
    """
    informed = sampling in ("informed", "informed_t1")
    if sampling not in ("random", "informed", "informed_t1"):
        raise NotImplementedError(f"sampling={sampling!r} not supported.")
    if informed and (iso_model_dir is None or source_code is None):
        raise ValueError("informed sampling requires iso_model_dir and source_code.")
    if bt_recovery_fn is None:
        from BT_utils.evalBT import evaluate_model_bt_recovery as bt_recovery_fn  # type: ignore

    # 1. reconstruct the exact study split (TF-free, deterministic at seed 42)
    train_pool, test_data = reconstruct_split(target_city, voter, root_path)

    # 2. zero-shot baseline on test_data (fresh load; predict only)
    tf.keras.backend.clear_session()
    zs_model = tf.keras.models.load_model(model_path)
    zs_test_ds = create_dataset(test_data, root_path, sub_bands, preprocess_fn, batch_size)
    zs_preds, zs_probs = evaluate_model(zs_model, zs_test_ds)
    y_true = np.array(test_data["y"], dtype=int).flatten()
    zero_accuracy = float((y_true == np.array(zs_preds, dtype=int).flatten()).mean())
    zero_bt = bt_recovery_fn(test_data, zs_probs)
    del zs_model, zs_test_ds

    # 3. fresh model for finetuning + pinned seed
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    model = tf.keras.models.load_model(model_path)

    # 4. sample (random or informed) + layerwise setup
    if informed:
        from finetune_informed import (load_iso_models, categorize_pool_for_sampling,
                                        informed_sample_unified)
        from finetune_data import all_target_data as _atd
        from BT_utils.evalBT import estimate_bt_scores_from_predictions as _bt_est
        iso = load_iso_models(source_code, iso_model_dir, voter=voter)
        atd = _atd(train_pool, test_data)
        aoa_labels, sf_pair_flags = categorize_pool_for_sampling(
            model, train_pool, root_path, sub_bands, iso, preprocess_fn,
            evaluate_fn=evaluate_model, bt_estimator=_bt_est, all_target_data=atd)
        finetune_data, sample_breakdown = informed_sample_unified(
            train_pool, aoa_labels, sf_pair_flags, sample_size, seed, temperature=1.0)
    else:
        finetune_data = random_sample(train_pool, sample_size, seed)
        sample_breakdown = None

    part = setup_layerwise_training(model, base_lr)
    if progress:
        print(f"      [Layerwise] block1={len(part['block1'])} block2={len(part['block2'])} "
              f"block3={len(part['block3'])} head={len(part['head'])} bn={part['n_bn']}")

    # 5. fit (shuffle default True; seed pinned above)
    train_ds = create_dataset(finetune_data, root_path, sub_bands, preprocess_fn, batch_size)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=patience, restore_best_weights=True)
    model.fit(train_ds, epochs=epochs, callbacks=[early_stop], verbose=0)

    # 6. gate metrics on test_data
    test_ds = create_dataset(test_data, root_path, sub_bands, preprocess_fn, batch_size)
    test_preds, test_probs = evaluate_model(model, test_ds)
    accuracy = float((y_true == np.array(test_preds, dtype=int).flatten()).mean())
    bt_metrics = bt_recovery_fn(test_data, test_probs)
    spearman = bt_metrics.get("spearman")

    selected_ids = [
        (_extract_id(l), _extract_id(r))
        for l, r in zip(finetune_data["left"], finetune_data["right"])
    ]

    return {
        "model": model,
        "metrics": {
            "accuracy": accuracy,
            "spearman": spearman,
            "improvement": accuracy - zero_accuracy,
            "sp_improvement": (spearman - zero_bt.get("spearman"))
            if spearman is not None and zero_bt.get("spearman") is not None else None,
        },
        "zero": {"accuracy": zero_accuracy, "spearman": zero_bt.get("spearman")},
        "selected_ids": selected_ids,
        "n_finetune": int(len(finetune_data)),
        "sampling": sampling, "sample_size": sample_size, "seed": seed,
    }
