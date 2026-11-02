import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import csv
import sqlite3
from osgeo import gdal
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Lambda, Input, Dense, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from archs.base_archs import *
from train_voter_600k import *
from datetime import datetime
from time import perf_counter
from sklearn.model_selection import KFold
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


d_src = sys.argv[1]
city = sys.argv[2] # 3 letter city codes
base_name = sys.argv[3] # communities/planners/combined
training = int(sys.argv[4])
out_model_base_name = f"{city}_voter_{base_name}"

# Randomly sample from existing votes
def reduce_data_random(df, frac):
    sampled_df = df.sample(frac=frac)
    return sampled_df

# Bradley-Terry probability layer
def bradley_terry_probability(strengths):
    """
    Compute Bradley-Terry probability from two strength scores.
    P(left > right) = exp(strength_left) / (exp(strength_left) + exp(strength_right))
                    = 1 / (1 + exp(strength_right - strength_left))
                    = sigmoid(strength_left - strength_right)
    
    Args:
        strengths: [strength_left, strength_right] - list of two tensors
    Returns:
        probability that left > right
    """
    strength_left, strength_right = strengths
    # Compute difference and apply sigmoid for Bradley-Terry probability
    diff = strength_left - strength_right
    prob = tf.nn.sigmoid(diff)
    return prob

def create_bradley_terry_model(base_model, n_height, n_width, n_channels):
    """
    Create a Bradley-Terry model that outputs individual strengths and combines them.
    
    Args:
        base_model: The base feature extractor model
        n_height, n_width, n_channels: Image dimensions
    
    Returns:
        A model that takes two images and outputs Bradley-Terry probability
    """
    # Inputs
    input_layer_l = Input(shape=(n_height, n_width, n_channels), name='left_image')
    input_layer_r = Input(shape=(n_height, n_width, n_channels), name='right_image')
    
    # Extract features using the base model (shared weights)
    features_l = base_model(input_layer_l)
    features_r = base_model(input_layer_r)
    
    # Flatten features
    features_l_flat = Flatten()(features_l)
    features_r_flat = Flatten()(features_r)
    
    # Add a final dense layer to get strength scores (single value per image)
    strength_layer = Dense(1, activation='linear', name='strength_output')
    strength_l = strength_layer(features_l_flat)
    strength_r = strength_layer(features_r_flat)
    
    # Compute Bradley-Terry probability
    bt_prob = Lambda(bradley_terry_probability, name='bradley_terry_prob')([strength_l, strength_r])
    
    # Create the full model
    model = Model(inputs=[input_layer_l, input_layer_r], outputs=bt_prob)
    
    return model

def partimap_bradley_terry_voter(euclid, se, n_width, n_height, n_channels):
    """
    Create a Bradley-Terry voter model based on the partimap architecture.
    This creates a feature extractor and wraps it in Bradley-Terry probability computation.
    """
    # Create base feature extractor (similar to original but without final classification layer)
    input_img = Input(shape=(n_height, n_width, n_channels))
    
    in_shape = input_img.shape
    feat_extractor = mVGG_feature_extractor(in_shape[1], in_shape[2], in_shape[3], se=se)
    features = feat_extractor(input_img)
    
    # Create base feature extractor model
    base_model = Model(inputs=input_img, outputs=features, name='feature_extractor')
    
    # Wrap in Bradley-Terry model
    bt_model = create_bradley_terry_model(base_model, n_height, n_width, n_channels)
    
    return bt_model

def prep_train_test_split(data, train_index, test_index, sub_bands, d_src):
    train_fold_df, test_fold_df = data.iloc[train_index], data.iloc[test_index]
    train_df, val_df = train_test_split(train_fold_df, test_size=0.0625, random_state=42) # 75-5-20 split (75 training, 5 validation, kfold test 20)
    train_data = load_data(train_df, sub_bands, d_src)
    val_data = load_data(val_df, sub_bands, d_src)
    train_images_l, train_images_r, train_labels, t_left, t_right, t_w = zip(*train_data)
    val_images_l, val_images_r, val_labels, v_left, v_right, v = zip(*val_data)
    # Convert the lists to NumPy arrays for easier manipulation
    train_images_l = np.array(train_images_l)
    train_images_r = np.array(train_images_r)
    val_images_l = np.array(val_images_l)
    val_images_r = np.array(val_images_r)
    img_in_t = [train_images_l, train_images_r]
    img_in_v = [val_images_l, val_images_r]
    
    # For Bradley-Terry, labels are binary (0 or 1) not one-hot encoded
    # 0 means left was chosen, 1 means right was chosen
    train_labels = np.array(train_labels).astype(np.float32)
    val_labels = np.array(val_labels).astype(np.float32)
    train_weights = np.array(t_w)
    return img_in_t, train_labels, train_weights, img_in_v, val_labels, test_fold_df

# Train the AI-voter with Bradley-Terry model
def train_voter(data, build_model, d_src, frac, kf, sub_bands, n_height, n_width, n_channels, n_epochs, n_batch, init_wt_fn):
    lr_scheduler = LearningRateScheduler(reduce_lr_on_plateau)
    with open("votes/{}_{}_frac{}_bradley_terry.csv".format(out_model_base_name, d_src, frac), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bands', 'accuracy', 'fold', 'frac'])
        test_pred_df = pd.DataFrame()
        fold = 1
        # best_acc = 0
        # best_fold = 1
        models = []
        # Use binary crossentropy since we're predicting probabilities
        loss = 'huber_loss'
        metrics = ['binary_accuracy']
        reg_strength = 0.001
        
        for train_index, test_index in kf.split(data):
            lr = 0.001
            opt = Adam(learning_rate=lr)
            i = fold-1
            # A new model per fold: appending one shared object would leave every
            # entry holding the last fold's weights, so the ensemble would be one model.
            models.append(build_model())
            models[i]._name = models[i]._name + "_" + str(frac) + "_" + str(fold)
            
            # Add regularization to the model (if needed)
            for layer in models[i].layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = l2(reg_strength)
            
            test_fold_predictions = []
            print("Fold:", fold)
            img_in_t, train_labels, t_w, img_in_v, val_labels, test_fold_df = prep_train_test_split(data, train_index, test_index, sub_bands, d_src)
            
            # Callbacks
            tensorboard_log_dir = "logs/votes/{}_{}_{}_f{}_bt".format(out_model_base_name, d_src, frac, fold)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir)
            ckpt_out = "trained_models/voter/{}_{}_{}_f{}_best_bradley_terry.ckpt".format(out_model_base_name, d_src, frac, fold)
            checkpoint = ModelCheckpoint(ckpt_out,
                                        save_best_only=True,
                                        monitor=f'val_{metrics[0]}',
                                        mode='max',  # Changed to max since we want higher accuracy
                                        verbose=0)
            early_stopping = EarlyStopping(monitor=f'val_{metrics[0]}', patience=50, restore_best_weights=True, verbose=0)

            print("Training with Bradley-Terry model...")
            # Train the model on the training set
            models[i].compile(optimizer=opt, loss=loss, metrics=metrics)
            models[i].load_weights(init_wt_fn)
            models[i].fit(img_in_t, train_labels,
                          sample_weight=t_w,
                          validation_data=(img_in_v, val_labels),
                          epochs=n_epochs, batch_size=n_batch,
                          callbacks=[checkpoint, tensorboard_callback, early_stopping, lr_scheduler],
                          verbose=1)

            # Reload best checkpoint weights — early_stopping restores best weights
            # only if it triggered; explicitly reloading the checkpoint guarantees
            # the model in memory always matches the best saved checkpoint.
            models[i].load_weights(ckpt_out)

            # Testing
            print("Testing...")
            test_data = load_data(test_fold_df, sub_bands, d_src)
            test_images_l, test_images_r, test_labels, test_left, test_right, test_wt = zip(*test_data)
            test_images_l = np.array(test_images_l)
            test_images_r = np.array(test_images_r)
            img_in_test = [test_images_l, test_images_r]
            test_labels = np.array(test_labels).astype(np.float32)
            
            test_score = models[i].evaluate(img_in_test, test_labels, verbose=0)
            # Predictions are now Bradley-Terry probabilities
            test_fold_predictions = models[i].predict(img_in_test).flatten().tolist()
            fold_value = np.full(len(test_fold_predictions), fold)
            d_value = np.full(len(test_fold_predictions), d_src)
            
            t_fold_df = pd.DataFrame({'fold': fold_value, 'left': test_left, 'right': test_right, 
                                     'bt_probability': test_fold_predictions, 'd_src': d_value})
            test_pred_df = pd.concat([test_pred_df, t_fold_df], ignore_index=True)
            print(f"Test Accuracy: {test_score[1]:.4f}")
            writer.writerow([d_src, f"{test_score[1]:.4f}", fold, frac])
            fold += 1
    
        # Build ensemble from the 5 best-checkpoint fold models — no further training
        ensemble_model = build_ensemble_voter(models, n_height, n_width, n_channels, d_src, frac)
        
    return test_pred_df, ensemble_model

# Ensemble model builder for Bradley-Terry
# Combines the 5 K-fold models by averaging their Bradley-Terry probabilities.
# No further training is performed — each fold model already carries its best
# checkpoint weights (reloaded after fit()), so training the ensemble on any
# data split would introduce leakage or redundancy.
def build_ensemble_voter(models, n_height, n_width, n_channels, d_src, frac):
    # Shared input layers
    input_layer_l = Input(shape=(n_height, n_width, n_channels))
    input_layer_r = Input(shape=(n_height, n_width, n_channels))

    # Collect each fold model's Bradley-Terry probability output
    predictions = []
    for model in models:
        prediction = model([input_layer_l, input_layer_r])
        predictions.append(prediction)

    # Average the probabilities across all fold models
    averaged_prediction = tf.keras.layers.average(predictions)

    ensemble_model = Model(inputs=[input_layer_l, input_layer_r], outputs=averaged_prediction)
    ensemble_model._name = ensemble_model._name + "_ens_bt_" + str(frac)

    # Save the full model so tf.keras.models.load_model() works downstream.
    # The .ckpt path is intentionally preserved to match the filename expected
    # by the transfer learning script.
    save_path = "trained_models/voter/{}_ensembleVoter_{}_{}_bradley_terry.ckpt".format(
        out_model_base_name, d_src, frac)
    ensemble_model.save(save_path)
    print(f"Ensemble model saved to: {save_path}")

    return ensemble_model

# Test the AI-voter with Bradley-Terry probabilities
def test_voter(model, pred_exist, test_set, d_src, frac):
    sub_bands, _, _, _ = get_sub_bands(d_src) 
    test_set['left_id'] = test_set['left'].apply(extract_tile_id).astype(np.int64)
    test_set['right_id'] = test_set['right'].apply(extract_tile_id).astype(np.int64)
    pred_exist['left_id'] = pred_exist['left'].apply(extract_tile_id).astype(np.int64)
    pred_exist['right_id'] = pred_exist['right'].apply(extract_tile_id).astype(np.int64)
    pred_exist['y_pred'] = pred_exist.apply(choose_left_right_bt, axis=1)
    
    merged_df = test_set.merge(pred_exist,
                               on=['left_id', 'right_id'],
                               how='outer', indicator=True)
    for_testing_df = merged_df[merged_df['_merge'] == 'left_only']
    for_testing_df = for_testing_df.rename(columns={'left_x': 'left', 'right_x': 'right'})
    
    test_data = load_data(for_testing_df, sub_bands, d_src)
    test_images_l, test_images_r, test_labels, test_left, test_right, test_w = zip(*test_data)
    test_images_l = np.array(test_images_l)
    test_images_r = np.array(test_images_r)
    img_in_test = [test_images_l, test_images_r]
    test_labels = np.array(test_labels)
    
    # Ensemble predict - returns Bradley-Terry probabilities
    test_fold_predictions = model.predict(img_in_test).flatten().tolist()
    
    t_l = pd.Series(test_left).apply(extract_tile_id).astype(np.int64)
    t_r = pd.Series(test_right).apply(extract_tile_id).astype(np.int64)
    t_fold_df = pd.DataFrame({'left_id': t_l, 'right_id': t_r, 'bt_probability': test_fold_predictions})
    
    final_test_df = pd.concat([t_fold_df[['left_id', 'right_id', 'bt_probability']],
                               pred_exist[['left_id', 'right_id', 'bt_probability']]],
                              ignore_index=True)
    final_test_df['y'] = final_test_df.apply(choose_left_right_bt, axis=1)
    return final_test_df

def load_test_image(d_src, tile_id):
    path = f"votes/2024/{city}/{d_src}"
    sub_bands, _, _, _ = get_sub_bands(d_src)
    img = load_and_preprocess_image(path + f"{city}_{d_src}_{tile_id}.tif", sub_bands)
    img = np.transpose(img, (1, 2, 0))
    img = tf.expand_dims(img, 0)
    return img

def extract_tile_id(filename):
    parts = filename.split('_')
    tile_id = parts[-1].split('.')[0]  # Remove the file extension
    return tile_id

def choose_left_right_bt(row):
    """
    Convert Bradley-Terry probability to binary choice.
    bt_probability > 0.5 means right is preferred (returns 1)
    bt_probability < 0.5 means left is preferred (returns 0)
    """
    if row.bt_probability < 0.5:
        return 0
    else:
        return 1

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

def import_votes(db_path, tbl_name, n_votes=-1):
    sql = f"select * from " + tbl_name
    con, all_data = retrieve_db(db_path, sql)
    if n_votes != -1:
        all_data = all_data.sample(n=n_votes)
    return all_data
    
def main():
    # Parameters
    db_path = f'votes/db/{city}/{city}_voterAI.db'
    
    se = 1
    euclid = 0
    n_epochs = 1000
    n_width = 20
    n_height = 20
    n_channels = 2
    n_batch = 64
    sub_bands, n_width, n_height, n_channels = get_sub_bands(d_src)
        
    # Model configuration - using Bradley-Terry model
    def build_model():
        return partimap_bradley_terry_voter(euclid=euclid,
                                            se=se,
                                            n_width=n_width,
                                            n_height=n_height,
                                            n_channels=n_channels)
    model = build_model()
    
    data_frac = [1]
    # Data loading
    print("Loading data...")
    
    if "planners" in base_name:
        pl_tbl_name = f'{city}_planners_votes'
        pl_data = import_votes(db_path, pl_tbl_name)
        all_data = pl_data.copy()
    elif "communities" in base_name:
        cm_tbl_name = f'{city}_communities_votes'
        cm_data = import_votes(db_path, cm_tbl_name)
        all_data = cm_data.copy()
    elif "combined" in base_name:
        cb_tbl_name = f'{city}_combined_votes'
        cb_data = import_votes(db_path, cb_tbl_name)
        all_data = cb_data.copy()
    
    for frac in data_frac:
        init_wt_fn = f"votes/weights/{out_model_base_name}_initWeights_" + d_src + "_frac" + str(frac) + "_bt.h5"
        if training == 1:
            num_folds = 5
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            fold = 1
            data = reduce_data_random(all_data, frac)
            print(f"Training Bradley-Terry model on {len(data)} out of {len(all_data)} samples")
            
            test_pred_df, ens_model = train_voter(data, build_model, d_src, frac, kf, sub_bands, n_height, n_width, n_channels, n_epochs, n_batch, init_wt_fn)
            
            # Save results with Bradley-Terry probabilities
            print("\nBradley-Terry probabilities preview:")
            print(test_pred_df.head())
            
        else:
            print(model.summary())
            model.save_weights(init_wt_fn)
            print("Saved initial weights: " + init_wt_fn)

if __name__ == "__main__":
    main()