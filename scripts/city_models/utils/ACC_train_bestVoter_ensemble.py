import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import csv
import sqlite3
from osgeo import gdal
from train_voter_600k import *
from datetime import datetime
from time import perf_counter
from sklearn.model_selection import KFold
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.regularizers import l2

d_src = sys.argv[1]
city = sys.argv[2] # 3 letter city codes
base_name = sys.argv[3] # communities/planners/combined
training = int(sys.argv[4])
out_model_base_name = f"{city}_voter_{base_name}"

# Randomly sample from existing votes
def reduce_data_random(df, frac):
    sampled_df = df.sample(frac=frac)
    return sampled_df
def prep_train_test_split(data, train_index, test_index, sub_bands, d_src):
    num_classes = 2
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
    
    # Convert labels to one-hot encoded format
    train_labels = tf.one_hot(train_labels, depth=num_classes).numpy()  # One-hot encode train labels
    val_labels = tf.one_hot(val_labels, depth=num_classes).numpy()
    train_weights = np.array(t_w)
    return img_in_t, train_labels, train_weights, img_in_v, val_labels, test_fold_df

# Train the AI-voter, df is training set (reduced or not), does 5Kfold-CV
# returns model
def train_voter(data, model, d_src, frac, kf, sub_bands, n_height, n_width, n_channels, n_epochs, n_batch, init_wt_fn):
    num_classes = 2
    lr_scheduler = LearningRateScheduler(reduce_lr_on_plateau)
    with open("votes/perf_metrics/{}/{}_{}_frac{}_dcai.csv".format(city, out_model_base_name,d_src, frac), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['bands', 'accuracy', 'fold', 'frac'])
        test_pred_df = pd.DataFrame()
        fold = 1
        best_acc = 0
        best_fold = 1
        models = []
        # loss = 'huber_loss'
        loss = 'categorical_crossentropy'
        metrics=['binary_accuracy']
        reg_strength = 0.001
        for train_index, test_index in kf.split(data):
            lr = 0.001
            opt = Adam(learning_rate=lr)
            i = fold-1
            models.append(model)
            models[i]._name = models[i]._name + "_" + str(frac) + "_" + str(fold)
            # Add regularization to the model (if needed)
            for layer in models[i].layers:
                if hasattr(layer, 'kernel_regularizer'):
                    layer.kernel_regularizer = l2(reg_strength)
            
            test_fold_predictions = []
            print("Fold:", fold)
            img_in_t, train_labels, t_w, img_in_v, val_labels, test_fold_df = prep_train_test_split(data, train_index, test_index, sub_bands, d_src)
            # Callbacks
            tensorboard_log_dir = "logs/votes/{}/{}_{}_{}_f{}".format(city, out_model_base_name,d_src, frac, fold)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir)
            ckpt_out = "trained_models/voter/{}_{}_{}_f{}_best_dcai.ckpt".format(out_model_base_name,d_src, frac, fold)
            checkpoint = ModelCheckpoint(ckpt_out,
                                        save_best_only=True,
                                        monitor=f'val_{metrics[0]}',
                                        mode='min', verbose=0)
            early_stopping = EarlyStopping(monitor=f'val_{metrics[0]}', patience=50, restore_best_weights=True, verbose =0)

            print("Training...")
            # Train the model on the training set
            models[i].compile(optimizer=opt, loss=loss, metrics=metrics)
            models[i].load_weights(init_wt_fn)
            models[i].fit(img_in_t, train_labels,\
                          sample_weight = t_w,\
                          validation_data=(img_in_v, val_labels),\
                          epochs=n_epochs, batch_size=n_batch,\
                          callbacks=[checkpoint, tensorboard_callback, early_stopping,lr_scheduler],
                          verbose=1)

            # Testing
            print("Testing...")
            test_data = load_data(test_fold_df, sub_bands, d_src)
            test_images_l, test_images_r, test_labels, test_left, test_right, test_wt = zip(*test_data)
            test_images_l = np.array(test_images_l)
            test_images_r = np.array(test_images_r)
            img_in_test = [test_images_l, test_images_r]
            # Convert labels to one-hot encoded format
            test_labels = tf.one_hot(test_labels, depth=num_classes).numpy()
            test_score = models[i].evaluate(img_in_test, test_labels, verbose=0)
            test_fold_predictions = models[i].predict(img_in_test).tolist()
            fold_value = np.full(len(test_fold_predictions), fold)
            d_value = np.full(len(test_fold_predictions), d_src)
            # print(f"{len(fold_value)}, {len(test_left)}, {len(test_right)}, {len(test_fold_predictions)}")
            t_fold_df = pd.DataFrame({'fold': fold_value,'left': test_left , 'right': test_right, 'output': test_fold_predictions, 'd_src': d_value})
            t_fold_df['output'] = t_fold_df['output'].apply(lambda x: x[0])
            test_pred_df = pd.concat([test_pred_df, t_fold_df], ignore_index=True)
            print(f"Test: {test_score[1]:.4f}")
            writer.writerow([d_src, f"{test_score[1]:.4f}", fold, frac])
            # if test_score[1] > best_acc:
            #     best_ckpt = ckpt_out
            #     best_fold = fold
            #     best_acc = test_score[1]
            fold+=1
    
        # Ensemble model training
        ens_lr = 0.01
        ens_opt = Adam(ens_lr)
        ensemble_model = train_ensemble_voter(data, models, n_height, n_width, n_channels, n_batch, d_src, frac, loss, ens_opt, metrics)
        # writer.writerow([d_src, f"{ens_test_score[1]:.4f}", 'ens', frac])
        # print(test_pred_df.head())
    return test_pred_df, ensemble_model

# Ensemble model training
def train_ensemble_voter(data, models, n_height, n_width, n_channels, n_batch, d_src, frac, loss, opt, metrics):
    num_classes = 2
    sub_bands, _, _, _ = get_sub_bands(d_src) 
    # Data prep
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
    train_data = load_data(train_df, sub_bands, d_src)
    val_data = load_data(val_df, sub_bands, d_src)
    train_images_l, train_images_r, train_labels, t_left, t_right, t_w = zip(*train_data)
    val_images_l, val_images_r, val_labels, v_left, v_right, v_w = zip(*val_data)
    # Convert the lists to NumPy arrays for easier manipulation
    train_images_l = np.array(train_images_l)
    train_images_r = np.array(train_images_r)
    val_images_l = np.array(val_images_l)
    val_images_r = np.array(val_images_r)
    img_in_t = [train_images_l, train_images_r]
    img_in_v = [val_images_l, val_images_r]
    
    train_labels = tf.one_hot(train_labels, depth=num_classes).numpy()  # One-hot encode train labels
    val_labels = tf.one_hot(val_labels, depth=num_classes).numpy()
    train_weights = np.array(t_w)
    # Model prep
    input_layer_l = Input(shape=(n_height, n_width, n_channels))
    input_layer_r = Input(shape=(n_height, n_width, n_channels))
    predictions = []
    for model in models:
        # Pass the input image through model i
        prediction = model([input_layer_l, input_layer_r])
        predictions.append(prediction)
    averaged_prediction = tf.keras.layers.average(predictions)
    
    ensemble_model = Model(inputs=[input_layer_l, input_layer_r], outputs=averaged_prediction)
    ensemble_model._name = ensemble_model._name + "_ens_" + str(frac)
    ensemble_model.compile(loss=loss, optimizer=opt, metrics=metrics)
    checkpoint = ModelCheckpoint("trained_models/voter/{}_ensembleVoter_{}_{}_dcai.ckpt".format(out_model_base_name,d_src, frac),
                                 save_best_only=True,
                                 monitor=f'val_{metrics[0]}',
                                 mode='min', verbose=0)
    early_stopping = EarlyStopping(monitor=f'val_{metrics[0]}', patience=25, restore_best_weights=True, verbose=0)

    # Model training
    ensemble_model.fit(img_in_t, train_labels,\
                      validation_data=(img_in_v, val_labels),\
                      sample_weight = train_weights,\
                      epochs=100, batch_size=n_batch,\
                      callbacks=[checkpoint, early_stopping], verbose=1)
    # test_score = ensemble_model.evaluate(img_in_ts, test_labels, verbose=0)
    # print(f"TEST: BinAcc: {test_score[1]:.4f}")
    return ensemble_model

# Test the AI-voter, if reduced dataset, take KFold-CV output + predict on remaining test
def test_voter(model, pred_exist, test_set, d_src, frac):
    sub_bands, _, _, _ = get_sub_bands(d_src) 
    test_set['left_id'] = test_set['left'].apply(extract_tile_id).astype(np.int64)
    test_set['right_id'] = test_set['right'].apply(extract_tile_id).astype(np.int64)
    pred_exist['left_id'] = pred_exist['left'].apply(extract_tile_id).astype(np.int64)
    pred_exist['right_id'] = pred_exist['right'].apply(extract_tile_id).astype(np.int64)
    pred_exist['y_pred'] = pred_exist.apply(choose_left_right, axis=1)
    merged_df = test_set.merge(pred_exist,
                               on=['left_id', 'right_id'],
                               how='outer', indicator=True)
    for_testing_df = merged_df[merged_df['_merge'] == 'left_only']
    for_testing_df = for_testing_df.rename(columns={'left_x': 'left', 'right_x': 'right'})
    # print(for_testing_df.columns)
    test_data = load_data(for_testing_df, sub_bands, d_src)
    test_images_l, test_images_r, test_labels, test_left, test_right = zip(*test_data)
    test_images_l = np.array(test_images_l)
    test_images_r = np.array(test_images_r)
    img_in_test = [test_images_l, test_images_r]
    test_labels = np.array(test_labels)
    # test_score = model.evaluate(img_in_test, test_labels, verbose=0)
    
    # Ensemble predict
    test_fold_predictions = model.predict(img_in_test).tolist()
    # frac_value = np.full(len(test_fold_predictions), frac)
    # d_value = np.full(len(test_fold_predictions), d_src)
    t_l = pd.Series(test_left).apply(extract_tile_id).astype(np.int64)
    t_r = pd.Series(test_right).apply(extract_tile_id).astype(np.int64)
    t_fold_df = pd.DataFrame({'left_id': t_l , 'right_id': t_r, 'output': test_fold_predictions})
    t_fold_df['output'] = t_fold_df['output'].apply(lambda x: x[0])
    
    final_test_df = pd.concat([t_fold_df[['left_id','right_id', 'output']],
                               pred_exist[['left_id','right_id', 'output']]],
                              ignore_index=True)
    final_test_df['y'] = final_test_df.apply(choose_left_right, axis=1)
    return final_test_df

def load_test_image(d_src, tile_id):
    path = f"votes/2024/{city}/{d_src}"
    sub_bands, _, _, _ = get_sub_bands(d_src)
    img = load_and_preprocess_image(path+f"{city}_{d_src}_{tile_id}.tif", sub_bands)
    img = np.transpose(img, (1,2,0))
    img = tf.expand_dims(img, 0)
    return img
def extract_tile_id(filename):
    parts = filename.split('_')
    tile_id = parts[-1].split('.')[0] # Remove the file extension
    return tile_id
def choose_left_right(row):
    if row.output < 0.5:
        return 0
    else:
        return 1

def get_sub_bands(d_src):
    n_width = 10
    n_height = 10
    if d_src == "RGBN":
        sub_bands = [0,1,2,6]
    elif d_src == "RGB":
        sub_bands = [0,1,2]
    elif d_src == "RGN":
        sub_bands = [0,1,6]
    elif d_src == "all":
        sub_bands = [0,1,2,3,4,5,6,7,8,9]
    elif d_src == "S1":
        sub_bands = [0,1]
    elif d_src == "wv3":
        sub_bands = [0,1,2,3]
        n_width = 50
        n_height = 50
    elif d_src == "S1_200m":
        sub_bands = [0,1]
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
    # out_tbl = 'GHA_communities_votes'
    
    # Parameters
    db_path = f'votes/db/{city}/{city}_voterAI.db'
    pl_tbl_name = f'{city}_planners_votes'
    cm_tbl_name = f'{city}_communities_votes'
    cb_tbl_name = f'{city}_combined_votes'
    se=1
    euclid=0
    n_epochs = 1000
    n_width = 20
    n_height = 20
    n_channels = 2
    n_batch = 64
    sub_bands, n_width, n_height, n_channels = get_sub_bands(d_src)
        
    # Model configuration
    model = partimap_voter(euclid=euclid,
                           se=se,
                           n_width=n_width,
                           n_height=n_height,
                           n_channels=n_channels)
    
    # data_frac = np.arange(0.1,1.1,0.1)
    data_frac = [1]
    # Data loading
    print("Loading...")
    
    
    if "planners" in base_name:
        pl_data = import_votes(db_path, pl_tbl_name)
        all_data = pl_data.copy()
    elif "communities" in base_name:
        cm_data = import_votes(db_path, cm_tbl_name)
        all_data = cm_data.copy()
    elif "combined" in base_name:
        cb_data = import_votes(db_path, cb_tbl_name)
        all_data = cb_data.copy()
    
    for frac in data_frac:
        # frac = round(frac, 1)
        init_wt_fn = f"votes/weights/{out_model_base_name}_initWeights_"+d_src+"_frac"+str(frac)+".h5"
        if training == 1:
            num_folds = 5
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            fold = 1
            data = reduce_data_random(all_data, frac)
            print(f"Training on {len(data)} out of {len(all_data)} samples")
            
            test_pred_df, ens_model = train_voter(data, model, d_src, frac, kf, sub_bands, n_height, n_width, n_channels, n_epochs, n_batch, init_wt_fn)
            # test_pred_df = test_voter(ens_model, test_pred_df, all_data, d_src, frac)
            # print(test_pred_df.head())
            # out_tbl_final = (out_tbl+"_"+d_src+"_ensemble_f"+str(frac)).replace('.', '_')
            # print(f"Saved {out_tbl_final}")
            # test_pred_df.to_sql(out_tbl_final, con = con, if_exists= 'replace')
        else:
            model.save_weights(init_wt_fn)
            print("Saved " + init_wt_fn)

if __name__ == "__main__":
    main()
