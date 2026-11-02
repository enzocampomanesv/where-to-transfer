import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import csv
import sqlite3
from osgeo import gdal
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from archs.base_archs import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from datetime import datetime
from time import perf_counter
import absl.logging
from sklearn.model_selection import KFold

absl.logging.set_verbosity(absl.logging.ERROR)

def retrieve_db(db_path, sql):
    con = sqlite3.connect(db_path)
    return con, pd.read_sql(sql, con = con)

def load_data(data, sub_bands, d_src, norm_data=False):
    data_copy = data.copy()
    if ((d_src == "S1") | (d_src == "wv3")):
        data_copy.loc[:, 'left'] = data_copy['left'].str.replace(r'S2', d_src, regex=True)
        data_copy.loc[:, 'right'] = data_copy['right'].str.replace(r'S2', d_src, regex=True)
    l_fp = data_copy['left'].values
    r_fp = data_copy['right'].values
    labels = data_copy['y'].values
    weights = data_copy['weight'].values
    y = np.array(labels)
    w = np.array(weights)
    imgs_l = load_and_transpose(l_fp, sub_bands, norm_data)
    imgs_r = load_and_transpose(r_fp, sub_bands, norm_data)
    # Crop the images: remove the bottom row and right column
    imgs_l = np.array([img[:20,:20,:] for img in imgs_l])  # Shape: (num_samples, 20, 20, num_bands)
    imgs_r = np.array([img[:20,:20,:] for img in imgs_r])  # Shape: (num_samples, 20, 20, num_bands)
    dataset = list(zip(imgs_l, imgs_r, y, np.array(l_fp),np.array(r_fp), w))
    return dataset
def load_and_preprocess_image(fp, sub_bands, norm_data=False):
    # Load image using GDAL
    # print(fp,end='\r')
    img_dataset = gdal.Open(fp)
    img_array = img_dataset.ReadAsArray()
    if norm_data:
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    if sub_bands is not None:
        img_array = img_array[sub_bands,:,:]
    return img_array
def load_and_transpose(fp, sub_bands, norm_data=False):
    # Load and process each image individually to handle different sizes
    processed_images = []
    for path in fp:
        img = load_and_preprocess_image(path, sub_bands, norm_data)
        # img shape is (bands, H, W)
        # Transpose to (H, W, bands)
        img_transposed = np.transpose(img, (1, 2, 0))
        # Crop to 20x20 (handles both 21x21 and 20x20 inputs)
        img_cropped = img_transposed[:20, :20, :]
        processed_images.append(img_cropped)
    
    # Now all images are (20, 20, bands), safe to create array
    return np.array(processed_images)

# def load_and_transpose(fp, sub_bands, norm_data=False):
#     d_img = np.array([load_and_preprocess_image(path, sub_bands, norm_data) for path in fp])
#     return np.transpose(d_img, (0, 2, 3, 1))

# def base_regr(in_layer, se = 0):
#     in_shape = in_layer.shape
#     out_model_regr = mVGG_feature_extractor(in_shape[1], in_shape[2], in_shape[3], se=se)(in_layer)
#     # out_model_regr = dn_feature_extractor(in_shape[1], in_shape[2], in_shape[3], se=se)(in_layer)
#     return out_model_regr
def base_regr(in_layer, se = 0):
    in_shape = in_layer.shape
    out_model_regr = mVGG_feature_extractor(in_shape[1], in_shape[2], in_shape[3], se=se)(in_layer)
    # out_model_regr = dn_feature_extractor(in_shape[1], in_shape[2], in_shape[3], se=se)(in_layer)
    return out_model_regr
def partimap_voter(euclid = 0, se = 0, n_width = 20, n_height = 20, n_channels = 10):
    input_image_l = Input(shape=(n_width, n_height, n_channels)) # Sentinel 2 left
    input_image_r = Input(shape=(n_width, n_height, n_channels)) # Sentinel 2 right
    in_shape = input_image_r.shape
    feat_extractor = mVGG_feature_extractor(in_shape[1], in_shape[2], in_shape[3], se=se)
    featEx_l = feat_extractor(input_image_l)
    featEx_r = feat_extractor(input_image_r)
    if euclid == 0:
        concat = Concatenate()([featEx_l, featEx_r])
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(concat)
        x = BatchNormalization()(x)
        # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        # x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        # x = BatchNormalization()(x)
        # Flatten, Concat, and Fully Connected Layers
        x = Flatten()(x)
        # x = Dense(256, activation='relu')(x)
        # x = Dense(256, activation='relu')(x)
        # outputs = Dense(1, activation='sigmoid')(x)
        outputs = Dense(2, activation='softmax')(x)
    else:
        def distance(vects):
            x, y = vects
            diff = x - y
            return diff
        def dist_output_shape(shapes):
            shape1, shape2 = shapes
            return (shape1[0], 1)
        # measure the similarity of the two vector outputs
        x = Lambda(distance, name="output_layer", output_shape=dist_output_shape)([featEx_l, featEx_r])
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        # outputs = Dense(1, activation='sigmoid')(x) # original sigmoid output
        outputs = Dense(2, activation='softmax')(x) # trial softmax classes output
    # Create the full model
    full_model = Model(inputs=[input_image_l, input_image_r], outputs=outputs)
    
    print(full_model.summary())
    return full_model

def reduce_lr_on_plateau(epoch, lr):
    """Reduce learning rate by half every x epochs."""
    if (epoch % 100 == 0) & epoch > 0:
        return lr / 10
    else:
        return lr
    
def main():
    # Parameters
    db_path = 'votes/NBO_voterAI.db'
    tbl_name = 'slum_image_pairs_500k_train'
    out_tbl = 'slum_500k_test_votes'
    d_src = sys.argv[1]
    se=int(sys.argv[2])
    euclid=int(sys.argv[3])
    training = int(sys.argv[4])
    
    n_epochs = 1000
    n_width = 20
    n_height = 20
    n_channels = 4
    n_batch = 128
    loss = 'huber_loss'
    lr_scheduler = LearningRateScheduler(reduce_lr_on_plateau)
    metrics=['binary_accuracy']
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
    n_channels = len(sub_bands)
    if se == 1:
        se_fn = "se"
    else:
        se_fn = ""
    if euclid == 1:
        euc_fn = "diff"
    else:
        euc_fn = ""
    # Model configuration
    model = partimap_voter(euclid=euclid,
                           se=se,
                           n_width=n_width,
                           n_height=n_height,
                           n_channels=n_channels)
    start_time = perf_counter()
    init_wt_fn = "votes/weights/NBO_voter_500k_initWeights_"+d_src+"_"+se_fn+"_"+euc_fn+"_huber.h5"
    test_pred_df = pd.DataFrame()
    if training == 1:
        # Data loading
        print("Loading...")
        sql = f"select * from " + tbl_name
        con, data = retrieve_db(db_path, sql)
        num_folds = 5
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        
        fold = 1
        with open('votes/NBO_voter_500k_'+d_src+'_'+se_fn+"_"+euc_fn+'.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for train_index, test_index in kf.split(data):
                lr = 0.02
                opt = Adam(learning_rate=lr)
                model.compile(optimizer=opt, loss=loss, metrics=metrics)
                test_fold_predictions = []
                print("Fold:", fold)
                train_fold_df, test_fold_df = data.iloc[train_index], data.iloc[test_index]

                train_df, val_df = train_test_split(train_fold_df, test_size=0.3, random_state=42)
                train_data = load_data(train_df, sub_bands, d_src)
                val_data = load_data(val_df, sub_bands, d_src)
                train_images_l, train_images_r, train_labels, t_left, t_right = zip(*train_data)
                val_images_l, val_images_r, val_labels, v_left, v_right = zip(*val_data)
                # Convert the lists to NumPy arrays for easier manipulation
                train_images_l = np.array(train_images_l)
                train_images_r = np.array(train_images_r)
                val_images_l = np.array(val_images_l)
                val_images_r = np.array(val_images_r)
                img_in_t = [train_images_l, train_images_r]
                img_in_v = [val_images_l, val_images_r]
                train_labels = np.array(train_labels)
                val_labels = np.array(val_labels)

                # Callbacks
                tensorboard_log_dir = "logs/votes/NBO_voter_500k_{}_{}_{}_f{}".format(d_src, se_fn, euc_fn, fold)
                tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir)
                checkpoint = ModelCheckpoint(
                    "trained_models/voter/NBO_voter_500k_{}_{}_{}_f{}_best.ckpt".format(d_src, se_fn, euc_fn, fold),
                    save_best_only=True, monitor='val_loss', mode='min', verbose=0
                )
                early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose =0)
                # model.load_weights("NBO/weights/NBO_pmap_initWeights_" + fusion_type + "_" + base_model+se_fn+".h5")

                print("Training...")
                # Train the model on the training set
                model.load_weights(init_wt_fn)
                model.fit(img_in_t, train_labels,\
                          validation_data=(img_in_v, val_labels),\
                          epochs=n_epochs, batch_size=n_batch,\
                          callbacks=[checkpoint, tensorboard_callback, early_stopping,lr_scheduler], verbose=1)

                # Testing
                print("Testing...")
                test_data = load_data(test_fold_df, sub_bands, d_src)
                test_images_l, test_images_r, test_labels, test_left, test_right = zip(*test_data)
                test_images_l = np.array(test_images_l)
                test_images_r = np.array(test_images_r)
                img_in_test = [test_images_l, test_images_r]
                test_labels = np.array(test_labels)
                test_score = model.evaluate(img_in_test, test_labels, verbose=0)
                test_fold_predictions = model.predict(img_in_test).tolist()
                fold_value = np.full(len(test_fold_predictions), fold)
                d_value = np.full(len(test_fold_predictions), d_src)
                # print(f"{len(fold_value)}, {len(test_left)}, {len(test_right)}, {len(test_fold_predictions)}")
                t_fold_df = pd.DataFrame({'fold': fold_value,'left': test_left , 'right': test_right, 'pred': test_fold_predictions, 'd_src': d_value})
                t_fold_df['pred'] = t_fold_df['pred'].apply(lambda x: x[0])
                test_pred_df = pd.concat([test_pred_df, t_fold_df], ignore_index=True)
                print(f"Test: {test_score[1]:.4f}")
                writer.writerow([d_src, se_fn, f"{test_score[1]:.4f}", fold])
                fold += 1
        # Save df to sqlite
        test_pred_df.to_sql(out_tbl+"_"+d_src, con = con, if_exists= 'replace')
    else:
        with open('votes/NBO_voter_500k_'+d_src+'_'+se_fn+"_"+euc_fn+'.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['bands', 'se', 'accuracy', 'fold'])
        model.save_weights(init_wt_fn)
        print("Saved " + init_wt_fn)
if __name__ == "__main__":
    main()
