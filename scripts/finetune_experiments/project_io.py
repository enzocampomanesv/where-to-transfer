import numpy as np
from osgeo import gdal

def load_and_preprocess_image(fp, sub_bands, norm_data=False):
    img_dataset = gdal.Open(fp)
    img_array = img_dataset.ReadAsArray()
    img_dataset = None
    if norm_data:
        img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    if sub_bands is not None:
        img_array = img_array[sub_bands,:,:]
    return np.transpose(img_array[:, :20, :20], (1, 2, 0))

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