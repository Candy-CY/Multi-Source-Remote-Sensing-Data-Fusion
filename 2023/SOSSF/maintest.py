
from __future__ import print_function
import os
import numpy as np
import tifffile as tiff
import pandas as pd
import Model
import cv2
from generators_SiamUnet import mybatch_generator_prediction
from utils import get_input_image_names_bi
import tensorflow as tf
import keras
import rasterio

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=str, default='0', help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--modelname', dest='modelname',type=str,  default='Model', help='choose model')
parser.add_argument('--dataset_dir', dest='dataset_dir',type=str, default='/mnt/data/xiayu/Paper2/ReMake2/Dataset', help='path')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.use_gpu

def prediction():
    model = Model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.load_weights(weights_path , by_name=True)

    print("\nExperiment name: ", experiment_name)
    print("Prediction started... ")
    print("Input image size = ", (in_rows, in_cols))
    print("Number of input spectral bands = ", num_of_channels)
    print("Batch size = ", batch_sz)

    imgs_mask_test = model.predict_generator(
        generator=mybatch_generator_prediction(test_img, in_rows, in_cols, batch_sz, max_bit),
        steps=np.ceil(len(test_img) / batch_sz))

    print("Saving predicted cloud masks on disk... \n")

    pred_dir = experiment_name + '_test'
    if not os.path.exists(os.path.join( pred_dir)):
        os.mkdir(os.path.join(pred_dir))

    for image, image_id in zip(imgs_mask_test, test_ids):
        with rasterio.open(os.path.join(TEST_FOLDER, 'MODIS_6bands', str(image_id))) as ds:
           profile = ds.profile
        img_result = (image.astype(np.float32))
        img_saveresult = img_result.transpose(2, 0, 1)
        with rasterio.open(os.path.join(pred_dir, str(image_id)), mode='w', **profile) as dst:
           dst.write(img_saveresult)



TRAIN_FOLDER = args.dataset_dir +'/DatasetTrain'
VAL_FOLDER = args.dataset_dir +'/DatasetVal'
TEST_FOLDER = args.dataset_dir +'/DatasetTest'


in_rows = 512
in_cols = 512
num_of_channels = 2
num_of_classes = 1
batch_sz = 1
max_bit = 1  # maximum gray level in landsat 8 images
experiment_name = args.modelname
weights_path = experiment_name + '.h5'


# getting input images names

test_img, test_ids = get_input_image_names_bi(TEST_FOLDER, if_train=False)


prediction()
