from __future__ import print_function
from sklearn.model_selection import train_test_split
import os
import numpy as np
from utils import ADAMLearningRateTracker
import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from Generator_Load import mybatch_generator_train, mybatch_generator_validation
import pandas as pd
from utils import get_input_image_names_bi
from keras.utils.vis_utils import plot_model
import keras
from keras import losses
from keras import backend as K

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=str, default='0', help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--modelname', dest='modelname',type=str,  default='Model', help='choose model')
parser.add_argument('--dataset_dir', dest='dataset_dir',type=str, default='/mnt/data/xiayu/Paper2/ReMake2/Dataset', help='path')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES']=args.use_gpu

def train():
    model = Model.model_arch(input_rows=in_rows,
                                       input_cols=in_cols,
                                       num_of_channels=num_of_channels,
                                       num_of_classes=num_of_classes)
    model.compile(optimizer=Adam(lr=starting_learning_rate), loss=losses.mean_absolute_error, metrics=[keras.metrics.mae])
    model.summary()

    model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', mode='min', save_best_only=True)
    lr_reducer = ReduceLROnPlateau(factor=decay_factor, cooldown=0, patience=patience, min_lr=end_learning_rate, verbose=1)
    csv_logger = CSVLogger(experiment_name + '_log_1.log')
    
    train_img_split = train_img
    train_msk_split =train_msk
    val_img_split = val_img
    val_msk_split = val_msk
    if train_resume:
        model.load_weights(weights_path)
        print("\nTraining resumed...")
    else:
        print("\nTraining started from scratch... ")

    print("Experiment name: ", experiment_name)
    print("Input image size: ", (in_rows, in_cols))
    print("Number of input spectral bands: ", num_of_channels)
    print("Learning rate: ", starting_learning_rate)
    print("Batch size: ", batch_sz, "\n")
    print(mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit))

    model.fit_generator(
        generator=mybatch_generator_train(list(zip(train_img_split, train_msk_split)), in_rows, in_cols, batch_sz, max_bit),
        steps_per_epoch=np.ceil(len(train_img_split) / batch_sz), epochs=max_num_epochs, verbose=1,
        validation_data=mybatch_generator_validation(list(zip(val_img_split, val_msk_split)), in_rows, in_cols, 1, max_bit),
        validation_steps=np.ceil(len(val_img_split) / 1),
        callbacks=[model_checkpoint, lr_reducer, ADAMLearningRateTracker(end_learning_rate),csv_logger])


TRAIN_FOLDER = args.dataset_dir +'/DatasetTrain'
VAL_FOLDER = args.dataset_dir +'/DatasetVal'
TEST_FOLDER = args.dataset_dir +'/DatasetTest'

in_rows = 512
in_cols = 512
num_of_channels = 8
num_of_classes = 1
starting_learning_rate = 1e-4
end_learning_rate = 1e-8
max_num_epochs = 2000  # just a huge number. The actual training should not be limited by this value
val_ratio = 0.2
patience = 15
decay_factor = 0.7
batch_sz = 4
max_bit = 1  # maximum gray level in landsat 8 images
experiment_name = args.modelname
weights_path =  experiment_name + '.h5'
train_resume = False

# getting input images names

train_img, train_msk = get_input_image_names_bi(TRAIN_FOLDER, if_train=True)
val_img, val_msk = get_input_image_names_bi(VAL_FOLDER, if_train=True)
train()
