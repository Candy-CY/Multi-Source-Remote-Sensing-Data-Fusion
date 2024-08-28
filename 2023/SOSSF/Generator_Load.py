import random
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import tifffile as tiff


def mybatch_generator_train(zip_list, img_rows, img_cols, batch_size, shuffle=True, max_possible_input_value=1):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:
        if shuffle:
            random.shuffle(zip_list)

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image1_list = []
        image2_list = []
        mask_list = []
        for file, mask in batch_files:

            image1 = tiff.imread(file[0])
            image2 = tiff.imread(file[1])
            mask = tiff.imread(mask)


            image1 = np.nan_to_num(image1)
            image2= np.nan_to_num(image2)
            mask = np.nan_to_num(mask)


            image1_list.append(image1)
            image2_list.append(image2)
            mask_list.append(mask)

        counter += 1
        image1_list = np.array(image1_list)
        image2_list = np.array(image2_list)
        mask_list = np.array(mask_list)
        yield ([image1_list,image2_list], mask_list)

        if counter == number_of_batches:
            if shuffle:
                random.shuffle(zip_list)
            counter = 0

def mybatch_generator_validation(zip_list, img_rows, img_cols, batch_size, shuffle=False, max_possible_input_value=1):
    number_of_batches = np.ceil(len(zip_list) / batch_size)
    if shuffle:
        random.shuffle(zip_list)
    counter = 0

    while True:

        batch_files = zip_list[batch_size * counter:batch_size * (counter + 1)]
        image1_list = []
        image2_list = []
        mask_list = []
        for file, mask in batch_files:

            image1 = tiff.imread(file[0])
            image2 = tiff.imread(file[1])
            mask = tiff.imread(mask)


            image1 = np.nan_to_num(image1)
            image2= np.nan_to_num(image2)
            mask = np.nan_to_num(mask)


            image1_list.append(image1)
            image2_list.append(image2)
            mask_list.append(mask)

        counter += 1
        image1_list = np.array(image1_list)
        image2_list = np.array(image2_list)
        mask_list = np.array(mask_list)
        yield ([image1_list,image2_list], mask_list)

        if counter == number_of_batches:
            counter = 0

def mybatch_generator_prediction(tstfiles, img_rows, img_cols, batch_size, max_possible_input_value=1):
    number_of_batches = np.ceil(len(tstfiles) / batch_size)
    counter = 0

    while True:

        beg = batch_size * counter
        end = batch_size * (counter + 1)
        batch_files = tstfiles[beg:end]
        image1_list = []
        image2_list = []


        for file in batch_files:
            image1 = tiff.imread(file[0])
            image2 = tiff.imread(file[1])

            image1 = np.nan_to_num(image1)
            image2 = np.nan_to_num(image2)


            image1_list.append(image1)
            image2_list.append(image2)

        counter += 1
        # print('counter = ', counter)
        image1_list = np.array(image1_list)
        image2_list = np.array(image2_list)

        yield ([image1_list, image2_list])

        if counter == number_of_batches:
            counter = 0