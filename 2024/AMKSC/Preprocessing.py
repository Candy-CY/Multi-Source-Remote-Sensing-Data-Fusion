# coding: utf-8

# EMP-->KNN-->MajoriryFilter-->(select training samples again)
#  -->KNN-->MajorityFilter-->... repeat this process
from __future__ import print_function

from collections.abc import Iterable

import numpy as np

from rolling_window import rolling_window as rw
import spectral as spy

# import tensorflow as tf
# from sklearn.feature_extraction import image

CLASS_MAP_COLOR = {
    '0': [255, 255, 255], '1': [50, 106, 179], '2': [83, 171, 72], '3': [137, 186, 67], '4': [66, 132, 91],
    '5': [60, 131, 69], '6': [144, 82, 54], '7': [105, 188, 200], '8': [254, 0, 250], '9': [199, 176, 201], '10': [218, 51, 44],
    '11': [119, 35, 36], '12': [87, 88, 90], '13': [224, 219, 84], '14': [217, 142, 52], '15': [84, 48, 126],
    '16': [227, 119, 91], '17': [157, 87, 150]
}
CLASS_MAP_COLOR_16 = [[82, 171, 72], [136, 186, 68], [66, 132, 92], [58, 132, 69], [144, 82, 54], [105, 188, 200],
                     [255, 255, 255], [199, 176, 201],
                     [218, 51, 44], [119, 35, 35], [87, 88, 90], [224, 220, 84], [217, 142, 52], [84, 48, 126],
                     [227, 119, 91], [157, 87, 150]]

CLASS_MAP_COLOR_B = [[252, 0, 3], [0, 123, 3], [250, 213, 2], [5, 253, 251], [254, 0, 250], [5, 250, 5], [4, 2, 252],
                     [1, 1, 1], [145, 30, 180], [0, 128, 128], [0, 130, 200], [245, 130, 49]]

CLASS_MAP_COLOR_8 = [[56,108,53], [228,55,38], [230,240,85], [170,228,49], [147,75,48], [0,204,0], [174,239,235], [86,132,193]]


class Processor:
    def __init__(self):
        pass

    def prepare_data(self, img_path, gt_path=None):
        if img_path[-3:] == 'mat':
            import scipy.io as sio
            img_mat = sio.loadmat(img_path)
            img_keys = img_mat.keys()
            img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']

            if gt_path is not None:
                gt_mat = sio.loadmat(gt_path)
                gt_keys = gt_mat.keys()
                gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
                return img_mat.get(img_key[0]).astype('float32'), gt_mat.get(gt_key[0]).astype('int8')
            else:
                return img_mat.get(img_key[0]).astype('float32')
        else:
            import spectral as spy
            img = spy.open_image(img_path).load()
            gt = spy.open_image(gt_path)
            # a = spy.principal_components()
            # a.transform()
            img = img.astype(np.float32)
            return img, gt.read_band(0)

    def get_correct(self, img, gt):
        """
        :param img: 3D arr
        :param gt: 2D arr
        :return: covert arr  [n_samples,n_bands]
        """
        gt_1D = gt.reshape(-1)
        index = gt_1D.nonzero()
        gt_correct = gt_1D[index]
        img_2D = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
        img_correct = img_2D[index]
        return img_correct, gt_correct

    def get_tr_tx_index(self, y, test_size=0.9):
        from sklearn.model_selection import train_test_split
        train_index, test_index, y_train_, y_test_ = \
            train_test_split(np.arange(0, y.shape[0]), y, test_size=test_size)
        return train_index, test_index

    def divide_img_blocks(self, img, gt, block_size=(5, 5)):
        """
        split image into a*b blocks, the edge filled with its mirror
        :param img:
        :param gt:
        :param block_size; tuple of size, it must be odd and >=3
        :return: correct image blocks
        """
        # TODO: padding edge with mirror
        w_1, w_2 = int((block_size[0] - 1) / 2), int((block_size[1] - 1) / 2)
        img_padding = np.pad(img, ((w_1, w_2),
                                   (w_1, w_2), (0, 0)), 'reflect')
        gt_padding = np.pad(gt, ((w_1, w_2),
                                 (w_1, w_2)), 'reflect')
        img_blocks = rw(img_padding, block_size, axes=(1, 0))  # divide data into 5x5 blocks
        gt_blocks = rw(gt_padding, block_size, axes=(1, 0))
        i_1, i_2 = int((block_size[0] - 1) / 2), int((block_size[0] - 1) / 2)
        nonzero_index = gt_blocks[:, :, i_1, i_2].nonzero()
        img_blocks_nonzero = img_blocks[nonzero_index]
        gt_blocks_nonzero = (gt_blocks[:, :, i_1, i_2])[nonzero_index]
        return img_blocks_nonzero, gt_blocks_nonzero

    def get_HSI_patches(self, x, gt, ksize, stride=(1, 1), padding='reflect', indix=False):
        """
        extract HSI spectral-spatial
        :param x: 3-D HSI (n_row, n_clm, n_band)
        :param gt: 2-D ground truth
        :param ksize: must be odd numbers, i.e. (3, 3) (7,7) ...
        :param stride:
        :param padding: padding mode: constant, reflect (default), etc.
        :return: (n_sample, ksize1, ksize2, n_band)
        """
        # # padding with boundary pixels
        import tensorflow as tf
        new_height = np.ceil(x.shape[0] / stride[0])
        new_width = np.ceil(x.shape[1] / stride[1])
        pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
        pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
        pad_top = int(pad_needed_height / 2)
        pad_down = int(pad_needed_height - pad_top)
        pad_left = int(pad_needed_width / 2)
        pad_right = int(pad_needed_width - pad_left)
        x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
        gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
        n_row, n_clm, n_band = x.shape
        x = np.reshape(x, (1, n_row, n_clm, n_band))
        y = np.reshape(gt, (1, n_row, n_clm, 1))
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        ksizes_ = (1, ksize[0], ksize[1], 1)
        strides_ = (1, stride[0], stride[1], 1)
        with tf.Session() as sess:
            x_patches = tf.extract_image_patches(x, ksizes_, strides_, rates=(1, 1, 1, 1), padding='VALID').eval()
            y_patches = tf.extract_image_patches(y, ksizes_, strides_, rates=(1, 1, 1, 1), padding='VALID').eval()
            sess.close()
        x_patches = np.reshape(x_patches, (-1, x_patches.shape[-1]))
        x_patches = np.reshape(x_patches, (-1, ksize[0], ksize[1], n_band))

        y_patches = np.reshape(y_patches, (-1, y_patches.shape[-1]))
        y_patches = np.reshape(y_patches, (-1, ksize[0], ksize[1], 1))

        i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[0] - 1) // 2)
        y_center_label = np.reshape(y_patches[:, i_1, i_2, :], -1)
        nonzero_index = np.nonzero(y_center_label)
        x_patches_nonzero = x_patches[nonzero_index]
        y_patches_nonzero = y_center_label[nonzero_index]
        if indix is True:
            return x_patches_nonzero, y_patches_nonzero, nonzero_index
        return x_patches_nonzero, y_patches_nonzero

    def get_HSI_patches_rw(self, x, gt, ksize, stride=(1, 1), padding='reflect', is_indix=False, is_labeled=True):
        """
        extract HSI spectral-spatial
        :param x: 3-D HSI (n_row, n_clm, n_band)
        :param gt: 2-D ground truth
        :param ksize: must be odd numbers, i.e. (3, 3) (7,7) ...
        :param stride:
        :param padding: padding mode: constant, reflect (default), etc.
        :return: (n_sample, ksize1, ksize2, n_band)
        """
        # # padding with boundary pixels
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)
        n_row_init, n_col_init, _ = x.shape
        new_height = np.ceil(x.shape[0] / stride[0])
        new_width = np.ceil(x.shape[1] / stride[1])
        pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
        pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]
        pad_top = int(pad_needed_height / 2)
        pad_down = int(pad_needed_height - pad_top)
        pad_left = int(pad_needed_width / 2)
        pad_right = int(pad_needed_width - pad_left)
        x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
        gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)
        n_row, n_clm, n_band = x.shape
        x = np.reshape(x, (n_row, n_clm, n_band))
        y = np.reshape(gt, (n_row, n_clm))
        ksizes_ = (ksize[0], ksize[1])
        x_patches = rw(x, ksizes_, axes=(1, 0))  # divide data into 5x5 blocks
        y_patches = rw(y, ksizes_, axes=(1, 0))
        i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[0] - 1) // 2)
        if not is_labeled:
            x_patches = x_patches.reshape((n_row_init * n_col_init, n_band, ksize[0], ksize[1]))
            y_patches = y_patches[:, :, i_1, i_2].reshape(-1)
        else:
            nonzero_index = y_patches[:, :, i_1, i_2].nonzero()
            x_patches = x_patches[nonzero_index]
            y_patches = (y_patches[:, :, i_1, i_2])[nonzero_index]
        x_patches = np.transpose(x_patches, [0, 2, 3, 1])
        if is_indix:
            return x_patches, y_patches, nonzero_index
        return x_patches, y_patches

    def split_tr_tx(self, X, y, test_size=0.4):
        """
        X_train, X_test, y_train, y_test
        :param X:
        :param y:
        :param test_size:
        :return:
        """
        from sklearn.cross_validation import train_test_split
        return train_test_split(X, y, test_size=test_size)

    def split_each_class(self, X, y, each_train_size=10):
        X_tr, y_tr, X_ts, y_ts = [], [], [], []
        for c in np.unique(y):
            y_index = np.nonzero(y == c)[0]
            np.random.shuffle(y_index)
            cho, non_cho = np.split(y_index, [each_train_size, ])
            X_tr.append(X[cho])
            y_tr.append(y[cho])
            X_ts.append(X[non_cho])
            y_ts.append(y[non_cho])
        X_tr, X_ts, y_tr, y_ts = np.asarray(X_tr), np.asarray(X_ts), np.asarray(y_tr), np.asarray(y_ts)
        return X_tr.reshape(X_tr.shape[0] * X_tr.shape[1], X.shape[1]), \
               X_ts.reshape(X_ts.shape[0] * X_ts.shape[1], X.shape[1]), \
               y_tr.flatten(), y_ts.flatten()

    def stratified_train_test_index(self, y, train_size, seed=None):
        """
        :param y: labels
        :param train_size: int, absolute number for each classes; float [0., 1.], percentage of each classes;
               or list/tuple of training size of each class
        :return:
        """
        from collections import Iterable
        np.random.seed(seed)
        train_idx, test_idx = [], []
        # # select sample according to given size
        for i, c in enumerate(np.unique(y)):
            idx = np.nonzero(y == c)[0]
            np.random.shuffle(idx)
            num = np.sum(y == c)
            if isinstance(train_size, Iterable):
                train_size_ = train_size[i]
                if train_size_ >= num:
                    raise Exception(
                        'given size {:d} should not be greater than real size {:d}'.format(train_size[i], train_size_))
            elif 0. < train_size < 1.:
                train_size_ = int(np.ceil(train_size * num))
            else:
                if train_size >= num * 0.9:  # np.ceil(num * 0.5):  # # num_labeled in this class should be less than 50%
                    train_size_ = 15  # int(np.ceil(0.5 * num))
                else:
                    train_size_ = np.copy(train_size)
            train_idx += idx[:train_size_].tolist()
            test_idx += idx[train_size_:].tolist()
        train_idx = np.asarray(train_idx).reshape(-1)
        test_idx = np.asarray(test_idx).reshape(-1)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        return train_idx, test_idx

    def save_experiment(self, y_pre, y_test, file_neme=None, parameters=None):
        """
        save classification results and experiment parameters into files for k-folds cross validation.
        :param y_pre:
        :param y_test:
        :param parameters:
        :return:
        """
        import os
        home = os.getcwd() + '/experiments'
        if not os.path.exists(home):
            os.makedirs(home)
        if parameters == None:
            parameters = [None]
        if file_neme == None:
            file_neme = home + '/scores.npz'
        else:
            file_neme = home + '/' + file_neme + '.npz'

        '''save results and scores into a numpy file'''
        ca, oa, aa, kappa = [], [], [], []
        if np.array(y_pre).shape.__len__() > 1:  # that means test data tested k times
            for y in y_pre:
                ca_, oa_, aa_, kappa_ = self.score(y_test, y)
                ca.append(ca_), oa.append(oa_), aa.append(aa_), kappa.append(kappa_)
        else:
            ca, oa, aa, kappa = self.score(y_test, y_pre)
        np.savez(file_neme, y_test=y_test, y_pre=y_pre, CA=np.array(ca), OA=np.array(oa), AA=aa, Kappa=kappa,
                 param=parameters)
        print('the experiments have been saved in experiments/scores.npz')

    # def get_train_test_indexes(self, train_size, gt):
    #     """
    #
    #     :param train_size:
    #     :param gt:
    #     :return:
    #     """
    #     gt_1D = gt.reshape(-1)
    #     samples_correct = gt_1D[gt_1D.nonzero()]
    #     n_samples = samples_correct.shape[0]  # the num of available samples
    #     classes = {}
    #     for i in np.unique(samples_correct):
    #         classes[i] = len(np.nonzero(samples_correct == i)[0])
    #     if train_size >= min(classes.values()):
    #             train_size = min(classes.values())
    #     train_indexes = np.empty((0))
    #     test_indexes = np.empty((0))
    #     for key in classes:
    #         size_ci = classes[key]
    #         index_ci = np.nonzero(gt_1D == key)[0]  # 1 dim: (row,col=None)
    #         index_train__ = np.empty(0)
    #         if train_size > 0 and train_size < 1.:
    #             # slip data as percentage for each classes
    #             index_train__ = np.random.choice(index_ci, int(size_ci * train_size), replace=False)
    #         else:
    #             # slip data as form of fixed numbers
    #             index_train__ = np.random.choice(index_ci, int(train_size), replace=False)
    #         index_test__ = np.setdiff1d(index_ci,index_train__)
    #         train_indexes = np.append(train_indexes,index_train__)
    #         test_indexes = np.append(test_indexes,index_test__)
    #     return train_indexes.astype(np.int64),test_indexes.astype(np.int64)

    def majority_filter(self, classes_map, selems):
        """
        :param classes_map: 2 dim image
        :param selems: elements: [disk(1),square(2)...]
        :return:
        """
        from skimage.filters.rank import modal
        # from skimage.morphology import disk,square
        classes_map__ = classes_map.astype(np.uint16)  # convert dtype to uint16
        out = classes_map__
        for selem in selems:
            out = modal(classes_map__, selem)
            classes_map__ = out
        return out.astype(np.int8)

    def score(self, y_test, y_predicted):
        """
        calculate the accuracy and other criterion according to predicted results
        :param y_test:
        :param y_predicted:
        :return: ca, oa, aa, kappa
        """
        from sklearn.metrics import accuracy_score
        '''overall accuracy'''
        oa = accuracy_score(y_test, y_predicted)
        '''average accuracy for each classes'''
        n_classes = max([np.unique(y_test).__len__(), np.unique(y_predicted).__len__()])
        ca = []
        for c in np.unique(y_test):
            y_c = y_test[np.nonzero(y_test == c)]  # find indices of each classes
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        aa = ca.mean()

        '''kappa'''
        kappa = self.kappa(y_test, y_predicted)
        return ca, oa, aa, kappa

    def result2gt(self, y_predicted, test_indexes, gt):
        """

        :param y_predicted:
        :param test_indexes: indexes got from ground truth
        :param gt: 2-dim img
        :return:
        """
        n_row, n_col = gt.shape
        gt_1D = gt.reshape((n_row * n_col))
        gt_1D[test_indexes] = y_predicted
        return gt_1D.reshape(n_row, n_col)

    def extended_morphological_profile(self, components, disk_radius):
        """

        :param components:
        :param disk_radius:
        :return:2-dim emp
        """
        rows, cols, bands = components.shape
        n = disk_radius.__len__()
        import numpy as np
        emp = np.zeros((rows * cols, bands * (2 * n + 1)))
        from skimage.morphology import opening, closing, disk
        for band in range(bands):
            position = band * (n * 2 + 1) + n
            emp_ = np.zeros((rows, cols, 2 * n + 1))
            emp_[:, :, n] = components[:, :, band]
            i = 1
            for r in disk_radius:
                closed = closing(components[:, :, band], selem=disk(r))
                opened = opening(components[:, :, band], selem=disk(r))
                emp_[:, :, n - i] = closed
                emp_[:, :, n + i] = opened
                i += 1
            emp[:, position - n:position + n + 1] = emp_.reshape((rows * cols, 2 * n + 1))
        return emp.reshape(rows, cols, bands * (2 * n + 1))

    def texture_feature(self, components, theta_arr=None, frequency_arr=None):
        """
        extract the texture features
        :param components:
        :param theta_arr:
        :param frequency_arr:
        :return:
        """
        if theta_arr == None:
            theta_arr = np.arange(0, 8) * np.pi / 4  # 8 orientations
        if frequency_arr == None:
            frequency_arr = np.pi / (2 ** np.arange(1, 5))  # 4 frequency

        from skimage.filters import gabor
        results = []
        for img in components.transpose():
            for theta in theta_arr:
                for fre in frequency_arr:
                    filt_real, filt_imag = gabor(img, frequency=fre, theta=theta)
                    results.append(filt_real)
        return np.array(results).transpose()

    def pca_transform(self, n_components, samples):
        """

        :param n_components:
        :param samples: [nb_samples, bands]/or [n_row, n_column, n_bands]
        :return:
        """
        HSI_or_not = samples.shape.__len__() == 3  # denotes HSI data
        n_row, n_column, n_bands = 0, 0, 0
        if HSI_or_not:
            n_row, n_column, n_bands = samples.shape
            samples = samples.reshape((n_row * n_column, n_bands))
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        trans_samples = pca.fit_transform(samples)
        if HSI_or_not:
            return trans_samples.reshape((n_row, n_column, n_components))
        return trans_samples

    def normlize_HSI(self, img):
        from sklearn.preprocessing import normalize
        n_row, n_column, n_bands = img.shape
        norm_img = normalize(img.reshape(n_row * n_column, n_bands))
        return norm_img.reshape(n_row, n_column, n_bands)

    def each_class_OA(self, y_test, y_predicted):
        """
        get each OA for all classes respectively
        :param y_test:
        :param y_predicted:
        :return:{}
        """
        classes = np.unique(y_test)
        results = []
        for c in classes:
            y_c = y_test[np.nonzero(y_test == c)]  # find indices of each classes
            y_c_p = y_predicted[np.nonzero(y_test == c)]
            acurracy = self.score(y_c, y_c_p)
            results.append(acurracy)
        return np.array(results)

    def kappa(self, y_test, y_predicted):
        from sklearn.metrics import cohen_kappa_score
        return round(cohen_kappa_score(y_test, y_predicted), 3)

    def color_legend(self, color_map, label):
        """

        :param color_map: 1-n color map in range 0-255
        :param label: label list
        :return:
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        size = len(label)
        patchs = []
        m = 255.  # float(color_map.max())
        color_map_ = (color_map / m)[1:]
        for i in range(0, size):
            patchs.append(mpatches.Patch(color=color_map_[i], label=label[i]))
        # plt.legend(handles=patchs)
        return patchs

    def get_tr_ts_index_num(self, y, n_labeled=10):
        import random
        classes = np.unique(y)
        X_train_index, X_test_index = np.empty(0, dtype='int8'), np.empty(0, dtype='int8')
        for c in classes:
            index_c = np.nonzero(y == c)[0]
            random.shuffle(index_c)
            X_train_index = np.append(X_train_index, index_c[:n_labeled])
            X_test_index = np.append(X_test_index, index_c[n_labeled:])
        return X_train_index, X_test_index

    def save_res_4kfolds_cv(self, y_tests, y_pres, file_name=None, verbose=False):
        """
        save experiment results for k-folds cross validation
        :param y_pres: predicted labels, k*Ntest
        :param y_tests: true labels, k*Ntest
        :param file_name:
        :return:
        """
        ca, oa, aa, kappa = [], [], [], []
        for y_t, y_p in zip(y_tests, y_pres):
            ca_, oa_, aa_, kappa_ = self.score(y_t, y_p)
            ca.append(np.asarray(ca_)), oa.append(np.asarray(oa_)), aa.append(np.asarray(aa_)),
            kappa.append(np.asarray(kappa_))
        ca = np.asarray(ca) * 100
        oa = np.asarray(oa) * 100
        aa = np.asarray(aa) * 100
        kappa = np.asarray(kappa)
        ca_mean, ca_std = np.round(ca.mean(axis=0), 2), np.round(ca.std(axis=0), 2)
        oa_mean, oa_std = np.round(oa.mean(), 2), np.round(oa.std(), 2)
        aa_mean, aa_std = np.round(aa.mean(), 2), np.round(aa.std(), 2)
        kappa_mean, kappa_std = np.round(kappa.mean(), 4), np.round(kappa.std(), 4)
        if file_name is not None:
            file_name = 'scores.npz'
            np.savez(file_name, y_test=y_tests, y_pre=y_pres,
                     ca_mean=ca_mean, ca_std=ca_std,
                     oa_mean=oa_mean, oa_std=oa_std,
                     aa_mean=aa_mean, aa_std=aa_std,
                     kappa_mean=kappa_mean, kappa_std=kappa_std)
            print('the experiments have been saved in ', file_name)

        if verbose is True:
            print('---------------------------------------------')
            print('ca\t\t', '\toa\t\t', '\taa\t\t', '\tkappa\t\t')
            for mean, std in zip(ca_mean, ca_std):
                print('{:.2f}+-{:.2f}'.format(mean, std))
            print('---------------------------------------------')
            print('{:.2f}+-{:.2f}'.format(oa_mean, oa_std))
            print('{:.2f}+-{:.2f}'.format(aa_mean, aa_std))
            print('{:.4f}+-{:.4f}'.format(kappa_mean, kappa_std))

        # return ca, oa, aa, kappa
        return np.asarray([ca_mean, ca_std]), np.asarray([aa_mean, aa_std]), \
               np.asarray([oa_mean, oa_std]), np.asarray([kappa_mean, kappa_std])

    # def view_clz_map(self, gt, y_index, y_predicted, save_path=None, show_error=False):
    #     """
    #     view HSI classification results
    #     :param gt:
    #     :param y_index: index of excluding 0th classes
    #     :param y_predicted:
    #     :param show_error:
    #     :return:
    #     """
    #     n_row, n_column = gt.shape
    #     gt_1d = gt.reshape(-1).copy()
    #     nonzero_index = gt_1d.nonzero()
    #     gt_corrected = gt_1d[nonzero_index]
    #     if show_error:
    #         t = y_predicted.copy()
    #         correct_index = np.nonzero(y_predicted == gt_corrected[y_index])
    #         t[correct_index] = 0  # leave error
    #         gt_corrected[:] = 0
    #         gt_corrected[y_index] = t
    #         gt_1d[nonzero_index] = t
    #     else:
    #         gt_corrected[y_index] = y_predicted
    #         gt_1d[nonzero_index] = gt_corrected
    #     gt_map = gt_1d.reshape((n_row, n_column)).astype('uint8')
    #     spy.imshow(classes=gt_map)
    #     if save_path != None:
    #         spy.save_rgb(save_path, gt_map, colors=spy.spy_colors)
    #         print('the figure is saved in ', save_path)

    def split_source_target(self, X, y, split_attribute_index, split_threshold, save_name=None):
        """
        split source/target domain data for transfer learning according to attribute
        :param X:
        :param y:
        :param split_attribute_index:
        :param split_threshold: split condition. e.g if 1.2 those X[:,index] >= 1.2 are split into source
        :param save_name:
        :return:
        """
        source_index = np.nonzero(X[:, split_attribute_index] >= split_threshold)
        target_index = np.nonzero(X[:, split_attribute_index] < split_threshold)
        X_source = X[source_index]
        X_target = X[target_index]
        y_source = y[source_index].astype('int')
        y_target = y[target_index].astype('int')
        if save_name is not None:
            np.savez(save_name, X_source=X_source, X_target=X_target, y_source=y_source, y_target=y_target)
        return X_source, X_target, y_source, y_target

    def results_to_cvs(self, res_file_name, save_name):
        import csv
        dt = np.load(res_file_name)
        ca_mean = np.round(dt['CA'].mean(axis=0) * 100, 2)
        ca_std = np.round(dt['CA'].std(axis=0), 2)
        oa_mean = np.round(dt['OA'].mean() * 100, 2)
        oa_std = np.round(dt['OA'].std(axis=0), 2)
        aa_mean = np.round(dt['AA'].mean() * 100, 2)
        aa_std = np.round(dt['AA'].std(axis=0), 2)
        kappa_mean = np.round(dt['Kappa'].mean(), 3)
        kappa_std = np.round(dt['Kappa'].std(axis=0), 2)
        with open(save_name, 'wb') as f:
            writer = csv.writer(f)
            for i in zip(ca_mean, ca_std):
                writer.writerow(i)
            writer.writerow([oa_mean, oa_std])
            writer.writerow([aa_mean, aa_std])
            writer.writerow([kappa_mean, kappa_std])

    def view_clz_map_spyversion4single_img(self, gt, y_test_index, y_predicted, save_path=None, show_error=False,
                                           show_axis=False):
        """
        view HSI classification results
        :param gt:
        :param y_test_index: test index of excluding 0th classes
        :param y_predicted:
        :param show_error:
        :return:
        """
        n_row, n_column = gt.shape
        gt_1d = gt.reshape(-1).copy()
        nonzero_index = gt_1d.nonzero()
        gt_corrected = gt_1d[nonzero_index]
        if show_error:
            t = y_predicted.copy()
            correct_index = np.nonzero(y_predicted == gt_corrected[y_test_index])
            t[correct_index] = 0  # leave error
            gt_corrected[:] = 0
            gt_corrected[y_test_index] = t
            gt_1d[nonzero_index] = t
        else:
            gt_corrected[y_test_index] = y_predicted
            gt_1d[nonzero_index] = gt_corrected
        gt_map = gt_1d.reshape((n_row, n_column)).astype('uint8')
        spy.imshow(classes=gt_map)
        if save_path != None:
            import matplotlib.pyplot as plt
            spy.save_rgb('temp.png', gt_map, colors=spy.spy_colors)
            if show_axis:
                plt.savefig(save_path, format='eps', bbox_inches='tight')
            else:
                plt.axis('off')
                plt.savefig(save_path, format='eps', bbox_inches='tight')
            # self.classification_map(gt_map, gt, 24, save_path)
            print('the figure is saved in ', save_path)

    def classification_map(self, map, groundTruth, dpi, savePath):
        import matplotlib.pyplot as plt
        fig = plt.figure(frameon=False)
        fig.set_size_inches(groundTruth.shape[1] * 2.0 / dpi, groundTruth.shape[0] * 2.0 / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.add_axes(ax)

        ax.imshow(map, aspect='normal')
        plt.imshow()
        fig.savefig(savePath, dpi=dpi, format='eps')
        return 0

    def view_clz_map_mlpversion(self, test_index, results, sub_indexes, labels, save_name=None):
        """ visualize image with 2 rows and 3 columns with the color legend for knn classification
            --------
            Usage:
                res = [gt, y_pre_spectral, y_pre_shape, y_pre_texture, y_pre_stack, y_pre_kernel]
                sub_index = [331, 332, 333, 334, 335, 336, 313]
                labels = ['(a) groundtruth', r'(b) $kNN_{spectral}$', r'(c) $kNN_{shape}$', r'(d) $kNN_{texture}$',
                r'(e) $kNN_{stack}$', r'(f) $kNN_{multi}$']
                view_clz_map_mlpversion(tx_index, res, sub_index, labels, save_name='./experiments/paviaU_class_map.eps')
        """
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
        import copy
        n_res = results.__len__()
        gt = copy.deepcopy(results[0])
        n_row, n_column = gt.shape
        gt_1d = gt.reshape(-1).copy()
        nonzero_index = gt_1d.nonzero()
        for i in range(n_res):
            if i == 0:
                gt_map = gt
            else:
                gt_corrected = copy.deepcopy(gt_1d[nonzero_index])
                gt_corrected[test_index] = results[i]
                gt_1d_temp = copy.deepcopy(gt.reshape(-1))
                gt_1d_temp[nonzero_index] = gt_corrected
                gt_map = gt_1d_temp.reshape((n_row, n_column)).astype('uint8')
            axe = plt.subplot(sub_indexes[i])
            im = axe.imshow(gt_map, cmap='jet')
            axe.set_title(labels[i], fontdict={'fontsize': 10})
            axe.axis('off')
        values = np.unique(gt.ravel())
        # get the colors of the values, according to the
        # colormap used by imshow
        colors = [im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color
        patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=values[i])) for i in range(len(values))]
        # put those patched as legend-handles into the legend
        axe_legend = plt.subplot(sub_indexes[-1])
        axe_legend.legend(handles=patches, loc=10, ncol=6)
        axe_legend.axis('off')

        # save image
        plt.show()
        plt.savefig(save_name, format='eps', dpi=1000)
        print('the figure is saved in ', save_name)

    def show_class_map(self, y_pre, y_indx, gt, show=True, save=False):
        """
        :param y_pre:
        :param y_indx: predicted pixel index in the ground truth
        :param gt:
        :param save:
        :return:
        """
        import copy
        import matplotlib.pyplot as plt
        gt = gt.astype('int16')
        gt_pre = copy.deepcopy(gt)
        gt_pre_flatten = gt_pre.reshape(-1)
        gt_pre_flatten[y_indx] = y_pre
        gt_pre_2d = np.reshape(gt_pre_flatten, gt.shape)
        fig, ax = plt.subplots()
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # cmap = ListedColormap(np.array(spy_colors) / 255.)
        # cmap = (np.array(self.class_colors) / 255.) #[500:1000, :]
        # gt_pre_2d[np.where(gt_pre_2d == 0)] = 50
        ax.imshow(gt_pre_2d, cmap='nipy_spectral', interpolation='nearest')  # spectral
        plt.axis('off')
        plt.tight_layout()
        if save is not False:
            plt.savefig(save, format='pdf', bbox_inches='tight')
        if show:
            plt.show()

    def standardize_label(self, y):
        """
        standardize the classes label into 0-k
        :param y: 
        :return: 
        """
        import copy
        classes = np.unique(y)
        standardize_y = copy.deepcopy(y)
        for i in range(classes.shape[0]):
            standardize_y[np.nonzero(y == classes[i])] = i
        return standardize_y

    def one2array(self, y):
        n_classes = np.unique(y).__len__()
        y_expected = np.zeros((y.shape[0], n_classes))
        for i in range(y.shape[0]):
            y_expected[i][y[i]] = 1
        return y_expected

    def zca_whitening(self, x, epsilon=1e-6, mean=None, whitening=None):
        '''
        Applies ZCA whitening the the input data.
        Arguments:
            x: numpy array of shape (batch_size, dim). If the input has
                more than 2 dimensions (such as images), it will be flatten the
                data.
            epsilon: an hyper-parameter called the whitening coefficient, default is 1e-6
            mean: numpy array of shape (dim) that will be used as the mean.
                If None (Default), the mean will be computed from the input data.
            whitening: numpy array shaped (dim, dim) that will be used as the
                whitening matrix. If None (Default), the whitening matrix will be
                computed from the input data.
        Returns:
            white_data: numpy array with whitened data. Has the same shape as
                the input.
            mean: numpy array of shape (dim) that contains the mean of each input
                dimension. If mean was provided as input, this is a copy of it.
            whitening:  numpy array of shape (dim, dim) that contains the whitening
                matrix. If whitening was provided as input, this is a copy of it.
        '''
        if not x.size:
            # Simply return if data_set is empty
            return x, mean, whitening
        data_shape = x.shape
        size = data_shape[0]
        white_data = x.reshape((size, -1))

        if mean is None:
            # No mean matrix, we must compute it
            mean = white_data.mean(axis=0)
        # Remove mean
        white_data -= mean

        # If no whitening matrix, we must compute it
        if whitening is None:
            cov = np.dot(white_data.T, white_data) / size
            U, S, V = np.linalg.svd(cov)
            whitening = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + epsilon))), U.T)

        white_data = np.dot(white_data, whitening)
        return white_data.reshape(data_shape), mean, whitening

    def colorize_map(self, gt_pred, colors=None, background_color=None):
        """
        make a colorful RGB classification map using specified color list
        :param gt_pred: predicted maps with s size of height*width
        :param colors: RGB color list for every class
        :return: colorized RGB map
        """
        if colors is None:
            colors = CLASS_MAP_COLOR
        if isinstance(colors, dict):
            colors = list(colors.values())
        n_row, n_col = gt_pred.shape
        map_color = np.zeros((n_row, n_col, 3))
        if background_color is not None:  # # use background color
            if background_color == 'w':
                map_color = np.zeros((n_row, n_col, 3)) + 255
            elif background_color == 'b':
                map_color = np.zeros((n_row, n_col, 3))
            elif isinstance(background_color, Iterable) and len(background_color) == 3:
                map_color = np.ones((n_row, n_col, 3)) * np.asarray(background_color)
            else:
                raise Exception('background should be white or black')
        clz = np.unique(gt_pred)
        if background_color is not None:
            clz = clz[1:]
        assert len(clz) <= len(colors)
        for clz_i, color_i in zip(clz, colors):
            indx = np.where(gt_pred == clz_i)
            map_color[indx] = color_i
        return map_color.astype('int')

    def prepare_data_public(self, img_path, gt_tr_path, gt_te_path):

        import scipy.io as sio
        img_mat = sio.loadmat(img_path)
        img_keys = img_mat.keys()
        img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']

        gt_mat_tr = sio.loadmat(gt_tr_path)
        gt_key_tr = [k for k in gt_mat_tr.keys() if k != '__version__' and k != '__header__' and k != '__globals__']

        gt_mat_te = sio.loadmat(gt_te_path)
        gt_key_te = [k for k in gt_mat_te.keys() if k != '__version__' and k != '__header__' and k != '__globals__']

        img = img_mat[img_key[0]].astype('float64')
        gt_tr = gt_mat_tr[gt_key_tr[0]].astype('int8')
        gt_te = gt_mat_te[gt_key_te[0]].astype('int8')
        return img, gt_tr, gt_te
