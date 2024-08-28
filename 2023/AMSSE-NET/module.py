# -*- coding:utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
import random
import torch
from einops import rearrange
from operator import truediv
from sklearn.decomposition import PCA

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def pad_with_zeros(X, margin=2):
    """apply zero padding to X with margin(w, h, c)"""  #(c, w, h)

    new_X = np.zeros((X.shape[0], X.shape[1] + 2 * margin, X.shape[2] + 2 * margin))
    x_offset = margin
    y_offset = margin
    new_X[:, x_offset:X.shape[1] + x_offset, y_offset:X.shape[2] + y_offset] = X
    return new_X

def pixel_select(Y, train_ratio):
    """
    :param Y:
    :param train_ratio: 训练集比率
    :return:
    """
    test_pixels = Y.copy()  # 复制Y到test_pixels
    kinds = np.unique(Y).shape[0] - 1  # np.unique(Y)=array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],dtype=uint8) ,kinds=分类种类数
    # print(kinds)
    for i in range(kinds):
        num = np.sum(Y == (i + 1))  # 计算每个类总共有多少样本 ,从Y=1到Y=16
        train_num = int(round(num * train_ratio))
        temp1 = np.where(Y == (i + 1))  # 返回标签满足第i+1类的位置索引，第一次循环返回第一类的索引
        temp2 = random.sample(range(num), train_num)  # get random sequence,random.sample表示从某一序列中随机获取所需个数（train_num）的数并以片段的形式输出,,再这里将随机从每个种类中挑选train_num个样本
        for i in temp2:
            test_pixels[temp1[0][temp2], temp1[1][temp2]] = 0  # 除去训练集样本

    train_pixels = Y - test_pixels
    return train_pixels, test_pixels

def GetImageCubes(input_data, pixels_select, windowSize=11):  # 这里的label_select就是train_pixels/test_pixels
    Band = input_data.shape[2]
    kind = np.unique(pixels_select).shape[0] - 1  # 得到测试或者训练集中的种类数
    # print(kind)
    # print('input_data.shape:', input_data.shape)
    paddingdata = np.pad(input_data, ((30, 30), (30, 30), (0, 0)),
                         "constant")  # 采用边缘值填充 [203, 203, 200]                 可以作为超参数
    paddinglabel = np.pad(pixels_select, ((30, 30), (30, 30)), "constant")  # 此处"constant"应改为"edge"
    # 得到 label的 pixel坐标位置,去除背景元素
    # p rint('paddingdata.shape:', paddingdata.shape)
    # print('paddinglabel.shape:', paddinglabel.shape)
    pixel = np.where(paddinglabel != 0)  # pixel = np.where(label_select != 0)  ，这里的pixel是坐标数据，不是光谱数据
    # the number of batch
    num = np.sum(pixels_select != 0)  # 参与分类的像素点个数
    batch_out = np.zeros([num, windowSize, windowSize, Band])
    batch_label = np.zeros([num, kind])
    for i in range(num):  # 得到每个像素点的batch，在这里为19*19的方块
        row_start = pixel[0][i] - windowSize // 2
        row_end = pixel[0][i] + windowSize // 2 + 1
        col_start = pixel[1][i] - windowSize // 2
        col_end = pixel[1][i] + windowSize // 2 + 1
        batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :]  # 得到一个数据块
        # batch_out[i, :, :, :] = paddingdata[col_start:col_end, row_start:row_end, :]  # 得到一个数据块
        temp = (paddinglabel[pixel[0][i], pixel[1][i]] - 1)  # temp = (label_selct[pixel[0][i],pixel[1][i]]-1)
        batch_label[i, temp] = 1  # 独热编码，并且是从零开始的
    # 修改合适三维卷积输入维度 [depth height weight]
    batch_out = batch_out.swapaxes(1, 3)
    # batch_out = batch_out.swapaxes(1, 2)
    batch_label = np.argmax(batch_label, axis=-1)
    # batch_out = batch_out[:, :, :, :, np.newaxis]           # np.newaxis:增加维度
    # print('batch_out.shape:', batch_out.shape)
    return batch_out, batch_label

def split_train_test_set(X, y, train_ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=345,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def weight_init(layer):
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(layer.weight, val=1.0)
        torch.nn.init.constant_(layer.bias, val=0.0)
    elif isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, val=0.0)

def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)  # 获取confusion_matrix的主对角线所有数值
    list_raw_sum = np.sum(confusion_matrix, axis=1)  # 将主对角线所有数求和
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # list_diag/list_raw_sum  对角线各个数字/对角线所有数字的总和
    average_acc = np.mean(each_acc)  #
    return np.round(each_acc, 4), average_acc


def GetImageCubes_all(input_data, pixels_select, windowSize=11):  # 这里的label_select就是train_pixels/test_pixels
    pixels_select=pixels_select+1
    Band = input_data.shape[2]
    kind = np.unique(pixels_select).shape[0]-1  # 得到测试或者训练集中的种类数
    # print(kind)
    # print('input_data.shape:', input_data.shape)
    paddingdata = np.pad(input_data, ((30, 30), (30, 30), (0, 0)),
                         "constant")  # 采用边缘值填充 [203, 203, 200]                 可以作为超参数
    paddinglabel = np.pad(pixels_select, ((30, 30), (30, 30)), "constant")  # 此处"constant"应改为"edge"
    # 得到 label的 pixel坐标位置,去除背景元素
    # p rint('paddingdata.shape:', paddingdata.shape)
    # print('paddinglabel.shape:', paddinglabel.shape)
    pixel = np.where(paddinglabel != 0)  # pixel = np.where(label_select != 0)  ，这里的pixel是坐标数据，不是光谱数据
    # the number of batch
    num = np.sum(pixels_select != 0)  # 参与分类的像素点个数
    batch_out = np.zeros([num, windowSize, windowSize, Band])
    batch_label = np.zeros([num, kind])
    for i in range(num):  # 得到每个像素点的batch，在这里为19*19的方块
        row_start = pixel[0][i] - windowSize // 2
        row_end = pixel[0][i] + windowSize // 2 + 1
        col_start = pixel[1][i] - windowSize // 2
        col_end = pixel[1][i] + windowSize // 2 + 1
        batch_out[i, :, :, :] = paddingdata[row_start:row_end, col_start:col_end, :]  # 得到一个数据块
        # batch_out[i, :, :, :] = paddingdata[col_start:col_end, row_start:row_end, :]  # 得到一个数据块
        temp = (paddinglabel[pixel[0][i], pixel[1][i]] - 1)  # temp = (label_selct[pixel[0][i],pixel[1][i]]-1)
        # batch_label[i, temp] = 1  # 独热编码，并且是从零开始的
    # 修改合适三维卷积输入维度 [depth height weight]
    batch_out = batch_out.swapaxes(1, 3)
    # batch_out = batch_out.swapaxes(1, 2)
    # batch_label = np.argmax(batch_label, axis=-1)
    # batch_out = batch_out[:, :, :, :, np.newaxis]           # np.newaxis:增加维度
    # print('batch_out.shape:', batch_out.shape)
    return batch_out

