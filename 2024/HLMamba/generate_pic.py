import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import datetime
import torch.utils.data as Data


def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

################# Houston datset ###########################
# def list_to_colormap(x_list):
#     y = np.zeros((x_list.shape[0], 3))
#     for index, item in enumerate(x_list):
#         if item == 0:
#             y[index] = np.array([76, 188, 56]) / 255.
#         if item == 1:
#             y[index] = np.array([128, 204, 42]) / 255.
#         if item == 2:
#             y[index] = np.array([64, 138, 88]) / 255.
#         if item == 3:
#             y[index] = np.array([56, 138, 62]) / 255.
#         if item == 4:
#             y[index] = np.array([144,72,47]) / 255.
#         if item == 5:
#             y[index] = np.array([114,208,210]) / 255.
#         if item == 6:
#             y[index] = np.array([255,255,255]) / 255.
#         if item == 7:
#             y[index] = np.array([201,169,206]) / 255.
#         if item == 8:
#             y[index] = np.array([232,33,39]) / 255.
#         if item == 9:
#             y[index] = np.array([121,31,36]) / 255.
#         if item == 10:
#             y[index] = np.array([62,99,183]) / 255.
#         if item == 11:
#             y[index] = np.array([223,230,49]) / 255.
#         if item == 12:
#             y[index] = np.array([226,134,32]) / 255.
#         if item == 13:
#             y[index] = np.array([80,41,137]) / 255.
#         if item == 14:
#             y[index] = np.array([243,99,77]) / 255.
#         if item == 15:
#             y[index] = np.array([255, 215, 0]) / 255.
#         if item == 16:
#             y[index] = np.array([0, 0, 0]) / 255.
#         if item == 17:
#             y[index] = np.array([215, 255, 0]) / 255.
#         if item == 18:
#             y[index] = np.array([0, 255, 215]) / 255.
#         if item == -1:
#             y[index] = np.array([0, 0, 0]) / 255.
#     return y

################# MUUFL datset ###########################
def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0,128,1]) / 255.
        if item == 1:
            y[index] = np.array([0,255,1]) / 255.
        if item == 2:
            y[index] = np.array([2,1,203]) / 255.
        if item == 3:
            y[index] = np.array([254,203,0]) / 255.
        if item == 4:
            y[index] = np.array([252,0,49]) / 255.
        if item == 5:
            y[index] = np.array([114,208,210]) / 255.
        if item == 6:
            y[index] = np.array([102,0,205]) / 255.
        if item == 7:
            y[index] = np.array([254,126,151]) / 255.
        if item == 8:
            y[index] = np.array([201,102,0]) / 255.
        if item == 9:
            y[index] = np.array([254,254,0]) / 255.
        if item == 10:
            y[index] = np.array([204,66,100]) / 255.
        if item == 11:
            y[index] = np.array([223,230,49]) / 255.
        if item == 12:
            y[index] = np.array([226,134,32]) / 255.
        if item == 13:
            y[index] = np.array([80,41,137]) / 255.
        if item == 14:
            y[index] = np.array([243,99,77]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y

################# Trento datset ###########################
# def list_to_colormap(x_list):
#     y = np.zeros((x_list.shape[0], 3))
#     for index, item in enumerate(x_list):
#         if item == 0:
#             y[index] = np.array([0,128,1]) / 255.
#         if item == 1:
#             y[index] = np.array([78,200,237]) / 255.
#         if item == 2:
#             y[index] = np.array([59,86,167]) / 255.
#         if item == 3:
#             y[index] = np.array([254,210,13]) / 255.
#         if item == 4:
#             y[index] = np.array([238,52,37]) / 255.
#         if item == 5:
#             y[index] = np.array([125,21,22]) / 255.
#         if item == 6:
#             y[index] = np.array([121,21,22]) / 255.
#         if item == 7:
#             y[index] = np.array([201,169,206]) / 255.
#         if item == 8:
#             y[index] = np.array([232,33,39]) / 255.
#         if item == 9:
#             y[index] = np.array([121,31,36]) / 255.
#         if item == 10:
#             y[index] = np.array([62,99,183]) / 255.
#         if item == 11:
#             y[index] = np.array([223,230,49]) / 255.
#         if item == 12:
#             y[index] = np.array([226,134,32]) / 255.
#         if item == 13:
#             y[index] = np.array([80,41,137]) / 255.
#         if item == 14:
#             y[index] = np.array([243,99,77]) / 255.
#         if item == 15:
#             y[index] = np.array([255, 215, 0]) / 255.
#         if item == 16:
#             y[index] = np.array([0, 0, 0]) / 255.
#         if item == 17:
#             y[index] = np.array([215, 255, 0]) / 255.
#         if item == 18:
#             y[index] = np.array([0, 255, 215]) / 255.
#         if item == -1:
#             y[index] = np.array([0, 0, 0]) / 255.
#     return y



def generate_png(all_iter, net, gt_hsi, device, total_indices, Dataset, HSIOnly=False):
    pred_test = []

    # for x1, x2, y in all_iter: # 黑色背景
    a_list = []
    for x1, x2, x3, x4, x5, x6, x7, x8, x9 in all_iter:  # 全像素
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        x5 = x5.to(device)
        x6 = x6.to(device)
        x7 = x7.to(device)
        x8 = x8.to(device)
        x9 = x9.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        if HSIOnly:
            # y_hat = net(x1)
            pred_test.extend(np.array(net(x1).cpu().argmax(axis=1)))
        else:
            y_hat= net(x1, x2, x3, x4, x5, x6, x7, x8, x9)

            A = y_hat.detach().cpu().numpy()    #[64,11]
            a_list.append(A)

            pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))

    pre = np.vstack(a_list)
    print(pre.shape)
    sio.savemat('/home/liaodl/Project/paperone/GLT/T-SNE/predicted_matrix_'+Dataset+'.mat', {'pre': pre})       

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            x_label[i] = 16

    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    print(y_list.shape)
    print(gt_hsi.shape[0])
    print(gt_hsi.shape[1])
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')

    # classification_map(y_re, gt_hsi, 300,
    #                     './classification_maps/' + Dataset + '_' + day_str + '.png')
    # classification_map(gt_re, gt_hsi, 300,
    #                     './classification_maps/' + Dataset + '_' + day_str + '_gt.png')
