# -*- coding:utf-8 -*-
# @Time       :2022/5/19 下午4:11
# @AUTHOR     :DingKexin
# @FileName   :utils.py
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io as sio


def normalize(input2):
    input2_normalize = np.zeros(input2.shape)
    for i in range(input2.shape[2]):
        input2_max = np.max(input2[:, :, i])
        input2_min = np.min(input2[:, :, i])
        input2_normalize[:, :, i] = (input2[:, :, i] - input2_min) / (input2_max - input2_min)

    return input2_normalize


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False


def train_patch(Data1, Data2, Data3, patchsize, pad_width, Label,ALL_Indices):
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)

    Data3 = Data3.reshape([m1, n1, -1])
    [m3, n3, l3] = np.shape(Data3)

    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    for i in range(l3):
        Data3[:, :, i] = (Data3[:, :, i] - Data3[:, :, i].min()) / (Data3[:, :, i].max() - Data3[:, :, i].min())
    x3 = Data3  

    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    x3_pad = np.empty((m3 + patchsize, n3 + patchsize, l3), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2
    for i in range(l3):
        temp = x3[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x3_pad[:, :, i] = temp2

    # construct the training and testing set
    if ALL_Indices:
        [ind1, ind2] = np.where(Label >= 0)
    else:
        [ind1, ind2] = np.where(Label > 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainPatch3 = np.empty((TrainNum, l3, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patch3 = x3_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch3 = np.transpose(patch3, (2, 0, 1))
        TrainPatch3[i, :, :, :] = patch3
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainPatch3 = torch.from_numpy(TrainPatch3)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainPatch3, TrainLabel

def train_patch1(Data1, Data2, Data3, patchsize, pad_width, Label):
    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])
    [m2, n2, l2] = np.shape(Data2)

    Data3 = Data3.reshape([m1, n1, -1])
    [m3, n3, l3] = np.shape(Data3)

    for i in range(l1):
        Data1[:, :, i] = (Data1[:, :, i] - Data1[:, :, i].min()) / (Data1[:, :, i].max() - Data1[:, :, i].min())
    x1 = Data1
    for i in range(l2):
        Data2[:, :, i] = (Data2[:, :, i] - Data2[:, :, i].min()) / (Data2[:, :, i].max() - Data2[:, :, i].min())
    x2 = Data2
    for i in range(l3):
        Data3[:, :, i] = (Data3[:, :, i] - Data3[:, :, i].min()) / (Data3[:, :, i].max() - Data3[:, :, i].min())
    x3 = Data3  

    x1_pad = np.empty((m1 + patchsize, n1 + patchsize, l1), dtype='float32')
    x2_pad = np.empty((m2 + patchsize, n2 + patchsize, l2), dtype='float32')
    x3_pad = np.empty((m3 + patchsize, n3 + patchsize, l3), dtype='float32')
    for i in range(l1):
        temp = x1[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x1_pad[:, :, i] = temp2
    for i in range(l2):
        temp = x2[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x2_pad[:, :, i] = temp2
    for i in range(l3):
        temp = x3[:, :, i]
        temp2 = np.pad(temp, pad_width, 'symmetric')
        x3_pad[:, :, i] = temp2

    # construct the training and testing set
    [ind1, ind2] = np.where(Label >= 0)
    TrainNum = len(ind1)
    TrainPatch1 = np.empty((TrainNum, l1, patchsize, patchsize), dtype='float32')
    TrainPatch2 = np.empty((TrainNum, l2, patchsize, patchsize), dtype='float32')
    TrainPatch3 = np.empty((TrainNum, l3, patchsize, patchsize), dtype='float32')
    TrainLabel = np.empty(TrainNum)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i in range(len(ind1)):
        patch1 = x1_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch1 = np.transpose(patch1, (2, 0, 1))
        TrainPatch1[i, :, :, :] = patch1
        patch2 = x2_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch2 = np.transpose(patch2, (2, 0, 1))
        TrainPatch2[i, :, :, :] = patch2
        patch3 = x3_pad[(ind3[i] - pad_width):(ind3[i] + pad_width), (ind4[i] - pad_width):(ind4[i] + pad_width), :]
        patch3 = np.transpose(patch3, (2, 0, 1))
        TrainPatch3[i, :, :, :] = patch3
        patchlabel = Label[ind1[i], ind2[i]]
        TrainLabel[i] = patchlabel
    # change data to the input type of PyTorch
    TrainPatch1 = torch.from_numpy(TrainPatch1)
    TrainPatch2 = torch.from_numpy(TrainPatch2)
    TrainPatch3 = torch.from_numpy(TrainPatch3)
    TrainLabel = torch.from_numpy(TrainLabel) - 1
    TrainLabel = TrainLabel.long()
    return TrainPatch1, TrainPatch2, TrainPatch3, TrainLabel


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k, v))


def train_epoch(model, train_loader, criterion, optimizer):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data11, batch_data21, batch_data31, batch_data12, batch_data22, batch_data32, batch_data13, batch_data23, batch_data33, batch_target) in enumerate(train_loader):
        batch_data11 = batch_data11.cuda()
        batch_data21 = batch_data21.cuda()
        batch_data31 = batch_data31.cuda()
        batch_data12 = batch_data12.cuda()
        batch_data22 = batch_data22.cuda()
        batch_data32 = batch_data32.cuda()
        batch_data13 = batch_data13.cuda()
        batch_data23 = batch_data23.cuda()
        batch_data33 = batch_data33.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        batch_pred = model(batch_data11, batch_data21, batch_data31, batch_data12, batch_data22, batch_data32, batch_data13, batch_data23, batch_data33)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data11.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


def valid_epoch(model, valid_loader, criterion):
    objs = AverageMeter()
    top1 = AverageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data11, batch_data21, batch_data31, batch_data12, batch_data22, batch_data32, batch_data13, batch_data23, batch_data33, batch_target) in enumerate(valid_loader):
        batch_data11 = batch_data11.cuda()
        batch_data21 = batch_data21.cuda()
        batch_data31 = batch_data31.cuda()
        batch_data12 = batch_data12.cuda()
        batch_data22 = batch_data22.cuda()
        batch_data32 = batch_data32.cuda()
        batch_data13 = batch_data13.cuda()
        batch_data23 = batch_data23.cuda()
        batch_data33 = batch_data33.cuda()
        batch_target = batch_target.cuda()

        batch_pred = model(batch_data11, batch_data21, batch_data31, batch_data12, batch_data22, batch_data32, batch_data13, batch_data23, batch_data33)

        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data11.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def split_train_test_labels(label_matrix, num_samples_per_class, random_seed=None):
    np.random.seed(random_seed)
    
    num_classes = label_matrix.max()  # Assuming classes are numbered consecutively starting from 1
    
    train_indices = []
    test_indices = []
    
    for class_label in range(1, num_classes + 1):
        class_indices = np.where(label_matrix == class_label)
        class_indices = np.array(class_indices).T
        
        # Shuffle indices
        np.random.shuffle(class_indices)
        
        train_indices.extend(class_indices[:num_samples_per_class])
        test_indices.extend(class_indices[num_samples_per_class:])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    train_labels = np.zeros_like(label_matrix)
    test_labels = np.copy(label_matrix)
    
    # Construct training set
    for idx in train_indices:
        train_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]
    
    # Construct test set
    for idx in test_indices:
        test_labels[idx[0], idx[1]] = label_matrix[idx[0], idx[1]]
    
    # Set selected training indices to 0 in test set
    test_labels[train_indices[:, 0], train_indices[:, 1]] = 0

    sio.savemat("train_labels.mat", {'train_labels': train_labels})
    sio.savemat("test_labels.mat", {'test_labels': test_labels})
    
    return train_labels, test_labels


