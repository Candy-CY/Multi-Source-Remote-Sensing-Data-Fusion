# -*- coding:utf-8 -*-

import torch
import torch.optim as optim
from einops import rearrange
from model import *
from loss import MarginLoss
import time
import matplotlib.pyplot as plt
import numpy as np


def train_best_model(dataset, train_iter, device, epoches, ITER, TRAIN_SIZE, TEST_SIZE, TOTAL_SIZE, test_iter, num_classes, in_channels_2, windowSize):
    for index_iter in range(ITER):
        train_loss_list = []
        train_acc_list = []
        oa_list = []
        oa_need_list = []
        epoch_need_list = []
        x_epoch = []
        eye = torch.eye(int(max(train_iter.dataset.tensors[2]) + 1)).cuda()
        net1 = fusion_main(15, in_channels_2, num_classes, windowSize).to(device)

        # net.apply(weight_init)  #网络权重初始化
        optimizer1 = optim.Adam(net1.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)
        lr_adjust = torch.optim.lr_scheduler.StepLR(optimizer=optimizer1, step_size=80, gamma=0.5, last_epoch=-1)
        net1.train()
        # loss1 = torch.nn.MSELoss(reduction='mean')
        loss1 = torch.nn.CrossEntropyLoss()
        print('\niter:', index_iter + 1)
        print('TRAIN_SIZE: ', TRAIN_SIZE)
        print('TEST_SIZE: ', TEST_SIZE)
        print('TOTAL_SIZE: ', TOTAL_SIZE)
        print(
            '--------------------------------------------------Training on {}--------------------------------------------------\n'.format(
                device))
        start = time.time()
        for epoch in range(epoches):
            train_acc_sum, train_loss_sum = 0.0, 0.0

            time_epoch = time.time()
            # lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0.0, last_epoch=-1)
            for step, (X_hsi, X_lidar, target) in enumerate(train_iter):
                X_hsi = X_hsi.to(device)
                X_lidar = X_lidar.to(device)
                target = target.to(device)
                target = target.to(torch.int64)
                target_hot = eye[target]

                # 前向传播
                out, out0, out1, out2, beta = net1(X_hsi, X_lidar)
                # print(target_hot)
                # print(out)
                l1 = beta[0] * loss1(out0, target_hot.float()) + beta[1] * loss1(out1, target_hot.float()) + beta[
                    2] * loss1(out2, target_hot.float())
                # l1 = loss1(out, x1)+loss1(out, x2)
                # print(beta)
                # 反向传播及优化
                optimizer1.zero_grad()  # 梯度清零
                l1.backward()
                optimizer1.step()
                train_loss_sum += l1.cpu().item()
                train_acc_sum += (out.argmax(dim=-1) == target_hot.argmax(dim=-1).to(device)).float().sum().cpu().item()
            lr_adjust.step()
            print(optimizer1.state_dict()['param_groups'][0]['lr'])

            print('epoch %d, train loss %.6f, train acc %.4f, time %.2f sec' % (
                epoch + 1, train_loss_sum / len(train_iter.dataset), train_acc_sum / len(train_iter.dataset),
                time.time() - time_epoch))
            train_loss_list.append(train_loss_sum / len(train_iter.dataset))  # / batch_count)
            train_acc_list.append(train_acc_sum / len(train_iter.dataset))
            if train_loss_list[-1] <= min(train_loss_list) and epoch>=50:
                torch.save(net1.state_dict(), './models/' + dataset + '.pt')
                print('**Successfully Saved Best hsi model parametres!***\n')  # 保存在训练集上损失值最好的模型效果
                x_epoch.append(epoch)

        End = time.time()
        print('***Training End! Total Time %.1f sec***' % (End - start))
        # print(max(oa_list))
        print(epoch_need_list)
        print(oa_need_list)
    return net1


