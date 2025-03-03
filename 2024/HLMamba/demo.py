import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
from HLMamba_Net import HLMamba
import numpy as np
import time
import os
import scipy.io as sio
from utils import train_patch,train_patch1, setup_seed, output_metric, print_args, train_epoch, valid_epoch, split_train_test_labels
from generate_pic import generate_png
from torchsummary import summary

# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser("HLMamba")
parser.add_argument('--gpu_id', default='3', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')  # Muufl 200
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')  # diffGrad 1e-3
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--dataset', choices=['Muufl', 'Trento', 'Houston'], default='Houston', help='dataset to use')
parser.add_argument('--num_classes', choices=[11, 6, 15], default=15, help='number of classes')
parser.add_argument('--flag_test', choices=['test', 'train', 'pretrain'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--train_num', type=int, default=20, help='number of per-class')
parser.add_argument('--patches1', type=int, default=8, help='number1 of patches')
parser.add_argument('--patches2', type=int, default=16, help='number2 of patches')
parser.add_argument('--patches3', type=int, default=24, help='number3 of patches')
parser.add_argument('--training_mode', choices=['one_time', 'ten_times', 'test_all', 'train_standard'], default='one_time', help='training times')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def train_1times():
    # setup_seed(args.seed)
    # -------------------------------------------------------------------------------
    # prepare data
    if args.dataset == 'Houston':
        DataPath1 = r'./dataset/houston_hsi.mat'
        DataPath2 = r'./dataset/houston_lidar.mat'
        LabelPath_10TIMES = sio.loadmat('./dataset/houston_gt.mat')
        LabelPath_10TIMES = LabelPath_10TIMES['gt']
        gt = LabelPath_10TIMES.reshape(np.prod(LabelPath_10TIMES.shape[:2]),)
        print('gt',gt.shape)
        CLASSES_NUM = max(gt)
        gt = LabelPath_10TIMES
        print('The class numbers of the HSI data is:', CLASSES_NUM)
        Data1 = loadmat(DataPath1)['houston_hsi']  # (349,1905,144)
        Data2 = loadmat(DataPath2)['houston_lidar']  # (349,1905)

    elif args.dataset == 'Muufl':
        DataPath1 = r'./dataset/muufl_hsi.mat'
        DataPath2 = r'./dataset/muufl_lidar.mat'
        LabelPath_10TIMES = sio.loadmat('./dataset/muufl_gt.mat')
        LabelPath_10TIMES = LabelPath_10TIMES['muufl_gt']
        gt = LabelPath_10TIMES.reshape(np.prod(LabelPath_10TIMES.shape[:2]),)
        CLASSES_NUM = max(gt)
        gt = LabelPath_10TIMES
        print('The class numbers of the HSI data is:', CLASSES_NUM)
        Data1 = loadmat(DataPath1)['muufl_hsi']  # (349,1905,144)
        Data2 = loadmat(DataPath2)['muufl_lidar']  # (349,1905)
    
    elif args.dataset == 'Trento':
        DataPath1 = r'./dataset/trento_hsi.mat'
        DataPath2 = r'./trento_lidar.mat'
        LabelPath_10TIMES = sio.loadmat('./dataset/trento_gt.mat')
        LabelPath_10TIMES = LabelPath_10TIMES['gt']
        gt = LabelPath_10TIMES.reshape(np.prod(LabelPath_10TIMES.shape[:2]),)
        print(LabelPath_10TIMES.shape)
        CLASSES_NUM = max(gt)
        gt = LabelPath_10TIMES
        print('gt',gt.shape)
        print('The class numbers of the HSI data is:', CLASSES_NUM)
        Data1 = loadmat(DataPath1)['HSI']  # (349,1905,144)
        Data2 = loadmat(DataPath2)['LiDAR']  # (349,1905)
    
    train_num = args.train_num
    TrLabel_10TIMES, TsLabel_10TIMES = split_train_test_labels(LabelPath_10TIMES, train_num)

    Data1 = Data1.astype(np.float32)
    Data2 = Data2.astype(np.float32)

    gradient = np.gradient(Data2)
    Data3 = np.sqrt(gradient[0]**2)

    patchsize1 = args.patches1  # input spatial size for 2D-CNN
    pad_width1 = np.floor(patchsize1 / 2)
    pad_width1 = int(pad_width1)  # 8
    patchsize2 = args.patches2  # input spatial size for 2D-CNN
    pad_width2 = np.floor(patchsize2 / 2)
    pad_width2 = int(pad_width2)  # 8
    patchsize3 = args.patches3  # input spatial size for 2D-CNN
    pad_width3 = np.floor(patchsize3 / 2)
    pad_width3 = int(pad_width3)  # 8
    TrainPatch11, TrainPatch21, TrainPatch31, TrainLabel = train_patch(Data1, Data2, Data3, patchsize1, pad_width1, TrLabel_10TIMES, False)
    TestPatch11, TestPatch21, TestPatch31, TestLabel = train_patch(Data1, Data2, Data3, patchsize1, pad_width1, TsLabel_10TIMES, False)

    TrainPatch12, TrainPatch22, TrainPatch32, _ = train_patch(Data1, Data2, Data3, patchsize2, pad_width2, TrLabel_10TIMES, False)
    TestPatch12, TestPatch22, TestPatch32, _ = train_patch(Data1, Data2, Data3, patchsize2, pad_width2, TsLabel_10TIMES, False)

    TrainPatch13, TrainPatch23, TrainPatch33, _ = train_patch(Data1, Data2, Data3, patchsize3, pad_width3, TrLabel_10TIMES, False)
    TestPatch13, TestPatch23, TestPatch33, _ = train_patch(Data1, Data2, Data3, patchsize3, pad_width3, TsLabel_10TIMES, False)

    train_dataset = Data.TensorDataset(TrainPatch11, TrainPatch21, TrainPatch31, TrainPatch12, TrainPatch22,TrainPatch32, TrainPatch13, TrainPatch23,TrainPatch33, TrainLabel)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = Data.TensorDataset(TestPatch11, TestPatch21, TestPatch31, TestPatch12, TestPatch22, TestPatch32, TestPatch13, TestPatch23,TestPatch33, TestLabel)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    [m1, n1, l1] = np.shape(Data1)
    Data2 = Data2.reshape([m1, n1, -1])  # when lidar is one band, this is used
    Data3 = Data3.reshape([m1, n1, -1]) 
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    height3, width3, band3 = Data3.shape
    # data size
    print("height1={0},width1={1},band1={2}".format(height1, width1, band1))
    print("height2={0},width2={1},band2={2}".format(height2, width2, band2))
    print("height2={0},width2={1},band2={2}".format(height3, width3, band3))
    # -------------------------------------------------------------------------------
    # create model
    model = HLMamba(l1=band1, l2=band2, patch_size=args.patches1, num_patches=64, num_classes=args.num_classes,
                encoder_embed_dim=64, decoder_embed_dim=32,
                en_depth=3, en_heads=4, de_depth=1, de_heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1)
    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

    
    summary(model, input_size=[(band1, 8, 8), (band2, 8, 8), (band2, 8, 8), (band1, 16, 16), (band2, 16, 16), (band2, 16, 16),(band1, 24, 24), (band2, 24, 24), (band2, 24, 24)], batch_size=-1)  # 输入的大小根据实际情况调整
    print('Total params: %.2f' % (sum(p.numel() for p in model.parameters())))

    # -------------------------------------------------------------------------------
    # train and test
    if args.flag_test == 'train':
        BestAcc = 0
        val_acc = []
        print("start training")
        tic = time.time()
        for epoch in range(args.epoches):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, train_loader, criterion, optimizer)
            OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
            print("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                  .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
            scheduler.step()

            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, test_loader, criterion)
                OA2, AA2, Kappa2, CA2 = output_metric(tar_v, pre_v)
                val_acc.append(OA2)
                print("Every 5 epochs' records:")
                print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
                print(CA2)
                if OA2 > BestAcc:
                    torch.save(model.state_dict(), './Wights/Best_' + args.dataset  + '_Net.pkl')
                    BestAcc = OA2

        toc = time.time()
        model.eval()
        model.load_state_dict(torch.load('./Wights/Best_' + args.dataset  + '_Net.pkl'))
        tar_v, pre_v = valid_epoch(model, test_loader, criterion) 
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        print("Final records:")
        print("Maxmial Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        print(CA)
        print("Running Time: {:.2f}".format(toc - tic))
        print("**************************************************")
        print("Parameter:")
        print_args(vars(args))

if __name__ == '__main__':
    setup_seed(args.seed)
    if args.training_mode == 'one_time':
        train_1times()




