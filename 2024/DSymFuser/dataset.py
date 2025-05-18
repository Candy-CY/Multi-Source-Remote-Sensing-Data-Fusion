import os
import random
import numpy as np
import torch
from scipy import io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor


class HXDataset(Dataset):

    def __init__(self, args, hsi, lidar, gt, mask, transform=ToTensor(), train = False):

        modes = ['symmetric', 'reflect']
        self.train = train
        self.mask = mask
        self.pad = args.patch_size // 2
        self.patch_size = args.patch_size

        self.hsi = np.pad(hsi, ((self.pad, self.pad),
                                (self.pad, self.pad), (0, 0)), mode=modes[self.patch_size % 2])
        if lidar.ndim == 2:
            self.lidar = np.pad(lidar, ((self.pad, self.pad),
                                    (self.pad, self.pad)), mode=modes[self.patch_size % 2])
        elif lidar.ndim == 3:
            self.lidar = np.pad(lidar, ((self.pad, self.pad),
                                    (self.pad, self.pad), (0, 0)), mode=modes[self.patch_size % 2])
        self.gt = gt
        if transform:
            self.transform = transform

    def __getitem__(self, index):
        h, w = self.mask[index, :]
        hsi = self.hsi[h: h + self.patch_size, w: w + self.patch_size]
        lidar = self.lidar[h: h + self.patch_size, w: w + self.patch_size]
        if self.transform:
            hsi = self.transform(hsi).float()
            lidar = self.transform(lidar).float()
            if self.train:
                trans = [transforms.RandomHorizontalFlip(0.5),
                         transforms.RandomVerticalFlip(0.5)]
                i = random.randint(0, 1)
                hsi = trans[i](hsi)
                lidar = trans[i](lidar)
        gt = torch.tensor(self.gt[h, w] - 1).long()
        return hsi, lidar, gt

    def __len__(self):
        return self.mask.shape[0]


def load_processed_dataset(args):
    hsi = io.loadmat(os.path.join(args.dataset_dir, args.dataset_name, "HSI.mat"))['HSI'].astype(np.float32)
    H, W, C = hsi.shape
    hsi = hsi.reshape(-1, C)
    hsi = MinMaxScaler().fit_transform(hsi)
    if args.use_pca:
        hsi = PCA(n_components=args.pca_component).fit_transform(hsi)
    else:
        ...
    hsi = hsi.reshape(H, W, -1)
    if args.dataset_name == 'Augsburg' or args.dataset_name == 'Berlin':
        lidar = io.loadmat(os.path.join(args.dataset_dir, args.dataset_name, "SAR.mat"))['SAR'].astype(np.float32)
    else:
        lidar = io.loadmat(os.path.join(args.dataset_dir, args.dataset_name, "LiDAR.mat"))['LiDAR'].astype(np.float32)
    if lidar.ndim == 2: lidar = np.expand_dims(lidar, axis=2)
    H, W, C = lidar.shape
    lidar = lidar.reshape(-1, C)
    lidar = MinMaxScaler().fit_transform(lidar) # StandardScaler.fit_transform
    lidar = lidar.reshape(H, W, -1)

    gt = io.loadmat(os.path.join(args.dataset_dir, args.dataset_name, "gt.mat"))['gt'].astype(np.int32)

    return hsi, lidar, gt

def select_mask(args, gt):  # divide dataset into train and test datasets

    if args.dataset_name == 'MUUFL':
        amount = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
    elif args.dataset_name == 'Trento':
        amount = [129, 125, 105, 154, 184, 122]
    elif args.dataset_name == 'Houston2013':
        amount = [198, 190, 192, 188, 186, 182, 196, 191, 193, 191, 181, 192, 184, 181, 187]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}. Dataset does not exist.")

    ignored_label = 0
    mask = np.ones(shape=gt.shape, dtype=bool)
    mask[gt == ignored_label] = False
    x_pos, y_pos = np.nonzero(mask)

    whole_loc = {}
    train = {}
    test = {}
    m = int(np.max(gt))  # num_classes
    for i in range(m):
        indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if gt[x, y] == i + 1])
        np.random.shuffle(indices)
        whole_loc[i] = indices
        nb_val = int(amount[i])
        train[i] = indices[-nb_val:]
        test[i] = indices[:-nb_val]

    whole_indices = whole_loc[0]
    train_indices = train[0]
    test_indices = test[0]
    for i in range(1, m):
        whole_indices = np.concatenate((whole_indices, whole_loc[i]), axis=0)
        train_indices = np.concatenate((train_indices, train[i]), axis=0)
        test_indices = np.concatenate((test_indices, test[i]), axis=0)

    return whole_indices, train_indices, test_indices


def make_dataloader(args):

    hsi, lidar, gt = load_processed_dataset(args)

    if os.path.exists(os.path.join(args.dataset_dir, args.dataset_name, "TRLabel.mat")):
        train_mask = io.loadmat(os.path.join(args.dataset_dir, args.dataset_name, "TRLabel.mat"))['TRLabel'].astype(np.int64)
        x_pos, y_pos = np.nonzero(train_mask)
        train_indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        test_mask = io.loadmat(os.path.join(args.dataset_dir, args.dataset_name, "TSLabel.mat"))['TSLabel'].astype(np.int64)
        x_pos, y_pos = np.nonzero(test_mask)
        test_indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        whole_indices = np.concatenate((train_indices, test_indices), axis=0)

    else:
        whole_indices, train_indices, test_indices = select_mask(args, gt)

    # whole_indices, train_indices, test_indices = select_mask(args, gt)
    # all_indices = np.argwhere(np.ones(shape=gt.shape))

    train_datasets = HXDataset(args, hsi, lidar, gt, train_indices, train = True)
    HSI, X, _ = train_datasets[0]
    print('Size of HSI data: {}'.format(HSI.shape))
    print('Size of X data: {}'.format(X.shape))
    print('Number of Training dataset: {}'.format(len(train_datasets)))
    test_datasets = HXDataset(args, hsi, lidar, gt, test_indices, train = False)
    print('Number of Test dataset: {}'.format(len(test_datasets)))
    # whole_datasets = HXDataset(args, hsi, lidar, gt, whole_indices, train = False)
    # all_datasets = HXDataset(args, hsi, lidar, gt, all_indices, train = False)

    train_loader = DataLoader(
        train_datasets, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(
        test_datasets, batch_size=args.batch_size, shuffle=False)
    # whole_loader = DataLoader(
    #     whole_datasets, batch_size=1, shuffle=False, num_workers=8)
    # all_loader = DataLoader(
    #     all_datasets, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print("Success!")

    return train_loader, test_loader