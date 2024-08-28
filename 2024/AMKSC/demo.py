import sys
import time
import warnings

from Preprocessing import Processor, CLASS_MAP_COLOR_16, CLASS_MAP_COLOR_B, CLASS_MAP_COLOR_8

warnings.filterwarnings("ignore")

sys.path.append('./')
sys.path.append('/root/python_codes/')
import os
import numpy as np
import torch
from scipy.io import loadmat

import argparse
from utils import yaml_config_hook, metric, initialization_utils, load_multimodal_data
# from model import KSCAnchorGraph
# from model_sum1 import KSCAnchorGraph
from AMKSC import AMKSC
from utils.superpixel_anchors import generate_anchors
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print(f'using {DEVICE}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    initialization_utils.set_global_random_seed(seed=args.seed)

    root = args.dataset_root

    # prepare data
    if args.dataset == "Houston":
        # im_1, im_2, im_3 = 'data_HS_LR', 'data_LiDAR', 'data_MS_HR'
        # gt_ = 'GT-ALL'
        # img_path = (root + im_1 + '.mat', root + im_2 + '.mat', root + im_3 + '.mat')
        # data_name = (im_1, im_2, im_3)

        im_1, im_2 = 'data_HS_LR', 'data_LiDAR'
        gt_ = 'GT-ALL'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        data_name = (im_1, im_2)
    elif args.dataset == "Trento":
        im_1, im_2 = 'Trento-HSI', 'Trento-Lidar'
        gt_ = 'Trento-GT'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        # img_path = (root + im_2 + '.mat', )
        CLASS_MAP_COLOR_16 = CLASS_MAP_COLOR_8
        data_name = (im_1, im_2)
    elif args.dataset == "Finland":
        im_1, im_2 = 'HSI', 'RGB-Downsample'
        gt_ = 'GT'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        data_name = (im_1, im_2)
    elif args.dataset == "Berlin":
        im_1, im_2 = 'data_HS_LR', 'data_SAR_HR'
        gt_ = 'GT-ALL'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        data_name = (im_1, im_2)
    elif args.dataset == "Augsburg":
        im_1, im_2, im_3 = 'data_HS_LR', 'data_SAR_HR', 'data_DSM'
        gt_ = 'GT-ALL'
        # img_path = (root + im_1 + '.mat', root + im_2 + '.mat', root + im_3 + '.mat')
        # data_name = (im_1, im_2, im_3)
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        data_name = (im_1, im_2)

    elif args.dataset == "MDAS-Sub1":
        im_1, im_2, im_3, im_4 = 'MDAS-Sub1-HSI', 'MDAS-Sub1-MSI', 'MDAS-Sub1-SAR', 'MDAS-Sub1-DSM'
        gt_ = 'MDAS-Sub1-GT'
        # img_path = (root + im_1 + '.mat', root + im_2 + '.mat', root + im_3 + '.mat', root + im_4 + '.mat')
        # data_name = (im_1, im_2, im_3, im_4)
        img_path = (root + im_1 + '.mat', root + im_4 + '.mat')
        data_name = (im_1, im_4)

    elif args.dataset == "MUUFL":
        im_1, im_2 = 'HSI', 'LiDAR_data_first_return'  # 'LiDAR_data_first_last_return'
        gt_ = 'GT'
        img_path = (root + im_1 + '.mat', root + im_2 + '.mat')
        data_name = (im_1, im_2)
    else:
        raise NotImplementedError
    gt_path = root + gt_ + '.mat'

    for i, p_ in enumerate(img_path):
        print(f'modality #{i + 1}: {p_}')
    modality_list, y, gt = load_multimodal_data(gt_path, *img_path, patch_size=(args.patch_size, args.patch_size),
                                                is_labeled=args.is_labeled_pixel)

    if args.is_labeled_pixel:
        class_num = len(np.unique(y))
        y_labeled = np.copy(y)
    else:
        indx_labeled = np.nonzero(y)[0]
        y_labeled = y[indx_labeled]
        class_num = len(np.unique(y)) - 1
    data_size = modality_list[0].shape
    spatial_size = gt.shape
    for i, m_ in enumerate(modality_list):
        print(f'modality # {i + 1} shape: {m_.shape}')
    print('# classes:', class_num)
    print('\n args:', args)

    if args.anchor_type == 'precomputed':
        seg_labels = []
        for i, name_ in enumerate(data_name):
            path_superpixel = f'./superpixel_utils/seg_labels/{args.dataset}-{name_}-seg-labels.mat'
            labels = loadmat(path_superpixel)['labels']
            seg_labels.append(labels)
        precomputed_anchors = generate_anchors(modality_list, superpixel_labels=seg_labels)
    else:
        precomputed_anchors = None
    start_time = time.time()
    # model = KSCAnchorGraph(class_num, n_anchor=args.n_anchor, lambda_=args.weight_decay, anchor_type=args.anchor_type,
    #                        precomputed_anchors=precomputed_anchors, smooth_coef=args.smooth_coef,
    #                        max_iter=args.max_iter, epsilon=1e-5, device=DEVICE, kernel=args.kernel)
    model = AMKSC(spatial_size, class_num, n_anchor=args.n_anchor, lambda_tv=args.lambda_TV,
                              anchor_type=args.anchor_type, precomputed_anchors=precomputed_anchors,
                              smooth_coef=args.smooth_coef, weight_decay=args.weight_decay,
                              max_iter=args.max_iter, n_kernel_sampling=args.n_kernel_sampling, epsilon=1e-5,
                              seed=args.seed, device=DEVICE, kernel=args.kernel)
    model.train(modality_list)
    y_pred = model.predict()
    print('learned weights:', model.mu)

    if not args.is_labeled_pixel:
        y_pred_labeled = y_pred[indx_labeled]
        y_pred_2D = y_pred.reshape(gt.shape)
    else:
        y_pred_labeled = y_pred
    acc, kappa, nmi, ari, pur, bcubed_F, ca = metric.cluster_accuracy(y_labeled, y_pred_labeled)
    print('OA = {:.4f} Kappa = {:.4f} NMI = {:.4f} ARI = {:.4f} Purity = {:.4f}  BCubed F = {:.4f}'.format(acc, kappa,
                                                                                                           nmi, ari,
                                                                                                           pur,
                                                                                                           bcubed_F))
    for i, ca_ in enumerate(ca):
        print(f'class #{i} ACC: {ca_:.4f}')
    running_time = time.time() - start_time
    print(f'running time: {running_time:.3f} s')

    Z = model.Z.cpu().numpy()
    emb = model.embedding
    mu = model.mu.cpu().numpy()
    singular_values = model.singular_values
    doubly_stochastic_Z = model.doubly_stochastic_Z.cpu().numpy()
    np.savez(f'results/{args.dataset}-logs.npz', Z=Z, normlized_Z=doubly_stochastic_Z, embedding=emb,
             singular_values=singular_values,
             mu=mu, time=running_time, acc=acc,
             kappa=kappa, nmi=nmi, ari=ari, pur=pur, bcubed_F=bcubed_F, ca=ca, y_pred=y_pred_2D, params=args)

    plt.matshow(Z[0, :].reshape(gt.shape), cmap='jet')

    # plt.show()
    # ## ====================================
    # # show classification map
    # # ======================================
    p = Processor()
    save_name = f'Figures/classmap-{args.dataset}-{acc * 10000:.0f}.pdf'
    gt_color = p.colorize_map(y_pred_2D, colors=CLASS_MAP_COLOR_16, background_color=None)

    fig, ax = plt.subplots()
    ax.imshow(gt_color)
    plt.axis('off')
    plt.tight_layout()
    print(save_name)
    fig.savefig(save_name, format='pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

