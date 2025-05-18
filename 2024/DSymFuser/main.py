import argparse
import os
import random
import numpy as np
import torch

from utils.dataset import make_dataloader
from train import train, validation
from model.SFT import Proposed
from utils.FocalLoss import FocalLoss

parser = argparse.ArgumentParser(description="multi_modal RS classification")
parser.add_argument("--model_name", type=str, default='proposed')
parser.add_argument("--dataset_name", type=str, default="Houston2013")
parser.add_argument("--dataset_dir", type=str, default="./datasets")
parser.add_argument("--use_pca", type=bool, default=False)
parser.add_argument("--pca_component", type=int, default=30) # 30
parser.add_argument("--patch_size", type=int, default=11)

parser.add_argument("--epoch", type=int, default=200) # 200 # 150
parser.add_argument("--warmup_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=16) # 16 # 512
parser.add_argument("--lr", type=float, default=0.0001) # 0.0001 # 0.0005
parser.add_argument("--weight_decay", type=float, default=0.01) # 0.1
parser.add_argument("--use_self_scheduler", type=bool, default=True)

parser.add_argument("--seed", type=int, default=100) # 100
parser.add_argument("--saving_path", type=str, default="./model_params")
parser.add_argument("--device", type=str, default="1")
parser.add_argument("--is_train", type=bool, default=True)
parser.add_argument("--test_freq", type=int, default=1)

# def setup_seed(seed):
#     random.seed(seed)  # 设置Python内置的随机数生成器种子
#     np.random.seed(seed)  # 设置NumPy随机数生成器种子
#     os.environ['PYTHONHASHSEED'] = str(args.seed)
#     torch.manual_seed(seed)  # 设置PyTorch CPU随机数生成器种子
#     torch.cuda.manual_seed(seed)  # 设置PyTorch GPU随机数生成器种子
#     torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机数生成器种子（如果有多个GPU）
#     torch.backends.cudnn.deterministic = True  # 确保每次调用返回相同的结果
#     torch.backends.cudnn.benchmark = False  # 关闭优化

args = parser.parse_args()

# setup_seed(args.seed)

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_loader, test_loader= make_dataloader(args)

model = Proposed(args.dataset_name).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

criterion = FocalLoss()

if args.is_train:

    train(args, model, optimizer, criterion, train_loader, test_loader, device, scheduler)

else:
    model.load_state_dict(
        torch.load(os.path.join(args.saving_path, 'model_best.pth')))
    test_loss, results=validation(model, criterion, test_loader, device)
    aa, oa, kappa = results['AA'], results['OA'], results['Kappa']
    print(f'OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}')
