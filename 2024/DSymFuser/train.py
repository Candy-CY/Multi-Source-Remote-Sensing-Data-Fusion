import os
import math
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
import time

def train(args, model, optimizer, criterion, train_loader, test_loader, device, scheduler=None):

    best_acc = 0.
    for epoch in range(args.epoch):

        start_time = time.time()

        model.train()

        losses = AverageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_idx, (hsi, lidar, batch_target) in enumerate(train_loader):

            hsi, lidar, batch_target = hsi.to(device), lidar.to(device), batch_target.to(device)

            batch_out = model(hsi, lidar)
            loss = criterion(batch_out, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.use_self_scheduler:
                lr = step_learning_rate(args, epoch, optimizer)
                # lr = cosine_learning_rate(args, epoch, optimizer)
            else:
                scheduler.step()

            losses.update(loss.data, batch_target.shape[0])
            batch_pred = np.argmax(batch_out.detach().cpu().numpy(), axis=1)
            batch_target = batch_target.detach().cpu().numpy()

            tar = np.append(tar, batch_pred)
            pre = np.append(pre, batch_target)

        # train_loss = losses.avg
        # oa, pa, kappa, aa = output_metric(tar, pre)

        end_time = time.time()
        training_time = end_time - start_time
        # print(f"Training time for one epoch: {training_time} seconds")

        if (epoch % args.test_freq == 0) | (epoch == args.epoch - 1):
        # if ((epoch % args.test_freq == 0) | (epoch == args.epoch - 1)) & (epoch >= 50):

            test_loss, results = validation(model, criterion, test_loader, device)
            aa, oa, kappa = results['AA'], results['OA'], results['Kappa']

            is_best = oa >= best_acc
            best_acc = max(oa, best_acc)
            save_checkpoint(model, is_best, args.saving_path, args.dataset_name, epoch=epoch, acc=best_acc)

            print(f'Epoch: {epoch}, epoch_time: {training_time}, '
                  f'OA: {oa:.4f}, AA: {aa:.4f}, Kappa: {kappa:.4f}, best_acc: {best_acc:.4f}')



def validation(model, criterion, test_loader, device):

    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        tar = np.array([])
        pre = np.array([])
        for batch_idx, (hsi, lidar, batch_target) in enumerate(test_loader):

            hsi, lidar, batch_target = hsi.to(device), lidar.to(device), batch_target.to(device)

            batch_out = model(hsi, lidar)
            loss = criterion(batch_out, batch_target)

            losses.update(loss.data, batch_target.shape[0])
            batch_pred = np.argmax(batch_out.detach().cpu().numpy(), axis=1)
            batch_target = batch_target.detach().cpu().numpy()

            tar = np.append(tar, batch_pred)
            pre = np.append(pre, batch_target)

        total_loss = losses.avg
        results = compute_metrics(tar, pre)

    return total_loss, results


def save_checkpoint(network, is_best, saving_path, dataset_name, **kwargs):
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path, exist_ok=True)

    if is_best:
        tqdm.write("epoch = {epoch}: best validation OA = {acc:.4f}".format(**kwargs))
        torch.save(network.state_dict(), os.path.join(saving_path, dataset_name, 'model_best.pth'))
    else:  # save the ckpt for each 10 epoch
        if kwargs['epoch'] % 10 == 0:
            torch.save(network.state_dict(), os.path.join(saving_path, dataset_name, 'model.pth'))


class AverageMeter(object):

  def __init__(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def compute_metrics(pred, target):
    """Compute and print metrics (OA, PA, AA, Kappa)

    Args:
        pred: list of predicted labels
        target: list of target labels

    Returns:
        {Confusion Matrix, OA, PA, AA, Kappa}
    """

    results = {}

    # compute Overall Accuracy
    cm = confusion_matrix(target, pred)
    results['Confusion matrix'] = cm

    # compute Overall Accuracy (OA)
    oa = 1. * np.trace(cm) / np.sum(cm)
    results['OA'] = oa

    # compute Producer Accuracy (PA)
    n_classes = cm.shape[0]
    pa = np.array([1. * cm[i, i] / np.sum(cm[i, :]) for i in range(n_classes)])
    results['PA'] = pa

    # compute Average Accuracy (AA)
    aa = np.mean(pa)
    results['AA'] = aa

    # compute kappa coefficient
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(np.sum(cm) * np.sum(cm))
    kappa = (oa - pe) / (1 - pe)
    results['Kappa'] = kappa

    return results


def step_learning_rate(args, epoch, optimizer):
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        # lr_adj = (epoch + 1) / warm_epochs
        lr_adj = 5.
    elif epoch < int(0.4 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.7 * total_epochs):
        lr_adj = 0.1
    elif epoch < int(1 * total_epochs):
        lr_adj = 0.01
    else:
        lr_adj = 0.

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


def cosine_learning_rate(args, epoch, optimizer):
    """Cosine Learning rate
    """
    total_epochs = args.epoch
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = 1.
    else:
        lr_adj = 1 / 2 * (1 + math.cos(math.pi * epoch / (total_epochs - warm_epochs)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj


