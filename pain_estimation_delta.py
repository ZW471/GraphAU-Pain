"""
pain_estimation_delta.py
Test whether DeltaMEFARG (ΔGraph) improves pain estimation on UNBC+DISFA.

Pipeline:
  - Optionally resume from a DISFA/UNBC AU-pretrained checkpoint (--resume).
  - Build per-subject neutral-frame pairs from the UNBC pspi split files.
  - Fine-tune DeltaMEFARG on those pairs and evaluate with the same F1/acc
    metrics as pain_estimation_full.py.

Usage:
    python pain_estimation_delta.py --dataset UNBC --fold 1 \\
        --resume checkpoints/disfa_pretrained.pth \\
        --use_delta_graph \\
        --exp-name delta_graph_test

Ablation (graph-level PE-score delta instead of node-level):
    python pain_estimation_delta.py ... (omit --use_delta_graph)
"""

import os
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

from model.ANFL import DeltaMEFARG
from dataset import default_loader
from utils import (
    AverageMeter, adjust_learning_rate,
    image_train, image_test,
    statistics_softmax, update_statistics_list, calc_f1_score, calc_acc,
    load_state_dict, WeightedCrossEntropyLoss,
    UNBC_pain_infolist, UNBC_pain_infolist_binary,
)
from conf import get_config, set_logger, set_outdir, set_env


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UNBCDelta(Dataset):
    """
    UNBC pain dataset with per-subject neutral frame pairing.

    Each __getitem__ returns (img_expr, img_neu, label) where:
      img_expr  — the expressive frame at this index (training augmentation)
      img_neu   — the subject's neutral reference (test-time transform, fixed)
      label     — float tensor of the pspi label (one-hot or scalar)

    Neutral selection strategy (per subject, chosen once at __init__):
      1. First frame in the split whose pspi class == 0 (no pain).
      2. If none, first frame with class == 1.
      3. If still none, first frame available.
    """

    def __init__(self, root_path, train=True, fold=1,
                 train_transform=None, test_transform=None,
                 crop_size=224, loader=default_loader):
        self._root_path = root_path
        self._train = train
        self._train_tf = train_transform
        self._test_tf = test_transform
        self.crop_size = crop_size
        self.loader = loader
        # Match UNBC convention: 172 → img/, else resized_img/
        img_dir = 'img' if crop_size == 172 else 'resized_img'
        self.img_folder = os.path.join(root_path, img_dir)

        split = 'train' if train else 'test'
        img_list_path = os.path.join(
            root_path, 'list',
            f'UNBC_{split}_img_path_fold{fold}.txt')
        label_path = os.path.join(
            root_path, 'list',
            f'UNBC_{split}_pspi_fold{fold}.txt')

        raw_paths = [l.strip() for l in open(img_list_path).readlines()]
        labels_raw = np.loadtxt(label_path)   # [N] or [N, C]

        # Derive scalar class index for neutral selection
        if labels_raw.ndim == 2:
            label_classes = np.argmax(labels_raw, axis=1).astype(int)
        else:
            label_classes = labels_raw.astype(int)

        # Group by subject (first directory component of path)
        subj_ids = [p.split('/')[0] for p in raw_paths]
        subj_to_indices: dict[str, list[int]] = {}
        for i, s in enumerate(subj_ids):
            subj_to_indices.setdefault(s, []).append(i)

        # Pick one neutral index per subject
        neutral_idx: dict[str, int] = {}
        for subj, idxs in subj_to_indices.items():
            chosen = None
            for target_cls in range(int(label_classes.max()) + 1):
                candidates = [i for i in idxs if label_classes[i] == target_cls]
                if candidates:
                    chosen = candidates[0]
                    break
            neutral_idx[subj] = chosen if chosen is not None else idxs[0]

        self.raw_paths = raw_paths
        self.labels_raw = labels_raw
        self.label_classes = label_classes
        self.subj_ids = subj_ids
        self.neutral_idx = neutral_idx
        # Number of pain classes — used for on-the-fly one-hot if labels are 1D
        self._num_pain_cls = int(label_classes.max()) + 1

    def __len__(self):
        return len(self.raw_paths)

    def __getitem__(self, index):
        expr_path = self.raw_paths[index]
        subj = self.subj_ids[index]
        neu_path = self.raw_paths[self.neutral_idx[subj]]

        img_expr = self.loader(os.path.join(self.img_folder, expr_path))
        img_neu = self.loader(os.path.join(self.img_folder, neu_path))

        label = self.labels_raw[index]
        # Ensure one-hot vector for WeightedCrossEntropyLoss / statistics_softmax
        if self.labels_raw.ndim == 1:
            oh = np.zeros(self._num_pain_cls, dtype=np.float32)
            oh[int(label)] = 1.0
            label = oh

        if self._train:
            w, h = img_expr.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._train_tf is not None:
                img_expr = self._train_tf(img_expr, flip, offset_x, offset_y)
            # Neutral: fixed test-time transform (no random crop/flip)
            if self._test_tf is not None:
                img_neu = self._test_tf(img_neu)
        else:
            if self._test_tf is not None:
                img_expr = self._test_tf(img_expr)
                img_neu = self._test_tf(img_neu)

        return img_expr, img_neu, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloader(conf):
    print('==> Preparing data...')
    assert conf.dataset == 'UNBC', f'UNBCDelta only supports UNBC, got {conf.dataset}'

    train_tf = image_train(crop_size=conf.crop_size)
    test_tf = image_test(crop_size=conf.crop_size)

    trainset = UNBCDelta(
        conf.dataset_path, train=True, fold=conf.fold,
        train_transform=train_tf, test_transform=test_tf,
        crop_size=conf.crop_size)
    valset = UNBCDelta(
        conf.dataset_path, train=False, fold=conf.fold,
        train_transform=train_tf, test_transform=test_tf,
        crop_size=conf.crop_size)

    train_loader = DataLoader(
        trainset, batch_size=conf.batch_size, shuffle=True,
        num_workers=conf.num_workers)
    val_loader = DataLoader(
        valset, batch_size=conf.batch_size, shuffle=False,
        num_workers=conf.num_workers)

    return train_loader, val_loader, len(trainset), len(valset)


# ---------------------------------------------------------------------------
# Train / Val loops
# ---------------------------------------------------------------------------

def train(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    net.train()
    loader_len = len(train_loader)
    for batch_idx, (x_expr, x_neu, targets) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs,
                             conf.learning_rate, batch_idx, loader_len)
        targets = targets.float()
        if torch.cuda.is_available():
            x_expr, x_neu, targets = x_expr.cuda(), x_neu.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(x_expr, x_neu)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), x_expr.size(0))
    return losses.avg


def val(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (x_expr, x_neu, targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                x_expr, x_neu, targets = x_expr.cuda(), x_neu.cuda(), targets.cuda()
            outputs = net(x_expr, x_neu)
            loss = criterion(outputs, targets)
            losses.update(loss.data.item(), x_expr.size(0))
            update_list = statistics_softmax(outputs, targets.detach())
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1, f1_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1, f1_list, mean_acc, acc_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(conf):
    num_pain_classes = 2 if conf.binary else 3
    dataset_info = UNBC_pain_infolist_binary if conf.binary else UNBC_pain_infolist

    train_loader, val_loader, train_n, val_n = get_dataloader(conf)
    logging.info(f'Fold: [{conf.fold} | {conf.N_fold}  val_data_num: {val_n}]')

    # Class weights (same file as pain_estimation_full.py)
    weight_path = os.path.join(
        conf.dataset_path, 'list',
        f'{conf.dataset}_pspi_w_fold{conf.fold}.txt')
    train_weight = torch.from_numpy(np.loadtxt(weight_path))

    # Model: shared backbone+head architecture, new delta classifier
    net = DeltaMEFARG(
        num_classes=conf.num_classes,          # AU nodes (8 for DISFA/UNBC)
        backbone=conf.arc,
        neighbor_num=conf.neighbor_num,
        metric=conf.metric,
        num_pain_classes=num_pain_classes,
        use_delta_graph=conf.use_delta_graph,  # node-level vs PE-score delta
    )

    if conf.resume != '':
        logging.info(f'Resume from | {conf.resume} |')
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    criterion = WeightedCrossEntropyLoss(weight=train_weight)
    optimizer = optim.AdamW(
        net.parameters(), betas=(0.9, 0.999),
        lr=conf.learning_rate, weight_decay=conf.weight_decay)
    logging.info(f'Init LR: {conf.learning_rate}  use_delta_graph: {conf.use_delta_graph}')

    for epoch in range(conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch: [{epoch + 1} | {conf.epochs}  LR: {lr}]')

        train_loss = train(conf, net, train_loader, optimizer, epoch, criterion)
        val_loss, val_f1, val_f1_list, val_acc, val_acc_list = val(
            net, val_loader, criterion)

        logging.info(
            f'Epoch: {epoch + 1}  train_loss: {train_loss:.5f}  '
            f'val_loss: {val_loss:.5f}  '
            f'val_mean_f1: {100. * val_f1:.2f}  '
            f'val_mean_acc: {100. * val_acc:.2f}')
        logging.info({'F1-score-list:'})
        logging.info(dataset_info(val_f1_list))
        logging.info({'Acc-list:'})
        logging.info(dataset_info(val_acc_list))

        # Save every epoch + keep a rolling "current" checkpoint
        ckpt = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(
            conf['outdir'], f'epoch{epoch + 1}_model_fold{conf.fold}.pth'))
        torch.save(ckpt, os.path.join(
            conf['outdir'], f'cur_model_fold{conf.fold}.pth'))


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
