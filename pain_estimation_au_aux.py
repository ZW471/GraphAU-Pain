"""Pain estimation with AU auxiliary loss.

Two-stage usage:

  Stage 1 — SynPAIN backbone pretrain (headless):
    python pain_estimation_au_aux.py --dataset SynPAIN --arc resnet50 \
        --crop-size 172 --epochs 10 -b 32 -lr 1e-5 --fold 1 \
        --binary True --num_classes 10 --neighbor_num 4 \
        --exp-name run9/synpain_bb_r50

    Trains BackboneOnlyPain on binary pain.  The checkpoint saves only
    backbone weights (classifier stripped) so downstream load_state_dict
    populates only the backbone of the full model.

  Stage 2 — UNBC finetune with AU aux:
    python pain_estimation_au_aux.py --dataset UNBC --arc resnet50 \
        --crop-size 172 --epochs 20 -b 64 -lr 1e-4 --fold 1 \
        --num_classes 10 --neighbor_num 4 --label_path original_unbc \
        --lam 1.0 --resume results/run9/synpain_bb_r50/.../best_model_fold1.pth \
        --exp-name run9/unbc_au_aux_r50

    Trains FullPictureMEFARGAuAux (backbone-agnostic) with:
      loss = pain_loss + lam * au_loss
    Pain loss: WeightedCrossEntropyLoss on 3-class PSPI labels.
    AU loss:   WeightedAsymmetricLoss on multi-label AU binary labels.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import logging
import random
from PIL import Image
from collections import OrderedDict

from model.ANFL import FullPictureMEFARGAuAux, BackboneOnlyPain
from dataset_synpain import SynPAINSingle
from utils import *
from conf import get_config, set_logger, set_outdir, set_env


# ---------------------------------------------------------------------------
# Dataset: UNBC with both AU labels (stage 1) and pain labels (stage 3)
# ---------------------------------------------------------------------------

def _pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class UNBCAuPain(Dataset):
    """UNBC loader that returns (img, au_label, pain_label) for joint training."""

    def __init__(self, root_path, train=True, fold=1, transform=None,
                 crop_size=172, label_path='', loader=_pil_loader):
        self._train = train
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(
            root_path, 'img' if crop_size == 172 else 'resized_img')

        list_dir = (os.path.join(root_path, 'list', label_path)
                    if label_path else os.path.join(root_path, 'list'))
        split = 'train' if train else 'test'

        img_path = os.path.join(list_dir, f'UNBC_{split}_img_path_fold{fold}.txt')
        au_path = os.path.join(list_dir, f'UNBC_{split}_label_fold{fold}.txt')
        pain_path = os.path.join(list_dir, f'UNBC_{split}_pspi_fold{fold}.txt')

        self.image_list = open(img_path).readlines()
        self.au_labels = np.loadtxt(au_path)
        self.pain_labels = np.loadtxt(pain_path)
        assert len(self.image_list) == len(self.au_labels) == len(self.pain_labels)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = self.loader(os.path.join(self.img_folder_path,
                                       self.image_list[index].strip()))
        if self._train:
            w, h = img.size
            offset_y = random.randint(0, h - self.crop_size)
            offset_x = random.randint(0, w - self.crop_size)
            flip = random.randint(0, 1)
            if self._transform is not None:
                img = self._transform(img, flip, offset_x, offset_y)
        else:
            if self._transform is not None:
                img = self._transform(img)
        return img, self.au_labels[index], self.pain_labels[index]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def get_dataloader(conf):
    print('==> Preparing data...')
    if conf.dataset == 'UNBC':
        label_path = getattr(conf, 'label_path', '')
        trainset = UNBCAuPain(conf.dataset_path, train=True, fold=conf.fold,
                              transform=image_train(crop_size=conf.crop_size),
                              crop_size=conf.crop_size, label_path=label_path)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size,
                                  shuffle=True, num_workers=conf.num_workers)
        valset = UNBCAuPain(conf.dataset_path, train=False, fold=conf.fold,
                            transform=image_test(crop_size=conf.crop_size),
                            label_path=label_path)
        val_loader = DataLoader(valset, batch_size=conf.batch_size,
                                shuffle=False, num_workers=conf.num_workers)
    elif conf.dataset == 'SynPAIN':
        metadata_file = getattr(conf, 'metadata_file', 'metadata.csv')
        trainset = SynPAINSingle(conf.dataset_path, split='train',
                                 metadata_file=metadata_file,
                                 transform=image_train(crop_size=conf.crop_size),
                                 crop_size=conf.crop_size)
        train_loader = DataLoader(trainset, batch_size=conf.batch_size,
                                  shuffle=True, num_workers=conf.num_workers)
        valset = SynPAINSingle(conf.dataset_path, split='val',
                               metadata_file=metadata_file,
                               transform=image_test(crop_size=conf.crop_size),
                               crop_size=conf.crop_size)
        val_loader = DataLoader(valset, batch_size=conf.batch_size,
                                shuffle=False, num_workers=conf.num_workers)
    else:
        raise ValueError(f'Unsupported dataset: {conf.dataset}')
    return train_loader, val_loader, len(trainset), len(valset)


# ---------------------------------------------------------------------------
# AU weight computation (mirrors tool/UNBC_calculate_AU_class_weights.py)
# ---------------------------------------------------------------------------

def compute_au_weights(au_labels):
    """Compute inverse-frequency AU class weights from label matrix [N, C]."""
    occur_rate = (au_labels > 0).mean(axis=0).clip(min=1e-10)
    w = 1.0 / occur_rate
    w = w / w.sum() * len(occur_rate)
    return torch.from_numpy(w).float()


# ---------------------------------------------------------------------------
# Train / Val — UNBC (dual output)
# ---------------------------------------------------------------------------

def train_unbc(conf, net, train_loader, optimizer, epoch,
               pain_criterion, au_criterion, au_loss_weight):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs, au_targets, pain_targets) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs,
                             conf.learning_rate, batch_idx, train_loader_len)
        au_targets = au_targets.float()
        pain_targets = pain_targets.float()
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            au_targets = au_targets.cuda()
            pain_targets = pain_targets.cuda()
        optimizer.zero_grad()
        pain_logits, au_logits = net(inputs)
        pain_loss = pain_criterion(pain_logits, pain_targets)
        au_loss = au_criterion(au_logits, au_targets)
        loss = pain_loss + au_loss_weight * au_loss
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg


def val_unbc(net, val_loader, pain_criterion, au_criterion, au_loss_weight):
    losses = AverageMeter()
    net.eval()
    pain_stats = None
    au_stats = None
    for batch_idx, (inputs, au_targets, pain_targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            au_targets = au_targets.float()
            pain_targets = pain_targets.float()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                au_targets = au_targets.cuda()
                pain_targets = pain_targets.cuda()
            pain_logits, au_logits = net(inputs)
            pain_loss = pain_criterion(pain_logits, pain_targets)
            au_loss = au_criterion(au_logits, au_targets)
            loss = pain_loss + au_loss_weight * au_loss
            losses.update(loss.data.item(), inputs.size(0))
            # Pain metrics (softmax multi-class)
            update_list = statistics_softmax(pain_logits, pain_targets.detach())
            pain_stats = update_statistics_list(pain_stats, update_list)
            # AU metrics (multi-label binary, threshold 0.5)
            update_list = statistics(au_logits, au_targets.detach(), 0.5)
            au_stats = update_statistics_list(au_stats, update_list)
    pain_f1, pain_f1_list = calc_f1_score(pain_stats)
    pain_acc, pain_acc_list = calc_acc(pain_stats)
    au_f1, au_f1_list = calc_f1_score(au_stats)
    au_acc, au_acc_list = calc_acc(au_stats)
    return (losses.avg,
            pain_f1, pain_f1_list, pain_acc, pain_acc_list,
            au_f1, au_f1_list, au_acc, au_acc_list)


# ---------------------------------------------------------------------------
# Train / Val — SynPAIN (single output, backbone-only)
# ---------------------------------------------------------------------------

def train_synpain(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    net.train()
    train_loader_len = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        adjust_learning_rate(optimizer, epoch, conf.epochs,
                             conf.learning_rate, batch_idx, train_loader_len)
        targets = targets.float()
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), inputs.size(0))
    return losses.avg


def val_synpain(net, val_loader, criterion):
    losses = AverageMeter()
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            targets = targets.float()
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.data.item(), inputs.size(0))
            update_list = statistics_softmax(outputs, targets.detach())
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1, f1_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return losses.avg, mean_f1, f1_list, mean_acc, acc_list


# ---------------------------------------------------------------------------
# Backbone-only checkpoint saving (strips classifier head)
# ---------------------------------------------------------------------------

def save_backbone_only(net, epoch, optimizer, path):
    """Save checkpoint containing only backbone weights (no classifier)."""
    raw_sd = net.state_dict()
    bb_sd = OrderedDict()
    for k, v in raw_sd.items():
        clean = k.replace('module.', '')
        if clean.startswith('backbone.'):
            bb_sd[clean] = v
    checkpoint = {
        'epoch': epoch,
        'state_dict': bb_sd,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(conf):
    start_epoch = 0
    train_loader, val_loader, train_data_num, val_data_num = get_dataloader(conf)
    logging.info("Fold: [{} | {}  val_data_num: {} ]".format(
        conf.fold, conf.N_fold, val_data_num))

    au_loss_weight = conf.lam  # --lam flag controls AU auxiliary weight

    if conf.dataset == 'SynPAIN':
        # ----- SynPAIN backbone pretrain (headless) -----
        dataset_info = UNBC_pain_infolist_binary
        net = BackboneOnlyPain(num_classes=conf.num_classes, backbone=conf.arc,
                               neighbor_num=conf.neighbor_num, metric=conf.metric,
                               binary=conf.binary)
        # Pain weights (inverse-frequency from train split)
        train_labels = np.array([s['label'] for s in train_loader.dataset.data_list])
        n_pos = float((train_labels == 1).sum())
        n_neg = float((train_labels == 0).sum())
        total = max(1.0, n_pos + n_neg)
        pain_weight = torch.tensor([total / (2 * max(1.0, n_neg)),
                                    total / (2 * max(1.0, n_pos))], dtype=torch.float32)

        if conf.resume != '':
            logging.info("Resume from | {} ]".format(conf.resume))
            net = load_state_dict(net, conf.resume)
        if torch.cuda.is_available():
            net = nn.DataParallel(net).cuda()

        criterion = WeightedCrossEntropyLoss(weight=pain_weight)
        optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999),
                                lr=conf.learning_rate, weight_decay=conf.weight_decay)
        print('init lr:', conf.learning_rate)

        best_val_f1 = -1.0
        for epoch in range(start_epoch, conf.epochs):
            lr = optimizer.param_groups[0]['lr']
            logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
            train_loss = train_synpain(conf, net, train_loader, optimizer, epoch, criterion)
            val_loss, val_f1, val_f1_list, val_acc, val_acc_list = val_synpain(
                net, val_loader, criterion)

            logging.info({'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  '
                          'val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                          .format(epoch + 1, train_loss, val_loss,
                                  100. * val_f1, 100. * val_acc)})
            logging.info({'F1-score-list:'})
            logging.info(dataset_info(val_f1_list))
            logging.info({'Acc-list:'})
            logging.info(dataset_info(val_acc_list))

            # Save backbone-only checkpoint (strip classifier)
            save_backbone_only(net, epoch, optimizer,
                               os.path.join(conf['outdir'],
                                            f'epoch{epoch+1}_model_fold{conf.fold}.pth'))
            save_backbone_only(net, epoch, optimizer,
                               os.path.join(conf['outdir'],
                                            f'cur_model_fold{conf.fold}.pth'))
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                save_backbone_only(net, epoch, optimizer,
                                   os.path.join(conf['outdir'],
                                                f'best_model_fold{conf.fold}.pth'))

    elif conf.dataset == 'UNBC':
        # ----- UNBC finetune with AU auxiliary loss -----
        if conf.binary:
            pain_info = UNBC_pain_infolist_binary
        else:
            pain_info = UNBC_pain_infolist

        label_path = getattr(conf, 'label_path', '')
        num_au = conf.num_classes
        # Determine if DISFA-labeled (8-AU) or original_unbc (10-AU)
        use_disfa = (num_au == 8)

        net = FullPictureMEFARGAuAux(num_classes=num_au, backbone=conf.arc,
                                      neighbor_num=conf.neighbor_num,
                                      metric=conf.metric, binary=conf.binary)
        if conf.resume != '':
            logging.info("Resume from | {} ]".format(conf.resume))
            net = load_state_dict(net, conf.resume)

        # Pain weights
        weight_dir = (os.path.join(conf.dataset_path, 'list', label_path)
                      if label_path else os.path.join(conf.dataset_path, 'list'))
        pain_weight = torch.from_numpy(np.loadtxt(
            os.path.join(weight_dir, f'{conf.dataset}_pspi_w_fold{conf.fold}.txt')))

        # AU weights — load file if exists, else compute from labels
        au_weight_path = os.path.join(weight_dir, f'{conf.dataset}_weight_fold{conf.fold}.txt')
        if os.path.exists(au_weight_path):
            au_weight = torch.from_numpy(np.loadtxt(au_weight_path))
        else:
            logging.info(f"AU weight file not found at {au_weight_path}, computing from labels")
            au_weight = compute_au_weights(train_loader.dataset.au_labels)
        logging.info(f"AU loss weight (lam): {au_loss_weight}")

        if torch.cuda.is_available():
            net = nn.DataParallel(net).cuda()
            au_weight = au_weight.cuda()

        pain_criterion = WeightedCrossEntropyLoss(weight=pain_weight)
        au_criterion = WeightedAsymmetricLoss(weight=au_weight)
        optimizer = optim.AdamW(net.parameters(), betas=(0.9, 0.999),
                                lr=conf.learning_rate, weight_decay=conf.weight_decay)
        print('init lr:', conf.learning_rate, ' au_loss_weight:', au_loss_weight)

        best_val_f1 = -1.0
        for epoch in range(start_epoch, conf.epochs):
            lr = optimizer.param_groups[0]['lr']
            logging.info("Epoch: [{} | {} LR: {} ]".format(epoch + 1, conf.epochs, lr))
            train_loss = train_unbc(conf, net, train_loader, optimizer, epoch,
                                    pain_criterion, au_criterion, au_loss_weight)
            (val_loss,
             val_pain_f1, val_pain_f1_list, val_pain_acc, val_pain_acc_list,
             val_au_f1, val_au_f1_list, val_au_acc, val_au_acc_list
             ) = val_unbc(net, val_loader, pain_criterion, au_criterion, au_loss_weight)

            logging.info({'Epoch:  {}   train_loss: {:.5f}  val_loss: {:.5f}  '
                          'val_mean_f1_score {:.2f},val_mean_acc {:.2f}'
                          .format(epoch + 1, train_loss, val_loss,
                                  100. * val_pain_f1, 100. * val_pain_acc)})
            logging.info({'Pain F1-score-list:'})
            logging.info(pain_info(val_pain_f1_list))
            logging.info({'Pain Acc-list:'})
            logging.info(pain_info(val_pain_acc_list))
            logging.info({'AU F1-score-list (mean {:.2f}):'.format(100. * val_au_f1)})
            logging.info(UNBC_infolist(val_au_f1_list, use_disfa=use_disfa))
            logging.info({'AU Acc-list (mean {:.2f}):'.format(100. * val_au_acc)})
            logging.info(UNBC_infolist(val_au_acc_list, use_disfa=use_disfa))

            checkpoint = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(
                conf['outdir'], f'epoch{epoch+1}_model_fold{conf.fold}.pth'))
            torch.save(checkpoint, os.path.join(
                conf['outdir'], f'cur_model_fold{conf.fold}.pth'))

            # Track best by pain F1 (primary objective)
            if val_pain_f1 > best_val_f1:
                best_val_f1 = val_pain_f1
                torch.save(checkpoint, os.path.join(
                    conf['outdir'], f'best_model_fold{conf.fold}.pth'))


if __name__ == "__main__":
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
