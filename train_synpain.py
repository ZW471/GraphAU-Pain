"""
SynPAIN Pretraining Script
==========================
Trains DeltaMEFARG on SynPAIN using pair-input delta-graph modeling.
Optionally runs demographic bias evaluation after each epoch.

Usage:
    python train_synpain.py \
        --dataset SynPAIN \
        --training_stage synpain_pretrain \
        --use_pair_input \
        --use_delta_graph \
        --arc swin_transformer_base \
        --epochs 20 \
        --batch-size 32 \
        --learning-rate 1e-5 \
        --exp-name synpain_delta \
        [--resume path/to/disfa_stage1.pth] \
        [--eval_by_group] \
        [--save_demographics]

UNBC single-frame finetuning is unaffected — run pain_estimation_full.py as before.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from conf import get_config, set_env, set_logger, set_outdir
from dataset_synpain import SynPAIN, synpain_collate_fn
from eval_demographics import evaluate_by_demographics
from model.ANFL import DeltaMEFARG
from utils import AverageMeter, adjust_learning_rate, load_state_dict


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------

def get_dataloader(conf):
    print('==> Preparing SynPAIN data...')
    metadata_file = getattr(conf, 'metadata_file', 'metadata.csv')

    trainset = SynPAIN(
        root_path=conf.dataset_path,
        split='train',
        metadata_file=metadata_file,
        crop_size=conf.crop_size,
    )
    valset = SynPAIN(
        root_path=conf.dataset_path,
        split='val',
        metadata_file=metadata_file,
        crop_size=conf.crop_size,
    )
    train_loader = DataLoader(
        trainset,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        collate_fn=synpain_collate_fn,
    )
    val_loader = DataLoader(
        valset,
        batch_size=conf.batch_size,
        shuffle=False,
        num_workers=conf.num_workers,
        collate_fn=synpain_collate_fn,
    )
    return train_loader, val_loader, len(trainset), len(valset)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(conf, net, train_loader, optimizer, epoch, criterion):
    losses = AverageMeter()
    net.train()
    loader_len = len(train_loader)

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch+1}")):
        adjust_learning_rate(
            optimizer, epoch, conf.epochs, conf.learning_rate, batch_idx, loader_len
        )

        x_neu  = batch['x_neu'].cuda()  if torch.cuda.is_available() else batch['x_neu']
        x_expr = batch['x_expr'].cuda() if torch.cuda.is_available() else batch['x_expr']
        labels = batch['label'].cuda()  if torch.cuda.is_available() else batch['label']

        optimizer.zero_grad()
        # DeltaMEFARG forward: pair input → pain logits [B, num_pain_classes]
        outputs = net(x_expr, x_neu)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), x_neu.size(0))

    return losses.avg


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def val(conf, net, val_loader, criterion, epoch):
    """
    Validate model. Collects predictions for overall and per-group metrics.

    Returns:
        val_loss (float)
        val_acc  (float)
        val_f1   (float)
        demo_results (dict | None)  — populated when eval_by_group=True
    """
    losses = AverageMeter()
    net.eval()

    all_labels = []
    all_scores = []   # positive-class probabilities for AUROC
    all_preds  = []

    # Demographic buffers (populated only when eval_by_group is set)
    all_demo = {'age_group': [], 'gender': [], 'ethnicity': []}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val epoch {epoch+1}"):
            x_neu  = batch['x_neu'].cuda()  if torch.cuda.is_available() else batch['x_neu']
            x_expr = batch['x_expr'].cuda() if torch.cuda.is_available() else batch['x_expr']
            labels = batch['label'].cuda()  if torch.cuda.is_available() else batch['label']

            outputs = net(x_expr, x_neu)  # [B, num_pain_classes]
            loss = criterion(outputs, labels)
            losses.update(loss.item(), x_neu.size(0))

            probs      = torch.softmax(outputs, dim=-1)      # [B, num_pain_classes]
            pos_scores = probs[:, 1].cpu().numpy()           # probability of pain (class 1)
            preds      = outputs.argmax(dim=-1).cpu().numpy()
            labels_np  = labels.cpu().numpy()

            all_labels.extend(labels_np.tolist())
            all_scores.extend(pos_scores.tolist())
            all_preds.extend(preds.tolist())

            if getattr(conf, 'eval_by_group', False):
                for key in all_demo:
                    all_demo[key].extend(batch.get(key, ['unknown'] * len(labels_np)))

    # Overall metrics
    labels_arr = np.array(all_labels)
    preds_arr  = np.array(all_preds)
    val_acc = float((labels_arr == preds_arr).mean())

    try:
        from sklearn.metrics import f1_score as sk_f1
        val_f1 = float(sk_f1(labels_arr, preds_arr, average='binary', zero_division=0))
    except ImportError:
        # Fallback manual F1 if sklearn not available
        tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
        fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
        fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
        val_f1 = 2 * tp / (2 * tp + fp + fn + 1e-20)

    # Demographic evaluation
    demo_results = None
    if getattr(conf, 'eval_by_group', False):
        save_dir = conf.get('outdir') if getattr(conf, 'save_demographics', False) else None
        demo_results = evaluate_by_demographics(
            all_labels, all_scores, all_preds,
            all_demographics=all_demo,
            output_dir=save_dir,
            prefix=f'epoch{epoch+1}_',
        )

    return losses.avg, val_acc, val_f1, demo_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(conf):
    # ---- Data ----
    train_loader, val_loader, train_n, val_n = get_dataloader(conf)
    logging.info(f"SynPAIN splits — train: {train_n}  val: {val_n}")

    # ---- Model ----
    num_pain_classes = getattr(conf, 'num_pain_classes', 2)
    use_delta_graph  = getattr(conf, 'use_delta_graph', True)

    net = DeltaMEFARG(
        num_classes=conf.num_classes,
        backbone=conf.arc,
        neighbor_num=conf.neighbor_num,
        metric=conf.metric,
        num_pain_classes=num_pain_classes,
        use_delta_graph=use_delta_graph,
    )

    # Optionally warm-start from a DISFA stage1 or stage2 checkpoint
    if conf.resume:
        logging.info(f"Loading pretrained weights from {conf.resume}")
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    # ---- Loss & optimiser ----
    # Plain cross-entropy: labels are class indices (not one-hot)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        net.parameters(),
        lr=conf.learning_rate,
        weight_decay=conf.weight_decay,
        betas=(0.9, 0.999),
    )
    logging.info(f"Initial LR: {conf.learning_rate}  | Use ΔGraph: {use_delta_graph}")

    # ---- Training loop ----
    best_val_f1 = 0.0

    for epoch in range(conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch [{epoch+1}/{conf.epochs}]  LR={lr:.2e}")

        train_loss = train(conf, net, train_loader, optimizer, epoch, criterion)
        val_loss, val_acc, val_f1, demo_results = val(
            conf, net, val_loader, criterion, epoch
        )

        logging.info(
            f"  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"val_acc={100.*val_acc:.2f}%  val_f1={100.*val_f1:.2f}%"
        )

        # Save checkpoint every epoch
        checkpoint = {
            'epoch':      epoch,
            'state_dict': net.state_dict(),
            'optimizer':  optimizer.state_dict(),
        }
        ckpt_path = os.path.join(conf['outdir'], f'epoch{epoch+1}_synpain.pth')
        torch.save(checkpoint, ckpt_path)

        # Track best model by val F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(checkpoint, os.path.join(conf['outdir'], 'best_synpain.pth'))
            logging.info(f"  ** New best val F1: {100.*best_val_f1:.2f}%")

    logging.info(f"Training complete. Best val F1: {100.*best_val_f1:.2f}%")


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    conf = get_config()
    set_env(conf)
    set_outdir(conf)
    set_logger(conf)
    main(conf)
