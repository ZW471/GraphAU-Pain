"""Evaluate SynPAIN pretrain models with demographic breakdown.

FM checkpoints contain the full model and are evaluated directly.
HL checkpoints contain only backbone weights (classifier stripped), so we
do a linear probe: freeze backbone, train a fresh classifier for 3 epochs
on the SynPAIN train set, then evaluate on val.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python eval_synpain_demographics.py
"""

import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_synpain import SynPAINSingle
from eval_demographics import evaluate_by_demographics
from model.ANFL import BackboneOnlyPain, FullPictureMEFARGGeneric
from utils import load_state_dict, image_train, image_test

# ── Configuration ────────────────────────────────────────────────────
DATA_ROOT = "data/SynPAIN"
METADATA  = "metadata.csv"
RESULTS   = "results/run10"
OUTPUT    = "summary/run10-splitfix/demographics"
PROBE_EPOCHS = 3

MODELS = {
    "hl_full_r50": dict(
        cls=BackboneOnlyPain, backbone="resnet50", crop=172,
        ckpt=f"{RESULTS}/hl_full_r50/bs_32_seed_0_lr_1e-05/best_model_fold1.pth",
        backbone_only=True,
    ),
    "hl_full_swin": dict(
        cls=BackboneOnlyPain, backbone="swin_transformer_base", crop=224,
        ckpt=f"{RESULTS}/hl_full_swin/bs_32_seed_0_lr_1e-05/best_model_fold1.pth",
        backbone_only=True,
    ),
    "fm_full_r50": dict(
        cls=FullPictureMEFARGGeneric, backbone="resnet50", crop=172,
        ckpt=f"{RESULTS}/fm_full_r50/bs_32_seed_0_lr_1e-05/best_model_fold1.pth",
        backbone_only=False,
    ),
    "fm_full_swin": dict(
        cls=FullPictureMEFARGGeneric, backbone="swin_transformer_base", crop=224,
        ckpt=f"{RESULTS}/fm_full_swin/bs_32_seed_0_lr_1e-05/best_model_fold1.pth",
        backbone_only=False,
    ),
}

DEMO_KEYS = ("gender", "age_group")


# ── Load demographics from CSV (aligned with SynPAINSingle order) ───
def load_demographics(root, metadata_file, split):
    meta_path = os.path.join(root, metadata_file)
    demos = []
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != split:
                continue
            demos.append({
                "gender": row.get("gender", "unknown").strip(),
                "age_group": row.get("age_group", "unknown").strip(),
            })
    return demos


def linear_probe(model, device, crop_size):
    """Freeze backbone, train classifier on SynPAIN train for a few epochs."""
    # Freeze everything except classifier
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Linear probe: {trainable} trainable params, {PROBE_EPOCHS} epochs")

    train_ds = SynPAINSingle(
        root_path=DATA_ROOT, split="train", metadata_file=METADATA,
        transform=image_train(crop_size=crop_size), crop_size=crop_size,
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(PROBE_EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            gt = labels.argmax(dim=1)
            logits = model(imgs)
            loss = criterion(logits, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == gt).sum().item()
            total += imgs.size(0)
        print(f"  Probe epoch {ep+1}/{PROBE_EPOCHS}: loss={total_loss/total:.4f} acc={correct/total:.4f}")

    # Unfreeze for clean state
    for p in model.parameters():
        p.requires_grad = True


def evaluate_model(model, device, crop_size, demographics_list):
    """Run inference on SynPAIN val and return labels/scores/preds."""
    transform_test = image_test(crop_size=crop_size)
    ds = SynPAINSingle(
        root_path=DATA_ROOT, split="val", metadata_file=METADATA,
        transform=transform_test, crop_size=crop_size,
    )
    assert len(ds) == len(demographics_list)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    model.eval()
    all_labels, all_scores, all_preds = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            gt = labels.argmax(dim=1)
            all_labels.extend(gt.cpu().tolist())
            all_scores.extend(probs[:, 1].cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    return all_labels, all_scores, all_preds


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_demographics = load_demographics(DATA_ROOT, METADATA, "val")
    print(f"Val samples: {len(val_demographics)}")

    all_results = {}

    for name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}")
        print(f"{'='*60}")

        # Build model — use num_classes=10 to match checkpoint AU head size
        model = cfg["cls"](
            num_classes=10, backbone=cfg["backbone"],
            neighbor_num=4, metric="dots", binary=True,
        )
        load_state_dict(model, cfg["ckpt"])
        model = model.to(device)

        if cfg["backbone_only"]:
            # HL checkpoint: backbone only → train linear classifier
            linear_probe(model, device, cfg["crop"])

        all_labels, all_scores, all_preds = evaluate_model(
            model, device, cfg["crop"], val_demographics,
        )

        demo_dict = {key: [d[key] for d in val_demographics] for key in DEMO_KEYS}
        results = evaluate_by_demographics(
            all_labels, all_scores, all_preds,
            all_demographics=demo_dict,
            demo_keys=DEMO_KEYS,
            output_dir=OUTPUT,
            prefix=f"{name}_",
        )
        all_results[name] = results

    # Save combined results
    combined_path = os.path.join(OUTPUT, "all_demographics.json")
    def nan_to_none(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: nan_to_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [nan_to_none(v) for v in obj]
        return obj

    with open(combined_path, "w") as f:
        json.dump(nan_to_none(all_results), f, indent=2)
    print(f"\nCombined results saved → {combined_path}")


if __name__ == "__main__":
    main()
