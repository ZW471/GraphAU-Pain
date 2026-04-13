# Pain Estimation with AU Auxiliary Loss & SynPAIN Pretraining

## Overview

This project trains pain estimation models on UNBC with Action Unit (AU)
classification as an auxiliary loss. Optionally, the backbone (or full model)
can be pretrained on SynPAIN before UNBC fine-tuning.

## How AU Auxiliary Loss Works

The standard pain estimation pipeline feeds an image through:

```
Image -> Backbone -> AU GNN Head -> Pain Classifier -> pain_logits
```

The AU GNN head produces per-AU logits as an intermediate step, but in the
standard setup only the final pain classification loss backpropagates through
the whole network. The AU node representations are learned indirectly.

With AU auxiliary loss, we **directly supervise the AU GNN head** using
ground-truth AU binary labels from UNBC:

```
loss = pain_loss + lam * au_loss
```

- `pain_loss`: WeightedCrossEntropyLoss on 3-class PSPI labels (no-pain / mild / pain)
- `au_loss`: WeightedAsymmetricLoss on multi-label AU binary labels
- `lam`: weight controlling AU loss contribution (default 1.0; set to 0 to disable)

The model class `FullPictureMEFARGAuAux` returns `(pain_logits, au_logits)` so
both losses can be computed. Its checkpoint is fully interchangeable with
`FullPictureMEFARGGeneric` (identical state_dict keys).

## SynPAIN Composite Image Fix

SynPAIN images are **1312x736 side-by-side composites** (neutral face on the
left, expressive face on the right). Before the fix, the whole composite was
fed to the model — after `Resize(256)` the image became 456x256, and crops of
172 or 224 captured random patches straddling both faces. Test-time `CenterCrop`
always grabbed the exact boundary between faces.

The fix (in `SynPAINSingle.__getitem__`) crops the right half before transforms:

```python
w, h = img.size
img = img.crop((w // 2, 0, w, h))  # right half = expressive face
```

This improved SynPAIN pretrain F1 by +10 to +26 points on most subsets.
**All commands below use the fixed code.**

## Two Training Stages

### Stage 1: SynPAIN Pretrain (optional)

Two pretrain modes are available:

**Headless (HL)** — `pain_estimation_au_aux.py` with `BackboneOnlyPain`:

```
Image -> Backbone -> mean-pool -> Linear -> binary pain logits
```

No AU GNN head is involved. After training, only the **backbone weights** are
saved (the classifier head is stripped). This isolates the question: "does
SynPAIN teach the backbone useful visual features for pain?"

**Full model (FM)** — `pain_estimation_full_generic.py` with `FullPictureMEFARGGeneric`:

```
Image -> Backbone -> AU GNN Head -> Pain Classifier -> binary pain logits
```

The full model (backbone + GNN + classifier) is saved. This transfers both
backbone features and learned AU graph structure to downstream UNBC fine-tuning.
FM pretrain consistently outperforms HL for UNBC transfer.

SynPAIN subsets available:
- `metadata.csv` — full dataset
- `metadata_part1_only.csv` — Part 1 subset
- `metadata_part2_only.csv` — Part 2 subset

### Stage 2: UNBC Fine-tuning

Trains the full `FullPictureMEFARGAuAux` model on UNBC with the combined loss.
If a SynPAIN pretrained checkpoint is provided via `--resume`, the backbone
weights (and GNN weights for FM checkpoints) are loaded (strict=False, so
mismatched keys are skipped).

Two AU label vocabularies are available:
- **DISFA-derived labels** (default, 8 AUs): `data/UNBC/list/` — use `--num_classes 8`, no `--label_path`
- **Original UNBC labels** (10 AUs): `data/UNBC/list/original_unbc/` — use `--num_classes 10 --label_path original_unbc`

---

## Commands to Reproduce All Experiments

All commands assume you are in the project root directory.
Set `PYTHONUNBUFFERED=1` for real-time logging.

### 1. UNBC-only with DISFA labels (8-AU)

**Without AU aux** (`--lam 0`):

```bash
# ResNet-50
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --num_classes 8 --neighbor_num 4 --lam 0 \
    --exp-name run9/unbc_no_aux_disfa_labels_r50 \
    --gpu_ids 0

# Swin Transformer
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --num_classes 8 --neighbor_num 4 --lam 0 \
    --exp-name run9/unbc_no_aux_disfa_labels_swin \
    --gpu_ids 0
```

**With AU aux** (`--lam 1.0`):

```bash
# ResNet-50
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --num_classes 8 --neighbor_num 4 --lam 1.0 \
    --exp-name run9/unbc_au_aux_disfa_labels_r50 \
    --gpu_ids 0

# Swin Transformer
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --num_classes 8 --neighbor_num 4 --lam 1.0 \
    --exp-name run9/unbc_au_aux_disfa_labels_swin \
    --gpu_ids 0
```

### 2. UNBC-only with original UNBC labels (10-AU)

**Without AU aux** (`--lam 0`):

```bash
# ResNet-50
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 0 \
    --exp-name run9/unbc_no_aux_ori_labels_r50 \
    --gpu_ids 0

# Swin Transformer
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 0 \
    --exp-name run9/unbc_no_aux_ori_labels_swin \
    --gpu_ids 0
```

**With AU aux** (`--lam 1.0`):

```bash
# ResNet-50
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --exp-name run9/unbc_au_aux_only_r50 \
    --gpu_ids 0

# Swin Transformer
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --exp-name run9/unbc_au_aux_only_swin \
    --gpu_ids 0
```

### 3. SynPAIN headless pretrain -> UNBC fine-tune (Run 10, original UNBC labels)

Each configuration has two steps: pretrain on SynPAIN, then fine-tune on UNBC.

**SynPAIN(Full), ResNet-50:**

```bash
# Stage 1: SynPAIN headless pretrain
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc resnet50 --crop-size 172 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata.csv \
    --exp-name run10/hl_full_r50 \
    --gpu_ids 0

# Stage 2: UNBC fine-tune
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run10/hl_full_r50/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run10/unbc_from_hl_full_r50 \
    --gpu_ids 0
```

**SynPAIN(Full), Swin Transformer:**

```bash
# Stage 1
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc swin_transformer_base --crop-size 224 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata.csv \
    --exp-name run10/hl_full_swin \
    --gpu_ids 0

# Stage 2
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run10/hl_full_swin/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run10/unbc_from_hl_full_swin \
    --gpu_ids 0
```

Replace `metadata.csv` with `metadata_part1_only.csv` or `metadata_part2_only.csv`
and adjust exp-names for Part 1 / Part 2 subsets.

### 4. SynPAIN full-model pretrain -> UNBC fine-tune (Run 10, original UNBC labels)

Uses `pain_estimation_full_generic.py` for Stage 1 (saves full model checkpoint
including GNN weights).

**SynPAIN(Full), ResNet-50:**

```bash
# Stage 1: SynPAIN full-model pretrain
uv run python pain_estimation_full_generic.py \
    --dataset SynPAIN --arc resnet50 --crop-size 172 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata.csv \
    --exp-name run10/fm_full_r50 \
    --gpu_ids 0

# Stage 2: UNBC fine-tune
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run10/fm_full_r50/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run10/unbc_from_fm_full_r50 \
    --gpu_ids 0
```

**SynPAIN(Full), Swin Transformer:**

```bash
# Stage 1
uv run python pain_estimation_full_generic.py \
    --dataset SynPAIN --arc swin_transformer_base --crop-size 224 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata.csv \
    --exp-name run10/fm_full_swin \
    --gpu_ids 0

# Stage 2
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run10/fm_full_swin/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run10/unbc_from_fm_full_swin \
    --gpu_ids 0
```

Replace `metadata.csv` with `metadata_part1_only.csv` or `metadata_part2_only.csv`
and adjust exp-names for Part 1 / Part 2 subsets.

---

## Evaluation

After training, evaluate all experiments using `eval_all_unbc.py`. Pass the
experiment names registered in the `REGISTRY` dict:

```bash
# Run 9 experiments
uv run python eval_all_unbc.py \
    run9_unbc_au_aux_only_r50 \
    run9_unbc_au_aux_only_swin \
    run9_unbc_au_aux_disfa_labels_r50 \
    run9_unbc_au_aux_disfa_labels_swin \
    run9_unbc_no_aux_disfa_labels_r50 \
    run9_unbc_no_aux_disfa_labels_swin \
    run9_unbc_no_aux_ori_labels_r50 \
    run9_unbc_no_aux_ori_labels_swin \
    run9_unbc_au_aux_from_full_r50 \
    run9_unbc_au_aux_from_full_swin \
    run9_unbc_au_aux_from_part1_r50 \
    run9_unbc_au_aux_from_part1_swin \
    run9_unbc_au_aux_from_part2_r50 \
    run9_unbc_au_aux_from_part2_swin

# Run 10 experiments (with composite image fix)
uv run python eval_all_unbc.py \
    run10_unbc_from_hl_full_r50 \
    run10_unbc_from_hl_full_swin \
    run10_unbc_from_hl_part1_r50 \
    run10_unbc_from_hl_part1_swin \
    run10_unbc_from_hl_part2_r50 \
    run10_unbc_from_hl_part2_swin \
    run10_unbc_from_fm_full_r50 \
    run10_unbc_from_fm_full_swin \
    run10_unbc_from_fm_part1_r50 \
    run10_unbc_from_fm_part1_swin \
    run10_unbc_from_fm_part2_r50 \
    run10_unbc_from_fm_part2_swin
```

Demographic fairness evaluation on SynPAIN val set:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python eval_synpain_demographics.py
```

---

## Key Flags Reference

| Flag | Description | Default |
|---|---|---|
| `--dataset` | `UNBC` or `SynPAIN` | required |
| `--arc` | `resnet50` or `swin_transformer_base` | `resnet50` |
| `--crop-size` | 172 for ResNet-50, 224 for Swin | 224 |
| `--epochs` | Training epochs | 10 |
| `-b` | Batch size | 64 |
| `-lr` | Learning rate | 1e-4 |
| `--fold` | Cross-validation fold | 1 |
| `--num_classes` | Number of AUs (8 for DISFA labels, 10 for original) | from config |
| `--neighbor_num` | K for GNN top-K neighbors | from config |
| `--label_path` | Subdirectory under `data/UNBC/list/` for labels. Empty = DISFA labels, `original_unbc` = original 10-AU labels | `""` |
| `--lam` | AU auxiliary loss weight. 0 = no AU aux, 1.0 = equal weight | 0.001 |
| `--resume` | Path to pretrained checkpoint (backbone-only or full) | `""` |
| `--binary` | Binary pain classification for SynPAIN pretrain | False |
| `--metadata_file` | SynPAIN metadata CSV (`metadata.csv`, `metadata_part1_only.csv`, `metadata_part2_only.csv`) | `metadata.csv` |
| `--exp-name` | Experiment output directory under `results/` | required |
| `--gpu_ids` | GPU device ID | 0 |

## Experiment Summary

### Run 9 (before composite image fix)

14 experiments: 8 UNBC-only baselines + 6 headless SynPAIN pretrain.

Best overall: **Swin + DISFA labels + AU aux = 67.30 F1** (Pain-class F1: 70.75)

Best SynPAIN transfer: **SynPAIN(Full) -> Swin + AU aux = 55.28 F1**

### Run 10 (with composite image fix)

12 experiments: 6 headless (HL) + 6 full-model (FM) SynPAIN pretrain, all with AU aux (original UNBC labels).

Best SynPAIN transfer: **SynPAIN-FM(Part 1) -> Swin + AU aux = 61.34 F1** (Pain-class F1: 58.42)

Key findings:
- Composite image fix improved SynPAIN pretrain F1 by +10 to +26 points
- Full-model pretrain (FM) consistently outperforms headless (HL) for UNBC transfer
- Part 1 subset produces the best downstream results despite lowest pretrain F1
- FM Swin shows near-zero age bias on SynPAIN (F1 gap: 0.05%)
