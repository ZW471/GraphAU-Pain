# Run 9: Pain Estimation with AU Auxiliary Loss

## Overview

This run uses `pain_estimation_au_aux.py` to train pain estimation models on UNBC
with Action Unit (AU) classification as an auxiliary loss. Optionally, the backbone
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

## Two Training Stages

### Stage 1: SynPAIN Backbone Pretrain (optional)

Trains a **headless** model (`BackboneOnlyPain`) on the synthetic SynPAIN dataset
for binary pain classification (pain vs no-pain). This model is just:

```
Image -> Backbone -> mean-pool -> Linear -> binary pain logits
```

No AU GNN head is involved. After training, only the **backbone weights** are
saved (the classifier head is stripped). This isolates the question: "does
SynPAIN teach the backbone useful visual features for pain?"

SynPAIN subsets available:
- `metadata.csv` — full dataset
- `metadata_part1_only.csv` — Part 1 subset
- `metadata_part2_only.csv` — Part 2 subset

### Stage 2: UNBC Fine-tuning

Trains the full `FullPictureMEFARGAuAux` model on UNBC with the combined loss.
If a SynPAIN pretrained checkpoint is provided via `--resume`, the backbone
weights are loaded (strict=False, so non-backbone keys are ignored).

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

### 3. SynPAIN pretrain -> UNBC fine-tune with AU aux (original UNBC labels)

Each configuration has two steps: pretrain on SynPAIN, then fine-tune on UNBC.

**SynPAIN(full) -> UNBC, ResNet-50:**

```bash
# Stage 1: SynPAIN pretrain
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc resnet50 --crop-size 172 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata.csv \
    --exp-name run9/synpain_bb_full_r50 \
    --gpu_ids 0

# Stage 2: UNBC fine-tune
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run9/synpain_bb_full_r50/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run9/unbc_au_aux_from_full_r50 \
    --gpu_ids 0
```

**SynPAIN(full) -> UNBC, Swin Transformer:**

```bash
# Stage 1
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc swin_transformer_base --crop-size 224 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata.csv \
    --exp-name run9/synpain_bb_full_swin \
    --gpu_ids 0

# Stage 2
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run9/synpain_bb_full_swin/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run9/unbc_au_aux_from_full_swin \
    --gpu_ids 0
```

**SynPAIN(Part1) -> UNBC, ResNet-50:**

```bash
# Stage 1
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc resnet50 --crop-size 172 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata_part1_only.csv \
    --exp-name run9/synpain_bb_part1_r50 \
    --gpu_ids 0

# Stage 2
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run9/synpain_bb_part1_r50/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run9/unbc_au_aux_from_part1_r50 \
    --gpu_ids 0
```

**SynPAIN(Part1) -> UNBC, Swin Transformer:**

```bash
# Stage 1
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc swin_transformer_base --crop-size 224 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata_part1_only.csv \
    --exp-name run9/synpain_bb_part1_swin \
    --gpu_ids 0

# Stage 2
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run9/synpain_bb_part1_swin/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run9/unbc_au_aux_from_part1_swin \
    --gpu_ids 0
```

**SynPAIN(Part2) -> UNBC, ResNet-50:**

```bash
# Stage 1
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc resnet50 --crop-size 172 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata_part2_only.csv \
    --exp-name run9/synpain_bb_part2_r50 \
    --gpu_ids 0

# Stage 2
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc resnet50 --crop-size 172 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run9/synpain_bb_part2_r50/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run9/unbc_au_aux_from_part2_r50 \
    --gpu_ids 0
```

**SynPAIN(Part2) -> UNBC, Swin Transformer:**

```bash
# Stage 1
uv run python pain_estimation_au_aux.py \
    --dataset SynPAIN --arc swin_transformer_base --crop-size 224 \
    --epochs 10 -b 32 -lr 1e-5 --fold 1 \
    --binary True --num_classes 10 --neighbor_num 4 \
    --metadata_file metadata_part2_only.csv \
    --exp-name run9/synpain_bb_part2_swin \
    --gpu_ids 0

# Stage 2
uv run python pain_estimation_au_aux.py \
    --dataset UNBC --arc swin_transformer_base --crop-size 224 \
    --epochs 20 -b 64 -lr 1e-4 --fold 1 \
    --label_path original_unbc --num_classes 10 --neighbor_num 4 --lam 1.0 \
    --resume results/run9/synpain_bb_part2_swin/bs_32_seed_0_lr_1e-05/best_model_fold1.pth \
    --exp-name run9/unbc_au_aux_from_part2_swin \
    --gpu_ids 0
```

---

## Evaluation

After training, evaluate all experiments using `eval_all_unbc.py`. Pass the
experiment names registered in the `REGISTRY` dict:

```bash
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

Total: 14 experiments

| Category | Count | Description |
|---|---|---|
| UNBC + DISFA labels (8-AU) | 4 | no-aux and AU-aux, R50 and Swin |
| UNBC + original labels (10-AU) | 4 | no-aux and AU-aux, R50 and Swin |
| SynPAIN -> UNBC + AU aux | 6 | 3 SynPAIN subsets x 2 backbones, original labels |

Best result: **Swin + DISFA labels + AU aux = 67.30 F1** (Pain-class F1: 70.75)
