# GraphAU-Pain Extension: SynPAIN + ΔGraph + Demographic Bias Evaluation

This document covers the MVP extension built on top of the original GraphAU-Pain codebase.
The baseline is fully preserved — all new functionality is opt-in via config flags.

---

## Table of contents

1. [What was added](#what-was-added)
2. [Requirements](#requirements)
3. [Architecture overview](#architecture-overview)
4. [SynPAIN metadata format](#synpain-metadata-format)
5. [Running the extension](#running-the-extension)
6. [Demographic bias evaluation](#demographic-bias-evaluation)
7. [New CLI flags](#new-cli-flags)
8. [Checkpoints](#checkpoints)
9. [File change summary](#file-change-summary)
10. [Backward compatibility](#backward-compatibility)
11. [Known limitations](#known-limitations)
12. [Troubleshooting](#troubleshooting)

---

## What was added

| Capability | Key files |
|---|---|
| SynPAIN dataset loader (pair-input + demographics) | `dataset_synpain.py` |
| Pair-input ΔGraph model | `model/ANFL.py` — `DeltaMEFARG` class |
| SynPAIN pretraining script | `train_synpain.py` |
| Demographic bias evaluation | `eval_demographics.py` |
| SynPAIN dataset config | `config/SynPAIN_config.yaml` |
| New CLI flags | `conf.py` |

---

## Requirements

### Python packages

All original GraphAU-Pain dependencies are unchanged. One additional package is needed
for demographic evaluation only:

```bash
pip install scikit-learn
```

scikit-learn is required for `eval_demographics.py` (AUROC, balanced accuracy).
Importing the module is always safe — the dependency is checked lazily when
`evaluate_by_demographics()` is first called. Scripts run fine without scikit-learn
as long as `--eval_by_group` is not set.

### Pre-trained backbone weights

`DeltaMEFARG` uses the same backbone loading as the original models. Pre-trained
ResNet and Swin Transformer weights should already be in `checkpoints/` if the
original codebase is working.

---

## Architecture overview

### Internal tensor shapes

Both supported backbones return **3D token sequences**, not flat vectors:

| Backbone | After last stage | After `view + permute` / patch embed | Shape |
|---|---|---|---|
| ResNet-50 | `[B, 2048, 7, 7]` | `view(b,c,-1).permute(0,2,1)` | `[B, 49, 2048]` |
| Swin-Base | `[B, 49, 1024]` | avgpool commented out — returned as-is | `[B, 49, 1024]` |

`LinearBlock` and `HeadPEAU` operate on these 3D tensors throughout. `DeltaMEFARG`
uses the same backbone setup and produces fully dynamic output dimensions
(`out_channels = in_channels // 2` for Swin, `// 4` for ResNet).

### SynPAIN pretraining (pair-input ΔGraph mode)

```
  x_neu  ──┐
            ├─► shared backbone → global_linear → HeadPEAU (AU graph)
  x_expr ──┘
                │
                ├─► f_v_neu   [B, nAU, d]   per-AU node embeddings, neutral frame
                └─► f_v_expr  [B, nAU, d]   per-AU node embeddings, expressive frame

  H_delta      = f_v_expr - f_v_neu        [B, nAU, d]   ← node-level ΔGraph
  delta_pooled = H_delta.mean(dim=1)       [B, d]        ← mean over AU nodes
  logits       = classifier(delta_pooled)  [B, num_pain_classes]
```

`--use_delta_graph` off (ablation): uses `pe_score_expr - pe_score_neu` instead
(graph-level difference of sum-pooled embeddings, shape `[B, d]`).

### UNBC finetuning (single-frame mode, unchanged)

```
  x ──► backbone → global_linear → HeadPEAU → FullPictureMEFARG → pain logits
```

The only model-level change is a `return_node_features=False` flag on `HeadPEAU`.
All existing call sites omit the flag and get the original return signature.

---

## SynPAIN metadata format

Place a CSV file at `data/SynPAIN/metadata.csv`
(the path is controlled by `dataset_path` in `config/SynPAIN_config.yaml`).

### Required columns

| Column | Type | Description |
|---|---|---|
| `neutral_path` | str | Path to neutral image, relative to `dataset_path` |
| `expr_path` | str | Path to expressive image, relative to `dataset_path` |
| `label` | int | `0` = no pain, `1` = pain |

### Optional columns

Parsed when present; `'unknown'` placeholder used when absent.

| Column | Description |
|---|---|
| `split` | `train` / `val` / `test` — if absent, **all rows appear in every split** |
| `age_group` | e.g. `young`, `middle`, `elder` |
| `gender` | e.g. `M`, `F` |
| `ethnicity` | e.g. `White`, `EastAsian`, `Black` |
| `subject_id` | any string identifier |

### Example

```
neutral_path,expr_path,label,split,age_group,gender,ethnicity,subject_id
imgs/s01_neu.jpg,imgs/s01_expr.jpg,1,train,adult,F,White,S01
imgs/s02_neu.jpg,imgs/s02_expr.jpg,0,train,elder,M,EastAsian,S02
imgs/s03_neu.jpg,imgs/s03_expr.jpg,1,val,young,F,Black,S03
```

### Image size note

The transform pipeline resizes images to 256 then crops to `crop_size` (default 224),
consistent with all existing datasets (BP4D/DISFA/UNBC). SynPAIN images should be at
least 224×224. Paired neutral and expressive frames must have the same spatial resolution
so that identical random crops can be applied.

---

## Running the extension

### SynPAIN pretraining

Minimal run (node-level ΔGraph, no demographic logging):
```bash
python train_synpain.py \
  --dataset SynPAIN \
  --use_pair_input \
  --use_delta_graph \
  --arc swin_transformer_base \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 1e-5 \
  --exp-name synpain_delta
```

With demographic bias evaluation logged after each epoch:
```bash
python train_synpain.py \
  --dataset SynPAIN \
  --use_pair_input \
  --use_delta_graph \
  --exp-name synpain_delta \
  --eval_by_group \
  --save_demographics
```

Warm-start from a DISFA stage-1 checkpoint (recommended):
```bash
python train_synpain.py \
  --dataset SynPAIN \
  --use_pair_input \
  --use_delta_graph \
  --resume results/disfa_stage1/best_model.pth \
  --exp-name synpain_delta_warmstart
```

Graph-level delta ablation (omit `--use_delta_graph`):
```bash
python train_synpain.py \
  --dataset SynPAIN \
  --use_pair_input \
  --exp-name synpain_graphlevel
```

### UNBC finetuning (unchanged baseline)

```bash
python pain_estimation_full.py \
  --dataset UNBC \
  --fold 1 \
  --arc swin_transformer_base \
  --epochs 20 \
  --binary True \
  --exp-name unbc_finetune_fold1
```

Finetuning from a SynPAIN pretrained checkpoint:
```bash
python pain_estimation_full.py \
  --dataset UNBC \
  --fold 1 \
  --binary True \
  --resume results/synpain_delta/best_synpain.pth \
  --exp-name unbc_from_synpain_fold1
```

> **Note:** `pain_estimation_full.py` uses `FullPictureMEFARG` which has a hardcoded
> `nn.Linear(2048, 36)` backbone projection. It is designed for ResNet-50 only.
> See [Known limitations](#known-limitations) for details.

---

## Demographic bias evaluation

Evaluation runs automatically during `train_synpain.py` validation when `--eval_by_group`
is set. scikit-learn must be installed.

It can also be called standalone from any evaluation script:

```python
from eval_demographics import evaluate_by_demographics

results = evaluate_by_demographics(
    all_labels,       # list[int]   — ground-truth binary labels
    all_scores,       # list[float] — positive-class probabilities (for AUROC)
    all_preds,        # list[int]   — binary hard predictions
    all_demographics={
        'gender':    ['F', 'M', ...],
        'ethnicity': ['White', 'EastAsian', ...],
        'age_group': ['adult', 'elder', ...],
    },
    output_dir='results/my_run',  # None to skip saving files
    prefix='epoch5_',
)
```

### Output format

**Terminal:**
```
============================================================
  Demographic Bias Evaluation
============================================================

[ETHNICITY]
  White               n=120    AUROC=0.8412  F1=0.7610  BalAcc=0.7900
  EastAsian           n=95     AUROC=0.7901  F1=0.7100  BalAcc=0.7450
  Black               n=80     AUROC=0.8100  F1=0.7300  BalAcc=0.7600
  >> Bias gap (AUROC): 0.0511  |  Worst-group AUROC: 0.7901

[GENDER]
  F                   n=180    AUROC=0.8250  F1=0.7500  BalAcc=0.7800
  M                   n=115    AUROC=0.8010  F1=0.7200  BalAcc=0.7600
  >> Bias gap (AUROC): 0.0240  |  Worst-group AUROC: 0.8010
============================================================
```

**JSON** (`epoch5_demographics.json`):
```json
{
  "ethnicity": {
    "White":     {"auroc": 0.8412, "f1": 0.7610, "balanced_acc": 0.7900, "n_samples": 120},
    "EastAsian": {"auroc": 0.7901, "f1": 0.7100, "balanced_acc": 0.7450, "n_samples": 95},
    "Black":     {"auroc": 0.8100, "f1": 0.7300, "balanced_acc": 0.7600, "n_samples": 80}
  },
  "ethnicity_bias_gap_auroc": 0.0511,
  "ethnicity_worst_group_auroc": 0.7901,
  "ethnicity_bias_gap_f1": 0.051,
  "ethnicity_worst_group_f1": 0.71,
  "gender": { "..." },
  "gender_bias_gap_auroc": 0.024,
  "gender_worst_group_auroc": 0.801
}
```

Groups with only one class present in labels receive `auroc: null` (cannot be computed).
All other metrics are always populated.

**CSV** (`epoch5_demographics.csv`):
```
attribute,group,n_samples,auroc,f1,balanced_acc
ethnicity,White,120,0.8412,0.7610,0.7900
ethnicity,EastAsian,95,0.7901,0.7100,0.7450
gender,F,180,0.8250,0.7500,0.7800
gender,M,115,0.8010,0.7200,0.7600
```

---

## New CLI flags

All flags are optional. When omitted they default to values that reproduce the
original GraphAU-Pain behaviour exactly.

| Flag | Default | Description |
|---|---|---|
| `--use_pair_input` | off | Enable pair-input mode (x_neu + x_expr); used by `train_synpain.py` |
| `--use_delta_graph` | off | Node-level ΔGraph (H_expr − H_neu); requires `--use_pair_input` |
| `--training_stage` | `unbc_finetune` | `disfa_pretrain` / `synpain_pretrain` / `unbc_finetune` |
| `--eval_by_group` | off | Compute per-subgroup metrics each validation epoch; requires scikit-learn |
| `--save_demographics` | off | Write JSON + CSV to `outdir`; only active when `--eval_by_group` is set |

---

## Checkpoints

All checkpoints follow the existing project format:

```python
{'epoch': int, 'state_dict': OrderedDict, 'optimizer': OrderedDict}
```

Files saved by `train_synpain.py`:

| File | Saved when |
|---|---|
| `epoch{N}_synpain.pth` | Every epoch |
| `best_synpain.pth` | New best validation F1 |

`load_state_dict()` from `utils.py` loads with `strict=False`, so a DISFA or BP4D
stage-1 checkpoint can be used to warm-start `DeltaMEFARG` even though the final
classifier head is new and will not match.

---

## File change summary

```
Modified (2 files)
├── model/ANFL.py
│     HeadPEAU: added return_node_features=False parameter (backward-compatible)
│     DeltaMEFARG: new class appended at the bottom of the file
└── conf.py
      +5 argparse flags (all default-safe for existing scripts)
      +SynPAIN branch in get_config()

Added (4 files)
├── config/SynPAIN_config.yaml   dataset path, AU count, pain class count
├── dataset_synpain.py           SynPAIN Dataset class + synpain_collate_fn
├── eval_demographics.py         per-subgroup metrics, bias gap, JSON/CSV output
└── train_synpain.py             SynPAIN pretraining script

Unchanged
├── pain_estimation_full.py      UNBC pain classification baseline
├── train_stage1/2/3.py          AU detection pretraining
├── dataset.py                   BP4D / DISFA / UNBC loaders
├── utils.py                     losses, metrics, image transforms
└── model/MEFL.py                stage-2 edge feature model
```

---

## Backward compatibility

The original GraphAU-Pain baseline is fully preserved.

| Component | Status |
|---|---|
| `FullPictureMEFARG.forward(x)` | ✅ Return signature unchanged |
| `HeadPEAU` existing call sites | ✅ New `return_node_features` defaults to `False` |
| `pain_estimation_full.py` | ✅ Runs without modification |
| `train_stage1/2/3.py` | ✅ Unaffected |
| `dataset.py` | ✅ Unaffected |
| Existing `.pth` checkpoints | ✅ Loadable via `load_state_dict(strict=False)` |
| New argparse flags in old scripts | ✅ All default to `False` / `'unbc_finetune'`; zero behavioural effect |

No existing class was renamed, removed, or had its default behaviour changed.

---

## Known limitations

### `FullPictureMEFARG` is ResNet-50 only (pre-existing issue)

`FullPictureMEFARG` (used for UNBC finetuning) contains a hardcoded
`nn.Linear(2048, 36)` backbone projection. This dimension is correct for ResNet-50
(`in_channels = 2048`) but will crash with Swin Transformer (`in_channels = 1024`).
This is a pre-existing issue in the original codebase; the extension does not modify it.

**`DeltaMEFARG` does not have this problem.** All projection dimensions are computed
dynamically from the backbone and work with any supported backbone.

### SynPAIN pretraining is binary only (MVP)

The current training script (`train_synpain.py`) uses binary pain/non-pain labels.
A regression hook (`num_pain_classes=1` + MSE loss) is left as a future extension.

### No fold-based cross-validation for SynPAIN

`train_synpain.py` uses a single train/val split defined by the `split` column in the
metadata CSV. The `--fold` and `--N-fold` flags are ignored by this script.

---

## Troubleshooting

**`FileNotFoundError: SynPAIN metadata not found`**
→ Check that `data/SynPAIN/metadata.csv` exists (or adjust `dataset_path` in
`config/SynPAIN_config.yaml`).

**`RuntimeError: No samples found for split='val'`**
→ The metadata CSV has no rows with `split=val`. Either add a `split` column or
remove it entirely (all rows will then be used for every split, which is useful
for quick tests but not for real training).

**`ImportError: scikit-learn is required for demographic evaluation`**
→ `pip install scikit-learn`, or drop `--eval_by_group` if you only want loss/acc
metrics during training.

**`AssertionError: Input image size does not match model`** (Swin Transformer)
→ The Swin backbone asserts that input height and width match the model's `img_size`
(default 224). Ensure `--crop-size 224` and that SynPAIN images are at least 224×224.

**`RuntimeError: mat1 and mat2 shapes cannot be multiplied`** in `FullPictureMEFARG`
→ You are using `pain_estimation_full.py` with a Swin Transformer backbone. Switch
to `--arc resnet50`; the hardcoded `Linear(2048, 36)` in `FullPictureMEFARG` only
supports ResNet-50. `DeltaMEFARG` (used by `train_synpain.py`) does not have this
constraint.
