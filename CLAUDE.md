# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraphAU-Pain is a multi-stage pain estimation framework that uses Graph Neural Networks (GNNs) to model Action Unit (AU) relationships from facial expressions. It supports three datasets (BP4D, DISFA, UNBC) and an extension dataset (SynPAIN) with pair-input delta-graph modeling.

The project is adapted from [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU) and requires pretrained backbone weights from that project.

## Common Commands

### Data Preparation (must run before training)
```bash
# Calculate class weights for AU detection
python tool/UNBC_calculate_AU_class_weights.py
# Calculate class weights for pain estimation
python tool/UNBC_calculate_pspi_class_weights.py
```

### Training
```bash
# Stage 1: AU representation learning
python train_stage1.py --dataset UNBC --arc resnet50 --exp-name train_unbc -b 16 -lr 0.0001 --fold 1

# Pain estimation (full pipeline, requires pretrained AU checkpoint)
python pain_estimation_full.py --dataset UNBC --arc resnet50 --exp-name full_network -b 64 -lr 0.0001 --fold 1 --crop-size 172 --resume path/to/pretrained_model.pth

# SynPAIN pretraining with delta-graph
python train_synpain.py --dataset SynPAIN --use_pair_input --use_delta_graph --arc swin_transformer_base --epochs 20 -b 32 -lr 1e-5 --exp-name synpain_delta
```

### Testing
```bash
python test_au.py        # AU detection evaluation
python test_pain.py      # Pain classification evaluation
```

Set `PYTHONUNBUFFERED=1` for real-time logging.

## Architecture

### Multi-Stage Training Pipeline
1. **Stage 1** (`train_stage1.py`): AU representation learning with backbone + GNN
2. **Stage 2** (`train_stage2.py`): Graph structure learning with edge features
3. **Stage 3** (`pain_estimation_full.py`): Pain estimation using learned AU representations

### Key Model Classes (model/ANFL.py)
- **`MEFARG`**: Core AU detection model = Backbone + GlobalLinear + HeadPEAU (GNN-based AU graph)
- **`FullPictureMEFARG`**: AU model + pain classification head for UNBC. **Hardcoded for ResNet-50 only** (`nn.Linear(2048, 36)`) — will crash with Swin Transformer.
- **`DeltaMEFARG`**: Pair-input model for SynPAIN. Computes node-level delta: `H_delta = f_v_expr - f_v_neu`. Works with any backbone (dynamic projection dimensions).
- **`GNN`**: Graph neural network with dynamic graph construction via top-K nearest neighbors (supports dots/cosine/l1 metrics)

### Alternative Model (model/MEFL.py)
Edge-feature GNN variant (`MEFARGWithGraphRepresentation`) — used for stage 2 graph learning.

### Backbones (model/resnet.py, model/swin_transformer.py)
- ResNet-18/50/101: outputs `[B, 49, 2048]` (for ResNet-50)
- Swin Transformer tiny/small/base: outputs `[B, 49, 1024]` (for base)
- Swin Transformer requires `--crop-size 224` exactly (positional embedding constraint)

### Configuration System
- `conf.py`: Argparse-based config with YAML dataset overrides from `config/` directory
- Dataset configs in `config/{BP4D,DISFA,UNBC,SynPAIN}_config.yaml` define `dataset_path`, `num_classes`, `neighbor_num`
- `--dataset` flag selects which YAML config to load

### Data Pipeline
- `dataset.py`: Loaders for BP4D, DISFA, UNBC with 3-fold cross-validation
- `dataset_synpain.py`: Pair-input loader (neutral + expressive frames) with demographic metadata; uses CSV `split` column instead of folds
- Data lives in `data/<DATASET>/img/` (images) and `data/<DATASET>/list/` (labels, weights, AU relations)
- Image lists for cross-validation splits are in `data_paths/`

### Losses (utils.py)
- `WeightedAsymmetricLoss`: Multi-label BCE with class weights (AU detection)
- `WeightedCrossEntropyLoss`: Softmax CE with class weights (pain classification)
- `FocalLoss`, `WeightedMSELoss`: Alternatives for hard examples and regression

### Outputs
- Results saved to `results/<exp-name>/` with checkpoints (`epoch*.pth`), logs, and optionally demographic evaluation JSON/CSV
- Checkpoint format: `{'epoch': int, 'state_dict': OrderedDict, 'optimizer': OrderedDict}`
- `load_state_dict()` in utils.py uses `strict=False` for cross-stage transfer

## Important Constraints

- UNBC default crop size is 172 (`--crop-size 172`), other datasets use 224
- `FullPictureMEFARG` only works with ResNet-50; use `DeltaMEFARG` for Swin Transformer
- When changing the number of predicted AUs: update YAML config `num_classes`, regenerate label/weight list files, and verify the `statistics` function in utils.py
- SynPAIN extension flags (`--use_pair_input`, `--use_delta_graph`, `--eval_by_group`, `--save_demographics`) all default off and do not affect baseline scripts
- Demographic evaluation (`eval_demographics.py`) requires scikit-learn
