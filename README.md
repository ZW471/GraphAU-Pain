# GraphAU-Pain: Training and Pain Estimation

This repository contains tools and scripts for training models and performing pain estimation using the UNBC dataset. The configuration files provided enable users to train and test models effectively.

## Overview

The project consists of the following main scripts:

1. **`train_stage1.py`**: A script to perform the first stage of training (pretraining and fine-tuning the AU representation module) with configurable model architectures and parameters. Use **[UNBC_calculate_AU_class_weights.py](tool/UNBC_calculate_AU_class_weights.py)** to calculate AU class weights before running this script.
2. **`pain_estimation_full.py`**: A script to train the model for pain estimation, supporting advanced configurations such as resuming from a checkpoint. Use **[UNBC_calculate_pspi_class_weights.py](tool/UNBC_calculate_pspi_class_weights.py)** to calculate pain class weights before running this script.
3. **`pain_estimation_delta.py`**: A script to test whether the ΔGraph (DeltaMEFARG) model improves pain estimation on UNBC. Each UNBC frame is automatically paired with a per-subject neutral reference frame; the model classifies the node-level delta between expressive and neutral AU graphs. Supports loading DISFA/AU-pretrained backbone weights via `--resume`. See [README_extension.md](README_extension.md) for details.


## Requirements

Ensure you meet the following requirements before running the scripts:

- Python 3.9+
- Installed dependencies for the project environment.
- Access to the UNBC dataset, and store your processed dataset in `data\UNBC` folder, and use `img` and `list` subfolder to store the images and labels. Sample files are provided.
- Pretrained weights for CNN backbone and AU recognition module from [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU?tab=readme-ov-file).

## Usage

### Train Stage 1: Representation Learning Module Training for AU Prediction Using `train_stage1.py`

The script `train_stage1.py` is used for training models. Use the following command to execute the training:

```bash
python train_stage1.py --dataset UNBC --arc resnet50 --exp-name train_unbc -b 16 -lr 0.0001 --fold 1
```

#### Arguments:
- `--dataset`: Specify the dataset to use (e.g., `UNBC`).
- `--arc`: Backbone model architecture (e.g., `resnet50`).
- `--exp-name`: Name of the experiment for logging and saving results.
- `-b`: Batch size (e.g., `16`).
- `-lr`: Learning rate (e.g., `0.0001`).
- `--fold`: Specify fold for cross-validation.
- `--resume`: Use this parameter for supervised fine-tuning. Provide the path to a pretrained model.

You can configure additional parameters if needed.

### ΔGraph Pain Estimation on UNBC Using `pain_estimation_delta.py`

Tests whether the delta-graph representation (expressive − neutral AU node features) improves pain classification on UNBC. Frames are auto-paired per subject; no extra annotation is needed beyond the existing pspi list files.

```bash
# Node-level ΔGraph (full model)
python pain_estimation_delta.py --dataset UNBC --fold 1 \
    --arc resnet50 --exp-name delta_graph_fold1 \
    --resume path/to/disfa_pretrained.pth \
    --use_delta_graph

# Ablation: graph-level delta (PE-score only, omit --use_delta_graph)
python pain_estimation_delta.py --dataset UNBC --fold 1 \
    --arc resnet50 --exp-name pe_score_delta_fold1

# Binary classification
python pain_estimation_delta.py --dataset UNBC --fold 1 \
    --binary True --use_delta_graph \
    --exp-name delta_graph_binary_fold1
```

#### Additional Arguments:
- `--use_delta_graph`: Enable node-level ΔGraph (H\_expr − H\_neu). Omit for PE-score-level ablation.
- `--binary`: Binary (pain / no-pain) classification instead of 3-class.
- All other arguments (`--arc`, `-b`, `-lr`, `--fold`, `--resume`, etc.) are the same as `pain_estimation_full.py`.

---

### Pain Estimation with Pre-trained Weights Using `pain_estimation_full.py`

The script `pain_estimation_full.py` executes the pain estimation pipeline with an existing checkpoint. Use the following command:

```bash
python pain_estimation_full.py --dataset UNBC --arc resnet50 --exp-name full_network -b 64 -lr 0.0001 --fold 1 --crop-size 172 --resume path/to/pretrained_model.pth
```

#### Arguments:
- `--dataset`: Specify the dataset to use (e.g., `UNBC`).
- `--arc`: Backbone model architecture (e.g., `resnet50`).
- `--exp-name`: Name of the experiment for logging.
- `-b`: Batch size (e.g., `64`).
- `-lr`: Learning rate (e.g., `0.0001`).
- `--fold`: Specify the cross-validation fold (e.g., `1`).
- `--crop-size`: Crop size for input images (e.g., `172`).
- `--resume`: Path to the pre-trained model for resuming.

## Outputs

Results are saved in the `results` directory, organized by the experiment name specified in the `--exp-name` parameter.

For example:
- Logs, model checkpoints, and outputs for a training session using `train_stage1.py` are saved under `results/<exp-name>/`.
- Testing or evaluation results for pain estimation using `pain_estimation_full.py` are saved under `results/<exp-name>/`.

## Notes

- Set the environment variable `PYTHONUNBUFFERED=1` to ensure real-time logging and debugging.
- Use the provided run configurations in the `.run.xml` files to set up your IDE (e.g., PyCharm) for quick and error-free execution.
- the lists of images used in fine-tuning and training is available in `data_paths/`

## License

This project is licensed under [MIT License].

## Acknowledgements

This study involved secondary analyses of an existing dataset that has been described and cited accordingly.
No human subject data are collected in this study. All experiments and analysis are conducted on public available dataset that has been described and cited accordingly. Implementation code are open source, including preprocessing, model training, and post-hoc analysis, via this repository. The AU representation learning module is an adapted version of the ANFL module from the [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU?tab=readme-ov-file) AU occurrence prediction model.