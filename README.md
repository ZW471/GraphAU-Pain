# GraphAU-Pain: Training and Pain Estimation

This repository contains tools and scripts for training models and performing pain estimation using the UNBC dataset. The configuration files provided enable users to train and test models effectively.

## Overview

The project consists of the following main scripts:

1. **`train_stage1.py`**: A script to perform the first stage of training using the UNBC dataset with configurable model architectures and parameters.
2. **`pain_estimation_full.py`**: A script to perform pain estimation using a pre-trained model, supporting advanced configurations such as resuming from a checkpoint.

## Requirements

Ensure you meet the following requirements before running the scripts:

- Python 3.9+
- Installed dependencies for the project environment.
- Access to the UNBC dataset.

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

## License

This project is licensed under [MIT License].

## Acknowledgements

The project utilizes:
- The UNBC dataset.
- `resnet50` architecture for model training and evaluation.