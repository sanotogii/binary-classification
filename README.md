# Binary Image Classification

## Overview

This project trains a binary classifier on images stored in [data/alma](data/alma) and [data/glaci](data/glaci). The training script deduplicates files by content hash, creates train/val/test splits, trains a ResNet-18 model, and saves artifacts for inference.

## Project Structure

- [main.py](main.py): data preparation + model training + metrics
- [app.py](app.py): Streamlit app to test the trained model
- data/: original images
- data_split/: generated train/val/test splits
- artifacts/: trained model + metadata + metrics

## Setup

Create and activate your venv, then install dependencies (CPU-only PyTorch wheels are used):

- pip install -r requirements.txt

## Train

Run training (also builds train/val/test splits):

- python main.py --force-split

Artifacts:

- artifacts/model.pt
- artifacts/meta.json
- artifacts/metrics.json

## Streamlit App

- streamlit run app.py

## Data Preparation Details

The script:

1. Scans both class folders.
2. Removes duplicates using MD5 hash.
3. Splits each class into train/val/test with default 70/15/15.
4. Copies unique files into data_split/.

## Metrics (latest run)

These values are updated automatically after training.

- Accuracy (test): 1.0000
- Precision (test): 1.0000
- Recall (test): 1.0000
- F1-score (test): 1.0000

## Notes

- The positive class is the class with the higher index in the `class_to_idx` mapping saved in artifacts/meta.json.
- Image size defaults to 224x224 with ImageNet normalization.
