# Binary Image Classification — Full Technical Documentation

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [Architecture](#4-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Evaluation & Metrics](#6-evaluation--metrics)
7. [Inference Application](#7-inference-application)
8. [Project Structure](#8-project-structure)
9. [How to Reproduce](#9-how-to-reproduce)
10. [Technical Details & Design Decisions](#10-technical-details--design-decisions)

---

## 1. Project Overview

This project implements an end-to-end **binary image classification** system that distinguishes between two classes of images: **alma** and **glaci**. The system leverages **transfer learning** with the MobileNetV2 architecture pre-trained on ImageNet, fine-tuned for our specific binary task. A Streamlit web application provides a user-friendly interface for real-time inference.

**Key Technologies:**

- **TensorFlow / Keras** — Deep learning framework
- **MobileNetV2** — Pre-trained backbone (ImageNet weights)
- **Streamlit** — Web-based inference UI
- **scikit-learn** — Evaluation metrics
- **Matplotlib** — Learning curve visualization

---

## 2. Problem Statement

### The Challenge

Given a collection of photographs belonging to two distinct categories — **alma** and **glaci** — the goal is to build a classifier that can automatically and accurately assign new, unseen images to the correct class.

### Why This Matters

Manual classification of images is time-consuming, error-prone, and does not scale. An automated deep learning approach provides:

- **Speed** — Classify images in milliseconds
- **Consistency** — No human fatigue or bias
- **Scalability** — Handle thousands of images without additional effort

### Success Criteria

- High accuracy on unseen test data (>95%)
- Fast inference time suitable for interactive use
- Simple deployment via a web interface

---

## 3. Dataset

### Raw Data

| Class     | Total Images | Format |
| --------- | ------------ | ------ |
| `alma`    | 300          | JPG    |
| `glaci`   | 375          | JPG    |
| **Total** | **675**      |        |

The raw images are stored in:

- `data/alma/` — 300 images
- `data/glaci/` — 375 images

Images are photographs captured with a camera (filenames like `IMG_XXXX.JPG`). Some images include copies (files with `- Copie` in the name), which may be duplicates.

### Data Split

The data is split into three subsets for proper model evaluation:

| Split          | alma | glaci | Total | Purpose                                  |
| -------------- | ---- | ----- | ----- | ---------------------------------------- |
| **Train**      | 105  | 92    | 197   | Model weight optimization                |
| **Validation** | 19   | 17    | 36    | Hyperparameter tuning & early monitoring |
| **Test**       | 19   | 17    | 36    | Final unbiased performance evaluation    |

> **Note:** The effective dataset size after deduplication is 269 images (from 675 raw), indicating significant duplication in the original data.

### Preprocessing

At load time, images undergo the following transformations:

1. **Resizing** — All images are resized to **96×96 pixels** (configurable via `--img-size`)
2. **Batching** — Images are grouped into batches of **32** (configurable via `--batch-size`)
3. **MobileNetV2 Preprocessing** — Pixel values are scaled from `[0, 255]` to `[-1, 1]` using `tf.keras.applications.mobilenet_v2.preprocess_input()`
4. **Prefetching** — `tf.data.AUTOTUNE` is used to overlap data loading with training for performance
5. **Error Handling** — `.ignore_errors()` skips corrupted image files gracefully

---

## 4. Architecture

### Overview

The model uses **transfer learning** with **MobileNetV2** as the feature extractor (backbone), with a custom classification head on top.

```
┌──────────────────────────────────────┐
│          Input Layer (96×96×3)        │
├──────────────────────────────────────┤
│   MobileNetV2 Preprocessing          │
│   (Scale pixels to [-1, 1])          │
├──────────────────────────────────────┤
│                                      │
│   MobileNetV2 (Frozen)               │
│   - Pre-trained on ImageNet          │
│   - 2.2M parameters (not trained)    │
│   - include_top=False                │
│   - Output: Feature maps             │
│                                      │
├──────────────────────────────────────┤
│   Global Average Pooling 2D          │
│   (Reduces spatial dims → vector)    │
├──────────────────────────────────────┤
│   Dropout (rate=0.2)                 │
│   (Regularization)                   │
├──────────────────────────────────────┤
│   Dense (1 unit, Sigmoid)            │
│   (Binary classification output)     │
└──────────────────────────────────────┘
```

### Why MobileNetV2?

| Feature                  | Benefit                                                      |
| ------------------------ | ------------------------------------------------------------ |
| Lightweight              | ~3.4M total parameters, fast inference                       |
| Pre-trained on ImageNet  | Rich visual feature representations                          |
| Inverted Residuals       | Efficient computation with depth-wise separable convolutions |
| Small input size support | Works well with 96×96 images                                 |

### Layer-by-Layer Breakdown

1. **Input Layer** (`tf.keras.Input`)
   - Shape: `(96, 96, 3)` — 96×96 RGB images
   - Accepts raw pixel values in `[0, 255]`

2. **MobileNetV2 Preprocessing** (`mobilenet_v2.preprocess_input`)
   - Normalizes pixel values from `[0, 255]` to `[-1, 1]`
   - Required by MobileNetV2's expected input distribution

3. **MobileNetV2 Backbone** (Frozen)
   - Pre-trained on ImageNet (1000 classes, 1.2M images)
   - `include_top=False` — removes the original classification head
   - `base.trainable = False` — all backbone weights are frozen
   - The backbone acts as a **fixed feature extractor**
   - Outputs a 3D feature map (spatial dimensions depend on input size)

4. **Global Average Pooling 2D**
   - Converts the 3D feature map to a 1D feature vector
   - Takes the spatial average of each feature channel
   - Reduces parameters and prevents overfitting compared to Flatten

5. **Dropout (0.2)**
   - Randomly zeros 20% of activations during training
   - Regularization technique to prevent overfitting
   - Inactive during inference

6. **Dense (1, Sigmoid)**
   - Single output neuron for binary classification
   - Sigmoid activation outputs a probability in `[0, 1]`
   - Threshold at 0.5: `< 0.5 → alma`, `≥ 0.5 → glaci`

### Transfer Learning Strategy

The approach uses **feature extraction** (not fine-tuning):

- The MobileNetV2 backbone is completely **frozen** — its weights are not updated during training
- Only the **classification head** (Global Average Pooling → Dropout → Dense) is trained
- This drastically reduces the number of trainable parameters and training time
- Works well when the target dataset is small and similar enough to ImageNet data

---

## 5. Training Pipeline

### Training Configuration

| Parameter      | Default Value                    | CLI Flag       |
| -------------- | -------------------------------- | -------------- |
| Data directory | `data_split`                     | `--data-dir`   |
| Image size     | 96×96                            | `--img-size`   |
| Batch size     | 32                               | `--batch-size` |
| Epochs         | 10                               | `--epochs`     |
| Model output   | `artifacts/2-epoches-test.keras` | `--model-out`  |

### Compilation Settings

| Setting   | Value               | Rationale                                           |
| --------- | ------------------- | --------------------------------------------------- |
| Optimizer | Adam                | Adaptive learning rate, good default for most tasks |
| Loss      | Binary Crossentropy | Standard loss for binary classification             |
| Metric    | Accuracy            | Intuitive monitoring metric during training         |

### Training Flow (`main.py`)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Parse command-line arguments                                 │
│ 2. Load train / val / test datasets from data_split/            │
│ 3. Build MobileNetV2 model with custom head                    │
│ 4. Compile model (Adam + Binary Crossentropy)                  │
│ 5. Train model with validation monitoring                      │
│ 6. Save trained model to artifacts/*.keras                     │
│ 7. Plot and save learning curves                               │
│ 8. Evaluate on train / val / test sets                         │
│    - Accuracy, Precision, Recall, F1-Score                     │
│ 9. Save metrics to artifacts/metrics.json                      │
│ 10. Save metadata (class names, img size) to artifacts/meta.json│
│ 11. Update README.md with latest test metrics                  │
└─────────────────────────────────────────────────────────────────┘
```

### Artifacts Produced

After training, the following files are generated in `artifacts/`:

| File                   | Description                                              |
| ---------------------- | -------------------------------------------------------- |
| `2-epoches-test.keras` | Serialized trained Keras model                           |
| `meta.json`            | Class names (`["alma", "glaci"]`) and image size (96)    |
| `metrics.json`         | Train/val/test metrics (accuracy, precision, recall, F1) |
| `learning_curves.png`  | Loss and accuracy plots over epochs                      |

---

## 6. Evaluation & Metrics

### Metrics Used

| Metric        | Formula                                         | What It Measures                             |
| ------------- | ----------------------------------------------- | -------------------------------------------- |
| **Accuracy**  | (TP + TN) / (TP + TN + FP + FN)                 | Overall correctness                          |
| **Precision** | TP / (TP + FP)                                  | Of predicted positives, how many are correct |
| **Recall**    | TP / (TP + FN)                                  | Of actual positives, how many were found     |
| **F1-Score**  | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision & recall          |

### Latest Results

| Split          | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| **Train**      | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| **Validation** | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| **Test**       | 1.0000   | 1.0000    | 1.0000 | 1.0000   |

> **Note:** Perfect scores across all splits suggest the two classes are visually very distinct, making the classification task straightforward for the model. This is a good sign — the model generalizes perfectly to unseen data.

### Prediction Threshold

- The sigmoid output produces a probability `p ∈ [0, 1]`
- If `p ≥ 0.5` → predicted class is **glaci** (index 1)
- If `p < 0.5` → predicted class is **alma** (index 0)
- Confidence is calculated as `max(p, 1-p)`

---

## 7. Inference Application

### Streamlit App (`app.py`)

The project includes a **Streamlit web application** for interactive inference.

#### Features

- Upload a JPG/PNG image
- Displays the uploaded image
- Shows predicted class and confidence percentage
- Displays raw probability distribution for both classes
- Shows latest training metrics from `metrics.json`

#### App Flow

```
User uploads image
       ↓
Image converted to RGB (if needed)
       ↓
Resized to 96×96
       ↓
Converted to numpy array
       ↓
Batch dimension added (1, 96, 96, 3)
       ↓
Model prediction (sigmoid output)
       ↓
Threshold at 0.5 → class label
       ↓
Display: class name + confidence %
```

#### Performance Optimization

- Model is loaded once and cached using `@st.cache_resource`
- Subsequent predictions reuse the cached model

---

## 8. Project Structure

```
binary-classification/
│
├── main.py                   # Training script (entry point)
├── app.py                    # Streamlit inference application
├── requirements.txt          # Python dependencies
├── README.md                 # Project readme
├── DOCUMENTATION.md          # This file
│
├── data/                     # Raw image data
│   ├── alma/                 # Class 0 images (300 JPGs)
│   └── glaci/                # Class 1 images (375 JPGs)
│
├── data_split/               # Train/val/test splits
│   ├── train/
│   │   ├── alma/             # 105 training images
│   │   └── glaci/            # 92 training images
│   ├── val/
│   │   ├── alma/             # 19 validation images
│   │   └── glaci/            # 17 validation images
│   └── test/
│       ├── alma/             # 19 test images
│       └── glaci/            # 17 test images
│
└── artifacts/                # Training outputs
    ├── 2-epoches-test.keras  # Trained model
    ├── meta.json             # Class names + image size
    ├── metrics.json          # Evaluation metrics
    └── learning_curves.png   # Loss & accuracy plots
```

---

## 9. How to Reproduce

### Prerequisites

- Python 3.8+
- pip

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies:

- `tensorflow-cpu` — Deep learning framework (CPU variant)
- `pillow` — Image I/O
- `streamlit` — Web application framework
- `scikit-learn` — Evaluation metrics

### Step 2: Prepare Data

Place images into the appropriate class folders:

```
data/alma/    ← place "alma" images here
data/glaci/   ← place "glaci" images here
```

Create the data split manually or with a script into `data_split/train`, `data_split/val`, and `data_split/test`.

### Step 3: Train the Model

```bash
python main.py
```

With custom parameters:

```bash
python main.py --epochs 20 --img-size 128 --batch-size 16 --model-out artifacts/my_model.keras
```

### Step 4: Run the Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 10. Technical Details & Design Decisions

### Why Transfer Learning over Training from Scratch?

| Aspect           | From Scratch         | Transfer Learning (this project) |
| ---------------- | -------------------- | -------------------------------- |
| Data required    | Thousands+           | Hundreds suffice                 |
| Training time    | Hours/days           | Minutes                          |
| Feature quality  | Learned from 0       | Rich ImageNet features           |
| Overfitting risk | High with small data | Low (frozen backbone)            |

With only ~270 unique images, training a CNN from scratch would almost certainly overfit. Transfer learning with a frozen MobileNetV2 backbone provides powerful pre-learned visual features that generalize well.

### Why 96×96 Input Size?

- Smaller than the standard 224×224 used in many models
- **Faster training and inference** due to fewer pixels
- Sufficient for this task since the two classes are visually distinct
- Reduces memory footprint

### Why Freeze the Entire Backbone?

- The dataset is small (~270 images) — fine-tuning all layers would cause overfitting
- MobileNetV2's ImageNet features are general-purpose and transfer well
- Only ~1,281 parameters are trainable (the Dense layer + bias), making training extremely fast

### Why Binary Crossentropy + Sigmoid (not Softmax)?

For binary classification with a single output neuron:

- **Sigmoid** maps the output to `[0, 1]` — interpretable as "probability of class 1"
- **Binary Crossentropy** is the mathematically correct loss for this setup
- More efficient than 2-class softmax (fewer parameters, equivalent performance)

### Why Adam Optimizer?

- Combines momentum and adaptive learning rates
- Works well out of the box without extensive hyperparameter tuning
- Standard choice for transfer learning tasks

### Potential Improvements

1. **Data Augmentation** — Random flips, rotations, brightness changes to increase effective training data
2. **Fine-tuning** — Unfreeze top layers of MobileNetV2 for domain-specific feature learning
3. **Larger Input Size** — Increase from 96×96 to 224×224 for more detail
4. **Cross-Validation** — K-fold validation for more robust performance estimates
5. **Early Stopping** — Stop training when validation loss stops improving
6. **Learning Rate Scheduling** — Reduce learning rate on plateau for better convergence
