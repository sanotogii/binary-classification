# Presentation: Binary Image Classification with Transfer Learning

## How to Use These Slides

This file contains the full content for a presentation. Each `---` separator represents a new slide. You can:

1. Copy the content into **Google Slides**, **PowerPoint**, or **Canva**
2. Use a Markdown-to-slides tool like **Marp**, **reveal.js**, or **Slidev**
3. Use the Marp format headers below directly (install Marp CLI or VS Code extension)

---

<!--
marp: true
theme: default
paginate: true
backgroundColor: #fff
style: |
  section { font-family: 'Segoe UI', Arial, sans-serif; }
  h1 { color: #1a73e8; }
  h2 { color: #333; }
  table { font-size: 0.85em; }
-->

---

# Binary Image Classification

## Using Transfer Learning with MobileNetV2

**Classifying images into two categories: alma vs. glaci**

_Built with TensorFlow, Keras & Streamlit_

---

# Agenda

1. **The Problem** — What we needed to solve
2. **The Dataset** — What data we had
3. **Our Approach** — Transfer learning strategy
4. **Architecture** — MobileNetV2 + custom head
5. **Training Pipeline** — How we trained the model
6. **Results** — Performance metrics
7. **Live Demo** — Streamlit web app
8. **Lessons & Next Steps**

---

# The Problem

## Manual Image Classification Doesn't Scale

- We have hundreds of images belonging to **two categories** (alma & glaci)
- Manually sorting them is **slow**, **error-prone**, and **tedious**
- We need an **automated system** that can classify new images **instantly**

## Our Goal

> Build a deep learning model that accurately classifies images
> into "alma" or "glaci" — and make it accessible through a web interface.

---

# The Dataset

## Raw Data

| Class     | Images | Format |
| --------- | ------ | ------ |
| **alma**  | 300    | JPG    |
| **glaci** | 375    | JPG    |
| **Total** | 675    |        |

## After Deduplication & Splitting

| Split          | alma | glaci | Total |
| -------------- | ---- | ----- | ----- |
| **Train**      | 105  | 92    | 197   |
| **Validation** | 19   | 17    | 36    |
| **Test**       | 19   | 17    | 36    |

> ~269 unique images after removing duplicates

---

# The Challenge

## Why Not Train from Scratch?

- Only **~270 unique images** — far too few for training a CNN from zero
- Risk of **severe overfitting** — model memorizes training data
- Would need **thousands** of images per class for acceptable results

## The Bottleneck

| Problem          | Impact                              |
| ---------------- | ----------------------------------- |
| Small dataset    | Not enough to learn visual features |
| Duplicate images | Inflated raw count (675 → 269)      |
| No augmentation  | Limited diversity in training data  |

**We need a smarter approach...**

---

# Our Solution: Transfer Learning

## Leverage Pre-Trained Knowledge

Instead of learning visual features from scratch, we **reuse** a model already trained on **1.2 million images** (ImageNet).

```
ImageNet (1.2M images, 1000 classes)
            ↓
  MobileNetV2 learned rich visual features
  (edges, textures, shapes, objects)
            ↓
  We FREEZE these features
            ↓
  Train ONLY a small classification head
  for our specific 2-class problem
```

## Why This Works

- Pre-trained features **generalize** to new tasks
- We only train **~1,281 new parameters** (not millions)
- Training takes **minutes**, not hours

---

# Architecture Overview

## Model Structure

```
┌──────────────────────────────────┐
│     Input Image (96×96×3)        │  ← RGB image
├──────────────────────────────────┤
│     Preprocessing                │  ← Scale to [-1, 1]
├──────────────────────────────────┤
│                                  │
│     MobileNetV2 (FROZEN)         │  ← Pre-trained on ImageNet
│     ~2.2M parameters             │  ← NOT updated during training
│     Feature Extractor            │
│                                  │
├──────────────────────────────────┤
│     Global Average Pooling       │  ← Compress features to vector
├──────────────────────────────────┤
│     Dropout (20%)                │  ← Prevent overfitting
├──────────────────────────────────┤
│     Dense (1, Sigmoid)           │  ← Output: probability [0,1]
└──────────────────────────────────┘
         ↓
    p ≥ 0.5 → glaci
    p < 0.5 → alma
```

---

# Why MobileNetV2?

## A Lightweight Yet Powerful Backbone

| Feature                              | Benefit                     |
| ------------------------------------ | --------------------------- |
| **Small size** (~3.4M params)        | Fast inference, low memory  |
| **ImageNet pre-trained**             | Rich visual feature library |
| **Depthwise separable convolutions** | Efficient computation       |
| **Inverted residual blocks**         | Better gradient flow        |
| **Works at 96×96**                   | Flexible input size support |

## Compared to Alternatives

| Model           | Parameters | Inference Speed |
| --------------- | ---------- | --------------- |
| **MobileNetV2** | **3.4M**   | **Fast** ✓      |
| ResNet-50       | 25.6M      | Medium          |
| VGG-16          | 138M       | Slow            |

**Best trade-off** between performance and efficiency for our use case.

---

# Key Design Decisions

## 1. Frozen Backbone (No Fine-Tuning)

- With only ~270 images, fine-tuning would **overfit**
- Frozen = only the classification head learns
- Only **~1,281 trainable parameters**

## 2. Image Size: 96×96

- Smaller than standard 224×224
- **Faster** training and inference
- Sufficient for our visually distinct classes

## 3. Sigmoid + Binary Crossentropy

- Single output neuron → probability of class 1
- Most efficient setup for binary classification

## 4. Adam Optimizer

- Adaptive learning rate — works well out of the box
- No extensive hyperparameter tuning required

---

# Training Pipeline

## End-to-End Workflow

```
1. Load Data          → Train / Val / Test from data_split/
2. Build Model        → MobileNetV2 + custom head
3. Compile            → Adam + Binary Crossentropy
4. Train              → Fit on training data, monitor validation
5. Save Model         → artifacts/model.keras
6. Plot Curves        → artifacts/learning_curves.png
7. Evaluate           → Accuracy, Precision, Recall, F1
8. Save Metrics       → artifacts/metrics.json
9. Save Metadata      → artifacts/meta.json (class names, img size)
```

## Command

```bash
python main.py --epochs 10 --img-size 96 --batch-size 32
```

---

# Results

## Perfect Classification Across All Splits

| Split          | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| **Train**      | 100%     | 100%      | 100%   | 100%     |
| **Validation** | 100%     | 100%      | 100%   | 100%     |
| **Test**       | 100%     | 100%      | 100%   | 100%     |

## What This Means

- The model **perfectly** classifies all images in the test set
- The two classes (alma vs. glaci) are **visually very distinct**
- Transfer learning features from ImageNet are **more than sufficient**
- No signs of overfitting (train = val = test performance)

---

# Understanding the Metrics

| Metric        | What It Tells Us                              |
| ------------- | --------------------------------------------- |
| **Accuracy**  | % of all predictions that were correct        |
| **Precision** | Of predicted positives, how many were correct |
| **Recall**    | Of actual positives, how many did we find     |
| **F1-Score**  | Balance between precision and recall          |

## Why All Four?

- **Accuracy alone can be misleading** with imbalanced classes
- Precision and Recall catch different types of errors:
  - **False Positive** (predicted glaci, was alma) → hurts Precision
  - **False Negative** (missed a glaci) → hurts Recall
- **F1-Score** gives a single balanced number

---

# Live Demo: Streamlit App

## Web Interface for Real-Time Predictions

```
streamlit run app.py
```

### Features:

- **Upload** any JPG/PNG image
- **Instant** classification result
- Shows **confidence percentage**
- Displays **raw probability** for both classes
- Shows **latest training metrics**

### How It Works:

1. User uploads an image
2. Image is resized to 96×96 and preprocessed
3. Model runs forward pass (cached for speed)
4. Sigmoid output → class label + confidence
5. Results displayed immediately

---

# Technical Stack

| Component     | Technology         | Purpose                    |
| ------------- | ------------------ | -------------------------- |
| Deep Learning | TensorFlow / Keras | Model building & training  |
| Backbone      | MobileNetV2        | Feature extraction         |
| Data Loading  | tf.keras.utils     | Image dataset pipeline     |
| Evaluation    | scikit-learn       | Precision, Recall, F1      |
| Visualization | Matplotlib         | Learning curves            |
| Web App       | Streamlit          | Interactive inference UI   |
| Image I/O     | Pillow             | Image loading & conversion |

---

# Lessons Learned

## What Worked Well

- **Transfer learning** eliminated the small dataset problem
- **MobileNetV2** provided excellent features with minimal compute
- **Freezing the backbone** prevented overfitting on 270 images
- **Streamlit** made deployment incredibly simple

## What We'd Do Differently

- Add **data augmentation** (flips, rotations, color jitter) for robustness
- Try **fine-tuning** the top layers of MobileNetV2
- Use **larger input images** (224×224) for potentially more detail
- Implement **early stopping** to avoid unnecessary epochs
- Add **cross-validation** for more robust evaluation

---

# Potential Improvements

| Improvement             | Expected Benefit                         | Effort |
| ----------------------- | ---------------------------------------- | ------ |
| Data Augmentation       | More robust to variations                | Low    |
| Fine-Tuning Top Layers  | Better domain-specific features          | Low    |
| Larger Image Size       | Capture finer details                    | Low    |
| Early Stopping          | Faster training, prevent overfitting     | Low    |
| Learning Rate Scheduler | Better convergence                       | Medium |
| Grad-CAM Visualization  | Explainability — see what model looks at | Medium |
| Multi-class Extension   | Support more than 2 categories           | Medium |

---

# Summary

## Problem → Solution → Result

```
PROBLEM:    Manual classification of 675+ images
                    ↓
APPROACH:   Transfer Learning with MobileNetV2
            - Frozen backbone (ImageNet features)
            - Custom binary classification head
            - Only ~1,281 trainable parameters
                    ↓
RESULT:     100% accuracy on all splits
            Fast inference via Streamlit web app
            Training completed in minutes
```

## Key Takeaway

> **Transfer learning** makes it possible to build high-accuracy
> image classifiers with **minimal data** and **minimal compute**.

---

# Thank You

## Questions?

**Project Repository Structure:**

- `main.py` — Training pipeline
- `app.py` — Streamlit inference app
- `artifacts/` — Model, metrics, plots
- `data_split/` — Train/Val/Test images

**Run it yourself:**

```bash
pip install -r requirements.txt
python main.py          # Train
streamlit run app.py    # Predict
```

---
