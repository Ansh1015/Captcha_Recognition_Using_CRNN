# CAPTCHA Recognition Using CRNN

A deep learning system that decodes 5-character CAPTCHAs with **99.0% word accuracy** using a Convolutional Recurrent Neural Network (CRNN) with Bidirectional LSTMs, multi-head self-attention, and CTC decoding. Ships with a pre-trained model and a Streamlit web interface for live inference.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Training](#training)
- [Model Limitations](#model-limitations)
- [Tech Stack](#tech-stack)

---

## Overview

This project solves the CAPTCHA recognition problem end-to-end — from raw image to decoded text — without requiring character segmentation. The CRNN architecture treats the CAPTCHA as a sequence prediction problem: the CNN extracts visual features column by column, the BiLSTM layers model character order, and CTC decoding collapses the output sequence into the final text.

The app ships with a pre-trained model. You do not need to train anything to start predicting.

---

## Architecture

```
Input Image (50 × 200 × 1, grayscale)
        │
        ▼
  Rescaling [0, 255] → [0, 1]
        │
        ▼
  Transpose (H × W → W × H, width becomes sequence axis)
        │
        ▼
  CNN Block 1 — Conv2D(64) × 2  + BatchNorm + ReLU + MaxPool(2,2) + Dropout(0.10)
  CNN Block 2 — Conv2D(128) × 2 + BatchNorm + ReLU + MaxPool(2,2) + Dropout(0.15)
  CNN Block 3 — Conv2D(256) × 2 + BatchNorm + ReLU + MaxPool(2,1) + Dropout(0.20)
        │
        ▼
  Reshape → (25 time steps × 3072 features)
        │
        ▼
  Dense Projection (256 units, ReLU) + Dropout(0.25)
        │
        ▼
  BiLSTM (128 units, bidirectional) + Dropout(0.25)
  BiLSTM (64 units,  bidirectional) + Dropout(0.20)
        │
        ▼
  Multi-Head Self-Attention (4 heads, key_dim=32) + Residual + LayerNorm
        │
        ▼
  Dense Output (num_classes + 1, softmax)
        │
        ▼
  CTC Greedy Decode → Predicted Text
```

**Key design choices:**

| Choice | Reason |
|---|---|
| VGG-style double conv blocks | Richer feature maps with fewer pooling-induced artifacts |
| Width as sequence axis | Each column of feature maps becomes one time step |
| Two BiLSTM layers | Deeper sequence modelling; second layer refines first layer's context |
| Multi-head self-attention | Global context across all 25 time steps without sequential bottleneck |
| CTC loss | No need for explicit character alignment during training |
| Segmentation-free | Works on variable-width characters without bounding boxes |

---

## Dataset

**captcha_images_v2** — a widely used CAPTCHA benchmark dataset.

| Property | Value |
|---|---|
| Total images | 1,040 |
| Image dimensions | 200 × 50 px (grayscale) |
| Label length | 5 characters (fixed) |
| Vocabulary | 19 characters: `2 3 4 5 6 7 8 b c d e f g m n p w x y` |
| Train split | 728 images (70%) |
| Validation split | 208 images (20%) |
| Test split | 104 images (10%) |

Data augmentation applied to training images: Gaussian noise (σ ≈ 2%), random brightness (±10%), random contrast ([0.8, 1.2]).

---

## Performance

Evaluated on the held-out test split (104 images, never seen during training):

| Metric | Value |
|---|---|
| Word Accuracy (exact 5-char match) | **99.04%** |
| Character Accuracy | **99.81%** |
| Character Error Rate (CER) | **0.19%** |
| Correct / Total | 103 / 104 |

Training converged in ~10 epochs with the default configuration. Validation loss tracked training loss throughout — no overfitting observed.

---

## Project Structure

```
Captcha_Recognition_Using_CRNN/
├── app.py                          # Streamlit web application
├── model.py                        # CRNN model definition
├── train.py                        # Training pipeline
├── data_utils.py                   # Data loading, augmentation, CTC decoding, metrics
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows one-click launcher
│
├── saved_model/
│   ├── crnn_model.weights.h5       # Pre-trained weights (~30 MB)
│   ├── metadata.json               # Character vocabulary + training metrics
│   ├── training_history.json       # Per-epoch loss history
│   ├── loss_curve.png              # Training/validation loss plot
│   └── sample_predictions.png     # Test set prediction grid
│
└── Captcha_Recognition_using_CRNN/
    └── captcha_images_v2/          # Dataset images (extracted on first run)
```

---

## Installation

**Requirements:** Python 3.9 or later, pip.

```bash
# 1. Clone the repository
git clone https://github.com/Ansh1015/Captcha_Recognition_Using_CRNN.git
cd Captcha_Recognition_Using_CRNN

# 2. Install dependencies
pip install -r requirements.txt
```

Dependencies:

| Package | Version |
|---|---|
| tensorflow | ≥ 2.13.0 |
| streamlit | ≥ 1.28.0 |
| numpy | ≥ 1.24.0 |
| matplotlib | ≥ 3.7.0 |
| Pillow | ≥ 10.0.0 |
| pandas | ≥ 2.0.0 |

---

## Running the App

**Windows (one-click):**

```
Double-click run_app.bat
```

**All platforms:**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### App Tabs

**Predict**
- Upload any CAPTCHA image (PNG/JPG) for instant inference.
- Roll random samples from the dataset to see live batch predictions with accuracy indicators.
- Vocabulary activation bar chart shows which characters the model activated for each prediction.

**Dashboard**
- Training/validation loss curves (linear and log scale).
- Test-set performance metrics.
- Sample prediction grid from the held-out test set.
- Full character vocabulary and model specification.

**Retrain**
- Re-run training with custom hyperparameters (epochs, batch size, learning rate, augmentation, attention).
- The pre-trained model remains active while you experiment.
- Live loss curve updates during training.

> **Note:** The pre-trained model loads automatically on startup. No training required before predicting.

---

## Training

To retrain from the command line:

```bash
python train.py
```

Or with custom hyperparameters:

```python
from train import train_model

results = train_model(
    epochs=30,
    batch_size=16,
    learning_rate=1e-3,
    use_attention=True,
    augment=True,
)
print(f"Word accuracy: {results['metrics']['word_accuracy']*100:.2f}%")
```

**Default hyperparameters:**

| Parameter | Default |
|---|---|
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 4) |
| Early stopping | patience 7, restore best weights |
| Augmentation | Enabled |
| Self-attention | Enabled |

Trained weights, metadata, loss curve, and sample predictions are saved to `saved_model/` automatically.

---

## Model Limitations

This model is purpose-built for the `captcha_images_v2` dataset. It will not generalize well to:

- **Uppercase letters** — the vocabulary contains only lowercase letters and digits `2–8`. Characters like `A`, `B`, `M` are not in the training distribution.
- **Different visual styles** — different fonts, colors, or noise patterns from other CAPTCHA systems are out of distribution.
- **Non-fixed-length CAPTCHAs** — the model expects exactly 5 characters.
- **Languages or symbols** — only the 19 characters listed in the dataset vocabulary are supported.

For general-purpose CAPTCHA recognition across multiple systems, consider fine-tuning on a broader dataset or using a pre-trained scene-text recognizer.

---

## Tech Stack

- **TensorFlow / Keras** — model definition, training, CTC loss and decoding
- **Streamlit** — web interface for live inference and dashboard
- **NumPy / Pillow** — image preprocessing
- **Matplotlib** — training visualizations
- **Python 3.9+**
