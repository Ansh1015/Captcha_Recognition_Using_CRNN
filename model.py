"""Improved CRNN model for CAPTCHA recognition.

Improvements over baseline:
- Double conv blocks (VGG-style) for richer feature extraction
- Second BiLSTM layer for deeper sequence modeling
- Multi-head self-attention for global context
- Dynamic batch-size-safe CTC loss
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Bidirectional, Conv2D,
    Dense, Dropout, Input, Lambda, LayerNormalization, LSTM,
    MaxPooling2D, MultiHeadAttention, Rescaling, Reshape,
)
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def ctc_loss_fn(y_true, y_pred):
    """CTC loss that handles any batch size (including the last partial batch)."""
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
    input_len = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_len = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_lengths = input_len * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_lengths = label_len * tf.ones(shape=(batch_len, 1), dtype="int64")

    return K.ctc_batch_cost(y_true, y_pred, input_lengths, label_lengths)


def build_crnn_model(
    image_height: int,
    image_width: int,
    num_classes: int,
    use_attention: bool = True,
) -> Model:
    """
    Build the improved CRNN model.

    Architecture:
        Input (H x W x 1)
        → Rescaling → Transpose (W x H x 1)
        → CNN block-1: Conv64 × 2 + BN + Pool(2,2)
        → CNN block-2: Conv128 × 2 + BN + Pool(2,2)
        → CNN block-3: Conv256 × 2 + BN + Pool(2,1)
        → Reshape → Dense(256) → Dropout
        → BiLSTM(128) → BiLSTM(64)
        → [optional] MultiHeadAttention + residual + LayerNorm
        → Dense(num_classes + 1, softmax)

    After 3 poolings on the width axis and 2 poolings on the height axis:
        seq_len  = image_width  // 8   (e.g. 200 → 25)
        feat_dim = (image_height // 4) * 256  (e.g. 50 → 3072)
    """
    inp = Input(shape=(image_height, image_width, 1), name="input_image")

    # Normalise to [0, 1]
    x = Rescaling(1.0 / 255)(inp)

    # Width → sequence axis
    x = Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1, 3]), name="transpose")(x)

    # ── Block 1 ─────────────────────────────────────────────────────
    for suffix in ("a", "b"):
        x = Conv2D(64, (3, 3), padding="same",
                   kernel_initializer="he_normal", name=f"conv1{suffix}")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)
    x = Dropout(0.10)(x)

    # ── Block 2 ─────────────────────────────────────────────────────
    for suffix in ("a", "b"):
        x = Conv2D(128, (3, 3), padding="same",
                   kernel_initializer="he_normal", name=f"conv2{suffix}")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)
    x = Dropout(0.15)(x)

    # ── Block 3 ─────────────────────────────────────────────────────
    for suffix in ("a", "b"):
        x = Conv2D(256, (3, 3), padding="same",
                   kernel_initializer="he_normal", name=f"conv3{suffix}")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    x = MaxPooling2D((2, 1), name="pool3")(x)   # keep height; halve width
    x = Dropout(0.20)(x)

    # ── Reshape to sequence ──────────────────────────────────────────
    seq_len = image_width // 8            # 25
    feat_dim = (image_height // 4) * 256  # 3072
    x = Reshape(target_shape=(seq_len, feat_dim), name="reshape")(x)

    # ── Dense projection ────────────────────────────────────────────
    x = Dense(256, activation="relu",
               kernel_initializer="he_normal", name="dense_proj")(x)
    x = Dropout(0.25)(x)

    # ── Recurrent layers ────────────────────────────────────────────
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25),
                       name="bilstm1")(x)
    x = Bidirectional(LSTM(64,  return_sequences=True, dropout=0.20),
                       name="bilstm2")(x)

    # ── Self-attention ───────────────────────────────────────────────
    if use_attention:
        residual = x
        attn = MultiHeadAttention(num_heads=4, key_dim=32, name="mha")(x, x)
        x = Add()([residual, attn])
        x = LayerNormalization(name="ln_attn")(x)

    # ── Output ───────────────────────────────────────────────────────
    out = Dense(num_classes + 1, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="improved_CRNN")
