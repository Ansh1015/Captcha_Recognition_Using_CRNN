"""Data loading, augmentation, and TF Dataset utilities."""

import json
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf


# ─── Dataset extraction ───────────────────────────────────────────────────────

def extract_dataset(zip_path: str, extract_dir: str) -> Path:
    """Extract zip archive only when the target directory does not yet exist."""
    out_dir = Path(extract_dir)
    if not out_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir.parent)
        print(f"Extracted {zip_path} -> {out_dir}")
    return out_dir


# ─── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(image_dir: str):
    """
    Load all CAPTCHA image paths and labels from a directory.

    Returns
    -------
    image_paths   : list[str]
    labels        : list[str]
    all_chars     : list[str]  sorted unique characters
    char_to_int   : dict[str, int]
    int_to_char   : dict[int, str]
    max_label_len : int
    """
    image_dir = Path(image_dir)
    image_paths = sorted(str(p) for p in image_dir.glob("*.png"))
    labels = [Path(p).stem for p in image_paths]

    all_chars = sorted(set("".join(labels)))
    char_to_int = {c: i for i, c in enumerate(all_chars)}
    int_to_char = {i: c for c, i in char_to_int.items()}
    max_label_len = max(len(lbl) for lbl in labels)

    return image_paths, labels, all_chars, char_to_int, int_to_char, max_label_len


# ─── Image preprocessing ──────────────────────────────────────────────────────

def preprocess_image(image_path, height: int, width: int) -> tf.Tensor:
    """Read, decode, and resize a PNG. Works with both str and tf.string tensors."""
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_png(raw, channels=1)
    img = tf.image.resize(img, (height, width))
    img = tf.cast(img, tf.float32)
    return img


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_image(image: tf.Tensor, label: tf.Tensor):
    """
    Lightweight augmentation for CAPTCHA images.

    • Gaussian noise  σ ≈ 2 % of pixel range
    • Random brightness ± 10 %
    • Random contrast  [0.8, 1.2]

    Images are float32 in [0, 255]; normalised internally for ops.
    """
    img = image / 255.0
    noise = tf.random.normal(tf.shape(img), stddev=0.02)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.80, upper=1.20)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img * 255.0, label


# ─── TF Dataset creation ──────────────────────────────────────────────────────

def create_tf_datasets(
    image_paths,
    labels,
    char_to_int,
    max_label_len: int,
    image_height: int,
    image_width: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    batch_size: int = 16,
    augment: bool = True,
    seed: int = 42,
):
    """
    Build tf.data.Dataset objects for train / validation / test splits.

    Returns (train_ds, val_ds, test_ds, split_sizes_dict).
    """
    total = len(image_paths)
    n_train = int(train_ratio * total)
    n_val   = int(val_ratio   * total)
    n_test  = total - n_train - n_val

    # Encode + pad labels to fixed length (CTC needs rectangular tensors)
    encoded_labels = []
    for lbl in labels:
        enc = [char_to_int[c] for c in lbl]
        enc += [enc[-1]] * (max_label_len - len(enc))
        encoded_labels.append(enc)
    encoded_labels = np.array(encoded_labels, dtype=np.float32)

    # Lazy dataset: store paths as strings, load images on demand
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(image_paths), encoded_labels)
    )
    ds = ds.shuffle(total, seed=seed, reshuffle_each_iteration=False)

    def load(path, label):
        return preprocess_image(path, image_height, image_width), label

    # Reshuffle training split each epoch so batches vary between epochs
    train_ds = (
        ds.take(n_train)
          .shuffle(n_train, reshuffle_each_iteration=True)
          .map(load, num_parallel_calls=tf.data.AUTOTUNE)
    )
    val_ds   = ds.skip(n_train).take(n_val).map(load, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds  = ds.skip(n_train + n_val).take(n_test).map(load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds  .batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds .batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, {"train": n_train, "val": n_val, "test": n_test}


# ─── Decoding helpers ─────────────────────────────────────────────────────────

def decode_predictions(raw_preds: tf.Tensor, int_to_char: dict) -> list:
    """CTC-greedy-decode a batch of model outputs → list of strings."""
    batch_size = tf.shape(raw_preds)[0]
    seq_len    = tf.shape(raw_preds)[1]
    input_lengths = tf.fill([batch_size], seq_len)

    decoded, _ = tf.keras.backend.ctc_decode(
        raw_preds, input_length=input_lengths, greedy=True
    )
    results = []
    for row in decoded[0].numpy():
        text = "".join(int_to_char[idx] for idx in row if idx != -1)
        results.append(text)
    return results


# ─── Metrics ──────────────────────────────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def calculate_metrics(model, dataset, int_to_char: dict) -> dict:
    """
    Evaluate model on a tf.data.Dataset.

    Returns dict with:
        word_accuracy  – exact full-string match rate
        char_accuracy  – per-character correct rate
        cer            – mean Character Error Rate
        total          – samples evaluated
        correct        – exact matches
        predictions    – list of (pred, true) tuples
    """
    total = correct = char_correct = char_total = 0
    cer_sum = 0.0
    all_preds = []

    for batch_imgs, batch_labels in dataset:
        preds = model.predict(batch_imgs, verbose=0)
        pred_texts = decode_predictions(preds, int_to_char)

        for i, pred in enumerate(pred_texts):
            true_indices = [int(x) for x in batch_labels[i].numpy()]
            true_text = "".join(int_to_char[idx] for idx in true_indices)

            if pred == true_text:
                correct += 1
            total += 1

            for pc, tc in zip(pred.ljust(len(true_text)), true_text):
                char_correct += int(pc == tc)
                char_total += 1

            ed = levenshtein(pred, true_text)
            cer_sum += ed / max(len(true_text), 1)

            all_preds.append((pred, true_text))

    return {
        "word_accuracy": correct / total if total else 0.0,
        "char_accuracy": char_correct / char_total if char_total else 0.0,
        "cer":           cer_sum / total if total else 0.0,
        "total":         total,
        "correct":       correct,
        "predictions":   all_preds,
    }


# ─── Metadata persistence ─────────────────────────────────────────────────────

def save_metadata(
    path: str,
    char_to_int: dict,
    int_to_char: dict,
    all_chars: list,
    max_label_len: int,
    image_height: int,
    image_width: int,
    metrics: Optional[dict] = None,
    use_attention: bool = True,
):
    """Persist character mapping and model settings to a JSON file."""
    data = {
        "char_to_int":   char_to_int,
        "int_to_char":   {str(k): v for k, v in int_to_char.items()},
        "all_chars":     all_chars,
        "max_label_len": max_label_len,
        "image_height":  image_height,
        "image_width":   image_width,
        "use_attention": use_attention,
    }
    if metrics:
        data["metrics"] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in metrics.items()
            if k != "predictions"
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_metadata(path: str) -> dict:
    """Load metadata JSON, restoring int_to_char keys as ints."""
    with open(path) as f:
        data = json.load(f)
    data["int_to_char"] = {int(k): v for k, v in data["int_to_char"].items()}
    return data
