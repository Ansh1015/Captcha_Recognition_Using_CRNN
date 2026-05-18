"""
Training pipeline for the improved CRNN CAPTCHA recognizer.

Run standalone:
    python train.py

Or import train_model() from Streamlit / other scripts.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

from data_utils import (
    calculate_metrics,
    create_tf_datasets,
    extract_dataset,
    load_dataset,
    load_metadata,
    save_metadata,
)
from model import build_crnn_model, ctc_loss_fn

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
ZIP_PATH     = PROJECT_ROOT / "Captcha_Recognition_using_CRNN" / "captcha_images_v2.zip"
IMAGE_DIR    = PROJECT_ROOT / "Captcha_Recognition_using_CRNN" / "captcha_images_v2"
SAVE_DIR     = PROJECT_ROOT / "saved_model"
WEIGHTS_PATH = SAVE_DIR / "crnn_model.weights.h5"
META_PATH    = SAVE_DIR / "metadata.json"
HISTORY_PATH = SAVE_DIR / "training_history.json"
SAMPLE_PATH  = SAVE_DIR / "sample_predictions.png"

# ─── Default hyperparameters ──────────────────────────────────────────────────
DEFAULTS = dict(
    image_height  = 50,
    image_width   = 200,
    batch_size    = 16,
    epochs        = 30,
    learning_rate = 1e-3,
    use_attention = True,
    augment       = True,
)


# ─── Progress callback ────────────────────────────────────────────────────────

class ProgressCallback(tf.keras.callbacks.Callback):
    """
    Writes per-epoch metrics to a JSON file so external processes
    (e.g. Streamlit) can poll it for live updates.
    """

    def __init__(self, progress_file: Path, total_epochs: int,
                 on_epoch_end_fn=None):
        super().__init__()
        self.progress_file = Path(progress_file)
        self.total_epochs  = total_epochs
        self._fn           = on_epoch_end_fn   # optional callable(epoch, logs)
        self._history      = {"loss": [], "val_loss": []}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._history["loss"].append(float(logs.get("loss", 0)))
        self._history["val_loss"].append(float(logs.get("val_loss", 0)))

        payload = {
            "epoch":       epoch + 1,
            "total":       self.total_epochs,
            "loss":        float(logs.get("loss", 0)),
            "val_loss":    float(logs.get("val_loss", 0)),
            "lr":          float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),
            "history":     self._history,
            "done":        False,
        }
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, "w") as f:
            json.dump(payload, f)

        if self._fn:
            self._fn(epoch, logs)

    def on_train_end(self, logs=None):
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                payload = json.load(f)
            payload["done"] = True
            with open(self.progress_file, "w") as f:
                json.dump(payload, f)


# ─── Sample-prediction visualiser ────────────────────────────────────────────

def save_sample_grid(model, test_ds, int_to_char, max_label_len, save_path, n=10):
    """Save a PNG grid (5-column) showing model predictions vs ground truth."""
    from data_utils import decode_predictions

    # Collect up to n samples
    samples = []
    for batch_imgs, batch_labels in test_ds:
        if len(samples) >= n:
            break
        preds      = model.predict(batch_imgs, verbose=0)
        pred_texts = decode_predictions(preds, int_to_char)

        for i in range(min(len(pred_texts), n - len(samples))):
            true_text = "".join(
                int_to_char[int(x)] for x in batch_labels[i].numpy()
            )
            samples.append({
                "image": batch_imgs[i, :, :, 0].numpy(),
                "pred":  pred_texts[i],
                "true":  true_text,
                "ok":    pred_texts[i] == true_text,
            })

    n_s  = len(samples)
    cols = 5
    rows = max(1, (n_s + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.0),
                              squeeze=False)
    fig.suptitle("Test Set Predictions", fontsize=13, fontweight="bold")

    for i, s in enumerate(samples):
        r, c  = i // cols, i % cols
        ax    = axes[r, c]
        color = "green" if s["ok"] else "red"
        mark  = "+" if s["ok"] else "x"
        ax.imshow(s["image"], cmap="gray")
        ax.set_title(
            f'[{mark}] pred: "{s["pred"]}"\ntrue: "{s["true"]}"',
            fontsize=7, color=color,
        )
        ax.axis("off")

    # Hide any unused grid cells
    for i in range(n_s, rows * cols):
        axes[i // cols, i % cols].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    return save_path


# ─── Core training function ───────────────────────────────────────────────────

def train_model(
    image_height:  int   = DEFAULTS["image_height"],
    image_width:   int   = DEFAULTS["image_width"],
    batch_size:    int   = DEFAULTS["batch_size"],
    epochs:        int   = DEFAULTS["epochs"],
    learning_rate: float = DEFAULTS["learning_rate"],
    use_attention: bool  = DEFAULTS["use_attention"],
    augment:       bool  = DEFAULTS["augment"],
    on_epoch_end_fn=None,
    zip_path:      str   = str(ZIP_PATH),
    image_dir:     str   = str(IMAGE_DIR),
    save_dir:      str   = str(SAVE_DIR),
):
    """
    Full training pipeline.

    Parameters
    ----------
    on_epoch_end_fn : callable(epoch, logs) | None
        Optional hook for streaming per-epoch metrics to a UI.

    Returns
    -------
    dict with keys: model, history, metrics, metadata
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────
    extract_dataset(zip_path, image_dir)
    image_paths, labels, all_chars, char_to_int, int_to_char, max_label_len = (
        load_dataset(image_dir)
    )

    train_ds, val_ds, test_ds, splits = create_tf_datasets(
        image_paths, labels, char_to_int, max_label_len,
        image_height, image_width,
        batch_size=batch_size,
        augment=augment,
    )

    print(f"Dataset: {splits['train']} train / {splits['val']} val / {splits['test']} test")
    print(f"Characters ({len(all_chars)}): {''.join(all_chars)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = build_crnn_model(
        image_height, image_width,
        num_classes=len(all_chars),
        use_attention=use_attention,
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=ctc_loss_fn,
    )
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────
    progress_file = save_dir / "progress.json"
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(save_dir / "crnn_model.weights.h5"),
            monitor="val_loss", save_best_only=True,
            save_weights_only=True, verbose=1,
        ),
        ProgressCallback(
            progress_file=progress_file,
            total_epochs=epochs,
            on_epoch_end_fn=on_epoch_end_fn,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    # EarlyStopping(restore_best_weights=True) already restored the best epoch
    # weights into memory when training ended. Re-save them so the file
    # always reflects exactly what is evaluated below. Keras 3's
    # ModelCheckpoint can write stale/incorrect weights in some TF builds.
    weights_path = save_dir / "crnn_model.weights.h5"
    model.save_weights(str(weights_path))

    # ── Evaluate ──────────────────────────────────────────────────────
    print("\nEvaluating on test set …")
    metrics = calculate_metrics(model, test_ds, int_to_char)

    print(f"\n{'='*50}")
    print(f"  Test Word Accuracy : {metrics['word_accuracy']*100:.2f}%")
    print(f"  Test Char Accuracy : {metrics['char_accuracy']*100:.2f}%")
    print(f"  Test CER           : {metrics['cer']*100:.2f}%")
    print(f"  Correct / Total    : {metrics['correct']} / {metrics['total']}")
    print(f"{'='*50}\n")

    # ── Save metadata ─────────────────────────────────────────────────
    save_metadata(
        str(save_dir / "metadata.json"),
        char_to_int, int_to_char, all_chars,
        max_label_len, image_height, image_width,
        metrics,
        use_attention=use_attention,
    )

    # Save training history
    hist_dict = {
        k: [float(v) for v in vals]
        for k, vals in history.history.items()
    }
    hist_dict["splits"] = splits
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    # Save loss curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist_dict["loss"],     label="Training Loss",   color="#1f77b4")
    ax.plot(hist_dict["val_loss"], label="Validation Loss", color="#ff7f0e")
    best_ep = int(np.argmin(hist_dict["val_loss"]))
    ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.6,
               label=f"Best epoch {best_ep+1}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("CTC Loss")
    ax.set_title("Training History"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_dir / "loss_curve.png"), dpi=120)
    plt.close()

    # Save sample predictions grid
    save_sample_grid(model, test_ds, int_to_char, max_label_len,
                     save_path=str(save_dir / "sample_predictions.png"))

    print(f"Artefacts saved to {save_dir}")

    return {
        "model":    model,
        "history":  hist_dict,
        "metrics":  metrics,
        "metadata": load_metadata(str(save_dir / "metadata.json")),
    }


# ─── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    results = train_model()
    m = results["metrics"]
    print(f"Training complete. Word accuracy: {m['word_accuracy']*100:.2f}%")
