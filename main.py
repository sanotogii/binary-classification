import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras import layers, models




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN binary classifier (TensorFlow)")
    parser.add_argument("--data-dir", default="data_split", help="Folder with train/val/test")
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of epochs with no improvement before stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum change in monitored metric to qualify as an improvement",
    )
    parser.add_argument("--model-out", default="artifacts/algae.keras")
    return parser.parse_args()


def load_dataset(path: Path, img_size: int, batch_size: int, shuffle: bool):
    if not path.exists():
        return None
        
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
    ).ignore_errors().prefetch(tf.data.AUTOTUNE)


def build_model(img_size: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze pretrained weights

    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    # MobileNetV2 expects pixels in [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs, outputs)


def evaluate_metrics(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict:
    if dataset is None:
        return {}

    y_true = []
    y_pred = []
    
    # Iterate over the dataset to get all predictions and labels
    print("Evaluating...")
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        # Convert probabilities to binary predictions (threshold 0.5)
        preds_binary = (preds.flatten() >= 0.5).astype(int)
        
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds_binary.tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def plot_learning_curves(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Plot training & validation loss and accuracy curves."""
    epochs = range(1, len(history.history["loss"]) + 1)
    plot_kwargs = dict(marker="o", markersize=4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history.history["loss"], label="Train", **plot_kwargs)
    if "val_loss" in history.history:
        ax1.plot(epochs, history.history["val_loss"], label="Validation", **plot_kwargs)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(list(epochs))
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history.history["accuracy"], label="Train", **plot_kwargs)
    if "val_accuracy" in history.history:
        ax2.plot(epochs, history.history["val_accuracy"], label="Validation", **plot_kwargs)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_xticks(list(epochs))
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Learning curves saved to {output_path}")


def update_readme_metrics(readme_path: Path, test_metrics: dict) -> None:
    if not readme_path.exists():
        return
    content = readme_path.read_text()

    metrics_lines = [
        "- Accuracy (test): {value:.4f}",
        "- Precision (test): {value:.4f}",
        "- Recall (test): {value:.4f}",
        "- F1-score (test): {value:.4f}",
    ]
    replacements = {
        "Accuracy": test_metrics.get("accuracy", 0.0),
        "Precision": test_metrics.get("precision", 0.0),
        "Recall": test_metrics.get("recall", 0.0),
        "F1-score": test_metrics.get("f1_score", 0.0),
    }

    updated_lines = []
    for line in content.splitlines():
        updated = line
        for key, value in replacements.items():
            pattern = rf"^- {re.escape(key)} \(test\):"
            if re.match(pattern, line):
                template = next(t for t in metrics_lines if t.startswith(f"- {key}"))
                updated = template.format(value=value)
                break
        updated_lines.append(updated)

    readme_path.write_text("\n".join(updated_lines) + "\n")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    print(f"Loading data from {data_dir}...")
    train_ds = load_dataset(data_dir / "train", args.img_size, args.batch_size, shuffle=True)
    val_ds = load_dataset(data_dir / "val", args.img_size, args.batch_size, shuffle=False)
    test_ds = load_dataset(data_dir / "test", args.img_size, args.batch_size, shuffle=False)

    if train_ds is None:
        print("Training dataset not found! Please run data preparation first.")
        return

    model = build_model(args.img_size)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    monitor_metric = "val_loss" if val_ds is not None else "loss"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    print("Starting training...")
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=args.epochs,
        callbacks=callbacks,
    )

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {args.model_out}...")
    model.save(args.model_out)

    plot_learning_curves(history, artifacts_dir / "learning_curves.png")

    metrics = {}
    print("Collecting metrics...")
    metrics["train"] = evaluate_metrics(model, train_ds)
    if val_ds:
        metrics["val"] = evaluate_metrics(model, val_ds)
    if test_ds:
        metrics["test"] = evaluate_metrics(model, test_ds)
        update_readme_metrics(Path("README.md"), metrics["test"])

    with (artifacts_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    
    # Save some meta info for the app
    # We retrieve class names from the directory structure because PrefetchDataset loses the attribute
    class_names = sorted([d.name for d in (data_dir / "train").iterdir() if d.is_dir()])
    with (artifacts_dir / "meta.json").open("w") as handle:
        json.dump({
            "class_names": class_names,
            "img_size": args.img_size
        }, handle, indent=2)


if __name__ == "__main__":
    main()
