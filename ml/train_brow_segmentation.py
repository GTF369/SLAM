#!/usr/bin/env python3
"""Train a compact eyebrow-ROI segmentation model and export a LiteRT/TFLite model.

The training data must be manually annotated with appropriate permission. Each image
must be a cropped single-brow ROI with a same-size binary mask. White mask pixels are
eyebrow hair/visible eyebrow boundary, black is background. A manifest.csv contains
image,mask,case_id columns with paths relative to the dataset root.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


IMAGE_SIZE = 256


def read_manifest(root: Path) -> list[dict[str, str]]:
    manifest = root / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"Missing {manifest}. Run annotate_brow_masks.py and check its output.")
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image", "mask", "case_id"}
    if not rows or not required.issubset(rows[0].keys()):
        raise SystemExit("manifest.csv must have image, mask, case_id columns.")
    for i, row in enumerate(rows, start=2):
        for key in ("image", "mask", "case_id"):
            if not row.get(key, "").strip():
                raise SystemExit(f"manifest.csv row {i} has an empty {key}.")
        if not (root / row["image"]).is_file() or not (root / row["mask"]).is_file():
            raise SystemExit(f"manifest.csv row {i} refers to a missing image or mask.")
    return rows


def load_samples(root: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    groups: list[str] = []
    for row in rows:
        image = tf.keras.utils.load_img(root / row["image"], color_mode="rgb", target_size=(IMAGE_SIZE, IMAGE_SIZE), interpolation="bilinear")
        mask = tf.keras.utils.load_img(root / row["mask"], color_mode="grayscale", target_size=(IMAGE_SIZE, IMAGE_SIZE), interpolation="nearest")
        x = tf.keras.utils.img_to_array(image).astype(np.float32) / 255.0
        y = (tf.keras.utils.img_to_array(mask) >= 127.5).astype(np.float32)
        images.append(x)
        masks.append(y)
        groups.append(row["case_id"].strip())
    return np.stack(images), np.stack(masks), np.asarray(groups)


def group_split(groups: np.ndarray, seed: int, test_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(groups.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_cases = set(unique[:n_test])
    test_indices = np.asarray([i for i, group in enumerate(groups) if group in test_cases], dtype=np.int64)
    train_indices = np.asarray([i for i, group in enumerate(groups) if group not in test_cases], dtype=np.int64)
    if not len(train_indices) or not len(test_indices):
        raise SystemExit("Could not create a case-grouped holdout split.")
    return train_indices, test_indices


def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predicted = (y_pred >= 0.5).astype(np.float32)
    numerator = 2.0 * float((predicted * y_true).sum()) + 1e-6
    denominator = float(predicted.sum() + y_true.sum()) + 1e-6
    return numerator / denominator


def iou_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predicted = (y_pred >= 0.5).astype(bool)
    truth = y_true.astype(bool)
    intersection = np.logical_and(predicted, truth).sum()
    union = np.logical_or(predicted, truth).sum()
    return float((intersection + 1e-6) / (union + 1e-6))


def dice_bce(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    axes = [1, 2, 3]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + tf.reduce_mean(dice_loss)


def conv_block(x, filters: int):
    x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)


def build_model():
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="brow_roi_rgb_0_1")
    e1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(e1)
    e2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(e2)
    e3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPooling2D()(e3)
    bottleneck = conv_block(p3, 96)
    u3 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(bottleneck)
    u3 = tf.keras.layers.Concatenate()([u3, e3])
    d3 = conv_block(u3, 64)
    u2 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(d3)
    u2 = tf.keras.layers.Concatenate()([u2, e2])
    d2 = conv_block(u2, 32)
    u1 = tf.keras.layers.UpSampling2D(interpolation="bilinear")(d2)
    u1 = tf.keras.layers.Concatenate()([u1, e1])
    d1 = conv_block(u1, 16)
    output = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="brow_mask_probability")(d1)
    return tf.keras.Model(inputs, output, name="brow_roi_unet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/brow_segmentation_candidate.tflite"))
    parser.add_argument("--metrics", type=Path, default=Path("ml/brow_segmentation_metrics.json"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-cases", type=int, default=50,
                        help="Minimum distinct annotated cases for an exploratory model. Does not imply production readiness.")
    parser.add_argument("--promote-to-app", action="store_true",
                        help="Copy candidate into app assets only when the grouped holdout Dice meets the threshold.")
    parser.add_argument("--min-validation-dice", type=float, default=0.70,
                        help="Minimum grouped holdout Dice required for --promote-to-app.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    rows = read_manifest(args.dataset)
    x, y, groups = load_samples(args.dataset, rows)
    unique_cases = sorted(set(groups.tolist()))
    if len(unique_cases) < args.min_cases:
        raise SystemExit(
            f"Refusing to train: dataset has {len(unique_cases)} distinct case IDs, below --min-cases={args.min_cases}. "
            "Use consented, manually reviewed labels. Small-sample experiments require an explicit lower threshold and are not production models."
        )

    train_idx, test_idx = group_split(groups, args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    # Apply geometric augmentation to both image and mask so supervision remains aligned.
    def augment_pair(image, mask):
        flip = tf.random.uniform(()) > 0.5
        image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
        return tf.clip_by_value(image, 0.0, 1.0), mask

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(x_train), seed=args.seed, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(args.batch_size).map(
        augment_pair,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=dice_bce,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]
    history = model.fit(
        train_ds,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    predicted = model.predict(x_test, batch_size=args.batch_size, verbose=0)
    metrics = {
        "model": "compact_brow_roi_unet",
        "input_shape": [1, IMAGE_SIZE, IMAGE_SIZE, 3],
        "output_shape": [1, IMAGE_SIZE, IMAGE_SIZE, 1],
        "distinct_cases": len(unique_cases),
        "train_cases": len(set(groups[train_idx].tolist())),
        "validation_cases": len(set(groups[test_idx].tolist())),
        "train_rois": int(len(train_idx)),
        "validation_rois": int(len(test_idx)),
        "validation_dice": dice_score(y_test, predicted),
        "validation_iou": iou_score(y_test, predicted),
        "best_validation_loss": float(min(history.history.get("val_loss", [float("nan")]))),
        "warning": "Grouped internal holdout only. Independent prospective validation and target-device testing are required before commercial use.",
    }

    # The Android runtime expects float32 RGB [0,1] input and a float32
    # [1,256,256,1] sigmoid probability map.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    model_bytes = converter.convert()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(model_bytes)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Candidate TFLite model written to {args.output}")
    if args.promote_to_app:
        if metrics["validation_dice"] < args.min_validation_dice:
            raise SystemExit(
                f"Model not promoted: grouped validation Dice={metrics['validation_dice']:.3f} "
                f"is below --min-validation-dice={args.min_validation_dice:.3f}. "
                "Review annotations and collect more consented cases."
            )
        app_path = Path("app/src/main/assets/brow_segmentation.tflite")
        app_path.parent.mkdir(parents=True, exist_ok=True)
        app_path.write_bytes(model_bytes)
        print(f"Candidate copied into app assets: {app_path}")
        print("This is not a production release. Independent client and target-device validation remains required.")
    else:
        print("Candidate was not added to the app. Review the metric JSON and validation examples first.")
    print("Do not treat the grouped holdout score as proof of clinical, demographic, or commercial generalization.")


if __name__ == "__main__":
    main()
