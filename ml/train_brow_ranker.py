#!/usr/bin/env python3
"""Train a grouped, preference-ranking baseline from the app's anonymous CSV export.

This script deliberately refuses to fabricate a model when there are too few cases.
The generated JSON is consumed by BrowRecommendationEngine in the Android app.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "face_width_height_ratio",
    "interocular_face_width_ratio",
    "jaw_face_width_ratio",
    "brow_eye_distance_ratio",
    "brow_asymmetry_ratio",
    "eye_line_tilt",
    "thickness",
    "length",
    "arch",
    "style_soft",
    "style_defined",
]
REQUIRED = {
    "case_id", "style", "thickness", "length", "arch",
    "rating", "face_width_height_ratio", "interocular_face_width_ratio",
    "jaw_face_width_ratio", "brow_eye_distance_ratio", "brow_asymmetry_ratio", "eye_line_tilt",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("CSV is empty.")
    missing = REQUIRED.difference(rows[0].keys())
    if missing:
        raise ValueError("Missing required CSV columns: " + ", ".join(sorted(missing)))
    return rows


def build_matrix(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for row_num, row in enumerate(rows, start=2):
        case_id = row["case_id"].strip()
        style = row["style"].strip().lower()
        if not case_id:
            raise ValueError(f"Row {row_num} has no case_id. Cases must be grouped to avoid train/test leakage.")
        if style not in {"natural", "soft", "defined"}:
            raise ValueError(f"Row {row_num} has unknown style {style!r}.")
        try:
            thickness = float(row["thickness"])
            length = float(row["length"])
            arch = float(row["arch"])
            rating = int(float(row["rating"]))
            face_ratio = float(row["face_width_height_ratio"])
            interocular_ratio = float(row["interocular_face_width_ratio"])
            jaw_ratio = float(row["jaw_face_width_ratio"])
            brow_eye_ratio = float(row["brow_eye_distance_ratio"])
            asymmetry = float(row["brow_asymmetry_ratio"])
            eye_tilt = float(row["eye_line_tilt"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {row_num} has an invalid numeric feature.") from exc
        values = [
            face_ratio, interocular_ratio, jaw_ratio, brow_eye_ratio, asymmetry, eye_tilt,
            thickness, length, arch,
            1.0 if style == "soft" else 0.0,
            1.0 if style == "defined" else 0.0,
        ]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Row {row_num} contains NaN or infinite feature values.")
        if rating < 1 or rating > 5:
            raise ValueError(f"Row {row_num} rating must be in the 1–5 range.")
        features.append(values)
        # A rating is the label. The final-selected flag is not a model input and is not
        # used as the label, because a client may like more than one candidate.
        labels.append(1 if rating >= 4 else 0)
        groups.append(case_id)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), np.asarray(groups)


def choose_group_split(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    for attempt in range(40):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        train_idx, test_idx = next(splitter.split(x, y, groups))
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
            return train_idx, test_idx
    raise ValueError("Could not create a case-grouped train/test split with both rating classes. Collect more varied ratings across more clients.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Anonymous preference feedback CSV exported from the app")
    parser.add_argument("--output", type=Path, default=Path("app/src/main/assets/brow_ranker.json"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    x, y, groups = build_matrix(rows)
    case_count = len(np.unique(groups))
    if case_count < 30:
        raise SystemExit(
            f"Refusing to train: only {case_count} unique cases. Collect at least 30 distinct cases for an initial experiment; "
            "for any production claim, target hundreds or more and use a separate prospective validation set."
        )
    if len(np.unique(y)) != 2:
        raise SystemExit("Ratings do not contain both positive (4–5) and negative (1–3) examples.")

    train_idx, test_idx = choose_group_split(x, y, groups, args.seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    model.fit(x_train, y[train_idx])
    predicted = model.predict(x_test)
    scores = model.predict_proba(x_test)[:, 1]
    metrics = {
        "group_split": "GroupShuffleSplit by case_id, 80/20",
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_cases": int(len(np.unique(groups[train_idx]))),
        "test_cases": int(len(np.unique(groups[test_idx]))),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], predicted)),
        "roc_auc": float(roc_auc_score(y[test_idx], scores)),
        "note": "Exploratory only. Scores are not clinically or statistically calibrated probabilities.",
    }
    # The holdout metrics above remain tied to models fitted only on training cases.
    # Once that assessment is recorded, refit the deployable candidate on all available
    # consented cases so the final JSON does not discard the grouped holdout observations.
    scaler = StandardScaler()
    x_all = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    model.fit(x_all, y)
    metrics["candidate_fit_scope"] = "Refit on all available cases after grouped holdout evaluation"
    metrics["holdout_metrics_are_from_final_all_data_fit"] = False

    payload = {
        "model_version": 1,
        "feature_order": FEATURE_NAMES,
        "means": [float(v) for v in scaler.mean_],
        "scales": [float(v) if float(v) != 0.0 else 1.0 for v in scaler.scale_],
        "coefficients": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "training_metrics": metrics,
        "trained_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "source_cases": case_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Model JSON written to: {args.output}")
    print("Review group-level performance and professional safety before deploying. Do not describe the score as a guaranteed best design.")


if __name__ == "__main__":
    main()
