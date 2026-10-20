#!/usr/bin/env python3
"""Train a grouped preference model with optional calibration and independent validation.

The output is an Android asset candidate. It is deliberately NOT deployable by default.
For a user-facing preference estimate, use disjoint development, calibration, and external
validation customers. An explicit human-review flag is required to mark the model releasable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
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
    "native_brow_area_fraction",
    "native_brow_side_imbalance",
    "native_brow_mask_available",
    "native_shape_mode",
]
REQUIRED = {
    "case_id", "style", "thickness", "length", "arch", "rating",
    "face_width_height_ratio", "interocular_face_width_ratio",
    "jaw_face_width_ratio", "brow_eye_distance_ratio",
    "brow_asymmetry_ratio", "eye_line_tilt",
    "native_brow_area_fraction", "native_brow_side_imbalance",
    "native_brow_mask_available", "native_shape_mode",
}


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    missing = REQUIRED.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{path} is missing columns: " + ", ".join(sorted(missing)))
    return rows


def build_matrix(rows: list[dict[str, Any]], source: str = "CSV"):
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    for row_num, row in enumerate(rows, start=2):
        case_id = row["case_id"].strip()
        style = row["style"].strip().lower()
        if not case_id:
            raise ValueError(f"{source} row {row_num} has no case_id; group by anonymous customer.")
        if style not in {"natural", "soft", "defined"}:
            raise ValueError(f"{source} row {row_num} has unknown style {style!r}.")
        try:
            thickness, length, arch = (float(row[k]) for k in ("thickness", "length", "arch"))
            rating = float(row["rating"])
            face_ratio = float(row["face_width_height_ratio"])
            interocular_ratio = float(row["interocular_face_width_ratio"])
            jaw_ratio = float(row["jaw_face_width_ratio"])
            brow_eye_ratio = float(row["brow_eye_distance_ratio"])
            asymmetry = float(row["brow_asymmetry_ratio"])
            eye_tilt = float(row["eye_line_tilt"])
            native_area = float(row["native_brow_area_fraction"])
            native_side_imbalance = float(row["native_brow_side_imbalance"])
            native_mask_available = float(row["native_brow_mask_available"])
            native_shape_mode = float(row["native_shape_mode"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source} row {row_num} has an invalid numeric value.") from exc
        values = [
            face_ratio, interocular_ratio, jaw_ratio, brow_eye_ratio, asymmetry, eye_tilt,
            thickness, length, arch,
            1.0 if style == "soft" else 0.0,
            1.0 if style == "defined" else 0.0,
            native_area,
            native_side_imbalance,
            native_mask_available,
            native_shape_mode,
        ]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{source} row {row_num} contains NaN or infinity.")
        if not (0.0 <= native_area <= 1.0 and 0.0 <= native_side_imbalance <= 1.0):
            raise ValueError(f"{source} row {row_num} has out-of-range brow-mask fractions.")
        if native_mask_available not in (0.0, 1.0) or native_shape_mode not in (0.0, 1.0):
            raise ValueError(f"{source} row {row_num} must use 0/1 for mask availability and native shape mode.")
        if not math.isfinite(rating) or rating < 1 or rating > 5:
            raise ValueError(f"{source} row {row_num} rating must be in the 1–5 range.")
        features.append(values)
        labels.append(1 if rating >= 4 else 0)
        groups.append(case_id)
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{source} must contain both rating classes (1–3 and 4–5).")
    return x, y, group_array


def choose_group_split(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int):
    for attempt in range(80):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + attempt)
        train_idx, test_idx = next(splitter.split(x, y, groups))
        if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2:
            return train_idx, test_idx
    raise ValueError("Could not form a case-grouped 80/20 split containing both labels. Collect more diverse cases.")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        selected = (probabilities >= edges[i]) & (
            probabilities < edges[i + 1] if i < bins - 1 else probabilities <= edges[i + 1]
        )
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(float(np.mean(probabilities[selected])) - float(np.mean(y[selected])))
    return ece


def calibration_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    baseline = float(np.mean(y))
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y, probabilities >= 0.5)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "prevalence_baseline_brier_score": float(baseline * (1.0 - baseline)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "expected_calibration_error_10_bins": expected_calibration_error(y, probabilities),
        "positive_rate": baseline,
    }


def geometric_reference_scores(x: np.ndarray) -> np.ndarray:
    """Reproduce the currently shipped geometric family score for a candidate row."""
    result = []
    for row in x:
        shape_signal = float(np.clip((row[0] - 0.72) / 0.18, -1.0, 1.0))
        penalty = float(np.clip(row[4] * 35.0, 0.0, 5.0))
        if row[9] > 0.5:
            score = 78.0 + (
                -shape_signal * 6.0 if shape_signal < 0.0 else -shape_signal * 2.0
            ) - penalty * 0.6
        elif row[10] > 0.5:
            score = 76.0 + (
                shape_signal * 7.0 if shape_signal > 0.0 else shape_signal * 2.0
            ) - penalty * 0.4
        else:
            score = 82.0 - abs(shape_signal) * 3.5 - penalty
        result.append(score)
    return np.asarray(result, dtype=np.float64)


def clustered_auc_comparison_ci(
    y: np.ndarray,
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    groups: np.ndarray,
    seed: int = 2026,
    draws: int = 1000,
) -> dict[str, float]:
    """Customer-cluster bootstrap for AUC and paired AUC gain over the rule baseline."""
    unique_groups = np.unique(groups)
    indices_by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    difference_values: list[float] = []
    for _ in range(draws):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = np.concatenate([indices_by_group[group] for group in sampled_groups])
        sampled_y = y[sampled_indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        candidate_auc = float(roc_auc_score(sampled_y, candidate_scores[sampled_indices]))
        baseline_auc = float(roc_auc_score(sampled_y, baseline_scores[sampled_indices]))
        auc_values.append(candidate_auc)
        difference_values.append(candidate_auc - baseline_auc)
    if len(auc_values) < max(100, draws // 2):
        raise ValueError("Too few valid customer-cluster bootstrap draws. Collect more varied customer ratings.")
    return {
        "auc_ci95_low": float(np.quantile(auc_values, 0.025)),
        "auc_ci95_high": float(np.quantile(auc_values, 0.975)),
        "auc_gain_over_geometry_ci95_low": float(np.quantile(difference_values, 0.025)),
        "auc_gain_over_geometry_ci95_high": float(np.quantile(difference_values, 0.975)),
        "valid_bootstrap_draws": len(auc_values),
    }


def fit_base_model(x: np.ndarray, y: np.ndarray, seed: int):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    # Do not use class_weight='balanced' here: it changes the class prior, making
    # predict_proba unsuitable as a direct estimate of the observed positive-label rate.
    model = LogisticRegression(max_iter=3000, random_state=seed)
    model.fit(scaled, y)
    return scaler, model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Development CSV exported by the app")
    parser.add_argument("--calibration-csv", type=Path, help="Consent-based calibration customers, disjoint from development")
    parser.add_argument("--external-validation-csv", type=Path, help="Untouched external validation customers")
    parser.add_argument("--output", type=Path, default=Path("app/src/main/assets/brow_ranker.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--mark-reviewed-after-human-review",
        action="store_true",
        help="Mark releasable only after you have reviewed reports and met all explicit validation gates",
    )
    args = parser.parse_args()

    if bool(args.calibration_csv) != bool(args.external_validation_csv):
        raise SystemExit("Provide both --calibration-csv and --external-validation-csv, or neither.")
    if args.mark_reviewed_after_human_review and not args.calibration_csv:
        raise SystemExit("Human release approval requires both separate calibration and external validation CSVs.")

    dev_rows = read_rows(args.csv)
    x, y, groups = build_matrix(dev_rows, "development CSV")
    dev_cases = set(groups.tolist())
    if len(dev_cases) < 30:
        raise SystemExit(f"Only {len(dev_cases)} development cases. Collect at least 30 distinct customers for an exploratory candidate.")
    train_idx, test_idx = choose_group_split(x, y, groups, args.seed)
    holdout_scaler, holdout_model = fit_base_model(x[train_idx], y[train_idx], args.seed)
    holdout_metrics = calibration_metrics(
        y[test_idx],
        holdout_model.predict_proba(holdout_scaler.transform(x[test_idx]))[:, 1],
    )
    holdout_metrics.update({
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_cases": int(len(np.unique(groups[train_idx]))),
        "test_cases": int(len(np.unique(groups[test_idx]))),
        "split": "GroupShuffleSplit by case_id, 80/20",
    })

    scaler, model = fit_base_model(x, y, args.seed)
    calibrator_intercept = 0.0
    calibrator_slope = 1.0
    calibration_report: dict[str, Any] = {
        "status": "not_fitted",
        "n_cases": 0,
        "note": "Base logistic score is not approved as a calibrated customer preference probability.",
    }
    external_report: dict[str, Any] = {}
    release_approved = False

    if args.calibration_csv and args.external_validation_csv:
        cal_rows = read_rows(args.calibration_csv)
        cal_x, cal_y, cal_groups = build_matrix(cal_rows, "calibration CSV")
        ext_rows = read_rows(args.external_validation_csv)
        ext_x, ext_y, ext_groups = build_matrix(ext_rows, "external validation CSV")
        cal_cases = set(cal_groups.tolist())
        ext_cases = set(ext_groups.tolist())
        if dev_cases & cal_cases or dev_cases & ext_cases or cal_cases & ext_cases:
            raise SystemExit("Development, calibration, and external validation case_id sets must be disjoint.")
        if len(cal_cases) < 50:
            raise SystemExit(f"Only {len(cal_cases)} calibration cases. Need at least 50 distinct cases for the release gate.")
        if len(ext_cases) < 100:
            raise SystemExit(f"Only {len(ext_cases)} external validation cases. Need at least 100 distinct cases for the release gate.")

        cal_scaled = scaler.transform(cal_x)
        cal_raw_logits = model.decision_function(cal_scaled).reshape(-1, 1)
        # Fit Platt scaling on customers never used for base-model training.
        calibrator = LogisticRegression(C=1000.0, max_iter=2000, random_state=args.seed)
        calibrator.fit(cal_raw_logits, cal_y)
        calibrator_intercept = float(calibrator.intercept_[0])
        calibrator_slope = float(calibrator.coef_[0, 0])
        calibration_report = {
            "status": "platt_scaling",
            "n_cases": len(cal_cases),
            "n_rows": int(len(cal_y)),
            "positive_rate": float(np.mean(cal_y)),
        }

        ext_logits = model.decision_function(scaler.transform(ext_x))
        ext_probs = sigmoid(calibrator_intercept + calibrator_slope * ext_logits)
        external_report = calibration_metrics(ext_y, ext_probs)
        reference_scores = geometric_reference_scores(ext_x)
        reference_auc = float(roc_auc_score(ext_y, reference_scores))
        comparison = clustered_auc_comparison_ci(
            ext_y, ext_probs, reference_scores, ext_groups, seed=args.seed
        )
        external_report.update({
            "n_cases": len(ext_cases),
            "n_rows": int(len(ext_y)),
            "split": "independent customer-level CSV, disjoint from development and calibration",
            "geometry_rule_baseline_roc_auc": reference_auc,
            "customer_cluster_bootstrap": comparison,
        })
        gates = {
            "external_cases_at_least_100": len(ext_cases) >= 100,
            "roc_auc_at_least_0_60": external_report["roc_auc"] >= 0.60,
            "auc_ci_lower_above_chance": comparison["auc_ci95_low"] > 0.50,
            "auc_gain_over_rule_lower_ci_above_zero": comparison["auc_gain_over_geometry_ci95_low"] > 0.0,
            "brier_better_than_prevalence_baseline": (
                external_report["brier_score"] < external_report["prevalence_baseline_brier_score"]
            ),
            "ece_at_most_0_10": external_report["expected_calibration_error_10_bins"] <= 0.10,
        }
        external_report["release_gates"] = gates
        external_report["passed_release_gates"] = all(gates.values())
        if args.mark_reviewed_after_human_review and not all(gates.values()):
            raise SystemExit("Release approval refused: one or more external validation gates failed. See external_validation in the candidate JSON/console report.")
        release_approved = bool(args.mark_reviewed_after_human_review and all(gates.values()))

    payload = {
        "model_version": 3,
        "feature_order": FEATURE_NAMES,
        "means": [float(v) for v in scaler.mean_],
        "scales": [float(v) if float(v) != 0.0 else 1.0 for v in scaler.scale_],
        "coefficients": [float(v) for v in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "calibration_intercept": calibrator_intercept,
        "calibration_slope": calibrator_slope,
        "calibration_status": calibration_report["status"],
        "release_approved": release_approved,
        "holdout_metrics": holdout_metrics,
        "calibration": calibration_report,
        "external_validation": external_report,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_cases": len(dev_cases),
        "release_note": (
            "Human-reviewed release candidate. Scores estimate the probability that a candidate receives rating 4 or 5 in the observed feedback process; "
            "they do not prove objective beauty, suitability, or treatment outcome."
            if release_approved else
            "Research candidate only. The Android app must not load this model until separate calibration, external validation, and human review pass."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "development_holdout": holdout_metrics,
        "calibration": calibration_report,
        "external_validation": external_report,
        "release_approved": release_approved,
    }, indent=2))
    print(f"Candidate JSON written to: {args.output}")
    if not release_approved:
        print("NOT RELEASE APPROVED. No real customer preference data is created by this script; collect consented, separate case groups and review the external test first.")


if __name__ == "__main__":
    main()
