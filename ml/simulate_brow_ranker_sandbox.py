#!/usr/bin/env python3
"""Exercise the preference ranking and probability-calibration pipeline on synthetic data.

This is a sandbox validation fixture only. It must never be treated as real salon data
or copied into app/src/main/assets/brow_ranker.json for release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_brow_ranker import build_matrix, calibration_metrics, fit_base_model, sigmoid

FEATURE_MEANS = np.asarray([
    0.78, 0.38, 0.80, 0.055, 0.025, 0.0, 1.0, 1.0, 1.0,
    0.0, 0.0, 0.12, 0.04, 0.85, 0.5,
])
FEATURE_SCALES = np.asarray([
    0.08, 0.05, 0.12, 0.02, 0.02, 0.08, 0.25, 0.15, 0.25,
    1.0, 1.0, 0.07, 0.03, 0.35, 0.5,
])
SYNTHETIC_WEIGHTS = np.asarray([
    0.55, -0.35, 0.25, -0.20, -0.90, 0.15, 0.75, -0.40, 0.70,
    -0.55, 0.55, 0.20, -0.30, 0.25, 0.15,
])


def make_synthetic_rows(case_count: int, prefix: str, seed: int) -> tuple[list[dict[str, str]], list[float]]:
    """Make synthetic customer groups from a known logistic preference function."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    probabilities: list[float] = []
    styles = [
        ("natural", 0, 1.00, 1.00, 0.92),
        ("soft", 1, 0.82, 0.94, 0.72),
        ("defined", 2, 1.18, 1.06, 1.22),
    ]
    for case_index in range(case_count):
        case_id = f"SYNTH-{prefix}-{case_index:04d}"
        face_width_height = float(np.clip(rng.normal(0.78, 0.075), 0.60, 0.98))
        interocular = float(np.clip(rng.normal(0.38, 0.045), 0.25, 0.52))
        jaw_ratio = float(np.clip(rng.normal(0.80, 0.10), 0.55, 1.02))
        brow_eye = float(np.clip(rng.normal(0.055, 0.018), 0.015, 0.12))
        asymmetry = float(np.clip(rng.gamma(1.6, 0.012), 0.0, 0.09))
        eye_tilt = float(np.clip(rng.normal(0.0, 0.06), -0.18, 0.18))
        native_area = float(np.clip(rng.normal(0.12, 0.045), 0.01, 0.28))
        side_imbalance = float(np.clip(rng.normal(0.04, 0.025), 0.0, 0.18))
        mask_available = float(rng.random() < 0.82)
        native_shape_mode = float(rng.random() < 0.45)
        customer_noise = float(rng.normal(0.0, 0.25))

        for style, _, thickness, length, arch in styles:
            # Candidate parameter variation is synthetic and deliberately simple.
            thickness_i = float(np.clip(thickness + rng.normal(0.0, 0.035), 0.55, 1.45))
            length_i = float(np.clip(length + rng.normal(0.0, 0.025), 0.80, 1.20))
            arch_i = float(np.clip(arch + rng.normal(0.0, 0.045), 0.55, 1.45))
            features = np.asarray([
                face_width_height, interocular, jaw_ratio, brow_eye, asymmetry, eye_tilt,
                thickness_i, length_i, arch_i,
                float(style == "soft"), float(style == "defined"),
                native_area, side_imbalance, mask_available, native_shape_mode,
            ])
            z = -0.22 + float(np.dot((features - FEATURE_MEANS) / FEATURE_SCALES, SYNTHETIC_WEIGHTS))
            z += customer_noise
            probability = float(1.0 / (1.0 + np.exp(-np.clip(z, -20, 20))))
            positive = bool(rng.random() < probability)
            probabilities.append(probability)
            rows.append({
                "case_id": case_id,
                "style": style,
                "thickness": f"{thickness_i:.8f}",
                "length": f"{length_i:.8f}",
                "arch": f"{arch_i:.8f}",
                "rating": "5" if positive else "2",
                "face_width_height_ratio": f"{face_width_height:.8f}",
                "interocular_face_width_ratio": f"{interocular:.8f}",
                "jaw_face_width_ratio": f"{jaw_ratio:.8f}",
                "brow_eye_distance_ratio": f"{brow_eye:.8f}",
                "brow_asymmetry_ratio": f"{asymmetry:.8f}",
                "eye_line_tilt": f"{eye_tilt:.8f}",
                "native_brow_area_fraction": f"{native_area:.8f}",
                "native_brow_side_imbalance": f"{side_imbalance:.8f}",
                "native_brow_mask_available": f"{mask_available:.0f}",
                "native_shape_mode": f"{native_shape_mode:.0f}",
            })
    return rows, probabilities


def run_sandbox(seed: int = 2026) -> dict:
    dev_rows, _ = make_synthetic_rows(300, "DEV", seed)
    cal_rows, _ = make_synthetic_rows(90, "CAL", seed + 1)
    ext_rows, ext_truth = make_synthetic_rows(160, "EXT", seed + 2)
    x_dev, y_dev, dev_groups = build_matrix(dev_rows, "synthetic development")
    x_cal, y_cal, cal_groups = build_matrix(cal_rows, "synthetic calibration")
    x_ext, y_ext, ext_groups = build_matrix(ext_rows, "synthetic external")
    dev_cases, cal_cases, ext_cases = map(set, (dev_groups.tolist(), cal_groups.tolist(), ext_groups.tolist()))
    if dev_cases & cal_cases or dev_cases & ext_cases or cal_cases & ext_cases:
        raise AssertionError("Synthetic customer groups leaked across data partitions")

    scaler, model = fit_base_model(x_dev, y_dev, seed)
    calibration_logits = model.decision_function(scaler.transform(x_cal)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000.0, max_iter=2000, random_state=seed)
    calibrator.fit(calibration_logits, y_cal)
    ext_logits = model.decision_function(scaler.transform(x_ext))
    probabilities = sigmoid(float(calibrator.intercept_[0]) + float(calibrator.coef_[0, 0]) * ext_logits)
    metrics = calibration_metrics(y_ext, probabilities)
    true_probability = np.asarray(ext_truth, dtype=np.float64)
    probability_mae_vs_known_generator = float(np.mean(np.abs(probabilities - true_probability)))

    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise AssertionError("Sandbox calibrated outputs must be finite probabilities in [0,1]")
    if metrics["brier_score"] >= metrics["prevalence_baseline_brier_score"]:
        raise AssertionError("Sandbox model failed to beat the synthetic prevalence-only baseline")
    if metrics["roc_auc"] < 0.60:
        raise AssertionError("Sandbox model failed the synthetic ranking-discrimination screen")

    return {
        "test_type": "synthetic_ranking_and_calibration_pipeline_not_real_customer_evidence",
        "seed": seed,
        "cases": {
            "development": len(dev_cases),
            "calibration": len(cal_cases),
            "external": len(ext_cases),
        },
        "rows": {"development": len(y_dev), "calibration": len(y_cal), "external": len(y_ext)},
        "external_metrics": metrics,
        "mean_absolute_probability_error_vs_known_synthetic_generator": probability_mae_vs_known_generator,
        "output_probability_range": [float(probabilities.min()), float(probabilities.max())],
        "release_approved": False,
        "note": (
            "The generator created the labels and ground-truth probabilities. This verifies "
            "software/calibration behavior only. It is not evidence of real salon preference, "
            "and the model must not be used to report real-customer probabilities."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("brow_ranker_sandbox_report.json"))
    args = parser.parse_args()
    report = run_sandbox(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Synthetic-only report written to: {args.output}")


if __name__ == "__main__":
    main()
