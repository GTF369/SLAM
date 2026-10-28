#!/usr/bin/env python3
"""Synthetic-only test of a calibrated three-candidate conditional choice model.

This script creates synthetic candidate sets with a known utility function. Its report
must never be used as real salon evidence or copied into app assets for release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_brow_choice_ranker import (
    build_choice_sessions,
    expand_choice_features,
    softmax,
    train_choice_model,
)

# Utility depends on the face-by-candidate interactions as well as candidate parameters.
# Those interactions are crucial because face-only terms cancel within conditional softmax.
TRUE_WEIGHTS = np.asarray([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.35, 0.25, 0.20, 0.10, 0.08, 0.0, 0.0, 0.0, 0.0,
    2.8, 1.8, 0.8, -0.9, 0.7, 1.1, 3.0, 4.0, -3.4,
    1.6, 1.2, 2.5, 1.8, -4.0, -2.8, 0.8, 1.0, 0.6, -1.3, -1.8,
], dtype=np.float64)
GENERATOR_TEMPERATURE = 0.65


def make_synthetic_rows(case_count: int, prefix: str, seed: int) -> tuple[list[dict[str, str]], dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []
    truths: dict[str, list[float]] = {}
    base_templates = [
        ("natural", 0, 1.00, 1.00, 0.92),
        ("soft", 1, 0.82, 0.94, 0.72),
        ("defined", 2, 1.18, 1.06, 1.22),
    ]
    for case_index in range(case_count):
        case_id = f"SYNTH-{prefix}-{case_index:04d}"
        session_id = f"SESSION-{prefix}-{case_index:04d}"
        face_width = float(np.clip(rng.normal(0.78, 0.075), 0.60, 0.98))
        interocular = float(np.clip(rng.normal(0.38, 0.045), 0.25, 0.52))
        jaw_ratio = float(np.clip(rng.normal(0.80, 0.10), 0.55, 1.02))
        brow_eye = float(np.clip(rng.normal(0.055, 0.018), 0.015, 0.12))
        asymmetry = float(np.clip(rng.gamma(1.6, 0.012), 0.0, 0.09))
        eye_tilt = float(np.clip(rng.normal(0.0, 0.06), -0.18, 0.18))
        native_area = float(np.clip(rng.normal(0.12, 0.045), 0.01, 0.28))
        side_imbalance = float(np.clip(rng.normal(0.04, 0.025), 0.0, 0.18))
        mask_available = float(rng.random() < 0.82)
        native_shape = float(rng.random() < 0.45)

        session_rows: list[dict[str, str]] = []
        utilities: list[float] = []
        for style, style_idx, thickness_base, length_base, arch_base in base_templates:
            thickness = float(np.clip(thickness_base + rng.normal(0.0, 0.025), 0.55, 1.45))
            length = float(np.clip(length_base + rng.normal(0.0, 0.02), 0.80, 1.20))
            arch = float(np.clip(arch_base + rng.normal(0.0, 0.035), 0.55, 1.45))
            base = np.asarray([
                face_width, interocular, jaw_ratio, brow_eye, asymmetry, eye_tilt,
                thickness, length, arch,
                float(style_idx == 1), float(style_idx == 2),
                native_area, side_imbalance, mask_available, native_shape,
            ], dtype=np.float64)
            utility = float(expand_choice_features(base) @ TRUE_WEIGHTS)
            # Session-level style preference noise is shared across candidates and therefore
            # cancels from the conditional choice. Candidate-specific noise remains observable.
            utility += float(rng.normal(0.0, 0.12))
            utilities.append(utility)
            session_rows.append({
                "case_id": case_id,
                "session_id": session_id,
                "style": style,
                "thickness": f"{thickness:.8f}",
                "length": f"{length:.8f}",
                "arch": f"{arch:.8f}",
                "rating": "5",
                "selected": "false",
                "face_width_height_ratio": f"{face_width:.8f}",
                "interocular_face_width_ratio": f"{interocular:.8f}",
                "jaw_face_width_ratio": f"{jaw_ratio:.8f}",
                "brow_eye_distance_ratio": f"{brow_eye:.8f}",
                "brow_asymmetry_ratio": f"{asymmetry:.8f}",
                "eye_line_tilt": f"{eye_tilt:.8f}",
                "native_brow_area_fraction": f"{native_area:.8f}",
                "native_brow_side_imbalance": f"{side_imbalance:.8f}",
                "native_brow_mask_available": f"{mask_available:.0f}",
                "native_shape_mode": f"{native_shape:.0f}",
            })
        true_probabilities = softmax(np.asarray(utilities), GENERATOR_TEMPERATURE)
        selected_index = int(rng.choice(3, p=true_probabilities))
        session_rows[selected_index]["selected"] = "true"
        truths[session_id] = true_probabilities.tolist()
        rows.extend(session_rows)
    return rows, truths


def _sessions(case_count: int, prefix: str, seed: int):
    rows, truths = make_synthetic_rows(case_count, prefix, seed)
    return build_choice_sessions(rows, f"synthetic {prefix}"), truths


def run_sandbox(seed: int = 2026) -> dict:
    development, _ = _sessions(300, "DEV", seed)
    calibration, _ = _sessions(90, "CAL", seed + 1)
    external, true_probs = _sessions(160, "EXT", seed + 2)
    report = train_choice_model(development, calibration, external, seed=seed)
    # The synthetic training/evaluation pipeline cannot grant real-world release approval.
    report["release_approved"] = False
    report["test_type"] = "synthetic_conditional_choice_probability_not_real_customer_evidence"
    report["synthetic_true_probabilities_used_for_generation_only"] = True
    probability_errors = []
    means = np.asarray(report["means"])
    scales = np.asarray(report["scales"])
    coefficients = np.asarray(report["coefficients"])
    for session in external:
        standardized_x = (session["x"] - means) / scales
        predicted_probabilities = softmax(
            standardized_x @ coefficients,
            report["temperature"],
        )
        probability_errors.append(float(np.mean(np.abs(
            predicted_probabilities - np.asarray(true_probs[session["session_id"]])
        ))))
    report["mean_probability_mae_vs_synthetic_generator"] = float(np.mean(probability_errors))
    report["note"] = (
        "Synthetic data validates conditional-softmax fitting, temperature scaling and "
        "candidate-set probability arithmetic only. These numbers are not real salon "
        "preferences and must never be presented as real customer suitability probabilities."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("brow-choice-sandbox-report.json"))
    args = parser.parse_args()
    report = run_sandbox(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "test_type": report["test_type"],
        "n_development_sessions": report["development_sessions"],
        "n_calibration_sessions": report["calibration"]["n_sessions"],
        "external": report["external_validation"],
        "mean_probability_mae_vs_synthetic_generator": report["mean_probability_mae_vs_synthetic_generator"],
        "release_approved": report["release_approved"],
    }, indent=2))
    print(f"Synthetic-only report written to: {args.output}")


if __name__ == "__main__":
    main()
