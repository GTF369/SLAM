#!/usr/bin/env python3
"""Train a customer-choice model over three eyebrow candidates in a single session.

Each valid session must contain exactly three candidates (natural, soft, defined) for
one anonymous customer and exactly one final selection. Probabilities are conditional on
the candidate set shown in that session. The model is intentionally not releasable until
disjoint calibration and external customer sessions pass the gates and a human reviewer
explicitly approves the result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_brow_ranker import FEATURE_NAMES, geometric_reference_scores

FEATURE_COLUMNS = {
    "face_width_height_ratio",
    "interocular_face_width_ratio",
    "jaw_face_width_ratio",
    "brow_eye_distance_ratio",
    "brow_asymmetry_ratio",
    "eye_line_tilt",
    "native_brow_area_fraction",
    "native_brow_side_imbalance",
    "native_brow_mask_available",
    "native_shape_mode",
}
REQUIRED = {
    "case_id", "session_id", "style", "thickness", "length", "arch", "selected",
    *FEATURE_COLUMNS,
}
EXPECTED_STYLES = {"natural", "soft", "defined"}
DESIGN_INTENT_INDEX = {
    "均衡自然": 0,
    "偏柔和": 1,
    "偏立体": 2,
    "balanced_natural": 0,
    "soft": 1,
    "defined": 2,
}
REGULARIZATION = 0.02
CHOICE_INTERACTION_NAMES = [
    "face_width_height_x_thickness",
    "face_width_height_x_length",
    "face_width_height_x_arch",
    "interocular_ratio_x_thickness",
    "interocular_ratio_x_arch",
    "jaw_ratio_x_arch",
    "brow_eye_distance_x_thickness",
    "brow_eye_distance_x_arch",
    "brow_asymmetry_x_arch",
    "face_width_height_x_style_soft",
    "face_width_height_x_style_defined",
    "brow_eye_distance_x_style_soft",
    "brow_eye_distance_x_style_defined",
    "brow_asymmetry_x_style_soft",
    "brow_asymmetry_x_style_defined",
    "eye_line_tilt_x_style_soft",
    "eye_line_tilt_x_style_defined",
    "native_brow_area_fraction_x_thickness",
    "native_brow_side_imbalance_x_thickness",
    "native_brow_side_imbalance_x_arch",
    "design_intent_soft_x_style_soft",
    "design_intent_soft_x_style_defined",
    "design_intent_defined_x_style_soft",
    "design_intent_defined_x_style_defined",
]
CHOICE_FEATURE_NAMES = FEATURE_NAMES + CHOICE_INTERACTION_NAMES


def expand_choice_features(base: np.ndarray, design_intent: int = 0) -> np.ndarray:
    """Expand face/candidate features and interactions with the explicitly stated style intent."""
    values = np.asarray(base, dtype=np.float64)
    if values.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} base features, got {values.shape}")
    if design_intent not in (0, 1, 2):
        raise ValueError("design_intent must be 0 (balanced), 1 (soft), or 2 (defined)")
    face_width, interocular, jaw, brow_eye, asymmetry, eye_tilt = values[:6]
    thickness, length, arch = values[6:9]
    style_soft, style_defined = values[9:11]
    native_area, side_imbalance = values[11:13]
    interactions = np.asarray([
        face_width * thickness,
        face_width * length,
        face_width * arch,
        interocular * thickness,
        interocular * arch,
        jaw * arch,
        brow_eye * thickness,
        brow_eye * arch,
        asymmetry * arch,
        face_width * style_soft,
        face_width * style_defined,
        brow_eye * style_soft,
        brow_eye * style_defined,
        asymmetry * style_soft,
        asymmetry * style_defined,
        eye_tilt * style_soft,
        eye_tilt * style_defined,
        native_area * thickness,
        side_imbalance * thickness,
        side_imbalance * arch,
    ], dtype=np.float64)
    intent_soft = 1.0 if design_intent == 1 else 0.0
    intent_defined = 1.0 if design_intent == 2 else 0.0
    intent_interactions = np.asarray([
        intent_soft * style_soft,
        intent_soft * style_defined,
        intent_defined * style_soft,
        intent_defined * style_defined,
    ], dtype=np.float64)
    return np.concatenate((values, interactions, intent_interactions))


def parse_selected(raw: str, *, source: str, row_num: int) -> bool:
    value = raw.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"{source} row {row_num} has invalid selected value {raw!r}; use true/false or 1/0.")


def feature_row(row: dict[str, str], *, source: str, row_num: int) -> np.ndarray:
    style = row["style"].strip().lower()
    if style not in EXPECTED_STYLES:
        raise ValueError(f"{source} row {row_num} has unknown style {style!r}.")
    try:
        values = [
            float(row["face_width_height_ratio"]),
            float(row["interocular_face_width_ratio"]),
            float(row["jaw_face_width_ratio"]),
            float(row["brow_eye_distance_ratio"]),
            float(row["brow_asymmetry_ratio"]),
            float(row["eye_line_tilt"]),
            float(row["thickness"]),
            float(row["length"]),
            float(row["arch"]),
            1.0 if style == "soft" else 0.0,
            1.0 if style == "defined" else 0.0,
            float(row["native_brow_area_fraction"]),
            float(row["native_brow_side_imbalance"]),
            float(row["native_brow_mask_available"]),
            float(row["native_shape_mode"]),
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} row {row_num} has an invalid numeric feature.") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{source} row {row_num} contains NaN or infinity.")
    if not (0.0 <= values[11] <= 1.0 and 0.0 <= values[12] <= 1.0):
        raise ValueError(f"{source} row {row_num} has out-of-range mask fractions.")
    if values[13] not in (0.0, 1.0) or values[14] not in (0.0, 1.0):
        raise ValueError(f"{source} row {row_num} must use 0/1 for mask availability and native-shape mode.")
    raw_intent = (row.get("design_intent") or "均衡自然").strip()
    if raw_intent not in DESIGN_INTENT_INDEX:
        raise ValueError(
            f"{source} row {row_num} has unknown design_intent {raw_intent!r}; "
            "use 均衡自然, 偏柔和, or 偏立体."
        )
    return expand_choice_features(
        np.asarray(values, dtype=np.float64),
        design_intent=DESIGN_INTENT_INDEX[raw_intent],
    )


def build_choice_sessions(rows: list[dict[str, str]], source: str = "CSV") -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"{source} is empty.")
    missing = REQUIRED.difference(rows[0].keys())
    if missing:
        raise ValueError(f"{source} is missing columns: " + ", ".join(sorted(missing)))

    grouped: dict[str, list[tuple[dict[str, str], int]]] = {}
    for row_num, row in enumerate(rows, start=2):
        case_id = row.get("case_id", "").strip()
        session_id = row.get("session_id", "").strip()
        if not case_id or not session_id:
            raise ValueError(
                f"{source} row {row_num} needs stable case_id and session_id. "
                "The current APK export does not contain session_id. Prepare a session CSV with exactly three style candidates " +
                "(natural, soft, defined) per session and exactly one selected candidate before using this experimental trainer."
            )
        grouped.setdefault(session_id, []).append((row, row_num))

    sessions: list[dict[str, Any]] = []
    bad: list[str] = []
    for session_id, records in grouped.items():
        case_ids = {row["case_id"].strip() for row, _ in records}
        styles = [row["style"].strip().lower() for row, _ in records]
        selected_indices = [
            index for index, (row, row_num) in enumerate(records)
            if parse_selected(row.get("selected", ""), source=source, row_num=row_num)
        ]
        if len(records) != 3 or len(case_ids) != 1 or set(styles) != EXPECTED_STYLES or len(set(styles)) != 3:
            bad.append(session_id)
            continue
        if len(selected_indices) != 1:
            bad.append(session_id)
            continue
        # Sort candidates by a fixed style order so the session representation is reproducible.
        style_order = {"natural": 0, "soft": 1, "defined": 2}
        sorted_records = sorted(records, key=lambda item: style_order[item[0]["style"].strip().lower()])
        matrix = np.stack([
            feature_row(row, source=source, row_num=row_num)
            for row, row_num in sorted_records
        ])
        selected_row = records[selected_indices[0]][0]
        selected_style = selected_row["style"].strip().lower()
        selected_index = style_order[selected_style]
        sessions.append({
            "session_id": session_id,
            "case_id": next(iter(case_ids)),
            "x": matrix,
            "choice": selected_index,
        })
    if bad:
        preview = ", ".join(bad[:5])
        raise ValueError(
            f"{source} has {len(bad)} invalid choice sessions (examples: {preview}). "
            "Each session must have exactly one natural, one soft and one defined candidate, "
            "all from the same customer, with exactly one final selection."
        )
    if not sessions:
        raise ValueError(f"{source} contains no complete three-candidate choice sessions.")
    return sessions


def read_sessions(path: Path, source: str) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return build_choice_sessions(rows, source)


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def fit_conditional_model(sessions: list[dict[str, Any]], seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del seed  # deterministic optimizer; retained for CLI compatibility and reproducibility
    raw = np.concatenate([session["x"] for session in sessions], axis=0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw)
    cursor = 0
    prepared: list[tuple[np.ndarray, int]] = []
    for session in sessions:
        count = len(session["x"])
        prepared.append((scaled[cursor:cursor + count], int(session["choice"])))
        cursor += count
    feature_count = scaled.shape[1]

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loss = 0.0
        gradient = np.zeros_like(beta)
        for matrix, selected_index in prepared:
            scores = matrix @ beta
            probabilities = np.exp(scores - np.max(scores))
            probabilities /= probabilities.sum()
            loss += _logsumexp(scores) - float(scores[selected_index])
            gradient += matrix.T @ probabilities - matrix[selected_index]
        divisor = float(len(prepared))
        loss = loss / divisor + 0.5 * REGULARIZATION * float(beta @ beta)
        gradient = gradient / divisor + REGULARIZATION * beta
        return loss, gradient

    result = minimize(
        fun=lambda beta: objective(beta)[0],
        x0=np.zeros(feature_count, dtype=np.float64),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Conditional-choice optimization failed: {result.message}")
    return scaler.mean_.copy(), scaler.scale_.copy(), result.x.copy()


def session_logits(session: dict[str, Any], means: np.ndarray, scales: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    scaled = (session["x"] - means) / np.where(scales == 0.0, 1.0, scales)
    return scaled @ coefficients


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite.")
    scaled = np.asarray(scores, dtype=np.float64) / temperature
    shifted = scaled - np.max(scaled)
    exp_scores = np.exp(np.clip(shifted, -700, 0))
    return exp_scores / exp_scores.sum()


def fit_temperature(sessions: list[dict[str, Any]], means: np.ndarray, scales: np.ndarray, coefficients: np.ndarray) -> float:
    logits = [session_logits(s, means, scales, coefficients) for s in sessions]
    targets = [int(s["choice"]) for s in sessions]

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        total = 0.0
        for scores, target in zip(logits, targets):
            probabilities = softmax(scores, temperature)
            total -= math.log(max(float(probabilities[target]), 1e-12))
        return total / len(logits)

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 1000},
    )
    if not result.success or not math.isfinite(result.x):
        raise ValueError("Could not fit choice-model temperature on calibration sessions.")
    return math.exp(float(result.x))


def _cluster_bootstrap_gain(
    sessions: list[dict[str, Any]],
    predictions: list[np.ndarray],
    seed: int = 2026,
    draws: int = 1000,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for session, probabilities in zip(sessions, predictions):
        candidate_correct = int(int(np.argmax(probabilities)) == int(session["choice"]))
        baseline = geometric_reference_scores(session["x"])
        baseline_correct = int(int(np.argmax(baseline)) == int(session["choice"]))
        grouped.setdefault(session["case_id"], []).append(float(candidate_correct - baseline_correct))
    customer_mean_gain = {customer: float(np.mean(gains)) for customer, gains in grouped.items()}
    customers = list(customer_mean_gain)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(customers, size=len(customers), replace=True)
        # Weight each customer equally, even when they have multiple design sessions.
        values.append(float(np.mean([customer_mean_gain[customer] for customer in sampled])))
    return {
        "top1_gain_ci95_low": float(np.quantile(values, 0.025)),
        "top1_gain_ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_draws": int(draws),
    }


def choice_metrics(
    sessions: list[dict[str, Any]],
    means: np.ndarray,
    scales: np.ndarray,
    coefficients: np.ndarray,
    temperature: float,
    *,
    seed: int = 2026,
    include_bootstrap: bool = False,
) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    top1: list[float] = []
    nll: list[float] = []
    brier: list[float] = []
    ece_rows: list[tuple[float, float]] = []
    baseline_correct: list[float] = []
    for session in sessions:
        scores = session_logits(session, means, scales, coefficients)
        probabilities = softmax(scores, temperature)
        choice = int(session["choice"])
        correct = float(int(np.argmax(probabilities)) == choice)
        predictions.append(probabilities)
        top1.append(correct)
        nll.append(-math.log(max(float(probabilities[choice]), 1e-12)))
        target = np.zeros_like(probabilities)
        target[choice] = 1.0
        brier.append(float(np.sum((probabilities - target) ** 2)))
        ece_rows.append((float(np.max(probabilities)), correct))
        baseline = geometric_reference_scores(session["x"])
        baseline_correct.append(float(int(np.argmax(baseline)) == choice))
    ece = 0.0
    for bin_index in range(10):
        lo, hi = bin_index / 10.0, (bin_index + 1) / 10.0
        selected = [
            correct for confidence, correct in ece_rows
            if confidence >= lo and (confidence < hi if bin_index < 9 else confidence <= hi)
        ]
        if selected:
            ece += len(selected) / len(ece_rows) * abs(float(np.mean(selected)) - float(np.mean([confidence for confidence, _ in ece_rows if confidence >= lo and (confidence < hi if bin_index < 9 else confidence <= hi)])))
    result: dict[str, Any] = {
        "n_sessions": len(sessions),
        "n_cases": len({s["case_id"] for s in sessions}),
        "top1_accuracy": float(np.mean(top1)),
        "geometry_rule_top1_accuracy": float(np.mean(baseline_correct)),
        "top1_gain_over_geometry": float(np.mean(top1) - np.mean(baseline_correct)),
        "mean_negative_log_likelihood": float(np.mean(nll)),
        "multiclass_brier_score": float(np.mean(brier)),
        "uniform_choice_brier_baseline": 2.0 / 3.0,
        "expected_calibration_error_10_bins": float(ece),
        "mean_max_probability": float(np.mean([float(p.max()) for p in predictions])),
    }
    if include_bootstrap:
        result["customer_cluster_bootstrap"] = _cluster_bootstrap_gain(sessions, predictions, seed=seed)
    return result


def release_gates(metrics: dict[str, Any]) -> dict[str, bool]:
    ci = metrics.get("customer_cluster_bootstrap", {})
    return {
        "external_cases_at_least_100": int(metrics["n_cases"]) >= 100,
        "external_sessions_at_least_100": int(metrics["n_sessions"]) >= 100,
        "top1_accuracy_at_least_0_40": float(metrics["top1_accuracy"]) >= 0.40,
        "top1_gain_ci_lower_above_zero": float(ci.get("top1_gain_ci95_low", -1.0)) > 0.0,
        "multiclass_brier_better_than_uniform": float(metrics["multiclass_brier_score"]) < 2.0 / 3.0,
        "ece_at_most_0_10": float(metrics["expected_calibration_error_10_bins"]) <= 0.10,
        "nll_better_than_uniform": float(metrics["mean_negative_log_likelihood"]) < math.log(3.0),
    }


def train_choice_model(
    development: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    external: list[dict[str, Any]],
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    dev_cases = {s["case_id"] for s in development}
    cal_cases = {s["case_id"] for s in calibration}
    ext_cases = {s["case_id"] for s in external}
    if dev_cases & cal_cases or dev_cases & ext_cases or cal_cases & ext_cases:
        raise ValueError("Development, calibration, and external customers must be disjoint.")
    if len(dev_cases) < 30 or len(cal_cases) < 50 or len(ext_cases) < 100:
        raise ValueError("Need at least 30 development, 50 calibration, and 100 external customers.")
    means, scales, coefficients = fit_conditional_model(development, seed)
    temperature = fit_temperature(calibration, means, scales, coefficients)
    calibration_metrics_report = choice_metrics(
        calibration, means, scales, coefficients, temperature, seed=seed
    )
    external_report = choice_metrics(
        external, means, scales, coefficients, temperature, seed=seed, include_bootstrap=True
    )
    gates = release_gates(external_report)
    external_report["release_gates"] = gates
    external_report["passed_release_gates"] = all(gates.values())
    return {
        "model_version": 2,
        "model_type": "conditional_choice",
        "feature_order": CHOICE_FEATURE_NAMES,
        "means": means.tolist(),
        "scales": [float(v) if float(v) != 0.0 else 1.0 for v in scales],
        "coefficients": coefficients.tolist(),
        "temperature": float(temperature),
        "calibration_status": "temperature_scaling",
        "release_approved": False,
        "development_cases": len(dev_cases),
        "development_sessions": len(development),
        "calibration": calibration_metrics_report,
        "calibration_cases": len(cal_cases),
        "external_validation": external_report,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_note": (
            "Research candidate only. Probabilities are conditional on the three candidate styles "
            "shown in a design session and require explicit human release approval."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_csv", type=Path)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--external-validation-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/output/brow_choice_ranker_candidate.json"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mark-reviewed-after-human-review", action="store_true")
    args = parser.parse_args()

    development = read_sessions(args.development_csv, "development CSV")
    calibration = read_sessions(args.calibration_csv, "calibration CSV")
    external = read_sessions(args.external_validation_csv, "external validation CSV")
    try:
        payload = train_choice_model(development, calibration, external, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    payload["release_approved"] = bool(
        args.mark_reviewed_after_human_review and payload["external_validation"]["passed_release_gates"]
    )
    if args.mark_reviewed_after_human_review and not payload["external_validation"]["passed_release_gates"]:
        raise SystemExit("Choice model failed an external release gate. Candidate not approved.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "calibration": payload["calibration"],
        "external_validation": payload["external_validation"],
        "release_approved": payload["release_approved"],
    }, indent=2))
    if not payload["release_approved"]:
        print("NOT RELEASE APPROVED. Never treat synthetic or unreviewed choice model probabilities as real customer probabilities.")


if __name__ == "__main__":
    main()
