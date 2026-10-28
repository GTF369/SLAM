#!/usr/bin/env python3
"""Public-image technical stress test for eyebrow ranking and relative-depth sensitivity.

This is deliberately NOT a preference-learning experiment. Public portraits have no per-person
ratings for our candidate edits, so normalized geometry scores are reference weights only.
MediaPipe z coordinates are relative estimates, not millimetre skin-surface measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from train_brow_ranker import geometric_reference_scores


def _load_cv2():
    """Load OpenCV only for image operations so pure geometry tests need no cv2 wheel."""
    try:
        import cv2 as cv2_module
    except ImportError as exc:
        raise RuntimeError("Install opencv-python-headless to run image processing.") from exc
    return cv2_module


CANDIDATES = (
    {"style": "natural", "thickness": 1.00, "length": 1.00, "arch": 0.92, "style_index": 0},
    {"style": "soft", "thickness": 0.82, "length": 0.94, "arch": 0.72, "style_index": 1},
    {"style": "defined", "thickness": 1.18, "length": 1.06, "arch": 1.22, "style_index": 2},
)
BROW_LANDMARK_SETS = ((70, 63, 105, 66, 107), (300, 293, 334, 296, 336))
CURVATURE_REGIONS = ("左眉头", "左眉身", "左眉峰", "左眉尾", "右眉头", "右眉身", "右眉峰", "右眉尾")
REFERENCE_WEIGHT_TEMPERATURE = 5.0
MAX_IMAGE_SIDE = 1024

SOURCES = {
    "mary_lynn_reed.jpg": {
        "title": "Mary Lynn Reed in 2023", "author": "Bravoyodel",
        "license": "CC BY-SA 4.0",
        "source_page": "https://commons.wikimedia.org/wiki/File:Mary_Lynn_Reed_in_2023.jpg",
    },
    "myriam_charlier.jpg": {
        "title": "Myriam Charlier, Portrait (2)", "author": "Horst J. Meuter",
        "license": "CC BY-SA 4.0",
        "source_page": "https://commons.wikimedia.org/wiki/File:Myriam_Charlier,_Portrait_(2).jpg",
    },
    "dr_mary_yang.jpg": {
        "title": "Dr. Mary Yang", "author": "IJFIPM",
        "license": "CC BY-SA 3.0 / GFDL",
        "source_page": "https://commons.wikimedia.org/wiki/File:Dr._Mary_Yang.jpg",
    },
    "female_officer_saluting.jpg": {
        "title": "Female officer saluting",
        "author": "U.S. Navy photo by Mass Communication Specialist 2nd Class Kevin S. O'Brien",
        "license": "U.S. federal government work / public domain in the United States",
        "source_page": "https://commons.wikimedia.org/wiki/File:Female_officer_saluting.jpg",
    },
    "judith_driscoll.jpg": {
        "title": "Prof. Judith Driscoll, University of Cambridge",
        "author": "Trinity College, Cambridge / Library staff of Trinity College",
        "license": "CC BY-SA 4.0",
        "source_page": "https://commons.wikimedia.org/wiki/File:Prof._Judith_Driscoll,_University_of_Cambridge.jpg",
    },
}


def resize_for_test(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    cv2 = _load_cv2()
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image.copy()
    scale = max_side / float(largest)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def reference_weights(scores: Iterable[float], temperature: float = REFERENCE_WEIGHT_TEMPERATURE) -> np.ndarray:
    """Normalize rule scores into relative reference weights; not empirical probability."""
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty, finite one-dimensional sequence")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    exps = np.exp((values - values.max()) / temperature)
    return exps / exps.sum()


def circular_arc_factor(chord_mm: float, radius_mm: float) -> float:
    """Theoretical arc/chord factor for a circular arc, under explicitly assumed geometry."""
    if not (math.isfinite(chord_mm) and math.isfinite(radius_mm)):
        raise ValueError("chord and radius must be finite")
    if chord_mm <= 0 or radius_mm <= 0 or chord_mm >= 2.0 * radius_mm:
        raise ValueError("require 0 < chord_mm < 2 * radius_mm")
    theta = 2.0 * math.asin(chord_mm / (2.0 * radius_mm))
    return (radius_mm * theta) / chord_mm


def relative_depth_factor(planar_px: float, depth_delta_px: float) -> float:
    """Mirror the app's experimental relative-z factor, not physical skin-surface curvature."""
    if not (math.isfinite(planar_px) and math.isfinite(depth_delta_px)) or planar_px <= 0:
        return float("nan")
    return min(1.4, max(1.0, math.hypot(planar_px, depth_delta_px) / planar_px))


def image_augmentations(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Fixed acquisition perturbations. They are repeated views, not independent people."""
    cv2 = _load_cv2()
    h, w = image.shape[:2]
    variants: list[tuple[str, np.ndarray]] = [("original", image.copy())]
    variants.extend([
        ("brightness_090", cv2.convertScaleAbs(image, alpha=0.90, beta=0)),
        ("brightness_110", cv2.convertScaleAbs(image, alpha=1.10, beta=0)),
        ("contrast_090", cv2.convertScaleAbs(image, alpha=0.90, beta=8)),
        ("contrast_110", cv2.convertScaleAbs(image, alpha=1.10, beta=-8)),
        ("blur_3x3", cv2.GaussianBlur(image, (3, 3), 0)),
    ])
    for angle in (-2.0, 2.0):
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        variants.append((f"rotate_{angle:+.0f}deg", cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )))
    for scale in (0.96, 1.04):
        resized = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        rh, rw = resized.shape[:2]
        if scale < 1:
            canvas = cv2.copyMakeBorder(
                resized, (h - rh) // 2, h - rh - (h - rh) // 2,
                (w - rw) // 2, w - rw - (w - rw) // 2, cv2.BORDER_REFLECT_101,
            )
        else:
            y0, x0 = max(0, (rh - h) // 2), max(0, (rw - w) // 2)
            canvas = resized[y0:y0 + h, x0:x0 + w]
        variants.append((f"scale_{scale:.2f}", canvas))
    for gamma in (0.90, 1.10):
        lut = np.clip((np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)
        variants.append((f"gamma_{gamma:.2f}", cv2.LUT(image, lut)))
    for crop_fraction in (0.015, 0.03):
        dx, dy = int(round(w * crop_fraction)), int(round(h * crop_fraction))
        if 2 * dx < w and 2 * dy < h:
            crop = image[dy:h - dy, dx:w - dx]
            variants.append((f"crop_{crop_fraction:.3f}", cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)))
    return variants


def landmark_array(landmarks: Any, width: int, height: int) -> np.ndarray:
    if len(landmarks) < 455:
        raise ValueError(f"need at least 455 facial landmarks, got {len(landmarks)}")
    return np.asarray([(float(lm.x) * width, float(lm.y) * height, float(lm.z) * width) for lm in landmarks], dtype=np.float64)


def face_measurements(points: np.ndarray) -> dict[str, float]:
    def distance(a: int, b: int) -> float:
        return float(np.linalg.norm(points[a, :2] - points[b, :2]))
    face_width = max(distance(234, 454), 1.0)
    face_height = max(distance(10, 152), 1.0)
    interocular = distance(33, 263)
    jaw_width = distance(172, 397)
    brow_eye = (
        abs(points[105, 1] - points[159, 1]) + abs(points[334, 1] - points[386, 1])
    ) / 2.0 / face_height
    asymmetry = abs(points[105, 1] - points[334, 1]) / face_height
    eye_tilt = math.degrees(math.atan2(
        points[263, 1] - points[33, 1], points[263, 0] - points[33, 0]
    )) / 90.0
    return {
        "face_width_height_ratio": face_width / face_height,
        "interocular_face_width_ratio": interocular / face_width,
        "jaw_face_width_ratio": jaw_width / face_width,
        "brow_eye_distance_ratio": brow_eye,
        "brow_asymmetry_ratio": asymmetry,
        "eye_line_tilt": eye_tilt,
        "interocular_distance_px": interocular,
        "face_width_px": face_width,
        "face_height_px": face_height,
    }


def candidate_matrix(features: dict[str, float]) -> np.ndarray:
    rows = []
    for candidate in CANDIDATES:
        style_index = int(candidate["style_index"])
        rows.append([
            features["face_width_height_ratio"],
            features["interocular_face_width_ratio"],
            features["jaw_face_width_ratio"],
            features["brow_eye_distance_ratio"],
            features["brow_asymmetry_ratio"],
            features["eye_line_tilt"],
            candidate["thickness"], candidate["length"], candidate["arch"],
            1.0 if style_index == 1 else 0.0,
            1.0 if style_index == 2 else 0.0,
            0.0, 0.0, 0.0, 0.0,  # no reviewed native-brow mask for these public photos
        ])
    return np.asarray(rows, dtype=np.float64)


def score_candidates(features: dict[str, float]) -> list[dict[str, Any]]:
    matrix = candidate_matrix(features)
    scores = geometric_reference_scores(matrix)
    weights = reference_weights(scores)
    return [{
        "style": candidate["style"],
        "geometry_reference_score": float(score),
        "normalized_reference_weight_temperature_5": float(weight),
        "thickness": candidate["thickness"],
        "length": candidate["length"],
        "arch": candidate["arch"],
        "not_a_preference_probability": True,
    } for candidate, score, weight in zip(CANDIDATES, scores, weights)]


def catmull_rom(points: np.ndarray, samples_per_segment: int = 10) -> np.ndarray:
    if len(points) < 2:
        return points.astype(np.int32)
    output: list[np.ndarray] = []
    for i in range(len(points) - 1):
        p0, p1 = points[max(0, i - 1)], points[i]
        p2, p3 = points[i + 1], points[min(len(points) - 1, i + 2)]
        for j in range(samples_per_segment):
            t = j / float(samples_per_segment)
            t2, t3 = t * t, t * t * t
            output.append(0.5 * (
                2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            ))
    output.append(points[-1])
    return np.asarray(output, dtype=np.float64)


def candidate_brow_curve(points: np.ndarray, landmark_ids: Iterable[int], *, length_factor: float, arch_factor: float) -> np.ndarray:
    anchors = points[list(landmark_ids), :2].astype(np.float64)
    anchors = anchors[np.argsort(anchors[:, 0])]
    x_mid = float((anchors[0, 0] + anchors[-1, 0]) / 2.0)
    anchors[:, 0] = x_mid + (anchors[:, 0] - x_mid) * length_factor
    x0, y0 = anchors[0]
    x1, y1 = anchors[-1]
    span = max(float(x1 - x0), 1.0)
    for index, anchor in enumerate(anchors):
        t = float((anchor[0] - x0) / span)
        baseline_y = y0 + t * (y1 - y0)
        anchors[index, 1] = baseline_y + arch_factor * (anchor[1] - baseline_y)
    return np.rint(catmull_rom(anchors)).astype(np.int32)


def draw_candidate(image: np.ndarray, points: np.ndarray, candidate: dict[str, Any], label: str) -> np.ndarray:
    cv2 = _load_cv2()
    overlay = image.copy()
    interocular_px = float(np.linalg.norm(points[33, :2] - points[263, :2]))
    line_width = max(2, int(round(interocular_px * 0.042 * float(candidate["thickness"]))))
    colors = {"natural": (45, 38, 31), "soft": (70, 52, 64), "defined": (28, 25, 23)}
    color = colors[str(candidate["style"])]
    for ids in BROW_LANDMARK_SETS:
        curve = candidate_brow_curve(
            points, ids, length_factor=float(candidate["length"]), arch_factor=float(candidate["arch"])
        )
        cv2.polylines(overlay, [curve], False, color, line_width, cv2.LINE_AA)
    output = cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0)
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (250, 250, 250), -1)
    cv2.putText(output, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    return output


def relative_depth_factors(points: np.ndarray) -> dict[str, float]:
    """Mirror the app's experimental MediaPipe-z factor; it is not physical curvature."""
    factors: dict[str, float] = {}
    nose_x = float(points[1, 0])
    for ids in BROW_LANDMARK_SETS:
        usable = sorted(ids, key=lambda i: points[i, 0])
        side_center = float(np.mean([points[i, 0] for i in usable]))
        anatomical_right = side_center < nose_x
        names = ["右眉尾", "右眉峰", "右眉身", "右眉头"] if anatomical_right else ["左眉头", "左眉身", "左眉峰", "左眉尾"]
        for index in range(len(usable) - 1):
            a, b = usable[index], usable[index + 1]
            planar = math.hypot(float(points[b, 0] - points[a, 0]), float(points[b, 1] - points[a, 1]))
            factor = relative_depth_factor(planar, float(points[b, 2] - points[a, 2]))
            if math.isfinite(factor):
                factors[names[index]] = factor
    return factors


def percentile(values: list[float], q: float) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.quantile(np.asarray(clean), q)) if clean else None


def import_landmarker(model_path: Path):
    try:
        import mediapipe as mp
    except Exception as exc:
        raise RuntimeError("MediaPipe is required. Install a compatible mediapipe wheel.") from exc
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_facial_transformation_matrixes=False,
        output_face_blendshapes=False,
    )
    return mp, mp.tasks.vision.FaceLandmarker.create_from_options(options)


def detect_points(mp: Any, landmarker: Any, image: np.ndarray) -> np.ndarray | None:
    cv2 = _load_cv2()
    if image is None or image.size == 0:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    except Exception:
        return None
    if not result.face_landmarks:
        return None
    chosen = max(
        result.face_landmarks,
        key=lambda lm: (max(p.x for p in lm) - min(p.x for p in lm)) * (max(p.y for p in lm) - min(p.y for p in lm)),
    )
    return landmark_array(chosen, image.shape[1], image.shape[0])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("no_rows\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(image_dir: Path, face_model: Path, output_dir: Path) -> dict[str, Any]:
    cv2 = _load_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    private_preview_dir = output_dir / "_private_previews"
    private_preview_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not images:
        raise ValueError(f"No images found in {image_dir}")

    mp, landmarker = import_landmarker(face_model)
    image_rows: list[dict[str, Any]] = []
    candidate_rows_out: list[dict[str, Any]] = []
    depth_rows_out: list[dict[str, Any]] = []
    try:
        for image_path in images:
            image = cv2.imread(str(image_path))
            source = SOURCES.get(image_path.name, {
                "title": image_path.name, "author": "Unknown", "license": "See source page", "source_page": "",
            })
            if image is None:
                image_rows.append({"image": image_path.name, "status": "unreadable_image", "source_title": source["title"]})
                continue
            image = resize_for_test(image)
            baseline_points = detect_points(mp, landmarker, image)
            transforms = image_augmentations(image)
            if baseline_points is None:
                image_rows.append({
                    "image": image_path.name, "status": "no_face_landmarks", "source_title": source["title"],
                    "source_page": source["source_page"], "license": source["license"],
                    "perturbation_attempts": len(transforms), "successful_perturbation_detections": 0,
                })
                continue

            features = face_measurements(baseline_points)
            scored = score_candidates(features)
            best_style = max(scored, key=lambda row: row["geometry_reference_score"])["style"]
            for row in scored:
                candidate_rows_out.append({
                    "image": image_path.name, "source_title": source["title"],
                    "style": row["style"], "geometry_reference_score": row["geometry_reference_score"],
                    "normalized_reference_weight_temperature_5": row["normalized_reference_weight_temperature_5"],
                    "thickness": row["thickness"], "length": row["length"], "arch": row["arch"],
                    "not_a_preference_probability": True,
                })
                preview = draw_candidate(image, baseline_points, row, f"{source['title']} | {row['style']}")
                cv2.imwrite(str(private_preview_dir / f"{image_path.stem}_{row['style']}.jpg"), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

            top_styles: list[str] = []
            factor_samples: dict[str, list[float]] = defaultdict(list)
            success = 0
            for transform_name, variant in transforms:
                points = detect_points(mp, landmarker, variant)
                if points is None:
                    continue
                success += 1
                variant_scores = score_candidates(face_measurements(points))
                top_styles.append(max(variant_scores, key=lambda row: row["geometry_reference_score"])["style"])
                for region, factor in relative_depth_factors(points).items():
                    factor_samples[region].append(float(factor))
                    depth_rows_out.append({
                        "image": image_path.name, "source_title": source["title"],
                        "augmentation": transform_name, "region": region, "relative_z_factor": factor,
                        "not_metric_mm": True, "not_validated_surface_curvature": True,
                    })
            stability = sum(x == best_style for x in top_styles) / len(top_styles) if top_styles else None
            image_rows.append({
                "image": image_path.name, "status": "ok", "source_title": source["title"],
                "source_page": source["source_page"], "license": source["license"],
                "face_width_height_ratio": features["face_width_height_ratio"],
                "interocular_face_width_ratio": features["interocular_face_width_ratio"],
                "jaw_face_width_ratio": features["jaw_face_width_ratio"],
                "brow_eye_distance_ratio": features["brow_eye_distance_ratio"],
                "brow_asymmetry_ratio": features["brow_asymmetry_ratio"],
                "eye_line_tilt": features["eye_line_tilt"],
                "geometry_rule_top1": best_style,
                "top1_stability_under_image_perturbations": stability,
                "successful_perturbation_detections": success,
                "perturbation_attempts": len(transforms),
                "normalized_reference_weights_are_probabilities": False,
                "media_pipe_depth_is_metric": False,
            })
            for region in CURVATURE_REGIONS:
                values = factor_samples.get(region, [])
                if values:
                    depth_rows_out.append({
                        "image": image_path.name, "source_title": source["title"], "augmentation": "SUMMARY",
                        "region": region, "relative_z_factor_median": percentile(values, 0.50),
                        "p10_across_image_perturbations": percentile(values, 0.10),
                        "p90_across_image_perturbations": percentile(values, 0.90),
                        "n_successful_augmentations": len(values),
                        "not_metric_mm": True, "not_validated_surface_curvature": True,
                    })
    finally:
        landmarker.close()

    write_csv(output_dir / "image_level_results.csv", image_rows)
    write_csv(output_dir / "candidate_reference_weights.csv", candidate_rows_out)
    write_csv(output_dir / "relative_depth_sensitivity.csv", depth_rows_out)

    surface_rows: list[dict[str, Any]] = []
    for chord in (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0):
        for radius in (15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0):
            if chord >= 2.0 * radius:
                continue
            factor = circular_arc_factor(chord, radius)
            surface_rows.append({
                "assumed_chord_mm": chord, "assumed_local_radius_mm": radius,
                "theoretical_arc_to_chord_factor": factor,
                "extra_path_length_pct_if_applied": (factor - 1.0) * 100.0,
                "status": "synthetic_surface_scenario_not_measured_face_data",
            })
    write_csv(output_dir / "synthetic_surface_sensitivity.csv", surface_rows)

    valid_images = [row for row in image_rows if row.get("status") == "ok"]
    top_counts = Counter(row["geometry_rule_top1"] for row in valid_images)
    stability_values = [float(row["top1_stability_under_image_perturbations"]) for row in valid_images
                        if row.get("top1_stability_under_image_perturbations") is not None]
    factors_grid = [float(row["theoretical_arc_to_chord_factor"]) for row in surface_rows]
    summary = {
        "experiment_type": "public_image_engineering_stress_test",
        "not_real_customer_preference_model": True,
        "n_images_downloaded": len(images),
        "n_images_with_detected_face_landmarks": len(valid_images),
        "candidate_styles": [row["style"] for row in CANDIDATES],
        "perturbations_per_image_including_original": len(image_augmentations(np.zeros((32, 32, 3), dtype=np.uint8))),
        "geometry_rule_top1_image_counts": dict(top_counts),
        "geometry_rule_top1_image_share": {style: count / len(valid_images) for style, count in top_counts.items()} if valid_images else {},
        "mean_top1_stability_under_image_perturbations": float(np.mean(stability_values)) if stability_values else None,
        "min_top1_stability_under_image_perturbations": float(np.min(stability_values)) if stability_values else None,
        "synthetic_surface_sweep": {
            "n_assumed_chord_radius_scenarios": len(factors_grid),
            "factor_min": float(np.min(factors_grid)),
            "factor_p10_scenario_grid": float(np.quantile(factors_grid, 0.10)),
            "factor_median_scenario_grid": float(np.median(factors_grid)),
            "factor_p90_scenario_grid": float(np.quantile(factors_grid, 0.90)),
            "factor_max": float(np.max(factors_grid)),
            "interpretation": "Scenario-grid quantiles, not a population interval or prediction interval.",
        },
        "source_manifest": SOURCES,
        "limitations": [
            "No customer preference ratings exist for these public portraits; normalized geometry weights are not preference probabilities.",
            "The top-style share is the fraction of this small image sample ranked first by a fixed geometric rule, not a probability of human preference.",
            "Image perturbations test landmark/ranking stability, not aesthetic validity or customer generalization.",
            "MediaPipe z is a relative depth estimate, not calibrated skin-surface geometry in millimetres.",
            "Synthetic arc factors test formula sensitivity only; chord and radius values are assumptions, not measurements.",
            "This convenience sample is not representative or a consented salon training set.",
        ],
    }
    (output_dir / "public_image_experiment.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = [
        "# Public-image brow engineering stress test", "",
        f"- Images downloaded: {len(images)}",
        f"- Images with detected face landmarks: {len(valid_images)}",
        f"- Candidate styles: {', '.join(row['style'] for row in CANDIDATES)}",
        f"- Mean top-1 ranking stability under image perturbations: {'not available' if not stability_values else f'{np.mean(stability_values):.1%}'}",
        f"- Top-ranked style counts in this image sample: {dict(top_counts)}", "",
        "## Interpretation", "",
        "The scores are the app's existing geometric baseline. Normalized weights are a softmax "
        "transformation of rule scores with an arbitrary temperature of 5. They are not trained, "
        "calibrated or validated customer preference probabilities.",
        "",
        "Repeated brightness, contrast, rotation, scale, gamma, blur and crop perturbations estimate "
        "ranking stability to small image changes. They are not independent people and do not prove "
        "aesthetic correctness.",
        "",
        "The relative-depth file reproduces the app's experimental MediaPipe-z ratio. It is not a "
        "measurement of skin-surface arc length in millimetres.",
        "",
        "The circular-surface sweep is a mathematical sensitivity check over an assumed chord/radius "
        "grid, not evidence of real face geometry or real printed transfer accuracy.",
        "",
        "## Inputs and licences", "",
    ]
    for filename, source in SOURCES.items():
        report.append(f"- **{source['title']}** — {source['author']}; {source['license']}; source: {source['source_page']}; test file: {filename}.")
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    attribution = [
        "PUBLIC-IMAGE BROW EXPERIMENT: IMAGE ATTRIBUTION",
        "Source portraits are downloaded only at workflow runtime. Rendered face derivatives are kept out of the uploaded artifact.",
        "CC BY-SA-derived works must comply with attribution and share-alike terms.",
        "",
    ]
    for source in SOURCES.values():
        attribution.extend([
            f"Title: {source['title']}", f"Author: {source['author']}",
            f"Licence: {source['license']}", f"Source: {source['source_page']}", "",
        ])
    (output_dir / "ATTRIBUTION.txt").write_text("\n".join(attribution) + "\n", encoding="utf-8")

    # Keep the image derivatives internal to the Actions runner. Only numerical reports are uploaded.
    for preview in private_preview_dir.glob("*"):
        preview.unlink()
    private_preview_dir.rmdir()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("public-image-brow-report"))
    args = parser.parse_args()
    summary = run_experiment(args.image_dir, args.face_model, args.output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
