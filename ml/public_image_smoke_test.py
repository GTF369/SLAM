#!/usr/bin/env python3
"""Smoke-test the exact bundled MediaPipe and EasyPortrait ONNX assets on public photos.

This is a pipeline smoke test, not a segmentation accuracy benchmark: test photos have
no professionally annotated eyebrow masks. It uses the app's crop, normalization,
class 6/7 mapping, ROI bounds and adaptive-contrast fallback.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageEnhance, ImageFilter

PARSING_SIZE = 384
MEAN_BGR = np.array([143.55267075, 132.96705975, 126.94924335], dtype=np.float32)
STD_BGR = np.array([60.2625333, 60.32740275, 59.30988645], dtype=np.float32)
BROW_SETS = [
    [70, 63, 105, 66, 107],
    [300, 293, 334, 296, 336],
]
REGION_NAMES = ["image-left eyebrow ROI", "image-right eyebrow ROI"]


def point_pixels(landmark: Any, width: int, height: int) -> tuple[float, float]:
    return float(landmark.x) * width, float(landmark.y) * height


def region_bounds(points: list[tuple[float, float]], ids: list[int], width: int, height: int):
    valid = [points[i] for i in ids if i < len(points)]
    if len(valid) < 4:
        return None
    xs = [p[0] for p in valid]
    ys = [p[1] for p in valid]
    span = max(8.0, max(xs) - min(xs))
    pad_x = max(6.0, span * 0.14)
    pad_top = max(8.0, span * 0.42)
    pad_bottom = max(7.0, span * 0.25)
    x0 = max(0, min(width - 1, int(min(xs)) - int(pad_x)))
    x1 = max(x0 + 1, min(width, int(max(xs)) + int(pad_x) + 1))
    y0 = max(0, min(height - 1, int(min(ys)) - int(pad_top)))
    y1 = max(y0 + 1, min(height, int(max(ys)) + int(pad_bottom) + 1))
    return [x0, y0, x1, y1]


def app_face_crop(image: np.ndarray, points: list[tuple[float, float]]):
    h, w = image.shape[:2]
    if len(points) <= 454:
        return None
    left_x, _ = points[234]
    right_x, _ = points[454]
    _, top_y = points[10]
    _, bottom_y = points[152]
    face_width = abs(right_x - left_x)
    face_height = abs(bottom_y - top_y)
    if not np.isfinite([face_width, face_height]).all() or min(face_width, face_height) < 16:
        return None
    requested_side = int(max(face_width, face_height) * 1.22)
    side = max(64, min(requested_side, min(w, h)))
    center_x = (left_x + right_x) / 2.0
    center_y = (top_y + bottom_y) / 2.0
    crop_left = max(0, min(w - side, int(center_x - side / 2.0)))
    crop_top = max(0, min(h - side, int(center_y - side / 2.0)))
    crop = image[crop_top:crop_top + side, crop_left:crop_left + side]
    if crop.size == 0:
        return None
    scaled = np.asarray(Image.fromarray(crop).resize((PARSING_SIZE, PARSING_SIZE), Image.Resampling.BILINEAR))
    # Match the Android app: BGR, NCHW, pixels in [0,255], model mean/std.
    bgr = scaled[:, :, ::-1].astype(np.float32)
    tensor = ((bgr - MEAN_BGR) / STD_BGR).transpose(2, 0, 1)[None, ...].copy()
    return tensor, crop_left, crop_top, side


def map_labels_to_roi(labels: np.ndarray, crop_left: int, crop_top: int, side: int, bounds: list[int]):
    x0, y0, x1, y1 = bounds
    roi_w, roi_h = x1 - x0, y1 - y0
    mask = np.zeros((roi_h, roi_w), dtype=bool)
    for y in range(roi_h):
        model_y = int(((y0 + y + 0.5 - crop_top) * PARSING_SIZE) / side)
        if not 0 <= model_y < PARSING_SIZE:
            continue
        xs = np.arange(roi_w, dtype=np.float32) + x0 + 0.5
        model_x = ((xs - crop_left) * PARSING_SIZE / side).astype(np.int32)
        in_range = (model_x >= 0) & (model_x < PARSING_SIZE)
        row_labels = np.zeros(roi_w, dtype=np.uint8)
        row_labels[in_range] = labels[model_y, model_x[in_range]]
        mask[y, :] = (row_labels == 6) | (row_labels == 7)
    return mask


def adaptive_mask(roi: np.ndarray, sensitivity: float = 0.7884):
    rgb = roi.astype(np.float32)
    luma = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.int16)
    h, w = luma.shape
    yy, xx = np.indices((h, w))
    edge_x = max(1, w // 8)
    top_band = max(1, h // 4)
    bottom_start = h * 3 // 4
    border = (yy < top_band) | (yy >= bottom_start) | (xx < edge_x) | (xx >= w - edge_x)
    border_samples = np.sort(luma[border])
    skin_ref = int(border_samples[len(border_samples) // 2]) if len(border_samples) else 150
    threshold = max(28, min(175, int(skin_ref * max(0.45, min(0.92, sensitivity)))))
    raw = luma < threshold
    filtered = np.zeros_like(raw, dtype=bool)
    if h > 2 and w > 2:
        neighbors = np.zeros((h - 2, w - 2), dtype=np.uint8)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                neighbors += raw[dy:dy + h - 2, dx:dx + w - 2]
        filtered[1:-1, 1:-1] = raw[1:-1, 1:-1] & (neighbors >= 2)
    return filtered


def run_case(image_path: Path, case_name: str, landmarker, session, out_dir: Path):
    original = Image.open(image_path).convert("RGB")
    image = np.asarray(original)
    h, w = image.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    t0 = time.perf_counter()
    landmarks_result = landmarker.detect(mp_image)
    face_ms = (time.perf_counter() - t0) * 1000.0
    report: dict[str, Any] = {
        "case": case_name, "source_file": image_path.name, "width": w, "height": h,
        "face_detected": bool(landmarks_result.face_landmarks),
        "face_landmark_count": 0, "face_landmark_ms": round(face_ms, 2),
    }
    if not landmarks_result.face_landmarks:
        report.update({"status": "FAIL", "reason": "MediaPipe did not detect a face"})
        return report

    landmarks = landmarks_result.face_landmarks[0]
    points = [point_pixels(p, w, h) for p in landmarks]
    report["face_landmark_count"] = len(points)
    bounds = [region_bounds(points, ids, w, h) for ids in BROW_SETS]
    if any(b is None for b in bounds) or len(points) < 455:
        report.update({"status": "FAIL", "reason": "Insufficient landmarks for app brow ROIs"})
        return report

    crop_result = app_face_crop(image, points)
    if crop_result is None:
        report.update({"status": "FAIL", "reason": "Could not make app-equivalent face crop"})
        return report
    tensor, crop_left, crop_top, side = crop_result
    input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
    t0 = time.perf_counter()
    logits = session.run([output_meta.name], {input_meta.name: tensor})[0]
    onnx_ms = (time.perf_counter() - t0) * 1000.0
    if logits.shape != (1, 19, PARSING_SIZE, PARSING_SIZE) or not np.isfinite(logits).all():
        report.update({"status": "FAIL", "reason": f"Unexpected model output {logits.shape} or non-finite logits"})
        return report
    labels = logits[0].argmax(axis=0).astype(np.uint8)
    counts = np.bincount(labels.reshape(-1), minlength=19)
    report["parser_inference_ms_cpu"] = round(onnx_ms, 2)
    report["parser_output_shape"] = list(logits.shape)
    report["face_crop_xyxy"] = [crop_left, crop_top, crop_left + side, crop_top + side]
    report["parser_class_pixels"] = {str(i): int(counts[i]) for i in range(19)}
    report["class_6_7_pixels_percent_of_face_crop"] = round(float(np.mean((labels == 6) | (labels == 7)) * 100), 3)
    local_brow_labels = (labels == 6) | (labels == 7)
    ys, xs = np.where(local_brow_labels)
    if len(xs):
        report["class_6_7_bbox_model_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        source_x0 = max(0, crop_left + int(xs.min() * side / PARSING_SIZE))
        source_y0 = max(0, crop_top + int(ys.min() * side / PARSING_SIZE))
        source_x1 = min(w, crop_left + int((xs.max() + 1) * side / PARSING_SIZE) + 1)
        source_y1 = min(h, crop_top + int((ys.max() + 1) * side / PARSING_SIZE) + 1)
        report["class_6_7_bbox_source_xyxy_approx"] = [source_x0, source_y0, source_x1, source_y1]
    else:
        report["class_6_7_bbox_model_xyxy"] = None
        report["class_6_7_bbox_source_xyxy_approx"] = None
    report["roi_results"] = []
    final_mask = np.zeros((h, w), dtype=bool)

    for name, b in zip(REGION_NAMES, bounds):
        assert b is not None
        x0, y0, x1, y1 = b
        model_mask = map_labels_to_roi(labels, crop_left, crop_top, side, b)
        roi = image[y0:y1, x0:x1]
        model_count = int(model_mask.sum())
        minimum = max(5, int(model_mask.size * 0.003))
        used_model = model_count >= minimum
        chosen = model_mask if used_model else adaptive_mask(roi)
        final_mask[y0:y1, x0:x1] |= chosen
        report["roi_results"].append({
            "name": name, "bounds_xyxy": b, "roi_pixels": int(model_mask.size),
            "model_brow_pixels": model_count,
            "model_brow_fraction": round(float(model_count / max(1, model_mask.size)), 4),
            "model_mask_accepted_by_app_gate": bool(used_model),
            "selected_method": "EasyPortrait class 6/7" if used_model else "adaptive contrast fallback",
            "final_mask_pixels": int(chosen.sum()),
            "final_mask_fraction": round(float(chosen.mean()), 4),
        })

    overlay = image.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 220
    green[:, :, 2] = 170
    alpha = 0.56
    overlay[final_mask] = (
        overlay[final_mask].astype(np.float32) * (1 - alpha)
        + green[final_mask].astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    # Outline the two evaluated ROI bounds, so the output is easy to inspect.
    for b in bounds:
        x0, y0, x1, y1 = b
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 60, 30), 1)
    output_path = out_dir / f"{case_name}_brow_overlay.jpg"
    Image.fromarray(overlay).save(output_path, quality=90)
    # Separate the model-only labels from the fallback path to debug coordinate
    # transforms and label-map assumptions. The source is the same crop used by the app.
    face_crop_mask = Image.fromarray((local_brow_labels.astype(np.uint8) * 255)).resize(
        (side, side), Image.Resampling.NEAREST
    )
    model_only = np.zeros((h, w), dtype=bool)
    model_only[crop_top:crop_top + side, crop_left:crop_left + side] = np.asarray(face_crop_mask) > 0
    model_overlay = image.copy()
    orange = np.zeros_like(model_overlay)
    orange[:, :, 0] = 255
    orange[:, :, 1] = 70
    model_overlay[model_only] = (
        model_overlay[model_only].astype(np.float32) * 0.42
        + orange[model_only].astype(np.float32) * 0.58
    ).clip(0, 255).astype(np.uint8)
    model_path = out_dir / f"{case_name}_model_classes_6_7.jpg"
    Image.fromarray(model_overlay).save(model_path, quality=90)
    report["preview_file"] = output_path.name
    report["model_only_preview_file"] = model_path.name
    report["total_final_mask_pixels"] = int(final_mask.sum())
    report["total_final_mask_fraction_of_image"] = round(float(final_mask.mean()), 6)
    report["status"] = "PASS_SMOKE_ONLY" if final_mask.sum() > 0 else "FAIL"
    report["accuracy_claim"] = "No ground-truth masks supplied; this tests face detection, tensor inference and app-equivalent mask path only."
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--parser-model", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create stress variants from a public demo portrait as well as testing original sources.
    demo_candidates = [args.images_dir / "virtual_makeup_demo.png"]
    demo_path = next((p for p in demo_candidates if p.is_file()), None)
    if demo_path:
        demo = Image.open(demo_path).convert("RGB")
        ImageEnhance.Brightness(demo).enhance(0.55).save(args.images_dir / "demo_low_light.jpg", quality=88)
        demo.filter(ImageFilter.GaussianBlur(radius=2)).save(args.images_dir / "demo_blurred.jpg", quality=88)
        demo.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(args.images_dir / "demo_mirrored.jpg", quality=88)

    base_options = mp.tasks.BaseOptions(model_asset_path=str(args.face_model))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    session = ort.InferenceSession(str(args.parser_model), providers=["CPUExecutionProvider"])
    reports = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        for path in sorted(args.images_dir.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            try:
                reports.append(run_case(path, path.stem, landmarker, session, args.output_dir))
            except Exception as exc:
                reports.append({"case": path.stem, "source_file": path.name, "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"})

    summary = {
        "test_type": "public_image_model_smoke_test_not_android_runtime",
        "face_model": args.face_model.name,
        "parser_model": args.parser_model.name,
        "onnx_input": {"name": session.get_inputs()[0].name, "shape": session.get_inputs()[0].shape},
        "onnx_output": {"name": session.get_outputs()[0].name, "shape": session.get_outputs()[0].shape},
        "cases": reports,
        "passed": sum(item.get("status") == "PASS_SMOKE_ONLY" for item in reports),
        "failed": sum(item.get("status") == "FAIL" for item in reports),
        "limitations": [
            "This does not install or launch the ARM64 Android APK.",
            "Public photos have no expert eyebrow-mask ground truth, so mask accuracy is not established.",
            "The Python implementation follows the app's model preprocessing, label mapping and ROI fallback rules, but is not bit-exact Android image rendering.",
            "Real OPPO Find X7 and printer/transfer validation is still required."
        ],
    }
    (args.output_dir / "smoke_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not reports or summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
