#!/usr/bin/env python3
"""Generate AI pseudo-labels by blind-scoring eyebrow overlays on licensed public portraits.

These judgments are model-generated design opinions, NOT real customer preference labels,
not calibrated probabilities, and not a releasable customer-preference model.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from public_image_brow_benchmark import (  # noqa: E402
    BROW_LANDMARK_SETS,
    CANDIDATES,
    SOURCES,
    candidate_brow_curve,
    detect_points,
    face_measurements,
    import_landmarker,
    resize_for_test,
)

CHOICE_STYLES = ("natural", "soft", "defined")
OPTION_IDS = ("A", "B", "C")
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "option_id": {"type": "string", "enum": ["A", "B", "C"]},
                    "compatibility_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "naturalness_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "proportion_fit_score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "notes": {"type": "string"},
                },
                "required": [
                    "option_id", "compatibility_score", "naturalness_score",
                    "proportion_fit_score", "notes",
                ],
                "additionalProperties": False,
            },
            "minItems": 3,
            "maxItems": 3,
        },
        "uncertainty": {"type": "string"},
    },
    "required": ["scores", "uncertainty"],
    "additionalProperties": False,
}


def _face_crop_bounds(image: np.ndarray, points: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    xmin, ymin = np.min(points[:, :2], axis=0)
    xmax, ymax = np.max(points[:, :2], axis=0)
    face_width = max(float(xmax - xmin), 1.0)
    face_height = max(float(ymax - ymin), 1.0)
    left = max(0, int(xmin - 0.15 * face_width))
    right = min(width, int(xmax + 0.15 * face_width))
    top = max(0, int(ymin - 0.12 * face_height))
    bottom = min(height, int(ymax + 0.12 * face_height))
    if right <= left or bottom <= top:
        return 0, 0, width, height
    return left, top, right, bottom


def draw_blind_candidate(
    image: np.ndarray, points: np.ndarray, candidate: dict[str, Any], option_id: str
) -> np.ndarray:
    """Render all candidates with one brow colour so colour cannot reveal the style label."""
    canvas = image.copy()
    interocular = float(np.linalg.norm(points[33, :2] - points[263, :2]))
    stroke = max(2, int(round(interocular * 0.042 * float(candidate["thickness"]))))
    colour = (38, 35, 31)  # identical for all options
    overlay = canvas.copy()
    for landmark_ids in BROW_LANDMARK_SETS:
        curve = candidate_brow_curve(
            points,
            landmark_ids,
            length_factor=float(candidate["length"]),
            arch_factor=float(candidate["arch"]),
        )
        if len(curve) >= 3:
            curve_f = curve.astype(np.float32)
            tangents = np.gradient(curve_f, axis=0)
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            norms[norms < 1e-4] = 1.0
            normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / norms
            half_width = max(1.0, stroke * 0.68)
            upper = curve_f + normals * half_width
            lower = curve_f - normals * half_width
            polygon = np.rint(np.vstack((upper, lower[::-1]))).astype(np.int32)
            cv2.fillPoly(overlay, [polygon], colour, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [curve], False, colour, max(1, stroke // 2), cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.76, canvas, 0.24, 0.0)

    left, top, right, bottom = _face_crop_bounds(canvas, points)
    crop = canvas[top:bottom, left:right].copy()
    if crop.size == 0:
        crop = canvas
    target_height = 620
    scale = min(1.0, target_height / max(crop.shape[0], 1))
    if scale < 1.0:
        crop = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    header = np.full((42, crop.shape[1], 3), 246, dtype=np.uint8)
    cv2.putText(header, f"OPTION {option_id}", (12, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.78, (28, 28, 28), 2, cv2.LINE_AA)
    return np.vstack((header, crop))


def make_contact_sheet(
    image: np.ndarray,
    points: np.ndarray,
    order: list[dict[str, Any]],
    option_ids: list[str],
) -> np.ndarray:
    panels = [
        draw_blind_candidate(image, points, candidate, option_id)
        for candidate, option_id in zip(order, option_ids)
    ]
    max_h = max(panel.shape[0] for panel in panels)
    padded: list[np.ndarray] = []
    for panel in panels:
        if panel.shape[0] < max_h:
            panel = cv2.copyMakeBorder(
                panel, 0, max_h - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                value=(246, 246, 246),
            )
        padded.append(panel)
    return np.hstack(padded)


def as_data_url(image: np.ndarray, max_side: int = 1900) -> str:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(
            image, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Could not encode contact sheet.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def validate_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("scores"), list):
        raise ValueError("The model response did not match the expected JSON object.")
    rows = value["scores"]
    if len(rows) != 3:
        raise ValueError("The model must score exactly three options.")
    if {row.get("option_id") for row in rows if isinstance(row, dict)} != set(OPTION_IDS):
        raise ValueError("The model must return each of A, B and C exactly once.")
    for row in rows:
        for key in ("compatibility_score", "naturalness_score", "proportion_fit_score"):
            score = row.get(key)
            if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 100:
                raise ValueError(f"Invalid {key} for option {row.get('option_id')}.")
        if not isinstance(row.get("notes"), str):
            raise ValueError("Candidate notes must be strings.")
    value["scores"].sort(key=lambda row: row["option_id"])
    return value


def query_vision_model(
    client: OpenAI,
    model: str,
    contact_sheet: np.ndarray,
    candidate_facts: dict[str, dict[str, float]],
) -> dict[str, Any]:
    data_url = as_data_url(contact_sheet)
    prompt = (
        "You are doing a blinded, repeatable eyebrow-design compatibility review. The image is a "
        "three-panel contact sheet labelled OPTION A, OPTION B and OPTION C. Each panel shows the "
        "same publicly licensed portrait with one candidate brow overlay. The prompt below gives the "
        "numeric design parameters mapped to those option letters. All overlays use the same colour. "
        "Score visible visual compatibility only: harmony with eye spacing and brow-eye distance, "
        "alignment with the existing brow region, balance of head/body/tail, naturalness, symmetry, "
        "and how well the design matches the stated intent 'balanced and natural'. Do not infer "
        "identity, ethnicity, health, personality, hidden taste or what a real customer would choose. "
        "Do not rank people or judge their inherent beauty. The overlays are simplified engineering "
        "previews, not photorealistic tattoo simulations. Return separate 0-100 relative design scores "
        "for each option and brief image-grounded reasons. These numbers are not probabilities and "
        "are not evidence of actual customer preferences. If the differences are too small to judge, "
        "say so in uncertainty.\n\nCandidate parameters by option:\n"
        + json.dumps(candidate_facts, ensure_ascii=False, sort_keys=True)
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "brow_design_compatibility_review",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            },
        },
        messages=[
            {
                "role": "system",
                "content": (
                    "Return only the requested structured JSON. Be conservative, consistent, and "
                    "explicit about uncertainty. Never describe an AI score as a customer preference probability."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The vision model returned an empty response.")
    return validate_judgment(json.loads(content))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("no_rows\n", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it as a repository Actions secret, then rerun this workflow."
        )
    if args.repeats < 3 or args.repeats > 9:
        raise ValueError("--repeats must be between 3 and 9 to limit cost and avoid overcounting repeats.")
    if not args.image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in args.image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise ValueError("No images found in --image-dir.")

    client = OpenAI(api_key=api_key, timeout=90.0, max_retries=2)
    mp, landmarker = import_landmarker(args.face_model)
    judgment_rows: list[dict[str, Any]] = []
    pseudo_choice_rows: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []
    all_scores: dict[str, list[float]] = defaultdict(list)
    winner_counts: Counter[str] = Counter()
    attempts = 0
    errors: list[dict[str, str]] = []
    try:
        for image_path in images:
            image = cv2.imread(str(image_path))
            source = SOURCES.get(image_path.name, {
                "title": image_path.name,
                "author": "User-provided",
                "license": "Verify the source licence before sending to a model",
                "source_page": "",
            })
            if image is None:
                image_summaries.append({"image": image_path.name, "status": "unreadable_image"})
                continue
            image = resize_for_test(image)
            points = detect_points(mp, landmarker, image)
            if points is None:
                image_summaries.append({
                    "image": image_path.name, "status": "no_face_landmarks",
                    "source_page": source["source_page"], "license": source["license"],
                })
                continue

            features = face_measurements(points)
            candidate_by_style = {str(candidate["style"]): candidate for candidate in CANDIDATES}
            case_id = "public_" + image_path.stem
            per_style_wins: Counter[str] = Counter()
            per_style_scores: dict[str, list[float]] = defaultdict(list)
            successes = 0
            for repeat_index in range(1, args.repeats + 1):
                rng = random.Random(args.seed + repeat_index * 7919 + sum(image_path.name.encode("utf-8")))
                order = list(CANDIDATES)
                rng.shuffle(order)
                options = list(OPTION_IDS)
                style_to_option = {
                    str(candidate["style"]): option_id
                    for candidate, option_id in zip(order, options)
                }
                option_to_style = {value: key for key, value in style_to_option.items()}
                facts = {
                    style_to_option[style]: {
                        "thickness": float(candidate["thickness"]),
                        "length": float(candidate["length"]),
                        "arch": float(candidate["arch"]),
                    }
                    for style, candidate in candidate_by_style.items()
                }
                contact_sheet = make_contact_sheet(image, points, order, options)
                attempts += 1
                try:
                    judgment = query_vision_model(client, args.model, contact_sheet, facts)
                except Exception as exc:
                    errors.append({
                        "image": image_path.name,
                        "repeat": str(repeat_index),
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    })
                    continue
                successes += 1
                scores_by_option = {row["option_id"]: row for row in judgment["scores"]}
                winner_option = max(
                    OPTION_IDS,
                    key=lambda option_id: scores_by_option[option_id]["compatibility_score"],
                )
                winner_style = option_to_style[winner_option]
                per_style_wins[winner_style] += 1
                winner_counts[winner_style] += 1
                session_id = f"llm_pseudo_{image_path.stem}_rep{repeat_index:02d}"
                for style, candidate in candidate_by_style.items():
                    option_id = style_to_option[style]
                    score_row = scores_by_option[option_id]
                    per_style_scores[style].append(float(score_row["compatibility_score"]))
                    all_scores[style].append(float(score_row["compatibility_score"]))
                    judgment_rows.append({
                        "case_id": case_id,
                        "session_id": session_id,
                        "image": image_path.name,
                        "source_title": source["title"],
                        "source_page": source["source_page"],
                        "license": source["license"],
                        "model": args.model,
                        "repeat": repeat_index,
                        "option_id": option_id,
                        "style": style,
                        "thickness": candidate["thickness"],
                        "length": candidate["length"],
                        "arch": candidate["arch"],
                        "compatibility_score_0_100": score_row["compatibility_score"],
                        "naturalness_score_0_100": score_row["naturalness_score"],
                        "proportion_fit_score_0_100": score_row["proportion_fit_score"],
                        "top_rank_in_this_ai_judgment": style == winner_style,
                        "ai_notes": score_row["notes"],
                        "uncertainty": judgment["uncertainty"],
                        "label_provenance": "vision_model_pseudo_label_not_customer_feedback",
                    })
                    pseudo_choice_rows.append({
                        "case_id": case_id,
                        "session_id": session_id,
                        "style": style,
                        "thickness": candidate["thickness"],
                        "length": candidate["length"],
                        "arch": candidate["arch"],
                        "selected": str(style == winner_style).lower(),
                        "face_width_height_ratio": features["face_width_height_ratio"],
                        "interocular_face_width_ratio": features["interocular_face_width_ratio"],
                        "jaw_face_width_ratio": features["jaw_face_width_ratio"],
                        "brow_eye_distance_ratio": features["brow_eye_distance_ratio"],
                        "brow_asymmetry_ratio": features["brow_asymmetry_ratio"],
                        "eye_line_tilt": features["eye_line_tilt"],
                        "native_brow_area_fraction": 0.0,
                        "native_brow_side_imbalance": 0.0,
                        "native_brow_mask_available": 0.0,
                        "native_shape_mode": 0.0,
                        "design_intent": "均衡自然",
                        "label_source": "vision_model_pseudo_label_not_customer_feedback",
                    })

            image_summaries.append({
                "case_id": case_id,
                "image": image_path.name,
                "source_title": source["title"],
                "source_page": source["source_page"],
                "author": source["author"],
                "license": source["license"],
                "status": "ok" if successes else "model_scoring_failed",
                "successful_model_judgments": successes,
                "requested_repeats": args.repeats,
                "winner_frequency_by_style": {
                    style: (per_style_wins[style] / successes if successes else None)
                    for style in CHOICE_STYLES
                },
                "mean_ai_compatibility_score_by_style": {
                    style: (float(np.mean(per_style_scores[style])) if per_style_scores[style] else None)
                    for style in CHOICE_STYLES
                },
                "interpretation": "AI judgment frequency only; not a human-preference probability",
            })
    finally:
        landmarker.close()

    write_csv(args.output_dir / "llm_candidate_judgments.csv", judgment_rows)
    write_csv(args.output_dir / "pseudo_choice_training_format.csv", pseudo_choice_rows)
    write_csv(args.output_dir / "image_level_pseudo_scores.csv", image_summaries)
    write_csv(args.output_dir / "api_errors.csv", errors)
    metadata = {
        "experiment_type": "vision_model_pseudo_label_pilot",
        "model": args.model,
        "repeats_requested_per_image": args.repeats,
        "n_images": len(images),
        "n_images_with_successful_judgments": sum(row.get("successful_model_judgments", 0) > 0 for row in image_summaries),
        "n_api_judgment_attempts": attempts,
        "n_successful_api_judgments": len(judgment_rows) // 3,
        "n_failed_api_judgments": len(errors),
        "labels_are_real_customer_preferences": False,
        "scores_are_calibrated_probabilities": False,
        "pseudo_label_method": "blind option-letter shuffling; repeated vision-model compatibility ratings on simplified overlays",
        "training_data_warning": "Do not use these pseudo-labels as calibration or external validation data and do not release a probability model trained only on them.",
        "source_manifest": SOURCES,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    successful = [row for row in image_summaries if row.get("successful_model_judgments", 0) > 0]
    summary_lines = [
        "# Vision-model eyebrow pseudo-label pilot",
        "",
        f"- Vision model: \`{args.model}\`",
        f"- Public portrait files inspected: {len(images)}",
        f"- Portraits with successful model judgments: {len(successful)}",
        f"- Successful blind judgments: {len(judgment_rows) // 3}",
        f"- Failed model judgments: {len(errors)}",
        "",
        "## Pseudo-label summary",
        "",
        "| Candidate style | Mean AI compatibility score (0–100) | AI top-rank frequency |",
        "|---|---:|---:|",
    ]
    total_winners = sum(winner_counts.values())
    for style in CHOICE_STYLES:
        mean_score = float(np.mean(all_scores[style])) if all_scores[style] else float("nan")
        winner_share = winner_counts[style] / total_winners if total_winners else float("nan")
        summary_lines.append(
            f"| {style} | {'n/a' if not np.isfinite(mean_score) else f'{mean_score:.1f}'} "
            f"| {'n/a' if not np.isfinite(winner_share) else f'{winner_share:.1%}'} |"
        )
    summary_lines += [
        "",
        "## Interpretation and limitations",
        "",
        "This pilot asks a vision-capable model to score simplified overlays on public portraits. "
        "Repeated calls and randomized A/B/C labels reduce some position and naming bias but do not "
        "make the judgements independent or representative of real customers.",
        "",
        "**AI top-rank frequency is not a customer-preference probability.** Scores are not calibrated, "
        "and the model may be systematically biased toward one style. The dataset can support debugging "
        "and a clearly labelled AI-design prior only. It must not be used as calibration or external "
        "validation data, and it must not by itself enable the Android app's customer-probability output.",
        "",
        "The previews are simplified geometric overlays generated from MediaPipe landmarks rather than "
        "photorealistic brow rendering. They can be affected by landmark errors, existing brow visibility, "
        "lighting, hair colour, camera angle, demographic coverage, prompt choice, provider-model drift and "
        "correlated repeated judgements.",
        "",
        "## Public-image licence / attribution",
        "",
    ]
    for filename, source in SOURCES.items():
        if any(row["image"] == filename for row in successful):
            summary_lines.append(
                f"- **{source['title']}** — {source['author']}; {source['license']}; "
                f"[source page]({source['source_page']}); local file \`{filename}\`."
            )
    (args.output_dir / "REPORT.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENAI_BROW_VISION_MODEL", "gpt-4o"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    metadata = run(args)
    if metadata["n_successful_api_judgments"] == 0:
        raise SystemExit("All model judgments failed. See api_errors.csv in the artifact.")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
