#!/usr/bin/env python3
"""Locally annotate eyebrow ROI masks for brow-segmentation training.

For consented customer images named like CASE123_left.jpg and CASE123_right.jpg:
1. Draw a rectangle around one eyebrow plus margin matching the Android ROI.
2. In the crop view, outline the eyebrow hair/body region as a polygon.
3. Press s to save the crop and mask, then select another ROI or move to the next image.

Keys in crop view: left click adds a vertex, c closes/fills polygon, u undoes one vertex,
r resets mask, s saves ROI/mask, n skips ROI, q quits.
All processing is local. Never upload identifiable customer photos or masks without
specific authorization and a lawful data-handling plan.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def fit_size(image: np.ndarray, max_width: int = 1080, max_height: int = 780) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_width / max(1, w), max_height / max(1, h))
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else image.copy()
    return resized, scale


def annotate_polygon(crop: np.ndarray) -> np.ndarray | None:
    display, scale = fit_size(crop, 760, 560)
    points: list[tuple[int, int]] = []
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    window = "Brow mask editor - click vertices; c close; u undo; r reset; s save; n skip; q quit"

    def redraw():
        canvas = display.copy()
        if points:
            scaled = np.asarray([(round(x * scale), round(y * scale)) for x, y in points], dtype=np.int32)
            for p in scaled:
                cv2.circle(canvas, tuple(p), 3, (0, 255, 255), -1)
            if len(scaled) > 1:
                cv2.polylines(canvas, [scaled], False, (0, 255, 255), 2)
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        overlay = cv2.resize(overlay, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST)
        active = overlay[:, :, 0] > 0
        canvas[active] = (0.45 * canvas[active] + 0.55 * np.array([40, 210, 130])).astype(np.uint8)
        cv2.imshow(window, canvas)

    def callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((round(x / scale), round(y / scale)))
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    redraw()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == ord("c"):
            if len(points) >= 3:
                poly = np.asarray(points, dtype=np.int32)
                cv2.fillPoly(mask, [poly], 255)
                points.clear()
                redraw()
        elif key == ord("u"):
            if points:
                points.pop()
                redraw()
        elif key == ord("r"):
            mask[:] = 0
            points.clear()
            redraw()
        elif key == ord("s"):
            if int(mask.max()) == 0 and len(points) >= 3:
                cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
            cv2.destroyWindow(window)
            return mask if int(mask.max()) > 0 else None
        elif key == ord("n"):
            cv2.destroyWindow(window)
            return None
        elif key == ord("q"):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt
    cv2.destroyWindow(window)


def select_roi(image: np.ndarray) -> tuple[int, int, int, int] | None:
    display, scale = fit_size(image)
    selected: list[tuple[int, int]] = []
    drawing = {"active": False}

    def callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected[:] = [(x, y), (x, y)]
            drawing["active"] = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
            selected[1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["active"]:
            selected[1] = (x, y)
            drawing["active"] = False

    window = "Select one eyebrow ROI then press Enter; n next photo; q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, callback)
    while True:
        canvas = display.copy()
        if len(selected) == 2:
            cv2.rectangle(canvas, selected[0], selected[1], (40, 220, 140), 2)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (10, 13):
            cv2.destroyWindow(window)
            if len(selected) != 2:
                return None
            x0, x1 = sorted([selected[0][0], selected[1][0]])
            y0, y1 = sorted([selected[0][1], selected[1][1]])
            if x1 - x0 < 24 or y1 - y0 < 12:
                return None
            return (
                max(0, round(x0 / scale)),
                max(0, round(y0 / scale)),
                min(image.shape[1], round(x1 / scale)),
                min(image.shape[0], round(y1 / scale)),
            )
        if key == ord("n"):
            cv2.destroyWindow(window)
            return None
        if key == ord("q"):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Folder with consented source photos")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for images/, masks/, manifest.csv")
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit("No images found in --images.")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    manifest_path = args.output / "manifest.csv"
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    existing_images: set[str] = set()
    existing_masks: set[str] = set()
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as existing_handle:
            existing_reader = csv.DictReader(existing_handle)
            if not existing_reader.fieldnames or not {"image", "mask", "case_id"}.issubset(existing_reader.fieldnames):
                raise SystemExit("Existing manifest.csv has an invalid header. Back it up and create a new output directory.")
            for existing_row in existing_reader:
                existing_images.add(existing_row.get("image", ""))
                existing_masks.add(existing_row.get("mask", ""))
    roi_counts: dict[str, int] = {}
    try:
        with manifest_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image", "mask", "case_id"])
            if write_header:
                writer.writeheader()
            for path in paths:
                source = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if source is None:
                    print(f"Skipping unreadable image: {path}")
                    continue
                # Use names such as CASE123_left.jpg and CASE123_right.jpg so
                # the same anonymous customer remains in one train/test split group.
                case_id = path.stem.split("_")[0]
                while True:
                    try:
                        bounds = select_roi(source)
                    except KeyboardInterrupt:
                        raise
                    if bounds is None:
                        break
                    x0, y0, x1, y1 = bounds
                    crop = source[y0:y1, x0:x1].copy()
                    mask = annotate_polygon(crop)
                    if mask is None:
                        continue
                    number = roi_counts.get(case_id, 0) + 1
                    while True:
                        stem = f"{case_id}_{path.stem}_roi{number:02d}"
                        image_name = f"images/{stem}.png"
                        mask_name = f"masks/{stem}.png"
                        if (
                            image_name not in existing_images
                            and mask_name not in existing_masks
                            and not (args.output / image_name).exists()
                            and not (args.output / mask_name).exists()
                        ):
                            break
                        number += 1
                    roi_counts[case_id] = number
                    existing_images.add(image_name)
                    existing_masks.add(mask_name)
                    if not cv2.imwrite(str(args.output / image_name), crop):
                        print(f"Unable to save image crop for {path}")
                        continue
                    if not cv2.imwrite(str(args.output / mask_name), mask):
                        print(f"Unable to save mask for {path}")
                        continue
                    writer.writerow({"image": image_name, "mask": mask_name, "case_id": case_id})
                    handle.flush()
                    print(f"Saved {image_name} and {mask_name}")
    except KeyboardInterrupt:
        print("Annotation stopped by user.")
    finally:
        cv2.destroyAllWindows()
    print(f"Manifest: {manifest_path}")
    print("Review every mask before training. Dataset quality matters more than raw image count.")


if __name__ == "__main__":
    main()
