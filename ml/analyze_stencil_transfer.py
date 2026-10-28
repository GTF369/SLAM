#!/usr/bin/env python3
"""Audit measured brow-stencil transfer errors by customer, printer profile, region, and correction method.

This script summarizes physical measurements. It does not infer 3D anatomy, create correction
coefficients, or authorize a print correction. Set acceptance thresholds before examining results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_COLUMNS = {
    "case_id",
    "printer_profile",
    "region",
    "surface_arc_mm",
    "template_path_mm",
    "endpoint_error_mm",
    "calibration_box_mm",
    "correction_method",
}


def _number(value: str | None, field: str, row_number: int, *, optional: bool = False) -> float | None:
    if value is None or value.strip() == "":
        if optional:
            return None
        raise ValueError(f"CSV row {row_number}: missing {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"CSV row {row_number}: {field} must be finite")
    return result


def read_measurements(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            profile = (row.get("printer_profile") or "").strip()
            region = (row.get("region") or "").strip()
            method = (row.get("correction_method") or "").strip()
            if not case_id or not profile or not region or not method:
                raise ValueError(
                    f"CSV row {row_number}: case_id, printer_profile, region, and correction_method are required"
                )
            surface_arc = _number(row.get("surface_arc_mm"), "surface_arc_mm", row_number)
            template_path = _number(row.get("template_path_mm"), "template_path_mm", row_number)
            calibration_box = _number(row.get("calibration_box_mm"), "calibration_box_mm", row_number)
            endpoint_error = _number(
                row.get("endpoint_error_mm"), "endpoint_error_mm", row_number, optional=True
            )
            assert surface_arc is not None and template_path is not None and calibration_box is not None
            if surface_arc <= 0 or template_path <= 0:
                raise ValueError(f"CSV row {row_number}: arc and template path lengths must be positive")
            if not 16.0 <= calibration_box <= 24.0:
                raise ValueError(
                    f"CSV row {row_number}: calibration_box_mm must be between 16 and 24 mm; "
                    "inspect the printed calibration square"
                )
            if endpoint_error is not None and endpoint_error < 0:
                raise ValueError(f"CSV row {row_number}: endpoint_error_mm cannot be negative")
            rows.append({
                "case_id": case_id,
                "printer_profile": profile,
                "region": region,
                "correction_method": method,
                "surface_arc_mm": surface_arc,
                "template_path_mm": template_path,
                "endpoint_error_mm": endpoint_error,
                "calibration_box_mm": calibration_box,
            })
    if not rows:
        raise ValueError(f"CSV contains no measurement rows: {path}")
    return rows


def case_level_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average repeated measurements within a case before calculating population statistics."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["printer_profile"],
            row["region"],
            row["correction_method"],
            row["case_id"],
        )
        grouped[key].append(row)

    collapsed: list[dict[str, Any]] = []
    for (profile, region, method, case_id), items in sorted(grouped.items()):
        surface = float(np.mean([x["surface_arc_mm"] for x in items]))
        template = float(np.mean([x["template_path_mm"] for x in items]))
        endpoints = [x["endpoint_error_mm"] for x in items if x["endpoint_error_mm"] is not None]
        box = float(np.mean([x["calibration_box_mm"] for x in items]))
        signed_error = template - surface
        collapsed.append({
            "printer_profile": profile,
            "region": region,
            "correction_method": method,
            "case_id": case_id,
            "repeat_measurements": len(items),
            "surface_arc_mm": surface,
            "template_path_mm": template,
            "signed_length_error_mm": signed_error,
            "absolute_length_error_mm": abs(signed_error),
            "relative_length_error_pct": 100.0 * signed_error / surface,
            "endpoint_error_mm": float(np.mean(endpoints)) if endpoints else None,
            "calibration_box_mm": box,
            "absolute_scale_error_mm": abs(box - 20.0),
        })
    return collapsed


def bootstrap_mean_ci(values: list[float], *, seed: int = 2026, draws: int = 2000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    data = np.asarray(values, dtype=float)
    if len(data) == 1:
        value = float(data[0])
        return (value, value)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(draws, len(data)))
    means = data[indices].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def summarize(
    rows: list[dict[str, Any]],
    *,
    min_cases: int = 30,
    max_p90_abs_length_error_mm: float | None = None,
    max_p90_endpoint_error_mm: float | None = None,
    max_scale_error_mm: float | None = None,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    cases = case_level_rows(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        grouped[(row["printer_profile"], row["region"], row["correction_method"])].append(row)

    screen_requested = (
        max_p90_abs_length_error_mm is not None
        and max_p90_endpoint_error_mm is not None
    )
    result: list[dict[str, Any]] = []
    for index, ((profile, region, method), items) in enumerate(sorted(grouped.items())):
        signed = [float(x["signed_length_error_mm"]) for x in items]
        absolute = [float(x["absolute_length_error_mm"]) for x in items]
        relative = [float(x["relative_length_error_pct"]) for x in items]
        endpoints = [float(x["endpoint_error_mm"]) for x in items if x["endpoint_error_mm"] is not None]
        scale_errors = [float(x["absolute_scale_error_mm"]) for x in items]
        ci_low, ci_high = bootstrap_mean_ci(signed, seed=seed + index)
        p90_abs = float(np.quantile(absolute, 0.90))
        p90_endpoint = float(np.quantile(endpoints, 0.90)) if endpoints else None
        p90_scale = float(np.quantile(scale_errors, 0.90))
        enough_cases = len(items) >= min_cases
        gates: dict[str, bool] = {}
        if screen_requested:
            gates["minimum_independent_cases"] = enough_cases
            gates["length_error_p90_within_limit"] = p90_abs <= float(max_p90_abs_length_error_mm)
            gates["endpoint_error_p90_within_limit"] = (
                p90_endpoint is not None
                and p90_endpoint <= float(max_p90_endpoint_error_mm)
            )
            if max_scale_error_mm is not None:
                gates["scale_error_p90_within_limit"] = p90_scale <= max_scale_error_mm
            screen_passed: bool | None = all(gates.values())
        else:
            screen_passed = None
        result.append({
            "printer_profile": profile,
            "region": region,
            "correction_method": method,
            "n_cases": len(items),
            "n_raw_measurements": sum(int(x["repeat_measurements"]) for x in items),
            "mean_signed_length_error_mm": float(np.mean(signed)),
            "median_signed_length_error_mm": float(np.median(signed)),
            "p90_absolute_length_error_mm": p90_abs,
            "max_absolute_length_error_mm": float(max(absolute)),
            "mean_relative_length_error_pct": float(np.mean(relative)),
            "p90_endpoint_error_mm": p90_endpoint,
            "endpoint_measurement_cases": len(endpoints),
            "p90_absolute_scale_error_mm": p90_scale,
            "mean_signed_error_ci95_low_mm": ci_low,
            "mean_signed_error_ci95_high_mm": ci_high,
            "screen_gates": gates,
            "screen_passed": screen_passed,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Transfer-validation CSV exported by the Android app")
    parser.add_argument("--output", type=Path, default=Path("stencil_transfer_summary.csv"))
    parser.add_argument("--min-cases", type=int, default=30)
    parser.add_argument("--max-p90-abs-length-error-mm", type=float)
    parser.add_argument("--max-p90-endpoint-error-mm", type=float)
    parser.add_argument("--max-scale-error-mm", type=float)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.min_cases < 1:
        parser.error("--min-cases must be at least 1")
    for name in (
        "max_p90_abs_length_error_mm",
        "max_p90_endpoint_error_mm",
        "max_scale_error_mm",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if (args.max_p90_abs_length_error_mm is None) != (args.max_p90_endpoint_error_mm is None):
        parser.error("For screening, provide both --max-p90-abs-length-error-mm and --max-p90-endpoint-error-mm")

    raw = read_measurements(args.csv)
    summaries = summarize(
        raw,
        min_cases=args.min_cases,
        max_p90_abs_length_error_mm=args.max_p90_abs_length_error_mm,
        max_p90_endpoint_error_mm=args.max_p90_endpoint_error_mm,
        max_scale_error_mm=args.max_scale_error_mm,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if summaries:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
    print(json.dumps({
        "n_raw_measurements": len(raw),
        "n_case_level_rows": len(case_level_rows(raw)),
        "summary_csv": str(args.output),
        "summaries": summaries,
        "note": (
            "screen_passed is only an engineering screen defined by user-provided thresholds. "
            "It is not clinical validation and does not automatically enable a correction."
        ),
    }, indent=2, allow_nan=False))
    print(f"Summary written to: {args.output}")


if __name__ == "__main__":
    main()
