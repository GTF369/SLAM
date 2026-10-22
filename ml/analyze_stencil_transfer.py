#!/usr/bin/env python3
"""Summarize stencil transfer error by printer profile and eyebrow region.

This tool is diagnostic. It never writes print correction coefficients or changes app
settings. Repeated rows from one anonymous customer are averaged within that customer
before population summaries and uncertainty intervals are computed.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED = {
    "case_id", "printer_profile", "region", "curvature_method", "skin_arc_mm",
    "printed_template_path_mm", "endpoint_alignment_error_mm",
}


def quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_mean_ci(
    values: list[float], seed: int = 2026, draws: int = 3000
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean, resampling independent customer summaries."""
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(draws)
    ]
    return quantile(means, 0.025), quantile(means, 0.975)


def _finite_positive(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be a finite positive number")
    return value


def summarize_transfer_rows(
    rows: list[dict[str, str]],
    *,
    min_cases: int = 30,
    max_p90_abs_length_error_mm: float | None = None,
    max_p90_endpoint_error_mm: float | None = None,
    min_endpoint_coverage: float = 0.80,
) -> list[dict[str, Any]]:
    """Return one summary per printer profile, eyebrow region and curvature method.

    Threshold-based screening is available only when both physical error tolerances
    are explicitly supplied. A pass means eligible for human review, never auto-release.
    """
    if min_cases < 2:
        raise ValueError("min_cases must be at least 2")
    if not 0.0 <= min_endpoint_coverage <= 1.0:
        raise ValueError("min_endpoint_coverage must be between 0 and 1")
    for name, value in (
        ("max_p90_abs_length_error_mm", max_p90_abs_length_error_mm),
        ("max_p90_endpoint_error_mm", max_p90_endpoint_error_mm),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")

    grouped: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    for line_no, row in enumerate(rows, start=2):
        missing = REQUIRED.difference(row.keys())
        if missing:
            raise ValueError("Missing required transfer columns: " + ", ".join(sorted(missing)))
        case_id = row.get("case_id", "").strip()
        profile = row.get("printer_profile", "").strip()
        region = row.get("region", "").strip()
        method = row.get("curvature_method", "").strip()
        if not case_id or not profile or not region or not method:
            raise ValueError(
                f"row {line_no} has an empty case_id, printer_profile, region, or curvature_method"
            )
        try:
            skin_arc = _finite_positive(row, "skin_arc_mm")
            printed_path = _finite_positive(row, "printed_template_path_mm")
            signed_error = printed_path - skin_arc
            raw_endpoint = row.get("endpoint_alignment_error_mm", "").strip()
            endpoint_error: float | None = None
            if raw_endpoint:
                endpoint_error = float(raw_endpoint)
                if not math.isfinite(endpoint_error) or endpoint_error < 0:
                    raise ValueError("endpoint_alignment_error_mm must be finite and non-negative")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"row {line_no} contains invalid physical measurements: {exc}") from exc
        grouped[(profile, region, method, case_id)].append({
            "signed_length_error_mm": signed_error,
            "absolute_length_error_mm": abs(signed_error),
            "endpoint_alignment_error_mm": (
                endpoint_error if endpoint_error is not None else float("nan")
            ),
        })

    # Keep no-correction and measured-curvature samples separate. Pooling them would
    # hide whether a correction method improves or worsens physical transfer error.
    by_profile_region: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (profile, region, method, case_id), repeated in grouped.items():
        length_errors = [r["signed_length_error_mm"] for r in repeated]
        absolute_errors = [r["absolute_length_error_mm"] for r in repeated]
        endpoint_errors = [
            r["endpoint_alignment_error_mm"] for r in repeated
            if math.isfinite(r["endpoint_alignment_error_mm"])
        ]
        by_profile_region[(profile, region, method)].append({
            "case_id": case_id,
            "signed_length_error_mm": statistics.fmean(length_errors),
            "absolute_length_error_mm": statistics.fmean(absolute_errors),
            "endpoint_alignment_error_mm": (
                statistics.fmean(endpoint_errors) if endpoint_errors else float("nan")
            ),
            "repeat_measurements": len(repeated),
        })

    summaries: list[dict[str, Any]] = []
    thresholds_supplied = (
        max_p90_abs_length_error_mm is not None
        and max_p90_endpoint_error_mm is not None
    )
    for (profile, region, method), cases in sorted(by_profile_region.items()):
        signed = [r["signed_length_error_mm"] for r in cases]
        absolute = [r["absolute_length_error_mm"] for r in cases]
        endpoint = [
            r["endpoint_alignment_error_mm"] for r in cases
            if math.isfinite(r["endpoint_alignment_error_mm"])
        ]
        endpoint_coverage = len(endpoint) / len(cases)
        low, high = bootstrap_mean_ci(
            signed,
            seed=2026 + sum(ord(char) for char in profile + region),
        )
        if len(cases) < min_cases:
            gate = "insufficient_independent_cases"
        elif not thresholds_supplied:
            gate = "descriptive_only_thresholds_not_supplied"
        elif endpoint_coverage < min_endpoint_coverage:
            gate = "insufficient_endpoint_measurements"
        elif (
            quantile(absolute, 0.90) <= float(max_p90_abs_length_error_mm)
            and quantile(endpoint, 0.90) <= float(max_p90_endpoint_error_mm)
        ):
            gate = "passes_screening_human_review_required"
        else:
            gate = "fails_screening"

        summaries.append({
            "printer_profile": profile,
            "region": region,
            "curvature_method": method,
            "n_measurements": sum(r["repeat_measurements"] for r in cases),
            "n_cases": len(cases),
            "mean_signed_length_error_mm": statistics.fmean(signed),
            "mean_absolute_length_error_mm": statistics.fmean(absolute),
            "median_absolute_length_error_mm": statistics.median(absolute),
            "p90_absolute_length_error_mm": quantile(absolute, 0.90),
            "maximum_absolute_length_error_mm": max(absolute),
            "mean_signed_length_error_ci95_low_mm": low,
            "mean_signed_length_error_ci95_high_mm": high,
            "n_cases_with_endpoint_error": len(endpoint),
            "endpoint_measurement_coverage": endpoint_coverage,
            "mean_endpoint_alignment_error_mm": statistics.fmean(endpoint) if endpoint else "",
            "p90_endpoint_alignment_error_mm": quantile(endpoint, 0.90) if endpoint else "",
            "screening_status": gate,
            "interpretation": (
                "Diagnostic only. Do not enable automatic correction from this table. "
                "Review independent cases, repeatability, printer profile stability, "
                "measurement protocol, and real-skin transfer before any controlled release."
            ),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV exported by the Android transfer-validation recorder")
    parser.add_argument("--output", type=Path, default=Path("stencil_transfer_summary.csv"))
    parser.add_argument("--min-cases", type=int, default=30)
    parser.add_argument("--max-p90-abs-length-error-mm", type=float)
    parser.add_argument("--max-p90-endpoint-error-mm", type=float)
    parser.add_argument("--min-endpoint-coverage", type=float, default=0.80)
    args = parser.parse_args()

    if (args.max_p90_abs_length_error_mm is None) != (args.max_p90_endpoint_error_mm is None):
        parser.error("Supply both physical error tolerances to run screening; omit both for descriptive summaries.")
    with args.csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise SystemExit("Input must contain: " + ", ".join(sorted(REQUIRED)))
        rows = list(reader)
    if not rows:
        raise SystemExit("Input contains no transfer-validation rows.")

    summaries = summarize_transfer_rows(
        rows,
        min_cases=args.min_cases,
        max_p90_abs_length_error_mm=args.max_p90_abs_length_error_mm,
        max_p90_endpoint_error_mm=args.max_p90_endpoint_error_mm,
        min_endpoint_coverage=args.min_endpoint_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {args.output} ({len(summaries)} profile-region summaries)")
    print("Results are descriptive or a configurable engineering screen. A screen pass only triggers human review; it never activates app correction coefficients.")


if __name__ == "__main__":
    main()
