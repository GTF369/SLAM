#!/usr/bin/env python3
"""Summarize measured eyebrow surface/chord ratios by region.

Input is the CSV exported by the Android app. Output is descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * p
    low = int(position)
    high = min(low + 1, len(values) - 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def bootstrap_mean_ci(values: list[float], seed: int = 2026, draws: int = 5000) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.fmean(sample))
    return quantile(means, 0.025), quantile(means, 0.975)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Curvature calibration CSV exported from the app")
    parser.add_argument("--output", type=Path, default=Path("curvature_summary.csv"))
    args = parser.parse_args()

    grouped: dict[str, list[tuple[str, float, float | None]]] = defaultdict(list)
    legacy_without_case_id = False
    with args.csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"region", "arc_mm", "chord_mm", "correction_factor"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("Input must contain region, arc_mm, chord_mm, correction_factor columns.")
        has_case_id = "case_id" in (reader.fieldnames or [])
        legacy_without_case_id = not has_case_id
        for row_num, row in enumerate(reader, start=2):
            try:
                arc = float(row["arc_mm"])
                chord = float(row["chord_mm"])
                factor = float(row["correction_factor"])
            except (TypeError, ValueError):
                continue
            if chord <= 0 or arc < chord or factor <= 0:
                continue
            case_id = row.get("case_id", "").strip() if has_case_id else ""
            if not case_id:
                # For legacy files without subject grouping, every row is treated as its own
                # independent unit, and output explicitly warns that repeated measures cannot be clustered.
                case_id = f"legacy-row-{row_num}"
            try:
                raw_radius = float(row.get("estimated_radius_mm", ""))
                radius = raw_radius if math.isfinite(raw_radius) and raw_radius > 0 else None
            except (TypeError, ValueError):
                radius = None
            # Use recomputed ratios to guard against inconsistent stored factors.
            grouped[row["region"].strip()].append((case_id, arc / chord, radius))

    if not grouped:
        raise SystemExit("No valid curvature measurements found.")

    fields = [
        "region", "n_measurements", "n_cases", "mean_measurement_factor", "median_factor",
        "p10_factor", "p90_factor", "sample_min", "sample_max",
        "n_radius_measurements", "median_estimated_radius_mm", "p10_estimated_radius_mm",
        "p90_estimated_radius_mm", "bootstrap_case_mean_ci95_low",
        "bootstrap_case_mean_ci95_high", "interpretation"
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for region, rows in sorted(grouped.items()):
            values = [value for _, value, _ in rows]
            case_values: dict[str, list[float]] = defaultdict(list)
            radius_values: list[float] = []
            for case_id, value, radius in rows:
                case_values[case_id].append(value)
                if radius is not None:
                    radius_values.append(radius)
            # Average repeated measurements within each subject before resampling subjects.
            case_means = [statistics.fmean(items) for items in case_values.values()]
            low, high = bootstrap_mean_ci(case_means, seed=2026 + len(region))
            n = len(values)
            n_cases = len(case_means)
            warnings = []
            if n_cases < 30:
                warnings.append("insufficient distinct cases")
            if legacy_without_case_id:
                warnings.append("legacy file had no case_id, repeated measures could not be grouped")
            warnings.append("descriptive only; independent validation required")
            writer.writerow({
                "region": region,
                "n_measurements": n,
                "n_cases": n_cases,
                "mean_measurement_factor": f"{statistics.fmean(values):.5f}",
                "median_factor": f"{statistics.median(values):.5f}",
                "p10_factor": f"{quantile(values, 0.10):.5f}",
                "p90_factor": f"{quantile(values, 0.90):.5f}",
                "sample_min": f"{min(values):.5f}",
                "sample_max": f"{max(values):.5f}",
                "n_radius_measurements": len(radius_values),
                "median_estimated_radius_mm": f"{statistics.median(radius_values):.5f}" if radius_values else "",
                "p10_estimated_radius_mm": f"{quantile(radius_values, 0.10):.5f}" if radius_values else "",
                "p90_estimated_radius_mm": f"{quantile(radius_values, 0.90):.5f}" if radius_values else "",
                "bootstrap_case_mean_ci95_low": f"{low:.5f}",
                "bootstrap_case_mean_ci95_high": f"{high:.5f}",
                "interpretation": "; ".join(warnings),
            })
    print(f"Wrote {args.output}")
    print("P10–P90 describe measurement rows. The bootstrap confidence interval resamples anonymous cases after averaging repeated measurements within each case.")
    if legacy_without_case_id:
        print("WARNING: input lacked case_id, so old rows were treated as distinct cases. Confidence intervals may be overconfident if a person has repeated records.")
    print("These statistics are not guaranteed anatomical limits or individual prediction bounds. Do not apply a region-level factor clinically until measurement repeatability and independent validation are established.")


if __name__ == "__main__":
    main()
