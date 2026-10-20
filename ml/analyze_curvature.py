#!/usr/bin/env python3
"""Summarize measured eyebrow surface/chord ratios by region.

Input is the CSV exported by the Android app. Output is descriptive only.
"""
from __future__ import annotations

import argparse
import csv
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

    grouped: dict[str, list[float]] = defaultdict(list)
    with args.csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"region", "arc_mm", "chord_mm", "correction_factor"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise SystemExit("Input must contain region, arc_mm, chord_mm, correction_factor columns.")
        for row in reader:
            try:
                arc = float(row["arc_mm"])
                chord = float(row["chord_mm"])
                factor = float(row["correction_factor"])
            except (TypeError, ValueError):
                continue
            if chord <= 0 or arc < chord or factor <= 0:
                continue
            # Use recomputed ratios to guard against inconsistent stored factors.
            grouped[row["region"].strip()].append(arc / chord)

    if not grouped:
        raise SystemExit("No valid curvature measurements found.")

    fields = [
        "region", "n", "mean_factor", "median_factor", "p10_factor", "p90_factor",
        "sample_min", "sample_max", "bootstrap_mean_ci95_low", "bootstrap_mean_ci95_high",
        "interpretation"
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for region, values in sorted(grouped.items()):
            low, high = bootstrap_mean_ci(values, seed=2026 + len(region))
            n = len(values)
            writer.writerow({
                "region": region,
                "n": n,
                "mean_factor": f"{statistics.fmean(values):.5f}",
                "median_factor": f"{statistics.median(values):.5f}",
                "p10_factor": f"{quantile(values, 0.10):.5f}",
                "p90_factor": f"{quantile(values, 0.90):.5f}",
                "sample_min": f"{min(values):.5f}",
                "sample_max": f"{max(values):.5f}",
                "bootstrap_mean_ci95_low": f"{low:.5f}",
                "bootstrap_mean_ci95_high": f"{high:.5f}",
                "interpretation": (
                    "preliminary, insufficient sample" if n < 30
                    else "descriptive sample interval, requires independent validation"
                ),
            })
    print(f"Wrote {args.output}")
    print("P10–P90 are sample quantiles, not guaranteed anatomical limits. Bootstrap intervals describe uncertainty in the sample mean, not individual prediction bounds.")
    print("Do not apply a region-level factor to clinical use until the measurement protocol and an independent validation study are complete.")


if __name__ == "__main__":
    main()
