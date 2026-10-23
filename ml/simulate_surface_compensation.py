#!/usr/bin/env python3
"""Run deterministic synthetic 3D-brow surface compensation experiments.

A PASS shows formula consistency for known synthetic geometry only. It does not establish
real customer accuracy, skin-transfer accuracy, or replacement of physical measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from surface_compensation import synthetic_surface_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("surface_compensation_sandbox_report.json"))
    args = parser.parse_args()
    scenarios = [
        ("flat frontal surface", 0.0, 0.0, 650.0),
        ("mild convexity", 2.0, 0.0, 620.0),
        ("moderate convexity", 5.0, 0.0, 600.0),
        ("strong convexity", 8.0, 0.0, 600.0),
        ("asymmetric convexity", 5.0, 1.5, 580.0),
        ("opposite asymmetry", 5.0, -1.5, 580.0),
    ]
    cases = [
        synthetic_surface_case(
            name=name, frontal_bulge_mm=bulge, asymmetry_mm=asymmetry,
            camera_distance_mm=distance,
        )
        for name, bulge, asymmetry, distance in scenarios
    ]
    report = {
        "test_type": "synthetic_known_geometry_not_customer_validation",
        "camera_model": "pinhole projection with known 3D surface coordinates",
        "cases": cases,
        "summary": {
            "n_cases": len(cases),
            "n_regions": sum(len(case["segments"]) for case in cases),
            "max_segment_error_mm": max(case["max_segment_error_mm"] for case in cases),
            "max_total_length_error_mm": max(case["absolute_total_length_error_mm"] for case in cases),
            "all_factors_in_supported_range": all(
                0.75 <= segment["factor"] <= 1.60
                for case in cases for segment in case["segments"]
            ),
        },
        "limitations": [
            "Surface coordinates are synthetic and fully known by construction.",
            "This proves numerical consistency of the mapping formula only.",
            "Real users still require measured surface arcs, same-endpoint photo landmarks, measured interpupillary distance, independent physical/3D reference, target-device checks, and real stencil-transfer validation.",
            "Do not describe this report as validated clinical, aesthetic, or physical-transfer accuracy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Detailed synthetic report: {args.output}")


if __name__ == "__main__":
    main()
