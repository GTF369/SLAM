#!/usr/bin/env python3
"""Split preference feedback into disjoint customer-level development/calibration/test CSVs.

This is a reproducible convenience split, not a substitute for a prospective external study.
All rows with the same anonymous case_id always go to the same output file.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def split_by_case(rows: list[dict[str, str]], seed: int = 2026, development_ratio: float = 0.70,
                  calibration_ratio: float = 0.15) -> dict[str, list[dict[str, str]]]:
    if not rows:
        raise ValueError("Input CSV is empty.")
    headers = set(rows[0])
    required = {"case_id", "rating"}
    if not required.issubset(headers):
        raise ValueError("Input CSV must contain case_id and rating columns.")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for i, row in enumerate(rows, start=2):
        case_id = row.get("case_id", "").strip()
        if not case_id:
            raise ValueError(f"Row {i} has an empty case_id.")
        grouped[case_id].append(row)
    if not (0.5 <= development_ratio < 0.9):
        raise ValueError("development_ratio must be in [0.5, 0.9).")
    if not (0.05 <= calibration_ratio <= 0.25):
        raise ValueError("calibration_ratio must be in [0.05, 0.25].")
    if development_ratio + calibration_ratio >= 1.0:
        raise ValueError("The remaining external validation ratio must be positive.")
    case_ids = list(grouped)
    random.Random(seed).shuffle(case_ids)
    n = len(case_ids)
    n_dev = int(n * development_ratio)
    n_cal = int(n * calibration_ratio)
    if n_dev < 1 or n_cal < 1 or n - n_dev - n_cal < 1:
        raise ValueError("Not enough distinct customer cases to create three non-empty splits.")
    assignments = {
        "development": case_ids[:n_dev],
        "calibration": case_ids[n_dev:n_dev + n_cal],
        "external_validation": case_ids[n_dev + n_cal:],
    }
    outputs = {
        name: [row for cid in ids for row in grouped[cid]]
        for name, ids in assignments.items()
    }
    for name, split_rows in outputs.items():
        labels = {1 if float(row["rating"]) >= 4 else 0 for row in split_rows}
        if len(labels) < 2:
            raise ValueError(
                f"{name} split does not contain both rating classes. Collect more varied feedback "
                "or create a reviewed split that still keeps each customer wholly in one set."
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Combined app preference feedback CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/data_splits"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--development-ratio", type=float, default=0.70)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    args = parser.parse_args()
    with args.csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no header.")
        headers = reader.fieldnames
        rows = list(reader)
    try:
        splits = split_by_case(rows, args.seed, args.development_ratio, args.calibration_ratio)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, split_rows in splits.items():
        path = args.output_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(split_rows)
        cases = {row["case_id"] for row in split_rows}
        report[name] = {
            "path": str(path),
            "rows": len(split_rows),
            "cases": len(cases),
            "positive_rows": sum(float(row["rating"]) >= 4 for row in split_rows),
        }
    ids = {name: {row["case_id"] for row in data} for name, data in splits.items()}
    assert ids["development"].isdisjoint(ids["calibration"])
    assert ids["development"].isdisjoint(ids["external_validation"])
    assert ids["calibration"].isdisjoint(ids["external_validation"])
    print("Customer-grouped split report (development holdout is still required):")
    for name, item in report.items():
        print(f"- {name}: {item['cases']} cases, {item['rows']} rows, {item['positive_rows']} positive rows -> {item['path']}")
    print("Warning: random held-out customers from the same collection are not the same as a prospective future cohort.")


if __name__ == "__main__":
    main()
