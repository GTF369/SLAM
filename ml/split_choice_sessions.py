#!/usr/bin/env python3
"""Split real three-candidate eyebrow choices by customer, not by row or session.

Default proportions are 60% development, 15% calibration, 25% external validation.
All sessions for a pseudonymous customer remain in one split. The script never creates
synthetic labels and refuses incomplete or inconsistent choice sessions.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from train_brow_choice_ranker import build_choice_sessions

STYLE_NAMES = ("natural", "soft", "defined")


def split_choice_rows(
    rows: list[dict[str, str]],
    *,
    seed: int = 2026,
    development_ratio: float = 0.60,
    calibration_ratio: float = 0.15,
    attempts: int = 5000,
) -> dict[str, list[dict[str, str]]]:
    if not rows:
        raise ValueError("Input CSV is empty.")
    if not (0.50 <= development_ratio < 0.85):
        raise ValueError("development_ratio must be in [0.50, 0.85).")
    if not (0.05 <= calibration_ratio < 0.30):
        raise ValueError("calibration_ratio must be in [0.05, 0.30].")
    if development_ratio + calibration_ratio >= 1.0:
        raise ValueError("The external validation ratio must be positive.")

    # Reuse the exact same schema, feature, and one-of-each-style checks as model training.
    sessions = build_choice_sessions(rows, "Input CSV")
    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    sessions_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case_id"].strip()].append(row)
    for session in sessions:
        sessions_by_case[session["case_id"]].append(session)

    case_ids = list(rows_by_case)
    n_cases = len(case_ids)
    n_dev = int(n_cases * development_ratio)
    n_cal = int(n_cases * calibration_ratio)
    n_external = n_cases - n_dev - n_cal
    if min(n_dev, n_cal, n_external) < 1:
        raise ValueError("Not enough distinct customers to make three non-empty splits.")

    # Try deterministic customer-level shuffles until every split has examples of all
    # three observed choices. If the preference distribution is too skewed, fail loudly.
    for attempt in range(attempts):
        shuffled = case_ids.copy()
        random.Random(seed + attempt).shuffle(shuffled)
        assignments = {
            "development": set(shuffled[:n_dev]),
            "calibration": set(shuffled[n_dev:n_dev + n_cal]),
            "external_validation": set(shuffled[n_dev + n_cal:]),
        }
        valid = True
        for ids in assignments.values():
            labels = {
                int(session["choice"])
                for case_id in ids
                for session in sessions_by_case[case_id]
            }
            if labels != {0, 1, 2}:
                valid = False
                break
        if not valid:
            continue
        output = {
            name: [row for case_id in case_ids if case_id in ids for row in rows_by_case[case_id]]
            for name, ids in assignments.items()
        }
        id_sets = {name: set(ids) for name, ids in assignments.items()}
        assert id_sets["development"].isdisjoint(id_sets["calibration"])
        assert id_sets["development"].isdisjoint(id_sets["external_validation"])
        assert id_sets["calibration"].isdisjoint(id_sets["external_validation"])
        return output

    raise ValueError(
        f"Could not create three customer-disjoint splits with all three chosen styles after "
        f"{attempts} attempts. Collect more varied, consented customer sessions or review "
        "the split manually without placing a customer in multiple sets."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV exported by the app's three-candidate feedback form")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/choice_splits"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--development-ratio", type=float, default=0.60)
    parser.add_argument("--calibration-ratio", type=float, default=0.15)
    args = parser.parse_args()

    with args.csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no header.")
        headers = reader.fieldnames
        rows = list(reader)
    try:
        splits = split_choice_rows(
            rows,
            seed=args.seed,
            development_ratio=args.development_ratio,
            calibration_ratio=args.calibration_ratio,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    for name, split_rows in splits.items():
        path = args.output_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(split_rows)
        sessions = build_choice_sessions(split_rows, name)
        report[name] = {
            "path": str(path),
            "customers": len({row["case_id"] for row in split_rows}),
            "sessions": len(sessions),
            "rows": len(split_rows),
            "selected_styles": {
                STYLE_NAMES[style]: sum(session["choice"] == style for session in sessions)
                for style in range(3)
            },
        }
    print(json.dumps({
        "warning": (
            "Random held-out customers are not a prospective future cohort. "
            "Do not approve model probabilities unless external gates pass and a human reviewer signs off."
        ),
        "splits": report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
