#!/usr/bin/env python3
"""Split observed three-candidate feedback by anonymous customer, never by row/session."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

REQUIRED = {"case_id", "session_id", "style", "selected"}
EXPECTED_STYLES = {"natural", "soft", "defined"}


def read_and_validate(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no header.")
        rows = list(reader)
        headers = list(reader.fieldnames)
    missing = REQUIRED.difference(headers)
    if missing:
        raise ValueError("Input CSV missing required columns: " + ", ".join(sorted(missing)))
    if not rows:
        raise ValueError("Input CSV is empty.")

    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line, row in enumerate(rows, start=2):
        case_id = row["case_id"].strip()
        session_id = row["session_id"].strip()
        style = row["style"].strip().lower()
        selected = row["selected"].strip().lower()
        if not case_id or not session_id:
            raise ValueError(f"CSV row {line} needs non-empty case_id and session_id.")
        if style not in EXPECTED_STYLES:
            raise ValueError(f"CSV row {line} has unknown style {style!r}.")
        if selected not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"CSV row {line} has invalid selected value {row['selected']!r}.")
        by_session[session_id].append(row)

    for session_id, records in by_session.items():
        cases = {row["case_id"].strip() for row in records}
        styles = [row["style"].strip().lower() for row in records]
        selected_count = sum(
            row["selected"].strip().lower() in {"true", "1", "yes"} for row in records
        )
        if len(records) != 3 or len(cases) != 1 or set(styles) != EXPECTED_STYLES or len(set(styles)) != 3:
            raise ValueError(
                f"Session {session_id!r} must contain exactly natural, soft and defined "
                "once each for one case_id."
            )
        if selected_count != 1:
            raise ValueError(f"Session {session_id!r} must contain exactly one selected candidate.")

    return headers, rows


def split_rows(
    rows: list[dict[str, str]],
    *,
    seed: int = 2026,
    development_fraction: float = 0.40,
    calibration_fraction: float = 0.20,
) -> dict[str, list[dict[str, str]]]:
    by_session: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_session[row.get("session_id", "").strip()].append(row)
    for session_id, records in by_session.items():
        case_ids = {row.get("case_id", "").strip() for row in records}
        styles = [row.get("style", "").strip().lower() for row in records]
        selected_count = sum(
            row.get("selected", "").strip().lower() in {"true", "1", "yes"}
            for row in records
        )
        if (
            not session_id
            or len(records) != 3
            or len(case_ids) != 1
            or "" in case_ids
            or set(styles) != EXPECTED_STYLES
            or len(set(styles)) != 3
        ):
            raise ValueError(
                f"Session {session_id!r} must contain exactly natural, soft and defined "
                "once each for one customer."
            )
        if selected_count != 1:
            raise ValueError(f"Session {session_id!r} must contain exactly one selected candidate.")

    if not (0.0 < development_fraction < 1.0):
        raise ValueError("development_fraction must lie strictly between 0 and 1.")
    if not (0.0 < calibration_fraction < 1.0):
        raise ValueError("calibration_fraction must lie strictly between 0 and 1.")
    if development_fraction + calibration_fraction >= 1.0:
        raise ValueError("Development plus calibration fractions must leave rows for external validation.")

    rows_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case_id"].strip()].append(row)
    case_ids = sorted(rows_by_case)
    if len(case_ids) < 3:
        raise ValueError("Need at least three distinct customer cases to create non-empty splits.")

    import numpy as np

    rng = np.random.default_rng(seed)
    shuffled = list(np.asarray(case_ids, dtype=object)[rng.permutation(len(case_ids))])
    n_development = max(1, int(round(len(shuffled) * development_fraction)))
    n_calibration = max(1, int(round(len(shuffled) * calibration_fraction)))
    if n_development + n_calibration >= len(shuffled):
        n_development = max(1, int(len(shuffled) * development_fraction))
        n_calibration = max(1, int(len(shuffled) * calibration_fraction))
    if n_development + n_calibration >= len(shuffled):
        raise ValueError("Too few distinct customers for the requested split fractions.")

    ids = {
        "development": set(shuffled[:n_development]),
        "calibration": set(shuffled[n_development:n_development + n_calibration]),
        "external_validation": set(shuffled[n_development + n_calibration:]),
    }
    if set.union(*ids.values()) != set(case_ids):
        raise AssertionError("Split lost or introduced customer IDs.")
    if any(ids[a] & ids[b] for a in ids for b in ids if a < b):
        raise AssertionError("Customer IDs leaked across splits.")

    output = {
        name: [row for row in rows if row["case_id"].strip() in members]
        for name, members in ids.items()
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Exported brow_choice_feedback CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("ml/data_splits/choice"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--development-fraction", type=float, default=0.40)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    args = parser.parse_args()

    headers, rows = read_and_validate(args.csv)
    parts = split_rows(
        rows,
        seed=args.seed,
        development_fraction=args.development_fraction,
        calibration_fraction=args.calibration_fraction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in parts.items():
        path = args.output_dir / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split)

    report: dict[str, Any] = {}
    for name, split in parts.items():
        report[name] = {
            "rows": len(split),
            "sessions": len({row["session_id"].strip() for row in split}),
            "distinct_customers": len({row["case_id"].strip() for row in split}),
        }
    report["release_minimums"] = {
        "development_customers": 30,
        "calibration_customers": 50,
        "external_validation_customers": 100,
        "expected_total_minimum_at_40_20_40_split": 250,
    }
    report["warning"] = (
        "These files are a grouped random split from the collected cohort, not a prospective "
        "future-cohort validation. Final evidence should include new customers collected after "
        "model development and calibration have been frozen."
    )
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
