import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_stencil_transfer import read_measurements, case_level_rows, summarize


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "case_id", "printer_profile", "region", "surface_arc_mm", "template_path_mm",
        "endpoint_error_mm", "calibration_box_mm", "correction_method", "timestamp",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows(case_count: int = 3) -> list[dict[str, str]]:
    rows = []
    for i in range(case_count):
        base = 20.0 + i
        rows.append({
            "case_id": f"CASE-{i}",
            "printer_profile": "printer-A-A4-100pct",
            "region": "左眉身",
            "surface_arc_mm": str(base),
            "template_path_mm": str(base + (0.2 if i % 2 == 0 else -0.1)),
            "endpoint_error_mm": str(0.3 + 0.1 * i),
            "calibration_box_mm": "20.1",
            "correction_method": "无补偿",
            "timestamp": str(1000 + i),
        })
    return rows


class TransferAuditTests(unittest.TestCase):
    def test_repeated_rows_are_collapsed_to_customer_level(self):
        rows = sample_rows(2)
        repeated = dict(rows[0])
        repeated["surface_arc_mm"] = "22"
        repeated["template_path_mm"] = "22.2"
        rows.append(repeated)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transfer.csv"
            write_rows(path, rows)
            parsed = read_measurements(path)
        cases = case_level_rows(parsed)
        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0]["repeat_measurements"], 2)
        self.assertAlmostEqual(cases[0]["signed_length_error_mm"], 0.2, places=6)

    def test_summary_has_physical_error_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transfer.csv"
            write_rows(path, sample_rows(4))
            summary = summarize(read_measurements(path), min_cases=3)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_cases"], 4)
        self.assertGreater(summary[0]["p90_absolute_length_error_mm"], 0)
        self.assertAlmostEqual(summary[0]["p90_absolute_scale_error_mm"], 0.1)

    def test_screen_fails_when_endpoint_measurements_are_missing(self):
        rows = sample_rows(3)
        for row in rows:
            row["endpoint_error_mm"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transfer.csv"
            write_rows(path, rows)
            summary = summarize(
                read_measurements(path),
                min_cases=3,
                max_p90_abs_length_error_mm=1.0,
                max_p90_endpoint_error_mm=1.0,
            )
        self.assertFalse(summary[0]["screen_gates"]["endpoint_error_p90_within_limit"])
        self.assertFalse(summary[0]["screen_passed"])

    def test_bad_calibration_box_is_rejected(self):
        rows = sample_rows(1)
        rows[0]["calibration_box_mm"] = "30"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "transfer.csv"
            write_rows(path, rows)
            with self.assertRaisesRegex(ValueError, "calibration_box_mm"):
                read_measurements(path)


if __name__ == "__main__":
    unittest.main()
