"""Tests for case-balanced stencil-transfer error analysis."""
import unittest

from ml.analyze_stencil_transfer import summarize_transfer_rows


def row(case, skin="10.0", printed="10.5", endpoint="0.5", profile="printer-A", region="左眉身"):
    return {
        "case_id": case,
        "printer_profile": profile,
        "region": region,
        "curvature_method": "无曲率修正",
        "skin_arc_mm": skin,
        "printed_template_path_mm": printed,
        "endpoint_alignment_error_mm": endpoint,
    }


class StencilTransferAnalysisTests(unittest.TestCase):
    def test_repeated_measurements_are_averaged_within_customer(self):
        rows = [
            row("A", printed="10.4", endpoint="0.4"),
            row("A", printed="10.6", endpoint="0.6"),
            row("B", printed="9.8", endpoint="0.8"),
            row("C", printed="10.1", endpoint="0.2"),
        ]
        summary = summarize_transfer_rows(rows, min_cases=3)[0]
        self.assertEqual(summary["n_measurements"], 4)
        self.assertEqual(summary["n_cases"], 3)
        self.assertAlmostEqual(summary["mean_signed_length_error_mm"], (0.5 - 0.2 + 0.1) / 3)
        self.assertAlmostEqual(summary["mean_endpoint_alignment_error_mm"], (0.5 + 0.8 + 0.2) / 3)
        self.assertEqual(summary["screening_status"], "descriptive_only_thresholds_not_supplied")

    def test_screening_requires_customer_count_and_both_error_thresholds(self):
        rows = [
            row("A", printed="10.4", endpoint="0.4"),
            row("A", printed="10.6", endpoint="0.6"),
            row("B", printed="9.8", endpoint="0.8"),
            row("C", printed="10.1", endpoint="0.2"),
        ]
        summary = summarize_transfer_rows(
            rows,
            min_cases=3,
            max_p90_abs_length_error_mm=1.0,
            max_p90_endpoint_error_mm=1.0,
            min_endpoint_coverage=0.8,
        )[0]
        self.assertEqual(summary["screening_status"], "passes_screening_human_review_required")

        too_few = summarize_transfer_rows(
            rows,
            min_cases=30,
            max_p90_abs_length_error_mm=1.0,
            max_p90_endpoint_error_mm=1.0,
        )[0]
        self.assertEqual(too_few["screening_status"], "insufficient_independent_cases")

    def test_profiles_and_regions_are_summarized_separately(self):
        rows = [
            row("A", profile="printer-A", region="左眉头"),
            row("B", profile="printer-B", region="左眉头", printed="12.0"),
            row("C", profile="printer-A", region="右眉尾", printed="9.0"),
        ]
        summaries = summarize_transfer_rows(rows, min_cases=2)
        self.assertEqual(len(summaries), 3)
        self.assertEqual(
            {(x["printer_profile"], x["region"]) for x in summaries},
            {("printer-A", "左眉头"), ("printer-B", "左眉头"), ("printer-A", "右眉尾")},
        )

    def test_invalid_physical_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_transfer_rows([row("A", skin="0")])
        with self.assertRaises(ValueError):
            summarize_transfer_rows([row("A", endpoint="-1")])
        with self.assertRaises(ValueError):
            summarize_transfer_rows([row("", printed="10")])


if __name__ == "__main__":
    unittest.main()
