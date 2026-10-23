"""Unit tests for the synthetic conditional-choice probability sandbox."""

import unittest

from ml.simulate_brow_choice_sandbox import run_sandbox


class BrowChoiceSandboxTests(unittest.TestCase):
    def test_choice_probabilities_sum_to_one_without_approving_synthetic_model(self):
        report = run_sandbox(seed=42)
        self.assertFalse(report["release_approved"])
        self.assertEqual(
            report["test_type"],
            "synthetic_conditional_choice_calibration_not_real_customer_evidence",
        )
        self.assertEqual(report["cases"], {"development": 300, "calibration": 90, "external": 160})
        self.assertLessEqual(report["maximum_probability_sum_error"], 1e-10)
        self.assertGreaterEqual(report["external_metrics"]["top1_accuracy"], 0.30)
        self.assertLess(report["mean_absolute_probability_error_vs_known_synthetic_generator"], 0.30)
        lo, hi = report["predicted_probability_range"]
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)


if __name__ == "__main__":
    unittest.main()
