"""Unit checks for the explicitly synthetic ranker/calibration sandbox."""
import unittest

from ml.simulate_brow_ranker_sandbox import run_sandbox


class BrowRankerSandboxTests(unittest.TestCase):
    def test_sandbox_calibrates_without_claiming_real_release(self):
        report = run_sandbox(seed=42)
        self.assertFalse(report["release_approved"])
        self.assertEqual(report["test_type"], "synthetic_ranking_and_calibration_pipeline_not_real_customer_evidence")
        self.assertEqual(report["cases"], {"development": 300, "calibration": 90, "external": 160})
        metrics = report["external_metrics"]
        self.assertGreaterEqual(metrics["roc_auc"], 0.60)
        self.assertLess(metrics["brier_score"], metrics["prevalence_baseline_brier_score"])
        self.assertGreaterEqual(report["output_probability_range"][0], 0.0)
        self.assertLessEqual(report["output_probability_range"][1], 1.0)


if __name__ == "__main__":
    unittest.main()
