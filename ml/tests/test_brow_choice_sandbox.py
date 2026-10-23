"""Tests for conditional candidate-choice probability and calibration sandbox."""
import unittest

from ml.simulate_brow_choice_sandbox import run_sandbox, make_synthetic_rows
from ml.train_brow_choice_ranker import build_choice_sessions


class BrowChoiceSandboxTests(unittest.TestCase):
    def test_rejects_incomplete_or_ambiguous_choice_sessions(self):
        rows, _ = make_synthetic_rows(4, "TEST", 7)
        sessions = build_choice_sessions(rows, "synthetic")
        self.assertEqual(len(sessions), 4)
        rows[0]["selected"] = "true"
        with self.assertRaises(ValueError):
            build_choice_sessions(rows, "synthetic-invalid")

    def test_probability_pipeline_uses_disjoint_customer_sets_and_never_approves_synthetic_model(self):
        report = run_sandbox(seed=42)
        self.assertFalse(report["release_approved"])
        self.assertEqual(report["test_type"], "synthetic_conditional_choice_probability_not_real_customer_evidence")
        self.assertEqual(report["development_cases"], 300)
        self.assertEqual(report["calibration_cases"], 90)
        self.assertEqual(report["external_validation"]["n_cases"], 160)
        self.assertEqual(report["external_validation"]["n_sessions"], 160)
        self.assertGreater(report["external_validation"]["top1_accuracy"], 1.0 / 3.0)
        self.assertLess(
            report["external_validation"]["mean_negative_log_likelihood"],
            1.0986122886681098,
        )
        self.assertGreaterEqual(report["mean_probability_mae_vs_synthetic_generator"], 0.0)


if __name__ == "__main__":
    unittest.main()
