"""Unit tests for the synthetic conditional-choice probability sandbox."""

import unittest

from ml.simulate_brow_choice_sandbox import run_sandbox
from ml.train_brow_choice_ranker import CHOICE_FEATURE_NAMES, expand_choice_features


class BrowChoiceSandboxTests(unittest.TestCase):
    def test_explicit_design_intent_changes_style_interactions(self):
        base = [0.0] * 15
        base[9] = 0.4
        base[10] = 0.25
        import numpy as np
        base_array = np.asarray(base, dtype=float)
        balanced = expand_choice_features(base_array, design_intent=0)
        soft = expand_choice_features(base_array, design_intent=1)
        defined = expand_choice_features(base_array, design_intent=2)
        self.assertEqual(len(CHOICE_FEATURE_NAMES), 39)
        self.assertEqual(len(balanced), 39)
        index = {name: CHOICE_FEATURE_NAMES.index(name) for name in CHOICE_FEATURE_NAMES}
        self.assertAlmostEqual(balanced[index["design_intent_soft_x_style_soft"]], 0.0)
        self.assertAlmostEqual(soft[index["design_intent_soft_x_style_soft"]], 0.4)
        self.assertAlmostEqual(soft[index["design_intent_soft_x_style_defined"]], 0.25)
        self.assertAlmostEqual(defined[index["design_intent_defined_x_style_soft"]], 0.4)
        self.assertAlmostEqual(defined[index["design_intent_defined_x_style_defined"]], 0.25)

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
