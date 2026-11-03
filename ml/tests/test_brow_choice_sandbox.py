"""Unit tests for the synthetic conditional-choice probability sandbox."""

import unittest

from ml.simulate_brow_choice_sandbox import make_synthetic_rows, run_sandbox
from ml.train_brow_choice_ranker import (
    CHOICE_FEATURE_NAMES, build_choice_sessions, expand_choice_features,
)


class BrowChoiceSandboxTests(unittest.TestCase):
    def test_choice_trainer_accepts_export_schema_with_consent_and_positions(self):
        rows, _ = make_synthetic_rows(1, "SCHEMA", 11)
        sessions = build_choice_sessions(rows, "synthetic schema test")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["x"].shape, (3, 39))

    def test_choice_trainer_rejects_unconsented_feedback(self):
        rows, _ = make_synthetic_rows(1, "NO-CONSENT", 12)
        rows[0]["consent_recorded"] = "false"
        with self.assertRaisesRegex(ValueError, "not marked as consented"):
            build_choice_sessions(rows, "unconsented test")

    def test_choice_trainer_rejects_inconsistent_candidate_display_positions(self):
        rows, _ = make_synthetic_rows(1, "BAD-POSITION", 13)
        rows[1]["display_position"] = rows[0]["display_position"]
        with self.assertRaisesRegex(ValueError, "invalid choice sessions"):
            build_choice_sessions(rows, "display-position test")

    def test_choice_trainer_rejects_mixed_intent_within_session(self):
        rows, _ = make_synthetic_rows(1, "MIXED-INTENT", 14)
        rows[1]["design_intent"] = "偏立体"
        with self.assertRaisesRegex(ValueError, "inconsistent design intent"):
            build_choice_sessions(rows, "mixed-intent test")

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
