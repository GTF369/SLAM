import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_brow_ranker import (
    build_matrix,
    expand_fit_features,
    FEATURE_NAMES,
    choose_group_split,
    expected_calibration_error,
    calibration_metrics,
    geometric_reference_scores,
    clustered_auc_comparison_ci,
    read_rows,
)


def synthetic_rows(case_count=12):
    """Synthetic rows for algorithm tests only, not model training or product evidence."""
    rows = []
    for i in range(case_count):
        for style, rating in (("natural", 5 if i % 2 == 0 else 2), ("soft", 2 if i % 2 == 0 else 5)):
            rows.append({
                "case_id": f"SYNTHETIC-{i:03d}",
                "feedback_schema_version": "3",
                "design_intent": ("偏柔和" if i % 3 == 0 else "偏立体" if i % 3 == 1 else "均衡自然"),
                "style": style,
                "thickness": "1.0",
                "length": "1.0",
                "arch": "1.0",
                "rating": str(rating),
                "face_width_height_ratio": str(0.7 + (i % 5) * 0.03),
                "interocular_face_width_ratio": "0.38",
                "jaw_face_width_ratio": "0.80",
                "brow_eye_distance_ratio": "0.04",
                "brow_asymmetry_ratio": "0.01",
                "eye_line_tilt": "0.0",
                "native_brow_area_fraction": "0.12",
                "native_brow_side_imbalance": "0.02",
                "native_brow_mask_available": "1",
                "native_shape_mode": "1",
            })
    return rows


class BrowRankerTests(unittest.TestCase):
    def test_fit_training_rejects_legacy_general_satisfaction_rows(self):
        row = synthetic_rows(1)[0]
        row["feedback_schema_version"] = "2"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "explicit eyebrow-fit ratings"):
                read_rows(path)

    def test_build_matrix_labels_and_groups(self):
        x, y, groups = build_matrix(synthetic_rows())
        self.assertEqual(x.shape, (24, 19))
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertEqual(len(np.unique(groups)), 12)

    def test_explicit_intent_candidate_interactions_are_encoded(self):
        base = [0.7, 0.38, 0.80, 0.04, 0.01, 0.0, 0.82, 0.94, 0.72,
                1.0, 0.0, 0.12, 0.02, 1.0, 0.0]
        soft_intent = expand_fit_features(base, "偏柔和")
        defined_intent = expand_fit_features(base[:9] + [0.0, 1.0] + base[11:], "偏立体")
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self.assertEqual(len(soft_intent), 19)
        self.assertAlmostEqual(soft_intent[index["design_intent_soft_x_style_soft"]], 1.0)
        self.assertAlmostEqual(soft_intent[index["design_intent_soft_x_style_defined"]], 0.0)
        self.assertAlmostEqual(defined_intent[index["design_intent_defined_x_style_soft"]], 0.0)
        self.assertAlmostEqual(defined_intent[index["design_intent_defined_x_style_defined"]], 1.0)

    def test_group_split_has_no_case_leakage(self):
        x, y, groups = build_matrix(synthetic_rows(30))
        train, test = choose_group_split(x, y, groups, seed=13)
        self.assertTrue(set(groups[train]).isdisjoint(set(groups[test])))
        self.assertEqual(set(np.unique(y[train])), {0, 1})
        self.assertEqual(set(np.unique(y[test])), {0, 1})

    def test_expected_calibration_error_is_zero_for_perfect_binary_predictions(self):
        y = np.asarray([0, 0, 1, 1])
        probs = np.asarray([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(expected_calibration_error(y, probs), 0.0)

    def test_geometric_baseline_returns_expected_family_order_scores(self):
        x, _, _ = build_matrix(synthetic_rows(4))
        scores = geometric_reference_scores(x)
        self.assertEqual(scores.shape, (8,))
        self.assertTrue(np.isfinite(scores).all())

    def test_cluster_bootstrap_compares_candidate_to_rule_baseline(self):
        rows = synthetic_rows(50)
        x, y, groups = build_matrix(rows)
        # A deterministic score that follows the high/low rating labels.
        candidate = np.where(y == 1, 0.9, 0.1)
        baseline = geometric_reference_scores(x)
        comparison = clustered_auc_comparison_ci(
            y, candidate, baseline, groups, seed=23, draws=300
        )
        self.assertGreater(comparison["auc_ci95_low"], 0.9)
        self.assertGreater(comparison["auc_gain_over_geometry_ci95_low"], 0.0)
        self.assertGreaterEqual(comparison["valid_bootstrap_draws"], 150)

    def test_calibration_metrics_include_prevalence_baseline(self):
        y = np.asarray([0, 0, 1, 1])
        probs = np.asarray([0.1, 0.2, 0.8, 0.9])
        metrics = calibration_metrics(y, probs)
        self.assertLess(metrics["brier_score"], metrics["prevalence_baseline_brier_score"])
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
