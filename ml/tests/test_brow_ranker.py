import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_brow_ranker import (
    build_matrix,
    choose_group_split,
    expected_calibration_error,
    calibration_metrics,
)


def synthetic_rows(case_count=12):
    """Synthetic rows for algorithm tests only, not model training or product evidence."""
    rows = []
    for i in range(case_count):
        for style, rating in (("natural", 5 if i % 2 == 0 else 2), ("soft", 2 if i % 2 == 0 else 5)):
            rows.append({
                "case_id": f"SYNTHETIC-{i:03d}",
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
            })
    return rows


class BrowRankerTests(unittest.TestCase):
    def test_build_matrix_labels_and_groups(self):
        x, y, groups = build_matrix(synthetic_rows())
        self.assertEqual(x.shape, (24, 11))
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertEqual(len(np.unique(groups)), 12)

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

    def test_calibration_metrics_include_prevalence_baseline(self):
        y = np.asarray([0, 0, 1, 1])
        probs = np.asarray([0.1, 0.2, 0.8, 0.9])
        metrics = calibration_metrics(y, probs)
        self.assertLess(metrics["brier_score"], metrics["prevalence_baseline_brier_score"])
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
