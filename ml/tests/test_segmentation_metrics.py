"""Unit tests for customer-balanced eyebrow segmentation metrics."""
import unittest

import numpy as np

from ml.segmentation_metrics import evaluate_segmentation_predictions


class SegmentationMetricsTests(unittest.TestCase):
    def test_perfect_predictions_score_one_at_all_levels(self):
        truth = np.array([
            [[1, 0], [0, 0]],
            [[1, 1], [0, 0]],
        ], dtype=np.float32)
        result = evaluate_segmentation_predictions(truth, truth, ["A", "B"])
        for key in (
            "micro_dice", "micro_iou", "roi_macro_dice", "roi_macro_iou",
            "customer_macro_dice", "customer_macro_iou",
            "customer_macro_precision", "customer_macro_recall",
        ):
            self.assertAlmostEqual(result[key], 1.0, places=7)
        self.assertEqual(result["n_test_rois"], 2)
        self.assertEqual(result["n_test_cases"], 2)

    def test_customer_macro_does_not_overweight_repeat_rois(self):
        truth = np.array([
            [[1, 0], [0, 0]],  # customer A, ROI 1
            [[1, 1], [0, 0]],  # customer A, ROI 2, completely missed
            [[1, 1], [0, 0]],  # customer B, ROI 1, correct
        ], dtype=np.float32)
        probability = np.array([
            [[1, 0], [0, 0]],
            [[0, 0], [0, 0]],
            [[1, 1], [0, 0]],
        ], dtype=np.float32)
        result = evaluate_segmentation_predictions(
            truth, probability, ["A", "A", "B"]
        )
        self.assertAlmostEqual(result["roi_macro_dice"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(result["customer_macro_dice"], 0.75, places=6)
        self.assertEqual(result["n_test_cases"], 2)

    def test_false_positive_and_false_negative_rates_are_reported(self):
        truth = np.array([[[1, 0], [0, 0]]], dtype=np.float32)
        probability = np.array([[[0, 1], [0, 0]]], dtype=np.float32)
        result = evaluate_segmentation_predictions(truth, probability, ["A"])
        self.assertAlmostEqual(result["micro_precision"], 0.0)
        self.assertAlmostEqual(result["micro_recall"], 0.0)
        self.assertGreater(result["micro_false_positive_rate"], 0.0)
        self.assertGreater(result["micro_false_negative_rate"], 0.0)

    def test_rejects_missing_case_ids_and_invalid_probabilities(self):
        truth = np.zeros((1, 2, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            evaluate_segmentation_predictions(truth, truth, [""])
        with self.assertRaises(ValueError):
            evaluate_segmentation_predictions(truth, truth + 2.0, ["A"])
        with self.assertRaises(ValueError):
            evaluate_segmentation_predictions(truth, truth, ["A", "B"])


if __name__ == "__main__":
    unittest.main()
