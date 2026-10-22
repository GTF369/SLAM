"""Robust eyebrow-segmentation metrics with equal weighting by customer.

The model is evaluated at ROI, customer and pooled-pixel levels. Pooled pixel scores
alone can be dominated by large masks and repeat photos from a small number of people,
so customer-macro scores are reported alongside them.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def _divide(numerator: float, denominator: float, *, empty_value: float = 0.0) -> float:
    return numerator / denominator if denominator else empty_value


def _binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    predicted = np.asarray(predicted, dtype=bool).reshape(-1)
    if truth.shape != predicted.shape or truth.size == 0:
        raise ValueError("Truth and prediction must have the same non-empty shape.")

    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    tn = int(np.logical_and(~truth, ~predicted).sum())
    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    return {
        "dice": _divide(2.0 * tp, float(dice_denominator), empty_value=1.0),
        "iou": _divide(float(tp), float(iou_denominator), empty_value=1.0),
        "precision": _divide(float(tp), float(tp + fp), empty_value=1.0),
        "recall": _divide(float(tp), float(tp + fn), empty_value=1.0),
        "specificity": _divide(float(tn), float(tn + fp), empty_value=1.0),
        "false_positive_rate": _divide(float(fp), float(fp + tn), empty_value=0.0),
        "false_negative_rate": _divide(float(fn), float(fn + tp), empty_value=0.0),
    }


def evaluate_segmentation_predictions(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    case_ids: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Return pooled-pixel, per-ROI macro and per-customer macro metrics.

    Inputs may have shape (N,H,W) or (N,H,W,1). case_ids[i] identifies the anonymous
    customer for ROI i. Every customer contributes once to the case-macro score even
    if they have multiple photos or left/right eyebrow ROIs.
    """
    truth_array = np.asarray(y_true)
    probability_array = np.asarray(y_probability, dtype=np.float64)
    if truth_array.shape != probability_array.shape:
        raise ValueError("y_true and y_probability must have identical shapes.")
    if truth_array.ndim < 3 or truth_array.shape[0] == 0:
        raise ValueError("Expected a non-empty batch of image masks.")
    if not np.isfinite(truth_array).all() or not np.isfinite(probability_array).all():
        raise ValueError("Masks and probabilities must not contain NaN or infinity.")
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if len(case_ids) != truth_array.shape[0]:
        raise ValueError("There must be exactly one case_id per ROI.")

    normalized_case_ids = [str(value).strip() for value in case_ids]
    if any(not value for value in normalized_case_ids):
        raise ValueError("case_ids must be non-empty anonymous customer IDs.")

    truth = truth_array >= 0.5
    predicted = probability_array >= threshold
    pooled = _binary_metrics(truth, predicted)
    roi_metrics = [_binary_metrics(truth[i], predicted[i]) for i in range(len(case_ids))]

    metrics_by_case: dict[str, list[dict[str, float]]] = defaultdict(list)
    for case_id, metrics in zip(normalized_case_ids, roi_metrics):
        metrics_by_case[case_id].append(metrics)

    metric_names = tuple(roi_metrics[0].keys())
    roi_macro = {name: float(np.mean([m[name] for m in roi_metrics])) for name in metric_names}
    case_metrics = {
        case_id: {
            name: float(np.mean([row[name] for row in rows]))
            for name in metric_names
        }
        for case_id, rows in metrics_by_case.items()
    }
    case_macro = {
        name: float(np.mean([row[name] for row in case_metrics.values()]))
        for name in metric_names
    }

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "n_test_rois": int(len(case_ids)),
        "n_test_cases": int(len(metrics_by_case)),
    }
    result.update({f"micro_{name}": value for name, value in pooled.items()})
    result.update({f"roi_macro_{name}": value for name, value in roi_macro.items()})
    result.update({f"customer_macro_{name}": value for name, value in case_macro.items()})
    return result
