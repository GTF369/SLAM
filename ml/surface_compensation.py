"""Geometry helpers for measured 2D-to-surface brow length compensation.

This module does not reconstruct an unknown human face from a single image. It maps a
measured skin-surface arc length to the corresponding projected segment length after
scale calibration from a known physical interpupillary distance. Synthetic helpers
provide a deterministic software test using known 3D surfaces.
"""
from __future__ import annotations

from math import exp, hypot, isfinite, sqrt
from typing import Sequence

MIN_COMPENSATION_FACTOR = 0.75
MAX_COMPENSATION_FACTOR = 1.60


def _point(values: Sequence[float], expected: int, label: str) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(f"{label} must have {expected} coordinates")
    result = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{label} contains NaN or infinity")
    return result


def polyline_length_2d(points: Sequence[Sequence[float]]) -> float:
    """Length of a projected image/template polyline, in its input units."""
    if len(points) < 2:
        return 0.0
    values = [_point(point, 2, "2D point") for point in points]
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(values, values[1:]))


def polyline_length_3d(points: Sequence[Sequence[float]]) -> float:
    """Length of a measured 3D surface polyline in the coordinates' length unit."""
    if len(points) < 2:
        return 0.0
    values = [_point(point, 3, "3D point") for point in points]
    return sum(
        sqrt(sum((b[axis] - a[axis]) ** 2 for axis in range(3)))
        for a, b in zip(values, values[1:])
    )


def compensation_factor_from_measurement(
    surface_arc_mm: float,
    projected_segment_px: float,
    pupil_distance_px: float,
    interpupillary_distance_mm: float,
    *,
    enforce_supported_range: bool = True,
) -> float:
    """Compute measured surface arc / calibrated 2D projected segment length.

    This intentionally does not use the surface-arc / spatial-chord ratio as the
    final factor. The projected segment is first converted to millimetres using the
    physical interpupillary distance measured for the current photograph.
    """
    values = (
        surface_arc_mm,
        projected_segment_px,
        pupil_distance_px,
        interpupillary_distance_mm,
    )
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("All measurement inputs must be finite")
    if surface_arc_mm <= 0 or projected_segment_px < 2 or pupil_distance_px < 5:
        raise ValueError("Surface arc and calibrated photo segment must be positive and usable")
    if not 35 <= interpupillary_distance_mm <= 90:
        raise ValueError("Measured interpupillary distance must be in [35, 90] mm")
    projected_segment_mm = projected_segment_px / pupil_distance_px * interpupillary_distance_mm
    if not isfinite(projected_segment_mm) or projected_segment_mm <= 0:
        raise ValueError("Could not calibrate the projected segment length")
    factor = surface_arc_mm / projected_segment_mm
    if not isfinite(factor):
        raise ValueError("Compensation factor is not finite")
    if enforce_supported_range and not MIN_COMPENSATION_FACTOR <= factor <= MAX_COMPENSATION_FACTOR:
        raise ValueError(
            f"Compensation factor {factor:.4f} is outside the supported "
            f"[{MIN_COMPENSATION_FACTOR}, {MAX_COMPENSATION_FACTOR}] range"
        )
    return factor


def apply_segment_factors(
    points: Sequence[Sequence[float]],
    factors: Sequence[float],
    *,
    recenter: bool = True,
) -> list[tuple[float, float]]:
    """Scale each 2D segment vector, matching the app's landmark mapping.

    Factors must contain one value per adjacent pair. Recentring keeps the midpoint
    of the two path endpoints fixed while allowing the required span to change.
    """
    values = [_point(point, 2, "2D point") for point in points]
    if len(values) < 2 or len(factors) != len(values) - 1:
        raise ValueError("Need one finite factor per adjacent 2D segment")
    normalized_factors = [float(value) for value in factors]
    if not all(
        isfinite(value) and MIN_COMPENSATION_FACTOR <= value <= MAX_COMPENSATION_FACTOR
        for value in normalized_factors
    ):
        raise ValueError("Every factor must be within the supported range")
    adjusted = [values[0]]
    for index, factor in enumerate(normalized_factors):
        ax, ay = values[index]
        bx, by = values[index + 1]
        px, py = adjusted[-1]
        adjusted.append((px + (bx - ax) * factor, py + (by - ay) * factor))
    if recenter:
        old_mid = ((values[0][0] + values[-1][0]) / 2, (values[0][1] + values[-1][1]) / 2)
        new_mid = ((adjusted[0][0] + adjusted[-1][0]) / 2, (adjusted[0][1] + adjusted[-1][1]) / 2)
        adjusted = [(x + old_mid[0] - new_mid[0], y + old_mid[1] - new_mid[1]) for x, y in adjusted]
    return adjusted


def perspective_project(
    points: Sequence[Sequence[float]],
    focal_length_px: float,
    *,
    principal_x: float = 0.0,
    principal_y: float = 0.0,
) -> list[tuple[float, float]]:
    """Project camera-space (x, y, z) points with a pinhole camera."""
    if not isfinite(focal_length_px) or focal_length_px <= 0:
        raise ValueError("Focal length must be positive and finite")
    projected = []
    for point in points:
        x, y, z = _point(point, 3, "camera-space point")
        if z <= 0:
            raise ValueError("Every projected point must be in front of the camera")
        projected.append((
            principal_x + focal_length_px * x / z,
            principal_y + focal_length_px * y / z,
        ))
    return projected


def synthetic_surface_case(
    *,
    name: str,
    frontal_bulge_mm: float,
    asymmetry_mm: float,
    camera_distance_mm: float = 600.0,
    focal_length_px: float = 1800.0,
    interpupillary_distance_mm: float = 63.0,
    samples_per_region: int = 40,
) -> dict:
    """Create a known synthetic brow path on a curved face patch.

    This is a math/implementation fixture, not a human anatomy model or customer-data
    substitute. Four regions are sampled densely in 3D so their arc length is known.
    """
    if samples_per_region < 4 or camera_distance_mm <= 0:
        raise ValueError("Invalid synthetic camera or region sample count")
    half_span = 30.0
    region_count = 4
    total_steps = region_count * samples_per_region

    def surface_point(u: float) -> tuple[float, float, float]:
        x = half_span * u
        y = -4.0 + 2.5 * u * u
        dome = frontal_bulge_mm * max(0.0, 1.0 - (x / 38.0) ** 2) * exp(-(y / 24.0) ** 2)
        depth = dome + asymmetry_mm * u
        return x, y, camera_distance_mm - depth

    dense_surface = [surface_point(-1.0 + 2.0 * i / total_steps) for i in range(total_steps + 1)]
    dense_image = perspective_project(dense_surface, focal_length_px)

    def project_pupil(x: float) -> tuple[float, float, float]:
        y = 0.0
        u = x / half_span
        dome = frontal_bulge_mm * max(0.0, 1.0 - (x / 38.0) ** 2) * exp(-(y / 24.0) ** 2)
        depth = dome + asymmetry_mm * u
        return x, y, camera_distance_mm - depth

    pupil_image = perspective_project(
        [project_pupil(-interpupillary_distance_mm / 2.0),
         project_pupil(interpupillary_distance_mm / 2.0)],
        focal_length_px,
    )
    pupil_distance_px = hypot(
        pupil_image[1][0] - pupil_image[0][0],
        pupil_image[1][1] - pupil_image[0][1],
    )

    anchor_indices = [i * samples_per_region for i in range(region_count + 1)]
    projected_anchors = [dense_image[index] for index in anchor_indices]
    segments = []
    raw_total_mm = 0.0
    corrected_total_mm = 0.0
    target_total_mm = 0.0
    factors = []

    for region_index in range(region_count):
        start = anchor_indices[region_index]
        end = anchor_indices[region_index + 1]
        surface_arc = polyline_length_3d(dense_surface[start:end + 1])
        projected_chord_px = polyline_length_2d(projected_anchors[region_index:region_index + 2])
        projected_chord_mm = projected_chord_px * interpupillary_distance_mm / pupil_distance_px
        factor = compensation_factor_from_measurement(
            surface_arc, projected_chord_px, pupil_distance_px, interpupillary_distance_mm
        )
        corrected_length_mm = projected_chord_mm * factor
        raw_total_mm += projected_chord_mm
        corrected_total_mm += corrected_length_mm
        target_total_mm += surface_arc
        factors.append(factor)
        segments.append({
            "region_index": region_index,
            "surface_arc_mm": surface_arc,
            "projected_chord_mm": projected_chord_mm,
            "factor": factor,
            "corrected_length_mm": corrected_length_mm,
            "absolute_error_mm": abs(corrected_length_mm - surface_arc),
        })

    adjusted = apply_segment_factors(projected_anchors, factors)
    endpoint_shift_px = hypot(
        adjusted[-1][0] - projected_anchors[-1][0],
        adjusted[-1][1] - projected_anchors[-1][1],
    )
    return {
        "name": name,
        "camera_distance_mm": camera_distance_mm,
        "frontal_bulge_mm": frontal_bulge_mm,
        "asymmetry_mm": asymmetry_mm,
        "interpupillary_distance_px": pupil_distance_px,
        "segments": segments,
        "raw_projected_total_mm": raw_total_mm,
        "surface_arc_total_mm": target_total_mm,
        "compensated_total_mm": corrected_total_mm,
        "raw_absolute_total_length_error_mm": abs(raw_total_mm - target_total_mm),
        "compensated_absolute_total_length_error_mm": abs(corrected_total_mm - target_total_mm),
        "absolute_total_length_error_mm": abs(corrected_total_mm - target_total_mm),
        "relative_total_error_reduction": (
            1.0 - abs(corrected_total_mm - target_total_mm) / abs(raw_total_mm - target_total_mm)
            if abs(raw_total_mm - target_total_mm) > 1e-12
            else 0.0
        ),
        "max_segment_error_mm": max(segment["absolute_error_mm"] for segment in segments),
        "endpoint_shift_px": endpoint_shift_px,
        "note": (
            "Synthetic geometry only. Demonstrates the calibration mathematics, not "
            "real face reconstruction, printing accuracy, or skin-transfer validation."
        ),
    }
