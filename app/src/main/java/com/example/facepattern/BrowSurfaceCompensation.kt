package com.example.facepattern

/**
 * Small, testable geometry core for measured photo-to-surface length mapping.
 *
 * This does not infer a face's hidden 3D shape. It maps a physically measured
 * surface-arc length to the same segment's photo-projected length, scaled by the
 * measured interpupillary distance. A result outside the permitted engineering
 * range is rejected rather than silently clamped.
 */
internal object BrowSurfaceCompensation {
    const val MIN_FACTOR = 0.75f
    const val MAX_FACTOR = 1.60f
    const val MIN_IPD_MM = 35f
    const val MAX_IPD_MM = 90f
    const val MIN_PROJECTED_SEGMENT_PX = 2f
    const val MIN_PUPIL_DISTANCE_PX = 5f

    fun projectedSegmentMillimetres(
        projectedSegmentPixels: Float,
        pupilDistancePixels: Float,
        measuredInterpupillaryDistanceMm: Float
    ): Float? {
        if (!projectedSegmentPixels.isFinite() ||
            !pupilDistancePixels.isFinite() ||
            !measuredInterpupillaryDistanceMm.isFinite()
        ) return null
        if (projectedSegmentPixels < MIN_PROJECTED_SEGMENT_PX ||
            pupilDistancePixels < MIN_PUPIL_DISTANCE_PX ||
            measuredInterpupillaryDistanceMm !in MIN_IPD_MM..MAX_IPD_MM
        ) return null
        val result = projectedSegmentPixels / pupilDistancePixels * measuredInterpupillaryDistanceMm
        return result.takeIf { it.isFinite() && it > 0f }
    }

    fun factorFromMeasurements(
        measuredSurfaceArcMm: Float,
        projectedSegmentPixels: Float,
        pupilDistancePixels: Float,
        measuredInterpupillaryDistanceMm: Float
    ): Float? {
        if (!measuredSurfaceArcMm.isFinite() || measuredSurfaceArcMm <= 0f) return null
        val projectedMm = projectedSegmentMillimetres(
            projectedSegmentPixels,
            pupilDistancePixels,
            measuredInterpupillaryDistanceMm
        ) ?: return null
        val factor = measuredSurfaceArcMm / projectedMm
        return factor.takeIf { it.isFinite() && it in MIN_FACTOR..MAX_FACTOR }
    }

    fun validateCompleteFactors(
        factors: Map<String, Float>,
        requiredRegions: Set<String>
    ): Boolean {
        return factors.keys.containsAll(requiredRegions) &&
            requiredRegions.all { region ->
                val factor = factors[region]
                factor != null && factor.isFinite() && factor in MIN_FACTOR..MAX_FACTOR
            }
    }
}
