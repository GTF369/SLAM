package com.example.facepattern

/**
 * Bounded, explainable geometry prior for initial brow candidates.
 * This is not a learned preference model and makes no claim about personality.
 */
internal object BrowGeometryAdapter {
    fun personalizeCandidates(
        candidates: List<BrowParams>,
        widthHeightRatio: Float,
        interocularFaceWidthRatio: Float,
        jawFaceWidthRatio: Float,
        browEyeDistanceRatio: Float,
        browAsymmetryRatio: Float,
        eyeLineTilt: Float,
        nativeBrowAreaFraction: Float,
        nativeBrowSideImbalance: Float,
        nativeBrowMaskAvailable: Float
    ): List<BrowParams> {
        fun safe(v: Float, fallback: Float = 0f) = if (v.isFinite()) v else fallback
        fun centered(v: Float, center: Float, span: Float) =
            ((safe(v) - center) / span).coerceIn(-1f, 1f)

        val face = centered(widthHeightRatio, 0.72f, 0.18f)
        val eyes = centered(interocularFaceWidthRatio, 0.40f, 0.08f)
        val jaw = centered(jawFaceWidthRatio, 0.75f, 0.15f)
        val browGap = centered(browEyeDistanceRatio, 0.06f, 0.04f)
        val tilt = (safe(eyeLineTilt) * 4f).coerceIn(-1f, 1f)
        val asymmetry = (safe(browAsymmetryRatio) / 0.08f).coerceIn(0f, 1f)
        val reviewed = safe(nativeBrowMaskAvailable) >= 0.5f
        val maskThin = if (reviewed) {
            ((0.13f - safe(nativeBrowAreaFraction, 0.13f)) / 0.13f).coerceIn(-1f, 1f)
        } else 0f
        val maskImbalance = if (reviewed) safe(nativeBrowSideImbalance).coerceIn(0f, 1f) else 0f

        return candidates.map { base ->
            base.copy(
                thickness = (base.thickness - face * 0.04f + jaw * 0.025f + maskThin * 0.06f)
                    .coerceIn(0.55f, 1.45f),
                length = (base.length + face * 0.025f + eyes * 0.02f + jaw * 0.015f)
                    .coerceIn(0.80f, 1.20f),
                arch = (base.arch + face * 0.045f + eyes * 0.025f + browGap * 0.035f +
                    tilt * 0.025f - asymmetry * 0.025f - maskImbalance * 0.025f)
                    .coerceIn(0.55f, 1.45f)
            )
        }
    }

    /**
     * Apply a small, reversible style preference to the candidate set.
     * 0 = balanced, 1 = softer, 2 = more defined. No personality is inferred.
     */
    fun applyStyleIntent(candidates: List<BrowParams>, preferredStyle: Int): List<BrowParams> {
        val (thicknessShift, lengthShift, archShift) = when (preferredStyle.coerceIn(0, 2)) {
            1 -> Triple(-0.025f, -0.01f, -0.065f)
            2 -> Triple(0.035f, 0.015f, 0.075f)
            else -> Triple(0f, 0f, 0f)
        }
        return candidates.map { base ->
            base.copy(
                thickness = (base.thickness + thicknessShift).coerceIn(0.55f, 1.45f),
                length = (base.length + lengthShift).coerceIn(0.80f, 1.20f),
                arch = (base.arch + archShift).coerceIn(0.55f, 1.45f)
            )
        }
    }
}
