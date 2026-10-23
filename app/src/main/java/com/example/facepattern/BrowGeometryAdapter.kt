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
}
