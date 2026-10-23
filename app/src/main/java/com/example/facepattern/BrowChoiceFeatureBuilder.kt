package com.example.facepattern

/**
 * Feature expansion for the conditional-choice probability model.
 *
 * The first 15 values must match BrowRecommendationEngine's candidate feature vector.
 * Interaction features allow a face measurement to change the utility difference between
 * candidate designs. Face-only additive terms would cancel when probabilities are normalized
 * with softmax within a choice set, producing the same ranking for every face.
 */
internal object BrowChoiceFeatureBuilder {
    val featureOrder: List<String> = listOf(
        "face_width_height_ratio",
        "interocular_face_width_ratio",
        "jaw_face_width_ratio",
        "brow_eye_distance_ratio",
        "brow_asymmetry_ratio",
        "eye_line_tilt",
        "thickness",
        "length",
        "arch",
        "style_soft",
        "style_defined",
        "native_brow_area_fraction",
        "native_brow_side_imbalance",
        "native_brow_mask_available",
        "native_shape_mode",
        "face_width_height_x_thickness",
        "face_width_height_x_length",
        "face_width_height_x_arch",
        "interocular_ratio_x_thickness",
        "interocular_ratio_x_arch",
        "jaw_ratio_x_arch",
        "brow_eye_distance_x_thickness",
        "brow_eye_distance_x_arch",
        "brow_asymmetry_x_arch",
        "face_width_height_x_style_soft",
        "face_width_height_x_style_defined",
        "brow_eye_distance_x_style_soft",
        "brow_eye_distance_x_style_defined",
        "brow_asymmetry_x_style_soft",
        "brow_asymmetry_x_style_defined",
        "eye_line_tilt_x_style_soft",
        "eye_line_tilt_x_style_defined",
        "native_brow_area_fraction_x_thickness",
        "native_brow_side_imbalance_x_thickness",
        "native_brow_side_imbalance_x_arch"
    )

    fun expand(base: FloatArray): FloatArray {
        require(base.size == 15) { "Expected 15 base candidate features, got ${base.size}" }
        require(base.all { it.isFinite() }) { "Candidate features must be finite" }
        val faceWidth = base[0]
        val interocular = base[1]
        val jaw = base[2]
        val browEye = base[3]
        val asymmetry = base[4]
        val eyeTilt = base[5]
        val thickness = base[6]
        val length = base[7]
        val arch = base[8]
        val styleSoft = base[9]
        val styleDefined = base[10]
        val nativeArea = base[11]
        val sideImbalance = base[12]
        val interactions = floatArrayOf(
            faceWidth * thickness,
            faceWidth * length,
            faceWidth * arch,
            interocular * thickness,
            interocular * arch,
            jaw * arch,
            browEye * thickness,
            browEye * arch,
            asymmetry * arch,
            faceWidth * styleSoft,
            faceWidth * styleDefined,
            browEye * styleSoft,
            browEye * styleDefined,
            asymmetry * styleSoft,
            asymmetry * styleDefined,
            eyeTilt * styleSoft,
            eyeTilt * styleDefined,
            nativeArea * thickness,
            sideImbalance * thickness,
            sideImbalance * arch
        )
        return base + interactions
    }
}
