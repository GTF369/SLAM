package com.example.facepattern

import android.content.res.AssetManager
import org.json.JSONObject
import kotlin.math.exp

/**
 * Loads only a reviewed, calibrated three-candidate model. Absence or an unapproved
 * artifact deliberately returns null so the UI cannot turn a heuristic into a probability.
 */
internal data class BrowChoiceCandidate(
    val style: String,
    val thickness: Double,
    val length: Double,
    val arch: Double,
    val nativeBrowAreaFraction: Double,
    val nativeBrowSideImbalance: Double,
    val nativeBrowMaskAvailable: Double,
    val nativeShapeMode: Double
)

internal class BrowChoiceRanker private constructor(
    private val means: DoubleArray,
    private val scales: DoubleArray,
    private val coefficients: DoubleArray,
    private val temperature: Double
) {
    fun probabilities(
        faceFeatures: DoubleArray,
        candidates: List<BrowChoiceCandidate>,
        designIntent: Int
    ): List<Double> {
        require(faceFeatures.size == 6) { "Six face-geometry features are required." }
        require(candidates.size == 3) { "Exactly three candidates are required." }
        require(designIntent in 0..2) { "Unknown stated design intent." }
        val expectedStyles = listOf("natural", "soft", "defined")
        require(candidates.map { it.style } == expectedStyles) {
            "Candidates must be ordered natural, soft, defined."
        }

        val logits = candidates.map { candidate ->
            val base = doubleArrayOf(
                faceFeatures[0], faceFeatures[1], faceFeatures[2],
                faceFeatures[3], faceFeatures[4], faceFeatures[5],
                candidate.thickness, candidate.length, candidate.arch,
                if (candidate.style == "soft") 1.0 else 0.0,
                if (candidate.style == "defined") 1.0 else 0.0,
                candidate.nativeBrowAreaFraction,
                candidate.nativeBrowSideImbalance,
                candidate.nativeBrowMaskAvailable,
                candidate.nativeShapeMode
            )
            val expanded = expandFeatures(base, designIntent)
            var score = 0.0
            for (i in expanded.indices) {
                score += ((expanded[i] - means[i]) / scales[i]) * coefficients[i]
            }
            score
        }
        val maxLogit = logits.maxOrNull() ?: return emptyList()
        val values = logits.map { exp(((it - maxLogit) / temperature).coerceIn(-700.0, 0.0)) }
        val total = values.sum()
        if (!total.isFinite() || total <= 0.0) return emptyList()
        return values.map { it / total }
    }

    private fun expandFeatures(base: DoubleArray, designIntent: Int): DoubleArray {
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
        // Keep this interaction order identical to ml/train_brow_choice_ranker.py.
        val interactions = doubleArrayOf(
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
        val intentSoft = if (designIntent == 1) 1.0 else 0.0
        val intentDefined = if (designIntent == 2) 1.0 else 0.0
        val intentInteractions = doubleArrayOf(
            intentSoft * styleSoft,
            intentSoft * styleDefined,
            intentDefined * styleSoft,
            intentDefined * styleDefined
        )
        return base + interactions + intentInteractions
    }

    companion object {
        private const val ASSET_NAME = "brow_choice_ranker.json"
        private const val EXPECTED_FEATURES = 39

        fun load(assets: AssetManager): BrowChoiceRanker? {
            return try {
                val payload = assets.open(ASSET_NAME).bufferedReader(Charsets.UTF_8).use { it.readText() }
                val json = JSONObject(payload)
                if (json.optInt("model_version", 0) < 2) return null
                if (json.optString("model_type") != "conditional_choice") return null
                if (!json.optBoolean("release_approved", false)) return null
                if (json.optString("calibration_status") != "temperature_scaling") return null
                if (json.optJSONObject("external_validation")?.optBoolean("passed_release_gates", false) != true) return null

                fun readArray(name: String): DoubleArray {
                    val array = json.getJSONArray(name)
                    return DoubleArray(array.length()) { index -> array.getDouble(index) }
                }

                val means = readArray("means")
                val scales = readArray("scales")
                val coefficients = readArray("coefficients")
                val temperature = json.getDouble("temperature")
                if (means.size != EXPECTED_FEATURES ||
                    scales.size != EXPECTED_FEATURES ||
                    coefficients.size != EXPECTED_FEATURES ||
                    means.any { !it.isFinite() } ||
                    scales.any { !it.isFinite() || it <= 0.0 } ||
                    coefficients.any { !it.isFinite() } ||
                    !temperature.isFinite() || temperature <= 0.0
                ) return null

                BrowChoiceRanker(means, scales, coefficients, temperature)
            } catch (_: Exception) {
                null
            }
        }
    }
}
