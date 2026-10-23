package com.example.facepattern

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/**
 * Research-only population aesthetic prior trained on SCUT-FBP5500 human ratings.
 *
 * Output is an averaged facial-attractiveness score in [1, 5], NOT an individual's
 * probability of choosing an eyebrow style. The original data/model terms apply.
 */
internal object BrowAestheticPrior {
    private const val ASSET_NAME = "brow_aesthetic_resnet50.onnx"
    private const val SIZE = 224
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    @Volatile private var attempted = false
    private var environment: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String? = null

    fun isAvailable(context: Context): Boolean =
        ASSET_NAME in (context.assets.list("")?.toSet().orEmpty())

    fun sourceLabel(): String =
        "已启用 SCUT-FBP5500 人类评分训练的群体审美参考模型。每个候选显示整脸审美评分预测（1–5 分），不代表顾客选择概率或眉形客观适配概率。模型反映特定数据集的群体偏好，仅供个人非商业研究参考。"

    @Synchronized
    private fun load(context: Context): Boolean {
        if (attempted) return session != null
        attempted = true
        var candidate: OrtSession? = null
        var options: OrtSession.SessionOptions? = null
        return try {
            if (!isAvailable(context)) return false
            val bytes = context.assets.open(ASSET_NAME).use { it.readBytes() }
            val env = OrtEnvironment.getEnvironment()
            options = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(2)
            }
            candidate = env.createSession(bytes, options)
            val name = candidate.inputNames.firstOrNull() ?: return false
            val shape = candidate.getInputInfo()[name]?.info
            val tensorShape = (shape as? ai.onnxruntime.TensorInfo)?.shape()
            if (tensorShape == null || tensorShape.size != 4 ||
                tensorShape[1] != 3L || tensorShape[2] != SIZE.toLong() ||
                tensorShape[3] != SIZE.toLong()) {
                return false
            }
            environment = env
            session = candidate
            inputName = name
            candidate = null
            true
        } catch (_: Exception) {
            false
        } finally {
            try { candidate?.close() } catch (_: Exception) {}
            try { options?.close() } catch (_: Exception) {}
        }
    }

    /**
     * Renders and scores the landmark-bounded face crop. Two forward passes with
     * horizontal-flip TTA mirror the published inference recipe.
     */
    @Synchronized
    fun score(context: Context, renderedCandidate: Bitmap, landmarks: List<PointF>): Float? {
        if (renderedCandidate.isRecycled || landmarks.size < 455 || !load(context)) return null
        val activeSession = session ?: return null
        val activeInput = inputName ?: return null
        val points = landmarks.take(468).filter { it.x.isFinite() && it.y.isFinite() }
        if (points.size < 200) return null

        val minX = points.minOf { it.x }
        val maxX = points.maxOf { it.x }
        val minY = points.minOf { it.y }
        val maxY = points.maxOf { it.y }
        val faceWidth = maxX - minX
        val faceHeight = maxY - minY
        if (!faceWidth.isFinite() || !faceHeight.isFinite() || faceWidth < 32f || faceHeight < 32f) return null

        val left = (minX - faceWidth * 0.12f).toInt().coerceIn(0, renderedCandidate.width - 1)
        val top = (minY - faceHeight * 0.10f).toInt().coerceIn(0, renderedCandidate.height - 1)
        val right = (maxX + faceWidth * 0.12f).toInt().coerceIn(left + 1, renderedCandidate.width)
        val bottom = (maxY + faceHeight * 0.07f).toInt().coerceIn(top + 1, renderedCandidate.height)
        if (right - left < 32 || bottom - top < 32) return null

        var crop: Bitmap? = null
        var resized: Bitmap? = null
        return try {
            crop = Bitmap.createBitmap(renderedCandidate, left, top, right - left, bottom - top)
            resized = Bitmap.createScaledBitmap(crop, SIZE, SIZE, true)
            val ordinary = runOne(activeSession, activeInput, resized, flip = false)
            val flipped = runOne(activeSession, activeInput, resized, flip = true)
            if (!ordinary.isFinite() || !flipped.isFinite()) null
            else ((ordinary + flipped) * 0.5f).coerceIn(1f, 5f)
        } catch (_: Exception) {
            null
        } finally {
            try { if (resized != null && resized !== crop && !resized.isRecycled) resized.recycle() } catch (_: Exception) {}
            try { if (crop != null && !crop.isRecycled) crop.recycle() } catch (_: Exception) {}
        }
    }

    private fun runOne(
        activeSession: OrtSession,
        activeInput: String,
        bitmap: Bitmap,
        flip: Boolean
    ): Float {
        val pixels = IntArray(SIZE * SIZE)
        bitmap.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        val values = FloatArray(3 * plane)
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                val sourceX = if (flip) SIZE - 1 - x else x
                val color = pixels[y * SIZE + sourceX]
                val index = y * SIZE + x
                val red = ((color shr 16) and 0xff) / 255f
                val green = ((color shr 8) and 0xff) / 255f
                val blue = (color and 0xff) / 255f
                values[index] = (red - mean[0]) / std[0]
                values[plane + index] = (green - mean[1]) / std[1]
                values[2 * plane + index] = (blue - mean[2]) / std[2]
            }
        }
        val tensor = OnnxTensor.createTensor(
            environment ?: OrtEnvironment.getEnvironment(),
            FloatBuffer.wrap(values),
            longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
        )
        try {
            activeSession.run(mapOf(activeInput to tensor)).use { result ->
                val output = result[0].value
                val score = when (output) {
                    is Array<*> -> (output.firstOrNull() as? FloatArray)?.firstOrNull()
                    is FloatArray -> output.firstOrNull()
                    is Number -> output.toFloat()
                    else -> null
                } ?: throw IllegalStateException("Unexpected SCUT regressor output tensor")
                return score
            }
        } finally {
            tensor.close()
        }
    }
}
