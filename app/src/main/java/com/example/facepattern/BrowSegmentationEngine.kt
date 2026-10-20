package com.example.facepattern

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import org.tensorflow.lite.Interpreter

internal data class BrowSegmentationResult(
    val overlay: Bitmap,
    val offsetX: Int,
    val offsetY: Int,
    val method: String,
    val maskQualityScore: Float,
    val selectedPixelCount: Int,
    val regions: List<IntArray>
)

/**
 * Brow-region pre-segmentation.
 *
 * When a customer-trained brow_segmentation.tflite model is present, it is used.
 * Otherwise this falls back to an adaptive dark-hair contrast mask. The fallback
 * is intentionally described as a pre-segmentation aid and is not a medical/
 * cosmetic-grade segmentation guarantee.
 */
internal object BrowSegmentationEngine {
    private const val MODEL_SIZE = 256
    private val browSets = listOf(
        intArrayOf(70, 63, 105, 66, 107),
        intArrayOf(300, 293, 334, 296, 336)
    )
    private var modelAttempted = false
    private var model: Interpreter? = null

    @Synchronized
    private fun loadModel(context: Context): Interpreter? {
        if (modelAttempted) return model
        modelAttempted = true
        try {
            val assets = context.assets.list("")?.toSet().orEmpty()
            if ("brow_segmentation.tflite" !in assets) return null
            val bytes = context.assets.open("brow_segmentation.tflite").use { it.readBytes() }
            val mapped = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            mapped.put(bytes)
            mapped.rewind()
            val candidate = Interpreter(mapped, Interpreter.Options().setNumThreads(2))
            val input = candidate.getInputTensor(0).shape()
            val output = candidate.getOutputTensor(0).shape()
            if (!input.contentEquals(intArrayOf(1, MODEL_SIZE, MODEL_SIZE, 3)) ||
                !output.contentEquals(intArrayOf(1, MODEL_SIZE, MODEL_SIZE, 1))) {
                candidate.close()
                return null
            }
            model = candidate
        } catch (_: Exception) {
            try { model?.close() } catch (_: Exception) {}
            model = null
        }
        return model
    }

    fun segment(
        context: Context,
        source: Bitmap,
        landmarks: List<PointF>,
        sensitivity: Float
    ): BrowSegmentationResult? {
        if (source.isRecycled || landmarks.size < 455) return null
        val regions = browSets.mapNotNull { ids -> regionBounds(source, landmarks, ids) }
        if (regions.size != 2) return null
        val left = regions.minOf { it[0] }
        val top = regions.minOf { it[1] }
        val right = regions.maxOf { it[2] }
        val bottom = regions.maxOf { it[3] }
        if (right <= left || bottom <= top) return null

        val overlay = Bitmap.createBitmap(right - left, bottom - top, Bitmap.Config.ARGB_8888)
        val colors = IntArray(overlay.width * overlay.height)
        val loadedModel = loadModel(context)
        var totalSelected = 0
        var confidenceSum = 0f
        var processedRegions = 0

        for (bounds in regions) {
            val x0 = bounds[0]
            val y0 = bounds[1]
            val x1 = bounds[2]
            val y1 = bounds[3]
            val roi = Bitmap.createBitmap(source, x0, y0, x1 - x0, y1 - y0)
            val prediction = try {
                if (loadedModel != null) predictWithModel(loadedModel, roi) else null
            } catch (_: Exception) {
                null
            }
            val mask = prediction ?: adaptiveMask(roi, sensitivity)
            val count = mask.count { it }
            val ratio = count.toFloat() / mask.size.coerceAtLeast(1)
            val localConfidence = when {
                ratio in 0.02f..0.28f -> 0.82f
                ratio in 0.005f..0.40f -> 0.58f
                else -> 0.25f
            }
            confidenceSum += localConfidence
            processedRegions++
            totalSelected += count

            var localIndex = 0
            for (y in 0 until roi.height) {
                val targetY = y0 - top + y
                for (x in 0 until roi.width) {
                    if (mask[localIndex]) {
                        val targetX = x0 - left + x
                        colors[targetY * overlay.width + targetX] = Color.argb(172, 71, 190, 157)
                    }
                    localIndex++
                }
            }
            if (roi !== source && !roi.isRecycled) roi.recycle()
        }
        overlay.setPixels(colors, 0, overlay.width, 0, 0, overlay.width, overlay.height)
        val usedModel = loadedModel != null && regions.size == 2
        return BrowSegmentationResult(
            overlay = overlay,
            offsetX = left,
            offsetY = top,
            method = if (usedModel) "本地眉毛分割模型" else "自适应对比度预分割",
            maskQualityScore = if (processedRegions == 0) 0f else confidenceSum / processedRegions,
            selectedPixelCount = totalSelected,
            regions = regions.map { it.copyOf() }
        )
    }

    private fun regionBounds(source: Bitmap, points: List<PointF>, ids: IntArray): IntArray? {
        val valid = ids.map { points.getOrNull(it) }.filterNotNull()
            .filter { it.x.isFinite() && it.y.isFinite() }
        if (valid.size < 4) return null
        val span = max(8f, valid.maxOf { it.x } - valid.minOf { it.x })
        val padX = max(6f, span * 0.14f).toInt()
        val padTop = max(8f, span * 0.42f).toInt()
        val padBottom = max(7f, span * 0.25f).toInt()
        val x0 = (valid.minOf { it.x }.toInt() - padX).coerceIn(0, source.width - 1)
        val x1 = (valid.maxOf { it.x }.toInt() + padX + 1).coerceIn(x0 + 1, source.width)
        val y0 = (valid.minOf { it.y }.toInt() - padTop).coerceIn(0, source.height - 1)
        val y1 = (valid.maxOf { it.y }.toInt() + padBottom + 1).coerceIn(y0 + 1, source.height)
        return intArrayOf(x0, y0, x1, y1)
    }

    private fun predictWithModel(interpreter: Interpreter, roi: Bitmap): BooleanArray? {
        val scaled = Bitmap.createScaledBitmap(roi, MODEL_SIZE, MODEL_SIZE, true)
        val input = ByteBuffer.allocateDirect(MODEL_SIZE * MODEL_SIZE * 3 * 4).order(ByteOrder.nativeOrder())
        for (y in 0 until MODEL_SIZE) {
            for (x in 0 until MODEL_SIZE) {
                val pixel = scaled.getPixel(x, y)
                input.putFloat(Color.red(pixel) / 255f)
                input.putFloat(Color.green(pixel) / 255f)
                input.putFloat(Color.blue(pixel) / 255f)
            }
        }
        if (scaled !== roi && !scaled.isRecycled) scaled.recycle()
        input.rewind()
        val output = ByteBuffer.allocateDirect(MODEL_SIZE * MODEL_SIZE * 4).order(ByteOrder.nativeOrder())
        interpreter.run(input, output)
        output.rewind()
        val probabilities = FloatArray(MODEL_SIZE * MODEL_SIZE) { output.float }
        val mask = BooleanArray(roi.width * roi.height)
        var k = 0
        for (y in 0 until roi.height) {
            val sy = ((y + 0.5f) * MODEL_SIZE / roi.height).toInt().coerceIn(0, MODEL_SIZE - 1)
            for (x in 0 until roi.width) {
                val sx = ((x + 0.5f) * MODEL_SIZE / roi.width).toInt().coerceIn(0, MODEL_SIZE - 1)
                mask[k++] = probabilities[sy * MODEL_SIZE + sx] >= 0.5f
            }
        }
        return mask
    }

    private fun adaptiveMask(roi: Bitmap, sensitivity: Float): BooleanArray {
        val pixels = IntArray(roi.width * roi.height)
        roi.getPixels(pixels, 0, roi.width, 0, 0, roi.width, roi.height)
        val luma = IntArray(pixels.size)
        for (i in pixels.indices) {
            val p = pixels[i]
            luma[i] = (0.299 * Color.red(p) + 0.587 * Color.green(p) + 0.114 * Color.blue(p)).toInt()
        }

        val borderSamples = ArrayList<Int>()
        val edgeX = max(1, roi.width / 8)
        val topBand = max(1, roi.height / 4)
        val bottomStart = roi.height * 3 / 4
        for (y in 0 until roi.height) {
            for (x in 0 until roi.width) {
                if (y < topBand || y >= bottomStart || x < edgeX || x >= roi.width - edgeX) {
                    borderSamples.add(luma[y * roi.width + x])
                }
            }
        }
        borderSamples.sort()
        val skinReference = if (borderSamples.isEmpty()) 150 else borderSamples[borderSamples.size / 2]
        val threshold = (skinReference * sensitivity.coerceIn(0.45f, 0.92f)).toInt().coerceIn(28, 175)
        val raw = BooleanArray(luma.size) { luma[it] < threshold }
        val filtered = BooleanArray(raw.size)
        for (y in 1 until roi.height - 1) {
            for (x in 1 until roi.width - 1) {
                val index = y * roi.width + x
                if (!raw[index]) continue
                var neighbours = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if ((dx != 0 || dy != 0) && raw[(y + dy) * roi.width + x + dx]) neighbours++
                    }
                }
                filtered[index] = neighbours >= 2
            }
        }
        return filtered
    }

    fun close() {
        synchronized(this) {
            try { model?.close() } catch (_: Exception) {}
            model = null
            modelAttempted = false
        }
    }
}
