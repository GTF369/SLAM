package com.example.facepattern

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.os.SystemClock
import java.nio.ByteBuffer
import java.nio.ByteOrder
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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

private data class FaceParsingMap(
    val labels: ByteArray,
    val cropLeft: Int,
    val cropTop: Int,
    val cropSize: Int
)

/**
 * Brow-region pre-segmentation.
 *
 * When the LiteRT faceparsing.tflite model is present, classes 2 and 3 are used
 * as the left/right eyebrow masks. The model is run on a square crop of the full
 * face because it was trained for whole-face parsing, not isolated brow crops.
 * If loading or inference fails, the engine falls back to an adaptive local mask.
 * The learned model is a starting point and still requires professional review.
 */
internal object BrowSegmentationEngine {
    private const val PARSING_MODEL_SIZE = 384
    private const val CUSTOM_MODEL_SIZE = 256
    private const val PARSING_CLASSES = 8
    // EasyPortrait face-parsing BGR input statistics, applied to pixel values in [0,255].
    private val meanBgr = floatArrayOf(143.55267f, 132.96706f, 126.94924f)
    private val stdBgr = floatArrayOf(60.26253f, 60.32740f, 59.30989f)
    private enum class ModelKind { CUSTOM_BROW }
    private val browSets = listOf(
        intArrayOf(70, 63, 105, 66, 107),
        intArrayOf(300, 293, 334, 296, 336)
    )
    private var modelAttempted = false
    private var model: Interpreter? = null
    private var modelKind: ModelKind? = null
    private var onnxAttempted = false
    private var onnxEnvironment: OrtEnvironment? = null
    private var onnxSession: OrtSession? = null
    private var onnxInputName = "image"
    private var onnxOutputName = "logits"

    @Synchronized
    private fun loadModel(context: Context): Interpreter? {
        if (modelAttempted) return model
        modelAttempted = true
        val assets = context.assets.list("")?.toSet().orEmpty()
        if ("brow_segmentation.tflite" !in assets) return null
        var candidate: Interpreter? = null
        try {
            val bytes = context.assets.open("brow_segmentation.tflite").use { it.readBytes() }
            val mapped = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            mapped.put(bytes)
            mapped.rewind()
            candidate = Interpreter(mapped, Interpreter.Options().setNumThreads(2))
            val input = candidate.getInputTensor(0).shape()
            val output = candidate.getOutputTensor(0).shape()
            if (input.contentEquals(intArrayOf(1, CUSTOM_MODEL_SIZE, CUSTOM_MODEL_SIZE, 3)) &&
                output.contentEquals(intArrayOf(1, CUSTOM_MODEL_SIZE, CUSTOM_MODEL_SIZE, 1))) {
                model = candidate
                modelKind = ModelKind.CUSTOM_BROW
            } else {
                candidate.close()
            }
        } catch (_: Exception) {
            try { candidate?.close() } catch (_: Exception) {}
            model = null
            modelKind = null
        }
        return model
    }

    @Synchronized
    private fun loadFaceParsingSession(context: Context): OrtSession? {
        if (onnxAttempted) return onnxSession
        onnxAttempted = true
        if ("faceparsing.onnx" !in context.assets.list("")?.toSet().orEmpty()) return null
        var candidate: OrtSession? = null
        var options: OrtSession.SessionOptions? = null
        try {
            val modelBytes = context.assets.open("faceparsing.onnx").use { it.readBytes() }
            val environment = OrtEnvironment.getEnvironment()
            options = OrtSession.SessionOptions()
            candidate = environment.createSession(modelBytes, options)
            val inputName = candidate.inputNames.firstOrNull()
            val outputName = candidate.outputNames.firstOrNull()
            if (inputName == null || outputName == null) {
                candidate.close()
                return null
            }
            onnxEnvironment = environment
            onnxSession = candidate
            onnxInputName = inputName
            onnxOutputName = outputName
            return onnxSession
        } catch (_: Exception) {
            try { candidate?.close() } catch (_: Exception) {}
            onnxSession = null
            onnxEnvironment = null
            return null
        } finally {
            try { options?.close() } catch (_: Exception) {}
        }
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
        val activeModelKind = modelKind
        val faceParsingSession = if (loadedModel == null) loadFaceParsingSession(context) else null
        val faceParsing = try {
            if (faceParsingSession != null) {
                predictFaceParsing(faceParsingSession, source, landmarks)
            } else {
                null
            }
        } catch (_: Exception) {
            null
        }
        var totalSelected = 0
        var qualitySum = 0f
        var processedRegions = 0
        var modelAppliedRegions = 0

        for (bounds in regions) {
            val x0 = bounds[0]
            val y0 = bounds[1]
            val x1 = bounds[2]
            val y1 = bounds[3]
            val roi = Bitmap.createBitmap(source, x0, y0, x1 - x0, y1 - y0)
            val modelMask = try {
                when {
                    activeModelKind == ModelKind.CUSTOM_BROW ->
                        loadedModel?.let { predictWithCustomBrowModel(it, roi) }
                    faceParsing != null -> maskFromFaceParsing(faceParsing, bounds)
                    else -> null
                }
            } catch (_: Exception) {
                null
            }
            val modelPixels = modelMask?.count { it } ?: 0
            // A nearly empty model output can occur under makeup, harsh shadows or
            // unusual lighting. Use the editable contrast mask instead of returning
            // a blank eyebrow in that case.
            val useModelMask = modelMask != null &&
                modelPixels >= max(5, (modelMask.size * 0.003f).toInt())
            val mask = if (useModelMask) {
                modelAppliedRegions++
                modelMask!!
            } else {
                adaptiveMask(roi, sensitivity)
            }
            val count = mask.count { it }
            val ratio = count.toFloat() / mask.size.coerceAtLeast(1)
            val localQuality = when {
                ratio in 0.02f..0.28f -> 0.82f
                ratio in 0.005f..0.40f -> 0.58f
                else -> 0.25f
            }
            qualitySum += localQuality
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
            if (!roi.isRecycled) roi.recycle()
        }
        overlay.setPixels(colors, 0, overlay.width, 0, 0, overlay.width, overlay.height)
        val usedModel = modelAppliedRegions > 0
        val modelLabel = if (activeModelKind == ModelKind.CUSTOM_BROW) {
            "客户数据训练的眉毛分割模型"
        } else {
            "EasyPortrait BiSeNet 人脸解析模型"
        }
        val method = when {
            !usedModel -> "自适应对比度预分割"
            modelAppliedRegions == regions.size -> modelLabel
            else -> "$modelLabel + 对比度补底"
        }
        return BrowSegmentationResult(
            overlay = overlay,
            offsetX = left,
            offsetY = top,
            method = method,
            maskQualityScore = if (processedRegions == 0) 0f else qualitySum / processedRegions,
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

    private fun predictWithCustomBrowModel(interpreter: Interpreter, roi: Bitmap): BooleanArray {
        val scaled = Bitmap.createScaledBitmap(roi, CUSTOM_MODEL_SIZE, CUSTOM_MODEL_SIZE, true)
        try {
            val input = ByteBuffer.allocateDirect(
                CUSTOM_MODEL_SIZE * CUSTOM_MODEL_SIZE * 3 * 4
            ).order(ByteOrder.nativeOrder())
            val pixels = IntArray(CUSTOM_MODEL_SIZE * CUSTOM_MODEL_SIZE)
            scaled.getPixels(pixels, 0, CUSTOM_MODEL_SIZE, 0, 0, CUSTOM_MODEL_SIZE, CUSTOM_MODEL_SIZE)
            for (pixel in pixels) {
                input.putFloat(Color.red(pixel) / 255f)
                input.putFloat(Color.green(pixel) / 255f)
                input.putFloat(Color.blue(pixel) / 255f)
            }
            input.rewind()
            val output = ByteBuffer.allocateDirect(
                CUSTOM_MODEL_SIZE * CUSTOM_MODEL_SIZE * 4
            ).order(ByteOrder.nativeOrder())
            interpreter.run(input, output)
            output.rewind()
            val probabilities = FloatArray(CUSTOM_MODEL_SIZE * CUSTOM_MODEL_SIZE) { output.float }
            val mask = BooleanArray(roi.width * roi.height)
            var k = 0
            for (y in 0 until roi.height) {
                val sy = ((y + 0.5f) * CUSTOM_MODEL_SIZE / roi.height).toInt().coerceIn(0, CUSTOM_MODEL_SIZE - 1)
                for (x in 0 until roi.width) {
                    val sx = ((x + 0.5f) * CUSTOM_MODEL_SIZE / roi.width).toInt().coerceIn(0, CUSTOM_MODEL_SIZE - 1)
                    mask[k++] = probabilities[sy * CUSTOM_MODEL_SIZE + sx] >= 0.5f
                }
            }
            return mask
        } finally {
            if (scaled !== roi && !scaled.isRecycled) scaled.recycle()
        }
    }

    private fun predictFaceParsing(
        session: OrtSession,
        source: Bitmap,
        landmarks: List<PointF>
    ): FaceParsingMap? {
        if (source.width < 64 || source.height < 64 || landmarks.size <= 454) return null
        val leftPoint = landmarks[234]
        val rightPoint = landmarks[454]
        val topPoint = landmarks.getOrNull(10) ?: return null
        val bottomPoint = landmarks.getOrNull(152) ?: return null
        val faceWidth = abs(rightPoint.x - leftPoint.x)
        val faceHeight = abs(bottomPoint.y - topPoint.y)
        if (!faceWidth.isFinite() || !faceHeight.isFinite() || faceWidth < 16f || faceHeight < 16f) {
            return null
        }

        // Keep the original image aspect ratio inside a square face crop. The
        // model is trained on a full face, unlike the previous small brow-ROI model.
        val requestedSide = (max(faceWidth, faceHeight) * 1.22f).toInt()
        val side = requestedSide.coerceIn(64, min(source.width, source.height))
        val centerX = (leftPoint.x + rightPoint.x) / 2f
        val centerY = (topPoint.y + bottomPoint.y) / 2f
        val cropLeft = (centerX - side / 2f).toInt().coerceIn(0, source.width - side)
        val cropTop = (centerY - side / 2f).toInt().coerceIn(0, source.height - side)
        val faceCrop = Bitmap.createBitmap(source, cropLeft, cropTop, side, side)
        val scaled = Bitmap.createScaledBitmap(faceCrop, PARSING_MODEL_SIZE, PARSING_MODEL_SIZE, true)
        if (scaled !== faceCrop && !faceCrop.isRecycled) faceCrop.recycle()

        try {
            val pixels = IntArray(PARSING_MODEL_SIZE * PARSING_MODEL_SIZE)
            scaled.getPixels(pixels, 0, PARSING_MODEL_SIZE, 0, 0, PARSING_MODEL_SIZE, PARSING_MODEL_SIZE)
            val inputBuffer = ByteBuffer.allocateDirect(
                PARSING_MODEL_SIZE * PARSING_MODEL_SIZE * 3 * 4
            ).order(ByteOrder.nativeOrder()).asFloatBuffer()

            // EasyPortrait BiSeNet uses NCHW BGR pixels in [0,255] and the
            // per-channel mean/std published with its ONNX model.
            for (channel in 0..2) {
                for (pixel in pixels) {
                    val channelValue = when (channel) {
                        0 -> Color.blue(pixel).toFloat()
                        1 -> Color.green(pixel).toFloat()
                        else -> Color.red(pixel).toFloat()
                    }
                    inputBuffer.put((channelValue - meanBgr[channel]) / stdBgr[channel])
                }
            }
            inputBuffer.rewind()
            val environment = onnxEnvironment ?: OrtEnvironment.getEnvironment()
            val inputTensor = OnnxTensor.createTensor(
                environment,
                inputBuffer,
                longArrayOf(1L, 3L, PARSING_MODEL_SIZE.toLong(), PARSING_MODEL_SIZE.toLong())
            )
            try {
                val outputs = session.run(mapOf(onnxInputName to inputTensor))
                try {
                    val outputTensor = outputs.get(onnxOutputName) as? OnnxTensor ?: return null
                    val logits = outputTensor.floatBuffer
                    val planeSize = PARSING_MODEL_SIZE * PARSING_MODEL_SIZE
                    val labels = ByteArray(planeSize)
                    for (pixel in 0 until planeSize) {
                        var bestClass = 0
                        var bestValue = logits.get(pixel)
                        for (classIndex in 1 until PARSING_CLASSES) {
                            val value = logits.get(classIndex * planeSize + pixel)
                            if (value > bestValue) {
                                bestValue = value
                                bestClass = classIndex
                            }
                        }
                        labels[pixel] = bestClass.toByte()
                    }
                    return FaceParsingMap(labels, cropLeft, cropTop, side)
                } finally {
                    outputs.close()
                }
            } finally {
                inputTensor.close()
            }
        } finally {
            if (!scaled.isRecycled) scaled.recycle()
        }
    }

    private fun maskFromFaceParsing(
        parsed: FaceParsingMap,
        bounds: IntArray
    ): BooleanArray {
        val x0 = bounds[0]
        val y0 = bounds[1]
        val width = bounds[2] - x0
        val height = bounds[3] - y0
        val result = BooleanArray(width * height)
        var index = 0
        for (y in 0 until height) {
            val imageY = y0 + y
            val modelY = (((imageY + 0.5f - parsed.cropTop) * PARSING_MODEL_SIZE) / parsed.cropSize)
                .toInt()
            if (modelY !in 0 until PARSING_MODEL_SIZE) {
                index += width
                continue
            }
            for (x in 0 until width) {
                val imageX = x0 + x
                val modelX = (((imageX + 0.5f - parsed.cropLeft) * PARSING_MODEL_SIZE) / parsed.cropSize)
                    .toInt()
                if (modelX in 0 until PARSING_MODEL_SIZE) {
                    val label = parsed.labels[modelY * PARSING_MODEL_SIZE + modelX].toInt() and 0xFF
                    // EasyPortrait classes: 2 = left brow, 3 = right brow.
                    result[index] = label == 2 || label == 3
                }
                index++
            }
        }
        return result
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
            try { onnxSession?.close() } catch (_: Exception) {}
            model = null
            modelKind = null
            onnxSession = null
            onnxEnvironment = null
            onnxAttempted = false
            modelAttempted = false
        }
    }
}
