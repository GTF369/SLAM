package com.example.facepattern

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.app.AlertDialog
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.graphics.Shader
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import android.print.PageRange
import android.print.PrintAttributes
import android.print.PrintDocumentAdapter
import android.print.PrintDocumentInfo
import android.provider.MediaStore
import android.print.PrintManager
import android.print.pdf.PrintedPdfDocument
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.animation.LinearInterpolator
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.exifinterface.media.ExifInterface
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import java.util.UUID
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sin

private data class BrowParams(
    val thickness: Float = 1.0f,
    val length: Float = 1.0f,
    val arch: Float = 1.0f,
    val style: Int = 0
)

private data class BrowPlan(
    val name: String,
    val params: BrowParams,
    val useNativeShape: Boolean = false,
    val segmentationSnapshot: BrowSegmentationResult? = null
)

private data class NativeBrowContour(
    val regionIndex: Int,
    val xStart: Int,
    val top: FloatArray,
    val bottom: FloatArray,
    val regionBounds: IntArray
)

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var browView: BrowDesignView
    private lateinit var status: TextView
    private lateinit var liveControls: LinearLayout
    private lateinit var designControls: LinearLayout
    private lateinit var plansRow: LinearLayout
    private lateinit var recommendationRow: LinearLayout
    private lateinit var recommendationLabel: TextView
    private lateinit var segmentationLabel: TextView
    private lateinit var segmentationSensitivitySeek: SeekBar
    private lateinit var pauseButton: Button
    private lateinit var thicknessSeek: SeekBar
    private lateinit var lengthSeek: SeekBar
    private lateinit var archSeek: SeekBar

    private var landmarker: FaceLandmarker? = null
    private var cameraExecutor: ExecutorService? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private var imageExecutor: ExecutorService? = null
    private var captureInProgress = false
    private var currentCaseId: String = UUID.randomUUID().toString().take(8)
    private val lastTimestamp = AtomicLong(0L)
    private val cameraPermission = 701
    private var frameWidth = 640
    private var frameHeight = 480
    private var designMode = false
    private var capturedPhoto: Bitmap? = null
    // MediaPipe z is a relative depth estimate, scaled to image width rather than millimetres.
    private var capturedLandmarkDepths: List<Float> = emptyList()
    private var params = BrowParams()
    private val history = mutableListOf<BrowParams>()
    private val savedPlans = mutableListOf<BrowPlan>()
    private var transitionAnimator: ValueAnimator? = null
    private var useNativeShape = false
    private var segmentationResult: BrowSegmentationResult? = null
    private var showNativeSegmentation = false
    private var segmentationSensitivity = 0.72f
    private var segmentationGeneration = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startFaceTracking()
        } else {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), cameraPermission)
        }
    }

    private fun buildUi() {
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        browView = BrowDesignView(this)
        browView.setParams(params)

        val root = FrameLayout(this)
        root.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        root.addView(browView, FrameLayout.LayoutParams(-1, -1))

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(14))
            background = GradientDrawable().apply {
                setColor(0xEC171A20.toInt())
                cornerRadius = dp(22).toFloat()
                setStroke(dp(1), 0x32FFFFFF)
            }
        }
        panel.addView(TextView(this).apply {
            text = "眉形设计助手"
            textSize = 18f
            setTextColor(0xFFF3F0E7.toInt())
            typeface = android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL)
            setPadding(dp(2), 0, dp(2), dp(2))
        })
        status = TextView(this).apply {
            text = "正在启动摄像头，请正对镜头并保持光线充足"
            textSize = 12f
            setTextColor(0xFFD3D8D0.toInt())
            setPadding(dp(2), dp(2), dp(2), dp(8))
        }
        panel.addView(status)

        liveControls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addButtonRow(this, listOf(
                "拍照设计" to { captureForDesign() },
                "重置" to { browView.setParams(BrowParams()); params = BrowParams() }
            ))
        }
        panel.addView(liveControls)

        designControls = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        recommendationLabel = TextView(this).apply {
            textSize = 11f
            setTextColor(0xFFE0E1E5.toInt())
            setPadding(dp(2), dp(2), dp(2), dp(6))
        }
        designControls.addView(recommendationLabel)
        recommendationRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        designControls.addView(recommendationRow)
        segmentationLabel = TextView(this).apply {
            text = "原生眉区识别：尚未运行。当前模型会在本机处理照片。"
            textSize = 11f
            setTextColor(0xFFD3D8D0.toInt())
            setPadding(dp(2), dp(6), dp(2), dp(4))
        }
        designControls.addView(segmentationLabel)
        val segmentationRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        segmentationRow.addView(TextView(this).apply {
            text = "识别灵敏度"
            textSize = 12f
            setTextColor(0xFFE9EAE4.toInt())
        }, LinearLayout.LayoutParams(dp(82), -2))
        segmentationSensitivitySeek = SeekBar(this).apply {
            max = 100
            progress = 72
            progressTintList = android.content.res.ColorStateList.valueOf(0xFF78D8B6.toInt())
            thumbTintList = android.content.res.ColorStateList.valueOf(0xFF78D8B6.toInt())
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    if (fromUser) {
                        segmentationSensitivity = 0.45f + progress / 100f * 0.47f
                        if (showNativeSegmentation) runSegmentation()
                    }
                }
            })
        }
        segmentationRow.addView(segmentationSensitivitySeek, LinearLayout.LayoutParams(0, dp(38), 1f))
        designControls.addView(segmentationRow)
        addButtonRow(designControls, listOf(
            "识别原生眉区" to { toggleNativeSegmentation() },
            "隐藏识别层" to {
                showNativeSegmentation = false
                browView.setSegmentationEditMode(false)
                browView.setSegmentationResult(segmentationResult, false)
            }
        ))
        addButtonRow(designControls, listOf(
            "画入缺失眉区" to {
                showNativeSegmentation = true
                browView.setSegmentationEditMode(true, includePixels = true)
                toggleNativeSegmentation()
            },
            "擦除误识别" to {
                showNativeSegmentation = true
                browView.setSegmentationEditMode(true, includePixels = false)
                toggleNativeSegmentation()
            },
            "完成修正" to { browView.setSegmentationEditMode(false) }
        ))
        addButtonRow(designControls, listOf(
            "以原生眉形作底稿" to { toggleNativeShapeDesign() },
            "使用参数化底稿" to { toggleNativeShapeDesign(forceOff = true) }
        ))
        thicknessSeek = addSlider(designControls, "粗细", 100, 50) { p ->
            params = params.copy(thickness = 0.55f + p / 100f * 0.90f)
            browView.setParams(params)
        }
        lengthSeek = addSlider(designControls, "长度", 100, 50) { p ->
            params = params.copy(length = 0.80f + p / 100f * 0.40f)
            browView.setParams(params)
        }
        archSeek = addSlider(designControls, "弧度", 100, 50) { p ->
            params = params.copy(arch = 0.55f + p / 100f * 0.90f)
            browView.setParams(params)
        }
        addButtonRow(designControls, listOf(
            "暂停 / 继续" to { toggleTransition() },
            "回撤一步" to { undoAdjustment() },
            "保存方案" to { savePlan() }
        )).also { row ->
            pauseButton = row.getChildAt(0) as Button
        }
        plansRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
        }
        designControls.addView(plansRow)
        addButtonRow(designControls, listOf(
            "方案对比" to { showComparison() },
            "保存效果图" to { savePreviewToGallery() },
            "1:1 打印" to { askPrintCalibration() }
        ))
        addButtonRow(designControls, listOf(
            "导出原始标注照片" to { exportOriginalPhotoForAnnotation() }
        ))
        addButtonRow(designControls, listOf(
            "模型来源与许可" to { showModelLicenseNotice() }
        ))
        addButtonRow(designControls, listOf(
            "顾客偏好" to { recordPreference() },
            "导出偏好数据" to { exportPreferenceCsv() },
            "清除本机记录" to { clearLocalRecords() }
        ))
        addButtonRow(designControls, listOf(
            "曲率标定" to { showCurvatureCalibration() },
            "重新拍摄" to { retakePhoto() }
        ))
        val note = TextView(this).apply {
            text = "打印前需实测瞳距。打印页附 20 mm 校准框，打印后务必量取确认。单张正面照不能可靠恢复眉弓三维曲率，本版不自动做三维曲面校正。"
            textSize = 11f
            setTextColor(0xFFD5D7DC.toInt())
            setPadding(dp(2), dp(6), dp(2), 0)
        }
        designControls.addView(note)
        designControls.visibility = View.GONE
        panel.addView(designControls)

        val scroll = ScrollView(this).apply {
            isFillViewport = false
            clipToPadding = false
            setPadding(dp(6), dp(4), dp(6), dp(6))
            background = GradientDrawable().apply {
                setColor(Color.TRANSPARENT)
                cornerRadius = dp(24).toFloat()
            }
            addView(panel)
        }
        root.addView(scroll, FrameLayout.LayoutParams(-1, dp(370), Gravity.BOTTOM).apply {
            leftMargin = dp(8)
            rightMargin = dp(8)
            bottomMargin = dp(8)
        })
        setContentView(root)
        // Android 15/16 enforces edge-to-edge layout for newer target SDKs.
        // Keep the studio controls above the system navigation area.
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, windowInsets ->
            val systemBars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            val params = scroll.layoutParams as FrameLayout.LayoutParams
            params.leftMargin = dp(8) + systemBars.left
            params.rightMargin = dp(8) + systemBars.right
            params.bottomMargin = dp(8) + systemBars.bottom
            scroll.layoutParams = params
            windowInsets
        }
        ViewCompat.requestApplyInsets(root)
    }

    private fun addTextLabel(parent: LinearLayout, text: String) {
        parent.addView(TextView(this).apply {
            this.text = text
            textSize = 11f
            setTextColor(0xFFE0E1E5.toInt())
            setPadding(dp(2), dp(2), dp(2), dp(6))
        })
    }

    private fun addButtonRow(parent: LinearLayout, items: List<Pair<String, () -> Unit>>): LinearLayout {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        items.forEach { (label, action) ->
            val isPrimary = label in setOf("拍照设计", "保存方案", "保存效果图", "1:1 打印", "生成打印模板")
            val button = Button(this).apply {
                text = label
                textSize = 12f
                isAllCaps = false
                minHeight = dp(42)
                setPadding(dp(4), 0, dp(4), 0)
                setTextColor(if (isPrimary) 0xFF1B201A.toInt() else 0xFFE9EAE4.toInt())
                background = GradientDrawable().apply {
                    cornerRadius = dp(13).toFloat()
                    setColor(if (isPrimary) 0xFFD5DDBF.toInt() else 0xFF30343D.toInt())
                    if (!isPrimary) setStroke(dp(1), 0x30FFFFFF)
                }
                stateListAnimator = null
                setOnClickListener { action() }
            }
            row.addView(button, LinearLayout.LayoutParams(0, dp(44), 1f).apply {
                setMargins(dp(2), dp(2), dp(2), dp(4))
            })
        }
        parent.addView(row)
        return row
    }

    private fun addSlider(parent: LinearLayout, title: String, maxValue: Int, initial: Int, onChange: (Int) -> Unit): SeekBar {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val label = TextView(this).apply {
            text = title
            textSize = 13f
            setTextColor(Color.WHITE)
        }
        row.addView(label, LinearLayout.LayoutParams(dp(42), -2))
        val seek = SeekBar(this).apply {
            max = maxValue
            progressTintList = android.content.res.ColorStateList.valueOf(0xFFD5DDBF.toInt())
            thumbTintList = android.content.res.ColorStateList.valueOf(0xFFD5DDBF.toInt())
            progress = initial
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onStartTrackingTouch(seekBar: SeekBar?) {
                    stopTransition()
                    history.add(params.copy())
                }
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    if (fromUser) onChange(progress)
                }
            })
        }
        row.addView(seek, LinearLayout.LayoutParams(0, dp(38), 1f))
        parent.addView(row)
        return seek
    }

    private fun startFaceTracking() {
        try {
            landmarker?.close()
            landmarker = FaceLandmarker.createFromOptions(
                this,
                FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumFaces(1)
                    .setOutputFacialTransformationMatrixes(true)
                    .build()
            )
            cameraExecutor?.shutdown()
            cameraExecutor = Executors.newSingleThreadExecutor()
            val providerFuture = ProcessCameraProvider.getInstance(this)
            providerFuture.addListener({
                try {
                    val provider = providerFuture.get()
                    cameraProvider = provider
                    val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                    val stillCapture = ImageCapture.Builder()
                        .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                        .setResolutionSelector(
                            ResolutionSelector.Builder()
                                .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                                .build()
                        )
                        .setTargetRotation(previewView.display?.rotation ?: android.view.Surface.ROTATION_0)
                        .build()
                    imageCapture = stillCapture
                    val analysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()
                    analysis.setAnalyzer(cameraExecutor!!) { image ->
                        try {
                            val bitmap = imageToBitmap(image)
                            val rotated = if (image.imageInfo.rotationDegrees != 0) {
                                val matrix = Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }
                                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true).also {
                                    if (it !== bitmap) bitmap.recycle()
                                }
                            } else bitmap
                            try {
                                frameWidth = rotated.width
                                frameHeight = rotated.height
                                val mpImage = BitmapImageBuilder(rotated).build()
                                try {
                                    val now = System.nanoTime() / 1_000_000L
                                    val timestamp = lastTimestamp.updateAndGet { old -> maxOf(now, old + 1L) }
                                    val result = landmarker?.detectForVideo(mpImage, timestamp)
                                    val normalized = result?.faceLandmarks()?.firstOrNull()
                                        ?.map { PointF(it.x(), it.y()) } ?: emptyList()
                                    runOnUiThread {
                                        if (!designMode) {
                                            val mapped = normalized.map { mapToPreview(it) }
                                            browView.setFacePoints(mapped)
                                            status.text = if (mapped.size >= 400) {
                                                "已识别面部关键点 · 请让顾客自然睁眼、正视镜头"
                                            } else {
                                                "请正对前置摄像头，避免强逆光"
                                            }
                                        }
                                    }
                                } finally {
                                    mpImage.close()
                                }
                            } finally {
                                if (!rotated.isRecycled) rotated.recycle()
                            }
                        } catch (_: Exception) {
                            // Skip a broken camera frame. The next frame will be analyzed.
                        } finally {
                            image.close()
                        }
                    }
                    provider.unbindAll()
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis, stillCapture)
                    status.text = "摄像头已启动，请将脸放在画面中央"
                } catch (e: Exception) {
                    status.text = "摄像头启动失败，请检查权限后重试"
                }
            }, ContextCompat.getMainExecutor(this))
        } catch (e: Exception) {
            status.text = "人脸识别初始化失败：" + (e.message ?: "请重启应用")
        }
    }

    private fun mapToPreview(p: PointF): PointF {
        val vw = max(1, browView.width).toFloat()
        val vh = max(1, browView.height).toFloat()
        val iw = max(1, frameWidth).toFloat()
        val ih = max(1, frameHeight).toFloat()
        val scale = max(vw / iw, vh / ih)
        val displayW = iw * scale
        val displayH = ih * scale
        val offsetX = (vw - displayW) / 2f
        val offsetY = (vh - displayH) / 2f
        val mirroredX = iw * (1f - p.x)
        return PointF(offsetX + mirroredX * scale, offsetY + p.y * ih * scale)
    }

    private fun imageToBitmap(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer.duplicate()
        val width = image.width
        val height = image.height
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val i = y * plane.rowStride + x * plane.pixelStride
                val r = buffer.get(i).toInt() and 0xff
                val g = buffer.get(i + 1).toInt() and 0xff
                val b = buffer.get(i + 2).toInt() and 0xff
                val a = buffer.get(i + 3).toInt() and 0xff
                pixels[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun captureForDesign() {
        if (captureInProgress) return
        val anonymousCode = EditText(this).apply {
            hint = "匿名顾客代码，同一顾客重复到店请沿用"
            setSingleLine(true)
            setText(currentCaseId)
        }
        val consentText = TextView(this).apply {
            text = "请先向顾客说明并取得同意。照片会在本机内存中用于眉形设计，当前原型不会上传照片。保存效果图或导出原始标注照片会将含人脸的图片写入相册。偏好评分和曲率数据使用下方同一匿名代码保存在本机，取得相应授权后可导出作算法研究。"
            textSize = 13f
            setPadding(dp(20), dp(4), dp(20), dp(8))
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(consentText)
            addView(anonymousCode, LinearLayout.LayoutParams(-1, -2).apply {
                setMargins(dp(20), dp(2), dp(20), dp(2))
            })
            addView(TextView(this@MainActivity).apply {
                text = "新顾客可使用新的随机代码。同一顾客跨次测量需由纹绣师在合规的本地记录中沿用同一个匿名代码，不要输入姓名、电话或预约号。"
                textSize = 11f
                setPadding(dp(20), dp(4), dp(20), dp(8))
            })
        }
        AlertDialog.Builder(this)
            .setTitle("顾客拍摄与数据使用确认")
            .setView(form)
            .setNegativeButton("取消", null)
            .setNeutralButton("随机新代码") { _, _ ->
                currentCaseId = UUID.randomUUID().toString().take(8)
                Toast.makeText(this, "已生成新的匿名代码：$currentCaseId，请再次点击拍照设计", Toast.LENGTH_LONG).show()
            }
            .setPositiveButton("同意并拍照") { _, _ ->
                val safeCode = anonymousCode.text.toString().trim()
                    .filter { it.isLetterOrDigit() || it == '_' || it == '-' }
                    .take(32)
                if (safeCode.isBlank()) {
                    Toast.makeText(this, "请输入匿名代码或先生成随机代码", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                currentCaseId = safeCode
                captureHighResolutionPhoto()
            }
            .show()
    }

    private fun captureHighResolutionPhoto() {
        if (captureInProgress) return
        val livePoints = browView.facePoints()
        if (livePoints.size < 400) {
            Toast.makeText(this, "尚未稳定识别人脸，请正视镜头后再拍照", Toast.LENGTH_SHORT).show()
            return
        }
        val stillCapture = imageCapture
        if (stillCapture == null) {
            Toast.makeText(this, "相机尚未准备好，请稍后重试", Toast.LENGTH_SHORT).show()
            return
        }
        captureInProgress = true
        status.text = "正在拍摄高分辨率照片并校验人脸关键点…"
        val file = File(cacheDir, "brow_capture_${System.currentTimeMillis()}.jpg")
        val output = ImageCapture.OutputFileOptions.Builder(file).build()
        stillCapture.takePicture(
            output,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                    if (imageExecutor == null || imageExecutor?.isShutdown != false) {
                        imageExecutor = Executors.newSingleThreadExecutor()
                    }
                    imageExecutor?.execute {
                        var bitmap: Bitmap? = null
                        var points: List<PointF> = emptyList()
                        var depths: List<Float> = emptyList()
                        var failure: String? = null
                        var detector: FaceLandmarker? = null
                        var mpImage: com.google.mediapipe.framework.image.MPImage? = null
                        try {
                            bitmap = decodeCapturedPhoto(file)
                            if (bitmap == null) {
                                failure = "无法读取拍摄照片"
                            } else {
                                detector = FaceLandmarker.createFromOptions(
                                    this@MainActivity,
                                    FaceLandmarker.FaceLandmarkerOptions.builder()
                                        .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                                        .setRunningMode(RunningMode.IMAGE)
                                        .setNumFaces(1)
                                        .build()
                                )
                                mpImage = BitmapImageBuilder(bitmap).build()
                                val result = detector.detect(mpImage)
                                val detected = result.faceLandmarks().firstOrNull()
                                points = detected?.map {
                                    PointF(it.x() * bitmap!!.width, it.y() * bitmap!!.height)
                                } ?: emptyList()
                                depths = detected?.map { it.z() } ?: emptyList()
                                if (points.size < 400) failure = "照片中的人脸关键点不足，请正视镜头并保持眉眼无遮挡"
                            }
                        } catch (e: Exception) {
                            failure = "照片分析失败：" + (e.message ?: "请调整姿势后重试")
                        } finally {
                            try { mpImage?.close() } catch (_: Exception) {}
                            try { detector?.close() } catch (_: Exception) {}
                            file.delete()
                        }
                        runOnUiThread {
                            captureInProgress = false
                            val photo = bitmap
                            if (failure != null || photo == null) {
                                photo?.recycle()
                                status.text = failure ?: "拍摄失败，请重试"
                                Toast.makeText(this@MainActivity, status.text, Toast.LENGTH_LONG).show()
                            } else {
                                val warning = photoQualityWarning(photo, points)
                                if (warning == null) {
                                    enterDesignMode(photo, points, depths)
                                } else {
                                    AlertDialog.Builder(this@MainActivity)
                                        .setTitle("拍摄质量提醒")
                                        .setMessage(warning + "\n\n继续使用可能增加眉形定位和打印拓描误差。")
                                        .setNegativeButton("重新拍摄") { _, _ ->
                                            photo.recycle()
                                            retakePhoto()
                                        }
                                        .setPositiveButton("仍然使用") { _, _ -> enterDesignMode(photo, points, depths) }
                                        .show()
                                }
                            }
                        }
                    } ?: run {
                        captureInProgress = false
                        file.delete()
                        Toast.makeText(this@MainActivity, "照片分析线程不可用，请重试", Toast.LENGTH_LONG).show()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    captureInProgress = false
                    file.delete()
                    status.text = "拍照失败：" + (exception.message ?: "请重试")
                    Toast.makeText(this@MainActivity, status.text, Toast.LENGTH_LONG).show()
                }
            }
        )
    }

    private fun decodeCapturedPhoto(file: File): Bitmap? {
        val decoded = BitmapFactory.decodeFile(file.absolutePath) ?: return null
        val exif = try { ExifInterface(file.absolutePath) } catch (_: Exception) { null }
        val orientation = exif?.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        ) ?: ExifInterface.ORIENTATION_NORMAL
        val transform = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> transform.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> transform.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> transform.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> transform.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> transform.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> {
                transform.postRotate(90f)
                transform.postScale(-1f, 1f)
            }
            ExifInterface.ORIENTATION_TRANSVERSE -> {
                transform.postRotate(270f)
                transform.postScale(-1f, 1f)
            }
        }
        val oriented = if (orientation == ExifInterface.ORIENTATION_NORMAL) decoded else {
            Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, transform, true).also {
                if (it !== decoded) decoded.recycle()
            }
        }
        // Match the front-camera mirror shown to the customer and the design overlay.
        val mirror = Matrix().apply { postScale(-1f, 1f, oriented.width / 2f, oriented.height / 2f) }
        return Bitmap.createBitmap(oriented, 0, 0, oriented.width, oriented.height, mirror, true).also {
            if (it !== oriented) oriented.recycle()
        }
    }

    private fun photoQualityWarning(photo: Bitmap, points: List<PointF>): String? {
        if (points.size <= 454) return "关键点不足，无法确认照片质量。"
        val leftCheek = points[234]
        val rightCheek = points[454]
        val faceWidth = hypot((rightCheek.x - leftCheek.x).toDouble(), (rightCheek.y - leftCheek.y).toDouble()).toFloat()
        val centerX = (leftCheek.x + rightCheek.x) / 2f
        val warnings = mutableListOf<String>()
        if (faceWidth < photo.width * 0.28f) warnings.add("脸部在照片中偏小，请让顾客靠近一些后重新拍摄")
        if (abs(centerX - photo.width / 2f) > photo.width * 0.16f) warnings.add("脸部没有位于画面中央")
        if (points.size > 263) {
            val leftEye = points[33]
            val rightEye = points[263]
            val rollDegrees = Math.toDegrees(kotlin.math.atan2(
                (rightEye.y - leftEye.y).toDouble(),
                (rightEye.x - leftEye.x).toDouble()
            )).toFloat()
            if (abs(rollDegrees) > 12f) warnings.add("头部有明显倾斜，请尽量让双眼连线接近水平")
        }
        // Sample the face region sparsely so exposure/contrast checks remain inexpensive
        // even on high-resolution photos.
        val faceLeft = min(leftCheek.x, rightCheek.x).toInt().coerceIn(0, photo.width - 1)
        val faceRight = max(leftCheek.x, rightCheek.x).toInt().coerceIn(faceLeft + 1, photo.width)
        val foreheadY = points.getOrNull(10)?.y ?: 0f
        val chinY = points.getOrNull(152)?.y ?: photo.height.toFloat()
        val faceTop = min(foreheadY, chinY).toInt().coerceIn(0, photo.height - 1)
        val faceBottom = max(foreheadY, chinY).toInt().coerceIn(faceTop + 1, photo.height)
        val stepX = max(4, (faceRight - faceLeft) / 36)
        val stepY = max(4, (faceBottom - faceTop) / 36)
        var lumaSum = 0.0
        var lumaSquareSum = 0.0
        var sampleCount = 0
        var gradientSum = 0.0
        var gradientCount = 0
        fun lumaAt(x: Int, y: Int): Double {
            val pixel = photo.getPixel(x.coerceIn(0, photo.width - 1), y.coerceIn(0, photo.height - 1))
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            val blue = Color.blue(pixel)
            return 0.299 * red + 0.587 * green + 0.114 * blue
        }
        var y = faceTop
        while (y < faceBottom) {
            var x = faceLeft
            while (x < faceRight) {
                val luma = lumaAt(x, y)
                lumaSum += luma
                lumaSquareSum += luma * luma
                sampleCount++
                if (x + stepX < faceRight) {
                    gradientSum += abs(luma - lumaAt(x + stepX, y))
                    gradientCount++
                }
                if (y + stepY < faceBottom) {
                    gradientSum += abs(luma - lumaAt(x, y + stepY))
                    gradientCount++
                }
                x += stepX
            }
            y += stepY
        }
        if (sampleCount > 0) {
            val meanLuma = lumaSum / sampleCount
            val variance = max(0.0, lumaSquareSum / sampleCount - meanLuma * meanLuma)
            val contrast = kotlin.math.sqrt(variance)
            val meanGradient = if (gradientCount > 0) gradientSum / gradientCount else 0.0
            if (meanLuma < 58.0) warnings.add("面部整体偏暗，建议使用正面柔和补光后重拍")
            if (meanLuma > 218.0) warnings.add("面部可能过曝，请降低强光或调整曝光")
            if (contrast < 15.0 && meanGradient < 4.0) warnings.add("面部细节或对比度不足，请检查对焦与光线")
        }
        return warnings.takeIf { it.isNotEmpty() }?.joinToString("\n")
    }

    private fun refreshRecommendations(points: List<PointF>) {
        recommendationRow.removeAllViews()
        val ranked = BrowRecommendationEngine.rank(this, points)
        recommendationLabel.text = BrowRecommendationEngine.sourceLabel(this)
        ranked.forEach { recommendation ->
            val button = Button(this).apply {
                text = if (BrowRecommendationEngine.hasApprovedModel(this@MainActivity)) {\n                    "${recommendation.name}\\n偏好估计 ${recommendation.score}%"\n                } else {\n                    "${recommendation.name}\\n规则参考分 ${recommendation.score}"\n                }
                textSize = 11f
                minHeight = dp(52)
                setPadding(dp(2), 0, dp(2), 0)
                contentDescription = recommendation.reason
                setOnClickListener {
                    Toast.makeText(this@MainActivity, recommendation.reason, Toast.LENGTH_SHORT).show()
                    animateTo(recommendation.params)
                }
            }
            recommendationRow.addView(button, LinearLayout.LayoutParams(0, dp(54), 1f).apply {
                setMargins(dp(2), dp(2), dp(2), dp(4))
            })
        }
    }

    private fun enterDesignMode(
        photo: Bitmap,
        photoPoints: List<PointF>,
        normalizedDepths: List<Float> = emptyList()
    ) {
        stopTransition()
        segmentationGeneration++
        savedPlans.forEach { it.segmentationSnapshot?.overlay?.recycle() }
        savedPlans.clear()
        if (::plansRow.isInitialized) refreshPlanButtons()
        segmentationResult?.overlay?.recycle()
        segmentationResult = null
        showNativeSegmentation = false
        useNativeShape = false
        browView.setNativeShapeMode(false)
        if (::segmentationLabel.isInitialized) segmentationLabel.text = "原生眉区识别：尚未运行"
        // Do not carry one customer's candidate plans into the next customer's session.
        savedPlans.clear()
        if (::plansRow.isInitialized) refreshPlanButtons()
        capturedPhoto?.recycle()
        capturedPhoto = photo
        capturedLandmarkDepths = if (normalizedDepths.size == photoPoints.size &&
            normalizedDepths.all { it.isFinite() }) normalizedDepths.map { it * photo.width } else emptyList()
        designMode = true
        browView.setPhoto(photo)
        browView.setFacePoints(photoPoints)
        previewView.visibility = View.GONE
        designControls.visibility = View.VISIBLE
        liveControls.visibility = View.GONE
        params = BrowParams()
        history.clear()
        browView.setParams(params)
        syncSliders()
        refreshRecommendations(photoPoints)
        cameraProvider?.unbindAll()
        imageCapture = null
        landmarker?.close()
        landmarker = null
        cameraExecutor?.shutdown()
        cameraExecutor = null
        status.text = "高分辨率照片已分析，可调整眉形并保存多个方案"
    }

    private fun retakePhoto() {
        stopTransition()
        segmentationGeneration++
        segmentationResult?.overlay?.recycle()
        segmentationResult = null
        showNativeSegmentation = false
        useNativeShape = false
        browView.setNativeShapeMode(false)
        if (::segmentationLabel.isInitialized) segmentationLabel.text = "原生眉区识别：尚未运行"
        browView.setSegmentationResult(null, false)
        designMode = false
        capturedPhoto?.recycle()
        capturedPhoto = null
        capturedLandmarkDepths = emptyList()
        browView.setPhoto(null)
        browView.setParams(params)
        previewView.visibility = View.VISIBLE
        designControls.visibility = View.GONE
        liveControls.visibility = View.VISIBLE
        browView.setFacePoints(emptyList())
        status.text = "重新拍摄中，请让顾客自然正视镜头"
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startFaceTracking()
        }
    }

    private fun animateTo(target: BrowParams) {
        history.add(params.copy())
        stopTransition()
        val from = params
        val animator = ValueAnimator.ofFloat(0f, 1f).apply {
            duration = 900L
            interpolator = LinearInterpolator()
            addUpdateListener {
                val t = it.animatedValue as Float
                params = BrowParams(
                    from.thickness + (target.thickness - from.thickness) * t,
                    from.length + (target.length - from.length) * t,
                    from.arch + (target.arch - from.arch) * t,
                    if (t < 0.5f) from.style else target.style
                )
                browView.setParams(params)
                syncSliders()
            }
            addListener(object : AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: Animator) {
                    if (transitionAnimator === animation) {
                        transitionAnimator = null
                        pauseButton.text = "暂停 / 继续"
                    }
                }
            })
        }
        transitionAnimator = animator
        pauseButton.text = "暂停"
        animator.start()
    }

    private fun toggleTransition() {
        val animator = transitionAnimator
        if (animator == null) {
            Toast.makeText(this, "先选择一种眉形，渐变预览才会启动", Toast.LENGTH_SHORT).show()
        } else if (animator.isPaused) {
            animator.resume()
            pauseButton.text = "暂停"
        } else if (animator.isRunning) {
            animator.pause()
            pauseButton.text = "继续"
        }
    }

    private fun stopTransition() {
        transitionAnimator?.cancel()
        transitionAnimator = null
        if (::pauseButton.isInitialized) pauseButton.text = "暂停 / 继续"
    }

    private fun undoAdjustment() {
        stopTransition()
        if (history.isEmpty()) {
            Toast.makeText(this, "没有可回撤的调整", Toast.LENGTH_SHORT).show()
            return
        }
        params = history.removeAt(history.lastIndex)
        browView.setParams(params)
        syncSliders()
    }

    private fun syncSliders() {
        if (!::thicknessSeek.isInitialized) return
        thicknessSeek.progress = (((params.thickness - 0.55f) / 0.90f) * 100f).toInt().coerceIn(0, 100)
        lengthSeek.progress = (((params.length - 0.80f) / 0.40f) * 100f).toInt().coerceIn(0, 100)
        archSeek.progress = (((params.arch - 0.55f) / 0.90f) * 100f).toInt().coerceIn(0, 100)
    }

    private fun savePlan() {
        if (capturedPhoto == null) return
        if (savedPlans.size >= 5) {
            Toast.makeText(this, "最多保存 5 套方案，请先对比并移除旧方案后再继续", Toast.LENGTH_LONG).show()
            return
        }
        val input = EditText(this).apply {
            hint = "例如：自然款 A"
            setSingleLine(true)
            setText("方案 " + (savedPlans.size + 1))
        }
        AlertDialog.Builder(this)
            .setTitle("保存当前眉形")
            .setView(input)
            .setNegativeButton("取消", null)
            .setPositiveButton("保存") { _, _ ->
                val name = input.text.toString().trim().ifBlank { "方案 " + (savedPlans.size + 1) }
                val maskSnapshot = segmentationResult?.let { result ->
                    if (result.overlay.isRecycled) null
                    else result.copy(overlay = result.overlay.copy(Bitmap.Config.ARGB_8888, true))
                }
                savedPlans.add(BrowPlan(name, params.copy(), useNativeShape, maskSnapshot))
                refreshPlanButtons()
                status.text = "已保存 " + name + "，可以继续调整并保存其他版本"
            }
            .show()
    }

    private fun applyPlan(plan: BrowPlan) {
        stopTransition()
        history.add(params.copy())
        segmentationResult?.overlay?.recycle()
        segmentationResult = plan.segmentationSnapshot?.let { snapshot ->
            if (snapshot.overlay.isRecycled) null
            else snapshot.copy(overlay = snapshot.overlay.copy(Bitmap.Config.ARGB_8888, true))
        }
        showNativeSegmentation = false
        useNativeShape = plan.useNativeShape && segmentationResult != null
        browView.setSegmentationResult(segmentationResult, false)
        browView.setNativeShapeMode(useNativeShape)
        params = plan.params.copy()
        browView.setParams(params)
        syncSliders()
        status.text = if (useNativeShape) "正在预览：${plan.name}（原生眉形轮廓）" else "正在预览：" + plan.name
    }

    private fun refreshPlanButtons() {
        plansRow.removeAllViews()
        if (savedPlans.isEmpty()) {
            plansRow.visibility = View.GONE
            return
        }
        plansRow.visibility = View.VISIBLE
        savedPlans.forEachIndexed { index, plan ->
            val b = Button(this).apply {
                text = plan.name
                textSize = 11f
                setOnClickListener { applyPlan(plan) }
                setOnLongClickListener {
                    AlertDialog.Builder(this@MainActivity)
                        .setTitle("移除方案")
                        .setMessage("确定删除“${plan.name}”吗？")
                        .setNegativeButton("取消", null)
                        .setPositiveButton("删除") { _, _ ->
                            if (index in savedPlans.indices) {
                                savedPlans.removeAt(index).segmentationSnapshot?.overlay?.recycle()
                                refreshPlanButtons()
                                status.text = "已删除方案，可继续保存其他版本"
                            }
                        }
                        .show()
                    true
                }
            }
            plansRow.addView(b, LinearLayout.LayoutParams(0, dp(40), 1f).apply {
                setMargins(dp(2), dp(2), dp(2), dp(2))
            })
        }
    }

    private fun showComparison() {
        val photo = capturedPhoto
        val points = browView.facePoints()
        if (photo == null || savedPlans.isEmpty()) {
            Toast.makeText(this, "请先保存至少一套方案", Toast.LENGTH_SHORT).show()
            return
        }
        val horizontal = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8), dp(8), dp(8), dp(8))
        }
        savedPlans.forEach { plan ->
            val column = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(dp(4), dp(4), dp(4), dp(4))
            }
            val image = ImageView(this).apply {
                setImageBitmap(BrowRenderer.renderPreview(
                    photo, points, plan.params, 300, plan.segmentationSnapshot, plan.useNativeShape
                ))
                scaleType = ImageView.ScaleType.FIT_CENTER
                contentDescription = plan.name
            }
            column.addView(image, LinearLayout.LayoutParams(dp(205), dp(245)))
            column.addView(TextView(this).apply {
                text = plan.name
                textSize = 13f
                setTextColor(Color.WHITE)
                gravity = Gravity.CENTER
            })
            column.addView(Button(this).apply {
                text = "选用"
                setOnClickListener { applyPlan(plan) }
            })
            horizontal.addView(column, LinearLayout.LayoutParams(dp(220), -2))
        }
        val scroller = HorizontalScrollView(this).apply {
            addView(horizontal)
            setBackgroundColor(0xFF181A20.toInt())
        }
        AlertDialog.Builder(this)
            .setTitle("方案对比")
            .setView(scroller)
            .setPositiveButton("完成", null)
            .show()
    }

    private fun exportOriginalPhotoForAnnotation() {
        val photo = capturedPhoto ?: run {
            Toast.makeText(this, "请先拍照", Toast.LENGTH_SHORT).show()
            return
        }
        AlertDialog.Builder(this)
            .setTitle("导出原始人脸照片")
            .setMessage("该操作会将未叠加眉形的完整人脸照片保存到手机相册，供你在获得单独明确授权后进行本地人工标注和训练。照片可识别顾客身份，请勿未经授权上传、共享或用于其他用途。")
            .setNegativeButton("取消", null)
            .setPositiveButton("保存原始照片") { _, _ ->
                val filename = "${currentCaseId}_source_${System.currentTimeMillis()}.png"
                var uri: android.net.Uri? = null
                var output: java.io.OutputStream? = null
                try {
                    val values = ContentValues().apply {
                        put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                        put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/BrowDesign/AnnotationSource")
                        put(MediaStore.Images.Media.IS_PENDING, 1)
                    }
                    uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                        ?: throw IllegalStateException("无法创建原始照片文件")
                    output = contentResolver.openOutputStream(uri)
                        ?: throw IllegalStateException("无法写入照片文件")
                    if (!photo.compress(Bitmap.CompressFormat.PNG, 100, output)) {
                        throw IllegalStateException("照片编码失败")
                    }
                    output.close()
                    output = null
                    values.clear()
                    values.put(MediaStore.Images.Media.IS_PENDING, 0)
                    contentResolver.update(uri, values, null, null)
                    Toast.makeText(this, "原始标注照片已保存到 Pictures/BrowDesign/AnnotationSource", Toast.LENGTH_LONG).show()
                } catch (e: Exception) {
                    if (uri != null) contentResolver.delete(uri, null, null)
                    Toast.makeText(this, "导出失败：" + (e.message ?: "请重试"), Toast.LENGTH_LONG).show()
                } finally {
                    try { output?.close() } catch (_: Exception) {}
                }
            }
            .show()
    }

    private fun savePreviewToGallery() {
        val photo = capturedPhoto
        if (photo == null) {
            Toast.makeText(this, "请先拍摄并设计眉形", Toast.LENGTH_SHORT).show()
            return
        }
        val preview = try {
            BrowRenderer.renderPreview(
                photo, browView.facePoints(), params, photo.width, segmentationResult, useNativeShape
            )
        } catch (e: Exception) {
            Toast.makeText(this, "生成效果图失败：" + (e.message ?: "请重试"), Toast.LENGTH_LONG).show()
            return
        }
        val filename = "眉形方案_${System.currentTimeMillis()}.png"
        var output: java.io.OutputStream? = null
        var uri: android.net.Uri? = null
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                    put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/BrowDesign")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
                uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                if (uri == null) throw IllegalStateException("无法创建图片文件")
                output = contentResolver.openOutputStream(uri) ?: throw IllegalStateException("无法写入图片文件")
                if (!preview.compress(Bitmap.CompressFormat.PNG, 100, output)) throw IllegalStateException("图片编码失败")
                output.close()
                output = null
                values.clear()
                values.put(MediaStore.Images.Media.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
                Toast.makeText(this, "效果图已保存到相册的 Pictures/BrowDesign", Toast.LENGTH_LONG).show()
            } else {
                val directory = File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "BrowDesign")
                if (!directory.exists() && !directory.mkdirs()) throw IllegalStateException("无法创建保存目录")
                val file = File(directory, filename)
                FileOutputStream(file).use { stream ->
                    if (!preview.compress(Bitmap.CompressFormat.PNG, 100, stream)) throw IllegalStateException("图片编码失败")
                }
                Toast.makeText(this, "效果图已保存到应用图片目录：" + file.absolutePath, Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            if (uri != null) contentResolver.delete(uri, null, null)
            Toast.makeText(this, "保存失败：" + (e.message ?: "请重试"), Toast.LENGTH_LONG).show()
        } finally {
            try { output?.close() } catch (_: Exception) {}
            preview.recycle()
        }
    }

    private fun showCurvatureCalibration() {
        val field = EditText(this).apply {
            hint = "表面弧长实测值（毫米）"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
        }
        val chord = EditText(this).apply {
            hint = "同一段两端直线距离（毫米）"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER or android.text.InputType.TYPE_NUMBER_FLAG_DECIMAL
        }
        val region = android.widget.Spinner(this).apply {
            adapter = android.widget.ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("左眉头", "左眉身", "左眉峰", "左眉尾", "右眉头", "右眉身", "右眉峰", "右眉尾")
            )
        }
        val caseIdField = EditText(this).apply {
            hint = "本次匿名顾客代码"
            setSingleLine(true)
            setText(currentCaseId)
            isEnabled = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(TextView(this@MainActivity).apply {
                text = "用于建立个体和区域的实测曲率数据。用柔性尺测量皮肤表面弧长，同时记录同一段两端的直线距离。不要用照片像素代替实测毫米。系统还会按局部圆弧假设估计半径，该估计不是三维面部重建结果。"
                textSize = 13f
            })
            addView(caseIdField)
            addView(region)
            addView(chord)
            addView(field)
        }
        AlertDialog.Builder(this)
            .setTitle("眉弓曲率标定记录")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("计算并保存") { _, _ ->
                val arcMm = field.text.toString().toFloatOrNull()
                val chordMm = chord.text.toString().toFloatOrNull()
                if (arcMm == null || chordMm == null || chordMm <= 0f || arcMm < chordMm || arcMm > chordMm * 1.5708f) {
                    Toast.makeText(this, "请检查实测值。单段圆弧模型要求弧长不小于弦长，且弧长/弦长不超过 π/2。", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val factor = arcMm / chordMm
                val estimatedRadiusMm = estimateCircularArcRadius(chordMm, arcMm)
                val prefs = getSharedPreferences("brow_curvature_samples", Context.MODE_PRIVATE)
                val old = org.json.JSONArray(prefs.getString("records", "[]"))
                val enteredCaseId = caseIdField.text.toString().trim().ifBlank { currentCaseId.take(8) }
                val item = org.json.JSONObject().apply {
                    put("case_id", enteredCaseId)
                    put("region", region.selectedItem.toString())
                    put("arc_mm", arcMm.toDouble())
                    put("chord_mm", chordMm.toDouble())
                    put("factor", factor.toDouble())
                    put("estimated_radius_mm", estimatedRadiusMm?.toDouble() ?: org.json.JSONObject.NULL)
                    put("timestamp", System.currentTimeMillis())
                }
                old.put(item)
                prefs.edit().putString("records", old.toString()).apply()
                val samples = (0 until old.length()).mapNotNull { index ->
                    old.optJSONObject(index)?.takeIf { it.optString("region") == region.selectedItem.toString() }
                        ?.optDouble("factor")?.takeIf { it.isFinite() }
                }
                val interval = if (samples.size >= 5) {
                    val sorted = samples.sorted()
                    val low = sorted[((sorted.size - 1) * 0.10).toInt()]
                    val high = sorted[((sorted.size - 1) * 0.90).toInt()]
                    "该区域已有 ${samples.size} 条记录，样本 10%–90% 分位区间约为 ${"%.3f".format(low)}–${"%.3f".format(high)}。这只是本机样本的经验区间，不代表人群通用区间。"
                } else {
                    "该区域已记录 ${samples.size} 条样本。建议至少采集 30–50 个不同顾客的标准化记录，再评估区间稳定性。"
                }
                val radiusText = if (estimatedRadiusMm == null) {
                    "半径估计不稳定，可能接近平坦"
                } else {
                    "局部圆弧近似半径 R≈${"%.1f".format(estimatedRadiusMm)} mm，曲率约 ${"%.5f".format(1f / estimatedRadiusMm)} mm⁻¹"
                }
                Toast.makeText(this, "实测修正系数 k = ${"%.3f".format(factor)}。\n" + radiusText + "\n" + interval, Toast.LENGTH_LONG).show()
            }
            .setNeutralButton("导出 CSV") { _, _ -> exportCurvatureCsv() }
            .show()
    }

    private fun estimateCircularArcRadius(chordMm: Float, arcMm: Float): Float? {
        val ratio = arcMm / chordMm
        if (ratio <= 1.00005f) return null
        var low = 0.0001
        var high = Math.PI
        repeat(80) {
            val theta = (low + high) / 2.0
            val value = theta / (2.0 * kotlin.math.sin(theta / 2.0))
            if (value < ratio.toDouble()) low = theta else high = theta
        }
        val theta = (low + high) / 2.0
        val denominator = 2.0 * kotlin.math.sin(theta / 2.0)
        if (denominator <= 0.0) return null
        val radius = chordMm / denominator
        if (!radius.isFinite() || radius > 1_000_000.0) return null
        return radius.toFloat()
    }

    private fun exportCurvatureCsv() {
        val records = org.json.JSONArray(getSharedPreferences("brow_curvature_samples", Context.MODE_PRIVATE).getString("records", "[]"))
        if (records.length() == 0) {
            Toast.makeText(this, "暂时没有曲率标定记录", Toast.LENGTH_SHORT).show()
            return
        }
        val csv = buildString {
            append("case_id,region,arc_mm,chord_mm,correction_factor,estimated_radius_mm,timestamp\n")
            for (i in 0 until records.length()) {
                val row = records.optJSONObject(i) ?: continue
                append(row.optString("case_id")).append(',')
                append(row.optString("region")).append(',')
                append(row.optDouble("arc_mm")).append(',')
                append(row.optDouble("chord_mm")).append(',')
                val radius = row.optDouble("estimated_radius_mm", Double.NaN)
                append(row.optDouble("factor")).append(',')
                append(if (radius.isFinite()) radius else "").append(',')
                append(row.optLong("timestamp")).append('\n')
            }
        }
        val filename = "brow_curvature_calibration_${System.currentTimeMillis()}.csv"
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, filename)
                    put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/BrowDesign")
                    put(MediaStore.Downloads.IS_PENDING, 1)
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: throw IllegalStateException("无法创建 CSV 文件")
                contentResolver.openOutputStream(uri)?.use { it.write(csv.toByteArray(Charsets.UTF_8)) }
                    ?: throw IllegalStateException("无法写入 CSV")
                values.clear()
                values.put(MediaStore.Downloads.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
                Toast.makeText(this, "曲率数据已导出到 Downloads/BrowDesign", Toast.LENGTH_LONG).show()
            } else {
                val directory = File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "BrowDesign")
                if (!directory.exists()) directory.mkdirs()
                val file = File(directory, filename)
                file.writeText(csv, Charsets.UTF_8)
                Toast.makeText(this, "曲率数据已导出：" + file.absolutePath, Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "导出失败：" + (e.message ?: "请重试"), Toast.LENGTH_LONG).show()
        }
    }

    private fun toggleNativeSegmentation() {
        if (capturedPhoto == null) {
            Toast.makeText(this, "请先拍照进入设计模式", Toast.LENGTH_SHORT).show()
            return
        }
        showNativeSegmentation = true
        val existing = segmentationResult
        if (existing != null) {
            browView.setSegmentationResult(existing, true)
            segmentationLabel.text = "识别方式：${existing.method} · 掩膜质量提示 ${(existing.maskQualityScore * 100).toInt()}%（启发式指标，非模型置信概率）· 请检查识别边界"
        } else {
            runSegmentation()
        }
    }

    private fun toggleNativeShapeDesign(forceOff: Boolean = false) {
        if (forceOff) {
            useNativeShape = false
            browView.setNativeShapeMode(false)
            status.text = "已切换到参数化眉形底稿"
            return
        }
        val result = segmentationResult
        if (result == null) {
            Toast.makeText(this, "请先识别原生眉区，并检查/修正识别结果", Toast.LENGTH_LONG).show()
            toggleNativeSegmentation()
            return
        }
        if (!browView.hasUsableNativeContour()) {
            segmentationLabel.text = "当前掩膜没有提取出连续眉形轮廓，请调整灵敏度或手工补画"
            Toast.makeText(this, "没有可靠的眉形轮廓，请先修正绿色识别层", Toast.LENGTH_LONG).show()
            showNativeSegmentation = true
            browView.setSegmentationResult(result, true)
            return
        }
        useNativeShape = true
        showNativeSegmentation = false
        browView.setSegmentationEditMode(false)
        browView.setSegmentationResult(result, false)
        browView.setNativeShapeMode(true)
        segmentationLabel.text = "正在使用原生眉区轮廓生成底稿。请调整后人工确认边界。"
        status.text = "已切换到原生眉形轮廓底稿，可微调粗细、长度和弧度"
    }

    private fun runSegmentation() {
        val photo = capturedPhoto ?: return
        val points = browView.facePoints()
        if (points.size < 455) {
            segmentationLabel.text = "关键点不足，不能生成眉区识别结果"
            return
        }
        if (imageExecutor == null || imageExecutor?.isShutdown != false) {
            imageExecutor = Executors.newSingleThreadExecutor()
        }
        val generation = ++segmentationGeneration
        segmentationLabel.text = "正在识别原生眉区…"
        imageExecutor?.execute {
            val result = try {
                BrowSegmentationEngine.segment(this, photo, points, segmentationSensitivity)
            } catch (_: Exception) {
                null
            }
            runOnUiThread {
                if (generation != segmentationGeneration || capturedPhoto !== photo) {
                    result?.overlay?.recycle()
                    return@runOnUiThread
                }
                segmentationResult?.overlay?.recycle()
                segmentationResult = result
                if (result == null) {
                    segmentationLabel.text = "识别失败，请调整光线或重新拍摄"
                    browView.setSegmentationResult(null, false)
                    showNativeSegmentation = false
                } else {
                    browView.setSegmentationResult(result, showNativeSegmentation)
                    browView.setNativeShapeMode(useNativeShape)
                    segmentationLabel.text = "识别方式：${result.method} · 掩膜质量提示 ${(result.maskQualityScore * 100).toInt()}%（按像素占比估算，非模型置信概率）· 识别像素 ${result.selectedPixelCount} · 请人工检查边界"
                }
            }
        }
    }

    private fun recordPreference() {
        val points = browView.facePoints()
        if (capturedPhoto == null || points.size < 455) {
            Toast.makeText(this, "请先完成拍照和眉形设计", Toast.LENGTH_SHORT).show()
            return
        }
        val anonymousClientCode = EditText(this).apply {
            hint = "本次匿名顾客代码"
            isSingleLine = true
            setText(currentCaseId)
            isEnabled = false
        }
        val ratingLabel = TextView(this).apply {
            text = "顾客满意度：5 / 5"
            textSize = 15f
        }
        val ratingBar = SeekBar(this).apply {
            max = 4
            progress = 4
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    ratingLabel.text = "顾客满意度：${progress + 1} / 5"
                }
            })
        }
        val selected = CheckBox(this).apply {
            text = "顾客最终选择此方案"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), 0)
            addView(TextView(this@MainActivity).apply {
                text = "记录的是匿名几何特征、方案参数和顾客反馈，不保存照片或身份信息。需先获得顾客授权。"
                textSize = 13f
            })
            addView(anonymousClientCode)
            addView(ratingLabel)
            addView(ratingBar)
            addView(selected)
        }
        AlertDialog.Builder(this)
            .setTitle("记录顾客偏好")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("保存反馈") { _, _ ->
                val features = BrowRecommendationEngine.measurements(points)
                val dataset = org.json.JSONArray(
                    getSharedPreferences("brow_preference_feedback", Context.MODE_PRIVATE).getString("records", "[]")
                )
                val consentedAnonymousId = anonymousClientCode.text.toString().trim()
                    .filter { it.isLetterOrDigit() || it == '_' || it == '-' }
                    .take(32)
                    .ifBlank { currentCaseId }
                val item = org.json.JSONObject().apply {
                    put("case_id", consentedAnonymousId)
                    put("style", when (params.style) { 1 -> "soft"; 2 -> "defined"; else -> "natural" })
                    put("thickness", params.thickness.toDouble())
                    put("length", params.length.toDouble())
                    put("arch", params.arch.toDouble())
                    put("rating", ratingBar.progress + 1)
                    put("selected", selected.isChecked)
                    put("face_width_height_ratio", features.widthHeightRatio.toDouble())
                    put("interocular_face_width_ratio", features.interocularFaceWidthRatio.toDouble())
                    put("jaw_face_width_ratio", features.jawFaceWidthRatio.toDouble())
                    put("brow_eye_distance_ratio", features.browEyeDistanceRatio.toDouble())
                    put("brow_asymmetry_ratio", features.browAsymmetryRatio.toDouble())
                    put("eye_line_tilt", features.eyeLineTilt.toDouble())
                    put("timestamp", System.currentTimeMillis())
                }
                dataset.put(item)
                getSharedPreferences("brow_preference_feedback", Context.MODE_PRIVATE).edit()
                    .putString("records", dataset.toString())
                    .apply()
                Toast.makeText(this, "已保存匿名偏好记录，可导出 CSV 用于后续模型训练", Toast.LENGTH_LONG).show()
            }
            .show()
    }

    private fun showModelLicenseNotice() {
        val notice = try {
            assets.open("THIRD_PARTY_MODEL_NOTICES.txt").bufferedReader(Charsets.UTF_8).use { it.readText() }
        } catch (_: Exception) {
            "无法读取模型来源说明。请查阅项目仓库中的 docs/MODEL_PROVENANCE.md。"
        }
        val body = TextView(this).apply {
            text = notice
            textSize = 12f
            setTextIsSelectable(true)
            setPadding(dp(16), dp(8), dp(16), dp(8))
        }
        val scroll = ScrollView(this).apply {
            addView(body)
        }
        AlertDialog.Builder(this)
            .setTitle("模型来源与许可")
            .setView(scroll)
            .setPositiveButton("关闭", null)
            .show()
    }

    private fun exportPreferenceCsv() {
        val records = org.json.JSONArray(
            getSharedPreferences("brow_preference_feedback", Context.MODE_PRIVATE).getString("records", "[]")
        )
        if (records.length() == 0) {
            Toast.makeText(this, "暂无顾客偏好记录", Toast.LENGTH_SHORT).show()
            return
        }
        val csv = buildString {
            append("case_id,style,thickness,length,arch,rating,selected,face_width_height_ratio,interocular_face_width_ratio,jaw_face_width_ratio,brow_eye_distance_ratio,brow_asymmetry_ratio,eye_line_tilt,timestamp\n")
            for (i in 0 until records.length()) {
                val row = records.optJSONObject(i) ?: continue
                append(row.optString("case_id")).append(',')
                append(row.optString("style")).append(',')
                append(row.optDouble("thickness")).append(',')
                append(row.optDouble("length")).append(',')
                append(row.optDouble("arch")).append(',')
                append(row.optInt("rating")).append(',')
                append(row.optBoolean("selected")).append(',')
                append(row.optDouble("face_width_height_ratio")).append(',')
                append(row.optDouble("interocular_face_width_ratio")).append(',')
                append(row.optDouble("jaw_face_width_ratio")).append(',')
                append(row.optDouble("brow_eye_distance_ratio")).append(',')
                append(row.optDouble("brow_asymmetry_ratio")).append(',')
                append(row.optDouble("eye_line_tilt")).append(',')
                append(row.optLong("timestamp")).append('\n')
            }
        }
        val filename = "brow_preference_feedback_${System.currentTimeMillis()}.csv"
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, filename)
                    put(MediaStore.Downloads.MIME_TYPE, "text/csv")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/BrowDesign")
                    put(MediaStore.Downloads.IS_PENDING, 1)
                }
                val uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: throw IllegalStateException("无法创建 CSV 文件")
                contentResolver.openOutputStream(uri)?.use { it.write(csv.toByteArray(Charsets.UTF_8)) }
                    ?: throw IllegalStateException("无法写入 CSV")
                values.clear()
                values.put(MediaStore.Downloads.IS_PENDING, 0)
                contentResolver.update(uri, values, null, null)
                Toast.makeText(this, "偏好数据已导出到 Downloads/BrowDesign", Toast.LENGTH_LONG).show()
            } else {
                val directory = File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "BrowDesign")
                if (!directory.exists()) directory.mkdirs()
                val file = File(directory, filename)
                file.writeText(csv, Charsets.UTF_8)
                Toast.makeText(this, "偏好数据已导出：" + file.absolutePath, Toast.LENGTH_LONG).show()
            }
        } catch (e: Exception) {
            Toast.makeText(this, "导出失败：" + (e.message ?: "请重试"), Toast.LENGTH_LONG).show()
        }
    }

    private fun clearLocalRecords() {
        AlertDialog.Builder(this)
            .setTitle("清除应用内记录")
            .setMessage("将删除本机保存的匿名偏好反馈与曲率测量记录。已导出的 CSV 文件和已保存到相册的设计图不会自动删除，需要到下载目录或相册中单独清除。此操作无法撤销。")
            .setNegativeButton("取消", null)
            .setPositiveButton("清除") { _, _ ->
                getSharedPreferences("brow_preference_feedback", Context.MODE_PRIVATE).edit().clear().apply()
                getSharedPreferences("brow_curvature_samples", Context.MODE_PRIVATE).edit().clear().apply()
                savedPlans.forEach { it.segmentationSnapshot?.overlay?.recycle() }
                savedPlans.clear()
                refreshPlanButtons()
                Toast.makeText(this, "应用内匿名记录已清除", Toast.LENGTH_LONG).show()
            }
            .show()
    }

    private val curvatureRegionNames = listOf(
        "左眉头", "左眉身", "左眉峰", "左眉尾",
        "右眉头", "右眉身", "右眉峰", "右眉尾"
    )

    private fun currentCurvatureFactorsForPrint(): Map<String, Float> {
        val records = org.json.JSONArray(
            getSharedPreferences("brow_curvature_samples", Context.MODE_PRIVATE).getString("records", "[]")
        )
        val valuesByRegion = mutableMapOf<String, MutableList<Float>>()
        for (i in 0 until records.length()) {
            val row = records.optJSONObject(i) ?: continue
            if (row.optString("case_id") != currentCaseId) continue
            val region = row.optString("region")
            val factor = row.optDouble("factor", Double.NaN).toFloat()
            if (region !in curvatureRegionNames || !factor.isFinite() || factor !in 1.0f..1.5708f) continue
            valuesByRegion.getOrPut(region) { mutableListOf() }.add(factor)
        }
        return valuesByRegion.mapValues { (_, factors) ->
            val sorted = factors.sorted()
            if (sorted.size % 2 == 1) sorted[sorted.size / 2]
            else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2f
        }
    }

    /**
     * Uses MediaPipe's relative z estimate only as a rough local arc-length proxy.
     * It is NOT metric skin-surface length and must remain opt-in until compared with
     * physical measurements. Values are clamped to limit the influence of noisy z.
     */
    private fun estimateDepthBasedCurvatureFactors(
        points: List<PointF>,
        depths: List<Float>
    ): Map<String, Float> {
        if (points.size < 455 || depths.size != points.size ||
            depths.any { !it.isFinite() }) return emptyMap()
        val noseX = points.getOrNull(1)?.x ?: return emptyMap()
        val browLandmarkSets = listOf(
            intArrayOf(70, 63, 105, 66, 107),
            intArrayOf(300, 293, 334, 296, 336)
        )
        val factors = linkedMapOf<String, Float>()
        for (ids in browLandmarkSets) {
            val usable = ids.filter { it in points.indices && it in depths.indices }
                .filter { points[it].x.isFinite() && points[it].y.isFinite() }
                .sortedBy { points[it].x }
            if (usable.size != ids.size) continue
            val sideCenter = usable.map { points[it].x }.average().toFloat()
            val anatomicalRight = sideCenter < noseX
            val names = if (anatomicalRight) {
                listOf("右眉尾", "右眉峰", "右眉身", "右眉头")
            } else {
                listOf("左眉头", "左眉身", "左眉峰", "左眉尾")
            }
            for (i in 0 until usable.lastIndex) {
                val a = usable[i]
                val b = usable[i + 1]
                val dx = points[b].x - points[a].x
                val dy = points[b].y - points[a].y
                val dz = depths[b] - depths[a]
                val planar = hypot(dx.toDouble(), dy.toDouble()).toFloat()
                val spatial = hypot(planar.toDouble(), dz.toDouble()).toFloat()
                val factor = if (planar > 0.5f && spatial.isFinite()) {
                    (spatial / planar).coerceIn(1f, 1.4f)
                } else {
                    1f
                }
                factors[names[i]] = factor
            }
        }
        return factors
    }

    private fun askPrintCalibration() {
        val points = browView.facePoints()
        if (points.size < 478 || capturedPhoto == null) {
            Toast.makeText(this, "未检测到完整虹膜关键点，暂时无法可靠建立 1:1 打印比例，请重新拍摄", Toast.LENGTH_LONG).show()
            return
        }
        val measuredFactors = currentCurvatureFactorsForPrint()
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "请输入实测瞳距（毫米）"
            setText("")
        }
        val calibrationInfo = TextView(this).apply {
            text = "已记录当前顾客的曲率区域：${measuredFactors.size}/8。必须使用柔性尺测表面弧长，并测量同一段两端距离。"
            textSize = 12f
            setPadding(dp(4), dp(6), dp(4), dp(6))
        }
        val printScalePrefs = getSharedPreferences("brow_print_calibration", Context.MODE_PRIVATE)
        val savedPrintScaleCorrection = printScalePrefs.getFloat("scale_correction", 1f)
        val printScaleInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "上一张打印稿中 20 mm 方框的实测边长，未测量留空"
            setText("")
        }
        val printScaleState = TextView(this).apply {
            text = "当前已保存打印比例修正系数：x${(savedPrintScaleCorrection * 1000f).roundToInt() / 1000f}。留空会继续使用该系数。"
            textSize = 11f
            setPadding(dp(4), dp(2), dp(4), dp(4))
        }
        val resetPrintScale = CheckBox(this).apply {
            text = "更换打印机、纸张或打印设置后，重置修正系数"
            isChecked = false
        }
        val applyCorrection = CheckBox(this).apply {
            text = "应用此顾客已实测的区域长度修正（优先使用实测值）"
            // Physical measurements remain opt-in and have priority over estimated depth.
            isChecked = false
            isEnabled = measuredFactors.size == curvatureRegionNames.size
        }
        val depthPreviewFactors = estimateDepthBasedCurvatureFactors(points, capturedLandmarkDepths)
        val applyDepthCorrection = CheckBox(this).apply {
            text = "尝试 MediaPipe 深度估计的区域长度修正（实验性）"
            isChecked = false
            isEnabled = depthPreviewFactors.size == curvatureRegionNames.size
        }
        val note = TextView(this).apply {
            text = "单张正面照片无法恢复真实三维眉弓。MediaPipe 的 z 值是相对深度估计，不是毫米级皮肤表面测量。优先使用顾客实测值；深度估计修正默认关闭，仅用于试验比较。所有方案都必须先通过打印后的 20 mm 校准。"
            textSize = 11f
            setPadding(dp(4), dp(2), dp(4), dp(2))
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(TextView(this@MainActivity).apply {
                text = "1:1 打印校准"
                textSize = 16f
                setTextColor(0xFF262A22.toInt())
            })
            addView(TextView(this@MainActivity).apply {
                text = "测量顾客两瞳孔中心的实际距离并输入毫米数。"
                textSize = 12f
            })
            addView(input)
            addView(TextView(this@MainActivity).apply {
                text = "打印比例校准（可选，当前已保存系数会持续生效）"
                textSize = 12f
                setPadding(dp(4), dp(4), dp(4), 0)
            })
            addView(printScaleInput)
            addView(TextView(this@MainActivity).apply {
                text = "首次打印先留空。之后输入上一张打印稿中 20 mm 方框的实测边长，例如 19.5 mm，应用会在当前修正系数基础上更新校准值。校准通过后，后续打印留空即可沿用修正系数。请保持打印机、纸张和打印设置一致。"
                textSize = 11f
                setPadding(dp(4), dp(2), dp(4), dp(4))
            })
            addView(printScaleState)
            addView(resetPrintScale)
            addView(calibrationInfo)
            addView(applyCorrection)
            addView(applyDepthCorrection)
            addView(note)
        }
        AlertDialog.Builder(this)
            .setTitle("生成 1:1 拓描模板")
            .setMessage("打印时选择 A4 与 100% / 实际尺寸，关闭“适合页面”等自动缩放。首次把打印比例校准留空，打印后测量 20 mm 校准框，再把实际边长填入后重打并复测。")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("生成打印模板") { _, _ ->
                val measuredMm = input.text.toString().toFloatOrNull()
                if (measuredMm == null || measuredMm !in 35f..90f) {
                    Toast.makeText(this, "请输入合理的实测瞳距（35–90 毫米）", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val measuredSquareText = printScaleInput.text.toString().trim()
                val measuredSquareMm = if (measuredSquareText.isEmpty()) null else measuredSquareText.toFloatOrNull()
                if (measuredSquareText.isNotEmpty() &&
                    (measuredSquareMm == null || measuredSquareMm !in 16f..24f)) {
                    Toast.makeText(this, "校准框实测边长应为 16–24 毫米，首次打印可留空", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val correctionBase = if (resetPrintScale.isChecked) 1f else
                    getSharedPreferences("brow_print_calibration", Context.MODE_PRIVATE)
                        .getFloat("scale_correction", 1f)
                val printScaleCorrection = measuredSquareMm?.let { correctionBase * 20f / it } ?: correctionBase
                if (printScaleCorrection !in 0.80f..1.25f) {
                    Toast.makeText(this, "计算出的打印修正超出 0.80–1.25。请检查是否选择了实际尺寸、A4 纸张和正确的校准框，必要时重置比例。", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                getSharedPreferences("brow_print_calibration", Context.MODE_PRIVATE).edit()
                    .putFloat("scale_correction", printScaleCorrection)
                    .apply()

                val pixelDistance = BrowRenderer.pupilDistance(points)
                if (pixelDistance < 5f) {
                    Toast.makeText(this, "未能定位瞳孔，请重新拍摄正面照片", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val useMeasuredCurvature = applyCorrection.isChecked
                val useDepthCurvature = applyDepthCorrection.isChecked && !useMeasuredCurvature
                if (useMeasuredCurvature && measuredFactors.size != curvatureRegionNames.size) {
                    Toast.makeText(this, "8 个区域的实测值不完整，不能应用实测区域修正", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val depthFactors = if (useDepthCurvature) {
                    estimateDepthBasedCurvatureFactors(points, capturedLandmarkDepths)
                } else {
                    emptyMap()
                }
                if (useDepthCurvature && depthFactors.size != curvatureRegionNames.size) {
                    Toast.makeText(this, "MediaPipe 深度关键点不完整，不能应用估计修正", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val useCurvature = useMeasuredCurvature || useDepthCurvature
                val factorsToApply = when {
                    useMeasuredCurvature -> measuredFactors
                    useDepthCurvature -> depthFactors
                    else -> emptyMap()
                }
                val template = BrowRenderer.createTemplate(
                    points,
                    params,
                    capturedPhoto!!.width,
                    capturedPhoto!!.height,
                    factorsToApply,
                    segmentationResult,
                    useNativeShape
                )
                if (template == null) {
                    Toast.makeText(this, "眉形模板生成失败，请重新拍摄", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                printTemplate(template, pixelDistance / measuredMm, useCurvature, printScaleCorrection)
            }
            .show()
    }

    private fun printTemplate(
        template: Bitmap,
        pixelsPerMm: Float,
        curvatureCorrected: Boolean = false,
        printScaleCorrection: Float = 1f
    ) {
        val printManager = getSystemService(Context.PRINT_SERVICE) as PrintManager
        val jobName = "眉形拓描模板"
        val defaultAttributes = PrintAttributes.Builder()
            .setMediaSize(PrintAttributes.MediaSize.ISO_A4)
            .setResolution(PrintAttributes.Resolution("brow_300dpi", "300 dpi", 300, 300))
            .setColorMode(PrintAttributes.COLOR_MODE_MONOCHROME)
            .setMinMargins(PrintAttributes.Margins(200, 200, 200, 200))
            .build()

        val adapter = object : PrintDocumentAdapter() {
            private var attributes: PrintAttributes = defaultAttributes

            override fun onLayout(
                oldAttributes: PrintAttributes?,
                newAttributes: PrintAttributes?,
                cancellationSignal: CancellationSignal?,
                callback: LayoutResultCallback,
                extras: Bundle?
            ) {
                if (cancellationSignal?.isCanceled == true) {
                    callback.onLayoutCancelled()
                    return
                }
                attributes = newAttributes ?: defaultAttributes
                val info = PrintDocumentInfo.Builder(jobName)
                    .setContentType(PrintDocumentInfo.CONTENT_TYPE_DOCUMENT)
                    .setPageCount(1)
                    .build()
                callback.onLayoutFinished(info, true)
            }

            override fun onWrite(
                pages: Array<out PageRange>,
                destination: ParcelFileDescriptor,
                cancellationSignal: CancellationSignal?,
                callback: WriteResultCallback
            ) {
                val document = PrintedPdfDocument(this@MainActivity, attributes)
                try {
                    if (cancellationSignal?.isCanceled == true) {
                        callback.onWriteCancelled()
                        return
                    }
                    val page = document.startPage(1)
                    val canvas = page.canvas
                    val content = document.pageContentRect
                    val pointsPerPixel = 72f / (25.4f * pixelsPerMm) * printScaleCorrection
                    val widthPts = template.width * pointsPerPixel
                    val heightPts = template.height * pointsPerPixel
                    val calibrationPts = 20f * 72f / 25.4f * printScaleCorrection
                    val gapPts = 12f
                    val hasScaleCorrection = kotlin.math.abs(printScaleCorrection - 1f) > 0.001f
                    val calibrationLabelHeightPts = 12f +
                        (if (hasScaleCorrection) 10f else 0f) +
                        (if (curvatureCorrected) 10f else 0f)
                    val totalHeightPts = heightPts + gapPts + calibrationPts + calibrationLabelHeightPts
                    if (widthPts > content.width() || totalHeightPts > content.height()) {
                        document.finishPage(page)
                        callback.onWriteFailed("模板及 20 毫米校准框超过 A4 可打印范围，请检查瞳距和照片比例")
                        return
                    }
                    val left = content.left + (content.width() - widthPts) / 2f
                    val groupTop = content.top + (content.height() - totalHeightPts) / 2f
                    val dest = RectF(left, groupTop, left + widthPts, groupTop + heightPts)
                    canvas.drawBitmap(template, null, dest, Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG))
                    val boxLeft = content.left + (content.width() - calibrationPts) / 2f
                    val boxTop = groupTop + heightPts + gapPts
                    val calibrationPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.BLACK
                        style = Paint.Style.STROKE
                        strokeWidth = 0.6f
                    }
                    canvas.drawRect(RectF(boxLeft, boxTop, boxLeft + calibrationPts, boxTop + calibrationPts), calibrationPaint)
                    val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.BLACK
                        textSize = 8f
                        textAlign = Paint.Align.CENTER
                        style = Paint.Style.FILL
                    }
                    canvas.drawText("20 mm calibration", boxLeft + calibrationPts / 2f, boxTop + calibrationPts + 9f, labelPaint)
                    var annotationLine = 1
                    if (hasScaleCorrection) {
                        val correctionLabel = "打印比例修正 x${(printScaleCorrection * 1000f).roundToInt() / 1000f}"
                        canvas.drawText(
                            correctionLabel,
                            content.exactCenterX(),
                            boxTop + calibrationPts + 9f + 11f * annotationLine,
                            labelPaint
                        )
                        annotationLine++
                    }
                    if (curvatureCorrected) {
                        canvas.drawText(
                            "区域长度修正已启用（实验性）",
                            content.exactCenterX(),
                            boxTop + calibrationPts + 9f + 11f * annotationLine,
                            labelPaint
                        )
                    }
                    document.finishPage(page)
                    FileOutputStream(destination.fileDescriptor).use { document.writeTo(it) }
                    callback.onWriteFinished(arrayOf(PageRange.ALL_PAGES))
                } catch (e: Exception) {
                    callback.onWriteFailed(e.message ?: "生成打印文件失败")
                } finally {
                    document.close()
                    template.recycle()
                }
            }
        }
        printManager.print(jobName, adapter, defaultAttributes)
    }

    override fun onResume() {
        super.onResume()
        if (!designMode && landmarker == null &&
            ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startFaceTracking()
        }
    }

    override fun onPause() {
        super.onPause()
        stopTransition()
    }

    override fun onDestroy() {
        cameraProvider?.unbindAll()
        landmarker?.close()
        landmarker = null
        cameraExecutor?.shutdown()
        cameraExecutor = null
        imageExecutor?.shutdown()
        imageExecutor = null
        imageCapture = null
        segmentationGeneration++
        segmentationResult?.overlay?.recycle()
        segmentationResult = null
        BrowSegmentationEngine.close()
        capturedPhoto?.recycle()
        capturedPhoto = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == cameraPermission && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startFaceTracking()
        } else {
            status.text = "请允许摄像头权限，才能拍摄并识别人脸"
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
}

private data class BrowRecommendation(
    val name: String,
    val score: Int,
    val params: BrowParams,
    val reason: String
)

private data class BrowFaceMeasurements(
    val widthHeightRatio: Float,
    val interocularFaceWidthRatio: Float,
    val jawFaceWidthRatio: Float,
    val browEyeDistanceRatio: Float,
    val browAsymmetryRatio: Float,
    val eyeLineTilt: Float
)

/**
 * An explainable geometric baseline for prototyping candidate ordering.
 * Scores are not probabilities and must not be presented as validated AI output.
 */
private object BrowRecommendationEngine {
    private data class TrainedModel(
        val means: FloatArray,
        val scales: FloatArray,
        val coefficients: FloatArray,
        val intercept: Float,
        val calibrationIntercept: Float,
        val calibrationSlope: Float
    )
    private var loadAttempted = false
    private var trainedModel: TrainedModel? = null

    fun measurements(points: List<PointF>): BrowFaceMeasurements {
        if (points.size <= 454) return BrowFaceMeasurements(0f, 0f, 0f, 0f, 0f, 0f)
        val faceWidth = distance(points[234], points[454]).coerceAtLeast(1f)
        val faceHeight = distance(
            points.getOrNull(10) ?: points[234],
            points.getOrNull(152) ?: points[454]
        ).coerceAtLeast(1f)
        val interocular = if (points.size > 263) distance(points[33], points[263]) else 0f
        val jawWidth = if (points.size > 397) distance(points[172], points[397]) else faceWidth * 0.8f
        val browEyeDistance = if (points.size > 386) {
            (abs(points[105].y - points[159].y) + abs(points[334].y - points[386].y)) / 2f / faceHeight
        } else 0f
        val asymmetry = if (points.size > 334) abs(points[105].y - points[334].y) / faceHeight else 0f
        val eyeTilt = if (points.size > 263) {
            (Math.toDegrees(kotlin.math.atan2(
                (points[263].y - points[33].y).toDouble(),
                (points[263].x - points[33].x).toDouble()
            )) / 90.0).toFloat()
        } else 0f
        return BrowFaceMeasurements(
            faceWidth / faceHeight,
            interocular / faceWidth,
            jawWidth / faceWidth,
            browEyeDistance,
            asymmetry,
            eyeTilt
        )
    }

    fun hasApprovedModel(context: Context): Boolean = loadModel(context) != null

    fun sourceLabel(context: Context): String =
        if (loadModel(context) != null) {
            "已启用经过独立顾客集校准、验证并完成人工审核的偏好估计模型。分数表示顾客给出 4–5 分评价的估计倾向，不代表客观审美或治疗效果保证。"
        } else {
            "当前使用几何规则参考分，并非 AI 概率。偏好模型只有经过独立校准、外部顾客验证和人工审核后才会启用。"
        }

    fun rank(context: Context, points: List<PointF>): List<BrowRecommendation> {
        val features = measurements(points)
        val templates = listOf(
            BrowRecommendation("自然款", 82, BrowParams(1.0f, 1.0f, 0.92f, 0), "自然款：较保守的参数基线，仍需纹绣师确认"),
            BrowRecommendation("柔和款", 78, BrowParams(0.82f, 0.94f, 0.72f, 1), "柔和款：较低粗细与弧度的参数预设"),
            BrowRecommendation("立体款", 76, BrowParams(1.18f, 1.06f, 1.22f, 2), "立体款：较高粗细与弧度的参数预设")
        )
        val model = loadModel(context)
        if (model != null) {
            // Search a small, bounded parameter grid within each style family. This
            // personalizes thickness, length and arch only through a reviewed model.
            // Without a reviewed model the app does not pretend these are learned parameters.
            val offsets = listOf(-0.12f, 0f, 0.12f)
            return templates.map { family ->
                val candidates = mutableListOf<BrowRecommendation>()
                for (thicknessDelta in offsets) {
                    for (lengthDelta in listOf(-0.06f, 0f, 0.06f)) {
                        for (archDelta in offsets) {
                            val candidateParams = family.params.copy(
                                thickness = (family.params.thickness + thicknessDelta).coerceIn(0.55f, 1.45f),
                                length = (family.params.length + lengthDelta).coerceIn(0.80f, 1.20f),
                                arch = (family.params.arch + archDelta).coerceIn(0.55f, 1.45f)
                            )
                            val x = floatArrayOf(
                                features.widthHeightRatio,
                                features.interocularFaceWidthRatio,
                                features.jawFaceWidthRatio,
                                features.browEyeDistanceRatio,
                                features.browAsymmetryRatio,
                                features.eyeLineTilt,
                                candidateParams.thickness,
                                candidateParams.length,
                                candidateParams.arch,
                                if (candidateParams.style == 1) 1f else 0f,
                                if (candidateParams.style == 2) 1f else 0f
                            )
                            val preference = predict(model, x)
                            candidates.add(
                                family.copy(
                                    name = "${family.name}·自适应",
                                    score = (preference * 100f).roundToInt().coerceIn(0, 100),
                                    params = candidateParams,
                                    reason = "${family.name}：参数组合经已审核的顾客偏好模型排序，仍需顾客和纹绣师确认"
                                )
                            )
                        }
                    }
                }
                candidates.maxBy { it.score }
            }.sortedByDescending { it.score }
        }
        val shapeSignal = ((features.widthHeightRatio - 0.72f) / 0.18f).coerceIn(-1f, 1f)
        val penalty = (features.browAsymmetryRatio * 35f).coerceIn(0f, 5f)
        return listOf(
            templates[0].copy(score = (82f - abs(shapeSignal) * 3.5f - penalty).roundToInt().coerceIn(60, 95)),
            templates[1].copy(score = (78f + (if (shapeSignal < 0f) -shapeSignal * 6f else -shapeSignal * 2f) - penalty * 0.6f).roundToInt().coerceIn(60, 95)),
            templates[2].copy(score = (76f + (if (shapeSignal > 0f) shapeSignal * 7f else shapeSignal * 2f) - penalty * 0.4f).roundToInt().coerceIn(60, 95))
        ).sortedByDescending { it.score }
    }

    private fun loadModel(context: Context): TrainedModel? {
        if (loadAttempted) return trainedModel
        loadAttempted = true
        return try {
            val json = context.assets.open("brow_ranker.json").bufferedReader(Charsets.UTF_8).use { it.readText() }
            val root = org.json.JSONObject(json)
            // Only a model explicitly human-reviewed after independent calibration and
            // external validation may provide percentages in the live app.
            if (root.optInt("model_version", -1) != 2) return null
            if (!root.optBoolean("release_approved", false)) return null
            if (root.optString("calibration_status") != "platt_scaling") return null
            val external = root.optJSONObject("external_validation") ?: return null
            val gates = external.optJSONObject("release_gates") ?: return null
            if (external.optInt("n_cases", 0) < 100 ||
                external.optDouble("roc_auc", 0.0) < 0.60 ||
                external.optDouble("brier_score", Double.POSITIVE_INFINITY) >=
                    external.optDouble("prevalence_baseline_brier_score", 0.0) ||
                external.optDouble("expected_calibration_error_10_bins", Double.POSITIVE_INFINITY) > 0.10 ||
                !external.optBoolean("passed_release_gates", false) ||
                !gates.optBoolean("external_cases_at_least_100", false) ||
                !gates.optBoolean("roc_auc_at_least_0_60", false) ||
                !gates.optBoolean("brier_better_than_prevalence_baseline", false) ||
                !gates.optBoolean("ece_at_most_0_10", false)) return null
            val expectedFeatures = listOf(
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
                "style_defined"
            )
            val featureOrder = root.optJSONArray("feature_order") ?: return null
            if (featureOrder.length() != expectedFeatures.size) return null
            for (i in expectedFeatures.indices) {
                if (featureOrder.optString(i) != expectedFeatures[i]) return null
            }
            val meansJson = root.getJSONArray("means")
            val scalesJson = root.getJSONArray("scales")
            val coefficientsJson = root.getJSONArray("coefficients")
            if (meansJson.length() != 11 || scalesJson.length() != 11 || coefficientsJson.length() != 11) return null
            trainedModel = TrainedModel(
                FloatArray(11) { meansJson.getDouble(it).toFloat() },
                FloatArray(11) { scalesJson.getDouble(it).toFloat().let { scale -> if (abs(scale) < 1e-8f) 1f else scale } },
                FloatArray(11) { coefficientsJson.getDouble(it).toFloat() },
                root.getDouble("intercept").toFloat(),
                root.getDouble("calibration_intercept").toFloat(),
                root.getDouble("calibration_slope").toFloat()
            )
            trainedModel
        } catch (_: Exception) {
            null
        }
    }

    private fun predict(model: TrainedModel, x: FloatArray): Float {
        var logit = model.intercept
        for (i in x.indices) {
            val scaled = (x[i] - model.means[i]) / model.scales[i]
            logit += model.coefficients[i] * scaled
        }
        val bounded = logit.coerceIn(-20f, 20f)
        val calibratedLogit = (
            model.calibrationIntercept + model.calibrationSlope * bounded
        ).coerceIn(-20f, 20f)
        return (1f / (1f + kotlin.math.exp(-calibratedLogit)))
    }

    private fun distance(a: PointF, b: PointF): Float =
        hypot((a.x - b.x).toDouble(), (a.y - b.y).toDouble()).toFloat()
}

private class BrowDesignView(context: Context) : View(context) {
    private var photo: Bitmap? = null
    private var points: List<PointF> = emptyList()
    private var params: BrowParams = BrowParams()

    fun setPhoto(bitmap: Bitmap?) {
        photo = bitmap
        invalidate()
    }

    fun setFacePoints(newPoints: List<PointF>) {
        points = newPoints.map { PointF(it.x, it.y) }
        invalidate()
    }

    fun facePoints(): List<PointF> = points.map { PointF(it.x, it.y) }

    private var segmentationResult: BrowSegmentationResult? = null
    private var nativeContours: List<NativeBrowContour> = emptyList()
    private var showSegmentation: Boolean = false
    private var useNativeShape: Boolean = false
    private var editSegmentationMask: Boolean = false
    private var includeEditedPixels: Boolean = true

    fun setSegmentationResult(result: BrowSegmentationResult?, visible: Boolean) {
        segmentationResult = result
        showSegmentation = visible
        nativeContours = result?.let { BrowRenderer.extractNativeContours(it) } ?: emptyList()
        if (!visible) editSegmentationMask = false
        invalidate()
    }

    fun setNativeShapeMode(enabled: Boolean) {
        useNativeShape = enabled && nativeContours.isNotEmpty()
        invalidate()
    }

    fun hasUsableNativeContour(): Boolean = nativeContours.size >= 2

    fun setSegmentationEditMode(enabled: Boolean, includePixels: Boolean = true) {
        editSegmentationMask = enabled
        includeEditedPixels = includePixels
        invalidate()
    }

    override fun onTouchEvent(event: android.view.MotionEvent): Boolean {
        if (!editSegmentationMask || !showSegmentation || event.action !in listOf(
                android.view.MotionEvent.ACTION_DOWN,
                android.view.MotionEvent.ACTION_MOVE,
                android.view.MotionEvent.ACTION_UP
            )) return super.onTouchEvent(event)
        val image = photo ?: return false
        val result = segmentationResult ?: return false
        if (image.isRecycled || result.overlay.isRecycled) return false
        val scale = max(width.toFloat() / image.width.coerceAtLeast(1), height.toFloat() / image.height.coerceAtLeast(1))
        val imageLeft = (width - image.width * scale) / 2f
        val imageTop = (height - image.height * scale) / 2f
        val imageX = (event.x - imageLeft) / scale
        val imageY = (event.y - imageTop) / scale
        val maskX = imageX - result.offsetX
        val maskY = imageY - result.offsetY
        if (maskX < 0f || maskX >= result.overlay.width.toFloat() ||
            maskY < 0f || maskY >= result.overlay.height.toFloat()) {
            return true
        }
        val brushRadius = max(4f, (8f * resources.displayMetrics.density) / scale)
        val maskCanvas = Canvas(result.overlay)
        val maskPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            if (includeEditedPixels) {
                color = Color.argb(172, 71, 190, 157)
                xfermode = null
            } else {
                color = Color.TRANSPARENT
                xfermode = android.graphics.PorterDuffXfermode(android.graphics.PorterDuff.Mode.CLEAR)
            }
        }
        maskCanvas.drawCircle(maskX, maskY, brushRadius, maskPaint)
        if (event.action == android.view.MotionEvent.ACTION_UP) {
            nativeContours = BrowRenderer.extractNativeContours(result)
            if (useNativeShape && nativeContours.isEmpty()) useNativeShape = false
        }
        invalidate()
        return true
    }

    fun setParams(newParams: BrowParams) {
        params = newParams
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val image = photo
        if (image != null && !image.isRecycled) {
            val scale = max(width.toFloat() / image.width.coerceAtLeast(1), height.toFloat() / image.height.coerceAtLeast(1))
            val displayWidth = image.width * scale
            val displayHeight = image.height * scale
            val left = (width - displayWidth) / 2f
            val top = (height - displayHeight) / 2f
            val destination = RectF(left, top, left + displayWidth, top + displayHeight)
            canvas.drawBitmap(image, null, destination, Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG))
            val currentSegmentation = segmentationResult
            if (showSegmentation && currentSegmentation != null && !currentSegmentation.overlay.isRecycled) {
                val x0 = left + currentSegmentation.offsetX * scale
                val y0 = top + currentSegmentation.offsetY * scale
                val x1 = x0 + currentSegmentation.overlay.width * scale
                val y1 = y0 + currentSegmentation.overlay.height * scale
                canvas.drawBitmap(
                    currentSegmentation.overlay,
                    null,
                    RectF(x0, y0, x1, y1),
                    Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
                )
            }
            if (useNativeShape && nativeContours.isNotEmpty()) {
                BrowRenderer.drawNativeContours(canvas, nativeContours, points, params, scale, left, top)
            } else {
                val screenPoints = points.map { p ->
                    PointF(left + p.x * scale, top + p.y * scale)
                }
                BrowRenderer.drawBrows(canvas, screenPoints, params)
            }
        } else {
            BrowRenderer.drawBrows(canvas, points, params)
        }
    }
}

private object BrowRenderer {
    private val browSets = listOf(
        intArrayOf(70, 63, 105, 66, 107),
        intArrayOf(300, 293, 334, 296, 336)
    )
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun drawBrows(
        canvas: Canvas,
        points: List<PointF>,
        params: BrowParams,
        outlineOnly: Boolean = false,
        curvatureFactors: Map<String, Float> = emptyMap()
    ) {
        if (points.size < 455) return
        val noseX = points.getOrNull(1)?.x ?: points.map { it.x }.average().toFloat()
        browSets.forEach { ids ->
            val sideCenter = ids.map { index -> points.getOrNull(index) }.filterNotNull().map { it.x }.average().toFloat()
            // The displayed selfie image is mirrored. Convert its screen side to anatomical side.
            val anatomicalRight = sideCenter < noseX
            drawOne(canvas, points, ids, params, outlineOnly, curvatureFactors, anatomicalRight)
        }
    }

    private fun drawOne(
        canvas: Canvas,
        points: List<PointF>,
        ids: IntArray,
        params: BrowParams,
        outlineOnly: Boolean,
        curvatureFactors: Map<String, Float>,
        anatomicalRight: Boolean
    ) {
        // Use all detected brow-region landmarks instead of forcing a generic arch between two endpoints.
        val ordered = ids.map { index -> points.getOrNull(index) }
            .filterNotNull()
            .filter { it.x.isFinite() && it.y.isFinite() }
            .sortedBy { it.x }
        if (ordered.size < 4) return
        val segmentRegions = if (anatomicalRight) {
            listOf("右眉尾", "右眉峰", "右眉身", "右眉头")
        } else {
            listOf("左眉头", "左眉身", "左眉峰", "左眉尾")
        }
        val workingPoints = if (curvatureFactors.isEmpty()) {
            ordered
        } else {
            val cumulative = mutableListOf(ordered.first().x)
            for (i in 0 until ordered.lastIndex) {
                val rawFactor = curvatureFactors[segmentRegions[i]] ?: 1f
                val factor = rawFactor.coerceIn(1f, 1.4f)
                val dx = ordered[i + 1].x - ordered[i].x
                cumulative.add(cumulative.last() + dx * factor)
            }
            val originalCenter = (ordered.first().x + ordered.last().x) / 2f
            val correctedCenter = (cumulative.first() + cumulative.last()) / 2f
            ordered.mapIndexed { index, point -> PointF(cumulative[index] + originalCenter - correctedCenter, point.y) }
        }
        val anchorStart = workingPoints.first()
        val anchorEnd = workingPoints.last()
        val naturalSpan = hypot(
            (anchorEnd.x - anchorStart.x).toDouble(),
            (anchorEnd.y - anchorStart.y).toDouble()
        ).toFloat()
        if (naturalSpan < 8f) return

        val centerX = (anchorStart.x + anchorEnd.x) / 2f
        val displaySpan = naturalSpan * params.length
        val centerline = (0..48).map { index ->
            val t = index / 48f
            val base = sampleCatmullRom(workingPoints, t)
            val archDelta = naturalSpan * 0.075f * (params.arch - 1f) * 4f * t * (1f - t)
            PointF(centerX + (base.x - centerX) * params.length, base.y - archDelta)
        }
        val halfThickness = displaySpan * 0.055f * params.thickness
        val top = mutableListOf<PointF>()
        val bottom = mutableListOf<PointF>()
        for (i in centerline.indices) {
            val current = centerline[i]
            val before = centerline[(i - 1).coerceAtLeast(0)]
            val after = centerline[(i + 1).coerceAtMost(centerline.lastIndex)]
            val tx = after.x - before.x
            val ty = after.y - before.y
            val tangentLength = max(0.001f, hypot(tx.toDouble(), ty.toDouble()).toFloat())
            var nx = -ty / tangentLength
            var ny = tx / tangentLength
            if (ny > 0f) { nx = -nx; ny = -ny }
            val t = i.toFloat() / centerline.lastIndex
            val taper = 0.12f + 0.88f * sin(Math.PI * t).toFloat().coerceAtLeast(0f)
            val thickness = halfThickness * taper
            top.add(PointF(current.x + nx * thickness, current.y + ny * thickness))
            bottom.add(PointF(current.x - nx * thickness, current.y - ny * thickness))
        }

        val path = Path()
        path.moveTo(top.first().x, top.first().y)
        top.drop(1).forEach { path.lineTo(it.x, it.y) }
        bottom.asReversed().forEach { path.lineTo(it.x, it.y) }
        path.close()
        if (outlineOnly) {
            paint.shader = null
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = max(1.0f, displaySpan * 0.003f)
            paint.strokeJoin = Paint.Join.ROUND
            paint.strokeCap = Paint.Cap.ROUND
            paint.color = Color.BLACK
            canvas.drawPath(path, paint)
            paint.style = Paint.Style.FILL
            paint.strokeWidth = 1f
            return
        }
        val colors = when (params.style) {
            1 -> intArrayOf(Color.argb(140, 94, 65, 51), Color.argb(165, 112, 77, 59))
            2 -> intArrayOf(Color.argb(175, 45, 29, 22), Color.argb(210, 55, 34, 25))
            else -> intArrayOf(Color.argb(150, 76, 48, 34), Color.argb(185, 92, 58, 40))
        }
        paint.style = Paint.Style.FILL
        paint.shader = LinearGradient(
            anchorStart.x,
            centerline.minOf { it.y },
            anchorEnd.x,
            max(anchorStart.y, anchorEnd.y),
            colors[0],
            colors[1],
            Shader.TileMode.CLAMP
        )
        canvas.drawPath(path, paint)
        paint.shader = null
    }

    private fun sampleCatmullRom(points: List<PointF>, t: Float): PointF {
        if (points.size < 2) return points.firstOrNull() ?: PointF(0f, 0f)
        val position = t.coerceIn(0f, 1f) * (points.size - 1)
        val segment = position.toInt().coerceIn(0, points.size - 2)
        val u = position - segment
        val p0 = points[(segment - 1).coerceAtLeast(0)]
        val p1 = points[segment]
        val p2 = points[segment + 1]
        val p3 = points[(segment + 2).coerceAtMost(points.lastIndex)]
        val u2 = u * u
        val u3 = u2 * u
        fun interpolate(a: Float, b: Float, c: Float, d: Float): Float =
            0.5f * (
                2f * b +
                (-a + c) * u +
                (2f * a - 5f * b + 4f * c - d) * u2 +
                (-a + 3f * b - 3f * c + d) * u3
            )
        return PointF(
            interpolate(p0.x, p1.x, p2.x, p3.x),
            interpolate(p0.y, p1.y, p2.y, p3.y)
        )
    }

    fun extractNativeContours(result: BrowSegmentationResult): List<NativeBrowContour> {
        val mask = result.overlay
        if (mask.isRecycled || result.regions.isEmpty()) return emptyList()
        val extracted = mutableListOf<NativeBrowContour>()
        result.regions.forEachIndexed { regionIndex, bounds ->
            if (bounds.size < 4) return@forEachIndexed
            val x0 = bounds[0]
            val y0 = bounds[1]
            val x1 = bounds[2]
            val y1 = bounds[3]
            val width = x1 - x0
            if (width < 8 || y1 <= y0) return@forEachIndexed
            val topAll = FloatArray(width) { Float.NaN }
            val bottomAll = FloatArray(width) { Float.NaN }
            for (dx in 0 until width) {
                val sx = x0 + dx
                val mx = sx - result.offsetX
                if (mx !in 0 until mask.width) continue
                var minY = Int.MAX_VALUE
                var maxY = Int.MIN_VALUE
                var count = 0
                for (sy in y0 until y1) {
                    val my = sy - result.offsetY
                    if (my !in 0 until mask.height) continue
                    if (Color.alpha(mask.getPixel(mx, my)) >= 80) {
                        minY = min(minY, sy)
                        maxY = max(maxY, sy)
                        count++
                    }
                }
                if (count >= 2 && minY <= maxY) {
                    topAll[dx] = minY.toFloat()
                    bottomAll[dx] = maxY.toFloat()
                }
            }
            val validIndices = topAll.indices.filter { topAll[it].isFinite() && bottomAll[it].isFinite() }
            if (validIndices.size < max(8, width / 12)) return@forEachIndexed
            val first = validIndices.first()
            val last = validIndices.last()
            val top = topAll.copyOfRange(first, last + 1)
            val bottom = bottomAll.copyOfRange(first, last + 1)
            fun fillGaps(values: FloatArray) {
                val nextValid = IntArray(values.size) { -1 }
                var next = -1
                for (i in values.indices.reversed()) {
                    if (values[i].isFinite()) next = i
                    nextValid[i] = next
                }
                var previous = -1
                for (i in values.indices) {
                    if (values[i].isFinite()) {
                        previous = i
                        continue
                    }
                    val following = nextValid[i]
                    values[i] = when {
                        previous < 0 && following >= 0 -> values[following]
                        following < 0 && previous >= 0 -> values[previous]
                        previous >= 0 && following >= 0 -> {
                            val t = (i - previous).toFloat() / (following - previous).coerceAtLeast(1)
                            values[previous] + (values[following] - values[previous]) * t
                        }
                        else -> 0f
                    }
                }
            }
            fillGaps(top)
            fillGaps(bottom)
            val smoothedTop = FloatArray(top.size)
            val smoothedBottom = FloatArray(bottom.size)
            for (i in top.indices) {
                var topSum = 0f
                var bottomSum = 0f
                var count = 0
                for (j in max(0, i - 2)..min(top.lastIndex, i + 2)) {
                    topSum += top[j]
                    bottomSum += bottom[j]
                    count++
                }
                val center = (topSum / count + bottomSum / count) / 2f
                val half = max(0.75f, abs(bottomSum / count - topSum / count) / 2f)
                smoothedTop[i] = center - half
                smoothedBottom[i] = center + half
            }
            extracted.add(
                NativeBrowContour(
                    regionIndex = regionIndex,
                    xStart = x0 + first,
                    top = smoothedTop,
                    bottom = smoothedBottom,
                    regionBounds = bounds.copyOf()
                )
            )
        }
        return extracted
    }

    fun drawNativeContours(
        canvas: Canvas,
        contours: List<NativeBrowContour>,
        landmarks: List<PointF>,
        params: BrowParams,
        scale: Float = 1f,
        imageLeft: Float = 0f,
        imageTop: Float = 0f,
        outlineOnly: Boolean = false,
        curvatureFactors: Map<String, Float> = emptyMap()
    ) {
        if (contours.isEmpty()) return
        val noseX = landmarks.getOrNull(1)?.x ?: landmarks.map { it.x }.average().toFloat()
        contours.forEach { contour ->
            val width = min(contour.top.size, contour.bottom.size)
            if (width < 8) return@forEach
            val imageCenterX = contour.xStart + (width - 1) / 2f
            val anatomicalRight = imageCenterX < noseX
            val segmentNames = if (anatomicalRight) {
                listOf("右眉尾", "右眉峰", "右眉身", "右眉头")
            } else {
                listOf("左眉头", "左眉身", "左眉峰", "左眉尾")
            }
            val cumulative = FloatArray(width)
            cumulative[0] = 0f
            for (i in 1 until width) {
                val t = (i - 0.5f) / (width - 1).coerceAtLeast(1)
                val segment = min(3, (t * 4f).toInt())
                val factor = (curvatureFactors[segmentNames[segment]] ?: 1f).coerceIn(1f, 1.4f)
                cumulative[i] = cumulative[i - 1] + factor
            }
            val cumulativeCenter = (cumulative.first() + cumulative.last()) / 2f
            val displaySpan = max(1f, cumulative.last() * params.length)
            val topPathPoints = ArrayList<PointF>(width)
            val bottomPathPoints = ArrayList<PointF>(width)
            for (i in 0 until width) {
                val t = i.toFloat() / (width - 1).coerceAtLeast(1)
                val x = imageCenterX + (cumulative[i] - cumulativeCenter) * params.length
                val upper = contour.top[i]
                val lower = contour.bottom[i]
                val centerY = (upper + lower) / 2f
                val halfThickness = max(0.7f, abs(lower - upper) * 0.5f * params.thickness)
                val archDelta = displaySpan * 0.075f * (params.arch - 1f) * 4f * t * (1f - t)
                val xOnCanvas = imageLeft + x * scale
                val upperOnCanvas = imageTop + (centerY - halfThickness - archDelta) * scale
                val lowerOnCanvas = imageTop + (centerY + halfThickness - archDelta) * scale
                topPathPoints.add(PointF(xOnCanvas, upperOnCanvas))
                bottomPathPoints.add(PointF(xOnCanvas, lowerOnCanvas))
            }
            val path = Path()
            path.moveTo(topPathPoints.first().x, topPathPoints.first().y)
            topPathPoints.drop(1).forEach { path.lineTo(it.x, it.y) }
            bottomPathPoints.asReversed().forEach { path.lineTo(it.x, it.y) }
            path.close()
            if (outlineOnly) {
                paint.shader = null
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = max(0.8f, displaySpan * scale * 0.003f)
                paint.strokeJoin = Paint.Join.ROUND
                paint.strokeCap = Paint.Cap.ROUND
                paint.color = Color.BLACK
                canvas.drawPath(path, paint)
                paint.style = Paint.Style.FILL
                paint.strokeWidth = 1f
                return@forEach
            }
            val colors = when (params.style) {
                1 -> intArrayOf(Color.argb(140, 94, 65, 51), Color.argb(165, 112, 77, 59))
                2 -> intArrayOf(Color.argb(175, 45, 29, 22), Color.argb(210, 55, 34, 25))
                else -> intArrayOf(Color.argb(150, 76, 48, 34), Color.argb(185, 92, 58, 40))
            }
            paint.style = Paint.Style.FILL
            paint.shader = LinearGradient(
                topPathPoints.first().x,
                topPathPoints.minOf { it.y },
                topPathPoints.last().x,
                bottomPathPoints.maxOf { it.y },
                colors[0],
                colors[1],
                Shader.TileMode.CLAMP
            )
            canvas.drawPath(path, paint)
            paint.shader = null
        }
    }

    fun pupilDistance(points: List<PointF>): Float {
        // Do not silently substitute eye-corner distance for interpupillary distance:
        // that would introduce a systematic physical print-scale error.
        if (points.size < 478) return 0f
        fun average(indices: IntRange): PointF {
            var x = 0f
            var y = 0f
            indices.forEach { x += points[it].x; y += points[it].y }
            return PointF(x / indices.count(), y / indices.count())
        }
        val leftIris = average(468..472)
        val rightIris = average(473..477)
        return hypot((leftIris.x - rightIris.x).toDouble(), (leftIris.y - rightIris.y).toDouble()).toFloat()
    }

    fun renderPreview(
        source: Bitmap,
        points: List<PointF>,
        params: BrowParams,
        maxWidth: Int,
        segmentation: BrowSegmentationResult? = null,
        useNativeShape: Boolean = false
    ): Bitmap {
        val scale = min(1f, maxWidth.toFloat() / source.width.coerceAtLeast(1))
        val width = max(1, (source.width * scale).toInt())
        val height = max(1, (source.height * scale).toInt())
        val scaled = Bitmap.createScaledBitmap(source, width, height, true)
        val output = try {
            scaled.copy(Bitmap.Config.ARGB_8888, true) ?: throw IllegalStateException("无法创建方案预览")
        } finally {
            if (scaled !== source && !scaled.isRecycled) scaled.recycle()
        }
        if (useNativeShape && segmentation != null && !segmentation.overlay.isRecycled) {
            val contours = extractNativeContours(segmentation)
            if (contours.size >= 2) {
                drawNativeContours(Canvas(output), contours, points, params, width.toFloat() / source.width)
            } else {
                val scaledPoints = points.map { PointF(it.x * width / source.width, it.y * height / source.height) }
                drawBrows(Canvas(output), scaledPoints, params)
            }
        } else {
            val scaledPoints = points.map { PointF(it.x * width / source.width, it.y * height / source.height) }
            drawBrows(Canvas(output), scaledPoints, params)
        }
        return output
    }

    fun createTemplate(
        points: List<PointF>,
        params: BrowParams,
        width: Int,
        height: Int,
        curvatureFactors: Map<String, Float> = emptyMap(),
        segmentation: BrowSegmentationResult? = null,
        useNativeShape: Boolean = false
    ): Bitmap? {
        if (points.size < 455 || width <= 0 || height <= 0) return null
        val contours = if (useNativeShape && segmentation != null && !segmentation.overlay.isRecycled) {
            extractNativeContours(segmentation)
        } else {
            emptyList()
        }
        if (useNativeShape && contours.size < 2) return null

        val landmarkPoints = browSets.flatMap { ids -> ids.map { index -> points.getOrNull(index) }.filterNotNull() }
        if (landmarkPoints.isEmpty()) return null
        val faceWidth = if (points.size > 454) abs(points[454].x - points[234].x) else width * 0.35f
        val maxCorrection = curvatureFactors.values.maxOrNull()?.coerceIn(1f, 1.4f) ?: 1f
        val marginX = max(8f, faceWidth * (0.06f + (maxCorrection - 1f) * 0.08f))
        val marginY = max(8f, faceWidth * 0.11f)

        val shapeLeft = if (contours.isNotEmpty()) contours.minOf { it.xStart.toFloat() }
            else landmarkPoints.minOf { it.x }
        val shapeRight = if (contours.isNotEmpty()) contours.maxOf { (it.xStart + it.top.size).toFloat() }
            else landmarkPoints.maxOf { it.x }
        val shapeTop = if (contours.isNotEmpty()) contours.minOf { it.top.minOrNull() ?: it.regionBounds[1].toFloat() }
            else landmarkPoints.minOf { it.y }
        val shapeBottom = if (contours.isNotEmpty()) contours.maxOf { it.bottom.maxOrNull() ?: it.regionBounds[3].toFloat() }
            else landmarkPoints.maxOf { it.y }
        val left = (shapeLeft - marginX).toInt().coerceIn(0, width - 1)
        val top = (shapeTop - marginY).toInt().coerceIn(0, height - 1)
        val right = (shapeRight + marginX).toInt().coerceIn(left + 1, width)
        val bottom = (shapeBottom + marginY).toInt().coerceIn(top + 1, height)
        val template = Bitmap.createBitmap(right - left, bottom - top, Bitmap.Config.ARGB_8888)
        if (contours.isNotEmpty()) {
            drawNativeContours(
                Canvas(template), contours, points, params,
                scale = 1f, imageLeft = -left.toFloat(), imageTop = -top.toFloat(),
                outlineOnly = true, curvatureFactors = curvatureFactors
            )
        } else {
            val shiftedPoints = points.map { point -> PointF(point.x - left, point.y - top) }
            drawBrows(Canvas(template), shiftedPoints, params, outlineOnly = true, curvatureFactors = curvatureFactors)
        }
        // The customer-facing preview is mirrored like a selfie. Flip only the
        // physical stencil back so left/right anatomy maps correctly on the customer.
        val unmirrored = Bitmap.createBitmap(
            template, 0, 0, template.width, template.height,
            Matrix().apply { postScale(-1f, 1f, template.width / 2f, template.height / 2f) },
            true
        )
        if (unmirrored !== template) template.recycle()
        return unmirrored
    }

    fun createTemplate(
        points: List<PointF>,
        params: BrowParams,
        width: Int,
        height: Int,
        curvatureFactors: Map<String, Float> = emptyMap()
    ): Bitmap? {
        if (points.size < 455 || width <= 0 || height <= 0) return null
        val browPoints = browSets.flatMap { ids -> ids.map { index -> points.getOrNull(index) }.filterNotNull() }
        if (browPoints.isEmpty()) return null
        val faceWidth = if (points.size > 454) abs(points[454].x - points[234].x) else width * 0.35f
        val maxCorrection = curvatureFactors.values.maxOrNull()?.coerceIn(1f, 1.4f) ?: 1f
        val marginX = max(8f, faceWidth * (0.06f + (maxCorrection - 1f) * 0.9f))
        val marginY = max(8f, faceWidth * 0.11f)
        val left = (browPoints.minOf { it.x } - marginX).toInt().coerceIn(0, width - 1)
        val top = (browPoints.minOf { it.y } - marginY).toInt().coerceIn(0, height - 1)
        val right = (browPoints.maxOf { it.x } + marginX).toInt().coerceIn(left + 1, width)
        val bottom = (browPoints.maxOf { it.y } + marginY).toInt().coerceIn(top + 1, height)
        val shiftedPoints = points.map { point -> PointF(point.x - left, point.y - top) }
        // Only allocate the small eyebrow region, not a full-resolution transparent
        // bitmap the size of the whole face photo. This reduces peak memory use.
        val template = Bitmap.createBitmap(right - left, bottom - top, Bitmap.Config.ARGB_8888)
        drawBrows(Canvas(template), shiftedPoints, params, outlineOnly = true, curvatureFactors = curvatureFactors)
        // The customer-facing preview is mirrored like a selfie. Flip only the
        // physical stencil back so left/right anatomy maps correctly on the customer.
        val unmirrored = Bitmap.createBitmap(
            template, 0, 0, template.width, template.height,
            Matrix().apply { postScale(-1f, 1f, template.width / 2f, template.height / 2f) },
            true
        )
        if (unmirrored !== template) template.recycle()
        return unmirrored
    }
}