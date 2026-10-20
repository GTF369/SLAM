package com.example.facepattern

import android.animation.Animator
import android.animation.AnimatorListenerAdapter
import android.animation.ValueAnimator
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.graphics.Shader
import android.os.Bundle
import android.os.CancellationSignal
import android.os.ParcelFileDescriptor
import android.print.PageRange
import android.print.PrintAttributes
import android.print.PrintDocumentAdapter
import android.print.PrintDocumentInfo
import android.print.PrintManager
import android.print.pdf.PrintedPdfDocument
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.animation.LinearInterpolator
import android.widget.Button
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
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

private data class BrowParams(
    val thickness: Float = 1.0f,
    val length: Float = 1.0f,
    val arch: Float = 1.0f,
    val style: Int = 0
)

private data class BrowPlan(val name: String, val params: BrowParams)

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var browView: BrowDesignView
    private lateinit var status: TextView
    private lateinit var liveControls: LinearLayout
    private lateinit var designControls: LinearLayout
    private lateinit var plansRow: LinearLayout
    private lateinit var pauseButton: Button
    private lateinit var thicknessSeek: SeekBar
    private lateinit var lengthSeek: SeekBar
    private lateinit var archSeek: SeekBar

    private var landmarker: FaceLandmarker? = null
    private var cameraExecutor: ExecutorService? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val lastTimestamp = AtomicLong(0L)
    private val cameraPermission = 701
    private var frameWidth = 640
    private var frameHeight = 480
    private var designMode = false
    private var capturedPhoto: Bitmap? = null
    private var params = BrowParams()
    private val history = mutableListOf<BrowParams>()
    private val savedPlans = mutableListOf<BrowPlan>()
    private var transitionAnimator: ValueAnimator? = null

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
            setPadding(dp(12), dp(10), dp(12), dp(12))
            setBackgroundColor(Color.argb(232, 18, 20, 25))
        }
        status = TextView(this).apply {
            text = "正在启动摄像头，请正对镜头并保持光线充足"
            textSize = 13f
            setTextColor(Color.WHITE)
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
        addTextLabel(designControls, "选择基础眉形 · 当前为可调原型，尚未训练真实 AI 适配度模型")
        addButtonRow(designControls, listOf(
            "自然款" to { animateTo(BrowParams(1.0f, 1.0f, 0.92f, 0)) },
            "柔和款" to { animateTo(BrowParams(0.82f, 0.94f, 0.72f, 1)) },
            "英气款" to { animateTo(BrowParams(1.18f, 1.06f, 1.22f, 2)) }
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
            "1:1 打印" to { askPrintCalibration() },
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
            addView(panel)
        }
        root.addView(scroll, FrameLayout.LayoutParams(-1, dp(345), Gravity.BOTTOM))
        setContentView(root)
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
            val button = Button(this).apply {
                text = label
                textSize = 12f
                minHeight = dp(42)
                setPadding(dp(3), 0, dp(3), 0)
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
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumFaces(1)
                    .setOutputFacialTransformationMatrixes(true)
                    .setResultListener { result, _ ->
                        val normalized = result.faceLandmarks().firstOrNull()?.map { PointF(it.x(), it.y()) } ?: emptyList()
                        runOnUiThread {
                            if (!designMode) {
                                val mapped = normalized.map { mapToPreview(it) }
                                browView.setFacePoints(mapped)
                                if (mapped.size >= 400) {
                                    status.text = "已识别面部关键点 · 请让顾客自然睁眼、正视镜头"
                                } else {
                                    status.text = "请正对前置摄像头，避免强逆光"
                                }
                            }
                        }
                    }
                    .setErrorListener { error ->
                        runOnUiThread { if (!designMode) status.text = "人脸识别错误：" + (error.message ?: "请重新尝试") }
                    }
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
                            frameWidth = rotated.width
                            frameHeight = rotated.height
                            val mpImage = BitmapImageBuilder(rotated).build()
                            val now = System.nanoTime() / 1_000_000L
                            val timestamp = lastTimestamp.updateAndGet { old -> maxOf(now, old + 1L) }
                            landmarker?.detectAsync(mpImage, timestamp)
                            if (rotated !== bitmap) rotated.recycle()
                        } catch (_: Exception) {
                            // Skip a broken camera frame. The next frame will be analyzed.
                        } finally {
                            image.close()
                        }
                    }
                    provider.unbindAll()
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
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
        val points = browView.facePoints()
        if (points.size < 400) {
            Toast.makeText(this, "尚未稳定识别人脸，请正视镜头后再拍照", Toast.LENGTH_SHORT).show()
            return
        }
        val bitmap = previewView.bitmap
        if (bitmap == null) {
            Toast.makeText(this, "拍照失败，请重新尝试", Toast.LENGTH_SHORT).show()
            return
        }
        capturedPhoto?.recycle()
        capturedPhoto = bitmap.copy(Bitmap.Config.ARGB_8888, false)
        designMode = true
        browView.setPhoto(capturedPhoto)
        browView.setFacePoints(points)
        previewView.visibility = View.GONE
        designControls.visibility = View.VISIBLE
        liveControls.visibility = View.GONE
        params = BrowParams()
        history.clear()
        browView.setParams(params)
        syncSliders()
        status.text = "已冻结照片，可选眉形并调整粗细、长度和弧度"
        cameraProvider?.unbindAll()
        landmarker?.close()
        landmarker = null
        cameraExecutor?.shutdown()
        cameraExecutor = null
    }

    private fun retakePhoto() {
        stopTransition()
        designMode = false
        capturedPhoto?.recycle()
        capturedPhoto = null
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
                savedPlans.add(BrowPlan(name, params.copy()))
                refreshPlanButtons()
                status.text = "已保存 " + name + "，可以继续调整并保存其他版本"
            }
            .show()
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
                setOnClickListener {
                    stopTransition()
                    history.add(params.copy())
                    params = plan.params.copy()
                    browView.setParams(params)
                    syncSliders()
                    status.text = "正在预览：" + plan.name
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
                setImageBitmap(BrowRenderer.renderPreview(photo, points, plan.params, 300))
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
                setOnClickListener {
                    params = plan.params.copy()
                    browView.setParams(params)
                    syncSliders()
                    status.text = "已选用：" + plan.name
                }
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

    private fun askPrintCalibration() {
        val points = browView.facePoints()
        if (points.size < 470 || capturedPhoto == null) {
            Toast.makeText(this, "当前照片的面部关键点不足，无法建立打印比例", Toast.LENGTH_LONG).show()
            return
        }
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "实际瞳距（毫米）"
            setText("63")
        }
        AlertDialog.Builder(this)
            .setTitle("1:1 打印校准")
            .setMessage("请先用尺测量顾客双眼瞳孔中心间的实际距离，并输入毫米数。默认值仅用于演示，不能代替实测。打印时选择 A4 与 100% / 实际尺寸，打印后量取页面中的 20 毫米方框，确认打印缩放无误。")
            .setView(input)
            .setNegativeButton("取消", null)
            .setPositiveButton("生成打印模板") { _, _ ->
                val measuredMm = input.text.toString().toFloatOrNull()
                if (measuredMm == null || measuredMm !in 35f..90f) {
                    Toast.makeText(this, "请输入合理的实测瞳距（35–90 毫米）", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val pixelDistance = BrowRenderer.pupilDistance(points)
                if (pixelDistance < 5f) {
                    Toast.makeText(this, "未能定位瞳孔，请重新拍摄正面照片", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val template = BrowRenderer.createTemplate(points, params, capturedPhoto!!.width, capturedPhoto!!.height)
                if (template == null) {
                    Toast.makeText(this, "眉形模板生成失败，请重新拍摄", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                printTemplate(template, pixelDistance / measuredMm)
            }
            .show()
    }

    private fun printTemplate(template: Bitmap, pixelsPerMm: Float) {
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
                    val pointsPerPixel = 72f / (25.4f * pixelsPerMm)
                    val widthPts = template.width * pointsPerPixel
                    val heightPts = template.height * pointsPerPixel
                    val calibrationPts = 20f * 72f / 25.4f
                    val gapPts = 12f
                    val totalHeightPts = heightPts + gapPts + calibrationPts
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

    fun setParams(newParams: BrowParams) {
        params = newParams
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val image = photo
        if (image != null && !image.isRecycled) {
            canvas.drawBitmap(image, null, RectF(0f, 0f, width.toFloat(), height.toFloat()), Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG))
        }
        BrowRenderer.drawBrows(canvas, points, params)
    }
}

private object BrowRenderer {
    private val browSets = listOf(
        intArrayOf(70, 63, 105, 66, 107),
        intArrayOf(300, 293, 334, 296, 336)
    )
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)

    fun drawBrows(canvas: Canvas, points: List<PointF>, params: BrowParams, outlineOnly: Boolean = false) {
        if (points.size < 455) return
        browSets.forEach { ids -> drawOne(canvas, points, ids, params, outlineOnly) }
    }

    private fun drawOne(canvas: Canvas, points: List<PointF>, ids: IntArray, params: BrowParams, outlineOnly: Boolean) {
        val valid = ids.map { index -> points.getOrNull(index) }.filterNotNull().filter { it.x.isFinite() && it.y.isFinite() }
        if (valid.size < 4) return
        val nose = points.getOrNull(1) ?: return
        val inner = valid.minByOrNull { abs(it.x - nose.x) } ?: return
        val outer = valid.maxByOrNull { abs(it.x - inner.x) } ?: return
        val span = hypot((outer.x - inner.x).toDouble(), (outer.y - inner.y).toDouble()).toFloat() * params.length
        if (span < 8f) return
        val dx = outer.x - inner.x
        val dy = outer.y - inner.y
        val baseStart = PointF(inner.x + dx * (1f - params.length) * 0.5f, inner.y + dy * (1f - params.length) * 0.5f)
        val baseEnd = PointF(outer.x - dx * (1f - params.length) * 0.5f, outer.y - dy * (1f - params.length) * 0.5f)
        val peakY = min(baseStart.y, baseEnd.y) - span * 0.08f * params.arch
        val control = PointF((baseStart.x + baseEnd.x) / 2f, 2f * peakY - (baseStart.y + baseEnd.y) / 2f)
        val top = mutableListOf<PointF>()
        val bottom = mutableListOf<PointF>()
        val halfThickness = span * 0.055f * params.thickness
        for (i in 0..32) {
            val t = i / 32f
            val oneMinus = 1f - t
            val x = oneMinus * oneMinus * baseStart.x + 2f * oneMinus * t * control.x + t * t * baseEnd.x
            val y = oneMinus * oneMinus * baseStart.y + 2f * oneMinus * t * control.y + t * t * baseEnd.y
            val tx = 2f * oneMinus * (control.x - baseStart.x) + 2f * t * (baseEnd.x - control.x)
            val ty = 2f * oneMinus * (control.y - baseStart.y) + 2f * t * (baseEnd.y - control.y)
            val length = max(0.001f, hypot(tx.toDouble(), ty.toDouble()).toFloat())
            var nx = -ty / length
            var ny = tx / length
            if (ny > 0f) { nx = -nx; ny = -ny }
            val taper = 0.12f + 0.88f * sin(Math.PI * t).toFloat().coerceAtLeast(0f)
            val thick = halfThickness * taper
            top.add(PointF(x + nx * thick, y + ny * thick))
            bottom.add(PointF(x - nx * thick, y - ny * thick))
        }
        val path = Path()
        path.moveTo(top.first().x, top.first().y)
        top.drop(1).forEach { path.lineTo(it.x, it.y) }
        bottom.asReversed().forEach { path.lineTo(it.x, it.y) }
        path.close()
        if (outlineOnly) {
            paint.shader = null
            paint.style = Paint.Style.STROKE
            paint.strokeWidth = max(1.0f, span * 0.003f)
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
        paint.shader = LinearGradient(baseStart.x, peakY, baseEnd.x, max(baseStart.y, baseEnd.y), colors[0], colors[1], Shader.TileMode.CLAMP)
        canvas.drawPath(path, paint)
        paint.shader = null
    }

    fun pupilDistance(points: List<PointF>): Float {
        if (points.size >= 478) {
            fun average(indices: IntRange): PointF {
                var x = 0f
                var y = 0f
                indices.forEach { x += points[it].x; y += points[it].y }
                return PointF(x / indices.count(), y / indices.count())
            }
            val left = average(468..472)
            val right = average(473..477)
            return hypot((left.x - right.x).toDouble(), (left.y - right.y).toDouble()).toFloat()
        }
        if (points.size > 263) {
            val left = points[33]
            val right = points[263]
            return hypot((left.x - right.x).toDouble(), (left.y - right.y).toDouble()).toFloat()
        }
        return 0f
    }

    fun renderPreview(source: Bitmap, points: List<PointF>, params: BrowParams, maxWidth: Int): Bitmap {
        val scale = min(1f, maxWidth.toFloat() / source.width.coerceAtLeast(1))
        val width = max(1, (source.width * scale).toInt())
        val height = max(1, (source.height * scale).toInt())
        val output = Bitmap.createScaledBitmap(source, width, height, true).copy(Bitmap.Config.ARGB_8888, true)
        val scaledPoints = points.map { PointF(it.x * width / source.width, it.y * height / source.height) }
        drawBrows(Canvas(output), scaledPoints, params)
        return output
    }

    fun createTemplate(points: List<PointF>, params: BrowParams, width: Int, height: Int): Bitmap? {
        if (points.size < 455) return null
        val full = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        drawBrows(Canvas(full), points, params, outlineOnly = true)
        val browPoints = browSets.flatMap { ids -> ids.map { index -> points.getOrNull(index) }.filterNotNull() }
        if (browPoints.isEmpty()) {
            full.recycle()
            return null
        }
        val faceWidth = if (points.size > 454) abs(points[454].x - points[234].x) else width * 0.35f
        val marginX = max(8f, faceWidth * 0.06f)
        val marginY = max(8f, faceWidth * 0.11f)
        val left = (browPoints.minOf { it.x } - marginX).toInt().coerceIn(0, width - 1)
        val top = (browPoints.minOf { it.y } - marginY).toInt().coerceIn(0, height - 1)
        val right = (browPoints.maxOf { it.x } + marginX).toInt().coerceIn(left + 1, width)
        val bottom = (browPoints.maxOf { it.y } + marginY).toInt().coerceIn(top + 1, height)
        val cropped = Bitmap.createBitmap(full, left, top, right - left, bottom - top)
        full.recycle()
        return cropped
    }
}