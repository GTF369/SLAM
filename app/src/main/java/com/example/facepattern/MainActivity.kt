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
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.CancellationSignal
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.os.ParcelFileDescriptor
import android.print.PageRange
import android.print.PrintAttributes
import android.print.PrintDocumentAdapter
import android.print.PrintDocumentInfo
import android.provider.MediaStore
import android.print.PrintManager
import android.print.pdf.PrintedPdfDocument
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
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
import androidx.activity.result.contract.ActivityResultContracts
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
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec
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

internal data class BrowParams(
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

private data class RestoredBrowCase(
    val recordId: String,
    val displayName: String,
    val anonymousCaseId: String,
    val currentParams: BrowParams,
    val currentUseNativeShape: Boolean,
    val segmentationReviewed: Boolean,
    val showSegmentation: Boolean,
    val currentSegmentation: BrowSegmentationResult?,
    val landmarks: List<PointF>,
    val depths: List<Float>,
    val plans: List<BrowPlan>
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
    // Shared by every candidate feedback row generated during the current design session.
    private var currentDesignSessionId: String = UUID.randomUUID().toString()
    private var activeStoredCaseId: String? = null
    private var activeStoredCaseName: String = ""
    private var pendingRestoredCase: RestoredBrowCase? = null
    private var caseSaveInProgress = false
    private val lastTimestamp = AtomicLong(0L)
    private val cameraPermission = 701
    private var frameWidth = 640
    private var frameHeight = 480
    private var designMode = false
    private var capturedPhoto: Bitmap? = null
    // MediaPipe z is a relative depth estimate, scaled to image width rather than millimetres.
    private var capturedLandmarkDepths: List<Float> = emptyList()
    private var params = BrowParams()
    // Explicit style intent is a user preference, not an inferred personality attribute.
    private var selectedDesignIntent: Int = 0
    private val designIntentLabels = listOf("均衡自然", "偏柔和", "偏立体")
    private val history = mutableListOf<BrowParams>()
    private val savedPlans = mutableListOf<BrowPlan>()
    private var transitionAnimator: ValueAnimator? = null
    // A longer default makes it practical to pause and compare gradual changes.
    private var transitionDurationMs: Long = 4_000L
    private var useNativeShape = false
    private var segmentationResult: BrowSegmentationResult? = null
    private var showNativeSegmentation = false
    // Only reviewed/confirmed segmentation masks may contribute brow-shape features
    // to preference collection and any future trained model.
    private var segmentationReviewed = false
    private var segmentationSensitivity = 0.72f
    private var segmentationGeneration = 0
    private val importPhotoLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            analyzeImportedPhoto(uri)
        }
    }
    private var pendingPortableCaseExport: ByteArray? = null
    private val portableCaseExportLauncher =
        registerForActivityResult(ActivityResultContracts.CreateDocument("application/octet-stream")) { uri ->
            val payload = pendingPortableCaseExport
            pendingPortableCaseExport = null
            if (uri == null) {
                payload?.fill(0)
                return@registerForActivityResult
            }
            if (payload == null) {
                Toast.makeText(this, "备份内容已失效，请重新导出", Toast.LENGTH_LONG).show()
                return@registerForActivityResult
            }
            Thread({
                var failure: String? = null
                try {
                    contentResolver.openOutputStream(uri, "w")?.use { output ->
                        output.write(payload)
                        output.flush()
                    } ?: throw IllegalStateException("无法写入所选备份位置")
                } catch (e: Exception) {
                    failure = e.message ?: "备份文件写入失败"
                    try { contentResolver.delete(uri, null, null) } catch (_: Exception) {}
                } finally {
                    payload.fill(0)
                }
                runOnUiThread {
                    Toast.makeText(
                        this,
                        if (failure == null) "加密案例备份已保存。请妥善保管独立口令，丢失后无法恢复。"
                        else "备份写入失败：" + failure,
                        Toast.LENGTH_LONG
                    ).show()
                }
            }, "PortableCaseExport").start()
        }
    private val portableCaseImportLauncher =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) promptPortableCaseImportPassword(uri)
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        // Instrumentation tests can load a synthetic, non-personal portrait to exercise
        // editing and encrypted case persistence without camera permissions or public photos.
        // Production launches never set these private extras.
        if (intent.getBooleanExtra("com.example.facepattern.INSTRUMENTATION_TEST_MODE", false)) {
            val publicPhotoUri = intent.getStringExtra("com.example.facepattern.PUBLIC_PHOTO_URI")
            if (!publicPhotoUri.isNullOrBlank()) {
                // CI stores a public portrait in the instrumentation APK, writes it to
                // MediaStore, then passes its URI through the same production importer.
                analyzeImportedPhoto(Uri.parse(publicPhotoUri))
            } else if (intent.getBooleanExtra("com.example.facepattern.SYNTHETIC_DESIGN_MODE", false)) {
                val (testPhoto, testLandmarks) = createInstrumentationPortrait()
                enterDesignMode(testPhoto, testLandmarks)
            } else {
                status.text = "仪器测试模式。相机未启动。"
            }
            return
        }
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startFaceTracking()
        } else {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), cameraPermission)
        }
    }

    private fun createInstrumentationPortrait(): Pair<Bitmap, List<PointF>> {
        val width = 800
        val height = 1000
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        canvas.drawColor(Color.rgb(241, 235, 228))
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        paint.color = Color.rgb(52, 39, 35)
        canvas.drawOval(118f, 80f, 682f, 925f, paint)
        paint.color = Color.rgb(224, 181, 153)
        canvas.drawOval(135f, 115f, 665f, 900f, paint)
        paint.color = Color.rgb(75, 51, 43)
        canvas.drawArc(125f, 55f, 675f, 440f, 180f, 180f, false, paint)
        paint.color = Color.rgb(45, 36, 34)
        canvas.drawOval(260f, 392f, 365f, 425f, paint)
        canvas.drawOval(435f, 392f, 540f, 425f, paint)
        paint.color = Color.rgb(66, 45, 39)
        canvas.drawOval(265f, 342f, 370f, 366f, paint)
        canvas.drawOval(430f, 342f, 535f, 366f, paint)
        paint.color = Color.rgb(249, 247, 240)
        canvas.drawOval(298f, 394f, 327f, 422f, paint)
        canvas.drawOval(468f, 394f, 497f, 422f, paint)
        paint.color = Color.rgb(49, 40, 38)
        canvas.drawCircle(313f, 408f, 12f, paint)
        canvas.drawCircle(483f, 408f, 12f, paint)
        paint.color = Color.rgb(133, 87, 68)
        paint.strokeWidth = 7f
        paint.strokeCap = Paint.Cap.ROUND
        canvas.drawLine(400f, 412f, 382f, 515f, paint)
        canvas.drawLine(382f, 515f, 410f, 523f, paint)
        paint.color = Color.rgb(151, 68, 67)
        paint.strokeWidth = 10f
        canvas.drawLine(335f, 650f, 465f, 650f, paint)

        val points = MutableList(478) { PointF(width * 0.5f, height * 0.5f) }
        fun point(index: Int, x: Float, y: Float) {
            points[index] = PointF(width * x, height * y)
        }
        point(234, 0.20f, 0.50f)
        point(454, 0.80f, 0.50f)
        point(10, 0.50f, 0.12f)
        point(152, 0.50f, 0.88f)
        point(1, 0.50f, 0.54f)
        point(33, 0.36f, 0.41f)
        point(263, 0.64f, 0.41f)
        point(70, 0.31f, 0.36f)
        point(63, 0.345f, 0.35f)
        point(105, 0.38f, 0.345f)
        point(66, 0.415f, 0.35f)
        point(107, 0.45f, 0.365f)
        point(300, 0.55f, 0.365f)
        point(293, 0.585f, 0.35f)
        point(334, 0.62f, 0.345f)
        point(296, 0.655f, 0.35f)
        point(336, 0.69f, 0.36f)
        return bitmap to points
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
                "导入照片" to { importPhotoFromGallery() }
            ))
            addButtonRow(this, listOf(
                "本机案例库" to { openSavedCaseLibrary() },
                "重置" to { browView.setParams(BrowParams()); params = BrowParams() }
            ))
            addButtonRow(this, listOf(
                "导出加密案例" to { exportPortableCaseBackup() },
                "导入加密案例" to { launchPortableCaseImport() }
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
        selectedDesignIntent = getSharedPreferences("brow_design_preference", Context.MODE_PRIVATE)
            .getInt("style_intent", 0).coerceIn(0, 2)
        val styleIntentRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        styleIntentRow.addView(TextView(this).apply {
            text = "风格倾向"
            textSize = 12f
            setTextColor(Color.WHITE)
        }, LinearLayout.LayoutParams(dp(66), -2))
        val styleIntentSpinner = android.widget.Spinner(this).apply {
            adapter = android.widget.ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                designIntentLabels
            )
            setSelection(selectedDesignIntent, false)
            onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
                override fun onItemSelected(
                    parent: android.widget.AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long
                ) {
                    val next = position.coerceIn(0, 2)
                    val changed = selectedDesignIntent != next
                    selectedDesignIntent = next
                    getSharedPreferences("brow_design_preference", Context.MODE_PRIVATE).edit()
                        .putInt("style_intent", next)
                        .apply()
                    if (changed && capturedPhoto != null && ::recommendationRow.isInitialized) {
                        refreshRecommendations(browView.facePoints())
                    }
                }
            }
        }
        styleIntentRow.addView(styleIntentSpinner, LinearLayout.LayoutParams(0, dp(42), 1f))
        designControls.addView(styleIntentRow)
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
                // Editing invalidates the previous review until the artist confirms again.
                segmentationReviewed = false
                useNativeShape = false
                browView.setNativeShapeMode(false)
                showNativeSegmentation = true
                browView.setSegmentationEditMode(true, includePixels = true)
                toggleNativeSegmentation()
            },
            "擦除误识别" to {
                segmentationReviewed = false
                useNativeShape = false
                browView.setNativeShapeMode(false)
                showNativeSegmentation = true
                browView.setSegmentationEditMode(true, includePixels = false)
                toggleNativeSegmentation()
            },
            "完成修正" to {
                browView.setSegmentationEditMode(false)
                segmentationReviewed = segmentationResult != null
                if (segmentationReviewed) {
                    segmentationLabel.text = "已由纹绣师确认识别层。掩膜特征可用于本次设计反馈，仍需检查实际眉形。"
                    refreshRecommendations(browView.facePoints())
                }
            }
        ))
        addButtonRow(designControls, listOf(
            "撤销笔划" to { undoMaskStroke() },
            "重做笔划" to { redoMaskStroke() }
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
        val durationRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val durationLabel = TextView(this).apply {
            text = "渐变时长 4 秒"
            textSize = 12f
            setTextColor(Color.WHITE)
        }
        durationRow.addView(durationLabel, LinearLayout.LayoutParams(dp(92), -2))
        val durationSeek = SeekBar(this).apply {
            max = 8
            progress = 2
            progressTintList = android.content.res.ColorStateList.valueOf(0xFFD5DDBF.toInt())
            thumbTintList = android.content.res.ColorStateList.valueOf(0xFFD5DDBF.toInt())
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onStopTrackingTouch(seekBar: SeekBar?) = Unit
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    if (fromUser) {
                        transitionDurationMs = (progress + 2L) * 1_000L
                        durationLabel.text = "渐变时长 ${progress + 2} 秒"
                    }
                }
            })
        }
        durationRow.addView(durationSeek, LinearLayout.LayoutParams(0, dp(38), 1f))
        designControls.addView(durationRow)
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
            "保存本机案例" to { saveCurrentCaseToLibrary() },
            "案例库" to { openSavedCaseLibrary() }
        ))
        addButtonRow(designControls, listOf(
            "导出加密案例" to { exportPortableCaseBackup() },
            "导入加密案例" to { launchPortableCaseImport() }
        ))
        addButtonRow(designControls, listOf(
            "导出方案册 PDF" to { exportPlanBookPdf() }
        ))
        addButtonRow(designControls, listOf(
            "导出原始标注照片" to { exportOriginalPhotoForAnnotation() }
        ))
        addButtonRow(designControls, listOf(
            "模型来源与许可" to { showModelLicenseNotice() }
        ))
        addButtonRow(designControls, listOf(
            "匿名案例编号" to { editAnonymousCaseId() },
            "顾客偏好" to { recordPreference() },
            "导出偏好数据" to { exportPreferenceCsv() }
        ))
        addButtonRow(designControls, listOf(
            "清除本机记录" to { clearLocalRecords() }
        ))
        addButtonRow(designControls, listOf(
            "曲率标定" to { showCurvatureCalibration() },
            "转印误差记录" to { showTransferValidationDialog() }
        ))
        addButtonRow(designControls, listOf(
            "导出转印验证 CSV" to { exportTransferValidationCsv() },
            "重新拍摄" to { retakePhoto() }
        ))
        val note = TextView(this).apply {
            text = "打印前需实测瞳距并复核 20 mm 校准框。可选的分区曲面补偿须先实测八个眉区的表面弧长。单张正面照不能自动恢复眉弓三维曲率，未经实际转印验证不应视为已验证精度。"
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

    private fun importPhotoFromGallery() {
        if (captureInProgress) return
        val anonymousCode = EditText(this).apply {
            hint = "匿名顾客代码，同一顾客重复到店请沿用"
            setSingleLine(true)
            setText(currentCaseId)
        }
        val notice = TextView(this).apply {
            text = "选择本机相册或下载目录中的照片。照片只在本机内存中分析，不会上传。为保持与前置摄像头预览一致，导入图会水平镜像后进入设计。请确认顾客已同意将该照片用于眉形方案设计。"
            textSize = 13f
            setPadding(dp(20), dp(4), dp(20), dp(8))
        }
        val consent = CheckBox(this).apply {
            text = "我已确认顾客同意将所选照片用于本次眉形设计"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(notice)
            addView(consent)
            addView(anonymousCode, LinearLayout.LayoutParams(-1, -2).apply {
                setMargins(dp(20), dp(2), dp(20), dp(2))
            })
            addView(TextView(this@MainActivity).apply {
                text = "不要填写姓名、电话号码或预约号。新顾客请点击“随机新代码”，同一顾客的授权研究记录应使用稳定匿名编号。"
                textSize = 11f
                setPadding(dp(20), dp(4), dp(20), dp(8))
            })
        }
        AlertDialog.Builder(this)
            .setTitle("导入顾客照片")
            .setView(form)
            .setNegativeButton("取消", null)
            .setNeutralButton("随机新代码") { _, _ ->
                currentCaseId = UUID.randomUUID().toString().take(8)
                Toast.makeText(this, "已生成新的匿名代码：$currentCaseId，请再次点击导入照片", Toast.LENGTH_LONG).show()
            }
            .setPositiveButton("同意并选择照片") { _, _ ->
                val safeCode = anonymousCode.text.toString().trim()
                    .filter { it.isLetterOrDigit() || it == '_' || it == '-' }
                    .take(32)
                if (safeCode.isBlank()) {
                    Toast.makeText(this, "请输入匿名代码或先生成随机代码", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                currentCaseId = safeCode
                importPhotoLauncher.launch(arrayOf("image/*"))
            }
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val chooseButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    chooseButton.isEnabled = consent.isChecked
                    consent.setOnCheckedChangeListener { _, checked -> chooseButton.isEnabled = checked }
                }
                dialog.show()
            }
    }

    private fun analyzeImportedPhoto(uri: Uri) {
        if (captureInProgress) return
        captureInProgress = true
        status.text = "正在本机解码导入照片并分析人脸关键点…"
        if (imageExecutor == null || imageExecutor?.isShutdown != false) {
            imageExecutor = Executors.newSingleThreadExecutor()
        }
        val executor = imageExecutor
        if (executor == null) {
            captureInProgress = false
            Toast.makeText(this, "照片分析线程不可用，请重试", Toast.LENGTH_LONG).show()
            return
        }
        executor.execute {
            var photo: Bitmap? = null
            var points: List<PointF> = emptyList()
            var depths: List<Float> = emptyList()
            var failure: String? = null
            var detector: FaceLandmarker? = null
            var mpImage: com.google.mediapipe.framework.image.MPImage? = null
            try {
                photo = decodeImportedPhoto(uri)
                if (photo == null) {
                    failure = "无法读取该图片。请尝试 JPEG 或 PNG 格式的本地文件。"
                } else {
                    detector = FaceLandmarker.createFromOptions(
                        this@MainActivity,
                        FaceLandmarker.FaceLandmarkerOptions.builder()
                            .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                            .setRunningMode(RunningMode.IMAGE)
                            .setNumFaces(1)
                            .build()
                    )
                    mpImage = BitmapImageBuilder(photo).build()
                    val result = detector.detect(mpImage)
                    val detected = result.faceLandmarks().firstOrNull()
                    points = detected?.map {
                        PointF(it.x() * photo!!.width, it.y() * photo!!.height)
                    } ?: emptyList()
                    depths = detected?.map { it.z() } ?: emptyList()
                    if (points.size < 400) {
                        failure = "图片中未识别到足够的人脸关键点。请使用单人、正脸、眉眼无遮挡且清晰的照片。"
                    }
                }
            } catch (e: Exception) {
                failure = "导入图片分析失败：" + (e.message ?: "请尝试另一张照片")
            } finally {
                try { mpImage?.close() } catch (_: Exception) {}
                try { detector?.close() } catch (_: Exception) {}
            }
            runOnUiThread {
                captureInProgress = false
                val selectedPhoto = photo
                if (failure != null || selectedPhoto == null) {
                    selectedPhoto?.recycle()
                    status.text = failure ?: "导入照片失败，请重试"
                    Toast.makeText(this@MainActivity, status.text, Toast.LENGTH_LONG).show()
                } else {
                    val warning = photoQualityWarning(selectedPhoto, points)
                    if (warning == null) {
                        enterDesignMode(selectedPhoto, points, depths)
                    } else {
                        AlertDialog.Builder(this@MainActivity)
                            .setTitle("导入照片质量提醒")
                            .setMessage(warning + "\n\n导入图片可能经过裁剪、滤镜或透视处理。继续前请检查双眼、眉头和眉尾定位。")
                            .setNegativeButton("重新选择") { _, _ ->
                                selectedPhoto.recycle()
                                status.text = "可以重新选择一张清晰的正面照片"
                            }
                            .setPositiveButton("仍然使用") { _, _ -> enterDesignMode(selectedPhoto, points, depths) }
                            .show()
                    }
                }
            }
        }
    }

    private fun decodeImportedPhoto(uri: Uri): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val boundsStream = contentResolver.openInputStream(uri) ?: return null
        boundsStream.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > 4096) sample *= 2
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: return null
        val orientation = try {
            contentResolver.openInputStream(uri)?.use { ExifInterface(it) }
                ?.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
                ?: ExifInterface.ORIENTATION_NORMAL
        } catch (_: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
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
        // Keep the same internal left/right convention as the selfie-camera design view.
        return Bitmap.createBitmap(
            oriented, 0, 0, oriented.width, oriented.height,
            Matrix().apply { postScale(-1f, 1f, oriented.width / 2f, oriented.height / 2f) }, true
        ).also {
            if (it !== oriented) oriented.recycle()
        }
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
        val consent = CheckBox(this).apply {
            text = "我已向顾客说明用途并取得本次拍摄和眉形设计同意"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(consentText)
            addView(consent)
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
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val captureButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    captureButton.isEnabled = consent.isChecked
                    consent.setOnCheckedChangeListener { _, checked -> captureButton.isEnabled = checked }
                }
                dialog.show()
            }
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
        val ranked = BrowRecommendationEngine.rank(
            this, points, segmentationResult, segmentationReviewed, useNativeShape,
            preferredStyle = selectedDesignIntent
        )
        recommendationLabel.text = BrowRecommendationEngine.sourceLabel(this) +
            "\n当前风格倾向：${designIntentLabels[selectedDesignIntent]}。倾向可随时修改，最终仍由顾客与设计师确认。"
        ranked.forEach { recommendation ->
            val button = Button(this).apply {
                text = when (recommendation.probabilityType) {
                    "conditional_choice" -> "${recommendation.name}\n本组选择概率 ${String.format(java.util.Locale.getDefault(), "%.1f", (recommendation.probability ?: 0f) * 100f)}%"
                    "high_rating" -> "${recommendation.name}\n获得 4–5 分概率 ${String.format(java.util.Locale.getDefault(), "%.1f", (recommendation.probability ?: 0f) * 100f)}%"
                    else -> "${recommendation.name}\n规则参考分 ${recommendation.score}"
                }
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
        // Keep multiple candidate feedback rows linked to one choice task. Reopening or
        // importing a design starts a new session while preserving the anonymous customer ID.
        currentDesignSessionId = UUID.randomUUID().toString()
        val restoredCase = pendingRestoredCase
        pendingRestoredCase = null
        if (restoredCase != null) {
            activeStoredCaseId = restoredCase.recordId
            activeStoredCaseName = restoredCase.displayName
            currentCaseId = restoredCase.anonymousCaseId
        } else {
            activeStoredCaseId = null
            activeStoredCaseName = ""
        }
        // Keep the operator-confirmed anonymous case ID from the capture/import dialog.
        // The same ID can then group authorized preference ratings and repeat measures.
        segmentationGeneration++
        savedPlans.forEach { it.segmentationSnapshot?.overlay?.recycle() }
        savedPlans.clear()
        if (::plansRow.isInitialized) refreshPlanButtons()
        segmentationResult?.overlay?.recycle()
        segmentationResult = restoredCase?.currentSegmentation
        showNativeSegmentation = restoredCase?.showSegmentation ?: false
        segmentationReviewed = restoredCase?.segmentationReviewed ?: false
        useNativeShape = restoredCase?.currentUseNativeShape ?: false
        browView.setNativeShapeMode(false)
        if (::segmentationLabel.isInitialized) segmentationLabel.text = "原生眉区识别：尚未运行"
        // A restored case owns its loaded snapshots. A new customer starts without old plans.
        savedPlans.clear()
        if (restoredCase != null) savedPlans.addAll(restoredCase.plans)
        if (::plansRow.isInitialized) refreshPlanButtons()
        capturedPhoto?.recycle()
        capturedPhoto = photo
        capturedLandmarkDepths = when {
            restoredCase != null && restoredCase.depths.size == photoPoints.size &&
                restoredCase.depths.all { it.isFinite() } -> restoredCase.depths.toList()
            normalizedDepths.size == photoPoints.size && normalizedDepths.all { it.isFinite() } ->
                normalizedDepths.map { it * photo.width }
            else -> emptyList()
        }
        designMode = true
        browView.setPhoto(photo)
        browView.setFacePoints(photoPoints)
        previewView.visibility = View.GONE
        designControls.visibility = View.VISIBLE
        liveControls.visibility = View.GONE
        params = restoredCase?.currentParams ?: BrowParams()
        history.clear()
        browView.setParams(params)
        browView.setSegmentationResult(segmentationResult, showNativeSegmentation)
        browView.setNativeShapeMode(useNativeShape)
        syncSliders()
        refreshRecommendations(photoPoints)
        cameraProvider?.unbindAll()
        imageCapture = null
        landmarker?.close()
        landmarker = null
        cameraExecutor?.shutdown()
        cameraExecutor = null
        status.text = if (restoredCase != null) {
            "已恢复本机案例“${restoredCase.displayName}”。方案参数、已保存方案和已确认的识别轮廓已载入。请重新检查人脸定位后再打印。"
        } else {
            "高分辨率照片已分析，正在自动识别原生眉区并生成初稿。请核查识别轮廓后再用于打印。"
        }
        // The synthetic instrumentation portrait deliberately skips native inference.
        // This verifies the real editing screen and encrypted case persistence without
        // pretending synthetic landmarks are real customers or model-accuracy evidence.
        val instrumentationMode = intent.getBooleanExtra(
            "com.example.facepattern.INSTRUMENTATION_TEST_MODE", false
        )
        if (instrumentationMode) {
            val publicPhotoMode = !intent.getStringExtra("com.example.facepattern.PUBLIC_PHOTO_URI").isNullOrBlank()
            status.text = when {
                restoredCase != null -> "仪器测试模式：本机案例已恢复，可继续编辑。"
                publicPhotoMode -> "仪器测试模式：真实公开照片已导入。"
                else -> "仪器测试模式：合成测试照片已加载。"
            }
            if (publicPhotoMode && intent.getBooleanExtra("com.example.facepattern.TEST_RUN_SEGMENTATION", false)) {
                runSegmentation()
            }
        } else if (segmentationResult == null) {
            // New images start a fresh segmentation pass. Stored cases keep their reviewed
            // mask snapshot so manual edits are not silently replaced by a new inference.
            runSegmentation()
        }
    }

    private fun retakePhoto() {
        stopTransition()
        activeStoredCaseId = null
        activeStoredCaseName = ""
        segmentationGeneration++
        segmentationResult?.overlay?.recycle()
        segmentationResult = null
        showNativeSegmentation = false
        segmentationReviewed = false
        useNativeShape = false
        browView.setNativeShapeMode(false)
        if (::segmentationLabel.isInitialized) segmentationLabel.text = "原生眉区识别：尚未运行"
        browView.setSegmentationResult(null, false)
        designMode = false
        capturedPhoto?.recycle()
        capturedPhoto = null
        capturedLandmarkDepths = emptyList()
        segmentationReviewed = false
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
            duration = transitionDurationMs
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
        // Freeze an in-progress morph before opening the name dialog so the saved plan
        // is exactly the intermediate state the customer asked to keep.
        stopTransition()
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
        segmentationReviewed = useNativeShape
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
        stopTransition()
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
            val parameterSummary = String.format(
                java.util.Locale.getDefault(),
                "粗细 ×%.2f  长度 ×%.2f\n弧度 ×%.2f",
                plan.params.thickness,
                plan.params.length,
                plan.params.arch
            )
            column.addView(TextView(this).apply {
                text = parameterSummary
                textSize = 10f
                setTextColor(0xFFD3D6DA.toInt())
                gravity = Gravity.CENTER
                setPadding(dp(2), dp(2), dp(2), dp(4))
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

    private fun exportPlanBookPdf() {
        val photo = capturedPhoto
        val points = browView.facePoints()
        if (photo == null || photo.isRecycled || points.size < 455) {
            Toast.makeText(this, "请先拍摄并完成眉形设计", Toast.LENGTH_SHORT).show()
            return
        }
        if (savedPlans.isEmpty()) {
            Toast.makeText(this, "请先保存至少一套方案", Toast.LENGTH_SHORT).show()
            return
        }

        val consent = CheckBox(this).apply {
            text = "我已取得顾客同意，将含人脸的方案预览导出为 PDF"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(TextView(this@MainActivity).apply {
                text = "PDF 将包含人脸效果预览、方案名称和参数，并保存到本机下载目录。文件含可识别的人脸图像。该方案册不是 1:1 拓描模板，实际打印尺寸仍须使用专用打印校准流程。"
                textSize = 13f
            })
            addView(consent)
        }
        AlertDialog.Builder(this)
            .setTitle("导出眉形方案册")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("确认并导出") { _, _ ->
                var document: android.graphics.pdf.PdfDocument? = null
                var output: java.io.OutputStream? = null
                var uri: android.net.Uri? = null
                try {
                    val pageWidth = 595
                    val pageHeight = 842
                    val margin = 28f
                    val gap = 14f
                    val itemsPerPage = 4
                    val cardWidth = (pageWidth - margin * 2f - gap) / 2
                    val cardHeight = 352f
                    val headerHeight = 72f
                    val footerY = pageHeight - 20f
                    val doc = android.graphics.pdf.PdfDocument()
                    document = doc

                    val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.rgb(34, 39, 34)
                        textSize = 21f
                        typeface = android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL)
                    }
                    val subtitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.DKGRAY
                        textSize = 9f
                    }
                    val cardPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.rgb(247, 247, 242)
                        style = Paint.Style.FILL
                    }
                    val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        color = Color.rgb(218, 220, 211)
                        style = Paint.Style.STROKE
                        strokeWidth = 0.8f
                    }
                    val bitmapPaint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
                    val plans = savedPlans.toList()
                    val pageCount = (plans.size + itemsPerPage - 1) / itemsPerPage

                    for (pageIndex in 0 until pageCount) {
                        val pageInfo = android.graphics.pdf.PdfDocument.PageInfo.Builder(
                            pageWidth, pageHeight, pageIndex + 1
                        ).create()
                        val page = doc.startPage(pageInfo)
                        val canvas = page.canvas
                        canvas.drawColor(Color.WHITE)
                        canvas.drawText("眉形设计方案册", margin, 34f, titlePaint)
                        canvas.drawText(
                            "匿名案例 " + currentCaseId + "    生成时间 " +
                                java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.getDefault()).format(java.util.Date()),
                            margin, 52f, subtitlePaint
                        )
                        canvas.drawText(
                            "效果预览，仅用于方案沟通；不是 1:1 拓描稿，也不代表已验证的三维曲面贴合。",
                            margin, 66f, subtitlePaint
                        )

                        val startIndex = pageIndex * itemsPerPage
                        val endIndex = min(plans.size, startIndex + itemsPerPage)
                        for (index in startIndex until endIndex) {
                            val localIndex = index - startIndex
                            val column = localIndex % 2
                            val row = localIndex / 2
                            val x = margin + column * (cardWidth + gap)
                            val y = headerHeight + row * (cardHeight + gap)
                            val card = RectF(x, y, x + cardWidth, y + cardHeight)
                            canvas.drawRoundRect(card, 8f, 8f, cardPaint)
                            canvas.drawRoundRect(card, 8f, 8f, borderPaint)
                            val plan = plans[index]
                            val cardTitlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                                color = Color.rgb(34, 39, 34)
                                textSize = 13f
                                typeface = android.graphics.Typeface.create("sans-serif-medium", android.graphics.Typeface.NORMAL)
                            }
                            canvas.drawText((index + 1).toString() + ". " + plan.name.take(22), x + 10f, y + 18f, cardTitlePaint)

                            val preview = BrowRenderer.renderPreview(
                                photo, points, plan.params, 480, plan.segmentationSnapshot, plan.useNativeShape
                            )
                            try {
                                val imageBounds = RectF(x + 10f, y + 28f, x + cardWidth - 10f, y + 270f)
                                val fit = min(
                                    imageBounds.width() / preview.width.coerceAtLeast(1),
                                    imageBounds.height() / preview.height.coerceAtLeast(1)
                                )
                                val drawWidth = preview.width * fit
                                val drawHeight = preview.height * fit
                                val left = imageBounds.centerX() - drawWidth / 2f
                                val top = imageBounds.centerY() - drawHeight / 2f
                                canvas.drawBitmap(
                                    preview, null,
                                    RectF(left, top, left + drawWidth, top + drawHeight),
                                    bitmapPaint
                                )
                            } finally {
                                if (!preview.isRecycled) preview.recycle()
                            }

                            val parameterPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                                color = Color.rgb(47, 51, 45)
                                textSize = 10f
                            }
                            canvas.drawText("粗细 x" + String.format(java.util.Locale.US, "%.2f", plan.params.thickness), x + 10f, y + 292f, parameterPaint)
                            canvas.drawText("长度 x" + String.format(java.util.Locale.US, "%.2f", plan.params.length), x + 10f, y + 308f, parameterPaint)
                            canvas.drawText("弧度 x" + String.format(java.util.Locale.US, "%.2f", plan.params.arch), x + 10f, y + 324f, parameterPaint)
                            val styleName = when (plan.params.style) {
                                1 -> "柔和款"
                                2 -> "立体款"
                                else -> "自然款"
                            }
                            val baseName = if (plan.useNativeShape) " · 原生眉区轮廓" else " · 参数化底稿"
                            canvas.drawText("风格：" + styleName + baseName, x + 10f, y + 340f, parameterPaint)
                        }
                        canvas.drawText(
                            "请由纹绣师与顾客共同确认。打印时关闭自动缩放并测量专用 20 mm 校准框。",
                            margin, footerY, subtitlePaint
                        )
                        canvas.drawText((pageIndex + 1).toString() + " / " + pageCount, pageWidth - margin - 28f, footerY, subtitlePaint)
                        doc.finishPage(page)
                    }

                    val filename = "眉形方案册_" + System.currentTimeMillis() + ".pdf"
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                        val values = ContentValues().apply {
                            put(MediaStore.Downloads.DISPLAY_NAME, filename)
                            put(MediaStore.Downloads.MIME_TYPE, "application/pdf")
                            put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/BrowDesign")
                            put(MediaStore.Downloads.IS_PENDING, 1)
                        }
                        uri = contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                            ?: throw IllegalStateException("无法创建 PDF 文件")
                        output = contentResolver.openOutputStream(uri!!)
                            ?: throw IllegalStateException("无法写入 PDF 文件")
                        doc.writeTo(output!!)
                        output.close()
                        output = null
                        values.clear()
                        values.put(MediaStore.Downloads.IS_PENDING, 0)
                        contentResolver.update(uri!!, values, null, null)
                        Toast.makeText(this, "方案册已保存到 Downloads/BrowDesign/" + filename, Toast.LENGTH_LONG).show()
                    } else {
                        val directory = File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "BrowDesign")
                        if (!directory.exists() && !directory.mkdirs()) throw IllegalStateException("无法创建保存目录")
                        val file = File(directory, filename)
                        FileOutputStream(file).use { stream -> doc.writeTo(stream) }
                        Toast.makeText(this, "方案册已保存：" + file.absolutePath, Toast.LENGTH_LONG).show()
                    }
                } catch (e: Exception) {
                    if (uri != null) contentResolver.delete(uri!!, null, null)
                    Toast.makeText(this, "导出方案册失败：" + (e.message ?: "请重试"), Toast.LENGTH_LONG).show()
                } finally {
                    try { output?.close() } catch (_: Exception) {}
                    try { document?.close() } catch (_: Exception) {}
                }
            }
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val exportButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    exportButton.isEnabled = consent.isChecked
                    consent.setOnCheckedChangeListener { _, checked -> exportButton.isEnabled = checked }
                }
                dialog.show()
            }
    }

    private fun exportOriginalPhotoForAnnotation() {
        val photo = capturedPhoto ?: run {
            Toast.makeText(this, "请先拍照", Toast.LENGTH_SHORT).show()
            return
        }
        val exportConsent = CheckBox(this).apply {
            text = "我已获得顾客对导出完整人脸照片并用于人工标注的单独明确授权"
            isChecked = false
        }
        val exportForm = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), 0)
            addView(TextView(this@MainActivity).apply {
                text = "该操作会将未叠加眉形的完整人脸照片保存到手机相册。照片可识别顾客身份，请勿未经授权上传、共享或用于其他用途。"
                textSize = 13f
            })
            addView(exportConsent)
        }
        AlertDialog.Builder(this)
            .setTitle("导出原始人脸照片")
            .setView(exportForm)
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
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val exportButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    exportButton.isEnabled = exportConsent.isChecked
                    exportConsent.setOnCheckedChangeListener { _, checked -> exportButton.isEnabled = checked }
                }
                dialog.show()
            }
    }


    private fun savePreviewToGallery() {
        val photo = capturedPhoto
        if (photo == null || photo.isRecycled) {
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
        val consent = CheckBox(this).apply {
            text = "我已取得顾客同意，将包含人脸和眉形的效果图保存到相册"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(TextView(this@MainActivity).apply {
                text = "效果图包含可识别的人脸。保存后会写入系统相册，不会由应用自动上传。不同意时可以取消，并继续在应用内设计。"
                textSize = 13f
            })
            addView(consent)
        }
        AlertDialog.Builder(this)
            .setTitle("保存眉形效果图")
            .setView(form)
            .setNegativeButton("取消") { _, _ ->
                if (!preview.isRecycled) preview.recycle()
            }
            .setPositiveButton("保存效果图") { _, _ ->
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
                    if (!preview.isRecycled) preview.recycle()
                }
            }
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val saveButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    saveButton.isEnabled = consent.isChecked
                    consent.setOnCheckedChangeListener { _, checked -> saveButton.isEnabled = checked }
                }
                dialog.show()
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
        val localUseConsent = CheckBox(this).apply {
            text = "我已取得顾客同意，将该测量值保存在本机，用于此顾客当前眉形设计与打印"
            isChecked = false
        }
        val researchConsent = CheckBox(this).apply {
            text = "另外同意将此测量值用于匿名跨顾客曲率统计与研究（可选）"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(TextView(this@MainActivity).apply {
                text = "测量记录保存在本机，用于当前顾客的分区打印补偿。请使用柔性尺测量，弧长与弦长必须对应同一段的端点，照片像素不能代替毫米。只有额外勾选研究授权后，该记录才纳入跨顾客 P10–P90 统计和曲率研究 CSV 导出。局部圆弧半径仅为几何近似，不是三维面部重建。"
                textSize = 13f
            })
            addView(localUseConsent)
            addView(researchConsent)
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
                    put("research_consent", researchConsent.isChecked)
                    put("timestamp", System.currentTimeMillis())
                }
                old.put(item)
                prefs.edit().putString("records", old.toString()).apply()
                val factorsByCase = mutableMapOf<String, MutableList<Float>>()
                for (index in 0 until old.length()) {
                    val row = old.optJSONObject(index) ?: continue
                    if (row.optString("region") != region.selectedItem.toString()) continue
                    if (!row.optBoolean("research_consent", true)) continue
                    val caseId = row.optString("case_id").trim()
                    val factorValue = row.optDouble("factor", Double.NaN)
                    if (caseId.isBlank() || !factorValue.isFinite() || factorValue !in 1.0..1.5708) continue
                    factorsByCase.getOrPut(caseId) { mutableListOf() }.add(factorValue.toFloat())
                }
                // Collapse repeated measurements within the same anonymous client before
                // reporting an empirical range, otherwise frequent repeat measurements
                // would be incorrectly treated as independent customers.
                val caseMeans = factorsByCase.values.map { values -> values.average().toFloat() }
                val interval = if (caseMeans.size >= 30) {
                    val low = empiricalQuantile(caseMeans, 0.10)
                    val high = empiricalQuantile(caseMeans, 0.90)
                    "该区域已有 ${caseMeans.size} 位不同顾客。按顾客内重复测量取均值后，样本 P10–P90 经验区间约为 ${"%.3f".format(low)}–${"%.3f".format(high)}。这是当前样本的描述性区间，不是个体预测区间，也不代表通用解剖范围。"
                } else {
                    "该区域已有 ${caseMeans.size} 位不同顾客的记录。至少需要约 30–50 位不同顾客才能初步评估人群区间，正式应用仍需独立测量与转印验证。"
                }
                val radiusText = if (estimatedRadiusMm == null) {
                    "半径估计不稳定，可能接近平坦"
                } else {
                    "局部圆弧近似半径 R≈${"%.1f".format(estimatedRadiusMm)} mm，曲率约 ${"%.5f".format(1f / estimatedRadiusMm)} mm⁻¹"
                }
                Toast.makeText(this, "实测弧长/空间弦长比 k = ${"%.3f".format(factor)}（打印时另按照片投影长度计算补偿）。\n" + radiusText + "\n" + interval, Toast.LENGTH_LONG).show()
            }
            .setNeutralButton("导出 CSV") { _, _ -> exportCurvatureCsv() }
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val saveButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    saveButton.isEnabled = localUseConsent.isChecked
                    localUseConsent.setOnCheckedChangeListener { _, checked ->
                        saveButton.isEnabled = checked
                    }
                }
                dialog.show()
            }
    }

    private fun empiricalQuantile(values: List<Float>, probability: Double): Float {
        if (values.isEmpty()) return Float.NaN
        val sorted = values.sorted()
        val p = probability.coerceIn(0.0, 1.0)
        val position = (sorted.size - 1) * p
        val lower = position.toInt()
        val upper = min(sorted.lastIndex, lower + 1)
        val weight = (position - lower).toFloat()
        return sorted[lower] * (1f - weight) + sorted[upper] * weight
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
        val researchRecords = (0 until records.length())
            .mapNotNull { records.optJSONObject(it) }
            .filter { it.optBoolean("research_consent", true) }
        if (researchRecords.isEmpty()) {
            Toast.makeText(this, "暂无明确授权用于跨顾客研究的曲率记录", Toast.LENGTH_SHORT).show()
            return
        }
        val csv = buildString {
            append("case_id,region,arc_mm,chord_mm,correction_factor,estimated_radius_mm,timestamp,research_consent\n")
            for (row in researchRecords) {
                append(row.optString("case_id")).append(',')
                append(row.optString("region")).append(',')
                append(row.optDouble("arc_mm")).append(',')
                append(row.optDouble("chord_mm")).append(',')
                val radius = row.optDouble("estimated_radius_mm", Double.NaN)
                append(row.optDouble("factor")).append(',')
                append(if (radius.isFinite()) radius else "").append(',')
                append(row.optLong("timestamp")).append(',')
                append(true).append('\n')
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
            refreshRecommendations(browView.facePoints())
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
        // Choosing this action is an explicit acceptance of the visible mask.
        // From this point, mask-derived features may be included in anonymous feedback.
        segmentationReviewed = true
        useNativeShape = true
        showNativeSegmentation = false
        browView.setSegmentationEditMode(false)
        browView.setSegmentationResult(result, false)
        browView.setNativeShapeMode(true)
        segmentationLabel.text = "已确认原生眉区轮廓。仍请在预览和打印前复核边界。"
        status.text = "已切换到原生眉形轮廓底稿，可微调粗细、长度和弧度"
        refreshRecommendations(browView.facePoints())
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
        // A newly generated mask has not yet been reviewed, even when the previous
        // mask was accepted. Do not use its derived features until the artist confirms it.
        segmentationReviewed = false
        useNativeShape = false
        browView.setNativeShapeMode(false)
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
                    // Show the editable mask immediately after capture so the artist can
                    // visually verify the upstream parser's label mapping and contour.
                    showNativeSegmentation = true
                    browView.setSegmentationResult(result, true)
                    val contourUsable = browView.hasUsableNativeContour()
                    useNativeShape = false
                    browView.setNativeShapeMode(false)
                    segmentationLabel.text = "识别方式：${result.method} · 掩膜质量提示 ${(result.maskQualityScore * 100).toInt()}%（按像素占比估算，非模型置信概率）· 识别像素 ${result.selectedPixelCount} · 请人工检查边界"
                    status.text = if (contourUsable) {
                        "已识别原生眉区，请检查绿色覆盖层。点击“完成修正”确认掩膜，或直接点“以原生眉形作底稿”接受并使用。"
                    } else {
                        "眉区轮廓无法可靠提取，当前保留参数化初稿。可调整灵敏度或手工补画/擦除后重新检查。"
                    }
                    refreshRecommendations(points)
                }
            }
        }
    }

    private fun editAnonymousCaseId() {
        val input = EditText(this).apply {
            setSingleLine(true)
            hint = "例如随机编号 CUST_7F2A"
            setText(currentCaseId)
            inputType = InputType.TYPE_CLASS_TEXT
        }
        val explanation = TextView(this).apply {
            text = "编号用于把同一顾客的多套方案和重复测量归为同一统计案例。若同一顾客重新拍摄或回访并参与数据收集，请在取得授权后重复使用同一随机编号。不要使用姓名、手机号、预约号或其他可识别身份的信息。点击“新随机编号”可为另一位顾客生成编号。"
            textSize = 13f
            setPadding(dp(16), dp(8), dp(16), dp(4))
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(explanation)
            addView(input)
        }
        AlertDialog.Builder(this)
            .setTitle("匿名案例编号")
            .setView(content)
            .setNegativeButton("取消", null)
            .setNeutralButton("新随机编号") { _, _ ->
                currentCaseId = UUID.randomUUID().toString().take(8)
                Toast.makeText(this, "已生成新的随机案例编号", Toast.LENGTH_SHORT).show()
            }
            .setPositiveButton("保存编号") { _, _ ->
                val sanitized = input.text.toString().trim()
                    .filter { it.isLetterOrDigit() || it == '_' || it == '-' }
                    .take(32)
                if (sanitized.length < 4) {
                    Toast.makeText(this, "请输入至少 4 位随机编号，不要使用个人身份信息", Toast.LENGTH_LONG).show()
                } else {
                    currentCaseId = sanitized
                    Toast.makeText(this, "匿名案例编号已更新，后续偏好和曲率记录将使用该编号", Toast.LENGTH_LONG).show()
                }
            }
            .show()
    }

    private fun undoMaskStroke() {
        if (browView.undoLastMaskStroke()) {
            segmentationReviewed = false
            useNativeShape = false
            browView.setNativeShapeMode(false)
            segmentationLabel.text = "已撤销上一笔掩膜编辑，请复核轮廓后点击“完成修正”。"
        } else {
            Toast.makeText(this, "没有可撤销的掩膜笔划", Toast.LENGTH_SHORT).show()
        }
    }

    private fun redoMaskStroke() {
        if (browView.redoLastMaskStroke()) {
            segmentationReviewed = false
            useNativeShape = false
            browView.setNativeShapeMode(false)
            segmentationLabel.text = "已重做掩膜笔划，请复核轮廓后点击“完成修正”。"
        } else {
            Toast.makeText(this, "没有可重做的掩膜笔划", Toast.LENGTH_SHORT).show()
        }
    }


    private fun caseLibraryRoot(): File = File(filesDir, "brow_cases")

    private fun encodeBrowParams(value: BrowParams): org.json.JSONObject =
        org.json.JSONObject().apply {
            put("thickness", value.thickness.toDouble())
            put("length", value.length.toDouble())
            put("arch", value.arch.toDouble())
            put("style", value.style)
        }

    private fun decodeBrowParams(value: org.json.JSONObject?): BrowParams {
        if (value == null) return BrowParams()
        return BrowParams(
            thickness = value.optDouble("thickness", 1.0).toFloat().coerceIn(0.55f, 1.45f),
            length = value.optDouble("length", 1.0).toFloat().coerceIn(0.80f, 1.20f),
            arch = value.optDouble("arch", 1.0).toFloat().coerceIn(0.55f, 1.45f),
            style = value.optInt("style", 0).coerceIn(0, 2)
        )
    }

    private fun duplicateSegmentation(value: BrowSegmentationResult?): BrowSegmentationResult? {
        if (value == null || value.overlay.isRecycled) return null
        val bitmap = value.overlay.copy(Bitmap.Config.ARGB_8888, true) ?: return null
        return value.copy(overlay = bitmap, regions = value.regions.map { it.copyOf() })
    }

    private val caseEncryptionAlias = "brow_case_file_key_v1"
    private val caseEncryptionMagic = "BROWCASE1".toByteArray(Charsets.US_ASCII)

    private fun caseEncryptionKey(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore")
        keyStore.load(null)
        val existing = keyStore.getKey(caseEncryptionAlias, null) as? SecretKey
        if (existing != null) return existing

        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(
                caseEncryptionAlias,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        return generator.generateKey()
    }

    private fun encryptCaseBlob(plaintext: ByteArray): ByteArray {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, caseEncryptionKey())
        val iv = cipher.iv ?: throw IllegalStateException("Android Keystore 未生成加密随机数")
        val ciphertext = cipher.doFinal(plaintext)
        return ByteArrayOutputStream(caseEncryptionMagic.size + 1 + iv.size + ciphertext.size).use { output ->
            output.write(caseEncryptionMagic)
            output.write(iv.size)
            output.write(iv)
            output.write(ciphertext)
            output.toByteArray()
        }
    }

    private fun decryptCaseBlob(payload: ByteArray): ByteArray {
        if (payload.size < caseEncryptionMagic.size + 1 + 12 + 16) {
            throw IllegalArgumentException("加密病例文件长度无效")
        }
        for (i in caseEncryptionMagic.indices) {
            if (payload[i] != caseEncryptionMagic[i]) throw IllegalArgumentException("病例文件不是受支持的加密格式")
        }
        val ivLength = payload[caseEncryptionMagic.size].toInt() and 0xff
        val ivStart = caseEncryptionMagic.size + 1
        val ciphertextStart = ivStart + ivLength
        if (ivLength !in 12..16 || ciphertextStart >= payload.size) {
            throw IllegalArgumentException("加密病例文件头损坏")
        }
        val iv = payload.copyOfRange(ivStart, ciphertextStart)
        val ciphertext = payload.copyOfRange(ciphertextStart, payload.size)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, caseEncryptionKey(), GCMParameterSpec(128, iv))
        return cipher.doFinal(ciphertext)
    }

    private fun writeEncryptedBitmapVersioned(
        bitmap: Bitmap,
        destination: File,
        format: Bitmap.CompressFormat,
        quality: Int
    ): Boolean {
        val temporary = File(destination.parentFile, destination.name + ".tmp")
        return try {
            val plainImage = ByteArrayOutputStream().use { output ->
                if (!bitmap.compress(format, quality, output)) throw IllegalStateException("图片编码失败")
                output.toByteArray()
            }
            val encryptedImage = encryptCaseBlob(plainImage)
            FileOutputStream(temporary).use { stream ->
                stream.write(encryptedImage)
                stream.fd.sync()
            }
            if (destination.exists() && !destination.delete()) throw IllegalStateException("无法更新加密图片文件")
            if (!temporary.renameTo(destination)) {
                temporary.copyTo(destination, overwrite = true)
                temporary.delete()
            }
            destination.isFile && destination.length() > 0L
        } catch (_: Exception) {
            temporary.delete()
            false
        }
    }

    private fun decodeCaseBitmap(directory: File, fileName: String): Bitmap? {
        if (fileName.isBlank() || File(fileName).name != fileName) return null
        val source = File(directory, fileName)
        if (!source.isFile) return null
        return try {
            if (fileName.endsWith(".enc", ignoreCase = true)) {
                val plaintext = decryptCaseBlob(source.readBytes())
                BitmapFactory.decodeByteArray(plaintext, 0, plaintext.size)
            } else {
                // Backward compatibility for cases saved by older unencrypted versions.
                BitmapFactory.decodeFile(source.absolutePath)
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun writeMaskSnapshot(
        directory: File,
        prefix: String,
        suffix: Long,
        result: BrowSegmentationResult?
    ): org.json.JSONObject? {
        if (result == null || result.overlay.isRecycled) return null
        val fileName = prefix + "_" + suffix + ".png.enc"
        val target = File(directory, fileName)
        if (!writeEncryptedBitmapVersioned(result.overlay, target, Bitmap.CompressFormat.PNG, 100)) {
            throw IllegalStateException("无法保存识别轮廓")
        }
        val regionsJson = org.json.JSONArray()
        result.regions.forEach { region ->
            val regionJson = org.json.JSONArray()
            region.forEach { regionJson.put(it) }
            regionsJson.put(regionJson)
        }
        return org.json.JSONObject().apply {
            put("mask_file", fileName)
            put("offset_x", result.offsetX)
            put("offset_y", result.offsetY)
            put("method", result.method)
            put("mask_quality_score", result.maskQualityScore.toDouble())
            put("selected_pixel_count", result.selectedPixelCount)
            put("regions", regionsJson)
        }
    }

    private fun readMaskSnapshot(directory: File, metadata: org.json.JSONObject?): BrowSegmentationResult? {
        if (metadata == null) return null
        val fileName = metadata.optString("mask_file", "")
        if (fileName.isBlank() || File(fileName).name != fileName) return null
        val bitmap = decodeCaseBitmap(directory, fileName) ?: return null
        val regions = mutableListOf<IntArray>()
        val regionArray = metadata.optJSONArray("regions")
        if (regionArray != null) {
            for (i in 0 until regionArray.length()) {
                val row = regionArray.optJSONArray(i) ?: continue
                if (row.length() < 4) continue
                regions.add(IntArray(row.length()) { index -> row.optInt(index) })
            }
        }
        if (regions.size < 2) {
            bitmap.recycle()
            return null
        }
        return BrowSegmentationResult(
            overlay = bitmap,
            offsetX = metadata.optInt("offset_x", 0),
            offsetY = metadata.optInt("offset_y", 0),
            method = metadata.optString("method", "本机保存的轮廓"),
            maskQualityScore = metadata.optDouble("mask_quality_score", 0.0).toFloat().coerceIn(0f, 1f),
            selectedPixelCount = metadata.optInt("selected_pixel_count", 0).coerceAtLeast(0),
            regions = regions
        )
    }

    private fun readCaseManifest(directory: File): org.json.JSONObject? {
        val candidates = listOf(
            File(directory, "case.json.enc"),
            File(directory, "case.json.enc.bak"),
            // Read legacy manifests from the earlier app version, then migrate on next save.
            File(directory, "case.json"),
            File(directory, "case.json.bak")
        )
        for (candidate in candidates) {
            if (!candidate.isFile) continue
            try {
                val bytes = candidate.readBytes()
                val plaintext = if (candidate.name.endsWith(".enc") || candidate.name.endsWith(".enc.bak")) {
                    decryptCaseBlob(bytes)
                } else {
                    bytes
                }
                return org.json.JSONObject(String(plaintext, Charsets.UTF_8))
            } catch (_: Exception) {
                // Try the last known good backup if the newest write was interrupted.
            }
        }
        return null
    }

    private fun writeCaseSnapshot(
        recordId: String,
        displayName: String,
        anonymousId: String,
        photo: Bitmap,
        currentParams: BrowParams,
        currentUseNativeShape: Boolean,
        reviewed: Boolean,
        showSegmentation: Boolean,
        currentSegmentation: BrowSegmentationResult?,
        points: List<PointF>,
        depths: List<Float>,
        plans: List<BrowPlan>
    ): Boolean {
        val directory = File(caseLibraryRoot(), recordId)
        if (!directory.exists() && !directory.mkdirs()) return false
        val timestamp = System.currentTimeMillis()
        val photoFileName = "photo_" + timestamp + ".jpg.enc"
        if (!writeEncryptedBitmapVersioned(photo, File(directory, photoFileName), Bitmap.CompressFormat.JPEG, 94)) {
            return false
        }
        return try {
            val currentMask = writeMaskSnapshot(directory, "mask_current", timestamp, currentSegmentation)
            val planJson = org.json.JSONArray()
            val keepFiles = mutableSetOf(photoFileName)
            currentMask?.optString("mask_file")?.takeIf { it.isNotBlank() }?.let { keepFiles.add(it) }
            plans.forEachIndexed { index, plan ->
                val mask = writeMaskSnapshot(directory, "mask_plan_" + index, timestamp, plan.segmentationSnapshot)
                mask?.optString("mask_file")?.takeIf { it.isNotBlank() }?.let { keepFiles.add(it) }
                planJson.put(org.json.JSONObject().apply {
                    put("name", plan.name.take(80))
                    put("params", encodeBrowParams(plan.params))
                    put("use_native_shape", plan.useNativeShape)
                    put("mask", mask ?: org.json.JSONObject.NULL)
                })
            }
            val manifest = org.json.JSONObject().apply {
                put("schema_version", 1)
                put("record_id", recordId)
                put("display_name", displayName.take(100))
                put("anonymous_case_id", anonymousId)
                put("saved_at", timestamp)
                put("photo_file", photoFileName)
                put("current_params", encodeBrowParams(currentParams))
                put("current_use_native_shape", currentUseNativeShape)
                put("segmentation_reviewed", reviewed)
                put("show_segmentation", showSegmentation)
                put("current_segmentation", currentMask ?: org.json.JSONObject.NULL)
                put("landmarks", org.json.JSONArray().apply {
                    points.forEach { point ->
                        put(org.json.JSONObject().apply {
                            put("x", point.x.toDouble())
                            put("y", point.y.toDouble())
                        })
                    }
                })
                put("landmark_depths", org.json.JSONArray().apply {
                    depths.forEach { depth -> put(depth.toDouble()) }
                })
                put("plans", planJson)
                put("storage_note", "Case photo, masks and manifest are encrypted with an Android Keystore AES-GCM key. Files are stored in app-private internal storage and are not uploaded. Copying the files to another device alone does not migrate the Keystore key.")
            }
            val manifestFile = File(directory, "case.json.enc")
            val backupFile = File(directory, "case.json.enc.bak")
            val temporary = File(directory, "case.json.enc.tmp")
            if (manifestFile.isFile) manifestFile.copyTo(backupFile, overwrite = true)
            val encryptedManifest = encryptCaseBlob(manifest.toString(2).toByteArray(Charsets.UTF_8))
            FileOutputStream(temporary).use { stream ->
                stream.write(encryptedManifest)
                stream.fd.sync()
            }
            if (manifestFile.exists() && !manifestFile.delete()) throw IllegalStateException("无法替换加密案例索引")
            if (!temporary.renameTo(manifestFile)) {
                temporary.copyTo(manifestFile, overwrite = true)
                temporary.delete()
            }
            if (!manifestFile.isFile || manifestFile.length() == 0L) throw IllegalStateException("加密案例索引保存失败")
            backupFile.delete()
            // Remove plaintext legacy manifests only after the new encrypted manifest is committed.
            File(directory, "case.json").delete()
            File(directory, "case.json.bak").delete()
            File(directory, "case.json.tmp").delete()
            // Remove stale encrypted or legacy image snapshots only after the new manifest is committed.
            keepFiles.add("case.json.enc")
            directory.listFiles()?.filter { it.isFile && (it.name.startsWith("photo_") || it.name.startsWith("mask_")) }
                ?.filter { it.name !in keepFiles }
                ?.forEach { it.delete() }
            true
        } catch (_: Exception) {
            false
        }
    }

    private fun saveCurrentCaseToLibrary() {
        if (caseSaveInProgress) {
            Toast.makeText(this, "案例仍在保存，请勿重复提交", Toast.LENGTH_SHORT).show()
            return
        }
        val photo = capturedPhoto
        if (photo == null || photo.isRecycled) {
            Toast.makeText(this, "请先拍摄或导入照片并完成设计", Toast.LENGTH_SHORT).show()
            return
        }
        val currentPoints = browView.facePoints()
        if (currentPoints.size < 400 || currentPoints.any { !it.x.isFinite() || !it.y.isFinite() }) {
            Toast.makeText(this, "当前人脸关键点不足，无法保存可准确恢复的案例。请重新识别照片后再保存。", Toast.LENGTH_LONG).show()
            return
        }
        val nameInput = EditText(this).apply {
            hint = "例如：顾客案例 A"
            setSingleLine(true)
            setText(activeStoredCaseName.ifBlank { "眉形案例 " + currentCaseId })
        }
        val note = TextView(this).apply {
            text = "案例照片、识别轮廓和关键点索引会在应用私有目录中使用 Android Keystore AES-GCM 加密保存，不会上传。请先取得顾客对本机留存的授权。加密密钥绑定本设备，单独复制文件夹无法迁移。卸载应用或清除应用数据会删除案例及密钥。"
            textSize = 12f
            setPadding(dp(18), dp(4), dp(18), dp(8))
        }
        val consent = CheckBox(this).apply {
            text = "我已取得顾客对本机保存人脸照片、识别轮廓和眉形方案的明确授权"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(note)
            addView(consent)
            addView(nameInput)
        }
        AlertDialog.Builder(this)
            .setTitle("保存案例到本机")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("保存") { _, _ ->
                val displayName = nameInput.text.toString().trim().ifBlank { "眉形案例 " + currentCaseId }
                    .take(100)
                val recordId = activeStoredCaseId ?: UUID.randomUUID().toString()
                val photoSnapshot = try { photo.copy(Bitmap.Config.ARGB_8888, false) } catch (_: Exception) { null }
                if (photoSnapshot == null) {
                    Toast.makeText(this, "无法复制照片，请重试", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                caseSaveInProgress = true
                val currentMask = duplicateSegmentation(segmentationResult)
                val planSnapshots = savedPlans.map { plan ->
                    plan.copy(params = plan.params.copy(), segmentationSnapshot = duplicateSegmentation(plan.segmentationSnapshot))
                }
                val paramsSnapshot = params.copy()
                val nativeSnapshot = useNativeShape
                val reviewSnapshot = segmentationReviewed
                val showSnapshot = showNativeSegmentation
                val anonymousSnapshot = currentCaseId
                val pointsSnapshot = currentPoints.map { PointF(it.x, it.y) }
                val depthSnapshot = if (capturedLandmarkDepths.size == pointsSnapshot.size &&
                    capturedLandmarkDepths.all { it.isFinite() }) capturedLandmarkDepths.toList() else emptyList()
                status.text = "正在将案例保存到本机…"
                Thread({
                    val saved = try {
                        writeCaseSnapshot(
                            recordId, displayName, anonymousSnapshot, photoSnapshot, paramsSnapshot,
                            nativeSnapshot, reviewSnapshot, showSnapshot, currentMask,
                            pointsSnapshot, depthSnapshot, planSnapshots
                        )
                    } catch (_: Exception) {
                        false
                    } finally {
                        if (!photoSnapshot.isRecycled) photoSnapshot.recycle()
                        currentMask?.overlay?.let { if (!it.isRecycled) it.recycle() }
                        planSnapshots.forEach { it.segmentationSnapshot?.overlay?.let { bitmap -> if (!bitmap.isRecycled) bitmap.recycle() } }
                    }
                    runOnUiThread {
                        caseSaveInProgress = false
                        if (saved) {
                            activeStoredCaseId = recordId
                            activeStoredCaseName = displayName
                            status.text = "案例已保存到本机，可从“本机案例库”重新打开。"
                            Toast.makeText(this, "案例已保存", Toast.LENGTH_LONG).show()
                        } else {
                            status.text = "案例保存失败，请检查存储空间后重试"
                            Toast.makeText(this, status.text, Toast.LENGTH_LONG).show()
                        }
                    }
                }, "BrowCaseSave").start()
            }
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val saveButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    saveButton.isEnabled = consent.isChecked
                    consent.setOnCheckedChangeListener { _, checked -> saveButton.isEnabled = checked }
                }
                dialog.show()
            }
    }


    private fun exportPortableCaseBackup() {
        val root = caseLibraryRoot()
        val directories = root.listFiles()
            ?.filter { it.isDirectory && readCaseManifest(it) != null }
            ?.sortedByDescending { it.lastModified() }
            .orEmpty()
        if (directories.isEmpty()) {
            Toast.makeText(this, "本机案例库为空。请先保存一个案例再导出备份。", Toast.LENGTH_LONG).show()
            return
        }
        val names = directories.map { directory ->
            val metadata = readCaseManifest(directory)
            val display = metadata?.optString("display_name", directory.name).orEmpty()
                .ifBlank { directory.name }
            val code = metadata?.optString("anonymous_case_id", "").orEmpty()
            val date = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.getDefault())
                .format(java.util.Date(metadata?.optLong("saved_at", directory.lastModified()) ?: directory.lastModified()))
            "$display  ·  $date" + if (code.isBlank()) "" else "\n匿名编号：$code"
        }.toTypedArray()
        var selectedIndex = 0
        AlertDialog.Builder(this)
            .setTitle("选择要导出的已保存案例")
            .setSingleChoiceItems(names, selectedIndex) { _, which -> selectedIndex = which }
            .setMessage("导出的是案例库中最近一次保存的版本。若当前编辑尚未保存，请先保存案例。备份将使用独立口令加密，不会导出本设备的 Android Keystore 密钥。")
            .setNegativeButton("取消", null)
            .setPositiveButton("下一步") { _, _ ->
                val selected = directories.getOrNull(selectedIndex)
                if (selected == null) {
                    Toast.makeText(this, "没有选中有效案例", Toast.LENGTH_SHORT).show()
                } else {
                    promptPortableCaseExportPassword(selected)
                }
            }
            .show()
    }

    private fun promptPortableCaseExportPassword(directory: File) {
        val password = EditText(this).apply {
            hint = "独立备份口令，至少 12 个字符"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setSingleLine(true)
        }
        val confirmation = EditText(this).apply {
            hint = "再次输入备份口令"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setSingleLine(true)
        }
        val consent = CheckBox(this).apply {
            text = "我已取得顾客对导出加密案例备份的明确授权"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(TextView(this@MainActivity).apply {
                text = "备份包含顾客照片、眉形轮廓和方案参数。文件将在你选择的位置保存，应用不会上传。口令不存储在应用中，忘记口令后无法恢复。建议使用唯一且足够长的口令，并与备份文件分开保管。"
                textSize = 13f
            })
            addView(password)
            addView(confirmation)
            addView(consent)
        }
        AlertDialog.Builder(this)
            .setTitle("创建口令保护的案例备份")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("加密并导出", null)
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val exportButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    fun refreshEnabled() {
                        val first = password.text.toString()
                        exportButton.isEnabled = consent.isChecked &&
                            first.length >= 12 && first == confirmation.text.toString()
                    }
                    val watcher = object : TextWatcher {
                        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
                        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = refreshEnabled()
                        override fun afterTextChanged(s: Editable?) = Unit
                    }
                    password.addTextChangedListener(watcher)
                    confirmation.addTextChangedListener(watcher)
                    consent.setOnCheckedChangeListener { _, _ -> refreshEnabled() }
                    refreshEnabled()
                    exportButton.setOnClickListener {
                        val first = password.text.toString()
                        if (first.length < 12 || first != confirmation.text.toString() || !consent.isChecked) {
                            Toast.makeText(this, "请确认口令不少于 12 个字符、两次输入一致并取得授权", Toast.LENGTH_LONG).show()
                            return@setOnClickListener
                        }
                        val passwordChars = first.toCharArray()
                        password.text?.clear()
                        confirmation.text?.clear()
                        dialog.dismiss()
                        createPortableCaseBackup(directory, passwordChars)
                    }
                }
                dialog.show()
            }
    }


    private fun createPortableCaseBackup(directory: File, password: CharArray) {
        status.text = "正在生成口令保护的案例备份…"
        Thread({
            var packedBytes: ByteArray? = null
            var encryptedBundle: ByteArray? = null
            val plainFiles = linkedMapOf<String, ByteArray>()
            var displayName = "眉形案例"
            var failure: String? = null
            try {
                val manifest = readCaseManifest(directory) ?: throw IllegalStateException("案例索引损坏，无法导出")
                displayName = manifest.optString("display_name", "眉形案例").take(50).ifBlank { "眉形案例" }
                val fileNames = PortableCaseBackup.referencedFiles(manifest)
                for (name in fileNames) {
                    val source = File(directory, name)
                    if (!source.isFile || source.length() > PortableCaseBackup.MAX_ARCHIVE_BYTES) {
                        throw IllegalStateException("案例文件缺失或过大：$name。旧格式案例请先打开并重新保存。")
                    }
                    plainFiles[name] = decryptCaseBlob(source.readBytes())
                }
                val packed = PortableCaseBackup.pack(manifest, plainFiles)
                packedBytes = packed
                encryptedBundle = PortableCaseBackup.encrypt(packed, password)
            } catch (e: Exception) {
                failure = e.message ?: "案例加密失败"
            } finally {
                password.fill('\u0000')
                plainFiles.values.forEach { it.fill(0) }
                packedBytes?.fill(0)
            }
            val payload = encryptedBundle
            runOnUiThread {
                status.text = if (failure == null) "加密案例包已准备好，请选择保存位置" else "案例备份失败"
                if (failure != null || payload == null) {
                    encryptedBundle?.fill(0)
                    Toast.makeText(this, "案例备份失败：" + (failure ?: "加密结果为空"), Toast.LENGTH_LONG).show()
                    return@runOnUiThread
                }
                pendingPortableCaseExport?.fill(0)
                pendingPortableCaseExport = payload
                val safeName = displayName
                    .map { if (it.isLetterOrDigit() || it == '_' || it == '-') it else '_' }
                    .joinToString("")
                    .trim('_')
                    .take(32)
                    .ifBlank { "brow_case" }
                portableCaseExportLauncher.launch("${safeName}_${System.currentTimeMillis()}.browbackup")
            }
        }, "PortableCaseExportBuild").start()
    }

    private fun launchPortableCaseImport() {
        portableCaseImportLauncher.launch(arrayOf("*/*"))
    }

    private fun promptPortableCaseImportPassword(uri: Uri) {
        val password = EditText(this).apply {
            hint = "输入备份口令（至少 12 个字符）"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setSingleLine(true)
        }
        val consent = CheckBox(this).apply {
            text = "我已取得顾客对导入、恢复这份人脸案例的明确授权"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(TextView(this@MainActivity).apply {
                text = "应用会先验证备份口令、加密完整性和文件结构，再将导入案例重新使用本设备的 Keystore 加密保存。原备份文件不会被修改，内容不会上传。"
                textSize = 13f
            })
            addView(password)
            addView(consent)
        }
        AlertDialog.Builder(this)
            .setTitle("导入加密案例备份")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("验证并导入", null)
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val importButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    fun refreshEnabled() {
                        importButton.isEnabled = consent.isChecked && password.text.length >= 12
                    }
                    password.addTextChangedListener(object : TextWatcher {
                        override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
                        override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) = refreshEnabled()
                        override fun afterTextChanged(s: Editable?) = Unit
                    })
                    consent.setOnCheckedChangeListener { _, _ -> refreshEnabled() }
                    refreshEnabled()
                    importButton.setOnClickListener {
                        val pass = password.text.toString()
                        if (pass.length < 12 || !consent.isChecked) return@setOnClickListener
                        val passwordChars = pass.toCharArray()
                        password.text?.clear()
                        dialog.dismiss()
                        importPortableCaseBackup(uri, passwordChars)
                    }
                }
                dialog.show()
            }
    }

    private fun readBoundedInput(uri: Uri, maxBytes: Int): ByteArray {
        val result = ByteArrayOutputStream()
        contentResolver.openInputStream(uri)?.use { input ->
            val buffer = ByteArray(8192)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                if (result.size().toLong() + count.toLong() > maxBytes.toLong()) {
                    throw IllegalArgumentException("备份文件超过大小限制")
                }
                result.write(buffer, 0, count)
            }
        } ?: throw IllegalStateException("无法读取选定备份文件")
        return result.toByteArray().also {
            if (it.isEmpty()) throw IllegalArgumentException("备份文件为空")
        }
    }

    private fun validatePortableCaseImages(archive: PortableCaseBackup.Archive) {
        val manifest = archive.manifest
        val photoName = manifest.optString("photo_file", "")
        val referenced = PortableCaseBackup.referencedFiles(manifest)
        require(referenced.contains(photoName)) { "备份缺少案例照片" }
        archive.files.forEach { (name, bytes) ->
            require(name in referenced) { "备份包含不属于当前案例的文件" }
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
            val pixels = bounds.outWidth.toLong() * bounds.outHeight.toLong()
            require(
                bounds.outWidth in 1..8000 &&
                    bounds.outHeight in 1..8000 &&
                    pixels in 1..16_000_000L
            ) {
                "备份图像尺寸无效或超过安全限制：$name"
            }
        }
    }


    private fun importPortableCaseBackup(uri: Uri, password: CharArray) {
        status.text = "正在验证并导入加密案例…"
        Thread({
            var bundle: ByteArray? = null
            var archiveBytes: ByteArray? = null
            var archive: PortableCaseBackup.Archive? = null
            var targetDirectory: File? = null
            var failure: String? = null
            var importedName = "眉形案例"
            try {
                val inputBundle = readBoundedInput(uri, PortableCaseBackup.MAX_ENCRYPTED_BUNDLE_BYTES)
                bundle = inputBundle
                val decryptedArchive = PortableCaseBackup.decrypt(inputBundle, password)
                archiveBytes = decryptedArchive
                val parsedArchive = PortableCaseBackup.unpack(decryptedArchive)
                archive = parsedArchive
                validatePortableCaseImages(parsedArchive)

                val sourceManifest = parsedArchive.manifest
                importedName = sourceManifest.optString("display_name", "眉形案例").take(100).ifBlank { "眉形案例" }
                val newRecordId = UUID.randomUUID().toString()
                val target = File(caseLibraryRoot(), newRecordId)
                targetDirectory = target
                check(target.mkdirs()) { "无法创建导入案例目录" }

                parsedArchive.files.forEach { (name, plaintext) ->
                    val destination = File(target, name)
                    val encryptedFile = encryptCaseBlob(plaintext)
                    try {
                        FileOutputStream(destination).use { output ->
                            output.write(encryptedFile)
                            output.fd.sync()
                        }
                    } finally {
                        encryptedFile.fill(0)
                    }
                }

                val manifest = org.json.JSONObject(sourceManifest.toString()).apply {
                    put("record_id", newRecordId)
                    put("display_name", importedName)
                    put("saved_at", System.currentTimeMillis())
                    put(
                        "storage_note",
                        "Portable backup imported and re-encrypted on this device with a new Android Keystore AES-GCM key. The portable backup was protected by an independent user-supplied password. The password is not stored."
                    )
                }
                val manifestBytes = manifest.toString().toByteArray(Charsets.UTF_8)
                val encryptedManifest = encryptCaseBlob(manifestBytes)
                try {
                    FileOutputStream(File(target, "case.json.enc")).use { output ->
                        output.write(encryptedManifest)
                        output.fd.sync()
                    }
                } finally {
                    manifestBytes.fill(0)
                    encryptedManifest.fill(0)
                }

                val reread = readCaseManifest(target) ?: throw IllegalStateException("导入案例索引无法重新读取")
                require(reread.optString("record_id") == newRecordId) { "导入案例编号校验失败" }
                val restoredPhoto = decodeCaseBitmap(target, reread.optString("photo_file", ""))
                    ?: throw IllegalStateException("导入案例照片无法恢复")
                restoredPhoto.recycle()

                val currentMaskValue = reread.opt("current_segmentation")
                if (currentMaskValue is org.json.JSONObject) {
                    val restoredMask = readMaskSnapshot(target, currentMaskValue)
                        ?: throw IllegalStateException("导入案例当前眉区轮廓无法恢复")
                    if (!restoredMask.overlay.isRecycled) restoredMask.overlay.recycle()
                }
                val savedPlans = reread.optJSONArray("plans") ?: org.json.JSONArray()
                for (index in 0 until savedPlans.length()) {
                    val plan = savedPlans.optJSONObject(index) ?: continue
                    val maskValue = plan.opt("mask")
                    if (maskValue is org.json.JSONObject) {
                        val restoredMask = readMaskSnapshot(target, maskValue)
                            ?: throw IllegalStateException("导入案例第 ${index + 1} 套方案轮廓无法恢复")
                        if (!restoredMask.overlay.isRecycled) restoredMask.overlay.recycle()
                    }
                }
                target.listFiles()?.filter { it.name.endsWith(".tmp") }?.forEach { it.delete() }
            } catch (e: Exception) {
                failure = when (e) {
                    is javax.crypto.AEADBadTagException -> "口令错误，或备份数据已被篡改"
                    is java.security.GeneralSecurityException -> "无法解密备份。请检查口令和文件完整性。"
                    else -> e.message ?: "导入失败"
                }
                targetDirectory?.deleteRecursively()
            } finally {
                password.fill('\u0000')
                bundle?.fill(0)
                archiveBytes?.fill(0)
                archive?.files?.values?.forEach { it.fill(0) }
            }
            runOnUiThread {
                if (failure == null) {
                    status.text = "加密案例已恢复到本机，可在案例库中打开"
                    Toast.makeText(this, "已导入“$importedName”。案例已用本设备密钥重新加密。", Toast.LENGTH_LONG).show()
                    openSavedCaseLibrary()
                } else {
                    status.text = "加密案例导入失败"
                    Toast.makeText(this, "导入失败：" + failure, Toast.LENGTH_LONG).show()
                }
            }
        }, "PortableCaseImport").start()
    }

    private fun openSavedCaseLibrary() {
        val root = caseLibraryRoot()
        val directories = root.listFiles()?.filter { it.isDirectory && readCaseManifest(it) != null }
            ?.sortedByDescending { it.lastModified() }.orEmpty()
        if (directories.isEmpty()) {
            Toast.makeText(this, "本机案例库为空。完成设计后点击“保存本机案例”", Toast.LENGTH_LONG).show()
            return
        }
        val names = directories.map { directory ->
            val metadata = readCaseManifest(directory)
            val name = metadata?.optString("display_name", directory.name).orEmpty().ifBlank { directory.name }
            val anonymous = metadata?.optString("anonymous_case_id", "").orEmpty()
            val savedAt = metadata?.optLong("saved_at", directory.lastModified()) ?: directory.lastModified()
            val date = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.getDefault())
                .format(java.util.Date(savedAt))
            name + "  ·  " + date + if (anonymous.isBlank()) "" else "\n匿名编号：" + anonymous
        }.toTypedArray()
        var selectedIndex = 0
        AlertDialog.Builder(this)
            .setTitle("本机案例库（保存在本设备）")
            .setSingleChoiceItems(names, selectedIndex) { _, which -> selectedIndex = which }
            .setNegativeButton("关闭", null)
            .setNeutralButton("删除案例") { dialog, _ ->
                val target = directories.getOrNull(selectedIndex) ?: return@setNeutralButton
                dialog.dismiss()
                val metadata = readCaseManifest(target)
                val caseName = metadata?.optString("display_name", target.name) ?: target.name
                AlertDialog.Builder(this)
                    .setTitle("删除案例")
                    .setMessage("确定删除“" + caseName + "”及其本机照片、识别轮廓和方案吗？此操作无法撤销。已导出的照片或 PDF 不会自动删除。")
                    .setNegativeButton("取消", null)
                    .setPositiveButton("删除") { _, _ ->
                        val activeId = activeStoredCaseId
                        val removed = target.deleteRecursively()
                        if (removed && activeId == target.name) {
                            activeStoredCaseId = null
                            activeStoredCaseName = ""
                        }
                        Toast.makeText(this, if (removed) "案例已删除" else "删除失败，请重试", Toast.LENGTH_SHORT).show()
                        openSavedCaseLibrary()
                    }
                    .show()
            }
            .setPositiveButton("打开") { _, _ ->
                val target = directories.getOrNull(selectedIndex)
                if (target != null) loadStoredCase(target.name)
            }
            .show()
    }

    private fun loadStoredCase(recordId: String) {
        val directory = File(caseLibraryRoot(), recordId)
        status.text = "正在读取本机案例…"
        if (imageExecutor == null || imageExecutor?.isShutdown != false) {
            imageExecutor = Executors.newSingleThreadExecutor()
        }
        val executor = imageExecutor ?: return
        executor.execute {
            var photo: Bitmap? = null
            var detector: FaceLandmarker? = null
            var mpImage: com.google.mediapipe.framework.image.MPImage? = null
            var restored: RestoredBrowCase? = null
            var points: List<PointF> = emptyList()
            var depths: List<Float> = emptyList()
            var error: String? = null
            try {
                val manifest = readCaseManifest(directory) ?: throw IllegalStateException("案例索引损坏")
                if (manifest.optInt("schema_version", 0) != 1) throw IllegalStateException("不支持的案例版本")
                val photoName = manifest.optString("photo_file", "")
                if (photoName.isBlank() || File(photoName).name != photoName) throw IllegalStateException("案例照片路径无效")
                photo = decodeCaseBitmap(directory, photoName)
                    ?: throw IllegalStateException("案例照片解密或读取失败。若病例文件来自另一台设备，需在原设备内导出可移交的加密包，不能只复制应用目录。")
                val currentMask = readMaskSnapshot(directory, manifest.optJSONObject("current_segmentation"))
                val planList = mutableListOf<BrowPlan>()
                val storedPlans = manifest.optJSONArray("plans") ?: org.json.JSONArray()
                for (index in 0 until storedPlans.length()) {
                    val item = storedPlans.optJSONObject(index) ?: continue
                    val paramsValue = decodeBrowParams(item.optJSONObject("params"))
                    val maskValue = readMaskSnapshot(directory, item.optJSONObject("mask"))
                    planList.add(BrowPlan(
                        name = item.optString("name", "方案 " + (index + 1)).take(80),
                        params = paramsValue,
                        useNativeShape = item.optBoolean("use_native_shape", false) && maskValue != null,
                        segmentationSnapshot = maskValue
                    ))
                }
                restored = RestoredBrowCase(
                    recordId = recordId,
                    displayName = manifest.optString("display_name", recordId),
                    anonymousCaseId = manifest.optString("anonymous_case_id", UUID.randomUUID().toString().take(8)),
                    currentParams = decodeBrowParams(manifest.optJSONObject("current_params")),
                    currentUseNativeShape = manifest.optBoolean("current_use_native_shape", false) && currentMask != null,
                    segmentationReviewed = manifest.optBoolean("segmentation_reviewed", false) && currentMask != null,
                    showSegmentation = manifest.optBoolean("show_segmentation", false),
                    currentSegmentation = currentMask,
                    landmarks = emptyList(),
                    depths = emptyList(),
                    plans = planList
                )
                val savedPoints = manifest.optJSONArray("landmarks")
                val restoredPoints = if (savedPoints != null && savedPoints.length() >= 400) {
                    List(savedPoints.length()) { index ->
                        val row = savedPoints.optJSONObject(index)
                            ?: throw IllegalStateException("案例关键点数据损坏")
                        PointF(row.getDouble("x").toFloat(), row.getDouble("y").toFloat()).also { point ->
                            if (!point.x.isFinite() || !point.y.isFinite()) {
                                throw IllegalStateException("案例关键点包含无效坐标")
                            }
                        }
                    }
                } else {
                    emptyList()
                }
                val savedDepths = manifest.optJSONArray("landmark_depths")
                val restoredDepths = if (savedDepths != null && savedDepths.length() == restoredPoints.size) {
                    List(savedDepths.length()) { index -> savedDepths.getDouble(index).toFloat() }
                        .takeIf { values -> values.all { it.isFinite() } }.orEmpty()
                } else {
                    emptyList()
                }
                if (restoredPoints.size >= 400) {
                    // Reuse the exact coordinates from the design session so reopening a case
                    // cannot drift due to changed model weights, lighting, or another inference pass.
                    points = restoredPoints
                    depths = restoredDepths
                } else {
                    // Compatibility path for older case manifests without stored face geometry.
                    detector = FaceLandmarker.createFromOptions(
                        this@MainActivity,
                        FaceLandmarker.FaceLandmarkerOptions.builder()
                            .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                            .setRunningMode(RunningMode.IMAGE)
                            .setNumFaces(1)
                            .build()
                    )
                    mpImage = BitmapImageBuilder(photo).build()
                    val detected = detector.detect(mpImage).faceLandmarks().firstOrNull()
                    points = detected?.map { PointF(it.x() * photo!!.width, it.y() * photo!!.height) } ?: emptyList()
                    // Match enterDesignMode's internal convention. MediaPipe relative z
                    // scaled by photo width is still not a metric skin-surface measurement.
                    depths = detected?.map { it.z() * photo!!.width } ?: emptyList()
                }
                restored = restored.copy(landmarks = points.map { PointF(it.x, it.y) }, depths = depths.toList())
                if (points.size < 400) throw IllegalStateException("案例缺少有效的人脸关键点，且无法从照片恢复定位")
            } catch (e: Exception) {
                error = e.message ?: "读取案例失败"
                restored?.currentSegmentation?.overlay?.let { if (!it.isRecycled) it.recycle() }
                restored?.plans?.forEach { it.segmentationSnapshot?.overlay?.let { bitmap -> if (!bitmap.isRecycled) bitmap.recycle() } }
                restored = null
            } finally {
                try { mpImage?.close() } catch (_: Exception) {}
                try { detector?.close() } catch (_: Exception) {}
            }
            runOnUiThread {
                val loadedPhoto = photo
                val caseState = restored
                if (error != null || loadedPhoto == null || caseState == null) {
                    loadedPhoto?.let { if (!it.isRecycled) it.recycle() }
                    status.text = error ?: "案例读取失败"
                    Toast.makeText(this@MainActivity, status.text, Toast.LENGTH_LONG).show()
                } else {
                    pendingRestoredCase = caseState
                    enterDesignMode(loadedPhoto, points, depths)
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
        val consent = CheckBox(this).apply {
            text = "我已取得顾客同意，将匿名几何特征与本次偏好评分用于改进推荐模型"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), 0)
            addView(TextView(this@MainActivity).apply {
                text = "记录的是匿名几何特征、方案参数和顾客反馈，不保存照片或身份信息。若本次比较了多套候选，请分别记录每套，所有候选会共享本次设计会话编号，且“顾客最终选择此方案”只勾选最终选中的一套。不同意时可以取消，设计功能不受影响。"
                textSize = 13f
            })
            addView(consent)
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
                val features = BrowRecommendationEngine.measurements(points, segmentationResult, segmentationReviewed, useNativeShape)
                val dataset = org.json.JSONArray(
                    getSharedPreferences("brow_preference_feedback", Context.MODE_PRIVATE).getString("records", "[]")
                )
                val consentedAnonymousId = anonymousClientCode.text.toString().trim()
                    .filter { it.isLetterOrDigit() || it == '_' || it == '-' }
                    .take(32)
                    .ifBlank { currentCaseId }
                val item = org.json.JSONObject().apply {
                    put("case_id", consentedAnonymousId)
                    put("session_id", currentDesignSessionId)
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
                    put("native_brow_area_fraction", features.nativeBrowAreaFraction.toDouble())
                    put("native_brow_side_imbalance", features.nativeBrowSideImbalance.toDouble())
                    put("native_brow_mask_available", features.nativeBrowMaskAvailable.toDouble())
                    put("native_shape_mode", features.nativeShapeMode.toDouble())
                    put("design_intent", designIntentLabels[selectedDesignIntent])
                    put("timestamp", System.currentTimeMillis())
                }
                dataset.put(item)
                getSharedPreferences("brow_preference_feedback", Context.MODE_PRIVATE).edit()
                    .putString("records", dataset.toString())
                    .apply()
                Toast.makeText(this, "已保存匿名偏好记录，可导出 CSV 用于后续模型训练", Toast.LENGTH_LONG).show()
            }
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val saveButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    saveButton.isEnabled = consent.isChecked
                    consent.setOnCheckedChangeListener { _, checked -> saveButton.isEnabled = checked }
                }
                dialog.show()
            }
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
            append("case_id,session_id,style,thickness,length,arch,rating,selected,face_width_height_ratio,interocular_face_width_ratio,jaw_face_width_ratio,brow_eye_distance_ratio,brow_asymmetry_ratio,eye_line_tilt,native_brow_area_fraction,native_brow_side_imbalance,native_brow_mask_available,native_shape_mode,timestamp,design_intent\n")
            for (i in 0 until records.length()) {
                val row = records.optJSONObject(i) ?: continue
                append(row.optString("case_id")).append(',')
                append(row.optString("session_id")).append(',')
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
                append(row.optDouble("native_brow_area_fraction")).append(',')
                append(row.optDouble("native_brow_side_imbalance")).append(',')
                append(row.optDouble("native_brow_mask_available")).append(',')
                append(row.optDouble("native_shape_mode")).append(',')
                append(row.optLong("timestamp")).append(',')
                append(row.optString("design_intent", "均衡自然")).append('\n')
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

    private fun showTransferValidationDialog() {
        val prefs = getSharedPreferences("brow_print_calibration", Context.MODE_PRIVATE)
        val profile = EditText(this).apply {
            setSingleLine(true)
            hint = "打印机、纸张、驱动配置名称"
            setText(prefs.getString("last_profile", "默认打印配置"))
        }
        val region = android.widget.Spinner(this).apply {
            adapter = android.widget.ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                curvatureRegionNames
            )
        }
        val curvatureMode = android.widget.Spinner(this).apply {
            adapter = android.widget.ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("无曲率修正", "顾客实测区域长度修正")
            )
        }
        val skinArc = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "同一端点间的皮肤表面弧长（毫米）"
        }
        val printedPath = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "已打印模板中同一段路径长度（毫米）"
        }
        val endpointError = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "实际转印后最大端点偏差（毫米，可留空）"
        }
        val note = TextView(this).apply {
            text = "用于验证真实纸面和皮肤转印误差，不会自动更改打印系数。必须对同一眉区、同一对端点测量。模板长度沿打印轮廓量取，皮肤弧长沿皮肤表面量取。端点偏差填写转印后眉头/眉尾对应定位点的最大偏离距离。请先取得顾客对这些匿名测量数据的使用授权。"
            textSize = 12f
            setPadding(dp(16), dp(4), dp(16), dp(8))
        }
        val consent = CheckBox(this).apply {
            text = "我已取得顾客同意，记录匿名的打印与转印几何误差"
            isChecked = false
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
            addView(note)
            addView(consent)
            addView(TextView(this@MainActivity).apply { text = "匿名案例编号：$currentCaseId"; textSize = 12f })
            addView(profile)
            addView(region)
            addView(curvatureMode)
            addView(skinArc)
            addView(printedPath)
            addView(endpointError)
        }
        AlertDialog.Builder(this)
            .setTitle("打印与转印误差记录")
            .setView(form)
            .setNegativeButton("取消", null)
            .setPositiveButton("保存验证记录") { _, _ ->
                val profileName = profile.text.toString().trim().take(80)
                val skinLength = skinArc.text.toString().toFloatOrNull()
                val templateLength = printedPath.text.toString().toFloatOrNull()
                val endpointText = endpointError.text.toString().trim()
                val endpointValue = if (endpointText.isBlank()) null else endpointText.toFloatOrNull()
                if (profileName.length < 3) {
                    Toast.makeText(this, "请填写有效的打印配置名称", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                if (skinLength == null || templateLength == null ||
                    !skinLength.isFinite() || !templateLength.isFinite() ||
                    skinLength !in 5f..100f || templateLength !in 5f..100f) {
                    Toast.makeText(this, "皮肤弧长和模板路径长度必须为 5–100 毫米之间的实测值", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                if (endpointText.isNotBlank() &&
                    (endpointValue == null || !endpointValue.isFinite() || endpointValue !in 0f..50f)) {
                    Toast.makeText(this, "端点偏差应为 0–50 毫米，未测量可留空", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val activePrintScale = printScaleForProfile(profileName)
                val signedError = templateLength - skinLength
                val record = org.json.JSONObject().apply {
                    put("record_id", UUID.randomUUID().toString())
                    put("case_id", currentCaseId)
                    put("printer_profile", profileName)
                    put("region", region.selectedItem.toString())
                    put("curvature_method", curvatureMode.selectedItem.toString())
                    put("skin_arc_mm", skinLength.toDouble())
                    put("printed_template_path_mm", templateLength.toDouble())
                    put("signed_length_error_mm", signedError.toDouble())
                    put("absolute_length_error_mm", abs(signedError).toDouble())
                    put("length_error_percent", (signedError / skinLength * 100f).toDouble())
                    put("endpoint_alignment_error_mm", endpointValue?.toDouble() ?: org.json.JSONObject.NULL)
                    put("print_scale_correction", activePrintScale.toDouble())
                    put("timestamp", System.currentTimeMillis())
                }
                val store = getSharedPreferences("brow_transfer_validation", Context.MODE_PRIVATE)
                val records = org.json.JSONArray(store.getString("records", "[]"))
                records.put(record)
                store.edit().putString("records", records.toString()).apply()
                Toast.makeText(
                    this,
                    "已保存匿名转印验证记录。该记录仅用于分析，不会自动调整任何打印参数。",
                    Toast.LENGTH_LONG
                ).show()
            }
            .create().also { dialog ->
                dialog.setOnShowListener {
                    val saveButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
                    saveButton.isEnabled = consent.isChecked
                    consent.setOnCheckedChangeListener { _, checked -> saveButton.isEnabled = checked }
                }
                dialog.show()
            }
    }

    private fun csvCell(value: String): String {
        val safe = if (value.firstOrNull()?.let { it in setOf('=', '+', '-', '@') } == true) "'$value" else value
        return "\"${safe.replace("\"", "\"\"")}\""
    }

    private fun exportTransferValidationCsv() {
        val records = org.json.JSONArray(
            getSharedPreferences("brow_transfer_validation", Context.MODE_PRIVATE).getString("records", "[]")
        )
        if (records.length() == 0) {
            Toast.makeText(this, "暂无转印验证记录", Toast.LENGTH_SHORT).show()
            return
        }
        val csv = buildString {
            append("case_id,printer_profile,region,curvature_method,skin_arc_mm,printed_template_path_mm,signed_length_error_mm,absolute_length_error_mm,length_error_percent,endpoint_alignment_error_mm,print_scale_correction,timestamp\n")
            for (i in 0 until records.length()) {
                val row = records.optJSONObject(i) ?: continue
                val endpoint = row.opt("endpoint_alignment_error_mm")
                append(csvCell(row.optString("case_id"))).append(',')
                append(csvCell(row.optString("printer_profile"))).append(',')
                append(csvCell(row.optString("region"))).append(',')
                append(csvCell(row.optString("curvature_method"))).append(',')
                append(row.optDouble("skin_arc_mm")).append(',')
                append(row.optDouble("printed_template_path_mm")).append(',')
                append(row.optDouble("signed_length_error_mm")).append(',')
                append(row.optDouble("absolute_length_error_mm")).append(',')
                append(row.optDouble("length_error_percent")).append(',')
                append(if (endpoint == null || endpoint == org.json.JSONObject.NULL) "" else row.optDouble("endpoint_alignment_error_mm").toString()).append(',')
                append(row.optDouble("print_scale_correction")).append(',')
                append(row.optLong("timestamp")).append('\n')
            }
        }
        val filename = "brow_transfer_validation_${System.currentTimeMillis()}.csv"
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
                val publish = ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }
                contentResolver.update(uri, publish, null, null)
            } else {
                val directory = File(getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS), "BrowDesign")
                if (!directory.exists()) directory.mkdirs()
                File(directory, filename).writeText(csv, Charsets.UTF_8)
            }
            Toast.makeText(this, "已导出转印验证 CSV 到 Downloads/BrowDesign", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "导出失败：" + (e.message ?: "请重试"), Toast.LENGTH_LONG).show()
        }
    }

    private fun clearLocalRecords() {
        AlertDialog.Builder(this)
            .setTitle("清除应用内记录")
            .setMessage("将删除本机保存的匿名偏好反馈、曲率测量和转印误差验证记录，并清除当前会话中的方案。已经保存到“本机案例库”的照片、识别轮廓和方案不会被此按钮删除，需要进入案例库逐项删除。已导出的 CSV、PDF 和相册图片也不会自动删除。此操作无法撤销。")
            .setNegativeButton("取消", null)
            .setPositiveButton("清除") { _, _ ->
                getSharedPreferences("brow_preference_feedback", Context.MODE_PRIVATE).edit().clear().apply()
                getSharedPreferences("brow_curvature_samples", Context.MODE_PRIVATE).edit().clear().apply()
                getSharedPreferences("brow_transfer_validation", Context.MODE_PRIVATE).edit().clear().apply()
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

    /**
     * Return customer-level median surface arc lengths. The stored arc/chord ratio is
     * useful as a measurement sanity check, but is NOT the final image-to-surface scale.
     * Printing recomputes that scale from the current photo's projected segment length
     * and the measured interpupillary distance.
     */
    private fun currentCurvatureArcLengthsForPrint(): Map<String, Float> {
        val records = org.json.JSONArray(
            getSharedPreferences("brow_curvature_samples", Context.MODE_PRIVATE).getString("records", "[]")
        )
        val valuesByRegion = mutableMapOf<String, MutableList<Float>>()
        for (i in 0 until records.length()) {
            val row = records.optJSONObject(i) ?: continue
            if (row.optString("case_id") != currentCaseId) continue
            val region = row.optString("region")
            val arcMm = row.optDouble("arc_mm", Double.NaN).toFloat()
            val chordMm = row.optDouble("chord_mm", Double.NaN).toFloat()
            val ratio = if (chordMm > 0f) arcMm / chordMm else Float.NaN
            if (region !in curvatureRegionNames ||
                !arcMm.isFinite() || !chordMm.isFinite() ||
                arcMm <= 0f || chordMm <= 0f ||
                !ratio.isFinite() || ratio !in 1.0f..1.5708f) continue
            valuesByRegion.getOrPut(region) { mutableListOf() }.add(arcMm)
        }
        return valuesByRegion.mapValues { (_, arcs) ->
            val sorted = arcs.sorted()
            if (sorted.size % 2 == 1) sorted[sorted.size / 2]
            else (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2f
        }
    }

    /** Pixel chord of the exact landmark segment used for the selected anatomical region. */
    private fun projectedBrowSegmentPixels(points: List<PointF>, region: String): Float? {
        if (points.size < 455 || region !in curvatureRegionNames) return null
        val noseX = points.getOrNull(1)?.x ?: return null
        val browSets = listOf(
            intArrayOf(70, 63, 105, 66, 107),
            intArrayOf(300, 293, 334, 296, 336)
        )
        for (ids in browSets) {
            val ordered = ids.map { index -> points.getOrNull(index) }.filterNotNull()
                .filter { it.x.isFinite() && it.y.isFinite() }
                .sortedBy { it.x }
            if (ordered.size != ids.size) continue
            val sideCenter = ordered.map { it.x }.average().toFloat()
            val anatomicalRight = sideCenter < noseX
            val segmentNames = if (anatomicalRight) {
                listOf("右眉尾", "右眉峰", "右眉身", "右眉头")
            } else {
                listOf("左眉头", "左眉身", "左眉峰", "左眉尾")
            }
            val segmentIndex = segmentNames.indexOf(region)
            if (segmentIndex < 0) continue
            val a = ordered[segmentIndex]
            val b = ordered[segmentIndex + 1]
            return kotlin.math.hypot((b.x - a.x).toDouble(), (b.y - a.y).toDouble()).toFloat()
        }
        return null
    }

    /**
     * Map each physically measured surface arc to its projected length in the current
     * photo, converted to millimetres using measured interpupillary distance. This is
     * the image-to-surface scale, unlike arc_mm / spatial_chord_mm.
     */
    private fun currentCurvatureFactorsForPrint(
        points: List<PointF>,
        interpupillaryDistanceMm: Float,
        surfaceArcsMm: Map<String, Float>
    ): Map<String, Float> {
        if (!interpupillaryDistanceMm.isFinite() || interpupillaryDistanceMm !in 35f..90f) {
            return emptyMap()
        }
        val pupilDistancePx = BrowRenderer.pupilDistance(points)
        if (!pupilDistancePx.isFinite() || pupilDistancePx < 5f) return emptyMap()
        val factors = linkedMapOf<String, Float>()
        for (region in curvatureRegionNames) {
            val surfaceArcMm = surfaceArcsMm[region] ?: continue
            val projectedSegmentPx = projectedBrowSegmentPixels(points, region) ?: continue
            if (!surfaceArcMm.isFinite() || surfaceArcMm <= 0f ||
                !projectedSegmentPx.isFinite() || projectedSegmentPx < 2f) continue
            val factor = BrowSurfaceCompensation.factorFromMeasurements(
                measuredSurfaceArcMm = surfaceArcMm,
                projectedSegmentPixels = projectedSegmentPx,
                pupilDistancePixels = pupilDistancePx,
                measuredInterpupillaryDistanceMm = interpupillaryDistanceMm
            ) ?: continue
            factors[region] = factor
        }
        return factors
    }

    // No single-image relative-z estimator participates in physical print geometry.

    private fun printProfileSlug(raw: String): String = raw.trim().lowercase()
        .map { if (it.isLetterOrDigit()) it else '_' }
        .joinToString("")
        .replace(Regex("_+"), "_")
        .trim('_')
        .take(48)
        .ifBlank { "default" }

    private fun printProfilePreferenceKey(raw: String): String =
        "scale_correction_profile_${printProfileSlug(raw)}"

    private fun printScaleForProfile(raw: String): Float {
        val prefs = getSharedPreferences("brow_print_calibration", Context.MODE_PRIVATE)
        val key = printProfilePreferenceKey(raw)
        if (prefs.contains(key)) return prefs.getFloat(key, 1f)
        // Migrate the former global coefficient only into the default profile.
        return if (printProfileSlug(raw) == printProfileSlug("默认打印配置")) {
            prefs.getFloat("scale_correction", 1f)
        } else {
            1f
        }
    }

    private fun askPrintCalibration() {
        val points = browView.facePoints()
        if (points.size < 478 || capturedPhoto == null) {
            Toast.makeText(this, "未检测到完整虹膜关键点，暂时无法可靠建立 1:1 打印比例，请重新拍摄", Toast.LENGTH_LONG).show()
            return
        }
        val measuredArcs = currentCurvatureArcLengthsForPrint()
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "请输入实测瞳距（毫米）"
            setText("")
        }
        val calibrationInfo = TextView(this).apply {
            text = "已记录当前顾客的表面弧长：${measuredArcs.size}/8 区域。打印时会用当前照片各眉区的投影长度（按实测瞳距换算）计算补偿比例。"
            textSize = 12f
            setPadding(dp(4), dp(6), dp(4), dp(6))
        }
        val printScalePrefs = getSharedPreferences("brow_print_calibration", Context.MODE_PRIVATE)
        val profileInput = EditText(this).apply {
            hint = "例如：办公室打印机型号_A4_普通纸_驱动名称"
            setSingleLine(true)
            setText(printScalePrefs.getString("last_profile", "默认打印配置"))
        }
        val savedPrintScaleCorrection = printScaleForProfile(profileInput.text.toString())
        val printScaleInput = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
            hint = "上一张打印稿中 20 mm 方框的实测边长，未测量留空"
            setText("")
        }
        val printScaleState = TextView(this).apply {
            text = "配置：${profileInput.text}。当前比例修正系数：x${(savedPrintScaleCorrection * 1000f).roundToInt() / 1000f}。仅适用于同一打印机、纸张和设置。"
            textSize = 11f
            setPadding(dp(4), dp(2), dp(4), dp(4))
        }
        profileInput.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                val name = s?.toString().orEmpty().trim().ifBlank { "未命名配置" }
                val scale = printScaleForProfile(name)
                printScaleState.text = "配置：$name。当前比例修正系数：x${(scale * 1000f).roundToInt() / 1000f}。仅适用于同一打印机、纸张和设置。"
            }
            override fun afterTextChanged(s: Editable?) = Unit
        })
        val resetPrintScale = CheckBox(this).apply {
            text = "更换打印机、纸张或打印设置后，重置修正系数"
            isChecked = false
        }
        val applyCorrection = CheckBox(this).apply {
            text = "应用此顾客已实测的眉弓曲面长度补偿（仍需转印验证）"
            // Physical measurements remain opt-in and have priority over estimated depth.
            isChecked = false
            isEnabled = measuredArcs.size == curvatureRegionNames.size
        }
        val note = TextView(this).apply {
            text = "曲面补偿以真实测得的八个眉区表面弧长为依据，并按本张照片的眉区投影段长度与实测瞳距换算比例。MediaPipe 相对 z 不作为毫米级深度使用。若补偿比例超出 0.75–1.60，程序会拒绝该次补偿。该方法是分区弧长映射，不等于单张照片三维重建，仍需独立 3D/物理测量与真实转印验证。更换打印机、纸张或驱动设置后，仍需重新测量 20 mm 校准框。"
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
                text = "打印配置名称（设备、纸张和驱动设置）"
                textSize = 12f
                setPadding(dp(4), dp(4), dp(4), 0)
            })
            addView(profileInput)
            addView(TextView(this@MainActivity).apply {
                text = "打印比例校准（可选，按上方配置分别保存）"
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
                val printProfileName = profileInput.text.toString().trim().take(80)
                if (printProfileName.length < 3) {
                    Toast.makeText(this, "请为打印机、纸张和驱动设置填写一个易识别的配置名称", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val printProfileKey = printProfilePreferenceKey(printProfileName)
                val correctionBase = if (resetPrintScale.isChecked) 1f else printScaleForProfile(printProfileName)
                val printScaleCorrection = measuredSquareMm?.let { correctionBase * 20f / it } ?: correctionBase
                if (printScaleCorrection !in 0.80f..1.25f) {
                    Toast.makeText(this, "计算出的打印修正超出 0.80–1.25。请检查是否选择了实际尺寸、A4 纸张和正确的校准框，必要时重置比例。", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                printScalePrefs.edit()
                    .putFloat(printProfileKey, printScaleCorrection)
                    .putString("last_profile", printProfileName)
                    .apply()

                val pixelDistance = BrowRenderer.pupilDistance(points)
                if (pixelDistance < 5f) {
                    Toast.makeText(this, "未能定位瞳孔，请重新拍摄正面照片", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val useMeasuredCurvature = applyCorrection.isChecked
                if (useMeasuredCurvature && measuredArcs.size != curvatureRegionNames.size) {
                    Toast.makeText(this, "8 个区域的真实表面弧长记录不完整，不能应用曲面补偿", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                // Convert each measured skin-surface arc to the current photo's projected
                // segment scale. Do not treat arc/chord ratio or MediaPipe relative-z as
                // the final image-to-surface correction factor.
                val factorsToApply = if (useMeasuredCurvature) {
                    currentCurvatureFactorsForPrint(points, measuredMm, measuredArcs)
                } else {
                    emptyMap()
                }
                if (useMeasuredCurvature && factorsToApply.size != curvatureRegionNames.size) {
                    Toast.makeText(this, "无法将八个实测弧长与当前照片眉区及瞳距匹配，请重新检查关键点和测量端点", Toast.LENGTH_LONG).show()
                    return@setPositiveButton
                }
                val outOfRangeFactors = factorsToApply.filterValues {
                    !it.isFinite() || it !in 0.75f..1.60f
                }
                if (useMeasuredCurvature && outOfRangeFactors.isNotEmpty()) {
                    Toast.makeText(
                        this,
                        "曲面补偿比例超出 0.75–1.60，已拒绝本次补偿。请核对弧长、端点、照片姿态和瞳距。",
                        Toast.LENGTH_LONG
                    ).show()
                    return@setPositiveButton
                }
                val useCurvature = useMeasuredCurvature
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
        // Test-mode launches intentionally use synthetic landmarks or a supplied test
        // image. Do not start the camera pipeline on resume, which would load a native
        // MediaPipe library that may not exist in the x86_64 instrumentation APK.
        val instrumentationTestMode = intent.getBooleanExtra(
            "com.example.facepattern.INSTRUMENTATION_TEST_MODE", false
        )
        if (!instrumentationTestMode && !designMode && landmarker == null &&
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
    val reason: String,
    // Present only when a separately calibrated and reviewed probability model is loaded.
    val probability: Float? = null,
    val probabilityType: String? = null,
    val choiceLogit: Float? = null
)

private data class BrowFaceMeasurements(
    val widthHeightRatio: Float,
    val interocularFaceWidthRatio: Float,
    val jawFaceWidthRatio: Float,
    val browEyeDistanceRatio: Float,
    val browAsymmetryRatio: Float,
    val eyeLineTilt: Float,
    val nativeBrowAreaFraction: Float = 0f,
    val nativeBrowSideImbalance: Float = 0f,
    val nativeBrowMaskAvailable: Float = 0f,
    val nativeShapeMode: Float = 0f
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
    private data class TrainedChoiceModel(
        val means: FloatArray,
        val scales: FloatArray,
        val coefficients: FloatArray,
        val temperature: Float
    )
    private var loadAttempted = false
    private var trainedModel: TrainedModel? = null
    private var choiceLoadAttempted = false
    private var trainedChoiceModel: TrainedChoiceModel? = null

    fun measurements(
        points: List<PointF>,
        segmentation: BrowSegmentationResult? = null,
        maskReviewed: Boolean = false,
        nativeShapeMode: Boolean = false
    ): BrowFaceMeasurements {
        val maskFeatures = nativeBrowMaskFeatures(segmentation, maskReviewed)
        if (points.size <= 454) {
            return BrowFaceMeasurements(0f, 0f, 0f, 0f, 0f, 0f,
                maskFeatures[0], maskFeatures[1], maskFeatures[2], if (nativeShapeMode) 1f else 0f)
        }
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
            eyeTilt,
            maskFeatures[0],
            maskFeatures[1],
            maskFeatures[2],
            if (nativeShapeMode) 1f else 0f
        )
    }

    /**
     * Mask area fractions describe the reviewed pixels relative to each model ROI.
     * They are engineering features, not measurements of real hair thickness or
     * proof that the parser mask is anatomically correct.
     */
    private fun nativeBrowMaskFeatures(
        segmentation: BrowSegmentationResult?,
        reviewed: Boolean
    ): FloatArray {
        if (!reviewed || segmentation == null || segmentation.overlay.isRecycled ||
            segmentation.regions.size < 2) return floatArrayOf(0f, 0f, 0f)
        return try {
            val mask = segmentation.overlay
            val pixels = IntArray(mask.width * mask.height)
            mask.getPixels(pixels, 0, mask.width, 0, 0, mask.width, mask.height)
            val fractions = segmentation.regions.take(2).mapNotNull { bounds ->
                if (bounds.size < 4) return@mapNotNull null
                val x0 = (bounds[0] - segmentation.offsetX).coerceIn(0, mask.width)
                val y0 = (bounds[1] - segmentation.offsetY).coerceIn(0, mask.height)
                val x1 = (bounds[2] - segmentation.offsetX).coerceIn(x0, mask.width)
                val y1 = (bounds[3] - segmentation.offsetY).coerceIn(y0, mask.height)
                val area = (x1 - x0) * (y1 - y0)
                if (area <= 0) return@mapNotNull null
                var selected = 0
                for (y in y0 until y1) {
                    val row = y * mask.width
                    for (x in x0 until x1) {
                        if (Color.alpha(pixels[row + x]) > 0) selected++
                    }
                }
                selected.toFloat() / area.toFloat()
            }
            if (fractions.size != 2) floatArrayOf(0f, 0f, 0f)
            else floatArrayOf(
                ((fractions[0] + fractions[1]) / 2f).coerceIn(0f, 1f),
                abs(fractions[0] - fractions[1]).coerceIn(0f, 1f),
                1f
            )
        } catch (_: Exception) {
            floatArrayOf(0f, 0f, 0f)
        }
    }

    fun hasApprovedModel(context: Context): Boolean =
        loadChoiceModel(context) != null || loadModel(context) != null

    fun sourceLabel(context: Context): String = when {
        loadChoiceModel(context) != null ->
            "已启用经过独立顾客集校准、外部验证和人工审核的条件选择模型。概率表示顾客在本次展示的自然款、柔和款和立体款三项中选择各候选的估计概率，三项合计为 100%。它不是客观审美概率，也不保证治疗结果。"
        loadModel(context) != null ->
            "已启用经过独立顾客集校准、外部验证和人工审核的高分模型。百分比表示候选获得 4–5 分评价的估计概率，各候选概率不要求合计为 100%。"
        else ->
            "当前使用几何规则参考分，并非 AI 概率。只有经过真实顾客偏好数据训练、独立校准验证并完成人工审核的模型才会显示概率。"
    }

    fun rank(
        context: Context,
        points: List<PointF>,
        segmentation: BrowSegmentationResult?,
        maskReviewed: Boolean,
        nativeShapeMode: Boolean,
        preferredStyle: Int = 0
    ): List<BrowRecommendation> {
        val features = measurements(points, segmentation, maskReviewed, nativeShapeMode)
        val baseTemplates = listOf(
            BrowRecommendation("自然款", 82, BrowParams(1.0f, 1.0f, 0.92f, 0), "自然款几何初稿"),
            BrowRecommendation("柔和款", 78, BrowParams(0.82f, 0.94f, 0.72f, 1), "柔和款几何初稿"),
            BrowRecommendation("立体款", 76, BrowParams(1.18f, 1.06f, 1.22f, 2), "立体款几何初稿")
        )
        // Without a trained preference model, the initial contour still receives a
        // bounded, explainable geometry adjustment. This is not a learned probability.
        val adaptedParams = BrowGeometryAdapter.personalizeCandidates(
            baseTemplates.map { it.params },
            widthHeightRatio = features.widthHeightRatio,
            interocularFaceWidthRatio = features.interocularFaceWidthRatio,
            jawFaceWidthRatio = features.jawFaceWidthRatio,
            browEyeDistanceRatio = features.browEyeDistanceRatio,
            browAsymmetryRatio = features.browAsymmetryRatio,
            eyeLineTilt = features.eyeLineTilt,
            nativeBrowAreaFraction = features.nativeBrowAreaFraction,
            nativeBrowSideImbalance = features.nativeBrowSideImbalance,
            nativeBrowMaskAvailable = features.nativeBrowMaskAvailable
        )
        val intentAdjustedParams = BrowGeometryAdapter.applyStyleIntent(
            adaptedParams, preferredStyle.coerceIn(0, 2)
        )
        val templates = baseTemplates.mapIndexed { index, candidate ->
            candidate.copy(
                params = intentAdjustedParams[index],
                reason = "${candidate.name}：根据面部比例、可选的已确认原生眉区和用户风格倾向做了有限幅度的几何适配。结果是可编辑初稿，风格倾向不是人格识别。"
            )
        }
        val choiceModel = loadChoiceModel(context)
        if (choiceModel != null) {
            return rankWithChoiceModel(features, templates, choiceModel)
        }
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
                                if (candidateParams.style == 2) 1f else 0f,
                                features.nativeBrowAreaFraction,
                                features.nativeBrowSideImbalance,
                                features.nativeBrowMaskAvailable,
                                features.nativeShapeMode
                            )
                            val preference = predict(model, x)
                            candidates.add(
                                family.copy(
                                    name = "${family.name}·自适应",
                                    score = (preference * 100f).roundToInt().coerceIn(0, 100),
                                    params = candidateParams,
                                    reason = "${family.name}：参数组合经已审核的顾客偏好模型排序，仍需顾客和纹绣师确认",
                                    probability = preference,
                                    probabilityType = "high_rating"
                                )
                            )
                        }
                    }
                }
                candidates.maxBy { it.probability ?: it.score.toFloat() / 100f }
            }.sortedByDescending { it.probability ?: it.score.toFloat() / 100f }
        }
        val shapeSignal = ((features.widthHeightRatio - 0.72f) / 0.18f).coerceIn(-1f, 1f)
        val eyeSpacingSignal = ((features.interocularFaceWidthRatio - 0.40f) / 0.08f).coerceIn(-1f, 1f)
        val jawSignal = ((features.jawFaceWidthRatio - 0.75f) / 0.15f).coerceIn(-1f, 1f)
        val browGapSignal = ((features.browEyeDistanceRatio - 0.06f) / 0.04f).coerceIn(-1f, 1f)
        val tiltSignal = (features.eyeLineTilt * 4f).coerceIn(-1f, 1f)
        val maskImbalance = if (features.nativeBrowMaskAvailable >= 0.5f) {
            features.nativeBrowSideImbalance.coerceIn(0f, 1f)
        } else 0f
        val penalty = (features.browAsymmetryRatio * 35f).coerceIn(0f, 5f)
        val naturalScore = 82f - abs(shapeSignal) * 3.5f - penalty +
            (if (preferredStyle == 0) 4f else 0f) +
            (if (preferredStyle == 1) 1f else 0f) +
            (if (preferredStyle == 2) 1f else 0f)
        val softScore = 78f +
            (-jawSignal).coerceAtLeast(0f) * 3.5f +
            (-eyeSpacingSignal).coerceAtLeast(0f) * 2f +
            browGapSignal.coerceAtLeast(0f) * 1.5f -
            penalty * 0.6f +
            (if (preferredStyle == 1) 8f else 0f)
        val definedScore = 76f +
            jawSignal.coerceAtLeast(0f) * 3f +
            eyeSpacingSignal.coerceAtLeast(0f) * 2f +
            shapeSignal.coerceAtLeast(0f) * 2f +
            tiltSignal.coerceAtLeast(0f) -
            penalty * 0.4f -
            maskImbalance * 2f +
            (if (preferredStyle == 2) 8f else 0f)
        return listOf(
            templates[0].copy(score = naturalScore.roundToInt().coerceIn(60, 95)),
            templates[1].copy(score = softScore.roundToInt().coerceIn(60, 95)),
            templates[2].copy(score = definedScore.roundToInt().coerceIn(60, 95))
        ).sortedByDescending { it.score }
    }

    private fun candidateFeatures(features: BrowFaceMeasurements, candidateParams: BrowParams): FloatArray =
        floatArrayOf(
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
            if (candidateParams.style == 2) 1f else 0f,
            features.nativeBrowAreaFraction,
            features.nativeBrowSideImbalance,
            features.nativeBrowMaskAvailable,
            features.nativeShapeMode
        )

    private fun rankWithChoiceModel(
        features: BrowFaceMeasurements,
        templates: List<BrowRecommendation>,
        model: TrainedChoiceModel
    ): List<BrowRecommendation> {
        val logits = FloatArray(templates.size) { index ->
            BrowChoiceFeatureBuilder.utility(
                candidateFeatures(features, templates[index].params),
                model.means,
                model.scales,
                model.coefficients
            )
        }
        val probabilities = BrowChoiceFeatureBuilder.probabilities(logits, model.temperature)
        return templates.mapIndexed { index, candidate ->
            candidate.copy(
                score = (probabilities[index] * 100f).roundToInt().coerceIn(0, 100),
                reason = "经独立校准的条件选择模型估计了顾客在当前三款候选中选择此方案的概率，仍需顾客和纹绣师确认。",
                probability = probabilities[index],
                probabilityType = "conditional_choice",
                choiceLogit = logits[index]
            )
        }.sortedByDescending { it.probability ?: 0f }
    }

    private fun loadChoiceModel(context: Context): TrainedChoiceModel? {
        if (choiceLoadAttempted) return trainedChoiceModel
        choiceLoadAttempted = true
        return try {
            val json = context.assets.open("brow_choice_ranker.json")
                .bufferedReader(Charsets.UTF_8).use { it.readText() }
            val root = org.json.JSONObject(json)
            if (root.optInt("model_version", -1) != 1 ||
                root.optString("model_type") != "conditional_choice" ||
                !root.optBoolean("release_approved", false) ||
                root.optString("calibration_status") != "temperature_scaling") return null
            val external = root.optJSONObject("external_validation") ?: return null
            val gates = external.optJSONObject("release_gates") ?: return null
            val requiredGates = listOf(
                "external_cases_at_least_100",
                "external_sessions_at_least_100",
                "top1_accuracy_at_least_0_40",
                "top1_gain_ci_lower_above_zero",
                "multiclass_brier_better_than_uniform",
                "ece_at_most_0_10",
                "nll_better_than_uniform"
            )
            if (requiredGates.any { !gates.optBoolean(it, false) } ||
                !external.optBoolean("passed_release_gates", false) ||
                external.optInt("n_cases", 0) < 100 ||
                external.optInt("n_sessions", 0) < 100 ||
                external.optDouble("top1_accuracy", 0.0) < 0.40 ||
                external.optDouble("multiclass_brier_score", Double.POSITIVE_INFINITY) >=
                    external.optDouble("uniform_choice_brier_baseline", 2.0 / 3.0) ||
                external.optDouble("expected_calibration_error_10_bins", Double.POSITIVE_INFINITY) > 0.10 ||
                external.optDouble("mean_negative_log_likelihood", Double.POSITIVE_INFINITY) >= kotlin.math.ln(3.0)) {
                return null
            }
            val bootstrap = external.optJSONObject("customer_cluster_bootstrap") ?: return null
            if (bootstrap.optDouble("top1_gain_ci95_low", Double.NEGATIVE_INFINITY) <= 0.0) return null

            val expectedFeatureOrder = BrowChoiceFeatureBuilder.featureOrder
            val featureOrder = root.optJSONArray("feature_order") ?: return null
            if (featureOrder.length() != expectedFeatureOrder.size) return null
            for (index in expectedFeatureOrder.indices) {
                if (featureOrder.optString(index) != expectedFeatureOrder[index]) return null
            }
            val meansJson = root.getJSONArray("means")
            val scalesJson = root.getJSONArray("scales")
            val coefficientsJson = root.getJSONArray("coefficients")
            if (meansJson.length() != expectedFeatureOrder.size ||
                scalesJson.length() != expectedFeatureOrder.size ||
                coefficientsJson.length() != expectedFeatureOrder.size) return null
            val means = FloatArray(expectedFeatureOrder.size) { meansJson.getDouble(it).toFloat() }
            val scales = FloatArray(expectedFeatureOrder.size) {
                scalesJson.getDouble(it).toFloat().let { scale -> if (abs(scale) < 1e-8f) 1f else scale }
            }
            val coefficients = FloatArray(expectedFeatureOrder.size) { coefficientsJson.getDouble(it).toFloat() }
            val temperature = root.getDouble("temperature").toFloat()
            if (!temperature.isFinite() || temperature <= 0f ||
                !means.all { it.isFinite() } ||
                !scales.all { it.isFinite() && it > 0f } ||
                !coefficients.all { it.isFinite() }) return null
            trainedChoiceModel = TrainedChoiceModel(means, scales, coefficients, temperature)
            trainedChoiceModel
        } catch (_: Exception) {
            null
        }
    }

    private fun loadModel(context: Context): TrainedModel? {
        if (loadAttempted) return trainedModel
        loadAttempted = true
        return try {
            val json = context.assets.open("brow_ranker.json").bufferedReader(Charsets.UTF_8).use { it.readText() }
            val root = org.json.JSONObject(json)
            // Only a model explicitly human-reviewed after independent calibration and
            // external validation may provide percentages in the live app.
            if (root.optInt("model_version", -1) != 3) return null
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
                !gates.optBoolean("auc_ci_lower_above_chance", false) ||
                !gates.optBoolean("auc_gain_over_rule_lower_ci_above_zero", false) ||
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
                "style_defined",
                "native_brow_area_fraction",
                "native_brow_side_imbalance",
                "native_brow_mask_available",
                "native_shape_mode"
            )
            val featureOrder = root.optJSONArray("feature_order") ?: return null
            if (featureOrder.length() != expectedFeatures.size) return null
            for (i in expectedFeatures.indices) {
                if (featureOrder.optString(i) != expectedFeatures[i]) return null
            }
            val meansJson = root.getJSONArray("means")
            val scalesJson = root.getJSONArray("scales")
            val coefficientsJson = root.getJSONArray("coefficients")
            if (meansJson.length() != 15 || scalesJson.length() != 15 || coefficientsJson.length() != 15) return null
            trainedModel = TrainedModel(
                FloatArray(15) { meansJson.getDouble(it).toFloat() },
                FloatArray(15) { scalesJson.getDouble(it).toFloat().let { scale -> if (abs(scale) < 1e-8f) 1f else scale } },
                FloatArray(15) { coefficientsJson.getDouble(it).toFloat() },
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

internal class BrowDesignView(context: Context) : View(context) {
    private data class MaskStroke(val before: Map<Int, Int>, val after: Map<Int, Int>)

    private val maskUndoStack = mutableListOf<MaskStroke>()
    private val maskRedoStack = mutableListOf<MaskStroke>()
    private var activeStrokeBefore: LinkedHashMap<Int, Int>? = null

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
        if (segmentationResult?.overlay !== result?.overlay) {
            maskUndoStack.clear()
            maskRedoStack.clear()
            activeStrokeBefore = null
        }
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

    fun canUndoMaskStroke(): Boolean = maskUndoStack.isNotEmpty()
    fun canRedoMaskStroke(): Boolean = maskRedoStack.isNotEmpty()

    fun undoLastMaskStroke(): Boolean {
        if (maskUndoStack.isEmpty()) return false
        val action = maskUndoStack.removeAt(maskUndoStack.lastIndex)
        applyPixelState(action.before)
        maskRedoStack.add(action)
        refreshContoursAfterMaskEdit()
        return true
    }

    fun redoLastMaskStroke(): Boolean {
        if (maskRedoStack.isEmpty()) return false
        val action = maskRedoStack.removeAt(maskRedoStack.lastIndex)
        applyPixelState(action.after)
        maskUndoStack.add(action)
        refreshContoursAfterMaskEdit()
        return true
    }

    private fun applyPixelState(state: Map<Int, Int>) {
        val overlay = segmentationResult?.overlay ?: return
        if (overlay.isRecycled || state.isEmpty()) return
        val pixels = IntArray(overlay.width * overlay.height)
        overlay.getPixels(pixels, 0, overlay.width, 0, 0, overlay.width, overlay.height)
        state.forEach { (index, color) ->
            if (index in pixels.indices) pixels[index] = color
        }
        overlay.setPixels(pixels, 0, overlay.width, 0, 0, overlay.width, overlay.height)
    }

    private fun refreshContoursAfterMaskEdit() {
        val result = segmentationResult
        nativeContours = if (result == null || result.overlay.isRecycled) emptyList()
            else BrowRenderer.extractNativeContours(result)
        if (useNativeShape && nativeContours.isEmpty()) useNativeShape = false
        invalidate()
    }

    private fun rememberPixelsBeforeDab(overlay: Bitmap, centerX: Float, centerY: Float, radius: Float) {
        val before = activeStrokeBefore ?: return
        val x0 = kotlin.math.floor(centerX - radius - 1f).toInt().coerceIn(0, overlay.width - 1)
        val y0 = kotlin.math.floor(centerY - radius - 1f).toInt().coerceIn(0, overlay.height - 1)
        val x1 = kotlin.math.ceil(centerX + radius + 1f).toInt().coerceIn(0, overlay.width - 1)
        val y1 = kotlin.math.ceil(centerY + radius + 1f).toInt().coerceIn(0, overlay.height - 1)
        for (y in y0..y1) {
            val row = y * overlay.width
            for (x in x0..x1) {
                val index = row + x
                if (!before.containsKey(index)) before[index] = overlay.getPixel(x, y)
            }
        }
    }

    private fun finishMaskStroke() {
        val before = activeStrokeBefore ?: return
        activeStrokeBefore = null
        val result = segmentationResult ?: return
        if (before.isEmpty() || result.overlay.isRecycled) return
        val after = LinkedHashMap<Int, Int>(before.size)
        before.keys.forEach { index ->
            val x = index % result.overlay.width
            val y = index / result.overlay.width
            after[index] = result.overlay.getPixel(x, y)
        }
        maskUndoStack.add(MaskStroke(before.toMap(), after.toMap()))
        if (maskUndoStack.size > 30) maskUndoStack.removeAt(0)
        maskRedoStack.clear()
        refreshContoursAfterMaskEdit()
    }

    override fun onTouchEvent(event: android.view.MotionEvent): Boolean {
        if (!editSegmentationMask || !showSegmentation || event.actionMasked !in listOf(
                android.view.MotionEvent.ACTION_DOWN,
                android.view.MotionEvent.ACTION_MOVE,
                android.view.MotionEvent.ACTION_UP,
                android.view.MotionEvent.ACTION_CANCEL
            )) return super.onTouchEvent(event)
        val image = photo ?: return false
        val result = segmentationResult ?: return false
        if (image.isRecycled || result.overlay.isRecycled) return false

        if (event.actionMasked == android.view.MotionEvent.ACTION_DOWN) {
            activeStrokeBefore = LinkedHashMap()
            maskRedoStack.clear()
        }
        if (event.actionMasked == android.view.MotionEvent.ACTION_CANCEL) {
            val before = activeStrokeBefore
            activeStrokeBefore = null
            if (before != null) applyPixelState(before)
            refreshContoursAfterMaskEdit()
            return true
        }

        val scale = max(width.toFloat() / image.width.coerceAtLeast(1), height.toFloat() / image.height.coerceAtLeast(1))
        val imageLeft = (width - image.width * scale) / 2f
        val imageTop = (height - image.height * scale) / 2f
        val imageX = (event.x - imageLeft) / scale
        val imageY = (event.y - imageTop) / scale
        val maskX = imageX - result.offsetX
        val maskY = imageY - result.offsetY
        if (maskX < 0f || maskX >= result.overlay.width.toFloat() ||
            maskY < 0f || maskY >= result.overlay.height.toFloat()) {
            if (event.actionMasked == android.view.MotionEvent.ACTION_UP) finishMaskStroke()
            return true
        }

        val brushRadius = max(4f, (8f * resources.displayMetrics.density) / scale)
        rememberPixelsBeforeDab(result.overlay, maskX, maskY, brushRadius)
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
        if (event.actionMasked == android.view.MotionEvent.ACTION_UP) finishMaskStroke()
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
            // Scale each 2D segment to its measured skin-surface arc length after
            // calibration by the current photo's projected segment and interpupillary distance.
            // Scale both tangent coordinates so sloped segments keep their direction.
            val cumulative = mutableListOf(PointF(ordered.first().x, ordered.first().y))
            for (i in 0 until ordered.lastIndex) {
                val rawFactor = curvatureFactors[segmentRegions[i]] ?: 1f
                val factor = rawFactor.coerceIn(0.75f, 1.6f)
                val dx = ordered[i + 1].x - ordered[i].x
                val dy = ordered[i + 1].y - ordered[i].y
                val previous = cumulative.last()
                cumulative.add(PointF(previous.x + dx * factor, previous.y + dy * factor))
            }
            val originalCenter = PointF(
                (ordered.first().x + ordered.last().x) / 2f,
                (ordered.first().y + ordered.last().y) / 2f
            )
            val correctedCenter = PointF(
                (cumulative.first().x + cumulative.last().x) / 2f,
                (cumulative.first().y + cumulative.last().y) / 2f
            )
            cumulative.map { point ->
                PointF(
                    point.x + originalCenter.x - correctedCenter.x,
                    point.y + originalCenter.y - correctedCenter.y
                )
            }
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
            val originalCenterYs = FloatArray(width) { i ->
                (contour.top[i] + contour.bottom[i]) / 2f
            }
            val browLandmarkIds = if (anatomicalRight) {
                intArrayOf(300, 293, 334, 296, 336)
            } else {
                intArrayOf(70, 63, 105, 66, 107)
            }
            val sortedLandmarkX = browLandmarkIds.map { index -> landmarks.getOrNull(index)?.x }.filterNotNull().sorted()
            val cumulativeX = FloatArray(width)
            val cumulativeY = FloatArray(width)
            cumulativeX[0] = 0f
            cumulativeY[0] = originalCenterYs[0]
            for (i in 1 until width) {
                val sourceImageX = contour.xStart + i - 0.5f
                val segment = if (sortedLandmarkX.size == 5) {
                    when {
                        sourceImageX < sortedLandmarkX[1] -> 0
                        sourceImageX < sortedLandmarkX[2] -> 1
                        sourceImageX < sortedLandmarkX[3] -> 2
                        else -> 3
                    }
                } else {
                    min(3, (((i - 0.5f) / (width - 1).coerceAtLeast(1)) * 4f).toInt())
                }
                val factor = (curvatureFactors[segmentNames[segment]] ?: 1f).coerceIn(0.75f, 1.6f)
                // Apply each region's factor between its matching landmark boundaries,
                // and scale both tangent components along the reviewed brow centreline.
                cumulativeX[i] = cumulativeX[i - 1] + factor
                cumulativeY[i] = cumulativeY[i - 1] + (originalCenterYs[i] - originalCenterYs[i - 1]) * factor
            }
            val cumulativeCenterX = (cumulativeX.first() + cumulativeX.last()) / 2f
            val cumulativeCenterY = (cumulativeY.first() + cumulativeY.last()) / 2f
            val originalCenterY = (originalCenterYs.first() + originalCenterYs.last()) / 2f
            val centerlineSpan = kotlin.math.hypot(
                (cumulativeX.last() - cumulativeX.first()).toDouble(),
                (cumulativeY.last() - cumulativeY.first()).toDouble()
            ).toFloat()
            val displaySpan = max(1f, centerlineSpan * params.length)
            val topPathPoints = ArrayList<PointF>(width)
            val bottomPathPoints = ArrayList<PointF>(width)
            for (i in 0 until width) {
                val t = i.toFloat() / (width - 1).coerceAtLeast(1)
                val x = imageCenterX + (cumulativeX[i] - cumulativeCenterX) * params.length
                val upper = contour.top[i]
                val lower = contour.bottom[i]
                val centerY = cumulativeY[i] - cumulativeCenterY + originalCenterY
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
        val maxCorrection = (curvatureFactors.values.maxOfOrNull { abs(it - 1f) }?.plus(1f) ?: 1f).coerceIn(1f, 1.6f)
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
        val maxCorrection = (curvatureFactors.values.maxOfOrNull { abs(it - 1f) }?.plus(1f) ?: 1f).coerceIn(1f, 1.6f)
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