package com.example.facepattern

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.atan2
import kotlin.math.hypot

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var overlay: FaceOverlay
    private lateinit var status: TextView
    private var landmarker: FaceLandmarker? = null
    private var cameraExecutor: ExecutorService? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private val lastTimestamp = AtomicLong(0L)
    private val cameraPermission = 701

    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            try {
                val bitmap = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                if (bitmap != null) {
                    overlay.setPattern(bitmap)
                    status.text = "花纹已载入，可拖动、缩放和旋转"
                } else Toast.makeText(this, "无法读取这张图片", Toast.LENGTH_SHORT).show()
            } catch (_: Exception) {
                Toast.makeText(this, "图片读取失败，请换一张图片", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startFaceTracking()
        } else {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), cameraPermission)
        }
    }

    private fun buildUi() {
        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
        overlay = FaceOverlay(this)
        val root = FrameLayout(this)
        root.addView(previewView, FrameLayout.LayoutParams(-1, -1))
        root.addView(overlay, FrameLayout.LayoutParams(-1, -1))

        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(14))
            setBackgroundColor(0xCC101014.toInt())
        }
        status = TextView(this).apply {
            text = "正在启动摄像头…"
            textSize = 14f
            setTextColor(Color.WHITE)
            setPadding(0, 0, 0, dp(6))
        }
        panel.addView(status)
        val row = LinearLayout(this).apply { gravity = Gravity.CENTER }
        row.addView(Button(this).apply {
            text = "导入花纹"
            setOnClickListener { imagePicker.launch("image/*") }
        }, LinearLayout.LayoutParams(0, dp(48), 1f))
        row.addView(Button(this).apply {
            text = "重置位置"
            setOnClickListener { overlay.resetTransform() }
        }, LinearLayout.LayoutParams(0, dp(48), 1f))
        panel.addView(row)
        root.addView(panel, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        setContentView(root)
    }

    private fun startFaceTracking() {
        try {
            landmarker = FaceLandmarker.createFromOptions(
                this,
                FaceLandmarker.FaceLandmarkerOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath("face_landmarker.task").build())
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setNumFaces(1)
                    .setOutputFacialTransformationMatrixes(true)
                    .setResultListener { result, _ ->
                        val points = result.faceLandmarks().firstOrNull()?.map { PointF(it.x(), it.y()) } ?: emptyList()
                        runOnUiThread {
                            overlay.updateFace(points)
                            status.text = if (points.isNotEmpty()) "已识别人脸 · 单指移动，双指缩放和旋转" else "请正对前置摄像头，保持光线充足"
                        }
                    }
                    .setErrorListener { error ->
                        runOnUiThread { status.text = "人脸识别暂不可用：${error.message ?: "请重新打开应用"}" }
                    }
                    .build()
            )
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
                            val mpImage = BitmapImageBuilder(rotated).build()
                            val now = System.nanoTime() / 1_000_000L
                            val timestamp = lastTimestamp.updateAndGet { old -> maxOf(now, old + 1L) }
                            landmarker?.detectAsync(mpImage, timestamp)
                            rotated.recycle()
                        } catch (_: Exception) {
                            // A single bad camera frame should not stop the live analyzer.
                        } finally {
                            image.close()
                        }
                    }
                    provider.unbindAll()
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
                    status.text = "摄像头已启动，请将脸放入画面"
                } catch (e: Exception) {
                    status.text = "摄像头启动失败，请检查权限后重试"
                }
            }, ContextCompat.getMainExecutor(this))
        } catch (e: Exception) {
            status.text = "人脸识别初始化失败，请检查网络后重新安装"
        }
    }

    private fun imageToBitmap(image: androidx.camera.core.ImageProxy): Bitmap {
        val plane = image.planes[0]
        val buffer = plane.buffer.duplicate()
        val width = image.width
        val height = image.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val pixels = IntArray(width * height)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val i = y * rowStride + x * pixelStride
                val r = buffer.get(i).toInt() and 0xff
                val g = buffer.get(i + 1).toInt() and 0xff
                val b = buffer.get(i + 2).toInt() and 0xff
                val a = buffer.get(i + 3).toInt() and 0xff
                pixels[y * width + x] = (a shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    override fun onResume() {
        super.onResume()
        if (::previewView.isInitialized && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED && landmarker == null) {
            startFaceTracking()
        }
    }

    override fun onDestroy() {
        cameraProvider?.unbindAll()
        landmarker?.close()
        landmarker = null
        cameraExecutor?.shutdown()
        cameraExecutor = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == cameraPermission && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startFaceTracking()
        } else {
            status.text = "请允许使用摄像头，应用才能识别人脸"
        }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
}

private class FaceOverlay(context: Context) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private var pattern: Bitmap? = null
    private var points: List<PointF> = emptyList()
    private var offsetX = 0f
    private var offsetY = 0f
    private var userScale = 0.58f
    private var userRotation = 0f
    private var dragX = 0f
    private var dragY = 0f
    private var lastAngle = 0f
    private var multiTouch = false
    private var previousDistance = 0f
    private var previousMidX = 0f
    private var previousMidY = 0f
    private var downX = 0f
    private var downY = 0f

    fun setPattern(bitmap: Bitmap) {
        pattern = bitmap
        invalidate()
    }

    fun updateFace(newPoints: List<PointF>) {
        points = newPoints
        invalidate()
    }

    fun resetTransform() {
        offsetX = 0f
        offsetY = 0f
        userScale = 0.58f
        userRotation = 0f
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val image = pattern ?: return
        if (points.size < 455 || width == 0 || height == 0) return
        // Camera preview is mirrored for a natural front-camera selfie view.
        fun screen(index: Int): PointF = PointF((1f - points[index].x) * width, points[index].y * height)
        val left = screen(234)
        val right = screen(454)
        val nose = screen(1)
        val faceWidth = hypot((right.x - left.x).toDouble(), (right.y - left.y).toDouble()).toFloat()
        if (faceWidth < 10f) return
        val stickerWidth = faceWidth * userScale
        val stickerHeight = stickerWidth * image.height.toFloat() / image.width.coerceAtLeast(1)
        val cx = nose.x + offsetX
        val cy = nose.y + offsetY
        canvas.save()
        canvas.rotate(userRotation, cx, cy)
        val dst = RectF(cx - stickerWidth / 2f, cy - stickerHeight / 2f, cx + stickerWidth / 2f, cy + stickerHeight / 2f)
        paint.alpha = 230
        canvas.drawBitmap(image, null, dst, paint)
        canvas.restore()
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                downX = event.x; downY = event.y
                dragX = offsetX; dragY = offsetY
                multiTouch = false
                return true
            }
            MotionEvent.ACTION_POINTER_DOWN -> if (event.pointerCount >= 2) {
                multiTouch = true
                lastAngle = angle(event)
                previousDistance = distance(event)
                previousMidX = (event.getX(0) + event.getX(1)) / 2f
                previousMidY = (event.getY(0) + event.getY(1)) / 2f
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (event.pointerCount >= 2) {
                    val a = angle(event)
                    var d = a - lastAngle
                    while (d > Math.PI) d -= (2 * Math.PI).toFloat()
                    while (d < -Math.PI) d += (2 * Math.PI).toFloat()
                    userRotation += Math.toDegrees(d.toDouble()).toFloat()
                    lastAngle = a
                    val dist = distance(event)
                    if (previousDistance > 0f) userScale = (userScale * dist / previousDistance).coerceIn(0.12f, 1.8f)
                    previousDistance = dist
                    invalidate()
                } else if (!multiTouch) {
                    offsetX = dragX + event.x - downX
                    offsetY = dragY + event.y - downY
                    invalidate()
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                multiTouch = false
                return true
            }
        }
        return true
    }

    private fun distance(e: MotionEvent): Float = hypot((e.getX(1) - e.getX(0)).toDouble(), (e.getY(1) - e.getY(0)).toDouble()).toFloat()
    private fun angle(e: MotionEvent): Float = atan2((e.getY(1) - e.getY(0)).toDouble(), (e.getX(1) - e.getX(0)).toDouble()).toFloat()
}
