package com.example.facepattern

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
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
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.atan2

class MainActivity : AppCompatActivity() {
    private var session: Session? = null
    private lateinit var glView: GLSurfaceView
    private lateinit var renderer: FaceRenderer
    private lateinit var status: TextView
    private var installRequested = false
    private val permissionCode = 410

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) try {
            val bitmap = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
            if (bitmap != null) { renderer.setSticker(bitmap); status.text = "花纹已载入 · 单指拖动，双指缩放和旋转" }
        } catch (_: Exception) { Toast.makeText(this, "图片读取失败", Toast.LENGTH_SHORT).show() }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        glView = GLSurfaceView(this).apply { setEGLContextClientVersion(2); preserveEGLContextOnPause = true; renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY }
        renderer = FaceRenderer(this) { found -> runOnUiThread { status.text = if (found) "已识别人脸 · 单指拖动，双指缩放和旋转" else "请将脸对准前置摄像头" } }
        glView.setRenderer(renderer)
        glView.setOnTouchListener(TouchController(renderer, this))
        val root = FrameLayout(this)
        root.addView(glView, FrameLayout.LayoutParams(-1, -1))
        val panel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(14), dp(10), dp(14), dp(14)); setBackgroundColor(0xCC101014.toInt()) }
        status = TextView(this).apply { text = "正在准备 AR 人脸追踪…"; textSize = 14f; setTextColor(-1); setPadding(0, 0, 0, dp(6)) }
        panel.addView(status)
        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER }
        buttons.addView(Button(this).apply { text = "导入花纹"; setOnClickListener { picker.launch("image/*") } }, LinearLayout.LayoutParams(0, dp(48), 1f))
        buttons.addView(Button(this).apply { text = "重置位置"; setOnClickListener { renderer.resetPattern() } }, LinearLayout.LayoutParams(0, dp(48), 1f))
        panel.addView(buttons)
        root.addView(panel, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        setContentView(root)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) createSession()
        else ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), permissionCode)
    }

    private fun createSession() {
        try {
            when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> { installRequested = true; return }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }
            val s = Session(this)
            s.configure(Config(s).apply { augmentedFaceMode = Config.AugmentedFaceMode.MESH3D })
            session = s
            renderer.setSession(s)
            status.text = "AR 已启动 · 请将脸对准前置摄像头"
        } catch (_: UnavailableException) { status.text = "此设备无法启动 ARCore，请检查设备兼容性" }
        catch (e: Exception) { status.text = "AR 初始化失败：${e.localizedMessage ?: "请检查设备兼容性"}" }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED && session == null) createSession()
        try { session?.resume(); if (::glView.isInitialized) glView.onResume() }
        catch (_: CameraNotAvailableException) { status.text = "摄像头不可用，请关闭其他使用摄像头的应用"; session?.close(); session = null }
        catch (_: Exception) { }
    }
    override fun onPause() { if (::glView.isInitialized) glView.onPause(); session?.pause(); super.onPause() }
    override fun onDestroy() { session?.close(); session = null; super.onDestroy() }
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionCode && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) createSession()
        else status.text = "需要摄像头权限才能进行人脸贴合"
    }
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
}

private class TouchController(private val renderer: FaceRenderer, context: android.content.Context) : View.OnTouchListener {
    private var lastX = 0f
    private var lastY = 0f
    private var lastAngle = 0f
    private var multi = false
    private val scaleDetector = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean { renderer.scalePattern(detector.scaleFactor); return true }
    })
    override fun onTouch(v: View, e: MotionEvent): Boolean {
        scaleDetector.onTouchEvent(e)
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> { lastX = e.x; lastY = e.y; multi = false }
            MotionEvent.ACTION_POINTER_DOWN -> { multi = true; if (e.pointerCount >= 2) lastAngle = angle(e) }
            MotionEvent.ACTION_MOVE -> {
                if (e.pointerCount >= 2) {
                    val a = angle(e)
                    var d = a - lastAngle
                    while (d > Math.PI) d -= (2 * Math.PI).toFloat()
                    while (d < -Math.PI) d += (2 * Math.PI).toFloat()
                    renderer.rotatePattern(d); lastAngle = a
                } else if (!scaleDetector.isInProgress && !multi) renderer.movePattern((e.x-lastX)/v.width*0.42f, -(e.y-lastY)/v.height*0.55f)
                lastX = e.x; lastY = e.y
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> multi = false
        }
        return true
    }
    private fun angle(e: MotionEvent) = atan2((e.getY(1)-e.getY(0)).toDouble(), (e.getX(1)-e.getX(0)).toDouble()).toFloat()
}

private class FaceRenderer(private val context: android.content.Context, private val onFace: (Boolean) -> Unit) : GLSurfaceView.Renderer {
    private var session: Session? = null
    private var cameraTex = 0
    private var cameraProgram = 0
    private var stickerProgram = 0
    private var stickerTex = 0
    private var sticker: Bitmap? = null
    private var uploaded: Bitmap? = null
    private var centerX = 0f
    private var centerY = 0.015f
    private var size = 0.09f
    private var scale = 1f
    private var rotation = 0f
    private val projection = FloatArray(16)
    private val view = FloatArray(16)
    private val model = FloatArray(16)
    private val mv = FloatArray(16)
    private val mvp = FloatArray(16)
    private var faceVisible = false
    fun setSession(s: Session) { session = s; if (cameraTex != 0) s.setCameraTextureNames(intArrayOf(cameraTex)) }
    @Synchronized fun setSticker(b: Bitmap) { sticker = b; size = 0.09f }
    @Synchronized fun movePattern(dx: Float, dy: Float) { centerX = (centerX+dx).coerceIn(-0.16f,0.16f); centerY = (centerY+dy).coerceIn(-0.16f,0.16f) }
    @Synchronized fun scalePattern(f: Float) { scale = (scale*f).coerceIn(0.2f,4f) }
    @Synchronized fun rotatePattern(d: Float) { rotation += d }
    @Synchronized fun resetPattern() { centerX=0f; centerY=0.015f; scale=1f; rotation=0f }

    override fun onSurfaceCreated(gl: GL10?, cfg: EGLConfig?) {
        cameraTex = externalTexture()
        cameraProgram = program(CAM_V, CAM_F)
        stickerProgram = program(STICK_V, STICK_F)
        session?.setCameraTextureNames(intArrayOf(cameraTex))
    }
    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        GLES20.glViewport(0,0,w,h)
        session?.setDisplayGeometry(android.view.Surface.ROTATION_0,w,h)
    }
    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClearColor(0f,0f,0f,1f); GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        try {
            val frame = s.update()
            if (frame.camera.trackingState != TrackingState.TRACKING) return
            frame.camera.getProjectionMatrix(projection,0,0.1f,100f)
            frame.camera.getViewMatrix(view,0)
            drawCamera()
            GLES20.glEnable(GLES20.GL_DEPTH_TEST)
            val face = s.getAllTrackables(com.google.ar.core.AugmentedFace::class.java).firstOrNull { it.trackingState == TrackingState.TRACKING }
            if (face != null) {
                if (!faceVisible) { faceVisible=true; onFace(true) }
                drawFace(face)
            } else if (faceVisible) { faceVisible=false; onFace(false) }
        } catch (_: Exception) { }
    }
    private fun drawCamera() {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST); GLES20.glUseProgram(cameraProgram)
        val pos = floats(floatArrayOf(-1f,-1f, 1f,-1f, -1f,1f, 1f,1f))
        val uv = floats(floatArrayOf(0f,1f, 1f,1f, 0f,0f, 1f,0f))
        val p = GLES20.glGetAttribLocation(cameraProgram,"aPosition")
        val t = GLES20.glGetAttribLocation(cameraProgram,"aTexCoord")
        GLES20.glEnableVertexAttribArray(p); GLES20.glVertexAttribPointer(p,2,GLES20.GL_FLOAT,false,0,pos)
        GLES20.glEnableVertexAttribArray(t); GLES20.glVertexAttribPointer(t,2,GLES20.GL_FLOAT,false,0,uv)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0); GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,cameraTex)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(cameraProgram,"uTexture"),0)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP,0,4)
        GLES20.glDisableVertexAttribArray(p); GLES20.glDisableVertexAttribArray(t)
    }
    @Synchronized private fun drawFace(face: com.google.ar.core.AugmentedFace) {
        val bitmap = sticker ?: return
        if (stickerTex == 0) { stickerTex = texture(bitmap); uploaded = bitmap }
        else if (uploaded !== bitmap) { GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,stickerTex); android.opengl.GLUtils.texImage2D(GLES20.GL_TEXTURE_2D,0,bitmap,0); uploaded=bitmap }
        val vb = face.meshVertices.duplicate(); vb.rewind()
        val vertexData = FloatArray(vb.remaining()); vb.get(vertexData)
        val ib = face.meshTriangleIndices.duplicate(); ib.rewind()
        val indexData = ShortArray(ib.remaining()); ib.get(indexData)
        val vbuf = floats(vertexData)
        val ibuf = shorts(indexData)
        face.centerPose.toMatrix(model,0)
        Matrix.multiplyMM(mv,0,view,0,model,0); Matrix.multiplyMM(mvp,0,projection,0,mv,0)
        GLES20.glUseProgram(stickerProgram)
        val p = GLES20.glGetAttribLocation(stickerProgram,"aPosition")
        GLES20.glEnableVertexAttribArray(p); GLES20.glVertexAttribPointer(p,3,GLES20.GL_FLOAT,false,0,vbuf)
        GLES20.glUniformMatrix4fv(GLES20.glGetUniformLocation(stickerProgram,"uMvp"),1,false,mvp,0)
        GLES20.glUniform2f(GLES20.glGetUniformLocation(stickerProgram,"uCenter"),centerX,centerY)
        GLES20.glUniform2f(GLES20.glGetUniformLocation(stickerProgram,"uSize"),size*scale,size*scale*bitmap.height.toFloat()/bitmap.width.coerceAtLeast(1))
        GLES20.glUniform1f(GLES20.glGetUniformLocation(stickerProgram,"uAngle"),rotation)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1); GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,stickerTex)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(stickerProgram,"uSticker"),1)
        GLES20.glEnable(GLES20.GL_BLEND); GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA,GLES20.GL_ONE_MINUS_SRC_ALPHA)
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,indexData.size,GLES20.GL_UNSIGNED_SHORT,ibuf)
        GLES20.glDisable(GLES20.GL_BLEND); GLES20.glDisableVertexAttribArray(p)
    }
    private fun externalTexture(): Int {
        val a=IntArray(1); GLES20.glGenTextures(1,a,0); GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,a[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR); GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE); GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE)
        return a[0]
    }
    private fun texture(b: Bitmap): Int {
        val a=IntArray(1); GLES20.glGenTextures(1,a,0); GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,a[0])
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR); GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE); GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE)
        android.opengl.GLUtils.texImage2D(GLES20.GL_TEXTURE_2D,0,b,0); return a[0]
    }
    companion object {
        private fun floats(a: FloatArray): FloatBuffer = ByteBuffer.allocateDirect(a.size*4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply { put(a); position(0) }
        private fun shorts(a: ShortArray): ShortBuffer = ByteBuffer.allocateDirect(a.size*2).order(ByteOrder.nativeOrder()).asShortBuffer().apply { put(a); position(0) }
        private fun sh(type: Int,src: String): Int { val id=GLES20.glCreateShader(type); GLES20.glShaderSource(id,src); GLES20.glCompileShader(id); val ok=IntArray(1); GLES20.glGetShaderiv(id,GLES20.GL_COMPILE_STATUS,ok,0); if(ok[0]==0) throw IllegalStateException(GLES20.glGetShaderInfoLog(id)); return id }
        private fun program(v: String,f: String): Int { val id=GLES20.glCreateProgram(); GLES20.glAttachShader(id,sh(GLES20.GL_VERTEX_SHADER,v)); GLES20.glAttachShader(id,sh(GLES20.GL_FRAGMENT_SHADER,f)); GLES20.glLinkProgram(id); val ok=IntArray(1); GLES20.glGetProgramiv(id,GLES20.GL_LINK_STATUS,ok,0); if(ok[0]==0) throw IllegalStateException(GLES20.glGetProgramInfoLog(id)); return id }
        private const val CAM_V = "attribute vec2 aPosition; attribute vec2 aTexCoord; varying vec2 vTex; void main(){gl_Position=vec4(aPosition,0.0,1.0);vTex=aTexCoord;}"
        private const val CAM_F = "#extension GL_OES_EGL_image_external : require\nprecision mediump float; varying vec2 vTex; uniform samplerExternalOES uTexture; void main(){gl_FragColor=texture2D(uTexture,vTex);}"
        private const val STICK_V = "attribute vec3 aPosition; uniform mat4 uMvp; varying vec2 vLocal; void main(){vLocal=aPosition.xy;gl_Position=uMvp*vec4(aPosition,1.0);}"
        private const val STICK_F = "precision mediump float; varying vec2 vLocal; uniform vec2 uCenter; uniform vec2 uSize; uniform float uAngle; uniform sampler2D uSticker; void main(){vec2 d=vLocal-uCenter;float c=cos(uAngle);float s=sin(uAngle);vec2 q=vec2(c*d.x+s*d.y,-s*d.x+c*d.y)/max(uSize,vec2(0.0001))+vec2(0.5);if(q.x<0.0||q.x>1.0||q.y<0.0||q.y>1.0)discard;vec4 color=texture2D(uSticker,vec2(q.x,1.0-q.y));if(color.a<0.02)discard;gl_FragColor=color;}"
    }
}
