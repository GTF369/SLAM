package com.example.facepattern

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session

/**
 * A capability-only probe. It does not start the camera stream, acquire depth frames,
 * or claim that the device has millimetre-accurate brow measurements.
 */
class ArCoreDepthProbeActivity : AppCompatActivity() {
    private lateinit var status: TextView
    private var userRequestedInstall = true
    private var checking = false

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                checkDepthSupport()
            } else {
                status.text = "未获得相机权限，无法检测 ARCore 深度能力。"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 48, 48, 48)
        }
        status = TextView(this).apply {
            textSize = 16f
            text = "此页面只检测设备是否支持 ARCore Depth API，不会扫描或保存顾客面部数据。"
        }
        root.addView(status, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(Button(this).apply {
            text = "重新检测"
            setOnClickListener { checkDepthSupport() }
        })
        root.addView(Button(this).apply {
            text = "返回眉形设计"
            setOnClickListener { finish() }
        })
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            checkDepthSupport()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun checkDepthSupport() {
        if (checking || isFinishing || isDestroyed) return
        checking = true
        status.text = "正在检查 ARCore 服务与设备深度能力……"
        ArCoreApk.getInstance().checkAvailabilityAsync(this) { availability ->
            if (isFinishing || isDestroyed) {
                checking = false
                return@checkAvailabilityAsync
            }
            if (!availability.isSupported) {
                checking = false
                status.text = "当前设备或 ARCore 服务暂不支持 ARCore。可以继续使用二维眉形设计功能。"
                return@checkAvailabilityAsync
            }
            createSessionAndCheckDepth()
        }
    }

    private fun createSessionAndCheckDepth() {
        try {
            when (ArCoreApk.getInstance().requestInstall(this, userRequestedInstall)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                    userRequestedInstall = false
                    checking = false
                    status.text = "请完成 Google Play 服务 for AR 的安装或更新，然后返回本页面重新检测。"
                    return
                }
                ArCoreApk.InstallStatus.INSTALLED -> Unit
            }

            val session = Session(this)
            val depthSupported = try {
                val supported = session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                if (supported) {
                    val config = session.config
                    config.depthMode = Config.DepthMode.AUTOMATIC
                    session.configure(config)
                }
                supported
            } finally {
                session.close()
            }
            status.text = if (depthSupported) {
                "检测通过：当前设备支持 ARCore Depth API。\n\n注意：这只是设备能力检测，不代表眉弓区域已经达到毫米级测量精度。本版本尚未采集深度帧。下一步将加入 Raw Depth、置信度和点云导出，并通过标准头模验证。"
            } else {
                "ARCore 可用，但当前设备不支持 Depth API。二维眉形设计和打印功能仍可继续使用。"
            }
        } catch (error: Exception) {
            status.text = "ARCore 检测未完成：${error.message ?: error.javaClass.simpleName}\n\n可以先继续使用二维设计功能，稍后重新检测。"
        } finally {
            checking = false
        }
    }
}
