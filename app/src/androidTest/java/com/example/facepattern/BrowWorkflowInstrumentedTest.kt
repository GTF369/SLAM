package com.example.facepattern

import android.content.ContentValues
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.view.MotionEvent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.UiScrollable
import androidx.test.uiautomator.UiSelector
import androidx.test.uiautomator.Until
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.TimeUnit
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.json.JSONArray
import org.json.JSONObject
import java.security.GeneralSecurityException

@RunWith(AndroidJUnit4::class)
class BrowWorkflowInstrumentedTest {
    private lateinit var device: UiDevice

    private fun launch(syntheticDesign: Boolean) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        device = UiDevice.getInstance(instrumentation)
        val intent = Intent(instrumentation.targetContext, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            putExtra("com.example.facepattern.INSTRUMENTATION_TEST_MODE", true)
            putExtra("com.example.facepattern.SYNTHETIC_DESIGN_MODE", syntheticDesign)
        }
        instrumentation.startActivitySync(intent)
        assertTrue(
            "Main screen did not start",
            device.wait(Until.hasObject(By.text("眉形设计助手")), TimeUnit.SECONDS.toMillis(10))
        )
    }

    private fun scrollTo(text: String) {
        val scrollable = UiScrollable(UiSelector().scrollable(true))
        scrollable.setAsVerticalList()
        assertTrue("Cannot find control: $text", scrollable.scrollTextIntoView(text))
        assertNotNull(device.findObject(By.text(text)))
    }

    private fun launchPublicPhoto(uri: Uri) {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        device = UiDevice.getInstance(instrumentation)
        val intent = Intent(instrumentation.targetContext, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            putExtra("com.example.facepattern.INSTRUMENTATION_TEST_MODE", true)
            putExtra("com.example.facepattern.PUBLIC_PHOTO_URI", uri.toString())
            putExtra("com.example.facepattern.TEST_RUN_SEGMENTATION", true)
        }
        instrumentation.startActivitySync(intent)
        assertTrue(
            "Main screen did not start for the public portrait",
            device.wait(Until.hasObject(By.text("眉形设计助手")), TimeUnit.SECONDS.toMillis(10))
        )
    }

    private fun typeIntoOnlyDialogField(value: String) {
        val field = device.findObject(By.clazz("android.widget.EditText"))
        assertNotNull("Expected a text field in the dialog", field)
        field.text = value
    }

    private fun clickText(text: String) {
        val element: UiObject2? = device.findObject(By.text(text))
        assertNotNull("Cannot find visible text: $text", element)
        element!!.click()
    }

    @Test
    fun measuredSurfaceCompensationUsesPhotoScaleAndRejectsInvalidFactors() {
        val projectedMm = BrowSurfaceCompensation.projectedSegmentMillimetres(
            projectedSegmentPixels = 100f,
            pupilDistancePixels = 630f,
            measuredInterpupillaryDistanceMm = 63f
        )
        assertNotNull("Calibrated projected segment should be valid", projectedMm)
        assertEquals(10f, projectedMm!!, 1e-5f)

        val factor = BrowSurfaceCompensation.factorFromMeasurements(
            measuredSurfaceArcMm = 12f,
            projectedSegmentPixels = 100f,
            pupilDistancePixels = 630f,
            measuredInterpupillaryDistanceMm = 63f
        )
        assertNotNull("Measured surface arc should produce a valid factor", factor)
        assertEquals(1.2f, factor!!, 1e-5f)

        assertEquals(0.75f, BrowSurfaceCompensation.factorFromMeasurements(
            7.5f, 100f, 630f, 63f
        )!!, 1e-5f)
        assertEquals(1.6f, BrowSurfaceCompensation.factorFromMeasurements(
            16f, 100f, 630f, 63f
        )!!, 1e-5f)
        assertEquals(null, BrowSurfaceCompensation.factorFromMeasurements(
            16.1f, 100f, 630f, 63f
        ))
        assertEquals(null, BrowSurfaceCompensation.factorFromMeasurements(
            0f, 100f, 630f, 63f
        ))
        assertEquals(null, BrowSurfaceCompensation.factorFromMeasurements(
            12f, 1f, 630f, 63f
        ))
        assertEquals(null, BrowSurfaceCompensation.factorFromMeasurements(
            12f, 100f, 630f, Float.NaN
        ))

        val regions = setOf("左眉头", "左眉身", "左眉峰", "左眉尾")
        assertTrue(BrowSurfaceCompensation.validateCompleteFactors(
            mapOf("左眉头" to 1.1f, "左眉身" to 1.2f, "左眉峰" to 1.0f, "左眉尾" to 1.3f),
            regions
        ))
        assertFalse(BrowSurfaceCompensation.validateCompleteFactors(
            mapOf("左眉头" to 1.1f, "左眉身" to 1.2f, "左眉峰" to 1.0f),
            regions
        ))
        assertFalse(BrowSurfaceCompensation.validateCompleteFactors(
            mapOf("左眉头" to 1.1f, "左眉身" to 1.61f, "左眉峰" to 1.0f, "左眉尾" to 1.3f),
            regions
        ))
    }

    @Test
    fun browMaskAreaGateRejectsEmptyAndOversizedContours() {
        assertTrue("A normal 10% brow ROI mask should be accepted", isPlausibleBrowMask(100, 1000))
        assertTrue("The lower 2% boundary should be accepted", isPlausibleBrowMask(20, 1000))
        assertTrue("The upper 28% boundary should be accepted", isPlausibleBrowMask(280, 1000))
        assertFalse("An empty contour must be rejected", isPlausibleBrowMask(0, 1000))
        assertFalse("A 0.98% speck must not be treated as an eyebrow", isPlausibleBrowMask(64, 6534))
        assertFalse("A 33% dark patch must not be treated as an eyebrow", isPlausibleBrowMask(333, 1000))
        assertTrue("A few border pixels should be tolerated", isPlausibleBrowMask(200, 1000, 2))
        assertFalse("A mask leaking into the ROI border should be rejected", isPlausibleBrowMask(200, 1000, 20))
        assertFalse("Invalid region size must be rejected", isPlausibleBrowMask(0, 0))
        val borderMask = BooleanArray(100)
        for (y in 0 until 10) for (x in 0 until 10) {
            if (x < 2 || y < 2 || x >= 8 || y >= 8) borderMask[y * 10 + x] = true
        }
        assertEquals("Border count must match the two-pixel ROI frame", 64, countBrowMaskBorderPixels(borderMask, 10, 10))
    }

    @Test
    fun launchWithoutCameraShowsPhotoImportAndLocalCaseLibrary() {
        launch(syntheticDesign = false)
        assertNotNull(device.findObject(By.text("拍照设计")))
        assertNotNull(device.findObject(By.text("导入照片")))
        assertNotNull(device.findObject(By.text("本机案例库")))
        assertNotNull(device.findObject(By.text("导出加密案例")))
        assertNotNull(device.findObject(By.text("导入加密案例")))
        assertTrue(device.hasObject(By.text("仪器测试模式。相机未启动。")))
    }

    @Test
    fun publicPortraitRunsThroughAndroidImportLandmarksAndBrowSegmentation() {
        // This CI variant is intentionally x86_64, while the production app is ARM64-only.
        // The MediaPipe JNI runtime is not bundled for x86_64. The equivalent public-image
        // inference pipeline is exercised by ml/public_image_smoke_test.py; run this
        // Android import-to-landmarks test only on a supported ARM64 Android runtime.
        assumeTrue(
            "MediaPipe public-photo import requires an ARM64 Android runtime; see Python model smoke test.",
            Build.SUPPORTED_ABIS.any { it == "arm64-v8a" }
        )
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val publicImageBytes = instrumentation.context.assets.open("public_portrait.jpg").use { it.readBytes() }
        assertTrue("CI public portrait asset is empty", publicImageBytes.isNotEmpty())

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "brow-ci-public.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/BrowDesignTest")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw AssertionError("Unable to create temporary MediaStore image for Android import test")
        try {
            context.contentResolver.openOutputStream(uri, "w")?.use { output ->
                output.write(publicImageBytes)
                output.flush()
            } ?: throw AssertionError("Unable to write test portrait to MediaStore")
            val ready = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
            context.contentResolver.update(uri, ready, null, null)

            launchPublicPhoto(uri)
            // Some publicly sourced portraits trigger the ordinary quality warning. Accepting
            // it is part of the production flow; the test never bypasses the image importer.
            if (device.wait(Until.hasObject(By.text("仍然使用")), TimeUnit.SECONDS.toMillis(40))) {
                clickText("仍然使用")
            }

            assertTrue(
                "Imported public photo never produced a brow mask status in the Android app",
                device.wait(Until.hasObject(By.textContains("识别方式：")), TimeUnit.SECONDS.toMillis(75))
            )
            val methodLabel = device.findObject(By.textContains("识别方式："))
            assertNotNull("Expected the app to expose its segmentation method", methodLabel)
            assertTrue(
                "Segmentation status should include a quality report",
                methodLabel!!.text.contains("掩膜质量提示")
            )
        } finally {
            try { context.contentResolver.delete(uri, null, null) } catch (_: Exception) {}
            publicImageBytes.fill(0)
        }
    }

    @Test
    fun portableCaseBackupRoundTripsAndRejectsWrongPasswordOrTampering() {
        val manifest = JSONObject()
            .put("schema_version", 1)
            .put("record_id", "source-case")
            .put("display_name", "测试案例")
            .put("anonymous_case_id", "TESTCASE01")
            .put("saved_at", 1000L)
            .put("photo_file", "photo_test.jpg.enc")
            .put("current_segmentation", JSONObject.NULL)
            .put("plans", JSONArray())
        val originalImage = "synthetic-image-bytes-for-codec-test".toByteArray(Charsets.UTF_8)
        val archiveBytes = PortableCaseBackup.pack(manifest, mapOf("photo_test.jpg.enc" to originalImage))
        val password = "unique-test-passphrase-12".toCharArray()
        val encrypted = PortableCaseBackup.encrypt(archiveBytes, password)
        val decrypted = PortableCaseBackup.decrypt(encrypted, password)
        val restored = PortableCaseBackup.unpack(decrypted)
        assertEquals("测试案例", restored.manifest.getString("display_name"))
        assertTrue(restored.files["photo_test.jpg.enc"]!!.contentEquals(originalImage))

        var wrongPasswordRejected = false
        try {
            PortableCaseBackup.decrypt(encrypted, "different-wrong-password".toCharArray())
        } catch (_: GeneralSecurityException) {
            wrongPasswordRejected = true
        }
        assertTrue("Wrong backup passwords must not decrypt the package", wrongPasswordRejected)

        val tampered = encrypted.copyOf()
        tampered[tampered.lastIndex] = (tampered[tampered.lastIndex].toInt() xor 1).toByte()
        var tamperingRejected = false
        try {
            PortableCaseBackup.decrypt(tampered, password)
        } catch (_: GeneralSecurityException) {
            tamperingRejected = true
        }
        assertTrue("Authenticated encryption must reject modified ciphertext", tamperingRejected)

        val evilManifest = JSONObject(manifest.toString()).put("photo_file", "../outside.jpg.enc")
        var unsafeReferenceRejected = false
        try {
            PortableCaseBackup.pack(evilManifest, mapOf("../outside.jpg.enc" to originalImage))
        } catch (_: IllegalArgumentException) {
            unsafeReferenceRejected = true
        }
        assertTrue("Case manifests must not reference paths outside the case", unsafeReferenceRejected)

        val reservedManifestReference = JSONObject(manifest.toString()).put("photo_file", "case.json.enc")
        var reservedReferenceRejected = false
        try {
            PortableCaseBackup.pack(reservedManifestReference, mapOf("case.json.enc" to originalImage))
        } catch (_: IllegalArgumentException) {
            reservedReferenceRejected = true
        }
        assertTrue("Case resources must not collide with the encrypted manifest", reservedReferenceRejected)

        val hostileZipBuffer = ByteArrayOutputStream()
        ZipOutputStream(hostileZipBuffer).use { zip ->
            zip.putNextEntry(ZipEntry("manifest.json"))
            zip.write(manifest.toString().toByteArray(Charsets.UTF_8))
            zip.closeEntry()
            zip.putNextEntry(ZipEntry("data/../outside.jpg.enc"))
            zip.write(originalImage)
            zip.closeEntry()
        }
        var unsafeZipRejected = false
        try {
            PortableCaseBackup.unpack(hostileZipBuffer.toByteArray())
        } catch (_: IllegalArgumentException) {
            unsafeZipRejected = true
        }
        assertTrue("ZIP path traversal entries must be rejected", unsafeZipRejected)

        password.fill('\u0000')
        encrypted.fill(0)
        decrypted.fill(0)
        archiveBytes.fill(0)
    }

    @Test
    fun manualMaskStrokeSupportsUndoAndRedo() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val photo = Bitmap.createBitmap(200, 200, Bitmap.Config.ARGB_8888)
        val overlay = Bitmap.createBitmap(120, 20, Bitmap.Config.ARGB_8888)
        val regions = listOf(
            intArrayOf(40, 70, 80, 90),
            intArrayOf(120, 70, 160, 90),
        )
        val mask = BrowSegmentationResult(
            overlay = overlay,
            offsetX = 40,
            offsetY = 70,
            method = "instrumentation_test",
            maskQualityScore = 0.5f,
            selectedPixelCount = 0,
            regions = regions,
        )
        val values = IntArray(3)
        val flags = BooleanArray(4)
        try {
            instrumentation.runOnMainSync {
                val view = BrowDesignView(context)
                view.layout(0, 0, 200, 200)
                view.setPhoto(photo)
                view.setSegmentationResult(mask, visible = true)
                view.setSegmentationEditMode(enabled = true, includePixels = true)

                values[0] = Color.alpha(overlay.getPixel(20, 10))
                val eventTime = SystemClock.uptimeMillis()
                val down = MotionEvent.obtain(
                    eventTime, eventTime, MotionEvent.ACTION_DOWN, 60f, 80f, 0
                )
                val up = MotionEvent.obtain(
                    eventTime, eventTime + 50, MotionEvent.ACTION_UP, 60f, 80f, 0
                )
                try {
                    flags[0] = view.onTouchEvent(down)
                    flags[1] = view.onTouchEvent(up)
                } finally {
                    down.recycle()
                    up.recycle()
                }

                values[1] = Color.alpha(overlay.getPixel(20, 10))
                flags[2] = view.canUndoMaskStroke()
                flags[3] = view.undoLastMaskStroke()
                values[2] = Color.alpha(overlay.getPixel(20, 10))
                val canRedo = view.canRedoMaskStroke()
                val redo = view.redoLastMaskStroke()
                val afterRedo = Color.alpha(overlay.getPixel(20, 10))
                view.setPhoto(null)
                overlay.recycle()
                photo.recycle()

                assertEquals("Undo should be available after the stroke", true, flags[2])
                assertTrue("The stroke should be undoable", flags[3])
                assertEquals("Undo must restore transparency", 0, values[2])
                assertTrue("Undo should enable redo", canRedo)
                assertTrue("Redo should succeed", redo)
                assertTrue("Redo must restore the stroke pixels", afterRedo > 0)
            }
        } finally {
            if (!photo.isRecycled) photo.recycle()
            if (!overlay.isRecycled) overlay.recycle()
        }
        assertEquals("Original overlay should start transparent", 0, values[0])
        assertTrue("ACTION_DOWN must be handled", flags[0])
        assertTrue("ACTION_UP must be handled", flags[1])
        assertTrue("A manual stroke should add visible pixels", values[1] > 0)
    }

    @Test
    fun photoCaptureAndImportRequireExplicitConsent() {
        launch(syntheticDesign = false)
        clickText("拍照设计")
        assertTrue(
            "Capture consent dialog did not open",
            device.wait(Until.hasObject(By.text("顾客拍摄与数据使用确认")), TimeUnit.SECONDS.toMillis(3))
        )
        val captureButton = device.findObject(By.text("同意并拍照"))
        assertNotNull("Capture confirmation button missing", captureButton)
        assertTrue("Capture must remain disabled until consent is checked", !captureButton!!.isEnabled)
        clickText("我已向顾客说明用途并取得本次拍摄和眉形设计同意")
        assertTrue("Capture should be enabled after consent", device.findObject(By.text("同意并拍照")).isEnabled)
        clickText("取消")

        clickText("导入照片")
        assertTrue(
            "Photo-import consent dialog did not open",
            device.wait(Until.hasObject(By.text("导入顾客照片")), TimeUnit.SECONDS.toMillis(3))
        )
        val importButton = device.findObject(By.text("同意并选择照片"))
        assertNotNull("Photo import button missing", importButton)
        assertTrue("Photo import must remain disabled until consent is checked", !importButton!!.isEnabled)
        clickText("我已确认顾客同意将所选照片用于本次眉形设计")
        assertTrue("Photo import should be enabled after consent", device.findObject(By.text("同意并选择照片")).isEnabled)
        clickText("取消")
    }

    @Test
    fun previewAndPlanBookExportsRequireConsent() {
        launch(syntheticDesign = true)
        assertTrue(
            "Synthetic test design did not load",
            device.wait(Until.hasObject(By.textContains("合成测试照片已加载")), TimeUnit.SECONDS.toMillis(5))
        )

        scrollTo("保存效果图")
        clickText("保存效果图")
        assertTrue(
            "Preview export dialog did not open",
            device.wait(Until.hasObject(By.text("保存眉形效果图")), TimeUnit.SECONDS.toMillis(3))
        )
        val previewButton = device.findObject(By.text("保存效果图"))
        assertNotNull("Preview export action missing", previewButton)
        assertTrue("Preview export must be disabled before consent", !previewButton!!.isEnabled)
        clickText("我已取得顾客同意，将包含人脸和眉形的效果图保存到相册")
        assertTrue("Preview export should enable after consent", device.findObject(By.text("保存效果图")).isEnabled)
        clickText("取消")

        scrollTo("保存方案")
        clickText("保存方案")
        typeIntoOnlyDialogField("CI_EXPORT_PLAN")
        clickText("保存")

        scrollTo("导出方案册 PDF")
        clickText("导出方案册 PDF")
        assertTrue(
            "Plan-book export dialog did not open",
            device.wait(Until.hasObject(By.text("导出眉形方案册")), TimeUnit.SECONDS.toMillis(3))
        )
        val pdfButton = device.findObject(By.text("确认并导出"))
        assertNotNull("Plan-book export action missing", pdfButton)
        assertTrue("Plan-book export must be disabled before consent", !pdfButton!!.isEnabled)
        clickText("我已取得顾客同意，将含人脸的方案预览导出为 PDF")
        assertTrue("Plan-book export should enable after consent", device.findObject(By.text("确认并导出")).isEnabled)
        clickText("取消")
    }

    @Test
    fun portableBackupExportRequiresStrongMatchingPasswordAndConsent() {
        launch(syntheticDesign = true)
        assertTrue(
            "Synthetic test design did not load",
            device.wait(Until.hasObject(By.textContains("合成测试照片已加载")), TimeUnit.SECONDS.toMillis(5))
        )
        scrollTo("保存本机案例")
        clickText("保存本机案例")
        assertTrue(
            "Case-save dialog did not open",
            device.wait(Until.hasObject(By.text("保存案例到本机")), TimeUnit.SECONDS.toMillis(3))
        )
        typeIntoOnlyDialogField("CI_PORTABLE_BACKUP")
        clickText("我已取得顾客对本机保存人脸照片、识别轮廓和眉形方案的明确授权")
        clickText("保存")
        assertTrue(
            "The case should save before backup",
            device.wait(Until.hasObject(By.textContains("案例已保存到本机")), TimeUnit.SECONDS.toMillis(20))
        )

        scrollTo("导出加密案例")
        clickText("导出加密案例")
        assertTrue(
            "Saved-case chooser did not open",
            device.wait(Until.hasObject(By.text("选择要导出的已保存案例")), TimeUnit.SECONDS.toMillis(3))
        )
        clickText("下一步")
        assertTrue(
            "Portable backup password dialog did not open",
            device.wait(Until.hasObject(By.text("创建口令保护的案例备份")), TimeUnit.SECONDS.toMillis(3))
        )
        val passwordFields = device.findObjects(By.clazz("android.widget.EditText"))
        assertTrue("Expected password and confirmation fields", passwordFields.size >= 2)
        val exportButton = device.findObject(By.text("加密并导出"))
        assertNotNull("Backup export button is missing", exportButton)
        assertTrue("Backup export must require consent and a matching password", !exportButton!!.isEnabled)

        passwordFields[0].text = "correct-horse-battery-27"
        passwordFields[1].text = "correct-horse-battery-27"
        assertTrue("Backup export must require explicit consent", !device.findObject(By.text("加密并导出")).isEnabled)
        clickText("我已取得顾客对导出加密案例备份的明确授权")
        assertTrue("Valid password and consent should enable export", device.findObject(By.text("加密并导出")).isEnabled)
        clickText("取消")
    }

    @Test
    fun transferErrorRecordCanBeSavedAndExported() {
        launch(syntheticDesign = true)
        assertTrue(
            "Synthetic test design did not load",
            device.wait(Until.hasObject(By.textContains("合成测试照片已加载")), TimeUnit.SECONDS.toMillis(5))
        )

        scrollTo("转印误差记录")
        clickText("转印误差记录")
        assertTrue(
            "Transfer validation dialog did not open",
            device.wait(Until.hasObject(By.text("打印与转印误差记录")), TimeUnit.SECONDS.toMillis(3))
        )
        val fields = device.findObjects(By.clazz("android.widget.EditText"))
        assertTrue("Expected profile, skin arc, stencil length and endpoint error fields", fields.size >= 4)
        fields[0].text = "CI_Printer_A4_100pct"
        fields[1].text = "10.0"
        fields[2].text = "10.5"
        fields[3].text = "1.2"
        clickText("我已取得顾客同意，记录匿名的打印与转印几何误差")
        clickText("保存验证记录")

        scrollTo("导出转印验证 CSV")
        clickText("导出转印验证 CSV")
        assertTrue(
            "Transfer-validation CSV export did not complete",
            device.wait(Until.hasObject(By.textContains("已导出转印验证 CSV")), TimeUnit.SECONDS.toMillis(5))
        )
    }

    @Test
    fun savedPlanAndEncryptedCaseCanBeReopened() {
        launch(syntheticDesign = true)
        assertTrue(
            "Synthetic test design did not load",
            device.wait(Until.hasObject(By.textContains("合成测试照片已加载")), TimeUnit.SECONDS.toMillis(5))
        )

        scrollTo("保存方案")
        clickText("保存方案")
        assertTrue(
            "Save-plan dialog did not open",
            device.wait(Until.hasObject(By.text("保存当前眉形")), TimeUnit.SECONDS.toMillis(3))
        )
        typeIntoOnlyDialogField("CI方案_自然款")
        clickText("保存")
        assertTrue(
            "Plan was not added",
            device.wait(Until.hasObject(By.text("CI方案_自然款")), TimeUnit.SECONDS.toMillis(5))
        )

        scrollTo("保存本机案例")
        clickText("保存本机案例")
        assertTrue(
            "Case-save dialog did not open",
            device.wait(Until.hasObject(By.text("保存案例到本机")), TimeUnit.SECONDS.toMillis(3))
        )
        typeIntoOnlyDialogField("CI持久化案例")
        clickText("我已取得顾客对本机保存人脸照片、识别轮廓和眉形方案的明确授权")
        clickText("保存")
        assertTrue(
            "Case save did not finish",
            device.wait(Until.hasObject(By.textContains("案例已保存到本机")), TimeUnit.SECONDS.toMillis(20))
        )

        scrollTo("案例库")
        clickText("案例库")
        assertTrue(
            "Local case library did not open",
            device.wait(Until.hasObject(By.textContains("本机案例库")), TimeUnit.SECONDS.toMillis(3))
        )
        val savedCase = device.findObject(By.textContains("CI持久化案例"))
        assertNotNull("Saved case was not listed", savedCase)
        savedCase!!.click()
        clickText("打开")
        assertTrue(
            "Encrypted case did not reopen successfully",
            device.wait(Until.hasObject(By.textContains("本机案例已恢复")), TimeUnit.SECONDS.toMillis(20))
        )
        scrollTo("CI方案_自然款")
        assertNotNull("Plan name did not survive case serialization", device.findObject(By.text("CI方案_自然款")))
    }
}
