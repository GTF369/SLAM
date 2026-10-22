package com.example.facepattern

import android.Manifest
import android.graphics.Bitmap
import android.graphics.Color
import android.view.MotionEvent
import android.view.View
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.Until
import androidx.test.core.app.ActivityScenario
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BrowDesignSmokeTest {
    @get:Rule
    val grantCameraPermission: GrantPermissionRule =
        GrantPermissionRule.grant(Manifest.permission.CAMERA)

    @Test
    fun maskStrokeCanBeUndoneAndRedone() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val device = UiDevice.getInstance(instrumentation)
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        val photo = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.LTGRAY)
        }
        val overlay = Bitmap.createBitmap(300, 120, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.TRANSPARENT)
        }
        try {
            assertTrue("MainActivity did not become visible", device.wait(Until.hasObject(By.text("眉形设计助手")), 10_000))
            scenario.onActivity { activity ->
                val viewField = MainActivity::class.java.getDeclaredField("browView").apply { isAccessible = true }
                val view = viewField.get(activity) as View
                fun invokeViewMethod(name: String, parameterCount: Int, vararg args: Any?) {
                    val method = view.javaClass.declaredMethods.first {
                        it.name == name && it.parameterTypes.size == parameterCount
                    }.apply { isAccessible = true }
                    method.invoke(view, *args)
                }

                val result = BrowSegmentationResult(
                    overlay = overlay,
                    offsetX = 50,
                    offsetY = 50,
                    method = "instrumentation test mask",
                    maskQualityScore = 0.5f,
                    selectedPixelCount = 0,
                    regions = listOf(
                        intArrayOf(70, 60, 190, 110),
                        intArrayOf(210, 60, 340, 110)
                    )
                )
                invokeViewMethod("setPhoto", 1, photo)
                invokeViewMethod("setSegmentationResult", 2, result, true)
                invokeViewMethod("setSegmentationEditMode", 2, true, true)

                assertTrue("BrowDesignView has no measured size", view.width > 0 && view.height > 0)
                val scale = maxOf(
                    view.width.toFloat() / photo.width,
                    view.height.toFloat() / photo.height
                )
                val imageLeft = (view.width - photo.width * scale) / 2f
                val imageTop = (view.height - photo.height * scale) / 2f
                val imageX = 250f
                val imageY = 100f
                val touchX = imageLeft + imageX * scale
                val touchY = imageTop + imageY * scale
                assertTrue("Touch point is outside the view", touchX in 0f..view.width.toFloat() && touchY in 0f..view.height.toFloat())

                val downTime = SystemClock.uptimeMillis()
                val down = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, touchX, touchY, 0)
                val up = MotionEvent.obtain(downTime, downTime + 20L, MotionEvent.ACTION_UP, touchX, touchY, 0)
                try {
                    view.dispatchTouchEvent(down)
                    view.dispatchTouchEvent(up)
                } finally {
                    down.recycle()
                    up.recycle()
                }

                val maskX = (imageX - result.offsetX).toInt()
                val maskY = (imageY - result.offsetY).toInt()
                assertTrue("A brush stroke did not modify the mask", Color.alpha(overlay.getPixel(maskX, maskY)) > 0)

                val undoMethod = view.javaClass.getDeclaredMethod("undoLastMaskStroke").apply { isAccessible = true }
                assertEquals(true, undoMethod.invoke(view))
                assertEquals("Undo did not restore the original alpha", 0, Color.alpha(overlay.getPixel(maskX, maskY)))

                val redoMethod = view.javaClass.getDeclaredMethod("redoLastMaskStroke").apply { isAccessible = true }
                assertEquals(true, redoMethod.invoke(view))
                assertTrue("Redo did not restore the edited alpha", Color.alpha(overlay.getPixel(maskX, maskY)) > 0)

                invokeViewMethod("setSegmentationEditMode", 2, false, true)
                invokeViewMethod("setSegmentationResult", 2, null, false)
                invokeViewMethod("setPhoto", 1, null)
            }
        } finally {
            scenario.close()
            if (!photo.isRecycled) photo.recycle()
            if (!overlay.isRecycled) overlay.recycle()
        }
    }

    @Test
    fun localCaseIsPersistedAndListedInCaseLibrary() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val device = UiDevice.getInstance(instrumentation)
        val context = instrumentation.targetContext
        val root = File(context.filesDir, "brow_cases")
        root.deleteRecursively()
        val scenario = ActivityScenario.launch(MainActivity::class.java)
        val photo = Bitmap.createBitmap(96, 96, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.rgb(180, 160, 145))
        }
        val mask = Bitmap.createBitmap(60, 20, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.TRANSPARENT)
            setPixel(8, 6, Color.argb(172, 71, 190, 157))
        }
        val segmentation = BrowSegmentationResult(
            overlay = mask,
            offsetX = 20,
            offsetY = 20,
            method = "instrumentation test mask",
            maskQualityScore = 0.5f,
            selectedPixelCount = 1,
            regions = listOf(intArrayOf(25, 22, 45, 30), intArrayOf(50, 22, 70, 30))
        )
        val points = List(478) { android.graphics.PointF(48f, 48f) }.toMutableList().apply {
            this[10] = android.graphics.PointF(48f, 10f)
            this[152] = android.graphics.PointF(48f, 90f)
            this[234] = android.graphics.PointF(20f, 60f)
            this[454] = android.graphics.PointF(76f, 60f)
            this[33] = android.graphics.PointF(35f, 40f)
            this[263] = android.graphics.PointF(61f, 40f)
            this[172] = android.graphics.PointF(25f, 75f)
            this[397] = android.graphics.PointF(71f, 75f)
            this[70] = android.graphics.PointF(29f, 33f)
            this[63] = android.graphics.PointF(32f, 31f)
            this[105] = android.graphics.PointF(36f, 30f)
            this[66] = android.graphics.PointF(39f, 31f)
            this[107] = android.graphics.PointF(42f, 33f)
            this[300] = android.graphics.PointF(54f, 33f)
            this[293] = android.graphics.PointF(57f, 31f)
            this[334] = android.graphics.PointF(60f, 30f)
            this[296] = android.graphics.PointF(63f, 31f)
            this[336] = android.graphics.PointF(66f, 33f)
        }
        try {
            assertTrue("MainActivity did not become visible", device.wait(Until.hasObject(By.text("眉形设计助手")), 10_000))
            scenario.onActivity { activity ->
                MainActivity::class.java.getDeclaredField("capturedPhoto").apply {
                    isAccessible = true
                    set(activity, photo)
                }
                MainActivity::class.java.getDeclaredField("segmentationResult").apply {
                    isAccessible = true
                    set(activity, segmentation)
                }
                MainActivity::class.java.getDeclaredField("browView").apply {
                    isAccessible = true
                }.get(activity).javaClass.getDeclaredMethod("setFacePoints", List::class.java).apply {
                    isAccessible = true
                    invoke(MainActivity::class.java.getDeclaredField("browView").apply {
                        isAccessible = true
                    }.get(activity), points)
                }
                MainActivity::class.java.getDeclaredField("showNativeSegmentation").apply {
                    isAccessible = true
                    setBoolean(activity, true)
                }
                MainActivity::class.java.getDeclaredField("segmentationReviewed").apply {
                    isAccessible = true
                    setBoolean(activity, true)
                }
                MainActivity::class.java.getDeclaredField("currentCaseId").apply {
                    isAccessible = true
                    set(activity, "QA_CASE_2026")
                }
                MainActivity::class.java.getDeclaredMethod("saveCurrentCaseToLibrary").apply {
                    isAccessible = true
                    invoke(activity)
                }
            }
            assertTrue("Save dialog did not appear", device.wait(Until.hasObject(By.text("保存案例到本机")), 5_000))
            device.findObject(By.text("保存")).click()

            val deadline = SystemClock.uptimeMillis() + 15_000L
            var caseDirectory: File? = null
            while (SystemClock.uptimeMillis() < deadline && caseDirectory == null) {
                caseDirectory = root.listFiles()?.firstOrNull { File(it, "case.json").isFile }
                if (caseDirectory == null) SystemClock.sleep(200L)
            }
            assertNotNull("Case manifest was not written", caseDirectory)
            val savedDirectory = caseDirectory ?: throw AssertionError("Case manifest was not written")
            val manifest = File(savedDirectory, "case.json")
            assertTrue(manifest.readText(Charsets.UTF_8).contains("QA_CASE_2026"))
            val manifestJson = org.json.JSONObject(manifest.readText(Charsets.UTF_8))
            val photoName = manifestJson.getString("photo_file")
            assertTrue("Persisted case photo is missing", File(savedDirectory, photoName).isFile)
            val maskMetadata = manifestJson.getJSONObject("current_segmentation")
            val maskName = maskMetadata.getString("mask_file")
            assertTrue("Persisted segmentation mask is missing", File(savedDirectory, maskName).isFile)

            scenario.onActivity { activity ->
                MainActivity::class.java.getDeclaredMethod("openSavedCaseLibrary").apply {
                    isAccessible = true
                    invoke(activity)
                }
            }
            assertTrue(
                "The saved case was not listed",
                device.wait(Until.hasObject(By.textContains("眉形案例 QA_CASE_2026")), 5_000)
            )
            device.findObject(By.text("打开")).click()
            assertTrue(
                "The saved case did not restore the design state",
                device.wait(Until.hasObject(By.textContains("已恢复本机案例")), 10_000)
            )
            scenario.onActivity { activity ->
                val captured = MainActivity::class.java.getDeclaredField("capturedPhoto").apply {
                    isAccessible = true
                }.get(activity) as? Bitmap
                assertNotNull("Restored photo is missing in memory", captured)
                val restoredMask = MainActivity::class.java.getDeclaredField("segmentationResult").apply {
                    isAccessible = true
                }.get(activity) as? BrowSegmentationResult
                assertNotNull("Restored segmentation mask is missing in memory", restoredMask)
                assertTrue("Restored mask bitmap is empty", (restoredMask?.overlay?.width ?: 0) > 0)
            }
            device.pressBack()
        } finally {
            scenario.close()
            if (!photo.isRecycled) photo.recycle()
            if (!mask.isRecycled) mask.recycle()
            root.deleteRecursively()
        }
    }
}
