package com.example.facepattern

import android.content.Intent
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.UiScrollable
import androidx.test.uiautomator.UiSelector
import androidx.test.uiautomator.Until
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.TimeUnit

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
    fun launchWithoutCameraShowsPhotoImportAndLocalCaseLibrary() {
        launch(syntheticDesign = false)
        assertNotNull(device.findObject(By.text("拍照设计")))
        assertNotNull(device.findObject(By.text("导入照片")))
        assertNotNull(device.findObject(By.text("本机案例库")))
        assertTrue(device.hasObject(By.text("仪器测试模式。相机未启动。")))
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
