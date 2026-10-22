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
