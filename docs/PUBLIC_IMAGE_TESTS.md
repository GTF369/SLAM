# 网络公开图片烟测

## 测试图片

- `virtual_makeup_demo.png`: [otdnnc/virtual-makeup demo asset](https://github.com/otdnnc/virtual-makeup/blob/main/apps/web/public/f.png). 仓库代码许可标注 MIT。此图片只在 CI 运行时下载，不提交到本仓库。
- `unsplash_portrait.jpg`: [Unsplash portrait, photo id 1534528741775-53994a69daeb](https://unsplash.com/photos/9MR78HGoflw) 。从 Unsplash 图片 CDN 下载，遵循 [Unsplash License](https://unsplash.com/license)。此图片只在 CI 运行时下载，不提交到本仓库。

测试任务还会从 demo 人像派生低亮度、模糊和水平镜像版本。派生图仅保存在 CI 临时目录和短期工件中。

## 测试内容

CI 下载与 Android 包相同版本的 `face_landmarker.task` 与 `faceparsing.onnx`，然后运行 `ml/public_image_smoke_test.py`。脚本复现应用使用的 MediaPipe 人脸关键点、眉区 ROI、EasyPortrait 输入归一化与类别标签、掩膜坐标映射，以及弱掩膜回退到自适应对比度的逻辑。它输出每个案例的 JSON 检查结果、眉区覆盖图、仅模型类别覆盖图和 class 2–8 对照图。

这属于模型/图像处理流程烟测，不是 Android APK 运行测试，也不是分割准确率基准。图片没有纹绣师逐像素标注，不能计算 Dice/IoU，也不能据此声称个性化推荐有效。GitHub Actions 上传图像结果时使用 `if: always()`，即便烟测失败也尽量保留报告以便分析。

## 后续人工验收

1. 对每张输出覆盖图检查左右眉是否准确落在真实眉毛上，而不是眼睛、眼影、眉骨阴影或额头。
2. 用授权真实客户图像建立由专业人员标注的验证集，按顾客隔离开发集和测试集。
3. 在 OPPO Find X7 安装 ARM64 APK，验证相机授权、拍摄坐标、旋转/镜像、掩膜编辑、撤销、保存与方案比较。
4. 在指定打印机上验证 20 mm 校准框、模板尺寸和实际转印偏差。

## 已知限制

- 单张图片会因姿态、妆容、光照、眉毛稀疏程度和遮挡影响识别。
- 相对 MediaPipe z 值不是以毫米计量的真实皮肤表面深度。
- 公开图片仅用于软件健壮性检查；不得未经许可将人物图像用于训练、产品宣传或用户案例。
