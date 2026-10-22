# 模型来源、运行方式与使用限制

## 1. 内置眉区预分割模型

Android 构建流程从 EasyPortrait ONNX 模型仓库下载 `bisenet_fp.onnx`，并将资产命名为 `faceparsing.onnx`。

- 模型仓库：https://huggingface.co/sadzip/EasyPortrait-ONNX
- 原始训练项目：https://github.com/hukenovs/easyportrait
- 文件大小约 13.4 MB。
- 模型类别：BiSeNet face parsing。
- 输入形状：`[1,3,384,384]`，NCHW，BGR 通道顺序，像素值位于 0–255，再按模型卡中的 mean/std 标准化。
- 输入 mean：`[143.55267075,132.96705975,126.94924335]`。
- 输入 std：`[60.2625333,60.32740275,59.30988645]`。
- 模型卡概述为 8 类，但下载到的 `bisenet_fp.onnx` 在 ONNX Runtime CPU 上实际输出 19 通道，形状为 `[1,19,384,384]`。CI 工作流会实际执行推理并检查张量形状和数值是否有限。
- 初始版本错误地套用了常见 CelebAMask-HQ 的类别顺序，把第 6、7 类当作眉毛。2026-10-22 的公开图像诊断在模型输出第 2–8 类上逐通道可视化，发现本次下载的 EasyPortrait ONNX 输出中第 2、3 类与两侧眉毛区域对齐，而第 6、7 类主要落在眼部和嘴唇附近。因此 Android 端和冒烟测试已改用第 2、3 类。该诊断属于少量公开照片上的映射排错，不含专业眉毛标注真值，不能替代独立分割精度验证。任何掩膜仍需纹绣师复核。
- 如果模型文件无法加载、推理异常或掩膜几乎为空，应用退回自适应对比度预分割。
- GitHub Actions 的 `public-image-smoke-test` 会使用与 APK 相同来源的模型资产，在公开样例照片上执行 MediaPipe 关键点检出、ONNX 推理、当前类别映射、眉区 ROI 及回退逻辑，并上传覆盖层和类别诊断图。它不运行 Android APK，也没有专业标注真值，属于模型链路冒烟检查。
- ONNX 导出仓库声称转换结果已与原始 PyTorch/mmseg 输出完成数值与 argmax 一致性验证。此验证不等同于在本应用的顾客、相机或 OPPO Find X7 上完成验证。

模型输入、输出与类别来源：
https://huggingface.co/sadzip/EasyPortrait-ONNX

## 2. 许可与署名

EasyPortrait 仓库将导出模型仓库标注为 Apache-2.0。原始 EasyPortrait 项目声明其数据集与模型采用一个 CC BY-SA 4.0 的变体许可。模型作者在原仓库公开问题中也确认，已发布模型沿用相同许可。

- EasyPortrait 原项目许可说明：https://github.com/hukenovs/easyportrait
- 作者确认模型许可的公开回复：https://github.com/hukenovs/easyportrait/issues/25
- 具体许可文件：https://github.com/hukenovs/easyportrait/blob/main/license/en_us.pdf

本项目保留模型来源、署名和许可链接。CC BY-SA 通常要求适当署名，并在分发衍生材料时遵守相同方式共享条件。上游使用的是“变体”许可，使用者应在商用部署或重新分发前阅读具体许可文件并确认其条件。这里的说明不是法律意见。

## 3. 将来提供的客户自训模型

应用继续优先尝试加载由获授权顾客标注数据训练的 `brow_segmentation.tflite`。要求：

- 输入：float32 `[1,256,256,3]`，RGB，值域 0–1。
- 输出：float32 `[1,256,256,1]`，sigmoid 概率掩膜。
- 顾客案例 ID 必须按人分组，使同一顾客不会同时出现在训练、验证和测试集合。
- 独立分组测试达到内部阈值仍不代表已通过真实门店前瞻性验证。

只使用获得相应用途授权的照片和人工标注，并依授权范围控制访问、保留、导出和删除。

## 4. 眉形排序和三维校正

- 应用暂时没有通过独立验证的通用“最佳眉形”模型。眉形偏好具有个体性，因此排序仍采用可解释几何基线，需积累真实客户评分后才能训练个性化偏好模型。
- MediaPipe 的相对 z 只能用于离线研究参考，不是毫米级皮肤表面测量，也不能称为真实三维重建。为避免未标定深度改变实体拓描稿，当前版本已禁止该估计进入打印几何。未来只有在独立 3D/物理参考与真实转印误差验证通过后，才考虑重新开放。
- 物理打印比例校准只校正整页缩放，不会还原眉弓深度。应用不再把 MediaPipe 相对 z 自动转成打印修正。对区域实测弧长系数的手动选项仍属实验用途，实物打印/转印误差必须另行记录并用 `ml/analyze_stencil_transfer.py` 分析。使用前仍需以 20 mm 方框实测打印尺寸。
