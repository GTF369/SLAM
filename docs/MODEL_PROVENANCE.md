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
- 输出为 8 类像素 logits，输出形状 `[1,8,384,384]`。应用通过 argmax 获取每像素类别。
- 类别 2 表示左眉，类别 3 表示右眉。应用映射为原生眉区预分割掩膜，继续支持人工补画/擦除。
- 如果模型文件无法加载、推理异常或掩膜几乎为空，应用退回自适应对比度预分割。
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
- 应用使用 MediaPipe 的相对 z 深度估计，可选计算局部二维与近似三维线段长度比例。这个 z 值不是毫米级皮肤表面测量，不能称为真实三维重建。该功能默认关闭。
- 物理打印比例校准单独调整整页尺寸，不会还原眉弓深度。使用前仍需以 20 mm 方框实测打印尺寸。
