# 模型来源、运行方式与使用限制

## 1. 内置眉区预分割模型

应用构建流程从以下地址获取 `faceparsing.tflite`：

- 模型页：https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT
- 文件地址：https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT/resolve/main/faceparsing.tflite
- 架构：BiSeNet，ResNet-18 主干
- 权重文件约 53 MB，输入为 float32 `[1,3,512,512]`，RGB，按 ImageNet mean/std 归一化
- 输出为 float32 `[1,19,512,512]`，每个像素对 19 个类别的 logits
- 标签 2 为左眉毛，标签 3 为右眉毛
- 本应用取标签 2、3 生成原生眉区掩膜。如果模型加载失败、推理异常或预测几乎为空，则回退到自适应对比度预分割
- 来源模型页报告 Pixel 8a 的特定 LiteRT GPU 路径约 22 ms，但本应用使用普通 TensorFlow Lite Interpreter 路径，不得把该数字当作本应用在 OPPO Find X7 上的实测速度

### 许可说明

Hugging Face 模型页将转换后的文件标记为 MIT，模型源为 `zllrunning/face-parsing.PyTorch`。但相关模型基于 CelebAMask-HQ 的 19 类面部解析标注体系。CelebAMask-HQ 官方协议明确将数据集用途限制为非商业研究与教育，并限制对其图像及衍生数据进行商业利用。

因此，仓库虽可在构建时下载并集成该文件进行技术测试，但**不能据此保证使用该模型权重或输出进行收费纹绣服务已获许可**。在收费门店场景启用前，应向权利人确认书面授权，或使用来源、训练数据和权重许可均明确允许该用途的模型。不要把“只自用、不上架”误认为当然属于非商业使用。

原始数据集协议：
https://github.com/switchablenorms/CelebAMask-HQ

模型卡：
https://huggingface.co/litert-community/BiSeNet-Face-Parsing-LiteRT

## 2. 将来提供的客户自训模型

若获得适当授权的门店照片并完成眉毛掩膜人工复核，训练脚本可以生成 `brow_segmentation.tflite`。应用会优先加载该文件，要求：

- 输入：float32 `[1,256,256,3]`，RGB，值域 0–1
- 输出：float32 `[1,256,256,1]`，sigmoid 概率掩膜
- 用顾客案例 ID 分组，确保同一顾客不会跨训练集、验证集和测试集
- 达到独立分组测试门槛仍不代表真实门店已完成前瞻性验证

客户自训模型只应由获得相应用途授权的资料训练，并按原始照片授权范围控制访问、保留、导出和删除。

## 3. 眉形排序和三维校正

- 眉形推荐没有通过独立验证的通用“最佳眉形”权重。应用保留可解释的几何基线与手工调整，因为个人审美偏好必须由顾客反馈数据来验证。
- 眉弓三维曲面不能从单张正面照片唯一恢复。实测弧长/弦长修正仍是实验性功能，默认关闭。打印比例校准只能校正整页的缩放，不等同于三维曲面展开。
