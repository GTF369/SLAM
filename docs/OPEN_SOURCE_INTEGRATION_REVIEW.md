# 开源方案调研与集成决策

审查日期：2026-10-22

本文针对眉形设计助手的 Android 原型进行开源项目适用性审查。目标不是简单堆叠依赖，而是选择能提高眉区定位、编辑、个性化排序和曲面校正可信度的组件，同时避免把研究型重建结果误当作真实毫米级测量。

## 结论摘要

现有方案已经采用 MediaPipe Face Landmarker、EasyPortrait 导出的 ONNX 人脸解析模型、可人工编辑的眉区掩膜、参数化眉形、按匿名顾客分组的偏好模型训练流程、区域弧长/弦长实测以及 Android Print Framework。最值得补强的是验证质量，而不是替换整套技术栈。

本次分支进一步将打印比例校准按设备/纸张/驱动的用户命名配置隔离，防止不同打印设置误用同一个全局比例系数；并为眉区分割模型增加按 ROI 和按顾客汇总的评估指标。它不会自动产生真实顾客数据，也不会替代目标打印机的实测。此外，公开人像的逐通道可视化发现原先套用的第 6、7 类映射不正确，并已在 Android 引擎和 smoke-test 中改为第 2、3 类。这个修复解决了已观察到的类别索引错误，但并不证明它在真实纹绣顾客中的轮廓准确率。

## 开源项目审查

| 项目 | 可以借鉴的内容 | 不应直接假定的能力 | 集成决策 |
|---|---|---|---|
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | Android 端人脸关键点与视觉任务，已有依赖与照片坐标流程可复用 | 关键点和相对 z 不是皮肤表面的毫米级 3D 扫描 | 继续使用现有模块，重点补真机与照片质量验证 |
| [EasyPortrait](https://github.com/hukenovs/easyportrait) / [EasyPortrait ONNX 模型](https://huggingface.co/sadzip/EasyPortrait-ONNX) | 通用人脸解析的眉区预分割，能减少纯阈值分割的依赖 | 类别索引必须匹配实际下载的模型资产，化妆、阴影和浅色眉毛也可能导致错分 | 已用公开图像逐通道检查并把应用映射从 6/7 修正为 2/3；仍需真实纹绣标注集验证 |
| [BiSeNet 面部分割示例](https://github.com/waleed-Amjad/Face-parsing-segmentation) | PyTorch/ONNX 推理、类别掩膜与可视化评估的参考 | 仓库里出现模型权重不代表该权重已对本应用的客户分布验证，也不自动代表可商用 | 参考评估与排障方式，不盲目替换当前模型 |
| [Virtual_Makeup](https://github.com/Jayanths9/Virtual_Makeup) | MediaPipe 关键点、多边形区域与 OpenCV 图像叠加；可作为渲染回归案例 | 这是 Python 桌面应用，目标是颜色妆效，不负责个性化眉形推荐、物理打印或 3D 转印 | 作为视觉叠加参考，不作为 Android 运行时依赖；复制代码前先单独审查许可 |
| [virtual-makeup (React)](https://github.com/otdnnc/virtual-makeup) | 客户端人脸关键点、分割、可调预设与本地处理的 UI 设计参考；仓库标注 MIT | Web 技术栈不能直接提供 Android 原生相机、打印校准或眉弓物理测量 | 只参考隐私优先与预设交互，不迁移整套前端 |
| [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) | 单图 3DMM 面部密集对齐，官方仓库提供 ONNX 推理路径，代码许可证标注 MIT | 单图 3DMM 输出仍是统计模型估计；代码许可不自动涵盖每个模型权重、训练数据或衍生资产 | 可作为独立离线研究基线，与实测数据对比；在完成权重/资产许可审核和误差验证前不进入正式打印链路 |
| [DECA](https://github.com/yfeng95/DECA) | 从单张图像估计头部形状和细节几何，可做离线重建方法对照 | 依赖 PyTorch/PyTorch3D、模型数据与 FLAME 相关资产，不能直接视作轻量 Android 组件或真实物理扫描 | 暂不嵌入 Android；仅在独立研究环境评估，并先核对模型与数据许可 |

## 本次改动

### 1. 打印校准配置隔离

应用过去保存一个全局的打印比例修正值。现在可以为打印机、纸张和驱动组合填写配置名，并分别保存该配置的校准系数。旧版全局系数只作为“默认打印配置”的迁移回退，不会自动污染新配置。

打印服务无法可靠地向应用确认用户最终选择的每一台打印机，所以配置名称仍由操作者维护。每次变更打印机、纸张、驱动选项或“适合页面”设置，都必须重新校准，并实测 20 mm 方框。软件不能仅凭配置名保证输出真实尺寸。

### 2. 眉区分割评估按顾客平衡

新增 `ml/segmentation_metrics.py`，同时报告：

- 全部像素合并后的 Dice、IoU、Precision、Recall、Specificity、假阳性率和假阴性率。
- 每个眉区 ROI 的宏平均指标。
- 先对同一匿名顾客的多个 ROI 求均值，再对顾客等权平均的指标。

训练脚本保持原有 `test_dice` 与 `test_iou` 字段兼容性，同时保存更完整的评估字段。`--promote-to-app` 现在必须同时通过 pooled-pixel Dice 与 customer-macro Dice 阈值，防止整体像素分数掩盖某些顾客上的分割失败。但内部按顾客分组的测试依旧不是前瞻性外部验证。

### 3. 三维曲面校正的安全边界

真实的单个顾客眉弓表面并不能由一张正面照片唯一确定。3DDFA_V2 或 DECA 可以输出先验驱动的 3D 形状估计，但不能替代实测弧长、独立 3D 参考或真实皮肤转印误差测量。

推荐研究流程：

1. 将单图 3D 重建只用于生成候选几何参考，不把预测深度直接标为毫米。
2. 对左/右眉头、眉身、眉峰和眉尾用统一端点记录表面弧长、弦长、测量人员与重复读数。
3. 使用一批未参与拟合的顾客评估区域修正的误差及覆盖率。
4. 将最终的数字模板打印出来，测量 20 mm 标尺框，并记录轮廓尺寸和实际皮肤转印位置误差。
5. 使用应用的“转印误差记录”按打印配置与眉区记录皮肤弧长、模板路径长度和端点偏差。通过 `ml/analyze_stencil_transfer.py` 计算顾客级误差分布。
6. 在误差达到事先约定的门槛前，保持自动曲率修正默认关闭。MediaPipe 相对 z 目前不进入实体打印几何。

## 仍需真实数据或实体设备才能完成的事项

以下事项不能通过下载开源仓库或修改代码伪造完成：

- 真实顾客偏好排序模型：需要授权收集多候选评分，分离开发、校准和独立外部测试顾客。
- 专用眉毛分割模型：需要授权图片、专业标注、按顾客隔离的测试集和不同肤色、妆容、光照及眉毛密度的错误分析。
- 个体三维曲面：需要可靠的独立 3D/物理参考，且必须和打印转印误差关联验证。
- 打印闭环：需要在 OPPO Find X7、实际打印机、纸张、打印服务设置及真实拓描流程上逐项测试。
- 正式交付：还需要隐私与数据保留政策审查、模型资产许可审核、签名 Release 构建与升级测试。

## 参考链接

- MediaPipe：<https://github.com/google-ai-edge/mediapipe>
- EasyPortrait 及其许可证说明：<https://github.com/hukenovs/easyportrait>
- EasyPortrait ONNX：<https://huggingface.co/sadzip/EasyPortrait-ONNX>
- BiSeNet 人脸解析示例：<https://github.com/waleed-Amjad/Face-parsing-segmentation>
- MediaPipe + OpenCV 虚拟妆效：<https://github.com/Jayanths9/Virtual_Makeup>
- 浏览器端虚拟妆效：<https://github.com/otdnnc/virtual-makeup>
- 3DDFA_V2：<https://github.com/cleardusk/3DDFA_V2>
- DECA：<https://github.com/yfeng95/DECA>
