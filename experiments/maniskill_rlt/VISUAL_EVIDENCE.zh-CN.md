# 视觉证据诊断原型

这个分支用受控图像遮挡检查不同输出路径依赖哪些区域，帮助区分 VLA 参考动作与 Stage 2 actor 的失误。它只提供离线诊断接口和 CPU 测试；尚未接入真实 checkpoint、提取 attention 权重或验证机器人控制收益。

## 如何使用

在 `toolkits/rlt/visual_evidence.py` 中，`ImageRegion` 表示某个相机上的半开像素矩形；`diagnose_regions` 接收 `[B,C,H,W]` 浮点图像、区域列表与确定性预测回调。回调中固定语言、本体状态和 action noise，冻结模型并使用 eval 模式。分别返回 `vla_action`、`rl_token` 和 `actor_action`，不要将它们合为一张重要性图。

下面的微型例子演示调用合同，不是 VLA 适配器：

```python
import torch
from toolkits.rlt.visual_evidence import ImageRegion, diagnose_regions

images = {"main": torch.ones(1, 3, 8, 8)}
def predict(images):
    return {"example_feature": images["main"].mean((2, 3))}

report = diagnose_regions(
    images, [ImageRegion("main", 0, 0, 4, 4)], predict,
    replacement="zero",
)
```

结果包含每个输出路径、每个样本的 RMS 变化。提供同 shape 的示范目标后，还返回有符号的 MSE 增量：负值说明遮挡后更接近目标，不能截断成零。`mean` 用该相机各通道的空间均值填充；`zero` 的实际含义取决于图像归一化，不能一律称作黑色。两种替换都可能产生分布外输入。

每次诊断先对原图调用两次，检查输出逐元素一致，避免把采样噪声误当作区域敏感性；这只是运行检查，不证明回调绝无隐藏状态。N 个区域需要 N+2 次预测，最多 64 个区域。诊断不改变全局 RNG、模型参数或设备放置，也不保证用户回调无副作用。先在离线录制帧上调用，不要直接放进实时控制循环。

## 与论文及后续实验的关系

[ActGaze](https://arxiv.org/html/2609.28955v1) 已用反事实视觉干预形成动作相关 gaze 监督；[VLA-Trace](https://arxiv.org/html/2605.30117v2) 分析不同计算阶段的表征与行为。因此，这个工具不是新 attention 算法，也不将遮挡敏感性称作因果解释。

接真实模型时，先检查固定噪声下未遮挡输出与原始推理一致，再针对夹爪—木块前端—孔口关系、等面积背景和另一相机做对照。当前 API 没有 action 可执行性检查，没有自动选区域，更没有把未来结果或模拟器真值作为部署输入。下一步增加数据适配和带版本的离线输出；只有发现稳定失误机制后，再单独研究历史经验能否改善局部证据选择。

CPU 验收命令：`CUDA_VISIBLE_DEVICES='' python -m pytest tests/unit_tests/test_visual_evidence.py -q`。当前 8 项通过；未运行 GPU 或机器人评估。
