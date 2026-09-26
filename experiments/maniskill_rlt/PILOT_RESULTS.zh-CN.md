# 交互记忆：小规模实验与验收

这份记录说明 2026-09-26 的实际运行结果：记忆分支已经接通真实 GPU rollout、TD 更新和断点续跑，但尚未证明记忆能加快 RLT 收敛。先理解方法，再区分工程验证与效果验证。[English](PILOT_RESULTS.md)

## 方法如何接入原来的 RLT

原来的 actor 根据当前 RL token、关节状态和参考动作产生动作。新增模块只补充“此前命令与实际结果”：执行完成后，保存起始关节、实际执行的动作前缀、关节变化、执行长度与结束标记。下一次决策读取近期 4 条记录，并从最多 32 条 archive 中检索 4 条旧记录，压成 64 维上下文，拼接到原 actor/critic 输入。

![已实现的交互记忆架构](figures/architecture.svg)

蓝色为冻结模块，绿色为可训练网络，虚线为训练信号。图对应当前实现，不是 Zeva 原论文架构。提供可编辑 [SVG](figures/architecture.svg)、论文排版用 [PDF](figures/architecture.pdf) 和 [PNG](figures/architecture.png)，用 `python experiments/maniskill_rlt/draw_architecture.py` 重新生成。

读取器由 critic 的 TD loss 更新；actor 读取的上下文经过 `detach()`，actor 自身仍由 Q＋BC 目标更新。Stage 1 和 VLA 不变，没有 Stage 1B。replay 保存当时的记忆快照，不用后来的经验补写过去。这里只保存关节控制证据，没有历史视觉、触觉或力觉，也没有推导接触标签。

借鉴 [Zeva 官方实现](https://github.com/air-embodied-brain/Zeva) 的是“动作—实际变化”证据与不同时间尺度的记忆。Zeva 的视觉因果 token、阶段/效果训练、prompt 接口和部署时更新记忆但冻结网络的方案没有被完整移植；本实现在线更新 TD 读取器，不能称为 Zeva 复现或因果识别。

## 真实训练发现并修复的问题

最初 smoke 能正常退出，却有 17 个记忆读取器张量在 checkpoint 间完全不变。FSDP 默认参数展平使按参数名分组的 optimizer 没有把读取器正确分配给 critic。CPU 上有梯度，并不代表实际分布式 optimizer 会更新它。

实验 overlay 现在强制 `actor.fsdp_config.use_orig_params: True`，配置校验拒绝不兼容组合，并用生产 optimizer 分组逻辑增加回归测试。launcher 在结束后比较相邻 checkpoint 的 `memory_encoder.*`；没有真实变化就返回失败，而不是只检查进程退出码。

| 验证 | 结果 |
| --- | --- |
| 修复后的真实 GPU smoke | global step 1–4，退出码 0 |
| step 1 对比 step 4 | 17/17 个读取器张量更新，最大绝对变化 0.00057749 |
| 从 step 4 恢复到 step 6 | 退出码 0，step 5→6 仍有 17/17 个张量更新 |
| 两个 optimizer 的 step | 8→12，保留状态继续更新，不是重新初始化训练 |
| CPU 回归 | 195 passed，1 skipped，1 deselected |

续跑保存/恢复 actor、critic、target、optimizer、scheduler 和 replay；不代表模拟器状态及其活动记忆逐帧复现。跳过的是原有可选 Gemma3 检查；排除的是 baseline 已存在的 `VideoMetadata` 依赖不兼容测试，没有改依赖掩盖问题。

所有 GPU 实验仅用物理 GPU 2：CUDA 可见性、RLinf placement 和 Vulkan PCI 设备均隔离。GPU 0/1 的 Stage 1 进程未被停止或重启。每个 smoke 只用 2 个训练环境、1 个评估环境、40 个控制步的 episode，不用于估计成功率。

## 记忆是否提供了有效信息

为避免把工程通过当成算法有效，额外用真实示范数据训练了一个小型诊断网络。输入当前关节、接下来 10 步命令，以及实际读取器产生的历史上下文；目标是这 10 步真实关节变化。它不是在线 actor，不使用 VLA，不代表 RL 成功率。

无记忆对照使用相同网络结构，但将上下文置零；按完整 episode 划分训练与验证，固定三个种子 2026/2027/2028，训练预算一致。每个样本先取历史快照，再写入当前执行结果，避免未来泄漏。验证时另打乱历史而保留当前关节和命令。

| 数据与预算 | 种子 | 无记忆 MSE | 有记忆 MSE | 有记忆、打乱历史 MSE |
| --- | --- | --- | --- | --- |
| 32 episodes，300 updates | 2026 | 0.00028224 | 0.00029212 | 0.00179913 |
| 同上 | 2027 | 0.00027725 | 0.00029476 | 0.00152981 |
| 同上 | 2028 | 0.00025943 | 0.00025684 | 0.00074894 |
| 64 episodes，600 updates | 2026 | 0.00010699 | 0.00010731 | 0.00081252 |
| 同上 | 2027 | 0.00011660 | 0.00011120 | 0.00081112 |
| 同上 | 2028 | 0.00010306 | 0.00010496 | 0.00038523 |

结论是**没有跨种子一致的优势**。打乱历史会变差，说明网络使用了历史，但打乱也制造了不匹配输入，不能证明有用的物理因果推理。扩大这一小预算后仍无稳定优势，因此保留负结果，没有继续围绕同一验证集调参。32/64 episodes 两轮都属于探索性验证，不是独立最终测试。

一个待检验解释是：同一控制器的成功示范中，当前关节与命令已经能预测很多变化。下一轮应固定新的测试集，引入明确的隐藏动力学差异、失败/恢复数据，再与近期历史、检索旧历史、无记忆三组对照比较；不能仅用成功示范宣称跨任务经验迁移。

## 重现与定位结果

下面第一条启动器命令显式带 `--check`，只做预检；第二条才运行有界 smoke。`RLT_SMOKE_STEPS` 是停止的全局 step，范围 1–20，不是额外训练步数。续跑用已有 `global_step_N` 目录设置 `RLT_SMOKE_RESUME_DIR`，并将停止 step 至少提高 2，便于审计相邻新 checkpoint。

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory --check
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  RLT_SMOKE_STEPS=4 RLT_SMOKE_RAY_PORT=6402 \
  bash run_rlt_stage2_smoke_gpu2.sh --memory
```

预检不分配 GPU；第二条会启动真实小实验，GPU 2 忙碌时拒绝启动。launcher 不执行全局 `ray stop`，也不清理其他任务的文件。CPU 诊断由 `python -m toolkits.rlt.probe_interaction_memory --dataset DATASET --output NEW_OUTPUT --episodes 64 --steps 600` 执行；输出目录必须是新目录。

所有大文件留在 NAS 根目录 `/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill`，没有加入 Git：

| 相对目录 | 用途 |
| --- | --- |
| `runs/stage2_smoke/memory_stage2_gpu2_20260926_222443_2996233` | 修复前，读取器未更新；不是有效训练证据 |
| `runs/stage2_smoke/memory_stage2_gpu2_20260926_223451_3027410` | 修复后的 4-step smoke |
| `runs/stage2_smoke/memory_stage2_gpu2_20260926_223716_3036341` | 从 4 到 6 的续跑 |
| `research/zeva_probe_20260926_2234/results.json` | 第一轮探针，含 episode、数据哈希与逐种子结果 |
| `research/zeva_probe_20260926_2238/results.json` | 第二轮探针 |

核心代码沿闭环阅读：`rlinf/algorithms/rlt/interaction_memory.py` 存储与校验 → `rlinf/models/embodiment/modules/rlt_memory_encoder.py` 读取 → `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` 拼接 → `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` 更新。`toolkits/rlt/audit_adapter.py` 检查真实参数变化。两条研究分支保持独立，暂不合入 baseline。
