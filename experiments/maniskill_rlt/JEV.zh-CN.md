# 从有限动作修正中学习选择

这条分支研究：在输入信息、VLA 权重和奖励保持一致时，把 RLT 的连续动作生成换成有限候选选择，能否减少无效探索。实现名称为 **RLT Atomic Decisions**，基于 baseline `d9ba471e`，分支为 `research/rlt-jev-atomic-decisions`。这是借鉴 Jev 接口思想的本地 RL 实验，不是 Jev 模型复现，也没有调用其 API。

先看动作如何执行，再看反馈如何训练网络，最后按验收命令检查实现。目前完成 CPU 验证和配置检查；真实 ManiSkill、GPU/FSDP、跨任务收益尚未验证。

![推理和学习路径](figures/rlt_atomic_architecture.svg)

## 原 RLT 保留了什么

原 Stage 1 的 VLA 根据图像、任务文字和机器人状态输出参考动作；RLT encoder 压缩出 RL token。Stage 2 冻结这部分，原 MLP actor 输入 `z_rl`、`proprio`、`ref_chunk` 后生成连续动作 chunk，critic 根据奖励学习这些动作的价值。

本分支继续使用相同的冻结特征模型。实际配置中 `z_rl` 为 2048 维，`proprio` 为 9 维，参考动作是 10 × 8：7 个关节控制量加 1 个夹爪控制量。输入来源见 [`extract_rlt_obs`](../../rlinf/models/embodiment/openpi/tasks/eval.py)。ManiSkill 的 `proprio` 已经经过模型输入变换，不能直接把它当作未归一化的关节角。

奖励、episode 终止、原有动作路由和人工／expert 标记没有改动。提供的 smoke 沿用 baseline 的 `full_task`、`always_on` 路由，且自动 expert 关闭；这不意味着已经实现了新的关键阶段识别器或 VR 接管。

## 新 actor 具体选择什么

选择器不再自由输出 80 个连续数，而是在当前 VLA 参考动作附近选择一个完整候选。Panda 的候选上限是 17 个：

| 候选 | 含义 |
| --- | --- |
| `reference` | 沿用裁剪到环境范围的参考动作 |
| `dampen_arm` | 关节控制量向参考值的一半靠近，修正量仍受限 |
| `brake_arm` | 关节控制量向零靠近，修正量仍受限；不是急停 |
| `joint_0_plus/minus` 至 `joint_6_plus/minus` | 在某个关节的整段命令上增加／减少固定小量 |

[`JointActionCandidates`](../../rlinf/models/embodiment/modules/rlt_action_candidates.py) 默认将每一步、每个关节相对于**已裁剪参考动作**的改变量限制在 `radius=0.08` 内，最终命令处于 `[-1,1]`。这是 `pd_joint_delta_pos` 的归一化控制量，不是米、弧度或力。所有候选继承参考夹爪命令，因此这一版不能自主纠正参考动作中的夹爪开合错误。

完全相同的候选会去重并屏蔽，例如参考命令为零时，沿用、减缓和制动可能相同。选择概率只在有效候选间归一化。关节方向不等于末端的空间方向；10 步持续修正可能累积成明显位移，幅度限制不是碰撞安全证明。

[`RLTAtomicPolicy`](../../rlinf/models/embodiment/mlp_policy/rlt_atomic_policy.py) 复用三层 256 维 MLP，输出候选 logits 和概率。初始时有效的参考候选通常得到至少 90% 的概率。训练 rollout 按概率抽取一个候选，评估取最大概率候选；**不会执行候选的概率加权平均**。这可以避免“两个可行命令平均后变成另一个未经评估的命令”。

## 经验通过什么路径积累

每次执行后，原 replay 保存当前观测、实际动作、奖励、结束标记和下一观测。经验沉淀在 critic 与选择器参数中；这一版没有显式长期记忆、接触识别器或世界模型。

critic 继续对实际动作做 TD 更新。实际动作即使来自接管、并不属于当前候选集合，也可以送入连续动作 critic。下一状态的价值改为对有限候选精确求和：

```text
V_next = Σ_k p_online(k | next_obs) × min(Q1_target(next_obs, a_k), Q2_target(...))
y = Σ_i γ^i r_i + (1-done) × γ^n × V_next
L_critic = mean((Q(actual_obs, actual_action) - y)²)
```

这里 `n` 来自 replay 中 reward 向量的真实长度，而不是硬编码为 10。ManiSkill 的 done 包括 termination 和 truncation；保持 baseline 的 terminal 屏蔽语义，避免对 reset 后状态错误 bootstrap。每个候选的 Q 是网络估计，不是额外模拟执行产生的真实标签。

选择器则学习 critic 给出的改进分布。先给所有候选计算代价：

```text
c_k = -w_Q × Q1(obs,a_k) + w_BC × BC_k + λ × |Q1-Q2|
p_target(k) = softmax(log p_prior(k) - c_k / temperature)
L_selector = -Σ_k stop_gradient(p_target(k)) × log p_selector(k)
```

`BC_k` 在普通步骤上对齐参考动作，在有接管标记的步骤上对齐实际执行动作。`w_Q`、`w_BC` 继续使用 baseline 的权重／调度。新增 `atomic_temperature=0.1` 控制改进分布的集中程度，不能解释成成功率；可选 `atomic_disagreement_weight` 默认 0，即不使用双 Q 分歧惩罚。Q 分歧也不是经过校准的风险概率。

更新实现见 [`atomic_decision.py`](../../rlinf/algorithms/rlt/atomic_decision.py)，调用入口在 [`RLTACLossMixin`](../../rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)。critic TD 只更新 critic；选择器的交叉熵只更新 actor backbone 和 selector。目标分布、候选命令及 VLA 输入均不通过这条损失反传。训练时的 reference dropout 只影响 selector 的输入，不会把真实候选或 BC 目标改成零。

worker 在每个更新阶段清除另一方的残留梯度，使全模型梯度裁剪只计算当前更新的参数；该行为仅对新功能启用。CPU 测试覆盖了真实 worker 更新方法，分布式 FSDP 行为仍需 GPU 验证。

这是相对于原 RLT 的**有意算法变化**：有限动作空间、参考先验下的分布改进、选择器蒸馏，以及精确的有限候选 bootstrap。不能称作“只换了动作头，目标函数完全没变”。

## 为什么没有保留最初的加权 Q 损失

第一版尝试直接优化 `Σ p_k c_k`。在 CPU 合成检查中，选择器早期选错后变得过于集中，1000 次更新后仍未恢复；critic 已学好不代表 selector 能纠正。现在改为学习完整的改进分布，让低概率但应当选择的候选仍有明确梯度。

工具保留了 `--actor-update expected-cost` 重现原型对照，而生产代码只使用蒸馏更新。这个检查是具有充分动作覆盖的一步二次奖励任务，不包含碰撞、图像或插销动力学。它能检查学习实现，不能证明机器人样本效率或泛化。

## 从相关工作借鉴了什么

这里采用的是公开机制的组合，是否构成有价值的创新要由受控实验判断。

| 来源 | 借鉴 | 本实现的区别 |
| --- | --- | --- |
| [Jev 官方接口](https://docs.typesafe.ai/introduction) | 把决策表示为有限选项及概率 | 使用本地 RLT 特征和小 MLP，不复现其未公开训练配方 |
| [jev-libero 源码架构](https://github.com/Dimweaker/jev-libero/blob/main/docs/architecture.md) | 明确区分候选生成、选择和执行的责任 | 不读取其几何距离、接触力或模拟器预演结果；其示例视频省略预演/API 等待，不能直接当实时控制对照 |
| [Residual RL](https://arxiv.org/abs/1812.03201) | 利用已有控制先验，只学习修正 | 这里的修正离散、有界，围绕冻结 VLA chunk 构造 |
| [SayCan](https://arxiv.org/abs/2204.01691) | 用可执行能力／价值帮助选择 | 这里选的是细粒度关节修正，不是语言级完整技能 |
| [Q-chunking](https://arxiv.org/abs/2507.07969) | 保留动作分块与相应时域的价值学习 | 当前不变更 chunk 长度，避免把时域变化混入对照 |

本分支与 FLARE、Zeva 分工不同：FLARE 分支研究后果预测，Zeva 分支研究历史交互条件化，这条分支研究有限决策空间。先各自建立对照；只有确认瓶颈后，才考虑让 FLARE 预测候选后果、或让 Zeva 记忆调整候选选择。

## 如何判断值不值得继续

第一轮机器人对照应固定 Stage 1 checkpoint、任务、种子、环境数量、episode 长度、VLA 调用频率、奖励和接管预算，分别比较原连续 RLT 与本分支。500 个控制步的 episode 预算不同于 500 次 learner 更新；同时报告环境步、更新数和墙钟时间。

随后分开消融：仅 reference、均匀候选、学习选择；有／无参考先验；不同 radius／temperature；双 Q 惩罚开／关；固定相同窗口下的连续 residual actor。后者能区分收益来自“限制探索幅度”还是“离散选择”。这些是实验计划，并非都已提供自动运行器。

主要指标包括成功率曲线面积、达到指定成功率所需交互数、人工接管次数与时长、执行偏离、候选使用分布、`atomic/bc_error_floor` 和完整端到端延迟。最小 BC 误差很高说明候选无法表达接管动作；不能把接管强行映射成一个“正确候选 ID”。baseline 接管逻辑会替换部分 reference，因此解读该诊断时必须同时查看接管标记。

候选由模型打分的结果不能拿来充当真实反事实标签。迁移实验需要留出物体／任务、重新报告性能下降，并区分冻结迁移与额外训练；不能把在同一任务中继续训练称作泛化。原 RLT actor 本来就是小 MLP，这条分支不保证更快推理，且训练时额外评分 17 个候选会增加成本。

## 验收与启动边界

在独立目录执行以下 CPU 验收，不会启动训练进程或 Ray：

```bash
cd /home/luokz/rlinf_rlt/UPT_jev_dev
export RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$RLINF_VENV/bin/python" -m pytest tests/unit_tests/test_atomic_decision.py -q
bash run_rlt_atomic_gpu2.sh --check
```

合成学习对照的命令如下。每条默认运行 3 个种子，只使用 CPU；输出 JSON 是验证结果，不是论文任务成功率：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$RLINF_VENV/bin/python" -m toolkits.rlt.atomic_cpu_smoke --steps 1000 --actor-update distill
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$RLINF_VENV/bin/python" -m toolkits.rlt.atomic_cpu_smoke --steps 1000 --actor-update expected-cost
```

只有决定开始 GPU 验证后才执行 `bash run_rlt_atomic_gpu2.sh --probe`，再执行 `bash run_rlt_atomic_gpu2.sh --run`。本次开发没有执行这两条命令。无参数时 launcher 仅预检；`--run` 默认 2 次外层迭代，可通过 `RLT_ATOMIC_STEPS` 设为 1–20。GPU 2 忙碌则拒绝启动，不会杀死已有进程；独立 Ray 端口默认 6512，退出只清理自己创建的子进程。它不能阻止其他不使用同一锁的任务在检查后抢占 GPU，仍应先确认机器调度情况。

配置为 [`maniskill_rlt_stage2_atomic_gpu2.yaml`](../../examples/embodiment/config/maniskill_rlt_stage2_atomic_gpu2.yaml)，2 个训练环境、1 个评估环境、500 控制步，global batch 4、micro batch 2。日志用本地 TensorBoard，位于 `$RLT_STORAGE/runs/atomic_smoke/`；沿用 runner 的 checkpoint 目录结构。VLA 加载 Stage 1 step 2000，selector/critic 从头初始化；baseline 的 Stage 2 actor checkpoint 不能当作新 actor 的续跑权重。

验证记录见 [JEV_VERIFICATION.md](JEV_VERIFICATION.md)。本分支保持独立，没有合并到 baseline、FLARE、Zeva 或 VR。
