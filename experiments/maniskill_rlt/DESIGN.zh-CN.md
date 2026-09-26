# 用动作后果表征改进 RLT

这个方案检验：在冻结的 RLT 特征空间里学习交互后果，能否提高在线 actor–critic 的样本效率。它以官方 ManiSkill 复现为起点，增加轻量离线适配阶段，再把同一模块接入真实 transition 的在线学习。[小规模实验记录](PILOT_RESULTS.zh-CN.md) 新增了增量预测、架构图与真实数据/GPU 证据，尚未证明在线 RL 改善、物理规律或完整持续学习能力。[English](DESIGN.md)

## 1. 先回答一个可以验证的问题

主要假设是：在示范、冻结 VLA、环境交互预算、奖励、阶段切换规则和评估协议相同的条件下，动作条件化的后果表征能让 RLT 用更少的环境控制步达到指定成功率。机制是监督“发出的动作实际改变了什么”，而不是另造一个奖励函数。

第一个实验仍然是 joint-control peg insertion。接触恢复、减少干预、跨任务迁移是后续假设，不能从辅助 loss 下降直接推出。暂不同时加入显式知识库检索和自动阶段切换，以便分辨单一机制是否有效。

## 2. 借鉴了 FLARE 的什么，又改了什么

[FLARE 论文](https://arxiv.org/abs/2505.15659)及其[项目页面](https://research.nvidia.com/labs/gear/flare/)启发我们在学习动作时预测未来观测的表征。原方法把可学习的 future tokens 放进 diffusion/flow transformer，让它们与 action tokens 交互，并把中间层的未来特征对齐到未来视觉语言 embedding。原论文还包含 action-aware embedding 预训练和 EMA target 的训练设计。

这里是 **FLARE 启发的扩展，不是 FLARE 原样复现**。代码没有在 Pi0 action denoiser 内插入 future tokens，也没有复现其架构、数据与论文结果。我们使用一个独立小型 transformer，在现有的冻结 RLT 坐标系中预测动作后果，使 Stage 2 可以直接使用相同监督，不必每次 replay 更新都重新训练数十亿参数的视觉编码器。相对于其他 latent-dynamics RL 工作是否具备充分新颖性，仍需进一步文献检索，不能现在保证。

| 设计选择 | 本分支实现 | 原因与限制 |
| --- | --- | --- |
| 视觉骨干 | 完成 Stage 1A 后冻结 | 缓存与 replay 坐标稳定；无法找回原特征没有保留的接触信息 |
| 未来目标 | 冻结 RL token 与归一化本体状态 | 复用 Stage 2 输入，不生成未来图像 |
| 动作条件 | 实际执行的动作前缀与 horizon query | 预测以动作作为条件，不只依赖时间 |
| 动作相关性 | 当前 adapter 特征的辅助 chunk BC | 防止目标只关注视觉；不能保证控制相关性 |
| 在线更新 | TD 加真实 replay 的后果监督 | 不生成虚构 transition，不替换奖励 |
| 经验保留 | 模型参数、checkpoint 和离线缓存 | 尚不是可检索的长期经验库 |

## 3. 把 Stage 1 明确分成两个子阶段

Stage 1A 保持官方 OpenPI RLT SFT，包括原有 VLA 与 token 重建目标。当前正在运行的 baseline 不受影响。Stage 1B 是新增的小型 adapter 训练，需要一个已经完整保存、后续不再修改的 Stage 1A checkpoint。不能把这种分阶段适配描述成端到端修改 Pi0。

完整数据流如下：

```text
示范 RGB + 语言 + 关节状态
  -> 冻结 Stage 1A / 与 Stage 2 相同的预处理
  -> episode 缓存：z_t、p_t、a_t、frame 和 task ID
  -> Stage 1B：当前状态 adapter + 动作前缀 transformer
  -> sidecar checkpoint，附带权重、预处理、控制单位的哈希
  -> Stage 2：扩展 actor/critic 输入，并用真实 replay 更新后果预测
```

缓存工具调用 `Pi0Eval.extract_rlt_obs(include_reference=False)`，只跳过缓存不需要的参考动作解码，prefix 提取和本体状态归一化仍与 rollout 共用。该参数默认仍为 `True`，保留原推理行为和随机数消耗顺序。图像、数据和原始权重不提交到 Git。

当前 LeRobot v2.1 数据集包含 400 个 episode、28,681 帧、主视角与腕部 RGB、9 维 state、8 维 action，频率为 10 Hz。它没有显式接触、力或摩擦系数标注。动作必须使用原始 `pd_joint_delta_pos` 环境命令，范围为 [-1, 1]，不能用 SFT 内部归一化后的 action 替代；本体状态则已经过冻结 OpenPI transform 的归一化。

## 4. 模块具体学习什么

记 `z_t` 为 2048 维冻结 RL token，`p_t` 为 9 维归一化本体状态，`A[t:t+h]` 为 8 维实际动作序列。`LN` 表示不带可学习参数的逐 token layer normalization。两层 MLP 生成默认 128 维的 `e_t = encoder([LN(z_t), p_t])`。

对于 `h ∈ {1, 5, 10}`，两层、四个 attention heads 的 transformer 读取 `[e_t, action_0 + position_0, ..., action_(h-1) + position_(h-1), query_h]`。每个 horizon 单独计算，因此一步预测不能看到后续动作。三个独立初始化的预测 head 输出 `pred_z_h` 与 `pred_delta_p_h`。它们共享主干，不是三个完全独立的世界模型。

默认目标为：

```text
L_h = masked_mean(
    1 - cosine(pred_z_h, stopgrad(LN(z_(t+h))))
    + 0.1 * MSE(pred_z_h, stopgrad(LN(z_(t+h))))
    + 0.5 * SmoothL1(pred_delta_p_h, stopgrad(p_(t+h) - p_t))
)
L_offline = future_weight * mean_h(L_h)
            + 0.1 * MSE(anchor(e_t), stopgrad(LN(z_t)))
            + 0.1 * masked_MSE(tanh(behavior(e_t)), demonstrated_chunk)
```

mask 按有效数量归一化，分母最小为 1。每个 ensemble head 使用概率为 0.8 的独立 bootstrap mask；第一个 head 始终使用所有有效样本。评估关闭 bootstrap mask。在线 minibatch 全部终止时，辅助项返回可反向传播的零损失。

anchor 保留当前状态信息，BC 让 adapter 与动作相关。固定且非恒定的 target 能降低坍塌风险，但不能证明模型真正使用了动作：场景变化小时，“未来等于现在”可能已经很好。因此评估工具同时报告保持当前特征不变的预测、打乱动作的预测和各 horizon 的本体状态误差。最终依据必须是控制改善，不只是训练 loss。

本体运动响应预测是当前物理交互信号的切入口，可以反映约束下动作产生的变化。但遮挡、夹爪柔顺性和接触状态未必能从 `z_t,p_t` 观察到。代码不会把“移动慢”伪造为接触标签，也没有学习受力公式的结论。未来可以单独加入仿真真实接触或力监督，作为 privileged-information 消融；推理仍必须使用声明过的传感输入。

## 5. 怎样送入 Stage 2 在线迭代

sidecar 扩展而不替换原特征。actor 使用 `[reference_chunk, z_t, p_t, stopgrad(e_t)]`；critic 使用 `[z_t, p_t, e_t]` 和候选动作。原始信息仍然保留，避免强制依赖一个可能无效的压缩表示。辅助 BC head 不复制给 Stage 2 actor，否则 BC warm-start 会干扰“未来表征是否有效”的比较。

Stage 2 保留 RLT 原目标，只增加：

```text
L_critic_total = L_original_RLT_TD + 0.1 * L_online_world
L_online_world = L_H + 0.1 * L_anchor       # 不加在线辅助 BC
L_actor = 原 RLT 的 Q/BC 目标               # 保留原有干预目标
```

`future_weight` 同样作用于在线 future 项。critic optimizer 管理 sidecar 的全部参数，包括 query 和预测 head；actor 读取 detach 后的 adapter 特征。TD 梯度可以调整 adapter，辅助梯度继续训练后果预测。`target_update_type` 必须为 `all`，确保 target critic 的 adapter 同步进行 Polyak 更新。sidecar 是正常注册的 policy 子模块，会进入现有 checkpoint 和权重同步路径；单 GPU 多进程 FSDP/Ray smoke 与续跑已通过，多 GPU 扩展仍待验证。

当前 ManiSkill collector 在执行完整 action chunk 后给出 successor observation。`H=10`、控制频率 10 Hz 时，在线预测跨度是 **1 秒，不是 0.1 秒**。`replay_world_batch` 检查 reward slots 和动作数量是否匹配 H，使用真正路由/执行的动作，包括 expert 替换，不使用尚未执行的 student proposal。任何 termination 或 truncation 都会屏蔽整行：RLT 的终止 transition 可能拿当前观测代替 next_obs，不能把它当作“动作没有改变状态”的物理证据。

现有 replay schedule 只记录指定关键阶段，不是全任务 world-model 更新。在线主要训练 H，可能使共享主干遗忘较短 horizon；应保留离线诊断并测量遗忘。冻结特征坐标解决的是旧 replay 编码失效，不是所有灾难性遗忘。

## 6. 可选的探索范围约束

主实验默认保留原动作参数化。可选 `bounded_residual` 模式为：

```text
center = clip(reference_chunk, -1, 1)
radius = configured_radius                             # 固定范围消融
radius = max(min_radius, radius / (1 + variance/scale)) # 可选启发式
action = clip(center + radius * tanh(actor_sample), -1, 1)
```

variance 来自 ensemble 对参考 action chunk 未来的分歧，计算时不传播梯度。scale 为 0 表示关闭不确定性调整，不会除以零。默认 radius 为 0.2、下限 0.02，调整关闭；主配置也没有启用 bounded residual。

这种方式限制相对 base action 的偏离，不代表安全认证，也不是接触检测器。共享主干可能一致地预测错误；参考动作不佳时，太小的范围可能阻止恢复。因此先比较固定范围，再判断不确定性是否提供额外收益。诊断工具报告误差与分歧相关性、p95 尺度候选，不会自动完成校准或开启约束；在线分布变化后仍需重新校准。

裁剪改变了动作分布。代码只为接口兼容保留 baseline 的 pre-tanh logprobs，并禁止本研究路径使用非零 entropy。不能未经正确的分布推导，直接移用于带 entropy 的 SAC 或 PPO。

## 7. 数据来源、划分与训练参数

cache manifest 绑定 checkpoint 内容、norm stats 内容、解析后的特征配置、controller 和频率，并记录各 episode 的内容哈希。缓存只对相同来源续跑；缺帧、不连续 timestamp 会报错。全部 episode 完成后才发布 complete manifest。checkpoint 和 manifest 使用同目录临时文件加原子重命名，不要求 NAS 支持符号链接。

整个 episode ID 通过固定种子的哈希划入 90/10 分组。窗口不能跨 episode；不存在的 future/action 被 mask。这只是 adapter 验证集，不是独立 policy 测试集：当前 Stage 1A 可能已经看过全部 400 个 episode。完整泛化结论需要在 Stage 1A 前就定义独立 demonstration/task split，并另外使用未见过的环境种子。

Stage 1B 初始参数：小模型 fp32，全局 batch 256，microbatch 32，即累积 8 次；AdamW 学习率 3e-4，weight decay 0.01，warmup 500 步，cosine 衰减到峰值的 10%，最多 10,000 次 optimizer update。每 250 步验证，连续 12 次验证无改善则早停。这些是起始设置，不是收敛保证。`best.pt` 按辅助验证损失选择，还要检查动作敏感性与实际控制表现。`last.pt` 保存 optimizer、scheduler、CPU/CUDA RNG、sampler、配置、划分和 cache manifest，以便相同配置精确续跑。

Stage 1B 使用单进程、单 GPU，缓存特征上的小模型不需要 DDP。研究 Stage 2 使用两张可见 GPU：actor 在 0，rollout/仿真在 1；训练和评估各 8 个并行环境，actor 全局 batch 256、microbatch 64。schedule、奖励和切换阈值继承冻结 baseline。新路径的这些资源参数尚未经过显存或吞吐实测。

## 8. 怎样证明收益来自所设计的机制

先做廉价诊断，再投入在线 RL：检查打乱动作是否降低预测质量，比较 persistence，观察特征方差与各 horizon 的误差。不能被大量静止帧主导的低平均误差误导。

| 实验组 | Stage 1B | 在线后果损失 | 探索 |
| --- | --- | --- | --- |
| A | 无 sidecar，资源匹配的原 RLT | 关闭 | 原动作形式 |
| B | 同等规模 adapter，`future_weight=0`，只保留 anchor + BC | 关闭 | 原动作形式 |
| C | 完整未来预训练 | 关闭；TD 仍会更新 encoder | 原动作形式 |
| D | 完整未来预训练 | 开启 | 原动作形式 |
| E | D | 开启 | 固定 bounded residual |
| F | D | 开启 | bounded residual 加经过验证的 uncertainty scale |

C 不是“冻结 adapter”：关闭 `latent_world_weight` 只关闭 replay 自监督，TD 梯度仍然存在。B 控制额外容量和 BC 数据利用。A/B/C/D 必须使用相同 Stage 1A、示范、环境种子、预算与干预规则。E/F 改变了动作空间中的搜索范围，需要单独比较。提供的 matched-baseline 配置保留研究资源设置但去掉 sidecar。由于不假定已有更强 expert checkpoint，两份配置都关闭 expert takeover，因此不能用它们证明“减少了干预”。

初步实验至少使用 3 组配对随机种子，正式报告最好 5 组。主要指标为 success-vs-control-steps 的 AUC，以及达到预先约定且持续满足的成功率所需控制步，例如连续 3 次评估达到 80%。应计入所有环境控制步，包括 base/expert 控制阶段，不只计关键阶段 replay 行数。未达到阈值的 seed 应作为受限观测/失败报告，不能剔除。辅助报告 wall-clock、gradient updates、显存峰值、推理延迟、明确记录的碰撞和未见扰动下的恢复。评估需有足够独立 episode 并提供不确定性区间；8 个并行评估环境只表示并发数，不代表足够样本量。

研究干预时，各组开启完全相同的 expert checkpoint 与 takeover 规则，统计每 episode 和每成功 episode 的接管事件数、接管控制步，并保留无接管的成功率评估。不能通过放宽触发门槛或接受更多失败来“降低干预”。当前 expert 是仿真中的较强 policy，不一定是人类。

## 9. 从单任务到经验积累

先在 peg insertion 内测试接触敏感变化：未见孔隙大小、初始姿态偏差、接近误差，以及受控物理参数变化。区分分布内学习与扰动鲁棒性。仿真特权变量可以划分评估条件，但不能悄悄作为 actor 输入。

之后，在 embodiment、动作语义和观测一致的多个任务上预训练，做顺序学习，同时评估新任务样本效率与旧任务保持。held-out task 必须是真正不同任务，不能只是同一任务的新种子。缓存保留 task ID 以便划分分析，但本版没有实现 task-balanced sampler、通用多任务 gate 或 task-specific heads。ManiSkill RLT 当前针对 peg 的关键阶段和接管规则，不能只改 environment ID 就迁移。

未来显式记忆可以存 `(feature-version, task, context, executed action, outcome, error, confidence, recovery)`，比较无记忆、近期历史、固定档案和持续更新档案。它还需要检索延迟预算、冲突处理、证据阈值和独立后续尝试。该扩展尚未实现；目前的持久变化在模型参数及标准 replay/checkpoint 中，默认 replay 有限，也不会自动保留无限经验档案。

## 10. 验收门槛与局限

代码验收应先通过：baseline 关闭兼容、episode 边界、未来 mask、target stop-grad、chunk 时间对齐、终止 mask、optimizer 参数归属、CPU 确定性续跑及配置一致性测试。GPU 2 缓存提取与短程 Stage 2 保存/恢复已记录在 [小规模实验结果](PILOT_RESULTS.zh-CN.md)。单元测试不能替代这些集成验证，收敛优势仍需受控实验。

如果预测对动作不敏感、差于 persistence、只改善 teacher loss，或只减少更新次数却不改善交互预算下的成功率，应停止或修订假设。无干预实验不能证明干预减少；对应实验完成前，不声称物理理解、跨任务泛化、终身成长或保证收敛。

明天需要的科研判断：是否接受 Stage 1A/1B 的拆分，还是随后投入 in-DiT 修改；选定主要成功率阈值与 held-out 扰动；选择 expert/无 expert 协议；决定真实接触监督和显式记忆档案中哪一个作为下一项独立变量。这些决策不影响现在检查代码和运行单元测试。
