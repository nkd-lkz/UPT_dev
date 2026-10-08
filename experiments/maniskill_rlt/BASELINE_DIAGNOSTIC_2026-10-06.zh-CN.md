# 诊断 RLT 的模仿误差与 Q 引导偏移

这份协议说明怎样先用同一批缓存数据比较 BC-only 与 Q+BC，再做冻结参数的仿真评估。CPU 诊断复用正式小网络和损失；实际进度与结果以 campaign manifest 为准。英文对应页为 [baseline diagnostic](BASELINE_DIAGNOSTIC_2026-10-06.md)。

10 月 6 日队列已等待超时。[10 月 8 日接续协议](CONTACT_FACTORIAL_2026-10-08.zh-CN.md)重新启动冻结复评，并把接触阶段变化和驱动条件变化分开诊断。下文历史预算对应此前启动记录。

## 为什么先做这个诊断

10-04 冻结复评中，零 context 为 16/64，响应记忆为 15/64，reference 为 19/64。只有 26/64 回合进入 actor 阶段。局部记忆不能直接修复此前的失败。本轮区分模仿误差与 actor 的 Q 项影响，不预设记忆收益。

## 固定开发数据

旧 checkpoint 的 replay 索引多于实际保存文件。下面的命令只读取存在且在索引中的文件，记录缺失数量和哈希，并保留原 checkpoint。非有限值、未记录的 transition、冲突终止标志及干预数据会被拒绝。

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$RLINF_VENV/bin/python" \
  toolkits/rlt/diagnose_actor_learning.py prepare \
  --replay "$SOURCE_REPLAY" --output "$CAMPAIGN/cache" \
  --limit 2048 --split-seed 601
```

`SOURCE_REPLAY` 指向零 context 第 275 轮的 `actor/sac_components/replay_buffer/rank_0`。`CAMPAIGN` 使用新的输出目录，`RLINF_VENV` 指向现有兼容环境。缓存包含当前／后继冻结特征、reference、实际动作、奖励、结束标志及当时的历史；不运行 VLA 或 simulator。

按采集模型版本分组，约五分之一版本用于验证，相邻动作块不独立拆分。这是开发集划分，不代表未见物理条件。旧 checkpoint 已接触这些数据；只有新初始化的 head 才拥有保留验证集。

## 比较同预算学习

两组使用零 context 的 resolved config，从同一 seed 重新初始化。网络、数据、采样顺序、microbatch、critic／actor 更新次数一致。两组都训练 critic；BC-only 仅把 actor 的 Q 系数置零，保留 BC 与 reference dropout。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 "$RLINF_VENV/bin/python" \
  toolkits/rlt/diagnose_actor_learning.py train \
  --cache "$CAMPAIGN/cache" --config "$CAMPAIGN/source-config.yaml" \
  --output "$CAMPAIGN/training" --steps 2048 --seed 1234 --threads 2
```

batch 为 32，microbatch 为 8，每次 critic 更新都执行一次 actor 更新。每组执行 2,048 次 critic／2,048 次 actor 更新。实际 pilot 已把底层默认的 4:1 比例覆盖为 1:1。前 512 次 critic 更新的 Q 系数为零，之后 512 次线性增加到 0.05；BC 系数保持 7.0。公共零 Q 热身期的两组权重应完全一致。

工具直接调用 `RLTACLossMixin` 的正式损失，只跳过 worker 计时装饰器。保留参数归属、Adam 默认值、随机动作采样、microbatch 顺序、全模型梯度裁剪与 target EMA。不复现 FSDP、增长中的 replay 或旧 optimizer 状态。

指标包括确定性 reference MSE／MAE、动作饱和比例、双 Q 分歧及同一状态的 Q1 动作排序。Q 排序不是反事实真实回报。每 256 次 critic 更新保存诊断记录；提前固定第 2048 步用于闭环评估，不在最终 seed 上选最佳 checkpoint。

## 等 GPU 可用后评估控制

两个有界队列各等待空闲 GPU 最多 12 小时，累计评估运行最多 4 小时。`wait_for_gpu.py` 同时检查低显存与无计算进程；portable launcher 随后取项目锁并再次检查显存。不停止已有进程；资源竞争时启动失败退出。

GPU 0 评估 BC-only 与 reference-only。GPU 1 评估 Q+BC 与旧零 context checkpoint。固定 4101–4104 四个 seed，各 16 个 lane、500 tick，每组 64 回合。无 expert、零 context、冻结权重。BC-only／Q+BC 是匹配训练对照；旧 checkpoint 训练历史及预算不同，仅作为描述性参照。

campaign launcher 显式设置 `RLT_MEMORY_READER=zero`、`RLT_EVAL_CHECKPOINT`、`RLT_EVAL_VARIANT`、`RLT_EVAL_SEED`、`RLT_EVAL_ENVS=16`、`RLT_EPISODE_STEPS=500`，调用 `run_rlt_portable.sh --memory --eval`。preflight 不算 GPU 运行；设备占用时报告等待。

## 根据结果决定下一步

若 Q+BC 增大模仿误差，先看是否影响真实成功率，再调整 Q 系数。若两组都学不好，检查输入、dropout 和数据覆盖。reference 在 gate 前失败，单独定位。最终 seed 不参与 checkpoint 选择。

baseline 学习验收后，再测正确／无／匹配错误历史；随后做条件 A→B→A，比较清空、保留、时间衰减与后果误差降权；最后做记忆 × 已通过恢复验收的固定 expert。这些后续实验不自动启动。报告逐次尝试的无辅助成功率与纠错时长。

CPU 测试覆盖数据拒绝、版本分组、缺失文件计数、原索引不变、相同初始化与采样、零 Q 热身一致及启用 Q 后的差异。GPU 检查拒绝被占用或无法读取的设备；设备占用时仿真验证仍待执行。
## 10-06 晚间续跑与经验诊断

本节说明 baseline 之后怎样继续验证历史信息，实际进度以 campaign 状态文件为准。两卡等待延长为 24 小时，每张卡实际执行总预算为 6 小时。`toolkits/rlt/run_research_queue.py` 保存逐任务状态、checkpoint 哈希和退出码。已完成且复验通过的任务可跳过；失败或被中断的任务必须先检查再重试。新队列采用固定提交的独立 checkout，保留原 baseline 版本；旧归档缺少 Git 元数据，而启动器在预检后会调用 `git rev-parse`。

16 个 baseline run 全部通过回合、权重、路由与配对初态检查后，两卡先分别执行小型仿真 smoke，再运行响应诊断。这些检查只说明数据有效，不证明 baseline 收敛，也不会自动触发新 RL 或 expert 训练。

| GPU | 先执行 | baseline 与 smoke 验收后 |
|---|---|---|
| 0 | BC-only、reference-only，各四个 seed | 56 对响应试验，每个条件四次查询 |
| 1 | Q+BC、旧零 context head，各四个 seed | 六对稳定条件／A→B→A 响应流 |

响应试验在两种机械臂驱动刚度下分别采集八个真实的 10 tick 动作块，再保留标定历史、复位到相同种子的查询场景。这是使用仿真复位能力的特权诊断，不能当作正常部署流程。两条件的完整仿真状态哈希、当前 qpos、速度和查询命令必须一致。错误历史来自配对条件，历史命令和有效槽数量保持匹配；隐藏刚度标签不进入预测器。按 pair 0–31／32–39／40–55 划分训练、验证与测试。三个 seed 分别训练同容量无历史／固定响应头，各 512 次更新；初始化和采样匹配，只报告预定最后一步。工程 smoke 只用两对数据，不训练模型。

`rlinf/algorithms/rlt/response_context.py` 实现四种诊断读取规则：每次明确的尝试起点清空、始终保留、时间衰减、按最近四条已完成响应计算误差权重。容量均为 32。误差权重可以恢复，但近期一致性不是校准后的可信度。该模块尚未接入正式 actor。真实 A→B→A 与稳定条件使用相同尝试边界；条件编号和未来后果都不传给 reader。

首轮合成测试采用线性系统、六个 seed 和固定阈值，不属于机器人仿真。恢复到 A 后，前四块 MSE 为保留 0.000389、误差降权 0.000696。这个反例不支持直接把降权升级到 RL；仿真诊断用于检验适用范围。

在已有环境中，用新输出目录运行 CPU 诊断：

```bash
CUDA_VISIBLE_DEVICES='' "$RLINF_VENV/bin/python" -m toolkits.rlt.probe_memory_conditions synthetic --output "$CAMPAIGN/synthetic"
```

物理 GPU 空闲后，用同一模块执行 `matched --gpu 0 --output "$CAMPAIGN/matched"` 或 `shift --gpu 1 --output "$CAMPAIGN/shift"`；去掉 CPU 示例中的 `CUDA_VISIBLE_DEVICES=''`，由工具隔离所选 GPU。增加 `--smoke` 先做接口验收。工具取得项目 GPU 锁，并在创建仿真前再次检查显存和计算进程。指标单位是原始关节位移平方误差；这些诊断都不测量任务成功率、纠错节省或迁移收益。
