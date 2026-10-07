# 诊断 RLT 的模仿误差与 Q 引导偏移

这份协议说明怎样先用同一批缓存数据比较 BC-only 与 Q+BC，再做冻结参数的仿真评估。CPU 诊断复用正式小网络和损失；实际进度与结果以 campaign manifest 为准。英文对应页为 [baseline diagnostic](BASELINE_DIAGNOSTIC_2026-10-06.md)。

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
