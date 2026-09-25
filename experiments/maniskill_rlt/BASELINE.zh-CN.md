# ManiSkill RLT Baseline — 2026-09-25

此快照固定引入未来 latent 预测之前的 ManiSkill RLT 复现代码，作为后续实验的比较基准。目前已验证 Stage 1 持续训练；Stage 2 成功率和最终收敛尚未验证。

## 代码与分支

- 上游代码快照：`b85c07175b10017bf58ab83e1b1eee99666d0626`。
- Baseline：`baseline/maniskill-rlt-2026-09-25`。
- 后续实验：`research/rlt-flare-latent-dynamics`。
- Stage 1：`examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml`。
- Stage 2 AC：`examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`。
- Stage 2 TD3：`examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml`。

运行中的训练目录保留在 baseline 分支，实验在独立 worktree 中开发。不要把算法实验合并回此快照；若需要更新基准实现，应另建带日期的 baseline。

## 当日 Stage 1 运行记录

`run_rlt_stage1_2gpu.sh` 保留当日实际执行的主机专用命令和路径，其中没有内嵌凭据。换机器时需修改路径，并登录正确的 W&B 账号。当日运行沿用了机器上已有的 W&B 登录状态。

| 配置 | 实际值 |
| --- | --- |
| Run | `maniskill_rlt_stage1_2xl40_20260925_163418` |
| GPU | 两张 L40，物理编号 0、1 |
| 每卡 micro batch / global batch | 1 / 256，梯度累积 128 次 |
| 最大 step / 保存间隔 | 2000 / 250 |
| 学习率 / warmup | 2.5e-5 / 500 step |
| Scheduler | cosine，总长度 10000，最低 2.5e-6 |
| 精度 | OpenPI 原生混合参数精度；AMP 关闭 |
| FSDP | no_shard；gradient checkpointing 关闭 |
| 显存分配器 | `expandable_segments:True` |
| Ray | 2 个 GPU 资源，8 GiB object store，临时文件在 /dev/shm，spill 在 NAS |

Scheduler 长度按实际运行记录为 10000，而训练在 2000 step 停止，因此不会走完完整 cosine 衰减。在约 step 200 时进程正常，训练 loss 从约 3.25 降至 0.583。这些数字用于确认训练进展，不能替代独立任务成功率评测。checkpoint、数据、缓存和凭据属于外部产物，不提交 Git；首个预期 checkpoint 为 `global_step_250/actor`。

## 环境与数据

Python 3.11；torch 2.11.0+cu128；Ray 2.58.0；ManiSkill 3.0.0b22；SAPIEN 3.0.1；rlinf-openpi 0.1.1；transformers 4.57.6；W&B 0.25.0。安装入口为 `requirements/install.sh`；这些版本记录本机状态，不是可移植的依赖锁文件。

数据来自 `RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ`，本地名称为 `maniskill_peginsertionside_joint`，使用已经计算的 `norm_stats.json`。基础权重为 `pi05_base`；动作是 8 维 joint delta，Panda 本体状态为 9 维。Stage 2 配置使用 `PegInsertionSideWideClearance-v1`，不要与渲染 smoke test 使用的窄孔 `PegInsertionSide-v1` 混淆。

## Stage 2 待验收事项

启动前需要选择并评测 Stage 1 checkpoint，配置更强的 expert checkpoint 或明确关闭接管，再根据可用 GPU 调整并行环境数。原 Stage 2 具有基于仿真状态的阶段切换和可选 expert 接管；减少干预的比较必须统计这两类帮助。此快照不宣称 Stage 2 已验证，分支整理期间没有启动新训练。

参见 [English record](BASELINE.md) 和仓库已有的中英文 RLT 文档。
