# 在 GPU 2 验证 Baseline Stage 2

这份说明帮助你在 Stage 1 继续使用 GPU 0/1 时，用物理 GPU 2 验证 Stage 2 的权重加载、仿真采样、replay、actor–critic 更新、评估和 checkpoint 保存。先执行路径与配置预检，再通过 tmux 启动，最后依据训练日志判断链路是否通过。这是工程 smoke，不是正式训练，也不能用于判断收敛或泛化。

## 配置与权重

独立配置是 `examples/embodiment/config/maniskill_rlt_stage2_smoke_gpu2.yaml`，启动入口是仓库根目录的 `run_rlt_stage2_smoke_gpu2.sh`。它们不会改写原有 Stage 1、Stage 2 配置，也不包含 FLARE 实验模块。

默认固定读取 `runs/stage1/maniskill_rlt_stage1_2xl40_20260925_163418/checkpoints/global_step_750/actor/model_state_dict/full_weights.pt`，相对目录根为 `/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill`。step 750 是已落盘的 checkpoint，并非当前训练进度。归一化文件使用同一根目录下的 `datasets/lerobot/maniskill_peginsertionside_joint/norm_stats.json`，不会重新计算。

Stage 1 权重送入冻结的 `rollout.rlt_feature_model`；Stage 2 的小型 MLP actor 和 twin-Q 从头初始化，不能把 Stage 1 的权重传给 `actor.model.model_path`。预算如下：

| 项目 | Smoke 设置 |
| --- | --- |
| GPU | `CUDA_VISIBLE_DEVICES=2`；RLinf 硬件 rank `2-2`；worker 内 CUDA ordinal 为 0；渲染按 GPU 2 的 PCI 地址绑定 |
| 训练 / 评估环境数 | 2 / 1 |
| 每轮控制步 / action chunk | 40 / 10 |
| 外层迭代数 | 2，评估和保存间隔均为 1 |
| global / micro batch | 4 / 2，单卡累积 2 个 micro batch |
| replay 最小样本数 | 4；首轮包含至多 8 个 chunk transition |
| 更新预算 | 初始 warmup 2 次；每轮最多 2 次；actor/critic 更新比 1:1 |
| 日志 | 本地 TensorBoard 和文本日志 |

为了不依赖早期 checkpoint 已经学会抓取，smoke 使用 `trigger_mode: always_on`，强制进入关键控制阶段。首轮仍由 baseline 的 warmup 逻辑选择参考动作，完成更新后可切换到 MLP actor。expert 接管关闭，`rollout.expert_model: null` 避免额外加载一套 OpenPI。传感器仍为 384×384，动作与本体维度沿用 baseline；这些设置验证工程链路，不构成正式任务的评估协议。成功率为 0 并不等于 smoke 失败。

## 预检与启动

先验证 checkpoint、归一化文件、Vulkan 文件和 Hydra 配置。预检只读取文件，不启动 Ray、不分配 GPU，也不创建输出目录：

```bash
cd /home/luokz/rlinf_rlt/UPT_dev
bash run_rlt_stage2_smoke_gpu2.sh --check
```

出现 `Preflight OK` 后，在 tmux 中启动实际测试：

```bash
tmux new-session -s rlt_stage2_smoke
cd /home/luokz/rlinf_rlt/UPT_dev
bash run_rlt_stage2_smoke_gpu2.sh
```

脚本自动激活 `.venv` 并配置已验证过的 headless Vulkan 路径；按 `Ctrl-b`，再按 `d` 可退出 tmux 界面，测试继续运行。用 `tmux attach -t rlt_stage2_smoke` 返回会话。脚本检查 GPU 2 已用显存不超过 1024 MiB，否则拒绝启动；这不是对其他进程的 GPU 预留，请勿同时在 GPU 2 启动其他任务。

Ray 默认使用 GCS 6382、client 6383、dashboard 6384；dashboard 绑定 127.0.0.1，agent 和 worker 端口自动分配，避免与 Stage 1 的默认端口冲突。CPU 资源额度为 8，对象存储为 2 GiB。Ray socket 和临时目录位于独立的 `/dev/shm/rlt2.*`，对象溢写及训练产物在 NAS。显式 `RAY_ADDRESS` 指向本次创建的 head，避免加入 Stage 1 的 Ray。退出时只向本脚本的训练进程和 Ray head 发送信号；不要执行全局 `ray stop`、`pkill ray`。CPU、内存带宽和 NAS 吞吐仍与 Stage 1 共享，Stage 1 可能出现少量速度波动。

如需另一个已完整保存的 checkpoint，启动前设置 actor 目录；脚本不会自动追踪最新目录：

```bash
export RLT_STAGE1_ACTOR=/absolute/path/to/checkpoints/global_step_1000/actor
bash run_rlt_stage2_smoke_gpu2.sh --check
```

上面的路径是占位示例，需要替换成已存在的目录。端口占用时可设置 `RLT_SMOKE_RAY_PORT=6390`，同时预留其后的两个端口；启动器拒绝占用 Stage 1 原端口 6379 及恢复端口 6385–6387。`RLT_STORAGE`、`RLT_DATASET_DIR` 和 `RLINF_VULKAN_PREFIX` 也支持环境变量覆盖。

## 日志与通过标准

启动时打印实际输出目录，形式为 `$RLT_STORAGE/runs/stage2_smoke/stage2_gpu2_<时间>_<PID>/`。该目录保存 `train.log`、`ray-head.log`、`resolved-config.yaml`、`tensorboard/`，小型 actor–critic checkpoint 保存在 `stage2_smoke/checkpoints/global_step_<N>/actor/`。Ray 临时日志保留在启动时打印的 `/dev/shm/rlt2.*`，便于排查失败；脚本不会自动删除诊断数据。

在另一个终端查看脚本打印的真实路径：

```bash
tail -f /absolute/path/to/the/printed/run/directory/train.log
```

通过标准是：两轮完成且退出码为 0；replay 确实收到样本；`train/rlt/critic_updates_run` 和 `train/rlt/actor_updates_run` 为正；loss 为有限数；评估完成；至少一份 Stage 2 checkpoint 写入。进程存活、GPU 占用上升或仅有评估输出，都不能单独证明通过。W&B 默认不启用，避免账户权限问题影响测试；指标可以从 TensorBoard 读取。

GPU placement 探针已通过；完整仿真和训练的实际结果在取得后记录于下方。smoke 通过后再用正式 baseline 配置恢复 `auto` 阶段切换、正式预算及评估协议，并选择质量合适的 Stage 1 checkpoint；不要把 smoke 配置直接扩展为收敛实验。

## GPU 隔离与恢复记录

2026-09-26 的首次 GPU smoke 暴露了原配置错误：RLinf 独立枚举三张物理卡，`0-0` 不会自动映射到外层 `CUDA_VISIBLE_DEVICES=2`。worker 实际进入 GPU 0，引发 smoke 与 Stage 1 OOM。Stage 1 最后显示 889 步，最近保存点为 750。另一个错误来自关闭 dashboard 后异常清理无法调用 State API。CPU 配置检查不足以验证 GPU 隔离。

修复后的启动器先解析三个组件的实际 placement，再启动一个真实 RLinf worker，检查可见设备为 `2` 且 CUDA UUID 与物理 GPU 2 相同，只分配一个标量。所有训练启动都强制执行此检查；也可单独运行：

```bash
bash run_rlt_stage2_smoke_gpu2.sh --probe
```

只有出现 `GPU_PROBE_OK` 并以 0 退出才算隔离通过。2026-09-26 已完成该实测，GPU 2 UUID 为 `4662787b-485a-0e8f-e4b2-dd47352ed69c`。这项检查不能代替完整仿真和训练 smoke。

Stage 1 使用 `run_rlt_stage1_resume750.sh` 从原 step 750 恢复，保存到新的时间戳目录；Ray 端口为 6385–6387。恢复读取 DCP 的模型、optimizer 和学习率调度状态。OpenPI 数据迭代器没有独立保存于该 checkpoint，因此不声称与中断前的数据顺序逐位一致。运行前会拒绝 GPU 0/1 已被占用的情况，不要重复启动恢复任务。

首次恢复在将 Adam 状态复制到 CUDA 时 OOM，尚未推进新训练步。DCP 为建立加载模板，先初始化了 GPU optimizer 状态；`Optimizer.load_state_dict` 在释放这些状态之前又分配了替换状态。现在 `Checkpoint.load_state_dict` 在 `cpu_offload=True` 时先将旧 optimizer 状态移至 CPU，再装载 checkpoint 中的状态。保留已初始化的条目，而不是清空它们，避免触发另一次临时 optimizer 初始化。模型权重、Adam 动量及计数、scheduler 和已保存的 RNG 状态仍按 checkpoint 恢复。这项修复只改变恢复期间的内存放置，不改变训练算法或 batch size；`local_shard` 路径保持不变。

`tests/unit_tests/test_checkpoint.py` 覆盖单个及多个 optimizer 的保存/恢复、CPU 上下一步更新逐值一致性、CUDA 恢复显存峰值，以及真实 FSDP `NO_SHARD` 的 DCP 磁盘往返和后续更新。2026-09-26 在 GPU 2 上运行，4 项测试通过；仅 CPU 执行时跳过两项 CUDA 测试。恢复启动器核验个人账号 `c6522513`，固定 entity 为 `c6522513-sustech`；HTTPS 默认使用用户已有的本机 17891 端口代理，并让本地 Ray 地址绕过代理。启动时需保持代理可用，或事先提供可用的 `HTTPS_PROXY`。这些服务器专用设置仅位于启动器中，不写入通用模型配置。

## 2026-09-26 实测结果

Stage 1 从 750 恢复后，在 GPU 0/1 上完成了 751–754 步，loss 和梯度均为有限值。新 run 位于存储根目录下的 `runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016`，tmux 窗口为 `rlt_stage1_resume750_fixed:0`，W&B run 为 `c6522513-sustech/rlinf-rlt/8oblhtio`。第 751 步 loss 为 0.447，第 754 步为 0.449。下一次计划保存 checkpoint 是第 1000 步；本次验证没有等待该保存点或训练收敛。

确认第 751 步完成后，才在 tmux `rlt_stage2_smoke_verified` 启动 GPU 2 的完整 smoke。`runs/stage2_smoke/stage2_gpu2_20260926_154350_2095986` 完成两轮并以 0 退出。TensorBoard 中的实测指标如下：

| 指标 | 第 1 轮 | 第 2 轮 |
| --- | --- | --- |
| Actor / critic 更新次数 | 2 / 2 | 2 / 2 |
| Replay 样本数 | 6 | 12 |
| Actor loss | 0.430630 | 0.711137 |
| Critic loss | 0.116520 | 0.010061 |
| 评估成功率 | 0 | 0 |

所有已记录 scalar 都是有限数。两次评估完成；`global_step_1` 和 `global_step_2` 均包含非空的 `actor/model_state_dict/full_weights.pt`、DCP metadata 和 DCP shard。实测三个 smoke 组件都位于物理 GPU 2。清理 smoke 后，GPU 上仅剩两个 Stage 1 进程，Stage 1 已推进到 754。这证明工程链路通过，不代表任务能力已收敛；本次未启用 expert 接管。TensorBoard 的 step 从 0 计数（0/1），checkpoint 目录从 1 计数（1/2）。

此外，Ruff、shell 语法检查、37 项 CPU 测试和中英文 Sphinx 构建均通过，构建警告为 0。CPU 测试跳过 2 项 CUDA 测试，排除 2 项会启动 Ray 的无关测试。仓库全局文档扫描仍报告未修改 RST 中已有的 61 项 inline markup 警告和 26 个未解析名称，不在本次恢复修复范围内。
