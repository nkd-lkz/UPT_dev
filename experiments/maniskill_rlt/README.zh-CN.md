# ManiSkill RLT baseline 与未来表征研究

最新证据见 [2026-09-30 审查与小实验](AUDIT_2026-09-30.zh-CN.md)：包含本轮修改、可复查命令及验证边界。下文保留先前设计和验收历史。

这个目录记录复现边界，并提供 FLARE 启发分支的代码验收和运行入口。[2026-09-26 小规模实验记录](PILOT_RESULTS.zh-CN.md) 补充了已实现架构图、真实数据独立测试、GPU smoke 与续跑证据，尚未证明在线 RL 收敛更快。[English](README.md)

## 分支与阅读顺序

初始 baseline 快照为 `7db62813`，对应官方源码快照 `b85c07175b10017bf58ab83e1b1eee99666d0626`。`baseline/maniskill-rlt-2026-09-25` 后续包含了恢复与隔离修复 `ff566637`，本研究分支将该修复 cherry-pick 为 `43a0e615`。Stage 1/2 源码与启动脚本不代表两个阶段已经收敛。原工作目录仍为 `/home/luokz/rlinf_rlt/UPT_dev`。

研究分支是 `research/rlt-flare-latent-dynamics`，工作目录为 `/home/luokz/rlinf_rlt/UPT_flare_dev`。它复用现有 Python 环境，但源码目录独立。不要在正在训练的原目录切换分支。

建议依次阅读[baseline 来源](BASELINE.zh-CN.md)、[算法设计与消融](DESIGN.zh-CN.md)、[验证结果与未通过的集成门槛](VERIFICATION.md)。主要入口为：

- `toolkits/rlt/cache_latents.py`：不可变 episode 特征缓存。
- `toolkits/rlt/train_latent_world.py`：离线 Stage 1B、验证与续跑。
- `toolkits/rlt/evaluate_latent_world.py`：各 horizon 的动作相关性和不确定性诊断。
- `rlinf/models/embodiment/modules/rlt_latent_world.py`：未来表征模型。
- `rlinf/algorithms/rlt/latent_world.py`：实际执行 chunk 的 replay 对齐及来源检查。
- `toolkits/rlt/preflight.py`：不启动训练的配置/文件检查。

下面是以后明确启动时使用的通用命令，使用 GPU 0 或双 GPU 配置；baseline 占用 GPU 0/1 时不要执行。要在空闲 GPU 2 上隔离运行，请使用 [小规模实验入口](PILOT_RESULTS.zh-CN.md)。保存了中间 checkpoint 不等于已经收敛。

## 准备路径，不启动任务

使用研究源码目录，把大文件放到 NAS。Stage 1 checkpoint 必须已经完整保存且不再变化，不能选择仍在写入的文件，也不能使用 `pi05_base`。

```bash
cd /home/luokz/rlinf_rlt/UPT_flare_dev
source /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export EMBODIED_PATH="$PWD/examples/embodiment"
export RLT_STORAGE=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill
export RLT_DATASET="$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint"
export RLT_NORM_STATS="$RLT_DATASET/norm_stats.json"
export RLT_STAGE1_CHECKPOINT="/replace/with/completed/global_step_N/actor"
export RLT_LATENT_CACHE="$RLT_STORAGE/research/latent_cache_stage1_N"
export RLT_WORLD_RUN="$RLT_STORAGE/research/stage1b_seed2026_run01"
export RLT_WORLD_CHECKPOINT="$RLT_WORLD_RUN/best.pt"
export RLT_STAGE2_RUN="$RLT_STORAGE/research/stage2_seed1234_run01"
```

占位符必须在确认 checkpoint 后替换。`norm_stats.json` 必须是 Stage 1/2 使用的同一份归一化文件，如果实际存放位置不同，需要调整路径。换一个 encoder checkpoint 就使用新的 cache 目录；每个实验使用独立输出目录。venv、socket 和构建缓存不要放到 CIFS。

下面是只用 CPU、不接触 GPU 或 Ray session 的代码检查：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/unit_tests/test_models.py tests/unit_tests/test_data.py \
  tests/unit_tests/test_worker.py -k latent_world -q
CUDA_VISIBLE_DEVICES='' python toolkits/rlt/preflight.py --config-only
```

`--config-only` 只验证配置能否组合，不代表文件存在或已具备运行条件。Stage 1B 生成 checkpoint 后去掉该参数，才会比较真实文件的 feature contract。完整读取 Stage 1 权重做哈希在 NAS 上可能较慢，但不会把 VLA 加载到 GPU。

## Stage 1B：资源空闲后再执行

缓存工具在一张 GPU 上运行冻结 VLA 推理，之后的小模型训练只读取缓存。已经完成的有界真实数据实验见 [实验记录](PILOT_RESULTS.zh-CN.md)；下面的通用命令仍需等所选设备空闲后执行。

```bash
CUDA_VISIBLE_DEVICES=0 python toolkits/rlt/cache_latents.py \
  --config experiments/maniskill_rlt/config/cache_latents.yaml --batch-size 2

# 确认 W&B 账号/entity 前默认离线，避免再次写到无法访问的账号。
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python toolkits/rlt/train_latent_world.py \
  --config experiments/maniskill_rlt/config/stage1b.yaml

CUDA_VISIBLE_DEVICES='' python toolkits/rlt/evaluate_latent_world.py \
  --checkpoint "$RLT_WORLD_CHECKPOINT" --cache-dir "$RLT_LATENT_CACHE"

# 中断后，仅使用同一 run/config/cache 续跑：
CUDA_VISIBLE_DEVICES=0 python toolkits/rlt/train_latent_world.py \
  --config experiments/maniskill_rlt/config/stage1b.yaml \
  --resume "$RLT_WORLD_RUN/last.pt"
```

长命令可放进有名称的 tmux session，用 Ctrl-b、d 脱离终端。在线 W&B 需要先登录目标账号，并在训练前设置 `WANDB_ENTITY` 和 `WANDB_MODE=online`。不要提交凭据。之前 baseline 的 W&B 可见性问题与 GitHub 授权无关，本分支不会擅自修改运行中任务的登录状态。

Stage 1B 在 `RLT_WORLD_RUN` 下写入 `best.pt`、`last.pt`、`metrics.jsonl` 和 W&B 文件。非空目录必须显式 `--resume` 才能使用；续跑要求配置完全相同，包括 `max_steps`。不同实验预算应使用新配置和新目录。不支持多个进程同时写同一缓存或 run。

## Stage 2：先做集成检查，再做正式实验

Stage 2 还需要之前验证过的无桌面 Vulkan 环境。复用服务器本地 loader/ICD 即可，不需要桌面；不要 source baseline 启动脚本来设置环境，因为它会启动训练。

```bash
export RLINF_VULKAN_PREFIX=/home/luokz/.local/rlinf-vulkan
export PATH="$RLINF_VULKAN_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.1/lib64"
export SAPIEN_VULKAN_LIBRARY_PATH="$RLINF_VULKAN_PREFIX/lib/libvulkan.so.1.4.357"
export VK_DRIVER_FILES="$RLINF_VULKAN_PREFIX/share/vulkan/icd.d/nvidia_headless_icd.json"
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
unset DISPLAY WAYLAND_DISPLAY VK_LAYER_PATH VK_INSTANCE_LAYERS __NV_PRIME_RENDER_OFFLOAD

CUDA_VISIBLE_DEVICES='' python toolkits/rlt/preflight.py

# 只在 baseline 已结束、没有其他任务占用该 Ray cluster 时执行：
CUDA_VISIBLE_DEVICES=0,1 python examples/embodiment/train_embodied_agent.py \
  --config-name maniskill_rlt_stage2_latent_world \
  runner.max_epochs=2 algorithm.rlt_schedule.enable=False \
  algorithm.replay_buffer.min_buffer_size=1 algorithm.update_epoch=1 \
  algorithm.train_actor_steps=2 algorithm.critic_actor_ratio=1 \
  runner.save_interval=1 runner.val_check_interval=1
```

这只是集成检查，不是正式实验：有意关闭了正常 RLT warmup。它仍会采集 rollout，可能需要一些时间。需要看到至少一次 optimizer update、有限的 Q/world losses、实际保存的 checkpoint，并测试恢复与 rollout 权重同步。若关键阶段 gate 没有记录足够 transition，应排查原因，不能把“进程启动”当成通过。Ray 临时目录和 socket 放在本地磁盘或 `/dev/shm`，object spilling/checkpoint 放 NAS；不要为此停止共享 Ray 服务，也不要接入仍在运行的 baseline cluster。

正式实验使用没有 smoke overrides 的研究配置。对照使用 `maniskill_rlt_stage2_matched_baseline`，并换一个 `RLT_STAGE2_RUN`。两者都关闭 expert takeover，使用相同 Stage 1 checkpoint、资源设置和原 schedule。要评估干预减少，必须为两者设置完全相同的真实 expert checkpoint 和 takeover 规则。当前首先能够回答的是无接管成功率和样本效率。

`examples/embodiment/config/rlt_research/ac_baseline.yaml` 是有意保留的冻结参数快照，去掉只能在 primary config 中使用的 Hydra 元信息。这样既不修改官方配置，也避免嵌套继承其 `hydra.searchpath` 导致组合失败。测试核对缓存/rollout 特征配置和对照组资源。

## 明天的验收边界

先检查代码、测试和 Stage 1A/1B 设计，再决定是否启动新任务。本版没有实现显式长期记忆、接触/力标签、新任务 gate 或 in-DiT FLARE，也没有证明收敛加速、成功率提升、干预减少或迁移。设计文档给出了可证伪实验，并列明后续需要研究者决定的事项。
