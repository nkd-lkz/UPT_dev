# 验收 RLT 交互记忆分支

先读 [10-06 baseline 诊断](BASELINE_DIAGNOSTIC_2026-10-06.zh-CN.md)：在缓存特征上匹配比较 BC-only／Q+BC，再做冻结参数的仿真评估。[10-03 记忆实验](MATCHED_CAMPAIGN_2026-10-03.md)及下方旧 overnight pilot 保留历史用途。

## 在新服务器运行有界实验

先在相邻的 `UPT_dev/.venv` 安装 OpenPI/ManiSkill 环境，并准备共享数据集、Stage 1 第 2000 步权重、tokenizer 和仿真资源。然后在本分支目录执行：

```bash
bash run_rlt_overnight.sh memory 1
```

第二个参数是物理 GPU 编号。启动器先核对实际 worker 的 CUDA UUID，再测试一个 RGB 环境的 reset、取图和 step，随后加载 policy。两轮集成 smoke 和新增模块参数更新检查通过后，会重新初始化一个 pilot：2 个训练环境、4 个固定评估环境、每回合 500 控制步、global batch 32、micro batch 8、512 次 BC 热身更新、每轮最多 64 次更新。每 50 轮评估、保存视频和 checkpoint。自动 expert 接管关闭；记忆分支使用 attention 读取器。

默认达到 1000 个外层训练轮次或 12 小时就结束，以先达到者为准。`RLT_NIGHT_HOURS` 支持 1～24；`RLT_NIGHT_STEPS` 支持 20～5000 之间的 10 的倍数。`RLT_NIGHT_INTERVAL` 默认 50，必须整除训练轮数且至少留下两个 checkpoint（20 轮实验需设为 10）。退出码 124 表示达到时间上限，应使用最后一次定期 checkpoint，正在进行的更新不会保存。probe、smoke 或参数更新检查失败会停止对应分支。日志和实际配置保存到 `$RLT_STORAGE/runs/inspur_memory`，可用 `RLT_OUTPUT_ROOT` 改目录。这些 pilot 尚不能证明收敛，也不能代替同配置 baseline 对照。

每个任务独立管理 Ray head、端口和物理 GPU 锁，每个阶段自动选择三个连续空闲端口，避免复用 smoke 端口。过夜脚本忽略 `RLT_SMOKE_RAY_PORT`；单独调用 portable 脚本仍可指定端口，已占用时拒绝启动。显卡忙时拒绝启动。渲染设备使用查询到的 PCI 地址和本机 NVIDIA EGL 库。如果库路径不同，可设置 `SAPIEN_VULKAN_LIBRARY_PATH` 和 `RLT_NVIDIA_EGL_LIBRARY`。优先使用已存在的 `$HOME/.local/rlinf-vulkan/lib/libvulkan.so.1.4.357`，否则使用系统 loader；不复制其他主机的 NVIDIA 驱动或 C++ 库。原 GPU 2 启动脚本保持原样。

验证范围：9 月 30 日 A6000 实验已通过 CUDA UUID/RGB probe 和两轮新增模块更新 smoke；随后长实验因固定端口仍被占用，在启动 Ray 前失败。新的端口选择和 smoke 到 pilot 切换已通过 CPU 回归测试，修复后的长实验仍需在目标服务器运行。测试还覆盖 head/client/dashboard 端口占用、非法预算和忙卡拒绝。



最新证据见 [2026-09-30 审查与小实验](AUDIT_2026-09-30.zh-CN.md)：包含本轮修改、可复查命令及验证边界。下文保留先前设计和验收历史。

这份入口说明帮助你检查独立记忆分支，不启动训练。先读 [算法设计与代码索引](DESIGN.zh-CN.md)，再用下面的 CPU 测试和只读预检确认安装与接口；[最新小规模实验](PILOT_RESULTS.zh-CN.md) 提供架构图、真实 GPU 更新与续跑证据，以及预测探针的负结果。[验收记录](VERIFICATION.md) 保留初版历史记录。

分支：`research/rlt-zeva-interaction-memory`。baseline 起点：`ff56663769fd00f6108c39195888f5d55cb8a737`。Stage 1、FLARE 和 VR 分支保持独立。

## 不启动训练的检查

下面复用已有 Python，只运行合成输入的单元测试；模拟器边界使用替身，不运行真实仿真。反向传播测试只检查梯度路径，不产生训练 checkpoint。

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m pytest \
  tests/unit_tests/test_interaction_memory.py -q
```

下面仅检查 Hydra 配置、已有 Stage 1 权重、归一化文件和 Vulkan 文件是否存在，不加载权重、不启动 Ray、不分配 GPU、不创建训练目录。输出中的 GPU 2 是未来启动配置，不是这次 GPU 执行的证明。

```bash
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory --check
```

应看到 `Interaction memory: True` 和 `Preflight OK`。`RLT_STAGE1_ACTOR` 可指定其他完整落盘的 Stage 1 actor 目录；默认是 baseline 启动器中的 step 750。

## 将来获准后才运行的 smoke

当前交付没有执行下面的命令。你确认可以用 GPU 2 后，启动器会沿用原有占用检查、独立 Ray 和物理 GPU UUID 探针，再跑两轮小预算更新。它不会证明收敛。

```bash
cd /home/luokz/rlinf_rlt/UPT_zeva_dev
RLINF_VENV=/home/luokz/rlinf_rlt/UPT_dev/.venv \
  bash run_rlt_stage2_smoke_gpu2.sh --memory
```

需要先看 GPU 实测、loss、replay 记忆字段、checkpoint 保存与恢复，才能进入正式实验。输出位于 NAS 的 `runs/stage2_smoke/memory_stage2_gpu2_<时间>_<PID>/`；实验名为 `stage2_memory_smoke`。不加 `--memory` 仍运行原 baseline smoke。
