# 验收 RLT 交互记忆分支

这份入口说明帮助你检查独立记忆分支，不启动训练。先读 [算法设计与代码索引](DESIGN.zh-CN.md)，再用下面的 CPU 测试和只读预检确认安装与接口；[验收记录](VERIFICATION.md) 区分已经通过和仍待 GPU 验证的部分。

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
