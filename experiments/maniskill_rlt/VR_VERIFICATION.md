# VR Prototype Verification / VR 原型验证

This record separates server-side evidence from Windows/PICO hardware acceptance. 本记录区分服务器侧软件验证与 Windows/PICO 实机验收，不能把前者当作后者。

## Online Update — 2026-09-27 / 在线更新

The standalone single-step learner now passes real GPU 2 simulation/RPC/update smoke. Run instructions: [English](VR_ONLINE.md) / [中文](VR_ONLINE.zh-CN.md). 新增的是独立单步 learner，不是正式 64 环境 worker 集成；Windows/PICO 到在线 learner 的跨机器验收仍待操作者完成。

Two 40-transition trials used the completed Stage 1 step1500 export, CPU physics, PyTorch IK and Vulkan on `pci:0000:e1:00.0`. The final trial is NAS `runs/vr_online/20260927_162119_1187984`, beneath `/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill`. Both exited 0. 两次均通过，最终 trial 增加了恢复后继续 optimizer 更新的检查；`result.json` 明确记录 `physical_pico_tested=false`。

| Check / 检查 | Measured Result / 实测 |
|---|---|
| Executed transitions / 实际仿真 transition | 40; two timed-out episodes / 两个超时回合 |
| Scripted takeover samples / 脚本接管样本 | 12; not physical human interventions / 不是真实人工操作 |
| Critic / actor updates | 33 / 33 |
| Published actor version / 发布版本 | 32, used for subsequent simulator actions / 已用于后续仿真动作 |
| Actor / critic maximum weight change | 0.00355607 / 0.00254076 |
| Duplicate upload / 重复上传 | Same sequence acknowledged without reinsertion / 未重复入库 |
| Checkpoint resume / 恢复 | Model exact reload; nonempty optimizer states; update 33→34 on replay / 精确加载，保留 optimizer，再更新一步 |
| Peak Torch allocation / Torch 峰值分配 | 9767.38 MiB; excludes driver/Vulkan allocations / 不含驱动及 Vulkan |
| CPU regression / CPU 回归 | 26 passed, 2 optional hardware tests skipped |

CUDA identity was checked before model allocation against physical GPU 2 UUID `GPU-4662787b-485a-0e8f-e4b2-dd47352ed69c`. Baseline GPU 0/1 PIDs 2061116/2061119 remained alive before and after; GPU 2 was released after each trial. 没有启动 Ray，没有停止或重启 baseline，没有修改共享训练环境依赖。Smoke checkpoints contain scripted intervention demonstrations and are engineering artifacts, not trained human-expert policies / smoke 权重含脚本样本，不作为人工专家训练成果。

Tests cover GPU busy rejection/UUID selection, terminal and time-limit targets, executed human BC targets, actor publication, exact CPU optimizer/RNG/replay continuation, ordered/deduplicated authenticated uploads, episode continuity, and fail-closed optimizer faults. Ruff, shell syntax and whitespace checks pass. Both Sphinx trees build with zero warnings; unchanged global scans retain 61 markup and 26 symbol findings. 新 Markdown 按 `refine-docs` 与 `docs-check` 核对命令及 EN/ZH 范围；未改 public RST 页面。

The following sections preserve the earlier inference-only milestone, not the current online evidence / 以下为早期仅推理阶段记录，最新状态以上文为准。

## Scope / 范围

The branch adds standalone tools, tests and documentation; baseline training files are unchanged. 分支只新增独立工具、测试和文档，不修改 baseline 训练入口，也不合并 FLARE。Test date / 测试日期：2026-09-26。

| Component / 组件 | Evidence / 证据 |
|---|---|
| Action ownership / 动作控制权 | Chunk interruption, reset invalidation, tracking/UI fault latches, terminal blocking, delayed real RPC reply rejection / 中断 chunk、重置废弃回复、故障后禁止自动恢复、回合结束保护、真实延迟回复拒绝 |
| Transport / 通信 | Loopback TCP, token failure, environment mismatch, size cap, incomplete connection, lossless RGB codec / 本机 TCP、认证失败、环境不一致、消息上限、断连、无损图像 |
| Recording / 记录 | Executed actions and human mask; read with `allow_pickle=False`; refuse existing output directory / 实际动作及人工标签；禁止覆盖已有目录 |
| VR adapter / 手柄接口 | Vendor SDK mock checks valid/invalid tracking, button mapping and idempotent shutdown / SDK mock 验证跟踪、按键与重复关闭；非真机证明 |
| Simulation / 仿真 | Linux CPU physics + GPU 2 Vulkan, actual baseline cameras, IK at current and displaced TCP, reset/step; Windows CPU-render path reached IK initialization / Linux CPU 物理与 GPU 2 渲染，真实相机、IK、重置与步进；Windows CPU-render 路径已到达 IK 初始化 |
| Stage1 inference / Stage 1 推理 | Actual step750 weights; two `10×8` chunks and simulator steps; approximately 370/161 ms model calls / 真实 step750，两次动作与步进；不包含校园网 |
| Stage2 inference / Stage 2 推理 | Actual smoke step2 head + Stage1 features, through PNG/TCP RPC to an executed simulator step / 真实 smoke step2 head，经 PNG/TCP 到仿真步进；非收敛或在线训练证明 |
| Static checks / 静态检查 | Ruff lint/format, CLI help, whitespace checks / lint、格式、命令帮助与空白检查 |

GPU tests select `CUDA_VISIBLE_DEVICES=2` and render backend `pci:0000:e1:00.0`. GPU 0/1 training was left running; GPU 2 returned to 4 MiB afterward. GPU 测试没有停止或重启任何训练进程。

Server test environment / 服务器测试环境：Python 3.11.14, ManiSkill 3.0.0b22, SAPIEN 3.0.1, Torch 2.11.0+cu128, NumPy 1.26.4. Windows instructions select a separate Torch 2.7.1/cu126 environment; that exact Windows combination is not yet tested / Windows 环境组合尚未实测。

## Reproduce / 复查

Run from this branch's repository root with its code on `PYTHONPATH`. 普通测试不需要 GPU；网络测试需要允许本机 socket。

```bash
python -m pytest tests/unit_tests/test_remote_teleop.py -q
ruff check toolkits/rlt_vr tests/unit_tests/test_remote_teleop.py
ruff format --check toolkits/rlt_vr tests/unit_tests/test_remote_teleop.py
```

Ordinary run after the Windows IK backend change / Windows IK 后端修改后的普通运行：18 passed, 2 hardware tests skipped. The two additional tests require an explicitly selected free GPU and the server's working headless Vulkan environment / 另外两个测试需显式选定空闲 GPU 并设置已验证的 Vulkan 环境：

```bash
export CUDA_VISIBLE_DEVICES=2
export RLT_VR_SIM_TEST=1
export RLT_VR_RENDER_BACKEND=pci:0000:e1:00.0
export RLT_VR_STAGE1="$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_2xl40_20260925_163418/checkpoints/global_step_750/actor"
export RLT_VR_DATASET="$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint"
export RLT_VR_ACTOR="$RLT_STORAGE/runs/stage2_smoke/stage2_gpu2_20260926_154350_2095986/stage2_smoke/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
python -m pytest tests/unit_tests/test_remote_teleop.py -q
```

`RLT_STORAGE` is the NAS project root defined in the guide; omit `RLT_VR_ACTOR` to test base inference. Hardware tests use trusted checkpoints and perform inference only / 路径变量沿用指南，省略 `RLT_VR_ACTOR` 则验证 base；硬件测试不训练。

The earlier Pinocchio-backed implementation passed 19 hardware-enabled tests in 51.55 s. The new PyTorch IK path still requires a repeated Linux hardware check and the Windows check below; do not treat the old run as evidence for the replacement backend. Both Sphinx language builds previously reported zero build warnings. This is a focused suite, not the full repository test suite / 旧 Pinocchio 实现曾通过 19 项含硬件测试；新 PyTorch IK 路径仍需重跑 Linux 硬件检查和下述 Windows 检查，不能沿用旧结果作为新后端证据。此前中英文 Sphinx 构建均为 0 警告；本次为定向回归，未运行全仓测试。

## Still Required / 尚待验收

Windows/PICO, campus-network p50/p95, GUI responsiveness, controller calibration, and operator intervention on the physical borrowed device remain untested. 本地硬件不可远程访问，不能声称 VR 已可用或 RTX 4060 已达到某个帧率。

Native stereo VR rendering, Windows-to-Ubuntu VR input bridging and automatic baseline phase routing remain unimplemented. 在线 replay 已由上文独立单步入口接通；正式 baseline 的 10 步 partial-chunk 集成仍待开发。No real-robot safety guarantee is provided / 不具备真机安全保证。

The bilingual guide was checked against CLI signatures, environment/action dimensions and checkpoint loading. Sphinx checks cover the unchanged documentation trees, not these standalone Markdown pages. 本次指南按 `refine-docs` / `docs-check` 保持中英文流程和命令一致；全仓检查仍有 61 个既有 markup 提示及 26 个既有符号提示，未改动相关页面。
