# VR Prototype Verification / VR 原型验证

This record separates server-side evidence from Windows/PICO hardware acceptance. 本记录区分服务器侧软件验证与 Windows/PICO 实机验收，不能把前者当作后者。

## Engineering Repair — 2026-10-06 / 控制与采集链路修复

Protocol 2 decouples durable admission from the single model-owner thread, bounds the local outbox/server inbox and retains raw journals. Pause reasons survive recovery, Cartesian targets are smoothed/rate-limited, joint commands respect soft limits, and human BC requires explicit fragment approval. Both guides describe the upgrade, controls and recovery: [English](VR_ONLINE.md) / [中文](VR_ONLINE.zh-CN.md).

协议 2 将持久化接收确认与模型处理分开，本地 outbox 和服务端 inbox 均有显式上限，保留原始记录。控制增加持续暂停原因、目标平滑和速度限制、关节软限位，人工 BC 必须明确批准片段。新旧协议及 schema-1/schema-2 checkpoint 不混用，首次验收从新 run 开始。原服务器进程与 GPU 0/1 训练未被停止或替换。

| Check / 检查 | Result / 结果 |
|---|---|
| Focused CPU regression / 定向 CPU 回归 | **47 passed, 2 optional hardware tests skipped**, 7.77 s; no GPU allocation / 不分配 GPU |
| Actual client loop / 真实客户端主循环 | Fake external VR/display/simulation devices; tracking/outbox fault → visible latch → release → Y/N review → re-anchor; no stepping during latch / 仅外部设备使用 fake，覆盖锁存恢复和审核 |
| Actual TCP and learner / 真实 TCP 与 learner | Receipt and health return while extraction is blocked; ordered disk outbox, quotas, duplicate/conflicting retries, worker fault and durable replay recovery / 提特征阻塞时仍可确认与查询，验证有界队列、去重和恢复 |
| Quality and control chain / 质量与切换链 | Unapproved/rejected human data excluded from BC; synthetic features exercise reference → approved correction → update/publication → actor-labelled execution → human again / 合成特征验证流程，非实机效果 |
| Actual Panda URDF / 真实 Panda 运动学 | 60 smoothed/bounded IK iterations reach a nearby 3-D target within 0.5 mm in forward kinematics / FK 误差低于 0.5 mm，不含物理接触 |
| Pilot path/config preflight / pilot 预检 | Passed with Stage 1 step 2000, no CUDA/RPC execution / 只验证路径和配置 |
| Shared NAS persistence / 共享 NAS 落盘 | 2 tests passed in 24.32 s: disk-outbox ingestion and acknowledged-receipt recovery after a worker fault; artifacts in NAS `runs/vr_online/protocol2_cpu_check.qjRIaUsE` / 使用真实 CIFS 挂载验证 fsync、替换与恢复，不是吞吐基准 |
| Static/docs / 静态与文档 | Ruff lint/format and whitespace checks; both Sphinx trees: 0 build warnings; scoped guide markup/symbol checks pass / 两种语言构建无警告，指南按代码人工交叉核对 |
| CPU-render integration attempt / CPU 渲染集成尝试 | **Failed** before IK: SAPIEN could not create a supported physical device named `cpu`, including an explicit lavapipe/loader retry / 本机 CPU 渲染未通过，未改用占用中的 GPU |
| New GPU/Windows acceptance / 新版 GPU 与 Windows 验收 | **Not run**; GPU 2 still serves the old VR process, and physical PICO/Windows GUI is not accessible here / 仍待两端升级后现场验收 |

The 47-pass command disables optional hardware tests. The separately attempted renderer check failed; it is not hidden by that pass count. This does not establish Windows smoothness or task-success improvement. Synchronous local fsync/IK/render can still stall the UI, and arrivals above processing throughput eventually trigger bounded backpressure. Unacknowledged Windows records are retained, but automatic local-outbox restart/reconciliation is not implemented. Server recovery requires all ancestor inbox directories named by its checkpoint.

47 项通过来自禁用硬件测试的 CPU 回归；另行尝试的渲染检查确实失败，不计作通过。本次不能宣称 Windows 已流畅、正式 Stage 2 已打通或成功率提高。本地同步落盘、IK、渲染仍可能拖慢 UI；持续采集快于服务端处理时仍会按配额反压。未确认上传的 Windows 记录会保留，但没有自动重启本地 outbox 的功能；服务端恢复依赖 checkpoint 引用的所有历史 inbox 目录。

The following sections preserve historical evidence for earlier revisions. 以下保留旧版本记录，不能替代本次协议 2 的硬件验收。

## Continuation — 2026-09-30 / 较长人工验收入口

`run_rlt_vr_hil_pilot.sh check` passed with Stage 1 step 2000, without CUDA allocation. CPU RPC/learner/control regression: **34 passed, 2 optional hardware skips**. Added update-budget reporting, publication imitation gate, resumed gate/metrics state, shared GPU-2 lease, per-run takeover summary and pre-allocation configuration validation. This does not add production ten-step/multi-environment integration. No new full VLA GPU smoke or physical PICO acceptance was run in this continuation.

已通过 step 2000 路径与 pilot 配置预检；CPU 回归 34 项通过、2 项硬件可选跳过。新增训练预算、发布时 imitation 门槛、门槛与指标恢复、GPU 2 互斥锁、接管统计和加载权重前的配置校验。本次仍未验证正式 10 步多环境接管，也没有将脚本样本记为真人实验。

## Original Online Update — 2026-09-27 / 原在线更新

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
