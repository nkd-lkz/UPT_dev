# 单环境 VR 在线学习验收

这份指南把已经可用的 Windows 仿真与 PICO 接管连接到 GPU 2 的在线 learner。先启动服务器，再建立隧道，最后用手柄产生真实训练样本。当前是单环境、单步动作的工程 smoke，不是 64 环境训练，也不是已收敛模型。

## 先明确训练的范围

手柄只控制 Windows 上的一个仿真实例。每次动作执行完成后，客户端保存原始 transition，再异步上传；服务器用冻结 Stage 1 提取两端特征，加入 replay 并更新小型 actor/critic。更新后的完整 actor 快照供后续推理使用。

```text
Windows：当前观察 → 服务器推理 → P 键执行模型动作
               ↘ 侧握键接管 → 本地 IK → 实际动作
                                          ↓
                    执行、显示、记录真实下一状态
                                          ↓ 后台上传（有限队列）
GPU 2：冻结 Stage 1 → 特征 replay + 人工样本 replay
                           ↓ TD + Q/BC 更新
                    actor / twin critic → 发布 actor → 下次推理
```

第一版使用 `horizon=1`，冻结 VLA 仍产生 10 步参考，只取首步作为参考 BC 目标。这样人工接管不会把未执行的 chunk 尾部误当成训练数据。此 head 与正式 baseline 的 10 步 head 不兼容；自动阶段切换、自动 OpenPI expert 均未接入本入口。用户按 `P` 选择模型控制，按住 grip 选择人工控制。

`online_smoke.yaml` 默认 replay 最少 8 条后开始更新，batch 8，最多 128 次更新，每 4 次发布权重，每 16 条保存 checkpoint。存在人工数据时，batch 的一半从人工 replay 采样，另一半从包含全部动作的 replay 采样，因此实际人工占比可能超过一半。人工样本的 BC 目标是实际接管动作；其他样本的 BC 目标是冻结 VLA 参考。critic 使用真实 sparse reward，不把按 grip 当作成功奖励。终止和超时均阻断 bootstrap，匹配本次显式配置。

网络上传和本地控制异步，服务器内部推理与更新串行；这不承诺每秒 10 次远程推理或训练。队列满、跟踪失效、回复过期或网络错误会暂停控制，不能用扩大队列掩盖长期处理速度不足。达到 128 次后停止优化但仍可采集和推理。

## 启动隔离的服务器

需要较长的人工验收时，在本目录运行以下入口。它默认使用已完成的 Stage 1 step 2000，使用 `online_pilot.yaml`：最少 64 条 replay、batch 32、最多 5000 次 learner 更新、每 8 次发布、每 50 条保存。actor 至少热身 128 次，且最近发布 minibatch 的 imitation MSE 不超过 0.01 才能参与模型控制；否则继续参考动作。这个门槛不是独立评估或安全认证。

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
bash run_rlt_vr_hil_pilot.sh check
tmux new-session -s rlt_vr_hil 'bash run_rlt_vr_hil_pilot.sh run; exec bash'
```

在 tmux 提示中输入至少 32 字符的连接口令，Windows 使用相同口令。新入口与研究训练共用 GPU 2 互斥锁，并继续拒绝 GPU 2 已被占用的情况；`check` 只校验配置和路径，不分配 GPU。不会修改正在运行的 GPU 0/1 baseline。这里仍是一个 Windows 环境和 `horizon=1`，不能加载正式 10 步 Stage 2 head，也没有合并至 64 环境训练。下面保留较短的 128 次更新 smoke 命令。

服务器固定按物理 GPU 2 的 UUID 设置 CUDA 可见性，并验证唯一可见设备；GPU 2 已占用超过 1 GiB 时拒绝启动。真实 smoke 另用 PCI 地址固定 Vulkan。不会连接 Stage 1 的 Ray，也不会执行 `ray stop`。请使用一个新的 tmux 会话，在其中执行：

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
export RLT_STAGE1_ACTOR=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill/runs/stage1/maniskill_rlt_stage1_resume750_20260926_153016/checkpoints/global_step_2000/actor
read -rsp '输入至少32字符的本次连接口令: ' RLT_VR_TOKEN
export RLT_VR_TOKEN
bash run_rlt_vr_online_gpu2.sh run
```

自行选择本次随机口令，在 Windows 输入同一个值；不要提交到 Git 或发到聊天。出现 `Online learner ready` 后再连接。输出保存到 NAS 的 `runs/vr_online/时间_PID/`，包括 `config.json`、`metrics.jsonl` 和 `learner.pt`。此 smoke 使用 JSONL 指标，不创建 W&B run。

## 连接 Windows 并接管

保持 PICO 企业串流与 SteamVR 正常跟踪。在一个 Windows PowerShell 窗口建立 SSH 隧道，`服务器地址` 替换为平时 SSH 使用的地址：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 8775:127.0.0.1:8775 luokz@服务器地址
```

这个窗口保持打开。在第二个 PowerShell 窗口更新 VR 分支，输入与服务器相同的口令，再启动客户端：

```powershell
Set-Location "C:\Users\lkz\Desktop\code\UPT_vr_dev"
git pull --ff-only origin feature/rlt-pico-vr-intervention
$env:RLT_VR_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host '连接口令' -AsSecureString)).Password
$vrRecord = "C:\Users\lkz\Desktop\rlt-records\online-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
conda run --no-capture-output --name rlt-vr python -X faulthandler -u -m toolkits.rlt_vr.client --online --port 8775 --render-backend cpu --record "$vrRecord" --max-episode-steps 1000 --log-interval 1
```

先保持暂停，检查画面与手柄跟踪。按 `P` 请求模型动作；随后握住侧握键，缓慢移动手柄，确认控制权切为 human。新按一次扳机切换夹爪开闭。松开 grip 后保持暂停，再按 `P` 才回到模型。`R` 重置回合，空格暂停，`Q` 退出。1000 步只用于人工验收，不与正式 baseline 评估配置的成功率直接比较。

终端的 `Online learner` 显示 `ack`、待上传数量和服务器指标。验收应同时看到 `human_accepted` 增加、`update_step` 增加、`policy_version` 增加，而不是仅有机械臂跟随。保留本地 `.npz`；服务器仅保存特征 replay，本地记录才包含原始图像。

## 中断与恢复

pilot 必须用相同配置恢复，统计入口读取原始执行日志：

```bash
bash run_rlt_vr_hil_pilot.sh run --resume /绝对路径/旧run/learner.pt
/home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m toolkits.rlt_vr.summarize_online /绝对路径/run
```

报告包含接管步数、连续接管段数、已结束回合结果和实际 learner 更新数。`human` 是客户端声明的控制来源，脚本注入也可以设置它，因此不能单凭该字段宣称真实人类实验。日志里的 `service_ms` 衡量服务端提特征与学习处理时间，不是端到端控制延迟。训练预算用完会显示 `update_budget_exhausted=true`；采集和推理继续，优化停止。

恢复须先停止旧客户端，再在服务器按 Ctrl-C。请求执行期间强制中断可能使 learner 进入故障状态，此时不会覆盖上一个有效 checkpoint。正常退出保存完整网络、target、两个 optimizer、replay、发布版本与 RNG。只有原来使用 `online_smoke.yaml` 的短 smoke 才用下面命令；pilot 必须使用上面的 `run_rlt_vr_hil_pilot.sh`，不能混用配置：

```bash
bash run_rlt_vr_online_gpu2.sh run --resume /绝对路径/旧run/learner.pt
```

恢复后重新启动 Windows 客户端，使用新的录制目录与新回合。活动仿真状态和未确认上传队列不恢复；不要重发旧会话的数据。配置与冻结 Stage 1/归一化身份必须匹配。最后一次 checkpoint 后的样本可能需要从本地记录另行整理，本版尚未实现 journal 自动重放。

## 已验证与待验收

服务器真实仿真 smoke 已验证同一 RPC、冻结 Stage 1 特征、脚本接管样本、TD/BC 更新、actor 发布、去重和 optimizer/replay 恢复；结果见 [VR_VERIFICATION.md](VR_VERIFICATION.md)。Windows/PICO 实际接管到 learner 的跨机器测试仍需操作者完成，不能把脚本接管记作人工成功率。

本 smoke 通过后，才继续开发 10 步部分 chunk 的掩码与折扣、正式 RLinf worker/replay 入口、阶段路由以及双卡 64 环境训练。当前所有新增入口独立于 baseline，暂不合并。
