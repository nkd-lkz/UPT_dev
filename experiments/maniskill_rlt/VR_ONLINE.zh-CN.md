# 单环境 VR 在线学习验收

这份指南说明如何验收 Windows 仿真、PICO 人工纠正与 GPU 2 learner 的完整闭环。先同步两端协议，再测试本地控制、上传与更新，最后确认新 actor 实际执行后仍能被人工接管。当前是独立 `horizon=1` 工程试验，**不是正式 64 环境、10 步 head 的 Stage 2 baseline**。

## 同步两端并开始新试验

本次更新使用 `online_protocol=2`，改变了接收确认和质量标签的含义。旧客户端与新服务端不能混用。源码更新不会升级正在运行的进程：先退出 Windows 客户端，在原 VR 服务端终端按 Ctrl-C 正常停止，再启动新版。不要停止 GPU 0/1 baseline 或其他 GPU 2 作业。

checkpoint 使用 schema 2；旧 schema 1 把所有接管动作都用于 BC，不能直接恢复到新版。首次验收创建新 run，不传 `--resume`，保留旧日志和权重。先检查配置，再在新 tmux 会话启动：

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
bash run_rlt_vr_hil_pilot.sh check
tmux new-session -s rlt_vr_hil_v2 'bash run_rlt_vr_hil_pilot.sh run; exec bash'
```

`check` 不分配 GPU。启动时输入至少 32 字符的口令，Windows 使用相同值；不要提交口令或发到聊天。入口按物理 GPU 2 UUID 隔离，使用互斥锁，超过 1 GiB 已用显存时拒绝启动。不会运行 `ray stop` 或修改 baseline。等待 `Online learner ready` 后再连接。

pilot 默认使用 Stage 1 step 2000、最少 64 条 replay、batch 32、最多 5000 次 learner 更新，每 8 次发布、每 50 条保存 checkpoint。actor 至少完成 128 次更新，且最近发布 minibatch 的有效 BC MSE 不超过 0.01 才能执行；否则继续 reference。这个门槛不是独立评估或安全认证。短 smoke 的 `run_rlt_vr_online_gpu2.sh run` 仍使用 8 条起训、batch 8、最多 128 次更新，每 4 次发布、每 16 条保存；两种配置不能混用恢复。

## 在 Windows 验证本地控制

服务端就绪后，保持企业串流和 SteamVR 跟踪正常。在一个 PowerShell 窗口建立隧道，将 `服务器地址` 替换为实际 SSH 地址：

```powershell
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8775:127.0.0.1:8775 luokz@服务器地址
```

保持隧道窗口打开。在另一个 PowerShell 窗口同步 VR 分支并启动新录制；两端必须使用同一版代码：

```powershell
Set-Location "C:\Users\lkz\Desktop\code\UPT_vr_dev"
git pull --ff-only origin feature/rlt-pico-vr-intervention
$env:RLT_VR_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host '服务器端相同的连接口令' -AsSecureString)).Password
$vrRecord = "C:\Users\lkz\Desktop\rlt-records\acceptance-v2-" + (Get-Date -Format 'yyyyMMdd-HHmmss')
conda run --no-capture-output --name rlt-vr python -X faulthandler -u -m toolkits.rlt_vr.client --online --port 8775 --render-backend cpu --record "$vrRecord" --max-episode-steps 1000 --log-interval 1
```

客户端默认暂停。先按住侧握键并缓慢移动；每次重新接管，会以当前手柄和 TCP 建立新锚点。新按一次扳机切换夹爪目标，松握不会自动张开夹爪。输入与显示仍在本地线程，远程推理不会直接阻塞这个线程。

遥操作保持 10 Hz 仿真控制频率，默认目标平滑时间常数 0.12 s、平移上限 0.12 m/s、旋转上限 0.8 rad/s；关节命令限制为 0.025 rad/步，并限制相邻命令变化，距关节边界保留 0.02 rad。相对锚点仍限制 15 cm / 30°。这些是目标和命令限制，不保证实际速度、避碰或真机安全。`--tcp-speed` 和 `--angular-speed` 可调整目标速度，不能靠放大上限解决接触卡住。

## 区分接管与可用于模仿的纠正

机械臂跟随正常后，按 `P` 请求模型动作，再尝试中途接管。松开 grip 后确认刚才的片段：`Y` 表示可用于 BC，`N` 表示排除 BC。确认后重新握住或按 `P` 继续。`R` 重置，空格暂停，`Q` 退出。结束或重置时未确认的记录标为 `unreviewed`，不会默认批准。

| 数据标签 | replay 与训练用途 |
|---|---|
| `human + approved` | 普通 replay 和纠正 replay；实际动作作为 BC 目标 |
| `human + rejected/unreviewed` | 保留真实动作和结果，参与 critic；不直接作为 BC 目标 |
| `reference/actor + policy` | 普通 replay；BC 使用冻结 VLA 的首个参考动作 |

原始 `.npz` 不被改写，审核写入 `review_*.json`。批准是操作者判断，不是自动成功检测。当前整段确认，不支持撤销已上传的批准，应短段操作并及时审核。纠正 replay 采样半个 batch，另一半来自全部 replay，所以实际人工占比可以高于一半。奖励仍是真实 sparse reward，不因按键或批准而增加；当前配置终止和超时均阻断 bootstrap。

## 识别暂停原因和处理进度

确认片段后后台才上传，严格保持执行顺序；后续 transition 也不能越过待审核片段。窗口持续显示暂停原因与恢复提示，安全暂停不会因为队列或跟踪恢复就自动运动。

| 原因 | 操作者下一步 |
|---|---|
| `tracking_invalid` / `ui_stall` | 检查 SteamVR 或本地卡顿；松握后重新接管 |
| `ik_failed` / `no_progress` | 检查目标、接触和姿态；松握、缩小目标并重建锚点 |
| 相对运动限幅 | 松握、重新定位手柄再握住；限幅本身不结束回合 |
| `review_required` | 松握后按 `Y` 或 `N` |
| `local_outbox_full` | 审核片段并等待后台接收；队列腾出后松握再接管 |
| `upload_error` / `server_storage_full` / `local_storage_full` | 停止操作，保留记录，检查连接、服务端故障或配额 |
| `episode_ended` | 只能用 `R` 开启下一回合；握键不能解除 |

连续 20 个控制步中，目标位置误差超过 2.5 cm 且每步实际移动不足 0.2 mm，会触发 `no_progress`。它不能证明碰撞，也不能区分接触、限位和动力学滞后。日志分别记录 IK、仿真 step、落盘耗时，及目标误差、关节限幅和实际 TCP 位移。原始样本含 `teleop_json`，稀疏控制事件保存在 `events.jsonl`。

服务端先校验并同步落盘到 `inbox/`，再确认接收；单个后台线程独占模型，交替处理推理和提特征/学习。前台网络接收不等待学习完成，但单次模型操作不可抢占，仍不承诺 10 Hz 远程推理或优化。

| 可见计数 | 含义 |
|---|---|
| `received` / `received_sequence` | 服务端已持久化的序号，从 0 开始，不是更新次数 |
| `processed` / `sequence` | 已完成提特征和入 replay 的客户端序号 |
| `local_pending` / `outbox` | 排队和正在上传的记录，包括待审核片段 |
| `server_pending` / `pending_learning` | 已确认接收但尚未处理完的记录 |
| `approved_accepted`、`update_step`、`policy_version` | 已处理批准纠正数、优化次数、发布版本 |

默认本地 outbox 最多 512 条，仅排队文件路径；本地记录配额 8 GiB，至少保留 128 MiB 空闲。服务端待处理最多 128 条，journal 配额 4 GiB、空闲保留 1 GiB。满额反压并最终暂停，不能丢弃 transition 或无限扩大队列。连续握住约 51 秒可能用尽 512 条待审核预算，应分段操作。

本地写入仍同步 `fsync`，慢磁盘仍可能影响操作；这次解耦了网络和学习，不宣称硬实时控制。checkpoint、特征 replay 等不计入原始 journal 配额，需要额外空间。记录不会自动删除。

## 恢复时保留证据

正常结束时先按 `Q`，等待上传退出，再在服务器按 Ctrl-C。Windows 的 `receipt.json` 注明最后一次确认序号和服务端 run；服务器保存 `inbox/` 原始请求、`metrics.jsonl`、`learner.pt` 和 `config.json`，不创建 W&B run。故障后保留两端全部文件，不要将旧样本改成新 session 后全部重发，否则可能重复训练。

同一协议、配置和冻结特征身份的 schema-2 checkpoint 可以恢复：

```bash
bash run_rlt_vr_hil_pilot.sh run --resume /绝对路径/旧run/learner.pt
/home/luokz/rlinf_rlt/UPT_dev/.venv/bin/python -m toolkits.rlt_vr.summarize_online /绝对路径/run
```

恢复会自动重放最后有效 checkpoint 之后已持久化的服务端 receipts，再允许新 Windows 会话。checkpoint 引用全部历史 `inbox/` 目录，不能删除、移动或单独复制 `learner.pt`。本地仿真状态和未被服务端确认的 Windows 文件不自动恢复；保留它们与 `receipt.json`，另行核对，不假定它们已学习。旧 run 与新 run 分开统计，模型累计计数可跨 run 增长。

报告区分人工数、批准纠正数以及真实 reference/actor 执行步数。`policy_version` 增长本身不代表 actor 已执行，要看到 `policy_source=actor` 的真实 transition。`service_ms` 是提特征/学习耗时，不是端到端延迟。`update_budget_exhausted=true` 表示优化停止，采集和推理仍可继续。

## 完整闭环验收

先用短工程记录验收，不直接恢复正式采集：reference 实际执行 → 人工纠正 → 松握并 `Y` 确认 → `approved_accepted`、`update_step`、`policy_version` 增长 → 通过门槛后 `P` 实际执行 actor → 再次人工接管。未达到门槛时应继续 reference，不能降低门槛伪装验收成功。

CPU 测试覆盖状态机、真实客户端循环、本机 socket、顺序上传、延迟学习、配额、批准标签、故障恢复，以及使用合成特征的 reference/actor/人工切换。它们不证明 Windows/PICO 流畅，也不证明任务成功率提高。历史 GPU smoke 不代表新异步版本已通过 GPU 验收；本次范围与未通过项见 [验证记录](VR_VERIFICATION.md)。自动阶段路由、OpenPI expert、正式 10 步 partial-chunk 和 64 环境 worker/replay 集成仍未接入此入口。
