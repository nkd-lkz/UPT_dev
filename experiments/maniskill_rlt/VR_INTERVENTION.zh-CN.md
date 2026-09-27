# 本地仿真与 VR 接管，服务器推理

本指南帮助你在 Windows 笔记本上运行单个 ManiSkill 环境、显示相机画面并用 PICO 手柄接管动作，同时让服务器加载 RLT checkpoint 生成动作。先验证本地仿真，再验证手柄，最后连接推理服务；全部通过后按 [单环境在线验收](VR_ONLINE.zh-CN.md) 连接 GPU 2 learner。

开发分支为 `feature/rlt-pico-vr-intervention`，起点是 baseline 的 `ff566637`，独立工作目录为 `/home/luokz/rlinf_rlt/UPT_vr_dev`。本分支没有合入 baseline，也不包含 FLARE 算法改动。运行中的 Stage 1 和原 Stage 2 入口保持不变。

## 两台电脑各自运行什么

Windows 独立处理人工控制，因此 VR 接管不必等待服务器的推理回复。网络仍会影响自动执行的等待时间；它不再承担仿真画面的持续传输。

```text
PICO → PICO Business Streaming / SteamVR → 右手柄位姿、按键
                                                ↓
Windows：CPU 物理仿真 ← 本地 IK / 接管控制 ← 操作者
         ↓ 两路 RGB、9 维 qpos               ↑ 本地相机窗口
         └──────── SSH 隧道 ────────────────┐
服务器：冻结的 Stage 1 特征模型 → 可选 Stage 2 actor
         └──────── 10×8 动作 ───────────→ Windows 逐步执行
```

动作仍是 `pd_joint_delta_pos`：7 维归一化关节增量加 1 维夹爪命令。手柄的 6DoF 位姿先经过坐标转换与 IK，不能把 6 维笛卡尔运动直接塞给 8 维 RLT 动作。两路相机沿用 baseline 的 `3rd_view_camera` 和 `wide_hand_camera`，模型输入为 384×384 RGB，使用 PNG 无损传输。

当前画面是本地 OpenCV 双相机窗口，不是头部跟踪的双目 VR 渲染。先在电脑屏幕上观察；头显桌面视图是否占用 SteamVR 输入需要实测，不能把它视为已经完成的沉浸式 VR 应用。

## Windows 能否运行

RTX 4060 8 GB 不需要加载 VLA 权重，预计适合尝试单环境渲染，但 CPU 性能、驱动和 SteamVR 共同负载决定实际速度。官方支持矩阵列出 Windows 的 CPU 仿真和渲染，未支持 GPU 仿真；本客户端固定 `num_envs=1`、`sim_backend=physx_cpu`。Windows 默认使用 `render_backend=cpu`，避免 SAPIEN 3.0.1 的 CUDA 图像互操作路径；Linux 默认仍为 `gpu`。不要用 WSL 替代原生 Windows 的图形环境。[ManiSkill 系统支持](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#system-support)

已确认 SAPIEN 3.0.1 发布了 `cp311-win_amd64` wheel，但安装包存在不等于这台笔记本已验收。[SAPIEN 发布清单](https://pypi.org/project/sapien/3.0.1/#files)

在 Windows PowerShell 中，用单独环境安装客户端。以下 PyTorch 使用 CUDA 12.6 构建，而不是服务器的 CUDA 12.8 构建；版本命令来自 [PyTorch 官方历史版本](https://pytorch.org/get-started/previous-versions/#v271)。不要在笔记本运行完整 RLinf 服务器安装脚本。

```powershell
git clone --branch feature/rlt-pico-vr-intervention https://github.com/nkd-lkz/UPT_dev.git UPT_vr_dev
cd UPT_vr_dev
py -3.11 -m venv .venv-vr
$PY = ".\.venv-vr\Scripts\python.exe"
& $PY -m pip install --upgrade pip
& $PY -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
& $PY -m pip install -r toolkits/rlt_vr/requirements-client.txt
& $PY -m pip check
& $PY -m toolkits.rlt_vr.preflight --probe sim --steps 20
```

预检会构建正式插销任务、检查两路图像与 IK，再执行 20 个保持机械臂位置的控制步。它不弹窗，也不连接服务器；成功时输出 `PASS` 和步耗时。客户端使用 ManiSkill 已固定依赖的 `pytorch_kinematics` 求解 Panda IK，不需要另装 Pinocchio；Windows 下 Pinocchio 的 LLVM OpenMP 与 PyTorch 的 Intel OpenMP 不能安全地加载在同一仿真进程。如果 wheel 或 Vulkan 不可用，保留完整报错，不要悄悄升级仿真版本后声称与 baseline 一致。版本不一致需要单独回归。

## 验证 PICO 输入并单独接管

本机仿真通过后，再连接 PICO Business Streaming 和 SteamVR，确认右手柄被识别。优先尝试 USB，减少头显无线链路的不确定性。PICO 官方说明该软件的 PC 端为 Windows，并依赖 SteamVR。[PICO Business Streaming](https://business.picoxr.com/us/software/streaming-assistant)

```powershell
& $PY -m toolkits.rlt_vr.preflight --probe vr --steps 60
& $PY -m toolkits.rlt_vr.client --manual-only --record C:\rlt-records\manual-001 --max-episode-steps 1000 --log-interval 1
```

预检持续打印 `valid`、按键位图、grip、trigger、模拟量 `trigger_value` 和位置。按住侧握键应令 `grip=True`；按下食指扳机应令 `trigger=True` 或提高 `trigger_value`。正常跟踪时，移动过程中 `valid` 应保持为真。默认按键编号为 2 和 33，可用 `--clutch-button`、`--trigger-button` 修改；客户端还会读取 SteamVR 声明的扳机轴，数值达到 `--trigger-threshold 0.6` 时视为按下。不能绕过 `valid` 检查来强制移动机械臂。

| 操作 | 本地行为 |
|---|---|
| 按住右手 grip | 以当前 TCP 和手柄位姿建立相对移动起点，接管机械臂 |
| 移动或转动手柄 | 本地 IK 生成关节增量；位移缩放 0.5，单次握持最多偏移 15 cm、旋转 30° |
| 接管期间新按一次 trigger | 切换夹爪开/闭目标；松开 trigger 不改变目标 |
| 松开 grip | 暂停；保留夹爪目标，不自动恢复模型动作 |
| `P` | 从当前新观测请求模型动作；manual-only 下无效 |
| 空格 / 跟踪失效 / UI 卡顿 / IK 失败 | 暂停；松开 grip 后才能重新接管 |
| `R` | 新回合，仍从暂停状态开始；废弃旧回合的推理回复 |
| `Q` / Esc / 关闭窗口 | 退出并关闭本地资源 |

先在空中测试三个平移方向与旋转方向，再尝试夹取。默认 OpenVR 的右/上/后坐标映射到机器人前/左/上坐标，可用 `--yaw-degrees` 校正站立朝向。IK 不等于避障，也不保证接触稳定；目前限制只服务于仿真调试，不能用于真机安全控制。

终端默认每秒输出一条 telemetry，其中包含控制权、跟踪状态、原始按键、扳机模拟量、请求与实际采用的相对运动、最大机械臂动作、夹爪命令、仿真步耗时和显示循环耗时；相机窗口也会显示按键与耗时摘要。相对运动达到边界时，客户端会保持边界目标并显示 `Motion limit reached`。此时松开 grip，把手柄移回舒适位置，再次握住以建立新起点。如果仿真活动空间确实需要扩大，可显式设置不超过 30 cm 的范围，例如 `--max-displacement 0.25`；完成全部方向校准前保留默认值。

训练任务注册的单回合上限是 100 个控制步，按 10 Hz 计算只有约 10 秒实际运动。人工校准命令使用 `--max-episode-steps 1000`，把可执行时间扩展到约 100 秒，不改变控制频率和模型输入。客户端会分别输出 `terminated` 与 `truncated`：任务成功对应前者，达到 wrapper 的步数上限对应后者；两种情况都需要按 `R` 开始新回合。对比任务级评测结果时仍应使用 baseline 的 100 步。

显示循环的 watchdog 会在仿真和 SteamVR 初始化完成后才开始计时，只有单次循环超过 `--stall-timeout 2.0` 才暂停。watchdog、跟踪或 IK 故障触发后，松开一次 grip 即可解除安全锁。排查 Windows 慢渲染时最多可把阈值提高到 10 秒，但这会推迟故障检测，不能作为真机参数。

## 启动服务器推理

本地接管可用后再占用推理 GPU。以下命令使用现有服务器环境，只启动冻结模型；不会启动 Ray、训练、W&B 或服务器仿真。先用 `nvidia-smi` 确认 GPU 2 仍空闲。

两端需要设置同一个至少 32 字符的随机口令。保存在自己的密码管理器中，在两端终端隐藏输入；不要发到聊天、提交到 Git 或写进启动脚本。

```bash
cd /home/luokz/rlinf_rlt/UPT_vr_dev
source /home/luokz/rlinf_rlt/UPT_dev/.venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=4
read -rs -p 'RLT VR token: ' RLT_VR_TOKEN
export RLT_VR_TOKEN
export RLT_STORAGE=/mnt/nas_ailab_434/Personal_File/luokz/rlinf_rlt_maniskill
python -m toolkits.rlt_vr.server \
  --stage1 "$RLT_STORAGE/runs/stage1/maniskill_rlt_stage1_2xl40_20260925_163418/checkpoints/global_step_750/actor" \
  --dataset "$RLT_STORAGE/datasets/lerobot/maniskill_peginsertionside_joint"
```

出现 `Inference ready` 后才连接。默认执行 Stage 1 的参考动作 `ref_chunk`。如需验证 Stage 2 head，额外提供 `--actor /绝对路径/actor/model_state_dict/full_weights.pt`。该模式每个 chunk 都使用 actor，不复刻 baseline 的阶段自动切换，也不是专家模型接管；实际人工动作仍由 Windows 产生。只加载自己可信的权重。

## Windows 连接并接管模型

推理服务只监听服务器的 `127.0.0.1:8765`。校园网内不一定允许主机互访；先确认 SSH 可达，再通过隧道连接，避免把图像和凭据明文暴露到校园网。

在单独的 PowerShell 窗口保持隧道运行，将服务器地址替换为实际 SSH 地址：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 127.0.0.1:8765:127.0.0.1:8765 luokz@SERVER_IP
```

回到客户端终端，输入与服务器相同的口令，依次测网络、真实推理和交互窗口：

```powershell
$env:RLT_VR_TOKEN = [System.Net.NetworkCredential]::new("", (Read-Host "RLT VR token" -AsSecureString)).Password
& $PY -m toolkits.rlt_vr.preflight --probe network --steps 20
& $PY -m toolkits.rlt_vr.preflight --probe inference --steps 5
& $PY -m toolkits.rlt_vr.client --record C:\rlt-records\intervention-001
```

窗口初始暂停。按 `P` 开始模型执行；中途握住 grip 应立即清除剩余动作，并在下一个可执行的本地控制步进入人工控制。也可先用 `--no-vr` 验证键盘暂停与自动执行；这不能作为 VR 验收。

物理控制频率保留 baseline 的 10 Hz，每次最多接收 10 步动作。显示窗口可处理更频繁的输入，但相机画面随仿真步更新，不承诺 60 FPS。等待推理时暂停物理时间，窗口和 VR 读取继续工作；人工可以在等待期间接管。超过 `--reply-ttl`（默认 10 秒）的回复不执行，旧回合或旧控制状态的回复也不执行。这是可暂停的采集工具，不是保持真实时间的控制系统。

网络预检的 p95 只代表小消息往返。完整推理预检包含 PNG 编码、传输、模型推理和一步仿真，更接近实际等待时间。不要把服务器上的 161 ms 推理耗时当作笔记本的实测延迟。`--record` 使用本地 SSD；每步保存前后两路 RGB，未压缩数据约 1.8 MB/步，长时间采集应检查磁盘容量，结束后再迁移 NAS。

## 接管数据如何进入后续 Stage 2

本次交付先封闭接管验证链路：本地决定并执行动作，然后保存实际发生的 transition。`metadata.json` 记录环境接口，逐步 `.npz` 保存前后图像、qpos、实际动作、原始 reward、terminated/truncated、episode、model_id 和 `human_intervention`。只按过接管键但没有执行成功，不会生成“人工动作”标签。

新增的 `--online` 模式已将实际 transition 上传给独立的单步 learner，具体命令和边界见 [单环境在线验收](VR_ONLINE.zh-CN.md)。这些文件仍不能直接送入正式 baseline 的 10 步 replay。后续与正式 Stage 2 合并时必须继续验证：

1. 在 takeover 边界切分 chunk，为不足 10 步的片段提供有效步数和 mask，不能补假动作。
2. 用同一冻结模型生成起止 RLT 特征；以实际执行动作构造 replay，保留逐步人工标记及 policy 版本。
3. 对齐 reward 累积、discount、success/termination、time-limit bootstrap 和 reset 后的 final observation。
4. 在 learner 端接收已执行 transition，确认实际训练路径消费它；加入人工样本的采样或模仿损失时单独做对照。
5. 完成 Windows/PICO 实机验收，再讨论合入 baseline。`server.py` 仍仅推理，新增 `online_server.py` 才包含在线更新；服务器脚本接管 smoke 不能替代真实手柄验收。

## 备用 Ubuntu 主机与验收范围

Ubuntu RTX 3060 12 GB 可以复用这套 CPU 单环境客户端和 SSH 推理协议；先保持相同仿真版本和接口，不能因换机器就改变控制语义。但 PICO Business Streaming 的 PC 端是 Windows 软件，不能承诺头显连接会自动迁移到 Ubuntu。届时还需验证 Linux 可用的 VR runtime，或增加 Windows 到 Ubuntu 的独立输入桥接；本分支尚未实现后者。

测试记录见 [VR_VERIFICATION.md](VR_VERIFICATION.md)。合并前最低验收：Windows reset/step 与图像、六轴方向、夹爪保持与切换、模型执行途中接管、松手后重新推理、拔线/遮挡/断网停止、旧回复不生效、回合结束不可继续步进，以及录制动作和实际动作一致。整个链路只用于仿真。

核心代码位于 [toolkits/rlt_vr](../../toolkits/rlt_vr)：先读 `simulation.py` 和 `client.py` 理解本地控制，再读 `control.py` 的权限切换，最后读 `protocol.py`、`server.py` 和 `vr.py`。通信工具使用标准库 logging，避免让 Windows 客户端因日志依赖引入 Ray；服务器模型部分沿用 RLinf 日志。
