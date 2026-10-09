验收 Pi0.5 RLT Baseline 后再迁移环境
====================================

本指南说明如何在 LIBERO 上执行有预算限制的 AlphaBrain RLT/Pi0.5 开发实验，并固定后续迁移 RoboDojo 所需的动作接口。代码位于独立目录 ``UPT_rlt_pi05_dev``，分支为 ``research/rlt-pi05-libero-robodojo``。先验证公开 VLA，再验收新 RLT encoder 和小 actor，最后运行配对的在线学习试验；此前 Qwen/RLT_a 的实验保持独立。

确定权重与实验协议
-------------------

新路线冻结带有 flow-matching 动作专家的 Pi0.5 VLA。AlphaBrain 的 ``RLT/encoder_decoder.py`` 使用 encoder 和 cross-attention 重建 decoder，将 VLM 最后层特征压缩为一个 2048 维 token。小 actor 直接输出完整的 10×7 动作 chunk；在线更新不会训练 flow-matching 专家。

源码固定为 ``604924beb77b04b0da49326dfae6ea423a27d28a``。公开模型采用 `AlphaBrainGroup/paligemma-pi0-libero-all4suite <https://huggingface.co/AlphaBrainGroup/paligemma-pi0-libero-all4suite>`_ 的 ``482fcfea4d6a5cee97456f50b1f5aeeae539a1c8`` 版本。虽然配置保留旧类名，但设置了 ``pi05: true``。该模型联合微调过四套 LIBERO 任务，适合作为工程起点，不能称为少示范 RLT 权重，也不能据此宣称未见任务迁移。

适配入口显式映射旧类名，严格加载全部权重，保留 LIBERO 原生带符号夹爪动作，并且只应用一次模型内的 MEAN_STD 变换。encoder 所需的 512 个真实图像位置特征来自 reference 自己的推理前向，避免采集与部署使用不同 prefix。固定噪声的一致性检查要求提取特征前后的 reference 动作完全相同。

.. warning::

   当前是新 RLT 架构的工程试跑，不是未经修改的上游训练复现。Stage 1 使用 reference rollout 的观测，小 actor 热身关闭 reference dropout。小头复用已有验收 learner，但输入改为 2048 维、chunk 改为十步、夹爪判定改为带符号阈值。奖励乘五，并按实际执行的仿真 tick 折扣。这些选择写入 ``PROTOCOL``，结果必须与论文分数区分。

准备已校验的模型
-----------------

使用已准备的 LIBERO acceptance 环境，避免修改正在运行的旧任务的依赖。``run_rlt_pi05.sh`` 默认选择相邻目录的 ``.venv-libero-acceptance`` 以及独立的 EGL 和依赖目录；其他主机可通过 ``RLT_PI05_VENV`` 和 ``RLT_EXTERNAL_ROOT`` 指定路径。推理阶段关闭在线 W&B 和 Hugging Face 查询。

模型权重约 9.4 GB，应存放在大容量文件系统。下载器支持续传，核对 Git blob 或 LFS 哈希后才写入 ``verified.json``：

.. code-block:: bash

   export RLT_PI05_ASSETS=/path/to/large-disk/pi05-assets
   export RLT_PI05_SOURCE=/path/to/AlphaBrain
   python -m toolkits.rlt.pi05_assets --destination "$RLT_PI05_ASSETS"

源码目录必须处于指定 commit 且没有修改。推理只读取模型目录，不改写公开配置，也不会静默忽略缺失权重。

启动分阶段验收
---------------

队列取得研究任务锁后，会连续两次检查指定 GPU 的显存和计算进程，确认空闲才启动任务，不会停止其他用户的进程。选用新的输出目录，并明确指定物理 GPU 编号：

.. code-block:: bash

   export RLT_PI05_RUN=/path/to/large-disk/new-pi05-pilot
   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_campaign run \
     --root "$RLT_PI05_RUN" --assets "$RLT_PI05_ASSETS" \
     --source "$RLT_PI05_SOURCE" --gpu 2

整个队列最多执行十二小时，等待资源也计入预算；每个 GPU 任务最多两小时。前置验收失败会跳过依赖它的阶段，进程正常退出也必须结合结果 manifest 判断是否完成。

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 阶段
     - 预算与数据
     - 验收条件
   * - Reference 采集
     - Task 0 的公开初态 0–9，每回合最多 320 tick
     - 至少成功五次；特征提取不改变动作
   * - Encoder 与 BC 热身
     - 重建更新 300 次，BC 更新 3000 次；按完整回合划分 80/20
     - 重建误差和动作跟随误差下降；留出集夹爪不一致率不超过 10%
   * - 热身闭环检查
     - Reference 与 actor 配对测试初态 10–14
     - 初态哈希一致；reference 至少成功三次；actor 最多少成功一次
   * - 在线试跑
     - BC-only 与 Q+BC 从同一热身模型启动；使用初态 20–29；每个 chunk 更新五次
     - 数值有限，记录实际 actor/critic 更新次数
   * - 试跑比较
     - Reference、仅热身、BC-only、Q+BC 测试初态 15–19
     - 保留全部结果，包括零成功

这些小规模开发集用于发现接入错误，不等于此前的独立 frozen-200 复评、完整任务集验收或具有统计效力的提升证据。调参和报告研究结果之前，应另行保留独立测试集。

检查输出与状态
---------------

先读取 ``queue/status.json``，区分等待、失败和完成。每个任务的 ``manifest.json`` 记录源码、模型版本、适配代码哈希、依赖版本、seed、设备映射和退出状态；``episodes.json`` 记录初态哈希、成功标志和实际执行 tick。热身成功后生成 ``warmup/warmup.pt``，在线阶段正常执行后保存 ``q_bc/online.pt`` 和 ``bc_only/online.pt``。

以下命令生成简要报告，将尚无结果与完成后的零成功率区分开。队列退出时也会写入 ``summary.json``：

.. code-block:: bash

   ../.venv-libero-acceptance/bin/python -m toolkits.rlt.pi05_campaign status \
     --root "$RLT_PI05_RUN"

开发服务器上的当前实验输出位于 NAS 的 ``research/rlt_pi05_20261009/pilot-v2``，对应 tmux 会话 ``rlt_pi05_baseline_1009``。此前 ``pilot`` 队列在等待 GPU 时停止，以补入仅热身对照，原有记录保持不变。单独的 ``cpu-audit/manifest.json`` 记录严格加载和合成输入前向；CPU 检查不是 LIBERO 成功率。失败 manifest 与被跳过的阶段都会保留，不能将它们标为已完成训练。

没有空闲 GPU 时，可以用 ``cpu-reference`` 通过 CPU 推理和 OSMesa 软件渲染检查真实仿真闭环。若 OSMesa 解压在系统路径之外，需要设置 ``LD_LIBRARY_PATH``。该模式不会占用 CUDA 设备，必填的 ``--gpu`` 参数在此被忽略。CPU BF16 autocast 推理较慢，其结果不能替代配对 GPU 评估：

.. code-block:: bash

   bash run_rlt_pi05.sh cpu-reference --source "$RLT_PI05_SOURCE" \
     --assets "$RLT_PI05_ASSETS" --output /path/to/new-cpu-smoke \
     --gpu 2 --episodes 1 --state-start 49 --seed 45000

准备 RoboDojo 接口
-------------------

LIBERO 试跑通过后，先迁移观测和动作接口，再增加训练。`RoboDojo <https://github.com/RoboDojo-Benchmark/RoboDojo>`_ 当前公开的是 eval-only 仿真与任务框架，policy server 由 XPolicyLab 管理。本次核查版本分别为 RoboDojo ``cb2a37d58dbbeb2777b0c2b30e97ecb8ab91095c`` 和 XPolicyLab ``53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2``。

首个迁移目标采用 ARX X5 关节控制接口：头部及双腕 RGB 相机、14 维状态，以及以弧度表示的机械臂绝对关节位置和 [0, 1] 夹爪开度。本次检查的实现中，夹爪状态来自命令开度，并非独立测得的真实关节响应。LIBERO 的七维末端动作模型不能通过补零或改名变成该接口的模型。

``toolkits/rlt/robodojo_boundary.py`` 在不启动 Isaac Sim 的情况下测试 policy 侧接口。构造 ``RoboDojoBoundary`` 时需要兼容的推理 backend 和明确的归一化标识；每回合先调用 ``reset``，再通过 ``update_obs_batch`` 写入带环境 ID 的观测，最后用 ``get_action_batch`` 请求当前活跃环境的动作。返回值是按请求顺序排列的原生命令字典，环境批次缩小时也保持对应关系；旧观测、非法夹爪值和 LIBERO 维度的输出会被拒绝。该组件目前尚不加载 RoboDojo policy，也不包含仿真训练循环。

当前服务器的迁移快照保留了较早的源码和 bridge 原型，但缺少原机器的仿真环境与资产。后续验收顺序是固定 Isaac runtime 与单任务资产，准备适配该机器人的 Pi0.5 权重或执行监督适配，完成原生 reference 评估，再补齐包含 reset、reward、termination 和 truncation 的在线 transition contract。初始 baseline 关闭人工、planner 和 memory，后续引入时分别记录其成本。
