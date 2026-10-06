从公开模型验证 LIBERO 上的 RLT 变体
====================================

本页说明如何在独立科研分支验证公开的 LIBERO RLT 变体，并区分模型复评、训练链路验证和原论文复现。先用相同任务初态比较冻结 VLA 与公开 learner，再考虑重新训练。现有 ManiSkill baseline、planner 和 VR 分支均不被替换。

为什么先选择 LIBERO
--------------------

要快速建立可核对的参照，代码、权重、动作定义和评估协议必须同时可用。截至 2026-10-06，本分支选择 AlphaBrain 的 QwenOFT + ``RLT_a`` 公开模型作为第一项验证，不把迁移到另一仿真器本身当作解决性能问题。

相关工作中的“RLT baseline”并不完全相同：

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - 工作
     - 实际比较对象
     - 对本次复现的意义
   * - `eRLT <https://arxiv.org/html/2610.00913v1>`_
     - LIBERO / RoboTwin 仿真中的 RLT* 共享 DSRL latent-noise RL 框架；不是直接照搬 RLinf 的动作空间训练。
     - 可借鉴表征对照设计，但不能把它的结果视为当前 ManiSkill 配置的验收标准；尚未核实独立官方仿真代码入口。
   * - `RouteRLT <https://arxiv.org/html/2609.26467v1>`_
     - 冻结 SmolVLA，在 LIBERO Object 的 3 个任务比较阶段 specialist 和控制权路由。
     - oracle 路由 RLT 为 91.11±0.96%，RouteRLT 为 92.22±2.55%；是论文报告而非本机复测，且不说明 VR 是必要条件。
   * - `AlphaBrain <https://github.com/AlphaBrainGroup/AlphaBrain>`_
     - 同时提供完整 token 的 RLT 与 action-query 压缩版 RLT_a。已核实的公开权重属于 QwenOFT / RLT_a。
     - 模型和评估源码可直接核查，适合先复评；不能将 RLT_a 的分数标成 RLinf pi0.5 RLT 的复现分数。

这里的 RLT_a 将 action-query 特征压缩到 256 维；公开 actor 直接输出 8×7 动作，并非结构上保证零初始化等于 reference 的残差 actor。RLinf ManiSkill 的 backbone、阶段门控、奖励和控制器均不同。原 ManiSkill Stage 2 下降仍需用匹配初态、actor 实际介入比例、动作归一化和 BC 能力检查来定位，不能归因于“没有 VR”。

准备独立代码与模型
------------------

本地目录为 ``UPT_libero_dev``，Git 分支为 ``research/rlt-libero-reproduction``。适配器固定外部源码版本和两个 Hugging Face 模型版本，不调用上游默认的多 GPU shell 启动器，也不运行其按用户名清理进程的命令。

在有 Git 权限的机器上准备源码；外部 Git checkout 应放在支持常规文件权限的本地盘，不放在本机 CIFS 网盘。

.. code-block:: bash

   export ALPHABRAIN_SOURCE="$HOME/rlinf_rlt/third_party/AlphaBrain"
   git clone https://github.com/AlphaBrainGroup/AlphaBrain.git "$ALPHABRAIN_SOURCE"
   git -C "$ALPHABRAIN_SOURCE" checkout --detach 604924beb77b04b0da49326dfae6ea423a27d28a

已有该目录时跳过 clone；检查版本，不覆盖本地修改。Python 环境需要 Torch、LIBERO 及仿真资源、MuJoCo/EGL、Transformers、Accelerate、FlashAttention、qwen-vl-utils 和上游 QwenOFT 依赖。CPU 单元测试或 ``check`` 通过不代表这些运行依赖全部通过。不要为此直接升级正在运行 baseline 的共享环境。

本机额外依赖放在 ``$HOME/rlinf_rlt/third_party/alphabrain-deps``，EGL loader 放在 ``$HOME/rlinf_rlt/third_party/alphabrain-egl``，没有修改共享 venv。

已准备好的 supermicro 可以直接使用 ``bash run_rlt_libero.sh check``。这个入口默认使用相邻 ``UPT_dev/.venv``、上述独立依赖目录和 NAS 上已下载的模型；不会安装软件或启动训练。运行时传入 ``evaluate`` 或 ``train`` 及后文参数，指定新的结果目录。其他机器可设置 ``RLINF_VENV``、``RLT_EXTERNAL_ROOT``、``RLT_ALPHABRAIN_SOURCE`` 和 ``RLT_LIBERO_ASSETS``。不使用脚本时，等价的环境设置为：

.. code-block:: bash

   export PYTHONPATH="$HOME/rlinf_rlt/third_party/alphabrain-deps${PYTHONPATH:+:$PYTHONPATH}"
   export LD_LIBRARY_PATH="$HOME/rlinf_rlt/third_party/alphabrain-egl/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
   export __EGL_VENDOR_LIBRARY_FILENAMES="$HOME/rlinf_rlt/third_party/alphabrain-egl/nvidia.json"
   export TMPDIR=/dev/shm
   export PYTHONDONTWRITEBYTECODE=1

其他机器应使用自己的系统 EGL，不要照抄不存在的本地 loader 路径。2026-10-06 已用当前共享 Python 环境完成 LIBERO Goal task 0 的 reset 和 10 tick 渲染交互；这不等于模型评估通过。

模型存放在容量充足的结果盘。命令固定下载 revision，不默认使用最新权重，也不下载本项目的 ManiSkill 数据。

.. code-block:: bash

   export LIBERO_RLT_STORAGE=/path/to/large-disk/libero-rlt/assets
   python -m toolkits.rlt.libero_reproduction download \
     --source "$ALPHABRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE"
   python -m toolkits.rlt.libero_reproduction check \
     --source "$ALPHABRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE"

VLA 来自 ``AlphaBrainGroup/qwenoft-5traj-libero-goal``，revision 为 ``91ea0817f556bf50d64008187519de19458f8188``；learner 来自 ``AlphaBrainGroup/alphabrain-rlt-5traj-alltasks-libero-goal``，revision 为 ``b5c0080e46f4b900c049e76bd9f3df68befac12a``。后者默认使用 ``rl_offpolicy_iter_00400``。``check`` 只核对源码版本与必需文件，不加载模型、不占 GPU。

先做配对评估
------------

空闲 GPU 上先跑同一任务的两个初态，验证模型加载、环境交互和录像。输出目录必须尚不存在。

.. code-block:: bash

   python -m toolkits.rlt.libero_reproduction evaluate \
     --source "$ALPHABRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE" \
     --gpu 2 --tasks 0 --states 0 1 --video \
     --output /path/to/new/libero-release-smoke

入口只暴露指定 GPU，并在启动前拒绝显存占用超过 512 MiB 的卡。CUDA 通过物理 UUID 选择设备，EGL 通过同一 UUID 查找渲染索引；两种编号不一致时不会默认使用 GPU 0。环境进程只需 CPU 物理与 EGL，因此隐藏其 CUDA 设备，避免 robosuite 混淆两个编号空间。无法唯一匹配设备时直接报错。

两组串行共享冻结 VLA，均使用作者的归一化、夹爪处理、8 tick chunk、10 tick 稳定等待及任务时限；reference 组保留 VLA 输出数值，仅转成 NumPy 支持的 float32，learner 组加载公开 encoder/actor。评估没有 planner 或真人辅助。视频由作者评估函数保存，不含最初的稳定等待。适配层也把环境进程的文本日志分流到 stderr，避免污染 stdout 上的二进制协议；没有改写任务动力学或奖励。

``reference.json``、``rlt_a.json`` 按 task/state 保存结果；``summary.json`` 额外统计“仅 reference 成功”和“仅 learner 成功”的配对差异。只有全部执行成功才将 ``manifest.json`` 的 ``complete`` 设为 true。任务 ID 范围为 0..9，初态 ID 为 0..49，重复或越界 ID 会报错，不做取模重复。

两个 episode 只用于链路验证。随后用 ``--tasks 0 1 2 3 4 5 6 7 8 9`` 和明确的 ``--states`` 列表扩大评估。全套 10×50 是公开 benchmark 状态复评，不应宣称这些初态从未用于公开模型训练，也不能把挑选某个任务后的结果外推为全套成功率。

2026-10-06 本机的第二次 smoke 已完成：LIBERO Goal task 0，初态 0、1，seed 42，reference 与公开 RLT_a 均为 2/2，四段 MP4 已保存。结果位于 NAS 的 research/libero_reproduction_20261006/release_smoke_v2，索引见 experiments/libero_rlt/validation_20261006.json。这只能证明公开权重、动作处理与环境闭环通过，不能证明 RL 增益或复现作者 10×50 分数。首次运行因 EGL 编号与物理 GPU 编号不同而主动停止，修正后使用 UUID 匹配；中断运行不计为通过。

再启动受控训练
--------------

2026-10-06 已完成 10 个任务、每任务两个初态的对照：reference 为 13/20，公开 RLT_a 为 11/20；共同成功 11 个、仅 reference 成功 2 个、共同失败 7 个。结果没有证明 RL 增益，下面的训练用于检查集成与学习过程，不代表已经找到有性能优势的配方。

评估继续使用未修改的上游源码。训练另用经过审查的运行时补丁，修复持久化 worker 的 UUID/EGL 映射、转 NumPy 前的 float32 转换、worker 错误传播和逐轮指标保存。从 UPT_libero_dev 目录执行以下命令，一次性创建训练 worktree 并应用补丁：

.. code-block:: bash

   export RLT_ALPHABRAIN_TRAIN_SOURCE="$HOME/rlinf_rlt/third_party/AlphaBrain_rlt_runtime"
   git -C "$ALPHABRAIN_SOURCE" worktree add --detach \
     "$RLT_ALPHABRAIN_TRAIN_SOURCE" 604924beb77b04b0da49326dfae6ea423a27d28a
   git -C "$RLT_ALPHABRAIN_TRAIN_SOURCE" am \
     "$PWD/experiments/libero_rlt/patches/0001-fix-libero-propagate-rollout-faults-and-respect-expl.patch" \
     "$PWD/experiments/libero_rlt/patches/0002-fix-libero-bound-rollout-lifetime-and-publish-small.patch"

已经应用两个补丁的训练 worktree 跳过这组命令；只有第一个补丁时，仅应用第二个。适配器要求干净的 Git tree 为 ``7abd269c822569642f94f79948e4b0869e6d25d5``；``git am`` 后的 commit 时间可以不同，实际 commit 会写入 manifest。启动脚本只在 ``train`` 模式选择这个目录；可用 ``RLT_ALPHABRAIN_TRAIN_SOURCE`` 修改路径。

训练入口使用同一个冻结公开 encoder，但重新初始化 actor 和 critic；不是从公开 step 400 恢复 optimizer，也不是完整 token RLT。先运行 20 次外层迭代的小实验。

.. code-block:: bash

   python -m toolkits.rlt.libero_reproduction train \
     --source "$RLT_ALPHABRAIN_TRAIN_SOURCE" --storage "$LIBERO_RLT_STORAGE" \
     --gpu 2 --tasks 0 --iterations 20 \
     --output /path/to/new/libero-train-pilot

预算为单 GPU、2 个环境、每轮 4 回合、5 轮 VLA 热身、batch 128、每轮最多 128 次 TD 更新、replay 容量 20,000。默认不启用 W&B 或并发评估，只在最后保存 checkpoint；这些小预算不是作者的完整训练配方，也不保证收敛。入口在结束时检查缺轮、零新增环境步、非有限标量、热身后没有 actor 更新，以及缺少最终权重等问题。``metrics.json`` 只表示进度；只有完整 manifest 才表示指定运行完成。

训练后用同一配对评估入口，并通过 ``--learner-dir /path/to/checkpoints/rl_offpolicy_iter_00020`` 指定新的 encoder/actor。独立评估前不能把训练退出码 0 解释为性能提高。CLI、来源版本和实验元数据会随结果保存；训练级复现还需要扩大样本、匹配预算和重复训练种子。

每轮 128 次只是上限，不是实际梯度更新数。更新量为 ``new_transitions × utd_ratio / batch_size`` 向下取整，至少一次且不超过上限。上游的 actor 延迟计数在每个外层迭代重新开始：一次 critic 更新、延迟参数为二的轮次不会更新 actor，但仍会记录 ``actor_loss`` 占位值。第二个补丁改用全局延迟计数并直接记录 optimizer 步数；适配器只对旧版记录重建预算，并标明来源。不能把 100 个外层迭代理解成 12,800 次 critic 更新，也不能仅凭 loss 字段存在认定 actor 更新过。

第一轮 20 次迭代已用第一个补丁完成并保存权重，但暴露了两个额外限制：默认累计 500 次更新才发布权重，高于小实验的总更新量，导致热身后的 rollout 权重长期不变；退出时也关闭了仍在采集多余第 21 批的 worker。第二个补丁只生成所需批次，并在关闭 socket 前确认采集线程退出。适配器设置 ``RLT_LIBERO_SYNC_UPDATES=1``，发布与完整采集 pass 共用锁。这是明确的小规模运行修正，不是未修改的上游运行。队列中的 100 轮将使用第二个补丁，不能把它与原 20 轮的差异全部归因于训练时长；新版本的 GPU 验收仍需等待该运行的 manifest 和计数。

运行有时限的夜间队列
--------------------

本机计划 ``experiments/libero_rlt/overnight_20261006.json`` 串行安排八项任务：20 轮训练验收与配对评估、两组 60 步 planner 对照及报告、公开模型 100 个配对样本评估，最后是 100 轮从头训练及其评估。较长训练以短实验完整执行为前提，不预先保证性能提升；它使用相同初始化重新训练，不伪装成恢复了 optimizer。

队列只接受物理 GPU 2。即使 GPU 0/1 在保存期间看起来空闲，也继续留给已有 baseline。连续两次确认 GPU 2 空闲才启动任务，每项有超时，总时限为 12 小时且包含等待；前置任务失败会跳过依赖任务，只向自己创建的进程组发送结束信号。队列不能锁住其他外部程序，启动器还会再次检查设备；GPU 2 一直繁忙时不能保证启动。复用这个带日期的计划前，先复制并将结果路径改为新目录。

.. code-block:: bash

   python -m toolkits.rlt.experiment_queue \
     --plan experiments/libero_rlt/overnight_20261006.json \
     --output /path/to/new/queue --check
   python -u -m toolkits.rlt.experiment_queue \
     --plan experiments/libero_rlt/overnight_20261006.json \
     --output /path/to/new/queue

第一条命令不分配 GPU；第二条保存 ``plan.json``、原子更新的 ``status.json`` 和逐任务日志。声明了实验 manifest 的任务，必须同时退出码为 0 且 manifest 完整才算通过。训练仅在末轮保存 checkpoint，只为最后的 task 0 对照启用视频。等待、超时和跳过都不会标成已完成实验。

复现边界
--------

本分支没有把外部模型注册成 RLinf 模型类型，也没有将 ManiSkill checkpoint 转成 LIBERO 模型。它先提供可审计的外部公开模型复评入口，以及受控训练适配。独立评估分数只以成功完成的本机结果文件为准；上游宣称的 benchmark 分数不能填作本机结果。后续完整 token RLT 需要另外准备匹配的 Stage 1 encoder，不能混用 RLT_a 权重。

2026-10-06 核对的官方 RLinf main 为 ``c70606f08cdca259b8dec03d4430926b5b8fac9d``。未合并的 `PR #1623 <https://github.com/RLinf/RLinf/pull/1623>`_ 涉及 replay checkpoint 保留范围，`PR #1527 <https://github.com/RLinf/RLinf/pull/1527>`_ 涉及 RLT 阶段路由；这里只记录相关性，没有自动合并到本地 baseline。
