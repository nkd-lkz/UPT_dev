验证规划器辅助的 RLT
======================

本页帮助你先验证仿真规划器能否提供正确记录的 RLT 纠正数据，再开展辅助训练对照。流程分为规划器可行性、离线 actor 验收、在线接入和自主评估四步。这个科研分支保留原无辅助 baseline；planner 接管不计作真人干预。

什么时候切换控制权
------------------

原 reference policy 先负责抓取和搬运。baseline 的自动门控要求：连续抓稳、任务尚未成功、peg 头在孔坐标系中的 x 不小于 -0.16 m，且 y、z 的绝对值分别不超过 1.5 倍孔半径。进入后一般锁存到回合结束。rollout 还检查 learner 是否完成热身：正式 baseline 要求收集后至少更新 30,000 次，不能把它理解为 30,000 个环境步。

新增加的是另一层仅用于训练的接管判断。它在 action chunk 边界读取当前和过去的执行证据，默认条件如下。

.. list-table::
   :header-rows: 1

   * - 触发原因
     - 判断与恢复方式
   * - 接近阶段超时
     - 180 个控制 tick 后仍未进入关键阶段：必要时重新抓取，然后对齐、插入。
   * - 丢失抓取
     - 先前观察到抓住，随后失去抓取：只在支持的工作区内尝试重新抓取。
   * - 插入停滞
     - 已进入关键阶段且仍抓住，x 减横向误差的得分连续 30 tick 未改善 0.003 m：退到预插入位置，重新对齐、插入。

控制频率为 10 Hz，因此两个等待预算分别对应 18 秒和 3 秒仿真时间，不是包含模型推理的墙钟时间。每回合最多尝试一次，planner 最多执行 300 tick。状态无效、控制器不兼容、恢复中再次失去抓取、规划失败或预算耗尽不会无限重试。恢复失败后记录原因并交还控制权，原环境的终止和超时规则保持生效。碰撞本身不是失败标准，也没有回退仿真时间。

先确认规划动作确实可执行
------------------------

规划器使用仿真物体位姿和几何真值，但动作仍通过现有的归一化 ``pd_joint_delta_pos`` 控制器执行。每步用规划关节目标减去真实当前关节位置，再除以 0.1 并截断至动作范围。规划器内部不偷偷调用 ``env.step`` 或 reset。

激活共享环境后，在新分支目录执行少量训练侧初态验证。即使不输出相机画面，SAPIEN 仍需要有效的 Vulkan loader/ICD；本次 CPU 测试使用 Lavapipe。输出目录必须尚不存在。

.. code-block:: bash

   python -m toolkits.rlt.planner_probe \
     --output /path/to/new/planner-reset --seeds 2026 2027 2028
   python -m toolkits.rlt.planner_probe \
     --output /path/to/new/planner-held --case held_recovery \
     --seeds 2026 2027 2028 2029 2030 2031
   python -m toolkits.rlt.planner_probe \
     --output /path/to/new/planner-drop --case dropped_recovery \
     --seeds 2026 2027 2028 2029 2030 2031

后两组先真实执行一段规划器前缀。held 在即将插入时换成从当前状态新建的规划器；drop 先真实张开夹爪 15 tick，再重新规划。这些是受控恢复案例，不是从 baseline 失败分布抽取的测试。NPZ 保存实际命令、关节前后状态、奖励、终止标志和动作来源；扰动动作不作为 expert 标签。``--video --gpu 2`` 需要空闲的渲染 GPU，入口自动选择该卡的 PCI 地址。GPU 录像测试在额外训练侧 seed 2032、2033、2034 上完成 3/3 已抓取接手，分别执行 128、123、129 tick，并保存了包含真实前缀的 MP4。

再用已有演示检查 actor
----------------------

actor 的输入除了 RL token 和本体状态，还包括独立预测的 VLA reference chunk。必须从动作执行前的观测生成 reference，不能把演示答案填进输入。等待指定 GPU 空闲后，先导出正确缓存，再在 CPU 上拟合。

.. code-block:: bash

   python -m toolkits.rlt.cache_actor_demos \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --output /path/to/new/actor-cache --gpu 2 \
     --episodes 0 1 2 3 4 5 6 7 8 9 10 11
   python -m toolkits.rlt.actor_bc \
     --cache /path/to/new/actor-cache --output /path/to/new/actor-bc --steps 500

拟合程序按完整 episode 做 3:1 划分，使用 actor 有界的确定性输出，不跨回合拼接动作 chunk，同时报告总体、手臂和夹爪误差。``best_actor.pt`` 只保存 actor 参数，不是可直接恢复的 Stage 2 checkpoint。离线误差不能认证闭环能力，也不会自动初始化在线 pilot。

旧 FLARE 缓存没有 reference 预测。显式设置 ``--reference-mode zero-diagnostic`` 只能做零 reference 消融，不能替代正常 actor 验收。新的 ``reference_cache_v1`` 已用 Stage 1 step 2000 导出真实动作前预测：9 个训练 episode、3 个验证 episode，500 次更新将验证 MSE 从 0.191056 降至最佳 0.025444。不过，同一验证集上直接执行 reference 的动作误差为 0.015863，仍优于小 actor。延长到 3,000 次更新仅将最佳值改善到 0.025000，最后一次验证误差反而回升至 0.029288。当前不能把离线拟合成功当作通过 actor 能力验收，更不能据此直接放大训练。

单独检查 reference 蒸馏
-----------------------

要区分“模仿演示”和“模仿 reference”，可在同一缓存上设置 ``--target-source reference``，仅替换监督目标，不改变输入来源。同样的 episode 划分下，3,000 次 CPU 蒸馏更新得到针对 reference 的最佳验证 MSE 0.010878。这里的目标与上面的演示 MSE 不同，不能直接比较两个数字，也不能据此将 actor 用作 Stage 2 初始化或声称自主成功率提高。

.. code-block:: bash

   python -m toolkits.rlt.actor_bc \
     --cache /path/to/new/actor-cache --output /path/to/new/reference-distill \
     --steps 3000 --target-source reference

最后开展匹配的在线小实验
------------------------

完成前面的检查后，再比较关闭和开启 planner 的两组。它们共用 ``rlt_planner_pilot``：1 个训练环境、20 个外层训练 step、512 次热身更新，每个外层 step 最多更新 128 次。无辅助评估明确使用 12026 至 12045 共 20 个不同 seed，不是把同一初态重复 20 次；后续评估循环使用这一固定序列。这是工程验证配置，与正式 64 环境 baseline 不同，不能直接当作后者的匹配复现。解读分数前，必须确认 learner 已完成热身，并检查实际由 actor 控制的次数。

.. code-block:: bash

   python -m toolkits.rlt.planner_experiment --check --gpu 2 --arm planner \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --output /path/to/new/stage2-planner

GPU 和存储条件满足后，去掉 ``--check`` 才会启动。无辅助对照使用 ``--arm none`` 和另一个新输出目录。``--steps`` 默认 20，``--eval-episodes`` 默认 20；两组必须使用相同参数。入口不仅设置 CUDA mask，还显式设置 RLinf 的物理硬件 placement，拒绝繁忙 GPU，并创建独立 Ray head。临时文件默认放在 ``--scratch-root /dev/shm``，至少需要 4 GiB；结果盘至少需要 5 GiB。退出只清理自己启动的进程，不停止别人的 Ray 集群。评估同时关闭 planner 和可选模型 expert。

两组完成后用报告入口核对实际配置、独立 seed 数和热身状态。尚未完成热身的 reference 主导运行不会被报告为两种已训练 actor 的对照。

.. code-block:: bash

   python -m toolkits.rlt.planner_report \
     --none /path/to/completed/stage2-none \
     --planner /path/to/completed/stage2-planner \
     --output /path/to/new/comparison.json

planner 的实际动作与来源标签会经过 transition 传输进入 replay，作为 BC 监督和 critic 的实际动作；原先的动作前 reference 保持不变。``planner_mask_ratio`` 与 ``bc_planner_loss`` 单独统计，不计入真人指标。恢复失败的真实 transition 仍保留，不改写成成功。

拆开前段失败与插入能力
----------------------

完整任务评估仍是主指标。新增的插入诊断在 reset 后真实执行规划器前缀，确认物体已抓住、接近孔口且尚未成功，再关闭后续 expert 动作并评估 checkpoint。不回退状态，也不重置回合计时；前缀消耗原有 500 tick 时限，成本单独记录为 ``fixture_prefix_ticks``，之后的 policy 执行计为 ``insertion_policy_ticks``。起点不符合条件就让本次测试失败，不能悄悄丢掉这个 seed。

对同一份权重、同一组明确初态，分别输出到两个新目录：

.. code-block:: bash

   python -m toolkits.rlt.planner_experiment --gpu 2 --arm none \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --checkpoint /path/to/final/model.pt --eval-episodes 50 \
     --eval-scope full_task --output /path/to/new/full-task
   python -m toolkits.rlt.planner_experiment --gpu 2 --arm none \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --checkpoint /path/to/final/model.pt --eval-episodes 50 \
     --eval-scope insertion --output /path/to/new/insertion

独立权重评估绕过 learner 热身门槛：完整任务仍沿用环境原有阶段门控，插入诊断则从起点就请求 actor。这是在测这份权重，不是验收热身状态。后者只能称为插入子任务结果，不能算完整任务自主成功率。前缀使用仿真几何真值，生成的状态分布也未必等同于 reference 失败时到达的状态。

检验纠正动作是否被 actor 利用
-----------------------------

在新的训练命令加 ``--export-corrections``，保存实际进入 replay 的 transition，包括动作前原始 reference、提交给环境的命令、终止掩码和 planner 来源。没有有效 transition 的回合也保留清单记录。仅在采集进程成功结束后封存缓存；拟合前校验哈希和冻结特征协议，中断采集的缓存不能直接使用。诊断器只针对已核实的 Panda 控制器，保留原命令并另行还原控制器裁剪至 [-1, 1] 后的归一化命令，报告裁剪计数，不改写封存缓存，也不改变在线 replay；其他控制器映射直接拒绝使用。

两种 actor 目标共用一份封存数据：

.. code-block:: bash

   python -m toolkits.rlt.correction_fit \
     --cache /path/to/completed/planner/corrections \
     --output /path/to/new/correction-fit \
     --updates 2000 --warmup-updates 128 --seed 1234

如果要检查已有 actor 是否能利用纠正，追加 ``--initial-weights /path/to/actor/model_state_dict/full_weights.pt``。两组严格加载同一份有限值权重，重新创建 optimizer 和 target，报告初始权重 SHA-256；这不是 resume。仍须保证该 actor 使用缓存声明的 Stage 1 特征及归一化数据。随机初始化与热启动属于不同实验，不能合并其 MSE。

CPU 诊断按时间顺序将完整回合以 3:1 分为训练和验证集，两组共享初始化、minibatch 索引和随机数。为匹配优化工作量，两组都训练 critic，只有 Q＋BC 在热身后将 Q 梯度用于 actor；每四次 critic 更新对应一次 actor 更新。BC 在成功回合的 planner tick 上使用实际纠正动作，在非 planner tick 上使用原 reference。失败 planner 动作保留给 TD，但不作 BC 标签；终止后的填充不计入 BC 或奖励和。成功回合筛选只是粗粒度质量规则，不保证每个动作都最优。现有在线 loss 不变，仍使用原来的 intervention 标签。

报告包含留出回合上的纠正、手臂、夹爪和 reference MSE，被拒绝的纠正 tick 数、命令饱和比例、数据与 minibatch 哈希及准确更新次数。合格标签要求导出的 transition 中出现正奖励；如果成功发生在未记录阶段，则保守地不视为合格纠正。按固定更新预算选取最终权重，不根据测试成功率挑模型。用前面的两种评估命令分别测试 ``bc_only/model.pt`` 和 ``q_bc/model.pt``，必须保持相同 Stage 1 特征与归一化数据。离线误差下降不等于闭环收益；这些权重用于评估，不是可恢复的分布式 checkpoint。至少需要四个非空回合，且训练、验证两边都有成功纠正；不足时补采训练侧数据，不能借用评估失败样本。

按真实交互预算开展多种子对照
----------------------------

旧实验匹配回合数，却让 planner 组执行更少环境 tick、更新更多次 learner。新入口因此支持二选一的 ``--control-budget`` 和 ``--update-budget``：前者计入所有真实训练 tick，包括 reference、learner、planner 和交还前的保持动作，并在最后一回合精确截断；后者限制累计 critic 更新，另外报告 actor 更新。达到预算时强制评估和保存权重。此时 ``--steps`` 仅作为回合数安全上限，先达到安全上限则报错。本协议只支持全新、同步、单环境训练，禁止 resume 和重叠 bootstrap。

campaign 默认只打印六条串行训练命令，加 ``--execute`` 才执行。每个训练结束后，还会单独评估最终权重的插入能力。不终止任何已有 GPU 任务；遇到占用会明确拒绝启动，不自动等待。

.. code-block:: bash

   python -m toolkits.rlt.planner_campaign \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" --gpu 2 \
     --seeds 1234 1235 1236 --eval-episodes 50 \
     --budget-basis control --budget 30000 --max-episodes 1000 \
     --protocol complete --output /path/to/new/control-budget-campaign

另开目录，用 ``--budget-basis updates --budget 2000`` 补充匹配优化次数的实验；只有 actor、critic 两种更新计数都一致，才能称为优化工作量匹配。报告保留每个训练 seed、成功率均值、训练 seed 间标准差、配对差值、失败阶段、planner 尝试与 tick 成本及更新数。两组使用相同的 50 个不同评估 seed：12026–12075，不将它们加入训练。三个训练 seed 与 50 个初态仍属于开发实验，不自动构成显著性或泛化证明。插入结果与完整任务对照分开保存。新入口显式设置环境和 actor 的训练 seed，不能与环境 seed 为 2026 的旧 pilot 悄悄合并。

将有限纠正后交还作为独立协议
----------------------------

默认 ``--protocol complete`` 保留原规划器完成剩余任务的行为。新增 ``--protocol preinsert_handoff`` 在重新抓取、对齐后，正式插入前停止；要求仍抓住、尚未成功、孔坐标系 x 不小于 -0.16 m、横向距离范数不超过 0.045 m。不承诺从任意接触状态恢复，触发条件和 300 tick 的规划动作预算仍然受限。

如果规划动作在 chunk 中途结束或失败，监督器先保持当前手臂位置，到下一个 chunk 边界再使用基于新观测生成的 policy 动作，不执行介入前缓存的剩余旧命令。保持最多占一个 chunk，计入 planner 和真实交互成本，另记为 ``planner_handoff_hold_ticks``。``planner_handoffs`` 统计已完成的交还流程，不是自主成功次数。交给 reference 还是 actor 仍取决于原有门控和 learner 热身，因此仅看到 handoff 不能证明 learner 已执行。请另开 campaign 目录，在同一声明协议下与无辅助组比较，不混入旧的完整恢复实验结果。

用两张空闲 GPU 热启动训练
-------------------------

如果要检验纠正数据能否改善已有小 actor，请单独建立只继承权重的训练实验。双卡入口把 learner 放在物理 GPU 0，冻结 VLA、rollout 和单个 CPU 物理仿真环境放在 GPU 1。启动前要求两卡空闲，不占用 GPU 2；使用私有 Ray，并在启动前归档运行代码、建立本地源码快照。

选取已验证可读的 Stage 2 ``actor/model_state_dict/full_weights.pt`` 和新的输出目录。在 tmux 会话中执行以下命令，使断开终端后训练继续运行：

.. code-block:: bash

   python -m toolkits.rlt.planner_stage2 \
     --stage1 "$RLT_STAGE1_ACTOR" --dataset "$RLT_DATASET_DIR" \
     --initial-weights /path/to/stage2/actor/model_state_dict/full_weights.pt \
     --output /path/to/new/planner-stage2 \
     --steps 400 --eval-interval 50 --eval-episodes 50 --save-interval 100 \
     --wandb-entity YOUR_ENTITY

追加 ``--check`` 只在 CPU 上验证配置和初始 actor/critic 权重，不启动训练。正常运行会启用在线 W&B 与 TensorBoard，在输出目录保存 ``launch.json``、``resolved.yaml``、``source.tar.gz``、初始权重和 ``train.log``。高频 backend 写入使用单独的本地持久目录 ``$HOME/rlinf_rlt/run_metrics/<output-name>``，避免远程文件系统阻塞；可用 ``--metrics-dir`` 修改。解析后的 ``runner.logger.backend_log_path`` 和 manifest 都记录该位置，planner 报告会从这里读取指标。请同时保留这个目录和 checkpoint 输出，为两处输出预留空间；默认关闭视频。私有 Ray 默认从 6575 端口开始，可用 ``--port`` 指定另一组三个连续空闲端口。

这不是原无辅助 baseline 的完整续训：只继承 actor/critic 权重，重新建立 replay、optimizer、target 初始化及计数器；target 从同一份权重初始化。热身先收集 256 条 replay 样本，再执行 4096 次 critic 更新，每轮最多更新 128 次；actor 权重调度也使用 4096 次更新的热身和渐变阶段。训练使用完整恢复的 planner 协议，单独记录其成本。每 50 轮在 seed 12026–12075 上关闭 planner 和模型 expert 评估，每 100 轮及训练结束时保存。learner 尚未就绪时的评估仍回退到 reference，必须结合就绪状态解读成功率。在这组开发初态上选定 checkpoint 后，最终结果还须使用未参与选模的初态检验。

如何解读结果与限制
------------------

第一轮 20 步分布式对照中，无辅助为 5/20，planner 辅助训练后为 3/20；之后的 60 回合对照分别为 4/20 和 7/20。但两组真实训练 tick 为 24,091 与 14,203，actor 更新为 173 与 257 次，辅助组另用了 4,392 个 planner tick。这些单种子、工作量不匹配的小实验不能证明可靠自主提升。上述预算协议是新的实验，不能沿用旧分数作为其有效性证据。

2026-10-06 的 CPU PhysX/MPLib 开发测试完成了 3/3 初态求解、6/6 已抓取接手、6/6 受控掉落恢复；上述 GPU reference 导出和额外 3 段录像也已完成。测试覆盖 Torch/NumPy 动作、实际动作 replay、来源标签、reference 泄漏、终止冻结和 reset。分布式测试发现并修正了 NumPy 动作兼容及终止 transition 的 replay 字段不一致问题：planner 来源现在只保留在元数据，不进入 policy 观测。这些少量开发样本不能证明任意状态恢复能力或 learner 提升。分布式对照结果与撤掉 expert 后的自主收益仍须分别验收。

用 ``failure_before_actor_phase``、``failure_after_actor_phase``、``success_before_actor_phase`` 和 ``success_with_actor_phase`` 定位 baseline 的瓶颈。planner 辅助的接近阶段数据可以增加训练状态覆盖，但不会改变评估门控；从未负责接近阶段的 actor，不能直接纠正冻结 reference 在该阶段的失败。自主成功率、planner 步数与尝试次数、失败类别和额外 expert 数据预算需要分别报告。不要用正式评估失败继续训练后，再把同一批初态当作未见测试。

2026-10-07 的开发验收确认了 220 和 3,000 个真实控制 tick 的精确停止。后者采集 12 回合、118 条进入 replay 的 transition，planner 占 827 tick；因为不足 128 条热身阈值，在线 learner 没有更新。在这份封存数据上，两组 CPU 拟合各做 500 次 critic、125 次 actor 更新，用 8 个非空回合训练、3 个回合验证。验证纠正 MSE 从初始化的 0.17130 降到 BC-only 的 0.07227、Q＋BC 的 0.07894；但手臂单独 MSE 反而由 0.05263 升到 0.06772、0.07641，整体改善主要来自夹爪。验证集中只有 65 个合格 planner tick，不能据此验收 actor 能力或宣布自主提升。另一次单回合插入烟测分别记录了 83 步前缀和 417 步 policy 执行，证明的是评估口径分离，不是性能。

之后，两份拟合权重在 seed 12026、12027 上均完成了 2/2 插入子任务，前缀结束后没有 planner 动作。两组相同的平均前缀为 87 tick，BC-only 平均再执行 26 tick，Q＋BC 为 21 tick。真实物理回归中，保持手臂零增量、夹爪闭合的固定命令在三个起点均未完成任务（seed 12026–12028）。这些只是少量集成验收，不能说明 Q 提高成功率，更不能证明完整任务提升。三训练 seed 的公平长实验尚未运行。代码验收包括 37 项 planner 测试全部通过（含真实物理），扩展 CPU 回归 240 项通过、12 项跳过；另有一项在未修改 baseline 也能复现的 Transformers 视频元数据兼容失败，已从扩展回归中单独排除。中英文文档构建均通过。

晚间接续：固定权重验收与有限纠正
--------------------------------

2026-10-07 的 ``planner_warm1100_1007_150132_retry2`` 已完成 400 回合：实际训练 102,648 tick，其中 planner 31,819 tick，尝试 281 次、失败 83 次，critic 更新 8,986 次、actor 更新 2,247 次。辅助训练成功 327/400 不等于自主能力。固定 50 seed、关闭 expert 的各次评估在 30%–42% 波动，最终 18/50；其中 28/50 在 actor phase 前失败、4/50 在其后失败。还没有相同协议下的初始化 checkpoint 复评，不能将这轮训练称为提升。

同晚从 baseline step 1100 权重开始，在前述 3,000-tick 缓存上完成一组配对诊断：各 2,000 次 critic、500 次 actor 更新。验证纠正 MSE 为初始 0.12085、BC-only 0.11575、Q＋BC 0.21194；手臂 MSE 分别为 0.13782、0.13126、0.23839。只涉及 3 个留出回合、65 个合格纠正 tick；表明该设置下 Q 项可能破坏跟随，尚无闭环结论。不要据此直接选择更多在线 Q 更新。

随后以优化器／批采样 seed 1234、1235、1236 重复诊断，BC-only 的纠正 MSE 分别为 0.11575、0.11504、0.11902，Q＋BC 为 0.21194、0.16522、0.17791。三组共用同一数据和初始权重，不是三个独立在线训练 seed。排队的闭环对照使用重复实验前已选定的 seed 1234，没有按最低验证误差挑选权重。

从旧 replay 重建回合需要额外检查。``toolkits.rlt.correction_export`` 只接受完成、无淘汰、同步单环境的 transition checkpoint，按真实 done 分组并验证相邻端点；遇到非终止记录间隙则拒绝封存。上述 400 回合 replay 在 transition 109 与 110 之间出现间隙，无法可靠恢复整次回合；失败导出保留为未封存证据。这不证明已有 TD 端点错误，但不能猜测回合边界后用于跨回合验证。新实验始终在采集时用 ``--export-corrections`` 保存真实回合分组，包括空 replay 回合。

要扩大插入起点检查，可以只使用 CPU 物理后端：

.. code-block:: bash

   python -m toolkits.rlt.planner_probe --case insertion_fixture \
     --seeds 12026 12027 12028 --output /path/to/new/fixture-control

该 probe 执行 planner 前缀，检查仍抓住且未成功，再保持手臂零增量、夹爪闭合直到原回合结束；逐步保存实际动作。报告将 ``fixture_ready_rate`` 与 ``hold_success_rate`` 分开，不输出 learner 成功率。仍需要可用的 Vulkan loader；CPU 检查应使用隔离的软件 Vulkan ICD，不能借用其他用户正在使用的 GPU。

扩大到 seed 12026–12075 后，CPU probe 在 48/50 个初态建立有效起点；seed 12067 耗尽 300 tick 前缀预算，seed 12070 抓取规划失败。保持不动的续接均未成功。因此夜间插入诊断用 ``--fixture-report /path/to/fixtures50_cpu/summary.json`` 显式声明这 48 个条件初态，同时保留 ``--eval-episodes 50`` 表示候选总体。所有 checkpoint 使用相同 48 个初态，manifest 保存原始 50 个 seed、两项排除原因和 probe 哈希。筛选只依赖 learner 评估前的 planner 起点可行性；运行中若再次建立起点失败仍会中止，不会自动跳过。完整任务评估继续使用全部 50 个初态，不能把其分母换成 48。

单卡 pilot 现在也支持 ``--initial-weights`` 热启动训练、``--wandb`` 启用 W&B，以及 ``--checkpoint-run`` 选择已完成训练的最后预算 checkpoint。固定权重评估不再依赖暖身计数；checkpoint 与初始权重记录 SHA-256。每次运行归档包含评估入口的源码，并从私有快照执行。指标默认保存在本地 ``$HOME/rlinf_rlt/run_metrics/<output-name>_<path-hash>``，不同 seed 不共用事件文件；NAS 保留 checkpoint、解析配置和运行 manifest。

本机夜间计划 ``experiments/maniskill_rlt/overnight_gpu0_20261007.json`` 对初始化、planner400、BC-only、Q＋BC 分别跑 50 个完整任务和上述 48 个已验证起点的插入诊断。``overnight_gpu1_20261007.json`` 则以三个训练 seed 比较无辅助与 ``preinsert_handoff``，各使用同一初始权重和 15,000 个实际控制 tick，并开启 W&B；主要对照完成后补充插入评估。两者均为全新实验，不沿用先前分数。

等待执行使用 LIBERO 分支中的有限队列：

.. code-block:: bash

   python ../UPT_libero_dev/toolkits/rlt/experiment_queue.py \
     --plan experiments/maniskill_rlt/overnight_gpu0_20261007.json \
     --output /path/to/new/queue-gpu0 --check

去掉 ``--check`` 才启动等待。日期化计划包含本机路径，复用前须更换所有输出目录。队列只使用计划显式指定的物理 GPU，连续两次空闲才启动；按卡独立锁，12 小时总预算包含等待，逐任务另有超时，只清理自己创建的进程组。本次 tmux 为 ``rlt_planner_audit_1007`` 与 ``rlt_planner_handoff_1007``，NAS 的 ``research/planner_overnight_20261007/queue_gpu0_v2/status.json`` 和 ``queue_gpu1_v2/status.json`` 才是执行状态依据。原队列在尚未启动 GPU 工作、仍等待空闲时暂停，以补充起点覆盖率声明，旧诊断继续保留。汇总入口可用 ``planner_report --evaluation-runs ... --output ...`` 分开报告完整任务与插入，或用 ``--comparisons`` 汇总三个已完成训练配对。排队、通过配置检查、拟合误差改善均不能替代自主评估结果。

仓库内的 ``experiments/maniskill_rlt/overnight_results_20261007.json`` 汇总已完成证据，并单独指向队列实时状态。晚间版本通过 46 项 planner 测试（包含 CPU 物理），以及 223 项扩展 CPU 测试、6 项跳过；已有的 Transformers 视频元数据兼容用例继续明确排除。离线诊断中的数据质量掩码没有悄悄改变分布式在线 BC 规则。
