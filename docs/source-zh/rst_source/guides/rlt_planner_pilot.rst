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

如何解读结果与限制
------------------

第一轮 20 步分布式对照已完成：相同的 20 个评估初态上，无辅助为 5/20，planner 辅助训练后为 3/20。辅助训练期间完成任务 18/20，但没有转化为观测到的自主收益。外层 step 数一致，不代表 transition 数和 actor 更新数一致。接下来的 60 步对照保持相同配置，分别报告真实控制 tick、expert 动作和 learner 更新预算，不将训练时的成功当作学到的成功。

2026-10-06 的 CPU PhysX/MPLib 开发测试完成了 3/3 初态求解、6/6 已抓取接手、6/6 受控掉落恢复；上述 GPU reference 导出和额外 3 段录像也已完成。测试覆盖 Torch/NumPy 动作、实际动作 replay、来源标签、reference 泄漏、终止冻结和 reset。分布式测试发现并修正了 NumPy 动作兼容及终止 transition 的 replay 字段不一致问题：planner 来源现在只保留在元数据，不进入 policy 观测。这些少量开发样本不能证明任意状态恢复能力或 learner 提升。分布式对照结果与撤掉 expert 后的自主收益仍须分别验收。

用 ``failure_before_actor_phase``、``failure_after_actor_phase``、``success_before_actor_phase`` 和 ``success_with_actor_phase`` 定位 baseline 的瓶颈。planner 辅助的接近阶段数据可以增加训练状态覆盖，但不会改变评估门控；从未负责接近阶段的 actor，不能直接纠正冻结 reference 在该阶段的失败。自主成功率、planner 步数与尝试次数、失败类别和额外 expert 数据预算需要分别报告。不要用正式评估失败继续训练后，再把同一批初态当作未见测试。
