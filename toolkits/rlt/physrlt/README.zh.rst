PhysRLT 小规模开发实验
=====================

本说明帮助你运行 LIBERO-Pro 小规模实验，检查真实执行响应与几何变化监督能否改善动作侧表征。代码冻结 AlphaBrain 发布的 π₀.₅，读取已有动作专家前向计算的特征，先比较等容量表征，再开展短程 BC／Q＋BC 控制实验。离线预测证据与闭环控制结果分别记录。

固定实验计划
------------

本机配置复用已安装的模拟器和校验过的权重。在新工作区按 README.rst 的 prepare 命令建立独立输出目录，生成明确的任务、初态、数据划分和依赖关系。task 0 使用 object 扰动，task 3 使用 lan 扰动；每任务含 6 个训练、2 个验证和 2 个开发初态。这些公开初态曾用于 baseline 开发，本轮不能作为论文独立测试集。

计划首先执行 20 个控制步的真实环境采集及动作一致性检查，通过后每任务采集 10 回合、每回合最多 320 个控制步。隔一个 chunk 加入小幅平移噪声，用于识别命令—响应关系，因此采集成功率不能充当冻结 reference 的成功率。

运行和检查
----------

计划固定后，按 README.rst 的 tmux 命令启动独立 supervisor。它只使用 GPU 0、1，等待连续空闲 30 秒才启动；每个任务设卡死检测超时，整个有限实验队列没有 12 小时截止。单个任务失败会阻断其依赖项，其他独立任务继续。重启时若记录中仍有运行任务，必须先核对进程和输出。

通过 status.json 查看任务状态，stdout.log 排查错误，各任务 result JSON 确认完成。离线 table.csv 比较预测误差，control_summary.csv 区分 reference、热身、在线训练与冻结复评，并保留数据划分和分母。记录直接保存在本地，不依赖 W&B 或网络服务。

理解实验对照
------------

七种表征均使用相同的 128 维瓶颈和解码器容量：动作特征重建、几何监督、响应监督、联合监督、打乱联合标签、prefix 特征联合监督，以及只用当前本体与 reference 的联合监督。每种方法在 seed 31、47、59 上各训练 400 步，再使用统一的冻结线性探针，与“本体状态＋动作”和“状态不变”预测比较；探针正则参数只根据验证初态选择。

可选的 ``flow_recon`` 与 ``flow_joint`` 保持 512 维输入和 128 维瓶颈不变，但从冻结 Pi0.5 的 flow-matching 生成轨迹中等距读取四个 action-expert 状态，不再只读取最后一次 forward。它们要求新采集的数据显式包含 ``flow_feature``；旧数据会直接报错，不会静默替换为末端特征。该对照只检验动作生成轨迹表征，不能把去噪时间解释为物理执行时间。

四种表征进入控制实验，均训练 actor BC 1,500 步，使用 seed 31、47。它们共享 AlphaBrain actor／critic 结构与超参数。只有验证集手臂 MSE 小于 0.02、夹爪分歧小于 0.05，且至少成功一次，才追加最多 1,040 个在线控制步。不同方法通过门槛的情况不同时，只能作为开发诊断，不能声称完成公平的大规模 RL 对比。

本轮固定特征投影和 128 维表征与原始 2048 维 RLT encoder 不同，属于匹配的科研变体，不能写成原算法的精确复现。原 baseline 工作区保持不变。

物理量与因果边界
----------------

Encoder 只读取执行前特征、当前本体和 reference；后果解码器读取真正接受的命令前缀，预测第 3、10 个控制步的后果。未来测量只作为训练标签，未执行到的终止时刻被屏蔽。关节角、末端／夹爪位置、经相机标定转换的米制深度都有明确单位。物体目标采用世界坐标位置，排除“物体未动、末端移动也会变化”的物体—末端相对观测。

关节角和末端位置在理想标定下可由正运动学关联，但二者不作为独立信息源来宣称贡献。采集保留完整的 7 维机械臂关节、3 维末端位置和 2 维夹爪位置，便于检查模型误差、标定误差及控制器响应；训练另设完整响应、机械臂关节＋夹爪、末端＋夹爪三组同预算消融。``arm_q_gripper_joint`` 与 ``eef_gripper_joint`` 只改变响应监督维度，输入、128 维瓶颈、几何监督和更新数保持一致。

在把实测执行信息接入 policy 前，先运行 ``gain_probe`` 做无训练的可辨识性检查。它恢复同一个已记录 simulator state，发出相同命令 chunk，只改变真正执行的平移增益；分数是实测末端位移在发出命令方向上的投影。独立 development 门槛要求每个时间尺度至少 6 对样本、配对排序准确率不低于 75%，并且 3 tick 和 10 tick 的“正常增益减低增益”平均 margin 均为正。该检查只能证明已完成 transition 包含局部执行条件信息，不能证明控制收益，也不允许当前决策读取未来测量。

``response_adaptation`` 是下一步闭环筛选，不是最终 RLT 方法。它只根据此前已经完成的 chunk 估计有界平移修正，不读取配置中的真实执行增益。匹配 reference 不作修正；``fixed_high`` 不读取响应，始终采用最大修正；``delayed_high`` 同样不读取响应，但等待与 adaptive 相同的最少观测数后再采用最大修正；``oracle`` 只是读取增益并执行有界固定补偿的对照，而不是真正的最优 policy。``delayed_oracle`` 等待与 adaptive 相同数量的完整 chunk，再按声明的执行增益补偿，但不读取这些 chunk 的响应，用于区分响应估计收益与修正时机收益。``reversed`` 则故意沿错误方向使用响应估计。正常响应分数必须来自独立训练布局；先用两个 development 布局冻结设置，再通过显式 offset 评测剩余布局。

传入 ``--nominal-dataset`` 后，程序使用动作命令和当前本体状态，借助条件 ridge 预测标称的三维末端位移。ridge 强度只在 validation 布局上选择，构造特征前排除全部 development 和 confirmation 记录；若该预测器不能在 validation 上优于零位移基线，或预测位移与实测位移的平均方向余弦不为正，运行会直接失败。每个已完成 chunk 将实测位移与“预测位移乘以已知动作修正幅度”比较，分别估计 x／y／z 执行增益，不把修正后的动作重新送入回归器。``--execution-gains`` 可以施加声明过的三轴压力条件，adaptive 组则只能读取已经完成的响应。这些比率及其有界修正是待接入 RLT 的诊断性 Physical Token，不能单独作为任务收益结论。

``--nominal-model diagonal`` 用三个拟合得到的“命令—位移”系数替代灵活的 ridge。它的表达能力更弱，但必须在 validation 的每个轴上恢复 0.75 到 1.25 范围内的绝对增益。该物理约束对照用于检查：当只有一个执行轴发生变化时，逐轴响应自适应能否超过统一动作缩放。

因果校准器累积有效激励，不遗忘较早的 chunk。validation 预先声明的 0.8--1.2 deadband 会保持标称轴不变；只有幅度足够且持续存在的响应偏差才能修正对应轴，避免低速或接触阶段的测量噪声演变成三轴动作放大。

``object_motion_probe`` 在不训练 policy 的条件下检查稀疏物体演化瓶颈。它只根据训练配对数据选择动作敏感度最高的物体，通过 leave-one-training-layout-out 交叉验证选择 ridge 系数，最后只在互斥的 development 配对状态上评估。四组输入严格匹配：本体＋reference 动作、最后一次 action-expert 特征、四点 flow-path 特征和 VLM prefix 特征。输出同时报告直接位移回归、运动事件门控回归、零运动基线及符号反转预测对照。运动阈值使用米制并需要进行敏感性检查；物体世界坐标和配对执行结果都不会进入部署输入。

``sparse_motion`` 将上述诊断实现为冻结 physical readout 上的小型可训练头。模型显式区分“物体是否运动”和“发生运动时的条件位移”，并可加入配对候选动作的几何效果监督。直接回归、事件门控、逐点监督、配对监督及符号反转配对五组严格共享训练与 development 初态、优化预算和目标物体。该结果只属于离线候选动作后果证据；冻结预测头尚未通过预先声明的独立闭环评测前，不能计为 policy 或 online RL 收益。

打开评估结果前，先固定配对干预时刻。``tick_selection`` 只读取 train 布局的物体位移，按预先声明的布局覆盖数筛选候选，并限制相邻时刻的最小间隔；将输出的 ``selected_ticks`` 传给 ``counterfactual --ticks``。该结果记录测量数据的采集协议，不代表表征或控制性能有所提升。

随后用 ``goal_ranking`` 检查冻结 sparse head 能否将配对后果转化为有效的动作排序。程序比较训练集选定物体到声明目标物体的预测距离缩短量与实测距离缩短量，报告排序准确率和 regret，并排除低于阈值的物理平局。仿真绝对物体位置只用于定义该诊断标签，不进入 policy 输入；该实验用于隔离仍然缺失的目标对齐机制，不能视为可部署控制器或 online RL 结果。

解释较高的配对排序分数前，先检查 ``constant_sign_accuracy_pct``：固定方向修正也可能解决方向分布单一的样本。``candidate_summary`` 只合并数据集、readout、目标、状态身份及绝对起点一致的多轴报告，分别报告可区分样本上的准确率、全样本 regret 和每个固定候选动作的结果。最优固定候选的上界使用评估结果选定，仅作为诊断参照。未完成的 10-tick 时间窗会被明确排除并记录。

使用 ``sparse_motion --object-rank 1`` 检验第二个被操作物体；默认 rank 为零，保留原先选择最敏感物体的行为。物体排序仅依据训练布局中的配对作用强度。``--arms gated_pair reversed_gated_pair`` 可运行有限预算的方向对照，``state_action_joint`` 提供相同预算的简单输入对照。``evaluate_object_bank`` 比较两个或更多冻结预测头的作用预测，判断哪个物体受影响最大或没有物体移动，报告 macro recall、固定主要物体和固定无运动对照。这衡量的是物理作用对象辨别能力，尚不代表任务目标选择、policy 成功率或动态路由的计算效率。

``--additional-train-pairs PATH ...`` 将其他干预轴的数据加入同一个预测头；每个目录必须包含已完成的 ``measurements.pt`` 和 ``result.json``，且干预幅度相同。重复的轴会在拟合前被拒绝。一个更新 2,000 步的共享头可以与两个各更新 1,000 步的独立头比较，保持总更新预算一致，但模型容量有所不同。Checkpoint 记录全部训练数据摘要与支持的干预轴。``sum_object_progress`` 对身份及起点一致的物体报告，等权相加源物体到目标的距离缩短量；该诊断尚未编码任务阶段或环境成功条件。``goal_ranking --score-mode reference_centered`` 在比较候选动作前减去每个预测头的 reference 进展预测，用于检查跨轴排序误差是否来自各头的分数偏移。

独立 confirmation 通过后，可以用 ``control --sparse-motion-heads LABEL=PATH`` 将冻结预测头给出的“候选动作相对 reference 的运动事件与位移”附加到 critic。``--sparse-zero-controls LABEL`` 保留相同的 256 维 critic 容量，但将物理通道替换为零。加载时会核对 sparse head 与当前 physical readout checkpoint 的摘要；版本不一致时在 rollout 前直接失败。

每轮有限预算的闭环筛选结束后，运行 ``control_report --runs PATH ...``，将不同修正轴、方法、split 分母、平均 ticks、正 replay 数量及跳过更新的原因写入同一份 CSV 和 JSON。该工具只接受完整且不可变的 run，并始终分开 validation 与 development；它负责忠实汇总协议，不负责挑选最优方法，也不会把不同轴合并成成功率结论。

离线消融完成后，使用 ``prepare-response-check`` 建立独立闭环筛选。三组共享训练 seed、验证／开发初态和 1,500 步 actor BC；``online-ticks`` 固定为 0，不用 Q＋BC 放大尚未确认的目标差异。该筛选只判断冗余响应目标是否影响 BC 控制，不是正式成功率实验。

``balanced_joint`` 保留三类响应，但分别计算机械臂关节、末端位置和夹爪位置误差，再对三组等权平均。它用于检查完整 12 维目标按维度平均时是否过度强调 7 维关节变化；该方法仍是多视角辅助监督，不把 FK 相关量解释为彼此独立的物理信息。

``outcomes`` 和 ``mixed`` 显式物理量部署只使用对应消融实际监督过的响应字段；未监督的 decoder 输出会替换为训练归一化均值，避免随机、未训练的数值进入 actor 或 critic。纯 latent 部署不受影响。

本轮不加入历史检索、agent、语言模型 API 或接触传感监督。代码检查原生动作范围与测量有效性，不使用仿真专有的接触对、冲量或摩擦参数，也不强加末端和物体位移相等。深度和物体状态仅作为训练期监督，不增加部署传感器；几何分支是短时预测器，不是 PrismWAM 深度扩散模型的复现。这里的“沉淀”指通过后果预测监督学习表征，不是教师—学生知识蒸馏，也不等同于 sim-to-real。

预测误差降低后，仍需重复闭环收益才能支持方法价值。本轮不足以证明减少人工接管、跨生成头通用性或吞吐量提升。墙钟时间、推理开销与交互效率分别报告；正式论文实验需要重新建立独立数据划分。

针对性迭代
----------

首轮结果提示：动作侧特征并不稳定优于 prefix 或“本体＋动作”。第二轮使用 prepare-followup，--base-root 指向首轮目录，--root 指向新目录；CPU 探针立即运行，GPU 闭环等待首轮结束。各方法复用已采集数据，因此属于迭代开发对比，不是独立确认实验。

hybrid_recon 与 hybrid_joint 把固定特征维度各分一半给两个来源；joint_recon 保留权重 0.1 的重建损失；factorized_joint 分别监督 latent 的两半，policy 仍读取完整 latent。residual_joint 预测简单“本体＋动作”模型无法解释的执行响应，拟合先验时排除目标训练回合；这是经验响应先验，不是已知物理定律。state_action_joint 重复作为简单输入对照。第二轮仍使用 400 步表征训练、3 个离线 seed，并开展单 seed 控制筛选。

诊断项 token_without_candidate 仅从探针输入中移除实际接受的命令，token 仍包含 reference 意图，因此不是完全无动作基线。要证明反事实动作效果正确，还需要匹配干预实验；观察性预测误差本身不够。

修正控制协议并使用预测物理量
--------------------------

控制协议第 2 版保证 critic 的 replay、actor 和 target 使用相同的 policy 提议动作域；实际接受的原生命令另存，用于物理监督。发现这一不一致后，首轮在线结果已作废；交互数据、离线探针、reference 和 BC 评测仍可使用。解释早期在线结果前应先检查 INVALIDATION.json。

使用 prepare-control-check，并指定 --base-root 和 --after-root，可在第二轮结束后接续修正的匹配对照。除四种基础表征外，joint_outcomes 直接向 policy 提供预测的归一化关节／末端／夹爪运动、降采样深度变化和物体运动；joint_mixed 将它们与一半 latent 结合。两者只用 reference 条件下的预测，不读未来实测值；部署保留解码器，因此时延统计必须计入这部分开销。

最后在训练初态上筛查平移命令增益与执行延迟。它属于自定义软件执行扰动，不是官方 LIBERO-Pro 成绩，也不等于修改摩擦或刚度。未把待执行命令加入状态前，代码拒绝延迟条件下的在线 RL。任务连续 10 分钟没有日志进展会被停止，启动时的 GPU 抢占竞态仅作有限次数重试。

保留原有动作表征
----------------

使用 prepare-anchor-check，保持 --base-root，并将 --after-root 指向修正控制实验，可以检查物理监督是否丢失了控制所需的动作信息。四组均保留一个冻结重建表征，分别与另一个重建、几何、响应或联合表征组合，固定投影回 128 维。每组使用两个各训练 400 步的 encoder 和同样的 actor；双重建对照采用不同初始化 seed，避免重复特征造成秩退化。该实验检验表征互补性，不增加 policy 容量，也不预设物理监督一定有效。

用新开发初态复核
----------------

prepare-repeat 排队运行两个训练 seed（47、59）、六个新开发初态（10—15），并分别测试原始条件和 0.7 平移命令增益。在动作特征与融合特征内，分别比较重建监督和联合物理监督。有限计划共 16 个任务，每个任务两种方法，各训练 BC 1,500 步、在线最多 1,040 个控制步。表征仍由原始条件数据训练，因此增益实验检查迁移和短程 RL 适应，不等于已在变化物理条件下训练表征。每组匹配对照使用相同任务、初态、训练 seed 和预算。

新初态身份不得与原先实验重叠，其身份由任务集、任务、公开 state 和扰动共同确定。Collection 的 reset hash 只作为来源记录，因为跨运行时或资产版本不能要求模拟器状态逐字节一致；同一次控制实验以每个 case 的首次 reset 为配对基准，之后所有方法的数值物理状态必须在绝对误差 ``1e-9`` 内一致。验证集保持不变。这些新增公开初态仍属于开发数据，不能宣称是未经使用的论文测试集。同一批布局上的多个 seed 不会增加独立初态数。所有方法和条件都要记录，包括验证门槛未通过和表现退化的情况。

阅读结果台账
------------

运行 python -m toolkits.rlt.physrlt.report --root <各轮实验的父目录>，可根据原始记录生成 overview/index.html、JSON 台账及 CSV 表格。在独立 tmux 中增加 --watch，可每分钟更新一次，直到已准备的队列结束；同一父目录只运行一个报告生成进程。报告保留未完成任务、数据划分、独立初态数及作废在线结果，不用预期成绩补齐缺失结果，也不把重复 reference 评测合并成独立样本。

检查候选动作的真实效果
----------------------

prepare-paired-actions 确认首轮数据和表征已完成后，在授权空闲卡上排队运行无需训练的诊断。代码重放已记录的命令前缀，核对本体与几何状态一致，再从同一状态执行两个方向相反、幅度受限的平移改动。比较预测后果之差、实测后果之差和“没有动作效应”的零预测对照；原始测量与预测保存在 measurements.pt。对照包括联合监督、打乱标签及“本体＋动作”的解码器，不使用重建模型中未经训练的后果解码器。默认 pilot 最多检查两个训练布局上的十二个状态；显式声明的接触阶段扩展可覆盖全部六个训练布局，但仍只能检验局部动作敏感性，不能证明任务成功率提升。

默认的 ``--paired-split train`` 复现上述诊断；准备新实验时设置 ``--paired-split development``，可把原有 development 布局用于配对动作采集。这些布局未参与 readout 拟合，但仍属于开发数据，不是论文最终测试集。

``python -m toolkits.rlt.physrlt.effect_probe`` 读取不可变的 transition 数据、``measurements.pt``、对应的纯测量 ``result.json`` 和一个 ``joint`` checkpoint，并按布局交叉使用配对标签：一个配对布局用于更新，另一个用于评估，随后交换。base、同数据逐点监督、正确差分监督和符号反转差分负对照都从同一 checkpoint 开始；``--steps`` 上限为 400。输出保留来源哈希和全部 fold。由于基础 readout 已见过两个训练布局的普通 transition，这仍是配对标签层面的开发证据，不能解释为未见布局泛化。

独立的 development split 采集完成后，同时传入 ``--evaluation-measurements`` 和 ``--evaluation-paired-result``。此时模块使用全部训练布局配对数据更新，只在未参与 readout 拟合和配对效果更新的 development 布局上评估。代码拒绝不完整记录、split 不匹配、布局重叠以及经过动作裁剪后幅度改变的干预。该结果属于更严格的后果预测证据，仍不能代替任务成功率或在线 RL 收益。

均匀选择执行时刻可能让 paired-action 数据主要覆盖自由空间运动，使零物体变化成为很强的基线。接触阶段筛选必须在采集命令中显式记录时刻，并保留独立布局评估。``object_effect`` 与 ``reversed_object_effect`` 只用候选动作之间的实测物体位移差更新 geometry decoder，前者使用正确方向，后者使用符号反转负对照；encoder、重建头和 response decoder 保持不变。只有正确方向同时优于原联合 effect、反向对照和零变化基线，才允许进入闭环实验。

完成独立 development 评估时，程序还会在 ``checkpoints/`` 下保存逐点监督、正确差分监督和符号反转差分监督三类 readout。每个 checkpoint 都记录 transition 数据哈希、配对测量哈希、监督组别、更新数和 seed。通过 ``control`` 的 ``--external-readouts LABEL=PATH`` 显式指定文件；同时出现在 ``--candidate-critics`` 中的 label 不改变 actor 输入，而是针对 replay 动作、target 动作和 actor 候选动作分别调用冻结后果解码器，并将当前 latent 与预测后果压缩为 128 维 critic 状态。环境仍执行二值夹爪命令；只有冻结后果模型内部使用有界 straight-through 近似向 actor 传递梯度。控制结果会记录 readout 哈希及条件化协议。

列入 ``--relative-critics`` 的 label 改为向 critic 提供“候选动作预测后果减去冻结 reference 动作预测后果”。同一个冻结 decoder 分别计算两个动作，可消除与动作无关的绝对值偏差，并让部署接口更贴合配对差分监督；critic 仍单独读取候选动作，actor 结构不变。绝对后果与相对后果 label 不能重叠。只有在同一相对接口下比较逐点 decoder 和配对差分 decoder，才能把差异归因于 effect 监督。

``--correction-limit`` 启用一项独立的 reference 锚定控制变体。actor 仍预测完整动作 chunk，但环境实际执行且 critic 读取的动作会变换为 ``reference + limit * tanh((raw - reference) / limit)``；BC 继续约束 actor 的原始输出，target policy smoothing 也经过同一有界变换。该选项默认关闭，因此不会改变 baseline。实验报告必须把它视为匹配的 actor 参数化对照，而不是 physical readout 自身的证据：如果逐点监督和配对差分监督都受益，只能归因于 reference 锚定；只有配对差分组进一步优于同配置逐点组，才能支持 effect 监督收益。

``--correction-axes`` 可进一步把修正限制到显式列出的动作维度；其他维度严格等于 reference，actor 也不会从这些维度收到 Q 梯度。这只是监督支持集匹配诊断，不是预期的通用方法。例如，成对数据只干预第 0 维时，不能据此让 effect decoder 指导任意旋转、平移或夹爪修正。完整方法必须让 paired evidence 覆盖 policy 实际允许修改的动作子空间。

``--residual-output`` 是一项独立的 actor 参数化对照。它将最后一层零初始化，并把 actor 原生输出解释为 reference 周围的有界 residual；BC 的目标因此是零 residual。所有表征从完全相同的 reference 行为开始，可以避免“先学会复制完整动作”的随机误差掩盖 token 差异。该选项不是原始 RLT actor，报告时必须与默认直接动作输出分开。

``--deferred-online-updates`` 会先完成有界在线 rollout，再按合格 transition 数执行等量更新。它与 ``--residual-output`` 配合时，不同表征会在 critic 分化前从同一初始 policy 分布采集数据。每次运行还会对提议动作、控制器接受命令、奖励和终止字段计算哈希，避免意外的数据分布差异被忽略。该诊断用于区分表征价值与即时 on-policy 漂移，不是默认的 online RLT 调度。

稀疏成功奖励还需要区分“有学习证据”和“全零回报”。``--require-positive-replay`` 在匹配 replay 没有任何正奖励 transition 时跳过 deferred actor／critic 更新，使 policy 保持 zero-residual reference，而不是跟随只见过零回报的随机初始化 critic 漂移。它属于独立协议，必须重跑相同协议的 action-reconstruction 对照，不能直接与未门控实验比较。

``--fixed-replays LABEL=PATH ...`` 用于离线隔离优化 seed。它要求启用 deferred 更新，并为每个方法显式提供 replay；加载器会拒绝不完整 chunk、非有限 tensor 和缺失字段，记录源文件摘要，并继续执行轨迹哈希审计。不同表征必须使用同一条物理轨迹各自重新编码后的 replay，不能直接复用另一种表征的 latent。该协议检验“面对相同正奖励证据，哪种表征更易学习”，不代表在线探索性能。

闭环成功数很少时，还要检查 critic 是否真的学会了回报排序。``critic_audit`` 从不可变 online checkpoint 读取一个未参与优化的 replay，按 episode 边界重建稀疏成功回报，报告 Q 排序 AUC、Monte Carlo 回报误差、双 Q 分歧及 actor 相对 reference 的 Q 增量。全零 replay 只用于检查虚假高分，不能计算 AUC。该审计不把训练 replay 上的拟合度当作泛化证据，也不使用 validation／development 闭环结果更新模型。

原有 ``candidate_relative`` 为保持 128 维 critic 输入，会用 64 维相对后果替换一半 latent；这同时改变了物理信息和原始表征容量。``--concat-control-critics`` 与 ``--concat-relative-critics`` 提供容量匹配的诊断：两组 critic 都接收完整 128 维 token 和额外 128 维通道，对照通道恒为零，物理通道为候选动作相对 reference 的预测后果。actor 仍只读取原始 token，训练 replay 与更新预算保持不变。只有物理组相对零通道对照改善 held-out Q 排序并带来跨 seed 闭环收益，才能说明显式物理后果对 critic 有用。

物理后果通道还可能与 base token 存在尺度差异。``--unit-concat-relative-critics`` 只归一化 68 个有效相对后果坐标的方向，critic 仍从原始候选动作读取幅度；符号反转 decoder 必须作为负对照，防止把任意放大信号误认为物理收益。若要避免物理监督同时改变 actor 的 base token，使用 ``--critic-readouts LABEL=PATH`` 指定独立 Physical Token encoder；固定 replay 还需用 ``--fixed-critic-replays`` 提供逐行匹配的物理视图。加载器要求 base 与物理 replay 的动作、控制器命令、奖励和终止哈希完全一致，actor 始终读取 base token，只有候选后果 decoder 读取 Physical Token。

当 actor gradient 对稀疏奖励 critic 过于敏感时，``--conservative-selectors`` 提供不改变训练数据的冻结评估诊断。它在 reference 与 actor 提议之间生成固定数量的插值候选，以双 Q 较小值减去 Q 分歧惩罚作为分数；只有最佳候选相对 reference 超过 ``--candidate-selection-margin`` 才执行修正，否则回退到 reference。选择器只在 critic 已更新且采用确定性评估时生效，并记录非 reference 选择比例、平均插值尺度和分数增量。action control 与 Physical Token 必须使用相同候选数、margin、分歧惩罚和 checkpoint；该实验只能检验物理条件化是否改善候选排序，不能视为额外在线数据带来的收益。

``--actor-zero-concat`` 与 ``--actor-physical-concat`` 用于检验 grounded readout 是否必须直接进入动作生成路径，而不只是进入 critic。两组 actor 输入均为 256 维：容量对照将完整 128 维 base token 与零向量拼接，物理组将同一 base token 与独立的 128 维 Physical Token 拼接；两组的实际 replay、actor／critic 更新次数和 base token 完全匹配。只有物理组跨多个优化 seed 优于零通道对照，才能支持物理表征收益；若只优于原 128 维 actor，无法排除模型容量增加带来的影响。

``--actor-physical-outcomes-concat`` 保持同一个 256 维接口，但将受物理监督的 encoder latent 替换为冻结 decoder 对 reference 动作预测的显式归一化后果，以区分“监督塑造的 latent”和类似 WAM 的预测后果表征。部署时只使用当前观测、本体状态、reference chunk 与冻结模型预测；实测未来响应和特权几何仍只作为训练标签。只有同时比较零通道对照、physical latent 与 explicit outcomes，才能把收益归因于显式物理量。

``--actor-physical-gates`` 为指定 actor token 的后半部分加入一个可学习的标量门。门从零开始，因此在 actor 权重一致时，物理组和零通道组的初始动作逐元素相同；只有 Q＋BC 梯度持续支持物理信息时，冻结的物理通道才会逐步开启。匹配对照中的所有组（包括零通道容量对照）都应使用相同门控结构，并同时报告最终门值和冻结评测结果。该门用于保护 reference 动作先验，本身不能证明物理量有效。

首轮闭环门槛应在 ``effect_probe`` 中设置 ``--parameter-scope decoders``。逐点、正确差分及符号反转差分监督此时冻结 encoder 和重建头，只更新响应与几何 decoder；三者提供给 actor 的 token 与起始 ``joint`` readout 逐参数一致，后续控制差异不能再归因于 actor 表征变化。默认的 ``all`` scope 仍可用于表征学习诊断，但必须与 decoder-only 结果分开报告。

候选动作条件化 critic 只是有预算上限的开发实验，不会因为接口实现就自动成为正结果。需要在相同 warmup 状态、在线交互量和评估布局下比较原始 latent critic、原始联合 readout 的候选条件化、同数据逐点监督和配对差分监督。BC 热身仍须通过动作保真度与 validation 成功门槛，才允许在线更新；仅有预测误差改善不能触发更大训练预算。

目前各 supervisor 在启动前申请共享研究 GPU 锁，子进程继承并持有到退出；进度检查同时读取回合／效果结果和终端日志。早先 pilot-v3 的守护进程启动时仍使用只检查 stdout 的旧逻辑，误停了仍在产出结果的实验；其中已完成、协议正确的回合仍属于有效开发数据。prepare-control-recovery 在新目录仅补跑未完成的 mixed-token 对照和被阻塞的 reference 扰动筛查。进程中断不能解释为算法负结果。

旧 pilot-v4 守护进程也中断了 task 0 的 anchor 实验。prepare-anchor-recovery 只补跑其中响应和联合两组；已经完成的 task 3 及 task 0 的重建／几何两组结果仍保留。所有补跑使用新目录，不覆盖原始失败记录。

更多诊断布局可用 ``counterfactual --cases 5 --case-start 0`` 和 ``--case-start 5`` 按固定顺序分批测量。起点只在选定 split 内计数，每个进程仍至多处理六个布局，并记录 case ID；每批使用独立输出目录。批次应在查看结果之前确定。

检验执行响应与物体运动的耦合
------------------------------

使用 ``transport_probe`` 检验先预测末端执行响应、再预测物体运动能否改善成对后果预测。该实验将预测的三维末端响应乘以取值为 [0, 1] 的学习系数，近似局部运动传递关系。它不是通用接触规律；旋转、滑动和释放可能不满足这一近似。输入仅包含当前本体状态、reference 命令与冻结 readout 特征，未来实测响应和物体位置仅用于监督。

以下命令读取 task 0 的 multiphase 实验目录，并写入新的不可覆盖结果目录。``development`` 对应两个开发布局，``confirmation`` 对应此前使用过的六个诊断布局；后者不是全新最终测试集。

.. code-block:: bash

   python -m toolkits.rlt.physrlt.transport_probe \
     --root <task0-multiphase-root> \
     --output <new-output-directory> --split confirmation

报告比较直接预测、学习耦合系数、训练集均值系数、本体与动作耦合以及打乱标签。使用未来真值的 oracle 单独列示，不可部署。岭回归正则通过训练布局留一验证选择。误差区间按布局重采样，而非将相关 transition 视为独立样本；反复开发后的区间仍属探索性结果，也不能证明控制收益。小样本或全零奖励闭环结果表示证据不足，不应在充分匹配的实验之前据此淘汰表征方案。

在中等难度任务上筛选
----------------------

扩大 seed 或任务之前，先选择一个成功率非零且未饱和的 LIBERO-Pro 单元。``medium_gate`` 默认使用 task 5 的 object 扰动，对应 baseline 审计中冻结 reference 成功率为 40% 的单元。状态 0—5 用于训练，6—9 用于验证，30—31 用于配对后果开发，32—35 用于闭环开发；配置通过开发门槛前，不读取状态 10—29。

先生成不可变的初态和实验队列，再检查生成的命令：

.. code-block:: bash

   python -m toolkits.rlt.physrlt.medium_gate prepare \
     --root <experiment-root> \
     --source <alphabrain-source> \
     --assets <pi05-checkpoint> \
     --python <libero-python> \
     --code "$PWD" \
     --gpu 2

   python -m toolkits.rlt.experiment_queue \
     --plan <experiment-root>/plan.json \
     --output <experiment-root>/queue \
     --check

该计划先在 400 步 encoder 更新、1,500 步 actor BC 更新和 1,040 个在线控制步的相同预算下复现冻结 π₀.₅ 与 AlphaBrain RLT 对照，再让 reference 约束的 residual actor 比较动作重建 readout、物理 latent、逐点后果 decoder 和配对后果 decoder。配对后果组只有在开发集上比逐点组多成功一次，或成功数相同且 capped mean ticks 至少降低 5%、验证表现不退化时，才进入下一阶段。

队列结束后，从原始 episode 记录生成统一结果表：

.. code-block:: bash

   python -m toolkits.rlt.physrlt.medium_gate summarize \
     --root <experiment-root>

``comparison.csv`` 与 ``comparison.md`` 会同时保留待完成、负结果和提升结果。单 seed 开发门槛通过后才能增加 seed 47 和 59；这一结果本身不能作为论文结论。正式读取状态 10—29 前，必须先冻结配置。
