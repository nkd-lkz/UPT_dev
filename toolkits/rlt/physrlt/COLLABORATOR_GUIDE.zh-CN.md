# PhysRLT 合作开发说明

这份说明帮助你在另一台服务器复现当前科研分支，并独立验证“动作端物理表征是否提升轻量在线 RL”这一核心假设。先完成环境和单任务复现，再认领多轴动作后果实验；暂时不要扩大 benchmark、训练 seed 或读取预留测试初态。

## 当前研究问题

冻结的 π₀.₅ 能产生可用的 reference action，但动作生成特征主要服务于生成动作，不保证表达命令执行后的关节响应、末端运动和物体几何变化。当前方法从 action-generation pathway 读取特征，将已完成交互的实测后果作为训练标签，学习 128 维 action-side physical representation，再交给 reference-anchored actor/critic 做小预算在线适应。

现阶段只检验三个问题：

1. 动作端表征能否预测候选动作的短期执行后果；
2. 正确的配对动作 effect 监督是否优于逐点后果监督和打乱标签；
3. 在相同初态、网络容量和交互预算下，物理表征能否提高成功率，或在成功率相同时减少控制步数。

历史记忆、外部语言模型、contact sensor 和大规模 benchmark 扩展不在当前实验范围内。几何、关节和末端量只作为训练监督；部署时仍使用原有相机、本体状态和 reference action。

## 获取代码

仓库：<https://github.com/nkd-lkz/UPT_dev>

共享分支：`research/physrlt-support-gate-20261027`

```bash
git clone https://github.com/nkd-lkz/UPT_dev.git
cd UPT_dev
git switch research/physrlt-support-gate-20261027
git rev-parse HEAD
git switch -c research/physrlt-<你的名字>-multiaxis
```

最后一条命令建立个人开发分支。不要直接改写共享分支；提交时使用 Conventional Commit，并通过 `git commit -s` 添加 DCO 签名。

主要代码位于：

| 路径 | 作用 |
|---|---|
| `toolkits/rlt/physrlt/collect.py` | 采集冻结 π₀.₅ 特征、reference action 和实测 transition |
| `toolkits/rlt/physrlt/model.py` | 128 维 encoder 与后果 decoder |
| `toolkits/rlt/physrlt/probe.py` | 动作重建、几何、响应、联合及负对照 readout |
| `toolkits/rlt/physrlt/counterfactual.py` | 从相同 simulator prefix 执行正负配对动作 |
| `toolkits/rlt/physrlt/effect_probe.py` | 比较逐点、正确 effect 和反号 effect 监督 |
| `toolkits/rlt/physrlt/control.py` | reference-anchored actor/critic 与短程在线训练 |
| `toolkits/rlt/physrlt/medium_gate.py` | 固定任务、初态、预算并生成统一结果表 |
| `tests/unit_tests/test_physrlt.py` | 数据边界、模型接口和实验协议的回归测试 |

## 准备运行环境

已验证的主机使用 Python 3.11.14、PyTorch 2.11.0、Transformers 4.53.2、NumPy 1.26.4、Robosuite 1.4.1、`rlinf-libero` 0.1.3 和 `rlinf-liberopro` 0.1.3。建议使用至少 24 GB 显存的 NVIDIA GPU，并将代码、Python 环境和实验输出放在支持符号链接及文件权限的本地磁盘；模型权重可以放在大容量磁盘。

当前安装脚本尚未把 `openpi + liberopro` 暴露为一个组合目标。先创建独立 Python 3.11 环境，按 AlphaBrain 的 `requirements.txt` 安装 π₀.₅ 依赖，再安装固定版本的 LIBERO 包：

```bash
uv venv --python 3.11 .venv-physrlt
source .venv-physrlt/bin/activate
uv pip install -r /path/to/AlphaBrain/requirements.txt
uv pip install rlinf-libero==0.1.3 rlinf-liberopro==0.1.3
libero-download-assets --skip-existing
liberopro-download-assets --skip-existing
```

如果上游 requirements 改变了 PyTorch、Transformers 或 NumPy，先恢复上面的已验证版本并保存 `pip freeze`。不要复用正在运行其他训练的环境。

将 AlphaBrain 固定到已经审计的提交：

```bash
git clone https://github.com/AlphaBrainGroup/AlphaBrain.git /path/to/AlphaBrain
git -C /path/to/AlphaBrain checkout 604924beb77b04b0da49326dfae6ea423a27d28a
test -z "$(git -C /path/to/AlphaBrain status --porcelain)"
```

下载公开的四套 LIBERO π₀.₅ 权重。下载器固定 Hugging Face revision `482fcfea4d6a5cee97456f50b1f5aeeae539a1c8`，并校验 `model.safetensors`：

```bash
python -m toolkits.rlt.pi05_assets --destination /path/to/pi05-assets
```

## 先通过最小验收

在启动 GPU 实验前运行测试：

```bash
python -m pytest -q \
  tests/unit_tests/test_physrlt.py \
  tests/unit_tests/test_experiment_queue.py \
  tests/unit_tests/test_pi05_baseline.py
```

随后为 LIBERO-Pro `task5-object` 生成新的开发计划。每个输出目录只生成一次；`--python-deps` 用于依赖装在独立目录的情况，全部装在 venv 时可以省略。只有系统 OpenGL/EGL 库放在非标准路径时才传 `--library-path`。

```bash
python -m toolkits.rlt.physrlt.medium_gate prepare \
  --root /path/to/experiments/task5-object-seed31 \
  --source /path/to/AlphaBrain \
  --assets /path/to/pi05-assets \
  --python "$(which python)" \
  --code "$PWD" \
  --gpu 0 \
  --task 5 \
  --perturbation object

python -m toolkits.rlt.experiment_queue \
  --plan /path/to/experiments/task5-object-seed31/plan.json \
  --output /path/to/experiments/task5-object-seed31/queue \
  --check
```

先检查 `plan.json` 中的 Python、AlphaBrain、权重和输出路径，再运行队列。启动后保留原始 `status.json`、日志、episode 记录和失败产物；不要只保存最好的一次结果。

## 建议认领：多轴动作后果 Grounding

当前配对动作实验只覆盖动作第 0 维，actor 也只能修改这一维。你负责把科研证据扩展到平移维度 0、1、2，但仍保持单任务、小数据和固定预算。

具体工作按以下顺序推进：

1. 扩展 `counterfactual.py`，从同一个已记录 simulator prefix 分别执行每个轴的 `+offset` 和 `-offset`。每个分支先核对 reset hash、本体状态和几何状态，再执行候选动作；动作被裁剪后幅度不一致时，该样本必须作废。
2. 扩展 `effect_probe.py`，把 axis id 作为显式条件或分别训练等容量 decoder。比较 base、pointwise、correct effect、reversed effect 和 shuffled-label 五组，所有组从同一个 checkpoint 开始，更新数相同。
3. 先只做离线门槛：在未参与 readout 拟合的 development 布局上，correct effect 的动作差分预测误差应稳定优于 pointwise 和 reversed effect。若不成立，停止闭环训练并分析数据对齐、动作裁剪和监督尺度。
4. 离线门槛通过后，将 `--correction-axes` 扩展为 `0 1 2`。保持 1,500 步 BC、最多 1,040 个在线控制步、correction limit 0.1、seed 31，与 pointwise decoder 做匹配闭环比较。
5. 只有 correct effect 比 pointwise 多成功一次，或成功数相同且 capped mean ticks 至少降低 5%、validation 不退化，才增加 seed 47 和 59。不要提前读取 states 10—29；它们留给配置冻结后的确认实验。

第一轮应交付下面这张表，而不是长时间训练日志：

| 方法 | 轴覆盖 | 离线 effect error | Validation 成功数 | Development 成功率 | Mean ticks | Online ticks | GPU 小时 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pointwise | 0/1/2 |  |  |  |  | 1040 |  |  |
| Correct effect | 0/1/2 |  |  |  |  | 1040 |  |  |
| Reversed effect | 0/1/2 |  |  |  |  | 0 |  |  |

## 实验边界

- 训练、validation、development 和最终确认初态必须分开；调参后不能把相同初态改名为测试集。
- 每组共享 reference checkpoint、actor 初始化、网络容量、训练 seed、交互预算和评测初态。
- 未来 transition 只能作为训练标签，不能进入执行时输入；未完成的 horizon 必须 mask。
- 二值夹爪动作、原生动作范围和 reference anchoring 不能因方法不同而改变。
- W&B 可以关闭；本地原始记录是验收依据。启用 W&B 时，不得因网络失败阻断本地训练。
- 零成功、validation gate 失败、超时和代码错误分别记录，不能统一写成算法失败。
- 不要从多次随机复评中挑最好结果。重试必须说明原因，并保留前一次产物。

## 回传代码和结果

代码提交前运行：

```bash
python -m pytest -q tests/unit_tests/test_physrlt.py
ruff check toolkits/rlt/physrlt tests/unit_tests/test_physrlt.py
ruff format --check toolkits/rlt/physrlt tests/unit_tests/test_physrlt.py
git diff --check
```

推送个人分支后，提供以下内容：

1. 分支名称和完整 commit SHA；
2. `pip freeze`、GPU 型号、驱动和 CUDA 版本；
3. 不可变的 `plan.json`、代码 revision 和输入初态清单；
4. 原始 result/episode JSON、统一 CSV、运行日志和失败记录；
5. 一段结论：哪些比较支持假设、哪些不支持、下一步是否满足扩大实验的预设门槛。

完整算法背景和已有消融见 [README.zh.rst](README.zh.rst)。当前所有结果均属于开发证据，不能直接写成论文最终结论。
