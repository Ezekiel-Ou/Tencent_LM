# 训练与复现说明

本文说明五个项目的环境准备、算法选择、训练检查、模型加载与评测记录方式。各项目的算法设计和历史成绩见根目录 [README](README.md) 及对应的 `introduction/README.md`。

源码核对基线为 [`ea14758958bdb8c8d35c592c698e9e9da65558ed`](https://github.com/Ezekiel-Ou/Tencent_LM/commit/ea14758958bdb8c8d35c592c698e9e9da65558ed)。下述配置值来自该版本；复现时应记录实际使用的提交和配置修改。

## 1. 复现前需要准备什么

本仓库保存算法代码、部分平台配置、地图转移数据和历史实验展示。完整训练还需要腾讯开悟对应赛题的官方工程、运行环境及 SDK。

| 项目 | 准备要求 |
| --- | --- |
| 运行环境 | 使用与赛题和接口版本匹配的开悟镜像或官方工程，保留其启动脚本、环境服务与协议文件 |
| Python 依赖 | 使用平台配套的 PyTorch、NumPy 和 SDK；代码涉及 `kaiwu_agent`、`kaiwudrl`、`tools`、`common_python` 等模块，不同项目所需模块有所不同 |
| 模型文件 | 仓库未附训练权重；继续训练或评估历史模型时，需要另行准备兼容检查点 |
| 对手模型 | 使用自定义对手 ID 时，需要在当前平台模型管理和 `kaiwu.json` 中配置可用模型 |
| 实验记录 | 保存源码提交、镜像版本、依赖版本、配置快照、初始权重和评测对手，才能解释结果差异 |

仓库没有统一的依赖锁定文件、完整的独立启动工程或已验证的最低 GPU 规格。Python、PyTorch、CUDA 和驱动版本应以实际使用的配套环境为准，并在实验记录中保存；仅在普通 Python 环境中安装 PyTorch 不足以启动完整训练。

五个项目应放在各自的赛题工程中运行。它们重复使用 `agent_ppo`、`conf` 等模块名，将多个项目同时加入同一 Python 导入路径可能加载错误实现。

## 2. 选择项目与算法入口

训练配置分为三个层次：

| 层次 | 文件 | 作用 |
| --- | --- | --- |
| 平台运行配置 | 各项目的 `conf/configure_app.toml`、`conf/app_conf_*.toml` | 策略选择、样本池、训练批次、预加载和评估模型路径；部分字段由平台启动脚本生成或覆盖 |
| 算法映射 | 各项目的 `conf/algo_conf_*.toml` | 将算法名映射到 Agent、训练、评估和考试工作流 |
| 算法与环境参数 | 所选 `agent_*/conf/conf.py`、`agent_*/conf/train_env_conf.toml` | 网络、优化参数、奖励和对局设置；部分运行逻辑还会动态更新配置 |

仓库已提供的训练实现如下：

| 项目 | 算法名 | Agent 与训练实现 | 映射来源 |
| --- | --- | --- | --- |
| 峡谷漫步 PPO | `diy` | `agent_ppo` | [algo_conf_gorge_walk.toml](gorge_walk/conf/algo_conf_gorge_walk.toml) |
| 峡谷漫步动态规划 | `dynamic_programming` | `agent_dynamic_programming` | 同上 |
| 重返秘境 DQN | `dqn` | `agent_dqn` | [algo_conf_back_to_the_realm.toml](back_to_the_realm/conf/algo_conf_back_to_the_realm.toml) |
| 重返秘境阶段目标 DQN | `target_dqn` | `agent_target_dqn` | 同上 |
| 峡谷追猎 PPO | `ppo` | `agent_ppo` | [algo_conf_gorge_chase.toml](gorge_chase/conf/algo_conf_gorge_chase.toml) |
| 王者荣耀 1v1 | `diy` 或 `ppo` | 两个入口均指向 `agent_ppo` | [algo_conf_hok1v1.toml](honor_of_kings_1v1/conf/algo_conf_hok1v1.toml) |
| 王者荣耀 3v3 | `ppo` | `agent_ppo` | [algo_conf_hok3v3_collaboration.toml](honor_of_kings_3v3/conf/algo_conf_hok3v3_collaboration.toml) |

启动前要核对平台启动参数、`configure_app.toml` 中的 `app.algo`（若存在）和 `app_conf_*.toml` 中各策略的 `algo`，确认它们最终选择同一套实现。实际配置加载顺序取决于配套 SDK，应从启动日志再次确认 Agent 和工作流的模块路径。

当前上传版本有以下需要人工确认的地方：

- **峡谷漫步**：[configure_app.toml](gorge_walk/conf/configure_app.toml) 写的是 `algo = "diy"`，而 [app_conf_gorge_walk.toml](gorge_walk/conf/app_conf_gorge_walk.toml) 的 `train_one` 写的是 `dynamic_programming`。先选择 PPO 或动态规划，再统一相关入口。
- **重返秘境**：[configure_app.toml](back_to_the_realm/conf/configure_app.toml) 写的是 `target_dqn`，而 [app_conf_back_to_the_realm.toml](back_to_the_realm/conf/app_conf_back_to_the_realm.toml) 的 `train_one` 写的是 `dqn`。选择阶段目标版本时，应核对策略也指向 `target_dqn`。
- **王者荣耀 3v3**：`diy` 指向 `agent_ppo_debug`，用于官方英雄技能和指令调试。正常训练使用 `ppo`；活动参数由 [FrozenV1Config、V1TrainingConfig 和 Config](honor_of_kings_3v3/agent_ppo/conf/conf.py) 定义，`LegacyConfig` 不能作为当前网络的参数依据。
- 算法映射中仍保留部分未随仓库上传的模板目录，例如 `agent_monte_carlo`、`agent_q_learning`、`agent_sarsa`、`agent_diy`。选择这些入口前需确认官方工程中确有对应实现。

峡谷漫步的 `configure_app.toml` 还保留 `sample_dim = 6`，PPO 的 [Config.SAMPLE_DIM](gorge_walk/agent_ppo/conf/conf.py) 则为 505。其 [PPO 工作流](gorge_walk/agent_ppo/workflow/train_workflow.py) 当前直接调用 `agent.learn(g_data)`，不能简单套用其他项目的样本传输方式。使用 Learner/Reverb 等平台链路时，应按所选实现和 SDK 核对实际序列化格式、样本维度与批次，避免仅凭模板字段修改维度。

## 3. 环境配置与训练参数

### 3.1 对局环境

下表路径均相对于仓库根目录；工作流读取配置时，路径相对于对应项目的工作目录。

| 项目 | 环境配置 | 当前主要设置 |
| --- | --- | --- |
| 峡谷漫步 | [PPO](gorge_walk/agent_ppo/conf/train_env_conf.toml)、[动态规划](gorge_walk/agent_dynamic_programming/conf/train_env_conf.toml) | 起点 `[29, 9]`，终点 `[11, 55]`，固定 10 个宝箱，单局最多 2000 步 |
| 重返秘境 | [target_dqn](back_to_the_realm/agent_target_dqn/conf/train_env_conf.toml)、[dqn](back_to_the_realm/agent_dqn/conf/train_env_conf.toml) | 起点 2，终点 1，随机 13 个宝箱，闪现技能，单局最多 2000 步 |
| 峡谷追猎 | [train_env_conf.toml](gorge_chase/agent_ppo/conf/train_env_conf.toml) | 地图 1～10，`map_random = false`，10 个宝箱，2 个增益，单局最多 1000 步 |
| 王者荣耀 1v1 | [train_env_conf.toml](honor_of_kings_1v1/agent_ppo/conf/train_env_conf.toml) | 混合训练对手；`eval_interval = 20`；阵容和召唤师技能还受工作流与算法配置控制 |
| 王者荣耀 3v3 | [train_env_conf.toml](honor_of_kings_3v3/agent_ppo/conf/train_env_conf.toml) | 自博弈，自动切换监控阵营；协作指令持续 300 帧、冷却 100 帧 |

各项目还有以下运行条件：

- **动态规划**：[训练工作流](gorge_walk/agent_dynamic_programming/workflow/train_workflow.py) 读取 [F_level_1.json](gorge_walk/conf/map_data/F_level_1.json) 后执行规划并保存策略表。应保持地图、起终点、固定宝箱设置与 [预设宝箱顺序](gorge_walk/agent_dynamic_programming/conf/conf.py) 一致。它的训练过程不等同于 PPO 的环境采样和梯度更新。
- **峡谷追猎**：当前 [CURRICULUM_AUTO](gorge_chase/agent_ppo/conf/conf.py) 为 `False`，读取普通 `train_env_conf.toml`。启用课程模式后，工作流才会选择 `train_env_conf_stage1.toml` 到 `stage4.toml`；`CURRICULUM_STAGE_OVERRIDE` 也要结合该开关使用。阶段由当前进程的局数推进，重启后的进度需要另行确认。
- **王者荣耀 1v1**：默认训练对手为 `selfplay`、`290819`、`278480`，权重为 `0.60 / 0.30 / 0.10`，评估对手池也包含这两个历史模型 ID。这些 ID 在新的账号或环境中不一定可用；要先导入对应模型并配置 `kaiwu.json`，或明确改成可用对手。
- **王者荣耀 3v3**：工作流按 [13 局阵容块](honor_of_kings_3v3/agent_ppo/conf/conf.py) 轮换双方官方英雄，TOML 中的英雄初值不代表训练全程固定阵容。`OPPONENT_MIX_ENABLED = False`，代码中的 9:1 混合对手调度未开启。`eval_interval = 10000000000`，不能据此认为训练中已持续评估对 `common_ai` 的胜率。

若 1v1 暂时没有历史对手模型，可在匹配的平台环境中先用 `common_ai` 验证流程。修改现有 `[episode]` 表中的下列字段，并保留其余所需配置，不要重复声明同名表：

```toml
[episode]
opponent_agent = "common_ai"
eval_interval = 20
eval_opponent_type = "common_ai"
eval_opponent_types = ["common_ai"]
```

这是流程验证用的对手设置。使用它得到的结果应单独记录，不能直接与历史模型对手下的成绩比较。

### 3.2 参数核对位置

运行前保存所选 `conf.py` 和平台 TOML 的完整快照。以下列出部分默认值，其他奖励、网络结构和调度参数以链接源码为准。

| 实现 | 参数来源 | 部分默认值 |
| --- | --- | --- |
| 峡谷漫步 PPO | [conf.py](gorge_walk/agent_ppo/conf/conf.py) | Actor / Critic 学习率 `3e-4 / 1e-3`，γ / λ 为 `0.95 / 0.95`，PPO 更新 4 轮，微批次 256 |
| 重返秘境 target_dqn | [conf.py](back_to_the_realm/agent_target_dqn/conf/conf.py) | 学习率 `1e-4`，γ 为 `0.995`，探索率从 `1.0` 衰减到 `0.05` |
| 峡谷追猎 PPO | [conf.py](gorge_chase/agent_ppo/conf/conf.py) | 初始学习率 `3e-4`，γ / λ 为 `0.995 / 0.97`，PPO clip 为 `0.2` |
| 王者荣耀 1v1 | [conf.py](honor_of_kings_1v1/agent_ppo/conf/conf.py) | 初始学习率 `3e-5`，目标学习率 `1e-5`，γ / λ 为 `0.995 / 0.95` |
| 王者荣耀 3v3 | [V1TrainingConfig](honor_of_kings_3v3/agent_ppo/conf/conf.py) | 学习率 `2e-4`，γ / λ 为 `0.997 / 0.97`，微批次 256，开启 AMP 支持 |

平台的 `train_batch_size` 与算法内部微批次不是同一个字段。例如 3v3 的平台训练批次为 1024，算法微批次为 256。修改批次或样本格式时，需要同时核对平台和算法的实际处理方式。

## 4. 从源码到首次训练

### 4.1 获取代码并确定工作目录

```bash
git clone https://github.com/Ezekiel-Ou/Tencent_LM.git
cd Tencent_LM
git rev-parse HEAD
git status --short

# 示例：选择王者荣耀 3v3；其他项目切换到对应目录
cd honor_of_kings_3v3
```

上述操作只取得源码。部署时，将所选项目的 `agent_*` 和本仓库提供的配置合并到对应赛题的官方工程中，保留官方启动脚本、SDK、环境接口、其他必需配置和平台生成文件。不要用本仓库的部分 `conf/` 文件替换掉整个官方配置工程。

实际运行目录应当能以相对路径访问 `conf/` 和所选 `agent_*/`。例如 3v3 的环境配置应能从运行目录定位到 `agent_ppo/conf/train_env_conf.toml`。

### 4.2 在配套环境中做启动检查

下面的命令面向平台镜像中的 Bash 与 Python，用于检查基础模块位置，不会启动训练：

```bash
python --version
python - <<'PY'
from importlib.util import find_spec

missing = []
for name in ("torch", "numpy", "kaiwudrl", "tools"):
    spec = find_spec(name)
    print(name, "MISSING" if spec is None else spec.origin)
    if spec is None:
        missing.append(name)
if missing:
    raise SystemExit("Missing platform dependencies: " + ", ".join(missing))
PY
```

根据所选项目，继续确认 `kaiwu_agent`、`common_python` 等依赖及其子模块可以导入，模块位置应来自对应 SDK 和官方工程。顶层模块存在仅说明基础路径可见，不能证明游戏服务、接口版本或训练流程兼容。

以 3v3 为例，可在对应工程根目录进一步检查 Agent 导入和 Python 语法：

```bash
python -c "from agent_ppo.agent import Agent; print(Agent.__module__)"
python -m compileall -q agent_ppo
```

这些检查仍不包含环境交互。随后按第 2 节统一算法选择，按第 5 节配置初始化模型，并确认对手池中的模型可用。

### 4.3 启动并确认训练链路

使用对应开悟镜像或工作台提供的训练入口启动任务。本仓库未附可统一运行五个项目的启动脚本；`agent_*/workflow/train_workflow.py` 是由平台注入环境和 Agent 后调用的回调，不能以直接执行该文件代替平台启动流程。

首次运行先完成以下检查，再扩大训练预算：

1. 启动日志显示正确的赛题、算法、Agent、工作目录及配置文件。
2. 环境能够成功 reset 并完成对局，观测、动作、奖励及终止标记正常。
3. 对需要采样学习的实现，确认样本正常送达、学习器发生更新，损失及梯度没有 NaN/Inf；同时检查 Actor 是否加载到更新后的模型。
4. 生成检查点后，通过配套评估入口重新加载该模型，确认文件格式、模型 ID 和推理流程一致。动态规划重点检查策略表的生成、保存与加载。
5. 保存此次运行的配置、日志和初始验证结果，再确定正式训练的资源和时长。

多数训练工作流包含持续运行循环，应通过平台任务预算、停止机制和实验计划控制训练时长。环境中的 `max_step` 是单局步数限制，不代表整个训练任务会在对应步数后结束。

## 5. 初始化、保存与加载模型

### 5.1 处理默认预加载设置

以下两个项目默认继续加载历史模型，直接照搬配置可能因文件缺失或协议不兼容而失败：

| 项目 | 配置来源 | 当前预加载设置 |
| --- | --- | --- |
| 王者荣耀 1v1 | [configure_app.toml](honor_of_kings_1v1/conf/configure_app.toml) | `preload_model = true`，目录 `/data/pre_model/ckpt`，ID `397045` |
| 王者荣耀 3v3 | [configure_app.toml](honor_of_kings_3v3/conf/configure_app.toml) | `preload_model = true`，目录 `/data/pre_model/ckpt`，ID `396387` |

根据实验目标选择初始化方式：

- **继续训练**：准备与当前网络和特征协议兼容的模型，配置正确目录及 ID，并检查实际加载日志。还应核对优化器、训练步数、学习率调度和随机状态恢复了哪些内容，仅加载网络权重不代表完整恢复训练现场。
- **从头训练**：将对应配置中的 `preload_model` 设为 `false`，使用配套平台的正常初始化流程，并记录这是重新训练实验。从头训练不能代替缺失的历史权重来证明赛时结果已复现。

重返秘境与峡谷追猎当前的 `preload_model` 为 `false`。峡谷漫步的该配置文件没有声明此开关，应按其 SDK 和所选实现核对初始化方式。

### 5.2 模型格式与评估路径

| 实现 | 保存方式与入口 | 加载时需要注意 |
| --- | --- | --- |
| 峡谷漫步动态规划 | [agent.py](gorge_walk/agent_dynamic_programming/agent.py)：`model.ckpt-<id>.npy` | 保存的是策略表，需要匹配状态编码和地图设置 |
| 峡谷漫步 PPO | [agent.py](gorge_walk/agent_ppo/agent.py)：`ppo_model_<id>.pt` | 工作流保存 `best` 和 `latest`；不能假定与其他项目的 `model.ckpt-<id>.pkl` 命名相同 |
| 重返秘境 | [target_dqn/agent.py](back_to_the_realm/agent_target_dqn/agent.py)、[dqn/agent.py](back_to_the_realm/agent_dqn/agent.py)：`model.ckpt-<id>.pkl` | 区分 DQN 与阶段目标版本，核对网络结构及加载日志 |
| 峡谷追猎 | [agent.py](gorge_chase/agent_ppo/agent.py)：`model.ckpt-<id>.pkl` | 核对特征、网络和环境设置 |
| 王者荣耀 1v1 | [agent.py](honor_of_kings_1v1/agent_ppo/agent.py) | 新检查点包含协议元数据，并保存 `.meta.json` 附属文件；模型归档时一并保留 |
| 王者荣耀 3v3 | [checkpoint_v1.py](honor_of_kings_3v3/agent_ppo/model/checkpoint_v1.py) | 检查点包含格式版本和协议签名；兼容迁移以加载器明确支持的范围为准 |

模型保存受平台 `dump_model_freq`、同步配置及部分工作流中的定时保存逻辑共同影响。应以实际产物和日志确认保存位置、时间与 ID，不要只根据训练时长推断模型步数。

各项目的 `eval_model_dir`、`eval_model_id` 是历史运行设置，不代表仓库已提供对应文件。正式评估前，按所选 SDK 的要求把它们指向实际存在且兼容的模型；峡谷漫步 PPO 还需确认评估入口能加载其实际保存的 `best` / `latest` ID 和文件名。

若协议签名、特征维度、循环状态或动作空间不匹配，应定位源码与检查点版本差异。不要通过忽略维度错误或绕过签名检查来认定模型已经成功复现。

## 6. 评估与历史结果比较

评估工作流由各项目的 `conf/algo_conf_*.toml` 指定。峡谷漫步、重返秘境的相关入口位于 SDK 的 `kaiwu_agent.*`；峡谷追猎、1v1、3v3 使用平台的 `tools.eval.workflow.*`。这些评估实现没有随仓库完整上传，需要配套环境提供。

建议先用少量对局确认模型加载和评估流程，再预先确定正式评测的局数、场景与统计口径。正式比较时固定待测检查点，避免使用训练中持续变化的 `latest` 模型。

| 项目 | 应固定并记录的条件 | 建议记录的指标 |
| --- | --- | --- |
| 峡谷漫步 | 地图、起终点、固定宝箱或随机宝箱设置、最大步数 | 到达率、平均得分、完成步数、宝箱数 |
| 重返秘境 | 宝箱数量与随机方式、起终点、技能、最大步数 | 完成率、平均得分、平均步数、宝箱数；区分 9 宝箱与 13 宝箱实验 |
| 峡谷追猎 | 地图集合、怪物出现与加速时间、技能冷却、宝箱与增益设置 | 平均得分、生存步数、宝箱数，并按地图分别统计 |
| 王者荣耀 1v1 | 对手模型与版本、英雄组合、召唤师技能、双方阵营 | 胜/负/平局数、胜率及分阵容、分阵营结果 |
| 王者荣耀 3v3 | 对手版本、双方官方英雄、模型控制英雄组合、协作指令设置、阵营 | 胜/负/平局数、胜率、资源与建筑推进、官方队友指令执行情况 |

训练塑形奖励与比赛得分、胜率应分别报告。平均得分使用全部已纳入评测的对局；胜率报告分母与平局处理方式；失败或中断对局单独记录及说明处理规则。条件不同的实验不能仅以总平均值直接比较。

仓库现有名次、比分和截图属于历史实验材料。要复现对应结果，还需要当时的代码、训练权重、平台版本、对手和评估条件；在缺少这些资料时，应将新实验标记为当前条件下的重新训练或重新评估。

随机性需要覆盖 Python、NumPy、PyTorch、阵容/对手采样和环境侧随机设置。当前源码没有统一的全局可复现启动器；例如 3v3 的混合对手调度在启用时还会使用进程 ID 参与种子构造，单独设置一个 PyTorch 种子不足以保证完整运行一致。记录实际可控的种子、并行度和不可控条件，必要时重复独立训练评估波动。

## 7. 每次实验的记录模板

建议将以下记录、配置快照、日志和模型校验值保存在平台实验存储或仓库外的归档目录。各字段填写实际信息，未执行的项目保留“未执行”。

```yaml
run_id: 待填写
source_commit: 待填写
source_patch: 无修改或归档补丁路径
project: honor_of_kings_3v3
algorithm: ppo
platform_image_and_sdk: 待填写镜像标签或摘要及SDK版本
python_pytorch_cuda_driver: 待填写实际版本
hardware_and_parallelism: 待填写
config_snapshots: 待填写配置快照位置
initialization: 从头训练或指定初始检查点
initial_checkpoint_sha256: 不适用或待填写
opponent_pool_and_lineups: 待填写对手版本及阵容设置
randomness: 待填写种子和不可控条件
training_budget_and_updates: 待填写实际时长及更新次数
evaluation_checkpoint_and_sha256: 未执行
evaluation_settings_and_games: 未执行
evaluation_metrics_and_failures: 未执行
logs_and_artifacts: 待填写归档位置
```

## 8. 常见问题定位

| 现象 | 优先检查 |
| --- | --- |
| 无法导入 `kaiwu_agent`、`kaiwudrl`、`tools` 或 `common_python` | 是否使用对应赛题的官方工程与镜像，SDK 路径是否正确 |
| 提示找不到 `agent_*` 或加载了另一项目的实现 | 算法映射是否选择了未提供的模板；工作目录和 Python 路径是否混入其他项目 |
| 启动后选择的算法与预期不同 | 启动参数、`configure_app.toml`、策略 `algo` 和启动日志是否一致 |
| 找不到预加载或评估模型 | 文件是否实际存在，目录、ID、命名及格式是否与 Agent 加载器匹配 |
| 对手模型 ID 不在有效模型池 | 是否已经导入模型并配置 `kaiwu.json`；训练对手池与评估对手池都需检查 |
| 样本、张量维度或模型协议错误 | 配置、Agent、特征、样本打包、模型和检查点是否来自同一兼容版本 |
| 峡谷追猎修改阶段文件后没有生效 | `CURRICULUM_AUTO` 是否开启，工作流实际选择了哪个环境文件 |
| 3v3 没有得到独立基准对手评估结果 | 默认评估间隔极大；应单独安排并记录固定模型、阵容和换边评测 |

本文完成了源码、配置引用和文档相对链接的静态核对；未在腾讯开悟环境中执行完整训练或游戏评测。实际复现结果应按第 7 节另行记录。
