# 王者荣耀 3v3 v3.5：协作任务状态机与奖励迭代

包含宏观任务、官方队友协作任务、奖励调整、安全边界和统一任务状态机等迭代。

## 来源与归档范围

| 项目 | 内容 |
| --- | --- |
| 来源仓库 | `dccc444/kaiwu_team` |
| 原始提交 | [`c860f3e6ce17`](https://github.com/dccc444/kaiwu_team/commit/c860f3e6ce171518d2591c2aba696b25e3cca887) |
| 源提交作者 | [Dccc4 / dccc444](https://github.com/dccc444) |
| 源提交作者时间 | `2026-07-31T15:56:39+08:00` |
| 上游提交说明 | PPO 3.5: Full implementation - M01/C01/C02/C03 macro objectives, R01-R05 reward updates, F01 feature semantics, S01 safety boundary, and unified task state machine |
| 运行入口 | `ppo` → `agent_ppo` |
| 源码文件数 | 83 |

这是带来源记录的版本快照，不是原仓库完整 Git 历史的镜像。源文件字节和文件模式记录在 [snapshot.json](snapshot.json) 中；归档提交本身使用本次整合的时间，原始 SHA 和作者时间作为来源信息保留。上游的快照导入时间不自动等同于实际开发、训练或比赛发生时间。

仅归档该版本已跟踪的算法源码、运行辅助代码和必要配置。官方指南、开发指引、设计归档、`kaiwu.json`、模型权重和运行产物不在此次归档范围内。原仓库为私有时，外部读者可能无法打开源提交链接；本目录与文件指纹可用于独立检查归档内容。

## 运行与比较

配套平台、依赖、模型和评测准备见[训练与复现说明](../../../docs/training_and_reproduction.md)。实际复现时，将本目录视为所选历史项目的根目录，把算法和配置合并到该赛题的官方工程中，再使用匹配的平台入口。

算法包名、配置文件名、导入路径及检查点协议沿用该版本的约定。不要把不同版本的 `agent_*` 或 `conf` 放入同一个 Python 导入路径；历史检查点也不保证能被当前最终版直接加载。

本次归档验证了源码指纹、Python 语法及 TOML/JSON 解析；未执行开悟平台训练或游戏评测。历史版本用于查看演进和复现对照，不包含本次新增的性能或比赛成绩结论。

返回[版本总索引](../../README.md)或[当前最终版](../../../honor_of_kings_3v3/introduction/README.md)。
