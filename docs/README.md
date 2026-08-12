# 文档中心

本文档目录只保留当前有效的说明。首次使用建议按“项目概览 → 对应运行手册 → 故障排查”的顺序阅读。

## 1. 项目概览

| 文档 | 用途 |
|---|---|
| [项目架构](architecture.md) | 了解模块职责、依赖方向和运行边界 |
| [端到端工作流](workflow.md) | 了解输入、处理、输出及文件流向 |
| [方法说明](methodology.md) | 了解标签、Reward、AlphaEval 和防泄漏原则 |
| [实验管理](experiments.md) | 统一实验命名、配置冻结和复现要求 |
| [常见故障排查](troubleshooting.md) | 排查 Colab、DDB、MemMap、PPU、LightGBM 和 RQAlphaPlus 问题 |

## 2. 运行手册

### 日频

- [日频完整运行手册](guides/daily/runbook.md)：Colab A100 训练、产物下载及本地回测的主入口。
- [日频本地完整训练](guides/daily/local_training.md)：在本机用一条流水线完成数据准备、训练、筛选和融合。
- [CPU 训练与性能参数](guides/daily/cpu_training.md)：只在需要 CPU 线程配置、单阶段训练或性能排查时阅读。

### 分钟频

- [分钟表达式、GFlowNet 与 PPU 后处理](guides/minute/expression_gflownet.md)：分钟特征/算子、PPU 全量内存训练及训练后 AlphaEval、LightGBM、结果打包的主入口。
- [DolphinDB 分钟数据 CPU 训练](guides/minute/ddb_cpu.md)：DDB 字段审计和兼容的流式训练路径。
- [DDB 与训练分离的 MemMap 方案](guides/minute/ddb_memmap.md)：低内存训练机的正式部署方案。

### 指数增强与回测

- [三指数增强运行手册](guides/index_enhancement/runbook.md)：历史成分/权重、指数标签、P0–P3 研究流程和三指数回测。
- [万 8 基线与组合优化](guides/index_enhancement/fee_0008_portfolio.md)：费用口径、基准约束组合和参数调整顺序。
- [RQAlphaPlus 环境配置](guides/backtest/rqalpha_plus_setup.md)：本地授权环境安装和启动配置。

## 3. 研究归档

- [指数增强实验分析](reports/index_enhancement_analysis_20260729.md)：阶段性实验结果和结论。
- [指数增强路线图](reports/index_enhancement_roadmap.md)：后续优化清单及完成状态。

研究归档用于追溯历史结论，不替代当前配置和运行手册。个人汇报材料、演示文稿及自动生成报告不属于公开运行文档，不在本索引中维护。

## 4. 维护规则

- 新操作步骤优先补充到对应主手册，避免新增内容重复的小文档。
- 运行参数以 `configs/` 中的当前配置为准；文档中的数值仅作说明。
- 过时的交接快照、系统缓存和工具检查文件不进入 `docs/`。
- 阶段性结果放入 `reports/`；可执行步骤放入 `guides/`。
