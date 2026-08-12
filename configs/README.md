# 配置目录

- `daily/`：日频 Colab 正式、快速、GFlowNet CPU 单阶段和本地完整训练。
- `minute/`：分钟频本地、CPU MemMap和PPU RAM训练。
- `index_enhancement/`：指数Universe、标签、模型和组合配置。
- `baselines/`：已冻结的实验与费用基线。

命令行应显式传入配置路径。代码中的默认路径与这里的分类保持一致。

日频本地完整训练使用 `daily/local.yaml`，所有模型与评价产物写入
`results/daily_local/`，不覆盖 A100 或原 CPU 单阶段实验。
