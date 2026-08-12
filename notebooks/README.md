# Notebook目录

- `pipelines/`：可独立运行的完整流水线，包括 Colab A100 和本地 CPU 日频版本。
- `stages/`：数据、表达式、训练、筛选、融合和回测的分阶段调试入口。
- `deployment/`：特定服务器环境的部署Notebook。
- `archived/`：保留用于追溯、但不再作为当前入口的旧Notebook。

正式算法应位于`src/`；Notebook只负责环境准备、编排、分析和展示。

`deployment/minute_ppu_ram.ipynb` 是 PPU 分钟频完整入口：训练完成后继续执行
产物验收、AlphaEval、LightGBM 和本地回测产物打包。若已有训练产物，可关闭
`RUN_GFLOWNET_TRAINING`，从后处理阶段继续。
