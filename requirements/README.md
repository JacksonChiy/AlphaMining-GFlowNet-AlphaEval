# 依赖分层

- `base.txt`：数据处理、表达式计算和评估所需的基础依赖。
- `training.txt`：在基础依赖上增加 PyTorch、LightGBM 和绘图工具。
- `ddb.txt`：分钟频 DolphinDB 训练环境，包含训练依赖和 DolphinDB Python API。
- `dev.txt`：开发与测试环境，包含训练依赖、Pytest 和 Jupyter。

根目录 `requirements.txt` 与 `requirements-ddb.txt` 是兼容入口，已有命令可以继续使用。
