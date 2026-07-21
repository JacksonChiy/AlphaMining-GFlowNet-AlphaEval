# 实验目录说明

每次正式运行创建一个 `experiments/<experiment_id>/` 目录，用于保存冻结后的 `config.yaml`、因子元数据、模型指标、预测结果，以及运行 RQAlphaPlus 后生成的回测报告。

`experiment_id` 用于把配置、模型和评价结果绑定到同一次实验，建议采用 `exp001`、`exp002` 等递增命名，或包含 UTC 时间戳的唯一名称。

生成的实验目录默认不提交到 Git。需要公开的配置和汇总指标应先进行人工审核；模型检查点通过 GitHub Release 或 Git LFS 发布，私有行情数据不得发布。
