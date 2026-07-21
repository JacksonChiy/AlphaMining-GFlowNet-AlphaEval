# 本地数据说明

请将私有日频行情文件放到 `data/price.csv`。该文件已被 Git 忽略，本项目不会自动下载，也不会上传到 GitHub。

规范字段为 `date`、`code`、`open`、`high`、`low`、`close`、`volume`；可选字段为 `amount`、`vwap`、`adj_factor`、`industry` 和 `market_cap`。加载器会自动映射常见中文字段名及数据商别名。

`industry` 和 `market_cap` 是可选的风险暴露输入。如果缺少其中任一字段，实验元数据会明确记录对应风险惩罚未启用，不会用虚构数据补全。

预处理阶段在本地生成 `data/daily_price.pkl` 和数据质量报告。原始 CSV、处理后的 PKL 以及其他大型数据文件均不应提交到 GitHub。

字段明细、数据检查和执行方法见[完整运行手册](../docs/运行手册.md)。
