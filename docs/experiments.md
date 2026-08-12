# 实验管理

## 原则

一次实验必须能由“代码 commit + 冻结配置 + 输入数据指纹 + 随机种子”复现。配置文件是参数真源，Notebook 中不维护另一套隐式参数。未经统一费用和日期口径的回测不能直接横向比较。

## 推荐目录

```text
experiments/<experiment_id>/
├── config.yaml
├── experiment_manifest.json
├── factor_results.csv
├── model_metrics.csv
└── backtest_report/
```

推荐命名为 `exp_YYYYMMDD_HHMMSS_<简短目的>`，例如 `exp_20260812_093000_csi500_huber`。不要再使用 `final2`、`latest_new` 等无法解释差异的名称。

## 每次必须记录

- Git commit、分支和工作区是否干净；
- 数据日期范围、股票数、交易日数和文件哈希；
- 训练/验证/回测切分及标签定义；
- GFlowNet 网络、epoch、轨迹数、学习率、seed 和 Reward 权重；
- 因子池大小、覆盖率门槛、AlphaEval/DPP 选择数；
- LightGBM 目标、窗口、purge 和 Universe；
- RQAlphaPlus 全部生效配置、费用、滑点、基准与调仓规则；
- 模型指标、IC/单调性、跟踪误差、换手和超额收益。

## 当前实验资产

- `experiments/baseline_v2_index_excess_l2_equal_weight/`：指数加权超额收益标签与 L2/等权组合基线；
- `experiments/fee_0008_baseline_20260729/`：统一万 8 费用后的冻结回测；
- `configs/index_enhancement/model_experiments.yaml`：指数模型标签和损失函数实验集合；
- `configs/baselines/fee_0008.yaml`：费用与组合约束的可复用基线。

历史分析见 `docs/reports/`。这些记录反映当时实验，不代表当前默认配置已经达到相同结果。

## 比较检查清单

- [ ] 数据区间与 Universe 相同；
- [ ] 标签定义、purge 和可交易掩码相同；
- [ ] 回测初始资金、万 8 费用、滑点与税率相同；
- [ ] 基准和指数权重时点相同；
- [ ] 报告中同时给出绝对收益、超额收益、跟踪误差、换手和最大回撤；
- [ ] 只改变一个主要实验变量，或明确记录联合变更。

大型模型与结果默认不提交 Git。确认需要发布时使用 Release 或 Git LFS，并在发布前检查是否含行情、服务器地址、账号或凭据。
