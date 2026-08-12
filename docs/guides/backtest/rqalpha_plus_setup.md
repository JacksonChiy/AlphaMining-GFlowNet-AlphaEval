# RQAlphaPlus 配置说明

RQAlphaPlus 是米筐的授权软件，本项目不会用自研模拟器替代它。请通过已获授权的米筐渠道安装 `rqalpha_plus`，并提前准备本地数据包。为保证实验可复现，运行时不启用自动下载行情数据。

本阶段只在本地运行，不在 Google Colab 运行。Colab 使用 `notebooks/pipelines/daily_colab_a100.ipynb` 生成 `alphamining_colab_outputs.zip`；把压缩包放到本地仓库根目录后，运行 `notebooks/stages/06_rqalpha_backtest.ipynb` 即可解压并校验训练产物。

## 运行前检查

请确认以下条件均已满足：

- 当前 Python 环境能够成功执行 `import rqalpha_plus`；
- 本地 RQAlphaPlus 数据包存在且当前账户有读取权限；
- 已生成 `results/lightgbm/prediction_score.csv`；
- 预测文件包含 `signal_date`、`code`、`prediction_score`；
- 股票代码格式能够被 RQAlphaPlus 识别，例如 `000001.XSHE`、`600000.XSHG`。

## 运行命令

在项目根目录执行：

```bash
python -m rqalpha_strategy.run_backtest \
  --config configs/daily/training.yaml \
  --bundle ~/.rqalpha-plus/bundle \
  --predictions results/lightgbm/prediction_score.csv \
  --output-dir results/backtest_report
```

如数据包不在默认位置，请将 `--bundle` 后的路径替换为实际授权数据包目录。

回测参数默认从所选 YAML 文件的 `backtest` 段读取。程序会将最终生效的资金、基准、持仓、排名平滑和换手控制参数打印到终端，并保存为 `results/backtest_report/backtest_effective_config.json`，避免 Notebook 只显示配置却没有传给回测程序。

本项目是纯 A 股策略，运行配置会显式关闭 `option`、`fund`、`convertible` 和 `spot` 模块，防止框架因缺少期权 instruments bundle 而尝试调用未初始化的 RQData。配置同时显式设置 `capital_gain_tax_rate: 0.0`，并使用新版 `stock_min_commission` 参数。股票行情 bundle 仍必须覆盖预测分数的全部回测日期。

## 策略与输出

策略仅使用严格早于交易日的信号。股票截面排名先按配置权重做历史平滑，再按 `rebalance_days` 调仓；旧持仓可在排名缓冲区内继续保留，同时受最短持有期和单次最大替换比例约束。目标组合最终等权持有，分析器会在 `results/backtest_report/` 下写入绩效汇总、净值曲线、持仓和交易历史。

完整步骤与故障排查见[日频运行手册](../daily/runbook.md)。

## 官方参考

- [RQAlphaPlus 配置 API](https://www.ricequant.com/doc/rqalpha-plus/api/config)
- [RQAlphaPlus 入口 API](https://www.ricequant.com/doc/rqalpha-plus/api/entrypoint)
- [RQAlphaPlus 下单 API](https://www.ricequant.com/doc/rqalpha-plus/api/order-api)
