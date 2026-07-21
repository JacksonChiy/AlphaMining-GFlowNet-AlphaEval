# RQAlphaPlus 配置说明

RQAlphaPlus 是米筐的授权软件，本项目不会用自研模拟器替代它。请通过已获授权的米筐渠道安装 `rqalpha_plus`，并提前准备本地数据包。为保证实验可复现，运行时不启用自动下载行情数据。

本阶段只在本地运行，不在 Google Colab 运行。Colab 使用 `notebooks/00_colab_full_pipeline_A100.ipynb` 生成 `alphamining_colab_outputs.zip`；把压缩包放到本地仓库根目录后，运行 `notebooks/06_rqalpha_backtest.ipynb` 即可解压并校验训练产物。

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
  --bundle ~/.rqalpha-plus/bundle \
  --predictions results/lightgbm/prediction_score.csv \
  --output-dir results/backtest_report
```

如数据包不在默认位置，请将 `--bundle` 后的路径替换为实际授权数据包目录。

## 策略与输出

策略仅使用严格早于交易日的最近一期信号，每 5 个交易日调仓，选择预测分数最高的 Top N 股票并等权持有。分析器会在 `results/backtest_report/` 下写入绩效汇总、净值曲线、持仓和交易历史。

完整步骤与故障排查见[运行手册](运行手册.md)。

## 官方参考

- [RQAlphaPlus 配置 API](https://www.ricequant.com/doc/rqalpha-plus/api/config)
- [RQAlphaPlus 入口 API](https://www.ricequant.com/doc/rqalpha-plus/api/entrypoint)
- [RQAlphaPlus 下单 API](https://www.ricequant.com/doc/rqalpha-plus/api/order-api)
