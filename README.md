# AlphaMining-GFlowNet-AlphaEval

## 项目简介

本仓库是研报《基于 GFlowNet 和 AlphaEval 的分钟频因子挖掘筛选框架》的一个**日频可运行精简复现**。项目从本地私有行情文件 `price.csv` 出发，依次完成数据预处理、表达式因子挖掘、AlphaEval 因子筛选、LightGBM 因子融合和 RQAlphaPlus 策略回测。

本项目仅用于量化研究与工程复现，不构成投资建议。代码不会调用任何外部行情接口；原始数据、模型权重、实验输出和回测结果均默认由 Git 忽略。

首次运行请先阅读[完整运行手册](docs/运行手册.md)。

## 研究背景

原研报使用 GFlowNet 生成具有多样性的公式型 Alpha，通过 Transformer 表示状态，并使用 Trajectory Balance 目标训练；随后用 AlphaEval 风格的预测能力、时间稳定性、扰动鲁棒性、金融逻辑性和多样性评价进行筛选，再由 LightGBM 融合。

本日频版本保留上述核心思想，并做出以下明确适配：

- 文法采用日频 OHLCV/VWAP 特征，时间窗口为 5、10、20、40、60 个交易日；
- 前缀表达式树的构建过程具有唯一父状态，因此 TB 中的反向策略项为 `log PB = 0`；
- 奖励为 `abs(RankIC) × (1 + 截断后的 LongIR) × RiskPenalty`；
- 仅当数据中真实存在行业和市值字段时，才启用对应风险暴露惩罚；
- 金融逻辑评价使用确定性的表达式复杂度与深度评分，不调用外部大模型；
- DPP 阶段在质量加权的半正定相似度核上进行贪心 MAP 筛选；
- 因子计算与未来收益标签严格隔离。

公式、评价方法及防止未来数据泄漏的边界详见[日频复现方法说明](docs/methodology.md)。

## 运行环境

模型训练面向 Google Colab + NVIDIA A100，并启用 PyTorch 混合精度训练。打开 `notebooks/03_train_gflownet_A100.ipynb`，把 Colab 硬件加速器设为 **A100 GPU** 后按顺序运行。Notebook 会输出并校验：

- CUDA 是否可用及 CUDA 运行时版本；
- GPU 型号；
- GPU 总显存；
- PyTorch 版本；
- A100 强制校验结果。

本地数据准备与单元测试：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

macOS 上的 LightGBM 还需要 OpenMP 运行库：

```bash
brew install libomp
```

RQAlphaPlus 是授权软件，需要通过米筐授权渠道独立安装，详见 [RQAlphaPlus 配置说明](docs/rqalpha_plus_setup.md)。

## 数据集

将本地文件放到 `data/price.csv`。支持的规范字段如下：

| 字段 | 含义 | 是否必需 |
|---|---|---|
| `date` | 交易日期 | 是 |
| `code` | 股票代码 | 是 |
| `open`、`high`、`low`、`close` | 日频开高低收价格 | 是 |
| `volume` | 日成交量 | 是 |
| `amount`、`vwap` | 成交额与成交量加权平均价 | 否 |
| `adj_factor` | 时点可得的复权因子 | 否 |
| `industry`、`market_cap` | 行业与市值风险暴露 | 否 |

加载器会自动识别常见中文字段名和数据商别名，并完成日期转换、按股票与日期稳定排序、重复与无效记录处理、仅在单只股票内部前向填充、非有限值处理、同日缩尾、可选复权和同日截面标准化。全程不使用后向填充，最终生成 `data/daily_price.pkl` 和数据质量报告。

## 训练与完整流水线

所有 Notebook 均包含 GitHub clone、依赖安装、GPU 状态检查和配置读取。默认 clone 本仓库；如需使用个人分支，可设置环境变量 `ALPHAMINING_REPO_URL`。

按以下顺序运行：

1. `01_data_prepare.ipynb`：数据检查与预处理；
2. `02_expression_engine.ipynb`：表达式生成、序列化与执行；
3. `03_train_gflownet_A100.ipynb`：A100 上训练 GFlowNet；
4. `04_alpha_eval.ipynb`：AlphaEval 评价与 DPP 筛选；
5. `05_lgbm_model.ipynb`：滚动 LightGBM 融合；
6. `06_rqalpha_backtest.ipynb`：在已获授权的 RQAlphaPlus 环境中回测。

也可以在项目根目录编排前五个阶段：

```bash
python -m scripts.run_daily_pipeline --pool-size 100
```

如需继续运行授权回测阶段，增加 `--rqalpha-bundle /path/to/bundle`。

正式训练默认强制要求 A100。`--allow-non-a100` 仅用于小规模代码路径冒烟测试，不得用于正式实验结果。

训练阶段保存 `checkpoints/gflownet_best.pt`，随后重新加载检查点，并生成 `factor_001`、`factor_002` 等因子元数据和因子值矩阵。

## GFlowNet 模型

状态包含动作 Token、部分表达式、当前与最大深度、算子数量、特征数量及归一化节点统计。Transformer Encoder 预测下一个合法的特征、算子或窗口动作，非法文法动作会被屏蔽。训练目标为：

```text
(logZ + sum(logPF) - logReward - sum(logPB))^2
```

采样使用 on-policy 轨迹，奖励按规范化表达式字符串缓存。检查点包含模型状态、优化器状态、`logZ`、训练配置、词表与训练历史。

## AlphaEval 与 LightGBM

`results/alpha_eval_result.csv` 至少包含 `factor`、`IC`、`RankIC`、`ICIR`、`Sharpe`、`complexity` 和 `score`，并附带滚动 IC、扰动鲁棒性、RRE 与 DPP 诊断结果。

LightGBM 使用滚动训练窗和 5 个交易日的 purge 间隔，预测标签为：

```text
close(t+5) / close(t+1) - 1
```

模型保存最新检查点，并输出每日股票预测分数和截面排名。

## RQAlphaPlus 回测

仓库**不包含自研回测器**。策略通过 RQAlphaPlus 的 `run_file` 和 `order_target_portfolio` 运行，只使用满足 `signal_date < trade_date` 的最近一期预测分数，选择 Top N 股票等权持有，每 5 个交易日调仓。

默认配置：

- 初始资金：人民币 1,000,000 元；
- 基准：`000300.XSHG`（沪深 300）；
- 交易费用：A 股默认佣金与时点印花税；
- 滑点：价格比例滑点 0.001；
- 报告目录：`results/backtest_report/`。

RQAlphaPlus 输出年度收益、总收益、Sharpe、最大回撤、波动率、换手率、净值曲线、持仓和交易明细。

## 实验结果

只有在用户提供 `price.csv`、在 Colab A100 完成训练并使用有效 RQAlphaPlus 数据包回测后，才会产生真实研究结果。本仓库不会伪造模型权重或业绩数据。预期产物如下：

```text
checkpoints/gflownet_best.pt
results/gflownet_training_metrics.csv
results/alpha_pool.csv
results/alpha_factor_matrix.pkl
results/alpha_eval_result.csv
results/lightgbm/lgbm_model.joblib
results/lightgbm/model_metrics.csv
results/lightgbm/feature_importance.csv
results/lightgbm/prediction_score.csv
results/backtest_report/
```

## 实验版本管理

`configs/training_config.yaml` 是默认实验配置。每次正式运行应创建 `experiments/<experiment_id>/`，保存冻结配置、因子结果、模型指标和回测报告。实验产物默认不提交；经确认的检查点应通过 GitHub Release 或 Git LFS 发布。

建议阶段标签：

- `v0.1-data-pipeline`
- `v0.2-expression-engine`
- `v0.3-gflownet`
- `v0.4-alphaeval`
- `v0.5-backtest`
- `v1.0-release`

## 项目结构

```text
AlphaMining-GFlowNet-AlphaEval/
├── configs/                 # 训练与回测配置
├── data/                    # 私有数据放置说明
├── docs/                    # 方法、运行与授权环境文档
├── experiments/             # 按 experiment_id 组织的实验
├── notebooks/               # 六阶段 Colab Notebook
├── rqalpha_strategy/        # RQAlphaPlus 策略与入口
├── scripts/                 # 完整流水线编排脚本
├── src/                     # 核心 Python 模块
│   ├── alpha_eval/
│   ├── data_loader/
│   ├── expression/
│   ├── gflownet/
│   ├── model/
│   ├── operators/
│   └── utils/
└── tests/                   # 单元测试
```

## 后续工作

- 使用 MemMap、分块缓存、Numba 和多进程扩展分钟频数据；
- 增加更多研报算子，包括时序二元算子；
- 接入严格时点一致的行业、市值和 Barra 风险暴露并进行中性化；
- 引入带 embargo 的嵌套验证和完全隔离的最终研究期；
- 分布式奖励计算与更大规模的 GFlowNet 策略网络。
