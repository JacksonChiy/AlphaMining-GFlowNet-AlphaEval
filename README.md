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
- 奖励为 `abs(RankIC) × (1 + 截断后的 LongIR) × RiskPenalty × CoveragePenalty`；
- 覆盖率同时检查有效观测占比和满足最小股票数的交易日占比；低覆盖表达式会被降权，低于默认 80% 门槛时禁止进入因子池；
- 仅当数据中真实存在行业和市值字段时，才启用对应风险暴露惩罚；
- 金融逻辑评价使用确定性的表达式复杂度与深度评分，不调用外部大模型；
- DPP 阶段在质量加权的半正定相似度核上进行贪心 MAP 筛选；
- 因子计算与未来收益标签严格隔离。

公式、评价方法及防止未来数据泄漏的边界详见[日频复现方法说明](docs/methodology.md)。

## 运行环境

模型训练面向 Google Colab + NVIDIA A100，并启用 PyTorch 混合精度训练。推荐直接打开 `notebooks/00_colab_full_pipeline_A100.ipynb`，在一次 Colab 会话中完成数据准备、GFlowNet、AlphaEval、LightGBM 和产物下载。把 Colab 硬件加速器设为 **A100 GPU** 后按顺序运行全部单元格。Notebook 会输出并校验：

- CUDA 是否可用及 CUDA 运行时版本；
- GPU 型号；
- GPU 总显存；
- PyTorch 版本；
- A100 强制校验结果。

Notebook 默认 `FAST_MODE=False`，因为三指数正式训练必须覆盖全部历史成分股。GFlowNet Reward 和 AlphaEval 使用 2020–2023 年，选出的表达式在 2020–2026 完整序列上重算；随后在沪深300、中证500和中证1000各自的历史 Universe 内，以指数加权超额收益为目标分别进行带 5 日 purge 的 LightGBM walk-forward 训练，只输出 2024 年以后的预测。`FAST_MODE=True` 仍可用于800只股票的流水线冒烟测试，但不能生成完整三指数增强模型。

Notebook 默认 `REUSE_EXISTING_ALPHA_POOL=False`，确保日期切换后真正使用 2020–2023 年重新训练。只有已经用相同训练区间完成 GFlowNet 并保留 `results/alpha_pool.csv` 时，才可手工改为 `True`，从保存的 token 恢复表达式并重算 2020–2026 因子。命令行也可运行 `python -m src.gflownet.recompute_factors --config configs/quick_training_config.yaml`。

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

### DolphinDB分钟数据MemMap CPU训练

`cpu-training`分支支持DDB与训练分离部署。远程DolphinDB只在首次构建时按交易日传输数据；构建器使用多线程独立DDB session并行写入不同交易日切片。训练服务器将19个分钟通道保存为按年组织的`float32 (n_days, 241, n_stocks)`本地MemMap；GFlowNet训练阶段不再连接DDB。Reward默认直接在NumPy三维数组上执行，只读取表达式需要的通道，并按日期小块持续打印进度。排名、滚动、掩码、相关性、协方差、加权均值、高阶矩和趋势统计均提供批量向量化路径；全空组直接跳过，缺分钟与并列值保留精确兼容路径。Block会按`close`、`ret+vol`等通道依赖自动打包，同一任务共享MemMap切片和公共子表达式缓存。每个表达式的中间结果合并保存为二维因子MemMap、完成位图和元数据三个文件，训练中断后可以续算，旧版partial小文件会自动迁移；只有日内Reduce后的日频结果才转换为Pandas。旧Pandas执行器保留为显式兼容回退，不会创建原始分钟PKL。

```bash
python scripts/prepare_ddb_minute.py \
  --config configs/minute_training_cpu_ddb.yaml \
  --audit-only

python scripts/audit_ddb_minute_quality.py \
  --config configs/minute_training_cpu_ddb.yaml \
  --scope grid

python scripts/build_ddb_memmap.py \
  --config configs/minute_training_cpu_ddb.yaml

python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute_training_cpu_ddb.yaml
```

可在训练服务器运行分钟执行器基准：

```bash
python scripts/benchmark_minute_numpy.py --days 10 --minutes 241 --stocks 256 --repeats 5
```

先运行`python scripts/build_ddb_memmap.py --config configs/minute_training_cpu_ddb.yaml`完成可断点续传构建，再运行CPU训练入口。正式运行前必须把结束日期改为数据库实际最后交易日，并确认OHLC已经复权后设置`prices_are_adjusted: true`。完整两机部署步骤见[DDB与训练分离的MemMap运行手册](docs/DDB与训练分离的MemMap运行手册.md)。

### 推荐：单个 Colab Notebook

Colab 一次只需打开：

```text
notebooks/00_colab_full_pipeline_A100.ipynb
```

该 Notebook 包含 GitHub clone、依赖安装、A100 检查、配置读取、从 Google Drive 复制本地数据、GFlowNet、AlphaEval、三指数标签、三套 LightGBM 和产物打包。请将行情保存为 `MyDrive/price.csv`，并将 `index_components.csv.gz`、`index_weights.csv.gz` 保存到 `MyDrive/AlphaMiningData/`。最终下载：

```text
alphamining_colab_outputs.zip
```

压缩包包含 GFlowNet 检查点、因子池、因子矩阵、AlphaEval 结果，以及三套指数 LightGBM 模型与 `prediction_score.csv`。RQAlphaPlus 不在 Colab 运行。

### 分阶段 Notebook

原有 Notebook 继续保留，用于单独调试各阶段。所有 Notebook 默认 clone 本仓库；如需使用个人分支，可设置环境变量 `ALPHAMINING_REPO_URL`。

按以下顺序运行：

1. `01_data_prepare.ipynb`：数据检查与预处理；
2. `02_expression_engine.ipynb`：表达式生成、序列化与执行；
3. `03_train_gflownet_A100.ipynb`：A100 上训练 GFlowNet；
4. `04_alpha_eval.ipynb`：AlphaEval 评价与 DPP 筛选；
5. `05_lgbm_model.ipynb`：滚动 LightGBM 融合；
6. `06_rqalpha_backtest.ipynb`：在本地已获授权的 RQAlphaPlus 环境中导入 Colab 产物并回测。

也可以在项目根目录编排前五个阶段：

```bash
python -m scripts.run_daily_pipeline --pool-size 100
```

上述命令仅编排 Colab 训练阶段；RQAlphaPlus 请在本地通过独立 Notebook 或回测入口运行。

正式训练默认强制要求 A100。`--allow-non-a100` 仅用于小规模代码路径冒烟测试，不得用于正式实验结果。

训练阶段保存 `checkpoints/gflownet_best.pt`，随后重新加载检查点，并生成 `factor_001`、`factor_002` 等因子元数据和因子值矩阵。

训练过程中每条轨迹都实时打印 epoch/step、全局 step、总体完成百分比、表达式、动作数、reward、RankIC、LongIR、风险惩罚、观测覆盖率、有效交易日覆盖率、覆盖率惩罚、`logPF`、单轨迹 TB loss 和耗时；Reward 进度同时输出子表达式缓存命中数、未命中数与命中率。每个 epoch 更新参数后，再打印平均与最高奖励、平均 RankIC、平均覆盖率、`logZ`、梯度范数、学习率、缓存命中/等待/内存、耗时、最佳检查点状态及 A100 显存。逐轨迹明细写入 `results/gflownet_trajectory_metrics.csv`，逐 epoch 汇总写入 `results/gflownet_training_metrics.csv`。因子池生成阶段也会为每次尝试打印接受、重复或低覆盖拒绝状态。

为提高 A100 利用率，同一 epoch 的轨迹按批次进行 Transformer 推理，而不是逐轨迹执行 batch size 1；Reward 对批内唯一表达式使用多线程并行，RankIC、Top 10% LongIR、行业与市值暴露均使用向量化计算。九个日频时序算子会把数据组织成“股票 × 时间”张量，在 PyTorch CUDA 上分块执行；没有 CUDA 时自动回退到 Pandas。表达式执行器还会跨表达式复用结构相同的子树，例如多个因子共同包含的 `ts_mean(close,20)` 只计算一次。缓存支持并发 single-flight、有界 LRU 和内存上限，不会因表达式池增长而无限占用 RAM。默认 `reward_workers: 4`，可根据 Colab CPU 核数调整。逐步日志保留，但每个 epoch 只进行一次批量 GPU→CPU 指标同步。

分钟MemMap训练会额外输出分阶段性能日志：底层区分缓存扫描、数据读取、NumPy算子、
缓存写入和结果组装；Reward批次区分Block执行、表达式组装和金融指标评价；每个epoch
区分采样、损失构造、反向传播、参数更新和checkpoint，并直接标出当前`bottleneck`。
详细字段解释见[《DDB与训练分离的MemMap运行手册》](docs/DDB与训练分离的MemMap运行手册.md)。

## GFlowNet 模型

状态包含动作 Token、部分表达式、当前与最大深度、算子数量、特征数量及归一化节点统计。Transformer Encoder 预测下一个合法的特征、算子或窗口动作，非法文法动作会被屏蔽。训练目标为：

```text
(logZ + sum(logPF) - logReward - sum(logPB))^2
```

采样使用 on-policy 轨迹，奖励按规范化表达式字符串缓存。检查点包含模型状态、优化器状态、`logZ`、训练配置、词表与训练历史。

覆盖率惩罚定义为：

```text
effective_coverage = min(有效因子观测数 / 可评价观测数,
                         有效交易日数 / 可评价交易日数)
CoveragePenalty = min(1, (effective_coverage / min_coverage) ^ power)
```

默认 `min_coverage: 0.80`、`coverage_penalty_power: 2.0`。Reward 会连续惩罚低覆盖表达式；生成和保存因子池时还会执行 80% 硬门槛，避免稀疏表达式继续进入 AlphaEval、LightGBM 和本地回测。

## AlphaEval 与 LightGBM

`results/alpha_eval_result.csv` 至少包含 `factor`、`IC`、`RankIC`、`ICIR`、`Sharpe`、`complexity` 和 `score`，并附带滚动 IC、扰动鲁棒性、RRE 与 DPP 诊断结果。

AlphaEval 的日度相关性和排名稳定性使用向量化计算，并逐因子打印四阶段耗时、总进度和预计剩余时间。DPP 按交易日均匀抽样，快速/正式配置最多使用 300,000/500,000 行构建多样性矩阵；因子预测能力指标仍使用完整样本。

LightGBM 使用滚动训练窗和 5 个交易日的 purge 间隔，预测标签为：

```text
close(t+5) / close(t+1) - 1
```

模型保存最新检查点，并输出每日股票预测分数和截面排名。

## 本地 RQAlphaPlus 回测

RQAlphaPlus **不在 Colab 运行**。将 Colab 下载的 `alphamining_colab_outputs.zip` 放到本地仓库根目录，然后运行 `notebooks/06_rqalpha_backtest.ipynb`。本地 Notebook 会解压并校验产物；策略实际读取 Alpha 因子经过 LightGBM 融合后的 `results/lightgbm/prediction_score.csv`。

仓库**不包含自研回测器**。策略通过 RQAlphaPlus 的 `run_file` 和 `order_target_portfolio` 运行，只使用满足 `signal_date < trade_date` 的最近一期预测分数。Notebook 和命令行入口默认从 `configs/training_config.yaml` 的 `backtest` 段读取全部参数；命令行显式参数可有意覆盖配置。

为降低原始 Top-N 策略的高换手，策略先对最近三个信号截面的股票排名按 `0.5/0.3/0.2` 加权平滑，再应用持仓排名缓冲、最短持有期和单次替换比例上限。正式配置下，Top 20 旧持仓只要仍在前 40 名即可保留，每次最多替换 25% 的目标股票，并至少持有 10 个交易日。替换比例限制的是名单变动，不是成交金额的严格上限；真实换手仍以 RQAlphaPlus 报告为准。

当前参数以配置文件为准，例如：

- `initial_cash`：初始资金；
- `benchmark`：业绩基准；
- `top_n`：持股数量；
- `rebalance_days`：调仓周期；
- `rank_smoothing_weights`：当前及历史信号排名的平滑权重；
- `hold_buffer_rank`：允许旧持仓继续保留的最差平滑排名；
- `max_replacement_ratio`：单次调仓最多替换的目标股票比例；
- `min_holding_days`：最短持有交易日数；
- 交易费用：A 股默认佣金与时点印花税；
- `slippage`：价格比例滑点；
- 报告目录：`results/backtest_report/`。

回测开始前会打印最终生效参数，并保存为 `results/backtest_report/backtest_effective_config.json`，Notebook 会逐项校验它与所选 YAML 配置一致。

RQAlphaPlus 输出年度收益、总收益、Sharpe、最大回撤、波动率、换手率、净值曲线、持仓和交易明细。

## 三类指数增强

项目支持在沪深300、中证500和中证1000的历史成分股内分别训练和回测。历史成分与指数权重各下载一次并保存到本地；`t+5/t+1` 原始收益、指数加权收益、超额收益、截面 Rank 和涨跌停可交易标签均由本地原始行情生成，后续训练、诊断和回测不重复调用RQData。

```bash
# 仅首次或主动更新时运行：每个指数调用一次RQData
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.universe

# 仅首次或主动更新时运行：每个指数一次完整区间权重查询
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.weights

# 纯本地生成三套指数内超额收益标签
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.labels

# 冻结当前基线并诊断 IC、分组收益、换手成本和现金拖累
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.diagnostics \
  --output-dir results/index_enhancement_diagnostics

# 纯本地生成三套指数内预测文件
python -m src.index_enhancement.builder

# 本地RQAlphaPlus分别回测三个指数
/opt/miniconda3/envs/rqsdk/bin/python -m rqalpha_strategy.run_index_enhancement
```

详细参数、输出结构和防未来成分泄漏检查见[指数增强运行手册](docs/指数增强运行手册.md)。

统一万8费用基线冻结、RQAlphaPlus 完整配置留档和基准权重约束组合优化见[万8基线冻结与指数增强组合优化运行手册](docs/万8基线冻结与指数增强组合优化运行手册.md)。

## 实验结果

只有在用户提供 `price.csv`、在 Colab A100 完成训练并使用有效 RQAlphaPlus 数据包回测后，才会产生真实研究结果。本仓库不会伪造模型权重或业绩数据。预期产物如下：

```text
checkpoints/gflownet_best.pt
alphamining_colab_outputs.zip
results/gflownet_training_metrics.csv
results/gflownet_trajectory_metrics.csv
results/alpha_pool.csv
results/alpha_factor_matrix.pkl
results/alpha_factor_matrix_oos.pkl
results/alpha_eval_result.csv
results/index_enhancement/csi300/
results/index_enhancement/csi500/
results/index_enhancement/csi1000/
results/index_enhancement_backtest/
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
├── notebooks/               # Colab 一体化训练与分阶段调试 Notebook
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

分钟频表达式系统已经加入研报图表 27–30 的 21 个特征、15 个分钟算子、14 个掩码算子和 16 个聚合算子，并提供独立的 A100 训练入口。字段口径、算子语义、服务器训练命令与输出说明见[分钟频表达式与 GFlowNet 训练手册](docs/分钟频表达式与GFlowNet训练手册.md)。

没有 GPU 时，可使用独立的日频/分钟频 CPU 配置和强制 CPU 启动器。线程调优、运行命令、输出目录与排错方法见[CPU 训练运行手册](docs/CPU训练运行手册.md)。

远程 DolphinDB 分钟表可通过字段审计、日期分块缓存和 CPU 训练入口接入。连接配置、复权确认和完整命令见[DolphinDB 分钟数据 CPU 训练手册](docs/DolphinDB分钟数据CPU训练手册.md)。

- 使用 MemMap、分块缓存、Numba 和多进程扩展分钟频数据；
- 增加更多研报算子，包括时序二元算子；
- 接入严格时点一致的行业、市值和 Barra 风险暴露并进行中性化；
- 引入带 embargo 的嵌套验证和完全隔离的最终研究期；
- 分布式奖励计算与更大规模的 GFlowNet 策略网络。
