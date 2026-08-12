# 沪深300、中证500和中证1000指数增强运行手册

## 1. 实现范围

本模块复用同一份 LightGBM 全市场预测分数，在历史时点成分股内分别构建：

- 沪深300增强：`000300.XSHG`，默认选择前60只；
- 中证500增强：`000905.XSHG`，默认选择前100只；
- 中证1000增强：`000852.XSHG`，默认选择前200只。

当前版本已经进入阶段 1：只允许使用当日历史成分股，并用本地历史指数权重构造相对 Benchmark 的 `t+5/t+1` 超额收益标签。指数权重当前用于标签和诊断；组合端仍是 Top-N 等权，下一阶段再改为“基准权重 + 主动偏离”。

## 2. 一次性下载历史成分股

只有这一步调用RQData接口。请在已经配置好RQData授权的本地环境执行：

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.universe \
  --config configs/index_enhancement/default.yaml
```

程序对三个指数分别调用一次：

```python
rqdatac.index_components(
    order_book_id,
    date=None,
    start_date=start_date,
    end_date=end_date,
    market="cn",
    return_create_tm=False,
)
```

输出：

```text
data/index_components.csv.gz
data/index_components.csv.gz.metadata.json
```

压缩CSV字段为：

```text
date,index_key,index_code,index_name,code
```

文件已存在时程序默认报错并停止，以免日常运行重复访问接口。只有主动更新成分股数据时才使用：

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.universe --force
```

历史指数权重也只下载一次，每个指数使用一次完整区间查询，总计三次：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  /opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.weights \
  --config configs/index_enhancement/default.yaml
```

输出 `data/index_weights.csv.gz` 及 metadata。程序会把 RQData 的月度快照前向填充结果逐日归一到 1；文件存在时默认拒绝覆盖。RQData/RQAlphaPlus 均不使用代理。

## 3. 生成严格的指数内训练标签

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.labels \
  --config configs/index_enhancement/default.yaml
```

本步骤仅读取本地 `price.csv`、成分股和权重文件，输出：

```text
results/index_enhancement_labels/
├── manifest.json
├── csi300/labels.pkl
├── csi500/labels.pkl
└── csi1000/labels.pkl
```

标签定义为 `close(t+5) / close(t+1) - 1`。先计算个股收益，再按信号日历史指数权重计算 Benchmark 收益，并生成 `target_excess_return` 与 `target_cross_sectional_rank`。买入日涨停、卖出日跌停、停牌样本按原始 `limit_up/limit_down/volume/amount` 标记；这些未来状态只用于标签有效性，不作为特征。

分别训练指数内超额收益模型：

```bash
python -m src.model.run_lightgbm \
  --label-path results/index_enhancement_labels/csi300/labels.pkl \
  --target-type excess_return \
  --output-dir results/index_enhancement/csi300
```

将命令中的 `csi300` 依次替换为 `csi500`、`csi1000`。因子矩阵必须同时包含 2020–2023 训练期和 2024–2026 预测期。回测入口会优先读取各目录中新训练生成的 `prediction_score.csv`，其次才读取旧版过滤流程的 `.csv.gz`。

## 4. 生成三套指数增强预测输入

该阶段及后续阶段只读取本地文件，不调用RQData：

```bash
python -m src.index_enhancement.builder \
  --config configs/index_enhancement/default.yaml
```

程序将 `results/lightgbm/prediction_score.csv` 与历史成分股按 `signal_date + code` 精确匹配，防止使用当前成分股回填历史。输出：

```text
results/index_enhancement/
├── manifest.json
├── csi300/prediction_score.csv.gz
├── csi500/prediction_score.csv.gz
└── csi1000/prediction_score.csv.gz
```

每个文件包含：

```text
signal_date,code,prediction_score,index_key,index_code,index_name,
prediction_rank,universe_size
```

如果成分股文件缺少任何预测日期，构建阶段会直接报错，不会用未来成分或最近成分静默填充。

## 5. 分别运行三类指数增强回测

在本地RQAlphaPlus授权环境执行：

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m rqalpha_strategy.run_index_enhancement \
  --config configs/index_enhancement/default.yaml
```

三个回测使用独立进程和独立目录：

```text
results/index_enhancement_backtest/
├── csi300/
├── csi500/
└── csi1000/
```

回测配置会显式关闭与本策略无关的 `rqfactor`、期权、基金等模块，避免它们在启动时额外访问外部服务。股票回测行情仍只读取本地RQAlphaPlus bundle。

只回测其中一个指数：

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m rqalpha_strategy.run_index_enhancement \
  --indexes csi300
```

先检查命令但不运行回测：

```bash
python -m rqalpha_strategy.run_index_enhancement --dry-run
```

## 6. 冻结并诊断阶段 0 基线

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.diagnostics \
  --config configs/index_enhancement/default.yaml \
  --price data/price.csv \
  --output-dir results/index_enhancement_diagnostics
```

输出 SHA256 基线清单、日度/年度 RankIC、五分组收益、Universe 覆盖率、年度/月度表现、换手成本和现金拖累。该命令只读本地文件，不启动回测、不调用 RQData。

## 7. P0–P3 研究优化流程

### P0–P1：冻结基线并诊断信号

三套指数标签已经生成时执行：

```bash
python -m src.index_enhancement.research_audit \
  --config configs/index_enhancement/default.yaml \
  --baseline-id baseline_v2_index_excess_l2_equal_weight
```

结果保存在 `experiments/<baseline_id>/`，包含输入文件 SHA256、年度/月度表现、换手成本、现金拖累、成熟标签覆盖率、日度/年度 RankIC、Q5-Q1、分组单调性、Top-N 目标收益和排名稳定性。本地缺少 Colab 生成的 `labels.pkl` 时，程序仍会冻结模型与回测，但跳过信号诊断。

### P2：对比 LightGBM 训练目标

实验矩阵位于 `configs/index_enhancement/model_experiments.yaml`：

| 实验标签 | 训练目标 |
|---|---|
| `cross_sectional_rank` | 截面 Rank 回归 |
| `excess_huber` | Huber 超额收益回归 |
| `excess_top_weighted` | 最高 20% 样本三倍权重 |
| `lambdarank` | 每日股票作为一个排序组 |
| `excess_l2` | 旧 L2 基线 |

例如，只运行 Rank 和 LambdaRank：

```bash
python -m src.index_enhancement.model_experiments \
  --experiments cross_sectional_rank,lambdarank
```

每组结果写入 `results/index_model_experiments/<experiment>/<index>/`。每个滚动窗口保存一个 `lgbm_window_NNN.joblib`，同时保存最后模型、预测、指标和特征重要性。不同实验必须使用独立输出目录，禁止覆盖结果后再比较。

### P3：三指数独立 AlphaEval

```bash
python -m src.alpha_eval.run_index_evaluation \
  --config configs/daily/training.yaml \
  --target-column target_excess_return
```

主要输出：

```text
results/index_alpha_eval/
├── manifest.json
├── csi300/alpha_eval_result.csv
├── csi300/selected_factors.csv
├── csi500/...
└── csi1000/...
```

每个指数仅使用训练期历史成分、可交易的指数超额标签和该 Universe 内的因子值。LightGBM 实验优先读取对应指数的筛选结果；不存在时才降级使用全市场 AlphaEval 结果。

Colab 一体化 Notebook `notebooks/pipelines/daily_colab_a100.ipynb` 按以下顺序执行：

```text
全市场 GFlowNet
→ 全市场初筛
→ 三指数标签
→ 三指数独立 AlphaEval/DPP
→ 指定实验的三指数 LightGBM
→ 保存全部滚动模型
→ 打包下载
```

Notebook 中的 `INDEX_MODEL_EXPERIMENT` 可设为 `cross_sectional_rank`、`excess_huber`、`excess_top_weighted` 或 `lambdarank`。

## 8. 默认参数

参数位于 `configs/index_enhancement/default.yaml`：

| 指数 | Benchmark | Top N | 持仓缓冲排名 |
|---|---|---:|---:|
| 沪深300 | 000300.XSHG | 60 | 120 |
| 中证500 | 000905.XSHG | 100 | 200 |
| 中证1000 | 000852.XSHG | 200 | 400 |

其余参数继承 `configs/daily/training.yaml`，包括初始资金、调仓周期、滑点、排名平滑、单次替换上限和最短持有期。

## 9. 结果比较

三类指数增强应分别比较：

- 年化收益和年化超额收益；
- Information Ratio；
- Tracking Error；
- 最大回撤和最大超额回撤；
- 年化双边换手率与交易成本；
- 分年度表现和压力区间表现。

不能只根据绝对收益选择最佳指数，还应同时检查超额收益的稳定性与跟踪误差。
