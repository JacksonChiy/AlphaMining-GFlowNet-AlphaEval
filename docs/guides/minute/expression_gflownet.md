# 分钟频表达式与 GFlowNet 训练手册

## 1. 实现范围

本模块复现研报《基于 GFlowNet 和 AlphaEval 的分钟频因子挖掘筛选框架》图表 27、28、29、30 中的分钟特征与算子。表达式先在每个“交易日 × 股票”的分钟序列内计算，最后必须经过 `r_*` 聚合成日频因子，才能接入现有日频 Reward、AlphaEval、LightGBM 和本地 RQAlphaPlus 回测。

对应代码：

- `src/operators/minute.py`：分钟特征、分钟算子、掩码和日内聚合；
- `src/expression/minute.py`：分钟表达式树、序列化和执行；
- `src/gflownet/minute_grammar.py`：分钟 GFlowNet 前缀语法与 71 维动作空间；
- `src/gflownet/minute_reward.py`：分钟表达式输出与日频 Reward 对齐；
- `src/gflownet/run_minute_training.py`：A100 混合精度训练入口；
- `configs/minute/training.yaml`：分钟训练配置。

## 2. 输入数据约定

训练入口读取 DolphinDB 导出或服务器本地缓存后的长表，支持 `pkl`、`parquet`、`csv`。规范字段如下：

| 字段 | 含义 |
|---|---|
| `date` | 交易日 |
| `datetime` | 分钟时间戳 |
| `code` | 股票代码 |
| `open/high/low/close` | 复权后分钟 OHLC |
| `vol` | 分钟成交量 |
| `amount` | 分钟成交额 |

同一 `date-code-datetime` 不允许重复。执行前固定按 `date, code, datetime` 稳定排序。DolphinDB 原表字段不同，应在抽取层完成 mapping；未经字段审计，不在表达式层猜测远端列名。

## 3. 图表 27：分钟特征

共 21 个叶子：

- 原始字段：`open, high, low, close, vol, amount`；
- 简单派生：`ret, vwap, hl_pct, bar_pos, amihud, rv, signed_vol, signed_amt`；
- 累积派生：`typical, vwap_cum, twap, obv, pvt`；
- 宏叶子：`logret, oc_ret`。

研报图表只给出了派生字段名称，没有逐项给出数学定义。本项目采用以下可复核口径；如果输入表已经包含同名预计算列，代码优先使用输入值，不覆盖供应方口径。

| 特征 | 缺列时的实现口径 |
|---|---|
| `ret` | `close / delay(close,1) - 1`，每日开盘首分钟为空 |
| `vwap` | `amount / vol` |
| `hl_pct` | `(high-low) / abs(close)` |
| `bar_pos` | `(close-low) / (high-low)` |
| `amihud` | `abs(ret) / abs(amount)` |
| `rv` | `ret²` |
| `signed_vol` | `sign(ret) × vol` |
| `signed_amt` | `sign(ret) × amount` |
| `typical` | `(high+low+close) / 3` |
| `vwap_cum` | 日内累计 `amount / vol` |
| `twap` | 日内 `close` 的扩展均值 |
| `obv` | 日内累计 `sign(ret) × vol` |
| `pvt` | 日内累计 `ret × vol` |
| `logret` | `log(close / delay(close,1))` |
| `oc_ret` | `close / open - 1` |

所有除法均使用安全分母；零分母、无穷值转换为空值。

## 4. 图表 28：分钟算子

共 15 个：

- 收益：`m_ret, m_logret`；
- 滞后与变化：`m_delay, m_delta`；
- 滚动统计：`m_ma, m_std`；
- 日内标准化：`m_rank, m_zscore`；
- 非线性：`m_abs, m_sign, m_log`；
- 二元运算：`m_add, m_sub, m_mul, m_div`。

`m_delay/m_delta/m_ma/m_std` 均在单日单股内部计算，绝不跨交易日。滚动统计的最小有效样本数为 `max(2, window/2)`。`m_rank/m_zscore` 使用当日完整分钟序列，适用于收盘后生成的日频信号，不应被解释为盘中实时可交易信号。

## 5. 图表 29：掩码算子

共 14 个：

- 位置：`m_head, m_tail, m_mid`；
- 极值：`m_top, m_bot, m_xtreme`；
- 分位数：`m_above, m_below, m_inner, m_outer`；
- 条件选择：`m_at_top, m_at_bot`；
- 条件过滤：`m_when_pos, m_when_gt`。

实现约定：`m_mid(x,w)` 选日内居中的 `w` 条；`m_xtreme` 选偏离日内中位数最大的 `w` 条；`inner/outer` 以内四分位距为界；`m_when_gt(x,y)` 保留 `y` 高于自身日内中位数的位置。掩码通过把未选位置设为空值实现，后续 `r_*` 只聚合保留值。

## 6. 图表 30：日内聚合算子

共 16 个：

- 基础：`r_mean, r_std, r_sum`；
- 极值：`r_max, r_min, r_median`；
- 首尾：`r_first, r_last`；
- 分布：`r_skew, r_kurt`；
- 路径：`r_slope, r_rsquare, r_argmax`；
- 二元：`r_corr, r_cov, r_wmean`。

聚合按“交易日 × 股票”输出一个值。`r_slope/r_rsquare` 对有效分钟值和顺序下标做线性回归；`r_argmax` 返回最大值位置在日内的归一化坐标 `[0,1]`；`r_wmean(x,y)` 把 `y` 作为权重。

## 7. 表达式语法与示例

分钟树根节点必须是 Reduce，防止 GFlowNet 产生无法与日频标签对齐的分钟向量：

```text
BlockExpr := ReduceOp(MinExpr)
           | ReduceOp(MaskExpr)
           | ReduceOp(MinExpr, MinExpr)
```

示例：

```text
r_corr(ret,signed_amt)
r_mean(m_head(ret,5))
r_mean(m_at_top(ret,amount,20))
r_slope(vwap_cum)
```

前缀 token 以窗口紧跟算子的形式保存，例如：

```python
["r_mean", "m_at_top", "W20", "ret", "amount"]
```

## 8. 服务器/A100 训练

先准备：

```text
data/minute_price.pkl
data/daily_price.pkl
```

其中分钟表用于执行表达式，日频表用于构造 `t+1 → t+5` 收益标签和计算截面 RankIC、Top 10% 组合收益、风险惩罚与覆盖率惩罚。修改 `configs/minute/training.yaml` 后运行：

```bash
python -m src.gflownet.run_minute_training \
  --config configs/minute/training.yaml
```

仅做 CPU 小样本冒烟测试时可添加 `--allow-non-a100`；正式训练默认强制检测 NVIDIA A100，并使用 PyTorch mixed precision。

## 9. 输出

| 输出 | 含义 |
|---|---|
| `checkpoints/gflownet_minute_best.pt` | 最优分钟 GFlowNet checkpoint |
| `results/minute_gflownet_training_metrics.csv` | 每轮 loss/reward/IC/覆盖率/耗时/GPU 日志 |
| `results/minute_gflownet_trajectory_metrics.csv` | 每条轨迹的表达式、reward 和 TB loss |
| `results/minute_alpha_pool.csv` | 分钟 Alpha 表达式库和 prefix tokens |
| `results/minute_alpha_factor_matrix.pkl` | 聚合后的日频因子矩阵，可直接接 AlphaEval/LightGBM |

需要在 2024–2026 年回测时，应把 2020–2026 分钟数据传给因子池执行；GFlowNet 的 Reward 仍只切片使用 2020–2023 年，输出矩阵则覆盖完整历史，避免重新训练表达式。

## 10. PPU 训练结束后的完整处理

阿里 PPU 全量内存入口为：

```text
notebooks/deployment/minute_ppu_ram.ipynb
```

Notebook 已包含训练结束后的完整流程，不需要再手工拼接命令：

```text
PPU GFlowNet 训练
→ 加载最佳 checkpoint
→ 生成分钟 Alpha Pool
→ 在 2020–2026 分钟数据上执行表达式
→ 日内聚合为日频因子矩阵
→ 产物与覆盖率验收
→ AlphaEval + DPP
→ LightGBM 滚动融合
→ 打包本地回测产物
```

### 10.1 首次完整运行

在 Notebook 的“选择运行阶段”单元格保持：

```python
RUN_GFLOWNET_TRAINING = True
RUN_ALPHA_EVAL = True
RUN_LIGHTGBM = True
PACKAGE_RESULTS = True
```

然后从上到下执行。分钟训练结束不是只保存模型：
`src/gflownet/run_minute_training.py` 会重新加载最佳 checkpoint，生成 Alpha Pool，并把
分钟表达式按日内 `r_*` 算子聚合为覆盖 2020–2026 的日频因子矩阵。
训练入口还会把同一份 DDB 日频聚合结果独立保存为
`results/minute_ppu_ddb_ram/daily_price.pkl`。该文件供 Reward、AlphaEval 和
LightGBM 共用，不再依赖大型 RAM 快照是否成功落盘。

### 10.2 训练已经完成时继续后处理

如果以下文件已经存在：

```text
checkpoints/gflownet_minute_ppu_ddb_ram_best.pt
results/minute_ppu_ddb_ram/alpha_pool.csv
results/minute_ppu_ddb_ram/alpha_factor_matrix.csv.gz
```

将：

```python
RUN_GFLOWNET_TRAINING = False
```

然后从“验收分钟训练产物并转换格式”单元格继续执行。这样不会重新读取 DDB、恢复
700GB RAM 数据或重新训练 GFlowNet。

旧版本训练可能没有独立的 `daily_price.pkl`。Notebook 会先尝试从旧 RAM 快照复制；
如果快照中也没有，则自动执行：

```bash
python scripts/export_ddb_daily.py \
  --config configs/minute/ppu_ddb_ram.yaml
```

该命令只在 DolphinDB 端分块聚合日频 OHLCV，不加载全量分钟数组、不生成 MemMap、
不训练 GFlowNet。默认每次查询 20 个交易日，以避免 DDB 的分区数限制；仍超限时使用
`--chunk-days 5`。输出为 `results/minute_ppu_ddb_ram/daily_price.pkl` 及其
`.metadata.json` 审计文件。

如果 AlphaEval 已经完成，只重新训练 LightGBM：

```python
RUN_GFLOWNET_TRAINING = False
RUN_ALPHA_EVAL = False
RUN_LIGHTGBM = True
PACKAGE_RESULTS = True
```

如果只需要重新打包：

```python
RUN_GFLOWNET_TRAINING = False
RUN_ALPHA_EVAL = False
RUN_LIGHTGBM = False
PACKAGE_RESULTS = True
```

跳过阶段时 Notebook 会检查对应文件是否存在，缺失时立即报错，不会静默使用其他
实验目录中的产物。

### 10.3 因子矩阵验收

Notebook 自动检查：

- checkpoint、Alpha Pool、日频因子矩阵和独立日频行情文件是否存在；
- `date-code` 是否重复；
- 是否存在因子列；
- 每个因子覆盖率是否达到 `reward.min_coverage`，默认 80%；
- 因子日期范围、股票数和交易日数；
- 将 `alpha_factor_matrix.csv.gz` 转为 AlphaEval/LightGBM 可直接读取的 Pickle。

转换结果为：

```text
results/minute_ppu_ddb_ram/alpha_factor_matrix.pkl
```

分钟原始数组和 RAM 快照不会进入后处理计算；后续全部在日频层运行。

### 10.4 AlphaEval

AlphaEval 使用 `configs/minute/ppu_ddb_ram.yaml` 中独立的 `alpha_eval` 配置，只评价
2020–2023 样本内因子，输出：

```text
results/minute_ppu_ddb_ram/alpha_eval_result.csv
```

主要指标包括 RankIC、ICIR、Top 组合 Sharpe、滚动稳定性、扰动鲁棒性、复杂度、
DPP 多样性和最终入选标记。默认 `dpp_k=30`。

### 10.5 LightGBM

LightGBM 使用 DPP 入选分钟因子，预测标签为：

```text
close(t+5) / close(t+1) - 1
```

训练与预测之间保留 5 个交易日 purge；2020–2023 提供训练历史，只输出
2024–2026 的样本外分数：

```text
results/minute_ppu_ddb_ram/lightgbm/
├── prediction_score.csv
├── model_metrics.csv
├── feature_importance.csv
├── lgbm_model.joblib
└── lgbm_window_*.joblib
```

### 10.6 打包和下载

Notebook 最后生成：

```text
results/minute_ppu_ddb_ram/postprocess_manifest.json
results/minute_ppu_ddb_ram/minute_ppu_artifacts.zip
```

压缩包包含：

- 最佳 GFlowNet checkpoint；
- 分钟 Alpha Pool；
- AlphaEval 结果；
- GFlowNet 训练与轨迹指标；
- LightGBM 最新模型、模型指标、特征重要性和预测分数；
- 本次配置和后处理 manifest。

压缩包不会包含全量 RAM 快照、原始分钟数据或完整因子矩阵，避免下载几十至数百 GB
数据。若需要在本地重新做 AlphaEval、相关性或风格归因，再单独下载
`alpha_factor_matrix.csv.gz`。

### 10.7 本地 RQAlphaPlus 回测

把压缩包解压到仓库根目录，运行：

```bash
python -m rqalpha_strategy.run_backtest \
  --config configs/minute/ppu_ddb_ram.yaml \
  --bundle ~/.rqalpha-plus/bundle \
  --predictions results/minute_ppu_ddb_ram/lightgbm/prediction_score.csv \
  --output-dir results/minute_ppu_ddb_ram/backtest_report
```

RQAlphaPlus 只在本地授权环境运行，并会清除代理变量。回测读取的是分钟因子经
AlphaEval 和 LightGBM 融合后的日频预测分数，不读取原始分钟数据。

## 11. 当前边界

本次完成的是图表 27–30 的表达式计算和 GFlowNet 搜索闭环。DolphinDB 远端表名、分区、字段类型和复权字段仍需在内网服务器做字段审计后再固化抽取脚本；大规模全市场分钟训练下一步应按研报方案接入 `(year, channel) -> (day, minute, stock)` MemMap/分块缓存，避免长表全量驻留内存。
