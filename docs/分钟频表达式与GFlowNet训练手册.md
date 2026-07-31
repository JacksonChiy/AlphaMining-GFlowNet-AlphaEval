# 分钟频表达式与 GFlowNet 训练手册

## 1. 实现范围

本模块复现研报《基于 GFlowNet 和 AlphaEval 的分钟频因子挖掘筛选框架》图表 27、28、29、30 中的分钟特征与算子。表达式先在每个“交易日 × 股票”的分钟序列内计算，最后必须经过 `r_*` 聚合成日频因子，才能接入现有日频 Reward、AlphaEval、LightGBM 和本地 RQAlphaPlus 回测。

对应代码：

- `src/operators/minute.py`：分钟特征、分钟算子、掩码和日内聚合；
- `src/expression/minute.py`：分钟表达式树、序列化和执行；
- `src/gflownet/minute_grammar.py`：分钟 GFlowNet 前缀语法与 71 维动作空间；
- `src/gflownet/minute_reward.py`：分钟表达式输出与日频 Reward 对齐；
- `src/gflownet/run_minute_training.py`：A100 混合精度训练入口；
- `configs/minute_training_config.yaml`：分钟训练配置。

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

其中分钟表用于执行表达式，日频表用于构造 `t+1 → t+5` 收益标签和计算截面 RankIC、Top 10% 组合收益、风险惩罚与覆盖率惩罚。修改 `configs/minute_training_config.yaml` 后运行：

```bash
python -m src.gflownet.run_minute_training \
  --config configs/minute_training_config.yaml
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

## 10. 当前边界

本次完成的是图表 27–30 的表达式计算和 GFlowNet 搜索闭环。DolphinDB 远端表名、分区、字段类型和复权字段仍需在内网服务器做字段审计后再固化抽取脚本；大规模全市场分钟训练下一步应按研报方案接入 `(year, channel) -> (day, minute, stock)` MemMap/分块缓存，避免长表全量驻留内存。
