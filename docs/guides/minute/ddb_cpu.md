# DolphinDB 分钟数据 CPU 训练手册

> 当前默认配置已切换为DDB与训练分离的MemMap模式。本文件后半部分保留`load_mode: stream`兼容路径；正式两机部署请优先使用[《DDB与训练分离的MemMap运行手册》](ddb_memmap.md)。

## 1. 数据源字段审计

当前确认的 DolphinDB 分钟表字段为：

```text
sym, date, time, open, high, low, close,
volume, amount, tradeCount
```

字段映射如下：

| DolphinDB 字段 | 训练标准字段 | 状态 | 用途 |
|---|---|---|---|
| `sym` | `code` | direct | 股票代码与分组键 |
| `date` | `date` | direct | 交易日 |
| `date + time` | `datetime` | derived | 分钟时间戳 |
| `open` | `open` | direct | 分钟开盘价 |
| `high` | `high` | direct | 分钟最高价 |
| `low` | `low` | direct | 分钟最低价 |
| `close` | `close` | direct | 分钟收盘价 |
| `volume` | `vol` | direct | 分钟成交量 |
| `amount` | `amount` | direct | 分钟成交额 |
| `tradeCount` | `trade_count` | direct | 成交笔数，目前保留用于审计，尚未进入 Grammar |

程序连接后会执行 `schema(loadTable(...)).colDefs`，检查：

- 字段是否真实存在；
- `sym` 是否为 SYMBOL/STRING；
- `date` 是否为 DATE；
- `time` 是否为 MINUTE/SECOND/TIME/NANOTIME 等时间类型；
- 行情和成交字段是否为数值类型；
- 数据实际最早、最晚日期是否覆盖配置区间；
- OHLC 是否已经复权。

流式模式的审计结果保存为：

```text
results/minute_cpu_ddb/field_audit.json
```

其中包含字段映射、源表类型、时间范围、标准输出约定和实际生成的 `dataSql`。

## 2. 当前必须确认的信息

正式运行前需要确认：

1. DolphinDB 主机和端口；
2. 用户名和密码；
3. DFS 数据库路径，例如 `dfs://minuteBars`；
4. 分区表名；
5. `TradeDays` 所在的独立 DFS 数据库路径；
6. 表中的 OHLC 是否已经完成前复权或后复权。

分钟表和交易日表可以位于两个不同的DFS数据库。`TradeDays`的日期列不会由代码猜测：如果Schema中恰好只有一个`DATE`列，程序会自动识别；如果存在多个`DATE`列，需在配置中明确填写：

```yaml
trade_days_table: TradeDays
trade_days_date_column: 实际日期列名
```

当前字段没有 `adjFactor`。如果 OHLC 是未复权价格，不能直接用于跨日未来收益标签，应先在 DolphinDB 表中完成复权或补充复权因子。

## 3. 安装

进入服务器上的项目目录：

```bash
cd /path/to/AlphaMining-GFlowNet-AlphaEval

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-ddb.txt
```

`requirements-ddb.txt` 在原训练依赖基础上增加 DolphinDB 官方 Python API。官方 API 使用 `Session.connect` 建立连接，并通过 `Session.run` 返回 Pandas DataFrame；任务结束后代码会显式关闭 Session。

## 4. 设置连接信息

连接信息只从环境变量读取，不写进 YAML 和 Git：

```bash
export DDB_HOST='内网服务器地址'
export DDB_PORT='8848'
export DDB_USER='用户名'

read -s DDB_PASSWORD
export DDB_PASSWORD

export DDB_DATABASE='dfs://实际数据库路径'
export DDB_TABLE='实际分区表名'
export DDB_TRADE_DAYS_DATABASE='dfs://交易日数据库路径'
```

`read -s` 输入密码时终端不会显示字符，也不会把密码写入 shell 命令历史。

检查非敏感变量：

```bash
echo "$DDB_HOST:$DDB_PORT"
echo "$DDB_DATABASE/$DDB_TABLE"
echo "$DDB_TRADE_DAYS_DATABASE/TradeDays"
```

不要运行 `echo "$DDB_PASSWORD"`。

Windows PowerShell 永久保存到当前用户时使用：

```powershell
[Environment]::SetEnvironmentVariable("DDB_HOST", "服务器地址", "User")
[Environment]::SetEnvironmentVariable("DDB_PORT", "8848", "User")
[Environment]::SetEnvironmentVariable("DDB_USER", "用户名", "User")
[Environment]::SetEnvironmentVariable("DDB_PASSWORD", "密码", "User")
[Environment]::SetEnvironmentVariable("DDB_DATABASE", "dfs://CYC", "User")
[Environment]::SetEnvironmentVariable("DDB_TABLE", "minute_bar", "User")
[Environment]::SetEnvironmentVariable("DDB_TRADE_DAYS_DATABASE", "dfs://实际交易日库", "User")
```

保存后关闭并重新打开 PowerShell。

## 5. 修改复权确认

打开：

```text
configs/minute/cpu_ddb_memmap.yaml
```

默认值为：

```yaml
prices_are_adjusted: false
```

先完成字段审计。只有确认 OHLC 已经复权后才改为：

```yaml
prices_are_adjusted: true
```

如果保持 `false`，审计可以运行，但数据抽取和训练会停止。

## 6. 第一步：只做字段审计

```bash
source .venv/bin/activate

python scripts/prepare_ddb_minute.py \
  --config configs/minute/cpu_ddb_memmap.yaml \
  --audit-only
```

输出：

```text
results/minute_cpu_ddb/field_audit.json
```

检查：

- `data[*].requiredFields` 不应出现 `missing`；
- `timeRange.sourceStart/sourceEnd` 应覆盖研究区间；
- `sourceAudit.source_schema` 的类型应符合实际表结构；
- `pricesAreAdjusted` 应与真实行情口径一致；
- `dataSql` 中只应包含已确认的十个源字段。

## 7. 第二步：确认研报式流模式

配置应保持：

```yaml
load_mode: stream
chunk_days: 1
audit_chunk_days: 120
daily_aggregate_chunk_days: 120
trade_days_table: TradeDays
trade_days_database_env: DDB_TRADE_DAYS_DATABASE
pushdown_enabled: true
pushdown_fallback: true
```

`chunk_days`按交易日而不是自然日计数，建议保持1天；程序先查询`TradeDays`，周末和节假日不会再向分钟表发起空查询。另外两个参数只执行聚合统计并返回小结果，可以使用120天以减少远程调用次数。

确认复权并把 `prices_are_adjusted` 改成 `true` 后运行：

```bash
python scripts/prepare_ddb_minute.py \
  --config configs/minute/cpu_ddb_memmap.yaml
```

该命令在流模式下只检查字段和日期，不下载分钟文件。训练时优先执行服务端分钟算子和Reduce，SQL结构为：

```dos
source = select ... from loadTable("dfs://数据库", "表名")
         where date >= 交易日, date <= 交易日
         order by date, sym, time
vectors = select ..., mavg(...), rank(...), ...
          from source context by date, sym csort time
select avg(...), corr(...), ...
from vectors group by date, sym
```

因此网络传输的是每日每只股票一行的日频Block，而不是约百万行的原始分钟数据。

命令应输出：

```text
[DDB] stream_ready raw_minute_files=false
```

程序不会创建 `minute_*.pkl` 或 `daily_price_ddb.pkl`。标签所需日行情直接由 DolphinDB 聚合后返回内存：

- open：当日第一根分钟开盘价；
- high：当日最高价；
- low：当日最低价；
- close：当日最后一根分钟收盘价；
- volume/amount/trade_count：日内求和；
- vwap：`amount / volume`。

## 8. 第三步：CPU 训练

推荐命令：

```bash
python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute/cpu_ddb_memmap.yaml
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_cpu.py --mode minute --config configs/minute/cpu_ddb_memmap.yaml
```

该命令会自动执行：

```text
DDB schema审计
→ DolphinDB服务端聚合日频行情
→ 构造t+1到t+5标签
→ GFlowNet批量生成表达式
→ 从TradeDays读取2020–2023真实交易日
→ 支持的分钟算子 + Mask + Reduce下推到DDB生成日频Block
→ 暂不支持下推的复杂算子使用NumPy分组内核回退
→ 复用日频Block缓存并执行日频Tree
→ 计算IC、LongIR、风险和覆盖率Reward
→ CPU GFlowNet训练
→ 生成Alpha Pool
→ 按同一规则计算2020–2026已选因子
→ 保存日频因子CSV.GZ
```

同一批表达式共享一次DolphinDB分区装载，而不是每个表达式分别扫描。重复日内Block直接从内存缓存命中。日志会明确显示执行计划：

```text
[DDB] trade_days_loaded count=...
[DDBReward] execution_plan pushdown_blocks=... numpy_fallback_blocks=... coarse_screen=false
[DDBPushdown] chunk=... blocks=... daily_rows=...
[GFlowNet] reward_progress ... cache_hit_rate=...
```

当前安全下推范围包括基础/衍生分钟特征、四则运算、分钟收益/排名/ZScore、`m_delay/m_delta/m_ma/m_std`、基于排名或分位数的Mask，以及常用Reduce（均值、标准差、求和、极值、中位数、首尾、相关、协方差、加权均值）。`r_skew/r_kurt/r_slope/r_rsquare/r_argmax`和位置型`m_head/m_tail/m_mid`使用NumPy回退，以保证与本地定义一致。

这里没有粗筛：每个GFlowNet候选表达式都进入完整区间Reward评估。`coarse_screen=false`会写入每批日志。

## 9. CPU 训练输出

```text
checkpoints/gflownet_minute_cpu_ddb_best.pt

results/minute_cpu_ddb/
├── field_audit.json
├── gflownet_training_metrics.csv
├── gflownet_trajectory_metrics.csv
├── alpha_pool.csv
└── alpha_factor_matrix.csv.gz
```

`alpha_factor_matrix.csv.gz` 是研报定义的“分钟信息生成、日内聚合后输出”的日频因子矩阵，可以继续进入 AlphaEval、LightGBM 和本地 RQAlphaPlus 回测。原始分钟数据不会落地到本机。

## 10. 防未来数据泄露说明

- DDB 查询固定按 `date, sym, time` 排序；
- `m_delay/m_delta/m_ma/m_std` 只在同一交易日、同一股票内部计算；
- 分钟因子只使用信号日当日及以前的行情；
- `m_rank/m_zscore` 使用完整当日分钟序列，因此信号定义为收盘后信号；
- 标签为 `close(t+5) / close(t+1) - 1`，只进入 Reward/评价，不进入表达式计算；
- 日内Block按完整交易日计算，不在交易日内部截断，因此Mask、日内rank和Reduce语义不变；
- Reduce之后的 `ts_*` 日频算子在完整日频Block上执行，滚动窗口只使用当日及历史日；
- 行业和市值字段不在当前分钟表中，因此相关风险惩罚不会生效。如需正式风险中性化，应增加时点一致的行业/市值数据源。

## 11. 内存与速度

DDB Reward和最终因子池执行均由`TradeDays`驱动。完全可下推的表达式只在Python中保留日频标签面板和Reduce后的日频Block；包含回退算子的批次才会临时读取一个交易日的原始分钟行，计算完成后立即释放。

研报使用MemMap减少重复读取；DDB版本用两层机制替代：

- 一批轨迹共享一次DDB扫描；
- `date_scope + block_expr`对应的日频Block在内存LRU中复用。
- DDB服务端完成分钟算子和日内Reduce，显著减少网络传输；
- 复杂回退Reduce使用NumPy数组内核，不再使用`DataFrameGroupBy.apply`。

CPU 首次验证建议临时改为：

```yaml
training:
  epochs: 2
  trajectories_per_epoch: 4
  reward_workers: 1

pipeline:
  pool_size: 5
  pool_attempts: 50
```

同时先把DDB日期缩短到3至6个月，验证字段、Reward、checkpoint和因子池后，再恢复完整区间。流模式必须保持 `reward_workers: 1`，否则会产生重复数据库扫描。

## 12. 常见错误

### Missing DolphinDB connection environment variables

当前 shell 没有设置完整的连接环境变量。重新执行第 4 节。

### DolphinDB field audit failed

实际 schema 与用户提供的字段名或类型不一致。查看 `field_audit.json`，不要在代码中猜测替换列名。

### prices_are_adjusted=true

尚未确认复权。确认数据口径后修改配置；如果是未复权价格，应先补充复权处理。

### source starts later / ends earlier

DDB表的实际日期范围不能覆盖配置日期。错误信息会显示`requested_end`和`actual_last_trade_date`；将`dataset.dolphindb.end_date`与`dataset.out_of_sample_end_date`改成实际最后交易日。

### TradeDays date column

如果`TradeDays`没有DATE列或存在多个DATE列，审计会停止并打印全部列名。请查看真实Schema后设置`trade_days_date_column`，不要凭名称猜测。

### Missing DolphinDB TradeDays database environment variable

当前Shell没有设置`DDB_TRADE_DAYS_DATABASE`。该变量应填写`TradeDays`所在的完整`dfs://`库路径，它与分钟表使用的`DDB_DATABASE`相互独立。

### pushdown_failed fallback_to_numpy=true

DDB版本或函数语法与当前环境不兼容。默认会自动转为NumPy并保证训练继续，但速度会下降。先把`pushdown_fallback: false`运行一小段日期，可获得原始服务端错误并定位不兼容算子；修正后再恢复完整训练。

### 内存不足

保持`chunk_days: 1`，减小训练日期范围或使用明确的股票Universe。不要随机删除分钟行，因为会破坏日内路径、Mask和聚合算子的定义。
