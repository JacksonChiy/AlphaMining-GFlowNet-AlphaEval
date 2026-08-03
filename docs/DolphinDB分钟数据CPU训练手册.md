# DolphinDB 分钟数据 CPU 训练手册

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
5. 表中的 OHLC 是否已经完成前复权或后复权。

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
```

`read -s` 输入密码时终端不会显示字符，也不会把密码写入 shell 命令历史。

检查非敏感变量：

```bash
echo "$DDB_HOST:$DDB_PORT"
echo "$DDB_DATABASE/$DDB_TABLE"
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
```

保存后关闭并重新打开 PowerShell。

## 5. 修改复权确认

打开：

```text
configs/minute_training_cpu_ddb.yaml
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
  --config configs/minute_training_cpu_ddb.yaml \
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
```

`chunk_days`控制返回原始分钟行的Reward查询，建议保持1天；另外两个参数只执行聚合统计并返回小结果，可以使用120天以减少远程调用次数。

确认复权并把 `prices_are_adjusted` 改成 `true` 后运行：

```bash
python scripts/prepare_ddb_minute.py \
  --config configs/minute_training_cpu_ddb.yaml
```

该命令在流模式下只检查字段和日期，不下载分钟文件。训练时每个交易日执行的 SQL 结构为：

```dos
select sym, date, time, open, high, low, close,
       volume, amount, tradeCount
from loadTable("dfs://数据库", "表名")
where date >= 开始日期, date <= 结束日期
order by date, sym, time
```

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
  --config configs/minute_training_cpu_ddb.yaml
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_cpu.py --mode minute --config configs/minute_training_cpu_ddb.yaml
```

该命令会自动执行：

```text
DDB schema审计
→ DolphinDB服务端聚合日频行情
→ 构造t+1到t+5标签
→ GFlowNet批量生成表达式
→ 按交易日流式读取2020–2023分钟数据
→ 分钟算子 + Mask + Reduce生成日频Block
→ 复用日频Block缓存并执行日频Tree
→ 计算IC、LongIR、风险和覆盖率Reward
→ CPU GFlowNet训练
→ 生成Alpha Pool
→ 一次流式扫描计算2020–2026已选因子
→ 保存日频因子CSV.GZ
```

同一批表达式共享一次DolphinDB扫描，而不是每个表达式分别扫描。重复日内Block直接从内存缓存命中：

```text
[DDBReward] chunk_complete ... new_blocks=... expressions=...
[GFlowNet] reward_progress ... cache_hit_rate=...
```

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

DDB Reward和最终因子池执行均按交易日流式查询。内存中只同时存在一个查询分块、日频标签面板和Reduce后的日频Block，不再持有完整训练区间分钟表。

研报使用MemMap减少重复读取；DDB版本用两层机制替代：

- 一批轨迹共享一次DDB扫描；
- `date_scope + block_expr`对应的日频Block在内存LRU中复用。

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

### 内存不足

保持`chunk_days: 1`，减小训练日期范围或使用明确的股票Universe。不要随机删除分钟行，因为会破坏日内路径、Mask和聚合算子的定义。
