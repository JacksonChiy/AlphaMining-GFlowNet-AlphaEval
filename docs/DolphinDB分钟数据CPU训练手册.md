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

审计结果保存为：

```text
data/minute_ddb_cache/field_audit.json
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

## 7. 第二步：分块抽取并缓存

确认复权并把配置改成 `true` 后运行：

```bash
python scripts/prepare_ddb_minute.py \
  --config configs/minute_training_cpu_ddb.yaml
```

默认每 20 个自然日查询一次：

```yaml
chunk_days: 20
```

每个分块执行的 SQL 结构为：

```dos
select sym, date, time, open, high, low, close,
       volume, amount, tradeCount
from loadTable("dfs://数据库", "表名")
where date >= 开始日期, date <= 结束日期
order by date, sym, time
```

缓存输出：

```text
data/minute_ddb_cache/
├── manifest.json
├── field_audit.json
├── minute_20200101_20200120.pkl
├── minute_20200121_20200209.pkl
└── ...

data/daily_price_ddb.pkl
```

`daily_price_ddb.pkl` 由分钟数据聚合：

- open：当日第一根分钟开盘价；
- high：当日最高价；
- low：当日最低价；
- close：当日最后一根分钟收盘价；
- volume/amount/trade_count：日内求和；
- vwap：`amount / volume`。

强制重新抽取：

```bash
python scripts/prepare_ddb_minute.py \
  --config configs/minute_training_cpu_ddb.yaml \
  --force-refresh
```

## 8. 第三步：CPU 训练

推荐命令：

```bash
python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute_training_cpu_ddb.yaml
```

该命令会自动执行：

```text
DDB schema审计
→ 分日期块抽取或复用现有缓存
→ 分钟字段标准化
→ 聚合日频行情和构造t+1到t+5标签
→ 加载2020–2023分钟训练区间
→ CPU GFlowNet训练
→ 生成Alpha Pool
→ 按缓存分区计算2020–2026日频因子值
→ 保存因子矩阵
```

训练期间不会重复请求 DolphinDB；存在相同 fingerprint 的完整缓存时会直接输出：

```text
[DDB] cache_reused ...
```

## 9. CPU 训练输出

```text
checkpoints/gflownet_minute_cpu_ddb_best.pt

results/minute_cpu_ddb/
├── field_audit.json
├── gflownet_training_metrics.csv
├── gflownet_trajectory_metrics.csv
├── alpha_pool.csv
└── alpha_factor_matrix.pkl
```

`alpha_factor_matrix.pkl` 已经是日频矩阵，可以继续进入 AlphaEval、LightGBM 和本地 RQAlphaPlus 回测。

## 10. 防未来数据泄露说明

- DDB 查询固定按 `date, sym, time` 排序；
- `m_delay/m_delta/m_ma/m_std` 只在同一交易日、同一股票内部计算；
- 分钟因子只使用信号日当日及以前的行情；
- `m_rank/m_zscore` 使用完整当日分钟序列，因此信号定义为收盘后信号；
- 标签为 `close(t+5) / close(t+1) - 1`，只进入 Reward/评价，不进入表达式计算；
- 因子池对完整历史的执行按日期缓存分区完成，因为分钟表达式不会跨交易日，分区计算不会改变结果；
- 行业和市值字段不在当前分钟表中，因此相关风险惩罚不会生效。如需正式风险中性化，应增加时点一致的行业/市值数据源。

## 11. 内存与速度

DDB 抽取和最终因子池执行均按日期分区，不会一次性下载完整远程表。GFlowNet Reward 训练仍需要在内存中持有配置的训练区间分钟数据，这是当前 Pandas 表达式引擎的边界。

CPU 首次验证建议临时改为：

```yaml
training:
  epochs: 2
  trajectories_per_epoch: 4
  reward_workers: 2

pipeline:
  pool_size: 5
  pool_attempts: 50
```

同时先把 DDB 的抽取日期缩短到几个月，验证字段、Reward、checkpoint 和因子池后，再恢复完整区间。缩短区间只用于系统测试，正式实验仍应保持训练期和样本外期隔离。

## 12. 常见错误

### Missing DolphinDB connection environment variables

当前 shell 没有设置完整的连接环境变量。重新执行第 4 节。

### DolphinDB field audit failed

实际 schema 与用户提供的字段名或类型不一致。查看 `field_audit.json`，不要在代码中猜测替换列名。

### prices_are_adjusted=true

尚未确认复权。确认数据口径后修改配置；如果是未复权价格，应先补充复权处理。

### source starts later / ends earlier

DDB 表的实际日期范围不能覆盖配置日期。修改抽取日期或补齐源数据。

### 内存不足

减小训练日期范围或使用明确的股票 Universe。不要随机删除分钟行，因为会破坏日内路径、Mask 和聚合算子的定义。
