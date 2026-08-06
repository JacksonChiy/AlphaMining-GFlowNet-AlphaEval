# DDB与训练分离的MemMap运行手册

## 1. 部署结构

```text
DolphinDB服务器
  └── minute_bar与TradeDays
          │ 一次性内网读取
          ▼
Windows训练服务器本地NVMe
  ├── 按年/通道float32 MemMap
  ├── 日频行情CSV.GZ
  ├── 持久化日内Block缓存
  └── GFlowNet CPU训练
```

MemMap只保存在训练服务器，不占用DDB服务器磁盘。构建完成后，GFlowNet训练阶段不连接DDB。

## 2. 更新代码

```powershell
cd D:\alpha-mining-gflow-net-alpha-eval
git switch cpu-training
git pull --ff-only gitee cpu-training
```

## 3. 准备训练服务器本地磁盘

推荐使用独立NVMe，例如：

```text
E:\AlphaMining\minute_memmap
E:\AlphaMining\block_cache
```

建议预留150至250GB空间。不要使用SMB、NAS或其他网络映射盘。

永久设置目录：

```powershell
[Environment]::SetEnvironmentVariable(
    "ALPHAMINING_MEMMAP_DIR",
    "E:\AlphaMining\minute_memmap",
    "User"
)
[Environment]::SetEnvironmentVariable(
    "ALPHAMINING_BLOCK_CACHE_DIR",
    "E:\AlphaMining\block_cache",
    "User"
)
```

当前PowerShell立即生效：

```powershell
$env:ALPHAMINING_MEMMAP_DIR = "E:\AlphaMining\minute_memmap"
$env:ALPHAMINING_BLOCK_CACHE_DIR = "E:\AlphaMining\block_cache"
```

## 4. 仅在构建阶段配置DDB

```powershell
$env:DDB_HOST = "DDB服务器IP"
$env:DDB_PORT = "8848"
$env:DDB_USER = "用户名"
$env:DDB_PASSWORD = "密码"
$env:DDB_DATABASE = "dfs://分钟数据库"
$env:DDB_TABLE = "minute_bar"
$env:DDB_TRADE_DAYS_DATABASE = "dfs://交易日数据库"
```

这些变量只用于构建MemMap。构建结束后训练不再读取DDB。

## 5. 检查配置

打开`configs/minute_training_cpu_ddb.yaml`，确认：

```yaml
dataset:
  source: dolphindb
  mining_start_date: '2020-01-01'
  mining_end_date: '2023-12-31'
  out_of_sample_start_date: '2024-01-01'
  out_of_sample_end_date: '数据库实际最后日期'

  dolphindb:
    load_mode: memmap
    start_date: '2020-01-01'
    end_date: '数据库实际最后日期'
    trade_days_database_env: DDB_TRADE_DAYS_DATABASE
    trade_days_table: TradeDays
    prices_are_adjusted: true

  memmap:
    root_env: ALPHAMINING_MEMMAP_DIR
    block_cache_env: ALPHAMINING_BLOCK_CACHE_DIR
    expected_minutes: 241
    stock_tile_size: 256
    workers: 4
    flush_every_days: 1
    force_rebuild: false
```

只有确认分钟OHLC已复权后才能设置`prices_are_adjusted: true`。

如果实际分钟网格不是241个点，构建程序会停止并打印实际数量。先核对时间字段口径，再修改`expected_minutes`，不要为了通过检查直接改数字。

## 6. 安装环境

```powershell
cd D:\alpha-mining-gflow-net-alpha-eval
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-ddb.txt
```

## 7. 先构建短区间MemMap

首次验证建议复制配置：

```powershell
Copy-Item configs\minute_training_cpu_ddb.yaml configs\minute_training_memmap_smoke.yaml
```

将日期改为约3个月，并使用一个单独目录：

```powershell
$env:ALPHAMINING_MEMMAP_DIR = "E:\AlphaMining\minute_memmap_smoke"
$env:ALPHAMINING_BLOCK_CACHE_DIR = "E:\AlphaMining\block_cache_smoke"
```

运行构建：

```powershell
python scripts/build_ddb_memmap.py `
  --config configs/minute_training_memmap_smoke.yaml
```

日志顺序：

```text
[DDB] audit_progress ...
[DDB] trade_days_loaded ...
[MemMapBuild] minute_grid_ready count=241 ...
[MemMapBuild] year_start ...
[MemMapBuild] day_complete ...
[MemMapBuild] complete ...
```

构建中断后，使用相同配置和目录重新执行同一命令，会从`progress.json`记录的完整交易日继续。

## 8. MemMap目录

```text
E:\AlphaMining\minute_memmap\
├── manifest.json
├── field_audit.json
├── stocks.npy
├── minute_grid.npy
├── daily_price.csv.gz
├── 2020\
│   ├── dates.npy
│   ├── open.npy
│   ├── high.npy
│   ├── low.npy
│   ├── close.npy
│   ├── volume相关及派生通道.npy
│   ├── valid_mask.npy
│   └── progress.json
├── 2021\
└── ...
```

每个通道的形状为：

```text
(n_trade_days, 241, n_stocks)
```

数值通道使用`float32`，有效值Mask使用`uint8`。没有生成原始分钟PKL。
`field_audit.json`记录构建时的字段映射、日期覆盖和复权检查结果，便于在无DDB连接的训练服务器上追溯数据口径。

训练计算按`year × stock_tile`切分，并由`workers`个loky进程只读共享操作系统Page Cache。Windows首次建议使用`workers: 2`，确认内存稳定后再提高到4或8。

## 9. 构建完成后断开DDB

可在当前PowerShell删除连接变量，验证训练不使用DDB：

```powershell
Remove-Item Env:DDB_HOST -ErrorAction SilentlyContinue
Remove-Item Env:DDB_PORT -ErrorAction SilentlyContinue
Remove-Item Env:DDB_USER -ErrorAction SilentlyContinue
Remove-Item Env:DDB_PASSWORD -ErrorAction SilentlyContinue
Remove-Item Env:DDB_DATABASE -ErrorAction SilentlyContinue
Remove-Item Env:DDB_TABLE -ErrorAction SilentlyContinue
Remove-Item Env:DDB_TRADE_DAYS_DATABASE -ErrorAction SilentlyContinue
```

保留：

```powershell
echo $env:ALPHAMINING_MEMMAP_DIR
echo $env:ALPHAMINING_BLOCK_CACHE_DIR
```

## 10. MemMap训练

```powershell
python scripts/train_cpu.py `
  --mode minute `
  --config configs/minute_training_cpu_ddb.yaml `
  --threads 8
```

正常日志必须包含：

```text
[MinuteGFlowNet] memmap_enabled remote_ddb_queries_during_training=0
[MemMapReward] execution_plan ... remote_ddb_queries=0 coarse_screen=false
[MemMapRead] progress=...
```

不应再出现：

```text
[DDBStream] minute_rows=...
[DDBPushdown] ...
```

## 11. Block持久化缓存

每个日内Block保存为：

```text
E:\AlphaMining\block_cache\
├── 表达式哈希.npy
└── 表达式哈希.json
```

缓存键包含：

```text
分钟源指纹 + 日期范围 + 完整block表达式
```

同一数据源和日期范围下，重新运行训练可直接命中已有Block。源表、复权状态或日期范围发生变化时不会错误复用旧缓存。

## 12. 正式构建与训练

短区间验证通过后，将环境变量恢复到正式目录，配置恢复到完整日期：

```powershell
$env:ALPHAMINING_MEMMAP_DIR = "E:\AlphaMining\minute_memmap"
$env:ALPHAMINING_BLOCK_CACHE_DIR = "E:\AlphaMining\block_cache"

python scripts/build_ddb_memmap.py `
  --config configs/minute_training_cpu_ddb.yaml

python scripts/train_cpu.py `
  --mode minute `
  --config configs/minute_training_cpu_ddb.yaml `
  --threads 8
```

第一次构建耗时较长，但后续epoch、重复实验和中断重跑都不再传输远程分钟数据。

## 13. 重新构建

以下变化需要使用新目录或完整重建：

- 日期范围变化；
- 分钟源库或表变化；
- 复权口径变化；
- 通道定义变化；
- 股票代码或分钟时间网格口径变化。

推荐使用新目录保留旧实验。确需覆盖时设置：

```yaml
memmap:
  force_rebuild: true
```

确认构建完成后再恢复为`false`。
