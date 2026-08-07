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
    minute_sessions:
      - ['09:31:00', '11:30:00']
      - ['13:01:00', '15:00:00']
    minute_extra_times:
      - '09:25:00'
    stock_tile_size: 256
    build_workers: 4
    workers: 4
    flush_every_days: 8
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

在构建前，先运行只读质量审计：

```powershell
python scripts/audit_ddb_minute_quality.py `
  --config configs/minute_training_cpu_ddb.yaml `
  --scope grid
```

结果保存在`results/minute_cpu_ddb/data_quality/`：

- `summary.json`：总体通过/失败状态和问题日期数；
- `date_quality.csv`：每个交易日的实际分钟数、多出/缺失时间、重复行和值异常；
- `time_presence.csv`：每个时间点出现在多少个交易日，以及是否属于预期连续竞价时段；
- `duplicate_keys.csv`：重复的`date + sym + time`键；
- `value_issues.csv`：空值、非正价格、OHLC关系错误、负成交量/成交额。

当需要继续定位到具体股票时，运行：

```powershell
python scripts/audit_ddb_minute_quality.py `
  --config configs/minute_training_cpu_ddb.yaml `
  --scope full
```

`full`会额外生成`problem_symbols.csv`，数据库计算量和本地传输量都大于`grid`，建议先跑`grid`。固定的241分钟口径为`09:25`集合竞价1一根 + `09:31-11:30` 120根 + `13:01-15:00` 120根。其他时间点会在MemMap构建时被显式过滤，且不参与派生分钟特征或日频OHLCV的计算。

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
[MemMapBuild] year_start ... build_workers=4 ...
[MemMapBuild] day_complete ...
[MemMapBuild] batch_flushed ...
[MemMapBuild] complete ...
```

构建中断后，使用相同配置和目录重新执行同一命令，会跳过`progress.json`记录的所有已完成交易日，不要求并行任务按日期顺序完成。

### 构建并发参数

- `build_workers`：构建线程数，每个线程创建独立DolphinDB session。
- `workers`：训练时的MemMap读取和因子Block并发，与`build_workers`无关。
- `flush_every_days`：每批至少完成多少个交易日后统一落盘，实际批次不会小于`build_workers`。

Windows 16核CPU建议从`build_workers: 4`开始。如果DDB服务器CPU、网络和并发连接数仍有余量，可提高到8；如果出现连接拒绝、查询排队或服务器CPU持续满载，应降回2至4，不建议直接设成16。

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
`minute_grid_audit.json`保存抽样日的原始时间网格差异、被排除时点和最终选中的241个时间点。

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
[MemMapReward] numpy_start ... date_chunk_days=10 ... blocks_per_task=2 ... estimated_peak_mb_per_worker=...
[MemMapReward] numpy_progress tasks=... rate=... eta=... read_sum_seconds=... compute_sum_seconds=... cache_write_seconds=...
[MemMapReward] stage_complete stage=partial_cache_scan ...
[MemMapReward] stage_complete stage=task_execution ...
[MemMapReward] stage_complete stage=result_materialization ...
[MemMapReward] stage_summary total_seconds=... partial_cache_scan_seconds=... task_execution_seconds=...
[MinuteRewardBatch] stage_summary ... block_execution_seconds=... expression_assembly_seconds=... factor_evaluation_seconds=...
[GFlowNet] stage_summary epoch=... sampling_seconds=... reward_seconds=... backward_seconds=... optimizer_seconds=... bottleneck=...
[MemMapReward] numpy_complete ... pandas_rows_built=0
```

### 10.1 分阶段性能日志怎么读

日志按三层输出，不需要开启额外配置：

1. `MemMapReward`：分钟块内部耗时。`partial_cache_scan`是完成位图扫描，
   `task_execution`是并行读取和NumPy算子计算，`result_materialization`是把二维日频结果
   整理为因子序列。
2. `MinuteRewardBatch`：一批表达式的耗时。重点区分`block_execution`、
   `expression_assembly`和`factor_evaluation`（IC、IR、覆盖率及风险惩罚）。
3. `GFlowNet`：每个epoch的耗时。日志直接给出`bottleneck`，候选值包括
   `sampling`、`reward`、`loss_build`、`trajectory_logging`、`backward`、
   `optimizer`和`checkpoint`。

`worker_read_sum_seconds`与`worker_compute_sum_seconds`是所有并行Worker耗时之和，
用于判断I/O和算子计算的相对占比；它们不是墙钟时间，多Worker并行时可能大于
`task_execution seconds`。`input_gb / task_execution seconds`可近似观察实际读取吞吐。
`parent_cache_write_seconds`较高通常表示本地缓存落盘受限；`partial_cache_scan`较高则说明
完成位图扫描或旧缓存迁移是主要开销。

不应再出现：

```text
[DDBStream] minute_rows=...
[DDBPushdown] ...
```

## 11. Block持久化缓存

计算中的小块和完成后的日内Block分别保存为：

```text
E:\AlphaMining\block_cache\
├── partials_v2\日期范围哈希\表达式哈希\
│   ├── values.npy
│   ├── completed.npy
│   └── metadata.json
├── 表达式哈希.npy
└── 表达式哈希.json
```

缓存键包含：

```text
分钟源指纹 + 日期范围 + 完整block表达式
```

同一数据源和日期范围下，重新运行训练可直接命中已有Block。源表、复权状态或日期范围发生变化时不会错误复用旧缓存。

训练仍按`reward_chunk_days × stock_tile_size`计算，但不再为每个计算块创建小文件。
每个表达式只使用一个二维`values.npy`、一个同形状完成位图`completed.npy`和一个
`metadata.json`。主进程先刷新因子值，再设置完成位图；异常中断最多造成少量安全重算，
不会把半写结果当成有效缓存。日期块和股票块参数发生变化后，完成位图仍可按新切片命中。
只有日内Reduce完成后的二维日频结果才转换为Pandas，远程DDB查询数始终为0。

旧版`partials`小文件会在首次访问时自动复制进`partials_v2`，日志显示：

```text
partial_cache=consolidated_v2 files_per_block=3 legacy_migrated=...
```

旧目录不会自动删除，避免迁移中断时丢失恢复点。确认完整Block已经生成后，可自行归档旧目录。

默认优化参数：

```yaml
memmap:
  workers: 4
  reward_backend: numpy
  reward_chunk_days: 10
  reward_blocks_per_task: 2
  reward_group_by_channels: true
  reward_cache_commit_tasks: 10
  reward_parallel_backend: loky
  numpy_fallback: true
```

内存不足时先把`reward_blocks_per_task`调为1，再把`workers`调为2；仍不足时把
`reward_chunk_days`调为5。机械硬盘任务过碎且内存充足时可把`reward_chunk_days`提高到20。
`reward_backend: pandas`只用于结果核验或兼容未来尚未
迁移的新算子，不建议正式训练使用。

若出现以下警告：

```text
A worker stopped while some jobs were given to the executor
```

通常表示Windows终止了内存峰值过高的子进程，而不是DolphinDB或SATA磁盘卡住。新版会
缩小表达式批次；若worker仍真正退出，会打印`worker_terminated sequential_resume=true`
并从已落盘的小块单进程续算。稳定性优先时可改为：

```yaml
memmap:
  workers: 2
  reward_chunk_days: 5
  reward_blocks_per_task: 1
  reward_parallel_backend: threading
```

`threading`不会创建`loky`子进程，但包含大量Python分组循环的算子可能更慢。正式训练建议
使用标准64位Python 3.11或3.12虚拟环境；Windows embeddable发行版更适合程序嵌入，
不建议作为长时间多进程训练环境。

## 11.1 分钟算子向量化

正式训练默认使用批量NumPy路径：

- `m_rank`对无并列的有效组批量排序，存在并列值时保留平均排名精确路径；
- `m_zscore`沿分钟轴批量计算样本标准差；
- `m_ma`和`m_std`对完整241分钟组使用二维累计和，缺分钟组按有效行压缩计算；
- `m_top`、`m_bot`、`m_xtreme`和条件掩码使用批量排名或分位数；
- `r_corr`、`r_cov`、`r_wmean`、`r_skew`、`r_kurt`、`r_slope`、
  `r_rsquare`、`r_argmax`使用批量中心化统计公式；
- 上市前、退市后或全天无分钟记录的股票日直接跳过。

所有滚动运算仍只使用当日当前分钟及之前的数据；日期和证券分组边界没有改变。
可以用以下命令测试当前机器的单任务速度：

```powershell
python scripts/benchmark_minute_numpy.py `
  --days 10 `
  --minutes 241 `
  --stocks 256 `
  --repeats 5
```

输出中的`million_block_elements_per_second`用于比较同一台机器、相同Python环境和相同
代码版本，不建议直接与不同CPU或不同任务形状的结果比较。

## 11.2 按通道依赖打包Block

`reward_group_by_channels: true`会在不改变`reward_blocks_per_task`的前提下重新排列
待计算Block。只依赖`close`的表达式优先进入同一任务，依赖`ret`和`vol`的表达式优先
进入另一任务，从而共享一次MemMap切片读取和同一执行器中的公共子表达式缓存。

程序同时计算原顺序与依赖分组的计划成本；若依赖分组没有减少通道切片，则自动保留
原顺序，因此该优化不会增加计划读取次数。训练日志示例：

```text
channel_grouping=true channel_slices=4200/6800 channel_slice_savings=38.2%
```

这里的`slices`是“一个任务读取一个通道切片”的次数，不是DolphinDB查询次数。训练期间
远程查询仍为0。关闭该优化只需设置：

```yaml
memmap:
  reward_group_by_channels: false
```

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
