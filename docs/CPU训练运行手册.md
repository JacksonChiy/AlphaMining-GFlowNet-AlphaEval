# CPU 训练运行手册

## 1. 适用范围

本手册对应 `cpu-training` 版本，支持：

- 日频 GFlowNet Alpha 挖掘；
- 分钟频 GFlowNet Alpha 挖掘；
- 强制使用 CPU，即使机器安装了 CUDA 或存在可用 GPU；
- 独立保存 CPU checkpoint、训练日志、Alpha Pool 和因子矩阵，不覆盖 GPU 实验。

CPU 版本适合本地功能验证、算子调试、小规模 Universe、短周期实验和服务器没有 GPU 的场景。全市场、多年分钟数据的主要耗时在表达式执行和 Reward 计算，CPU 正式训练通常明显慢于 A100，不能把它视为 GPU 版本的等速替代品。

## 2. 文件说明

| 文件 | 作用 |
|---|---|
| `scripts/train_cpu.py` | 推荐的 CPU 启动入口，在导入计算库前固定线程和隐藏 CUDA |
| `configs/training_cpu.yaml` | 日频 CPU 配置 |
| `configs/minute_training_cpu.yaml` | 分钟频 CPU 配置 |
| `src/gflownet/run_training.py` | 日频训练，支持显式 `device=cpu` / `--cpu` |
| `src/gflownet/run_minute_training.py` | 分钟训练，支持显式 `device=cpu` / `--cpu` |

## 3. 安装环境

建议使用 Python 3.10 或 3.11：

```bash
cd "/path/to/AlphaMining-GFlowNet-AlphaEval"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

检查 PyTorch CPU：

```bash
python -c "import os, torch; print('CPU逻辑核心:', os.cpu_count()); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

系统有 CUDA 也没有关系，推荐入口会设置 `CUDA_VISIBLE_DEVICES` 为空并向训练器显式传入 `cpu`。

## 4. 数据准备

### 4.1 日频训练

准备：

```text
data/daily_price.pkl
```

它由原日频数据准备流程生成。默认研究区间：

- 训练/挖掘：2020-01-01 至 2023-12-31；
- 样本外因子：2024-01-01 至 2026-12-31；
- 标签：`close(t+5) / close(t+1) - 1`。

### 4.2 分钟频训练

准备：

```text
data/minute_price.pkl
data/daily_price.pkl
```

分钟长表至少包含：

```text
date, datetime, code, open, high, low, close, vol, amount
```

分钟表达式在单日单股内部执行，经过 `r_*` 聚合后生成日频因子，再与 `daily_price.pkl` 中构造的未来收益标签对齐。

## 5. 推荐运行方式

### 5.1 日频 CPU 训练

```bash
source .venv/bin/activate

python scripts/train_cpu.py \
  --mode daily \
  --config configs/training_cpu.yaml
```

### 5.2 分钟频 CPU 训练

```bash
source .venv/bin/activate

python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute_training_cpu.yaml
```

临时覆盖 PyTorch 计算线程数：

```bash
python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute_training_cpu.yaml \
  --threads 12
```

临时覆盖 Alpha Pool 数量：

```bash
python scripts/train_cpu.py \
  --mode minute \
  --pool-size 20
```

也可以直接调用模块：

```bash
python -m src.gflownet.run_training \
  --cpu --config configs/training_cpu.yaml

python -m src.gflownet.run_minute_training \
  --cpu --config configs/minute_training_cpu.yaml
```

直接模块入口会强制训练设备为 CPU；但推荐使用 `scripts/train_cpu.py`，因为它能在导入底层计算库之前设置 BLAS 和 PyTorch 线程。

`scripts/train_cpu.py`还会把终端的标准输出、警告、错误和异常堆栈实时追加到
`outputs.log_dir`配置的目录。默认文件名包含运行模式、微秒级启动时间和进程号，
因此多次运行不会互相覆盖。可用`--log-dir`临时改目录，或用`--log-file`指定文件。

## 6. 如何确认没有使用 GPU

启动日志应包含：

```text
[CPUTraining] runtime_start ... cuda_visible=False
[GFlowNet] ... training_device=cpu
```

或：

```text
[MinuteGFlowNet] ... training_device=cpu
[GFlowNet] training_start device=cpu ... amp=False
```

CPU 配置明确设置：

```yaml
mixed_precision: false
```

CPU 不使用 CUDA AMP，也不会产生 GPU 显存日志。

## 7. CPU 参数说明

```yaml
cpu_runtime:
  torch_threads: 8
  interop_threads: 2
  blas_threads: 1
```

- `torch_threads`：Transformer 矩阵计算使用的 CPU 线程；
- `interop_threads`：PyTorch 算子之间的调度线程；
- `blas_threads`：NumPy/Pandas 底层 BLAS 单次运算线程数；
- `training.reward_workers`：并发计算不同表达式 Reward 的线程数。

不要同时把 `torch_threads`、`blas_threads` 和 `reward_workers` 都设置得很大。默认把 `blas_threads` 固定为 1，是为了降低 Reward 并发时的线程争抢。

建议起点：

| 机器 | torch_threads | reward_workers | blas_threads |
|---|---:|---:|---:|
| 8 核 | 4–6 | 2 | 1 |
| 12–16 核 | 8 | 4 | 1 |
| 24–32 核 | 12–16 | 4–8 | 1 |

实际最优值取决于内存带宽、分钟数据大小和表达式结构。比较参数时应观察每轮日志中的 `reward_seconds`，而不是只看 CPU 使用率。

## 8. 默认 CPU 模型为什么更小

GPU 分钟模型默认是：

```yaml
hidden_dim: 256
num_layers: 4
trajectories_per_epoch: 32
```

CPU 分钟模型默认调整为：

```yaml
hidden_dim: 128
num_layers: 2
trajectories_per_epoch: 8
max_depth: 5
max_nodes: 12
```

这主要减少 Transformer 前向采样和过深表达式的计算量。它不会改变图表 27–30 的特征和算子集合，但搜索容量低于 A100 正式模型。CPU 结果应单独作为一个实验版本比较，不应与 GPU checkpoint 混用。

## 9. 快速冒烟测试

第一次运行建议修改 CPU 配置：

```yaml
training:
  epochs: 2
  trajectories_per_epoch: 4
  reward_workers: 2

pipeline:
  pool_size: 5
  pool_attempts: 50
```

确认以下内容正常后，再提高训练规模：

1. 每条 trajectory 都输出表达式、Reward、RankIC、覆盖率和 TB Loss；
2. 每轮生成训练指标；
3. checkpoint 可以保存并重新加载；
4. Alpha Pool 能生成；
5. 因子矩阵日期和股票代码与日频数据一致。

## 10. 输出位置

日频 CPU：

```text
checkpoints/gflownet_daily_cpu_best.pt
results/daily_cpu/gflownet_training_metrics.csv
results/daily_cpu/gflownet_trajectory_metrics.csv
results/daily_cpu/alpha_pool.csv
results/daily_cpu/alpha_factor_matrix.pkl
results/daily_cpu/alpha_factor_matrix_oos.pkl
```

分钟频 CPU：

```text
checkpoints/gflownet_minute_cpu_best.pt
results/minute_cpu/gflownet_training_metrics.csv
results/minute_cpu/gflownet_trajectory_metrics.csv
results/minute_cpu/alpha_pool.csv
results/minute_cpu/alpha_factor_matrix.pkl
```

CPU 输出与 A100 输出完全分离，后续 AlphaEval 和 LightGBM 只需读取对应目录中的因子矩阵与 Alpha Pool。

使用远程 DolphinDB 分钟表时，参见 [DolphinDB 分钟数据 CPU 训练手册](DolphinDB分钟数据CPU训练手册.md)。

## 11. 常见问题

### 训练开始后 CPU 使用率很高但 Reward 很慢

通常是线程过度订阅或分钟表达式本身计算量大。先把：

```yaml
reward_workers: 2
blas_threads: 1
```

然后比较 `reward_seconds`。

### 内存不足

分钟长表和 21 个派生通道会占用较多内存。CPU 版目前适合按时间段或 Universe 分批验证；全市场多年分钟正式训练仍建议使用研报中的 MemMap/年度分块方案。

### CPU 训练是否会比 GPU 更快

Transformer 采样一般是 GPU 更快；Pandas Reward 计算主要依赖 CPU。小模型、小样本下 GPU 数据调度开销可能不明显，但全量分钟训练通常仍应使用 A100。CPU 版的核心价值是可运行、可调试和不依赖 GPU 资源。
