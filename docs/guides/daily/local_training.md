# 日频本地完整训练手册

## 1. 能做什么

日频本地入口在一台普通 CPU 机器上完成：

```text
price.csv
→ 数据预处理
→ GFlowNet 日频因子挖掘
→ 因子池与 2020–2026 因子矩阵
→ AlphaEval + DPP
→ LightGBM 滚动训练
→ prediction_score.csv
→ 可选本地 RQAlphaPlus 回测
```

与 `scripts/train_cpu.py --mode daily` 的区别是：后者只运行 GFlowNet 和因子池；`scripts/train_daily_local.py` 是完整、可按阶段续跑的本地流水线。

## 2. 环境和数据

推荐 Python 3.10–3.13，并在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

把本地日频行情保存为 `data/price.csv`。代码不调用外部行情接口。

## 3. 首次快速验证

先复制一份配置，避免修改正式配置：

```bash
cp configs/daily/local.yaml configs/daily/local_smoke.yaml
```

将测试配置调小：

```yaml
dataset:
  max_stocks: 300
  universe_start_date: '2020-07-01'
  universe_end_date: '2020-12-31'

training:
  epochs: 2
  trajectories_per_epoch: 4
  reward_workers: 2

alpha_eval:
  dpp_k: 5

lightgbm:
  min_train_days: 252
  train_window_days: 504
  n_estimators: 50

pipeline:
  pool_size: 5
  pool_attempts: 50
```

然后运行到 LightGBM：

```bash
python scripts/train_daily_local.py \
  --config configs/daily/local_smoke.yaml \
  --to-stage lightgbm
```

## 4. 正式完整运行

```bash
python scripts/train_daily_local.py \
  --config configs/daily/local.yaml \
  --from-stage prepare \
  --to-stage lightgbm
```

如果 `data/daily_price.pkl` 已由相同数据和配置生成，可跳过 CSV 预处理：

```bash
python scripts/train_daily_local.py \
  --config configs/daily/local.yaml \
  --from-stage prepare \
  --to-stage lightgbm \
  --reuse-prepared-data
```

也可以打开 `notebooks/pipelines/daily_local_cpu.ipynb`。Notebook 最终仍调用同一个独立进程入口，保证线程设置与命令行一致。

## 5. 中断后续跑

每个阶段开始和结束都会原子更新：

```text
results/daily_local/pipeline_manifest.json
```

如果 GFlowNet 和因子矩阵已经完成，从 AlphaEval 继续：

```bash
python scripts/train_daily_local.py \
  --config configs/daily/local.yaml \
  --from-stage alpha_eval \
  --to-stage lightgbm
```

如果只想重新训练 LightGBM：

```bash
python scripts/train_daily_local.py \
  --config configs/daily/local.yaml \
  --from-stage lightgbm \
  --to-stage lightgbm
```

续跑前会检查上游文件；缺失时会明确列出所需产物，不会静默从错误目录读取。

## 6. 复用已挖掘因子

已有相同训练区间和 Grammar 生成的 `results/daily_local/alpha_pool.csv` 时，可以不重新训练 GFlowNet，而是在当前完整行情上恢复表达式并重算因子：

```bash
python scripts/train_daily_local.py \
  --config configs/daily/local.yaml \
  --from-stage gflownet \
  --to-stage lightgbm \
  --reuse-alpha-pool
```

因子必须先在 2020–2026 完整序列上执行，再截取 2024–2026 样本外部分，这样滚动算子在 2024 年初仍有历史窗口。

## 7. 可选本地回测

只有本机已安装授权 RQAlphaPlus 时才运行：

```bash
python scripts/train_daily_local.py \
  --config configs/daily/local.yaml \
  --from-stage backtest \
  --to-stage backtest \
  --rqalpha-bundle ~/.rqalpha-plus/bundle
```

也可在首次运行时将 `--to-stage` 设置为 `backtest` 并传 bundle。回测自动读取配置中的排名平滑、持仓缓冲、最短持有、万 8 手续费和滑点；RQAlphaPlus 运行前会清除代理环境变量。

## 8. 输出文件

```text
data/daily_price.pkl
checkpoints/gflownet_daily_local_best.pt
results/daily_local/
├── logs/
├── data_quality_report.json
├── pipeline_manifest.json
├── gflownet_training_metrics.csv
├── gflownet_trajectory_metrics.csv
├── alpha_pool.csv
├── alpha_factor_matrix.pkl
├── alpha_factor_matrix_oos.pkl
├── alpha_eval_result.csv
├── lightgbm/
│   ├── prediction_score.csv
│   ├── model_metrics.csv
│   ├── feature_importance.csv
│   └── lgbm_model.joblib
└── backtest_report/
```

日志包含每个阶段、每条轨迹、Reward、AlphaEval 因子进度和 LightGBM 滚动窗口进度。

## 9. 性能建议

- 先用 300 只股票、2 个 epoch 做完整冒烟测试；
- `blas_threads` 保持 1，避免与 `reward_workers` 过度争抢；
- 12–16 核机器从 `torch_threads=8`、`reward_workers=4` 开始；
- 日频滚动算子在本地 CPU 默认使用 Pandas；
- `--reuse-alpha-pool` 适合固定表达式后反复扩展行情、调 AlphaEval 或 LightGBM；
- 本地正式全市场搜索会明显慢于 Colab A100，适合调试、复现和中小规模实验。
