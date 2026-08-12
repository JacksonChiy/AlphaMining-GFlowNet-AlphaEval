# AlphaMining-GFlowNet-AlphaEval

## 项目概览

本项目复现研报《基于 GFlowNet 和 AlphaEval 的分钟频因子挖掘筛选框架》，目前同时维护两条研究流水线：

- 日频：`price.csv` → 数据预处理 → GFlowNet 因子挖掘 → AlphaEval 筛选 → LightGBM 融合 → 本地 RQAlphaPlus 回测；
- 分钟频：远程 DolphinDB → 按 241 根有效分钟构建 MemMap 或全量 RAM 数据 → 分钟表达式 GFlowNet → 日频 Reward → 因子池输出。

日频 GFlowNet 正式训练面向 Google Colab A100；分钟频支持 CPU、DolphinDB/MemMap 分离和阿里 PPU 大内存模式。行情、模型、日志和实验结果默认不提交 Git。本项目只用于研究与工程复现，不构成投资建议。

首次使用请从[日频运行手册](docs/guides/daily/runbook.md)或[分钟频文档](docs/guides/minute/expression_gflownet.md)开始。完整数据流见[工作流说明](docs/workflow.md)。

## 项目结构

```text
AlphaMining-GFlowNet-AlphaEval/
├── configs/                    # 按日频、分钟频、指数增强和基线分类的配置
├── data/                       # 私有数据及 raw/processed/external 放置说明
├── docs/
│   ├── guides/                 # 可执行运行手册
│   ├── reports/                # 历史分析、路线图与交接记录
│   ├── architecture.md         # 模块架构与依赖关系
│   ├── workflow.md             # 端到端数据流和产物
│   ├── experiments.md          # 实验命名、冻结和复现规则
│   └── troubleshooting.md      # 常见故障处理
├── experiments/                # 已冻结实验清单；大型产物默认忽略
├── notebooks/
│   ├── pipelines/              # 一体化正式流水线
│   ├── stages/                 # 分阶段调试入口
│   ├── deployment/             # PPU 等特定环境入口
│   └── archived/               # 只用于追溯的旧 Notebook
├── outputs/                    # 新代码推荐使用的统一输出根目录
├── rqalpha_strategy/           # RQAlphaPlus 策略与本地回测入口
├── scripts/                    # 命令行编排、审计与训练脚本
├── src/
│   ├── data_loader/            # 日频预处理、DDB、MemMap/RAM 数据层
│   ├── expression/             # 日频与分钟频表达式树、词法和编译
│   ├── operators/              # Pandas、NumPy、PyTorch 因子算子
│   ├── gflownet/               # Transformer 策略、TB 训练和 Reward
│   ├── alpha_eval/             # 因子评价、稳定性、鲁棒性和 DPP
│   ├── model/                  # LightGBM 滚动融合
│   ├── index_enhancement/      # 指数 Universe、标签和组合优化
│   └── utils/                  # 配置、实验和通用工具
└── tests/                      # 端到端关键边界与单元测试
```

目录设计依据与移动记录见 [RESTRUCTURE_PLAN.md](RESTRUCTURE_PLAN.md)，整理前审计见 [PROJECT_AUDIT.md](PROJECT_AUDIT.md)。

## 环境准备

推荐 Python 3.10–3.13。开发机从仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

Windows 激活命令为 `.venv\Scripts\activate`。也可按用途安装：

```bash
pip install -r requirements/base.txt       # 数据处理与评价
pip install -r requirements/training.txt   # PyTorch、LightGBM 和绘图
pip install -r requirements/ddb.txt        # 分钟频 DolphinDB 训练
pip install -r requirements/dev.txt        # 测试与 Jupyter
```

macOS 的 LightGBM 可能还需要 `brew install libomp`。RQAlphaPlus 为授权软件，不在公开依赖文件中，安装方法见 [RQAlphaPlus 环境说明](docs/guides/backtest/rqalpha_plus_setup.md)。

## 数据准备

日频原始数据放在 `data/price.csv`，至少包含日期、证券代码、开高低收和成交量；加载器会自动映射常见中英文字段名，并处理排序、缺失、异常、可选复权和截面标准化。推荐字段为：

```text
date, code, open, high, low, close, volume, amount, vwap,
adj_factor, industry, market_cap
```

预处理产物为 `data/daily_price.pkl` 和 `results/data_quality_report.json`。原始数据绝不应上传 Git；目录规则见 [data/README.md](data/README.md)。

分钟数据读取 DolphinDB 表字段：`sym, date, time, open, high, low, close, volume, amount, tradeCount`。有效日内网格固定为 9:25 一根、9:31–11:30 共 120 根、13:01–15:00 共 120 根。DDB 与训练机可分离：先落本地 MemMap 再训练；大内存 PPU 也可从 DDB 加载后保存 NPY 快照，下次直接从磁盘恢复到 RAM。

## 如何运行

### 日频 Colab A100

正式入口是 `notebooks/pipelines/daily_colab_a100.ipynb`。Notebook 包含仓库克隆、依赖安装、A100 检查、Google Drive 数据复制、训练、模型保存和加载测试。默认训练期为 2020–2023，2024–2026 用于样本外预测和本地回测；未来 5 日标签定义为：

```text
close(t+5) / close(t+1) - 1
```

Colab 只训练和导出模型产物，不运行 RQAlphaPlus。分阶段排错可依次运行 `notebooks/stages/` 下的 01–05；06 回测 Notebook 必须放在本地授权环境执行。

命令行日频流水线：

```bash
python -m scripts.run_daily_pipeline \
  --config configs/daily/training.yaml \
  --pool-size 100
```

### 日频或分钟频 CPU

```bash
python scripts/train_cpu.py \
  --mode daily \
  --config configs/daily/cpu.yaml

python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute/cpu.yaml
```

CPU 线程、日志和快速验证说明见 [CPU 训练手册](docs/guides/daily/cpu_training.md)。

### DolphinDB + MemMap 分钟训练

```bash
python scripts/prepare_ddb_minute.py \
  --config configs/minute/cpu_ddb_memmap.yaml \
  --audit-only

python scripts/audit_ddb_minute_quality.py \
  --config configs/minute/cpu_ddb_memmap.yaml \
  --scope grid

python scripts/build_ddb_memmap.py \
  --config configs/minute/cpu_ddb_memmap.yaml

python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute/cpu_ddb_memmap.yaml
```

首次构建才访问 DDB，训练期远程查询数为 0。详细部署见 [DDB/MemMap 手册](docs/guides/minute/ddb_memmap.md)。PPU 全量内存入口为 `notebooks/deployment/minute_ppu_ram.ipynb`，配置为 `configs/minute/ppu_ddb_ram.yaml`。

### 指数增强与本地回测

指数增强支持沪深 300、中证 500 和中证 1000。历史成分与权重一次性从 RQData 拉取后保存为本地文件，其余模块不重复请求接口：

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.universe
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.weights
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.labels
python -m src.index_enhancement.builder
/opt/miniconda3/envs/rqsdk/bin/python -m rqalpha_strategy.run_index_enhancement
```

RQAlphaPlus 不使用代理，回测读取 Colab 输出的预测分数。换手控制包含排名平滑、持仓缓冲、最短持有期与单次替换比例；费用基线统一为万 8。详见[指数增强手册](docs/guides/index_enhancement/runbook.md)与[组合优化手册](docs/guides/index_enhancement/fee_0008_portfolio.md)。

## 训练、筛选与评价

GFlowNet 用 Transformer Encoder 输出满足表达式文法的动作概率，并以 Trajectory Balance 训练：

```text
(logZ + sum(logPF) - logReward - sum(logPB))²
```

Reward 综合绝对 RankIC、多头组合 IR、行业/市值风险惩罚和覆盖率惩罚。低覆盖表达式先连续降权，再由因子池覆盖率硬门槛拒绝，避免后续回测因稀疏信号失败。

AlphaEval 从预测能力、滚动稳定性、扰动鲁棒性、表达式逻辑和 DPP 多样性筛选因子。LightGBM 在带 5 日 purge 的滚动窗口中融合因子，输出每日 `prediction_score` 和截面排名。算法边界与防未来数据泄漏检查见[方法说明](docs/methodology.md)。

## 输出

当前历史代码仍使用 `results/`、`checkpoints/` 和 `experiments/`，这些路径为兼容性暂不强制迁移；新增工具应优先写入 `outputs/`。典型产物包括：

```text
checkpoints/gflownet_best.pt
results/gflownet_training_metrics.csv
results/alpha_pool.csv
results/alpha_factor_matrix.pkl
results/alpha_factor_matrix_oos.pkl
results/alpha_eval_result.csv
results/lightgbm/prediction_score.csv
results/backtest_report/
experiments/<experiment_id>/config.yaml
```

真实权重、私有行情和回测结果不随仓库发布。运行前后每个文件的生产者与消费者见[工作流说明](docs/workflow.md)。

## 配置

- `configs/daily/training.yaml`：日频正式配置；
- `configs/daily/quick.yaml`：小规模冒烟测试；
- `configs/daily/cpu.yaml`：日频 CPU；
- `configs/minute/training.yaml`：分钟频基础训练；
- `configs/minute/cpu_ddb_memmap.yaml`：DDB/MemMap CPU；
- `configs/minute/ppu_ddb_ram.yaml`：PPU 全量 RAM 与磁盘快照；
- `configs/index_enhancement/default.yaml`：三指数数据、模型、组合和回测；
- `configs/baselines/fee_0008.yaml`：万 8 冻结基线。

正式实验不要直接覆盖基线配置；应复制配置并在 `experiments/<experiment_id>/` 保存快照。规范见[实验管理](docs/experiments.md)。

## 开发说明

- 算法实现必须位于 `src/`，Notebook 只做环境准备、编排和展示；
- 所有路径从仓库根目录或配置解析，不写用户绝对路径；
- 不提交密钥、DDB 口令、RQData 凭据、行情、模型和大体积输出；
- 每个独立改动使用一个语义化 Git commit；
- 提交前运行 `pytest -q`，并确认 Notebook 可被 JSON/IPython 正常解析；
- 不确定的历史文件先记录或归档，不直接删除。

更多信息： [架构](docs/architecture.md) · [工作流](docs/workflow.md) · [实验](docs/experiments.md) · [排错](docs/troubleshooting.md)

```mermaid
flowchart LR
    A["行情数据"] --> B["预处理与时点校验"]
    B --> C["GFlowNet 表达式挖掘"]
    C --> D["AlphaEval + DPP 筛选"]
    D --> E["LightGBM 滚动融合"]
    E --> F["预测分数"]
    F --> G["本地 RQAlphaPlus 回测"]
```
