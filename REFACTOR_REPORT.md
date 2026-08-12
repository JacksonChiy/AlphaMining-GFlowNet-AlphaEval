# AlphaMining 项目整理完成报告

完成日期：2026-08-12

工作分支：`feature/ppu-training`

整理原则：保持模型、Reward、标签、因子、组合和回测算法不变；不删除私有数据与历史实验资产。

## Before

整理前项目已经具备完整研究能力，但可维护性问题主要集中在：

```text
repository/
├── configs/*.yaml                     # 日频、分钟频、指数配置全部平铺
├── notebooks/*.ipynb                  # 正式、分阶段、部署方案混在一起
├── 06_rqalpha_backtest.ipynb          # 根目录重复/遗留版本
├── docs/*.md                          # 运行手册、报告、交接材料没有边界
├── src/data_loader/minute_memmap.py   # 约 1562 行，多种职责耦合
├── requirements.txt                   # 运行、训练、测试、Notebook 依赖混合
├── requirements-ddb.txt
├── results*/、checkpoints/、*.zip     # 本地大型产物分散但已被忽略
└── 无 pyproject.toml、架构/工作流/实验/排错文档
```

README 仍将项目描述为“日频精简复现”，没有准确覆盖分钟 DolphinDB、MemMap/RAM、PPU 和三指数增强。配置与 Notebook 文件名也无法直接表达场景和用途。

详细整理前快照见 [PROJECT_AUDIT.md](PROJECT_AUDIT.md)。

## After

```text
AlphaMining-GFlowNet-AlphaEval/
├── README.md
├── PROJECT_AUDIT.md
├── RESTRUCTURE_PLAN.md
├── REFACTOR_REPORT.md
├── pyproject.toml
├── requirements/
│   ├── base.txt
│   ├── training.txt
│   ├── ddb.txt
│   └── dev.txt
├── configs/
│   ├── daily/
│   ├── minute/
│   ├── index_enhancement/
│   └── baselines/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── docs/
│   ├── guides/{daily,minute,index_enhancement,backtest}/
│   ├── reports/
│   ├── architecture.md
│   ├── workflow.md
│   ├── experiments.md
│   └── troubleshooting.md
├── notebooks/
│   ├── pipelines/
│   ├── stages/
│   ├── deployment/
│   └── archived/
├── outputs/
├── archive/
├── scripts/
├── src/
├── rqalpha_strategy/
└── tests/
```

## Changes

### Moved and renamed

- 10 个配置按日频、分钟频、指数增强和冻结基线归类；
- 9 个当前 Notebook 按完整流水线、分阶段和部署场景归类并统一命名；
- 根目录旧回测 Notebook 移入 `notebooks/archived/`，当前入口保持唯一；
- 12 份已跟踪文档分为 `docs/guides/` 和 `docs/reports/`；
- 所有当前源码、测试、Notebook 和维护文档中的旧路径已更新。

### Refactored

- 从 `minute_memmap.py` 抽出 `minute_dense.py`，集中管理 19 个分钟通道和单日稠密 NumPy 构建；
- 抽出 `minute_ram_cache.py`，集中管理 RAM 快照指纹、manifest 校验和 NPY/JSON 原子写入；
- 保留 `_build_dense_minute_channels` 兼容别名，没有改变已有算子数值语义；
- 分钟主文件减少约 300 行，MemMap/RAM Store 的公共接口保持不变。

### Created

- `pyproject.toml`：Python 3.10–3.13、包发现、可选依赖和 Pytest 配置；
- 四层依赖文件与根目录兼容入口；
- 配置、Notebook、数据、输出和归档目录说明；
- 中文 README、架构、工作流、实验管理、排错和文档索引；
- PPU 全量 RAM 数据的本地 NPY 快照：配置指纹命中时无需再次查询 DDB。

### Git hygiene

- 新增忽略 `outputs/` 产物、RAM 快照、离线包、日志、检查器中间文件和 PowerPoint 锁文件；
- 保持 `README.md` 等目录说明可跟踪；
- 未发现源码或已跟踪配置中的明文 token/password；连接凭据继续通过环境变量提供。

### Deleted

没有删除原始数据、模型、结果、实验、压缩包或未确认文档。本轮也没有清理本地缓存文件，以避免产生不可逆影响。

## Independent Git commits

每个阶段均形成独立提交：

```text
334d4c5 feat: add eager RAM disk cache for PPU training
ff52a4f docs: audit current project structure
e6a1558 docs: plan project restructuring
7a8e5d1 chore: reorganize project structure
fd76b74 refactor: extract minute storage utilities
8a1bb74 chore: standardize dependencies and git hygiene
8f081b8 docs: rebuild project documentation
(本次提交) docs: add refactor completion report
```

## Important Files

以后继续开发时，建议优先查看：

| 文件 | 用途 |
|---|---|
| `README.md` | 从零安装、数据、训练、回测和配置入口 |
| `docs/workflow.md` | 每一步输入、处理和输出的数据走向 |
| `docs/architecture.md` | 模块职责和依赖边界 |
| `configs/daily/training.yaml` | 日频正式训练与回测参数真源 |
| `configs/minute/cpu_ddb_memmap.yaml` | DDB/MemMap 分离训练配置 |
| `configs/minute/ppu_ddb_ram.yaml` | PPU 全量 RAM 与磁盘快照配置 |
| `notebooks/pipelines/daily_colab_a100.ipynb` | Colab A100 日频一体化入口 |
| `notebooks/pipelines/daily_local_cpu.ipynb` | 日频本地 CPU 一体化入口 |
| `notebooks/deployment/minute_ppu_ram.ipynb` | PPU 分钟频训练入口 |
| `scripts/train_cpu.py` | CPU/PPU 统一训练启动器与日志入口 |
| `src/gflownet/trainer.py` | GFlowNet 采样与 TB loss |
| `src/gflownet/reward.py`、`memmap_reward.py` | 日频和分钟 MemMap Reward |
| `src/data_loader/minute_dense.py` | DDB 单日数据到 NumPy 通道 |
| `src/data_loader/minute_ram_cache.py` | PPU RAM 快照底层协议 |
| `rqalpha_strategy/strategy.py` | RQAlphaPlus 实际策略逻辑 |

## How to Run

### 安装和测试

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

### 日频完整训练

在 Colab A100 打开并运行：

```text
notebooks/pipelines/daily_colab_a100.ipynb
```

或在仓库根目录运行命令行编排：

```bash
python -m scripts.run_daily_pipeline \
  --config configs/daily/training.yaml \
  --pool-size 100
```

### DDB/MemMap 分钟训练

```bash
python scripts/prepare_ddb_minute.py \
  --config configs/minute/cpu_ddb_memmap.yaml \
  --audit-only

python scripts/build_ddb_memmap.py \
  --config configs/minute/cpu_ddb_memmap.yaml

python scripts/train_cpu.py \
  --mode minute \
  --config configs/minute/cpu_ddb_memmap.yaml
```

### PPU 全量 RAM 训练

在 PPU JupyterLab 中打开：

```text
notebooks/deployment/minute_ppu_ram.ipynb
```

首次运行从 DDB 加载并保存快照；以后配置与数据指纹一致时直接从硬盘恢复至 RAM。该模式不使用 MemMap。

### 本地指数增强回测

```bash
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.universe
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.weights
/opt/miniconda3/envs/rqsdk/bin/python -m src.index_enhancement.labels
python -m src.index_enhancement.builder
/opt/miniconda3/envs/rqsdk/bin/python -m rqalpha_strategy.run_index_enhancement
```

RQAlphaPlus 在本地授权环境执行，并明确不使用代理。

## Validation

整理完成后的验收结果：

- `pytest -q`：121 passed，1 skipped；
- 全部已跟踪 Python 文件编译通过；
- 10 个 Notebook 均可解析为合法 JSON，代码单元可由 IPython 转换并进行语法检查；
- 当前 Markdown 相对链接检查通过；
- 当前源码、配置、测试和维护文档中的旧配置/Notebook/文档路径引用为 0；
- `git diff --check` 通过；
- 未改变任何核心算法默认参数和数值实现。

## Remaining Issues

1. `src/gflownet/memmap_reward.py` 和 `src/data_loader/dolphindb_minute.py` 仍然较长，后续应在专项测试保护下拆分；
2. `rqalpha_strategy/run_backtest.py` 的配置装配、环境变量和报告提取仍可分层；
3. 历史代码继续写入 `results/`、`checkpoints/` 和 `experiments/`，尚未强制迁移到 `outputs/`；
4. 顶级包仍名为 `src`，这是兼容性优先下保留的已知技术债；
5. 尚未建立 CI、静态类型检查和格式检查；
6. 根目录本地仍有数 GB 压缩包和多套结果目录，已忽略但会增加索引/备份成本；
7. 未跟踪的 PPTX、HTML、讲稿、PPU 手册和报告脚本用途未确认，因此保持原地且未提交；
8. DolphinDB、RQData、RQAlphaPlus 和真实 A100/PPU 的外部集成需要在对应授权环境做最终运行验收。

## Recommended Next Steps

1. 在 PPU 上用小日期区间验证 RAM 快照首次写入、二次命中和断点恢复，再扩展到全量；
2. 建立 GitHub/Gitee CI，至少执行 Python 3.10/3.12 的 Pytest、Notebook 解析和敏感信息扫描；
3. 将 `memmap_reward.py` 拆成缓存、任务计划和 NumPy 执行器三个模块；
4. 为实验 manifest 增加 Git commit、输入文件哈希和 RQAlphaPlus 生效配置的自动记录；
5. 逐步让新实验写入 `outputs/<experiment_id>/`，保留旧路径读取兼容层；
6. 人工确认未跟踪演示材料后，再决定纳入 `docs/reports/`、移出仓库或删除临时副本。
