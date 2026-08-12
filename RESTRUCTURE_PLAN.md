# AlphaMining 项目整理计划

制定日期：2026-08-12  
依据：`PROJECT_AUDIT.md`  
约束：不改变模型、因子、标签、组合优化和回测算法；不删除原始数据、实验结果、模型或未确认资产。

## 1. 目标结构

项目不强制迁移为新的包名，以避免破坏大量`src.*`导入。目标是在现有可运行结构上增加清晰分类：

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
│   ├── dev.txt
│   └── ddb.txt
├── requirements.txt                 # 兼容入口
├── requirements-ddb.txt             # 兼容入口
├── configs/
│   ├── daily/
│   ├── minute/
│   ├── index_enhancement/
│   └── baselines/
├── data/
│   ├── README.md
│   ├── raw/.gitkeep
│   ├── processed/.gitkeep
│   └── external/.gitkeep
├── docs/
│   ├── architecture.md
│   ├── workflow.md
│   ├── experiments.md
│   ├── troubleshooting.md
│   ├── guides/
│   │   ├── daily/
│   │   ├── minute/
│   │   ├── backtest/
│   │   └── index_enhancement/
│   └── reports/
├── notebooks/
│   ├── pipelines/
│   ├── stages/
│   ├── deployment/
│   └── archived/
├── outputs/
│   └── README.md                    # 只记录约定，实际产物忽略
├── scripts/
├── src/
│   ├── alpha_eval/
│   ├── data_loader/
│   │   ├── minute_dense.py
│   │   ├── minute_memmap.py
│   │   └── minute_ram_cache.py
│   ├── expression/
│   ├── gflownet/
│   ├── index_enhancement/
│   ├── model/
│   ├── operators/
│   └── utils/
├── rqalpha_strategy/
├── tests/
└── archive/
    └── README.md
```

## 2. 执行阶段与独立提交

### 阶段A：冻结现状审计

状态：已完成。

- 创建`PROJECT_AUDIT.md`；
- 不修改其他文件；
- 提交：`docs: audit current project structure`。

### 阶段B：计划与风险边界

本阶段仅创建本文档。

- 列出目标结构；
- 明确移动映射、兼容策略和不处理资产；
- 提交：`docs: plan project restructuring`。

### 阶段C：整理项目结构

计划提交：`chore: reorganize project structure`。

执行内容：

1. 创建`configs/daily`、`configs/minute`、`configs/index_enhancement`和`configs/baselines`；
2. 使用`git mv`移动配置，并全仓更新路径引用；
3. 创建`notebooks/pipelines`、`notebooks/stages`、`notebooks/deployment`和`notebooks/archived`；
4. 移动Notebook并更新测试、README、文档和Notebook内部路径；
5. 将根目录遗留`06_rqalpha_backtest.ipynb`归档为带明确说明的旧版本；
6. 创建`data/raw`、`data/processed`、`data/external`占位说明，不移动或删除现有私有数据；
7. 创建`outputs/README.md`和`archive/README.md`，定义生成物与历史代码策略。

配置移动映射：

| 当前路径 | 目标路径 |
|---|---|
| `configs/training_config.yaml` | `configs/daily/training.yaml` |
| `configs/quick_training_config.yaml` | `configs/daily/quick.yaml` |
| `configs/training_cpu.yaml` | `configs/daily/cpu.yaml` |
| `configs/minute_training_config.yaml` | `configs/minute/training.yaml` |
| `configs/minute_training_cpu.yaml` | `configs/minute/cpu.yaml` |
| `configs/minute_training_cpu_ddb.yaml` | `configs/minute/cpu_ddb_memmap.yaml` |
| `configs/minute_training_ppu_ddb_ram.yaml` | `configs/minute/ppu_ddb_ram.yaml` |
| `configs/index_enhancement.yaml` | `configs/index_enhancement/default.yaml` |
| `configs/index_model_experiments.yaml` | `configs/index_enhancement/model_experiments.yaml` |
| `configs/fee_0008_baseline.yaml` | `configs/baselines/fee_0008.yaml` |

Notebook移动映射：

| 当前路径 | 目标路径 |
|---|---|
| `notebooks/00_colab_full_pipeline_A100.ipynb` | `notebooks/pipelines/daily_colab_a100.ipynb` |
| `notebooks/01_data_prepare.ipynb` | `notebooks/stages/01_data_prepare.ipynb` |
| `notebooks/02_expression_engine.ipynb` | `notebooks/stages/02_expression_engine.ipynb` |
| `notebooks/03_train_gflownet_A100.ipynb` | `notebooks/stages/03_train_gflownet_a100.ipynb` |
| `notebooks/04_alpha_eval.ipynb` | `notebooks/stages/04_alpha_eval.ipynb` |
| `notebooks/05_lgbm_model.ipynb` | `notebooks/stages/05_lightgbm.ipynb` |
| `notebooks/06_rqalpha_backtest.ipynb` | `notebooks/stages/06_rqalpha_backtest.ipynb` |
| `notebooks/07_train_minute_ppu_jupyterlab.ipynb` | `notebooks/deployment/minute_ppu_memmap.ipynb` |
| `notebooks/08_train_minute_ppu_ddb_ram.ipynb` | `notebooks/deployment/minute_ppu_ram.ipynb` |
| 根目录`06_rqalpha_backtest.ipynb` | `notebooks/archived/rqalpha_backtest_legacy.ipynb` |

兼容策略：移动后必须全仓更新路径引用；不保留重复Notebook副本。配置CLI默认值同步更新，并由测试验证。

### 阶段D：拆分共享分钟数据工具

计划提交：`refactor: extract minute storage utilities`。

执行内容：

1. 从`minute_memmap.py`提取纯NumPy稠密分钟构建到`minute_dense.py`；
2. 从`minute_memmap.py`提取RAM快照格式、校验和原子读写到`minute_ram_cache.py`；
3. `minute_memmap.py`保留MemMap/RAM Store编排和原有公共导出；
4. 更新测试导入，确保数值结果、缓存指纹和日志不变；
5. 不修改分钟特征公式和缺失分钟语义。

本轮暂不拆`memmap_reward.py`和`dolphindb_minute.py`，避免把一个整理任务扩大为算法级重构；将其记录为后续事项。

### 阶段E：依赖与Git规范

计划提交：`chore: standardize dependencies and git hygiene`。

执行内容：

1. 创建最小运行、训练、开发和DDB依赖分层；
2. 保留根目录requirements兼容入口；
3. 创建`pyproject.toml`，声明Python版本、包发现和pytest配置；
4. 扩展`.gitignore`：RAM快照、离线包、日志、临时演示检查文件、PowerPoint锁文件；
5. 不从Git历史中删除任何现有文件，不自动安装或升级依赖。

### 阶段F：重建项目文档

计划提交：`docs: rebuild project documentation`。

执行内容：

1. 重写README，使其覆盖日频、分钟频、PPU、指数增强和本地RQAlphaPlus；
2. 创建`docs/architecture.md`；
3. 创建`docs/workflow.md`；
4. 创建`docs/experiments.md`；
5. 创建`docs/troubleshooting.md`；
6. 把现有运行手册按主题移动到`docs/guides/`，阶段报告移动到`docs/reports/`；
7. 更新全部相对链接和运行命令。

`docs/项目交接记忆.md`含个人绝对路径，将归档到`docs/reports/legacy_handover.md`并加“历史环境信息”说明，不作为当前运行依据。

### 阶段G：最终验证与报告

计划提交：`docs: add refactor completion report`。

执行内容：

1. 运行完整pytest；
2. 对所有Python文件执行编译检查；
3. 校验所有Notebook JSON与代码单元语法；
4. 扫描失效的旧配置/Notebook/文档路径；
5. 创建`REFACTOR_REPORT.md`，记录Before、After、Changes、Important Files、How to Run、Remaining Issues和Recommended Next Steps。

## 3. 不执行的删除

以下内容不会在本轮自动删除：

- `data/`中的任何数据；
- `results*`、`experiments/*/`和`checkpoints/`；
- 根目录压缩包；
- 未跟踪PPTX、HTML和报告；
- 用户修改的README流程图；
- 授权软件、bundle和本地环境。

对于`.DS_Store`、`__pycache__`、`.pytest_cache`、`~$*.pptx`和`*.inspect.ndjson`，只完善忽略规则；是否删除本地副本留给用户决定。

## 4. 行为保持与验收标准

每个执行阶段都必须满足：

1. `git diff --check`通过；
2. 路径移动后旧引用扫描结果为零，或被明确标记为历史文本；
3. `PYTHONPATH=. pytest -q`保持通过；
4. 不改变Reward、标签、因子、LightGBM、组合优化或回测参数默认值；
5. 不提交数据、模型、结果、压缩包、密钥或服务器凭据；
6. 每个阶段形成独立Git commit。

## 5. 预期风险

- Notebook内部可能包含旧路径字符串，移动后必须修改并重新做AST检查；
- 配置路径被README、文档、Python默认参数和测试多处引用，需要机械替换后全仓扫描；
- 报告文档中的旧命令可能是历史记录，不应静默改写实验事实；移动时应保留历史语境；
- RAM缓存快照可能达到数十GB，必须继续被Git忽略；
- `src`包未标准化命名是已知技术债，本轮保持兼容优先。
