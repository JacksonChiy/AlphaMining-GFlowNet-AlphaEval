# AlphaMining 项目审计报告

审计日期：2026-08-12  
审计分支：`feature/ppu-training`  
审计原则：本阶段只读取和记录，不移动、不删除、不重写算法。

## 1. 当前目录结构

以下结构同时区分了 Git 已跟踪内容与仅存在于本地工作区的研究资产。省略了 `.git/`、`.venv/`、`__pycache__/` 和目录内部的大量生成文件。

```text
AlphaMining-GFlowNet-AlphaEval/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-ddb.txt
├── .gitignore
├── 06_rqalpha_backtest.ipynb          # 根目录遗留副本，与 notebooks/ 下版本不同
├── configs/                           # 日频、分钟频、指数增强和费用基线配置
├── data/
│   ├── README.md                      # Git 跟踪
│   └── ...                            # 本地私有行情和指数数据，约 1.4 GB
├── docs/                              # 方法、运行手册、阶段报告及本地演示材料
├── experiments/
│   ├── README.md                      # Git 跟踪
│   └── ...                            # 本地实验目录，默认忽略
├── notebooks/
│   ├── 00_colab_full_pipeline_A100.ipynb
│   ├── 01_data_prepare.ipynb
│   ├── 02_expression_engine.ipynb
│   ├── 03_train_gflownet_A100.ipynb
│   ├── 04_alpha_eval.ipynb
│   ├── 05_lgbm_model.ipynb
│   ├── 06_rqalpha_backtest.ipynb
│   ├── 07_train_minute_ppu_jupyterlab.ipynb
│   └── 08_train_minute_ppu_ddb_ram.ipynb
├── rqalpha_strategy/                  # RQAlphaPlus 策略、配置装配和本地回测入口
├── scripts/                           # 面向用户的编排、审计、构建和训练入口
├── src/
│   ├── alpha_eval/                    # AlphaEval 评价与指数内评价
│   ├── data_loader/                   # 日频 CSV、DolphinDB、质量审计、MemMap/RAM
│   ├── expression/                    # 日频/分钟表达式树、序列化和执行缓存
│   ├── gflownet/                      # 文法、策略网络、训练器、奖励和因子池
│   ├── index_enhancement/             # Universe、权重、标签、诊断、组合优化和归因
│   ├── model/                         # LightGBM 融合
│   ├── operators/                     # 日频、分钟频和 PyTorch 时序算子
│   ├── utils/                         # 日期区间和实验目录工具
│   └── runtime_logging.py             # 训练终端日志持久化
├── tests/                             # 19 个测试模块
├── checkpoints/                       # 本地模型，约 37 MB，Git 忽略
├── results*/                          # 多套本地实验结果，合计约 1.3 GB，Git 忽略
├── *.zip                              # Colab产物和报告压缩包，合计约 5.8 GB，Git 忽略
└── offline_packages/                  # 本地离线依赖目录
```

### Git 跟踪范围

Git 当前主要跟踪：53 个 `src` 文件、19 个测试、10 个配置、9 个脚本、9 个 Notebook、13 份文档/报告，以及项目元数据。大型数据、模型、结果和压缩包均未被 Git 跟踪。

## 2. 核心模块说明

### 2.1 正式入口

| 入口 | 用途 | 主要输出 |
|---|---|---|
| `notebooks/00_colab_full_pipeline_A100.ipynb` | 日频 Colab A100 一体化训练 | checkpoint、因子池、AlphaEval、指数模型和预测文件 |
| `scripts/run_daily_pipeline.py` | 日频命令行编排 | 与日频训练阶段相同，不负责授权回测 |
| `scripts/train_cpu.py` | CPU/PPU 日频或分钟训练启动器 | 日志、checkpoint、训练指标、因子池 |
| `src/gflownet/run_minute_training.py` | 分钟训练实际入口 | 分钟 GFlowNet 结果和因子矩阵 |
| `scripts/build_ddb_memmap.py` | DDB 与 CPU 训练分离时构建本地 MemMap | 年度通道数组、mask、manifest |
| `rqalpha_strategy/run_backtest.py` | 本地 RQAlphaPlus 单策略回测 | 回测配置、指标、净值、持仓和交易明细 |
| `rqalpha_strategy/run_index_enhancement.py` | 沪深300/中证500/中证1000批量回测 | 三指数独立回测目录 |

### 2.2 数据读取流程

日频流程：

```text
data/price.csv
→ src/data_loader/preprocess.py
→ 字段映射、清洗、复权、截面处理
→ data/daily_price.pkl + results/data_quality_report.json
```

分钟流程存在三种明确模式：

```text
DolphinDB
├── cache：分块保存分钟文件
├── memmap：构建年度磁盘数组，训练与DDB分离
└── ram：DDB直读普通NumPy数组；可保存磁盘快照，下次完整载入RAM
```

`src/data_loader/dolphindb_minute.py`负责连接、字段审计、交易日、SQL和日频聚合；`minute_quality_audit.py`负责完整质量审计；`minute_memmap.py`当前同时承担MemMap构建、MemMap读取、NumPy稠密特征构造、RAM加载和RAM快照职责。

### 2.3 模型训练流程

```text
规范化行情
→ Expression Grammar / Expression Tree
→ Transformer Policy采样表达式轨迹
→ 表达式执行与Reward评估
→ Trajectory Balance Loss反向传播
→ 最佳checkpoint
→ Alpha Pool及完整区间因子矩阵
```

日频和分钟频共享`src/gflownet/model.py`及基础训练器结构。分钟频额外使用分钟文法、NumPy三维执行器和日内Reduce结果缓存。训练器输出逐轨迹与逐epoch指标，并保存最优模型。

### 2.4 筛选、融合和回测流程

```text
Alpha Pool
→ AlphaEval：预测力、稳定性、扰动、复杂度、多样性/DPP
→ 选定因子矩阵
→ LightGBM walk-forward融合
→ prediction_score.csv
→ RQAlphaPlus本地回测
```

指数增强在此基础上加入历史成分、指数权重、`t+5/t+1`超额标签、涨跌停可交易约束、组合优化、诊断和风险归因。

### 2.5 配置体系

当前配置覆盖四组场景：

- 日频正式/快速/CPU：`training_config.yaml`、`quick_training_config.yaml`、`training_cpu.yaml`；
- 分钟本地/CPU/DDB/PPU：`minute_training_config.yaml`、`minute_training_cpu.yaml`、`minute_training_cpu_ddb.yaml`、`minute_training_ppu_ddb_ram.yaml`；
- 指数增强：`index_enhancement.yaml`、`index_model_experiments.yaml`；
- 冻结基线：`fee_0008_baseline.yaml`。

配置已覆盖大部分路径和超参数，但文件仍平铺在一个目录，命名没有清晰表达“频率/运行平台/实验用途”的层级。

## 3. 当前存在的问题

### 3.1 结构与命名

1. 根目录存在`06_rqalpha_backtest.ipynb`，与`notebooks/06_rqalpha_backtest.ipynb`内容不同，无法仅凭名称判断权威版本。
2. Notebook同时混合日频正式流程、分阶段调试、旧PPU MemMap模式和新PPU RAM模式，缺少用途分组。
3. 配置文件平铺，`training_config`、`training_cpu`、`quick_training_config`等名称无法直观体现日频；分钟配置则包含频率前缀，命名不统一。
4. 文档同时包含长期维护文档、阶段性报告、交接记忆、运行手册和演示材料，没有稳定文档与历史材料的边界。
5. `src`作为顶级包能够运行，但名称过于泛化；此时整体改包名会制造大量无价值diff，因此本轮不建议强制迁移为`src/alphamining/`。

### 3.2 核心代码职责过重

1. `src/data_loader/minute_memmap.py`约1562行，同时包含配置、分钟时间解析、NumPy特征、MemMap构建/读取、RAM Store和RAM磁盘快照，已超过单一模块职责。
2. `src/gflownet/memmap_reward.py`约1084行，同时包含缓存格式、任务编排、进程/线程并行和Reward执行。
3. `src/data_loader/dolphindb_minute.py`约926行，同时包含配置、审计、SQL、缓存抽取和日频聚合。
4. `rqalpha_strategy/run_backtest.py`约511行，配置解析、环境变量装配、RQAlpha运行和报告提取耦合。
5. 多个模块可直接以`python -m src...`运行，同时`scripts/`又提供部分相同层级的入口，用户入口与内部模块入口界限不完全统一。

### 3.3 路径和配置

1. 核心Python代码基本没有用户绝对路径，这是优点；硬编码主要位于Colab、PPU Notebook及部署文档，属于平台路径，但应集中说明。
2. `docs/项目交接记忆.md`包含个人本地绝对路径，不适合作为长期公共文档。
3. Notebook自行修改配置、环境变量和路径；配置覆盖关系应在文档中明确。
4. README仍以“日频精简复现”为开头，但项目已包含完整分钟DDB、MemMap、RAM/PPU和三指数增强，描述已落后于实际能力。
5. README的分钟章节重点仍是`cpu-training` MemMap分支，尚未完整纳入PPU RAM快照路径。

### 3.4 数据、结果和临时资产

1. 本地`data/`约1.4GB，结果目录约1.3GB，压缩包约5.8GB；虽然被`.gitignore`覆盖，但全部位于仓库根附近，易造成误操作、备份缓慢和编辑器索引压力。
2. 根目录有多份`alphamining_colab_outputs*.zip`和`Gflownet报告.zip`，命名包含`hubor`拼写错误、日期版本和无日期版本，属于应外移或归档的本地产物。
3. 存在多个`results_<experiment>`平行目录，实验元数据没有统一由`experiments/<experiment_id>/`索引。
4. 工作区存在`.DS_Store`、`__pycache__`、`.pytest_cache`。它们已被忽略，可安全清理，但本轮不会擅自删除。
5. `docs/`内存在未跟踪PPTX、`*.inspect.ndjson`及报告生成脚本；用途需用户确认后再决定归档或纳入版本控制。

### 3.5 Notebook与正式代码

1. 重要算法大部分已经提取到`src/`，Notebook主要负责环境、编排和展示，这是当前优点。
2. 根目录Notebook副本破坏了单一权威来源。
3. `07_train_minute_ppu_jupyterlab.ipynb`仍代表MemMap路径，`08`代表RAM路径，但编号没有表达两者是并列部署方案而非顺序步骤。
4. Notebook内的Git分支、仓库地址和平台数据路径需要被视为部署参数，而不是模型逻辑。

### 3.6 Git与依赖

1. `.gitignore`已覆盖Python缓存、Notebook缓存、模型、结果、实验输出、压缩包、大型数据、环境和IDE文件，基础较好。
2. 尚未显式忽略RAM快照目录、离线包目录、PowerPoint临时锁文件`~$*`和`*.inspect.ndjson`。
3. `requirements.txt`把运行依赖、训练依赖、可视化依赖、测试和Jupyter混在一起；RQAlphaPlus正确地未作为公开pip依赖。
4. `requirements-ddb.txt`直接包含完整requirements，导致仅做DDB抽取也会安装Torch、Jupyter和LightGBM。
5. 没有`pyproject.toml`，因此缺少统一的Python版本、pytest配置、包元数据和可选依赖声明。
6. 代码中未发现明文Token或密码；DDB连接使用环境变量。未跟踪二进制演示材料无法通过文本扫描确认敏感信息。

### 3.7 测试与质量

1. 当前完整测试基线为`121 passed, 1 skipped`，覆盖日频数据、表达式、GFlowNet、分钟DDB/RAM、AlphaEval、LightGBM、指数增强和RQAlpha策略。
2. 尚无静态检查、格式检查或CI配置。
3. 测试需要`PYTHONPATH=.`运行，反映项目尚未建立标准可安装包配置。
4. 超长模块缺少更细粒度的单元边界，但关键分钟NumPy语义已有与Pandas实现的一致性测试。

## 4. 风险与处理等级

### 4.1 核心代码：暂时不要移动

- `src/expression/`、`src/operators/`：表达式和算子语义核心。
- `src/gflownet/`：模型、TB训练、奖励与因子池核心。
- `src/alpha_eval/`、`src/model/`：筛选与融合核心。
- `src/index_enhancement/`：Universe、标签、组合和归因核心。
- `rqalpha_strategy/strategy.py`：授权回测运行时直接加载。
- `configs/*.yaml`：当前Notebook、文档和命令行大量直接引用。

这些文件可以在测试保护下拆分或分类，但不应一次性整体移动或改包名。

### 4.2 建议拆分重构

- `src/data_loader/minute_memmap.py`：优先拆出`minute_dense.py`、`minute_ram_cache.py`，保留兼容导入。
- `src/gflownet/memmap_reward.py`：后续按缓存、任务编排、执行器拆分。
- `src/data_loader/dolphindb_minute.py`：后续按配置/SQL/Loader拆分。
- `rqalpha_strategy/run_backtest.py`：后续分离配置装配和报告输出。

为了保持算法不变，本轮只建议先拆分最明确、测试覆盖最充分的RAM缓存和稠密分钟工具。

### 4.3 建议归档，不直接删除

- 根目录`06_rqalpha_backtest.ipynb`：先移至`notebooks/archived/`并标注来源，保留`notebooks/06...`为当前入口。
- 阶段性分析报告、交接记忆：移至`docs/archive/`或`docs/reports/`。
- 旧PPU MemMap Notebook：移至明确的`notebooks/deployment/`并保留为MemMap方案，而不是删除。
- 未跟踪PPTX、HTML、NDJSON：在确认是否属于项目交付前保持原位，不纳入自动整理提交。

### 4.4 可以安全删除，但本轮暂不执行

- `.DS_Store`；
- 所有`__pycache__/`和`*.pyc`；
- `.pytest_cache/`；
- PowerPoint临时锁文件`~$*.pptx`；
- 已确认仅为检查器中间结果的`*.inspect.ndjson`。

删除这些文件不影响源码，但仍应在独立清理阶段执行并记录。

### 4.5 不得删除

- `data/`中的原始或处理后数据；
- `results*`中的历史实验和回测结果；
- `checkpoints/`模型；
- Colab产物压缩包；
- 未确认用途的PPTX/报告；
- RQAlphaPlus授权环境或bundle。

这些资产可以通过文档和目录约定管理，但不能未经确认删除。

## 5. 依赖关系概览

```text
scripts / notebooks / rqalpha_strategy
                 │
                 ▼
       src.data_loader ───────► src.operators
                 │                    │
                 ▼                    ▼
        src.expression ───────► src.gflownet
                                      │
                    ┌─────────────────┼───────────────┐
                    ▼                 ▼               ▼
             src.alpha_eval       src.model   src.index_enhancement
                    │                 │               │
                    └─────────────────┴───────┬───────┘
                                              ▼
                                    rqalpha_strategy
```

可选外部边界：DolphinDB Python API仅用于分钟数据接入；RQData仅用于一次性下载指数数据；RQAlphaPlus仅用于本地授权回测；Google Drive仅用于Colab数据与产物传输。

## 6. 审计结论

项目不是“算法散落在Notebook中的原型”，而是已经形成了测试覆盖较好的研究代码库。最安全且高收益的整理路径是：

1. 保持`src`核心包和算法接口不变；
2. 先建立清晰的文档、配置、Notebook和输出分类；
3. 归档根目录遗留Notebook，不删除数据和实验；
4. 在兼容导出的前提下拆分分钟超长模块；
5. 增加`pyproject.toml`和分层依赖文件；
6. 最后更新README、运行手册、架构、工作流和实验索引。

具体移动、重命名、归档和代码拆分将在`RESTRUCTURE_PLAN.md`中逐项列出，得到可验证计划后再执行。
