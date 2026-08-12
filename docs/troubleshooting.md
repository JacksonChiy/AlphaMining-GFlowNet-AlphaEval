# 常见故障排查

## Colab 没有使用 A100

先在 Colab 运行 GPU 检查单元格，确认 CUDA 可用、型号包含 A100 且显存符合预期。正式配置会拒绝非 A100；`--allow-non-a100` 只用于小数据冒烟测试。

## LightGBM 没有产生滚动预测窗口

报错通常表示可用交易日不足以同时容纳 `min_train_days`、5 日 purge 和预测窗口。检查数据截止日期和有效样本数；快速验证可降低 `min_train_days`，正式实验应扩大训练期，不能取消 purge。

## 已保存表达式无法导入或恢复

确认仓库代码和 Colab 产物来自同一 commit，并从仓库根目录运行。旧环境若缺少新导出符号，先拉取对应分支并重启 Kernel，避免 Python 继续使用已缓存的旧模块。

## DolphinDB 查询分区过多

不要对整表执行无日期过滤的 `min/max/count`，也不要一次查询跨越大量日期分区。审计和加载器应按 `TradeDays` 分块或逐日查询；若交易日表在另一数据库，在配置中分别填写交易日数据库与表。结束日期必须小于等于数据实际最后交易日。

## 分钟网格不是 241 根

本项目只接收 9:25、9:31–11:30、13:01–15:00。先运行质量审计定位盘前、午间或重复记录，不要简单把 `expected_minutes` 改成源表的异常数量。

## MemMap 构建显示一个 worker

确认设置的是当前配置中的 `build_workers`，而不是 Reward 的 `workers`；多线程构建还需要 loader factory 为每个 worker 建立独立 DDB session。日志中的 build worker 数才是实际生效值。

## PPU 加载 DDB 很慢

观察 `[DDBRAM] load_progress` 的阶段占比。常见瓶颈是字符串/时间标准化、特征派生和索引校验，而非网络。全量 RAM 模式首次加载完成后会保存 NPY 快照；下次配置指纹命中时应显示 `DDBRAMCache hit` 和 `ddb_queries=0`。

## Joblib worker 停止或内存异常

大数组不要用 Loky 多进程复制；RAM 模式必须使用 `reward_parallel_backend: threading`。MemMap 模式可使用多进程，但应降低 worker 数和每任务 block 数，并确认磁盘缓存没有损坏。

## 表达式产生全空值

日志若指出某个 block 没有值，优先检查该表达式的掩码、最小样本数、窗口长度和源通道覆盖率。当前执行器会对低覆盖因子施加 Reward 惩罚并在入池前执行硬门槛；不要为绕过错误直接填充未来数据。

## RQAlphaPlus 提示 `rqdatac is not initialized`

关闭不需要的期权模块，确认本地 RQData 已初始化且 bundle 可用。RQAlphaPlus 不走 `127.0.0.1:7890` 代理。授权安装、bundle 和配置示例见 `guides/backtest/rqalpha_plus_setup.md`。

## 回测参数没有进入策略

查看 `backtest_effective_config.json`，它记录 RQAlphaPlus 实际生效配置。命令行显式参数优先于 YAML；Notebook 必须把选定配置路径传给启动器，不能只在单元格中修改一个未被使用的字典。

## 测试与导入失败

从仓库根目录执行：

```bash
pip install -r requirements.txt
pytest -q
```

如果 Jupyter 安装到了另一个环境，应使用该环境的 Python 安装并注册 Kernel：`python -m ipykernel install --user --name <环境名>`。验证当前解释器可运行 `python -c "import sys; print(sys.executable)"`。
