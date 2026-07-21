# AlphaMining-GFlowNet-AlphaEval

## Project Introduction

This repository is a runnable **daily-frequency mini replication** of the report
《基于 GFlowNet 和 AlphaEval 的分钟频因子挖掘筛选框架》. It implements the complete
research path from a private local `price.csv` to expression mining, factor
evaluation, LightGBM score fusion, and an RQAlphaPlus strategy.

The project is research software, not investment advice. No external market-data
API is called. Raw data, checkpoints, model outputs, and backtest results are
ignored by Git.

## Research Background

The report describes a GFlowNet that samples a diverse population of formula
alphas, using a Transformer representation and Trajectory Balance training. It
then applies AlphaEval-style predictive-power, temporal-stability, perturbation,
financial-logic, and diversity checks before LightGBM fusion.

This daily MVP keeps those ideas while making the following explicit adaptations:

- the grammar is the requested OHLCV/VWAP daily grammar with windows 5/10/20/40/60;
- prefix-tree construction has a unique parent, so the backward-policy term in TB
  is exactly `log PB = 0`;
- the reward is `abs(RankIC) * (1 + clipped LongIR) * RiskPenalty`;
- industry and market-cap penalties are only enabled when those real fields exist;
- financial-logic evaluation is deterministic complexity/depth scoring rather
  than an external LLM call;
- the DPP stage uses greedy MAP selection on a quality-weighted PSD similarity
  kernel;
- all labels are forward 5-day returns and are isolated from factor calculation.

See [docs/methodology.md](docs/methodology.md) for formulas and leakage controls.

## Environment

Model training is designed for Google Colab with an NVIDIA A100 and PyTorch AMP.
Open `notebooks/03_train_gflownet_A100.ipynb`, choose **A100 GPU**, and run all
cells. The notebook prints:

- CUDA availability and runtime;
- GPU model;
- total GPU memory;
- PyTorch version;
- an explicit A100 assertion.

Local setup for preprocessing and tests:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

On macOS, the LightGBM wheel also requires the OpenMP runtime (`brew install
libomp`). Colab's Linux runtime does not need that extra step.

RQAlphaPlus is licensed and is installed separately through an authorized
Ricequant channel. See [docs/rqalpha_plus_setup.md](docs/rqalpha_plus_setup.md).

## Dataset

Put the local file at `data/price.csv`. Required canonical fields are:

| Field | Meaning |
|---|---|
| `date` | trading date |
| `code` | security identifier |
| `open`, `high`, `low`, `close` | daily prices |
| `volume` | daily volume |
| `amount`, `vwap` | optional turnover and VWAP |
| `adj_factor` | optional point-in-time adjustment factor |
| `industry`, `market_cap` | optional risk-exposure fields |

English vendor aliases and common Chinese names are mapped automatically. The
pipeline converts dates, sorts by security/time, removes duplicate/invalid rows,
forward-fills prices only within each security, handles non-finite values,
winsorizes within each date, applies an optional adjustment factor, and adds
same-date cross-sectional z-scores. It writes `data/daily_price.pkl` and a data
quality report. Back-filling is never used.

## Training

Every notebook contains a GitHub clone cell, dependency installation, GPU status,
and config loading. In Colab set the optional environment variable
`ALPHAMINING_REPO_URL` to your fork URL or replace the placeholder.

Run notebooks in order:

1. `01_data_prepare.ipynb`
2. `02_expression_engine.ipynb`
3. `03_train_gflownet_A100.ipynb`
4. `04_alpha_eval.ipynb`
5. `05_lgbm_model.ipynb`
6. `06_rqalpha_backtest.ipynb` (in an environment licensed for RQAlphaPlus)

The same stages can be orchestrated from the project root:

```bash
python -m scripts.run_daily_pipeline --pool-size 100
# Add --rqalpha-bundle /path/to/bundle to run the licensed backtest stage.
```

Formal training enforces an A100 by default. `--allow-non-a100` exists only for
small code-path smoke tests and must not be used for reported experiments.

The training notebook saves `checkpoints/gflownet_best.pt`, reloads it, and
generates `factor_001`, `factor_002`, ... plus a factor-value matrix.

## Model

The state includes action tokens, the partial expression, current/max depth,
operator count, feature count, and normalized node statistics. A Transformer
Encoder predicts the next valid feature/operator/window action. Invalid grammar
actions are masked. Training minimizes:

```text
(logZ + sum(logPF) - logReward - sum(logPB))^2
```

Sampling is on-policy and rewards are cached by canonical expression string.
Checkpoint contents include model/optimizer state, `logZ`, configs, vocabulary,
and training history.

## AlphaEval and LightGBM

`alpha_eval_result.csv` includes at least `factor`, `IC`, `RankIC`, `ICIR`,
`Sharpe`, `complexity`, and `score`, plus rolling-IC, robustness, RRE and DPP
diagnostics. LightGBM uses a rolling window with a five-trading-day purge gap,
predicts future five-day return, saves its latest model, and emits daily scores
and ranks.

## Backtest

The repository does **not** contain a custom backtester. The strategy calls
RQAlphaPlus `run_file` and `order_target_portfolio`. It uses only the latest score
whose `signal_date` is strictly earlier than the current trading day, buys Top N
stocks equally, and rebalances every five trading days.

Default configuration:

- initial capital: CNY 1,000,000;
- benchmark: `000300.XSHG` (CSI 300);
- A-share default commissions/tax with point-in-time stamp tax enabled;
- price-ratio slippage: 0.001;
- report directory: `results/backtest_report/`.

RQAlphaPlus produces its summary, annual/total return, Sharpe, max drawdown,
volatility, turnover, equity curve, positions and trade history.

## Results

Real research outputs are generated only after the user supplies `price.csv`,
trains on Colab A100, and runs RQAlphaPlus with a valid local bundle. The project
does not ship fabricated checkpoints or performance numbers. Expected artifacts:

```text
checkpoints/gflownet_best.pt
results/alpha_pool.csv
results/alpha_factor_matrix.pkl
results/alpha_eval_result.csv
results/lightgbm/lgbm_model.joblib
results/lightgbm/prediction_score.csv
results/backtest_report/
```

## Experiment Versioning

`configs/training_config.yaml` is the source configuration. Each real run should
create `experiments/<experiment_id>/` and copy the frozen config and metrics into
that directory. Generated experiments are ignored; publish approved checkpoints
through a GitHub Release or Git LFS.

Recommended tags:

- `v0.1-data-pipeline`
- `v0.2-expression-engine`
- `v0.3-gflownet`
- `v0.4-alphaeval`
- `v0.5-backtest`
- `v1.0-release`

## Project Structure

```text
AlphaMining-GFlowNet-AlphaEval/
├── configs/
├── data/
├── docs/
├── experiments/
├── notebooks/
├── rqalpha_strategy/
├── src/
│   ├── alpha_eval/
│   ├── data_loader/
│   ├── expression/
│   ├── gflownet/
│   ├── model/
│   ├── operators/
│   └── utils/
└── tests/
```

## Future Work

- minute-level features with MemMap, block caching, Numba and multiprocessing;
- more report operators, including time-series binary operators;
- point-in-time industry/size/Barra exposure data and neutralization;
- embargoed nested validation and a fully held-out final research period;
- distributed reward evaluation and larger GFlowNet policies.
