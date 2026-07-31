"""运行沪深300、中证500和中证1000的风险因子归因。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.index_enhancement.factor_attribution import (
    AttributionConfig,
    attribute_index,
    estimate_factor_returns,
    load_factor_exposures,
    load_price_returns,
)


INDEX_NAMES = {"csi300": "沪深300", "csi500": "中证500", "csi1000": "中证1000"}


def log(message: str) -> None:
    print(f"[归因] {message}", flush=True)


def plot_index(output: Path, index_key: str) -> None:
    exposure = pd.read_csv(output / "active_exposure_summary.csv", index_col="factor")
    contribution = pd.read_csv(output / "daily_return_attribution.csv", parse_dates=["date"])
    style = exposure.loc[exposure["group"].eq("style")].sort_values("mean")
    industry = exposure.loc[exposure["group"].eq("industry")]
    industry = industry.loc[industry["mean_abs"].nlargest(15).index].sort_values("mean")

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    colors = np.where(style["mean"] >= 0, "#C44E52", "#4C72B0")
    axes[0].barh(style.index, style["mean"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("平均主动风格暴露")
    axes[0].set_xlabel("标准化暴露差")
    colors = np.where(industry["mean"] >= 0, "#C44E52", "#4C72B0")
    axes[1].barh(industry.index, industry["mean"], color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("主要行业超配/低配")
    axes[1].set_xlabel("组合权重减基准权重")
    fig.suptitle(f"{INDEX_NAMES[index_key]}：主动风险暴露")
    fig.tight_layout()
    fig.savefig(output / "active_exposure.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    contribution = contribution.set_index("date")
    cumulative = contribution[
        [
            "realized_active_return",
            "style_contribution",
            "industry_contribution",
            "residual_contribution",
        ]
    ].cumsum()
    cumulative.columns = ["实际主动收益", "风格贡献", "行业贡献", "残差贡献"]
    ax = cumulative.plot(figsize=(13, 7), linewidth=1.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{INDEX_NAMES[index_key]}：累计加法收益归因")
    ax.set_ylabel("累计贡献")
    ax.grid(alpha=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(output / "cumulative_return_attribution.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def table(frame: pd.DataFrame, columns: list[str], rows: int = 5) -> str:
    selected = frame[columns].head(rows).copy()
    for column in columns:
        selected[column] = selected[column].map(lambda value: f"{value:.4f}")
    return selected.to_markdown()


def build_report(root: Path, regression_diagnostics: pd.DataFrame, results: list[dict]) -> None:
    lines = [
        "# 三指数增强风险因子归因报告",
        "",
        "## 方法与口径",
        "",
        "本报告使用每日实际持仓市值计算股票组合权重，并与同日指数成分权重比较。主动风险暴露等于组合风险暴露减基准风险暴露。",
        "",
        "因RQData官方因子收益和完整协方差尚未下载，本报告使用前一交易日51维风险暴露，对下一交易日股票收益进行每日截面岭回归，估计本地因子收益。该处理严格滞后一期，不使用未来信息。残差同时包含个股选择、现金、交易成本、滑点以及模型不能解释的收益。",
        "",
        f"因子收益回归共覆盖 {len(regression_diagnostics)} 个交易日，平均截面样本数 {regression_diagnostics['observations'].mean():.0f}，平均截面R²为 {regression_diagnostics['r_squared'].mean():.2%}。",
        "",
    ]
    for result in results:
        key = result["index_key"]
        path = root / key
        exposure = pd.read_csv(path / "active_exposure_summary.csv", index_col="factor")
        factor_contribution = pd.read_csv(path / "factor_contribution_summary.csv", index_col="factor")
        styles = exposure.loc[exposure["group"].eq("style")]
        industries = exposure.loc[exposure["group"].eq("industry")]
        style_high = styles.sort_values("mean", ascending=False)
        style_low = styles.sort_values("mean")
        industry_abs = industries.reindex(industries["mean_abs"].sort_values(ascending=False).index)
        factor_abs = factor_contribution.reindex(
            factor_contribution["cumulative_contribution"].abs().sort_values(ascending=False).index
        )
        lines.extend(
            [
                f"## {INDEX_NAMES[key]}",
                "",
                f"归因区间：{result['start_date']} 至 {result['end_date']}，共 {result['attribution_days']} 个交易日。组合暴露平均覆盖率为 {result['mean_portfolio_exposure_coverage']:.2%}，基准暴露平均覆盖率为 {result['mean_benchmark_exposure_coverage']:.2%}。",
                "",
                f"累计加法主动收益为 {result['cumulative_realized_active_return_additive']:.2%}；其中风格贡献 {result['cumulative_style_contribution']:.2%}，行业贡献 {result['cumulative_industry_contribution']:.2%}，残差贡献 {result['cumulative_residual_contribution']:.2%}。",
                "",
                "### 主要风格超配",
                "",
                table(style_high, ["mean", "mean_abs", "latest"]),
                "",
                "### 主要风格低配",
                "",
                table(style_low, ["mean", "mean_abs", "latest"]),
                "",
                "### 主要行业主动偏离",
                "",
                table(industry_abs, ["mean", "mean_abs", "latest"], rows=8),
                "",
                "### 主要收益贡献来源",
                "",
                table(factor_abs, ["cumulative_contribution", "annualized_contribution"], rows=10),
                "",
                f"![{INDEX_NAMES[key]}主动暴露]({key}/active_exposure.png)",
                "",
                f"![{INDEX_NAMES[key]}收益归因]({key}/cumulative_return_attribution.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 使用限制",
            "",
            "本次属于基于本地估计因子收益的研究归因，不等同于RQData官方风险模型收益归因。待完整因子协方差、特异风险和特异收益数据可用后，应复算预测跟踪误差、边际风险贡献和官方口径归因。",
            "",
        ]
    )
    (root / "因子归因报告.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure-root", default="data/rqdata/factor_exposure")
    parser.add_argument("--price", default="data/price.csv")
    parser.add_argument("--index-weights", default="data/index_weights.csv.gz")
    parser.add_argument("--backtest-root", default="results/index_enhancement_optimizer_backtest")
    parser.add_argument("--output", default="results/factor_attribution")
    parser.add_argument("--start-date", default="2024-01-02")
    parser.add_argument("--end-date", default="2026-07-17")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AttributionConfig(start_date=args.start_date, end_date=args.end_date)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    log("读取2024—2026年RQData风险暴露")
    exposure, styles, industries = load_factor_exposures(
        args.exposure_root, config.start_date, config.end_date
    )
    log(f"风险暴露行数={len(exposure):,}，风格={len(styles)}，行业={len(industries)}")
    log("读取价格并计算复权收盘价收益")
    returns = load_price_returns(args.price, config.start_date, config.end_date)
    log(f"收益记录={len(returns):,}，开始每日截面回归")
    factor_returns, diagnostics = estimate_factor_returns(
        exposure, returns, styles + industries, config
    )
    factor_returns.to_csv(output / "estimated_factor_returns.csv")
    diagnostics.to_csv(output / "factor_return_diagnostics.csv", index=False)
    log(
        f"因子收益日期={len(factor_returns)}，平均R²={diagnostics['r_squared'].mean():.2%}"
    )

    results = []
    for index_key in ["csi300", "csi500", "csi1000"]:
        log(f"计算{INDEX_NAMES[index_key]}主动暴露与收益归因")
        result = attribute_index(
            index_key=index_key,
            backtest_dir=Path(args.backtest_root) / index_key,
            benchmark_weight_path=args.index_weights,
            exposure=exposure,
            factor_returns=factor_returns,
            styles=styles,
            industries=industries,
            output_dir=output / index_key,
            config=config,
        )
        plot_index(output / index_key, index_key)
        results.append(result)
        log(
            f"{INDEX_NAMES[index_key]}：风格={result['cumulative_style_contribution']:.2%}，"
            f"行业={result['cumulative_industry_contribution']:.2%}，"
            f"残差={result['cumulative_residual_contribution']:.2%}"
        )

    (output / "attribution_manifest.json").write_text(
        json.dumps(
            {
                "method": "lagged_exposure_cross_sectional_ridge",
                "official_rqdata_factor_returns": False,
                "style_factors": styles,
                "industry_factors": industries,
                "regression_days": len(factor_returns),
                "mean_cross_section_r_squared": float(diagnostics["r_squared"].mean()),
                "indexes": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    build_report(output, diagnostics, results)
    log(f"全部完成：{output / '因子归因报告.md'}")


if __name__ == "__main__":
    main()
