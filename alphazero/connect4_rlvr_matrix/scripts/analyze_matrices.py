from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


GROUP_ORDER = ["baseline", "noise_c001", "noise_c005", "noise_c010", "nonshared"]
GROUP_LABELS = {
    "baseline": "baseline shared",
    "noise_c001": "noise c=0.01",
    "noise_c005": "noise c=0.05",
    "noise_c010": "noise c=0.10",
    "nonshared": "non-shared",
}
GROUP_COLORS = {
    "baseline": "#1f77b4",
    "noise_c001": "#2ca02c",
    "noise_c005": "#ff7f0e",
    "noise_c010": "#d62728",
    "nonshared": "#9467bd",
}

RUN_RE = re.compile(
    r"^connect4_fast_(?P<tag>baseline_rollout32|nonshared_rollout32|noise_c001|noise_c005|noise_c010)_seed(?P<seed>\d+)_(?P<stamp>\d+)$"
)

CURVE_METRICS = [
    "rollout/p",
    "info/H_p_bits",
    "info/I_cumulative_bits",
    "entropy/action_avg",
    "entropy/seq_cumulative",
    "reward_info/H_correct_bits",
    "reward_info/H_spec_bits",
    "reward_info/H_combined_bits",
    "reward_info/H_sum_components_bits",
    "noise/std",
    "loss/value",
    "train/success_rate",
    "val/success_rate",
]

FINAL_METRICS = [
    "rollout/p",
    "info/H_p_bits",
    "info/I_cumulative_bits",
    "entropy/action_avg",
    "entropy/seq_cumulative",
    "train/success_rate",
    "val/success_rate",
    "reward_info/H_correct_bits",
    "reward_info/H_spec_bits",
    "reward_info/H_combined_bits",
    "reward_info/H_sum_components_bits",
    "noise/c",
    "noise/std",
    "loss/value",
    "selfplay/shared_params",
    "offline_train/position_policy_acc",
    "offline_train/position_top3_acc",
    "offline_train/position_value_mse",
    "offline_val/position_policy_acc",
    "offline_val/position_top3_acc",
    "offline_val/position_value_mse",
    "offline_train/position_policy_nll",
    "offline_val/position_policy_nll",
]

OFFLINE_METRICS = [
    "offline_train/position_policy_acc",
    "offline_train/position_top3_acc",
    "offline_train/position_value_mse",
    "offline_val/position_policy_acc",
    "offline_val/position_top3_acc",
    "offline_val/position_value_mse",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Connect4 fast ablation matrix.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--summary-csv", default="runs/summary.csv")
    parser.add_argument("--offline-csv", default="runs/position_eval_summary.csv")
    parser.add_argument("--figures-dir", default="runs/analysis_figures")
    parser.add_argument("--tables-dir", default="runs/analysis_tables")
    parser.add_argument("--report", default="runs/analysis_report.md")
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    report_path = Path(args.report)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(runs_dir)
    metrics_df, final_df = load_metric_records(runs)
    summary_df = read_optional_csv(Path(args.summary_csv))
    offline_df = read_optional_csv(Path(args.offline_csv))

    final_df = merge_external_summaries(final_df, summary_df, offline_df)
    validate_matrix(final_df, offline_df)

    step_agg = aggregate_step_curves(metrics_df)
    final_agg = aggregate_final(final_df)
    offline_agg = aggregate_final(final_df[["group", "seed", "run_id", *OFFLINE_METRICS]])

    final_df.to_csv(tables_dir / "final_by_run.csv", index=False)
    final_agg.to_csv(tables_dir / "final_by_group.csv", index=False)
    step_agg.to_csv(tables_dir / "step_curves_by_group.csv", index=False)
    offline_agg.to_csv(tables_dir / "offline_by_group.csv", index=False)

    figure_paths = make_figures(step_agg, final_df, figures_dir, save_pdf=not args.no_pdf)
    report_text = build_report(final_df, final_agg, figure_paths, report_path)
    report_path.write_text(report_text, encoding="utf-8")

    diagnostics = {
        "formal_runs": int(len(final_df)),
        "groups": {
            group: sorted(final_df.loc[final_df["group"] == group, "seed"].astype(int).tolist())
            for group in GROUP_ORDER
        },
        "figures": [str(path) for path in figure_paths],
        "tables": [
            str(tables_dir / "final_by_run.csv"),
            str(tables_dir / "final_by_group.csv"),
            str(tables_dir / "step_curves_by_group.csv"),
            str(tables_dir / "offline_by_group.csv"),
        ],
    }
    (tables_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    print(f"read {len(final_df)} formal runs")
    print(f"wrote {len(figure_paths)} png figures to {figures_dir}")
    print(f"wrote report to {report_path}")


def discover_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "metrics.jsonl").exists():
            continue
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            continue
        config = read_json(run_dir / "config.json")
        runs.append(
            {
                "run_id": run_dir.name,
                "run_name": config.get("run_name", run_dir.name),
                "group": parsed["group"],
                "seed": int(parsed["seed"]),
                "path": run_dir,
            }
        )
    return sorted(runs, key=lambda row: (GROUP_ORDER.index(row["group"]), row["seed"]))


def parse_run_id(run_id: str) -> dict[str, Any] | None:
    match = RUN_RE.match(run_id)
    if not match:
        return None
    tag = match.group("tag")
    if tag == "baseline_rollout32":
        group = "baseline"
    elif tag == "nonshared_rollout32":
        group = "nonshared"
    else:
        group = tag
    return {"group": group, "seed": int(match.group("seed"))}


def load_metric_records(runs: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for run in runs:
        run_records = []
        metrics_path = run["path"] / "metrics.jsonl"
        with metrics_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                run_records.append(item)
                row = {
                    "run_id": run["run_id"],
                    "run_name": run["run_name"],
                    "group": run["group"],
                    "seed": run["seed"],
                    "step": item.get("step"),
                }
                for metric in CURVE_METRICS:
                    row[metric] = item.get(metric, math.nan)
                records.append(row)
        if not run_records:
            continue
        last = run_records[-1]
        final = {
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "group": run["group"],
            "seed": run["seed"],
            "steps_logged": len(run_records),
            "final_step": last.get("step"),
        }
        for metric in FINAL_METRICS:
            final[metric] = last.get(metric, math.nan)
        final_rows.append(final)
    return pd.DataFrame(records), pd.DataFrame(final_rows)


def merge_external_summaries(
    final_df: pd.DataFrame, summary_df: pd.DataFrame, offline_df: pd.DataFrame
) -> pd.DataFrame:
    out = final_df.copy()
    if not summary_df.empty:
        summary_cols = [
            col
            for col in summary_df.columns
            if col in {"run_id", "val/success_rate_mean", "val/success_rate_std", "train/success_rate_mean", "train/success_rate_std"}
        ]
        if "run_id" in summary_cols:
            out = out.merge(summary_df[summary_cols], on="run_id", how="left")
    if not offline_df.empty:
        offline_cols = ["run_id", "checkpoint_step", *[col for col in offline_df.columns if col.startswith("offline_")]]
        offline_cols = list(dict.fromkeys(offline_cols))
        out = out.drop(columns=[col for col in offline_cols if col in out.columns and col != "run_id"], errors="ignore")
        out = out.merge(offline_df[offline_cols], on="run_id", how="left")
    out["reward_info/H_redundancy_bits"] = (
        out["reward_info/H_sum_components_bits"] - out["reward_info/H_combined_bits"]
    )
    out["offline_gap/position_policy_acc"] = (
        out["offline_train/position_policy_acc"] - out["offline_val/position_policy_acc"]
    )
    out["offline_gap/position_top3_acc"] = (
        out["offline_train/position_top3_acc"] - out["offline_val/position_top3_acc"]
    )
    out["offline_gap/position_value_mse"] = (
        out["offline_val/position_value_mse"] - out["offline_train/position_value_mse"]
    )
    return out


def validate_matrix(final_df: pd.DataFrame, offline_df: pd.DataFrame) -> None:
    if len(final_df) != 15:
        raise ValueError(f"expected 15 formal runs, found {len(final_df)}")
    for group in GROUP_ORDER:
        seeds = sorted(final_df.loc[final_df["group"] == group, "seed"].astype(int).tolist())
        if seeds != [0, 1, 2]:
            raise ValueError(f"group {group} has seeds {seeds}, expected [0, 1, 2]")
    missing = [col for col in OFFLINE_METRICS if col not in final_df.columns or final_df[col].isna().any()]
    if missing:
        raise ValueError(f"missing offline metrics after merge: {missing}")
    formal_offline = offline_df[offline_df["run_id"].isin(final_df["run_id"])] if not offline_df.empty else pd.DataFrame()
    if len(formal_offline) != 15:
        raise ValueError(f"expected 15 formal offline rows, found {len(formal_offline)}")


def aggregate_step_curves(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, step), chunk in metrics_df.groupby(["group", "step"], sort=False):
        row: dict[str, Any] = {"group": group, "step": step, "n": int(chunk["run_id"].nunique())}
        for metric in CURVE_METRICS:
            values = pd.to_numeric(chunk[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["group_order"] = out["group"].map({group: i for i, group in enumerate(GROUP_ORDER)})
    return out.sort_values(["group_order", "step"]).drop(columns=["group_order"])


def aggregate_final(final_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in final_df.columns
        if col not in {"run_id", "run_name", "group", "seed"}
        and pd.api.types.is_numeric_dtype(final_df[col])
    ]
    rows = []
    for group, chunk in final_df.groupby("group", sort=False):
        row: dict[str, Any] = {"group": group, "n": int(len(chunk))}
        for metric in metric_cols:
            values = pd.to_numeric(chunk[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["group_order"] = out["group"].map({group: i for i, group in enumerate(GROUP_ORDER)})
    return out.sort_values("group_order").drop(columns=["group_order"])


def make_figures(step_agg: pd.DataFrame, final_df: pd.DataFrame, figures_dir: Path, save_pdf: bool) -> list[Path]:
    paths = []
    paths.append(plot_curve_grid(step_agg, ["rollout/p", "info/H_p_bits", "info/I_cumulative_bits"], figures_dir / "01_rollout_p_h_i.png", save_pdf))
    paths.append(plot_curve_grid(step_agg, ["entropy/action_avg", "entropy/seq_cumulative"], figures_dir / "02_entropy_curves.png", save_pdf))
    paths.append(
        plot_curve_grid(
            step_agg,
            [
                "reward_info/H_correct_bits",
                "reward_info/H_spec_bits",
                "reward_info/H_combined_bits",
                "reward_info/H_sum_components_bits",
            ],
            figures_dir / "03_reward_information.png",
            save_pdf,
        )
    )
    paths.append(plot_curve_grid(step_agg, ["noise/std", "loss/value", "entropy/action_avg", "rollout/p"], figures_dir / "04_noise_training_effects.png", save_pdf))
    paths.append(plot_offline_bars(final_df, figures_dir / "05_offline_val_performance.png", save_pdf))
    paths.append(plot_offline_gaps(final_df, figures_dir / "06_offline_train_val_gap.png", save_pdf))
    paths.append(plot_shared_nonshared(step_agg, final_df, figures_dir / "07_shared_vs_nonshared.png", save_pdf))
    return paths


def plot_curve_grid(step_agg: pd.DataFrame, metrics: list[str], path: Path, save_pdf: bool) -> Path:
    ncols = 2 if len(metrics) > 1 else 1
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, metric in zip(axes.ravel(), metrics):
        for group in GROUP_ORDER:
            chunk = step_agg[step_agg["group"] == group]
            x = chunk["step"].astype(float).to_numpy()
            mean = chunk[f"{metric}_mean"].astype(float).to_numpy()
            std = chunk[f"{metric}_std"].fillna(0).astype(float).to_numpy()
            ax.plot(x, mean, label=GROUP_LABELS[group], color=GROUP_COLORS[group], linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std, color=GROUP_COLORS[group], alpha=0.12, linewidth=0)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_figure(fig, path, save_pdf)
    return path


def plot_offline_bars(final_df: pd.DataFrame, path: Path, save_pdf: bool) -> Path:
    metrics = [
        "offline_val/position_policy_acc",
        "offline_val/position_top3_acc",
        "offline_val/position_value_mse",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, metrics):
        means = []
        stds = []
        labels = []
        colors = []
        for group in GROUP_ORDER:
            values = pd.to_numeric(final_df.loc[final_df["group"] == group, metric], errors="coerce")
            means.append(values.mean())
            stds.append(values.std(ddof=1))
            labels.append(GROUP_LABELS[group])
            colors.append(GROUP_COLORS[group])
        ax.bar(range(len(labels)), means, yerr=stds, capsize=3, color=colors, alpha=0.85)
        ax.set_title(metric)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, save_pdf)
    return path


def plot_offline_gaps(final_df: pd.DataFrame, path: Path, save_pdf: bool) -> Path:
    metrics = [
        ("offline_gap/position_policy_acc", "train acc - val acc"),
        ("offline_gap/position_top3_acc", "train top3 - val top3"),
        ("offline_gap/position_value_mse", "val mse - train mse"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (metric, title) in zip(axes, metrics):
        means = []
        stds = []
        labels = []
        colors = []
        for group in GROUP_ORDER:
            values = pd.to_numeric(final_df.loc[final_df["group"] == group, metric], errors="coerce")
            means.append(values.mean())
            stds.append(values.std(ddof=1))
            labels.append(GROUP_LABELS[group])
            colors.append(GROUP_COLORS[group])
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar(range(len(labels)), means, yerr=stds, capsize=3, color=colors, alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, save_pdf)
    return path


def plot_shared_nonshared(step_agg: pd.DataFrame, final_df: pd.DataFrame, path: Path, save_pdf: bool) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    curve_metrics = ["rollout/p", "entropy/action_avg"]
    for ax, metric in zip(axes.ravel()[:2], curve_metrics):
        for group in ["baseline", "nonshared"]:
            chunk = step_agg[step_agg["group"] == group]
            x = chunk["step"].astype(float).to_numpy()
            mean = chunk[f"{metric}_mean"].astype(float).to_numpy()
            std = chunk[f"{metric}_std"].fillna(0).astype(float).to_numpy()
            ax.plot(x, mean, label=GROUP_LABELS[group], color=GROUP_COLORS[group], linewidth=2)
            ax.fill_between(x, mean - std, mean + std, color=GROUP_COLORS[group], alpha=0.12, linewidth=0)
        ax.set_title(metric)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    bar_metrics = ["offline_val/position_policy_acc", "offline_val/position_value_mse"]
    for ax, metric in zip(axes.ravel()[2:], bar_metrics):
        groups = ["baseline", "nonshared"]
        means = [final_df.loc[final_df["group"] == group, metric].mean() for group in groups]
        stds = [final_df.loc[final_df["group"] == group, metric].std(ddof=1) for group in groups]
        ax.bar(range(2), means, yerr=stds, capsize=3, color=[GROUP_COLORS[group] for group in groups], alpha=0.85)
        ax.set_title(metric)
        ax.set_xticks(range(2), [GROUP_LABELS[group] for group in groups], rotation=15, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, path, save_pdf)
    return path


def save_figure(fig: plt.Figure, png_path: Path, save_pdf: bool) -> None:
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    if save_pdf:
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_report(
    final_df: pd.DataFrame, final_agg: pd.DataFrame, figure_paths: list[Path], report_path: Path
) -> str:
    rel_figs = [
        os.path.relpath(path, start=report_path.parent).replace(os.sep, "/")
        for path in figure_paths
    ]
    baseline = group_stats(final_df, "baseline")
    nonshared = group_stats(final_df, "nonshared")
    best_policy = best_group(final_agg, "offline_val/position_policy_acc_mean", higher=True)
    best_top3 = best_group(final_agg, "offline_val/position_top3_acc_mean", higher=True)
    best_mse = best_group(final_agg, "offline_val/position_value_mse_mean", higher=False)

    lines = [
        "# Matrix 分析报告",
        "",
        "本报告分析 `runs/` 下 15 个正式 Connect4 fast matrix run：5 个设置，每个设置 3 个 seed。训练时 `success_rate` 作为辅助参考；主要 performance 口径使用离线 fixed-position evaluation。",
        "",
        "## 数据与 caveat",
        "",
        "- 正式分组：`baseline`, `noise_c001`, `noise_c005`, `noise_c010`, `nonshared`。",
        "- 每组 seed：0, 1, 2。",
        "- 离线 fixed-position labels 来自 depth=4 alpha-beta negamax + heuristic，不是完美 solver。",
        "- 当前任务是 Connect4 RLVR/AlphaZero 原型，不是 Coding Agent RLVR。",
        "",
        "## 1. rollout=32 的 p/H/I",
        "",
        f"![rollout p/H/I]({rel_figs[0]})",
        "",
        f"baseline 末步 `rollout/p` 为 {fmt_mean_std(baseline, 'rollout/p')}，`H_p_bits` 为 {fmt_mean_std(baseline, 'info/H_p_bits')}，`I_cumulative_bits` 为 {fmt_mean_std(baseline, 'info/I_cumulative_bits')}。non-shared 的 `rollout/p` 为 {fmt_mean_std(nonshared, 'rollout/p')}，通常更高，但累计信息量不一定更高，说明自博弈参数共享方式改变了访问分布和策略集中程度。",
        "",
        "## 2. action entropy 和 sequence cumulative entropy",
        "",
        f"![entropy curves]({rel_figs[1]})",
        "",
        f"baseline 末步 action entropy 为 {fmt_mean_std(baseline, 'entropy/action_avg')}，sequence cumulative entropy 为 {fmt_mean_std(baseline, 'entropy/seq_cumulative')}。non-shared action entropy 为 {fmt_mean_std(nonshared, 'entropy/action_avg')}，但 sequence cumulative entropy 更低，提示 episode 轨迹分布更集中或有效长度更短。noise 组需要结合 reward noise 与 value loss 一起看，不能只用 entropy 判断 performance。",
        "",
        "## 3. train/validation performance",
        "",
        f"![offline validation performance]({rel_figs[4]})",
        "",
        f"离线 validation 上，policy acc 最好的是 `{best_policy[0]}`（{best_policy[1]:.4f}），top3 acc 最好的是 `{best_top3[0]}`（{best_top3[1]:.4f}），value MSE 最低的是 `{best_mse[0]}`（{best_mse[1]:.4f}）。训练中的 `train/success_rate` / `val/success_rate` 多数 run 已接近或达到饱和，因此这里只作为辅助指标。",
        "",
        f"![offline train val gap]({rel_figs[5]})",
        "",
        "train/validation gap 图显示固定局面上的泛化差异并不完全等同于 self-play success rate。policy acc gap 接近 0 时更可信；value MSE gap 为正表示 validation value 拟合更差。",
        "",
        "## 4. combination reward 信息量",
        "",
        f"![reward information]({rel_figs[2]})",
        "",
        f"baseline 末步 `H_combined_bits` 为 {fmt_mean_std(baseline, 'reward_info/H_combined_bits')}，`H_sum_components_bits` 为 {fmt_mean_std(baseline, 'reward_info/H_sum_components_bits')}，二者差值为 {fmt_mean_std(baseline, 'reward_info/H_redundancy_bits')}。这个差值可以作为 correctness/spec 组件相关性或冗余的线索；差值越大，组件信息越不像独立相加。",
        "",
        "## 5. reward noise 影响",
        "",
        f"![noise effects]({rel_figs[3]})",
        "",
        "noise 影响重点看 `noise/std`, `loss/value`, `entropy/action_avg`, `rollout/p` 和离线 `position_value_mse`。从聚合结果看，noise 系数越高并不保证 policy acc 提升；较高 noise 通常会直接抬高 reward/value 目标的不确定性，应优先关注 validation value MSE 是否恶化。",
        "",
        "## 6. shared vs non-shared self-play 差异",
        "",
        f"![shared vs nonshared]({rel_figs[6]})",
        "",
        f"baseline 离线 validation policy acc 为 {fmt_mean_std(baseline, 'offline_val/position_policy_acc')}，value MSE 为 {fmt_mean_std(baseline, 'offline_val/position_value_mse')}；non-shared policy acc 为 {fmt_mean_std(nonshared, 'offline_val/position_policy_acc')}，value MSE 为 {fmt_mean_std(nonshared, 'offline_val/position_value_mse')}。non-shared 在 self-play 成功率上同样容易饱和，但固定局面 value MSE 明显更差，说明它的训练动态不应只用 gameplay success rate 评价。",
        "",
        "## 输出文件",
        "",
        "- 本次仓库归档路径：`alphazero/connect4_rlvr_matrix/analysis_tables/` 与 `alphazero/connect4_rlvr_matrix/analysis_figures/`。",
        "- 脚本默认本地输出：`runs/analysis_report.md`, `runs/analysis_tables/`, `runs/analysis_figures/`。",
    ]
    return "\n".join(lines) + "\n"


def group_stats(final_df: pd.DataFrame, group: str) -> pd.DataFrame:
    return final_df[final_df["group"] == group]


def fmt_mean_std(group_df: pd.DataFrame, metric: str) -> str:
    values = pd.to_numeric(group_df[metric], errors="coerce")
    return f"{values.mean():.4f} +/- {values.std(ddof=1):.4f}"


def best_group(final_agg: pd.DataFrame, metric: str, higher: bool) -> tuple[str, float]:
    values = final_agg[["group", metric]].dropna()
    idx = values[metric].idxmax() if higher else values[metric].idxmin()
    row = values.loc[idx]
    return str(row["group"]), float(row[metric])


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
