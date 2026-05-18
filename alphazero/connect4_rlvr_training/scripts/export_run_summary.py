from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SUMMARY_KEYS = [
    "rollout/p",
    "info/H_p_bits",
    "info/I_cumulative_bits",
    "entropy/action_avg",
    "entropy/seq_cumulative",
    "train/success_rate",
    "val/success_rate",
    "train/position_policy_acc",
    "train/position_top3_acc",
    "train/position_value_mse",
    "val/position_policy_acc",
    "val/position_top3_acc",
    "val/position_value_mse",
    "reward_info/H_correct_bits",
    "reward_info/H_spec_bits",
    "reward_info/H_combined_bits",
    "reward_info/H_sum_components_bits",
    "rlvr/correctness_verifier",
    "rlvr/spec_verifier",
    "rlvr/combined_reward",
    "noise/c",
    "noise/std",
    "selfplay/shared_params",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out", default="runs/summary.csv")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(Path(args.runs_dir).glob("*")):
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        config = load_json(run_dir / "config.json")
        rows.append(summarize_run(run_dir.name, config, records))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "env",
        "seed",
        "steps_logged",
        "final_step",
        *SUMMARY_KEYS,
        "val/success_rate_mean",
        "val/success_rate_std",
        "train/success_rate_mean",
        "train/success_rate_std",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"wrote {len(rows)} run summaries to {out_path}")


def summarize_run(run_id: str, config: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    last = records[-1]
    row: dict[str, Any] = {
        "run_id": run_id,
        "run_name": config.get("run_name", run_id),
        "env": config.get("env", {}).get("name", ""),
        "seed": config.get("seed", ""),
        "steps_logged": len(records),
        "final_step": last.get("step", ""),
    }
    for key in SUMMARY_KEYS:
        row[key] = last.get(key, "")
    for key in ["val/success_rate", "train/success_rate"]:
        values = [float(record[key]) for record in records if key in record]
        row[f"{key}_mean"] = mean(values) if values else ""
        row[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
    return row


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
