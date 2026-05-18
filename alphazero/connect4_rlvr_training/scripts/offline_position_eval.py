from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ibrlvr.envs import make_initial_state
from ibrlvr.eval_positions import build_position_sets, evaluate_labeled_positions
from ibrlvr.model import PolicyValueNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--pattern", default="connect4_fast_*")
    parser.add_argument("--out", default="runs/position_eval_summary.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--valid-size", type=int, default=256)
    parser.add_argument("--min-moves", type=int, default=4)
    parser.add_argument("--max-moves", type=int, default=18)
    parser.add_argument("--solver-depth", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    rows = []
    position_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for run_dir in sorted(Path(args.runs_dir).glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        checkpoint = latest_checkpoint(run_dir)
        if not config_path.exists() or checkpoint is None:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        inject_position_eval_config(config, args)
        cache_key = (config["env"]["name"], int(config.get("seed", 0)))
        if cache_key not in position_cache:
            print(f"building position labels for env={cache_key[0]} seed={cache_key[1]}", flush=True)
            position_cache[cache_key] = build_position_sets(config, int(config.get("seed", 0)))
            print(f"built position labels for env={cache_key[0]} seed={cache_key[1]}", flush=True)
        row = evaluate_run(
            run_dir=run_dir,
            config=config,
            checkpoint=checkpoint,
            device=device,
            top_k=args.top_k,
            batch_size=args.batch_size,
            position_sets=position_cache[cache_key],
        )
        rows.append(row)
        (run_dir / "position_eval.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        print(f"evaluated {run_dir.name} at {checkpoint.name}", flush=True)

    write_summary(Path(args.out), rows)
    print(f"wrote {len(rows)} rows to {args.out}")


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


def inject_position_eval_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    config.setdefault("evaluation", {})["position_eval"] = {
        "enabled": True,
        "train_size": args.train_size,
        "valid_size": args.valid_size,
        "min_moves": args.min_moves,
        "max_moves": args.max_moves,
        "solver_depth": args.solver_depth,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
    }


def evaluate_run(
    run_dir: Path,
    config: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    top_k: int,
    batch_size: int,
    position_sets: dict[str, Any],
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    initial_state = make_initial_state(config["env"]["name"])
    model_cfg = config.get("model", {})
    model = PolicyValueNet(
        action_size=initial_state.action_size,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_blocks=int(model_cfg.get("num_blocks", 3)),
        input_size=int(np.prod(initial_state.observation().shape)),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    model.eval()

    metrics: dict[str, float] = {}
    metrics.update(
        evaluate_labeled_positions(
            model=model,
            device=device,
            positions=position_sets["train_positions"],
            prefix="offline_train",
            top_k=top_k,
            batch_size=batch_size,
        )
    )
    metrics.update(
        evaluate_labeled_positions(
            model=model,
            device=device,
            positions=position_sets["valid_positions"],
            prefix="offline_val",
            top_k=top_k,
            batch_size=batch_size,
        )
    )

    return {
        "run_id": run_dir.name,
        "run_name": config.get("run_name", run_dir.name),
        "env": config.get("env", {}).get("name", ""),
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(payload.get("step", -1)),
        **metrics,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_name",
        "env",
        "seed",
        "checkpoint",
        "checkpoint_step",
        "offline_train/position_policy_acc",
        "offline_train/position_top3_acc",
        "offline_train/position_value_mse",
        "offline_train/position_policy_nll",
        "offline_train/position_count",
        "offline_val/position_policy_acc",
        "offline_val/position_top3_acc",
        "offline_val/position_value_mse",
        "offline_val/position_policy_nll",
        "offline_val/position_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


if __name__ == "__main__":
    main()
