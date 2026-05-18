from __future__ import annotations

import json
from pathlib import Path

from ibrlvr.cli.train import load_config, train


def test_training_smoke_writes_required_metrics(tmp_path: Path) -> None:
    config = load_config(Path("configs/tictactoe_smoke.yaml"))
    config["output_dir"] = str(tmp_path)
    config["run_name"] = "pytest_smoke"
    config["training"]["steps"] = 2
    config["training"]["episodes_per_step"] = 1
    config["training"]["batch_size"] = 2
    config["training"]["eval_interval"] = 1
    config["training"]["checkpoint_interval"] = 2
    config["mcts"]["num_rollouts"] = 4
    config["wandb"]["mode"] = "disabled"
    train(config)

    run_dirs = list(tmp_path.glob("pytest_smoke_*"))
    assert len(run_dirs) == 1
    metrics = [json.loads(line) for line in (run_dirs[0] / "metrics.jsonl").read_text().splitlines()]
    last = metrics[-1]
    required = {
        "rollout/p",
        "info/H_p_bits",
        "info/I_cumulative_bits",
        "entropy/action_avg",
        "entropy/seq_cumulative",
        "train/success_rate",
        "val/success_rate",
        "reward_info/H_sum_components_bits",
        "rlvr/correctness_verifier",
        "rlvr/spec_verifier",
        "rlvr/combined_reward",
        "noise/std",
        "selfplay/shared_params",
    }
    assert required.issubset(last)
