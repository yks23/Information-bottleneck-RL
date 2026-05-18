from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from ibrlvr.cli.train import load_config, train
from ibrlvr.envs.tictactoe import TicTacToeState
from ibrlvr.mcts import MCTS
from ibrlvr.metrics import InformationAccumulator, binary_entropy, categorical_entropy, reward_information
from ibrlvr.model import PolicyValueNet


def test_metrics() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert math.isclose(binary_entropy(0.5), 1.0, rel_tol=1e-9)
    assert math.isclose(categorical_entropy([0.5, 0.5], base=2.0), 1.0, rel_tol=1e-9)

    acc = InformationAccumulator()
    assert math.isclose(acc.update(0.5)["info/I_cumulative_bits"], 1.0, rel_tol=1e-9)
    assert math.isclose(acc.update(1.0)["info/I_cumulative_bits"], 1.0, rel_tol=1e-9)

    info = reward_information(
        correct=[1.0, 0.0, 1.0, 0.0],
        spec=[1.0, 1.0, 1.0, 1.0],
        combined=[1.1, 0.1, 1.1, 0.1],
    )
    assert math.isclose(info["reward_info/H_correct_bits"], 1.0, rel_tol=1e-9)
    assert info["reward_info/H_spec_bits"] == 0.0
    assert math.isclose(info["reward_info/H_combined_bits"], 1.0, rel_tol=1e-9)


def test_env_and_mcts() -> None:
    state = TicTacToeState((1, 1, 0, -1, -1, 0, 0, 0, 0), 1)
    assert state.legal_actions() == [2, 5, 6, 7, 8]
    assert state.step(2).winner() == 1

    model = PolicyValueNet(action_size=9, hidden_size=16, num_blocks=1)
    policy, stats = MCTS(model, torch.device("cpu"), num_rollouts=8, c_puct=1.5).run(state)
    assert np.isclose(policy.sum(), 1.0)
    assert policy[0] == 0.0
    assert policy[1] == 0.0
    assert stats["rollout/p"] >= 0.0


def test_training_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = load_config(Path("configs/tictactoe_smoke.yaml"))
        config["output_dir"] = tmp
        config["run_name"] = "stdlib_smoke"
        config["training"]["steps"] = 2
        config["training"]["episodes_per_step"] = 1
        config["training"]["batch_size"] = 2
        config["training"]["eval_interval"] = 1
        config["training"]["checkpoint_interval"] = 2
        config["mcts"]["num_rollouts"] = 4
        config["wandb"]["mode"] = "disabled"
        train(config)

        run_dirs = list(Path(tmp).glob("stdlib_smoke_*"))
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
            "noise/std",
            "selfplay/shared_params",
        }
        assert required.issubset(last.keys())


if __name__ == "__main__":
    test_metrics()
    test_env_and_mcts()
    test_training_smoke()
    print("all tests passed")
