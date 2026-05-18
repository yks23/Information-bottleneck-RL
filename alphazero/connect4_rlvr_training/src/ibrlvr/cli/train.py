from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to load config files") from exc

from ibrlvr.eval_positions import build_position_sets, evaluate_labeled_positions
from ibrlvr.envs import make_initial_state, make_validation_states
from ibrlvr.logging import ExperimentLogger
from ibrlvr.mcts import MCTS, select_action
from ibrlvr.metrics import InformationAccumulator, categorical_entropy, reward_information
from ibrlvr.model import PolicyValueNet
from ibrlvr.replay import ReplayBuffer, Transition
from ibrlvr.rewards import make_rlvr_verifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--override", action="append", default=[], help="Override key=value, supports dotted keys.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    for override in args.override:
        apply_override(config, override)
    train(config)


def train(config: dict[str, Any]) -> None:
    seed = int(config.get("seed", 0))
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(config.get("device", "cpu"))
    initial_state = make_initial_state(config["env"]["name"])

    model_cfg = config.get("model", {})
    model = PolicyValueNet(
        action_size=initial_state.action_size,
        hidden_size=int(model_cfg.get("hidden_size", 128)),
        num_blocks=int(model_cfg.get("num_blocks", 3)),
        input_size=int(np.prod(initial_state.observation().shape)),
    ).to(device)
    opponent = clone_model(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    replay = ReplayBuffer(int(config["training"]["replay_capacity"]), rng)
    info_accumulator = InformationAccumulator()
    position_sets = build_position_sets(config, seed)

    run_id = f"{config.get('run_name', 'alphazero')}_{int(time.time())}"
    run_dir = Path(config.get("output_dir", "runs")) / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(run_dir, config)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    try:
        steps = int(config["training"]["steps"])
        for step in range(1, steps + 1):
            shared_params = bool(config["self_play"].get("shared_params", True))
            player_models = {1: model, -1: model if shared_params else opponent}
            episodes, generation_metrics = generate_episodes(config, player_models, device, rng, info_accumulator)
            replay.add_many([transition for episode in episodes for transition in episode.transitions])

            train_metrics = {}
            if len(replay) >= int(config["training"]["batch_size"]):
                train_metrics = train_batch(model, optimizer, replay, int(config["training"]["batch_size"]), device)

            if not shared_params and step % int(config["self_play"]["opponent_update_interval"]) == 0:
                opponent.load_state_dict(model.state_dict())

            metrics = {**generation_metrics, **train_metrics, "replay/size": float(len(replay))}
            if step % int(config["training"]["eval_interval"]) == 0 or step == 1:
                metrics.update(evaluate(config, model, device, rng, split="val"))
                metrics.update(evaluate(config, model, device, rng, split="train"))
                metrics.update(evaluate_position_sets(config, model, device, position_sets))

            logger.log(step, metrics)

            if step % int(config["training"]["checkpoint_interval"]) == 0 or step == steps:
                save_checkpoint(checkpoint_dir, step, model, optimizer, config)
                prune_checkpoints(checkpoint_dir, int(config["training"].get("keep_last_checkpoints", 3)))
    finally:
        logger.close()


class Episode:
    def __init__(self, transitions: list[Transition], metrics: dict[str, float]) -> None:
        self.transitions = transitions
        self.metrics = metrics


def generate_episodes(
    config: dict[str, Any],
    player_models: dict[int, torch.nn.Module],
    device: torch.device,
    rng: np.random.Generator,
    info_accumulator: InformationAccumulator,
) -> tuple[list[Episode], dict[str, float]]:
    episodes = []
    correct_rewards: list[float] = []
    spec_rewards: list[float] = []
    combined_rewards: list[float] = []
    noisy_rewards: list[float] = []
    rollout_ps: list[float] = []
    action_avg_entropies: list[float] = []
    seq_entropies: list[float] = []
    root_values: list[float] = []
    visit_entropies: list[float] = []

    reward_cfg = config["reward"]
    verifier = make_rlvr_verifier(reward_cfg)
    mcts_cfg = config["mcts"]
    for _ in range(int(config["training"]["episodes_per_step"])):
        state = make_initial_state(config["env"]["name"])
        transitions: list[Transition] = []
        state_players: list[int] = []
        step_entropies: list[float] = []
        while not state.is_terminal():
            model = player_models[state.player]
            search = MCTS(
                model=model,
                device=device,
                num_rollouts=int(mcts_cfg["num_rollouts"]),
                c_puct=float(mcts_cfg["c_puct"]),
            )
            policy, stats = search.run(state)
            rollout_ps.append(stats["rollout/p"])
            root_values.append(stats["mcts/root_value"])
            visit_entropies.append(stats["mcts/root_visit_entropy"])
            entropy = categorical_entropy(policy.tolist(), base=2.0)
            step_entropies.append(entropy)
            transitions.append(Transition(obs=state.observation(), policy=policy, value=0.0))
            state_players.append(state.player)
            action = select_action(policy, float(mcts_cfg.get("temperature", 1.0)), rng)
            state = state.step(action)

        reward = verifier.score(state, info_accumulator.bits, rng)

        trainable_transitions: list[Transition] = []
        for transition, player in zip(transitions, state_players, strict=True):
            value = state.terminal_value(player)
            if not bool(config["self_play"].get("shared_params", True)) and player == -1:
                continue
            else:
                transition.value = float(reward.noisy if value > 0 else -reward.noisy if value < 0 else 0.0)
            trainable_transitions.append(transition)

        correct_rewards.append(reward.correctness)
        spec_rewards.append(reward.spec)
        combined_rewards.append(reward.combined)
        noisy_rewards.append(reward.noisy)
        action_avg_entropies.append(float(np.mean(step_entropies)) if step_entropies else 0.0)
        seq_entropies.append(float(np.sum(step_entropies)))
        episodes.append(Episode(transitions=trainable_transitions, metrics={}))

    reward_info = reward_information(correct_rewards, spec_rewards, combined_rewards)
    metrics = {
        "rollout/p": float(np.mean(rollout_ps)) if rollout_ps else 0.0,
        **info_accumulator.update(float(np.mean(rollout_ps)) if rollout_ps else 0.0),
        "entropy/action_avg": float(np.mean(action_avg_entropies)) if action_avg_entropies else 0.0,
        "entropy/seq_cumulative": float(np.mean(seq_entropies)) if seq_entropies else 0.0,
        "reward/correct": float(np.mean(correct_rewards)) if correct_rewards else 0.0,
        "reward/spec": float(np.mean(spec_rewards)) if spec_rewards else 0.0,
        "reward/combined": float(np.mean(combined_rewards)) if combined_rewards else 0.0,
        "reward/noisy": float(np.mean(noisy_rewards)) if noisy_rewards else 0.0,
        "rlvr/correctness_verifier": float(np.mean(correct_rewards)) if correct_rewards else 0.0,
        "rlvr/spec_verifier": float(np.mean(spec_rewards)) if spec_rewards else 0.0,
        "rlvr/combined_reward": float(np.mean(combined_rewards)) if combined_rewards else 0.0,
        "noise/c": float(reward_cfg.get("noise_c", 0.0)),
        "noise/std": float(np.sqrt(max(float(reward_cfg.get("noise_c", 0.0)) * info_accumulator.bits, 0.0))),
        "mcts/root_value": float(np.mean(root_values)) if root_values else 0.0,
        "mcts/root_visit_entropy": float(np.mean(visit_entropies)) if visit_entropies else 0.0,
        "selfplay/shared_params": float(bool(config["self_play"].get("shared_params", True))),
        **reward_info,
    }
    return episodes, metrics


def train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    batch = replay.sample(batch_size)
    obs = torch.tensor(np.stack([item.obs for item in batch]), dtype=torch.float32, device=device)
    target_policy = torch.tensor(np.stack([item.policy for item in batch]), dtype=torch.float32, device=device)
    target_value = torch.tensor([item.value for item in batch], dtype=torch.float32, device=device)
    logits, value = model(obs)
    policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    value_loss = F.mse_loss(value, target_value)
    loss = policy_loss + value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss/policy": float(policy_loss.item()),
        "loss/value": float(value_loss.item()),
        "loss/total": float(loss.item()),
    }


@torch.no_grad()
def evaluate(
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    rng: np.random.Generator,
    split: str,
) -> dict[str, float]:
    games = 16 if split == "val" else 8
    rewards = []
    lengths = []
    for game_idx in range(games):
        states = make_validation_states(config["env"]["name"])
        state = states[game_idx % len(states)] if split == "val" else make_initial_state(config["env"]["name"])
        length = 0
        while not state.is_terminal():
            obs = torch.tensor(state.observation(), dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs)
            logits_np = logits.squeeze(0).detach().cpu().numpy()
            logits_np[state.legal_mask() == 0.0] = -1e9
            action = int(np.argmax(logits_np))
            state = state.step(action)
            length += 1
        rewards.append(1.0 if state.winner() != 0 else 0.0)
        lengths.append(length)
    return {
        f"{split}/success_rate": float(np.mean(rewards)),
        f"{split}/avg_reward": float(np.mean(rewards)),
        f"{split}/avg_episode_length": float(np.mean(lengths)),
    }


@torch.no_grad()
def evaluate_position_sets(
    config: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    position_sets: dict[str, Any],
) -> dict[str, float]:
    cfg = config.get("evaluation", {}).get("position_eval", {})
    if not position_sets:
        return {}
    top_k = int(cfg.get("top_k", 3))
    batch_size = int(cfg.get("batch_size", 256))
    metrics: dict[str, float] = {}
    metrics.update(
        evaluate_labeled_positions(
            model=model,
            device=device,
            positions=position_sets.get("train_positions", []),
            prefix="train",
            top_k=top_k,
            batch_size=batch_size,
        )
    )
    metrics.update(
        evaluate_labeled_positions(
            model=model,
            device=device,
            positions=position_sets.get("valid_positions", []),
            prefix="val",
            top_k=top_k,
            batch_size=batch_size,
        )
    )
    return metrics

def clone_model(model: PolicyValueNet) -> PolicyValueNet:
    return copy.deepcopy(model)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    path = checkpoint_dir / f"step_{step:07d}.pt"
    torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config}, path)


def prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    for path in checkpoints[:-keep_last]:
        path.unlink()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_override(config: dict[str, Any], item: str) -> None:
    key, raw_value = item.split("=", 1)
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = parse_value(raw_value)


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
