from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ibrlvr.envs import GameState, make_initial_state


@dataclass(frozen=True)
class LabeledPosition:
    state: GameState
    action: int
    value: float


def build_position_sets(config: dict[str, Any], seed: int) -> dict[str, list[LabeledPosition]]:
    cfg = config.get("evaluation", {}).get("position_eval", {})
    if not bool(cfg.get("enabled", False)):
        return {}

    env_name = config["env"]["name"]
    train_size = int(cfg.get("train_size", 128))
    valid_size = int(cfg.get("valid_size", 128))
    min_moves = int(cfg.get("min_moves", 4))
    max_moves = int(cfg.get("max_moves", 18))
    solver_depth = int(cfg.get("solver_depth", 4))
    return {
        "train_positions": generate_labeled_positions(
            env_name=env_name,
            size=train_size,
            seed=seed + 1009,
            min_moves=min_moves,
            max_moves=max_moves,
            solver_depth=solver_depth,
        ),
        "valid_positions": generate_labeled_positions(
            env_name=env_name,
            size=valid_size,
            seed=seed + 2003,
            min_moves=min_moves,
            max_moves=max_moves,
            solver_depth=solver_depth,
        ),
    }


def generate_labeled_positions(
    env_name: str,
    size: int,
    seed: int,
    min_moves: int,
    max_moves: int,
    solver_depth: int,
) -> list[LabeledPosition]:
    rng = np.random.default_rng(seed)
    positions: list[LabeledPosition] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    max_attempts = max(size * 100, 1000)

    while len(positions) < size and attempts < max_attempts:
        attempts += 1
        state = make_initial_state(env_name)
        target_moves = int(rng.integers(min_moves, max_moves + 1))
        for _ in range(target_moves):
            if state.is_terminal():
                break
            legal = state.legal_actions()
            state = state.step(int(rng.choice(legal)))
        if state.is_terminal() or not state.legal_actions():
            continue
        key = (state.canonical_key() if hasattr(state, "canonical_key") else repr(state), state.player)
        if key in seen:
            continue
        seen.add(key)
        action, value = solve_reference(state, solver_depth)
        positions.append(LabeledPosition(state=state, action=action, value=value))

    if len(positions) < size:
        raise RuntimeError(f"only generated {len(positions)} labeled positions out of requested {size}")
    return positions


def solve_reference(state: GameState, depth: int) -> tuple[int, float]:
    cache: dict[tuple[Any, int], float] = {}
    best_action = ordered_legal_actions(state)[0]
    best_value = -float("inf")
    alpha = -float("inf")
    beta = float("inf")
    for action in ordered_legal_actions(state):
        value = -negamax(state.step(action), depth - 1, -beta, -alpha, cache)
        if value > best_value:
            best_value = value
            best_action = action
        alpha = max(alpha, best_value)
    return best_action, float(np.clip(best_value, -1.0, 1.0))


def negamax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    cache: dict[tuple[Any, int], float],
) -> float:
    key = (state_key(state), depth)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if state.is_terminal():
        value = float(state.terminal_value(state.player))
        cache[key] = value
        return value
    if depth <= 0:
        value = heuristic_value(state)
        cache[key] = value
        return value

    best_value = -float("inf")
    original_alpha = alpha
    cutoff = False
    for action in ordered_legal_actions(state):
        value = -negamax(state.step(action), depth - 1, -beta, -alpha, cache)
        best_value = max(best_value, value)
        alpha = max(alpha, best_value)
        if alpha >= beta:
            cutoff = True
            break
    if not cutoff and best_value > original_alpha:
        cache[key] = float(best_value)
    return float(best_value)


def state_key(state: GameState) -> Any:
    if hasattr(state, "canonical_key"):
        return state.canonical_key()
    return repr(state)


def ordered_legal_actions(state: GameState) -> list[int]:
    actions = state.legal_actions()
    if hasattr(state, "cols"):
        center = int(getattr(state, "cols")) // 2
        return sorted(actions, key=lambda action: (abs(action - center), action))
    return actions


def heuristic_value(state: GameState) -> float:
    if state.is_terminal():
        return float(state.terminal_value(state.player))
    if all(hasattr(state, attr) for attr in ["rows", "cols", "connect", "board"]):
        return connect4_heuristic(state)
    return 0.0


def connect4_heuristic(state: GameState) -> float:
    rows = int(getattr(state, "rows"))
    cols = int(getattr(state, "cols"))
    connect = int(getattr(state, "connect"))
    board = np.array(getattr(state, "board"), dtype=np.int8).reshape(rows, cols)
    player = int(state.player)

    score = 0.0
    center = cols // 2
    score += 0.05 * float(np.sum(board[:, center] == player))
    score -= 0.05 * float(np.sum(board[:, center] == -player))

    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(rows):
        for col in range(cols):
            for dr, dc in directions:
                end_r = row + (connect - 1) * dr
                end_c = col + (connect - 1) * dc
                if not (0 <= end_r < rows and 0 <= end_c < cols):
                    continue
                window = [int(board[row + k * dr, col + k * dc]) for k in range(connect)]
                score += window_score(window, player)
                score -= window_score(window, -player)
    return float(np.tanh(score / 4.0))


def window_score(window: list[int], player: int) -> float:
    own = sum(value == player for value in window)
    opp = sum(value == -player for value in window)
    empty = sum(value == 0 for value in window)
    if own > 0 and opp > 0:
        return 0.0
    if own == 4:
        return 10.0
    if own == 3 and empty == 1:
        return 2.0
    if own == 2 and empty == 2:
        return 0.5
    if own == 1 and empty == 3:
        return 0.05
    return 0.0


@torch.no_grad()
def evaluate_labeled_positions(
    model: torch.nn.Module,
    device: torch.device,
    positions: list[LabeledPosition],
    prefix: str,
    top_k: int,
    batch_size: int = 256,
) -> dict[str, float]:
    if not positions:
        return {}

    total = 0
    top1 = 0
    topk = 0
    value_errors: list[float] = []
    policy_nll: list[float] = []

    for start in range(0, len(positions), batch_size):
        batch = positions[start : start + batch_size]
        obs = torch.tensor(np.stack([item.state.observation() for item in batch]), dtype=torch.float32, device=device)
        logits, values = model(obs)
        logits_np = logits.detach().cpu().numpy()
        values_np = values.detach().cpu().numpy()

        for idx, item in enumerate(batch):
            mask = item.state.legal_mask()
            masked_logits = logits_np[idx].copy()
            masked_logits[mask == 0.0] = -1e9
            ranking = np.argsort(masked_logits)[::-1]
            k = min(top_k, int(mask.sum()))
            total += 1
            top1 += int(int(ranking[0]) == item.action)
            topk += int(item.action in set(int(action) for action in ranking[:k]))
            value_errors.append(float((values_np[idx] - item.value) ** 2))
            target = torch.tensor([item.action], dtype=torch.long, device=device)
            policy_nll.append(float(F.cross_entropy(logits[idx].unsqueeze(0), target).item()))

    return {
        f"{prefix}/position_policy_acc": float(top1 / max(total, 1)),
        f"{prefix}/position_top{top_k}_acc": float(topk / max(total, 1)),
        f"{prefix}/position_value_mse": float(np.mean(value_errors)) if value_errors else 0.0,
        f"{prefix}/position_policy_nll": float(np.mean(policy_nll)) if policy_nll else 0.0,
        f"{prefix}/position_count": float(total),
    }
