from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ibrlvr.envs import GameState
from ibrlvr.metrics import categorical_entropy


@dataclass
class Node:
    state: GameState
    prior: float = 0.0
    parent: Node | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    def __init__(self, model: torch.nn.Module, device: torch.device, num_rollouts: int, c_puct: float) -> None:
        self.model = model
        self.device = device
        self.num_rollouts = num_rollouts
        self.c_puct = c_puct

    @torch.no_grad()
    def run(self, state: GameState) -> tuple[np.ndarray, dict[str, float]]:
        root = Node(state=state)
        self._expand(root)
        success_values: list[float] = []
        for _ in range(self.num_rollouts):
            node = root
            search_path = [node]
            while node.expanded() and not node.state.is_terminal():
                _, node = self._select_child(node)
                search_path.append(node)
            value = node.state.terminal_value(node.state.player) if node.state.is_terminal() else self._expand(node)
            success_values.append(1.0 if value > 0.0 else 0.0)
            self._backpropagate(search_path, value)

        visits = np.zeros(state.action_size, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        policy = visits / max(float(visits.sum()), 1.0)
        stats = {
            "rollout/p": float(np.mean(success_values)) if success_values else 0.0,
            "mcts/root_value": root.value,
            "mcts/root_visit_entropy": categorical_entropy(policy.tolist(), base=2.0),
        }
        return policy, stats

    def _select_child(self, node: Node) -> tuple[int, Node]:
        total_visits = max(1, node.visit_count)
        best_score = -float("inf")
        best_action = -1
        best_child: Node | None = None
        for action, child in node.children.items():
            prior_score = self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count)
            value_score = -child.value
            score = value_score + prior_score
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            raise RuntimeError("cannot select from an unexpanded node")
        return best_action, best_child

    @torch.no_grad()
    def _expand(self, node: Node) -> float:
        obs = torch.tensor(node.state.observation(), dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(obs)
        logits_np = logits.squeeze(0).detach().cpu().numpy()
        mask = node.state.legal_mask()
        logits_np[mask == 0.0] = -1e9
        probs = _softmax(logits_np)
        for action in node.state.legal_actions():
            node.children[action] = Node(
                state=node.state.step(action),
                prior=float(probs[action]),
                parent=node,
            )
        return float(value.item())

    def _backpropagate(self, search_path: list[Node], value: float) -> None:
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value


def select_action(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    legal = np.asarray(policy, dtype=np.float64)
    if temperature <= 1e-6:
        return int(np.argmax(legal))
    adjusted = np.power(legal, 1.0 / temperature)
    if adjusted.sum() <= 0.0:
        adjusted = np.ones_like(adjusted) / len(adjusted)
    else:
        adjusted = adjusted / adjusted.sum()
    return int(rng.choice(len(adjusted), p=adjusted))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
