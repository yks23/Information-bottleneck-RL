from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.data: deque[Transition] = deque(maxlen=capacity)

    def add_many(self, transitions: list[Transition]) -> None:
        self.data.extend(transitions)

    def sample(self, batch_size: int) -> list[Transition]:
        if len(self.data) < batch_size:
            raise ValueError("not enough replay data")
        idxs = self.rng.choice(len(self.data), size=batch_size, replace=False)
        return [self.data[int(idx)] for idx in idxs]

    def __len__(self) -> int:
        return len(self.data)
