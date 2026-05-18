from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        action_size: int,
        hidden_size: int = 128,
        num_blocks: int = 3,
        input_size: int = 27,
    ) -> None:
        super().__init__()
        self.action_size = action_size
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            *[ResidualBlock(hidden_size) for _ in range(num_blocks)],
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Tanh())

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(obs)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)
