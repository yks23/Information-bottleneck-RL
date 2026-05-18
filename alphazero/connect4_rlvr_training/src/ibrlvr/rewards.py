from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class VerifiableState(Protocol):
    def winner(self) -> int: ...

    def spec_reward(self) -> float: ...


@dataclass(frozen=True)
class RLVRReward:
    """Reward from verifiers, not a learned reward model."""

    correctness: float
    spec: float
    combined: float
    noisy: float
    noise_std: float


class RLVRVerifier:
    """Computes verifiable reward components and their configured combination."""

    def __init__(self, correctness_weight: float, spec_weight: float, noise_c: float) -> None:
        self.correctness_weight = correctness_weight
        self.spec_weight = spec_weight
        self.noise_c = noise_c

    def score(
        self,
        terminal_state: VerifiableState,
        information_bits: float,
        rng: np.random.Generator,
    ) -> RLVRReward:
        correctness = 1.0 if terminal_state.winner() != 0 else 0.0
        spec = float(terminal_state.spec_reward())
        combined = self.correctness_weight * correctness + self.spec_weight * spec
        noise_std = float(np.sqrt(max(self.noise_c * information_bits, 0.0)))
        noisy = combined + float(rng.normal(0.0, noise_std)) if noise_std > 0.0 else combined
        return RLVRReward(
            correctness=correctness,
            spec=spec,
            combined=combined,
            noisy=noisy,
            noise_std=noise_std,
        )


def make_rlvr_verifier(config: dict[str, float]) -> RLVRVerifier:
    return RLVRVerifier(
        correctness_weight=float(config["correctness_weight"]),
        spec_weight=float(config["format_weight"]),
        noise_c=float(config.get("noise_c", 0.0)),
    )
