from __future__ import annotations

import math

from ibrlvr.metrics import InformationAccumulator, binary_entropy, categorical_entropy, reward_information
from ibrlvr.rewards import make_rlvr_verifier


class TerminalForReward:
    def __init__(self, winner: int, spec: float) -> None:
        self._winner = winner
        self._spec = spec

    def winner(self) -> int:
        return self._winner

    def spec_reward(self) -> float:
        return self._spec


def test_binary_entropy_edges_and_midpoint() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert math.isclose(binary_entropy(0.5), 1.0, rel_tol=1e-9)


def test_information_accumulator_sums_step_entropy() -> None:
    acc = InformationAccumulator()
    first = acc.update(0.5)
    second = acc.update(1.0)
    assert math.isclose(first["info/I_cumulative_bits"], 1.0, rel_tol=1e-9)
    assert math.isclose(second["info/I_cumulative_bits"], 1.0, rel_tol=1e-9)


def test_categorical_entropy_action_distribution() -> None:
    assert math.isclose(categorical_entropy([0.5, 0.5], base=2.0), 1.0, rel_tol=1e-9)
    assert categorical_entropy([1.0, 0.0], base=2.0) == 0.0


def test_reward_information_tracks_components_and_combination() -> None:
    info = reward_information(
        correct=[1.0, 0.0, 1.0, 0.0],
        spec=[1.0, 1.0, 1.0, 1.0],
        combined=[1.1, 0.1, 1.1, 0.1],
    )
    assert math.isclose(info["reward_info/H_correct_bits"], 1.0, rel_tol=1e-9)
    assert info["reward_info/H_spec_bits"] == 0.0
    assert math.isclose(info["reward_info/H_combined_bits"], 1.0, rel_tol=1e-9)
    assert math.isclose(info["reward_info/H_sum_components_bits"], 1.0, rel_tol=1e-9)


def test_rlvr_verifier_combines_verifiable_components() -> None:
    verifier = make_rlvr_verifier({"correctness_weight": 1.0, "format_weight": 0.25, "noise_c": 0.0})
    reward = verifier.score(TerminalForReward(winner=1, spec=1.0), information_bits=10.0, rng=__import__("numpy").random.default_rng(0))
    assert reward.correctness == 1.0
    assert reward.spec == 1.0
    assert reward.combined == 1.25
    assert reward.noisy == 1.25
    assert reward.noise_std == 0.0
