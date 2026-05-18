from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


EPS = 1e-12


def binary_entropy(p: float, base: float = 2.0) -> float:
    p = min(max(float(p), 0.0), 1.0)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    value = -(p * math.log(p + EPS) + (1.0 - p) * math.log(1.0 - p + EPS))
    return value / math.log(base)


def categorical_entropy(probs: Iterable[float], base: float = math.e) -> float:
    arr = np.asarray(list(probs), dtype=np.float64)
    arr = arr[arr > 0.0]
    if arr.size == 0:
        return 0.0
    value = -float(np.sum(arr * np.log(arr + EPS)))
    if base == math.e:
        return max(value, 0.0)
    return max(value / math.log(base), 0.0)


class InformationAccumulator:
    def __init__(self) -> None:
        self.bits = 0.0
        self.nats = 0.0

    def update(self, p: float) -> dict[str, float]:
        h_bits = binary_entropy(p, base=2.0)
        h_nats = binary_entropy(p, base=math.e)
        self.bits += h_bits
        self.nats += h_nats
        return {
            "info/H_p_bits": h_bits,
            "info/H_p_nats": h_nats,
            "info/I_cumulative_bits": self.bits,
            "info/I_cumulative_nats": self.nats,
        }


def bernoulli_information(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    positives = float(np.mean(arr > 0.0))
    return binary_entropy(positives, base=2.0)


def reward_information(correct: list[float], spec: list[float], combined: list[float]) -> dict[str, float]:
    h_correct = bernoulli_information(correct)
    h_spec = bernoulli_information(spec)
    h_combined = categorical_entropy(_normalized_histogram(combined), base=2.0)
    return {
        "reward_info/H_correct_bits": h_correct,
        "reward_info/H_spec_bits": h_spec,
        "reward_info/H_combined_bits": h_combined,
        "reward_info/H_sum_components_bits": h_correct + h_spec,
    }


def _normalized_histogram(values: list[float]) -> list[float]:
    if not values:
        return []
    rounded = [round(float(value), 6) for value in values]
    unique, counts = np.unique(rounded, return_counts=True)
    del unique
    probs = counts.astype(np.float64) / float(np.sum(counts))
    return probs.tolist()
