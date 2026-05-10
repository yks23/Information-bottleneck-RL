import numpy as np


def discrete_entropy(values, base=2):
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    log_fn = np.log2 if base == 2 else np.log
    return float(-(p * log_fn(p + 1e-12)).sum())


def binary_entropy_from_successes(successes, base=2):
    successes = np.asarray(successes).astype(float)
    if successes.size == 0:
        return 0.0
    p = successes.mean()
    log_fn = np.log2 if base == 2 else np.log
    return float(-(p * log_fn(p + 1e-12) + (1 - p) * log_fn(1 - p + 1e-12)))


def group_binary_entropy(rewards, base=2):
    """
    rewards: array shaped [num_prompts, group_size], values in {0, 1}
    returns: entropy per prompt, p per prompt
    """
    rewards = np.asarray(rewards).astype(float)
    p = rewards.mean(axis=1)
    log_fn = np.log2 if base == 2 else np.log
    h = -(p * log_fn(p + 1e-12) + (1 - p) * log_fn(1 - p + 1e-12))
    return h, p


def group_stats(rewards):
    rewards = np.asarray(rewards).astype(float)
    h, p = group_binary_entropy(rewards)
    return {
        "avg_group_entropy": float(h.mean()),
        "median_group_entropy": float(np.median(h)),
        "all_fail_ratio": float((p == 0).mean()),
        "mixed_group_ratio": float(((p > 0) & (p < 1)).mean()),
        "all_correct_ratio": float((p == 1).mean()),
        "effective_update_fraction": float(((p > 0) & (p < 1)).mean()),
        "mean_reward": float(p.mean()),
    }
