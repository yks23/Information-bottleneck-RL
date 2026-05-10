"""Offline group sampling skeleton with dummy verifier fallback."""

import argparse
import random

from src.common.entropy import binary_entropy_from_successes
from src.common.logging_utils import save_table


def sample_dummy_rewards(num_prompts, group_size, p=0.3):
    return [[1 if random.random() < p else 0 for _ in range(group_size)] for _ in range(num_prompts)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--out", default="logs/llm/offline_groups.csv")
    args = parser.parse_args()

    groups = sample_dummy_rewards(args.num_prompts, args.group_size)
    rows = []
    for i, rewards in enumerate(groups):
        p = sum(rewards) / len(rewards)
        group_type = "mixed" if 0 < p < 1 else ("all_correct" if p == 1 else "all_fail")
        rows.append(
            {
                "prompt_id": i,
                "group_size": args.group_size,
                "mean_reward": p,
                "group_entropy": binary_entropy_from_successes(rewards),
                "group_type": group_type,
            }
        )

    save_table(args.out, rows)
    print(f"Saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
