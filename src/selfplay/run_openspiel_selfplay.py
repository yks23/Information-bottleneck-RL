"""Starter script: random-vs-random outcome entropy logging."""

import argparse
import random

import pyspiel

from src.common.entropy import discrete_entropy
from src.common.logging_utils import save_table


def rollout_random(game_name: str):
    game = pyspiel.load_game(game_name)
    state = game.new_initial_state()
    while not state.is_terminal():
        legal = state.legal_actions()
        state.apply_action(random.choice(legal))
    ret = state.returns()[0]
    return -1 if ret < 0 else (1 if ret > 0 else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="tic_tac_toe")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", default="logs/selfplay/random_baseline.csv")
    args = parser.parse_args()

    outcomes = [rollout_random(args.game) for _ in range(args.episodes)]
    p_win = sum(o == 1 for o in outcomes) / len(outcomes)
    p_draw = sum(o == 0 for o in outcomes) / len(outcomes)
    p_loss = sum(o == -1 for o in outcomes) / len(outcomes)

    row = {
        "step": 0,
        "game": args.game,
        "condition": "random_vs_random",
        "seed": 0,
        "p_win": p_win,
        "p_draw": p_draw,
        "p_loss": p_loss,
        "outcome_entropy": discrete_entropy(outcomes),
        "eval_win_rate": p_win,
    }
    save_table(args.out, [row])
    print(row)


if __name__ == "__main__":
    main()
