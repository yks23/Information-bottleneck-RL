from __future__ import annotations

import numpy as np
import torch

from ibrlvr.envs.tictactoe import TicTacToeState
from ibrlvr.envs import make_initial_state
from ibrlvr.envs.connect4 import Connect4State
from ibrlvr.mcts import MCTS
from ibrlvr.model import PolicyValueNet


def test_tictactoe_legal_actions_and_winner() -> None:
    state = TicTacToeState((1, 1, 0, -1, -1, 0, 0, 0, 0), 1)
    assert state.legal_actions() == [2, 5, 6, 7, 8]
    next_state = state.step(2)
    assert next_state.winner() == 1
    assert next_state.terminal_value(1) == 1.0
    assert next_state.terminal_value(-1) == -1.0


def test_mcts_visit_distribution_respects_rollouts_and_legal_mask() -> None:
    model = PolicyValueNet(action_size=9, hidden_size=16, num_blocks=1)
    state = TicTacToeState((1, 1, 0, -1, -1, 0, 0, 0, 0), 1)
    policy, stats = MCTS(model, torch.device("cpu"), num_rollouts=8, c_puct=1.5).run(state)
    assert np.isclose(policy.sum(), 1.0)
    assert policy[0] == 0.0
    assert policy[1] == 0.0
    assert stats["rollout/p"] >= 0.0
    assert stats["mcts/root_visit_entropy"] >= 0.0


def test_connect4_small_drop_winner_and_spec_reward() -> None:
    state = Connect4State()
    for action in [2, 1, 2, 1, 2, 0, 2]:
        state = state.step(action)
    assert state.winner() == 1
    assert state.terminal_value(1) == 1.0
    assert state.terminal_value(-1) == -1.0
    assert state.spec_reward() == 1.0


def test_mcts_supports_connect4_small_shape() -> None:
    state = make_initial_state("connect4_small")
    model = PolicyValueNet(action_size=state.action_size, hidden_size=16, num_blocks=1, input_size=75)
    policy, stats = MCTS(model, torch.device("cpu"), num_rollouts=4, c_puct=1.5).run(state)
    assert np.isclose(policy.sum(), 1.0)
    assert policy.shape == (5,)
    assert stats["mcts/root_visit_entropy"] >= 0.0
