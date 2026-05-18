from typing import Protocol

import numpy as np

from ibrlvr.envs.connect4 import Connect4State, validation_states as connect4_validation_states
from ibrlvr.envs.tictactoe import TicTacToeState, validation_states as tictactoe_validation_states


class GameState(Protocol):
    player: int

    @property
    def action_size(self) -> int: ...

    def legal_actions(self) -> list[int]: ...

    def legal_mask(self) -> np.ndarray: ...

    def step(self, action: int) -> "GameState": ...

    def winner(self) -> int: ...

    def is_terminal(self) -> bool: ...

    def terminal_value(self, perspective: int) -> float: ...

    def observation(self) -> np.ndarray: ...

    def spec_reward(self) -> float: ...


def make_initial_state(name: str) -> GameState:
    if name == "tictactoe":
        return TicTacToeState()
    if name == "connect4":
        return Connect4State(rows=6, cols=7, connect=4)
    if name == "connect4_small":
        return Connect4State()
    raise ValueError(f"unsupported environment: {name}")


def make_validation_states(name: str) -> list[GameState]:
    if name == "tictactoe":
        return tictactoe_validation_states()
    if name == "connect4":
        return connect4_validation_states(rows=6, cols=7, connect=4)
    if name == "connect4_small":
        return connect4_validation_states()
    raise ValueError(f"unsupported environment: {name}")


__all__ = ["Connect4State", "GameState", "TicTacToeState", "make_initial_state", "make_validation_states"]
