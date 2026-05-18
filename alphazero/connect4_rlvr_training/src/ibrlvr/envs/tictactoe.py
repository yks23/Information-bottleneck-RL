from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TicTacToeState:
    board: tuple[int, ...] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    player: int = 1

    @property
    def action_size(self) -> int:
        return 9

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        return [idx for idx, value in enumerate(self.board) if value == 0]

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[self.legal_actions()] = 1.0
        return mask

    def step(self, action: int) -> TicTacToeState:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}")
        board = list(self.board)
        board[action] = self.player
        return TicTacToeState(tuple(board), -self.player)

    def winner(self) -> int:
        lines = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        )
        for a, b, c in lines:
            total = self.board[a] + self.board[b] + self.board[c]
            if total == 3:
                return 1
            if total == -3:
                return -1
        return 0

    def is_terminal(self) -> bool:
        return self.winner() != 0 or all(value != 0 for value in self.board)

    def terminal_value(self, perspective: int) -> float:
        winner = self.winner()
        if winner == perspective:
            return 1.0
        if winner == -perspective:
            return -1.0
        return 0.0

    def observation(self) -> np.ndarray:
        board = np.array(self.board, dtype=np.float32).reshape(3, 3)
        current = (board == self.player).astype(np.float32)
        opponent = (board == -self.player).astype(np.float32)
        player_plane = np.full((3, 3), self.player, dtype=np.float32)
        return np.stack([current, opponent, player_plane], axis=0)

    def canonical_key(self) -> tuple[tuple[int, ...], int]:
        return self.board, self.player

    def spec_reward(self) -> float:
        return 1.0


def make_initial_state(name: str) -> TicTacToeState:
    if name != "tictactoe":
        raise ValueError(f"unsupported environment: {name}")
    return TicTacToeState()


def validation_states() -> list[TicTacToeState]:
    return [
        TicTacToeState(),
        TicTacToeState((1, -1, 0, 0, 1, 0, 0, -1, 0), 1),
        TicTacToeState((1, 1, 0, -1, -1, 0, 0, 0, 0), 1),
        TicTacToeState((-1, 0, 1, 0, 1, 0, 0, -1, 0), -1),
    ]
