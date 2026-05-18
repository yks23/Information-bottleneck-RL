from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Connect4State:
    rows: int = 5
    cols: int = 5
    connect: int = 4
    board: tuple[int, ...] | None = None
    player: int = 1

    def __post_init__(self) -> None:
        if self.board is None:
            object.__setattr__(self, "board", tuple([0] * (self.rows * self.cols)))
        if len(self.board or ()) != self.rows * self.cols:
            raise ValueError("board size does not match rows*cols")

    @property
    def action_size(self) -> int:
        return self.cols

    def legal_actions(self) -> list[int]:
        if self.is_terminal():
            return []
        board = self._array()
        return [col for col in range(self.cols) if board[0, col] == 0]

    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.float32)
        mask[self.legal_actions()] = 1.0
        return mask

    def step(self, action: int) -> Connect4State:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action}")
        board = self._array()
        for row in range(self.rows - 1, -1, -1):
            if board[row, action] == 0:
                board[row, action] = self.player
                break
        return Connect4State(self.rows, self.cols, self.connect, tuple(board.reshape(-1).tolist()), -self.player)

    def winner(self) -> int:
        board = self._array()
        directions = ((0, 1), (1, 0), (1, 1), (1, -1))
        for row in range(self.rows):
            for col in range(self.cols):
                player = int(board[row, col])
                if player == 0:
                    continue
                for dr, dc in directions:
                    end_r = row + (self.connect - 1) * dr
                    end_c = col + (self.connect - 1) * dc
                    if not (0 <= end_r < self.rows and 0 <= end_c < self.cols):
                        continue
                    if all(int(board[row + k * dr, col + k * dc]) == player for k in range(self.connect)):
                        return player
        return 0

    def is_terminal(self) -> bool:
        return self.winner() != 0 or all(value != 0 for value in self.board or ())

    def terminal_value(self, perspective: int) -> float:
        winner = self.winner()
        if winner == perspective:
            return 1.0
        if winner == -perspective:
            return -1.0
        return 0.0

    def observation(self) -> np.ndarray:
        board = self._array()
        current = (board == self.player).astype(np.float32)
        opponent = (board == -self.player).astype(np.float32)
        player_plane = np.full((self.rows, self.cols), self.player, dtype=np.float32)
        return np.stack([current, opponent, player_plane], axis=0)

    def canonical_key(self) -> tuple[tuple[int, ...], int]:
        return tuple(self.board or ()), self.player

    def spec_reward(self) -> float:
        """Second reward source: terminal winner has meaningful center control."""
        winner = self.winner()
        if winner == 0:
            return 0.0
        board = self._array()
        center_cols = {self.cols // 2}
        if self.cols >= 5:
            center_cols.add(self.cols // 2 - 1)
            center_cols.add(self.cols // 2 + 1)
        winner_center = sum(int(board[row, col] == winner) for row in range(self.rows) for col in center_cols)
        opponent_center = sum(int(board[row, col] == -winner) for row in range(self.rows) for col in center_cols)
        return 1.0 if winner_center >= opponent_center else 0.0

    def _array(self) -> np.ndarray:
        return np.array(self.board, dtype=np.int8).reshape(self.rows, self.cols)


def validation_states(rows: int = 5, cols: int = 5, connect: int = 4) -> list[Connect4State]:
    states: list[Connect4State] = [Connect4State(rows, cols, connect)]
    center = cols // 2
    left = max(center - 1, 0)
    right = min(center + 1, cols - 1)
    edge = cols - 1
    state = Connect4State(rows, cols, connect)
    for action in [center, left, center, left, center, 0]:
        state = state.step(action)
    states.append(state)
    state = Connect4State(rows, cols, connect)
    for action in [0, right, 1, right, 2, edge]:
        state = state.step(action)
    states.append(state)
    state = Connect4State(rows, cols, connect)
    for action in [edge, center, right, center, left, center]:
        state = state.step(action)
    states.append(state)
    return states
