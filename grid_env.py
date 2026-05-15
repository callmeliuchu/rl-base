from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Position = tuple[int, int]


@dataclass(frozen=True)
class GridConfig:
    rows: int = 5
    cols: int = 5
    goal_pos: Position = (3, 2)
    terminal_goal: bool = False
    orange_cells: tuple[Position, ...] = (
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 1),
    )
    step_reward: float = 0.0
    boundary_reward: float = -1.0
    orange_reward: float = -1.0
    goal_reward: float = 1.0
    max_steps: int = 50


class GridWorldEnv(gym.Env[np.ndarray | int, int]):
    """A deterministic grid world with orange penalty cells and a target."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: GridConfig | None = None,
        observation_mode: str = "index",
        start_mode: str = "random",
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GridConfig()
        self.observation_mode = observation_mode
        self.start_mode = start_mode
        self.render_mode = render_mode

        self.action_to_delta: dict[int, Position] = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }
        self.action_names = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
        }

        self.action_space = spaces.Discrete(4)
        if observation_mode == "index":
            self.observation_space = spaces.Discrete(
                self.config.rows * self.config.cols
            )
        elif observation_mode == "coords":
            self.observation_space = spaces.Box(
                low=np.array([0, 0], dtype=np.int32),
                high=np.array(
                    [self.config.rows - 1, self.config.cols - 1], dtype=np.int32
                ),
                shape=(2,),
                dtype=np.int32,
            )
        else:
            raise ValueError("observation_mode must be 'index' or 'coords'")

        self.orange_cells = set(self.config.orange_cells)
        self.valid_start_positions = [
            (row, col)
            for row in range(self.config.rows)
            for col in range(self.config.cols)
            if (row, col) != self.config.goal_pos
        ]

        self.agent_pos: Position = self.config.goal_pos
        self.steps = 0

    def states(self) -> list[Position]:
        return [
            (row, col)
            for row in range(self.config.rows)
            for col in range(self.config.cols)
        ]

    def actions(self) -> list[int]:
        return list(self.action_to_delta.keys())

    def is_terminal(self, pos: Position) -> bool:
        return self.config.terminal_goal and pos == self.config.goal_pos

    def is_goal(self, pos: Position) -> bool:
        return pos == self.config.goal_pos

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray | int, dict]:
        super().reset(seed=seed)
        self.steps = 0

        if self.start_mode == "fixed":
            self.agent_pos = (0, 0)
        elif self.start_mode == "random":
            index = self.np_random.integers(0, len(self.valid_start_positions))
            self.agent_pos = self.valid_start_positions[int(index)]
        else:
            raise ValueError("start_mode must be 'fixed' or 'random'")

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[np.ndarray | int, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")

        self.steps += 1
        next_pos, reward, terminated = self.transition(self.agent_pos, action)
        self.agent_pos = next_pos
        truncated = self.steps >= self.config.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> str | None:
        symbols: list[str] = []
        for row in range(self.config.rows):
            line: list[str] = []
            for col in range(self.config.cols):
                pos = (row, col)
                if pos == self.agent_pos:
                    cell = "A"
                elif pos == self.config.goal_pos:
                    cell = "G"
                elif pos in self.orange_cells:
                    cell = "O"
                else:
                    cell = "."
                line.append(cell)
            symbols.append(" ".join(line))

        board = "\n".join(symbols)
        if self.render_mode == "human":
            print(board)
            print()
            return None
        return board

    def encode(self, pos: Position) -> int:
        row, col = pos
        return row * self.config.cols + col

    def decode(self, index: int) -> Position:
        return divmod(index, self.config.cols)

    def transition(self, pos: Position, action: int) -> tuple[Position, float, bool]:
        if self.is_terminal(pos):
            return pos, 0.0, True

        delta = self.action_to_delta[action]
        candidate = (pos[0] + delta[0], pos[1] + delta[1])

        if not self._inside(candidate):
            return pos, self.config.boundary_reward, False
        if candidate == self.config.goal_pos:
            return candidate, self.config.goal_reward, self.config.terminal_goal
        if candidate in self.orange_cells:
            return candidate, self.config.orange_reward, False
        return candidate, self.config.step_reward, False

    def _move(self, pos: Position, delta: Position) -> Position:
        row = min(max(pos[0] + delta[0], 0), self.config.rows - 1)
        col = min(max(pos[1] + delta[1], 0), self.config.cols - 1)
        return (row, col)

    def _inside(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.config.rows and 0 <= pos[1] < self.config.cols

    def _get_obs(self) -> np.ndarray | int:
        if self.observation_mode == "index":
            return self.encode(self.agent_pos)
        return np.array(self.agent_pos, dtype=np.int32)

    def _get_info(self) -> dict:
        return {
            "position": self.agent_pos,
            "steps": self.steps,
        }


def greedy_rollout(env: GridWorldEnv, q_table: np.ndarray) -> list[Position]:
    obs, _ = env.reset(seed=0)
    visited: list[Position] = [env.agent_pos]
    done = False

    while not done:
        action = int(np.argmax(q_table[int(obs)]))
        obs, _, terminated, truncated, _ = env.step(action)
        visited.append(env.agent_pos)
        done = terminated or truncated

    return visited


def format_path(path: Iterable[Position]) -> str:
    return " -> ".join(f"({row},{col})" for row, col in path)
