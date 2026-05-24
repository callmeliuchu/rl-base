"""
env3：5x5 网格世界，Gymnasium 风格接口（独立文件，不依赖 gym / env2）。
仅包含环境动力学：reset / step / render / close，不含策略 π。

策略评估、值迭代、MC 等见 env3_mc.py（外部维护 policy，调用 env.transition）。

规则（0-index，row=0 为最上行）:
  - 目标 (3, 2)，进入 reward +1
  - 橙色格进入 reward -1
  - 撞墙 reward -1，留在原地
  - 其余 0
  - 默认 continuing：进目标不自动终止

用法:
    env = GridWorldEnv()
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    env.close()
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# 世界常量
# ---------------------------------------------------------------------------

N = 5
GOAL = (3, 2)
ORANGE = frozenset(
    {
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (3, 3),
        (4, 1),
    }
)
GAMMA = 0.9

# 动作向量 [dr, dc]：停、右、下、左、上
ACTION_VECTORS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
)

# ---------------------------------------------------------------------------
# 轻量 Space（接口对齐 gym，无第三方依赖）
# ---------------------------------------------------------------------------


class Discrete:
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def contains(self, x: Any) -> bool:
        return isinstance(x, int) and 0 <= x < self.n

    def sample(self, rng: random.Random | None = None) -> int:
        r = rng or random
        return r.randrange(self.n)

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class Box:
    def __init__(self, low, high, shape=(2,), dtype="int32") -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def contains(self, x: Any) -> bool:
        if len(x) != 2:
            return False
        return (
            self.low[0] <= x[0] <= self.high[0]
            and self.low[1] <= x[1] <= self.high[1]
        )

    def __repr__(self) -> str:
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"


# ---------------------------------------------------------------------------
# 转移与奖励（确定性）
# ---------------------------------------------------------------------------


def _in_bounds(row: int, col: int, rows: int, cols: int) -> bool:
    return 0 <= row < rows and 0 <= col < cols


def compute_reward_and_next(
    pos: tuple[int, int],
    delta: tuple[int, int],
    *,
    goal: tuple[int, int],
    orange: frozenset[tuple[int, int]],
    rows: int,
    cols: int,
    boundary_reward: float = -1.0,
    orange_reward: float = -1.0,
    goal_reward: float = 1.0,
    step_reward: float = -0.1,
) -> tuple[float, tuple[int, int]]:
    """返回 (reward, next_pos)。"""
    nr, nc = pos[0] + delta[0], pos[1] + delta[1]
    if not _in_bounds(nr, nc, rows, cols):
        return boundary_reward, pos
    next_pos = (nr, nc)
    if next_pos == goal:
        return goal_reward, next_pos
    if next_pos in orange:
        return orange_reward, next_pos
    return step_reward, next_pos


@dataclass(frozen=True)
class EnvConfig:
    rows: int = N
    cols: int = N
    goal_pos: tuple[int, int] = GOAL
    orange_cells: frozenset[tuple[int, int]] = ORANGE
    gamma: float = GAMMA
    max_steps: int = 200
    terminal_on_goal: bool = False
    boundary_reward: float = -1.0
    orange_reward: float = -10.0  # make_env() 实际使用此项（非 compute_reward_and_next 的默认参数）
    goal_reward: float = 1.0
    step_reward: float = 0.0


# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------


class GridWorldEnv:
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    ACTION_STAY = 0
    ACTION_RIGHT = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_UP = 4

    action_meanings = ("stay", "right", "down", "left", "up")

    def __init__(
        self,
        config: EnvConfig | None = None,
        observation_mode: str = "index",
        start_mode: str = "fixed",
        render_mode: str | None = None,
    ) -> None:
        self.config = config or EnvConfig()
        self.observation_mode = observation_mode
        self.start_mode = start_mode
        self.render_mode = render_mode

        self.action_space = Discrete(len(ACTION_VECTORS))
        if observation_mode == "index":
            self.observation_space = Discrete(self.config.rows * self.config.cols)
        elif observation_mode == "coords":
            self.observation_space = Box(
                low=[0, 0],
                high=[self.config.rows - 1, self.config.cols - 1],
            )
        else:
            raise ValueError("observation_mode must be 'index' or 'coords'")

        self._rng = random.Random()
        self._agent_pos: tuple[int, int] = (0, 0)
        self._steps = 0
        self._closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[int | list[int], dict]:
        if self._closed:
            raise RuntimeError("环境已 close，请重新构造实例")

        if seed is not None:
            self._rng.seed(seed)

        options = options or {}
        self._steps = 0

        if "start_pos" in options:
            pos = tuple(options["start_pos"])
            if not self._in_bounds(pos):
                raise ValueError(f"非法起点: {pos}")
            self._agent_pos = pos
        elif "start_index" in options:
            self._agent_pos = self.decode(int(options["start_index"]))
        elif self.start_mode == "fixed":
            self._agent_pos = (0, 0)
        elif self.start_mode == "random":
            candidates = [
                (r, c)
                for r in range(self.config.rows)
                for c in range(self.config.cols)
                if (r, c) != self.config.goal_pos
            ]
            self._agent_pos = self._rng.choice(candidates)
        else:
            raise ValueError("start_mode must be 'fixed' or 'random'")

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[int | list[int], float, bool, bool, dict]:
        if self._closed:
            raise RuntimeError("环境已 close")
        if not self.action_space.contains(action):
            raise ValueError(f"非法动作: {action}")

        self._steps += 1
        reward, terminated, next_pos = self._transition(self._agent_pos, action)
        self._agent_pos = next_pos
        truncated = self._steps >= self.config.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> str | None:
        lines: list[str] = []
        for r in range(self.config.rows):
            row: list[str] = []
            for c in range(self.config.cols):
                pos = (r, c)
                if pos == self._agent_pos:
                    ch = "A"
                elif pos == self.config.goal_pos:
                    ch = "G"
                elif pos in self.config.orange_cells:
                    ch = "O"
                else:
                    ch = "."
                row.append(ch)
            lines.append(" ".join(row))
        board = "\n".join(lines)

        if self.render_mode == "human":
            print(board)
            print()
            return None
        return board

    def close(self) -> None:
        self._closed = True

    def encode(self, pos: tuple[int, int]) -> int:
        r, c = pos
        return r * self.config.cols + c

    def decode(self, index: int) -> tuple[int, int]:
        return divmod(index, self.config.cols)

    def states(self) -> list[tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.config.rows)
            for c in range(self.config.cols)
        ]

    def action_vector(self, action: int) -> tuple[int, int]:
        return ACTION_VECTORS[action]

    def transition(
        self, pos: tuple[int, int], action: int
    ) -> tuple[tuple[int, int], float, bool]:
        """确定性转移，可供 DP / Q-learning 直接调用。"""
        reward, next_pos = compute_reward_and_next(
            pos,
            self.action_vector(action),
            goal=self.config.goal_pos,
            orange=self.config.orange_cells,
            rows=self.config.rows,
            cols=self.config.cols,
            boundary_reward=self.config.boundary_reward,
            orange_reward=self.config.orange_reward,
            goal_reward=self.config.goal_reward,
            step_reward=self.config.step_reward,
        )
        terminated = (
            self.config.terminal_on_goal and next_pos == self.config.goal_pos
        )
        return next_pos, reward, terminated

    def _transition(
        self, pos: tuple[int, int], action: int
    ) -> tuple[float, bool, tuple[int, int]]:
        next_pos, reward, terminated = self.transition(pos, action)
        return reward, terminated, next_pos

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        return _in_bounds(pos[0], pos[1], self.config.rows, self.config.cols)

    def _get_obs(self) -> int | list[int]:
        if self.observation_mode == "index":
            return self.encode(self._agent_pos)
        return [self._agent_pos[0], self._agent_pos[1]]

    def _get_info(self) -> dict:
        return {
            "position": self._agent_pos,
            "steps": self._steps,
            "goal": self.config.goal_pos,
        }


def make_env(**kwargs) -> GridWorldEnv:
    return GridWorldEnv(**kwargs)


if __name__ == "__main__":
    env = make_env(render_mode="human")
    obs, info = env.reset(seed=42)
    print("reset:", obs, info)

    for t in range(8):
        action = env.action_space.sample(env._rng)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step {t}: action={env.action_meanings[action]} "
            f"obs={obs} reward={reward} terminated={terminated} "
            f"truncated={truncated} pos={info['position']}"
        )
        if terminated or truncated:
            break

    env.render()
    env.close()
