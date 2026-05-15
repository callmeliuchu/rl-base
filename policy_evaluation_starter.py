from __future__ import annotations

from typing import Callable

from grid_env import GridWorldEnv, Position


PolicyFn = Callable[[Position], int]


def evaluate_policy(
    env: GridWorldEnv,
    policy: PolicyFn,
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> dict[Position, float]:
    """Iterative policy evaluation for a deterministic policy."""
    values = {state: 0.0 for state in env.states()}

    for iteration in range(max_iterations):
        delta = 0.0
        new_values = values.copy()

        for state in env.states():
            if env.is_terminal(state):
                new_values[state] = 0.0
                continue

            action = policy(state)
            next_state, reward, terminated = env.transition(state, action)

            # TODO:
            # Implement V^pi(s) = r + gamma * V^pi(s')
            # In the default config, the goal is NOT terminal.
            # If you switch env.config.terminal_goal=True, use `terminated` here.
            raise NotImplementedError("Fill in evaluate_policy()")

            delta = max(delta, abs(updated_value - values[state]))
            new_values[state] = updated_value

        values = new_values
        if delta < theta:
            break

    return values


def screenshot_policy(state: Position) -> int:
    """
    Optional helper if you want to evaluate the arrows from the screenshot.
    Actions: 0=up, 1=right, 2=down, 3=left
    """
    mapping = {
        (0, 0): 1,
        (0, 1): 1,
        (0, 2): 1,
        (0, 3): 2,
        (0, 4): 2,
        (1, 0): 0,
        (1, 1): 0,
        (1, 2): 1,
        (1, 3): 2,
        (1, 4): 2,
        (2, 0): 0,
        (2, 1): 3,
        (2, 2): 2,
        (2, 3): 1,
        (2, 4): 2,
        (3, 0): 0,
        (3, 1): 1,
        (3, 3): 3,
        (3, 4): 2,
        (4, 0): 0,
        (4, 1): 1,
        (4, 2): 0,
        (4, 3): 3,
        (4, 4): 3,
    }
    return mapping[state]


def print_value_grid(env: GridWorldEnv, values: dict[Position, float]) -> None:
    print("      c1      c2      c3      c4      c5")
    for row in range(env.config.rows):
        cells = [f"r{row + 1}"]
        for col in range(env.config.cols):
            cells.append(f"{values[(row, col)]:7.3f}")
        print(" ".join(cells))


if __name__ == "__main__":
    env = GridWorldEnv(observation_mode="index", start_mode="fixed")
    values = evaluate_policy(env, screenshot_policy)
    print_value_grid(env, values)
