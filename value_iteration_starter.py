from __future__ import annotations

import numpy as np

from grid_env import GridWorldEnv, Position


def initialize_values(env: GridWorldEnv) -> dict[Position, float]:
    """Initialize V(s). Change this if you want a non-zero initialization."""
    return {state: 0.0 for state in env.states()}


def one_step_lookahead(
    env: GridWorldEnv,
    state: Position,
    values: dict[Position, float],
    gamma: float,
) -> list[float]:
    """Return the value of each action under the current V(s)."""
    action_values: list[float] = []

    for action in env.actions():
        next_state, reward, terminated = env.transition(state, action)

        # TODO:
        # Implement q(s, a) = r + gamma * V(s')
        # In the default config, the goal is NOT terminal.
        # If you switch env.config.terminal_goal=True, use `terminated` here.
        raise NotImplementedError("Fill in one_step_lookahead()")

        action_values.append(q_sa)

    return action_values


def value_iteration(
    env: GridWorldEnv,
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_iterations: int = 10_000,
) -> tuple[dict[Position, float], dict[Position, int]]:
    """Compute V*(s) and a greedy policy."""
    values = initialize_values(env)

    for iteration in range(max_iterations):
        delta = 0.0
        new_values = values.copy()

        for state in env.states():
            if env.is_terminal(state):
                new_values[state] = 0.0
                continue

            action_values = one_step_lookahead(env, state, values, gamma)

            # TODO:
            # Replace the placeholder below with the Bellman optimality backup.
            raise NotImplementedError("Fill in the Bellman backup in value_iteration()")

            delta = max(delta, abs(best_value - values[state]))
            new_values[state] = best_value

        values = new_values
        if delta < theta:
            break

    policy: dict[Position, int] = {}
    for state in env.states():
        if env.is_terminal(state):
            continue

        action_values = one_step_lookahead(env, state, values, gamma)

        # TODO:
        # Pick the greedy action under the converged values.
        raise NotImplementedError("Fill in greedy policy extraction")

        policy[state] = best_action

    return values, policy


def print_value_grid(env: GridWorldEnv, values: dict[Position, float]) -> None:
    print("      c1      c2      c3      c4      c5")
    for row in range(env.config.rows):
        cells = [f"r{row + 1}"]
        for col in range(env.config.cols):
            cells.append(f"{values[(row, col)]:7.3f}")
        print(" ".join(cells))


def print_policy_grid(env: GridWorldEnv, policy: dict[Position, int]) -> None:
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    print("      c1 c2 c3 c4 c5")
    for row in range(env.config.rows):
        cells = [f"r{row + 1}"]
        for col in range(env.config.cols):
            pos = (row, col)
            if env.is_terminal(pos):
                cells.append(" G")
            elif env.is_goal(pos):
                cells.append(f"G{arrows[policy[pos]]}")
            else:
                cells.append(f" {arrows[policy[pos]]}")
        print(" ".join(cells))


if __name__ == "__main__":
    env = GridWorldEnv(observation_mode="index", start_mode="fixed")
    values, policy = value_iteration(env)
    print("Optimal state value:")
    print_value_grid(env, values)
    print()
    print("Greedy policy:")
    print_policy_grid(env, policy)
