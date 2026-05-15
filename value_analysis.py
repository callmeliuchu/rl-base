from __future__ import annotations

import numpy as np

from grid_env import GridWorldEnv, Position


Policy = dict[Position, int]


ARROW_POLICY: Policy = {
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


def value_iteration(
    env: GridWorldEnv, gamma: float = 0.9, tol: float = 1e-10
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(env.observation_space.n, dtype=np.float64)

    while True:
        delta = 0.0
        new_values = values.copy()
        for state in range(env.observation_space.n):
            pos = env.decode(state)
            if env.is_terminal(pos):
                new_values[state] = 0.0
                continue

            q_values = []
            for action in range(env.action_space.n):
                next_pos, reward, terminated = env.transition(pos, action)
                next_state = env.encode(next_pos)
                q = reward + gamma * values[next_state] * (not terminated)
                q_values.append(q)
            best_value = max(q_values)
            new_values[state] = best_value
            delta = max(delta, abs(best_value - values[state]))

        values = new_values
        if delta < tol:
            break

    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    for state in range(env.observation_space.n):
        pos = env.decode(state)
        if env.is_terminal(pos):
            continue
        for action in range(env.action_space.n):
            next_pos, reward, terminated = env.transition(pos, action)
            next_state = env.encode(next_pos)
            q_table[state, action] = reward + gamma * values[next_state] * (
                not terminated
            )
    return values, q_table


def evaluate_policy(
    env: GridWorldEnv,
    policy: Policy,
    gamma: float = 0.9,
    tol: float = 1e-10,
) -> np.ndarray:
    values = np.zeros(env.observation_space.n, dtype=np.float64)

    while True:
        delta = 0.0
        new_values = values.copy()
        for state in range(env.observation_space.n):
            pos = env.decode(state)
            if env.is_terminal(pos):
                new_values[state] = 0.0
                continue

            action = policy.get(pos)
            if action is None:
                continue

            next_pos, reward, terminated = env.transition(pos, action)
            next_state = env.encode(next_pos)
            new_value = reward + gamma * values[next_state] * (not terminated)
            new_values[state] = new_value
            delta = max(delta, abs(new_value - values[state]))

        values = new_values
        if delta < tol:
            break

    return values


def format_value_grid(
    env: GridWorldEnv,
    values: np.ndarray,
) -> str:
    rows: list[str] = []
    for row in range(env.config.rows):
        cells: list[str] = []
        for col in range(env.config.cols):
            pos = (row, col)
            if pos in env.orange_cells:
                cells.append(f"[{values[env.encode(pos)]:5.3f}]")
            else:
                cells.append(f"{values[env.encode(pos)]:7.3f}")
        rows.append(" ".join(cells))
    return "\n".join(rows)


if __name__ == "__main__":
    env = GridWorldEnv(observation_mode="index", start_mode="fixed")
    optimal_values, _ = value_iteration(env, gamma=0.9)
    policy_values = evaluate_policy(env, ARROW_POLICY, gamma=0.9)

    print("Optimal value grid V*(s):")
    print(format_value_grid(env, optimal_values))
    print()
    print("Arrow policy value grid V^pi(s):")
    print(format_value_grid(env, policy_values))
