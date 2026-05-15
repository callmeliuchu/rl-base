from __future__ import annotations

import numpy as np

from grid_env import GridWorldEnv, format_path, greedy_rollout


def train_q_learning(
    episodes: int = 4000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon: float = 0.15,
) -> np.ndarray:
    env = GridWorldEnv(observation_mode="index", start_mode="random")
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)

    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[obs]))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            target = reward + gamma * np.max(q_table[next_obs]) * (not terminated)
            q_table[obs, action] += alpha * (target - q_table[obs, action])

            obs = next_obs
            done = terminated or truncated

    return q_table


if __name__ == "__main__":
    q_table = train_q_learning()
    env = GridWorldEnv(observation_mode="index", start_mode="fixed")
    path = greedy_rollout(env, q_table)

    print("Learned greedy path from (0,0):")
    print(format_path(path))
    print()
    print("Final board:")
    print(env.render())
