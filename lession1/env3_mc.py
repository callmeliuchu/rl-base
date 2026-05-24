"""
基于 env3.GridWorldEnv 的 Monte Carlo / 策略迭代（独立模块，不修改环境类）。

策略 π 由调用方维护为 dict[(row,col), list[float]]，即 π(a|s) 概率向量。
mc_policy_iteration / v2 使用确定性 π；mc_policy_iteration3 为 MC ε-greedy（Algorithm 5.3）。

用法:
    from env3 import make_env
    from env3_mc import default_policy, mc_policy_iteration3, print_values, print_policy

    env = make_env()
    v, policy = mc_policy_iteration3(env, epsilon=0.1)
    print_values(env, v)
    print_policy(env, policy)
"""

from __future__ import annotations

import copy
import random
from collections import defaultdict
from typing import TYPE_CHECKING

from env3 import GAMMA, GridWorldEnv, make_env, make_pg_env

if TYPE_CHECKING:
    pass

# π(a|s)：每个状态对应长度为 n_actions 的概率，和为 1
Policy = dict[tuple[int, int], list[float]]
Transition = tuple[tuple[int, int], int, float, tuple[int, int]]


def deterministic_probs(n_actions: int, action: int) -> list[float]:
    """单动作概率 1，其余为 0。"""
    probs = [0.0] * n_actions
    probs[action] = 1.0
    return probs


def set_deterministic_action(
    policy: Policy,
    pos: tuple[int, int],
    action: int,
    n_actions: int,
) -> None:
    """将 π(·|s) 设为确定性：只执行 action。"""
    policy[pos] = deterministic_probs(n_actions, action)


def default_policy(env: GridWorldEnv | None = None) -> Policy:
    """课件初始策略（当前为全 stay，确定性概率分布）。"""
    if env is None:
        env = make_env()
    n_actions = env.action_space.n
    stay = GridWorldEnv.ACTION_STAY
    policy: Policy = {}
    for i in range(env.config.rows):
        for j in range(env.config.cols):
            policy[(i, j)] = deterministic_probs(n_actions, stay)
    return policy


def uniform_policy(env: GridWorldEnv | None = None) -> Policy:
    """均匀随机策略 π(a|s) = 1/|A|。"""
    if env is None:
        env = make_env()
    n_actions = env.action_space.n
    p = 1.0 / n_actions
    policy: Policy = {}
    for i in range(env.config.rows):
        for j in range(env.config.cols):
            policy[(i, j)] = [p] * n_actions
    return policy


def policy_prob(policy: Policy, pos: tuple[int, int], action: int) -> float:
    """π(a|s)。"""
    probs = policy.get(pos)
    if probs is None or action >= len(probs):
        return 0.0
    return probs[action]


def policy_action_at(policy: Policy, pos: tuple[int, int]) -> int | None:
    """确定性策略下的动作：argmax_a π(a|s)。"""
    probs = policy.get(pos)
    if probs is None:
        return None
    return max(range(len(probs)), key=lambda a: probs[a])


def sample_action(probs: list[float]) -> int:
    """按 π(·|s) 采样一个动作。"""
    return random.choices(range(len(probs)), weights=probs, k=1)[0]


def epsilon_greedy_probs(
    q: dict[tuple[tuple[int, int], int], float],
    s: tuple[int, int],
    n_actions: int,
    epsilon: float,
) -> list[float]:
    """
    Algorithm 5.3 的 policy improvement：
      π(a*|s) = 1 - (|A|-1)/|A| * ε
      π(a|s)  = ε/|A|  (a ≠ a*)
    """
    probs = [epsilon / n_actions] * n_actions
    best_a = max(range(n_actions), key=lambda a: q.get((s, a), 0.0))
    probs[best_a] = 1.0 - epsilon + epsilon / n_actions
    return probs


def rollout_trajectory_stochastic(
    env: GridWorldEnv,
    start_pos: tuple[int, int],
    policy: Policy,
    max_steps: int = 100,
) -> list[Transition]:
    """全程按 π(·|s) 采样动作生成 episode。"""
    traj: list[Transition] = []
    pos = start_pos
    probs = policy.get(pos)
    if probs is None:
        return traj
    action = sample_action(probs)

    for _ in range(max_steps):
        next_pos, reward, terminated = env.transition(pos, action)
        traj.append((pos, action, reward, next_pos))
        if terminated:
            break
        pos = next_pos
        probs = policy.get(pos)
        if probs is None:
            break
        action = sample_action(probs)
    return traj


def rollout_trajectory(
    env: GridWorldEnv,
    start_pos: tuple[int, int],
    start_action: int,
    policy: Policy,
    max_steps: int = 100,
) -> list[Transition]:
    """
    轨迹生成：
      - 第 1 步：强制 start_action（用于估计 Q(s, start_action)）
      - 第 2 步起：按 π(·|s) 的 argmax 取动作（当前为确定性，不随机采样）
    """
    traj: list[Transition] = []
    pos = start_pos
    action = start_action

    for _ in range(max_steps):
        next_pos, reward, terminated = env.transition(pos, action)
        traj.append((pos, action, reward, next_pos))
        if terminated:
            break
        action = policy_action_at(policy, next_pos)
        if action is None:
            break
        pos = next_pos
    return traj


def discount_returns(
    trajectory: list[Transition],
    gamma: float = GAMMA,
) -> list[float]:
    """每一步的折扣回报 G_t。"""
    ret: list[float] = []
    g = 0.0
    for _pos, _act, reward, _next in reversed(trajectory):
        g = g * gamma + reward
        ret.append(g)
    ret.reverse()
    return ret


def mc_policy_iteration(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    gamma: float | None = None,
    theta: float = 1e-8,
    max_steps: int = 100,
) -> tuple[list[list[float]], Policy]:
    """
    MC 策略迭代（外部维护 policy，不写入 env）:
      1) 对每个 s,a 采轨迹，用 G_0 比较 Q(s,a)
      2) 贪心更新 π(·|s) 为确定性分布
      3) V(s) <- 最优动作的样本回报
    返回 (V, policy)。
    """
    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(policy if policy is not None else default_policy(env))
    n_actions = env.action_space.n
    n = env.config.rows
    v = [[0.0] * n for _ in range(n)]

    while True:
        delta = 0.0
        v_new = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                pos = (i, j)
                best_g = float("-inf")
                best_act: int | None = None
                for action in range(n_actions):
                    traj = rollout_trajectory(
                        env, pos, action, policy, max_steps=max_steps
                    )
                    if not traj:
                        continue
                    g0 = discount_returns(traj, gamma)[0]
                    if g0 > best_g:
                        best_g = g0
                        best_act = action
                if best_act is not None:
                    set_deterministic_action(policy, pos, best_act, n_actions)
                    v_new[i][j] = best_g
                else:
                    v_new[i][j] = v[i][j]
                delta = max(delta, abs(v_new[i][j] - v[i][j]))
        v = v_new
        if delta < theta:
            break
    return v, policy


def mc_policy_iteration2(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 5000,
) -> tuple[list[list[float]], Policy]:
    """
    MC Exploring Starts 风格（π 为概率表，改进步仍设为确定性 π）。
    """
    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(policy if policy is not None else default_policy(env))
    n_actions = env.action_space.n
    n = env.config.rows
    q: dict[tuple[tuple[int, int], int], float] = {}
    ret: dict[tuple[tuple[int, int], int], float] = {}
    c: dict[tuple[tuple[int, int], int], int] = {}

    for _ in range(num_episodes):
        pos = (random.randint(0, n - 1), random.randint(0, n - 1))
        action = random.randint(0, n_actions - 1)
        traj = rollout_trajectory(
            env, pos, action, policy, max_steps=max_steps
        )
        if not traj:
            continue
        g = 0.0
        for pos, act, r, _next_pos in reversed(traj):
            g = g * gamma + r
            sa = (pos, act)
            ret[sa] = ret.get(sa, 0.0) + g
            c[sa] = c.get(sa, 0) + 1
            q[sa] = ret[sa] / c[sa]

            best = float("-inf")
            best_act: int | None = None
            for a in range(n_actions):
                sa_ = (pos, a)
                if sa_ in q and best < q[sa_]:
                    best = q[sa_]
                    best_act = a
            if best_act is not None:
                set_deterministic_action(policy, pos, best_act, n_actions)

    v = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = (i, j)
            v[i][j] = max(
                (q.get((s, a), float("-inf")) for a in range(n_actions)),
                default=float("-inf"),
            )
            if v[i][j] == float("-inf"):
                v[i][j] = 0.0
    return v, policy


def mc_policy_iteration3(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    REINFORCE with a minimal state baseline:
      - episode 按当前 π(·|s) 采样动作
      - 用状态历史回报的运行平均作为 baseline b(s)
      - 用 advantage = G_t - b(s_t) 更新策略
    未传 policy 时从均匀策略开始。返回 (V, policy)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    n = env.config.rows
    q: dict[tuple[tuple[int, int], int], float] = {}
    returns: dict[tuple[tuple[int, int], int], float] = {}
    num: dict[tuple[tuple[int, int], int], int] = {}

    for _ in range(num_episodes):
        s0 = (random.randint(0, n - 1), random.randint(0, n - 1))
        traj = rollout_trajectory_stochastic(
            env, s0, policy, max_steps=max_steps
        )
        if not traj:
            continue

        g = 0.0
        for s, a, r, _next_pos in reversed(traj):
            g = gamma * g + r
            sa = (s, a)
            returns[sa] = returns.get(sa, 0.0) + g
            num[sa] = num.get(sa, 0) + 1
            q[sa] = returns[sa] / num[sa]
            policy[s] = epsilon_greedy_probs(q, s, n_actions, epsilon)

    v = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = (i, j)
            probs = policy.get(s, [1.0 / n_actions] * n_actions)
            v[i][j] = sum(
                probs[a] * q.get((s, a), 0.0) for a in range(n_actions)
            )
    return v, policy


def sarsa(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    MC ε-Greedy（Sutton & Barto Algorithm 5.3）:
      - episode 按当前 π(·|s) 采样动作（概率探索）
      - 用 Returns/Num 估计 q(s,a)
      - π(·|s) ← ε-greedy(q, s)
    未传 policy 时从均匀策略开始。
    返回 (V, policy)，V(s) = Σ_a π(a|s) q(s,a)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    s = (0,0)
    for _ in range(num_episodes):
        for _ in range(max_steps):
            probs = policy.get(s)
            a = sample_action(probs)
            next_pos, r, terminated = env.transition(s, a)
            next_probs = policy.get(next_pos)
            next_a = sample_action(next_probs)
            q[(s,a)] = q.get((s,a),0) - 0.001 * (q.get((s,a),0) - (r + gamma * q.get((next_pos,next_a),0)))
            policy[s] = epsilon_greedy_probs(q,s,n_actions,epsilon)
            if terminated:
                break
            s = next_pos

    n = env.config.rows
    v = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            st = (i, j)
            probs = policy.get(st, [1.0 / n_actions] * n_actions)
            v[i][j] = sum(probs[a] * q.get((st, a), 0.0) for a in range(n_actions))
    return v, policy



def qleaning_onpolicy(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    MC ε-Greedy（Sutton & Barto Algorithm 5.3）:
      - episode 按当前 π(·|s) 采样动作（概率探索）
      - 用 Returns/Num 估计 q(s,a)
      - π(·|s) ← ε-greedy(q, s)
    未传 policy 时从均匀策略开始。
    返回 (V, policy)，V(s) = Σ_a π(a|s) q(s,a)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    s = (0,0)
    for _ in range(num_episodes):
        for _ in range(max_steps):
            probs = policy.get(s)
            a = sample_action(probs)
            next_pos, r, terminated = env.transition(s, a)
            next_probs = policy.get(next_pos)
            next_a = sample_action(next_probs)
            best = float('-inf')
            best_act = None
            for a_ in range(n_actions):
                if q.get((next_pos,a_),float('-inf')) > best:
                    best = q.get((next_pos,a_))
                    best_act = a_
            q[(s,a)] = q.get((s,a),0) - 0.001 * (q.get((s,a),0) - (r + gamma * q.get((next_pos,best_act),0)))
            policy[s] = epsilon_greedy_probs(q,s,n_actions,epsilon)
            if terminated:
                break
            s = next_pos

    n = env.config.rows
    v = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            st = (i, j)
            probs = policy.get(st, [1.0 / n_actions] * n_actions)
            v[i][j] = sum(probs[a] * q.get((st, a), 0.0) for a in range(n_actions))
    return v, policy



def qleaning_offpolicy(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    MC ε-Greedy（Sutton & Barto Algorithm 5.3）:
      - episode 按当前 π(·|s) 采样动作（概率探索）
      - 用 Returns/Num 估计 q(s,a)
      - π(·|s) ← ε-greedy(q, s)
    未传 policy 时从均匀策略开始。
    返回 (V, policy)，V(s) = Σ_a π(a|s) q(s,a)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    policy_new = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    s = (0,0)
    for _ in range(num_episodes):
        for _ in range(max_steps):
            probs = policy.get(s)
            a = sample_action(probs)
            next_pos, r, terminated = env.transition(s, a)
            next_probs = policy.get(next_pos)
            next_a = sample_action(next_probs)
            best = float('-inf')
            best_act = None
            for a_ in range(n_actions):
                if q.get((next_pos,a_),float('-inf')) > best:
                    best = q.get((next_pos,a_))
                    best_act = a_
            q[(s,a)] = q.get((s,a),0) - 0.001 * (q.get((s,a),0) - (r + gamma * q.get((next_pos,best_act),0)))
            policy_new[s] = epsilon_greedy_probs(q,s,n_actions,0)
            if terminated:
                break
            s = next_pos

    n = env.config.rows
    v = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            st = (i, j)
            probs = policy_new.get(st, [1.0 / n_actions] * n_actions)
            v[i][j] = sum(probs[a] * q.get((st, a), 0.0) for a in range(n_actions))
    return v, policy_new




def qleaning_offpolicy(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    MC ε-Greedy（Sutton & Barto Algorithm 5.3）:
      - episode 按当前 π(·|s) 采样动作（概率探索）
      - 用 Returns/Num 估计 q(s,a)
      - π(·|s) ← ε-greedy(q, s)
    未传 policy 时从均匀策略开始。
    返回 (V, policy)，V(s) = Σ_a π(a|s) q(s,a)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    gamma = env.config.gamma if gamma is None else gamma
    policy = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    policy_new = copy.deepcopy(
        policy if policy is not None else uniform_policy(env)
    )
    n_actions = env.action_space.n
    q: dict[tuple[tuple[int, int], int], float] = {}
    s = (0,0)
    for _ in range(num_episodes):
        for _ in range(max_steps):
            probs = policy.get(s)
            a = sample_action(probs)
            next_pos, r, terminated = env.transition(s, a)
            next_probs = policy.get(next_pos)
            next_a = sample_action(next_probs)
            best = float('-inf')
            best_act = None
            for a_ in range(n_actions):
                if q.get((next_pos,a_),float('-inf')) > best:
                    best = q.get((next_pos,a_))
                    best_act = a_
            q[(s,a)] = q.get((s,a),0) - 0.001 * (q.get((s,a),0) - (r + gamma * q.get((next_pos,best_act),0)))
            policy_new[s] = epsilon_greedy_probs(q,s,n_actions,0)
            if terminated:
                break
            s = next_pos

    n = env.config.rows
    v = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            st = (i, j)
            probs = policy_new.get(st, [1.0 / n_actions] * n_actions)
            v[i][j] = sum(probs[a] * q.get((st, a), 0.0) for a in range(n_actions))
    return v, policy_new




import torch
from torch import nn
from torch.functional import F

class PolicyNet(nn.Module):

    def __init__(self,state_dim,action_dim):
        super().__init__()
        layers = [nn.Linear(state_dim,100),nn.ReLU(),nn.Linear(100,action_dim)]
        self.mlp = nn.Sequential(*layers)
    
    # def sample_action(self,state):
    #     with torch.no_grad():
    #         probs = self.forward(state) # action_dim
    #         return sample_action(probs.tolist())

    def forward(self,state):
        ## state state_dim
        logits = self.mlp(state) # action_dim
        probs = F.softmax(logits,dim=-1)
        return probs




def calc_rewards(arr,gamma):
    t = 0
    ret = []
    for r in reversed(arr):
        t = t * gamma + r
        ret.append(t)
    ret.reverse()
    return ret


from torch.optim import Adam

def policy_gradient(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    MC ε-Greedy（Sutton & Barto Algorithm 5.3）:
      - episode 按当前 π(·|s) 采样动作（概率探索）
      - 用 Returns/Num 估计 q(s,a)
      - π(·|s) ← ε-greedy(q, s)
    未传 policy 时从均匀策略开始。
    返回 (V, policy)，V(s) = Σ_a π(a|s) q(s,a)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    n_actions = env.action_space.n
    n = env.config.rows
    policy1 = PolicyNet(n*n,n_actions)
    optim = Adam(policy1.parameters(),lr=0.003)
    goal = env.config.goal_pos
    baseline_sum: dict[tuple[int, int], float] = defaultdict(float)
    baseline_count: dict[tuple[int, int], int] = defaultdict(int)

    gamma = env.config.gamma if gamma is None else gamma
    def make_state(s):
        vec = [0.0] * (n*n)
        index = s[0] * n  + s[1]
        vec[index] = 1.0
        return torch.tensor(vec)
    

    for ep in range(num_episodes):
        s = (0, 0)  # 与 eval 同起点，否则很难学到从 (0,0) 出发
        traj = []
        rws = []
        for _ in range(max_steps):
            state = make_state(s)
            probs = policy1(state)
            a = sample_action(probs.tolist())
            next_pos, r, terminated = env.transition(s, a)
            traj.append((s,a,probs[a],r,next_pos))
            rws.append(r)
            if terminated or next_pos == goal:
                break
            s = next_pos
        
        returns = calc_rewards(rws, gamma)

        loss = 0
        for i in range(len(traj)):
            s,a,prob,_,next_pos = traj[i]
            g = returns[i]
            count = baseline_count[s]
            baseline = baseline_sum[s] / count if count > 0 else 0.0
            advantage = g - baseline
            loss += (-prob.log() * advantage)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        for i in range(len(traj)):
            s, _, _, _, _ = traj[i]
            baseline_sum[s] += returns[i]
            baseline_count[s] += 1

        ## eval
        if ep % 100 == 0:
            rs = 0
            count = 0
            s = (0, 0)
            path_steps: list[str] = [f"{s}"]
            for _ in range(max_steps):
                state = make_state(s)
                probs = policy1(state)
                # a = sample_action(probs.tolist())
                a = torch.argmax(probs, dim=-1).item()
                next_pos, r, terminated = env.transition(s, a)
                rs += r
                count += 1
                path_steps.append(
                    f"{action_name(env, a)}->{next_pos}(r={r:g})"
                )
                if terminated or next_pos == goal:
                    break
                s = next_pos
            shown = path_steps[:10]
            tail = "..." if len(path_steps) > 10 else ""
            print("eval", ep, f"G_0={rs:.3f}", "->".join(shown) + tail)
    

    def mc_return_from(s_start: tuple[int, int]) -> float:
        """从 s_start 出发按 π 采样一条轨迹，返回 G_0。"""
        s = s_start
        rewards: list[float] = []
        with torch.no_grad():
            for _ in range(max_steps):
                probs = policy1(make_state(s))
                a = sample_action(probs.tolist())
                next_pos, r, terminated = env.transition(s, a)
                rewards.append(r)
                if terminated:
                    break
                s = next_pos
        if not rewards:
            return 0.0
        return calc_rewards(rewards, gamma)[0]

    value_rollouts = 100
    policy1.eval()
    n = env.config.rows
    v = [[0.0] * n for _ in range(n)]
    policy_table: Policy = {}
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                s = (i, j)
                v[i][j] = sum(
                    mc_return_from(s) for _ in range(value_rollouts)
                ) / value_rollouts
                policy_table[s] = policy1(make_state(s)).tolist()
    return v, policy_table

            


class CriticNet(nn.Module):

    def __init__(self,state_dim,action_dim):
        super().__init__()
        layers = [nn.Linear(state_dim+action_dim,100),nn.ReLU(),nn.Linear(100,1)]
        self.mlp = nn.Sequential(*layers)
    

    def forward(self,state,action):
        ## state state_dim
        merge = torch.concat([state,action],dim=-1)
        values = self.mlp(merge) # action_dim
        return values



def qac(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    MC ε-Greedy（Sutton & Barto Algorithm 5.3）:
      - episode 按当前 π(·|s) 采样动作（概率探索）
      - 用 Returns/Num 估计 q(s,a)
      - π(·|s) ← ε-greedy(q, s)
    未传 policy 时从均匀策略开始。
    返回 (V, policy)，V(s) = Σ_a π(a|s) q(s,a)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    n_actions = env.action_space.n
    n = env.config.rows
    policy1 = PolicyNet(n*n,n_actions)
    critic = CriticNet(n*n,n_actions)
    optim = Adam(policy1.parameters(),lr=0.003)
    critic_optim = Adam(critic.parameters(),lr=0.003)
    goal = env.config.goal_pos

    gamma = env.config.gamma if gamma is None else gamma
    def make_state(s):
        vec = [0.0] * (n*n)
        index = s[0] * n  + s[1]
        vec[index] = 1.0
        return torch.tensor(vec)
    
    def make_action(action):
        vec = [0.0] * n_actions
        vec[action] = 1.0
        return torch.tensor(vec)
    

    for ep in range(num_episodes):
        s = (random.randint(0,n-1), random.randint(0,n-1))  # 与 eval 同起点，否则很难学到从 (0,0) 出发
        state = make_state(s)
        probs = policy1(state)
        a = sample_action(probs.tolist())
        next_pos, r, terminated = env.transition(s, a)
        next_pos = make_state(next_pos)
        next_probs = policy1(next_pos)
        next_action = sample_action(next_probs.tolist())
        next_action = make_action(next_action)
        with torch.no_grad():
            qsa = critic(state,make_action(a))
            td = qsa - (r + gamma * critic(next_pos,next_action))
        
        loss = -qsa * probs[a].log()
        optim.zero_grad()
        loss.backward()
        optim.step()

        critic_loss = td * critic(state,make_action(a))
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

  

        ## eval
        if ep % 100 == 0:
            print('loss',loss,critic_loss)
            rs = 0
            count = 0
            s = (0, 0)
            path_steps: list[str] = [f"{s}"]
            for _ in range(max_steps):
                state = make_state(s)
                probs = policy1(state)
                # a = sample_action(probs.tolist())
                a = torch.argmax(probs, dim=-1).item()
                next_pos, r, terminated = env.transition(s, a)
                rs += r
                count += 1
                path_steps.append(
                    f"{action_name(env, a)}->{next_pos}(r={r:g})"
                )
                if terminated or next_pos == goal:
                    break
                s = next_pos
            shown = path_steps[:10]
            tail = "..." if len(path_steps) > 10 else ""
            print("eval", ep, f"G_0={rs:.3f}", "->".join(shown) + tail)
    

    policy1.eval()
    critic.eval()
    v = [[0.0] * n for _ in range(n)]
    policy_table: Policy = {}
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                s = (i, j)
                state = make_state(s)
                probs = policy1(state)
                v[i][j] = sum(
                    probs[a].item() * critic(state, make_action(a)).item()
                    for a in range(n_actions)
                )
                policy_table[s] = probs.tolist()
    return v, policy_table




class ValueNet(nn.Module):

    def __init__(self,state_dim):
        super().__init__()
        layers = [nn.Linear(state_dim,100),nn.ReLU(),nn.Linear(100,1)]
        self.mlp = nn.Sequential(*layers)
    

    def forward(self,state):
        ## state state_dim
        values = self.mlp(state) # action_dim
        return values


def a2c(
    env: GridWorldEnv,
    policy: Policy | None = None,
    *,
    epsilon: float = 0.1,
    gamma: float | None = None,
    max_steps: int = 100,
    num_episodes: int = 50_000,
) -> tuple[list[list[float]], Policy]:
    """
    MC ε-Greedy（Sutton & Barto Algorithm 5.3）:
      - episode 按当前 π(·|s) 采样动作（概率探索）
      - 用 Returns/Num 估计 q(s,a)
      - π(·|s) ← ε-greedy(q, s)
    未传 policy 时从均匀策略开始。
    返回 (V, policy)，V(s) = Σ_a π(a|s) q(s,a)。
    """
    if not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be in (0, 1]")

    gamma = env.config.gamma if gamma is None else gamma
    n_actions = env.action_space.n
    n = env.config.rows
    policy1 = PolicyNet(n*n,n_actions)
    critic = ValueNet(n*n)
    optim = Adam(policy1.parameters(),lr=0.003)
    critic_optim = Adam(critic.parameters(),lr=0.003)
    goal = env.config.goal_pos

    gamma = env.config.gamma if gamma is None else gamma
    def make_state(s):
        vec = [0.0] * (n*n)
        index = s[0] * n  + s[1]
        vec[index] = 1.0
        return torch.tensor(vec)
    

    for ep in range(num_episodes):
        s = (random.randint(0,n-1), random.randint(0,n-1))  # 与 eval 同起点，否则很难学到从 (0,0) 出发

        state = make_state(s)
        probs = policy1(state)
        a = sample_action(probs.tolist())
        next_pos, r, terminated = env.transition(s, a)
        next_pos1 = make_state(next_pos)

        with torch.no_grad():
            v = critic(state)
            td = v - (r + gamma * critic(next_pos1))
        
        loss = td * probs[a].log()
        optim.zero_grad()
        loss.backward()
        optim.step()

        critic_loss = td * critic(state)
        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        s = next_pos



  

        ## eval
        if ep % 100 == 0:
            # print('loss',loss,critic_loss)
            rs = 0
            count = 0
            s = (0, 0)
            path_steps: list[str] = [f"{s}"]
            for _ in range(max_steps):
                state = make_state(s)
                probs = policy1(state)
                # a = sample_action(probs.tolist())
                a = torch.argmax(probs, dim=-1).item()
                next_pos, r, terminated = env.transition(s, a)
                rs += r
                count += 1
                path_steps.append(
                    f"{action_name(env, a)}->{next_pos}(r={r:g})"
                )
                if terminated or next_pos == goal:
                    break
                s = next_pos
            shown = path_steps[:10]
            tail = "..." if len(path_steps) > 10 else ""
            print("eval", ep, f"G_0={rs:.3f}", "->".join(shown) + tail)
    

    policy1.eval()
    critic.eval()
    v = [[0.0] * n for _ in range(n)]
    policy_table: Policy = {}
    with torch.no_grad():
        for i in range(n):
            for j in range(n):
                s = (i, j)
                state = make_state(s)
                probs = policy1(state)
                v[i][j] = critic(state).item()
                policy_table[s] = probs.tolist()
    return v, policy_table


def action_name(env: GridWorldEnv, action: int) -> str:
    return env.action_meanings[action]


def print_values(
    env: GridWorldEnv,
    v: list[list[float]],
    title: str = "",
) -> None:
    if title:
        print(title)
    for i in range(env.config.rows):
        row = []
        for j in range(env.config.cols):
            pos = (i, j)
            tag = ""
            if pos == env.config.goal_pos:
                tag = "G"
            elif pos in env.config.orange_cells:
                tag = "O"
            row.append(f"{v[i][j]:7.2f}{tag}")
        print("  " + " ".join(row))


def print_policy(env: GridWorldEnv, policy: Policy) -> None:
    """打印 argmax π(·|s)（确定性时即唯一动作）。"""
    print("策略 pi（箭头，argmax π）:")
    for i in range(env.config.rows):
        row = []
        for j in range(env.config.cols):
            if (i, j) == env.config.goal_pos:
                row.append("  G ")
            else:
                act = policy_action_at(policy, (i, j))
                row.append(f"{action_name(env, act):>3}" if act is not None else "  ?")
        print("  " + " ".join(row))


if __name__ == "__main__":
    env = make_env()
    policy = uniform_policy(env)

    print("=== 初始策略（均匀）===")
    print_policy(env, policy)

    v, policy = a2c(env, policy, epsilon=0.1, num_episodes=90000)
    print_values(env, v, title="\nV（MC 估计 V^π，按 π 采样 rollout）:")
    print_policy(env, policy)
