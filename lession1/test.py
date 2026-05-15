from math import gamma
from numpy import matrix


ACTIONS = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]
GAMMA = 0.9


def reward(s, a):
    # a == [0,1],[0,-1],[1,0],[-1,0]
    # s [0,0],[0,1],[1,0],[1,1]
    next_s = [s[0] + a[0], s[1] + a[1]]
    for o in next_s:
        if o not in [0, 1]:
            return -1, 1, s  # 撞墙：惩罚，留在原状态
    if next_s == [1, 1]:
        return 1, 1, next_s
    if next_s == [0, 1]:
        return -1, 1, next_s
    return 0, 1, next_s


def pi(s, a):
    if s == [0, 0]:
        if a == [0, 1] or a == [1, 0]:
            return 0.5
        return 0

    if s == [0, 1]:
        if a == [1, 0]:
            return 1
        return 0

    if s == [1, 0]:
        if a == [0, 1]:
            return 1
        return 0

    if s == [1, 1]:
        if a == [0, 0]:
            return 1
        return 0

    return 0

def calc():
    rewards = []
    for i in range(2):
        for j in range(2):
            total_r = 0
            for action in ACTIONS:
                prob = pi([i,j],action)
                if prob == 0:
                    continue
                r,_,_ = reward([i,j],action)
                total_r += r * prob
            rewards.append(total_r)

    ### expectation : rewards
    #   p(s'|s)
    matrix = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]

    for i in range(2):
        for j in range(2):
            for action in ACTIONS:
                prob = pi([i,j],action)
                next_ij = [i+action[0],j+action[1]]
                if next_ij[0] in [0,1] and next_ij[1] in [0,1]:
                    index1 = i * 2 + j
                    index2 = next_ij[0] * 2 + next_ij[1]
                    matrix[index1][index2] = prob
    import numpy as np

    rewards = np.array(rewards)
    matrix = np.array(matrix)
    print(matrix,rewards)
    ans = np.linalg.inv(np.eye(4,4) - GAMMA * matrix) @ rewards
    print(ans)


def evaluate_policy(gamma=GAMMA, theta=1e-8):
    """迭代策略评估：V^pi(s) = sum_a pi(a|s) * (R + gamma * V^pi(s'))"""
    v = [[0.0, 0.0], [0.0, 0.0]]
    while True:
        delta = 0.0
        v_new = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                s = [i, j]
                total = 0.0
                for a in ACTIONS:
                    prob = pi(s, a)
                    if prob == 0:
                        continue
                    r, _, next_s = reward(s, a)
                    ni, nj = next_s
                    total += prob * (r + gamma * v[ni][nj])
                v_new[i][j] = total
                delta = max(delta, abs(v_new[i][j] - v[i][j]))
        v = v_new
        if delta < theta:
            break
    return v


def compute_q(v, gamma=GAMMA):
    """Q^pi(s,a) = R(s,a) + gamma * V^pi(s')（确定性转移）"""
    q = {}
    for i in range(2):
        for j in range(2):
            s = [i, j]
            for a in ACTIONS:
                if pi(s, a) == 0:
                    continue
                r, _, next_s = reward(s, a)
                ni, nj = next_s
                q[(tuple(s), tuple(a))] = r + gamma * v[ni][nj]
    return q


STATES = [[0, 0], [0, 1], [1, 0], [1, 1]]


def make_deterministic_policy(action_per_state):
    """action_per_state: {(0,0): [1,0], ...} -> pi(a|s) in {0,1}"""

    def policy(s, a):
        chosen = action_per_state[tuple(s)]
        return 1.0 if a == chosen else 0.0

    return policy


def evaluate_policy_fn(policy_fn, gamma=GAMMA, theta=1e-8):
    v = [[0.0, 0.0], [0.0, 0.0]]
    while True:
        delta = 0.0
        v_new = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                s = [i, j]
                total = 0.0
                for a in ACTIONS:
                    prob = policy_fn(s, a)
                    if prob == 0:
                        continue
                    r, _, next_s = reward(s, a)
                    ni, nj = next_s
                    total += prob * (r + gamma * v[ni][nj])
                v_new[i][j] = total
                delta = max(delta, abs(v_new[i][j] - v[i][j]))
        v = v_new
        if delta < theta:
            break
    return v


def brute_force_trajectory(s, depth, policy_fn, gamma=GAMMA):
    """
    暴力展开轨迹树（深度 depth）：
    对每个动作分支算 r + γ * 子树，按 policy_fn(a|s) 加权。
    depth→∞ 时趋近 V^pi(s)，但深度一大分支数爆炸。
    """
    if depth == 0:
        return 0.0
    total = 0.0
    for a in ACTIONS:
        prob = policy_fn(s, a)
        if prob == 0:
            continue
        r, _, next_s = reward(s, a)
        total += prob * (r + gamma * brute_force_trajectory(next_s, depth - 1, policy_fn, gamma))
    return total


def brute_force_trajectory_optimal(s, depth, gamma=GAMMA):
    """暴力：每步取最优动作（有限深度下的最优决策树）。"""
    if depth == 0:
        return 0.0
    best = float("-inf")
    for a in ACTIONS:
        r, _, next_s = reward(s, a)
        best = max(best, r + gamma * brute_force_trajectory_optimal(next_s, depth - 1, gamma))
    return best


def _dominates(v, u):
    """v 在每个状态上都不差于 u，且至少一个状态严格更好。"""
    not_worse = all(v[i][j] >= u[i][j] for i in range(2) for j in range(2))
    strictly_better = any(v[i][j] > u[i][j] for i in range(2) for j in range(2))
    return not_worse and strictly_better


def brute_force_enumerate_policies(gamma=GAMMA):
    """
    暴力枚举所有确定性策略（每状态 5 个动作 -> 5^4=625 种），
    对每种策略做策略评估，保留 Pareto 最优（分量意义下的 V 最大）。
    状态空间小时可行；大 MDP 不可行。
    """
    from itertools import product

    best_v = [[float("-inf"), float("-inf")], [float("-inf"), float("-inf")]]
    best_policy = None
    for choices in product(ACTIONS, repeat=len(STATES)):
        action_map = {tuple(STATES[i]): choices[i] for i in range(len(STATES))}
        policy_fn = make_deterministic_policy(action_map)
        v = evaluate_policy_fn(policy_fn, gamma=gamma)
        if _dominates(v, best_v):
            best_v = v
            best_policy = action_map
    return best_v, best_policy


def value_iteration(gamma=GAMMA, theta=1e-8):
    """
    无 pi 时求 V*：Bellman 最优备份
    V*(s) <- max_a [ R(s,a) + gamma * V*(s') ]
    只需 reward / 转移，不需要预设策略。
    """
    v = [[0.0, 0.0], [0.0, 0.0]]
    while True:
        delta = 0.0
        v_new = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            for j in range(2):
                s = [i, j]
                best = float("-inf")
                for a in ACTIONS:
                    r, _, next_s = reward(s, a)
                    ni, nj = next_s
                    best = max(best, r + gamma * v[ni][nj])
                v_new[i][j] = best
                delta = max(delta, abs(v_new[i][j] - v[i][j]))
        v = v_new
        if delta < theta:
            break
    return v


def greedy_policy_from_v(v, gamma=GAMMA):
    """V* 收敛后：pi*(s) = argmax_a [ R + gamma * V*(s') ]"""
    policy = {}
    for i in range(2):
        for j in range(2):
            s = [i, j]
            best_a, best_q = None, float("-inf")
            for a in ACTIONS:
                r, _, next_s = reward(s, a)
                q = r + gamma * v[next_s[0]][next_s[1]]
                if q > best_q:
                    best_q, best_a = q, a
            policy[tuple(s)] = best_a
    return policy


def solve_optimal(gamma=GAMMA, theta=1e-8):
    """无 pi：值迭代得 V*，再贪心提取 pi*。"""
    v_star = value_iteration(gamma, theta)
    pi_star = greedy_policy_from_v(v_star, gamma)
    return v_star, pi_star


if __name__ == "__main__":
    calc()

    v = evaluate_policy()
    q = compute_q(v)

    print("=== 策略评估 (Bellman 迭代) ===")
    print("V^pi:")
    for row in v:
        print("  ", [round(x, 4) for x in row])

    print("\nQ^pi (pi(a|s) > 0):")
    for (s, a), val in sorted(q.items()):
        print(f"  s={list(s)} a={list(a)}  Q={round(val, 4)}")

    print("\n=== 暴力 1：展开轨迹树 (固定 pi，分支数 ~5^depth) ===")
    for depth in [3, 5, 8, 10]:
        bf = brute_force_trajectory([0, 0], depth, pi)
        print(f"  depth={depth:2d}  V^pi(0,0) ≈ {bf:.4f}  (真值 {v[0][0]:.4f})")

    print("\n=== 暴力 2：枚举全部确定性策略 (625 种) ===")
    v_star_bf, pi_star = brute_force_enumerate_policies()
    print("V* (暴力枚举策略):")
    for row in v_star_bf:
        print("  ", [round(x, 4) for x in row])
    print("最优确定性策略 pi*(s):")
    for s, a in sorted(pi_star.items()):
        print(f"  s={list(s)} -> a={a}")

    v_star, pi_star_vi = solve_optimal()
    print("\n=== 无 pi：值迭代 -> V* 与 pi* ===")
    print("V*:")
    for row in v_star:
        print("  ", [round(x, 4) for x in row])
    print("pi*(贪心):")
    for s, a in sorted(pi_star_vi.items()):
        print(f"  s={list(s)} -> a={a}")

    print("\n=== 暴力 3：有限深度最优决策树 (非无穷 horizon) ===")
    for depth in [3, 5, 8, 10]:
        bf = brute_force_trajectory_optimal([0, 0], depth)
        print(f"  depth={depth:2d}  V~*(0,0) ≈ {bf:.4f}  (V*(0,0)={v_star[0][0]:.4f})")
