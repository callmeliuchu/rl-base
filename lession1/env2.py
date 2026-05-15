"""
环境 2：5x5 网格（行/列从 1 开始编号时，目标在 (4,3)）
坐标用 0-index：s = [row, col]，row=0 为最上一行。
动作：与 test.py 相同 — [0,1]右 [0,-1]左 [1,0]下 [-1,0]上 [0,0]停
"""

import numpy as np

ACTIONS = [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]
# 方向别名（对应图上箭头）
RIGHT, LEFT, DOWN, UP, STAY = [0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]

GAMMA = 0.9
N = 5

# 1-index (4,3) -> 0-index [3, 2]
GOAL = [3, 2]

# 橙色格（1-index 见课件图）
ORANGE = {
    tuple(x)
    for x in [
        [1, 1],
        [1, 2],  # (2,2)(2,3)
        [2, 2],  # (3,3)
        [3, 1],
        [3, 3],  # (4,2)(4,4)
        [4, 1],  # (5,2)
    ]
}

# 图上绿色箭头：每格确定性动作（0-index）
POLICY_ACTION = {
    (0, 0): RIGHT,
    (0, 1): RIGHT,
    (0, 2): RIGHT,
    (0, 3): DOWN,
    (0, 4): DOWN,
    (1, 0): UP,
    (1, 1): UP,
    (1, 2): RIGHT,
    (1, 3): DOWN,
    (1, 4): DOWN,
    (2, 0): UP,
    (2, 1): LEFT,
    (2, 2): DOWN,
    (2, 3): RIGHT,
    (2, 4): DOWN,
    (3, 0): UP,
    (3, 1): RIGHT,
    (3, 2): STAY,  # 目标格
    (3, 3): LEFT,
    (3, 4): DOWN,
    (4, 0): UP,
    (4, 1): RIGHT,
    (4, 2): UP,
    (4, 3): LEFT,
    (4, 4): LEFT,
}


def in_bounds(i, j):
    return 0 <= i < N and 0 <= j < N


def reward(s, a):
    # a == RIGHT, LEFT, DOWN, UP, STAY
    next_s = [s[0] + a[0], s[1] + a[1]]
    if not in_bounds(next_s[0], next_s[1]):
        return -1, 1, s  # 撞墙：惩罚，留在原状态
    if next_s == GOAL:
        return 1, 1, next_s
    if tuple(next_s) in ORANGE:
        return -1, 1, next_s
    return 0, 1, next_s


def pi(s, a):
    chosen = POLICY_ACTION.get(tuple(s))
    if chosen is None:
        return 0.0
    return 1.0 if a == chosen else 0.0


def state_index(i, j):
    return i * N + j


def all_states():
    return [[i, j] for i in range(N) for j in range(N)]


def calc():
    """解析解：V^pi = (I - gamma * P)^-1 R（与 test.py 中 calc 相同思路）"""
    rewards = []
    for i in range(N):
        for j in range(N):
            total_r = 0.0
            for action in ACTIONS:
                prob = pi([i, j], action)
                if prob == 0:
                    continue
                r, _, _ = reward([i, j], action)
                total_r += r * prob
            rewards.append(total_r)

    size = N * N
    matrix = [[0.0] * size for _ in range(size)]
    for i in range(N):
        for j in range(N):
            for action in ACTIONS:
                prob = pi([i, j], action)
                if prob == 0:
                    continue
                _, _, next_s = reward([i, j], action)
                ni, nj = next_s
                if in_bounds(ni, nj):
                    idx1 = state_index(i, j)
                    idx2 = state_index(ni, nj)
                    matrix[idx1][idx2] += prob

    rewards_arr = np.array(rewards)
    p_mat = np.array(matrix)
    v_vec = np.linalg.inv(np.eye(size) - GAMMA * p_mat) @ rewards_arr
    return v_vec.reshape(N, N), p_mat, rewards_arr


def evaluate_policy(gamma=GAMMA, theta=1e-8):
    v = [[0.0] * N for _ in range(N)]
    while True:
        delta = 0.0
        v_new = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
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
    q = {}
    for i in range(N):
        for j in range(N):
            s = [i, j]
            chosen = POLICY_ACTION.get((i, j))
            if chosen is None:
                continue
            r, _, next_s = reward(s, chosen)
            q[(i, j)] = r + gamma * v[next_s[0]][next_s[1]]
    return q


def value_iteration(gamma=GAMMA, theta=1e-8):
    v = [[0.0] * N for _ in range(N)]
    while True:
        delta = 0.0
        v_new = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
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
    policy = {}
    for i in range(N):
        for j in range(N):
            s = [i, j]
            best_a, best_q = None, float("-inf")
            for a in ACTIONS:
                r, _, next_s = reward(s, a)
                q = r + gamma * v[next_s[0]][next_s[1]]
                if q > best_q:
                    best_q, best_a = q, a
            policy[(i, j)] = best_a
    return policy


def action_name(a):
    names = {tuple(STAY): "停", tuple(UP): "上", tuple(DOWN): "下", tuple(LEFT): "左", tuple(RIGHT): "右"}
    return names.get(tuple(a), str(a))


def print_grid_values(v, title=""):
    if title:
        print(title)
    for i in range(N):
        row = []
        for j in range(N):
            tag = ""
            if [i, j] == GOAL:
                tag = "G"
            elif (i, j) in ORANGE:
                tag = "O"
            row.append(f"{v[i][j]:7.2f}{tag}")
        print("  " + " ".join(row))


def print_policy():
    print("策略 pi（箭头）:")
    for i in range(N):
        row = []
        for j in range(N):
            if [i, j] == GOAL:
                row.append("  G ")
            else:
                row.append(f"{action_name(POLICY_ACTION[(i, j)]):>3}")
        print("  " + " ".join(row))


if __name__ == "__main__":
    print("=== 环境2：5x5，目标(4,3)，橙色=惩罚 -1 ===\n")
    print_policy()

    v_exact, _, _ = calc()
    print_grid_values(v_exact, "\nV^pi（矩阵解析解）:")

    v_iter = evaluate_policy()
    print_grid_values(v_iter, "\nV^pi（迭代策略评估）:")

    max_diff = max(abs(v_exact[i][j] - v_iter[i][j]) for i in range(N) for j in range(N))
    print(f"\n两种方法最大误差: {max_diff:.2e}")

    v_star = value_iteration()
    print_grid_values(v_star, "\nV*（值迭代，无 pi）:")
