# Grid RL Environment

这是一个最小可用的 `Gymnasium` 风格 grid world，适合拿来自己写：

- state value / value iteration
- policy evaluation
- Q-learning
- 后续再扩展到 DQN / PPO

当前环境对应你给的 5x5 图，规则是：

- 蓝色格子是目标 `goal`
- 橙色格子可以进入，但进入时 reward = `-1`
- 撞边界 reward = `-1`，并留在原地
- 进入目标 reward = `+1`
- 其他转移 reward = `0`
- 默认是 continuing task：到了目标以后还能继续移动，不会自动终止
- discount factor 默认用 `gamma = 0.9`

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 你现在直接能用的文件

```bash
grid_env.py
value_iteration_starter.py
policy_evaluation_starter.py
train_q_learning.py
```

说明：

- [grid_env.py](/Users/liuchu/codes/rl-base/grid_env.py)：环境本身，已经可直接调用
- [value_iteration_starter.py](/Users/liuchu/codes/rl-base/value_iteration_starter.py)：你自己填写最优 state value 的框架
- [policy_evaluation_starter.py](/Users/liuchu/codes/rl-base/policy_evaluation_starter.py)：你自己填写固定策略 state value 的框架
- [train_q_learning.py](/Users/liuchu/codes/rl-base/train_q_learning.py)：保留的 Q-learning 示例

## 环境接口

你自己算 state value 时，主要用这几个接口：

```python
env.states()                # 所有状态，返回 [(0,0), ...]
env.actions()               # 所有动作，返回 [0,1,2,3]
env.transition(state, a)    # 返回 (next_state, reward, terminated)
env.is_terminal(state)      # 默认恒为 False，除非你把 terminal_goal=True
```

动作编码：

```python
0 = up
1 = right
2 = down
3 = left
```

## 运行方式

等你把 `TODO` 填完以后，可以直接运行：

```bash
python3 value_iteration_starter.py
python3 policy_evaluation_starter.py
```

## 建议

1. 先用全 0 初始化 `V(s)`
2. 先写 policy evaluation，再写 value iteration
3. 先确认 terminal state 的处理约定，再写 Bellman 更新
4. 如果算出来不对，先打印单个状态的 4 个 `Q(s,a)` 检查

如果你想切回 episodic 版本，可以这样初始化：

```python
from grid_env import GridConfig, GridWorldEnv

env = GridWorldEnv(config=GridConfig(terminal_goal=True))
```
