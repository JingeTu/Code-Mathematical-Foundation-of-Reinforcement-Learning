import numpy as np

R = 5
C = 5
A = 5

N = R * C

action_grid = [
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
[1, 1, 1, 1, 1],
]
# action_grid = (np.random.rand(R, C) * A).astype(int)

# Policy Matrix.
# pi[i][j] mean the probability of `at state i choose action j`.
pi = np.zeros((N, 5))
for r_ in range(R):
    for c_ in range(C):
        pi[r_ * C + c_][action_grid[r_][c_]] = 1

# Block states.
blocks = np.array([
[0,  0,  0,  0, 0],
[0, -10, -10,  0, 0],
[0,  0, -10,  0, 0],
[0, -10,  1, -10, 0],
[0, -10,  0,  0, 0],
])

action_to_direction = {}
action_to_direction[0] = np.array([-1, 0])
action_to_direction[1] = np.array([0, 1])
action_to_direction[2] = np.array([1, 0])
action_to_direction[3] = np.array([0, -1])
action_to_direction[4] = np.array([0, 0])

gamma = 0.9

# State Values.
v = np.ones((R * C, 1))

# Episode Length.
E = 20

q_table = np.zeros((N, A))

num_episode = 10000

Return = np.zeros((N, A))
Num = np.zeros((N, A))

def random_select():
    r_start = np.random.randint(0, R)
    c_start = np.random.randint(0, C)
    a_start = np.random.randint(0, A)
    return r_start, c_start, a_start

# np.random.seed(42)
for episode in range(num_episode):
    # 1. Random start.
    r_start, c_start, a_start = random_select()
    state = (r_start, c_start)

    # 2. Generate a Episode.
    episode_history = []
    for e in range(E):
        r_, c_ = state
        a = a_start if e == 0 else np.argmax(pi[r_ * C + c_])
        delta = action_to_direction[a]
        r_p = r_ + delta[0]
        c_p = c_ + delta[1]

        if r_p >= R or r_p < 0 or c_p >= C or c_p < 0:
            reward = -1
            next_state = (r_, c_)
        else:
            reward = blocks[r_p][c_p]
            next_state = (r_p, c_p)

        episode_history.append((state, a, reward))
        state = next_state

    g = 0
    for e in reversed(range(E)):
        (r_, c_), a, reward = episode_history[e]
        # print(f'episode {episode} e {e}', (r_, c_))
        g = gamma * g + reward
        # 3. Update Return and Num.
        Return[r_ * C + c_][a] += g
        Num[r_ * C + c_][a] += 1
        # 4. Policy evaluation.
        q_table[r_ * C + c_][a] = Return[r_ * C + c_][a] / Num[r_ * C + c_][a]
        # 5. Policy improvement.
        max_a = np.argmax(q_table[r_ * C + c_])
        pi[r_ * C + c_] = 0
        pi[r_ * C + c_][max_a] = 1

action_grid = np.zeros((R, C))
for r_ in range(R):
    for c_ in range(C):
        for a in range(A):
            if pi[r_ * C + c_][a] == 1:
                action_grid[r_][c_] = a
print('Conveged Action Grid:\n', action_grid)
