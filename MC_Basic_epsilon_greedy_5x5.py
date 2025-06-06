import numpy as np
from tqdm import tqdm

R = 5
C = 5
A = 5

N = R * C

epsilon = 0.1
# epsilon = 0.0

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
pi = np.zeros((N, A))
for r_ in range(R):
    for c_ in range(C):
        pi[r_ * C + c_] = epsilon / A
        pi[r_ * C + c_][action_grid[r_][c_]] = 1 - epsilon / A * (A - 1)
        # pi[r_ * C + c_] = 1 / A
        # pi[r_ * C + c_][action_grid[r_][c_]] = 1

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
E = 20000

q_table = np.zeros((N, A))

num_episode = 200

def random_select():
    r_start = np.random.randint(0, R)
    c_start = np.random.randint(0, C)
    a_start = np.random.randint(0, A)
    return r_start, c_start, a_start

# np.random.seed(42)
for episode in tqdm(range(num_episode)):
    # 1. Random start.
    r_start, c_start, a_start = random_select()
    state = (r_start, c_start)

    # 2. Generate a Episode.
    episode_history = []
    for e in range(E):
        r_, c_ = state
        a = a_start
        if e != 0:
            # Method 1. Wrong code for random choise based on pi[r_ * C + c_]
            # rn = np.random.random()
            # accum = 0.
            # for a in range(A):
            #     accum += pi[r_ * C + c_][a]
            #     if accum >= rn:
            #         break
            # Method 2. Too slow.
            a = np.random.choice(range(A), size=1, replace=False, p=pi[r_ * C + c_])[0]
            # Method 3. Same as Method 1.
            # cumsum = np.cumsum(pi[r_ * C + c_])
            # rn = np.random.random()
            # rn *= cumsum[-1]
            # a = np.searchsorted(cumsum, rn, side='right')
            # Method 4. Also wrong...
            # prob = pi[r_ * C + c_]
            # prob_sum = np.sum(prob)
            # if not np.isclose(prob_sum, 1.0):
            #     prob /= prob_sum
            # rn = np.random.random()
            # accum = 0.0
            # for a in range(A):
            #     accum += prob[a]
            #     if accum >= rn:
            #         break

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

    # ATTENTION: Initialize Return and Num every episode.
    Return = np.zeros((N, A))
    Num = np.zeros((N, A))
    g = 0
    for e in reversed(range(len(episode_history))):
        (r_, c_), a, reward = episode_history[e]
        g = gamma * g + reward
        # 3. Update Return and Num.
        Return[r_ * C + c_][a] += g
        Num[r_ * C + c_][a] += 1
        # 4. Policy evaluation.
        q_table[r_ * C + c_][a] = Return[r_ * C + c_][a] / Num[r_ * C + c_][a]
        # 5. Policy improvement.
        max_a = np.argmax(q_table[r_ * C + c_])
        pi[r_ * C + c_] = epsilon / A
        pi[r_ * C + c_][max_a] = 1 - epsilon / A * (A - 1)
    # ATTENTION: Decrease epsilon.
    if epsilon > 0.001:
        epsilon -= 0.001

action_grid = np.zeros((R, C))
for r_ in range(R):
    for c_ in range(C):
        a = np.argmax(pi[r_ * C + c_])
        action_grid[r_][c_] = a
print('Conveged Action Grid:\n', action_grid)
