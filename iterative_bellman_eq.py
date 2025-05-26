import numpy as np

R = 5
C = 5
A = 5

N = R * C

# Policy Matrix.
# pi[i][j] mean the probability of `at state i choose action j`.
pi = np.zeros((N, 5))

# For Fig 2.7 (a) 1.
action_grid = [
[1, 1, 1, 2, 2],
[0, 0, 1, 2, 2],
[0, 3, 2, 1, 2],
[0, 1, 4, 3, 2],
[0, 1, 0, 3, 3],
]
# # For Fig 2.7 (a) 2.
# action_grid = [
# [1, 1, 1, 1, 2],
# [0, 0, 1, 1, 2],
# [0, 3, 2, 1, 2],
# [0, 1, 4, 3, 2],
# [0, 1, 0, 3, 3],
# ]
# # For Fig 2.7 (b) 1.
# action_grid = [
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1],
# ]
# # For Fig 2.7 (b) 2.
# action_grid = [
# [1, 3, 3, 0, 0],
# [2, 4, 1, 2, 1],
# [3, 1, 2, 3, 4],
# [4, 2, 0, 0, 1],
# [4, 1, 4, 1, 4],
# ]
for r in range(R):
    for c in range(C):
        pi[r * C + c][action_grid[r][c]] = 1

# Block states.
blocks = np.array([
[0,  0,  0,  0, 0],
[0, -1, -1,  0, 0],
[0,  0, -1,  0, 0],
[0, -1,  1, -1, 0],
[0, -1,  0,  0, 0],
])

action_to_direction = {}
action_to_direction[0] = np.array([-1, 0])
action_to_direction[1] = np.array([0, 1])
action_to_direction[2] = np.array([1, 0])
action_to_direction[3] = np.array([0, -1])
action_to_direction[4] = np.array([0, 0])
# Immediate Reward.
r = np.zeros((R * C, 1))
# State transition Matrix.
P_pi = np.zeros((N, N))
for r_ in range(R):
    for c_ in range(C):
        # target r[r_ * C + c_]
        for a in range(A):
            delta = action_to_direction[a]
            r_p = r_ + delta[0]
            c_p = c_ + delta[1]
            if r_p >= R or r_p < 0 or c_p >= C or c_p < 0:
                next_block = -1
                P_pi[r_ * C + c_][r_ * C + c_] += pi[r_ * C + c_][a]
            else:
                next_block = blocks[r_p][c_p]
                P_pi[r_ * C + c_][r_p * C + c_p] += pi[r_ * C + c_][a]
            r[r_ * C + c_] += pi[r_ * C + c_][a] * next_block

assert np.all(np.sum(P_pi, axis = 0) == 1.)

gamma = 0.9

# State Values.
v = np.ones((R * C, 1))

for iter in range(1000):
    temp_v = r + gamma * P_pi @ v
    v_2d = np.round(temp_v.reshape(R, C), decimals=1)
    print(f'After iter {iter} State Values:\n{v_2d}')
    max_delta = abs(np.max(v - temp_v))
    v = temp_v
    if max_delta < 1e-6:
        break
