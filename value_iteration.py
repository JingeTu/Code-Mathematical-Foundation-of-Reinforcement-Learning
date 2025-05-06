import numpy as np

R = 2
C = 2
A = 5

N = R * C

# Policy Matrix.
# pi[i][j] mean the probability of `at state i choose action j`.
pi = np.zeros((N, 5))

# Block states.
blocks = np.array([
[0, -1],
[0, 1],
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

q_table = np.zeros((N, A))

for k in range(1000):
    v_kp1 = np.ones((R * C, 1))
    for r_ in range(R):
        for c_ in range(C):
            # q value calculation.
            max_q = None
            max_a = None
            for a in range(A):
                pi[r_ * C + c_][a] = 0
                delta = action_to_direction[a]
                r_p = r_ + delta[0]
                c_p = c_ + delta[1]
                if r_p >= R or r_p < 0 or c_p >= C or c_p < 0:
                    q_table[r_ * C + c_][a] = -1 + gamma * v[r_ * C + c_]
                else:
                    q_table[r_ * C + c_][a] = blocks[r_p][c_p] + gamma * v[r_p * C + c_p]
                if max_q is None or q_table[r_ * C + c_][a] > max_q:
                    max_q = q_table[r_ * C + c_][a]
                    max_a = a
            pi[r_ * C + c_][max_a] = 1
            # State update.
            v_kp1[r_ * C + c_] = max_q
    inf_norm = np.linalg.norm(v_kp1 - v, ord=np.inf)
    v = v_kp1
    if inf_norm < 1e-6:
        print(f'Converged in iter {k + 1}.')
        break
v_2d = v.reshape(R, C)
print('State Value(v):\n', v_2d)
action_grid = np.zeros((R, C))
for r_ in range(R):
    for c_ in range(C):
        for a in range(A):
            if pi[r_ * C + c_][a] == 1:
                action_grid[r_][c_] = a
print('Policy:\n', action_grid)
