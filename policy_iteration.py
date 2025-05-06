import numpy as np

R = 1
C = 2
A = 5

N = R * C

action_grid = [
[3, 3],
]

# Policy Matrix.
# pi[i][j] mean the probability of `at state i choose action j`.
pi = np.zeros((N, 5))
for r_ in range(R):
    for c_ in range(C):
        pi[r_ * C + c_][action_grid[r_][c_]] = 1

# Block states.
blocks = np.array([
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
# State trainsition Matrix.
P_pi = np.zeros((N, N))

q_table = np.zeros((N, A))

for k in range(1000):
    # 1. Policy evaluation.
    # 1.1. State transition Matrix Update.
    # Immediate Reward. (This is based on v and pi, so should initialize here.)
    r = np.zeros((R * C, 1))
    for r_ in range(R):
        for c_ in range(C):
            # target r[r_ * C + c_]
            for a in range(A):
                delta = action_to_direction[a]
                r_p = r_ + delta[0]
                c_p = c_ + delta[1]
                if r_p >= R or r_p < 0 or c_p >= C or c_p < 0:
                    next_block = -1
                else:
                    next_block = blocks[r_p][c_p]
                    P_pi[r_ * C + c_][r_p * C + c_p] = pi[r_ * C + c_][a]
                r[r_ * C + c_] += pi[r_ * C + c_][a] * next_block
    # 1.2. Calculate v.
    for l in range(1000):
        temp_v = r + gamma * P_pi @ v
        max_delta = abs(np.max(v - temp_v))
        v = temp_v
        if max_delta < 1e-6:
            print(f'iter {k} Policy evaluation iter {l}.')
            break
    # 2. Policy improvement.
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

print('Conveged Policy(pi):\n', pi)
action_grid = np.zeros((R, C))
for r_ in range(R):
    for c_ in range(C):
        for a in range(A):
            if pi[r_ * C + c_][a] == 1:
                action_grid[r_][c_] = a
print('Conveged Action Grid:\n', action_grid)
