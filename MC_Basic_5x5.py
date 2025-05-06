import numpy as np

R = 5
C = 5
A = 5

N = R * C

action_grid = [
[4, 4, 4, 4, 4],
[4, 4, 4, 4, 4],
[4, 4, 4, 4, 4],
[4, 4, 4, 4, 4],
[4, 4, 4, 4, 4],
]

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

def traverse(r_, c_, depth):
    # E, A, gamma, C
    if r_ >= R or r_ < 0 or c_ >= C or c_ < 0:
        if depth < E:
            return -1 + gamma * traverse(r_, c_, depth + 1)
        return -1
    if depth >= E:
        return 0.
    act_val = blocks[r_][c_]
    for a in range(A):
        if pi[r_ * C + c_][a] == 0.:
            continue
        delta = action_to_direction[a]
        r_p = r_ + delta[0]
        c_p = c_ + delta[1]
        act_val += gamma * traverse(r_p, c_p, depth + 1)
    return act_val

memo_last_pi = np.full((R, C), -1)

for k in range(1000):
    # Attention: updated policy can not put into pi immediately.
    pi_temp = np.zeros((N, 5))
    pi_changed = False
    for r_ in range(R):
        for c_ in range(C):
            max_q = None
            max_a = None
            q_acts = np.zeros((A, 1))
            for a in range(A):
                pi_temp[r_ * C + c_][a] = 0
                delta = action_to_direction[a]
                r_p = r_ + delta[0]
                c_p = c_ + delta[1]
                q_acts[a] = traverse(r_p, c_p, 1)
                if max_q is None or max_q < q_acts[a]:
                    max_q = q_acts[a]
                    max_a = a
            if memo_last_pi[r_][c_] != max_a:
                print(f'k {k} ({r_}, {c_}) {memo_last_pi[r_][c_]} -> {max_a}')
                pi_changed = True
            pi_temp[r_ * C + c_][max_a] = 1
            memo_last_pi[r_][c_] = max_a
    pi = pi_temp
    if not pi_changed:
        print(f'Converged in iter {k + 1}.')
        break

action_grid = np.zeros((R, C))
for r_ in range(R):
    for c_ in range(C):
        for a in range(A):
            if pi[r_ * C + c_][a] == 1:
                action_grid[r_][c_] = a
print('Conveged Action Grid:\n', action_grid)
