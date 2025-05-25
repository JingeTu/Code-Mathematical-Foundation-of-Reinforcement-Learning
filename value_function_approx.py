import numpy as np

R = 5
C = 5
A = 5

N = R * C

# Policy Matrix.
# pi[i][j] mean the probability of `at state i choose action j`.
pi = np.zeros((N, 5))

for r in range(R):
    for c in range(C):
        pi[r * C + c] = 1 / A

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
v = np.zeros((R * C, 1))

for iter in range(1000):
    temp_v = r + gamma * P_pi @ v
    v_2d = temp_v.reshape(R, C)
    print(f'After iter {iter} State Values:\n{np.round(v_2d, decimals=1)}')
    max_delta = abs(np.max(v - temp_v))
    v = temp_v
    if max_delta < 1e-6:
        break

def feature_vector(y, x):
    x = x / C
    y = y / R
    # return np.array([1, x, y, x ** 2, y **2, x * y])
    return np.array([1, x, y, x ** 2, y **2, x * y, x ** 3, y ** 3, x ** 2 * y, x * y ** 2])

# w = np.random.normal(loc=0, scale=0.01, size=6)
w = np.random.normal(loc=0, scale=0.01, size=10)

# def feature_vector(y, x):
#     idx = y * C + x
#     ret = np.zeros(R * C)
#     ret[idx] = 1
#     return ret

# w = np.random.normal(loc=0, scale=0.01, size=R * C)

num_episode = 100000

E = 500

# alpha = 1.
alpha = 0.001
r_target = 3
c_target = 2

# Can not converge.

for episode in range(num_episode):
    r_start = np.random.randint(0, R)
    c_start = np.random.randint(0, C)
    a_start = np.random.randint(0, A)
    # a_start = np.random.choice(range(A), size=1, replace=False, p=pi[r_start * C + c_start])[0]
    state = (r_start, c_start)

    episode_history = []
    for e in range(E):
        r_, c_ = state
        a = a_start
        if e != 0:
            # a = np.random.choice(range(A), size=1, replace=False, p=pi[r_ * C + c_])[0]
            # As uniform distribution, this will be more efficient.
            a = np.random.randint(0, A)

        delta = action_to_direction[a]
        r_p = r_ + delta[0]
        c_p = c_ + delta[1]

        if r_p >= R or r_p < 0 or c_p >= C or c_p < 0 or blocks[r_p][c_p] == -1:
            reward = -1
            next_state = (r_, c_)
        else:
            reward = blocks[r_p][c_p]
            next_state = (r_p, c_p)

        episode_history.append((state, a, reward))
        state = next_state

        if next_state[0] == r_target and next_state[1] == c_target:
            break

    for e in range(1, len(episode_history)):
        (r_last, c_last), a_last, reward_last = episode_history[e - 1]
        (r_cur, c_cur), _, _ = episode_history[e]
        phi_last = feature_vector(r_last, c_last)
        phi_cur = feature_vector(r_cur, c_cur)
        w = w + alpha * (reward_last + gamma * phi_cur.dot(w) - phi_last.dot(w)) * phi_last

    if episode % 500 == 0:
        v_est = np.zeros((R * C, 1))
        for r_ in range(0, R):
            for c_ in range(0, C):
                v_est[r_ * C + c_] = feature_vector(r_, c_).dot(w)

        print(f'episode {episode}/{num_episode} Diff State Values: {np.linalg.norm(v_est - v)}')
        print(f'v_est_2d: {v_est.reshape(R, C)}')