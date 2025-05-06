import math
import numpy as np

def func1(x):
    return 0.5 * x

def func2(x):
    return 0.5 * math.sin(x)

def func3(A, x):
    return A @ x

converge_path = [10.]
error_path = []
for i in range(0, 1000):
    x0 = converge_path[-1]
    x1 = func1(x0)
    if abs(x1 - x0) < 1e-6:
        break
    error_path.append(x1 - x0)
    converge_path.append(x1)
print('func1 converge_path:', converge_path)
print('func1 error_path:', error_path)

converge_path = [math.pi/2]
error_path = []
for i in range(0, 1000):
    x0 = converge_path[-1]
    x1 = func2(x0)
    if abs(x1 - x0) < 1e-6:
        break
    error_path.append(x1 - x0)
    converge_path.append(x1)
print('func2 converge_path:', converge_path)
print('func2 error_path:', error_path)

# A = np.asarray([
# [0, 0.5, 0.5, 0],
# [0, 0, 0, 1],
# [0, 0, 0, 1],
# [0, 0, 0, 1],
# ])

def create_matrix_spectral(gamma, size=(4,4), seed=None):
    if seed is not None:
        np.random.seed(seed)
    A = np.random.randn(*size)
    current_norm = np.linalg.norm(A, 2)
    A_scaled = (gamma / current_norm) * A
    return A_scaled

# # Construction method 1
# A = np.random.rand(4, 4)
# A /= 4
# # Construction method 2
# A = np.random.rand(4, 4)
# A = (2 * A - 1) / 4
# # A = 2 * A / 4 - 1 # Attention, this is wrong.
# Construction method 3
A = create_matrix_spectral(0.9)
x = np.random.rand(4, 1)
for i in range(0, 1000):
    x1 = func3(A, x)
    delta = np.linalg.norm(x1 - x)
    print(f'func3 iter {i} abs(delta) {abs(delta)}')
    x = x1
    if abs(delta) < 1e-6:
        break
