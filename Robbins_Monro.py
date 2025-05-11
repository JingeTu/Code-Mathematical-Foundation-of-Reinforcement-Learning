import numpy as np

def g(x):
    return np.tanh(x)

def g_tilt(x):
    # eta = np.random.normal(loc=0, scale=1)
    eta = 0
    return g(x) + eta, eta

etas = []
ws = []
w = 1.5
ws.append(w)
for k in range(1, 1000):
    y, eta = g_tilt(w)
    w = w - 1 / k * y
    ws.append(w)
    etas.append(eta)
print(ws)
