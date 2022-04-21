import numpy as np

def f():
    return np.array([1]), np.array([1])

u, v = np.array([1]), np.array([1])

temp = [u, v]
temp[0] += np.array(f())[0]

print(temp, u, v)
