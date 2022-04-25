#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from datetime import timedelta
import time
from copy import deepcopy as copy
from core_simulator import CoreSimulatorGpu
from core_simulator_np import CoreSimulatorNp

# Initialize testing parameters
DEBUG = False
LAPLACIAN = True
ABS = True
np.random.seed(0)

# Initialize simulation hyper-parameters
rd_types = ['generalized', 'gray_scott']
num_iters = 1 if DEBUG else 2000
grid_size = 256                        
dt = 0.001

# Original v and u
v_og = np.random.rand(grid_size, grid_size)
u_og = np.random.rand(grid_size, grid_size)

# Makes the Laplacian zero for the first iteration
if not LAPLACIAN:
    v_og = np.zeros(v_og.shape)
    u_og = np.zeros(v_og.shape)

# Initialize simulation parameters
F, k = np.array([np.random.random(), np.random.random()])
rho_np = (2 * np.random.rand(2, 3, 3, 3) - 1).round(decimals=3)
kap_np = (2 * np.random.rand(2, 3, 3, 3) - 1).round(decimals=3)

if ABS:
    rho_np = np.abs(rho_np)
    kap_np = np.abs(kap_np)

rho_gm, kap_gm, mu, nu = -.5, .238, 1.0, .9 #np.random.rand(4)
dv, du = 1, 1

# Run simulation on GPU
start = time.time()
v, u = copy(v_og), copy(u_og)
sonic = CoreSimulatorGpu(v, u, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, dv, du, rd_types)
v, u = sonic.simulate(dt, num_iters)
end = time.time()

# Save results
v_g, u_g = copy(v), copy(u)
d_g = timedelta(seconds=(end-start) / num_iters)
total_g = timedelta(seconds=(end-start))

# Check on CPU with Numpy
start = time.time()
v, u = copy(v_og), copy(u_og)
knuckles = CoreSimulatorNp(v, u, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, dv, du, rd_types)
v, u = knuckles.simulate(dt, num_iters)
end = time.time()

# Save results
v_np, u_np = copy(v), copy(u)
d_np = timedelta(seconds=(end-start) / num_iters)
total_np = timedelta(seconds=(end-start))

print(f'Deltv_g {d_g}   Total time_g: {total_g}')
print(f'Deltv {d_np}   Total time_np: {total_np}')
print(f'Speedup: {round(d_np / d_g, 3)}X')

print('Original v', v_og, sep='\n')
print('GPU:', v_g, 'CPU', v_np, sep='\n')
print('Diff:', v_g - v_np, sep='\n')

assert np.allclose(v_g, v_np)
assert np.allclose(u_g, u_np)

print("All tests cleared.")
