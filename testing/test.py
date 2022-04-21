#!/usr/bin/env python
# -*- coding: utf-8 -*-
from asyncio import new_event_loop
from re import U
from turtle import update
import numpy as np
import pyopencl as cl
from datetime import timedelta
import time
from copy import deepcopy as copy
from core_simulator import CoreSimulator
from core_simulator_np import CoreSimulatorNp

# Initialize testing parameters
DEBUG = False
LAPLACIAN = True

# Initialize simulation hyper-parameters
rd_types = [] if DEBUG else ['generalized', 'gray_scott', 'gierer_mienhardt']
num_iters = 1 if DEBUG else 200      
grid_size = 256                        
dt = 0.001

# Original v and u
v_og = np.random.rand(grid_size, grid_size).astype(np.float32)
u_og = np.random.rand(grid_size, grid_size).astype(np.float32)

# Makes the Laplacian zero for the first iteration
if not LAPLACIAN:
    v_og = 1 + np.zeros(v_og.shape).astype(np.float32)
    u_og = 1 + np.zeros(v_og.shape).astype(np.float32)

# Initialize simulation parameters
F, k = np.random.random(), np.random.random()
rho_np = np.random.rand(2, 3, 3, 3).astype(np.float32) 
kap_np = np.random.rand(2, 3, 3, 3).astype(np.float32)
rho_gm, kap_gm, mu, nu = np.random.rand(4)

# Run simulation on GPU
start = time.time()
v, u = copy(v_og), copy(u_og)
sonic = CoreSimulator(v, u, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types)
v, u = sonic.simulate(dt, num_iters)
end = time.time()

# Save results
v_g, u_g = copy(v), copy(u)
d_g = timedelta(seconds=(end-start) / num_iters)
total_g = timedelta(seconds=(end-start))

# Check on CPU with Numpy:
start = time.time()
v, u = copy(v_og), copy(u_og)
knuckles = CoreSimulatorNp(v, u, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types)
v, u = knuckles.simulate(dt, num_iters)
end = time.time()

# Save results
v_np, u_np = copy(v), copy(u)
d_np = timedelta(seconds=(end-start) / num_iters)
total_np = timedelta(seconds=(end-start))

print(f'Deltv_g {d_g}   Total time_g: {total_g}')
print(f'Deltv {d_np}   Total time_np: {total_np}')
print(f'Speedup: {round(d_np / d_g, 3)}X')

if DEBUG:
    print('Original v', v_og, sep='\n')
    print('GPU:', v_g, 'CPU', v_np, sep='\n')
    print('Diff:', v_g - v_np, sep='\n')

assert np.allclose(v_g, v_np)
assert np.allclose(u_g, u_np)

print("All tests cleared.")
