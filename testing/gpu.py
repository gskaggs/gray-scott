#!/usr/bin/env python
# -*- coding: utf-8 -*-
from asyncio import new_event_loop
from re import U
from turtle import update
import numpy as np
import pyopencl as cl
from datetime import timedelta
from reaction_diffusion_np import generalized_np, gray_scott_np, laplacian, gierer_mienhardt
import time
from copy import deepcopy as copy
from core_simulator import CoreSimulator

GENERALIZED = True
TEST = False
LAP = True

N = 256
dt = 0.001
v_np_og = np.random.rand(N, N).astype(np.float32)
u_np_og = np.random.rand(N, N).astype(np.float32)

# Makes the Laplacian zero for the first iteration
if not LAP:
    v_np_og = 1 + np.zeros(v_np_og.shape).astype(np.float32)
    u_np_og = 1 + np.zeros(v_np_og.shape).astype(np.float32)

v_np, u_np = copy(v_np_og), copy(u_np_og)

F, k = np.random.random(), np.random.random()

rho_np = np.random.rand(2, 3, 3, 3).astype(np.float32) 
kap_np = np.random.rand(2, 3, 3, 3).astype(np.float32)

rho_gm, kap_gm, mu, nu = np.random.rand(4)

v_update, u_update = np.zeros(v_np.shape).astype(np.float32), np.zeros(v_np.shape).astype(np.float32)
updates = np.array((v_update, u_update))

count = 1 if TEST else 200
rd_types = [] if TEST else ['generalized', 'gray_scott', 'gierer_mienhardt']

start = time.time()

sonic = CoreSimulator(v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types)
v_np, u_np = sonic.simulate(dt, count)

v_g = copy(v_np)
u_g = copy(u_np)

end = time.time()
d_g = timedelta(seconds=(end-start) / count)
total_g = timedelta(seconds=(end-start))

# Check on CPU with Numpy:
v_np, u_np = copy(v_np_og), copy(u_np_og)
v_update, u_update = np.zeros(v_np.shape)[1:-1, 1:-1], np.zeros(u_np.shape)[1:-1, 1:-1]
updates_np = np.array((v_update, u_update))

start = time.time()
for _ in range(count):
    updates_np = np.zeros(updates_np.shape)
    updates_np[0] += laplacian(v_np)
    updates_np[1] += laplacian(u_np)

    if 'generalized' in rd_types:
        updates_np += np.array(generalized_np(rho_np, kap_np, v_np, u_np))[:, 1:-1, 1:-1]
    if 'gray_scott' in rd_types:
        updates_np += np.array(gray_scott_np(F, k, v_np, u_np))[:, 1:-1, 1:-1]
    if 'gierer_mienhardt' in rd_types:
        updates_np += np.array(gierer_mienhardt(v_np, u_np, rho_gm, kap_gm, mu, nu))
    
    v_np[1:-1, 1:-1] += dt * updates_np[0]
    u_np[1:-1, 1:-1] += dt * updates_np[1]

end = time.time()

d_np = timedelta(seconds=(end-start) / count)
total_np = timedelta(seconds=(end-start))

print(f'Deltv_g {d_g}   Total time_g: {total_g}')
print(f'Deltv_np {d_np}   Total time_np: {total_np}')
print(f'Speedup: {round(d_np / d_g, 3)}X')

if TEST:
    print('Original v', v_np_og, sep='\n')
    print('GPU:', v_g, 'CPU', v_np, sep='\n')
    print('Diff:', v_g - v_np, sep='\n')
# print('GPU:', updates, 'NP', updates_np,sep='\n')

# print(updates, updates_np)

assert np.allclose(v_g, v_np)
assert np.allclose(u_g, u_np)

print("All tests cleared.")
