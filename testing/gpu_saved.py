#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pyopencl as cl
from datetime import timedelta
from reaction_diffusion_np import generalized_np, gray_scott_np
from reaction_diffusion import generalized, gray_scott
import time

N = 256
v_np = np.random.rand(N, N).astype(np.float32)
u_np = np.random.rand(N, N).astype(np.float32)
rho_np = np.random.rand(5, 5).astype(np.float32)
kap_np = np.random.rand(5, 5).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
pow_np = np.uint(5)
v_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=v_np)
u_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=u_np)
rho_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rho_np)
kap_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=kap_np)

kernel_file = open('generalized.c', 'r')
kernel = kernel_file.read()

prg = cl.Program(ctx, kernel).build()
v_update_g = cl.Buffer(ctx, mf.WRITE_ONLY, v_np.nbytes)
u_update_g = cl.Buffer(ctx, mf.WRITE_ONLY, v_np.nbytes)

start = time.time()
count = 10
for _ in range(count):
    prg.iterate(queue, v_np.shape, None, rho_g, kap_g, v_g, u_g, v_update_g)
end = time.time()
d_g = timedelta(seconds=(end-start) / count)

res = np.empty_like(v_np)
cl.enqueue_copy(queue, res, v_update_g)

# Check on CPU with Numpy:

start = time.time()
for _ in range(count):
    res_np = np.empty_like(v_np)
    generalized_np(rho_np, kap_np, v_np, u_np, res_np)
end = time.time()

d_np = timedelta(seconds=(end-start) / count)

print(f'Deltv_g {d_g}')
print(f'Deltv_np {d_np}')
print(f'Speedup: {round(d_np / d_g, 3)}X')

# print('GPU:', res, 'NP', res_np,sep='\n')

assert np.allclose(res, res_np)

print("All tests cleared.")
