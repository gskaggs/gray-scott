import numpy as np
import pyopencl as cl
import pyopencl.array
from datetime import timedelta
import time

def init_opencl():
    ctx = cl.create_some_context()
    return ctx, cl.CommandQueue(ctx), cl.mem_flags

class GpuSimulator():
    def __init__(self):
        self.ctx, self.queue, self.mf = init_opencl()

    def simulate(self):
        raise NotImplementedError

class GrayScottSimulator(GpuSimulator):
    def __init__(self, v_np, u_np, F, k):
        super().__init__()
        self.F = F
        self.k = k
        self.v_np = v_np
        self.u_np = u_np

        kernel_file = open('gray_scott.c', 'r')
        kernel = kernel_file.read()

        self.prg = cl.Program(self.ctx, kernel).build()
        self.v_update_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, v_np.nbytes)
        self.u_update_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, v_np.nbytes)

    def simulate(self):
        self.v_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.v_np)
        self.u_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.u_np)

        self.prg.iterate(self.queue, self.v_np.shape, None,  self.v_g, self.u_g, np.float32(self.F), np.float32(self.k), self.v_update_g, self.u_update_g)
        v_update_np, u_update_np = np.empty_like(self.v_np), np.empty_like(self.v_np)
        cl.enqueue_copy(self.queue, u_update_np, self.u_update_g)
        cl.enqueue_copy(self.queue, v_update_np, self.v_update_g)
        return v_update_np, u_update_np

class UpdateSimulator(GpuSimulator):
    def __init__(self, v_np, u_np, v_update_np, u_update_np, dt):
        super().__init__()
        self.v_update_np = v_update_np
        self.u_update_np = u_update_np
        self.v_np = v_np
        self.u_np = u_np
        self.dt = dt

        kernel_file = open('update.c', 'r')
        kernel = kernel_file.read()

        self.prg = cl.Program(self.ctx, kernel).build()

    def simulate(self):
        self.v_g = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.v_np.nbytes)
        self.u_g = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.v_np.nbytes)
        cl.enqueue_copy(self.queue, self.v_g, self.v_np)
        cl.enqueue_copy(self.queue, self.u_g, self.u_np)

        # temp = cl.array.to_device(self.queue, self.v_np)
        # print(temp.get() - self.v_np)  

        v_update_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.v_update_np)
        u_update_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.u_update_np)

        # print(self.v_update_np)
        # cl.enqueue_copy(self.queue, v_update_g, self.v_update_np)
        # cl.enqueue_copy(self.queue, self.v_update_np, v_update_g)

        self.prg.iterate(self.queue, self.v_update_np.shape, None, self.v_g, v_update_g, np.float32(self.dt))
        self.prg.iterate(self.queue, self.u_update_np.shape, None, self.u_g, u_update_g, np.float32(self.dt))

        v_np, u_np = np.empty_like(self.v_np), np.empty_like(self.v_np)
        cl.enqueue_copy(self.queue, v_np, self.v_g)
        cl.enqueue_copy(self.queue, u_np, self.u_g)
        
        # return temp.get(), temp.get()
        return v_np, u_np


class LaplacianSimulator(GpuSimulator):
    def __init__(self, v_np, u_np):
        super().__init__()
        self.v_np = v_np
        self.u_np = u_np

        kernel_file = open('laplacian.c', 'r')
        kernel = kernel_file.read()

        self.prg = cl.Program(self.ctx, kernel).build()
        self.v_update_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, v_np.nbytes)
        self.u_update_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, v_np.nbytes)

    def simulate(self):
        self.u_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.u_np)
        self.v_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.v_np)

        self.prg.iterate(self.queue, self.v_np.shape, None, self.v_g, self.v_update_g)
        self.prg.iterate(self.queue, self.u_np.shape, None, self.u_g, self.u_update_g)

        v_update_np, u_update_np = np.empty_like(self.v_np), np.empty_like(self.v_np)
        cl.enqueue_copy(self.queue, v_update_np, self.v_update_g)
        cl.enqueue_copy(self.queue, u_update_np, self.u_update_g)
        
        return v_update_np, u_update_np

class GeneralizedSimulator(GpuSimulator):
    def __init__(self, v_np, u_np, rho_np, kap_np):
        super().__init__()
        self.v_np = v_np
        self.u_np = u_np
        self.rho_np = rho_np
        self.kap_np = kap_np

        self.rho_g = [cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=rho_np[i]) for i in range(2)]
        self.kap_g = [cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=kap_np[i]) for i in range(2)]

        kernel_file = open('generalized.c', 'r')
        kernel = kernel_file.read()

        self.prg = cl.Program(self.ctx, kernel).build()
        self.v_update_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, v_np.nbytes)
        self.u_update_g = cl.Buffer(self.ctx, self.mf.WRITE_ONLY, v_np.nbytes)

    def simulate(self):
        self.u_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.u_np)
        self.v_g = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.v_np)

        self.prg.iterate(self.queue, self.v_np.shape, None, self.rho_g[0], self.kap_g[0], self.v_g, self.u_g, self.v_update_g)
        self.prg.iterate(self.queue, self.v_np.shape, None, self.rho_g[1], self.kap_g[1], self.v_g, self.u_g, self.u_update_g)

        v_update_np, u_update_np = np.empty_like(self.v_np), np.empty_like(self.v_np)
        cl.enqueue_copy(self.queue, v_update_np, self.v_update_g)
        cl.enqueue_copy(self.queue, u_update_np, self.u_update_g)
        
        return v_update_np, u_update_np


