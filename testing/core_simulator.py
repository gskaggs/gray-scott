import numpy as np
import pyopencl as cl

def init_opencl():
    ctx = cl.create_some_context()
    return ctx, cl.CommandQueue(ctx), cl.mem_flags

class CoreSimulator():
    def __init__(self, v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types):
        # Initialize some opencl stuff
        self.ctx, self.queue, self.mf = init_opencl()

        # Set type to float32 or these won't align in GPU memory
        self.v_np = v_np.astype(np.float32)
        self.u_np = u_np.astype(np.float32)
        self.rho_np = rho_np.astype(np.float32)
        self.kap_np = kap_np.astype(np.float32)
        self.F = np.float32(F)
        self.k = np.float32(k)
        self.rho_gm = np.float32(rho_gm)
        self.kap_gm = np.float32(kap_gm)
        self.mu = np.float32(mu)
        self.nu = np.float32(nu)
        self.rd_types = rd_types

        # Build programs:
        # Laplacian Program
        kernel_file = open('./c/laplacian.c', 'r')
        kernel = kernel_file.read()

        prg = cl.Program(self.ctx, kernel).build()
        self.lap = prg.iterate

        # Generalized Reaction-Diffusion Program
        kernel_file = open('./c/generalized.c', 'r')
        kernel = kernel_file.read()

        prg = cl.Program(self.ctx, kernel).build()
        self.gen_rd = prg.iterate

        # Gray Scott Reaction-Diffusion Program
        kernel_file = open('./c/gray_scott.c', 'r')
        kernel = kernel_file.read()

        prg = cl.Program(self.ctx, kernel).build()
        self.gs = prg.iterate

        # Gray Scott Reaction-Diffusion Program
        kernel_file = open('./c/gierer_mienhardt.c', 'r')
        kernel = kernel_file.read()

        prg = cl.Program(self.ctx, kernel).build()
        self.gm = prg.iterate

        # Update Program
        kernel_file = open('./c/update.c', 'r')
        kernel = kernel_file.read()

        prg = cl.Program(self.ctx, kernel).build()
        self.update = prg.iterate

        # Allocate memory:
        # V and U
        self.v_g = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.v_np.nbytes)
        self.u_g = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.v_np.nbytes)
        cl.enqueue_copy(self.queue, self.v_g, self.v_np)
        cl.enqueue_copy(self.queue, self.u_g, self.u_np)

        # Updates 
        # (We don't set this memory to any particular value here. 
        # Instead, we set it in laplacian.c)
        # Also, note that we assume the update matrix has same size as n_np
        self.v_update_g = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.v_np.nbytes)
        self.u_update_g = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=self.v_np.nbytes)

        # Simulation Paramaters
        # Index 0 is for V, Index 1 is for U
        self.rho_g = [cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=rho_np[i]) for i in range(2)]
        self.kap_g = [cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=kap_np[i]) for i in range(2)]

    def simulate(self, dt, T):
        dt = np.float32(dt)
        for _ in range(T):
            # Laplacian
            self.lap(self.queue, self.v_np.shape, None, self.v_g, self.v_update_g)
            self.lap(self.queue, self.u_np.shape, None, self.u_g, self.u_update_g)

            # Generalized Reaction Diffusion
            if 'generalized' in self.rd_types:
                self.gen_rd(self.queue, self.v_np.shape, None, self.rho_g[0], self.kap_g[0], self.v_g, self.u_g, self.v_update_g)
                self.gen_rd(self.queue, self.v_np.shape, None, self.rho_g[1], self.kap_g[1], self.v_g, self.u_g, self.u_update_g)

            if 'gray_scott' in self.rd_types:
                self.gs(self.queue, self.v_np.shape, None,  self.v_g, self.u_g, self.F, self.k, self.v_update_g, self.u_update_g)

            if 'gierer_mienhardt' in self.rd_types:
                self.gm(self.queue, self.v_np.shape, None,  self.v_g, self.u_g, self.rho_gm, self.kap_gm, self.mu, self.nu, self.v_update_g, self.u_update_g)

            # Update V and U
            self.update(self.queue, self.v_np.shape, None, self.v_g, self.v_update_g, dt)
            self.update(self.queue, self.v_np.shape, None, self.u_g, self.u_update_g, dt)

        # Get values
        cl.enqueue_copy(self.queue, self.v_np, self.v_g)
        cl.enqueue_copy(self.queue, self.u_np, self.u_g)

        return self.v_np, self.u_np
