import numpy as np
import pyopencl as cl
import pyopencl.array as array
import pyopencl.clmath as clmath

def init_opencl():
    ctx = cl.create_some_context()
    return ctx, cl.CommandQueue(ctx), cl.mem_flags

class CoreSimulator():
    def __init__(self, v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types):
        # Set type to float32 or these won't align in GPU memory
        self.v_np = v_np.astype(np.float64)
        self.u_np = u_np.astype(np.float64)
        self.rho_np = rho_np.astype(np.float64)
        self.kap_np = kap_np.astype(np.float64)
        self.F = np.float64(F)
        self.k = np.float64(k)
        self.rho_gm = np.float64(rho_gm)
        self.kap_gm = np.float64(kap_gm)
        self.mu = np.float64(mu)
        self.nu = np.float64(nu)
        self.rd_types = rd_types
        self.ROUND = False
        self.round_const = 10**6

class CoreSimulatorGpu(CoreSimulator):
    def __init__(self, v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types):
        super().__init__(v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types)
        
        # Initialize some opencl stuff
        self.ctx, self.queue, self.mf = init_opencl()

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
        self.v_g = array.to_device(self.queue, self.v_np)
        self.u_g = array.to_device(self.queue, self.u_np)

        # Updates 
        # (We don't set this memory to any particular value here. 
        # Instead, we set it in laplacian.c)
        # Also, note that we assume the update matrix has same size as n_np
        self.v_update_np = np.empty_like(self.v_np)
        self.u_update_np = np.empty_like(self.u_np)
        self.v_update_g = array.to_device(self.queue, self.v_update_np)
        self.u_update_g = array.to_device(self.queue, self.u_update_np)

        # Simulation Paramaters
        # Index 0 is for V, Index 1 is for U
        self.rho_g = [cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=rho_np[i]) for i in range(2)]
        self.kap_g = [cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=kap_np[i]) for i in range(2)]

    def simulate(self, dt, T):
        dt = np.float64(dt)
        for _ in range(T):
            # Laplacian
            self.lap(self.queue, self.v_np.shape, None, self.v_g.data, self.v_update_g.data)
            self.lap(self.queue, self.u_np.shape, None, self.u_g.data, self.u_update_g.data)

            # Generalized Reaction Diffusion
            if 'generalized' in self.rd_types:
                self.gen_rd(self.queue, self.v_np.shape, None, self.rho_g[0], self.kap_g[0], self.v_g.data, self.u_g.data, self.v_update_g.data)
                self.gen_rd(self.queue, self.v_np.shape, None, self.rho_g[1], self.kap_g[1], self.v_g.data, self.u_g.data, self.u_update_g.data)

            if 'gray_scott' in self.rd_types:
                self.gs(self.queue, self.v_np.shape, None,  self.v_g.data, self.u_g.data, self.F, self.k, self.v_update_g.data, self.u_update_g.data)

            if 'gierer_mienhardt' in self.rd_types:
                self.gm(self.queue, self.v_np.shape, None,  self.v_g.data, self.u_g.data, self.rho_gm, self.kap_gm, self.mu, self.nu, self.v_update_g.data, self.u_update_g.data)

            # Update V and U
            self.update(self.queue, self.v_np.shape, None, self.v_g.data, self.v_update_g.data, dt)
            self.update(self.queue, self.v_np.shape, None, self.u_g.data, self.u_update_g.data, dt)

            if self.ROUND:
                self.v_g = clmath.round(self.round_const * self.v_g) / self.round_const
                self.u_g = clmath.round(self.round_const * self.u_g) / self.round_const

        # Get values
        self.v_np = self.v_g.get()
        self.u_np = self.u_g.get()

        return self.v_np, self.u_np
