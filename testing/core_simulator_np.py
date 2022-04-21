import numpy as np
from core_simulator_np_utils import generalized_np, gray_scott_np, laplacian, gierer_mienhardt

class CoreSimulatorNp():
    def __init__(self, v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, rd_types):
        self.v_np = v_np
        self.u_np = u_np
        self.rho_np = rho_np
        self.kap_np = kap_np
        self.F = F
        self.k = k
        self.rho_gm = rho_gm
        self.kap_gm = kap_gm
        self.mu = mu
        self.nu = nu
        self.rd_types = rd_types

        v_update, u_update = np.zeros(v_np.shape)[1:-1, 1:-1], np.zeros(u_np.shape)[1:-1, 1:-1]
        self.updates_np = np.array((v_update, u_update))

    def simulate(self, dt, T):
        for _ in range(T):
            self.updates_np = np.zeros(self.updates_np.shape)
            self.updates_np[0] += laplacian(self.v_np)
            self.updates_np[1] += laplacian(self.u_np)

            if 'generalized' in self.rd_types:
                self.updates_np += np.array(generalized_np(self.rho_np, self.kap_np, self.v_np, self.u_np))[:, 1:-1, 1:-1]
            if 'gray_scott' in self.rd_types:
                self.updates_np += np.array(gray_scott_np(self.F, self.k, self.v_np, self.u_np))[:, 1:-1, 1:-1]
            if 'gierer_mienhardt' in self.rd_types:
                self.updates_np += np.array(gierer_mienhardt(self.v_np, self.u_np, self.rho_gm, self.kap_gm, self.mu, self.nu))
            
            self.v_np[1:-1, 1:-1] += dt * self.updates_np[0]
            self.u_np[1:-1, 1:-1] += dt * self.updates_np[1]

        return self.v_np, self.u_np