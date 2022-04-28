import numpy as np
from python.core_simulator_np_utils import generalized_np, gray_scott_np, laplacian, gierer_mienhardt, update_ghosts
from python.core_simulator import CoreSimulator

class CoreSimulatorNp(CoreSimulator):
    def __init__(self, v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, dv, du, rd_types):
        super().__init__(v_np, u_np, rho_np, kap_np, F, k, rho_gm, kap_gm, mu, nu, dv, du, rd_types)

        v_update, u_update = np.zeros(v_np.shape)[1:-1, 1:-1], np.zeros(u_np.shape)[1:-1, 1:-1]
        self.updates_np = np.array((v_update, u_update))

    def simulate(self, dt, T):
        for _ in range(T):
            self.updates_np = np.zeros(self.updates_np.shape)
            self.updates_np[0] += self.dv * laplacian(self.v_np)
            self.updates_np[1] += self.du * laplacian(self.u_np)

            if 'generalized' in self.rd_types:
                self.updates_np += np.array(generalized_np(self.rho_np, self.kap_np, self.v_np, self.u_np))[:, 1:-1, 1:-1]
            if 'gray_scott' in self.rd_types:
                self.updates_np += np.array(gray_scott_np(self.F, self.k, self.v_np, self.u_np))[:, 1:-1, 1:-1]
            if 'gierer_mienhardt' in self.rd_types:
                self.updates_np += np.array(gierer_mienhardt(self.v_np, self.u_np, self.rho_gm, self.kap_gm, self.mu, self.nu))
            
            self.v_np[1:-1, 1:-1] += dt * self.updates_np[0]
            self.u_np[1:-1, 1:-1] += dt * self.updates_np[1]

            np.clip(self.v_np[1:-1, 1:-1], 0, 5, out=self.v_np[1:-1, 1:-1])
            np.clip(self.u_np[1:-1, 1:-1], 0, 5, out=self.u_np[1:-1, 1:-1])

            update_ghosts(self.v_np)
            update_ghosts(self.u_np)

            if self.ROUND:
                self.v_np = np.round(self.round_const * self.v_np) / self.round_const
                self.u_np = np.round(self.round_const * self.u_np) / self.round_const

        return self.v_np, self.u_np