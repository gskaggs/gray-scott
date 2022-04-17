# File       : gray_scott.py
# Created    : Sat Jan 30 2021 05:12:47 PM (+0100)
# Description: Gray-Scott reaction-diffusion
# Copyright 2021 ETH Zurich. All Rights Reserved.
import os
import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt
from PIL import Image as im
from enum_util import RdType
from reaction_diffusion_np import gray_scott_np, generalized_np
from reaction_diffusion import GrayScottSimulator, GeneralizedSimulator

class GrayScott:
    """
    Gray-Scott reaction-diffusion system.
    http://mrob.com/pub/comp/xmorphia/index.html
    https://www.lanevol.org/resources/gray-scott

    Reactions:
        U + 2V → 3V
        V → P (P is an inert product)
    """

    def __init__(self,
                 *,
                 chromosome,
                 Du=2.0e-5,
                 Dv=1.0e-5,
                 x0=-1,
                 x1=1,
                 N=256,
                 Fo=0.8,
                 initial_condition='trefethen',
                 second_order=False,
                 movie=False,
                 outdir='.',
                 name='',
                 rd_types=['gray_scott']):
        """
        Constructor.
        The domain is a square.

        Arguments
            F: parameter F in governing equations
            k: parameter k (k) in governing equations
            Du: diffusivity of species U
            Dv: diffusivity of species V
            x0: left domain coordinate
            x1: right domain coordinate
            N: number of nodes in x and y
            Fo: Fourier number (<= 1)

            initial_condition: type of initial condition to be used
            second_order: use Heun's method (2nd-order Runge-Kutta)
            movie: create a movie
            outdir: output directory
        """
        # gray-scott parameters
        self.F = chromosome.get_param('F')
        self.k = chromosome.get_param('k')

        # grierer-mienhardt parameters
        self.rho = chromosome.get_param('rho')
        self.mu = chromosome.get_param('mu')
        self.nu = chromosome.get_param('nu')
        self.kappa = chromosome.get_param('kappa')

        # generalized parameters
        self.gen_params = chromosome.gen_params

        Nnodes = N + 1
        self.Fo = Fo
        self.x0 = x0
        self.x1 = x1

        # options
        self.name = name
        self.movie = movie
        self.outdir = outdir
        self.dump_count = 0
        self.second_order = second_order

        # grid spacing
        L = x1 - x0
        dx = L / N

        # intermediates
        self.fa = Du / dx**2
        self.fs = Dv / dx**2
        self.dt = Fo * dx**2 / (4*max(Du, Dv))

        self.rd_type = RdType(rd_types)

        if len(rd_types) != 1 or not self.rd_type.GRAY_SCOTT:
            self.dt /= 10
            self.fs = .1
            self.fa = 2
            # print('dt', self.dt)
            # print('Du', self.fa, '\nDv', self.fs)

        if self.rd_type.GENERALIZED:
            self.rho_np, self.kap_np = self.gen_params[0], self.gen_params[1]

        # nodal grid (+ghosts)
        x = np.linspace(x0-dx, x1+dx, Nnodes+2)
        y = np.linspace(x0-dx, x1+dx, Nnodes+2)
        self.x, self.y = np.meshgrid(x, y)

        # initial condition
        self.u = np.zeros((len(x), len(y))).astype(np.float32)
        self.v = np.zeros((len(x), len(y))).astype(np.float32)
        if self.second_order:
            self.u_tmp = np.zeros((len(x), len(y)))
            self.v_tmp = np.zeros((len(x), len(y)))

        if initial_condition == 'trefethen':
            self._trefethen_IC()
        elif initial_condition == 'random':
            self._random_IC()
        else:
            raise RuntimeError(
                f"Unknown initial condition type: `{initial_condition}`")

        # populate ghost cells
        self.update_ghosts(self.u)
        self.update_ghosts(self.v)

        self.v_view = self.v[1:-1, 1:-1]
        self.u_view = self.u[1:-1, 1:-1]

        if self.rd_type.GRAY_SCOTT:
            self.grey_scott_sim = GrayScottSimulator(self.v, self.u, self.F, self.k)
        if self.rd_type.GENERALIZED:
            self.generalized_sim = GeneralizedSimulator(self.v, self.u, self.rho_np, self.kap_np)

    def integrate(self, t0, t1, *, dump_freq=100, report=50, dirichlet_vis=False, should_dump=False, fitness='pattern'):
        """
        Integrate system.

        Arguments:
            t0: start time
            r1: end time
            dump_freq: dump frequency in steps
            report: stdout report frequency
        
        Returns:
            pattern: is True if simulation terminates
                     in a Turing pattern
        """
        t = t0
        s = 0
        latest = 0
        while t < t1 and not np.isnan(np.sum(self.v)):
            if should_dump and s % dump_freq == 0:
                self._dump(s, t)
            t = self.update(time=t)
            if (t1 - t) < self.dt:
                self.dt = t1 - t
            s += 1

            if fitness=='pattern' and self._check_pattern():
                latest = t
            

        pattern = self._check_pattern()
        image = self._dump(s, t, dirichlet_vis)
        
        if fitness=='dirichlet':
            latest = self._dirichlet()

        return pattern, latest, image


    def update(self, *, time=0):
        """
        Perform time integration step

        Arguments:
            time: current time

        Returns:
            Time after integration
        """
        if self.second_order:
            self._heun()
        else:
            self._euler()

        # update ghost cells
        self.update_ghosts(self.u)
        self.update_ghosts(self.v)

        return time + self.dt

    def _dirichlet(self):
        '''
        Computes a proxy for the Dirichlet energy of v.
        This is used as a heuristic for how complex the pattern is.
        '''
        v_view = self.v[1:-1, 1:-1]
        grad = np.gradient(v_view / np.max(v_view))
        grad = [l**2 for l in grad]
        return sum(sum(sum(grad)))

    def _check_pattern(self):
        """
        Check if a Turing pattern is present
        """
        # internal domain
        v_flat = self.v[1:-1, 1:-1].flatten()

        theta = 0.001
        diff  = max(v_flat) - min(v_flat) 

        return diff > theta

    def _gierer_mienhardt(self, a_update, h_update):
        """
        1st order Euler step
        """
        # internal domain
        a_view = self.v[1:-1, 1:-1]
        h_view = self.u[1:-1, 1:-1]

        # advance state (Euler step)
        # print(np.max(a_view), np.max(h_view))
        a2 = np.power(a_view, 2)
        ah2 = h_view * (1 + self.kappa * a2)
        a2_ah2 = a2 / ah2 

        a_update += self.rho * (a2_ah2 - self.mu * a_view) # Gierer-Mienhardt
        h_update += self.rho * (a2 - self.nu * h_view)# Gierer-Mienhardt

        # np.clip(h_view, 0.001, 10, out=h_view)
        # np.clip(a_view, 0.001, 10, out=a_view)    


    def _euler(self):
        """
        1st order Euler step
        """

        # advance state (Euler step)
        # print(np.max(v_view), np.max(u_view))

        v_update = self.fs * self.laplacian(self.v) # Diffusion
        u_update = self.fa * self.laplacian(self.u) # Diffusion
        updates = np.array([v_update, u_update])

        if self.rd_type.GIERER_MIENHARDT:
            self._gierer_mienhardt(v_update, u_update)
        if self.rd_type.GRAY_SCOTT:
            # updates += np.array(gray_scott_np(self.F, self.k, self.v_view, self.u_view))
            updates += np.array(self.grey_scott_sim.simulate())[:, 1:-1, 1:-1]
            # print('Updates', updates, 'V', self.v_view, sep='\n')

        if self.rd_type.GENERALIZED:
            updates += np.array(self.generalized_sim.simulate())[:, 1:-1, 1:-1]
            # updates += np.array(generalized_np(self.rho_np, self.kap_np, self.v_view, self.u_view))

        v_update, u_update = updates[0], updates[1]
        self.v_view += self.dt * v_update
        self.u_view += self.dt * u_update
        # np.clip(u_view, 0.001, 10, out=u_view)
        # np.clip(v_view, 0.001, 10, out=v_view)


    def _heun(self):
        """
        2nd order Heun's method
        """
        # internal domain
        u_view = self.u[1:-1, 1:-1]
        v_view = self.v[1:-1, 1:-1]
        u_vtmp = self.u_tmp[1:-1, 1:-1]
        v_vtmp = self.v_tmp[1:-1, 1:-1]

        # 1st stage
        uv2 = u_view * v_view**2
        u_rhs1 = self.fa * self.laplacian(self.u) - uv2 + self.F * (1 - u_view)
        v_rhs1 = self.fs * self.laplacian(self.v) + uv2 - (self.F + self.k) * v_view
        u_vtmp = u_view + self.dt * u_rhs1
        v_vtmp = v_view + self.dt * v_rhs1
        self.update_ghosts(self.u_tmp)
        self.update_ghosts(self.v_tmp)

        # 2nd stage
        uv2 = u_vtmp * v_vtmp**2
        u_rhs2 = self.fa * self.laplacian(self.u_tmp) - uv2 + self.F * (1 - u_vtmp)
        v_rhs2 = self.fs * self.laplacian(self.v_tmp) + uv2 - (self.F + self.k) * v_vtmp
        u_view += 0.5 * self.dt * (u_rhs1 + u_rhs2)
        v_view += 0.5 * self.dt * (v_rhs1 + v_rhs2)


    def _dump(self, step, time, dirichlet_vis=False, *, both=False, save=False):
        """
        Dump snapshot

        Arguments:
            step: step ID
            time: current time
            both: if true, dump contours for both species U and V
        """
        if np.max(self.v) > 10 or True:
            print('Max v', np.max(self.v))

        V = (255 * self.v[1:-1, 1:-1]).astype(np.uint8)
        grad = [l**2 for l in np.gradient(self.v[1:-1, 1:-1])]
        grad = grad[0] + grad[1]
        grad = 255 * grad / np.max(grad)
        grad = grad.astype(np.uint8)
        if dirichlet_vis:
            image = im.fromarray(np.append(V, grad, 1))      
        else:
            image = im.fromarray(V)    
        if save: 
           image.save(os.path.join(self.outdir, self.name + f"_frame_{self.dump_count:06d}.png"))
        return image


    def _old_dump(self, step, time, *, both=False):
        plt.switch_backend('Agg') 

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        x = self.x[1:-1, 1:-1]
        y = self.y[1:-1, 1:-1]
        U = self.u[1:-1, 1:-1]
        V = self.v[1:-1, 1:-1]
        if both:
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            cs0 = ax[0].contourf(x, y, U, levels=50, cmap='jet')
            cs1 = ax[1].contourf(x, y, V, levels=50, cmap='jet')
            fig.suptitle(f"time = {time:e}")
            lim = (self.x0, self.x1)
            species = ("U", "V")
            for a, l in zip(ax, species):
                a.set_title(f"Species {l}")
                a.set_xlabel("x")
                a.set_ylabel("y")
                a.set_xlim(lim)
                a.set_ylim(lim)
                a.set_aspect('equal')
        else: # only species V
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.contourf(x, y, V, levels=50, cmap='jet')
            fig.suptitle(f"time = {time:e}")
            lim = (self.x0, self.x1)
            ax.set_title(f"Species V")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
        fig.savefig(os.path.join(self.outdir, self.name + f"_frame_{self.dump_count:06d}.png"), dpi=400)
        plt.close(fig)
        self.dump_count += 1


    def _render_frames(self):
        cmd = ['ffmpeg', '-framerate', '24', '-i',
                os.path.join(self.outdir, 'frame_%06d.png'), '-b:v', '90M',
                '-vcodec', 'mpeg4', os.path.join(self.outdir, self.name + '.mp4')]
        sp.run(cmd)


    def _random_IC(self):
        """
        Random initial condition
        """
        dim = self.u.shape
        self.u = np.ones(dim) / 2 + 0.5 * np.random.uniform(0, 1, dim)
        self.v = np.ones(dim) / 4 + 0.5 * np.random.uniform(0, 1, dim)

    def _trefethen_IC(self):
        """
        Initial condition for the example used by N. Trefethen at
        https://www.chebfun.org/examples/pde/GrayScott.html
        """
        x = self.x[1:-1, 1:-1]
        y = self.y[1:-1, 1:-1]
        self.u[1:-1, 1:-1] = 1 - np.exp(-80*((x+0.05)**2 + (y+0.05)**2))
        self.v[1:-1, 1:-1] = np.exp(-80*((x-0.05)**2 + (y-0.05)**2))


    @staticmethod
    def laplacian(a):
        """
        Discretization of Laplacian operator

        Arguments:
            a: 2D array
        """
        return a[2:, 1:-1] + a[1:-1, 2:] + a[0:-2, 1:-1] + a[1:-1, 0:-2] - 4 * a[1:-1, 1:-1]

    @staticmethod
    def update_ghosts(v):
        """
        Apply periodic boundary conditions

        Arguments:
            v: 2D array
        """
        v[0, :] = v[-2, :]
        v[:, 0] = v[:, -2]
        v[-1, :] = v[1, :]
        v[:, -1] = v[:, 1]
