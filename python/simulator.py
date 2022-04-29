# File       : gray_scott.py
# Created    : Sat Jan 30 2021 05:12:47 PM (+0100)
# Description: Gray-Scott reaction-diffusion
# Copyright 2021 ETH Zurich. All Rights Reserved.
import os
from pickletools import uint8
import numpy as np
from PIL import Image as im
from python.enum_util import RdType
from python.core_simulator import CoreSimulatorGpu
from python.core_simulator_np import CoreSimulatorNp
from python.colors import viridis

class ReactionDiffusionSimulator:
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
                 movie=False,
                 outdir='.',
                 name='',
                 use_cpu=True,
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

        # grid spacing
        L = x1 - x0
        dx = L / N

        # intermediates
        # fs refers to dv, fa refers to du
        self.du = Du / dx**2
        self.dv = Dv / dx**2
        self.dt = Fo * dx**2 / (4*max(Du, Dv))

        self.use_cpu = use_cpu
        self.rd_type = RdType(rd_types)

        if self.rd_type.GRAY_SCOTT:
            self.dt = .5
            self.du = 0.32768
            self.dv = 0.16384

        if self.rd_type.GIERER_MIENHARDT:
            self.dt = .05
            self.du = 2
            self.dv = .1
            # print('dt', self.dt)
            # print('Du', self.du, '\nDv', self.dv)

        if self.rd_type.GENERALIZED:
            self.dt = 0.01
            self.du = 1
            self.dv = 1

        self.dv = chromosome.dv 
        self.du = chromosome.du

        # print(f'du {self.du}, dv {self.dv}')
        
        self.rho_np, self.kap_np = self.gen_params[0], self.gen_params[1]

        # nodal grid (+ghosts)
        x = np.linspace(x0-dx, x1+dx, Nnodes+2)
        y = np.linspace(x0-dx, x1+dx, Nnodes+2)
        self.x, self.y = np.meshgrid(x, y)

        # initial condition
        self.u = np.zeros((len(x), len(y))).astype(np.float64)
        self.v = np.zeros((len(x), len(y))).astype(np.float64)

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

        if not self.use_cpu: 
            self.simulator = CoreSimulatorGpu(self.v, self.u, self.rho_np, self.kap_np, self.F, self.k, \
                self.rho, self.kappa, self.mu, self.nu, self.dv, self.du, rd_types)
        else:
            self.simulator = CoreSimulatorNp(self.v, self.u, self.rho_np, self.kap_np, self.F, self.k, \
                self.rho, self.kappa, self.mu, self.nu, self.dv, self.du, rd_types)

    def integrate(self, final_t, *, dirichlet_vis=False, fitness='pattern'):
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
        latest = 0
        num_iters = int(final_t / self.dt)

        self.v, self.u = self.simulator.simulate(self.dt, num_iters)

        pattern = self._check_pattern()
        image = self._dump(dirichlet_vis)
        
        if fitness=='dirichlet':
            latest = self._dirichlet()
            if np.isnan(latest):
                latest = -1

        return pattern, latest, image

    def _dirichlet(self):
        '''
        Computes a proxy for the Dirichlet energy of v.
        This is used as a heuristic for how complex the pattern is.
        '''
        v_view = self.v[1:-1, 1:-1]

        max_v = np.max(v_view)
        if max_v < 10e-7 or max_v > 4.5:
            return 0

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

    def to_color_array(self, array):
        max_val = np.max(array)
        if max_val < 10e-7:
            return (255*np.array([[viridis[0]] * array.shape[0]]* array.shape[1])).astype(np.uint8)

        array = array / max_val
        np.clip(array, 0, 1, out=array)

        result = 255*np.array([[viridis[int(x * (len(viridis)-1))] for x in row] for row in array])
        result = result.astype(np.uint8)
        return result

    def _dump(self, dirichlet_vis=False, *, both=False, save=False):
        """
        Dump snapshot

        Arguments:
            step: step ID
            time: current time
            both: if true, dump contours for both species U and V
        """

        # print('Max v', np.max(self.v))

        V = self.to_color_array(self.v[1:-1, 1:-1])

        grad = [l**2 for l in np.gradient(self.v[1:-1, 1:-1])]
        grad = grad[0] + grad[1]
        grad = self.to_color_array(grad)

        if dirichlet_vis:
            image = im.fromarray(np.append(V, grad, 1), 'RGB')      
        else:
            image = im.fromarray(V, 'RGB')    
        if save: 
           image.save(os.path.join(self.outdir, self.name + f"_frame_{self.dump_count:06d}.png"))
        return image


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
