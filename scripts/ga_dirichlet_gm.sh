#!/bin/bash

python3 sim_driver_tui.py --genetic_algorithm -fitness dirichlet -t 6 -T 200 -num_iters 2 \
-rd gierer_mienhardt \
-rho .5 \
-mu 1 \
-nu .9 \
-kappa .1 .8 3 \
--dirichlet_vis