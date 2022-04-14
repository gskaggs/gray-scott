#!/bin/bash

python3 run_simulation.py --genetic_algorithm -fitness dirichlet -t 6 -T 200 -num_iters 3 \
-rd gierer_mienhardt \
-rho .5 \
-mu 1 \
-nu .9 \
-kappa .1 .8 6 \
--dirichlet_vis