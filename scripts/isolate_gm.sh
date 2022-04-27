#!/bin/bash

python3 sim_driver_tui.py --param_search \
-rd gierer_mienhardt \
-t 6 -T 50 \
-rho .5 \
-mu 1 \
-nu .9 \
-kappa .238 \
--dirichlet_vis \
--use_cpu 