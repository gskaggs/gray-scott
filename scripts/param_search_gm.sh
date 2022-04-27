#!/bin/bash

python3 sim_driver_tui.py --param_search \
-t 6 -T 3000 \
-p_a 0 1 5 \
-p_a0 0 0 1 \
-u_a 0 .5 2 \
-p_h 0 1 5 \
-p_h0 0 0 1 \
-u_h 0 .5 2 \
-kappa 0 .54 2 \
--dirichlet_vis