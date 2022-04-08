#!/bin/bash

python3 run_simulation.py --param_search \
-t 1 -T 3000 \
-p_a .5 .5 1 \
-p_a0 0 0 1 \
-u_a .5 .5 1 \
-p_h .5 .5 1 \
-p_h0 0 0 1 \
-u_h .45 .45 1 \
-kappa .54 .54 1 \
--dirichlet_vis