#!/bin/bash

python3 ./sim_driver_tui.py --genetic_algorithm \
 -fitness dirichlet  \
 -num_iters 20 \
 -T 2000 \
 -num_individuals 30 \
 -rd gierer_mienhardt