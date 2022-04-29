#!/bin/bash

python3 ./sim_driver_tui.py --genetic_algorithm \
 -fitness dirichlet  \
 -num_iters 10 \
 -T 500 \
 -num_individuals 20 \
 -rd gierer_mienhardt