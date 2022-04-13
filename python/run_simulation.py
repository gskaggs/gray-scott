#!/usr/bin/env python3
# File       : run_simulation.py
# Created    : Sat Jan 30 2021 09:13:51 PM (+0100)
# Description: Gray-Scott driver.  Use the --help argument for all options
# Copyright 2021 ETH Zurich. All Rights Reserved.
from email.policy import default
from fileinput import filename
import threading
import matplotlib
matplotlib.use('Agg')

import argparse
import pickle
import os
import itertools
import numpy as np

# the file gray_scott.py must be in the PYTHONPATH or in the current directory
from gray_scott import GrayScott
from ga import Chromosome
from ga import apply_fitness_function
from thread_util import run_threads
from thread_util import ThreadSafeIterable
from process_util import start_processes
from process_util import ProcessSafeIterable
from process_util import end_processes
from print_args import load_args
from PIL import Image as im
from PIL import ImageFont
from PIL import ImageDraw 
from multiprocessing import Queue
from datetime import timedelta
import time
import psutil

param_names = ['F', 'k', 'rho', 'mu', 'nu', 'kappa']

def parse_args():
    """
    Driver arguments.  These are passed to the GrayScott class
    """
    parser = argparse.ArgumentParser()
    global param_names
    for name in param_names:
        parser.add_argument('-' + name, default=[0, 0, 1], type=float, nargs='+')

    parser.add_argument('-num_iters', default=1, type=int, help='How many generations of ga to run.')
    parser.add_argument('-fitness', default='dirichlet', type=str, help='The kind of fitness function to use.')
    parser.add_argument('-rd', default='gray-scott', type=str, help='The kind of reaction diffussion equation to use.')

    parser.add_argument('-T', '--end_time', default=3500, type=float, help='Final time')
    parser.add_argument('-d', '--dump_freq', default=100, type=int, help='Dump frequency (integration steps)')
    parser.add_argument('--demo', action='store_true', help='Run demo (https://www.chebfun.org/examples/pde/GrayScott.html)')
    parser.add_argument('--param_search', action='store_true', help='Run param search')
    parser.add_argument('--resume_file', default='resume.pkl', type=str, help='Where intermediate program values should be stored for genetic algorithm')
    parser.add_argument('--resume', action='store_true', help='Restart a simulation with a given setup stored in --resume_file')
    parser.add_argument('--genetic_algorithm', action='store_true', help='Run genetic algorithm')
    parser.add_argument('--movie', action='store_true', help='Create a movie (requires ffmpeg)')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory')
    parser.add_argument('--should_dump', action='store_true', default=False, help='Actually create png files during simulation')
    parser.add_argument('--dirichlet_vis', action='store_true', default=False, help='Visualize gradient side by side with sims')
    parser.add_argument('--name', default='', type=str, help='Name of the simulation, used to save the results')
    parser.add_argument('-t', '--num_processes', default=6, type=int, help='Number of threads for the simulation')
    return parser.parse_known_args()


def demo(args):
    """
    Reproduces the example at https://www.chebfun.org/examples/pde/GrayScott.html
    Pass the --demo option to the driver to run this demo.
    """
    rolls = GrayScott(F=0.04, k=0.06, movie=False, outdir=".", name="Rolls")
    rolls.integrate(0, 3500, dump_freq=args.dump_freq, should_dump=True)
    
    gs = GrayScott(F=args.feed_rate, kappa=args.death_rate, movie=args.movie, outdir=args.outdir, name=args.name)
    gs.integrate(0, args.end_time, dump_freq=args.dump_freq, should_dump=args.should_dump)


def create_img_grid(images, text):
    W, H = images[0][0].width, images[0][0].height
    rows, cols = len(images), len(images[0])
    grid = im.new("L", (rows*W, cols*H))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    
    for i in range(rows):
        for j in range(cols):
            x, y = i*W, j*H
            grid.paste(images[i][j], (x, y))
            draw.rectangle((x,y,x+W,y+10),fill=(0))
            draw.text((x, y),text[i][j],(255),font=font)

    return grid


def process_function_ga(chromosomes, modified, args):
    while True:
        c = chromosomes.get()
        if c == 'DONE':
            break

        sim = GrayScott(chromosome=c, movie=False, outdir="./garbage", rd=args.rd)
        pattern, latest, image = sim.integrate(0, args.end_time, dump_freq=100, report=250, should_dump=args.should_dump, dirichlet_vis=args.dirichlet_vis, fitness=args.fitness) 
        c.set_fitness(latest)
        c.set_pattern(pattern)
        c.set_image(image)
        modified.put(c)

    modified.put('DONE')


def grid_w_h(chromosomes):
    N = len(chromosomes)
    for W in range(N):
        if W**2 >= N and N % W == 0:
            return W, N//W

    return N, 1

def present_chromosomes(chromosomes, cur_iter, args):
    W, H = grid_w_h(chromosomes)
    img_text = [['' for _ in range(H)] for _ in range(W)]
    images   = [[None for _ in range(H)] for _ in range(W)]
    successful_params = []

    for i in range(W):
        for j in range(H):
            cur = H*i+j
            c = chromosomes[cur]
            F, k, fitness = round(c.F, 4), round(c.k, 4), round(c.fitness, 2)
            img_text[i][j] = f'#{cur+1}: F={F}, K={k}'
            if args.dirichlet_vis:
                img_text[i][j] = img_text[i][j] + f', Fit={fitness}'
            images[i][j]   = chromosomes[cur].image
            if c.pattern:
                successful_params.append(c.get_params())

    sim_type = 'Paramater search' if args.param_search else 'Genetic algorithm'
    last_gen = cur_iter == args.num_iters or args.param_search
    if last_gen:
        print(f"{sim_type} terminated with {len(successful_params)} turing patterns out of {len(chromosomes)} chromosomes")
        for params in successful_params:
            print(params)

    grid = create_img_grid(images, img_text)
    sim_type = 'param_search' if args.param_search else args.fitness
    sim_id = f'./results/{args.rd}/{sim_type}_{len(chromosomes)}_{args.end_time}_{cur_iter}'
    img_file, param_file = sim_id + '.png', sim_id + '.pkl'

    count = 1
    while os.path.exists(img_file) or os.path.exists(param_file):
        img_file, param_file = sim_id + f'({count})' + '.png', sim_id + f'({count})' + '.pkl'
        count += 1

    if last_gen:
        try:
            grid.show()
        except:
            pass

    print('Saving simulation image at', img_file)
    grid.save(img_file)

    with open(param_file, 'wb') as file:
        pickle.dump((chromosomes, cur_iter, args), file)

def prep_sim(chromosomes, cur_iter, args):
    if args.param_search:
        print('Beginning param search')
    else:
        print(f"GA Iteration {cur_iter} of {args.num_iters}")

    for _ in range(args.num_processes):
        chromosomes.append("DONE")

    q, modified = Queue(), Queue()
    for c in chromosomes:
        q.put(c)
    chromosomes = q

    return chromosomes, modified

def run_generation(chromosomes, cur_iter, args):
    # Prepare process safe queues
    chromosomes, modified = prep_sim(chromosomes, cur_iter, args)

    # Do the simulations
    start = time.time()
    processes = start_processes(args.num_processes, process_function_ga, (chromosomes, modified, args))
    chromosomes = end_processes(processes, modified, args.num_processes)
    end = time.time()
    print(f'Generation {cur_iter} time taken {timedelta(seconds=end-start)}')

    # Save the results
    chromosomes.sort(key=lambda c: -c.fitness) # sorted by decreasing fitness
    present_chromosomes(chromosomes, cur_iter, args)

    return chromosomes

def init_chromosomes(args):
    global param_names
    args_map = vars(args)
    param_bounds = [args_map[param_name] for param_name in param_names]
    
    for bounds in param_bounds:
        # If only one value given, that's the only value the sim will use.
        if len(bounds) == 1:
            bounds.append(bounds[0])
            bounds.append(1)

        bounds[2] = int(bounds[2]) # So that np.linspace works

    # bounds[0] is first value in range, bounds[1] is last, and 
    # bounds[2] is the number of values to take on
    param_bounds = [np.linspace(*bounds) for bounds in param_bounds]
    chromosomes = []

    for param_combo in itertools.product(*param_bounds):
        rd_params = {}
        for i in range(len(param_names)):
            rd_params[param_names[i]] = param_combo[i] 
        chromosomes.append(Chromosome(rd_params))

    return chromosomes


def resume_ga(args):
    chromosomes = init_chromosomes(args)

    cur_iter, num_iters = 1, args.num_iters
    
    if os.path.exists(args.resume_file):
        with open(args.resume_file, 'rb') as file:
            chromosomes, cur_iter, num_iters = pickle.load(file)
            chromosomes = apply_fitness_function(chromosomes, 'user_input')

    else:
        with open(args.resume_file, 'w') as file:
            # Just making the file for now
            pass

    chromosomes = run_generation(chromosomes, cur_iter, args)

    with open(args.resume_file, 'wb') as file:
        pickle.dump((chromosomes, cur_iter+1, num_iters), file)


def genetic_algorithm(args):
    chromosomes = init_chromosomes(args)

    num_iters = args.num_iters

    for cur_iter in range(1, num_iters+1):
        chromosomes = run_generation(chromosomes, cur_iter, args)
        chromosomes = apply_fitness_function(chromosomes, 'default')


def param_search(args):
    chromosomes = init_chromosomes(args)
    cur_iter = 1
    chromosomes = run_generation(chromosomes, cur_iter, args)


def resume_sim(args):
    chromosomes, cur_iter, args = load_args(args.resume_file)

    if args.param_search:
        param_search(args)
        return

    while cur_iter <= args.num_iters:
        chromosomes = run_generation(chromosomes, cur_iter, args)
        chromosomes = apply_fitness_function(chromosomes, 'default')
        cur_iter += 1


def make_output_dirs(args):
    dirs = ['garbage', 'results', f'results/{args.rd}']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def main():
    args, _ = parse_args()

    print(f'thread count per core: {psutil.cpu_count() // psutil.cpu_count(logical=False)}')
    print(f"Num cores = {psutil.cpu_count(logical=False)}")
    print(f'Num_processes = {args.num_processes}')

    make_output_dirs(args)

    if args.demo:
        demo(args)
        return

    if args.resume:
        resume_sim(args)
        return

    if args.param_search:
        param_search(args)
        return

    if args.genetic_algorithm and args.fitness == 'user':
        resume_ga(args)
        return

    if args.genetic_algorithm:
        genetic_algorithm(args)
        return

    print('Sim type not specified')

if __name__ == "__main__":
    main()
