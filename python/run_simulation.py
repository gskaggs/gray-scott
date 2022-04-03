#!/usr/bin/env python3
# File       : run_simulation.py
# Created    : Sat Jan 30 2021 09:13:51 PM (+0100)
# Description: Gray-Scott driver.  Use the --help argument for all options
# Copyright 2021 ETH Zurich. All Rights Reserved.
import threading
import matplotlib
matplotlib.use('Agg')

import argparse
import pickle
import os

# the file gray_scott.py must be in the PYTHONPATH or in the current directory
from gray_scott import GrayScott
from ga import Chromosome
from ga import apply_fitness_function
from thread_util import run_threads
from thread_util import ThreadSafeIterable
from process_util import start_processes
from process_util import ProcessSafeIterable
from process_util import end_processes
from PIL import Image as im
from PIL import ImageFont
from PIL import ImageDraw 
from multiprocessing import Queue
from datetime import timedelta
import time
import psutil

def parse_args():
    """
    Driver arguments.  These are passed to the GrayScott class
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-Nf', default=20, type=int, help='')
    parser.add_argument('-Nk', default=10, type=int, help='')
    parser.add_argument('-F0', default=0.01, type=float, help='')
    parser.add_argument('-F1', default=0.11, type=float, help='')
    parser.add_argument('-k0', default=0.04, type=float, help='')
    parser.add_argument('-k1', default=0.08, type=float, help='')
    parser.add_argument('-num_iters', default=10, type=int, help='How many generations of ga to run.')
    parser.add_argument('-fitness', default='dirichlet', type=str, help='The kind of fitness function to use.')
    parser.add_argument('-rd', default='gray-scott', type=str, help='The kind of reaction diffussion equation to use.')

    parser.add_argument('-F', '--feed_rate', default=0.04, type=float, help='Feed rate F')
    parser.add_argument('-k', '--death_rate', default=0.06, type=float, help='Death rate kappa')
    parser.add_argument('-T', '--end_time', default=3500, type=float, help='Final time')
    parser.add_argument('-d', '--dump_freq', default=100, type=int, help='Dump frequency (integration steps)')
    parser.add_argument('--demo', action='store_true', help='Run demo (https://www.chebfun.org/examples/pde/GrayScott.html)')
    parser.add_argument('--param_search', action='store_true', help='Run param search')
    parser.add_argument('--resume_file', default='resume.pkl', type=str, help='Where intermediate program values should be stored for genetic algorithm')
    parser.add_argument('--genetic_algorithm', action='store_true', help='Run genetic algorithm')
    parser.add_argument('--movie', action='store_true', help='Create a movie (requires ffmpeg)')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory')
    parser.add_argument('--should_dump', action='store_true', default=True, help='Actually create png files during simulation')
    parser.add_argument('--name', default='', type=str, help='Name of the simulation, used to save the results')
    parser.add_argument('-t', '--num_processes', default=6, type=int, help='Number of threads for the simulation')
    return parser.parse_known_args()


def demo(args):
    """
    Reproduces the example at https://www.chebfun.org/examples/pde/GrayScott.html
    Pass the --demo option to the driver to run this demo.
    """
    rolls = GrayScott(F=0.04, kappa=0.06, movie=False, outdir=".", name="Rolls")
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
        F, k = c.F, c.k
        sim = GrayScott(F=F, kappa=k, movie=False, outdir="./garbage", name=f"{F}_{k}")
        pattern, latest, image = sim.integrate(0, args.end_time, dump_freq=100, report=250, should_dump=False, fitness=args.fitness) 
        c.set_fitness(latest)
        c.set_pattern(pattern)
        c.set_image(image)
        modified.put(c)

    modified.put('DONE')


def present_chromosomes(chromosomes, cur_iter, args):
    Nf, Nk = args.Nf, args.Nk
    img_text = [['' for _ in range(Nk)] for _ in range(Nf)]
    images   = [[None for _ in range(Nk)] for _ in range(Nf)]
    successful_params = []

    for i in range(Nf):
        for j in range(Nk):
            cur = Nk*i+j
            c = chromosomes[cur]
            F, k = round(c.F, 4), round(c.k, 4) 
            img_text[i][j] = f'#{cur+1}: F={F}, K={k}'
            images[i][j]   = chromosomes[cur].image
            if c.pattern:
                successful_params.append((F, k))

    grid = create_img_grid(images, img_text)
    grid.save(f'./results/{args.rd}/{args.fitness}_{Nf}_{Nk}_{args.end_time}_{cur_iter}.png')

    sim_type = 'Paramater search' if args.param_search else 'Genetic algorithm'
    if cur_iter == args.num_iters or args.param_search:
        print(f"{sim_type} terminated with {len(successful_params)} turing patterns out of {len(chromosomes)} chromosomes")
        for params in successful_params:
            print(f"F={params[0]}, k={params[1]}")

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
    F0, F1, k0, k1 = args.F0, args.F1, args.k0, args.k1
    Nf, Nk = args.Nf, args.Nk  # We'll have Nf * Nk chromosomes
    df, dk = (F1 - F0) / Nf, (k1 - k0) / Nk

    chromosomes = []
    for i in range(Nf):
        for j in range(Nk):
            F, k = round(F0 + i * df, 3), round(k0 + j * dk, 3)
            chromosomes.append(Chromosome(F, k))

    return chromosomes


def resume_ga(args):
    chromosomes = init_chromosomes(args)

    cur_iter, Nf, Nk, num_iters = 1, args.Nf, args.Nk, args.num_iters
    
    if os.path.exists(args.resume_file):
        with open(args.resume_file, 'rb') as file:
            chromosomes, Nf, Nk, cur_iter, num_iters = pickle.load(file)
            chromosomes = apply_fitness_function(chromosomes, 'user_input')

    else:
        with open(args.resume_file, 'w') as file:
            # Just making the file for now
            pass

    chromosomes = run_generation(chromosomes, cur_iter, args)

    with open(args.resume_file, 'wb') as file:
        pickle.dump((chromosomes, Nf, Nk, cur_iter+1, num_iters), file)


def genetic_algorithm(args):
    chromosomes = init_chromosomes(args)

    num_iters = args.num_iters

    for cur_iter in range(num_iters):
        chromosomes = run_generation(chromosomes, cur_iter, args)
        chromosomes = apply_fitness_function(chromosomes, 'default')


def param_search(args):
    chromosomes = init_chromosomes(args)
    cur_iter = 1
    chromosomes = run_generation(chromosomes, cur_iter, args)


def make_output_dirs(args):
    dirs = ['garbage', 'results', f'results/{args.rd}']
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def main():
    args, _ = parse_args()

    print(f'thread count per core: {psutil.cpu_count() // psutil.cpu_count(logical=False)}')
    print(f"Num cores = {psutil.cpu_count(logical=False)}")
    print('Num_processes = ', args.num_processes)

    make_output_dirs(args)

    if args.demo:
        demo(args)
        return

    if args.param_search:
        param_search(args)
        return

    if args.genetic_algorithm:
        resume_ga(args)
        #genetic_algorithm(args)
        return

    print('Sim type not specified')

if __name__ == "__main__":
    main()
