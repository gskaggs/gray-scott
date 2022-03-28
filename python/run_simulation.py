#!/usr/bin/env python3
# File       : run_simulation.py
# Created    : Sat Jan 30 2021 09:13:51 PM (+0100)
# Description: Gray-Scott driver.  Use the --help argument for all options
# Copyright 2021 ETH Zurich. All Rights Reserved.
import matplotlib
matplotlib.use('Agg')

import argparse

# the file gray_scott.py must be in the PYTHONPATH or in the current directory
from gray_scott import GrayScott
from ga import Chromosome
from ga import apply_fitness_function
from thread_util import run_threads
from thread_util import ThreadSafeIterable
from PIL import Image as im
from PIL import ImageFont
from PIL import ImageDraw 
import psutil

def parse_args():
    """
    Driver arguments.  These are passed to the GrayScott class
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--feed_rate', default=0.04, type=float, help='Feed rate F')
    parser.add_argument('-k', '--death_rate', default=0.06, type=float, help='Death rate kappa')
    parser.add_argument('-T', '--end_time', default=3500, type=float, help='Final time')
    parser.add_argument('-d', '--dump_freq', default=100, type=int, help='Dump frequency (integration steps)')
    parser.add_argument('--demo', action='store_true', help='Run demo (https://www.chebfun.org/examples/pde/GrayScott.html)')
    parser.add_argument('--param_search', action='store_true', help='Run param search')
    parser.add_argument('--genetic_algorithm', action='store_true', help='Run genetic algorithm')
    parser.add_argument('--movie', action='store_true', help='Create a movie (requires ffmpeg)')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory')
    parser.add_argument('--should_dump', action='store_true', help='Actually create png files during simulation')
    parser.add_argument('--name', default='', type=str, help='Name of the simulation, used to save the results')
    parser.add_argument('-t', '--num_threads', default=6, type=int, help='Number of threads for the simulation')
    return parser.parse_known_args()


def demo(args):
    """
    Reproduces the example at https://www.chebfun.org/examples/pde/GrayScott.html
    Pass the --demo option to the driver to run this demo.
    """

    # 1. Rolls
    # rolls = GrayScott(F=0.04, kappa=0.06, movie=True, outdir="demo_rolls")
    # rolls.integrate(0, 3500, dump_freq=args.dump_freq)

    # # 2. Spots
    # spots = GrayScott(F=0.025, kappa=0.06, movie=True, outdir="demo_spots")
    # spots.integrate(0, 3500, dump_freq=args.dump_freq)

    # 1. Rolls
    rolls = GrayScott(F=0.04, kappa=0.06, movie=False, outdir=".", name="Rolls")
    rolls.integrate(0, 3500, dump_freq=args.dump_freq, should_dump=False)

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

def param_search(args):
    """
    Searchers the space of parameters for Turing patterns
    """
    F0, F1, k0, k1 = 0.01, .11, 0.04, .08
    Nf, Nk = 10, 4   # We'll have Nf * Nk simulations
    df, dk = (F1 - F0) / Nf, (k1 - k0) / Nk

    successul_params = []
    images = [[None for _ in range(Nk)] for _ in range(Nf)]
    param_seeds = [(i, j) for i in range(Nf) for j in range(Nk)]
    param_seeds = ThreadSafeIterable(param_seeds)

    def thread_function(param_seeds, successful_params):
        param_seed = param_seeds.next()
        while param_seed is not None:
            i, j = param_seed

            F, k = round(F0 + i * df, 3), round(k0 + j * dk, 3)
            print(f"Beginning sim: F={F}, k={k}")

            sim = GrayScott(F=F, kappa=k, movie=False, outdir="./garbage", name=f"{F}_{k}")
            pattern, _, image = sim.integrate(0, 3500, dump_freq=args.dump_freq, report=250, should_dump=False)
            images[i][j] = image
            if pattern:
                successful_params.append((F, k))

            param_seed = param_seeds.next()

    run_threads(args.num_threads, thread_function, (param_seeds, successul_params))
    num_successes = len(successul_params)

    img_text = [[f'F={round(F0 + i * df, 3)}, K={round(k0 + j * dk, 3)}' for j in range(Nk)] for i in range(Nf)]
    grid = create_img_grid(images, img_text)

    grid.save('param_search.png')
    print(f"Param search terminated with {num_successes} turing patterns")
    for params in successul_params:
        print(f"F={params[0]}, k={params[1]}")


def genetic_algorithm(args):
    F0, F1, k0, k1 = 0.01, .11, 0.04, .08
    Nf, Nk = 5, 4   # We'll have Nf * Nk chromosomes
    N = Nf * Nk
    df, dk = (F1 - F0) / Nf, (k1 - k0) / Nk

    chromosomes = []
    for i in range(Nf):
        for j in range(Nk):
            F, k = round(F0 + i * df, 3), round(k0 + j * dk, 3)
            chromosomes.append(Chromosome(F, k))

    num_successes = 0
    successul_params = []
    num_iters = 10

    for iter in range(num_iters):
        print(f"GA Iteration {iter} of {num_iters}")
        chromosomes = ThreadSafeIterable(chromosomes)

        def thread_function(chromosomes):
            c = chromosomes.next()
            while c is not None:
                F, k = c.F, c.k
                sim = GrayScott(F=F, kappa=k, movie=False, outdir=".", name=f"{F}_{k}")
                pattern, latest, image = sim.integrate(0, 20, dump_freq=args.dump_freq, report=250, should_dump=False) 
                c.set_fitness(latest)
                c.set_pattern(pattern)
                c.set_image(image)

                c = chromosomes.next()

        run_threads(args.num_threads, thread_function, (chromosomes,))
        chromosomes = chromosomes.get_data()

        chromosomes.sort(key=lambda c: -c.fitness) # sorted by decreasing fitness

        img_text = [['' for _ in range(Nk)] for _ in range(Nf)]
        images   = [[None for _ in range(Nk)] for _ in range(Nf)]

        num_successes = 0
        successul_params = []

        for i in range(Nf):
            for j in range(Nk):
                cur = Nk*i+j
                c = chromosomes[cur]
                F, k = round(c.F, 4), round(c.k, 4) 
                img_text[i][j] = f'#{cur+1}: F={F}, K={k}'
                images[i][j]   = chromosomes[cur].image
                if c.pattern:
                    num_successes += 1
                    successul_params.append((F, k))
                    

        grid = create_img_grid(images, img_text)
        grid.save(f'ga_search_iter_{iter}.png')

        # Fitness function
        chromosomes = apply_fitness_function(chromosomes, 'default')
        


    print(f"Genetic algorithm terminated with {num_successes} turing patterns out of {N} chromosomes")
    for params in successul_params:
        print(f"F={params[0]}, k={params[1]}")



def main():
    args, _ = parse_args()

    print(f'thread count per core: {psutil.cpu_count() // psutil.cpu_count(logical=False)}')
    print('Num_threads =', args.num_threads)

    if args.demo:
        demo(args)
        return

    if args.param_search:
        param_search(args)
        return

    if args.genetic_algorithm:
        genetic_algorithm(args)
        return

    gs = GrayScott(F=args.feed_rate, kappa=args.death_rate, movie=args.movie, outdir=args.outdir, name=args.name)
    gs.integrate(0, args.end_time, dump_freq=args.dump_freq, should_dump=args.should_dump)


if __name__ == "__main__":
    main()
