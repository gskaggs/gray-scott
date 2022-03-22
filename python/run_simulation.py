#!/usr/bin/env python3
# File       : run_simulation.py
# Created    : Sat Jan 30 2021 09:13:51 PM (+0100)
# Description: Gray-Scott driver.  Use the --help argument for all options
# Copyright 2021 ETH Zurich. All Rights Reserved.
import argparse

# the file gray_scott.py must be in the PYTHONPATH or in the current directory
from gray_scott import GrayScott
from ga import Chromosome


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


def param_search(args):
    """
    Searchers the space of parameters for Turing patterns
    """
    F0, F1, k0, k1 = 0.01, .11, 0.04, .08
    Nf, Nk = 10, 4   # We'll have Nf * Nk simulations
    df, dk = (F1 - F0) / Nf, (k1 - k0) / Nk

    num_successes = 0
    successul_params = []
    for i in range(Nf):
        for j in range(Nk):
            F, k = round(F0 + i * df, 3), round(k0 + j * dk, 3)
            print(f"Beginning sim: F={F}, k={k}")

            sim = GrayScott(F=F, kappa=k, movie=False, outdir=".", name=f"{F}_{k}")
            pattern, _ = sim.integrate(0, 2000, dump_freq=args.dump_freq, report=250, should_dump=False)

            if pattern:
                num_successes += 1
                successul_params.append((F, k))

    print(f"Param search terminated with {num_successes} turing patterns")
    for params in successul_params:
        print(f"F={params[0]}, k={params[1]}")


def genetic_algorithm(args):
    F0, F1, k0, k1 = 0.01, .11, 0.04, .08
    Nf, Nk = 1, 1   # We'll have Nf * Nk chromosomes
    N = Nf * Nk
    df, dk = (F1 - F0) / Nf, (k1 - k0) / Nk

    chromosomes = []
    for i in range(Nf):
        for j in range(Nk):
            F, k = round(F0 + i * df, 3), round(k0 + j * dk, 3)
            chromosomes.append(Chromosome(F, k))

    num_iters = 3
    for i in range(num_iters):
        print(f"GA Iteration {i} of {num_iters}")
        for c in chromosomes:
            F, k = c.F, c.k
            sim = GrayScott(F=F, kappa=k, movie=False, outdir=".", name=f"{F}_{k}")
            pattern, latest = sim.integrate(0, 2000, dump_freq=args.dump_freq, report=250, should_dump=False) 
            c.set_fitness(latest)

        chromosomes.sort(key=lambda c: -c.fitness) # sorted by decreasing fitness
        for j in range(N//2):
            print(j, chromosomes[j].F)
            chromosomes[2*j] = chromosomes[j].crossover(chromosomes[j+1])
            chromosomes[2*j].mutate()

    num_successes = 0
    successul_params = []
    print("GA: Checking for turing patterns")
    for c in chromosomes:
        F, k = c.F, c.k
        sim = GrayScott(F=F, kappa=k, movie=False, outdir=".", name=f"{F}_{k}")
        pattern, latest = sim.integrate(0, 2000, dump_freq=args.dump_freq, report=250, should_dump=False)
        if pattern:
            num_successes += 1
            successul_params.append((F, k))

    print(f"Genetic algorithm terminated with {num_successes} turing patterns out of {N} chromosomes")
    for params in successul_params:
        print(f"F={params[0]}, k={params[1]}")



def main():
    args, _ = parse_args()

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
