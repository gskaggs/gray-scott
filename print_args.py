#!/usr/bin/env python3

import argparse
from ast import parse
import pickle
import os
import numpy as np
from python.simulator import ReactionDiffusionSimulator
from python.init_utils import load_args
from PIL import Image as im

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_file', type=str, help='')
    parser.add_argument('--target_chromosome', type=int, default=None)
    return parser.parse_known_args()

def load_args(resume_file):
    assert os.path.exists(resume_file), 'Resume file doesn\'t exist.'
    with open(resume_file, 'rb') as file: 
        return pickle.load(file)

def print_all_chromosomes(chromosomes, args, cur_iter):
    print('Cur iteration', cur_iter)
    print('Args', args)
    for i in range(len(chromosomes)):
        c = chromosomes[i]
        print('Chromosome', i)
        print(f'Fitness={c.fitness}')
        print(c._rd_params)
        print(c.du, c.dv)

def make_sim_image(v):
    N = len(v)
    result = np.zeros((N, N, 3))
    max_v = np.max(v)
    if max_v < 10e-7:
        return result.astype(np.uint8)

    v = v / max_v
    
    def color(value):
        theta = .5
        color1 = (255, 0, 0)
        color2 = (255, 255, 255)
        return color1 if value < theta else color2

    color_array = np.array([[color(value) for value in row] for row in v]).astype(np.uint8)
    return im.fromarray(color_array, 'RGB')    

def print_target_chromsome(c, c_idx, args):
    print('Chromosome', c_idx)
    print(f'Fitness={c.fitness}')
    print('Rd model', args.rd)
    print(c._rd_params)
    if 'generalized' in args.rd:
        print(c.gen_params)
    
    sim = ReactionDiffusionSimulator(chromosome=c, movie=False, outdir="./garbage", use_cpu=args.use_cpu, rd_types=args.rd)
    sim.integrate(args.end_time, dirichlet_vis=args.dirichlet_vis, fitness=args.fitness) 
    
    image_file = './results/targeted/' + '_'.join(sorted(args.rd)) + '.png'
    count = 1
    while os.path.exists(image_file):
        image_file = './results/targeted/' + '_'.join(sorted(args.rd)) + f'_v{count}' + '.png'
        count += 1

    image = make_sim_image(sim.v)
    image.show()
    image.save(image_file)
    print('Saved file at ' + image_file)

def main():
    args, _ = parse_args()
    target_chromosome = args.target_chromosome
    chromosomes, cur_iter, args = load_args(args.resume_file)

    if not target_chromosome:
        print_all_chromosomes(chromosomes, args, cur_iter)
    else:
        c_idx = target_chromosome-1
        assert c_idx < len(chromosomes) and c_idx >= 0, 'invalid targeted chromosome'
        c = chromosomes[c_idx]
        print_target_chromsome(c, c_idx, args)


if __name__ == "__main__":
    main()