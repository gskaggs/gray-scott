#!/usr/bin/env python3

import argparse
from ast import parse
import pickle
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_file', type=str, help='')
    return parser.parse_known_args()

def load_args(resume_file):
    assert os.path.exists(resume_file), 'Resume file doesn\'t exist.'
    with open(resume_file, 'rb') as file: 
        return pickle.load(file)

def main():
    args, _ = parse_args()
    chromosomes, cur_iter, args = load_args(args.resume_file)

    print('Cur iteration', cur_iter)
    print('Args', args)
    for i in range(len(chromosomes)):
        c = chromosomes[i]
        print('Chromosome', i)
        print(f'Fitness={c.fitness}')
        print(c._rd_params)

if __name__ == "__main__":
    main()