from python.ga import Chromosome
from multiprocessing import Queue
import itertools
import numpy as np
import os

param_names = ['F', 'k', 'rho', 'mu', 'nu', 'kappa']
ranges = {'F': (.01, .11), 'k': (0.01, .11), 'rho': (.3, .7), 'nu': (.7, 1.1), 'kappa': (.1, .8)}

def load_args(resume_file):
    assert os.path.exists(resume_file), 'Resume file doesn\'t exist.'
    with open(resume_file, 'rb') as file: 
        return pickle.load(file)


def init_gen_params():
    '''
    Axis 1: u, v
    Axis 2: Terms 1, 2, 3
    Axis 3: v pow
    Axis 4: u pow
    '''
    rho = (2 * np.random.rand(2, 3, 3, 3) - 1).round(decimals=3)
    kap = (2 * np.random.rand(2, 3, 3, 3) - 1).round(decimals=3)

    return np.array([rho, kap]).astype(np.float64)

def init_rd_params():
    rd_params = {}
    for k in ranges:
        rd_params[k] = np.random.uniform(*ranges[k])
    return rd_params

def init_dif_genes(rd):
    if 'gierer_mienhardt' in rd:
        du = [1.5, 2.5]
        dv = [.05, .15]
    elif 'gray_scott' in rd:
        du = [.27, .37]
        dv = [.1, .2]
    elif 'generalized' in rd:
        du = [.5, 1.5]
        dv = [.5, 1.5]
    else:
        raise NotImplementedError
    
    return np.random.uniform(*du), np.random.uniform(*dv)

def init_chromosomes(args):
    result = []

    for _ in range(args.num_individuals):
        rd_params = init_rd_params()
        gen_params = init_gen_params()
        du, dv = init_dif_genes(args.rd)
        c = Chromosome(rd_params, du, dv, gen_params)
        result.append(c)
    
    return result

def init_chromosomes_old(args):
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
        
        for _ in range(args.num_generalized):
            gen_params = init_gen_params()
            chromosomes.append(Chromosome(rd_params, du, dv, gen_params))

        if 'generalized' not in args.rd:
            chromosomes.append(Chromosome(rd_params, du, dv))

    return chromosomes

def prep_sim(chromosomes, cur_iter, args):
    if args.param_search:
        print('Beginning param search')
    else:
        if not args.test_speed:
            print(f"GA Iteration {cur_iter} of {args.num_iters}")

    for _ in range(args.num_processes):
        chromosomes.append("DONE")

    q, modified = Queue(), Queue()
    for c in chromosomes:
        q.put(c)
    chromosomes = q

    return chromosomes, modified