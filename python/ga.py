import numpy as np

class Chromosome:
    def __init__(self, rd_params, gen_params=None):
        self._rd_params = rd_params
        self.gen_params = gen_params if gen_params is not None else np.zeros((2,3,3,3))
        self.F = self.get_param('F')
        self.k = self.get_param('k')

    def get_param(self, param_name):
        return self._rd_params.get(param_name, 0)

    def get_params(self):
        return self._rd_params

    def mutate(self):
        for k in self._rd_params:
            self._rd_params[k] += np.random.normal(0, .001)

        self.gen_params += np.random.normal(0, .005, self.gen_params.shape)

    def crossover(self, other):    
        new_rd_params = {}
        for k in self._rd_params:
            options = (self._rd_params[k], other._rd_params[k])
            new_rd_params[k] = np.random.choice(options)

        new_gen_params = np.empty_like(self.gen_params)
        for k in range(len(self.gen_params)):
            for i in range(len(self.gen_params[k])):
                for j in range(len(self.gen_params[k][i])):
                    for l in range(len(self.gen_params[k][i][j])):
                        options = (self.gen_params[k][i][j][l], other.gen_params[k][i][j][l])
                        if self.fitness == -1:
                            options = (np.random.random(), np.random.random())
                        new_gen_params[k][i][j][l] = np.random.choice(options)


        result = Chromosome(new_rd_params, new_gen_params)
        result.mutate()

        return result

def set_fitness(chromosomes, preferred):
    total_chromosomes = len(chromosomes)
    num_preferred = len(preferred)
    num_remaining = total_chromosomes - num_preferred

    preferred = set([idx-1 for idx in preferred])
    fitness_preferred = .9
    fitness_remaining = .1

    for idx in range(total_chromosomes):
        c = chromosomes[idx]
        if idx in preferred:
            c.fitness = fitness_preferred / num_preferred
        else:
            c.fitness = fitness_remaining / num_remaining    

def apply_selection(chromosomes):
    total_chromosomes = len(chromosomes)
    new_generation    = []

    total_fitness = sum(c.fitness for c in chromosomes)
    probabilities = [c.fitness / total_fitness for c in chromosomes]

    for _ in range(total_chromosomes):
        mate1, mate2 = np.random.choice(chromosomes, 2, p=probabilities)
        child = mate1.crossover(mate2)
        child.mutate()
        new_generation.append(child)
    
    return new_generation

