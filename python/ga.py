import numpy as np

class Chromosome:
    def __init__(self, rd_params, du, dv, gen_params=None):
        self._rd_params = rd_params
        self.gen_params = gen_params if gen_params is not None else np.zeros((2,2,3,3,3))
        self.F = self.get_param('F')
        self.k = self.get_param('k')
        self.du = du
        self.dv = dv
        self.fitness = 0

    def get_param(self, param_name):
        return self._rd_params.get(param_name, 0)

    def get_params(self):
        return self._rd_params

    def mutate(self):
        sd = {'F': .001, 'k': .001, 'rho': .01, 'mu': .01, 'nu': .01, 'kappa': .01}
        for k in self._rd_params:
            self._rd_params[k] += np.random.normal(0, sd[k])

        self.du += np.random.normal(0, .02)
        self.dv += np.random.normal(0, .02)

        self.gen_params += np.random.normal(0, .05, self.gen_params.shape)

    def crossover(self, other):    
        new_rd_params = {}

        for k in self._rd_params:
            options = (self._rd_params[k], other._rd_params[k])
            new_rd_params[k] = np.random.choice(options)

        new_rho = np.empty_like(self.gen_params[0])
        new_kap = np.empty_like(self.gen_params[1])

        for k in range(len(new_rho)):
            for i in range(len(new_rho[k])):
                for j in range(len(new_rho[k][i])):
                    for l in range(len(new_rho[k][i][j])):
                        rho_options = [self.gen_params[0][k][i][j][l], other.gen_params[0][k][i][j][l]]
                        kap_options = [self.gen_params[1][k][i][j][l], other.gen_params[1][k][i][j][l]]
                        if self.fitness == 0:
                            rho_options[0] = 2*np.random.random() - 1
                            kap_options[0] = 2*np.random.random() - 1
                        if other.fitness == 0:
                            rho_options[1] = 2*np.random.random() - 1
                            kap_options[1] = 2*np.random.random() - 1
                            
                        new_rho[k][i][j][l] = np.random.choice(rho_options)
                        new_kap[k][i][j][l] = np.random.choice(kap_options)

        du = np.random.choice((self.du, other.du))
        dv = np.random.choice((self.dv, other.dv))

        new_gen_params = np.array([new_rho, new_kap]).astype(np.float64)

        result = Chromosome(new_rd_params, du, dv, new_gen_params)
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
    if total_fitness == 0:
        probabilities = [1 / len(chromosomes) for c in chromosomes]
    else:
        probabilities = [c.fitness / total_fitness for c in chromosomes]

    for _ in range(total_chromosomes):
        mate1, mate2 = np.random.choice(chromosomes, 2, p=probabilities)
        child = mate1.crossover(mate2)
        child.mutate()
        new_generation.append(child)
    
    return new_generation

