from curses import newpad
import numpy as np

class Chromosome:
    def __init__(self, rd_params, gen_params=None):
        self._rd_params = rd_params
        self.gen_params = gen_params if gen_params is not None else []
        self.F = self.get_param('F')
        self.k = self.get_param('k')

    def get_param(self, param_name):
        return self._rd_params.get(param_name, 0)

    def get_params(self):
        return self._rd_params

    @property
    def fitness(self):
        return self._fitness

    @property
    def image(self):
        return self._image

    @property
    def pattern(self):
        return self._pattern

    def mutate(self):
        for k in self._rd_params:
            self._rd_params[k] += np.random.normal(0, .001)

    def set_pattern(self, pattern):
        self._pattern = pattern

    def set_fitness(self, fitness):
        self._fitness = fitness

    def set_image(self, image):
        self._image = image

    def crossover(self, other):    
        new_rd_params = {}
        for k in self._rd_params:
            options = (self._rd_params[k], other._rd_params[k])
            new_rd_params[k] = np.random.choice(options)

        new_gen_params = np.zeros(np.shape(self.gen_params)).tolist()
        for k in range(len(self.gen_params)):
            for i in range(len(self.gen_params[k])):
                for j in range(len(self.gen_params[k][i])):
                    options = (self.gen_params[k][i][j], other.gen_params[k][i][j])
                    new_gen_params[k][i][j] = np.random.choice(options)


        result = Chromosome(new_rd_params, new_gen_params)
        result.mutate()

        return result

def apply_fitness_function(chromosomes, type):
    N = len(chromosomes)
    result = [None] * N

    if type == 'default':
        chromosomes.sort(key=lambda c: -c.fitness) # sorted by decreasing fitness
        eps = 0.2 # Percent of population which survives to next round
        survivors = max(int(eps * N), 1)
        best = range(survivors)
        for i in range(survivors):
            result[i] = chromosomes[i]
        for i in range(survivors, N):
            mate1, mate2 = tuple(map(lambda x: chromosomes[x], np.random.choice(best, 2)))
            result[i] = mate1.crossover(mate2)

    else:
        best = []
        eps = .25
        survivors = max(int(eps*N), 1)
        while True:
            try:
                best = list(map(int, input(f'Top {survivors} performers:').strip().split(' ')))
            except:
                print("Input integers please.")
                continue 
            if len(best) != survivors:
                continue
            
            if any(map(lambda x: x < 1 or x > N, best)):
                continue
            
            # 0-index
            best = list(map(lambda x: x-1, best))

            break
        
        for i in range(survivors):
            result[i] = chromosomes[best[i]]

        for i in range(survivors, N):
            mate1, mate2 = tuple(map(lambda x: chromosomes[x], np.random.choice(best, 2)))
            result[i] = mate1.crossover(mate2)

    return result