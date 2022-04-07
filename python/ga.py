import numpy as np

class Chromosome:
    def __init__(self, rd_params):
        self._rd_params = rd_params
        self.F = self.get_param('F')
        self.k = self.get_param('k')

    def get_param(self, param_name):
        return self._rd_params.get(param_name, 0)

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
        for i in range(len(self.rd_params)):
            self.rd_params[i] +=  np.random.normal(0, .001)

    def set_pattern(self, pattern):
        self._pattern = pattern

    def set_fitness(self, fitness):
        self._fitness = fitness

    def set_image(self, image):
        self._image = image

    def crossover(self, other):
        loc = np.random.choice([0, 1, 2])
        
        F = other.F if loc == 0 else self.F
        k = self.k if loc == 2 else other.k
        result = Chromosome([F, k])
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