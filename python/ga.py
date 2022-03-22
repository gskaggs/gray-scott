import numpy as np

class Chromosome:
    def __init__(self, F, k):
        self._F = F
        self._k = k

    @property 
    def F(self):
        return self._F

    @property
    def k(self):
        return self._k

    @property
    def fitness(self):
        return self._fitness

    def mutate(self):
        self._F += np.random.normal(0, .001)
        self._k += np.random.normal(0, .001)

    def set_fitness(self, fitness):
        self._fitness = fitness

    def crossover(self, other):
        loc = np.random.choice([0, 1, 2])
        
        F = other.F if loc == 0 else self.F
        k = self.k if loc == 2 else other.k

        return Chromosome(F, k)
