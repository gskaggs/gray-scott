from python.simulator import ReactionDiffusionSimulator
from python.ga import Chromosome

c = Chromosome({}, 0, 0)
sim = ReactionDiffusionSimulator(chromosome=c)


_, _, image = sim.integrate(0)
image.show()