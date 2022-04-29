import pandas as pd
from PIL import Image as im
import time
from datetime import timedelta

from python.init_utils import init_chromosomes, prep_sim
from python.present_utils import present_chromosomes
from sim_driver_tui import process_function_ga
from python.process_util import start_processes, end_processes

class GuiSimulationDriver():
    def __init__(self):
        self.rd = ['gray_scott']
        self.F = [.01, .08, 3]
        self.k = [.04, .06, 2]
        self.rho = [0]
        self.mu = [0]
        self.nu = [0]
        self.kappa = [0]
        self.num_generalized = 0
        self.param_search = False
        self.num_individuals = 6
        self.num_iters = -1
        self.num_processes = 6
        self.use_cpu = False
        self.fitness = 'dirichlet'
        self.end_time = 200
        self.dirichlet_vis = False
        self.chromosomes = init_chromosomes(self)

    def reset(self):
        pass

    def get_spreadsheet(self):
        return pd.DataFrame(list(range(20)))

    def run_generation(self, generation_id):
        # Prepare process safe queues
        self.chromosomes, modified = prep_sim(self.chromosomes, generation_id, self)
        
        # Do the simulations
        start = time.time()
        processes = start_processes(self.num_processes, process_function_ga, (self.chromosomes, modified, self))
        self.chromosomes = end_processes(processes, modified, self.num_processes)
        end = time.time()
        print(f'Generation {generation_id} time taken {timedelta(seconds=end-start)}')

        # Save the results
        self.chromosomes.sort(key=lambda c: -c.fitness) # sorted by decreasing fitness
        image = present_chromosomes(self.chromosomes, generation_id, self)

        width, height = image.size
        width, height = 2000, int(height * 2000 / width)
        image = image.resize((width, height))

        return image


if __name__ == "__main__":
    driver = GuiSimulationDriver()
    image = driver.run_generation(1)
    image.show()