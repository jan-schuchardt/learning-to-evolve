"""
This module contains strategies for repairing genomes encoding invalid solutions.
"""

from abc import abstractmethod
from random import uniform

import numpy as np

from genomes import KStepRealValuedGenome, OneStepRealValuedGenome
from strategies.strategy import Strategy


class RepairStrategy(Strategy):
    """
    This 
    """
    def __init__(self, problem=None):
        self.problem = problem

    @abstractmethod
    def repair(self, population):
        """
        Repairs the genome of all individuals in-place.
        """
        pass


class TravellingSalesmanRepairStrategy(RepairStrategy):

    def repair(self, population):
        pass


class BinaryDropoutRepairStrategy(RepairStrategy):

    """
    This strategy repairs invalid solutions to the knapsack problem
    by randomly removing selected items, until the accumulated weight
    is lower than the weight limit.
    """
    def repair(self, population):
        for genome in population:
            self.repair_genome(genome)

    def repair_genome(self, genome):
        while not self.problem.is_feasible(genome):
            non_zero_indeces = np.nonzero(genome.data)[0]
            change_index = np.random.choice(non_zero_indeces)

            genome.data[change_index] = 0


class RealValuedResamplingRepairStrategy(RepairStrategy):

    """
    This strategy repairs invalid genomes for continuous optimization
    by randomly resampling them from the problem domain.
    Step sizes are reset to the initial step size.
    """

    def __init__(self,  initial_step_size, problem=None):
        super().__init__(problem)
        self.initial_step_size = initial_step_size

    def repair(self, population):
        for genome in population:
            if not self.problem.is_feasible(genome):
                self.repair_genome(genome)

    def repair_genome(self, genome):
        data = np.array([uniform(low, high) for low, high in self.problem.constraints])
        genome.data = data

        if isinstance(genome, KStepRealValuedGenome):
            genome.step_sizes = np.full(self.problem.num_dimensions, self.initial_step_size)

        if isinstance(genome, OneStepRealValuedGenome):
            genome.step_size = self.initial_step_size
