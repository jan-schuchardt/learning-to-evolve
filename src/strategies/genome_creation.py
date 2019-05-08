"""
This module contains strategies for creating genomes for the different problem types.
"""

from abc import abstractmethod
from random import uniform

import numpy as np

from genomes import (BinaryGenome, IntegerGenome, KStepRealValuedGenome,
                     OneStepRealValuedGenome, RealValuedGenome)
from strategies.strategy import Strategy


class GenomeCreationStrategy(Strategy):
    """
    The base class for all genome creation strategies
    """
    def __init__(self, problem=None):
        self.problem = problem

    @abstractmethod
    def create(self, *args, data=None, **kwargs):
        pass


class BinaryGenomeCreationStrategy(GenomeCreationStrategy):
    """
    The genome creation strategy for evolutionary algorithms using a binary encoding
    """

    def create(self, *args, data=None, **kwargs):
        """
        Creates a binary genome. If no data is provided, the genome is initialized randomly.
        """
        if data is not None:
            return BinaryGenome(data)
        else:
            data = np.random.randint(0, high=2, size=self.problem.num_dimensions, dtype='int')
            return BinaryGenome(data)


class IntegerGenomeCreationStrategy(GenomeCreationStrategy):
    """
    The genome creation strategy for evolutionary algorithms using an integer-valued encoding
    """

    def create(self, *args, data=None, **kwargs):
        """
        Creates an integer-valued genome. If no data is provided, the genome is initialized randomly.
        """

        if data is not None:
            return IntegerGenome(data)
        else:
            data = np.random.permutation(self.problem.num_dimensions)
            return IntegerGenome(data)


class RealValuedGenomeCreationStrategy(GenomeCreationStrategy):
    """
    The genome creation strategy for evolutionary algorithms using a real-valued encoding.
    Supports:
    -genomes without additional control chromosomes,
    -genomes with a single step size chromosome
    -genomes with k step size chromosomes (one per problem dimension)
    """

    control_types = ['none', 'one', 'k']

    def __init__(self, initial_step_size, control_type='one', problem=None):
        if control_type not in RealValuedGenomeCreationStrategy.control_types:
            raise ValueError('Invalid control type')

        super().__init__(problem)
        self.control_type = control_type
        self.initial_step_size = initial_step_size

    def create(self, *args, data=None, **kwargs):

        step_size = kwargs.pop('step_size', None)
        step_sizes = kwargs.pop('step_sizes', None)

        # Randomly sample genome data from problem domain, if no data is provided
        if data is None:
            data = np.array([uniform(low, high) for low, high in self.problem.constraints])

        # No control chromosomes
        if self.control_type == 'none':
            return RealValuedGenome(data)

        # One step size chromosome
        elif self.control_type == 'one':
            if step_size == None:
                step_size = self.initial_step_size
            return OneStepRealValuedGenome(data, step_size)

        # K step size chromosomes, one per problem dimension
        elif self.control_type == 'k':
            if step_sizes == None:
                step_sizes = np.full(self.problem.num_dimensions, self.initial_step_size)

            return KStepRealValuedGenome(data, step_sizes)
