"""
This module contains different variants of representing
soltuions in genomes.
"""

from abc import ABC, abstractmethod


class Genome(ABC):
    """
    A genome holds some sort of encoded solution and
    can decode it. It also stores the solution's fitness
    """

    @abstractmethod
    def decode(self):
        """
        :return: A decoding of the solution stored in the genome
        """
        pass

    @abstractmethod
    def length(self):
        pass


class BinaryGenome(Genome):
    """
    BinaryGenome stores the solution as a binary sequence
    """

    def __init__(self, data, fitness=0):
        """
        :parameter length: The length of the stored data
        :parameter data: The data to store in the genome. If None,
        a random binary sequence is generated
        :parameter fitness: The fitness of the stored solution
        """
        self.fitness = fitness
        self.data = data

    def decode(self):
        return self.data

    def length(self):
        return len(self.data)


class IntegerGenome(Genome):
    """
    IntegerGenome stores the solution as a sequency of integers
    """

    def __init__(self, data, fitness=0):
        self.fitness = fitness
        self.data = data

    def decode(self):
        return self.data

    def length(self):
        return len(self.data)


class RealValuedGenome(Genome):
    """
    RealValuedGenome stores the solution as a sequence of floats
    """

    def __init__(self, data, fitness=0):
        self.fitness = fitness
        self.data = data

    def decode(self):
        return self.data

    def length(self):
        return len(self.data)


class OneStepRealValuedGenome(RealValuedGenome):
    """
    OneStepRealValuedGenoem stores the solution as a sequence of floats
    and additionally stores one step size per individuals.
    """

    def __init__(self, data, step_size, fitness=0):
        super().__init__(data, fitness=fitness)
        self.step_size = step_size


class KStepRealValuedGenome(RealValuedGenome):
    """
    KStepRealValuedGenome stores the solution as a sequence of floats
    and additionally stores k step sizes per individuals, one
    per problem dimension.
    """

    def __init__(self, data, step_sizes, fitness=0):
        super().__init__(data, fitness)
        self.step_sizes = step_sizes
