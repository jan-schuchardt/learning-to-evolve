"""
This module contains strategies for encoding the state of a generation
for different optimization problems
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from genomes import OneStepRealValuedGenome, KStepRealValuedGenome


class EncodingStrategy(ABC):
    """
    An EncodingStrategy encodes the state of a population on a specific problem instance
    """

    def __init__(self, problem=None, num_problem_dimensions=None):
        self.problem = problem
        self.num_problem_dimensions = num_problem_dimensions

    @abstractmethod
    def encode(self, population, generations_left):
        """
        :parameter population: The population to encode
        :parameter generations_left: Encoding of the remaining number of generations
        :returns: The encoded state of a population
        """
        pass

    @abstractmethod
    def num_channels(self):
        pass


class ContinuousFunctionEncodingStrategy(EncodingStrategy):
    """
    Encoding strategy to be used for continuous optimization problems.
    """

    def encode(self, population, generations_left):
        """
        :returns: 4D tensor with the following channels:
                    -genome of each individual;
                    -fitness of each individual;
                    -step size(s) of each individual
                    -remaining number of generations
        """

        # Feature 0 : Solution of each individual
        population_data = torch.tensor(
            [genome.data for genome in population]
            ).unsqueeze(0).float()

        # Feature 1 : Fitness of each individual
        population_fitness = torch.tensor([[genome.fitness for genome in population]])
        population_fitness = population_fitness.transpose(0, 1)
        population_fitness = population_fitness.expand_as(population_data).float()
        if torch.isnan(population_fitness).any():
                raise ValueError('nan detected in fitness')

        # Feature 2: Step size or step sizes of each individual
        if isinstance(population[0], OneStepRealValuedGenome):
            step_sizes = torch.tensor([[genome.step_size for genome in population]])
            step_sizes = step_sizes.transpose(0, 1)
            step_sizes = - step_sizes.log10()
            if torch.isnan(step_sizes).any():
                raise ValueError('nan detected in step sizes')

        elif isinstance(population[0], KStepRealValuedGenome):
            step_sizes = torch.tensor([genome.step_sizes for genome in population])
            step_sizes = - step_sizes.log10()
            if torch.isnan(step_sizes).any():
                raise ValueError('nan detected in step sizes')

        step_sizes = step_sizes.expand_as(population_data).float()

        # Feature 3: Time left
        generations_left = torch.tensor(generations_left).float().expand_as(population_data)

        # Create #Features x #individuals x individual_length tensor
        state = torch.cat([population_data,
                            population_fitness,
                            step_sizes,
                            generations_left], 0)

        if torch.isnan(state).any():
            raise ValueError('Nan detected in state encoding')

        # Add batch_dimension of size 1
        state = state.unsqueeze(0)

        return state.float()

    def num_channels(self):
        return 4


class KnapsackEncodingStrategy(EncodingStrategy):
    """
    Encoding strategy to be used for the 0-1 knapsack problem.
    """

    def encode(self, population, generations_left):
        """
        :returns: 4D tensor with the following channels:
                    -genome of each individual;
                    -fitness of each individual;
                    -item weights
                    -item values
                    -problem weight limits
                    -remaining number of generations
        """
        # Feature 0 : Solution of each individual
        population_data = torch.tensor(
            [genome.data for genome in population]
            ).unsqueeze(0).float()

        # Feature 1 : Fitness of each individual
        population_fitness = np.array([[genome.fitness for genome in population]])
        population_fitness = torch.tensor([[genome.fitness for genome in population]])
        population_fitness = population_fitness.transpose(0, 1)
        population_fitness = population_fitness.expand_as(population_data).float()

        # Feature 2 : Weight of each item
        item_weights = torch.tensor([self.problem.get_weights()])
        item_weights = item_weights.expand_as(population_data).float()

        # Feature 3 : Value of each item
        item_values = torch.tensor([self.problem.get_values()])
        item_values = item_values.expand_as(population_data).float()

        # Feature 4: Weight limit
        weight_limit = torch.tensor(self.problem.weight_limit)
        weight_limit = weight_limit.expand_as(population_data)

        # Feature 5: Generations_left
        generations_left = torch.tensor(generations_left).float()
        generations_left = generations_left.expand_as(population_data)

        # Create #Features x #individuals x individual_length tensor
        state = torch.cat([population_data,
                            population_fitness,
                            item_weights,
                            item_values,
                            weight_limit,
                            generations_left], 0)

        # Add batch_dimension of size 1
        state = state.unsqueeze(0)

        return state.float()

    def num_channels(self):
        return 6


class TravellingSalesmanEncodingStrategy(EncodingStrategy):

    def encode(self, population, generations_left):
        """
        :returns: 4D tensor with the following channels:
                    -genome of each individual;
                    -fitness of each individual;
                    -remaining number of generations;
                    -Adjacency 
        """

        # Feature 0 : Solution of each individual
        population_data = torch.tensor(
            [genome.data for genome in population]
            ).unsqueeze(0).float()

        # Feature 1 : Fitness of each individual
        population_fitness = np.array([[genome.fitness for genome in population]])
        population_fitness = torch.tensor([[genome.fitness for genome in population]])
        population_fitness = population_fitness.transpose(0, 1)
        population_fitness = population_fitness.expand_as(population_data).float()

        # Feature 2: Generations_left
        generations_left = torch.tensor(generations_left).float()
        generations_left = generations_left.expand_as(population_data)

        # Remaining Features: Adjacency Information
        adjacency_matrix = torch.tensor(self.problem.adjacency_matrix).float()

        """
        Permutate the adjacency matrix, so that:
        Entry (i, j) of adjacency channel k-1 corresponds to the distance from node x to node k,
        where x is the node number stored in chromosome j of individual i
        """
        adjacency_information = adjacency_matrix[population_data[0].long()].permute(2, 0, 1)

        state = torch.cat([population_data,
                           population_fitness,
                           generations_left,
                           adjacency_information], 0)

        state = state.unsqueeze(0)

        return state.float()

    def num_channels(self):
        return 3 + self.num_problem_dimensions
