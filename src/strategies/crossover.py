"""
This module contains strategies for performing crossover.
"""

from abc import abstractmethod
from copy import deepcopy
from random import sample, uniform

import numpy as np

import torch
from neural_networks.networks import CombinedActorCriticNetwork
from reinforcement.ppo import PPOStrategy
from strategies.strategy import Strategy
from torch.autograd import Variable
from torch.distributions.categorical import Categorical


class CrossoverStrategy(Strategy):
    """
    The base class for crossover strategies.
    """
    @abstractmethod
    def crossover(self, parent_pairs, generations_left):
        """
        Performs crossover for a given set of parent pairs.

        :param parent_pairs: An iterable of tuples of genomes
        :param generatins_left: A float representation of the remaining number of generations

        :returns: A list of offspring genomes
        """
        pass


class UniformCrossoverStrategy(CrossoverStrategy):
    """
    Uniform crossover for evolutionary algorithms with a binary encoding.
    
    With a probability of crossover_rate, two parents are crossed over to create two children.
    Else, the parents are copied directly.

    If parents are crossed over, there is a 50% chance for each chromosome value in parent 1
    to be exchanged with that of parent 2.
    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            child_1, child_2 = self.crossover_pair(parent_1, parent_2)
            offspring.extend([child_1, child_2])

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            child_1, child_2 = deepcopy(parent_1), deepcopy(parent_2)

        else:

            length = parent_1.length()
            data_child_1 = np.empty(length, dtype='int')
            data_child_2 = np.empty(length, dtype='int')

            for i, (chromosome1, chromosome2) in enumerate(zip(parent_1.data, parent_2.data)):

                if uniform(0, 1) < 0.5:
                    data_child_1[i] = chromosome1
                    data_child_2[i] = chromosome2
                else:
                    data_child_1[i] = chromosome2
                    data_child_2[i] = chromosome1

            child_1 = self.genome_creation_strategy.create(data=data_child_1)
            child_2 = self.genome_creation_strategy.create(data=data_child_2)
            
        return child_1, child_2


class CopyCrossoverStrategy(CrossoverStrategy):
    """
    A placeholder crossover strategy for algorithms that do not require crossover.
    Returns all parents in the given parent pairs.
    """

    def __init__(self, genome_creation_strategy):
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for pair in parent_pairs:
            offspring.append(deepcopy(pair[0]))

        return offspring


class OnePointCrossoverStrategy(CrossoverStrategy):
    """
    One-point crossover for evolutionary algorithms with a permutation-based integer encoding,
    as defined in https://pdfs.semanticscholar.org/54bf/a05a2a993ba1b8896901751846f32566e701.pdf
    
    With a probability of crossover_rate, two parents are crossed over to create one child.
    Else, the fitter parent is copied directly

    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            if uniform(0, 1) < 0.5:
                child = self.crossover_pair(parent_1, parent_2)
            else:
                child = self.crossover_pair(parent_2, parent_1)
            offspring.append(child)

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            if parent_1.fitness > parent_2.fitness:
                child = deepcopy(parent_1)
            else:
                child = deepcopy(parent_2)

        else:

            length = parent_1.length()
            data_child = np.empty(length, dtype='int')

            crossover_position = np.random.randint(0, length)

            data_1 = parent_1.data[0:crossover_position]
            data_2 = parent_2.data[np.isin(parent_2.data, data_1, invert=True)]

            data_child[0:crossover_position] = data_1
            data_child[crossover_position:] = data_2

            child = self.genome_creation_strategy.create(data=data_child)
            
        return child


class TwoPointCrossoverStrategy(CrossoverStrategy):
    """
    Two-point crossover for evolutionary algorithms with a permutation-based integer encoding,
    as defined in https://pdfs.semanticscholar.org/54bf/a05a2a993ba1b8896901751846f32566e701.pdf
    
    With a probability of crossover_rate, two parents are crossed over to create one child.
    Else, the fitter parent is copied directly

    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            if uniform(0, 1) < 0.5:
                child = self.crossover_pair(parent_1, parent_2)
            else:
                child = self.crossover_pair(parent_2, parent_1)
            offspring.append(child)

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            if parent_1.fitness > parent_2.fitness:
                child = deepcopy(parent_1)
            else:
                child = deepcopy(parent_2)

        else:

            length = parent_1.length()
            data_child = np.empty(length, dtype='int')

            positions = np.sort(np.random.randint(0, length, 2))

            data_2 = parent_1.data[positions[0]:positions[1]]

            data_1 = parent_2.data[:positions[1]][
                    np.isin(parent_2.data[:positions[1]], data_2, invert=True)
                ]

            data_3 = parent_2.data[positions[1]:][
                    np.isin(parent_2.data[positions[1]:], data_2, invert=True)
                ]

            data_remainder = np.concatenate((data_3, data_1))

            data_child[positions[0]:positions[1]] = data_2

            data_child[positions[1]:] = data_remainder[:len(data_child)-positions[1]]

            data_child[:positions[0]] = data_remainder[
                    len(data_child)-positions[1]:len(data_child)-positions[1]+positions[0]
                ]

            child = self.genome_creation_strategy.create(data=data_child)

        return child


class LinearCrossoverStrategy(CrossoverStrategy):
    """
    Linear order crossover for evolutionary algorithms with a permutation-based integer encoding,
    as defined in https://pdfs.semanticscholar.org/54bf/a05a2a993ba1b8896901751846f32566e701.pdf

    With a probability of crossover_rate, two parents are crossed over to create one child.
    Else, the fitter parent is copied directly

    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            if uniform(0, 1) < 0.5:
                child = self.crossover_pair(parent_1, parent_2)
            else:
                child = self.crossover_pair(parent_2, parent_1)
            offspring.append(child)

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            if parent_1.fitness > parent_2.fitness:
                child = deepcopy(parent_1)
            else:
                child = deepcopy(parent_2)

        else:

            length = parent_1.length()
            data_child = np.empty(length, dtype='int')

            positions = np.sort(np.random.randint(0, length, 2))

            data_1 = parent_1.data[positions[0]:positions[1]]

            data_2 = parent_2.data[np.isin(parent_2.data, data_1, invert=True)]

            data_child[positions[0]:positions[1]] = data_1
            data_child[:positions[0]] = data_2[:positions[0]]
            data_child[positions[1]:] = data_2[positions[0]:]

            child = self.genome_creation_strategy.create(data=data_child)
            
        return child


class CyclicCrossoverStrategy(CrossoverStrategy):
    """
    Cyclic crossover for evolutionary algorithms with a permutation-based integer encoding,
    as defined in https://pdfs.semanticscholar.org/54bf/a05a2a993ba1b8896901751846f32566e701.pdf
    
    With a probability of crossover_rate, two parents are crossed over to create one child.
    Else, the fitter parent is copied directly

    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            if uniform(0, 1) <= 0.5:
                child = self.crossover_pair(parent_1, parent_2)
            else:
                child = self.crossover_pair(parent_2, parent_1)

            offspring.append(child)

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            if parent_1.fitness > parent_2.fitness:
                child = deepcopy(parent_1)
            else:
                child = deepcopy(parent_2)

        else:

            length = parent_1.length()
            data_child = np.empty(length, dtype='int')

            starting_index = np.random.randint(0, length)
            values = [parent_1.data[starting_index]]

            index = np.where(parent_1.data == parent_2.data[starting_index])[0][0]
            while not index == starting_index:
                values.append(parent_1.data[index])
                index = np.where(parent_1.data == parent_2.data[index])[0][0]

            mask_1 = np.isin(parent_1.data, values, invert=False)
            mask_2 = np.invert(mask_1)

            data_child[mask_1] = parent_1.data[mask_1]
            data_child[mask_2] = parent_2.data[mask_2]

            child = self.genome_creation_strategy.create(data_child)

        return child


class PositionCrossoverStrategy(CrossoverStrategy):
    """
    Position-based crossover for evolutionary algorithms with a permutation-based integer encoding,
    as defined in https://pdfs.semanticscholar.org/54bf/a05a2a993ba1b8896901751846f32566e701.pdf
    
    With a probability of crossover_rate, two parents are crossed over to create one child.
    Else, the fitter parent is copied directly

    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            if uniform(0, 1) <= 0.5:
                child = self.crossover_pair(parent_1, parent_2)
            else:
                child = self.crossover_pair(parent_2, parent_1)

            offspring.append(child)

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            if parent_1.fitness > parent_2.fitness:
                child = deepcopy(parent_1)
            else:
                child = deepcopy(parent_2)

        else:

            length = parent_1.length()
            data_child = np.empty(length, dtype='int')

            values = np.random.choice(length, np.random.choice(length), replace=False)

            mask_1 = np.isin(parent_1.data, values)
            mask_2 = np.isin(parent_2.data, values, invert=True)

            data_child[mask_1] = parent_1.data[mask_1]
            data_child[np.invert(mask_1)] = parent_2.data[mask_2]

            child = self.genome_creation_strategy.create(data_child)

        return child


class OrderCrossoverStrategy(CrossoverStrategy):
    """
    Order-based crossover for evolutionary algorithms with a permutation-based integer encoding,
    as defined in https://pdfs.semanticscholar.org/54bf/a05a2a993ba1b8896901751846f32566e701.pdf
    
    With a probability of crossover_rate, two parents are crossed over to create one child.
    Else, the fitter parent is copied directly

    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            if uniform(0, 1) <= 0.5:
                child = self.crossover_pair(parent_1, parent_2)
            else:
                child = self.crossover_pair(parent_2, parent_1)

            offspring.append(child)

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            if parent_1.fitness > parent_2.fitness:
                child = deepcopy(parent_1)
            else:
                child = deepcopy(parent_2)

        else:

            length = parent_1.length()
            data_child = np.empty(length, dtype='int')

            values = np.random.choice(length, np.random.choice(length), replace=False)

            mask_1 = np.isin(parent_1.data, values)
            mask_2 = np.isin(parent_2.data, values)

            data_child[mask_2] = parent_1.data[mask_1]
            data_child[np.invert(mask_2)] = parent_2.data[np.invert(mask_2)]

            child = self.genome_creation_strategy.create(data_child)

        return child


class PartiallyMappedCrossoverStrategy(CrossoverStrategy):
    """
    Partially mapped crossover for evolutionary algorithms with permutation-based integer encoding,
    as defined in https://pdfs.semanticscholar.org/54bf/a05a2a993ba1b8896901751846f32566e701.pdf

    With a probability of crossover_rate, two parents are crossed over to create one child.
    Else, the fitter parent is copied directly

    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

    def crossover(self, parent_pairs, generations_left):

        offspring = []

        for parent_1, parent_2 in parent_pairs:
            if uniform(0, 1) <= 0.5:
                child = self.crossover_pair(parent_1, parent_2)
            else:
                child = self.crossover_pair(parent_2, parent_1)

            offspring.append(child)

        return offspring

    def crossover_pair(self, parent_1, parent_2):
        if uniform(0, 1) >= self.crossover_rate:

            if parent_1.fitness > parent_2.fitness:
                child = deepcopy(parent_1)
            else:
                child = deepcopy(parent_2)

        else:
            length = parent_1.length()
            data_child = -1 * np.ones(length, dtype='int')

            positions = np.sort(np.random.randint(0, length+1, 2))
            
            data_child[positions[0]:positions[1]] = parent_1.data[positions[0]:positions[1]]

            for value in parent_2.data[positions[0]:positions[1]]:
                if value not in parent_1.data[positions[0]:positions[1]]:
                    self.place_value(parent_1, parent_2, data_child, value, value, positions)

            mask = np.isin(parent_2.data, data_child, invert=True)

            data_child[mask] = parent_2.data[mask]

            child = self.genome_creation_strategy.create(data_child)

        return child

    def place_value(self, parent_1, parent_2, data_child, search_value, save_value, positions):
        """
        This recursive method looks for a place to position a specific value in the
        genome of a child. Google PMX for a detailed explanation.

        :param parent_1: The first parent
        :param parent_2: The second parent
        :param data_child: The child genome to place values in
        :param search_value: Value, whose position in parent 2's genome is to be found
        :param save_value: Value to be stored
        :param positions: Left and right border of serach interval.
        """
        index = np.where(parent_2.data == search_value)[0][0]

        if index < positions[0] or index >= positions[1]:

            data_child[index] = save_value

        else:

            self.place_value(parent_1,
                             parent_2,
                             data_child,
                             parent_1.data[index],
                             save_value,
                             positions)


class PPOGlobalCrossoverOperatorSelectionStrategy(PPOStrategy):
    """
    A proximal policy optimization-based strategies
    for selecting from the different crossover operators
    used for permutation-based integer genomes.

    Samples from a categorical distribution are taken to choose one of the operators.
    """

    def __init__(self,
                 encoding_strategy,
                 genome_creation_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_global_crossover_operator_selection',
                 learning_rate=1e-5,
                 discount_factor=0.99,
                 variance_bias_factor=0.98,
                 num_hidden_layers=1,
                 num_neurons=128,
                 batch_size=32,
                 clipping_value=0.2,
                 num_training_epochs=4,
                 dim_elimination_max_pooling=False,
                 entropy_factor=0.1,
                 entropy_factor_decay=0.05,
                 min_entropy_factor=0.01,
                 value_loss_factor=0.5,
                 crossover_rate=0.9,
                 ):
        """
        :param crossover_rate: The probability, with which crossover is performed,
        instead of copying the fittest parent
        """

        self.operators = [
            OnePointCrossoverStrategy(crossover_rate, genome_creation_strategy),
            TwoPointCrossoverStrategy(crossover_rate, genome_creation_strategy),
            LinearCrossoverStrategy(crossover_rate, genome_creation_strategy),
            CyclicCrossoverStrategy(crossover_rate, genome_creation_strategy),
            PositionCrossoverStrategy(crossover_rate, genome_creation_strategy),
            OrderCrossoverStrategy(crossover_rate, genome_creation_strategy),
            PartiallyMappedCrossoverStrategy(crossover_rate, genome_creation_strategy)
        ]

        network = CombinedActorCriticNetwork(
            2 * encoding_strategy.num_channels(),
            len(self.operators),
            eliminate_length_dimension=True,
            eliminate_population_dimension=True,
            dim_elimination_max_pooling=dim_elimination_max_pooling,
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons
        ).cuda()

        super().__init__(network,
                         encoding_strategy,
                         weight_file_name,
                         training=training,
                         learning_rate=learning_rate,
                         num_actors=num_actors,
                         episode_length=episode_length,
                         discount_factor=discount_factor,
                         variance_bias_factor=variance_bias_factor,
                         batch_size=batch_size,
                         clipping_value=clipping_value,
                         num_training_epochs=num_training_epochs,
                         finite_environment=True,
                         entropy_factor=entropy_factor,
                         entropy_factor_decay=entropy_factor_decay,
                         min_entropy_factor=min_entropy_factor,
                         value_loss_factor=value_loss_factor
                         )

    def select_action(self, state):
        self.optimizer.zero_grad()

        distribution_params, _ = self.network(Variable(state).cuda())
        
        distribution = self.create_distribution(distribution_params)
        
        action = distribution.sample()

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def crossover(self, parent_pairs, generations_left):
        parents = list(zip(*parent_pairs))

        # Calculate state by concatenating encoding of "first" parents and "second" parents
        state_1 = self.generate_encoded_state(parents[0], generations_left)
        state_2 = self.generate_encoded_state(parents[1], generations_left)

        state = torch.cat([state_1, state_2], 1)

        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        return self.operators[action].crossover(parent_pairs, generations_left)

    def create_distribution(self, distribution_params):

        distribution_params = distribution_params.max(dim=2)[0]

        return Categorical(logits = distribution_params)


class PPOIndividualCrossoverOperatorSelectionStrategy(PPOStrategy):
    """
    A proximal policy optimization-based strategies for selecting from the different crossover operators
    used for permutation-based integer genomes. For each pair of parents, a different
    crossover operator can be chosen

    Samples from a categorical distribution are taken to choose the operators.
    """

    def __init__(self,
                 encoding_strategy,
                 genome_creation_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_individual_crossover_operator_selection',
                 learning_rate=1e-5,
                 discount_factor=0.99,
                 variance_bias_factor=0.98,
                 num_hidden_layers=1,
                 num_neurons=128,
                 batch_size=32,
                 clipping_value=0.2,
                 num_training_epochs=4,
                 dim_elimination_max_pooling=False,
                 entropy_factor=0.1,
                 entropy_factor_decay=0.05,
                 min_entropy_factor=0.01,
                 value_loss_factor=0.5,
                 crossover_rate=0.9,
                 ):
        """
        :param crossover_rate: The probability, with which crossover is performed, instead of copying the fittest parent
        """

        self.operators = [
            OnePointCrossoverStrategy(crossover_rate, genome_creation_strategy),
            TwoPointCrossoverStrategy(crossover_rate, genome_creation_strategy),
            LinearCrossoverStrategy(crossover_rate, genome_creation_strategy),
            CyclicCrossoverStrategy(crossover_rate, genome_creation_strategy),
            PositionCrossoverStrategy(crossover_rate, genome_creation_strategy),
            OrderCrossoverStrategy(crossover_rate, genome_creation_strategy),
            PartiallyMappedCrossoverStrategy(crossover_rate, genome_creation_strategy)
        ]

        network = CombinedActorCriticNetwork(
            2 * encoding_strategy.num_channels(),
            len(self.operators),
            eliminate_length_dimension=True,
            eliminate_population_dimension=False,
            dim_elimination_max_pooling=dim_elimination_max_pooling,
            num_hidden_layers=num_hidden_layers,
            num_neurons=num_neurons
        ).cuda()

        super().__init__(network,
                         encoding_strategy,
                         weight_file_name,
                         training=training,
                         learning_rate=learning_rate,
                         num_actors=num_actors,
                         episode_length=episode_length,
                         discount_factor=discount_factor,
                         variance_bias_factor=variance_bias_factor,
                         batch_size=batch_size,
                         clipping_value=clipping_value,
                         num_training_epochs=num_training_epochs,
                         finite_environment=True,
                         entropy_factor=entropy_factor,
                         entropy_factor_decay=entropy_factor_decay,
                         min_entropy_factor=min_entropy_factor,
                         value_loss_factor=value_loss_factor
                         )

    def select_action(self, state):
        self.optimizer.zero_grad()

        distribution_params, _ = self.network(Variable(state).cuda())

        
        distribution = self.create_distribution(distribution_params)
        
        action = distribution.sample()

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def crossover(self, parent_pairs, generations_left):
        parents = list(zip(*parent_pairs))
        
        state_1 = self.generate_encoded_state(parents[0], generations_left)
        state_2 = self.generate_encoded_state(parents[1], generations_left)

        state = torch.cat([state_1, state_2], 1)

        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        offspring = []
        
        # For each pair of parents, apply the chosen operator
        for parent_pair, operator_index in zip(parent_pairs, action.cpu().numpy()[0]):
            offspring.extend(self.operators[operator_index].crossover([parent_pair], generations_left))

        return offspring

    def create_distribution(self, distribution_params):

        # Transpose, because we want one value per pair, not one value per chromosome
        return Categorical(logits = distribution_params.transpose(1, 2))


class RandomCrossoverOperatorSelectionStrategy(CrossoverStrategy):
    """
    A strategy for randomly selecting one of the crossover operators for permutation-based,
    integer genomes.

    All crossover operators have a uniform probability
    """

    def __init__(self, crossover_rate, genome_creation_strategy):
        super().__init__()
        self.crossover_rate = crossover_rate
        self.genome_creation_strategy = genome_creation_strategy

        self.operators = [
            OnePointCrossoverStrategy(crossover_rate, genome_creation_strategy),
            TwoPointCrossoverStrategy(crossover_rate, genome_creation_strategy),
            LinearCrossoverStrategy(crossover_rate, genome_creation_strategy),
            CyclicCrossoverStrategy(crossover_rate, genome_creation_strategy),
            PositionCrossoverStrategy(crossover_rate, genome_creation_strategy),
            OrderCrossoverStrategy(crossover_rate, genome_creation_strategy),
            PartiallyMappedCrossoverStrategy(crossover_rate, genome_creation_strategy)
        ]

    def crossover(self, parent_pairs, generations_left):

        operator = sample(self.operators, 1)

        return operator[0].crossover(parent_pairs, generations_left)
