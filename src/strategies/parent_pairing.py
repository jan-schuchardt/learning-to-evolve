"""
This module contains strategies for selecting pairs of parents from populations
"""

from abc import abstractmethod
from random import choice, sample

import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal

from neural_networks.networks import CombinedActorCriticNetwork
from reinforcement.ppo import PPOStrategy
from strategies.strategy import Strategy


class ParentPairingStrategy(Strategy):
    """
    This is the base class for all parent pairing strategies.
    """

    def __init__(self, num_pairs, **kwargs):
        """
        :param num_pairs: The number of parent pairs to be generated.
        """
        self.num_pairs = num_pairs

    @abstractmethod
    def select(self, population, generations_left):
        """
        Takes an interable of individuals and returns a list of tuples,
        containing the paired parents.
        """
        pass


class TournamentParentPairingStrategy(ParentPairingStrategy):
    """
    This strategy implements tournament selection with tournament size 2.
    For each pair of parents, the first parent is selected by taking the fitter
    of two randomly chosen individuals.
    The second parent is selected in the same fashion.
    """

    def select(self, population, generations_left):
        parent_pairs = []

        for _ in range(self.num_pairs):
            parent_1 = self.perform_tournament(population)
            remaining_population = population[:]
            remaining_population.remove(parent_1)
            parent_2 = self.perform_tournament(remaining_population)

            pair = (parent_1, parent_2)
            parent_pairs.append(pair)

        return parent_pairs
        
    def perform_tournament(self, population):
        if len(population) == 1:
            return population[0]
        else:
            genome1, genome2 = tuple(sample(population, 2))

        if genome1.fitness > genome2.fitness:
            return genome1
        else:
            return genome2


class RandomSingleParentPairingStrategy(ParentPairingStrategy):
    """
    This strategy serves as a placeholder for evolutionary algorithms
    that do not perform crossover.
    A list of single-element tuples with individuals randomly sampled
    from the population is returned.
    """

    def select(self, population, generations_left):
        parent_pairs = []

        for _ in range(self.num_pairs):
            parent_pairs.append((choice(population), ))

        return parent_pairs


class PPOFitnessShapingTournamentSelection(PPOStrategy):

    """
    This strategy implements fitness shaping for the genetic algorithms.
    Fitness values are multiplied with values selected by the policy.
    Afterwards, tournament selection is performed, using the altered fitness values.
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 num_pairs,
                 training=False,
                 weight_file_name='ppo_fitness_shaping_tournament',
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
                 ):
        """
        :param num_pairs: Number of parent pairs that have to be generated
        """

        self.tournament_strategy = TournamentParentPairingStrategy(num_pairs)
        self.num_pairs = num_pairs

        network = CombinedActorCriticNetwork(
            encoding_strategy.num_channels(),
            2,
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

        if self.training:
            distribution_params, _ = self.network(Variable(state).cuda())
 
            distribution = self.create_distribution(distribution_params)
            
            action = distribution.sample()
        else:
            distribution_params, _ = self.network(Variable(state).cuda())

            action = distribution_params[:, 0, :]

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def select(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)

        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        parents_factors = {}

        # Store factor, by which the fitness of each parent is multiplied
        for fitness_factor, individual in zip(action.cpu().numpy()[0], population):
            parents_factors[individual] = fitness_factor

        parent_pairs = []

        # Perform tournament selection to pair parents
        for _ in range(self.num_pairs):
            parent_1 = self.perform_tournament(population, parents_factors)
            remaining_population = population[:]
            remaining_population.remove(parent_1)
            parent_2 = self.perform_tournament(remaining_population, parents_factors)

            pair = (parent_1, parent_2)
            parent_pairs.append(pair)

        return parent_pairs
        
    def perform_tournament(self, parents, parents_factors):
        """
        :param parents: List of available parents
        :param parents_factors: Value, by which each parent's fitness is to be multiplied
        """
        if len(parents) == 1:
            return parents[0]

        parent1, parent2 = tuple(sample(parents, 2))

        if parent1.fitness * parents_factors[parent1] > parent2.fitness * parents_factors[parent2]:
            return parent1
        else:
            return parent2

    def create_distribution(self, distribution_params):

        return Normal(distribution_params[:, 0, :], F.softplus(distribution_params[:, 0, :]))
