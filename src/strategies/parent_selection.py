"""
This module contains strategies for performing parent selection
"""

from abc import abstractmethod
from math import floor

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal

from neural_networks.networks import CombinedActorCriticNetwork
from reinforcement.ppo import PPOStrategy
from strategies.strategy import Strategy


class ParentSelectionStrategy(Strategy):
    """
    This is the base class for aprent selection
    """

    @abstractmethod
    def select(self, population, generations_left):
        pass


class RankedSelectionStrategy(ParentSelectionStrategy):

    """
    This method implements ranked parent selection, to be used for continuous optimization
    The percentage fittest individuals are chosen as parents deterministically.
    """
    
    def __init__(self, percentage, prefer_higher_fitness=True):

        if percentage <= 0 or percentage > 1:
            raise ValueError('Selection percentage must be in (0, 1]')

        self.prefer_higher_fitness = prefer_higher_fitness
        self.percentage = percentage

    def select(self, population, generations_left):

        sorted_population = sorted(population,
                                   key=(lambda x: x.fitness),
                                   reverse=self.prefer_higher_fitness)

        # Round non-integer numbers to next lower integer
        cutoff_index = floor(len(population) * self.percentage)
        return sorted_population[:cutoff_index]


class PPOFitnessShapingRankedSelection(PPOStrategy):

    """
    This strategy implements fitness shaping for the continuous optimization algorithm.
    Fitness values are multiplied with values selected by the policy.
    Afterwards, the percentage fittest individuals are selected deterministically.
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 percentage,
                 training=False,
                 weight_file_name='ppo_fitness_shaping_ranked',
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
                 prefer_higher_fitness=True
                 ):
        """
        :param percentage: The percentage of fittest individuals to be selected from the population
        :param prefer_higher_fitness: Whether a higher or lower fitness
                                      is preferred during selection
        """

        self.percentage = percentage
        self.prefer_higher_fitness = prefer_higher_fitness

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

        individual_factors = {}

        # Store factors by which each individual's fitness is to be multiplied
        for fitness_factor, individual in zip(action.cpu().numpy()[0], population):

            individual_factors[individual] = fitness_factor

        # Select fittest individuals, based on altered fitness values
        sorted_population = sorted(population,
                                   key=(lambda x: x.fitness * individual_factors[x]),
                                   reverse=self.prefer_higher_fitness)

        cutoff_index = floor(len(population) * self.percentage)

        return sorted_population[:cutoff_index]

    def create_distribution(self, distribution_params):

        return Normal(distribution_params[:, 0, :], F.softplus(distribution_params[:, 0, :]))
