"""
This module contains strategies for survivor selection
"""

from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal

from neural_networks.networks import CombinedActorCriticNetwork
from reinforcement.ppo import PPOStrategy
from strategies.strategy import Strategy


class SurvivorSelectionStrategy(Strategy):

    """
    This is the base class for survivor selection
    """
    @abstractmethod
    def select(self, previous_population, offspring, generations_left):
        """
        :param previous_population: Iterable, containing individuals of previous generation
        :param offspring: Iterable, containing offspring of previous generation
        :param generations_left: Float-encoding of remaining generations
        :returns: List of individuals from both input populations, selected for survival
        """
        pass


class ReplacingSurvivorSelectionStrategy(SurvivorSelectionStrategy):
    """
    This strategy implements replacing survivor selection.
    The elite_size fittest individuals from the previous population are chosen.
    The remaining number of survivors are the fittest individuals from the offspring population.
    """

    def __init__(self, elite_size=0, prefer_higher_fitness=True):
        self.elite_size = elite_size
        self.prefer_higher_fitness = prefer_higher_fitness

    def select(self, previous_population, offspring, generations_left):

        sorted_population = sorted(previous_population,
                                   reverse=self.prefer_higher_fitness,
                                   key=(lambda x: x.fitness))

        sorted_offspring = sorted(offspring,
                                  reverse=self.prefer_higher_fitness,
                                  key=(lambda x: x.fitness))

        elite = sorted_population[:self.elite_size]

        selected_offspring = sorted_offspring[:(len(previous_population) - self.elite_size)]

        return (elite + selected_offspring)


class RankedSurvivorSelectionStrategy(SurvivorSelectionStrategy):

    """
    This strategy implements ranked survivor selection.
    The fittest individuals from the union of the previous and the offspring generation
    are selected for survival.
    """

    def __init__(self, prefer_higher_fitness=True):
        self.prefer_higher_fitness = prefer_higher_fitness

    def select(self, previous_population, offspring, genetations_left):
        sorted_population = sorted(previous_population + offspring,
                                   reverse=self.prefer_higher_fitness,
                                   key=(lambda x: x.fitness))

        return sorted_population[:len(previous_population)]


class PPOSurvivorSelection(PPOStrategy):

    """
    This strategy implements proximal polciy optimization - based survivor selection.
    The policy outputs a real-valued value for each individual from the previous and offspring 
    generation.
    Those with the highest values are selected for survival.

    Actions are taken as samples from independent normal distributions.

    Deterministic elitism can additionally be enforced. In this case, 
    """

    def __init__(self,
                 encoding_strategy,
                 population_size,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_survivor_selection',
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
                 value_loss_factor=0.5):

        self.population_size = population_size

        network = CombinedActorCriticNetwork(
            encoding_strategy.num_channels(),
            2,
            eliminate_length_dimension=True,
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
            if torch.isnan(distribution_params).any():
                    raise ValueError('Nan detected')

            distribution = self.create_distribution(distribution_params)

            action = distribution.sample()
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        else:
            action, _ = self.network(Variable(state).cuda())

            # Take mean determinstically during validation
            action = action[:, 0, :]

        return action.detach()

    def perform_action(self, action, population):
        # weird bug causes wrong outputs for .topk on cuda tensors
        action = action.cpu().view(-1).numpy()

        num_survivors = self.population_size

        # Select the individuals with the highest output number
        survivor_indeces = np.argpartition(action, num_survivors)[-num_survivors:]

        survivors = [population[i] for i in survivor_indeces]

        return survivors

    def select(self, previous_population, offspring, generations_left):
        candidates = previous_population + offspring

        state = self.generate_encoded_state(candidates, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        selection = self.perform_action(action, candidates)

        return selection

    def create_distribution(self, distribution_params):
        return Normal(distribution_params[:, 0, :], F.softplus(distribution_params[:, 1, :]))