"""
This module contains adaptation strategies that can be used for different purposes.
"""

from itertools import compress
from random import sample

import torch
from torch.autograd import Variable
from torch.distributions.binomial import Binomial

from neural_networks.networks import CombinedActorCriticNetwork
from reinforcement.ppo import PPOStrategy


class PPOPopulationSubsetSelection(PPOStrategy):
    """
    This proximal policy optimization based strategy selects a subset of arbitrary size 
    from a population.

    This strategy uses a binomial distribution to take actions:
    The samples of the distribution are directly used as a binary map for selecting individuals
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_population_subset',
                 learning_rate=1e-3,
                 min_subset_size=0,
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
        """
        :param min_subset_size: The minimum number of individuals that has to be selected.
                                If too few are selected, min_subset_size random individuals are returned
        """

        self.min_subset_size = min_subset_size

        num_output_channels = 1

        network = CombinedActorCriticNetwork(
            encoding_strategy.num_channels(),
            num_output_channels,
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
                         entropy_factor=entropy_factor,
                         entropy_factor_decay=entropy_factor_decay,
                         min_entropy_factor=min_entropy_factor,
                         value_loss_factor=value_loss_factor
                         )

    def select_action(self, state):

        self.optimizer.zero_grad()

        distribution_params, _ = self.network(Variable(state).cuda())

        distribution = self.create_distribution(distribution_params)
        if torch.isnan(distribution_params).any():
            raise ValueError('Nan detected')

        action = distribution.sample()

        self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach().cpu()

    def perform_action(self, action, population):

        # Use binary map to select elements form population
        population_subset = list(compress(population, action.view(-1)))

        # Randomly sample, if two few individuals are chosen
        if len(population_subset) < self.min_subset_size:
            population_subset = sample(population, self.min_subset_size)

        return population_subset

    def select(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)

        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action

        population_subset = self.perform_action(action, population)

        return population_subset

    def create_distribution(self, distribution_params):

        return Binomial(1, logits=distribution_params)
