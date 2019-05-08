"""
This module contains functionality shared by all reinforcement learning algorithms
"""

from abc import abstractmethod

import torch

from neural_networks.utils import ExperienceStore, load_weights, store_weights
from strategies.strategy import Strategy


class ReinforcementLearningStrategy(Strategy):
    """
    The super class for all reinforcement learning algorithms.
    """
    def __init__(self,
                 encoding_strategy,
                 weight_file_name,
                 training=False,
                 experience_store_length=100000,
                 batch_size=32):
        """
        :param encoding_strategy: Strategy for generating state tensor from population data
        :param weight_file_name: Name to be used when storing network weights
        :param training: Whether training or validation is performed
        :param experience_store_length: Number of experience samples to store
        :param batch_size: Number of samples to train on per iteration
        """

        self.encoding_strategy = encoding_strategy
        self.experience_store_length = experience_store_length

        self.experience_store = ExperienceStore(batch_size, experience_store_length)

        self.last_experience = {}

        self.weight_file_name = weight_file_name

        self.training = training

    @abstractmethod
    def optimize_model(self):
        """
        Optimize the neural network models, using past experience samples
        """
        pass

    def reward(self, reward, new_state):
        """
        Reward the agent upon arriving in a new state. Has to be called before performing the next action
        """
        self.experience_store.append(
            self.last_experience['state'],
            self.last_experience['action'],
            reward,
            new_state
        )

        self.last_experience = {}

    def store_weights(self, weight_store_folder, iteration):
        """
        Store the weights of the acting neural network

        :param weight_store_folder: Folder to store weights in
        :param iteration: Number of the current trainign iteration
        """
        acting_network = self.get_acting_network()

        if weight_store_folder is not None:
            store_weights(acting_network, weight_store_folder, iteration, self.weight_file_name)

    def load_weights(self, weight_load_folder):
        """
        Load the weights of the acting neural network from the specified weight_load_folder
        """
        acting_network = self.get_acting_network()

        if weight_load_folder is not None:
            load_weights(acting_network, f'{weight_load_folder}/{self.weight_file_name}')

    def generate_encoded_state(self, population, generations_left):
        """
        Encode the current state of an evolutionary algorithm, using the given encoding strategy

        :param population: The current population of an evolutionary algorithm
        :param generations_left: Float, encoding the number of remaining generations
        :returns: Tensor encoding state of the evolutionary algorithm
        """
        return self.encoding_strategy.encode(population, generations_left)

    def soft_update(self, network, target_network, tracking_rate):
        """
        Change weights of the target network in direction of the main network, when using separate 
        target and training networks.

        :param network: Main neural network
        :param network: Target neural network
        :param tracking_rate: How much the target network follows the main network. 0:No changes. 1: Copy weights
        """
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tracking_rate) + param.data * tracking_rate)

    def hard_update(self, network, target_network):
        """
        Make weights of the target network match the weights of the main neural network, when using separate 
        target and training networks.
        """
        target_network.load_state_dict(network.state_dict())

    def preprocess_experience_samples(self):
        """
        Takes samples from the experience store and converts them into torch tensors.

        :returns: a tuple of states, actions, reward and new states
        """

        states, actions, rewards, new_states = self.experience_store.sample()

        # transform given tuples into tensors
        states = torch.cat(list(states)).cuda()

        actions = torch.cat(list(actions))

        rewards = torch.tensor(list(rewards)).float().cuda()

        new_states = torch.cat(list(new_states)).cuda()

        return states, actions, rewards, new_states

    def update_exploration_rate(self):
        """
        Updates the degree to which exploration is performed
        """
        pass

    @abstractmethod
    def get_acting_network(self):
        """
        Returns the network that is relevant for taking actions during validation
        """
        pass


class OrnsteinUhlenbeckProcess:
    """
    A mean-reverting process, can be used for directed exploration
    https://de.wikipedia.org/wiki/Ornstein-Uhlenbeck-Prozess
    """

    def __init__(self, mean_reversion=0.8, mean=0, scale=1, uniform=True):
        self.mean_reversion = mean_reversion
        self.mean = mean
        self.scale = scale
        self.uniform = uniform

    def sample(self, action):
        if self.uniform:
            t = self.mean_reversion * (self.mean - action) + self.scale * (torch.rand_like(action) - 0.5)
        else:
            t = self.mean_reversion * (self.mean - action) + self.scale * torch.randn_like(action)

        return t
