"""
This module contains utility classes and methods for use
in combination with the implemented neural networks
"""

import os.path
from collections import deque
from random import sample

import torch
import torch.nn as nn


class ExperienceStore():
    """
    Implements a circular queue of fixed length that can be used to store
    (state, action, reward, new_state) tuples and randomly
    sample from it. Useful for qlearning / vanilla actor critic / ddpg
    """

    def __init__(self, batch_size, memory_size):
        """
        :param batch_size: The number of elements that is sampled from the store at once
        :param memory_size: The number of experiences that can be stored
        """
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.memory = deque(maxlen=memory_size)

    def is_ready(self):
        """
        Returns whether or not there are enough experiences stored to start sampling

        :returns: True = Ready to start sampling; False= Not ready
        """
        return len(self.memory) >= self.batch_size

    def sample(self):
        """
        :returns: Randomly sampled batch_size elements from the experience store,
        """
        sampled_batch = sample(self.memory, self.batch_size)

        # Output: 4 lists (states, actions, rewards, new_states)
        return zip(*sampled_batch)

    def append(self, state, action, reward, new_state):
        """
        Adds a tuple of state, action, reward, new_state to the experience store

        :param state: The state to store
        :param action: The action to store
        :param reward: The reward to store
        :param new_state: The new state to store
        """
        self.memory.append((state, action, reward, new_state))


def store_weights(network, directory, iteration, weight_file_name):
    """
    Stores the weights of a neural network during a specific training iteration
    in a directory.
    Weights get stored as directory/weights{iteration}

    :param network: A neural network model to store
    :param directory: The name of the directory to store the weights in
    :param iteration: The training iteration associated with the weights
    """

    iteration_directory = f'{directory}/weights{iteration}'
    if not os.path.isdir(iteration_directory):
        os.makedirs(iteration_directory)

    file_path = f'{iteration_directory}/{weight_file_name}'

    torch.save(network.state_dict(), file_path)


def load_weights(network, weight_file):
    """
    Loads stored weights from the given weight_file into the given neural network model.
    Neural network must have exactly the same weight dimensions as the one used to create the weight_file

    :parameter network: The network to load the weights into
    :parameter weight_file: The name of the file to load the weights from
    """
    network.load_state_dict(torch.load(weight_file))


def init_weights(layer):
    """
    Initializes a Conv2D layer using xavier_uniform for weights and a normal distribution for the bias

    """
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_normal_(layer.weight.data)
    if isinstance(layer, nn.Conv1d):
        nn.init.xavier_normal_(layer.weight.data)