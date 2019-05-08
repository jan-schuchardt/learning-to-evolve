"""
This module contains different neural network models for application
in evolutionary algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_networks.utils import init_weights


def add_broadcasted(x):
    """
    Take a tensor of size Batch x Features x LengthH x LengthV
    and appends channel which are created through max-pooling
    followed by broadcasting along the vertical / horizontal
    spatial dimension
    """
    # Take max and add back dimension that was deleted through taking max
    if x.dim() == 4:
        max_horizontal = x.max(3)[0].unsqueeze(3)
        max_horizontal_broadcasted = max_horizontal.expand_as(x)

        max_vertical = x.max(2)[0].unsqueeze(2)
        max_vertical_broadcasted = max_vertical.expand_as(x)

        # Concatenate original x and broadcasted information along feature dimension
        return torch.cat([x, max_horizontal_broadcasted, max_vertical_broadcasted], 1)

    # If there is only one spatial dimension, only perform pooling and broadcasted along that dimension
    elif x.dim() == 3:
        max_vertical = x.max(2)[0].unsqueeze(2)
        max_vertical_broadcasted = max_vertical.expand_as(x)

        # Concatenate original x and broadcasted information along feature dimension
        return torch.cat([x, max_vertical_broadcasted], 1)


class CombinedActorCriticNetwork(nn.Module):
    """
    A neural network using 2D convolutions and pooling+broadcasting to process
    sets of genomes with variable length.

    Outputs both the parameters of an action distribution, and the associated value approximate..

    All input genomes chromosomes are evaluated with the same weights through
    2d convolution.
    Global information is generated through horizontal and vertical
    broadcasting + pooling along the spatial dimensions
    """

    def __init__(self,
                 num_input_channels,
                 num_output_channels,
                 eliminate_length_dimension=True,
                 eliminate_population_dimension=False,
                 dim_elimination_max_pooling=False,
                 num_hidden_layers=1,
                 num_neurons=128):
        """
        :param num_input_channels: Number of input channel
        :param num_output_channels: Number of output channels for the actor component
        :param eliminate_length_dimension: Whether to eliminate the 4th dimension of the actor output
        :param eliminate_population_dimension: Whether to also eliminate the 3rd dimension of the actor output
        :param dim_eliminiation_max_pooling: Whether to use max- or mean-pooling
        :param num_hidden_layers: Number of layers between first convolution and the two output layers 
        :param num_neurons: Number of neurons / filters per conv layer
        :param eliminate_length_dimension: Whether to eliminate
        """

        super().__init__()

        self.num_output_channels = num_output_channels
        self.num_input_channels = num_input_channels
        self.eliminiate_length_dimension = eliminate_length_dimension
        self.eliminate_population_dimension = eliminate_population_dimension
        self.dim_elimination_max_pooling = dim_elimination_max_pooling

        self.num_hidden_layers = num_hidden_layers

        self.input_norm = nn.BatchNorm2d(num_input_channels)

        self.layers = nn.ModuleList()

        # Generate input and hidden layers
        for layer_number in range(num_hidden_layers + 1):

            if layer_number == 0:
                conv_layer = nn.Conv2d(num_input_channels * 3, num_neurons, 1)
            else:
                conv_layer = nn.Conv2d(num_neurons * 3, num_neurons, 1)

            self.layers.append(conv_layer)

        # Generate output layers

        if self.eliminiate_length_dimension:
            self.output_layer_actor = nn.Conv1d(num_neurons * 2, num_output_channels, 1)
        else:
            self.output_layer_actor = nn.Conv2d(num_neurons * 3, num_output_channels, 1)

        self.output_layer_critic = nn.Conv1d(num_neurons * 2, 1, 1)

        self.apply(init_weights)

    def forward(self, state):
        """
        Generates action distribution parameters and value approximates for the given state tensor

        :param state: 4D state tensor of Batchsize x #Channels x P x G
        :returns: Tuple of a tensor of action distribution parameters and a tensor of value approximates
        """
        x = state

        for i in range(self.num_hidden_layers + 1):
            x = add_broadcasted(x)
            
            x = F.leaky_relu(self.layers[i](x))

        action_distributions = x
        values = x

        # Eliminate dimensions before the output layers
        if self.dim_elimination_max_pooling:
            if self.eliminiate_length_dimension:
                action_distributions = action_distributions.max(3)[0]
                if self.eliminate_population_dimension:
                    action_distributions = action_distributions.max(2)[0].unsqueeze(2)
            values = values.max(3)[0]
        else:
            if self.eliminiate_length_dimension:
                action_distributions = action_distributions.mean(3)
                if self.eliminate_population_dimension:
                    action_distributions = action_distributions.mean(2).unsqueeze(2)
            values = values.mean(3)

        # Calculate action output
        action_distributions = self.output_layer_actor(add_broadcasted(action_distributions))

        # Calculate value approximate
        values = self.output_layer_critic(add_broadcasted(values))

        values = values.sum(2).view(-1)
            
        return action_distributions, values


