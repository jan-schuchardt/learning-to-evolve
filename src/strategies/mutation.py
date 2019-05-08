"""
This module contains strategies for performing mutation.
"""

from abc import abstractmethod
from random import uniform

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.beta import Beta
from torch.distributions.binomial import Binomial
from torch.distributions.normal import Normal

from neural_networks.networks import CombinedActorCriticNetwork
from reinforcement.ppo import PPOStrategy
from strategies.strategy import Strategy


class MutationStrategy(Strategy):
    """
    The base class for all mutation strategies
    """

    @abstractmethod
    def mutate(self, genome, generations_left):
        """
        Mutates a genome in-place
        """
        pass


class BinaryMutationStrategy(MutationStrategy):

    """
    This class implements binary uniform mutation.

    Each bit in a genome is flipped with a probability of mutation_rate.
    """

    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, population, generations_left):
        for genome in population:
            self.mutate_genome(genome)

    def mutate_genome(self, genome):
        for i in range(len(genome.data)):

            if uniform(0, 1) < self.mutation_rate:
                genome.data[i] = 1 - genome.data[i]


class OneStepUncorrelatedMutation(MutationStrategy):

    """
    This class implements continuous onestep uncorellated mutation.

    The step size of each individual is mutated through multiplication with a sample
    from a log-normal distribution with \sigma=learning_parameter
    The solution-encoding part of the genome is then mutated by adding samples
    from independent normal distributions with \sigma=step size

    """

    def __init__(self, learning_parameter, minimum_step_size):
        self.minimum_step_size = minimum_step_size
        self.learning_parameter = learning_parameter

    def mutate(self, population, generations_left):
        for genome in population:
            self.mutate_genome(genome)

    def mutate_genome(self, genome):
        perturbation = np.random.lognormal(sigma=self.learning_parameter)

        new_step_size = genome.step_size * perturbation
        new_step_size = min(1, max(new_step_size, self.minimum_step_size))

        genome.step_size = new_step_size

        genome.data += np.random.normal(scale=genome.step_size, size=genome.data.size)


class KStepUncorrelatedMutation(MutationStrategy):
    """
    This class implements continuous k-step uncorellated mutation.

    The step sizes of each individual are mutated through multiplication with samples from
    independent log-normal distribution with \sigma=learning_parameter_1 and
    \sigma=learning_parameter_2
    The solution-encoding part of the genome is then mutated by adding samples
    from independent normal distributions with \sigma=step sizes

    """

    def __init__(self, learning_parameter_1, learning_parameter_2, minimum_step_size):
        self.minimum_step_size = minimum_step_size
        self.learning_parameter_1 = learning_parameter_1
        self.learning_parameter_2 = learning_parameter_2

    def mutate(self, population, generations_left):
        for genome in population:
            self.mutate_genome(genome)

    def mutate_genome(self, genome):

        perturbation = np.random.lognormal(sigma=self.learning_parameter_1,
                                           size=np.size(genome.step_sizes))

        perturbation *= np.random.lognormal(sigma=self.learning_parameter_2,
                                            size=np.size(genome.step_sizes))

        new_step_sizes = genome.step_sizes * perturbation
        new_step_sizes = np.minimum(1.0, np.maximum(new_step_sizes, self.minimum_step_size))

        genome.step_sizes = new_step_sizes

        genome.data += np.random.normal(scale=genome.step_sizes)


class InversionMutationStrategy(MutationStrategy):
    """
    This class implements inversion mutation for permutation-based genomes with integer encoding.

    Two random positions in the genome are chosen, the order of the chromosome values between
    them is then inverted.

    The operator is applied to each individual with a probability of mutation_rate.
    """

    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, population, generations_left):
        for genome in population:
            self.mutate_genome(genome)

    def mutate_genome(self, genome):
        if uniform(0, 1) < self.mutation_rate:

            positions = np.sort(np.random.randint(0, len(genome.data)+1, 2))

            subsequence = genome.data[positions[0]:positions[1]]
            genome.data[positions[0]:positions[1]] = np.flip(subsequence, 0)


class PPOIndividualMutationRateControl(PPOStrategy):

    """
    This strategy performs binary mutation, but controls the mutation_rate
    of each individual separately.

    Mutation rates are taken as samples from independent beta distributions.
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_binary_individual_mutation',
                 learning_rate=1e-5,
                 discount_factor=0.99,
                 variance_bias_factor=0.98,
                 num_hidden_layers=1,
                 num_neurons=128,
                 batch_size=32,
                 clipping_value=0.2,
                 num_training_epochs=4,
                 dim_elimination_max_pooling=False,
                 fixed_std_deviation=-1,
                 entropy_factor=0.1,
                 entropy_factor_decay=0.05,
                 min_entropy_factor=0.01,
                 value_loss_factor=0.5):

        num_output_channels = 2

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
        else:
            # Select mean deterministically during validation
            action, _ = self.network(Variable(state).cuda())
            alpha = F.softplus(action[:, 0, :]) + 1
            beta = F.softplus(action[:, 1, :]) + 1
            action = alpha / (alpha + beta)

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def perform_action(self, action, population):
        mutation_maps = self.generate_mutation_maps(action)

        for genome, mutation_map in zip(population, mutation_maps):
            mutated_data = np.abs(genome.data - mutation_map)
            genome.data = mutated_data

    def mutate(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        self.perform_action(action, population)

    def create_distribution(self, distribution_params):

            alpha = F.softplus(distribution_params[:, 0, :]) + 1
            beta = F.softplus(distribution_params[:, 1, :]) + 1
            return Beta(alpha, beta)

    def generate_mutation_maps(self, action):
        """
        Create the binary map describing how the chromosome of each individual is mutated.
        For each individual, the probability of one chromosome flipping is defined
        by the sampled action.
        """
        
        genome_length = self.encoding_strategy.problem.num_dimensions

        # Expand action, so that all chromosomes of an individual
        # have the same mutation probability
        binomial_probs = action.view(-1).unsqueeze(1).expand(-1, genome_length)

        return Binomial(1, probs=binomial_probs).sample().cpu().numpy()


class PPOComponentLevelBinaryMutation(PPOStrategy):
    """
    This strategy mutates binary genomes by outputting a binary matrix. The matrix indicates
    which chromosomes should be flipped.

    Actions are taken by sampling from bernoulli trials.
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_binary_chromosome_mutation',
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

        num_output_channels = 1

        network = CombinedActorCriticNetwork(
            encoding_strategy.num_channels(),
            num_output_channels,
            eliminate_length_dimension=False,
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

    def perform_action(self, action, population):
        mutation_maps = action[0][0].cpu().numpy()

        for genome, mutation_map in zip(population, mutation_maps):
            mutated_data = np.abs(genome.data - mutation_map)
            genome.data = mutated_data

    def mutate(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        self.perform_action(action, population)

    def create_distribution(self, distribution_params):

        return Binomial(1, logits=distribution_params)


class PPOGlobalMutationRateControl(PPOStrategy):
    """
    This strategy controls binary mutation by outputting a single mutation rate.

    Mutation rates are taken as samples from a beta distribution
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_binary_global_mutation',
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

        num_output_channels = 2

        network = CombinedActorCriticNetwork(
            encoding_strategy.num_channels(),
            num_output_channels,
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

        if self.training:
            distribution_params, _ = self.network(Variable(state).cuda())

            distribution = self.create_distribution(distribution_params)

            action = distribution.sample()
        else:
            # Take mean determinstically during validation
            action, _ = self.network(Variable(state).cuda())
            action = action.max(dim=2)[0]
            alpha = F.softplus(action[:, 0]) + 1
            beta = F.softplus(action[:, 1]) + 1
            action = alpha / (alpha + beta)

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def perform_action(self, action, population):

        mutation_rate = action[0].cpu().numpy()

        for genome in population:
            self.mutate_genome(genome, mutation_rate)

    def mutate_genome(self, genome, mutation_rate):
        for i in range(len(genome.data)):
            if uniform(0, 1) < mutation_rate:
                genome.data[i] = 1 - genome.data[i]

    def mutate(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        self.perform_action(action, population)

    def create_distribution(self, distribution_params):
        distribution_params = distribution_params.max(dim=2)[0]

        alpha = F.softplus(distribution_params[:, 0]) + 1
        beta = F.softplus(distribution_params[:, 1]) + 1
        return Beta(alpha, beta)


class PPOComponentLevelStepSizeControl(PPOStrategy):

    """
    This strategy implements component level step size control.
    Each individual holds one step size per problem dimensions.
    Mutation is performed by multiplying step sizes with the actions of the policy.
    The solution-encoding chromosomes are then mutated with samples from a normal
    distributino with std = step_sizes.

    Samples from independent normal distributions are used as actions (followed by softplus)
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_mutate_kstep_uncorrelated_mutation',
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
                 minimum_step_size=1e-5):
        """
        :param minimum_step_size: The lower limit for step sizes
        """

        num_output_channels = 2

        self.minimum_step_size = minimum_step_size

        network = CombinedActorCriticNetwork(
            encoding_strategy.num_channels(),
            num_output_channels,
            eliminate_length_dimension=False,
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

        else:
            # During validation, the mean is taken deterministically
            distribution_params, _ = self.network(Variable(state).cuda())
            action = distribution_params[:, 0, :, :]

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def perform_action(self, action, population):
        perturbations = F.softplus(action[0])
        perturbations = perturbations.cpu().numpy()

        for genome, perturbation in zip(population, perturbations):
            new_step_sizes = genome.step_sizes * perturbation
            new_step_sizes = np.minimum(1.0, np.maximum(new_step_sizes, self.minimum_step_size))

            genome.step_sizes = new_step_sizes
            genome.data += np.random.normal(scale=genome.step_sizes)

    def mutate(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        self.perform_action(action, population)

    def create_distribution(self, distribution_params):

        return Normal(distribution_params[:, 0, :, :], F.softplus(distribution_params[:, 1, :, :]))


class PPOIndividualLevelLearningParameterControl(PPOStrategy):

    """
    This strategy implements individual-level learning parameter control.
    The step size of each genome is mutated with a sample from a log-normal distribution.
    The parameter \sigma is controlled by the strategy.

    Actions are taken as samples from a normal distribution, followed by softplus.
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_mutate_onestep_individual_learning_rates',
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
                 minimum_step_size=1e-5):
        """
        :param minimum_step_size: The lower limit for step sizes
        """

        num_output_channels = 2

        self.minimum_step_size = minimum_step_size

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
        else:
            # During validation, the mean is taken deterministically
            distribution_params, _ = self.network(Variable(state).cuda())
            action = distribution_params[:, 0, :]

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def perform_action(self, action, population):
        learning_parameters = F.softplus(action[0]).cpu().numpy()

        for genome, learning_parameter in zip(population, learning_parameters):

            perturbation = np.random.lognormal(sigma=learning_parameter)

            new_step_size = genome.step_size * perturbation
            new_step_size = min(1.0, max(new_step_size, self.minimum_step_size))

            genome.step_size = new_step_size

            genome.data += np.random.normal(scale=genome.step_size, size=genome.data.size)

    def mutate(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        self.perform_action(action, population)

    def create_distribution(self, distribution_params):

        return Normal(distribution_params[:, 0, :], F.softplus(distribution_params[:, 1, :]))


class PPOIndividualLevelStepSizeControl(PPOStrategy):
    """
    This strategy implements individual-level step size control.
    Each individual holds one step size used for mutation. Step sizes are multiplied
    with the actions of the trained policy.
    The solution-encoding genomes are then mutated by adding samples from a normal distribution
    with std=step size.

    Actions are taken as samples from a normal distribution, followed by softplus
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_mutate_onestep_uncorrelated_mutation',
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
                 minimum_step_size=1e-5):

        num_output_channels = 2

        self.minimum_step_size = minimum_step_size

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
        else:
            # During validation, the mean is taken deterministically
            distribution_params, _ = self.network(Variable(state).cuda())
            action = distribution_params[:, 0, :]

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def perform_action(self, action, population):
        perturbations = F.softplus(action[0]).cpu().numpy()

        for genome, perturbation in zip(population, perturbations):

            new_step_size = genome.step_size * perturbation
            new_step_size = min(1.0, max(new_step_size, self.minimum_step_size))

            genome.step_size = new_step_size

            genome.data += np.random.normal(scale=genome.step_size, size=genome.data.size)

    def mutate(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        self.perform_action(action, population)

    def create_distribution(self, distribution_params):

        return Normal(distribution_params[:, 0, :], F.softplus(distribution_params[:, 1, :]))


class PPOPopulationLevelLearningParameterControl(PPOStrategy):
    """
    This strategy implements population-level learning parameter control.
    The policy decides on a single learning parameter.
    The step size of each individual is mutated by multiplying with a sample from
    a log-normal distribution with \sigma = learning_parameter.
    The solution-encoding chromosomes are mutated with adding samples from independent
    normal distributions with std=step size

    Actions are taken as samples from a normal distribution, followed by softplus. 
    """

    def __init__(self,
                 encoding_strategy,
                 num_actors,
                 episode_length,
                 training=False,
                 weight_file_name='ppo_mutate_onestep_global_learning_rate',
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
                 minimum_step_size=1e-5):

        num_output_channels = 2

        self.minimum_step_size = minimum_step_size

        network = CombinedActorCriticNetwork(
            encoding_strategy.num_channels(),
            num_output_channels,
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

        if self.training:
            distribution_params, _ = self.network(Variable(state).cuda())
            if torch.isnan(distribution_params).any():
                    raise ValueError('Nan detected')

            distribution = self.create_distribution(distribution_params)

            action = distribution.sample()
        else:
            # During validation, the mean is taken deterministically
            distribution_params, _ = self.network(Variable(state).cuda())
            action = distribution_params.max(dim=2)[0][:, 0]

        if self.training:
            self.last_experience['log_prob'] = distribution.log_prob(action).sum().detach().cpu()

        return action.detach()

    def perform_action(self, action, population):
        mutation_learning_rate = F.softplus(action[0]).cpu().numpy()

        for genome in population:
            perturbation = np.random.lognormal(sigma=mutation_learning_rate)

            new_step_size = genome.step_size * perturbation
            new_step_size = min(1.0, max(new_step_size, self.minimum_step_size))

            genome.step_size = new_step_size

            genome.data += np.random.normal(scale=genome.step_size, size=genome.data.size)

    def mutate(self, population, generations_left):
        state = self.generate_encoded_state(population, generations_left)
        action = self.select_action(state)

        self.last_experience['state'] = state
        self.last_experience['action'] = action.cpu()

        self.perform_action(action, population)

    def create_distribution(self, distribution_params):

        distribution_params = distribution_params.max(dim=2)[0]

        return Normal(distribution_params[:, 0], F.softplus(distribution_params[:, 1]))
