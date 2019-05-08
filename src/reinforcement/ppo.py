from abc import abstractmethod
from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim

from reinforcement.reinforcement import ReinforcementLearningStrategy


class PPOStrategy(ReinforcementLearningStrategy):
    """
    Super class for all proximal policy optimization based adaptation methods.
    To be used with a combined actor-critic network.
    Implements entropy-based exploration and generalized advantage estimation.

    PPO-based strategies have to store the state, action and its log-proability each time an action is performed.
    See any of implemented strategies for reference, or consult the provided readme.
    """

    def __init__(self,
                 network,
                 encoding_strategy,
                 weight_file_name,
                 training=False,
                 learning_rate=1e-5,
                 num_actors=4,
                 episode_length=20,
                 discount_factor=0.99,
                 variance_bias_factor=0.98,
                 clipping_value=0.2,
                 batch_size=32,
                 num_training_epochs=4,
                 finite_environment=True,
                 entropy_factor=0.1,
                 entropy_factor_decay=0.05,
                 min_entropy_factor=0.01,
                 value_loss_factor=0.5
                 ):
        """
        :param network: A combined actor-critic neural network
        :param encoding_strategy: The strategy to be used to generate state inputs
        :param weight_file_name: File name to use for storing network weights
        :param training: Whether strategy is used for validation or training
        :param learning_rate: Learning rate to be used for training
        :param num_actors: Number of actors collecting experience samples
        :param episode_length: Number of actions to be performed by each actor before optimization
        :param discount_factor: Discount factor \gamma of the markov decision process
        :param variance_bias_factor: \lambda of generalized advantage estimation
        :param clipping_value: Clipping value \epsilon of proximal policy optimization.
        :param batch_size: Number of training samples per training epoch
        :param num_training_epochs: Number of forward-backward passes over all experience samples
        :param finite_environment_true: Whether the last state transition
                                        always ends in a terminal state of the environment.
                                        If yes, the value estimate for the final state is always 0
        :param entropy_factor: The initial exploration-controlling coefficient \alpha_e
        :param entropy_factor_decay: The percentage of the initial entropy factor
                                     that is substracted from \alpha_e after each training
        :param min_entropy_factor: The lower threshold for decaying entropy factors
        :param value_loss_factor: The ratio between actor and critic loss
        """

        super().__init__(
            encoding_strategy,
            weight_file_name,
            training=training,
            batch_size=batch_size
            )

        # PPO is an on-policy method, so we discard training samples after each time that training is performed
        # We manage our own experience store, instead of using the Deque-like experience store used by other methods
        self.actor_experience_store = []

        self.batch_size = batch_size

        self.num_training_epochs = num_training_epochs

        self.num_actors = num_actors
        self.episode_length = episode_length

        self.clipping_value = clipping_value

        self.network = network

        self.critic_loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.variance_bias_factor = variance_bias_factor
        self.discount_factor = discount_factor

        self.finite_environment = finite_environment

        self.episode_memory = []

        self.entropy_factor = entropy_factor
        self.original_entropy_factor = entropy_factor
        self.entropy_factor_decay = entropy_factor_decay
        self.min_entropy_factor = min_entropy_factor

        self.value_loss_factor = value_loss_factor

    @abstractmethod
    def create_distribution(self, distribution_params):
        """
        Creates a probability distribution, parameterized by distribution_params.

        This probability distribution should then be used for taking actions through sampling.
        """
        pass

    def optimize_model(self):

        for _ in range(self.num_training_epochs):

            for mini_batch in self.generate_mini_batches():

                (states,
                    actions,
                    log_probs_old,
                    _,
                    _,
                    advantages,
                    returns) = self.preprocess_actor_experience_samples(mini_batch)

                loss = self.calc_loss(states, actions, log_probs_old, advantages, returns)

                loss.backward()
                self.optimizer.step()

        self.actor_experience_store = []
        self.update_exploration_rate()
        torch.cuda.empty_cache()

    def calc_loss(self, states, actions, log_probs_old, advantages, returns):
        """
        Calculates the combined actor, critic and exploration loss, as defined in
        https://arxiv.org/abs/1707.06347
        """
        self.optimizer.zero_grad()

        distribution_params, values = self.network(states)

        # Calculate the actor loss

        distribution = self.create_distribution(distribution_params)
        actions_log_probabilities = distribution.log_prob(actions)

        batch_size = actions_log_probabilities.size()[0]
        actions_log_probabilities = actions_log_probabilities.view(batch_size, -1).sum(1)

        prob_ratio = torch.exp(actions_log_probabilities - log_probs_old)

        surrogate_objective = torch.min(prob_ratio * advantages,
                                        torch.clamp(
                                            prob_ratio,
                                            min=1-self.clipping_value,
                                            max=1+self.clipping_value)
                                        * advantages)

        # Calculate the critic loss

        critic_loss = self.critic_loss_function(values, returns)

        # Combine actor and critic loss

        loss = - surrogate_objective + self.value_loss_factor * critic_loss

        # Add exploration loss

        if isinstance(distribution, torch.distributions.binomial.Binomial):
            # Using the binomial distribution as a bernoulli process
            # requires our own entropy calculation.

            probs = distribution.probs.view(batch_size, -1)
            entropies = - (probs * probs.clamp(min=0.000001, max=0.999999).log2()
                           - (1 - probs) * (1 - probs).clamp(min=0.000001, max=0.999999).log2())

            joint_entropy = entropies.sum(1)
            loss -= self.entropy_factor * joint_entropy

        else:

            loss -= self.entropy_factor * distribution.entropy().view(batch_size, -1).sum(1)

        loss = loss.mean()

        return loss

    def reward(self, reward, new_state):
        # Store experience
        self.episode_memory.append((self.last_experience['state'],
                                    self.last_experience['action'],
                                    self.last_experience['log_prob'],
                                    reward,
                                    new_state))

        self.last_experience = {}

        """
        After one experience-gathering episode is completed, calculate:
        -The advantage estimate, used for training the actor (Generalized advantage estimation)
        -The accumulated rewards ("returns"), used for training the critic
        """
        if len(self.episode_memory) == self.episode_length:

            # Convert data to cuda tensors

            states, actions, log_probs, rewards, new_states = zip(*self.episode_memory)
            self.episode_memory = []

            combined_states = torch.cat(list(states)).cuda()

            state_values = self.network.forward(combined_states)[1].detach()

            # Calculate advantage and returns

            for t in range(self.episode_length):
                advantage = 0
                returns = 0

                if t is not self.episode_length - 1:
                    for delta_t in range(self.episode_length - t):

                        x = t + delta_t

                        if self.finite_environment and delta_t == self.episode_length - t - 1:

                            #  if the end of an episode means the end of the environment,
                            # the terminal value should be 0
                            residual = rewards[x] + self.discount_factor * 0 - state_values[x]

                        else:

                            residual = (rewards[x]
                                        + self.discount_factor * state_values[x + 1]
                                        - state_values[x])

                        advantage += ((self.variance_bias_factor * self.discount_factor)
                                      ** delta_t) * residual

                        returns += (self.discount_factor ** delta_t) * rewards[x]

                else:
                    advantage = rewards[t] - state_values[t]
                    returns = rewards[t]

                # Store data for later training

                self.actor_experience_store.append(
                    (
                        states[t].detach(),
                        actions[t].detach(),
                        log_probs[t].detach(),
                        rewards[t],
                        new_states[t].detach(),
                        advantage.detach().cpu(),
                        returns
                    )
                )

    def update_exploration_rate(self):
        self.entropy_factor = max(self.entropy_factor - self.entropy_factor_decay
                                  * self.original_entropy_factor,
                                  self.min_entropy_factor)

    def generate_mini_batches(self):
        """
        Separate the gathered experiences into mini batches

        :returns: A list of minibatches, containing experience samples
        """
        shuffle(self.actor_experience_store)
        if len(self.actor_experience_store) % self.batch_size is not 0:
            raise ValueError('Batchsize must be a divisor of episode length')

        mini_batches = []

        for i in range(int(len(self.actor_experience_store) / self.batch_size)):

            mini_batches.append(
                self.actor_experience_store[i * self.batch_size: (i + 1) * self.batch_size]
            )

        return mini_batches

    def get_acting_network(self):
        return self.network

    def preprocess_actor_experience_samples(self, minibatch):
        states, actions, log_probs, rewards, new_states, advantages, returns = zip(*minibatch)

        # transform given tuples into tensors
        states = torch.cat(list(states)).cuda()

        actions = torch.cat(list(actions)).float().cuda()

        log_probs = torch.stack(log_probs).float().cuda()

        rewards = torch.tensor(list(rewards)).float().cuda()

        new_states = torch.cat(list(new_states)).cuda()

        advantages = torch.tensor(list(advantages)).cuda()

        returns = torch.tensor(list(advantages)).cuda()

        return states, actions, log_probs, rewards, new_states, advantages, returns

        