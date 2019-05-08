"""
This module contains the main class for evolutionary algorithms
"""

from abc import ABC
from collections import deque
from math import log

import numpy as np
import torch

from input_output.stats_exporter import StatsExporter
from problems import ContinuousFunction
from reinforcement.reinforcement import ReinforcementLearningStrategy


class EvolutionaryAlgorithm(ABC):
    """
    A trainable evolutionary algorithm, going through the steps of:

    -Initialization
    -Parent selection
    -Crossover
    -Mutation
    -Survivor Selection

    How these steps are performed is based on the assigned strategy.
    """

    def __init__(self,
                 encoding_strategy,
                 crossover_strategy,
                 genome_creation_strategy,
                 mutation_strategy,
                 parent_pairing_strategy,
                 parent_selection_strategy,
                 repair_strategy,
                 survivor_selection_strategy,
                 training=False,
                 population_size=50,
                 num_generations=100):

        self.problem = None

        self.encoding_strategy = encoding_strategy
        self.parent_selection_strategy = parent_selection_strategy
        self.parent_pairing_strategy = parent_pairing_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.genome_creation_strategy = genome_creation_strategy
        self.population_size = population_size
        self.num_generations = num_generations
        self.repair_strategy = repair_strategy
        self.survivor_selection_strategy = survivor_selection_strategy
        training = training

        self.population = []

        self.strategies = [self.parent_selection_strategy,
                           self.parent_pairing_strategy,
                           self.crossover_strategy,
                           self.mutation_strategy,
                           self.survivor_selection_strategy]

    def solve(self,
              problems,
              training=False,
              stats_exporter=None,
              num_actors=1,
              train_per_generation=False):
        """
        Applies the evolutionary algorithm to all given problem instances.
        When training, the evoltuionary algorithm is applied num_actors times
        to each problem instance.

        :param problems: List of problem instances to be solved
        :param training: Whether the evolutionary algorithm is run in training or validation mode
        :stats_exporter: The stats exporter to store per-generation data
        :num_actors: Number of times, the ea is applied to each problem instance during training
        :train_per_generation: Whether to optimize strategies after every single iteration,
                               or after all runs of the evolutionary algorithm are completed.
                               Should be set to False, as long as you are using PPO

        :returns: Best solution found in the final generation
        """

        solutions = {}

        for problem_number, problem in enumerate(problems):

            self.problem = problem

            # Update strategies requiring problem information
            self.genome_creation_strategy.problem = problem
            self.encoding_strategy.problem = problem
            self.repair_strategy.problem = problem

            self.reset()

            if training:

                # actor_populations keeps track of the population managed by each actor
                actor_populations = []
                for actor in range(num_actors):

                    actor_populations.append(self.population)
                    self.reset()

                for actor in range(num_actors):
                    self.population = actor_populations[actor]

                    # Reward for actions taken in generation n
                    # to be used for rewarding agents in generation n+1
                    delayed_reward = None

                    for generation in range(self.num_generations):

                        delayed_reward = self.perform_generation(generation,
                                                                 delayed_reward=delayed_reward,
                                                                 training=True)

                        actor_populations[actor] = self.population

                        # During training, we only store stats for a single actor
                        if actor == num_actors-1:
                            fitnesses = self.get_population_fitnesses(evaluate_problem=True)

                            stats_exporter.store_stats(problem_number,
                                                       generation,
                                                       np.array(fitnesses))

                        if train_per_generation:
                            self.optimize_strategies()

            else:
                for generation in range(self.num_generations):

                    self.perform_generation(generation, training=training)

                    fitnesses = self.get_population_fitnesses(evaluate_problem=True)
                    stats_exporter.store_stats(problem_number, generation, np.array(fitnesses))

            solutions[problem_number] = self.get_fittest_individual()

        if not train_per_generation and training:
            self.optimize_strategies()

        if stats_exporter is not None:
            stats_exporter.write_run()

        return solutions

    def perform_generation(self, generation, delayed_reward=None, training=False):
        """
        Performs a single generation, consisting of:
        parent selection, crossover, mutation and survivor selection.
        Rewards the adaptation strategies, if training is performed.

        Strategies can only be rewarded upon reaching the next state of the markov decision process,
        which is in the following generation.
        Consequently, rewarding of strategies is always delayed by one generation.

        :param generation: The number of the current generation
        :param delayed_reward: The reward for the actions performed in the last generation
        :param training: Whether the evolutionary algorithm is used in training mode.
                         If True, strategies are rewarded and optimized

        :returns: Reward for actions performed in this generation
        """
        generations_left = (self.num_generations - generation) / self.num_generations

        # No need to reward when not training
        if not training:
            self.update_population_fitness()

            # Parent selection
            parents = self.parent_selection_strategy.select(self.population, generations_left)

            # Parent pairing
            parent_pairings = self.parent_pairing_strategy.select(parents, generations_left)

            # Crossover
            offspring = self.crossover_strategy.crossover(parent_pairings, generations_left)
            self.update_fitness_and_repair(offspring)

            # Mutation
            self.mutation_strategy.mutate(offspring, generations_left)
            self.update_fitness_and_repair(offspring)

            # Survivor Selection
            next_generation = self.survivor_selection_strategy.select(self.population,
                                                                      offspring,
                                                                      generations_left)

            self.population = next_generation
            self.update_population_fitness()

        # Reward, when training.
        # TODO: De-uglify this part of the code
        else:
            self.update_population_fitness()
            initial_best_fitness = self.get_best_fitness()

            # Parent selection

            if delayed_reward is not None:
                self.reward_strategy(self.parent_selection_strategy,
                                     delayed_reward,
                                     self.population,
                                     generations_left)

            parents = self.parent_selection_strategy.select(self.population, generations_left)

            # Parent pairing

            if delayed_reward is not None:
                self.reward_strategy(self.parent_pairing_strategy,
                                     delayed_reward,
                                     parents,
                                     generations_left)

            parent_pairings = self.parent_pairing_strategy.select(parents, generations_left)

            # Crossover

            if delayed_reward is not None:
                self.reward_crossover_strategy(self.crossover_strategy,
                                               delayed_reward,
                                               parent_pairings,
                                               generations_left)

            offspring = self.crossover_strategy.crossover(parent_pairings, generations_left)
            self.update_fitness_and_repair(offspring)

            # Mutation

            if delayed_reward is not None:
                self.reward_strategy(self.mutation_strategy,
                                     delayed_reward,
                                     offspring,
                                     generations_left)

            self.mutation_strategy.mutate(offspring, generations_left)
            self.update_fitness_and_repair(offspring)

            # Survivor selection

            if delayed_reward is not None:
                self.reward_strategy(self.survivor_selection_strategy,
                                     delayed_reward,
                                     self.population + offspring,
                                     generations_left)

            next_generation = self.survivor_selection_strategy.select(self.population,
                                                                      offspring,
                                                                      generations_left)

            self.population = next_generation
            self.update_population_fitness()

            final_best_fitness = self.get_best_fitness()

            next_delayed_reward = self.calc_reward(initial_best_fitness, final_best_fitness)

            if generation == self.num_generations - 1:
                # Give reward for transition into the terminal state
                # the given population does not really matter, because V(s_terminal) = 0

                self.reward_strategy(self.parent_selection_strategy,
                                     delayed_reward,
                                     self.population,
                                     generations_left)

                self.reward_strategy(self.parent_pairing_strategy,
                                     delayed_reward,
                                     parents,
                                     generations_left)

                self.reward_crossover_strategy(self.crossover_strategy,
                                               delayed_reward,
                                               parent_pairings,
                                               generations_left)

                self.reward_strategy(self.mutation_strategy,
                                     delayed_reward,
                                     offspring,
                                     generations_left)

                self.reward_strategy(self.survivor_selection_strategy,
                                     delayed_reward,
                                     self.population + offspring,
                                     generations_left)

            else:
                return next_delayed_reward

    def validate(self, problems, num_runs, weight_load_folder, stats_file=None):
        """
        Loads neural network weights from a specified folder
        and then applies the evolutionary algorithm num_runs times to each problem instance.

        Also prints mean and median performance statistics for all problem instances.

        :param problems: List of problem instances to be used for validation
        :param num_runs: Number of times to apply EA to each problem instance
        :param weight_load_folder: Folder to load neural network weights for strategies from
        :param stats_file: Filename of file to store stats in
        """
        self.load_weights(weight_load_folder)
        solution_store = []

        # Create stats exporter
        stats_exporter = None

        if stats_file is not None:

            stats_exporter = StatsExporter(len(problems),
                                           self.num_generations,
                                           self.population_size,
                                           stats_file)

        # Apply solver num_runs times
        for run in range(num_runs):

            if stats_exporter is not None:
                solutions = self.solve(problems, stats_exporter=stats_exporter)

            else:
                solutions = self.solve(problems)

            solution_store.append(solutions)

        # Print median and mean fitness of the best solution found for each problem instance
        for problem_number, problem in enumerate(problems):

                if isinstance(problem, ContinuousFunction):

                        solution_fitnesses = [
                            problem.evaluate(stored_solutions[problem_number], fitness=False)
                            for stored_solutions in solution_store
                        ]

                        mean_solution_fitness = -np.log10(
                                                    np.mean(np.maximum(1e-20,  solution_fitnesses))
                                                )
                        median_solution_fitness = -np.log10(
                                                    np.median(np.maximum(1e-20, solution_fitnesses))
                                                )

                else:
                    solution_fitnesses = [stored_solutions[problem_number].fitness
                                          for stored_solutions in solution_store]

                    mean_solution_fitness = np.mean(solution_fitnesses)
                    median_solution_fitness = np.median(solution_fitnesses)

                print(f'Problem {problem_number}: {median_solution_fitness} ; {mean_solution_fitness}')

    def train(self,
              problems,
              num_iterations,
              weight_store_folder,
              weight_store_iterations,
              stats_file=None, num_actors=1,
              train_per_generation=False):

        """
        Trains the evolutionary algorithm's strategies on a set of problem instances.

        Also prints mean and median performance statistics for all problem instances.

        :param problems: List of problem instances
        :param num_iterations: Number of training iterations to be performed
        :param weight_store_folder: Folder to store weights in
        :param weight_store_iterations: After how many iterations weights are to be stored
        :param stats_file: Name of file to store stats in
        :param num_actors: Number of times that the evolutionary algorithm is applied
                           to each problem instance per iteration
        :param train_per_generation: Whether to optimize the strategies after each generation,
                                     or after one complete iteration.
                                     Should be "False", as long as you are using PPO
        """

        # During training, the performance of our strategies changes over time
        # We store the solutions of the past 25 iterations to print stats
        solution_store = deque([], 25)        

        # Create stats exporter
        stats_exporter = None

        if stats_file is not None:

            stats_exporter = StatsExporter(len(problems),
                                           self.num_generations,
                                           self.population_size, stats_file)

        for iteration in range(num_iterations):

            # Store weights
            if iteration % weight_store_iterations == 0 and weight_store_folder is not None:
                self.store_weights(weight_store_folder, iteration)

            # Perform one iteration
            solutions = self.solve(problems,
                                   training=True,
                                   stats_exporter=stats_exporter,
                                   num_actors=num_actors,
                                   train_per_generation=train_per_generation
                                   )

            solution_store.append(solutions)

            # Output mean and median performance over the last 25 training iterations

            if iteration % weight_store_iterations == 0:

                print(f'iteration {iteration}')

                for problem_number, problem in enumerate(problems):

                    if isinstance(problem, ContinuousFunction):

                        solution_fitnesses = [
                            problem.evaluate(stored_solutions[problem_number], fitness=False)
                            for stored_solutions in solution_store
                            ]

                        mean_solution_fitness = -np.log10(
                                                    np.mean(np.maximum(1e-20,  solution_fitnesses))
                                                )
                        median_solution_fitness = -np.log10(
                                                    np.median(np.maximum(1e-20, solution_fitnesses))
                                                )
                    else:

                        solution_fitnesses = [
                            stored_solutions[problem_number].fitness
                            for stored_solutions in solution_store
                            ]

                        mean_solution_fitness = np.mean(solution_fitnesses)
                        median_solution_fitness = np.median(solution_fitnesses)

                    last_solution_fitness = solutions[problem_number].fitness

                    print(f'Problem {problem_number}: {last_solution_fitness} ; {median_solution_fitness}; {mean_solution_fitness}')

    def store_weights(self, weight_store_folder, iteration):
        """
        Store the neural network weights of all trainable strategies in the provided folder

        :param weight_store_folder: Name of folder to store weights in
        :param iteration: Number of current training iteration
        """
        for strategy in self.strategies:
            if isinstance(strategy, ReinforcementLearningStrategy):

                strategy.store_weights(weight_store_folder, iteration)

    def load_weights(self, weight_load_folder):
        """
        Loads the neural network weights of all trainable strateges from the provided folder
        """
        for strategy in self.strategies:
            if isinstance(strategy, ReinforcementLearningStrategy):

                strategy.load_weights(weight_load_folder)

    def optimize_strategies(self):
        """
        Use the reinforcement learning loss function to optimize strategies
        """
        for strategy in self.strategies:
            if isinstance(strategy, ReinforcementLearningStrategy):
                strategy.optimize_model()

    def calc_reward(self, old_best_fitness, new_best_fitness):
        """
        Calculate the reward for an action, based on the change in best fitness
        """
        if isinstance(self.problem, ContinuousFunction):
            return float(new_best_fitness - old_best_fitness)
        else:
            if old_best_fitness == 0:
                if new_best_fitness == 0:
                    return 0
                else:
                    return 100 * log(new_best_fitness, 10)
            if new_best_fitness == 0:
                return - 100 * abs(log(1 + old_best_fitness, 10))
            return 100 * log(new_best_fitness / old_best_fitness, 10)

    def reward_strategy(self, strategy, reward, new_state_population, generations_left):
        """
        Rewards a trainable strategy for its last action

        :param strategy: The strategy to reward
        :param reward: The reward for the strategies last action
        :param new_state_population: The ea-population, after the action was performed
        :param generations_left: The remaining number of generations
        """
        if isinstance(strategy, ReinforcementLearningStrategy):
            new_state = strategy.generate_encoded_state(new_state_population, generations_left)

            strategy.reward(reward, new_state)

    def reward_crossover_strategy(self, strategy, reward, new_state_pairs, generations_left):
        """
        Rewards a trainable crossover strategy for its last action.
        Crossover strategies do not operate on populations, but on parent pairs.
        Consequently, their new state is to be encoded differently.

        :param strategy: The strategy to reward
        :param reward: The reward for the strategies last action
        :param new_state_population: The ea-population, after the action was performed
        :param generations_left: The remaining number of generations
        """
        if isinstance(strategy, ReinforcementLearningStrategy):
            parents = list(zip(*new_state_pairs))
            
            # Encode first parents of each pair
            state_1 = strategy.generate_encoded_state(parents[0], generations_left)

            # Encode second parents of each pair
            state_2 = strategy.generate_encoded_state(parents[0], generations_left)

            # Combine partial states into complete state

            new_state = torch.cat([state_1, state_2], 1)

            strategy.reward(reward, new_state)

    def reset(self):
        """
        Initializes the population randomly
        """

        self.population = []
        for _ in range(self.population_size):
            new_genome = self.genome_creation_strategy.create()

            self.population.append(new_genome)

        self.repair_strategy.repair(self.population)

    def update_population_fitness(self, population=None):
        """
        Updates the fitness values of a population with regards to the given problem

        :parameter population: If None, the fitness of all indivudals in the self.population is
        updated.
        If not None, the fitness of all individuals in population are updated
        """
        # Run on own population, if no population is given
        if population is None:
            for individual in self.population:
                individual.fitness = self.problem.evaluate(individual)
        else:
            for individual in population:
                individual.fitness = self.problem.evaluate(individual)

    def update_fitness_and_repair(self, population):
        self.repair_strategy.repair(population)
        self.update_population_fitness(population)

    def get_best_fitness(self, population=None):
        """
        Returns the fitness of the fittest individual in given the population.
        If no population is provided, the current population of the evolutioanry algorithm is used.
        """
        if population is None:
            population = self.population

        fittest_individual = self.get_fittest_individual(population)

        return self.problem.evaluate(fittest_individual)

    def get_fittest_individual(self, population=None):
        """
        Returns the fittest individual in the given population.
        If no population is provided, the current population of the evolutioanry algorithm is used.
        """
        if population is None:
            population = self.population

        if isinstance(self.problem, ContinuousFunction):
            fittest_individual = max(population, key=lambda x: self.problem.evaluate(x))
        else:
            fittest_individual = max(population, key=lambda x: self.problem.evaluate(x))

        return fittest_individual

    def get_population_fitnesses(self, population=None, evaluate_problem=False):
        """
        Returns the fitness of the fittest individual in the population.
        If no population is provided, the current population of the evolutioanry algorithm is used.

        :param population: The population to select the fittest individual from
        :param evaluate_problem: Whether to return the fitness (False)
                                 or the value of the actual objective function (True)
        """
        if population is None:
            population = self.population

        if evaluate_problem:
            return [self.problem.evaluate(individual, fitness=False) for individual in population]

        else:
            return [individual.fitness for individual in population]
