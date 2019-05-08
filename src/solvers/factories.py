"""
This module contains factories for assembling evolutionary algorithms from different strategies.
It defines one AbstractFactory for all evolutionary algorithms,
and one factory per problem type.

This code is not as bad as it looks, it is just way too long,
because of the strategies' parameter lists.

In essence, all evolutionary algorithms are assembled from a set of strategies,
using the parameters passed from the config file.
"""

from abc import ABC, abstractmethod
from math import ceil

from solvers.encoding import (ContinuousFunctionEncodingStrategy,
                              KnapsackEncodingStrategy,
                              TravellingSalesmanEncodingStrategy)
from solvers.evolutionary import EvolutionaryAlgorithm

from strategies.crossover import (CopyCrossoverStrategy,
                                  CyclicCrossoverStrategy,
                                  LinearCrossoverStrategy,
                                  OnePointCrossoverStrategy,
                                  OrderCrossoverStrategy,
                                  PartiallyMappedCrossoverStrategy,
                                  PositionCrossoverStrategy,
                                  PPOGlobalCrossoverOperatorSelectionStrategy,
                                  PPOIndividualCrossoverOperatorSelectionStrategy,
                                  RandomCrossoverOperatorSelectionStrategy,
                                  TwoPointCrossoverStrategy,
                                  UniformCrossoverStrategy)

from strategies.general_purpose import PPOPopulationSubsetSelection

from strategies.genome_creation import (BinaryGenomeCreationStrategy,
                                        IntegerGenomeCreationStrategy,
                                        RealValuedGenomeCreationStrategy)

from strategies.mutation import (BinaryMutationStrategy,
                                 InversionMutationStrategy,
                                 KStepUncorrelatedMutation,
                                 OneStepUncorrelatedMutation,
                                 PPOComponentLevelBinaryMutation,
                                 PPOGlobalMutationRateControl,
                                 PPOIndividualMutationRateControl,
                                 PPOComponentLevelStepSizeControl,
                                 PPOPopulationLevelLearningParameterControl,
                                 PPOIndividualLevelLearningParameterControl,
                                 PPOIndividualLevelStepSizeControl)

from strategies.parent_pairing import (PPOFitnessShapingTournamentSelection,
                                       RandomSingleParentPairingStrategy,
                                       TournamentParentPairingStrategy)

from strategies.parent_selection import (RankedSelectionStrategy,
                                         PPOFitnessShapingRankedSelection)

from strategies.repair import (BinaryDropoutRepairStrategy,
                               RealValuedResamplingRepairStrategy,
                               TravellingSalesmanRepairStrategy)

from strategies.survivor_selection import (PPOSurvivorSelection,
                                           RankedSurvivorSelectionStrategy,
                                           ReplacingSurvivorSelectionStrategy)


class EvolutionaryAlgorithmFactory(ABC):
    """
    The abstract factory for creating evolutionary algorithms
    """

    def __init__(self, encoding_strategy):
        self.encoding_strategy = encoding_strategy
        self.training = False

    def create(self, training, options):
        """
        Creates an evolutionary algorithm, based on the provided options
        """
        self.training = training
        genome_creation_strategy = self.select_genome_creation_strategy(options)

        crossover_strategy = self.select_crossover_strategy(genome_creation_strategy,
                                                            options)

        mutation_strategy = self.select_mutation_strategy(options)

        parent_pairing_strategy = self.select_parent_pairing_strategy(options)

        parent_selection_strategy = self.select_parent_selection_strategy(options)

        repair_strategy = self.select_repair_strategy(options)

        survivor_selection_strategy = self.select_survivor_selection_strategy(options)

        return EvolutionaryAlgorithm(
            self.encoding_strategy,
            crossover_strategy,
            genome_creation_strategy,
            mutation_strategy,
            parent_pairing_strategy,
            parent_selection_strategy,
            repair_strategy,
            survivor_selection_strategy,
            population_size=options['general']['population_size'],
            num_generations=options['general']['num_generations']
        )

    @abstractmethod
    def select_crossover_strategy(self, genome_creation_strategy, options):
        """
        Returns the crossover strategy to be used, based on the problem type and the 
        provided options
        """
        pass

    @abstractmethod
    def select_genome_creation_strategy(self, options):
        """
        Returns the genome creation strategy to be used, based on the problem type and the 
        provided options
        """
        pass

    @abstractmethod
    def select_mutation_strategy(self, options):
        """
        Returns the mutation strategy to be used, based on the problem type and the 
        provided options
        """
        pass

    @abstractmethod
    def select_parent_pairing_strategy(self, options):
        """
        Returns the parent pairing strategy to be used, based on the problem type and the 
        provided options
        """
        pass

    @abstractmethod
    def select_parent_selection_strategy(self, options):
        """
        Returns the parent selection strategy to be used, based on the problem type and the 
        provided options
        """
        pass

    @abstractmethod
    def select_repair_strategy(self, options):
        """
        Returns the genome repair strategy to be used, based on the problem type and the 
        provided options
        """
        pass

    @abstractmethod
    def select_survivor_selection_strategy(self, options):
        """
        Returns the survivor selection strategy to be used, based on the problem type and the 
        provided options
        """
        pass


class ContinuousFunctionEvolutionaryAlgorithmFactory(EvolutionaryAlgorithmFactory):
    """
    The factory for creating evolutionary algorithms for performing continuous optimization
    """

    def __init__(self):
        encoding_strategy = ContinuousFunctionEncodingStrategy()
        super().__init__(encoding_strategy)

    def create(self, training, options):
        return super().create(training, options)

    def select_crossover_strategy(self, genome_creation_strategy, options):
        return CopyCrossoverStrategy(genome_creation_strategy)

    def select_genome_creation_strategy(self, options):
        one_step_strategies = [
            'real_onestep',
            'ppo_real_individual_learning_parameter_control',
            'ppo_real_individual_step_size_control',
            'ppo_real_population_learning_parameter_control'
        ]

        k_step_strategies = [
            'real_kstep',
            'ppo_real_component_step_size_control'
        ]

        mutation_strategy = options['mutation']['strategy']
        initial_step_size = options['mutation']['initial_step_size']

        if mutation_strategy in one_step_strategies:
            return RealValuedGenomeCreationStrategy(
                initial_step_size,
                control_type='one'
            )

        elif mutation_strategy in k_step_strategies:
            return RealValuedGenomeCreationStrategy(
                initial_step_size,
                control_type='k'
            )

        else:
            raise ValueError('Invalid mutation strategy')

    def select_mutation_strategy(self, options):
        mutation_strategy = options['mutation']['strategy']
        minimum_step_size = options['mutation']['minimum_step_size']

        if mutation_strategy == 'real_onestep':
            strategy = OneStepUncorrelatedMutation(
                options['mutation']['learning_parameter_evolutionary_strategy_1'],
                minimum_step_size
            )

        elif mutation_strategy == 'real_kstep':
            strategy = KStepUncorrelatedMutation(
                options['mutation']['learning_parameter_evolutionary_strategy_1'],
                options['mutation']['learning_parameter_evolutionary_strategy_2'],
                minimum_step_size
            )

        elif mutation_strategy == 'ppo_real_component_step_size_control':
            strategy = PPOComponentLevelStepSizeControl(
                 self.encoding_strategy,
                 options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                 options['general']['num_generations'],
                 training=self.training,
                 learning_rate=options['hyper_params']['learning_rate_actor'],
                 discount_factor=options['hyper_params']['discount_factor'],
                 variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                 num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                 num_neurons=options['hyper_params']['num_neurons'],
                 batch_size=options['hyper_params']['batch_size'],
                 clipping_value=options['hyper_params']['clipping_value'],
                 num_training_epochs=options['hyper_params']['num_training_epochs'],
                 dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                 entropy_factor=options['hyper_params']['entropy_factor'],
                 entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                 min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                 value_loss_factor=options['hyper_params']['value_loss_factor'],
                 minimum_step_size=options['mutation']['minimum_step_size'])

        elif mutation_strategy == 'ppo_real_individual_learning_parameter_control':

            strategy = PPOIndividualLevelLearningParameterControl(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor'],
                minimum_step_size=options['mutation']['minimum_step_size'])

        elif mutation_strategy == 'ppo_real_individual_step_size_control':

            strategy = PPOIndividualLevelStepSizeControl(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor'],
                minimum_step_size=options['mutation']['minimum_step_size'])

        elif mutation_strategy == 'ppo_real_population_learning_parameter_control':

            strategy = PPOPopulationLevelLearningParameterControl(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor'],
                minimum_step_size=options['mutation']['minimum_step_size'])

        else:
            raise ValueError('Invalid mutation strategy')

        return strategy

    def select_parent_pairing_strategy(self, options):
        return RandomSingleParentPairingStrategy(options['general']['population_size'])

    def select_parent_selection_strategy(self, options):
        parent_selection_strategy = options['parent_selection']['strategy']

        if parent_selection_strategy == 'ranked':
            return RankedSelectionStrategy(options['parent_selection']['percentage'], prefer_higher_fitness=True)

        elif parent_selection_strategy == 'ppo':

            strategy = PPOPopulationSubsetSelection(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                learning_rate=options['hyper_params']['learning_rate_actor'],
                min_subset_size=1,
                training=self.training,
                num_neurons=options['hyper_params']['num_neurons'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                batch_size=options['hyper_params']['batch_size'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        elif parent_selection_strategy == 'ppo_fitness_shaping':

            strategy = PPOFitnessShapingRankedSelection(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                options['parent_selection']['percentage'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor'],
                prefer_higher_fitness=True
            )

        else:
            raise ValueError('Invalid choice of parent selection strategy')

        return strategy

    def select_repair_strategy(self, options):
        return RealValuedResamplingRepairStrategy(options['mutation']['initial_step_size'])

    def select_survivor_selection_strategy(self, options):
        survivor_selection_strategy = options['survivor_selection']['strategy']

        if survivor_selection_strategy == 'replacing':
            strategy = ReplacingSurvivorSelectionStrategy(options['survivor_selection']['elite_size'], prefer_higher_fitness=True)

        elif survivor_selection_strategy == 'ppo':

            strategy = PPOSurvivorSelection(
                self.encoding_strategy,
                options['general']['population_size'],
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        elif survivor_selection_strategy == 'ranked':

            strategy = RankedSurvivorSelectionStrategy(prefer_higher_fitness=True)

        else:
            raise ValueError('Invalid chocie of survivor selection strategy')

        return strategy


class KnapsackEvolutionaryAlgorithmFactory(EvolutionaryAlgorithmFactory):
    """
    The factory for creating evolutionary algorithms for solving the 0-1 knapsack problem
    """

    def __init__(self):
        encoding_strategy = KnapsackEncodingStrategy()
        super().__init__(encoding_strategy)

    def create(self, training, options):
        return super().create(training, options)

    def select_crossover_strategy(self, genome_creation_strategy, options):

        if options['crossover']['strategy'] == 'uniform':
            strategy = UniformCrossoverStrategy(options['crossover']['crossover_rate'],
                                                genome_creation_strategy)

        else:
            raise ValueError('Invalid choice of crossover strategy')

        return strategy

    def select_genome_creation_strategy(self, options):

        return BinaryGenomeCreationStrategy()

    def select_mutation_strategy(self, options):

        mutation_strategy = options['mutation']['strategy']

        if mutation_strategy == 'binary_random':
            strategy = BinaryMutationStrategy(options['mutation']['mutation_rate'])

        elif mutation_strategy == 'ppo_binary_individual_mutation_rate_control':

            strategy = PPOIndividualMutationRateControl(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        elif mutation_strategy == 'ppo_binary_component_binary_mutation':

            strategy = PPOComponentLevelBinaryMutation(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        elif mutation_strategy == 'ppo_binary_global_mutation_rate_control':

            strategy = PPOGlobalMutationRateControl(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        else:
            raise ValueError('Invalid choice of mutation strategy')

        return strategy

    def select_parent_pairing_strategy(self, options):
        needed_num_children = options['general']['population_size']
        num_pairs = ceil(needed_num_children / 2)

        if options['parent_pairing']['strategy'] == 'tournament':
            strategy = TournamentParentPairingStrategy(num_pairs)

        elif options['parent_pairing']['strategy'] == 'ppo_fitness_shaping_tournament':
            strategy = PPOFitnessShapingTournamentSelection(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                num_pairs,
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        else:
            raise ValueError('Invalid choice of pairing strategy')

        return strategy

    def select_parent_selection_strategy(self, options):
        parent_selection_strategy = options['parent_selection']['strategy']
        learning_rate_actor = options['hyper_params']['learning_rate_actor']
        num_neurons = options['hyper_params']['num_neurons']
        num_hidden_layers = options['hyper_params']['num_hidden_layers']
        discount_factor = options['hyper_params']['discount_factor']

        if parent_selection_strategy == 'ppo':

            strategy = PPOPopulationSubsetSelection(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                learning_rate=learning_rate_actor,
                min_subset_size=2,
                training=self.training,
                num_neurons=num_neurons,
                num_hidden_layers=num_hidden_layers,
                discount_factor=discount_factor,
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                batch_size=options['hyper_params']['batch_size'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        elif parent_selection_strategy == 'ranked':

            strategy = RankedSelectionStrategy(
                options['parent_selection']['percentage']
            )
        
        else:
            raise ValueError('Invalid choice of parent selection strategy ')

        return strategy

    def select_repair_strategy(self, options):
        
        return BinaryDropoutRepairStrategy()

    def select_survivor_selection_strategy(self, options):
        elite_size = options['survivor_selection']['elite_size']
        survivor_selection_strategy = options['survivor_selection']['strategy']

        if survivor_selection_strategy == 'replacing':
            strategy = ReplacingSurvivorSelectionStrategy(elite_size=elite_size)

        elif survivor_selection_strategy == 'ppo':

            strategy = PPOSurvivorSelection(
                self.encoding_strategy,
                options['general']['population_size'],
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        elif survivor_selection_strategy == 'ranked':

            strategy = RankedSurvivorSelectionStrategy(prefer_higher_fitness=True)

        else:
            raise ValueError('Invalid chocie of survivor selection strategy')

        return strategy


class TravellingSalesmanEvolutionaryAlgorithmFactory(EvolutionaryAlgorithmFactory):
    """
    The factory for creating evolutionary algorithms for solving the traveling salesman problem
    """

    def __init__(self):
        encoding_strategy = TravellingSalesmanEncodingStrategy()
        super().__init__(encoding_strategy)

    def create(self, training, options):
        self.encoding_strategy.num_problem_dimensions = options['general']['num_problem_dimensions']
        return super().create(training, options)

    def select_crossover_strategy(self, genome_creation_strategy, options):

        if options['crossover']['strategy'] == 'onepoint':
            strategy = OnePointCrossoverStrategy(options['crossover']['crossover_rate'], genome_creation_strategy)

        elif options['crossover']['strategy'] == 'twopoint':
            strategy = TwoPointCrossoverStrategy(options['crossover']['crossover_rate'], genome_creation_strategy)

        elif options['crossover']['strategy'] == 'linear':
            strategy = LinearCrossoverStrategy(options['crossover']['crossover_rate'], genome_creation_strategy)

        elif options['crossover']['strategy'] == 'cyclic':
            strategy = CyclicCrossoverStrategy(options['crossover']['crossover_rate'], genome_creation_strategy)

        elif options['crossover']['strategy'] == 'position':
            strategy = PositionCrossoverStrategy(options['crossover']['crossover_rate'], genome_creation_strategy)

        elif options['crossover']['strategy'] == 'order':
            strategy = OrderCrossoverStrategy(options['crossover']['crossover_rate'], genome_creation_strategy)

        elif options['crossover']['strategy'] == 'partially_mapped':
            strategy = PartiallyMappedCrossoverStrategy(options['crossover']['crossover_rate'], genome_creation_strategy)

        elif options['crossover']['strategy'] == 'ppo_operator_selection_global':
            strategy = PPOGlobalCrossoverOperatorSelectionStrategy(
                self.encoding_strategy,
                genome_creation_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor'],
                crossover_rate=options['crossover']['crossover_rate']
            )

        elif options['crossover']['strategy'] == 'ppo_operator_selection_individual':
            strategy = PPOIndividualCrossoverOperatorSelectionStrategy(
                self.encoding_strategy,
                genome_creation_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor'],
                crossover_rate=options['crossover']['crossover_rate']
            )

        elif options['crossover']['strategy'] == 'random_operator_selection':
            strategy = RandomCrossoverOperatorSelectionStrategy(
                options['crossover']['crossover_rate'],
                genome_creation_strategy
            )
        
        else:
            raise ValueError('Invalid choice of crossover strategy')

        return strategy
            
    def select_genome_creation_strategy(self, options):

        return IntegerGenomeCreationStrategy()

    def select_mutation_strategy(self, options):

        mutation_strategy = options['mutation']['strategy']

        if mutation_strategy == 'inversion':
            strategy = InversionMutationStrategy(options['mutation']['mutation_rate'])

        else:
            raise ValueError('Invalid choice of mutation strategy')

        return strategy

    def select_parent_pairing_strategy(self, options):
        needed_num_children = options['general']['population_size']

        num_pairs = needed_num_children

        if options['parent_pairing']['strategy'] == 'tournament':
            strategy = TournamentParentPairingStrategy(num_pairs)

        elif options['parent_pairing']['strategy'] == 'ppo_fitness_shaping_tournament':
            strategy = PPOFitnessShapingTournamentSelection(
                self.encoding_strategy,
                options['hyper_params']['num_actors'] * options['training']['num_training_problems'],
                options['general']['num_generations'],
                num_pairs,
                training=self.training,
                learning_rate=options['hyper_params']['learning_rate_actor'],
                discount_factor=options['hyper_params']['discount_factor'],
                variance_bias_factor=options['hyper_params']['variance_bias_factor'],
                num_hidden_layers=options['hyper_params']['num_hidden_layers'],
                num_neurons=options['hyper_params']['num_neurons'],
                batch_size=options['hyper_params']['batch_size'],
                clipping_value=options['hyper_params']['clipping_value'],
                num_training_epochs=options['hyper_params']['num_training_epochs'],
                dim_elimination_max_pooling=options['hyper_params']['dim_elimination_max_pooling'],
                entropy_factor=options['hyper_params']['entropy_factor'],
                entropy_factor_decay=options['hyper_params']['entropy_factor_decay'],
                min_entropy_factor=options['hyper_params']['min_entropy_factor'],
                value_loss_factor=options['hyper_params']['value_loss_factor']
            )

        else:
            raise ValueError('Invalid choice of pairing strategy')

        return strategy

    def select_parent_selection_strategy(self, options):
        parent_selection_strategy = options['parent_selection']['strategy']

        if parent_selection_strategy == 'ranked':

            strategy = RankedSelectionStrategy(
                options['parent_selection']['percentage']
            )

        else:
            raise ValueError('Invalid choice of parent selection strategy ')

        return strategy

    def select_repair_strategy(self, options):

        return TravellingSalesmanRepairStrategy()

    def select_survivor_selection_strategy(self, options):
        elite_size = options['survivor_selection']['elite_size']
        survivor_selection_strategy = options['survivor_selection']['strategy']

        if survivor_selection_strategy == 'replacing':
            strategy = ReplacingSurvivorSelectionStrategy(elite_size=elite_size)

        elif survivor_selection_strategy == 'ranked':

            strategy = RankedSurvivorSelectionStrategy(prefer_higher_fitness=True)

        else:
            raise ValueError('Invalid chocie of survivor selection strategy')

        return strategy
