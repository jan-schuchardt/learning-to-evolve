"""
This script contains the functionality for applying solvers to problems
and to initiate training.
"""

import json
import sys
from copy import deepcopy
from itertools import product

from input_output.problem_loader import (ContinousFunctionLoader,
                                         KnapsackProblemLoader,
                                         TravellingSalesmanProblemLoader)
from solvers.factories import (ContinuousFunctionEvolutionaryAlgorithmFactory,
                               KnapsackEvolutionaryAlgorithmFactory,
                               TravellingSalesmanEvolutionaryAlgorithmFactory)


def load_config(config_file_name):
    """
    Returns config loaded from file with given file name
    """
    with open(config_file_name) as file:
        config = json.load(file)

    return config


def generate_factory_configs(config):
    """
    Splits up single config object into multiple configs for generating EA-factories,
    one per combination of parameters.
    Also returns subfolder names for each combination of parameters, and a combination
    of the parameter-value pairs (only for parameters with multiple values).

        :param config: Dict containing parameter-value or parameter-(value-list) pairs
        :returns: Tuple of factory configs, folder names and parameter-value pairs
    """
    iterable_parameters = []  # List of parameters with multiple values
    iterable_values = []

    for category_key in config:
        for param_key, param_value in config[category_key].items():
            if isinstance(param_value, list):
                iterable_parameters.append((category_key, param_key))
                iterable_values.append(param_value)

    factory_configs = []
    folder_names = []
    parameter_value_pairs = []

    # Generate one config object per combination of parameters
    for parameter_combination in product(*iterable_values):

        factory_config = deepcopy(config)
        folder_name = factory_config['training']['output_folder']

        parameter_value_pair = []

        for keys, value in zip(iterable_parameters, parameter_combination):
            factory_config[keys[0]][keys[1]] = value

            folder_name += f'/{keys[1]}_{value}'

            parameter_value_pair.append((keys[0], keys[1], value))

        factory_configs.append(factory_config)
        folder_names.append(folder_name)
        parameter_value_pairs.append(parameter_value_pair)

    return factory_configs, folder_names, parameter_value_pairs


def main(config_file_name):
    """
    Loads config from given file, creates solvers, which are then trained or validated
    """

    config = load_config(config_file_name)

    if config['general']['problem_type'] == 'continuous':
        problem_loader = ContinousFunctionLoader()        
        solver_factory = ContinuousFunctionEvolutionaryAlgorithmFactory()
    
    elif config['general']['problem_type'] == 'knapsack':
        problem_loader = KnapsackProblemLoader()
        solver_factory = KnapsackEvolutionaryAlgorithmFactory()

    elif config['general']['problem_type'] == 'travelling_salesman':
        problem_loader = TravellingSalesmanProblemLoader()
        solver_factory = TravellingSalesmanEvolutionaryAlgorithmFactory()

    else:
        raise ValueError('Invalid Problem Type')

    # Load problems

    training_problems = problem_loader.load(config['training']['training_problems'])

    validation_problems = problem_loader.load(config['validation']['validation_problems'])

    # Create EA-factory config and folder names for each combination of parameter-values

    factory_configs, folder_names, parameter_value_pairs = generate_factory_configs(config)

    for factory_config, folder_name, parameter_value_pair in zip(factory_configs,
                                                                 folder_names,
                                                                 parameter_value_pairs):

        print('___________________________________________________________________________________')
        print(f'{folder_name}\n')
        print(f'{parameter_value_pair}\n')

        # Perform training or validation for each training session
        # with a specific combination of parameters

        for training_session in range(factory_config['training']['num_training_sessions']):
            session_folder = folder_name + f'/session_{training_session}'

            weight_store_iterations = factory_config['training']['weight_store_iterations']
            num_iterations = factory_config['training']['num_iterations']

            print(f'Session: {training_session}\n')

            if factory_config['training']['train']:

                # Create solver, using this specific combination of parameter-values
                solver = solver_factory.create(True, factory_config)

                print(f'XXXXX training XXXXX\n')

                solver.train(
                    training_problems,
                    num_iterations,
                    session_folder + '/weights',
                    weight_store_iterations,
                    num_actors=factory_config['hyper_params']['num_actors'],
                    stats_file=session_folder + '/training/stats',
                    train_per_generation=config['training']['train_per_generation']
                )

            if factory_config['validation']['validate']:

                weight_load_iterations = config['validation']['weight_load_iterations']

                # Create solver, using this specific combination of parameter-values
                solver = solver_factory.create(False, factory_config)

                i = 0

                # Load trained weights from every (weight_load_iteration)s and validate
                while i * weight_load_iterations <= (
                            (num_iterations / weight_store_iterations) * weight_store_iterations
                ):

                    print(f'\nValidating iteration {i * weight_load_iterations}')

                    # Store validation results in standard folder,
                    # or in designated validation_output_folder

                    if config['validation']['validation_output_folder'] == "":

                        solver.validate(
                            validation_problems,
                            config['validation']['num_runs'],
                            session_folder + f'/weights/weights{i * weight_load_iterations}',
                            # todo: Rename epoch folder to iteration folder (After old results are no longer needed)
                            stats_file=session_folder + f'/validation/epoch{i * weight_load_iterations}/stats'
                        )

                    else:
                        stats_folder = config['validation']['validation_output_folder']
                        parameter_suffixes = session_folder[len(config['training']['output_folder']):]

                        solver.validate(
                            validation_problems,
                            config['validation']['num_runs'],
                            session_folder + f'/weights/weights{i * weight_load_iterations}',
                            #todo: Rename epoch folder to iteration folder (After old results are no longer needed)
                            stats_file=stats_folder + parameter_suffixes + f'/epoch{i * weight_load_iterations}/stats'
                        )
                    i += 1


if __name__ == "__main__":

    if len(sys.argv) is not 2:
        print(f'Usage: {sys.argv[0]} config-file')
        sys.exit(0)
    else:
        config_file_name = sys.argv[1]

    main(config_file_name)
