"""
This module contains functionality for loading problem sets from files
"""

import json

import numpy as np

from problems import *


class KnapsackProblemLoader():

    def load(self, file_name):
        with open(file_name, 'r') as infile:
            problem_data = json.load(infile)

        problems = []

        for problem in problem_data:
            weight_limit = problem[0]
            item_weights = problem[1]
            item_values = problem[2]

            problems.append(KnapSackProblem(item_weights, item_values, weight_limit))

        return problems


class TravellingSalesmanProblemLoader():

    def load(self, file_name):
        with open(file_name, 'r') as infile:
            problem_data = json.load(infile)

        problems = []

        for problem in problem_data:

            problems.append(TravellingSalesmanProblem(np.array(problem), False))

        return problems


class ContinousFunctionLoader():

    function_type_mapping = {
        'rastrigin': RastriginFunction,
        'ackley': AckleyFunction,
        'sphere': SphereFunction,
        'rosenbrock': RosenbrockFunction,
        'beale': BealeFunction,
        'goldstein-price': GoldsteinPriceFunction,
        'booth': BoothFunction,
        'bukin-6': BukinFunctionN6,
        'matyas': MatyasFunction,
        'levi-13': LeviFunctionN13,
        'himmelblau': HimmelblausFunction,
        'camel-3': ThreeHumpCamelFunction,
        'easom': EasomFunction,
        'cross-in-tray': CrossInTrayFunction,
        'eggholder': EggHolderFunction,
        'hoelder': HoelderTableFunction,
        'mccormick': McCormickFunction,
        'schaffer-2': SchafferFunctionN2,
        'schaffer-4': SchafferFunctionN4,
        'styblinski': StyblinskiTangFunction
    }

    def load(self, file_name):
        with open(file_name, 'r') as infile:
            problem_data = json.load(infile)

        problems = []

        for problem in problem_data:
            function_type = problem[0]
            num_dimensions = problem[1]

            if function_type not in ContinousFunctionLoader.function_type_mapping:
                raise ValueError('Invalid problem type: ' + function_type)

            else:
                function_class = ContinousFunctionLoader.function_type_mapping[function_type]
                problems.append(function_class(num_dimensions))

        return problems
