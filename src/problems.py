"""
This module contains different problem types to be solved by our evolutioanry algorithms.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from itertools import compress

import numpy as np


class Problem(ABC):
    """
    A Problem is anything that has a given number of dimensions
    and can map a possible solution to a fitness
    """

    def __init__(self, num_dimensions):
        """
        :parameter num_dimensions: The number of dimensions of the problem.
        Equals the length of the phenotype of a solution genome.
        """
        self.num_dimensions = num_dimensions

    @abstractmethod
    def evaluate(self, solution, fitness=True):
        """
        :parameter solution: The solution to evaluate
        :return: The fitness of the solution
        """
        pass


class KnapSackProblem(Problem):
    """
    Implements the one-dimensional 0-1 knapsack problem

    The KnapSackProblem is a combinatorial optimization problem where there are
    n items, each associated with a value and weight.

    The goal is to maximize the value of all chosen items while keeping the the sum
    of the weights of all chosen items under a certain limit.
    """

    def __init__(self, item_weights, item_values, weight_limit):
        """
        Creates an instance of the KnapSackProblem.
        item_weights and item_values must be ordered iterables
        , with item_weights[n] and item_values[n] storing the weight and value
        for the same item n.
        """

        super().__init__(len(item_weights))

        self.weight_limit = weight_limit

        Item = namedtuple('Item', ['weight', 'value'])
        self.items = []

        for weight, value in zip(item_weights, item_values):
            self.items.append(Item(weight=weight, value=value))

    def evaluate(self, solution, fitness=True):
        item_selection = solution.decode().data

        picked_items = list(compress(self.items, item_selection))

        value = 0

        for item in picked_items:
            value += item.value

        return value

    def is_feasible(self, solution):
        """
        Check whether the given solution violates the weight constraint

        :return: True, if the solution does not violate the weight constraint.
        Else False
        """
        item_selection = solution.decode().data
        picked_items = list(compress(self.items, item_selection))

        weight_list = [item.weight for item in picked_items]

        return sum(weight_list) <= self.weight_limit

    def get_weights(self):
        """
        :return: A list of the weights of all items
        """
        return [item.weight for item in self.items]

    def get_values(self):
        """
        :return: A list of the values of all items
        """
        return [item.value for item in self.items]


class TravellingSalesmanProblem(Problem):
    """
    Implements the traveling salesman problem for fully connected graphs.
    The goal is to find a hamilton circle with maximum weights in the
    graph.

    Edge weights are encoded in an adjacency matrix.
    """
    def __init__(self, adjacency_matrix, check_feasibility):

        super().__init__(len(adjacency_matrix))

        self.adjacency_matrix = adjacency_matrix
        self.check_feasibility = check_feasibility

    def evaluate(self, solution, fitness=True):
        value = 0

        # Iterate over genome, adding up distance from node in chromosome k to chromosome k+1
        # Wrap around at end of genome
        for i, start in enumerate(solution.data[:-1]):
            value += self.adjacency_matrix[start, solution.data[i+1]]

        value += self.adjacency_matrix[solution.data[-1], solution.data[0]]

        return value

    def is_feasible(self, solution):
        """
        Returns, whether the solution is a permutation of node indeces.
        """
        if not self.check_feasibility:
            return True

        else:
            sorted_data = np.sort(solution.data)

            if np.array_equal(sorted_data, np.arange(len(sorted_data))):
                return True
            else:
                raise ValueError('Genomes should always be valid permutations')


class ContinuousFunction(Problem):
    """
    This is the base class for all continuous optimization functions.

    We differentiate between fitness calculation and evaluating the objective function.
    In the former case, values are clipped to max(1e-20, actual_value).
    In the latter, the unchanged value is used (e.g. for validation).

    TODO: Replace hard-caded function implementations with interface to some library.
    """

    def __init__(self, num_dimensions, constraints):
        """
        :param num_dimensions: Number of problem dimensions
        :param constraints: Iterable of tuples, each tuple containing a lower and upper limit for one problem dimension.
        """
        super().__init__(num_dimensions)
        self.constraints = constraints

    def is_feasible(self, solution):
        for constraint, value in zip(self.constraints, solution.data):
            if value < constraint[0] or value > constraint[1]:
                return False
        
        return True


class RastriginFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions < 1:
            raise ValueError('Unsupported number of dimensions')

        constraints = []

        for _ in range(num_dimensions):
            constraint = (-1, 1)
            constraints.append(constraint)

        super().__init__(num_dimensions, constraints)

    def evaluate(self, solution, fitness=True):
        x = solution.data * 5.12
        tmp = np.sum(np.square(x) - 10*np.cos(2 * np.pi * x))
        
        result = (10 * self.num_dimensions + tmp)

        if fitness:
            return -np.log10(np.maximum(1e-20, result))
        else:
            return result


class AckleyFunction(ContinuousFunction):
    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)

    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 5
        y = solution.data[1] * 5

        tmp_1 = -20 * np.exp(-0.2 *np.sqrt(0.5 * (x**2 + y**2)))
        tmp_2 = - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        result = tmp_1 + tmp_2 + np.e + 20

        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class SphereFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions < 1:
            raise ValueError('Unsupported number of dimensions')

        constraints = []

        for _ in range(num_dimensions):
            constraint = (-1, 1)
            constraints.append(constraint)

        super().__init__(num_dimensions, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data * 20

        result = np.sum(np.square(x))

        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class RosenbrockFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions < 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = []

        for _ in range(num_dimensions):
            constraint = (-1, 1)
            constraints.append(constraint)

        super().__init__(num_dimensions, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data * 10

        result = 0
        for i in range(self.num_dimensions - 1):

            result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i]**2) ** 2
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class BealeFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 4.5
        y = solution.data[1] * 4.5

        result = (1.5 - x + x*y) ** 2
        result += (2.25 -x + x*(y**2)) ** 2
        result += (2.625 -x + x * (y ** 3)) ** 2
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class GoldsteinPriceFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 2
        y = solution.data[1] * 2

        tmp_1 = (x + y + 1) ** 2
        tmp_1 *= (19 - 14*x + 3 * (x**2) - 14 * y + 6 * x * y + 3 * (y ** 2))
        tmp_1 += 1

        tmp_2 = (2 *x - 3 * y) ** 2
        tmp_2 *= (18 - 32 * x +12 * (x ** 2) + 48* y - 36 * x * y + 27 * (y ** 2))
        tmp_2 += 30

        result = tmp_1 * tmp_2
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result - 3))
        else:
            return result - 3


class BoothFunction(ContinuousFunction):

    def __init__(self, num_dimensions, fitness=True):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 10
        y = solution.data[1] * 10

        result = (x + 2 *y - 7) ** 2
        result += (2 * x  + y - 5) ** 2

        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class BukinFunctionN6(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 5 - 10
        y = solution.data[1] * 3

        result = 100 * np.sqrt(np.abs(y - 0.01 * (x ** 2)))
        result += 0.01 * np.abs (x + 10)
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class MatyasFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 10
        y = solution.data[1] * 10

        result = 0.26 * (x ** 2 + y **2) - 0.48 * x * y
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class LeviFunctionN13(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 10
        y = solution.data[1] * 10

        tmp_1 = np.sin(3 * np.pi * x) ** 2
        
        tmp_2 = (x - 1) ** 2
        tmp_2 *= (1 + np.sin(3 * np.pi * y) ** 2)

        tmp_3 = (y - 1) ** 2
        tmp_3 *= (1 + np.sin(2 * np.pi * y) ** 2)

        result = tmp_1 + tmp_2 + tmp_3
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result

    
class HimmelblausFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 5
        y = solution.data[1] * 5

        result = (x ** 2 + y - 11) ** 2 + (x + y **2 -7) **2

        
        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class ThreeHumpCamelFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 5
        y = solution.data[1] * 5

        result = 2 * x**2 - 1.05 * x**4 + (x**6) / 6 + x * y + y **2
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class EasomFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 100
        y = solution.data[1] * 100

        result = -1 * np.cos(x) * np.cos(y)
        result *= np.exp(-1 * ((x - np.pi) **2 + (y - np.pi) ** 2))

        if fitness:
            return -np.log10(np.maximum(1e-20,result + 1))
        else:
            return result + 1


class CrossInTrayFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 10
        y = solution.data[1] * 10

        tmp_1 = np.sqrt(x **2 + y **2) / np.pi
        tmp_1 = np.exp(np.abs(100 - tmp_1))

        tmp_2 = np.sin(x) * np.sin(y)

        result = np.abs(tmp_2 * tmp_1) +1
        result = result ** 0.1
        result *= -0.0001
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result + 2.06261187082274))
        else:
            return result + 2.06261187082274


class EggHolderFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 512
        y = solution.data[1] * 512

        tmp_1 = x / 2 + y + 47
        tmp_1 = np.sin(np.sqrt(np.abs(tmp_1)))
        tmp_1 = -1 * (y + 47) * tmp_1

        tmp_2 = x * np.sin(np.sqrt(np.abs(x - (y + 47))))

        result = tmp_1 - tmp_2

        if fitness:
            return -np.log10(np.maximum(1e-20,result + 959.640662720850743))
        else:
            return result + 959.640662720850743


class HoelderTableFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 10
        y = solution.data[1] * 10

        tmp_1 = 1 - np.sqrt(x **2 + y **2) / np.pi
        tmp_1 = np.exp(np.abs(tmp_1))

        tmp_2 = np.sin(x) * np.cos(y)

        result = -1 * np.abs(tmp_1 * tmp_2)

        if fitness:
            return -np.log10(np.maximum(1e-20,result + 19.20850256788676))
        else:
            return result + 19.20850256788676


class McCormickFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 2.75 + 1.25
        y = solution.data[1] * 3.5 + 0.5

        result = np.sin(x + y)
        result += (x - y) ** 2
        result -= 1.5 * x
        result += 2.5 * y + 1

        if fitness:
            return -np.log10(np.maximum(1e-20,result + 1.913222954981038))
        else:
            return result + 1.913222954981038

class SchafferFunctionN2(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 100
        y = solution.data[1] * 100

        tmp_1 = np.sin(x **2 - y**2) ** 2 - 0.5
        tmp_2 = (1 + 0.001 * (x**2 + y **2)) ** 2

        result = 0.5 + tmp_1 / tmp_2

        if fitness:
            return -np.log10(np.maximum(1e-20,result))
        else:
            return result


class SchafferFunctionN4(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions is not 2:
            raise ValueError('Unsupported number of dimensions')

        constraints = [(-1, 1), (-1, 1)]

        super().__init__(2, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data[0] * 100
        y = solution.data[1] * 100

        tmp_1 = np.cos(np.sin(np.abs(x **2 - y**2))) ** 2 - 0.5
        tmp_2 = (1 + 0.001 * (x**2 + y **2)) ** 2

        result = 0.5 + tmp_1 / tmp_2

        if fitness:
            return -np.log10(np.maximum(1e-20,result - 0.292579))
        else:
            return result - 0.292579


class StyblinskiTangFunction(ContinuousFunction):

    def __init__(self, num_dimensions):
        if num_dimensions < 1:
            raise ValueError('Unsupported number of dimensions')

        constraints = []

        for _ in range(num_dimensions):
            constraint = (-1, 1)
            constraints.append(constraint)

        super().__init__(num_dimensions, constraints)
    
    def evaluate(self, solution, fitness=True):
        x = solution.data * 5

        result = np.sum(x ** 4 - 16 * (x ** 2) + 5 * x) / 2
        
        if fitness:
            return -np.log10(np.maximum(1e-20,result + 39.16617 * self.num_dimensions))
        else:
            return result + 39.16617 * self.num_dimensions