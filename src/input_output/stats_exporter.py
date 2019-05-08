"""
This module contains the StatsExporter class, which can be used to store
per-generation data for runs of a solver.
"""

import os.path

import numpy as np


class StatsExporter():
    """
    The StatsExporter class is used to buffer per-generation data for a single ongoing run of a solver.
    Once all data has been stored, it can be written to a specified file
    """
    def __init__(self, num_problems, num_generations, num_values, stats_file):
        """
            :param num_problems: Number of problem instances
            :param num_generations: Number of generations per run of a solver
            :param num_values: Number of values to be stored per generation
            :param stats_file: Name of file to store data in
        """   
        self.num_problems = num_problems
        self.num_generations = num_generations
        self.num_values = num_values
        
        self.stats_file = stats_file

        if os.path.isfile(stats_file):
            raise FileExistsError('Given stats file already exists')
        else:
            if not os.path.exists(os.path.dirname(stats_file)):
                os.makedirs(os.path.dirname(stats_file))

        self.data = np.empty((1, num_problems, num_generations, num_values))

    def store_stats(self, problem, generation, values):
        """
        Buffer values (e.g. fitness) for a specified generation on a specific problem instance
        """

        self.data[-1, problem, generation] = values

    def write_run(self):
        """
        Writes the buffered data, clears the buffer
        """
        np.save(self.stats_file, self.data)

        empty = np.empty((1, self.num_problems, self.num_generations, self.num_values))
        self.data = np.concatenate((self.data, empty), 0)

    def clear(self):
        """
        Removes all buffered data.
        """
        self.data = np.empty((1, self.num_problems, self.num_generations, self.num_values))
