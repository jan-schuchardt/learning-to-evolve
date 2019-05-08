"""
This script implements generation of instances of the traveling salesman problem with
fully connected graphs.
"""

import json
import os.path
import sys
from optparse import OptionParser
from random import uniform

import numpy as np


def main(target_file, options):
    if os.path.isfile(target_file):
        print("File already exists")
        return

    else:
        with open(target_file, 'w') as outfile:

            problems = []

            for problem in range(options.num_problems):


                # Generate random adjacency matrix with edge weights \in [0,1]
                random_adjacency = np.random.rand(options.num_cities, options.num_cities)

                # Ensure symmetry
                problem = np.tril(random_adjacency) + np.tril(random_adjacency, -1).T

                problems.append(problem.tolist())

            json.dump(problems, outfile)


if __name__ == "__main__":
    usage = "usage: %prog [options] output-file"
    parser = OptionParser(usage=usage)

    parser.add_option("--num-problems",
                      dest="num_problems",
                      help="The number of problem instances to generate",
                      default=1,
                      type='int')
    parser.add_option("--num-cities",
                      dest="num_cities",
                      help="The number of cities per problem instance",
                      default=50,
                      type='int')

    (options, args) = parser.parse_args()

    if not args:
        parser.print_help()
        sys.exit(1)
    else:
        target_file = args[0]

    main(target_file, options)
