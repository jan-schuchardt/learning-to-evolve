"""
This script implements generation of instances of the one-dimensional 0-1 knapsack problem
"""

import json
import os.path
import sys
from optparse import OptionParser
from random import uniform


def main(target_file, options):
    """
    Generates knapsack problem instances and stores them in given json with name "target_file".
    Weights and weight limits are randomly chosen, based on the given options.
    """
    if os.path.isfile(target_file):
        print("File already exists")
        return

    else:
        with open(target_file, 'w') as outfile:

            problems = []

            for problem in range(options.num_problems):

                weights = []
                values = []

                for _ in range(options.num_items):
                    weight = uniform(0, options.max_item_weight)
                    value = uniform(0, options.max_item_value)

                    weights.append(weight)
                    values.append(value)

                weight_limit = uniform(options.min_weight_limit, options.max_weight_limit)

                problems.append((weight_limit, weights, values))

            json.dump(problems, outfile)


if __name__ == "__main__":
    usage = "usage: %prog [options] output-file"
    parser = OptionParser(usage=usage)

    parser.add_option("--max-weight-limit",
                      dest="max_weight_limit",
                      help="The highest possible weight limit",
                      default=4,
                      type='float')
    parser.add_option("--min-weight-limit",
                      dest="min_weight_limit",
                      help="The lowest possible weight limit",
                      default=4,
                      type='float')
    parser.add_option("--num-problems",
                      dest="num_problems",
                      help="The number of problem instances to generate",
                      default=1,
                      type='int')
    parser.add_option("--max-item-value",
                      dest="max_item_value",
                      help="The maximum value a single item can have",
                      default=1,
                      type='float')
    parser.add_option("--max-item-weight",
                      dest="max_item_weight",
                      help="The maximum weight a single item can have",
                      default=1,
                      type='float')
    parser.add_option("--num-items",
                      dest="num_items",
                      help="The number of items per problem instance",
                      default=100,
                      type='int')

    (options, args) = parser.parse_args()

    if not args:
        parser.print_help()
        sys.exit(1)
    else:
        target_file = args[0]

    main(target_file, options)
