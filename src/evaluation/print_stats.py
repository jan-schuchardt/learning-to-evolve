"""
This module implements plotting of previously stored statistics
"""

import os
import sys
from fnmatch import fnmatch
from optparse import OptionParser

import numpy


def print_latex_table_row(data, digits, notation_char, generations):
    print(' & '.join(
        ['{:.{digits}{notation}}'.format(
            x, digits=digits, notation=notation_char)
         for x in data[generations]]
        ) + '\\\\')


def print_merged_files(file_names, mean_function, notation_char, best_selection_function, options):
    data_sets = []
    for file_name in file_names:
        root = file_name
        pattern = "stats.npy"

        for path, subdirs, files in os.walk(root):

            for name in files:
                #todo: Rename epoch folder to iteration folder (After old results are no longer needed)
                if fnmatch(name, pattern) and not '/training' in path and f'epoch{options.iteration}' in path:

                    data_sets.append(numpy.load(path + '/' + name))


    data = numpy.stack(data_sets)

    #  data = data[:, :, :, :-1, :]

    data = best_selection_function(data, axis=4)

    if options.individual_problems:
        data = mean_function(data, axis=(1))
    else:
        data = mean_function(data, axis=(1, 2))

    if options.use_log:
        data = numpy.log10(data)

    if options.individual_problems:
        for i in range(len(data[0, :])):
            problem_data = data[:, i]
            print('_________')
            minimum = numpy.min(problem_data, axis=0)
            mean = numpy.mean(problem_data, axis=0)
            median = numpy.median(problem_data, axis=0)
            maximum = numpy.max(problem_data, axis=0)

            metrics = [minimum, mean, median, maximum]
            for metric in metrics:
                print_latex_table_row(
                    metric,
                    options.num_digits,
                    notation_char,
                    options.generations)

    else:
        minimum = numpy.min(data, axis=0)
        mean = numpy.mean(data, axis=0)
        median = numpy.median(data, axis=0)
        maximum = numpy.max(data, axis=0)

        metrics = [minimum, mean, median, maximum]
        for metric in metrics:
            print_latex_table_row(
                metric,
                options.num_digits,
                notation_char,
                options.generations)


def print_seperate_files(file_names, mean_function, notation_char, best_selection_function, options):
    for file_name in file_names:
        print('')
        """
        Iterates over file_names, opens each associated file to generate plot

        :param file_names: iterable of strings
        """

        data = numpy.load(file_name)
        #  data = data[:, :, :-1, :]
        data = best_selection_function(data, axis=3)

        if options.run is None and options.problem is None:

            if options.individual_problems:
                data = mean_function(data, axis=0)
                if options.use_log:
                    data = numpy.log10(data)
                for problem_data in data[:]:
                    print_latex_table_row(
                        problem_data,
                        options.num_digits,
                        notation_char,
                        options.generations)

            else:
                data = mean_function(data, axis=(0, 1))
                if options.use_log:
                    data = numpy.log10(data)
                print_latex_table_row(
                    data,
                    options.num_digits,
                    notation_char,
                    options.generations)

        elif options.run is None and options.problem is not None:

            data = data[:, options.problem, :]
            data = mean_function(data, axis=0)
            if options.use_log:
                data = numpy.log10(data)
            print_latex_table_row(
                data,
                options.num_digits,
                notation_char,
                options.generations)

        elif options.run is not None and options.problem is None:
            data = data[options.run, :, :]

            if options.individual_problems:
                if options.use_log:
                    data = numpy.log10(data)
                for problem_data in data[:]:
                    print_latex_table_row(
                        problem_data,
                        options.num_digits,
                        notation_char,
                        options.generations)

            else:
                data = mean_function(data, axis=0)
                if options.use_log:
                    data = numpy.log10(data)
                print_latex_table_row(
                    data,
                    options.num_digits,
                    notation_char,
                    options.generations)

        elif options.run is not None and options.problem is not None:

            data = data[options.run, options.problem, :]
            if options.use_log:
                data = numpy.log10(data)
            print_latex_table_row(
                data,
                options.num_digits,
                notation_char,
                options.generations)


def main(file_names, options):
    if options.median:
        mean_function = numpy.median
    else:
        mean_function = numpy.mean

    if options.use_scientific_notation:
        notation_char = 'E'
    else:
        notation_char = 'f'

    if options.use_min:
        best_selection_function = numpy.min
    else:
        best_selection_function = numpy.max

    if options.merge:
        print_merged_files(file_names, mean_function, notation_char, best_selection_function, options)

    else:
        print_seperate_files(file_names, mean_function, notation_char, best_selection_function, options)


if __name__ == "__main__":
    usage = "usage: %prog [options] input_files"
    parser = OptionParser(usage=usage)

    parser.add_option("-g", "--generation",
                      action="append",
                      type='int',
                      dest="generations")
    parser.add_option("--run",
                      dest="run",
                      help="The run to process results from",
                      type=int)
    parser.add_option("--problem",
                      dest="problem",
                      help="The problem to process results from",
                      type=int)
    parser.add_option("--individual-problems",
                      dest="individual_problems",
                      help="Whether to keep data on all problems are just take mean",
                      default=False,
                      action="store_true"
                     )
    parser.add_option("--median",
                      dest="median",
                      help="Whether to use median or mean for estimating performance",
                      default=False,
                      action="store_true"
                     ),
    parser.add_option("--digits",
                      dest="num_digits",
                      help="",
                      default=3,
                      type=int)
    parser.add_option("--iteration",
                      dest="iteration",
                      help="",
                      default=None,
                      type=int),
    parser.add_option("--scientific",
                      dest="use_scientific_notation",
                      help="",
                      default=False,
                      action="store_true"
                     )
    parser.add_option("--min",
                      dest="use_min",
                      help="",
                      default=False,
                      action="store_true"
                     )
    parser.add_option("--log",
                      dest="use_log",
                      help="",
                      default=False,
                      action="store_true"
                     )
    parser.add_option("--merge",
                      dest="merge",
                      help="",
                      default=False,
                      action="store_true"
                     )

    (options, args) = parser.parse_args()

    if not args:
        parser.print_help()
        sys.exit(1)
    else:
        main(args, options)
