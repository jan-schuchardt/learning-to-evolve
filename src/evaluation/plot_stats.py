"""
This script implements plotting of previously stored statistics.
Supports two modes:

Per-generation data for multiple iterations
or
Per-iteration data for multiple training sessions
"""

import os.path
import sys
from fnmatch import fnmatch
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy import loadtxt


def generate_datasets(file_names, best_individual_fn, feature_fn, options):
    """
    Loads data for later plotting from files with given names.


        :param file_names: List of file names to load data from
        :param best_individual_fn: Function used for selecting fittest individual
        :param feature_fn: Function to extract features across problems / runs (e.g. mean or median)
        :param options: Other options
        :returns: List of numpy arrays for later plotting
    """

    datasets = []

    for file_name_index, (file_name, label) in enumerate(zip(file_names[1:], options.labels)):

        if options.training_progress:
            root = file_name
            pattern = "stats.npy"

            iteration_datasets = {}

            # Load data from all training iterations
            for path, subdirs, files in os.walk(root):
                for name in files:
                    if fnmatch(name, pattern) and not '/training' in path:
                        #todo: Rename epoch folder to iteration folder (After old results are no longer needed)
                        key =  int(path[path.find('epoch')+5:])
                        iteration_datasets[key] = np.load(path + '/'  + name)

            iteration_datasets_ordered = []

            # Order data by their iteration number
            for key in sorted (iteration_datasets):
                iteration_datasets_ordered.append(iteration_datasets[key])

            
            # Combine iteration data, extract fittest individual per iteration
            data = np.stack(iteration_datasets_ordered, axis=0)
            #data = data[:, :, :, :-1, :]
            data = best_individual_fn(data[:, :, :, -1, :], axis=3)

            # Constrain data to specific run, problem instance or both
            # Extract features across remaining dimensions

            if options.run is not None and options.problem is not None:
                data = data[:, options.run, options.problem]
                datasets.append(feature_fn(data, axis=1))

            elif options.run is None and options.problem is not None:
                data = data[:, :, options.problem]
                datasets.append(feature_fn(data, axis=1))

            elif options.run is not None and options.problem is None:
                data = data[:, options.run, :]
                datasets.append(feature_fn(data, axis=1))

            elif options.run is None and options.problem is None:
                datasets.append(feature_fn(data, axis=(1,2)))

        elif options.merge:
            root = file_name
            pattern = 'stats.npy'

            if file_name_index == len(options.labels) -1:
                data = np.load(file_name)
                #data = data[:, :, :-1, :]
                data = best_individual_fn(data, axis=3)

                if options.problem is not None:
                    data = data[:, options.problem, :]
                    data = feature_fn(data, axis=(0))
                else:
                    data = feature_fn(data, axis=(0, 1))
                
                datasets.append(data)

            else:
                # Load data from all training iterations
                for path, subdirs, files in os.walk(root):
                    for name in files:
                        if fnmatch(name, pattern) and not '/training' in path and f'epoch{options.iteration}' in path:
                            #todo: Rename epoch folder to iteration folder (After old results are no longer needed)
                            
                            data = np.load(f'{path}/{name}')
                            #data = data[:, :, :-1, :]
                            data = best_individual_fn(data, axis=3)

                            if options.problem is not None:
                                data = data[:, options.problem, :]
                                data = feature_fn(data, axis=(0))
                            else:
                                data = feature_fn(data, axis=(0, 1))
                            
                            datasets.append(data)

        else:
            if not os.path.isfile(file_name):
                raise FileNotFoundError('File does not exist: ' + file_name)

            # Extract data of fittest individual
            data = np.load(file_name)
            #data = data[:, :, :-1, :]
            data = best_individual_fn(data, axis=3)

            # Constrain data to specific run, problem instance or both
            # Extract features across remaining dimensions

            if options.run is not None and options.problem is not None:
                data = data[options.run, options.problem, :]

            elif options.run is None and options.problem is not None:
                data = data[:, options.problem, :]
                data = feature_fn(data, axis=0)

            elif options.run is not None and options.problem is None:
                data = data[options.run, :, :]

                if not options.individual_problems:
                    data = feature_fn(data, axis=0)

            elif options.run is None and options.problem is None:
                if options.individual_problems:
                    data = feature_fn(data, axis=0)
                else:
                    data = feature_fn(data, axis=(0, 1))

            if options.individual_problems and data.ndim == 2:
                datasets.extend([problem_data for problem_data in data])
            else:
                datasets.append(data)

    return datasets

def generate_colors(datasets, options):
    """
    Generates graph color for each given datasets, using the color map specified in the options.
    Colors are assigned with uniform distances, between min_color and max_color of the colormap.

    :returns: List of RGB colors
    """
    color_map = plt.get_cmap(name=options.color_map_name)

    if options.merge:
        return [color_map(options.color_min)] * (len(datasets)-1) + [color_map(options.color_max)]

    colors = []

    num_colors = int(len(datasets) / options.num_agents)

    for i in range(num_colors):
        color_index = (i + 0.5) / num_colors
        color_index = (options.color_max - options.color_min) * color_index
        color_index = color_index + options.color_min
        color = color_map(color_index)
        colors.append(color)

    ret = []

    # Ensure that each agent of the same parameter combination uses the same color
    for i in range(len(colors)):
        ret.extend([colors[i]] * options.num_agents)

    return ret

def generate_labels(datasets, options):
    """
    Generates graph labels for all datasets.
    Appends the problem number to the labels specificed by the user, if individual problems are to be plotted.

    :return: List of label strings
    """
    if options.individual_problems:
        num_datasets_per_label = int(len(datasets) / len(options.labels))

        labels = []

        for i in range(len(options.labels)):
            label_datasets = datasets[i * num_datasets_per_label : i*num_datasets_per_label + num_datasets_per_label]

            for j in range(len(label_datasets)):
                labels.append(options.labels[i] + str(j))

        return labels

    else:
        return options.labels

def main(file_names, options):
    """
    Iterates over file_names, opens each associated file to generate plot
    """

    # Select functions to be used for plotting and feature extraction
    if options.median:
        feature_fn = np.median
    else:
        feature_fn = np.mean

    if options.use_min:
        best_individual_fn = np.min
    else:
        best_individual_fn = np.max

    if options.use_log:
        plot_fn = plt.semilogy
    else:
        plot_fn = plt.plot

    datasets = generate_datasets(file_names, best_individual_fn, feature_fn, options)

    if options.training_progress:
        colors = generate_colors(datasets[:-1], options)
    else:
        colors = generate_colors(datasets, options)

    labels = generate_labels(datasets, options)

    plots = []

    # Plot loaded data, using generated colors and labels
    if options.training_progress:
        x_axis = [options.num_training_iterations / (len(datasets[0])-1) * x for x in range(len(datasets[0]))]

        for data, color in zip(datasets[:-1], colors):
            plot = plot_fn(
                [options.num_training_iterations / (len(data)-1) * x for x in range(len(data))],
                data,
                color=color,
                linewidth=options.dot_size,
                linestyle='-',
                alpha=options.alpha
                )

            plots.append(plot)

        # The last dataset is reserved for the benchmark method and is plotted with a different style
        plot = plot_fn(
                x_axis,
                np.repeat(datasets[-1], len(x_axis), axis=0),
                color='black',
                linewidth=1.5,
                linestyle=':',
                )

        plots.append(plot)

        plt.xticks(x_axis)

    else:
        for data, color in zip(datasets, colors):

            plot = plot_fn(
                range(len(data))[0::options.resolution],
                data[0::options.resolution],
                color=color,
                linewidth=options.dot_size,
                alpha = options.alpha
            )

            plots.append(plot)

    # Axis labels and description
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(fontsize=options.axis_fontsize)
    plt.yticks(fontsize=options.axis_fontsize)

    plt.xlabel(options.xlabel, fontsize=options.axis_fontsize)
    plt.ylabel(options.ylabel, fontsize=options.axis_fontsize)

    # Assign labels to plots
    # Ensure that agents with the same combination of hyper-parameters receive the same label
    if options.merge:
        plt.legend(
            [plot[0] for plot in plots][::options.num_agents],
            labels,
            loc=options.legend_location,
            fontsize=options.legend_fontsize,
            title=options.legend_title,
            fancybox=True,
            ncol=options.legend_columns)

    else:
        plt.legend(
            [plot[0] for plot in plots][::options.num_agents],
            labels[::options.num_agents],
            loc=options.legend_location,
            fontsize=options.legend_fontsize,
            title=options.legend_title,
            fancybox=True,
            ncol=options.legend_columns)

    min_value = min([data.min() for data in datasets])
    max_value = max([data.max() for data in datasets])

    # Change range of values that are displayed in the plot (looks a bit nicer than default matplotlib)
    if options.use_log:
        plt.ylim(min_value * 0.5, max_value * 2)
    else:
        spread = max_value - min_value
        plt.ylim(min_value - 0.1 * spread, max_value + 0.15 * spread)


    plt.savefig(file_names[0], format=options.format, dpi=1080, bbox_inches='tight')


if __name__ == "__main__":
    usage = "usage: %prog [options] output_file input_files"
    parser = OptionParser(usage=usage)

    parser.add_option("--format",
                      dest="format",
                      help="The output file type. Options: svg, png, pdf",
                      choices=['svg', 'png', 'pdf'],
                      default='png',
                      type='choice')
    parser.add_option("--resolution",
                      dest="resolution",
                      help="Every nth step to take into consideration in plot",
                      default='1',
                      type='int')
    parser.add_option("--dot-size",
                      dest="dot_size",
                      help="Size of the dots of each individual graph",
                      default='1',
                      type='float')
    parser.add_option("--label",
                      action="append",
                      dest="labels")
    parser.add_option("--run",
                      dest="run",
                      help="The run number to process results from",
                      type=int)
    parser.add_option("--problem",
                      dest="problem",
                      help="The problem to process results from",
                      type=int)
    parser.add_option("--individual-problems",
                      dest="individual_problems",
                      help="Whether to plot data for each problem instance separately",
                      default=False,
                      action="store_true"
                     )
    parser.add_option("--median",
                      dest="median",
                      help="Whether to use median or mean for estimating performance",
                      default=False,
                      action="store_true"
                     ),
    parser.add_option("--log",
                      dest="use_log",
                      help="Whether to use logarithmic base 10 scaling of fitness-axis",
                      default=False,
                      action="store_true"
                     ),
    parser.add_option("--min",
                      dest="use_min",
                      help="Whether the best individual of a population has the lowest or highest fitness value",
                      default=False,
                      action="store_true"
                     )
    parser.add_option("--color-map",
                      dest="color_map_name",
                      help="Name of the colormap to use",
                      default="jet")
    parser.add_option("--color-min",
                      dest="color_min",
                      help="Lower limit of colors taken from the specified colormap (between 0 and 1)",
                      default='0',
                      type='float')
    parser.add_option("--color-max",
                      dest="color_max",
                      help="Upper limit of colors taken from the specified colormap (between 0 and 1)",
                      default='1',
                      type='float')
    parser.add_option("--xlabel",
                      dest="xlabel",
                      help="Label of X axis",
                      default="Generation")
    parser.add_option("--ylabel",
                      dest="ylabel",
                      help="Label of Y axis",
                      default="Mean best fitness")
    parser.add_option("--legend-location",
                      dest="legend_location",
                      help="Position of legend (use matplotlib keywords)",
                      default="lower right")
    parser.add_option("--legend-title",
                      dest="legend_title")
    parser.add_option("--legend-fontsize",
                      dest="legend_fontsize",
                      default="small")
    parser.add_option("--axis-fontsize",
                      dest="axis_fontsize",
                      default="medium")
    parser.add_option("--legend-columns",
                      dest="legend_columns",
                      help="Number of columns for entries of the plot's legend",
                      default=1,
                      type=int)
    parser.add_option("--training-progress",
                      dest="training_progress",
                      default=False,
                      help="Whether to plot per-iteration training or per-generation optimization data.",
                      action="store_true")
    parser.add_option("--merge",
                      dest="merge",
                      default=False,
                      help="Whether to plot per-generation for all subfolders or not.",
                      action="store_true")
    parser.add_option("--iteration",
                      dest="iteration",
                      help="Training iteration to take results from",
                      default=500,
                      type=int)
    parser.add_option("--num-training-iterations",
                      dest="num_training_iterations",
                      help="Number of iterations used for training. Used for entries on the x-axis",
                      default=100,
                      type=int)
    parser.add_option("--alpha",
                      dest="alpha",
                      default=1,
                      help="Opacity of the generated graphs",
                      type=float)
    parser.add_option("--num-agents",
                      dest="num_agents",
                      help="Number of agents trained with the same combination of hyper-parameters",
                      default=1,
                      type=int)

    (options, args) = parser.parse_args()

    if not args:
        parser.print_help()
        sys.exit(1)
    elif len(args) < 2:
        parser.print_help()
        sys.exit(1)
    elif len(options.labels) is not len(args)-1:
        print('Number of labels must match number of stats files')
        sys.exit(1)
    else:
        main(args, options)
