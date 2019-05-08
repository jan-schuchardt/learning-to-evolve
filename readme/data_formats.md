## Data formats

This readme describes the file formats used by the different problem types.  
It also explains the file format used by the stats exporter for generating stats.npy files.

### Knapsack problem

A knapsack problem set is a json file, containing an array of problem instances.  
Each problem instance is an array, containing:
* A weight limit
* An array of item weights
* An array of item values

### Traveling salesman problem

A traveling salesman problem set is a json file, containing an array of problem instances.  
Each problem instance is a symmetric distance matrix, encoded as an array of arrays.

### Continuous objective functions

Currently, there is only a limited number of hard-coded objective functions available.  
Problem files only serve to specify which objective functions should be used for training / iteration.

A continuous problem set is a json file, containing an array of problem instances.  
Each problem instance is defined by an array, consisting of:

* The name of the objective function
* The number of objective function dimensions (Should be set to 2. A few objective functions can also be used with a higher number of dimensions)

Example: [["rastrigin", 2], ["rosenbrock",2]]

### stats.npy

All stats files are saved 4-dimensional numpy arrays. Stat files are currently used to store the fitness of every individual, in every generation, for every problem, for every run / iteration.

Consequently, the array-dimensions are:

* run (when validating) / iteration (when training)
* problem instance (ordered as in the problem set)
* generation
* individual