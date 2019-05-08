## Problem generation

There are two scripts for generating problem instances:  

* evaluation/generate_knapsack.py [options] _output-file_  
Generates a set of problem instances for the 0-1 knapsack problem and stores them as a .json-file _output-file_, for later use by apply_solver.

* evaluation/generate_tsp.py [options] _output-file_  
Generates a set of problem instances for the traveling salesman problem and stores them as a .json-file _output-file_, for later use by apply_solver.

## generate_knapsack

The generate_knapsack script generates --num-problems different problem instances.  
For each problem instance, the weight limit is uniformly sampled from [--min-weight-limit, --max-weigh-limit].  
Per problem instance, --num-items items are generated with uniform sampling, with item weights from [0, --max-item-weight] and item values from [0, --max-item-value].

## generate_tsp

The generate_tsp script generates --num-problems problem instances.  
Each problem instance has --num-cities nodes.  
Distances are uniformly sampled from [0, 1]. The generated distance matrix is symmetric.