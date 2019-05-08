## Todo

There, there are a few implementation-specific aspects to be considered for future work.

### Population / Individual representation
Currently, each individual is an object, the population of an evolutionary algorithm is represented as a list of individuals.  
This is nice and object-orientied. However, it limits efficiency.  
It would be better to encode the entire population in a torch tensor or numpy array, and implement all operators using efficient array operations.  
This would also eliminate the step of generating new encodings in every generation.

### Parallelization
One nice thing about evolutionary algorithms is that they are embarassingly parallel.  
However, the current implementation does not use multi-threading in any way.
Strategies should ideally be parallelized.
Instead of running evolutionary sequentially during training, they should be run in parallel (Given enough RAM / VRAM).

### Better decoupling of PPO and the evolutionary algorithm
Currently, the "num_actors" parameter of all ppo-based methods is set to num_actors * num_training/validation_problems of the config file.  
The "episode_length" parameter of all ppo-based methods is set to "num_generations" of the config file.  
This ensures that advantage estimates are always calculated for one run of the evolutionary algorithm and then stored, before the next run of the evolutioanry algorithm.  
This is a bit hacky and a better way needs to be found, if there is to be an unlimited number of problem instances, a varying number of generations or something else.

### Interface to continuous objective functions
The different continuous objective functions are currently hardcoded, which makes problems.py pretty ugly.  
It would be better to create an interface to a library, e.g. https://github.com/DEAP/deap

### Eliminate strategy parameter lists
All RL-based strategies have really long parameter lists.  
If we want solvers/factories.py to be prettier, the parameter lists should be replaced with **kwargs.

### Training with large training sets
If we want to train with large or randomly generated training sets, iterating over all problem instances in each iteration would make training super inefficient,
as the policies only changes to a limited degree in each generation, due to the use of PPO.
Instead, a subset of limited size should be sampled from the training set in each generation.
