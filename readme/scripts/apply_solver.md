# Training and Validation of Evolutionary Algorithms

apply_solver.py [options] _config-file_

is the main script of this project. It reads the _config-file_ to generate evolutionary algorithms, which are then trained and/or validated
on problem sets.
The default config file can be found under data/.

Currently, the program is only designed for learning:
* from a limited training set
* for a fixed number of generations
* for a specific problem type

There are seven key options:

* problem_type: Currently either [knapsack, travelling_salesman or continuous]. Specifies the problem type, thus selecting the baseline evolutionary algorithm to use.
* output_folder: Specifies where all neural network weights and stats files are to be stored.  
If you want to reuse generated weight files for later validation, set output_folder to the previously path, and specify a separate validation_output_folder.
* validation_output_folder: If != "", all stats files generated during validation are stored in this folder.
* train: Whether training should be performed
* validate: Whether validation
* training_problems: Specifies the file containing problem instances for training.
* validation_problems: Specifies the file containing problem instances for validation.

All other parameters are used to:
* Specify the components / strategies of the evolutionary algorithm
* Specify the parameters thereof
* Specify deep reinforcement learning hyper-parameters.

If you want to try multiple combinations of hparameter-values, surround parameter-values with [] to create a list (e.g.: "num_hidden_layers": [2,3,4]).  
All combinations of parameter values are then trained and/or validated.  
When multiple parameter values are tested, a hierarchical folder structure is automatically generated under output_folder, with one folder-level par varied parameter.

If you have trained for multiple combinations of parameters, but only want to validate for a subset of the parameter combinations:
* copy the config file
* remove all unwanted parameter values
* specify a validation_output_folder


## Training procedure:
Training is performed in the following way:

For each combination of parameters, **training_sessions** evolutionary algorithms are created.  
Each evolutionary algorithm is trained **num_iterations** on the training_problems set.  
In each iteration, an evolutionary algorithm is applied **num_actors** times to each training problem instance for **num_generations** generations.  
At the end of each iteration, **num_training_epochs** passes over all gathered experience samples are performed for optimizing the neural network.  
After every "**weight_store_iterations**"th iteration, the neural network weights are stored for later validation, and mean and median performance on all problem instances for the last few iterations is output.

## Validation procedure:
Validation is performed in the following way.

The neural networks of every "**weight_load_iterations**"th training iteration are loaded for each combination of paramters and every training session.  
Each evolutionary algorithm is applied **num_runs** times to each problem instance in the validation_problems set.  
The fitness of every individual, for every generation, for every problem instance, for every run, is stored in a stats.npy file.

## Parameters
This section provides a brief explanation of all available options in the config file.

### hyper-params
This section contains deep-reinforcement learning hyper-parameters.

* learning_rate_actor: The learning rate to use for gradient descent optimization of the neural network weights
* batch_size: The batch size to use for training on the experiences samples of every iteratation
* num_neurons: The number of neurons / filters of each hidden convolutional layer
* discount_factor: The discount factor \gamma of the markov decision process. Must be in (0, 1)
* variance_bias_factor: The variance-bias-tradeoff factor \tau of generalized advantage estimation. Must be in (0, 1). Smaller --> Less variance, more bias
* clipping_value: The clipping value \epsilon of proximal policy optimization. Must be in (0, \infty). Larger --> Bigger changes of the policy per epoch
* num_actors: Number of times an ea is applied to each problem training instance per iteration
* num_training_epochs: Number of optimization passes over the training samples gathered in one iteration
* dim_elimination_max_pooling: Whether to use max or mean pooling in the neural network.
* entropy_factor: The exploration-controlling coefficient \alpha_e of proximal policy optimization. Larger --> More exploration
* entropy_factor_decay: Decreases the entropy factor by entropy_factor_decay * (original entropy_factor) in each generation
* min_entropy_factor: Lower limit for decay of the entropy factor
* value_loss_factor: Ratio between actor and critic loss

### general
This section contains basic parameters of all evolutionary algorithms.

* problem_type: Either "knapsack", "continuous" or "travelling_salesman". Must match the training_problems and validation_problems!
* num_problem_dimensions: The number of problem dimensions. Only relevant for travelling_salesman, because the number of input channels is dependant on the problem dimensionality (when using the current encoding)
* num_generations: The number of generations to run the evolutionary algorithm for on each problem instance.
* population_size: The number of individuals in a population

### training
This section contains all parameters controlling the training procedure.

* output_folder: The folder to store weights and training stats.npy's in
* training_problems: A json containing the problem instances
* num_training_problems: The number of problems in the training set (Must be currently be set by hand)
* num_training_sessions: The number of evolutionary algorithms trained per combination of hyper-parameters.
* num_iterations: The number of training iterations
* weight_store_iterations: The number of iterations after which weights are to be stored
* train: Whether to train or not (If not, only the validation step is performed)
* train_per_generation: Whether optimization should be performed after every iteration or every generation. Should always be set to **False**, when using the current proximal-policy-optimization implementation!

### validation
This section contains all parameters controlling the validation procedure.

* validation_problems: A json containing the validation instances
* validation_output_folder: Optional, seperate folder for storing validation stats.npy files
* weight_load_iterations: Weights are loaded from the results of every "weight_load_iterations"th training iteration
* num_runs: The number of times to apply each evolutionary algorithm to every problem instance
* validate: Whether to perform validation or not. If False, only training is performed

### parent_selection
This section contains parameters for the parent selection strategy

* strategy: The strategy to use for parent selection.  
Current options: ["ppo", "ranked", "ppo_fitness_shaping"]
* percentage: The parent percentage to use for "ranked" and "ppo_fitness_shaping".


### parent_pairing
This section contains parameters for the parent pairing strategy. The baseline algorithm for continuous optimization does not use parent pairing, so a placeholder strategy is always used.

* strategy: The parent pairing strategy to use.  
Current options: ["ppo_fitness_shaping_tournament", "ppo_fitness_shaping_tournament"]

### crossover
This section contains parameters for the crossover strategy. The baseline algorithm for continuous optimization does not use crossover, so a placeholder strategy is always used.

* strategy: The crossover strategy to use.  
Current options: ["uniform", "onepoint", "twopoint", "linear", "cyclic", "order", "position", "partially_mapped", "ppo_operator_selection_global", "ppo_operator_selection_individual", "random_operator_selection"]
* crossover_rate: The probability for performing crossover instead of copying parents. Used by some operators

### mutation
This section contains paramters for the mutation strategy.

* strategy: The mutation strategy to use.  
Current options: ["binary_random", "ppo_binary_individual_mutation_rate_control", "ppo_binary_component_binary_mutation", "ppo_binary_global_mutation_rate_control", "inversion", "real_onestep", "real_kstep", "ppo_real_component_step_size_control", "ppo_real_individual_learning_parameter_control", "ppo_real_individual_step_size_control", "ppo_real_population_learning_parameter_control"]
* mutation_rate: The probability for mutation, used by some of the baseline strategies.
* initial_step_size: The initial step size for real-valued mutation for continuous optimization
* minimum_step_size: The lower limit for step sizes, when performing real-valued mutation for continuous optimization
* learning_parameter_evolutionary_strategy_1: The learning parameter used by "real_onestep" mutation and "real_kstep" mutation
* learning_parameter_evolutionary_strategy_2 : An additional learning parameter used by "real_kstep" mutation

### survivor_selection
This section contains paramters for the survivor selection strategy

* strategy: The survivor selection strategy to use.  
Current options: ["replacing", "ppo", "ranked"]. "Replacing" is the normal strategy used in my thesis. "Ranked" selects the fittest individuals from both the parent and offspring generation
* elite_size: The elite size for "replacing" survivor selection
