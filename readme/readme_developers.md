## Writing your own strategies

ppo.py, evolutionary.py and factories.py take care of most of the complexity,
so writing your own trainable strategies is straight-forward.

All adaptation strategies follow the same boilerplate-template (just look at any of the RL-strategies).

To create a new strategy, create a class inheriting from ReinforcementLearningStrategy (just copy any other strategy),
and overwrite create_distribution(), using the probability distribution you want.

All strategies use the same pattern:

1. Encode state using self.generate_encoded_state()
2. Input state into neural network, pass the actor output to create_distribution()
3. Sample from the generated solution
4. Store the state, sampled action, and the log-probability of the action in self.last_experience
5. Do something with your sampled action

After you have defined your strategy, you just have to add it to the EvolutionaryAlgorithmFactory of your choice in solvers/factories.py.  

1. Identifiy the select_XXX_strategy() method that your strategy belongs in
2. Add an if-statement, comparing the strategy name from the config file with the name you want to give your strategy
3. Call the constructor of your strategy with the parameter values from the config file.

Just copy-paste from any other strategy, factories.py is always following the same pattern.

## Adding new problem types

If you want to extend the code to new problem types, you have to go through the following steps:

1. Create a new class inheriting from the Problem abstract base class in problems.py. This can also be a wrapper for an external benchmark library.
2. Create a new ProblemLoader class in input_output/problem_loader that generates a list of problem instances
3. (Optional) Define a new genome type in genomes.py and a corresponding GenomeCreationStrategy in strategies/genome_creation.py
4. (Optional) Create new strategies that implement evolutionary operators for this problem type, as explained above
5. Create a new EvolutionaryALgorithmFactory in solvers/factories.py that returns a collection of strategies for this specific problem type
6. Initialize the ProblemLoader and EvolutionaryAlgorithmFactory in the beginning of the main-method in apply_solver.py

Note: You might also have to adjust the calc_reward method in evolutionary.py to ensure that appropriate rewards calculated based on the change in fitness.
The current implementation is not compatible with negative fitness values and might not handle some action cases that could appear with new problem types.