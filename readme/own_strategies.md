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