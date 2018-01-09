# GeneticNeuralNetworks

Usually, feedforward neural networks are optimized by backpropagation method, to get the derivatives of weights and biases, and optimize them during each iteration of stochastic gradient descending. However, sometimes it is difficult to get the derivatives of the cost functions and sometimes the cost functions are not the same with the metrics. There are alternatives for backpropagation methods, such as genetic algorithms which can search for optimal parameters using iterations of populations and generations.

The conventional genetic algorithm can generate a large population of specified neural networks with different parameters, each neural network with a specific parameter combinations is an individual. Then through calculation of fitnesses, cross-over, mutations, the algorithm tries to evolve the population in terms of the fitness.

An important factor of applying genetic algorithm to real-world problem is encoding, namely encoding the key factors into genes, such as binary coding, namely encode a parameter as a series of 0s or 1s.  However, In this project, we are going to encode the weights and biases with quantums. The encoding method refers to Dr. Wang Zhiteng's [paper](https://wenku.baidu.com/view/8fe5bc385a8102d276a22f8b.html).

Conventional Genetic Algorithm:
Steps:
- Create a population consist of individual neural network with randomly assigned weights
- Calculate the fitness of each individual according to the accuracy of prediction
- Select a proportion of the population and make crossover between two individuals, replace the parents by the children
- Muate several individuals at random points
- Repeat the process until the loop ends or the ideal weights are found

Unlike conventional genetic algorithm, there are four major steps in a quantum genetic algorithm:
- Create Population, randomly create parameters and generate neural networks according to the parameters
- Calculate fitness, calculate the fitness of each individual neural network
- Quantum rotating gates, compared to conventional genetic algorithm, QGA uses quantum rotating gates to update chromosomes.
- Mutate, select certain individual neural networks and make mutations on their chromosomes in terms of NOT gating.

Environment:
- Anconda Python 3.5
- Win10
- Intel i5

Packages:
- Numpy
- Scikit-learn
- Copy

Dataset:
- Iris
- MNIST

Note, genetic algorithms require intensive computing, because of its parallel charateristics, it will be better to run it on clusters. 
Still, more work to be continued....
