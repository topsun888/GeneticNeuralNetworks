# GeneticNeuralNetworks
Use genetic algorithm to update weights rather than backpropagation method. We can view the weights and biases as the genes of a specified neural network, all we need is to search for optimal weights and biases through generations' iterations.
Steps:
- Create a population consist of individual neural network with randomly assigned weights
- Calculate the fitness of each individual according to the accuracy of prediction
- Select a proportion of the population and make crossover between two individuals, replace the parents by the children
- Muate several individuals at random points
- Repeat the process until the loop ends or the ideal weights are found

Package:
- Python Numpy
- Copy

Note, genetic algorithms require intensive computing, because of its parallel charateristics, it will be better to run it on clusters. 
Still, more work to be continued....
