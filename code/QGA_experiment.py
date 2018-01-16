import numpy as np
import math
import os
from time import time
from sklearn.model_selection import train_test_split
from individual import feedforwardnetwork
from sklearn import datasets
from population import initPopulation
from quantumGA import QGA
import tensorflow.examples.tutorials.mnist.input_data as input_data
qubit_len = 10
def initializeWeights(layers):
    if type(layers) is not list:
        print('Wrong input!')
        return None
    layers_num = len(layers)
    weights = []
    for l in range(layers_num-1):
        #Generate angles
        weight = {}
        #Each weight varibale is encoded with 4 qubits
        #With the 4 qubits, we can map them into a value
        degrees = 0.25 * math.pi * np.ones((layers[l], layers[l+1], qubit_len))
        weight['sin'] = np.sin(degrees)
        weight['cos'] = np.cos(degrees)
        weight['degree'] = degrees
        weights.append(weight)
    return weights

def initializeBiases(layers):
    if type(layers) is not list:
        print('Wrong input!')
        return None
    layers_num = len(layers)
    biases = []
    for l in range(layers_num-1):
        bias = {}
        #Each bias is encoded in 4 qubits
        #With the 4 qubits, we can map them into a value
        degrees = 0.25 * math.pi * np.ones((layers[l+1], qubit_len))
        bias['sin'] = np.sin(degrees)
        bias['cos'] = np.cos(degrees)
        bias['degrees'] = degrees
        biases.append(bias)
    return biases
        
if __name__ == '__main__':
    #Get data set
    #iris = datasets.load_iris()
    #X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=1)
    print(os.getcwd())
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=False)
    
    X_train, y_train, X_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    layers = [784, 10]
    #layers = [4, 20, 3]
    #Test the individual object
    weights_qubit = initializeWeights(layers)
    biases_qubit = initializeBiases(layers)
    fw = feedforwardnetwork(X_train, y_train, weights_qubit, biases_qubit)
    start = time()
    fw.proceed()
    end = time()
    print(end - start)
    print(fw.fitness)
    #fw.savaParams()
    #Test the population
    pop = initPopulation(layers)
    population = pop.generateBatchPop(X_train, y_train, params_path='model/mnist.pickle')
    #Test Quantum Genetic Algorithm
    qga = QGA(population)
    #Try 20 generations
    start = time()
    best = qga.proceed(300)
    end = time()
    print('Training Time:', round(end - start, 4))
    accuracy = best.evaluateTestData(X_test, y_test)
    print('Testing Accuracy:', round(accuracy, 4))
    best.savaParams(path='mnist.pickle')
