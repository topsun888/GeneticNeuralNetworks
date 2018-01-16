import math
import pickle
import os
import numpy as np
from individual import feedforwardnetwork
class initPopulation:
    def __init__(self, layers, pop_size=200, qubit_len=8, X_train=None, y_train=None):
        '''
        Args:
        pop_size: number of population, integer
        layers: neuron number for each layer, list
        '''
        self.pop_size = pop_size
        self.layers = layers
        self.qubit_len = qubit_len
        self.x_train = X_train
        self.y_train = y_train
        
    def __loadWeightBias__(self, path):
        '''
        Load parameters such as weights and biases from local disk
        Args:
        path: the file path of the saved parameters
        the file is a dictionary containing both weights and biases
        '''
        if not os.path.exists(path):
            print(path + 'Not Found')
            return None, None
        pkl_file = open(path, 'rb')
        data = pickle.load(pkl_file)
        #Qubit biases and weights
        weights = data['weights']
        biases =data['biases']
        pkl_file.close()
        return weights, biases



    def __initializeWeights__(self):
        '''
        Initialize weights between each two neigbouring layers
        '''
        layers = self.layers
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
            #degrees = 0.25 * np.pi * np.ones((layers[l], layers[l+1], qubit_len))
            degrees = np.ones((layers[l], layers[l+1], self.qubit_len)) * math.pi/4
            weight['sin'] = np.sin(degrees)
            weight['cos'] = np.cos(degrees)
            weight['degree'] = degrees
            weights.append(weight)
        return weights

    def __initializeBiases__(self):
        '''
        Initialize biases for each layer(except for input layer)
        '''
        layers = self.layers
        if type(layers) is not list:
            print('Wrong input!')
            return None
        layers_num = len(layers)
        biases = []
        for l in range(layers_num-1):
            bias = {}
            #Each bias is encoded in 4 qubits
            #With the 4 qubits, we can map them into a value
            #degrees = 0.25 * np.pi * np.ones((layers[l], layers[l+1], qubit_len))
            degrees = np.ones((layers[l+1], self.qubit_len)) * math.pi/4
            bias['sin'] = np.sin(degrees)
            bias['cos'] = np.cos(degrees)
            bias['degree'] = degrees
            biases.append(bias)
        return biases
    
    def generatePop(self, X_train, y_train):
        '''
        Generate a group of weights at random
        '''
        population = []
        for i in range(self.pop_size):
            weights = self.__initializeWeights__()
            biases = self.__initializeBiases__()
            #Initialize an individual
            individual = feedforwardnetwork(X_train, y_train, weights, biases)
            #Calculate fitness
            individual.proceed()
            population.append(individual)
        return population

    def generateBatchPop(self, X_train, y_train, batch_size=1024, params_path=None):
        '''
        Generate a group of weights at random
        Args:
        X_train: input feature matrix, numpy, [rows, columns]
        y_train: input label, numpy
        batch_size: number of input data
        params_path: path of parameter files
        '''
        #Load parameters
        if params_path is not None and os.path.exists(params_path):
            print('Load parameters...')
            weights, biases = self.__loadWeightBias__(params_path)
        else:
            print('Initialize pparameters...')
            weights = self.__initializeWeights__()
            biases = self.__initializeBiases__()
        population = []
        for i in range(self.pop_size):
            #Initialize an individual
            if len(y_train) <= batch_size:#if the sample is too small
                batch_train, batch_label = X_train, y_train
            else:
                selected = np.random.choice(len(y_train), batch_size)
                batch_train, batch_label = X_train[selected], y_train[selected]
            individual = feedforwardnetwork(batch_train, batch_label, weights, biases)
            #Calculate fitness
            individual.proceed()
            population.append(individual)
        print('Population')
        return population