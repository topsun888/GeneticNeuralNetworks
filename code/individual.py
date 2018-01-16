import pickle
import os
import numpy as np
class feedforwardnetwork:
    '''
    Args:
    input_data: input matrix,[batch_size X feature_num], array
    input_labels: one-hot labels, array
    weights_qubit: define the weights, qubit encoding, list
    biases_qubit: define the biases, qubit_encoding, list
    '''
    def __init__(self, input_data, input_labels, weights_qubit, biases_qubit):
        self.input_data = input_data
        self.input_labels = input_labels
        self.weights_qubit = weights_qubit
        self.biases_qubit = biases_qubit
        
    #Define sigmoid function
    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))
    
    #Calculate the accuracy
    def calAccuracy(self, y, y_test):
        '''Calculate Accuracy'''
        return np.mean(y==y_test)
    
    def collapse(self, alpha, beta):
        '''
        Collapse the quantum state into a binary value
        To calculate values later
        Args:
        alpha: the first state
        beta: the second state
        '''
        pick = np.random.uniform(0, 1, alpha.shape)
        #If the random value greater than alpha square, the return true
        #Note this can lead to uncertainties for the results
        states = np.where(pick > alpha**2, 1, 0)
        return states

    
    def binary2decimal(self, states, bound):
        '''
        Map binary values of a variable into a decimal value
        Each parameter can be mapped into binary bits
        Args:
        states: binary bits, like [0 1 1 0]
        bound: the border of the variable
        
        Return: variable values
        '''
        shape = states.shape
        #For weights
        if len(shape) > 2:
            qubit_num = shape[-1]
            values = np.zeros((shape[0], shape[1]))
            for l in np.arange(qubit_num):
                values += states[:, :, l] * (2**l)
                values = -bound + values/(2**qubit_num-1)*2*bound
        #For biases
        else:
            qubit_num = shape[-1]
            values = np.zeros((shape[0]))
            for l in np.arange(qubit_num):
                values += states[:, l] * (2**l)
                values = -bound + values/(2**qubit_num-1)*2*bound
        return values
    
    def quantum2value(self):
        '''
        Map varible qubits into decimal values
        '''
        self.weights = []
        self.weights_bits = []
        for weight_qubit in self.weights_qubit:
            states = self.collapse(weight_qubit['sin'], weight_qubit['cos'])
            values = self.binary2decimal(states, 0.5)
            self.weights.append(values)
            self.weights_bits.append(states)
        self.biases = []
        self.biases_bits = []
        for bias_qubit in self.biases_qubit:
            states = self.collapse(bias_qubit['sin'], bias_qubit['cos'])
            values = self.binary2decimal(states, 0.5)
            self.biases.append(values)
            self.biases_bits.append(states)
        #return self.weights, self.biases
        
    #Create a function to do predictions
    #Map a one-hot vector to a number
    def vec2num(self, data, label_num=10):
        '''Make predictions'''
        if len(data.shape) < 2:
            print('The input has too few dimensions')
            return None
        #Select the class which has largest probability
        #predictions = [(z.argmax()+ 1)%label_num for z in data]
        predictions = data.argmax(axis=1)
        return np.array(predictions)
    
    
    def costFuncWithReg(self, h, lambda1=0.01):
        '''
        Calculate the cost of neural network
        Note here we use cross entropy
        Regularization is also taken into account
        '''
        y = self.input_labels
        if h is None or y is None or self.weights is None:
            print('Invalid Input!')
            return None
        sample_num = len(y)#Length of y
        #Cost of errors
        total = -np.mean(np.sum(y*np.log(h), axis=1))
        #Cost of regularization
        weights = np.array(self.weights)
        reg = 0
        for wgt in weights:
            reg += np.sum(wgt**2) * lambda1/2/sample_num
        total +=  reg    
        return total 
    
    def proceed(self):
        '''
        Finish the procedure of feed forward network 
        and calculate the output
        '''
        self.quantum2value()
        h, output_h, input_z = self.feedforwardNeuralNetwork(self.input_data)
        #labels = self.vec2num(self.input_labels)
        labels = self.input_labels
        predictions = self.vec2num(h)
        #self.fitness = self.costFuncWithReg(h)
        self.accuracy = self.calAccuracy(labels, predictions)
        self.fitness = self.accuracy
        
    def updateWeightBias(self, new_weights_qubit, new_biases_qubit):
        '''
        Update weights and biases
        '''
        self.weights_qubit = new_weights_qubit
        self.biases_qubit = new_biases_qubit

    def savaParams(self, path='data.pickle'):
        '''
        Save current parameters such as weights and biases
        Args:
        path: the file path and name
        '''
        data = {}
        data['weights'] = self.weights_qubit
        data['biases'] = self.biases_qubit
        try:
            path = 'model/' + path
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print('Parameters Saved!')
        except OSError:
            print('Failing in I/O operations!')
        except ValueError:
            print('Failing in saving the values')
        


        
    def evaluateTestData(self, new_input_data, new_input_labels):
        '''
        Update data
        '''
        X, y = new_input_data, new_input_labels
        h, output_h, input_z = self.feedforwardNeuralNetwork(X)
        labels = y
        predictions = self.vec2num(h)
        accuracy = self.calAccuracy(labels, predictions)
        return accuracy
    
    #weights: weights for each layer, a list
    def feedforwardNeuralNetwork(self, X):
        '''Calculate feedforward propagation output'''
        ######Deal with extreme cases###
        if X is None or self.weights is None:
            print('Invalid Input!')
            return None
        dim = X.shape
        if len(dim) < 2:
            print('X has too less variables')
            return None

        #####Define variables###########
        layer_num = len(self.weights)
        output_h = []#Output for each layer
        output_h.append(X)#The first layer is equal to input X
        input_z = []#Input for each layer, starts from the second layer
        #####Make alculations for each layer, except the input layer
        for i in range(layer_num):
            z = np.dot(output_h[i], self.weights[i])
            z += self.biases[i]
            h = self.sigmoid(z)
            output_h.append(h)
            input_z.append(z)
        return h, output_h, input_z    