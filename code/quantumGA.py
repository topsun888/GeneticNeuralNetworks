import copy
import math
from multiprocessing import Pool  
import numpy as np

class QGA:
    def __init__(self, population):
        '''
        Initialize genetic algorithm
        '''
        self.population = copy.deepcopy(population)
        self.pop_size = len(population)
        
    
    def rotationMatrix(self, sgn, delta):
        '''
        Calculation the matrix of rotation gate
        Args:
        sgn: sign of the angle rotation direction, +1 or -1
        delta: shift angle of the rotation gate
        '''
        e = sgn *delta
        U = np.array([[np.cos(e), -np.sin(e)], [np.sin(e), np.cos(e)]])
        return U
    
    def rotationAngleDirection(self, params):
        '''
        Calculate rotation angles and directions for each individual
        Args:
        params: tuple, (bestIndividual, currentIndividual)
        '''
        bestIndividual = params[0]
        obj = params[1]
        #Initialize the shift angle
        delta_theta = 0.01 * math.pi
        #Compare the fitness
        fitness_flag = obj.fitness > bestIndividual.fitness
        #Traverse each weight layer
        for j in np.arange(len(obj.weights_qubit)):
            #The jth layer
            #Traverse each parameter in this layer
            #qubit contains alpha and beta
            #Alpha size: layer(i), layer(i+1), bit length
            #A qubit is a pair of alpha and beta
            weight_alpha = obj.weights_qubit[j]['sin']
            #beta size: layer(i), layer(i+1), bit length
            weight_beta = obj.weights_qubit[j]['cos']
            #Original angles:
            degrees = obj.weights_qubit[j]['degree']
            #state size: layer(i), layer(i+1), bit length
            #state, e.g [0 1 1 0]
            weight_bit = obj.weights_bits[j]
            #Traverse each parameter
            best_weight_bit = bestIndividual.weights_bits[j]
            #Search the table
            criteria = (weight_bit + best_weight_bit) == 1
            #Calculate shift angles for each parameter
            delta = criteria * delta_theta
            #Calculate the sign of shift angles, use matrix as much as possible
            sgns = np.zeros(weight_bit.shape)#Initialize it with zeros
            #Try to avoid loops, use matrix operation as much as mossible
            #IF xi=0 besti=1, then the difference will be -1
            current_best_bit_flag = weight_bit - best_weight_bit
            #Create a matrix of fitness flag with the same shape as the weight
            fitness_flags = np.ones(weight_bit.shape) * fitness_flag
            #Map 0 into -1
            fitness_flags = np.where(fitness_flags>0, 1, -1)
            alpha_beta_pos = (weight_alpha * weight_beta) > 0
            alpha_beta_neg = (weight_alpha * weight_beta) < 0
            alpha_zero = weight_alpha == 0
            beta_zero = weight_beta == 0
            #if alpha * beta>0
            sgns += current_best_bit_flag * fitness_flags * alpha_beta_pos
            #if alpha * beta<0
            sgns += (-1)*current_best_bit_flag * fitness_flags * alpha_beta_neg
            #if alpha = 0
            #Gnerate +1 -1 at random
            direction = np.random.choice([1, -1], size=weight_bit.shape)
            criteria = current_best_bit_flag * fitness_flags * alpha_zero < 0
            sgns += criteria * direction
            #if beta = 0
            criteria = current_best_bit_flag * fitness_flags * beta_zero > 0
            sgns += criteria * direction
            #Calculate shift angles
            angles = delta * sgns
            #Calculate new angles
            degrees = degrees - angles
            obj.weights_qubit[j]['sin'] = np.sin(degrees)
            obj.weights_qubit[j]['cos'] = np.cos(degrees)
            obj.weights_qubit[j]['degree'] = degrees
        #Update bias
        for j in np.arange(len(obj.biases_qubit)):
            #The jth bias
            #Traverse each parameter in this layer
            #qubit contains alpha and beta
            #Alpha size: layer(i) bit length
            #A qubit is a pair of alpha and beta
            bias_alpha = obj.biases_qubit[j]['sin']
            #beta size: layer(i), layer(i+1), bit length
            bias_beta = obj.biases_qubit[j]['cos']
            #Original angles:
            degrees = obj.biases_qubit[j]['degree']
            #state size: layer(i), layer(i+1), bit length
            #state, e.g [0 1 1 0]
            bias_bit = obj.biases_bits[j]
            #Traverse each parameter
            best_bias_bit = bestIndividual.biases_bits[j]
            #Search the table
            criteria = (bias_bit + best_bias_bit) == 1
            #Calculate shift angles for each parameter
            delta = criteria * delta_theta
            #Calculate the sign of shift angles
            #Shape: bias number * bit length
            sgns = np.zeros(bias_bit.shape)#Initialize it with zeros
            #Try to avoid loops, use matrix operation as much as mossible
            #IF xi=0 besti=1, then the difference will be -1
            current_best_bit_flag = bias_bit - best_bias_bit
            #Create a matrix of fitness flag with the same shape as the weight
            fitness_flags = np.ones(bias_bit.shape) * fitness_flag
            #Map 0 into -1
            fitness_flags = np.where(fitness_flags>0, 1, -1)
            alpha_beta_pos = (bias_alpha * bias_beta) > 0
            alpha_beta_neg = (bias_alpha * bias_beta) < 0
            alpha_zero = bias_alpha == 0
            beta_zero = bias_beta == 0
            #if alpha * beta>0
            sgns += current_best_bit_flag * fitness_flags * alpha_beta_pos
            #if alpha * beta<0
            sgns += (-1)*current_best_bit_flag * fitness_flags * alpha_beta_neg
            #if alpha = 0
            #Gnerate +1 -1 at random
            direction = np.random.choice([1, -1], size=bias_bit.shape)
            criteria = current_best_bit_flag * fitness_flags * alpha_zero < 0
            sgns += criteria * direction
            #if beta = 0
            criteria = current_best_bit_flag * fitness_flags * beta_zero > 0
            sgns += criteria * direction
            #Calculate shift angles
            angles = delta * sgns
            #Calculate new angles
            degrees = degrees - angles
            obj.biases_qubit[j]['sin'] = np.sin(degrees)
            obj.biases_qubit[j]['cos'] = np.cos(degrees)
            obj.biases_qubit[j]['degree'] = degrees
        obj.proceed()
        return obj
    
    def rotatingGates(self, bestIndividual_index):
        '''
        Rotate gates of quantum registra,
        Note, we try to make use of numpy's matrix operations to speed
        Computation
        Args:
        bestIndividual_index: the index of the best individual
        '''
        bestIndividual = self.population[bestIndividual_index]
        bestIndividuals = [bestIndividual] * self.pop_size
        params = zip(bestIndividuals, self.population)
        pool = Pool()
        self.population = pool.map(self.rotationAngleDirection, params)
        pool.close()
        pool.join()
        #Traverse each individual
        #for i in np.arange(self.pop_size):
            #obj = self.population[i]
            #obj = self.rotationAngleDirection(bestIndividual, obj)
            #self.population[i] = obj
            
    def NotGates(self, ratio=0.1):
        '''
        Rotate gates of quantum registra,
        Note, we try to make use of numpy's matrix operations to speed
        Computation
        Args:
        bestIndividual_index: the index of the best individual
        '''
        #Traverse each individual
        num = int(self.pop_size * ratio)
        #pops = np.random.choice(self.population, num)
        #Note the individuals of the population are refered
        #And the parameters point to the address of individuals
        #Once the individuals which are used as parameter change
        #The original population will change as well
        #pool = Pool()
        #_ = pool.map(self.mutation, pops)
        #pool.close()
        #pool.join()
        indice = np.random.choice(self.pop_size, num)
        for i in indice:
            obj = self.population[i]
            obj = self.mutation(obj)
            self.population[i] = obj
    
    def mutation(self, obj):
        '''
        Mutation at several random point within an individual
        '''
        #Traverse each weight layer
        for j in np.arange(len(obj.weights_qubit)):
            #The jth layer
            #Traverse each parameter in this layer
            #qubit contains alpha and beta
            #Alpha size: layer(i), layer(i+1), bit length
            #A qubit is a pair of alpha and beta
            weight_alpha = obj.weights_qubit[j]['sin']
            #beta size: layer(i), layer(i+1), bit length
            weight_beta = obj.weights_qubit[j]['cos']
            #Degrees
            degrees = obj.weights_qubit[j]['degree']
            picks = np.random.uniform(0, 1, size=degrees.shape)
            #state size: layer(i), layer(i+1), bit length
            alpha_flag = weight_alpha < picks
            beta_flag = weight_beta < picks
            degrees = degrees - alpha_flag*beta_flag*math.pi/2
            obj.weights_qubit[j]['sin'] = np.sin(degrees)
            obj.weights_qubit[j]['cos'] = np.cos(degrees)
            obj.weights_qubit[j]['degree'] = degrees
        #Update bias
        for j in np.arange(len(obj.biases_qubit)):
            #The jth bias
            #Traverse each parameter in this layer
            #qubit contains alpha and beta
            #Alpha size: layer(i) bit length
            #A qubit is a pair of alpha and beta
            bias_alpha = obj.biases_qubit[j]['sin']
            #beta size: layer(i), layer(i+1), bit length
            bias_beta = obj.biases_qubit[j]['cos']
            #Original angles:
            degrees = obj.biases_qubit[j]['degree']
            picks = np.random.uniform(0, 1, size=degrees.shape)
            alpha_flag = bias_alpha < picks
            beta_flag = bias_beta < picks
            degrees = degrees - alpha_flag*beta_flag*math.pi/2
            #Calculate new angles
            degrees = degrees - alpha_flag*beta_flag*math.pi/2
            obj.biases_qubit[j]['sin'] = np.sin(degrees)
            obj.biases_qubit[j]['cos'] = np.cos(degrees)
            obj.biases_qubit[j]['degree'] = degrees
        obj.proceed()
        return obj
        
    def proceed(self, generation_num = 1):
        '''
        Execute quantum rotation and not gating
        '''
        #Keep the best individual
        best_fitness, optimal_index, _ = self.findMaximalIndividual()
        #Note the variables are referenced, we need to copy in a deep way
        best_individual = copy.deepcopy(self.population[optimal_index])
        for i in np.arange(generation_num):
            #Quantum Rotation Gate
            if i % 5 == 0:
                print('Best Training Accuracy:', round(best_fitness, 4))
            #test_accuracy = best_individual.evaluateTestData(X_test, y_test)
            #print('Testing Accuracy:', round(test_accuracy, 4))
            self.rotatingGates(optimal_index)
            #Mutation
            self.NotGates()
            fitness, max_index, min_index = self.findMaximalIndividual()
            if fitness < best_fitness:
                #Use deep copy or the two objects will be the same actually
                #Keep the original best individual
                self.population[min_index] = copy.deepcopy(best_individual)
                optimal_index = min_index
            else:
                best_fitness = fitness
                optimal_index = max_index
                #Note deep copy here
                best_individual = copy.deepcopy(self.population[optimal_index])
        print('Final Training Accuracy:', round(best_fitness, 4))
        return best_individual
        
    def findMaximalIndividual(self):
        fitnesses = np.array([one.fitness for one in self.population])
        optimal_value = max(fitnesses)
        optimal_index = fitnesses.argmax()
        minimal_index = fitnesses.argmin()
        return optimal_value, optimal_index, minimal_index