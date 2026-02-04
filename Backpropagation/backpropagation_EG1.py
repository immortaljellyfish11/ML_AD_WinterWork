import math
import random

class NeuralNetwork:
    def __init__(self, layersize, learning_rate=0.15):
        self.learning_rate = learning_rate
        self.layersize = layersize   # multilayer structure
        self.layer_num = len(layersize)
        self.weights = []
        self.biases = []
        self.activations = []

        for i in range(self.layer_num - 1):
            # initialize weights randomly between -1 and 1
            weight_matrix = [[random.uniform(-1, 1) for _ in range(self.layersize[i + 1])] 
                           for _ in range(self.layersize[i])]# weight from layer L to layer L+1
            self.weights.append(weight_matrix)
            bias_vector = [random.uniform(-0.15, 0.15) for _ in range(self.layersize[i + 1])]
            self.biases.append(bias_vector)
        # z = W_i*a + b, Where a is activation from previous layer,W_i is weight matrix,b is bias vector

    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, input):
        current_input = input
        self.activations = [input]
        for l in range(self.layer_num - 1):     
            next_activation = []
            for j in range(self.layersize[l + 1]):
                active = self.biases[l][j] 
                for i in range(self.layersize[l]):# matrix multiplication
                    active += current_input[i] * self.weights[l][i][j]  
                next_activation.append(self.sigmoid(active))
            current_input = next_activation
            self.activations.append(next_activation)
        return self.activations[-1]# output layer
    
    def backward(self, target):  # the core part of this algorithm 
        output_errors = []
        final_output = self.activations[-1]
        for i in range(len(final_output)):
            error = target[i]- final_output[i]
            output_errors.append(error)
        
        loss = 0
        for err in output_errors:
            loss += err ** 2
        loss = loss / 2

        output_gradients = []
        for i in range(len(final_output)):# use loss function :(0.5(t-y)^2) 
            gradient = output_errors[i] * self.sigmoid_derivative(final_output[i])
            output_gradients.append(gradient)
        
        for l in range(self.layer_num - 2, -1, -1):# backpropagation value
            current_activations = self.activations[l]
            for i in range(self.layersize[l]):  # refresh the weights
                for j in range(self.layersize[l + 1]):
                    self.weights[l][i][j] += self.learning_rate * output_gradients[j] * current_activations[i] 
            
            for j in range(self.layersize[l + 1]):# refresh the biases
                self.biases[l][j] += self.learning_rate * output_gradients[j]
            
            if l > 0:
                hidden_errors = [0] * self.layersize[l]# calculate the hidden layer errors
                for i in range(self.layersize[l]):
                    error = 0
                    for j in range(self.layersize[l + 1]):
                        error += output_gradients[j] * self.weights[l][i][j]
                    hidden_errors[i] = error
                
                output_gradients = []
                for i in range(self.layersize[l]):
                    gradient = hidden_errors[i] * self.sigmoid_derivative(self.activations[l][i])
                    output_gradients.append(gradient)
        return loss
    
    def train(self, input, target):
        self.forward(input)
        loss = self.backward(target)
        return loss

    def reasoning(self, input):
        return self.forward(input)
    


