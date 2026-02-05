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

        for i in range(self.layer_num-1):
            weight_matrix = [[random.uniform(-0.5,0.5)for _ in range(self.layersize[i])] for _ in range(self.layersize[i+1])]
            self.weights.append(weight_matrix)
            bias_vector = [random.uniform(-0.1,0.1) for _ in range(self.layersize[i+1])]
            self.biases.append(bias_vector)
        # z = W_i*a + b, Where a is activation from previous layer,W_i is weight matrix, b is bias vector     

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def dsigmoid(self, x):
        return x * (1 - x)
    
    def forward(self,input):
        current_input = input
        self.activations = [input]
        for i in range(self.layer_num-1):
            next_activation = []
            for j in range(self.layersize[i+1]):
                activate = self.biases[i][j]
                for k in range(len(current_input)):
                    activate += current_input[k] * self.weights[i][j][k]
                next_activation.append(self.sigmoid(activate))
            self.activations.append(next_activation)
            current_input = next_activation
        return self.activations[-1]
    
    def backward(self,target):#  the core part of backpropagation
        output_errors = []
        output = self.activations[-1]
        for i in range(len(output)):
            error = target[i]-output[i]
            output_errors.append(error)
        
        Loss = 0
        for e in output_errors:
            Loss += e**2
        Loss = Loss/2

        for j in range(len(output)):
            gradient = output_errors[j] * self.dsigmoid(output[j])
            # update weights and biases for output layer
            for k in range(self.layersize[-2]):
                delta_w = self.learning_rate * gradient * self.activations[-2][k]
                self.weights[-1][j][k] += delta_w
            delta_b = self.learning_rate * gradient
            self.biases[-1][j] += delta_b       
        # propagate errors backward
        for l in range(self.layer_num-3,-1,-1):
            layer_errors = [] 
            for i in range(self.layersize[l+1]):# we compute the errors for layer l+1
                error = 0
                for j in range(self.layersize[l+2]):
                    error += output_errors[j] * self.weights[l+1][j][i]
                gradient = error * self.dsigmoid(self.activations[l+1][i]) 
                # the gradient for layer l is Sigma{errors^{L+1}}*dsigmoid(activation)
                layer_errors.append(gradient)
                for k in range(self.layersize[l]):
                    delta_w = self.learning_rate * gradient * self.activations[l][k]
                    self.weights[l][i][k] += delta_w
                delta_b = self.learning_rate * gradient
                self.biases[l][i] += delta_b
            # update weights and biases for layer l+1
            output_errors = layer_errors
        return Loss
    
    def train(self, input, target):
        self.forward(input)
        loss = self.backward(target)
        return loss
    def reasoning(self, input):
        output = self.forward(input)
        return output