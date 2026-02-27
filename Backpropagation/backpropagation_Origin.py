import math
import random
import pickle

class NeuralNetwork:
    def __init__(self, layersize, learning_rate=0.15, use_relu=False):
        self.learning_rate = learning_rate
        self.layersize = layersize   # multilayer structure
        self.layer_num = len(layersize)
        self.use_relu = use_relu  # Use ReLU for hidden layers
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []  # Store pre-activation values for ReLU

        # Conservative initialization to avoid dying ReLU
        for i in range(self.layer_num-1):
            if use_relu:
                # Smaller initialization for ReLU to prevent dying neurons
                weight_matrix = [[random.uniform(-0.3,0.3) for _ in range(self.layersize[i])] 
                               for _ in range(self.layersize[i+1])]
                bias_vector = [0.01 for _ in range(self.layersize[i+1])]  # Small positive bias
            else:
                weight_matrix = [[random.uniform(-0.5,0.5) for _ in range(self.layersize[i])] 
                               for _ in range(self.layersize[i+1])]
                bias_vector = [random.uniform(-0.1,0.1) for _ in range(self.layersize[i+1])]
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
        # z = W_i*a + b, Where a is activation from previous layer,W_i is weight matrix, b is bias vector     

    def sigmoid(self, x):
        # Numerical stability
        if x < -500:
            return 0.0
        if x > 500:
            return 1.0
        return 1 / (1 + math.exp(-x))
    
    def dsigmoid(self, a):
        # Derivative of sigmoid (input is activation)
        return a * (1 - a)
    
    def relu(self, x):
        return max(0.0, x)
    
    def drelu(self, z):
        # Derivative of ReLU (input is pre-activation z)
        return 1.0 if z > 0 else 0.0
    
    def forward(self, input):
        current_input = input
        self.activations = [input]
        self.z_values = []
        
        for i in range(self.layer_num-1):
            next_activation = []
            layer_z = []
            
            for j in range(self.layersize[i+1]):
                # Compute z = W*x + b
                z = self.biases[i][j]
                for k in range(len(current_input)):
                    z += current_input[k] * self.weights[i][j][k]
                
                layer_z.append(z)
                
                # Apply activation: ReLU for hidden, Sigmoid for output
                if self.use_relu and i < self.layer_num - 2:  # Hidden layer
                    activation = self.relu(z)
                else:  # Output layer or non-ReLU mode
                    activation = self.sigmoid(z)
                
                next_activation.append(activation)
            
            self.z_values.append(layer_z)
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

        # compute gradients (deltas) for output layer and update its weights/biases
        output_deltas = []
        for j in range(len(output)):
            gradient = output_errors[j] * self.dsigmoid(output[j])
            output_deltas.append(gradient)
            # update weights and biases for output layer
            for k in range(self.layersize[-2]):
                delta_w = self.learning_rate * gradient * self.activations[-2][k]
                self.weights[-1][j][k] += delta_w
            delta_b = self.learning_rate * gradient
            self.biases[-1][j] += delta_b

        # use output layer deltas for backpropagation to previous layers
        output_errors = output_deltas

        # propagate errors backward for remaining hidden layers
        for l in range(self.layer_num-3,-1,-1):
            layer_errors = [] 
            for i in range(self.layersize[l+1]):# we compute the errors for layer l+1
                error = 0
                for j in range(self.layersize[l+2]):
                    error += output_errors[j] * self.weights[l+1][j][i]
                gradient = error * self.dsigmoid(self.activations[l+1][i]) 
                # the gradient for layer L is Sigma{errors^{L+1}}*dsigmoid(activation)
                layer_errors.append(gradient)
                for k in range(len(self.activations[l])):
                    delta_w = self.learning_rate * gradient * self.activations[l][k]
                    self.weights[l][i][k] += delta_w
                delta_b = self.learning_rate * gradient
                self.biases[l][i] += delta_b
            # update weights and biases for layer l+1
            output_errors = layer_errors
            # print(layer_errors, "\n")
        return Loss

    def train(self, input, target):
        self.forward(input)
        loss = self.backward(target)
        return loss
    
    def backward_cross_entropy(self, target):
        """
        Backpropagation using Binary Cross Entropy (BCE) loss with sigmoid output.
        For sigmoid + BCE, output layer delta is (target - output).
        """
        output = self.activations[-1]

        # numerical stability for log
        eps = 1e-12

        # BCE loss
        loss = 0
        for i in range(len(output)):
            a = min(max(output[i], eps), 1 - eps)
            y = target[i]
            loss += -(y * math.log(a) + (1 - y) * math.log(1 - a))
        loss = loss / len(output)

        # output layer delta: (y - a)
        output_deltas = []
        for j in range(len(output)):
            delta = target[j] - output[j]
            output_deltas.append(delta)

            # update weights and biases for output layer
            for k in range(self.layersize[-2]):
                delta_w = self.learning_rate * delta * self.activations[-2][k]
                self.weights[-1][j][k] += delta_w
            delta_b = self.learning_rate * delta
            self.biases[-1][j] += delta_b

        # propagate deltas backward for hidden layers
        output_errors = output_deltas
        for l in range(self.layer_num-3, -1, -1):
            layer_errors = []
            for i in range(self.layersize[l+1]):
                error = 0
                for j in range(self.layersize[l+2]):
                    error += output_errors[j] * self.weights[l+1][j][i]
                
                # Apply correct derivative
                if self.use_relu:
                    gradient = error * self.drelu(self.z_values[l][i])
                else:
                    gradient = error * self.dsigmoid(self.activations[l+1][i])
                
                # Gradient clipping to prevent NaN
                gradient = max(min(gradient, 10.0), -10.0)
                
                layer_errors.append(gradient)
                for k in range(len(self.activations[l])):
                    delta_w = self.learning_rate * gradient * self.activations[l][k]
                    self.weights[l][i][k] += delta_w
                delta_b = self.learning_rate * gradient
                self.biases[l][i] += delta_b

            output_errors = layer_errors

        return loss
    

    def train_cross_entropy(self, input, target):
        self.forward(input)
        loss = self.backward_cross_entropy(target)
        return loss

    def reasoning(self, input):
        output = self.forward(input)
        return output