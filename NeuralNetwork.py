import numpy as np
from sigmoid import *
from sigmoid_derivative import *

class NeuralNetwork():
    def __init__(self,
                 n_in,
                 n_hidden,
                 n_out):
        """
        :param n_in     : number of elements of each input sample
        :param n_hidden : number of hidden states
        :param n_out    : number of output states
        """
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        
        # Initialize weight matrices of the hidden layer and the output layer.
        # Both layers will have a bias term, hence we need to add one more weight.
        self.w_hidden = np.random.rand(n_in + 1, n_hidden)
        self.w_out = np.random.rand(n_hidden + 1, n_out)
        
        # initialize the states of the hidden neurons and the output neurons
        self.y_hidden = np.zeros((n_hidden))
        self.y_out = np.zeros((n_out))
    
    def reset_activations(self):
        # reset neuron activations
        self.y_hidden.fill(0)
        self.y_out.fill(0)
    
    def forward_pass(self, inputs):
        # We will calculate the output(s), by feeding the inputs forward through the network
        
        # If a forward pass has occured before (i.e., bias term has been appended to y_hidden), then we have to remove the bias from the hidden neurons
        if len(self.y_hidden)==(self.n_hidden+1):
            self.y_hidden = self.y_hidden[1:]
        
        # set hidden states and output states to zero
        self.reset_activations()
        
        # append term to be multiplied with the hidden layer's bias
        inputs = np.append(1, inputs)
        
        # activate hidden neurons
        for i in range(self.n_hidden):
            hidden_neuron = 0.0
            for j in range(len(inputs)):
                hidden_neuron += + inputs[j] * self.w_hidden[j,i]
            self.y_hidden[i] = sigmoid(hidden_neuron)
        
        # append term to be multiplied with the output layer's bias
        self.y_hidden = np.append(1.0, self.y_hidden)
        
        # activate output neurons
        for i in range(self.n_out):
            output_neuron = 0.0
            for j in range(len(self.y_hidden)):
                output_neuron += self.y_hidden[j] * self.w_out[j,i]
            self.y_out[i] = sigmoid(output_neuron)
        
        predictions = self.y_out.copy()
        
        return predictions
    
    def backward_pass(self, inputs, targets, learning_rate):
        # We will backpropagate the error and perform gradient descent on the network weights
        
        # We compute the error between predictions and targets
        J = 0.5 * np.sum( np.power(self.y_out - targets, 2) )
        
        # append term that was multiplied with the hidden layer's bias
        inputs = np.append(1, inputs)
        
        # Step 1. Output deltas are used to update the weights of the output layer
        output_deltas = np.zeros((self.n_out))
        outputs = self.y_out.copy()

        for i in range(self.n_out):
            #########################################
            # Write your code here
            # compute output_deltas : delta_k = (y_k - t_k) * g'(x_k)
            if type(targets) is np.ndarray:
                output_deltas[i]=(outputs[i]-targets[i])*sigmoid_derivative(outputs[i])
            else:
                output_deltas[i] = (outputs[i] - targets) * sigmoid_derivative(outputs[i])
            
            # output_deltas[i] = ...
            ########################################/
        
        # Step 2. Hidden deltas are used to update the weights of the hidden layer
        hidden_deltas = np.zeros((len(self.y_hidden)))
        
        # Create a for loop, to iterate over the hidden neurons.
        # Then, for each hidden neuron, create another for loop, to iterate over the output neurons
        for i in range(len(hidden_deltas)):
            #########################################
            # Write your code here
            # compute hidden_deltas
            
            #...
            #...
            #...
            sigma=0
            for k in range(len(output_deltas)):
                sigma=sigma+self.w_out[i,k]*output_deltas[k]
            g=sigmoid_derivative(self.y_hidden[i])
            hidden_deltas[i] = g*sigma
            
            ########################################/

        # Step 3. update the weights of the output layer
        for i in range(len(self.y_hidden)):
            for j in range(len(output_deltas)):
                #########################################
                # Write your code here
                # update the weights of the output layer
                self.w_out[i, j] = self.w_out[i, j] - learning_rate * output_deltas[j] * self.y_hidden[i]
                # self.w_out[i,j] = ...
                ########################################/
        
        # we will remove the bias that was appended to the hidden neurons, as there is no
        # connection to it from the hidden layer
        # hence, we also have to keep only the corresponding deltas
        hidden_deltas = hidden_deltas[1:]
        
        # Step 4. update the weights of the hidden layer
        # Create a for loop, to iterate over the inputs.
        # Then, for each input, create another for loop, to iterate over the hidden deltas
        for i in range(len(inputs)):
            for j in range(len(hidden_deltas)):
                #########################################
                # Write your code here
                # update the weights of the hidden layer
                try:
                    self.w_hidden[i,j] = self.w_hidden[i,j] - learning_rate * hidden_deltas[j] * inputs[j]
                except IndexError:
                    self.w_hidden[i,4] = self.w_hidden[i,4] - learning_rate * hidden_deltas[4] * inputs[4]
                # self.w_hidden[i,j] = ...
                ########################################/
        
        return J
