import pandas as pd
import numpy as np

from math import e
from copy import copy
from random import random, shuffle, seed as random_seed


def load_data(path):
    return pd.read_excel(path, sheet_name='measurement', usecols=[4,5,6,7])


class MultilayerPerceptron:
    # Activation functions
    __sigmoid = lambda x: 1 / (1 + e**(-x))
    __tanh = lambda x: 2 / (1 + e**(-2*x)) -1
    __relu = lambda x: max(0, x)

    # Derivatives of activation functions
    __sigmoid_derivative = lambda x: MultilayerPerceptron.__sigmoid(x) * (1 - MultilayerPerceptron.__sigmoid(x))
    __tanh_derivative = lambda x: 1 - MultilayerPerceptron.__tanh(x)**2
    __relu_derivative = lambda x: int(x >= 0)


    random_seed(1)


    def __init__(self, inputs_number, outputs_number, *args, learning_factor=.1):
        """
        Instantiates an MLP - a Multilayer Perceptron.
        @requires inputs_number - number of input data pieces
        @requires outputs_number - number of output neurons
        @requires args - set of layer sizes one by one
        """

        self.weights = list()
        self.biases = list()
        self.learning_factor = learning_factor
        self.activation_function = MultilayerPerceptron.__sigmoid
        self.activation_function_derivative = MultilayerPerceptron.__sigmoid_derivative
        previous_layer_neurons = inputs_number

        for arg in args:
            self.weights.append(
                np.array([1 - 2*random() for x in range(arg * previous_layer_neurons)])
                    .reshape(arg, previous_layer_neurons))
            self.biases.append(np.ones(arg))
            previous_layer_neurons = arg

        self.weights.append(
            np.array([1 - 2*random() for x in range(outputs_number * previous_layer_neurons)])
                .reshape(outputs_number, previous_layer_neurons))
        self.biases.append(np.ones(outputs_number))


    def feed_forward(self, inputs):
        layer_input = inputs

        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            sigma = weights @ layer_input + biases
            output = np.array( list( map(self.activation_function, sigma)))
            self.pre_squashing.append(sigma)
            self.outputs.append(output)
            layer_input = output

        output = self.weights[-1] @ layer_input + self.biases[-1]
        return output


    def get_network_cost(self, output, reference):
        if len(output) != len(reference):
            raise Error('Network output should be the same length as reference')

        cost = 0
        for i in range(len(output)):
            cost += (reference[i] - output[i])**2

        return cost


    def backpropagate(self, input, output, reference):
        self.layer_errors = []
        self.layer_errors.append((output - reference) * self.activation_function_derivative(output))
        # self.layer_errors.append(output - reference)

        # calculating node errors
        for sigma, activation, i in zip(self.pre_squashing[::-1], self.outputs[::-1], reversed(range(len(self.outputs)+1))):
            if self.activation_function == MultilayerPerceptron.__sigmoid:
                derivative_values = activation * (1 - activation)
            elif self.activation_function == MultilayerPerceptron.__tanh:
                derivative_values = 1 - activation**2
            else:
                derivative_values = self.activation_function_derivative(sigma)

            self.layer_errors.append(self.weights[i].T @ self.layer_errors[-1] * derivative_values)
        self.layer_errors.reverse()

        # calculating weights changes
        for i in range(len(self.weights)):
            if i == 0:
                input_array = input
            else:
                input_array = self.outputs[i-1]

            self.weights[i] += np.matrix(self.layer_errors[i]).T @ np.matrix(input_array * self.learning_factor)
            self.biases[i] += self.layer_errors[i] * self.learning_factor


    def train(self, epochs):
        self.pre_squashing = []
        self.data = self.data / 1000;

        for record in enumerate(self.data):
            self.outputs = []
            measurement = np.array([ record[1][0], record[1][1] ])
            reference = np.array([ record[1][2], record[1][3] ])

            output = self.feed_forward(measurement)
            self.backpropagate(measurement, output, reference)


    def get_data(self, path):
        df = load_data(path)
        self.data = pd.DataFrame.to_numpy(df)


# debug
if __name__ == '__main__':
    # path_template = 'data/pozyxAPI_only_localization_measurement{}.xlsx'
    # path_template = path_template.format(1)

    tpath = '/home/pawel/Pulpit/SISE/Localization_NN/data/pozyxAPI_only_localization_measurement1.xlsx'

    mlp = MultilayerPerceptron(2, 2, 6, 5)
    mlp.get_data(tpath)
    mlp.train(1000)

    df = load_data(tpath)
