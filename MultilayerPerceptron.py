import pandas as pd
import numpy as np

from Functions import *
from math import e
from copy import copy
from random import random, shuffle, seed as random_seed

class MultilayerPerceptron:
    # Activation functions
    __sigmoid = lambda x: 1 / (1 + e**(-x))
    __tanh = lambda x: 2 / (1 + e**(-2*x)) -1
    __relu = lambda x: max(0, x)

    # Derivatives of activation functions
    __sigmoid_derivative = lambda x: MultilayerPerceptron.__sigmoid(x) * (1 - MultilayerPerceptron.__sigmoid(x))
    __tanh_derivative = lambda x: 1 - MultilayerPerceptron.__tanh(x)**2
    __relu_derivative = lambda x: int(x >= 0)


    def __init__(self, inputs_number, outputs_number, *args, learning_factor=.01, weights=None, biases=None):
        """
        Instantiates an MLP - a Multilayer Perceptron.
        @requires inputs_number - number of input data pieces
        @requires outputs_number - number of output neurons
        @requires args - set of layer sizes one by one
        """
        '6.5.2_factor=0.1_take-0.npy'

        self.filename = ''
        self.weights = list()
        self.biases = list()
        self.learning_factor = learning_factor
        self.activation_function = MultilayerPerceptron.__tanh
        self.activation_function_derivative = MultilayerPerceptron.__tanh_derivative

        if weights is not None:
            self.weights = weights
            self.biases = biases
        else:
            previous_layer_neurons = inputs_number
            for arg in args:
                self.weights.append(np.random.uniform(-1, 1, (arg, previous_layer_neurons)))
                self.biases.append(np.ones(arg).reshape(arg,1))
                previous_layer_neurons = arg
                self.filename += str(arg) + '.'
            self.filename += str(outputs_number) + '_factor={}_'.format(learning_factor)

            self.weights.append(
                np.array([1 - 2*random() for x in range(outputs_number * previous_layer_neurons)])
                    .reshape(outputs_number, previous_layer_neurons))
            self.biases.append(np.ones(outputs_number).reshape(outputs_number,1))

        self.test_data = pd.DataFrame.to_numpy(pd.read_excel('data/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx', usecols=[4,5,6,7]).dropna()) / 1000

    def feed_forward(self, inputs):
        layer_input = inputs.reshape(len(inputs),1)

        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            sigma = weights @ layer_input + biases
            output = np.array( list( map(self.activation_function, sigma)))
            self.pre_squashing.append(sigma)
            self.outputs.append(output)
            layer_input = output

        output = self.weights[-1] @ layer_input + self.biases[-1]
        return output


    def test_feed(self, inputs):
        layer_input = inputs.reshape(len(inputs), 1)

        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            sigma = weights @ layer_input + biases
            output = np.array( list( map(self.activation_function, sigma)))
            layer_input = output

        output = self.weights[-1] @ layer_input + self.biases[-1]
        return output


    def backpropagate(self, input, output, reference):
        # calculating node errors
        self.layer_errors.append(np.matrix(reference - output))
        for weights in self.weights[:0:-1]:
            self.layer_errors.append(weights.T * self.layer_errors[-1])
        self.layer_errors.reverse()

        # loop for every layer but the last one
        for sigma, activation, errors, i in zip(self.pre_squashing, self.outputs, self.layer_errors, range(len(self.weights)-1)):
            # getting layer outputs' derivatives
            if self.activation_function == MultilayerPerceptron.__sigmoid:
                derivative_values = activation * (1 - activation)
            elif self.activation_function == MultilayerPerceptron.__tanh:
                derivative_values = 1 - activation**2
            else:
                derivative_values = self.activation_function_derivative(sigma)

            if not i:
                layer_input = input.reshape(len(input),1)
            else:
                layer_input = self.outputs[i-1]

            # calculating weights and biases changes
            self.weight_changes[i] += np.multiply(errors, derivative_values) * layer_input.T * self.learning_factor
            self.bias_changes[i] += errors * self.learning_factor

        # for the last layer it differs
        self.weight_changes[len(self.weights)-1] += self.layer_errors[-1] @ self.outputs[-1].T * self.learning_factor
        self.bias_changes[len(self.weights)-1] += self.layer_errors[-1] * self.learning_factor


    def train(self, epochs, take):
        error_check_cycle = 10
        self.weight_changes = [np.zeros_like(weights) for weights in self.weights]
        self.bias_changes = [np.zeros_like(biases) for biases in self.biases]
        record_counter = 0

        for epoch in range(epochs):
            if (epoch+1) % error_check_cycle == 0 and epoch < epochs-1:
                self.test_error()

            np.random.shuffle(self.data)
            for record in enumerate(self.data):
                self.pre_squashing = []
                self.layer_errors = []
                self.outputs = []

                record_counter += 1
                measurement = np.array([ record[1][0], record[1][1] ])
                reference = np.array([ record[1][2], record[1][3] ])
                output = self.feed_forward(measurement)
                self.backpropagate(measurement, output, reference.reshape(2,1))

                if record_counter % 64 == 0 or record_counter == len(self.data):
                    for i in range(len(self.weights)):
                        self.weights[i] += self.weight_changes[i] / record_counter
                        self.biases[i] += self.bias_changes[i] / record_counter
                        self.weight_changes = [wc * 0 for wc in self.weight_changes]
                        self.bias_changes = [bc * 0 for bc in self.bias_changes]
        save_numpy_file(self.weights, self.biases, self.filename+'take-{}.npy'.format(take))


    def test_error(self):
        errors = []
        for record in self.test_data:
            measurement = np.array([ record[0], record[1] ])
            reference = np.array([ record[2], record[3] ])
            output = self.test_feed(measurement)
            error = output_error(output.reshape(1, len(output)), reference)
            errors.append(error)
        print(np.mean(np.array(errors), axis=0))


    def test_model(self, take=0):
        self.test_outputs = []
        errors = []

        for record in self.test_data:
            measurement = np.array([ record[0], record[1] ])
            reference = np.array([ record[2], record[3] ])
            output = self.test_feed(measurement)
            error = output_error(output, reference)
            errors.append(error)
            self.test_outputs.append(measurement.tolist() + reference.tolist() + output.tolist() + error.tolist())

        mean_errors = np.mean(np.array(errors), axis=0)
        labels = ['measurement x', 'measurement y', 'reference x', 'reference y', 'output x', 'output y', 'error x', 'error y']
        output_frame = pd.DataFrame(self.test_outputs, columns=labels)
        output_frame.to_csv(self.filename+'take-{}_MSE={:.3f}.csv'.format(take, np.mean(np.square(errors))))
        print('Final network errors:', mean_errors)


    def get_data(self, path):
        df = load_data(path)
        self.data = pd.DataFrame.to_numpy(df) / 1000


# debug
if __name__ == '__main__':
    # path_template = 'data/pozyxAPI_only_localization_measurement{}.xlsx'
    # path_template = path_template.format(1)

    tpath = '/home/pawel/Pulpit/SISE/Localization_NN/data/pozyxAPI_only_localization_measurement1.xlsx'

    # mpath = '/home/pawel/Pulpit/SISE/Localization_NN/training77.npy'
    # model = load_numpy_file(mpath)
    # mlp = MultilayerPerceptron(0, 0, weights=model[0], biases=model[1])
    # mlp.test_model()
    # exit()

    mlp = MultilayerPerceptron(2, 2, 6, 5)
    mlp.get_data(tpath)

    s_time = current_time()
    for t in range(1):
        i_time = current_time()
        mlp.train(1000, t)
        mlp.test_model(t)
        print('Model obtained in {} ms'.format(current_time() - i_time))
    print("Finished in {} ms".format(current_time() - s_time))
