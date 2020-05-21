import pandas as pd

from math import e
from copy import copy
from NeuronLayer import NeuronLayer
from OutputLayer import OutputLayer


def load_data(path):
    return pd.read_excel(path, sheet_name='measurement', usecols=[2,4,5,6,7])


class MultilayerPerceptron:
    __sigmoid = lambda x: 1 / (1 + e**(-x))
    __tanh = lambda x: 2 / (1 + e**(-2*x)) -1

    def __init__(self, inputs_number, outputs_number, *args):
        self.layers = list()
        self.activation_function = MultilayerPerceptron.__sigmoid
        previous_layer_neurons = 0

        for arg in args:
            self.layers.append(
                NeuronLayer(previous_layer_neurons, arg, self.activation_function)
                if previous_layer_neurons
                else NeuronLayer(inputs_number, arg, self.activation_function))
            previous_layer_neurons = arg
        self.layers.append(OutputLayer(previous_layer_neurons, outputs_number))


    def feed_forward(self, inputs):
        layer_input = copy(inputs)
        print(layer_input)
        for layer in self.layers:
            layer_input = layer.feed(layer_input)
        return layer_input


    def backpropagate(self, correct_values, produced_values):
        # //TODO: implement backpropagation
        pass

    def train(self):
        for instance in self.measurements:
            feed_forward(instance)


    def get_data(self, path):
        df = load_data(path)
        self.measurements = zip(df['measurement x'] / 100, df['measurement y'] / 1000)
        self.references =  zip(df['reference x'] / 1000, df['reference y'] / 1000)


# debug
if __name__ == '__main__':
    # path_template = 'data/pozyxAPI_only_localization_measurement{}.xlsx'
    # path_template = path_template.format(1)

    tpath = '/home/pawel/Pulpit/SISE/Localization_NN/data/pozyxAPI_only_localization_measurement1.xlsx'

    mlp = MultilayerPerceptron(2, 2, 6, 5)
    # mlp.get_data(tpath)

    df = load_data(tpath)
    a = (df['measurement x'][0], df['measurement y'][0])
    print(mlp.feed_forward([2.071,4.075]))
