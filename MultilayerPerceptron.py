import pandas as pd

from math import e
from NeuronLayer import NeuronLayer
from copy import copy


def __load_data(path):
    return pd.read_excel(path, sheet_name='measurement', usecols=[2,4,5,6,7])


class MultilayerPerceptron:
    __sigmoid = lambda x: 1 / (1 + e**(-x))
    __tanh = lambda x: 2 / (1 + e**(-2*x)) -1

    def __init__(self, inputs_number, *args):
        self.layers = list()
        self.activation_function = MultilayerPerceptron.__sigmoid
        previous_layer_neurons = 0

        for arg in args:
            self.layers.append(
                NeuronLayer(previous_layer_neurons, arg, self.activation_function)
                if previous_layer_neurons
                else NeuronLayer(inputs_number, arg, self.activation_function))
            previous_layer_neurons = arg


    def feed_forward(self, inputs):
        layer_input = copy(inputs)
        print(layer_input)
        for layer in self.layers:
            layer_input = layer.feed(layer_input)
        return layer_input


    def backpropagate(self, correct_values, produced_values):
        # //TODO: implement backpropagation
        pass


# debug
if __name__ == '__main__':
    df = __load_data('/home/pawel/Pulpit/SISE/Localization_NN/data/pozyxAPI_only_localization_measurement1.xlsx')
    print(df)

    mlp = MultilayerPerceptron(3, 3, 6, 5)

    a = [1,2,3]
    print(mlp.feed_forward(a))
