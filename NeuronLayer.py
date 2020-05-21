# from library.Neuron import Neuron
import numpy as np

from random import random, seed as random_seed

from math import e  # delete this line


class NeuronLayer:

    def __init__(self, neuron_inputs, neurons_number, activation_function, weights=None):
        random_seed(1)

        if weights:
            self.weights = weights
        else:
            self.weights = np.array([1 - 2*random() for x in \
                      range(neuron_inputs * neurons_number)]) \
                      .reshape(neurons_number, neuron_inputs)
        self.activation_function = activation_function


    def feed(self, inputs):
        return list(map(self.activation_function, self.weights @ inputs))


    def adjust_weights(self, deltas):
        # TODO: implement backpropagation
        pass


# debug
if __name__ == '__main__':
    n = NeuronLayer(5,7,lambda x: 1 / (1 + e**(-x)))
    print(n.weights)
    l = [2,5,3,6,4]
    print(n.feed(l))
