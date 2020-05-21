# from library.Neuron import Neuron
import numpy as np

from random import random, seed as random_seed


class OutputLayer:

    def __init__(self, neuron_inputs, neurons_number, weights=None):
        random_seed(1)

        if weights:
            self.weights = weights
        else:
            self.weights = np.array([1 - 2*random() for x in \
                      range(neuron_inputs * neurons_number)]) \
                      .reshape(neurons_number, neuron_inputs)


    def feed(self, inputs):
        return self.weights @ inputs


    def adjust_weights(self, deltas):
        # TODO: implement backpropagation
        pass


# debug
if __name__ == '__main__':
    n = NeuronLayer(5,2)
    print(n.weights)
    l = [2,5,3,6,4]
    print(n.feed(l))
