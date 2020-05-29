import numpy as np
import pandas as pd

from time import time

current_time = lambda: round(time() * 1000)
output_error = lambda output, reference: reference - output
MSE = lambda o, r: np.mean( np.square(r - o) )

def load_data(path):
    return pd.read_excel(path, sheet_name='measurement', usecols=[4,5,6,7])


def save_numpy_file(weights, biases, filename):
    with open(filename+'.npy', 'wb') as file:
        results = np.array([weights, biases])
        np.save(file, results)


def load_numpy_file(path):
    with open(path, 'rb') as file:
        return np.load(file, allow_pickle=True)


def get_network_euclidean_error(output, reference):
    """
    Returns euclidean value of the error -> sqrt( (x - x.)^2 + (y - y.)^2 )
    """
    if len(output) != len(reference):
        raise Error('Network output should be the same length as reference')
    # TODO: implement or remove


def get_network_cost(output, reference):
    """
    Returns squared error values
    """
    if len(output) != len(reference):
        raise Error('Network output should be the same length as reference')

    cost = 0
    for i in range(len(output)):
        cost += (reference[i] - output[i])**2
    return cost

def benchmark(repeats, function, input):
    stime = current_time()
    for i in range(10000):
        pass
    print(current_time()-stime, 'ms')
