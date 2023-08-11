"""
S. Pavlioglou.

Reproducing the results of the original ipynb
"""

import numpy as np
from rnn_utils import *
from public_tests import *

"""
INPUT, x:

as input we are going to have a tensor of dimension 3.
the shape of the tensor is (n_x, m, T_x) where:
    - n_x is the size of the vocabulary 
    - m is the size of the batch that we are feeding the
        solver with at any given moment in time t
    - T_x is the size of the time dimension for this 
        particular sequence


HIDDEN STATE, a: 

this is the information that is passing form the one
time step to the next. 
    - for single training example length=n_a
    - we include the mini-batch size => (n_a, m)
    - then we can think of all the activations of the network
        as a single tensor combined. in that case the dimension
        would be (n_a, m, T_x)


PREDICTION, y^ (y_pred):

dimension: (n_y, m, T_y) where: 
    - n_y number of words in the output vocab
    - m number of examples in the mini batch
    - T_y number of sequence steps in the pred space


"""

def rnn_cell_forward(xt, a_prev, parameters):
    """
    Function that simulates the computations that are 
    taking place inside the recurring RNN cell. 

    Arguments:
    [x_t] :: the input of time instance t (n_x, m)
    [a_prev] :: the activation of the previous cell (n_a, m)
    [parameters] :: dictionary of the weights that are to be tuned for multiplying
            - with the input: Wax (n_a, n_x)
            - with the activation: Waa (n_a, n_a)
            - for the output: Wya (n_y, n_a)
            - ba, activation bias (n_a, 1)
            - by, prediction bias (n_y, 1)

    Returns:
    [a_next] :: activation towards the next cell (n_a, m)
    [yt_pred] :: prediction at timestep t (n_y, m)
    [cache] :: tuple of values needed for the backward pass,
                contains (a_next, a_prev, xt, parameters, )

                
    """
    # extracting the parameters
    wax = parameters['Wax']
    waa = parameters['Waa']
    wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    # using numpy for the matrix multiplications
    a_next = np.tanh(np.dot(waa, a_prev) + np.dot(wax, xt) + ba)
    yt_pred = softmax(np.dot(wya, a_prev) + by)
    cache = (a_next, yt_pred, xt, parameters)
    
    return a_next, yt_pred, cache



def rnn_forward(x, a0, parameters):
    """
    Function that uses the rnn_cell_forward to actually 
    simulate a forward pass. 

    Arguments: 
    [x] :: the total input data for all time steps combined (n_x, m, T_x)
    [a0] :: initial hidden state (n_a, m)
    [parametrs] :: same as above

    Returns: 
    [a] :: total hidden state matrix (n_a, m, T_x)
    [y_pred] :: total predictions matrix (n_y, m, T_y)
    [caches] :: values needed for the backward pass
    """

    # Initialize the cashes list
    caches = []

    # Retrieve the dimensions of the various vars
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape
    T_y = T_x # in this case

    # Initialize a and y_pred with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_y))

    # loop over all time steps
    for t in range(T_x):
        # update the next hidden state, compute the prediction, get the cache
        pass
        # save the value of the new next hidden state in a
        pass
        # save the value of the prediction in y
        pass
        # append cache toe cashes

    caches = (cache, x)
    return a, y_pred, caches