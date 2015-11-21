import theano
import theano.tensor as T
import numpy as np

from sklearn.utils import check_random_state

floatX = theano.config['floatX']


class RNN(object):
    """Simple simple RNN"""

    def __init__(self, n_input, n_hidden,
                 n_representation=None
                 random_state=42):
        self.n_input = n_input
        self.n_representation = n_representation
        self.n_hidden = n_hidden
        self.random_state = random_state

    def initialize_params(self):
        """Set all shared variables necessary. Initialize them.

        Note: This could be made more flexible and accept already
        created variables from somewhere else. But for now it creates them."""

        self.rng_ = check_random_state(self.random_state)
        self.Wrec_ = theano.shared(
            (.1 * self.rng_.rand(self.n_hidden,
                                 self.n_hidden)).astype(floatX),
            name='Wrec')
        if n_representation is not None:
            self.Wrep_ = theano.shared(
                (.1 * self.rng_.rand(self.n_input,
                                     self.n_representation)).astype(floatX),
                name='Wrep')
        self.Win_ = theano.shared(
            (.1 * self.rng_.rand(self.n_representation,
                                 self.n_hidden)).astype(floatX),
            name='Win')

        self.brec_ = theano.shared(
            np.zeros(self.n_hidden, dtype=floatX),
            name='brec')
        self.bin_ = theano.shared(
            np.zeros(self.n_hidden, dtype=floatX),
            name='bin')

        self.params_ = [self.Wrep_, self.Win_, self.Wrec_,
                        self.brep_, self.bin_, self.brec_]

    def build_expression(self, input_expression=None):

        if input_expression is None:
            
