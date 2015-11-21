# The simplest RNN possible, using tensorflow 
import numpy as np
import tensorflow as tf

rng = np.random.RandomState(42)

dim_input = 11
dim_hidden = 128

# embed each input as a vector
W_in = tf.Variable(rng.uniform(-.1, .1,
	(dim_input, dim_hidden)).astype('float32'))

# recurrent matrix
W_rec = tf.Variable(rng.uniform(-.1, .1,
	(dim_input, dim_hidden)).astype('float32'))


# recurrence step
def recurrence(inp, state):
    
