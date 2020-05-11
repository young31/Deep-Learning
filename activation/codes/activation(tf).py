import tensorflow as tf
import numpy as np

def swish(x):
    return x*tf.sigmoid(x)


def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))


def mish(x):
    return x*tf.tanh(tf.softplus(x))